"""Benchmark domain types: prompts, generation settings, per-run and aggregate metrics.

Benchmark results are persisted as JSON and compared across devices and profiles, so
comparability is part of the domain: two results may only be compared when the prompts,
the sampling settings, the device, the pipeline's runtime properties and the relevant
package versions match.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from yonga.domain.hardware import normalise_device

#: Current benchmark record schema version.
BENCH_SCHEMA_VERSION = 1

#: Packages whose version must match before two benchmark results may be compared.
COMPARABILITY_PACKAGES: tuple[str, ...] = (
    "openvino",
    "openvino_genai",
    "openvino_tokenizers",
)

_LENIENT = ConfigDict(extra="ignore")


def _sha256_text(text: str) -> str:
    """Return the hex SHA-256 digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalise_package_name(name: str) -> str:
    """Normalise a distribution name so ``openvino-genai`` and ``openvino_genai`` match."""
    return name.strip().lower().replace("-", "_")


def _normalise_property_name(name: str) -> str:
    """Normalise a runtime property name; OpenVINO GenAI spells them upper case."""
    return name.strip().upper()


class PromptFixture(BaseModel):
    """A fixed benchmark prompt, identified by id and content hash."""

    model_config = _LENIENT

    id: str
    text: str
    category: str = "general"

    @property
    def sha256(self) -> str:
        """Hex SHA-256 digest of the exact prompt text."""
        return _sha256_text(self.text)


class GenerationSettings(BaseModel):
    """Sampling and length settings used for generation and benchmarking."""

    model_config = _LENIENT

    max_new_tokens: int = 256
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    do_sample: bool = False
    repetition_penalty: float | None = None
    seed: int | None = None
    stop_strings: list[str] = Field(default_factory=list)
    """Model-level end markers the runtime stops at natively, never shown.

    They come from the registry chat profile, not from a client: a client's
    stop sequences are matched on the visible answer by the chat service.
    """

    def canonical_payload(self) -> dict[str, object]:
        """Return the sampling-relevant fields in a canonical, hashable form.

        ``stop_strings`` joins the payload only when there are some, so every
        fingerprint recorded before the field existed still matches.
        """
        payload: dict[str, object] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
        }
        if self.stop_strings:
            payload["stop_strings"] = sorted(set(self.stop_strings))
        return payload

    def fingerprint(self) -> str:
        """Stable SHA-256 fingerprint of the sampling-relevant settings."""
        canonical = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return _sha256_text(canonical)


class RunMetrics(BaseModel):
    """Measurements captured for a single generation run."""

    model_config = _LENIENT

    prompt_id: str
    prompt_sha256: str
    run_index: int
    warmup: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    ttft_ms: float | None = None
    throughput_tps: float | None = None
    tpot_ms: float | None = None
    wall_seconds: float = 0.0
    load_seconds: float | None = None
    peak_rss_bytes: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the run completed without a recorded error."""
        return self.error is None


class AggregateMetrics(BaseModel):
    """Summary statistics over the measured (non-warmup) runs of one prompt."""

    model_config = _LENIENT

    prompt_id: str
    runs: int = 0
    ttft_ms_mean: float | None = None
    ttft_ms_median: float | None = None
    ttft_ms_stdev: float | None = None
    throughput_tps_mean: float | None = None
    throughput_tps_median: float | None = None
    tpot_ms_mean: float | None = None
    tpot_ms_median: float | None = None
    output_tokens_mean: float | None = None
    wall_seconds_mean: float | None = None


def _mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or None when there is nothing to average."""
    if not values:
        return None
    return float(statistics.fmean(values))


def _median(values: Sequence[float]) -> float | None:
    """Median, or None when there is nothing to summarise."""
    if not values:
        return None
    return float(statistics.median(values))


def _stdev(values: Sequence[float]) -> float | None:
    """Sample standard deviation, defined only for two or more samples."""
    if len(values) < 2:
        return None
    return float(statistics.stdev(values))


def aggregate_runs(prompt_id: str, runs: Iterable[RunMetrics]) -> AggregateMetrics:
    """Aggregate the measured runs of one prompt, ignoring warmups and missing metrics.

    Runs belonging to another prompt and runs flagged as warmups are dropped. Every
    metric is summarised over the runs that actually reported it, so a partially
    instrumented backend still produces useful aggregates. The standard deviation is
    reported only when at least two samples are available.
    """
    measured = [run for run in runs if run.prompt_id == prompt_id and not run.warmup]
    ttft = [run.ttft_ms for run in measured if run.ttft_ms is not None]
    throughput = [run.throughput_tps for run in measured if run.throughput_tps is not None]
    tpot = [run.tpot_ms for run in measured if run.tpot_ms is not None]
    output_tokens = [float(run.output_tokens) for run in measured if run.output_tokens is not None]
    wall = [run.wall_seconds for run in measured]
    return AggregateMetrics(
        prompt_id=prompt_id,
        runs=len(measured),
        ttft_ms_mean=_mean(ttft),
        ttft_ms_median=_median(ttft),
        ttft_ms_stdev=_stdev(ttft),
        throughput_tps_mean=_mean(throughput),
        throughput_tps_median=_median(throughput),
        tpot_ms_mean=_mean(tpot),
        tpot_ms_median=_median(tpot),
        output_tokens_mean=_mean(output_tokens),
        wall_seconds_mean=_mean(wall),
    )


class BenchmarkResult(BaseModel):
    """A complete, reproducible benchmark record for one artifact on one device."""

    model_config = _LENIENT

    schema_version: int = BENCH_SCHEMA_VERSION
    benchmark_id: str
    created_at: datetime
    suite: str = "default"
    artifact_id: str
    artifact_path: Path
    artifact_manifest_sha256: str | None = None
    repo_id: str
    profile: str = ""
    device: str
    hardware: dict[str, str] = Field(default_factory=dict)
    package_versions: dict[str, str] = Field(default_factory=dict)
    runtime_properties: dict[str, str] = Field(default_factory=dict)
    settings: GenerationSettings = Field(default_factory=GenerationSettings)
    settings_fingerprint: str = ""
    warmups: int = 0
    runs: int = 0
    compile_seconds: float | None = None
    run_metrics: list[RunMetrics] = Field(default_factory=list)
    aggregates: list[AggregateMetrics] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @field_validator("device")
    @classmethod
    def _normalise_device_name(cls, value: str) -> str:
        return normalise_device(value)

    @model_validator(mode="after")
    def _fill_fingerprint(self) -> BenchmarkResult:
        """Derive the settings fingerprint when the caller did not supply one."""
        if not self.settings_fingerprint:
            self.settings_fingerprint = self.settings.fingerprint()
        return self

    @property
    def has_measurements(self) -> bool:
        """True when at least one prompt produced a successful measured run.

        One :class:`AggregateMetrics` row is written per prompt in the suite
        whether or not anything succeeded, so the *presence* of aggregates says
        nothing about whether the benchmark measured anything. ``runs`` counts
        the successful measured runs behind a row, so a benchmark in which every
        run raised has aggregates but no measurements — and must be reported as a
        generation failure rather than as success with warnings.
        """
        return any(aggregate.runs for aggregate in self.aggregates)

    @property
    def prompt_signature(self) -> tuple[tuple[str, str], ...]:
        """Sorted ``(prompt_id, prompt_sha256)`` pairs of the measured runs."""
        return tuple(
            sorted(
                {(run.prompt_id, run.prompt_sha256) for run in self.run_metrics if not run.warmup}
            )
        )

    def aggregate_for(self, prompt_id: str) -> AggregateMetrics | None:
        """Return the stored aggregate for ``prompt_id``, if present."""
        for aggregate in self.aggregates:
            if aggregate.prompt_id == prompt_id:
                return aggregate
        return None

    def comparable_to(self, other: BenchmarkResult) -> tuple[bool, list[str]]:
        """Return whether two results may be compared, plus the reasons they may not.

        ``runtime_properties`` counts. ``SPEC.md`` section 6.5 forbids comparing
        runs made "using different prompt/generation settings", and the
        properties handed to the GenAI pipeline — ``ATTENTION_BACKEND``,
        ``MAX_PROMPT_LEN``, ``MIN_RESPONSE_LEN`` — are exactly that: they decide
        how the model is executed, not merely where. Two artifacts whose
        profiles declare different pipeline properties produce different numbers
        on the same device with the same prompts, so they are reported as
        incomparable with the differing keys named.
        """
        reasons: list[str] = []
        if self.schema_version != other.schema_version:
            reasons.append(
                f"benchmark schema versions differ ({self.schema_version} vs "
                f"{other.schema_version})"
            )
        if self.settings_fingerprint != other.settings_fingerprint:
            reasons.append("generation settings differ")
        if self.device != other.device:
            reasons.append(f"devices differ ({self.device} vs {other.device})")
        mine = self.prompt_signature
        theirs = other.prompt_signature
        if mine != theirs:
            if {pid for pid, _ in mine} != {pid for pid, _ in theirs}:
                reasons.append("prompt sets differ")
            else:
                reasons.append("prompt texts differ for the same prompt ids")
        reasons.extend(_runtime_property_reasons(self.runtime_properties, other.runtime_properties))
        reasons.extend(_package_version_reasons(self.package_versions, other.package_versions))
        return (not reasons, reasons)

    def to_json(self) -> dict[str, object]:
        """Return a fully JSON-serialisable representation of the benchmark result."""
        return dict(self.model_dump(mode="json"))


def _package_version_reasons(mine: dict[str, str], theirs: dict[str, str]) -> list[str]:
    """Report comparability-critical package versions that differ between two runs."""
    left = {_normalise_package_name(name): value for name, value in mine.items()}
    right = {_normalise_package_name(name): value for name, value in theirs.items()}
    reasons: list[str] = []
    for package in COMPARABILITY_PACKAGES:
        first = left.get(package)
        second = right.get(package)
        if first == second:
            continue
        reasons.append(f"{package} versions differ ({first or 'unknown'} vs {second or 'unknown'})")
    return reasons


def _runtime_property_reasons(mine: dict[str, str], theirs: dict[str, str]) -> list[str]:
    """Report the pipeline runtime properties that differ between two runs.

    Every key either side recorded is compared, because a property set on one
    side and absent on the other is still a difference: the side that did not
    record it ran on the plugin's default, which is a different configuration
    rather than an unknown one. Names are folded to upper case first, so a
    hand-written record spelling ``attention_backend`` matches the registry's
    ``ATTENTION_BACKEND`` instead of being reported as two missing properties.
    """
    left = {_normalise_property_name(name): value for name, value in mine.items()}
    right = {_normalise_property_name(name): value for name, value in theirs.items()}
    reasons: list[str] = []
    for name in sorted(set(left) | set(right)):
        first = left.get(name)
        second = right.get(name)
        if first == second:
            continue
        reasons.append(
            f"runtime property {name} differs "
            f"({'unset' if first is None else first} vs {'unset' if second is None else second})"
        )
    return reasons
