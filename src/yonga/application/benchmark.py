"""Benchmark service: repeatable, self-describing measurements of one artifact.

A benchmark number is worthless without the context that produced it, so every
result written here records what ``SPEC.md`` section 6.5 lists: the hardware and
package versions, the artifact id and the sha256 of its manifest, the device and
its runtime properties, the generation settings and their fingerprint, the exact
sha256 of every prompt, the compile/load time, and per run the TTFT, token counts,
throughput, TPOT, wall time, peak process RSS and any error.

Three rules shape the implementation:

* **Warmups never enter the aggregates.** The first generation pays for lazy
  plugin initialisation and a cold cache; including it would flatter the numbers.
  Warmup runs are recorded (they are evidence) and dropped by
  :func:`~yonga.domain.benchmarks.aggregate_runs`.
* **The prompt text is the fixture.** ``src/yonga/data/prompts.json`` is data, not
  documentation: its sha256 goes into every :class:`RunMetrics`, so editing a
  prompt makes old results visibly incomparable instead of silently wrong.
* **Comparability is checked, not assumed.** Two results produced with different
  generation settings, prompt sets, devices or OpenVINO versions are not
  comparable. :meth:`BenchmarkService.compare` refuses to present them as such and
  returns the reasons rather than hiding them.

A run that raises or overruns its budget becomes an error entry. It never
propagates out and takes the other runs' measurements with it.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yonga.adapters.runtime_genai import ChatHistory, ChatRuntime
from yonga.application.jobs import JobContext
from yonga.application.probe import ArtifactStoreLike, LocalArtifactStore, resolve_artifact
from yonga.application.progress import ProgressKind
from yonga.config.paths import Paths
from yonga.config.settings import Settings
from yonga.domain.artifacts import ArtifactManifest, manifest_path
from yonga.domain.benchmarks import (
    AggregateMetrics,
    BenchmarkResult,
    GenerationSettings,
    PromptFixture,
    RunMetrics,
    aggregate_runs,
)
from yonga.domain.diagnostics import (
    CompileError,
    ConfigurationError,
    Diagnostic,
    FailureCategory,
    Severity,
    ValidationError,
    YongaError,
    diagnostic,
)
from yonga.domain.hardware import HardwareInventory, normalise_device
from yonga.domain.models import PipelineKind
from yonga.util.hashing import sha256_file, short_hash
from yonga.util.redaction import redact_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yonga.adapters import HardwareProbe

__all__ = [
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_SUITE",
    "PROMPTS_RESOURCE",
    "BenchmarkComparison",
    "BenchmarkService",
    "ComparisonRow",
    "PromptLibrary",
    "RuntimeFactory",
    "benchmark_result_path",
    "load_prompt_library",
    "load_result",
    "peak_rss_bytes",
]

#: Suite used when the caller does not name one.
DEFAULT_SUITE = "standard"

#: Benchmark generation length. Deliberately not ``Settings.max_new_tokens``: chat
#: wants a generous ceiling, a benchmark wants a short, fixed, comparable budget.
DEFAULT_MAX_NEW_TOKENS = 256

#: Package-data location of the prompt fixtures.
PROMPTS_RESOURCE = "data/prompts.json"

#: Packages recorded with every result; the first three decide comparability.
_RECORDED_PACKAGES: tuple[str, ...] = (
    "openvino",
    "openvino_genai",
    "openvino_tokenizers",
    "optimum",
    "optimum_intel",
    "nncf",
    "transformers",
    "torch",
    "huggingface_hub",
    "python",
)

_KIB = 1024
_UNSAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")
_DEFAULT_DEVICE = "CPU"


def _now() -> datetime:
    """Current UTC timestamp."""
    return datetime.now(UTC)


# --- ports ----------------------------------------------------------------------------


class RuntimeFactory(Protocol):
    """Opens a loaded model on a device.

    :func:`yonga.adapters.runtime_genai.open_runtime` satisfies this; a test
    double returning scripted metrics satisfies it just as well, which is what
    keeps the service testable with no OpenVINO installed.
    """

    def __call__(
        self,
        artifact_dir: Path,
        *,
        device: str = _DEFAULT_DEVICE,
        pipeline: PipelineKind = PipelineKind.LLM,
        properties: Mapping[str, str] | None = None,
        settings: Settings | None = None,
    ) -> ChatRuntime:
        """Load ``artifact_dir`` on ``device`` and return a runtime."""
        ...


#: Reads the process' peak resident set size in bytes, or ``None`` when unknown.
PeakMemoryReader = Callable[[], int | None]


def peak_rss_bytes() -> int | None:
    """Best-effort peak resident set size of this process, in bytes.

    ``resource.getrusage`` is preferred because it reports the *peak* (Linux
    returns ``ru_maxrss`` in KiB), which is what ``SPEC.md`` section 6.5 asks for;
    ``psutil`` only exposes the current RSS and is used as the fallback. Both are
    optional: a missing reading is recorded as ``None`` and breaks nothing.
    """
    with suppress(ImportError, OSError, ValueError, AttributeError):
        import resource

        maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if maxrss > 0:
            return maxrss * _KIB
    with suppress(Exception):
        import psutil

        return int(psutil.Process().memory_info().rss)
    return None


# --- prompt fixtures ------------------------------------------------------------------


class PromptLibrary(BaseModel):
    """The benchmark prompt fixtures and the suites built from them."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = 1
    prompts: list[PromptFixture] = Field(default_factory=list)
    suites: dict[str, list[str]] = Field(default_factory=dict)
    default_suite: str = DEFAULT_SUITE

    @model_validator(mode="after")
    def _check_references(self) -> PromptLibrary:
        """Reject a library whose suites name prompts that do not exist."""
        known = {prompt.id for prompt in self.prompts}
        if len(known) != len(self.prompts):
            raise ValueError("prompt ids must be unique")
        for suite, ids in self.suites.items():
            missing = [name for name in ids if name not in known]
            if missing:
                raise ValueError(f"suite {suite!r} names unknown prompts: {', '.join(missing)}")
            if not ids:
                raise ValueError(f"suite {suite!r} is empty")
        return self

    @property
    def suite_names(self) -> list[str]:
        """Available suite names, sorted."""
        return sorted(self.suites)

    def prompt(self, prompt_id: str) -> PromptFixture:
        """Return one prompt fixture by id."""
        for prompt in self.prompts:
            if prompt.id == prompt_id:
                return prompt
        raise ConfigurationError(
            diagnostic(
                "bench.prompt.unknown",
                FailureCategory.CONFIGURATION_ERROR,
                "Unknown benchmark prompt",
                f"No benchmark prompt is called {prompt_id!r}.",
                remediation=[f"Known prompts: {', '.join(p.id for p in self.prompts)}"],
            )
        )

    def suite(self, name: str) -> list[PromptFixture]:
        """Return the prompt fixtures of one suite, in their declared order."""
        key = (name or self.default_suite).strip()
        ids = self.suites.get(key)
        if ids is None:
            raise ConfigurationError(
                diagnostic(
                    "bench.suite.unknown",
                    FailureCategory.CONFIGURATION_ERROR,
                    "Unknown benchmark suite",
                    f"There is no benchmark suite called {key!r}.",
                    remediation=[f"Available suites: {', '.join(self.suite_names)}"],
                    context={"suite": key},
                )
            )
        return [self.prompt(prompt_id) for prompt_id in ids]


_CACHED_LIBRARY: PromptLibrary | None = None


def load_prompt_library(path: Path | None = None) -> PromptLibrary:
    """Load the prompt fixtures, from ``path`` or from the shipped package data.

    The shipped library is cached because it is immutable data; an explicit
    ``path`` is always re-read so tests can vary it.
    """
    # The shipped fixtures are immutable package data, so caching them is safe.
    global _CACHED_LIBRARY
    if path is None and _CACHED_LIBRARY is not None:
        return _CACHED_LIBRARY
    raw = _read_prompt_text(path)
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise ConfigurationError(
            diagnostic(
                "bench.prompts.malformed",
                FailureCategory.CONFIGURATION_ERROR,
                "Benchmark prompt fixtures are not valid JSON",
                f"Failed to parse the prompt fixtures: {exc}",
                remediation=["Reinstall yonga; the prompt fixtures ship with the package."],
            ),
            cause=exc,
        ) from exc
    try:
        library = PromptLibrary.model_validate(payload)
    except ValueError as exc:
        raise ConfigurationError(
            diagnostic(
                "bench.prompts.invalid",
                FailureCategory.CONFIGURATION_ERROR,
                "Benchmark prompt fixtures are invalid",
                str(exc),
                remediation=["Reinstall yonga; the prompt fixtures ship with the package."],
            ),
            cause=exc,
        ) from exc
    if path is None:
        _CACHED_LIBRARY = library
    return library


def _read_prompt_text(path: Path | None) -> str:
    """Read the prompt fixture document from disk or from the installed package."""
    if path is not None:
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigurationError(
                diagnostic(
                    "bench.prompts.unreadable",
                    FailureCategory.CONFIGURATION_ERROR,
                    "Benchmark prompt fixtures could not be read",
                    f"Failed to read {path}: {exc}",
                    context={"path": str(path)},
                ),
                cause=exc,
            ) from exc
    with suppress(OSError, ModuleNotFoundError, FileNotFoundError):
        return resources.files("yonga").joinpath(PROMPTS_RESOURCE).read_text(encoding="utf-8")
    fallback = Path(__file__).resolve().parent.parent / PROMPTS_RESOURCE
    try:
        return fallback.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - only when the install is broken
        raise ConfigurationError(
            diagnostic(
                "bench.prompts.missing",
                FailureCategory.CONFIGURATION_ERROR,
                "Benchmark prompt fixtures are missing",
                f"{PROMPTS_RESOURCE} was not found in the installed package.",
                remediation=["Reinstall yonga; the prompt fixtures ship with the package."],
            ),
            cause=exc,
        ) from exc


# --- comparison -----------------------------------------------------------------------


class ComparisonRow(BaseModel):
    """One prompt compared between two benchmark results."""

    model_config = ConfigDict(extra="ignore")

    prompt_id: str
    baseline: AggregateMetrics | None = None
    candidate: AggregateMetrics | None = None
    throughput_ratio: float | None = None
    ttft_ratio: float | None = None


class BenchmarkComparison(BaseModel):
    """The verdict on whether two benchmark results may be compared at all."""

    model_config = ConfigDict(extra="ignore")

    comparable: bool = False
    reasons: list[str] = Field(default_factory=list)
    baseline_id: str
    candidate_id: str
    rows: list[ComparisonRow] = Field(default_factory=list)
    diagnostic: Diagnostic | None = None


# --- persistence ----------------------------------------------------------------------


def _safe_id(value: str) -> str:
    """Reduce a string to characters that are safe in a file name."""
    cleaned = _UNSAFE_ID.sub("-", value.strip()).strip("-.")
    return cleaned or "benchmark"


def benchmark_result_path(paths: Paths, result: BenchmarkResult) -> Path:
    """Return the file a benchmark result is saved to."""
    return paths.benchmarks_dir / f"{_safe_id(result.benchmark_id)}.json"


def load_result(path: Path) -> BenchmarkResult:
    """Read a saved benchmark document back into a :class:`BenchmarkResult`."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValidationError(
            diagnostic(
                "bench.result.unreadable",
                FailureCategory.CONFIGURATION_ERROR,
                "Benchmark result could not be read",
                f"Failed to read {path}: {exc}",
                context={"path": str(path)},
            ),
            cause=exc,
        ) from exc
    try:
        return BenchmarkResult.model_validate(payload)
    except ValueError as exc:
        raise ValidationError(
            diagnostic(
                "bench.result.schema_mismatch",
                FailureCategory.CONFIGURATION_ERROR,
                "Benchmark result does not match the expected schema",
                f"{path} could not be validated: {exc}",
                context={"path": str(path)},
            ),
            cause=exc,
        ) from exc


# --- the service ----------------------------------------------------------------------


class BenchmarkService:
    """Runs a prompt suite against one artifact on one device and records everything."""

    def __init__(
        self,
        runtime_factory: RuntimeFactory,
        artifact_store: ArtifactStoreLike | None = None,
        hardware_probe: HardwareProbe | None = None,
        settings: Settings | None = None,
        *,
        prompts: PromptLibrary | None = None,
        peak_memory: PeakMemoryReader | None = None,
        classifier: Callable[[str, str], Diagnostic | None] | None = None,
    ) -> None:
        """Wire the service to its adapters.

        ``classifier`` turns a load failure's raw text into a registry diagnosis,
        exactly as ``probe`` and ``chat`` do; without one the failure is reported
        as the runtime described it.

        ``artifact_store`` defaults to a
        :class:`~yonga.application.probe.LocalArtifactStore`, ``prompts`` to the
        shipped fixtures, and ``peak_memory`` to :func:`peak_rss_bytes`. The
        hardware probe is optional: without it the result simply records no
        hardware identity rather than failing.
        """
        self._factory = runtime_factory
        self._settings = settings
        self._hardware = hardware_probe
        self._classifier = classifier
        paths = None if settings is None else settings.paths
        self._store: ArtifactStoreLike = artifact_store or LocalArtifactStore(paths)
        self._prompts = prompts or load_prompt_library()
        self._peak_memory: PeakMemoryReader = peak_memory or peak_rss_bytes

    @property
    def prompts(self) -> PromptLibrary:
        """The prompt fixtures this service benchmarks with."""
        return self._prompts

    # --- running ----------------------------------------------------------

    def _classified(
        self,
        error: YongaError,
        device: str,
        manifest: ArtifactManifest | None = None,
    ) -> YongaError:
        """The registry's diagnosis of a load failure, when a rule recognises it.

        The diagnosis names the model the artifact was prepared from, as
        ``probe`` and ``chat`` do, so its ``yonga prepare MODEL`` steps print
        the real repository instead of the placeholder.
        """
        if self._classifier is None:
            return error
        raw = error.diagnostic.raw_excerpt or error.diagnostic.message
        found = self._classifier(raw, device) if raw.strip() else None
        if found is None or found.matched_rule_id is None:
            return error
        repo_id = None if manifest is None else manifest.resolved_source.repo_id
        if repo_id and not found.context.get("repo_id"):
            found = found.model_copy(update={"context": {**found.context, "repo_id": repo_id}})
        return YongaError(found)

    def run(
        self,
        artifact: str | Path,
        *,
        device: str | None = None,
        suite: str = DEFAULT_SUITE,
        runs: int | None = None,
        warmups: int | None = None,
        ctx: JobContext,
        generation: GenerationSettings | None = None,
        properties: Mapping[str, str] | None = None,
        pipeline: PipelineKind | None = None,
        timeout_seconds: float | None = None,
        save: bool = True,
    ) -> BenchmarkResult:
        """Benchmark ``artifact`` and return the full result.

        ``runs`` and ``warmups`` default to ``Settings.bench_runs`` (3) and
        ``Settings.bench_warmups`` (1). Warmups are measured and recorded but are
        excluded from the aggregates. A run that raises is recorded as an error
        entry and the remaining runs still complete. When ``save`` is true the
        document is written under ``Paths.benchmarks_dir``; the file it lands in
        is :func:`benchmark_result_path`.
        """
        artifact_dir = resolve_artifact(self._store, artifact)
        manifest = self._store.manifest_for(artifact_dir)
        fixtures = self._prompts.suite(suite)
        suite_name = (suite or self._prompts.default_suite).strip()

        measured_runs = self._positive(runs, self._setting_int("bench_runs", 3), "runs")
        warmup_runs = self._non_negative(warmups, self._setting_int("bench_warmups", 1), "warmups")
        target = self._resolve_device(device, manifest)
        settings = generation or GenerationSettings(
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            do_sample=False,
        )
        runtime_properties = self._runtime_properties(manifest, target, properties)
        kind = pipeline or (manifest.pipeline if manifest is not None else PipelineKind.LLM)
        if kind is PipelineKind.UNKNOWN:
            kind = PipelineKind.LLM

        errors: list[str] = []
        inventory = self._inventory(errors)
        created_at = _now()

        ctx.emit(
            ProgressKind.STAGE,
            f"Loading {artifact_dir.name} on {target}",
            device=target,
        )
        compile_seconds: float | None = None
        runtime: ChatRuntime | None = None
        started = time.perf_counter()
        try:
            runtime = self._factory(
                artifact_dir,
                device=target,
                pipeline=kind,
                properties=dict(runtime_properties) or None,
                settings=self._settings,
            )
        except YongaError as exc:
            # A model that cannot load has no benchmark. Writing an empty result
            # and exiting 2 hid the reason; ``probe`` and ``chat`` classify the
            # same failure and exit 51, and so does this now.
            raise self._classified(exc, target, manifest) from exc
        except Exception as exc:
            raw = redact_text(_error_text(exc))
            raise self._classified(
                CompileError(
                    diagnostic(
                        "bench.load_failed",
                        FailureCategory.UNKNOWN_RUNTIME_ERROR,
                        f"Could not load the model on {target}",
                        raw,
                        raw_excerpt=raw,
                    )
                ),
                target,
                manifest,
            ) from exc
        else:
            compile_seconds = time.perf_counter() - started
            reported = getattr(runtime, "load_seconds", None)
            if isinstance(reported, int | float) and float(reported) > 0.0:
                compile_seconds = float(reported)

        run_metrics: list[RunMetrics] = []
        if runtime is not None:
            try:
                run_metrics = self._run_suite(
                    runtime,
                    fixtures,
                    settings=settings,
                    runs=measured_runs,
                    warmups=warmup_runs,
                    ctx=ctx,
                    errors=errors,
                    timeout_seconds=timeout_seconds,
                )
            finally:
                with suppress(Exception):
                    runtime.close()

        aggregates = [
            aggregate_runs(fixture.id, [run for run in run_metrics if run.ok])
            for fixture in fixtures
        ]

        result = BenchmarkResult(
            benchmark_id=_benchmark_id(created_at, manifest, target, suite_name),
            created_at=created_at,
            suite=suite_name,
            artifact_id=_artifact_id(manifest, artifact_dir),
            artifact_path=artifact_dir,
            artifact_manifest_sha256=_manifest_sha256(artifact_dir),
            repo_id=_repo_id(manifest, artifact_dir),
            profile=_profile(manifest),
            device=target,
            hardware=_hardware_identity(inventory),
            package_versions=_package_versions(inventory),
            runtime_properties=dict(runtime_properties),
            settings=settings,
            settings_fingerprint=settings.fingerprint(),
            warmups=warmup_runs,
            runs=measured_runs,
            compile_seconds=compile_seconds,
            run_metrics=run_metrics,
            aggregates=aggregates,
            errors=errors,
        )
        if save:
            path = self.save(result)
            ctx.emit(
                ProgressKind.STAGE,
                f"Benchmark saved to {path}",
                status="saved",
                path=str(path),
            )
        return result

    def _run_suite(
        self,
        runtime: ChatRuntime,
        fixtures: Sequence[PromptFixture],
        *,
        settings: GenerationSettings,
        runs: int,
        warmups: int,
        ctx: JobContext,
        errors: list[str],
        timeout_seconds: float | None,
    ) -> list[RunMetrics]:
        """Execute every prompt's warmups and measured runs, in order."""
        collected: list[RunMetrics] = []
        total = len(fixtures) * (runs + warmups)
        done = 0
        for fixture in fixtures:
            for index in range(warmups + runs):
                if ctx.check_cancelled():
                    errors.append("benchmark cancelled before completing every run")
                    return collected
                warmup = index < warmups
                done += 1
                label = "warmup" if warmup else "run"
                ctx.emit(
                    ProgressKind.STAGE,
                    f"{fixture.id}: {label} {index + 1}",
                    current=done,
                    total=total,
                )
                metrics = self._measure(
                    runtime,
                    fixture,
                    run_index=index,
                    warmup=warmup,
                    settings=settings,
                    timeout_seconds=timeout_seconds,
                )
                collected.append(metrics)
                if metrics.error is not None:
                    errors.append(f"{fixture.id} {label} {index + 1}: {metrics.error}")
                    ctx.emit(
                        ProgressKind.LOG,
                        f"{fixture.id} {label} {index + 1} failed: {metrics.error}",
                        severity=Severity.ERROR,
                    )
                    if not warmup:
                        # Keep going: the point of several runs is that one bad
                        # run must not cost the measurements of the others.
                        continue
        return collected

    def _measure(
        self,
        runtime: ChatRuntime,
        fixture: PromptFixture,
        *,
        run_index: int,
        warmup: bool,
        settings: GenerationSettings,
        timeout_seconds: float | None,
    ) -> RunMetrics:
        """Run one generation and turn it into a :class:`RunMetrics` record.

        ``run_index`` counts within the prompt's own sequence, warmups first.
        Nothing raises out of here: a failed generation becomes an ``error`` field.
        """
        history = ChatHistory()
        history.add_user(fixture.text)
        started = time.perf_counter()
        try:
            # Every run — warmups included — is its own conversation. A stateful
            # runtime that carried the previous run's turns into this one would
            # be measuring a longer context than the fixture describes, so its
            # TTFT and token counts would not be comparable with the other runs
            # (SPEC.md section 6.5). Independence is declared here rather than
            # inferred from the history, which repeats the same prompt verbatim.
            runtime.begin_conversation()
            generated = runtime.generate(history, settings)
        except Exception as exc:
            return RunMetrics(
                prompt_id=fixture.id,
                prompt_sha256=fixture.sha256,
                run_index=run_index,
                warmup=warmup,
                wall_seconds=time.perf_counter() - started,
                peak_rss_bytes=self._read_peak_memory(),
                error=_error_text(exc),
            )
        wall = time.perf_counter() - started
        error: str | None = None
        if timeout_seconds is not None and wall > timeout_seconds:
            error = (
                f"the run took {wall:.1f} s, over the {timeout_seconds:.1f} s budget; "
                "generation cannot be interrupted mid-call, so it was allowed to finish "
                "and is excluded from the aggregates"
            )
        return RunMetrics(
            prompt_id=fixture.id,
            prompt_sha256=fixture.sha256,
            run_index=run_index,
            warmup=warmup,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
            ttft_ms=generated.ttft_ms,
            throughput_tps=generated.throughput_tps,
            tpot_ms=generated.tpot_ms,
            wall_seconds=generated.wall_seconds or wall,
            load_seconds=generated.load_seconds,
            peak_rss_bytes=self._read_peak_memory(),
            error=error,
        )

    def _read_peak_memory(self) -> int | None:
        """Read peak RSS without ever letting the reader fail a run."""
        try:
            return self._peak_memory()
        except Exception:
            return None

    # --- persistence ------------------------------------------------------

    def save(self, result: BenchmarkResult, *, paths: Paths | None = None) -> Path:
        """Write the result as JSON under ``Paths.benchmarks_dir`` and return the path.

        The document is exactly ``BenchmarkResult.to_json()``, so it round-trips
        through :meth:`BenchmarkResult.model_validate` without loss.
        """
        target = paths or self._paths()
        directory = target.benchmarks_dir
        directory.mkdir(parents=True, exist_ok=True)
        path = benchmark_result_path(target, result)
        document = json.dumps(result.to_json(), indent=2, sort_keys=True, ensure_ascii=False)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(document + "\n", encoding="utf-8")
        tmp.replace(path)
        return path

    # --- comparison -------------------------------------------------------

    def compare(
        self,
        baseline: BenchmarkResult,
        candidate: BenchmarkResult,
    ) -> BenchmarkComparison:
        """Compare two results, refusing when they are not comparable.

        Delegates the verdict to
        :meth:`~yonga.domain.benchmarks.BenchmarkResult.comparable_to`. When it
        says no, the comparison carries every reason and **no** rows: a refusal
        that hides why it refused is worse than no answer at all.
        """
        comparable, reasons = baseline.comparable_to(candidate)
        if not comparable:
            return BenchmarkComparison(
                comparable=False,
                reasons=reasons,
                baseline_id=baseline.benchmark_id,
                candidate_id=candidate.benchmark_id,
                rows=[],
                diagnostic=diagnostic(
                    "bench.not_comparable",
                    FailureCategory.CONFIGURATION_ERROR,
                    "These benchmark results are not comparable",
                    "; ".join(reasons),
                    severity=Severity.WARNING,
                    remediation=[
                        "Re-run both benchmarks with the same suite, device and "
                        "generation settings.",
                        "Benchmark results are only comparable within one OpenVINO release family.",
                    ],
                    context={
                        "baseline": baseline.benchmark_id,
                        "candidate": candidate.benchmark_id,
                    },
                ),
            )
        rows = [
            _comparison_row(prompt_id, baseline, candidate)
            for prompt_id in sorted({aggregate.prompt_id for aggregate in baseline.aggregates})
        ]
        return BenchmarkComparison(
            comparable=True,
            reasons=[],
            baseline_id=baseline.benchmark_id,
            candidate_id=candidate.benchmark_id,
            rows=rows,
        )

    # --- helpers ----------------------------------------------------------

    def _paths(self) -> Paths:
        """Return the managed paths, raising when the service has no settings."""
        if self._settings is None:
            raise ConfigurationError(
                diagnostic(
                    "bench.paths.unknown",
                    FailureCategory.CONFIGURATION_ERROR,
                    "Benchmark results have nowhere to go",
                    "This BenchmarkService was built without settings, so the managed "
                    "benchmarks directory is unknown.",
                    remediation=["Pass settings=..., or call save(result, paths=...)."],
                )
            )
        return self._settings.paths

    def _setting_int(self, name: str, default: int) -> int:
        """Read an integer setting, falling back to ``default``."""
        if self._settings is None:
            return default
        value = getattr(self._settings, name, default)
        return int(value) if isinstance(value, int) else default

    @staticmethod
    def _positive(value: int | None, default: int, label: str) -> int:
        """Validate a strictly positive count."""
        resolved = default if value is None else int(value)
        if resolved < 1:
            raise ConfigurationError(
                diagnostic(
                    "bench.count.invalid",
                    FailureCategory.CONFIGURATION_ERROR,
                    f"Invalid number of {label}",
                    f"{label} must be at least 1, got {resolved}.",
                )
            )
        return resolved

    @staticmethod
    def _non_negative(value: int | None, default: int, label: str) -> int:
        """Validate a count that may be zero."""
        resolved = default if value is None else int(value)
        if resolved < 0:
            raise ConfigurationError(
                diagnostic(
                    "bench.count.invalid",
                    FailureCategory.CONFIGURATION_ERROR,
                    f"Invalid number of {label}",
                    f"{label} must not be negative, got {resolved}.",
                )
            )
        return resolved

    def _resolve_device(self, device: str | None, manifest: ArtifactManifest | None) -> str:
        """Pick the device to benchmark on.

        An explicit device wins. Otherwise the artifact's recommended runtime
        device is used, then the first device whose probe succeeded, then the
        configured default policy, and finally CPU — which is always present.
        """
        requested = (device or "").strip()
        if requested and requested.casefold() != "auto":
            return normalise_device(requested)
        if manifest is not None:
            if manifest.recommended_runtime is not None:
                return normalise_device(manifest.recommended_runtime.device)
            supported = manifest.supported_devices
            if supported:
                return supported[0]
        if self._settings is not None:
            policy = self._settings.default_device.value
            if policy and policy.casefold() != "auto":
                return normalise_device(policy)
        return _DEFAULT_DEVICE

    @staticmethod
    def _runtime_properties(
        manifest: ArtifactManifest | None,
        device: str,
        override: Mapping[str, str] | None,
    ) -> dict[str, str]:
        """Runtime properties for the pipeline, recorded with the result.

        The manifest's recommended properties are only used when they were
        recorded for this device: a GPU's ``ATTENTION_BACKEND`` says nothing about
        the NPU, and recording it there would make the result a lie.
        """
        if override is not None:
            return {str(key): str(value) for key, value in override.items()}
        if manifest is None or manifest.recommended_runtime is None:
            return {}
        recommended = manifest.recommended_runtime
        if normalise_device(recommended.device) != normalise_device(device):
            return {}
        return {str(key): str(value) for key, value in recommended.properties.items()}

    def _inventory(self, errors: list[str]) -> HardwareInventory | None:
        """Collect the hardware inventory, never failing the benchmark over it."""
        if self._hardware is None:
            return None
        try:
            return self._hardware.inventory()
        except Exception as exc:
            errors.append(f"hardware inventory unavailable: {_error_text(exc)}")
            return None


# --- record building ------------------------------------------------------------------


def _error_text(exc: BaseException) -> str:
    """Render an exception for the record, redacted and bounded."""
    return redact_text(f"{type(exc).__name__}: {exc}")[:2000]


def _benchmark_id(
    created_at: datetime,
    manifest: ArtifactManifest | None,
    device: str,
    suite: str,
) -> str:
    """Build a readable, file-safe, collision-resistant benchmark id."""
    stamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    artifact = manifest.artifact_id if manifest is not None else "artifact"
    token = short_hash(f"{artifact}|{device}|{suite}|{created_at.isoformat()}", 8)
    return _safe_id(f"{stamp}-{device.lower()}-{suite}-{token}")


def _artifact_id(manifest: ArtifactManifest | None, artifact_dir: Path) -> str:
    """The artifact's recorded id, or its directory name when it has no manifest."""
    if manifest is not None and manifest.artifact_id:
        return manifest.artifact_id
    return artifact_dir.name


def _repo_id(manifest: ArtifactManifest | None, artifact_dir: Path) -> str:
    """The source repository the artifact was built from."""
    if manifest is not None and manifest.resolved_source.repo_id:
        return manifest.resolved_source.repo_id
    return artifact_dir.name


def _profile(manifest: ArtifactManifest | None) -> str:
    """The conversion profile the artifact was built with."""
    if manifest is None:
        return ""
    return manifest.export.profile


def _manifest_sha256(artifact_dir: Path) -> str | None:
    """Digest of the artifact manifest file, or ``None`` when there is none."""
    path = manifest_path(artifact_dir)
    if not path.is_file():
        return None
    with suppress(OSError):
        return sha256_file(path)
    return None


def _hardware_identity(inventory: HardwareInventory | None) -> dict[str, str]:
    """Flatten the hardware facts worth recording with a benchmark."""
    if inventory is None:
        return {}
    host = inventory.host
    data: dict[str, str] = {
        "platform": host.platform,
        "machine": host.machine,
        "kernel": host.kernel,
        "python_version": host.python_version,
        "memory_total_bytes": str(inventory.memory.total_bytes),
    }
    if host.cpu_model:
        data["cpu_model"] = host.cpu_model
    if host.cpu_count is not None:
        data["cpu_count"] = str(host.cpu_count)
    if host.distro_pretty:
        data["distro"] = host.distro_pretty
    devices = inventory.available_device_names
    if devices:
        data["devices"] = ", ".join(devices)
    for device in inventory.devices:
        name = device.normalised_name or device.name
        data[f"device.{name}"] = device.full_name
        if device.architecture:
            data[f"device.{name}.architecture"] = device.architecture
    npu = inventory.npu
    if npu is not None and npu.driver_version:
        data["npu_driver_version"] = npu.driver_version
    return {key: value for key, value in data.items() if value}


def _package_versions(inventory: HardwareInventory | None) -> dict[str, str]:
    """Record the package versions a benchmark's comparability depends on."""
    from yonga.registry.resolver import installed_registry_versions

    versions: dict[str, str] = {}
    for name, value in installed_registry_versions().items():
        if name in _RECORDED_PACKAGES and value:
            versions[name] = value
    if inventory is not None:
        for package in inventory.packages:
            if package.version:
                versions.setdefault(package.name.replace("-", "_"), package.version)
    return versions


def _comparison_row(
    prompt_id: str,
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
) -> ComparisonRow:
    """Build one prompt's comparison row, tolerating missing aggregates."""
    left = baseline.aggregate_for(prompt_id)
    right = candidate.aggregate_for(prompt_id)
    return ComparisonRow(
        prompt_id=prompt_id,
        baseline=left,
        candidate=right,
        throughput_ratio=_ratio(
            None if left is None else left.throughput_tps_mean,
            None if right is None else right.throughput_tps_mean,
        ),
        ttft_ratio=_ratio(
            None if left is None else left.ttft_ms_mean,
            None if right is None else right.ttft_ms_mean,
        ),
    )


def _ratio(baseline: float | None, candidate: float | None) -> float | None:
    """Return ``candidate / baseline``, or ``None`` when either side is missing."""
    if baseline is None or candidate is None or baseline == 0.0:
        return None
    return candidate / baseline
