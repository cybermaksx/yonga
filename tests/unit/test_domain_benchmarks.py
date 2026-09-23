"""Unit tests for yonga.domain.benchmarks."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from yonga.domain.benchmarks import (
    BENCH_SCHEMA_VERSION,
    BenchmarkResult,
    GenerationSettings,
    PromptFixture,
    RunMetrics,
    aggregate_runs,
)

PROMPT = PromptFixture(id="short", text="Explain what an NPU is in one sentence.")


def run(
    index: int,
    *,
    warmup: bool = False,
    ttft: float | None = None,
    throughput: float | None = None,
    tpot: float | None = None,
    output_tokens: int | None = None,
    wall: float = 1.0,
    prompt_id: str = PROMPT.id,
) -> RunMetrics:
    return RunMetrics(
        prompt_id=prompt_id,
        prompt_sha256=PROMPT.sha256,
        run_index=index,
        warmup=warmup,
        ttft_ms=ttft,
        throughput_tps=throughput,
        tpot_ms=tpot,
        output_tokens=output_tokens,
        wall_seconds=wall,
    )


# --- prompts and settings ---------------------------------------------------------------


def test_prompt_sha256_matches_hashlib() -> None:
    expected = hashlib.sha256(PROMPT.text.encode("utf-8")).hexdigest()
    assert PROMPT.sha256 == expected
    assert len(PROMPT.sha256) == 64


def test_fingerprint_is_stable_and_sampling_sensitive() -> None:
    base = GenerationSettings(max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True)
    same = GenerationSettings(max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True)
    assert base.fingerprint() == same.fingerprint()
    assert len(base.fingerprint()) == 64

    for changed in (
        base.model_copy(update={"temperature": 0.8}),
        base.model_copy(update={"top_p": 0.95}),
        base.model_copy(update={"top_k": 40}),
        base.model_copy(update={"max_new_tokens": 512}),
        base.model_copy(update={"do_sample": False}),
        base.model_copy(update={"repetition_penalty": 1.1}),
        base.model_copy(update={"seed": 1234}),
    ):
        assert changed.fingerprint() != base.fingerprint()


def test_fingerprint_canonical_payload_is_json_serialisable() -> None:
    settings = GenerationSettings()
    assert json.dumps(settings.canonical_payload())


# --- aggregation ------------------------------------------------------------------------


def test_aggregate_runs_ignores_warmups() -> None:
    runs = [
        run(0, warmup=True, ttft=999.0, throughput=1.0, tpot=999.0, output_tokens=1, wall=9.0),
        run(1, ttft=100.0, throughput=10.0, tpot=50.0, output_tokens=64, wall=2.0),
        run(2, ttft=200.0, throughput=20.0, tpot=70.0, output_tokens=96, wall=4.0),
    ]
    aggregate = aggregate_runs(PROMPT.id, runs)
    assert aggregate.prompt_id == PROMPT.id
    assert aggregate.runs == 2
    assert aggregate.ttft_ms_mean == 150.0
    assert aggregate.ttft_ms_median == 150.0
    assert aggregate.throughput_tps_mean == 15.0
    assert aggregate.tpot_ms_mean == 60.0
    assert aggregate.output_tokens_mean == 80.0
    assert aggregate.wall_seconds_mean == 3.0
    assert aggregate.ttft_ms_stdev is not None


def test_aggregate_runs_tolerates_missing_metrics() -> None:
    runs = [
        run(0, warmup=True, ttft=1.0),
        run(1, ttft=100.0, throughput=None, tpot=None, output_tokens=None, wall=2.0),
        run(2, ttft=None, throughput=20.0, tpot=None, output_tokens=10, wall=4.0),
        run(3, ttft=None, throughput=None, tpot=None, output_tokens=None, wall=6.0),
    ]
    aggregate = aggregate_runs(PROMPT.id, runs)
    assert aggregate.runs == 3
    assert aggregate.ttft_ms_mean == 100.0
    assert aggregate.ttft_ms_stdev is None  # only one sample
    assert aggregate.throughput_tps_mean == 20.0
    assert aggregate.tpot_ms_mean is None
    assert aggregate.tpot_ms_median is None
    assert aggregate.output_tokens_mean == 10.0
    assert aggregate.wall_seconds_mean == 4.0


def test_aggregate_runs_stdev_needs_two_samples() -> None:
    one = aggregate_runs(PROMPT.id, [run(1, ttft=100.0)])
    assert one.ttft_ms_stdev is None
    two = aggregate_runs(PROMPT.id, [run(1, ttft=100.0), run(2, ttft=120.0)])
    assert two.ttft_ms_stdev is not None
    assert round(two.ttft_ms_stdev, 6) == round(14.142135623730951, 6)


def test_aggregate_runs_with_only_warmups_or_no_runs() -> None:
    empty = aggregate_runs(PROMPT.id, [])
    assert empty.runs == 0
    assert empty.ttft_ms_mean is None
    assert empty.wall_seconds_mean is None

    warmups_only = aggregate_runs(PROMPT.id, [run(0, warmup=True, ttft=5.0)])
    assert warmups_only.runs == 0
    assert warmups_only.ttft_ms_mean is None


def test_aggregate_runs_ignores_other_prompts() -> None:
    runs = [run(1, ttft=100.0), run(2, ttft=300.0, prompt_id="medium")]
    assert aggregate_runs(PROMPT.id, runs).runs == 1
    assert aggregate_runs(PROMPT.id, runs).ttft_ms_mean == 100.0


# --- comparability ----------------------------------------------------------------------


def make_result(**overrides: object) -> BenchmarkResult:
    payload: dict[str, object] = {
        "benchmark_id": "bench-1",
        "created_at": datetime(2026, 9, 22, 11, 0, tzinfo=UTC),
        "suite": "default",
        "artifact_id": "qwen35-abc123-gpu",
        "artifact_path": Path("/artifacts/qwen35/abc123--gpu"),
        "repo_id": "nDimensional/Qwen3.5-9B-Uncensored-Safetensors",
        "profile": "gpu-int4-asym-g128",
        "device": "GPU",
        "hardware": {"cpu": "Intel Core Ultra 7 268V"},
        "package_versions": {
            "openvino": "2026.4.0",
            "openvino_genai": "2026.4.0.0",
            "openvino_tokenizers": "2026.4.0.0",
        },
        "runtime_properties": {"ATTENTION_BACKEND": "SDPA"},
        "settings": GenerationSettings(max_new_tokens=256),
        "warmups": 1,
        "runs": 2,
        "run_metrics": [run(0, warmup=True, ttft=5.0), run(1, ttft=100.0), run(2, ttft=120.0)],
    }
    payload.update(overrides)
    return BenchmarkResult.model_validate(payload)


def test_result_defaults_and_fingerprint_filling() -> None:
    result = make_result()
    assert result.schema_version == BENCH_SCHEMA_VERSION
    assert result.settings_fingerprint == result.settings.fingerprint()
    assert result.device == "GPU"
    assert json.dumps(result.to_json())


def test_identical_results_are_comparable() -> None:
    ok, reasons = make_result().comparable_to(make_result())
    assert ok
    assert reasons == []


def test_results_differ_by_device() -> None:
    ok, reasons = make_result().comparable_to(make_result(device="npu.0"))
    assert not ok
    assert any("devices differ" in reason for reason in reasons)


def test_results_differ_by_generation_settings() -> None:
    other = make_result(settings=GenerationSettings(max_new_tokens=512))
    ok, reasons = make_result().comparable_to(other)
    assert not ok
    assert "generation settings differ" in reasons


def test_results_differ_by_sampling_even_with_same_length() -> None:
    other = make_result(settings=GenerationSettings(max_new_tokens=256, temperature=0.9))
    ok, reasons = make_result().comparable_to(other)
    assert not ok
    assert "generation settings differ" in reasons


def test_results_differ_by_prompt_set() -> None:
    other = make_result(run_metrics=[run(1, ttft=100.0, prompt_id="medium")])
    ok, reasons = make_result().comparable_to(other)
    assert not ok
    assert "prompt sets differ" in reasons


def test_results_differ_by_prompt_text_for_same_id() -> None:
    mutated = run(1, ttft=100.0)
    mutated.prompt_sha256 = "0" * 64
    other = make_result(run_metrics=[mutated])
    ok, reasons = make_result().comparable_to(other)
    assert not ok
    assert "prompt texts differ for the same prompt ids" in reasons


def test_results_differ_by_runtime_properties() -> None:
    """A different pipeline configuration is not a comparable benchmark.

    Reproduced against two real GPU results on this host: the artifact built
    with ``gpu-int4-asym-g128`` records ``ATTENTION_BACKEND=SDPA`` because its
    profile declares it, the one built with ``npu-int4-sym-cw`` records nothing.
    Everything else matched, so the pair was reported comparable with no reasons
    at all, against ``SPEC.md`` section 6.5.
    """
    ok, reasons = make_result().comparable_to(make_result(runtime_properties={}))
    assert not ok
    assert any("ATTENTION_BACKEND" in reason for reason in reasons)
    assert any("unset" in reason for reason in reasons)


def test_results_differ_by_a_runtime_property_value() -> None:
    """Two values for the same property are as incomparable as one missing value."""
    other = make_result(runtime_properties={"ATTENTION_BACKEND": "PA"})
    ok, reasons = make_result().comparable_to(other)
    assert not ok
    assert any("SDPA" in reason and "PA" in reason for reason in reasons)


def test_runtime_property_names_are_compared_case_insensitively() -> None:
    """OpenVINO property names are upper case; a lower-case record is the same key."""
    other = make_result(runtime_properties={"attention_backend": "SDPA"})
    ok, reasons = make_result().comparable_to(other)
    assert ok
    assert reasons == []


def test_results_differ_by_package_versions() -> None:
    other = make_result(
        package_versions={
            "openvino": "2026.2.0",
            "openvino-genai": "2026.4.0.0",
            "openvino_tokenizers": "2026.4.0.0",
        }
    )
    ok, reasons = make_result().comparable_to(other)
    assert not ok
    assert any(reason.startswith("openvino versions differ") for reason in reasons)
    # openvino-genai and openvino_genai are the same distribution.
    assert not any("genai" in reason for reason in reasons)


def test_missing_package_version_breaks_comparability() -> None:
    ok, reasons = make_result().comparable_to(make_result(package_versions={}))
    assert not ok
    assert any("unknown" in reason for reason in reasons)


def test_schema_version_mismatch_breaks_comparability() -> None:
    ok, reasons = make_result().comparable_to(make_result(schema_version=2))
    assert not ok
    assert any("schema versions differ" in reason for reason in reasons)


def test_aggregate_lookup_on_result() -> None:
    result = make_result(aggregates=[aggregate_runs(PROMPT.id, make_result().run_metrics)])
    assert result.aggregate_for(PROMPT.id) is not None
    assert result.aggregate_for("missing") is None


def test_has_measurements_is_false_when_every_run_failed() -> None:
    """One aggregate row is written per prompt even when nothing succeeded.

    The row exists so the summary table can show the prompt, but it carries
    ``runs=0``. Exit-code policy has to look at ``runs``, not at the presence of
    the row, or a benchmark in which every generation raised reports success
    with warnings instead of a generation failure.
    """
    empty = aggregate_runs("short", [])
    assert empty.runs == 0

    result = make_result(
        run_metrics=[],
        aggregates=[empty],
        errors=["short run 1: GenerationError: synthetic generation failure"],
    )

    assert result.aggregates  # the row is present ...
    assert result.has_measurements is False  # ... but nothing was measured


def test_has_measurements_is_true_when_at_least_one_run_succeeded() -> None:
    result = make_result(aggregates=[aggregate_runs("short", [run(1, ttft=100.0)])])

    assert result.has_measurements is True
