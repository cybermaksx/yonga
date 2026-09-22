"""Benchmark service tests: warmups, prompt hashes, schema, errors and comparability.

The runtime is scripted so the arithmetic is checkable by hand: a benchmark whose
mean is wrong is worse than no benchmark, so the maths is asserted against values
computed in the test rather than against the service's own output.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pytest

from yonga.adapters.runtime_genai import ChatHistory, ChatRuntime, GenerationResult
from yonga.application.benchmark import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_SUITE,
    BenchmarkService,
    PromptLibrary,
    benchmark_result_path,
    load_prompt_library,
    load_result,
    peak_rss_bytes,
)
from yonga.application.jobs import JobContext
from yonga.application.probe import LocalArtifactStore
from yonga.application.progress import CollectingSink
from yonga.config.settings import Settings
from yonga.domain.benchmarks import BenchmarkResult, GenerationSettings
from yonga.domain.diagnostics import ConfigurationError
from yonga.domain.hardware import (
    DeviceKind,
    HardwareInventory,
    HostInfo,
    MemoryInfo,
    OpenVINODevice,
    PackageInfo,
)
from yonga.domain.models import PipelineKind
from yonga.util.jsonlog import null_logger

# --- the prompt fixtures are a contract -----------------------------------------------

#: sha256 of the exact prompt text shipped in ``src/yonga/data/prompts.json``.
#: These are deliberately hard-coded: if a prompt is edited, every benchmark
#: recorded with the old text becomes incomparable, and this test is the alarm.
EXPECTED_PROMPT_SHA256: dict[str, str] = {
    "short-instruction": "ceee5e267606f227c88d5e9751b8d50371ce549b53d79eb518758e061ce5c722",
    "medium-explanation": "30cffdd34c254e040099c05c90a233ae768b1bec2cfdbfeadf7bd1e402b41bc7",
    "long-context": "acb9d81eac90a8260d56d4c0ea68a45a0d4ba4442ae8239ef5d18dd57a747dea",
    "long-context-analysis": "078529c6b1c0b04276a0db4b9add15aa6e514546a62acd1396998ba884fe9da1",
}


# --- doubles --------------------------------------------------------------------------


class ScriptedRuntime:
    """A ChatRuntime returning canned metrics, or raising a canned exception."""

    def __init__(
        self,
        script: Mapping[str, Sequence[GenerationResult | Exception]] | None = None,
        *,
        default: GenerationResult | None = None,
        load_seconds: float = 0.75,
    ) -> None:
        self.script = {key: list(value) for key, value in (script or {}).items()}
        self.default = default or make_generation(ttft_ms=100.0, throughput_tps=20.0)
        self.load_seconds = load_seconds
        self.calls: list[str] = []
        self.closed = False

    def generate(self, history: ChatHistory, config: GenerationSettings) -> GenerationResult:
        prompt = history.messages[-1].content
        key = _prompt_key(prompt)
        self.calls.append(key)
        queue = self.script.get(key)
        if not queue:
            return self.default
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


class RecordingFactory:
    """A RuntimeFactory that records how it was called."""

    def __init__(self, runtime: ChatRuntime | None = None, error: Exception | None = None) -> None:
        self.runtime = runtime
        self.error = error
        self.calls: list[tuple[Path, str, PipelineKind, dict[str, str]]] = []

    def __call__(
        self,
        artifact_dir: Path,
        *,
        device: str = "CPU",
        pipeline: PipelineKind = PipelineKind.LLM,
        properties: Mapping[str, str] | None = None,
        settings: Settings | None = None,
    ) -> ChatRuntime:
        self.calls.append((artifact_dir, device, pipeline, dict(properties or {})))
        if self.error is not None:
            raise self.error
        assert self.runtime is not None
        return self.runtime


class FakeHardwareProbe:
    """A HardwareProbe returning the reference host's identity."""

    def __init__(self, inventory: HardwareInventory | None = None) -> None:
        self.inventory_value = inventory or make_inventory()

    def inventory(self, *, full: bool = False) -> HardwareInventory:
        return self.inventory_value


# --- builders -------------------------------------------------------------------------


def _prompt_key(text: str) -> str:
    """Map a prompt's text back to its fixture id."""
    for prompt in load_prompt_library().prompts:
        if prompt.text == text:
            return prompt.id
    return "unknown"


def make_generation(
    *,
    ttft_ms: float | None = 100.0,
    throughput_tps: float | None = 20.0,
    tpot_ms: float | None = 50.0,
    input_tokens: int | None = 42,
    output_tokens: int | None = 256,
    wall_seconds: float = 10.0,
) -> GenerationResult:
    """A generation result with the metrics a real perf-metrics read would carry."""
    return GenerationResult(
        text="answer",
        display_text="answer",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        ttft_ms=ttft_ms,
        throughput_tps=throughput_tps,
        tpot_ms=tpot_ms,
        wall_seconds=wall_seconds,
        load_seconds=0.75,
        stop_reason="length",
    )


def make_inventory() -> HardwareInventory:
    """The reference host, as ``doctor`` would report it."""
    return HardwareInventory(
        host=HostInfo(
            platform="Linux",
            machine="x86_64",
            kernel="6.17.0-generic",
            python_version="3.11.9",
            distro_pretty="Ubuntu 26.04 LTS",
            cpu_model="Intel(R) Core(TM) Ultra 7 268V",
            cpu_count=8,
        ),
        memory=MemoryInfo(total_bytes=33_102_655_488, available_bytes=20_000_000_000),
        devices=[
            OpenVINODevice(
                name="CPU",
                kind=DeviceKind.CPU,
                full_name="Intel(R) Core(TM) Ultra 7 268V",
            ),
            OpenVINODevice(
                name="GPU",
                kind=DeviceKind.GPU,
                full_name="Intel(R) Arc(TM) Graphics (iGPU)",
                architecture="GPU: vendor=0x8086 arch=v20.4.4",
            ),
            OpenVINODevice(name="NPU", kind=DeviceKind.NPU, full_name="Intel(R) AI Boost"),
        ],
        openvino_available=True,
        packages=[
            PackageInfo(name="openvino", version="2026.4.0", importable=True),
            PackageInfo(name="openvino-genai", version="2026.4.0.0", importable=True),
        ],
    )


def make_settings(tmp_path: Path) -> Settings:
    """Settings whose managed directories live under ``tmp_path``."""
    return Settings(
        config_dir=tmp_path / "config",
        cache_dir=tmp_path / "cache",
        data_dir=tmp_path / "data",
    )


@pytest.fixture
def job_ctx(tmp_path: Path) -> Iterator[tuple[JobContext, CollectingSink]]:
    """A job context that logs nowhere and collects its progress events."""
    settings = make_settings(tmp_path)
    sink = CollectingSink()
    logger = null_logger("job-bench")
    try:
        yield (
            JobContext(
                job_id="job-bench",
                logger=logger,
                paths=settings.paths,
                settings=settings,
                progress=sink,
            ),
            sink,
        )
    finally:
        logger.close()


def make_artifact(tmp_path: Path) -> Path:
    """A minimal prepared-artifact directory."""
    artifact = tmp_path / "artifacts" / "acme__model__cpu"
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    return artifact


def make_service(
    tmp_path: Path,
    runtime: ChatRuntime,
    *,
    hardware: FakeHardwareProbe | None = None,
    peak_memory: object = None,
) -> tuple[BenchmarkService, RecordingFactory]:
    """Build the service under test together with its factory double."""
    factory = RecordingFactory(runtime)
    settings = make_settings(tmp_path)
    service = BenchmarkService(
        factory,
        LocalArtifactStore(settings.paths),
        hardware if hardware is not None else FakeHardwareProbe(),
        settings,
        peak_memory=peak_memory,  # type: ignore[arg-type]
    )
    return service, factory


# --- the prompt library ---------------------------------------------------------------


def test_the_shipped_prompts_have_exact_stable_hashes() -> None:
    """The prompt text is the fixture; its digest is part of the contract."""
    library = load_prompt_library()
    digests = {prompt.id: prompt.sha256 for prompt in library.prompts}
    assert digests == EXPECTED_PROMPT_SHA256


def test_the_library_has_a_short_a_medium_and_a_long_prompt() -> None:
    """AGENT_PROMPT 23 requires all three shapes, each with a stable id and category."""
    library = load_prompt_library()
    by_id = {prompt.id: prompt for prompt in library.prompts}

    assert by_id["short-instruction"].category == "instruction"
    assert by_id["medium-explanation"].category == "explanation"
    assert by_id["long-context"].category == "long_context"
    assert all(prompt.id and prompt.category for prompt in library.prompts)
    lengths = [len(by_id[name].text) for name in ("short-instruction", "medium-explanation")]
    assert lengths[0] < lengths[1] < len(by_id["long-context"].text)


def test_the_three_suites_exist_and_grow() -> None:
    """quick <= standard <= long, and every suite names real prompts."""
    library = load_prompt_library()
    assert library.suite_names == ["long", "quick", "standard"]
    quick = [prompt.id for prompt in library.suite("quick")]
    standard = [prompt.id for prompt in library.suite("standard")]
    long_suite = [prompt.id for prompt in library.suite("long")]
    assert set(quick) < set(standard) < set(long_suite)
    assert library.default_suite == DEFAULT_SUITE


def test_an_unknown_suite_is_refused_with_the_available_ones() -> None:
    """A typo must say what the options are, not fall back to a default."""
    with pytest.raises(ConfigurationError) as excinfo:
        load_prompt_library().suite("turbo")
    assert excinfo.value.diagnostic.code == "bench.suite.unknown"
    assert "standard" in " ".join(excinfo.value.diagnostic.remediation)


def test_a_library_naming_a_missing_prompt_is_rejected() -> None:
    """A suite may not reference a prompt that does not exist."""
    with pytest.raises(ValueError, match="unknown prompts"):
        PromptLibrary.model_validate(
            {
                "prompts": [{"id": "a", "text": "hello", "category": "instruction"}],
                "suites": {"quick": ["a", "ghost"]},
            }
        )


# --- warmups --------------------------------------------------------------------------


def test_warmups_are_recorded_but_excluded_from_the_aggregates(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """The warmup is evidence, not a measurement."""
    ctx, _ = job_ctx
    runtime = ScriptedRuntime(
        {
            "short-instruction": [
                # The warmup is deliberately terrible; it must not move the mean.
                make_generation(ttft_ms=9000.0, throughput_tps=1.0, tpot_ms=900.0),
                make_generation(ttft_ms=100.0, throughput_tps=20.0, tpot_ms=50.0),
                make_generation(ttft_ms=200.0, throughput_tps=30.0, tpot_ms=40.0),
                make_generation(ttft_ms=300.0, throughput_tps=40.0, tpot_ms=30.0),
            ]
        }
    )
    service, _ = make_service(tmp_path, runtime)

    result = service.run(
        make_artifact(tmp_path), device="CPU", suite="quick", runs=3, warmups=1, ctx=ctx
    )

    assert [run.warmup for run in result.run_metrics] == [True, False, False, False]
    assert result.warmups == 1
    assert result.runs == 3
    aggregate = result.aggregate_for("short-instruction")
    assert aggregate is not None
    assert aggregate.runs == 3
    assert aggregate.ttft_ms_mean == pytest.approx((100.0 + 200.0 + 300.0) / 3)
    assert aggregate.ttft_ms_median == pytest.approx(200.0)


def test_mean_and_median_are_computed_over_the_measured_runs(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """Hand-checked arithmetic over four runs with an even-length median."""
    ctx, _ = job_ctx
    runtime = ScriptedRuntime(
        {
            "short-instruction": [
                make_generation(ttft_ms=10.0, throughput_tps=1.0, tpot_ms=4.0, wall_seconds=1.0),
                make_generation(ttft_ms=20.0, throughput_tps=2.0, tpot_ms=3.0, wall_seconds=2.0),
                make_generation(ttft_ms=30.0, throughput_tps=3.0, tpot_ms=2.0, wall_seconds=3.0),
                make_generation(ttft_ms=40.0, throughput_tps=4.0, tpot_ms=1.0, wall_seconds=4.0),
            ]
        }
    )
    service, _ = make_service(tmp_path, runtime)

    result = service.run(
        make_artifact(tmp_path), device="CPU", suite="quick", runs=4, warmups=0, ctx=ctx
    )

    aggregate = result.aggregate_for("short-instruction")
    assert aggregate is not None
    assert aggregate.runs == 4
    assert aggregate.ttft_ms_mean == pytest.approx(25.0)
    assert aggregate.ttft_ms_median == pytest.approx(25.0)
    assert aggregate.ttft_ms_stdev == pytest.approx(12.909944487358056)
    assert aggregate.throughput_tps_mean == pytest.approx(2.5)
    assert aggregate.throughput_tps_median == pytest.approx(2.5)
    assert aggregate.tpot_ms_mean == pytest.approx(2.5)
    assert aggregate.wall_seconds_mean == pytest.approx(2.5)
    assert aggregate.output_tokens_mean == pytest.approx(256.0)


def test_zero_warmups_is_allowed(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """``--warmups 0`` measures every run."""
    ctx, _ = job_ctx
    service, _ = make_service(tmp_path, ScriptedRuntime())
    result = service.run(
        make_artifact(tmp_path), device="CPU", suite="quick", runs=2, warmups=0, ctx=ctx
    )
    assert [run.warmup for run in result.run_metrics] == [False, False]


def test_run_and_warmup_counts_come_from_settings(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """Defaults are 1 warmup and 3 runs, straight from Settings."""
    ctx, _ = job_ctx
    service, _ = make_service(tmp_path, ScriptedRuntime())
    result = service.run(make_artifact(tmp_path), device="CPU", suite="quick", ctx=ctx)
    assert (result.warmups, result.runs) == (1, 3)
    assert len(result.run_metrics) == 4


# --- what gets recorded ---------------------------------------------------------------


def test_every_run_records_the_exact_prompt_hash(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """The digest recorded per run is the digest of the shipped fixture."""
    ctx, _ = job_ctx
    service, _ = make_service(tmp_path, ScriptedRuntime())

    result = service.run(
        make_artifact(tmp_path), device="CPU", suite="standard", runs=1, warmups=0, ctx=ctx
    )

    recorded = {run.prompt_id: run.prompt_sha256 for run in result.run_metrics}
    assert recorded == {
        "short-instruction": EXPECTED_PROMPT_SHA256["short-instruction"],
        "medium-explanation": EXPECTED_PROMPT_SHA256["medium-explanation"],
        "long-context": EXPECTED_PROMPT_SHA256["long-context"],
    }
    assert result.prompt_signature == tuple(sorted(recorded.items()))


def test_artifact_hardware_and_runtime_identity_are_present(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """SPEC 6.5: a result must carry the context that produced it."""
    ctx, _ = job_ctx
    artifact = make_artifact(tmp_path)
    service, factory = make_service(tmp_path, ScriptedRuntime())

    result = service.run(
        artifact,
        device="gpu",
        suite="quick",
        runs=1,
        warmups=1,
        ctx=ctx,
        properties={"ATTENTION_BACKEND": "SDPA"},
    )

    assert result.artifact_path == artifact
    assert result.artifact_id == artifact.name
    assert result.device == "GPU"
    assert result.suite == "quick"
    assert result.schema_version == 1
    assert result.benchmark_id
    # hardware identity
    assert result.hardware["cpu_model"] == "Intel(R) Core(TM) Ultra 7 268V"
    assert result.hardware["device.GPU"] == "Intel(R) Arc(TM) Graphics (iGPU)"
    assert result.hardware["devices"] == "CPU, GPU, NPU"
    # software identity
    assert "python" in result.package_versions
    # runtime identity
    assert result.runtime_properties == {"ATTENTION_BACKEND": "SDPA"}
    assert result.compile_seconds == pytest.approx(0.75)
    assert result.settings.max_new_tokens == DEFAULT_MAX_NEW_TOKENS
    assert result.settings_fingerprint == result.settings.fingerprint()
    assert factory.calls[0][1] == "GPU"
    assert factory.calls[0][3] == {"ATTENTION_BACKEND": "SDPA"}
    # per-run measurements
    measured = next(run for run in result.run_metrics if not run.warmup)
    assert measured.input_tokens == 42
    assert measured.output_tokens == 256
    assert measured.ttft_ms == pytest.approx(100.0)
    assert measured.tpot_ms == pytest.approx(50.0)
    assert measured.wall_seconds == pytest.approx(10.0)


def test_the_manifest_hash_is_recorded_when_there_is_a_manifest(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """The artifact's manifest digest pins the exact build that was measured."""
    ctx, _ = job_ctx
    artifact = make_artifact(tmp_path)
    service, _ = make_service(tmp_path, ScriptedRuntime())

    without = service.run(artifact, device="CPU", suite="quick", runs=1, warmups=0, ctx=ctx)
    assert without.artifact_manifest_sha256 is None

    manifest_dir = artifact / ".yonga"
    manifest_dir.mkdir(exist_ok=True)
    (manifest_dir / "manifest.json").write_text("{}\n", encoding="utf-8")

    with_manifest = service.run(artifact, device="CPU", suite="quick", runs=1, warmups=0, ctx=ctx)
    assert with_manifest.artifact_manifest_sha256 == (
        "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"
    )


def test_a_missing_peak_rss_does_not_break_the_result(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """Peak RSS is best-effort: absent means None, not a failed benchmark."""
    ctx, _ = job_ctx
    service, _ = make_service(tmp_path, ScriptedRuntime(), peak_memory=lambda: None)

    result = service.run(
        make_artifact(tmp_path), device="CPU", suite="quick", runs=2, warmups=0, ctx=ctx
    )

    assert all(run.peak_rss_bytes is None for run in result.run_metrics)
    assert all(run.error is None for run in result.run_metrics)
    assert result.errors == []


def test_a_raising_peak_rss_reader_does_not_break_the_result(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """A memory reader that explodes must not take the benchmark with it."""

    def exploding() -> int | None:
        raise OSError("procfs is gone")

    ctx, _ = job_ctx
    service, _ = make_service(tmp_path, ScriptedRuntime(), peak_memory=exploding)
    result = service.run(
        make_artifact(tmp_path), device="CPU", suite="quick", runs=1, warmups=0, ctx=ctx
    )
    assert result.run_metrics[0].peak_rss_bytes is None


def test_peak_rss_is_read_on_this_host() -> None:
    """The default reader really works here; it is allowed to be unavailable."""
    value = peak_rss_bytes()
    assert value is None or value > 0


# --- errors ---------------------------------------------------------------------------


def test_a_run_that_raises_is_an_error_entry_and_the_others_still_complete(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """One bad run must not cost the measurements of the others."""
    ctx, _ = job_ctx
    runtime = ScriptedRuntime(
        {
            "short-instruction": [
                make_generation(ttft_ms=100.0),
                RuntimeError("Exception from src/inference: out of device memory"),
                make_generation(ttft_ms=300.0),
            ]
        }
    )
    service, _ = make_service(tmp_path, runtime)

    result = service.run(
        make_artifact(tmp_path), device="CPU", suite="quick", runs=3, warmups=0, ctx=ctx
    )

    assert len(result.run_metrics) == 3
    failed = [run for run in result.run_metrics if run.error is not None]
    assert len(failed) == 1
    assert "out of device memory" in (failed[0].error or "")
    assert failed[0].run_index == 1
    assert len(result.errors) == 1
    assert "short-instruction run 2" in result.errors[0]
    # The failed run is excluded from the aggregate; the other two are not.
    aggregate = result.aggregate_for("short-instruction")
    assert aggregate is not None
    assert aggregate.runs == 2
    assert aggregate.ttft_ms_mean == pytest.approx(200.0)


def test_a_failed_load_is_recorded_rather_than_raised(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """A model that will not load still produces a saved, honest record."""
    ctx, _ = job_ctx
    settings = make_settings(tmp_path)
    factory = RecordingFactory(error=RuntimeError("NPU compiler rejected the graph"))
    service = BenchmarkService(
        factory, LocalArtifactStore(settings.paths), FakeHardwareProbe(), settings
    )

    result = service.run(make_artifact(tmp_path), device="NPU", suite="quick", ctx=ctx)

    assert result.run_metrics == []
    assert result.compile_seconds is None
    assert result.errors and "NPU compiler rejected the graph" in result.errors[0]
    assert result.aggregates[0].runs == 0


def test_an_overrun_run_is_recorded_as_a_timeout_error(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """A run over its budget is an error entry, excluded from the aggregate."""
    ctx, _ = job_ctx
    runtime = ScriptedRuntime(
        {"short-instruction": [make_generation(wall_seconds=90.0), make_generation()]}
    )
    service, _ = make_service(tmp_path, runtime)

    result = service.run(
        make_artifact(tmp_path),
        device="CPU",
        suite="quick",
        runs=2,
        warmups=0,
        ctx=ctx,
        timeout_seconds=0.0,
    )

    assert all("budget" in (run.error or "") for run in result.run_metrics)
    assert result.aggregate_for("short-instruction") is not None
    assert result.aggregate_for("short-instruction").runs == 0  # type: ignore[union-attr]


def test_the_runtime_is_always_closed(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """A benchmark must not leak a compiled pipeline, even when a run failed."""
    ctx, _ = job_ctx
    runtime = ScriptedRuntime({"short-instruction": [RuntimeError("boom")]})
    service, _ = make_service(tmp_path, runtime)
    service.run(make_artifact(tmp_path), device="CPU", suite="quick", runs=1, warmups=0, ctx=ctx)
    assert runtime.closed is True


def test_an_invalid_run_count_is_refused(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """Zero runs is a user error, not an empty benchmark."""
    ctx, _ = job_ctx
    service, _ = make_service(tmp_path, ScriptedRuntime())
    with pytest.raises(ConfigurationError) as excinfo:
        service.run(make_artifact(tmp_path), device="CPU", suite="quick", runs=0, ctx=ctx)
    assert excinfo.value.diagnostic.code == "bench.count.invalid"


# --- persistence ----------------------------------------------------------------------


def test_the_saved_json_validates_and_round_trips(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """The document on disk is exactly what BenchmarkResult.model_validate accepts."""
    ctx, sink = job_ctx
    settings = make_settings(tmp_path)
    service, _ = make_service(tmp_path, ScriptedRuntime())

    result = service.run(
        make_artifact(tmp_path), device="CPU", suite="standard", runs=2, warmups=1, ctx=ctx
    )

    path = benchmark_result_path(settings.paths, result)
    assert path.is_file()
    assert path.parent == settings.paths.benchmarks_dir

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["suite"] == "standard"
    assert len(document["run_metrics"]) == 9

    restored = BenchmarkResult.model_validate(document)
    assert restored.to_json() == result.to_json()
    assert load_result(path).benchmark_id == result.benchmark_id
    assert any("Benchmark saved" in event.message for event in sink.events)


def test_saving_can_be_turned_off(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """A dry benchmark leaves nothing behind."""
    ctx, _ = job_ctx
    settings = make_settings(tmp_path)
    service, _ = make_service(tmp_path, ScriptedRuntime())
    result = service.run(
        make_artifact(tmp_path), device="CPU", suite="quick", runs=1, ctx=ctx, save=False
    )
    assert not benchmark_result_path(settings.paths, result).exists()


def test_the_benchmark_id_is_file_safe(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """The id is used as a file name, so it may not carry a path separator."""
    ctx, _ = job_ctx
    service, _ = make_service(tmp_path, ScriptedRuntime())
    result = service.run(make_artifact(tmp_path), device="CPU", suite="quick", runs=1, ctx=ctx)
    assert "/" not in result.benchmark_id
    assert result.benchmark_id.startswith("20")
    assert "cpu" in result.benchmark_id


# --- comparability --------------------------------------------------------------------


def _two_results(
    tmp_path: Path,
    ctx: JobContext,
    *,
    second_settings: GenerationSettings | None = None,
    second_device: str = "CPU",
    second_suite: str = "quick",
) -> tuple[BenchmarkResult, BenchmarkResult, BenchmarkService]:
    """Run the same benchmark twice, varying only what the caller asks for."""
    service, _ = make_service(tmp_path, ScriptedRuntime())
    baseline = service.run(
        make_artifact(tmp_path), device="CPU", suite="quick", runs=2, warmups=0, ctx=ctx
    )
    candidate = service.run(
        make_artifact(tmp_path),
        device=second_device,
        suite=second_suite,
        runs=2,
        warmups=0,
        ctx=ctx,
        generation=second_settings,
    )
    return baseline, candidate, service


def test_two_identical_runs_are_comparable(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """Same suite, same device, same settings, same versions: comparable."""
    ctx, _ = job_ctx
    baseline, candidate, service = _two_results(tmp_path, ctx)

    comparison = service.compare(baseline, candidate)

    assert comparison.comparable is True
    assert comparison.reasons == []
    assert comparison.diagnostic is None
    assert [row.prompt_id for row in comparison.rows] == ["short-instruction"]
    assert comparison.rows[0].throughput_ratio == pytest.approx(1.0)


def test_comparison_refuses_differing_generation_settings_and_says_why(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """docs/TESTING.md 9: incompatible settings are refused, with the reason."""
    ctx, _ = job_ctx
    baseline, candidate, service = _two_results(
        tmp_path,
        ctx,
        second_settings=GenerationSettings(max_new_tokens=512, do_sample=False),
    )

    comparison = service.compare(baseline, candidate)

    assert comparison.comparable is False
    assert "generation settings differ" in comparison.reasons
    assert comparison.rows == []
    assert comparison.diagnostic is not None
    assert comparison.diagnostic.code == "bench.not_comparable"
    assert "generation settings differ" in comparison.diagnostic.message
    assert comparison.baseline_id == baseline.benchmark_id
    assert comparison.candidate_id == candidate.benchmark_id


def test_comparison_refuses_a_different_device_and_says_which(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """SPEC 6.5: do not compare different devices."""
    ctx, _ = job_ctx
    baseline, candidate, service = _two_results(tmp_path, ctx, second_device="GPU")

    comparison = service.compare(baseline, candidate)

    assert comparison.comparable is False
    assert any("devices differ (CPU vs GPU)" in reason for reason in comparison.reasons)
    assert comparison.rows == []


def test_comparison_refuses_a_different_prompt_set(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """A longer suite is a different experiment, not a better score."""
    ctx, _ = job_ctx
    baseline, candidate, service = _two_results(tmp_path, ctx, second_suite="standard")

    comparison = service.compare(baseline, candidate)

    assert comparison.comparable is False
    assert "prompt sets differ" in comparison.reasons


def test_comparison_refuses_an_edited_prompt(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """Same prompt id, different text: the digest is what decides."""
    ctx, _ = job_ctx
    baseline, candidate, service = _two_results(tmp_path, ctx)
    tampered = candidate.model_copy(deep=True)
    for run in tampered.run_metrics:
        run.prompt_sha256 = "0" * 64

    comparison = service.compare(baseline, tampered)

    assert comparison.comparable is False
    assert "prompt texts differ for the same prompt ids" in comparison.reasons


def test_comparison_refuses_a_different_openvino_version(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """Numbers from two OpenVINO releases are not the same experiment."""
    ctx, _ = job_ctx
    baseline, candidate, service = _two_results(tmp_path, ctx)
    other = candidate.model_copy(deep=True)
    other.package_versions["openvino"] = "2025.3.0"

    comparison = service.compare(baseline, other)

    assert comparison.comparable is False
    assert any("openvino versions differ" in reason for reason in comparison.reasons)


# --- device and pipeline selection ----------------------------------------------------


def test_the_device_defaults_to_cpu_without_a_manifest(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """With nothing to go on, benchmark where every host can run."""
    ctx, _ = job_ctx
    service, factory = make_service(tmp_path, ScriptedRuntime())
    result = service.run(make_artifact(tmp_path), suite="quick", runs=1, ctx=ctx)
    assert result.device == "CPU"
    assert factory.calls[0][1] == "CPU"
    assert factory.calls[0][2] is PipelineKind.LLM


def test_a_directory_outside_the_catalogue_is_accepted(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """Benchmarking a bare export directory must not require a catalogue entry."""
    ctx, _ = job_ctx
    loose = tmp_path / "somewhere" / "export"
    loose.mkdir(parents=True)
    service, _ = make_service(tmp_path, ScriptedRuntime())
    result = service.run(loose, device="CPU", suite="quick", runs=1, ctx=ctx)
    assert result.artifact_path == loose
    assert result.repo_id == "export"
