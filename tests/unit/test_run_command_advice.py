"""What the commands that run a prepared artifact tell the user, and only once.

``chat``, ``bench``, ``probe``, ``eval`` and ``publish`` share their advice: a
failure on one device offers the same command on a device the artifact already
passed on (never a new ``yonga prepare`` of the model in hand), a command with
no job never prints a ``JOB_ID`` placeholder, a model id that is not prepared
gets its ``yonga prepare`` command, and a model id that matched several ready
artifacts says which one ran.
"""

from __future__ import annotations

import io
import json
import shlex
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from tests.unit.test_artifact_store import make_manifest as store_manifest
from tests.unit.test_artifact_store import publish
from tests.unit.test_benchmark_service import (
    FakeHardwareProbe,
    RecordingFactory,
    ScriptedRuntime,
    make_artifact,
    make_service,
)
from tests.unit.test_chat_service import NPU_COMPILER_PASS_TAIL, failing_service, make_manifest
from tests.unit.test_cli import (
    FakeRuntimeFactory,
    FakeWiring,
    RefusingRuntimeFactory,
    _chat_wiring,
    _root,
    _staged_artifact,
    invoke,
)
from tests.unit.test_prepare_planning import (
    SAFETENSORS_REPO,
    FakeExporter,
    FakeHub,
    FakeRuntimeProbe,
    build_service,
    make_snapshot,
    run_prepare,
    safetensors_metadata,
    settings_for,
)
from yonga.application import export_reuse
from yonga.application.artifact_refs import artifact_not_found, newest_choice_note
from yonga.application.artifacts import ArtifactStore
from yonga.application.benchmark import BenchmarkService
from yonga.application.chat import ChatOptions
from yonga.application.jobs import JobContext, JobStateStore
from yonga.application.probe import LocalArtifactStore, resolve_artifact
from yonga.application.progress import CollectingSink
from yonga.config.settings import Settings
from yonga.domain.artifacts import (
    ArtifactManifest,
    ArtifactSummary,
    DeviceProbeReport,
    ProbeResult,
    ProbeStatus,
    manifest_path,
)
from yonga.domain.diagnostics import (
    RERUN_PLACEHOLDER,
    CompileError,
    ConfigurationError,
    Diagnostic,
    ExitCode,
    FailureCategory,
    YongaError,
    diagnostic,
    for_artifact_run,
    with_rerun,
    without_job,
)
from yonga.domain.plans import ExportRequest, ExportResult, JobStage, JobStatus
from yonga.registry.loader import load_builtin_registry
from yonga.registry.resolver import RegistryResolver
from yonga.ui.chat_tui import render_answer
from yonga.ui.console import diagnostic_lines
from yonga.util.jsonlog import null_logger

ARTIFACT_ID = "e3014dda15c8a5be"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """A private XDG root, as ``test_cli`` uses."""
    root = tmp_path / "home"
    (root / "config").mkdir(parents=True)
    return root


@pytest.fixture
def job_ctx(tmp_path: Path) -> Iterator[tuple[JobContext, CollectingSink]]:
    """A job context that logs nowhere and collects its progress events."""
    settings = settings_for(tmp_path)
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


def npu_refusal(**fields: object) -> Diagnostic:
    """The registry's NPU compiler refusal, as ``classify`` hands it back for a model."""
    resolver = RegistryResolver(load_builtin_registry())
    found = resolver.classify(raw_error=NPU_COMPILER_PASS_TAIL, device="NPU", stage="chat")
    assert found.matched_rule_id == "npu-compiler-pass-failed"
    context = {**found.context, "repo_id": "Qwen/Qwen3-0.6B"}
    return found.model_copy(update={"context": context, **fields})


def passed_on(*devices: str) -> dict[str, DeviceProbeReport]:
    """Device probe reports that passed on ``devices``."""
    return {
        device: DeviceProbeReport(
            device=device,
            status=ProbeStatus.OK,
            submodels=[ProbeResult(device=device, status=ProbeStatus.OK)],
        )
        for device in devices
    }


# --- the shared rewrite ---------------------------------------------------------------


def test_a_fallback_the_artifact_runs_on_is_offered_as_the_same_command() -> None:
    """The reproduced retest: a GPU artifact refused by the NPU compiler in ``chat``."""
    item = for_artifact_run(
        npu_refusal(),
        run_command=f"yonga chat {ARTIFACT_ID}",
        devices=["GPU"],
        has_job=False,
    )

    assert (
        f"Run this artifact on the GPU instead: yonga chat {ARTIFACT_ID} --device gpu"
        in item.remediation
    )
    assert not any("yonga prepare" in step for step in item.remediation)
    assert not any("JOB_ID" in step or "yonga logs" in step for step in item.remediation)
    lines = diagnostic_lines(item)
    assert "Recommended fallback: GPU, CPU" in lines
    assert f"Run: yonga chat {ARTIFACT_ID} --device gpu" in lines
    assert not any("JOB_ID" in line for line in lines)


def test_a_fallback_the_artifact_never_passed_on_keeps_the_prepare_advice() -> None:
    """An NPU-only artifact was never shown to run on the GPU: preparing one is right."""
    item = for_artifact_run(
        npu_refusal(),
        run_command=f"yonga chat {ARTIFACT_ID}",
        devices=["NPU"],
        has_job=False,
    )

    assert any(
        step.endswith("yonga prepare Qwen/Qwen3-0.6B --device gpu") for step in item.remediation
    )
    assert "Run: yonga prepare Qwen/Qwen3-0.6B --device gpu" in diagnostic_lines(item)


def test_a_command_with_a_job_keeps_the_job_log_step_for_its_id() -> None:
    """``bench`` and ``probe`` open a job: the step is filled once the id is known."""
    item = for_artifact_run(
        npu_refusal(),
        run_command=f"yonga bench {ARTIFACT_ID}",
        devices=["GPU"],
        has_job=True,
    )
    filled = item.model_copy(update={"job_id": "2f4d7f83-0864"})

    assert any("yonga logs 2f4d7f83-0864 --raw" in line for line in diagnostic_lines(filled))


def test_without_job_keeps_a_diagnostic_that_has_its_job() -> None:
    item = npu_refusal(job_id="abc")
    assert without_job(item) is item
    assert not any("JOB_ID" in step for step in without_job(npu_refusal()).remediation)


# --- chat ----------------------------------------------------------------------------


def test_chat_on_a_gpu_artifact_refused_by_the_npu_offers_chat_on_the_gpu(
    tmp_path: Path,
) -> None:
    service, artifact = failing_service(tmp_path, NPU_COMPILER_PASS_TAIL)
    manifest = make_manifest(artifact_id=ARTIFACT_ID, device_probes=passed_on("GPU"))
    manifest_path(artifact).write_text(manifest.to_json_text(), encoding="utf-8")
    session = service.open_artifact(artifact, options=ChatOptions(device="npu", command="chat"))

    with pytest.raises(YongaError) as caught:
        session.load()

    report = caught.value.diagnostic
    assert (
        f"Run this artifact on the GPU instead: yonga chat {ARTIFACT_ID} --device gpu"
        in report.remediation
    )
    assert not any("JOB_ID" in step for step in report.remediation)
    assert report.context["fallback_command"] == f"yonga chat {ARTIFACT_ID} --device gpu"


def test_chat_prints_no_job_placeholder_for_an_npu_refusal(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """``chat`` opens no job, so "yonga logs JOB_ID --raw" can only mislead."""
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    manifest = make_manifest(artifact_id=ARTIFACT_ID, device_probes=passed_on("GPU"))
    target = manifest_path(artifact)
    target.parent.mkdir(parents=True)
    target.write_text(manifest.to_json_text(), encoding="utf-8")
    error = CompileError(
        diagnostic(
            "runtime.genai.load_failed",
            FailureCategory.UNKNOWN_RUNTIME_ERROR,
            "Could not load the model on NPU",
            NPU_COMPILER_PASS_TAIL,
            raw_excerpt=NPU_COMPILER_PASS_TAIL,
        )
    )

    result = invoke(
        runner,
        home,
        ["chat", str(artifact), "--device", "npu"],
        factory=_chat_wiring(RefusingRuntimeFactory(error)),
        stdin="",
    )

    err = " ".join(result.stderr.split())
    assert result.exit_code == int(ExitCode.COMPILE), result.output
    assert "JOB_ID" not in err
    assert f"Run: {_root(home)} chat {ARTIFACT_ID} --device gpu" in err
    assert " prepare " not in err


def test_a_registry_step_needing_a_job_is_dropped_for_a_command_without_one(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """The last line of defence: a classified failure in a job-less command.

    Without a manifest nothing is known about the artifact, so the advice is not
    rewritten; the step that needs a job is still never printed.
    """
    error = CompileError(npu_refusal())
    result = invoke(
        runner,
        home,
        ["chat", str(_staged_artifact(tmp_path)), "--device", "npu"],
        factory=_chat_wiring(RefusingRuntimeFactory(error)),
        stdin="",
    )
    err = " ".join(result.stderr.split())
    assert result.exit_code == int(ExitCode.COMPILE), result.output
    assert "The NPU compiler refused this graph" in err
    assert "JOB_ID" not in err
    assert "yonga logs" not in err


# --- bench and probe -----------------------------------------------------------------


def _artifact_with_manifest(tmp_path: Path, manifest: ArtifactManifest) -> Path:
    artifact = make_artifact(tmp_path)
    target = manifest_path(artifact)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(manifest.to_json_text(), encoding="utf-8")
    return artifact


def test_bench_on_a_gpu_artifact_refused_by_the_npu_offers_bench_on_the_gpu(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    ctx, _ = job_ctx
    settings = Settings.model_validate(
        {
            "config_dir": tmp_path / "config",
            "cache_dir": tmp_path / "cache",
            "data_dir": tmp_path / "data",
        }
    )
    manifest = make_manifest(artifact_id=ARTIFACT_ID, device_probes=passed_on("GPU"))
    artifact = _artifact_with_manifest(tmp_path, manifest)
    resolver = RegistryResolver(load_builtin_registry())
    service = BenchmarkService(
        RecordingFactory(error=RuntimeError(NPU_COMPILER_PASS_TAIL)),
        LocalArtifactStore(settings.paths),
        FakeHardwareProbe(),
        settings,
        classifier=lambda raw, device: resolver.classify(
            raw_error=raw, device=device, stage="bench"
        ),
    )

    with pytest.raises(YongaError) as caught:
        service.run(artifact, device="NPU", suite="quick", ctx=ctx)

    report = caught.value.diagnostic
    expected = f"yonga bench {ARTIFACT_ID} --suite quick --device gpu"
    assert f"Run this artifact on the GPU instead: {expected}" in report.remediation
    assert report.context["fallback_command"] == expected
    # bench has a job: its id fills the step when the CLI reports the failure.
    assert any("JOB_ID" in step for step in report.remediation)


def test_probe_advice_names_the_probe_of_the_artifact(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    artifact = _staged_artifact(tmp_path)
    manifest = make_manifest(artifact_id=ARTIFACT_ID, device_probes=passed_on("GPU"))
    target = manifest_path(artifact)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(manifest.to_json_text(), encoding="utf-8")
    runtime = FakeRuntimeProbe(compile_failures={"NPU": NPU_COMPILER_PASS_TAIL})

    result = invoke(
        runner,
        home,
        ["probe", str(artifact), "--device", "npu", "--quick"],
        factory=lambda settings: FakeWiring(settings, runtime=runtime),
    )

    err = " ".join(result.stderr.split())
    assert f"Run: {_root(home)} probe {ARTIFACT_ID} --device gpu" in err, result.output
    assert " prepare " not in err


# --- an unknown reference ------------------------------------------------------------


def test_probe_and_bench_tell_a_model_id_to_prepare_it_first(tmp_path: Path) -> None:
    paths = settings_for(tmp_path).paths
    paths.ensure()

    for command in ("probe", "bench"):
        with pytest.raises(ConfigurationError) as caught:
            resolve_artifact(LocalArtifactStore(paths), "acme/not-here", command=command)
        found = caught.value.diagnostic
        assert found.code == "artifact.not_prepared"
        assert "Prepare it first: yonga prepare acme/not-here" in found.remediation
        # The command to repeat is the user's own, which only the CLI knows; it
        # is never rebuilt from the reference, and never shown as a placeholder.
        assert f"Then run the same command again: {RERUN_PLACEHOLDER}" in found.remediation
        assert f"yonga {command} acme/not-here" not in " ".join(found.remediation)
        assert "Then run the same command again." in with_rerun(found, None).remediation
        assert not any("MODEL" in step for step in found.remediation)


def test_an_artifact_id_that_is_gone_is_pointed_at_yonga_list(tmp_path: Path) -> None:
    paths = settings_for(tmp_path).paths
    paths.ensure()

    with pytest.raises(ConfigurationError) as caught:
        resolve_artifact(LocalArtifactStore(paths), "cc24636a2f49356a")
    found = caught.value.diagnostic
    assert found.code == "artifact.not_found"
    assert found.title == "No prepared artifact has that id"
    assert found.remediation[0] == "See what is already prepared: yonga list"
    assert not any("yonga prepare cc24636a2f49356a" in step for step in found.remediation)
    # Nor a placeholder: pasted, OWNER/NAME is looked up on the Hub as a model.
    assert not any("yonga prepare" in step for step in found.remediation)
    assert not any("OWNER" in step for step in found.remediation)


def test_eval_keeps_its_code_and_gains_the_prepare_advice(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    from yonga.application.evaluation import EvaluationService

    settings = settings_for(tmp_path)
    service = EvaluationService(
        ArtifactStore(settings.paths, settings),
        lambda path, device: pytest.fail("nothing may be scored"),
        settings,
    )

    with pytest.raises(ConfigurationError) as caught:
        service.evaluate("acme/not-here", "acme/also-not-here")
    found = caught.value.diagnostic
    assert found.code == "eval.artifact_not_found"
    assert any(step.endswith("yonga prepare acme/also-not-here") for step in found.remediation)


def test_publish_keeps_its_code_and_gains_the_prepare_advice() -> None:
    found = artifact_not_found("acme/x", command="publish", code="publish.artifact_not_found")
    assert found.code == "publish.artifact_not_found"
    typed = "yonga publish acme/x --repo me/x --yes"
    assert f"Then run the same command again: {typed}" in with_rerun(found, typed).remediation


# --- several ready artifacts ---------------------------------------------------------


def _summary(artifact_id: str, *, profile: str, device: str, hours: int) -> ArtifactSummary:
    return ArtifactSummary(
        artifact_id=artifact_id,
        path=Path("/artifacts") / artifact_id,
        repo_id="Qwen/Qwen3-0.6B",
        profile=profile,
        recommended_device=device,
        created_at=datetime(2026, 10, 11, tzinfo=UTC) + timedelta(hours=hours),
        ready=True,
    )


def test_the_note_names_the_artifact_that_runs_and_how_many_there_were() -> None:
    newest = _summary("e3014dda15c8a5be", profile="gpu-int4-asym-g128-r08", device="GPU", hours=2)
    older = _summary("270a26c6e2903431", profile="cpu-int8-asym-ref", device="CPU", hours=1)

    assert newest_choice_note("Qwen/Qwen3-0.6B", [newest, older]) == (
        "Using e3014dda15c8a5be (gpu-int4-asym-g128-r08, GPU), the newest of 2 ready "
        "artifacts for Qwen/Qwen3-0.6B; pass the artifact id to choose another."
    )
    assert newest_choice_note("Qwen/Qwen3-0.6B", [newest]) is None
    unready = older.model_copy(update={"ready": False})
    assert newest_choice_note("Qwen/Qwen3-0.6B", [newest, unready]) is None


def test_chat_with_a_model_id_says_which_of_its_artifacts_it_uses(
    runner: CliRunner, home: Path
) -> None:
    settings = settings_for(home)
    store = ArtifactStore(settings.paths, settings)
    start = datetime.now(tz=UTC) - timedelta(hours=2)
    for index, (artifact_id, profile, device) in enumerate(
        (("aaaa000000000001", "cpu-int8", "CPU"), ("bbbb000000000002", "gpu-int4-asym-g128", "GPU"))
    ):
        manifest = store_manifest(
            artifact_id=artifact_id,
            profile=profile,
            device=device,
            created_at=start + timedelta(hours=index),
        )
        publish(store, home, manifest=manifest, name=f"staging-{index}")
    repo_id = manifest.resolved_source.repo_id

    result = invoke(
        runner,
        home,
        ["chat", repo_id],
        factory=lambda settings: FakeWiring(settings, runtime_factory=FakeRuntimeFactory()),
        stdin="",
    )

    err = " ".join(result.stderr.split())
    assert (
        "Using bbbb000000000002 (gpu-int4-asym-g128, GPU), the newest of 2 ready artifacts "
        f"for {repo_id}; pass the artifact id to choose another." in err
    ), result.output
    single = invoke(
        runner,
        home,
        ["chat", "aaaa000000000001"],
        factory=lambda settings: FakeWiring(settings, runtime_factory=FakeRuntimeFactory()),
        stdin="",
    )
    assert "the newest of" not in single.stderr


# --- bench prints where it saved once ------------------------------------------------


def test_bench_names_the_saved_file_once(runner: CliRunner, home: Path, tmp_path: Path) -> None:
    artifact = _staged_artifact(tmp_path)
    result = invoke(
        runner,
        home,
        ["bench", str(artifact), "--device", "cpu", "--suite", "quick", "--runs", "1"],
    )

    assert result.exit_code == int(ExitCode.SUCCESS), result.output
    saved = list((home / "data" / "benchmarks").glob("*.json"))
    assert len(saved) == 1
    output = (result.stdout + result.stderr).replace("\n", "")
    assert output.count(saved[0].name) == 1, output


def test_the_bench_service_records_the_saved_file_without_announcing_it(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    ctx, sink = job_ctx
    service, _ = make_service(tmp_path, ScriptedRuntime())

    service.run(make_artifact(tmp_path), device="CPU", suite="quick", runs=1, warmups=0, ctx=ctx)

    assert not any("saved" in event.message.lower() for event in sink.events)


# --- an answer Markdown would draw as nothing ----------------------------------------


def _rendered(text: str) -> str:
    console = Console(file=io.StringIO(), width=60, color_system=None)
    console.print(render_answer(text))
    output = console.file.getvalue()  # type: ignore[attr-defined]
    assert isinstance(output, str)
    return output


@pytest.mark.parametrize(
    ("answer", "shown"),
    [
        ("391.", "391."),
        ("1984)", "1984)"),
        ("391.\n\nThat is the product of 17 and 23.", "391."),
        ("> 391.", "391."),
        ("#", "#"),
        ("```", "```"),
    ],
)
def test_an_answer_markdown_would_hide_is_shown_as_written(answer: str, shown: str) -> None:
    """The retest's empty panel: Qwen3 answered "361." and Markdown drew an empty list."""
    assert shown in _rendered(answer)


def test_a_real_numbered_list_is_still_a_list() -> None:
    rendered = _rendered("1. one\n2. two")
    assert "one" in rendered and "two" in rendered
    assert "1." not in rendered and "\\" not in rendered


# --- a cancelled preparation ---------------------------------------------------------


class CancelledExporter(FakeExporter):
    """An exporter the user interrupted, the way the real one reports it."""

    def export(self, request: ExportRequest, ctx: JobContext) -> ExportResult:
        self.requests.append(request)
        return ExportResult(
            success=False,
            argv=["optimum-cli", "export", "openvino"],
            returncode=-2,
            duration_seconds=9.6,
            stderr_tail=["Traceback (most recent call last):", "KeyboardInterrupt"],
            diagnostic=diagnostic(
                "export.cancelled",
                FailureCategory.USER_CANCELLED,
                "Conversion cancelled",
                "The conversion was cancelled before it finished. No artifact was produced.",
                remediation=[
                    f"Run the same command to start the conversion again: {RERUN_PLACEHOLDER}"
                ],
            ),
        )


def _records(settings: Settings, job_id: str) -> list[Mapping[str, object]]:
    path = settings.paths.job_log_path(job_id)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_cancelled_export_is_recorded_as_cancelled_not_failed(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    hub = FakeHub(
        repos={SAFETENSORS_REPO: safetensors_metadata()},
        snapshots={SAFETENSORS_REPO: make_snapshot(tmp_path, "snapshot-safetensors")},
    )
    service = build_service(settings=settings, hub=hub, exporter=CancelledExporter())
    plan = service.plan(SAFETENSORS_REPO, device="cpu")

    with pytest.raises(YongaError) as caught:
        run_prepare(service, plan, settings)

    assert caught.value.diagnostic.category is FailureCategory.USER_CANCELLED
    events = [(record["level"], record["event"]) for record in _records(settings, plan.job_id)]
    assert ("warning", "export.cancelled") in events
    assert ("warning", "job.cancelled") in events
    assert not any(level == "error" for level, _ in events), events


def test_a_cancelled_job_records_the_profile_its_plan_chose(tmp_path: Path) -> None:
    """The state document said ``profile_id: null`` for a conversion cancelled half way."""
    settings = settings_for(tmp_path)
    hub = FakeHub(
        repos={SAFETENSORS_REPO: safetensors_metadata()},
        snapshots={SAFETENSORS_REPO: make_snapshot(tmp_path, "snapshot-safetensors")},
    )
    service = build_service(settings=settings, hub=hub, exporter=CancelledExporter())
    plan = service.plan(SAFETENSORS_REPO, device="cpu")

    with pytest.raises(YongaError):
        run_prepare(service, plan, settings)

    state = JobStateStore.load(settings.paths.job_state_path(plan.job_id)).state
    assert state.stage is JobStage.CANCELLED
    assert state.profile_id == plan.profile.id
    assert state.target_device == plan.target_device


def test_the_cancelled_conversion_advice_prints_the_users_command(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    hub = FakeHub(
        repos={SAFETENSORS_REPO: safetensors_metadata()},
        snapshots={SAFETENSORS_REPO: make_snapshot(tmp_path, "snapshot-safetensors")},
    )
    args = ["prepare", SAFETENSORS_REPO, "--device", "cpu", "--profile", "cpu-int8-asym-ref"]

    result = invoke(
        runner,
        home,
        [*args, "--yes"],
        factory=lambda settings: FakeWiring(settings, hub=hub, exporter=CancelledExporter()),
    )

    assert result.exit_code == int(ExitCode.CANCELLED), result.output
    err = " ".join(result.stderr.split())
    expected = f"{_root(home)} {shlex.join(args)} --yes"
    assert f"Run the same command to start the conversion again: {expected}" in err
    assert RERUN_PLACEHOLDER not in err


class InterruptingRuntimeProbe(FakeRuntimeProbe):
    """Ctrl-C lands while the device probe compiles, after the export finished."""

    def compile(
        self, path: Path, device: str, properties: Mapping[str, str] | None = None
    ) -> ProbeResult:
        raise KeyboardInterrupt


def test_a_retry_after_a_cancel_says_the_export_came_from_a_cancelled_job(
    tmp_path: Path,
) -> None:
    settings = settings_for(tmp_path)
    hub = FakeHub(
        repos={SAFETENSORS_REPO: safetensors_metadata()},
        snapshots={SAFETENSORS_REPO: make_snapshot(tmp_path, "snapshot-safetensors")},
    )
    interrupted = build_service(settings=settings, hub=hub, runtime=InterruptingRuntimeProbe())
    first = interrupted.plan(SAFETENSORS_REPO, device="gpu")
    with pytest.raises(KeyboardInterrupt):
        run_prepare(interrupted, first, settings)
    staging = settings.paths.staging_dir / first.job_id / "model"
    assert export_reuse.checkpoint_path(staging).is_file()

    service = build_service(settings=settings, hub=hub)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    assert plan.reusable_export is not None
    assert plan.reusable_export.job_status is JobStatus.CANCELLED
    assert plan.reusable_export.job_label == "cancelled job"

    outcome = run_prepare(service, plan, settings)
    assert any(f"reused from cancelled job {first.job_id}" in n for n in outcome.manifest.notes)
    assert not any("failed job" in note for note in outcome.manifest.notes)


def test_an_unknown_source_job_is_called_an_earlier_one() -> None:
    from yonga.domain.plans import ReusableExport

    reusable = ReusableExport(
        job_id="j", path=Path("/x"), completed_at=datetime.now(tz=UTC), fingerprint="f"
    )
    assert reusable.job_label == "earlier job"
    assert reusable.model_copy(update={"job_status": JobStatus.FAILED}).job_label == "failed job"
