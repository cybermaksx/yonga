"""Probe service tests: roles, CPU-only tokenizers, classification, honest auto mode.

The runtime probe is faked so these run on any machine, but every raw error string
below was captured from this project's reference host (see ``HOST_FACTS.md``) and
is matched against the real shipped registry — the classification assertions are
therefore about real behaviour, not about a mock.
"""

from __future__ import annotations

import json
import shlex
import textwrap
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.support import SmokeChildRunner
from yonga.adapters.smoke_child import ChildProcessSmokeTester, verdict_line
from yonga.application.jobs import JobContext
from yonga.application.probe import (
    DEFAULT_DEVICE_ORDER,
    TOKENIZER_DEVICE,
    ArtifactStoreLike,
    LocalArtifactStore,
    ProbeOutcome,
    ProbeService,
    resolve_artifact,
    role_device_matrix,
)
from yonga.application.progress import CollectingSink, ProgressKind
from yonga.config.settings import Settings
from yonga.domain.artifacts import (
    ArtifactManifest,
    DeviceProbeReport,
    ExportRecord,
    IRDescription,
    ManifestModelRef,
    PortDescription,
    ProbeResult,
    ProbeStatus,
    RecommendedRuntime,
    SubmodelRole,
    manifest_path,
    role_for_filename,
)
from yonga.domain.diagnostics import (
    ConfigurationError,
    ExitCode,
    FailureCategory,
    Severity,
    YongaError,
)
from yonga.domain.models import PipelineKind
from yonga.registry.loader import load_builtin_registry
from yonga.registry.resolver import RegistryResolver
from yonga.util.jsonlog import JobLogger, null_logger

# --- real host facts ------------------------------------------------------------------

HOST_VERSIONS = {
    "openvino": "2026.4.0-22959-99c81491cc3-releases/2026/4",
    "openvino_genai": "2026.4.0.0",
    "openvino_tokenizers": "2026.4.0.0",
    "transformers": "5.5.4",
}

#: Captured from ``core.compile_model(read_model("openvino_model.xml"), "NPU")``.
NPU_DYNAMIC_SHAPE_ERROR = textwrap.dedent(
    """
    Exception from src/inference/src/cpp/core.cpp:117:
    Exception from src/inference/src/dev/plugin.cpp:54:
    Exception from src/plugins/intel_npu/src/plugin/src/plugin.cpp:552:
    Exception from src/plugins/intel_npu/src/compiler_adapter/src/ze_graph_ext_wrappers.cpp:425:
    Compilation failed. Level0 pfnCreate2 result: ZE_RESULT_ERROR_INVALID_ARGUMENT, code 0x78000004
     - generic error code for invalid arguments . [NPU_DRV] Driver reports a failure from
    vclAllocatedExecutableCreate4, return code: 2013265924
    [NPU_VCL] Compiler returned msg:
    Exception from src/core/src/partial_shape.cpp:261:
    to_shape was called on a dynamic shape.
    """
).strip()

#: Captured from ``Core().read_model("openvino_tokenizer.xml")`` before repair.
TOKENIZER_STRING_ERROR = textwrap.dedent(
    """
    Exception from src/inference/src/cpp/core.cpp:84:
    Check 'is_valid_type_idx(type_idx)' failed at src/core/src/type/element_type.cpp:115:
    Unsupported element type: STRiNG
    """
).strip()

#: Captured from ``Core().read_model()`` on a truncated IR on the reference host.
MALFORMED_IR_ERROR = (
    "RuntimeError: Exception from src/inference/src/cpp/core.cpp:84:\n"
    "Check 'res.status == pugi::status_ok' failed at "
    "src/frontends/ir/src/input_model.cpp:219:\n"
    "Start-end tags mismatch at offset 27"
)

#: Every IR file name the OpenVINO GenAI export writes (docs/MODEL_PIPELINE.md 8).
GENAI_IR_NAMES: tuple[tuple[str, SubmodelRole], ...] = (
    ("openvino_language_model.xml", SubmodelRole.LANGUAGE),
    ("openvino_text_embeddings_model.xml", SubmodelRole.TEXT_EMBEDDINGS),
    ("openvino_vision_embeddings_model.xml", SubmodelRole.VISION_EMBEDDINGS),
    ("openvino_vision_embeddings_merger_model.xml", SubmodelRole.VISION_EMBEDDINGS_MERGER),
    ("openvino_vision_embeddings_pos_model.xml", SubmodelRole.VISION_EMBEDDINGS_POS),
    ("openvino_resampler_model.xml", SubmodelRole.RESAMPLER),
    ("openvino_tokenizer.xml", SubmodelRole.TOKENIZER),
    ("openvino_detokenizer.xml", SubmodelRole.DETOKENIZER),
    ("openvino_model.xml", SubmodelRole.LANGUAGE),
)


# --- doubles --------------------------------------------------------------------------


class FakeRuntimeProbe:
    """A RuntimeProbe that reads and compiles nothing but answers like the real one."""

    def __init__(
        self,
        *,
        devices: Sequence[str] = ("CPU", "GPU", "NPU"),
        compile_errors: Mapping[str, str] | None = None,
        unreadable: Mapping[str, str] | None = None,
        tokenizer_error: str | None = None,
    ) -> None:
        self.devices = list(devices)
        self.compile_errors = dict(compile_errors or {})
        self.unreadable = dict(unreadable or {})
        self.tokenizer_error = tokenizer_error
        self.compiled: list[tuple[str, str]] = []
        self.tokenizer_calls = 0

    def available_devices(self) -> list[str]:
        return list(self.devices)

    def read_ir(self, path: Path) -> IRDescription:
        role = role_for_filename(path.name)
        read_error = self.unreadable.get(path.name)
        if read_error is not None:
            return IRDescription(path=path, role=role, readable=False, read_error=read_error)
        return IRDescription(
            path=path,
            role=role,
            friendly_name=path.stem,
            inputs=[PortDescription(name="input_ids", element_type="i64", shape="[?,?]")],
            outputs=[PortDescription(name="logits", element_type="f32", shape="[?,?,32000]")],
            readable=True,
            bytes_total=1024,
        )

    def compile(
        self,
        path: Path,
        device: str,
        properties: Mapping[str, str] | None = None,
    ) -> ProbeResult:
        self.compiled.append((path.name, device))
        error = self.compile_errors.get(device) or self.compile_errors.get(f"{device}:{path.name}")
        if error is not None:
            return ProbeResult(
                device=device,
                role=role_for_filename(path.name),
                path=path,
                status=ProbeStatus.FAILED,
                duration_seconds=0.01,
                message=error,
                properties=dict(properties or {}),
            )
        return ProbeResult(
            device=device,
            role=role_for_filename(path.name),
            path=path,
            status=ProbeStatus.OK,
            duration_seconds=0.01,
            properties=dict(properties or {}),
        )

    def read_tokenizer(self, artifact: Path) -> ProbeResult:
        self.tokenizer_calls += 1
        if self.tokenizer_error is not None:
            return ProbeResult(
                device=TOKENIZER_DEVICE,
                role=SubmodelRole.TOKENIZER,
                path=artifact,
                status=ProbeStatus.FAILED,
                message=self.tokenizer_error,
            )
        return ProbeResult(
            device=TOKENIZER_DEVICE,
            role=SubmodelRole.TOKENIZER,
            path=artifact,
            status=ProbeStatus.OK,
            duration_seconds=0.01,
        )


# --- builders -------------------------------------------------------------------------


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
    logger = null_logger("job-probe")
    try:
        yield (
            JobContext(
                job_id="job-probe",
                logger=logger,
                paths=settings.paths,
                settings=settings,
                progress=sink,
            ),
            sink,
        )
    finally:
        logger.close()


def resolver() -> RegistryResolver:
    """A resolver over the shipped registry, bound to the reference host versions."""
    return RegistryResolver(load_builtin_registry(), versions=HOST_VERSIONS)


def make_artifact(root: Path, names: Sequence[str] = ()) -> Path:
    """Create an artifact directory holding the named (empty) IR files."""
    artifact = root / "artifact"
    artifact.mkdir(parents=True, exist_ok=True)
    chosen = names or (
        "openvino_model.xml",
        "openvino_tokenizer.xml",
        "openvino_detokenizer.xml",
    )
    for name in chosen:
        (artifact / name).write_text("<net/>", encoding="utf-8")
        (artifact / name.replace(".xml", ".bin")).write_bytes(b"")
    return artifact


def write_manifest(artifact: Path, *, repo_id: str, model_type: str | None = None) -> None:
    """Write a minimal but valid manifest into an artifact directory."""
    manifest = ArtifactManifest(
        artifact_id="artifact-0001",
        created_at=datetime(2026, 9, 22, 12, 0, tzinfo=UTC),
        tool_version="0.1.0",
        requested_model=ManifestModelRef(repo_id=repo_id),
        resolved_source=ManifestModelRef(repo_id=repo_id, resolved_revision="abc123"),
        model_type=model_type,
        export=ExportRecord(profile="cpu-int4-asym-g128"),
    )
    path = manifest_path(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json_text(), encoding="utf-8")


class FakeSmokeTester:
    """A SmokeTester that records its calls and returns a scripted outcome."""

    def __init__(self, *, error: str | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, object]] = []

    def smoke_test(
        self,
        artifact_dir: Path,
        *,
        device: str,
        pipeline: object,
        properties: Mapping[str, str],
        prompt: str,
        max_new_tokens: int,
    ) -> ProbeResult:
        self.calls.append({"device": device, "properties": dict(properties)})
        if self.error is not None:
            return ProbeResult(device=device, status=ProbeStatus.FAILED, message=self.error)
        return ProbeResult(device=device, status=ProbeStatus.OK, message="generated 24 token(s)")


def service(
    runtime: FakeRuntimeProbe,
    tmp_path: Path,
    *,
    store: ArtifactStoreLike | None = None,
    smoke: FakeSmokeTester | None = None,
    child: SmokeChildRunner | None = None,
) -> ProbeService:
    """Build the service under test; ``child`` scripts the ``--timeout`` child process."""
    return ProbeService(
        runtime,
        resolver(),
        store,
        make_settings(tmp_path),
        smoke_tester=smoke,
        isolated_smoke_tester=None if child is None else ChildProcessSmokeTester(child),
    )


# --- role assignment ------------------------------------------------------------------


@pytest.mark.parametrize(("name", "role"), GENAI_IR_NAMES)
def test_every_genai_filename_gets_its_role(name: str, role: SubmodelRole) -> None:
    """Every file name the GenAI export writes maps to the documented role."""
    assert role_for_filename(name) is role


def test_inventory_assigns_a_role_to_every_ir(tmp_path: Path) -> None:
    """The service inventories every top-level IR and records its ports."""
    artifact = make_artifact(tmp_path, [name for name, _ in GENAI_IR_NAMES])
    runtime = FakeRuntimeProbe()
    irs = service(runtime, tmp_path).inventory(artifact)

    assert len(irs) == len(GENAI_IR_NAMES)
    by_name = {ir.path.name: ir.role for ir in irs}
    for name, role in GENAI_IR_NAMES:
        assert by_name[name] is role
    assert all(ir.inputs and ir.outputs for ir in irs)
    assert irs[0].inputs[0].element_type == "i64"
    assert irs[0].outputs[0].shape == "[?,?,32000]"


def test_inventory_ignores_the_tool_owned_metadata_directory(tmp_path: Path) -> None:
    """``.yonga`` is tool metadata, not model content."""
    artifact = make_artifact(tmp_path)
    write_manifest(artifact, repo_id="acme/model")
    (artifact / ".yonga" / "stray.xml").write_text("<net/>", encoding="utf-8")

    irs = service(FakeRuntimeProbe(), tmp_path).inventory(artifact)
    assert [ir.path.name for ir in irs] == [
        "openvino_detokenizer.xml",
        "openvino_model.xml",
        "openvino_tokenizer.xml",
    ]


# --- tokenizers stay on the CPU -------------------------------------------------------


def test_tokenizer_is_probed_on_cpu_even_when_npu_was_requested(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """Tokenizers are string-tensor graphs: they are never sent to the NPU."""
    artifact = make_artifact(tmp_path)
    runtime = FakeRuntimeProbe()
    ctx, _ = job_ctx

    reports = service(runtime, tmp_path).probe(artifact, device="npu", ctx=ctx)

    assert [report.device for report in reports] == ["NPU"]
    tokenizer = [
        result
        for result in reports[0].submodels
        if result.role in {SubmodelRole.TOKENIZER, SubmodelRole.DETOKENIZER}
    ]
    assert tokenizer and all(result.device == "CPU" for result in tokenizer)
    # No tokenizer IR was ever handed to Core.compile_model on the NPU.
    assert all(
        not name.startswith(("openvino_tokenizer", "openvino_detokenizer"))
        for name, _device in runtime.compiled
    )
    assert runtime.compiled == [("openvino_model.xml", "NPU")]
    assert runtime.tokenizer_calls == 1


def test_tokenizer_is_probed_once_and_shared_across_devices(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """Probing three devices must not read the tokenizer three times."""
    artifact = make_artifact(tmp_path)
    runtime = FakeRuntimeProbe()
    ctx, _ = job_ctx

    reports = service(runtime, tmp_path).probe(artifact, device="auto", ctx=ctx)

    assert len(reports) == 3
    assert runtime.tokenizer_calls == 1


def test_an_artifact_without_a_tokenizer_is_not_invented(tmp_path: Path) -> None:
    """No tokenizer IR means no tokenizer probe, rather than a fabricated failure."""
    artifact = make_artifact(tmp_path, ["openvino_model.xml"])
    runtime = FakeRuntimeProbe()

    outcome = service(runtime, tmp_path).probe_detailed(artifact, device="CPU")

    assert outcome.tokenizer is None
    assert runtime.tokenizer_calls == 0
    assert [result.role for result in outcome.reports[0].submodels] == [SubmodelRole.LANGUAGE]


# --- classification -------------------------------------------------------------------


def test_a_failing_compile_is_classified_not_a_raw_traceback(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """The real NPU rejection becomes a typed diagnostic, raw text preserved."""
    artifact = make_artifact(tmp_path)
    write_manifest(artifact, repo_id="acme/model")
    runtime = FakeRuntimeProbe(compile_errors={"NPU": NPU_DYNAMIC_SHAPE_ERROR})
    ctx, sink = job_ctx

    reports = service(runtime, tmp_path).probe(artifact, device="npu", ctx=ctx)

    report = reports[0]
    # No pipeline was available to settle the raw failure, so the device is not
    # passed -- but the raw compile is recorded as deferred, not as the verdict.
    assert report.status is ProbeStatus.FAILED
    failed = next(result for result in report.submodels if result.status is ProbeStatus.DEFERRED)
    assert failed.diagnostic is not None
    assert failed.diagnostic.code == "npu.compiler.dynamic_shape"
    assert failed.diagnostic.category is FailureCategory.COMPILER_UNSUPPORTED_GRAPH
    # "Don't blame the driver": the compiler answered, so the environment is fine.
    assert failed.diagnostic.environment_broken is False
    assert failed.diagnostic.retry_same_device is False
    assert "gpu" in [device.lower() for device in failed.diagnostic.fallback_devices]
    # The raw runtime text survives, both on the result and inside the diagnostic.
    assert "to_shape was called on a dynamic shape" in (failed.message or "")
    assert "to_shape was called on a dynamic shape" in (failed.diagnostic.raw_excerpt or "")
    assert report.diagnostic is not None
    assert report.diagnostic.code == "npu.compiler.dynamic_shape"
    assert "unproven" in report.diagnostic.message
    assert any(event.kind is ProgressKind.PROBE for event in sink.events)


# --- raw NPU compile failures the GenAI pipeline must settle -------------------------


def test_a_deferred_npu_failure_passes_when_the_pipeline_generates(tmp_path: Path) -> None:
    """The dynamic-shape raw failure is overruled by a pipeline that really generates.

    This is the reference-host finding: a raw compile_model of an LLM fails on
    the NPU for every model, while LLMPipeline reshapes and runs.
    """
    artifact = make_artifact(tmp_path)
    write_manifest(artifact, repo_id="acme/model")
    runtime = FakeRuntimeProbe(compile_errors={"NPU": NPU_DYNAMIC_SHAPE_ERROR})
    smoke = FakeSmokeTester()

    report = service(runtime, tmp_path, smoke=smoke).probe(artifact, device="npu")[0]

    assert report.status is ProbeStatus.OK
    assert report.smoke_test is not None and report.smoke_test.status is ProbeStatus.OK
    raw = [result for result in report.submodels if result.status is ProbeStatus.DEFERRED]
    assert raw, "the raw compile attempt must stay on record"
    assert "to_shape was called on a dynamic shape" in (raw[0].message or "")
    assert [call["device"] for call in smoke.calls] == ["NPU"]


def test_a_deferred_npu_failure_fails_with_the_pipeline_error(tmp_path: Path) -> None:
    """When the pipeline fails too, the pipeline's error is the verdict."""
    artifact = make_artifact(tmp_path)
    write_manifest(artifact, repo_id="acme/model", model_type="qwen3_5")
    runtime = FakeRuntimeProbe(compile_errors={"NPU": NPU_DYNAMIC_SHAPE_ERROR})
    smoke = FakeSmokeTester(error="OneShotBufferizeVPU2VPUIP failed : Only supports FP16.")

    report = service(runtime, tmp_path, smoke=smoke).probe(artifact, device="npu")[0]

    assert report.status is ProbeStatus.FAILED
    assert report.diagnostic is not None
    assert report.diagnostic.code == "npu.compiler.unsupported_graph.qwen35"


def test_a_conclusive_failure_never_builds_a_pipeline(tmp_path: Path) -> None:
    """Only a deferred failure is settled on the pipeline; a real one stands."""
    artifact = make_artifact(tmp_path)
    write_manifest(artifact, repo_id="acme/model", model_type="qwen3_5")
    runtime = FakeRuntimeProbe(
        compile_errors={"NPU": "OneShotBufferizeVPU2VPUIP failed : Only supports FP16."}
    )
    smoke = FakeSmokeTester()

    report = service(runtime, tmp_path, smoke=smoke).probe(artifact, device="npu")[0]

    assert report.status is ProbeStatus.FAILED
    assert smoke.calls == []
    assert report.smoke_test is None


def test_a_successful_compile_never_builds_a_pipeline(tmp_path: Path) -> None:
    """A probe stays a cheap compile check when nothing needs settling."""
    artifact = make_artifact(tmp_path)
    write_manifest(artifact, repo_id="acme/model")
    smoke = FakeSmokeTester()

    report = service(FakeRuntimeProbe(), tmp_path, smoke=smoke).probe(artifact, device="gpu")[0]

    assert report.status is ProbeStatus.OK
    assert smoke.calls == []


def test_another_devices_runtime_properties_are_not_applied(tmp_path: Path) -> None:
    """A GPU artifact's recommended properties are not forced onto the NPU pipeline."""
    artifact = make_artifact(tmp_path)
    manifest = ArtifactManifest(
        artifact_id="artifact-0002",
        created_at=datetime(2026, 9, 22, 12, 0, tzinfo=UTC),
        tool_version="0.1.0",
        requested_model=ManifestModelRef(repo_id="acme/model"),
        resolved_source=ManifestModelRef(repo_id="acme/model", resolved_revision="abc123"),
        export=ExportRecord(profile="gpu-int4-asym-g128"),
        recommended_runtime=RecommendedRuntime(
            device="GPU", pipeline=PipelineKind.LLM, properties={"ATTENTION_BACKEND": "SDPA"}
        ),
    )
    path = manifest_path(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json_text(), encoding="utf-8")
    runtime = FakeRuntimeProbe(compile_errors={"NPU": NPU_DYNAMIC_SHAPE_ERROR})
    smoke = FakeSmokeTester()

    service(runtime, tmp_path, smoke=smoke).probe(artifact, device="npu")

    assert smoke.calls == [{"device": "NPU", "properties": {}}]


def test_a_failing_tokenizer_read_is_classified(tmp_path: Path) -> None:
    """The real STRiNG casing defect classifies as an incompatible tokenizer IR."""
    artifact = make_artifact(tmp_path)
    runtime = FakeRuntimeProbe(tokenizer_error=TOKENIZER_STRING_ERROR)

    outcome = service(runtime, tmp_path).probe_detailed(artifact, device="CPU")

    assert outcome.tokenizer is not None
    assert outcome.tokenizer.device == "CPU"
    assert outcome.tokenizer.diagnostic is not None
    assert outcome.tokenizer.diagnostic.category is FailureCategory.TOKENIZER_IR_INCOMPATIBLE
    assert "STRiNG" in (outcome.tokenizer.message or "")
    assert outcome.reports[0].status is ProbeStatus.FAILED


def test_an_unreadable_ir_becomes_a_classified_failure(tmp_path: Path) -> None:
    """An IR that cannot even be read is reported as a diagnostic, not skipped."""
    artifact = make_artifact(tmp_path)
    runtime = FakeRuntimeProbe(unreadable={"openvino_model.xml": MALFORMED_IR_ERROR})

    outcome = service(runtime, tmp_path).probe_detailed(artifact, device="CPU")

    language = next(
        result for result in outcome.reports[0].submodels if result.role is SubmodelRole.LANGUAGE
    )
    assert language.status is ProbeStatus.FAILED
    assert language.diagnostic is not None
    # No registry rule covers a malformed XML, so the honest answer is "unknown" --
    # but it is still a typed diagnostic with the raw text attached, not a traceback.
    assert language.diagnostic.category is FailureCategory.UNKNOWN_RUNTIME_ERROR
    assert "Start-end tags mismatch" in (language.diagnostic.raw_excerpt or "")
    assert "Start-end tags mismatch" in (language.message or "")
    assert runtime.compiled == []


# --- auto mode ------------------------------------------------------------------------


def test_auto_reports_every_candidate_device_honestly(
    tmp_path: Path,
    job_ctx: tuple[JobContext, CollectingSink],
) -> None:
    """Auto probes every available device and never hides the one that failed."""
    artifact = make_artifact(tmp_path)
    runtime = FakeRuntimeProbe(compile_errors={"NPU": NPU_DYNAMIC_SHAPE_ERROR})
    ctx, _ = job_ctx

    outcome = service(runtime, tmp_path).probe_detailed(artifact, device="auto", ctx=ctx)

    assert outcome.candidates == list(DEFAULT_DEVICE_ORDER)
    assert [report.device for report in outcome.reports] == ["NPU", "GPU", "CPU"]
    statuses = {report.device: report.status for report in outcome.reports}
    assert statuses["NPU"] is ProbeStatus.FAILED
    assert statuses["GPU"] is ProbeStatus.OK
    assert statuses["CPU"] is ProbeStatus.OK
    assert outcome.working_devices == ["GPU", "CPU"]
    assert outcome.ok is True


def test_auto_only_probes_devices_openvino_exposes(tmp_path: Path) -> None:
    """A device the runtime cannot see is not a candidate."""
    artifact = make_artifact(tmp_path)
    runtime = FakeRuntimeProbe(devices=("CPU", "GPU"))

    outcome = service(runtime, tmp_path).probe_detailed(artifact, device="auto")

    assert outcome.candidates == ["GPU", "CPU"]


def test_an_explicit_device_never_falls_back(tmp_path: Path) -> None:
    """SPEC 7.1: an explicit --device must not silently move elsewhere."""
    artifact = make_artifact(tmp_path)
    runtime = FakeRuntimeProbe(compile_errors={"NPU": NPU_DYNAMIC_SHAPE_ERROR})

    outcome = service(runtime, tmp_path).probe_detailed(artifact, device="npu")

    assert outcome.candidates == ["NPU"]
    assert [report.device for report in outcome.reports] == ["NPU"]
    assert outcome.ok is False
    assert outcome.working_devices == []


def test_a_device_is_never_ok_when_one_of_its_compiles_failed(tmp_path: Path) -> None:
    """A partly failing device is failed, not "mostly working"."""
    artifact = make_artifact(
        tmp_path,
        [
            "openvino_language_model.xml",
            "openvino_text_embeddings_model.xml",
            "openvino_tokenizer.xml",
        ],
    )
    runtime = FakeRuntimeProbe(
        compile_errors={"GPU:openvino_text_embeddings_model.xml": "boom"},
    )

    outcome = service(runtime, tmp_path).probe_detailed(artifact, device="GPU", all_submodels=True)

    assert outcome.reports[0].status is ProbeStatus.FAILED
    assert outcome.ok is False


# --- submodel selection ---------------------------------------------------------------


def test_default_probes_only_the_language_model(tmp_path: Path) -> None:
    """The cheap default answers "can this run here at all"."""
    artifact = make_artifact(
        tmp_path,
        [
            "openvino_language_model.xml",
            "openvino_text_embeddings_model.xml",
            "openvino_vision_embeddings_model.xml",
            "openvino_tokenizer.xml",
        ],
    )
    runtime = FakeRuntimeProbe()

    service(runtime, tmp_path).probe(artifact, device="GPU")

    assert runtime.compiled == [("openvino_language_model.xml", "GPU")]


def test_all_submodels_probes_every_numeric_ir(tmp_path: Path) -> None:
    """--all-submodels compiles every numeric IR, and still no tokenizer."""
    artifact = make_artifact(
        tmp_path,
        [
            "openvino_language_model.xml",
            "openvino_text_embeddings_model.xml",
            "openvino_vision_embeddings_model.xml",
            "openvino_tokenizer.xml",
            "openvino_detokenizer.xml",
        ],
    )
    runtime = FakeRuntimeProbe()

    service(runtime, tmp_path).probe(artifact, device="GPU", all_submodels=True)

    assert sorted(name for name, _ in runtime.compiled) == [
        "openvino_language_model.xml",
        "openvino_text_embeddings_model.xml",
        "openvino_vision_embeddings_model.xml",
    ]


# --- the documented role/device matrix ------------------------------------------------


def test_role_device_matrix_matches_the_documented_shape(tmp_path: Path) -> None:
    """docs/MODEL_PIPELINE.md 10: {"language": {"GPU": "ok", "NPU": "failed"}}."""
    artifact = make_artifact(tmp_path, ["openvino_language_model.xml"])
    runtime = FakeRuntimeProbe(
        devices=("GPU", "NPU"),
        compile_errors={"NPU": NPU_DYNAMIC_SHAPE_ERROR},
    )

    outcome = service(runtime, tmp_path).probe_detailed(artifact, device="auto")

    assert outcome.role_matrix() == {"language": {"NPU": "failed", "GPU": "ok"}}


def test_role_device_matrix_keeps_the_worse_status(tmp_path: Path) -> None:
    """One role probed twice on one device reports the worse outcome."""
    reports = [
        DeviceProbeReport(
            device="GPU",
            status=ProbeStatus.FAILED,
            submodels=[
                ProbeResult(device="GPU", role=SubmodelRole.LANGUAGE, status=ProbeStatus.OK),
                ProbeResult(device="GPU", role=SubmodelRole.LANGUAGE, status=ProbeStatus.FAILED),
            ],
        )
    ]
    assert role_device_matrix(reports) == {"language": {"GPU": "failed"}}


# --- artifact resolution --------------------------------------------------------------


def test_local_store_resolves_a_directory_and_reads_its_manifest(tmp_path: Path) -> None:
    """A plain directory path resolves to itself and its manifest is parsed."""
    artifact = make_artifact(tmp_path)
    write_manifest(artifact, repo_id="acme/model", model_type="qwen3_5")
    store = LocalArtifactStore(make_settings(tmp_path).paths)

    assert store.find(artifact) == artifact
    manifest = store.manifest_for(artifact)
    assert manifest is not None
    assert manifest.artifact_id == "artifact-0001"
    assert manifest.model_type == "qwen3_5"


def test_local_store_finds_an_artifact_by_repository_id(tmp_path: Path) -> None:
    """A repository id resolves through the catalogue, not by guessing a path."""
    paths = make_settings(tmp_path).paths
    paths.ensure()
    artifact = paths.artifacts_dir / "acme__model__cpu"
    artifact.mkdir(parents=True)
    write_manifest(artifact, repo_id="acme/model")

    assert LocalArtifactStore(paths).find("acme/model") == artifact


def test_local_store_reports_a_miss_rather_than_guessing(tmp_path: Path) -> None:
    """A reference that matches nothing is None, not a nearby directory."""
    paths = make_settings(tmp_path).paths
    paths.ensure()
    assert LocalArtifactStore(paths).find("acme/nope") is None


def test_an_unresolvable_reference_raises_with_remediation(tmp_path: Path) -> None:
    """resolve_artifact turns a miss into a typed, actionable failure."""
    paths = make_settings(tmp_path).paths
    paths.ensure()

    with pytest.raises(ConfigurationError) as excinfo:
        resolve_artifact(LocalArtifactStore(paths), "acme/nope")
    assert excinfo.value.diagnostic.code == "artifact.not_found"
    assert excinfo.value.diagnostic.remediation


def test_the_probe_service_refuses_an_unresolvable_reference(tmp_path: Path) -> None:
    """The refusal reaches the caller of probe(), not a silent empty report."""
    paths = make_settings(tmp_path).paths
    paths.ensure()
    with pytest.raises(ConfigurationError):
        service(FakeRuntimeProbe(), tmp_path).probe("acme/nope", device="CPU")


def test_an_artifact_without_manifest_still_probes(tmp_path: Path) -> None:
    """A bare export directory has no manifest; the probe must not need one."""
    artifact = make_artifact(tmp_path)
    outcome = service(FakeRuntimeProbe(), tmp_path).probe_detailed(artifact, device="CPU")
    assert outcome.reports[0].status is ProbeStatus.OK


# --- the multiprocessing guard --------------------------------------------------------


def test_the_entry_point_is_guarded() -> None:
    """OpenVINO's forkserver re-imports __main__: the entry point must be guarded."""
    source = Path(
        str(
            __import__("yonga.application.probe", fromlist=["__file__"]).__file__,
        )
    ).read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in source
    assert "main()" in source


def test_no_progress_sink_is_required(tmp_path: Path) -> None:
    """The service works without a job context, for one-off diagnostics."""
    artifact = make_artifact(tmp_path)
    reports = service(FakeRuntimeProbe(), tmp_path).probe(artifact, device="CPU")
    assert [report.device for report in reports] == ["CPU"]
    assert reports[0].attempted_at is not None


def test_a_large_pipeline_build_is_announced_as_slow(
    tmp_path: Path, job_ctx: tuple[JobContext, CollectingSink]
) -> None:
    """A 9B NPU build ran past 15 minutes on the reference host; warn before waiting."""
    artifact = make_artifact(tmp_path)
    write_manifest(artifact, repo_id="acme/model")
    with (artifact / "openvino_model.bin").open("wb") as weights:
        weights.truncate(3 * 2**30)  # sparse: no disk is actually used
    runtime = FakeRuntimeProbe(compile_errors={"NPU": NPU_DYNAMIC_SHAPE_ERROR})
    ctx, sink = job_ctx

    service(runtime, tmp_path, smoke=FakeSmokeTester()).probe(artifact, device="npu", ctx=ctx)

    messages = [event.message for event in sink.events]
    assert any("can take many minutes" in message for message in messages)


def test_a_small_pipeline_build_is_not_announced_as_slow(
    tmp_path: Path, job_ctx: tuple[JobContext, CollectingSink]
) -> None:
    artifact = make_artifact(tmp_path)
    write_manifest(artifact, repo_id="acme/model")
    runtime = FakeRuntimeProbe(compile_errors={"NPU": NPU_DYNAMIC_SHAPE_ERROR})
    ctx, sink = job_ctx

    service(runtime, tmp_path, smoke=FakeSmokeTester()).probe(artifact, device="npu", ctx=ctx)

    assert not any("can take many minutes" in event.message for event in sink.events)


# --- --quick and --timeout: a deferred device the probe chose not to settle ------------


def ok_child_line(device: str = "NPU") -> str:
    """What ``python -m yonga.adapters.smoke_child`` prints after generating."""
    result = ProbeResult(device=device, status=ProbeStatus.OK, message="generated 24 token(s)")
    return verdict_line(result)


def test_quick_never_builds_a_pipeline_and_reports_inconclusive(
    tmp_path: Path, job_ctx: tuple[JobContext, CollectingSink]
) -> None:
    """``--quick`` stops at the raw compile: a deferred device is neither ok nor failed."""
    artifact = make_artifact(tmp_path)
    write_manifest(artifact, repo_id="acme/model")
    runtime = FakeRuntimeProbe(compile_errors={"NPU": NPU_DYNAMIC_SHAPE_ERROR})
    smoke = FakeSmokeTester()
    ctx, sink = job_ctx

    outcome = service(runtime, tmp_path, smoke=smoke).probe_detailed(
        artifact, device="npu", quick=True, ctx=ctx
    )

    report = outcome.reports[0]
    assert smoke.calls == []
    assert report.status is ProbeStatus.INCONCLUSIVE
    assert report.smoke_test is None
    assert outcome.ok is False
    assert outcome.inconclusive is True
    assert outcome.quick is True
    # The raw attempt stays on record, still marked deferred.
    assert any(result.status is ProbeStatus.DEFERRED for result in report.submodels)
    assert report.diagnostic is not None
    assert report.diagnostic.code == "probe.inconclusive.quick"
    assert report.diagnostic.severity is Severity.WARNING
    assert report.diagnostic.context["deferred_code"] == "npu.compiler.dynamic_shape"
    assert "deferred" in report.diagnostic.title
    assert any("--timeout" in step for step in report.diagnostic.remediation)
    assert not any("Building the GenAI pipeline" in event.message for event in sink.events)


def test_quick_remediation_repeats_the_probe_as_a_command_that_can_be_pasted(
    tmp_path: Path,
) -> None:
    """The reference is shell-quoted, and ``--all-submodels`` is kept."""
    artifact = make_artifact(tmp_path / "my models")
    write_manifest(artifact, repo_id="acme/model")
    runtime = FakeRuntimeProbe(compile_errors={"NPU": NPU_DYNAMIC_SHAPE_ERROR})

    outcome = service(runtime, tmp_path).probe_detailed(
        str(artifact), device="npu", all_submodels=True, quick=True
    )

    found = outcome.reports[0].diagnostic
    assert found is not None and found.code == "probe.inconclusive.quick"
    settle, bounded = (step.split(": ", 1)[1] for step in found.remediation)
    probe = ["yonga", "probe", str(artifact), "--device", "npu", "--all-submodels"]
    assert shlex.split(settle) == probe
    assert shlex.split(bounded) == [*probe, "--timeout", "SECONDS"]


def test_quick_does_not_change_a_device_that_needs_no_pipeline(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)
    smoke = FakeSmokeTester()

    outcome = service(FakeRuntimeProbe(), tmp_path, smoke=smoke).probe_detailed(
        artifact, device="gpu", quick=True
    )

    assert outcome.reports[0].status is ProbeStatus.OK
    assert outcome.inconclusive is False
    assert smoke.calls == []


def test_quick_keeps_a_definite_failure_on_another_submodel(tmp_path: Path) -> None:
    """An unsettled deferral never hides a submodel that failed outright."""
    artifact = make_artifact(
        tmp_path, ["openvino_language_model.xml", "openvino_vision_embeddings_model.xml"]
    )
    write_manifest(artifact, repo_id="acme/model", model_type="qwen3_5")
    runtime = FakeRuntimeProbe(
        compile_errors={
            "NPU:openvino_language_model.xml": NPU_DYNAMIC_SHAPE_ERROR,
            "NPU:openvino_vision_embeddings_model.xml": (
                "OneShotBufferizeVPU2VPUIP failed : Only supports FP16."
            ),
        }
    )

    outcome = service(runtime, tmp_path).probe_detailed(
        artifact, device="npu", all_submodels=True, quick=True
    )

    report = outcome.reports[0]
    assert report.status is ProbeStatus.FAILED
    assert outcome.inconclusive is False
    assert report.diagnostic is not None
    assert report.diagnostic.code == "npu.compiler.unsupported_graph.qwen35"
    # The language model itself was never settled; only the vision encoder failed.
    assert outcome.role_matrix() == {
        "language": {"NPU": "inconclusive"},
        "vision_embeddings": {"NPU": "failed"},
    }


def test_a_pipeline_build_stopped_at_the_timeout_is_inconclusive(
    tmp_path: Path, job_ctx: tuple[JobContext, CollectingSink]
) -> None:
    """``--timeout`` runs the build in a child; stopping it is not a compiler verdict."""
    artifact = make_artifact(tmp_path)
    write_manifest(artifact, repo_id="acme/model")
    runtime = FakeRuntimeProbe(compile_errors={"NPU": NPU_DYNAMIC_SHAPE_ERROR})
    smoke = FakeSmokeTester()
    child = SmokeChildRunner(timed_out=True, stderr=["[ERROR] [vpux-compiler] still compiling"])
    ctx, sink = job_ctx

    outcome = service(runtime, tmp_path, smoke=smoke, child=child).probe_detailed(
        artifact, device="npu", timeout=30, ctx=ctx
    )

    report = outcome.reports[0]
    assert smoke.calls == [], "a bounded build never runs in-process"
    assert [timeout for _, timeout in child.calls] == [30]
    assert report.status is ProbeStatus.INCONCLUSIVE
    assert report.smoke_test is not None
    assert report.smoke_test.status is ProbeStatus.INCONCLUSIVE
    assert report.diagnostic is not None
    assert report.diagnostic.code == "probe.inconclusive.timeout"
    assert report.diagnostic.category is not FailureCategory.COMPILER_UNSUPPORTED_GRAPH
    assert "30 s" in report.diagnostic.message
    assert outcome.inconclusive is True
    assert outcome.timeout_seconds == 30
    assert outcome.role_matrix()["language"] == {"NPU": "inconclusive"}
    assert any("child process stopped after 30 s" in event.message for event in sink.events)


def test_a_child_that_generates_passes_the_device(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)
    write_manifest(artifact, repo_id="acme/model")
    runtime = FakeRuntimeProbe(compile_errors={"NPU": NPU_DYNAMIC_SHAPE_ERROR})
    child = SmokeChildRunner(stdout=["[ INFO ] noise", ok_child_line()])

    outcome = service(runtime, tmp_path, child=child).probe_detailed(
        artifact, device="npu", timeout=600
    )

    report = outcome.reports[0]
    assert report.status is ProbeStatus.OK
    assert report.smoke_test is not None and report.smoke_test.ok
    assert outcome.ok is True
    assert outcome.role_matrix()["language"] == {"NPU": "ok"}


def test_a_child_crash_is_a_failure_classified_by_the_registry(tmp_path: Path) -> None:
    """No verdict from the child: its stderr is the raw text the registry reads."""
    artifact = make_artifact(tmp_path)
    write_manifest(artifact, repo_id="acme/model", model_type="qwen3_5")
    runtime = FakeRuntimeProbe(compile_errors={"NPU": NPU_DYNAMIC_SHAPE_ERROR})
    child = SmokeChildRunner(
        stderr=["OneShotBufferizeVPU2VPUIP failed : Only supports FP16.", "Aborted"],
        returncode=-6,
    )

    outcome = service(runtime, tmp_path, child=child).probe_detailed(
        artifact, device="npu", timeout=600
    )

    report = outcome.reports[0]
    assert report.status is ProbeStatus.FAILED
    assert outcome.inconclusive is False
    assert report.smoke_test is not None
    assert report.smoke_test.status is ProbeStatus.FAILED
    assert report.diagnostic is not None
    assert report.diagnostic.code == "npu.compiler.unsupported_graph.qwen35"


def test_without_a_timeout_the_pipeline_is_still_built_in_process(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)
    write_manifest(artifact, repo_id="acme/model")
    runtime = FakeRuntimeProbe(compile_errors={"NPU": NPU_DYNAMIC_SHAPE_ERROR})
    smoke = FakeSmokeTester()
    child = SmokeChildRunner(stdout=[ok_child_line()])

    report = service(runtime, tmp_path, smoke=smoke, child=child).probe(artifact, device="npu")[0]

    assert report.status is ProbeStatus.OK
    assert len(smoke.calls) == 1
    assert child.calls == []


def test_the_slow_build_notice_names_the_options_that_bound_it(
    tmp_path: Path, job_ctx: tuple[JobContext, CollectingSink]
) -> None:
    artifact = make_artifact(tmp_path)
    write_manifest(artifact, repo_id="acme/model")
    with (artifact / "openvino_model.bin").open("wb") as weights:
        weights.truncate(3 * 2**30)  # sparse: no disk is actually used
    runtime = FakeRuntimeProbe(compile_errors={"NPU": NPU_DYNAMIC_SHAPE_ERROR})
    ctx, sink = job_ctx

    service(runtime, tmp_path, smoke=FakeSmokeTester()).probe(artifact, device="npu", ctx=ctx)

    notice = next(event.message for event in sink.events if "many minutes" in event.message)
    assert "--timeout" in notice and "--quick" in notice


@pytest.mark.parametrize("timeout", [0, -5, float("nan"), float("inf")])
def test_a_timeout_must_be_a_positive_number_of_seconds(tmp_path: Path, timeout: float) -> None:
    artifact = make_artifact(tmp_path)
    child = SmokeChildRunner()

    with pytest.raises(ConfigurationError) as raised:
        service(FakeRuntimeProbe(), tmp_path, child=child).probe(
            artifact, device="npu", timeout=timeout
        )

    assert raised.value.diagnostic.code == "probe.timeout.invalid"
    assert child.calls == []


def test_quick_and_a_timeout_cannot_be_combined(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)

    with pytest.raises(ConfigurationError) as raised:
        service(FakeRuntimeProbe(), tmp_path, child=SmokeChildRunner()).probe(
            artifact, device="npu", quick=True, timeout=10
        )

    assert raised.value.diagnostic.code == "probe.options.conflict"


@pytest.mark.parametrize(
    ("args", "code"),
    [
        (["--timeout", "0"], "probe.timeout.invalid"),
        (["--timeout", "-1"], "probe.timeout.invalid"),
        (["--timeout", "nan"], "probe.timeout.invalid"),
        (["--quick", "--timeout", "5"], "probe.options.conflict"),
    ],
)
def test_the_manual_entry_point_refuses_bad_options_with_70_not_2(
    args: list[str], code: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """2 means inconclusive there, so an option error must not exit with it."""
    from yonga.application.probe import main

    assert main(["some-artifact", *args]) == int(ExitCode.REGISTRY)
    captured = capsys.readouterr()
    assert code in captured.err
    assert captured.out == ""


def test_a_timeout_without_a_child_process_tester_is_refused(tmp_path: Path) -> None:
    """Never a silent fall back to an unbounded in-process build."""
    artifact = make_artifact(tmp_path)

    with pytest.raises(ConfigurationError) as raised:
        service(FakeRuntimeProbe(), tmp_path, smoke=FakeSmokeTester()).probe(
            artifact, device="npu", timeout=10
        )

    assert raised.value.diagnostic.code == "probe.timeout.unavailable"


def test_the_role_matrix_reads_an_inconclusive_device_as_inconclusive() -> None:
    deferred = ProbeResult(device="NPU", role=SubmodelRole.LANGUAGE, status=ProbeStatus.DEFERRED)
    reports = [
        DeviceProbeReport(device="NPU", status=ProbeStatus.INCONCLUSIVE, submodels=[deferred]),
        DeviceProbeReport(
            device="GPU",
            status=ProbeStatus.OK,
            submodels=[
                ProbeResult(device="GPU", role=SubmodelRole.LANGUAGE, status=ProbeStatus.OK)
            ],
        ),
    ]

    assert role_device_matrix(reports) == {"language": {"NPU": "inconclusive", "GPU": "ok"}}


def test_inconclusive_ranks_between_ok_and_failed_in_the_role_matrix() -> None:
    def result(role: SubmodelRole, status: ProbeStatus) -> ProbeResult:
        return ProbeResult(device="NPU", role=role, status=status)

    reports = [
        DeviceProbeReport(
            device="NPU",
            status=ProbeStatus.FAILED,
            submodels=[
                result(SubmodelRole.LANGUAGE, ProbeStatus.OK),
                result(SubmodelRole.LANGUAGE, ProbeStatus.INCONCLUSIVE),
                result(SubmodelRole.VISION_EMBEDDINGS, ProbeStatus.FAILED),
                result(SubmodelRole.VISION_EMBEDDINGS, ProbeStatus.INCONCLUSIVE),
            ],
        )
    ]

    assert role_device_matrix(reports) == {
        "language": {"NPU": "inconclusive"},
        "vision_embeddings": {"NPU": "failed"},
    }


def test_an_inconclusive_outcome_is_one_where_nothing_was_decided(tmp_path: Path) -> None:
    def outcome(*statuses: ProbeStatus) -> ProbeOutcome:
        names = ("NPU", "GPU", "CPU")
        return ProbeOutcome(
            artifact_dir=tmp_path,
            reports=[
                DeviceProbeReport(device=name, status=status)
                for name, status in zip(names, statuses, strict=False)
            ],
        )

    assert outcome(ProbeStatus.INCONCLUSIVE).inconclusive is True
    assert outcome(ProbeStatus.INCONCLUSIVE, ProbeStatus.SKIPPED).inconclusive is True
    assert outcome(ProbeStatus.INCONCLUSIVE, ProbeStatus.OK).inconclusive is False
    assert outcome(ProbeStatus.INCONCLUSIVE, ProbeStatus.FAILED).inconclusive is False
    assert outcome(ProbeStatus.FAILED).inconclusive is False


def test_a_deferred_raw_compile_is_reported_as_deferred_not_as_an_error(
    tmp_path: Path, job_ctx: tuple[JobContext, CollectingSink]
) -> None:
    """The progress line must not shout "failed" about a compile the pipeline decides."""
    artifact = make_artifact(tmp_path)
    write_manifest(artifact, repo_id="acme/model")
    runtime = FakeRuntimeProbe(compile_errors={"NPU": NPU_DYNAMIC_SHAPE_ERROR})
    ctx, sink = job_ctx

    service(runtime, tmp_path).probe(artifact, device="npu", quick=True, ctx=ctx)

    language = [event for event in sink.events if event.message == "language on NPU: deferred"]
    assert len(language) == 1
    assert language[0].status == "deferred"
    assert language[0].severity is Severity.INFO
    assert not any(event.message == "language on NPU: failed" for event in sink.events)


# --- a runtime that is not installed --------------------------------------------------


class MissingRuntimeProbe(FakeRuntimeProbe):
    """A runtime probe on a core-only install: ``openvino`` does not import.

    Mirrors :class:`~yonga.adapters.openvino_probe.OpenVINOProbe` there: every
    IR is unreadable with the import error as its text.
    """

    def available(self) -> tuple[bool, str | None]:
        return (False, "ModuleNotFoundError: No module named 'openvino'")

    def read_ir(self, path: Path) -> IRDescription:
        return IRDescription(
            path=path,
            role=role_for_filename(path.name),
            readable=False,
            read_error="ModuleNotFoundError: No module named 'openvino'",
        )


def test_a_missing_openvino_is_an_environment_failure_not_an_unrecognised_one(
    tmp_path: Path,
) -> None:
    """A core-only install is told to install the extra, with exit 10, as bench does."""
    artifact = make_artifact(tmp_path)
    write_manifest(artifact, repo_id="acme/model")
    runtime = MissingRuntimeProbe()

    with pytest.raises(YongaError) as excinfo:
        service(runtime, tmp_path).probe_detailed(artifact, device="cpu")

    found = excinfo.value.diagnostic
    assert found.code == "runtime.openvino.missing"
    assert found.category is FailureCategory.PACKAGE_ABI_MISMATCH
    assert excinfo.value.exit_code() is ExitCode.ENVIRONMENT_UNUSABLE
    assert any("pip install 'yonga[openvino]'" in step for step in found.remediation)
    assert "No module named 'openvino'" in found.message
    assert runtime.compiled == []
    assert runtime.tokenizer_calls == 0


def test_a_runtime_that_cannot_say_whether_it_imports_is_taken_at_its_word(
    tmp_path: Path,
) -> None:
    """The ``RuntimeProbe`` port does not require ``available()``: no check, no refusal."""
    artifact = make_artifact(tmp_path)

    reports = service(FakeRuntimeProbe(), tmp_path).probe(artifact, device="cpu")

    assert reports[0].status is ProbeStatus.OK


def test_every_probe_result_and_its_raw_text_goes_to_the_job_log(tmp_path: Path) -> None:
    """The terminal summarises a compiler dump; the job log keeps all of it."""
    artifact = make_artifact(tmp_path)
    write_manifest(artifact, repo_id="acme/model")
    runtime = FakeRuntimeProbe(compile_errors={"NPU": NPU_DYNAMIC_SHAPE_ERROR})
    settings = make_settings(tmp_path)
    log_path = tmp_path / "job.jsonl"
    logger = JobLogger("job-log", log_path)
    ctx = JobContext(job_id="job-log", logger=logger, paths=settings.paths, settings=settings)
    try:
        service(runtime, tmp_path).probe(artifact, device="npu", quick=True, ctx=ctx)
    finally:
        logger.close()

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    results = [record for record in records if record["event"] == "probe.result"]
    language = next(record for record in results if record["role"] == "language")
    assert language["device"] == "NPU"
    assert language["status"] == "deferred"
    assert language["code"] == "npu.compiler.dynamic_shape"
    assert "to_shape was called on a dynamic shape" in language["raw"]
    assert any(record["role"] == "tokenizer" for record in results)


def test_the_manual_entry_point_reports_a_refusal_instead_of_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No such artifact (or no OpenVINO) is a typed refusal with its own exit code."""
    import yonga.config.settings as settings_module
    from yonga.application.probe import main

    monkeypatch.setattr(settings_module, "load_settings", lambda: make_settings(tmp_path))

    assert main([str(tmp_path / "missing")]) == int(ExitCode.REGISTRY)
    captured = capsys.readouterr()
    assert "artifact.not_found" in captured.err
    assert captured.out == ""
