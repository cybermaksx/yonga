"""Device policy, validation, repair and probing behaviour of ``PrepareService``.

The rules under test are the ones ``SPEC.md`` section 7 and ``AGENT_PROMPT.md``
section 19 spell out and that are easy to get subtly wrong:

* an explicitly requested device that fails stops the job — no silent switch;
* ``--device auto`` falls back only when the classifier says the environment is
  healthy, so a permission problem is diagnosed instead of being hidden;
* a validation failure is never final until registry repair detection has run,
  and ``--no-repair`` turns that detection into a typed failure rather than a
  silent skip;
* file digests are computed *after* repairs, so the manifest describes the bytes
  that were actually published.

The failure strings are the real ones captured on the reference host and recorded
in ``HOST_FACTS.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.test_prepare_planning import (
    BAD_PRECISION,
    MISMATCHED_EXPORT_VERSIONS,
    SAFETENSORS_REPO,
    FakeExporter,
    FakeFilesystemProbe,
    FakeHub,
    FakeRuntimeProbe,
    FakeSmokeTester,
    build_service,
    builtin_resolver,
    make_snapshot,
    reference_inventory,
    run_prepare,
    safetensors_metadata,
    settings_for,
)
from yonga.application.progress import CollectingSink, ProgressKind
from yonga.config.settings import Settings
from yonga.domain.artifacts import ProbeResult, ProbeStatus
from yonga.domain.diagnostics import (
    DeviceUnsupportedError,
    ExitCode,
    ExportError,
    FailureCategory,
    PreflightError,
    RepairError,
    ValidationError,
    diagnostic,
)
from yonga.domain.plans import CONFIRMATION_DEVICE_FALLBACK, ConfirmationReason
from yonga.util.hashing import sha256_file

# --- real failure signatures captured on the reference host --------------------------

#: The documented Qwen3.5 NPU compiler refusal (``SPEC.md`` section 20).
NPU_UNSUPPORTED_GRAPH = (
    "Exception from src/plugins/intel_npu/src/plugin/src/plugin.cpp:552:\n"
    "OneShotBufferizeVPU2VPUIP failed : Only supports FP16."
)

#: A genuinely broken environment: the kernel refused the accelerator node.
NPU_PERMISSION_DENIED = (
    "Exception from src/inference/src/cpp/core.cpp:117:\n"
    "Failed to open /dev/accel/accel0: Permission denied"
)

#: What a full filesystem looks like coming out of the exporter.
DISK_FULL = "RuntimeError: Failed to save model: No space left on device"

#: How the Qwen3.5 export really dies on ``transformers 5.5.4``: optimum-intel
#: 2.2.0's model patcher imports a symbol that release no longer exposes. The
#: architecture looks supported right up to this line — 5.5.4 loads
#: ``Qwen3_5Config`` and reports ``model_type: qwen3_5`` — which is exactly why
#: only the declared version pin can predict the failure.
QWEN35_CACHE_IMPORT_ERROR = (
    'File ".../optimum/exporters/openvino/model_patcher.py", line 10049, in __init__\n'
    "    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache\n"
    "ImportError: cannot import name 'Qwen3_5DynamicCache' from "
    "'transformers.models.qwen3_5.modeling_qwen3_5'"
)

#: A registry document that makes the NPU the recommended device for one exact
#: repository, so auto-selection tries it first and the fallback path is reached.
#: A ``repo_ids`` match scores higher specificity than the built-in family rule,
#: so the override is deterministic rather than a priority race.
NPU_FIRST_REGISTRY = f"""
schema_version: 1
architectures:
  - id: arch-test-npu-first
    schema_version: 1
    status: active
    priority: 100
    summary: Test rule that believes the NPU can run this repository
    match:
      repo_ids:
        - {SAFETENSORS_REPO}
    export:
      pipeline: vlm
      trust_remote_code: false
    devices:
      cpu:
        status: supported
        evidence: official
      gpu:
        status: supported
        evidence: official
      npu:
        status: supported
        evidence: tested
    recommended_device: npu
    profiles:
      npu: npu-int4-sym-cw
      gpu: gpu-int4-asym-g128
      cpu: cpu-int4-asym-g128
    chat_profile: chat-qwen3_5
"""


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings confined to the test's temporary directory."""
    return settings_for(tmp_path)


@pytest.fixture
def hub(tmp_path: Path) -> FakeHub:
    """A hub carrying only the safetensors acceptance fixture."""
    return FakeHub(
        repos={SAFETENSORS_REPO: safetensors_metadata()},
        snapshots={SAFETENSORS_REPO: make_snapshot(tmp_path, "snapshot-safetensors")},
    )


# --- explicit device: never switch ---------------------------------------------------


def test_explicit_npu_failure_never_falls_back(settings: Settings, hub: FakeHub) -> None:
    """``--device npu`` that fails stops with the diagnosis and a suggested command."""
    runtime = FakeRuntimeProbe(compile_failures={"NPU": NPU_UNSUPPORTED_GRAPH})
    service = build_service(settings=settings, hub=hub, runtime=runtime)
    plan = service.plan(SAFETENSORS_REPO, device="npu")
    assert plan.compatibility.explicit_request is True
    assert plan.target_device == "NPU"

    with pytest.raises(DeviceUnsupportedError) as caught:
        run_prepare(service, plan, settings)

    failure = caught.value.diagnostic
    assert failure.code == "npu.compiler.unsupported_graph.qwen35"
    assert failure.environment_broken is False
    assert "will not silently prepare for another device" in failure.message
    assert any("--device gpu" in line for line in failure.remediation)
    # Nothing was ever compiled on another device.
    assert {device for _name, device in runtime.compiled} == {"NPU"}
    assert not plan.artifact_dir.exists()


def test_explicit_device_absent_from_openvino_stops_at_preflight(
    settings: Settings, hub: FakeHub
) -> None:
    """A requested device OpenVINO does not expose fails before any download."""
    from tests.unit.test_prepare_planning import FakeHardwareProbe

    hardware = FakeHardwareProbe(reference_inventory(devices=("CPU", "GPU")))
    service = build_service(settings=settings, hub=hub, hardware=hardware)
    plan = service.plan(SAFETENSORS_REPO, device="npu")

    with pytest.raises(DeviceUnsupportedError) as caught:
        run_prepare(service, plan, settings)

    assert caught.value.diagnostic.code == "device.not_present"
    assert caught.value.exit_code() is ExitCode.DEVICE_UNSUPPORTED
    assert hub.downloaded == []


# --- auto device: fall back, but only for a healthy environment ----------------------


def test_auto_falls_back_on_an_unsupported_graph(settings: Settings, hub: FakeHub) -> None:
    """A classified compatibility limit with a healthy environment permits GPU."""
    runtime = FakeRuntimeProbe(compile_failures={"NPU": NPU_UNSUPPORTED_GRAPH})
    smoke = FakeSmokeTester()
    service = build_service(
        settings=settings,
        hub=hub,
        runtime=runtime,
        smoke=smoke,
        resolver=builtin_resolver([NPU_FIRST_REGISTRY]),
    )
    plan = service.plan(SAFETENSORS_REPO, device="auto")
    assert plan.target_device == "NPU"
    assert plan.compatibility.explicit_request is False
    sink = CollectingSink()

    outcome = run_prepare(service, plan, settings, sink=sink)

    assert outcome.device == "GPU"
    assert outcome.fallbacks == ["GPU"]
    probes = outcome.manifest.device_probes
    assert probes["NPU"].status is ProbeStatus.FAILED
    assert probes["NPU"].diagnostic is not None
    assert probes["NPU"].diagnostic.code == "npu.compiler.unsupported_graph.qwen35"
    assert probes["NPU"].diagnostic.environment_broken is False
    assert probes["GPU"].status is ProbeStatus.OK
    assert outcome.manifest.supported_devices == ["GPU"]
    # The fallback is explained in the UI stream, not just buried in the manifest.
    fallback_events = [
        event for event in sink.of_kind(ProgressKind.PROBE) if event.status == "fallback"
    ]
    assert fallback_events
    assert "not a broken NPU" in fallback_events[0].message
    assert any("Fell back from NPU to GPU." in note for note in outcome.manifest.notes)
    assert smoke.calls[0][0] == "GPU"


def test_auto_stops_on_a_permission_denied_environment(settings: Settings, hub: FakeHub) -> None:
    """A broken environment is diagnosed, never hidden behind a working GPU."""
    runtime = FakeRuntimeProbe(compile_failures={"NPU": NPU_PERMISSION_DENIED})
    service = build_service(
        settings=settings,
        hub=hub,
        runtime=runtime,
        resolver=builtin_resolver([NPU_FIRST_REGISTRY]),
    )
    plan = service.plan(SAFETENSORS_REPO, device="auto")
    assert plan.target_device == "NPU"

    with pytest.raises(DeviceUnsupportedError) as caught:
        run_prepare(service, plan, settings)

    failure = caught.value.diagnostic
    assert failure.code == "npu.device.permission_denied"
    assert failure.environment_broken is True
    assert any("did not fall back" in line for line in failure.remediation)
    assert {device for _name, device in runtime.compiled} == {"NPU"}
    assert not plan.artifact_dir.exists()


def test_declining_the_fallback_confirmation_stops_the_job(
    settings: Settings, hub: FakeHub
) -> None:
    """Auto mode asks at the moment it wants to switch, and a no is honoured."""
    runtime = FakeRuntimeProbe(compile_failures={"NPU": NPU_UNSUPPORTED_GRAPH})
    service = build_service(
        settings=settings,
        hub=hub,
        runtime=runtime,
        resolver=builtin_resolver([NPU_FIRST_REGISTRY]),
    )
    plan = service.plan(SAFETENSORS_REPO, device="auto")
    # Nothing speculative is asked up front; the question only arrives when the
    # NPU probe has actually failed.
    assert CONFIRMATION_DEVICE_FALLBACK not in {r.code for r in plan.confirmations}
    asked: list[str] = []

    def refuse_fallback(reason: ConfirmationReason) -> bool:
        asked.append(reason.code)
        return reason.code != CONFIRMATION_DEVICE_FALLBACK

    with pytest.raises(DeviceUnsupportedError):
        run_prepare(service, plan, settings, confirm=refuse_fallback)

    assert CONFIRMATION_DEVICE_FALLBACK in asked
    assert {device for _name, device in runtime.compiled} == {"NPU"}


def test_smoke_test_failure_is_treated_as_a_device_failure(
    settings: Settings, hub: FakeHub
) -> None:
    """Compiling is not enough: generation has to work before a device counts."""
    smoke = FakeSmokeTester(failures={"NPU": "RuntimeError: generation produced no tokens"})
    service = build_service(
        settings=settings,
        hub=hub,
        smoke=smoke,
        resolver=builtin_resolver([NPU_FIRST_REGISTRY]),
    )
    plan = service.plan(SAFETENSORS_REPO, device="auto")

    outcome = run_prepare(service, plan, settings)

    assert outcome.device == "GPU"
    assert outcome.manifest.device_probes["NPU"].status is ProbeStatus.FAILED
    assert outcome.manifest.device_probes["NPU"].smoke_test is not None
    assert outcome.manifest.device_probes["GPU"].smoke_test is not None
    assert outcome.manifest.device_probes["GPU"].smoke_test.ok is True


# --- probe hygiene -------------------------------------------------------------------


def test_tokenizers_are_never_compiled_on_an_accelerator(settings: Settings, hub: FakeHub) -> None:
    """Tokenizer IRs stay on the CPU; only numeric submodels reach the GPU."""
    runtime = FakeRuntimeProbe()
    service = build_service(settings=settings, hub=hub, runtime=runtime)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    outcome = run_prepare(service, plan, settings)

    tokenizer_names = {"openvino_tokenizer.xml", "openvino_detokenizer.xml"}
    assert not [name for name, _device in runtime.compiled if name in tokenizer_names]
    assert {device for _name, device in runtime.compiled} == {"GPU"}
    skipped = [
        result
        for result in outcome.manifest.device_probes["GPU"].submodels
        if result.status is ProbeStatus.SKIPPED
    ]
    assert {result.device for result in skipped} == {"CPU"}
    assert {result.role.value for result in skipped if result.role} == {
        "tokenizer",
        "detokenizer",
    }


def test_genai_only_properties_never_reach_a_compile_probe(
    settings: Settings, hub: FakeHub
) -> None:
    """``ATTENTION_BACKEND`` is a pipeline property; compiling with it would fail."""
    runtime = FakeRuntimeProbe()
    smoke = FakeSmokeTester()
    service = build_service(settings=settings, hub=hub, runtime=runtime, smoke=smoke)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    assert plan.profile.runtime_properties == {"ATTENTION_BACKEND": "SDPA"}

    outcome = run_prepare(service, plan, settings)

    compiled = outcome.manifest.device_probes["GPU"].submodels
    for result in compiled:
        assert "ATTENTION_BACKEND" not in result.properties
    # ...but the smoke test, which builds a real pipeline, does receive it.
    assert smoke.calls[0][1] == {"ATTENTION_BACKEND": "SDPA"}
    assert outcome.manifest.recommended_runtime is not None
    assert outcome.manifest.recommended_runtime.properties == {"ATTENTION_BACKEND": "SDPA"}
    assert 16 <= smoke.calls[0][2] <= 32


# --- validation and repair -----------------------------------------------------------


def test_repairs_run_before_validation_failure_is_declared(
    settings: Settings, hub: FakeHub
) -> None:
    """The STRiNG defect is repaired in staging and recorded, not reported as failure."""
    runtime = FakeRuntimeProbe()
    service = build_service(settings=settings, hub=hub, runtime=runtime)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    sink = CollectingSink()

    outcome = run_prepare(service, plan, settings, sink=sink)

    assert [record.repair_id for record in outcome.repairs] == [
        "ov-tokenizer-precision-string-casing"
    ]
    record = outcome.repairs[0]
    assert record.applied is True
    assert record.rule_id == "ov-tokenizer-precision-string-casing"
    assert record.safety == "automatic"
    assert sorted(record.files) == ["openvino_detokenizer.xml", "openvino_tokenizer.xml"]
    assert record.before_hashes != record.after_hashes
    # The manifest keeps the whole repair history.
    assert outcome.manifest.repairs == outcome.repairs
    # Validation was attempted, failed, and then succeeded after the repair.
    assert runtime.tokenizer_reads == 2
    assert outcome.validation is not None
    assert outcome.validation.ok is True
    assert outcome.validation.tokenizer_read_ok is True
    assert outcome.validation.tokenizer_compile_ok is True
    assert sink.of_kind(ProgressKind.REPAIR)
    # And the published files really no longer carry the defect.
    published = outcome.artifact_dir / "openvino_tokenizer.xml"
    assert BAD_PRECISION not in published.read_text(encoding="utf-8")


def test_file_hashes_are_computed_after_repairs(settings: Settings, hub: FakeHub) -> None:
    """A digest recorded before the repair would describe bytes nobody ever gets."""
    service = build_service(settings=settings, hub=hub)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    outcome = run_prepare(service, plan, settings)

    record = outcome.repairs[0]
    manifest_files = {entry.path: entry.sha256 for entry in outcome.manifest.files}
    for name in record.files:
        published = outcome.artifact_dir / name
        assert manifest_files[name] == sha256_file(published)
        assert manifest_files[name] == record.after_hashes[name]
        assert manifest_files[name] != record.before_hashes[name]
    # The manifest on disk says the same thing.
    written = json.loads(
        (outcome.artifact_dir / ".yonga" / "manifest.json").read_text(encoding="utf-8")
    )
    on_disk = {entry["path"]: entry["sha256"] for entry in written["files"]}
    assert on_disk["openvino_tokenizer.xml"] == manifest_files["openvino_tokenizer.xml"]


def test_no_repair_turns_detection_into_a_typed_failure(settings: Settings, hub: FakeHub) -> None:
    """``--no-repair`` must fail loudly with exit code 42, never skip silently."""
    service = build_service(settings=settings, hub=hub)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    with pytest.raises(RepairError) as caught:
        run_prepare(service, plan, settings, allow_repairs=False)

    assert caught.value.exit_code() is ExitCode.REPAIR_DENIED
    assert caught.value.diagnostic.code == "repair.declined"
    assert "ov-tokenizer-precision-string-casing" in caught.value.diagnostic.message
    assert not plan.artifact_dir.exists()


def test_repairs_never_touch_the_downloaded_source(settings: Settings, hub: FakeHub) -> None:
    """The Hub snapshot standing in for the shared cache is left exactly as it was."""
    snapshot = hub.snapshots[SAFETENSORS_REPO]
    (snapshot / "openvino_tokenizer.xml").write_text(
        f"<net><port {BAD_PRECISION}/></net>\n", encoding="utf-8"
    )
    before = sha256_file(snapshot / "openvino_tokenizer.xml")
    service = build_service(settings=settings, hub=hub)

    run_prepare(service, service.plan(SAFETENSORS_REPO, device="gpu"), settings)

    assert sha256_file(snapshot / "openvino_tokenizer.xml") == before


def test_validation_failure_without_a_matching_repair_is_final(
    settings: Settings, hub: FakeHub
) -> None:
    """When nothing in the registry matches, validation failure is reported as such."""
    exporter = FakeExporter(write_tokenizer=False, numeric_models=())
    service = build_service(settings=settings, hub=hub, exporter=exporter)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    with pytest.raises(ValidationError) as caught:
        run_prepare(service, plan, settings)

    assert caught.value.exit_code() in {ExitCode.VALIDATION, ExitCode.INTERNAL}
    assert not plan.artifact_dir.exists()


# --- export failure ------------------------------------------------------------------


def test_export_failure_is_classified_and_kept_raw(settings: Settings, hub: FakeHub) -> None:
    """A full filesystem is diagnosed as such, and nothing is published."""
    exporter = FakeExporter(succeed=False, stderr_tail=[DISK_FULL])
    service = build_service(settings=settings, hub=hub, exporter=exporter)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    with pytest.raises(ExportError) as caught:
        run_prepare(service, plan, settings)

    failure = caught.value.diagnostic
    assert failure.code == "storage.insufficient_disk"
    assert failure.raw_excerpt is not None
    assert "No space left on device" in failure.raw_excerpt
    assert not plan.artifact_dir.exists()
    # The raw text also reached the job log.
    log_text = settings.paths.job_log_path(plan.job_id).read_text(encoding="utf-8")
    assert "No space left on device" in log_text


def test_an_unmet_exporter_pin_stops_the_job_before_the_export_can_fail(
    settings: Settings, hub: FakeHub
) -> None:
    """The export that cannot succeed is never launched.

    This is the observed defect in miniature: the exporter double here reports
    the reference host's real ``transformers 5.5.4`` *and* would fail with the
    authentic ``Qwen3_5DynamicCache`` import error. Preflight refuses first, so
    the user gets the package diagnostic instead of a generic conversion failure
    minutes and 17.5 GiB later.
    """
    exporter = FakeExporter(
        versions=dict(MISMATCHED_EXPORT_VERSIONS),
        succeed=False,
        stderr_tail=[QWEN35_CACHE_IMPORT_ERROR],
    )
    service = build_service(settings=settings, hub=hub, exporter=exporter)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    with pytest.raises(PreflightError) as caught:
        run_prepare(service, plan, settings)

    failure = caught.value.diagnostic
    assert failure.category is FailureCategory.PACKAGE_ABI_MISMATCH
    assert caught.value.exit_code() is ExitCode.ENVIRONMENT_UNUSABLE
    assert exporter.requests == []
    assert hub.downloaded == []
    assert not plan.artifact_dir.exists()


def test_export_argv_is_recorded_in_the_manifest(settings: Settings, hub: FakeHub) -> None:
    """Reproducibility: the exact command line lands in the manifest, redacted."""
    service = build_service(settings=settings, hub=hub)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    outcome = run_prepare(service, plan, settings)

    export = outcome.manifest.export
    assert export.backend == "optimum-intel"
    assert export.profile == "gpu-int4-asym-g128"
    assert export.argv[:5] == ["python", "-m", "optimum.commands.optimum_cli", "export", "openvino"]
    assert export.argv[-1] == str(plan.staging_dir)
    assert export.arguments["weight_format"] == "int4"
    assert export.arguments["group_size"] == 128
    assert export.arguments["symmetric"] is False
    assert export.package_versions["transformers"] == "5.2.1"
    assert export.duration_seconds is not None


# --- disk plan sanity ----------------------------------------------------------------


def test_cramped_artifact_filesystem_is_reported_separately(
    settings: Settings, hub: FakeHub
) -> None:
    """A temp filesystem with room but no room for the artifact still fails preflight."""
    probe = FakeFilesystemProbe(
        small_roots=(settings.paths.data_dir,),
        small_free=8 * (1024**3),
    )
    service = build_service(settings=settings, hub=hub, filesystem=probe)

    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    assert plan.disk.temp.sufficient is True
    assert plan.disk.artifact_sufficient is False
    assert plan.disk.sufficient is False


# --- a raw NPU compile is not a verdict: the GenAI pipeline decides ------------------
#
# On the reference host a raw ``Core.compile_model`` of Qwen3-0.6B (INT4 sym, g128)
# fails on the NPU with the dynamic-shape message below, while ``LLMPipeline`` on
# the same NPU reshapes and generates. Before deferral existed, the raw failure
# ended the probe, so yonga could not prepare an NPU artifact for any LLM at all.

#: The raw-compile rejection, verbatim from the reference host.
NPU_DYNAMIC_SHAPE = (
    "Exception from src/plugins/intel_npu/src/compiler_adapter/src/ze_graph_ext_wrappers.cpp:425:\n"
    "Compilation failed. Level0 pfnCreate2 result: ZE_RESULT_ERROR_INVALID_ARGUMENT\n"
    "[NPU_VCL] Compiler returned msg:\n"
    "Exception from src/core/src/partial_shape.cpp:261:\n"
    "to_shape was called on a dynamic shape."
)


class SkippingSmokeTester(FakeSmokeTester):
    """A smoke tester that never builds a pipeline -- no evidence either way."""

    def smoke_test(self, artifact_dir, *, device, pipeline, properties, prompt, max_new_tokens):
        self.calls.append((device, dict(properties), max_new_tokens))
        return ProbeResult(device=device, status=ProbeStatus.SKIPPED, message="not run")


class GenericFailureSmokeTester(FakeSmokeTester):
    """Fails the way the real adapter does: a generic diagnostic around raw text."""

    raw: str = ""

    def smoke_test(self, artifact_dir, *, device, pipeline, properties, prompt, max_new_tokens):
        self.calls.append((device, dict(properties), max_new_tokens))
        generic = diagnostic(
            "runtime.genai.load_failed",
            FailureCategory.UNKNOWN_RUNTIME_ERROR,
            f"Could not load the model on {device}",
            "Compilation failed",
            raw_excerpt=self.raw,
        )
        return ProbeResult(
            device=device,
            status=ProbeStatus.FAILED,
            diagnostic=generic,
            message=generic.message,
        )


def test_a_deferred_npu_compile_is_settled_by_the_pipeline(
    settings: Settings, hub: FakeHub
) -> None:
    """The dynamic-shape raw failure is overruled when the pipeline generates."""
    runtime = FakeRuntimeProbe(compile_failures={"NPU": NPU_DYNAMIC_SHAPE})
    smoke = FakeSmokeTester()
    service = build_service(settings=settings, hub=hub, runtime=runtime, smoke=smoke)
    plan = service.plan(SAFETENSORS_REPO, device="npu")

    outcome = run_prepare(service, plan, settings)

    report = outcome.manifest.device_probes["NPU"]
    assert report.status is ProbeStatus.OK
    assert report.smoke_test is not None and report.smoke_test.status is ProbeStatus.OK
    raw = [result for result in report.submodels if result.status is ProbeStatus.DEFERRED]
    assert raw, "the raw compile attempt must stay on record"
    assert "to_shape was called on a dynamic shape" in (raw[0].message or "")
    # The NPU profile's static-shape budget reached the pipeline.
    npu_calls = [properties for device, properties, _ in smoke.calls if device == "NPU"]
    assert npu_calls and npu_calls[0]["MAX_PROMPT_LEN"] == "1024"
    assert npu_calls[0]["MIN_RESPONSE_LEN"] == "128"


def test_a_deferred_npu_compile_without_pipeline_evidence_is_not_a_pass(
    settings: Settings, hub: FakeHub
) -> None:
    """Deferral never passes a device on its own; no pipeline evidence means failed."""
    runtime = FakeRuntimeProbe(compile_failures={"NPU": NPU_DYNAMIC_SHAPE})
    service = build_service(
        settings=settings, hub=hub, runtime=runtime, smoke=SkippingSmokeTester()
    )
    plan = service.plan(SAFETENSORS_REPO, device="npu")

    with pytest.raises(DeviceUnsupportedError) as caught:
        run_prepare(service, plan, settings)

    assert caught.value.diagnostic.code == "npu.compiler.dynamic_shape"
    assert "unproven" in caught.value.diagnostic.message
    assert not plan.artifact_dir.exists()


def test_a_pipeline_failure_is_classified_by_the_registry_not_the_adapter(
    settings: Settings, hub: FakeHub
) -> None:
    """A generic adapter diagnostic must not hide the compiler signature it carries.

    The real adapter reports every pipeline load failure as
    ``runtime.genai.load_failed``; the NPU compiler's reason arrives only in the
    raw text (captured from the native descriptors). The registry must see it.
    """
    runtime = FakeRuntimeProbe(compile_failures={"NPU": NPU_DYNAMIC_SHAPE})
    smoke = GenericFailureSmokeTester()
    smoke.raw = (
        "Compilation failed\n--- native output ---\n"
        "[ERROR] [vpux-compiler] OneShotBufferizeVPU2VPUIP failed : Only supports FP16."
    )
    service = build_service(settings=settings, hub=hub, runtime=runtime, smoke=smoke)
    plan = service.plan(SAFETENSORS_REPO, device="npu")

    with pytest.raises(DeviceUnsupportedError) as caught:
        run_prepare(service, plan, settings)

    assert caught.value.diagnostic.code == "npu.compiler.unsupported_graph.qwen35"
    assert caught.value.diagnostic.environment_broken is False


def test_a_conclusive_npu_compile_failure_still_stops_before_the_pipeline(
    settings: Settings, hub: FakeHub
) -> None:
    """Only a failure the registry marks deferrable is handed to the pipeline."""
    runtime = FakeRuntimeProbe(compile_failures={"NPU": NPU_UNSUPPORTED_GRAPH})
    smoke = FakeSmokeTester()
    service = build_service(settings=settings, hub=hub, runtime=runtime, smoke=smoke)
    plan = service.plan(SAFETENSORS_REPO, device="npu")

    with pytest.raises(DeviceUnsupportedError):
        run_prepare(service, plan, settings)

    assert [device for device, _, _ in smoke.calls if device == "NPU"] == []


def test_the_signature_is_found_even_after_a_long_traceback(
    settings: Settings, hub: FakeHub
) -> None:
    """A capped, head-first excerpt must not truncate away the decisive native line.

    The NPU compiler's reason is captured native output appended *after* the
    exception text, so a long traceback plus plugin noise would push it past the
    excerpt's cap. Classification reads the full message instead.
    """
    long_text = (
        "Compilation failed\n"
        + "Exception from src/inference/src/dev/plugin.cpp:54:\n" * 200
        + "--- native output ---\n"
        + "[INFO] plugin chatter\n" * 200
        + "[ERROR] [vpux-compiler] OneShotBufferizeVPU2VPUIP failed : Only supports FP16."
    )

    class LongFailureSmokeTester(FakeSmokeTester):
        def smoke_test(self, artifact_dir, *, device, pipeline, properties, prompt, max_new_tokens):
            self.calls.append((device, dict(properties), max_new_tokens))
            generic = diagnostic(
                "runtime.genai.load_failed",
                FailureCategory.UNKNOWN_RUNTIME_ERROR,
                f"Could not load the model on {device}",
                long_text,
                raw_excerpt=long_text,
            )
            assert "OneShotBufferize" not in (generic.raw_excerpt or ""), "precondition"
            return ProbeResult(
                device=device, status=ProbeStatus.FAILED, diagnostic=generic, message=long_text
            )

    runtime = FakeRuntimeProbe(compile_failures={"NPU": NPU_DYNAMIC_SHAPE})
    service = build_service(
        settings=settings, hub=hub, runtime=runtime, smoke=LongFailureSmokeTester()
    )
    plan = service.plan(SAFETENSORS_REPO, device="npu")

    with pytest.raises(DeviceUnsupportedError) as caught:
        run_prepare(service, plan, settings)

    assert caught.value.diagnostic.code == "npu.compiler.unsupported_graph.qwen35"
