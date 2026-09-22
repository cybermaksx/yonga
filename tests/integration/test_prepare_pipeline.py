"""End-to-end preparation against the real subprocess and process boundaries.

Unlike the unit tests, this module wires the *real*
:class:`~yonga.adapters.subprocess.SubprocessRunner` and
:class:`~yonga.adapters.exporter_optimum.OptimumExporter` to
``tests/fixtures/fake_optimum_cli.py``. Nothing about the export path is
simulated: an actual child process is spawned with an argv list and
``shell=False``, it reads the environment yonga gave it, and it writes the files
back. That is what makes the ``TMPDIR`` redirection, the argv recording, the
redaction and the atomic publication real assertions rather than assertions about
a stub.

The module-level ``if __name__`` guard requirement from ``AGENT_PROMPT.md``
section 16.4 applies to the fixture script, which carries one.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from tests.unit.test_prepare_planning import (
    FakeHub,
    FakeRuntimeProbe,
    FakeSmokeTester,
    approve_all,
    build_service,
    builtin_resolver,
    make_snapshot,
    reference_inventory,
    run_prepare,
    safetensors_metadata,
    settings_for,
    write_tokenizer_pair,
)
from yonga.adapters.exporter_optimum import OptimumExporter
from yonga.adapters.subprocess import SubprocessRunner
from yonga.application.artifacts import ArtifactStore
from yonga.application.jobs import JobContext
from yonga.application.prepare import PrepareService
from yonga.application.progress import CollectingSink, ProgressKind
from yonga.config.settings import Settings
from yonga.domain.artifacts import ProbeResult, ProbeStatus
from yonga.domain.benchmarks import GenerationSettings
from yonga.domain.diagnostics import ExportError
from yonga.domain.models import (
    FormatObservation,
    ModelFormat,
    ModelIdentity,
    ModelMetadata,
    PipelineKind,
    RepoFile,
)
from yonga.domain.plans import ConversionProfile, ExportRequest, ExportResult
from yonga.util.hashing import sha256_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FAKE_CLI = FIXTURES / "fake_optimum_cli.py"

REPO_ID = "acme/tiny-llm"
SHA = "1f2e3d4c5b6a79808172635445362718091a2b3c"

#: A syntactically valid, entirely fictitious token, matching the fixture script.
FAKE_TOKEN = "hf_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7"

#: Two registry profiles whose ``export_extra_args`` steer the fixture exporter.
#: Driving the fake through registry data (rather than by mutating the plan) keeps
#: the test on the same path production code takes.
TEST_REGISTRY = f"""
schema_version: 1
profiles:
  - id: test-export-ok
    schema_version: 1
    status: active
    priority: 100
    summary: Fixture profile whose export succeeds and prints a token
    device: gpu
    quantization:
      weight_format: int4
      symmetric: false
      ratio: 1.0
      group_size: 128
    runtime:
      ATTENTION_BACKEND: SDPA
    export_extra_args:
      - "--emit-secret"
  - id: test-export-disk-full
    schema_version: 1
    status: active
    priority: 100
    summary: Fixture profile whose export fails with a full filesystem
    device: cpu
    quantization:
      weight_format: int4
      symmetric: true
      ratio: 1.0
      group_size: -1
    export_extra_args:
      - "--fail-with"
      - "1"
      - "--stderr-line"
      - "RuntimeError: Failed to save model: No space left on device"
architectures:
  - id: arch-test-tiny-llm
    schema_version: 1
    status: active
    priority: 100
    summary: Fixture architecture for the integration pipeline
    match:
      repo_ids:
        - {REPO_ID}
    export:
      pipeline: llm
      task: text-generation-with-past
      trust_remote_code: false
    devices:
      cpu:
        status: supported
        evidence: official
      gpu:
        status: supported
        evidence: official
    recommended_device: gpu
    profiles:
      gpu: test-export-ok
      cpu: test-export-disk-full
    chat_profile: chat-default
"""


def llm_metadata() -> ModelMetadata:
    """Metadata for a small decoder-only repository (one language model, no vision)."""
    return safetensors_metadata(
        REPO_ID,
        sha=SHA,
        weight_bytes=2_000_000_000,
        total_bytes=2_100_000_000,
        model_type="llama",
        pipeline_tag="text-generation",
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings confined to the test's temporary directory."""
    return settings_for(tmp_path)


@pytest.fixture
def hub(tmp_path: Path) -> FakeHub:
    """A hub serving the integration fixture repository."""
    return FakeHub(
        repos={REPO_ID: llm_metadata()},
        snapshots={REPO_ID: make_snapshot(tmp_path, "snapshot")},
    )


def real_exporter() -> OptimumExporter:
    """The production exporter, pointed at the fixture CLI instead of Optimum."""
    return OptimumExporter(
        runner=SubprocessRunner(extra_secrets=(FAKE_TOKEN,)),
        launcher=[sys.executable, str(FAKE_CLI)],
        timeout=120.0,
    )


def integration_service(
    settings: Settings,
    hub: FakeHub,
    *,
    runtime: FakeRuntimeProbe | None = None,
    smoke: FakeSmokeTester | None = None,
) -> PrepareService:
    """Build a service whose export path is a real child process."""
    return build_service(
        settings=settings,
        hub=hub,
        exporter=real_exporter(),
        runtime=runtime if runtime is not None else FakeRuntimeProbe(),
        smoke=smoke if smoke is not None else FakeSmokeTester(),
        resolver=builtin_resolver([TEST_REGISTRY]),
    )


def test_full_pipeline_through_a_real_subprocess(settings: Settings, hub: FakeHub) -> None:
    """Plan, preflight, export, validate, probe and publish, end to end."""
    service = integration_service(settings, hub)
    plan = service.plan(REPO_ID, device="gpu")
    assert plan.profile.id == "test-export-ok"
    assert plan.task == "text-generation-with-past"
    sink = CollectingSink()

    outcome = run_prepare(service, plan, settings, sink=sink)

    published = outcome.artifact_dir
    assert published == plan.artifact_dir
    assert published.is_dir()
    assert not plan.staging_dir.exists()
    assert not settings.paths.job_tmp_dir(plan.job_id).exists()

    # --- the child process really ran, with the argv and environment we intended
    child_argv = json.loads((published / "argv.json").read_text(encoding="utf-8"))
    assert child_argv[0] == "export"
    assert child_argv[1] == "openvino"
    assert "--model" in child_argv
    assert child_argv[child_argv.index("--model") + 1] == str(hub.snapshots[REPO_ID])
    assert child_argv[-1] == str(plan.staging_dir)
    assert child_argv[child_argv.index("--weight-format") + 1] == "int4"
    # The equals form is used so a leading-dash value can never look like a flag.
    assert "--group-size=128" in child_argv

    managed_temp = str(settings.paths.job_tmp_dir(plan.job_id))
    child_env = json.loads((published / "env.json").read_text(encoding="utf-8"))
    assert child_env["TMPDIR"] == managed_temp
    assert child_env["TMP"] == managed_temp
    assert child_env["TEMP"] == managed_temp

    # --- the manifest records what happened, redacted
    manifest = outcome.manifest
    assert manifest.schema_version == 1
    assert manifest.export.backend == "optimum-intel"
    assert manifest.export.profile == "test-export-ok"
    # The manifest keeps the full command, launcher included; the child only ever
    # sees the part after the launcher.
    assert manifest.export.argv[-len(child_argv) :] == child_argv
    assert manifest.export.argv[0] == sys.executable
    assert manifest.export.argv[-1] == str(plan.staging_dir)
    assert "--emit-secret" in manifest.export.argv
    assert manifest.requested_model.repo_id == REPO_ID
    assert manifest.resolved_source.resolved_revision == SHA
    assert manifest.recommended_runtime is not None
    assert manifest.recommended_runtime.device == "GPU"
    assert manifest.recommended_runtime.properties == {"ATTENTION_BACKEND": "SDPA"}
    assert manifest.device_probes["GPU"].status is ProbeStatus.OK
    assert manifest.device_probes["GPU"].smoke_test is not None
    assert manifest.host["devices"] == "CPU,GPU,NPU"

    # --- every published file is inventoried with a digest that matches disk
    recorded = {entry.path: entry.sha256 for entry in manifest.files}
    assert "openvino_model.xml" in recorded
    for name, digest in recorded.items():
        assert digest == sha256_file(published / name)

    # --- no secret leaks into the manifest, the artifact metadata or the job log
    manifest_text = (published / ".yonga" / "manifest.json").read_text(encoding="utf-8")
    assert FAKE_TOKEN not in manifest_text
    log_text = settings.paths.job_log_path(plan.job_id).read_text(encoding="utf-8")
    assert FAKE_TOKEN not in log_text
    assert "***REDACTED***" in log_text

    # --- the store agrees the artifact is ready
    store = ArtifactStore(settings.paths, settings)
    summaries = store.list_artifacts()
    assert [summary.path for summary in summaries] == [published]
    assert summaries[0].ready is True
    assert summaries[0].repo_id == REPO_ID
    assert store.find(REPO_ID) == published

    # The counter lives in current/total; the message is the bare label.
    converting = sink.of_kind(ProgressKind.STAGE)[4]
    assert (converting.message, converting.current, converting.total) == ("Converting", 5, 7)
    stages = [event.message for event in sink.of_kind(ProgressKind.STAGE)]
    assert stages[-1].startswith("Artifact ready at ")


def test_failing_export_is_classified_and_publishes_nothing(
    settings: Settings, hub: FakeHub
) -> None:
    """A non-zero exit from the real child is diagnosed, and no artifact appears."""
    service = integration_service(settings, hub)
    plan = service.plan(REPO_ID, device="cpu", profile_id="test-export-disk-full")

    with pytest.raises(ExportError) as caught:
        run_prepare(service, plan, settings)

    failure = caught.value.diagnostic
    assert failure.code == "storage.insufficient_disk"
    assert failure.raw_excerpt is not None
    assert "No space left on device" in failure.raw_excerpt
    assert not plan.artifact_dir.exists()
    assert ArtifactStore(settings.paths, settings).list_artifacts() == []
    # The staging directory is left behind, still marked as an unfinished build.
    assert (plan.staging_dir / ".yonga-incomplete").is_file()


def test_incomplete_staging_is_never_listed_as_ready(settings: Settings, hub: FakeHub) -> None:
    """A failed run leaves staging behind; nothing about it may look ready."""
    service = integration_service(settings, hub)
    plan = service.plan(REPO_ID, device="cpu", profile_id="test-export-disk-full")
    with pytest.raises(ExportError):
        run_prepare(service, plan, settings)

    store = ArtifactStore(settings.paths, settings)
    # Even if a manifest somehow ends up in the staging tree, it is not an artifact.
    metadata_dir = plan.staging_dir / ".yonga"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "manifest.json").write_text("{}", encoding="utf-8")

    assert store.list_artifacts() == []
    assert store.is_ready(plan.staging_dir) is False


def test_reuse_skips_the_subprocess_entirely(settings: Settings, hub: FakeHub) -> None:
    """The second prepare of an identical revision and profile spawns no child."""
    service = integration_service(settings, hub)
    first = run_prepare(service, service.plan(REPO_ID, device="gpu"), settings)
    first_mtime = (first.artifact_dir / "argv.json").stat().st_mtime_ns

    second_plan = service.plan(REPO_ID, device="gpu")
    assert second_plan.reuse_existing_artifact == first.artifact_dir
    second = run_prepare(service, second_plan, settings)

    assert second.reused is True
    assert second.artifact_dir == first.artifact_dir
    assert (first.artifact_dir / "argv.json").stat().st_mtime_ns == first_mtime
    assert second.manifest.artifact_id == first.manifest.artifact_id


@pytest.mark.openvino
def test_published_tokenizers_load_in_the_real_openvino_runtime(
    settings: Settings, hub: FakeHub, tmp_path: Path
) -> None:
    """The repaired artifact this pipeline publishes is genuinely loadable.

    The export fixture cannot emit tokenizer IR, so the authentic broken fixture
    is injected through a thin exporter wrapper; everything after that — repair
    detection, the repair itself and the final check — is the real code path
    against the installed OpenVINO 2026.4 runtime.
    """
    openvino_probe = pytest.importorskip("yonga.adapters.openvino_probe")

    class TokenizerWritingExporter:
        """The real exporter, plus the tokenizer pair a real export would emit."""

        def __init__(self) -> None:
            self._inner = real_exporter()

        def requirements(self) -> dict[str, str | None]:
            """Delegate to the wrapped exporter."""
            return self._inner.requirements()

        def export(self, request: ExportRequest, ctx: JobContext) -> ExportResult:
            """Run the real export, then add the tokenizer files it cannot write."""
            result = self._inner.export(request, ctx)
            if result.success:
                write_tokenizer_pair(request.output_path, broken=True)
            return result

    service = build_service(
        settings=settings,
        hub=hub,
        exporter=TokenizerWritingExporter(),
        runtime=FakeRuntimeProbe(),
        smoke=FakeSmokeTester(),
        resolver=builtin_resolver([TEST_REGISTRY]),
    )
    plan = service.plan(REPO_ID, device="gpu")

    outcome = run_prepare(service, plan, settings, confirm=approve_all)

    assert [record.repair_id for record in outcome.repairs] == [
        "ov-tokenizer-precision-string-casing"
    ]
    probe = openvino_probe.OpenVINOProbe()
    if not probe.available()[0]:
        pytest.skip("OpenVINO is not importable on this host")
    result = probe.read_tokenizer(outcome.artifact_dir)
    assert result.status is ProbeStatus.OK, result.message
    assert result.properties["tokenizer_read"] == "ok"
    assert result.properties["detokenizer_compile"] == "ok"
    assert result.device == "CPU"


def test_reference_inventory_matches_the_host_facts_fixture() -> None:
    """Guard: the inventory the tests plan against is the documented one."""
    inventory = reference_inventory()
    assert inventory.available_device_names == ["CPU", "GPU", "NPU"]
    assert "intel.npu.lunar_lake" in inventory.device_tags


def test_profile_flags_reach_the_command_line(settings: Settings, hub: FakeHub) -> None:
    """Quantization settings become the exact flags Optimum Intel documents."""
    service = integration_service(settings, hub)
    plan = service.plan(REPO_ID, device="gpu")
    profile: ConversionProfile = plan.profile

    outcome = run_prepare(service, plan, settings)

    argv = outcome.manifest.export.argv
    assert profile.quantization.symmetric is False
    assert "--sym" not in argv  # omitting --sym is how asymmetric is spelled
    assert argv[argv.index("--weight-format") + 1] == "int4"
    assert "--group-size=128" in argv
    assert argv[argv.index("--ratio") + 1] == "1.0"
    assert "--trust-remote-code" not in argv


# --- the real stack ------------------------------------------------------------------


class GenAISmokeTester:
    """A :class:`~yonga.application.prepare.SmokeTester` over OpenVINO GenAI.

    This is the shape the CLI is expected to inject: the application layer keeps
    the port, and the concrete pipeline is built here, at the edge, where
    ``openvino_genai`` may legitimately be imported.
    """

    def smoke_test(
        self,
        artifact_dir: Path,
        *,
        device: str,
        pipeline: PipelineKind,
        properties: Mapping[str, str],
        prompt: str,
        max_new_tokens: int,
    ) -> ProbeResult:
        """Build the real pipeline, generate a few tokens and report the outcome."""
        from yonga.adapters.runtime_genai import ChatHistory, open_runtime

        started = time.perf_counter()
        history = ChatHistory()
        history.add_user(prompt)
        config = GenerationSettings(max_new_tokens=max_new_tokens, do_sample=False)
        runtime = open_runtime(
            artifact_dir,
            device=device,
            pipeline=pipeline,
            properties=dict(properties),
        )
        try:
            result = runtime.generate(history, config)
        finally:
            runtime.close()
        return ProbeResult(
            device=device,
            status=ProbeStatus.OK,
            duration_seconds=time.perf_counter() - started,
            message=f"generated {result.output_tokens} token(s)",
            properties={key: str(value) for key, value in properties.items()},
        )


@pytest.mark.openvino
def test_real_openvino_artifact_is_repaired_probed_and_published(
    tmp_path: Path, broken_artifact: Path
) -> None:
    """The whole pipeline against the installed OpenVINO 2026.4 stack.

    The source is a real, unrepaired OpenVINO export produced by the actual
    Optimum CLI, so every step after the download is genuine: the published IR is
    read by the real runtime, the ``STRiNG`` defect is detected and repaired by
    the real repair, the language model is compiled on the real CPU plugin, and a
    real GenAI pipeline generates tokens from it.

    The managed-temp requirement is lowered for this run because the fixture
    model is a few megabytes; the policy itself is exercised by
    ``tests/unit/test_prepare_planning.py``.
    """
    from yonga.adapters.filesystem import FilesystemAdapter
    from yonga.adapters.openvino_probe import OpenVINOProbe
    from yonga.adapters.system_linux import DeviceSnapshot, LinuxSystemProbe

    probe = OpenVINOProbe()
    available, error = probe.available()
    if not available:
        pytest.skip(f"OpenVINO is not importable: {error}")

    def device_source() -> DeviceSnapshot:
        """Hand the Linux probe the devices without it importing OpenVINO itself."""
        return DeviceSnapshot(devices=tuple(probe.devices()), available=True)

    repo_id = "hf-internal-testing/tiny-random-LlamaForCausalLM-ov"
    sha = "0" * 40
    metadata = ModelMetadata(
        identity=ModelIdentity(repo_id=repo_id, revision=sha),
        resolved_revision=sha,
        license="apache-2.0",
        pipeline_tag="text-generation",
        library_name="openvino",
        model_type="llama",
        config={"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
        files=[RepoFile(path="openvino_model.bin", size=7_000_000)],
        formats=[
            FormatObservation(
                format=ModelFormat.OPENVINO_IR,
                files=["openvino_model.xml", "openvino_model.bin"],
                total_bytes=7_000_000,
                complete=True,
            )
        ],
    )
    settings = settings_for(
        tmp_path,
        temp_min_bytes=1 * 1024**3,
        temp_safety_margin_bytes=256 * 1024**2,
    )
    hub = FakeHub(repos={repo_id: metadata}, snapshots={repo_id: broken_artifact})
    service = PrepareService(
        hub_client=hub,
        exporter=real_exporter(),
        runtime_probe=probe,
        hardware_probe=LinuxSystemProbe(
            settings=settings, paths=settings.paths, device_source=device_source
        ),
        filesystem=FilesystemAdapter(),
        registry_resolver=builtin_resolver(),
        artifact_store=ArtifactStore(settings.paths, settings),
        settings=settings,
        smoke_tester=GenAISmokeTester(),
    )

    plan = service.plan(repo_id, device="cpu")
    assert plan.resolved_source.already_openvino is True
    assert plan.disk.sufficient is True

    outcome = run_prepare(service, plan, settings)

    # The source in the Hub cache was never touched.
    assert (broken_artifact / "openvino_tokenizer.xml").read_text(encoding="utf-8").count(
        'precision="STRiNG"'
    ) == 2
    # The published artifact was repaired, validated, compiled and generated from.
    assert [record.repair_id for record in outcome.repairs] == [
        "ov-tokenizer-precision-string-casing"
    ]
    assert outcome.validation is not None
    assert outcome.validation.ok is True
    assert outcome.validation.tokenizer_compile_ok is True
    assert outcome.validation.openvino_version is not None
    assert outcome.device == "CPU"
    report = outcome.manifest.device_probes["CPU"]
    assert report.status is ProbeStatus.OK
    assert report.smoke_test is not None
    assert report.smoke_test.ok is True
    assert outcome.manifest.export.backend == "openvino-ir-passthrough"
    assert ArtifactStore(settings.paths, settings).list_artifacts()[0].ready is True
    published = outcome.artifact_dir / "openvino_tokenizer.xml"
    assert 'precision="STRiNG"' not in published.read_text(encoding="utf-8")
    assert 'element_type="string"' in published.read_text(encoding="utf-8")
