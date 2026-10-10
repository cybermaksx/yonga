"""Export failures that must be diagnosed, and diagnosed before the download where possible.

Each case here came from a real run on the reference host (``build/e2e``):

* SmolLM2-360M (hidden size 960) with a group-128 profile: NNCF listed 118
  refused nodes after its exception line, which pushed that line out of the
  exporter's 40-line tail, so the known ``InvalidGroupSizeError`` signature was
  never seen and the job ended as a generic ``export.failed`` after a 4.4 GiB
  download. The group size cannot divide 960, which ``config.json`` says before
  anything is downloaded.
* A pip install with torch from the CPU index and torchvision from PyPI: every
  export died on ``operator torchvision::nms does not exist``, unclassified,
  while every version check passed.
* Remediation steps that printed ``MODEL`` and ``JOB_ID`` although both were
  known, and listed the same ``--device gpu`` command twice.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pytest

from tests.unit.test_benchmark_service import FakeHardwareProbe as BenchHardwareProbe
from tests.unit.test_benchmark_service import RecordingFactory
from tests.unit.test_benchmark_service import make_settings as bench_settings
from tests.unit.test_chat_service import make_manifest
from tests.unit.test_exporter_optimum import LAUNCHER, FakeRunner, make_request
from tests.unit.test_prepare_planning import (
    FakeExporter,
    FakeHub,
    build_service,
    make_snapshot,
    run_prepare,
    safetensors_metadata,
)
from yonga.adapters import ChecksToolchainImport
from yonga.adapters import exporter_optimum as exporter_module
from yonga.adapters.exporter_optimum import TAIL_LINES, OptimumExporter
from yonga.application.benchmark import BenchmarkService
from yonga.application.jobs import JobContext, open_job
from yonga.application.probe import LocalArtifactStore
from yonga.config.settings import Settings, load_settings
from yonga.domain.artifacts import manifest_path
from yonga.domain.diagnostics import (
    Diagnostic,
    ExitCode,
    FailureCategory,
    PreflightError,
    RegistryError,
    YongaError,
    diagnostic,
    fill_placeholders,
)
from yonga.domain.models import ModelIdentity, ModelMetadata
from yonga.domain.plans import QuantizationSettings
from yonga.registry.loader import load_builtin_registry
from yonga.registry.resolver import (
    GROUP_SIZE_MISMATCH_CODE,
    RegistryResolver,
    channel_widths,
    group_size_conflict,
)
from yonga.ui.console import diagnostic_lines

SMOLLM2_REPO = "HuggingFaceTB/SmolLM2-360M-Instruct"

HOST_VERSIONS: dict[str, str | None] = {
    "openvino": "2026.4.0",
    "openvino_genai": "2026.4.0.0",
    "openvino_tokenizers": "2026.4.0.0",
    "transformers": "5.5.4",
}

# The shape of the real SmolLM2-360M failure: the exception line, then one line
# per refused node, 118 of them (job 79c02ccd on the reference host).
NNCF_HEADER = (
    "nncf.errors.InvalidGroupSizeError: Failed to apply group-wise quantization with group "
    "size value 128."
)
SMOLLM2_STDERR = [
    "Traceback (most recent call last):",
    '  File "/env/lib/python3.14/site-packages/nncf/quantization/algorithms/'
    'weight_compression/algorithm.py", line 679, in validate_group_size',
    "    raise nncf.InvalidGroupSizeError(msg)",
    NNCF_HEADER,
    "Ensure that the group size is divisible by the channel size, or consider setting",
    "`group_size_fallback_mode` to IGNORE or ADJUST. Failed nodes:",
    *(
        f'\t"__module.model.layers.{index // 4}.mlp.up_proj/ov_ext::linear/MatMul" '
        "(channel size: 960)"
        for index in range(118)
    ),
]

# The torchvision/torch build mismatch: the root cause is a chained exception
# whose header sits above the final one (from the pip E2E run's job log).
TORCHVISION_STDERR = [
    "Traceback (most recent call last):",
    '  File "/venv/lib/python3.14/site-packages/torch/library.py", line 354, in _register_fake',
    *("    frame line" for _ in range(40)),
    "RuntimeError: operator torchvision::nms does not exist",
    "",
    "The above exception was the direct cause of the following exception:",
    "",
    "Traceback (most recent call last):",
    *("    frame line" for _ in range(30)),
    "ModuleNotFoundError: Could not import module 'PreTrainedModel'. Are this object's "
    "requirements defined correctly?",
]


def resolver() -> RegistryResolver:
    """The shipped registry with the reference host's versions."""
    return RegistryResolver(load_builtin_registry(), versions=HOST_VERSIONS)


def smollm2_metadata(**config: object) -> ModelMetadata:
    """Hub metadata for SmolLM2-360M-Instruct as the Hub reports it."""
    payload: dict[str, object] = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 960,
        "intermediate_size": 2560,
    }
    payload.update(config)
    return safetensors_metadata(
        repo_id=SMOLLM2_REPO, model_type="llama", pipeline_tag="text-generation"
    ).model_copy(
        update={
            "config": payload,
            "architectures": ["LlamaForCausalLM"],
            "parameter_count": 361_821_120,
        }
    )


@pytest.fixture
def job(tmp_path: Path) -> Iterator[JobContext]:
    """A real job context rooted inside ``tmp_path``."""
    settings = load_settings(
        env={},
        overrides={
            "data_dir": str(tmp_path / "data"),
            "cache_dir": str(tmp_path / "cache"),
            "config_dir": str(tmp_path / "config"),
        },
    )
    with open_job(settings=settings, requested_model="acme/model", job_id="job-c3") as ctx:
        yield ctx


# --- the exception line survives a long failure listing --------------------------------


def test_a_long_node_list_does_not_push_the_exception_out_of_the_tail(job: JobContext) -> None:
    runner = FakeRunner(returncode=1, stderr=SMOLLM2_STDERR)
    result = OptimumExporter(runner, launcher=LAUNCHER).export(make_request(job), job)

    assert NNCF_HEADER in result.stderr_tail
    # The tail itself is untouched: it still ends with the last line printed.
    assert result.stderr_tail[-TAIL_LINES:] == SMOLLM2_STDERR[-TAIL_LINES:]
    assert any("lines omitted" in line for line in result.stderr_tail)
    assert result.diagnostic is not None
    assert NNCF_HEADER in (result.diagnostic.raw_excerpt or "")


def test_the_kept_exception_line_is_classified_as_the_group_size_mismatch(
    job: JobContext,
) -> None:
    runner = FakeRunner(returncode=1, stderr=SMOLLM2_STDERR)
    result = OptimumExporter(runner, launcher=LAUNCHER).export(make_request(job), job)

    classified = resolver().classify(
        raw_error="\n".join([*result.stdout_tail, *result.stderr_tail]),
        device="CPU",
        stage="export",
    )
    assert classified.code == GROUP_SIZE_MISMATCH_CODE
    assert classified.exit_code() is ExitCode.REGISTRY


def test_a_chained_root_cause_above_the_tail_is_kept_too(job: JobContext) -> None:
    runner = FakeRunner(returncode=1, stderr=TORCHVISION_STDERR)
    result = OptimumExporter(runner, launcher=LAUNCHER).export(make_request(job), job)

    assert "RuntimeError: operator torchvision::nms does not exist" in result.stderr_tail
    assert result.stderr_tail[-1].startswith("ModuleNotFoundError")


def test_a_successful_export_keeps_a_plain_tail(job: JobContext) -> None:
    runner = FakeRunner(returncode=0, stderr=SMOLLM2_STDERR)
    result = OptimumExporter(runner, launcher=LAUNCHER).export(make_request(job), job)

    assert result.stderr_tail == SMOLLM2_STDERR[-TAIL_LINES:]


# --- torchvision built against another torch ------------------------------------------


def test_the_torchvision_build_mismatch_is_an_environment_failure() -> None:
    result = resolver().classify(
        raw_error="\n".join(TORCHVISION_STDERR[-35:] + TORCHVISION_STDERR[:45]),
        stage="export",
    )

    assert result.code == "export.torchvision.build_mismatch"
    assert result.matched_rule_id == "torchvision-torch-build-mismatch"
    assert result.category is FailureCategory.PACKAGE_ABI_MISMATCH
    assert result.exit_code() is ExitCode.ENVIRONMENT_UNUSABLE
    assert result.environment_broken is True
    assert result.fallback_devices == []
    joined = " ".join(result.remediation)
    assert "torch torchvision" in joined
    assert "https://download.pytorch.org/whl/cpu" in joined


def test_the_installation_guide_installs_torchvision_from_the_cpu_index_too() -> None:
    """The documented pip path is how the mismatch was produced on the reference host."""
    text = (Path(__file__).resolve().parents[2] / "docs" / "INSTALL.md").read_text(encoding="utf-8")
    assert "pip install torch --index-url" not in text
    assert "pip install torch torchvision --index-url" in text


# --- the exporter is imported before the download -------------------------------------


@pytest.fixture
def optimum_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Behave as if the Optimum CLI module were installed beside yonga."""
    monkeypatch.setattr(exporter_module, "_module_importable", lambda name: True)


@pytest.mark.usefixtures("optimum_present")
def test_an_import_that_works_is_checked_once_per_environment(tmp_path: Path) -> None:
    runner = FakeRunner(returncode=0)
    exporter = OptimumExporter(runner, cache_dir=tmp_path / "cache")

    assert exporter.toolchain_import_failure() is None
    assert exporter.toolchain_import_failure() is None
    assert len(runner.calls) == 1
    argv = runner.calls[0].argv
    assert argv[1:3] == ["-P", "-c"]
    assert "import optimum.exporters.openvino" in argv[3]


@pytest.mark.usefixtures("optimum_present")
def test_an_import_that_fails_reports_its_error_and_is_asked_again(tmp_path: Path) -> None:
    runner = FakeRunner(returncode=1, stderr=TORCHVISION_STDERR)
    exporter = OptimumExporter(runner, cache_dir=tmp_path / "cache")

    failure = exporter.toolchain_import_failure()
    assert failure is not None
    assert "operator torchvision::nms does not exist" in failure
    assert exporter.toolchain_import_failure() is not None
    assert len(runner.calls) == 2


def test_nothing_is_imported_without_an_exporter_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exporter_module, "_module_importable", lambda name: False)
    runner = FakeRunner(returncode=1)
    assert OptimumExporter(runner).toolchain_import_failure() is None
    assert runner.calls == []


def test_the_exporter_satisfies_the_import_check_protocol() -> None:
    assert isinstance(OptimumExporter(FakeRunner()), ChecksToolchainImport)


class BrokenImportExporter(FakeExporter):
    """An exporter whose environment cannot import the export toolchain."""

    def __init__(self, failure: str | None) -> None:
        super().__init__()
        self.failure = failure
        self.checked = 0

    def toolchain_import_failure(self) -> str | None:
        """Report the canned import failure."""
        self.checked += 1
        return self.failure


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings confined to the test's temporary directory."""
    return Settings.model_validate(
        {
            "config_dir": tmp_path / "config",
            "cache_dir": tmp_path / "cache",
            "data_dir": tmp_path / "data",
        }
    )


def qwen3_hub(tmp_path: Path) -> FakeHub:
    """A hub with one small safetensors model whose group size fits."""
    repo = "Qwen/Qwen3-0.6B"
    meta = safetensors_metadata(
        repo_id=repo, model_type="qwen3", pipeline_tag="text-generation"
    ).model_copy(
        update={
            "config": {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 1024,
                "intermediate_size": 3072,
            },
            "architectures": ["Qwen3ForCausalLM"],
        }
    )
    return FakeHub(repos={repo: meta}, snapshots={repo: make_snapshot(tmp_path, "qwen3")})


def test_prepare_refuses_before_the_download_when_the_exporter_cannot_import(
    settings: Settings, tmp_path: Path
) -> None:
    hub = qwen3_hub(tmp_path)
    exporter = BrokenImportExporter("\n".join(TORCHVISION_STDERR))
    service = build_service(settings=settings, hub=hub, exporter=exporter)
    plan = service.plan("Qwen/Qwen3-0.6B", device="cpu")

    with pytest.raises(PreflightError) as caught:
        run_prepare(service, plan, settings)

    assert caught.value.diagnostic.code == "export.torchvision.build_mismatch"
    assert caught.value.exit_code() is ExitCode.ENVIRONMENT_UNUSABLE
    assert hub.downloaded == []
    assert exporter.requests == []


def test_an_unrecognised_import_failure_still_stops_before_the_download(
    settings: Settings, tmp_path: Path
) -> None:
    hub = qwen3_hub(tmp_path)
    exporter = BrokenImportExporter("SyntaxError: something nobody has seen before")
    service = build_service(settings=settings, hub=hub, exporter=exporter)
    plan = service.plan("Qwen/Qwen3-0.6B", device="cpu")

    with pytest.raises(PreflightError) as caught:
        run_prepare(service, plan, settings)

    assert caught.value.diagnostic.category is FailureCategory.PACKAGE_ABI_MISMATCH
    assert caught.value.exit_code() is ExitCode.ENVIRONMENT_UNUSABLE
    assert hub.downloaded == []


def test_a_working_import_lets_the_preparation_through(settings: Settings, tmp_path: Path) -> None:
    hub = qwen3_hub(tmp_path)
    exporter = BrokenImportExporter(None)
    service = build_service(settings=settings, hub=hub, exporter=exporter)
    plan = service.plan("Qwen/Qwen3-0.6B", device="cpu")

    run_prepare(service, plan, settings)

    assert exporter.checked == 1
    assert hub.downloaded == ["Qwen/Qwen3-0.6B"]


# --- a group size the model cannot take is ruled out before the download ---------------


def test_channel_widths_come_from_the_config_or_its_text_config() -> None:
    assert channel_widths(smollm2_metadata()) == {"hidden_size": 960, "intermediate_size": 2560}
    nested = ModelMetadata(
        identity=ModelIdentity(repo_id="acme/vlm"),
        config={"text_config": {"hidden_size": 1024, "intermediate_size": True}},
    )
    assert channel_widths(nested) == {"hidden_size": 1024}
    assert channel_widths(None) == {}


@pytest.mark.parametrize(
    ("quantization", "widths", "conflicts"),
    [
        (QuantizationSettings(group_size=128, ratio=1.0), {"hidden_size": 960}, True),
        # Most of the weights read the hidden size, so a mixed ratio cannot help.
        (QuantizationSettings(group_size=128, ratio=0.8), {"hidden_size": 960}, True),
        (QuantizationSettings(group_size=128, ratio=1.0), {"hidden_size": 1024}, False),
        # Only the down projections read the intermediate size: certain at 1.0 only.
        (QuantizationSettings(group_size=128, ratio=1.0), {"intermediate_size": 4864}, False),
        (QuantizationSettings(group_size=64, ratio=1.0), {"intermediate_size": 4900}, True),
        (QuantizationSettings(group_size=128, ratio=0.8), {"intermediate_size": 4900}, False),
        (QuantizationSettings(group_size=-1, ratio=1.0), {"hidden_size": 960}, False),
        (QuantizationSettings(weight_format="int8"), {"hidden_size": 960}, False),
        (QuantizationSettings(group_size=128), {}, False),
    ],
)
def test_a_group_size_conflict_is_reported_only_when_certain(
    quantization: QuantizationSettings, widths: Mapping[str, int], conflicts: bool
) -> None:
    assert (group_size_conflict(quantization, widths) is not None) is conflicts


def _profile_for(device: str, meta: ModelMetadata) -> tuple[str, str]:
    registry = resolver()
    arch_rule, _ = registry.resolve_architecture(meta)
    profile, explanation = registry.resolve_profile(device=device, arch_rule=arch_rule, meta=meta)
    assert explanation.profile_selection is not None
    return profile.id, explanation.profile_selection.sentence


def test_smollm2_on_the_cpu_gets_a_profile_that_can_quantize_it() -> None:
    profile_id, sentence = _profile_for("cpu", smollm2_metadata())

    assert profile_id == "cpu-int8-asym-ref"
    assert "cpu-int4-asym-g128-r08 cannot quantize this model" in sentence
    assert "hidden_size of 960" in sentence


def test_smollm2_on_the_npu_gets_the_channel_wise_profile() -> None:
    profile_id, sentence = _profile_for("npu", smollm2_metadata())

    assert profile_id == "npu-int4-sym-cw"
    assert "npu-int4-sym-g128 cannot quantize this model" in sentence


def test_no_gpu_profile_fits_smollm2_and_that_is_said_before_the_download() -> None:
    with pytest.raises(RegistryError) as caught:
        _profile_for("gpu", smollm2_metadata())

    found = caught.value.diagnostic
    assert found.code == GROUP_SIZE_MISMATCH_CODE
    assert found.exit_code() is ExitCode.REGISTRY
    assert found.context["repo_id"] == SMOLLM2_REPO
    # The way out is a device whose own choice fits, never the NPU profile on the
    # GPU: its MAX_PROMPT_LEN fails the GPU probe (verified on the reference host).
    steps = fill_placeholders(found).remediation
    assert any(
        "npu-int4-sym-cw fits" in step
        and step.endswith(f"yonga prepare {SMOLLM2_REPO} --device npu")
        for step in steps
    )
    assert any(
        "cpu-int8-asym-ref fits" in step
        and step.endswith(f"yonga prepare {SMOLLM2_REPO} --device cpu")
        for step in steps
    )
    assert not any("--profile npu-int4-sym-cw" in step for step in steps)


def test_a_model_whose_widths_fit_keeps_its_usual_profile() -> None:
    profile_id, _sentence = _profile_for("cpu", smollm2_metadata(hidden_size=1024))
    assert profile_id == "cpu-int4-asym-g128-r08"


def test_without_widths_nothing_is_ruled_out() -> None:
    meta = smollm2_metadata()
    meta = meta.model_copy(update={"config": {"model_type": "llama"}})
    profile_id, _sentence = _profile_for("cpu", meta)
    assert profile_id == "cpu-int4-asym-g128-r08"


def test_an_explicit_profile_that_cannot_fit_is_used_but_flagged() -> None:
    meta = smollm2_metadata()
    registry = resolver()
    arch_rule, _ = registry.resolve_architecture(meta)
    profile, explanation = registry.resolve_profile(
        device="cpu", arch_rule=arch_rule, meta=meta, explicit_profile_id="cpu-int4-asym-g128"
    )

    assert profile.id == "cpu-int4-asym-g128"
    assert any(GROUP_SIZE_MISMATCH_CODE in note for note in explanation.notes)


def smollm2_hub(tmp_path: Path) -> FakeHub:
    """A hub carrying SmolLM2-360M-Instruct."""
    return FakeHub(
        repos={SMOLLM2_REPO: smollm2_metadata()},
        snapshots={SMOLLM2_REPO: make_snapshot(tmp_path, "smollm2")},
    )


def test_an_explicit_gpu_is_refused_at_planning_without_downloading(
    settings: Settings, tmp_path: Path
) -> None:
    hub = smollm2_hub(tmp_path)
    service = build_service(settings=settings, hub=hub)

    with pytest.raises(RegistryError) as caught:
        service.plan(SMOLLM2_REPO, device="gpu")

    assert caught.value.diagnostic.code == GROUP_SIZE_MISMATCH_CODE
    assert hub.downloaded == []


def test_auto_moves_past_a_device_with_no_fitting_profile(
    settings: Settings, tmp_path: Path
) -> None:
    service = build_service(settings=settings, hub=smollm2_hub(tmp_path))

    plan = service.plan(SMOLLM2_REPO, device="auto")

    assert plan.compatibility.selected_device != "GPU"
    assert "GPU" not in plan.compatibility.fallback_candidates
    assert plan.profile.quantization.group_size in (None, -1)
    assert any("No GPU profile can quantize" in reason for reason in plan.compatibility.reasons)


def test_a_cpu_plan_says_why_the_int4_profiles_were_passed_over(
    settings: Settings, tmp_path: Path
) -> None:
    service = build_service(settings=settings, hub=smollm2_hub(tmp_path))

    plan = service.plan(SMOLLM2_REPO, device="cpu")

    assert plan.profile.id == "cpu-int8-asym-ref"
    assert plan.profile_selection is not None
    assert "cannot quantize this model" in plan.profile_selection.sentence


# --- remediation names the real model and job ------------------------------------------


def _with_steps(*steps: str, **fields: object) -> Diagnostic:
    return diagnostic(
        "test.code",
        FailureCategory.COMPILER_UNSUPPORTED_GRAPH,
        "Title",
        "Message.",
        remediation=list(steps),
        **fields,
    )


def test_known_values_replace_the_placeholders() -> None:
    item = _with_steps(
        "Run this model on the GPU instead: yonga prepare MODEL --device gpu",
        "The failing pass is in the job log: yonga logs JOB_ID --raw",
        job_id="0a12fcce",
        context={"repo_id": "Qwen/Qwen3.5-0.8B"},
    )

    filled = fill_placeholders(item)

    assert filled.remediation == [
        "Run this model on the GPU instead: yonga prepare Qwen/Qwen3.5-0.8B --device gpu",
        "The failing pass is in the job log: yonga logs 0a12fcce --raw",
    ]
    assert fill_placeholders(filled) == filled


def test_an_unknown_value_leaves_its_placeholder() -> None:
    item = _with_steps("yonga logs JOB_ID --raw, then yonga prepare MODEL", job_id="job-1")
    assert fill_placeholders(item).remediation == [
        "yonga logs job-1 --raw, then yonga prepare MODEL"
    ]


def test_only_whole_placeholder_words_are_replaced() -> None:
    item = _with_steps(
        "Set MODEL_DIR, not MODEL-ID; then yonga inspect MODEL",
        context={"repo_id": "acme/model"},
    )
    assert fill_placeholders(item).remediation == [
        "Set MODEL_DIR, not MODEL-ID; then yonga inspect acme/model"
    ]


def test_a_step_repeating_an_earlier_command_is_dropped() -> None:
    item = _with_steps(
        "Run this model on the GPU instead: yonga prepare MODEL --device gpu",
        "Re-check after an OpenVINO and NPU compiler update.",
        "Prepare for GPU instead: yonga prepare Qwen/Qwen3.5-0.8B --device gpu",
        "Prepare for CPU instead: yonga prepare Qwen/Qwen3.5-0.8B --device cpu",
        context={"repo_id": "Qwen/Qwen3.5-0.8B"},
    )

    steps = fill_placeholders(item).remediation

    assert steps == [
        "Run this model on the GPU instead: yonga prepare Qwen/Qwen3.5-0.8B --device gpu",
        "Re-check after an OpenVINO and NPU compiler update.",
        "Prepare for CPU instead: yonga prepare Qwen/Qwen3.5-0.8B --device cpu",
    ]


def test_the_classifier_fills_in_what_it_knows() -> None:
    meta = ModelMetadata(
        identity=ModelIdentity(repo_id="Qwen/Qwen3-0.6B"),
        model_type="qwen3",
        architectures=["Qwen3ForCausalLM"],
    )
    result = resolver().classify(
        raw_error="[vpux-compiler] Failed Pass MultiClusterStrategyAssignment",
        device="NPU",
        meta=meta,
        stage="probe",
        job_id="aeeb6a07",
    )

    joined = "\n".join(result.remediation)
    assert "yonga prepare Qwen/Qwen3-0.6B --device gpu" in joined
    assert "yonga logs aeeb6a07 --raw" in joined
    assert "MODEL" not in joined and "JOB_ID" not in joined


def test_rendering_fills_a_job_id_attached_after_classification() -> None:
    item = _with_steps("The failing pass is in the job log: yonga logs JOB_ID --raw")
    late = item.model_copy(update={"job_id": "70ab5941"})

    lines = diagnostic_lines(late)

    assert "  - The failing pass is in the job log: yonga logs 70ab5941 --raw" in lines


def test_a_bench_diagnosis_names_the_model_the_artifact_came_from(tmp_path: Path) -> None:
    settings = bench_settings(tmp_path)
    artifact = tmp_path / "artifacts" / "qwen35"
    artifact.mkdir(parents=True)
    (artifact / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    manifest = make_manifest()
    target = manifest_path(artifact)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(manifest.to_json_text(), encoding="utf-8")
    registry = resolver()

    def classify(raw: str, device: str) -> Diagnostic | None:
        return registry.classify(raw_error=raw, device=device, stage="bench")

    service = BenchmarkService(
        RecordingFactory(error=RuntimeError("[NPU_VCL] Compiler returned msg: no")),
        LocalArtifactStore(settings.paths),
        BenchHardwareProbe(),
        settings,
        classifier=classify,
    )
    with (
        open_job(settings=settings, requested_model="bench", job_id="job-bench") as ctx,
        pytest.raises(YongaError) as caught,
    ):
        service.run(artifact, device="NPU", suite="quick", ctx=ctx)

    found = caught.value.diagnostic
    assert found.code == "npu.compiler.pass_failed"
    assert found.context["repo_id"] == manifest.resolved_source.repo_id
    steps: Sequence[str] = fill_placeholders(found).remediation
    assert any(f"yonga prepare {manifest.resolved_source.repo_id}" in step for step in steps)
