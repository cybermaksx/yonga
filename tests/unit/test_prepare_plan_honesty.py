"""What the preparation plan promises, and what the job then actually does.

Every test here started as a real run on the reference host where the plan and
the job disagreed: a re-run that asked about an export it was never going to
do, a download several times the model, an explicit device the registry already
knew could not work, a core-only install blamed on the host, a confirmation
nobody could see reported as declined, a good adopted export exiting 2, an
artifact estimate three times too large, an auto-selection reason that was not
the reason, a gated repository discovered only at the download, and a plan
silent about which Python the export runs in.
"""

from __future__ import annotations

import io
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from huggingface_hub.utils import filter_repo_objects  # type: ignore[attr-defined]
from rich.console import Console

from tests.unit.test_prepare_managed_env import FakeEnvironments
from tests.unit.test_prepare_planning import (
    SAFETENSORS_REPO,
    FakeExporter,
    FakeHardwareProbe,
    FakeHub,
    FakeRuntimeProbe,
    build_service,
    make_snapshot,
    reference_inventory,
    run_prepare,
    safetensors_metadata,
    settings_for,
)
from yonga.adapters.hf import detect_formats
from yonga.application.jobs import JobContext
from yonga.config.settings import Settings
from yonga.domain.diagnostics import (
    CancelledError,
    DeviceUnsupportedError,
    ExitCode,
    HubAuthError,
    PreflightError,
)
from yonga.domain.hardware import HuggingFaceAuthInfo
from yonga.domain.models import ModelFormat, ModelIdentity, ModelMetadata, RepoFile
from yonga.domain.plans import (
    CONFIRMATION_REMOTE_CODE,
    ConfirmationReason,
    PreparationPlan,
    QuantizationSettings,
    bytes_to_download,
    download_ignore_patterns,
    estimate_artifact_bytes,
    is_ignored,
)
from yonga.ui.tables import preparation_plan

MIB = 1024**2

SMOL_REPO = "HuggingFaceTB/SmolLM2-360M-Instruct"
SMOL_SHA = "a10cc1512eabd3dde888204e902eca88bddb4951"

#: The files of the real SmolLM2-360M-Instruct snapshot (sizes rounded), which
#: ``prepare`` downloaded in full: 4.4 GiB for a 690 MiB model.
SMOL_FILES: tuple[tuple[str, int], ...] = (
    ("model.safetensors", 723_674_912),
    ("config.json", 861),
    ("generation_config.json", 132),
    ("tokenizer.json", 2_104_556),
    ("tokenizer_config.json", 3_764),
    ("special_tokens_map.json", 655),
    ("vocab.json", 800_662),
    ("merges.txt", 466_391),
    ("README.md", 4_000),
    (".gitattributes", 1_519),
    ("all_results.json", 300),
    ("train_results.json", 300),
    ("eval_results.json", 300),
    ("trainer_state.json", 90_000),
    ("training_args.bin", 7_416),
    *((f"onnx/model{suffix}.onnx", 470_000_000) for suffix in ("", "_fp16", "_int8", "_q4")),
    *((f"onnx/model{suffix}.onnx", 390_000_000) for suffix in ("_bnb4", "_q4f16", "_uint8")),
    ("onnx/model_quantized.onnx", 390_000_000),
    ("runs/Oct31_09-01-58_ip/events.out.tfevents.1730371773.ip.451351.1", 6_000),
    ("runs/Oct31_09-01-58_ip/events.out.tfevents.1730365788.ip.451351.0", 9_000_000),
)


def repo_metadata(
    repo_id: str,
    files: Sequence[tuple[str, int]],
    *,
    sha: str,
    model_type: str = "llama",
    parameter_count: int | None = None,
    gated: bool = False,
) -> ModelMetadata:
    """Metadata for a repository listing ``files``, with formats detected as the adapter does."""
    entries = [RepoFile(path=path, size=size, lfs=size > 1_000_000) for path, size in files]
    return ModelMetadata(
        identity=ModelIdentity(repo_id=repo_id, revision=sha),
        resolved_revision=sha,
        license="apache-2.0",
        pipeline_tag="text-generation",
        library_name="transformers",
        model_type=model_type,
        config={"model_type": model_type, "architectures": ["LlamaForCausalLM"]},
        files=entries,
        formats=detect_formats(entries),
        parameter_count=parameter_count,
        parameter_dtypes={"BF16": parameter_count} if parameter_count else {},
        gated=gated,
    )


@dataclass
class RecordingHub(FakeHub):
    """A hub that remembers which patterns each download was given."""

    ignored: list[list[str] | None] = field(default_factory=list)
    logged_in: bool = False

    def download(
        self,
        source: ModelIdentity,
        *,
        ctx: JobContext | None = None,
        allow_patterns: Sequence[str] | None = None,
        ignore_patterns: Sequence[str] | None = None,
    ) -> Path:
        """Record the ignore patterns, then behave as the plain fake does."""
        self.ignored.append(None if ignore_patterns is None else list(ignore_patterns))
        return super().download(
            source, ctx=ctx, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns
        )

    def auth_info(self) -> HuggingFaceAuthInfo:
        """Report a session with or without a token, as configured."""
        return HuggingFaceAuthInfo(logged_in=self.logged_in)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return settings_for(tmp_path)


@pytest.fixture
def hub(tmp_path: Path) -> RecordingHub:
    return RecordingHub(
        repos={
            SAFETENSORS_REPO: safetensors_metadata(),
            SMOL_REPO: repo_metadata(SMOL_REPO, SMOL_FILES, sha=SMOL_SHA),
        },
        snapshots={
            SAFETENSORS_REPO: make_snapshot(tmp_path, "snapshot-safetensors"),
            SMOL_REPO: make_snapshot(tmp_path, "snapshot-smol"),
        },
    )


def render(plan: PreparationPlan) -> str:
    """The plan panel as plain text, wide enough that sentences stay on one line."""
    console = Console(width=240, no_color=True, file=io.StringIO())
    console.print(preparation_plan(plan))
    output = console.file
    assert isinstance(output, io.StringIO)
    return output.getvalue()


def refuse_everything(reason: ConfirmationReason) -> bool:
    """A non-interactive session without ``--yes``: every question is answered no."""
    return False


# --- PREP-1: a reused artifact asks nothing --------------------------------------------


def test_a_reused_artifact_asks_no_confirmation_and_plans_no_work(
    settings: Settings, hub: RecordingHub
) -> None:
    """Nothing is exported, so there is no repair, download or environment to confirm."""
    service = build_service(settings=settings, hub=hub)
    first = run_prepare(service, service.plan(SAFETENSORS_REPO, device="gpu"), settings)

    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    assert plan.reuse_existing_artifact == first.artifact_dir
    assert plan.confirmations == []
    assert plan.expected_repairs == []
    assert plan.export_environment is None
    assert not plan.exporter_requirements_block
    outcome = run_prepare(service, plan, settings, confirm=refuse_everything)
    assert outcome.reused is True
    assert hub.downloaded == [SAFETENSORS_REPO]
    text = render(plan)
    assert "Reusing artifact" in text
    for absent in ("Expected download", "Estimated temp", "Confirmation required", "repairs"):
        assert absent not in text


def test_a_reuse_that_vanished_before_the_job_does_not_export_unconfirmed(
    settings: Settings, hub: RecordingHub
) -> None:
    """The plan dropped its confirmations for the reuse; exporting anyway would skip them."""
    service = build_service(settings=settings, hub=hub)
    first = run_prepare(service, service.plan(SAFETENSORS_REPO, device="gpu"), settings)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    (first.artifact_dir / ".yonga" / "manifest.json").unlink()

    with pytest.raises(PreflightError) as caught:
        run_prepare(service, plan, settings, confirm=refuse_everything)

    assert caught.value.diagnostic.code == "prepare.reuse_unavailable"
    assert caught.value.exit_code() is ExitCode.PREFLIGHT
    assert hub.downloaded == [SAFETENSORS_REPO]


# --- PREP-2: download only what the conversion reads -----------------------------------


def test_the_download_skips_onnx_runs_and_training_files(
    settings: Settings, hub: RecordingHub
) -> None:
    """SmolLM2's snapshot was 4.4 GiB; the export reads the 690 MiB of safetensors and configs."""
    service = build_service(settings=settings, hub=hub)

    plan = service.plan(SMOL_REPO, device="cpu")

    paths = [path for path, _ in SMOL_FILES]
    fetched = set(filter_repo_objects(paths, ignore_patterns=plan.download_ignore_patterns))
    assert "model.safetensors" in fetched
    for kept in ("config.json", "tokenizer.json", "tokenizer_config.json", "merges.txt"):
        assert kept in fetched
    assert not any(path.startswith(("onnx/", "runs/")) for path in fetched)
    assert "training_args.bin" not in fetched
    # The plan's figure is exactly what huggingface_hub will fetch, not the repository.
    sizes = dict(SMOL_FILES)
    assert plan.disk.expected_download_bytes == sum(sizes[path] for path in fetched)
    assert plan.disk.expected_download_bytes < 700 * MIB
    assert fetched == {
        path for path in paths if not is_ignored(path, plan.download_ignore_patterns)
    }

    run_prepare(service, plan, settings)
    assert hub.ignored == [plan.download_ignore_patterns]


def test_bin_weights_are_skipped_only_when_safetensors_carry_the_model() -> None:
    """A repository with both formats is fetched once; a ``.bin``-only one keeps its weights."""
    both = repo_metadata(
        "acme/both",
        [
            ("model.safetensors", 1_000_000_000),
            ("pytorch_model.bin", 1_000_000_000),
            ("config.json", 800),
        ],
        sha="b" * 40,
    )
    patterns = download_ignore_patterns(
        both.files, both.formats, source_format=ModelFormat.SAFETENSORS
    )
    assert is_ignored("pytorch_model.bin", patterns)
    assert not is_ignored("model.safetensors", patterns)
    assert bytes_to_download(both.files, patterns) == 1_000_000_800

    only_bin = repo_metadata(
        "acme/bin", [("pytorch_model.bin", 1_000_000_000), ("config.json", 800)], sha="c" * 40
    )
    patterns = download_ignore_patterns(
        only_bin.files, only_bin.formats, source_format=ModelFormat.PYTORCH_BIN
    )
    assert not is_ignored("pytorch_model.bin", patterns)


def test_remote_code_downloads_the_whole_repository() -> None:
    """The repository's own code may open any file it ships, so nothing is filtered."""
    meta = repo_metadata(SMOL_REPO, SMOL_FILES, sha=SMOL_SHA)
    assert (
        download_ignore_patterns(
            meta.files, meta.formats, source_format=ModelFormat.SAFETENSORS, remote_code=True
        )
        == []
    )


def test_a_path_with_glob_characters_is_excluded_literally() -> None:
    """An excluded weight file is matched by its exact path, brackets included."""
    meta = repo_metadata(
        "acme/odd",
        [("model.safetensors", 10), ("weights[1].bin", 10), ("weights1.bin", 10)],
        sha="d" * 40,
    )
    patterns = download_ignore_patterns(
        meta.files, meta.formats, source_format=ModelFormat.SAFETENSORS
    )
    assert is_ignored("weights[1].bin", patterns)
    fetched = set(filter_repo_objects(["weights[1].bin", "weights1.bin"], ignore_patterns=patterns))
    assert fetched == set()


# --- PREP-3: an explicit device the registry rules out ---------------------------------


def test_an_explicit_device_the_registry_rules_out_stops_before_the_download(
    settings: Settings, hub: RecordingHub
) -> None:
    """Qwen3.5 on the NPU is unsupported (tested): exit 50 with the GPU fallback, no download."""
    runtime = FakeRuntimeProbe()
    service = build_service(settings=settings, hub=hub, runtime=runtime)
    plan = service.plan(SAFETENSORS_REPO, device="npu")

    assert plan.device_known_unsupported
    assert plan.confirmations == []
    with pytest.raises(DeviceUnsupportedError) as caught:
        run_prepare(service, plan, settings, confirm=refuse_everything)

    failure = caught.value.diagnostic
    assert failure.code == "device.known_unsupported"
    assert caught.value.exit_code() is ExitCode.DEVICE_UNSUPPORTED
    assert failure.message.startswith("NPU preparation cannot be completed")
    assert failure.fallback_devices[0] == "GPU"
    assert any("--device gpu" in step for step in failure.remediation)
    assert hub.downloaded == []
    assert runtime.compiled == []
    assert "BLOCKED" in render(plan)


def test_auto_mode_is_not_refused_for_a_device_it_would_never_pick(
    settings: Settings, hub: RecordingHub
) -> None:
    """Only an explicit request is refused; auto-selection already skips the NPU."""
    service = build_service(settings=settings, hub=hub)

    plan = service.plan(SAFETENSORS_REPO, device="auto")

    assert plan.target_device != "NPU"
    assert not plan.device_known_unsupported


# --- PREP-4: the OpenVINO runtime is missing -------------------------------------------


def test_a_missing_runtime_is_named_before_any_device_or_environment_decision(
    settings: Settings, hub: RecordingHub
) -> None:
    """Exit 10 naming the extra; no managed environment is planned or confirmed."""
    inventory = reference_inventory(devices=()).model_copy(
        update={"openvino_available": False, "openvino_import_error": "No module named 'openvino'"}
    )
    exporter = FakeExporter(versions={"transformers": None, "optimum-intel": None})
    envs = FakeEnvironments(settings.paths.envs_dir)
    service = build_service(
        settings=settings,
        hub=hub,
        hardware=FakeHardwareProbe(inventory),
        exporter=exporter,
        environments=envs,
    )
    # With the runtime present the same plan would build a managed environment.
    healthy = build_service(settings=settings, hub=hub, exporter=exporter, environments=envs)
    assert healthy.plan(SAFETENSORS_REPO, device="cpu").export_environment is not None

    plan = service.plan(SAFETENSORS_REPO, device="cpu")

    assert plan.runtime_missing
    assert plan.export_environment is None
    assert envs.ensured == []
    assert plan.confirmations == []
    with pytest.raises(PreflightError) as caught:
        run_prepare(service, plan, settings)
    assert caught.value.exit_code() is ExitCode.ENVIRONMENT_UNUSABLE
    assert caught.value.diagnostic.code == "runtime.openvino.missing"
    assert "pip install 'yonga[openvino]'" in caught.value.diagnostic.remediation[0]
    assert hub.downloaded == []
    assert "yonga[openvino]" in render(plan)


# --- PREP-5: a question nobody could answer --------------------------------------------


def test_an_unattended_refusal_is_not_reported_as_declined(
    settings: Settings, hub: RecordingHub
) -> None:
    """No prompt was shown: the diagnostic says so and names ``--yes``."""
    service = build_service(settings=settings, hub=hub)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    assert plan.confirmations

    with pytest.raises(CancelledError) as caught:
        with_unattended(service, plan, settings)

    failure = caught.value.diagnostic
    assert failure.code == "job.confirmation_unavailable"
    assert caught.value.exit_code() is ExitCode.CANCELLED
    assert "declined" not in failure.message
    assert "could not be asked" in failure.message
    assert any("--yes" in step for step in failure.remediation)
    assert hub.downloaded == []


def test_an_unattended_security_question_says_yes_cannot_answer_it(
    settings: Settings, tmp_path: Path
) -> None:
    """Remote code needs a person at a terminal; ``--yes`` is not offered as the way out."""
    repo = "acme/remote-code-model"
    hub = RecordingHub(
        repos={repo: safetensors_metadata(repo, sha="a" * 40, requires_remote_code=True)},
        snapshots={repo: make_snapshot(tmp_path)},
    )
    service = build_service(settings=settings, hub=hub)
    plan = service.plan(repo, device="gpu", trust_remote_code=True)

    with pytest.raises(CancelledError) as caught:
        with_unattended(service, plan, settings, confirm=lambda r: not r.security_sensitive)

    failure = caught.value.diagnostic
    assert failure.context["confirmation"] == CONFIRMATION_REMOTE_CODE
    assert any("--yes never approves" in step for step in failure.remediation)
    assert not any("Re-run the same command with --yes" in step for step in failure.remediation)


def with_unattended(
    service: object,
    plan: object,
    settings: Settings,
    *,
    confirm: object = refuse_everything,
) -> object:
    """Run a plan the way the CLI does when stdin is not a terminal."""
    from yonga.application.jobs import open_job
    from yonga.application.progress import CollectingSink

    with open_job(
        settings=settings,
        requested_model=plan.requested.repo_id,  # type: ignore[attr-defined]
        job_id=plan.job_id,  # type: ignore[attr-defined]
        progress=CollectingSink(),
    ) as ctx:
        return service.prepare(  # type: ignore[attr-defined]
            plan, confirm=confirm, ctx=ctx, unattended=True
        )


# --- PREP-6: an adopted export is a fact, not a warning -------------------------------


def test_an_adopted_export_is_a_note_not_a_warning(settings: Settings, hub: RecordingHub) -> None:
    """A good artifact from a retained export must not make the command exit 2."""
    from tests.unit.test_prepare_export_reuse import fail_after_export

    failed_job, _ = fail_after_export(settings, hub)
    service = build_service(settings=settings, hub=hub, exporter=FakeExporter())
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    assert plan.reusable_export is not None

    outcome = run_prepare(service, plan, settings)

    assert not any("reused from failed job" in warning for warning in outcome.warnings)
    assert any(f"reused from failed job {failed_job}" in note for note in outcome.notes)
    assert any(f"reused from failed job {failed_job}" in n for n in outcome.manifest.notes)


# --- PREP-7: the artifact estimate -----------------------------------------------------


@pytest.mark.parametrize(
    ("parameters", "quantization", "real_mib"),
    [
        # Artifacts measured on the reference host (``du -sm`` of the artifact).
        (751_632_384, QuantizationSettings(weight_format="int4", ratio=0.8), 430),
        (751_632_384, QuantizationSettings(weight_format="int4", ratio=1.0), 387),
        (494_032_768, QuantizationSettings(weight_format="int4", ratio=0.8), 363),
        (3_075_098_624, QuantizationSettings(weight_format="int4", ratio=1.0), 1664),
    ],
)
def test_the_artifact_estimate_is_near_what_the_profile_produces(
    parameters: int, quantization: QuantizationSettings, real_mib: int
) -> None:
    """The estimate was the BF16 source size, 3.3 times the real INT4 Qwen3-0.6B artifact."""
    estimate = estimate_artifact_bytes(parameters, quantization)
    assert estimate is not None
    assert 0.75 <= estimate / (real_mib * MIB) <= 1.3
    assert estimate < parameters * 2  # well below the BF16 source


def test_without_a_parameter_count_the_estimate_stays_conservative(
    settings: Settings, hub: RecordingHub
) -> None:
    """No count, no guess: the plan keeps the source weight size."""
    assert estimate_artifact_bytes(None, QuantizationSettings()) is None
    service = build_service(settings=settings, hub=hub)

    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    assert plan.disk.estimated_artifact_bytes == safetensors_metadata().weight_bytes


def test_the_plan_uses_the_profile_estimate(settings: Settings, tmp_path: Path) -> None:
    """With a Hub parameter count the plan shows the quantized size, not the source size."""
    repo = "acme/counted"
    meta = repo_metadata(
        repo,
        [("model.safetensors", 1_503_264_768), ("config.json", 800)],
        sha="e" * 40,
        parameter_count=751_632_384,
    )
    hub = RecordingHub(repos={repo: meta}, snapshots={repo: make_snapshot(tmp_path)})
    service = build_service(settings=settings, hub=hub)

    plan = service.plan(repo, device="cpu")

    estimate = plan.disk.estimated_artifact_bytes
    assert estimate is not None
    assert estimate < meta.weight_bytes / 2
    assert "Estimated artifact          about" in render(plan)


# --- PREP-8: the auto-selection reason -------------------------------------------------


def test_auto_selection_names_the_registry_recommendation(
    settings: Settings, hub: RecordingHub
) -> None:
    """Qwen3.5 recommends the GPU; that, not "most compatible", is why it was chosen."""
    service = build_service(settings=settings, hub=hub)

    plan = service.plan(SAFETENSORS_REPO, device="auto")

    assert plan.target_device == "GPU"
    assert plan.compatibility.reasons[0] == (
        "Auto-selection chose GPU: the registry recommends it for this architecture."
    )


def test_auto_selection_among_equals_names_the_default_order(
    settings: Settings, hub: RecordingHub
) -> None:
    """With no recommendation and equal support, the tie-break is the stated default order."""
    service = build_service(settings=settings, hub=hub)

    plan = service.plan(SMOL_REPO, device="auto")

    reason = plan.compatibility.reasons[0]
    assert reason.startswith(f"Auto-selection chose {plan.target_device}")
    assert "most compatible" not in reason


# --- PREP-9: a gated repository without a token ----------------------------------------


def test_a_gated_repository_without_a_token_stops_before_the_download(
    settings: Settings, tmp_path: Path
) -> None:
    """The plan says so, and the job exits 21 before fetching 4.6 GiB it cannot read."""
    repo = "meta-llama/Llama-3.2-1B-Instruct"
    meta = repo_metadata(repo, [("model.safetensors", 2_471_645_608)], sha="f" * 40, gated=True)
    hub = RecordingHub(repos={repo: meta}, snapshots={repo: make_snapshot(tmp_path)})
    service = build_service(settings=settings, hub=hub)

    plan = service.plan(repo, device="cpu")

    assert plan.auth_missing
    assert any("gated or private" in note for note in plan.known_limitations)
    assert plan.confirmations == []
    with pytest.raises(HubAuthError) as caught:
        run_prepare(service, plan, settings)
    assert caught.value.exit_code() is ExitCode.AUTH_REQUIRED
    assert hub.downloaded == []

    hub.logged_in = True
    assert not service.plan(repo, device="cpu").auth_missing


# --- PREP-10: where the export runs ----------------------------------------------------


def test_the_plan_always_says_where_the_export_runs(settings: Settings, hub: RecordingHub) -> None:
    """Without a managed environment the plan names this interpreter and why."""
    service = build_service(settings=settings, hub=hub)

    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    assert plan.export_environment is None
    text = render(plan)
    assert "Export environment" in text
    assert "this environment" in text
    assert sys.executable in text
    assert "the installed packages meet the exporter requirements" in text
