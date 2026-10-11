"""Unit tests for yonga.domain.plans."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from yonga.domain.compatibility import CompatibilityDecision, DeviceStatus, EvidenceLevel
from yonga.domain.hardware import FilesystemInfo, HardwareInventory
from yonga.domain.models import ModelFormat, ModelIdentity, PipelineKind, ResolvedSource
from yonga.domain.plans import (
    CONFIRMATION_DEVICE_FALLBACK,
    CONFIRMATION_LARGE_DOWNLOAD,
    CONFIRMATION_REMOTE_CODE,
    TEMP_ENV_KEYS,
    ConfirmationReason,
    ConversionProfile,
    DiskPlan,
    ExporterRequirements,
    ExportRequest,
    JobStage,
    JobState,
    PreparationPlan,
    QuantizationSettings,
    TempPlan,
)

GIB = 1024**3


def make_fs(
    path: str = "/home/user/.cache/yonga/tmp",
    *,
    fstype: str = "ext4",
    total: int = 100 * GIB,
    free: int = 50 * GIB,
    available: int | None = None,
) -> FilesystemInfo:
    return FilesystemInfo(
        path=Path(path),
        mount_point=Path("/"),
        fstype=fstype,
        total_bytes=total,
        free_bytes=free,
        available_bytes=free if available is None else available,
    )


def make_temp_plan(
    *, required: int, margin: int, available: int, free: int | None = None
) -> TempPlan:
    return TempPlan(
        job_temp_dir=Path("/home/user/.cache/yonga/tmp/job-1"),
        temp_filesystem=make_fs(free=free if free is not None else available, available=available),
        system_temp=make_fs("/tmp", fstype="tmpfs", total=16 * GIB, free=15 * GIB),
        required_bytes=required,
        safety_margin_bytes=margin,
    )


# --- TempPlan ---------------------------------------------------------------------------


def test_temp_plan_total_required_bytes() -> None:
    plan = make_temp_plan(required=20 * GIB, margin=5 * GIB, available=40 * GIB)
    assert plan.total_required_bytes == 25 * GIB


def test_temp_plan_sufficient_exactly_at_the_boundary() -> None:
    plan = make_temp_plan(required=20 * GIB, margin=5 * GIB, available=25 * GIB)
    assert plan.sufficient


def test_temp_plan_insufficient_one_byte_below_boundary() -> None:
    plan = make_temp_plan(required=20 * GIB, margin=5 * GIB, available=25 * GIB - 1)
    assert not plan.sufficient


def test_temp_plan_sufficiency_is_derived_not_trusted() -> None:
    plan = TempPlan(
        job_temp_dir=Path("/tmp/job"),
        temp_filesystem=make_fs(free=1 * GIB, available=1 * GIB),
        required_bytes=20 * GIB,
        safety_margin_bytes=5 * GIB,
        sufficient=True,
    )
    assert not plan.sufficient


def test_temp_plan_falls_back_to_free_bytes_when_available_is_unknown() -> None:
    plan = make_temp_plan(required=10 * GIB, margin=0, available=0, free=12 * GIB)
    assert plan.usable_free_bytes == 12 * GIB
    assert plan.sufficient


def test_temp_plan_env_points_every_variable_at_the_job_directory() -> None:
    plan = make_temp_plan(required=1, margin=0, available=GIB)
    env = plan.env()
    assert set(env) == {"TMPDIR", "TMP", "TEMP"}
    assert set(env) == set(TEMP_ENV_KEYS)
    assert set(env.values()) == {str(plan.job_temp_dir)}


# --- DiskPlan ---------------------------------------------------------------------------


def test_disk_plan_sufficiency_combines_temp_and_artifact() -> None:
    temp = make_temp_plan(required=20 * GIB, margin=5 * GIB, available=40 * GIB)
    disk = DiskPlan(
        temp=temp,
        artifact_filesystem=make_fs("/home/user/.local/share/yonga", free=30 * GIB),
        expected_download_bytes=18 * GIB,
        estimated_artifact_bytes=6 * GIB,
        artifact_sufficient=True,
    )
    assert disk.sufficient
    assert disk.artifact_free_bytes == 30 * GIB

    starved = DiskPlan(
        temp=make_temp_plan(required=20 * GIB, margin=5 * GIB, available=1 * GIB),
        artifact_filesystem=make_fs(free=30 * GIB),
        artifact_sufficient=True,
    )
    assert not starved.sufficient


# --- confirmations ----------------------------------------------------------------------


def test_remote_code_confirmation_is_always_security_sensitive() -> None:
    reason = ConfirmationReason(
        code=CONFIRMATION_REMOTE_CODE,
        message="This model requires trust_remote_code.",
    )
    assert reason.security_sensitive


def test_security_confirmations_filters_only_sensitive_reasons() -> None:
    plan = make_plan(
        confirmations=[
            ConfirmationReason(code=CONFIRMATION_LARGE_DOWNLOAD, message="18 GiB download"),
            ConfirmationReason(code=CONFIRMATION_REMOTE_CODE, message="remote code"),
            ConfirmationReason(
                code=CONFIRMATION_DEVICE_FALLBACK,
                message="NPU -> GPU",
                security_sensitive=False,
            ),
            ConfirmationReason(
                code="custom_policy",
                message="operator policy",
                security_sensitive=True,
            ),
        ]
    )
    assert [reason.code for reason in plan.security_confirmations] == [
        CONFIRMATION_REMOTE_CODE,
        "custom_policy",
    ]
    assert [reason.code for reason in plan.auto_approvable_confirmations] == [
        CONFIRMATION_LARGE_DOWNLOAD,
        CONFIRMATION_DEVICE_FALLBACK,
    ]


# --- ExporterRequirements ---------------------------------------------------------------


def test_exporter_requirements_problems_imply_unsatisfied() -> None:
    requirements = ExporterRequirements(
        transformers="==5.2.*",
        satisfied=True,
        problems=["transformers 5.5.4 does not satisfy ==5.2.*"],
    )
    assert not requirements.satisfied
    assert requirements.specifiers() == {"transformers": "==5.2.*"}


# --- PreparationPlan --------------------------------------------------------------------


def make_profile() -> ConversionProfile:
    return ConversionProfile(
        id="gpu-int4-asym-g128",
        device="gpu",
        summary="Default GPU INT4 profile",
        quantization=QuantizationSettings(weight_format="INT4", ratio=1.0, group_size=128),
        runtime_properties={"ATTENTION_BACKEND": "SDPA"},
        constraints={"openvino": ">=2026.2"},
    )


def test_profile_runtime_properties_apply_only_on_its_own_device() -> None:
    profile = make_profile()
    assert profile.runtime_properties_on("GPU.0") == {"ATTENTION_BACKEND": "SDPA"}
    assert profile.runtime_properties_on("gpu") == {"ATTENTION_BACKEND": "SDPA"}
    assert profile.runtime_properties_on("CPU") == {}
    assert profile.runtime_properties_on("NPU") == {}
    # A copy: the caller may extend it without changing the profile.
    profile.runtime_properties_on("GPU")["CACHE_DIR"] = "/x"
    assert profile.runtime_properties == {"ATTENTION_BACKEND": "SDPA"}


def make_plan(**overrides: object) -> PreparationPlan:
    requested = ModelIdentity(repo_id="HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive")
    source = ModelIdentity(
        repo_id="nDimensional/Qwen3.5-9B-Uncensored-Safetensors", revision="abc123"
    )
    temp = make_temp_plan(required=20 * GIB, margin=5 * GIB, available=40 * GIB)
    payload: dict[str, object] = {
        "job_id": "6f1b0f0a",
        "created_at": datetime(2026, 9, 22, 12, 30, 45, 123456, tzinfo=UTC),
        "requested": requested,
        "requested_revision": "main",
        "resolved_source": ResolvedSource(
            requested=requested,
            source=source,
            format=ModelFormat.SAFETENSORS,
            requires_confirmation=True,
        ),
        "architecture": "qwen3_5",
        "task": "image-text-to-text",
        "pipeline": PipelineKind.VLM,
        "hardware": HardwareInventory(openvino_available=True),
        "compatibility": CompatibilityDecision(
            requested_device="auto",
            selected_device="GPU",
            status=DeviceStatus.SUPPORTED,
            evidence_level=EvidenceLevel.TESTED,
            fallback_candidates=["CPU"],
        ),
        "target_device": "gpu",
        "profile": make_profile(),
        "exporter": ExporterRequirements(transformers="==5.2.*"),
        "disk": DiskPlan(
            temp=temp,
            artifact_filesystem=make_fs("/home/user/.local/share/yonga", free=30 * GIB),
            expected_download_bytes=18 * GIB,
            estimated_artifact_bytes=6 * GIB,
        ),
        "artifact_dir": Path("/home/user/.local/share/yonga/artifacts/a--b/abc123--gpu"),
        "staging_dir": Path("/home/user/.local/share/yonga/artifacts/.staging/6f1b0f0a"),
        "trust_remote_code_required": False,
        "trust_remote_code_enabled": False,
    }
    payload.update(overrides)
    return PreparationPlan.model_validate(payload)


def test_plan_normalises_target_device() -> None:
    assert make_plan(target_device="npu.0").target_device == "NPU"


def test_plan_blocked_when_remote_code_needed_but_not_enabled() -> None:
    assert make_plan(trust_remote_code_required=True).blocked
    assert not make_plan(trust_remote_code_required=True, trust_remote_code_enabled=True).blocked
    assert not make_plan().blocked


def test_plan_to_json_is_serialisable_and_round_trips() -> None:
    plan = make_plan(
        confirmations=[ConfirmationReason(code=CONFIRMATION_REMOTE_CODE, message="remote code")],
        reuse_existing_artifact=Path("/home/user/.local/share/yonga/artifacts/a--b/abc123--gpu"),
    )
    data = plan.to_json()

    text = json.dumps(data)
    assert isinstance(json.loads(text), dict)

    assert data["artifact_dir"] == str(plan.artifact_dir)
    created_at = data["created_at"]
    assert isinstance(created_at, str)
    assert datetime.fromisoformat(created_at) == plan.created_at
    assert data["pipeline"] == "vlm"
    assert data["target_device"] == "GPU"

    restored = PreparationPlan.model_validate(data)
    assert restored == plan
    assert restored.to_json() == data


def test_plan_json_contains_no_python_objects() -> None:
    data = make_plan().to_json()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                assert isinstance(key, str)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        else:
            assert value is None or isinstance(value, str | bool | int | float)

    walk(data)


# --- ExportRequest / JobState -----------------------------------------------------------


def test_export_request_model_reference_defaults_to_source_path() -> None:
    request = ExportRequest(
        source_path=Path("/snapshot/model"),
        output_path=Path("/staging/model"),
        profile=make_profile(),
    )
    assert request.resolved_model_ref == "/snapshot/model"
    assert not request.trust_remote_code

    overridden = request.model_copy(update={"model_ref": "org/repo"})
    assert overridden.resolved_model_ref == "org/repo"


def test_job_state_advance_records_history() -> None:
    created = datetime(2026, 9, 22, 8, 0, tzinfo=UTC)
    state = JobState(
        job_id="6f1b0f0a",
        created_at=created,
        updated_at=created,
        requested_model="org/repo",
    )
    assert state.stage.value == "created"

    first = datetime(2026, 9, 22, 8, 1, tzinfo=UTC)
    second = datetime(2026, 9, 22, 8, 2, tzinfo=UTC)
    state.advance(JobStage.INSPECTED, now=first)
    state.advance(JobStage.READY, now=second)

    assert state.stage is JobStage.READY
    assert state.updated_at == second
    assert state.history == [
        (first.isoformat(), "inspected"),
        (second.isoformat(), "ready"),
    ]
    assert JobStage.READY.is_terminal
    assert JobStage.FAILED.is_terminal
    assert not JobStage.EXPORTED.is_terminal
    assert json.dumps(state.to_json())


def test_quantization_weight_format_is_normalised() -> None:
    assert QuantizationSettings(weight_format=" INT4 ").weight_format == "int4"
