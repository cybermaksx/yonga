"""Preparation planning types: profiles, preflight/disk plans, export requests and job state.

Everything here is produced *before* side effects happen (or records what a side effect
did) and must stay JSON-serialisable so a plan can be printed, stored and replayed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from yonga.domain.compatibility import CompatibilityDecision
from yonga.domain.diagnostics import Diagnostic
from yonga.domain.hardware import FilesystemInfo, HardwareInventory, normalise_device
from yonga.domain.models import ModelIdentity, PipelineKind, ResolvedSource

_STRICT = ConfigDict(extra="forbid")
_LENIENT = ConfigDict(extra="ignore")

#: Stable confirmation codes used across the CLI and application services.
CONFIRMATION_THIRD_PARTY_DERIVATIVE = "third_party_derivative"
CONFIRMATION_REMOTE_CODE = "remote_code"
CONFIRMATION_LARGE_DOWNLOAD = "large_download"
CONFIRMATION_REPAIR = "repair"
CONFIRMATION_DEVICE_FALLBACK = "device_fallback"
CONFIRMATION_OVERWRITE_EXISTING = "overwrite_existing"

#: Confirmation codes that ``--yes`` must never auto-approve (see SPEC.md 18.1).
SECURITY_SENSITIVE_CONFIRMATION_CODES: frozenset[str] = frozenset({CONFIRMATION_REMOTE_CODE})

#: Environment variables that carry the managed temporary directory into subprocesses.
TEMP_ENV_KEYS: tuple[str, ...] = ("TMPDIR", "TMP", "TEMP")


def _json_safe(value: object) -> object:
    """Coerce an already JSON-dumped value into something ``json.dumps`` accepts."""
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return _json_safe(value.value)
    return str(value)


def _json_safe_mapping(data: dict[str, object]) -> dict[str, object]:
    """Apply :func:`_json_safe` to every value of a mapping."""
    return {str(key): _json_safe(value) for key, value in data.items()}


class QuantizationSettings(BaseModel):
    """Weight-compression settings mirroring a registry profile ``quantization`` block."""

    model_config = _LENIENT

    weight_format: str = "int4"
    symmetric: bool | None = None
    ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    group_size: int | None = None
    dataset: str | None = None
    awq: bool = False
    scale_estimation: bool = False
    all_layers: bool = False
    backup_precision: str | None = None

    @field_validator("weight_format")
    @classmethod
    def _normalise_weight_format(cls, value: str) -> str:
        return value.strip().lower()


class ConversionProfile(BaseModel):
    """Device-specific conversion and runtime settings resolved from the registry."""

    model_config = _LENIENT

    id: str
    device: str
    summary: str = ""
    quantization: QuantizationSettings
    runtime_properties: dict[str, str] = Field(default_factory=dict)
    export_extra_args: list[str] = Field(default_factory=list)
    constraints: dict[str, str] = Field(default_factory=dict)
    priority: int = 100
    rationale: str | None = None
    sources: list[dict[str, str]] = Field(default_factory=list)

    @field_validator("device")
    @classmethod
    def _normalise_device_name(cls, value: str) -> str:
        return normalise_device(value)


class ConfirmationReason(BaseModel):
    """One thing the user must confirm before a preparation may proceed."""

    model_config = _STRICT

    code: str
    message: str
    security_sensitive: bool = False

    @model_validator(mode="after")
    def _force_security_sensitive(self) -> ConfirmationReason:
        """Security-relevant codes are always security sensitive, never auto-approvable."""
        if self.code in SECURITY_SENSITIVE_CONFIRMATION_CODES and not self.security_sensitive:
            self.security_sensitive = True
        return self


class TempPlan(BaseModel):
    """Managed temporary-storage plan for one conversion job."""

    model_config = _STRICT

    job_temp_dir: Path
    temp_filesystem: FilesystemInfo
    system_temp: FilesystemInfo | None = None
    required_bytes: int = 0
    safety_margin_bytes: int = 0
    sufficient: bool = False
    system_temp_would_fail: bool = False
    notes: list[str] = Field(default_factory=list)

    @property
    def total_required_bytes(self) -> int:
        """Required working space plus the configured safety margin."""
        return self.required_bytes + self.safety_margin_bytes

    @property
    def usable_free_bytes(self) -> int:
        """Free space on the chosen temp filesystem, preferring the non-reserved figure."""
        return _usable_free_bytes(self.temp_filesystem)

    @model_validator(mode="after")
    def _recompute_sufficiency(self) -> TempPlan:
        """``sufficient`` is derived: free space must cover requirement plus margin."""
        self.sufficient = self.usable_free_bytes >= self.total_required_bytes
        return self

    def env(self) -> dict[str, str]:
        """Environment overrides pointing every temp variable at the managed directory."""
        target = str(self.job_temp_dir)
        return dict.fromkeys(TEMP_ENV_KEYS, target)


def _usable_free_bytes(filesystem: FilesystemInfo) -> int:
    """Return available bytes for an unprivileged writer, falling back to free bytes."""
    if filesystem.available_bytes > 0:
        return filesystem.available_bytes
    return filesystem.free_bytes


class DiskPlan(BaseModel):
    """Disk preflight result covering temp, download and final artifact space."""

    model_config = _STRICT

    temp: TempPlan
    artifact_filesystem: FilesystemInfo
    expected_download_bytes: int = 0
    estimated_artifact_bytes: int | None = None
    artifact_sufficient: bool = True
    notes: list[str] = Field(default_factory=list)

    @property
    def artifact_free_bytes(self) -> int:
        """Free space on the filesystem holding the final artifact directory."""
        return _usable_free_bytes(self.artifact_filesystem)

    @property
    def sufficient(self) -> bool:
        """True when both the temp and the artifact filesystems have enough room."""
        return self.temp.sufficient and self.artifact_sufficient


class ExporterRequirements(BaseModel):
    """Exporter-side package requirements for one architecture and their status."""

    model_config = _LENIENT

    transformers: str | None = None
    optimum_intel: str | None = None
    openvino: str | None = None
    nncf: str | None = None
    satisfied: bool = True
    problems: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _problems_imply_unsatisfied(self) -> ExporterRequirements:
        """Recorded problems always mean the requirements are not satisfied."""
        if self.problems:
            self.satisfied = False
        return self

    def specifiers(self) -> dict[str, str]:
        """Return the declared version specifiers keyed by distribution name."""
        declared = {
            "transformers": self.transformers,
            "optimum-intel": self.optimum_intel,
            "openvino": self.openvino,
            "nncf": self.nncf,
        }
        return {name: spec for name, spec in declared.items() if spec is not None}


class PreparationPlan(BaseModel):
    """Everything decided before a preparation job starts doing expensive work."""

    model_config = _LENIENT

    job_id: str
    created_at: datetime
    requested: ModelIdentity
    requested_revision: str | None = None
    resolved_source: ResolvedSource
    architecture: str | None = None
    """What this model *is* — ``llama``, ``qwen3_5`` — never a registry rule id.

    Same meaning as ``ArtifactManifest.architecture``, so the pre-flight panel
    and the manifest it becomes cannot disagree. The id of the rule that matched
    is ``architecture_rule_id``.
    """

    architecture_rule_id: str | None = None
    """The registry architecture rule that matched, when one did."""

    task: str | None = None
    pipeline: PipelineKind = PipelineKind.UNKNOWN
    hardware: HardwareInventory
    compatibility: CompatibilityDecision
    target_device: str
    profile: ConversionProfile
    exporter: ExporterRequirements
    disk: DiskPlan
    artifact_dir: Path
    staging_dir: Path
    trust_remote_code_required: bool = False
    trust_remote_code_enabled: bool = False
    expected_repairs: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)
    confirmations: list[ConfirmationReason] = Field(default_factory=list)
    reuse_existing_artifact: Path | None = None
    force: bool = False

    @field_validator("target_device")
    @classmethod
    def _normalise_target_device(cls, value: str) -> str:
        return normalise_device(value)

    @property
    def security_confirmations(self) -> list[ConfirmationReason]:
        """Confirmations that ``--yes`` must never auto-approve."""
        return [reason for reason in self.confirmations if reason.security_sensitive]

    @property
    def auto_approvable_confirmations(self) -> list[ConfirmationReason]:
        """Confirmations that ``--yes`` may approve on the user's behalf."""
        return [reason for reason in self.confirmations if not reason.security_sensitive]

    @property
    def blocked(self) -> bool:
        """True when the plan cannot run as-is (remote code needed but not enabled)."""
        return self.trust_remote_code_required and not self.trust_remote_code_enabled

    @property
    def exporter_requirements_block(self) -> bool:
        """True when an unmet exporter pin really will stop this job at preflight.

        Unmet pins are not always fatal. A repository that already publishes
        OpenVINO IR is copied and validated, never converted, so the exporter is
        not on that job's path at all and its version pins decide nothing.

        The preflight gate and the plan panel both read this one property, so
        the panel cannot announce a block the service will not perform.
        """
        return bool(self.exporter.problems) and not self.resolved_source.already_openvino

    def to_json(self) -> dict[str, object]:
        """Return a fully JSON-serialisable representation of the plan."""
        return _json_safe_mapping(self.model_dump(mode="json"))


class ExportRequest(BaseModel):
    """A single conversion/quantization request handed to an exporter adapter."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    source_path: Path
    output_path: Path
    task: str | None = None
    profile: ConversionProfile
    trust_remote_code: bool = False
    extra_env: dict[str, str] = Field(default_factory=dict)
    model_ref: str | None = None

    @property
    def resolved_model_ref(self) -> str:
        """What the exporter should pass as its model reference (path unless overridden)."""
        return self.model_ref if self.model_ref is not None else str(self.source_path)


class ExportResult(BaseModel):
    """Outcome of one exporter invocation; all captured text is already redacted."""

    model_config = _LENIENT

    success: bool
    argv: list[str] = Field(default_factory=list)
    env_overrides: dict[str, str] = Field(default_factory=dict)
    returncode: int = 0
    duration_seconds: float = 0.0
    stdout_tail: list[str] = Field(default_factory=list)
    stderr_tail: list[str] = Field(default_factory=list)
    log_path: Path | None = None
    package_versions: dict[str, str] = Field(default_factory=dict)
    diagnostic: Diagnostic | None = None


class JobStage(str, Enum):
    """States of the preparation state machine (docs/ARCHITECTURE.md section 5)."""

    CREATED = "created"
    INSPECTED = "inspected"
    SOURCE_RESOLVED = "source_resolved"
    PREFLIGHT_OK = "preflight_ok"
    DOWNLOADED = "downloaded"
    EXPORTED = "exported"
    VALIDATING = "validating"
    REPAIRED = "repaired"
    PROBED = "probed"
    READY = "ready"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """True for stages from which the job does not continue."""
        return self in {JobStage.READY, JobStage.FAILED}


class JobState(BaseModel):
    """Small persisted record describing an in-flight or finished preparation job."""

    model_config = _LENIENT

    job_id: str
    stage: JobStage = JobStage.CREATED
    created_at: datetime
    updated_at: datetime
    requested_model: str
    target_device: str | None = None
    profile_id: str | None = None
    artifact_dir: Path | None = None
    diagnostic: Diagnostic | None = None
    history: list[tuple[str, str]] = Field(default_factory=list)

    def advance(self, stage: JobStage, *, now: datetime | None = None) -> None:
        """Move the job to ``stage`` and append it to the history with a timestamp."""
        moment = now if now is not None else datetime.now(tz=UTC)
        self.stage = stage
        self.updated_at = moment
        self.history.append((moment.isoformat(), stage.value))

    def to_json(self) -> dict[str, object]:
        """Return a fully JSON-serialisable representation of the job state."""
        return _json_safe_mapping(self.model_dump(mode="json"))
