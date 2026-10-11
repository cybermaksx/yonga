"""Preparation planning types: profiles, preflight/disk plans, export requests and job state.

Everything here is produced *before* side effects happen (or records what a side effect
did) and must stay JSON-serialisable so a plan can be printed, stored and replayed.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from yonga.domain.compatibility import (
    CompatibilityDecision,
    DeviceStatus,
    EvidenceLevel,
    ProfileSelection,
)
from yonga.domain.diagnostics import Diagnostic
from yonga.domain.environments import EnvironmentSpec
from yonga.domain.hardware import FilesystemInfo, HardwareInventory, normalise_device
from yonga.domain.models import (
    FormatObservation,
    ModelFormat,
    ModelIdentity,
    PipelineKind,
    RepoFile,
    ResolvedSource,
)

_STRICT = ConfigDict(extra="forbid")
_LENIENT = ConfigDict(extra="ignore")

#: Stable confirmation codes used across the CLI and application services.
CONFIRMATION_THIRD_PARTY_DERIVATIVE = "third_party_derivative"
CONFIRMATION_REMOTE_CODE = "remote_code"
CONFIRMATION_LARGE_DOWNLOAD = "large_download"
CONFIRMATION_REPAIR = "repair"
CONFIRMATION_DEVICE_FALLBACK = "device_fallback"
CONFIRMATION_OVERWRITE_EXISTING = "overwrite_existing"
CONFIRMATION_MANAGED_ENVIRONMENT = "managed_export_environment"

#: Confirmation codes that ``--yes`` must never auto-approve (see SPEC.md 18.1).
SECURITY_SENSITIVE_CONFIRMATION_CODES: frozenset[str] = frozenset({CONFIRMATION_REMOTE_CODE})

#: Environment variables that carry the managed temporary directory into subprocesses.
TEMP_ENV_KEYS: tuple[str, ...] = ("TMPDIR", "TMP", "TEMP")

_QUANTIZATION_SECTION = "set it in the profile's quantization section"

#: ``optimum-cli export openvino`` options that a profile's ``export_extra_args``
#: may not pass, and why. Registry data -- built in, a signed bundle or a local
#: override -- tunes an export; it never chooses what is exported, where it goes,
#: or whether a repository's code runs.
RESERVED_EXPORT_OPTIONS: dict[str, str] = {
    "model": "yonga passes the verified local snapshot itself",
    "task": "yonga sets the task itself",
    "trust-remote-code": "remote code is enabled only by yonga's --trust-remote-code flag",
    "model-kwargs": "it reaches the model loader, which could enable remote code",
    "cache_dir": "the export reads only the local snapshot yonga passes",
    "cache-dir": "the export reads only the local snapshot yonga passes",
    "token": "registry data never carries a credential",
    "quantization-statistics-path": "it writes outside the output directory",
    "weight-format": _QUANTIZATION_SECTION,
    "sym": _QUANTIZATION_SECTION,
    "ratio": _QUANTIZATION_SECTION,
    "group-size": _QUANTIZATION_SECTION,
    "backup-precision": _QUANTIZATION_SECTION,
    "dataset": _QUANTIZATION_SECTION,
    "all-layers": _QUANTIZATION_SECTION,
    "awq": _QUANTIZATION_SECTION,
    "scale-estimation": _QUANTIZATION_SECTION,
}


def export_extra_arg_problems(arguments: Sequence[str]) -> list[str]:
    """Say why each of ``arguments`` may not be appended to an export command line.

    An empty list means the arguments are acceptable. The check follows how
    ``argparse`` reads a command line rather than comparing strings: a long
    option is refused when it is a prefix of a reserved one (``--trust-remote``
    is accepted as an abbreviation), ``--name=value`` is read as ``--name``, a
    bare ``--`` is refused because it turns what follows into positionals, and
    ``-m`` is refused with its value attached as well. Tokens without a leading
    dash are values and are left alone.
    """
    problems: list[str] = []
    for argument in arguments:
        text = str(argument)
        if text == "--":
            problems.append("'--' would turn the arguments after it into positionals")
        elif text.startswith("--"):
            name = text[2:].split("=", 1)[0]
            reserved = [option for option in RESERVED_EXPORT_OPTIONS if option.startswith(name)]
            if reserved:
                option = reserved[0]
                problems.append(f"{text!r} names --{option}: {RESERVED_EXPORT_OPTIONS[option]}")
        elif text.startswith("-m"):
            problems.append(f"{text!r} names -m: {RESERVED_EXPORT_OPTIONS['model']}")
    return problems


#: Repository paths an Optimum export or an OpenVINO IR copy never reads: other
#: runtimes' weights (ONNX, Flax, TensorFlow, Rust, Core ML), a model's native
#: checkpoint kept beside its Transformers one (``original/``, ``consolidated*``),
#: and training leftovers. ``fnmatch`` patterns, matched the way
#: ``huggingface_hub`` matches ``ignore_patterns`` (``*`` crosses ``/``).
EXPORT_IRRELEVANT_PATTERNS: tuple[str, ...] = (
    "*.onnx",
    "*.onnx_data",
    "*.onnx.data",
    "onnx/*",
    "*.msgpack",
    "*.h5",
    "*.tflite",
    "*.ot",
    "*.mlmodel",
    "*.mlpackage/*",
    "coreml/*",
    "original/*",
    "consolidated*.pth",
    "runs/*",
    "*tfevents*",
    "training_args.bin",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state*.pth",
    "*.gguf",
)

#: Bits one stored weight takes, per NNCF ``weight_format`` (and backup precision).
_WEIGHT_BITS: dict[str, int] = {
    "int4": 4,
    "nf4": 4,
    "mxfp4": 4,
    "fp4": 4,
    "e2m1": 4,
    "int8": 8,
    "int8_asym": 8,
    "int8_sym": 8,
    "mxfp8": 8,
    "fp8": 8,
    "f8e4m3": 8,
    "f8e5m2": 8,
    "fp16": 16,
    "bf16": 16,
    "none": 16,
    "fp32": 32,
}

#: How much larger a prepared artifact is than its parameters times their bits.
#: Embeddings and the output layer stay at INT8, group scales and zero points
#: add a little, and tokenizer IR and configs ride along. Measured on the
#: reference host: Qwen3-0.6B, Qwen2.5-0.5B, SmolLM3-3B, Qwen2.5-VL-3B and
#: Qwen3.5-0.8B artifacts came out at 0.83 to 1.85 times the bare figure,
#: most of them between 1.0 and 1.4 (the VLMs, whose vision tower is not
#: compressed like the language model, at the top).
ARTIFACT_SIZE_OVERHEAD: float = 1.2


def download_ignore_patterns(
    files: Sequence[RepoFile],
    formats: Sequence[FormatObservation],
    *,
    source_format: ModelFormat,
    remote_code: bool = False,
) -> list[str]:
    """Patterns for the repository files a conversion of ``source_format`` never reads.

    A snapshot of the whole repository can be several times the model: the
    SmolLM2 repositories carry eight ONNX variants and TensorBoard runs, Llama
    ships its native ``original/`` checkpoint beside the Transformers one, and
    many repositories publish both ``.safetensors`` and ``.bin`` weights. Only
    weights in the format the job reads are downloaded; config, tokenizer,
    chat-template and processor files are never filtered.

    Other weight formats are excluded by their exact paths, and only once the
    chosen format is complete, so a ``.bin`` that is the model's only weight
    file is never dropped. With remote code enabled nothing is filtered: the
    repository's own code may open any file it ships.
    """
    if remote_code:
        return []
    patterns = list(EXPORT_IRRELEVANT_PATTERNS)
    chosen = [observation for observation in formats if observation.format is source_format]
    if not chosen or not any(observation.complete for observation in chosen):
        return patterns
    keep = {path for observation in chosen for path in observation.files}
    for observation in formats:
        if observation.format in (source_format, ModelFormat.GGUF):
            continue
        patterns.extend(_glob_escape(path) for path in observation.files if path not in keep)
    index_suffix = _OTHER_INDEX_SUFFIXES.get(source_format, ())
    patterns.extend(
        _glob_escape(entry.path) for entry in files if entry.path.endswith(index_suffix)
    )
    return list(dict.fromkeys(patterns))


#: Shard index documents of the weight formats a job does not read, by the format it does.
_OTHER_INDEX_SUFFIXES: dict[ModelFormat, tuple[str, ...]] = {
    ModelFormat.SAFETENSORS: (".bin.index.json",),
    ModelFormat.PYTORCH_BIN: (".safetensors.index.json",),
    ModelFormat.OPENVINO_IR: (".safetensors.index.json", ".bin.index.json"),
}


def _glob_escape(path: str) -> str:
    """Escape ``fnmatch`` metacharacters so ``path`` matches only itself."""
    return "".join(f"[{char}]" if char in "*?[" else char for char in path)


def is_ignored(path: str, patterns: Sequence[str]) -> bool:
    """Whether ``huggingface_hub`` would skip ``path`` for these ``ignore_patterns``."""
    expanded = [pattern + "*" if pattern.endswith("/") else pattern for pattern in patterns]
    return any(fnmatch(path, pattern) for pattern in expanded)


def bytes_to_download(files: Sequence[RepoFile], patterns: Sequence[str]) -> int:
    """Bytes a download of ``files`` fetches when ``patterns`` are ignored."""
    return sum(entry.size or 0 for entry in files if not is_ignored(entry.path, patterns))


def estimate_artifact_bytes(
    parameter_count: int | None, quantization: QuantizationSettings
) -> int | None:
    """Estimate the prepared artifact's size from what the profile stores per weight.

    ``ratio`` of the weights take the profile's format and the rest its backup
    precision (NNCF's default backup is INT8), scaled by
    :data:`ARTIFACT_SIZE_OVERHEAD`. ``None`` when the parameter count or the
    format is unknown: then the caller falls back to the source weight size,
    which over-reserves rather than under-reserves.
    """
    if parameter_count is None or parameter_count <= 0:
        return None
    primary = _WEIGHT_BITS.get(quantization.weight_format)
    if primary is None:
        return None
    ratio = 1.0 if quantization.ratio is None else quantization.ratio
    backup_name = (quantization.backup_precision or "int8_asym").strip().lower()
    backup = _WEIGHT_BITS.get(backup_name, 8)
    bits = ratio * primary + (1.0 - ratio) * backup
    return int(parameter_count * bits / 8 * ARTIFACT_SIZE_OVERHEAD)


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

    def runtime_properties_on(self, device: str) -> dict[str, str]:
        """The runtime properties to apply on ``device``: this profile's own, or none.

        They were chosen for the profile's own device and only that device's
        plugin is known to accept them: the NPU's ``MAX_PROMPT_LEN`` makes the CPU
        and GPU plugins refuse the pipeline. A profile used elsewhere -- an
        explicit cross-device ``--profile``, or auto mode falling back from the
        NPU -- runs with the device's own defaults.
        """
        if normalise_device(device) != self.device:
            return {}
        return dict(self.runtime_properties)


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


class DeclaredRequirement(BaseModel):
    """A package range the exporter itself declares for an architecture."""

    model_config = ConfigDict(frozen=True)

    transformers: str
    """A PEP 440 specifier, e.g. ``>=4.57,<=5.0``."""
    source: str
    """Where it was read from, for the plan and the manifest."""


class ReusableExport(BaseModel):
    """A failed job's completed export that an identical preparation can adopt."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    """The job that ran the export."""
    path: Path
    """Its retained staging directory."""
    completed_at: datetime
    fingerprint: str


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
    export_environment: EnvironmentSpec | None = None
    """The managed environment the export will run in, when the installed packages
    cannot meet the architecture's exporter requirements."""
    export_environment_ready: bool = False
    """Whether :attr:`export_environment` is already built (no download needed)."""
    reusable_export: ReusableExport | None = None
    """A completed export, left by a failed job, that this plan will adopt instead
    of exporting again. ``None`` when there is none or reuse was declined."""
    profile_selection: ProfileSelection | None = None
    """Why :attr:`profile` was chosen; ``applies`` is False when no profile decides
    the quantization (a GGUF import or an OpenVINO copy)."""
    download_ignore_patterns: list[str] = Field(default_factory=list)
    """Repository paths the download skips, because this job never reads them.

    :attr:`DiskPlan.expected_download_bytes` is computed with the same patterns,
    so the plan announces what the download will actually fetch."""
    auth_missing: bool = False
    """The source repository is gated or private and no Hugging Face token was
    found, so its download is bound to be refused."""
    export_blocked: Diagnostic | None = None
    """A known exporter defect this export is certain to hit with the interpreter
    and packages that will run it (an issue rule's ``blocks_export``). The job
    stops before any question is asked unless ``--ignore-exporter-requirements``
    overrides it, as it overrides an unmet exporter pin."""

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
    def reuses_artifact(self) -> bool:
        """True when a ready artifact is returned as-is: nothing is downloaded or exported."""
        return self.reuse_existing_artifact is not None and not self.force

    @property
    def runtime_missing(self) -> bool:
        """True when the OpenVINO runtime cannot be imported in this environment.

        Every path -- export, GGUF import, IR copy -- validates and probes the
        result with OpenVINO, so without it a preparation cannot finish at all.
        """
        return not self.hardware.openvino_available and not self.hardware.devices

    @property
    def device_known_unsupported(self) -> bool:
        """True when the explicitly requested device is one the registry rules out.

        Only an ``unsupported`` verdict backed by official or tested evidence
        counts: weaker knowledge leaves the decision to the runtime probe. An
        automatic choice never selects such a device in the first place.
        """
        decision = self.compatibility
        return (
            decision.explicit_request
            and decision.status is DeviceStatus.UNSUPPORTED
            and decision.evidence_level.rank >= EvidenceLevel.TESTED.rank
        )

    @property
    def refused_before_work(self) -> bool:
        """True when the job will stop before preflight, whatever is confirmed."""
        return self.runtime_missing or self.device_known_unsupported or self.auth_missing

    @property
    def exporter_requirements_block(self) -> bool:
        """True when an unmet exporter pin really will stop this job at preflight.

        Unmet pins are not always fatal. A repository that already publishes
        OpenVINO IR is copied and validated, never converted, so the exporter is
        not on that job's path at all and its version pins decide nothing.

        The preflight gate and the plan panel both read this one property, so
        the panel cannot announce a block the service will not perform. A plan
        that reuses a ready artifact, or that stops before preflight for another
        reason, never reaches this gate.
        """
        return (
            bool(self.exporter.problems)
            and not self.resolved_source.already_openvino
            and self.resolved_source.gguf_file is None
            and self.export_environment is None
            and not self.reuses_artifact
            and not self.refused_before_work
        )

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
    interpreter: Path | None = None
    """Run the export with this Python -- a managed export environment's -- instead
    of the one yonga runs in. ``None`` means yonga's own interpreter."""
    environment_versions: dict[str, str] = Field(default_factory=dict)
    """Package versions installed in :attr:`interpreter`'s environment, recorded as
    the export's package versions so the manifest names what actually ran."""

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
    """States of the job state machine (docs/ARCHITECTURE.md section 5).

    ``ready`` is the end of a successful preparation; ``completed`` is the end
    of any other job that ran to its end (a probe or a benchmark), which has
    no artifact to make ready. ``cancelled`` is a job the user stopped, kept
    apart from ``failed`` so that a Ctrl-C does not read as a defect.
    """

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
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """True for stages from which the job does not continue."""
        return self in _TERMINAL_STAGES

    @property
    def succeeded(self) -> bool:
        """True for the terminal stages a job reaches by finishing its work."""
        return self in {JobStage.READY, JobStage.COMPLETED}


_TERMINAL_STAGES: frozenset[JobStage] = frozenset(
    {JobStage.READY, JobStage.COMPLETED, JobStage.FAILED, JobStage.CANCELLED}
)


class JobState(BaseModel):
    """Small persisted record describing an in-flight or finished job."""

    model_config = _LENIENT

    job_id: str
    stage: JobStage = JobStage.CREATED
    created_at: datetime
    updated_at: datetime
    requested_model: str
    command: str | None = None
    """The command that opened the job (``prepare``, ``probe``, ``bench``), when known."""
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


class JobStatus(str, Enum):
    """How a job ended, as ``yonga logs`` shows it."""

    OK = "ok"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RUNNING = "running"
    UNFINISHED = "unfinished"
    """Not terminal, and silent for longer than a live job ever is: the process died."""
    UNKNOWN = "unknown"
    """No readable state document: a log written before job states existed."""


class JobSummary(BaseModel):
    """One line of the job index: which job it was and how it ended."""

    model_config = _LENIENT

    job_id: str
    log_path: Path
    size_bytes: int = 0
    started_at: datetime | None = None
    ended_at: datetime | None = None
    command: str | None = None
    model: str | None = None
    device: str | None = None
    stage: JobStage | None = None
    status: JobStatus = JobStatus.UNKNOWN
    code: str | None = None
    """The diagnostic code of a failed or cancelled job."""

    @property
    def duration_seconds(self) -> float | None:
        """How long the job ran, once it has ended."""
        if self.started_at is None or self.ended_at is None:
            return None
        return max(0.0, (self.ended_at - self.started_at).total_seconds())

    def to_json(self) -> dict[str, object]:
        """Return a JSON-serialisable representation, duration included."""
        payload = _json_safe_mapping(self.model_dump(mode="json"))
        payload["duration_seconds"] = self.duration_seconds
        return payload
