"""Prepared-artifact domain types: submodel roles, probes, repairs, validation, manifest.

The artifact manifest is the canonical identity of a prepared model directory. It is
schema-versioned, must round-trip through JSON without loss, and must never contain
secrets (see :func:`assert_no_secrets`).
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

from yonga.domain.diagnostics import Diagnostic, FailureCategory, ValidationError, diagnostic
from yonga.domain.hardware import normalise_device
from yonga.domain.models import PipelineKind, SourceRelationship

#: Directory inside an artifact holding tool-owned metadata.
MANIFEST_DIRNAME = ".yonga"
#: File name of the artifact manifest inside :data:`MANIFEST_DIRNAME`.
MANIFEST_FILENAME = "manifest.json"
#: Current artifact manifest schema version.
MANIFEST_SCHEMA_VERSION = 1

_LENIENT = ConfigDict(extra="ignore")

_IR_SUFFIXES = (".xml", ".bin")


class SubmodelRole(str, Enum):
    """Role of one OpenVINO submodel inside a prepared artifact."""

    LANGUAGE = "language"
    TEXT_EMBEDDINGS = "text_embeddings"
    VISION_EMBEDDINGS = "vision_embeddings"
    VISION_EMBEDDINGS_MERGER = "vision_embeddings_merger"
    VISION_EMBEDDINGS_POS = "vision_embeddings_pos"
    RESAMPLER = "resampler"
    TOKENIZER = "tokenizer"
    DETOKENIZER = "detokenizer"
    OTHER = "other"

    @property
    def is_tokenizer(self) -> bool:
        """True for the tokenizer and detokenizer roles (CPU-hosted by default)."""
        return self in {SubmodelRole.TOKENIZER, SubmodelRole.DETOKENIZER}


# Exact stems produced by the OpenVINO GenAI export (docs/MODEL_PIPELINE.md section 8).
# Order matters for the fallback pass below: "merger" and "pos" must beat the plain
# "vision_embeddings" prefix, and "detokenizer" must beat "tokenizer".
_EXACT_ROLE_BY_STEM: dict[str, SubmodelRole] = {
    "openvino_language_model": SubmodelRole.LANGUAGE,
    "openvino_text_embeddings_model": SubmodelRole.TEXT_EMBEDDINGS,
    "openvino_vision_embeddings_merger_model": SubmodelRole.VISION_EMBEDDINGS_MERGER,
    "openvino_vision_embeddings_pos_model": SubmodelRole.VISION_EMBEDDINGS_POS,
    "openvino_vision_embeddings_model": SubmodelRole.VISION_EMBEDDINGS,
    "openvino_resampler_model": SubmodelRole.RESAMPLER,
    "openvino_tokenizer": SubmodelRole.TOKENIZER,
    "openvino_detokenizer": SubmodelRole.DETOKENIZER,
    "openvino_model": SubmodelRole.LANGUAGE,
}

_FALLBACK_ROLE_BY_MARKER: tuple[tuple[str, SubmodelRole], ...] = (
    ("detokenizer", SubmodelRole.DETOKENIZER),
    ("tokenizer", SubmodelRole.TOKENIZER),
    ("vision_embeddings_merger", SubmodelRole.VISION_EMBEDDINGS_MERGER),
    ("vision_embeddings_pos", SubmodelRole.VISION_EMBEDDINGS_POS),
    ("vision_embeddings", SubmodelRole.VISION_EMBEDDINGS),
    ("text_embeddings", SubmodelRole.TEXT_EMBEDDINGS),
    ("resampler", SubmodelRole.RESAMPLER),
    ("language_model", SubmodelRole.LANGUAGE),
)


def role_for_filename(name: str) -> SubmodelRole:
    """Map an exported OpenVINO file name to its submodel role.

    Accepts bare names or paths, is case-insensitive, and strips a trailing ``.xml`` or
    ``.bin`` suffix. Names carrying any other extension are not IR files, so only the
    exact export names are honoured for them. Anything unrecognised returns
    :attr:`SubmodelRole.OTHER`.
    """
    base = PurePosixPath(name.strip().replace("\\", "/")).name.lower()
    if not base:
        return SubmodelRole.OTHER
    stem = base
    for suffix in _IR_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    exact = _EXACT_ROLE_BY_STEM.get(stem)
    if exact is not None:
        return exact
    if PurePosixPath(stem).suffix:
        # e.g. "tokenizer_config.json": related to a role, but not an IR submodel.
        return SubmodelRole.OTHER
    for marker, role in _FALLBACK_ROLE_BY_MARKER:
        if marker in stem:
            return role
    return SubmodelRole.OTHER


# --- secret guard ---------------------------------------------------------------------

_SECRET_KEY_WORDS: frozenset[str] = frozenset(
    {
        "token",
        "secret",
        "password",
        "passwd",
        "pwd",
        "credential",
        "credentials",
        "apikey",
        "authorization",
        "bearer",
        "passphrase",
        "cookie",
    }
)

_SECRET_KEY_SUBSTRINGS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "apikey",
    "accesskey",
    "privatekey",
    "credential",
    "authorization",
    "passphrase",
    "bearer",
)

_WORD_SPLIT = re.compile(r"[^a-z0-9]+")
_NON_ALNUM = re.compile(r"[^a-z0-9]")


def _key_looks_secret(key: str) -> bool:
    """True when a mapping key name suggests it carries a credential."""
    lowered = key.lower()
    words = {word for word in _WORD_SPLIT.split(lowered) if word}
    if words & _SECRET_KEY_WORDS:
        return True
    compact = _NON_ALNUM.sub("", lowered)
    return any(marker in compact for marker in _SECRET_KEY_SUBSTRINGS)


def _find_secret_key(value: object, path: str = "") -> str | None:
    """Return the dotted path of the first secret-looking key, or None."""
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key)
            here = f"{path}.{key}" if path else key
            if _key_looks_secret(key):
                return here
            found = _find_secret_key(item, here)
            if found is not None:
                return found
        return None
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            found = _find_secret_key(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def assert_no_secrets(data: dict[str, object]) -> None:
    """Raise when a serialised structure contains a key whose name suggests a secret.

    Key names are checked recursively; value redaction is the job of
    ``yonga.util.redaction``. Used by the artifact store before writing a manifest.
    """
    offender = _find_secret_key(data)
    if offender is None:
        return
    raise ValidationError(
        diagnostic(
            "artifact.manifest.secret_key",
            FailureCategory.CONFIGURATION_ERROR,
            "Refusing to write a manifest containing a secret",
            f"The key {offender!r} looks like it carries a credential.",
            remediation=[
                "Remove the offending key before serialising the manifest.",
                "Route anything that may carry a token through yonga.util.redaction.",
            ],
        )
    )


def _json_safe(value: object) -> object:
    """Coerce an already JSON-dumped value into something ``json.dumps`` accepts."""
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=_by_key)}
    if isinstance(value, list | tuple | set | frozenset):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return _json_safe(value.value)
    return str(value)


def _by_key(item: tuple[object, object]) -> str:
    """Sort helper giving dictionaries a stable, string-based key order."""
    return str(item[0])


def _json_safe_mapping(data: dict[str, object]) -> dict[str, object]:
    """Apply :func:`_json_safe` to a mapping, sorting keys for stable output."""
    return {str(key): _json_safe(value) for key, value in sorted(data.items(), key=_by_key)}


def manifest_path(artifact_dir: Path) -> Path:
    """Return the manifest path for a prepared artifact directory."""
    return artifact_dir / MANIFEST_DIRNAME / MANIFEST_FILENAME


# --- validation / probing -------------------------------------------------------------


class PortDescription(BaseModel):
    """One input or output port of an OpenVINO IR."""

    model_config = _LENIENT

    name: str
    element_type: str = "unknown"
    shape: str = "unknown"


class IRDescription(BaseModel):
    """What was read from a single OpenVINO IR file."""

    model_config = _LENIENT

    path: Path
    role: SubmodelRole = SubmodelRole.OTHER
    friendly_name: str | None = None
    inputs: list[PortDescription] = Field(default_factory=list)
    outputs: list[PortDescription] = Field(default_factory=list)
    readable: bool = True
    read_error: str | None = None
    bin_path: Path | None = None
    bytes_total: int = 0


class ProbeStatus(str, Enum):
    """Outcome of a compile/load probe."""

    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"
    NOT_ATTEMPTED = "not_attempted"


class ProbeResult(BaseModel):
    """Result of compiling or loading one artifact component on one device."""

    model_config = _LENIENT

    device: str
    role: SubmodelRole | None = None
    path: Path | None = None
    status: ProbeStatus = ProbeStatus.NOT_ATTEMPTED
    duration_seconds: float | None = None
    diagnostic: Diagnostic | None = None
    properties: dict[str, str] = Field(default_factory=dict)
    message: str | None = None

    @field_validator("device")
    @classmethod
    def _normalise_device_name(cls, value: str) -> str:
        return normalise_device(value)

    @property
    def ok(self) -> bool:
        """True when the probe succeeded."""
        return self.status is ProbeStatus.OK


class DeviceProbeReport(BaseModel):
    """All probes performed for one device, plus the optional generation smoke test."""

    model_config = _LENIENT

    device: str
    status: ProbeStatus = ProbeStatus.NOT_ATTEMPTED
    submodels: list[ProbeResult] = Field(default_factory=list)
    smoke_test: ProbeResult | None = None
    diagnostic: Diagnostic | None = None
    attempted_at: datetime | None = None

    @field_validator("device")
    @classmethod
    def _normalise_device_name(cls, value: str) -> str:
        return normalise_device(value)

    @property
    def ok(self) -> bool:
        """True when the device probed successfully."""
        return self.status is ProbeStatus.OK


class RepairRecord(BaseModel):
    """A repair that was considered or applied to an artifact."""

    model_config = _LENIENT

    repair_id: str
    description: str = ""
    applied: bool = False
    safety: str = "manual_only"
    files: list[str] = Field(default_factory=list)
    before_hashes: dict[str, str] = Field(default_factory=dict)
    after_hashes: dict[str, str] = Field(default_factory=dict)
    backup_dir: str | None = None
    detail: dict[str, object] = Field(default_factory=dict)
    rule_id: str | None = None
    applied_at: datetime | None = None


class ValidationReport(BaseModel):
    """Result of validating a prepared artifact before declaring it ready."""

    model_config = _LENIENT

    ok: bool = False
    openvino_version: str | None = None
    tokenizers_version: str | None = None
    irs: list[IRDescription] = Field(default_factory=list)
    tokenizer_read_ok: bool | None = None
    detokenizer_read_ok: bool | None = None
    tokenizer_compile_ok: bool | None = None
    missing_files: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    diagnostic: Diagnostic | None = None
    validated_at: datetime | None = None

    def role(self, role: SubmodelRole) -> IRDescription | None:
        """Return the first IR description carrying ``role``, if any."""
        for ir in self.irs:
            if ir.role is role:
                return ir
        return None


class ArtifactFileRecord(BaseModel):
    """One file inside a prepared artifact, with its size and optional digest."""

    model_config = _LENIENT

    path: str
    size: int = 0
    sha256: str | None = None


class RecommendedRuntime(BaseModel):
    """Runtime settings the tool recommends for a prepared artifact."""

    model_config = _LENIENT

    device: str
    pipeline: PipelineKind = PipelineKind.UNKNOWN
    properties: dict[str, str] = Field(default_factory=dict)
    max_new_tokens: int | None = None
    generation_defaults: dict[str, float | int | bool | str] = Field(default_factory=dict)
    chat: dict[str, object] = Field(default_factory=dict)

    @field_validator("device")
    @classmethod
    def _normalise_device_name(cls, value: str) -> str:
        return normalise_device(value)


class ManifestModelRef(BaseModel):
    """A model reference as recorded in the manifest."""

    model_config = _LENIENT

    repo_id: str
    requested_revision: str | None = None
    resolved_revision: str | None = None
    relationship: SourceRelationship | None = None


class ExportRecord(BaseModel):
    """What the exporter did, recorded for reproducibility."""

    model_config = _LENIENT

    backend: str = "optimum-intel"
    profile: str
    arguments: dict[str, object] = Field(default_factory=dict)
    argv: list[str] = Field(default_factory=list)
    package_versions: dict[str, str] = Field(default_factory=dict)
    duration_seconds: float | None = None
    started_at: datetime | None = None


class ArtifactManifest(BaseModel):
    """Canonical, schema-versioned description of one prepared artifact."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    schema_version: int = MANIFEST_SCHEMA_VERSION
    artifact_id: str
    created_at: datetime
    tool_version: str
    requested_model: ManifestModelRef
    resolved_source: ManifestModelRef
    architecture: str | None = None
    architecture_rule_id: str | None = None
    model_type: str | None = None
    task: str | None = None
    pipeline: PipelineKind = PipelineKind.UNKNOWN
    export: ExportRecord
    repairs: list[RepairRecord] = Field(default_factory=list)
    validation: ValidationReport | None = None
    device_probes: dict[str, DeviceProbeReport] = Field(default_factory=dict)
    recommended_runtime: RecommendedRuntime | None = None
    source_license: str | None = None
    files: list[ArtifactFileRecord] = Field(default_factory=list)
    host: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sort_files(self) -> ArtifactManifest:
        """Keep the file inventory in a stable order so manifests diff cleanly."""
        self.files.sort(key=lambda record: record.path)
        return self

    @property
    def supported_devices(self) -> list[str]:
        """Normalised devices whose probe succeeded, in CPU/GPU/NPU-agnostic sort order."""
        return sorted(
            {normalise_device(device) for device, report in self.device_probes.items() if report.ok}
        )

    @property
    def ready(self) -> bool:
        """True when validation passed and at least one device probed successfully."""
        validated = self.validation is not None and self.validation.ok
        return validated and bool(self.supported_devices)

    def to_json(self) -> dict[str, object]:
        """Return a stable, secret-free, fully JSON-serialisable manifest mapping."""
        data = _json_safe_mapping(self.model_dump(mode="json", exclude_none=True))
        assert_no_secrets(data)
        return data

    def to_json_text(self) -> str:
        """Return the manifest as indented JSON text ready to be written to disk."""
        return json.dumps(self.to_json(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    @classmethod
    def model_validate_file(cls, path: Path) -> ArtifactManifest:
        """Read and validate a manifest file, raising a domain ValidationError on failure."""
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValidationError(
                diagnostic(
                    "artifact.manifest.unreadable",
                    FailureCategory.CONFIGURATION_ERROR,
                    "Artifact manifest could not be read",
                    f"Failed to read {path}: {exc}",
                    remediation=[
                        "Check that the artifact directory exists and is readable.",
                        "Re-run preparation to regenerate the artifact.",
                    ],
                    context={"path": str(path)},
                ),
                cause=exc,
            ) from exc
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise ValidationError(
                diagnostic(
                    "artifact.manifest.malformed_json",
                    FailureCategory.CONFIGURATION_ERROR,
                    "Artifact manifest is not valid JSON",
                    f"Failed to parse {path}: {exc}",
                    remediation=["Re-run preparation to regenerate the artifact manifest."],
                    context={"path": str(path)},
                ),
                cause=exc,
            ) from exc
        try:
            return cls.model_validate(payload)
        except PydanticValidationError as exc:
            raise ValidationError(
                diagnostic(
                    "artifact.manifest.schema_mismatch",
                    FailureCategory.CONFIGURATION_ERROR,
                    "Artifact manifest does not match the expected schema",
                    f"{path} could not be validated: {exc.error_count()} problem(s).",
                    remediation=[
                        "The artifact may have been written by a different tool version.",
                        "Re-run preparation to regenerate the artifact manifest.",
                    ],
                    raw_excerpt=str(exc)[:4000],
                    context={"path": str(path)},
                ),
                cause=exc,
            ) from exc


class ArtifactSummary(BaseModel):
    """Compact artifact description used by ``yonga list``."""

    model_config = _LENIENT

    artifact_id: str
    path: Path
    repo_id: str
    resolved_revision: str | None = None
    profile: str = ""
    recommended_device: str | None = None
    size_bytes: int = 0
    created_at: datetime
    ready: bool = False
    pipeline: PipelineKind = PipelineKind.UNKNOWN
    last_benchmark: datetime | None = None
