"""Structured diagnostics, failure categories, typed errors and exit codes.

Every user-visible failure in yonga is a :class:`Diagnostic`: a stable machine
readable code, a category, human text and remediation hints. The category maps
onto a stable process exit code through :data:`CATEGORY_EXIT_CODES`, which is the
single source of truth for ``docs/CLI_SPEC.md`` section 15.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from enum import Enum, IntEnum
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from yonga.domain.hardware import normalise_device

#: Hard ceiling for retained raw error text inside a diagnostic.
RAW_EXCERPT_MAX_CHARS: int = 4000

_TRUNCATION_MARKER = "\n[... truncated ...]"


class Severity(str, Enum):
    """How serious a diagnostic or health check is."""

    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    UNKNOWN = "unknown"


class FailureCategory(str, Enum):
    """Machine-readable failure classes produced by the classification pipeline."""

    DEVICE_UNAVAILABLE = "device_unavailable"
    DRIVER_MISSING = "driver_missing"
    PERMISSION_DENIED = "permission_denied"
    COMPILER_MISSING = "compiler_missing"
    COMPILER_UNSUPPORTED_GRAPH = "compiler_unsupported_graph"
    OUT_OF_MEMORY = "out_of_memory"
    INSUFFICIENT_DISK = "insufficient_disk"
    UNSUPPORTED_IR_TYPE = "unsupported_ir_type"
    TOKENIZER_IR_INCOMPATIBLE = "tokenizer_ir_incompatible"
    PACKAGE_ABI_MISMATCH = "package_abi_mismatch"
    SOURCE_FORMAT_UNSUPPORTED = "source_format_unsupported"
    REMOTE_CODE_REQUIRED = "remote_code_required"
    GATED_MODEL_AUTH_REQUIRED = "gated_model_auth_required"
    DOWNLOAD_FAILURE = "download_failure"
    CONFIGURATION_ERROR = "configuration_error"
    REGISTRY_ERROR = "registry_error"
    USER_CANCELLED = "user_cancelled"
    UNKNOWN_RUNTIME_ERROR = "unknown_runtime_error"


class ExitCode(IntEnum):
    """Stable process exit codes (``docs/CLI_SPEC.md`` section 15)."""

    SUCCESS = 0
    WARNINGS = 2
    ENVIRONMENT_UNUSABLE = 10
    SOURCE_RESOLUTION = 20
    AUTH_REQUIRED = 21
    REMOTE_CODE_REQUIRED = 22
    PREFLIGHT = 30
    CONVERSION = 40
    VALIDATION = 41
    REPAIR_DENIED = 42
    DEVICE_UNSUPPORTED = 50
    COMPILE = 51
    GENERATION = 52
    CANCELLED = 60
    REGISTRY = 70
    INTERNAL = 99


#: Complete, explicit mapping from failure category to process exit code.
#:
#: Every member of :class:`FailureCategory` must appear here; the unit tests
#: assert that. Exit codes 40 (conversion), 42 (repair denied) and 52
#: (generation) describe *where* a job died rather than *why*, so no category
#: maps to them: the typed errors below supply them via ``fallback_exit_code``
#: when the category is merely ``unknown_runtime_error`` (and ``bench`` returns
#: 52 itself when no run produced a measurement). Exit code 41
#: (validation) is both: it is the category code for ``unsupported_ir_type`` and
#: ``tokenizer_ir_incompatible``, and the fallback of :class:`ValidationError`.
CATEGORY_EXIT_CODES: Mapping[FailureCategory, ExitCode] = MappingProxyType(
    {
        FailureCategory.DEVICE_UNAVAILABLE: ExitCode.DEVICE_UNSUPPORTED,
        FailureCategory.DRIVER_MISSING: ExitCode.ENVIRONMENT_UNUSABLE,
        FailureCategory.PERMISSION_DENIED: ExitCode.ENVIRONMENT_UNUSABLE,
        FailureCategory.COMPILER_MISSING: ExitCode.ENVIRONMENT_UNUSABLE,
        FailureCategory.COMPILER_UNSUPPORTED_GRAPH: ExitCode.COMPILE,
        FailureCategory.OUT_OF_MEMORY: ExitCode.PREFLIGHT,
        FailureCategory.INSUFFICIENT_DISK: ExitCode.PREFLIGHT,
        FailureCategory.UNSUPPORTED_IR_TYPE: ExitCode.VALIDATION,
        FailureCategory.TOKENIZER_IR_INCOMPATIBLE: ExitCode.VALIDATION,
        FailureCategory.PACKAGE_ABI_MISMATCH: ExitCode.ENVIRONMENT_UNUSABLE,
        FailureCategory.SOURCE_FORMAT_UNSUPPORTED: ExitCode.SOURCE_RESOLUTION,
        FailureCategory.REMOTE_CODE_REQUIRED: ExitCode.REMOTE_CODE_REQUIRED,
        FailureCategory.GATED_MODEL_AUTH_REQUIRED: ExitCode.AUTH_REQUIRED,
        FailureCategory.DOWNLOAD_FAILURE: ExitCode.SOURCE_RESOLUTION,
        FailureCategory.CONFIGURATION_ERROR: ExitCode.REGISTRY,
        FailureCategory.REGISTRY_ERROR: ExitCode.REGISTRY,
        FailureCategory.USER_CANCELLED: ExitCode.CANCELLED,
        FailureCategory.UNKNOWN_RUNTIME_ERROR: ExitCode.INTERNAL,
    }
)

#: Ordering used to fold several severities into the worst one.
SEVERITY_RANK: Mapping[Severity, int] = MappingProxyType(
    {
        Severity.OK: 0,
        Severity.INFO: 1,
        Severity.UNKNOWN: 2,
        Severity.WARNING: 3,
        Severity.ERROR: 4,
    }
)


def exit_code_for_category(category: FailureCategory) -> ExitCode:
    """Return the stable exit code reserved for *category*."""
    return CATEGORY_EXIT_CODES[category]


def worst_severity(values: Iterable[Severity]) -> Severity:
    """Return the most serious severity in *values* (``Severity.OK`` when empty)."""
    worst = Severity.OK
    for value in values:
        if SEVERITY_RANK[value] > SEVERITY_RANK[worst]:
            worst = value
    return worst


class Diagnostic(BaseModel):
    """A classified, explainable failure or observation."""

    model_config = ConfigDict(extra="ignore")

    code: str
    category: FailureCategory
    severity: Severity = Severity.ERROR
    title: str
    message: str
    remediation: list[str] = Field(default_factory=list)
    fallback_devices: list[str] = Field(default_factory=list)
    environment_broken: bool = False
    retry_same_device: bool = False
    defer_to_pipeline: bool = False
    matched_rule_id: str | None = None
    raw_excerpt: str | None = None
    log_path: Path | None = None
    job_id: str | None = None
    context: dict[str, str] = Field(default_factory=dict)

    @field_validator("code")
    @classmethod
    def _check_code(cls, value: str) -> str:
        """A diagnostic without a stable code cannot be matched or documented."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("diagnostic code must not be empty")
        return cleaned

    @field_validator("fallback_devices")
    @classmethod
    def _normalise_fallback_devices(cls, value: list[str]) -> list[str]:
        """Canonicalise device names and drop blanks/duplicates, keeping order."""
        names: list[str] = []
        for raw in value:
            name = normalise_device(raw)
            if name and name not in names:
                names.append(name)
        return names

    @field_validator("raw_excerpt")
    @classmethod
    def _truncate_raw_excerpt(cls, value: str | None) -> str | None:
        """Keep retained raw text bounded; full text always stays in the job log."""
        if value is None:
            return None
        if len(value) <= RAW_EXCERPT_MAX_CHARS:
            return value
        keep = RAW_EXCERPT_MAX_CHARS - len(_TRUNCATION_MARKER)
        return value[:keep] + _TRUNCATION_MARKER

    def exit_code(self) -> ExitCode:
        """Return the process exit code for this diagnostic's category."""
        return CATEGORY_EXIT_CODES[self.category]

    def effective_exit_code(self) -> ExitCode:
        """Exit code that also respects severity.

        A diagnostic carried as information or as a warning is not a failure:
        those map to :data:`ExitCode.SUCCESS` and :data:`ExitCode.WARNINGS`
        respectively. Everything else uses the category table.
        """
        if self.severity in (Severity.OK, Severity.INFO):
            return ExitCode.SUCCESS
        if self.severity is Severity.WARNING:
            return ExitCode.WARNINGS
        return self.exit_code()

    def to_json(self) -> dict[str, object]:
        """Return a JSON-safe dictionary (enums as values, paths as strings)."""
        return dict(self.model_dump(mode="json"))


def diagnostic(
    code: str,
    category: FailureCategory,
    title: str,
    message: str,
    **fields: object,
) -> Diagnostic:
    """Build a :class:`Diagnostic`; extra keyword arguments set optional fields."""
    payload: dict[str, object] = {
        "code": code,
        "category": category,
        "title": title,
        "message": message,
    }
    payload.update(fields)
    return Diagnostic.model_validate(payload)


#: Upper-case words a remediation step uses for what it cannot know when it is
#: written: ``MODEL`` is the repository the job is about, ``JOB_ID`` the job.
#: A neighbouring letter, digit, underscore or hyphen makes it another word.
_PLACEHOLDER_PATTERN = re.compile(r"(?<![\w-])(MODEL|JOB_ID)(?![\w-])")

#: The command a remediation step ends with: ``Do this: yonga prepare X --device gpu``.
_TRAILING_COMMAND = re.compile(r":\s+`?(yonga\s[^`]*?)`?\.?\s*$")


def fill_placeholders(item: Diagnostic) -> Diagnostic:
    """Put the known repository and job id into ``item``'s remediation, once each.

    Registry remediation is data written before any job exists, so it says
    ``yonga prepare MODEL`` and ``yonga logs JOB_ID --raw``. Wherever the
    diagnostic knows the value -- ``job_id``, or the ``repo_id`` in its context
    -- the placeholder is replaced, so the step can be copied as it stands; an
    unknown value leaves the placeholder, which still reads as one.

    Filling can make two steps say the same thing: a registry rule's "run this
    model on the GPU instead: yonga prepare MODEL --device gpu" and ``prepare``'s
    own "Prepare for GPU instead: yonga prepare <repo> --device gpu". A step that
    repeats an earlier one, or ends in the same command as an earlier one, is
    dropped. Safe to apply more than once.
    """
    model = item.context.get("repo_id") or item.context.get("model")
    values = {name: value for name, value in (("MODEL", model), ("JOB_ID", item.job_id)) if value}

    def substitute(match: re.Match[str]) -> str:
        return values.get(match.group(1), match.group(1))

    steps: list[str] = []
    commands: set[str] = set()
    for step in item.remediation:
        filled = _PLACEHOLDER_PATTERN.sub(substitute, step) if values else step
        command = _TRAILING_COMMAND.search(filled.strip())
        key = " ".join(command.group(1).split()) if command is not None else None
        if filled in steps or (key is not None and key in commands):
            continue
        steps.append(filled)
        if key is not None:
            commands.add(key)
    if steps == item.remediation:
        return item
    return item.model_copy(update={"remediation": steps})


def without_job(item: Diagnostic) -> Diagnostic:
    """Drop the remediation steps that need a job id ``item`` does not have.

    ``yonga chat`` and ``yonga serve`` open no job, so a registry step such as
    "the failing pass is in the job log: yonga logs JOB_ID --raw" can never be
    filled for them, and printing the placeholder would send the user looking
    for a job that does not exist. The raw text such a step points at is still
    shown with ``--debug``. A diagnostic that has a job id is returned as it is.
    """
    if item.job_id:
        return item
    steps = [step for step in item.remediation if not _JOB_PLACEHOLDER.search(step)]
    if steps == item.remediation:
        return item
    return item.model_copy(update={"remediation": steps})


def for_artifact_run(
    item: Diagnostic,
    *,
    run_command: str,
    devices: Iterable[str],
    has_job: bool,
) -> Diagnostic:
    """Point ``item``'s advice at the prepared artifact a command was running.

    Registry remediation is written for ``prepare``: "run this model on the GPU
    instead: yonga prepare MODEL --device gpu". When ``chat``, ``bench`` or
    ``probe`` fails on one device with an artifact that already passed on that
    fallback device, preparing it again is the wrong advice: the same command
    with another ``--device`` is the right one. ``run_command`` is that command
    without its ``--device`` (``yonga chat e3014dda15c8a5be``), and ``devices``
    are the ones the artifact passed its probe on.

    A step that ends in ``yonga prepare <model> --device D`` for a device in
    ``devices`` is replaced by one that runs ``run_command`` there, and the
    "Run:" line under the fallback devices names it too. Steps for a device the
    artifact was never probed on keep their ``prepare`` advice. Without a job
    (``has_job`` false) the steps that need its id are dropped, see
    :func:`without_job`.
    """
    runnable = {name for name in (normalise_device(device) for device in devices) if name}
    steps: list[str] = []
    for step in item.remediation:
        device = _prepared_device(step)
        if device is not None and device in runnable:
            command = f"{run_command} --device {device.lower()}"
            step = f"Run this artifact on the {device} instead: {command}"
        steps.append(step)
    context = dict(item.context)
    first = item.fallback_devices[0] if item.fallback_devices else None
    if first is not None and first in runnable:
        context["fallback_command"] = f"{run_command} --device {first.lower()}"
    updated = item.model_copy(update={"remediation": steps, "context": context})
    if not has_job:
        updated = without_job(updated)
    return fill_placeholders(updated)


#: A remediation step naming the job id it cannot know yet.
_JOB_PLACEHOLDER = re.compile(r"(?<![\w-])JOB_ID(?![\w-])")

#: ``yonga prepare <model> --device <device>`` and nothing more, once whitespace is collapsed.
_PREPARE_FOR_DEVICE = re.compile(r"yonga prepare \S+ --device (\S+)")


def _prepared_device(step: str) -> str | None:
    """The device a step's trailing ``yonga prepare MODEL --device D`` prepares for."""
    command = _TRAILING_COMMAND.search(step.strip())
    if command is None:
        return None
    found = _PREPARE_FOR_DEVICE.fullmatch(" ".join(command.group(1).split()))
    if found is None:
        return None
    return normalise_device(found.group(1)) or None


class YongaError(Exception):
    """Base class for typed failures; always carries a :class:`Diagnostic`.

    The primary constructor takes a fully built diagnostic. For convenience at
    adapter boundaries a bare message may be passed instead, in which case the
    subclass' ``default_code``/``default_category``/``default_title`` are used.
    """

    default_code: ClassVar[str] = "internal.unknown_error"
    default_category: ClassVar[FailureCategory] = FailureCategory.UNKNOWN_RUNTIME_ERROR
    default_title: ClassVar[str] = "Unexpected internal error"
    #: Used by :meth:`exit_code` only when the category cannot say anything better
    #: than ``unknown_runtime_error``; see the note on :data:`CATEGORY_EXIT_CODES`.
    fallback_exit_code: ClassVar[ExitCode | None] = None

    def __init__(self, diagnostic: Diagnostic | str, *, cause: BaseException | None = None) -> None:
        if isinstance(diagnostic, str):
            diagnostic = Diagnostic(
                code=self.default_code,
                category=self.default_category,
                title=self.default_title,
                message=diagnostic,
            )
        super().__init__(diagnostic.message or diagnostic.title)
        self.diagnostic: Diagnostic = diagnostic
        self.cause: BaseException | None = cause
        if cause is not None:
            self.__cause__ = cause

    def exit_code(self) -> ExitCode:
        """Exit code for this failure, preferring the stage-specific fallback."""
        code = self.diagnostic.exit_code()
        if code is ExitCode.INTERNAL and self.fallback_exit_code is not None:
            return self.fallback_exit_code
        return code


class ConfigurationError(YongaError):
    """Invalid configuration file, environment variable or CLI combination."""

    default_code: ClassVar[str] = "config.invalid"
    default_category: ClassVar[FailureCategory] = FailureCategory.CONFIGURATION_ERROR
    default_title: ClassVar[str] = "Configuration error"
    fallback_exit_code: ClassVar[ExitCode | None] = ExitCode.REGISTRY


class RegistryError(YongaError):
    """The compatibility registry could not be loaded or resolved deterministically."""

    default_code: ClassVar[str] = "registry.invalid"
    default_category: ClassVar[FailureCategory] = FailureCategory.REGISTRY_ERROR
    default_title: ClassVar[str] = "Registry error"
    fallback_exit_code: ClassVar[ExitCode | None] = ExitCode.REGISTRY


class SourceResolutionError(YongaError):
    """No usable conversion source could be resolved for the requested model."""

    default_code: ClassVar[str] = "source.unresolved"
    default_category: ClassVar[FailureCategory] = FailureCategory.SOURCE_FORMAT_UNSUPPORTED
    default_title: ClassVar[str] = "Model source could not be resolved"
    fallback_exit_code: ClassVar[ExitCode | None] = ExitCode.SOURCE_RESOLUTION


class HubAuthError(YongaError):
    """The Hugging Face repository is gated or private and needs authentication."""

    default_code: ClassVar[str] = "hub.auth_required"
    default_category: ClassVar[FailureCategory] = FailureCategory.GATED_MODEL_AUTH_REQUIRED
    default_title: ClassVar[str] = "Authentication required"
    fallback_exit_code: ClassVar[ExitCode | None] = ExitCode.AUTH_REQUIRED


class RemoteCodeRequiredError(YongaError):
    """The model can only be loaded with ``trust_remote_code`` explicitly enabled."""

    default_code: ClassVar[str] = "source.remote_code_required"
    default_category: ClassVar[FailureCategory] = FailureCategory.REMOTE_CODE_REQUIRED
    default_title: ClassVar[str] = "Remote code approval required"
    fallback_exit_code: ClassVar[ExitCode | None] = ExitCode.REMOTE_CODE_REQUIRED


class PreflightError(YongaError):
    """Disk, memory or temporary-space preflight refused to start the job."""

    default_code: ClassVar[str] = "preflight.insufficient_resources"
    default_category: ClassVar[FailureCategory] = FailureCategory.INSUFFICIENT_DISK
    default_title: ClassVar[str] = "Preflight check failed"
    fallback_exit_code: ClassVar[ExitCode | None] = ExitCode.PREFLIGHT


class ExportError(YongaError):
    """Conversion/quantization of the source model failed."""

    default_code: ClassVar[str] = "export.failed"
    default_title: ClassVar[str] = "Conversion failed"
    fallback_exit_code: ClassVar[ExitCode | None] = ExitCode.CONVERSION


class ValidationError(YongaError):
    """The exported artifact did not pass validation.

    Note: this shadows :class:`pydantic.ValidationError` when both are imported
    into the same namespace; the contract fixes the name, so import it as
    ``from yonga.domain.diagnostics import ValidationError`` deliberately.
    """

    default_code: ClassVar[str] = "validation.failed"
    default_title: ClassVar[str] = "Artifact validation failed"
    fallback_exit_code: ClassVar[ExitCode | None] = ExitCode.VALIDATION


class RepairError(YongaError):
    """A required repair was denied, disabled or could not be applied safely."""

    default_code: ClassVar[str] = "repair.failed"
    default_title: ClassVar[str] = "Repair could not be applied"
    fallback_exit_code: ClassVar[ExitCode | None] = ExitCode.REPAIR_DENIED


class DeviceUnsupportedError(YongaError):
    """The requested device cannot run this model on this host."""

    default_code: ClassVar[str] = "device.unsupported"
    default_category: ClassVar[FailureCategory] = FailureCategory.DEVICE_UNAVAILABLE
    default_title: ClassVar[str] = "Requested device is not supported"
    fallback_exit_code: ClassVar[ExitCode | None] = ExitCode.DEVICE_UNSUPPORTED


class CompileError(YongaError):
    """An OpenVINO model failed to compile on the target device."""

    default_code: ClassVar[str] = "runtime.compile_failed"
    default_category: ClassVar[FailureCategory] = FailureCategory.COMPILER_UNSUPPORTED_GRAPH
    default_title: ClassVar[str] = "Model could not be compiled"
    fallback_exit_code: ClassVar[ExitCode | None] = ExitCode.COMPILE


class GenerationError(YongaError):
    """Text generation failed after the pipeline had been created."""

    default_code: ClassVar[str] = "runtime.generation_failed"
    default_title: ClassVar[str] = "Generation failed"
    fallback_exit_code: ClassVar[ExitCode | None] = ExitCode.GENERATION


class CancelledError(YongaError):
    """The user interrupted the operation.

    Note: this shadows :class:`builtins.CancelledError`/``asyncio.CancelledError``
    when star-imported; the contract fixes the name.
    """

    default_code: ClassVar[str] = "job.cancelled"
    default_category: ClassVar[FailureCategory] = FailureCategory.USER_CANCELLED
    default_title: ClassVar[str] = "Cancelled"
    fallback_exit_code: ClassVar[ExitCode | None] = ExitCode.CANCELLED


class HealthCheck(BaseModel):
    """One ``doctor`` check with its status and remediation hints."""

    model_config = ConfigDict(extra="ignore")

    id: str
    label: str
    status: Severity = Severity.UNKNOWN
    value: str | None = None
    detail: str | None = None
    remediation: list[str] = Field(default_factory=list)


class HealthSection(BaseModel):
    """A named group of ``doctor`` checks, e.g. ``Devices`` or ``Storage``."""

    model_config = ConfigDict(extra="ignore")

    id: str
    title: str
    checks: list[HealthCheck] = Field(default_factory=list)

    @property
    def status(self) -> Severity:
        """The worst status among this section's checks."""
        return worst_severity(check.status for check in self.checks)
