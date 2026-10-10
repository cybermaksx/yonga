"""The preparation pipeline: plan, preflight, download, export, validate, repair, probe, publish.

This is the service behind ``yonga prepare``. It owns the state machine in
``docs/ARCHITECTURE.md`` section 5 and every policy decision in
``docs/MODEL_PIPELINE.md``; the mechanism lives entirely behind injected adapter
Protocols, so the whole pipeline is testable on a machine with no Intel
accelerator and no network.

Six rules are load-bearing and each one exists because the naive alternative
fails on real hardware.

**Plan first, then work.** :meth:`PrepareService.plan` performs no expensive side
effect: it inspects metadata, resolves the source and the device, picks a
profile, and computes the disk plan. The user sees the whole plan — including the
temp requirement and every confirmation — before a single byte is downloaded.

**Never convert in the system temp directory.** ``/tmp`` is a RAM-backed tmpfs on
most desktop Linux installs (16 GiB on the reference host). Every export runs
with ``TMPDIR``/``TMP``/``TEMP`` redirected to ``~/.cache/yonga/tmp/<job-id>``,
and preflight fails *before* the download when the managed temp or the artifact
filesystem clearly lacks room.

**Exit code zero is not readiness.** A conversion that exits ``0`` still has to
be inventoried, read, repaired if a known defect is present, compiled on the
target device, and made to generate tokens before anything is published.

**Never silently change device.** An explicitly requested device that fails stops
the job with the classified diagnostic and a recommended fallback *command*.
``--device auto`` may fall back, but only when the classifier says the
environment is healthy and the graph is what the device cannot take: a permission
or driver problem is a broken environment, and hiding it behind the GPU would
stop the user from ever fixing it.

**Security confirmations are not negotiable.** ``trust_remote_code`` is enabled
only by the explicit flag. ``--yes`` can approve a large download or a repair; it
can never approve executing repository code.

**An export we already know cannot succeed never starts.** The registry declares
which package versions an architecture's exporter needs, preflight computes package compatibility
(``docs/MODEL_PIPELINE.md`` section 5), and an unmet pin stops the job there —
before the download, not several minutes into the quantization. Printing the
problem and converting anyway is what this pipeline used to do, and it cost a
17.5 GiB download per attempt. ``--ignore-exporter-requirements`` is the escape
hatch for a stale pin; like ``--trust-remote-code`` it is its own deliberate
flag and ``--yes`` does not reach it.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from yonga import __version__
from yonga.adapters import (
    DeclaresRequirements,
    DeclaresRequirementsIn,
    ExportEnvironments,
    Exporter,
    FilesystemProbe,
    GGUFImport,
    HardwareProbe,
    HubClient,
    ReportsToolchain,
    RuntimeProbe,
)
from yonga.adapters.filesystem import (
    format_bytes,
    is_tool_owned_incomplete,
    mark_incomplete,
    system_temp_dir,
)
from yonga.adapters.filesystem import plan_disk as plan_disk_usage
from yonga.application import export_reuse
from yonga.application.artifacts import ArtifactStore
from yonga.application.jobs import JobContext, new_job_id
from yonga.application.progress import ProgressKind
from yonga.application.resolve import SourcePolicy, SourceResolver, gguf_quant_label
from yonga.config.settings import DevicePolicy, Settings
from yonga.domain.artifacts import (
    MANIFEST_SCHEMA_VERSION,
    ArtifactManifest,
    DeviceProbeReport,
    ExportCheckpoint,
    ExportRecord,
    IRDescription,
    ManifestModelRef,
    ProbeResult,
    ProbeStatus,
    RecommendedRuntime,
    RepairRecord,
    SubmodelRole,
    ValidationReport,
    failure_text,
)
from yonga.domain.compatibility import (
    CompatibilityDecision,
    DeviceCompatibility,
    DeviceStatus,
    EvidenceLevel,
    ProfileSelection,
)
from yonga.domain.diagnostics import (
    CancelledError,
    DeviceUnsupportedError,
    Diagnostic,
    ExportError,
    FailureCategory,
    PreflightError,
    RemoteCodeRequiredError,
    RepairError,
    Severity,
    SourceResolutionError,
    ValidationError,
    YongaError,
    diagnostic,
)
from yonga.domain.environments import (
    ESTIMATED_ENVIRONMENT_BYTES,
    EnvironmentSpec,
    ExportEnvironment,
)
from yonga.domain.hardware import HardwareInventory, normalise_device
from yonga.domain.models import (
    ModelFormat,
    ModelIdentity,
    ModelMetadata,
    PipelineKind,
    ResolvedSource,
    SourceEvidence,
    SourceRelationship,
)
from yonga.domain.plans import (
    CONFIRMATION_DEVICE_FALLBACK,
    CONFIRMATION_LARGE_DOWNLOAD,
    CONFIRMATION_MANAGED_ENVIRONMENT,
    CONFIRMATION_OVERWRITE_EXISTING,
    CONFIRMATION_REMOTE_CODE,
    CONFIRMATION_REPAIR,
    CONFIRMATION_THIRD_PARTY_DERIVATIVE,
    TEMP_ENV_KEYS,
    ConfirmationReason,
    ConversionProfile,
    DiskPlan,
    ExporterRequirements,
    ExportRequest,
    ExportResult,
    JobStage,
    PreparationPlan,
)
from yonga.registry.resolver import RegistryResolver
from yonga.registry.schema import (
    ArchitectureRule,
    ChatProfileRule,
    ModelRule,
    ProfileRule,
    RepairRule,
)
from yonga.repairs import Repair, RepairContext, RepairMatch, apply_repairs
from yonga.util.hashing import sha256_text, short_hash
from yonga.util.redaction import redact_argv, redact_text
from yonga.util.versions import installed_version, parse_version, satisfies

__all__ = [
    "AUTO_DEVICE_PREFERENCE",
    "EXPECTED_EXPORT_FILES",
    "GENAI_ONLY_PROPERTIES",
    "IGNORE_EXPORTER_REQUIREMENTS_FLAG",
    "SMOKE_MAX_NEW_TOKENS",
    "SMOKE_PROMPT",
    "STAGE_LABELS",
    "TOTAL_STAGES",
    "ConfirmCallback",
    "PreparationOutcome",
    "PrepareService",
    "RequirementState",
    "SmokeTester",
    "requirement_state",
]

#: Stage labels. The renderer prefixes the counter, so an event carries the
#: bare label here and its position in ``current``/``total``
#: (``docs/CLI_SPEC.md`` section 6).
STAGE_LABELS: tuple[str, ...] = (
    "Inspecting source",
    "Resolving compatibility",
    "Preflight",
    "Downloading",
    "Converting",
    "Validating / repairing",
    "Probing runtime",
)

#: Number of stages in the preparation pipeline.
TOTAL_STAGES: int = len(STAGE_LABELS)

#: Prompt used for the post-compile generation smoke test. Deliberately trivial:
#: the test proves the pipeline generates at all, it is not a quality measurement.
SMOKE_PROMPT: str = "Hello"

#: Tokens generated by the smoke test (``SPEC.md`` section 12: a minimal smoke test).
SMOKE_MAX_NEW_TOKENS: int = 24

#: Properties that belong to an OpenVINO GenAI *pipeline*, not to ``Core.compile_model``.
#: ``ATTENTION_BACKEND`` in particular is a GenAI configuration key and is rejected by
#: the plugin when passed to a plain compile, so it is filtered out of compile probes
#: and applied only when the smoke test builds the real pipeline.
GENAI_ONLY_PROPERTIES: frozenset[str] = frozenset(
    {
        "ATTENTION_BACKEND",
        "DRAFT_MODEL",
        "GENERATION_CONFIG",
        "MAX_PROMPT_LEN",
        "MIN_RESPONSE_LEN",
        "PROMPT_LOOKUP",
        "SCHEDULER_CONFIG",
    }
)

#: File names an OpenVINO export is expected to produce, used at planning time to
#: predict which registry repairs are likely to apply before anything is converted.
EXPECTED_EXPORT_FILES: tuple[str, ...] = (
    "openvino_model.xml",
    "openvino_language_model.xml",
    "openvino_text_embeddings_model.xml",
    "openvino_tokenizer.xml",
    "openvino_detokenizer.xml",
)

#: Order auto-selection prefers among devices the registry rates equally.
#: GPU leads on purpose: on an Intel AI PC the iGPU takes by far the widest range
#: of graphs, the NPU is the pickiest, and CPU is the guaranteed last resort.
AUTO_DEVICE_PREFERENCE: tuple[str, ...] = ("GPU", "NPU", "CPU")

#: Source formats that can be handed to the exporter (or used directly) as-is.
_USABLE_SOURCE_FORMATS: frozenset[ModelFormat] = frozenset(
    {ModelFormat.OPENVINO_IR, ModelFormat.SAFETENSORS, ModelFormat.PYTORCH_BIN}
)

_STATUS_RANK: Mapping[DeviceStatus, int] = {
    DeviceStatus.SUPPORTED: 3,
    DeviceStatus.EXPERIMENTAL: 2,
    DeviceStatus.UNKNOWN: 1,
    DeviceStatus.UNSUPPORTED: 0,
}

_STRONG_EVIDENCE: frozenset[EvidenceLevel] = frozenset(
    {EvidenceLevel.OFFICIAL, EvidenceLevel.TESTED}
)

_AUTO_DEVICE_NAMES: frozenset[str] = frozenset({"", "AUTO"})

#: Backend recorded for an artifact that was already OpenVINO IR upstream.
_PASSTHROUGH_BACKEND: str = "openvino-ir-passthrough"

#: Backend recorded for a normal Optimum Intel conversion.
_EXPORT_BACKEND: str = "optimum-intel"

#: Recorded as the export backend of an artifact imported from a GGUF file.
_GGUF_BACKEND: str = "openvino-genai-gguf"

#: Distributions whose versions are reported in the validation record.
_OPENVINO_DISTRIBUTION: str = "openvino"
_TOKENIZERS_DISTRIBUTION: str = "openvino-tokenizers"

#: Name of the flag that overrides an unmet exporter pin. Spelled once so the
#: diagnostic, the plan panel's hint and the CLI option cannot drift apart.
IGNORE_EXPORTER_REQUIREMENTS_FLAG: str = "--ignore-exporter-requirements"

RequirementState = Literal["ok", "missing", "mismatch", "unverifiable"]
"""How one declared exporter version pin compares to what is installed."""

#: States that stop preparation. ``unverifiable`` is deliberately absent.
_BLOCKING_REQUIREMENT_STATES: frozenset[RequirementState] = frozenset({"missing", "mismatch"})


def requirement_state(installed: str | None, specifier: str) -> RequirementState:
    """Classify one declared exporter pin against the installed version.

    Four outcomes, kept distinct because only two of them may stop a job:

    ``missing``
        The distribution is not installed at all. The export cannot run.
    ``mismatch``
        A parseable version is installed and it contradicts the pin. This is the
        Qwen3.5 case: ``optimum-intel`` 2.2.0 imports ``Qwen3_5DynamicCache``,
        which ``transformers`` 5.5.4 no longer exposes, so the export dies deep
        inside the patcher after the weights have already been fetched.
    ``unverifiable``
        Something is installed but its version cannot be parsed, so the pin
        cannot be judged either way. :func:`~yonga.util.versions.satisfies`
        answers ``False`` here exactly as it does for a real contradiction,
        which is why a bare ``False`` must never be read as "wrong".
    ``ok``
        The installed version satisfies the pin, or there is no pin.
    """
    if installed is None:
        return "missing"
    if satisfies(installed, specifier):
        return "ok"
    if parse_version(installed) is None:
        return "unverifiable"
    return "mismatch"


ConfirmCallback = Callable[[ConfirmationReason], bool]
"""Asks the user to approve one :class:`ConfirmationReason`.

The CLI owns the ``--yes`` policy, but it must never auto-approve a reason whose
``security_sensitive`` flag is set (``SPEC.md`` section 6.3). The service enforces
the one case that cannot be delegated: ``trust_remote_code`` must already be
enabled by its own explicit flag before the pipeline will even start.
"""


@runtime_checkable
class SmokeTester(Protocol):
    """Builds a real generation pipeline and proves it produces tokens.

    Implemented by the CLI over :func:`yonga.adapters.runtime_genai.open_runtime`;
    kept as a port here so the application layer never constructs a GenAI
    pipeline itself. When no tester is injected the smoke test is recorded as
    :attr:`~yonga.domain.artifacts.ProbeStatus.SKIPPED` rather than silently
    claimed to have passed.
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
        """Generate a handful of tokens and report the outcome as a typed result."""
        ...


class PreparationOutcome(BaseModel):
    """What one preparation produced, successful or reused."""

    model_config = ConfigDict(extra="ignore")

    job_id: str
    artifact_dir: Path
    manifest: ArtifactManifest
    device: str
    reused: bool = False
    fallbacks: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    repairs: list[RepairRecord] = Field(default_factory=list)
    validation: ValidationReport | None = None
    duration_seconds: float = 0.0

    @property
    def fell_back(self) -> bool:
        """True when the job ended on a device other than the one first attempted."""
        return bool(self.fallbacks)


@dataclass
class _StageTracker:
    """Remembers which stage is running so a failure can be recorded against it."""

    stage: JobStage = JobStage.CREATED


@dataclass
class _ExportOutcome:
    """The export step's result, normalised across conversion and IR pass-through."""

    record: ExportRecord
    result: ExportResult | None = None
    notes: list[str] = field(default_factory=list)


def _runtime_task(task: str | None) -> str | None:
    """The export task yonga's runtime can use for a Hub pipeline tag.

    The Hub tags a causal LM ``text-generation``, and optimum-intel exports
    exactly that when asked: a stateless model with no KV cache and no
    ``beam_idx`` input, which OpenVINO GenAI's ``LLMPipeline`` cannot run
    ("Port for tensor name beam_idx was not found", observed exporting
    OLMo-2 on the reference host). Every text-generation artifact yonga makes
    is run by that pipeline, so the stateful ``-with-past`` export is the
    only useful one. A registry rule that names a task still wins.
    """
    if task == "text-generation":
        return "text-generation-with-past"
    return task


def _artifact_key(resolved: ResolvedSource, profile: ConversionProfile) -> str:
    """What names an artifact: its profile, or for a GGUF import the file's quantization.

    A GGUF import keeps the file's own quantization, so naming it after the
    profile would describe weights it does not have; and one repository
    publishes several files, which must not share an artifact directory.
    """
    if resolved.gguf_file is None:
        return profile.id
    return f"gguf-{gguf_quant_label(resolved.gguf_file)}-{profile.device.lower()}"


def _now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(tz=UTC)


def _as_identity(value: ModelIdentity | str) -> ModelIdentity:
    """Coerce a model reference into a :class:`ModelIdentity`."""
    if isinstance(value, ModelIdentity):
        return value
    return ModelIdentity.parse(value)


def _pinned(identity: ModelIdentity, metadata: ModelMetadata) -> ModelIdentity:
    """Return ``identity`` pinned to the revision the Hub resolved, when known."""
    resolved = metadata.resolved_revision
    if resolved:
        return identity.with_revision(resolved)
    return identity


def _tri_state(value: str | None) -> bool | None:
    """Translate an ``ok``/``failed``/``missing`` probe outcome into a tri-state."""
    if value == "ok":
        return True
    if value == "failed":
        return False
    return None


def _all_ok(*values: str | None) -> bool | None:
    """Fold several ``ok``/``failed`` outcomes into one tri-state."""
    states = [_tri_state(value) for value in values]
    if any(state is False for state in states):
        return False
    if all(state is True for state in states):
        return True
    return None


#: The artifact directory itself, in the relative form recorded paths use.
_ARTIFACT_ROOT: Path = Path()


def _artifact_relative(path: Path, staging: Path) -> Path:
    """Return ``path`` as it will read once ``staging`` has been published.

    Validation and probing run against the staging directory, so everything they
    record names ``<data>/artifacts/.staging/<job-id>/model/...`` — a directory
    that :meth:`ArtifactStore.finalize` renames onto the artifact directory and
    that therefore does not exist by the time anyone reads the manifest. Every
    such path is rewritten relative to the artifact directory instead, which is
    the form ``manifest.files`` already uses and the only one that survives the
    artifact being moved, copied to another host or published.

    Paths outside the staging root are left exactly as they are: they name
    something that is not part of this artifact, and inventing a relative path
    for them would be a lie rather than a translation.
    """
    if path == staging:
        return _ARTIFACT_ROOT
    if path.is_relative_to(staging):
        return path.relative_to(staging)
    return path


def _relocated_ir(ir: IRDescription, staging: Path) -> IRDescription:
    """Copy one IR description with its file paths made artifact-relative."""
    moved = ir.model_copy(deep=True)
    moved.path = _artifact_relative(ir.path, staging)
    if ir.bin_path is not None:
        moved.bin_path = _artifact_relative(ir.bin_path, staging)
    return moved


def _relocated_probe_result(result: ProbeResult, staging: Path) -> ProbeResult:
    """Copy one probe result with the file it named made artifact-relative."""
    moved = result.model_copy(deep=True)
    if result.path is not None:
        moved.path = _artifact_relative(result.path, staging)
    return moved


def _relocated_validation(report: ValidationReport, staging: Path) -> ValidationReport:
    """Copy a validation report so its IR paths outlive the staging directory."""
    moved = report.model_copy(deep=True)
    moved.irs = [_relocated_ir(ir, staging) for ir in report.irs]
    return moved


def _relocated_device_probe(report: DeviceProbeReport, staging: Path) -> DeviceProbeReport:
    """Copy a device probe report so its submodel paths outlive staging."""
    moved = report.model_copy(deep=True)
    moved.submodels = [_relocated_probe_result(result, staging) for result in report.submodels]
    if report.smoke_test is not None:
        moved.smoke_test = _relocated_probe_result(report.smoke_test, staging)
    return moved


def _relocated_export(record: ExportRecord, staging: Path) -> ExportRecord:
    """Copy an export record, adding where the command's output actually ended up.

    ``argv`` is deliberately *not* rewritten. It is the record of a command that
    was executed, and a command nobody ran is worth nothing in a bug report; its
    last token is the staging directory, exactly as the exporter received it. So
    the honest half stays literal and the usable half is recorded beside it:
    ``arguments["output_dir"]`` gives the published home of that output
    directory, relative to the artifact directory — ``"."``, the directory the
    manifest itself lives in. Re-running the export therefore means replaying
    ``argv`` with its final argument pointed at a directory of your choosing.
    """
    moved = record.model_copy(deep=True)
    moved.arguments = {
        **record.arguments,
        "output_dir": _artifact_relative(staging, staging).as_posix(),
    }
    return moved


class PrepareService:
    """Plans and runs model preparation, from Hub metadata to a published artifact."""

    def __init__(
        self,
        *,
        hub_client: HubClient,
        exporter: Exporter,
        runtime_probe: RuntimeProbe,
        hardware_probe: HardwareProbe,
        filesystem: FilesystemProbe,
        registry_resolver: RegistryResolver,
        artifact_store: ArtifactStore,
        settings: Settings,
        smoke_tester: SmokeTester | None = None,
        export_environments: ExportEnvironments | None = None,
        gguf_importer: GGUFImport | None = None,
    ) -> None:
        """Wire the service to its adapters.

        Every collaborator is a Protocol, so a test can drive the whole pipeline
        with fakes. ``smoke_tester`` is optional because generation needs
        OpenVINO GenAI: when it is absent the smoke test is *recorded as skipped*
        rather than assumed to pass.
        """
        self._hub = hub_client
        self._exporter = exporter
        self._runtime = runtime_probe
        self._hardware = hardware_probe
        self._filesystem = filesystem
        self._registry = registry_resolver
        self._artifacts = artifact_store
        self._settings = settings
        self._smoke = smoke_tester
        # Optional: without it an unmet exporter pin blocks at preflight, exactly
        # as it did before managed export environments existed.
        self._environments = export_environments
        self._gguf_importer = gguf_importer

    # ==================================================================
    # planning
    # ==================================================================

    def plan(
        self,
        identity: ModelIdentity | str,
        *,
        revision: str | None = None,
        device: str | DevicePolicy | None = None,
        profile_id: str | None = None,
        explicit_source: ModelIdentity | str | None = None,
        trust_remote_code: bool = False,
        force: bool = False,
        job: JobContext | None = None,
        managed_env: bool = True,
        reuse_export: bool = True,
        gguf_file: str | None = None,
    ) -> PreparationPlan:
        """Decide everything, change nothing.

        ``reuse_export`` lets the plan adopt a completed export that a failed job
        with identical inputs left behind (see :mod:`yonga.application.export_reuse`);
        the CLI turns it off for ``--fresh-export``.

        Metadata lookups are the only I/O: no download, no conversion, no
        directory is created. The returned plan is what the CLI prints before
        asking for confirmation, and what :meth:`prepare` then executes.
        """
        requested = _as_identity(identity)
        if revision:
            requested = requested.with_revision(revision)
        requested_revision = requested.revision

        job_id = job.job_id if job is not None else new_job_id()
        paths = job.paths if job is not None else self._settings.paths

        metadata = self._hub.inspect(requested)
        # Pin the requested repository to its immutable commit before anything else
        # (``docs/MODEL_PIPELINE.md`` section 4). ``requested_revision`` keeps what the
        # user actually asked for, so the manifest can record both.
        requested = _pinned(requested, metadata)
        if requested.revision is None:
            requested = self._pin_from_hub(requested)

        resolved_source = self._resolve_source(requested, metadata, explicit_source, gguf_file)
        source_meta = resolved_source.metadata or metadata

        inventory = self._hardware.inventory()
        host_tags = tuple(inventory.device_tags)
        arch_rule, arch_explanation = self._registry.resolve_architecture(source_meta, host_tags)
        model_rule, _model_explanation = self._registry.resolve_model_rule(source_meta)

        requested_device = self._requested_device(device)
        compatibility = self._decide_device(
            requested_device,
            arch_rule=arch_rule,
            model_rule=model_rule,
            inventory=inventory,
        )
        target_device = compatibility.selected_device or requested_device or "CPU"

        # A GGUF import keeps the file's own quantization and an OpenVINO source
        # is copied as published, so the model's size must not pick a profile
        # whose quantization would never be applied.
        size_applies = resolved_source.gguf_file is None and not resolved_source.already_openvino
        profile, profile_explanation = self._registry.resolve_profile(
            device=target_device,
            arch_rule=arch_rule,
            meta=source_meta,
            model_rule=model_rule,
            explicit_profile_id=profile_id,
            size_applies=size_applies,
        )
        profile_selection = profile_explanation.profile_selection
        if profile_selection is not None and not size_applies:
            profile_selection = profile_selection.model_copy(update={"applies": False})
        task = self._task(source_meta, arch_rule, model_rule)
        exporter_requirements, exporter_unverifiable = self._exporter_requirements(
            arch_rule,
            model_rule,
            model_type=source_meta.model_type,
            task=task,
            skip_exporter=resolved_source.gguf_file is not None or resolved_source.already_openvino,
        )

        source_sha = source_meta.resolved_revision or resolved_source.source.revision
        artifact_key = _artifact_key(resolved_source, profile)
        artifact_dir = self._artifacts.artifact_dir(
            resolved_source.source.repo_id, source_sha, artifact_key
        )
        staging_dir = self._artifacts.staging_model_dir(job_id)

        download_bytes = sum(entry.size or 0 for entry in source_meta.files)
        weight_bytes = source_meta.weight_bytes
        if resolved_source.gguf_file is not None:
            # One file of the repository's many quantizations is downloaded and imported.
            download_bytes = resolved_source.gguf_file_bytes or 0
            weight_bytes = resolved_source.gguf_file_bytes or weight_bytes
        disk = plan_disk_usage(
            job_temp_dir=paths.job_tmp_dir(job_id),
            artifact_dir=artifact_dir,
            source_weight_bytes=weight_bytes,
            settings=self._settings,
            expected_download_bytes=download_bytes,
            download_dir=inventory.hf_auth.cache_dir if inventory.hf_auth is not None else None,
            probe=self._filesystem,
        )

        pipeline = self._pipeline_kind(source_meta, arch_rule)
        remote_code_required = self._remote_code_required(source_meta, arch_rule, model_rule)
        expected_repairs = [rule.id for rule in self._expected_repairs()]
        known_limitations = list(arch_rule.known_limitations) if arch_rule is not None else []
        if arch_rule is not None:
            # On a successful match the only note resolution adds is how the
            # architecture was identified. The pipeline, the exporter pin and the
            # device recommendation below all descend from that match, so a plan
            # built on a weight-file hint says so before anything is downloaded
            # (``docs/CLI_SPEC.md`` section 1).
            known_limitations.extend(arch_explanation.notes)
        # Profile resolution composes a note whenever the profile it returns is
        # not the obvious one: an explicitly requested profile that targets
        # another device, one whose version constraints this host does not meet,
        # or a rule naming a profile the registry does not contain. Nothing else
        # ever says so, so dropping the explanation here made those sentences
        # unreachable. A user asking for `--profile npu-int4-sym-cw --device gpu`
        # gets told, in the plan, before anything is downloaded.
        known_limitations.extend(profile_explanation.notes)
        if source_meta.requires_remote_code and not remote_code_required:
            files = ", ".join(source_meta.remote_code_files) or "an auto_map"
            known_limitations.append(
                f"The repository ships its own modelling code ({files}). It is not run: "
                "Transformers implements this architecture natively, and remote code stays "
                "off."
            )
        if resolved_source.gguf_file is not None:
            types = ", ".join(
                f"{name} x{count}" for name, count in resolved_source.gguf_tensor_types.items()
            )
            known_limitations.extend(
                reason for reason in resolved_source.reasons if reason.startswith("Passed over")
            )
            known_limitations.append(
                f"{resolved_source.gguf_file} is imported directly by OpenVINO GenAI and keeps "
                f"its own quantization ({types}); the {profile.id} profile's quantization "
                "settings do not apply, only its runtime properties."
            )
        export_environment = self._plan_export_environment(
            exporter_requirements, resolved_source=resolved_source, allowed=managed_env
        )
        environment_ready = (
            export_environment is not None
            and self._environments is not None
            and self._environments.find(export_environment) is not None
        )
        if export_environment is not None:
            known_limitations.extend(
                f"The installed packages do not meet an exporter requirement ({problem}); the "
                f"export runs in managed environment {export_environment.name} instead."
                for problem in exporter_requirements.problems
            )
        else:
            known_limitations.extend(
                f"Exporter requirement not satisfied: {problem}"
                for problem in exporter_requirements.problems
            )
        known_limitations.extend(
            f"Exporter requirement could not be checked: {note}" for note in exporter_unverifiable
        )

        reuse = None if force else self._reusable_artifact(artifact_dir, source_sha, artifact_key)
        if reuse is None and profile_selection is not None and profile_selection.applies:
            known_limitations.extend(
                self._existing_under_other_profiles(artifact_dir, profile, profile_selection)
            )
        confirmations = self._confirmations(
            resolved_source=resolved_source,
            remote_code_required=remote_code_required,
            download_bytes=download_bytes,
            expected_repairs=expected_repairs,
            artifact_dir=artifact_dir,
            reuse=reuse,
            force=force,
            export_environment=None if environment_ready else export_environment,
        )

        plan = PreparationPlan(
            job_id=job_id,
            created_at=_now(),
            requested=requested,
            requested_revision=requested_revision,
            resolved_source=resolved_source,
            architecture=self._architecture_identity(source_meta, arch_rule),
            architecture_rule_id=arch_rule.id if arch_rule is not None else None,
            task=task,
            pipeline=pipeline,
            hardware=inventory,
            compatibility=compatibility,
            target_device=target_device,
            profile=profile,
            exporter=exporter_requirements,
            disk=disk,
            artifact_dir=artifact_dir,
            staging_dir=staging_dir,
            trust_remote_code_required=remote_code_required,
            trust_remote_code_enabled=bool(trust_remote_code),
            expected_repairs=expected_repairs,
            known_limitations=known_limitations,
            confirmations=confirmations,
            reuse_existing_artifact=reuse,
            force=force,
            export_environment=export_environment,
            export_environment_ready=environment_ready,
            profile_selection=profile_selection,
        )
        if not reuse_export or reuse is not None or resolved_source.already_openvino:
            return plan
        reusable = export_reuse.find_reusable(
            paths, self._export_fingerprint(plan), exclude_job=job_id
        )
        return plan if reusable is None else plan.model_copy(update={"reusable_export": reusable})

    # --- planning helpers -------------------------------------------------

    def _plan_export_environment(
        self,
        requirements: ExporterRequirements,
        *,
        resolved_source: ResolvedSource,
        allowed: bool,
    ) -> EnvironmentSpec | None:
        """The managed environment this export needs, or None if it needs none.

        Only an unmet pin calls for one, and only on a job that actually runs
        the exporter: an OpenVINO source is copied, never converted. Refusing is
        kept available -- ``--no-managed-env``, the ``managed_export_envs``
        setting, or a service built without an environment manager -- and then
        preflight blocks on the pin as before.
        """
        if (
            not requirements.problems
            or resolved_source.already_openvino
            or resolved_source.format is ModelFormat.GGUF
        ):
            return None
        if not allowed or not self._settings.managed_export_envs or self._environments is None:
            return None
        return self._environments.spec_for(requirements, reason="; ".join(requirements.problems))

    def _pin_from_hub(self, identity: ModelIdentity) -> ModelIdentity:
        """Resolve a branch or tag to a commit SHA when ``inspect`` did not.

        A failure here is not fatal: an unpinned revision costs reproducibility,
        not correctness, so it degrades to the reference the user gave.
        """
        try:
            resolved = self._hub.resolve_revision(identity)
        except (YongaError, LookupError, OSError):
            return identity
        return identity.with_revision(resolved) if resolved else identity

    def _requested_device(self, device: str | DevicePolicy | None) -> str:
        """Normalise the requested device, falling back to the configured default."""
        raw = device if device is not None else self._settings.default_device
        if isinstance(raw, DevicePolicy):
            raw = raw.value
        normalised = normalise_device(str(raw))
        return "AUTO" if normalised in _AUTO_DEVICE_NAMES else normalised

    def _resolve_source(
        self,
        requested: ModelIdentity,
        metadata: ModelMetadata,
        explicit_source: ModelIdentity | str | None,
        gguf_file: str | None = None,
    ) -> ResolvedSource:
        """Decide which repository will actually be converted, and record why.

        Follows ``docs/MODEL_PIPELINE.md`` section 3: use the requested repository
        when it carries weights the exporter understands, use a declared registry
        derivative when it only carries GGUF, and refuse rather than guess.
        """
        if explicit_source is not None:
            return self._explicit_source(requested, metadata, _as_identity(explicit_source))

        primary = metadata.primary_format
        if primary in _USABLE_SOURCE_FORMATS:
            reason = (
                "The requested repository is already OpenVINO IR."
                if primary is ModelFormat.OPENVINO_IR
                else f"The requested repository carries {primary.value} weights."
            )
            return ResolvedSource(
                requested=requested,
                source=_pinned(metadata.identity, metadata),
                relationship=SourceRelationship.ORIGINAL,
                format=primary,
                already_openvino=primary is ModelFormat.OPENVINO_IR,
                reasons=[reason],
                metadata=metadata,
            )

        return self._derivative_source(requested, metadata, gguf_file)

    def _explicit_source(
        self,
        requested: ModelIdentity,
        metadata: ModelMetadata,
        source: ModelIdentity,
    ) -> ResolvedSource:
        """Resolve a source the user named with ``--source``."""
        source_meta = self._hub.inspect(source)
        primary = source_meta.primary_format
        if primary not in _USABLE_SOURCE_FORMATS:
            raise SourceResolutionError(
                diagnostic(
                    "source.explicit_format_unsupported",
                    FailureCategory.SOURCE_FORMAT_UNSUPPORTED,
                    "The chosen source cannot be converted",
                    (
                        f"{source.repo_id} carries {primary.value} weights, which this "
                        "exporter cannot read."
                    ),
                    remediation=[
                        "Choose a source that publishes safetensors or OpenVINO IR.",
                        f"Inspect it first: yonga inspect {source.repo_id}",
                    ],
                    context={"requested": requested.repo_id, "source": source.repo_id},
                )
            )
        same_repo = source.repo_id == requested.repo_id
        relationship = (
            SourceRelationship.ORIGINAL if same_repo else SourceRelationship.LOSSLESS_CONVERSION
        )
        return ResolvedSource(
            requested=requested,
            source=_pinned(source, source_meta),
            relationship=relationship,
            format=primary,
            already_openvino=primary is ModelFormat.OPENVINO_IR,
            evidence=[
                SourceEvidence(
                    kind="user",
                    note="Source chosen explicitly on the command line with --source.",
                )
            ],
            requires_confirmation=not same_repo,
            reasons=[
                f"{source.repo_id} was named explicitly as the conversion source "
                f"for {requested.repo_id}.",
                f"The requested repository publishes {metadata.primary_format.value} weights.",
            ],
            metadata=source_meta,
        )

    def _derivative_source(
        self,
        requested: ModelIdentity,
        metadata: ModelMetadata,
        gguf_file: str | None = None,
    ) -> ResolvedSource:
        """Fall back to a registry-declared derivative, or refuse to guess."""
        candidates, _explanation = self._registry.resolve_derivatives(requested)
        rejected: list[str] = []
        for candidate in candidates:
            identity = ModelIdentity(repo_id=candidate.repo_id, revision=candidate.revision)
            try:
                candidate_meta = self._hub.inspect(identity)
            except YongaError as error:
                rejected.append(f"{candidate.repo_id}: {error.diagnostic.message}")
                continue
            if candidate_meta.primary_format not in _USABLE_SOURCE_FORMATS:
                rejected.append(
                    f"{candidate.repo_id} publishes {candidate_meta.primary_format.value} weights."
                )
                continue
            reasons = [
                f"{requested.repo_id} publishes {metadata.primary_format.value} only, "
                "which this backend cannot convert.",
                f"The registry declares {candidate.repo_id} as a "
                f"{candidate.relationship.value} of it.",
            ]
            if candidate.note:
                reasons.append(candidate.note)
            return ResolvedSource(
                requested=requested,
                source=_pinned(identity, candidate_meta),
                relationship=candidate.relationship,
                format=candidate_meta.primary_format,
                already_openvino=candidate_meta.primary_format is ModelFormat.OPENVINO_IR,
                evidence=candidate.source_evidence(),
                requires_confirmation=candidate.requires_confirmation,
                reasons=reasons,
                metadata=candidate_meta,
            )

        if metadata.primary_format is ModelFormat.GGUF:
            # Step 5 of SPEC.md section 8.1, only once no declared derivative is usable:
            # the resolver reads candidate file headers and imports one directly, or
            # refuses with the tensor types and architecture that ruled each file out.
            resolved = SourceResolver(self._hub).resolve_gguf(
                metadata,
                registry_resolver=self._registry,
                policy=SourcePolicy(gguf_file=gguf_file),
            )
            if rejected:
                resolved.reasons.append(
                    "Declared derivatives were unusable: " + "; ".join(rejected)
                )
            return resolved

        message = (
            f"{requested.repo_id} publishes {metadata.primary_format.value} weights, which "
            "this backend cannot convert, and no usable alternative source is declared."
        )
        raise SourceResolutionError(
            diagnostic(
                "source.format_unsupported",
                FailureCategory.SOURCE_FORMAT_UNSUPPORTED,
                "No usable conversion source",
                message if not rejected else message + " Rejected: " + "; ".join(rejected),
                remediation=[
                    "Name a safetensors repository explicitly: yonga prepare MODEL --source REPO",
                    "Or ask the publisher for a safetensors or OpenVINO IR release.",
                ],
                context={"requested": requested.repo_id, "format": metadata.primary_format.value},
            )
        )

    def _exporter_requirements(
        self,
        arch_rule: ArchitectureRule | None,
        model_rule: ModelRule | None,
        *,
        model_type: str | None = None,
        task: str | None = None,
        skip_exporter: bool = False,
    ) -> tuple[ExporterRequirements, list[str]]:
        """Compare the declared exporter constraints against this host.

        The registry's pins come first. Where it pins no Transformers version,
        the range optimum-intel's own export config declares for the model type
        is used instead, when the exporter can report one: that is the range the
        exporter will enforce anyway, so knowing it here lets preflight refuse,
        or a managed export environment satisfy it, before the download.

        A mismatch is reported, never fixed: yonga does not install or downgrade
        packages in the user's environment (``SPEC.md`` section 18.5).

        Returns the checked requirements plus the pins that could not be judged
        at all. ``problems`` is reserved for the two states that really do stop
        an export — an absent distribution and a contradicted version — because
        :attr:`ExporterRequirements.satisfied` is derived from it and preflight
        blocks on it. A version string nobody can parse is a gap in what yonga
        knows, not evidence that the export will fail, so it travels separately
        and lands in ``known_limitations`` where it informs without blocking.
        """
        spec = None
        if model_rule is not None and model_rule.export is not None:
            spec = model_rule.export
        elif arch_rule is not None:
            spec = arch_rule.export
        requirements = ExporterRequirements(
            transformers=spec.transformers if spec is not None else None,
            optimum_intel=spec.optimum_intel if spec is not None else None,
            openvino=spec.openvino if spec is not None else None,
            nncf=spec.nncf if spec is not None else None,
        )
        declared_by: str | None = None
        if (
            requirements.transformers is None
            and not skip_exporter
            and model_type
            and isinstance(self._exporter, DeclaresRequirements)
        ):
            declared = self._exporter.declared_requirements(model_type, task)
            if declared is not None:
                requirements = requirements.model_copy(
                    update={"transformers": declared.transformers}
                )
                declared_by = declared.source
        versions = self._exporter.requirements()
        problems: list[str] = []
        unverifiable: list[str] = []
        for name, specifier in requirements.specifiers().items():
            installed = versions.get(name)
            state = requirement_state(installed, specifier)
            if state == "ok":
                continue
            if state == "unverifiable":
                unverifiable.append(
                    f"{name} {specifier} is required, and the installed version "
                    f"{installed!r} could not be parsed, so the requirement could not be checked"
                )
                continue
            installed_text = (
                f"{installed} is installed" if installed else f"{name} is not installed"
            )
            origin = (
                f" (declared by {declared_by})" if name == "transformers" and declared_by else ""
            )
            problems.append(f"{name} {specifier} is required{origin}, but {installed_text}")
        if (
            not skip_exporter
            and isinstance(self._exporter, ReportsToolchain)
            and not self._exporter.toolchain_available()
            and not any(problem.startswith("optimum-intel ") for problem in problems)
        ):
            # No export toolchain beside yonga at all: a plain install, or the
            # AppImage. A managed environment can supply one; without it the
            # export could only fail, so preflight says so before the download.
            problems.append(
                "optimum-intel is required to convert this model, but optimum-intel is not "
                "installed alongside yonga"
            )
        return (
            requirements.model_copy(update={"problems": problems, "satisfied": not problems}),
            unverifiable,
        )

    def _decide_device(
        self,
        requested: str,
        *,
        arch_rule: ArchitectureRule | None,
        model_rule: ModelRule | None,
        inventory: HardwareInventory,
    ) -> CompatibilityDecision:
        """Turn registry knowledge plus the host inventory into a device decision."""
        per_device = self._registry.device_compatibility(
            arch_rule=arch_rule, inventory=inventory, model_rule=model_rule
        )
        by_name = {entry.device: entry for entry in per_device}
        available = list(inventory.available_device_names)
        matched = sorted({entry.rule_id for entry in per_device if entry.rule_id})
        explicit = requested != "AUTO"

        if explicit:
            entry = by_name.get(requested)
            reasons = [
                f"{requested} was requested explicitly; yonga never switches device on its own."
            ]
            if requested not in available:
                reasons.append(f"OpenVINO does not report {requested} on this host.")
            if entry is not None and entry.reason:
                reasons.append(entry.reason)
            return CompatibilityDecision(
                requested_device=requested,
                selected_device=requested,
                status=entry.status if entry is not None else DeviceStatus.UNKNOWN,
                evidence_level=entry.evidence if entry is not None else EvidenceLevel.UNKNOWN,
                matched_rule_ids=matched,
                reasons=reasons,
                fallback_candidates=self._ranked_devices(by_name, available, arch_rule, requested),
                per_device=per_device,
                explicit_request=True,
            )

        order = self._ranked_devices(by_name, available, arch_rule, None)
        if not order:
            return CompatibilityDecision(
                requested_device="AUTO",
                selected_device=None,
                status=DeviceStatus.UNSUPPORTED,
                evidence_level=EvidenceLevel.UNKNOWN,
                matched_rule_ids=matched,
                reasons=["No device reported by OpenVINO is usable for this model."],
                per_device=per_device,
                explicit_request=False,
            )
        selected = order[0]
        entry = by_name.get(selected)
        reasons = [f"Auto-selection chose {selected} as the most compatible available device."]
        if entry is not None and entry.reason:
            reasons.append(entry.reason)
        recommended = arch_rule.recommended_device if arch_rule is not None else None
        if recommended and recommended == selected:
            reasons.append(f"The registry recommends {selected} for this architecture.")
        return CompatibilityDecision(
            requested_device="AUTO",
            selected_device=selected,
            status=entry.status if entry is not None else DeviceStatus.UNKNOWN,
            evidence_level=entry.evidence if entry is not None else EvidenceLevel.UNKNOWN,
            matched_rule_ids=matched,
            reasons=reasons,
            fallback_candidates=order[1:],
            per_device=per_device,
            explicit_request=False,
        )

    @staticmethod
    def _ranked_devices(
        by_name: Mapping[str, DeviceCompatibility],
        available: Sequence[str],
        arch_rule: ArchitectureRule | None,
        exclude: str | None,
    ) -> list[str]:
        """Rank the usable devices, most compatible first.

        Devices the registry marks ``unsupported`` are dropped outright; the rest
        are ordered by declared status and then by
        :data:`AUTO_DEVICE_PREFERENCE`. A registry ``recommended_device`` backed
        by official or tested evidence is promoted to the front, which is what
        makes auto-selection compatibility-first rather than hardware-first.
        """
        candidates: list[str] = []
        for name in available:
            if exclude is not None and name == exclude:
                continue
            entry = by_name.get(name)
            if entry is not None and entry.status is DeviceStatus.UNSUPPORTED:
                continue
            candidates.append(name)

        def sort_key(name: str) -> tuple[int, int, str]:
            entry = by_name.get(name)
            status = entry.status if entry is not None else DeviceStatus.UNKNOWN
            preference = (
                AUTO_DEVICE_PREFERENCE.index(name)
                if name in AUTO_DEVICE_PREFERENCE
                else len(AUTO_DEVICE_PREFERENCE)
            )
            return (-_STATUS_RANK[status], preference, name)

        ordered = sorted(candidates, key=sort_key)
        recommended = arch_rule.recommended_device if arch_rule is not None else None
        if recommended and recommended in ordered:
            entry = by_name.get(recommended)
            if entry is None or entry.evidence in _STRONG_EVIDENCE:
                ordered.remove(recommended)
                ordered.insert(0, recommended)
        return ordered

    @staticmethod
    def _pipeline_kind(
        metadata: ModelMetadata,
        arch_rule: ArchitectureRule | None,
    ) -> PipelineKind:
        """Pick the runtime pipeline, preferring what the repository itself says."""
        inferred = metadata.pipeline_kind
        if inferred is not PipelineKind.UNKNOWN:
            return inferred
        if arch_rule is not None and arch_rule.export.pipeline is not PipelineKind.UNKNOWN:
            return arch_rule.export.pipeline
        return PipelineKind.UNKNOWN

    @staticmethod
    def _task(
        metadata: ModelMetadata,
        arch_rule: ArchitectureRule | None,
        model_rule: ModelRule | None,
    ) -> str | None:
        """Pick the export task, preferring the most specific registry rule."""
        if model_rule is not None and model_rule.export is not None and model_rule.export.task:
            return model_rule.export.task
        if arch_rule is not None and arch_rule.export.task:
            return arch_rule.export.task
        return _runtime_task(metadata.task)

    @staticmethod
    def _remote_code_required(
        metadata: ModelMetadata,
        arch_rule: ArchitectureRule | None,
        model_rule: ModelRule | None,
    ) -> bool:
        """Whether conversion cannot proceed without executing repository code.

        A repository's ``auto_map`` means remote code *unless* the registry
        declares that Transformers implements the architecture natively: then the
        repository's code is ignored with ``trust_remote_code`` off and never run.
        """
        spec = (
            model_rule.export
            if model_rule is not None and model_rule.export is not None
            else arch_rule.export
            if arch_rule is not None
            else None
        )
        if spec is not None and spec.trust_remote_code:
            return True
        if metadata.requires_remote_code:
            return not (spec is not None and spec.native_implementation)
        return False

    def _expected_repairs(self) -> list[RepairRule]:
        """Registry repairs that are likely to apply to a fresh export.

        No artifact exists yet, so the file *names* an export produces are matched
        without a content probe. The repair implementation re-checks the real file
        before touching it, so a false positive here only ever over-warns.
        """
        return self._registry.repairs_for(files=EXPECTED_EXPORT_FILES, contents_probe=None)

    def _existing_under_other_profiles(
        self,
        artifact_dir: Path,
        profile: ConversionProfile,
        selection: ProfileSelection,
    ) -> list[str]:
        """Say so when this revision is already prepared for the device under another profile.

        A changed default -- a new size band, a re-prioritised profile --
        produces a new artifact identity, never a silent replacement, so the
        old artifact stays and a fresh export starts. That is right, but a user
        who ran the same ``prepare`` last week deserves to hear it before the
        download, with the flag that keeps the old one.
        """
        if selection.basis == "explicit" or not artifact_dir.parent.is_dir():
            return []
        prefix = artifact_dir.name.split("--", 1)[0] + "--"
        notes: list[str] = []
        for sibling in sorted(artifact_dir.parent.iterdir()):
            if sibling == artifact_dir or not sibling.name.startswith(prefix):
                continue
            manifest = self._artifacts.manifest_for(sibling)
            if manifest is None or not manifest.ready or manifest.export.profile == profile.id:
                continue
            other = self._registry.registry.profile(manifest.export.profile)
            if other is None or other.device != profile.device:
                continue
            if not self._could_be_automatic(other):
                continue  # an eval reference, never a substitute for the default
            if not self._artifacts.is_ready(sibling):
                continue
            notes.append(
                f"This revision is already prepared as {other.id} ({sibling}). This plan exports "
                f"{profile.id} instead: {selection.sentence} Pass --profile {other.id} to use "
                "the existing artifact."
            )
        return notes

    def _could_be_automatic(self, rule: ProfileRule) -> bool:
        """Whether ``rule`` can be chosen without ``--profile`` for some model.

        True for a banded profile, for one an architecture or model rule names,
        and for an unbanded profile ranking at least with its device's
        unconditional default. The reference profiles (``cpu-int8-asym-ref``,
        ``cpu-fp16-ref``) sit far below that on purpose and are excluded.
        """
        registry = self._registry.registry
        if not rule.status.is_usable:
            return False
        if rule.applies_to.parameter_count is not None:
            return True
        carriers: list[ArchitectureRule | ModelRule] = [*registry.architectures, *registry.models]
        if any(rule.id in carrier.profiles.values() for carrier in carriers):
            return True
        defaults = [
            other.priority
            for other in registry.profiles
            if other.device == rule.device
            and other.status.is_usable
            and other.applies_to.is_generic
        ]
        return bool(defaults) and rule.priority >= max(defaults)

    def _reusable_artifact(
        self,
        artifact_dir: Path,
        source_sha: str | None,
        profile_id: str,
    ) -> Path | None:
        """Return an existing ready artifact for this exact source and profile.

        ``docs/MODEL_PIPELINE.md`` section 15: an identical source revision and
        profile that is already ready is returned rather than re-exported, unless
        ``--force`` was passed.
        """
        manifest = self._artifacts.manifest_for(artifact_dir)
        if manifest is None or not manifest.ready:
            return None
        if manifest.export.profile != profile_id:
            return None
        recorded = manifest.resolved_source.resolved_revision
        if source_sha is not None and recorded is not None and recorded != source_sha:
            return None
        if not self._artifacts.is_ready(artifact_dir):
            return None
        return artifact_dir

    def _confirmations(
        self,
        *,
        resolved_source: ResolvedSource,
        remote_code_required: bool,
        download_bytes: int,
        expected_repairs: Sequence[str],
        artifact_dir: Path,
        reuse: Path | None,
        force: bool,
        export_environment: EnvironmentSpec | None = None,
    ) -> list[ConfirmationReason]:
        """Collect everything the user has to agree to before work starts."""
        reasons: list[ConfirmationReason] = []
        if export_environment is not None:
            pins = ", ".join(
                f"{name}{specifier}"
                for name, specifier in export_environment.packages.items()
                if name == "transformers" and specifier
            )
            reasons.append(
                ConfirmationReason(
                    code=CONFIRMATION_MANAGED_ENVIRONMENT,
                    message=(
                        f"A managed export environment ({pins or export_environment.name}) will be "
                        f"built under {self._settings.paths.envs_dir}: up to about "
                        f"{format_bytes(ESTIMATED_ENVIRONMENT_BYTES)} of packages from PyPI and "
                        "the PyTorch CPU index. It is used only by the export, is reused by "
                        "later jobs, and your own Python environment is not changed."
                    ),
                )
            )
        if resolved_source.requires_confirmation or resolved_source.is_derivative:
            reasons.append(
                ConfirmationReason(
                    code=CONFIRMATION_THIRD_PARTY_DERIVATIVE,
                    message=(
                        f"{resolved_source.source.repo_id} will be converted instead of "
                        f"{resolved_source.requested.repo_id}. It is published by a third "
                        "party and yonga cannot vouch for its contents."
                    ),
                )
            )
        if remote_code_required:
            reasons.append(
                ConfirmationReason(
                    code=CONFIRMATION_REMOTE_CODE,
                    message=(
                        f"{resolved_source.source.repo_id} can only be converted by executing "
                        "Python code from the repository on this machine."
                    ),
                    security_sensitive=True,
                )
            )
        threshold = int(self._settings.download_confirm_threshold_bytes)
        if download_bytes > threshold:
            reasons.append(
                ConfirmationReason(
                    code=CONFIRMATION_LARGE_DOWNLOAD,
                    message=(
                        f"About {format_bytes(download_bytes)} will be downloaded, above the "
                        f"{format_bytes(threshold)} confirmation threshold."
                    ),
                )
            )
        if expected_repairs:
            reasons.append(
                ConfirmationReason(
                    code=CONFIRMATION_REPAIR,
                    message=(
                        "This export is expected to need a known artifact repair: "
                        + ", ".join(expected_repairs)
                    ),
                )
            )
        # A device fallback is *not* confirmed here. It is confirmed at the moment it
        # actually happens (see ``_probe``), because at planning time there is nothing
        # concrete to agree to: the plan already prints
        # ``compatibility.fallback_candidates``, and asking up front would train the
        # user to approve a switch that may never be needed.
        if reuse is None and artifact_dir.exists() and not is_tool_owned_incomplete(artifact_dir):
            reasons.append(
                ConfirmationReason(
                    code=CONFIRMATION_OVERWRITE_EXISTING,
                    message=(
                        f"{artifact_dir} already exists and will be replaced"
                        + (" because --force was passed." if force else ".")
                    ),
                )
            )
        return reasons

    # ==================================================================
    # execution
    # ==================================================================

    def prepare(
        self,
        plan: PreparationPlan,
        *,
        confirm: ConfirmCallback,
        ctx: JobContext,
        allow_repairs: bool = True,
        ignore_exporter_requirements: bool = False,
    ) -> PreparationOutcome:
        """Execute a plan through the full preparation state machine.

        Advances ``ctx`` at every transition and emits a ``STAGE`` progress event
        per step. Any typed failure is recorded against the stage it happened in
        and re-raised, so the CLI can render the diagnostic and exit with its
        stable code.

        Args:
            plan: The plan to execute, as returned by :meth:`plan`.
            confirm: Callback asked to approve each :class:`ConfirmationReason`.
            ctx: The open job context this preparation runs inside.
            allow_repairs: Whether known-defect repairs may modify the export.
            ignore_exporter_requirements: Proceed even when the registry's
                exporter version pins are unmet. Defaults to ``False``; the CLI
                only sets it from its own explicit flag, never from ``--yes``.
        """
        tracker = _StageTracker()
        try:
            return self._execute(
                plan,
                confirm=confirm,
                ctx=ctx,
                allow_repairs=allow_repairs,
                tracker=tracker,
                ignore_exporter_requirements=ignore_exporter_requirements,
            )
        except YongaError as error:
            self._clean_up_staging(ctx)
            ctx.fail(tracker.stage, error.diagnostic)
            raise

    @staticmethod
    def _clean_up_staging(ctx: JobContext) -> None:
        """Apply the staging retention policy to a failed preparation.

        ``SPEC.md`` section 16 wants raw diagnostics retained, so a staging
        directory holding a partial export survives and its path is announced —
        :meth:`JobContext.fail` then records it in the job state and in the
        failure diagnostic. A directory that produced no file is removed on the
        spot: an empty husk diagnoses nothing, and leaving one behind per failed
        run is how the hidden staging root filled up.
        """
        retained = ctx.retain_staging_for_diagnosis()
        if retained is None:
            return
        ctx.emit(
            ProgressKind.LOG,
            f"Partial output kept for diagnosis at {retained}",
            staging_dir=str(retained),
        )

    def _execute(
        self,
        plan: PreparationPlan,
        *,
        confirm: ConfirmCallback,
        ctx: JobContext,
        allow_repairs: bool,
        tracker: _StageTracker,
        ignore_exporter_requirements: bool = False,
    ) -> PreparationOutcome:
        """Run the state machine. See :meth:`prepare` for the caller contract."""
        started = time.perf_counter()
        source_meta = plan.resolved_source.metadata

        self._stage(ctx, 1)
        tracker.stage = JobStage.INSPECTED
        ctx.advance(JobStage.INSPECTED)

        self._stage(ctx, 2)
        tracker.stage = JobStage.SOURCE_RESOLVED
        ctx.advance(JobStage.SOURCE_RESOLVED)
        for reason in plan.resolved_source.reasons:
            ctx.emit(ProgressKind.LOG, reason)

        self._guard_remote_code(plan)
        self._approve(plan, confirm)

        reused = self._reuse(plan, ctx, started)
        if reused is not None:
            return reused

        self._stage(ctx, 3)
        tracker.stage = JobStage.PREFLIGHT_OK
        warnings = self._preflight(
            plan, ctx, ignore_exporter_requirements=ignore_exporter_requirements
        )
        staging = self._staging_dir(plan, ctx)
        adopted = self._adopt_export(plan, staging, ctx)
        # Built at preflight, before the model download: a failed build then costs
        # minutes, not a multi-gigabyte download that was never going to be used.
        environment = None if adopted is not None else self._ensure_export_environment(plan, ctx)
        ctx.advance(JobStage.PREFLIGHT_OK)

        self._stage(ctx, 4)
        tracker.stage = JobStage.DOWNLOADED
        if adopted is not None:
            ctx.emit(
                ProgressKind.DOWNLOAD,
                "Not needed: the export is reused",
                status="skipped",
            )
        snapshot = None if adopted is not None else self._download(plan, ctx)
        ctx.advance(JobStage.DOWNLOADED)

        self._stage(ctx, 5)
        tracker.stage = JobStage.EXPORTED
        if adopted is not None or snapshot is None:
            export = self._reused_export_outcome(adopted, plan.profile_selection)
            ctx.emit(
                ProgressKind.EXPORT,
                f"Reusing the completed export of job {export.record.reused_from_job}",
                status="reused",
            )
        else:
            export = self._convert(plan, snapshot, staging, ctx, environment=environment)
        warnings.extend(export.notes)
        ctx.advance(JobStage.EXPORTED)

        self._stage(ctx, 6)
        tracker.stage = JobStage.VALIDATING
        validation, repairs = self._validate_with_repairs(
            plan, staging, ctx, confirm=confirm, allow_repairs=allow_repairs, tracker=tracker
        )

        self._stage(ctx, 7)
        tracker.stage = JobStage.PROBED
        probes, device, fallbacks = self._probe(
            plan, staging, validation, ctx, confirm=confirm, source_meta=source_meta
        )
        ctx.advance(JobStage.PROBED)
        warnings.extend(
            f"Fell back from {previous} to {nxt}." for previous, nxt in pairwise(fallbacks)
        )

        # From here on nothing runs against the staging directory any more, and
        # it is about to be renamed onto the artifact directory. Rewrite the
        # paths it produced before they are frozen into the manifest and into
        # ``.yonga/validation.json``.
        validation = _relocated_validation(validation, staging)
        probes = {name: _relocated_device_probe(report, staging) for name, report in probes.items()}
        manifest = self._build_manifest(
            plan=plan,
            export=_relocated_export(export.record, staging),
            validation=validation,
            repairs=repairs,
            probes=probes,
            device=device,
            source_meta=source_meta,
            notes=warnings,
        )
        final = self._artifacts.finalize(
            staging,
            plan.artifact_dir,
            manifest,
            force=plan.force,
            progress=ctx.progress,
            validation=validation,
        )
        self._advance_ready(ctx, plan, final, device)
        ctx.emit(
            ProgressKind.STAGE,
            f"Artifact ready at {final}",
            current=TOTAL_STAGES,
            total=TOTAL_STAGES,
            status="ready",
            severity=Severity.OK,
        )
        return PreparationOutcome(
            job_id=ctx.job_id,
            artifact_dir=final,
            manifest=manifest,
            device=device,
            reused=False,
            fallbacks=fallbacks[1:],
            warnings=warnings,
            repairs=list(repairs),
            validation=validation,
            duration_seconds=time.perf_counter() - started,
        )

    # --- stage helpers ----------------------------------------------------

    @staticmethod
    def _stage(ctx: JobContext, number: int) -> None:
        """Emit the stage event for step ``number``.

        The message is the bare label. The ``current``/``total`` pair is what
        carries the position, and the renderer turns it into the ``"4/7"``
        prefix of ``docs/CLI_SPEC.md`` section 6 — putting the counter in the
        message too printed it twice.
        """
        ctx.emit(
            ProgressKind.STAGE,
            STAGE_LABELS[number - 1],
            current=number,
            total=TOTAL_STAGES,
        )

    @staticmethod
    def _advance_ready(ctx: JobContext, plan: PreparationPlan, final: Path, device: str) -> None:
        """Record the terminal ``ready`` transition together with what it produced."""
        ctx.logger.info(
            "job.stage",
            f"stage -> {JobStage.READY.value}",
            stage=JobStage.READY.value,
            artifact_dir=str(final),
        )
        if ctx.state is None:
            return
        ctx.state.advance(
            JobStage.READY,
            artifact_dir=final,
            profile_id=plan.profile.id,
            target_device=device,
        )

    @staticmethod
    def _guard_remote_code(plan: PreparationPlan) -> None:
        """Stop before any work when repository code would have to run unapproved.

        This is deliberately not a prompt. ``--yes`` must not be able to reach it,
        and neither must a confirmation callback: only the explicit
        ``--trust-remote-code`` flag enables it (``SPEC.md`` section 18.1).
        """
        if not plan.blocked:
            return
        raise RemoteCodeRequiredError(
            diagnostic(
                "source.remote_code_required",
                FailureCategory.REMOTE_CODE_REQUIRED,
                "Remote code approval required",
                (
                    f"{plan.resolved_source.source.repo_id} can only be converted by executing "
                    "Python code published in the repository. yonga will not do that unless you "
                    "ask for it explicitly."
                ),
                remediation=[
                    "Review the repository's custom modelling code first.",
                    (
                        "Then re-run with the explicit flag: "
                        f"yonga prepare {plan.requested.repo_id} --trust-remote-code"
                    ),
                    "--yes does not and will not enable remote code.",
                ],
                context={"repo_id": plan.resolved_source.source.repo_id},
            )
        )

    @staticmethod
    def _approve(plan: PreparationPlan, confirm: ConfirmCallback) -> None:
        """Ask for every confirmation the plan collected; a refusal cancels the job."""
        for reason in plan.confirmations:
            if confirm(reason):
                continue
            raise CancelledError(
                diagnostic(
                    "job.declined",
                    FailureCategory.USER_CANCELLED,
                    "Preparation cancelled",
                    f"Confirmation {reason.code!r} was declined: {reason.message}",
                    context={"confirmation": reason.code},
                )
            )

    def _reuse(
        self,
        plan: PreparationPlan,
        ctx: JobContext,
        started: float,
    ) -> PreparationOutcome | None:
        """Return the existing ready artifact when the plan says it can be reused."""
        existing = plan.reuse_existing_artifact
        if existing is None or plan.force:
            return None
        manifest = self._artifacts.manifest_for(existing)
        if manifest is None or not manifest.ready:
            return None
        device = (
            manifest.recommended_runtime.device
            if manifest.recommended_runtime is not None
            else plan.target_device
        )
        message = (
            f"An artifact for {plan.resolved_source.source.repo_id} at this revision and "
            f"profile {plan.profile.id} is already ready; reusing {existing}."
        )
        ctx.emit(ProgressKind.LOG, message, status="reused", severity=Severity.OK)
        ctx.logger.info("prepare.reused", message, artifact_dir=str(existing))
        self._advance_ready(ctx, plan, existing, device)
        return PreparationOutcome(
            job_id=ctx.job_id,
            artifact_dir=existing,
            manifest=manifest,
            device=device,
            reused=True,
            validation=manifest.validation,
            duration_seconds=time.perf_counter() - started,
        )

    # --- preflight --------------------------------------------------------

    def _preflight(
        self,
        plan: PreparationPlan,
        ctx: JobContext,
        *,
        ignore_exporter_requirements: bool = False,
    ) -> list[str]:
        """Point every temp variable at managed storage and refuse to start if tight.

        Returns non-fatal warnings worth recording in the manifest. Raises
        :class:`~yonga.domain.diagnostics.PreflightError` when the managed temp
        or the artifact filesystem clearly cannot hold the job (exit code 30),
        when the exporter's declared package pins are unmet (exit code 10), and
        :class:`~yonga.domain.diagnostics.DeviceUnsupportedError` when an
        explicitly requested device is not present at all.
        """
        warnings: list[str] = []
        temp_dir = ctx.temp_dir
        system_temp = system_temp_dir()
        try:
            resolved_temp = temp_dir.expanduser().resolve()
            resolved_system = system_temp.expanduser().resolve()
        except OSError:  # pragma: no cover - defensive
            resolved_temp, resolved_system = temp_dir, system_temp
        if resolved_temp == resolved_system:
            raise PreflightError(
                diagnostic(
                    "preflight.temp_is_system_temp",
                    FailureCategory.CONFIGURATION_ERROR,
                    "Managed temporary directory resolves to the system temp",
                    (
                        f"The job temp directory resolves to {resolved_system} itself. "
                        "yonga converts in its own managed, per-job directory so a small "
                        "/tmp tmpfs cannot kill a multi-gigabyte export."
                    ),
                    remediation=[
                        "Point YONGA_CACHE_DIR at a directory on a real disk.",
                    ],
                    context={"temp_dir": str(resolved_temp)},
                )
            )

        temp_dir.mkdir(parents=True, exist_ok=True)
        ctx.temp_env.update(dict.fromkeys(TEMP_ENV_KEYS, str(temp_dir)))
        ctx.logger.info(
            "preflight.temp",
            "conversion temp redirected to managed storage",
            temp_dir=str(temp_dir),
            system_temp=str(system_temp),
        )
        for note in plan.disk.temp.notes:
            ctx.emit(ProgressKind.LOG, note)
        if plan.disk.temp.system_temp_would_fail:
            warnings.append(
                f"The system temporary directory {system_temp} is too small for this "
                "conversion; the managed directory was used instead."
            )
        if plan.disk.temp.temp_filesystem.is_ram_backed:
            # Not fatal on its own: the space check below is what actually decides.
            # Saying so is still worth a note, because a RAM-backed cache directory is
            # almost always a misconfiguration rather than a choice.
            warnings.append(
                f"The managed temporary directory sits on a RAM-backed "
                f"{plan.disk.temp.temp_filesystem.fstype} filesystem; consider pointing "
                "YONGA_CACHE_DIR at persistent storage."
            )

        self._require_space(plan.disk)
        self._require_device(plan)
        warnings.extend(
            self._require_exporter_requirements(plan, ctx, ignored=ignore_exporter_requirements)
        )
        return warnings

    def _require_exporter_requirements(
        self,
        plan: PreparationPlan,
        ctx: JobContext,
        *,
        ignored: bool,
    ) -> list[str]:
        """Refuse to convert when the exporter's declared package pins are unmet.

        ``docs/MODEL_PIPELINE.md`` section 5 counts package compatibility among
        the things preflight computes, and ``SPEC.md`` section 10 says preflight
        exists to fail before expensive work. An unmet pin is knowledge that the
        export cannot succeed, so spending the download and the quantization to
        rediscover it several minutes later is exactly the waste this stage is
        supposed to prevent.

        The decision is keyed on :attr:`ExporterRequirements.problems` and never
        on a bare :func:`~yonga.util.versions.satisfies` result: that function
        also answers ``False`` for a version it cannot parse, and "unknown" must
        not masquerade as "wrong" (see :func:`requirement_state`).

        Returns the warnings to record when the block was explicitly overridden.
        """
        # Keyed on the plan's own property so the preflight gate and the plan
        # panel cannot drift: an already-OpenVINO source is copied rather than
        # converted (see :meth:`_convert`), and blocking it would refuse a job
        # that has no export to fail.
        if not plan.exporter_requirements_block:
            return []

        unmet = self._unmet_requirements(plan)
        if ignored:
            warnings: list[str] = []
            for problem in plan.exporter.problems:
                # The manifest is the artifact's lineage: an artifact built past a
                # declared pin must say so, naming the requirement and the version
                # that was actually installed, or nobody can explain it later.
                note = (
                    f"Exporter requirement overridden with "
                    f"{IGNORE_EXPORTER_REQUIREMENTS_FLAG}: {problem}"
                )
                warnings.append(note)
                ctx.emit(ProgressKind.LOG, note, severity=Severity.WARNING)
                ctx.logger.warning("preflight.exporter_requirement_overridden", problem)
            return warnings

        for problem in plan.exporter.problems:
            ctx.logger.error("preflight.exporter_requirement", problem)
        commands = [f"pip install '{name}{specifier}'" for name, specifier, _ in unmet]
        remediation = [
            *(f"Install the required version yourself: {command}" for command in commands),
            (
                "yonga never installs, upgrades or removes packages in your environment. "
                "Without --no-managed-env (and with managed_export_envs left on) it runs "
                "this export in an isolated environment it builds instead."
            ),
            (
                f"If you know this pin is stale, re-run with {IGNORE_EXPORTER_REQUIREMENTS_FLAG}: "
                f"yonga prepare {plan.requested.repo_id} {IGNORE_EXPORTER_REQUIREMENTS_FLAG}"
            ),
            (
                "--yes does not and will not bypass this check, only "
                f"{IGNORE_EXPORTER_REQUIREMENTS_FLAG} does."
            ),
        ]
        # PACKAGE_ABI_MISMATCH is the honest category. Every package is present and
        # importable; what is wrong is the combination of their versions — the
        # Qwen3.5 export dies because optimum-intel 2.2.0 imports
        # Qwen3_5DynamicCache, a symbol transformers 5.5.4 dropped although it
        # still loads Qwen3_5Config and reports model_type qwen3_5. It is not
        # INSUFFICIENT_DISK (no resource is short), not CONFIGURATION_ERROR or
        # REGISTRY_ERROR (nothing the user or the registry wrote is wrong) and not
        # SOURCE_FORMAT_UNSUPPORTED (the model is fine). That category maps to
        # ExitCode.ENVIRONMENT_UNUSABLE (10) rather than PREFLIGHT (30): 10 is the
        # right answer to *why* the job stopped even though it stopped *in*
        # preflight, so neither the category nor the table is bent to match.
        raise PreflightError(
            diagnostic(
                "preflight.exporter_requirement_unsatisfied",
                FailureCategory.PACKAGE_ABI_MISMATCH,
                "Exporter requirements are not satisfied",
                (
                    f"Converting {plan.resolved_source.source.repo_id} needs "
                    f"{self._requirement_summary(plan, unmet)} yonga stops here, before the "
                    f"{format_bytes(plan.disk.expected_download_bytes)} download and the "
                    "conversion, because this export cannot succeed as the environment stands."
                ),
                remediation=remediation,
                environment_broken=True,
                context={
                    "requirements": "; ".join(plan.exporter.problems)[:512],
                    "profile": plan.profile.id,
                },
            )
        )

    def _unmet_requirements(self, plan: PreparationPlan) -> list[tuple[str, str, str | None]]:
        """Return ``(distribution, specifier, installed)`` for every blocking pin."""
        versions = self._exporter.requirements()
        return [
            (name, specifier, versions.get(name))
            for name, specifier in plan.exporter.specifiers().items()
            if requirement_state(versions.get(name), specifier) in _BLOCKING_REQUIREMENT_STATES
        ]

    @staticmethod
    def _requirement_summary(
        plan: PreparationPlan,
        unmet: Sequence[tuple[str, str, str | None]],
    ) -> str:
        """Render one sentence naming each unmet pin and what is installed instead.

        Falls back to the plan's recorded problem text when the live environment
        no longer explains the plan — a plan is data and may outlive the versions
        it was computed against, and a diagnostic must never invent a detail.
        """
        if not unmet:
            return "; ".join(plan.exporter.problems) + "."
        parts = [
            (
                f"{name} {specifier}, but {installed} is installed"
                if installed
                else f"{name} {specifier}, but {name} is not installed"
            )
            for name, specifier, installed in unmet
        ]
        return "; ".join(parts) + "."

    @staticmethod
    def _require_space(disk: DiskPlan) -> None:
        """Fail preflight before any download when the disk plan is insufficient."""
        if disk.temp.sufficient and disk.artifact_sufficient:
            return
        if not disk.temp.sufficient:
            title = "Not enough temporary space"
            message = (
                f"The managed temporary directory {disk.temp.job_temp_dir} has "
                f"{format_bytes(disk.temp.usable_free_bytes)} free, but this conversion "
                f"needs {format_bytes(disk.temp.total_required_bytes)} "
                f"({format_bytes(disk.temp.required_bytes)} plus a "
                f"{format_bytes(disk.temp.safety_margin_bytes)} safety margin)."
            )
            remediation = [
                "Free space on the cache filesystem.",
                "Or point YONGA_CACHE_DIR at a larger filesystem.",
            ]
        else:
            title = "Not enough space for the artifact"
            message = (
                f"The artifact filesystem has {format_bytes(disk.artifact_free_bytes)} free, "
                f"which is not enough for an estimated "
                f"{format_bytes(disk.estimated_artifact_bytes or 0)} artifact plus "
                f"{format_bytes(disk.expected_download_bytes)} of downloads."
            )
            remediation = [
                "Free space on the data filesystem.",
                "Or point YONGA_DATA_DIR at a larger filesystem.",
            ]
        raise PreflightError(
            diagnostic(
                "preflight.insufficient_disk",
                FailureCategory.INSUFFICIENT_DISK,
                title,
                message,
                remediation=remediation,
                context={"notes": " ".join(disk.notes)[:512]},
            )
        )

    @staticmethod
    def _require_device(plan: PreparationPlan) -> None:
        """Refuse an explicitly requested device OpenVINO does not expose at all."""
        decision = plan.compatibility
        if not decision.explicit_request:
            return
        available = plan.hardware.available_device_names
        if plan.target_device in available:
            return
        alternatives = [name for name in decision.fallback_candidates if name in available]
        remediation = ["Run 'yonga doctor' to see why the device is not available."]
        remediation.extend(
            f"Or prepare for another device: yonga prepare {plan.requested.repo_id} "
            f"--device {name.lower()}"
            for name in alternatives
        )
        raise DeviceUnsupportedError(
            diagnostic(
                "device.not_present",
                FailureCategory.DEVICE_UNAVAILABLE,
                f"{plan.target_device} is not available",
                (
                    f"OpenVINO reports {', '.join(available) or 'no devices'} on this host, "
                    f"so {plan.target_device} cannot be used."
                ),
                remediation=remediation,
                fallback_devices=alternatives,
                context={"requested_device": plan.target_device},
            )
        )

    # --- download and conversion -----------------------------------------

    def _export_fingerprint(self, plan: PreparationPlan) -> str:
        """Digest what determines this plan's export output; see ``export_reuse``."""
        source = plan.resolved_source.source
        metadata = plan.resolved_source.metadata
        environment = plan.export_environment
        return export_reuse.export_fingerprint(
            source_repo_id=source.repo_id,
            source_revision=(metadata.resolved_revision if metadata else None) or source.revision,
            task=plan.task,
            profile=plan.profile.model_dump(mode="json"),
            trust_remote_code=plan.trust_remote_code_enabled,
            backend=_EXPORT_BACKEND,
            environment_key=None if environment is None else environment.key,
            package_versions={} if environment is not None else self._package_versions(),
        )

    def _adopt_export(
        self, plan: PreparationPlan, staging: Path, ctx: JobContext
    ) -> ExportCheckpoint | None:
        """Take over the plan's reusable export, if it has one and it checks out."""
        candidate = plan.reusable_export
        if candidate is None:
            return None
        adopted = export_reuse.adopt(candidate, staging, ctx)
        if adopted is None:
            ctx.emit(
                ProgressKind.LOG,
                f"The retained export of job {candidate.job_id} no longer matches; exporting.",
                severity=Severity.WARNING,
            )
            return None
        return adopted

    @staticmethod
    def _reused_export_outcome(
        checkpoint: ExportCheckpoint | None, selection: ProfileSelection | None
    ) -> _ExportOutcome:
        """The export stage's result when an earlier job's export was adopted.

        The weights are the earlier job's, but why the profile was chosen is
        this job's decision: the earlier job may have named the same profile
        with --profile where this one reached it by its size band.
        """
        assert checkpoint is not None  # guarded by the caller: no snapshot means adopted
        record = checkpoint.record.model_copy(
            update={"reused_from_job": checkpoint.job_id, "profile_selection": selection}
        )
        finished = checkpoint.completed_at.strftime("%Y-%m-%d %H:%M UTC")
        return _ExportOutcome(
            record=record,
            notes=[
                f"The export was reused from failed job {checkpoint.job_id}, which completed "
                f"it at {finished} with identical inputs; it was not run again."
            ],
        )

    def _download(self, plan: PreparationPlan, ctx: JobContext) -> Path:
        """Materialise the pinned source revision in the Hugging Face cache."""
        source = plan.resolved_source.source
        ctx.emit(
            ProgressKind.DOWNLOAD,
            f"Fetching {source} ({format_bytes(plan.disk.expected_download_bytes)})",
            status="started",
        )
        gguf_file = plan.resolved_source.gguf_file
        if gguf_file is not None:
            # A GGUF repository publishes one file per quantization; fetch only ours.
            snapshot = self._hub.download(source, ctx=ctx, allow_patterns=[gguf_file])
        else:
            snapshot = self._hub.download(source, ctx=ctx)
        ctx.logger.info("download.complete", "source snapshot ready", path=str(snapshot))
        return snapshot

    def _staging_dir(self, plan: PreparationPlan, ctx: JobContext) -> Path:
        """Return the staging directory this run exports into.

        The plan's staging directory is honoured when it belongs to the same job;
        otherwise the context wins, because writing into another job's staging
        area would break that job's cleanup.
        """
        if plan.job_id == ctx.job_id:
            return plan.staging_dir
        return self._artifacts.staging_model_dir(ctx.job_id)

    def _ensure_export_environment(
        self, plan: PreparationPlan, ctx: JobContext
    ) -> ExportEnvironment | None:
        """Build (or reuse) the managed environment the plan chose, if any."""
        spec = plan.export_environment
        if spec is None:
            return None
        if self._environments is None:
            raise PreflightError(
                diagnostic(
                    "export.env.unavailable",
                    FailureCategory.CONFIGURATION_ERROR,
                    "Managed export environments are unavailable",
                    "The plan needs a managed export environment, but this service was "
                    "built without an environment manager.",
                )
            )
        if not plan.export_environment_ready:
            self._require_environment_space(plan)
        environment = self._environments.ensure(spec, ctx)
        environment = self._within_declared_range(plan, environment, ctx)
        ctx.emit(
            ProgressKind.LOG,
            f"Exporting in managed environment {environment.spec.name}",
            key=environment.key,
        )
        return environment

    def _within_declared_range(
        self, plan: PreparationPlan, environment: ExportEnvironment, ctx: JobContext
    ) -> ExportEnvironment:
        """Swap in an environment whose Transformers the exporter will accept.

        Planning reads optimum-intel's Transformers range for the model type
        from the optimum-intel installed beside yonga. With none installed there
        was nobody to ask, so the range is read from the environment just built
        and, if its Transformers falls outside, one satisfying the range is
        built instead -- still before the download, where refusing is cheap.
        """
        optimum_intel = environment.installed.get("optimum-intel")
        if (
            plan.exporter.transformers is not None
            or not plan.architecture
            or not optimum_intel
            or not isinstance(self._exporter, DeclaresRequirementsIn)
            or self._environments is None
        ):
            return environment
        declared = self._exporter.declared_requirements_in(
            environment.python, optimum_intel, plan.architecture, plan.task
        )
        if declared is None or satisfies(
            environment.installed.get("transformers"), declared.transformers
        ):
            return environment
        requirements = plan.exporter.model_copy(update={"transformers": declared.transformers})
        spec = self._environments.spec_for(
            requirements,
            reason=f"transformers {declared.transformers} (declared by {declared.source})",
        )
        ctx.logger.info(
            "export.env.declared_range",
            "the environment's transformers is outside the exporter's declared range",
            installed=environment.installed.get("transformers"),
            declared=declared.transformers,
            key=spec.key,
        )
        if self._environments.find(spec) is None:
            self._require_environment_space(plan)
        return self._environments.ensure(spec, ctx)

    def _require_environment_space(self, plan: PreparationPlan) -> None:
        """Refuse to build an environment the cache filesystem cannot also hold.

        The environment and the export's managed temp directory both live under
        the cache directory, and both are needed at once. Temp space was already
        checked on its own; when the environment shares that filesystem it has
        to fit *on top of* the temp requirement, or the export would run out of
        room halfway through.
        """
        envs_dir = self._settings.paths.envs_dir
        usage = self._filesystem.usage(envs_dir)
        free = int(getattr(usage, "available_bytes", 0) or getattr(usage, "free_bytes", 0) or 0)
        if not free:
            return  # unknown: the filesystem probe already reported it as a warning
        needed = ESTIMATED_ENVIRONMENT_BYTES
        temp = plan.disk.temp.temp_filesystem
        shared = temp.mount_point is not None and temp.mount_point == getattr(
            usage, "mount_point", None
        )
        if shared:
            needed += plan.disk.temp.total_required_bytes
        if free >= needed:
            return
        also = " alongside the export's temporary files" if shared else ""
        raise PreflightError(
            diagnostic(
                "export.env.insufficient_disk",
                FailureCategory.INSUFFICIENT_DISK,
                "Not enough space to build the export environment",
                f"About {format_bytes(ESTIMATED_ENVIRONMENT_BYTES)} is needed under {envs_dir}"
                f"{also} ({format_bytes(needed)} in all); {format_bytes(free)} is free.",
                remediation=["Free space, or point YONGA_CACHE_DIR at a bigger disk."],
            )
        )

    def _convert(
        self,
        plan: PreparationPlan,
        snapshot: Path,
        staging: Path,
        ctx: JobContext,
        *,
        environment: ExportEnvironment | None = None,
    ) -> _ExportOutcome:
        """Produce OpenVINO IR in ``staging``, by conversion or by copying IR."""
        mark_incomplete(staging, job_id=ctx.job_id)
        if plan.resolved_source.already_openvino:
            return self._copy_openvino_source(plan, snapshot, staging, ctx)
        if plan.resolved_source.gguf_file is not None:
            return self._import_gguf(plan, snapshot, staging, ctx)
        return self._export(plan, snapshot, staging, ctx, environment=environment)

    def _import_gguf(
        self,
        plan: PreparationPlan,
        snapshot: Path,
        staging: Path,
        ctx: JobContext,
    ) -> _ExportOutcome:
        """Import the chosen GGUF file into staging with OpenVINO GenAI's reader.

        The Hub cache is only read: the importer links the file into staging and
        GenAI writes the IR beside the link. What comes out is validated,
        repaired and probed exactly like an optimum-intel export.
        """
        source = plan.resolved_source
        assert source.gguf_file is not None  # guarded by the caller
        if self._gguf_importer is None:
            raise ExportError(
                diagnostic(
                    "export.gguf_unavailable",
                    FailureCategory.CONFIGURATION_ERROR,
                    "GGUF import is unavailable",
                    "This preparation service was built without a GGUF importer.",
                )
            )
        started = _now()
        result = self._gguf_importer.import_file(snapshot / source.gguf_file, staging, ctx)
        if staging.is_dir():
            mark_incomplete(staging, job_id=ctx.job_id)
        record = ExportRecord(
            profile_selection=plan.profile_selection,
            backend=_GGUF_BACKEND,
            profile=_artifact_key(source, plan.profile),
            arguments={
                "gguf_file": source.gguf_file,
                "gguf_architecture": source.gguf_architecture,
                "tensor_types": dict(source.gguf_tensor_types),
                "task": plan.task,
                "weight_format": "gguf",
            },
            argv=redact_argv(result.argv),
            package_versions=result.package_versions,
            duration_seconds=result.duration_seconds,
            started_at=started,
        )
        if not result.success:
            report = result.diagnostic or diagnostic(
                "export.gguf_import_failed",
                FailureCategory.UNKNOWN_RUNTIME_ERROR,
                "The GGUF import failed",
                "OpenVINO GenAI did not produce an OpenVINO model.",
            )
            ctx.logger.error("export.failed", report.message, raw=report.raw_excerpt or "")
            raise ExportError(report)
        ctx.logger.info(
            "export.complete",
            "GGUF import finished",
            gguf_file=source.gguf_file,
            duration_seconds=result.duration_seconds,
        )
        return _ExportOutcome(record=record, result=result)

    def _copy_openvino_source(
        self,
        plan: PreparationPlan,
        snapshot: Path,
        staging: Path,
        ctx: JobContext,
    ) -> _ExportOutcome:
        """Copy an upstream OpenVINO IR repository into staging without re-exporting.

        ``docs/MODEL_PIPELINE.md`` section 3, case B. The Hub snapshot is read
        only: files are copied (never linked, never moved) so the shared cache is
        left exactly as it was, and validation then runs against our own copy.
        """
        started = _now()
        ctx.emit(
            ProgressKind.EXPORT,
            f"Using the published OpenVINO IR from {plan.resolved_source.source.repo_id}",
            status="copying",
        )
        begin = time.perf_counter()
        marker = staging / ".yonga-incomplete"
        try:
            shutil.copytree(snapshot, staging, symlinks=False, dirs_exist_ok=True)
        except OSError as exc:
            raise ExportError(
                diagnostic(
                    "export.copy_failed",
                    FailureCategory.UNKNOWN_RUNTIME_ERROR,
                    "The published OpenVINO artifact could not be copied",
                    f"Copying {snapshot} into {staging} failed: {exc}.",
                    remediation=["Check free space and permissions on the data filesystem."],
                    context={"staging": str(staging)},
                ),
                cause=exc,
            ) from exc
        if not marker.exists():
            mark_incomplete(staging, job_id=ctx.job_id)
        record = ExportRecord(
            profile_selection=plan.profile_selection,
            backend=_PASSTHROUGH_BACKEND,
            profile=plan.profile.id,
            arguments={
                "source_repo_id": plan.resolved_source.source.repo_id,
                "already_openvino": True,
            },
            argv=[],
            package_versions=self._package_versions(),
            duration_seconds=time.perf_counter() - begin,
            started_at=started,
        )
        return _ExportOutcome(
            record=record,
            notes=[
                f"{plan.resolved_source.source.repo_id} already publishes OpenVINO IR; "
                "it was copied and validated instead of being re-exported."
            ],
        )

    def _export(
        self,
        plan: PreparationPlan,
        snapshot: Path,
        staging: Path,
        ctx: JobContext,
        *,
        environment: ExportEnvironment | None = None,
    ) -> _ExportOutcome:
        """Run the exporter and turn a failure into a classified diagnostic."""
        started = _now()
        request = ExportRequest(
            source_path=snapshot,
            output_path=staging,
            task=plan.task,
            profile=plan.profile,
            trust_remote_code=plan.trust_remote_code_enabled,
            extra_env=dict(ctx.temp_env),
            model_ref=str(snapshot),
            interpreter=None if environment is None else environment.python,
            environment_versions={} if environment is None else dict(environment.installed),
        )
        result = self._exporter.export(request, ctx)
        # The exporter keeps the marker when it empties its output directory, but
        # the tool it runs owns that directory meanwhile. Re-mark whatever it left
        # behind — on success so an interrupted validation is still recognisable,
        # and on failure so the debris of a failed run is recognisably this job's
        # and not a user directory.
        if staging.is_dir():
            mark_incomplete(staging, job_id=ctx.job_id)
        record = ExportRecord(
            profile_selection=plan.profile_selection,
            backend=_EXPORT_BACKEND,
            profile=plan.profile.id,
            arguments=self._export_arguments(plan),
            argv=redact_argv(result.argv),
            package_versions=result.package_versions or self._package_versions(),
            duration_seconds=result.duration_seconds,
            started_at=started,
            environment_key=None if environment is None else environment.key,
        )
        if result.success:
            ctx.logger.info(
                "export.complete",
                "conversion finished",
                profile=plan.profile.id,
                duration_seconds=result.duration_seconds,
            )
            # Before validation touches anything: the checkpoint describes the
            # export exactly as the exporter left it.
            export_reuse.write_checkpoint(
                staging, fingerprint=self._export_fingerprint(plan), record=record, ctx=ctx
            )
            return _ExportOutcome(record=record, result=result)

        raw = "\n".join([*result.stdout_tail, *result.stderr_tail])
        ctx.logger.error(
            "export.failed",
            "conversion failed",
            returncode=result.returncode,
            profile=plan.profile.id,
            raw=redact_text(raw),
        )
        classified = self._classify(
            raw,
            device=plan.target_device,
            stage="export",
            ctx=ctx,
            metadata=plan.resolved_source.metadata,
            fallback=result.diagnostic,
        )
        raise ExportError(classified)

    @staticmethod
    def _export_arguments(plan: PreparationPlan) -> dict[str, object]:
        """Summarise the conversion settings for the manifest, without any paths."""
        quantization = plan.profile.quantization
        return {
            "task": plan.task,
            "weight_format": quantization.weight_format,
            "symmetric": quantization.symmetric,
            "ratio": quantization.ratio,
            "group_size": quantization.group_size,
            "dataset": quantization.dataset,
            "awq": quantization.awq,
            "scale_estimation": quantization.scale_estimation,
            "all_layers": quantization.all_layers,
            "backup_precision": quantization.backup_precision,
            "extra_args": list(plan.profile.export_extra_args),
            "trust_remote_code": plan.trust_remote_code_enabled,
        }

    def _package_versions(self) -> dict[str, str]:
        """Installed versions of every package the export result depends on."""
        return {
            name: version
            for name, version in self._exporter.requirements().items()
            if version is not None
        }

    # --- validation and repair -------------------------------------------

    def _validate_with_repairs(
        self,
        plan: PreparationPlan,
        staging: Path,
        ctx: JobContext,
        *,
        confirm: ConfirmCallback,
        allow_repairs: bool,
        tracker: _StageTracker,
    ) -> tuple[ValidationReport, list[RepairRecord]]:
        """Validate the export, repairing known defects once before giving up.

        A failing validation is never declared final until registry repair
        detection has run: the ``STRiNG`` tokenizer defect is produced by every
        current export, so failing here without looking would make the tool
        useless on its own reference stack.
        """
        ctx.advance(JobStage.VALIDATING)
        report = self._validate_once(plan, staging, ctx)
        if report.ok:
            return report, []

        rules = self._detect_repairs(staging)
        if not rules:
            raise ValidationError(self._validation_failure(report, plan, ctx))
        names = ", ".join(rule.id for rule in rules)
        if not allow_repairs:
            raise RepairError(
                diagnostic(
                    "repair.declined",
                    FailureCategory.UNKNOWN_RUNTIME_ERROR,
                    "A required repair was disabled",
                    (
                        f"Validation failed and the registry has a matching repair ({names}), "
                        "but repairs were disabled with --no-repair."
                    ),
                    remediation=[
                        "Re-run without --no-repair to apply the repair.",
                        "Or inspect the staged artifact and fix it by hand.",
                    ],
                    context={"rules": names, "staging": str(staging)},
                )
            )

        ctx.emit(ProgressKind.REPAIR, f"Applying registry repairs: {names}", status="started")
        context = RepairContext.for_job(
            ctx,
            staging,
            openvino_version=report.openvino_version,
            tokenizers_version=report.tokenizers_version,
        )
        records = apply_repairs(staging, rules, context, confirm=self._repair_confirm(confirm))
        applied = [record for record in records if record.applied]
        if not applied:
            raise RepairError(
                diagnostic(
                    "repair.not_applied",
                    FailureCategory.UNKNOWN_RUNTIME_ERROR,
                    "The required repair was not applied",
                    (
                        f"Validation failed and repair {names} matched, but it was declined or "
                        "could not run, so the artifact is still unusable."
                    ),
                    remediation=["Approve the repair, or fix the artifact by hand."],
                    context={"rules": names},
                )
            )
        tracker.stage = JobStage.REPAIRED
        ctx.advance(JobStage.REPAIRED)

        tracker.stage = JobStage.VALIDATING
        ctx.advance(JobStage.VALIDATING)
        report = self._validate_once(plan, staging, ctx)
        if not report.ok:
            raise ValidationError(self._validation_failure(report, plan, ctx))
        return report, records

    def _validate_once(
        self,
        plan: PreparationPlan,
        staging: Path,
        ctx: JobContext,
    ) -> ValidationReport:
        """Inventory the IR, read every numeric model and CPU-check the tokenizers."""
        ctx.emit(ProgressKind.VALIDATE, "Inventorying OpenVINO IR", status="started")
        irs = self._inventory(staging)
        ctx.emit(
            ProgressKind.VALIDATE,
            f"Read {len(irs)} IR file(s)",
            current=len(irs),
            total=len(irs),
        )
        ctx.emit(ProgressKind.VALIDATE, "Reading tokenizer and detokenizer on CPU")
        tokenizer = self._runtime.read_tokenizer(staging)

        properties = tokenizer.properties
        problems: list[str] = []
        missing: list[str] = []
        for ir in irs:
            if not ir.readable:
                problems.append(f"{ir.path.name}: {ir.read_error or 'could not be read'}")
        present_roles = {ir.role for ir in irs if ir.readable}
        for role in self._required_roles(plan.pipeline):
            if role not in present_roles:
                missing.append(role.value)
        if not irs:
            problems.append("The export produced no OpenVINO IR files at all.")
        if tokenizer.status is ProbeStatus.FAILED:
            problems.append(tokenizer.message or "Tokenizer validation failed.")

        ok = not problems and not missing
        report = ValidationReport(
            ok=ok,
            openvino_version=properties.get("openvino_version")
            or installed_version(_OPENVINO_DISTRIBUTION),
            tokenizers_version=properties.get("openvino_tokenizers_version")
            or installed_version(_TOKENIZERS_DISTRIBUTION),
            irs=irs,
            tokenizer_read_ok=_tri_state(properties.get("tokenizer_read")),
            detokenizer_read_ok=_tri_state(properties.get("detokenizer_read")),
            tokenizer_compile_ok=_all_ok(
                properties.get("tokenizer_compile"), properties.get("detokenizer_compile")
            ),
            missing_files=missing,
            problems=problems,
            validated_at=_now(),
        )
        if not ok:
            raw = tokenizer.message or (problems[0] if problems else "")
            report.diagnostic = self._classify(
                raw,
                device="CPU",
                stage="validate",
                ctx=ctx,
                metadata=plan.resolved_source.metadata,
                fallback=tokenizer.diagnostic,
            )
        return report

    def _inventory(self, staging: Path) -> list[IRDescription]:
        """Read every top-level ``*.xml`` IR in the staging directory."""
        try:
            candidates = sorted(item for item in staging.glob("*.xml") if item.is_file())
        except OSError:  # pragma: no cover - unreadable staging directory
            return []
        return [self._runtime.read_ir(item) for item in candidates]

    @staticmethod
    def _required_roles(pipeline: PipelineKind) -> tuple[SubmodelRole, ...]:
        """Roles an artifact of this pipeline kind must carry to be usable.

        Deliberately minimal: ``docs/MODEL_PIPELINE.md`` section 8 warns against
        demanding every role universally, because which submodels an export emits
        depends on the architecture.
        """
        if pipeline is PipelineKind.VLM:
            return (SubmodelRole.LANGUAGE, SubmodelRole.TEXT_EMBEDDINGS)
        return (SubmodelRole.LANGUAGE,)

    def _detect_repairs(self, staging: Path) -> list[RepairRule]:
        """Ask the registry which repairs match what is actually in ``staging``."""
        try:
            names = [item.name for item in staging.iterdir() if item.is_file()]
        except OSError:  # pragma: no cover - unreadable staging directory
            return []

        def contains(file_name: str, needle: str) -> bool:
            target = staging / file_name
            try:
                return needle in target.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return False

        return self._registry.repairs_for(files=names, contents_probe=contains)

    @staticmethod
    def _repair_confirm(confirm: ConfirmCallback) -> Callable[[Repair, RepairMatch], bool]:
        """Adapt the plan-level confirmation callback to the repair framework's."""

        def decide(repair: Repair, match: RepairMatch) -> bool:
            return confirm(
                ConfirmationReason(
                    code=CONFIRMATION_REPAIR,
                    message=(
                        f"Apply repair {repair.id} to {len(match.counts)} file(s): "
                        f"{repair.description}"
                    ),
                )
            )

        return decide

    def _validation_failure(
        self,
        report: ValidationReport,
        plan: PreparationPlan,
        ctx: JobContext,
    ) -> Diagnostic:
        """Build the diagnostic raised when validation cannot be rescued."""
        if report.diagnostic is not None:
            return report.diagnostic
        detail = "; ".join([*report.problems, *(f"missing {m}" for m in report.missing_files)])
        return self._classify(
            detail,
            device="CPU",
            stage="validate",
            ctx=ctx,
            metadata=plan.resolved_source.metadata,
            fallback=diagnostic(
                "validation.failed",
                FailureCategory.UNKNOWN_RUNTIME_ERROR,
                "Artifact validation failed",
                detail or "The exported artifact did not pass validation.",
                remediation=[
                    "Inspect the job log for the raw conversion output.",
                    "Re-run with --force to redo the conversion from scratch.",
                ],
            ),
        )

    # --- device probing ---------------------------------------------------

    def _probe(
        self,
        plan: PreparationPlan,
        staging: Path,
        validation: ValidationReport,
        ctx: JobContext,
        *,
        confirm: ConfirmCallback,
        source_meta: ModelMetadata | None,
    ) -> tuple[dict[str, DeviceProbeReport], str, list[str]]:
        """Compile and smoke-test on the target device, falling back only if allowed.

        Returns the per-device reports, the device that succeeded, and the ordered
        list of devices that were attempted (so the caller can explain a fallback).
        """
        explicit = plan.compatibility.explicit_request
        attempted: list[str] = []
        reports: dict[str, DeviceProbeReport] = {}
        device = plan.target_device

        while True:
            attempted.append(device)
            report = self._probe_device(plan, staging, device, validation, ctx, source_meta)
            reports[device] = report
            if report.ok:
                return reports, device, attempted

            failure = report.diagnostic or self._unclassified(device)
            if explicit:
                raise DeviceUnsupportedError(self._explicit_failure(plan, failure, device))
            if failure.environment_broken:
                raise DeviceUnsupportedError(self._broken_environment(failure, device))

            nxt = self._next_device(plan, failure, attempted)
            if nxt is None:
                raise DeviceUnsupportedError(self._exhausted(plan, failure, attempted))
            explanation = (
                f"{device} cannot run this model: {failure.title}. "
                f"This is a compatibility limit, not a broken {device}. Trying {nxt} instead."
            )
            ctx.emit(
                ProgressKind.PROBE,
                explanation,
                device=nxt,
                status="fallback",
                severity=Severity.WARNING,
                diagnostic=failure,
            )
            ctx.logger.warning(
                "probe.fallback", explanation, previous=device, next=nxt, code=failure.code
            )
            if not confirm(
                ConfirmationReason(code=CONFIRMATION_DEVICE_FALLBACK, message=explanation)
            ):
                raise DeviceUnsupportedError(self._exhausted(plan, failure, attempted))
            device = nxt

    def _probe_device(
        self,
        plan: PreparationPlan,
        staging: Path,
        device: str,
        validation: ValidationReport,
        ctx: JobContext,
        source_meta: ModelMetadata | None,
    ) -> DeviceProbeReport:
        """Compile every numeric submodel on ``device``, then run the smoke test.

        Tokenizers are never compiled here: they are string-tensor models that
        only the CPU plugin supports, and they were already read and CPU-compiled
        during validation. Each numeric submodel is recorded separately so a VLM
        whose vision tower behaves differently from its language model is visible
        in the manifest rather than averaged away.

        A raw compile failure the registry marks ``defer_to_pipeline`` does not
        decide the device. It is what a raw ``Core.compile_model`` hits on every
        dynamically shaped LLM on the NPU, whereas the GenAI pipeline reshapes
        first -- so the smoke test, which builds that pipeline, is the verdict.
        Deferral never makes a device look supported by itself: without a
        successful smoke test the device is reported failed.
        """
        attempted_at = _now()
        properties = self._compile_properties(plan.profile)
        results: list[ProbeResult] = []
        failure: Diagnostic | None = None
        deferred: Diagnostic | None = None
        numeric = 0

        for ir in validation.irs:
            if ir.role.is_tokenizer:
                results.append(
                    ProbeResult(
                        device="CPU",
                        role=ir.role,
                        path=ir.path,
                        status=ProbeStatus.SKIPPED,
                        message="Tokenizers are CPU-hosted; validated on CPU, never on an "
                        "accelerator.",
                    )
                )
                continue
            numeric += 1
            ctx.emit(
                ProgressKind.PROBE,
                f"Compiling {ir.path.name} on {device}",
                device=device,
                status="started",
            )
            result = self._runtime.compile(ir.path, device, properties)
            results.append(result)
            if result.ok:
                continue
            ctx.logger.error(
                "probe.compile_failed",
                "submodel compile failed",
                device=device,
                role=ir.role.value,
                file=ir.path.name,
                raw=redact_text(result.message or ""),
            )
            classified = self._classify(
                result.message or "",
                device=device,
                stage="probe",
                ctx=ctx,
                metadata=source_meta,
                fallback=result.diagnostic,
            )
            if classified.defer_to_pipeline:
                results[-1] = result.model_copy(
                    update={
                        "status": ProbeStatus.DEFERRED,
                        "diagnostic": classified,
                        "message": (
                            f"A raw compile on {device} is not conclusive for this graph; "
                            "the GenAI pipeline smoke test decides. Raw error: "
                            f"{result.message or classified.message}"
                        ),
                    }
                )
                if deferred is None:
                    deferred = classified
                continue
            if failure is None:
                failure = classified

        if numeric == 0:
            failure = diagnostic(
                "validation.no_numeric_ir",
                FailureCategory.UNKNOWN_RUNTIME_ERROR,
                "No numeric model to probe",
                f"{staging} contains no numeric OpenVINO IR, only tokenizer models.",
                remediation=["Re-run the conversion; the export is incomplete."],
            )
        if failure is not None:
            return DeviceProbeReport(
                device=device,
                status=ProbeStatus.FAILED,
                submodels=results,
                diagnostic=failure,
                attempted_at=attempted_at,
            )

        smoke = self._smoke_test(plan, staging, device, ctx)
        if smoke.status is ProbeStatus.FAILED:
            # The registry is consulted even when the adapter already attached a
            # generic diagnostic: a pipeline load failure arrives as
            # runtime.genai.load_failed, and only the registry can recognise
            # the compiler signature inside its raw text. The adapter's
            # diagnostic remains the fallback for text no rule matches.
            classified = self._classify(
                failure_text(smoke),
                device=device,
                stage="smoke",
                ctx=ctx,
                metadata=source_meta,
                fallback=smoke.diagnostic,
            )
            return DeviceProbeReport(
                device=device,
                status=ProbeStatus.FAILED,
                submodels=results,
                smoke_test=smoke,
                diagnostic=classified,
                attempted_at=attempted_at,
            )
        if deferred is not None and smoke.status is not ProbeStatus.OK:
            # The raw compile was set aside for the pipeline to judge, and the
            # pipeline was never exercised: there is no evidence either way, so
            # the device is not claimed.
            unsettled = deferred.model_copy(
                update={
                    "message": (
                        f"{deferred.message} The GenAI pipeline smoke test that settles this "
                        f"could not run ({smoke.message or smoke.status.value}), so {device} "
                        "support is unproven."
                    )
                }
            )
            return DeviceProbeReport(
                device=device,
                status=ProbeStatus.FAILED,
                submodels=results,
                smoke_test=smoke,
                diagnostic=unsettled,
                attempted_at=attempted_at,
            )
        if deferred is not None:
            ctx.logger.info(
                "probe.deferred_resolved",
                "raw compile failure overruled by a successful pipeline smoke test",
                device=device,
                code=deferred.code,
            )
        return DeviceProbeReport(
            device=device,
            status=ProbeStatus.OK,
            submodels=results,
            smoke_test=smoke,
            attempted_at=attempted_at,
        )

    @staticmethod
    def _compile_properties(profile: ConversionProfile) -> dict[str, str]:
        """Profile runtime properties minus the ones only a GenAI pipeline understands."""
        return {
            name: value
            for name, value in profile.runtime_properties.items()
            if name.upper() not in GENAI_ONLY_PROPERTIES
        }

    def _smoke_test(
        self,
        plan: PreparationPlan,
        staging: Path,
        device: str,
        ctx: JobContext,
    ) -> ProbeResult:
        """Generate a few tokens to prove the pipeline actually runs.

        The profile's full runtime properties are applied here — including the
        GenAI-only ones such as ``ATTENTION_BACKEND=SDPA`` — because this is the
        first point at which a real pipeline exists to accept them.
        """
        if self._smoke is None:
            return ProbeResult(
                device=device,
                status=ProbeStatus.SKIPPED,
                message="No generation smoke tester was configured for this run.",
            )
        ctx.emit(
            ProgressKind.PROBE,
            f"Generating {SMOKE_MAX_NEW_TOKENS} tokens on {device}",
            device=device,
            status="smoke",
        )
        try:
            return self._smoke.smoke_test(
                staging,
                device=device,
                pipeline=plan.pipeline,
                properties=dict(plan.profile.runtime_properties),
                prompt=SMOKE_PROMPT,
                max_new_tokens=SMOKE_MAX_NEW_TOKENS,
            )
        except YongaError as error:
            return ProbeResult(
                device=device,
                status=ProbeStatus.FAILED,
                diagnostic=error.diagnostic,
                message=error.diagnostic.message,
            )
        except Exception as exc:
            return ProbeResult(
                device=device,
                status=ProbeStatus.FAILED,
                message=redact_text(f"{type(exc).__name__}: {exc}"),
            )

    def _next_device(
        self,
        plan: PreparationPlan,
        failure: Diagnostic,
        attempted: Sequence[str],
    ) -> str | None:
        """Pick the next device auto mode may try, or ``None`` when there is none.

        The classifier's own ``fallback_devices`` lead, because a rule that knows
        this exact failure knows best where the model *does* run; the plan's
        ranked candidates are the generic backstop.
        """
        available = plan.hardware.available_device_names
        unsupported = {
            entry.device
            for entry in plan.compatibility.per_device
            if entry.status is DeviceStatus.UNSUPPORTED
        }
        seen = set(attempted)
        ordered: list[str] = [normalise_device(name) for name in failure.fallback_devices]
        ordered.extend(plan.compatibility.fallback_candidates)
        for name in ordered:
            if name in seen or name in unsupported or name not in available:
                continue
            return name
        return None

    def _classify(
        self,
        raw: str,
        *,
        device: str | None,
        stage: str,
        ctx: JobContext,
        metadata: ModelMetadata | None,
        fallback: Diagnostic | None,
    ) -> Diagnostic:
        """Run raw failure text through the registry classifier, keeping the original.

        The raw text is written to the job log before classification so the
        unedited message survives even when no rule matches it.
        """
        ctx.logger.error(
            "failure.raw",
            f"raw failure text captured at stage {stage}",
            stage=stage,
            device=device or "unknown",
            raw=redact_text(raw),
        )
        classified = self._registry.classify(
            raw_error=raw,
            device=device,
            meta=metadata,
            stage=stage,
            job_id=ctx.job_id,
            log_path=ctx.logger.path,
        )
        if classified.category is FailureCategory.UNKNOWN_RUNTIME_ERROR and fallback is not None:
            return fallback
        return classified

    @staticmethod
    def _unclassified(device: str) -> Diagnostic:
        """Diagnostic used when a device report failed without one of its own."""
        return diagnostic(
            "probe.failed",
            FailureCategory.UNKNOWN_RUNTIME_ERROR,
            f"{device} probe failed",
            f"The artifact could not be compiled on {device}.",
            remediation=["Inspect the job log for the raw runtime output."],
        )

    @staticmethod
    def _explicit_failure(plan: PreparationPlan, failure: Diagnostic, device: str) -> Diagnostic:
        """Add a recommended fallback *command* without changing the device.

        An explicit ``--device`` request is an instruction, not a preference: the
        job stops, explains, and tells the user exactly what to type next.
        """
        available = plan.hardware.available_device_names
        alternatives = [
            name
            for name in [normalise_device(n) for n in failure.fallback_devices]
            if name in available and name != device
        ]
        remediation = list(failure.remediation)
        remediation.extend(
            f"Prepare for {name} instead: yonga prepare {plan.requested.repo_id} "
            f"--device {name.lower()}"
            for name in alternatives
        )
        if not failure.environment_broken:
            remediation.append(
                f"Or let yonga choose: yonga prepare {plan.requested.repo_id} --device auto"
            )
        return failure.model_copy(
            update={
                "message": (
                    f"{failure.message} {device} was requested explicitly, so yonga will not "
                    "silently prepare for another device."
                ),
                "remediation": remediation,
                "fallback_devices": alternatives,
            }
        )

    @staticmethod
    def _broken_environment(failure: Diagnostic, device: str) -> Diagnostic:
        """Refuse to hide a broken environment behind an automatic fallback."""
        remediation = list(failure.remediation)
        remediation.append(
            f"yonga did not fall back to another device: {device} looks misconfigured rather "
            "than incompatible, and hiding that would stop you from fixing it."
        )
        return failure.model_copy(update={"remediation": remediation})

    @staticmethod
    def _exhausted(
        plan: PreparationPlan,
        failure: Diagnostic,
        attempted: Sequence[str],
    ) -> Diagnostic:
        """Diagnostic raised when auto mode has no device left to try."""
        remediation = list(failure.remediation)
        remediation.append(
            f"Devices already tried: {', '.join(attempted)}. "
            f"Run 'yonga inspect {plan.requested.repo_id}' to see what the registry knows."
        )
        return failure.model_copy(
            update={
                "message": (f"{failure.message} No further device is available to fall back to."),
                "remediation": remediation,
            }
        )

    # --- manifest ---------------------------------------------------------

    def _build_manifest(
        self,
        *,
        plan: PreparationPlan,
        export: ExportRecord,
        validation: ValidationReport,
        repairs: Sequence[RepairRecord],
        probes: Mapping[str, DeviceProbeReport],
        device: str,
        source_meta: ModelMetadata | None,
        notes: Sequence[str],
    ) -> ArtifactManifest:
        """Assemble the schema-1 manifest described by ``SPEC.md`` section 14.

        File digests are deliberately *not* filled in here:
        :meth:`ArtifactStore.finalize` computes them after every repair has run,
        so the manifest always describes the bytes that were actually published.

        Every path in ``validation``, ``probes`` and ``export`` has already been
        made artifact-relative by :func:`_artifact_relative`; see its docstring
        for why the export ``argv`` is the one thing left literal.
        """
        source = plan.resolved_source.source
        source_sha = (
            source_meta.resolved_revision if source_meta is not None else None
        ) or source.revision
        requested_sha = plan.requested.revision
        arch_rule, arch_explanation = self._registry.resolve_architecture(
            source_meta, tuple(plan.hardware.device_tags)
        )
        # The manifest is the durable record of how this artifact was produced,
        # so it keeps the identification chain too, not only its conclusion.
        manifest_notes = list(notes)
        if arch_rule is not None:
            manifest_notes.extend(arch_explanation.notes)
        selection = plan.profile_selection
        if selection is not None and selection.applies and selection.sentence:
            manifest_notes.append(f"Profile {selection.profile_id}: {selection.sentence}")
        chat_rule = self._registry.chat_profile(
            arch_rule=arch_rule, repo_id=source.repo_id, meta=source_meta
        )
        return ArtifactManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            artifact_id=self._artifact_id(
                source.repo_id, source_sha, _artifact_key(plan.resolved_source, plan.profile)
            ),
            created_at=_now(),
            tool_version=__version__,
            requested_model=ManifestModelRef(
                repo_id=plan.requested.repo_id,
                requested_revision=plan.requested_revision,
                resolved_revision=requested_sha,
            ),
            resolved_source=ManifestModelRef(
                repo_id=source.repo_id,
                requested_revision=source.revision,
                resolved_revision=source_sha,
                relationship=plan.resolved_source.relationship,
            ),
            architecture=self._architecture_identity(source_meta, arch_rule),
            architecture_rule_id=arch_rule.id if arch_rule is not None else None,
            model_type=source_meta.model_type if source_meta is not None else None,
            task=plan.task,
            pipeline=plan.pipeline,
            export=export,
            repairs=list(repairs),
            validation=validation,
            device_probes=dict(probes),
            recommended_runtime=RecommendedRuntime(
                device=device,
                pipeline=plan.pipeline,
                properties=dict(plan.profile.runtime_properties),
                max_new_tokens=int(self._settings.max_new_tokens),
                generation_defaults={},
                chat=self._chat_snapshot(chat_rule),
            ),
            source_license=source_meta.license if source_meta is not None else None,
            files=[],
            host=self._host_summary(plan.hardware, validation),
            notes=list(dict.fromkeys(manifest_notes)),
        )

    @staticmethod
    def _architecture_identity(
        meta: ModelMetadata | None,
        arch_rule: ArchitectureRule | None,
    ) -> str | None:
        """Return the architecture identity ``SPEC.md`` section 14 asks for.

        That is what this model *is* — ``qwen3_5`` — and never the id of the
        registry rule that happened to match it. A rule covers a family, so its
        id answers a different question: it is recorded separately, in
        ``architecture_rule_id``, so the lineage survives.

        The declaration wins when there is one. A repository that declares
        nothing — a GGUF-only repository has no ``config.json`` to declare in —
        falls back to the canonical model type of the rule that identified it,
        which is registry data rather than a guess made here.
        """
        if meta is not None:
            if meta.model_type:
                return meta.model_type
            if meta.architectures:
                return meta.architectures[0]
        if arch_rule is not None and arch_rule.match.model_types:
            return arch_rule.match.model_types[0]
        return None

    @staticmethod
    def _artifact_id(repo_id: str, source_sha: str | None, profile_id: str) -> str:
        """Return the stable identity of one (source revision, profile) artifact.

        Deterministic on purpose: the same source revision and profile always
        produce the same id, which is what lets reuse detection and the compile
        cache agree across runs.
        """
        payload = "\n".join([repo_id, source_sha or "", profile_id, str(MANIFEST_SCHEMA_VERSION)])
        return short_hash(sha256_text(payload), 16)

    @staticmethod
    def _chat_snapshot(rule: ChatProfileRule | None) -> dict[str, object]:
        """Freeze the registry chat profile into the manifest.

        Stored with the artifact rather than looked up later so that a registry
        change cannot retroactively alter how a prepared model talks.
        """
        if rule is None:
            return {}
        return {
            "id": rule.id,
            "supports_thinking_flag": rule.supports_thinking_flag,
            "thinking_template_variable": rule.thinking_template_variable,
            "default_thinking": rule.default_thinking,
            "reasoning_wrappers": [list(pair) for pair in rule.reasoning_wrappers],
            "stop_strings": list(rule.stop_strings),
        }

    @staticmethod
    def _host_summary(
        inventory: HardwareInventory,
        validation: ValidationReport,
    ) -> dict[str, str]:
        """Summarise the build host without recording user names or paths."""
        host = inventory.host
        summary = {
            "platform": host.platform,
            "machine": host.machine,
            "kernel": host.kernel,
            "python": host.python_version,
            "cpu": host.cpu_model or "unknown",
            "distro": host.distro_pretty or host.distro_id or "unknown",
            "devices": ",".join(inventory.available_device_names),
            "openvino": validation.openvino_version or "unknown",
            "openvino_tokenizers": validation.tokenizers_version or "unknown",
        }
        return {key: value for key, value in summary.items() if value}
