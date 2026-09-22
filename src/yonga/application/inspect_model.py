"""``yonga inspect``: low-cost model discovery before anything large is downloaded.

``SPEC.md`` section 6.2 makes one hard promise about this command: it must answer
"can this model run here, and how would it be prepared?" from metadata alone.
:class:`InspectService` therefore reads repository metadata and the handful of
small configuration files the Hub adapter already fetches, asks the registry what
is known about the architecture, and asks the host what devices exist. It never
calls :meth:`yonga.adapters.HubClient.download`, so a 19 GB repository costs a few
kilobytes to inspect.

The second promise is about honesty. Device compatibility in the report is
*evidence*, not inference: an entry says which registry rule made the claim and
how strong that evidence is. A device that OpenVINO exposes but the registry has
no claim about is reported as ``unknown`` — "the NPU is present" has never been a
reason to believe a model runs on it.

The service is pure application policy. It receives its collaborators
(:class:`~yonga.adapters.HubClient`,
:class:`~yonga.registry.resolver.RegistryResolver`,
:class:`~yonga.adapters.HardwareProbe`) already constructed by the CLI and emits
:class:`~yonga.application.progress.ProgressEvent` values instead of printing.

One deliberate coupling: the exporter-requirement check reuses
:func:`yonga.adapters.exporter_optimum.check_exporter_requirements`. That function
is pure policy — it compares declared specifiers against installed versions and
imports nothing heavier than :mod:`packaging` — and reusing it keeps ``inspect``
and ``prepare`` from ever disagreeing about whether this host can export a model.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from yonga.adapters import HardwareProbe, HubClient
from yonga.adapters.exporter_optimum import check_exporter_requirements
from yonga.application.progress import ProgressEvent, ProgressKind, ProgressSink, null_sink
from yonga.application.resolve import (
    DEFAULT_SOURCE_POLICY,
    SourceOption,
    SourcePolicy,
    describe_source_options,
    is_pinned_revision,
)
from yonga.config.settings import DevicePolicy, Settings
from yonga.domain.compatibility import DeviceCompatibility, DeviceStatus, ResolutionExplanation
from yonga.domain.hardware import HardwareInventory, normalise_device
from yonga.domain.models import ModelIdentity, ModelMetadata, PipelineKind
from yonga.domain.plans import ExporterRequirements
from yonga.registry.resolver import VERSION_DISTRIBUTIONS, RegistryResolver
from yonga.registry.schema import (
    IDENTITY_HINT_SOURCE,
    SPECIFICITY_ARCHITECTURE_ALIAS,
    ArchitectureRule,
    ExportSpec,
    ModelRule,
)

__all__ = ["InspectReport", "InspectService"]


def _distribution_versions(versions: Mapping[str, str | None]) -> dict[str, str | None]:
    """Re-key installed versions by distribution name as well as by constraint key.

    The registry constrains ``optimum_intel``; the requirement check looks the
    same package up as ``optimum-intel``. Publishing both spellings means neither
    side has to know about the other's naming.
    """
    result: dict[str, str | None] = dict(versions)
    for key, distribution in VERSION_DISTRIBUTIONS.items():
        if key in versions:
            result[distribution] = versions[key]
    return result


def _export_spec(
    architecture_rule: ArchitectureRule | None,
    model_rule: ModelRule | None,
) -> ExportSpec | None:
    """Return the export spec that applies, preferring an exact-model override."""
    if model_rule is not None and model_rule.export is not None:
        return model_rule.export
    if architecture_rule is not None:
        return architecture_rule.export
    return None


def _identified_from(
    architecture_rule: ArchitectureRule | None,
    explanation: ResolutionExplanation,
) -> str | None:
    """Name the metadata source that identified the architecture, when it was not ``config.json``.

    The registry demotes a rule it could reach only through an identity alias to
    :data:`~yonga.registry.schema.SPECIFICITY_ARCHITECTURE_ALIAS`, so the
    winning candidate's own score — a number the resolver already published,
    not a parse of its prose — says whether the identification rests on a hint
    read out of a weight file. ``None`` means the repository declared the
    identity itself, or that nothing matched at all.
    """
    if architecture_rule is None:
        return None
    for candidate in explanation.matched:
        if (
            candidate.rule_id == architecture_rule.id
            and candidate.specificity == SPECIFICITY_ARCHITECTURE_ALIAS
        ):
            return IDENTITY_HINT_SOURCE
    return None


def _requirements_from(spec: ExportSpec | None) -> ExporterRequirements:
    """Turn a registry export spec into declared exporter requirements."""
    if spec is None:
        return ExporterRequirements()
    return ExporterRequirements(
        transformers=spec.transformers,
        optimum_intel=spec.optimum_intel,
        openvino=spec.openvino,
        nncf=spec.nncf,
    )


class InspectReport(BaseModel):
    """Everything ``yonga inspect`` learned about one model, without downloading it.

    The report is the whole output of the command: the CLI renders it with
    :func:`yonga.ui.tables.inspect_report` and ``--json`` prints
    :meth:`to_json`. Nothing here required a weight shard.
    """

    model_config = ConfigDict(extra="forbid")

    requested: ModelIdentity
    """The repository and revision the user asked for, exactly as given."""

    requested_revision: str | None = None
    """The revision the user asked for, if any; may be a branch or tag."""

    resolved_revision: str | None = None
    """The immutable commit SHA the request resolved to, when the Hub reported one."""

    metadata: ModelMetadata
    """Repository metadata, with :attr:`resolved_revision` pinned when known."""

    architecture: str | None = None
    """The architecture class name, or the model type when no class is declared."""

    architecture_identified_from: str | None = None
    """The metadata source that identified the architecture, when ``config.json`` did not.

    ``None`` is the ordinary case: the repository declared its own identity (or
    nothing matched). A value — ``"GGUF metadata"`` — means the registry reached
    its rule through an identity *alias* matched against an identifier read out
    of the weight file, which is weaker than a declared ``model_type``. Every
    device claim in :attr:`compatibility` descends from that match, so this is
    stated as a field rather than left for a ``--json`` consumer to find in
    :attr:`notes`.
    """

    architecture_rule_id: str | None = None
    model_rule_id: str | None = None
    task: str | None = None
    pipeline: PipelineKind = PipelineKind.UNKNOWN

    compatibility: list[DeviceCompatibility] = Field(default_factory=list)
    """Per-device compatibility for *this* host, each entry citing its evidence."""

    host_devices: list[str] = Field(default_factory=list)
    """Devices OpenVINO actually exposes here, normalised and ordered."""

    recommended_device: str | None = None
    """The device the registry recommends for this architecture, if it names one."""

    configured_device: str | None = None
    """The device pinned in configuration, or ``None`` when the policy is ``auto``."""

    exporter: ExporterRequirements = Field(default_factory=ExporterRequirements)
    """Exporter package requirements from the architecture rule, checked against this host."""

    source_options: list[SourceOption] = Field(default_factory=list)
    """Candidate preparation routes (``docs/CLI_SPEC.md`` section 4)."""

    known_limitations: list[str] = Field(default_factory=list)
    trust_remote_code_required: bool = False
    offline: bool = False
    notes: list[str] = Field(default_factory=list)

    @property
    def usable_devices(self) -> list[str]:
        """Devices not recorded as unsupported, in report order.

        "Usable" includes ``unknown``: an absent claim is a reason to probe, not
        a reason to refuse.
        """
        return [
            entry.device
            for entry in self.compatibility
            if entry.status is not DeviceStatus.UNSUPPORTED
        ]

    def compatibility_for(self, device: str) -> DeviceCompatibility | None:
        """Return the compatibility entry for *device*, or ``None``."""
        name = normalise_device(device)
        for entry in self.compatibility:
            if entry.device == name:
                return entry
        return None

    def option(self, label: str) -> SourceOption | None:
        """Return the first source option carrying *label*, or ``None``."""
        for entry in self.source_options:
            if entry.label == label:
                return entry
        return None

    def to_json(self) -> dict[str, object]:
        """Return a JSON-safe dictionary (enums as values, paths as strings)."""
        return dict(self.model_dump(mode="json"))


class InspectService:
    """Builds an :class:`InspectReport` from metadata, registry rules and host facts."""

    def __init__(
        self,
        hub_client: HubClient,
        registry_resolver: RegistryResolver,
        hardware_probe: HardwareProbe,
        settings: Settings,
        *,
        progress: ProgressSink | None = None,
        source_policy: SourcePolicy | None = None,
    ) -> None:
        """Bind the collaborators the CLI constructed for this run.

        *source_policy* only affects how preparation routes are described; the
        service never resolves a source, because doing so would mean inspecting
        other repositories on the Hub.
        """
        self._hub = hub_client
        self._registry = registry_resolver
        self._hardware = hardware_probe
        self._settings = settings
        self._progress: ProgressSink = null_sink() if progress is None else progress
        self._source_policy = DEFAULT_SOURCE_POLICY if source_policy is None else source_policy

    def inspect(
        self,
        identity: ModelIdentity | str,
        *,
        revision: str | None = None,
    ) -> InspectReport:
        """Inspect *identity*, optionally pinned to *revision*.

        *revision* overrides any revision carried by *identity*. Raises the typed
        errors the Hub adapter raises — a missing repository, a gated repository
        or an offline cache miss are reported, never guessed around.
        """
        requested = ModelIdentity.parse(identity) if isinstance(identity, str) else identity
        if revision is not None:
            requested = requested.with_revision(revision)

        self._emit(
            ProgressKind.STAGE,
            f"Reading repository metadata for {requested.repo_id}",
            repo_id=requested.repo_id,
        )
        metadata = self._hub.inspect(requested, with_sizes=True)
        resolved_revision = self._resolved_revision(requested, metadata)
        if resolved_revision is not None and metadata.resolved_revision != resolved_revision:
            metadata = metadata.model_copy(update={"resolved_revision": resolved_revision})

        self._emit(
            ProgressKind.PROBE,
            "Collecting host devices and packages",
        )
        inventory = self._hardware.inventory(full=False)

        self._emit(
            ProgressKind.STAGE,
            "Matching compatibility rules for this host",
            repo_id=requested.repo_id,
        )
        architecture_rule, architecture_explanation = self._registry.resolve_architecture(
            metadata, inventory.device_tags
        )
        model_rule, _model_explanation = self._registry.resolve_model_rule(metadata)
        compatibility = self._registry.device_compatibility(
            arch_rule=architecture_rule,
            inventory=inventory,
            model_rule=model_rule,
        )

        spec = _export_spec(architecture_rule, model_rule)
        exporter = check_exporter_requirements(
            _requirements_from(spec),
            versions=_distribution_versions(self._registry.versions),
        )

        candidates, _derivative_explanation = self._registry.resolve_derivatives(requested)
        source_options = describe_source_options(metadata, candidates, policy=self._source_policy)

        report = InspectReport(
            requested=requested,
            requested_revision=requested.revision,
            resolved_revision=resolved_revision,
            metadata=metadata,
            architecture=self._architecture_label(metadata, architecture_rule),
            architecture_identified_from=_identified_from(
                architecture_rule, architecture_explanation
            ),
            architecture_rule_id=architecture_rule.id if architecture_rule is not None else None,
            model_rule_id=model_rule.id if model_rule is not None else None,
            task=metadata.task or metadata.pipeline_tag,
            pipeline=self._pipeline(metadata, spec),
            compatibility=compatibility,
            host_devices=list(inventory.available_device_names),
            recommended_device=(
                architecture_rule.recommended_device if architecture_rule is not None else None
            ),
            configured_device=self._configured_device(),
            exporter=exporter,
            source_options=source_options,
            known_limitations=self._known_limitations(architecture_rule, model_rule),
            trust_remote_code_required=(
                metadata.requires_remote_code or (spec is not None and spec.trust_remote_code)
            ),
            offline=self._settings.offline,
            notes=[],
        )
        report.notes.extend(self._notes(report, metadata, inventory, architecture_rule))
        if architecture_rule is not None:
            # On a successful match the only note resolution adds is how the
            # architecture was identified, which the user is owed when it did
            # not come from config.json.
            report.notes.extend(architecture_explanation.notes)
        self._emit(
            ProgressKind.STAGE,
            f"Inspection of {requested.repo_id} complete",
            repo_id=requested.repo_id,
        )
        return report

    # -- helpers --------------------------------------------------------- #

    def _resolved_revision(
        self,
        requested: ModelIdentity,
        metadata: ModelMetadata,
    ) -> str | None:
        """Return the commit SHA for *requested*, asking the Hub only if needed.

        ``inspect`` reports what it can rather than refusing: an unpinnable
        revision becomes a note here, and a hard failure only in
        :class:`~yonga.application.resolve.SourceResolver`, where reproducibility
        actually matters.
        """
        candidate = metadata.resolved_revision
        if is_pinned_revision(candidate):
            return candidate
        if is_pinned_revision(requested.revision):
            return requested.revision
        resolved = self._hub.resolve_revision(requested)
        return resolved if is_pinned_revision(resolved) else candidate

    @staticmethod
    def _architecture_label(
        metadata: ModelMetadata,
        architecture_rule: ArchitectureRule | None,
    ) -> str | None:
        """Return the architecture class name, falling back to the model type.

        When the repository declares neither — a GGUF-only repository has no
        ``config.json`` to declare them in — the matched registry rule supplies
        the canonical model type it is keyed on, so the report names the
        architecture the rule identified instead of leaving the field blank.
        The name still comes from registry data, never from this module.
        """
        if metadata.architectures:
            return metadata.architectures[0]
        if metadata.model_type:
            return metadata.model_type
        if architecture_rule is not None and architecture_rule.match.model_types:
            return architecture_rule.match.model_types[0]
        return None

    @staticmethod
    def _pipeline(metadata: ModelMetadata, spec: ExportSpec | None) -> PipelineKind:
        """Prefer the registry's declared pipeline over the metadata heuristic."""
        if spec is not None and spec.pipeline is not PipelineKind.UNKNOWN:
            return spec.pipeline
        return metadata.pipeline_kind

    @staticmethod
    def _known_limitations(
        architecture_rule: ArchitectureRule | None,
        model_rule: ModelRule | None,
    ) -> list[str]:
        """Collect the limitations declared by the matching rules, de-duplicated."""
        limitations: list[str] = []
        if architecture_rule is not None:
            limitations.extend(architecture_rule.known_limitations)
        if model_rule is not None:
            limitations.extend(model_rule.notes)
        seen: list[str] = []
        for entry in limitations:
            text = entry.strip()
            if text and text not in seen:
                seen.append(text)
        return seen

    def _configured_device(self) -> str | None:
        """Return the configured default device, or ``None`` when it is ``auto``."""
        policy = self._settings.default_device
        if policy is DevicePolicy.AUTO:
            return None
        return normalise_device(policy.value)

    @staticmethod
    def _notes(
        report: InspectReport,
        metadata: ModelMetadata,
        inventory: HardwareInventory,
        architecture_rule: ArchitectureRule | None,
    ) -> list[str]:
        """Compose the report's advisory notes. None of them assert compatibility."""
        notes: list[str] = []
        if not metadata.exists:
            notes.append(
                "The Hugging Face Hub did not confirm that this repository exists; every "
                "conclusion below is provisional."
            )
        if report.offline:
            notes.append(
                "Offline mode is enabled, so this report was built from the local Hugging Face "
                "cache and may be older than the repository."
            )
        if metadata.gated or metadata.private:
            notes.append(
                "This repository is gated or private; preparation needs a Hugging Face token "
                "with access to it."
            )
        if report.resolved_revision is None:
            notes.append(
                "No commit SHA could be resolved for this request. Preparation will refuse to "
                "run from a branch, so pin a revision before preparing."
            )
        if metadata.requires_remote_code:
            notes.append(
                "This repository ships custom modelling code. Preparation needs an explicit "
                "--trust-remote-code; --yes will not enable it."
            )
        if architecture_rule is None:
            notes.append(
                "No registry rule matches this architecture, so device support below is "
                "reported as unknown rather than guessed. Runtime probing decides."
            )
        if not inventory.openvino_available:
            error = inventory.openvino_import_error
            detail = f" ({error})" if error else ""
            notes.append(
                f"OpenVINO is not available to yonga in this process{detail}, so the device "
                f"list below comes from the registry alone and nothing was observed on the host."
            )
        if report.configured_device is not None:
            entry = report.compatibility_for(report.configured_device)
            if entry is not None and entry.status is DeviceStatus.UNSUPPORTED:
                notes.append(
                    f"Your configured default device {report.configured_device} is recorded as "
                    f"unsupported for this architecture; preparation will need another device."
                )
        if not report.exporter.satisfied:
            notes.extend(report.exporter.problems)
        return notes

    def _emit(
        self,
        kind: ProgressKind,
        message: str,
        **data: object,
    ) -> None:
        """Emit one progress event; rendering must never be able to fail inspection."""
        self._progress.emit(ProgressEvent(kind=kind, message=message, data=dict(data)))
