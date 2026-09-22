"""Conversion-source resolution: deciding which repository yonga actually converts.

This is the most safety-critical policy in the project. A model manager that
quietly swaps one set of weights for another is worse than one that refuses to
run, so every rule here is written to fail loudly rather than to guess.

``SPEC.md`` section 8.1 fixes the preference order and this module implements it
verbatim:

1. an existing, *complete* OpenVINO IR artifact in the requested repository is
   used directly — relationship :data:`SourceRelationship.ORIGINAL`,
   ``already_openvino=True``, no re-export;
2. safetensors in the requested repository are the preferred conversion source;
3. other standard Transformers weights (``pytorch_bin``) in that repository come
   next;
4. a derivative **declared in the registry** may be offered, but under the MVP
   default trust policy (``suggest``) it always carries
   ``requires_confirmation=True`` and a
   :class:`~yonga.domain.plans.ConfirmationReason` with code
   ``third_party_derivative``; ``never_auto`` candidates are informational and
   are never selected automatically at all;
5. direct GGUF conversion only when the backend explicitly declares support for
   the architecture. :data:`GGUF_BACKEND_ARCHITECTURES` is empty on every build
   shipped so far, so this path currently fails with a clear explanation instead
   of pretending that "a GGUF file exists" means "OpenVINO GenAI can read it";
6. otherwise :class:`~yonga.domain.diagnostics.SourceResolutionError`, listing
   the choices that would actually help.

Two invariants hold everywhere in this file:

* **Only registry-declared lineage counts.** There is no name-similarity,
  fuzzy-matching or Hub-search logic here, and there never may be. Two
  repositories whose names look alike are, as far as yonga is concerned,
  unrelated.
* **A resolved source is always pinned to an immutable commit SHA.** A branch or
  tag is never enough for a reproducible preparation job
  (``docs/MODEL_PIPELINE.md`` section 4), so resolution fails rather than hand a
  mutable reference to the downloader.

The module is pure application policy: it takes an injected
:class:`~yonga.adapters.HubClient` and a
:class:`~yonga.registry.resolver.RegistryResolver`, emits
:class:`~yonga.application.progress.ProgressEvent` values, and imports no
renderer, no Hub library and no inference stack.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from yonga.adapters import HubClient
from yonga.application.progress import ProgressEvent, ProgressKind, ProgressSink, null_sink
from yonga.domain.diagnostics import (
    CancelledError,
    Diagnostic,
    FailureCategory,
    Severity,
    SourceResolutionError,
    YongaError,
    diagnostic,
)
from yonga.domain.models import (
    FormatObservation,
    ModelFormat,
    ModelIdentity,
    ModelMetadata,
    ResolvedSource,
    SourceEvidence,
    SourceRelationship,
)
from yonga.domain.plans import CONFIRMATION_THIRD_PARTY_DERIVATIVE, ConfirmationReason
from yonga.registry.resolver import RegistryResolver
from yonga.registry.schema import DerivativeCandidate, ModelRule, TrustPolicy

__all__ = [
    "CODE_GGUF_UNSUPPORTED",
    "CODE_NO_USABLE_SOURCE",
    "CODE_REVISION_UNPINNED",
    "CONFIRMATION_EXPLICIT_SOURCE",
    "CONVERSION_FORMAT_PREFERENCE",
    "DEFAULT_SOURCE_POLICY",
    "GGUF_BACKEND_ARCHITECTURES",
    "OPTION_DERIVATIVE",
    "OPTION_GGUF",
    "OPTION_OPENVINO",
    "OPTION_PYTORCH",
    "OPTION_SAFETENSORS",
    "SourceOption",
    "SourcePolicy",
    "SourceResolution",
    "SourceResolver",
    "describe_source_options",
    "is_pinned_revision",
]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

GGUF_BACKEND_ARCHITECTURES: Final[tuple[str, ...]] = ()
"""Architectures the bundled OpenVINO GenAI build can consume *directly* from GGUF.

Empty on purpose. OpenVINO GenAI 2026.4 — the stack this release is tested
against — has no GGUF reader for any architecture yonga supports, so no model
qualifies for step 5 of the preference order. The tuple exists so that support,
when it lands, is a data change backed by evidence rather than an optimistic
branch. The presence of ``.gguf`` files in a repository is never evidence.
"""

CONVERSION_FORMAT_PREFERENCE: Final[tuple[ModelFormat, ...]] = (
    ModelFormat.OPENVINO_IR,
    ModelFormat.SAFETENSORS,
    ModelFormat.PYTORCH_BIN,
)
"""Formats a conversion can read, strongest first (``SPEC.md`` section 8.1)."""

CONFIRMATION_EXPLICIT_SOURCE: Final[str] = "explicit_source"
"""Confirmation code recorded when the user named the source with ``--source``.

It is deliberately *not* ``third_party_derivative``: per ``SPEC.md`` section 6.3
the user having chosen the repository already satisfies the derivative
confirmation, so yonga records the substitution instead of asking about it twice.
"""

CODE_NO_USABLE_SOURCE: Final[str] = "source.no_usable_source"
"""Diagnostic code for "nothing in or declared for this repo can be converted"."""

CODE_GGUF_UNSUPPORTED: Final[str] = "source.gguf_unsupported"
"""Diagnostic code for a GGUF-only repository the backend cannot read."""

CODE_REVISION_UNPINNED: Final[str] = "source.revision_unpinned"
"""Diagnostic code for a source that could not be pinned to a commit SHA."""

OPTION_OPENVINO: Final[str] = "Direct OpenVINO source"
OPTION_SAFETENSORS: Final[str] = "Safetensors source"
OPTION_PYTORCH: Final[str] = "Transformers weights"
OPTION_DERIVATIVE: Final[str] = "Known derivative"
OPTION_GGUF: Final[str] = "Direct GGUF conversion"

_COMMIT_SHA: Final[re.Pattern[str]] = re.compile(r"\A[0-9a-f]{40}\Z")

#: Order in which declared derivatives are considered. ``never_auto`` is absent:
#: such a candidate is reported to the user but never selected by this module.
_TRUST_ORDER: Final[dict[TrustPolicy, int]] = {
    TrustPolicy.AUTO_IF_TRUSTED: 0,
    TrustPolicy.SUGGEST: 1,
}

_FORMAT_REASONS: Final[dict[ModelFormat, str]] = {
    ModelFormat.OPENVINO_IR: (
        "The requested repository already publishes a complete OpenVINO IR artifact, "
        "so it is used directly and nothing is re-exported."
    ),
    ModelFormat.SAFETENSORS: (
        "The requested repository publishes safetensors weights, which are the "
        "preferred conversion source."
    ),
    ModelFormat.PYTORCH_BIN: (
        "The requested repository publishes standard Transformers weights "
        "(PyTorch .bin); no safetensors were available."
    ),
    ModelFormat.GGUF: (
        "The backend declares direct GGUF support for this architecture, so the "
        "requested repository is read as-is."
    ),
}


def is_pinned_revision(value: str | None) -> bool:
    """Return ``True`` when *value* is a full 40-character git commit SHA.

    "A branch name is not a pin" is policy, not Hub mechanism, so the check lives
    in the application layer rather than being imported from
    :mod:`yonga.adapters.hf` — which would drag ``huggingface_hub`` into every
    service that needs to know whether a revision is reproducible.
    """
    return value is not None and _COMMIT_SHA.match(value) is not None


# --------------------------------------------------------------------------- #
# Policy and view models
# --------------------------------------------------------------------------- #


class SourcePolicy(BaseModel):
    """How far source resolution may go without asking the user.

    The defaults are the MVP policy described in ``SPEC.md`` section 8: use what
    the requested repository already contains, offer a registry-declared
    derivative but never select one silently, and never invent GGUF support.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    allow_openvino_direct: bool = True
    """Whether an existing OpenVINO IR artifact may be used without re-exporting."""

    require_complete_openvino_ir: bool = True
    """Whether an IR artifact missing a ``.bin`` companion is rejected as a direct source."""

    allow_derivatives: bool = True
    """Whether registry-declared derivative sources may be offered at all."""

    honour_registry_auto_trust: bool = True
    """Whether ``trust: auto_if_trusted`` may select a derivative without confirmation.

    Setting this to ``False`` downgrades every declared candidate to "needs
    confirmation"; it never promotes one.
    """

    gguf_architectures: tuple[str, ...] = GGUF_BACKEND_ARCHITECTURES
    """Architectures the backend can read directly from GGUF, case-insensitively."""


DEFAULT_SOURCE_POLICY: Final[SourcePolicy] = SourcePolicy()
"""The policy every command uses unless the caller overrides it."""


class SourceOption(BaseModel):
    """One preparation route, as ``yonga inspect`` presents it.

    Field-for-field compatible with :class:`yonga.ui.tables.SourceOption` so the
    CLI can render it with
    ``tables.SourceOption.model_validate(option.model_dump())``. It is declared
    here because application services must not import the UI package.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    available: bool = False
    repo_id: str | None = None
    detail: str | None = None
    requires_confirmation: bool = False


class SourceResolution(BaseModel):
    """A resolved source plus everything a preparation plan must show and ask.

    :meth:`SourceResolver.resolve` returns only the
    :class:`~yonga.domain.models.ResolvedSource`, because that is what the rest
    of the pipeline consumes. The planner calls
    :meth:`SourceResolver.resolve_detailed` instead, because the confirmations
    and lineage warnings belong on the printed plan.
    """

    model_config = ConfigDict(extra="forbid")

    resolved: ResolvedSource
    confirmations: list[ConfirmationReason] = Field(default_factory=list)
    options: list[SourceOption] = Field(default_factory=list)
    lineage_warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    explicit: bool = False
    """True when the user named the source with ``--source``."""

    derivative_confirmation_satisfied: bool = False
    """True when the third-party-derivative confirmation is already answered.

    ``SPEC.md`` section 6.3: choosing the source explicitly *is* the approval, so
    yonga records :data:`CONFIRMATION_EXPLICIT_SOURCE` and does not additionally
    raise :data:`~yonga.domain.plans.CONFIRMATION_THIRD_PARTY_DERIVATIVE`.
    """

    @property
    def requires_confirmation(self) -> bool:
        """True when at least one confirmation is attached to this resolution."""
        return bool(self.confirmations)

    def confirmation_codes(self) -> list[str]:
        """Return the codes of the attached confirmations, in order."""
        return [reason.code for reason in self.confirmations]

    def to_json(self) -> dict[str, object]:
        """Return a JSON-safe dictionary (enums as values, paths as strings)."""
        return dict(self.model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def _observation(metadata: ModelMetadata, fmt: ModelFormat) -> FormatObservation | None:
    """Return the strongest non-empty observation of *fmt*, or ``None``.

    "Strongest" means complete before incomplete, then larger before smaller, so
    a repository that ships both a full and a partial set of shards is judged on
    the full one.
    """
    best: FormatObservation | None = None
    for candidate in metadata.formats:
        if candidate.format is not fmt or not candidate.files:
            continue
        if best is None:
            best = candidate
            continue
        if (candidate.complete, candidate.total_bytes) > (best.complete, best.total_bytes):
            best = candidate
    return best


def _architecture_tokens(metadata: ModelMetadata) -> list[str]:
    """Return the casefolded architecture identifiers declared by a repository."""
    tokens: list[str] = []
    for raw in (metadata.model_type, *metadata.architectures):
        if not raw:
            continue
        token = raw.strip().casefold()
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def _supported_gguf_architecture(metadata: ModelMetadata, policy: SourcePolicy) -> str | None:
    """Return the architecture the backend can read from GGUF, or ``None``.

    Matching is an exact, casefolded comparison against
    :attr:`SourcePolicy.gguf_architectures`. A repository whose architecture is
    unknown never matches: "we could not tell" is not support.
    """
    supported = {name.strip().casefold() for name in policy.gguf_architectures if name.strip()}
    if not supported:
        return None
    for token in _architecture_tokens(metadata):
        if token in supported:
            return token
    return None


def _architecture_label(metadata: ModelMetadata) -> str:
    """Human-readable architecture name for messages, or ``"this architecture"``."""
    if metadata.architectures:
        return metadata.architectures[0]
    if metadata.model_type:
        return metadata.model_type
    return "this architecture"


def _candidate_detail(candidate: DerivativeCandidate) -> str:
    """Describe one declared derivative for the inspect table."""
    parts = [f"declared {candidate.relationship.value}", f"trust {candidate.trust.value}"]
    if candidate.trust is TrustPolicy.NEVER_AUTO:
        parts.append("never selected automatically; pass --source to use it")
    if candidate.note:
        parts.append(candidate.note.strip())
    return "; ".join(parts)


def describe_source_options(
    metadata: ModelMetadata,
    candidates: Sequence[DerivativeCandidate] = (),
    *,
    policy: SourcePolicy | None = None,
) -> list[SourceOption]:
    """Describe every preparation route for a repository, without any I/O.

    This is what ``yonga inspect`` prints under "Preparation"
    (``docs/CLI_SPEC.md`` section 4). It reads the metadata already fetched and
    the derivative candidates the registry declared; it never contacts the Hub,
    so ``inspect`` stays a metadata-only command.
    """
    active = DEFAULT_SOURCE_POLICY if policy is None else policy
    options: list[SourceOption] = []

    openvino = _observation(metadata, ModelFormat.OPENVINO_IR)
    detail: str | None = None
    if openvino is not None and not openvino.complete:
        detail = "OpenVINO IR present but incomplete; a weight file is missing."
    elif openvino is not None and not active.allow_openvino_direct:
        detail = "OpenVINO IR present, but direct reuse is disabled by policy."
    options.append(
        SourceOption(
            label=OPTION_OPENVINO,
            available=openvino is not None and openvino.complete and active.allow_openvino_direct,
            repo_id=metadata.identity.repo_id if openvino is not None else None,
            detail=detail,
        )
    )

    safetensors = _observation(metadata, ModelFormat.SAFETENSORS)
    options.append(
        SourceOption(
            label=OPTION_SAFETENSORS,
            available=safetensors is not None,
            repo_id=metadata.identity.repo_id if safetensors is not None else None,
        )
    )

    pytorch = _observation(metadata, ModelFormat.PYTORCH_BIN)
    if pytorch is not None or safetensors is None:
        options.append(
            SourceOption(
                label=OPTION_PYTORCH,
                available=pytorch is not None,
                repo_id=metadata.identity.repo_id if pytorch is not None else None,
            )
        )

    if candidates:
        for candidate in candidates:
            options.append(
                SourceOption(
                    label=OPTION_DERIVATIVE,
                    available=active.allow_derivatives,
                    repo_id=candidate.repo_id,
                    detail=_candidate_detail(candidate),
                    requires_confirmation=candidate.requires_confirmation
                    or not active.honour_registry_auto_trust,
                )
            )
    else:
        options.append(
            SourceOption(
                label=OPTION_DERIVATIVE,
                available=False,
                detail="No derivative source is declared for this repository.",
            )
        )

    gguf = _observation(metadata, ModelFormat.GGUF)
    if gguf is not None:
        architecture = _supported_gguf_architecture(metadata, active)
        options.append(
            SourceOption(
                label=OPTION_GGUF,
                available=architecture is not None,
                repo_id=metadata.identity.repo_id,
                detail=None
                if architecture is not None
                else (
                    "OpenVINO GenAI cannot read GGUF for "
                    f"{_architecture_label(metadata)}; the presence of GGUF files is "
                    "not evidence that it can."
                ),
            )
        )

    return options


# --------------------------------------------------------------------------- #
# The resolver
# --------------------------------------------------------------------------- #


class SourceResolver:
    """Chooses the repository and revision a preparation job will convert.

    The resolver owns no state beyond its injected collaborators, so one instance
    is safe to reuse across models.
    """

    def __init__(self, hub: HubClient, *, progress: ProgressSink | None = None) -> None:
        """Bind the Hub client used to inspect and pin candidate repositories."""
        self._hub = hub
        self._progress: ProgressSink = null_sink() if progress is None else progress

    # -- public API ---------------------------------------------------- #

    def resolve(
        self,
        metadata: ModelMetadata,
        *,
        explicit_source: ModelIdentity | str | None = None,
        registry_resolver: RegistryResolver,
        policy: SourcePolicy | None = None,
    ) -> ResolvedSource:
        """Resolve the conversion source for the repository described by *metadata*.

        *metadata* is the requested repository's own metadata, as returned by
        :meth:`yonga.adapters.HubClient.inspect`. *explicit_source* is the user's
        ``--source`` argument, which is honoured but still recorded as a lineage
        warning. Raises
        :class:`~yonga.domain.diagnostics.SourceResolutionError` when nothing
        usable exists.

        Use :meth:`resolve_detailed` when the caller also needs the confirmation
        reasons and lineage warnings that belong on a preparation plan.
        """
        return self.resolve_detailed(
            metadata,
            explicit_source=explicit_source,
            registry_resolver=registry_resolver,
            policy=policy,
        ).resolved

    def options(
        self,
        metadata: ModelMetadata,
        *,
        registry_resolver: RegistryResolver,
        policy: SourcePolicy | None = None,
    ) -> list[SourceOption]:
        """Describe the routes available for a repository without resolving one."""
        candidates, _explanation = registry_resolver.resolve_derivatives(metadata.identity)
        return describe_source_options(metadata, candidates, policy=policy)

    def resolve_detailed(
        self,
        metadata: ModelMetadata,
        *,
        explicit_source: ModelIdentity | str | None = None,
        registry_resolver: RegistryResolver,
        policy: SourcePolicy | None = None,
    ) -> SourceResolution:
        """Resolve the conversion source and everything a plan must disclose."""
        active = DEFAULT_SOURCE_POLICY if policy is None else policy
        requested = metadata.identity
        candidates, derivative_explanation = registry_resolver.resolve_derivatives(requested)
        model_rule, _model_explanation = registry_resolver.resolve_model_rule(metadata)
        options = describe_source_options(metadata, candidates, policy=active)
        self._emit(
            ProgressKind.STAGE,
            f"Resolving a conversion source for {requested.repo_id}",
            repo_id=requested.repo_id,
        )

        if explicit_source is not None:
            return self._resolve_explicit(
                metadata,
                explicit_source=explicit_source,
                candidates=candidates,
                rule_id=derivative_explanation.winner_id,
                options=options,
                policy=active,
            )

        problems: list[str] = []

        openvino = _observation(metadata, ModelFormat.OPENVINO_IR)
        if openvino is not None:
            if not active.allow_openvino_direct:
                problems.append(
                    "An OpenVINO IR artifact is present but direct reuse is disabled by policy."
                )
            elif active.require_complete_openvino_ir and not openvino.complete:
                problems.append(
                    "The OpenVINO IR artifact in the requested repository is incomplete "
                    "(a weight file referenced by an .xml is missing), so it cannot be "
                    "used directly."
                )
            else:
                return self._in_repo(
                    metadata,
                    openvino,
                    options=options,
                    model_rule=model_rule,
                )

        for fmt in (ModelFormat.SAFETENSORS, ModelFormat.PYTORCH_BIN):
            observation = _observation(metadata, fmt)
            if observation is not None:
                return self._in_repo(metadata, observation, options=options, model_rule=model_rule)

        if active.allow_derivatives:
            resolution = self._resolve_derivative(
                metadata,
                candidates=candidates,
                rule_id=derivative_explanation.winner_id,
                options=options,
                policy=active,
                problems=problems,
            )
            if resolution is not None:
                return resolution
        elif candidates:
            problems.append(
                "Declared derivative sources exist but are disabled by the active policy."
            )

        gguf = _observation(metadata, ModelFormat.GGUF)
        if gguf is not None:
            architecture = _supported_gguf_architecture(metadata, active)
            if architecture is not None:
                return self._in_repo(metadata, gguf, options=options, model_rule=model_rule)
            raise SourceResolutionError(
                self._gguf_diagnostic(metadata, candidates=candidates, problems=problems)
            )

        raise SourceResolutionError(
            self._unresolved_diagnostic(metadata, candidates=candidates, problems=problems)
        )

    # -- resolution branches -------------------------------------------- #

    def _in_repo(
        self,
        metadata: ModelMetadata,
        observation: FormatObservation,
        *,
        options: Sequence[SourceOption],
        model_rule: ModelRule | None,
    ) -> SourceResolution:
        """Build the resolution for weights found in the requested repository."""
        pinned, sha = self._pin(metadata.identity, metadata)
        evidence = [
            SourceEvidence(
                kind="repo_files",
                note=(
                    f"{len(observation.files)} {observation.format.value} file(s) listed in "
                    f"{metadata.identity.repo_id} at {sha}."
                ),
            )
        ]
        rule_id: str | None = None
        already_openvino = observation.format is ModelFormat.OPENVINO_IR
        if (
            already_openvino
            and model_rule is not None
            and model_rule.artifact.format is ModelFormat.OPENVINO_IR
        ):
            rule_id = model_rule.id
            evidence.extend(source.as_evidence() for source in model_rule.sources)
        reasons = [_FORMAT_REASONS[observation.format]]
        if already_openvino:
            reasons.append("The artifact is still validated locally before it is declared ready.")
        resolved = ResolvedSource(
            requested=metadata.identity,
            source=pinned,
            relationship=SourceRelationship.ORIGINAL,
            format=observation.format,
            already_openvino=already_openvino,
            evidence=evidence,
            requires_confirmation=False,
            reasons=reasons,
            rule_id=rule_id,
            metadata=metadata.model_copy(update={"resolved_revision": sha}),
        )
        self._emit(
            ProgressKind.STAGE,
            f"Using {pinned.repo_id} ({observation.format.value}) as the conversion source",
            repo_id=pinned.repo_id,
            format=observation.format.value,
        )
        return SourceResolution(resolved=resolved, options=list(options))

    def _resolve_derivative(
        self,
        metadata: ModelMetadata,
        *,
        candidates: Sequence[DerivativeCandidate],
        rule_id: str | None,
        options: Sequence[SourceOption],
        policy: SourcePolicy,
        problems: list[str],
    ) -> SourceResolution | None:
        """Offer the best declared derivative, or ``None`` when none qualifies.

        ``never_auto`` candidates are reported in *problems* so the failure
        message can name them, but they are never returned: the user has to ask
        for one by name with ``--source``.
        """
        selectable: list[DerivativeCandidate] = []
        for candidate in candidates:
            if candidate.trust is TrustPolicy.NEVER_AUTO:
                problems.append(
                    f"{candidate.repo_id} is declared with trust policy 'never_auto', so yonga "
                    f"never selects it on its own; pass --source {candidate.repo_id} to use it "
                    f"deliberately."
                )
                continue
            selectable.append(candidate)
        selectable.sort(key=lambda item: _TRUST_ORDER[item.trust])

        for candidate in selectable:
            identity = ModelIdentity(repo_id=candidate.repo_id, revision=candidate.revision)
            try:
                derivative_metadata = self._hub.inspect(identity)
            except CancelledError:
                raise
            except YongaError as error:
                problems.append(
                    f"{candidate.repo_id} is declared as a source but could not be read: "
                    f"{error.diagnostic.message}"
                )
                continue
            observation = self._conversion_observation(derivative_metadata)
            if observation is None:
                problems.append(
                    f"{candidate.repo_id} is declared as a source but publishes no weights the "
                    f"OpenVINO exporter can read."
                )
                continue
            pinned, sha = self._pin(identity, derivative_metadata)
            needs_confirmation = (
                candidate.requires_confirmation or not policy.honour_registry_auto_trust
            )
            reason_text = (
                f"The requested repository {metadata.identity.repo_id} publishes no weights the "
                f"OpenVINO exporter can read. The registry declares {candidate.repo_id} as a "
                f"{candidate.relationship.value} of it."
            )
            resolved = ResolvedSource(
                requested=metadata.identity,
                source=pinned,
                relationship=candidate.relationship,
                format=observation.format,
                already_openvino=observation.format is ModelFormat.OPENVINO_IR,
                evidence=candidate.source_evidence(),
                requires_confirmation=needs_confirmation,
                reasons=[reason_text],
                rule_id=rule_id,
                metadata=derivative_metadata.model_copy(update={"resolved_revision": sha}),
            )
            confirmations: list[ConfirmationReason] = []
            if needs_confirmation:
                confirmations.append(
                    ConfirmationReason(
                        code=CONFIRMATION_THIRD_PARTY_DERIVATIVE,
                        message=(
                            f"A third-party derivative will be used: yonga would convert "
                            f"{candidate.repo_id} instead of {metadata.identity.repo_id}. The "
                            f"registry declares it as a {candidate.relationship.value} under "
                            f"trust policy '{candidate.trust.value}', which means it is a "
                            f"suggestion and never a silent substitution. Approve it only if "
                            f"you accept that lineage."
                        ),
                        security_sensitive=False,
                    )
                )
                self._emit(
                    ProgressKind.STAGE,
                    f"{candidate.repo_id} is offered as a declared derivative and needs approval",
                    severity=Severity.WARNING,
                    repo_id=candidate.repo_id,
                    trust=candidate.trust.value,
                )
            return SourceResolution(
                resolved=resolved,
                confirmations=confirmations,
                options=list(options),
                lineage_warnings=[
                    f"{candidate.repo_id} is a third-party derivative of "
                    f"{metadata.identity.repo_id}. The lineage is the one declared in the "
                    f"registry; yonga did not verify the weights themselves."
                ],
                notes=list(problems),
            )
        return None

    def _resolve_explicit(
        self,
        metadata: ModelMetadata,
        *,
        explicit_source: ModelIdentity | str,
        candidates: Sequence[DerivativeCandidate],
        rule_id: str | None,
        options: Sequence[SourceOption],
        policy: SourcePolicy,
    ) -> SourceResolution:
        """Honour a user-chosen ``--source``, recording the lineage it asserts.

        The source is used even when the registry knows nothing about it, because
        the user asserted the substitution — but the assertion is written down:
        the relationship stays :data:`SourceRelationship.ORIGINAL` unless the
        registry actually declares one, and a lineage warning always accompanies
        the result. The one thing an explicit source cannot do is buy GGUF
        support the backend does not have.
        """
        identity = (
            ModelIdentity.parse(explicit_source)
            if isinstance(explicit_source, str)
            else explicit_source
        )
        requested = metadata.identity
        same_repo = identity.repo_id == requested.repo_id
        source_metadata = metadata if same_repo else self._hub.inspect(identity)
        observation = self._conversion_observation(source_metadata)
        if observation is None:
            gguf = _observation(source_metadata, ModelFormat.GGUF)
            if gguf is not None and _supported_gguf_architecture(source_metadata, policy) is None:
                raise SourceResolutionError(
                    self._gguf_diagnostic(source_metadata, candidates=candidates, problems=[])
                )
            raise SourceResolutionError(
                self._unresolved_diagnostic(source_metadata, candidates=candidates, problems=[])
            )
        pinned, sha = self._pin(identity, source_metadata)

        declared = self._declared_candidate(identity, candidates)
        relationship = (
            declared.relationship if declared is not None else SourceRelationship.ORIGINAL
        )
        evidence = [
            SourceEvidence(
                kind="user_choice",
                note=f"{identity.repo_id} was named explicitly with --source.",
            )
        ]
        if declared is not None:
            evidence.extend(declared.source_evidence())

        lineage_warnings = [
            f"Lineage was asserted by you, not verified by yonga: {identity.repo_id} will be "
            f"converted for {requested.repo_id}."
            if not same_repo
            else (
                f"{identity.repo_id} was named explicitly with --source; it is the requested "
                f"repository itself."
            )
        ]
        if declared is None and not same_repo:
            lineage_warnings.append(
                "The registry declares no lineage between these repositories. yonga never infers "
                "one from repository names."
            )
        reasons = [
            f"{identity.repo_id} was selected explicitly with --source, which overrides the "
            f"preference order in SPEC.md section 8.1."
        ]
        if declared is not None:
            reasons.append(
                f"The registry declares it as a {declared.relationship.value} of "
                f"{requested.repo_id}."
            )

        confirmations: list[ConfirmationReason] = []
        if not same_repo:
            confirmations.append(
                ConfirmationReason(
                    code=CONFIRMATION_EXPLICIT_SOURCE,
                    message=(
                        f"You chose {identity.repo_id} as the conversion source for "
                        f"{requested.repo_id}. yonga records the substitution and its lineage "
                        f"warning in the artifact manifest; because you named the source "
                        f"yourself, the third-party derivative confirmation is already "
                        f"satisfied."
                    ),
                    security_sensitive=False,
                )
            )
            self._emit(
                ProgressKind.STAGE,
                f"Using explicitly requested source {identity.repo_id}",
                severity=Severity.WARNING,
                repo_id=identity.repo_id,
            )

        resolved = ResolvedSource(
            requested=requested,
            source=pinned,
            relationship=relationship,
            format=observation.format,
            already_openvino=observation.format is ModelFormat.OPENVINO_IR,
            evidence=evidence,
            requires_confirmation=not same_repo,
            reasons=reasons,
            rule_id=rule_id if declared is not None else None,
            metadata=source_metadata.model_copy(update={"resolved_revision": sha}),
        )
        return SourceResolution(
            resolved=resolved,
            confirmations=confirmations,
            options=list(options),
            lineage_warnings=lineage_warnings,
            explicit=True,
            derivative_confirmation_satisfied=True,
        )

    # -- shared mechanism ------------------------------------------------ #

    @staticmethod
    def _declared_candidate(
        identity: ModelIdentity,
        candidates: Sequence[DerivativeCandidate],
    ) -> DerivativeCandidate | None:
        """Return the registry candidate for *identity*, matched by exact repo id.

        Exact equality only. A repository that merely looks related to a declared
        candidate is not one.
        """
        for candidate in candidates:
            if candidate.repo_id.casefold() == identity.repo_id.casefold():
                return candidate
        return None

    @staticmethod
    def _conversion_observation(metadata: ModelMetadata) -> FormatObservation | None:
        """Return the best weight format an export can read, or ``None``."""
        for fmt in CONVERSION_FORMAT_PREFERENCE:
            observation = _observation(metadata, fmt)
            if observation is None:
                continue
            if fmt is ModelFormat.OPENVINO_IR and not observation.complete:
                continue
            return observation
        return None

    def _pin(
        self,
        identity: ModelIdentity,
        metadata: ModelMetadata | None = None,
    ) -> tuple[ModelIdentity, str]:
        """Pin *identity* to an immutable commit SHA, asking the Hub if needed.

        A branch or tag is never accepted: ``docs/MODEL_PIPELINE.md`` section 4
        requires every download and every manifest entry for a job to name one
        commit.
        """
        revision: str | None = None
        if metadata is not None and is_pinned_revision(metadata.resolved_revision):
            revision = metadata.resolved_revision
        elif is_pinned_revision(identity.revision):
            revision = identity.revision
        if revision is None:
            revision = self._hub.resolve_revision(identity)
        if not is_pinned_revision(revision):
            raise SourceResolutionError(
                diagnostic(
                    CODE_REVISION_UNPINNED,
                    FailureCategory.SOURCE_FORMAT_UNSUPPORTED,
                    "Source revision could not be pinned",
                    (
                        f"{identity.repo_id} could not be pinned to an immutable commit. "
                        f"yonga refuses to prepare from a branch or tag, because the artifact "
                        f"would not be reproducible."
                    ),
                    remediation=[
                        f"Pass an explicit commit, e.g. --revision <40-character sha> for "
                        f"{identity.repo_id}.",
                        "Check that the repository and revision exist and that you have access.",
                    ],
                    context={"repo_id": identity.repo_id, "revision": identity.revision or ""},
                )
            )
        return identity.with_revision(revision), revision

    def _emit(
        self,
        kind: ProgressKind,
        message: str,
        *,
        severity: Severity = Severity.INFO,
        **data: object,
    ) -> None:
        """Emit one progress event; a renderer failure must never break resolution."""
        self._progress.emit(
            ProgressEvent(kind=kind, message=message, severity=severity, data=dict(data))
        )

    # -- failure diagnostics --------------------------------------------- #

    @staticmethod
    def _choices(
        metadata: ModelMetadata,
        candidates: Sequence[DerivativeCandidate],
        problems: Sequence[str],
    ) -> list[str]:
        """Build the remediation list shared by both failure diagnostics."""
        remediation = list(problems)
        if candidates:
            names = ", ".join(candidate.repo_id for candidate in candidates)
            remediation.append(
                f"Declared alternative source(s) for this repository: {names}. Review the "
                f"lineage first, then pass --source <repo id> to use one."
            )
        else:
            remediation.append(
                f"No alternative source is declared for {metadata.identity.repo_id}. yonga will "
                f"not guess one from repository names."
            )
        remediation.extend(
            [
                "Pass --source <repo id> to name a repository that publishes safetensors or "
                "standard Transformers weights for this model.",
                "Or declare the lineage in a registry override so the substitution is "
                "auditable (see docs/REGISTRY_SPEC.md).",
            ]
        )
        return remediation

    @staticmethod
    def _gguf_diagnostic(
        metadata: ModelMetadata,
        *,
        candidates: Sequence[DerivativeCandidate],
        problems: Sequence[str],
    ) -> Diagnostic:
        """Explain that GGUF weights exist but cannot be read for this architecture."""
        return diagnostic(
            CODE_GGUF_UNSUPPORTED,
            FailureCategory.SOURCE_FORMAT_UNSUPPORTED,
            "This repository publishes GGUF weights only",
            (
                f"{metadata.identity.repo_id} publishes GGUF weights only. The OpenVINO GenAI "
                f"build yonga is running has no direct GGUF reader for "
                f"{_architecture_label(metadata)}, and the presence of .gguf files is not "
                f"evidence that it has one. yonga stops here rather than start a conversion "
                f"that cannot succeed."
            ),
            remediation=SourceResolver._choices(metadata, candidates, problems),
            context={"repo_id": metadata.identity.repo_id, "format": ModelFormat.GGUF.value},
        )

    @staticmethod
    def _unresolved_diagnostic(
        metadata: ModelMetadata,
        *,
        candidates: Sequence[DerivativeCandidate],
        problems: Sequence[str],
    ) -> Diagnostic:
        """Explain that nothing convertible was found in or declared for a repository."""
        observed = sorted(
            {observation.format.value for observation in metadata.formats if observation.files}
        )
        seen = ", ".join(observed) if observed else "no recognised weight files"
        return diagnostic(
            CODE_NO_USABLE_SOURCE,
            FailureCategory.SOURCE_FORMAT_UNSUPPORTED,
            "No usable conversion source",
            (
                f"yonga found {seen} in {metadata.identity.repo_id}, and none of it is a source "
                f"the OpenVINO exporter can read. No declared derivative could stand in either."
            ),
            remediation=SourceResolver._choices(metadata, candidates, problems),
            context={"repo_id": metadata.identity.repo_id, "formats": seen},
        )
