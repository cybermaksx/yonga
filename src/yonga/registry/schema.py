"""Typed records for the compatibility registry.

The registry turns fast-moving knowledge — which architecture runs on which
device, which exporter version it needs, what a known failure looks like, which
repair fixes it — into *data* instead of conditionals scattered through the
codebase. This module defines the shape of that data and nothing else: it reads
no files, performs no resolution and imports nothing heavier than Pydantic.

Two policies are baked into the types themselves:

* Registry data never carries executable instructions. A repair rule names a
  built-in Python implementation by key (:data:`KNOWN_REPAIR_IMPLEMENTATIONS`);
  it can never describe a find/replace to run.
* Every rule can be traced. Each rule carries ``sources``, and matching is
  explainable through :class:`MatchOutcome`, which records *why* a rule matched
  or was rejected.

Matching semantics (shared by every rule kind, see :class:`RuleMatch`):

* Inside one criterion the entries are alternatives (OR).
* ``model_types`` and ``architectures`` form a single *identity* criterion that
  is satisfied when either side matches, because a repository may expose only
  one of the two.
* ``aliases`` extends that same identity criterion to :attr:`identity hints
  <MatchSubject.identity_hints>`: identifiers a repository carries somewhere
  other than ``config.json``. A hint is compared *only* against ``aliases`` and
  never against ``model_types`` or ``architectures``, so teaching yonga a new
  spelling of an architecture is always a deliberate, reviewable registry edit
  rather than a fuzzy string match that happens to fire.
* Different criteria are combined with AND.
* An empty criterion imposes no constraint.

Identity hints are evidence, not resemblance
--------------------------------------------

:func:`subject_from_metadata` fills :attr:`MatchSubject.identity_hints` from the
GGUF file metadata the Hub reports (``gguf_metadata["architecture"]``) and from
nothing else. That value is read out of the weight file itself, so it is a
statement the artifact makes about what it is.

Repository *tags* are deliberately excluded. They are free-form text any
uploader can write, and matching a tag such as ``qwen`` or ``qwen3.5`` would be
exactly the "infer that a similarly named repository is equivalent" mistake
``SPEC.md`` section 8.1 forbids. A tag may well be true; it is not evidence.

A hint is consulted *only* when the repository declares no identity of its own:
an empty ``model_type`` and an empty ``architectures`` list. A GGUF header is
evidence in the *absence* of a declaration, never a correction of one. When the
repository declares something, the declaration wins, so an identity no rule
recognises resolves to *nothing* rather than to whatever rule the header
happens to name — a conflict between a declaration and a header means
"unknown", never "pick the other one". Substituting the header's architecture
for a declared one would hand the user another model's exporter pins, pipeline
and device verdict, which ``SPEC.md`` section 3 forbids ("silently substitute a
semantically different model") and section 8.1 forbids inferring.

Specificity ladder
------------------

Exact repository id (100) > architecture plus device/host family (80) >
architecture (60) > architecture reached through an alias
(:data:`SPECIFICITY_ARCHITECTURE_ALIAS`, 50) > task (40) > generic (20). The
alias rung sits below every declared-identity rung on purpose: a rule that
recognises the model type a repository actually declares must always beat one
that had to fall back on a hint found outside ``config.json``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, ClassVar, Literal

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from yonga.domain.compatibility import DeviceStatus, EvidenceLevel
from yonga.domain.hardware import normalise_device
from yonga.domain.models import ModelFormat, PipelineKind, SourceEvidence, SourceRelationship
from yonga.domain.plans import (
    ConversionProfile,
    QuantizationSettings,
    export_extra_arg_problems,
)

#: Schema version understood by this build of the loader.
#:
#: Version 2 added :attr:`RuleMatch.parameter_count` (a profile's model-size
#: band) and :attr:`ProfileRule.evidence`. Version 3 added
#: :attr:`ChatProfileRule.tool_calling`. Older files still load unchanged:
#: every new field defaults to "no constraint" or "not declared".
SCHEMA_VERSION: int = 3

#: Oldest yonga release that understands each registry schema version, written
#: into a remote bundle as ``min_tool_version`` so an older yonga refuses the
#: bundle at ``registry update`` instead of failing to parse it.
SCHEMA_MIN_TOOL_VERSION: dict[int, str] = {2: "0.3.0", 3: "0.4.0"}

#: The schema version a rule must declare to use a parameter-count band.
PARAMETER_BAND_SCHEMA_VERSION: int = 2

#: The schema version a chat profile must declare to carry ``tool_calling``.
TOOL_CALLING_SCHEMA_VERSION: int = 3

#: Specificity of a rule that names an exact repository id.
SPECIFICITY_REPO_ID: int = 100
#: Specificity of an architecture rule that also constrains device or host family.
SPECIFICITY_ARCHITECTURE_FAMILY: int = 80
#: Specificity of a plain architecture (model type / architecture class) rule.
SPECIFICITY_ARCHITECTURE: int = 60
#: Specificity of an architecture rule reached only through an identity *alias*.
#:
#: Below :data:`SPECIFICITY_ARCHITECTURE` on purpose: an alias match means the
#: repository never declared the model type, so any rule that did recognise a
#: declared identity is the better answer.
SPECIFICITY_ARCHITECTURE_ALIAS: int = 50
#: Specificity of a task-scoped rule.
SPECIFICITY_TASK: int = 40
#: Specificity of a generic device or profile rule.
SPECIFICITY_GENERIC: int = 20

#: Human-readable name of the source :func:`subject_from_metadata` reads hints from.
#:
#: Kept beside the reader so the sentence ``yonga registry explain`` prints and
#: the code that fills :attr:`MatchSubject.identity_hints` cannot drift apart.
IDENTITY_HINT_SOURCE: str = "GGUF metadata"

#: Built-in repair implementations a registry rule is allowed to select.
#:
#: The name is a key into the Python repair registry; YAML must never describe
#: the transformation itself. ``validate_registry`` rejects unknown names.
KNOWN_REPAIR_IMPLEMENTATIONS: frozenset[str] = frozenset({"tokenizer_string_precision"})

#: Tool-call output formats a chat profile's ``tool_calling`` may select.
#:
#: Each is a generic parser compiled into yonga
#: (:mod:`yonga.application.tool_calls`), parameterised by the rule's markers
#: and field names; registry data selects one and can never describe a new one.
#: ``validate_registry`` reports an unknown name, and a chat profile naming one
#: this build lacks has no tool calling.
#:
#: ``tagged_json``
#:     Each call is one JSON object between the ``open`` and ``close`` markers,
#:     repeatable, with ordinary text allowed around the blocks.
#: ``json_list``
#:     The whole answer is one JSON array of call objects, optionally after the
#:     ``open`` marker (which a detokenizer may already have removed, when the
#:     marker is a special token) and before the ``close`` marker.
KNOWN_TOOL_CALL_FORMATS: frozenset[str] = frozenset({"tagged_json", "json_list"})

#: Distribution names whose versions may appear in a ``constraints`` block.
KNOWN_CONSTRAINT_KEYS: frozenset[str] = frozenset(
    {
        "openvino",
        "openvino_genai",
        "openvino_tokenizers",
        "optimum",
        "optimum_intel",
        "nncf",
        "transformers",
        "torch",
        "huggingface_hub",
        "python",
    }
)

_STRICT = ConfigDict(extra="forbid")
_STRICT_ALIASED = ConfigDict(extra="forbid", populate_by_name=True)


def _checked_export_args(value: list[str]) -> list[str]:
    """Refuse export arguments that would override what yonga decides itself."""
    problems = export_extra_arg_problems(value)
    if problems:
        raise ValueError("export arguments not allowed in registry data: " + "; ".join(problems))
    return value


def _normalise_key(value: str) -> str:
    """Canonicalise a package/constraint key (``optimum-intel`` -> ``optimum_intel``)."""
    return value.strip().lower().replace("-", "_")


def _casefold_all(values: Iterable[str]) -> list[str]:
    """Casefold and de-duplicate a list of match tokens, preserving order."""
    seen: list[str] = []
    for value in values:
        cleaned = value.strip().casefold()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


class RuleStatus(str, Enum):
    """Lifecycle status of a registry rule."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"

    @property
    def is_usable(self) -> bool:
        """True when the rule may take part in resolution."""
        return self is not RuleStatus.DEPRECATED


class SourceKind(str, Enum):
    """Kind of provenance backing a registry claim."""

    OFFICIAL_DOCS = "official_docs"
    UPSTREAM_ISSUE = "upstream_issue"
    TESTED = "tested"
    MODEL_CARD = "model_card"
    COMMUNITY = "community"


class TrustPolicy(str, Enum):
    """How far a declared derivative source may be used without asking."""

    SUGGEST = "suggest"
    AUTO_IF_TRUSTED = "auto_if_trusted"
    NEVER_AUTO = "never_auto"

    @property
    def requires_confirmation(self) -> bool:
        """True when the user must approve before this source is used."""
        return self is not TrustPolicy.AUTO_IF_TRUSTED


class RepairSafety(str, Enum):
    """How much approval a repair needs before it may touch artifact files."""

    AUTOMATIC = "automatic"
    CONFIRM = "confirm"
    MANUAL = "manual"


class SourceRef(BaseModel):
    """One provenance entry backing a registry claim."""

    model_config = _STRICT

    kind: SourceKind
    url: str | None = None
    note: str | None = None

    def as_mapping(self) -> dict[str, str]:
        """Render as a flat string mapping for embedding in domain models."""
        data: dict[str, str] = {"kind": self.kind.value}
        if self.url is not None:
            data["url"] = self.url
        if self.note is not None:
            data["note"] = self.note
        return data

    def as_evidence(self) -> SourceEvidence:
        """Render as a :class:`~yonga.domain.models.SourceEvidence` entry."""
        return SourceEvidence(kind=self.kind.value, url=self.url, note=self.note)


class MatchOutcome(BaseModel):
    """Result of evaluating one rule's match block against a subject."""

    model_config = _STRICT

    matched: bool
    specificity: int = SPECIFICITY_GENERIC
    reasons: list[str] = Field(default_factory=list)

    alias_hint: str | None = None
    """The identity hint that satisfied identity through :attr:`RuleMatch.aliases`.

    ``None`` whenever identity was settled by a declared ``model_type`` or
    architecture class, so a caller can tell a hint-based identification apart
    from a ``config.json`` one without parsing :attr:`reasons`.
    """


class MatchSubject(BaseModel):
    """The facts a rule is matched against."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo_id: str | None = None
    model_type: str | None = None
    architectures: tuple[str, ...] = ()
    task: str | None = None
    device: str | None = None
    host_tags: tuple[str, ...] = ()

    identity_hints: tuple[str, ...] = ()
    """Identifiers discovered somewhere other than ``config.json``.

    Weaker than a declared ``model_type`` but still authoritative, because they
    are read out of the weight file rather than typed into a repository card.
    They are only ever compared against :attr:`RuleMatch.aliases`; see the
    module docstring for why repository tags are not hints.
    """

    parameter_count: int | None = Field(default=None, ge=1)
    """The model's parameter count as the Hub reports it, or ``None`` when unusable.

    Only :attr:`RuleMatch.parameter_count` reads it. ``None`` means the size is
    unknown, or known but not a fair basis for choosing a quantization (the
    weights are already packed integers, or the source keeps its own
    quantization); :attr:`parameter_count_basis` says which.
    """

    parameter_count_basis: str | None = None
    """Where :attr:`parameter_count` came from, or why it is missing. Printed verbatim."""

    @field_validator("device")
    @classmethod
    def _normalise_device_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalise_device(value) or None


_COUNT_SUFFIXES: dict[str, int] = {"k": 10**3, "m": 10**6, "b": 10**9, "t": 10**12}
_COUNT_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmbt])\s*$", re.IGNORECASE)
_COUNT_HINT = "write 2000000000, 2_000_000_000 or 2B (YAML reads 2e9 as text)"


def _parse_count(value: object) -> object:
    """Accept an integer, or a decimal with a K/M/B/T suffix; refuse everything else.

    A float is refused on purpose: ``yaml.safe_load`` reads ``2e9`` as the
    *string* ``"2e9"`` but ``2.0e+9`` as a float, and a threshold that silently
    depended on how it was spelled is not a threshold.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"a parameter count cannot be a boolean; {_COUNT_HINT}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip().replace("_", "")
        if stripped.isdigit():
            return int(stripped)
        found = _COUNT_PATTERN.match(value.replace("_", ""))
        if found is not None:
            try:
                amount = Decimal(found.group(1)) * _COUNT_SUFFIXES[found.group(2).lower()]
            except InvalidOperation as error:  # pragma: no cover - the pattern prevents it
                raise ValueError(f"unreadable parameter count {value!r}") from error
            if amount != amount.to_integral_value():
                raise ValueError(f"parameter count {value!r} is not a whole number")
            return int(amount)
    raise ValueError(f"unreadable parameter count {value!r}; {_COUNT_HINT}")


def format_parameter_count(count: int) -> str:
    """A short human spelling of a parameter count: ``752M``, ``1.24B``, ``8B``."""
    for suffix, scale in (("T", 10**12), ("B", 10**9), ("M", 10**6), ("K", 10**3)):
        if count >= scale:
            # Truncated, never rounded: 1,996,000,000 must not read as "2B" next
            # to a threshold of 2B it is below.
            precision = 2 if count < 10 * scale else 1
            step = scale // 10**precision
            whole, rest = divmod(count // step, 10**precision)
            text = f"{whole}.{rest:0{precision}d}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    return str(count)


class CountRange(BaseModel):
    """A half-open range of parameter counts: ``at_least`` <= count < ``below``.

    The unit is parameters as the Hugging Face Hub counts them from the
    safetensors headers: embeddings, an untied output layer and any vision
    tower included. Half-open so that adjacent bands never overlap: a model
    exactly at a threshold belongs to the upper band.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    at_least: int | None = Field(default=None, ge=1)
    below: int | None = Field(default=None, ge=1)

    @field_validator("at_least", "below", mode="before")
    @classmethod
    def _read_count(cls, value: object) -> object:
        return _parse_count(value)

    @model_validator(mode="after")
    def _bounded_and_ordered(self) -> CountRange:
        if self.at_least is None and self.below is None:
            raise ValueError("a parameter_count band needs at_least, below, or both")
        if self.at_least is not None and self.below is not None and self.at_least >= self.below:
            raise ValueError("parameter_count at_least must be smaller than below")
        return self

    def verdict(self, count: int | None) -> Literal["inside", "outside", "unknown"]:
        """Where ``count`` falls: inside the band, outside it, or unknown."""
        if count is None:
            return "unknown"
        if self.at_least is not None and count < self.at_least:
            return "outside"
        if self.below is not None and count >= self.below:
            return "outside"
        return "inside"

    def intersects(self, other: CountRange | None) -> bool:
        """True when one parameter count could fall inside both bands."""
        return _ranges_intersect(self, other)

    def predicate(self) -> str:
        """The band after "is": ``below 2B``, ``8B or more``, ``in the range 2B to below 8B``."""
        if self.at_least is None and self.below is not None:
            return f"below {format_parameter_count(self.below)}"
        if self.below is None and self.at_least is not None:
            return f"{format_parameter_count(self.at_least)} or more"
        return f"in the range {self.describe().removeprefix('of ')}"

    def describe(self) -> str:
        """The band after "models": ``below 2B``, ``of 8B or more``, ``of 2B to below 8B``."""
        if self.at_least is None and self.below is not None:
            return f"below {format_parameter_count(self.below)}"
        if self.below is None and self.at_least is not None:
            return f"of {format_parameter_count(self.at_least)} or more"
        assert self.at_least is not None and self.below is not None
        return (
            f"of {format_parameter_count(self.at_least)} to below "
            f"{format_parameter_count(self.below)}"
        )


def _ranges_intersect(left: CountRange | None, right: CountRange | None) -> bool:
    """True when some count satisfies both bands; ``None`` is unbounded."""
    if left is None or right is None:
        return True
    lows = [value for value in (left.at_least, right.at_least) if value is not None]
    highs = [value for value in (left.below, right.below) if value is not None]
    low = max(lows) if lows else 0
    return not highs or low < min(highs)


class RuleMatch(BaseModel):
    """Criteria selecting the subjects a rule applies to.

    See the module docstring for the OR/AND semantics. ``model_types``,
    ``architectures`` and ``aliases`` together form the single *identity*
    criterion, with ``aliases`` matched only against identity hints.
    """

    model_config = _STRICT_ALIASED

    repo_ids: list[str] = Field(default_factory=list)
    model_types: list[str] = Field(default_factory=list)
    architectures: list[str] = Field(default_factory=list)

    aliases: list[str] = Field(default_factory=list)
    """Alternative spellings this architecture is known by in other metadata sources.

    Only :attr:`MatchSubject.identity_hints` are compared against this list, so
    an alias can never widen what a declared ``model_type`` matches.
    """

    tasks: list[str] = Field(default_factory=list)
    devices: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("devices", "device"),
    )
    host_tags: list[str] = Field(default_factory=list)

    parameter_count: CountRange | None = None
    """The model sizes this rule applies to. Only conversion profiles may set it.

    It filters and never ranks: a band adds no specificity rung, so among
    profiles that match, priority decides. A model whose size is unknown is
    outside every band.
    """

    @field_validator("devices", mode="before")
    @classmethod
    def _accept_single_device(cls, value: object) -> object:
        """Allow ``device: npu`` as shorthand for ``devices: [npu]``."""
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("devices")
    @classmethod
    def _normalise_devices(cls, value: list[str]) -> list[str]:
        names: list[str] = []
        for raw in value:
            name = normalise_device(raw)
            if name and name not in names:
                names.append(name)
        return names

    @property
    def identity_tokens(self) -> list[str]:
        """Casefolded model types and architecture class names, combined."""
        return _casefold_all([*self.model_types, *self.architectures])

    @property
    def alias_tokens(self) -> list[str]:
        """Casefolded identity aliases, compared only against identity hints."""
        return _casefold_all(self.aliases)

    @property
    def is_generic(self) -> bool:
        """True when the rule constrains nothing at all."""
        return not (
            self.repo_ids
            or self.model_types
            or self.architectures
            or self.aliases
            or self.tasks
            or self.host_tags
            or self.parameter_count is not None
        )

    def static_specificity(self) -> int:
        """Best specificity this rule could score, ignoring the subject.

        A rule carrying both declared identities and aliases reports the
        declared rung: that is the strongest outcome it can reach, and
        :meth:`evaluate` lowers it to :data:`SPECIFICITY_ARCHITECTURE_ALIAS`
        for the runs that actually had to fall back on an alias. Callers that
        need every rung a rule can compete on — registry validation above all —
        want :meth:`resolvable_specificities` instead.
        """
        if self.repo_ids:
            return SPECIFICITY_REPO_ID
        if self.model_types or self.architectures:
            if self.devices or self.host_tags:
                return SPECIFICITY_ARCHITECTURE_FAMILY
            return SPECIFICITY_ARCHITECTURE
        if self.aliases:
            return SPECIFICITY_ARCHITECTURE_ALIAS
        if self.tasks:
            return SPECIFICITY_TASK
        return SPECIFICITY_GENERIC

    def evaluate(self, subject: MatchSubject) -> MatchOutcome:
        """Evaluate this match block against ``subject``.

        Returns a :class:`MatchOutcome` whose ``reasons`` explain the decision in
        user-facing English, so ``registry explain`` can print it verbatim.
        """
        reasons: list[str] = []

        if self.repo_ids:
            wanted = _casefold_all(self.repo_ids)
            actual = (subject.repo_id or "").strip().casefold()
            if actual not in wanted:
                return MatchOutcome(
                    matched=False,
                    specificity=self.static_specificity(),
                    reasons=[f"repository {subject.repo_id or '<unknown>'} is not listed"],
                )
            reasons.append(f"repository {subject.repo_id} matches exactly")

        identity = self.identity_tokens
        aliases = self.alias_tokens
        alias_hint: str | None = None
        if identity or aliases:
            declared = _casefold_all([subject.model_type or "", *subject.architectures])
            hit = next((token for token in declared if token in identity), None)
            if hit is not None:
                reasons.append(f"model identity {hit} is listed")
            elif declared:
                # The repository declared an identity this rule does not list.
                # The alias path stops here: a GGUF header is evidence only in
                # the ABSENCE of a declaration. Consulting it now would let a
                # header override a contradicting config.json and report the
                # repository as a different architecture entirely, with that
                # architecture's exporter pins, pipeline and device verdict.
                # The declaration wins, and a conflict means "unknown".
                return MatchOutcome(
                    matched=False,
                    specificity=self.static_specificity(),
                    reasons=self._identity_rejection(subject, hints_considered=False),
                )
            else:
                alias_hint = next(
                    (token for token in _casefold_all(subject.identity_hints) if token in aliases),
                    None,
                )
                if alias_hint is None:
                    return MatchOutcome(
                        matched=False,
                        specificity=self.static_specificity(),
                        reasons=self._identity_rejection(subject, hints_considered=bool(aliases)),
                    )
                reasons.append(_alias_reason(alias_hint))

        if self.tasks:
            wanted_tasks = _casefold_all(self.tasks)
            actual_task = (subject.task or "").strip().casefold()
            if actual_task not in wanted_tasks:
                return MatchOutcome(
                    matched=False,
                    specificity=self.static_specificity(),
                    reasons=[f"task {subject.task or '<unknown>'} is not listed"],
                )
            reasons.append(f"task {subject.task} is listed")

        if self.devices:
            if subject.device is None or subject.device not in self.devices:
                return MatchOutcome(
                    matched=False,
                    specificity=self.static_specificity(),
                    reasons=[f"device {subject.device or '<unspecified>'} is not listed"],
                )
            reasons.append(f"device {subject.device} is listed")

        if self.host_tags:
            wanted_tags = set(_casefold_all(self.host_tags))
            actual_tags = set(_casefold_all(subject.host_tags))
            overlap = sorted(wanted_tags & actual_tags)
            if not overlap:
                return MatchOutcome(
                    matched=False,
                    specificity=self.static_specificity(),
                    reasons=["host does not carry any of the required device tags"],
                )
            reasons.append(f"host tag {overlap[0]} is present")

        if self.parameter_count is not None:
            band = self.parameter_count
            verdict = band.verdict(subject.parameter_count)
            if verdict == "unknown":
                basis = subject.parameter_count_basis or "no parameter count is known"
                return MatchOutcome(
                    matched=False,
                    specificity=self.static_specificity(),
                    reasons=[
                        f"parameter count unknown ({basis}); this applies only to models "
                        f"{band.describe()}"
                    ],
                )
            assert subject.parameter_count is not None
            size = format_parameter_count(subject.parameter_count)
            if verdict == "outside":
                return MatchOutcome(
                    matched=False,
                    specificity=self.static_specificity(),
                    reasons=[f"{size} parameters is not {band.predicate()}"],
                )
            reasons.append(f"{size} parameters is {band.predicate()}")

        if not reasons:
            reasons.append("generic rule, no constraints")
        return MatchOutcome(
            matched=True,
            specificity=self._matched_specificity(alias_hint=alias_hint),
            reasons=reasons,
            alias_hint=alias_hint,
        )

    def _identity_rejection(self, subject: MatchSubject, *, hints_considered: bool) -> list[str]:
        """Explain why neither the declared identity nor a hint satisfied this rule."""
        described = subject.model_type or ", ".join(subject.architectures) or "<unknown>"
        reasons = [f"architecture/model type {described} is not listed"]
        if hints_considered:
            hints = ", ".join(subject.identity_hints) or "<none>"
            reasons.append(f"identity hints ({hints}) match no alias of this rule")
        elif self.alias_tokens and subject.identity_hints:
            hints = ", ".join(subject.identity_hints)
            reasons.append(
                f"identity hints ({hints}) were not consulted because the repository "
                f"declares an identity of its own"
            )
        return reasons

    def _matched_specificity(self, *, alias_hint: str | None) -> int:
        """Score a successful match, demoting one that only an alias rescued."""
        specificity = self.static_specificity()
        if alias_hint is None:
            return specificity
        return _alias_demoted(specificity)

    def resolvable_specificities(self) -> tuple[int, ...]:
        """Every specificity :meth:`evaluate` can actually score this block at.

        :meth:`static_specificity` reports only the best rung. A block that also
        carries ``aliases`` resolves at the demoted
        :data:`SPECIFICITY_ARCHITECTURE_ALIAS` rung whenever an alias rescued
        the match, so it competes with other rules on two rungs, not one.
        Registry validation compares every rung two rules share: a pair that
        ties only on the demoted rung would otherwise pass validation and then
        raise ``registry.ambiguous_rules`` at resolution time.
        """
        static = self.static_specificity()
        if not self.alias_tokens:
            return (static,)
        demoted = _alias_demoted(static)
        if demoted == static:
            return (static,)
        return (static, demoted)

    def overlaps(self, other: RuleMatch) -> bool:
        """True when both blocks could select the same subject.

        Used by registry validation to detect genuinely ambiguous rule pairs
        without having to enumerate every possible subject.
        """
        return (
            _tokens_overlap(_casefold_all(self.repo_ids), _casefold_all(other.repo_ids))
            and self._identity_overlaps(other)
            and _tokens_overlap(_casefold_all(self.tasks), _casefold_all(other.tasks))
            and _tokens_overlap(self.devices, other.devices)
            and _tokens_overlap(_casefold_all(self.host_tags), _casefold_all(other.host_tags))
            and _ranges_intersect(self.parameter_count, other.parameter_count)
        )

    def _identity_overlaps(self, other: RuleMatch) -> bool:
        """True when one subject could satisfy both blocks' identity criterion.

        A block that lists neither identities nor aliases constrains identity
        not at all and therefore overlaps everything. Otherwise the question is
        whether the two blocks name a token in common — comparing aliases with
        aliases (one hint could satisfy both) and each side's aliases with the
        other side's declared identities (the same name written as a hint on one
        side and as a declaration on the other). Two alias-bearing blocks naming
        *different* architectures are not overlapping: no subject carries two
        identities, and a subject carries at most one hint, so nothing could
        satisfy both. Reporting them would cap the registry at one alias rule
        per priority and make every further alias a hand-tuned priority.
        """
        mine_identity, mine_aliases = self.identity_tokens, self.alias_tokens
        their_identity, their_aliases = other.identity_tokens, other.alias_tokens
        if not (mine_identity or mine_aliases) or not (their_identity or their_aliases):
            return True
        pairs = (
            (mine_identity, their_identity),
            (mine_aliases, their_aliases),
            (mine_aliases, their_identity),
            (mine_identity, their_aliases),
        )
        return any(set(left) & set(right) for left, right in pairs)


def _alias_demoted(specificity: int) -> int:
    """Lower an architecture rung to the alias rung; other rungs are unchanged."""
    if specificity in (SPECIFICITY_ARCHITECTURE, SPECIFICITY_ARCHITECTURE_FAMILY):
        return SPECIFICITY_ARCHITECTURE_ALIAS
    return specificity


def _alias_reason(alias_hint: str) -> str:
    """Word an alias-only identity match the way ``registry explain`` prints it.

    The sentence names the alias and where the hint came from, so the audit
    trail never lets an alias match pass for a ``config.json`` identification.
    An alias only ever matches a subject that declares no identity, so the
    sentence can state that unconditionally.
    """
    return f"matched alias {alias_hint!r} from {IDENTITY_HINT_SOURCE} (no config.json)"


def _tokens_overlap(left: Sequence[str], right: Sequence[str]) -> bool:
    """True when two criterion lists could both be satisfied by one subject."""
    if not left or not right:
        return True
    return bool(set(left) & set(right))


class DeviceSupport(BaseModel):
    """What is known about running one architecture on one device class."""

    model_config = _STRICT

    status: DeviceStatus = DeviceStatus.UNKNOWN
    evidence: EvidenceLevel = EvidenceLevel.UNKNOWN
    constraints: dict[str, str] = Field(default_factory=dict)
    reason: str | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("constraints")
    @classmethod
    def _normalise_constraints(cls, value: dict[str, str]) -> dict[str, str]:
        return {_normalise_key(key): str(spec).strip() for key, spec in value.items()}


class ExportSpec(BaseModel):
    """Exporter requirements and pipeline hints for an architecture."""

    model_config = _STRICT

    transformers: str | None = None
    optimum_intel: str | None = None
    openvino: str | None = None
    nncf: str | None = None
    task: str | None = None
    pipeline: PipelineKind = PipelineKind.UNKNOWN
    trust_remote_code: bool = False
    native_implementation: bool = False
    """Transformers implements this architecture itself.

    A repository of such a model may still ship its own modelling code and an
    ``auto_map`` (Phi-4-mini does). With ``trust_remote_code`` off, Transformers
    ignores that code and uses its own implementation, so the export needs no
    remote code and the repository's Python is never run. Declaring this is a
    claim about Transformers, not about any repository, and it never *enables*
    remote code: a rule that sets both is contradictory and is rejected.
    """
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("extra_args")
    @classmethod
    def _no_reserved_options(cls, value: list[str]) -> list[str]:
        return _checked_export_args(value)

    @model_validator(mode="after")
    def _native_excludes_remote_code(self) -> ExportSpec:
        if self.native_implementation and self.trust_remote_code:
            raise ValueError("native_implementation and trust_remote_code contradict each other")
        return self

    def specifiers(self) -> dict[str, str]:
        """Return the non-empty version specifiers keyed by distribution name."""
        pairs = (
            ("transformers", self.transformers),
            ("optimum_intel", self.optimum_intel),
            ("openvino", self.openvino),
            ("nncf", self.nncf),
        )
        return {name: spec for name, spec in pairs if spec}


def export_needs_remote_code(spec: ExportSpec | None, *, repository_ships_code: bool) -> bool:
    """Whether exporting cannot proceed without executing repository code.

    *repository_ships_code* is what the repository declares (an ``auto_map``).
    That means remote code *unless* the applicable export spec declares that
    Transformers implements the architecture natively: then the repository's
    code is ignored with ``trust_remote_code`` off and never run. A spec that
    itself demands remote code always does. ``inspect`` and ``prepare`` both
    decide through this one function, so they cannot disagree.
    """
    if spec is not None and spec.trust_remote_code:
        return True
    if repository_ships_code:
        return not (spec is not None and spec.native_implementation)
    return False


class BaseRule(BaseModel):
    """Fields shared by every registry rule."""

    model_config = _STRICT

    #: Rule family, used in explanations and in :class:`Registry` indexes.
    kind: ClassVar[str] = "rule"

    id: str
    schema_version: int = SCHEMA_VERSION
    status: RuleStatus = RuleStatus.ACTIVE
    summary: str = ""
    sources: list[SourceRef] = Field(default_factory=list)
    priority: int = 100
    constraints: dict[str, str] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("rule id must not be empty")
        return cleaned

    @field_validator("constraints")
    @classmethod
    def _normalise_constraints(cls, value: dict[str, str]) -> dict[str, str]:
        return {_normalise_key(key): str(spec).strip() for key, spec in value.items()}

    @property
    def match_block(self) -> RuleMatch:
        """The rule's match criteria; generic when the rule has none."""
        return RuleMatch()

    @model_validator(mode="after")
    def _band_only_on_profiles(self) -> BaseRule:
        """A size band is a statement about quantization, so only profiles may carry one.

        Every rule kind shares :class:`RuleMatch`, so without this an
        architecture, issue or chat rule would accept ``parameter_count`` and
        silently mean something no resolver implements for it.
        """
        if self.kind != "profile" and self.match_block.parameter_count is not None:
            raise ValueError(
                f"{self.kind} rule {self.id}: parameter_count is only honoured in a "
                "conversion profile's applies_to"
            )
        return self

    def source_mappings(self) -> list[dict[str, str]]:
        """Provenance entries as flat string mappings."""
        return [source.as_mapping() for source in self.sources]


class ArchitectureRule(BaseRule):
    """Device support, exporter requirements and defaults for an architecture."""

    kind: ClassVar[str] = "architecture"

    match: RuleMatch = Field(default_factory=RuleMatch)
    export: ExportSpec = Field(default_factory=ExportSpec)
    devices: dict[str, DeviceSupport] = Field(default_factory=dict)
    recommended_device: str | None = None
    profiles: dict[str, str] = Field(default_factory=dict)
    chat_profile: str | None = None
    known_limitations: list[str] = Field(default_factory=list)

    @field_validator("devices")
    @classmethod
    def _normalise_device_keys(cls, value: dict[str, DeviceSupport]) -> dict[str, DeviceSupport]:
        return {normalise_device(key): support for key, support in value.items()}

    @field_validator("profiles")
    @classmethod
    def _normalise_profile_keys(cls, value: dict[str, str]) -> dict[str, str]:
        return {normalise_device(key): profile_id.strip() for key, profile_id in value.items()}

    @field_validator("recommended_device")
    @classmethod
    def _normalise_recommended(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalise_device(value) or None

    @property
    def match_block(self) -> RuleMatch:
        """The rule's match criteria."""
        return self.match

    def device_support(self, device: str) -> DeviceSupport | None:
        """Return the declared support entry for ``device``, if any."""
        return self.devices.get(normalise_device(device))

    def profile_id_for(self, device: str) -> str | None:
        """Return the preferred profile id for ``device``, if the rule names one."""
        return self.profiles.get(normalise_device(device))


class ProfileRule(BaseRule):
    """A conversion profile: quantization plus runtime properties for a device."""

    kind: ClassVar[str] = "profile"

    device: str
    format: ModelFormat = ModelFormat.OPENVINO_IR
    quantization: QuantizationSettings = Field(default_factory=QuantizationSettings)
    runtime: dict[str, str] = Field(default_factory=dict)
    export_extra_args: list[str] = Field(default_factory=list)
    applies_to: RuleMatch = Field(default_factory=RuleMatch)
    rationale: str | None = None
    evidence: EvidenceLevel = EvidenceLevel.UNKNOWN
    """How strong the evidence is that this profile is the right automatic choice.

    Required on a profile with a size band, because the band is a claim that
    models in it are better served by this quantization than by the default.
    """

    @field_validator("device")
    @classmethod
    def _normalise_device_name(cls, value: str) -> str:
        name = normalise_device(value)
        if not name:
            raise ValueError("profile device must not be empty")
        return name

    @model_validator(mode="after")
    def _band_needs_schema_and_evidence(self) -> ProfileRule:
        if (
            "evidence" in self.model_fields_set
            and self.schema_version < PARAMETER_BAND_SCHEMA_VERSION
        ):
            # A schema-1 reader rejects the unknown key, so a schema-1 file must not carry it.
            raise ValueError(
                f"profile {self.id}: evidence needs schema_version {PARAMETER_BAND_SCHEMA_VERSION}"
            )
        band = self.applies_to.parameter_count
        if band is None:
            return self
        if self.schema_version < PARAMETER_BAND_SCHEMA_VERSION:
            raise ValueError(
                f"profile {self.id}: parameter_count needs schema_version "
                f"{PARAMETER_BAND_SCHEMA_VERSION}"
            )
        if self.evidence is EvidenceLevel.UNKNOWN:
            raise ValueError(f"profile {self.id}: a size-banded profile must state its evidence")
        return self

    @field_validator("export_extra_args")
    @classmethod
    def _no_reserved_options(cls, value: list[str]) -> list[str]:
        return _checked_export_args(value)

    @field_validator("runtime")
    @classmethod
    def _normalise_runtime(cls, value: dict[str, str]) -> dict[str, str]:
        """Runtime property names are upper case in OpenVINO GenAI."""
        return {key.strip().upper(): str(item) for key, item in value.items()}

    @property
    def match_block(self) -> RuleMatch:
        """The rule's match criteria."""
        return self.applies_to

    def weights_label(self) -> str:
        """The weight compression in a few words: ``int4 asym g128 ratio 0.8``."""
        settings = self.quantization
        parts = [settings.weight_format]
        if settings.symmetric is not None:
            parts.append("sym" if settings.symmetric else "asym")
        if settings.group_size is not None:
            parts.append("channel-wise" if settings.group_size == -1 else f"g{settings.group_size}")
        if settings.ratio is not None and settings.ratio < 1.0:
            parts.append(f"ratio {settings.ratio:g}")
        if settings.awq:
            parts.append("awq")
        if settings.scale_estimation:
            parts.append("scale estimation")
        return " ".join(parts)

    def to_conversion_profile(self) -> ConversionProfile:
        """Convert to the domain :class:`~yonga.domain.plans.ConversionProfile`."""
        return ConversionProfile(
            id=self.id,
            device=self.device,
            summary=self.summary,
            quantization=self.quantization.model_copy(deep=True),
            runtime_properties=dict(self.runtime),
            export_extra_args=list(self.export_extra_args),
            constraints=dict(self.constraints),
            priority=self.priority,
            rationale=self.rationale,
            sources=self.source_mappings(),
        )


class IssueMatch(RuleMatch):
    """Match criteria for a known failure signature."""

    stages: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("stages", "stage"),
    )
    exception_contains: list[str] = Field(default_factory=list)
    file_names: list[str] = Field(default_factory=list)
    contains: str | None = None
    case_sensitive: bool = False

    @field_validator("stages", mode="before")
    @classmethod
    def _accept_single_stage(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("stages")
    @classmethod
    def _normalise_stages(cls, value: list[str]) -> list[str]:
        return _casefold_all(value)

    def signature_hit(self, text: str) -> str | None:
        """Return the longest signature found in ``text``, or ``None``."""
        if not self.exception_contains:
            return None
        haystack = text if self.case_sensitive else text.casefold()
        hits = [
            needle
            for needle in self.exception_contains
            if (needle if self.case_sensitive else needle.casefold()) in haystack
        ]
        if not hits:
            return None
        return max(hits, key=len)

    def stage_matches(self, stage: str | None) -> bool:
        """True when this rule applies to ``stage`` (no ``stages`` means any)."""
        if not self.stages:
            return True
        if stage is None:
            return False
        return stage.strip().casefold() in self.stages


class IssueClassification(BaseModel):
    """The diagnostic an issue rule produces."""

    model_config = _STRICT

    code: str
    category: str
    environment_broken: bool = False
    retry_same_device: bool = False
    defer_to_pipeline: bool = False
    """The probe that produced this failure is not how the runtime executes the model.

    Set it for a failure a raw ``Core.compile_model`` hits on every model of a
    kind -- a dynamically shaped LLM on the NPU -- where the OpenVINO GenAI
    pipeline, which reshapes before compiling, is the real test. A deferred
    failure is recorded but does not decide the device; the pipeline smoke test
    does. It never makes a device look supported on its own: with no pipeline
    evidence the device stays failed.
    """

    @field_validator("code")
    @classmethod
    def _check_code(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("classification code must not be empty")
        return cleaned

    @field_validator("category")
    @classmethod
    def _check_category(cls, value: str) -> str:
        return value.strip().lower()


class IssueRecommendation(BaseModel):
    """What the user should do about a classified failure."""

    model_config = _STRICT

    message: str | None = None
    title: str | None = None
    fallback_devices: list[str] = Field(default_factory=list)
    remediation: list[str] = Field(default_factory=list)
    repair: str | None = None

    @field_validator("fallback_devices")
    @classmethod
    def _normalise_devices(cls, value: list[str]) -> list[str]:
        names: list[str] = []
        for raw in value:
            name = normalise_device(raw)
            if name and name not in names:
                names.append(name)
        return names


class IssueRule(BaseRule):
    """A known failure signature and its classification."""

    kind: ClassVar[str] = "issue"

    match: IssueMatch = Field(default_factory=IssueMatch)
    classification: IssueClassification
    recommendation: IssueRecommendation = Field(default_factory=IssueRecommendation)

    @property
    def match_block(self) -> RuleMatch:
        """The rule's match criteria."""
        return self.match


class RepairMatch(BaseModel):
    """File-level criteria selecting artifacts a repair applies to."""

    model_config = _STRICT_ALIASED

    files: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("files", "file_names"),
    )
    contains: str | None = None

    def file_candidates(self, names: Iterable[str]) -> list[str]:
        """Return the entries of ``names`` this repair would inspect."""
        wanted = {name.strip() for name in self.files if name.strip()}
        if not wanted:
            return list(names)
        return [name for name in names if name in wanted]


class RepairAction(BaseModel):
    """The built-in implementation a repair rule selects."""

    model_config = _STRICT

    implementation: str
    safety: RepairSafety = RepairSafety.CONFIRM

    @field_validator("implementation")
    @classmethod
    def _check_implementation(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("repair implementation must not be empty")
        return cleaned


class RepairValidation(BaseModel):
    """Post-conditions that must hold once a repair has been applied."""

    model_config = _STRICT

    requires_tokenizer_read: bool = False
    postconditions: list[str] = Field(default_factory=list)


class RepairRule(BaseRule):
    """A detectable artifact defect and the built-in repair that fixes it."""

    kind: ClassVar[str] = "repair"

    match: RepairMatch = Field(default_factory=RepairMatch)
    repair: RepairAction
    validation: RepairValidation = Field(default_factory=RepairValidation)

    @property
    def implementation(self) -> str:
        """Key of the built-in Python repair implementation."""
        return self.repair.implementation

    @property
    def safety(self) -> RepairSafety:
        """How much approval this repair needs."""
        return self.repair.safety


class DerivativeCandidate(BaseModel):
    """One alternative repository that can stand in for a requested model."""

    model_config = _STRICT_ALIASED

    repo_id: str
    relationship: SourceRelationship = SourceRelationship.CONVERTED
    trust: TrustPolicy = TrustPolicy.SUGGEST
    revision: str | None = None
    note: str | None = None
    evidence: list[SourceRef] = Field(
        default_factory=list,
        validation_alias=AliasChoices("evidence", "sources"),
    )

    @property
    def requires_confirmation(self) -> bool:
        """True when the user must approve before this source may be used."""
        return self.trust.requires_confirmation

    def source_evidence(self) -> list[SourceEvidence]:
        """Provenance as domain :class:`~yonga.domain.models.SourceEvidence` entries."""
        return [item.as_evidence() for item in self.evidence]


class DerivativeRule(BaseRule):
    """Declared alternative sources for one requested repository."""

    kind: ClassVar[str] = "derivative"

    requested_repo: str
    candidates: list[DerivativeCandidate] = Field(default_factory=list)

    @field_validator("requested_repo")
    @classmethod
    def _check_requested_repo(cls, value: str) -> str:
        cleaned = value.strip().strip("/")
        if not cleaned:
            raise ValueError("requested_repo must not be empty")
        return cleaned

    @property
    def match_block(self) -> RuleMatch:
        """Derivative rules are keyed by the requested repository id."""
        return RuleMatch(repo_ids=[self.requested_repo])

    def matches_repo(self, repo_id: str) -> bool:
        """True when this rule declares sources for ``repo_id``."""
        return repo_id.strip().casefold() == self.requested_repo.casefold()


class GGUFRule(BaseRule):
    """What the installed OpenVINO GenAI can read directly from a GGUF file.

    Two facts decide it, both read from the file's own header, never from its
    name: ``general.architecture`` must be one the GenAI GGUF model builder
    implements, and every tensor must be of a type its reader can convert. A
    ``Q4_K_M`` label says nothing reliable -- one such file is all ``Q4_K`` and
    ``Q6_K``, another carries ``Q5_0`` tensors the reader rejects.

    ``tensor_types`` and ``header_architectures`` list what was *verified*, not
    what the library might support: a type with a dequantiser that was never
    tested stays out until it has been.
    """

    kind: ClassVar[str] = "gguf"

    header_architectures: list[str] = Field(default_factory=list)
    """``general.architecture`` values the GenAI GGUF builder accepts."""
    tensor_types: list[str] = Field(default_factory=list)
    """GGUF tensor type names (``Q4_K``, ``Q8_0``, ...) the reader converts."""
    preferred_files: list[str] = Field(default_factory=list)
    """Quantization labels to pick from, in order, when the user names no file."""

    @field_validator("header_architectures")
    @classmethod
    def _casefold_architectures(cls, value: list[str]) -> list[str]:
        return [item.strip().casefold() for item in value if item.strip()]

    @field_validator("tensor_types", "preferred_files")
    @classmethod
    def _upper_types(cls, value: list[str]) -> list[str]:
        return [item.strip().upper() for item in value if item.strip()]

    def reads_architecture(self, name: str | None) -> bool:
        """Whether the GGUF builder accepts ``general.architecture`` ``name``."""
        if not name:
            return False
        return name.strip().casefold() in self.header_architectures


class ArtifactSpec(BaseModel):
    """What a specific repository already contains."""

    model_config = _STRICT

    format: ModelFormat = ModelFormat.UNKNOWN
    already_prepared: bool = False


class ModelRule(BaseRule):
    """Behaviour that genuinely differs from the model's architecture."""

    kind: ClassVar[str] = "model"

    match: RuleMatch = Field(default_factory=RuleMatch)
    artifact: ArtifactSpec = Field(default_factory=ArtifactSpec)
    devices: dict[str, DeviceSupport] = Field(default_factory=dict)
    profiles: dict[str, str] = Field(default_factory=dict)
    chat_profile: str | None = None
    export: ExportSpec | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("devices")
    @classmethod
    def _normalise_device_keys(cls, value: dict[str, DeviceSupport]) -> dict[str, DeviceSupport]:
        return {normalise_device(key): support for key, support in value.items()}

    @field_validator("profiles")
    @classmethod
    def _normalise_profile_keys(cls, value: dict[str, str]) -> dict[str, str]:
        return {normalise_device(key): profile_id.strip() for key, profile_id in value.items()}

    @property
    def match_block(self) -> RuleMatch:
        """The rule's match criteria."""
        return self.match

    def device_support(self, device: str) -> DeviceSupport | None:
        """Return the declared support entry for ``device``, if any."""
        return self.devices.get(normalise_device(device))

    def profile_id_for(self, device: str) -> str | None:
        """Return the preferred profile id for ``device``, if the rule names one."""
        return self.profiles.get(normalise_device(device))


class GenerationDefaults(BaseModel):
    """Sampling defaults suggested for a model family."""

    model_config = _STRICT

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_new_tokens: int | None = None
    repetition_penalty: float | None = None
    do_sample: bool | None = None

    def merged(self, other: GenerationDefaults) -> GenerationDefaults:
        """Return a copy where every field explicitly set on ``other`` wins."""
        data = self.model_dump()
        for name in other.model_fields_set:
            data[name] = getattr(other, name)
        return GenerationDefaults.model_validate(data)

    def as_mapping(self) -> dict[str, float | int | bool]:
        """Return only the fields that carry a value."""
        data: dict[str, float | int | bool] = {}
        for name, value in self.model_dump().items():
            if value is not None:
                data[name] = value
        return data


class ToolCallingRule(BaseModel):
    """How a model family writes tool calls, as measured; part of a chat profile.

    ``yonga serve`` accepts OpenAI ``tools`` only for a model whose chat profile
    carries one of these, because a family that has not been seen producing
    calls in a known shape would be handed tools it cannot use, and its
    answers would never be read back as calls. The entry selects one of the
    parsers compiled into yonga (:data:`KNOWN_TOOL_CALL_FORMATS`) and fills in
    its markers and field names; it never describes a parser.

    ``evidence`` must be ``tested`` or ``official`` and ``sources`` must say
    where the format was seen: tool calling is enabled from measurements, not
    from a template's promise. ``template_contains``, when set, is a literal
    the artifact's own chat template must contain before tools are enabled for
    it, which keeps a family rule from switching tools on for a repackaged
    model whose template does not ask for this format.
    """

    model_config = _STRICT

    format: str
    """A key of :data:`KNOWN_TOOL_CALL_FORMATS`."""
    open: str | None = None
    """The marker in front of a call (``tagged_json``) or of the call list (``json_list``)."""
    close: str | None = None
    """The marker after a call (``tagged_json``) or after the call list (``json_list``)."""
    name_field: str = "name"
    """The key of a call object that holds the function's name."""
    arguments_field: str = "arguments"
    """The key of a call object that holds its arguments: an object, or JSON text of one."""
    template_contains: str | None = None
    evidence: EvidenceLevel
    sources: list[SourceRef] = Field(min_length=1)

    @field_validator("format", "name_field", "arguments_field")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @field_validator("open", "close", "template_contains")
    @classmethod
    def _empty_is_unset(cls, value: str | None) -> str | None:
        """Markers are literals, kept as written; only an empty one means "none"."""
        return value or None

    @model_validator(mode="after")
    def _measured_and_complete(self) -> ToolCallingRule:
        if self.evidence not in (EvidenceLevel.TESTED, EvidenceLevel.OFFICIAL):
            raise ValueError(
                "tool_calling needs evidence tested or official: tools are enabled only "
                "for a measured format"
            )
        if self.format == "tagged_json" and not (self.open and self.close):
            raise ValueError("tool_calling format tagged_json needs both open and close")
        return self

    def describe(self) -> str:
        """One line for ``registry explain``: the format, its markers and the evidence."""
        markers = " and ".join(marker for marker in (self.open, self.close) if marker)
        parts = [self.format]
        if markers:
            parts.append(f"({markers})")
        parts.append(f"[{self.evidence.value}]")
        if self.template_contains:
            parts.append(f"when the chat template contains {self.template_contains!r}")
        return " ".join(parts)


#: Thinking variables tried against an artifact's own chat template when a chat
#: profile names none of its own. One entry, because ``enable_thinking`` is the
#: spelling the Jinja templates in the wild actually use; a family that spells it
#: differently says so in its registry rule instead of being added here.
DEFAULT_THINKING_VARIABLE_CANDIDATES: tuple[str, ...] = ("enable_thinking",)


class ChatProfileRule(BaseRule):
    """Chat-template and generation policy for a model family.

    Thinking mode is requested through a chat-template variable, and this rule
    describes *how sure* the registry is about which variable that is:

    ``supports_thinking_flag: true``
        The family is known to honour :attr:`thinking_template_variable`; it is
        applied without asking the artifact. The validator makes the variable
        mandatory in this case, because a declared flag with no name drives
        nothing.
    ``supports_thinking_flag: false``
        An authoritative veto. No thinking variable is ever sent for this
        family, whatever its chat template happens to mention.
    *unset* (the default)
        The registry has no opinion, so the decision is made from evidence: the
        runtime applies the first of :attr:`thinking_variable_candidates` that
        the artifact's **own** chat template actually references. That is what
        makes a model nobody has written a rule for behave correctly, and it is
        why enumerating architectures here is not required.

    The source repository's template is never rewritten; the variable travels as
    chat-history extra context (``SPEC.md`` section 18.4, ``MODEL_PIPELINE.md``
    section 13).
    """

    kind: ClassVar[str] = "chat"

    match: RuleMatch = Field(default_factory=RuleMatch)
    supports_thinking_flag: bool | None = None
    thinking_template_variable: str | None = None
    thinking_variable_candidates: list[str] = Field(
        default_factory=lambda: [*DEFAULT_THINKING_VARIABLE_CANDIDATES]
    )
    """Variable names to look for in the artifact's chat template, in order.

    An empty list is a second way to switch detection off, for a family whose
    template mentions a thinking variable it does not honour.
    """

    default_thinking: bool = False
    reasoning_wrappers: list[tuple[str, str]] = Field(default_factory=list)
    defaults: GenerationDefaults = Field(default_factory=GenerationDefaults)
    thinking_defaults: GenerationDefaults = Field(default_factory=GenerationDefaults)
    stop_strings: list[str] = Field(default_factory=list)
    """Raw-text end markers for the family, applied natively by OpenVINO GenAI.

    Model-level data, such as an end-of-turn marker a template writes as text
    rather than as a special token. GenAI stops at the first one and leaves it
    out of the answer. It matches the raw output, inside a reasoning block too,
    which is why a client's ``stop`` sequences are matched separately, on the
    visible answer.
    """

    tool_calling: ToolCallingRule | None = None
    """How the family writes tool calls, when that has been measured (schema 3).

    Without it ``yonga serve`` refuses a request that offers ``tools``.
    """

    @field_validator("stop_strings", mode="after")
    @classmethod
    def _clean_stop_strings(cls, value: list[str]) -> list[str]:
        """Drop empty and duplicate stop strings, preserving the declared order.

        Whitespace is kept: ``"\\n\\n"`` is a meaningful marker. An empty string
        would match everywhere, and GenAI's own validation does not refuse it.
        """
        cleaned: list[str] = []
        for text in value:
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned

    @field_validator("thinking_variable_candidates", mode="after")
    @classmethod
    def _clean_candidates(cls, value: list[str]) -> list[str]:
        """Drop blank and duplicate candidate names, preserving the declared order."""
        cleaned: list[str] = []
        for name in value:
            token = name.strip()
            if token and token not in cleaned:
                cleaned.append(token)
        return cleaned

    @model_validator(mode="after")
    def _check_thinking_variable(self) -> ChatProfileRule:
        """A declared thinking flag is useless without the variable that drives it."""
        if self.supports_thinking_flag and not self.thinking_template_variable:
            raise ValueError("supports_thinking_flag requires thinking_template_variable")
        return self

    @model_validator(mode="after")
    def _tool_calling_needs_schema_three(self) -> ChatProfileRule:
        """An older reader rejects the unknown key, so an older rule must not carry it."""
        if self.tool_calling is not None and self.schema_version < TOOL_CALLING_SCHEMA_VERSION:
            raise ValueError(
                f"chat profile {self.id}: tool_calling needs schema_version "
                f"{TOOL_CALLING_SCHEMA_VERSION}"
            )
        return self

    @property
    def match_block(self) -> RuleMatch:
        """The rule's match criteria."""
        return self.match

    @property
    def thinking_candidates(self) -> tuple[str, ...]:
        """Variables to try against an artifact template, most preferred first.

        A named :attr:`thinking_template_variable` is the only candidate, because
        naming one is the registry saying which variable this family uses.
        """
        named = (self.thinking_template_variable or "").strip()
        if named:
            return (named,)
        return tuple(self.thinking_variable_candidates)

    def defaults_for(self, *, thinking: bool) -> GenerationDefaults:
        """Return the generation defaults for thinking or non-thinking mode."""
        if thinking:
            return self.defaults.merged(self.thinking_defaults)
        return self.defaults.model_copy(deep=True)

    def template_context(self, *, thinking: bool | None = None) -> dict[str, bool]:
        """Chat-template variables this rule declares *without* consulting an artifact.

        Only a rule that declares ``supports_thinking_flag: true`` produces one
        here. Detection from a real chat template needs the artifact and lives in
        ``yonga.adapters.runtime_genai``.
        """
        if not self.supports_thinking_flag or self.thinking_template_variable is None:
            return {}
        enabled = self.default_thinking if thinking is None else thinking
        return {self.thinking_template_variable: enabled}


#: Any concrete registry rule.
AnyRule = (
    ArchitectureRule
    | ProfileRule
    | IssueRule
    | RepairRule
    | DerivativeRule
    | ModelRule
    | ChatProfileRule
    | GGUFRule
)


#: Every section of a registry document that holds a list of rules.
_RULE_SECTIONS: tuple[str, ...] = (
    "architectures",
    "profiles",
    "issues",
    "repairs",
    "derivatives",
    "models",
    "chat",
    "chat_profiles",
    "gguf",
)


class RegistryDocument(BaseModel):
    """One parsed registry YAML file, before merging."""

    model_config = _STRICT_ALIASED

    schema_version: int = SCHEMA_VERSION
    origin: str = "<memory>"
    architectures: list[ArchitectureRule] = Field(default_factory=list)
    profiles: list[ProfileRule] = Field(default_factory=list)
    issues: list[IssueRule] = Field(default_factory=list)
    repairs: list[RepairRule] = Field(default_factory=list)
    derivatives: list[DerivativeRule] = Field(default_factory=list)
    models: list[ModelRule] = Field(default_factory=list)
    chat: list[ChatProfileRule] = Field(
        default_factory=list,
        validation_alias=AliasChoices("chat", "chat_profiles"),
    )
    gguf: list[GGUFRule] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _rules_inherit_the_document_version(cls, data: object) -> object:
        """A rule that does not state its schema_version has the document's.

        Without this a rule would default to the *build's* version, so a band
        in a file declaring ``schema_version: 1`` would pass the band's own
        version check, and a bundle built from that file would carry no
        ``min_tool_version`` -- an older yonga would install data it cannot
        parse. A rule declaring a newer version than its document is refused
        for the same reason.
        """
        if not isinstance(data, dict):
            return data
        declared = data.get("schema_version", SCHEMA_VERSION)
        if not isinstance(declared, int) or isinstance(declared, bool):
            return data
        updated = dict(data)
        for section in _RULE_SECTIONS:
            items = updated.get(section)
            if not isinstance(items, list):
                continue
            rules: list[object] = []
            for item in items:
                if isinstance(item, dict):
                    version = item.get("schema_version", declared)
                    if isinstance(version, int) and version > declared:
                        raise ValueError(
                            f"rule {item.get('id', '<no id>')!r} declares schema_version "
                            f"{version} but its document declares {declared}"
                        )
                    item = {**item, "schema_version": version}
                rules.append(item)
            updated[section] = rules
        return updated

    def rules(self) -> list[AnyRule]:
        """Every rule in this document, in section order."""
        collected: list[AnyRule] = []
        collected.extend(self.architectures)
        collected.extend(self.profiles)
        collected.extend(self.issues)
        collected.extend(self.repairs)
        collected.extend(self.derivatives)
        collected.extend(self.models)
        collected.extend(self.chat)
        collected.extend(self.gguf)
        return collected


class Registry(BaseModel):
    """The merged, indexed registry used by the resolver."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    architectures: list[ArchitectureRule] = Field(default_factory=list)
    profiles: list[ProfileRule] = Field(default_factory=list)
    issues: list[IssueRule] = Field(default_factory=list)
    repairs: list[RepairRule] = Field(default_factory=list)
    derivatives: list[DerivativeRule] = Field(default_factory=list)
    models: list[ModelRule] = Field(default_factory=list)
    chat: list[ChatProfileRule] = Field(default_factory=list)
    gguf: list[GGUFRule] = Field(default_factory=list)
    #: Rule id -> human readable origin of the layer that supplied it.
    origins: dict[str, str] = Field(default_factory=dict)
    #: Human readable origins of every layer that was merged, in order.
    layers: list[str] = Field(default_factory=list)
    #: Layers that are present but were left out, each saying why: an installed
    #: remote bundle that no longer verifies (``registry.remote.bad_signature:
    #: ...``). Loading carries on without them; every surface that reports on
    #: the registry says so instead of looking clean.
    ignored: list[str] = Field(default_factory=list)

    _index: dict[str, AnyRule] = PrivateAttr(default_factory=dict)

    def model_post_init(self, _context: Any, /) -> None:
        """Build the by-id index once the model is constructed."""
        index: dict[str, AnyRule] = {}
        for rule in self.all_rules():
            index[rule.id] = rule
        self._index = index

    def all_rules(self) -> list[AnyRule]:
        """Every rule of every kind, grouped by kind in a stable order."""
        collected: list[AnyRule] = []
        collected.extend(self.architectures)
        collected.extend(self.profiles)
        collected.extend(self.issues)
        collected.extend(self.repairs)
        collected.extend(self.derivatives)
        collected.extend(self.models)
        collected.extend(self.chat)
        collected.extend(self.gguf)
        return collected

    @property
    def rule_ids(self) -> list[str]:
        """Ids of every rule, in registry order."""
        return [rule.id for rule in self.all_rules()]

    def rule(self, rule_id: str) -> AnyRule | None:
        """Look up any rule by id."""
        return self._index.get(rule_id)

    def origin_of(self, rule_id: str) -> str | None:
        """Return the layer a rule came from, for explanations and debugging."""
        return self.origins.get(rule_id)

    def architecture(self, rule_id: str) -> ArchitectureRule | None:
        """Look up an architecture rule by id."""
        found = self._index.get(rule_id)
        return found if isinstance(found, ArchitectureRule) else None

    def profile(self, rule_id: str) -> ProfileRule | None:
        """Look up a profile rule by id."""
        found = self._index.get(rule_id)
        return found if isinstance(found, ProfileRule) else None

    def issue(self, rule_id: str) -> IssueRule | None:
        """Look up an issue rule by id."""
        found = self._index.get(rule_id)
        return found if isinstance(found, IssueRule) else None

    def repair(self, rule_id: str) -> RepairRule | None:
        """Look up a repair rule by id."""
        found = self._index.get(rule_id)
        return found if isinstance(found, RepairRule) else None

    def derivative(self, rule_id: str) -> DerivativeRule | None:
        """Look up a derivative rule by id."""
        found = self._index.get(rule_id)
        return found if isinstance(found, DerivativeRule) else None

    def model_rule(self, rule_id: str) -> ModelRule | None:
        """Look up an exact-model rule by id."""
        found = self._index.get(rule_id)
        return found if isinstance(found, ModelRule) else None

    def chat_profile(self, rule_id: str) -> ChatProfileRule | None:
        """Look up a chat profile rule by id."""
        found = self._index.get(rule_id)
        return found if isinstance(found, ChatProfileRule) else None

    def profiles_for_device(self, device: str) -> list[ProfileRule]:
        """Return every profile rule targeting ``device``."""
        wanted = normalise_device(device)
        return [rule for rule in self.profiles if rule.device == wanted]

    def derivatives_for(self, repo_id: str) -> list[DerivativeRule]:
        """Return every derivative rule declared for ``repo_id``."""
        return [rule for rule in self.derivatives if rule.matches_repo(repo_id)]

    def counts(self) -> dict[str, int]:
        """Number of rules per kind, for ``yonga registry validate`` output."""
        return {
            "architectures": len(self.architectures),
            "profiles": len(self.profiles),
            "issues": len(self.issues),
            "repairs": len(self.repairs),
            "derivatives": len(self.derivatives),
            "models": len(self.models),
            "chat": len(self.chat),
            "gguf": len(self.gguf),
        }


#: Why a source that keeps its own quantization has no size for matching.
SIZE_NOT_APPLICABLE: str = "the source keeps its own quantization"


def subject_from_metadata(
    metadata: object,
    *,
    device: str | None = None,
    host_tags: Sequence[str] = (),
    size_applies: bool = True,
) -> MatchSubject:
    """Build a :class:`MatchSubject` from a ``ModelMetadata``-like object.

    Accepts anything exposing ``identity``/``model_type``/``architectures``/
    ``task``/``gguf_metadata`` attributes, and tolerates ``None`` so
    classification still works when no metadata was gathered.

    :attr:`MatchSubject.identity_hints` is filled from
    ``gguf_metadata["architecture"]`` — the identifier stored inside the weight
    file — and from nothing else. ``tags`` are read by no branch here on
    purpose: see the module docstring.

    The parameter count comes from ``metadata.size_for_matching()`` when the
    object has one. ``size_applies=False`` leaves it unknown on purpose, for a
    GGUF import or an OpenVINO copy, whose quantization no profile decides.
    """
    repo_id: str | None = None
    model_type: str | None = None
    architectures: tuple[str, ...] = ()
    task: str | None = None
    identity_hints: tuple[str, ...] = ()
    if metadata is not None:
        identity = getattr(metadata, "identity", None)
        raw_repo = getattr(identity, "repo_id", None)
        if isinstance(raw_repo, str):
            repo_id = raw_repo
        raw_type = getattr(metadata, "model_type", None)
        if isinstance(raw_type, str):
            model_type = raw_type
        raw_architectures = getattr(metadata, "architectures", None)
        if isinstance(raw_architectures, Sequence) and not isinstance(raw_architectures, str):
            architectures = tuple(item for item in raw_architectures if isinstance(item, str))
        raw_task = getattr(metadata, "task", None)
        if isinstance(raw_task, str):
            task = raw_task
        raw_gguf = getattr(metadata, "gguf_metadata", None)
        if isinstance(raw_gguf, Mapping):
            raw_hint = raw_gguf.get("architecture")
            if isinstance(raw_hint, str) and raw_hint.strip():
                identity_hints = (raw_hint.strip(),)
    parameter_count: int | None = None
    count_basis: str | None = None
    if not size_applies:
        count_basis = SIZE_NOT_APPLICABLE
    elif metadata is not None:
        sizer = getattr(metadata, "size_for_matching", None)
        if callable(sizer):
            found = sizer()
            if isinstance(found, tuple) and len(found) == 2:
                raw_count, raw_basis = found
                if isinstance(raw_count, int) and not isinstance(raw_count, bool):
                    parameter_count = raw_count if raw_count > 0 else None
                if isinstance(raw_basis, str):
                    count_basis = raw_basis
    return MatchSubject(
        repo_id=repo_id,
        model_type=model_type,
        architectures=architectures,
        task=task,
        device=device,
        host_tags=tuple(host_tags),
        identity_hints=identity_hints,
        parameter_count=parameter_count,
        parameter_count_basis=count_basis,
    )


def constraint_problems(constraints: Mapping[str, str]) -> list[str]:
    """Return human-readable problems with a ``constraints`` block."""
    problems: list[str] = []
    for key, specifier in constraints.items():
        if key not in KNOWN_CONSTRAINT_KEYS:
            problems.append(f"unknown constraint key {key!r}")
        try:
            SpecifierSet(specifier.strip())
        except InvalidSpecifier:
            problems.append(f"unparsable version specifier {specifier!r} for {key!r}")
    return problems
