"""Device compatibility statuses, evidence levels and registry resolution explanations.

This module holds the *policy output* of compatibility reasoning: which device a model
should run on, how strong the evidence behind that claim is, and which registry rules
were considered before the decision was made. It holds no host facts (see
``yonga.domain.hardware``) and performs no I/O.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from yonga.domain.hardware import normalise_device

#: Placeholder used when the user did not request a specific device.
AUTO_DEVICE = "AUTO"

_STRICT = ConfigDict(extra="forbid")


class DeviceStatus(str, Enum):
    """How well a model/device pair is known to work."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    EXPERIMENTAL = "experimental"
    UNKNOWN = "unknown"

    @property
    def is_usable(self) -> bool:
        """True when the pair may be attempted (supported, experimental or unknown)."""
        return self is not DeviceStatus.UNSUPPORTED


class EvidenceLevel(str, Enum):
    """Strength of the knowledge backing a compatibility claim."""

    OFFICIAL = "official"
    TESTED = "tested"
    REPORTED = "reported"
    INFERRED = "inferred"
    UNKNOWN = "unknown"

    @property
    def rank(self) -> int:
        """Numeric strength of this evidence level; higher is stronger."""
        return _EVIDENCE_RANKS[self]


_EVIDENCE_RANKS: dict[EvidenceLevel, int] = {
    EvidenceLevel.OFFICIAL: 4,
    EvidenceLevel.TESTED: 3,
    EvidenceLevel.REPORTED: 2,
    EvidenceLevel.INFERRED: 1,
    EvidenceLevel.UNKNOWN: 0,
}


def strongest_evidence(levels: Iterable[EvidenceLevel]) -> EvidenceLevel:
    """Return the strongest level in ``levels``, or ``UNKNOWN`` when empty."""
    best = EvidenceLevel.UNKNOWN
    for level in levels:
        if level.rank > best.rank:
            best = level
    return best


def _normalise_requested_device(value: str) -> str:
    """Normalise a user-requested device, keeping the ``AUTO`` placeholder intact."""
    raw = value.strip()
    if not raw or raw.upper() == AUTO_DEVICE:
        return AUTO_DEVICE
    return normalise_device(raw)


class DeviceCompatibility(BaseModel):
    """Compatibility of one model with one normalised OpenVINO device."""

    model_config = _STRICT

    device: str
    status: DeviceStatus
    evidence: EvidenceLevel
    reason: str | None = None
    rule_id: str | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("device")
    @classmethod
    def _normalise_device_name(cls, value: str) -> str:
        return normalise_device(value)


class RuleCandidate(BaseModel):
    """One registry rule considered while resolving a decision."""

    model_config = _STRICT

    rule_id: str
    kind: str
    specificity: int = 0
    priority: int = 100
    matched: bool = False
    reasons: list[str] = Field(default_factory=list)


class ProfileSelection(BaseModel):
    """Why a conversion profile was chosen, in one record the plan, explain and manifest share.

    :attr:`sentence` is rendered once, by the resolver, so the three places
    that show it cannot word the same decision differently.

    It is persisted in the artifact manifest, so it is read the way the rest of
    the manifest is: unknown keys are ignored and :attr:`basis` and
    :attr:`band_evidence` are plain strings. A newer yonga that adds a basis
    or a field must not make this build lose sight of the whole artifact.
    """

    model_config = ConfigDict(extra="ignore")

    basis: str
    """``explicit``, ``named_by_model_rule``, ``named_by_architecture_rule``,
    ``size_band`` or ``device_default`` in this build."""
    profile_id: str
    rule_id: str | None = None
    """The model or architecture rule that named the profile, for the ``named_*`` bases."""
    parameter_count: int | None = None
    parameter_count_basis: str | None = None
    band: str | None = None
    """The winning profile's band, described (``below 2B``), when it has one."""
    band_evidence: str | None = None
    """The winning band's :class:`EvidenceLevel` value."""
    automatic_profile_id: str | None = None
    """For an explicit profile: what would have been chosen automatically, when different."""
    skipped: list[str] = Field(default_factory=list)
    """One sentence per size-banded profile that lost because of the model's size."""
    applies: bool = True
    """False when no profile decides the quantization (a GGUF import, an OpenVINO copy)."""
    sentence: str = ""


class ResolutionExplanation(BaseModel):
    """Audit trail of a registry resolution: candidates, winner and notes."""

    model_config = _STRICT

    subject: str
    candidates: list[RuleCandidate] = Field(default_factory=list)
    winner_id: str | None = None
    notes: list[str] = Field(default_factory=list)
    profile_selection: ProfileSelection | None = None

    @property
    def rejected(self) -> list[RuleCandidate]:
        """Candidates that did not match (or lost) the resolution."""
        return [candidate for candidate in self.candidates if not candidate.matched]

    @property
    def matched(self) -> list[RuleCandidate]:
        """Candidates that matched the subject."""
        return [candidate for candidate in self.candidates if candidate.matched]


class CompatibilityDecision(BaseModel):
    """The device decision for one preparation or run, with its evidence."""

    model_config = _STRICT

    requested_device: str = AUTO_DEVICE
    selected_device: str | None = None
    status: DeviceStatus = DeviceStatus.UNKNOWN
    evidence_level: EvidenceLevel = EvidenceLevel.UNKNOWN
    matched_rule_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    fallback_candidates: list[str] = Field(default_factory=list)
    per_device: list[DeviceCompatibility] = Field(default_factory=list)
    explanation: ResolutionExplanation | None = None
    explicit_request: bool = False

    @field_validator("requested_device")
    @classmethod
    def _normalise_requested(cls, value: str) -> str:
        return _normalise_requested_device(value)

    @field_validator("selected_device")
    @classmethod
    def _normalise_selected(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalise_device(value)

    @field_validator("fallback_candidates")
    @classmethod
    def _normalise_fallbacks(cls, value: list[str]) -> list[str]:
        seen: list[str] = []
        for name in value:
            normalised = normalise_device(name)
            if normalised not in seen:
                seen.append(normalised)
        return seen

    @property
    def is_auto(self) -> bool:
        """True when the user left device selection to the tool."""
        return self.requested_device == AUTO_DEVICE and not self.explicit_request

    @property
    def usable(self) -> bool:
        """True when a device was selected and its status does not forbid running."""
        return self.selected_device is not None and self.status.is_usable

    def compatibility_for(self, device: str) -> DeviceCompatibility | None:
        """Return the per-device entry for ``device`` (normalised), if present."""
        wanted = normalise_device(device)
        for entry in self.per_device:
            if entry.device == wanted:
                return entry
        return None
