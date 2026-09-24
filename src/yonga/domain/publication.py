"""Publication: what may be redistributed, and what a publication will contain.

``SPEC.md`` section 21 puts the requirements before the plumbing: the user
names the destination, the source licence permits redistribution, and the
model card tells the truth about lineage, quantization, repairs and devices.
This module holds the licence policy and the plan the CLI shows before a single
byte leaves the machine.

The licence policy is deliberately conservative and deliberately small. yonga
cannot read a licence; it can only recognise identifiers the Hugging Face Hub
uses for licences whose redistribution terms are well known. Everything else
-- no licence, ``other``, a custom name -- is :attr:`LicenseClass.UNKNOWN`, and
publishing it requires the user to state that they have read it
(``--license-reviewed``). ``--yes`` never makes that statement for them.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DeviceEvidence",
    "LicenseAssessment",
    "LicenseClass",
    "PublishFile",
    "PublishPlan",
    "PublisherIdentity",
    "classify_license",
]


class LicenseClass(str, Enum):
    """How a licence treats redistribution of a quantized derivative."""

    PERMISSIVE = "permissive"
    """Redistribution of derivatives allowed with attribution at most."""
    CONDITIONAL = "conditional"
    """Allowed, but the licence's own terms must travel with the weights."""
    RESTRICTED = "restricted"
    """Derivatives may not be redistributed (``-nd`` licences)."""
    UNKNOWN = "unknown"
    """No licence, or one yonga does not recognise: a human has to read it."""


_PERMISSIVE: frozenset[str] = frozenset(
    {
        "apache-2.0",
        "mit",
        "bsd-2-clause",
        "bsd-3-clause",
        "bsd-3-clause-clear",
        "isc",
        "zlib",
        "bsl-1.0",
        "unlicense",
        "cc0-1.0",
        "cc-by-4.0",
        "cc-by-sa-4.0",
    }
)

_CONDITIONAL: frozenset[str] = frozenset(
    {
        "llama2",
        "llama3",
        "llama3.1",
        "llama3.2",
        "llama3.3",
        "llama4",
        "gemma",
        "openrail",
        "openrail++",
        "bigscience-openrail-m",
        "creativeml-openrail-m",
        "bigscience-bloom-rail-1.0",
        "cc-by-nc-4.0",
        "cc-by-nc-sa-4.0",
    }
)

_RESTRICTED: frozenset[str] = frozenset({"cc-by-nd-4.0", "cc-by-nc-nd-4.0"})


def classify_license(identifier: str | None) -> LicenseClass:
    """Classify a Hub licence identifier; anything unrecognised is ``UNKNOWN``."""
    name = (identifier or "").strip().lower()
    if name in _PERMISSIVE:
        return LicenseClass.PERMISSIVE
    if name in _CONDITIONAL:
        return LicenseClass.CONDITIONAL
    if name in _RESTRICTED:
        return LicenseClass.RESTRICTED
    return LicenseClass.UNKNOWN


class LicenseAssessment(BaseModel):
    """The licence of one repository the artifact descends from."""

    model_config = ConfigDict(frozen=True)

    repo_id: str
    role: str
    """``requested`` (the model asked for) or ``source`` (the weights converted)."""
    license: str | None
    classification: LicenseClass


class DeviceEvidence(BaseModel):
    """What the artifact's probes say about one device."""

    model_config = ConfigDict(frozen=True)

    device: str
    status: str
    """``ok``, ``failed``, or ``not_tested``."""
    detail: str = ""
    attempted_at: str | None = None


class PublishFile(BaseModel):
    """One file of the upload: where it is, where it goes, and what it hashes to."""

    model_config = ConfigDict(frozen=True)

    path_in_repo: str
    local_path: Path | None = None
    """``None`` for generated files, whose bytes are in :attr:`PublishPlan.generated`."""
    size: int
    sha256: str


class PublishPlan(BaseModel):
    """Everything a publication will do, decided before anything is uploaded."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    artifact_dir: Path
    repo_id: str
    private: bool
    repo_exists: bool
    licenses: list[LicenseAssessment]
    license_reviewed: bool = False
    devices: list[DeviceEvidence]
    repairs: list[str] = Field(default_factory=list)
    files: list[PublishFile]
    generated: dict[str, str] = Field(default_factory=dict)
    """Path in the repository to text yonga wrote: the model card and the manifest."""
    warnings: list[str] = Field(default_factory=list)

    @property
    def upload_bytes(self) -> int:
        """Total size of everything uploaded."""
        return sum(item.size for item in self.files)

    @property
    def url(self) -> str:
        """Where the repository lives on the Hub."""
        return f"https://huggingface.co/{self.repo_id}"

    @property
    def supported_devices(self) -> list[str]:
        """Devices whose probe passed: the only ones the card may claim."""
        return [evidence.device for evidence in self.devices if evidence.status == "ok"]


class PublisherIdentity(BaseModel):
    """Who the stored credentials belong to, and whether they can write."""

    model_config = ConfigDict(frozen=True)

    name: str
    organizations: list[str] = Field(default_factory=list)
    can_write: bool | None = None
    """``False`` for a read-only token; ``None`` when the Hub does not say (fine-grained)."""

    def may_write_to(self, namespace: str) -> bool:
        """Whether ``namespace`` is this account or one of its organisations."""
        return namespace == self.name or namespace in self.organizations
