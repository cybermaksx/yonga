"""The OpenVINO compile cache as yonga sees it: entries, owners and their status.

Every pipeline yonga builds gets ``CACHE_DIR=<cache>/compile/<key>``, where the
key is a one-way digest of (artifact identity, device, OpenVINO version, runtime
properties); see :func:`yonga.adapters.openvino_probe.compile_cache_key`. A
digest cannot be read back, so yonga writes a small ownership record,
:data:`COMPILE_CACHE_ENTRY_FILENAME`, into each entry it creates or uses. The
record is what ``yonga cache list`` and ``yonga cache prune`` read to tell an
entry that still serves an artifact from one whose artifact is gone or whose
blobs the installed OpenVINO can no longer load.

The record's name starts with a dot and ends in ``.json``. OpenVINO only ever
opens the files it names itself (``<digest>.blob``, ``<digest>.cl_cache``) inside
a cache directory and never enumerates it, so the extra file is never mistaken
for a blob.

This module is pure: models and predicates, no filesystem access.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "COMPILE_CACHE_ENTRY_FILENAME",
    "COMPILE_CACHE_ENTRY_SCHEMA",
    "COMPILE_CACHE_KEY_LENGTH",
    "RECLAIMABLE_STATUSES",
    "CacheAttribution",
    "CacheEntryStatus",
    "CompileCacheEntry",
    "CompileCacheInventory",
    "CompileCachePrunePlan",
    "CompileCachePruneReport",
    "CompileCacheRecord",
    "PruneFailure",
    "is_cache_key",
]

#: Name of the ownership record yonga keeps inside each compile cache entry.
COMPILE_CACHE_ENTRY_FILENAME = ".yonga-entry.json"

#: Schema version of :class:`CompileCacheRecord`.
COMPILE_CACHE_ENTRY_SCHEMA = 1

#: Length of a compile cache key: 16 lowercase hexadecimal characters.
COMPILE_CACHE_KEY_LENGTH = 16

_KEY_PATTERN = re.compile(rf"[0-9a-f]{{{COMPILE_CACHE_KEY_LENGTH}}}")


def is_cache_key(name: str) -> bool:
    """Whether ``name`` has the exact shape of a compile cache key.

    Only directories named like this are ever treated as cache entries; anything
    else under the cache root is reported and left alone.
    """
    return _KEY_PATTERN.fullmatch(name) is not None


class CacheEntryStatus(str, Enum):
    """What an entry is worth keeping for."""

    IN_USE = "in-use"
    """Its artifact exists and the installed OpenVINO would load its blobs."""

    ORPHANED = "orphaned"
    """The artifact it was built for is gone."""

    STALE_RUNTIME = "stale-runtime"
    """Built by an OpenVINO release other than the installed one, so never reused."""

    UNATTRIBUTED = "unattributed"
    """No ownership record, and no known artifact produces its key."""


#: The statuses ``yonga cache prune`` removes without being asked for more.
RECLAIMABLE_STATUSES: frozenset[CacheEntryStatus] = frozenset(
    {CacheEntryStatus.ORPHANED, CacheEntryStatus.STALE_RUNTIME}
)


class CacheAttribution(str, Enum):
    """How an entry's owner was established."""

    RECORD = "record"
    """From the ownership record yonga wrote into the entry."""

    RECOMPUTED = "recomputed"
    """By recomputing the keys of known artifacts until one matched the name."""

    NONE = "none"
    """Not at all."""


class CompileCacheRecord(BaseModel):
    """The ownership record stored as :data:`COMPILE_CACHE_ENTRY_FILENAME`.

    ``fingerprint`` is the artifact half of the key exactly as the runtime
    computed it: the manifest's ``artifact_id`` for a prepared artifact, or a
    digest of the file listing for a directory without a manifest (a staging
    directory during ``prepare``'s smoke test). ``artifact_id`` is set only in
    the first case.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: int = COMPILE_CACHE_ENTRY_SCHEMA
    key: str
    fingerprint: str
    artifact_id: str | None = None
    artifact_path: str | None = None
    repo_id: str | None = None
    profile: str | None = None
    device: str
    openvino_version: str | None = None
    properties: dict[str, str] = Field(default_factory=dict)
    properties_fingerprint: str = ""
    created_at: datetime
    last_used_at: datetime


class CompileCacheEntry(BaseModel):
    """One directory under the compile cache root, measured and classified."""

    model_config = ConfigDict(extra="ignore")

    key: str
    path: Path
    size_bytes: int = 0
    file_count: int = 0
    last_used_at: datetime | None = None
    device: str | None = None
    openvino_version: str | None = None
    artifact_id: str | None = None
    artifact_path: Path | None = None
    repo_id: str | None = None
    profile: str | None = None
    owner_in_store: bool = False
    owner_exists: bool = False
    attribution: CacheAttribution = CacheAttribution.NONE
    status: CacheEntryStatus = CacheEntryStatus.UNATTRIBUTED
    reason: str = ""

    @property
    def reclaimable(self) -> bool:
        """Whether a plain ``yonga cache prune`` removes this entry."""
        return self.status in RECLAIMABLE_STATUSES


class CompileCacheInventory(BaseModel):
    """Everything under the compile cache root."""

    model_config = ConfigDict(extra="ignore")

    root: Path
    store_root: Path
    installed_openvino: str | None = None
    entries: list[CompileCacheEntry] = Field(default_factory=list)
    ignored: list[str] = Field(default_factory=list)
    """Names under the root that are not cache entries; never touched."""

    @property
    def total_bytes(self) -> int:
        """Bytes held by every entry."""
        return sum(entry.size_bytes for entry in self.entries)

    @property
    def reclaimable_bytes(self) -> int:
        """Bytes held by orphaned and stale-runtime entries."""
        return sum(entry.size_bytes for entry in self.entries if entry.reclaimable)

    def counts(self) -> dict[str, int]:
        """Entry count per status, every status present."""
        counts = {status.value: 0 for status in CacheEntryStatus}
        for entry in self.entries:
            counts[entry.status.value] += 1
        return counts

    def bytes_by_status(self) -> dict[str, int]:
        """Bytes per status, every status present."""
        totals = {status.value: 0 for status in CacheEntryStatus}
        for entry in self.entries:
            totals[entry.status.value] += entry.size_bytes
        return totals

    def to_json(self) -> dict[str, object]:
        """The ``yonga cache list --json`` payload (without the ``ok`` envelope)."""
        return {
            "compile_cache_dir": str(self.root),
            "artifacts_dir": str(self.store_root),
            "installed_openvino": self.installed_openvino,
            "count": len(self.entries),
            "total_bytes": self.total_bytes,
            "reclaimable_bytes": self.reclaimable_bytes,
            "by_status": self.counts(),
            "bytes_by_status": self.bytes_by_status(),
            "entries": [entry.model_dump(mode="json") for entry in self.entries],
            "ignored": list(self.ignored),
        }


class CompileCachePrunePlan(BaseModel):
    """The entries a prune would remove, and why each was chosen."""

    model_config = ConfigDict(extra="ignore")

    root: Path
    older_than_days: float | None = None
    unattributed: bool = False
    everything: bool = False
    installed_openvino: str | None = None
    entries: list[CompileCacheEntry] = Field(default_factory=list)
    reasons: dict[str, str] = Field(default_factory=dict)
    """Why each key was selected: a status value, ``older-than`` or ``all``."""

    @property
    def total_bytes(self) -> int:
        """Bytes the plan would free, measured before anything is deleted."""
        return sum(entry.size_bytes for entry in self.entries)

    def criteria(self) -> dict[str, object]:
        """The selection rules, for the ``--json`` document."""
        return {
            "orphaned": True,
            "stale_runtime": self.installed_openvino is not None,
            "older_than_days": self.older_than_days,
            "unattributed": self.unattributed,
            "all": self.everything,
        }


class PruneFailure(BaseModel):
    """An entry that could not be removed, and why."""

    model_config = ConfigDict(extra="ignore")

    key: str
    path: Path
    error: str


class CompileCachePruneReport(BaseModel):
    """What ``yonga cache prune`` removed."""

    model_config = ConfigDict(extra="ignore")

    dry_run: bool = False
    removed: list[str] = Field(default_factory=list)
    failed: list[PruneFailure] = Field(default_factory=list)
    planned_bytes: int = 0
    freed_bytes: int = 0
