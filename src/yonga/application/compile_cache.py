"""The ``yonga cache`` service: measure, attribute and prune the OpenVINO compile cache.

Every pipeline yonga builds compiles into ``<cache>/compile/<key>``, and before
this service existed nothing but ``yonga remove`` ever deleted an entry. Entries
for artifacts deleted by hand, for artifacts in another ``--data-dir``, for the
staging directory of a finished ``prepare``, and for blobs built by an OpenVINO
release that has since been upgraded all accumulated; the reference host held
276 GB in 203 entries.

Each entry is classified (:class:`~yonga.domain.compile_cache.CacheEntryStatus`):

* **in-use** -- its artifact exists and the installed OpenVINO would reuse it;
* **orphaned** -- the artifact it was built for is gone;
* **stale-runtime** -- an OpenVINO release other than the installed one built it,
  and since the runtime version is part of the key, it will never be reused;
* **unattributed** -- it carries no ownership record and no known artifact
  produces its key.

The owner comes from the ownership record yonga writes into each entry it uses
(:func:`yonga.adapters.openvino_probe.record_compile_cache_use`). An entry made
before those records existed is attributed by recomputing every key each known
artifact could have produced -- every artifact in the store, plus every
artifact a saved benchmark names -- across the devices, OpenVINO versions and
runtime properties yonga actually uses, the same derivation ``yonga remove``
relies on.

Deletion is narrow on purpose. Only *direct children* of the compile cache root
whose names have the exact shape of a key, and that are real directories rather
than symbolic links, are ever candidates; the check is repeated immediately
before each removal, and nothing else under the cache directory (the managed
temp directory, export environments, metadata) is in scope.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import stat
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from yonga.adapters.openvino_probe import (
    artifact_fingerprint,
    candidate_compile_cache_keys,
    read_compile_cache_record,
)
from yonga.application.artifacts import ArtifactStore
from yonga.config.paths import Paths
from yonga.domain.artifacts import ArtifactManifest
from yonga.domain.compile_cache import (
    CacheAttribution,
    CacheEntryStatus,
    CompileCacheEntry,
    CompileCacheInventory,
    CompileCachePrunePlan,
    CompileCachePruneReport,
    CompileCacheRecord,
    PruneFailure,
    is_cache_key,
)
from yonga.domain.diagnostics import ConfigurationError, FailureCategory, diagnostic
from yonga.domain.hardware import normalise_device
from yonga.util.versions import installed_version

__all__ = [
    "KEY_DEVICES",
    "CompileCacheService",
    "cache_entry_dirs",
    "check_older_than",
    "manifest_key_inputs",
]

#: Devices every artifact's keys are recomputed for, whatever its manifest says:
#: ``chat``, ``bench`` and ``serve`` accept any of them with ``--device``.
KEY_DEVICES: tuple[str, ...] = ("CPU", "GPU", "NPU")

#: The plan reason recorded for an entry chosen by ``--older-than``.
OLDER_THAN_REASON = "older-than"

#: The plan reason recorded for every entry with ``--all``.
ALL_REASON = "all"

_SECONDS_PER_DAY = 86_400.0


def cache_entry_dirs(root: Path) -> list[Path]:
    """Return the compile cache entries directly under ``root``, sorted by name.

    An entry is a real directory (never a symbolic link) whose name has the
    exact shape of a cache key. Everything else is not this service's to judge.
    """
    found: list[Path] = []
    try:
        children = list(os.scandir(root))
    except OSError:
        return found
    for child in children:
        if not is_cache_key(child.name):
            continue
        try:
            if child.is_symlink() or not child.is_dir(follow_symlinks=False):
                continue
        except OSError:
            continue
        found.append(root / child.name)
    return sorted(found)


def manifest_key_inputs(
    manifest: ArtifactManifest,
    *,
    extra_versions: Iterable[str | None] = (),
    extra_devices: Iterable[str] = (),
    extra_property_sets: Iterable[Mapping[str, str]] = (),
) -> tuple[list[str], list[str | None], list[dict[str, str]]]:
    """The devices, OpenVINO versions and property sets an artifact's keys may use.

    Devices: :data:`KEY_DEVICES` plus every device the manifest probed or
    recommends. Versions: "unknown" (``None``), the version that validated the
    artifact and the one that exported it. Property sets: none at all (what
    ``chat`` and ``bench`` use on a device other than the recommended one), the
    recommended runtime properties, and whatever a recorded smoke test applied.
    """
    devices = list(KEY_DEVICES)
    devices.extend(manifest.device_probes)
    runtime = manifest.recommended_runtime
    if runtime is not None:
        devices.append(runtime.device)
    devices.extend(extra_devices)

    versions: list[str | None] = [None]
    if manifest.validation is not None:
        versions.append(manifest.validation.openvino_version)
    versions.append(manifest.export.package_versions.get("openvino"))
    versions.extend(extra_versions)

    property_sets: list[dict[str, str]] = [{}]
    if runtime is not None and runtime.properties:
        property_sets.append(dict(runtime.properties))
    for report in manifest.device_probes.values():
        for result in report.submodels:
            if result.properties:
                property_sets.append(dict(result.properties))
    property_sets.extend(dict(item) for item in extra_property_sets)
    return (
        list(dict.fromkeys(name for name in devices if name)),
        list(dict.fromkeys(versions)),
        property_sets,
    )


def check_older_than(days: float | None) -> float | None:
    """Validate ``--older-than DAYS``: a finite, non-negative number of days."""
    if days is None:
        return None
    if not math.isfinite(days) or days < 0:
        raise ConfigurationError(
            diagnostic(
                "cache.prune.older_than_invalid",
                FailureCategory.CONFIGURATION_ERROR,
                "Invalid --older-than",
                f"--older-than takes a number of days of 0 or more, not {days!r}.",
                remediation=["For example: yonga cache prune --older-than 30"],
                context={"older_than": str(days)},
            )
        )
    return float(days)


@dataclass(frozen=True)
class _Owner:
    """An artifact a cache entry can belong to."""

    fingerprint: str
    artifact_id: str | None
    path: Path | None
    repo_id: str | None
    profile: str | None
    in_store: bool


@dataclass(frozen=True)
class _Attribution:
    """The owner and key inputs a recomputed key matched."""

    owner: _Owner
    device: str
    version: str | None


@dataclass(frozen=True)
class _BenchmarkClaim:
    """What a saved benchmark says about the artifact it measured."""

    artifact_id: str
    artifact_path: Path | None
    device: str | None
    properties: dict[str, str]
    openvino: str | None


def _now() -> datetime:
    return datetime.now(tz=UTC)


class CompileCacheService:
    """Inventory and prune ``paths.compile_cache_dir``.

    ``runtime_version`` returns the installed OpenVINO's full version string
    (the one the runtime folds into every key), or ``None`` when OpenVINO is not
    importable; with ``None`` no entry is ever called stale-runtime, because
    there is nothing to compare against. It is called at most once per
    inventory, and only when the cache holds an entry.
    """

    def __init__(
        self,
        paths: Paths,
        store: ArtifactStore,
        *,
        runtime_version: Callable[[], str | None],
        clock: Callable[[], datetime] = _now,
    ) -> None:
        """Bind the managed layout, the artifact store and the runtime version source."""
        self._paths = paths
        self._store = store
        self._runtime_version = runtime_version
        self._clock = clock

    @property
    def root(self) -> Path:
        """The compile cache root this service manages."""
        return self._paths.compile_cache_dir

    # --- inventory --------------------------------------------------------

    def inventory(self) -> CompileCacheInventory:
        """Measure and classify every entry under the compile cache root. Never deletes."""
        root = self.root
        directories = cache_entry_dirs(root)
        inventory = CompileCacheInventory(
            root=root,
            store_root=self._store.root,
            ignored=self._ignored_names(root, directories),
        )
        if not directories:
            return inventory

        installed = self._installed_version()
        inventory.installed_openvino = installed
        store = self._store_fingerprints()
        records = {directory: read_compile_cache_record(directory) for directory in directories}
        attribution: dict[str, _Attribution] | None = None
        if any(record is None for record in records.values()):
            attribution = self._attribution_table(store, records.values(), installed)

        entries: list[CompileCacheEntry] = []
        for directory in directories:
            size, count, newest = _measure(directory)
            record = records[directory]
            if record is not None:
                entry = self._recorded_entry(directory, record, store)
            else:
                matched = (attribution or {}).get(directory.name)
                entry = self._attributed_entry(directory, matched)
            entry.size_bytes = size
            entry.file_count = count
            entry.last_used_at = _latest(entry.last_used_at, newest)
            _classify(entry, installed=installed, store_root=self._store.root)
            entries.append(entry)
        entries.sort(key=lambda item: (-item.size_bytes, item.key))
        inventory.entries = entries
        return inventory

    def _installed_version(self) -> str | None:
        """The installed OpenVINO version, or ``None`` when it cannot be read."""
        try:
            version = self._runtime_version()
        except Exception:
            return None
        cleaned = (version or "").strip()
        return cleaned or None

    @staticmethod
    def _ignored_names(root: Path, directories: Sequence[Path]) -> list[str]:
        """Names under the root that are not cache entries."""
        kept = {directory.name for directory in directories}
        try:
            names = sorted(child.name for child in os.scandir(root))
        except OSError:
            return []
        return [name for name in names if name not in kept]

    def _store_fingerprints(self) -> dict[str, tuple[Path, ArtifactManifest | None]]:
        """Map the key fingerprint of every artifact in the store to its directory."""
        found: dict[str, tuple[Path, ArtifactManifest | None]] = {}
        for directory in self._store.artifact_directories():
            fingerprint = artifact_fingerprint(directory)
            found.setdefault(fingerprint, (directory, self._store.manifest_for(directory)))
        return found

    def _recorded_entry(
        self,
        directory: Path,
        record: CompileCacheRecord,
        store: Mapping[str, tuple[Path, ArtifactManifest | None]],
    ) -> CompileCacheEntry:
        """Build an entry from its ownership record."""
        recorded_path = Path(record.artifact_path) if record.artifact_path else None
        in_store = record.fingerprint in store
        current: Path | None
        if in_store:
            current = store[record.fingerprint][0]
            exists = True
        else:
            current = recorded_path
            exists = recorded_path is not None and _same_artifact(recorded_path, record.fingerprint)
        return CompileCacheEntry(
            key=directory.name,
            path=directory,
            last_used_at=record.last_used_at,
            device=record.device,
            openvino_version=record.openvino_version,
            artifact_id=record.artifact_id,
            artifact_path=current,
            repo_id=record.repo_id,
            profile=record.profile,
            owner_in_store=in_store,
            owner_exists=exists,
            attribution=CacheAttribution.RECORD,
        )

    @staticmethod
    def _attributed_entry(directory: Path, matched: _Attribution | None) -> CompileCacheEntry:
        """Build an entry for a directory with no ownership record."""
        if matched is None:
            return CompileCacheEntry(key=directory.name, path=directory)
        owner = matched.owner
        exists = owner.in_store or (
            owner.path is not None and _same_artifact(owner.path, owner.fingerprint)
        )
        return CompileCacheEntry(
            key=directory.name,
            path=directory,
            device=matched.device,
            openvino_version=matched.version,
            artifact_id=owner.artifact_id,
            artifact_path=owner.path,
            repo_id=owner.repo_id,
            profile=owner.profile,
            owner_in_store=owner.in_store,
            owner_exists=exists,
            attribution=CacheAttribution.RECOMPUTED,
        )

    # --- attribution --------------------------------------------------------

    def _attribution_table(
        self,
        store: Mapping[str, tuple[Path, ArtifactManifest | None]],
        records: Iterable[CompileCacheRecord | None],
        installed: str | None,
    ) -> dict[str, _Attribution]:
        """Recompute every key the known artifacts could have produced.

        Known artifacts are the store's and the ones saved benchmarks name. The
        OpenVINO versions tried are every version any of them, any ownership
        record and the installed runtime mention, so an entry built by an older
        release is attributed as long as anything remembers that release.
        """
        claims = self._benchmark_claims()
        versions: list[str | None] = [
            None,
            installed,
            installed_version("openvino"),
        ]
        versions.extend(record.openvino_version for record in records if record is not None)
        versions.extend(claim.openvino for claim in claims)
        for _, manifest in store.values():
            if manifest is None:
                continue
            if manifest.validation is not None:
                versions.append(manifest.validation.openvino_version)
            versions.append(manifest.export.package_versions.get("openvino"))
        global_versions = list(dict.fromkeys(versions))

        by_artifact: dict[str, list[_BenchmarkClaim]] = {}
        for claim in claims:
            by_artifact.setdefault(claim.artifact_id, []).append(claim)

        table: dict[str, _Attribution] = {}
        for fingerprint, (directory, manifest) in store.items():
            owner = _Owner(
                fingerprint=fingerprint,
                artifact_id=manifest.artifact_id if manifest is not None else None,
                path=directory,
                repo_id=manifest.resolved_source.repo_id if manifest is not None else None,
                profile=manifest.export.profile if manifest is not None else None,
                in_store=True,
            )
            measured = by_artifact.get(fingerprint, [])
            extra_devices = [claim.device for claim in measured if claim.device]
            extra_properties = [claim.properties for claim in measured]
            if manifest is not None:
                devices, _, property_sets = manifest_key_inputs(
                    manifest,
                    extra_devices=extra_devices,
                    extra_property_sets=extra_properties,
                )
            else:
                devices = [*KEY_DEVICES, *extra_devices]
                property_sets = [{}, *extra_properties]
            self._add_keys(table, owner, devices, global_versions, property_sets)

        for artifact_id, measured in by_artifact.items():
            if artifact_id in store:
                continue
            first = measured[0]
            owner = _Owner(
                fingerprint=artifact_id,
                artifact_id=artifact_id,
                path=first.artifact_path,
                repo_id=None,
                profile=None,
                in_store=False,
            )
            devices = [*KEY_DEVICES, *(claim.device for claim in measured if claim.device)]
            property_sets = [{}, *(claim.properties for claim in measured)]
            self._add_keys(table, owner, devices, global_versions, property_sets)
        return table

    @staticmethod
    def _add_keys(
        table: dict[str, _Attribution],
        owner: _Owner,
        devices: Sequence[str],
        versions: Sequence[str | None],
        property_sets: Sequence[Mapping[str, str]],
    ) -> None:
        """Add one owner's candidate keys; an earlier owner keeps a shared key."""
        keys = candidate_compile_cache_keys(
            owner.fingerprint,
            devices=devices,
            versions=versions,
            property_sets=property_sets,
        )
        for key, (device, version, _) in keys.items():
            table.setdefault(key, _Attribution(owner=owner, device=device, version=version))

    def _benchmark_claims(self) -> list[_BenchmarkClaim]:
        """Read what every saved benchmark recorded about its artifact. Never raises."""
        claims: list[_BenchmarkClaim] = []
        try:
            documents = sorted(self._paths.benchmarks_dir.glob("*.json"))
        except OSError:
            return claims
        for path in documents:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            artifact_id = payload.get("artifact_id")
            if not isinstance(artifact_id, str) or not artifact_id.strip():
                continue
            raw_path = payload.get("artifact_path")
            artifact_path = Path(raw_path) if isinstance(raw_path, str) and raw_path else None
            device = payload.get("device")
            properties = payload.get("runtime_properties")
            packages = payload.get("package_versions")
            openvino = packages.get("openvino") if isinstance(packages, dict) else None
            claims.append(
                _BenchmarkClaim(
                    artifact_id=artifact_id.strip(),
                    artifact_path=artifact_path,
                    device=normalise_device(device) if isinstance(device, str) else None,
                    properties=(
                        {str(key): str(value) for key, value in properties.items()}
                        if isinstance(properties, dict)
                        else {}
                    ),
                    openvino=openvino if isinstance(openvino, str) else None,
                )
            )
        return claims

    # --- prune --------------------------------------------------------------

    def plan_prune(
        self,
        inventory: CompileCacheInventory,
        *,
        older_than_days: float | None = None,
        unattributed: bool = False,
        everything: bool = False,
    ) -> CompileCachePrunePlan:
        """Choose the entries to remove. Deletes nothing.

        Orphaned and stale-runtime entries always qualify. ``older_than_days``
        adds any entry not used for that long, ``unattributed`` adds the entries
        nothing can be traced to, and ``everything`` takes the whole cache.
        """
        days = check_older_than(older_than_days)
        cutoff = self._clock() - timedelta(days=days) if days is not None else None
        plan = CompileCachePrunePlan(
            root=inventory.root,
            older_than_days=days,
            unattributed=unattributed,
            everything=everything,
            installed_openvino=inventory.installed_openvino,
        )
        for entry in inventory.entries:
            reason = _selection_reason(
                entry, cutoff=cutoff, unattributed=unattributed, everything=everything
            )
            if reason is None:
                continue
            plan.entries.append(entry)
            plan.reasons[entry.key] = reason
        return plan

    def prune(
        self, plan: CompileCachePrunePlan, *, dry_run: bool = False
    ) -> CompileCachePruneReport:
        """Remove the planned entries and report the bytes that actually went away."""
        report = CompileCachePruneReport(dry_run=dry_run, planned_bytes=plan.total_bytes)
        if dry_run:
            return report
        for entry in plan.entries:
            present = _is_real_dir(entry.path)
            error = _remove_entry(self.root, entry.path)
            if error is None:
                # An entry something else removed meanwhile freed nothing here.
                report.removed.append(entry.key)
                report.freed_bytes += entry.size_bytes if present else 0
                continue
            # Count only what this prune itself took away: whatever is left of
            # a real directory it started on, and nothing at all for a path that
            # turned out to be something else (a link, a file, a foreign path).
            if entry.path.parent == self.root and _is_real_dir(entry.path):
                remaining = _measure(entry.path)[0]
                report.freed_bytes += max(0, entry.size_bytes - remaining)
            report.failed.append(PruneFailure(key=entry.key, path=entry.path, error=error))
        return report


# --- helpers ----------------------------------------------------------------------------


def _selection_reason(
    entry: CompileCacheEntry,
    *,
    cutoff: datetime | None,
    unattributed: bool,
    everything: bool,
) -> str | None:
    """Why ``entry`` is pruned under these options, or ``None`` when it is kept."""
    if everything:
        return ALL_REASON
    if entry.reclaimable or (unattributed and entry.status is CacheEntryStatus.UNATTRIBUTED):
        return str(entry.status.value)
    if cutoff is not None and (entry.last_used_at is None or entry.last_used_at <= cutoff):
        return OLDER_THAN_REASON
    return None


def _classify(entry: CompileCacheEntry, *, installed: str | None, store_root: Path) -> None:
    """Set the entry's status and the one-line reason for it."""
    if entry.attribution is CacheAttribution.NONE:
        entry.status = CacheEntryStatus.UNATTRIBUTED
        entry.reason = (
            "no ownership record, and no artifact in the store or in a saved benchmark "
            "produces this key"
        )
        return
    owner = entry.artifact_id or (str(entry.artifact_path) if entry.artifact_path else "unknown")
    if not entry.owner_exists:
        entry.status = CacheEntryStatus.ORPHANED
        entry.reason = f"artifact {owner} is no longer in {store_root} or at its recorded path"
        return
    built = entry.openvino_version or "unknown"
    if installed is not None and built != installed:
        entry.status = CacheEntryStatus.STALE_RUNTIME
        entry.reason = f"built by OpenVINO {built}; the installed runtime is {installed}"
        return
    entry.status = CacheEntryStatus.IN_USE
    where = "in the store" if entry.owner_in_store else f"at {entry.artifact_path}"
    entry.reason = f"artifact {owner} is {where}"


def _is_real_dir(path: Path) -> bool:
    """Whether ``path`` is a directory itself, not a link to one. Never raises."""
    try:
        return stat.S_ISDIR(os.lstat(path).st_mode)
    except OSError:
        return False


def _same_artifact(path: Path, fingerprint: str) -> bool:
    """Whether ``path`` still holds the artifact whose key fingerprint is ``fingerprint``."""
    try:
        if not path.is_dir():
            return False
    except OSError:
        return False
    return artifact_fingerprint(path) == fingerprint


def _measure(directory: Path) -> tuple[int, int, datetime | None]:
    """Return ``(bytes, files, newest mtime)`` of one entry, never following a link.

    Symbolic links are neither followed nor counted. The directory's own mtime
    counts, so an empty entry still has an age.
    """
    size = 0
    count = 0
    newest: float | None = None
    try:
        newest = os.lstat(directory).st_mtime
    except OSError:
        return (0, 0, None)
    for current, _, files in os.walk(directory, followlinks=False):
        folder = Path(current)
        for name in files:
            try:
                info = (folder / name).lstat()
            except OSError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            size += info.st_size
            count += 1
            newest = max(newest, info.st_mtime)
    return (size, count, datetime.fromtimestamp(newest, tz=UTC))


def _latest(first: datetime | None, second: datetime | None) -> datetime | None:
    """The later of two optional moments."""
    if first is None:
        return second
    if second is None:
        return first
    return max(first, second)


def _remove_entry(root: Path, path: Path) -> str | None:
    """Delete one cache entry after re-checking it is one. Returns an error, or ``None``.

    The containment rules are checked again here, at the last moment, rather
    than trusted from the plan: the entry must be a direct child of ``root``,
    named like a key, and a real directory. A symbolic link is refused, and
    ``shutil.rmtree`` never follows one inside the tree.
    """
    if path.parent != root or not is_cache_key(path.name):
        return f"{path} is not a compile cache entry under {root}"
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        return f"{path} is not a directory; it was left alone"
    shutil.rmtree(path, ignore_errors=True)
    if path.is_symlink() or path.exists():
        return f"{path} could not be removed completely; check its permissions"
    return None
