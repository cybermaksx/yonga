"""The artifact store: directory layout, listing, lookup, publication and removal.

A *prepared artifact* is a directory of OpenVINO IR plus a tool-owned ``.yonga/``
metadata directory. The manifest inside it is the canonical identity; the
directory name (``<namespace>--<model>/<sha-short>--<profile>/``) is convenience
only, exactly as ``docs/ARCHITECTURE.md`` section 9 requires.

Three invariants shape this module.

**Nothing incomplete is ever advertised as ready.** A directory under the hidden
staging root, a directory whose manifest will not parse, a directory still
carrying the incomplete marker, and a manifest whose own ``ready`` property is
false are all reported with ``ready=False``. ``yonga list`` may still show them so
the user can clean them up, but no caller can mistake one for a usable model
(``SPEC.md`` section 24).

**Publication is all-or-nothing.** :meth:`ArtifactStore.finalize` hashes the
files, writes the manifest *inside* the staging directory, flushes it to stable
storage, and only then hands the directory to
:func:`yonga.adapters.filesystem.finalize_directory`, which renames it into place
on the same filesystem. A crash at any point leaves either the old state or the
new one, never half an artifact.

**Removal only touches what the tool created.** :meth:`ArtifactStore.remove`
refuses any path outside ``paths.artifacts_dir`` and any path inside a Hugging
Face cache root, whatever the caller passes (``SPEC.md`` sections 8.3 and 18.5).
It also refuses to take the compile cache with it when another directory still
claims the same manifest identity, and reports the bytes that actually went away
rather than the bytes it hoped to free.

The hidden staging root is swept rather than left to grow:
:meth:`ArtifactStore.sweep_staging` applies the retention policy of
``SPEC.md`` section 14 to ``paths.staging_dir``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Iterable, Iterator, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from yonga.adapters.filesystem import INCOMPLETE_MARKER_NAME, finalize_directory
from yonga.application.jobs import SweepReport, sweep_job_directories
from yonga.application.progress import NullSink, ProgressEvent, ProgressKind, ProgressSink
from yonga.config.paths import Paths
from yonga.config.settings import Settings
from yonga.domain.artifacts import (
    MANIFEST_DIRNAME,
    MANIFEST_FILENAME,
    ArtifactFileRecord,
    ArtifactManifest,
    ArtifactSummary,
    ValidationReport,
    manifest_path,
)
from yonga.domain.diagnostics import (
    ConfigurationError,
    FailureCategory,
    Severity,
    ValidationError,
    diagnostic,
)
from yonga.domain.models import ModelIdentity, PipelineKind
from yonga.repairs import huggingface_cache_roots
from yonga.util.hashing import sha256_file, short_hash

__all__ = [
    "HASH_SKIP_THRESHOLD_BYTES",
    "SHA_SHORT_LENGTH",
    "SOURCE_FILENAME",
    "STAGING_MODEL_DIRNAME",
    "UNPINNED_REVISION_COMPONENT",
    "VALIDATION_FILENAME",
    "ArtifactStore",
    "RemovalReport",
    "artifact_directory_name",
    "revision_component",
]

#: Name of the validation document written next to the manifest.
VALIDATION_FILENAME: str = "validation.json"

#: Name of the provenance document written next to the manifest.
SOURCE_FILENAME: str = "source.json"

#: Sub-directory of a job staging directory that holds the model itself.
STAGING_MODEL_DIRNAME: str = "model"

#: Characters of a commit SHA kept in the artifact directory name.
SHA_SHORT_LENGTH: int = 12

#: Directory-name component used when the source revision is unknown.
UNPINNED_REVISION_COMPONENT: str = "norev"

#: Files at or above this size are skipped when ``settings.hash_weights`` is off.
#: 64 MiB is comfortably above every configuration/tokenizer asset and comfortably
#: below any real weight shard, so the development shortcut never silently drops
#: the digest of a small file that costs nothing to hash.
HASH_SKIP_THRESHOLD_BYTES: int = 64 * 1024 * 1024

#: Names never included in the artifact file inventory.
_INVENTORY_EXCLUDED_NAMES: frozenset[str] = frozenset({INCOMPLETE_MARKER_NAME})

_HEX_RE = re.compile(r"[0-9a-fA-F]+")
_UNSAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def revision_component(revision: str | None) -> str:
    """Return the directory-name component standing for a source revision.

    A commit SHA is truncated to :data:`SHA_SHORT_LENGTH` characters so the
    directory stays readable; any other reference (a branch or tag name) is
    sanitised, and a long one is shortened with a digest suffix so two different
    references can never collide. ``None`` becomes
    :data:`UNPINNED_REVISION_COMPONENT`.
    """
    if revision is None:
        return UNPINNED_REVISION_COMPONENT
    text = revision.strip()
    if not text:
        return UNPINNED_REVISION_COMPONENT
    if _HEX_RE.fullmatch(text) and len(text) >= SHA_SHORT_LENGTH:
        return text[:SHA_SHORT_LENGTH].lower()
    safe = _UNSAFE_COMPONENT.sub("_", text)
    if len(safe) <= SHA_SHORT_LENGTH:
        return safe
    return f"{safe[:6]}-{short_hash(text, 5)}"


def artifact_directory_name(profile_id: str, revision: str | None) -> str:
    """Return the leaf directory name ``<sha-short>--<profile>`` for an artifact."""
    profile = _UNSAFE_COMPONENT.sub("_", profile_id.strip()) or "unknown-profile"
    return f"{revision_component(revision)}--{profile}"


class RemovalReport(BaseModel):
    """What :meth:`ArtifactStore.remove` deleted, or would delete on a dry run."""

    model_config = ConfigDict(extra="ignore")

    artifact_dir: Path
    artifact_id: str | None = None
    removed: bool = False
    dry_run: bool = False
    freed_bytes: int = 0
    removed_paths: list[Path] = Field(default_factory=list)
    compile_cache_paths: list[Path] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _emit(sink: ProgressSink, event: ProgressEvent) -> None:
    """Send one progress event, never letting a renderer fail a store operation."""
    with suppress(Exception):
        sink.emit(event)


def _directory_size(path: Path) -> int:
    """Sum the sizes of every regular file under ``path``. Never raises."""
    total = 0
    with suppress(OSError):
        for item in path.rglob("*"):
            if item.is_symlink():
                continue
            with suppress(OSError):
                if item.is_file():
                    total += item.stat().st_size
    return total


def _fsync_file(path: Path) -> None:
    """Flush one file to stable storage, ignoring filesystems that cannot."""
    with suppress(OSError):
        handle = os.open(path, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)


def _fsync_directory(path: Path) -> None:
    """Flush a directory entry so a rename cannot resurrect a missing file."""
    with suppress(OSError):
        handle = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(handle)
        finally:
            os.close(handle)


def _write_document(path: Path, text: str) -> None:
    """Write a metadata document and flush it to stable storage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    _fsync_file(path)


def _json_text(payload: object) -> str:
    """Render a JSON document the way every yonga metadata file is written."""
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _is_within(candidate: Path, root: Path) -> bool:
    """Whether ``candidate`` is ``root`` or lives underneath it."""
    try:
        return candidate == root or candidate.is_relative_to(root)
    except (OSError, ValueError):  # pragma: no cover - defensive
        return False


class ArtifactStore:
    """Owns the on-disk layout of prepared artifacts under ``paths.artifacts_dir``."""

    def __init__(self, paths: Paths, settings: Settings) -> None:
        """Create a store rooted at ``paths.artifacts_dir``.

        ``settings`` supplies the policy knobs the store honours — currently
        ``hash_weights``, which lets a developer skip multi-gigabyte digests while
        iterating. It defaults to on, so a published artifact is always fully
        hashed unless someone deliberately turned that off.
        """
        self._paths = paths
        self._settings = settings

    # --- identity and layout ------------------------------------------------

    @property
    def paths(self) -> Paths:
        """Resolved XDG paths this store operates in."""
        return self._paths

    @property
    def settings(self) -> Settings:
        """Settings whose policy this store honours."""
        return self._settings

    @property
    def root(self) -> Path:
        """Root directory holding every prepared artifact."""
        return self._paths.artifacts_dir

    def artifact_dir(
        self,
        source_repo_id: str,
        source_sha: str | None,
        profile_id: str,
    ) -> Path:
        """Return the final directory for one (source revision, profile) pair.

        The layout is ``<artifacts>/<namespace>--<model>/<sha-short>--<profile>/``.
        Nothing is created: this is a pure naming decision so that ``plan`` can
        show the user where an artifact *would* land without side effects.
        """
        family = ModelIdentity(repo_id=source_repo_id).slug
        return self.root / family / artifact_directory_name(profile_id, source_sha)

    def staging_model_dir(self, job_id: str) -> Path:
        """Return the staging directory one job exports into."""
        return self._paths.job_staging_dir(job_id) / STAGING_MODEL_DIRNAME

    # --- reading ------------------------------------------------------------

    def artifact_directories(self) -> list[Path]:
        """Every published artifact directory holding a manifest file, staging excluded.

        The manifest may still be unreadable; :meth:`manifest_for` says. Nothing is
        measured or hashed, so this is cheap enough to call on every command.
        """
        return list(self._artifact_dirs())

    def manifest_for(self, artifact_dir: Path) -> ArtifactManifest | None:
        """Read and validate the manifest of ``artifact_dir``, or ``None``.

        A manifest that is missing, unreadable, not JSON, or does not match the
        schema all return ``None``: the caller's job is to report the directory as
        not ready, not to crash on someone else's leftovers.
        """
        target = manifest_path(artifact_dir)
        if not target.is_file():
            return None
        try:
            return ArtifactManifest.model_validate_file(target)
        except ValidationError:
            return None

    def is_ready(self, artifact_dir: Path) -> bool:
        """Whether ``artifact_dir`` is a complete, validated, probe-backed artifact.

        Four independent conditions must hold: the directory is not under the
        hidden staging root, it carries no incomplete marker, its manifest parses,
        and the manifest itself reports validation success plus at least one
        device that probed successfully.
        """
        directory = Path(artifact_dir)
        if self._is_staging(directory):
            return False
        if (directory / INCOMPLETE_MARKER_NAME).exists():
            return False
        manifest = self.manifest_for(directory)
        return manifest is not None and manifest.ready

    def list_artifacts(self) -> list[ArtifactSummary]:
        """Summarise every artifact directory under the store root.

        Directories under ``.staging`` are never returned. A directory whose
        manifest cannot be read is returned with ``ready=False`` and a synthetic
        id so ``yonga list`` can show it and ``yonga remove`` can clean it up.
        Results are ordered newest first.
        """
        benchmarks = self._benchmark_index()
        summaries = [
            self._summarise(directory, benchmarks=benchmarks) for directory in self._artifact_dirs()
        ]
        summaries.sort(key=lambda item: (item.created_at, str(item.path)), reverse=True)
        return summaries

    def summarise(self, artifact_dir: Path) -> ArtifactSummary:
        """Summarise a single artifact directory."""
        return self._summarise(Path(artifact_dir), benchmarks=self._benchmark_index())

    def find(self, reference: str | Path) -> Path | None:
        """Resolve an artifact id, a repository id or a filesystem path to a directory.

        Matching is tried in decreasing order of precision: an existing directory
        on disk, an exact artifact id, an unambiguous artifact-id prefix, the
        directory name, and finally the requested or resolved repository id
        (optionally written ``repo_id@revision``). Ready artifacts always win over
        unready ones, and the newest wins among equals.
        """
        text = str(reference).strip()
        if not text:
            return None

        direct = self._as_directory(text)
        if direct is not None:
            return direct

        candidates = self.matches(text)
        return candidates[0].path if candidates else None

    def matches(self, reference: str) -> list[ArtifactSummary]:
        """Return every catalogued artifact the most precise matching rule selects.

        The rules are the ones :meth:`find` tries, in the same order, and only
        the first rule that matches anything contributes: an exact artifact id
        is never mixed with the other artifacts of its repository. The result is
        ordered best first (ready before unready, then newest), so ``find`` can
        take the head while a destructive caller can see that more than one
        artifact answered and refuse to guess. A filesystem path is not looked
        up here; it names one directory and never needs disambiguation.
        """
        text = str(reference).strip()
        if not text:
            return []
        summaries = self.list_artifacts()
        for matcher in (
            self._match_artifact_id,
            self._match_directory_name,
            self._match_repo_id,
            self._match_artifact_id_prefix,
        ):
            matched = [summary for summary in summaries if matcher(summary, text)]
            if matched:
                matched.sort(key=lambda item: (item.ready, item.created_at), reverse=True)
                return matched
        return []

    # --- publication --------------------------------------------------------

    def finalize(
        self,
        staging_dir: Path,
        final_dir: Path,
        manifest: ArtifactManifest,
        *,
        force: bool = False,
        progress: ProgressSink | None = None,
        validation: ValidationReport | None = None,
    ) -> Path:
        """Publish a staging directory as a finished artifact, atomically.

        The order matters and is the whole point of this method:

        1. every file in the staging directory is inventoried and hashed, and the
           inventory is written onto ``manifest`` in place (so the caller can read
           the digests back);
        2. ``.yonga/manifest.json``, ``.yonga/validation.json`` and
           ``.yonga/source.json`` are written **inside the staging directory** and
           flushed with ``fsync``;
        3. the staging directory is renamed onto ``final_dir``.

        Because the manifest is written before the rename, a reader can never
        observe an artifact directory without its manifest, and a crash leaves the
        staging directory behind rather than a half-published artifact.

        ``force`` is required to replace a destination that is not a tool-owned
        incomplete build. Returns the final directory.
        """
        staging = Path(staging_dir)
        final = Path(final_dir)
        sink = progress if progress is not None else NullSink()

        if not staging.is_dir():
            raise ValidationError(
                diagnostic(
                    "artifact.finalize.missing_staging",
                    FailureCategory.UNKNOWN_RUNTIME_ERROR,
                    "Nothing to publish",
                    f"The staging directory {staging} does not exist.",
                    remediation=["Re-run preparation; the export left nothing behind."],
                    context={"staging": str(staging)},
                )
            )

        manifest.files = self._inventory(staging, sink)
        report = validation if validation is not None else manifest.validation

        metadata_dir = staging / MANIFEST_DIRNAME
        metadata_dir.mkdir(parents=True, exist_ok=True)
        _write_document(metadata_dir / MANIFEST_FILENAME, manifest.to_json_text())
        _write_document(
            metadata_dir / VALIDATION_FILENAME,
            _json_text(report.model_dump(mode="json") if report is not None else {}),
        )
        _write_document(metadata_dir / SOURCE_FILENAME, _json_text(self._source_document(manifest)))
        _fsync_directory(metadata_dir)
        _fsync_directory(staging)

        notes: list[str] = []
        published = finalize_directory(staging, final, force=force, notes=notes)
        _fsync_directory(published.parent)
        for note in notes:
            _emit(
                sink,
                ProgressEvent(kind=ProgressKind.LOG, message=note, severity=Severity.INFO),
            )
        return published

    # --- removal ------------------------------------------------------------

    def remove(
        self,
        artifact: str | Path | ArtifactSummary,
        *,
        dry_run: bool = False,
    ) -> RemovalReport:
        """Delete one prepared artifact and the compile-cache entries keyed to it.

        Only the tool-owned artifact directory and directories under
        ``paths.compile_cache_dir`` are ever touched. A path outside
        ``paths.artifacts_dir``, the artifacts root itself, and any path resolving
        inside a Hugging Face cache root are all refused with a typed
        :class:`~yonga.domain.diagnostics.ConfigurationError`; the shared Hub cache
        is not this tool's to delete (``SPEC.md`` section 8.3).
        """
        target = self._removal_target(artifact)
        root = self._resolved_root()
        if target == root or not _is_within(target, root):
            raise ConfigurationError(
                diagnostic(
                    "artifact.remove.outside_store",
                    FailureCategory.CONFIGURATION_ERROR,
                    "Refusing to remove a path outside the artifact store",
                    f"{target} is not a prepared artifact inside {root}.",
                    remediation=[
                        "Pass an artifact id, a repository id, or a path under the store root.",
                        "Run 'yonga list' to see the prepared artifacts.",
                    ],
                    context={"path": str(target), "artifacts_dir": str(root)},
                )
            )
        for cache_root in huggingface_cache_roots():
            if _is_within(target, cache_root):
                raise ConfigurationError(
                    diagnostic(
                        "artifact.remove.hf_cache",
                        FailureCategory.CONFIGURATION_ERROR,
                        "Refusing to delete anything in the Hugging Face cache",
                        (
                            f"{target} resolves inside the shared Hugging Face cache "
                            f"({cache_root}); yonga never deletes downloaded sources."
                        ),
                        remediation=[
                            "Use the Hugging Face tooling to manage its own cache.",
                        ],
                        context={"path": str(target), "cache_root": str(cache_root)},
                    )
                )

        manifest = self.manifest_for(target)
        notes: list[str] = []
        if manifest is None:
            notes.append(f"{target} has no readable manifest; removing the directory anyway.")

        # The compile cache is keyed on the artifact's *identity*, so a second
        # directory carrying the same manifest is entitled to the same cache. A
        # copy of an artifact must therefore not take the original's compiled
        # blobs with it when it goes.
        shared = self._directories_sharing_identity(target, manifest)
        cache_dirs = [] if shared else self._compile_cache_dirs(manifest)
        if shared and manifest is not None:
            notes.append(
                f"The compile cache is keyed on artifact id {manifest.artifact_id}, which "
                f"{len(shared)} other directory in the store still claims; it is kept."
            )
        elif manifest is not None and not cache_dirs:
            notes.append("No compile-cache entries were found for this artifact.")

        removed_paths = [target, *cache_dirs]
        planned = [(path, _directory_size(path)) for path in removed_paths]

        report = RemovalReport(
            artifact_dir=target,
            artifact_id=manifest.artifact_id if manifest is not None else None,
            removed=False,
            dry_run=dry_run,
            freed_bytes=sum(size for _, size in planned),
            removed_paths=removed_paths,
            compile_cache_paths=cache_dirs,
            notes=notes,
        )
        if dry_run:
            return report

        shutil.rmtree(target, ignore_errors=True)
        for cache_dir in cache_dirs:
            shutil.rmtree(cache_dir, ignore_errors=True)
        self._prune_empty_parent(target, root)
        report.removed = not target.exists()
        # Report what actually went away, not what was hoped for: a tree that
        # survived because it was read-only must not be counted as reclaimed.
        report.freed_bytes = max(0, sum(size - _directory_size(path) for path, size in planned))
        return report

    # --- retention ----------------------------------------------------------

    def sweep_staging(
        self,
        *,
        protected: Iterable[str] = (),
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> SweepReport:
        """Retire stale staging directories under ``paths.staging_dir``.

        ``yonga list`` hides the staging root, so the space a failed preparation
        leaves there is invisible until the disk fills. This is the garbage
        collector that keeps it bounded: it removes the working directories of
        jobs that are neither running nor recent, honouring
        ``settings.workspace_retention_hours`` and
        ``settings.workspace_retention_max``. See
        :func:`yonga.application.jobs.sweep_job_directories` for the exact policy
        and the containment rules — nothing outside the direct children of the
        staging root is ever touched, and a live job's directory never is.
        """
        return sweep_job_directories(
            self._paths.staging_dir,
            paths=self._paths,
            settings=self._settings,
            protected=protected,
            now=now,
            dry_run=dry_run,
        )

    def staging_bytes(self) -> int:
        """Total bytes held under the hidden staging root. Never raises."""
        return _directory_size(self._paths.staging_dir)

    # --- internals ----------------------------------------------------------

    def _resolved_root(self) -> Path:
        """Return the artifacts root, resolved for containment checks."""
        try:
            return self.root.expanduser().resolve()
        except OSError:  # pragma: no cover - defensive
            return self.root

    def _is_staging(self, directory: Path) -> bool:
        """Whether ``directory`` lives under the hidden staging root.

        Any path component starting with a dot counts, which covers
        ``artifacts/.staging/<job>/model`` and any other hidden working area a
        future stage might introduce.
        """
        try:
            relative = directory.relative_to(self.root)
        except ValueError:
            return _is_within(directory, self._paths.staging_dir)
        return any(part.startswith(".") for part in relative.parts)

    def _artifact_dirs(self) -> Iterator[Path]:
        """Yield every non-staging directory under the root that holds a manifest."""
        root = self.root
        if not root.is_dir():
            return
        pattern = f"*/{MANIFEST_DIRNAME}/{MANIFEST_FILENAME}"
        seen: set[Path] = set()
        try:
            candidates = sorted(root.rglob(pattern))
        except OSError:  # pragma: no cover - unreadable store root
            return
        for found in candidates:
            directory = found.parent.parent
            if self._is_staging(directory) or directory in seen:
                continue
            seen.add(directory)
            yield directory

    def _summarise(
        self,
        directory: Path,
        *,
        benchmarks: Mapping[str, datetime],
    ) -> ArtifactSummary:
        """Build one summary, degrading gracefully when the manifest is unusable."""
        manifest = self.manifest_for(directory)
        size = _directory_size(directory)
        staging = self._is_staging(directory)
        incomplete = (directory / INCOMPLETE_MARKER_NAME).exists()
        if manifest is None:
            return ArtifactSummary(
                artifact_id=f"unknown:{directory.name}",
                path=directory,
                repo_id=self._repo_id_from_path(directory),
                profile=self._profile_from_path(directory),
                size_bytes=size,
                created_at=self._directory_mtime(directory),
                ready=False,
                pipeline=PipelineKind.UNKNOWN,
            )
        runtime = manifest.recommended_runtime
        return ArtifactSummary(
            artifact_id=manifest.artifact_id,
            path=directory,
            repo_id=manifest.resolved_source.repo_id,
            resolved_revision=manifest.resolved_source.resolved_revision,
            profile=manifest.export.profile,
            recommended_device=runtime.device if runtime is not None else None,
            size_bytes=size,
            created_at=manifest.created_at,
            ready=manifest.ready and not staging and not incomplete,
            pipeline=manifest.pipeline,
            last_benchmark=benchmarks.get(manifest.artifact_id),
        )

    @staticmethod
    def _directory_mtime(directory: Path) -> datetime:
        """Return the directory's modification time as an aware UTC timestamp."""
        try:
            return datetime.fromtimestamp(directory.stat().st_mtime, tz=UTC)
        except OSError:  # pragma: no cover - vanished between listing and stat
            return datetime.now(tz=UTC)

    def _repo_id_from_path(self, directory: Path) -> str:
        """Recover a best-effort repository id from the family directory name."""
        try:
            relative = directory.relative_to(self.root)
        except ValueError:
            return directory.parent.name.replace("--", "/", 1)
        if not relative.parts:
            return directory.name
        return relative.parts[0].replace("--", "/", 1)

    @staticmethod
    def _profile_from_path(directory: Path) -> str:
        """Recover a best-effort profile id from the leaf directory name."""
        name = directory.name
        if "--" in name:
            return name.split("--", 1)[1]
        return ""

    def _benchmark_index(self) -> dict[str, datetime]:
        """Map artifact id to the most recent benchmark timestamp on disk."""
        index: dict[str, datetime] = {}
        directory = self._paths.benchmarks_dir
        if not directory.is_dir():
            return index
        try:
            documents = sorted(directory.glob("*.json"))
        except OSError:  # pragma: no cover - unreadable benchmarks directory
            return index
        for document in documents:
            payload = self._read_json_object(document)
            if payload is None:
                continue
            artifact_id = payload.get("artifact_id")
            created = payload.get("created_at")
            if not isinstance(artifact_id, str) or not isinstance(created, str):
                continue
            try:
                moment = datetime.fromisoformat(created)
            except ValueError:
                continue
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=UTC)
            previous = index.get(artifact_id)
            if previous is None or moment > previous:
                index[artifact_id] = moment
        return index

    @staticmethod
    def _read_json_object(path: Path) -> dict[str, object] | None:
        """Read a JSON object from disk, returning ``None`` on any problem."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def _as_directory(self, reference: str) -> Path | None:
        """Return ``reference`` as an artifact directory if it points at one on disk."""
        candidate = Path(reference).expanduser()
        if candidate.name == MANIFEST_FILENAME and candidate.parent.name == MANIFEST_DIRNAME:
            candidate = candidate.parent.parent
        elif candidate.name == MANIFEST_DIRNAME:
            candidate = candidate.parent
        if not candidate.is_dir():
            return None
        if manifest_path(candidate).is_file():
            return candidate
        return None

    @staticmethod
    def _match_artifact_id(summary: ArtifactSummary, reference: str) -> bool:
        """Exact artifact-id match."""
        return summary.artifact_id == reference

    @staticmethod
    def _match_artifact_id_prefix(summary: ArtifactSummary, reference: str) -> bool:
        """Unambiguous artifact-id prefix match; short prefixes are rejected."""
        minimum_prefix = 6
        if len(reference) < minimum_prefix:
            return False
        return summary.artifact_id.startswith(reference)

    @staticmethod
    def _match_directory_name(summary: ArtifactSummary, reference: str) -> bool:
        """Match the leaf directory name or ``<family>/<leaf>``."""
        path = summary.path
        return reference in {path.name, f"{path.parent.name}/{path.name}"}

    @staticmethod
    def _match_repo_id(summary: ArtifactSummary, reference: str) -> bool:
        """Match the resolved repository id, with or without an ``@revision`` suffix."""
        repo_id, _, revision = reference.partition("@")
        if summary.repo_id.lower() != repo_id.strip().lower():
            return False
        if not revision:
            return True
        resolved = summary.resolved_revision or ""
        return resolved.startswith(revision) or revision.startswith(resolved[:SHA_SHORT_LENGTH])

    def _inventory(self, staging: Path, sink: ProgressSink) -> list[ArtifactFileRecord]:
        """Inventory and hash every file in ``staging``, emitting HASH progress.

        The tool-owned ``.yonga`` directory is excluded: it is written after this
        runs, so hashing it would record digests of files that do not exist yet.
        Digests of large files are skipped when ``settings.hash_weights`` is off,
        and the record then carries ``sha256=None`` rather than a wrong value.
        """
        files = sorted(
            item
            for item in staging.rglob("*")
            if item.is_file() and not item.is_symlink() and not self._excluded(staging, item)
        )
        total = len(files)
        records: list[ArtifactFileRecord] = []
        hash_everything = bool(self._settings.hash_weights)
        for index, item in enumerate(files, start=1):
            relative = item.relative_to(staging).as_posix()
            try:
                size = item.stat().st_size
            except OSError:  # pragma: no cover - vanished mid-inventory
                size = 0
            skip = not hash_everything and size >= HASH_SKIP_THRESHOLD_BYTES
            _emit(
                sink,
                ProgressEvent(
                    kind=ProgressKind.HASH,
                    message=relative,
                    current=index,
                    total=total,
                    status="skipped" if skip else "hashing",
                    data={"size_bytes": size},
                ),
            )
            digest: str | None = None
            if not skip:
                try:
                    digest = sha256_file(item)
                except OSError:  # pragma: no cover - unreadable file mid-inventory
                    digest = None
            records.append(ArtifactFileRecord(path=relative, size=size, sha256=digest))
        records.sort(key=lambda record: record.path)
        return records

    @staticmethod
    def _excluded(staging: Path, item: Path) -> bool:
        """Whether a staged file is tool metadata rather than artifact content."""
        relative = item.relative_to(staging)
        if relative.parts and relative.parts[0] == MANIFEST_DIRNAME:
            return True
        return item.name in _INVENTORY_EXCLUDED_NAMES

    @staticmethod
    def _source_document(manifest: ArtifactManifest) -> dict[str, object]:
        """Build the small provenance document written beside the manifest."""
        return {
            "artifact_id": manifest.artifact_id,
            "requested_model": manifest.requested_model.model_dump(mode="json"),
            "resolved_source": manifest.resolved_source.model_dump(mode="json"),
            "source_license": manifest.source_license,
            "architecture": manifest.architecture,
            "task": manifest.task,
            "pipeline": manifest.pipeline.value,
        }

    def _removal_target(self, artifact: str | Path | ArtifactSummary) -> Path:
        """Resolve whatever the caller passed into a concrete directory path."""
        if isinstance(artifact, ArtifactSummary):
            candidate: Path = artifact.path
        elif isinstance(artifact, Path):
            candidate = artifact
        else:
            candidate = self._unique_removal_match(artifact)
        try:
            return candidate.expanduser().resolve()
        except OSError:  # pragma: no cover - defensive
            return candidate

    def _unique_removal_match(self, reference: str) -> Path:
        """Resolve a textual reference to exactly one directory, or refuse.

        :meth:`find` settles a tie by taking the newest ready artifact, which is
        the right answer for a command that only reads. Deleting is not undone
        by asking again, so a repository id (or an id prefix, or a copied
        artifact id) that matches several artifacts is refused with every
        candidate named, even under ``--yes``.
        """
        direct = self._as_directory(str(reference).strip())
        if direct is not None:
            return direct
        candidates = self.matches(reference)
        if not candidates:
            raise ConfigurationError(
                diagnostic(
                    "artifact.remove.not_found",
                    FailureCategory.CONFIGURATION_ERROR,
                    "No such prepared artifact",
                    f"Nothing in the artifact store matches {reference!r}.",
                    remediation=["Run 'yonga list' to see the prepared artifacts."],
                    context={"reference": str(reference)},
                )
            )
        if len(candidates) == 1:
            return candidates[0].path
        # Name each candidate by its id unless two of them share one (a copied
        # artifact); then only the path tells them apart.
        ids = [summary.artifact_id for summary in candidates]
        by_path = len(set(ids)) != len(ids)
        remediation = [
            (
                f"yonga remove {summary.path if by_path else summary.artifact_id}"
                f"  ({summary.profile or 'unknown profile'}, "
                f"{summary.recommended_device or 'no device'}"
                f"{'' if summary.ready else ', not ready'})"
            )
            for summary in candidates
        ]
        remediation.append("Run 'yonga list' to compare them.")
        raise ConfigurationError(
            diagnostic(
                "artifact.remove.ambiguous",
                FailureCategory.CONFIGURATION_ERROR,
                "More than one prepared artifact matches",
                (
                    f"{reference!r} matches {len(candidates)} prepared artifacts; nothing was "
                    "removed. Name the one to delete by its artifact id."
                ),
                remediation=remediation,
                context={
                    "reference": str(reference),
                    "candidates": ", ".join(
                        str(summary.path) if by_path else summary.artifact_id
                        for summary in candidates
                    ),
                },
            )
        )

    def _directories_sharing_identity(
        self,
        target: Path,
        manifest: ArtifactManifest | None,
    ) -> list[Path]:
        """Return other artifact directories that claim ``manifest``'s identity.

        The manifest is the identity and the directory name is convenience
        (``docs/ARCHITECTURE.md`` section 9), so a copy of an artifact is the same
        artifact and shares its compile cache. Removing one copy must leave that
        cache alone while another directory can still use it.
        """
        if manifest is None:
            return []
        shared: list[Path] = []
        for directory in self._artifact_dirs():
            try:
                resolved = directory.expanduser().resolve()
            except OSError:  # pragma: no cover - defensive
                resolved = directory
            if resolved == target:
                continue
            other = self.manifest_for(directory)
            if other is not None and other.artifact_id == manifest.artifact_id:
                shared.append(directory)
        return shared

    def _compile_cache_dirs(self, manifest: ArtifactManifest | None) -> list[Path]:
        """Return the existing compile-cache directories keyed to this artifact.

        Two sources are combined. The ownership record yonga writes into every
        entry it uses names the artifact directly, which is the only way to find
        an entry built by an OpenVINO release the manifest never mentions. Entries
        made before those records existed are found the old way: the cache key is
        a digest of (artifact identity, device, OpenVINO version, runtime
        properties), so it cannot be inverted, and every key this artifact could
        plausibly have produced is recomputed from its own manifest and kept when
        the directory exists. :func:`yonga.application.compile_cache.manifest_key_inputs`
        decides which devices, versions and properties are tried, and
        :func:`yonga.application.compile_cache.known_key_versions` adds the
        installed runtime's full version, the versions other entries' records
        name and the ones saved benchmarks recorded -- the same derivation
        ``yonga cache list`` uses. The installed runtime is only asked (which
        imports OpenVINO) when an entry without a record exists.

        The identity half of the key is the manifest's ``artifact_id`` and nothing
        else. Digesting the directory's *contents* instead would make two
        byte-identical artifacts share a key even when they are different
        artifacts, and would make the key change under an artifact whenever a file
        in it changed. ``artifact_id`` is derived from the source repository,
        resolved revision and profile, which is exactly what a compiled blob is
        valid for. The OpenVINO version stays in the key because blobs are not
        portable across runtime releases (``docs/ARCHITECTURE.md`` section 7).
        The helpers are imported lazily: the key derivation lives in the OpenVINO
        probe adapter (a pure function with no OpenVINO import of its own), and
        deriving the key twice in two places would be a guaranteed source of drift.
        """
        if manifest is None:
            return []
        from yonga.adapters.openvino_probe import (
            candidate_compile_cache_keys,
            openvino_runtime_version,
            read_compile_cache_record,
        )
        from yonga.application.compile_cache import (
            cache_entry_dirs,
            known_key_versions,
            manifest_key_inputs,
        )

        cache_root = self._paths.compile_cache_dir
        found: list[Path] = []
        entries = cache_entry_dirs(cache_root)
        records = [read_compile_cache_record(entry) for entry in entries]
        for entry, record in zip(entries, records, strict=True):
            if record is not None and record.fingerprint == manifest.artifact_id:
                found.append(entry)
        if all(record is not None for record in records):
            # Every entry names its owner; nothing is left to recompute.
            return found
        installed: str | None = None
        with suppress(Exception):
            installed = openvino_runtime_version(strict=True)
        devices, versions, property_sets = manifest_key_inputs(
            manifest,
            extra_versions=known_key_versions(self._paths, records, installed=installed),
        )
        keys = candidate_compile_cache_keys(
            manifest.artifact_id,
            devices=devices,
            versions=versions,
            property_sets=property_sets,
        )
        names = {entry.name: entry for entry in entries}
        for key in keys:
            candidate = names.get(key)
            if candidate is not None and candidate not in found:
                found.append(candidate)
        return found

    @staticmethod
    def _prune_empty_parent(target: Path, root: Path) -> None:
        """Remove the family directory when the last artifact in it is gone."""
        parent = target.parent
        if parent == root or not _is_within(parent, root):
            return
        with suppress(OSError):
            if not any(parent.iterdir()):
                parent.rmdir()
