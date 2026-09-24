"""Reusing a completed export that a failed preparation left behind.

This is the whole of yonga's resume support, and it is deliberately narrow.
Downloads resume on their own through the Hugging Face cache. The export does
not: ``optimum-cli export openvino`` converts and quantizes in one in-process
run (``main_export`` calls ``export_from_model`` and then applies weight
compression before it returns), it writes nothing that marks a stage as done,
and a run killed part-way leaves files that are indistinguishable from a
finished run's except by being wrong. So an interrupted *export* restarts from
scratch, and ``docs/ARCHITECTURE.md`` section 5 forbids pretending otherwise.

What can be reused is an export that *finished*, when the job failed later: a
device probe that failed, a validation defect, a smoke test cut short by
Ctrl-C. On a 9B model that export is the half-hour part of the job. So the
moment the exporter succeeds, :func:`write_checkpoint` records an
:class:`~yonga.domain.artifacts.ExportCheckpoint` in the staging directory: a
fingerprint of everything that determines the output, and the size, mtime and
(for small files) digest of every file produced. A later preparation with the
same fingerprint adopts that directory with :func:`adopt` instead of
exporting, once every file has been shown to be exactly what the exporter
wrote. Files a repair rewrote are restored from the repair's own backup first,
so validation and repair then run against the export as it originally was.

Anything that does not check out is discarded and the export simply runs.
Reuse is an optimisation: it must never be the reason a preparation fails.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from yonga import __version__
from yonga.adapters.filesystem import INCOMPLETE_MARKER_NAME, mark_incomplete
from yonga.application.jobs import JobContext, live_job_ids
from yonga.config.paths import Paths
from yonga.domain.artifacts import (
    EXPORT_CHECKPOINT_FILENAME,
    MANIFEST_DIRNAME,
    CheckpointFile,
    ExportCheckpoint,
    ExportRecord,
)
from yonga.domain.plans import ReusableExport
from yonga.util.hashing import sha256_file

__all__ = [
    "HASHED_MAX_BYTES",
    "adopt",
    "checkpoint_path",
    "export_fingerprint",
    "find_reusable",
    "write_checkpoint",
]

#: Files at most this large are identified by digest, larger ones by size and
#: modification time. Every file a repair can rewrite (IR XML, tokenizer
#: models, JSON configs) is far below it; weight ``.bin`` files are far above.
HASHED_MAX_BYTES: int = 16 * 2**20

#: Where repairs keep the originals of the files they rewrite.
_REPAIR_BACKUPS = "repairs"


def checkpoint_path(staging: Path) -> Path:
    """The checkpoint file of the export in ``staging``."""
    return staging / MANIFEST_DIRNAME / EXPORT_CHECKPOINT_FILENAME


def export_fingerprint(
    *,
    source_repo_id: str,
    source_revision: str | None,
    task: str | None,
    profile: Mapping[str, object],
    trust_remote_code: bool,
    backend: str,
    environment_key: str | None,
    package_versions: Mapping[str, str],
) -> str:
    """Digest every input that determines what the exporter writes.

    The environment is identified by its key when the export runs in a managed
    one (the key already digests every pin), and by the installed exporter
    package versions otherwise. yonga's own version is included too: a release
    can change how a profile becomes exporter flags without changing the
    profile, and a stale export is worth less than a fresh one.
    """
    payload = {
        "tool_version": __version__,
        "source": {"repo_id": source_repo_id, "revision": source_revision},
        "task": task,
        "profile": dict(profile),
        "trust_remote_code": trust_remote_code,
        "backend": backend,
        "environment": (
            {"key": environment_key}
            if environment_key is not None
            else {"packages": dict(sorted(package_versions.items()))}
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_checkpoint(
    staging: Path,
    *,
    fingerprint: str,
    record: ExportRecord,
    ctx: JobContext,
) -> Path | None:
    """Record that the export in ``staging`` finished. Never raises.

    Must be called straight after the exporter succeeds, before validation or
    any repair has touched the output: the checkpoint describes the export as
    the exporter left it.
    """
    try:
        files = {
            relative: _file_state(path, hashed=path.stat().st_size <= HASHED_MAX_BYTES)
            for relative, path in _exported_files(staging)
        }
        checkpoint = ExportCheckpoint(
            fingerprint=fingerprint,
            job_id=ctx.job_id,
            completed_at=datetime.now(UTC),
            record=record,
            files=files,
        )
        target = checkpoint_path(staging)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.partial")
        partial.write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")
        partial.replace(target)
    except OSError as error:
        ctx.logger.warning(
            "export.checkpoint_failed",
            "could not record the export checkpoint; a retry will export again",
            error=str(error),
        )
        return None
    ctx.logger.info(
        "export.checkpoint_written",
        "export checkpoint recorded",
        files=len(files),
        fingerprint=fingerprint[:12],
    )
    return target


def find_reusable(
    paths: Paths,
    fingerprint: str,
    *,
    exclude_job: str | None = None,
) -> ReusableExport | None:
    """The newest retained export with ``fingerprint`` that looks intact. Read-only.

    Only direct children of the staging root are considered, never a live
    job's. This check is cheap -- every file present at its recorded size --
    because it runs while planning; :func:`adopt` does the thorough one.
    """
    root = paths.staging_dir
    if not root.is_dir():
        return None
    live = live_job_ids(paths)
    best: ReusableExport | None = None
    for job_dir in _children(root):
        if job_dir.name == exclude_job or job_dir.name in live:
            continue
        staging = job_dir / "model"
        checkpoint = _read_checkpoint(staging)
        if checkpoint is None or checkpoint.fingerprint != fingerprint:
            continue
        if not _sizes_match(staging, checkpoint):
            continue
        candidate = ReusableExport(
            job_id=job_dir.name,
            path=staging,
            completed_at=checkpoint.completed_at,
            fingerprint=checkpoint.fingerprint,
        )
        if best is None or candidate.completed_at > best.completed_at:
            best = candidate
    return best


def adopt(candidate: ReusableExport, staging: Path, ctx: JobContext) -> ExportCheckpoint | None:
    """Move ``candidate`` into ``staging`` and prove it intact, or return ``None``.

    The directory is claimed first, by a rename within the staging root, so two
    preparations can never adopt the same export. It is then checked file by
    file; a file a repair rewrote is restored from that repair's backup when the
    backup's digest is the one the exporter wrote. Everything the exporter did
    not write (the previous job's validation record, the repair backups) is
    removed, so what remains is the export exactly as it finished.

    On any mismatch the claimed directory is emptied and ``None`` returned: the
    caller then exports as though there had been nothing to reuse.
    """
    if not _claim(candidate.path, staging, ctx):
        return None
    with contextlib.suppress(OSError):
        candidate.path.parent.rmdir()  # the old job's now-empty staging husk

    checkpoint = _read_checkpoint(staging)
    problem = (
        "the checkpoint is unreadable"
        if checkpoint is None
        else "its fingerprint changed"
        if checkpoint.fingerprint != candidate.fingerprint
        else _restore_or_explain(staging, checkpoint)
    )
    if checkpoint is None or problem is not None:
        ctx.logger.warning(
            "export.reuse_rejected",
            f"the retained export of job {candidate.job_id} cannot be reused: {problem}",
            source_job=candidate.job_id,
        )
        _empty(staging)
        mark_incomplete(staging, job_id=ctx.job_id)
        return None

    _remove_unrecorded(staging, checkpoint)
    mark_incomplete(staging, job_id=ctx.job_id)
    ctx.logger.info(
        "export.reused",
        f"adopted the completed export of job {candidate.job_id}",
        source_job=candidate.job_id,
        files=len(checkpoint.files),
    )
    return checkpoint


# --- internals -------------------------------------------------------------------


def _children(root: Path) -> list[Path]:
    try:
        return sorted(child for child in root.iterdir() if child.is_dir())
    except OSError:
        return []


def _exported_files(staging: Path) -> list[tuple[str, Path]]:
    """Every regular file the exporter produced, as ``(posix relative path, path)``."""
    found: list[tuple[str, Path]] = []
    for path in sorted(staging.rglob("*")):
        relative = path.relative_to(staging)
        if relative.parts[0] in (MANIFEST_DIRNAME, INCOMPLETE_MARKER_NAME):
            continue
        if path.is_file() and not path.is_symlink():
            found.append((relative.as_posix(), path))
    return found


def _file_state(path: Path, *, hashed: bool) -> CheckpointFile:
    stat = path.stat()
    return CheckpointFile(
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        sha256=sha256_file(path) if hashed else None,
    )


def _read_checkpoint(staging: Path) -> ExportCheckpoint | None:
    target = checkpoint_path(staging)
    try:
        checkpoint = ExportCheckpoint.model_validate_json(target.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError):
        return None
    # Adoption writes to and deletes around these paths: one that could leave the
    # staging directory makes the whole checkpoint untrustworthy.
    if not all(_contained(relative) for relative in checkpoint.files):
        return None
    return checkpoint


def _contained(relative: str) -> bool:
    path = PurePosixPath(relative)
    return bool(path.parts) and not path.is_absolute() and ".." not in path.parts


def _sizes_match(staging: Path, checkpoint: ExportCheckpoint) -> bool:
    if not checkpoint.files:
        return False
    for relative, expected in checkpoint.files.items():
        try:
            if (staging / relative).stat().st_size != expected.size:
                return False
        except OSError:
            return False
    return True


def _claim(source: Path, staging: Path, ctx: JobContext) -> bool:
    """Rename ``source`` onto ``staging``, which must be absent or an empty husk."""
    try:
        if staging.exists():
            if any(child.name != INCOMPLETE_MARKER_NAME for child in staging.iterdir()):
                return False
            shutil.rmtree(staging)
        staging.parent.mkdir(parents=True, exist_ok=True)
        source.replace(staging)
    except OSError as error:
        # Most often another preparation claimed it first; either way, export.
        ctx.logger.info(
            "export.reuse_unavailable",
            "the retained export could not be claimed",
            error=str(error),
        )
        return False
    return True


def _restore_or_explain(staging: Path, checkpoint: ExportCheckpoint) -> str | None:
    try:
        return _restore(staging, checkpoint)
    except OSError as error:
        return f"restoring it failed: {error}"


def _restore(staging: Path, checkpoint: ExportCheckpoint) -> str | None:
    """Bring every recorded file back to its exported state; describe any failure."""
    backups = _repair_backups(staging)
    for relative, expected in checkpoint.files.items():
        path = staging / relative
        try:
            actual = path.stat()
        except OSError:
            return f"{relative} is missing"
        if expected.sha256 is None:
            if actual.st_size != expected.size or actual.st_mtime_ns != expected.mtime_ns:
                return f"{relative} changed after the export"
            continue
        if actual.st_size == expected.size and sha256_file(path) == expected.sha256:
            continue
        original = next(
            (
                backup
                for backup in backups.get(path.name, [])
                if backup.stat().st_size == expected.size and sha256_file(backup) == expected.sha256
            ),
            None,
        )
        if original is None:
            return f"{relative} changed after the export and no backup of the original exists"
        restored = path.with_name(f".{path.name}.restoring")
        shutil.copyfile(original, restored)
        restored.replace(path)
    return None


def _repair_backups(staging: Path) -> dict[str, list[Path]]:
    """Backups repairs made, by file name (repairs back up flat, by name)."""
    root = staging / MANIFEST_DIRNAME / _REPAIR_BACKUPS
    found: dict[str, list[Path]] = {}
    if not root.is_dir():
        return found
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            found.setdefault(path.name, []).append(path)
    return found


def _remove_unrecorded(staging: Path, checkpoint: ExportCheckpoint) -> None:
    """Delete everything but the exported files and the checkpoint itself."""
    keep = {staging / relative for relative in checkpoint.files}
    keep.add(checkpoint_path(staging))
    for path in sorted(staging.rglob("*"), reverse=True):  # children before parents
        if path in keep:
            continue
        with contextlib.suppress(OSError):
            if path.is_dir() and not path.is_symlink():
                path.rmdir()  # only succeeds once emptied, so kept files' parents stay
            else:
                path.unlink()


def _empty(staging: Path) -> None:
    for child in list(staging.iterdir()) if staging.is_dir() else []:
        with contextlib.suppress(OSError):
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
