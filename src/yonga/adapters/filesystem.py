"""Filesystem capacity probing, managed-temp policy and atomic directory publication.

Three concerns live here, all of them pure policy over ``os.statvfs`` and
``/proc/self/mounts``:

* :class:`FilesystemAdapter` answers "how much room is there, and what kind of
  filesystem is it?" for any path, existing or not. It implements the
  :class:`yonga.adapters.FilesystemProbe` protocol and never raises: an
  unreadable ``statvfs`` degrades to zeros plus a note, because a preflight
  check that crashes is worse than one that warns.
* :func:`plan_temp` / :func:`plan_disk` turn those facts into the managed-temp
  decision described in ``SPEC.md`` section 10. yonga deliberately does **not**
  use the system temporary directory for conversion work, so a 16 GiB ``/tmp``
  tmpfs is a warning, never an error, and the tool never advises enlarging it.
* :func:`finalize_directory` publishes a staging directory into its final
  location with a same-filesystem rename, refusing to clobber anything that is
  not a tool-owned incomplete job.

Nothing here imports OpenVINO, Torch or the Hugging Face libraries.
"""

from __future__ import annotations

import contextlib
import errno
import os
import shutil
import tempfile
import uuid
from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from yonga.domain.artifacts import manifest_path
from yonga.domain.diagnostics import (
    ConfigurationError,
    Diagnostic,
    FailureCategory,
    Severity,
    ValidationError,
)
from yonga.domain.hardware import FilesystemInfo
from yonga.domain.plans import DiskPlan, TempPlan

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yonga.adapters import FilesystemProbe
    from yonga.config.settings import Settings

__all__ = [
    "INCOMPLETE_MARKER_NAME",
    "MOUNTS_FILES",
    "FilesystemAdapter",
    "FilesystemReading",
    "MountEntry",
    "finalize_directory",
    "format_bytes",
    "is_tool_owned_incomplete",
    "mark_incomplete",
    "plan_disk",
    "plan_temp",
    "required_temp_bytes",
    "system_temp_dir",
]

#: Files consulted, in order, to learn mount points and filesystem types.
MOUNTS_FILES: tuple[Path, ...] = (Path("/proc/self/mounts"), Path("/proc/mounts"))

#: Marker file that makes a directory recognisably a tool-owned incomplete build.
INCOMPLETE_MARKER_NAME: str = ".yonga-incomplete"

#: Prefix of the temporary directory used when a cross-filesystem copy is needed.
_INCOMING_PREFIX: str = ".yonga-incoming-"

#: Prefix of the directory an overwritten target is moved aside to.
_REPLACED_PREFIX: str = ".yonga-replaced-"

_BYTE_UNITS: tuple[str, ...] = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")

#: ``/proc/self/mounts`` escapes these characters in the mount point field.
_MOUNT_ESCAPES: tuple[tuple[str, str], ...] = (
    ("\\011", "\t"),
    ("\\012", "\n"),
    ("\\040", " "),
    ("\\134", "\\"),
)

_STATS_UNAVAILABLE = "filesystem statistics unavailable"


def _error_text(exc: BaseException) -> str:
    """Return the most readable message an OS-level exception carries."""
    strerror = getattr(exc, "strerror", None)
    if isinstance(strerror, str) and strerror:
        return strerror
    return str(exc) or type(exc).__name__


def format_bytes(value: int) -> str:
    """Render a byte count in binary units, e.g. ``17.5 GiB``.

    Used for user-facing notes only; every stored number stays an exact ``int``.
    """
    if value < 0:
        return f"-{format_bytes(-value)}"
    size = float(value)
    for unit in _BYTE_UNITS:
        if size < 1024.0 or unit == _BYTE_UNITS[-1]:
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} {_BYTE_UNITS[-1]}"  # pragma: no cover - loop always returns


def system_temp_dir() -> Path:
    """Return the system temporary directory (``TMPDIR`` or ``/tmp``).

    This is the directory yonga *avoids* for conversion work; it is probed only
    so that ``doctor`` and the preflight report can explain why.
    """
    try:
        return Path(tempfile.gettempdir())
    except (OSError, ValueError):  # pragma: no cover - gettempdir is very robust
        return Path("/tmp")


@dataclass(frozen=True)
class MountEntry:
    """One line of ``/proc/self/mounts``."""

    source: str
    mount_point: Path
    fstype: str
    options: str = ""


@dataclass(frozen=True)
class FilesystemReading:
    """A :class:`FilesystemInfo` together with any degradation notes.

    ``FilesystemInfo`` has nowhere to record *why* a number is zero, so the
    adapter returns the notes alongside it and the planners fold them into the
    plan's ``notes`` list.
    """

    info: FilesystemInfo
    notes: tuple[str, ...] = ()


def _unescape_mount_field(value: str) -> str:
    """Decode the octal escapes the kernel writes into ``/proc/self/mounts``."""
    if "\\" not in value:
        return value
    result = value
    for escape, replacement in _MOUNT_ESCAPES:
        result = result.replace(escape, replacement)
    return result


def _absolute(path: Path) -> Path:
    """Return an absolute form of ``path`` without requiring it to exist."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    try:
        return Path.cwd() / candidate
    except OSError:  # pragma: no cover - deleted working directory
        return candidate


def _nearest_existing(path: Path) -> Path | None:
    """Return ``path`` or its closest existing ancestor, or ``None`` if there is none."""
    current = path
    while True:
        try:
            if current.exists() or current.is_symlink():
                return current
        except OSError:  # pragma: no cover - permission denied on a parent
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent


class FilesystemAdapter:
    """Reads capacity and type facts for the filesystem holding a path.

    Implements :class:`yonga.adapters.FilesystemProbe`. Every method is
    read-only and none of them raise: a failure becomes zeros plus a note so
    that preflight can warn instead of crashing (``docs/TESTING.md`` requires
    "filesystem statistics unavailable -> warning, not crash").
    """

    def __init__(self, *, mounts_files: Sequence[Path] | None = None) -> None:
        """Create an adapter.

        ``mounts_files`` overrides the ``/proc`` files consulted for mount
        points and filesystem types, which is what lets the planner tests
        simulate a tmpfs ``/tmp`` without root.
        """
        self._mounts_files: tuple[Path, ...] = (
            tuple(mounts_files) if mounts_files is not None else MOUNTS_FILES
        )

    # --- mount table ---------------------------------------------------
    def mounts(self) -> list[MountEntry]:
        """Parse the mount table, returning an empty list when it is unreadable."""
        for candidate in self._mounts_files:
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            return _parse_mounts(text)
        return []

    def mount_for(self, path: Path) -> MountEntry | None:
        """Return the longest mount point that contains ``path``.

        Later entries win over earlier ones with the same mount point, which is
        how the kernel reports a directory that has been mounted over.
        """
        return _longest_mount(self.mounts(), path)

    # --- capacity ------------------------------------------------------
    def usage(self, path: Path) -> FilesystemInfo:
        """Return a :class:`FilesystemInfo` for the filesystem holding ``path``.

        ``path`` does not need to exist: the nearest existing ancestor is used,
        which is the normal case for a job temp directory that has not been
        created yet.
        """
        return self.describe(path).info

    def describe(self, path: Path) -> FilesystemReading:
        """Like :meth:`usage`, but also returns notes explaining any missing data."""
        requested = _absolute(path)
        notes: list[str] = []

        existing = _nearest_existing(requested)
        if existing is None:
            notes.append(
                f"{_STATS_UNAVAILABLE} for {requested}: no existing parent directory was found."
            )
            return FilesystemReading(FilesystemInfo(path=requested), tuple(notes))

        if existing != requested:
            notes.append(f"{requested} does not exist yet; reporting the filesystem of {existing}.")

        try:
            resolved = existing.resolve()
        except OSError:  # pragma: no cover - symlink loop
            resolved = existing

        total = free = available = 0
        try:
            stats = os.statvfs(resolved)
        except (OSError, ValueError) as exc:
            notes.append(f"{_STATS_UNAVAILABLE} for {resolved}: {_error_text(exc)}.")
        else:
            block = stats.f_frsize or stats.f_bsize
            total = int(stats.f_blocks) * int(block)
            free = int(stats.f_bfree) * int(block)
            available = int(stats.f_bavail) * int(block)

        mount = _longest_mount(self.mounts(), resolved)
        if mount is None:
            notes.append(
                f"mount table entry for {resolved} not found; the filesystem type is unknown."
            )

        info = FilesystemInfo(
            path=requested,
            mount_point=mount.mount_point if mount is not None else None,
            fstype=mount.fstype if mount is not None else "unknown",
            total_bytes=total,
            free_bytes=free,
            available_bytes=available,
        )
        return FilesystemReading(info, tuple(notes))


def _parse_mounts(text: str) -> list[MountEntry]:
    """Parse the contents of a ``/proc/*/mounts`` file."""
    entries: list[MountEntry] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 3:
            continue
        mount_point = _unescape_mount_field(fields[1])
        if not mount_point:
            continue
        entries.append(
            MountEntry(
                source=_unescape_mount_field(fields[0]),
                mount_point=Path(mount_point),
                fstype=fields[2],
                options=fields[3] if len(fields) > 3 else "",
            )
        )
    return entries


def _longest_mount(entries: Sequence[MountEntry], path: Path) -> MountEntry | None:
    """Return the entry whose mount point is the longest prefix of ``path``."""
    target = _absolute(path)
    best: MountEntry | None = None
    best_length = -1
    for entry in entries:
        try:
            matched = target == entry.mount_point or target.is_relative_to(entry.mount_point)
        except (OSError, ValueError):  # pragma: no cover - exotic mount point
            continue
        if not matched:
            continue
        length = len(entry.mount_point.parts)
        # ``>=`` so a later line shadows an earlier mount of the same directory.
        if length >= best_length:
            best = entry
            best_length = length
    return best


def _usable_free(info: FilesystemInfo) -> int:
    """Free bytes an unprivileged writer may actually use."""
    if info.available_bytes > 0:
        return info.available_bytes
    return info.free_bytes


def _read(probe: FilesystemProbe | None, path: Path) -> FilesystemReading:
    """Read one filesystem through ``probe``, coercing whatever it returns.

    The ``FilesystemProbe`` protocol is declared as returning ``object`` so that
    the domain layer never leaks into the protocol module; this helper restores
    the concrete type and never lets a hostile fake break preflight.
    """
    adapter = probe if probe is not None else FilesystemAdapter()
    if isinstance(adapter, FilesystemAdapter):
        return adapter.describe(path)
    try:
        value = adapter.usage(path)
    except Exception as exc:  # a broken probe must never crash preflight
        return FilesystemReading(
            FilesystemInfo(path=_absolute(path)),
            (f"{_STATS_UNAVAILABLE} for {path}: {exc}.",),
        )
    if isinstance(value, FilesystemInfo):
        return FilesystemReading(value)
    try:
        return FilesystemReading(FilesystemInfo.model_validate(value))
    except Exception:  # the probe returned something that is not a FilesystemInfo
        return FilesystemReading(
            FilesystemInfo(path=_absolute(path)),
            (f"{_STATS_UNAVAILABLE} for {path}: the probe returned an unusable value.",),
        )


def required_temp_bytes(source_weight_bytes: int, settings: Settings) -> int:
    """Return the conservative working-space estimate from ``SPEC.md`` section 10.

    ``max(temp_min_bytes, source_weight_bytes * temp_size_multiplier)`` — a
    policy estimate, not a claim about what the exporter does internally.
    """
    weight = max(0, int(source_weight_bytes))
    scaled = int(weight * settings.temp_size_multiplier)
    return max(int(settings.temp_min_bytes), scaled)


def plan_temp(
    *,
    job_temp_dir: Path,
    source_weight_bytes: int,
    settings: Settings,
    probe: FilesystemProbe | None = None,
) -> TempPlan:
    """Decide whether the managed temporary directory can hold a conversion.

    yonga always converts inside ``job_temp_dir`` (under the configured cache
    directory), never in the system temporary directory. The plan therefore
    reports two independent facts:

    * ``sufficient`` — whether the *managed* filesystem has room for
      ``required_bytes`` plus ``safety_margin_bytes``;
    * ``system_temp_would_fail`` — whether ``/tmp`` alone would have been too
      small, which is informative but never an error, because ``TMPDIR``,
      ``TMP`` and ``TEMP`` are redirected for every child process.

    The notes deliberately never advise enlarging a tmpfs: growing a RAM-backed
    filesystem to hold a 19 GiB model is the wrong fix, and the tool has already
    avoided the problem.
    """
    temp_reading = _read(probe, job_temp_dir)
    system_dir = system_temp_dir()
    system_reading = _read(probe, system_dir)

    required = required_temp_bytes(source_weight_bytes, settings)
    margin = int(settings.temp_safety_margin_bytes)
    needed = required + margin

    temp_info = temp_reading.info
    system_info = system_reading.info
    temp_free = _usable_free(temp_info)
    system_free = _usable_free(system_info)
    system_temp_would_fail = system_free < needed

    notes: list[str] = [
        f"yonga uses a managed disk-backed temporary directory at {job_temp_dir}; "
        "TMPDIR, TMP and TEMP are redirected there for every conversion subprocess, "
        "so the size of the system temporary directory does not limit a conversion.",
        f"Working-space estimate: max({format_bytes(int(settings.temp_min_bytes))}, "
        f"{format_bytes(max(0, int(source_weight_bytes)))} x {settings.temp_size_multiplier:g}) "
        f"= {format_bytes(required)}, plus a {format_bytes(margin)} safety margin.",
    ]
    notes.extend(temp_reading.notes)
    notes.extend(system_reading.notes)

    if system_info.is_tmpfs:
        notes.append(
            f"The system temporary directory {system_dir} is a RAM-backed "
            f"{system_info.fstype} with {format_bytes(system_free)} free. "
            "This is a warning, not an error: yonga does not convert there."
        )
    if system_temp_would_fail:
        notes.append(
            f"A conversion run in {system_dir} would have failed "
            f"({format_bytes(system_free)} free against {format_bytes(needed)} needed); "
            "the managed directory is used instead."
        )

    if temp_free < needed:
        notes.append(
            f"Not enough space for the managed temporary directory: "
            f"{format_bytes(temp_free)} free on "
            f"{temp_info.mount_point or temp_info.path} against "
            f"{format_bytes(needed)} needed."
        )
        notes.append(
            "Free space on that filesystem, or point cache_dir (YONGA_CACHE_DIR) at a larger one."
        )

    return TempPlan(
        job_temp_dir=Path(job_temp_dir),
        temp_filesystem=temp_info,
        system_temp=system_info,
        required_bytes=required,
        safety_margin_bytes=margin,
        system_temp_would_fail=system_temp_would_fail,
        notes=notes,
    )


def plan_disk(
    *,
    job_temp_dir: Path,
    artifact_dir: Path,
    source_weight_bytes: int,
    settings: Settings,
    expected_download_bytes: int = 0,
    estimated_artifact_bytes: int | None = None,
    download_dir: Path | None = None,
    probe: FilesystemProbe | None = None,
) -> DiskPlan:
    """Combine the temp plan with the artifact filesystem and the download size.

    ``expected_download_bytes`` is what the Hugging Face cache still has to
    fetch. When ``download_dir`` resolves to the same mount point as
    ``artifact_dir`` those bytes are charged to the artifact filesystem as well;
    when ``download_dir`` is ``None`` the same filesystem is assumed, which is
    the conservative answer on a single-partition Linux install.

    ``estimated_artifact_bytes`` defaults to ``source_weight_bytes``: an INT4
    artifact is far smaller, but over-reserving only produces an early warning
    whereas under-reserving produces a failed export hours later.
    """
    probe = probe if probe is not None else FilesystemAdapter()
    temp = plan_temp(
        job_temp_dir=job_temp_dir,
        source_weight_bytes=source_weight_bytes,
        settings=settings,
        probe=probe,
    )

    artifact_reading = _read(probe, artifact_dir)
    artifact_info = artifact_reading.info
    artifact_free = _usable_free(artifact_info)

    download = max(0, int(expected_download_bytes))
    artifact_estimate = (
        max(0, int(estimated_artifact_bytes))
        if estimated_artifact_bytes is not None
        else max(0, int(source_weight_bytes))
    )
    margin = int(settings.temp_safety_margin_bytes)

    notes: list[str] = list(artifact_reading.notes)

    shares_filesystem = True
    if download_dir is not None:
        download_reading = _read(probe, download_dir)
        notes.extend(download_reading.notes)
        shares_filesystem = _same_filesystem(download_reading.info, artifact_info)
        if not shares_filesystem:
            notes.append(
                f"Downloads land on {download_reading.info.mount_point or download_dir}, "
                f"a different filesystem from the artifact directory; "
                f"{format_bytes(download)} is not charged to the artifact filesystem."
            )
    elif download > 0:
        notes.append(
            "The download cache is assumed to share the artifact filesystem; "
            "pass download_dir to check it separately."
        )

    charged_download = download if shares_filesystem else 0
    artifact_needed = artifact_estimate + charged_download + margin
    artifact_sufficient = artifact_free >= artifact_needed

    notes.append(
        f"Artifact filesystem {artifact_info.mount_point or artifact_info.path} "
        f"({artifact_info.fstype}) has {format_bytes(artifact_free)} free against "
        f"{format_bytes(artifact_needed)} needed "
        f"({format_bytes(artifact_estimate)} artifact"
        + (f" + {format_bytes(charged_download)} download" if charged_download else "")
        + f" + {format_bytes(margin)} margin)."
    )
    if not artifact_sufficient:
        notes.append(
            "Free space on the artifact filesystem, or point data_dir "
            "(YONGA_DATA_DIR) at a larger one."
        )

    return DiskPlan(
        temp=temp,
        artifact_filesystem=artifact_info,
        expected_download_bytes=download,
        estimated_artifact_bytes=artifact_estimate,
        artifact_sufficient=artifact_sufficient,
        notes=notes,
    )


def _same_filesystem(left: FilesystemInfo, right: FilesystemInfo) -> bool:
    """Whether two readings describe the same mount point."""
    if left.mount_point is None or right.mount_point is None:
        return True
    return left.mount_point == right.mount_point


# --- atomic directory publication ---------------------------------------------------


def mark_incomplete(directory: Path, *, job_id: str | None = None) -> Path:
    """Create the marker that makes ``directory`` a tool-owned incomplete build.

    The marker is what allows :func:`finalize_directory` to replace a leftover
    directory without ``--force``: yonga only ever deletes what it created.
    """
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / INCOMPLETE_MARKER_NAME
    payload = f"job_id={job_id or 'unknown'}\n"
    with contextlib.suppress(OSError):  # a read-only staging area is not fatal
        marker.write_text(payload, encoding="utf-8")
    return marker


def is_tool_owned_incomplete(directory: Path) -> bool:
    """Whether ``directory`` is an incomplete build that yonga itself created.

    True when the directory does not exist, is empty, or carries the
    :data:`INCOMPLETE_MARKER_NAME` marker *and* has no artifact manifest.
    Anything else — a user's own directory, or a finished artifact — is off
    limits without an explicit ``force``.
    """
    path = Path(directory)
    try:
        if not path.exists():
            return True
        if not path.is_dir() or path.is_symlink():
            return False
        if manifest_path(path).exists():
            return False
        if (path / INCOMPLETE_MARKER_NAME).exists():
            return True
        return not any(path.iterdir())
    except OSError:
        return False


def _refusal(final: Path) -> ConfigurationError:
    """Build the typed refusal raised when the destination is not ours to replace."""
    return ConfigurationError(
        Diagnostic(
            code="artifact.finalize.refused",
            category=FailureCategory.CONFIGURATION_ERROR,
            severity=Severity.ERROR,
            title="Refusing to overwrite an existing directory",
            message=(
                f"{final} already exists and is not an incomplete build created by yonga, "
                "so it will not be replaced."
            ),
            remediation=[
                "Re-run with --force to replace the existing directory.",
                "Or choose a different artifact directory.",
            ],
            context={"path": str(final)},
        )
    )


def finalize_directory(
    staging: Path,
    final: Path,
    *,
    force: bool = False,
    notes: MutableSequence[str] | None = None,
) -> Path:
    """Publish ``staging`` as ``final``, atomically where the filesystem allows.

    The rename is a single ``rename(2)`` when both paths live on the same
    filesystem, so an interrupted publication can never leave a half-populated
    artifact directory behind. Across filesystems the content is copied to a
    hidden sibling of ``final`` first and only then renamed into place, which
    keeps the same all-or-nothing guarantee at the cost of a copy; a note
    records that this happened.

    An existing ``final`` is replaced only when it is a tool-owned incomplete
    build (see :func:`is_tool_owned_incomplete`) or when ``force`` is set;
    otherwise a :class:`~yonga.domain.diagnostics.ConfigurationError` is raised
    and nothing is touched. ``notes``, when given, collects human-readable
    remarks about what happened.

    Returns ``final``.
    """
    staging = Path(staging)
    final = Path(final)
    collected: MutableSequence[str] = notes if notes is not None else []

    if not staging.is_dir():
        raise ValidationError(
            Diagnostic(
                code="artifact.finalize.missing_staging",
                category=FailureCategory.UNKNOWN_RUNTIME_ERROR,
                severity=Severity.ERROR,
                title="Staging directory is missing",
                message=f"Nothing to publish: {staging} is not a directory.",
                context={"staging": str(staging)},
            )
        )

    if staging == final:
        _clear_marker(staging)
        collected.append(f"{final} was already in its final location.")
        return final

    if final.exists() or final.is_symlink():
        if not force and not is_tool_owned_incomplete(final):
            raise _refusal(final)
        collected.append(
            f"Replaced an existing {'forced' if force else 'incomplete'} directory at {final}."
        )

    _clear_marker(staging)
    try:
        final.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _publish_failed(staging, final, exc) from exc

    backup: Path | None = None
    if final.exists() or final.is_symlink():
        backup = final.parent / f"{_REPLACED_PREFIX}{final.name}-{uuid.uuid4().hex[:8]}"
        try:
            final.replace(backup)
        except OSError as exc:
            raise _publish_failed(staging, final, exc) from exc

    try:
        _move_directory(staging, final, collected)
    except OSError as exc:
        if backup is not None:
            with contextlib.suppress(OSError):
                backup.replace(final)
        raise _publish_failed(staging, final, exc) from exc

    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)
    return final


def _move_directory(staging: Path, final: Path, collected: MutableSequence[str]) -> None:
    """Rename ``staging`` onto ``final``, copying across filesystems if needed."""
    try:
        staging.replace(final)
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM}:
            raise
        incoming = final.parent / f"{_INCOMING_PREFIX}{final.name}-{uuid.uuid4().hex[:8]}"
        collected.append(
            f"{staging} and {final} are on different filesystems, so the artifact was "
            "copied instead of renamed; the final directory still appears atomically."
        )
        try:
            shutil.copytree(staging, incoming, symlinks=True)
            incoming.replace(final)
        except OSError:
            shutil.rmtree(incoming, ignore_errors=True)
            raise
        shutil.rmtree(staging, ignore_errors=True)


def _clear_marker(directory: Path) -> None:
    """Remove the incomplete marker so a published artifact is not flagged."""
    with contextlib.suppress(OSError):
        (directory / INCOMPLETE_MARKER_NAME).unlink(missing_ok=True)


def _publish_failed(staging: Path, final: Path, exc: OSError) -> ValidationError:
    """Build the typed failure raised when publication itself fails."""
    return ValidationError(
        Diagnostic(
            code="artifact.finalize.failed",
            category=FailureCategory.UNKNOWN_RUNTIME_ERROR,
            severity=Severity.ERROR,
            title="Artifact could not be published",
            message=f"Could not move {staging} to {final}: {_error_text(exc)}.",
            context={"staging": str(staging), "final": str(final)},
        ),
        cause=exc,
    )
