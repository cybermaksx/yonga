"""The ``yonga self`` service: put the running AppImage on PATH, or take it off.

The AppImage is one file the user downloads, and a downloaded file is not a
command: ``yonga`` only works from a shell once a file named ``yonga`` sits in a
directory on ``PATH``. ``yonga self install`` copies the running AppImage to
``~/.local/bin/yonga`` -- the per-user directory the XDG base directory
specification names for executables, which Ubuntu's ``~/.profile`` puts on
``PATH`` at login when it exists -- and ``yonga self uninstall`` removes that
copy again.

The rules are narrow on purpose:

* **Only an AppImage installs itself.** The AppImage runtime sets ``APPIMAGE``
  to the image file and ``APPDIR`` to its mount point. Both must be set, the
  image must be a real AppImage, and the interpreter running this code must live
  inside ``APPDIR``: a pip-installed yonga started from a terminal that another
  AppImage launched inherits that application's ``APPIMAGE``, and copying *it*
  to ``~/.local/bin/yonga`` would be a bad surprise.
* **Only an AppImage is ever replaced or removed.** Whatever already sits at
  ``DIR/yonga`` is looked at with ``lstat`` and its first bytes, never run: an
  ELF file with the type-2 AppImage magic ``AI\\x02`` at offset 8 may be
  replaced (after a question) or removed; a symbolic link, a directory, a pip
  console script or any other file is left exactly as it is. A symbolic link is
  never followed for writing: the new file is written next to it and renamed
  over the *name*, and a link found there is refused before that point anyway.
* **The copy is atomic.** It is written to a temporary file in the same
  directory, flushed to disk, made ``0755`` and renamed over the target, so a
  shell never finds a half-written ``yonga``. What is at the target is checked
  again immediately before the rename.
* **Nothing else on the system changes.** No ``sudo``, no shell startup file is
  edited: when the directory is not on ``PATH`` the user is told the exact line
  to run. Models, logs and caches are left alone by ``uninstall``.

A small receipt, ``<data-dir>/self-install.json``, remembers the version and
SHA-256 of every copy this command made. It is the only way to name the version
of an AppImage without running it, so a later ``self install`` can say which
version it is about to replace. It is advisory: a missing or unreadable receipt
only makes that version "unknown".
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shlex
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from yonga import __version__
from yonga.config.paths import Paths
from yonga.domain.diagnostics import (
    RERUN_PLACEHOLDER,
    ConfigurationError,
    FailureCategory,
    PreflightError,
    YongaError,
    diagnostic,
)

__all__ = [
    "APPIMAGE_MAGIC",
    "APPIMAGE_MAGIC_OFFSET",
    "COMMAND_NAME",
    "ELF_MAGIC",
    "PIPX_INSTALL_HINT",
    "RECEIPT_FILENAME",
    "CommandRelation",
    "InstallAction",
    "InstallKind",
    "InstallPlan",
    "InstallResult",
    "InstallationStatus",
    "PathReport",
    "RunningInstallation",
    "SelfInstallService",
    "Target",
    "TargetKind",
    "UninstallPlan",
    "default_bin_dir",
    "detect_installation",
    "inspect_target",
    "is_appimage",
    "on_path",
]

#: The command name the copy is installed under.
COMMAND_NAME: str = "yonga"

#: Every AppImage is an ELF executable ...
ELF_MAGIC: bytes = b"\x7fELF"

#: ... carrying ``AI`` plus the image type in the ELF identification padding.
APPIMAGE_MAGIC: bytes = b"AI\x02"

#: Where :data:`APPIMAGE_MAGIC` sits: ``e_ident[EI_PAD]``, unused by ELF itself.
APPIMAGE_MAGIC_OFFSET: int = 8

#: The receipt of copies ``self install`` made, under the data directory.
RECEIPT_FILENAME: str = "self-install.json"

#: How a pip user gets a ``yonga`` command on PATH. Printed, never executed.
PIPX_INSTALL_HINT: str = "pipx install 'yonga[openvino,export]'"

#: Free space kept on the target filesystem beyond the copy itself.
SPACE_MARGIN_BYTES: int = 16 * 1024 * 1024

_CHUNK: int = 1 << 20
_RECEIPT_SCHEMA: int = 1


# ======================================================================
# what is running, and what sits at the target
# ======================================================================
class InstallKind(str, Enum):
    """How the running yonga was started."""

    APPIMAGE = "appimage"
    PYTHON = "python"


class TargetKind(str, Enum):
    """What ``DIR/yonga`` is, as far as ``lstat`` and its first bytes tell."""

    MISSING = "missing"
    APPIMAGE = "appimage"
    SYMLINK = "symlink"
    DIRECTORY = "directory"
    FILE = "file"
    """A regular file that is not an AppImage: a pip console script, a wrapper."""
    OTHER = "other"
    """A FIFO, socket or device node."""


def is_appimage(path: Path) -> bool:
    """Whether ``path`` itself (not a link target) is a type-2 AppImage.

    Only the first eleven bytes are read, through a descriptor opened with
    ``O_NOFOLLOW`` and ``O_NONBLOCK``, after ``lstat`` said it is a regular
    file: nothing is executed, no link is followed and no FIFO can block.
    """
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if not stat.S_ISREG(info.st_mode):
        return False
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return False
    try:
        head = os.read(descriptor, APPIMAGE_MAGIC_OFFSET + len(APPIMAGE_MAGIC))
    except OSError:
        return False
    finally:
        os.close(descriptor)
    end = APPIMAGE_MAGIC_OFFSET + len(APPIMAGE_MAGIC)
    return head.startswith(ELF_MAGIC) and head[APPIMAGE_MAGIC_OFFSET:end] == APPIMAGE_MAGIC


@dataclass(frozen=True)
class FileIdentity:
    """Enough of a ``stat`` result to notice that a file was swapped or changed."""

    device: int
    inode: int
    size: int
    mtime_ns: int

    @classmethod
    def of(cls, info: os.stat_result) -> FileIdentity:
        """Capture the identity of one ``stat`` result."""
        return cls(info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


@dataclass(frozen=True)
class Target:
    """What sits at ``DIR/yonga`` when it was looked at."""

    path: Path
    kind: TargetKind
    identity: FileIdentity | None = None
    link_target: str | None = None

    @property
    def size(self) -> int | None:
        """Size in bytes of a regular file, ``None`` for anything else."""
        if self.kind in (TargetKind.APPIMAGE, TargetKind.FILE) and self.identity is not None:
            return self.identity.size
        return None

    @property
    def modified(self) -> datetime | None:
        """Modification time of a regular file."""
        if self.identity is None or self.size is None:
            return None
        return datetime.fromtimestamp(self.identity.mtime_ns / 1e9, tz=UTC)

    def describe(self) -> str:
        """A short phrase naming what this is, for refusals."""
        if self.kind is TargetKind.SYMLINK:
            return f"a symbolic link to {self.link_target}"
        if self.kind is TargetKind.DIRECTORY:
            return "a directory"
        if self.kind is TargetKind.FILE:
            return "a file that is not an AppImage (a pip or pipx command, or a script)"
        if self.kind is TargetKind.OTHER:
            return "a special file (FIFO, socket or device)"
        if self.kind is TargetKind.APPIMAGE:
            return "an AppImage"
        return "nothing"


def inspect_target(path: Path) -> Target:
    """Classify ``path`` without following a symbolic link or running anything."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return Target(path, TargetKind.MISSING)
    identity = FileIdentity.of(info)
    if stat.S_ISLNK(info.st_mode):
        link: str | None = None
        with suppress(OSError):
            link = str(path.readlink())
        return Target(path, TargetKind.SYMLINK, identity, link)
    if stat.S_ISDIR(info.st_mode):
        return Target(path, TargetKind.DIRECTORY, identity)
    if not stat.S_ISREG(info.st_mode):
        return Target(path, TargetKind.OTHER, identity)
    kind = TargetKind.APPIMAGE if is_appimage(path) else TargetKind.FILE
    return Target(path, kind, identity)


@dataclass(frozen=True)
class RunningInstallation:
    """The yonga running this command: an AppImage, or a Python environment."""

    kind: InstallKind
    version: str
    prefix: Path
    """``sys.prefix``: the bundled interpreter inside an AppImage, the venv otherwise."""
    appimage: Path | None = None
    """The ``.AppImage`` file, resolved, when running from one."""
    appdir: Path | None = None
    note: str | None = None
    """Why ``APPIMAGE`` was set but not believed, when it was."""

    def to_json(self) -> dict[str, object]:
        """JSON-safe description."""
        return {
            "kind": self.kind.value,
            "version": self.version,
            "appimage": str(self.appimage) if self.appimage is not None else None,
            "prefix": str(self.prefix),
        }


def _inside(path: str, root: str) -> bool:
    """Whether ``path`` resolves to ``root`` or somewhere below it."""
    try:
        resolved = os.path.realpath(path)
        base = os.path.realpath(root)
        return os.path.commonpath([resolved, base]) == base
    except (OSError, ValueError):
        return False


def detect_installation(
    environ: Mapping[str, str],
    *,
    executable: str,
    prefix: str,
    version: str = __version__,
) -> RunningInstallation:
    """Tell an AppImage run from a Python environment (see the module docstring)."""
    python = RunningInstallation(InstallKind.PYTHON, version, Path(prefix))
    raw = environ.get("APPIMAGE", "")
    if not raw:
        return python
    appdir = environ.get("APPDIR", "")
    image = Path(raw)
    if not image.is_absolute():
        return _not_believed(python, f"APPIMAGE is not an absolute path: {raw}")
    real = Path(os.path.realpath(image))
    if not is_appimage(real):
        return _not_believed(python, f"APPIMAGE names {raw}, which is not an AppImage file.")
    if not appdir or not executable or not _inside(executable, appdir):
        return _not_believed(
            python,
            f"APPIMAGE names {raw}, but this yonga's interpreter ({executable}) is not "
            "inside that image; the variable was inherited from another application.",
        )
    return RunningInstallation(
        InstallKind.APPIMAGE, version, Path(prefix), appimage=real, appdir=Path(appdir)
    )


def _not_believed(python: RunningInstallation, note: str) -> RunningInstallation:
    """A Python installation, with the reason a set ``APPIMAGE`` was ignored."""
    return RunningInstallation(python.kind, python.version, python.prefix, note=note)


# ======================================================================
# PATH
# ======================================================================
def _home(environ: Mapping[str, str]) -> Path:
    """``$HOME`` when it is absolute, the password database otherwise."""
    raw = environ.get("HOME", "")
    if raw and Path(raw).is_absolute():
        return Path(raw)
    return Path.home()


def default_bin_dir(environ: Mapping[str, str]) -> Path:
    """``~/.local/bin``: the XDG per-user directory for executables."""
    return _home(environ) / ".local" / "bin"


def _search_path(environ: Mapping[str, str]) -> str:
    """The ``PATH`` a shell would search, ``os.defpath`` when it is unset."""
    value = environ.get("PATH")
    return os.defpath if value is None else value


def on_path(directory: Path, environ: Mapping[str, str]) -> bool:
    """Whether ``directory`` is on ``PATH``, comparing resolved paths.

    Relative entries (``.``, an empty entry) depend on the current directory at
    the moment a command is typed and do not count.
    """
    wanted = os.path.realpath(directory)
    for entry in _search_path(environ).split(os.pathsep):
        if entry and Path(entry).is_absolute() and os.path.realpath(entry) == wanted:
            return True
    return False


def find_command(environ: Mapping[str, str]) -> Path | None:
    """What typing ``yonga`` in a shell with this ``PATH`` would run."""
    found = shutil.which(COMMAND_NAME, path=_search_path(environ))
    return Path(found) if found else None


def _same_file(left: Path, right: Path) -> bool:
    """Whether two paths name one file."""
    try:
        return left.samefile(right)
    except OSError:
        return False


def _sha256(path: Path) -> str:
    """Hex SHA-256 of a file, following a link (reading is harmless)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def same_content(left: Path, right: Path) -> bool:
    """Same file, or same size and SHA-256."""
    if _same_file(left, right):
        return True
    try:
        if left.stat().st_size != right.stat().st_size:
            return False
        return _sha256(left) == _sha256(right)
    except OSError:
        return False


def _shell_path(directory: Path, home: Path) -> str:
    """``directory`` for a double-quoted shell word, as ``$HOME/...`` when under home."""
    try:
        relative = directory.relative_to(home)
    except ValueError:
        relative = None
    if relative is not None:
        rest = "" if str(relative) == "." else "/" + _escape_double_quoted(str(relative))
        return "$HOME" + rest
    return _escape_double_quoted(str(directory))


def _escape_double_quoted(text: str) -> str:
    """Escape the characters that keep their meaning inside double quotes."""
    for char in ("\\", '"', "$", "`"):
        text = text.replace(char, "\\" + char)
    return text


def _quote(path: Path) -> str:
    """A path quoted for a shell command line."""
    return shlex.quote(str(path))


@dataclass(frozen=True)
class PathReport:
    """Whether the installed command is what typing ``yonga`` runs, and what to do if not."""

    directory: Path
    target: Path
    on_path: bool
    command: Path | None
    """What ``shutil.which('yonga')`` finds with the current ``PATH``."""
    command_is_target: bool
    advice: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, object]:
        """JSON-safe form."""
        return {
            "bin_dir_on_path": self.on_path,
            "path_command": str(self.command) if self.command is not None else None,
            "path_command_is_target": self.command_is_target,
            "advice": list(self.advice),
        }


# ======================================================================
# doctor's view
# ======================================================================
class CommandRelation(str, Enum):
    """How the ``yonga`` found on PATH relates to the running installation."""

    THIS = "this"
    OTHER_APPIMAGE = "other_appimage"
    OTHER = "other"
    MISSING = "missing"


@dataclass(frozen=True)
class InstallationStatus:
    """How yonga is running and what ``yonga`` on PATH resolves to."""

    running: RunningInstallation
    command: Path | None
    relation: CommandRelation
    command_version: str | None = None
    default_bin_dir: Path | None = None

    def to_json(self) -> dict[str, object]:
        """JSON-safe form for ``doctor --json``."""
        return {
            **self.running.to_json(),
            "path_command": str(self.command) if self.command is not None else None,
            "path_command_relation": self.relation.value,
            "path_command_version": self.command_version,
        }


# ======================================================================
# plans and results
# ======================================================================
class InstallAction(str, Enum):
    """What ``self install`` will do."""

    INSTALL = "installed"
    REPLACE = "replaced"
    CURRENT = "already_installed"


@dataclass(frozen=True)
class InstallPlan:
    """Everything decided before a byte is written."""

    source: Path
    source_size: int
    version: str
    directory: Path
    target: Target
    action: InstallAction
    create_directory: bool = False
    previous_version: str | None = None
    previous_sha256: str | None = None


@dataclass(frozen=True)
class InstallResult:
    """What ``self install`` did."""

    plan: InstallPlan
    sha256: str
    created_directory: bool = False
    receipt_written: bool = True

    def to_json(self) -> dict[str, object]:
        """JSON-safe form."""
        plan = self.plan
        previous: dict[str, object] | None = None
        if plan.action is InstallAction.REPLACE:
            previous = {
                "version": plan.previous_version,
                "sha256": plan.previous_sha256,
                "size": plan.target.size,
            }
        return {
            "action": plan.action.value,
            "source": str(plan.source),
            "target": str(plan.target.path),
            "bin_dir": str(plan.directory),
            "version": plan.version,
            "size": plan.source_size,
            "sha256": self.sha256,
            "created_bin_dir": self.created_directory,
            "previous": previous,
        }


@dataclass(frozen=True)
class UninstallPlan:
    """What ``self uninstall`` found at the target."""

    directory: Path
    target: Target
    version: str | None = None
    running_copy: bool = False
    """The file to remove is the AppImage running this command."""


# ======================================================================
# the receipt
# ======================================================================
class InstallReceipts:
    """``<data-dir>/self-install.json``: version and digest of each copy made."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """Location of the receipt."""
        return self._path

    def _load(self) -> dict[str, dict[str, object]]:
        try:
            document = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        installs = document.get("installs") if isinstance(document, dict) else None
        if not isinstance(installs, dict):
            return {}
        return {str(key): value for key, value in installs.items() if isinstance(value, dict)}

    def version_of(self, sha256: str) -> str | None:
        """The version recorded for a copy with this digest, if any."""
        for entry in self._load().values():
            if entry.get("sha256") == sha256 and isinstance(entry.get("version"), str):
                return str(entry["version"])
        return None

    def record(self, target: Path, *, sha256: str, size: int, version: str, source: Path) -> bool:
        """Remember one copy; ``False`` when the receipt could not be written."""
        installs = self._load()
        installs[str(target)] = {
            "sha256": sha256,
            "size": size,
            "version": version,
            "source": str(source),
            "installed_at": datetime.now(UTC).isoformat(),
        }
        return self._save(installs)

    def forget(self, target: Path) -> bool:
        """Drop the entry of a removed copy."""
        installs = self._load()
        if installs.pop(str(target), None) is None:
            return True
        return self._save(installs)

    def _save(self, installs: dict[str, dict[str, object]]) -> bool:
        document = {"schema_version": _RECEIPT_SCHEMA, "installs": installs}
        temporary: str | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=".self-install-", suffix=".json", dir=self._path.parent
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2, sort_keys=True)
                handle.write("\n")
            Path(temporary).replace(self._path)
        except OSError:
            if temporary is not None:
                with suppress(OSError):
                    Path(temporary).unlink()
            return False
        return True


# ======================================================================
# the service
# ======================================================================
class SelfInstallService:
    """Plans and performs ``yonga self install`` and ``yonga self uninstall``."""

    def __init__(
        self,
        paths: Paths,
        *,
        environ: Mapping[str, str] | None = None,
        executable: str | None = None,
        prefix: str | None = None,
        version: str = __version__,
    ) -> None:
        """Bind the service to one environment; everything is read from it lazily."""
        self._paths = paths
        self._environ: Mapping[str, str] = os.environ if environ is None else environ
        self._executable = sys.executable if executable is None else executable
        self._prefix = sys.prefix if prefix is None else prefix
        self._version = version
        self.receipts = InstallReceipts(paths.data_dir / RECEIPT_FILENAME)

    # --- facts ---------------------------------------------------------
    def running(self) -> RunningInstallation:
        """The installation running this command."""
        return detect_installation(
            self._environ,
            executable=self._executable,
            prefix=self._prefix,
            version=self._version,
        )

    def default_bin_dir(self) -> Path:
        """``~/.local/bin`` for this environment's ``HOME``."""
        return default_bin_dir(self._environ)

    def bin_dir(self, option: Path | None) -> Path:
        """``--bin-dir`` made absolute (``~`` expanded), or the default."""
        if option is None:
            return self.default_bin_dir()
        text = str(option)
        if text == "~" or text.startswith("~/"):
            text = str(_home(self._environ)) + text[1:]
        return Path(os.path.normpath(Path(text).absolute()))

    def data_locations(self) -> dict[str, Path]:
        """Where models, logs and caches live: what ``uninstall`` leaves alone."""
        return {
            "data_dir": self._paths.data_dir,
            "cache_dir": self._paths.cache_dir,
            "config_dir": self._paths.config_dir,
        }

    # --- install -------------------------------------------------------
    def plan_install(self, bin_dir: Path | None) -> InstallPlan:
        """Decide what ``self install`` would do, refusing anything unsafe."""
        running = self.running()
        if running.kind is not InstallKind.APPIMAGE or running.appimage is None:
            raise _not_an_appimage(running)
        source = running.appimage
        directory = self.bin_dir(bin_dir)
        create = _check_directory(directory)
        target = inspect_target(directory / COMMAND_NAME)
        try:
            source_size = source.stat().st_size
        except OSError as error:
            raise _io_failure("read", source, error) from error

        previous_version: str | None = None
        previous_sha256: str | None = None
        if target.kind is TargetKind.MISSING:
            action = InstallAction.INSTALL
        elif target.kind is TargetKind.APPIMAGE:
            action = InstallAction.REPLACE
            if _same_file(source, target.path):
                action = InstallAction.CURRENT
            else:
                try:
                    previous_sha256 = _sha256(target.path)
                    if target.size == source_size and _sha256(source) == previous_sha256:
                        action = InstallAction.CURRENT
                except OSError as error:
                    raise _io_failure("read", target.path, error) from error
            if action is InstallAction.REPLACE and previous_sha256 is not None:
                previous_version = self.receipts.version_of(previous_sha256)
        else:
            raise _target_refused(target)

        if action is not InstallAction.CURRENT:
            _check_space(directory, source_size)
        return InstallPlan(
            source=source,
            source_size=source_size,
            version=running.version,
            directory=directory,
            target=target,
            action=action,
            create_directory=create,
            previous_version=previous_version,
            previous_sha256=previous_sha256,
        )

    def install(self, plan: InstallPlan) -> InstallResult:
        """Carry out a plan: copy, flush, chmod 0755, rename over the target."""
        if plan.action is InstallAction.CURRENT:
            try:
                current = _sha256(plan.target.path)
            except OSError as error:
                raise _io_failure("read", plan.target.path, error) from error
            return InstallResult(plan, current, receipt_written=self._record(plan, current))

        created = False
        if plan.create_directory:
            try:
                plan.directory.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                raise _io_failure("create", plan.directory, error) from error
            created = True

        temporary: str | None = None
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{COMMAND_NAME}-", suffix=".part", dir=plan.directory
            )
            digest = hashlib.sha256()
            copied = 0
            with os.fdopen(descriptor, "wb") as output, plan.source.open("rb") as source:
                while block := source.read(_CHUNK):
                    output.write(block)
                    digest.update(block)
                    copied += len(block)
                output.flush()
                os.fchmod(output.fileno(), 0o755)
                os.fsync(output.fileno())
            if copied != plan.source_size:
                raise _changed_during_copy(plan.source)
            _recheck(plan.target)
            Path(temporary).replace(plan.target.path)
            temporary = None
        except OSError as error:
            raise _io_failure("write", plan.target.path, error) from error
        finally:
            if temporary is not None:
                with suppress(OSError):
                    Path(temporary).unlink()
        _fsync_directory(plan.directory)
        sha256 = digest.hexdigest()
        return InstallResult(
            plan, sha256, created_directory=created, receipt_written=self._record(plan, sha256)
        )

    def _record(self, plan: InstallPlan, sha256: str) -> bool:
        return self.receipts.record(
            plan.target.path,
            sha256=sha256,
            size=plan.source_size,
            version=plan.version,
            source=plan.source,
        )

    # --- uninstall -----------------------------------------------------
    def plan_uninstall(self, bin_dir: Path | None) -> UninstallPlan:
        """Look at ``DIR/yonga``; refuse anything that is not an AppImage."""
        directory = self.bin_dir(bin_dir)
        target = inspect_target(directory / COMMAND_NAME)
        if target.kind is TargetKind.MISSING:
            return UninstallPlan(directory, target)
        if target.kind is not TargetKind.APPIMAGE:
            raise _target_refused(target, uninstall=True)
        version: str | None = None
        with suppress(OSError):
            version = self.receipts.version_of(_sha256(target.path))
        running = self.running()
        running_copy = running.appimage is not None and _same_file(running.appimage, target.path)
        if running_copy and version is None:
            version = running.version
        return UninstallPlan(directory, target, version=version, running_copy=running_copy)

    def uninstall(self, plan: UninstallPlan) -> bool:
        """Remove the AppImage the plan found; ``False`` when there was nothing."""
        if plan.target.kind is TargetKind.MISSING:
            return False
        _recheck(plan.target)
        try:
            plan.target.path.unlink()
        except FileNotFoundError:
            return False
        except OSError as error:
            raise _io_failure("remove", plan.target.path, error) from error
        _fsync_directory(plan.directory)
        self.receipts.forget(plan.target.path)
        return True

    # --- PATH ----------------------------------------------------------
    def path_report(self, directory: Path, target: Path, *, created: bool = False) -> PathReport:
        """Whether typing ``yonga`` now runs ``target``, and what to do if not."""
        listed = on_path(directory, self._environ)
        command = find_command(self._environ)
        is_target = command is not None and _same_file(command, target)
        advice: list[str] = []
        if not listed:
            advice.extend(self._path_advice(directory, command, created=created))
        elif not is_target and command is not None:
            advice.extend(
                [
                    f"'yonga' in this shell runs {command}, which comes before {directory} "
                    "on PATH: a different installation.",
                    f"Run {_quote(target)} by its full path, remove or rename {command}, "
                    f"or put {directory} earlier on PATH.",
                ]
            )
        elif not is_target:
            advice.append(
                f"{directory} is on PATH, but 'yonga' was not found in it; "
                "check that the directory is readable and the file executable."
            )
        return PathReport(directory, target, listed, command, is_target, advice)

    def _path_advice(self, directory: Path, command: Path | None, *, created: bool) -> list[str]:
        home = _home(self._environ)
        export = f'export PATH="{_shell_path(directory, home)}:$PATH"'
        if command is None:
            effect = "so 'yonga' is not found yet"
        else:
            effect = f"so 'yonga' still runs {command}, a different installation"
        lines = [f"{directory} is not on PATH in this shell, {effect}."]
        if os.path.realpath(directory) == os.path.realpath(self.default_bin_dir()):
            when = (
                "yonga just created it, so log out and back in once"
                if created
                else "log out and back in once"
            )
            lines.append(
                "Ubuntu's and Debian's ~/.profile add ~/.local/bin to PATH at login, but "
                f"only if the directory already existed then: {when}."
            )
            lines.append(f"For this shell right now: {export}")
            lines.append(
                "If it is still missing after logging in again, add that line to your shell's "
                "startup file (~/.profile, ~/.bashrc or ~/.zshrc) yourself."
            )
        else:
            lines.append(f"For this shell right now: {export}")
            lines.append(
                "To keep it, add that line to your shell's startup file (~/.profile, "
                "~/.bashrc or ~/.zshrc) yourself."
            )
        lines.append("yonga never edits shell startup files.")
        return lines

    # --- doctor --------------------------------------------------------
    def status(self) -> InstallationStatus:
        """How yonga runs and how the ``yonga`` on PATH relates to it."""
        running = self.running()
        command = find_command(self._environ)
        relation = CommandRelation.MISSING
        version: str | None = None
        if command is not None:
            relation = self._relation(running, command)
            if relation is CommandRelation.OTHER_APPIMAGE:
                with suppress(OSError):
                    version = self.receipts.version_of(_sha256(command))
            elif relation is CommandRelation.THIS:
                version = running.version
        return InstallationStatus(running, command, relation, version, self.default_bin_dir())

    def _relation(self, running: RunningInstallation, command: Path) -> CommandRelation:
        real = Path(os.path.realpath(command))
        if running.appimage is not None:
            if same_content(command, running.appimage):
                return CommandRelation.THIS
            return CommandRelation.OTHER_APPIMAGE if is_appimage(real) else CommandRelation.OTHER
        if _inside(str(real), str(running.prefix)) or _shebang_matches(real, self._executable):
            return CommandRelation.THIS
        return CommandRelation.OTHER_APPIMAGE if is_appimage(real) else CommandRelation.OTHER


def _shebang_matches(script: Path, executable: str) -> bool:
    """Whether a console script's ``#!`` names an interpreter beside ``executable``."""
    try:
        with script.open("rb") as handle:
            first = handle.readline(512)
    except OSError:
        return False
    if not first.startswith(b"#!"):
        return False
    words = first[2:].decode("utf-8", "replace").split()
    if not words:
        return False
    try:
        return os.path.realpath(Path(words[0]).parent) == os.path.realpath(Path(executable).parent)
    except (OSError, ValueError):
        return False


# ======================================================================
# checks and failures
# ======================================================================
def _nearest_existing(path: Path) -> Path:
    """``path`` or its closest ancestor that exists."""
    current = path
    while not os.path.lexists(current) and current.parent != current:
        current = current.parent
    return current


def _check_directory(directory: Path) -> bool:
    """Refuse a bin directory that cannot take the copy; ``True`` if it must be made."""
    if os.path.lexists(directory):
        if not directory.is_dir():
            raise ConfigurationError(
                diagnostic(
                    "self.bin_dir.not_a_directory",
                    FailureCategory.CONFIGURATION_ERROR,
                    "Not a directory",
                    f"{directory} exists and is not a directory.",
                    remediation=[
                        "Pass a directory with --bin-dir; ~/.local/bin is the default.",
                    ],
                    context={"bin_dir": str(directory)},
                )
            )
        if not os.access(directory, os.W_OK | os.X_OK):
            raise _not_writable(directory, directory)
        return False
    parent = _nearest_existing(directory)
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise _not_writable(directory, parent)
    return True


def _not_writable(directory: Path, blocking: Path) -> YongaError:
    """The refusal for a directory the user cannot write to."""
    where = (
        f"{directory} is not writable by you"
        if blocking == directory
        else f"{directory} does not exist and cannot be created: {blocking} is not writable by you"
    )
    return YongaError(
        diagnostic(
            "self.bin_dir.not_writable",
            FailureCategory.PERMISSION_DENIED,
            "Directory is not writable",
            f"{where}. yonga installs for your user only and never uses sudo.",
            remediation=[
                "Pick a directory you own: the default ~/.local/bin, or another one on "
                "your PATH, with --bin-dir DIR.",
                "Running yonga with sudo is not needed and not supported for this.",
            ],
            context={"bin_dir": str(directory)},
        )
    )


def _not_an_appimage(running: RunningInstallation) -> ConfigurationError:
    """The refusal outside an AppImage, with how a pip user gets a command instead."""
    script = running.prefix / "bin" / COMMAND_NAME
    if os.path.lexists(script):
        where = f"pip installed its 'yonga' command as {script}"
    else:
        where = f"it has no 'yonga' command in {script.parent}; 'python -m yonga' runs it"
    remediation = [
        f"This yonga runs from the Python environment {running.prefix}: {where}.",
        f"Activate that environment to have it on PATH: source {_quote(running.prefix)}/bin/"
        "activate",
        f"Or install yonga as a command for your user with pipx: {PIPX_INSTALL_HINT}",
        "'self install' is for the AppImage: ./yonga-X.Y.Z-x86_64.AppImage self install",
    ]
    message = "'yonga self install' copies the running AppImage, and this yonga is not one."
    if running.note:
        message = f"{message} {running.note}"
    return ConfigurationError(
        diagnostic(
            "self.not_appimage",
            FailureCategory.CONFIGURATION_ERROR,
            "Not running from an AppImage",
            message,
            remediation=remediation,
            context={"prefix": str(running.prefix)},
        )
    )


def _target_refused(target: Target, *, uninstall: bool = False) -> YongaError:
    """The refusal to touch anything at ``DIR/yonga`` that is not an AppImage."""
    verb = "remove" if uninstall else "replace"
    remediation = [f"{target.path} was left exactly as it is."]
    if target.kind is TargetKind.SYMLINK:
        remediation.append(
            "A link there usually comes from pipx or a manual setup; remove it yourself "
            "if you want the AppImage there instead."
        )
    elif target.kind is TargetKind.FILE:
        remediation.append(
            "If it is a pip or pipx install of yonga, remove that first (pipx uninstall yonga) "
            "if you want the AppImage instead."
        )
    if not uninstall:
        remediation.append("Or install the AppImage elsewhere: --bin-dir DIR")
    return ConfigurationError(
        diagnostic(
            "self.target.not_appimage",
            FailureCategory.CONFIGURATION_ERROR,
            "Something else is installed as yonga",
            f"{target.path} is {target.describe()}. yonga only {verb}s a yonga AppImage "
            "it can recognise there.",
            remediation=remediation,
            context={"target": str(target.path), "kind": target.kind.value},
        )
    )


def _check_space(directory: Path, size: int) -> None:
    """Refuse up front when the copy cannot fit."""
    try:
        free = shutil.disk_usage(_nearest_existing(directory)).free
    except OSError:
        return
    if free >= size + SPACE_MARGIN_BYTES:
        return
    raise PreflightError(
        diagnostic(
            "self.insufficient_disk",
            FailureCategory.INSUFFICIENT_DISK,
            "Not enough disk space",
            f"Copying the AppImage needs {size} bytes on the filesystem of {directory}; "
            f"{free} are free.",
            remediation=["Free some space, or choose another directory with --bin-dir DIR."],
        )
    )


def _recheck(target: Target) -> None:
    """Refuse when ``DIR/yonga`` changed since the plan was made."""
    now = inspect_target(target.path)
    if now.kind is target.kind and now.identity == target.identity:
        return
    raise ConfigurationError(
        diagnostic(
            "self.target.changed",
            FailureCategory.CONFIGURATION_ERROR,
            "The target changed",
            f"{target.path} changed while yonga was working on it; nothing was replaced.",
            remediation=[f"Run the command again: {RERUN_PLACEHOLDER}"],
        )
    )


def _changed_during_copy(source: Path) -> ConfigurationError:
    """The AppImage changed size while it was being copied."""
    return ConfigurationError(
        diagnostic(
            "self.source.changed",
            FailureCategory.CONFIGURATION_ERROR,
            "The AppImage changed while it was copied",
            f"{source} did not have the size it had when the copy started.",
            remediation=["Run the command again once the file is complete."],
        )
    )


def _io_failure(action: str, path: Path, error: OSError) -> YongaError:
    """Turn an ``OSError`` into a typed failure with the right exit code."""
    if error.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
        category = FailureCategory.PERMISSION_DENIED
        remediation = ["Pick a directory you own with --bin-dir DIR; sudo is not needed."]
    elif error.errno in (errno.ENOSPC, errno.EDQUOT):
        category = FailureCategory.INSUFFICIENT_DISK
        remediation = ["Free some space, or choose another directory with --bin-dir DIR."]
    else:
        category = FailureCategory.UNKNOWN_RUNTIME_ERROR
        remediation = ["Re-run with --debug to see the raw error."]
    return YongaError(
        diagnostic(
            f"self.io.{action}",
            category,
            f"Could not {action} {path.name}",
            f"Could not {action} {path}: {error.strerror or error}.",
            remediation=remediation,
            raw_excerpt=f"{type(error).__name__}: {error}",
        ),
        cause=error,
    )


def _fsync_directory(directory: Path) -> None:
    """Make a rename or unlink in ``directory`` durable; best effort."""
    try:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError:
        return
    try:
        with suppress(OSError):
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
