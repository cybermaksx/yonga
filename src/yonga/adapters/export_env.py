"""Build, verify and manage the isolated environments exports run in.

See :mod:`yonga.domain.environments` for why they exist. This adapter does the
mechanical part:

* **Spec.** :meth:`ExportEnvironmentManager.spec_for` mirrors the environment
  yonga is running in -- the same optimum-intel, nncf and torch -- and overrides
  only the packages whose installed version breaks a requirement. OpenVINO and
  OpenVINO Tokenizers are pinned to the runtime's own version, because the
  runtime has to read the IR the export writes.
* **Build.** ``python -m venv``, then torch and torchvision together from
  PyTorch's CPU index, then everything else from PyPI. Every step is an argv
  list through the injected :class:`~yonga.adapters.CommandRunner`, so nothing
  here uses a shell and every line of pip's output lands in the job log.
* **Completion.** The ``*-env.json`` marker is written *last*, after the
  environment has been imported and its versions checked against the spec. A
  directory without the marker is an interrupted build: it is never used, and it
  is rebuilt from scratch.
* **Concurrency.** A per-environment lock file stops two jobs building the same
  environment at once; the second waits and then reuses the first's result.

Environments are addressed only through the interpreter at ``bin/python``
(``python -m pip``, ``optimum.commands.optimum_cli`` run as a module), never through
console scripts, whose shebangs would pin an absolute build path. Every such
interpreter runs with ``-P``, so the directory yonga was started from is not put
first on ``sys.path``: a stray ``json.py`` or ``pip/`` there cannot run in its place.
Whatever imports OpenVINO there starts with its telemetry switched off
(:mod:`yonga.privacy`); yonga itself is not installed in these environments.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from yonga import APP_NAME, APP_SLUG
from yonga.application.progress import ProgressKind
from yonga.domain.diagnostics import (
    ConfigurationError,
    FailureCategory,
    PreflightError,
    diagnostic,
)
from yonga.domain.environments import (
    ENVIRONMENT_PACKAGES,
    ESTIMATED_ENVIRONMENT_BYTES,
    REFERENCE_TOOLCHAIN,
    RUNTIME_PACKAGES,
    TORCH_CPU_INDEX,
    TORCH_INDEX_PACKAGES,
    EnvironmentSpec,
    ExportEnvironment,
)
from yonga.privacy import python_code_argv
from yonga.util.versions import installed_version, satisfies

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yonga.adapters import CommandResult, CommandRunner
    from yonga.application.jobs import JobContext
    from yonga.config.paths import Paths
    from yonga.domain.plans import ExporterRequirements

__all__ = [
    "ESTIMATED_ENVIRONMENT_BYTES",
    "MARKER_FILENAME",
    "BaseInterpreter",
    "ExportEnvironmentManager",
    "requirement_specifiers",
]

#: Written last, once the environment is verified. Its absence means "incomplete".
MARKER_FILENAME: str = f"{APP_SLUG}-env.json"

#: ``ExporterRequirements`` field name -> the package it constrains.
_REQUIREMENT_PACKAGES: Mapping[str, str] = {
    "transformers": "transformers",
    "optimum_intel": "optimum-intel",
    "openvino": "openvino",
    "nncf": "nncf",
}

#: Printed by the freshly built interpreter: what it can import, and at what
#: versions. It runs inside the new environment, never in this process.
_VERIFY_SCRIPT = """
import importlib.metadata as md, json
versions = {}
for name in %r:
    try:
        versions[name] = md.version(name)
    except md.PackageNotFoundError:
        versions[name] = None
import optimum.exporters.openvino  # the export entry point must import
print(json.dumps(versions))
"""

_PIP_FLAGS: tuple[str, ...] = ("--disable-pip-version-check", "--no-input")

#: Name prefix of a copy of the bundled interpreter under the environments directory.
_BASE_PREFIX = "python-"

#: Suffix of a copy of the bundled interpreter still being made.
_PARTIAL_SUFFIX = ".partial"


@dataclass(frozen=True)
class BaseInterpreter:
    """A copy of a bundled interpreter that environments are made from."""

    path: Path
    size_bytes: int
    in_use: bool


def requirement_specifiers(requirements: ExporterRequirements) -> dict[str, str]:
    """The package specifiers an architecture's exporter requirements declare."""
    declared: dict[str, str] = {}
    for field_name, package in _REQUIREMENT_PACKAGES.items():
        specifier = getattr(requirements, field_name, None)
        if isinstance(specifier, str) and specifier.strip():
            declared[package] = specifier.strip()
    return declared


class ExportEnvironmentManager:
    """Creates and finds managed export environments under ``paths.envs_dir``."""

    def __init__(
        self,
        paths: Paths,
        runner: CommandRunner,
        *,
        offline: bool = False,
        base_python: Path | None = None,
        torch_index: str = TORCH_CPU_INDEX,
    ) -> None:
        """Bind the manager to its directory and command runner.

        ``base_python`` is the interpreter environments are created from; it
        defaults to the one running yonga, so an environment always matches the
        Python version yonga itself runs on.
        """
        self._paths = paths
        self._runner = runner
        self._offline = offline
        self._explicit_base = base_python is not None
        self._base_python = base_python or Path(sys.executable)
        self._torch_index = torch_index

    @property
    def root(self) -> Path:
        """Where managed environments live."""
        return self._paths.envs_dir

    # --- specification ---------------------------------------------------------

    def spec_for(
        self,
        requirements: ExporterRequirements,
        *,
        installed: Mapping[str, str | None] | None = None,
        reason: str = "",
    ) -> EnvironmentSpec:
        """The environment that satisfies ``requirements`` while changing least.

        Each package keeps its installed version when that version meets the
        requirement (or none is declared), so the environment differs from the
        one yonga is known to export with only where it must. A package that is
        not installed at all takes the declared specifier, or the latest release.

        Without optimum-intel there is no export toolchain to mirror -- a plain
        ``pip install yonga``, or the AppImage -- and a stray torch installed
        for something else is no evidence of what exports. Then everything but
        the :data:`RUNTIME_PACKAGES` comes from :data:`REFERENCE_TOOLCHAIN`,
        still under any declared requirement.
        """
        declared = requirement_specifiers(requirements)
        if installed is None:
            installed = {package: installed_version(package) for package in ENVIRONMENT_PACKAGES}
        no_toolchain = not installed.get("optimum-intel")
        packages: dict[str, str] = {}
        for package in ENVIRONMENT_PACKAGES:
            version = installed.get(package)
            if no_toolchain and package not in RUNTIME_PACKAGES:
                version = None
            specifier = declared.get(package)
            if version and (specifier is None or satisfies(version, specifier)):
                packages[package] = f"=={version}"
            elif specifier is not None:
                packages[package] = specifier
            elif no_toolchain:
                packages[package] = REFERENCE_TOOLCHAIN.get(package, "")
            else:
                packages[package] = ""
        python = f"{sys.version_info.major}.{sys.version_info.minor}"
        return EnvironmentSpec(python=python, packages=packages, reason=reason)

    # --- lookup ----------------------------------------------------------------

    def path_for(self, spec: EnvironmentSpec) -> Path:
        """The directory an environment built from ``spec`` lives in."""
        return self.root / spec.name

    def find(self, spec: EnvironmentSpec) -> ExportEnvironment | None:
        """Return the completed environment for ``spec``, or None if there is none."""
        return self._load(self.path_for(spec))

    def environments(self) -> list[ExportEnvironment]:
        """Every completed environment, newest first. Incomplete builds are skipped."""
        if not self.root.is_dir():
            return []
        found = [
            env
            for path in sorted(self.root.glob("export-*"))
            if path.is_dir() and (env := self._load(path)) is not None
        ]
        return sorted(found, key=lambda env: env.created_at, reverse=True)

    def resolve(self, reference: str) -> ExportEnvironment:
        """The one environment ``reference`` names: its key, an unambiguous prefix of it,
        or its directory name (``export-KEY``, as ``env list`` and the prepare plan show it).

        Raises :class:`~yonga.domain.diagnostics.ConfigurationError` when nothing,
        or more than one environment, matches.
        """
        wanted = reference.strip().removeprefix("export-")
        matches = (
            [env for env in self.environments() if env.key.startswith(wanted)] if wanted else []
        )
        if len(matches) != 1:
            detail = "matches several environments" if matches else "matches no environment"
            raise ConfigurationError(
                diagnostic(
                    "export.env.not_found",
                    FailureCategory.CONFIGURATION_ERROR,
                    "No such export environment",
                    f"{reference!r} {detail}.",
                    remediation=[f"List them with: {APP_NAME} env list"],
                )
            )
        return matches[0]

    def remove(self, reference: str) -> ExportEnvironment:
        """Delete one environment, named as :meth:`resolve` accepts.

        A copy of the bundled interpreter (see :meth:`base_interpreters`) that no
        environment is made from any more goes with it.
        """
        target = self.resolve(reference)
        _remove_owned(target.path, self.root)
        with suppress(OSError):
            (self.root / f"{target.spec.name}.lock").unlink()
        self.prune_base_interpreters()
        return target

    # --- base interpreter copies -------------------------------------------------

    def base_interpreters(self) -> list[BaseInterpreter]:
        """The copies of a bundled interpreter environments are made from.

        Only an AppImage makes them (:meth:`_base_interpreter`). They are not
        environments, but they are disk under :attr:`root` all the same, so
        ``env list`` shows them.
        """
        if not self.root.is_dir():
            return []
        return [
            BaseInterpreter(
                path=path, size_bytes=_tree_size(path), in_use=self._interpreter_in_use(path)
            )
            for path in sorted(self.root.glob(f"{_BASE_PREFIX}*"))
            if _is_base_copy(path)
        ]

    def prune_base_interpreters(self) -> list[Path]:
        """Delete every interpreter copy no environment is made from; return what went.

        A copy is kept while any environment directory -- complete or not --
        records it as its base, and while a build holds its lock: a build takes
        the lock shared from before it creates its virtual environment until it
        ends, so a copy is never pulled from under one. An interrupted copy
        (``.partial``) goes whenever its lock is free. Never raises for a copy it
        cannot remove: this is housekeeping.
        """
        removed: list[Path] = []
        if not self.root.is_dir():
            return removed
        for path in sorted(self.root.glob(f"{_BASE_PREFIX}*")):
            partial = path.name.endswith(_PARTIAL_SUFFIX)
            if not (partial or _is_base_copy(path)) or path.is_symlink() or not path.is_dir():
                continue
            name = path.name.removesuffix(_PARTIAL_SUFFIX)
            with suppress(OSError), (self.root / f"{name}.lock").open("a") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue  # a build is using it, or copying it
                try:
                    if partial or not self._interpreter_in_use(path):
                        _remove_owned(path, self.root)
                        removed.append(path)
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
        return removed

    def _interpreter_in_use(self, copy: Path) -> bool:
        """Whether any environment directory under :attr:`root` is made from ``copy``.

        A virtual environment names its base in ``pyvenv.cfg`` (``home = .../bin``).
        One whose configuration cannot be read counts as using it.
        """
        base = copy.resolve()
        for environment in self.root.glob("export-*"):
            if environment.is_symlink() or not environment.is_dir():
                continue
            config = environment / "pyvenv.cfg"
            if not config.exists():
                continue
            try:
                home = _venv_home(config.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                return True
            if home is not None and Path(home).resolve().is_relative_to(base):
                return True
        return False

    # --- building --------------------------------------------------------------

    def ensure(self, spec: EnvironmentSpec, ctx: JobContext) -> ExportEnvironment:
        """Return the environment for ``spec``, building and verifying it if needed."""
        existing = self.find(spec)
        if existing is not None:
            ctx.logger.info("export.env.reused", "managed export environment reused", key=spec.key)
            return existing
        if self._offline:
            raise PreflightError(
                diagnostic(
                    "export.env.offline",
                    FailureCategory.DOWNLOAD_FAILURE,
                    "The export environment cannot be built offline",
                    "This export needs a managed environment "
                    f"({_summary(spec)}) that has not been built yet, and building "
                    "one downloads packages.",
                    remediation=[
                        "Run once without offline mode to build it; it is reused afterwards.",
                    ],
                )
            )
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock(spec):
            # Another job may have finished the build while this one waited.
            existing = self.find(spec)
            if existing is not None:
                return existing
            return self._build(spec, ctx)

    def _build(self, spec: EnvironmentSpec, ctx: JobContext) -> ExportEnvironment:
        path = self.path_for(spec)
        if path.exists():
            # A directory without the marker is an interrupted build; start clean.
            _remove_owned(path, self.root)
        ctx.emit(
            ProgressKind.STAGE,
            f"Building managed export environment {spec.name} ({_summary(spec)})",
            status="env",
        )
        ctx.logger.info(
            "export.env.build",
            "building managed export environment",
            key=spec.key,
            packages=dict(spec.packages),
        )
        try:
            with self._base_interpreter(ctx) as base:
                environment = self._build_from(base, path, spec, ctx)
        except BaseException:
            # A failed build may leave a copy of the interpreter nothing uses.
            with suppress(OSError):
                self.prune_base_interpreters()
            raise
        _write_marker(path / MARKER_FILENAME, environment)
        ctx.logger.info(
            "export.env.ready",
            "managed export environment built",
            key=spec.key,
            size_bytes=environment.size_bytes,
            installed=environment.installed,
        )
        return environment

    def _build_from(
        self, base: Path, path: Path, spec: EnvironmentSpec, ctx: JobContext
    ) -> ExportEnvironment:
        """Create, install and verify the environment at ``path`` from ``base``."""
        python = path / "bin" / "python"
        steps: Sequence[tuple[str, list[str]]] = (
            ("creating the virtual environment", [str(base), "-P", "-m", "venv", str(path)]),
            (
                "installing torch and torchvision (CPU builds)",
                [
                    str(python),
                    "-P",
                    "-m",
                    "pip",
                    "install",
                    *_PIP_FLAGS,
                    "--index-url",
                    self._torch_index,
                    *(spec.requirement(name) for name in TORCH_INDEX_PACKAGES),
                ],
            ),
            (
                "installing the export packages",
                [
                    str(python),
                    "-P",
                    "-m",
                    "pip",
                    "install",
                    *_PIP_FLAGS,
                    *(
                        spec.requirement(name)
                        for name in ENVIRONMENT_PACKAGES
                        if name not in TORCH_INDEX_PACKAGES
                    ),
                ],
            ),
        )
        try:
            for label, argv in steps:
                ctx.emit(ProgressKind.LOG, f"Export environment: {label}", status="env")
                self._check(self._runner.run(argv, ctx=ctx), spec, label)
            installed = self._verify(python, spec, ctx)
        except BaseException:
            _remove_owned(path, self.root)
            raise
        return ExportEnvironment(
            spec=spec,
            path=path,
            installed=installed,
            created_at=datetime.now(UTC),
            size_bytes=_tree_size(path),
        )

    @contextmanager
    def _base_interpreter(self, ctx: JobContext) -> Iterator[Path]:
        """An interpreter that outlives this process, to create environments from.

        A virtual environment records its base interpreter's directory and
        breaks when that directory goes away. Inside an AppImage, yonga's own
        interpreter lives on a mount that exists only while the AppImage runs,
        at a new path every time, so an environment made from it would work for
        exactly one job. There the bundled interpreter -- without yonga's own
        site-packages -- is copied once under the environments directory, and
        environments are made from the copy.

        The copy's lock is held shared for as long as the caller builds, so
        :meth:`prune_base_interpreters` (which needs it exclusively) cannot
        remove the copy before the new environment records it as its base.
        """
        running = Path(sys.executable)
        appdir = os.environ.get("APPDIR")
        if self._explicit_base:
            yield self._base_python
            return
        if not appdir or not running.resolve().is_relative_to(Path(appdir).resolve()):
            yield running
            return
        prefix = Path(sys.base_prefix)
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        digest = hashlib.sha256(sys.version.encode("utf-8")).hexdigest()[:12]
        release = ".".join(str(part) for part in sys.version_info[:3])
        target = self.root / f"{_BASE_PREFIX}{release}-{digest}"
        interpreter = target / "bin" / f"python{version}"
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / f"{target.name}.lock").open("a") as handle:
            try:
                while True:
                    fcntl.flock(handle, fcntl.LOCK_EX)
                    if not interpreter.is_file():
                        ctx.emit(
                            ProgressKind.LOG,
                            "Export environment: keeping a copy of the bundled Python",
                        )
                        _copy_interpreter(prefix, target, version=version, root=self.root)
                        ctx.logger.info(
                            "export.env.base_copied", "bundled interpreter copied", path=str(target)
                        )
                    # Converting to shared is not atomic: a prune may slip in
                    # between, so the copy is checked again once it is held.
                    fcntl.flock(handle, fcntl.LOCK_SH)
                    if interpreter.is_file():
                        break
                yield interpreter
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _verify(self, python: Path, spec: EnvironmentSpec, ctx: JobContext) -> dict[str, str]:
        """Import the export entry point inside the environment and check every pin."""
        ctx.emit(ProgressKind.LOG, "Export environment: verifying", status="env")
        script = _VERIFY_SCRIPT % (list(ENVIRONMENT_PACKAGES),)
        result = self._runner.run(python_code_argv(python, script), ctx=ctx)
        self._check(result, spec, "verifying the environment")
        versions = _parse_versions(result.stdout_lines)
        problems = (
            ["the environment did not report its package versions"]
            if versions is None
            else [
                f"{name} {versions.get(name) or 'missing'} does not satisfy {specifier}"
                for name, specifier in spec.packages.items()
                if specifier and not satisfies(versions.get(name), specifier)
            ]
        )
        if versions is None or problems:
            raise PreflightError(
                diagnostic(
                    "export.env.verify_failed",
                    FailureCategory.PACKAGE_ABI_MISMATCH,
                    "The export environment does not match its specification",
                    "; ".join(problems),
                    remediation=["Re-run; the incomplete environment has been removed."],
                )
            )
        return {name: version for name, version in versions.items() if version}

    @staticmethod
    def _check(result: CommandResult, spec: EnvironmentSpec, label: str) -> None:
        if result.cancelled:
            raise PreflightError(
                diagnostic(
                    "export.env.cancelled",
                    FailureCategory.USER_CANCELLED,
                    "Building the export environment was cancelled",
                    f"Cancelled while {label}.",
                )
            )
        if result.returncode != 0:
            tail = "\n".join(list(result.stderr_lines)[-15:] or list(result.stdout_lines)[-15:])
            raise PreflightError(
                diagnostic(
                    "export.env.install_failed",
                    FailureCategory.DOWNLOAD_FAILURE,
                    "The export environment could not be built",
                    f"Failed while {label} ({_summary(spec)}); exit status {result.returncode}.",
                    remediation=[
                        "Check network access to PyPI and download.pytorch.org.",
                        f"The full pip output is in the job log: {APP_NAME} logs JOB_ID --raw",
                    ],
                    raw_excerpt=tail,
                )
            )

    # --- internals -------------------------------------------------------------

    def _load(self, path: Path) -> ExportEnvironment | None:
        marker = path / MARKER_FILENAME
        if not marker.is_file() or not (path / "bin" / "python").exists():
            return None
        try:
            environment = ExportEnvironment.model_validate_json(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return environment.model_copy(update={"path": path})

    @contextmanager
    def _lock(self, spec: EnvironmentSpec) -> Iterator[None]:
        lock_path = self.root / f"{spec.name}.lock"
        with lock_path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _summary(spec: EnvironmentSpec) -> str:
    """The pins that make this environment different, for messages."""
    shown = [
        f"{name}{spec.packages[name]}" for name in ("transformers",) if spec.packages.get(name)
    ]
    return ", ".join(shown) or spec.name


def _parse_versions(lines: Sequence[str]) -> dict[str, str | None] | None:
    for line in reversed(list(lines)):
        text = line.strip()
        if text.startswith("{"):
            with suppress(ValueError):
                data = json.loads(text)
                if isinstance(data, dict):
                    return {str(k): (str(v) if v is not None else None) for k, v in data.items()}
    return None


def _is_base_copy(path: Path) -> bool:
    """Whether ``path`` is a finished copy of a bundled interpreter."""
    name = path.name
    return (
        name.startswith(_BASE_PREFIX)
        and not name.endswith((_PARTIAL_SUFFIX, ".lock"))
        and path.is_dir()
        and not path.is_symlink()
    )


def _venv_home(config: str) -> str | None:
    """The ``home`` value of a ``pyvenv.cfg``: the base interpreter's directory."""
    for line in config.splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip().lower() == "home":
            return value.strip() or None
    return None


def _write_marker(path: Path, environment: ExportEnvironment) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(environment.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def _copy_interpreter(prefix: Path, target: Path, *, version: str, root: Path) -> None:
    """Copy a relocatable Python install to ``target``, leaving out its site-packages."""
    partial = target.with_name(f"{target.name}{_PARTIAL_SUFFIX}")
    if partial.exists():
        _remove_owned(partial, root)
    site = prefix / "lib" / f"python{version}" / "site-packages"

    def skip(directory: str, names: list[str]) -> set[str]:
        if Path(directory) == site:
            return set(names)
        return {name for name in names if name == "__pycache__"}

    shutil.copytree(prefix, partial, symlinks=True, ignore=skip)
    partial.rename(target)


def _tree_size(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        with suppress(OSError):
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
    return total


def _remove_owned(path: Path, root: Path) -> None:
    """Delete ``path`` only if it is a real directory directly inside ``root``."""
    if path.is_symlink() or path.parent.resolve() != root.resolve():
        raise ConfigurationError(
            diagnostic(
                "export.env.outside_root",
                FailureCategory.CONFIGURATION_ERROR,
                "Refusing to delete outside the environment store",
                f"{path} is not a managed export environment.",
            )
        )
    shutil.rmtree(path, ignore_errors=True)
