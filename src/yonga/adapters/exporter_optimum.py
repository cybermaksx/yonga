"""Optimum Intel export adapter: ``optimum-cli export openvino`` behind the Exporter port.

yonga orchestrates conversion, it does not implement it. This adapter is the only
place in the codebase that knows how Optimum Intel spells its command line flags;
the domain layer describes *what* it wants through
:class:`~yonga.domain.plans.QuantizationSettings` and this module translates that
into argv.

Design rules enforced here:

* the command is always an argv **list** executed with ``shell=False`` through an
  injected :class:`~yonga.adapters.CommandRunner`;
* ``optimum`` is never imported at module scope — only
  :func:`importlib.util.find_spec` is consulted, inside a function, and a missing
  exporter becomes a typed :class:`~yonga.domain.diagnostics.Diagnostic`;
* conversion writes into the job's staging directory and nowhere else, with
  ``TMPDIR``/``TMP``/``TEMP`` pointed at the managed disk-backed job temp
  (``SPEC.md`` section 10) so a small ``/tmp`` tmpfs cannot kill a 19 GB export;
* :meth:`OptimumExporter.export` never raises on a failed conversion. It returns
  an :class:`~yonga.domain.plans.ExportResult` carrying a default diagnostic that
  the registry classifier is free to refine;
* nothing captured — argv, environment overrides, output tails — leaves this
  module without passing through :mod:`yonga.util.redaction`.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from yonga.application.progress import ProgressKind
from yonga.domain.diagnostics import (
    ConfigurationError,
    Diagnostic,
    FailureCategory,
    Severity,
    diagnostic,
)
from yonga.domain.plans import (
    DeclaredRequirement,
    ExporterRequirements,
    ExportRequest,
    ExportResult,
    QuantizationSettings,
    export_extra_arg_problems,
)
from yonga.util.redaction import redact_argv, redact_env, redact_text
from yonga.util.versions import installed_version, satisfies

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yonga.adapters import CommandRunner
    from yonga.application.jobs import JobContext

__all__ = [
    "DISK_EXHAUSTION_SIGNATURES",
    "EXPORT_PACKAGES",
    "OPTIMUM_CLI_EXECUTABLE",
    "OPTIMUM_CLI_MODULE",
    "SUPPORTED_WEIGHT_FORMATS",
    "TAIL_LINES",
    "OptimumExporter",
    "build_export_argv",
    "check_exporter_requirements",
    "is_disk_exhaustion",
    "quantization_flags",
    "resolve_launcher",
]

#: Module path of the Optimum CLI entry point. Running it with ``python -m`` keeps
#: the export inside the interpreter yonga itself runs in (verified on this host).
OPTIMUM_CLI_MODULE: str = "optimum.commands.optimum_cli"

#: Console-script fallback used when the module above cannot be located.
OPTIMUM_CLI_EXECUTABLE: str = "optimum-cli"

#: Distributions whose versions are recorded with every export, for reproducibility.
EXPORT_PACKAGES: tuple[str, ...] = (
    "openvino",
    "optimum",
    "optimum-intel",
    "nncf",
    "transformers",
    "torch",
)

#: Weight formats accepted by ``optimum-cli export openvino`` (optimum-intel 2.2).
#: Informational only: the registry owns profile data, and a newer Optimum may add
#: formats, so an unknown value is still passed through rather than rejected here.
SUPPORTED_WEIGHT_FORMATS: frozenset[str] = frozenset(
    {"fp32", "fp16", "int8", "int4", "mxfp4", "nf4", "cb4"}
)

#: Lower-cased substrings that mean the conversion ran out of writable space.
DISK_EXHAUSTION_SIGNATURES: tuple[str, ...] = (
    "no space left on device",
    "basic_ios::clear: iostream error",
    "[errno 28]",
    "errno 28",
    "disk quota exceeded",
    "not enough free space",
)

#: How many trailing output lines are kept on an :class:`ExportResult`.
TAIL_LINES: int = 40

_INSTALL_EXPORT_EXTRA = 'pip install "yonga[export]"'
_INSTALL_OPTIMUM = 'pip install "optimum-intel[openvino]" nncf'


def _format_ratio(value: float) -> str:
    """Render a quantization ratio so ``1.0`` stays ``"1.0"`` and never becomes ``"1"``."""
    text = f"{float(value):.6f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text


def quantization_flags(settings: QuantizationSettings) -> list[str]:
    """Translate quantization settings into ``optimum-cli export openvino`` flags.

    Flag semantics that matter (verified against ``--help`` on this host):

    * ``--sym`` is emitted **only** when ``symmetric`` is ``True``; omitting it is
      how Optimum Intel expresses asymmetric integer weight compression, so
      ``symmetric=False`` and ``symmetric=None`` both produce no flag;
    * ``--group-size`` is emitted in the ``--group-size=<n>`` equals form so the
      per-column value ``-1`` can never be mistaken for another option.
    """
    flags: list[str] = []
    weight_format = settings.weight_format.strip()
    if weight_format:
        flags += ["--weight-format", weight_format]
    if settings.symmetric is True:
        flags.append("--sym")
    if settings.ratio is not None:
        flags += ["--ratio", _format_ratio(settings.ratio)]
    if settings.group_size is not None:
        flags.append(f"--group-size={int(settings.group_size)}")
    if settings.backup_precision:
        flags += ["--backup-precision", settings.backup_precision.strip()]
    if settings.dataset:
        flags += ["--dataset", settings.dataset.strip()]
    if settings.all_layers:
        flags.append("--all-layers")
    if settings.awq:
        flags.append("--awq")
    if settings.scale_estimation:
        flags.append("--scale-estimation")
    return flags


def build_export_argv(request: ExportRequest, *, launcher: Sequence[str]) -> list[str]:
    """Build the full export argv for ``request``, prefixed by ``launcher``.

    The output directory is always the **last** positional argument, matching the
    ``optimum-cli export openvino ... OUTPUT`` signature. ``--model`` receives a
    local snapshot path, so the exporter never re-downloads anything.

    The profile's ``export_extra_args`` are checked again here, not only when the
    registry is loaded, so a profile built in code cannot slip ``--trust-remote-code``,
    another ``--model`` or a ``--`` past the request either: any of them raises
    :class:`ConfigurationError`.
    """
    extras = [str(extra) for extra in request.profile.export_extra_args]
    problems = export_extra_arg_problems(extras)
    if problems:
        raise ConfigurationError(
            diagnostic(
                "export.reserved_extra_args",
                FailureCategory.CONFIGURATION_ERROR,
                "The conversion profile passes export arguments yonga reserves",
                f"Profile {request.profile.id}: " + "; ".join(problems) + ".",
                remediation=[
                    "Remove those arguments from the profile's export_extra_args.",
                    "Remote code is enabled only by --trust-remote-code on the command line.",
                ],
                context={"profile": request.profile.id},
            )
        )
    argv: list[str] = [*launcher, "export", "openvino", "--model", request.resolved_model_ref]
    if request.task:
        argv += ["--task", request.task]
    argv += quantization_flags(request.profile.quantization)
    if request.trust_remote_code:
        argv.append("--trust-remote-code")
    argv += extras
    argv.append(str(request.output_path))
    return argv


def _module_importable(name: str) -> bool:
    """Return whether ``name`` can be located without importing it into yonga's namespace."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def missing_exporter_diagnostic() -> Diagnostic:
    """Diagnostic for a host where neither the Optimum module nor its script exists."""
    return diagnostic(
        "export.optimum_cli_missing",
        FailureCategory.CONFIGURATION_ERROR,
        "Optimum Intel exporter is not installed",
        (
            "Model conversion needs Optimum Intel with its OpenVINO extra, but neither "
            f"the '{OPTIMUM_CLI_MODULE}' module nor an '{OPTIMUM_CLI_EXECUTABLE}' "
            "executable could be found for this Python interpreter."
        ),
        remediation=[
            f"Install the export extra: {_INSTALL_EXPORT_EXTRA}",
            f"Or install the upstream packages directly: {_INSTALL_OPTIMUM}",
            "Then re-run `yonga doctor` to confirm the exporter is visible.",
        ],
        environment_broken=True,
        context={"interpreter": sys.executable},
    )


def resolve_launcher() -> list[str]:
    """Return the argv prefix that runs the Optimum CLI, or raise ``ConfigurationError``.

    ``[sys.executable, "-P", "-m", "optimum.commands.optimum_cli"]`` is preferred
    because it pins the export to the interpreter yonga is running in — no ``PATH``
    games, no second virtual environment. ``-P`` keeps the current directory off
    ``sys.path``, so a module of the same name there cannot stand in for a real one.
    ``shutil.which("optimum-cli")`` is the fallback.
    """
    if _module_importable(OPTIMUM_CLI_MODULE):
        return [sys.executable, "-P", "-m", OPTIMUM_CLI_MODULE]
    executable = shutil.which(OPTIMUM_CLI_EXECUTABLE)
    if executable:
        return [executable]
    raise ConfigurationError(missing_exporter_diagnostic())


def is_disk_exhaustion(text: str) -> bool:
    """Return whether captured output carries the classic "out of working space" signature.

    ``basic_ios::clear: iostream error`` is what a failed C++ IR write looks like
    when the filesystem underneath ran out of room, which is exactly what happens
    when a multi-gigabyte conversion lands in a small ``/tmp`` tmpfs.
    """
    lowered = text.lower()
    return any(signature in lowered for signature in DISK_EXHAUSTION_SIGNATURES)


def check_exporter_requirements(
    requirements: ExporterRequirements,
    *,
    versions: Mapping[str, str | None],
) -> ExporterRequirements:
    """Compare declared version specifiers against installed versions.

    Returns a new :class:`ExporterRequirements` with ``satisfied`` and ``problems``
    filled in. yonga never installs or downgrades anything on the user's behalf in
    the MVP, so every problem names both the required specifier and what is
    actually installed and leaves the decision to the user.
    """
    problems: list[str] = list(requirements.problems)
    for name, specifier in requirements.specifiers().items():
        installed = versions.get(name)
        if installed is None:
            message = (
                f"{name} {specifier} is required to export this model, but {name} is not installed."
            )
        elif satisfies(installed, specifier):
            continue
        else:
            message = (
                f"{name} {specifier} is required to export this model, "
                f"but {installed} is installed. yonga does not change your environment "
                f"automatically — install the required version yourself, e.g. `pip install "
                f'"{name}{specifier}"`.'
            )
        if message not in problems:
            problems.append(message)
    payload = dict(requirements.model_dump())
    payload["problems"] = problems
    payload["satisfied"] = not problems
    return ExporterRequirements.model_validate(payload)


class OptimumExporter:
    """Runs ``optimum-cli export openvino`` as a supervised subprocess.

    The concrete :class:`~yonga.adapters.CommandRunner` is injected so tests can
    substitute a fake process, and so a future implementation can move to the
    in-process Optimum Python API behind the same ``Exporter`` port without any
    caller noticing.
    """

    def __init__(
        self,
        runner: CommandRunner | None = None,
        *,
        launcher: Sequence[str] | None = None,
        timeout: float | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        """Create an exporter.

        Args:
            runner: Command runner to execute through. Defaults to the real
                :class:`~yonga.adapters.subprocess.SubprocessRunner`.
            launcher: Explicit argv prefix, overriding launcher discovery. Only
                tests and advanced configuration should need this.
            timeout: Wall-clock limit handed to the runner, or ``None`` for none.
                Conversions legitimately take tens of minutes, so there is no
                default deadline.
        """
        self._runner = runner if runner is not None else _default_runner()
        self._launcher = list(launcher) if launcher is not None else None
        self._timeout = timeout
        self._cache_dir = cache_dir

    # --- argv ----------------------------------------------------------

    def resolve_launcher(self, request: ExportRequest | None = None) -> list[str]:
        """Return the launcher for ``request``. Raises ``ConfigurationError``.

        Precedence: a launcher given to the constructor (a test seam) wins, then
        the request's managed-environment interpreter, then discovery in the
        interpreter yonga runs in.
        """
        if self._launcher is not None:
            return list(self._launcher)
        if request is not None and request.interpreter is not None:
            return [str(request.interpreter), "-P", "-m", OPTIMUM_CLI_MODULE]
        return resolve_launcher()

    def build_argv(self, request: ExportRequest) -> list[str]:
        """Translate an :class:`ExportRequest` into the exact argv to execute."""
        return build_export_argv(request, launcher=self.resolve_launcher(request))

    # --- requirements --------------------------------------------------

    def toolchain_available(self) -> bool:
        """Whether the Optimum CLI can be launched from here: configured, importable or on PATH."""
        if self._launcher is not None:
            return True
        try:
            resolve_launcher()
        except ConfigurationError:
            return False
        return True

    def requirements(self) -> dict[str, str | None]:
        """Return installed versions of every package the export depends on."""
        return {name: installed_version(name) for name in EXPORT_PACKAGES}

    def declared_requirements(
        self, model_type: str, task: str | None
    ) -> DeclaredRequirement | None:
        """The Transformers range optimum-intel itself declares for ``model_type``.

        Each optimum-intel export config carries ``MIN_TRANSFORMERS_VERSION`` and
        often ``MAX_TRANSFORMERS_VERSION``, and the exporter refuses to run
        outside them ("The current version of Transformers does not allow for
        the export of the model. Maximum required is 5.0"). Reading the range
        here lets preflight know before the download, and lets a managed
        export environment satisfy it, for any architecture and without a
        registry entry per model. The registry's own pin, where one exists,
        still wins.

        The lookup imports optimum and torch, which takes seconds, so it runs in
        a child process and its answer is cached per optimum-intel version.
        ``None`` when optimum-intel is absent or declares nothing usable.
        """
        version = installed_version("optimum-intel")
        if version is None:
            return None
        return self.declared_requirements_in(Path(sys.executable), version, model_type, task)

    def declared_requirements_in(
        self, python: Path, optimum_intel: str, model_type: str, task: str | None
    ) -> DeclaredRequirement | None:
        """:meth:`declared_requirements`, asked of the optimum-intel under ``python``.

        The answer depends only on the optimum-intel version, so it shares the
        same cache whichever interpreter gave it.
        """
        version = optimum_intel
        if not model_type:
            return None
        key = f"{version}|{model_type}|{task or ''}"
        cached = self._read_bounds_cache()
        entry = cached.get(key)
        if entry is None:
            entry = self._lookup_bounds(python, model_type, task)
            if entry is None:
                return None
            cached[key] = entry
            self._write_bounds_cache(cached)
        low, high, config = entry.get("min"), entry.get("max"), entry.get("config")
        parts = [f">={low}" if low else "", f"<={high}" if high else ""]
        specifier = ",".join(part for part in parts if part)
        if not specifier:
            return None
        return DeclaredRequirement(
            transformers=specifier,
            source=f"optimum-intel {version} {config or 'export config'} for {model_type}",
        )

    def _lookup_bounds(
        self, python: Path, model_type: str, task: str | None
    ) -> dict[str, str | None] | None:
        script = (
            "import json, sys\n"
            "import optimum.exporters.openvino\n"
            "from optimum.exporters.tasks import TasksManager\n"
            "model_type, task = sys.argv[1], sys.argv[2] or 'text-generation-with-past'\n"
            "ctor = TasksManager.get_exporter_config_constructor(\n"
            "    exporter='openvino', model_type=model_type, task=task,\n"
            "    library_name='transformers')\n"
            "cls = getattr(ctor, 'func', ctor)\n"
            "def text(value):\n"
            "    return None if value is None else str(getattr(value, 'base_version', value))\n"
            "print(json.dumps({'min': text(getattr(cls, 'MIN_TRANSFORMERS_VERSION', None)),\n"
            "                  'max': text(getattr(cls, 'MAX_TRANSFORMERS_VERSION', None)),\n"
            "                  'config': cls.__name__}))\n"
        )
        result = self._runner.run([str(python), "-P", "-c", script, model_type, task or ""])
        if result.returncode != 0:
            return None
        for line in reversed(list(result.stdout_lines)):
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                return {key: parsed.get(key) for key in ("min", "max", "config")}
        return None

    def _bounds_cache_path(self) -> Path | None:
        return None if self._cache_dir is None else self._cache_dir / "exporter-bounds.json"

    def _read_bounds_cache(self) -> dict[str, dict[str, str | None]]:
        path = self._bounds_cache_path()
        if path is None:
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_bounds_cache(self, data: dict[str, dict[str, str | None]]) -> None:
        path = self._bounds_cache_path()
        if path is None:
            return
        with contextlib.suppress(OSError):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def check_requirements(self, req: ExporterRequirements) -> ExporterRequirements:
        """Fill ``satisfied``/``problems`` on ``req`` from this host's installed versions."""
        return check_exporter_requirements(req, versions=self.requirements())

    # --- export --------------------------------------------------------

    def export(self, request: ExportRequest, ctx: JobContext) -> ExportResult:
        """Convert and quantize ``request.source_path`` into ``request.output_path``.

        Never raises for a failed conversion: a non-zero exit, a cancellation or a
        refusal to start all come back as an :class:`ExportResult` whose
        ``success`` is ``False`` and whose ``diagnostic`` explains why.
        """
        # In a managed environment the versions that matter are the environment's,
        # not the ones installed alongside yonga.
        package_versions = dict(request.environment_versions) or {
            name: version for name, version in self.requirements().items() if version is not None
        }

        refusal = self._precheck(request, ctx)
        if refusal is not None:
            return self._refused(refusal, package_versions=package_versions, ctx=ctx)

        try:
            argv = self.build_argv(request)
        except ConfigurationError as error:
            return self._refused(
                error.diagnostic, package_versions=package_versions, ctx=ctx, argv=[]
            )

        # Only the differences are handed to the runner. The runner merges them
        # onto ``ctx.child_env()`` itself, so the child sees exactly the same
        # environment either way — but nothing inherited from this process is
        # passed along to be written into the job log.
        env = _environment_overrides(self._build_env(request, ctx))
        overrides = redact_env(env)

        ctx.logger.argv(
            "export.command",
            argv,
            profile=request.profile.id,
            output_dir=str(request.output_path),
            temp_dir=str(ctx.temp_dir),
        )
        ctx.emit(
            ProgressKind.EXPORT,
            f"Converting with profile {request.profile.id}",
            status="started",
            profile=request.profile.id,
        )

        cwd = ctx.temp_dir if ctx.temp_dir.is_dir() else None
        result = self._runner.run(argv, env=env, cwd=cwd, timeout=self._timeout, ctx=ctx)

        stdout_tail = _tail(result.stdout_lines)
        stderr_tail = _tail(result.stderr_lines)
        cancelled = bool(result.cancelled)
        success = result.returncode == 0 and not cancelled

        failure: Diagnostic | None = None
        if not success:
            failure = self._failure_diagnostic(
                request=request,
                ctx=ctx,
                returncode=result.returncode,
                cancelled=cancelled,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
            )

        ctx.emit(
            ProgressKind.EXPORT,
            "Conversion finished" if success else "Conversion failed",
            status="ok" if success else "failed",
            severity=Severity.INFO if success else Severity.ERROR,
            diagnostic=failure,
            profile=request.profile.id,
        )

        return ExportResult(
            success=success,
            argv=redact_argv(result.argv),
            env_overrides=overrides,
            returncode=result.returncode,
            duration_seconds=float(result.duration_seconds),
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            log_path=ctx.logger.path,
            package_versions=package_versions,
            diagnostic=failure,
        )

    # --- internals -----------------------------------------------------

    def _precheck(self, request: ExportRequest, ctx: JobContext) -> Diagnostic | None:
        """Validate the request and prepare the output directory; return a refusal or ``None``."""
        if ctx.check_cancelled():
            return self._cancelled_diagnostic(request, ctx)
        containment = _staging_violation(request.output_path, ctx)
        if containment is not None:
            return containment
        return _prepare_output_dir(request.output_path, ctx)

    def _build_env(self, request: ExportRequest, ctx: JobContext) -> dict[str, str]:
        """Build the subprocess environment, with the managed temp always winning."""
        env = ctx.child_env()
        env.update(request.extra_env)
        # SPEC.md section 10: conversion intermediates must land on the managed,
        # disk-backed temp, so nothing a caller passes may redirect them back to
        # a RAM-backed /tmp.
        env.update(ctx.temp_env)
        if ctx.settings.offline:
            env["HF_HUB_OFFLINE"] = "1"
        return env

    def _refused(
        self,
        failure: Diagnostic,
        *,
        package_versions: dict[str, str],
        ctx: JobContext,
        argv: list[str] | None = None,
    ) -> ExportResult:
        """Build the result for an export that never started."""
        ctx.logger.error("export.refused", failure.message, code=failure.code)
        ctx.emit(
            ProgressKind.EXPORT,
            failure.title,
            status="failed",
            severity=Severity.ERROR,
            diagnostic=failure,
        )
        return ExportResult(
            success=False,
            argv=redact_argv(argv or []),
            returncode=-1,
            duration_seconds=0.0,
            log_path=ctx.logger.path,
            package_versions=package_versions,
            diagnostic=failure,
        )

    def _cancelled_diagnostic(self, request: ExportRequest, ctx: JobContext) -> Diagnostic:
        """Diagnostic for a conversion the user interrupted."""
        return diagnostic(
            "export.cancelled",
            FailureCategory.USER_CANCELLED,
            "Conversion cancelled",
            "The conversion was cancelled before it finished. No artifact was produced.",
            remediation=["Re-run the same `yonga prepare` command to start the conversion again."],
            log_path=ctx.logger.path,
            job_id=ctx.job_id,
            context={"profile": request.profile.id},
        )

    def _failure_diagnostic(
        self,
        *,
        request: ExportRequest,
        ctx: JobContext,
        returncode: int,
        cancelled: bool,
        stdout_tail: Sequence[str],
        stderr_tail: Sequence[str],
    ) -> Diagnostic:
        """Attach a reasonable default diagnostic; the registry classifier may refine it."""
        if cancelled:
            return self._cancelled_diagnostic(request, ctx)

        excerpt = "\n".join([*stdout_tail, *stderr_tail]).strip()
        context = {
            "profile": request.profile.id,
            "returncode": str(returncode),
            "output_dir": str(request.output_path),
            "temp_dir": str(ctx.temp_dir),
        }

        if is_disk_exhaustion(excerpt):
            return diagnostic(
                "export.insufficient_disk",
                FailureCategory.INSUFFICIENT_DISK,
                "Conversion ran out of working space",
                (
                    "Conversion could not write its temporary model data.\n"
                    "The system temporary directory is normally a small RAM-backed tmpfs "
                    "(16 GiB on a typical desktop), which cannot hold the intermediates of a "
                    "multi-gigabyte model.\n"
                    f"yonga already points the conversion at its disk-backed cache temp "
                    f"({ctx.temp_dir}); that filesystem is the one that needs free space."
                ),
                remediation=[
                    f"Free space on the filesystem holding {ctx.temp_dir}.",
                    "Run `yonga doctor` to see filesystem types and free space.",
                    "Do not enlarge /tmp: yonga does not convert there.",
                ],
                raw_excerpt=excerpt or None,
                log_path=ctx.logger.path,
                job_id=ctx.job_id,
                context=context,
            )

        return diagnostic(
            "export.failed",
            FailureCategory.UNKNOWN_RUNTIME_ERROR,
            "Conversion failed",
            (
                f"The Optimum Intel exporter exited with status {returncode} while converting "
                f"with profile {request.profile.id}. The full command output is in the job log."
            ),
            remediation=[
                f"Inspect the job log: {ctx.logger.path}",
                "Re-run with `--debug` to see the raw exporter output.",
            ],
            raw_excerpt=excerpt or None,
            log_path=ctx.logger.path,
            job_id=ctx.job_id,
            context=context,
        )


def _default_runner() -> CommandRunner:
    """Instantiate the real subprocess runner, imported lazily to keep imports light."""
    from yonga.adapters.subprocess import SubprocessRunner
    from yonga.util.redaction import environment_secrets

    runner: CommandRunner = SubprocessRunner(extra_secrets=environment_secrets())
    return runner


def _tail(lines: Sequence[str], limit: int = TAIL_LINES) -> list[str]:
    """Return the last ``limit`` output lines, redacted and newline-stripped."""
    selected = list(lines)[-limit:] if limit > 0 else []
    return [redact_text(str(line).rstrip("\n")) for line in selected]


def _environment_overrides(env: Mapping[str, str]) -> dict[str, str]:
    """Return only the variables ``env`` adds to or changes from the current process."""
    base = os.environ
    return {key: value for key, value in env.items() if base.get(key) != value}


def _resolve(path: Path) -> Path:
    """Resolve a path without requiring it to exist, tolerating unreadable parents."""
    try:
        return path.expanduser().resolve()
    except OSError:  # pragma: no cover - only on a pathological filesystem
        return path.expanduser().absolute()


def _staging_violation(output_path: Path, ctx: JobContext) -> Diagnostic | None:
    """Refuse to export anywhere except inside this job's staging directory."""
    output = _resolve(output_path)
    staging = _resolve(ctx.staging_dir)
    if output == staging or output.is_relative_to(staging):
        return None
    return diagnostic(
        "export.output_not_staged",
        FailureCategory.CONFIGURATION_ERROR,
        "Conversion output directory is outside the job staging area",
        (
            "Conversion may only write into the tool-owned staging directory for this job. "
            f"The requested output directory {output} is not inside {staging}."
        ),
        remediation=[
            "This is an internal error; please report it with the job log.",
            f"Expected an output directory under {staging}.",
        ],
        log_path=ctx.logger.path,
        job_id=ctx.job_id,
        context={"output_dir": str(output), "staging_dir": str(staging)},
    )


def _prepare_output_dir(output_path: Path, ctx: JobContext) -> Diagnostic | None:
    """Create an empty output directory, clearing only tool-owned staging leftovers."""
    output = _resolve(output_path)
    try:
        if output.exists():
            if not output.is_dir():
                return diagnostic(
                    "export.output_not_a_directory",
                    FailureCategory.CONFIGURATION_ERROR,
                    "Conversion output path is not a directory",
                    f"{output} already exists and is not a directory.",
                    remediation=[f"Remove {output} and run the command again."],
                    log_path=ctx.logger.path,
                    job_id=ctx.job_id,
                )
            _clear_directory(output, ctx)
        else:
            output.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return diagnostic(
            "export.output_dir_unwritable",
            FailureCategory.CONFIGURATION_ERROR,
            "Conversion output directory could not be prepared",
            f"{output} could not be created or emptied: {redact_text(str(error))}",
            remediation=[
                f"Check permissions and free space on the filesystem holding {output}.",
            ],
            log_path=ctx.logger.path,
            job_id=ctx.job_id,
            context={"output_dir": str(output)},
        )
    return None


def _clear_directory(directory: Path, ctx: JobContext) -> None:
    """Empty a directory that is already known to be tool-owned job staging.

    Only reached for paths inside ``ctx.staging_dir``, which means the content can
    only be the remains of an interrupted run of *this* job. Nothing outside the
    staging area is ever deleted (``SPEC.md`` section 11).
    """
    children = list(directory.iterdir())
    if not children:
        return
    ctx.logger.info(
        "export.output_reset",
        "clearing leftover staging output from an earlier attempt",
        output_dir=str(directory),
        entries=len(children),
    )
    for child in children:
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
