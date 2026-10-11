"""The ``yonga`` command line: every command, every exit code, ``--json`` everywhere.

This module is the only place in the package that constructs concrete adapters
(through :mod:`yonga.application.wiring`), touches the console, or decides a
process exit code. Application services never print and never call ``sys.exit``;
they raise typed :class:`~yonga.domain.diagnostics.YongaError` failures, and the
mapping from those to the stable exit codes of ``docs/CLI_SPEC.md`` section 15
lives here.

Three rules are enforced by tests rather than by convention:

* **a traceback never reaches a user.** Every command body runs inside
  :func:`_dispatch`, which renders a typed failure as a short explanation with
  the command to run next, and an unexpected exception as exit 99 with a pointer
  to the job log. ``--debug`` adds the raw text, nothing else does;
* **``--json`` emits exactly one JSON document on stdout.** The console suppresses
  every decorative byte in JSON mode, diagnostics go to stderr, and even the C
  level output of an OpenVINO plugin is redirected away from stdout while a
  service runs;
* **``--yes`` is not a security override.** It answers ordinary questions only;
  ``trust_remote_code`` needs its own explicit flag and stops the job with exit 22
  otherwise.

Nothing runs at import time: the module only defines the Typer application, so a
``multiprocessing``/``forkserver`` child started by OpenVINO or NNCF that
re-imports ``__main__`` cannot re-execute the CLI.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
from rich.text import Text
from typer.core import TyperGroup

from yonga import APP_NAME, __version__
from yonga.application.artifacts import ArtifactStore
from yonga.application.benchmark import DEFAULT_SUITE, benchmark_result_path
from yonga.application.chat import ChatOptions, ChatService, ChatSession, ChatTurn
from yonga.application.inspect_model import InspectReport
from yonga.application.jobs import (
    JobContext,
    WorkspaceSweep,
    open_job,
    summarise_job,
    summarise_jobs,
    sweep_managed_workspaces,
)
from yonga.application.probe import ProbeOutcome, check_probe_options
from yonga.application.progress import ProgressEvent, ProgressKind, ProgressSink, null_sink
from yonga.application.self_install import InstallAction, InstallKind, PathReport, TargetKind
from yonga.application.wiring import Wiring, WiringFactory
from yonga.config.settings import Settings, load_settings
from yonga.domain.artifacts import ArtifactSummary, DeviceProbeReport, ProbeStatus
from yonga.domain.compatibility import (
    CompatibilityDecision,
    DeviceStatus,
    EvidenceLevel,
    strongest_evidence,
)
from yonga.domain.compile_cache import CacheEntryStatus, CompileCachePruneReport, entry_count
from yonga.domain.diagnostics import (
    CancelledError,
    ConfigurationError,
    DeviceUnsupportedError,
    Diagnostic,
    ExitCode,
    FailureCategory,
    RegistryError,
    YongaError,
    diagnostic,
    fill_placeholders,
)
from yonga.domain.hardware import normalise_device
from yonga.domain.invocation import CommandLine, Option
from yonga.domain.models import ModelIdentity
from yonga.registry.loader import registry_override_dirs, validate_registry
from yonga.registry.resolver import installed_registry_versions
from yonga.repairs import known_implementations
from yonga.ui import tables
from yonga.ui.chat_tui import ChatCommandKind, ChatSessionInfo, ChatView, parse_command
from yonga.ui.console import YongaConsole, human_bytes
from yonga.ui.progress import RichProgressSink
from yonga.util.redaction import redact_text

__all__ = ["CliState", "app", "cli_state", "main"]

#: Device spellings accepted by ``--device`` (``docs/CLI_SPEC.md`` sections 6, 9, 10).
DEVICE_CHOICES: tuple[str, ...] = ("auto", "npu", "gpu", "cpu")

#: Benchmark suites accepted by ``yonga bench --suite``.
SUITE_CHOICES: tuple[str, ...] = ("quick", "standard", "long")

#: How many job logs ``yonga logs`` lists when no job id is given.
LOG_LIST_LIMIT: int = 20

#: Below this, stale working directories are not worth a line of the user's
#: attention in ``yonga list``. Above it they are, because nothing else in the
#: tool ever mentions the hidden staging root.
STALE_NOTE_THRESHOLD_BYTES: int = 256 * 1024 * 1024


# ======================================================================
# state
# ======================================================================
@dataclass
class CliState:
    """Everything a command needs that the root callback resolved once.

    The adapter graph is built lazily: ``yonga --help`` and ``yonga --version``
    must work on a host with no OpenVINO installed, so no adapter is constructed
    until a command actually asks for one.
    """

    settings: Settings
    console: YongaConsole
    factory: WiringFactory
    job: JobContext | None = None
    command_line: CommandLine = field(default_factory=CommandLine)
    """What the user typed, for remediation that tells them to run it again."""
    _graph: Wiring | None = field(default=None, init=False, repr=False)
    _sink: ProgressSink | None = field(default=None, init=False, repr=False)

    @property
    def graph(self) -> Wiring:
        """The concrete adapter graph for this invocation, built on first use."""
        if self._graph is None:
            self._graph = self.factory(self.settings)
        return self._graph

    @property
    def json_mode(self) -> bool:
        """Whether stdout must carry exactly one JSON document and nothing else."""
        return self.console.json_mode

    def enable_json(self) -> None:
        """Switch the invocation into JSON mode (a command-level ``--json``)."""
        self.console.json_mode = True
        self.settings.json_output = True

    # --- output -------------------------------------------------------
    def emit(self, payload: Mapping[str, object], *, ok: bool = True) -> None:
        """Write the single success document in JSON mode; do nothing otherwise."""
        if not self.json_mode:
            return
        document: dict[str, object] = {"ok": ok}
        document.update(payload)
        self.console.print_json(document)

    def sink(self) -> ProgressSink:
        """Return this invocation's progress sink, building it once.

        Rich progress is silent in JSON mode by construction, but a null sink
        avoids building the machinery at all. The same sink is reused for every
        service in one command so a single live display owns the terminal.
        """
        if self._sink is None:
            self._sink = null_sink() if self.json_mode else RichProgressSink(self.console)
        return self._sink

    def close_sink(self) -> None:
        """Stop the live progress display, if one was started."""
        sink = self._sink
        self._sink = None
        closer = getattr(sink, "close", None)
        if callable(closer):
            with suppress(Exception):
                closer()

    # --- failure ------------------------------------------------------
    def fail(self, error: YongaError) -> ExitCode:
        """Render a typed failure and return its exit code."""
        return self.report(error.diagnostic, exit_code=exit_code_for(error))

    def rerun(self, *flags: str) -> str:
        """The command the user ran, with ``flags`` added, ready to paste."""
        return self.command_line.rerun(add=flags)

    def report(self, item: Diagnostic, *, exit_code: ExitCode) -> ExitCode:
        """Render one diagnostic the way the current output mode requires."""
        # The job id is often only known here, so placeholders are filled last.
        enriched = fill_placeholders(self.attach_job(item))
        self.console.diagnostic(
            enriched, debug_command=self.command_line.rerun(add_root=("--debug",))
        )
        if self.json_mode:
            self.console.print_json(error_document(enriched, exit_code=exit_code))
        return exit_code

    def attach_job(self, item: Diagnostic) -> Diagnostic:
        """Add the running job's id and log path when the diagnostic lacks them."""
        job = self.job
        if job is None or (item.job_id and item.log_path is not None):
            return item
        updates: dict[str, object] = {}
        if not item.job_id:
            updates["job_id"] = job.job_id
        if item.log_path is None:
            updates["log_path"] = job.paths.job_log_path(job.job_id)
        if not updates:
            return item
        return item.model_copy(update=updates)

    def unexpected(self, error: BaseException) -> ExitCode:
        """Render an unexpected exception as exit 99, never as a traceback."""
        logs_dir = self.settings.paths.logs_dir
        raw = _raw_excerpt(error, debug=self.console.debug)
        item = diagnostic(
            "internal.unexpected_error",
            FailureCategory.UNKNOWN_RUNTIME_ERROR,
            "Unexpected internal error",
            (
                "yonga hit a bug rather than a condition it knows how to explain. "
                "Nothing was left half-written: prepared artifacts are only published "
                "once they are complete."
            ),
            remediation=[
                "Re-run the same command with --debug to see the raw error: "
                + self.command_line.rerun(add_root=("--debug",)),
                f"Job logs are kept under {logs_dir}",
                "Please report the code above with the raw error.",
            ],
            raw_excerpt=raw,
        )
        return self.report(item, exit_code=ExitCode.INTERNAL)


def cli_state(ctx: typer.Context, *, json_output: bool = False) -> CliState:
    """Return the :class:`CliState` the root callback stored on the context."""
    state = ctx.find_object(CliState)
    if state is None:  # pragma: no cover - the root callback always sets it
        raise RuntimeError("the yonga root callback did not run")
    if json_output:
        state.enable_json()
    state.command_line = command_line(ctx)
    return state


def command_line(ctx: typer.Context) -> CommandLine:
    """Rebuild what the user typed from the parsed contexts, root options included.

    Click keeps no copy of the raw arguments, but it records where each
    parameter's value came from; only those given on the command line are
    repeated, in their long spelling. A value that looks like a secret is
    masked, and :class:`~yonga.domain.invocation.CommandLine` drops any option
    whose name says it carries one.
    """
    chain: list[Any] = []
    current: Any = ctx
    while current is not None:
        chain.append(current)
        current = current.parent
    chain.reverse()
    root, commands = chain[0], chain[1:]
    arguments: list[str] = []
    options: list[Option] = []
    for context in commands:
        found_arguments, found_options = _typed_parameters(context)
        arguments.extend(found_arguments)
        options.extend(found_options)
    return CommandLine(
        command=tuple(str(context.info_name) for context in commands),
        arguments=tuple(arguments),
        options=tuple(options),
        root_options=_typed_parameters(root)[1],
    )


def _typed_parameters(context: Any) -> tuple[tuple[str, ...], tuple[Option, ...]]:
    """The arguments and options of one context that came from the command line."""
    arguments: list[str] = []
    options: list[Option] = []
    for parameter in context.command.params:
        name = parameter.name
        if name is None or name not in context.params:
            continue
        source = context.get_parameter_source(name)
        if source is None or source.name != "COMMANDLINE":
            continue
        value = context.params[name]
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        if parameter.param_type_name == "argument":
            arguments.extend(_word(item) for item in values)
            continue
        flags = [*parameter.opts, *parameter.secondary_opts]
        spelling = next((flag for flag in parameter.opts if flag.startswith("--")), flags[0])
        if getattr(parameter, "is_flag", False):
            if value:
                options.append((spelling,))
            elif parameter.secondary_opts:
                options.append((parameter.secondary_opts[0],))
            continue
        options.extend((spelling, _word(item)) for item in values)
    return tuple(arguments), tuple(options)


def _word(value: object) -> str:
    """One parsed value as it would be typed, with anything secret-shaped masked."""
    raw = getattr(value, "value", value)
    if isinstance(raw, float) and raw.is_integer():
        raw = int(raw)  # ``--older-than 30`` parses to 30.0
    return redact_text(str(raw))


def exit_code_for(error: YongaError) -> ExitCode:
    """Map a typed failure onto its stable process exit code.

    The diagnostic's category decides, except for
    :class:`~yonga.domain.diagnostics.DeviceUnsupportedError`: that error *is*
    "the requested device cannot be used", which ``docs/CLI_SPEC.md`` section 15
    reserves code 50 for, whatever the underlying category (a compiler that
    cannot build the graph classifies as ``compiler_unsupported_graph``, but the
    answer the user needs is still "this device will not work, use that one").

    The exception is a diagnostic that marks the environment as broken (a device
    node the user may not open, a driver that is not initialised): the device
    would work once the host is fixed, so the answer is code 10, environment
    unusable, and not 50. Only ``prepare`` raises
    :class:`~yonga.domain.diagnostics.DeviceUnsupportedError`; ``probe``,
    ``chat`` and ``bench`` report the category's own code.
    """
    if isinstance(error, DeviceUnsupportedError):
        if error.diagnostic.environment_broken:
            return ExitCode.ENVIRONMENT_UNUSABLE
        return ExitCode.DEVICE_UNSUPPORTED
    return error.exit_code()


def error_document(item: Diagnostic, *, exit_code: ExitCode) -> dict[str, object]:
    """Build the stable ``--json`` failure document."""
    return {
        "ok": False,
        "exit_code": int(exit_code),
        "diagnostic": {
            "code": item.code,
            "category": item.category.value,
            "message": item.message,
            "fallback_devices": list(item.fallback_devices),
            "job_id": item.job_id,
            "title": item.title,
            "severity": item.severity.value,
            "remediation": list(item.remediation),
            "log_path": str(item.log_path) if item.log_path is not None else None,
        },
    }


def _raw_excerpt(error: BaseException, *, debug: bool) -> str:
    """Return the raw text kept for ``--debug``; a traceback only when asked for."""
    if debug:
        import traceback

        return "".join(traceback.format_exception(error)).strip()
    return f"{type(error).__name__}: {error}"


def _dispatch(state: CliState, action: Callable[[], ExitCode]) -> None:
    """Run one command body and turn its outcome into a process exit code.

    Every exception class is handled here so that no command can leak a traceback:
    typed failures keep their stable code, ``Ctrl-C`` is exit 60, a closed stdout
    is a quiet 0, and anything else is exit 99 with the raw text held back for
    ``--debug``.
    """
    try:
        try:
            code = action()
        finally:
            state.close_sink()
        # Commit the command's output here rather than leaving it to the
        # interpreter. A short answer fits in the pipe buffer, so writing it
        # never fails and the ``EPIPE`` surfaces in Click's own flush instead,
        # where it becomes a bare exit 1. Flushing inside this ``try`` is what
        # makes ``yonga registry paths | head -1`` end the same way
        # ``yonga logs JOB --raw | head -2`` does.
        sys.stdout.flush()
    except YongaError as error:
        code = state.fail(error)
    except BrokenPipeError:
        code = _closed_stdout()
    except SystemExit as error:
        if not _from_closed_stdout(error):
            raise
        code = _closed_stdout()
    except KeyboardInterrupt:
        code = state.report(
            diagnostic(
                "job.cancelled",
                FailureCategory.USER_CANCELLED,
                "Cancelled",
                "The command was interrupted before it finished.",
            ),
            exit_code=ExitCode.CANCELLED,
        )
    except Exception as error:  # the last line of defence; see the module docstring
        code = state.unexpected(error)
    raise typer.Exit(int(code))


def _from_closed_stdout(error: BaseException) -> bool:
    """Whether ``error`` is a library's reaction to stdout having been closed.

    Rich does not let :class:`BrokenPipeError` out of a ``print``:
    ``Console.on_broken_pipe`` silences the console and raises ``SystemExit(1)``
    instead. Without this, a table piped into ``head`` would end with a
    different exit code from a raw stream piped into ``head``, and a caller
    running under ``set -o pipefail`` would see one of them as a failure. The
    original exception is still on the chain, which is what this reads; any
    other ``SystemExit`` is left alone to do what it was raised to do.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, BrokenPipeError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _closed_stdout() -> ExitCode:
    """Treat a closed stdout as the normal end of output and exit quietly.

    ``yonga logs JOB --raw | head -2`` closes the pipe the moment ``head`` has
    what it asked for, and every later write raises :class:`BrokenPipeError`
    (Python ignores ``SIGPIPE``, so the process is not killed the way ``cat``
    is). That is the reader saying "enough", not a failure of this program, so
    nothing is printed, no diagnostic is rendered, and the exit code is 0 —
    letting it fall through to the catch-all in :func:`_dispatch` instead made
    piping into ``head``, ``less`` or ``jq`` look like exit 99 and a bug report.

    Silencing it takes two steps, because the *interpreter* flushes ``stdout``
    again at shutdown and would print ``Exception ignored ... BrokenPipeError``
    on stderr afterwards — the second error the shell must never see. Pointing
    the descriptor at ``/dev/null`` makes that last flush succeed silently; it
    is the idiom from the Python documentation's note on ``SIGPIPE``. Under a
    captured stream with no real descriptor ``fileno()`` raises, which is
    harmless: there is no shutdown flush to a dead pipe to silence.
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
    except OSError:  # pragma: no cover - /dev/null is always openable
        return ExitCode.SUCCESS
    try:
        os.dup2(devnull, sys.stdout.fileno())
    except (OSError, ValueError, AttributeError):
        pass
    finally:
        with suppress(OSError):
            os.close(devnull)
    return ExitCode.SUCCESS


@contextmanager
def _quiet_native_stdout(enabled: bool) -> Iterator[None]:
    """Redirect file-descriptor 1 to stderr while a service runs.

    The OpenVINO NPU plugin writes ``[ERROR] Upper bounds are not specified ...``
    to stdout from native code during a failed compile, which would corrupt the
    single JSON document a caller is parsing. Python-level output is unaffected:
    only the raw descriptor is moved, and it is restored afterwards.
    """
    if not enabled:
        yield
        return
    saved: int | None = None
    try:
        sys.stdout.flush()
        saved = os.dup(1)
        os.dup2(2, 1)
    except (OSError, ValueError, AttributeError):
        saved = None
    try:
        yield
    finally:
        if saved is not None:
            with suppress(OSError):
                os.dup2(saved, 1)
                os.close(saved)


def _device(value: str | None) -> str | None:
    """Validate and canonicalise a ``--device`` option."""
    if value is None:
        return None
    text = value.strip().lower()
    if text not in DEVICE_CHOICES:
        raise ConfigurationError(
            diagnostic(
                "cli.device.invalid",
                FailureCategory.CONFIGURATION_ERROR,
                "Unknown device",
                f"{value!r} is not a device yonga can target.",
                remediation=[f"Use one of: {', '.join(DEVICE_CHOICES)}."],
                context={"device": value},
            )
        )
    return "auto" if text == "auto" else normalise_device(text)


def _suite(value: str) -> str:
    """Validate a ``--suite`` option."""
    text = value.strip().lower()
    if text not in SUITE_CHOICES:
        raise ConfigurationError(
            diagnostic(
                "cli.suite.invalid",
                FailureCategory.CONFIGURATION_ERROR,
                "Unknown benchmark suite",
                f"{value!r} is not a benchmark suite.",
                remediation=[f"Use one of: {', '.join(SUITE_CHOICES)}."],
                context={"suite": value},
            )
        )
    return text


# ======================================================================
# application
# ======================================================================
def _usage_error_class() -> type[Exception]:
    """Click's ``UsageError``, as the installed typer raises it.

    typer 0.27 vendors click, so ``import click`` would name a different class
    from the one a parse failure raises there. ``typer.BadParameter`` is a
    subclass of the right one under every typer release.
    """
    for klass in typer.BadParameter.__mro__:
        if klass.__name__ == "UsageError" and issubclass(klass, Exception):
            return klass
    return typer.BadParameter  # pragma: no cover - click's BadParameter is a UsageError


_USAGE_ERROR: type[Exception] = _usage_error_class()


@contextmanager
def _usage_errors_exit_70() -> Iterator[None]:
    """Give a command line that cannot be parsed exit 70 instead of Click's 2.

    Click exits 2 for an unknown command or option, a missing argument or a
    value of the wrong type (``--timeout abc``). In yonga 2 means success with
    warnings or an inconclusive probe (``docs/CLI_SPEC.md`` section 15), and a
    script must be able to tell those from a typo; an invalid value that does
    parse (``--timeout nan``) is already a configuration error, exit 70. Only
    the code changes: Click still renders the message.
    """
    try:
        yield
    except _USAGE_ERROR as error:
        error.exit_code = int(ExitCode.REGISTRY)  # type: ignore[attr-defined]
        raise


class _YongaGroup(TyperGroup):
    """The root command group; see :func:`_usage_errors_exit_70`.

    Parsing the root's own options happens in ``make_context``; every
    subcommand, including the ``registry``, ``env``, ``cache`` and ``self``
    groups and their commands, is resolved and parsed inside ``invoke``.
    """

    def make_context(self, *args: Any, **kwargs: Any) -> Any:
        with _usage_errors_exit_70():
            return super().make_context(*args, **kwargs)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        with _usage_errors_exit_70():
            return super().invoke(*args, **kwargs)


app = typer.Typer(
    name=APP_NAME,
    help="Local AI model manager for Intel AI PCs (OpenVINO, Optimum Intel, Hugging Face).",
    no_args_is_help=True,
    add_completion=False,
    cls=_YongaGroup,
)
registry_app = typer.Typer(
    name="registry",
    help="Inspect, validate and update the compatibility registry.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(registry_app, name="registry")
env_app = typer.Typer(
    name="env",
    help="List and remove the managed environments exports run in.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(env_app, name="env")
cache_app = typer.Typer(
    name="cache",
    help="Show and prune the OpenVINO compile cache.",
    invoke_without_command=True,
    add_completion=False,
)
app.add_typer(cache_app, name="cache")
self_app = typer.Typer(
    name="self",
    help="Put the yonga AppImage on PATH as 'yonga', or take it off again.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(self_app, name="self")


def _version_callback(value: bool) -> None:
    """Print the version and exit; eager, so it needs no configuration."""
    if value:
        typer.echo(f"{APP_NAME} {__version__}")
        raise typer.Exit(0)


@app.callback()
def main(
    ctx: typer.Context,
    config: Path | None = typer.Option(
        None, "--config", metavar="PATH", help="Configuration file to load instead of the default."
    ),
    data_dir: Path | None = typer.Option(
        None, "--data-dir", metavar="PATH", help="Where prepared artifacts and logs are kept."
    ),
    cache_dir: Path | None = typer.Option(
        None, "--cache-dir", metavar="PATH", help="Where the managed temp and compile cache live."
    ),
    offline: bool = typer.Option(False, "--offline", help="Never contact the Hugging Face Hub."),
    json_output: bool = typer.Option(
        False, "--json", help="Emit one JSON document on stdout and no decoration."
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Explain what is happening."),
    debug: bool = typer.Option(False, "--debug", help="Include raw error text in failures."),
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Resolve configuration and build the state every subcommand shares."""
    console = YongaConsole(json_mode=json_output, verbose=verbose or debug, debug=debug)
    factory: WiringFactory = ctx.obj if callable(ctx.obj) else Wiring

    overrides: dict[str, object] = {
        "data_dir": data_dir,
        "cache_dir": cache_dir,
        "offline": True if offline else None,
        "json_output": True if json_output else None,
        "verbose": True if verbose else None,
        "debug": True if debug else None,
    }
    try:
        settings = load_settings(config_file=config, overrides=overrides)
    except YongaError as error:
        console.diagnostic(error.diagnostic)
        raise typer.Exit(int(exit_code_for(error))) from error

    for warning in settings.config_warnings:
        console.detail(warning)
    ctx.obj = CliState(settings=settings, console=console, factory=factory)


# ======================================================================
# doctor
# ======================================================================
@app.command()
def doctor(
    ctx: typer.Context,
    full: bool = typer.Option(False, "--full", help="Also report the loaded registry."),
    json_output: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Check this host: OpenVINO, devices, NPU access, storage and authentication."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(state, lambda: _doctor(state, full=full))


def _doctor(state: CliState, *, full: bool) -> ExitCode:
    """Run the health checks and render them (``docs/CLI_SPEC.md`` section 3)."""
    service = state.graph.doctor_service(progress=state.sink())
    with _quiet_native_stdout(state.json_mode):
        report = service.run(full=full)
    code = report.exit_code()
    if state.json_mode:
        state.emit(report.to_json(), ok=report.runtime_usable)
        return code
    state.console.print(tables.doctor_report(report.sections))
    for note in report.notes:
        state.console.print_error(note, style="dim")
    summary = report.summary()
    if summary is not None and code is ExitCode.ENVIRONMENT_UNUSABLE:
        state.console.print_error(summary)
    elif summary is not None:
        state.console.print_error(summary, style="dim")
    return code


# ======================================================================
# search
# ======================================================================
@app.command()
def search(
    ctx: typer.Context,
    query: str = typer.Argument(..., metavar="QUERY", help="Words to search the Hub for."),
    limit: int = typer.Option(20, "--limit", min=1, max=100, help="How many results to show."),
    device: str | None = typer.Option(
        None, "--device", help="Show compatibility for one device: npu, gpu, cpu or auto."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the results as JSON."),
) -> None:
    """Search the Hugging Face Hub and show what the registry knows about each result."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(state, lambda: _search(state, query, limit=limit, device=device))


def _search(state: CliState, query: str, *, limit: int, device: str | None) -> ExitCode:
    """One Hub request; no repository files are fetched (``docs/CLI_SPEC.md`` section 5)."""
    chosen = _device(device)
    report = state.graph.search_service().search(query, limit=limit)
    if state.json_mode:
        state.emit(report.to_json())
        return ExitCode.SUCCESS
    state.console.print(tables.search_results(report, device=chosen))
    return ExitCode.SUCCESS


# ======================================================================
# inspect
# ======================================================================
@app.command()
def inspect(
    ctx: typer.Context,
    model: str = typer.Argument(..., metavar="MODEL", help="Hugging Face repository id."),
    revision: str | None = typer.Option(None, "--revision", help="Branch, tag or commit sha."),
    files: bool = typer.Option(False, "--files", help="Also list the repository files."),
    json_output: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Describe a model and what preparing it on this host would involve."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(state, lambda: _inspect(state, model, revision=revision, files=files))


def _inspect(state: CliState, model: str, *, revision: str | None, files: bool) -> ExitCode:
    """Fetch metadata only — never weights (``docs/CLI_SPEC.md`` section 4)."""
    service = state.graph.inspect_service(progress=state.sink())
    report = service.inspect(model, revision=revision)
    if state.json_mode:
        state.emit(report.to_json())
        return ExitCode.SUCCESS
    state.console.print(
        tables.inspect_report(
            report.metadata,
            _decision(report),
            _source_options(report),
            show_files=files,
            architecture=report.architecture,
            identified_from=report.architecture_identified_from,
            remote_code_required=report.trust_remote_code_required,
        )
    )
    if report.known_limitations:
        state.console.section("Known limitations")
        for limitation in report.known_limitations:
            state.console.print(f"  - {limitation}")
    if report.trust_remote_code_required:
        state.console.print_error(
            "This model needs --trust-remote-code: preparing it would execute code from the "
            "repository.",
            style="yellow",
        )
    for note in report.notes:
        state.console.detail(note)
    return ExitCode.SUCCESS


def _source_options(report: InspectReport) -> list[tables.SourceOption]:
    """Adapt the service's source options to the renderer's field-identical type.

    The application layer declares its own ``SourceOption`` because
    :mod:`yonga.ui.tables` imports Rich; the two models have the same fields, so
    the hand-off is a validated round trip rather than a cast.
    """
    return [
        tables.SourceOption.model_validate(option.model_dump()) for option in report.source_options
    ]


def _decision(report: InspectReport) -> CompatibilityDecision:
    """Adapt an inspection's per-device rows to the renderer's decision type."""
    statuses = [entry.status for entry in report.compatibility]
    recommended = report.recommended_device
    status = DeviceStatus.UNKNOWN
    for entry in report.compatibility:
        if recommended is not None and entry.device == recommended:
            status = entry.status
    if recommended is None and DeviceStatus.SUPPORTED in statuses:
        status = DeviceStatus.SUPPORTED
    return CompatibilityDecision(
        requested_device=report.configured_device or "auto",
        selected_device=recommended,
        status=status,
        evidence_level=strongest_evidence(entry.evidence for entry in report.compatibility)
        if report.compatibility
        else EvidenceLevel.UNKNOWN,
        matched_rule_ids=[
            rule_id
            for rule_id in (report.architecture_rule_id, report.model_rule_id)
            if rule_id is not None
        ],
        per_device=report.compatibility,
        fallback_candidates=report.usable_devices,
    )


# ======================================================================
# prepare
# ======================================================================
@app.command()
def prepare(
    ctx: typer.Context,
    model: str = typer.Argument(..., metavar="MODEL", help="Hugging Face repository id."),
    revision: str | None = typer.Option(None, "--revision", help="Branch, tag or commit sha."),
    device: str = typer.Option("auto", "--device", help="auto, npu, gpu or cpu."),
    profile: str | None = typer.Option(None, "--profile", help="Conversion profile id."),
    source: str | None = typer.Option(
        None, "--source", help="Convert this repository instead of the requested one."
    ),
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help="Allow the conversion to execute Python published in the repository.",
    ),
    ignore_exporter_requirements: bool = typer.Option(
        False,
        "--ignore-exporter-requirements",
        help=(
            "Convert even though a declared exporter package version is unmet. "
            "Deliberate and separate: --yes never enables it."
        ),
    ),
    no_managed_env: bool = typer.Option(
        False,
        "--no-managed-env",
        help=(
            "Never build a managed export environment; refuse instead when the installed "
            "packages cannot meet the exporter requirements."
        ),
    ),
    gguf_file: str | None = typer.Option(
        None,
        "--gguf-file",
        help=(
            "For a GGUF repository, the file to import. By default yonga picks the first "
            "preferred quantization whose tensor types OpenVINO GenAI can read."
        ),
    ),
    fresh_export: bool = typer.Option(
        False,
        "--fresh-export",
        help=(
            "Export again even when a failed job left a completed export with identical "
            "inputs that could be reused."
        ),
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Answer ordinary questions with yes."),
    force: bool = typer.Option(False, "--force", help="Re-prepare even if an artifact exists."),
    no_repair: bool = typer.Option(False, "--no-repair", help="Never modify the exported IR."),
    json_output: bool = typer.Option(False, "--json", help="Emit the outcome as JSON."),
) -> None:
    """Download, convert, validate and probe a model into a prepared artifact."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(
        state,
        lambda: _prepare(
            state,
            model,
            revision=revision,
            device=device,
            profile=profile,
            source=source,
            trust_remote_code=trust_remote_code,
            ignore_exporter_requirements=ignore_exporter_requirements,
            yes=yes,
            force=force,
            no_repair=no_repair,
            # Ignoring the requirements means exporting in *this* environment
            # regardless; building a managed one would make the flag meaningless.
            managed_env=not no_managed_env and not ignore_exporter_requirements,
            reuse_export=not fresh_export,
            gguf_file=gguf_file,
        ),
    )


def _prepare(
    state: CliState,
    model: str,
    *,
    revision: str | None,
    device: str,
    profile: str | None,
    source: str | None,
    trust_remote_code: bool,
    ignore_exporter_requirements: bool,
    yes: bool,
    force: bool,
    no_repair: bool,
    managed_env: bool = True,
    reuse_export: bool = True,
    gguf_file: str | None = None,
) -> ExitCode:
    """Run the seven preparation stages (``docs/CLI_SPEC.md`` section 6)."""
    target = _device(device)
    service = state.graph.prepare_service()
    console = state.console

    def confirm(reason: object) -> bool:
        """Ask one confirmation; ``--yes`` never answers a security question."""
        message = getattr(reason, "message", str(reason))
        sensitive = bool(getattr(reason, "security_sensitive", False))
        return console.confirm(message, default=False, assume_yes=yes, security_sensitive=sensitive)

    with open_job(
        settings=state.settings,
        requested_model=model,
        progress=state.sink(),
        target_device=target,
        command="prepare",
    ) as job:
        state.job = job
        plan = service.plan(
            model,
            revision=revision,
            device=target,
            profile_id=profile,
            explicit_source=source,
            trust_remote_code=trust_remote_code,
            force=force,
            job=job,
            managed_env=managed_env,
            reuse_export=reuse_export,
            gguf_file=gguf_file,
            command_line=state.command_line,
        )
        console.print(tables.preparation_plan(plan))
        with _quiet_native_stdout(state.json_mode):
            outcome = service.prepare(
                plan,
                confirm=confirm,
                ctx=job,
                allow_repairs=not no_repair,
                ignore_exporter_requirements=ignore_exporter_requirements,
                unattended=not console.interactive,
            )

    warnings = list(outcome.warnings)
    if outcome.fell_back:
        warnings.append(
            f"{plan.target_device} could not run this model, so it was prepared for "
            f"{outcome.device} instead."
        )
    if state.json_mode:
        state.emit(
            {
                "job_id": outcome.job_id,
                "artifact_id": outcome.manifest.artifact_id,
                "artifact_dir": str(outcome.artifact_dir),
                "device": outcome.device,
                "profile": outcome.manifest.export.profile,
                "reused": outcome.reused,
                "fallbacks": list(outcome.fallbacks),
                "warnings": warnings,
                "notes": list(outcome.notes),
                "repairs": [record.repair_id for record in outcome.repairs],
                "duration_seconds": outcome.duration_seconds,
            }
        )
    else:
        console.section("Ready")
        console.print(f"  Artifact  {outcome.manifest.artifact_id}")
        console.print(f"  Path      {outcome.artifact_dir}")
        console.print(f"  Device    {outcome.device}")
        console.print(f"  Profile   {outcome.manifest.export.profile}")
        if outcome.reused:
            console.print("  Reused    an existing artifact at this revision and profile")
        for note in outcome.notes:
            console.print(f"  note: {note}", style="dim")
        for warning in warnings:
            console.print_error(f"  warning: {warning}", style="yellow")
        console.print("")
        console.print(f"Chat with it: yonga chat {outcome.manifest.artifact_id}")
    return ExitCode.WARNINGS if warnings else ExitCode.SUCCESS


# ======================================================================
# list / remove
# ======================================================================
@app.command("list")
def list_artifacts(
    ctx: typer.Context,
    clean: bool = typer.Option(
        False,
        "--clean",
        help="Retire stale staging and temp directories under the retention policy.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the catalogue as JSON."),
) -> None:
    """List the artifacts prepared on this host."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(state, lambda: _list(state, clean=clean))


def _stale_working_dirs(state: CliState, *, clean: bool) -> WorkspaceSweep | None:
    """Project (or perform) the working-directory sweep. Never fails the command."""
    try:
        return sweep_managed_workspaces(state.settings, dry_run=not clean)
    except Exception:  # pragma: no cover - a sweep must never break 'yonga list'
        return None


def _stale_note(state: CliState, sweep: WorkspaceSweep | None, *, clean: bool) -> None:
    """Print the one honest line about space held outside the artifact catalogue."""
    if sweep is None:
        return
    if clean:
        if sweep.removed_count:
            state.console.print(
                f"Reclaimed {human_bytes(sweep.freed_bytes)} from {sweep.removed_count} "
                "stale working directories."
            )
        else:
            state.console.print("No stale working directories to reclaim.")
        return
    if sweep.freed_bytes < STALE_NOTE_THRESHOLD_BYTES:
        return
    state.console.print(
        f"{human_bytes(sweep.freed_bytes)} in {sweep.removed_count} stale working "
        "directories can be reclaimed: yonga list --clean"
    )


def _list(state: CliState, *, clean: bool = False) -> ExitCode:
    """Render the prepared-artifact catalogue (``docs/CLI_SPEC.md`` section 7).

    ``--clean`` runs the working-directory retention sweep first; without it the
    sweep is only projected, and a non-trivial total earns a single line so the
    hidden staging root can never quietly eat the disk.
    """
    store: ArtifactStore = state.graph.artifact_store()
    sweep = _stale_working_dirs(state, clean=clean)
    summaries: Sequence[ArtifactSummary] = store.list_artifacts()
    if state.json_mode:
        document: dict[str, object] = {
            "artifacts": [summary.model_dump(mode="json") for summary in summaries],
            "count": len(summaries),
            "artifacts_dir": str(store.root),
        }
        # Silence is the honest answer when there is nothing stale: the key only
        # appears when it carries news, so a fresh home's document is unchanged.
        if clean or sweep is None or sweep.removed_count:
            document["stale_working_dirs"] = _stale_payload(state, sweep, clean=clean)
        state.emit(document)
        return ExitCode.SUCCESS
    if not summaries:
        state.console.print("No prepared artifacts yet.")
        state.console.print("Prepare one with: yonga prepare MODEL")
    else:
        state.console.print(tables.artifact_list(summaries))
    _stale_note(state, sweep, clean=clean)
    return ExitCode.SUCCESS


def _stale_payload(
    state: CliState,
    sweep: WorkspaceSweep | None,
    *,
    clean: bool,
) -> dict[str, object]:
    """Describe the working-directory sweep for the ``--json`` document."""
    paths = state.settings.paths
    payload: dict[str, object] = {
        "cleaned": clean,
        "staging_dir": str(paths.staging_dir),
        "temp_dir": str(paths.tmp_dir),
    }
    if sweep is None:
        payload["available"] = False
        return payload
    payload.update(
        {
            "available": True,
            "directories": sweep.removed_count,
            "bytes": sweep.freed_bytes,
            "retained_bytes": sweep.remaining_bytes,
        }
    )
    return payload


@app.command()
def remove(
    ctx: typer.Context,
    artifact: str = typer.Argument(..., metavar="ARTIFACT", help="Artifact id, repo id or path."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation."),
    json_output: bool = typer.Option(False, "--json", help="Emit the outcome as JSON."),
) -> None:
    """Remove a prepared artifact and its compile-cache entries."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(state, lambda: _remove(state, artifact, yes=yes))


def _remove(state: CliState, artifact: str, *, yes: bool) -> ExitCode:
    """Preview, confirm, then delete — never the Hugging Face cache (section 8)."""
    store = state.graph.artifact_store()
    console = state.console
    preview = store.remove(artifact, dry_run=True)

    console.print(f"Remove {preview.artifact_id or preview.artifact_dir.name}")
    console.print(f"  Path   {preview.artifact_dir}")
    console.print(f"  Frees  {human_bytes(preview.freed_bytes)}")
    for cache_dir in preview.compile_cache_paths:
        console.print(f"  Cache  {cache_dir}")
    for note in preview.notes:
        console.detail(note)
    console.print("The Hugging Face download cache is never touched.")

    if not console.confirm("Remove this artifact?", default=False, assume_yes=yes):
        raise CancelledError(
            diagnostic(
                "artifact.remove.declined",
                FailureCategory.USER_CANCELLED,
                "Nothing was removed",
                f"{preview.artifact_dir} was left in place.",
                remediation=[
                    f"Re-run with --yes to remove it without a prompt: {state.rerun('--yes')}"
                ],
            )
        )

    report = store.remove(preview.artifact_dir)
    if state.json_mode:
        state.emit(
            {
                "artifact_id": report.artifact_id,
                "artifact_dir": str(report.artifact_dir),
                "removed": report.removed,
                "freed_bytes": report.freed_bytes,
                "compile_cache_paths": [str(path) for path in report.compile_cache_paths],
            }
        )
    else:
        console.print(f"Removed {report.artifact_dir} ({human_bytes(report.freed_bytes)} freed).")
    return ExitCode.SUCCESS


# ======================================================================
# probe
# ======================================================================
@app.command()
def probe(
    ctx: typer.Context,
    artifact: str = typer.Argument(..., metavar="MODEL_OR_ARTIFACT", help="Artifact id or path."),
    device: str = typer.Option("auto", "--device", help="auto, npu, gpu or cpu."),
    all_submodels: bool = typer.Option(
        False, "--all-submodels", help="Compile every submodel, not only the language model."
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Never build the GenAI pipeline; a deferred raw compile is reported inconclusive.",
    ),
    timeout: float | None = typer.Option(
        None,
        "--timeout",
        metavar="SECONDS",
        help=(
            "Bound the GenAI pipeline build that settles a deferred compile: run it in a child "
            "process and stop it after SECONDS (the device is then inconclusive)."
        ),
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Compile a prepared artifact on a device and report what worked."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(
        state,
        lambda: _probe(
            state,
            artifact,
            device=device,
            all_submodels=all_submodels,
            quick=quick,
            timeout=timeout,
        ),
    )


def _probe(
    state: CliState,
    artifact: str,
    *,
    device: str,
    all_submodels: bool,
    quick: bool = False,
    timeout: float | None = None,
) -> ExitCode:
    """Run the per-submodel probe (``docs/CLI_SPEC.md`` section 11)."""
    target = _device(device) or "auto"
    check_probe_options(quick=quick, timeout=timeout)
    service = state.graph.probe_service()
    with open_job(
        settings=state.settings,
        requested_model=artifact,
        progress=_ProbeProgress(state.sink()),
        target_device=target,
        create_temp=False,
        command="probe",
    ) as job:
        state.job = job
        with _quiet_native_stdout(state.json_mode):
            outcome = service.probe_detailed(
                artifact,
                device=target,
                all_submodels=all_submodels,
                quick=quick,
                timeout=timeout,
                ctx=job,
            )
    if state.json_mode:
        payload = outcome.model_dump(mode="json")
        payload["role_matrix"] = outcome.role_matrix()
        payload["inconclusive"] = outcome.inconclusive
        state.emit(payload, ok=outcome.ok)
        return _probe_exit(outcome)
    state.console.print(tables.probe_report(outcome.reports, verbose=state.console.verbose))
    for note in outcome.notes:
        state.console.print_error(note, style="dim")
    unsettled = [report.device for report in outcome.reports if not report.ok]
    if outcome.ok and unsettled:
        # ``auto`` tries every candidate. A device that did not pass is worth
        # knowing about, but it is not a verdict on an artifact that runs.
        state.console.print_error(
            f"This artifact runs on {', '.join(outcome.working_devices)}. The result for "
            f"{', '.join(unsettled)} is for information only and changes nothing here.",
            style="dim",
        )
    if not outcome.ok:
        # The explanation shown is the one the exit code comes from: a definite
        # failure first, an inconclusive device only when nothing failed.
        for report in _probe_verdict_order(outcome):
            if report.diagnostic is not None:
                state.console.diagnostic(report.diagnostic)
                break
    return _probe_exit(outcome)


class _ProbeProgress:
    """The probe's progress sink, without a diagnostic block for every failed row.

    ``yonga probe`` explains its outcome once, when it knows it: a failure on
    one device is the verdict only when no other device passed, and with
    ``--device auto`` an NPU refusal printed in red, advice and all, read as
    the answer for an artifact that had just passed on the GPU. Each failure
    still has its progress line, its row in the report and its record in the
    job log.
    """

    def __init__(self, sink: ProgressSink) -> None:
        """Forward to ``sink``."""
        self._sink = sink

    def emit(self, event: ProgressEvent) -> None:
        """Forward ``event``, dropping a probe result's diagnostic."""
        if event.kind is ProgressKind.PROBE and event.diagnostic is not None:
            event = event.model_copy(update={"diagnostic": None})
        self._sink.emit(event)


def _probe_verdict_order(outcome: ProbeOutcome) -> list[DeviceProbeReport]:
    """The reports, inconclusive ones last: they are never the reason for a failure."""
    return sorted(outcome.reports, key=lambda report: report.status is ProbeStatus.INCONCLUSIVE)


def _probe_exit(outcome: ProbeOutcome) -> ExitCode:
    """Exit 0 when a device worked, otherwise the first failure's own code.

    A device left inconclusive by ``--quick`` or ``--timeout`` is not a failure:
    when nothing failed outright the probe exits 2, a warning, so a script can
    tell "not settled" from both "works" and "does not work".
    """
    if outcome.ok:
        return ExitCode.SUCCESS
    if outcome.inconclusive:
        return ExitCode.WARNINGS
    for report in _probe_verdict_order(outcome):
        if report.status is ProbeStatus.INCONCLUSIVE:
            continue
        if report.diagnostic is not None:
            # A probe that failed is a compilation failure by definition. An
            # unclassified diagnostic must not surface as exit 99 ("yonga hit a
            # bug") -- the same rule CompileError.fallback_exit_code encodes.
            if report.diagnostic.category is FailureCategory.UNKNOWN_RUNTIME_ERROR:
                return ExitCode.COMPILE
            return report.diagnostic.exit_code()
    return ExitCode.COMPILE


# ======================================================================
# chat
# ======================================================================
@app.command()
def chat(
    ctx: typer.Context,
    artifact: str = typer.Argument(..., metavar="MODEL_OR_ARTIFACT", help="Artifact id or path."),
    device: str | None = typer.Option(None, "--device", help="auto, npu, gpu or cpu."),
    max_new_tokens: int | None = typer.Option(None, "--max-new-tokens", help="Output token limit."),
    temperature: float | None = typer.Option(None, "--temperature", help="Sampling temperature."),
    top_p: float | None = typer.Option(None, "--top-p", help="Nucleus sampling probability."),
    top_k: int | None = typer.Option(None, "--top-k", help="Top-k sampling cutoff."),
    show_reasoning: bool = typer.Option(
        False, "--show-reasoning", help="Display hidden reasoning blocks."
    ),
    system: str | None = typer.Option(None, "--system", help="System prompt for the session."),
    json_output: bool = typer.Option(
        False, "--json", help="Not supported: a conversation is not one document."
    ),
) -> None:
    """Chat with a prepared artifact, locally."""
    state = cli_state(ctx, json_output=json_output)
    options = ChatOptions(
        device=_device(device),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        show_reasoning=True if show_reasoning else None,
        system_prompt=system,
        display_name=artifact,
    )
    _dispatch(state, lambda: _chat(state, artifact, options=options))


def _chat(state: CliState, artifact: str, *, options: ChatOptions) -> ExitCode:
    """Open a session and run the prompt loop (``docs/CLI_SPEC.md`` section 9)."""
    if state.json_mode:
        raise ConfigurationError(
            diagnostic(
                "cli.chat.json_unsupported",
                FailureCategory.CONFIGURATION_ERROR,
                "chat is an interactive command",
                "A conversation cannot be emitted as one JSON document.",
                remediation=[
                    "Drop --json to chat.",
                    "For a machine-readable run, use: yonga bench MODEL_OR_ARTIFACT --json",
                ],
            )
        )
    service: ChatService = state.graph.chat_service(progress=state.sink())
    session = service.open_session(artifact, options=options)
    try:
        # Load before the startup panel: a model that cannot compile on this
        # device fails the command here, with its category's exit code, instead
        # of failing every message and exiting 0.
        session.load()
        view = ChatView(state.console, _session_info(session), paths=state.settings.paths)
        try:
            view.show_startup()
            return _chat_loop(state, session, view)
        finally:
            view.close()
    finally:
        session.close()


def _session_info(session: ChatSession) -> ChatSessionInfo:
    """Adapt the service's description to the view's model."""
    return ChatSessionInfo.model_validate(session.describe().model_dump())


# ======================================================================
# serve
# ======================================================================
#: Environment variable holding the bearer token ``serve`` requires, when set.
#: Read from the environment only, never from an option: a token on the command
#: line is visible to every user of the machine through the process list.
API_KEY_ENV = "YONGA_API_KEY"

#: ``serve --max-body-mb``: the default, and the most it may be set to. The
#: default matches the API's own; the ceiling keeps one request from holding
#: more memory than a desktop can spare while it is read and decoded.
DEFAULT_MAX_BODY_MB = 8
MAX_BODY_MB_CEILING = 64


@app.command()
def serve(
    ctx: typer.Context,
    artifact: str = typer.Argument(..., metavar="MODEL_OR_ARTIFACT", help="Artifact id or path."),
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Address to bind. Loopback unless you say otherwise."
    ),
    port: int = typer.Option(8000, "--port", min=1, max=65535, help="TCP port to listen on."),
    device: str | None = typer.Option(None, "--device", help="auto, npu, gpu or cpu."),
    max_new_tokens: int | None = typer.Option(
        None, "--max-new-tokens", help="Default output token limit when a request sets none."
    ),
    show_reasoning: bool = typer.Option(
        False, "--show-reasoning", help="Return reasoning blocks instead of hiding them."
    ),
    max_body_mb: int = typer.Option(
        DEFAULT_MAX_BODY_MB,
        "--max-body-mb",
        min=1,
        max=MAX_BODY_MB_CEILING,
        help="Largest request body in MiB; embedded images count towards it.",
    ),
    carry_images_forward: bool = typer.Option(
        False,
        "--carry-images-forward",
        help="Move images sent with earlier messages onto the last one instead of refusing them.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Not supported: a server is not one document."
    ),
) -> None:
    """Serve a prepared artifact through an OpenAI-compatible HTTP API."""
    state = cli_state(ctx, json_output=json_output)
    options = ChatOptions(
        device=_device(device),
        max_new_tokens=max_new_tokens,
        show_reasoning=True if show_reasoning else None,
        display_name=artifact,
    )
    limits = _ServeLimits(
        max_body_bytes=max_body_mb * 1024 * 1024, carry_images_forward=carry_images_forward
    )
    _dispatch(
        state,
        lambda: _serve(state, artifact, options=options, host=host, port=port, limits=limits),
    )


@dataclass(frozen=True)
class _ServeLimits:
    """What ``serve`` accepts in a request, beyond what the model decides."""

    max_body_bytes: int = DEFAULT_MAX_BODY_MB * 1024 * 1024
    carry_images_forward: bool = False


def _serve(
    state: CliState,
    artifact: str,
    *,
    options: ChatOptions,
    host: str,
    port: int,
    limits: _ServeLimits | None = None,
) -> ExitCode:
    """Load the model once, then answer requests until interrupted."""
    if state.json_mode:
        raise ConfigurationError(
            diagnostic(
                "cli.serve.json_unsupported",
                FailureCategory.CONFIGURATION_ERROR,
                "serve is a long-running command",
                "A server cannot be emitted as one JSON document.",
                remediation=["Drop --json; GET /health reports the server state as JSON."],
            )
        )
    create_app = _serve_dependencies()
    _require_port(host, port)
    api_key = os.environ.get(API_KEY_ENV) or None
    console = state.console
    if not _is_loopback(host):
        warning = f"Binding {host} makes the model reachable from other machines."
        if api_key is None:
            warning += f" No {API_KEY_ENV} is set, so anyone who can reach it can use it."
        console.print_error(Text.assemble(("warning:", "yellow"), f" {warning}"))

    service: ChatService = state.graph.chat_service(progress=state.sink())
    from yonga.application.serve import ModelServer

    limits = limits if limits is not None else _ServeLimits()
    server = ModelServer(
        service.open_session(artifact, options=options),
        name=artifact,
        carry_images_forward=limits.carry_images_forward,
    )
    try:
        server.load()
        url = f"http://{_url_host(host)}:{port}/v1"
        console.print(
            Text.assemble(
                "Serving ",
                (server.model.id, "bold"),
                f" on {server.model.device} at ",
                (url, "cyan"),
                "  (API key required)" if api_key else "",
            )
        )
        if server.image_policy.accepted and not _pillow_available():
            console.print_error(
                Text.assemble(
                    ("warning:", "yellow"),
                    " Pillow is not installed, so requests with images will fail; text "
                    "works. pip install 'yonga[serve]' to accept images.",
                )
            )
        console.print("Press Ctrl+C to stop.", style="dim")
        app_instance = create_app(server, api_key=api_key, max_body_bytes=limits.max_body_bytes)
        _run_http(app_instance, host=host, port=port, verbose=state.settings.verbose)
    finally:
        server.close()
    return ExitCode.SUCCESS


# ======================================================================
# eval
# ======================================================================
@app.command("eval")
def eval_command(
    ctx: typer.Context,
    candidate: str = typer.Argument(..., metavar="CANDIDATE", help="Artifact to evaluate."),
    reference: str = typer.Option(
        ..., "--reference", help="Higher-precision artifact of the same weights to compare to."
    ),
    device: str = typer.Option("cpu", "--device", help="cpu (full precision) or gpu."),
    corpus: Path | None = typer.Option(
        None, "--corpus", help="Text to read instead of the shipped corpus (.json or .txt)."
    ),
    allow_different_sources: bool = typer.Option(
        False,
        "--allow-different-sources",
        help="Compare artifacts converted from different repositories (e.g. GGUF vs original).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
) -> None:
    """Measure how far CANDIDATE's predictions move from REFERENCE's (KL, top-1, perplexity)."""
    state = cli_state(ctx, json_output=json_output)

    def run() -> ExitCode:
        target = _eval_device(device)
        try:
            import numpy  # noqa: F401  (eval computes distributions with it)

            from yonga.application.evaluation import (
                evaluation_result_path,
                load_corpus,
                result_payload,
            )
        except ImportError as error:
            raise ConfigurationError(
                diagnostic(
                    "eval.dependencies.missing",
                    FailureCategory.PACKAGE_ABI_MISMATCH,
                    "eval needs the OpenVINO runtime",
                    f"yonga eval runs the IR through OpenVINO and needs numpy: {error}.",
                    remediation=["pip install 'yonga[openvino]'"],
                )
            ) from error

        service = state.graph.evaluation_service(progress=state.sink())
        text = load_corpus(corpus) if corpus is not None else None
        result = service.evaluate(
            candidate,
            reference,
            device=target,
            corpus=text,
            allow_different_sources=allow_different_sources,
        )
        path = evaluation_result_path(state.settings.paths, result)
        if state.json_mode:
            payload = result_payload(result)
            payload["result_path"] = str(path)
            state.emit(payload)
        else:
            state.console.print(tables.evaluation_result(result))
            state.console.print(f"Saved: {path}")
        return ExitCode.SUCCESS

    _dispatch(state, run)


#: The devices ``yonga eval`` scores on (``docs/CLI_SPEC.md`` section 13a).
EVAL_DEVICES: tuple[str, ...] = ("cpu", "gpu")


def _eval_device(value: str) -> str:
    """Validate ``eval --device``: CPU or GPU, refused up front otherwise.

    Scoring runs every passage through the IR as one prefill, which needs the
    dynamic shapes the NPU compiler rejects, so the NPU is refused before any
    model is loaded rather than after a compile failure. ``auto`` is CPU, the
    default, as it has always been.
    """
    chosen = _device(value)
    if chosen is None or chosen == "auto":
        return "CPU"
    if chosen.lower() not in EVAL_DEVICES:
        raise ConfigurationError(
            diagnostic(
                "eval.device.unsupported",
                FailureCategory.CONFIGURATION_ERROR,
                "eval runs on the CPU or the GPU",
                f"eval cannot score on {chosen}: it reads whole passages through the raw IR, "
                "whose dynamic shapes the NPU compiler does not take.",
                remediation=[
                    "Use --device cpu (full precision, the default) or --device gpu.",
                    "An artifact prepared for the NPU is evaluated on the CPU like any other.",
                ],
                context={"device": value},
            )
        )
    return chosen


# ======================================================================
# publish
# ======================================================================
@app.command()
def publish(
    ctx: typer.Context,
    artifact: str = typer.Argument(..., metavar="ARTIFACT", help="Prepared artifact id or path."),
    repo: str = typer.Option(
        ..., "--repo", help="Destination NAMESPACE/NAME on the Hub. Always explicit."
    ),
    public: bool = typer.Option(
        False, "--public", help="Create a public repository (the default is private)."
    ),
    update: bool = typer.Option(
        False, "--update", help="Write into a repository that already exists."
    ),
    license_reviewed: bool = typer.Option(
        False,
        "--license-reviewed",
        help=(
            "State that you read a license yonga cannot classify and it permits "
            "redistributing a quantized derivative. --yes never implies it."
        ),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Check everything and write the card locally; upload nothing."
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Where to write the card, manifest and upload list."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Approve the upload without a prompt."),
    json_output: bool = typer.Option(False, "--json", help="Emit the outcome as JSON."),
) -> None:
    """Upload a prepared artifact to the Hugging Face Hub with a generated model card."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(
        state,
        lambda: _publish(
            state,
            artifact,
            repo=repo,
            public=public,
            update=update,
            license_reviewed=license_reviewed,
            dry_run=dry_run,
            output=output,
            yes=yes,
        ),
    )


def _publish(
    state: CliState,
    artifact: str,
    *,
    repo: str,
    public: bool,
    update: bool,
    license_reviewed: bool,
    dry_run: bool,
    output: Path | None,
    yes: bool,
) -> ExitCode:
    """Plan, show, confirm and upload (``docs/CLI_SPEC.md`` section 14a)."""
    service = state.graph.publish_service()
    console = state.console
    plan = service.plan(
        artifact,
        repo,
        private=not public,
        update=update,
        license_reviewed=license_reviewed,
        dry_run=dry_run,
    )
    console.print(tables.publication_plan(plan))
    preview_dir = output
    if dry_run and preview_dir is None:
        preview_dir = Path.cwd() / f"yonga-publish-{plan.artifact_id}"
    written = service.write_preview(plan, preview_dir) if preview_dir is not None else []
    for path in written:
        console.detail(f"Wrote {path}")
    payload: dict[str, object] = {
        "artifact_id": plan.artifact_id,
        "repo_id": plan.repo_id,
        "private": plan.private,
        "repo_exists": plan.repo_exists,
        "licenses": [a.model_dump(mode="json") for a in plan.licenses],
        "devices": [d.model_dump(mode="json") for d in plan.devices],
        "repairs": plan.repairs,
        "files": [f.model_dump(mode="json", exclude={"local_path"}) for f in plan.files],
        "upload_bytes": plan.upload_bytes,
        "warnings": plan.warnings,
        "preview_dir": str(preview_dir) if preview_dir is not None else None,
        "dry_run": dry_run,
        "url": None,
    }
    if dry_run:
        console.print(f"Dry run: nothing was uploaded. The model card is in {preview_dir}.")
        state.emit(payload)
        return ExitCode.WARNINGS if plan.warnings else ExitCode.SUCCESS

    visibility = "private" if plan.private else "PUBLIC"
    question = (
        f"Upload {len(plan.files)} files ({human_bytes(plan.upload_bytes)}) to "
        f"{plan.url} as a {visibility} repository?"
    )
    if not console.confirm(question, default=False, assume_yes=yes):
        raise CancelledError(
            diagnostic(
                "publish.declined",
                FailureCategory.USER_CANCELLED,
                "Nothing was uploaded",
                f"{plan.repo_id} was left as it was.",
                remediation=[
                    f"Re-run with --yes to upload without a prompt: {state.rerun('--yes')}"
                ],
            )
        )
    console.print(f"Uploading to {plan.url} ...")
    url = service.publish(plan)
    console.print(Text.assemble(("Published", "green"), f" {plan.artifact_id}: {url}"))
    payload["url"] = url
    state.emit(payload)
    return ExitCode.WARNINGS if plan.warnings else ExitCode.SUCCESS


def _serve_dependencies() -> Callable[..., object]:
    """Import the optional web stack, or say exactly how to install it."""
    try:
        import uvicorn  # noqa: F401  (imported here only to prove it is installed)

        from yonga.api.app import create_app
    except ImportError as error:
        raise ConfigurationError(
            diagnostic(
                "serve.dependencies.missing",
                FailureCategory.PACKAGE_ABI_MISMATCH,
                "The serve extra is not installed",
                f"yonga serve needs FastAPI and uvicorn: {error}.",
                remediation=["pip install 'yonga[serve]'"],
            )
        ) from error
    return create_app


def _pillow_available() -> bool:
    from yonga.api.images import pillow_available

    return pillow_available()


def _run_http(app_instance: object, *, host: str, port: int, verbose: bool) -> None:
    """Run uvicorn in the foreground. A seam, so tests never open a socket."""
    import uvicorn

    uvicorn.run(
        app_instance,  # type: ignore[arg-type]
        host=host,
        port=port,
        log_level="info" if verbose else "warning",
        access_log=verbose,
    )


def _is_loopback(host: str) -> bool:
    import ipaddress

    if host.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _url_host(host: str) -> str:
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def _require_port(host: str, port: int) -> None:
    """Fail before loading a multi-gigabyte model when the port is not free."""
    import socket

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host.strip("[]"), port))
        except OSError as error:
            raise ConfigurationError(
                diagnostic(
                    "serve.address_unavailable",
                    FailureCategory.CONFIGURATION_ERROR,
                    f"Cannot listen on {host}:{port}",
                    f"Binding the address failed: {error.strerror or error}.",
                    remediation=[
                        "Pick another port with --port, or stop whatever is using this one.",
                    ],
                )
            ) from error


#: How often the chat loop wakes while a reply is generated, so Ctrl-C is seen promptly.
_TURN_POLL_SECONDS = 0.1


def _chat_loop(state: CliState, session: ChatSession, view: ChatView) -> ExitCode:
    """Read, dispatch and render until the user leaves.

    A failed message does not end the chat, but it is not forgotten either:
    leaving while the last message failed exits with that failure's code, so a
    piped conversation whose message never got an answer does not exit 0.
    """
    console = state.console
    failure: Diagnostic | None = None
    while True:
        try:
            line = view.prompt()
        except (EOFError, KeyboardInterrupt):
            console.print("")
            return _chat_exit(failure)
        if not line.strip():
            continue

        command = parse_command(line)
        if command is not None:
            if command.kind is ChatCommandKind.UNKNOWN:
                view.show_error(command.error or f"Unknown command: {line.strip()}")
                continue
            outcome = session.apply_command(command)
            if outcome.message:
                if outcome.ok:
                    console.print(outcome.message, style="dim")
                else:
                    view.show_error(outcome.message)
            if outcome.action.value == "help":
                view.show_help()
            elif outcome.action.value == "clear":
                view.clear()
            elif outcome.action.value == "model":
                view.info = _session_info(session)
                view.show_model()
            elif outcome.exit_requested:
                return _chat_exit(failure)
            continue

        with view.generating():
            turn = _send_cancellable(state, session, line)
        if turn.cancelled:
            console.print("Reply cancelled.", style="dim")
            continue
        if not turn.ok:
            if turn.diagnostic is not None:
                console.diagnostic(turn.diagnostic)
                failure = turn.diagnostic
            continue
        failure = None
        if session.show_reasoning and turn.reasoning_text:
            view.show_reasoning(turn.reasoning_text)
        view.show_answer(turn.result or turn.display_text)
        if session.stats:
            view.show_metrics(turn.result)


def _send_cancellable(state: CliState, session: ChatSession, line: str) -> ChatTurn:
    """Answer one message on a worker thread, so Ctrl-C can stop the reply.

    Generation runs in native code, where the interpreter cannot act on a Ctrl-C
    until it returns. Here the main thread only waits, so the first Ctrl-C
    reaches it at once and asks the session to stop at the next token; the loop
    then returns to the prompt with the conversation intact. A second Ctrl-C
    while that reply is still stopping ends the chat (exit 60).
    """
    cancel = threading.Event()
    turns: list[ChatTurn] = []
    errors: list[BaseException] = []

    def answer() -> None:
        try:
            turns.append(session.send(line, should_stop=cancel.is_set))
        except BaseException as error:  # re-raised on the main thread below
            errors.append(error)

    worker = threading.Thread(target=answer, name="yonga-chat-turn", daemon=True)
    started = False
    while True:
        # Starting the worker is inside the guard too: Thread.start() waits, and
        # a Ctrl-C landing there is the same first press as one landing later.
        try:
            if not started:
                started = True
                worker.start()
            if not worker.is_alive():
                break
            worker.join(_TURN_POLL_SECONDS)
        except KeyboardInterrupt:
            if cancel.is_set():
                raise
            cancel.set()
            state.console.print_error("Stopping the reply; Ctrl-C again quits.", style="dim")
    if errors:
        raise errors[0]
    # No turn at all means the Ctrl-C came before the worker could run.
    return turns[0] if turns else ChatTurn(cancelled=True)


def _chat_exit(failure: Diagnostic | None) -> ExitCode:
    """Exit 0, or the code of the failure that left the last message unanswered.

    A generation failure no rule classified keeps the generation stage's code
    (52), never 99: the runtime failed, not yonga.
    """
    if failure is None:
        return ExitCode.SUCCESS
    code = failure.exit_code()
    return ExitCode.GENERATION if code is ExitCode.INTERNAL else code


# ======================================================================
# bench
# ======================================================================
@app.command()
def bench(
    ctx: typer.Context,
    artifact: str = typer.Argument(..., metavar="MODEL_OR_ARTIFACT", help="Artifact id or path."),
    device: str | None = typer.Option(None, "--device", help="auto, npu, gpu or cpu."),
    suite: str = typer.Option(DEFAULT_SUITE, "--suite", help="quick, standard or long."),
    runs: int | None = typer.Option(None, "--runs", help="Measured runs per prompt."),
    warmups: int | None = typer.Option(None, "--warmups", help="Warmup runs per prompt."),
    json_output: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
) -> None:
    """Benchmark a prepared artifact and save a reproducible JSON result."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(
        state,
        lambda: _bench(state, artifact, device=device, suite=suite, runs=runs, warmups=warmups),
    )


def _bench(
    state: CliState,
    artifact: str,
    *,
    device: str | None,
    suite: str,
    runs: int | None,
    warmups: int | None,
) -> ExitCode:
    """Run the benchmark suite (``docs/CLI_SPEC.md`` section 10)."""
    target = _device(device)
    chosen = _suite(suite)
    service = state.graph.benchmark_service()
    with open_job(
        settings=state.settings,
        requested_model=artifact,
        progress=state.sink(),
        target_device=target,
        command="bench",
    ) as job:
        state.job = job
        with _quiet_native_stdout(state.json_mode):
            result = service.run(
                artifact,
                device=target,
                suite=chosen,
                runs=runs,
                warmups=warmups,
                ctx=job,
            )
    path = benchmark_result_path(state.settings.paths, result)
    if state.json_mode:
        payload = result.to_json()
        payload["result_path"] = str(path)
        state.emit(payload, ok=not result.errors)
        return _bench_exit(result.errors, result.has_measurements)
    state.console.print(tables.benchmark_summary(result))
    state.console.print(f"Saved: {path}")
    for error in result.errors:
        state.console.print_error(f"  {error}", style="yellow")
    return _bench_exit(result.errors, result.has_measurements)


def _bench_exit(errors: Sequence[str], measured: bool) -> ExitCode:
    """Success, success-with-warnings, or a generation failure when nothing ran."""
    if not errors:
        return ExitCode.SUCCESS
    return ExitCode.WARNINGS if measured else ExitCode.GENERATION


# ======================================================================
# registry
# ======================================================================
@env_app.command("list")
def env_list(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit the environments as JSON."),
) -> None:
    """Show the managed export environments and the disk they use."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(state, lambda: _env_list(state))


def _env_list(state: CliState) -> ExitCode:
    """List every completed managed export environment, and the interpreter copies under them."""
    manager = state.graph.export_environments()
    environments = manager.environments()
    interpreters = manager.base_interpreters()
    total = sum(environment.size_bytes for environment in environments)
    total += sum(interpreter.size_bytes for interpreter in interpreters)
    if state.json_mode:
        state.emit(
            {
                "environments": [
                    environment.model_dump(mode="json") for environment in environments
                ],
                "base_interpreters": [
                    {
                        "path": str(interpreter.path),
                        "size_bytes": interpreter.size_bytes,
                        "in_use": interpreter.in_use,
                    }
                    for interpreter in interpreters
                ],
                "total_bytes": total,
            }
        )
        return ExitCode.SUCCESS
    if not environments and not interpreters:
        state.console.print(
            "No managed export environments. yonga builds one when a model's exporter "
            "requirements cannot be met by the installed packages.",
            style="dim",
        )
        return ExitCode.SUCCESS
    for environment in environments:
        pins = ", ".join(
            f"{name}{specifier}"
            for name, specifier in environment.spec.packages.items()
            if specifier
        )
        state.console.print(
            f"{environment.key}  {human_bytes(environment.size_bytes):>9}  "
            f"{environment.created_at:%Y-%m-%d}  {environment.path}"
        )
        state.console.print(f"    {pins}", style="dim")
    for interpreter in interpreters:
        role = "the environments' base" if interpreter.in_use else "unused"
        state.console.print(
            f"base Python   {human_bytes(interpreter.size_bytes):>9}  {interpreter.path}  ({role})"
        )
    noun = "environment" if len(environments) == 1 else "environments"
    state.console.print(f"{len(environments)} {noun}, {human_bytes(total)} in total.")
    if interpreters:
        state.console.print(
            "The base Python is a copy of the AppImage's interpreter; it is removed with the "
            "last environment made from it.",
            style="dim",
        )
    return ExitCode.SUCCESS


@env_app.command("remove")
def env_remove(
    ctx: typer.Context,
    key: str = typer.Argument(..., metavar="KEY", help="Environment key, or a unique prefix."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation."),
    json_output: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
) -> None:
    """Delete one managed export environment. It is rebuilt if a later export needs it."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(state, lambda: _env_remove(state, key, yes=yes))


def _env_remove(state: CliState, key: str, *, yes: bool) -> ExitCode:
    """Name the environment, confirm, then remove it (``--json`` needs ``--yes``)."""
    manager = state.graph.export_environments()
    target = manager.resolve(key)
    question = f"Delete {target.path} ({human_bytes(target.size_bytes)})?"
    if not state.console.confirm(question, default=False, assume_yes=yes):
        raise CancelledError(
            diagnostic(
                "export.env.remove.declined",
                FailureCategory.USER_CANCELLED,
                "Nothing was removed",
                f"{target.path} was left in place.",
                remediation=[
                    f"Re-run with --yes to remove it without a prompt: {state.rerun('--yes')}"
                ],
            )
        )
    removed = manager.remove(target.key)
    if state.json_mode:
        state.emit({"removed": removed.key, "path": str(removed.path), "bytes": removed.size_bytes})
        return ExitCode.SUCCESS
    state.console.print(f"Removed {removed.path} ({human_bytes(removed.size_bytes)} freed).")
    return ExitCode.SUCCESS


# ======================================================================
# cache
# ======================================================================
@cache_app.callback()
def cache_main(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit the entries as JSON."),
) -> None:
    """Show and prune the OpenVINO compile cache. Alone, the same as 'cache list'."""
    if ctx.invoked_subcommand is not None:
        if json_output:
            cli_state(ctx, json_output=True)
        return
    state = cli_state(ctx, json_output=json_output)
    _dispatch(state, lambda: _cache_list(state))


@cache_app.command("list")
def cache_list(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit the entries as JSON."),
) -> None:
    """List compile cache entries: size, last use, owner, and whether they are still used."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(state, lambda: _cache_list(state))


def _cache_list(state: CliState) -> ExitCode:
    """Render the compile cache inventory (``docs/CLI_SPEC.md`` section 12b)."""
    inventory = state.graph.compile_cache_service().inventory()
    if state.json_mode:
        state.emit(inventory.to_json())
        return ExitCode.SUCCESS
    console = state.console
    console.print(f"Compile cache  {inventory.root}")
    if not inventory.entries:
        console.print("The compile cache is empty.", style="dim")
        return ExitCode.SUCCESS
    console.print(tables.compile_cache_entries(inventory.entries))
    counts = inventory.counts()
    breakdown = ", ".join(f"{count} {status}" for status, count in counts.items() if count)
    console.print(
        f"{entry_count(len(inventory.entries))}, {human_bytes(inventory.total_bytes)} in total "
        f"({breakdown})."
    )
    _cache_runtime_note(state, inventory.installed_openvino)
    if inventory.reclaimable_bytes:
        console.print(
            f"{human_bytes(inventory.reclaimable_bytes)} is orphaned or built by another "
            "OpenVINO release: yonga cache prune"
        )
    if counts[CacheEntryStatus.UNATTRIBUTED.value]:
        console.print(
            "Unattributed entries carry no ownership record and match no known artifact; "
            "'yonga cache prune --unattributed' removes them too.",
            style="dim",
        )
    for name in inventory.ignored:
        console.detail(f"Not a cache entry, left alone: {inventory.root / name}")
    return ExitCode.SUCCESS


def _cache_runtime_note(state: CliState, installed: str | None) -> None:
    """Say so when no entry can be called stale because OpenVINO is unreadable."""
    if installed is None:
        state.console.print_error(
            "The installed OpenVINO version could not be read, so no entry is marked "
            "stale-runtime.",
            style="dim",
        )


@cache_app.command("prune")
def cache_prune(
    ctx: typer.Context,
    older_than: float | None = typer.Option(
        None,
        "--older-than",
        metavar="DAYS",
        help="Also remove any entry not used for this many days.",
    ),
    unattributed: bool = typer.Option(
        False,
        "--unattributed",
        help="Also remove entries that cannot be traced to any artifact.",
    ),
    everything: bool = typer.Option(False, "--all", help="Empty the whole compile cache."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would go; delete nothing."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation."),
    json_output: bool = typer.Option(False, "--json", help="Emit the outcome as JSON."),
) -> None:
    """Remove orphaned and stale-runtime compile cache entries (and more, when asked)."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(
        state,
        lambda: _cache_prune(
            state,
            older_than=older_than,
            unattributed=unattributed,
            everything=everything,
            dry_run=dry_run,
            yes=yes,
        ),
    )


def _cache_prune(
    state: CliState,
    *,
    older_than: float | None,
    unattributed: bool,
    everything: bool,
    dry_run: bool,
    yes: bool,
) -> ExitCode:
    """Plan, show, confirm, then delete compile cache entries and report the bytes freed."""
    service = state.graph.compile_cache_service()
    inventory = service.inventory()
    plan = service.plan_prune(
        inventory,
        older_than_days=older_than,
        unattributed=unattributed,
        everything=everything,
    )
    console = state.console

    def document(report: CompileCachePruneReport) -> dict[str, object]:
        return {
            "compile_cache_dir": str(plan.root),
            "installed_openvino": plan.installed_openvino,
            "criteria": plan.criteria(),
            "dry_run": report.dry_run,
            "planned": [
                {**entry.model_dump(mode="json"), "selected_because": plan.reasons[entry.key]}
                for entry in plan.entries
            ],
            "planned_bytes": report.planned_bytes,
            "removed": list(report.removed),
            "used_since_plan": list(report.used_since_plan),
            "failed": [failure.model_dump(mode="json") for failure in report.failed],
            "freed_bytes": report.freed_bytes,
            "kept": len(inventory.entries) - len(plan.entries),
            "ignored": list(inventory.ignored),
        }

    if not plan.entries:
        if state.json_mode:
            state.emit(document(CompileCachePruneReport(dry_run=dry_run)))
            return ExitCode.SUCCESS
        if inventory.entries:
            console.print(
                f"Nothing to prune: {entry_count(len(inventory.entries))} "
                f"({human_bytes(inventory.total_bytes)}) are all kept under these options."
            )
        else:
            console.print("Nothing to prune: the compile cache is empty.")
        if inventory.entries:
            _cache_runtime_note(state, plan.installed_openvino)
        for name in inventory.ignored:
            console.detail(f"Not a cache entry, left alone: {inventory.root / name}")
        return ExitCode.SUCCESS

    console.print(tables.compile_cache_entries(plan.entries, reasons=plan.reasons))
    kept = len(inventory.entries) - len(plan.entries)
    console.print(
        f"Prune {len(plan.entries)} of {entry_count(len(inventory.entries))}, "
        f"{human_bytes(plan.total_bytes)}; {kept} kept. Only {plan.root} is touched."
    )
    _cache_runtime_note(state, plan.installed_openvino)
    if CacheEntryStatus.STALE_RUNTIME.value in plan.reasons.values():
        console.print(
            f"Stale-runtime means built by an OpenVINO other than {plan.installed_openvino}, "
            "the one running this command; an entry another install used after this "
            "release was first used here is kept as in-use.",
            style="dim",
        )
    for name in inventory.ignored:
        console.detail(f"Not a cache entry, left alone: {inventory.root / name}")
    if dry_run:
        state.emit(document(service.prune(plan, dry_run=True)))
        console.print("Dry run: nothing was removed.", style="dim")
        return ExitCode.SUCCESS

    if not console.confirm("Remove these compile cache entries?", default=False, assume_yes=yes):
        raise CancelledError(
            diagnostic(
                "cache.prune.declined",
                FailureCategory.USER_CANCELLED,
                "Nothing was pruned",
                f"The compile cache under {plan.root} was left as it was.",
                remediation=[
                    f"Re-run with --yes to prune without a prompt: {state.rerun('--yes')}"
                ],
            )
        )

    report = service.prune(plan)
    if state.json_mode:
        state.emit(document(report), ok=not report.failed)
    else:
        console.print(
            f"Removed {entry_count(len(report.removed))}, {human_bytes(report.freed_bytes)} freed."
        )
        for key in report.used_since_plan:
            console.print(f"Kept {plan.root / key}: it was used after the plan was made.")
        for failure in report.failed:
            console.print_error(f"Could not remove {failure.path}: {failure.error}")
    return ExitCode.WARNINGS if report.failed else ExitCode.SUCCESS


# ======================================================================
# self
# ======================================================================
_BIN_DIR_HELP = "Directory to install the 'yonga' command in (default: ~/.local/bin)."
_UNINSTALL_BIN_DIR_HELP = "Directory to remove the 'yonga' command from (default: ~/.local/bin)."


@self_app.command("install")
def self_install(
    ctx: typer.Context,
    bin_dir: Path | None = typer.Option(None, "--bin-dir", metavar="DIR", help=_BIN_DIR_HELP),
    yes: bool = typer.Option(False, "--yes", "-y", help="Replace an old AppImage without asking."),
    json_output: bool = typer.Option(False, "--json", help="Emit the outcome as JSON."),
) -> None:
    """Copy the running AppImage to ~/.local/bin/yonga so 'yonga' works in any shell."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(state, lambda: _self_install(state, bin_dir=bin_dir, yes=yes))


def _self_install(state: CliState, *, bin_dir: Path | None, yes: bool) -> ExitCode:
    """Plan, confirm a replacement, copy atomically, then check PATH (section 12c)."""
    service = state.graph.self_install_service()
    plan = service.plan_install(bin_dir)
    console = state.console
    target = plan.target.path
    if plan.action is InstallAction.REPLACE:
        old = f"yonga {plan.previous_version}" if plan.previous_version else "version unknown"
        modified = plan.target.modified
        when = f", modified {modified:%Y-%m-%d}" if modified is not None else ""
        console.print(f"{target} is another yonga AppImage.")
        console.print(f"  installed  {old}, {human_bytes(plan.target.size)}{when}")
        console.print(
            f"  this       yonga {plan.version}, {human_bytes(plan.source_size)}  {plan.source}"
        )
        if not console.confirm(f"Replace {target}?", default=False, assume_yes=yes):
            raise CancelledError(
                diagnostic(
                    "self.install.declined",
                    FailureCategory.USER_CANCELLED,
                    "Nothing was replaced",
                    f"{target} was left as it was.",
                    remediation=[
                        f"Re-run with --yes to replace it without a prompt: {state.rerun('--yes')}"
                    ],
                )
            )
    result = service.install(plan)
    report = service.path_report(plan.directory, target, created=result.created_directory)
    code = ExitCode.SUCCESS if report.command_is_target else ExitCode.WARNINGS
    if state.json_mode:
        state.emit({**result.to_json(), **report.to_json(), "exit_code": int(code)})
        return code
    if plan.action is InstallAction.CURRENT:
        console.print(f"yonga {plan.version} is already installed at {target}.")
    elif plan.action is InstallAction.REPLACE:
        console.print(f"Replaced {target} with yonga {plan.version}.")
    else:
        console.print(f"Installed yonga {plan.version} to {target}.")
    if result.created_directory:
        console.detail(f"Created {plan.directory}.")
    if not result.receipt_written:
        console.detail(f"Could not update {service.receipts.path}; the version is not recorded.")
    _render_path_report(state, report)
    return code


def _render_path_report(state: CliState, report: PathReport) -> None:
    """Say whether 'yonga' now works, or exactly what to do so it does."""
    console = state.console
    if report.command_is_target:
        console.print("Run 'yonga doctor' to check this host.", style="dim")
        return
    for line in report.advice:
        console.print_error(line, style="yellow")


@self_app.command("uninstall")
def self_uninstall(
    ctx: typer.Context,
    bin_dir: Path | None = typer.Option(
        None, "--bin-dir", metavar="DIR", help=_UNINSTALL_BIN_DIR_HELP
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation."),
    json_output: bool = typer.Option(False, "--json", help="Emit the outcome as JSON."),
) -> None:
    """Remove the AppImage copy at ~/.local/bin/yonga. Models and caches are kept."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(state, lambda: _self_uninstall(state, bin_dir=bin_dir, yes=yes))


def _self_uninstall(state: CliState, *, bin_dir: Path | None, yes: bool) -> ExitCode:
    """Remove ``DIR/yonga`` only when it is an AppImage, after confirmation."""
    service = state.graph.self_install_service()
    plan = service.plan_uninstall(bin_dir)
    console = state.console
    locations = service.data_locations()
    target = plan.target.path

    def document(removed: bool) -> dict[str, object]:
        return {
            "target": str(target),
            "removed": removed,
            "version": plan.version,
            "kept": {name: str(path) for name, path in locations.items()},
        }

    if plan.target.kind is TargetKind.MISSING:
        if state.json_mode:
            state.emit(document(False))
        else:
            console.print(f"Nothing to remove: there is no yonga AppImage at {target}.")
        return ExitCode.SUCCESS

    from_python = service.running().kind is InstallKind.PYTHON
    if from_python and not state.json_mode:
        # Typed from a pip or pipx install, 'self uninstall' could read as removing that.
        console.print(
            f"This yonga runs from a Python environment; 'self uninstall' removes only the "
            f"AppImage at {target}. To remove this installation, use pip or pipx yourself."
        )
    version = f"yonga {plan.version}" if plan.version else "version unknown"
    question = f"Remove {target} ({version}, {human_bytes(plan.target.size or 0)})?"
    if not console.confirm(question, default=False, assume_yes=yes):
        raise CancelledError(
            diagnostic(
                "self.uninstall.declined",
                FailureCategory.USER_CANCELLED,
                "Nothing was removed",
                f"{target} was left as it was.",
                remediation=[
                    f"Re-run with --yes to remove it without a prompt: {state.rerun('--yes')}"
                ],
            )
        )
    removed = service.uninstall(plan)
    if state.json_mode:
        state.emit(document(removed))
        return ExitCode.SUCCESS
    console.print(f"Removed {target}." if removed else f"{target} was already gone.")
    _render_kept_data(state, locations, from_python=from_python)
    return ExitCode.SUCCESS


def _render_kept_data(state: CliState, locations: Mapping[str, Path], *, from_python: bool) -> None:
    """Say what was left in place and how to remove it, without doing it."""
    console = state.console
    console.print(
        f"Kept: prepared models and logs in {locations['data_dir']}, caches in "
        f"{locations['cache_dir']}, settings in {locations['config_dir']}."
    )
    # A Python environment's yonga is still installed; only an AppImage user may have
    # just removed the only 'yonga' command there was.
    where = (
        "this Python environment's yonga still runs them"
        if from_python
        else "run them from the downloaded AppImage if 'yonga' is gone now"
    )
    console.print(
        "To free space: 'yonga remove ARTIFACT' deletes a model and 'yonga cache prune --all' "
        f"empties the compile cache ({where}). Deleting those directories yourself removes "
        "everything.",
        style="dim",
    )


@registry_app.command("validate")
def registry_validate(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
) -> None:
    """Load every registry layer and check it is internally consistent."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(state, lambda: _registry_validate(state))


def _registry_validate(state: CliState) -> ExitCode:
    """Deterministic validation of the merged registry (section 12)."""
    registry = state.graph.registry()
    problems = validate_registry(registry, known_implementations=known_implementations())
    counts = registry.counts()
    if state.json_mode:
        state.emit(
            {
                "schema_version": registry.schema_version,
                "layers": list(registry.layers),
                "counts": dict(counts),
                "problems": problems,
                "ignored": list(registry.ignored),
            },
            ok=not problems,
        )
    else:
        state.console.print(f"Schema version  {registry.schema_version}")
        for name, count in counts.items():
            state.console.print(f"  {name:<14} {count}")
        state.console.print(f"  layers         {len(registry.layers)}")
        if registry.ignored:
            # Validation covers what loaded; a layer that did not load would
            # otherwise pass unseen under "Registry is valid."
            state.console.section("Ignored")
            for reason in registry.ignored:
                state.console.print(f"  - {reason}")
        if problems:
            # The problem list is the command's result, not a side note, so it
            # belongs on stdout next to the counts; the summary diagnostic that
            # follows goes to stderr like every other failure.
            state.console.section("Problems")
            for problem in problems:
                state.console.print(f"  - {problem}")
        elif registry.ignored:
            state.console.print(
                "The loaded layers are valid; 'yonga registry status' says why one is not in use."
            )
        else:
            state.console.print("Registry is valid.")
    if problems:
        raise RegistryError(
            diagnostic(
                "registry.invalid",
                FailureCategory.REGISTRY_ERROR,
                "The registry is not valid",
                f"{len(problems)} problem(s) were found in the merged registry.",
                remediation=[
                    "Fix the reported rules in the override layer that introduced them.",
                    f"Override layers are read from: {_override_summary(state)}",
                ],
            )
        )
    return ExitCode.WARNINGS if registry.ignored else ExitCode.SUCCESS


@registry_app.command("profiles")
def registry_profiles(
    ctx: typer.Context,
    device: str | None = typer.Option(None, "--device", help="Only profiles for this device."),
    json_output: bool = typer.Option(False, "--json", help="Emit the profiles as JSON."),
) -> None:
    """List the conversion profiles '--profile' accepts."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(state, lambda: _registry_profiles(state, device=device))


def _registry_profiles(state: CliState, *, device: str | None) -> ExitCode:
    """Every profile of the merged registry, by device and then by priority."""
    target = _device(device)
    registry = state.graph.registry()
    chosen = [
        rule for rule in registry.profiles if target in (None, "auto") or rule.device == target
    ]
    chosen.sort(key=lambda rule: (rule.device, -rule.priority, rule.id))
    rows: list[dict[str, object]] = [
        {
            "id": rule.id,
            "device": rule.device,
            "weights": rule.weights_label(),
            "band": (
                rule.applies_to.parameter_count.describe()
                if rule.applies_to.parameter_count is not None
                else None
            ),
            "priority": rule.priority,
            "status": rule.status.value,
            "evidence": rule.evidence.value,
            "summary": rule.summary,
            "origin": registry.origins.get(rule.id),
        }
        for rule in chosen
    ]
    if state.json_mode:
        state.emit({"profiles": rows, "ignored": list(registry.ignored)})
        return ExitCode.SUCCESS
    if not rows:
        state.console.print(f"No registry profile targets {target}.")
        return ExitCode.SUCCESS
    state.console.print(tables.registry_profiles(rows))
    return ExitCode.SUCCESS


def _override_summary(state: CliState) -> str:
    """Describe where override layers are read from, for a remediation line."""
    directories = registry_override_dirs(state.settings.paths)
    return ", ".join(str(directory) for directory in directories) or "none"


@registry_app.command("explain")
def registry_explain(
    ctx: typer.Context,
    model: str = typer.Argument(..., metavar="MODEL", help="Hugging Face repository id."),
    device: str | None = typer.Option(None, "--device", help="Explain for this device only."),
    json_output: bool = typer.Option(False, "--json", help="Emit the explanation as JSON."),
) -> None:
    """Show which registry rules matched a model, and why they won."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(state, lambda: _registry_explain(state, model, device=device))


def _registry_explain(state: CliState, model: str, *, device: str | None) -> ExitCode:
    """Resolve every rule for one model and render the decision."""
    target = _device(device)
    graph = state.graph
    metadata = graph.hub_client().inspect(ModelIdentity(repo_id=model))
    with _quiet_native_stdout(state.json_mode):
        inventory = graph.hardware_probe().inventory()
    explanation = graph.registry_resolver().explain(
        meta=metadata,
        inventory=inventory,
        device=None if target in (None, "auto") else target,
    )
    ignored = list(graph.registry().ignored)
    if state.json_mode:
        state.emit({**explanation.model_dump(mode="json"), "ignored": ignored})
        return ExitCode.SUCCESS
    state.console.print(tables.registry_explanation(explanation))
    for reason in ignored:
        state.console.print_error(
            Text.assemble(("warning:", "yellow"), f" {reason}; explained without it.")
        )
    return ExitCode.SUCCESS


@registry_app.command("paths")
def registry_paths(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit the paths as JSON."),
) -> None:
    """Show where registry layers, configuration and managed data live."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(state, lambda: _registry_paths(state))


def _registry_paths(state: CliState) -> ExitCode:
    """Print the configuration and registry layer locations (section 12)."""
    from yonga.registry.loader import builtin_registry_appimage, builtin_registry_dir

    paths = state.settings.paths
    overrides = registry_override_dirs(paths)
    builtin = builtin_registry_dir()
    appimage = builtin_registry_appimage(builtin)
    payload: dict[str, object] = {
        "builtin": str(builtin) if builtin is not None else None,
        # The builtin path above is on the AppImage's per-run mount; this is the file to keep.
        "builtin_appimage": str(appimage) if appimage is not None else None,
        "overrides": [str(directory) for directory in overrides],
        "config_file": str(state.settings.config_file) if state.settings.config_file else None,
        "versions": installed_registry_versions(),
        "paths": {name: str(value) for name, value in paths.as_dict().items()},
    }
    if state.json_mode:
        state.emit(payload)
        return ExitCode.SUCCESS
    if appimage is not None:
        state.console.print(
            f"Builtin registry   inside {appimage} (read-only; its mount exists only while "
            "yonga runs, so 'yonga registry explain' and 'registry profiles' show the rules)"
        )
    else:
        state.console.print(f"Builtin registry   {payload['builtin']}")
    state.console.print(f"Override layers   {', '.join(str(d) for d in overrides) or 'none'}")
    state.console.print(f"Configuration file {payload['config_file'] or 'none'}")
    state.console.section("Managed directories")
    for name, value in paths.as_dict().items():
        state.console.print(f"  {name:<24} {value}")
    return ExitCode.SUCCESS


@registry_app.command("update")
def registry_update(
    ctx: typer.Context,
    url: str | None = typer.Option(
        None, "--url", help="Bundle URL (https); the signature is fetched from URL.sig."
    ),
    file: Path | None = typer.Option(None, "--file", help="A bundle already on disk."),
    signature: Path | None = typer.Option(
        None, "--signature", help="Its signature (default: FILE.sig)."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Install without a prompt."),
    json_output: bool = typer.Option(False, "--json", help="Emit the outcome as JSON."),
) -> None:
    """Fetch, verify and install a signed remote registry bundle."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(
        state,
        lambda: _registry_update(state, url=url, file=file, signature=signature, yes=yes),
    )


def _registry_update(
    state: CliState,
    *,
    url: str | None,
    file: Path | None,
    signature: Path | None,
    yes: bool,
) -> ExitCode:
    """Verify a bundle completely, show what it changes, then install it."""
    from yonga.registry.remote import RemoteRegistryStore, fetch

    settings = state.settings
    if settings.registry_update_policy == "never":
        raise ConfigurationError(
            diagnostic(
                "registry.remote.disabled",
                FailureCategory.CONFIGURATION_ERROR,
                "Remote registry updates are disabled",
                "registry_update_policy is 'never' in the configuration.",
                remediation=['Set registry_update_policy = "manual" to allow updates.'],
            )
        )
    if url is not None and file is not None:
        raise ConfigurationError(
            diagnostic(
                "registry.remote.conflicting_sources",
                FailureCategory.CONFIGURATION_ERROR,
                "Choose one bundle to update from",
                "--url and --file each name a bundle; only one can be installed.",
                remediation=["Pass either --url URL or --file BUNDLE, not both."],
            )
        )
    if signature is not None and file is None:
        raise ConfigurationError(
            diagnostic(
                "registry.remote.signature_without_file",
                FailureCategory.CONFIGURATION_ERROR,
                "--signature goes with --file",
                "A downloaded bundle's signature is always fetched from URL.sig.",
                remediation=["Pass --signature together with --file BUNDLE."],
            )
        )
    if file is not None:
        source = str(file)
        try:
            bundle_bytes = file.read_bytes()
            signature_bytes = (signature or file.with_name(file.name + ".sig")).read_bytes()
        except OSError as error:
            raise ConfigurationError(
                diagnostic(
                    "registry.remote.file_unreadable",
                    FailureCategory.CONFIGURATION_ERROR,
                    "The bundle could not be read",
                    str(error),
                )
            ) from error
    else:
        target = url or settings.registry_url
        if not target:
            raise ConfigurationError(
                diagnostic(
                    "registry.remote.no_source",
                    FailureCategory.CONFIGURATION_ERROR,
                    "No registry bundle to update from",
                    "Neither --url, --file nor registry_url in the configuration is set.",
                    remediation=["yonga registry update --url https://…/bundle.json"],
                )
            )
        if settings.offline:
            raise ConfigurationError(
                diagnostic(
                    "registry.remote.offline",
                    FailureCategory.CONFIGURATION_ERROR,
                    "Updating the registry needs the network",
                    "yonga is in offline mode.",
                    remediation=["Use --file with a bundle you already have."],
                )
            )
        source = target
        bundle_bytes = fetch(target)
        signature_bytes = fetch(target + ".sig", limit=64 * 1024)

    store = RemoteRegistryStore(settings.paths, registry_url=settings.registry_url)
    verified = store.verify(bundle_bytes, signature_bytes)
    current = state.graph.registry()
    incoming = [rule for document in verified.documents for rule in document.rules()]
    known = set(current.rule_ids)
    replaced = sorted(rule.id for rule in incoming if rule.id in known)
    added = sorted(rule.id for rule in incoming if rule.id not in known)
    bundle = verified.bundle
    payload: dict[str, object] = {
        "version": bundle.version,
        "key_id": verified.key_id,
        "sha256": verified.sha256,
        "created_at": bundle.created_at.isoformat(),
        "expires_at": bundle.expires_at.isoformat(),
        "description": bundle.description,
        "added": added,
        "replaced": replaced,
        "source": source,
        "installed": False,
    }
    console = state.console
    console.print(f"Signed by key {verified.key_id}; signature verified.")
    console.print(
        f"Bundle version {bundle.version}, created {bundle.created_at:%Y-%m-%d}, "
        f"expires {bundle.expires_at:%Y-%m-%d}"
    )
    if bundle.description:
        console.print(f"  {bundle.description}")
    console.print(f"Adds {len(added)} rule(s), replaces {len(replaced)}.")
    for rule_id in added:
        console.detail(f"+ {rule_id}")
    for rule_id in replaced:
        console.detail(f"~ {rule_id}")
    if not console.confirm("Install this registry bundle?", default=False, assume_yes=yes):
        raise CancelledError(
            diagnostic(
                "registry.remote.declined",
                FailureCategory.USER_CANCELLED,
                "The registry was not changed",
                "The bundle verified, but was not installed.",
                remediation=[
                    f"Re-run with --yes to install it without a prompt: {state.rerun('--yes')}"
                ],
            )
        )
    store.install(verified, bundle_bytes, signature_bytes, source=source)
    payload["installed"] = True
    console.print(
        Text.assemble(("Installed", "green"), f" registry bundle version {bundle.version}.")
    )
    state.emit(payload)
    return ExitCode.SUCCESS


@registry_app.command("status")
def registry_status(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit the status as JSON."),
) -> None:
    """Show the installed remote registry bundle and whether it still verifies."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(state, lambda: _registry_status(state))


def _registry_status(state: CliState) -> ExitCode:
    """Re-verify the installed bundle from scratch and report what is in use."""
    from yonga.registry.remote import RemoteRegistryStore, trusted_keys

    settings = state.settings
    store = RemoteRegistryStore(settings.paths, registry_url=settings.registry_url)
    installed = store.state()
    keys = trusted_keys(settings.paths)
    problem: str | None = None
    in_use = False
    if installed is not None:
        try:
            in_use = store.load() is not None and settings.registry_update_policy != "never"
        except RegistryError as error:
            problem = f"{error.diagnostic.code}: {error.diagnostic.message}"
    payload: dict[str, object] = {
        "policy": settings.registry_update_policy,
        "trusted_keys": [{"key_id": k.key_id, "file": str(k.source)} for k in keys],
        "installed": installed.model_dump(mode="json") if installed is not None else None,
        "in_use": in_use,
        "problem": problem,
    }
    if state.json_mode:
        state.emit(payload)
        return ExitCode.WARNINGS if problem else ExitCode.SUCCESS
    console = state.console
    console.print(f"Update policy   {settings.registry_update_policy}")
    console.print(
        f"Trusted keys    {', '.join(k.key_id for k in keys) or 'none'}  "
        f"({settings.paths.registry_keys_dir})"
    )
    if installed is None:
        console.print("Remote bundle   none installed; the built-in registry is in use.")
        return ExitCode.SUCCESS
    console.print(
        f"Remote bundle   version {installed.version}, key {installed.key_id}, "
        f"expires {installed.expires_at:%Y-%m-%d}, from {installed.source}"
    )
    if problem:
        console.print(Text.assemble(("Not in use:", "yellow"), f" {problem}"))
        return ExitCode.WARNINGS
    console.print("In use          " + ("yes" if in_use else "no (policy is 'never')"))
    return ExitCode.SUCCESS


@registry_app.command("reset")
def registry_reset(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit the outcome as JSON."),
) -> None:
    """Remove the installed remote bundle and go back to the built-in registry."""
    state = cli_state(ctx, json_output=json_output)

    def run() -> ExitCode:
        from yonga.registry.remote import RemoteRegistryStore

        removed = RemoteRegistryStore(state.settings.paths).reset()
        state.console.print(
            "Removed the remote registry bundle." if removed else "No remote bundle was installed."
        )
        state.emit({"removed": removed})
        return ExitCode.SUCCESS

    _dispatch(state, run)


# ======================================================================
# logs
# ======================================================================
@app.command()
def logs(
    ctx: typer.Context,
    job_id: str | None = typer.Argument(None, metavar="JOB_ID", help="Job id, or a prefix."),
    raw: bool = typer.Option(False, "--raw", help="Print the JSON Lines file verbatim."),
    bundle: bool = typer.Option(
        False,
        "--bundle",
        help="Write a redacted, anonymised .tar.gz for a bug report "
        "(log, state, manifest, host report).",
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Where to write the --bundle archive.", dir_okay=False
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the records as JSON."),
) -> None:
    """Show the job logs kept on disk, most recent first."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(state, lambda: _logs(state, job_id, raw=raw, bundle=bundle, output=output))


def _logs(
    state: CliState,
    job_id: str | None,
    *,
    raw: bool,
    bundle: bool = False,
    output: Path | None = None,
) -> ExitCode:
    """List job logs, or render one (``docs/CLI_SPEC.md`` section 13)."""
    logs_dir = state.settings.paths.logs_dir
    if bundle:
        if job_id is None:
            raise ConfigurationError(
                diagnostic(
                    "logs.bundle.no_job",
                    FailureCategory.CONFIGURATION_ERROR,
                    "--bundle needs a job",
                    "Name the job to bundle.",
                    remediation=["Find one with: yonga logs, then: yonga logs JOB_ID --bundle"],
                )
            )
        return _log_bundle(state, job_id, output=output)
    if job_id is None:
        return _log_index(state, logs_dir)
    path = _log_path(logs_dir, job_id)
    if raw:
        text = path.read_text(encoding="utf-8", errors="replace")
        sys.stdout.write(text if text.endswith("\n") or not text else text + "\n")
        return ExitCode.SUCCESS

    records = _log_records(path)
    summary = summarise_job(path, paths=state.settings.paths)
    if state.json_mode:
        state.emit(
            {
                "job_id": path.stem,
                "path": str(path),
                "summary": summary.to_json(),
                "records": records,
            }
        )
        return ExitCode.SUCCESS
    state.console.print(f"Job {path.stem}  ({path})")
    state.console.print(tables.job_heading(summary))
    for record in records:
        level = str(record.get("level", "info"))
        event = str(record.get("event", ""))
        message = str(record.get("message", ""))
        stamp = tables.local_clock(record.get("timestamp"))
        style = "red" if level == "error" else ("yellow" if level == "warning" else "")
        state.console.print(f"  {stamp}  {level:<7} {event:<26} {message}", style=style or None)
    return ExitCode.SUCCESS


def _log_bundle(state: CliState, job_id: str, *, output: Path | None) -> ExitCode:
    """Write a bug-report bundle for one job and say what went into it."""
    from yonga.application.report import build_bundle

    host: dict[str, object] | None = None
    try:
        with _quiet_native_stdout(True):
            host = state.graph.doctor_service().run().to_json()
    except Exception:
        # A host report makes a bundle more useful; failing to gather one must
        # not prevent sharing the log that describes the failure.
        host = None
    result = build_bundle(state.settings.paths, job_id, output=output, host_report=host)
    if state.json_mode:
        state.emit(result.to_json())
        return ExitCode.SUCCESS
    state.console.print(f"Bug-report bundle written to {result.path}")
    for member in result.members:
        state.console.print(f"  {member.name:<16} {member.description}")
    for note in result.notes:
        state.console.print_error(note, style="dim")
    state.console.print(
        "Secrets were redacted; your home directory, hostname, login name and Hugging Face "
        "account name anonymised. Read it before sharing.",
        style="dim",
    )
    return ExitCode.SUCCESS


def _log_index(state: CliState, logs_dir: Path) -> ExitCode:
    """Render the most recent jobs: what each ran, on what, and how it ended."""
    jobs = summarise_jobs(state.settings.paths, limit=LOG_LIST_LIMIT)
    if state.json_mode:
        state.emit(
            {
                "logs_dir": str(logs_dir),
                "jobs": [
                    {
                        **job.to_json(),
                        "path": str(job.log_path),
                        "modified": _modified(job.log_path),
                    }
                    for job in jobs
                ],
            }
        )
        return ExitCode.SUCCESS
    if not jobs:
        state.console.print(f"No job logs in {logs_dir}.")
        return ExitCode.SUCCESS
    state.console.print(f"Job logs in {logs_dir}")
    state.console.print(tables.job_list(jobs))
    state.console.print("")
    state.console.print("Show one with: yonga logs JOB_ID (a unique prefix is enough)")
    return ExitCode.SUCCESS


def _modified(path: Path) -> float | None:
    """A log file's modification time, or ``None`` when it vanished meanwhile."""
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _log_path(logs_dir: Path, job_id: str) -> Path:
    """Resolve a job id (or an unambiguous prefix) to its log file."""
    from yonga.application.report import find_job_log

    return find_job_log(logs_dir, job_id)


def _log_records(path: Path) -> list[dict[str, object]]:
    """Parse a JSON Lines job log, skipping any line that is not a record."""
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def run() -> None:
    """Console-script entry point.

    Kept as a function so that nothing executes at import time: an OpenVINO or
    NNCF ``forkserver`` child re-imports the main module, and an unguarded call
    here would re-run the whole command in every worker.
    """
    app()


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    run()
