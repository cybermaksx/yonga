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
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path

import typer

from yonga import APP_NAME, __version__
from yonga.application.artifacts import ArtifactStore
from yonga.application.benchmark import DEFAULT_SUITE, benchmark_result_path
from yonga.application.chat import ChatOptions, ChatService, ChatSession
from yonga.application.inspect_model import InspectReport
from yonga.application.jobs import JobContext, open_job
from yonga.application.probe import ProbeOutcome
from yonga.application.progress import ProgressSink, null_sink
from yonga.application.wiring import Wiring, WiringFactory
from yonga.config.settings import Settings, load_settings
from yonga.domain.artifacts import ArtifactSummary
from yonga.domain.compatibility import (
    CompatibilityDecision,
    DeviceStatus,
    EvidenceLevel,
    strongest_evidence,
)
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
)
from yonga.domain.hardware import normalise_device
from yonga.domain.models import ModelIdentity
from yonga.registry.loader import registry_override_dirs, validate_registry
from yonga.registry.resolver import installed_registry_versions
from yonga.repairs import known_implementations
from yonga.ui import tables
from yonga.ui.chat_tui import ChatCommandKind, ChatSessionInfo, ChatView, parse_command
from yonga.ui.console import YongaConsole, human_bytes
from yonga.ui.progress import RichProgressSink

__all__ = ["CliState", "app", "cli_state", "main"]

#: Device spellings accepted by ``--device`` (``docs/CLI_SPEC.md`` sections 6, 9, 10).
DEVICE_CHOICES: tuple[str, ...] = ("auto", "npu", "gpu", "cpu")

#: Benchmark suites accepted by ``yonga bench --suite``.
SUITE_CHOICES: tuple[str, ...] = ("quick", "standard", "long")

#: How many job logs ``yonga logs`` lists when no job id is given.
LOG_LIST_LIMIT: int = 20


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

    def report(self, item: Diagnostic, *, exit_code: ExitCode) -> ExitCode:
        """Render one diagnostic the way the current output mode requires."""
        enriched = self.attach_job(item)
        self.console.diagnostic(enriched)
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
                "Re-run the same command with --debug to see the raw error.",
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
    return state


def exit_code_for(error: YongaError) -> ExitCode:
    """Map a typed failure onto its stable process exit code.

    The diagnostic's category decides, except for
    :class:`~yonga.domain.diagnostics.DeviceUnsupportedError`: that error *is*
    "the requested device cannot be used", which ``docs/CLI_SPEC.md`` section 15
    reserves code 50 for, whatever the underlying category (a compiler that
    cannot build the graph classifies as ``compiler_unsupported_graph``, but the
    answer the user needs is still "this device will not work, use that one").
    """
    if isinstance(error, DeviceUnsupportedError):
        return ExitCode.DEVICE_UNSUPPORTED
    return error.exit_code()


def error_document(item: Diagnostic, *, exit_code: ExitCode) -> dict[str, object]:
    """Build the stable ``--json`` failure document (``AGENT_PROMPT.md`` section 26)."""
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
    typed failures keep their stable code, ``Ctrl-C`` is exit 60, and anything
    else is exit 99 with the raw text held back for ``--debug``.
    """
    try:
        try:
            code = action()
        finally:
            state.close_sink()
    except YongaError as error:
        code = state.fail(error)
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
app = typer.Typer(
    name=APP_NAME,
    help="Local AI model manager for Intel AI PCs (OpenVINO, Optimum Intel, Hugging Face).",
    no_args_is_help=True,
    add_completion=False,
)
registry_app = typer.Typer(
    name="registry",
    help="Inspect and validate the compatibility registry.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(registry_app, name="registry")


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
    if code is ExitCode.ENVIRONMENT_UNUSABLE:
        state.console.print_error("No usable OpenVINO device was found on this host.")
    elif code is ExitCode.WARNINGS:
        state.console.print_error("Usable, with warnings above.", style="dim")
    return code


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
        )
        console.print(tables.preparation_plan(plan))
        with _quiet_native_stdout(state.json_mode):
            outcome = service.prepare(
                plan,
                confirm=confirm,
                ctx=job,
                allow_repairs=not no_repair,
                ignore_exporter_requirements=ignore_exporter_requirements,
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
    json_output: bool = typer.Option(False, "--json", help="Emit the catalogue as JSON."),
) -> None:
    """List the artifacts prepared on this host."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(state, lambda: _list(state))


def _list(state: CliState) -> ExitCode:
    """Render the prepared-artifact catalogue (``docs/CLI_SPEC.md`` section 7)."""
    store: ArtifactStore = state.graph.artifact_store()
    summaries: Sequence[ArtifactSummary] = store.list_artifacts()
    if state.json_mode:
        state.emit(
            {
                "artifacts": [summary.model_dump(mode="json") for summary in summaries],
                "count": len(summaries),
                "artifacts_dir": str(store.root),
            }
        )
        return ExitCode.SUCCESS
    if not summaries:
        state.console.print("No prepared artifacts yet.")
        state.console.print("Prepare one with: yonga prepare MODEL")
        return ExitCode.SUCCESS
    state.console.print(tables.artifact_list(summaries))
    return ExitCode.SUCCESS


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
                remediation=["Re-run with --yes to remove it without a prompt."],
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
    json_output: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Compile a prepared artifact on a device and report what worked."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(state, lambda: _probe(state, artifact, device=device, all_submodels=all_submodels))


def _probe(state: CliState, artifact: str, *, device: str, all_submodels: bool) -> ExitCode:
    """Run the per-submodel probe (``docs/CLI_SPEC.md`` section 11)."""
    target = _device(device) or "auto"
    service = state.graph.probe_service()
    with open_job(
        settings=state.settings,
        requested_model=artifact,
        progress=state.sink(),
        target_device=target,
        create_temp=False,
    ) as job:
        state.job = job
        with _quiet_native_stdout(state.json_mode):
            outcome = service.probe_detailed(
                artifact, device=target, all_submodels=all_submodels, ctx=job
            )
    if state.json_mode:
        payload = outcome.model_dump(mode="json")
        payload["role_matrix"] = outcome.role_matrix()
        state.emit(payload, ok=outcome.ok)
        return _probe_exit(outcome)
    state.console.print(tables.probe_report(outcome.reports))
    for note in outcome.notes:
        state.console.print_error(note, style="dim")
    if not outcome.ok:
        for report in outcome.reports:
            if report.diagnostic is not None:
                state.console.diagnostic(report.diagnostic)
                break
    return _probe_exit(outcome)


def _probe_exit(outcome: ProbeOutcome) -> ExitCode:
    """Exit 0 when a device worked, otherwise the first failure's own code."""
    if outcome.ok:
        return ExitCode.SUCCESS
    for report in outcome.reports:
        if report.diagnostic is not None:
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
    view = ChatView(state.console, _session_info(session), paths=state.settings.paths)
    try:
        view.show_startup()
        return _chat_loop(state, session, view)
    finally:
        session.close()
        view.close()


def _session_info(session: ChatSession) -> ChatSessionInfo:
    """Adapt the service's description to the view's model."""
    return ChatSessionInfo.model_validate(session.describe().model_dump())


def _chat_loop(state: CliState, session: ChatSession, view: ChatView) -> ExitCode:
    """Read, dispatch and render until the user leaves."""
    console = state.console
    while True:
        try:
            line = view.prompt()
        except (EOFError, KeyboardInterrupt):
            console.print("")
            return ExitCode.SUCCESS
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
                return ExitCode.SUCCESS
            continue

        with view.generating():
            turn = session.send(line)
        if not turn.ok:
            if turn.diagnostic is not None:
                console.diagnostic(turn.diagnostic)
            continue
        view.show_answer(turn.result or turn.display_text)
        if session.show_reasoning and turn.reasoning_text:
            console.print(turn.reasoning_text, style="dim")
        if session.stats:
            view.show_metrics(turn.result)


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
        return _bench_exit(result.errors, bool(result.aggregates))
    state.console.print(tables.benchmark_summary(result))
    state.console.print(f"Saved: {path}")
    for error in result.errors:
        state.console.print_error(f"  {error}", style="yellow")
    return _bench_exit(result.errors, bool(result.aggregates))


def _bench_exit(errors: Sequence[str], measured: bool) -> ExitCode:
    """Success, success-with-warnings, or a generation failure when nothing ran."""
    if not errors:
        return ExitCode.SUCCESS
    return ExitCode.WARNINGS if measured else ExitCode.GENERATION


# ======================================================================
# registry
# ======================================================================
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
            },
            ok=not problems,
        )
    else:
        state.console.print(f"Schema version  {registry.schema_version}")
        for name, count in counts.items():
            state.console.print(f"  {name:<14} {count}")
        state.console.print(f"  layers         {len(registry.layers)}")
        if problems:
            # The problem list is the command's result, not a side note, so it
            # belongs on stdout next to the counts; the summary diagnostic that
            # follows goes to stderr like every other failure.
            state.console.section("Problems")
            for problem in problems:
                state.console.print(f"  - {problem}")
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
    if state.json_mode:
        state.emit(explanation.model_dump(mode="json"))
        return ExitCode.SUCCESS
    state.console.print(tables.registry_explanation(explanation))
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
    from yonga.registry.loader import builtin_registry_dir

    paths = state.settings.paths
    overrides = registry_override_dirs(paths)
    payload: dict[str, object] = {
        "builtin": str(builtin_registry_dir()) if builtin_registry_dir() is not None else None,
        "overrides": [str(directory) for directory in overrides],
        "config_file": str(state.settings.config_file) if state.settings.config_file else None,
        "versions": installed_registry_versions(),
        "paths": {name: str(value) for name, value in paths.as_dict().items()},
    }
    if state.json_mode:
        state.emit(payload)
        return ExitCode.SUCCESS
    state.console.print(f"Builtin registry   {payload['builtin']}")
    state.console.print(f"Override layers    {', '.join(str(d) for d in overrides) or 'none'}")
    state.console.print(f"Configuration file {payload['config_file'] or 'none'}")
    state.console.section("Managed directories")
    for name, value in paths.as_dict().items():
        state.console.print(f"  {name:<24} {value}")
    return ExitCode.SUCCESS


# ======================================================================
# logs
# ======================================================================
@app.command()
def logs(
    ctx: typer.Context,
    job_id: str | None = typer.Argument(None, metavar="JOB_ID", help="Job id, or a prefix."),
    raw: bool = typer.Option(False, "--raw", help="Print the JSON Lines file verbatim."),
    json_output: bool = typer.Option(False, "--json", help="Emit the records as JSON."),
) -> None:
    """Show the job logs kept on disk, most recent first."""
    state = cli_state(ctx, json_output=json_output)
    _dispatch(state, lambda: _logs(state, job_id, raw=raw))


def _logs(state: CliState, job_id: str | None, *, raw: bool) -> ExitCode:
    """List job logs, or render one (``docs/CLI_SPEC.md`` section 13)."""
    logs_dir = state.settings.paths.logs_dir
    if job_id is None:
        return _log_index(state, logs_dir)
    path = _log_path(logs_dir, job_id)
    if raw:
        text = path.read_text(encoding="utf-8", errors="replace")
        sys.stdout.write(text if text.endswith("\n") or not text else text + "\n")
        return ExitCode.SUCCESS

    records = _log_records(path)
    if state.json_mode:
        state.emit({"job_id": path.stem, "path": str(path), "records": records})
        return ExitCode.SUCCESS
    state.console.print(f"Job {path.stem}  ({path})")
    for record in records:
        level = str(record.get("level", "info"))
        event = str(record.get("event", ""))
        message = str(record.get("message", ""))
        stamp = str(record.get("timestamp", ""))[11:23]
        style = "red" if level == "error" else ("yellow" if level == "warning" else "")
        state.console.print(f"  {stamp}  {level:<7} {event:<26} {message}", style=style or None)
    return ExitCode.SUCCESS


def _log_index(state: CliState, logs_dir: Path) -> ExitCode:
    """Render the most recent job logs."""
    entries = sorted(
        (path for path in logs_dir.glob("*.jsonl") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:LOG_LIST_LIMIT]
    if state.json_mode:
        state.emit(
            {
                "logs_dir": str(logs_dir),
                "jobs": [
                    {
                        "job_id": path.stem,
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "modified": path.stat().st_mtime,
                    }
                    for path in entries
                ],
            }
        )
        return ExitCode.SUCCESS
    if not entries:
        state.console.print(f"No job logs in {logs_dir}.")
        return ExitCode.SUCCESS
    state.console.print(f"Job logs in {logs_dir}")
    for path in entries:
        state.console.print(f"  {path.stem}  {human_bytes(path.stat().st_size)}")
    state.console.print("")
    state.console.print("Show one with: yonga logs JOB_ID")
    return ExitCode.SUCCESS


def _log_path(logs_dir: Path, job_id: str) -> Path:
    """Resolve a job id (or an unambiguous prefix) to its log file."""
    exact = logs_dir / f"{job_id}.jsonl"
    if exact.is_file():
        return exact
    matches = sorted(path for path in logs_dir.glob(f"{job_id}*.jsonl") if path.is_file())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ConfigurationError(
            diagnostic(
                "logs.not_found",
                FailureCategory.CONFIGURATION_ERROR,
                "No such job log",
                f"No job log for {job_id!r} exists in {logs_dir}.",
                remediation=["List the jobs that do exist: yonga logs"],
                context={"job_id": job_id},
            )
        )
    raise ConfigurationError(
        diagnostic(
            "logs.ambiguous",
            FailureCategory.CONFIGURATION_ERROR,
            "That job id prefix is ambiguous",
            f"{len(matches)} job logs start with {job_id!r}.",
            remediation=["Use the full job id: yonga logs " + matches[0].stem],
            context={"job_id": job_id},
        )
    )


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
