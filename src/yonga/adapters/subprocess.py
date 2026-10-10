"""The single place in yonga where an external process is started.

Everything an external tool does — today only ``optimum-cli export openvino`` —
goes through :class:`SubprocessRunner`. Centralising it is what makes the
security rules in ``SPEC.md`` section 11 checkable in one file:

* the command is always an argv **list** with ``shell=False``; no model id, path
  or user string is ever interpolated into a command line;
* stdout and stderr are streamed line by line into the job's JSON Lines log,
  every line passing through :func:`yonga.util.redaction.redact_text` first;
* only the environment *overrides* are recorded, never the whole environment,
  and those are redacted too;
* the child is started with ``start_new_session=True`` so it owns its process
  group: Ctrl+C in the terminal does not leak into it, and cancellation is an
  explicit ``SIGINT`` → ``SIGTERM`` → ``SIGKILL`` escalation against that group;
* a non-zero exit is **returned**, never raised. Classifying the failure is the
  registry's job, not this adapter's.

The module name deliberately shadows the standard library inside this package;
absolute imports mean ``import subprocess`` below still resolves to the stdlib.
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING

from yonga.application.progress import ProgressKind
from yonga.util.redaction import redact_argv, redact_env, redact_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yonga.adapters import CommandResult, CommandRunner
    from yonga.application.jobs import JobContext
    from yonga.util.jsonlog import JobLogger

__all__ = [
    "DEFAULT_TAIL_LINES",
    "CompletedCommand",
    "SubprocessRunner",
    "default_line_selector",
    "progress_text",
]

#: Number of trailing output lines kept in memory for each stream.
DEFAULT_TAIL_LINES: int = 200

#: Exit code reported when the command could not be started at all.
SPAWN_FAILURE_RETURNCODE: int = 127

#: Lines that report a milestone of a conversion, surfaced as ``EXPORT`` progress.
_MILESTONE = re.compile(
    r"(exporting|converting|compressing|applying weight compression|weight compression|"
    r"statistics of the bitwidth|mixed-precision|searching for|saving|loading weights|"
    r"quantizing|calibrat|INFO:nncf)",
    re.IGNORECASE,
)

#: Lines worth surfacing verbatim even in a quiet run.
_NOTEWORTHY = re.compile(r"(traceback|error|failed|failure|warning|not supported)", re.IGNORECASE)

#: A carriage-return redraw of a progress bar; only the final segment is useful.
_PROGRESS_BAR = re.compile(r"\d+%\|")

#: The first box-drawing or block glyph and everything after it. Rich draws its
#: progress bars (``Applying Weight Compression ━━━ 100% • 0:00:05``) and its
#: tables (nncf's bit-width statistics) with these, so what precedes them is the
#: label and a line that starts with one is a table border or row.
_DRAWING = re.compile(r"\s*[\u2500-\u259f].*$", re.DOTALL)


def progress_text(line: str) -> str:
    """Return the part of one output line worth showing as progress.

    A Rich progress bar keeps its label and loses the bar; a table row or
    border becomes empty. The job log always keeps the full line.
    """
    return _DRAWING.sub("", line.strip())


def default_line_selector(line: str) -> ProgressKind | None:
    """Classify one captured output line for progress reporting.

    Returns :attr:`~yonga.application.progress.ProgressKind.EXPORT` for a
    conversion milestone, :attr:`~yonga.application.progress.ProgressKind.LOG`
    for a warning/error worth showing, and ``None`` for the overwhelming
    majority of lines, which belong in the log file and nowhere else.

    Parsing every progress bar is not required: an indeterminate status plus
    selected lines is enough.
    """
    text = progress_text(line)
    if not text:
        return None
    if _NOTEWORTHY.search(text):
        return ProgressKind.LOG
    if _PROGRESS_BAR.search(text):
        return None
    if _MILESTONE.search(text):
        return ProgressKind.EXPORT
    return None


def _recorded_overrides(
    candidates: Mapping[str, str],
    *,
    forced: Mapping[str, str],
) -> dict[str, str]:
    """Reduce a child environment to the variables that are genuinely overrides.

    Anything whose value is identical to the inherited environment is not an
    override and is dropped, so a caller that passes a complete environment
    records nothing more than a caller that passes three variables. The managed
    temp variables in ``forced`` are always kept: they are what makes a recorded
    conversion reproducible.
    """
    inherited = os.environ
    return {
        key: value
        for key, value in candidates.items()
        if key in forced or inherited.get(key) != value
    }


@dataclass
class CompletedCommand:
    """The outcome of one subprocess execution.

    Structurally satisfies :class:`yonga.adapters.CommandResult`. Everything
    stored here is already redacted, so the whole object can be serialised into
    a manifest or a log record without further filtering.
    """

    argv: Sequence[str]
    """The command line **after** :func:`~yonga.util.redaction.redact_argv`."""

    returncode: int
    duration_seconds: float
    stdout_lines: Sequence[str] = field(default_factory=list)
    """The last ``tail_lines`` redacted stdout lines."""

    stderr_lines: Sequence[str] = field(default_factory=list)
    """The last ``tail_lines`` redacted stderr lines."""

    cancelled: bool = False
    """``True`` when the run was stopped by Ctrl+C or a programmatic cancel."""

    timed_out: bool = False
    """``True`` when the run exceeded its ``timeout`` and was terminated."""

    env_overrides: Mapping[str, str] = field(default_factory=dict)
    """Only the variables this run *set*, redacted. Never the whole environment."""

    cwd: Path | None = None
    pid: int | None = None
    error: str | None = None
    """Populated when the process could not be started (e.g. executable missing)."""

    @property
    def ok(self) -> bool:
        """``True`` when the command ran to completion with exit code zero."""
        return self.returncode == 0 and not self.cancelled and not self.timed_out

    @property
    def output_tail(self) -> list[str]:
        """Both streams' tails, stderr last, for a compact failure excerpt."""
        return [*self.stdout_lines, *self.stderr_lines]


class SubprocessRunner:
    """Runs an external command as an argv list, never through a shell.

    Implements :class:`yonga.adapters.CommandRunner`.
    """

    def __init__(
        self,
        *,
        tail_lines: int = DEFAULT_TAIL_LINES,
        poll_interval: float = 0.1,
        sigint_grace_seconds: float = 5.0,
        sigterm_grace_seconds: float = 5.0,
        sigkill_grace_seconds: float = 2.0,
        line_selector: Callable[[str], ProgressKind | None] | None = None,
        extra_secrets: Sequence[str] = (),
    ) -> None:
        """Create a runner.

        ``tail_lines`` bounds the in-memory output tail; a conversion can emit
        hundreds of megabytes of progress bars, so the full stream only ever
        goes to the job log file. ``extra_secrets`` adds literal strings that
        must be masked in captured output on top of the pattern-based rules.
        """
        self._tail_lines = max(1, int(tail_lines))
        self._poll_interval = max(0.01, float(poll_interval))
        self._grace: tuple[tuple[int, float], ...] = (
            (int(signal.SIGINT), max(0.0, float(sigint_grace_seconds))),
            (int(signal.SIGTERM), max(0.0, float(sigterm_grace_seconds))),
            (
                int(getattr(signal, "SIGKILL", signal.SIGTERM)),
                max(0.0, float(sigkill_grace_seconds)),
            ),
        )
        self._line_selector = line_selector or default_line_selector
        self._extra_secrets: tuple[str, ...] = tuple(extra_secrets)

    # --- public API ----------------------------------------------------
    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float | None = None,
        ctx: JobContext | None = None,
    ) -> CompletedCommand:
        """Execute ``argv`` with ``shell=False`` and return a typed result.

        ``env`` holds *overrides* merged onto the inherited environment (or onto
        ``ctx.child_env()``, which already forces ``TMPDIR``/``TMP``/``TEMP`` at
        the managed temp directory). It is never the complete environment, and
        only these overrides are recorded on the result.

        That last sentence is enforced here rather than trusted: a caller that
        hands in a whole environment must not be able to dump every inherited
        variable into the job log, where name-based redaction cannot recognise a
        secret held under an unfamiliar name. Anything whose value merely repeats
        the inherited environment is therefore dropped from the record, while the
        managed temp variables are always kept because reproducing the run needs
        them.

        A non-zero exit code is returned, not raised. A failure to *start* the
        process is also returned, with ``returncode`` 127 and ``error`` set, so
        that callers have exactly one code path to classify.
        """
        command = [str(item) for item in argv]
        if not command:
            message = "A command must have at least one argument."
            raise ValueError(message)

        overrides = dict(env or {})
        forced = dict(ctx.temp_env) if ctx is not None else {}
        base = ctx.child_env() if ctx is not None else dict(os.environ)
        child_env = {**base, **overrides}
        recorded = _recorded_overrides({**forced, **overrides}, forced=forced)

        safe_argv = redact_argv(command)
        safe_overrides = redact_env(recorded)
        logger = ctx.logger.bind("subprocess") if ctx is not None else None
        if logger is not None:
            logger.argv(
                "subprocess.start",
                command,
                cwd=str(cwd) if cwd is not None else None,
                env_overrides=safe_overrides,
                timeout=timeout,
            )

        stdout_tail: deque[str] = deque(maxlen=self._tail_lines)
        stderr_tail: deque[str] = deque(maxlen=self._tail_lines)
        started = time.monotonic()

        try:
            # argv list + shell=False: nothing here is ever parsed by a shell.
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                cwd=str(cwd) if cwd is not None else None,
                env=child_env,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                shell=False,
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            duration = time.monotonic() - started
            message = redact_text(str(exc), self._extra_secrets)
            if logger is not None:
                logger.error("subprocess.spawn_failed", message, argv=safe_argv)
            return CompletedCommand(
                argv=safe_argv,
                returncode=SPAWN_FAILURE_RETURNCODE,
                duration_seconds=duration,
                stdout_lines=[],
                stderr_lines=[message],
                env_overrides=safe_overrides,
                cwd=cwd,
                error=message,
            )

        lock = threading.Lock()
        pumps = [
            threading.Thread(
                target=self._pump,
                args=(process.stdout, "stdout", stdout_tail, lock, ctx),
                name="yonga-stdout",
                daemon=True,
            ),
            threading.Thread(
                target=self._pump,
                args=(process.stderr, "stderr", stderr_tail, lock, ctx),
                name="yonga-stderr",
                daemon=True,
            ),
        ]
        for pump in pumps:
            pump.start()

        cancelled = False
        timed_out = False
        try:
            while True:
                try:
                    process.wait(timeout=self._poll_interval)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if ctx is not None and ctx.cancelled.is_set():
                    cancelled = True
                    self._stop(process, logger, "cancellation requested")
                    break
                if timeout is not None and (time.monotonic() - started) > timeout:
                    timed_out = True
                    self._stop(process, logger, f"timed out after {timeout:g}s")
                    break
        except KeyboardInterrupt:
            # Ctrl+C: the child has its own session, so it never saw the signal.
            cancelled = True
            self._stop(process, logger, "interrupted by the user")

        for pump in pumps:
            pump.join(timeout=5.0)

        duration = time.monotonic() - started
        returncode = process.poll()
        if returncode is None:  # pragma: no cover - only after an unkillable child
            returncode = -int(self._grace[-1][0])

        with lock:
            stdout_lines = list(stdout_tail)
            stderr_lines = list(stderr_tail)

        if logger is not None:
            logger.info(
                "subprocess.finished",
                f"exit {returncode}",
                returncode=returncode,
                duration_seconds=round(duration, 3),
                cancelled=cancelled,
                timed_out=timed_out,
            )

        return CompletedCommand(
            argv=safe_argv,
            returncode=int(returncode),
            duration_seconds=duration,
            stdout_lines=stdout_lines,
            stderr_lines=stderr_lines,
            cancelled=cancelled,
            timed_out=timed_out,
            env_overrides=safe_overrides,
            cwd=cwd,
            pid=process.pid,
        )

    # --- internals -----------------------------------------------------
    def _pump(
        self,
        stream: IO[str] | None,
        name: str,
        tail: deque[str],
        lock: threading.Lock,
        ctx: JobContext | None,
    ) -> None:
        """Consume one output stream line by line, redacting as it goes."""
        if stream is None:  # pragma: no cover - PIPE is always requested
            return
        try:
            for raw in stream:
                self._handle_line(raw, name, tail, lock, ctx)
        except (OSError, ValueError):  # pragma: no cover - pipe closed under us
            pass
        finally:
            with contextlib.suppress(OSError, ValueError):
                stream.close()

    def _handle_line(
        self,
        raw: str,
        name: str,
        tail: deque[str],
        lock: threading.Lock,
        ctx: JobContext | None,
    ) -> None:
        """Redact, log, buffer and optionally report one output line."""
        text = redact_text(raw.rstrip("\r\n"), self._extra_secrets)
        with lock:
            tail.append(text)
        if ctx is None:
            return
        with contextlib.suppress(Exception):
            ctx.logger.raw(name, text)
        # A carriage-return redraw arrives as one long line; only its last
        # segment is the current state of the progress bar.
        visible = progress_text(text.rsplit("\r", 1)[-1])
        if not visible:
            return
        kind = self._line_selector(visible)
        if kind is not None:
            ctx.emit(kind, visible, stream=name)

    def _stop(
        self,
        process: subprocess.Popen[str],
        logger: JobLogger | None,
        reason: str,
    ) -> None:
        """Escalate ``SIGINT`` → ``SIGTERM`` → ``SIGKILL`` against the child's group."""
        for number, grace in self._grace:
            if process.poll() is not None:
                return
            self._signal_group(process, number)
            if logger is not None:
                with contextlib.suppress(Exception):
                    logger.info(
                        "subprocess.signalled",
                        f"{reason}: sent {signal.Signals(number).name}",
                        signal=signal.Signals(number).name,
                    )
            if grace <= 0:
                continue
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=grace)
                return

    @staticmethod
    def _signal_group(process: subprocess.Popen[str], number: int) -> None:
        """Signal the child's whole process group, falling back to the child alone."""
        killpg = getattr(os, "killpg", None)
        getpgid = getattr(os, "getpgid", None)
        if killpg is not None and getpgid is not None:
            try:
                killpg(getpgid(process.pid), number)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            else:
                return
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError, ValueError):
            process.send_signal(number)


if TYPE_CHECKING:  # pragma: no cover - static contract checks only

    def _satisfies_result(value: CompletedCommand) -> CommandResult:
        """Assert to mypy that ``CompletedCommand`` implements ``CommandResult``."""
        return value

    def _satisfies_runner(value: SubprocessRunner) -> CommandRunner:
        """Assert to mypy that ``SubprocessRunner`` implements ``CommandRunner``."""
        return value
