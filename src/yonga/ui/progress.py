"""Rich renderer for the application's :class:`~yonga.application.progress.ProgressSink`.

Application services emit typed :class:`ProgressEvent` values and never import
Rich. This module is the terminal subscriber: it turns those events into a
determinate bar, an indeterminate spinner, a stage counter or a device-tagged
line, according to what the event actually carries.

Two hard rules apply here:

* it is completely silent in ``--json`` mode, so ``--json`` still emits exactly
  one JSON document;
* :meth:`RichProgressSink.emit` never raises — broken progress rendering must not
  be able to fail a preparation job.
"""

from __future__ import annotations

from contextlib import suppress
from types import TracebackType
from typing import TYPE_CHECKING

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

from yonga.application.progress import ProgressEvent, ProgressKind
from yonga.domain.diagnostics import Severity
from yonga.ui.console import YongaConsole, human_bytes, status_icon, status_style

if TYPE_CHECKING:
    from rich.progress import TaskID

__all__ = ["RichProgressSink"]

#: Kinds that deserve their own live task when the terminal can animate.
_TASK_KINDS: frozenset[ProgressKind] = frozenset(
    {
        ProgressKind.DOWNLOAD,
        ProgressKind.EXPORT,
        ProgressKind.VALIDATE,
        ProgressKind.REPAIR,
        ProgressKind.HASH,
    }
)

#: Human labels used when an event of that kind carries no message.
_KIND_LABELS: dict[ProgressKind, str] = {
    ProgressKind.DOWNLOAD: "Downloading",
    ProgressKind.EXPORT: "Converting",
    ProgressKind.VALIDATE: "Validating",
    ProgressKind.REPAIR: "Repairing",
    ProgressKind.HASH: "Hashing",
    ProgressKind.PROBE: "Probing",
    ProgressKind.STAGE: "Working",
    ProgressKind.LOG: "",
}

#: Event ``status`` values mapped onto a severity for icon/colour selection.
_STATUS_SEVERITY: dict[str, Severity] = {
    "ok": Severity.OK,
    "success": Severity.OK,
    "passed": Severity.OK,
    "done": Severity.OK,
    "started": Severity.INFO,
    "running": Severity.INFO,
    "skipped": Severity.INFO,
    "not_attempted": Severity.UNKNOWN,
    "unknown": Severity.UNKNOWN,
    "warning": Severity.WARNING,
    "failed": Severity.ERROR,
    "error": Severity.ERROR,
}


def _severity_for(event: ProgressEvent) -> Severity:
    """Pick the severity that best describes ``event``."""
    if event.status:
        mapped = _STATUS_SEVERITY.get(event.status.strip().lower())
        if mapped is not None:
            return mapped
    if event.diagnostic is not None:
        return event.diagnostic.severity
    return event.severity


class RichProgressSink:
    """Draws progress events on a terminal. Implements ``ProgressSink``.

    Use it as a context manager so the live display is always torn down::

        with RichProgressSink(console) as sink:
            service.prepare(..., progress=sink)

    Outside a terminal (a pipe, a CI log) no animation is started: stage and
    probe lines are still printed as plain text, which is what a log wants.
    In ``--json`` mode nothing is written at all.
    """

    def __init__(self, console: YongaConsole) -> None:
        """Bind the sink to a console; nothing is drawn until the first event."""
        self.console = console
        self._progress: Progress | None = None
        self._tasks: dict[ProgressKind, TaskID] = {}
        self._announced: set[ProgressKind] = set()
        self._closed = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def __enter__(self) -> RichProgressSink:
        """Enter the context; the live display starts with the first task."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Always stop the live display, even when the job failed."""
        self.close()

    def close(self) -> None:
        """Stop and discard the live display. Safe to call more than once."""
        self._closed = True
        progress, self._progress = self._progress, None
        self._tasks.clear()
        self._announced.clear()
        if progress is not None:
            with suppress(Exception):
                progress.stop()

    # ------------------------------------------------------------------
    # ProgressSink
    # ------------------------------------------------------------------
    def emit(self, event: ProgressEvent) -> None:
        """Render one event. Never raises."""
        try:
            self._emit(event)
        except Exception:
            # Rendering must never be able to fail a preparation job.
            with suppress(Exception):
                self.close()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _emit(self, event: ProgressEvent) -> None:
        """Dispatch one event to the renderer that suits its kind."""
        if self.console.json_mode or self._closed:
            return
        if event.kind is ProgressKind.STAGE:
            self._render_stage(event)
            return
        if event.kind is ProgressKind.PROBE:
            self._render_probe(event)
            return
        if event.kind is ProgressKind.LOG:
            self.console.detail(event.message)
            return
        if event.kind in _TASK_KINDS:
            self._render_task(event)

    def _render_stage(self, event: ProgressEvent) -> None:
        """Print a persistent ``3/7 Preflight`` line and clear any live task."""
        self._finish_tasks()
        label = event.message.strip() or _KIND_LABELS[ProgressKind.STAGE]
        line = Text()
        if event.current is not None and event.total:
            line.append(f"{event.current}/{event.total} ", style="dim")
        line.append(label, style="bold")
        self._write(line)

    def _render_probe(self, event: ProgressEvent) -> None:
        """Print a device-tagged outcome line such as ``GPU  compile ✓``."""
        severity = _severity_for(event)
        line = Text()
        line.append(f"{status_icon(severity)} ", style=status_style(severity))
        if event.device:
            line.append(f"{event.device:<4}", style="bold")
            line.append(" ")
        line.append(event.message.strip() or _KIND_LABELS[ProgressKind.PROBE])
        if event.status and not event.message:
            line.append(f" {event.status}", style=status_style(severity))
        self._write(line)
        if event.diagnostic is not None and severity is Severity.ERROR:
            self.console.diagnostic(event.diagnostic)

    def _render_task(self, event: ProgressEvent) -> None:
        """Update (or create) the live task for this event's kind."""
        description = event.message.strip() or _KIND_LABELS.get(event.kind, event.kind.value)
        if event.kind is ProgressKind.DOWNLOAD and event.total:
            moved = human_bytes(event.current)
            expected = human_bytes(event.total)
            description = f"{description} ({moved} / {expected})"

        if not self.console.animate:
            # No terminal: a bar would be noise, so only announce the phase once.
            if event.kind not in self._announced:
                self._announced.add(event.kind)
                self._write(Text(description, style="dim"))
            elif self.console.verbose:
                self.console.detail(description)
            return

        progress = self._ensure_progress()
        determinate = event.fraction is not None
        task_id = self._tasks.get(event.kind)
        if task_id is None:
            task_id = progress.add_task(
                description,
                total=float(event.total) if determinate and event.total else None,
                completed=int(event.current or 0),
            )
            self._tasks[event.kind] = task_id
            return
        progress.update(
            task_id,
            description=description,
            total=float(event.total) if determinate and event.total else None,
            completed=float(event.current or 0),
        )

    def _ensure_progress(self) -> Progress:
        """Start the live display on first use and return it."""
        if self._progress is None:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console.err,
                transient=True,
                auto_refresh=True,
                # stdout carries results (the --json document); a stderr progress
                # display must not hijack it, so Rich's global redirect stays off.
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._progress.start()
        return self._progress

    def _finish_tasks(self) -> None:
        """Remove every live task so the next stage starts with a clean display."""
        progress = self._progress
        self._tasks.clear()
        self._announced.clear()
        if progress is None:
            return
        with suppress(Exception):
            for task in list(progress.tasks):
                progress.remove_task(task.id)
            # Rich only re-reads the task table on refresh, so force one now:
            # otherwise the next printed line is followed by a stale frame.
            progress.refresh()

    def _write(self, renderable: Text) -> None:
        """Print a line above the live region, or plainly when nothing is live."""
        target = self._progress.console if self._progress is not None else self.console.err
        target.print(renderable, highlight=False)
