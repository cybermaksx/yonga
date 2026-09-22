"""Structured progress events emitted by application services.

Application services must never call into Rich (or any other renderer) directly.
They emit :class:`ProgressEvent` values into a :class:`ProgressSink`; the UI layer
subscribes and decides how to draw them. Keeping the event vocabulary here — in
``application`` rather than ``ui`` — means a future GUI or web frontend can consume
the same stream without the services importing a terminal library.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from yonga.domain.diagnostics import Diagnostic, Severity

__all__ = [
    "CallbackSink",
    "CollectingSink",
    "NullSink",
    "ProgressEvent",
    "ProgressKind",
    "ProgressSink",
    "null_sink",
]


class ProgressKind(str, Enum):
    """The kind of work an event describes.

    Renderers switch on this to pick a widget (spinner, bar, log line), so the
    set is deliberately small and stable.
    """

    STAGE = "stage"
    """A named pipeline stage started or finished (``current``/``total`` = step N of M)."""

    DOWNLOAD = "download"
    """Bytes or files moving from the Hub into the local cache."""

    EXPORT = "export"
    """Conversion/quantization progress; usually indeterminate."""

    VALIDATE = "validate"
    """IR inventory, tokenizer reads and submodel inspection."""

    REPAIR = "repair"
    """A repair was detected, applied or verified."""

    PROBE = "probe"
    """A compile or smoke-test attempt against a device."""

    HASH = "hash"
    """File hashing during artifact finalization."""

    LOG = "log"
    """A human-readable line worth surfacing (verbose mode)."""


class ProgressEvent(BaseModel):
    """A single observation about work in flight.

    Every field except ``kind`` is optional: a sink must render whatever it is
    given and never assume a field is present.
    """

    model_config = ConfigDict(frozen=True)

    kind: ProgressKind
    message: str = ""
    current: int | None = None
    total: int | None = None
    device: str | None = None
    status: str | None = None
    severity: Severity = Severity.INFO
    diagnostic: Diagnostic | None = None
    data: dict[str, object] = Field(default_factory=dict)

    @property
    def fraction(self) -> float | None:
        """Completion in ``[0.0, 1.0]``, or ``None`` when indeterminate."""
        if self.current is None or not self.total or self.total <= 0:
            return None
        return max(0.0, min(1.0, self.current / self.total))


@runtime_checkable
class ProgressSink(Protocol):
    """Receives progress events. Implementations must never raise."""

    def emit(self, event: ProgressEvent) -> None:
        """Consume one event."""
        ...


class NullSink:
    """Discards every event.

    Used by tests and by non-interactive code paths so services never need a
    ``if sink is not None`` branch.
    """

    def emit(self, event: ProgressEvent) -> None:
        """Ignore the event."""


class CallbackSink:
    """Adapts a plain callable into a :class:`ProgressSink`.

    Exceptions raised by the callback are swallowed: progress reporting must
    never be able to fail a preparation job.
    """

    def __init__(self, callback: Callable[[ProgressEvent], None]) -> None:
        self._callback = callback

    def emit(self, event: ProgressEvent) -> None:
        """Forward the event, ignoring renderer failures."""
        # A broken renderer must never be able to fail a preparation job.
        with suppress(Exception):
            self._callback(event)


class CollectingSink:
    """Records every event in memory. Intended for tests and ``--json`` runs."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        """Append the event to :attr:`events`."""
        self.events.append(event)

    def of_kind(self, kind: ProgressKind) -> list[ProgressEvent]:
        """Return the recorded events matching ``kind``."""
        return [event for event in self.events if event.kind is kind]


def null_sink() -> ProgressSink:
    """Return a sink that discards everything."""
    return NullSink()
