"""Route third-party library log records into the running job's log.

Some libraries yonga drives in-process attach their own ``stderr`` handler at
import time. ``huggingface_hub`` is the case that matters: it relays the Hub's
``X-HF-Warning`` response headers through it, so an anonymous download prints

    Warning: You are sending unauthenticated requests to the HF Hub. ...

straight into the middle of a live progress line. While a job runs, that text
belongs in the job's JSON Lines log, next to everything else the job did, and
not on the terminal.

:func:`route_library_logs` swaps a library logger's stream handlers for a
bridge, once per process. While a :func:`library_logs_to` block is active the
bridge writes every record into that job logger (redacted, like any other
record); outside one it hands the record to the handlers it replaced, so a
command that runs no job behaves exactly as before. The binding is process-wide
rather than per-thread on purpose: the Hub logs from its download worker
threads, which do not inherit the caller's context.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from yonga.util.jsonlog import (
    LEVEL_DEBUG,
    LEVEL_ERROR,
    LEVEL_INFO,
    LEVEL_WARNING,
    JobLogger,
)

__all__ = [
    "LIBRARY_LOG_EVENT",
    "library_logs_to",
    "route_library_logs",
]

#: Event name of a library record written into the job log.
LIBRARY_LOG_EVENT: str = "library.log"

_LOCK = threading.Lock()
_BOUND: list[JobLogger] = []
_INSTALLED: dict[str, _LibraryBridge] = {}


def _level_name(levelno: int) -> str:
    """Map a stdlib logging level onto the job log's level names."""
    if levelno >= logging.ERROR:
        return LEVEL_ERROR
    if levelno >= logging.WARNING:
        return LEVEL_WARNING
    if levelno >= logging.INFO:
        return LEVEL_INFO
    return LEVEL_DEBUG


def _bound_logger() -> JobLogger | None:
    """The job logger records currently go to, if a job is running."""
    with _LOCK:
        return _BOUND[-1] if _BOUND else None


class _LibraryBridge(logging.Handler):
    """Sends a library's records to the running job's log, or where they went before."""

    def __init__(self, fallback: Iterable[logging.Handler]) -> None:
        super().__init__(logging.NOTSET)
        self._fallback: tuple[logging.Handler, ...] = tuple(fallback)

    def emit(self, record: logging.LogRecord) -> None:
        """Write ``record`` into the job log, or to the replaced handlers outside a job."""
        target = _bound_logger()
        if target is None:
            for handler in self._fallback:
                if record.levelno >= handler.level:
                    handler.handle(record)
            return
        try:
            message = record.getMessage()
        except Exception:  # a malformed %-format must not break the library call
            message = str(record.msg)
        target.log(_level_name(record.levelno), LIBRARY_LOG_EVENT, message, library=record.name)


def route_library_logs(names: Iterable[str]) -> None:
    """Put the job-log bridge in front of each named library logger. Idempotent.

    Only stream handlers are replaced -- the ones that write to the terminal. A
    file handler someone configured on purpose is left alone.
    """
    with _LOCK:
        for name in names:
            if name in _INSTALLED:
                continue
            logger = logging.getLogger(name)
            streams = [
                handler
                for handler in logger.handlers
                if isinstance(handler, logging.StreamHandler)
                and not isinstance(handler, logging.FileHandler)
            ]
            for handler in streams:
                logger.removeHandler(handler)
            bridge = _LibraryBridge(streams)
            logger.addHandler(bridge)
            _INSTALLED[name] = bridge


@contextmanager
def library_logs_to(logger: JobLogger) -> Iterator[None]:
    """Send routed library records into ``logger`` for the duration of the block."""
    with _LOCK:
        _BOUND.append(logger)
    try:
        yield
    finally:
        with _LOCK:
            for index in range(len(_BOUND) - 1, -1, -1):
                if _BOUND[index] is logger:
                    del _BOUND[index]
                    break
