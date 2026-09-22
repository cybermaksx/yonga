"""Per-job JSON Lines logging.

The machine log is the second output channel described in ``SPEC.md``; the Rich
console is the first. Logging is best-effort by design: a failure to write a log
record must never abort a preparation job, and no record may contain a secret.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import IO, Final

from yonga import APP_SLUG
from yonga.util.redaction import redact_argv, redact_text

DEBUG_ENV_VAR: Final[str] = "YONGA_DEBUG_LOGGING"
"""Set this to a truthy value to see a one-off note when logging itself fails."""

RESERVED_KEYS: Final[frozenset[str]] = frozenset(
    {"timestamp", "job_id", "level", "component", "event", "message"}
)
"""Record keys that extra fields may not overwrite."""

LEVEL_DEBUG: Final[str] = "debug"
LEVEL_INFO: Final[str] = "info"
LEVEL_WARNING: Final[str] = "warning"
LEVEL_ERROR: Final[str] = "error"

MAX_RAW_LINE_CHARS: Final[int] = 8192
"""Captured subprocess lines longer than this are truncated in the log."""


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class _Sink:
    """Shared, thread-safe file handle behind a :class:`JobLogger` family."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._handle: IO[str] | None = None
        self._failed = False
        self._notified = False
        self._open()

    def _open(self) -> None:
        try:
            if self.path.parent and str(self.path.parent) not in {"", "."}:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8", errors="replace")
        except OSError as exc:
            self._handle = None
            self._failed = True
            self.note(f"could not open log file {self.path}: {exc}")

    def note(self, message: str) -> None:
        """Emit at most one diagnostic note about the logging subsystem itself."""
        if self._notified:
            return
        self._notified = True
        if not os.environ.get(DEBUG_ENV_VAR, "").strip():
            return
        with contextlib.suppress(Exception):  # pragma: no cover - stderr may be broken
            print(f"{APP_SLUG}: logging disabled: {message}", file=sys.stderr)

    def write(self, line: str) -> None:
        """Append one already-serialised record; swallows every failure."""
        with self._lock:
            handle = self._handle
            if handle is None:
                return
            try:
                handle.write(line)
                handle.write("\n")
                handle.flush()
            except Exception as exc:  # pragma: no cover - disk full, closed fd, ...
                self._handle = None
                self._failed = True
                self.note(f"could not write to {self.path}: {exc}")

    def close(self) -> None:
        """Close the underlying handle; safe to call repeatedly."""
        with self._lock:
            handle = self._handle
            self._handle = None
            if handle is None:
                return
            with contextlib.suppress(Exception):  # pragma: no cover - broken handle
                handle.close()

    @property
    def healthy(self) -> bool:
        """Whether records are still being written."""
        return self._handle is not None and not self._failed


class JobLogger:
    """Write structured JSON Lines records for one job.

    One JSON object per line, with ``timestamp``, ``job_id``, ``level``,
    ``component``, ``event`` and ``message`` keys plus any extra fields
    flattened alongside them. Every string that is written passes through
    :func:`yonga.util.redaction.redact_text`, and no method ever raises.
    """

    def __init__(
        self,
        job_id: str,
        path: Path,
        *,
        component: str = APP_SLUG,
        extra_secrets: Iterable[str] = (),
    ) -> None:
        self.job_id = job_id
        self.component = component
        self._extra_secrets: tuple[str, ...] = tuple(extra_secrets)
        self._sink = _Sink(path)

    @classmethod
    def _from_sink(
        cls,
        job_id: str,
        sink: _Sink,
        component: str,
        extra_secrets: tuple[str, ...],
    ) -> JobLogger:
        """Build a logger that shares an already-open sink."""
        child = cls.__new__(cls)
        child.job_id = job_id
        child.component = component
        child._extra_secrets = extra_secrets
        child._sink = sink
        return child

    @property
    def path(self) -> Path:
        """Path of the JSON Lines file this logger writes to."""
        return self._sink.path

    @property
    def healthy(self) -> bool:
        """Whether the log file is open and accepting records."""
        return self._sink.healthy

    def bind(self, component: str) -> JobLogger:
        """Return a child logger writing to the same file under a new component."""
        return JobLogger._from_sink(self.job_id, self._sink, component, self._extra_secrets)

    def with_secrets(self, extra_secrets: Iterable[str]) -> JobLogger:
        """Return a child logger that also masks the given literal secrets."""
        merged = tuple(dict.fromkeys((*self._extra_secrets, *extra_secrets)))
        return JobLogger._from_sink(self.job_id, self._sink, self.component, merged)

    # --- writing -------------------------------------------------------
    def _clean_text(self, value: str) -> str:
        """Redact one string value."""
        return redact_text(value, self._extra_secrets)

    def _clean_value(self, value: object, depth: int = 0) -> object:
        """Convert a field value into a redacted, JSON-serialisable value."""
        if depth > 8:
            return self._clean_text(str(value))
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            return self._clean_text(value)
        if isinstance(value, Path):
            return self._clean_text(str(value))
        if isinstance(value, Enum):
            return self._clean_value(value.value, depth + 1)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Mapping):
            return {
                self._clean_text(str(key)): self._clean_value(item, depth + 1)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set, frozenset)):
            return [self._clean_value(item, depth + 1) for item in value]
        return self._clean_text(str(value))

    def log(self, level: str, event: str, message: str = "", **fields: object) -> None:
        """Write one record; never raises, whatever the caller passes in."""
        self._emit(level, event, message, fields)

    def _emit(
        self,
        level: str,
        event: str,
        message: str,
        fields: Mapping[str, object],
    ) -> None:
        """Serialise and write one record, swallowing every failure."""
        try:
            record: dict[str, object] = {
                "timestamp": _utc_now(),
                "job_id": self._clean_text(str(self.job_id)),
                "level": str(level).lower(),
                "component": self._clean_text(str(self.component)),
                "event": self._clean_text(str(event)),
                "message": self._clean_text(str(message)),
            }
            for key, value in fields.items():
                name = str(key)
                if name in RESERVED_KEYS:
                    name = f"{name}_field"
                record[name] = self._clean_value(value)
            line = json.dumps(record, ensure_ascii=False, default=str, sort_keys=False)
            self._sink.write(redact_text(line, self._extra_secrets))
        except Exception:  # pragma: no cover - logging must never break the caller
            self._sink.note("record could not be serialised")

    def debug(self, event: str, message: str = "", **fields: object) -> None:
        """Write a ``debug`` level record."""
        self._emit(LEVEL_DEBUG, event, message, fields)

    def info(self, event: str, message: str = "", **fields: object) -> None:
        """Write an ``info`` level record."""
        self._emit(LEVEL_INFO, event, message, fields)

    def warning(self, event: str, message: str = "", **fields: object) -> None:
        """Write a ``warning`` level record."""
        self._emit(LEVEL_WARNING, event, message, fields)

    def error(self, event: str, message: str = "", **fields: object) -> None:
        """Write an ``error`` level record."""
        self._emit(LEVEL_ERROR, event, message, fields)

    def raw(self, stream: str, line: str) -> None:
        """Record one captured subprocess output line."""
        text = str(line).rstrip("\n")
        truncated = len(text) > MAX_RAW_LINE_CHARS
        if truncated:
            text = text[:MAX_RAW_LINE_CHARS]
        self._emit(
            LEVEL_DEBUG,
            "subprocess.output",
            text,
            {"stream": str(stream), "truncated": truncated},
        )

    def argv(self, event: str, argv: Sequence[str], **fields: object) -> None:
        """Record a command line with secret-bearing options masked."""
        self._emit(LEVEL_INFO, event, "", {"argv": redact_argv(argv), **fields})

    # --- lifecycle -----------------------------------------------------
    def close(self) -> None:
        """Close the shared log file; further records are dropped silently."""
        self._sink.close()

    def __enter__(self) -> JobLogger:
        """Enter a context manager that closes the log on exit."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the log when leaving the context."""
        self.close()


def null_logger(job_id: str = "null", *, component: str = APP_SLUG) -> JobLogger:
    """Return a logger that discards everything, for tests and non-job paths."""
    return JobLogger(job_id, Path(os.devnull), component=component)
