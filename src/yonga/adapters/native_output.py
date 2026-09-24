"""Capture what native code writes straight to file descriptors 1 and 2.

OpenVINO's C++ plugins log their most useful diagnostics directly to the
process's standard descriptors, bypassing Python entirely. The NPU compiler is
the case that matters: when it rejects a graph, the Python exception says only

    [NPU_VCL] Compiler returned msg:
    Compilation failed

while the actual reason -- for example
``[vpux-compiler] OneShotBufferizeVPU2VPUIP failed : Only supports FP16.`` --
is written by native code to stdout or stderr (both were observed on the
reference host, depending on how the process was started). Without capturing
it the registry classifier never sees the signature it was written for.

Capturing also protects ``--json``: native lines written to descriptor 1 while
a pipeline builds would otherwise land in front of the one JSON document the
CLI promises on stdout.

While the block runs, descriptors 1 and 2 are pointed at a temporary file, and
``sys.stdout`` / ``sys.stderr`` are pointed at duplicates of the *original*
descriptors, so Python-level output (Rich, logging) keeps reaching the
terminal. Rich resolves ``sys.stdout`` at every write, which is what makes this
work. The capture is process-global and meant to wrap one native call at a time.
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import IO

__all__ = [
    "MAX_CAPTURED_BYTES",
    "NativeOutput",
    "capture_native_output",
    "with_native_output",
]

#: Upper bound on the captured text. The tail is kept: a native failure is
#: reported last, after whatever the plugin printed on the way there.
MAX_CAPTURED_BYTES: int = 256 * 1024


@dataclass
class NativeOutput:
    """What native code wrote while a :func:`capture_native_output` block ran."""

    text: str = ""
    captured: bool = False
    """False when the descriptors could not be redirected and nothing was captured."""


@contextmanager
def capture_native_output() -> Iterator[NativeOutput]:
    """Redirect descriptors 1 and 2 to a buffer for the duration of the block.

    Never raises on its own account: if the descriptors cannot be duplicated
    the block runs uncaptured and :attr:`NativeOutput.captured` stays False.
    """
    result = NativeOutput()
    with tempfile.TemporaryFile() as sink:
        try:
            saved_out = os.dup(1)
        except OSError:
            yield result
            return
        try:
            saved_err = os.dup(2)
        except OSError:
            os.close(saved_out)
            yield result
            return

        python_out, python_err = sys.stdout, sys.stderr
        _flush_python(python_out, python_err)
        terminal_out: IO[str] | None = None
        terminal_err: IO[str] | None = None
        try:
            terminal_out = _reopen(saved_out, python_out)
            terminal_err = _reopen(saved_err, python_err)
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            sys.stdout, sys.stderr = terminal_out, terminal_err
            result.captured = True
            yield result
        finally:
            _flush_c_stdio()
            sys.stdout, sys.stderr = python_out, python_err
            with suppress(OSError):
                os.dup2(saved_out, 1)
            with suppress(OSError):
                os.dup2(saved_err, 2)
            for descriptor in (saved_out, saved_err):
                with suppress(OSError):
                    os.close(descriptor)
            for stream in (terminal_out, terminal_err):
                if stream is not None:
                    with suppress(Exception):
                        stream.close()
            result.text = _read_tail(sink)


def with_native_output(message: str, native: NativeOutput) -> str:
    """Append captured native text to an error message, if there is any."""
    extra = native.text.strip()
    if not extra:
        return message
    return f"{message}\n--- native output ---\n{extra}"


def _reopen(descriptor: int, like: object) -> IO[str]:
    """Open a text stream on a duplicate of ``descriptor`` matching ``like``'s encoding."""
    encoding = getattr(like, "encoding", None) or "utf-8"
    return os.fdopen(os.dup(descriptor), "w", buffering=1, encoding=encoding, errors="replace")


def _flush_python(*streams: object) -> None:
    """Flush Python-level buffers so nothing written before the block is captured."""
    for stream in streams:
        flush = getattr(stream, "flush", None)
        if callable(flush):
            with suppress(Exception):
                flush()


def _flush_c_stdio() -> None:
    """Flush every C stdio stream so buffered native output lands in the sink.

    Once descriptor 1 points at a file, C stdio switches to full buffering, so
    without this a native message could still be sitting in a buffer when the
    descriptor is restored -- and then appear on the terminal instead.
    """
    with suppress(Exception):
        import ctypes

        ctypes.CDLL(None).fflush(None)


def _read_tail(sink: IO[bytes]) -> str:
    """Read at most :data:`MAX_CAPTURED_BYTES` from the end of the sink."""
    try:
        size = sink.seek(0, os.SEEK_END)
        sink.seek(max(0, size - MAX_CAPTURED_BYTES))
        return sink.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
