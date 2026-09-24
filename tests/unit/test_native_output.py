"""Capture of what native code writes straight to descriptors 1 and 2.

The NPU compiler reports its real reason for rejecting a graph on the native
descriptors, not in the Python exception, so yonga captures them around every
native compile. ``os.write`` stands in for native code here: like a C++ plugin,
it goes straight to the descriptor and bypasses Python's ``sys.stdout``.
"""

from __future__ import annotations

import os

import pytest

from yonga.adapters.native_output import (
    MAX_CAPTURED_BYTES,
    NativeOutput,
    capture_native_output,
    with_native_output,
)


def test_native_writes_to_both_descriptors_are_captured(capfd: pytest.CaptureFixture[str]) -> None:
    with capture_native_output() as native:
        os.write(1, b"[ERROR] [vpux-compiler] native on stdout\n")
        os.write(2, b"[ERROR] native on stderr\n")

    assert native.captured is True
    assert "native on stdout" in native.text
    assert "native on stderr" in native.text
    leaked = capfd.readouterr()
    assert "native on" not in leaked.out
    assert "native on" not in leaked.err


def test_python_output_keeps_reaching_the_terminal(capfd: pytest.CaptureFixture[str]) -> None:
    """Rich and print write through sys.stdout, which must stay on the real stream."""
    with capture_native_output() as native:
        print("python-level progress line")

    assert "python-level progress line" in capfd.readouterr().out
    assert "python-level progress line" not in native.text


def test_descriptors_are_restored_after_the_block(capfd: pytest.CaptureFixture[str]) -> None:
    with capture_native_output():
        os.write(1, b"inside\n")
    os.write(1, b"after the block\n")

    assert "after the block" in capfd.readouterr().out


def test_an_exception_inside_the_block_still_restores_and_captures(
    capfd: pytest.CaptureFixture[str],
) -> None:
    holder: list[NativeOutput] = []
    with pytest.raises(RuntimeError), capture_native_output() as native:
        holder.append(native)
        os.write(2, b"reason before the failure\n")
        raise RuntimeError("compile failed")

    assert "reason before the failure" in holder[0].text
    os.write(1, b"still wired\n")
    assert "still wired" in capfd.readouterr().out


def test_capture_degrades_when_descriptors_cannot_be_duplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(_fd: int) -> int:
        raise OSError("no descriptors")

    monkeypatch.setattr(os, "dup", refuse)
    ran = False
    with capture_native_output() as native:
        ran = True

    assert ran is True
    assert native.captured is False
    assert native.text == ""


def test_only_the_tail_of_a_huge_output_is_kept() -> None:
    with capture_native_output() as native:
        os.write(1, b"x" * (MAX_CAPTURED_BYTES + 4096))
        os.write(1, b"THE-REAL-REASON\n")

    assert len(native.text.encode()) <= MAX_CAPTURED_BYTES
    assert native.text.rstrip().endswith("THE-REAL-REASON")


def test_native_text_is_appended_to_the_error_message() -> None:
    native = NativeOutput(text="  [vpux-compiler] Only supports FP16.\n", captured=True)
    combined = with_native_output("Compilation failed", native)

    assert combined.startswith("Compilation failed")
    assert "--- native output ---" in combined
    assert "Only supports FP16." in combined


def test_an_empty_capture_leaves_the_message_alone() -> None:
    assert with_native_output("Compilation failed", NativeOutput()) == "Compilation failed"
