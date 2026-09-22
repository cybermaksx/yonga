"""Tests for the Rich progress renderer's line shapes.

These are about what a user actually reads: exactly one stage counter per stage
line, and no bare filename printed with nothing to explain it.
"""

from __future__ import annotations

import io
from typing import Any

from yonga.application.progress import ProgressEvent, ProgressKind
from yonga.ui.console import YongaConsole, build_console
from yonga.ui.progress import RichProgressSink


def make_console(**kwargs: Any) -> tuple[YongaConsole, io.StringIO]:
    """Build a console writing into in-memory buffers, plus its stderr buffer."""
    err = io.StringIO()
    console = YongaConsole(
        out=build_console(file=io.StringIO(), width=100, force_terminal=False, no_color=True),
        err=build_console(file=err, width=100, force_terminal=False, no_color=True),
        env=kwargs.pop("env", {}),
        **kwargs,
    )
    return console, err


def hash_events(names: list[str]) -> list[ProgressEvent]:
    """Build the per-file hashing stream the artifact store emits."""
    return [
        ProgressEvent(
            kind=ProgressKind.HASH,
            message=name,
            current=index,
            total=len(names),
            status="hashing",
        )
        for index, name in enumerate(names, start=1)
    ]


FILES = ["chat_template.jinja", "config.json", "openvino_model.xml"]


# ----------------------------------------------------------------------
# stage counter
# ----------------------------------------------------------------------
def test_a_stage_line_carries_exactly_one_counter() -> None:
    """The renderer owns the ``4/7`` prefix, so the message must not repeat it."""
    console, err = make_console()

    with RichProgressSink(console) as sink:
        sink.emit(ProgressEvent(kind=ProgressKind.STAGE, message="Converting", current=5, total=7))

    line = err.getvalue().strip()
    assert line == "5/7 Converting"
    assert line.count("5/7") == 1


def test_a_stage_message_that_repeats_the_counter_would_be_visible() -> None:
    """Guards the assertion above: the duplication this test forbids is detectable."""
    console, err = make_console()

    with RichProgressSink(console) as sink:
        sink.emit(
            ProgressEvent(kind=ProgressKind.STAGE, message="5/7 Converting", current=5, total=7)
        )

    assert err.getvalue().strip().count("5/7") == 2


def test_a_stage_without_a_position_prints_no_counter() -> None:
    """Services that report a stage with no position get a bare label."""
    console, err = make_console()

    with RichProgressSink(console) as sink:
        sink.emit(ProgressEvent(kind=ProgressKind.STAGE, message="Loading the model"))

    assert err.getvalue().strip() == "Loading the model"


# ----------------------------------------------------------------------
# hashing
# ----------------------------------------------------------------------
def test_hashing_is_summarised_instead_of_naming_every_file() -> None:
    """A bare filename says nothing; the step is announced once, with a count."""
    console, err = make_console()

    with RichProgressSink(console) as sink:
        for event in hash_events(FILES):
            sink.emit(event)

    text = err.getvalue()
    assert "Hashing 3 files" in text
    assert "chat_template.jinja" not in text
    assert text.strip().count("Hashing") == 1


def test_hashing_one_file_is_not_pluralised() -> None:
    """``Hashing 1 file`` — a count is only worth printing if it reads correctly."""
    console, err = make_console()

    with RichProgressSink(console) as sink:
        for event in hash_events(["openvino_model.xml"]):
            sink.emit(event)

    assert "Hashing 1 file" in err.getvalue()
    assert "1 files" not in err.getvalue()


def test_verbose_hashing_names_each_file_with_context() -> None:
    """``--verbose`` earns the per-file detail, and it still reads as a sentence."""
    console, err = make_console(verbose=True)

    with RichProgressSink(console) as sink:
        for event in hash_events(FILES):
            sink.emit(event)

    text = err.getvalue()
    assert "Hashing 3 files" in text
    assert "Hashing chat_template.jinja" in text
    assert "Hashing openvino_model.xml" in text


def test_a_hash_event_without_a_total_still_says_something() -> None:
    """A sink must render whatever it is given, including a count-free event."""
    console, err = make_console()

    with RichProgressSink(console) as sink:
        sink.emit(ProgressEvent(kind=ProgressKind.HASH, message="chat_template.jinja"))

    text = err.getvalue()
    assert "Hashing" in text
    assert "chat_template.jinja" not in text


# ----------------------------------------------------------------------
# IR inventory
# ----------------------------------------------------------------------
def inventory_events() -> list[ProgressEvent]:
    """Build the per-submodel stream ``ProbeService._emit_inventory`` emits."""
    return [
        ProgressEvent(
            kind=ProgressKind.VALIDATE,
            message=f"{name}: {role}",
            status="read",
            data={"role": role, "inputs": "", "outputs": "", "bytes_total": 1},
        )
        for name, role in (
            ("openvino_detokenizer.xml", "detokenizer"),
            ("openvino_model.xml", "language"),
            ("openvino_tokenizer.xml", "tokenizer"),
        )
    ]


def test_the_ir_inventory_does_not_open_with_a_bare_filename() -> None:
    """``yonga probe`` used to start with ``openvino_detokenizer.xml: detokenizer``."""
    console, err = make_console()

    with RichProgressSink(console) as sink:
        for event in inventory_events():
            sink.emit(event)

    lines = [line for line in err.getvalue().splitlines() if line.strip()]
    assert lines[0].strip() == "Inventorying OpenVINO IR"
    assert not any(line.strip().startswith("openvino_") for line in lines)


def test_verbose_still_names_every_submodel() -> None:
    """The per-file detail is kept, not discarded — it just needs ``--verbose``."""
    console, err = make_console(verbose=True)

    with RichProgressSink(console) as sink:
        for event in inventory_events():
            sink.emit(event)

    text = err.getvalue()
    assert "Inventorying OpenVINO IR" in text
    for name in ("openvino_detokenizer.xml", "openvino_model.xml", "openvino_tokenizer.xml"):
        assert name in text


def test_a_validate_event_without_a_role_is_left_alone() -> None:
    """``prepare`` emits plain VALIDATE messages that are already good sentences."""
    console, err = make_console()

    with RichProgressSink(console) as sink:
        sink.emit(
            ProgressEvent(
                kind=ProgressKind.VALIDATE,
                message="Applying registry repairs: ov-tokenizer-precision-string-casing",
            )
        )

    assert "Applying registry repairs" in err.getvalue()
