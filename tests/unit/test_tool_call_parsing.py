"""Reading tool calls out of a visible answer: the parsers and the streaming gate.

The rules here are built by hand rather than taken from the built-in registry,
so the parsers are tested as the generic, data-driven code they are; the
built-in entries are covered in ``test_registry_tool_calling.py``.
"""

from __future__ import annotations

import json
import random
from itertools import count

import pytest

from yonga.application import tool_calls as module
from yonga.application.tool_calls import (
    ToolCallGate,
    new_call_id,
    parse_tool_calls,
    supported_format,
    tool_names,
)
from yonga.registry.schema import KNOWN_TOOL_CALL_FORMATS, SourceKind, SourceRef, ToolCallingRule

TESTED = [SourceRef(kind=SourceKind.TESTED, note="demo")]
TAGGED = ToolCallingRule(
    format="tagged_json",
    open="<tool_call>",
    close="</tool_call>",
    evidence="tested",
    sources=TESTED,
)
LISTED = ToolCallingRule(
    format="json_list", open="<|tool_call|>", evidence="tested", sources=TESTED
)
NAMES = frozenset({"get_weather", "get_time"})

WEATHER = '{"name": "get_weather", "arguments": {"city": "Ankara"}}'
TIME = '{"name": "get_time", "arguments": {"zone": "Europe/Istanbul"}}'


def ids() -> object:
    numbers = count(1)
    return lambda: f"call_{next(numbers)}"


def parse(text: str, rule: ToolCallingRule = TAGGED, names: frozenset[str] = NAMES):
    return parse_tool_calls(text, rule, names, new_id=ids())  # type: ignore[arg-type]


def test_every_known_format_has_a_parser() -> None:
    assert set(module._PARSERS) == set(KNOWN_TOOL_CALL_FORMATS)
    assert supported_format(TAGGED) and supported_format(LISTED)
    future = TAGGED.model_copy(update={"format": "from_the_future"})
    assert not supported_format(future)
    assert parse("hello", future).content == "hello"


def test_call_ids_are_fresh_and_openai_shaped() -> None:
    first, second = new_call_id(), new_call_id()
    assert first.startswith("call_") and first != second


def test_tool_names_reads_the_offered_functions() -> None:
    tools = [
        {"type": "function", "function": {"name": "get_weather"}},
        {"type": "function", "function": {"name": "get_time", "parameters": {}}},
        {"type": "function"},
    ]
    assert tool_names(tools) == NAMES


# --- tagged_json ------------------------------------------------------------------------


def test_one_tagged_call() -> None:
    parsed = parse(f"<tool_call>\n{WEATHER}\n</tool_call>")

    assert parsed.content == ""
    assert parsed.prefix == ""
    assert not parsed.unparsed
    (call,) = parsed.calls
    assert call.id == "call_1"
    assert call.name == "get_weather"
    assert json.loads(call.arguments) == {"city": "Ankara"}


def test_several_calls_with_text_around_them() -> None:
    text = f"Let me look.\n<tool_call>{WEATHER}</tool_call>\nand\n<tool_call>{TIME}</tool_call> ok"
    parsed = parse(text)

    assert [call.name for call in parsed.calls] == ["get_weather", "get_time"]
    assert [call.id for call in parsed.calls] == ["call_1", "call_2"]
    assert parsed.content == "Let me look.\nand\nok"
    assert parsed.prefix == "Let me look."


def test_an_answer_without_a_marker_is_plain_text() -> None:
    parsed = parse("  It is sunny in Ankara.  ")

    assert parsed.content == "It is sunny in Ankara."
    assert parsed.calls == []
    assert not parsed.unparsed


@pytest.mark.parametrize(
    "text",
    [
        f"<tool_call>{WEATHER}",  # never closed
        "<tool_call>{not json}</tool_call>",
        '<tool_call>["get_weather"]</tool_call>',
        '<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>',  # not offered
        '<tool_call>{"arguments": {"city": "Ankara"}}</tool_call>',  # no name
        '<tool_call>{"name": "get_weather", "arguments": [1]}</tool_call>',
        '<tool_call>{"name": "get_weather", "arguments": "not json"}</tool_call>',
        f"<tool_call>{WEATHER}</tool_call><tool_call>{{oops</tool_call>",  # all or nothing
    ],
)
def test_what_cannot_be_read_as_calls_is_returned_as_text(text: str) -> None:
    parsed = parse(f"Sure. {text}")

    assert parsed.calls == []
    assert parsed.content == f"Sure. {text}"
    assert parsed.prefix == parsed.content
    assert parsed.unparsed


def test_arguments_may_arrive_as_json_text_or_be_missing() -> None:
    encoded = '{"name": "get_weather", "arguments": "{\\"city\\": \\"Ankara\\"}"}'
    missing = '{"name": "get_time"}'
    parsed = parse(f"<tool_call>{encoded}</tool_call><tool_call>{missing}</tool_call>")

    assert [json.loads(call.arguments) for call in parsed.calls] == [{"city": "Ankara"}, {}]


def test_field_names_and_markers_come_from_the_rule() -> None:
    rule = ToolCallingRule(
        format="tagged_json",
        open="[CALL]",
        close="[/CALL]",
        name_field="function",
        arguments_field="parameters",
        evidence="tested",
        sources=TESTED,
    )
    parsed = parse('[CALL]{"function": "get_time", "parameters": {"zone": "UTC"}}[/CALL]', rule)

    (call,) = parsed.calls
    assert (call.name, json.loads(call.arguments)) == ("get_time", {"zone": "UTC"})
    assert parse(f"<tool_call>{WEATHER}</tool_call>", rule).calls == []


def test_non_ascii_arguments_survive() -> None:
    body = '{"name": "get_weather", "arguments": {"city": "İzmir"}}'
    (call,) = parse(f"<tool_call>{body}</tool_call>").calls
    assert "İzmir" in call.arguments


# --- json_list --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        f"[{WEATHER}, {TIME}]",
        f"<|tool_call|>[{WEATHER}, {TIME}]",
        f"<|tool_call|>\n[{WEATHER}, {TIME}]",
    ],
)
def test_a_json_list_answer_is_read_as_calls(text: str) -> None:
    parsed = parse(text, LISTED)

    assert [call.name for call in parsed.calls] == ["get_weather", "get_time"]
    assert parsed.content == ""
    assert not parsed.unparsed


def test_a_json_list_with_a_close_marker() -> None:
    rule = LISTED.model_copy(update={"close": "<|end|>"})
    assert len(parse(f"[{WEATHER}]<|end|>", rule).calls) == 1


def test_prose_is_not_a_json_list() -> None:
    parsed = parse(f"The call would be [{WEATHER}].", LISTED)

    assert parsed.calls == []
    assert not parsed.unparsed


@pytest.mark.parametrize("text", ["[1] A citation.", "[]", f"[{WEATHER},", '[{"name": "x"}]'])
def test_a_list_that_is_not_calls_is_returned_as_text(text: str) -> None:
    parsed = parse(text, LISTED)

    assert parsed.calls == []
    assert parsed.content == text
    assert parsed.unparsed


# --- the streaming gate -----------------------------------------------------------------


def run_gate(rule: ToolCallingRule, pieces: list[str]) -> tuple[list[str], str]:
    """Feed ``pieces``; return what each feed released and what ``finish`` added."""
    gate = ToolCallGate(rule, NAMES)
    released = [gate.feed(piece) for piece in pieces]
    return released, gate.finish()


def chunked(text: str, rng: random.Random) -> list[str]:
    pieces: list[str] = []
    position = 0
    while position < len(text):
        size = rng.randint(1, 7)
        pieces.append(text[position : position + size])
        position += size
    return pieces


TAGGED_ANSWERS = [
    "Plain answer, no calls.",
    f"<tool_call>{WEATHER}</tool_call>",
    f"Let me check. <tool_call>{WEATHER}</tool_call>",
    f"Checking.\n\n<tool_call>{WEATHER}</tool_call>\n<tool_call>{TIME}</tool_call>\nDone.",
    "Sure. <tool_call>{broken</tool_call>",
    "Ends on a partial <tool_",
    "A < sign and <tool_ca and more",
]


@pytest.mark.parametrize("text", TAGGED_ANSWERS)
@pytest.mark.parametrize("seed", range(5))
def test_the_gate_releases_exactly_the_content_however_it_is_chunked(text: str, seed: int) -> None:
    released, rest = run_gate(TAGGED, chunked(text, random.Random(seed)))

    assert "".join(released) + rest == parse(text).content


@pytest.mark.parametrize("seed", range(5))
def test_nothing_of_a_call_is_released_before_the_end(seed: int) -> None:
    text = f"Let me check. <tool_call>{WEATHER}</tool_call> Done."
    released, rest = run_gate(TAGGED, chunked(text, random.Random(seed)))

    early = "".join(released)
    assert early == "Let me check."
    assert "<" not in early and "Ankara" not in early
    assert rest == "\nDone."


def test_a_partial_marker_is_held_until_it_is_settled() -> None:
    gate = ToolCallGate(TAGGED, NAMES)

    assert gate.feed("Hello <tool") == "Hello"
    assert not gate.holding
    assert gate.feed("box> world") == " <toolbox> world"


def test_a_json_list_answer_is_held_from_its_first_character() -> None:
    gate = ToolCallGate(LISTED, NAMES)

    assert gate.feed("[") == ""
    assert gate.holding
    assert gate.feed(WEATHER + "]") == ""
    assert gate.finish() == ""


@pytest.mark.parametrize("text", [f"[{WEATHER}]", "[1] A citation.", "Plain text [x]."])
@pytest.mark.parametrize("seed", range(3))
def test_the_list_gate_releases_exactly_the_content(text: str, seed: int) -> None:
    released, rest = run_gate(LISTED, chunked(text, random.Random(seed)))

    assert "".join(released) + rest == parse(text, LISTED).content


def test_prose_passes_the_list_gate_as_it_arrives() -> None:
    gate = ToolCallGate(LISTED, NAMES)

    assert gate.feed("<|to") == ""  # could still be the marker
    assert gate.feed("ol is a word") == "<|tool is a word"
    assert gate.feed(" [x]") == " [x]"
