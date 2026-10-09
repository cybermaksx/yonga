"""Reading tool calls out of a model's answer, in the formats the registry can select.

A model that was offered tools answers either in prose or with one or more
calls written in its family's own format. Which format that is, is registry
data: a chat profile's :class:`~yonga.registry.schema.ToolCallingRule` names one
of the parsers here (:data:`~yonga.registry.schema.KNOWN_TOOL_CALL_FORMATS`) and
fills in its markers and field names. No family is named in this module.

Everything here runs on the answer with its reasoning removed, after the
reasoning sanitiser, so a call drafted inside a reasoning block is never read
as one; with ``--show-reasoning`` the chat service reads the answer after the
shown reasoning the same way (:func:`yonga.application.chat.split_leading_reasoning`).

The parsers are total and conservative. An answer that cannot be read as calls
to the offered functions -- malformed JSON, an unclosed block, a function that
was not offered, arguments that are not an object -- is returned unchanged as
text, never as a guessed call, and :attr:`ParsedAnswer.unparsed` says it looked
like one. Calls are all-or-nothing for the same reason: one unreadable block
makes the whole answer text.

:class:`ToolCallGate` is the streaming half. Text before a call is released as
it arrives; from the point where a call could begin, everything is held until
generation ends, because a call cannot be sent until it is complete and text
already sent cannot be taken back. The text it releases, joined with what
:meth:`ToolCallGate.finish` returns, is exactly :attr:`ParsedAnswer.content`.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NamedTuple

from yonga.adapters.runtime_genai import ToolCall
from yonga.registry.schema import KNOWN_TOOL_CALL_FORMATS, ToolCallingRule

__all__ = [
    "TOOL_CALLS_REASON",
    "ParsedAnswer",
    "ToolCallGate",
    "new_call_id",
    "parse_tool_calls",
    "supported_format",
    "tool_names",
]

#: ``GenerationResult.stop_reason`` of an answer that ended in tool calls.
TOOL_CALLS_REASON = "tool_calls"

#: What separates the pieces of text around call blocks in the content.
_SEPARATOR = "\n"


class ParsedAnswer(NamedTuple):
    """A visible answer split into its text and the calls it made."""

    content: str
    """The text outside the calls, stripped; the whole answer when it made none."""
    calls: list[ToolCall]
    prefix: str
    """The text in front of the first call (the whole content when there is none).

    A stop sequence that completes in here ended the answer before any call
    began, which is what the chat service needs to know.
    """
    unparsed: bool = False
    """True when the answer began like a call that could not be read as one."""


def new_call_id() -> str:
    """A fresh ``call_…`` identifier for one tool call."""
    return f"call_{uuid.uuid4().hex}"


def tool_names(tools: Iterable[Mapping[str, Any]]) -> frozenset[str]:
    """The function names an OpenAI ``tools`` list offers."""
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            names.add(function["name"])
    return frozenset(names)


def supported_format(rule: ToolCallingRule) -> bool:
    """Whether this build has a parser for ``rule``'s format.

    A registry layer written for a newer yonga may name a format this one lacks;
    ``validate_registry`` reports that, and such a profile has no tool calling.
    """
    return rule.format in KNOWN_TOOL_CALL_FORMATS and rule.format in _PARSERS


def parse_tool_calls(
    text: str,
    rule: ToolCallingRule,
    names: Iterable[str],
    *,
    new_id: Callable[[], str] = new_call_id,
) -> ParsedAnswer:
    """Split ``text``, a visible answer, into content and calls as ``rule`` describes.

    Only calls to ``names``, the functions the request offered, are accepted.
    Anything that cannot be read as such calls comes back as content, whole.
    """
    stripped = text.strip()
    parser = _PARSERS.get(rule.format)
    if parser is None:
        return ParsedAnswer(stripped, [], stripped)
    return parser(stripped, rule, frozenset(names), new_id)


class ToolCallGate:
    """Release a streamed visible answer up to where a tool call could begin.

    It sits after the reasoning sanitiser, whose output carries no leading or
    trailing whitespace, and before a client's stop matcher, which then sees
    only content. Held back are the tail that could still grow into the
    rule's opening marker and the whitespace in front of it; once a call
    section has begun everything is held until :meth:`finish`.
    """

    def __init__(self, rule: ToolCallingRule, names: Iterable[str]) -> None:
        """Gate an answer for ``rule``'s format and the offered function ``names``."""
        self._rule = rule
        self._names = frozenset(names)
        self._text = ""
        self._released = ""
        self._holding = False
        self._passing = False
        self._scanned = 0

    @property
    def holding(self) -> bool:
        """True once a call section has begun and the rest is being held."""
        return self._holding

    def feed(self, piece: str) -> str:
        """Take one piece of visible text; return what may be released now."""
        self._text += piece
        if self._holding:
            return ""
        if self._passing:
            return self._release(self._text)
        if self._rule.format == "json_list":
            return self._feed_list()
        return self._feed_tagged()

    def finish(self) -> str:
        """Return the rest of the content once generation has ended.

        The answer is parsed whole; what is returned completes the released
        text to :attr:`ParsedAnswer.content`. The calls themselves are read
        from the finished answer by the caller.
        """
        content = parse_tool_calls(self._text, self._rule, self._names).content
        if not content.startswith(self._released):
            return ""  # unreachable while the parsers keep the prefix verbatim
        rest = content[len(self._released) :]
        self._released = content
        return rest

    def _feed_tagged(self) -> str:
        opening = self._rule.open or ""
        start = self._text.find(opening, max(0, self._scanned - len(opening)))
        if start >= 0:
            self._holding = True
            return self._release(self._text[:start].rstrip())
        self._scanned = len(self._text)
        held = _partial_suffix(self._text, opening)
        return self._release(self._text[: len(self._text) - held].rstrip())

    def _feed_list(self) -> str:
        text = self._text.lstrip()
        opening = self._rule.open or ""
        starts = [marker for marker in (opening, "[") if marker]
        if any(text.startswith(marker) for marker in starts):
            self._holding = True
            return ""
        if any(marker.startswith(text) for marker in starts):
            return ""  # could still become a marker
        self._passing = True
        return self._release(self._text)

    def _release(self, text: str) -> str:
        if len(text) <= len(self._released) or not text.startswith(self._released):
            return ""
        delta = text[len(self._released) :]
        self._released = text
        return delta


def _partial_suffix(text: str, marker: str) -> int:
    """Length of the longest proper suffix of ``text`` that ``marker`` starts with."""
    for size in range(min(len(marker) - 1, len(text)), 0, -1):
        if marker.startswith(text[-size:]):
            return size
    return 0


# --- parsers ----------------------------------------------------------------------------


def _parse_tagged_json(
    text: str, rule: ToolCallingRule, names: frozenset[str], new_id: Callable[[], str]
) -> ParsedAnswer:
    """Calls as JSON objects between ``open`` and ``close``, text allowed around them."""
    opening, closing = rule.open or "", rule.close or ""
    first = text.find(opening) if opening else -1
    if first < 0 or not closing:
        return ParsedAnswer(text, [], text)
    segments: list[str] = []
    calls: list[ToolCall] = []
    position = 0
    while True:
        start = text.find(opening, position)
        if start < 0:
            segments.append(text[position:])
            break
        segments.append(text[position:start])
        end = text.find(closing, start + len(opening))
        if end < 0:
            return _unparsed(text)
        call = _call(text[start + len(opening) : end], rule, names, new_id)
        if call is None:
            return _unparsed(text)
        calls.append(call)
        position = end + len(closing)
    content = _SEPARATOR.join(part.strip() for part in segments if part.strip())
    return ParsedAnswer(content, calls, text[:first].strip())


def _parse_json_list(
    text: str, rule: ToolCallingRule, names: frozenset[str], new_id: Callable[[], str]
) -> ParsedAnswer:
    """The whole answer is a JSON array of calls, optionally inside ``open`` and ``close``."""
    body = text
    if rule.open and body.startswith(rule.open):
        body = body[len(rule.open) :].strip()
    elif not body.startswith("["):
        return ParsedAnswer(text, [], text)
    if rule.close and body.endswith(rule.close):
        body = body[: -len(rule.close)].strip()
    try:
        items = json.loads(body)
    except (ValueError, RecursionError):
        return _unparsed(text)
    if not isinstance(items, list) or not items:
        return _unparsed(text)
    calls: list[ToolCall] = []
    for item in items:
        call = _from_object(item, rule, names, new_id)
        if call is None:
            return _unparsed(text)
        calls.append(call)
    return ParsedAnswer("", calls, "")


def _unparsed(text: str) -> ParsedAnswer:
    """The answer as text, flagged: it began like a call but is not one."""
    return ParsedAnswer(text, [], text, unparsed=True)


def _call(
    body: str, rule: ToolCallingRule, names: frozenset[str], new_id: Callable[[], str]
) -> ToolCall | None:
    try:
        item = json.loads(body)
    except (ValueError, RecursionError):
        return None
    return _from_object(item, rule, names, new_id)


def _from_object(
    item: object, rule: ToolCallingRule, names: frozenset[str], new_id: Callable[[], str]
) -> ToolCall | None:
    """One call object as a :class:`ToolCall`, or ``None`` when it is not a valid one."""
    if not isinstance(item, dict):
        return None
    name = item.get(rule.name_field)
    if not isinstance(name, str) or name not in names:
        return None
    arguments = item.get(rule.arguments_field, {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except (ValueError, RecursionError):
            return None
    if not isinstance(arguments, dict):
        return None
    return ToolCall(id=new_id(), name=name, arguments=json.dumps(arguments, ensure_ascii=False))


_Parser = Callable[[str, ToolCallingRule, frozenset[str], Callable[[], str]], ParsedAnswer]

#: One parser per format in :data:`~yonga.registry.schema.KNOWN_TOOL_CALL_FORMATS`.
_PARSERS: Mapping[str, _Parser] = {
    "tagged_json": _parse_tagged_json,
    "json_list": _parse_json_list,
}
