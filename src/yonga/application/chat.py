"""The chat service: conversation state, generation policy and the reasoning sanitiser.

Three responsibilities live here and nowhere else.

*The sanitiser.* :func:`sanitize_reasoning` is a pure, total function that removes
recognised reasoning wrappers from raw model output. It is the layer that
actually guarantees hidden reasoning never reaches the screen, so it is written
defensively: an **unclosed** opening wrapper hides everything from the opener to
the end of the text, a closing wrapper with no opener hides everything back to
the last displayed boundary, and any internal failure hides the whole answer
rather than risk leaking a hidden block. The one thing it cannot deduce is that
the *prompt* opened a block, leaving a reply that contains no wrapper at all;
that state arrives as ``starts_in_reasoning``, established by
:func:`yonga.adapters.runtime_genai.resolve_starts_in_reasoning` from the
artifact's own chat template. A client's stop sequences are matched after it,
on the visible text only, by :class:`StopMatcher`, and so are tool calls
(:mod:`yonga.application.tool_calls`), in the format the chat profile declares.

*The session.* :class:`ChatSession` owns one conversation: the pipeline is built
once by the injected ``runtime_factory`` and reused for every turn, generation
defaults come from the registry chat profile for the model's architecture
rather than from global constants (``SPEC.md`` section 6.4),
and a failed turn returns a typed :class:`~yonga.domain.diagnostics.Diagnostic`
instead of raising, so the prompt comes back.

*Command execution.* ``yonga.ui.chat_tui`` parses a ``/command`` line; this module
applies it. The split matters because the validation rules and the resulting
state are policy, and policy does not belong in the renderer.

This module imports no renderer, no Hub client and no inference library. The
conversation types are re-exported from ``yonga.adapters.runtime_genai`` rather
than redefined, and the deprecated ``start_chat()``/``finish_chat()`` lifecycle
does not appear here at all.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from yonga.adapters.runtime_genai import (
    DEFAULT_REASONING_WRAPPERS,
    DEFAULT_THINKING_VARIABLES,
    ChatHistory,
    ChatImage,
    ChatMessage,
    ChatProfile,
    ChatProfileLike,
    ChatRuntime,
    GenerationResult,
    Role,
    StreamingChatRuntime,
    ToolCall,
    read_chat_template,
    resolve_starts_in_reasoning,
    resolve_thinking_variable,
)
from yonga.application.progress import ProgressEvent, ProgressKind, ProgressSink, null_sink
from yonga.application.tool_calls import (
    TOOL_CALLS_REASON,
    ParsedAnswer,
    ToolCallGate,
    parse_tool_calls,
    supported_format,
    tool_names,
)
from yonga.config.settings import DevicePolicy, Settings
from yonga.domain.artifacts import ArtifactManifest, manifest_path
from yonga.domain.benchmarks import GenerationSettings
from yonga.domain.diagnostics import (
    Diagnostic,
    FailureCategory,
    Severity,
    YongaError,
    diagnostic,
)
from yonga.domain.hardware import normalise_device
from yonga.domain.models import ModelIdentity, ModelMetadata, PipelineKind
from yonga.registry.resolver import RegistryResolver
from yonga.registry.schema import ChatProfileRule, GenerationDefaults, ToolCallingRule
from yonga.util.redaction import redact_text

__all__ = [
    "DEFAULT_REASONING_WRAPPERS",
    "MAX_OUTPUT_TOKENS",
    "STOP_SEQUENCE_REASON",
    "TEMPERATURE_RANGE",
    "TOP_K_MAX",
    "TOP_P_RANGE",
    "ArtifactStoreLike",
    "ChatAction",
    "ChatCommandName",
    "ChatHistory",
    "ChatImage",
    "ChatMessage",
    "ChatOptions",
    "ChatProfile",
    "ChatProfileLike",
    "ChatProfileView",
    "ChatRuntime",
    "ChatService",
    "ChatSession",
    "ChatSessionDescription",
    "ChatTurn",
    "CommandOutcome",
    "GenerationResult",
    "LeadingReasoning",
    "LoadClassifier",
    "ParsedCommand",
    "Role",
    "RuntimeFactory",
    "SanitizedAnswer",
    "StopMatcher",
    "StreamingSanitizer",
    "TextSink",
    "ToolCall",
    "chat_history_filename",
    "join_after_reasoning",
    "sanitize_answer",
    "sanitize_reasoning",
    "split_leading_reasoning",
    "truncate_at_stop",
]

#: Largest value ``/tokens`` accepts. Mirrors ``yonga.ui.chat_tui.MAX_OUTPUT_TOKENS``;
#: the service validates independently because it, not the parser, owns the policy.
MAX_OUTPUT_TOKENS: int = 1_048_576
#: Inclusive bounds for the sampling temperature.
TEMPERATURE_RANGE: tuple[float, float] = (0.0, 2.0)
#: Inclusive bounds for nucleus sampling.
TOP_P_RANGE: tuple[float, float] = (0.0, 1.0)
#: Largest ``top_k`` accepted; ``0`` disables top-k sampling.
TOP_K_MAX: int = MAX_OUTPUT_TOKENS

_HISTORY_SUFFIX = ".history"
_FALLBACK_DEVICE = "CPU"


# --------------------------------------------------------------------------------------
# the sanitiser
# --------------------------------------------------------------------------------------


class _Scan(NamedTuple):
    """Outcome of one pass over raw model text."""

    segments: list[str]
    blocks: list[str]
    removed: bool
    unterminated: bool


def _normalise_wrappers(wrappers: object) -> list[tuple[str, str]]:
    """Return usable ``(opening, closing)`` literal pairs, dropping anything malformed.

    Never raises: a registry document, a manifest snapshot or a caller may hand
    in something odd, and a malformed wrapper must degrade to "not configured"
    rather than to an exception during a chat turn.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    try:
        items = list(wrappers)  # type: ignore[call-overload]
    except TypeError:
        return pairs
    for item in items:
        if isinstance(item, str):
            continue
        try:
            parts = tuple(item)
        except TypeError:
            continue
        if len(parts) != 2:
            continue
        opening, closing = parts
        if not isinstance(opening, str) or not isinstance(closing, str):
            continue
        if not opening or not closing:
            continue
        pair = (opening, closing)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def _scan_from_inside(text: str, pairs: Sequence[tuple[str, str]]) -> _Scan:
    """Scan text whose first character is already inside a reasoning block.

    The caller has established that the *prompt* opened the block, so there is
    no opener to find in ``text``: everything up to the first closing wrapper is
    the reasoning, and the remainder is scanned normally. With no closing
    wrapper anywhere the block never ended, and the whole text is hidden.

    Splitting at the first closer rather than running the ordinary stack machine
    is what makes this safe: a stray opener *inside* the prompt-opened block
    would otherwise end the leading run and put the reasoning before it on
    screen.
    """
    cut = -1
    token_length = 0
    for _, closing in pairs:
        position = text.find(closing)
        if position < 0:
            continue
        longer = position == cut and len(closing) > token_length
        if cut < 0 or position < cut or longer:
            cut, token_length = position, len(closing)
    if cut < 0:
        return _Scan(
            segments=[],
            blocks=[_without_leading_opener(text, pairs)],
            removed=True,
            unterminated=True,
        )
    rest = _scan(text[cut + token_length :], pairs)
    return _Scan(
        segments=rest.segments,
        blocks=[_without_leading_opener(text[:cut], pairs), *rest.blocks],
        removed=True,
        unterminated=rest.unterminated,
    )


def _without_leading_opener(block: str, pairs: Sequence[tuple[str, str]]) -> str:
    """Drop an opening wrapper the model wrote at the start of a prompt-opened block.

    When a template can open the block in the prompt, ``starts_in_reasoning`` is
    set whenever thinking could not be proved off, and a model that opens the
    block itself then puts its own opener at the start of the reasoning. It is a
    tag, not reasoning, and ``--show-reasoning`` must not print it.
    """
    stripped = block.lstrip()
    for opening, _ in sorted(pairs, key=lambda pair: len(pair[0]), reverse=True):
        if stripped.startswith(opening):
            return stripped[len(opening) :]
    return block


def _scan(
    text: str,
    pairs: Sequence[tuple[str, str]],
    *,
    starts_in_reasoning: bool = False,
) -> _Scan:
    """Split ``text`` into displayable segments and hidden reasoning blocks.

    A single left-to-right pass with a stack of expected closing wrappers, so
    nested and repeated blocks both work. Four rules make it safe rather than
    merely correct:

    * ``starts_in_reasoning`` says the block was opened by the prompt, before
      this text began, and the pass starts inside it;
    * an opening wrapper that is never closed hides everything to end of text;
    * a closing wrapper with no matching opener hides everything back to the end
      of the previous block (the model's chat template can emit the opener
      itself, in which case the generated text starts *inside* the block);
    * only text outside every block is ever placed in a display segment.
    """
    if starts_in_reasoning:
        return _scan_from_inside(text, pairs)
    segments: list[str] = []
    blocks: list[str] = []
    stack: list[str] = []
    removed = False
    index = 0
    run_start = 0
    block_start = 0
    length = len(text)
    all_closings = {closing for _, closing in pairs}

    while index <= length:
        best_position = -1
        best_token = ""
        best_closing = ""
        best_is_opening = False

        for opening, closing in pairs:
            position = text.find(opening, index)
            if position < 0:
                continue
            if best_position < 0 or position < best_position:
                better = True
            elif position == best_position:
                better = len(opening) > len(best_token)
            else:
                better = False
            if better:
                best_position, best_token, best_closing, best_is_opening = (
                    position,
                    opening,
                    closing,
                    True,
                )

        for closing in sorted(set(stack)) if stack else sorted(all_closings):
            position = text.find(closing, index)
            if position < 0:
                continue
            if best_position < 0 or position < best_position:
                better = True
            elif position == best_position:
                better = len(closing) > len(best_token) or (
                    len(closing) == len(best_token) and bool(stack)
                )
            else:
                better = False
            if better:
                best_position, best_token, best_closing, best_is_opening = (
                    position,
                    closing,
                    closing,
                    False,
                )

        if best_position < 0:
            break

        if best_is_opening:
            if not stack:
                segments.append(text[run_start:best_position])
                block_start = best_position + len(best_token)
            stack.append(best_closing)
            index = best_position + len(best_token)
            continue

        if stack:
            while stack:
                if stack.pop() == best_token:
                    break
            if stack:
                index = best_position + len(best_token)
                continue
            blocks.append(text[block_start:best_position])
        else:
            blocks.append(text[run_start:best_position])
        removed = True
        index = best_position + len(best_token)
        run_start = index

    unterminated = bool(stack)
    if stack:
        blocks.append(text[block_start:])
        removed = True
    else:
        segments.append(text[run_start:])
    return _Scan(
        segments=segments,
        blocks=blocks,
        removed=removed,
        unterminated=unterminated,
    )


def _separator(gap: str) -> str:
    """Collapse the whitespace that surrounded a removed block into one separator."""
    if not gap:
        return ""
    newlines = gap.count("\n")
    if newlines >= 2:
        return "\n\n"
    if newlines == 1:
        return "\n"
    return " "


def _join_segments(segments: Sequence[str]) -> str:
    """Join display segments, collapsing the whitespace left where blocks were cut.

    Leading and trailing whitespace of the whole answer is dropped so a removed
    opening block cannot leave the reply starting with blank lines.
    """
    cores: list[str] = []
    gaps: list[str] = []
    pending = ""
    for segment in segments:
        core = segment.strip()
        if not core:
            pending += segment
            continue
        leading = segment[: len(segment) - len(segment.lstrip())]
        if cores:
            gaps.append(pending + leading)
        cores.append(core)
        pending = segment[len(segment.rstrip()) :]
    if not cores:
        return ""
    parts = [cores[0]]
    for position, core in enumerate(cores[1:]):
        parts.append(_separator(gaps[position]))
        parts.append(core)
    return "".join(parts)


class SanitizedAnswer(BaseModel):
    """What one raw model reply looks like after the reasoning policy is applied.

    The three flags describe the *shape* of what was hidden and never its
    content, which is what lets the UI explain an empty answer without being able
    to leak a word of the reasoning it is explaining.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    display_text: str
    """The text that may be shown."""

    reasoning_text: str | None = None
    """The hidden reasoning, reported so ``--show-reasoning`` can render it."""

    reasoning_hidden: bool = False
    """True when a reasoning block was found and removed from ``display_text``."""

    reasoning_unterminated: bool = False
    """True when a reasoning block was opened and never closed.

    The model ran out of tokens mid-thought, which is a different failure from
    having decided to say nothing.
    """

    @property
    def is_empty(self) -> bool:
        """True when nothing at all survived for display."""
        return not self.display_text.strip()


def sanitize_answer(
    text: str,
    wrappers: Sequence[tuple[str, str]],
    *,
    show: bool,
    starts_in_reasoning: bool = False,
) -> SanitizedAnswer:
    """Split raw model output into what to display and what to keep hidden.

    ``wrappers`` are literal ``(opening, closing)`` pairs taken from the registry
    chat profile, never hard-coded here; matching is exact and case-sensitive, so
    ordinary prose containing "think", "thinking" or a literal ``<thinking>`` tag
    that is not a configured wrapper passes through untouched. Text with no block
    at all is returned byte-for-byte; whitespace is only normalised where a block
    was actually removed.

    ``starts_in_reasoning`` is the caller's statement that generation began
    *inside* a reasoning block, because the chat template opened one in the
    prompt. That cannot be worked out from the text: reasoning with no tags and
    a plain answer are the same string. When it is true the text is hidden up to
    the first closing wrapper, and hidden **whole** — with
    ``reasoning_unterminated`` set — when there is no closing wrapper at all.
    The default of false leaves every existing caller's behaviour untouched;
    :func:`yonga.adapters.runtime_genai.resolve_starts_in_reasoning` is what
    establishes the value honestly, and errs toward hiding.

    With ``show=True`` the text is returned unchanged and the reasoning is still
    reported, so ``--show-reasoning`` can render it separately. Nothing is
    *hidden* in that mode, so ``reasoning_hidden`` stays false.

    The function is total. Any unexpected failure is treated as "we cannot prove
    this text is safe" and yields an empty display string rather than risking a
    leaked reasoning block.
    """
    raw = text if isinstance(text, str) else str(text)
    try:
        pairs = _normalise_wrappers(wrappers)
        if not pairs and not starts_in_reasoning:
            return SanitizedAnswer(display_text=raw)
        scan = _scan(raw, pairs, starts_in_reasoning=starts_in_reasoning)
        reasoning_blocks = [block.strip() for block in scan.blocks]
        reasoning = "\n\n".join(block for block in reasoning_blocks if block) or None
        if show:
            return SanitizedAnswer(
                display_text=raw,
                reasoning_text=reasoning,
                reasoning_unterminated=scan.unterminated,
            )
        display = raw if not scan.removed else _join_segments(scan.segments)
        return SanitizedAnswer(
            display_text=display,
            reasoning_text=reasoning,
            reasoning_hidden=scan.removed,
            reasoning_unterminated=scan.unterminated,
        )
    except Exception:  # pragma: no cover - defensive; the scan above is total
        if show:
            return SanitizedAnswer(display_text=raw)
        return SanitizedAnswer(display_text="", reasoning_hidden=True)


def sanitize_reasoning(
    text: str,
    wrappers: Sequence[tuple[str, str]],
    *,
    show: bool,
    starts_in_reasoning: bool = False,
) -> tuple[str, str | None]:
    """Return just ``(display_text, reasoning_text)`` from :func:`sanitize_answer`.

    The two-value form other components were written against; new callers that
    need to tell "said nothing" from "never finished thinking" apart should use
    :func:`sanitize_answer` instead.
    """
    answer = sanitize_answer(text, wrappers, show=show, starts_in_reasoning=starts_in_reasoning)
    return answer.display_text, answer.reasoning_text


class StreamingSanitizer:
    """Apply the reasoning policy to text that arrives a chunk at a time.

    What has been sent cannot be taken back, so a streamed answer may only ever
    be *extended*. Each chunk re-runs :func:`sanitize_answer` over everything
    received so far and releases only the part that can no longer change:

    * a tail that could be the start of an opening wrapper (``<th`` of
      ``<think>``) is held until the next chunk settles it;
    * leading and trailing whitespace is held, because removing a block later
      collapses the whitespace around it, and a space already sent would then
      no longer be a prefix of the answer.

    So nothing inside a reasoning block is ever released, an unterminated block
    releases nothing after its opener, and the concatenated output equals the
    final sanitised answer with its outer whitespace stripped.
    """

    def __init__(
        self,
        wrappers: Sequence[tuple[str, str]],
        *,
        show: bool = False,
        starts_in_reasoning: bool = False,
    ) -> None:
        """Bind the sanitiser to one reply's reasoning policy."""
        self._wrappers = list(wrappers)
        self._openers = [opening for opening, _ in _normalise_wrappers(wrappers) if opening]
        self._show = show
        self._starts_in_reasoning = starts_in_reasoning
        self._raw = ""
        self._released = ""

    @property
    def raw_text(self) -> str:
        """Everything received so far, unsanitised."""
        return self._raw

    def feed(self, chunk: str) -> str:
        """Take one chunk; return the text that may be released now (possibly empty)."""
        self._raw += chunk
        if self._show:
            return self._release(self._raw)
        return self._release(self._without_partial_opener(self._display()))

    def finish(self) -> str:
        """Return whatever is still held once generation has ended."""
        return self._release(self._raw if self._show else self._display())

    def _display(self) -> str:
        return sanitize_answer(
            self._raw,
            self._wrappers,
            show=False,
            starts_in_reasoning=self._starts_in_reasoning,
        ).display_text

    def _without_partial_opener(self, display: str) -> str:
        for opener in self._openers:
            for size in range(len(opener) - 1, 0, -1):
                fragment = opener[:size]
                if self._raw.endswith(fragment) and display.endswith(fragment):
                    return display[: -len(fragment)]
        return display

    def _release(self, text: str) -> str:
        settled = text.strip()
        if not settled.startswith(self._released):
            # Only reachable if the policy rewrote text already released; holding
            # back is the one safe answer, since nothing sent can be retracted.
            return ""
        delta = settled[len(self._released) :]
        self._released = settled
        return delta


#: What separates shown leading reasoning from the answer after it.
_REASONING_SEPARATOR = "\n\n"


def split_leading_reasoning(
    text: str,
    wrappers: Sequence[tuple[str, str]],
    *,
    starts_in_reasoning: bool = False,
) -> tuple[str, str]:
    """Split raw model output into the reasoning that opens it and the rest.

    The leading part ends after the last closing wrapper that still has
    nothing visible in front of it, by :func:`sanitize_answer`'s reading of the
    text; it is empty when the answer opens with visible text, and the whole
    text when nothing in it is visible. With ``--show-reasoning`` the leading
    part is what is shown as reasoning while tool calls are read from the rest
    only, so a call drafted while thinking is never made.
    """
    whole = sanitize_answer(text, wrappers, show=False, starts_in_reasoning=starts_in_reasoning)
    if not whole.display_text.strip():
        return text, ""
    closings = {closing for _, closing in _normalise_wrappers(wrappers)}
    ends: set[int] = set()
    for closing in closings:
        position = text.find(closing)
        while position >= 0:
            ends.add(position + len(closing))
            position = text.find(closing, position + 1)
    lead = 0
    for end in sorted(ends):
        prefix = sanitize_answer(
            text[:end], wrappers, show=False, starts_in_reasoning=starts_in_reasoning
        )
        if prefix.display_text.strip():
            break
        if not prefix.reasoning_unterminated:
            lead = end
    return text[:lead], text[lead:]


def join_after_reasoning(reasoning: str, text: str) -> str:
    """Put shown leading ``reasoning`` in front of ``text``, the answer after it."""
    if not reasoning:
        return text
    return f"{reasoning}{_REASONING_SEPARATOR}{text}" if text else reasoning


class LeadingReasoning:
    """Release the reasoning that opens a shown answer, and pass on the rest.

    The streaming half of :func:`split_leading_reasoning`, for a shown answer
    that is read for tool calls. Raw text is released as reasoning, exactly as
    ``--show-reasoning`` releases it, until something visible appears; the rest
    from there is returned raw, for the caller to sanitise and read as an
    answer, and :meth:`join` puts the separator in front of the first piece of
    it. Released reasoning followed by the joined answer equals
    :func:`join_after_reasoning` of the two halves, however it was chunked.
    """

    def __init__(
        self, wrappers: Sequence[tuple[str, str]], *, starts_in_reasoning: bool = False
    ) -> None:
        """Bind the splitter to one reply's reasoning policy."""
        self._wrappers = list(wrappers)
        self._openers = [opening for opening, _ in _normalise_wrappers(wrappers)]
        self._starts_in_reasoning = starts_in_reasoning
        self._probe = StreamingSanitizer(wrappers, starts_in_reasoning=starts_in_reasoning)
        self._raw = ""
        self._released = ""
        self._settled = False
        self._joined = False

    def feed(self, chunk: str) -> tuple[str, str]:
        """Take one raw chunk; return the reasoning to release and the raw answer after it."""
        if self._settled:
            return "", chunk
        self._raw += chunk
        if self._probe.feed(chunk):
            return self._settle()
        # A tail that could still open a block is held: it may yet be visible text.
        held = max(
            (
                size
                for opener in self._openers
                for size in range(1, len(opener))
                if self._raw.endswith(opener[:size])
            ),
            default=0,
        )
        return self._release(self._raw[: len(self._raw) - held]), ""

    def finish(self) -> tuple[str, str]:
        """Settle the split once generation has ended; return what :meth:`feed` would."""
        return ("", "") if self._settled else self._settle()

    def join(self, text: str) -> str:
        """Prefix the first non-empty piece of the answer with the separator."""
        if not text or self._joined:
            return text
        self._joined = True
        return _REASONING_SEPARATOR + text if self._released else text

    def _settle(self) -> tuple[str, str]:
        self._settled = True
        lead, rest = split_leading_reasoning(
            self._raw, self._wrappers, starts_in_reasoning=self._starts_in_reasoning
        )
        return self._release(lead), rest

    def _release(self, text: str) -> str:
        settled = text.strip()
        if not settled.startswith(self._released):
            return ""  # unreachable: the leading part only ever grows
        delta = settled[len(self._released) :]
        self._released = settled
        return delta


#: ``GenerationResult.stop_reason`` of an answer a client's stop sequence ended.
STOP_SEQUENCE_REASON = "stop_sequence"


def _first_stop(text: str, stops: Sequence[str]) -> int | None:
    """Start of the first stop sequence to complete in ``text``, if one does.

    "First to complete" is the match with the smallest end, ties going to the
    one that starts earlier: that is where generation would have stopped, and
    unlike the earliest *start* it does not depend on how the text was chunked.
    With ``["Observation:", "ion"]`` the answer ends before ``"ion"``, which
    completes while ``"Observation:"`` is still only a partial match.
    """
    found = [
        (index + len(stop), index)
        for index, stop in ((text.find(stop), stop) for stop in stops if stop)
        if index >= 0
    ]
    return min(found)[1] if found else None


def truncate_at_stop(text: str, stops: Sequence[str]) -> tuple[str, bool]:
    """Cut ``text`` before the first stop sequence to complete; return it and whether one did.

    ``text`` is the sanitised visible answer. The cut also drops the whitespace
    in front of the match, exactly as :class:`StopMatcher` releases it, so a
    streamed and an unstreamed answer to the same request read the same.
    """
    index = _first_stop(text, stops)
    if index is None:
        return text, False
    return text[:index].rstrip(), True


class StopMatcher:
    """Find a client's stop sequences in a visible answer that arrives in pieces.

    It sits *after* :class:`StreamingSanitizer`, so it only ever sees text that
    will be shown: a stop sequence inside a hidden reasoning block cannot end
    the answer, which a native GenAI stop string would. Each piece is appended
    to a small held tail and released except for:

    * the longest suffix that is still the start of some stop sequence, which
      the next piece may complete -- that is what finds a match split across
      chunk boundaries;
    * the whitespace in front of that suffix, because a match drops it
      (:func:`truncate_at_stop`) and released text cannot be taken back.

    On a match the text before it is released, :attr:`hit` turns true, and every
    later piece is swallowed. A match is found as soon as it completes, which is
    why the first stop sequence to *complete* is the one that counts. The
    concatenated output equals ``truncate_at_stop(answer, stops)[0]`` for the
    concatenated input, however it was chunked.
    """

    def __init__(self, stops: Sequence[str]) -> None:
        """Match ``stops``; empty strings are ignored."""
        self._stops = [stop for stop in stops if stop]
        self._held = ""
        self._hit = False

    @property
    def hit(self) -> bool:
        """True once a stop sequence matched."""
        return self._hit

    def feed(self, piece: str) -> str:
        """Take one piece of visible text; return what may be released now."""
        if self._hit:
            return ""
        text = self._held + piece
        index = _first_stop(text, self._stops)
        if index is not None:
            self._hit = True
            self._held = ""
            return text[:index].rstrip()
        cut = len(text) - self._partial_suffix(text)
        cut = len(text[:cut].rstrip())
        self._held = text[cut:]
        return text[:cut]

    def finish(self) -> str:
        """Return whatever is still held once the answer has ended without a match."""
        held, self._held = ("" if self._hit else self._held), ""
        return held

    def _partial_suffix(self, text: str) -> int:
        """Length of the longest suffix of ``text`` that starts some stop sequence."""
        longest = max((len(stop) for stop in self._stops), default=1) - 1
        for size in range(min(longest, len(text)), 0, -1):
            tail = text[-size:]
            if any(stop.startswith(tail) for stop in self._stops):
                return size
        return 0


class _SplitStopMatcher:
    """Match stop sequences in shown leading reasoning and in the answer apart.

    :class:`LeadingReasoning` joins the two with a separator of yonga's own,
    which the model never wrote, so a stop such as ``"\\n\\n"`` must not match
    it. Each half gets its own :class:`StopMatcher`, and the separator is put in
    front of the answer only after matching: a stop the model writes in the
    shown reasoning still ends the answer there, and one it writes in the
    answer ends it there. The concatenated output equals
    :func:`join_after_reasoning` of the two halves, each cut by
    :func:`truncate_at_stop` -- an answer is dropped when the reasoning was cut.
    """

    def __init__(self, leading: LeadingReasoning, stops: Sequence[str]) -> None:
        """Match ``stops`` on what ``leading`` splits; empty strings are ignored."""
        self._leading = leading
        self._reasoning = StopMatcher(stops)
        self._answer = StopMatcher(stops)

    @property
    def hit(self) -> bool:
        """True once a stop sequence matched in either half."""
        return self._reasoning.hit or self._answer.hit

    def feed(self, reasoning: str, answer: str) -> str:
        """Take released reasoning and the visible answer after it; return what to release."""
        released = self._reasoning.feed(reasoning) if reasoning else ""
        if answer and not self._reasoning.hit:
            # No more reasoning follows once the answer has begun.
            released += self._reasoning.finish()
            released += self._leading.join(self._answer.feed(answer))
        return released

    def finish(self) -> str:
        """Return whatever is still held once the answer has ended without a match."""
        released = self._reasoning.finish()
        if self._reasoning.hit:
            return released
        return released + self._leading.join(self._answer.finish())


#: Receives each piece of a streamed answer; returning ``True`` asks to stop.
TextSink = Callable[[str], bool | None]


# --------------------------------------------------------------------------------------
# injected ports
# --------------------------------------------------------------------------------------


@runtime_checkable
class RuntimeFactory(Protocol):
    """Builds a loaded :class:`~yonga.adapters.runtime_genai.ChatRuntime`.

    ``yonga.adapters.runtime_genai.open_runtime`` satisfies this by name and by
    signature; the CLI injects it, and tests inject a fake. The session calls it
    at most once, so an implementation may take as long as a compile needs.
    """

    def __call__(
        self,
        artifact_dir: Path,
        *,
        device: str,
        pipeline: PipelineKind,
        properties: Mapping[str, str] | None,
        chat_profile: ChatProfileLike | None,
        settings: Settings | None,
        thinking: bool | None,
    ) -> ChatRuntime:
        """Open a chat runtime for one prepared artifact."""
        ...


@runtime_checkable
class LoadClassifier(Protocol):
    """Turns raw runtime-load text into a registry diagnostic, or ``None``.

    Keeps :class:`ChatSession` free of a registry import while still letting a
    failed compile be explained by the same rules ``yonga probe`` uses.
    """

    def __call__(self, raw: str, device: str) -> Diagnostic | None:
        """Classify ``raw`` for ``device``; ``None`` means "nothing to add"."""
        ...


@runtime_checkable
class ArtifactStoreLike(Protocol):
    """The one thing the chat service needs from the artifact store."""

    def resolve(self, reference: str) -> Path:
        """Return the prepared artifact directory for a model id, artifact id or path.

        Raises a :class:`~yonga.domain.diagnostics.YongaError` when nothing
        matches; the chat service lets that propagate, because a session that
        cannot find its model has nothing to fall back to.
        """
        ...


@runtime_checkable
class ParsedCommand(Protocol):
    """A ``/command`` line already parsed by the UI.

    ``yonga.ui.chat_tui.ChatCommand`` satisfies this structurally, which is what
    keeps this module from importing the renderer. Only read-only access is
    required, and ``kind`` is read as a string so the UI's enum stays private to
    the UI.
    """

    @property
    def kind(self) -> str:
        """Canonical command name, such as ``tokens`` or ``top-p``."""
        ...

    @property
    def error(self) -> str | None:
        """Why the parser rejected the line, when it did."""
        ...

    @property
    def int_value(self) -> int | None:
        """The whole-number argument, when the command takes one."""
        ...

    @property
    def float_value(self) -> float | None:
        """The decimal argument, when the command takes one."""
        ...

    @property
    def bool_value(self) -> bool | None:
        """The ``on``/``off`` argument, when the command takes one."""
        ...


# --------------------------------------------------------------------------------------
# chat profile view
# --------------------------------------------------------------------------------------


def _default_wrappers() -> list[tuple[str, str]]:
    """Return a fresh copy of the fallback reasoning wrappers."""
    return [*DEFAULT_REASONING_WRAPPERS]


class ChatProfileView(BaseModel):
    """The chat policy one session runs under, normalised from its source.

    A registry :class:`~yonga.registry.schema.ChatProfileRule` is the preferred
    source; the artifact manifest's chat snapshot and a conservative built-in
    default are the fallbacks. Keeping one shape means the session never asks
    where its policy came from, and the view satisfies
    :class:`~yonga.adapters.runtime_genai.ChatProfileLike` so it can be handed
    straight to the runtime factory.
    """

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    summary: str = ""
    supports_thinking_flag: bool | None = None
    """``True`` declares a thinking flag, ``False`` vetoes one, ``None`` means detect."""

    thinking_template_variable: str | None = None
    thinking_variable_candidates: list[str] = Field(
        default_factory=lambda: [*DEFAULT_THINKING_VARIABLES]
    )
    default_thinking: bool = False
    reasoning_wrappers: list[tuple[str, str]] = Field(default_factory=_default_wrappers)
    defaults: GenerationDefaults = Field(default_factory=GenerationDefaults)
    thinking_defaults: GenerationDefaults = Field(default_factory=GenerationDefaults)
    stop_strings: list[str] = Field(default_factory=list)
    tool_calling: ToolCallingRule | None = None
    """How the family writes tool calls, when the registry has measured it."""

    @classmethod
    def from_rule(cls, rule: ChatProfileRule) -> ChatProfileView:
        """Build the view from a resolved registry chat profile."""
        return cls(
            id=rule.id,
            summary=rule.summary,
            supports_thinking_flag=rule.supports_thinking_flag,
            thinking_template_variable=rule.thinking_template_variable,
            thinking_variable_candidates=list(rule.thinking_variable_candidates),
            default_thinking=rule.default_thinking,
            reasoning_wrappers=list(rule.reasoning_wrappers) or _default_wrappers(),
            defaults=rule.defaults.model_copy(deep=True),
            thinking_defaults=rule.thinking_defaults.model_copy(deep=True),
            stop_strings=list(rule.stop_strings),
            tool_calling=rule.tool_calling.model_copy(deep=True) if rule.tool_calling else None,
        )

    @classmethod
    def from_snapshot(cls, data: Mapping[str, object]) -> ChatProfileView | None:
        """Build the view from a manifest's recorded chat profile, or ``None`` if unusable."""
        if not data:
            return None
        try:
            view = cls.model_validate(dict(data))
        except Exception:
            return None
        if not view.reasoning_wrappers:
            view.reasoning_wrappers = _default_wrappers()
        return view

    def defaults_for(self, *, thinking: bool) -> GenerationDefaults:
        """Return the generation defaults for thinking or non-thinking mode."""
        if thinking:
            return self.defaults.merged(self.thinking_defaults)
        return self.defaults.model_copy(deep=True)

    def generation_settings(
        self,
        *,
        thinking: bool,
        fallback_max_new_tokens: int,
    ) -> GenerationSettings:
        """Turn this profile's defaults into runnable generation settings.

        ``do_sample`` is inferred when the profile does not state it: a family
        that documents a temperature, a nucleus or a top-k expects sampling, and
        without the flag OpenVINO GenAI would silently decode greedily.

        The profile's ``stop_strings`` travel with the settings as model-level
        end markers, which the runtime applies natively (see
        :func:`~yonga.adapters.runtime_genai.build_generation_config`).
        """
        values = self.defaults_for(thinking=thinking)
        maximum = values.max_new_tokens or fallback_max_new_tokens
        return GenerationSettings(
            max_new_tokens=max(1, int(maximum)),
            temperature=values.temperature,
            top_p=values.top_p,
            top_k=values.top_k,
            do_sample=_infer_do_sample(values),
            repetition_penalty=values.repetition_penalty,
            stop_strings=[text for text in self.stop_strings if text],
        )


def _infer_do_sample(values: GenerationDefaults) -> bool:
    """Decide whether a profile's defaults describe sampling or greedy decoding."""
    if values.do_sample is not None:
        return values.do_sample
    if values.temperature is not None and values.temperature > 0.0:
        return True
    if values.top_p is not None and 0.0 < values.top_p < 1.0:
        return True
    return values.top_k is not None and values.top_k > 0


# --------------------------------------------------------------------------------------
# session description, options and turn results
# --------------------------------------------------------------------------------------


class ChatSessionDescription(BaseModel):
    """Everything the startup panel and ``/model`` display about a session.

    The field names match ``yonga.ui.chat_tui.ChatSessionInfo`` exactly, so the
    CLI renders a session with
    ``ChatSessionInfo.model_validate(description.model_dump())`` without this
    module ever importing the renderer.
    """

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    display_name: str
    artifact_id: str = ""
    repo_id: str | None = None
    revision: str | None = None
    device: str = "AUTO"
    device_full_name: str | None = None
    profile: str | None = None
    max_new_tokens: int | None = None
    show_reasoning: bool = False
    compile_cache_enabled: bool = False
    compile_cache_path: Path | None = None
    artifact_path: Path | None = None
    extra: dict[str, str] = Field(default_factory=dict)


class ChatOptions(BaseModel):
    """Command-line overrides applied when a session opens.

    Every field defaults to ``None`` meaning "use the registry chat profile, the
    artifact manifest or the configuration", in that order.
    """

    model_config = ConfigDict(extra="forbid")

    device: str | None = None
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    show_reasoning: bool | None = None
    thinking: bool | None = None
    system_prompt: str | None = None
    stats: bool = True
    display_name: str | None = None


class ChatTurn(BaseModel):
    """The outcome of one ``send``: a result, a diagnostic, or a cancelled reply.

    A failed turn is a value, not an exception: the chat loop prints the
    diagnostic and returns to the prompt with the session and its loaded pipeline
    intact. A turn the user stopped carries neither: ``cancelled`` is set and
    nothing of it was kept.
    """

    model_config = ConfigDict(extra="forbid")

    result: GenerationResult | None = None
    diagnostic: Diagnostic | None = None
    cancelled: bool = False

    @property
    def ok(self) -> bool:
        """True when the model answered."""
        return self.result is not None and self.diagnostic is None

    @property
    def display_text(self) -> str:
        """The sanitised answer, or an empty string after a failure."""
        return self.result.display_text if self.result is not None else ""

    @property
    def reasoning_text(self) -> str | None:
        """The hidden reasoning, when the model emitted any."""
        return self.result.reasoning_text if self.result is not None else None

    @property
    def reasoning_hidden(self) -> bool:
        """True when a reasoning block was removed from the answer."""
        return self.result.reasoning_hidden if self.result is not None else False

    @property
    def reasoning_unterminated(self) -> bool:
        """True when the model opened a reasoning block and never closed it."""
        return self.result.reasoning_unterminated if self.result is not None else False

    @property
    def output_tokens(self) -> int | None:
        """How many tokens the model generated, when the runtime reported it."""
        return self.result.output_tokens if self.result is not None else None


class ChatAction(str, Enum):
    """What the chat loop should do after a command was applied."""

    NONE = "none"
    """Nothing beyond printing the outcome message."""

    HELP = "help"
    """Print the command list."""

    CLEAR = "clear"
    """Clear the screen."""

    MODEL = "model"
    """Print the session description again."""

    EXIT = "exit"
    """Leave the chat session."""


class ChatCommandName(str, Enum):
    """Canonical ``/command`` names, mirroring ``yonga.ui.chat_tui.ChatCommandKind``."""

    HELP = "help"
    RESET = "reset"
    CLEAR = "clear"
    TOKENS = "tokens"
    TEMPERATURE = "temp"
    TOP_P = "top-p"
    TOP_K = "top-k"
    REASONING = "reasoning"
    STATS = "stats"
    MODEL = "model"
    EXIT = "exit"
    UNKNOWN = "unknown"


class CommandOutcome(BaseModel):
    """What applying one ``/command`` changed, and what the loop should do next."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: ChatAction = ChatAction.NONE
    message: str = ""
    ok: bool = True

    @property
    def exit_requested(self) -> bool:
        """True when the user asked to leave the session."""
        return self.action is ChatAction.EXIT


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def chat_history_filename(stem: str) -> str:
    """Return the prompt-history file name for a session identity.

    Matches ``yonga.ui.chat_tui.ChatSessionInfo.history_filename`` so the service
    and the view agree on one file under the XDG data directory.
    """
    cleaned = "".join(
        character if character.isalnum() or character in "-._" else "-" for character in stem
    ).strip("-")
    return f"{cleaned or 'session'}{_HISTORY_SUFFIX}"


def _command_name(kind: object) -> str:
    """Normalise a command kind, whether it arrives as an enum member or a string."""
    value = getattr(kind, "value", kind)
    return str(value).strip().lower().replace("_", "-")


def _finite_float(value: float) -> bool:
    """True when ``value`` is a real, finite number."""
    return value == value and value not in (float("inf"), float("-inf"))


def _empty_prompt_diagnostic() -> Diagnostic:
    """Diagnostic returned when the user pressed enter on an empty line."""
    return diagnostic(
        "chat.prompt.empty",
        FailureCategory.CONFIGURATION_ERROR,
        "Nothing to send",
        "Type a message, or /help to see the available commands.",
        severity=Severity.WARNING,
    )


def _tools_unsupported_diagnostic(reason: str | None) -> Diagnostic:
    """The diagnostic for a conversation offering tools to a model that takes none."""
    return diagnostic(
        "chat.tools_unsupported",
        FailureCategory.CONFIGURATION_ERROR,
        "This model takes no tools",
        reason or "Its chat profile declares no tool-call format.",
        remediation=["Run `yonga registry explain` for the model to see its chat profile."],
    )


def _resolve_tool_calling(
    profile: ChatProfileView, *, template: str | None
) -> tuple[ToolCallingRule | None, str | None]:
    """The tool-call format a session uses, or ``None`` and why there is none.

    Three things must hold: the chat profile declares a measured format, this
    build has its parser, and the artifact's own chat template contains the
    profile's ``template_contains`` literal when it names one. The last one is
    evidence about *this* artifact, as the thinking variable is: a family rule
    must not switch tools on for a model whose template asks for another format.
    """
    rule = profile.tool_calling
    name = profile.id or "the built-in default"
    if rule is None:
        return None, f"Its chat profile ({name}) declares no tool-call format."
    if not supported_format(rule):
        return None, (
            f"Its chat profile ({name}) names the tool-call format {rule.format!r}, "
            "which this yonga does not have."
        )
    needed = rule.template_contains
    if needed and (template is None or needed not in template):
        return None, (
            f"Its chat profile ({name}) expects {needed!r} in the chat template, "
            "and this artifact's template does not contain it."
        )
    return rule, None


def _unexpected_diagnostic(error: BaseException, *, device: str) -> Diagnostic:
    """Wrap an untyped failure from a runtime so the turn can still report it."""
    detail = redact_text(f"{type(error).__name__}: {error}")
    return diagnostic(
        "chat.generation.unexpected_error",
        FailureCategory.UNKNOWN_RUNTIME_ERROR,
        f"Generation failed on {device}",
        detail,
        remediation=[
            "Try the same prompt again, or /reset to start a fresh conversation.",
            "Run `yonga doctor` if the device stopped responding.",
        ],
        raw_excerpt=detail,
        context={"device": device},
    )


# --------------------------------------------------------------------------------------
# the session
# --------------------------------------------------------------------------------------


class ChatSession:
    """One conversation with one prepared artifact.

    The runtime is constructed on first use and then reused for every turn, so a
    long session pays the compile and load cost exactly once. Nothing here
    renders: the session returns values, and the CLI draws them.
    """

    def __init__(
        self,
        *,
        artifact_dir: Path,
        manifest: ArtifactManifest | None,
        profile: ChatProfileView,
        settings: Settings,
        runtime_factory: RuntimeFactory,
        device: str,
        pipeline: PipelineKind,
        properties: Mapping[str, str] | None = None,
        options: ChatOptions | None = None,
        progress: ProgressSink | None = None,
        classifier: LoadClassifier | None = None,
    ) -> None:
        """Bind a session to an artifact, a chat profile and a runtime factory.

        Nothing is loaded here: :meth:`load` (or the first :meth:`send`) builds
        the pipeline. ``yonga chat`` calls :meth:`load` before it draws the
        startup panel, so a model that cannot load fails the command with its
        own exit code instead of failing every message.

        ``classifier`` turns raw runtime text into a registry diagnostic. It is
        optional so a session can be built without a registry, and when it is
        absent a load failure is reported exactly as the runtime described it.
        """
        chosen = options if options is not None else ChatOptions()
        self._artifact_dir = Path(artifact_dir)
        self._manifest = manifest
        self._profile = profile
        self._settings = settings
        self._runtime_factory = runtime_factory
        self._device = normalise_device(device) or _FALLBACK_DEVICE
        self._pipeline = pipeline
        self._properties = dict(properties or {})
        self._options = chosen
        self._progress = progress if progress is not None else null_sink()
        self._classifier = classifier

        self._chat_template = read_chat_template(self._artifact_dir)
        self._thinking_variable = resolve_thinking_variable(profile, template=self._chat_template)
        self._thinking = _resolve_thinking(
            profile,
            chosen,
            thinking_available=self._thinking_variable is not None,
        )
        self._starts_in_reasoning = resolve_starts_in_reasoning(
            profile,
            template=self._chat_template,
            thinking=self._thinking,
        )
        self._tool_calling, self._tool_calling_refusal = _resolve_tool_calling(
            profile, template=self._chat_template
        )
        self._show_reasoning = (
            chosen.show_reasoning if chosen.show_reasoning is not None else settings.show_reasoning
        )
        self._stats = chosen.stats
        self._generation = profile.generation_settings(
            thinking=self._thinking,
            fallback_max_new_tokens=settings.max_new_tokens,
        )
        self._apply_option_overrides(chosen)

        self.history = ChatHistory()
        system = (chosen.system_prompt or "").strip()
        if system:
            self.history.add_system(system)

        self._runtime: ChatRuntime | None = None
        self._runtime_loads = 0
        self._closed = False

    # --- construction helpers -------------------------------------------

    def _apply_option_overrides(self, options: ChatOptions) -> None:
        """Apply the command-line overrides on top of the profile defaults."""
        if options.max_new_tokens is not None:
            self.set_tokens(options.max_new_tokens)
        if options.temperature is not None:
            self.set_temperature(options.temperature)
        if options.top_p is not None:
            self.set_top_p(options.top_p)
        if options.top_k is not None:
            self.set_top_k(options.top_k)

    # --- introspection ---------------------------------------------------

    @property
    def artifact_dir(self) -> Path:
        """The prepared artifact this session runs."""
        return self._artifact_dir

    @property
    def manifest(self) -> ArtifactManifest | None:
        """The artifact manifest, when one could be read."""
        return self._manifest

    @property
    def profile(self) -> ChatProfileView:
        """The chat profile the session's defaults came from."""
        return self._profile

    @property
    def device(self) -> str:
        """The OpenVINO device the runtime was asked for."""
        return self._device

    @property
    def pipeline(self) -> PipelineKind:
        """Whether this artifact is driven as an LLM or a VLM."""
        return self._pipeline

    @property
    def generation_settings(self) -> GenerationSettings:
        """A copy of the settings the next turn will use."""
        return self._generation.model_copy(deep=True)

    @property
    def reasoning_wrappers(self) -> list[tuple[str, str]]:
        """The wrapper pairs the sanitiser strips for this model family."""
        return list(self._profile.reasoning_wrappers)

    @property
    def show_reasoning(self) -> bool:
        """True when reasoning blocks are rendered instead of hidden."""
        return self._show_reasoning

    @property
    def thinking(self) -> bool:
        """True when the chat template was asked to enable thinking for this session."""
        return self._thinking

    @property
    def thinking_variable(self) -> str | None:
        """The chat-template variable this session drives, when the artifact has one.

        ``None`` means the artifact's own template references no variable this
        profile knows about, so no template variable is sent and the reasoning
        sanitiser is the only defence in play.
        """
        return self._thinking_variable

    @property
    def starts_in_reasoning(self) -> bool:
        """True when this artifact's chat template opens a reasoning block in the prompt.

        The reply then begins inside that block and carries no opening wrapper of
        its own, so the sanitiser is told to hide until the first closing wrapper
        — and to hide the whole reply when none arrives. It is true whenever the
        template can emit an opener and thinking could not be *proved* off, which
        is the direction ``SPEC.md`` section 6.4 requires; ``--show-reasoning``
        and ``/reasoning on`` both undo it.
        """
        return self._starts_in_reasoning

    @property
    def tool_calling(self) -> ToolCallingRule | None:
        """How this model writes tool calls, or ``None`` when it takes no tools.

        The chat profile's measured format, provided this build has its parser
        and the artifact's own chat template contains what the profile says it
        must (see :func:`_resolve_tool_calling`).
        """
        return self._tool_calling

    @property
    def tool_calling_refusal(self) -> str | None:
        """Why this model takes no tools, in a sentence; ``None`` when it takes them."""
        return self._tool_calling_refusal

    @property
    def stats(self) -> bool:
        """True when the metrics footer should be drawn."""
        return self._stats

    @property
    def runtime_loads(self) -> int:
        """How many times a runtime was constructed; one for the life of a session."""
        return self._runtime_loads

    @property
    def closed(self) -> bool:
        """True once :meth:`close` has run."""
        return self._closed

    @property
    def history_path(self) -> Path:
        """Where this session's prompt history lives, under the XDG data directory."""
        return self._settings.paths.chat_history_dir / chat_history_filename(self._identity_stem())

    def _identity_stem(self) -> str:
        """The most stable identity available for naming per-session files."""
        manifest = self._manifest
        if manifest is not None and manifest.artifact_id:
            return manifest.artifact_id
        if manifest is not None and manifest.requested_model.repo_id:
            return manifest.requested_model.repo_id
        return self.display_name

    @property
    def display_name(self) -> str:
        """Human-facing model name for the startup panel."""
        if self._options.display_name:
            return self._options.display_name
        manifest = self._manifest
        if manifest is not None and manifest.requested_model.repo_id:
            return manifest.requested_model.repo_id.rsplit("/", 1)[-1]
        return self._artifact_dir.name

    # --- lifecycle -------------------------------------------------------

    def load(self) -> ChatRuntime:
        """Build the runtime if it does not exist yet, and return it.

        Calling this repeatedly is free: the pipeline is created exactly once per
        session and reused for every turn.
        """
        if self._closed:
            raise YongaError(
                diagnostic(
                    "chat.session.closed",
                    FailureCategory.CONFIGURATION_ERROR,
                    "This chat session has been closed",
                    "Open a new session before sending another message.",
                )
            )
        runtime = self._runtime
        if runtime is None:
            self._emit(ProgressKind.STAGE, f"Loading the model on {self._device}...")
            try:
                runtime = self._runtime_factory(
                    self._artifact_dir,
                    device=self._device,
                    pipeline=self._pipeline,
                    properties=dict(self._properties),
                    chat_profile=self._profile,
                    settings=self._settings,
                    thinking=self._thinking,
                )
            except YongaError as error:
                raise self._classified(error) from error
            self._runtime = runtime
            self._runtime_loads += 1
        return runtime

    def _classified(self, error: YongaError) -> YongaError:
        """Re-report a load failure through the compatibility registry.

        The runtime adapter cannot reach the registry, so on its own it can only
        say "could not load the model" and suggest checking the driver — which is
        the opposite of the truth when the NPU compiler ran and refused a graph.
        Here the raw text goes through the same classifier ``probe`` uses, and the
        registry's answer replaces the generic one whenever a rule recognised it.
        An unrecognised failure keeps the runtime's own report, which at least
        names the device.
        """
        if self._classifier is None:
            return error
        raw = error.diagnostic.raw_excerpt or error.diagnostic.message
        if not raw.strip():
            return error
        classified = self._classifier(raw, self._device)
        if classified is None or classified.matched_rule_id is None:
            return error
        return YongaError(classified)

    def close(self) -> None:
        """Release the runtime. Idempotent, and safe to call from a finally block."""
        if self._closed:
            return
        self._closed = True
        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            with suppress(Exception):
                runtime.close()

    def __enter__(self) -> ChatSession:
        """Return the session so it can be used as a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the session when the ``with`` block ends."""
        self.close()

    # --- turns -----------------------------------------------------------

    def send(self, user_text: str, *, should_stop: Callable[[], bool] | None = None) -> ChatTurn:
        """Answer one user message.

        The raw model text is sanitised before it is returned *and* before it is
        stored in the conversation, so a hidden reasoning block can neither reach
        the screen nor be fed back as context on the next turn. With reasoning
        shown, ``display_text`` is still the answer alone and the reasoning comes
        separately in ``reasoning_text``, so the CLI can draw it first and the
        conversation keeps only the answer either way. A failure comes back as a
        :class:`ChatTurn` carrying a diagnostic; the session, its history and its
        loaded pipeline all survive it.

        ``should_stop`` is asked on every chunk a streaming runtime produces;
        once it answers true generation ends, the partial reply is dropped
        together with the user message, and the turn comes back ``cancelled``.
        That is how Ctrl-C stops a reply without ending the chat.
        """
        prompt = user_text.strip()
        if not prompt:
            return ChatTurn(diagnostic=_empty_prompt_diagnostic())
        self.history.add_user(prompt)
        try:
            runtime = self.load()
            self._emit(ProgressKind.STAGE, "Generating a reply...")
            if should_stop is not None and isinstance(runtime, StreamingChatRuntime):

                def stop_requested(_chunk: str) -> bool:
                    return should_stop()

                raw = runtime.generate_stream(self.history, self._generation, stop_requested)
            else:
                raw = runtime.generate(self.history, self._generation)
            if should_stop is not None and should_stop():
                self._rollback_user_turn()
                return ChatTurn(cancelled=True)
            answer = sanitize_answer(
                raw.text,
                self._profile.reasoning_wrappers,
                show=False,
                starts_in_reasoning=self._starts_in_reasoning,
            )
            display = answer.display_text
            result = raw.model_copy(
                update={
                    "display_text": display,
                    "reasoning_text": answer.reasoning_text,
                    "reasoning_hidden": answer.reasoning_hidden and not self._show_reasoning,
                    "reasoning_unterminated": answer.reasoning_unterminated,
                }
            )
        except YongaError as error:
            self._rollback_user_turn()
            self._emit_diagnostic(error.diagnostic)
            return ChatTurn(diagnostic=error.diagnostic)
        except Exception as error:
            self._rollback_user_turn()
            report = _unexpected_diagnostic(error, device=self._device)
            self._emit_diagnostic(report)
            return ChatTurn(diagnostic=report)
        self.history.add_assistant(display)
        return ChatTurn(result=result)

    def respond(
        self,
        history: ChatHistory,
        generation: GenerationSettings,
        *,
        on_text: TextSink | None = None,
        stop: Sequence[str] = (),
        parallel_tool_calls: bool = True,
        should_stop: Callable[[], bool] | None = None,
    ) -> ChatTurn:
        """Answer one complete conversation without touching this session's own.

        What an HTTP request needs: the client owns the conversation and sends
        all of it every time, so this reads ``history`` and never records
        anything. The loaded runtime, the chat profile and the reasoning policy
        are the session's, exactly as :meth:`send` uses them.

        With ``on_text`` the answer is streamed through a
        :class:`StreamingSanitizer`, so no reasoning text is ever released; a
        runtime that cannot stream answers in one piece through the same sink.
        Failures come back as a :class:`ChatTurn` carrying a diagnostic, and an
        exception raised by ``on_text`` itself propagates to the caller.

        ``stop`` holds a client's stop sequences. A :class:`StopMatcher` after
        the sanitiser looks for them in the visible answer only, and the first
        one to complete ends the answer before it: the runtime is asked to stop, the
        result's ``display_text`` is cut there (see :func:`truncate_at_stop`)
        and its ``stop_reason`` is :data:`STOP_SEQUENCE_REASON`. A runtime that
        can stream is streamed even without ``on_text`` then, so generation
        ends at the match instead of running on to its token limit.

        When ``history`` offers tools, the visible answer is read for calls in
        the format of :attr:`tool_calling` (a model without one gets a
        diagnostic instead). The calls go to the result's ``tool_calls``, its
        ``display_text`` keeps the text around them, and its ``stop_reason``
        is :data:`~yonga.application.tool_calls.TOOL_CALLS_REASON`; with
        ``parallel_tool_calls`` false only the first call is kept. Streamed,
        a :class:`~yonga.application.tool_calls.ToolCallGate` between the
        sanitiser and the stop matcher releases text until a call could
        begin, so stop sequences are matched on that text only, and a stop
        that completes before the first call ends the answer without calls.
        An answer that cannot be read as calls is returned as text. Calls
        are read outside reasoning whether it is shown or not: with
        ``--show-reasoning`` the reasoning that opens the answer is shown in
        front of the content (:func:`split_leading_reasoning`) and the rest is
        read as a hidden-reasoning answer would be. Stop sequences are then
        matched on the shown reasoning and on the answer apart, so the blank
        line yonga puts between them never matches one.

        ``should_stop`` is asked on every chunk the runtime streams, whatever
        is released: a client that went away ends generation even while the
        answer is being held back.
        """
        if not any(m.role == "user" and (m.content.strip() or m.images) for m in history.messages):
            return ChatTurn(diagnostic=_empty_prompt_diagnostic())
        rule = self._tool_calling if history.tools else None
        if history.tools and rule is None:
            return ChatTurn(diagnostic=_tools_unsupported_diagnostic(self._tool_calling_refusal))
        names = tool_names(history.tools)
        gate = ToolCallGate(rule, names) if rule is not None else None
        wrappers = self._profile.reasoning_wrappers
        # Shown reasoning is passed through in front of the answer; calls are
        # read from the rest only, as they would be with the reasoning hidden.
        leading = (
            LeadingReasoning(wrappers, starts_in_reasoning=self._starts_in_reasoning)
            if gate is not None and self._show_reasoning
            else None
        )
        streaming = StreamingSanitizer(
            wrappers,
            show=self._show_reasoning and leading is None,
            starts_in_reasoning=self._starts_in_reasoning and leading is None,
        )
        stops = [text for text in stop if text]
        matcher = StopMatcher(stops)
        # With shown reasoning each half is matched apart, so a stop never
        # matches the separator yonga puts between them.
        split = _SplitStopMatcher(leading, stops) if leading is not None else None

        def stopped() -> bool:
            return split.hit if split is not None else matcher.hit

        def visible(chunk: str) -> str:
            shown = ""
            if leading is not None:
                shown, chunk = leading.feed(chunk)
            released = streaming.feed(chunk) if chunk else ""
            if gate is not None and released:
                released = gate.feed(released)
            if split is not None:
                return split.feed(shown, released)
            return matcher.feed(released) if stops and released else released

        try:
            runtime = self.load()
            if (on_text is not None or stops) and isinstance(runtime, StreamingChatRuntime):

                def forward(chunk: str) -> bool | None:
                    if should_stop is not None and should_stop():
                        return True
                    released = visible(chunk)
                    wants_stop = on_text(released) if on_text is not None and released else None
                    return True if stopped() else wants_stop

                raw = runtime.generate_stream(history, generation, forward)
            else:
                raw = runtime.generate(history, generation)
                released = visible(raw.text)
                if on_text is not None and released:
                    on_text(released)
            answer = sanitize_answer(
                raw.text,
                wrappers,
                show=self._show_reasoning,
                starts_in_reasoning=self._starts_in_reasoning,
            )
        except YongaError as error:
            return ChatTurn(diagnostic=error.diagnostic)
        if on_text is not None:
            shown, tail = leading.finish() if leading is not None else ("", "")
            rest = (streaming.feed(tail) if tail else "") + streaming.finish()
            if gate is not None:
                rest = (gate.feed(rest) if rest else "") + gate.finish()
            if split is not None:
                rest = split.feed(shown, rest) + split.finish()
            elif stops:
                rest = (matcher.feed(rest) if rest else "") + matcher.finish()
            if rest:
                on_text(rest)
        update: dict[str, object] = {
            "display_text": answer.display_text,
            "reasoning_text": answer.reasoning_text,
            "reasoning_hidden": answer.reasoning_hidden,
            "reasoning_unterminated": answer.reasoning_unterminated,
        }
        if rule is None and not stops:
            return ChatTurn(result=raw.model_copy(update=update))
        shown = ""
        if rule is None:
            parsed = ParsedAnswer(answer.display_text.strip(), [], answer.display_text.strip())
        elif leading is None:
            parsed = parse_tool_calls(answer.display_text, rule, names)
        else:
            lead, after = split_leading_reasoning(
                raw.text, wrappers, starts_in_reasoning=self._starts_in_reasoning
            )
            hidden = sanitize_answer(after, wrappers, show=False).display_text
            parsed = parse_tool_calls(hidden, rule, names)
            shown = lead.strip()
        content, calls = parsed.content, parsed.calls
        if stops:
            # The same cuts the stream makes: shown reasoning and the answer are
            # matched apart, so the separator between them never matches, and a
            # stop that completes before the first call ends the answer there
            # with no call.
            reasoning_cut, matched = truncate_at_stop(shown, stops)
            if matched:
                shown, content, calls = reasoning_cut, "", []
            else:
                cut, matched = truncate_at_stop(parsed.prefix, stops)
                if matched:
                    content, calls = cut, []
                else:
                    content, matched = truncate_at_stop(content, stops)
            if matched:
                update["stop_reason"] = STOP_SEQUENCE_REASON
        content = join_after_reasoning(shown, content)
        update["display_text"] = content
        if calls:
            update["tool_calls"] = calls if parallel_tool_calls else calls[:1]
            if "stop_reason" not in update:
                update["stop_reason"] = TOOL_CALLS_REASON
        elif parsed.unparsed and "stop_reason" not in update:
            update["tool_calls_unparsed"] = True
        return ChatTurn(result=raw.model_copy(update=update))

    def _rollback_user_turn(self) -> None:
        """Drop the user message that produced no answer, keeping the history consistent."""
        messages = self.history.messages
        if messages and messages[-1].role == "user":
            messages.pop()

    def reset(self) -> None:
        """Forget the conversation, keeping the session's system prompt.

        The pipeline is deliberately not reloaded — that would pay the compile
        cost again — but it is *told* that a new conversation starts here. For
        the LLM path that changes nothing, since the whole conversation is
        re-supplied on every turn. For a stateful path it is the only way the
        runtime can know that retyping the first message is a fresh start rather
        than a repeat with nothing new in it.
        """
        self.history.reset(keep_system=True)
        runtime = self._runtime
        if runtime is not None:
            with suppress(Exception):
                runtime.begin_conversation()

    # --- settings --------------------------------------------------------

    def set_tokens(self, value: int) -> int:
        """Set the maximum number of tokens in a reply. Returns the value applied."""
        number = _require_int(value, "/tokens N")
        if number < 1 or number > MAX_OUTPUT_TOKENS:
            message = f"/tokens N expects a value between 1 and {MAX_OUTPUT_TOKENS}; got {number}."
            raise ValueError(message)
        self._generation.max_new_tokens = number
        return number

    def set_temperature(self, value: float) -> float:
        """Set the sampling temperature; ``0`` switches to greedy decoding."""
        number = _require_float(value, "/temp X")
        low, high = TEMPERATURE_RANGE
        if number < low or number > high:
            message = f"/temp X expects a value between {low:g} and {high:g}; got {number:g}."
            raise ValueError(message)
        self._generation.temperature = number
        self._generation.do_sample = number > 0.0
        return number

    def set_top_p(self, value: float) -> float:
        """Set nucleus sampling."""
        number = _require_float(value, "/top-p X")
        low, high = TOP_P_RANGE
        if number < low or number > high:
            message = f"/top-p X expects a value between {low:g} and {high:g}; got {number:g}."
            raise ValueError(message)
        self._generation.top_p = number
        return number

    def set_top_k(self, value: int) -> int:
        """Set top-k sampling; ``0`` disables it."""
        number = _require_int(value, "/top-k N")
        if number < 0 or number > TOP_K_MAX:
            message = f"/top-k N expects a value between 0 and {TOP_K_MAX}; got {number}."
            raise ValueError(message)
        self._generation.top_k = number or None
        return number

    def set_reasoning(self, show: bool) -> bool:
        """Show or hide the model's reasoning block. Returns the new state."""
        self._show_reasoning = bool(show)
        return self._show_reasoning

    def set_stats(self, enabled: bool) -> bool:
        """Show or hide the metrics footer. Returns the new state."""
        self._stats = bool(enabled)
        return self._stats

    # --- commands --------------------------------------------------------

    def apply_command(self, command: ParsedCommand) -> CommandOutcome:
        """Execute one parsed ``/command`` and report what the loop should do next.

        The UI parses and reports; this is where a command actually changes the
        session. Every rejection carries an English message the loop can print
        before returning to the prompt.
        """
        parser_error = command.error
        if parser_error:
            return CommandOutcome(ok=False, message=parser_error)
        name = _command_name(command.kind)
        try:
            return self._dispatch(name, command)
        except ValueError as error:
            return CommandOutcome(ok=False, message=str(error))

    def _dispatch(self, name: str, command: ParsedCommand) -> CommandOutcome:
        """Route one validated command name to the state it changes."""
        if name == ChatCommandName.HELP.value:
            return CommandOutcome(action=ChatAction.HELP)
        if name == ChatCommandName.CLEAR.value:
            return CommandOutcome(action=ChatAction.CLEAR)
        if name == ChatCommandName.MODEL.value:
            return CommandOutcome(action=ChatAction.MODEL)
        if name == ChatCommandName.EXIT.value:
            return CommandOutcome(action=ChatAction.EXIT, message="Goodbye.")
        if name == ChatCommandName.RESET.value:
            self.reset()
            return CommandOutcome(message="Conversation history cleared.")
        if name == ChatCommandName.TOKENS.value:
            applied = self.set_tokens(_required_int(command.int_value, "/tokens N"))
            return CommandOutcome(message=f"Maximum output tokens set to {applied}.")
        if name == ChatCommandName.TEMPERATURE.value:
            temperature = self.set_temperature(_required_float(command.float_value, "/temp X"))
            if temperature == 0.0:
                return CommandOutcome(message="Temperature set to 0; decoding is now greedy.")
            return CommandOutcome(message=f"Temperature set to {temperature:g}.")
        if name == ChatCommandName.TOP_P.value:
            nucleus = self.set_top_p(_required_float(command.float_value, "/top-p X"))
            return CommandOutcome(message=f"Top-p set to {nucleus:g}.")
        if name == ChatCommandName.TOP_K.value:
            applied = self.set_top_k(_required_int(command.int_value, "/top-k N"))
            if applied == 0:
                return CommandOutcome(message="Top-k sampling disabled.")
            return CommandOutcome(message=f"Top-k set to {applied}.")
        if name == ChatCommandName.REASONING.value:
            return self._apply_reasoning(_required_bool(command.bool_value, "/reasoning on|off"))
        if name == ChatCommandName.STATS.value:
            enabled = self.set_stats(_required_bool(command.bool_value, "/stats on|off"))
            state = "enabled" if enabled else "disabled"
            return CommandOutcome(message=f"Metrics footer {state}.")
        return CommandOutcome(
            ok=False,
            message=f"Unknown command '/{name}'. Type /help to see the available commands.",
        )

    def _apply_reasoning(self, show: bool) -> CommandOutcome:
        """Apply ``/reasoning on|off`` and explain what it can and cannot change."""
        self.set_reasoning(show)
        if not show:
            return CommandOutcome(message="Reasoning blocks are now hidden.")
        if self._thinking_variable is None and self._starts_in_reasoning:
            return CommandOutcome(
                message=(
                    "Reasoning blocks are now shown. This chat template opens one in the "
                    "prompt and nothing can switch it off, so replies were being hidden "
                    "until it closed."
                )
            )
        if self._thinking_variable is not None and not self._thinking:
            return CommandOutcome(
                message=(
                    "Reasoning blocks are now shown. Thinking mode is off for this session, "
                    "so the model may not produce any; restart the chat with --show-reasoning "
                    "to turn it on."
                )
            )
        return CommandOutcome(message="Reasoning blocks are now shown.")

    # --- description -----------------------------------------------------

    def describe(self) -> ChatSessionDescription:
        """Describe the session for the startup panel and for ``/model``."""
        manifest = self._manifest
        requested = manifest.requested_model if manifest is not None else None
        resolved = manifest.resolved_source if manifest is not None else None
        revision = None
        if resolved is not None:
            revision = resolved.resolved_revision or resolved.requested_revision
        if revision is None and requested is not None:
            revision = requested.resolved_revision or requested.requested_revision
        return ChatSessionDescription(
            display_name=self.display_name,
            artifact_id=manifest.artifact_id if manifest is not None else "",
            repo_id=requested.repo_id if requested is not None else None,
            revision=revision,
            device=self._device,
            device_full_name=None,
            profile=manifest.export.profile if manifest is not None else None,
            max_new_tokens=self._generation.max_new_tokens,
            show_reasoning=self._show_reasoning,
            compile_cache_enabled=self._settings.compile_cache_enabled,
            compile_cache_path=self._compile_cache_path(),
            artifact_path=self._artifact_dir,
            extra=self._description_extras(),
        )

    def _compile_cache_path(self) -> Path | None:
        """The compile cache directory in use, preferring what the runtime actually set."""
        if not self._settings.compile_cache_enabled:
            return None
        runtime = self._runtime
        properties = getattr(runtime, "properties", None)
        if isinstance(properties, Mapping):
            configured = properties.get("CACHE_DIR")
            if isinstance(configured, str) and configured:
                return Path(configured)
        return self._settings.paths.compile_cache_dir

    def _description_extras(self) -> dict[str, str]:
        """Extra startup-panel rows describing the generation policy in force.

        The chat profile and the thinking state are reported because they decide
        whether reasoning is suppressed at all, and a user staring at a short or
        empty answer needs to be able to see which rule produced it.
        """
        extras: dict[str, str] = {"Chat profile": self._profile.id or "built-in default"}
        extras["Thinking mode"] = self._thinking_summary()
        sampling = _sampling_summary(self._generation)
        if sampling:
            extras["Sampling"] = sampling
        extras["Metrics footer"] = "on" if self._stats else "off"
        return extras

    def _thinking_summary(self) -> str:
        """Describe the thinking state, naming the template variable that drives it.

        When no variable could be established the summary also has to say whether
        reasoning is being hidden *by assumption*: the template opens a block
        nothing can switch off, so replies are suppressed until it closes. A user
        who is not told that has no way to connect an empty panel to a decision
        this session made.
        """
        variable = self._thinking_variable
        if variable is None:
            if self._starts_in_reasoning:
                return (
                    "unavailable (no thinking variable; this template opens a reasoning "
                    "block, so replies are hidden until it closes)"
                )
            return "unavailable (this chat template has no thinking variable)"
        state = "on" if self._thinking else "off"
        return f"{state} ({variable}={str(self._thinking).lower()})"

    # --- progress --------------------------------------------------------

    def _emit(self, kind: ProgressKind, message: str) -> None:
        """Emit one progress event; a failing sink never breaks a turn."""
        with suppress(Exception):
            self._progress.emit(ProgressEvent(kind=kind, message=message, device=self._device))

    def _emit_diagnostic(self, report: Diagnostic) -> None:
        """Emit a failed turn's diagnostic; a failing sink never breaks a turn."""
        with suppress(Exception):
            self._progress.emit(
                ProgressEvent(
                    kind=ProgressKind.LOG,
                    message=report.title,
                    severity=Severity.ERROR,
                    diagnostic=report,
                    device=self._device,
                )
            )


def _sampling_summary(settings: GenerationSettings) -> str:
    """Render the active sampling settings as one short line."""
    if not settings.do_sample:
        return "greedy"
    parts: list[str] = []
    if settings.temperature is not None:
        parts.append(f"temp {settings.temperature:g}")
    if settings.top_p is not None:
        parts.append(f"top-p {settings.top_p:g}")
    if settings.top_k is not None:
        parts.append(f"top-k {settings.top_k}")
    return " • ".join(parts)


def _resolve_thinking(
    profile: ChatProfileView,
    options: ChatOptions,
    *,
    thinking_available: bool,
) -> bool:
    """Decide whether this session asks the chat template to enable thinking.

    An explicit ``ChatOptions.thinking`` wins. Otherwise asking to *see*
    reasoning on a model whose template actually has a thinking variable is taken
    as asking for it, since the alternative is a flag that visibly does nothing.
    Everything else follows the profile, which defaults to off.
    """
    if options.thinking is not None:
        return options.thinking
    if options.show_reasoning and thinking_available:
        return True
    return profile.default_thinking


def _require_int(value: int, usage: str) -> int:
    """Coerce a command argument to a whole number or raise a usage error."""
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"Usage: {usage}"
        raise ValueError(message)
    return value


def _require_float(value: float, usage: str) -> float:
    """Coerce a command argument to a finite number or raise a usage error."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        message = f"Usage: {usage}"
        raise ValueError(message)
    number = float(value)
    if not _finite_float(number):
        message = f"{usage} expects a finite number."
        raise ValueError(message)
    return number


def _required_int(value: int | None, usage: str) -> int:
    """Return a mandatory whole-number argument or raise a usage error."""
    if value is None:
        message = f"Usage: {usage}"
        raise ValueError(message)
    return value


def _required_float(value: float | None, usage: str) -> float:
    """Return a mandatory decimal argument or raise a usage error."""
    if value is None:
        message = f"Usage: {usage}"
        raise ValueError(message)
    return value


def _required_bool(value: bool | None, usage: str) -> bool:
    """Return a mandatory ``on``/``off`` argument or raise a usage error."""
    if value is None:
        message = f"Usage: {usage}"
        raise ValueError(message)
    return value


# --------------------------------------------------------------------------------------
# the service
# --------------------------------------------------------------------------------------


class ChatService:
    """Opens chat sessions for prepared artifacts.

    The service resolves three things the session cannot resolve for itself — the
    artifact directory, its manifest, and the registry chat profile for its
    architecture — and then hands the session everything it needs. It constructs
    no adapters: the runtime factory and the artifact store are injected, which
    is what keeps the inference libraries out of the application layer.
    """

    def __init__(
        self,
        runtime_factory: RuntimeFactory,
        artifact_store: ArtifactStoreLike,
        registry_resolver: RegistryResolver,
        settings: Settings,
        *,
        progress: ProgressSink | None = None,
    ) -> None:
        """Bind the service to its adapters, the resolved registry and the configuration."""
        self._runtime_factory = runtime_factory
        self._artifact_store = artifact_store
        self._resolver = registry_resolver
        self._settings = settings
        self._progress = progress if progress is not None else null_sink()

    @property
    def settings(self) -> Settings:
        """The configuration this service was built with."""
        return self._settings

    def open_session(self, reference: str, *, options: ChatOptions | None = None) -> ChatSession:
        """Open a session for a model id, artifact id or artifact path.

        The artifact store decides what ``reference`` means; a reference it
        cannot resolve raises, because a chat session has nothing to fall back
        to.
        """
        artifact_dir = self._artifact_store.resolve(reference)
        return self.open_artifact(artifact_dir, options=options)

    def open_artifact(
        self,
        artifact_dir: Path,
        *,
        options: ChatOptions | None = None,
        manifest: ArtifactManifest | None = None,
    ) -> ChatSession:
        """Open a session for an artifact directory that is already located."""
        chosen = options if options is not None else ChatOptions()
        directory = Path(artifact_dir)
        found = manifest if manifest is not None else self.load_manifest(directory)
        profile = self.chat_profile_for(found)
        device = self._resolve_device(found, chosen)
        return ChatSession(
            artifact_dir=directory,
            manifest=found,
            profile=profile,
            settings=self._settings,
            runtime_factory=self._runtime_factory,
            device=device,
            pipeline=_pipeline_for(found),
            properties=self._runtime_properties(found, device),
            options=chosen,
            progress=self._progress,
            classifier=self._load_classifier(found),
        )

    def _load_classifier(self, manifest: ArtifactManifest | None) -> LoadClassifier:
        """Return the callable a session uses to classify a runtime load failure."""
        meta = _metadata_from_manifest(manifest)

        def classify(raw: str, device: str) -> Diagnostic | None:
            try:
                return self._resolver.classify(
                    raw_error=raw, device=device, meta=meta, stage="chat"
                )
            except Exception:  # pragma: no cover - classify is documented total
                return None

        return classify

    # --- resolution ------------------------------------------------------

    def load_manifest(self, artifact_dir: Path) -> ArtifactManifest | None:
        """Read the artifact manifest, or ``None`` when it is missing or unreadable.

        A missing manifest is not fatal: the session falls back to the default
        chat profile and the configured device, which is exactly what a
        hand-placed artifact directory needs.
        """
        path = manifest_path(Path(artifact_dir))
        try:
            return ArtifactManifest.model_validate_file(path)
        except Exception:
            return None

    def chat_profile_for(self, manifest: ArtifactManifest | None) -> ChatProfileView:
        """Resolve the chat profile for an artifact, registry first.

        Order: the registry rule for the architecture or repository, then the
        chat profile recorded in the manifest when the artifact was prepared,
        then a conservative default that still strips ``<think>`` blocks. The
        generation defaults come from whichever of these won — never from a
        global constant.
        """
        rule = self._registry_chat_rule(manifest)
        if rule is not None:
            return ChatProfileView.from_rule(rule)
        snapshot = self._manifest_chat_snapshot(manifest)
        if snapshot is not None:
            return snapshot
        return ChatProfileView()

    def _registry_chat_rule(self, manifest: ArtifactManifest | None) -> ChatProfileRule | None:
        """Ask the resolver for this artifact's chat rule, tolerating a registry fault.

        The specificity preference — a family's own chat profile beating the one a
        catch-all architecture rule names — lives in
        :meth:`RegistryResolver.resolve_chat_profile`, so ``yonga chat`` and
        ``yonga registry explain`` cannot disagree about which profile won.
        """
        meta = _metadata_from_manifest(manifest)
        repo_id = meta.identity.repo_id if meta is not None else None
        try:
            architecture, _ = self._resolver.resolve_architecture(meta)
            return self._resolver.chat_profile(
                arch_rule=architecture,
                repo_id=repo_id,
                meta=meta,
            )
        except YongaError:
            # A registry authoring fault must not make the model unusable; the
            # built-in default still hides reasoning.
            return None

    def _manifest_chat_snapshot(self, manifest: ArtifactManifest | None) -> ChatProfileView | None:
        """Return the chat profile recorded in the manifest, when it carries one."""
        if manifest is None or manifest.recommended_runtime is None:
            return None
        return ChatProfileView.from_snapshot(manifest.recommended_runtime.chat)

    def _resolve_device(self, manifest: ArtifactManifest | None, options: ChatOptions) -> str:
        """Choose the device: explicit request, configured policy, then the manifest."""
        if options.device:
            return normalise_device(options.device) or _FALLBACK_DEVICE
        policy = self._settings.default_device
        if policy is not DevicePolicy.AUTO:
            return normalise_device(policy.value) or _FALLBACK_DEVICE
        if manifest is not None:
            recommended = manifest.recommended_runtime
            if recommended is not None and recommended.device:
                return normalise_device(recommended.device) or _FALLBACK_DEVICE
            supported = manifest.supported_devices
            if supported:
                return supported[0]
        return _FALLBACK_DEVICE

    def _runtime_properties(
        self,
        manifest: ArtifactManifest | None,
        device: str,
    ) -> dict[str, str]:
        """Return the manifest's runtime properties when they target this device.

        Properties are recorded per recommended device, so applying them to a
        different one would hand a device settings that were measured for
        another.
        """
        if manifest is None or manifest.recommended_runtime is None:
            return {}
        recommended = manifest.recommended_runtime
        if normalise_device(recommended.device) != device:
            return {}
        return dict(recommended.properties)


def _pipeline_for(manifest: ArtifactManifest | None) -> PipelineKind:
    """Return the pipeline kind to drive, defaulting to the text-only path."""
    if manifest is None:
        return PipelineKind.LLM
    if manifest.pipeline is PipelineKind.UNKNOWN:
        return PipelineKind.LLM
    return manifest.pipeline


def _metadata_from_manifest(manifest: ArtifactManifest | None) -> ModelMetadata | None:
    """Rebuild just enough model metadata for the registry to match rules against.

    The Hub is never contacted here: everything the matcher needs — repository
    id, model type, architecture and task — was recorded in the manifest when
    the artifact was prepared.
    """
    if manifest is None:
        return None
    identity = ModelIdentity(
        repo_id=manifest.resolved_source.repo_id or manifest.requested_model.repo_id,
        revision=manifest.resolved_source.resolved_revision,
    )
    architectures = [manifest.architecture] if manifest.architecture else []
    return ModelMetadata(
        identity=identity,
        model_type=manifest.model_type,
        architectures=architectures,
        task=manifest.task,
        pipeline_tag=None,
    )
