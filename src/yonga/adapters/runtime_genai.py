"""OpenVINO GenAI chat/bench runtime and the domain-facing chat types.

The UI and the application services talk to :class:`ChatRuntime`; only this
module ever imports ``openvino_genai``. The conversation types
(:class:`ChatMessage`, :class:`ChatHistory`, :class:`GenerationResult`) are
*ours*, deliberately distinct from ``openvino_genai.ChatHistory`` — the GenAI
object is built fresh inside :meth:`GenAIChatRuntime.generate` and never leaves
this file. ``application/chat.py`` should re-export these names rather than
redefine them.

Two upstream details are hidden here on purpose:

* ``start_chat()``/``finish_chat()`` are deprecated and must not appear in the
  domain interface. The LLM path does not drive chats with them: multi-turn
  state travels as an ``openvino_genai.ChatHistory`` passed to ``generate``.
  A history that does not end with a user turn — the follow-up after a tool
  result — is refused that way by GenAI 2026.4's stateful pipelines, so it is
  rendered with the model's own chat template and passed as tokens instead,
  with ``finish_chat()`` resetting the pipeline's state before and after that
  call (see :meth:`GenAIChatRuntime._generate_llm_rendered`). The VLM
  path does the same wherever ``VLMPipeline.generate`` accepts a history, which
  OpenVINO GenAI 2026.4 does. Older builds have no such overload, so there the
  adapter falls back to driving the deprecated lifecycle internally and feeding
  the pipeline only the newest user turn. Because that fallback is stateful,
  the domain interface carries :meth:`ChatRuntime.begin_conversation` — a
  caller's explicit statement that the next history opens a new conversation.
  The adapter cannot infer that on its own: two conversations may open with
  identical words.
* Reasoning ("thinking") is switched off through the chat template variable via
  ``ChatHistory.set_extra_context``. The source repository's chat template is
  never rewritten. *Which* variable to send is decided from evidence: unless the
  registry profile states the answer outright, the artifact's own chat template
  is read and a candidate is applied only when that template actually references
  it. See :func:`resolve_thinking_variable`.
* The same template evidence decides whether generation *begins* inside a
  reasoning block. A template that opens ``<think>`` in the prompt produces a
  reply with no opener in it at all, which no sanitiser can recognise on its
  own, so :func:`resolve_starts_in_reasoning` establishes that state here and
  fails safe when it cannot prove the block was closed.

:class:`GenerationResult` carries the **raw** model text. ``display_text`` is set
equal to ``text`` and ``reasoning_text`` to ``None`` here; hiding reasoning
blocks is the chat service's ``sanitize_reasoning`` step, which owns the
show/hide policy and the unclosed-wrapper rule.
"""

from __future__ import annotations

import json
import math
import re
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yonga.adapters.native_output import (
    NativeOutput,
    capture_native_output,
    with_native_output,
)
from yonga.adapters.openvino_probe import (
    artifact_fingerprint,
    compile_cache_dir,
    compile_cache_key,
    openvino_runtime_version,
    record_compile_cache_use,
)
from yonga.config.settings import Settings
from yonga.domain.benchmarks import GenerationSettings
from yonga.domain.diagnostics import (
    CompileError,
    Diagnostic,
    FailureCategory,
    GenerationError,
    YongaError,
    diagnostic,
)
from yonga.domain.models import PipelineKind
from yonga.util.redaction import redact_text

__all__ = [
    "CHAT_TEMPLATE_FILES",
    "DEFAULT_REASONING_WRAPPERS",
    "DEFAULT_THINKING_VARIABLES",
    "PROMPT_TOO_LONG",
    "ChatHistory",
    "ChatImage",
    "ChatMessage",
    "ChatProfile",
    "ChatProfileLike",
    "ChatRuntime",
    "GenAIChatRuntime",
    "GenerationMetrics",
    "GenerationResult",
    "Role",
    "ToolCall",
    "build_generation_config",
    "collect_metrics",
    "open_runtime",
    "read_chat_template",
    "resolve_starts_in_reasoning",
    "resolve_thinking_variable",
    "template_opens_reasoning",
    "template_references",
    "thinking_extra_context",
]

#: Reasoning wrappers understood by the default chat profile.
DEFAULT_REASONING_WRAPPERS: tuple[tuple[str, str], ...] = (("<think>", "</think>"),)

#: Thinking variables tried against an artifact's chat template when its chat
#: profile names none. Mirrors the registry's default so a profile-less runtime
#: behaves the same way as one driven by ``chat-default``.
DEFAULT_THINKING_VARIABLES: tuple[str, ...] = ("enable_thinking",)

#: Files an artifact may keep its chat template in, in the order they are read.
CHAT_TEMPLATE_FILES: tuple[str, ...] = (
    "chat_template.jinja",
    "chat_template.json",
    "tokenizer_config.json",
)

#: Diagnostic code of a prompt longer than a pipeline compiled for a fixed
#: prompt length accepts (the NPU's ``MAX_PROMPT_LEN``). It describes the
#: request, not the runtime, so it is a configuration error.
PROMPT_TOO_LONG = "runtime.genai.prompt_too_long"

#: How OpenVINO GenAI 2026.4 refuses such a prompt, whichever NPU pipeline
#: (stateful or static LLM, VLM) runs it: "... may only process prompts or hold
#: chat history up to 1024 tokens. 6423 is passed." The check runs before any
#: inference, so the refusal costs the tokenisation only, and its count is
#: GenAI's own -- a count of yonga's could disagree with it by the special
#: tokens.
_PROMPT_LIMIT = re.compile(r"may only process [^.]*?up to (\d+) tokens\.\s*(\d+) is passed")

_DEFAULT_DEVICE = "CPU"
_CACHE_DIR_PROPERTY = "CACHE_DIR"
_LENIENT = ConfigDict(extra="ignore")

#: The conversation roles the chat template contract uses. ``tool`` carries a
#: tool's result back to the model; only ``yonga serve`` sends one.
Role = Literal["system", "user", "assistant", "tool"]


# --- conversation types ---------------------------------------------------------------


class ChatImage(BaseModel):
    """One decoded image: 8-bit RGB pixels, row by row, three bytes per pixel.

    Already decoded on purpose, so this type needs neither Pillow nor numpy and
    the conversation types stay free of both; the adapter turns it into an
    ``openvino.Tensor`` immediately before generation. The pixels are kept out
    of ``repr`` so that a traceback or a log line can never carry an image.
    """

    model_config = ConfigDict(frozen=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    pixels: bytes = Field(repr=False)

    @model_validator(mode="after")
    def _pixels_fill_the_frame(self) -> ChatImage:
        if len(self.pixels) != self.width * self.height * 3:
            message = f"{len(self.pixels)} bytes cannot be a {self.width}x{self.height} RGB image."
            raise ValueError(message)
        return self


class ToolCall(BaseModel):
    """One function call the assistant made, in OpenAI's terms.

    ``arguments`` is JSON text, as OpenAI carries it on the wire; it is parsed
    into an object only where a chat template needs one (see
    :meth:`ChatMessage.as_payload`).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: str = "{}"

    def as_payload(self) -> dict[str, object]:
        """The call in the Hugging Face chat-template convention, arguments as an object.

        Templates such as Qwen's write ``arguments | tojson``, which would quote
        JSON text a second time; text that is not a JSON object stays text,
        and so does JSON nested too deeply to parse.
        """
        arguments: object = self.arguments
        with suppress(ValueError, RecursionError):
            parsed = json.loads(self.arguments)
            if isinstance(parsed, dict):
                arguments = parsed
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": arguments},
        }


class ChatMessage(BaseModel):
    """One turn in a conversation.

    ``images`` only ever travels to a VLM pipeline, and only on the newest user
    turn (see :meth:`GenAIChatRuntime._generate_vlm`). It is excluded from
    serialisation, so no image is ever written wherever a conversation is.

    ``tool_calls`` belongs to an assistant turn that called functions;
    ``tool_call_id`` and ``name`` to a ``tool`` turn carrying one call's result.
    The interactive chat never sets them.
    """

    model_config = _LENIENT

    role: Role
    content: str
    images: list[ChatImage] = Field(default_factory=list, exclude=True)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def as_payload(self) -> dict[str, object]:
        """Return the mapping GenAI expects: ``role`` and ``content``, plus any tool fields."""
        payload: dict[str, object] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [call.as_payload() for call in self.tool_calls]
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            payload["name"] = self.name
        return payload


class ChatHistory(BaseModel):
    """An ordered conversation, owned by the domain rather than by GenAI.

    Keeping our own type means the UI, the chat service and the transcript writer
    never import ``openvino_genai``; the adapter converts this into the GenAI
    object immediately before each generation.
    """

    model_config = _LENIENT

    messages: list[ChatMessage] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list, exclude=True)
    """Function definitions offered to the model, in OpenAI's ``tools`` shape.

    They reach the chat template through ``ChatHistory.set_tools``; only
    ``yonga serve`` sets them, and they are never serialised with the history.
    """

    def __len__(self) -> int:
        """Return the number of messages in the conversation."""
        return len(self.messages)

    def add_system(self, content: str) -> None:
        """Append a system message."""
        self.messages.append(ChatMessage(role="system", content=content))

    def add_user(self, content: str) -> None:
        """Append a user message."""
        self.messages.append(ChatMessage(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        """Append an assistant message."""
        self.messages.append(ChatMessage(role="assistant", content=content))

    def reset(self, keep_system: bool = True) -> None:
        """Clear the conversation, optionally preserving the leading system turns."""
        if keep_system:
            self.messages = [message for message in self.messages if message.role == "system"]
        else:
            self.messages = []

    @property
    def system_message(self) -> str | None:
        """The first system message's content, when the conversation has one."""
        return next((m.content for m in self.messages if m.role == "system"), None)

    def to_payload(self) -> list[dict[str, object]]:
        """Return the conversation as a list of role/content mappings."""
        return [message.as_payload() for message in self.messages]


class GenerationMetrics(BaseModel):
    """Performance numbers extracted from one ``PerfMetrics`` object.

    Every field is optional: a missing or raising metric getter yields ``None``
    and must never break generation (CLI_SPEC section 9).
    """

    model_config = _LENIENT

    input_tokens: int | None = None
    output_tokens: int | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    throughput_tps: float | None = None
    load_ms: float | None = None


class GenerationResult(BaseModel):
    """The outcome of one generation turn.

    ``text`` is the raw model output. ``display_text``, ``reasoning_text`` and the
    two ``reasoning_*`` flags are filled in by the chat service's sanitiser; this
    adapter sets ``display_text = text`` and leaves the rest at their defaults so
    a caller that does not sanitise still shows something truthful.

    ``reasoning_hidden`` and ``reasoning_unterminated`` exist so the UI can
    explain an empty answer without looking at the text again. They describe the
    *shape* of what was hidden, never its content: a renderer may say that 200
    tokens of reasoning were suppressed and that the block never closed, and must
    not be able to reconstruct a single word of it.
    """

    model_config = _LENIENT

    text: str
    display_text: str
    reasoning_text: str | None = None
    reasoning_hidden: bool = False
    """True when a reasoning block was found and removed from ``display_text``."""

    reasoning_unterminated: bool = False
    """True when a reasoning block was opened and never closed before the text ended.

    In practice this means the model ran into its token limit mid-thought, which
    is the difference between "it had nothing to say" and "it never got to the
    answer".
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    ttft_ms: float | None = None
    throughput_tps: float | None = None
    tpot_ms: float | None = None
    wall_seconds: float = 0.0
    load_seconds: float | None = None
    stop_reason: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    """Calls read out of the answer by the chat service; ``display_text`` is then what is left."""

    tool_calls_unparsed: bool = False
    """True when the answer began like a tool call that could not be read as one.

    It is then returned as text, as the model wrote it; the flag lets a server
    log that it happened without logging the text.
    """


# --- chat profile ---------------------------------------------------------------------


class ChatProfileLike(Protocol):
    """Structural type for a registry chat profile.

    The registry's ``ChatProfileRule`` satisfies this by declaring the same
    attribute names with the same types; the adapter reads nothing else.
    ``supports_thinking_flag`` is tri-state — ``None`` means "no opinion, decide
    from the artifact's chat template".
    """

    supports_thinking_flag: bool | None
    thinking_template_variable: str | None
    thinking_variable_candidates: list[str]
    default_thinking: bool


class ChatProfile(BaseModel):
    """Concrete chat profile, used when no registry rule matched.

    ``reasoning_wrappers`` is consumed by the chat service's sanitiser, not by
    this adapter.
    """

    model_config = _LENIENT

    supports_thinking_flag: bool | None = None
    thinking_template_variable: str | None = "enable_thinking"
    thinking_variable_candidates: list[str] = Field(
        default_factory=lambda: [*DEFAULT_THINKING_VARIABLES]
    )
    default_thinking: bool = False
    reasoning_wrappers: list[tuple[str, str]] = Field(
        default_factory=lambda: [*DEFAULT_REASONING_WRAPPERS]
    )


def read_chat_template(artifact_dir: Path) -> str | None:
    """Return the chat-template source an artifact ships, or ``None``.

    The template is read, never written: suppressing reasoning by editing the
    model's own template is forbidden (``SPEC.md`` section 18.4,
    ``MODEL_PIPELINE.md`` section 13), so this is evidence-gathering only.

    Exporters put the template in one of three places, checked in that order: a
    standalone ``chat_template.jinja``, a ``chat_template.json`` wrapper, or the
    ``chat_template`` key of ``tokenizer_config.json``. Newer Transformers
    releases may store a *list* of named templates there; every one of them is
    returned joined together, because any of them could be the one that runs.

    Never raises. An unreadable or malformed file is simply no evidence, and the
    caller then leaves the template exactly as the model author wrote it.
    """
    directory = Path(artifact_dir)
    for name in CHAT_TEMPLATE_FILES:
        path = directory / name
        try:
            if not path.is_file():
                continue
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        template = raw if path.suffix == ".jinja" else _template_from_json(raw)
        if template:
            return template
    return None


def _template_from_json(raw: str) -> str | None:
    """Pull the ``chat_template`` value out of a JSON config, whatever shape it has."""
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("chat_template")
    if isinstance(value, str):
        return value or None
    if isinstance(value, list):
        parts = [
            entry["template"]
            for entry in value
            if isinstance(entry, dict) and isinstance(entry.get("template"), str)
        ]
        return "\n".join(parts) or None
    return None


def template_references(template: str | None, variable: str) -> bool:
    """True when ``template`` uses ``variable`` as a name of its own.

    The match is on whole identifiers, so ``enable_thinking`` is not found inside
    ``disable_enable_thinking2``. A name reached through an attribute, as in
    ``message.enable_thinking``, does not count either: that is a property of a
    message, not a variable the caller can set through extra context.
    """
    name = (variable or "").strip()
    if not template or not name:
        return False
    pattern = rf"(?<![\w.]){re.escape(name)}(?![\w])"
    return re.search(pattern, template) is not None


def resolve_thinking_variable(
    profile: ChatProfileLike | None,
    *,
    template: str | None = None,
) -> str | None:
    """Return the chat-template variable that drives thinking mode, or ``None``.

    Three sources, in order of authority:

    1. ``supports_thinking_flag: false`` is a veto. Nothing is sent, whatever the
       template mentions.
    2. ``supports_thinking_flag: true`` is a declaration. The profile's
       ``thinking_template_variable`` is used as-is, because the registry author
       has stated the family honours it.
    3. Otherwise the candidates are tried against ``template`` and the first one
       the template actually references wins. A profile that names a variable
       makes that name the only candidate.

    With no template (``None``) there is no evidence, so only an explicit
    declaration can produce a variable. That is the safe direction to fail: a
    template is left exactly as its author wrote it.
    """
    if profile is None:
        return None
    declared = getattr(profile, "supports_thinking_flag", None)
    named = str(getattr(profile, "thinking_template_variable", "") or "").strip()
    if declared is False:
        return None
    if declared is True and named:
        return named
    candidates = (named,) if named else _candidate_variables(profile)
    return next((name for name in candidates if template_references(template, name)), None)


def _candidate_variables(profile: ChatProfileLike) -> tuple[str, ...]:
    """Return a profile's candidate variable names, tolerating an odd value."""
    raw = getattr(profile, "thinking_variable_candidates", None)
    if raw is None:
        return DEFAULT_THINKING_VARIABLES
    try:
        items = list(raw)
    except TypeError:
        return DEFAULT_THINKING_VARIABLES
    return tuple(item.strip() for item in items if isinstance(item, str) and item.strip())


def thinking_extra_context(
    profile: ChatProfileLike | None,
    *,
    thinking: bool | None = None,
    template: str | None = None,
) -> dict[str, bool]:
    """Return the chat-template variables to push through ``set_extra_context``.

    Returns an empty mapping when no thinking variable could be established,
    which is the safe default: an unknown template is left exactly as the model
    author wrote it. ``thinking`` overrides the profile's ``default_thinking``,
    and ``template`` is the artifact's own chat-template source, which is what
    turns a candidate name into evidence.
    """
    variable = resolve_thinking_variable(profile, template=template)
    if variable is None:
        return {}
    enabled = (
        bool(getattr(profile, "default_thinking", False)) if thinking is None else bool(thinking)
    )
    return {variable: enabled}


def _profile_wrappers(profile: ChatProfileLike | None) -> tuple[tuple[str, str], ...]:
    """Return a profile's reasoning wrappers, tolerating a profile that has none.

    ``reasoning_wrappers`` is not part of :class:`ChatProfileLike` — the sanitiser
    in the chat service is its real consumer — so it is read defensively here. A
    profile that carries no attribute at all falls back to the defaults; a
    profile that carries an *empty* list is taken at its word, because a family
    with no recognisable wrapper has no opener to find and no closer to wait for.
    """
    if profile is None:
        return DEFAULT_REASONING_WRAPPERS
    raw = getattr(profile, "reasoning_wrappers", None)
    if raw is None:
        return DEFAULT_REASONING_WRAPPERS
    try:
        items = list(raw)
    except TypeError:
        return DEFAULT_REASONING_WRAPPERS
    pairs: list[tuple[str, str]] = []
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
        if isinstance(opening, str) and isinstance(closing, str) and opening and closing:
            pairs.append((opening, closing))
    return tuple(pairs)


def template_opens_reasoning(
    template: str | None,
    wrappers: Sequence[tuple[str, str]] = DEFAULT_REASONING_WRAPPERS,
) -> bool:
    """True when ``template`` can emit a reasoning opener into the prompt itself.

    Chat templates write the opener as a literal inside the rendered assistant
    header — ``{{- '<think>\\n' }}`` — so the literal appearing anywhere in the
    template source is the evidence. It is deliberately not asked *whether* the
    branch that emits it runs: that depends on variables this process may not be
    able to set, and the whole point of the check is to decide what to assume
    when it cannot.

    Never raises. An unreadable template is no evidence, which is the answer the
    caller then has to fail safe about.
    """
    if not template:
        return False
    return any(opening in template for opening, _ in wrappers if opening)


def resolve_starts_in_reasoning(
    profile: ChatProfileLike | None,
    *,
    template: str | None = None,
    thinking: bool | None = None,
) -> bool:
    """Whether generation begins *inside* a reasoning block the prompt opened.

    A chat template that opens ``<think>`` in the prompt makes the model's reply
    start in the middle of a reasoning block, with no opener of its own. Such a
    reply is byte-for-byte indistinguishable from a plain answer, so the
    sanitiser cannot work it out and has to be told — see ``sanitize_answer``'s
    ``starts_in_reasoning``.

    The rule has two halves and errs toward hiding, which is what ``SPEC.md``
    section 6.4 requires of an unclosed block:

    1. the template contains a reasoning opener at all, and
    2. we could **not** establish that thinking is disabled.

    Pushing the template variable false (the Qwen3.5 path) closes the block
    inside the prompt, so the reply starts outside it and this is ``False``.
    Pushing it *true*, or not being able to push anything — an unconditionally
    opening template of the DeepSeek-R1-distill shape, where no variable guards
    the opener — leaves the block open, so this is ``True``.

    The cost of the assumption is a reply hidden in full when a model opens a
    block and answers without ever closing it. That is the correct trade under
    the spec, and it is recoverable: ``--show-reasoning`` and ``/reasoning on``
    both reveal the text, and the renderer says so rather than drawing a blank
    panel.
    """
    if not template_opens_reasoning(template, _profile_wrappers(profile)):
        return False
    context = thinking_extra_context(profile, thinking=thinking, template=template)
    disabled = any(value is False for value in context.values())
    return not disabled


# --- runtime protocol -----------------------------------------------------------------


@runtime_checkable
class ChatRuntime(Protocol):
    """A loaded model that can answer turns. Deliberately free of GenAI types."""

    def begin_conversation(self) -> None:
        """Declare that the next turn starts a brand-new conversation.

        Only the caller knows where one conversation ends and the next begins.
        A benchmark run, each of its warmups, and a ``/reset`` in chat each start
        one; a history that merely *looks* like the previous one — the same
        prompt fixture sent twice — is not a continuation of it.

        A **stateless** implementation may ignore this entirely: the LLM path
        re-supplies the whole history on every turn, so there is nothing stale to
        throw away, and the default below is exactly that no-op. A **stateful**
        implementation (the VLM path, which keeps a hidden pipeline-side session)
        **must** discard its session here, so the next turn is measured against an
        empty context rather than against whatever the last conversation left
        behind (``SPEC.md`` section 6.5).

        Must be idempotent, safe to call before the first turn, and must not raise.

        The body below is the stateless default — doing nothing is the correct
        behaviour for a runtime that holds no session — and is inherited by
        anything that subclasses the protocol explicitly.
        """
        return

    def generate(self, history: ChatHistory, config: GenerationSettings) -> GenerationResult:
        """Generate one assistant turn for ``history``."""
        ...

    def close(self) -> None:
        """Release the pipeline. Must be idempotent."""
        ...


#: Receives each decoded text chunk as it is generated. Returning ``True`` asks
#: the runtime to stop generating (the client went away, say); ``None`` or
#: ``False`` lets it continue.
TextCallback = Callable[[str], bool | None]


@runtime_checkable
class StreamingChatRuntime(ChatRuntime, Protocol):
    """A :class:`ChatRuntime` that can also report text as it is generated.

    Kept as its own protocol so every existing runtime stays a valid
    :class:`ChatRuntime` without growing a method it has no use for; a caller
    that wants streaming checks for this one and otherwise answers in one piece.
    """

    def generate_stream(
        self, history: ChatHistory, config: GenerationSettings, on_text: TextCallback
    ) -> GenerationResult:
        """Generate like :meth:`ChatRuntime.generate`, calling ``on_text`` per chunk.

        The returned result carries the whole raw text, exactly as ``generate``
        would have returned it.
        """
        ...


# --- lazy import ----------------------------------------------------------------------


def _is_staged(artifact_dir: Path, staging_dir: Path) -> bool:
    """Whether ``artifact_dir`` is a ``prepare`` job's work in progress, not a stored artifact."""
    with suppress(OSError, ValueError):
        return artifact_dir.resolve().is_relative_to(staging_dir.resolve())
    return False


def _import_genai() -> Any:
    """Import :mod:`openvino_genai` lazily. Raises ``ImportError`` when unavailable."""
    # Deliberately lazy: see the module docstring.
    import openvino_genai

    return openvino_genai


def _genai_missing_diagnostic(error: object) -> Diagnostic:
    """Build the diagnostic used when ``openvino_genai`` cannot be imported."""
    detail = redact_text(str(error))
    return diagnostic(
        "runtime.openvino_genai.missing",
        FailureCategory.PACKAGE_ABI_MISMATCH,
        "OpenVINO GenAI is not available",
        f"openvino_genai could not be imported, so no model can be run. {detail}",
        remediation=[
            "Install the runtime extras: pip install 'yonga[openvino]'",
            "openvino-genai must come from the same release family as openvino.",
        ],
        environment_broken=True,
        raw_excerpt=detail,
    )


def _require_genai() -> Any:
    """Return the ``openvino_genai`` module or raise a typed :class:`CompileError`."""
    try:
        return _import_genai()
    except Exception as exc:
        raise CompileError(_genai_missing_diagnostic(exc), cause=exc) from exc


# --- generation config ----------------------------------------------------------------


def build_generation_config(settings: GenerationSettings) -> Any:
    """Map :class:`GenerationSettings` onto an ``openvino_genai.GenerationConfig``.

    ``temperature``, ``top_p``, ``top_k`` and the RNG seed only take effect when
    ``do_sample`` is true, so they are applied only in that case — setting them
    for greedy decoding would silently suggest a sampling behaviour the runtime
    will not perform. ``repetition_penalty`` applies to both modes.

    ``stop_strings`` are the registry chat profile's model-level end markers.
    GenAI matches them natively, across chunk boundaries and without releasing
    a partial match to the streamer, and they are excluded from the output. They
    are set only when there are some, so a profile without any leaves GenAI's
    defaults untouched. The config is validated before it is returned.
    """
    genai = _require_genai()
    config = genai.GenerationConfig()
    config.max_new_tokens = max(1, int(settings.max_new_tokens))
    config.do_sample = bool(settings.do_sample)
    if settings.repetition_penalty is not None:
        config.repetition_penalty = float(settings.repetition_penalty)
    stop_strings = {text for text in settings.stop_strings if text}
    if stop_strings:
        config.stop_strings = stop_strings
        config.include_stop_str_in_output = False
    if settings.do_sample:
        if settings.temperature is not None:
            config.temperature = float(settings.temperature)
        if settings.top_p is not None:
            config.top_p = float(settings.top_p)
        if settings.top_k is not None:
            config.top_k = int(settings.top_k)
        if settings.seed is not None:
            config.rng_seed = int(settings.seed)
    try:
        config.validate()
    except Exception as exc:
        raise GenerationError(
            diagnostic(
                "runtime.generation.invalid_config",
                FailureCategory.CONFIGURATION_ERROR,
                "The generation settings were rejected by OpenVINO GenAI",
                redact_text(str(exc)),
                remediation=[
                    "Enable sampling (--temperature/--top-p/--top-k require do_sample).",
                    "Check that max_new_tokens is positive.",
                ],
                raw_excerpt=redact_text(str(exc)),
            ),
            cause=exc,
        ) from exc
    return config


# --- metrics --------------------------------------------------------------------------


def _finite(value: float) -> float | None:
    """Return ``value`` when it is a finite number, otherwise ``None``."""
    return value if math.isfinite(value) else None


def _metric_float(perf_metrics: object, getter: str) -> float | None:
    """Read one ``MeanStdPair`` (or plain float) metric defensively."""
    try:
        value = getattr(perf_metrics, getter)()
    except Exception:
        return None
    raw = getattr(value, "mean", value)
    try:
        return _finite(float(raw))
    except (TypeError, ValueError):
        return None


def _metric_int(perf_metrics: object, getter: str) -> int | None:
    """Read one integer metric defensively."""
    try:
        value = getattr(perf_metrics, getter)()
    except Exception:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def collect_metrics(perf_metrics: object | None) -> GenerationMetrics:
    """Extract every metric we report from a GenAI ``PerfMetrics`` object.

    Each getter is called in isolation: one metric that is absent, raises, or
    returns something unexpected costs exactly that one field. ``get_load_time``
    returns a plain float while the others return ``MeanStdPair``; both shapes are
    handled by reading an optional ``mean`` attribute.
    """
    if perf_metrics is None:
        return GenerationMetrics()
    return GenerationMetrics(
        input_tokens=_metric_int(perf_metrics, "get_num_input_tokens"),
        output_tokens=_metric_int(perf_metrics, "get_num_generated_tokens"),
        ttft_ms=_metric_float(perf_metrics, "get_ttft"),
        tpot_ms=_metric_float(perf_metrics, "get_tpot"),
        throughput_tps=_metric_float(perf_metrics, "get_throughput"),
        load_ms=_metric_float(perf_metrics, "get_load_time"),
    )


def _first_text(raw: object) -> str:
    """Return the first decoded string of a GenAI result object."""
    with suppress(Exception):
        texts = list(raw.texts)  # type: ignore[attr-defined]
        if texts:
            return str(texts[0])
    with suppress(Exception):
        return str(raw)
    return ""


class _RenderedTurnResult:
    """A GenAI ``EncodedResults``, read as the ``DecodedResults`` it stands for.

    The rendered-prompt route hands the pipeline tokens, so it gets tokens
    back; this carries their decoded text beside the untouched finish reasons
    and perf metrics, so the rest of the runtime reads both routes alike.
    """

    def __init__(self, *, text: str, finish_reasons: Any, perf_metrics: Any) -> None:
        self.texts = [text]
        self.finish_reasons = finish_reasons if finish_reasons is not None else []
        self.perf_metrics = perf_metrics


def _decoded_tokens(tokenizer: Any, raw: object) -> str:
    """Decode the first sequence of an ``EncodedResults``, skipping special tokens.

    Skipping them is what GenAI does when it decodes a string or history call.
    """
    sequences = list(raw.tokens)  # type: ignore[attr-defined]
    if not sequences:
        return ""
    return str(tokenizer.decode(list(sequences[0]), skip_special_tokens=True))


def _finish_reason(raw: object) -> str | None:
    """Return the first finish reason as a lower-case name, when reported."""
    with suppress(Exception):
        reasons = list(raw.finish_reasons)  # type: ignore[attr-defined]
        if reasons:
            first = reasons[0]
            name = getattr(first, "name", None)
            return str(name).lower() if name else str(first)
    return None


def _is_prefix(prefix: Sequence[ChatMessage], messages: Sequence[ChatMessage]) -> bool:
    """Return whether ``prefix`` is the leading run of ``messages``.

    This answers a question about *text*, not about identity: two different
    conversations that open with the same words are prefixes of one another.
    Callers must not read it as "this is a continuation" on its own — see
    :meth:`GenAIChatRuntime._continues_conversation`.
    """
    if len(prefix) > len(messages):
        return False
    return all(left == right for left, right in zip(prefix, messages, strict=False))


def _user_turn_text(messages: Sequence[ChatMessage]) -> str:
    """Join the non-empty user turns of ``messages`` into one prompt."""
    return "\n\n".join(
        message.content.strip()
        for message in messages
        if message.role == "user" and message.content.strip()
    )


def _has_user_input(messages: Sequence[ChatMessage]) -> bool:
    """Whether ``messages`` holds a user turn with something to answer: text or an image."""
    return any(
        message.role == "user" and (message.content.strip() or message.images)
        for message in messages
    )


#: GenAI's universal media tags, which it reads anywhere in a VLM prompt as
#: "image N" or "video N goes here". They are removed from the conversation's
#: text, so typing one cannot claim an image that was not attached (GenAI then
#: fails with "Missing image/video with index N") or move one that was. A
#: model's own vision tokens typed as text (such as Qwen2-VL's
#: ``<|vision_start|><|image_pad|><|vision_end|>``) are not removed: which
#: those are is per-family knowledge yonga does not carry.
_MEDIA_TAG = re.compile(r"<ov_genai_(?:image|video)_\d+>")


def _without_media_tags(text: str) -> str:
    """``text`` without universal media tags, removed until none is left.

    One pass is not enough: removing the inner tag of
    ``<ov_genai_<ov_genai_image_0>image_0>`` would join the rest into a new one.
    """
    while True:
        text, removed = _MEDIA_TAG.subn("", text)
        if not removed:
            return text


def _image_tensor(image: ChatImage) -> Any:
    """``image`` as the ``openvino.Tensor`` a VLM pipeline takes: uint8, height x width x 3.

    Imported here rather than at module level, like ``openvino_genai``, so the
    module loads without the OpenVINO extra. The tensor copies the pixels.
    """
    import numpy as np
    import openvino as ov

    array = np.frombuffer(image.pixels, dtype=np.uint8).reshape(image.height, image.width, 3)
    return ov.Tensor(array)


def _leading_system(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    """Return the run of system turns at the head of ``messages``.

    Only the leading run counts: what is recorded as already delivered has to be
    a genuine prefix of the conversation, or the next turn's "everything after
    what I sent" slice would skip real messages.
    """
    leading: list[ChatMessage] = []
    for message in messages:
        if message.role != "system":
            break
        leading.append(message)
    return leading


def _count(number: int, noun: str) -> str:
    """``the 30 offered tools``, ``the image``, or ``""`` for none."""
    if number == 0:
        return ""
    return f"the {noun}" if number == 1 else f"the {number} {noun}s"


def _one_of(choices: Sequence[str]) -> str:
    """``a``, ``a or b``, or ``a, b or c``, leaving out empty choices."""
    kept = [choice for choice in choices if choice]
    return " or ".join(kept) if len(kept) < 3 else f"{', '.join(kept[:-1])} or {kept[-1]}"


def _streamer(on_text: TextCallback | None) -> tuple[Any, list[BaseException]]:
    """Wrap ``on_text`` as a GenAI streamer, and a slot for anything it raised.

    GenAI calls the streamer from native code, where a Python exception would be
    reported as a generation failure. So a raising callback stops generation
    instead, and the caller re-raises what it recorded.
    """
    errors: list[BaseException] = []
    if on_text is None:
        return None, errors

    def stream(chunk: str) -> bool:
        if errors:
            return True
        try:
            return bool(on_text(chunk))
        except BaseException as error:  # re-raised by the caller, after GenAI returns
            errors.append(error)
            return True

    return stream, errors


def _streamer_kwargs(streamer: Any) -> dict[str, Any]:
    """``{"streamer": streamer}``, or nothing, so a non-streaming call is unchanged."""
    return {} if streamer is None else {"streamer": streamer}


# --- the runtime ----------------------------------------------------------------------


#: OpenVINO GenAI pipeline properties that must be passed as integers.
#: Registry and manifest data keep every property as a string, because it has to
#: round-trip through YAML and JSON, but GenAI extracts these typed: handed the
#: string "1024" for MAX_PROMPT_LEN it raises "Failed to extract MAX_PROMPT_LEN.
#: Type mismatch: expected types: int or int64_t" (observed on the reference host).
INTEGER_PIPELINE_PROPERTIES: frozenset[str] = frozenset({"MAX_PROMPT_LEN", "MIN_RESPONSE_LEN"})


def pipeline_kwargs(properties: Mapping[str, str]) -> dict[str, object]:
    """Convert string-valued properties into what the GenAI constructor accepts.

    Only properties known to need another type are converted; everything else
    passes through unchanged. A value that does not parse is left as given so
    GenAI reports the mismatch in its own words rather than yonga guessing.
    """
    converted: dict[str, object] = {}
    for name, value in properties.items():
        if name.upper() in INTEGER_PIPELINE_PROPERTIES:
            with suppress(ValueError):
                converted[name] = int(str(value).strip())
                continue
        converted[name] = value
    return converted


class GenAIChatRuntime:
    """An OpenVINO GenAI pipeline kept alive across turns.

    The pipeline object is created once, in the constructor, and reused for every
    turn: reloading per prompt would pay the compile/load cost again and again.
    Call :meth:`close` (or use the runtime as a context
    manager) to release it.
    """

    def __init__(
        self,
        artifact_dir: Path,
        *,
        device: str = _DEFAULT_DEVICE,
        pipeline: PipelineKind = PipelineKind.LLM,
        properties: Mapping[str, str] | None = None,
        chat_profile: ChatProfileLike | None = None,
        settings: Settings | None = None,
        thinking: bool | None = None,
    ) -> None:
        """Load ``artifact_dir`` on ``device``.

        Raises :class:`~yonga.domain.diagnostics.CompileError` when the package is
        missing, the directory is not an artifact, or the pipeline fails to load;
        the raw runtime message is preserved for the registry classifier.
        """
        self._artifact_dir = Path(artifact_dir)
        self._device = (device or "").strip() or _DEFAULT_DEVICE
        self._pipeline_kind = pipeline
        self._chat_profile = chat_profile
        self._thinking = thinking
        self._settings = settings if settings is not None else Settings()
        self._properties = self._resolve_properties(properties)
        self._chat_template = read_chat_template(self._artifact_dir)
        self._thinking_variable = resolve_thinking_variable(
            chat_profile, template=self._chat_template
        )
        self._starts_in_reasoning = resolve_starts_in_reasoning(
            chat_profile, template=self._chat_template, thinking=thinking
        )
        self._extra_context_error: str | None = None
        self._closed = False
        self._chat_started = False
        self._sent: list[ChatMessage] = []
        # Whether VLMPipeline.generate takes a ChatHistory: learnt on the first
        # VLM turn, since the answer depends on the installed GenAI build.
        self._vlm_accepts_history: bool | None = None
        self._load_seconds = 0.0
        self._genai: Any = _require_genai()
        self._pipeline: Any = self._open_pipeline()

    # --- construction helpers --------------------------------------------

    def _resolve_properties(self, properties: Mapping[str, str] | None) -> dict[str, str]:
        """Merge the profile's runtime properties with the managed compile cache.

        The cache directory is keyed with the OpenVINO version so a runtime
        upgrade can never be handed blobs built by the previous release
        (ARCHITECTURE section 7). An explicit ``CACHE_DIR`` from the caller wins,
        and is not yonga's to record.

        The managed directory gets an ownership record (artifact, device, runtime
        version, last use) so ``yonga cache prune`` can later tell whether its
        blobs still serve anything. Writing it never fails the load.

        An artifact still in the staging directory is never cached: see below.
        """
        resolved = {str(name): str(value) for name, value in (properties or {}).items()}
        if not self._settings.compile_cache_enabled or _CACHE_DIR_PROPERTY in resolved:
            return resolved
        if _is_staged(self._artifact_dir, self._settings.paths.staging_dir):
            # prepare's smoke test: the directory is about to move into the store
            # under another fingerprint, and OpenVINO keys its blobs by path, so
            # nothing could ever load them again. Caching would only leave an
            # orphaned entry the size of the model behind every prepare.
            return resolved
        fingerprint = artifact_fingerprint(self._artifact_dir)
        version = openvino_runtime_version()
        key = compile_cache_key(
            artifact_hash=fingerprint,
            device=self._device,
            openvino_version=version,
            properties=resolved,
        )
        directory = compile_cache_dir(self._settings.paths, key)
        record_compile_cache_use(
            directory,
            artifact_dir=self._artifact_dir,
            artifact_hash=fingerprint,
            device=self._device,
            openvino_version=version,
            properties=resolved,
        )
        resolved[_CACHE_DIR_PROPERTY] = str(directory)
        return resolved

    def _open_pipeline(self) -> Any:
        """Construct the GenAI pipeline for this artifact's pipeline kind."""
        if not self._artifact_dir.is_dir():
            raise CompileError(
                diagnostic(
                    "runtime.artifact.not_found",
                    FailureCategory.CONFIGURATION_ERROR,
                    "The prepared artifact directory does not exist",
                    f"No directory at {self._artifact_dir}.",
                    remediation=["Run `yonga prepare` first, or check the artifact path."],
                    context={"artifact_dir": str(self._artifact_dir)},
                )
            )
        started = time.perf_counter()
        # Compilation happens here. The NPU compiler reports its real reason on
        # the native descriptors rather than in the exception, so capture them.
        failure: Exception | None = None
        pipeline: Any = None
        with capture_native_output() as native:
            try:
                if self._pipeline_kind is PipelineKind.VLM:
                    pipeline = self._genai.VLMPipeline(
                        str(self._artifact_dir), self._device, **pipeline_kwargs(self._properties)
                    )
                else:
                    pipeline = self._genai.LLMPipeline(
                        str(self._artifact_dir), self._device, **pipeline_kwargs(self._properties)
                    )
            except Exception as exc:
                failure = exc
        if failure is not None:
            raise CompileError(
                self._load_failed_diagnostic(failure, native=native), cause=failure
            ) from failure
        self._load_seconds = time.perf_counter() - started
        return pipeline

    def _load_failed_diagnostic(
        self, error: BaseException, *, native: NativeOutput | None = None
    ) -> Diagnostic:
        """Build the load/compile failure diagnostic, keeping the raw runtime text."""
        text = str(error) if native is None else with_native_output(str(error), native)
        raw = redact_text(text)
        return diagnostic(
            "runtime.genai.load_failed",
            FailureCategory.UNKNOWN_RUNTIME_ERROR,
            f"Could not load the model on {self._device}",
            raw,
            remediation=[
                "Run `yonga doctor` to check the device and driver state.",
                "Try another device, for example --device cpu.",
            ],
            raw_excerpt=raw,
            context={
                "device": self._device,
                "pipeline": self._pipeline_kind.value,
                "artifact_dir": str(self._artifact_dir),
            },
        )

    # --- introspection ---------------------------------------------------

    @property
    def artifact_dir(self) -> Path:
        """The artifact directory this runtime loaded."""
        return self._artifact_dir

    @property
    def device(self) -> str:
        """The OpenVINO device string the pipeline was created with."""
        return self._device

    @property
    def pipeline_kind(self) -> PipelineKind:
        """Whether this runtime drives an LLM or a VLM pipeline."""
        return self._pipeline_kind

    @property
    def properties(self) -> dict[str, str]:
        """The pipeline properties actually applied, including ``CACHE_DIR``."""
        return dict(self._properties)

    @property
    def load_seconds(self) -> float:
        """Wall-clock seconds spent constructing the pipeline (compile + load)."""
        return self._load_seconds

    @property
    def thinking_variable(self) -> str | None:
        """The chat-template variable this runtime drives, when one was established.

        ``None`` means the artifact's template names no thinking variable this
        profile knows about, so the template is left untouched.
        """
        return self._thinking_variable

    @property
    def starts_in_reasoning(self) -> bool:
        """True when this artifact's template opens a reasoning block in the prompt.

        The reply then starts *inside* that block and carries no opener of its
        own, which the chat service's sanitiser cannot see for itself. Reported
        here because this is the layer that read the template and decided what
        could be pushed into it; see
        :func:`resolve_starts_in_reasoning` for the rule and why it fails safe.
        """
        return self._starts_in_reasoning

    @property
    def extra_context_error(self) -> str | None:
        """Why the chat-template variables could not be applied, when that failed."""
        return self._extra_context_error

    @property
    def closed(self) -> bool:
        """True once :meth:`close` has run."""
        return self._closed

    # --- generation ------------------------------------------------------

    def begin_conversation(self) -> None:
        """Discard any conversation the pipeline is still holding.

        The LLM path is stateless — its whole history travels with every
        ``generate`` call — so there is nothing to do for it. The VLM path keeps
        a hidden ``start_chat()`` session, and this is the only signal that the
        next history is a *new* conversation rather than a continuation that
        happens to begin with the same words.

        Idempotent, safe before the first turn and after :meth:`close`, and it
        never raises: refusing to start a conversation is not a failure mode a
        caller could do anything about.
        """
        if self._closed or self._pipeline is None:
            return
        if self._pipeline_kind is PipelineKind.VLM:
            self._end_vlm_chat()

    def generate(self, history: ChatHistory, config: GenerationSettings) -> GenerationResult:
        """Generate one assistant turn and collect its metrics.

        Raises :class:`~yonga.domain.diagnostics.GenerationError` on failure, with
        the raw runtime text preserved for the classifier.
        """
        return self._generate(history, config, None)

    def generate_stream(
        self, history: ChatHistory, config: GenerationSettings, on_text: TextCallback
    ) -> GenerationResult:
        """Generate one turn, handing each decoded chunk to ``on_text`` as it arrives."""
        return self._generate(history, config, on_text)

    def _generate(
        self, history: ChatHistory, config: GenerationSettings, on_text: TextCallback | None
    ) -> GenerationResult:
        """Shared body of :meth:`generate` and :meth:`generate_stream`."""
        if self._closed:
            raise GenerationError(
                diagnostic(
                    "runtime.genai.closed",
                    FailureCategory.CONFIGURATION_ERROR,
                    "The chat runtime has already been closed",
                    "Open a new runtime before generating again.",
                )
            )
        generation_config = build_generation_config(config)
        started = time.perf_counter()
        streamer, callback_error = _streamer(on_text)
        try:
            raw = self._generate_raw(history, generation_config, streamer)
        except YongaError:
            raise
        except Exception as exc:
            report = self._generation_failed_diagnostic(exc, history)
            raise GenerationError(report, cause=exc) from exc
        if callback_error:
            # The consumer's own failure stopped generation; it is not a model failure,
            # and the consumer is the one that has to hear about it.
            raise callback_error[0]
        wall_seconds = time.perf_counter() - started
        text = _first_text(raw)
        metrics = collect_metrics(getattr(raw, "perf_metrics", None))
        load_seconds = (
            metrics.load_ms / 1000.0 if metrics.load_ms is not None else self._load_seconds
        )
        return GenerationResult(
            text=text,
            display_text=text,
            reasoning_text=None,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            ttft_ms=metrics.ttft_ms,
            throughput_tps=metrics.throughput_tps,
            tpot_ms=metrics.tpot_ms,
            wall_seconds=wall_seconds,
            load_seconds=load_seconds,
            stop_reason=_finish_reason(raw),
        )

    def _generate_raw(
        self, history: ChatHistory, generation_config: Any, streamer: Any = None
    ) -> Any:
        """Dispatch to the LLM or VLM generation path."""
        images = self._turn_images(history)
        if self._pipeline_kind is PipelineKind.VLM:
            return self._generate_vlm(history, generation_config, streamer, images)
        return self._generate_llm(history, generation_config, streamer)

    def _turn_images(self, history: ChatHistory) -> list[ChatImage]:
        """The images this turn hands the pipeline: those of the newest user turn.

        OpenVINO GenAI 2026.4 binds images only to the last user message of a
        stateless history; an image tag in an earlier turn fails with "Missing
        image/video with index 0". So an image anywhere else is refused here
        rather than silently moved, and so is any image on an LLM pipeline. The
        serve layer refuses both before they get this far, naming the parameter.
        """
        messages = history.messages
        if not any(message.images for message in messages):
            return []
        if self._pipeline_kind is not PipelineKind.VLM:
            raise GenerationError(
                diagnostic(
                    "runtime.genai.images_unsupported",
                    FailureCategory.CONFIGURATION_ERROR,
                    "This model takes text only",
                    "Images need a vision-language (VLM) artifact.",
                )
            )
        last = messages[-1]
        if last.role != "user" or any(message.images for message in messages[:-1]):
            raise GenerationError(
                diagnostic(
                    "runtime.genai.images_not_last_turn",
                    FailureCategory.CONFIGURATION_ERROR,
                    "Images can only go with the newest user message",
                    "OpenVINO GenAI binds images to the last user turn of a conversation only.",
                )
            )
        return list(last.images)

    def _generate_llm(self, history: ChatHistory, generation_config: Any, streamer: Any) -> Any:
        """Generate with an ``LLMPipeline`` using a GenAI ``ChatHistory``.

        Passing the history object is what replaces the deprecated
        ``start_chat()``/``finish_chat()`` lifecycle: the whole conversation is
        re-supplied each turn and the pipeline applies the chat template itself.

        That holds only while the history ends with a user turn. OpenVINO GenAI
        2026.4's stateful and NPU pipelines (``ATTENTION_BACKEND=SDPA``, and
        every NPU LLM) refuse any other history with "Chat doesn't support
        switching between input types": a tool result or an assistant prefix in
        the last place fails, which is the second half of every tool round
        trip. Such a history goes through :meth:`_generate_llm_rendered`
        instead, on every backend, so a conversation cannot behave differently
        on GPU than it did on CPU.
        """
        genai_history = self._genai_history(history)
        messages = history.messages
        if messages and messages[-1].role != "user":
            return self._generate_llm_rendered(genai_history, generation_config, streamer)
        return self._pipeline.generate(
            genai_history, generation_config, **_streamer_kwargs(streamer)
        )

    def _generate_llm_rendered(
        self, genai_history: Any, generation_config: Any, streamer: Any
    ) -> _RenderedTurnResult:
        """Generate from a history that does not end with a user turn.

        The prompt is what the ``ChatHistory`` route would have built: the
        pipeline's own tokenizer applies the artifact's chat template to the
        same GenAI history — with its tools and template variables — adding the
        generation prompt, and encodes it without special tokens, exactly as
        GenAI does for a history. Handing the pipeline those tokens rather than
        a string keeps the template from being applied twice and keeps a BOS
        token from being added a second time. On the reference host this
        produces the same tokens, text and stop handling as the history route.

        The pipeline's state is reset on both sides of the call. Before it, the
        stateful pipeline still holds the previous history call's chat and
        would read these tokens as a continuation of it. After it, the GPU
        pipeline keeps this call's state, and the next history call then fails
        inside the sampler ("token_idx < seq_len") or miscounts its prompt —
        both seen on the reference host with GenAI 2026.4. ``finish_chat`` is
        deprecated as an API to drive chats with, but it is the only reset the
        pipeline exposes; the next history call opens a fresh chat on its own.
        """
        tokenizer = self._pipeline.get_tokenizer()
        prompt = tokenizer.apply_chat_template(genai_history, add_generation_prompt=True)
        inputs = tokenizer.encode(prompt, add_special_tokens=False)
        with suppress(Exception):
            # Encoded inputs are never templated; this only states the intent.
            generation_config.apply_chat_template = False
        self._close_llm_chat()
        try:
            raw = self._pipeline.generate(inputs, generation_config, **_streamer_kwargs(streamer))
        finally:
            # Never let a failed reset hide the answer, or the generation error.
            with suppress(Exception):
                self._close_llm_chat()
        return _RenderedTurnResult(
            text=_decoded_tokens(tokenizer, raw),
            finish_reasons=getattr(raw, "finish_reasons", None),
            perf_metrics=getattr(raw, "perf_metrics", None),
        )

    def _close_llm_chat(self) -> None:
        """Reset the LLM pipeline's chat and generation state, quietly.

        A pipeline without ``finish_chat`` (a future GenAI may drop it) is left
        alone: there is then no chat to close.
        """
        finish_chat = getattr(self._pipeline, "finish_chat", None)
        if finish_chat is None:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            finish_chat()

    def _genai_history(self, history: ChatHistory) -> Any:
        """Our history as an ``openvino_genai.ChatHistory``, with the template variables.

        Tool definitions go in through ``set_tools`` and tool turns as the
        message fields the chat template reads (``tool_calls``,
        ``tool_call_id``), so the artifact's own template renders them. GenAI
        2026.4 fills in for a template that has no tool support of its own.
        """
        genai_history = self._genai.ChatHistory()
        vlm = self._pipeline_kind is PipelineKind.VLM
        for message in history.messages:
            payload = message.as_payload()
            if vlm:
                payload["content"] = _without_media_tags(message.content)
            genai_history.append(payload)
        if history.tools:
            genai_history.set_tools([dict(tool) for tool in history.tools])
        extra = thinking_extra_context(
            self._chat_profile, thinking=self._thinking, template=self._chat_template
        )
        if extra:
            try:
                genai_history.set_extra_context(dict(extra))
            except Exception as exc:
                # The template variable is a best-effort hint; the chat service's
                # sanitiser is the defence that actually hides reasoning text.
                self._extra_context_error = redact_text(f"{type(exc).__name__}: {exc}")
        return genai_history

    def _generate_vlm(
        self,
        history: ChatHistory,
        generation_config: Any,
        streamer: Any,
        images: Sequence[ChatImage] = (),
    ) -> Any:
        """Generate with a ``VLMPipeline``, statelessly where the build allows it.

        OpenVINO GenAI 2026.4 accepts a ``ChatHistory`` here, exactly as the LLM
        pipeline does, and then the whole conversation travels with every turn.
        A build without that overload rejects the call with ``TypeError`` before
        generating anything; from then on the deprecated chat lifecycle is driven
        instead, entirely inside the adapter, handing the pipeline only the turns
        it has not seen.

        ``images`` belong to the newest user turn and go to the pipeline as
        ``images=``, without tags in the text: GenAI then places them ahead of
        that turn's words, in order. A call without images passes no
        ``images`` argument at all, so it is exactly what it was before images.
        """
        extra = _streamer_kwargs(streamer)
        if images:
            extra["images"] = [_image_tensor(image) for image in images]
        if self._vlm_accepts_history is not False:
            try:
                result = self._pipeline.generate(
                    self._genai_history(history), generation_config=generation_config, **extra
                )
            except TypeError:
                if self._vlm_accepts_history:
                    raise
                self._vlm_accepts_history = False
            else:
                self._vlm_accepts_history = True
                return result
        if history.tools or any(m.role == "tool" or m.tool_calls for m in history.messages):
            # The fallback hands the pipeline bare user text: a tool definition,
            # call or result would be dropped without a word.
            raise GenerationError(
                diagnostic(
                    "runtime.genai.tools_unsupported",
                    FailureCategory.CONFIGURATION_ERROR,
                    "This OpenVINO GenAI build cannot pass tools to a VLM",
                    "Tool calling on a VLM needs a VLMPipeline that accepts a ChatHistory.",
                )
            )
        prompt = self._vlm_turn_prompt(history)
        return self._pipeline.generate(prompt, generation_config=generation_config, **extra)

    def _vlm_turn_prompt(self, history: ChatHistory) -> str:
        """Return the text to send to the VLM pipeline for this turn.

        The pipeline remembers the conversation, so only the turns it has not
        seen are handed over. Deciding *what it has not seen* is the whole
        difficulty: the adapter sees histories, not conversation identities.

        Three guards, outermost first:

        1. :meth:`_continues_conversation` — a continuation must extend what was
           already delivered, not merely match it. A history equal to the last
           one is a second conversation that happens to start with the same
           words, which is exactly what a benchmark's repeated prompt fixture and
           a ``/reset`` followed by retyping the same line both look like.
        2. A pending slice with no user content, in a history that *does* carry
           one, means the bookkeeping has lost track. Restart and resend rather
           than tell the caller there is nothing to answer.
        3. Only a history with no user turn at all — a system-only conversation —
           still raises, because that is a genuine caller bug.
        """
        messages = list(history.messages)
        if not self._continues_conversation(messages):
            self._restart_vlm_chat(messages)
        pending = messages[len(self._sent) :]
        if not _has_user_input(pending) and _has_user_input(messages):
            self._restart_vlm_chat(messages)
            pending = messages[len(self._sent) :]
        if not _has_user_input(pending):
            raise GenerationError(
                diagnostic(
                    "runtime.generation.empty_prompt",
                    FailureCategory.CONFIGURATION_ERROR,
                    "There is no new user message to answer",
                    "Add a user message to the conversation before generating.",
                )
            )
        self._sent = messages
        return _without_media_tags(_user_turn_text(pending))

    def _continues_conversation(self, messages: Sequence[ChatMessage]) -> bool:
        """Whether ``messages`` continues the conversation the pipeline is holding.

        Equality of content is not identity of conversation, so a history that is
        exactly what was already delivered is *not* a continuation: it is a new
        conversation whose opening happens to repeat. A continuation has to
        extend the delivered turns, which means being strictly longer than them.
        """
        if not self._chat_started:
            return False
        if not _is_prefix(self._sent, messages):
            return False
        return len(messages) > len(self._sent)

    def _restart_vlm_chat(self, messages: Sequence[ChatMessage]) -> None:
        """(Re)start the hidden VLM chat session, seeding it with the system turn."""
        self._end_vlm_chat()
        system = next((m.content for m in messages if m.role == "system"), "")
        try:
            if system:
                self._pipeline.start_chat(system)
            else:
                self._pipeline.start_chat()
        except Exception as exc:
            raise GenerationError(self._generation_failed_diagnostic(exc), cause=exc) from exc
        self._chat_started = True
        # start_chat has consumed the system turn; everything after it is pending.
        self._sent = _leading_system(messages)

    def _end_vlm_chat(self) -> None:
        """Close the hidden VLM session, if one is open, and forget what was sent."""
        if self._chat_started:
            with suppress(Exception):
                self._pipeline.finish_chat()
            self._chat_started = False
        self._sent = []

    def _generation_failed_diagnostic(
        self, error: BaseException, history: ChatHistory | None = None
    ) -> Diagnostic:
        """Build the generation failure diagnostic, keeping the raw runtime text.

        ``history`` is the conversation that failed, when there is one, so a
        refusal of its length can say what in it counts.
        """
        raw = redact_text(str(error))
        limit = _PROMPT_LIMIT.search(raw)
        if limit is not None:
            return self._prompt_too_long_diagnostic(
                raw, limit=int(limit.group(1)), passed=int(limit.group(2)), history=history
            )
        return diagnostic(
            "runtime.genai.generation_failed",
            FailureCategory.UNKNOWN_RUNTIME_ERROR,
            f"Generation failed on {self._device}",
            raw,
            remediation=[
                "Retry with fewer tokens, or on another device.",
                "Run `yonga doctor` if the device stopped responding.",
            ],
            raw_excerpt=raw,
            context={
                "device": self._device,
                "pipeline": self._pipeline_kind.value,
                "artifact_dir": str(self._artifact_dir),
            },
        )

    def _prompt_too_long_diagnostic(
        self, raw: str, *, limit: int, passed: int, history: ChatHistory | None
    ) -> Diagnostic:
        """The pipeline refused a prompt longer than the length it was compiled for.

        Nothing is wrong with the device or the model: the same pipeline answers
        a shorter prompt, so this is the request's to change. The message itself
        says how, because it is all a client of ``yonga serve`` sees, and it
        names the offered tools and the images when the conversation has any:
        they count as much as its text does.
        """
        tools = len(history.tools) if history is not None else 0
        images = sum(len(m.images) for m in history.messages) if history is not None else 0
        counted = [_count(tools, "offered tool"), _count(images, "image")]
        fewer = ["offer fewer tools" if tools else "", "send fewer images" if images else ""]
        counting = " and ".join(part for part in counted if part)
        advice = _one_of(["Shorten the conversation", *fewer])
        return diagnostic(
            PROMPT_TOO_LONG,
            FailureCategory.CONFIGURATION_ERROR,
            "The prompt is too long for this model",
            f"It has {passed} tokens{f', counting {counting}' if counting else ''}, and this "
            f"model on {self._device} takes at most {limit}: its prompt length (MAX_PROMPT_LEN) "
            f"is fixed when it is compiled. {advice}, or use a GPU or CPU artifact of this "
            "model, whose prompt length is not fixed.",
            remediation=[
                "Prepare the model for the GPU, whose prompt length is not fixed: "
                "yonga prepare MODEL --device gpu",
            ],
            raw_excerpt=raw,
            context={
                "device": self._device,
                "pipeline": self._pipeline_kind.value,
                "prompt_tokens": str(passed),
                "max_prompt_tokens": str(limit),
            },
        )

    # --- lifecycle -------------------------------------------------------

    def close(self) -> None:
        """Release the pipeline. Idempotent, and safe to call from a finally block."""
        if self._closed:
            return
        self._closed = True
        self._end_vlm_chat()
        self._pipeline = None

    def __enter__(self) -> GenAIChatRuntime:
        """Return the runtime so it can be used as a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the runtime when the ``with`` block ends."""
        self.close()


def open_runtime(
    artifact_dir: Path,
    *,
    device: str = _DEFAULT_DEVICE,
    pipeline: PipelineKind = PipelineKind.LLM,
    properties: Mapping[str, str] | None = None,
    chat_profile: ChatProfileLike | None = None,
    settings: Settings | None = None,
    thinking: bool | None = None,
) -> ChatRuntime:
    """Open a chat runtime for a prepared artifact.

    This is the single entry point the chat and bench services use. ``pipeline``
    comes from the artifact manifest's ``PipelineKind``;
    :attr:`~yonga.domain.models.PipelineKind.UNKNOWN` is driven as an LLM, which
    is what every text-only export is. ``thinking`` overrides the chat profile's
    ``default_thinking``.
    """
    return GenAIChatRuntime(
        artifact_dir,
        device=device,
        pipeline=pipeline,
        properties=properties,
        chat_profile=chat_profile,
        settings=settings,
        thinking=thinking,
    )
