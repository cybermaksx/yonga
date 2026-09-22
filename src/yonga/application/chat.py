"""The chat service: conversation state, generation policy and the reasoning sanitiser.

Three responsibilities live here and nowhere else.

*The sanitiser.* :func:`sanitize_reasoning` is a pure, total function that removes
recognised reasoning wrappers from raw model output. It is the layer that
actually guarantees hidden reasoning never reaches the screen, so it is written
defensively: an **unclosed** opening wrapper hides everything from the opener to
the end of the text, a closing wrapper with no opener hides everything back to
the last displayed boundary, and any internal failure hides the whole answer
rather than risk leaking a hidden block.

*The session.* :class:`ChatSession` owns one conversation: the pipeline is built
once by the injected ``runtime_factory`` and reused for every turn (AGENT_PROMPT
section 34), generation defaults come from the registry chat profile for the
model's architecture rather than from global constants (AGENT_PROMPT section 22),
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

from collections.abc import Mapping, Sequence
from contextlib import suppress
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from yonga.adapters.runtime_genai import (
    DEFAULT_REASONING_WRAPPERS,
    ChatHistory,
    ChatMessage,
    ChatProfile,
    ChatProfileLike,
    ChatRuntime,
    GenerationResult,
    Role,
)
from yonga.application.progress import ProgressEvent, ProgressKind, ProgressSink, null_sink
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
from yonga.registry.schema import ChatProfileRule, GenerationDefaults
from yonga.util.redaction import redact_text

__all__ = [
    "DEFAULT_REASONING_WRAPPERS",
    "MAX_OUTPUT_TOKENS",
    "TEMPERATURE_RANGE",
    "TOP_K_MAX",
    "TOP_P_RANGE",
    "ArtifactStoreLike",
    "ChatAction",
    "ChatCommandName",
    "ChatHistory",
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
    "ParsedCommand",
    "Role",
    "RuntimeFactory",
    "chat_history_filename",
    "sanitize_reasoning",
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


def _scan(text: str, pairs: Sequence[tuple[str, str]]) -> _Scan:
    """Split ``text`` into displayable segments and hidden reasoning blocks.

    A single left-to-right pass with a stack of expected closing wrappers, so
    nested and repeated blocks both work. Three rules make it safe rather than
    merely correct:

    * an opening wrapper that is never closed hides everything to end of text;
    * a closing wrapper with no matching opener hides everything back to the end
      of the previous block (the model's chat template can emit the opener
      itself, in which case the generated text starts *inside* the block);
    * only text outside every block is ever placed in a display segment.
    """
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

    if stack:
        blocks.append(text[block_start:])
        removed = True
    else:
        segments.append(text[run_start:])
    return _Scan(segments=segments, blocks=blocks, removed=removed)


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


def sanitize_reasoning(
    text: str,
    wrappers: Sequence[tuple[str, str]],
    *,
    show: bool,
) -> tuple[str, str | None]:
    """Split raw model output into what to display and what to keep hidden.

    Returns ``(display_text, reasoning_text)``. ``reasoning_text`` is ``None``
    when no reasoning block was found, or when every block was empty.

    ``wrappers`` are literal ``(opening, closing)`` pairs taken from the registry
    chat profile, never hard-coded here; matching is exact and case-sensitive, so
    ordinary prose containing "think", "thinking" or a literal ``<thinking>`` tag
    that is not a configured wrapper passes through untouched. Text with no block
    at all is returned byte-for-byte; whitespace is only normalised where a block
    was actually removed.

    With ``show=True`` the text is returned unchanged and the reasoning is still
    reported, so ``--show-reasoning`` can render it separately.

    The function is total. Any unexpected failure is treated as "we cannot prove
    this text is safe" and yields an empty display string rather than risking a
    leaked reasoning block.
    """
    raw = text if isinstance(text, str) else str(text)
    try:
        pairs = _normalise_wrappers(wrappers)
        if not pairs:
            return raw, None
        scan = _scan(raw, pairs)
        reasoning_blocks = [block.strip() for block in scan.blocks]
        reasoning = "\n\n".join(block for block in reasoning_blocks if block) or None
        if show:
            return raw, reasoning
        if not scan.removed:
            return raw, reasoning
        return _join_segments(scan.segments), reasoning
    except Exception:  # pragma: no cover - defensive; the scan above is total
        return (raw, None) if show else ("", None)


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
    supports_thinking_flag: bool = False
    thinking_template_variable: str | None = None
    default_thinking: bool = False
    reasoning_wrappers: list[tuple[str, str]] = Field(default_factory=_default_wrappers)
    defaults: GenerationDefaults = Field(default_factory=GenerationDefaults)
    thinking_defaults: GenerationDefaults = Field(default_factory=GenerationDefaults)
    stop_strings: list[str] = Field(default_factory=list)

    @classmethod
    def from_rule(cls, rule: ChatProfileRule) -> ChatProfileView:
        """Build the view from a resolved registry chat profile."""
        return cls(
            id=rule.id,
            summary=rule.summary,
            supports_thinking_flag=rule.supports_thinking_flag,
            thinking_template_variable=rule.thinking_template_variable,
            default_thinking=rule.default_thinking,
            reasoning_wrappers=list(rule.reasoning_wrappers) or _default_wrappers(),
            defaults=rule.defaults.model_copy(deep=True),
            thinking_defaults=rule.thinking_defaults.model_copy(deep=True),
            stop_strings=list(rule.stop_strings),
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
    """The outcome of one ``send``: either a result or a diagnostic, never both.

    A failed turn is a value, not an exception: the chat loop prints the
    diagnostic and returns to the prompt with the session and its loaded pipeline
    intact.
    """

    model_config = ConfigDict(extra="forbid")

    result: GenerationResult | None = None
    diagnostic: Diagnostic | None = None

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
    ) -> None:
        """Bind a session to an artifact, a chat profile and a runtime factory.

        Nothing is loaded here: :meth:`load` (or the first :meth:`send`) builds
        the pipeline, so the caller can draw the startup panel first.
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

        self._thinking = _resolve_thinking(profile, chosen)
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
        session and reused for every turn (AGENT_PROMPT section 34).
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
            runtime = self._runtime_factory(
                self._artifact_dir,
                device=self._device,
                pipeline=self._pipeline,
                properties=dict(self._properties),
                chat_profile=self._profile,
                settings=self._settings,
                thinking=self._thinking,
            )
            self._runtime = runtime
            self._runtime_loads += 1
        return runtime

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

    def send(self, user_text: str) -> ChatTurn:
        """Answer one user message.

        The raw model text is sanitised before it is returned *and* before it is
        stored in the conversation, so a hidden reasoning block can neither reach
        the screen nor be fed back as context on the next turn. A failure comes
        back as a :class:`ChatTurn` carrying a diagnostic; the session, its
        history and its loaded pipeline all survive it.
        """
        prompt = user_text.strip()
        if not prompt:
            return ChatTurn(diagnostic=_empty_prompt_diagnostic())
        self.history.add_user(prompt)
        try:
            runtime = self.load()
            self._emit(ProgressKind.STAGE, "Generating a reply...")
            raw = runtime.generate(self.history, self._generation)
            display, reasoning = sanitize_reasoning(
                raw.text,
                self._profile.reasoning_wrappers,
                show=self._show_reasoning,
            )
            result = raw.model_copy(update={"display_text": display, "reasoning_text": reasoning})
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

    def _rollback_user_turn(self) -> None:
        """Drop the user message that produced no answer, keeping the history consistent."""
        messages = self.history.messages
        if messages and messages[-1].role == "user":
            messages.pop()

    def reset(self) -> None:
        """Forget the conversation, keeping the session's system prompt.

        The pipeline is deliberately untouched: for the LLM path the whole
        conversation is re-supplied on every turn, so there is nothing stale to
        discard and nothing to reload.
        """
        self.history.reset(keep_system=True)

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
        if self._profile.supports_thinking_flag and not self._thinking:
            return CommandOutcome(
                message=(
                    "Reasoning blocks are now shown. Thinking mode is off for this session, "
                    "so the model may not produce any; restart with --thinking to enable it."
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
        """Extra startup-panel rows describing the generation policy in force."""
        extras: dict[str, str] = {}
        if self._profile.id:
            extras["Chat profile"] = self._profile.id
        if self._profile.supports_thinking_flag:
            extras["Thinking mode"] = "on" if self._thinking else "off"
        sampling = _sampling_summary(self._generation)
        if sampling:
            extras["Sampling"] = sampling
        extras["Metrics footer"] = "on" if self._stats else "off"
        return extras

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


def _resolve_thinking(profile: ChatProfileView, options: ChatOptions) -> bool:
    """Decide whether this session asks the chat template to enable thinking.

    An explicit ``--thinking``/``--no-thinking`` wins. Otherwise asking to *see*
    reasoning on a family that has a thinking flag is taken as asking for it,
    since the alternative is a flag that visibly does nothing. Everything else
    follows the profile, which defaults to off.
    """
    if options.thinking is not None:
        return options.thinking
    if options.show_reasoning and profile.supports_thinking_flag:
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
        )

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
        """Ask the resolver for this artifact's chat rule, tolerating a registry fault."""
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
