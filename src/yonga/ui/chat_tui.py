"""Presentation layer for ``yonga chat``: panels, prompt, command parsing, metrics.

This module draws the chat session and parses what the user types. It does not
run a model, it does not own the conversation and — importantly — it does not
sanitise reasoning: hiding ``<think>`` blocks is the chat *service*'s job
(``yonga.application.chat``), because getting that wrong leaks hidden text.
Here we only render the text we are handed.

``prompt_toolkit`` is imported lazily inside :meth:`ChatView.prompt` so that
``--json`` runs, piped runs and ``import yonga.cli`` never construct a terminal
session.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field
from rich.box import ROUNDED
from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from yonga.ui.console import YongaConsole

if TYPE_CHECKING:
    from yonga.config.paths import Paths

__all__ = [
    "CHAT_COMMANDS",
    "ChatCommand",
    "ChatCommandKind",
    "ChatMetrics",
    "ChatSessionInfo",
    "ChatView",
    "help_text",
    "parse_command",
    "render_answer",
    "render_metrics",
    "session_facts",
    "startup_panel",
]

#: Largest value ``/tokens`` will accept; anything bigger is a typo, not a wish.
MAX_OUTPUT_TOKENS = 1_048_576
#: Inclusive bounds for ``/temp``.
TEMPERATURE_RANGE = (0.0, 2.0)
#: Inclusive bounds for ``/top-p``.
TOP_P_RANGE = (0.0, 1.0)


class ChatCommandKind(str, Enum):
    """Every slash command the chat loop understands."""

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


class ChatCommand(BaseModel):
    """A parsed slash command, with its validated argument or an error message.

    The UI parses and reports; the chat service executes. ``error`` is set — and
    is user-facing English — whenever the command was recognised but its
    argument was not usable, or the command itself is unknown.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ChatCommandKind
    raw: str
    name: str = ""
    argument: str = ""
    int_value: int | None = None
    float_value: float | None = None
    bool_value: bool | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the command and its argument are both usable."""
        return self.error is None


class _Spec(BaseModel):
    """Internal description of one command's name, aliases and help line."""

    model_config = ConfigDict(frozen=True)

    kind: ChatCommandKind
    usage: str
    help: str
    aliases: tuple[str, ...] = Field(default_factory=tuple)


#: The command table, in the order ``/help`` prints it.
CHAT_COMMANDS: tuple[_Spec, ...] = (
    _Spec(kind=ChatCommandKind.HELP, usage="/help", help="Show this command list."),
    _Spec(kind=ChatCommandKind.RESET, usage="/reset", help="Forget the conversation history."),
    _Spec(kind=ChatCommandKind.CLEAR, usage="/clear", help="Clear the screen."),
    _Spec(
        kind=ChatCommandKind.TOKENS,
        usage="/tokens N",
        help="Set the maximum number of tokens in a reply.",
    ),
    _Spec(
        kind=ChatCommandKind.TEMPERATURE,
        usage="/temp X",
        help="Set the sampling temperature (0.0-2.0).",
        aliases=("temperature",),
    ),
    _Spec(kind=ChatCommandKind.TOP_P, usage="/top-p X", help="Set nucleus sampling (0.0-1.0)."),
    _Spec(kind=ChatCommandKind.TOP_K, usage="/top-k N", help="Set top-k sampling (0 disables it)."),
    _Spec(
        kind=ChatCommandKind.REASONING,
        usage="/reasoning on|off",
        help="Show or hide the model's reasoning block.",
    ),
    _Spec(
        kind=ChatCommandKind.STATS,
        usage="/stats on|off",
        help="Show or hide the metrics footer.",
    ),
    _Spec(
        kind=ChatCommandKind.MODEL,
        usage="/model",
        help="Show the loaded model, device and profile.",
    ),
    _Spec(
        kind=ChatCommandKind.EXIT,
        usage="/exit",
        help="Leave the chat session.",
        aliases=("quit", "q", "bye"),
    ),
)

_BY_NAME: dict[str, _Spec] = {}
for _spec in CHAT_COMMANDS:
    _BY_NAME[_spec.kind.value] = _spec
    for _alias in _spec.aliases:
        _BY_NAME[_alias] = _spec

#: Commands that never take an argument.
_NO_ARGUMENT: frozenset[ChatCommandKind] = frozenset(
    {
        ChatCommandKind.HELP,
        ChatCommandKind.RESET,
        ChatCommandKind.CLEAR,
        ChatCommandKind.MODEL,
        ChatCommandKind.EXIT,
    }
)

_TRUE_WORDS: frozenset[str] = frozenset({"on", "true", "yes", "1", "show"})
_FALSE_WORDS: frozenset[str] = frozenset({"off", "false", "no", "0", "hide"})


def _normalise_name(name: str) -> str:
    """Canonicalise a command name: lowercase, underscores become hyphens."""
    return name.strip().lower().replace("_", "-")


def _fail(kind: ChatCommandKind, raw: str, name: str, argument: str, error: str) -> ChatCommand:
    """Build a rejected command carrying a user-facing explanation."""
    return ChatCommand(kind=kind, raw=raw, name=name, argument=argument, error=error)


def parse_command(line: str) -> ChatCommand | None:
    """Parse one input line into a :class:`ChatCommand`, or ``None`` if it is a prompt.

    A line is a command only when it starts with ``/``. Recognised commands with
    an unusable argument come back with ``error`` set rather than raising, so the
    chat loop can print the message and keep going. This function is pure.
    """
    stripped = line.strip()
    if not stripped.startswith("/") or stripped == "/":
        return None

    body = stripped[1:]
    head, _, tail = body.partition(" ")
    name = _normalise_name(head)
    argument = tail.strip()

    spec = _BY_NAME.get(name)
    if spec is None:
        return _fail(
            ChatCommandKind.UNKNOWN,
            stripped,
            name,
            argument,
            f"Unknown command '/{head}'. Type /help to see the available commands.",
        )

    kind = spec.kind
    if kind in _NO_ARGUMENT:
        if argument:
            return _fail(kind, stripped, name, argument, f"{spec.usage} does not take an argument.")
        return ChatCommand(kind=kind, raw=stripped, name=name)

    if not argument:
        return _fail(kind, stripped, name, argument, f"Usage: {spec.usage}")

    if kind is ChatCommandKind.TOKENS:
        return _parse_int(
            kind, stripped, name, argument, spec, minimum=1, maximum=MAX_OUTPUT_TOKENS
        )
    if kind is ChatCommandKind.TOP_K:
        return _parse_int(
            kind, stripped, name, argument, spec, minimum=0, maximum=MAX_OUTPUT_TOKENS
        )
    if kind is ChatCommandKind.TEMPERATURE:
        return _parse_float(kind, stripped, name, argument, spec, bounds=TEMPERATURE_RANGE)
    if kind is ChatCommandKind.TOP_P:
        return _parse_float(kind, stripped, name, argument, spec, bounds=TOP_P_RANGE)
    return _parse_switch(kind, stripped, name, argument, spec)


def _parse_int(
    kind: ChatCommandKind,
    raw: str,
    name: str,
    argument: str,
    spec: _Spec,
    *,
    minimum: int,
    maximum: int,
) -> ChatCommand:
    """Parse and range-check a whole-number argument."""
    try:
        value = int(argument, 10)
    except ValueError:
        return _fail(
            kind,
            raw,
            name,
            argument,
            f"'{argument}' is not a whole number. Usage: {spec.usage}",
        )
    if value < minimum or value > maximum:
        return _fail(
            kind,
            raw,
            name,
            argument,
            f"{spec.usage} expects a value between {minimum} and {maximum}; got {value}.",
        )
    return ChatCommand(kind=kind, raw=raw, name=name, argument=argument, int_value=value)


def _parse_float(
    kind: ChatCommandKind,
    raw: str,
    name: str,
    argument: str,
    spec: _Spec,
    *,
    bounds: tuple[float, float],
) -> ChatCommand:
    """Parse and range-check a decimal argument."""
    try:
        value = float(argument)
    except ValueError:
        return _fail(
            kind, raw, name, argument, f"'{argument}' is not a number. Usage: {spec.usage}"
        )
    if value != value or value in (float("inf"), float("-inf")):
        return _fail(
            kind, raw, name, argument, f"'{argument}' is not a finite number. Usage: {spec.usage}"
        )
    low, high = bounds
    if value < low or value > high:
        return _fail(
            kind,
            raw,
            name,
            argument,
            f"{spec.usage} expects a value between {low:g} and {high:g}; got {value:g}.",
        )
    return ChatCommand(kind=kind, raw=raw, name=name, argument=argument, float_value=value)


def _parse_switch(
    kind: ChatCommandKind, raw: str, name: str, argument: str, spec: _Spec
) -> ChatCommand:
    """Parse an ``on``/``off`` argument."""
    word = argument.strip().lower()
    if word in _TRUE_WORDS:
        return ChatCommand(kind=kind, raw=raw, name=name, argument=argument, bool_value=True)
    if word in _FALSE_WORDS:
        return ChatCommand(kind=kind, raw=raw, name=name, argument=argument, bool_value=False)
    return _fail(
        kind, raw, name, argument, f"Expected 'on' or 'off', not '{argument}'. Usage: {spec.usage}"
    )


@runtime_checkable
class ChatMetrics(Protocol):
    """What :func:`render_metrics` reads from a generation result.

    ``yonga.application.chat.GenerationResult`` satisfies this structurally; the
    protocol keeps the UI from importing the application layer, and every member
    is optional in practice because a runtime may not report it.
    """

    @property
    def input_tokens(self) -> int | None:
        """Prompt tokens fed to the model, when the runtime reported them."""

    @property
    def output_tokens(self) -> int | None:
        """Generated tokens, when the runtime reported them."""

    @property
    def ttft_ms(self) -> float | None:
        """Time to first token in milliseconds."""

    @property
    def throughput_tps(self) -> float | None:
        """Generation throughput in tokens per second."""

    @property
    def tpot_ms(self) -> float | None:
        """Time per output token in milliseconds."""


class ChatSessionInfo(BaseModel):
    """Everything the startup panel and ``/model`` need to describe a session.

    The chat service fills this in from the artifact manifest; the UI only reads
    it, which is what keeps this module free of adapter imports.
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

    @property
    def device_label(self) -> str:
        """``Intel Arc Graphics (GPU)`` when the full name is known, else ``GPU``."""
        if self.device_full_name:
            return f"{self.device_full_name} ({self.device})"
        return self.device

    @property
    def source_label(self) -> str:
        """``repo@abcdef123456`` with the revision abbreviated, or an em dash."""
        if not self.repo_id:
            return "—"
        revision = _abbreviate_revision(self.revision)
        return f"{self.repo_id}@{revision}" if revision else self.repo_id

    @property
    def history_filename(self) -> str:
        """Name of this session's persistent prompt-history file."""
        stem = "".join(
            char if char.isalnum() or char in "-._" else "-"
            for char in (self.artifact_id or self.repo_id or self.display_name or "session")
        ).strip("-")
        return f"{stem or 'session'}.history"


def _abbreviate_revision(revision: str | None, *, length: int = 12) -> str | None:
    """Shorten a 40-character commit sha; leave branch and tag names alone."""
    if revision is None:
        return None
    cleaned = revision.strip()
    if not cleaned:
        return None
    if len(cleaned) == 40 and all(char in "0123456789abcdef" for char in cleaned.lower()):
        return cleaned[:length]
    return cleaned


def startup_panel(info: ChatSessionInfo) -> RenderableType:
    """Render the panel shown when a chat session opens (``AGENT_PROMPT`` 22)."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True, style="dim")
    grid.add_column(overflow="fold")

    grid.add_row("Model", Text(info.display_name, style="bold"))
    grid.add_row("Source", Text(info.source_label))
    if info.profile:
        grid.add_row("Profile", Text(info.profile))
    grid.add_row("Device", Text(info.device_label))
    if info.max_new_tokens is not None:
        grid.add_row("Max output", Text(f"{info.max_new_tokens} tokens"))
    grid.add_row(
        "Reasoning display",
        Text("shown", style="yellow") if info.show_reasoning else Text("hidden", style="green"),
    )
    cache = Text("enabled", style="green") if info.compile_cache_enabled else Text("off")
    if info.compile_cache_enabled and info.compile_cache_path is not None:
        cache = Text.assemble(cache, ("  ", ""), (str(info.compile_cache_path), "dim"))
    grid.add_row("Compile cache", cache)
    for key, value in info.extra.items():
        grid.add_row(key, Text(value))

    return Panel(
        grid,
        title="Local Chat",
        subtitle="/help for commands",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2),
    )


def render_answer(markdown_text: str) -> RenderableType:
    """Render an already-sanitised answer as a Markdown panel.

    The text handed in must already have had any hidden reasoning removed by
    ``yonga.application.chat``; this function trusts and renders it.
    """
    body = markdown_text.strip()
    if not body:
        return Panel(
            Text("(the model returned no text)", style="dim"),
            border_style="dim",
            box=ROUNDED,
            padding=(1, 2),
        )
    return Panel(Markdown(body), border_style="green", box=ROUNDED, padding=(1, 2))


def render_metrics(result: ChatMetrics | None) -> Text:
    """Render the dim footer, omitting every metric the runtime did not report.

    With everything available this is::

        TTFT 180 ms • 24.6 tok/s • TPOT 40.6 ms/token • 512 → 380 tokens

    A missing metric is simply left out, and a result with no metrics at all
    renders as empty text. This never raises: a runtime that reports nothing
    must not be able to break the chat loop.
    """
    if result is None:
        return Text("", style="dim")

    parts: list[str] = []
    ttft = _finite(_attribute(result, "ttft_ms"))
    if ttft is not None:
        parts.append(f"TTFT {ttft:.0f} ms")
    throughput = _finite(_attribute(result, "throughput_tps"))
    if throughput is not None:
        parts.append(f"{throughput:.1f} tok/s")
    tpot = _finite(_attribute(result, "tpot_ms"))
    if tpot is not None:
        parts.append(f"TPOT {tpot:.1f} ms/token")

    input_tokens = _whole(_attribute(result, "input_tokens"))
    output_tokens = _whole(_attribute(result, "output_tokens"))
    if input_tokens is not None and output_tokens is not None:
        parts.append(f"{input_tokens} → {output_tokens} tokens")
    elif output_tokens is not None:
        parts.append(f"{output_tokens} output tokens")
    elif input_tokens is not None:
        parts.append(f"{input_tokens} input tokens")

    return Text(" • ".join(parts), style="dim")


def _attribute(result: object, name: str) -> object:
    """Read an optional metric attribute without ever raising."""
    try:
        return getattr(result, name, None)
    except Exception:
        return None


def _finite(value: object) -> float | None:
    """Coerce a metric to a finite float, or ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _whole(value: object) -> int | None:
    """Coerce a token count to a non-negative integer, or ``None``."""
    number = _finite(value)
    if number is None or number < 0:
        return None
    return int(number)


def help_text() -> RenderableType:
    """Render the ``/help`` table listing every command and what it does."""
    table = Table.grid(padding=(0, 3))
    table.add_column(no_wrap=True, style="bold")
    table.add_column(overflow="fold")
    for spec in CHAT_COMMANDS:
        table.add_row(spec.usage, Text(spec.help))
    return Group(
        Text("Commands", style="bold"),
        table,
        Text("Anything else is sent to the model.", style="dim"),
    )


class ChatView:
    """Draws one chat session and reads the user's input line.

    The view owns no conversation state: it renders what the chat service gives
    it and returns what the user typed.
    """

    def __init__(
        self,
        console: YongaConsole,
        info: ChatSessionInfo,
        *,
        paths: Paths | None = None,
        history_dir: Path | None = None,
    ) -> None:
        """Bind the view to a console and a session description.

        ``paths`` (or an explicit ``history_dir``) decides where the persistent
        prompt history lives; without either, history is kept in memory only.
        """
        self.console = console
        self.info = info
        self._history_dir = history_dir if history_dir is not None else _chat_history_dir(paths)
        self._session: Any = None

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------
    @property
    def history_path(self) -> Path | None:
        """``<chat history dir>/<artifact id>.history``, or ``None`` when unset."""
        if self._history_dir is None:
            return None
        return self._history_dir / self.info.history_filename

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def startup_panel(self) -> RenderableType:
        """Return the startup panel for this session."""
        return startup_panel(self.info)

    def show_startup(self) -> None:
        """Print the startup panel."""
        self.console.print(self.startup_panel())

    def show_help(self) -> None:
        """Print the command list."""
        self.console.print(help_text())

    def show_model(self) -> None:
        """Print the session description again, as ``/model`` does."""
        self.console.print(self.startup_panel())

    def render_answer(self, markdown_text: str) -> RenderableType:
        """Return the model's answer as a Markdown panel."""
        return render_answer(markdown_text)

    def show_answer(self, markdown_text: str) -> None:
        """Print the model's answer."""
        self.console.print(self.render_answer(markdown_text))

    def render_metrics(self, result: ChatMetrics | None) -> Text:
        """Return the dim metrics footer for ``result``."""
        return render_metrics(result)

    def show_metrics(self, result: ChatMetrics | None) -> None:
        """Print the metrics footer, unless there is nothing to show."""
        footer = self.render_metrics(result)
        if footer.plain:
            self.console.print(footer)

    def show_error(self, message: str) -> None:
        """Print a short, non-fatal chat error such as a bad command argument."""
        self.console.print_error(Text(message, style="yellow"))

    def clear(self) -> None:
        """Clear the screen, when the terminal supports it."""
        if self.console.show_decoration and self.console.out.is_terminal:
            self.console.out.clear()

    @contextmanager
    def generating(self, message: str = "Generating a reply...") -> Iterator[None]:
        """Show a spinner while the model runs; silent outside a terminal."""
        if not self.console.animate:
            yield
            return
        with self.console.err.status(Text(message, style="dim"), spinner="dots"):
            yield

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def prompt(self, message: str = "you > ") -> str:
        """Read one line from the user, with history and line editing.

        ``prompt_toolkit`` is imported here, not at module scope, so a
        non-interactive or ``--json`` run never builds a terminal session. Outside
        a terminal the line is read from stdin instead. ``EOFError`` (Ctrl-D) and
        ``KeyboardInterrupt`` (Ctrl-C) propagate so the chat loop can decide what
        they mean.
        """
        if not self.console.interactive:
            return input()
        session = self._prompt_session()
        line = session.prompt(message)
        return line if isinstance(line, str) else str(line)

    def _prompt_session(self) -> Any:
        """Build (once) the prompt_toolkit session, with persistent history."""
        if self._session is None:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import FileHistory, History, InMemoryHistory

            history: History = InMemoryHistory()
            path = self.history_path
            if path is not None:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    history = FileHistory(str(path))
                except OSError:
                    # A read-only data directory is not a reason to refuse a chat.
                    history = InMemoryHistory()
            self._session = PromptSession(history=history, enable_history_search=True)
        return self._session

    def close(self) -> None:
        """Drop the prompt session so its file handles are released."""
        self._session = None


def _chat_history_dir(paths: Paths | None) -> Path | None:
    """Return ``Paths.chat_history_dir`` when paths were supplied."""
    if paths is None:
        return None
    return paths.chat_history_dir


def session_facts(info: ChatSessionInfo) -> Mapping[str, str]:
    """Return the session description as flat strings, for ``--json`` output."""
    facts: dict[str, str] = {
        "display_name": info.display_name,
        "device": info.device_label,
        "reasoning": "shown" if info.show_reasoning else "hidden",
        "compile_cache": "enabled" if info.compile_cache_enabled else "off",
    }
    if info.artifact_id:
        facts["artifact_id"] = info.artifact_id
    if info.repo_id:
        facts["source"] = info.source_label
    if info.profile:
        facts["profile"] = info.profile
    if info.max_new_tokens is not None:
        facts["max_new_tokens"] = str(info.max_new_tokens)
    return facts
