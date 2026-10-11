"""Terminal output policy: colour, JSON mode, error rendering and confirmations.

This module is the single place that decides *how* yonga talks to a terminal. It
wraps :class:`rich.console.Console` and adds the project's output rules:

* ``NO_COLOR`` (and a non-TTY stdout) disable ANSI styling and animation;
* ``json_mode`` suppresses every decorative byte so ``--json`` writes exactly one
  JSON document to stdout;
* failures are rendered as short, plain-English explanations — never tracebacks —
  with the raw exception text shown only under ``--debug``;
* ``--yes`` may never auto-approve a security-sensitive question;
* a ``str`` is always plain text. Rich console markup is switched off, so a
  bracketed word in a message, a remediation command (``pip install
  'yonga[openvino]'``), a model id or a path is printed as written instead of
  being read as a style tag. Styling is applied with :class:`rich.text.Text`.

The UI renders; it never decides. Nothing here imports adapters, OpenVINO or the
Hugging Face Hub.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePath
from typing import IO

from pydantic import BaseModel
from rich.console import Console, Group, RenderableType
from rich.text import Text

from yonga.domain.diagnostics import Diagnostic, Severity, fill_placeholders
from yonga.util.redaction import redact_mapping, redact_text

__all__ = [
    "BYTE_UNITS",
    "SEVERITY_ICONS",
    "SEVERITY_STYLES",
    "YongaConsole",
    "build_console",
    "diagnostic_lines",
    "human_bytes",
    "human_duration",
    "json_ready",
    "render_diagnostic",
    "status_icon",
    "status_style",
]

#: Binary unit ladder used by :func:`human_bytes`.
BYTE_UNITS: tuple[str, ...] = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB")

#: Single-character status glyphs, keyed by severity.
#:
#: ``Severity`` has five members but only four states are meaningful to a reader,
#: so ``INFO`` — which is neither a pass nor a failure — gets a neutral bullet.
SEVERITY_ICONS: Mapping[Severity, str] = {
    Severity.OK: "✓",
    Severity.INFO: "·",
    Severity.WARNING: "!",
    Severity.ERROR: "✗",
    Severity.UNKNOWN: "?",
}

#: Rich style names, keyed by severity.
SEVERITY_STYLES: Mapping[Severity, str] = {
    Severity.OK: "green",
    Severity.INFO: "dim",
    Severity.WARNING: "yellow",
    Severity.ERROR: "red",
    Severity.UNKNOWN: "dim",
}

_UNKNOWN = "unknown"


def status_icon(severity: Severity) -> str:
    """Return the one-character glyph that represents ``severity``."""
    return SEVERITY_ICONS.get(severity, SEVERITY_ICONS[Severity.UNKNOWN])


def status_style(severity: Severity) -> str:
    """Return the Rich style name that represents ``severity``."""
    return SEVERITY_STYLES.get(severity, SEVERITY_STYLES[Severity.UNKNOWN])


def _as_float(value: object) -> float | None:
    """Coerce ``value`` to a finite float, or return ``None`` when impossible."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def human_bytes(value: int | float | None, *, precision: int | None = None) -> str:
    """Format a byte count using binary units (``1.5 KiB``, ``796 GiB``).

    The default precision is chosen for readability: whole bytes and values of
    ten or more print without decimals, smaller values keep one. Pass
    ``precision`` to force a fixed number of decimals (``18.8 GiB``). ``None``
    renders as ``"unknown"`` and negative values are clamped to zero.
    """
    amount = _as_float(value)
    if amount is None:
        return _UNKNOWN
    amount = max(amount, 0.0)

    index = 0
    while amount >= 1024.0 and index < len(BYTE_UNITS) - 1:
        amount /= 1024.0
        index += 1
    if index == 0:
        return f"{int(amount)} {BYTE_UNITS[0]}"

    digits = precision if precision is not None else (0 if amount >= 10 else 1)
    rendered = f"{amount:.{digits}f}"
    # Rounding can push the value back over the next unit boundary (1023.9 KiB).
    if float(rendered) >= 1024.0 and index < len(BYTE_UNITS) - 1:
        amount /= 1024.0
        index += 1
        digits = precision if precision is not None else (0 if amount >= 10 else 1)
        rendered = f"{amount:.{digits}f}"
    return f"{rendered} {BYTE_UNITS[index]}"


def human_duration(seconds: float | None) -> str:
    """Format a duration in seconds as ``180 ms`` / ``10.8 s`` / ``3m 24s`` / ``1h 05m``."""
    total = _as_float(seconds)
    if total is None:
        return _UNKNOWN
    total = max(total, 0.0)

    if total < 1.0:
        milliseconds = total * 1000.0
        if 0.0 < milliseconds < 1.0:
            return "<1 ms"
        return f"{milliseconds:.0f} ms"
    if total < 60.0:
        return f"{total:.1f} s"

    whole = round(total)
    minutes, remaining_seconds = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h {remaining_minutes:02d}m"


def json_ready(value: object) -> object:
    """Convert ``value`` into something :func:`json.dumps` accepts.

    Handles the types the domain layer actually produces: ``Path``, ``datetime``,
    ``Enum``, ``Decimal``, sets, tuples and Pydantic models. Anything else falls
    back to ``str``.
    """
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, BaseModel):
        return json_ready(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, PurePath | Path):
        return str(value)
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, Enum):
        return json_ready(value.value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes | bytearray):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, set | frozenset):
        return [json_ready(item) for item in sorted(value, key=str)]
    if isinstance(value, Sequence):
        return [json_ready(item) for item in value]
    if isinstance(value, Iterable):
        return [json_ready(item) for item in value]
    return str(value)


def _fallback_command(item: Diagnostic) -> str | None:
    """Return the command the user should run next, when one can be derived."""
    explicit = item.context.get("fallback_command")
    if explicit:
        return explicit
    if not item.fallback_devices:
        return None
    model = item.context.get("model") or item.context.get("repo_id")
    if not model:
        return None
    command = item.context.get("command", "prepare")
    device = item.fallback_devices[0].lower()
    return f"yonga {command} {model} --device {device}"


def diagnostic_lines(
    item: Diagnostic, *, debug: bool = False, debug_command: str | None = None
) -> list[str]:
    """Render ``item`` as plain text lines, in the order a reader needs them.

    This is the pure core shared by the coloured renderer and by the undecorated
    ``--json`` error output, which makes the wording testable without a terminal.
    The raw exception excerpt is included only when ``debug`` is true; in normal
    mode a traceback is never shown. Remediation placeholders are filled in
    here as well, so every surface prints the same copy-pasteable commands.
    ``debug_command`` is the user's own command with ``--debug`` added, which
    the hint about the raw error prints: ``--debug`` is a root option, so
    appending it to the command is a usage error.
    """
    item = fill_placeholders(item)
    lines: list[str] = [item.title.strip() or item.code]
    message = item.message.strip()
    if message:
        lines.extend(("", message))

    remediation = [step.strip() for step in item.remediation if step.strip()]
    if remediation:
        lines.append("")
        lines.extend(f"  - {step}" for step in remediation)

    fallback_devices = [name for name in item.fallback_devices if name]
    if fallback_devices:
        lines.extend(("", f"Recommended fallback: {', '.join(fallback_devices)}"))
    command = _fallback_command(item)
    if command:
        if not fallback_devices:
            lines.append("")
        lines.append(f"Run: {command}")

    if item.log_path is not None:
        lines.extend(("", f"Job log: {item.log_path}"))
    elif item.job_id:
        lines.extend(("", f"Job id: {item.job_id}"))

    if debug and item.raw_excerpt:
        lines.extend(("", "Raw error (--debug):"))
        lines.extend(f"  {line}" for line in redact_text(item.raw_excerpt).splitlines())
    elif item.raw_excerpt and debug_command:
        lines.extend(("", f"Re-run with --debug to see the raw error: {debug_command}"))
    elif item.raw_excerpt:
        lines.extend(("", "Re-run with --debug to see the raw error."))
    return lines


def render_diagnostic(
    item: Diagnostic, *, debug: bool = False, debug_command: str | None = None
) -> RenderableType:
    """Return a styled renderable for ``item`` (see :func:`diagnostic_lines`)."""
    severity = item.severity
    header = Text()
    header.append(f"{status_icon(severity)} ", style=status_style(severity))
    header.append(item.title.strip() or item.code, style=f"bold {status_style(severity)}")

    body = Text()
    for line in diagnostic_lines(item, debug=debug, debug_command=debug_command)[1:]:
        style = "dim" if line.startswith(("Job log:", "Job id:", "Re-run with")) else ""
        body.append(line, style=style)
        body.append("\n")
    return Group(header, body)


def _no_color_requested(env: Mapping[str, str]) -> bool:
    """Return whether the environment asks for uncoloured output."""
    if env.get("NO_COLOR") is not None:
        return True
    return env.get("TERM", "").strip().lower() == "dumb"


def _force_color_requested(env: Mapping[str, str]) -> bool:
    """Return whether the environment explicitly asks for colour anyway."""
    return bool(env.get("FORCE_COLOR"))


def build_console(
    *,
    stderr: bool = False,
    no_color: bool = False,
    file: IO[str] | None = None,
    force_terminal: bool | None = None,
    width: int | None = None,
) -> Console:
    """Create a Rich console with yonga's output policy already applied.

    When ``no_color`` is set the colour system is switched off entirely rather
    than merely desaturated, so ``NO_COLOR`` also removes bold and dim
    attributes: a terminal that asked for no escape sequences gets none.

    Markup is off: every string this console renders -- a printed line, a table
    cell, a column header -- is literal text, because most of them carry text
    yonga did not write (diagnostics, model ids, paths, upstream errors).
    """
    return Console(
        markup=False,
        file=file,
        stderr=stderr,
        no_color=no_color,
        color_system=None if no_color else "auto",
        highlight=False,
        soft_wrap=False,
        emoji=False,
        force_terminal=force_terminal,
        width=width,
    )


class YongaConsole:
    """The tool's terminal front end: one stdout console and one stderr console.

    ``stdout`` carries results (including the single ``--json`` document);
    ``stderr`` carries diagnostics, prompts and progress, so a piped stdout stays
    machine readable. Construct it with explicit ``out``/``err`` consoles in
    tests to capture output.
    """

    def __init__(
        self,
        *,
        json_mode: bool = False,
        verbose: bool = False,
        debug: bool = False,
        out: Console | None = None,
        err: Console | None = None,
        env: Mapping[str, str] | None = None,
        interactive: bool | None = None,
    ) -> None:
        """Build a console pair, honouring ``NO_COLOR`` and ``json_mode``."""
        self._env: dict[str, str] = dict(os.environ if env is None else env)
        self.json_mode = json_mode
        self.verbose = verbose
        self.debug = debug
        self.no_color = _no_color_requested(self._env) and not _force_color_requested(self._env)
        self._interactive = interactive

        self.out: Console = out if out is not None else build_console(no_color=self.no_color)
        self.err: Console = (
            err if err is not None else build_console(stderr=True, no_color=self.no_color)
        )

    # ------------------------------------------------------------------
    # capabilities
    # ------------------------------------------------------------------
    @property
    def show_decoration(self) -> bool:
        """True when panels, rules and tables may be drawn at all."""
        return not self.json_mode

    @property
    def animate(self) -> bool:
        """True when spinners and live progress bars are appropriate."""
        return self.show_decoration and self.err.is_terminal

    @property
    def color_enabled(self) -> bool:
        """True when ANSI styling will actually be emitted on stdout."""
        return self.out.color_system is not None and self.out.is_terminal

    @property
    def interactive(self) -> bool:
        """True when a question may be put to a human on this session."""
        if self._interactive is not None:
            return self._interactive
        if self.json_mode:
            return False
        return self.err.is_terminal and _stdin_is_tty()

    # ------------------------------------------------------------------
    # plain output
    # ------------------------------------------------------------------
    def print(self, renderable: RenderableType = "", *, style: str | None = None) -> None:
        """Write a renderable to stdout; silent in ``--json`` mode. A ``str`` is plain text."""
        if not self.show_decoration:
            return
        self.out.print(_plain(renderable), style=style, highlight=False)

    def print_error(self, renderable: RenderableType = "", *, style: str | None = None) -> None:
        """Write a renderable to stderr; silent in ``--json`` mode. A ``str`` is plain text."""
        if not self.show_decoration:
            return
        self.err.print(_plain(renderable), style=style, highlight=False)

    def detail(self, message: str) -> None:
        """Write a dim line to stderr, but only under ``--verbose``."""
        if not self.verbose:
            return
        self.print_error(Text(message, style="dim"))

    def blank(self) -> None:
        """Write one empty line to stdout; silent in ``--json`` mode."""
        self.print("")

    def rule(self, title: str = "") -> None:
        """Draw a horizontal rule with an optional title; silent in ``--json`` mode."""
        if not self.show_decoration:
            return
        self.out.rule(Text(title, style="bold") if title else "", style="dim")

    def section(self, title: str) -> None:
        """Start a titled section: a blank line then a bold heading."""
        if not self.show_decoration:
            return
        self.out.line()
        self.out.print(Text(title, style="bold"))

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------
    def print_json(self, data: object) -> None:
        """Write exactly one compact JSON document to stdout, then a newline.

        The payload is made JSON-safe (``Path``, ``datetime``, ``Enum``, Pydantic
        models) and mappings are passed through
        :func:`yonga.util.redaction.redact_mapping` so a secret-bearing key can
        never reach stdout. Rich never sees the text, so no wrapping, colour or
        highlighting can corrupt the document.
        """
        payload = json_ready(data)
        if isinstance(payload, dict):
            payload = redact_mapping(payload)
        text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str)
        stream: IO[str] = self.out.file
        stream.write(text + "\n")
        with suppress(OSError, ValueError):
            stream.flush()

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def diagnostic(
        self, d: Diagnostic, *, debug: bool | None = None, debug_command: str | None = None
    ) -> None:
        """Render a failure the way a human wants to read it.

        A short title, a plain-English explanation, the remediation steps, the
        recommended fallback command and the job log path. The raw exception is
        printed only when ``debug`` is true; a traceback is never printed. In
        ``--json`` mode the same text is written to stderr without any
        decoration, so stdout still holds exactly one JSON document.
        ``debug_command`` is the command that would show the raw exception.
        """
        show_raw = self.debug if debug is None else debug
        if not self.show_decoration:
            for line in diagnostic_lines(d, debug=show_raw, debug_command=debug_command):
                self.err.file.write(line + "\n")
            with suppress(OSError, ValueError):
                self.err.file.flush()
            return
        self.err.line()
        self.err.print(
            render_diagnostic(d, debug=show_raw, debug_command=debug_command), highlight=False
        )

    # ------------------------------------------------------------------
    # confirmation
    # ------------------------------------------------------------------
    def confirm(
        self,
        question: str,
        *,
        default: bool = False,
        assume_yes: bool = False,
        security_sensitive: bool = False,
    ) -> bool:
        """Ask a yes/no question, enforcing the ``--yes`` security policy.

        The policy (``SPEC.md`` section 18.1) is deliberately asymmetric:

        * ``security_sensitive=True`` — ``assume_yes`` is ignored. A human is
          always asked, and a non-interactive session is refused (``False``).
          This is what stops ``--yes`` from silently enabling
          ``trust_remote_code``.
        * otherwise — ``assume_yes`` answers ``True`` without prompting, a
          non-interactive session without ``--yes`` answers ``False``, and an
          interactive session asks.
        """
        if security_sensitive:
            if not self.interactive:
                self.detail(f"Refused without an interactive prompt: {question}")
                return False
            return self._ask(question, default=default)
        if assume_yes:
            return True
        if not self.interactive:
            return False
        return self._ask(question, default=default)

    def _ask(self, question: str, *, default: bool) -> bool:
        """Put the question to the user on stderr; ``False`` on EOF or interrupt."""
        from rich.prompt import Confirm

        try:
            # A ``str`` prompt would be parsed as markup; the question can name a model.
            prompt = Text(question, style="prompt")
            return bool(Confirm.ask(prompt, console=self.err, default=default))
        except (EOFError, KeyboardInterrupt):
            self.err.line()
            return False


def _plain(renderable: RenderableType) -> RenderableType:
    """Return ``renderable`` with a ``str`` turned into literal :class:`Text`.

    The consoles from :func:`build_console` already ignore markup; this keeps
    the rule when a caller (a test, an embedding application) hands
    :class:`YongaConsole` a Rich console of its own.
    """
    return Text(renderable) if isinstance(renderable, str) else renderable


def _stdin_is_tty() -> bool:
    """Return whether stdin is a terminal, defensively."""
    try:
        return bool(sys.stdin is not None and sys.stdin.isatty())
    except (AttributeError, OSError, ValueError):
        return False
