"""Tests for the chat presentation layer: panel, command parser, metrics footer."""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from yonga.config.paths import Paths
from yonga.ui.chat_tui import (
    CHAT_COMMANDS,
    ChatCommand,
    ChatCommandKind,
    ChatSessionInfo,
    ChatView,
    help_text,
    parse_command,
    render_answer,
    render_metrics,
    session_facts,
    startup_panel,
)
from yonga.ui.console import YongaConsole, build_console

SHA = "0123456789abcdef0123456789abcdef01234567"


def render(renderable: Any, *, width: int = 100) -> str:
    """Render a renderable to plain text the way a non-TTY terminal would."""
    from rich.console import Console

    buffer = io.StringIO()
    Console(file=buffer, width=width, force_terminal=False, no_color=True).print(renderable)
    return buffer.getvalue()


def session_info(**overrides: Any) -> ChatSessionInfo:
    """A fully populated chat session description."""
    payload: dict[str, Any] = {
        "display_name": "Qwen3.5-9B",
        "artifact_id": "qwen35-gpu",
        "repo_id": "Qwen/Qwen3.5-9B",
        "revision": SHA,
        "device": "GPU",
        "device_full_name": "Intel Arc Graphics",
        "profile": "gpu-int4-asym-g128",
        "max_new_tokens": 4096,
        "show_reasoning": False,
        "compile_cache_enabled": True,
        "compile_cache_path": Path("/home/user/.cache/yonga/compile/qwen35-gpu"),
    }
    payload.update(overrides)
    return ChatSessionInfo(**payload)


def metrics(**overrides: Any) -> SimpleNamespace:
    """A generation result stand-in carrying whichever metrics the test needs."""
    payload: dict[str, Any] = {
        "input_tokens": 512,
        "output_tokens": 380,
        "ttft_ms": 180.0,
        "throughput_tps": 24.6,
        "tpot_ms": 40.6,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def console_pair(**kwargs: Any) -> tuple[YongaConsole, io.StringIO, io.StringIO]:
    """A console writing into in-memory buffers."""
    out = io.StringIO()
    err = io.StringIO()
    return (
        YongaConsole(
            out=build_console(file=out, width=100, force_terminal=False, no_color=True),
            err=build_console(file=err, width=100, force_terminal=False, no_color=True),
            env={},
            **kwargs,
        ),
        out,
        err,
    )


def parsed(line: str) -> ChatCommand:
    """Parse a line that must be a command, narrowing away the ``None``."""
    command = parse_command(line)
    assert command is not None
    return command


# ----------------------------------------------------------------------
# command parsing
# ----------------------------------------------------------------------
def test_plain_text_is_not_a_command():
    assert parse_command("hello there") is None
    assert parse_command("") is None
    assert parse_command("   ") is None
    assert parse_command("/") is None
    assert parse_command("what do you think about /exit?") is None


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ("/help", ChatCommandKind.HELP),
        ("/reset", ChatCommandKind.RESET),
        ("/clear", ChatCommandKind.CLEAR),
        ("/model", ChatCommandKind.MODEL),
        ("/exit", ChatCommandKind.EXIT),
        ("/quit", ChatCommandKind.EXIT),
        ("  /HELP  ", ChatCommandKind.HELP),
    ],
)
def test_bare_commands_parse(line, kind):
    command = parsed(line)
    assert command is not None
    assert command.kind is kind
    assert command.ok


def test_bare_command_rejects_an_argument():
    command = parsed("/reset everything")
    assert command.kind is ChatCommandKind.RESET
    assert not command.ok
    assert command.error is not None and "does not take an argument" in command.error


def test_unknown_command_is_reported_with_help_hint():
    command = parsed("/nope 3")
    assert command.kind is ChatCommandKind.UNKNOWN
    assert not command.ok
    assert command.error is not None and "Unknown command '/nope'" in command.error
    assert command.error is not None and "/help" in command.error


@pytest.mark.parametrize(
    ("line", "expected"),
    [("/tokens 512", 512), ("/tokens 1", 1), ("/tokens   2048  ", 2048)],
)
def test_tokens_accepts_whole_numbers(line, expected):
    command = parsed(line)
    assert command.kind is ChatCommandKind.TOKENS
    assert command.ok
    assert command.int_value == expected


@pytest.mark.parametrize("line", ["/tokens abc", "/tokens 1.5", "/tokens 0", "/tokens -4"])
def test_tokens_rejects_bad_arguments(line):
    command = parsed(line)
    assert command.kind is ChatCommandKind.TOKENS
    assert not command.ok
    assert command.error
    assert command.int_value is None


def test_tokens_without_argument_shows_usage():
    command = parsed("/tokens")
    assert not command.ok
    assert command.error == "Usage: /tokens N"


def test_tokens_rejects_an_absurd_value():
    command = parsed("/tokens 999999999")
    assert not command.ok
    assert command.error is not None and "between 1 and" in command.error


@pytest.mark.parametrize(
    ("line", "expected"), [("/temp 0.7", 0.7), ("/temp 0", 0.0), ("/temperature 2", 2.0)]
)
def test_temperature_accepts_valid_values(line, expected):
    command = parsed(line)
    assert command.kind is ChatCommandKind.TEMPERATURE
    assert command.ok
    assert command.float_value == pytest.approx(expected)


@pytest.mark.parametrize("line", ["/temp 5", "/temp -1", "/temp hot", "/temp nan", "/temp inf"])
def test_temperature_rejects_out_of_range_and_junk(line):
    command = parsed(line)
    assert command.kind is ChatCommandKind.TEMPERATURE
    assert not command.ok
    assert command.error
    assert command.float_value is None


def test_temperature_error_names_the_bounds():
    error = parsed("/temp 5").error
    assert error is not None
    assert "between 0 and 2" in error


@pytest.mark.parametrize(("line", "expected"), [("/top-p 0.8", 0.8), ("/top_p 1", 1.0)])
def test_top_p_accepts_valid_values(line, expected):
    command = parsed(line)
    assert command.kind is ChatCommandKind.TOP_P
    assert command.ok
    assert command.float_value == pytest.approx(expected)


@pytest.mark.parametrize("line", ["/top-p 1.5", "/top-p -0.1", "/top-p most"])
def test_top_p_rejects_bad_values(line):
    assert not parsed(line).ok


@pytest.mark.parametrize(("line", "expected"), [("/top-k 20", 20), ("/top-k 0", 0)])
def test_top_k_accepts_valid_values(line, expected):
    command = parsed(line)
    assert command.kind is ChatCommandKind.TOP_K
    assert command.ok
    assert command.int_value == expected


@pytest.mark.parametrize("line", ["/top-k -1", "/top-k many", "/top-k 1.5"])
def test_top_k_rejects_bad_values(line):
    assert not parsed(line).ok


@pytest.mark.parametrize(
    ("line", "kind", "expected"),
    [
        ("/reasoning on", ChatCommandKind.REASONING, True),
        ("/reasoning OFF", ChatCommandKind.REASONING, False),
        ("/stats on", ChatCommandKind.STATS, True),
        ("/stats off", ChatCommandKind.STATS, False),
    ],
)
def test_switch_commands_parse(line, kind, expected):
    command = parsed(line)
    assert command.kind is kind
    assert command.ok
    assert command.bool_value is expected


@pytest.mark.parametrize("line", ["/reasoning maybe", "/stats 7", "/reasoning"])
def test_switch_commands_reject_other_words(line):
    command = parsed(line)
    assert not command.ok
    assert command.bool_value is None
    assert command.error


def test_every_documented_command_is_parseable():
    for spec in CHAT_COMMANDS:
        name = spec.usage.split(" ")[0]
        command = parsed(name)
        assert command is not None
        assert command.kind is spec.kind


def test_parsed_commands_are_immutable():
    command = parsed("/tokens 64")
    with pytest.raises(ValueError):
        command.int_value = 1


# ----------------------------------------------------------------------
# metrics footer
# ----------------------------------------------------------------------
def test_metrics_footer_with_every_metric():
    assert (
        render_metrics(metrics()).plain
        == "TTFT 180 ms • 24.6 tok/s • TPOT 40.6 ms/token • 512 → 380 tokens"
    )


def test_metrics_footer_omits_missing_metrics():
    footer = render_metrics(metrics(ttft_ms=None, tpot_ms=None)).plain
    assert footer == "24.6 tok/s • 512 → 380 tokens"
    assert "TTFT" not in footer
    assert "TPOT" not in footer


def test_metrics_footer_with_only_output_tokens():
    footer = render_metrics(
        metrics(ttft_ms=None, throughput_tps=None, tpot_ms=None, input_tokens=None)
    ).plain
    assert footer == "380 output tokens"


def test_metrics_footer_with_only_input_tokens():
    footer = render_metrics(
        metrics(ttft_ms=None, throughput_tps=None, tpot_ms=None, output_tokens=None)
    ).plain
    assert footer == "512 input tokens"


def test_metrics_footer_with_all_metrics_missing_is_empty():
    empty = metrics(
        ttft_ms=None, throughput_tps=None, tpot_ms=None, input_tokens=None, output_tokens=None
    )
    assert render_metrics(empty).plain == ""
    assert render_metrics(None).plain == ""


def test_metrics_footer_ignores_a_result_without_metric_attributes():
    assert render_metrics(SimpleNamespace()).plain == ""


def test_metrics_footer_ignores_non_numeric_metrics():
    assert render_metrics(metrics(ttft_ms="fast", throughput_tps=float("nan"))).plain == (
        "TPOT 40.6 ms/token • 512 → 380 tokens"
    )


def test_metrics_footer_is_dim():
    assert "dim" in str(render_metrics(metrics()).style)


def test_show_metrics_prints_nothing_when_there_is_nothing_to_show():
    console, out, _ = console_pair()
    view = ChatView(console, session_info())
    view.show_metrics(None)
    assert out.getvalue() == ""
    view.show_metrics(metrics())
    assert "TTFT 180 ms" in out.getvalue()


# ----------------------------------------------------------------------
# panels
# ----------------------------------------------------------------------
def test_startup_panel_shows_every_required_fact():
    text = render(startup_panel(session_info()))
    assert "Local Chat" in text
    assert "Qwen3.5-9B" in text
    assert "Qwen/Qwen3.5-9B@0123456789ab" in text
    assert SHA not in text
    assert "Intel Arc Graphics (GPU)" in text
    assert "gpu-int4-asym-g128" in text
    assert "4096 tokens" in text
    assert "hidden" in text
    assert "enabled" in text


def test_startup_panel_reports_visible_reasoning():
    assert "shown" in render(startup_panel(session_info(show_reasoning=True)))


def test_startup_panel_without_optional_facts():
    text = render(startup_panel(ChatSessionInfo(display_name="local model")))
    assert "local model" in text
    assert "AUTO" in text


def test_render_answer_is_a_markdown_panel():
    text = render(render_answer("# Title\n\nSome **bold** answer."))
    assert "Title" in text
    assert "Some bold answer." in text


def test_render_answer_handles_empty_text():
    assert "no text" in render(render_answer("   "))


def test_help_text_lists_every_command():
    text = render(help_text())
    for usage in (
        "/help",
        "/reset",
        "/clear",
        "/tokens N",
        "/temp X",
        "/top-p X",
        "/top-k N",
        "/reasoning on|off",
        "/stats on|off",
        "/model",
        "/exit",
    ):
        assert usage in text


def test_session_facts_are_flat_strings():
    facts = session_facts(session_info())
    assert facts["display_name"] == "Qwen3.5-9B"
    assert facts["reasoning"] == "hidden"
    assert facts["max_new_tokens"] == "4096"
    assert all(isinstance(value, str) for value in facts.values())


# ----------------------------------------------------------------------
# view plumbing
# ----------------------------------------------------------------------
def test_history_path_uses_the_chat_history_directory(tmp_path):
    paths = Paths(config_dir=tmp_path / "c", cache_dir=tmp_path / "k", data_dir=tmp_path / "d")
    console, _, _ = console_pair()
    view = ChatView(console, session_info(), paths=paths)
    assert view.history_path == paths.chat_history_dir / "qwen35-gpu.history"


def test_history_path_is_none_without_paths():
    console, _, _ = console_pair()
    assert ChatView(console, session_info()).history_path is None


def test_history_filename_is_filesystem_safe():
    info = ChatSessionInfo(display_name="x", artifact_id="", repo_id="Qwen/Qwen3.5-9B")
    assert info.history_filename == "Qwen-Qwen3.5-9B.history"


def test_generating_is_silent_without_a_terminal():
    console, out, err = console_pair()
    view = ChatView(console, session_info())
    with view.generating():
        pass
    assert out.getvalue() == ""
    assert err.getvalue() == ""


def test_json_mode_suppresses_chat_decoration():
    console, out, err = console_pair(json_mode=True)
    view = ChatView(console, session_info())
    view.show_startup()
    view.show_answer("hello")
    view.show_metrics(metrics())
    view.show_help()
    assert out.getvalue() == ""
    assert err.getvalue() == ""


def test_show_error_goes_to_stderr():
    console, out, err = console_pair()
    ChatView(console, session_info()).show_error("Usage: /tokens N")
    assert out.getvalue() == ""
    assert "Usage: /tokens N" in err.getvalue()


def test_prompt_reads_stdin_when_not_interactive(monkeypatch):
    console = YongaConsole(
        out=build_console(file=io.StringIO(), width=100, no_color=True),
        err=build_console(file=io.StringIO(), width=100, no_color=True),
        env={},
        interactive=False,
    )
    monkeypatch.setattr("builtins.input", lambda: "typed line")
    assert ChatView(console, session_info()).prompt() == "typed line"


# ----------------------------------------------------------------------
# import discipline
# ----------------------------------------------------------------------
def test_ui_imports_no_heavy_or_terminal_modules():
    script = (
        "import sys; import yonga.ui; import yonga.cli;"
        "banned = ['prompt_toolkit', 'openvino', 'openvino_genai', 'openvino_tokenizers',"
        " 'optimum', 'nncf', 'transformers', 'torch', 'huggingface_hub'];"
        "print(','.join(name for name in banned if name in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""
