"""Tests for the console output policy: colour, JSON mode, errors, confirmations."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from yonga.application.progress import ProgressEvent, ProgressKind, ProgressSink
from yonga.domain.diagnostics import Diagnostic, FailureCategory, Severity, diagnostic
from yonga.ui.console import (
    YongaConsole,
    build_console,
    diagnostic_lines,
    human_bytes,
    human_duration,
    json_ready,
    status_icon,
)
from yonga.ui.progress import RichProgressSink

RAW_TRACEBACK = (
    "RuntimeError: Exception from src/plugins/intel_npu/... ZE_RESULT_ERROR_INVALID_ARGUMENT"
)


def make_console(**kwargs: Any) -> tuple[YongaConsole, io.StringIO, io.StringIO]:
    """Build a console pair writing into in-memory buffers."""
    out = io.StringIO()
    err = io.StringIO()
    console = YongaConsole(
        out=build_console(file=out, width=100, force_terminal=False, no_color=True),
        err=build_console(file=err, width=100, force_terminal=False, no_color=True),
        env=kwargs.pop("env", {}),
        **kwargs,
    )
    return console, out, err


def npu_diagnostic() -> Diagnostic:
    """A realistic NPU compile failure carrying a raw excerpt and a fallback."""
    return diagnostic(
        "runtime.compile_failed",
        FailureCategory.COMPILER_UNSUPPORTED_GRAPH,
        "NPU compilation failed for this Qwen3.5 model",
        "The NPU is installed and visible, but the current compiler cannot compile "
        "this model graph.",
        severity=Severity.ERROR,
        remediation=["Prepare the model for the GPU instead."],
        fallback_devices=["GPU"],
        raw_excerpt=RAW_TRACEBACK,
        log_path=Path("/home/user/.local/share/yonga/logs/job.jsonl"),
        context={"model": "Qwen/Qwen3.5-9B", "command": "prepare"},
    )


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0 B"),
        (1, "1 B"),
        (512, "512 B"),
        (1023, "1023 B"),
        (1024, "1.0 KiB"),
        (1536, "1.5 KiB"),
        (10 * 1024, "10 KiB"),
        (1024**2 - 1, "1.0 MiB"),
        (1024**2, "1.0 MiB"),
        (1024**3, "1.0 GiB"),
        (16 * 1024**3, "16 GiB"),
        (796 * 1024**3, "796 GiB"),
        (1024**5, "1.0 PiB"),
        (-5, "0 B"),
        (None, "unknown"),
    ],
)
def test_human_bytes_boundaries(value, expected):
    assert human_bytes(value) == expected


def test_human_bytes_explicit_precision():
    assert human_bytes(int(18.8 * 1024**3), precision=1) == "18.8 GiB"
    assert human_bytes(1024**3, precision=2) == "1.00 GiB"


def test_human_bytes_rejects_non_numbers():
    assert human_bytes(float("nan")) == "unknown"
    assert human_bytes(float("inf")) == "unknown"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (None, "unknown"),
        (0.0, "0 ms"),
        (0.0000001, "<1 ms"),
        (0.18, "180 ms"),
        (1.0, "1.0 s"),
        (10.84, "10.8 s"),
        (61.0, "1m 01s"),
        (3600.0, "1h 00m"),
        (3900.0, "1h 05m"),
        (-3.0, "0 ms"),
    ],
)
def test_human_duration(seconds, expected):
    assert human_duration(seconds) == expected


def test_status_icon_covers_every_severity():
    icons = {severity: status_icon(severity) for severity in Severity}
    assert icons[Severity.OK] == "✓"
    assert icons[Severity.WARNING] == "!"
    assert icons[Severity.ERROR] == "✗"
    assert icons[Severity.UNKNOWN] == "?"
    assert len(icons) == len(Severity)


def test_json_ready_handles_paths_datetimes_and_models():
    payload = json_ready(
        {
            "path": Path("/tmp/x"),
            "when": datetime(2026, 9, 22, 10, 0, tzinfo=UTC),
            "severity": Severity.WARNING,
            "diag": npu_diagnostic(),
            "items": {3, 1, 2},
        }
    )
    assert isinstance(payload, dict)
    assert payload["path"] == "/tmp/x"
    assert payload["when"].startswith("2026-09-22T10:00")
    assert payload["severity"] == "warning"
    assert payload["diag"]["code"] == "runtime.compile_failed"
    assert sorted(payload["items"]) == [1, 2, 3]
    json.dumps(payload)


# ----------------------------------------------------------------------
# colour and decoration policy
# ----------------------------------------------------------------------
def test_no_color_environment_disables_colour():
    console = YongaConsole(env={"NO_COLOR": "1"})
    assert console.no_color is True
    assert console.color_enabled is False


def test_force_color_overrides_no_color():
    console = YongaConsole(env={"NO_COLOR": "1", "FORCE_COLOR": "1"})
    assert console.no_color is False


def test_dumb_terminal_disables_colour():
    assert YongaConsole(env={"TERM": "dumb"}).no_color is True


def test_no_color_output_contains_no_ansi_escapes():
    out = io.StringIO()
    console = YongaConsole(
        env={"NO_COLOR": "1"},
        out=build_console(file=out, width=100, force_terminal=True, no_color=True),
        err=build_console(file=io.StringIO(), width=100, force_terminal=True, no_color=True),
    )
    console.section("Devices")
    console.print("CPU ok")
    console.rule("Storage")
    text = out.getvalue()
    assert "\x1b[" not in text
    assert console.color_enabled is False
    assert "Devices" in text
    assert "Storage" in text


def test_non_terminal_disables_animation():
    console, _, _ = make_console()
    assert console.animate is False
    assert console.interactive is False


def test_json_mode_suppresses_all_decoration():
    console, out, err = make_console(json_mode=True)
    console.section("Devices")
    console.rule("Storage")
    console.print("decorative")
    console.print_error("also decorative")
    console.blank()
    assert out.getvalue() == ""
    assert err.getvalue() == ""
    assert console.show_decoration is False
    assert console.animate is False


def test_json_mode_emits_exactly_one_document():
    console, out, err = make_console(json_mode=True)
    console.section("ignored")
    console.print_json({"artifact": "abc", "path": Path("/data/a"), "size": 10})
    payload = out.getvalue()
    assert payload.count("\n") == 1
    document = json.loads(payload)
    assert document == {"artifact": "abc", "path": "/data/a", "size": 10}
    assert err.getvalue() == ""


def test_print_json_is_compact_and_unwrapped():
    console, out, _ = make_console(json_mode=True)
    console.print_json({"reasons": ["x" * 400], "n": 1})
    payload = out.getvalue()
    assert len(payload.splitlines()) == 1
    assert ", " not in payload
    assert json.loads(payload)["n"] == 1


def test_print_json_redacts_secret_bearing_keys():
    console, out, _ = make_console(json_mode=True)
    console.print_json({"hf_token": "hf_abcdefghijklmnopqrstuvwxyz", "model": "a/b"})
    document = json.loads(out.getvalue())
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in out.getvalue()
    assert document["model"] == "a/b"


# ----------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------
def test_diagnostic_hides_raw_exception_without_debug():
    console, out, err = make_console()
    console.diagnostic(npu_diagnostic(), debug=False)
    text = err.getvalue()
    assert "NPU compilation failed" in text
    assert "Recommended fallback: GPU" in text
    assert "yonga prepare Qwen/Qwen3.5-9B --device gpu" in text
    assert "Job log: /home/user/.local/share/yonga/logs/job.jsonl" in text
    assert "Prepare the model for the GPU instead." in text
    assert "ZE_RESULT_ERROR_INVALID_ARGUMENT" not in text
    assert "Traceback" not in text
    assert "--debug" in text
    assert out.getvalue() == ""


def test_diagnostic_shows_raw_exception_with_debug():
    console, _, err = make_console()
    console.diagnostic(npu_diagnostic(), debug=True)
    assert "ZE_RESULT_ERROR_INVALID_ARGUMENT" in err.getvalue()


def test_diagnostic_debug_defaults_to_console_flag():
    console, _, err = make_console(debug=True)
    console.diagnostic(npu_diagnostic())
    assert "ZE_RESULT_ERROR_INVALID_ARGUMENT" in err.getvalue()


def test_diagnostic_in_json_mode_keeps_stdout_clean():
    console, out, err = make_console(json_mode=True)
    console.diagnostic(npu_diagnostic())
    assert out.getvalue() == ""
    text = err.getvalue()
    assert "NPU compilation failed" in text
    # No panels, rules or icons: json mode output is undecorated.
    assert "─" not in text
    assert "✗" not in text


def test_diagnostic_lines_are_pure_and_ordered():
    lines = diagnostic_lines(npu_diagnostic(), debug=False)
    assert lines[0] == "NPU compilation failed for this Qwen3.5 model"
    joined = "\n".join(lines)
    assert joined.index("Recommended fallback") < joined.index("Job log:")
    assert RAW_TRACEBACK not in joined


def test_diagnostic_without_optional_fields_still_renders():
    minimal = diagnostic(
        "config.invalid",
        FailureCategory.CONFIGURATION_ERROR,
        "Configuration error",
        "The configuration file could not be parsed.",
    )
    console, _, err = make_console()
    console.diagnostic(minimal)
    text = err.getvalue()
    assert "Configuration error" in text
    assert "Job log" not in text


def test_diagnostic_prefers_explicit_fallback_command():
    item = diagnostic(
        "runtime.compile_failed",
        FailureCategory.COMPILER_UNSUPPORTED_GRAPH,
        "Compile failed",
        "No good.",
        fallback_devices=["GPU"],
        context={"fallback_command": "yonga prepare foo/bar --device gpu --profile x"},
    )
    assert "Run: yonga prepare foo/bar --device gpu --profile x" in diagnostic_lines(item)


# ----------------------------------------------------------------------
# confirmation policy
# ----------------------------------------------------------------------
class RecordingConsole(YongaConsole):
    """Console that records prompts instead of blocking on a terminal."""

    def __init__(self, answer: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.asked: list[str] = []
        self.answer = answer

    def _ask(self, question: str, *, default: bool) -> bool:
        self.asked.append(question)
        return self.answer


def recording(answer: bool = True, *, interactive: bool | None, **kwargs: Any) -> RecordingConsole:
    """Build a RecordingConsole writing into in-memory buffers."""
    return RecordingConsole(
        answer,
        out=build_console(file=io.StringIO(), width=100, force_terminal=False, no_color=True),
        err=build_console(file=io.StringIO(), width=100, force_terminal=False, no_color=True),
        env={},
        interactive=interactive,
        **kwargs,
    )


def test_security_sensitive_confirm_is_never_auto_approved_non_interactive():
    console = recording(answer=True, interactive=False)
    result = console.confirm(
        "Enable trust_remote_code?", default=False, assume_yes=True, security_sensitive=True
    )
    assert result is False
    assert console.asked == []


def test_security_sensitive_confirm_still_prompts_with_assume_yes():
    console = recording(answer=True, interactive=True)
    result = console.confirm(
        "Enable trust_remote_code?", default=False, assume_yes=True, security_sensitive=True
    )
    assert console.asked == ["Enable trust_remote_code?"]
    assert result is True


def test_security_sensitive_confirm_respects_a_no():
    console = recording(answer=False, interactive=True)
    assert (
        console.confirm("Enable trust_remote_code?", assume_yes=True, security_sensitive=True)
        is False
    )


def test_ordinary_confirm_is_auto_approved_by_assume_yes():
    console = recording(answer=False, interactive=False)
    assert console.confirm("Download 18 GiB?", assume_yes=True) is True
    assert console.asked == []


def test_non_interactive_without_assume_yes_declines():
    console = recording(answer=True, interactive=False)
    assert console.confirm("Download 18 GiB?", default=True) is False
    assert console.asked == []


def test_interactive_confirm_asks():
    console = recording(answer=True, interactive=True)
    assert console.confirm("Download 18 GiB?") is True
    assert console.asked == ["Download 18 GiB?"]


def test_json_mode_is_never_interactive():
    console, _, _ = make_console(json_mode=True)
    assert console.interactive is False


# ----------------------------------------------------------------------
# progress sink
# ----------------------------------------------------------------------
def test_rich_progress_sink_satisfies_the_protocol():
    console, _, _ = make_console()
    assert isinstance(RichProgressSink(console), ProgressSink)


def test_progress_sink_is_silent_in_json_mode():
    console, out, err = make_console(json_mode=True)
    with RichProgressSink(console) as sink:
        sink.emit(ProgressEvent(kind=ProgressKind.STAGE, message="Preflight", current=3, total=7))
        sink.emit(ProgressEvent(kind=ProgressKind.DOWNLOAD, current=1, total=2))
        sink.emit(ProgressEvent(kind=ProgressKind.PROBE, device="NPU", status="failed"))
    assert out.getvalue() == ""
    assert err.getvalue() == ""


def test_progress_sink_prints_stage_counter_without_a_terminal():
    console, _, err = make_console()
    with RichProgressSink(console) as sink:
        sink.emit(ProgressEvent(kind=ProgressKind.STAGE, message="Preflight", current=3, total=7))
    assert "3/7 Preflight" in err.getvalue()


def test_progress_sink_tags_probe_lines_with_the_device():
    console, _, err = make_console()
    with RichProgressSink(console) as sink:
        sink.emit(
            ProgressEvent(
                kind=ProgressKind.PROBE,
                device="NPU",
                status="failed",
                message="compile probe failed",
            )
        )
        sink.emit(
            ProgressEvent(kind=ProgressKind.PROBE, device="GPU", status="ok", message="compile ok")
        )
    text = err.getvalue()
    assert "NPU" in text
    assert "compile probe failed" in text
    assert "GPU" in text
    assert "✓" in text
    assert "✗" in text


def test_progress_sink_log_events_need_verbose():
    console, _, err = make_console()
    with RichProgressSink(console) as sink:
        sink.emit(ProgressEvent(kind=ProgressKind.LOG, message="chatty detail"))
    assert "chatty detail" not in err.getvalue()

    verbose, _, verbose_err = make_console(verbose=True)
    with RichProgressSink(verbose) as sink:
        sink.emit(ProgressEvent(kind=ProgressKind.LOG, message="chatty detail"))
    assert "chatty detail" in verbose_err.getvalue()


def test_progress_sink_never_raises():
    class Exploding(YongaConsole):
        @property
        def json_mode(self) -> bool:
            raise RuntimeError("boom")

        @json_mode.setter
        def json_mode(self, value: bool) -> None:
            pass

    console = Exploding(
        out=build_console(file=io.StringIO(), width=100, force_terminal=False),
        err=build_console(file=io.StringIO(), width=100, force_terminal=False),
        env={},
    )
    sink = RichProgressSink(console)
    sink.emit(ProgressEvent(kind=ProgressKind.STAGE, message="boom"))
    sink.close()


def test_progress_sink_close_is_idempotent():
    console, _, _ = make_console()
    sink = RichProgressSink(console)
    sink.close()
    sink.close()
