"""Unit tests for the subprocess adapter: no shell, redaction, cancellation."""

from __future__ import annotations

import json
import os
import subprocess as stdlib_subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from yonga.adapters.subprocess import (
    DEFAULT_TAIL_LINES,
    CompletedCommand,
    SubprocessRunner,
    default_line_selector,
)
from yonga.application.jobs import JobContext
from yonga.application.progress import CollectingSink, ProgressKind
from yonga.config.settings import load_settings
from yonga.util.jsonlog import JobLogger
from yonga.util.redaction import REDACTED

SECRET = "hf_abcdefghijklmnopqrstuvwxyz0123456789"


def _make_context(tmp_path: Path) -> tuple[JobContext, CollectingSink, Path]:
    """Build a real job context writing to a throwaway log file."""
    settings = load_settings(
        config_file=tmp_path / "absent-config.toml",
        env={},
        overrides={"cache_dir": tmp_path / "cache", "data_dir": tmp_path / "data"},
    )
    log_path = tmp_path / "job.jsonl"
    sink = CollectingSink()
    job_temp = tmp_path / "job-temp"
    job_temp.mkdir(exist_ok=True)
    ctx = JobContext(
        job_id="job-1",
        logger=JobLogger("job-1", log_path),
        paths=settings.paths,
        settings=settings,
        progress=sink,
        temp_env={"TMPDIR": str(job_temp), "TMP": str(job_temp), "TEMP": str(job_temp)},
    )
    return ctx, sink, log_path


@pytest.fixture
def job(tmp_path):
    """A real job context whose log file is closed when the test finishes."""
    ctx, sink, log_path = _make_context(tmp_path)
    try:
        yield ctx, sink, log_path
    finally:
        ctx.logger.close()


def python(code: str) -> list[str]:
    """Build an argv list running ``code`` in this interpreter."""
    return [sys.executable, "-c", textwrap.dedent(code)]


def read_log(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --- the shell must never be involved ------------------------------------------------


def test_popen_is_called_with_an_argv_list_and_no_shell(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    class Spy(stdlib_subprocess.Popen[bytes]):
        def __init__(self, args: Any, **kwargs: Any) -> None:
            captured["args"] = args
            captured["kwargs"] = dict(kwargs)
            super().__init__(args, **kwargs)

    monkeypatch.setattr("yonga.adapters.subprocess.subprocess.Popen", Spy)

    argv = [sys.executable, "-c", "print('ok')"]
    result = SubprocessRunner().run(argv, cwd=tmp_path)

    assert result.returncode == 0
    assert isinstance(captured["args"], list)
    assert captured["args"] == argv
    kwargs = captured["kwargs"]
    assert kwargs.get("shell", False) is False
    assert kwargs["start_new_session"] is True
    assert kwargs["env"] is not None


def test_a_path_with_spaces_and_shell_metacharacters_is_passed_verbatim(tmp_path):
    hostile = tmp_path / "a dir; rm -rf $HOME" / "model & more"
    hostile.mkdir(parents=True)
    marker = hostile / "sentinel.txt"
    argv = python(
        """
        import pathlib, sys
        pathlib.Path(sys.argv[1]).write_text("written", encoding="utf-8")
        """
    )
    result = SubprocessRunner().run([*argv, str(marker)])
    assert result.returncode == 0
    assert marker.read_text(encoding="utf-8") == "written"


def test_empty_argv_is_rejected():
    with pytest.raises(ValueError, match="at least one argument"):
        SubprocessRunner().run([])


# --- environment ---------------------------------------------------------------------


def test_temp_env_overrides_reach_the_child(tmp_path):
    job_temp = tmp_path / "managed-temp"
    job_temp.mkdir()
    result = SubprocessRunner().run(
        python(
            """
            import os
            print(os.environ["TMPDIR"], os.environ["TMP"], os.environ["TEMP"])
            """
        ),
        env={"TMPDIR": str(job_temp), "TMP": str(job_temp), "TEMP": str(job_temp)},
    )
    assert result.returncode == 0
    assert result.stdout_lines == [f"{job_temp} {job_temp} {job_temp}"]


def test_context_temp_env_reaches_the_child_and_is_recorded(job):
    ctx, _sink, _log = job
    result = SubprocessRunner().run(
        python(
            """
            import os
            print(os.environ.get("TMPDIR", "<unset>"))
            """
        ),
        ctx=ctx,
    )
    assert result.returncode == 0
    assert result.stdout_lines == [ctx.temp_env["TMPDIR"]]
    assert dict(result.env_overrides) == dict(ctx.temp_env)


def test_only_overrides_are_recorded_never_the_whole_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_UNRELATED_HOST_VARIABLE", "leaky")
    result = SubprocessRunner().run(python("print('x')"), env={"YONGA_MARKER": "1"})
    assert dict(result.env_overrides) == {"YONGA_MARKER": "1"}
    assert "SOME_UNRELATED_HOST_VARIABLE" not in result.env_overrides


def test_secret_environment_overrides_are_redacted(tmp_path):
    result = SubprocessRunner().run(python("print('x')"), env={"HF_TOKEN": SECRET})
    assert result.env_overrides["HF_TOKEN"] == REDACTED
    assert SECRET not in json.dumps(dict(result.env_overrides))


# --- redaction -----------------------------------------------------------------------


def test_secrets_in_argv_are_redacted_on_the_result(job):
    ctx, _sink, log_path = job
    result = SubprocessRunner().run(
        [*python("print('x')"), "--token", SECRET, f"--api-key={SECRET}"],
        ctx=ctx,
    )
    joined = " ".join(result.argv)
    assert SECRET not in joined
    assert REDACTED in joined
    assert SECRET not in log_path.read_text(encoding="utf-8")


def test_secrets_in_output_lines_are_redacted(job):
    ctx, _sink, log_path = job
    result = SubprocessRunner().run(
        python(
            f"""
            import sys
            print("downloading with token {SECRET}")
            print("Authorization: Bearer {SECRET}", file=sys.stderr)
            """
        ),
        ctx=ctx,
    )
    assert result.returncode == 0
    assert SECRET not in " ".join(result.stdout_lines)
    assert SECRET not in " ".join(result.stderr_lines)
    assert REDACTED in " ".join(result.stdout_lines)
    assert SECRET not in log_path.read_text(encoding="utf-8")


def test_extra_secret_literals_are_masked(tmp_path):
    literal = "s3cret-passphrase-value"
    result = SubprocessRunner(extra_secrets=[literal]).run(
        python(f"print('the value is {literal}')")
    )
    assert literal not in " ".join(result.stdout_lines)
    assert REDACTED in " ".join(result.stdout_lines)


# --- exit codes and streams ----------------------------------------------------------


def test_non_zero_exit_is_reported_not_raised(job):
    ctx, _sink, _log = job
    result = SubprocessRunner().run(
        python(
            """
            import sys
            print("partial work")
            print("boom", file=sys.stderr)
            sys.exit(42)
            """
        ),
        ctx=ctx,
    )
    assert result.returncode == 42
    assert result.ok is False
    assert result.cancelled is False
    assert result.stdout_lines == ["partial work"]
    assert result.stderr_lines == ["boom"]


def test_a_missing_executable_is_returned_not_raised(tmp_path):
    result = SubprocessRunner().run([str(tmp_path / "definitely-not-a-program")])
    assert result.returncode == 127
    assert result.error is not None
    assert result.stderr_lines
    assert result.ok is False


def test_output_tail_is_bounded(tmp_path):
    runner = SubprocessRunner(tail_lines=10)
    result = runner.run(
        python(
            """
            for index in range(500):
                print(f"line {index}")
            """
        )
    )
    assert result.returncode == 0
    assert len(result.stdout_lines) == 10
    assert result.stdout_lines[-1] == "line 499"
    assert result.stdout_lines[0] == "line 490"


def test_the_full_stream_still_reaches_the_job_log(job):
    ctx, _sink, log_path = job
    SubprocessRunner(tail_lines=5).run(
        python(
            """
            for index in range(50):
                print(f"line {index}")
            """
        ),
        ctx=ctx,
    )
    ctx.logger.close()
    raw = [r for r in read_log(log_path) if r.get("event") == "subprocess.output"]
    assert len(raw) == 50


def test_duration_is_measured(tmp_path):
    result = SubprocessRunner().run(python("import time; time.sleep(0.2)"))
    assert result.duration_seconds >= 0.15
    assert result.duration_seconds < 30


def test_cwd_is_honoured(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    result = SubprocessRunner().run(python("import os; print(os.getcwd())"), cwd=workdir)
    assert Path(result.stdout_lines[0]).resolve() == workdir.resolve()
    assert result.cwd == workdir


# --- cancellation and timeout --------------------------------------------------------


def test_programmatic_cancellation_terminates_the_child(tmp_path, job):
    ctx, _sink, _log = job
    marker = tmp_path / "started.txt"
    argv = python(
        f"""
        import pathlib, time
        pathlib.Path({str(marker)!r}).write_text("up", encoding="utf-8")
        time.sleep(120)
        """
    )

    def cancel_soon() -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.02)
        ctx.cancelled.set()

    threading.Thread(target=cancel_soon, daemon=True).start()
    started = time.monotonic()
    result = SubprocessRunner(sigint_grace_seconds=1.0, sigterm_grace_seconds=1.0).run(
        argv, ctx=ctx
    )
    elapsed = time.monotonic() - started

    assert result.cancelled is True
    assert result.returncode != 0
    assert elapsed < 30
    assert result.pid is not None
    with pytest.raises(OSError):
        os.kill(result.pid, 0)


def test_cancellation_kills_a_child_that_ignores_sigint(tmp_path, job):
    ctx, _sink, _log = job
    marker = tmp_path / "ready.txt"
    argv = python(
        f"""
        import pathlib, signal, time
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        pathlib.Path({str(marker)!r}).write_text("up", encoding="utf-8")
        while True:
            time.sleep(0.1)
        """
    )

    def cancel_soon() -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.02)
        ctx.cancelled.set()

    threading.Thread(target=cancel_soon, daemon=True).start()
    result = SubprocessRunner(sigint_grace_seconds=0.3, sigterm_grace_seconds=0.3).run(
        argv, ctx=ctx
    )

    assert result.cancelled is True
    assert result.returncode != 0


def test_keyboard_interrupt_is_forwarded_and_reported(job, monkeypatch):
    ctx, _sink, _log = job
    real_wait = stdlib_subprocess.Popen.wait
    state = {"raised": False}

    def wait(self, timeout=None):
        if not state["raised"]:
            state["raised"] = True
            raise KeyboardInterrupt
        return real_wait(self, timeout=timeout)

    monkeypatch.setattr(stdlib_subprocess.Popen, "wait", wait)
    result = SubprocessRunner(sigint_grace_seconds=1.0, sigterm_grace_seconds=1.0).run(
        python("import time; time.sleep(60)"), ctx=ctx
    )
    assert result.cancelled is True


def test_timeout_terminates_and_is_flagged(tmp_path):
    result = SubprocessRunner(
        poll_interval=0.05, sigint_grace_seconds=0.5, sigterm_grace_seconds=0.5
    ).run(python("import time; time.sleep(60)"), timeout=0.5)
    assert result.timed_out is True
    assert result.cancelled is False
    assert result.ok is False
    assert result.duration_seconds < 20


# --- progress ------------------------------------------------------------------------


def test_selected_lines_become_progress_events(job):
    ctx, sink, _log = job
    SubprocessRunner().run(
        python(
            """
            import sys
            print("Exporting model to OpenVINO IR")
            print("Applying Weight Compression")
            print("a completely ordinary line of chatter")
            print("ERROR: something went wrong", file=sys.stderr)
            """
        ),
        ctx=ctx,
    )
    export = [event.message for event in sink.of_kind(ProgressKind.EXPORT)]
    logs = [event.message for event in sink.of_kind(ProgressKind.LOG)]
    assert "Exporting model to OpenVINO IR" in export
    assert "Applying Weight Compression" in export
    assert any("something went wrong" in message for message in logs)
    assert "a completely ordinary line of chatter" not in export + logs


def test_no_progress_is_emitted_without_a_context():
    result = SubprocessRunner().run(python("print('Exporting model')"))
    assert result.stdout_lines == ["Exporting model"]


def test_default_line_selector_classification():
    assert default_line_selector("") is None
    assert default_line_selector("Exporting model") is ProgressKind.EXPORT
    assert default_line_selector("INFO:nncf:Statistics") is ProgressKind.EXPORT
    assert default_line_selector("Traceback (most recent call last):") is ProgressKind.LOG
    assert default_line_selector("just some text") is None
    assert default_line_selector("Loading weights:  40%|####      | 8/21") is None


def test_progress_uses_the_last_segment_of_a_carriage_return_redraw(job):
    ctx, sink, _log = job
    SubprocessRunner().run(
        python(
            r"""
            import sys
            sys.stdout.write("stale\rApplying Weight Compression done\n")
            """
        ),
        ctx=ctx,
    )
    messages = [event.message for event in sink.of_kind(ProgressKind.EXPORT)]
    assert messages == ["Applying Weight Compression done"]


# --- result type ---------------------------------------------------------------------


def test_completed_command_satisfies_the_command_result_shape():
    result = CompletedCommand(argv=["a"], returncode=0, duration_seconds=1.0)
    for attribute in (
        "argv",
        "returncode",
        "duration_seconds",
        "stdout_lines",
        "stderr_lines",
        "cancelled",
    ):
        assert hasattr(result, attribute)
    assert result.ok is True
    assert result.output_tail == []


def test_default_tail_is_two_hundred():
    assert DEFAULT_TAIL_LINES == 200
