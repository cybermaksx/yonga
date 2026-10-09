"""The smoke test in a child process: what ``yonga probe --timeout`` runs.

A native NPU compile cannot be interrupted in-process, so a bounded pipeline
build runs in ``python -P -m yonga.adapters.smoke_child``. These tests pin the
contract between the two sides: the command is an argv list, the child's
verdict is one JSON line found even among native noise, a build stopped at the
time limit is inconclusive rather than a failure, and a child that dies without
a verdict is a failure carrying its redacted stderr for the registry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support import SmokeChildRunner
from yonga.adapters.smoke_child import (
    SMOKE_CHILD_MODULE,
    ChildProcessSmokeTester,
    main,
    result_from_stdout,
)
from yonga.adapters.subprocess import SubprocessRunner
from yonga.application.probe import IsolatedSmokeTester
from yonga.domain.artifacts import ProbeResult, ProbeStatus
from yonga.domain.diagnostics import CancelledError, CompileError, FailureCategory, diagnostic
from yonga.domain.models import PipelineKind

FAKE_TOKEN = "hf_smokechildsecretvalue0000000000000"


def ok_line(device: str = "NPU") -> str:
    """The line a child prints after the pipeline generated."""
    result = ProbeResult(device=device, status=ProbeStatus.OK, message="generated 24 token(s)")
    return json.dumps(result.model_dump(mode="json"))


def run(tester: ChildProcessSmokeTester, artifact: Path, *, timeout: float = 30.0) -> ProbeResult:
    """Run one smoke test with the probe's own prompt settings."""
    return tester.smoke_test(
        artifact,
        device="NPU",
        pipeline=PipelineKind.LLM,
        properties={"MAX_PROMPT_LEN": "1024"},
        prompt="Hello",
        max_new_tokens=24,
        timeout=timeout,
    )


def test_it_satisfies_the_port_the_probe_service_depends_on() -> None:
    assert isinstance(ChildProcessSmokeTester(SmokeChildRunner()), IsolatedSmokeTester)


def test_the_child_is_an_argv_list_with_the_cwd_kept_off_sys_path(tmp_path: Path) -> None:
    """No shell, and ``-P`` so a stray module in the working directory cannot load."""
    runner = SmokeChildRunner(stdout=[ok_line()])
    tester = ChildProcessSmokeTester(runner, python="/opt/python/bin/python3")

    run(tester, tmp_path / "my model; rm -rf ~", timeout=42)

    (command, timeout), *_ = runner.calls
    assert command[:4] == ["/opt/python/bin/python3", "-P", "-m", SMOKE_CHILD_MODULE]
    # The path travels as one argument, however it is spelled.
    assert command[4] == str(tmp_path / "my model; rm -rf ~")
    assert command[command.index("--device") + 1] == "NPU"
    assert command[command.index("--pipeline") + 1] == "llm"
    assert command[command.index("--max-new-tokens") + 1] == "24"
    assert command[command.index("--property") + 1] == "MAX_PROMPT_LEN=1024"
    assert timeout == 42


def test_the_childs_verdict_is_returned(tmp_path: Path) -> None:
    result = run(ChildProcessSmokeTester(SmokeChildRunner(stdout=[ok_line()])), tmp_path)

    assert result.status is ProbeStatus.OK
    assert result.message == "generated 24 token(s)"


def test_the_verdict_is_found_among_native_output_before_and_after_it(tmp_path: Path) -> None:
    """Native libraries write to stdout too, even after the verdict as they exit."""
    stdout = ["[ INFO ] compiling", '{"not": "a result"', ok_line(), "{broken", "bye", ""]
    runner = SmokeChildRunner(stdout=stdout, returncode=-11)

    result = run(ChildProcessSmokeTester(runner), tmp_path)

    # A crash during teardown, after the pipeline generated, does not undo it.
    assert result.status is ProbeStatus.OK


def test_a_stop_at_the_time_limit_is_inconclusive_never_a_failure(tmp_path: Path) -> None:
    runner = SmokeChildRunner(timed_out=True, stderr=["[ERROR] [vpux-compiler] half way"])

    result = run(ChildProcessSmokeTester(runner), tmp_path, timeout=5)

    assert result.status is ProbeStatus.INCONCLUSIVE
    assert result.diagnostic is None
    assert "5 s" in (result.message or "")


def test_a_cancelled_child_cancels_the_probe(tmp_path: Path) -> None:
    with pytest.raises(CancelledError) as raised:
        run(ChildProcessSmokeTester(SmokeChildRunner(cancelled=True)), tmp_path)

    assert raised.value.diagnostic.code == "probe.cancelled"


def test_a_child_that_dies_without_a_verdict_fails_with_its_redacted_stderr(
    tmp_path: Path,
) -> None:
    stderr = [
        f"token={FAKE_TOKEN}",
        "OneShotBufferizeVPU2VPUIP failed : Only supports FP16.",
        "Segmentation fault",
    ]
    runner = SmokeChildRunner(stdout=["[ INFO ] compiling"], stderr=stderr, returncode=-11)

    result = run(ChildProcessSmokeTester(runner), tmp_path)

    assert result.status is ProbeStatus.FAILED
    assert "exited with -11" in (result.message or "")
    assert "Only supports FP16" in (result.message or "")
    assert FAKE_TOKEN not in (result.message or "")


@pytest.mark.parametrize(
    "lines",
    [
        [],
        ["plain text"],
        ["[1, 2, 3]"],
        ['{"device": "NPU", "status": "no-such-status"}'],
    ],
)
def test_anything_but_a_result_object_is_no_verdict(lines: list[str]) -> None:
    assert result_from_stdout(lines) is None


def test_the_child_prints_one_json_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` is the child: it runs the in-process smoke test and prints its result."""
    calls: list[dict[str, object]] = []

    class RecordingTester:
        def smoke_test(self, artifact_dir: Path, **kwargs: object) -> ProbeResult:
            calls.append({"artifact": artifact_dir, **kwargs})
            return ProbeResult(device="NPU", status=ProbeStatus.OK, message="generated 3 token(s)")

    monkeypatch.setattr("yonga.application.wiring.GenAISmokeTester", RecordingTester)

    code = main(
        [
            str(tmp_path),
            "--device",
            "NPU",
            "--pipeline",
            "llm",
            "--max-new-tokens",
            "3",
            "--property",
            "MAX_PROMPT_LEN=1024",
        ]
    )

    assert code == 0
    assert calls == [
        {
            "artifact": tmp_path,
            "device": "NPU",
            "pipeline": PipelineKind.LLM,
            "properties": {"MAX_PROMPT_LEN": "1024"},
            "prompt": "Hello",
            "max_new_tokens": 3,
        }
    ]
    verdict = result_from_stdout(capsys.readouterr().out.splitlines())
    assert verdict is not None and verdict.status is ProbeStatus.OK


def test_the_child_reports_a_typed_failure_as_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FailingTester:
        def smoke_test(self, artifact_dir: Path, **kwargs: object) -> ProbeResult:
            raise CompileError(
                diagnostic(
                    "runtime.genai.load_failed",
                    FailureCategory.UNKNOWN_RUNTIME_ERROR,
                    "Could not load the model on NPU",
                    "Compilation failed",
                )
            )

    monkeypatch.setattr("yonga.application.wiring.GenAISmokeTester", FailingTester)

    code = main([str(tmp_path), "--device", "NPU"])

    assert code == 1
    verdict = result_from_stdout(capsys.readouterr().out.splitlines())
    assert verdict is not None and verdict.status is ProbeStatus.FAILED
    assert verdict.diagnostic is not None
    assert verdict.diagnostic.code == "runtime.genai.load_failed"


def test_a_malformed_property_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        main([str(tmp_path), "--device", "NPU", "--property", "no-separator"])

    assert raised.value.code == 2


def test_a_real_child_process_returns_a_verdict(tmp_path: Path) -> None:
    """End to end through the real runner: the child starts, fails, and says why.

    The artifact does not exist, so the pipeline cannot be built whether or not
    OpenVINO is installed -- what matters is that the verdict came back as JSON
    from the child rather than as a crash without one.
    """
    tester = ChildProcessSmokeTester(SubprocessRunner())

    result = run(tester, tmp_path / "missing", timeout=120)

    assert result.status is ProbeStatus.FAILED
    assert "without a result" not in (result.message or "")
