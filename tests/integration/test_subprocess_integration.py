"""Integration test: run a real subprocess through the adapter end to end.

Nothing is mocked here. A genuine Python child process is started, writes to
both streams, sleeps briefly and exits non-zero; the adapter must capture both
streams, report the exit code, measure a plausible duration, redirect the
managed temp variables and write every line into the real job log.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from yonga.adapters.filesystem import plan_temp
from yonga.adapters.subprocess import SubprocessRunner
from yonga.application.jobs import JobContext
from yonga.application.progress import CollectingSink, ProgressKind
from yonga.config.settings import load_settings
from yonga.util.jsonlog import JobLogger
from yonga.util.redaction import REDACTED

CHILD = textwrap.dedent(
    """
    import os
    import sys
    import time

    print("Exporting model to OpenVINO IR")
    print(f"tmpdir={os.environ.get('TMPDIR', '<unset>')}")
    print("token is hf_abcdefghijklmnopqrstuvwxyz0123456789")
    sys.stdout.flush()
    print("WARNING: this is only a drill", file=sys.stderr)
    print(f"argv1={sys.argv[1]}", file=sys.stderr)
    sys.stderr.flush()
    time.sleep(0.25)
    sys.exit(3)
    """
)


@pytest.fixture
def context(tmp_path: Path):
    """A job context backed by a real settings object and a real log file."""
    settings = load_settings(
        config_file=tmp_path / "absent-config.toml",
        env={},
        overrides={"cache_dir": tmp_path / "cache", "data_dir": tmp_path / "data"},
    )
    settings.paths.ensure()
    job_temp = settings.paths.job_tmp_dir("integration-job")
    job_temp.mkdir(parents=True, exist_ok=True)
    log_path = settings.paths.job_log_path("integration-job")
    sink = CollectingSink()
    ctx = JobContext(
        job_id="integration-job",
        logger=JobLogger("integration-job", log_path),
        paths=settings.paths,
        settings=settings,
        progress=sink,
        temp_env={"TMPDIR": str(job_temp), "TMP": str(job_temp), "TEMP": str(job_temp)},
    )
    try:
        yield ctx, sink, log_path, job_temp
    finally:
        ctx.logger.close()


def test_real_subprocess_round_trip(context):
    ctx, sink, log_path, job_temp = context
    argv = [sys.executable, "-c", CHILD, "a value with spaces"]

    result = SubprocessRunner(tail_lines=50).run(argv, cwd=job_temp, ctx=ctx)

    # exit code and duration
    assert result.returncode == 3
    assert result.ok is False
    assert result.cancelled is False
    assert result.timed_out is False
    assert result.duration_seconds >= 0.2
    assert result.duration_seconds < 60
    assert result.pid is not None
    assert result.cwd == job_temp

    # both streams captured, separately
    stdout = result.stdout_lines
    stderr = result.stderr_lines
    assert "Exporting model to OpenVINO IR" in stdout
    assert f"tmpdir={job_temp}" in stdout
    assert "WARNING: this is only a drill" in stderr
    assert "argv1=a value with spaces" in stderr
    assert not any("WARNING" in line for line in stdout)

    # the managed temp directory was handed to the child
    assert dict(result.env_overrides) == dict(ctx.temp_env)

    # secrets are gone from the captured output and from the log file
    joined = " ".join([*stdout, *stderr])
    assert "hf_abcdefghijklmnopqrstuvwxyz0123456789" not in joined
    assert REDACTED in joined
    assert "hf_abcdefghijklmnopqrstuvwxyz0123456789" not in log_path.read_text(encoding="utf-8")

    # progress reached the sink
    assert any("Exporting model" in event.message for event in sink.of_kind(ProgressKind.EXPORT))
    assert any("drill" in event.message for event in sink.of_kind(ProgressKind.LOG))

    # the job log holds every line of both streams
    ctx.logger.close()
    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw = [record for record in records if record.get("event") == "subprocess.output"]
    assert {record["stream"] for record in raw} == {"stdout", "stderr"}
    assert len(raw) == 5  # 3 stdout + 2 stderr
    finished = [record for record in records if record.get("event") == "subprocess.finished"]
    assert finished and finished[-1]["returncode"] == 3


def test_managed_temp_plan_matches_the_context_environment(context):
    ctx, _sink, _log, job_temp = context
    plan = plan_temp(
        job_temp_dir=job_temp,
        source_weight_bytes=18_819_721_040,
        settings=ctx.settings,
    )
    assert plan.required_bytes == 47_049_302_600
    assert plan.env() == ctx.temp_env

    result = SubprocessRunner().run(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ['TMPDIR']); print(os.environ['TEMP'])",
        ],
        ctx=ctx,
    )
    assert result.returncode == 0
    assert result.stdout_lines == [str(job_temp), str(job_temp)]
