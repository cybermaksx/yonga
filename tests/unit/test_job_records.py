"""What a job leaves in its log and state document, and what ``yonga logs`` makes of it.

The job log and the state document are the record a user reads after the fact.
Four things went wrong with them on a real host:

* a probe or a benchmark that succeeded left a log holding nothing but
  ``job.created`` and a state stuck at ``created``, so it looked live for a day;
* a failure's raw text (the NPU compiler's pass and shape) reached only the
  state document, while the remediation pointed at ``yonga logs JOB --raw``;
* a Ctrl-C was recorded as ``job.failed`` against a stage the job never
  reached, and its gigabyte of conversion temp was kept "for diagnosis";
* ``yonga logs`` listed ids and sizes only, and printed UTC clock times with no
  zone, so six jobs, one of them failed, were indistinguishable.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from tests.unit.test_cli import (
    FakeRuntimeFactory,
    FakeWiring,
    _staged_artifact,
    invoke,
    one_document,
    records,
)
from yonga.adapters.runtime_genai import ChatProfileLike, ChatRuntime
from yonga.application.jobs import (
    LIVE_JOB_GRACE_HOURS,
    JobStateStore,
    live_job_ids,
    open_job,
    summarise_job,
    summarise_jobs,
)
from yonga.config.settings import Settings
from yonga.domain.diagnostics import CompileError, ExitCode, FailureCategory, diagnostic
from yonga.domain.models import PipelineKind
from yonga.domain.plans import JobStage, JobStatus
from yonga.ui.tables import job_list

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

#: The NPU compiler's own words, which only the raw text carries.
COMPILER_PASS = "MultiClusterStrategyAssignment Pass failed : Cannot get per cluster memory shapes"


def settings_for(root: Path) -> Settings:
    """Settings confined to ``root``."""
    return Settings.model_validate(
        {"config_dir": root / "config", "cache_dir": root / "cache", "data_dir": root / "data"}
    )


def log_records(settings: Settings, job_id: str) -> list[dict[str, object]]:
    """Parse one job's JSON Lines log."""
    path = settings.paths.job_log_path(job_id)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def events(settings: Settings, job_id: str) -> list[str]:
    """The event names of one job's log, in order."""
    return [str(record["event"]) for record in log_records(settings, job_id)]


def only_job_id(settings: Settings) -> str:
    """The id of the single job a command ran."""
    logs = sorted(settings.paths.logs_dir.glob("*.jsonl"))
    assert len(logs) == 1, logs
    return logs[0].stem


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Private directories for one test."""
    return settings_for(tmp_path)


@pytest.fixture
def runner() -> CliRunner:
    """A Typer runner with stdout and stderr captured separately."""
    return CliRunner()


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """A private XDG root, laid out the way ``tests/unit/test_cli.invoke`` expects."""
    root = tmp_path / "home"
    (root / "config").mkdir(parents=True)
    return root


@pytest.fixture
def three_hours_east(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run in a fixed zone three hours ahead of UTC (POSIX ``TZ`` signs are inverted)."""
    monkeypatch.setenv("TZ", "Etc/GMT-3")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


# --- a job that ran to its end -----------------------------------------------------


def test_a_job_that_needs_no_stages_still_ends_completed(settings: Settings) -> None:
    """A probe or a benchmark never reached a terminal stage, so it looked live for a day."""
    with open_job(settings=settings, requested_model=MODEL, job_id="probe-job") as ctx:
        assert ctx.state is not None

    store = JobStateStore.load(settings.paths.job_state_path("probe-job"))
    assert store.stage is JobStage.COMPLETED
    assert [stage for _, stage in store.state.history] == ["created", "completed"]
    assert events(settings, "probe-job")[-1] == "job.completed"
    assert "probe-job" not in live_job_ids(settings.paths)


def test_a_preparation_keeps_its_ready_stage_and_logs_the_end(settings: Settings) -> None:
    """``ready`` is already terminal; completing must not overwrite it."""
    with open_job(settings=settings, requested_model=MODEL, job_id="prepare-job") as ctx:
        ctx.advance(JobStage.READY)

    store = JobStateStore.load(settings.paths.job_state_path("prepare-job"))
    assert store.stage is JobStage.READY
    completed = [r for r in log_records(settings, "prepare-job") if r["event"] == "job.completed"]
    assert len(completed) == 1
    assert completed[0]["stage"] == "ready"
    assert isinstance(completed[0]["duration_seconds"], float)


def test_a_recorded_failure_is_not_turned_into_a_completion(settings: Settings) -> None:
    """A service may record its failure and return; the job stays failed."""
    failure = diagnostic("probe.failed", FailureCategory.UNKNOWN_RUNTIME_ERROR, "Failed", "boom")
    with open_job(settings=settings, requested_model=MODEL, job_id="recorded") as ctx:
        ctx.fail(JobStage.PROBED, failure)

    assert JobStateStore.load(settings.paths.job_state_path("recorded")).stage is JobStage.FAILED
    assert "job.completed" not in events(settings, "recorded")


def test_probe_and_bench_jobs_end_completed_through_the_cli(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """The real commands, end to end: the state is terminal and names the command."""
    settings = settings_for(home)
    artifact = _staged_artifact(tmp_path)

    probed = invoke(runner, home, ["probe", str(artifact), "--device", "cpu"])
    assert probed.exit_code == int(ExitCode.SUCCESS), probed.stderr
    probe_id = only_job_id(settings)
    probe_state = JobStateStore.load(settings.paths.job_state_path(probe_id)).state
    assert probe_state.stage is JobStage.COMPLETED
    assert probe_state.command == "probe"
    assert events(settings, probe_id)[-1] == "job.completed"

    for path in settings.paths.logs_dir.iterdir():
        path.unlink()
    benched = invoke(
        runner,
        home,
        ["bench", str(artifact), "--device", "cpu", "--suite", "quick", "--runs", "1"],
    )
    assert benched.exit_code == int(ExitCode.SUCCESS), benched.stderr
    bench_id = only_job_id(settings)
    bench_state = JobStateStore.load(settings.paths.job_state_path(bench_id)).state
    assert bench_state.stage is JobStage.COMPLETED
    assert bench_state.command == "bench"
    assert "job.completed" in events(settings, bench_id)


# --- a failure's raw text ----------------------------------------------------------


class RefusingNpuFactory(FakeRuntimeFactory):
    """A runtime factory whose NPU compile fails the way the real compiler does."""

    def __call__(
        self,
        artifact_dir: Path,
        *,
        device: str = "CPU",
        pipeline: PipelineKind = PipelineKind.LLM,
        properties: Mapping[str, str] | None = None,
        chat_profile: ChatProfileLike | None = None,
        settings: Settings | None = None,
        thinking: bool | None = None,
    ) -> ChatRuntime:
        """Refuse every device with the compiler's raw text."""
        raw = f"Compilation failed\n--- native output ---\n[ERROR] [vpux-compiler] {COMPILER_PASS}"
        raise CompileError(
            diagnostic(
                "runtime.compile_failed",
                FailureCategory.COMPILER_UNSUPPORTED_GRAPH,
                "Model could not be compiled",
                "The NPU compiler refused this graph.",
                raw_excerpt=raw,
            )
        )


def test_a_failure_writes_its_raw_text_into_the_job_log(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """``yonga logs JOB --raw`` must hold what the remediation says it holds."""
    settings = settings_for(home)
    artifact = _staged_artifact(tmp_path)
    factory = RefusingNpuFactory()
    wiring_for = lambda s: FakeWiring(s, runtime_factory=factory)  # noqa: E731

    failed = invoke(
        runner,
        home,
        ["bench", str(artifact), "--device", "npu", "--suite", "quick"],
        factory=wiring_for,
    )
    assert failed.exit_code != int(ExitCode.SUCCESS)
    job_id = only_job_id(settings)

    detail = [r for r in log_records(settings, job_id) if r["event"] == "job.error_detail"]
    assert len(detail) == 1
    assert COMPILER_PASS in str(detail[0]["raw"])
    assert events(settings, job_id).index("job.error_detail") < events(settings, job_id).index(
        "job.failed"
    )

    raw = invoke(runner, home, ["logs", job_id[:8], "--raw"])
    assert raw.exit_code == int(ExitCode.SUCCESS)
    assert COMPILER_PASS in raw.stdout


def test_a_failure_without_raw_text_writes_no_detail_record(settings: Settings) -> None:
    """The detail record is for the tool's own words; there is nothing to repeat otherwise."""
    failure = diagnostic("x.failed", FailureCategory.UNKNOWN_RUNTIME_ERROR, "Failed", "boom")
    with (
        pytest.raises(RuntimeError),
        open_job(settings=settings, requested_model=MODEL, job_id="plain") as ctx,
    ):
        ctx.fail(JobStage.EXPORTED, failure)
        raise RuntimeError("re-raised")

    assert "job.error_detail" not in events(settings, "plain")


# --- a cancelled job ---------------------------------------------------------------


def cancelled_export() -> CompileError:
    """The typed error the exporter's cancellation becomes."""
    return CompileError(
        diagnostic(
            "export.cancelled",
            FailureCategory.USER_CANCELLED,
            "Conversion cancelled",
            "The conversion was cancelled before it finished. No artifact was produced.",
        )
    )


def test_a_cancelled_conversion_is_recorded_as_cancelled_at_the_stage_it_reached(
    settings: Settings,
) -> None:
    """Not ``job.failed`` at ``exported``: the export never finished, and nothing was wrong."""
    error = cancelled_export()
    with (
        pytest.raises(CompileError),
        open_job(settings=settings, requested_model=MODEL, job_id="ctrl-c") as ctx,
    ):
        ctx.advance(JobStage.DOWNLOADED)
        ctx.fail(JobStage.EXPORTED, error.diagnostic)
        raise error

    log = log_records(settings, "ctrl-c")
    assert "job.failed" not in [r["event"] for r in log]
    (ended,) = [r for r in log if r["event"] == "job.cancelled"]
    assert ended["level"] == "warning"
    assert ended["stage"] == "downloaded"
    assert ended["failed_before"] == "exported"
    assert ended["code"] == "export.cancelled"
    store = JobStateStore.load(settings.paths.job_state_path("ctrl-c"))
    assert store.stage is JobStage.CANCELLED
    assert "ctrl-c" not in live_job_ids(settings.paths)


def test_a_cancelled_job_does_not_keep_its_temp_tree(settings: Settings) -> None:
    """A Ctrl-C needs no diagnosis; its gigabyte of tool intermediates used to stay for 72 h."""
    error = cancelled_export()
    with (
        pytest.raises(CompileError),
        open_job(settings=settings, requested_model=MODEL, job_id="ctrl-c") as ctx,
    ):
        temp = ctx.temp_dir
        (temp / "nncf-intermediate.bin").write_bytes(b"\x00" * 4096)
        ctx.fail(JobStage.EXPORTED, error.diagnostic)
        raise error

    assert not temp.exists()
    names = events(settings, "ctrl-c")
    assert "job.temp_discarded" in names
    assert "job.temp_retained" not in names


def test_an_interrupt_removes_the_temp_tree_too(settings: Settings) -> None:
    """Ctrl-C outside any service is the same cancellation."""
    with (
        pytest.raises(KeyboardInterrupt),
        open_job(settings=settings, requested_model=MODEL, job_id="interrupted") as ctx,
    ):
        temp = ctx.temp_dir
        (temp / "partial.bin").write_bytes(b"\x00" * 16)
        raise KeyboardInterrupt

    assert not temp.exists()
    assert "job.cancelled" in events(settings, "interrupted")


def test_a_failed_job_still_keeps_its_temp_tree_for_diagnosis(settings: Settings) -> None:
    """Only cancellation changed: a real failure is still diagnosable from its temp."""
    failure = diagnostic("export.failed", FailureCategory.UNKNOWN_RUNTIME_ERROR, "Failed", "1")
    with (
        pytest.raises(RuntimeError),
        open_job(settings=settings, requested_model=MODEL, job_id="broken") as ctx,
    ):
        temp = ctx.temp_dir
        (temp / "partial.bin").write_bytes(b"\x00" * 16)
        ctx.fail(JobStage.EXPORTED, failure)
        raise RuntimeError("re-raised")

    assert temp.is_dir()
    assert "job.temp_retained" in events(settings, "broken")


# --- the index ---------------------------------------------------------------------


def write_job(
    settings: Settings,
    job_id: str,
    *,
    stage: str,
    command: str | None,
    model: str,
    started: datetime,
    took: timedelta,
    code: str | None = None,
    category: str = "unknown_runtime_error",
) -> None:
    """Write a job's log and state document as a finished job leaves them."""
    paths = settings.paths
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = started.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    record = {"timestamp": stamp, "job_id": job_id, "level": "info", "event": "job.created"}
    paths.job_log_path(job_id).write_text(json.dumps(record) + "\n", encoding="utf-8")
    state: dict[str, object] = {
        "job_id": job_id,
        "stage": stage,
        "created_at": started.isoformat(),
        "updated_at": (started + took).isoformat(),
        "requested_model": model,
        "command": command,
        "target_device": "CPU",
    }
    if code is not None:
        state["diagnostic"] = {
            "code": code,
            "category": category,
            "title": "t",
            "message": "m",
        }
    paths.job_state_path(job_id).write_text(json.dumps(state), encoding="utf-8")


STARTED = datetime(2026, 10, 10, 22, 33, 24, 663000, tzinfo=UTC)


def test_the_index_says_which_job_is_which_and_how_it_ended(
    runner: CliRunner, home: Path, three_hours_east: None
) -> None:
    """Command, model, outcome, duration, and a local start time that names its zone."""
    settings = settings_for(home)
    write_job(
        settings,
        "93b14bf7-73b0-4018-86a9-ff49f76e2933",
        stage="cancelled",
        command="prepare",
        model=MODEL,
        started=STARTED,
        took=timedelta(seconds=29),
        code="export.cancelled",
        category="user_cancelled",
    )
    write_job(
        settings,
        "70ab5941-40bd-4302-8837-c0ef8b66832e",
        stage="failed",
        command="bench",
        model="e3014dda15c8a5be",
        started=STARTED + timedelta(minutes=5),
        took=timedelta(seconds=2),
        code="npu.compiler.pass_failed",
    )

    shown = invoke(runner, home, ["logs"])
    assert shown.exit_code == int(ExitCode.SUCCESS)
    out = shown.stdout
    assert "93b14bf7" in out
    assert "70ab5941" in out
    assert "prepare" in out
    assert "bench" in out
    assert MODEL in out
    assert "cancelled" in out
    assert "failed" in out
    assert "UTC+03:00" in out
    # 22:33 UTC is 01:33 the next day, three hours east.
    assert "2026-10-11 01:33" in out
    assert "29.0 s" in out

    listed = one_document(invoke(runner, home, ["--json", "logs"]))
    jobs = {job["job_id"]: job for job in records(listed["jobs"])}
    cancelled = jobs["93b14bf7-73b0-4018-86a9-ff49f76e2933"]
    assert cancelled["status"] == "cancelled"
    assert cancelled["command"] == "prepare"
    assert cancelled["code"] == "export.cancelled"
    assert cancelled["duration_seconds"] == 29.0
    assert jobs["70ab5941-40bd-4302-8837-c0ef8b66832e"]["status"] == "failed"


def test_one_job_is_shown_in_local_time_with_its_outcome(
    runner: CliRunner, home: Path, three_hours_east: None
) -> None:
    """The record clock was UTC with no marker; the user's clock read three hours later."""
    settings = settings_for(home)
    write_job(
        settings,
        "93b14bf7-73b0-4018-86a9-ff49f76e2933",
        stage="cancelled",
        command="prepare",
        model=MODEL,
        started=STARTED,
        took=timedelta(seconds=29),
        code="export.cancelled",
        category="user_cancelled",
    )

    shown = invoke(runner, home, ["logs", "93b14bf7"])
    assert shown.exit_code == int(ExitCode.SUCCESS)
    assert "01:33:24.663" in shown.stdout
    assert "22:33:24.663" not in shown.stdout
    assert "UTC+03:00" in shown.stdout
    assert "cancelled (export.cancelled)" in shown.stdout
    assert f"prepare {MODEL} CPU" in shown.stdout


def test_old_and_unfinished_jobs_are_summarised_honestly(settings: Settings) -> None:
    """A Ctrl-C an older release recorded as failed, a dead job, a log with no state."""
    now = datetime.now(tz=UTC)
    write_job(
        settings,
        "old-cancel",
        stage="failed",
        command=None,
        model=MODEL,
        started=now,
        took=timedelta(seconds=3),
        code="job.cancelled",
        category="user_cancelled",
    )
    write_job(
        settings,
        "dead",
        stage="exported",
        command="prepare",
        model=MODEL,
        started=now,
        took=timedelta(seconds=3),
    )
    state = settings.paths.job_state_path("dead")
    stale = (now - timedelta(hours=LIVE_JOB_GRACE_HOURS + 1)).timestamp()
    os.utime(state, (stale, stale))
    write_job(
        settings,
        "running",
        stage="downloaded",
        command="prepare",
        model=MODEL,
        started=now,
        took=timedelta(seconds=3),
    )
    settings.paths.job_log_path("stateless").write_text(
        json.dumps({"timestamp": "2026-09-22T10:11:12.345Z", "event": "job.created"}) + "\n",
        encoding="utf-8",
    )

    by_id = {job.job_id: job for job in summarise_jobs(settings.paths)}
    assert by_id["old-cancel"].status is JobStatus.CANCELLED
    assert by_id["dead"].status is JobStatus.UNFINISHED
    assert by_id["dead"].duration_seconds is None
    assert by_id["running"].status is JobStatus.RUNNING
    stateless = summarise_job(settings.paths.job_log_path("stateless"), paths=settings.paths)
    assert stateless.status is JobStatus.UNKNOWN
    assert stateless.started_at == datetime(2026, 9, 22, 10, 11, 12, 345000, tzinfo=UTC)


def test_a_narrow_index_drops_columns_before_it_hides_which_job_is_which(
    settings: Settings, three_hours_east: None
) -> None:
    """At 60 columns the id prefix, the command and the outcome are still whole."""
    write_job(
        settings,
        "61926a20-16b1-45e4-8392-5089d9f8c25e",
        stage="cancelled",
        command="bench",
        model="/home/someone/.local/share/yonga/artifacts/Qwen--Qwen2.5-0.5B-Instruct/"
        "7ae557604adf--cpu-int4-asym-g128-r08",
        started=STARTED,
        took=timedelta(seconds=6.6),
        code="job.cancelled",
        category="user_cancelled",
    )
    console = Console(width=60, record=True, color_system=None)
    console.print(job_list(summarise_jobs(settings.paths)))
    text = console.export_text()

    assert all(len(line) <= 60 for line in text.splitlines())
    assert "61926a20" in text
    assert "bench" in text
    assert "cancelled" in text
    # A path is shown by its last component, never as the home directory.
    assert "/home/someone" not in text
    assert "7ae557604adf" in text
