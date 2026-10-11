"""Retention of the two working roots a job owns: staging and the managed temp tree.

Both roots are invisible: ``yonga list`` hides ``artifacts/.staging`` by design,
and nothing ever mentioned ``cache/tmp``. Before this policy existed a failed
preparation left its whole staging directory behind for ever and a successful one
left the emptied husk, which is how a reference host accumulated 5.6 GB in 15
staging directories and 19 empty temp directories without a single line of output
saying so.

Every test here pins one half of the contract in ``SPEC.md`` section 14: what a
job leaves behind when it fails, what it leaves behind when it succeeds, and what
the bounded sweep is and is not allowed to delete. The safety rules are the
important ones — a sweep that follows a symlink or deletes a running job's export
is far worse than the leak it replaces.
"""

from __future__ import annotations

import json
import os
import shlex
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from yonga.adapters.filesystem import INCOMPLETE_MARKER_NAME
from yonga.application.jobs import (
    LIVE_JOB_GRACE_HOURS,
    JobStateStore,
    live_job_ids,
    open_job,
    sweep_job_directories,
    sweep_managed_workspaces,
)
from yonga.config.settings import Settings
from yonga.domain.diagnostics import FailureCategory, diagnostic
from yonga.domain.plans import JobStage

MODEL = "nDimensional/Qwen3.5-9B-Uncensored-Safetensors"


def settings_for(tmp_path: Path, **overrides: object) -> Settings:
    """Build settings confined to ``tmp_path``, with an explicit retention policy."""
    payload: dict[str, object] = {
        "config_dir": tmp_path / "config",
        "cache_dir": tmp_path / "cache",
        "data_dir": tmp_path / "data",
    }
    payload.update(overrides)
    return Settings.model_validate(payload)


def age(path: Path, *, hours: float) -> Path:
    """Backdate a directory's modification time by ``hours``."""
    moment = datetime.now(tz=UTC) - timedelta(hours=hours)
    os.utime(path, (moment.timestamp(), moment.timestamp()))
    return path


def make_working_dir(root: Path, name: str, *, content: bool = True, hours: float = 0.0) -> Path:
    """Create one job working directory under ``root``."""
    directory = root / name
    (directory / "model").mkdir(parents=True, exist_ok=True)
    (directory / "model" / INCOMPLETE_MARKER_NAME).write_text("job\n", encoding="utf-8")
    if content:
        (directory / "model" / "openvino_model.bin").write_bytes(b"\x00" * 2048)
    return age(directory, hours=hours)


def write_state(settings: Settings, job_id: str, stage: str, *, hours: float = 0.0) -> Path:
    """Write a job state document in ``stage`` and backdate it by ``hours``."""
    path = settings.paths.job_state_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=UTC)
    path.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "stage": stage,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "requested_model": MODEL,
            }
        ),
        encoding="utf-8",
    )
    return age(path, hours=hours)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings whose retention policy is short enough to observe in a test."""
    return settings_for(tmp_path, workspace_retention_hours=6, workspace_retention_max=1)


# --- what a job leaves behind --------------------------------------------------------


def test_a_successful_job_removes_its_temp_tree_and_its_staging_husk(
    settings: Settings,
) -> None:
    """Publication renames the model out of staging; the empty parent must go too."""
    with open_job(settings=settings, requested_model=MODEL, job_id="job-success") as ctx:
        (ctx.staging_dir / "model").mkdir(parents=True)
        staging = ctx.staging_dir
        temp = ctx.temp_dir
        assert temp.is_dir()

    assert not staging.exists()
    assert not temp.exists()


def test_a_failed_job_discards_a_staging_directory_that_holds_nothing(
    settings: Settings,
) -> None:
    """An export that wrote no file diagnoses nothing, so its husk is not kept."""
    with (
        pytest.raises(RuntimeError),
        open_job(settings=settings, requested_model=MODEL, job_id="job-empty") as ctx,
    ):
        staging = ctx.staging_dir
        (staging / "model").mkdir(parents=True)
        (staging / "model" / INCOMPLETE_MARKER_NAME).write_text("x\n", encoding="utf-8")
        raise RuntimeError("export died")

    assert not staging.exists()


def test_a_failed_job_keeps_a_staging_directory_that_holds_a_partial_export(
    settings: Settings,
) -> None:
    """SPEC section 16 keeps raw diagnostics: real bytes survive for inspection."""
    with (
        pytest.raises(RuntimeError),
        open_job(settings=settings, requested_model=MODEL, job_id="job-partial") as ctx,
    ):
        staging = ctx.staging_dir
        (staging / "model").mkdir(parents=True)
        (staging / "model" / "openvino_model.xml").write_text("<net/>\n", encoding="utf-8")
        raise RuntimeError("validation died")

    assert staging.is_dir()
    assert (staging / "model" / "openvino_model.xml").is_file()


def test_failing_a_job_records_the_retained_staging_path(settings: Settings) -> None:
    """The user must be able to find the partial export from the error alone."""
    failure = diagnostic(
        "export.failed",
        FailureCategory.UNKNOWN_RUNTIME_ERROR,
        "Export failed",
        "The exporter returned 1.",
    )
    with open_job(settings=settings, requested_model=MODEL, job_id="job-recorded") as ctx:
        (ctx.staging_dir / "model").mkdir(parents=True)
        (ctx.staging_dir / "model" / "openvino_model.xml").write_text("<net/>\n", encoding="utf-8")
        ctx.fail(JobStage.EXPORTED, failure)
        state_path = ctx.state.path if ctx.state is not None else None
        staging = ctx.staging_dir

    assert failure.context["staging_dir"] == str(staging)
    assert any(str(staging) in line for line in failure.remediation)
    assert state_path is not None
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["diagnostic"]["context"]["staging_dir"] == str(staging)


def test_the_staging_decision_is_taken_once(settings: Settings) -> None:
    """A service and ``open_job`` both call it; the second call must be a no-op."""
    with open_job(settings=settings, requested_model=MODEL, job_id="job-once") as ctx:
        (ctx.staging_dir / "model").mkdir(parents=True)
        (ctx.staging_dir / "model" / "openvino_model.xml").write_text("<net/>\n", encoding="utf-8")
        first = ctx.retain_staging_for_diagnosis()
        (ctx.staging_dir / "model" / "openvino_model.xml").unlink()
        second = ctx.retain_staging_for_diagnosis()

    assert first == ctx.staging_dir
    assert second == first


# --- the sweep -----------------------------------------------------------------------


def test_the_sweep_retires_directories_past_the_retention_window(settings: Settings) -> None:
    """Age is one of the two bounds on the staging root."""
    root = settings.paths.staging_dir
    root.mkdir(parents=True, exist_ok=True)
    old = make_working_dir(root, "old-job", hours=48)
    recent = make_working_dir(root, "recent-job", hours=1)

    report = sweep_job_directories(root, paths=settings.paths, settings=settings)

    assert old in report.removed
    assert recent in report.kept
    assert not old.exists()
    assert recent.is_dir()
    assert report.freed_bytes >= 2048


def test_the_sweep_keeps_only_the_configured_number_of_recent_directories(
    tmp_path: Path,
) -> None:
    """Count is the other bound: a burst of failures cannot fill the disk."""
    settings = settings_for(tmp_path, workspace_retention_hours=999, workspace_retention_max=2)
    root = settings.paths.staging_dir
    root.mkdir(parents=True, exist_ok=True)
    newest = make_working_dir(root, "job-c", hours=1)
    middle = make_working_dir(root, "job-b", hours=2)
    oldest = make_working_dir(root, "job-a", hours=3)

    report = sweep_job_directories(root, paths=settings.paths, settings=settings)

    assert set(report.kept) == {newest, middle}
    assert report.removed == [oldest]
    assert not oldest.exists()


def test_the_sweep_always_retires_an_empty_husk(tmp_path: Path) -> None:
    """A husk holding only the incomplete marker never counts as a diagnostic."""
    settings = settings_for(tmp_path, workspace_retention_hours=999, workspace_retention_max=99)
    root = settings.paths.staging_dir
    root.mkdir(parents=True, exist_ok=True)
    husk = make_working_dir(root, "job-husk", content=False, hours=0)
    real = make_working_dir(root, "job-real", hours=0)

    report = sweep_job_directories(root, paths=settings.paths, settings=settings)

    assert report.removed == [husk]
    assert real.is_dir()


def test_the_sweep_never_touches_a_running_jobs_directory(settings: Settings) -> None:
    """A live export is untouchable however old its directory looks."""
    root = settings.paths.staging_dir
    root.mkdir(parents=True, exist_ok=True)
    running = make_working_dir(root, "job-running", hours=48)
    finished = make_working_dir(root, "job-finished", hours=48)
    write_state(settings, "job-running", "exported")
    write_state(settings, "job-finished", "failed")

    report = sweep_job_directories(root, paths=settings.paths, settings=settings)

    assert running.is_dir()
    assert running in report.kept
    assert not finished.exists()


def test_the_sweep_respects_an_explicitly_protected_job(settings: Settings) -> None:
    """The job that asked for the sweep is never its own victim."""
    root = settings.paths.staging_dir
    root.mkdir(parents=True, exist_ok=True)
    mine = make_working_dir(root, "job-mine", hours=48)

    report = sweep_job_directories(
        root, paths=settings.paths, settings=settings, protected=("job-mine",)
    )

    assert mine.is_dir()
    assert report.removed == []


def test_the_sweep_refuses_symlinks_and_anything_outside_the_root(settings: Settings) -> None:
    """Following a link out of the staging root would delete somebody else's data."""
    root = settings.paths.staging_dir
    root.mkdir(parents=True, exist_ok=True)
    outside = settings.paths.data_dir / "precious"
    outside.mkdir(parents=True, exist_ok=True)
    (outside / "keep.txt").write_text("must survive\n", encoding="utf-8")
    age(outside, hours=72)
    link = root / "job-link"
    link.symlink_to(outside, target_is_directory=True)
    hidden = root / ".not-a-job"
    hidden.mkdir()
    age(hidden, hours=72)

    report = sweep_job_directories(root, paths=settings.paths, settings=settings)

    assert report.removed == []
    assert outside.is_dir()
    assert (outside / "keep.txt").is_file()
    assert link.is_symlink()
    assert hidden.is_dir()


def test_a_dry_run_reports_without_deleting(settings: Settings) -> None:
    """``yonga list`` projects the sweep on every run, so it must be side-effect free."""
    root = settings.paths.staging_dir
    root.mkdir(parents=True, exist_ok=True)
    old = make_working_dir(root, "job-old", hours=48)

    report = sweep_job_directories(root, paths=settings.paths, settings=settings, dry_run=True)

    assert report.dry_run is True
    assert report.removed == [old]
    assert report.freed_bytes >= 2048
    assert old.is_dir()


def test_the_sweep_survives_a_missing_root(settings: Settings) -> None:
    """A host that never prepared anything has no staging root at all."""
    report = sweep_job_directories(
        settings.paths.staging_dir / "nowhere",
        paths=settings.paths,
        settings=settings,
    )

    assert report.removed == []
    assert report.freed_bytes == 0


# --- the managed temp tree (the same policy, the other root) -------------------------


def test_managed_temp_directories_are_swept_like_staging(settings: Settings) -> None:
    """``open_job`` only ever removed the temp tree on the success path."""
    temp_root = settings.paths.tmp_dir
    temp_root.mkdir(parents=True, exist_ok=True)
    leaked = [make_working_dir(temp_root, f"temp-{index}", content=False) for index in range(5)]

    sweep = sweep_managed_workspaces(settings)

    assert sweep.temp.removed_count == len(leaked)
    assert not any(path.exists() for path in leaked)


def test_opening_a_job_sweeps_both_roots_but_spares_the_new_job(settings: Settings) -> None:
    """The bounded sweep runs on job startup, which is the only place it is free."""
    staging_root = settings.paths.staging_dir
    temp_root = settings.paths.tmp_dir
    staging_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    stale_staging = make_working_dir(staging_root, "stale-staging", hours=48)
    stale_temp = make_working_dir(temp_root, "stale-temp", hours=48)

    with open_job(settings=settings, requested_model=MODEL, job_id="fresh-job") as ctx:
        assert ctx.temp_dir.is_dir()
        assert not stale_staging.exists()
        assert not stale_temp.exists()


# --- liveness ------------------------------------------------------------------------


def test_live_job_ids_reads_the_state_documents(settings: Settings) -> None:
    """Only a non-terminal, recently written state means 'still running'."""
    write_state(settings, "running", "downloaded")
    write_state(settings, "done", "ready")
    write_state(settings, "broken", "failed")

    assert live_job_ids(settings.paths) == frozenset({"running"})


def test_an_interrupted_job_records_that_it_ended(settings: Settings) -> None:
    """Ctrl-C used to leave the state at the last stage, so the job looked live for a day."""
    with (
        pytest.raises(KeyboardInterrupt),
        open_job(settings=settings, requested_model=MODEL, job_id="interrupted") as ctx,
    ):
        ctx.advance(JobStage.VALIDATING)
        raise KeyboardInterrupt

    store = JobStateStore.load(settings.paths.job_state_path("interrupted"))
    assert store.stage is JobStage.CANCELLED
    assert store.state.diagnostic is not None
    assert store.state.diagnostic.code == "job.cancelled"
    assert "interrupted" not in live_job_ids(settings.paths)


def test_a_recorded_failure_is_not_overwritten_on_the_way_out(settings: Settings) -> None:
    recorded = diagnostic(
        "export.failed", FailureCategory.UNKNOWN_RUNTIME_ERROR, "Export failed", "boom"
    )
    with (
        pytest.raises(RuntimeError),
        open_job(settings=settings, requested_model=MODEL, job_id="failed") as ctx,
    ):
        ctx.fail(JobStage.EXPORTED, recorded)
        raise RuntimeError("the service re-raised")

    store = JobStateStore.load(settings.paths.job_state_path("failed"))
    assert store.state.diagnostic is not None
    assert store.state.diagnostic.code == "export.failed"


def test_live_job_ids_forgets_a_job_that_stopped_writing(settings: Settings) -> None:
    """A crashed job must not pin gigabytes for ever by never reaching a terminal stage."""
    write_state(settings, "abandoned", "exported", hours=LIVE_JOB_GRACE_HOURS + 1)

    assert live_job_ids(settings.paths) == frozenset()


def test_an_unreadable_state_document_counts_as_live(settings: Settings) -> None:
    """Refusing to delete is the safe answer when the state cannot be read."""
    path = settings.paths.job_state_path("garbled")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    assert "garbled" in live_job_ids(settings.paths)


def test_a_real_job_state_store_is_seen_as_live(settings: Settings) -> None:
    """The document ``JobStateStore`` actually writes must parse for the sweep."""
    JobStateStore.create(paths=settings.paths, job_id="written", requested_model=MODEL)

    assert "written" in live_job_ids(settings.paths)


# --- what the user is told -----------------------------------------------------------


def make_bulky_staging(home: Path, name: str, *, size: int) -> Path:
    """Create a stale staging directory of ``size`` apparent bytes (sparsely)."""
    directory = home / "data" / "artifacts" / ".staging" / name / "model"
    directory.mkdir(parents=True)
    with (directory / "openvino_language_model.bin").open("wb") as handle:
        handle.truncate(size)
    return age(directory.parent, hours=500)


def run_cli(home: Path, *args: str) -> Result:
    """Invoke the real CLI against a private XDG root."""
    from yonga.application.wiring import Wiring
    from yonga.cli import app

    return CliRunner().invoke(
        app,
        ["--data-dir", str(home / "data"), "--cache-dir", str(home / "cache"), *args],
        obj=lambda settings: Wiring(settings),
        env={
            "YONGA_CONFIG_DIR": str(home / "config"),
            "NO_COLOR": "1",
            "COLUMNS": "200",
        },
    )


def test_list_says_when_stale_working_directories_hold_real_space(tmp_path: Path) -> None:
    """``list`` hides the staging root, so it owes the user one honest line."""
    home = tmp_path / "home"
    (home / "config").mkdir(parents=True)
    stale = make_bulky_staging(home, "dead-job", size=3 * 1024**3)

    result = run_cli(home, "list")

    assert result.exit_code == 0
    root = shlex.join(
        ["yonga", "--data-dir", str(home / "data"), "--cache-dir", str(home / "cache")]
    )
    expected = f"in 1 stale working directory can be reclaimed: {root} list --clean"
    assert expected in " ".join(result.stdout.split())
    assert stale.is_dir()


def test_list_clean_reclaims_the_space_and_reports_it(tmp_path: Path) -> None:
    """``--clean`` is the explicit way to get the disk back."""
    home = tmp_path / "home"
    (home / "config").mkdir(parents=True)
    stale = make_bulky_staging(home, "dead-job", size=3 * 1024**3)

    result = run_cli(home, "list", "--clean")

    assert result.exit_code == 0
    assert "from 1 stale working directory." in result.stdout
    assert not stale.exists()


def test_list_stays_quiet_when_nothing_is_stale(tmp_path: Path) -> None:
    """A fresh installation must not be nagged about an empty staging root."""
    home = tmp_path / "home"
    (home / "config").mkdir(parents=True)

    result = run_cli(home, "list")

    assert result.exit_code == 0
    assert "--clean" not in result.stdout


def test_list_json_describes_the_stale_working_directories(tmp_path: Path) -> None:
    """``--json`` carries the same news as the human line, in one document."""
    home = tmp_path / "home"
    (home / "config").mkdir(parents=True)
    make_bulky_staging(home, "dead-job", size=2 * 1024**3)

    result = run_cli(home, "list", "--json")

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    stale = payload["stale_working_dirs"]
    assert stale["cleaned"] is False
    assert stale["directories"] == 1
    assert stale["bytes"] >= 2 * 1024**3
