"""Integration tests for the export adapter over a real subprocess.

These drive :class:`~yonga.adapters.exporter_optimum.OptimumExporter` through the
real ``SubprocessRunner`` against ``tests/fixtures/fake_optimum_cli.py``, so the
parts that only a genuine process can prove are actually proven: the argv list
reaches the child verbatim with no shell in between, ``TMPDIR`` really points at
the managed job temp, output is captured and redacted, non-zero exits become
typed diagnostics, and cancellation stops a running conversion.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest

from yonga.adapters import CommandRunner
from yonga.adapters.exporter_optimum import OptimumExporter
from yonga.application.jobs import JobContext, open_job
from yonga.config.settings import load_settings
from yonga.domain.diagnostics import FailureCategory
from yonga.domain.plans import ConversionProfile, ExportRequest, QuantizationSettings
from yonga.util.redaction import REDACTED

FAKE_CLI = Path(__file__).resolve().parents[1] / "fixtures" / "fake_optimum_cli.py"

#: The fake stands in for the whole ``python -m optimum.commands.optimum_cli`` prefix.
LAUNCHER = [sys.executable, str(FAKE_CLI)]


@pytest.fixture(scope="module")
def make_runner() -> Callable[[], CommandRunner]:
    """A factory for the real subprocess runner, skipped when it is not available yet."""
    module = pytest.importorskip(
        "yonga.adapters.subprocess",
        reason="the subprocess adapter is not available in this checkout",
    )
    factory = module.SubprocessRunner

    def build() -> CommandRunner:
        runner: CommandRunner = factory()
        return runner

    return build


@pytest.fixture
def job(tmp_path: Path) -> Iterator[JobContext]:
    """A real job context, with every managed directory inside ``tmp_path``."""
    settings = load_settings(
        env={},
        overrides={
            "data_dir": str(tmp_path / "data"),
            "cache_dir": str(tmp_path / "cache"),
            "config_dir": str(tmp_path / "config"),
        },
    )
    with open_job(settings=settings, requested_model="acme/model", job_id="fake-export") as ctx:
        yield ctx


def make_request(ctx: JobContext, *, extra_args: Sequence[str] = ()) -> ExportRequest:
    """An NPU channel-wise export request writing into this job's staging directory."""
    profile = ConversionProfile(
        id="npu-int4-sym-cw",
        device="npu",
        quantization=QuantizationSettings(
            weight_format="int4", symmetric=True, ratio=1.0, group_size=-1
        ),
        export_extra_args=list(extra_args),
    )
    return ExportRequest(
        source_path=Path("/snapshots/acme/model"),
        output_path=ctx.staging_dir / "model",
        task="text-generation-with-past",
        profile=profile,
    )


def read_json(path: Path) -> object:
    """Read one JSON document the fake exporter wrote."""
    return json.loads(path.read_text(encoding="utf-8"))


def test_argv_reaches_the_child_verbatim_with_no_shell(
    job: JobContext, make_runner: Callable[[], CommandRunner]
) -> None:
    hostile = "$(touch pwned); `id` && echo *"
    request = make_request(job, extra_args=["--stderr-line", hostile])
    result = OptimumExporter(make_runner(), launcher=LAUNCHER).export(request, job)

    assert result.success is True, result.stderr_tail
    received = read_json(request.output_path / "argv.json")
    assert isinstance(received, list)
    assert received[0] == "export"
    assert received[1] == "openvino"
    assert received[-1] == str(request.output_path)
    assert "--group-size=-1" in received
    assert received[received.index("--stderr-line") + 1] == hostile
    # A shell would have expanded the substitution or the glob.
    assert not (job.temp_dir / "pwned").exists()
    assert not (request.output_path / "pwned").exists()
    assert any(hostile in line for line in result.stderr_tail)


def test_required_npu_profile_flags_survive_the_round_trip(
    job: JobContext, make_runner: Callable[[], CommandRunner]
) -> None:
    request = make_request(job)
    OptimumExporter(make_runner(), launcher=LAUNCHER).export(request, job)
    received = read_json(request.output_path / "argv.json")
    assert isinstance(received, list)
    window = received[received.index("--weight-format") : received.index("--group-size=-1") + 1]
    assert window == ["--weight-format", "int4", "--sym", "--ratio", "1.0", "--group-size=-1"]


def test_managed_temp_directory_is_visible_to_the_child(
    job: JobContext, make_runner: Callable[[], CommandRunner]
) -> None:
    request = make_request(job)
    OptimumExporter(make_runner(), launcher=LAUNCHER).export(request, job)
    seen = read_json(request.output_path / "env.json")
    assert isinstance(seen, dict)
    assert seen["TMPDIR"] == str(job.temp_dir)
    assert seen["TMP"] == str(job.temp_dir)
    assert seen["TEMP"] == str(job.temp_dir)
    assert Path(str(seen["TMPDIR"])).is_dir()


def test_successful_export_writes_its_fixture_files_and_captures_logs(
    job: JobContext, make_runner: Callable[[], CommandRunner]
) -> None:
    request = make_request(job)
    result = OptimumExporter(make_runner(), launcher=LAUNCHER).export(request, job)

    assert result.success is True
    assert result.returncode == 0
    assert result.duration_seconds >= 0.0
    assert (request.output_path / "openvino_model.xml").is_file()
    assert (request.output_path / "config.json").is_file()
    assert any("Export finished" in line for line in result.stdout_tail)
    assert any("Applying Weight Compression" in line for line in result.stderr_tail)
    assert result.diagnostic is None
    assert result.package_versions


def test_job_log_records_the_export_command(
    job: JobContext, make_runner: Callable[[], CommandRunner]
) -> None:
    request = make_request(job)
    OptimumExporter(make_runner(), launcher=LAUNCHER).export(request, job)

    log_path = job.logger.path
    assert log_path is not None
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    events = {record.get("event") for record in records}
    assert "export.command" in events
    command = next(record for record in records if record.get("event") == "export.command")
    assert command["argv"][:2] == LAUNCHER


def test_secrets_never_reach_the_captured_output(
    job: JobContext, make_runner: Callable[[], CommandRunner]
) -> None:
    request = make_request(job, extra_args=["--emit-secret"])
    result = OptimumExporter(make_runner(), launcher=LAUNCHER).export(request, job)

    combined = " ".join([*result.stdout_tail, *result.stderr_tail])
    assert "hf_A1b2C3d4" not in combined
    assert REDACTED in combined

    log_text = Path(str(job.logger.path)).read_text(encoding="utf-8")
    assert "hf_A1b2C3d4" not in log_text


def test_nonzero_exit_becomes_a_typed_diagnostic(
    job: JobContext, make_runner: Callable[[], CommandRunner]
) -> None:
    request = make_request(job, extra_args=["--fail-with", "3", "--no-write-files"])
    result = OptimumExporter(make_runner(), launcher=LAUNCHER).export(request, job)

    assert result.success is False
    assert result.returncode == 3
    assert result.diagnostic is not None
    assert result.diagnostic.code == "export.failed"
    assert result.diagnostic.category is FailureCategory.UNKNOWN_RUNTIME_ERROR
    assert result.diagnostic.job_id == job.job_id
    assert not (request.output_path / "openvino_model.xml").exists()


def test_disk_exhaustion_signature_is_classified(
    job: JobContext, make_runner: Callable[[], CommandRunner]
) -> None:
    request = make_request(
        job,
        extra_args=[
            "--stderr-line",
            "RuntimeError: basic_ios::clear: iostream error",
            "--fail-with",
            "1",
            "--no-write-files",
        ],
    )
    result = OptimumExporter(make_runner(), launcher=LAUNCHER).export(request, job)

    assert result.diagnostic is not None
    assert result.diagnostic.code == "export.insufficient_disk"
    assert result.diagnostic.category is FailureCategory.INSUFFICIENT_DISK
    assert "tmpfs" in result.diagnostic.message


@pytest.mark.timeout(90)
def test_cancellation_stops_a_running_export(
    job: JobContext, make_runner: Callable[[], CommandRunner]
) -> None:
    request = make_request(job, extra_args=["--sleep", "60"])
    canceller = threading.Timer(1.5, job.cancelled.set)
    canceller.start()
    try:
        result = OptimumExporter(make_runner(), launcher=LAUNCHER).export(request, job)
    finally:
        canceller.cancel()

    assert result.success is False
    assert result.duration_seconds < 55.0
    assert result.diagnostic is not None
    assert result.diagnostic.code == "export.cancelled"
    assert result.diagnostic.category is FailureCategory.USER_CANCELLED


def test_fake_exporter_runs_standalone(tmp_path: Path) -> None:
    """The fixture must stay usable as a plain script, independent of yonga."""
    import subprocess

    output = tmp_path / "out"
    completed = subprocess.run(
        [sys.executable, str(FAKE_CLI), "export", "openvino", "--model", "x", str(output)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0
    assert (output / "openvino_model.xml").is_file()
