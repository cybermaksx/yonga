"""Unit tests for the Optimum Intel export adapter.

These never spawn a process: the :class:`~yonga.adapters.CommandRunner` is faked so
argv construction, environment handling, staging containment, redaction and
failure classification can be asserted deterministically. The real subprocess path
is covered by ``tests/integration/test_exporter_fake.py``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from yonga.adapters import CommandRunner, Exporter
from yonga.adapters.exporter_optimum import (
    EXPORT_PACKAGES,
    OPTIMUM_CLI_MODULE,
    OptimumExporter,
    build_export_argv,
    check_exporter_requirements,
    is_disk_exhaustion,
    quantization_flags,
    resolve_launcher,
)
from yonga.adapters.filesystem import INCOMPLETE_MARKER_NAME, mark_incomplete
from yonga.application.jobs import JobContext, open_job
from yonga.application.progress import CollectingSink, ProgressKind
from yonga.config.settings import load_settings
from yonga.domain.diagnostics import ConfigurationError, FailureCategory
from yonga.domain.plans import (
    ConversionProfile,
    ExporterRequirements,
    ExportRequest,
    QuantizationSettings,
)
from yonga.util.redaction import REDACTED
from yonga.util.versions import installed_version

LAUNCHER = ["/usr/bin/python3", "-m", OPTIMUM_CLI_MODULE]

# --- profile fixtures matching SPEC.md section 9 ---

PROFILE_FLAGS: dict[str, list[str]] = {
    "npu-int4-sym-cw": ["--weight-format", "int4", "--sym", "--ratio", "1.0", "--group-size=-1"],
    "npu-int4-sym-g128": ["--weight-format", "int4", "--sym", "--ratio", "1.0", "--group-size=128"],
    "gpu-int4-asym-g128": ["--weight-format", "int4", "--ratio", "1.0", "--group-size=128"],
    "cpu-int4-asym-g128": ["--weight-format", "int4", "--ratio", "1.0", "--group-size=128"],
}


def make_profile(profile_id: str) -> ConversionProfile:
    """Build one of the four required conversion profiles."""
    quantization = {
        "npu-int4-sym-cw": QuantizationSettings(
            weight_format="int4", symmetric=True, ratio=1.0, group_size=-1
        ),
        "npu-int4-sym-g128": QuantizationSettings(
            weight_format="int4", symmetric=True, ratio=1.0, group_size=128
        ),
        "gpu-int4-asym-g128": QuantizationSettings(
            weight_format="int4", symmetric=False, ratio=1.0, group_size=128
        ),
        "cpu-int4-asym-g128": QuantizationSettings(
            weight_format="int4", symmetric=False, ratio=1.0, group_size=128
        ),
    }[profile_id]
    device = profile_id.split("-", 1)[0]
    return ConversionProfile(id=profile_id, device=device, quantization=quantization)


# --- fake command runner -------------------------------------------------


@dataclass
class FakeCommandResult:
    """Stands in for a real ``CommandResult``."""

    argv: Sequence[str]
    returncode: int = 0
    duration_seconds: float = 1.5
    stdout_lines: Sequence[str] = field(default_factory=list)
    stderr_lines: Sequence[str] = field(default_factory=list)
    cancelled: bool = False
    timed_out: bool = False


@dataclass
class RecordedCall:
    """One captured invocation."""

    argv: list[str]
    env: dict[str, str]
    cwd: Path | None
    timeout: float | None


class FakeRunner:
    """A ``CommandRunner`` that records the call and replays a canned result."""

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: Sequence[str] = (),
        stderr: Sequence[str] = (),
        cancelled: bool = False,
    ) -> None:
        self.calls: list[RecordedCall] = []
        self._returncode = returncode
        self._stdout = list(stdout)
        self._stderr = list(stderr)
        self._cancelled = cancelled

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float | None = None,
        ctx: JobContext | None = None,
    ) -> FakeCommandResult:
        """Record the invocation and return the configured result."""
        self.calls.append(
            RecordedCall(argv=list(argv), env=dict(env or {}), cwd=cwd, timeout=timeout)
        )
        return FakeCommandResult(
            argv=list(argv),
            returncode=self._returncode,
            stdout_lines=self._stdout,
            stderr_lines=self._stderr,
            cancelled=self._cancelled,
        )


@pytest.fixture
def job(tmp_path: Path) -> Iterator[JobContext]:
    """A real job context rooted entirely inside ``tmp_path``."""
    settings = load_settings(
        env={},
        overrides={
            "data_dir": str(tmp_path / "data"),
            "cache_dir": str(tmp_path / "cache"),
            "config_dir": str(tmp_path / "config"),
        },
    )
    with open_job(settings=settings, requested_model="acme/model", job_id="job-test") as ctx:
        yield ctx


def make_request(
    job_ctx: JobContext,
    *,
    profile_id: str = "gpu-int4-asym-g128",
    task: str | None = "text-generation-with-past",
    trust_remote_code: bool = False,
    extra_args: Sequence[str] = (),
    extra_env: Mapping[str, str] | None = None,
    output_path: Path | None = None,
) -> ExportRequest:
    """Build an export request pointing at this job's staging directory."""
    profile = make_profile(profile_id)
    profile = profile.model_copy(update={"export_extra_args": list(extra_args)})
    return ExportRequest(
        source_path=Path("/snapshots/acme/model"),
        output_path=output_path or (job_ctx.staging_dir / "model"),
        task=task,
        profile=profile,
        trust_remote_code=trust_remote_code,
        extra_env=dict(extra_env or {}),
    )


# --- argv construction ---------------------------------------------------


#: These read the *really installed* exporter packages, which only the ``export``
#: extra provides; without it there is nothing to read, and that is not a failure.
requires_export_extra = pytest.mark.skipif(
    installed_version("optimum-intel") is None, reason="needs the yonga[export] extra"
)


@pytest.mark.parametrize("profile_id", sorted(PROFILE_FLAGS))
def test_required_profiles_produce_documented_flags(profile_id: str) -> None:
    profile = make_profile(profile_id)
    assert quantization_flags(profile.quantization) == PROFILE_FLAGS[profile_id]


def test_gpu_and_cpu_profiles_share_flags_but_not_identity() -> None:
    gpu = make_profile("gpu-int4-asym-g128")
    cpu = make_profile("cpu-int4-asym-g128")
    assert quantization_flags(gpu.quantization) == quantization_flags(cpu.quantization)
    assert gpu.id != cpu.id
    assert (gpu.device, cpu.device) == ("GPU", "CPU")


def test_sym_is_omitted_for_asymmetric_and_unset() -> None:
    asymmetric = QuantizationSettings(weight_format="int4", symmetric=False, ratio=1.0)
    unset = QuantizationSettings(weight_format="int4", symmetric=None, ratio=1.0)
    assert "--sym" not in quantization_flags(asymmetric)
    assert "--sym" not in quantization_flags(unset)
    assert "--sym" in quantization_flags(QuantizationSettings(weight_format="int4", symmetric=True))


def test_group_size_uses_equals_form_so_minus_one_is_never_a_flag() -> None:
    flags = quantization_flags(QuantizationSettings(weight_format="int4", group_size=-1))
    assert "--group-size=-1" in flags
    assert "-1" not in flags


def test_ratio_keeps_a_decimal_point() -> None:
    assert quantization_flags(QuantizationSettings(ratio=1.0))[-1] == "1.0"
    assert quantization_flags(QuantizationSettings(ratio=0.8))[-1] == "0.8"


def test_optional_data_aware_flags() -> None:
    settings = QuantizationSettings(
        weight_format="int4",
        symmetric=True,
        ratio=0.8,
        group_size=64,
        backup_precision="int8_asym",
        dataset="wikitext2",
        awq=True,
        scale_estimation=True,
        all_layers=True,
    )
    assert quantization_flags(settings) == [
        "--weight-format",
        "int4",
        "--sym",
        "--ratio",
        "0.8",
        "--group-size=64",
        "--backup-precision",
        "int8_asym",
        "--dataset",
        "wikitext2",
        "--all-layers",
        "--awq",
        "--scale-estimation",
    ]


def test_build_argv_puts_output_directory_last(tmp_path: Path) -> None:
    request = ExportRequest(
        source_path=tmp_path / "snapshot",
        output_path=tmp_path / "out",
        task="image-text-to-text",
        profile=make_profile("npu-int4-sym-cw"),
    )
    argv = build_export_argv(request, launcher=LAUNCHER)
    assert argv[:5] == [*LAUNCHER, "export", "openvino"]
    assert argv[5:9] == ["--model", str(tmp_path / "snapshot"), "--task", "image-text-to-text"]
    assert argv[9:] == [*PROFILE_FLAGS["npu-int4-sym-cw"], str(tmp_path / "out")]
    assert "--trust-remote-code" not in argv


def test_build_argv_uses_local_snapshot_path_not_a_repo_id(tmp_path: Path) -> None:
    request = ExportRequest(
        source_path=tmp_path / "snapshot",
        output_path=tmp_path / "out",
        profile=make_profile("gpu-int4-asym-g128"),
    )
    argv = build_export_argv(request, launcher=LAUNCHER)
    assert argv[argv.index("--model") + 1] == str(tmp_path / "snapshot")
    assert "--task" not in argv


def test_build_argv_honours_an_explicit_model_ref(tmp_path: Path) -> None:
    request = ExportRequest(
        source_path=tmp_path / "snapshot",
        output_path=tmp_path / "out",
        profile=make_profile("gpu-int4-asym-g128"),
        model_ref="/other/snapshot",
    )
    argv = build_export_argv(request, launcher=LAUNCHER)
    assert argv[argv.index("--model") + 1] == "/other/snapshot"


def test_trust_remote_code_only_when_explicitly_enabled(tmp_path: Path) -> None:
    profile = make_profile("gpu-int4-asym-g128")
    off = build_export_argv(
        ExportRequest(source_path=tmp_path / "s", output_path=tmp_path / "o", profile=profile),
        launcher=LAUNCHER,
    )
    on = build_export_argv(
        ExportRequest(
            source_path=tmp_path / "s",
            output_path=tmp_path / "o",
            profile=profile,
            trust_remote_code=True,
        ),
        launcher=LAUNCHER,
    )
    assert "--trust-remote-code" not in off
    assert "--trust-remote-code" in on
    # The flag must never end up after the positional output directory.
    assert on.index("--trust-remote-code") < len(on) - 1


def test_extra_args_sit_between_flags_and_the_output_directory(tmp_path: Path) -> None:
    profile = make_profile("gpu-int4-asym-g128").model_copy(
        update={"export_extra_args": ["--disable-stateful", "--num-samples", "32"]}
    )
    request = ExportRequest(source_path=tmp_path / "s", output_path=tmp_path / "o", profile=profile)
    argv = build_export_argv(request, launcher=LAUNCHER)
    assert argv[-4:] == ["--disable-stateful", "--num-samples", "32", str(tmp_path / "o")]


@pytest.mark.parametrize(
    "extra",
    [
        ["--trust-remote-code"],
        ["--trust-remote"],
        ["--model", "attacker/repo"],
        ["--model=attacker/repo"],
        ["-m", "attacker/repo"],
        ["-mattacker/repo"],
        ["--model-kwargs", '{"trust_remote_code": true}'],
        ["--", "/elsewhere"],
        ["--weight-format", "fp16"],
    ],
)
def test_reserved_extra_args_are_refused_when_the_argv_is_built(
    tmp_path: Path, extra: list[str]
) -> None:
    """A profile built in code never passes the registry's validator; the argv guard holds."""
    profile = make_profile("gpu-int4-asym-g128").model_copy(update={"export_extra_args": extra})
    request = ExportRequest(source_path=tmp_path / "s", output_path=tmp_path / "o", profile=profile)

    with pytest.raises(ConfigurationError) as caught:
        build_export_argv(request, launcher=LAUNCHER)
    assert caught.value.diagnostic.code == "export.reserved_extra_args"


def test_a_profile_with_reserved_extra_args_is_refused_without_running(job: JobContext) -> None:
    runner = FakeRunner()
    request = make_request(job, extra_args=["--trust-remote-code"])

    result = OptimumExporter(runner, launcher=LAUNCHER).export(request, job)

    assert runner.calls == []
    assert not result.success
    assert result.diagnostic is not None
    assert result.diagnostic.code == "export.reserved_extra_args"


# --- launcher discovery --------------------------------------------------


def test_resolve_launcher_prefers_the_module_in_this_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "yonga.adapters.exporter_optimum._module_importable",
        lambda name: name == OPTIMUM_CLI_MODULE,
    )
    assert resolve_launcher()[1:] == ["-P", "-m", OPTIMUM_CLI_MODULE]


def test_a_managed_interpreter_runs_without_the_current_directory_on_sys_path(
    tmp_path: Path,
) -> None:
    python = tmp_path / "env" / "bin" / "python"
    request = ExportRequest(
        source_path=tmp_path / "s",
        output_path=tmp_path / "o",
        profile=make_profile("gpu-int4-asym-g128"),
        interpreter=python,
    )
    assert OptimumExporter().resolve_launcher(request) == [
        str(python),
        "-P",
        "-m",
        OPTIMUM_CLI_MODULE,
    ]


def test_resolve_launcher_falls_back_to_the_console_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("yonga.adapters.exporter_optimum._module_importable", lambda name: False)
    monkeypatch.setattr(
        "yonga.adapters.exporter_optimum.shutil.which", lambda name: f"/usr/local/bin/{name}"
    )
    assert resolve_launcher() == ["/usr/local/bin/optimum-cli"]


def test_resolve_launcher_without_any_exporter_is_a_typed_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("yonga.adapters.exporter_optimum._module_importable", lambda name: False)
    monkeypatch.setattr("yonga.adapters.exporter_optimum.shutil.which", lambda name: None)
    with pytest.raises(ConfigurationError) as excinfo:
        resolve_launcher()
    diag = excinfo.value.diagnostic
    assert diag.code == "export.optimum_cli_missing"
    assert diag.category is FailureCategory.CONFIGURATION_ERROR
    assert any("yonga[export]" in line for line in diag.remediation)


def test_export_without_an_exporter_returns_a_result_instead_of_raising(
    job: JobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("yonga.adapters.exporter_optimum._module_importable", lambda name: False)
    monkeypatch.setattr("yonga.adapters.exporter_optimum.shutil.which", lambda name: None)
    runner = FakeRunner()
    result = OptimumExporter(runner).export(make_request(job), job)
    assert result.success is False
    assert result.diagnostic is not None
    assert result.diagnostic.code == "export.optimum_cli_missing"
    assert runner.calls == []


# --- export --------------------------------------------------------------


def test_export_runs_an_argv_list_and_creates_the_staging_output(job: JobContext) -> None:
    runner = FakeRunner(stdout=["Export finished"])
    request = make_request(job)
    result = OptimumExporter(runner, launcher=LAUNCHER).export(request, job)

    assert result.success is True
    assert result.returncode == 0
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call.argv[:3] == LAUNCHER
    assert call.argv[3:5] == ["export", "openvino"]
    assert call.argv[-1] == str(request.output_path)
    assert all(isinstance(item, str) for item in call.argv)
    assert request.output_path.is_dir()
    assert result.stdout_tail == ["Export finished"]
    assert result.log_path == job.logger.path


def test_export_points_temp_variables_at_the_managed_disk_backed_temp(job: JobContext) -> None:
    runner = FakeRunner()
    OptimumExporter(runner, launcher=LAUNCHER).export(make_request(job), job)
    env = runner.calls[0].env
    for key in ("TMPDIR", "TMP", "TEMP"):
        assert env[key] == str(job.temp_dir)


def test_extra_env_cannot_redirect_the_managed_temp(job: JobContext) -> None:
    runner = FakeRunner()
    request = make_request(job, extra_env={"TMPDIR": "/tmp", "OV_CACHE": "on"})
    result = OptimumExporter(runner, launcher=LAUNCHER).export(request, job)
    assert runner.calls[0].env["TMPDIR"] == str(job.temp_dir)
    assert runner.calls[0].env["OV_CACHE"] == "on"
    assert result.env_overrides["TMPDIR"] == str(job.temp_dir)
    assert result.env_overrides["OV_CACHE"] == "on"


def test_env_overrides_record_only_the_differences(job: JobContext) -> None:
    runner = FakeRunner()
    result = OptimumExporter(runner, launcher=LAUNCHER).export(make_request(job), job)
    assert set(result.env_overrides) <= {"TMPDIR", "TMP", "TEMP", "HF_HUB_OFFLINE"}


def test_the_runner_is_never_handed_the_whole_environment(job: JobContext) -> None:
    """What reaches ``run(env=...)`` is recorded in the job log, so it must be small.

    The exporter used to pass ``ctx.child_env()`` — a complete environment —
    which the runner wrote into ``subprocess.start`` verbatim. The child still
    sees the inherited environment, because the runner merges the overrides onto
    ``ctx.child_env()`` itself.
    """
    runner = FakeRunner()
    request = make_request(job, extra_env={"OV_CACHE": "on"})
    OptimumExporter(runner, launcher=LAUNCHER).export(request, job)
    passed = dict(runner.calls[0].env)
    assert set(passed) <= {"TMPDIR", "TMP", "TEMP", "HF_HUB_OFFLINE", "OV_CACHE"}
    assert "PATH" not in passed
    assert "HOME" not in passed
    assert passed["TMPDIR"] == str(job.temp_dir)


def test_offline_setting_reaches_the_subprocess(tmp_path: Path) -> None:
    settings = load_settings(
        env={},
        overrides={
            "data_dir": str(tmp_path / "data"),
            "cache_dir": str(tmp_path / "cache"),
            "config_dir": str(tmp_path / "config"),
            "offline": True,
        },
    )
    runner = FakeRunner()
    with open_job(settings=settings, requested_model="acme/model", job_id="offline-job") as ctx:
        OptimumExporter(runner, launcher=LAUNCHER).export(make_request(ctx), ctx)
    assert runner.calls[0].env["HF_HUB_OFFLINE"] == "1"


def test_export_records_package_versions_from_the_real_environment(job: JobContext) -> None:
    result = OptimumExporter(FakeRunner(), launcher=LAUNCHER).export(make_request(job), job)
    # This host has the full stack installed; at minimum the tool's own deps resolve.
    assert set(result.package_versions) <= set(EXPORT_PACKAGES)
    for version in result.package_versions.values():
        assert version


def test_export_refuses_an_output_directory_outside_staging(
    job: JobContext, tmp_path: Path
) -> None:
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "keep.txt").write_text("do not delete", encoding="utf-8")
    runner = FakeRunner()

    result = OptimumExporter(runner, launcher=LAUNCHER).export(
        make_request(job, output_path=outside), job
    )

    assert result.success is False
    assert result.diagnostic is not None
    assert result.diagnostic.code == "export.output_not_staged"
    assert result.diagnostic.category is FailureCategory.CONFIGURATION_ERROR
    assert runner.calls == []
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "do not delete"


def test_export_clears_only_tool_owned_staging_leftovers(job: JobContext) -> None:
    output = job.staging_dir / "model"
    output.mkdir(parents=True)
    (output / "stale.xml").write_text("partial", encoding="utf-8")
    (output / "nested").mkdir()
    sibling = job.staging_dir / "keep-me.txt"
    sibling.write_text("sibling", encoding="utf-8")

    OptimumExporter(FakeRunner(), launcher=LAUNCHER).export(
        make_request(job, output_path=output), job
    )

    assert list(output.iterdir()) == []
    assert sibling.read_text(encoding="utf-8") == "sibling"


def job_events(job: JobContext, event: str) -> list[dict[str, object]]:
    """The records of ``event`` in this job's log so far."""
    lines = job.logger.path.read_text(encoding="utf-8").splitlines()
    return [entry for entry in map(json.loads, lines) if entry["event"] == event]


def test_a_fresh_staging_directory_is_not_reported_as_leftovers(job: JobContext) -> None:
    """The caller's incomplete marker is not output from an earlier attempt."""
    output = job.staging_dir / "model"
    mark_incomplete(output, job_id=job.job_id)

    OptimumExporter(FakeRunner(), launcher=LAUNCHER).export(
        make_request(job, output_path=output), job
    )

    assert job_events(job, "export.output_reset") == []
    assert (output / INCOMPLETE_MARKER_NAME).exists()


def test_real_leftovers_are_reported_and_cleared_but_the_marker_stays(job: JobContext) -> None:
    output = job.staging_dir / "model"
    mark_incomplete(output, job_id=job.job_id)
    (output / "stale.xml").write_text("partial", encoding="utf-8")

    OptimumExporter(FakeRunner(), launcher=LAUNCHER).export(
        make_request(job, output_path=output), job
    )

    [reset] = job_events(job, "export.output_reset")
    assert reset["entries"] == 1
    assert [path.name for path in output.iterdir()] == [INCOMPLETE_MARKER_NAME]


def test_export_refuses_when_the_output_path_is_a_file(job: JobContext) -> None:
    job.staging_dir.mkdir(parents=True, exist_ok=True)
    target = job.staging_dir / "model"
    target.write_text("not a directory", encoding="utf-8")
    result = OptimumExporter(FakeRunner(), launcher=LAUNCHER).export(
        make_request(job, output_path=target), job
    )
    assert result.success is False
    assert result.diagnostic is not None
    assert result.diagnostic.code == "export.output_not_a_directory"
    assert target.read_text(encoding="utf-8") == "not a directory"


def test_nonzero_exit_attaches_a_default_diagnostic_and_never_raises(job: JobContext) -> None:
    runner = FakeRunner(
        returncode=1,
        stderr=["Traceback (most recent call last):", "ValueError: unsupported task"],
    )
    result = OptimumExporter(runner, launcher=LAUNCHER).export(make_request(job), job)

    assert result.success is False
    assert result.returncode == 1
    assert result.diagnostic is not None
    assert result.diagnostic.code == "export.failed"
    assert result.diagnostic.category is FailureCategory.UNKNOWN_RUNTIME_ERROR
    assert result.diagnostic.job_id == job.job_id
    assert result.diagnostic.raw_excerpt is not None
    assert "unsupported task" in result.diagnostic.raw_excerpt
    assert result.stderr_tail[-1].endswith("unsupported task")


@pytest.mark.parametrize(
    "line",
    [
        "OSError: [Errno 28] No space left on device",
        "RuntimeError: basic_ios::clear: iostream error",
    ],
)
def test_disk_exhaustion_is_classified_and_explained(job: JobContext, line: str) -> None:
    runner = FakeRunner(returncode=1, stderr=[line])
    result = OptimumExporter(runner, launcher=LAUNCHER).export(make_request(job), job)

    assert result.diagnostic is not None
    assert result.diagnostic.code == "export.insufficient_disk"
    assert result.diagnostic.category is FailureCategory.INSUFFICIENT_DISK
    assert "tmpfs" in result.diagnostic.message
    assert str(job.temp_dir) in result.diagnostic.message
    assert any("Do not enlarge /tmp" in item for item in result.diagnostic.remediation)


def test_is_disk_exhaustion_helper() -> None:
    assert is_disk_exhaustion("No space left on device") is True
    assert is_disk_exhaustion("basic_ios::clear: iostream error") is True
    assert is_disk_exhaustion("to_shape was called on a dynamic shape") is False


def test_cancellation_is_reported_as_user_cancelled(job: JobContext) -> None:
    runner = FakeRunner(returncode=-2, cancelled=True)
    result = OptimumExporter(runner, launcher=LAUNCHER).export(make_request(job), job)
    assert result.success is False
    assert result.diagnostic is not None
    assert result.diagnostic.code == "export.cancelled"
    assert result.diagnostic.category is FailureCategory.USER_CANCELLED


def test_a_cancelled_conversion_is_not_announced_as_failed(job: JobContext) -> None:
    """The live line read "Conversion failed" above a "Conversion cancelled" result."""
    sink = CollectingSink()
    job.progress = sink
    runner = FakeRunner(returncode=-2, cancelled=True)

    OptimumExporter(runner, launcher=LAUNCHER).export(make_request(job), job)

    last = sink.of_kind(ProgressKind.EXPORT)[-1]
    assert last.message == "Conversion cancelled"
    assert last.status == "cancelled"


def test_a_failed_conversion_is_still_announced_as_failed(job: JobContext) -> None:
    sink = CollectingSink()
    job.progress = sink

    OptimumExporter(FakeRunner(returncode=1), launcher=LAUNCHER).export(make_request(job), job)

    last = sink.of_kind(ProgressKind.EXPORT)[-1]
    assert (last.message, last.status) == ("Conversion failed", "failed")


def test_an_already_cancelled_job_never_starts_the_exporter(job: JobContext) -> None:
    job.cancelled.set()
    runner = FakeRunner()
    result = OptimumExporter(runner, launcher=LAUNCHER).export(make_request(job), job)
    assert runner.calls == []
    assert result.diagnostic is not None
    assert result.diagnostic.code == "export.cancelled"


def test_secrets_are_redacted_in_argv_and_captured_output(job: JobContext) -> None:
    token = "hf_" + "Z9y8X7w6V5u4T3s2R1q0P9o8N7m6L5k4J3"
    runner = FakeRunner(
        returncode=1,
        stdout=[f"Authorization: Bearer {token}"],
        stderr=[f"failed using {token}"],
    )
    # Any argument will do: redaction is by pattern, and --token itself is reserved.
    request = make_request(job, extra_args=["--pad-token-id", token])
    result = OptimumExporter(runner, launcher=LAUNCHER).export(request, job)

    assert token not in " ".join(result.argv)
    assert REDACTED in result.argv
    assert token not in " ".join(result.stdout_tail)
    assert token not in " ".join(result.stderr_tail)
    assert result.diagnostic is not None
    assert token not in (result.diagnostic.raw_excerpt or "")


def test_output_tails_are_bounded(job: JobContext) -> None:
    runner = FakeRunner(stdout=[f"line {index}" for index in range(500)])
    result = OptimumExporter(runner, launcher=LAUNCHER).export(make_request(job), job)
    assert len(result.stdout_tail) == 40
    assert result.stdout_tail[-1] == "line 499"


# --- requirements --------------------------------------------------------


@requires_export_extra
def test_requirements_returns_every_export_package() -> None:
    versions = OptimumExporter(FakeRunner(), launcher=LAUNCHER).requirements()
    assert set(versions) == set(EXPORT_PACKAGES)
    # This reference host has the full Intel stack installed.
    assert versions["openvino"] is not None
    assert versions["transformers"] is not None


@requires_export_extra
def test_check_requirements_flags_a_mismatch_against_the_really_installed_version() -> None:
    """A pin the installed transformers cannot satisfy is reported, naming both versions.

    The specifier is derived from whatever is installed rather than hard-coded, so
    this exercises the real ``requirements()`` reading without asserting anything
    about which transformers release happens to be on the host. Pinning a literal
    here would make the test fail on exactly the version the Qwen3.5 rule asks for.
    """
    exporter = OptimumExporter(FakeRunner(), launcher=LAUNCHER)
    installed = exporter.requirements()["transformers"]
    assert installed is not None

    major = int(installed.split(".")[0])
    impossible = f"=={major + 1}.0.*"

    checked = exporter.check_requirements(ExporterRequirements(transformers=impossible))

    assert checked.satisfied is False
    assert len(checked.problems) == 1
    problem = checked.problems[0]
    assert impossible in problem
    assert installed in problem
    assert "transformers" in problem


def test_check_requirements_passes_when_every_specifier_matches() -> None:
    checked = check_exporter_requirements(
        ExporterRequirements(transformers="==5.2.*", openvino=">=2026.2"),
        versions={"transformers": "5.2.1", "openvino": "2026.4.0-22959-99c81491cc3"},
    )
    assert checked.satisfied is True
    assert checked.problems == []


def test_check_requirements_reports_a_missing_package() -> None:
    checked = check_exporter_requirements(
        ExporterRequirements(nncf=">=2.19"),
        versions={"nncf": None},
    )
    assert checked.satisfied is False
    assert "not installed" in checked.problems[0]
    assert ">=2.19" in checked.problems[0]


def test_check_requirements_does_not_install_anything() -> None:
    """MVP policy: diagnose, name the required version, change nothing."""
    checked = check_exporter_requirements(
        ExporterRequirements(transformers="==5.2.*"),
        versions={"transformers": "5.5.4"},
    )
    assert "5.5.4" in checked.problems[0]
    assert "pip install" in checked.problems[0]


def test_check_requirements_keeps_pre_existing_problems() -> None:
    checked = check_exporter_requirements(
        ExporterRequirements(problems=["registry said so"]),
        versions={},
    )
    assert checked.problems == ["registry said so"]
    assert checked.satisfied is False


# --- protocol conformance ------------------------------------------------


def test_adapter_and_fake_satisfy_their_protocols() -> None:
    exporter = OptimumExporter(FakeRunner(), launcher=LAUNCHER)
    assert isinstance(exporter, Exporter)
    assert isinstance(FakeRunner(), CommandRunner)
