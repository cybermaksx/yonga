"""Managed export environments: specified minimally, built safely, never half-used."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pytest

from yonga.adapters.export_env import MARKER_FILENAME, ExportEnvironmentManager
from yonga.adapters.subprocess import CompletedCommand
from yonga.application.jobs import JobContext
from yonga.config.paths import Paths, default_paths
from yonga.config.settings import Settings
from yonga.domain.diagnostics import ConfigurationError, FailureCategory, PreflightError
from yonga.domain.environments import TORCH_CPU_INDEX, EnvironmentSpec
from yonga.domain.plans import ExporterRequirements
from yonga.util.jsonlog import null_logger

#: What yonga's own environment holds on the reference host, less the pin.
INSTALLED: dict[str, str | None] = {
    "torch": "2.14.0",
    "torchvision": "0.29.0",
    "transformers": "5.5.4",
    "optimum": "2.3.0",
    "optimum-intel": "2.2.0",
    "nncf": "3.4.0",
    "openvino": "2026.4.0",
    "openvino-tokenizers": "2026.4.0.0",
}
QWEN35 = ExporterRequirements(transformers="==5.2.*")


class FakeRunner:
    """Answers the three build steps and the verification script."""

    def __init__(
        self,
        *,
        fail_step: str | None = None,
        cancel_step: str | None = None,
        versions: Mapping[str, str | None] | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self.fail_step = fail_step
        self.cancel_step = cancel_step
        self.versions = dict(versions) if versions is not None else None

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float | None = None,
        ctx: JobContext | None = None,
    ) -> CompletedCommand:
        command = list(argv)
        assert all(isinstance(part, str) for part in command), "argv must be a list of str"
        self.calls.append(command)
        step = self._step(command)
        if step == self.cancel_step:
            return CompletedCommand(
                argv=command, returncode=-2, duration_seconds=0.0, cancelled=True
            )
        if step == self.fail_step:
            return CompletedCommand(
                argv=command, returncode=1, duration_seconds=0.0, stderr_lines=["ERROR: no network"]
            )
        if step == "venv":
            python = Path(command[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("#!/bin/sh\n")
        if step == "verify":
            reported = self.versions if self.versions is not None else self._pinned()
            return CompletedCommand(
                argv=command,
                returncode=0,
                duration_seconds=0.0,
                stdout_lines=[json.dumps(reported)],
            )
        return CompletedCommand(argv=command, returncode=0, duration_seconds=0.0)

    @staticmethod
    def _step(command: list[str]) -> str:
        if command[1:3] == ["-m", "venv"]:
            return "venv"
        if "--index-url" in command:
            return "torch"
        if "pip" in command:
            return "packages"
        return "verify"

    def _pinned(self) -> dict[str, str]:
        """Versions a correct build would report: the mirror, with transformers 5.2.0."""
        return {**{k: v for k, v in INSTALLED.items() if v}, "transformers": "5.2.0"}


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    built = default_paths(
        data_dir=tmp_path / "data", cache_dir=tmp_path / "cache", config_dir=tmp_path / "config"
    )
    built.ensure()
    return built


@pytest.fixture
def ctx(paths: Paths) -> Iterator[JobContext]:
    logger = null_logger()
    yield JobContext(job_id="job-1", logger=logger, paths=paths, settings=Settings())
    logger.close()


def manager(paths: Paths, runner: FakeRunner, *, offline: bool = False) -> ExportEnvironmentManager:
    return ExportEnvironmentManager(
        paths, runner, offline=offline, base_python=Path("/usr/bin/python3")
    )


# --- specification --------------------------------------------------------------------


def test_the_spec_mirrors_what_is_installed_and_changes_only_the_failing_pin(
    paths: Paths,
) -> None:
    spec = manager(paths, FakeRunner()).spec_for(QWEN35, installed=INSTALLED)

    assert spec.packages["transformers"] == "==5.2.*"
    assert spec.packages["torch"] == "==2.14.0"
    assert spec.packages["optimum-intel"] == "==2.2.0"
    assert spec.packages["nncf"] == "==3.4.0"
    # The runtime must read the IR the export writes: same OpenVINO release.
    assert spec.packages["openvino"] == "==2026.4.0"
    assert spec.packages["openvino-tokenizers"] == "==2026.4.0.0"


def test_a_missing_package_takes_the_declared_specifier_or_the_latest(paths: Paths) -> None:
    installed = {**INSTALLED, "optimum-intel": None, "nncf": None}
    requirements = ExporterRequirements(transformers="==5.2.*", optimum_intel=">=2.1")

    spec = manager(paths, FakeRunner()).spec_for(requirements, installed=installed)

    assert spec.packages["optimum-intel"] == ">=2.1"
    assert spec.packages["nncf"] == ""
    assert spec.requirement("nncf") == "nncf"


def test_the_key_identifies_the_package_set() -> None:
    one = EnvironmentSpec(python="3.14", packages={"transformers": "==5.2.*", "torch": "==2.14.0"})
    same = EnvironmentSpec(python="3.14", packages={"torch": "==2.14.0", "transformers": "==5.2.*"})
    other = EnvironmentSpec(
        python="3.14", packages={"transformers": "==5.3.*", "torch": "==2.14.0"}
    )

    assert one.key == same.key
    assert one.key != other.key
    assert EnvironmentSpec(python="3.13", packages=one.packages).key != one.key


def test_registry_data_cannot_name_a_package_of_its_own() -> None:
    """Data may choose a version for a known package, never a package to install."""
    with pytest.raises(ValueError, match="not a package"):
        EnvironmentSpec(python="3.14", packages={"evil-package": "==1.0"})
    with pytest.raises(ValueError, match="invalid specifier"):
        EnvironmentSpec(python="3.14", packages={"transformers": "; rm -rf /"})


# --- building -------------------------------------------------------------------------


def test_a_build_runs_venv_then_cpu_torch_then_the_pins_then_verifies(
    paths: Paths, ctx: JobContext
) -> None:
    runner = FakeRunner()
    envs = manager(paths, runner)
    spec = envs.spec_for(QWEN35, installed=INSTALLED)

    environment = envs.ensure(spec, ctx)

    venv, torch, packages, verify = runner.calls
    assert venv[:3] == ["/usr/bin/python3", "-m", "venv"]
    python = str(environment.python)
    assert torch[:4] == [python, "-m", "pip", "install"]
    assert torch[torch.index("--index-url") + 1] == TORCH_CPU_INDEX
    assert torch[-2:] == ["torch==2.14.0", "torchvision==0.29.0"]
    assert "transformers==5.2.*" in packages
    # Built against torch: a PyPI torchvision is the CUDA build and breaks the CPU torch.
    assert not any(part.startswith(("torch==", "torchvision")) for part in packages)
    assert verify[:2] == [python, "-c"]
    assert environment.installed["transformers"] == "5.2.0"
    assert (environment.path / MARKER_FILENAME).is_file()


def test_a_built_environment_is_reused_without_running_anything(
    paths: Paths, ctx: JobContext
) -> None:
    runner = FakeRunner()
    envs = manager(paths, runner)
    spec = envs.spec_for(QWEN35, installed=INSTALLED)
    first = envs.ensure(spec, ctx)
    runner.calls.clear()

    second = envs.ensure(spec, ctx)

    assert runner.calls == []
    assert second.key == first.key


def test_an_interrupted_build_is_never_used_and_is_rebuilt(paths: Paths, ctx: JobContext) -> None:
    runner = FakeRunner()
    envs = manager(paths, runner)
    spec = envs.spec_for(QWEN35, installed=INSTALLED)
    leftover = envs.path_for(spec)
    (leftover / "bin").mkdir(parents=True)
    (leftover / "bin" / "python").write_text("#!/bin/sh\n")
    (leftover / "half-installed").write_text("x")

    assert envs.find(spec) is None, "a directory without the marker is not an environment"
    envs.ensure(spec, ctx)

    assert not (leftover / "half-installed").exists()
    assert runner.calls[0][1:3] == ["-m", "venv"]


def test_a_failed_install_is_typed_and_leaves_nothing_behind(paths: Paths, ctx: JobContext) -> None:
    envs = manager(paths, FakeRunner(fail_step="packages"))
    spec = envs.spec_for(QWEN35, installed=INSTALLED)

    with pytest.raises(PreflightError) as caught:
        envs.ensure(spec, ctx)

    assert caught.value.diagnostic.code == "export.env.install_failed"
    assert caught.value.diagnostic.category is FailureCategory.DOWNLOAD_FAILURE
    assert "no network" in (caught.value.diagnostic.raw_excerpt or "")
    assert not envs.path_for(spec).exists()


def test_a_build_that_does_not_match_its_spec_is_rejected(paths: Paths, ctx: JobContext) -> None:
    """pip resolving to the wrong release must not produce a trusted environment."""
    wrong = {**{k: v for k, v in INSTALLED.items() if v}}  # transformers still 5.5.4
    envs = manager(paths, FakeRunner(versions=wrong))
    spec = envs.spec_for(QWEN35, installed=INSTALLED)

    with pytest.raises(PreflightError) as caught:
        envs.ensure(spec, ctx)

    assert caught.value.diagnostic.code == "export.env.verify_failed"
    assert "transformers 5.5.4 does not satisfy ==5.2.*" in caught.value.diagnostic.message
    assert not envs.path_for(spec).exists()


def test_a_cancelled_build_is_reported_as_cancelled(paths: Paths, ctx: JobContext) -> None:
    envs = manager(paths, FakeRunner(cancel_step="torch"))

    with pytest.raises(PreflightError) as caught:
        envs.ensure(envs.spec_for(QWEN35, installed=INSTALLED), ctx)

    assert caught.value.diagnostic.category is FailureCategory.USER_CANCELLED


def test_offline_with_no_environment_refuses_before_running_anything(
    paths: Paths, ctx: JobContext
) -> None:
    runner = FakeRunner()
    envs = manager(paths, runner, offline=True)

    with pytest.raises(PreflightError) as caught:
        envs.ensure(envs.spec_for(QWEN35, installed=INSTALLED), ctx)

    assert caught.value.diagnostic.code == "export.env.offline"
    assert runner.calls == []


def test_offline_still_reuses_an_environment_that_exists(paths: Paths, ctx: JobContext) -> None:
    spec = manager(paths, FakeRunner()).spec_for(QWEN35, installed=INSTALLED)
    manager(paths, FakeRunner()).ensure(spec, ctx)

    assert manager(paths, FakeRunner(), offline=True).ensure(spec, ctx).key == spec.key


# --- listing and removal --------------------------------------------------------------


def test_only_completed_environments_are_listed(paths: Paths, ctx: JobContext) -> None:
    envs = manager(paths, FakeRunner())
    built = envs.ensure(envs.spec_for(QWEN35, installed=INSTALLED), ctx)
    (paths.envs_dir / "export-000000000000").mkdir()

    assert [environment.key for environment in envs.environments()] == [built.key]


def test_remove_deletes_by_prefix_and_refuses_unknown_keys(paths: Paths, ctx: JobContext) -> None:
    envs = manager(paths, FakeRunner())
    built = envs.ensure(envs.spec_for(QWEN35, installed=INSTALLED), ctx)

    with pytest.raises(ConfigurationError):
        envs.remove("ffff")
    removed = envs.remove(built.key[:6])

    assert removed.key == built.key
    assert not built.path.exists()
    assert not (paths.envs_dir / f"{built.spec.name}.lock").exists()
    assert envs.environments() == []
