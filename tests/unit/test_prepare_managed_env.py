"""Preparation with a managed export environment instead of a refusal.

On the reference host Qwen3.5 exports only with ``transformers==5.2.*`` while
5.5.4 is installed. Before managed environments that stopped preparation at
preflight and left the user to reshape their own environment; now the export
runs in an isolated environment yonga builds -- confirmed in the plan, built
before the model download, and recorded in the manifest.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.unit.test_prepare_planning import (
    MISMATCHED_EXPORT_VERSIONS,
    SAFETENSORS_REPO,
    FakeExporter,
    FakeFilesystemProbe,
    FakeHub,
    build_service,
    make_snapshot,
    run_prepare,
    safetensors_metadata,
    settings_for,
)
from yonga.adapters.filesystem import system_temp_dir
from yonga.application.jobs import JobContext
from yonga.config.settings import Settings
from yonga.domain.diagnostics import FailureCategory, PreflightError, diagnostic
from yonga.domain.environments import EnvironmentSpec, ExportEnvironment
from yonga.domain.plans import CONFIRMATION_MANAGED_ENVIRONMENT, ExporterRequirements

ENV_INSTALLED = {"transformers": "5.2.0", "optimum-intel": "2.2.0", "openvino": "2026.4.0"}


class FakeEnvironments:
    """An ExportEnvironments double that never touches pip."""

    def __init__(
        self, root: Path, *, ready: bool = False, fail: PreflightError | None = None
    ) -> None:
        self.root = root
        self.ready = ready
        self.fail = fail
        self.ensured: list[EnvironmentSpec] = []

    def spec_for(
        self,
        requirements: ExporterRequirements,
        *,
        installed: object = None,
        reason: str = "",
    ) -> EnvironmentSpec:
        return EnvironmentSpec(
            python="3.14", packages={"transformers": requirements.transformers or ""}, reason=reason
        )

    def find(self, spec: EnvironmentSpec) -> ExportEnvironment | None:
        return self._environment(spec) if self.ready else None

    def ensure(self, spec: EnvironmentSpec, ctx: JobContext) -> ExportEnvironment:
        self.ensured.append(spec)
        if self.fail is not None:
            raise self.fail
        return self._environment(spec)

    def _environment(self, spec: EnvironmentSpec) -> ExportEnvironment:
        return ExportEnvironment(
            spec=spec,
            path=self.root / spec.name,
            installed=dict(ENV_INSTALLED),
            created_at=datetime(2026, 9, 24, tzinfo=UTC),
        )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return settings_for(tmp_path)


@pytest.fixture
def hub(tmp_path: Path) -> FakeHub:
    return FakeHub(
        repos={SAFETENSORS_REPO: safetensors_metadata()},
        snapshots={SAFETENSORS_REPO: make_snapshot(tmp_path, "snapshot-safetensors")},
    )


def mismatched() -> FakeExporter:
    return FakeExporter(versions=dict(MISMATCHED_EXPORT_VERSIONS))


def test_the_plan_chooses_an_environment_instead_of_a_block(
    settings: Settings, hub: FakeHub, tmp_path: Path
) -> None:
    envs = FakeEnvironments(tmp_path / "envs")
    service = build_service(settings=settings, hub=hub, exporter=mismatched(), environments=envs)

    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    assert plan.export_environment is not None
    assert plan.export_environment.packages["transformers"] == "==5.2.*"
    assert plan.exporter_requirements_block is False
    codes = [reason.code for reason in plan.confirmations]
    assert CONFIRMATION_MANAGED_ENVIRONMENT in codes
    managed = next(r for r in plan.confirmations if r.code == CONFIRMATION_MANAGED_ENVIRONMENT)
    assert managed.security_sensitive is False, "--yes may approve building an environment"
    assert any("managed environment" in note for note in plan.known_limitations)


def test_the_export_runs_in_the_environment_and_the_manifest_says_so(
    settings: Settings, hub: FakeHub, tmp_path: Path
) -> None:
    envs = FakeEnvironments(tmp_path / "envs")
    exporter = mismatched()
    service = build_service(settings=settings, hub=hub, exporter=exporter, environments=envs)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    outcome = run_prepare(service, plan, settings)

    (request,) = exporter.requests
    assert plan.export_environment is not None
    environment_python = tmp_path / "envs" / plan.export_environment.name / "bin" / "python"
    assert request.interpreter == environment_python
    assert request.environment_versions == ENV_INSTALLED
    assert outcome.manifest.export.environment_key == plan.export_environment.key


def test_an_existing_environment_needs_no_confirmation(
    settings: Settings, hub: FakeHub, tmp_path: Path
) -> None:
    envs = FakeEnvironments(tmp_path / "envs", ready=True)
    service = build_service(settings=settings, hub=hub, exporter=mismatched(), environments=envs)

    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    assert plan.export_environment_ready is True
    assert CONFIRMATION_MANAGED_ENVIRONMENT not in [reason.code for reason in plan.confirmations]


@pytest.mark.parametrize("how", ["flag", "setting", "no-manager"])
def test_refusing_managed_environments_restores_the_block(
    how: str, tmp_path: Path, hub: FakeHub
) -> None:
    settings = settings_for(tmp_path)
    if how == "setting":
        settings = settings.model_copy(update={"managed_export_envs": False})
    envs = None if how == "no-manager" else FakeEnvironments(tmp_path / "envs")
    exporter = mismatched()
    service = build_service(settings=settings, hub=hub, exporter=exporter, environments=envs)

    plan = service.plan(SAFETENSORS_REPO, device="gpu", managed_env=how != "flag")

    assert plan.export_environment is None
    with pytest.raises(PreflightError) as caught:
        run_prepare(service, plan, settings)
    assert caught.value.diagnostic.code == "preflight.exporter_requirement_unsatisfied"
    assert exporter.requests == []


def test_a_failed_build_stops_before_the_model_download(
    settings: Settings, hub: FakeHub, tmp_path: Path
) -> None:
    failure = PreflightError(
        diagnostic(
            "export.env.install_failed",
            FailureCategory.DOWNLOAD_FAILURE,
            "The export environment could not be built",
            "pip failed",
        )
    )
    envs = FakeEnvironments(tmp_path / "envs", fail=failure)
    service = build_service(settings=settings, hub=hub, exporter=mismatched(), environments=envs)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    with pytest.raises(PreflightError) as caught:
        run_prepare(service, plan, settings)

    assert caught.value.diagnostic.code == "export.env.install_failed"
    assert hub.downloaded == [], "the model must not be downloaded for a job that cannot export"


def test_satisfied_requirements_never_build_an_environment(
    settings: Settings, hub: FakeHub, tmp_path: Path
) -> None:
    envs = FakeEnvironments(tmp_path / "envs")
    exporter = FakeExporter()  # transformers 5.2.1, which meets ==5.2.*
    service = build_service(settings=settings, hub=hub, exporter=exporter, environments=envs)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    run_prepare(service, plan, settings)

    assert plan.export_environment is None
    assert envs.ensured == []
    assert exporter.requests[0].interpreter is None


def test_a_new_environment_must_fit_alongside_the_exports_temp_space(
    settings: Settings, hub: FakeHub, tmp_path: Path
) -> None:
    """Temp alone fits, temp plus the environment does not: refused, before building.

    The managed temp directory and the environment share the cache filesystem,
    so passing the temp check on its own is not enough.
    """
    envs = FakeEnvironments(tmp_path / "envs")
    probe = FakeFilesystemProbe(
        tmpfs_paths=(system_temp_dir(),), small_roots=(settings.paths.cache_dir,)
    )
    service = build_service(
        settings=settings, hub=hub, exporter=mismatched(), environments=envs, filesystem=probe
    )
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    probe.small_free = plan.disk.temp.total_required_bytes + 2**30  # temp fits, +3 GiB does not
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    assert plan.disk.temp.sufficient

    with pytest.raises(PreflightError) as caught:
        run_prepare(service, plan, settings)

    assert caught.value.diagnostic.code == "export.env.insufficient_disk"
    assert "alongside the export's temporary files" in caught.value.diagnostic.message
    assert envs.ensured == []
    assert hub.downloaded == []
