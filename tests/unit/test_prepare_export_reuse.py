"""Reusing a completed export that a failed preparation left behind.

The export is the expensive half of a preparation, and upstream cannot resume
one that was interrupted. What yonga can do is not throw away one that
*finished*: when a later stage fails (a device probe, a validation defect,
Ctrl-C during the smoke test), an identical retry adopts the retained export
-- but only after proving every file is what the exporter wrote.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.test_prepare_planning import (
    SAFETENSORS_REPO,
    FakeExporter,
    FakeHub,
    FakeRuntimeProbe,
    build_service,
    make_snapshot,
    run_prepare,
    safetensors_metadata,
    settings_for,
)
from yonga.application import export_reuse
from yonga.application.prepare import PrepareService
from yonga.config.settings import Settings
from yonga.domain.artifacts import MANIFEST_DIRNAME
from yonga.domain.diagnostics import YongaError

GPU_FAILURE = "Exception from src/plugins/intel_gpu: out of device memory"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return settings_for(tmp_path)


@pytest.fixture
def hub(tmp_path: Path) -> FakeHub:
    return FakeHub(
        repos={SAFETENSORS_REPO: safetensors_metadata()},
        snapshots={SAFETENSORS_REPO: make_snapshot(tmp_path, "snapshot-safetensors")},
    )


def fail_after_export(
    settings: Settings, hub: FakeHub, *, broken_tokenizer: bool = True
) -> tuple[str, Path]:
    """Run a preparation whose export succeeds and whose GPU probe fails.

    Returns the failed job's id and its retained staging directory. The fake
    export ships the ``STRiNG`` tokenizer defect, so by the time the probe fails
    the repair has already rewritten the tokenizer files.
    """
    runtime = FakeRuntimeProbe(compile_failures={"GPU": GPU_FAILURE})
    exporter = FakeExporter(broken_tokenizer=broken_tokenizer)
    service = build_service(settings=settings, hub=hub, runtime=runtime, exporter=exporter)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    with pytest.raises(YongaError):
        run_prepare(service, plan, settings)
    staging = settings.paths.staging_dir / plan.job_id / "model"
    assert export_reuse.checkpoint_path(staging).is_file(), "the export had completed"
    assert (staging / MANIFEST_DIRNAME / "repairs").is_dir() is broken_tokenizer
    return plan.job_id, staging


def healthy(settings: Settings, hub: FakeHub) -> tuple[PrepareService, FakeExporter]:
    exporter = FakeExporter()
    return build_service(settings=settings, hub=hub, exporter=exporter), exporter


def test_a_retry_adopts_the_completed_export_instead_of_exporting(
    settings: Settings, hub: FakeHub
) -> None:
    failed_job, retained = fail_after_export(settings, hub)
    downloads_before = len(hub.downloaded)
    service, exporter = healthy(settings, hub)

    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    assert plan.reusable_export is not None
    assert plan.reusable_export.job_id == failed_job

    outcome = run_prepare(service, plan, settings)

    assert exporter.requests == [], "the conversion must not run again"
    assert len(hub.downloaded) == downloads_before, "nor the download"
    assert outcome.manifest.export.reused_from_job == failed_job
    assert any(f"reused from failed job {failed_job}" in note for note in outcome.manifest.notes)
    # The restored originals were repaired again, by this job, and recorded.
    assert [record.repair_id for record in outcome.repairs] != []
    assert outcome.validation is not None
    assert outcome.validation.ok
    assert not retained.parent.exists(), "the failed job's staging husk is gone"


def test_fresh_export_declines_the_reuse(settings: Settings, hub: FakeHub) -> None:
    fail_after_export(settings, hub)
    service, exporter = healthy(settings, hub)

    plan = service.plan(SAFETENSORS_REPO, device="gpu", reuse_export=False)
    run_prepare(service, plan, settings)

    assert plan.reusable_export is None
    assert len(exporter.requests) == 1


def test_different_inputs_never_reuse(settings: Settings, hub: FakeHub) -> None:
    """Another profile is another export, even for the same source revision."""
    fail_after_export(settings, hub)
    service, _ = healthy(settings, hub)

    plan = service.plan(SAFETENSORS_REPO, device="cpu")

    assert plan.reusable_export is None


def test_different_exporter_packages_never_reuse(settings: Settings, hub: FakeHub) -> None:
    fail_after_export(settings, hub)
    upgraded = FakeExporter()
    upgraded.versions["nncf"] = "9.9.9"
    service = build_service(settings=settings, hub=hub, exporter=upgraded)

    assert service.plan(SAFETENSORS_REPO, device="gpu").reusable_export is None


@pytest.mark.parametrize("hashed", [True, False], ids=["by-digest", "by-size-and-mtime"])
def test_a_weight_file_changed_after_the_export_is_not_trusted(
    settings: Settings, hub: FakeHub, monkeypatch: pytest.MonkeyPatch, hashed: bool
) -> None:
    """Same size, different bytes: the planning check passes, adoption must not.

    Real weight files are far too large to hash, so they are identified by size
    and modification time; ``hashed=False`` makes the fixture's tiny ones behave
    the same way.
    """
    if not hashed:
        monkeypatch.setattr(export_reuse, "HASHED_MAX_BYTES", 0)
    # Without the tokenizer defect no repair rewrites anything, so only the
    # weights can be what gives the export away.
    _, retained = fail_after_export(settings, hub, broken_tokenizer=False)
    weights = retained / "openvino_language_model.bin"
    weights.write_bytes(b"\x01" * weights.stat().st_size)
    service, exporter = healthy(settings, hub)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    assert plan.reusable_export is not None

    outcome = run_prepare(service, plan, settings)

    assert len(exporter.requests) == 1, "the export ran instead"
    assert outcome.manifest.export.reused_from_job is None


def test_unhashed_files_that_did_not_change_are_trusted(
    settings: Settings, hub: FakeHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control for the case above: size and mtime alone do admit an intact file."""
    monkeypatch.setattr(export_reuse, "HASHED_MAX_BYTES", 0)
    failed_job, _ = fail_after_export(settings, hub, broken_tokenizer=False)
    service, exporter = healthy(settings, hub)

    outcome = run_prepare(service, service.plan(SAFETENSORS_REPO, device="gpu"), settings)

    assert exporter.requests == []
    assert outcome.manifest.export.reused_from_job == failed_job


def test_a_rewritten_file_without_a_backup_is_not_trusted(settings: Settings, hub: FakeHub) -> None:
    _, retained = fail_after_export(settings, hub)
    config = retained / "config.json"
    config.write_text('{"model_type": "something_else"}\n', encoding="utf-8")
    service, exporter = healthy(settings, hub)

    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    outcome = run_prepare(service, plan, settings)

    assert len(exporter.requests) == 1
    assert outcome.manifest.export.reused_from_job is None


def test_a_failed_export_leaves_nothing_to_reuse(settings: Settings, hub: FakeHub) -> None:
    exporter = FakeExporter(succeed=False, stderr_tail=["RuntimeError: boom"])
    service = build_service(settings=settings, hub=hub, exporter=exporter)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    with pytest.raises(YongaError):
        run_prepare(service, plan, settings)

    retry, _ = healthy(settings, hub)

    assert retry.plan(SAFETENSORS_REPO, device="gpu").reusable_export is None


def test_the_checkpoint_hashes_small_files_and_skips_its_own_metadata(
    settings: Settings, hub: FakeHub
) -> None:
    _, retained = fail_after_export(settings, hub)
    checkpoint = export_reuse._read_checkpoint(retained)

    assert checkpoint is not None
    assert "config.json" in checkpoint.files
    assert not any(
        name.startswith((MANIFEST_DIRNAME, ".yonga-incomplete")) for name in checkpoint.files
    )
    assert all(entry.sha256 is not None for entry in checkpoint.files.values())


def test_a_checkpoint_naming_a_path_outside_staging_is_ignored(
    settings: Settings, hub: FakeHub
) -> None:
    _, retained = fail_after_export(settings, hub)
    target = export_reuse.checkpoint_path(retained)
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["files"]["../../outside.txt"] = {"size": 0, "mtime_ns": 0, "sha256": None}
    target.write_text(json.dumps(payload), encoding="utf-8")
    service, _ = healthy(settings, hub)

    assert service.plan(SAFETENSORS_REPO, device="gpu").reusable_export is None
