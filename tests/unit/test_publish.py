"""``yonga publish``: SPEC section 21's requirements, each one pinned.

The artifact is produced by a real preparation against faked adapters, so the
manifest the card is generated from is exactly the shape production writes --
including a repair, because the fake export ships the tokenizer defect.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.unit.test_prepare_planning import (
    SAFETENSORS_REPO,
    FakeHub,
    build_service,
    make_snapshot,
    run_prepare,
    safetensors_metadata,
    settings_for,
)
from yonga.application.artifacts import ArtifactStore
from yonga.application.publish import PublishService
from yonga.application.report import anonymise
from yonga.config.settings import Settings
from yonga.domain.artifacts import (
    ArtifactManifest,
    DeviceProbeReport,
    ProbeStatus,
    manifest_path,
)
from yonga.domain.diagnostics import (
    ConfigurationError,
    FailureCategory,
    YongaError,
    diagnostic,
)
from yonga.domain.publication import LicenseClass, PublisherIdentity, classify_license
from yonga.registry.loader import load_builtin_registry
from yonga.registry.resolver import RegistryResolver

REPO = "someone/qwen35-openvino-int4"


@dataclass
class Commit:
    """One recorded upload."""

    repo_id: str
    private: bool
    files: dict[str, Path]
    generated: dict[str, str]
    message: str


class FakePublisher:
    """A HubPublisher that records instead of uploading."""

    def __init__(
        self,
        identity: PublisherIdentity | None = None,
        *,
        existing: bool = False,
    ) -> None:
        self._identity = identity
        self.existing = existing
        self.commits: list[Commit] = []

    def identity(self) -> PublisherIdentity | None:
        return self._identity

    def repo_exists(self, repo_id: str) -> bool:
        return self.existing

    def publish(
        self,
        repo_id: str,
        *,
        private: bool,
        files: Sequence[tuple[str, Path]],
        generated: Mapping[str, bytes],
        message: str,
    ) -> str:
        self.commits.append(
            Commit(
                repo_id=repo_id,
                private=private,
                files=dict(files),
                generated={k: v.decode() for k, v in generated.items()},
                message=message,
            )
        )
        return f"https://huggingface.co/{repo_id}/commit/abc123"


def someone(**overrides: object) -> PublisherIdentity:
    payload: dict[str, object] = {"name": "someone", "organizations": ["team"], "can_write": True}
    payload.update(overrides)
    return PublisherIdentity.model_validate(payload)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return settings_for(tmp_path)


@pytest.fixture
def hub(tmp_path: Path) -> FakeHub:
    snapshot = make_snapshot(tmp_path, "snapshot-safetensors")
    (snapshot / "LICENSE").write_text("Apache License 2.0\n", encoding="utf-8")
    return FakeHub(
        repos={SAFETENSORS_REPO: safetensors_metadata()},
        snapshots={SAFETENSORS_REPO: snapshot},
    )


@pytest.fixture
def artifact(settings: Settings, hub: FakeHub) -> ArtifactManifest:
    service = build_service(settings=settings, hub=hub)
    outcome = run_prepare(service, service.plan(SAFETENSORS_REPO, device="gpu"), settings)
    return outcome.manifest


def publisher_service(
    settings: Settings,
    hub: FakeHub,
    publisher: FakePublisher,
    tmp_path: Path,
) -> PublishService:
    return PublishService(
        store=ArtifactStore(settings.paths, settings),
        hub=hub,
        publisher=publisher,
        resolver=RegistryResolver(load_builtin_registry()),
        settings=settings,
        anonymiser=lambda text: anonymise(text, home=str(tmp_path), user=None, host=None),
    )


def rewrite_manifest(settings: Settings, manifest: ArtifactManifest, **changes: object) -> None:
    store = ArtifactStore(settings.paths, settings)
    directory = store.find(manifest.artifact_id)
    assert directory is not None
    updated = manifest.model_copy(update=changes)
    manifest_path(directory).write_text(updated.to_json_text(), encoding="utf-8")


# --- the licence policy ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("apache-2.0", LicenseClass.PERMISSIVE),
        ("MIT", LicenseClass.PERMISSIVE),
        ("llama3.1", LicenseClass.CONDITIONAL),
        ("cc-by-nc-4.0", LicenseClass.CONDITIONAL),
        ("cc-by-nd-4.0", LicenseClass.RESTRICTED),
        ("other", LicenseClass.UNKNOWN),
        (None, LicenseClass.UNKNOWN),
        ("my-custom-license", LicenseClass.UNKNOWN),
    ],
)
def test_licenses_are_classified_conservatively(name: str | None, expected: LicenseClass) -> None:
    assert classify_license(name) is expected


# --- a successful publication ---------------------------------------------------------


def test_the_plan_uploads_the_model_files_the_card_and_the_license(
    settings: Settings, hub: FakeHub, artifact: ArtifactManifest, tmp_path: Path
) -> None:
    publisher = FakePublisher(someone())
    service = publisher_service(settings, hub, publisher, tmp_path)

    plan = service.plan(artifact.artifact_id, REPO)
    url = service.publish(plan)

    (commit,) = publisher.commits
    uploaded = set(commit.files) | set(commit.generated)
    assert {record.path for record in artifact.files} <= uploaded
    assert {"README.md", "yonga-manifest.json", "LICENSE"} <= uploaded
    assert not any(name.startswith(".yonga") for name in uploaded), "tool metadata stays local"
    assert commit.private is True, "private unless --public"
    assert url.endswith("/commit/abc123")


def test_the_card_tells_the_truth_about_what_this_is(
    settings: Settings, hub: FakeHub, artifact: ArtifactManifest, tmp_path: Path
) -> None:
    service = publisher_service(settings, hub, FakePublisher(someone()), tmp_path)
    card = service.plan(artifact.artifact_id, REPO).generated["README.md"]

    assert "base_model_relation: quantized" in card
    assert f"base_model: {SAFETENSORS_REPO}" in card
    assert "not a fine-tune" in card
    assert "No weights were trained" in card
    assert "license: apache-2.0" in card
    assert artifact.resolved_source.resolved_revision is not None
    assert artifact.resolved_source.resolved_revision in card
    assert "Tested on: **GPU**" in card
    assert "| CPU | not tested |" in card
    assert "| NPU | not tested |" in card


def test_every_repair_is_disclosed_with_its_digests(
    settings: Settings, hub: FakeHub, artifact: ArtifactManifest, tmp_path: Path
) -> None:
    service = publisher_service(settings, hub, FakePublisher(someone()), tmp_path)
    plan = service.plan(artifact.artifact_id, REPO)
    card = plan.generated["README.md"]

    applied = [record for record in artifact.repairs if record.applied]
    assert applied, "the fixture export ships the tokenizer defect"
    for record in applied:
        assert record.repair_id in plan.repairs
        assert f"`{record.repair_id}`" in card
        for digest in [*record.before_hashes.values(), *record.after_hashes.values()]:
            assert digest in card


def test_every_file_is_listed_with_its_digest(
    settings: Settings, hub: FakeHub, artifact: ArtifactManifest, tmp_path: Path
) -> None:
    service = publisher_service(settings, hub, FakePublisher(someone()), tmp_path)
    card = service.plan(artifact.artifact_id, REPO).generated["README.md"]

    for record in artifact.files:
        assert record.sha256 is not None
        assert f"| `{record.path}` | {record.size} | `{record.sha256}` |" in card


def test_no_local_path_is_published(
    settings: Settings, hub: FakeHub, artifact: ArtifactManifest, tmp_path: Path
) -> None:
    service = publisher_service(settings, hub, FakePublisher(someone()), tmp_path)
    plan = service.plan(artifact.artifact_id, REPO)

    for name, text in plan.generated.items():
        assert str(tmp_path) not in text, f"{name} leaks a local path"
    assert "~/" in plan.generated["yonga-manifest.json"], "paths are anonymised, not dropped"


def test_a_failed_device_is_listed_as_failed_and_never_claimed(
    settings: Settings, hub: FakeHub, artifact: ArtifactManifest, tmp_path: Path
) -> None:
    failed = DeviceProbeReport(
        device="NPU",
        status=ProbeStatus.FAILED,
        diagnostic=diagnostic(
            "npu.compiler.unsupported_graph",
            FailureCategory.COMPILER_UNSUPPORTED_GRAPH,
            "NPU compiler rejected the graph",
            "unsupported operation",
        ),
    )
    rewrite_manifest(settings, artifact, device_probes={**artifact.device_probes, "NPU": failed})
    service = publisher_service(settings, hub, FakePublisher(someone()), tmp_path)
    plan = service.plan(artifact.artifact_id, REPO)
    card = plan.generated["README.md"]

    assert plan.supported_devices == ["GPU"]
    assert "| NPU | **failed** | `npu.compiler.unsupported_graph`" in card
    assert "**NPU: failed.**" in card
    assert "Tested on: **GPU**" in card


def test_an_inconclusive_device_is_neither_claimed_nor_called_failed(
    settings: Settings, hub: FakeHub, artifact: ArtifactManifest, tmp_path: Path
) -> None:
    """A probe stopped by --quick or --timeout proves nothing in either direction."""
    unsettled = DeviceProbeReport(
        device="NPU",
        status=ProbeStatus.INCONCLUSIVE,
        diagnostic=diagnostic(
            "probe.inconclusive.timeout",
            FailureCategory.UNKNOWN_RUNTIME_ERROR,
            "NPU is inconclusive: the pipeline build hit --timeout",
            "stopped",
            severity="warning",
        ),
    )
    rewrite_manifest(settings, artifact, device_probes={**artifact.device_probes, "NPU": unsettled})
    service = publisher_service(settings, hub, FakePublisher(someone()), tmp_path)
    plan = service.plan(artifact.artifact_id, REPO)
    card = plan.generated["README.md"]

    assert plan.supported_devices == ["GPU"]
    assert "| NPU | inconclusive |" in card
    assert "**NPU: inconclusive.**" in card
    assert "**NPU: failed.**" not in card
    assert "| NPU | works |" not in card


# --- refusals -------------------------------------------------------------------------


def refused(service: PublishService, reference: str, repo: str = REPO, **kwargs: bool) -> str:
    with pytest.raises(YongaError) as caught:
        service.plan(reference, repo, **kwargs)
    return caught.value.diagnostic.code


@pytest.mark.parametrize("repo", ["no-namespace", "a/b/c", "/leading", "bad name/x"])
def test_the_destination_must_be_an_explicit_repository_id(
    settings: Settings, hub: FakeHub, artifact: ArtifactManifest, tmp_path: Path, repo: str
) -> None:
    service = publisher_service(settings, hub, FakePublisher(someone()), tmp_path)

    assert refused(service, artifact.artifact_id, repo) == "publish.repo_invalid"


def test_offline_mode_refuses(hub: FakeHub, artifact: ArtifactManifest, tmp_path: Path) -> None:
    offline = settings_for(tmp_path, offline=True)
    service = publisher_service(offline, hub, FakePublisher(someone()), tmp_path)

    assert refused(service, artifact.artifact_id) == "publish.offline"


@pytest.mark.parametrize(
    ("identity", "code"),
    [
        (None, "publish.hub.not_logged_in"),
        (someone(can_write=False), "publish.hub.read_only_token"),
        (someone(name="other", organizations=[]), "publish.hub.foreign_namespace"),
    ],
)
def test_credentials_must_be_able_to_write_there(
    settings: Settings,
    hub: FakeHub,
    artifact: ArtifactManifest,
    tmp_path: Path,
    identity: PublisherIdentity | None,
    code: str,
) -> None:
    service = publisher_service(settings, hub, FakePublisher(identity), tmp_path)

    assert refused(service, artifact.artifact_id) == code


def test_a_dry_run_reports_credential_problems_instead_of_refusing(
    settings: Settings, hub: FakeHub, artifact: ArtifactManifest, tmp_path: Path
) -> None:
    service = publisher_service(settings, hub, FakePublisher(None), tmp_path)

    plan = service.plan(artifact.artifact_id, REPO, dry_run=True)

    assert any("would be refused" in warning for warning in plan.warnings)
    assert "README.md" in plan.generated


def test_an_organisation_namespace_is_allowed(
    settings: Settings, hub: FakeHub, artifact: ArtifactManifest, tmp_path: Path
) -> None:
    service = publisher_service(settings, hub, FakePublisher(someone()), tmp_path)

    assert service.plan(artifact.artifact_id, "team/model").repo_id == "team/model"


def test_an_existing_repository_needs_update(
    settings: Settings, hub: FakeHub, artifact: ArtifactManifest, tmp_path: Path
) -> None:
    service = publisher_service(settings, hub, FakePublisher(someone(), existing=True), tmp_path)

    assert refused(service, artifact.artifact_id) == "publish.repo_exists"
    plan = service.plan(artifact.artifact_id, REPO, update=True)
    assert plan.repo_exists
    assert any("already exists" in warning for warning in plan.warnings)


def test_a_license_forbidding_derivatives_refuses(
    settings: Settings, hub: FakeHub, artifact: ArtifactManifest, tmp_path: Path
) -> None:
    hub.repos[SAFETENSORS_REPO] = safetensors_metadata().model_copy(
        update={"license": "cc-by-nd-4.0"}
    )
    service = publisher_service(settings, hub, FakePublisher(someone()), tmp_path)

    assert refused(service, artifact.artifact_id) == "publish.license.restricted"
    assert refused(service, artifact.artifact_id, license_reviewed=True) == (
        "publish.license.restricted"
    ), "--license-reviewed cannot override a license that forbids it"


def test_an_unknown_license_needs_the_publisher_to_have_read_it(
    settings: Settings, hub: FakeHub, artifact: ArtifactManifest, tmp_path: Path
) -> None:
    hub.repos[SAFETENSORS_REPO] = safetensors_metadata().model_copy(update={"license": "other"})
    service = publisher_service(settings, hub, FakePublisher(someone()), tmp_path)

    assert refused(service, artifact.artifact_id) == "publish.license.unreviewed"
    card = service.plan(artifact.artifact_id, REPO, license_reviewed=True).generated["README.md"]
    assert "the publisher stated that they reviewed them" in card


def test_a_file_changed_since_preparation_refuses(
    settings: Settings, hub: FakeHub, artifact: ArtifactManifest, tmp_path: Path
) -> None:
    directory = ArtifactStore(settings.paths, settings).find(artifact.artifact_id)
    assert directory is not None
    (directory / "config.json").write_text('{"tampered": true}\n', encoding="utf-8")
    service = publisher_service(settings, hub, FakePublisher(someone()), tmp_path)

    assert refused(service, artifact.artifact_id) == "publish.files_changed"


def test_an_artifact_without_a_passing_probe_refuses(
    settings: Settings, hub: FakeHub, artifact: ArtifactManifest, tmp_path: Path
) -> None:
    rewrite_manifest(settings, artifact, device_probes={})
    service = publisher_service(settings, hub, FakePublisher(someone()), tmp_path)

    assert refused(service, artifact.artifact_id) == "publish.artifact_not_ready"


def test_an_unknown_artifact_refuses(settings: Settings, hub: FakeHub, tmp_path: Path) -> None:
    service = publisher_service(settings, hub, FakePublisher(someone()), tmp_path)

    with pytest.raises(ConfigurationError) as caught:
        service.plan("nope", REPO)
    assert caught.value.diagnostic.code == "publish.artifact_not_found"


def test_the_preview_writes_the_card_but_no_weights(
    settings: Settings, hub: FakeHub, artifact: ArtifactManifest, tmp_path: Path
) -> None:
    service = publisher_service(settings, hub, FakePublisher(someone()), tmp_path)
    plan = service.plan(artifact.artifact_id, REPO)

    written = service.write_preview(plan, tmp_path / "preview")

    assert sorted(path.name for path in written) == [
        "README.md",
        "upload.txt",
        "yonga-manifest.json",
    ]
    listing = (tmp_path / "preview" / "upload.txt").read_text(encoding="utf-8")
    assert "openvino_language_model.bin" in listing
