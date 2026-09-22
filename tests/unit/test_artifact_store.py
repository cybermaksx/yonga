"""The artifact store: layout, listing, lookup, atomic publication and removal.

The invariant these tests exist for is the one in ``SPEC.md`` section 24: nothing
that is incomplete, unparsable or unvalidated may ever be advertised as ready.
Each way an artifact can be broken gets its own case, because "ready" is what
``yonga chat`` and ``yonga bench`` trust.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from yonga.adapters.filesystem import INCOMPLETE_MARKER_NAME, mark_incomplete
from yonga.adapters.openvino_probe import artifact_fingerprint, compile_cache_key
from yonga.application.artifacts import (
    HASH_SKIP_THRESHOLD_BYTES,
    SOURCE_FILENAME,
    VALIDATION_FILENAME,
    ArtifactStore,
    artifact_directory_name,
    revision_component,
)
from yonga.application.progress import CollectingSink, ProgressKind
from yonga.config.settings import Settings
from yonga.domain.artifacts import (
    MANIFEST_DIRNAME,
    MANIFEST_FILENAME,
    ArtifactManifest,
    DeviceProbeReport,
    ExportRecord,
    ManifestModelRef,
    ProbeResult,
    ProbeStatus,
    RecommendedRuntime,
    ValidationReport,
    manifest_path,
)
from yonga.domain.diagnostics import ConfigurationError, ValidationError
from yonga.domain.models import PipelineKind, SourceRelationship
from yonga.util.hashing import sha256_file

REPO_ID = "nDimensional/Qwen3.5-9B-Uncensored-Safetensors"
SHA = "4623e504074fc2489f3af98dc3d68b8392bbc21f"
PROFILE = "gpu-int4-asym-g128"


def settings_for(tmp_path: Path, **overrides: object) -> Settings:
    """Build settings rooted entirely inside ``tmp_path``."""
    payload: dict[str, object] = {
        "config_dir": tmp_path / "config",
        "cache_dir": tmp_path / "cache",
        "data_dir": tmp_path / "data",
    }
    payload.update(overrides)
    return Settings.model_validate(payload)


def make_manifest(
    *,
    artifact_id: str = "a1b2c3d4e5f60718",
    repo_id: str = REPO_ID,
    revision: str = SHA,
    profile: str = PROFILE,
    ready: bool = True,
    device: str = "GPU",
    created_at: datetime | None = None,
    arguments: dict[str, object] | None = None,
) -> ArtifactManifest:
    """Build a manifest that is ready (or deliberately not) for store tests."""
    probe_status = ProbeStatus.OK if ready else ProbeStatus.FAILED
    return ArtifactManifest(
        artifact_id=artifact_id,
        created_at=created_at or datetime.now(tz=UTC),
        tool_version="0.1.0",
        requested_model=ManifestModelRef(repo_id=repo_id, resolved_revision=revision),
        resolved_source=ManifestModelRef(
            repo_id=repo_id,
            resolved_revision=revision,
            relationship=SourceRelationship.ORIGINAL,
        ),
        architecture="arch-qwen3_5",
        task="image-text-to-text",
        pipeline=PipelineKind.VLM,
        export=ExportRecord(
            profile=profile,
            argv=["python", "-m", "optimum.commands.optimum_cli", "export", "openvino"],
            arguments=arguments or {"weight_format": "int4"},
            package_versions={"openvino": "2026.4.0"},
        ),
        validation=ValidationReport(ok=ready, openvino_version="2026.4.0"),
        device_probes={
            device: DeviceProbeReport(
                device=device,
                status=probe_status,
                submodels=[ProbeResult(device=device, status=probe_status)],
            )
        },
        recommended_runtime=RecommendedRuntime(device=device, pipeline=PipelineKind.VLM),
        source_license="apache-2.0",
    )


def make_staging(tmp_path: Path, *, name: str = "staging", big_file: bool = False) -> Path:
    """Create a staging directory that looks like a finished export."""
    staging = tmp_path / name
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "openvino_language_model.xml").write_text("<net/>\n", encoding="utf-8")
    (staging / "openvino_language_model.bin").write_bytes(b"\x01" * 128)
    (staging / "config.json").write_text('{"model_type": "qwen3_5"}\n', encoding="utf-8")
    if big_file:
        (staging / "openvino_language_model.bin").write_bytes(
            b"\x02" * (HASH_SKIP_THRESHOLD_BYTES + 1)
        )
    mark_incomplete(staging, job_id="job-under-test")
    return staging


def publish(
    store: ArtifactStore,
    tmp_path: Path,
    *,
    manifest: ArtifactManifest | None = None,
    name: str = "staging",
    sink: CollectingSink | None = None,
) -> Path:
    """Publish a freshly made staging directory through the store."""
    document = manifest if manifest is not None else make_manifest()
    staging = make_staging(tmp_path, name=name)
    final = store.artifact_dir(
        document.resolved_source.repo_id,
        document.resolved_source.resolved_revision,
        document.export.profile,
    )
    return store.finalize(staging, final, document, progress=sink)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings confined to the test's temporary directory."""
    return settings_for(tmp_path)


@pytest.fixture
def store(settings: Settings) -> ArtifactStore:
    """An artifact store over those settings."""
    return ArtifactStore(settings.paths, settings)


# --- layout --------------------------------------------------------------------------


def test_artifact_dir_follows_the_documented_layout(
    store: ArtifactStore, settings: Settings
) -> None:
    """``<artifacts>/<namespace>--<model>/<sha-short>--<profile>/``."""
    directory = store.artifact_dir(REPO_ID, SHA, PROFILE)

    assert directory.parent.parent == settings.paths.artifacts_dir
    assert directory.parent.name == "nDimensional--Qwen3.5-9B-Uncensored-Safetensors"
    assert directory.name == f"{SHA[:12]}--{PROFILE}"
    # Naming is a pure decision: nothing is created on disk.
    assert not directory.exists()


def test_revision_component_handles_shas_branches_and_nothing() -> None:
    """A SHA is truncated, a branch name is sanitised, and an unknown one is named."""
    assert revision_component(SHA) == SHA[:12]
    assert revision_component("main") == "main"
    assert revision_component(None) == "norev"
    assert revision_component("") == "norev"
    assert revision_component("refs/pr/7") == "refs_pr_7"
    # A long reference is shortened deterministically, and two different long
    # references never collapse onto the same directory name.
    long_one = revision_component("a-very-long-branch-name-indeed")
    other = revision_component("a-very-long-branch-name-instead")
    assert len(long_one) <= 12
    assert long_one == revision_component("a-very-long-branch-name-indeed")
    assert long_one != other
    assert artifact_directory_name(PROFILE, SHA) == f"{SHA[:12]}--{PROFILE}"


# --- publication ---------------------------------------------------------------------


def test_finalize_publishes_atomically_with_metadata(store: ArtifactStore, tmp_path: Path) -> None:
    """The manifest is written inside staging and the directory is renamed into place."""
    manifest = make_manifest()
    staging = make_staging(tmp_path)
    final = store.artifact_dir(REPO_ID, SHA, PROFILE)

    published = store.finalize(staging, final, manifest)

    assert published == final
    assert not staging.exists()
    assert manifest_path(published).is_file()
    assert (published / MANIFEST_DIRNAME / VALIDATION_FILENAME).is_file()
    assert (published / MANIFEST_DIRNAME / SOURCE_FILENAME).is_file()
    assert not (published / INCOMPLETE_MARKER_NAME).exists()
    written = json.loads(manifest_path(published).read_text(encoding="utf-8"))
    assert written["artifact_id"] == manifest.artifact_id
    assert written["schema_version"] == 1
    source_doc = json.loads(
        (published / MANIFEST_DIRNAME / SOURCE_FILENAME).read_text(encoding="utf-8")
    )
    assert source_doc["resolved_source"]["repo_id"] == REPO_ID
    assert store.is_ready(published) is True


def test_finalize_hashes_every_file_and_emits_progress(
    store: ArtifactStore, tmp_path: Path
) -> None:
    """Digests are streamed, reported and written onto the manifest."""
    manifest = make_manifest()
    sink = CollectingSink()

    published = publish(store, tmp_path, manifest=manifest, sink=sink)

    recorded = {entry.path: entry.sha256 for entry in manifest.files}
    assert set(recorded) == {
        "config.json",
        "openvino_language_model.bin",
        "openvino_language_model.xml",
    }
    assert INCOMPLETE_MARKER_NAME not in recorded
    for name, digest in recorded.items():
        assert digest == sha256_file(published / name)
    events = sink.of_kind(ProgressKind.HASH)
    assert {event.message for event in events} == set(recorded)
    assert all(event.total == 3 for event in events)


def test_hash_weights_off_skips_large_files_only(tmp_path: Path) -> None:
    """The development shortcut skips big files and records no digest for them."""
    settings = settings_for(tmp_path, hash_weights=False)
    store = ArtifactStore(settings.paths, settings)
    manifest = make_manifest()
    staging = make_staging(tmp_path, big_file=True)

    published = store.finalize(staging, store.artifact_dir(REPO_ID, SHA, PROFILE), manifest)

    by_path = {entry.path: entry for entry in manifest.files}
    assert by_path["openvino_language_model.bin"].sha256 is None
    assert by_path["openvino_language_model.bin"].size > HASH_SKIP_THRESHOLD_BYTES
    # Small files are still hashed: skipping them would buy nothing.
    assert by_path["config.json"].sha256 == sha256_file(published / "config.json")


def test_finalize_refuses_to_replace_a_foreign_directory(
    store: ArtifactStore, tmp_path: Path
) -> None:
    """Only a tool-owned incomplete build is replaced without ``--force``."""
    final = store.artifact_dir(REPO_ID, SHA, PROFILE)
    final.mkdir(parents=True)
    (final / "someones-notes.txt").write_text("do not delete me\n", encoding="utf-8")
    staging = make_staging(tmp_path)

    with pytest.raises(ConfigurationError) as caught:
        store.finalize(staging, final, make_manifest())

    assert caught.value.diagnostic.code == "artifact.finalize.refused"
    assert (final / "someones-notes.txt").is_file()

    published = store.finalize(staging, final, make_manifest(), force=True)
    assert not (published / "someones-notes.txt").exists()


def test_finalize_without_staging_is_a_typed_failure(store: ArtifactStore, tmp_path: Path) -> None:
    """Publishing a directory that does not exist reports it rather than crashing."""
    with pytest.raises(ValidationError) as caught:
        store.finalize(
            tmp_path / "never-created",
            store.artifact_dir(REPO_ID, SHA, PROFILE),
            make_manifest(),
        )

    assert caught.value.diagnostic.code == "artifact.finalize.missing_staging"


def test_finalize_refuses_a_manifest_carrying_a_secret(
    store: ArtifactStore, tmp_path: Path
) -> None:
    """A key that looks like a credential stops publication (``SPEC.md`` section 14)."""
    manifest = make_manifest(arguments={"weight_format": "int4", "hf_token": "hf_abc"})
    staging = make_staging(tmp_path)

    with pytest.raises(ValidationError) as caught:
        store.finalize(staging, store.artifact_dir(REPO_ID, SHA, PROFILE), manifest)

    assert caught.value.diagnostic.code == "artifact.manifest.secret_key"


# --- readiness -----------------------------------------------------------------------


def test_staged_artifact_is_never_listed(store: ArtifactStore, settings: Settings) -> None:
    """A complete manifest under ``.staging`` is still not an artifact."""
    staged = settings.paths.staging_dir / "job-1" / "model"
    staged.mkdir(parents=True)
    manifest_path(staged).parent.mkdir(parents=True)
    manifest_path(staged).write_text(make_manifest().to_json_text(), encoding="utf-8")

    assert store.list_artifacts() == []
    assert store.is_ready(staged) is False
    # An explicit path still resolves — the caller is the one who asked for it —
    # but it can never be mistaken for a ready artifact.
    assert store.find(str(staged)) == staged


def test_incomplete_marker_blocks_readiness(store: ArtifactStore, tmp_path: Path) -> None:
    """A leftover marker means the build never finished, whatever the manifest says."""
    published = publish(store, tmp_path)
    (published / INCOMPLETE_MARKER_NAME).write_text("job_id=interrupted\n", encoding="utf-8")

    assert store.is_ready(published) is False
    summary = next(item for item in store.list_artifacts() if item.path == published)
    assert summary.ready is False


def test_unreadable_manifest_is_listed_but_never_ready(
    store: ArtifactStore, tmp_path: Path
) -> None:
    """A corrupt manifest is visible so it can be removed, and is never usable."""
    published = publish(store, tmp_path)
    manifest_path(published).write_text("{ this is not json", encoding="utf-8")

    assert store.manifest_for(published) is None
    assert store.is_ready(published) is False
    summaries = store.list_artifacts()
    assert len(summaries) == 1
    assert summaries[0].ready is False
    assert summaries[0].artifact_id.startswith("unknown:")
    assert summaries[0].repo_id == REPO_ID


def test_manifest_that_says_not_ready_is_not_ready(store: ArtifactStore, tmp_path: Path) -> None:
    """Validation failure or a failed probe both mean "not ready"."""
    published = publish(store, tmp_path, manifest=make_manifest(ready=False))

    assert store.is_ready(published) is False
    summary = store.summarise(published)
    assert summary.ready is False
    assert summary.recommended_device == "GPU"


def test_ready_summary_carries_the_facts_yonga_list_shows(
    store: ArtifactStore, tmp_path: Path
) -> None:
    """The summary is what ``yonga list`` renders, so every column has to be filled."""
    manifest = make_manifest()
    published = publish(store, tmp_path, manifest=manifest)

    summary = store.summarise(published)

    assert summary.ready is True
    assert summary.artifact_id == manifest.artifact_id
    assert summary.repo_id == REPO_ID
    assert summary.resolved_revision == SHA
    assert summary.profile == PROFILE
    assert summary.recommended_device == "GPU"
    assert summary.pipeline is PipelineKind.VLM
    assert summary.size_bytes > 0
    assert summary.last_benchmark is None


def test_last_benchmark_is_read_from_the_benchmark_directory(
    store: ArtifactStore, settings: Settings, tmp_path: Path
) -> None:
    """``yonga list`` shows when an artifact was last benchmarked."""
    manifest = make_manifest()
    published = publish(store, tmp_path, manifest=manifest)
    settings.paths.benchmarks_dir.mkdir(parents=True, exist_ok=True)
    older = datetime.now(tz=UTC) - timedelta(days=2)
    newer = datetime.now(tz=UTC) - timedelta(hours=1)
    for index, moment in enumerate((older, newer)):
        (settings.paths.benchmarks_dir / f"bench-{index}.json").write_text(
            json.dumps({"artifact_id": manifest.artifact_id, "created_at": moment.isoformat()}),
            encoding="utf-8",
        )

    summary = next(item for item in store.list_artifacts() if item.path == published)

    assert summary.last_benchmark is not None
    assert abs((summary.last_benchmark - newer).total_seconds()) < 1


# --- lookup --------------------------------------------------------------------------


def test_find_resolves_id_repo_directory_and_path(store: ArtifactStore, tmp_path: Path) -> None:
    """Every reference the CLI accepts resolves to the same directory."""
    manifest = make_manifest()
    published = publish(store, tmp_path, manifest=manifest)

    assert store.find(manifest.artifact_id) == published
    assert store.find(manifest.artifact_id[:8]) == published
    assert store.find(REPO_ID) == published
    assert store.find(f"{REPO_ID}@{SHA}") == published
    assert store.find(published.name) == published
    assert store.find(f"{published.parent.name}/{published.name}") == published
    assert store.find(str(published)) == published
    assert store.find(str(manifest_path(published))) == published
    assert store.find("nothing/at-all") is None
    assert store.find("") is None


def test_find_prefers_a_ready_artifact_over_a_broken_one(
    store: ArtifactStore, tmp_path: Path
) -> None:
    """Two artifacts for one repository: the usable one wins."""
    broken = publish(
        store,
        tmp_path,
        manifest=make_manifest(artifact_id="broken0000000000", ready=False),
        name="staging-broken",
    )
    good_manifest = make_manifest(artifact_id="good000000000000", profile="cpu-int4-asym-g128")
    good = publish(store, tmp_path, manifest=good_manifest, name="staging-good")

    assert broken != good
    assert store.find(REPO_ID) == good


# --- removal -------------------------------------------------------------------------


def test_remove_deletes_the_artifact_and_its_compile_cache(
    store: ArtifactStore, settings: Settings, tmp_path: Path
) -> None:
    """Only the tool-owned artifact and its own cache entries disappear."""
    manifest = make_manifest()
    published = publish(store, tmp_path, manifest=manifest)
    key = compile_cache_key(
        artifact_hash=artifact_fingerprint(published),
        device="GPU",
        openvino_version="2026.4.0",
    )
    cache_entry = settings.paths.compile_cache_dir / key
    cache_entry.mkdir(parents=True)
    (cache_entry / "blob.cl_cache").write_bytes(b"\x00" * 64)
    unrelated = settings.paths.compile_cache_dir / "someone-elses-key"
    unrelated.mkdir(parents=True)

    report = store.remove(manifest.artifact_id)

    assert report.removed is True
    assert report.artifact_id == manifest.artifact_id
    assert cache_entry in report.compile_cache_paths
    assert not published.exists()
    assert not cache_entry.exists()
    assert unrelated.exists()
    assert report.freed_bytes > 0
    # The now-empty family directory is tidied away too.
    assert not published.parent.exists()
    assert settings.paths.artifacts_dir.exists()


def test_remove_dry_run_changes_nothing(store: ArtifactStore, tmp_path: Path) -> None:
    """A dry run reports exactly what would go and leaves it in place."""
    published = publish(store, tmp_path)

    report = store.remove(published, dry_run=True)

    assert report.dry_run is True
    assert report.removed is False
    assert report.removed_paths[0] == published.resolve()
    assert published.exists()


def test_remove_refuses_a_path_outside_the_store(store: ArtifactStore, tmp_path: Path) -> None:
    """Anything outside ``paths.artifacts_dir`` is refused, including the root itself."""
    outsider = tmp_path / "not-ours"
    outsider.mkdir()

    with pytest.raises(ConfigurationError) as caught:
        store.remove(outsider)

    assert caught.value.diagnostic.code == "artifact.remove.outside_store"
    assert outsider.exists()

    store.root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ConfigurationError):
        store.remove(store.root)
    assert store.root.is_dir()


def test_remove_refuses_anything_inside_the_hugging_face_cache(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared Hub cache is never this tool's to delete (``SPEC.md`` section 8.3)."""
    hub_cache = settings.paths.artifacts_dir / "hf-cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))
    store = ArtifactStore(settings.paths, settings)
    target = hub_cache / "models--acme--thing"
    target.mkdir(parents=True)
    (target / "weights.safetensors").write_bytes(b"\x00" * 16)

    with pytest.raises(ConfigurationError) as caught:
        store.remove(target)

    assert caught.value.diagnostic.code == "artifact.remove.hf_cache"
    assert (target / "weights.safetensors").is_file()


def test_remove_reports_an_unknown_reference(store: ArtifactStore) -> None:
    """A reference that matches nothing is a typed error, not a silent no-op."""
    with pytest.raises(ConfigurationError) as caught:
        store.remove("acme/never-prepared")

    assert caught.value.diagnostic.code == "artifact.remove.not_found"


def test_remove_handles_an_artifact_without_a_manifest(
    store: ArtifactStore, tmp_path: Path
) -> None:
    """A corrupt artifact can still be cleaned up."""
    published = publish(store, tmp_path)
    (published / MANIFEST_DIRNAME / MANIFEST_FILENAME).unlink()

    report = store.remove(published)

    assert report.removed is True
    assert report.artifact_id is None
    assert any("no readable manifest" in note for note in report.notes)
