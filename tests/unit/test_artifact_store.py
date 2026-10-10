"""The artifact store: layout, listing, lookup, atomic publication and removal.

The invariant these tests exist for is the one in ``SPEC.md`` section 24: nothing
that is incomplete, unparsable or unvalidated may ever be advertised as ready.
Each way an artifact can be broken gets its own case, because "ready" is what
``yonga chat`` and ``yonga bench`` trust.
"""

from __future__ import annotations

import json
import os
import shutil
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


def _three_artifacts_of_one_repository(store: ArtifactStore, tmp_path: Path) -> list[Path]:
    """Publish one repository three times, for GPU, CPU and NPU, oldest first."""
    published: list[Path] = []
    start = datetime.now(tz=UTC) - timedelta(hours=3)
    for index, (artifact_id, profile, device) in enumerate(
        (
            ("aaaa000000000001", "gpu-int4-asym-g128", "GPU"),
            ("aaaa000000000002", "cpu-int8", "CPU"),
            ("bbbb000000000003", "npu-int4-sym-g128", "NPU"),
        )
    ):
        manifest = make_manifest(
            artifact_id=artifact_id,
            profile=profile,
            device=device,
            created_at=start + timedelta(hours=index),
        )
        published.append(publish(store, tmp_path, manifest=manifest, name=f"staging-{index}"))
    return published


def test_matches_returns_every_artifact_the_winning_rule_selects(
    store: ArtifactStore, tmp_path: Path
) -> None:
    """A repository id answers with all its artifacts, best first; an exact id with one."""
    gpu, cpu, npu = _three_artifacts_of_one_repository(store, tmp_path)

    assert [item.path for item in store.matches(REPO_ID)] == [npu, cpu, gpu]
    assert [item.path for item in store.matches("aaaa000000000002")] == [cpu]
    assert [item.path for item in store.matches("aaaa0000")] == [cpu, gpu]
    assert store.matches("nothing/at-all") == []
    # The read-only lookup still settles the tie on the newest.
    assert store.find(REPO_ID) == npu


@pytest.mark.parametrize("reference", [REPO_ID, f"{REPO_ID}@{SHA[:7]}", "aaaa0000"])
def test_remove_refuses_a_reference_that_matches_several_artifacts(
    store: ArtifactStore, tmp_path: Path, reference: str
) -> None:
    """Deleting is never a guess: every candidate is named and nothing is removed."""
    published = _three_artifacts_of_one_repository(store, tmp_path)

    with pytest.raises(ConfigurationError) as caught:
        store.remove(reference)

    found = caught.value.diagnostic
    assert found.code == "artifact.remove.ambiguous"
    assert caught.value.exit_code() == 70
    assert all(path.is_dir() for path in published)
    expected = [s.artifact_id for s in store.matches(reference)]
    assert len(expected) > 1
    for artifact_id in expected:
        assert any(f"yonga remove {artifact_id}" in step for step in found.remediation)
    assert found.context["candidates"] == ", ".join(expected)


def test_remove_by_a_unique_reference_still_removes_only_that_artifact(
    store: ArtifactStore, tmp_path: Path
) -> None:
    """An exact id (or a prefix only one id has) is not ambiguous."""
    gpu, cpu, npu = _three_artifacts_of_one_repository(store, tmp_path)

    report = store.remove("bbbb0000")

    assert report.removed is True
    assert not npu.exists()
    assert gpu.is_dir()
    assert cpu.is_dir()
    assert store.remove("aaaa000000000001").removed is True
    assert not gpu.exists()
    # One artifact left for the repository: its id is now unambiguous.
    assert store.remove(REPO_ID).removed is True
    assert not cpu.exists()


def test_remove_names_copies_of_one_artifact_by_path(store: ArtifactStore, tmp_path: Path) -> None:
    """Two directories claiming one artifact id can only be told apart by path."""
    original = publish(store, tmp_path)
    copy = original.parent / "copy-of-it"
    shutil.copytree(original, copy)
    artifact_id = make_manifest().artifact_id

    with pytest.raises(ConfigurationError) as caught:
        store.remove(artifact_id)

    remediation = caught.value.diagnostic.remediation
    assert any(f"yonga remove {original}" in step for step in remediation)
    assert any(f"yonga remove {copy}" in step for step in remediation)
    assert original.is_dir()
    assert copy.is_dir()


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


# --- compile-cache identity ----------------------------------------------------------


def cache_entry_for(settings: Settings, manifest: ArtifactManifest, *, size: int = 4096) -> Path:
    """Create the compile-cache directory an artifact's identity would produce."""
    key = compile_cache_key(
        artifact_hash=manifest.artifact_id,
        device="GPU",
        openvino_version="2026.4.0",
    )
    entry = settings.paths.compile_cache_dir / key
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "blob.cl_cache").write_bytes(b"\x00" * size)
    return entry


def copy_artifact(store: ArtifactStore, published: Path, *, family: str) -> Path:
    """Make a byte-identical copy of an artifact somewhere else in the store."""
    copy = store.root / family / published.name
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(published, copy)
    return copy


def test_the_compile_cache_key_follows_the_manifest_identity(
    store: ArtifactStore, settings: Settings, tmp_path: Path
) -> None:
    """The manifest is the identity; the bytes in the directory are not.

    Editing a file inside a prepared artifact must not move its compile cache to
    a different key, or the cache would be orphaned by every repair.
    """
    manifest = make_manifest()
    published = publish(store, tmp_path, manifest=manifest)
    cache_entry = cache_entry_for(settings, manifest)
    (published / "openvino_language_model.bin").write_bytes(b"\xff" * 4096)

    report = store.remove(published, dry_run=True)

    assert cache_entry in report.compile_cache_paths


def test_removing_a_copy_leaves_the_originals_compile_cache_alone(
    store: ArtifactStore, settings: Settings, tmp_path: Path
) -> None:
    """A copy is the same artifact, so it does not own the cache by itself.

    Keying the cache on the directory's contents made ``yonga remove`` on a
    byte-identical copy delete the compiled blobs the original was still using.
    """
    manifest = make_manifest()
    original = publish(store, tmp_path, manifest=manifest)
    cache_entry = cache_entry_for(settings, manifest)
    copy = copy_artifact(store, original, family="someone--else")

    report = store.remove(copy)

    assert report.removed is True
    assert not copy.exists()
    assert original.is_dir()
    assert cache_entry.is_dir()
    assert report.compile_cache_paths == []
    assert any("still claims" in note for note in report.notes)


def test_the_last_copy_standing_does_take_the_compile_cache(
    store: ArtifactStore, settings: Settings, tmp_path: Path
) -> None:
    """Once nothing else claims the identity, the cache is garbage and goes."""
    manifest = make_manifest()
    original = publish(store, tmp_path, manifest=manifest)
    cache_entry = cache_entry_for(settings, manifest)
    copy = copy_artifact(store, original, family="someone--else")
    store.remove(copy)

    report = store.remove(original)

    assert cache_entry in report.compile_cache_paths
    assert not cache_entry.exists()


def test_freed_bytes_counts_only_what_actually_went_away(
    store: ArtifactStore, settings: Settings, tmp_path: Path
) -> None:
    """Counting a cache that was deliberately kept doubled the reported saving."""
    manifest = make_manifest()
    original = publish(store, tmp_path, manifest=manifest)
    cache_entry = cache_entry_for(settings, manifest, size=64 * 1024)
    copy = copy_artifact(store, original, family="someone--else")
    copy_bytes = sum(item.stat().st_size for item in copy.rglob("*") if item.is_file())

    report = store.remove(copy)

    assert report.freed_bytes == copy_bytes
    assert report.freed_bytes < copy_bytes + cache_entry.stat().st_size


def test_freed_bytes_reports_nothing_when_removal_failed(
    store: ArtifactStore, tmp_path: Path
) -> None:
    """A tree that survived a read-only directory was still counted as reclaimed."""
    published = publish(store, tmp_path)
    weights = published / "openvino_language_model.bin"
    frozen = [published, *(item for item in published.rglob("*") if item.is_dir())]
    modes = [(item, item.stat().st_mode) for item in frozen]
    for item, _ in reversed(modes):
        item.chmod(0o500)
    try:
        report = store.remove(published)
    finally:
        for item, mode in modes:
            item.chmod(mode)

    assert report.removed is False
    assert weights.is_file()
    assert report.freed_bytes == 0


# --- staging retention ---------------------------------------------------------------


def test_sweep_staging_retires_stale_directories_under_the_staging_root(
    store: ArtifactStore, settings: Settings
) -> None:
    """``yonga list`` hides the staging root, so the store has to sweep it."""
    root = settings.paths.staging_dir
    root.mkdir(parents=True, exist_ok=True)
    stale = root / "dead-job"
    (stale / "model").mkdir(parents=True)
    (stale / "model" / "openvino_model.bin").write_bytes(b"\x00" * 8192)
    moment = (datetime.now(tz=UTC) - timedelta(hours=200)).timestamp()
    os.utime(stale, (moment, moment))

    report = store.sweep_staging()

    assert stale in report.removed
    assert not stale.exists()
    assert report.freed_bytes >= 8192
    assert store.staging_bytes() == 0


def test_sweep_staging_leaves_published_artifacts_alone(
    store: ArtifactStore, settings: Settings, tmp_path: Path
) -> None:
    """The sweep only ever touches direct children of the staging root."""
    published = publish(store, tmp_path)
    settings.paths.staging_dir.mkdir(parents=True, exist_ok=True)

    store.sweep_staging()

    assert published.is_dir()
    assert store.is_ready(published)


def test_a_stale_compile_cache_entry_never_breaks_a_removal(
    store: ArtifactStore, settings: Settings, tmp_path: Path
) -> None:
    """Changing how a key is derived leaves old cache directories behind.

    They are junk, not a hazard: an entry under a key this artifact cannot
    produce is simply not claimed, and a non-directory sitting where a key would
    be is not mistaken for one.
    """
    manifest = make_manifest()
    published = publish(store, tmp_path, manifest=manifest)
    cache_root = settings.paths.compile_cache_dir
    cache_root.mkdir(parents=True, exist_ok=True)
    stale = cache_root / "0123456789abcdef"
    stale.mkdir()
    (stale / "blob.cl_cache").write_bytes(b"\x00" * 32)
    imposter = cache_root / compile_cache_key(
        artifact_hash=manifest.artifact_id,
        device="GPU",
        openvino_version="2026.4.0",
    )
    imposter.write_text("not a cache directory\n", encoding="utf-8")

    report = store.remove(published)

    assert report.removed is True
    assert imposter not in report.compile_cache_paths
    assert stale.is_dir()
    assert imposter.is_file()
