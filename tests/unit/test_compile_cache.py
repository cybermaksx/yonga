"""The compile cache: ownership records, classification, attribution and pruning.

Nothing here needs OpenVINO. Entries are made the way the runtime makes them --
``compile_cache_key`` plus ``compile_cache_dir`` plus the ownership record -- or,
for the entries that predate the record, as bare key-named directories. The
invariants are the ones that make ``yonga cache prune`` safe to run: only
direct, key-named, real directories under the compile cache root are ever
removed, a symbolic link is never followed, and an entry whose artifact still
exists and whose blobs the installed runtime would load is kept unless the user
asks for more.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.unit.test_artifact_store import make_manifest, publish, settings_for
from yonga.adapters.openvino_probe import (
    artifact_fingerprint,
    compile_cache_dir,
    compile_cache_key,
    read_compile_cache_record,
    record_compile_cache_use,
)
from yonga.application.artifacts import ArtifactStore
from yonga.application.compile_cache import CompileCacheService, cache_entry_dirs
from yonga.config.settings import Settings
from yonga.domain.compile_cache import (
    COMPILE_CACHE_ENTRY_FILENAME,
    CacheAttribution,
    CacheEntryStatus,
    CompileCacheEntry,
    CompileCacheInventory,
    is_cache_key,
)
from yonga.domain.diagnostics import ConfigurationError

#: The runtime version the tests treat as installed. The fixture manifests
#: record the same string as the version that validated them.
INSTALLED = "2026.4.0"

#: A release the installed runtime is not.
OLDER = "2026.3.0"

#: A fixed "now" for every age-based decision.
NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings rooted entirely inside ``tmp_path``."""
    return settings_for(tmp_path)


@pytest.fixture
def store(settings: Settings) -> ArtifactStore:
    """An empty artifact store under ``tmp_path``."""
    return ArtifactStore(settings.paths, settings)


def service_for(
    settings: Settings, store: ArtifactStore, *, installed: str | None = INSTALLED
) -> CompileCacheService:
    """A service with a fixed runtime version and clock."""
    return CompileCacheService(
        settings.paths, store, runtime_version=lambda: installed, clock=lambda: NOW
    )


def recorded_entry(
    settings: Settings,
    artifact_dir: Path,
    *,
    device: str = "GPU",
    version: str | None = INSTALLED,
    properties: dict[str, str] | None = None,
    size: int = 1024,
    used: datetime | None = None,
) -> Path:
    """Make an entry exactly as the runtime does, ownership record included."""
    fingerprint = artifact_fingerprint(artifact_dir)
    key = compile_cache_key(
        artifact_hash=fingerprint, device=device, openvino_version=version, properties=properties
    )
    directory = compile_cache_dir(settings.paths, key)
    (directory / "4392266892912006757.blob").write_bytes(b"\x00" * size)
    record_compile_cache_use(
        directory,
        artifact_dir=artifact_dir,
        artifact_hash=fingerprint,
        device=device,
        openvino_version=version,
        properties=properties,
        now=used,
    )
    return directory


def legacy_entry(settings: Settings, key: str, *, size: int = 2048) -> Path:
    """Make an entry the way yonga did before ownership records existed."""
    directory = settings.paths.compile_cache_dir / key
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "10058747372601149534.cl_cache").write_bytes(b"\x01" * size)
    return directory


def age(directory: Path, moment: datetime) -> None:
    """Backdate every file in an entry, and the entry itself, to ``moment``."""
    stamp = moment.timestamp()
    for path in [*directory.rglob("*"), directory]:
        os.utime(path, (stamp, stamp), follow_symlinks=False)


def by_key(inventory: CompileCacheInventory) -> dict[str, CompileCacheEntry]:
    """Index an inventory's entries by key."""
    return {entry.key: entry for entry in inventory.entries}


# --- the ownership record --------------------------------------------------------------


def test_the_record_names_the_artifact_and_survives_refreshes(
    settings: Settings, store: ArtifactStore, tmp_path: Path
) -> None:
    manifest = make_manifest()
    published = publish(store, tmp_path, manifest=manifest)
    first = datetime(2026, 9, 1, tzinfo=UTC)
    directory = recorded_entry(
        settings,
        published,
        properties={"ATTENTION_BACKEND": "SDPA", "CACHE_DIR": "/ignored"},
        used=first,
    )

    record = read_compile_cache_record(directory)
    assert record is not None
    assert record.artifact_id == manifest.artifact_id == record.fingerprint
    assert record.repo_id == manifest.resolved_source.repo_id
    assert record.profile == manifest.export.profile
    assert record.properties == {"ATTENTION_BACKEND": "SDPA"}, "CACHE_DIR is not a key input"
    assert record.created_at == record.last_used_at == first

    record_compile_cache_use(
        directory,
        artifact_dir=published,
        artifact_hash=manifest.artifact_id,
        device="GPU",
        openvino_version=INSTALLED,
        properties={"ATTENTION_BACKEND": "SDPA"},
        now=NOW,
    )
    refreshed = read_compile_cache_record(directory)
    assert refreshed is not None
    assert refreshed.created_at == first
    assert refreshed.last_used_at == NOW


def test_a_record_that_does_not_belong_to_its_directory_is_ignored(
    settings: Settings, tmp_path: Path
) -> None:
    root = settings.paths.compile_cache_dir
    moved = root / "0123456789abcdef"
    moved.mkdir(parents=True)
    (moved / COMPILE_CACHE_ENTRY_FILENAME).write_text(
        json.dumps(
            {
                "key": "fedcba9876543210",
                "fingerprint": "x",
                "device": "CPU",
                "created_at": NOW.isoformat(),
                "last_used_at": NOW.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    garbled = root / "1111111111111111"
    garbled.mkdir()
    (garbled / COMPILE_CACHE_ENTRY_FILENAME).write_text("{not json", encoding="utf-8")
    linked = root / "2222222222222222"
    linked.mkdir()
    outside = tmp_path / "planted.json"
    outside.write_text("{}", encoding="utf-8")
    (linked / COMPILE_CACHE_ENTRY_FILENAME).symlink_to(outside)

    assert read_compile_cache_record(moved) is None
    assert read_compile_cache_record(garbled) is None
    assert read_compile_cache_record(linked) is None


def test_a_record_is_never_written_through_a_symbolic_link(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = tmp_path / "0123456789abcdef"
    link.symlink_to(target, target_is_directory=True)

    record_compile_cache_use(
        link,
        artifact_dir=tmp_path,
        artifact_hash="abc",
        device="CPU",
        openvino_version=INSTALLED,
    )

    assert not any(target.iterdir())


def test_only_key_shaped_names_are_entries() -> None:
    assert is_cache_key("00573215847e5a72")
    assert not is_cache_key("00573215847E5A72")
    assert not is_cache_key("00573215847e5a7")
    assert not is_cache_key("tmp")
    assert not is_cache_key("../00573215847e5a")


# --- classification --------------------------------------------------------------------


def test_every_entry_gets_an_owner_and_a_status(
    settings: Settings, store: ArtifactStore, tmp_path: Path
) -> None:
    kept = make_manifest(artifact_id="aaaaaaaaaaaaaaaa")
    published = publish(store, tmp_path, manifest=kept, name="staging-kept")
    gone_manifest = make_manifest(artifact_id="bbbbbbbbbbbbbbbb", profile="cpu-int4-asym-g128")
    gone = publish(store, tmp_path, manifest=gone_manifest, name="staging-gone")

    in_use = recorded_entry(settings, published, size=100)
    stale = recorded_entry(settings, published, version=OLDER, size=200)
    orphaned = recorded_entry(settings, gone, device="NPU", size=400)
    shutil.rmtree(gone)  # deleted by hand, behind yonga's back

    # Entries from before ownership records: one the store's artifact produces
    # (NPU, the version that validated it, no properties), one a saved
    # benchmark ties to an artifact that no longer exists, and one nothing does.
    legacy_kept = legacy_entry(
        settings,
        compile_cache_key(artifact_hash=kept.artifact_id, device="NPU", openvino_version=INSTALLED),
        size=800,
    )
    settings.paths.benchmarks_dir.mkdir(parents=True)
    (settings.paths.benchmarks_dir / "20260922T174112Z-cpu-quick-5971cce8.json").write_text(
        json.dumps(
            {
                "artifact_id": "cccccccccccccccc",
                "artifact_path": str(tmp_path / "other-data" / "artifact"),
                "device": "CPU",
                "runtime_properties": {"INFERENCE_PRECISION_HINT": "f32"},
                "package_versions": {"openvino": OLDER},
            }
        ),
        encoding="utf-8",
    )
    legacy_benchmarked = legacy_entry(
        settings,
        compile_cache_key(
            artifact_hash="cccccccccccccccc",
            device="CPU",
            openvino_version=OLDER,
            properties={"INFERENCE_PRECISION_HINT": "f32"},
        ),
        size=1600,
    )
    unknown = legacy_entry(settings, "0123456789abcdef", size=3200)

    inventory = service_for(settings, store).inventory()
    entries = by_key(inventory)

    assert entries[in_use.name].status is CacheEntryStatus.IN_USE
    assert entries[in_use.name].owner_in_store
    assert entries[in_use.name].attribution is CacheAttribution.RECORD
    assert entries[in_use.name].artifact_path == published

    assert entries[stale.name].status is CacheEntryStatus.STALE_RUNTIME
    assert OLDER in entries[stale.name].reason and INSTALLED in entries[stale.name].reason

    assert entries[orphaned.name].status is CacheEntryStatus.ORPHANED
    assert entries[orphaned.name].artifact_id == gone_manifest.artifact_id
    assert not entries[orphaned.name].owner_exists

    assert entries[legacy_kept.name].status is CacheEntryStatus.IN_USE
    assert entries[legacy_kept.name].attribution is CacheAttribution.RECOMPUTED
    assert entries[legacy_kept.name].device == "NPU"
    assert entries[legacy_kept.name].artifact_id == kept.artifact_id

    assert entries[legacy_benchmarked.name].status is CacheEntryStatus.ORPHANED
    assert entries[legacy_benchmarked.name].artifact_id == "cccccccccccccccc"
    assert entries[legacy_benchmarked.name].openvino_version == OLDER

    assert entries[unknown.name].status is CacheEntryStatus.UNATTRIBUTED
    assert entries[unknown.name].attribution is CacheAttribution.NONE

    assert inventory.total_bytes == 100 + 200 + 400 + 800 + 1600 + 3200 + _record_bytes(
        in_use, stale, orphaned
    )
    assert inventory.reclaimable_bytes == sum(
        entries[path.name].size_bytes for path in (stale, orphaned, legacy_benchmarked)
    )
    assert inventory.counts() == {
        "in-use": 2,
        "orphaned": 2,
        "stale-runtime": 1,
        "unattributed": 1,
    }
    assert [entry.size_bytes for entry in inventory.entries] == sorted(
        (entry.size_bytes for entry in inventory.entries), reverse=True
    )


def _record_bytes(*directories: Path) -> int:
    """Bytes the ownership records themselves add to their entries."""
    return sum(
        (directory / COMPILE_CACHE_ENTRY_FILENAME).stat().st_size for directory in directories
    )


def test_nothing_is_stale_when_the_installed_runtime_is_unknown(
    settings: Settings, store: ArtifactStore, tmp_path: Path
) -> None:
    published = publish(store, tmp_path)
    entry = recorded_entry(settings, published, version=OLDER)

    inventory = service_for(settings, store, installed=None).inventory()

    assert inventory.installed_openvino is None
    assert by_key(inventory)[entry.name].status is CacheEntryStatus.IN_USE


def test_an_artifact_outside_the_store_keeps_its_entry(
    settings: Settings, store: ArtifactStore, tmp_path: Path
) -> None:
    """Another ``--data-dir``'s artifact still exists, so its blobs are not orphaned."""
    other_settings = settings_for(tmp_path / "other")
    other_store = ArtifactStore(other_settings.paths, other_settings)
    elsewhere = publish(other_store, tmp_path / "other", manifest=make_manifest())
    entry = recorded_entry(settings, elsewhere)

    found = by_key(service_for(settings, store).inventory())[entry.name]

    assert found.status is CacheEntryStatus.IN_USE
    assert found.owner_exists and not found.owner_in_store
    shutil.rmtree(elsewhere)
    gone = by_key(service_for(settings, store).inventory())[entry.name]
    assert gone.status is CacheEntryStatus.ORPHANED


def test_a_smoke_test_entry_dies_with_its_staging_directory(
    settings: Settings, store: ArtifactStore
) -> None:
    """``prepare`` smoke-tests a staging directory that has no manifest yet."""
    staging = settings.paths.job_staging_dir("job-1") / "model"
    staging.mkdir(parents=True)
    (staging / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    entry = recorded_entry(settings, staging, device="CPU")
    record = read_compile_cache_record(entry)
    assert record is not None and record.artifact_id is None

    assert by_key(service_for(settings, store).inventory())[entry.name].status is (
        CacheEntryStatus.IN_USE
    )
    shutil.rmtree(staging)
    assert by_key(service_for(settings, store).inventory())[entry.name].status is (
        CacheEntryStatus.ORPHANED
    )


def test_what_is_not_an_entry_is_reported_and_never_measured(
    settings: Settings, store: ArtifactStore, tmp_path: Path
) -> None:
    root = settings.paths.compile_cache_dir
    root.mkdir(parents=True)
    (root / "notes").mkdir()
    (root / "README").write_text("mine\n", encoding="utf-8")
    outside = tmp_path / "outside" / "abcdefabcdefabcd"
    outside.mkdir(parents=True)
    (outside / "precious.bin").write_bytes(b"\x02" * 64)
    (root / "abcdefabcdefabcd").symlink_to(outside, target_is_directory=True)
    (root / "fedcba9876543210").write_text("a file named like a key\n", encoding="utf-8")

    inventory = service_for(settings, store).inventory()

    assert inventory.entries == []
    assert inventory.ignored == ["README", "abcdefabcdefabcd", "fedcba9876543210", "notes"]
    assert cache_entry_dirs(root) == []


def test_an_empty_or_missing_cache_never_asks_for_the_runtime_version(
    settings: Settings, store: ArtifactStore
) -> None:
    def refuse() -> str:
        raise AssertionError("the runtime version is only needed when there are entries")

    service = CompileCacheService(settings.paths, store, runtime_version=refuse)
    inventory = service.inventory()

    assert inventory.entries == []
    assert inventory.total_bytes == 0


# --- planning ----------------------------------------------------------------------------


def test_the_plan_takes_only_what_the_options_allow(
    settings: Settings, store: ArtifactStore, tmp_path: Path
) -> None:
    kept = make_manifest(artifact_id="aaaaaaaaaaaaaaaa")
    published = publish(store, tmp_path, manifest=kept)
    fresh = recorded_entry(settings, published, device="GPU", used=NOW - timedelta(days=1))
    age(fresh, NOW - timedelta(days=1))
    old = recorded_entry(settings, published, device="CPU", used=NOW - timedelta(days=40))
    age(old, NOW - timedelta(days=40))
    stale = recorded_entry(settings, published, device="NPU", version=OLDER)
    unknown = legacy_entry(settings, "0123456789abcdef")

    service = service_for(settings, store)
    inventory = service.inventory()

    default = service.plan_prune(inventory)
    assert {entry.key for entry in default.entries} == {stale.name}
    assert default.reasons == {stale.name: "stale-runtime"}

    unattributed = service.plan_prune(inventory, unattributed=True)
    assert {entry.key for entry in unattributed.entries} == {stale.name, unknown.name}

    aged = service.plan_prune(inventory, older_than_days=30)
    assert {entry.key for entry in aged.entries} == {stale.name, old.name}
    assert aged.reasons[old.name] == "older-than"

    everything = service.plan_prune(inventory, everything=True)
    assert {entry.key for entry in everything.entries} == {
        fresh.name,
        old.name,
        stale.name,
        unknown.name,
    }
    assert set(everything.reasons.values()) == {"all"}
    assert everything.total_bytes == inventory.total_bytes


@pytest.mark.parametrize("days", [-1.0, float("nan"), float("inf")])
def test_a_nonsensical_age_is_a_configuration_error(
    settings: Settings, store: ArtifactStore, days: float
) -> None:
    service = service_for(settings, store)
    with pytest.raises(ConfigurationError) as raised:
        service.plan_prune(service.inventory(), older_than_days=days)
    assert raised.value.diagnostic.code == "cache.prune.older_than_invalid"


# --- pruning -----------------------------------------------------------------------------


def test_prune_removes_the_plan_and_nothing_else(
    settings: Settings, store: ArtifactStore, tmp_path: Path
) -> None:
    published = publish(store, tmp_path)
    keep = recorded_entry(settings, published)
    stale = recorded_entry(settings, published, device="NPU", version=OLDER, size=4096)
    # NPU blobs are written read-only; the directory is what decides removal.
    for blob in stale.glob("*.blob"):
        blob.chmod(0o440)
    root = settings.paths.compile_cache_dir
    (root / "notes").mkdir()
    outside = tmp_path / "outside" / "abcdefabcdefabcd"
    outside.mkdir(parents=True)
    (outside / "precious.bin").write_bytes(b"\x02" * 64)
    (root / "abcdefabcdefabcd").symlink_to(outside, target_is_directory=True)

    service = service_for(settings, store)
    plan = service.plan_prune(service.inventory())
    planned = plan.total_bytes

    preview = service.prune(plan, dry_run=True)
    assert preview.dry_run and preview.removed == [] and preview.freed_bytes == 0
    assert preview.planned_bytes == planned
    assert stale.is_dir()

    report = service.prune(plan)

    assert report.removed == [stale.name]
    assert report.failed == []
    assert report.freed_bytes == planned >= 4096
    assert not stale.exists()
    assert keep.is_dir()
    assert (root / "notes").is_dir()
    assert (outside / "precious.bin").is_file()
    assert published.is_dir()


def test_prune_all_empties_the_cache_but_keeps_what_is_not_an_entry(
    settings: Settings, store: ArtifactStore, tmp_path: Path
) -> None:
    published = publish(store, tmp_path)
    recorded_entry(settings, published)
    legacy_entry(settings, "0123456789abcdef")
    root = settings.paths.compile_cache_dir
    (root / "README").write_text("mine\n", encoding="utf-8")

    service = service_for(settings, store)
    report = service.prune(service.plan_prune(service.inventory(), everything=True))

    assert len(report.removed) == 2
    assert sorted(path.name for path in root.iterdir()) == ["README"]


def test_prune_checks_containment_again_before_deleting(
    settings: Settings, store: ArtifactStore, tmp_path: Path
) -> None:
    """A plan is not trusted: an entry outside the root is refused at delete time."""
    foreign = tmp_path / "foreign" / "0123456789abcdef"
    foreign.mkdir(parents=True)
    (foreign / "keep.bin").write_bytes(b"\x03" * 10)
    service = service_for(settings, store)
    plan = service.plan_prune(service.inventory())
    plan.entries.append(CompileCacheEntry(key=foreign.name, path=foreign, size_bytes=10))
    plan.reasons[foreign.name] = "all"

    report = service.prune(plan)

    assert report.removed == []
    assert [failure.key for failure in report.failed] == [foreign.name]
    assert report.freed_bytes == 0
    assert (foreign / "keep.bin").is_file()


def test_prune_never_follows_a_link_that_replaced_an_entry(
    settings: Settings, store: ArtifactStore, tmp_path: Path
) -> None:
    published = publish(store, tmp_path)
    entry = recorded_entry(settings, published, version=OLDER)
    service = service_for(settings, store)
    plan = service.plan_prune(service.inventory())
    assert [item.key for item in plan.entries] == [entry.name]

    # Between the plan and the deletion the entry becomes a link elsewhere.
    shutil.rmtree(entry)
    target = tmp_path / "victim"
    target.mkdir()
    (target / "data.bin").write_bytes(b"\x04" * 32)
    entry.symlink_to(target, target_is_directory=True)

    report = service.prune(plan)

    assert report.removed == []
    assert report.failed and "not a directory" in report.failed[0].error
    assert report.freed_bytes == 0
    assert (target / "data.bin").is_file()


# --- yonga remove reuses the records -----------------------------------------------------


def test_remove_finds_entries_built_by_any_runtime_through_their_record(
    settings: Settings, store: ArtifactStore, tmp_path: Path
) -> None:
    """The manifest cannot predict a future runtime's key; the record names the owner."""
    manifest = make_manifest()
    published = publish(store, tmp_path, manifest=manifest)
    upgraded = recorded_entry(settings, published, device="CPU", version="2027.1.0-1-abc")
    someone_else = recorded_entry(settings, tmp_path / "unrelated-artifact-dir", device="CPU")

    report = store.remove(manifest.artifact_id)

    assert upgraded in report.compile_cache_paths
    assert not upgraded.exists()
    assert someone_else.exists()


def test_remove_finds_a_legacy_entry_for_a_device_the_manifest_never_probed(
    settings: Settings, store: ArtifactStore, tmp_path: Path
) -> None:
    """``chat --device cpu`` on a GPU artifact made a CPU entry the manifest knows nothing of."""
    manifest = make_manifest(device="GPU")
    publish(store, tmp_path, manifest=manifest)
    cpu = legacy_entry(
        settings,
        compile_cache_key(
            artifact_hash=manifest.artifact_id, device="CPU", openvino_version=INSTALLED
        ),
    )

    report = store.remove(manifest.artifact_id, dry_run=True)

    assert cpu in report.compile_cache_paths
