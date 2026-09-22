"""Unit tests for the filesystem probe, the temp/disk planner and finalization."""

from __future__ import annotations

import errno
import os
import pathlib
from pathlib import Path

import pytest

from yonga.adapters.filesystem import (
    INCOMPLETE_MARKER_NAME,
    FilesystemAdapter,
    finalize_directory,
    format_bytes,
    is_tool_owned_incomplete,
    mark_incomplete,
    plan_disk,
    plan_temp,
    required_temp_bytes,
    system_temp_dir,
)
from yonga.config.settings import load_settings
from yonga.domain.artifacts import manifest_path
from yonga.domain.diagnostics import ConfigurationError, ValidationError
from yonga.domain.hardware import FilesystemInfo

GIB = 2**30

# The exact acceptance fixture from HOST_FACTS.md / docs/TESTING.md.
SOURCE_WEIGHT_BYTES = 18_819_721_040
EXPECTED_REQUIRED_TEMP = 47_049_302_600
EXPECTED_SAFETY_MARGIN = 5_368_709_120
TMPFS_FREE = 16 * GIB
ROOT_FREE = 750 * GIB


def make_settings(tmp_path: Path):
    """Build deterministic settings that ignore the developer's own config."""
    return load_settings(
        config_file=tmp_path / "absent-config.toml",
        env={},
        overrides={"cache_dir": tmp_path / "cache", "data_dir": tmp_path / "data"},
    )


class FakeProbe:
    """A ``FilesystemProbe`` that answers from a table keyed by path prefix."""

    def __init__(self, table: dict[Path, FilesystemInfo]) -> None:
        self.table = table
        self.calls: list[Path] = []

    def usage(self, path: Path) -> FilesystemInfo:
        self.calls.append(Path(path))
        best: FilesystemInfo | None = None
        best_len = -1
        for prefix, info in self.table.items():
            matched = path == prefix or Path(path).is_relative_to(prefix)
            if matched and len(prefix.parts) > best_len:
                best = info
                best_len = len(prefix.parts)
        if best is None:
            return FilesystemInfo(path=Path(path))
        return best.model_copy(update={"path": Path(path)})


def tmpfs_and_root(managed_root: Path, *, root_free: int = ROOT_FREE) -> FakeProbe:
    """The docs/TESTING.md fixture: 16 GiB tmpfs /tmp, a large ext4 root."""
    return FakeProbe(
        {
            Path("/"): FilesystemInfo(
                path=Path("/"),
                mount_point=Path("/"),
                fstype="ext4",
                total_bytes=935 * GIB,
                free_bytes=root_free,
                available_bytes=root_free,
            ),
            Path(system_temp_dir()): FilesystemInfo(
                path=system_temp_dir(),
                mount_point=system_temp_dir(),
                fstype="tmpfs",
                total_bytes=TMPFS_FREE,
                free_bytes=TMPFS_FREE,
                available_bytes=TMPFS_FREE,
            ),
            managed_root: FilesystemInfo(
                path=managed_root,
                mount_point=Path("/"),
                fstype="ext4",
                total_bytes=935 * GIB,
                free_bytes=root_free,
                available_bytes=root_free,
            ),
        }
    )


# --- mount table parsing ------------------------------------------------------------


def write_mounts(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "mounts"
    path.write_text(body, encoding="utf-8")
    return path


def test_mounts_parses_fstypes_and_escapes(tmp_path):
    mounts = write_mounts(
        tmp_path,
        "tmpfs /run tmpfs rw,nosuid 0 0\n"
        "/dev/mapper/vg-root / ext4 rw,relatime 0 0\n"
        "tmpfs /tmp tmpfs rw,nosuid,size=16163408k 0 0\n"
        "server:/x /mnt/my\\040share nfs4 rw 0 0\n"
        "\n"
        "# comment line\n"
        "short line\n",
    )
    adapter = FilesystemAdapter(mounts_files=[mounts])
    entries = adapter.mounts()
    points = {str(entry.mount_point): entry.fstype for entry in entries}
    assert points["/"] == "ext4"
    assert points["/tmp"] == "tmpfs"
    assert points["/mnt/my share"] == "nfs4"


def test_mount_for_picks_the_longest_match(tmp_path):
    mounts = write_mounts(
        tmp_path,
        "/dev/root / ext4 rw 0 0\ntmpfs /tmp tmpfs rw 0 0\n/dev/sdb /tmp/deep xfs rw 0 0\n",
    )
    adapter = FilesystemAdapter(mounts_files=[mounts])
    assert adapter.mount_for(Path("/tmp/deep/inner/file")).fstype == "xfs"
    assert adapter.mount_for(Path("/tmp/other")).fstype == "tmpfs"
    assert adapter.mount_for(Path("/var/lib")).fstype == "ext4"


def test_mounts_falls_back_to_the_second_file(tmp_path):
    second = write_mounts(tmp_path, "/dev/root / ext4 rw 0 0\n")
    adapter = FilesystemAdapter(mounts_files=[tmp_path / "missing", second])
    assert [entry.fstype for entry in adapter.mounts()] == ["ext4"]


def test_mounts_missing_everywhere_is_empty(tmp_path):
    adapter = FilesystemAdapter(mounts_files=[tmp_path / "nope"])
    assert adapter.mounts() == []


# --- usage --------------------------------------------------------------------------


def test_usage_reports_real_statistics_for_an_existing_path(tmp_path):
    info = FilesystemAdapter().usage(tmp_path)
    assert info.total_bytes > 0
    assert info.available_bytes > 0
    assert info.mount_point is not None


def test_usage_walks_up_to_the_nearest_existing_ancestor(tmp_path):
    missing = tmp_path / "a" / "b" / "c" / "not-created-yet"
    reading = FilesystemAdapter().describe(missing)
    assert reading.info.path == missing
    assert reading.info.total_bytes > 0
    assert any("does not exist yet" in note for note in reading.notes)


def test_statvfs_failure_is_a_warning_not_a_crash(tmp_path, monkeypatch):
    def boom(_path):
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(os, "statvfs", boom)
    reading = FilesystemAdapter().describe(tmp_path)
    assert reading.info.total_bytes == 0
    assert reading.info.free_bytes == 0
    assert reading.info.available_bytes == 0
    assert any("filesystem statistics unavailable" in note for note in reading.notes)


def test_usage_of_an_unrooted_path_still_returns_a_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    info = FilesystemAdapter().usage(Path("relative/child"))
    assert info.path.is_absolute()


def test_is_tmpfs_flows_through_from_the_mount_table(tmp_path):
    mounts = write_mounts(tmp_path, f"tmpfs {tmp_path} tmpfs rw 0 0\n")
    info = FilesystemAdapter(mounts_files=[mounts]).usage(tmp_path / "child")
    assert info.fstype == "tmpfs"
    assert info.is_tmpfs
    assert info.is_ram_backed


# --- temp planner -------------------------------------------------------------------


def test_required_temp_uses_the_spec_formula(tmp_path):
    settings = make_settings(tmp_path)
    assert required_temp_bytes(SOURCE_WEIGHT_BYTES, settings) == EXPECTED_REQUIRED_TEMP
    # A tiny model still reserves the configured floor.
    assert required_temp_bytes(1024, settings) == settings.temp_min_bytes


def test_tmpfs_fixture_chooses_the_managed_disk_temp(tmp_path):
    settings = make_settings(tmp_path)
    job_temp = settings.paths.job_tmp_dir("job-1")
    probe = tmpfs_and_root(settings.paths.cache_dir)

    plan = plan_temp(
        job_temp_dir=job_temp,
        source_weight_bytes=SOURCE_WEIGHT_BYTES,
        settings=settings,
        probe=probe,
    )

    assert plan.required_bytes == EXPECTED_REQUIRED_TEMP
    assert plan.safety_margin_bytes == EXPECTED_SAFETY_MARGIN
    assert plan.total_required_bytes == EXPECTED_REQUIRED_TEMP + EXPECTED_SAFETY_MARGIN
    assert plan.sufficient is True
    assert plan.system_temp_would_fail is True
    assert plan.job_temp_dir == job_temp
    assert plan.temp_filesystem.fstype == "ext4"
    assert plan.system_temp is not None
    assert plan.system_temp.is_tmpfs
    assert plan.env() == {
        "TMPDIR": str(job_temp),
        "TMP": str(job_temp),
        "TEMP": str(job_temp),
    }


def test_tmpfs_fixture_never_advises_enlarging_tmpfs(tmp_path):
    settings = make_settings(tmp_path)
    plan = plan_temp(
        job_temp_dir=settings.paths.job_tmp_dir("job-1"),
        source_weight_bytes=SOURCE_WEIGHT_BYTES,
        settings=settings,
        probe=tmpfs_and_root(settings.paths.cache_dir),
    )
    joined = " ".join(plan.notes).lower()
    assert "managed disk-backed temporary directory" in joined
    assert "warning, not an error" in joined
    for forbidden in ("enlarge", "increase the size of /tmp", "remount", "mount -o size"):
        assert forbidden not in joined


def test_insufficient_cache_disk_is_not_sufficient(tmp_path):
    settings = make_settings(tmp_path)
    plan = plan_temp(
        job_temp_dir=settings.paths.job_tmp_dir("job-1"),
        source_weight_bytes=SOURCE_WEIGHT_BYTES,
        settings=settings,
        probe=tmpfs_and_root(settings.paths.cache_dir, root_free=10 * GIB),
    )
    assert plan.sufficient is False
    assert plan.system_temp_would_fail is True
    assert any("Not enough space" in note for note in plan.notes)


def test_plan_temp_survives_a_probe_that_raises(tmp_path):
    class Exploding:
        def usage(self, path: Path) -> FilesystemInfo:
            raise OSError("no")

    settings = make_settings(tmp_path)
    plan = plan_temp(
        job_temp_dir=settings.paths.job_tmp_dir("job-1"),
        source_weight_bytes=SOURCE_WEIGHT_BYTES,
        settings=settings,
        probe=Exploding(),
    )
    assert plan.sufficient is False
    assert any("filesystem statistics unavailable" in note for note in plan.notes)


def test_plan_temp_clamps_a_negative_source_size(tmp_path):
    settings = make_settings(tmp_path)
    plan = plan_temp(
        job_temp_dir=settings.paths.job_tmp_dir("job-1"),
        source_weight_bytes=-5,
        settings=settings,
        probe=tmpfs_and_root(settings.paths.cache_dir),
    )
    assert plan.required_bytes == settings.temp_min_bytes


# --- disk planner -------------------------------------------------------------------


def test_plan_disk_combines_temp_artifact_and_download(tmp_path):
    settings = make_settings(tmp_path)
    plan = plan_disk(
        job_temp_dir=settings.paths.job_tmp_dir("job-1"),
        artifact_dir=settings.paths.artifacts_dir / "model",
        source_weight_bytes=SOURCE_WEIGHT_BYTES,
        settings=settings,
        expected_download_bytes=18_842_727_977,
        estimated_artifact_bytes=6 * GIB,
        probe=tmpfs_and_root(tmp_path),
    )
    assert plan.temp.sufficient is True
    assert plan.artifact_sufficient is True
    assert plan.sufficient is True
    assert plan.expected_download_bytes == 18_842_727_977
    assert plan.estimated_artifact_bytes == 6 * GIB
    assert plan.artifact_filesystem.fstype == "ext4"


def test_plan_disk_reports_an_artifact_filesystem_that_is_too_small(tmp_path):
    settings = make_settings(tmp_path)
    plan = plan_disk(
        job_temp_dir=settings.paths.job_tmp_dir("job-1"),
        artifact_dir=settings.paths.artifacts_dir / "model",
        source_weight_bytes=SOURCE_WEIGHT_BYTES,
        settings=settings,
        expected_download_bytes=SOURCE_WEIGHT_BYTES,
        estimated_artifact_bytes=50 * GIB,
        probe=tmpfs_and_root(tmp_path, root_free=60 * GIB),
    )
    assert plan.temp.sufficient is True
    assert plan.artifact_sufficient is False
    assert plan.sufficient is False
    assert any("Free space on the artifact filesystem" in note for note in plan.notes)


def test_plan_disk_defaults_the_artifact_estimate_to_the_source_size(tmp_path):
    settings = make_settings(tmp_path)
    plan = plan_disk(
        job_temp_dir=settings.paths.job_tmp_dir("job-1"),
        artifact_dir=settings.paths.artifacts_dir / "model",
        source_weight_bytes=SOURCE_WEIGHT_BYTES,
        settings=settings,
        probe=tmpfs_and_root(tmp_path),
    )
    assert plan.estimated_artifact_bytes == SOURCE_WEIGHT_BYTES


def test_plan_disk_does_not_charge_a_download_on_another_filesystem(tmp_path):
    settings = make_settings(tmp_path)
    artifacts = settings.paths.artifacts_dir / "model"
    downloads = tmp_path / "elsewhere" / "hub"
    probe = FakeProbe(
        {
            Path("/"): FilesystemInfo(
                path=Path("/"), mount_point=Path("/"), fstype="ext4", free_bytes=ROOT_FREE
            ),
            system_temp_dir(): FilesystemInfo(
                path=system_temp_dir(),
                mount_point=system_temp_dir(),
                fstype="tmpfs",
                free_bytes=TMPFS_FREE,
            ),
            settings.paths.cache_dir: FilesystemInfo(
                path=settings.paths.cache_dir,
                mount_point=Path("/"),
                fstype="ext4",
                free_bytes=ROOT_FREE,
            ),
            artifacts: FilesystemInfo(
                path=artifacts,
                mount_point=Path("/data"),
                fstype="ext4",
                free_bytes=40 * GIB,
            ),
            downloads: FilesystemInfo(
                path=downloads,
                mount_point=Path("/hub"),
                fstype="btrfs",
                free_bytes=200 * GIB,
            ),
        }
    )
    plan = plan_disk(
        job_temp_dir=settings.paths.job_tmp_dir("job-1"),
        artifact_dir=artifacts,
        source_weight_bytes=SOURCE_WEIGHT_BYTES,
        settings=settings,
        expected_download_bytes=SOURCE_WEIGHT_BYTES,
        estimated_artifact_bytes=6 * GIB,
        download_dir=downloads,
        probe=probe,
    )
    assert plan.artifact_sufficient is True
    assert any("different filesystem" in note for note in plan.notes)


# --- finalization -------------------------------------------------------------------


def build_staging(root: Path, name: str = "staging") -> Path:
    staging = root / name
    (staging / "sub").mkdir(parents=True)
    (staging / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    (staging / "sub" / "extra.bin").write_bytes(b"\x00\x01")
    mark_incomplete(staging, job_id="job-1")
    return staging


def test_finalize_moves_the_directory_and_clears_the_marker(tmp_path):
    staging = build_staging(tmp_path)
    final = tmp_path / "artifacts" / "model"

    result = finalize_directory(staging, final)

    assert result == final
    assert not staging.exists()
    assert (final / "openvino_model.xml").read_text(encoding="utf-8") == "<net/>"
    assert (final / "sub" / "extra.bin").read_bytes() == b"\x00\x01"
    assert not (final / INCOMPLETE_MARKER_NAME).exists()


def test_finalize_refuses_to_clobber_a_foreign_directory(tmp_path):
    staging = build_staging(tmp_path)
    final = tmp_path / "artifacts" / "model"
    final.mkdir(parents=True)
    (final / "my-own-notes.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(ConfigurationError) as excinfo:
        finalize_directory(staging, final)

    assert excinfo.value.diagnostic.code == "artifact.finalize.refused"
    assert (final / "my-own-notes.txt").read_text(encoding="utf-8") == "keep me"
    assert staging.exists()


def test_finalize_refuses_to_clobber_a_finished_artifact(tmp_path):
    staging = build_staging(tmp_path)
    final = tmp_path / "artifacts" / "model"
    manifest = manifest_path(final)
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")

    assert is_tool_owned_incomplete(final) is False
    with pytest.raises(ConfigurationError):
        finalize_directory(staging, final)


def test_finalize_replaces_a_tool_owned_incomplete_directory(tmp_path):
    staging = build_staging(tmp_path)
    final = tmp_path / "artifacts" / "model"
    mark_incomplete(final, job_id="older-job")
    (final / "half-written.bin").write_bytes(b"junk")

    assert is_tool_owned_incomplete(final) is True
    assert finalize_directory(staging, final) == final
    assert not (final / "half-written.bin").exists()
    assert (final / "openvino_model.xml").exists()
    assert not any(entry.name.startswith(".yonga-replaced-") for entry in final.parent.iterdir())


def test_finalize_replaces_an_empty_directory(tmp_path):
    staging = build_staging(tmp_path)
    final = tmp_path / "artifacts" / "model"
    final.mkdir(parents=True)
    assert finalize_directory(staging, final) == final
    assert (final / "openvino_model.xml").exists()


def test_finalize_force_replaces_a_foreign_directory(tmp_path):
    staging = build_staging(tmp_path)
    final = tmp_path / "artifacts" / "model"
    final.mkdir(parents=True)
    (final / "someone-elses.txt").write_text("gone", encoding="utf-8")

    notes: list[str] = []
    assert finalize_directory(staging, final, force=True, notes=notes) == final
    assert not (final / "someone-elses.txt").exists()
    assert (final / "openvino_model.xml").exists()
    assert any("forced" in note for note in notes)


def test_finalize_missing_staging_raises_a_typed_error(tmp_path):
    with pytest.raises(ValidationError) as excinfo:
        finalize_directory(tmp_path / "nothing", tmp_path / "final")
    assert excinfo.value.diagnostic.code == "artifact.finalize.missing_staging"


def test_finalize_falls_back_to_copy_across_filesystems(tmp_path, monkeypatch):
    staging = build_staging(tmp_path)
    final = tmp_path / "artifacts" / "model"
    original_replace = pathlib.Path.replace

    def fake_replace(self, target):
        if self == staging:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return original_replace(self, target)

    monkeypatch.setattr(pathlib.Path, "replace", fake_replace)

    notes: list[str] = []
    assert finalize_directory(staging, final, notes=notes) == final
    assert (final / "sub" / "extra.bin").read_bytes() == b"\x00\x01"
    assert not staging.exists()
    assert any("different filesystems" in note for note in notes)
    assert not any(entry.name.startswith(".yonga-incoming-") for entry in final.parent.iterdir())


def test_finalize_is_a_no_op_when_staging_is_already_final(tmp_path):
    staging = build_staging(tmp_path)
    notes: list[str] = []
    assert finalize_directory(staging, staging, notes=notes) == staging
    assert not (staging / INCOMPLETE_MARKER_NAME).exists()
    assert notes


def test_is_tool_owned_incomplete_rules(tmp_path):
    assert is_tool_owned_incomplete(tmp_path / "absent") is True

    empty = tmp_path / "empty"
    empty.mkdir()
    assert is_tool_owned_incomplete(empty) is True

    marked = tmp_path / "marked"
    mark_incomplete(marked)
    (marked / "partial.bin").write_bytes(b"x")
    assert is_tool_owned_incomplete(marked) is True

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("hi", encoding="utf-8")
    assert is_tool_owned_incomplete(foreign) is False

    a_file = tmp_path / "a-file"
    a_file.write_text("x", encoding="utf-8")
    assert is_tool_owned_incomplete(a_file) is False


# --- misc ---------------------------------------------------------------------------


def test_format_bytes_is_readable():
    assert format_bytes(0) == "0 B"
    assert format_bytes(1536) == "1.5 KiB"
    assert format_bytes(20 * GIB) == "20.0 GiB"


def test_system_temp_dir_is_absolute():
    assert system_temp_dir().is_absolute()
