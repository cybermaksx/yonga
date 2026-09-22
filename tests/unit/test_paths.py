"""Tests for the XDG directory layout."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from yonga.config.paths import Paths, default_paths


def test_defaults_without_xdg_variables(tmp_path: Path) -> None:
    home = tmp_path / "home"
    paths = default_paths({"HOME": str(home)})

    assert paths.config_dir == home / ".config" / "yonga"
    assert paths.cache_dir == home / ".cache" / "yonga"
    assert paths.data_dir == home / ".local" / "share" / "yonga"


def test_absolute_xdg_variables_win(tmp_path: Path) -> None:
    env = {
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "cfg"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
    }
    paths = default_paths(env)

    assert paths.config_dir == tmp_path / "cfg" / "yonga"
    assert paths.cache_dir == tmp_path / "cache" / "yonga"
    assert paths.data_dir == tmp_path / "data" / "yonga"


@pytest.mark.parametrize("bad", ["", "   ", "relative/dir", "./dir", "~/dir"])
def test_relative_or_empty_xdg_variables_are_ignored(tmp_path: Path, bad: str) -> None:
    home = tmp_path / "home"
    env = {
        "HOME": str(home),
        "XDG_CONFIG_HOME": bad,
        "XDG_CACHE_HOME": bad,
        "XDG_DATA_HOME": bad,
    }
    paths = default_paths(env)

    assert paths.config_dir == home / ".config" / "yonga"
    assert paths.cache_dir == home / ".cache" / "yonga"
    assert paths.data_dir == home / ".local" / "share" / "yonga"


def test_relative_home_falls_back_to_real_home() -> None:
    paths = default_paths({"HOME": "relative/home"})

    assert paths.config_dir.is_absolute()
    assert paths.config_dir == Path.home() / ".config" / "yonga"


def test_explicit_overrides_beat_xdg(tmp_path: Path) -> None:
    env = {"HOME": str(tmp_path), "XDG_DATA_HOME": str(tmp_path / "xdg")}
    paths = default_paths(env, data_dir=tmp_path / "explicit")

    assert paths.data_dir == tmp_path / "explicit"
    assert paths.cache_dir == tmp_path / ".cache" / "yonga"


def test_derived_paths(tmp_path: Path) -> None:
    paths = Paths(
        config_dir=tmp_path / "cfg",
        cache_dir=tmp_path / "cache",
        data_dir=tmp_path / "data",
    )

    assert paths.config_file == tmp_path / "cfg" / "config.toml"
    assert paths.registry_overrides_dir == tmp_path / "cfg" / "registry-overrides"
    assert paths.tmp_dir == tmp_path / "cache" / "tmp"
    assert paths.compile_cache_dir == tmp_path / "cache" / "compile"
    assert paths.metadata_cache_dir == tmp_path / "cache" / "metadata"
    assert paths.artifacts_dir == tmp_path / "data" / "artifacts"
    assert paths.staging_dir == tmp_path / "data" / "artifacts" / ".staging"
    assert paths.benchmarks_dir == tmp_path / "data" / "benchmarks"
    assert paths.logs_dir == tmp_path / "data" / "logs"
    assert paths.chat_history_dir == tmp_path / "data" / "chat-history"


def test_job_paths(tmp_path: Path) -> None:
    paths = Paths(
        config_dir=tmp_path / "cfg",
        cache_dir=tmp_path / "cache",
        data_dir=tmp_path / "data",
    )
    job_id = "0b3d7a1e"

    assert paths.job_tmp_dir(job_id) == paths.tmp_dir / job_id
    assert paths.job_staging_dir(job_id) == paths.staging_dir / job_id
    assert paths.job_log_path(job_id) == paths.logs_dir / f"{job_id}.jsonl"
    assert paths.job_state_path(job_id) == paths.logs_dir / f"{job_id}.state.json"


@pytest.mark.parametrize("job_id", ["", ".", "..", "a/b", "a\\b"])
def test_job_id_traversal_is_rejected(tmp_path: Path, job_id: str) -> None:
    paths = Paths(
        config_dir=tmp_path / "cfg",
        cache_dir=tmp_path / "cache",
        data_dir=tmp_path / "data",
    )

    with pytest.raises(ValueError, match="Invalid job id"):
        paths.job_tmp_dir(job_id)


def test_ensure_creates_every_managed_directory(tmp_path: Path) -> None:
    paths = default_paths({"HOME": str(tmp_path)})
    paths.ensure()

    for directory in paths.managed_dirs:
        assert directory.is_dir()
    paths.ensure()  # idempotent


def test_ensure_never_creates_a_hugging_face_cache(tmp_path: Path) -> None:
    paths = default_paths({"HOME": str(tmp_path)})
    paths.ensure()

    assert not (tmp_path / ".cache" / "huggingface").exists()


def test_paths_is_frozen(tmp_path: Path) -> None:
    paths = default_paths({"HOME": str(tmp_path)})

    with pytest.raises(dataclasses.FrozenInstanceError):
        paths.config_dir = tmp_path  # type: ignore[misc]
