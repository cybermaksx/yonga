"""XDG-compliant filesystem layout for every directory the tool owns.

Nothing in this module touches the Hugging Face cache: yonga references Hub
snapshots, it never manages them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from yonga import APP_SLUG

XDG_CONFIG_HOME: str = "XDG_CONFIG_HOME"
XDG_CACHE_HOME: str = "XDG_CACHE_HOME"
XDG_DATA_HOME: str = "XDG_DATA_HOME"

DEFAULT_CONFIG_HOME: str = ".config"
DEFAULT_CACHE_HOME: str = ".cache"
DEFAULT_DATA_HOME: str = ".local/share"

CONFIG_FILENAME: str = "config.toml"
REGISTRY_OVERRIDES_DIRNAME: str = "registry-overrides"

_JOB_ID_FORBIDDEN: frozenset[str] = frozenset({"", ".", ".."})


def _home_dir(env: Mapping[str, str]) -> Path:
    """Return the user's home directory, preferring an absolute ``$HOME``."""
    raw = env.get("HOME", "")
    if raw:
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate
    return Path.home()


def _xdg_dir(env: Mapping[str, str], variable: str, default_relative: str) -> Path:
    """Resolve one XDG base directory.

    Per the XDG Base Directory Specification a variable that is unset, empty or
    not an absolute path must be treated as if it were unset, so relative values
    silently fall back to the specified default.
    """
    raw = env.get(variable, "")
    if raw:
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate
    return _home_dir(env) / default_relative


def _validate_job_id(job_id: str) -> str:
    """Reject job identifiers that could escape the managed directories."""
    if job_id in _JOB_ID_FORBIDDEN or "/" in job_id or "\\" in job_id or "\x00" in job_id:
        message = f"Invalid job id: {job_id!r}"
        raise ValueError(message)
    return job_id


@dataclass(frozen=True)
class Paths:
    """Every managed directory, derived from the three XDG base directories."""

    config_dir: Path
    cache_dir: Path
    data_dir: Path

    # --- configuration -------------------------------------------------
    @property
    def config_file(self) -> Path:
        """Path of the user TOML configuration file."""
        return self.config_dir / CONFIG_FILENAME

    @property
    def registry_overrides_dir(self) -> Path:
        """Directory holding user-supplied compatibility registry overrides."""
        return self.config_dir / REGISTRY_OVERRIDES_DIRNAME

    @property
    def registry_keys_dir(self) -> Path:
        """Public keys the user trusts to sign remote registry bundles.

        Configuration, not data: trusting a key is a decision the user makes by
        putting a file here, and nothing yonga downloads can add one.
        """
        return self.config_dir / "registry-keys"

    @property
    def remote_registry_dir(self) -> Path:
        """The verified remote registry bundle, re-verified every time it is loaded."""
        return self.data_dir / "registry-remote"

    # --- cache ---------------------------------------------------------
    @property
    def tmp_dir(self) -> Path:
        """Disk-backed working directory used instead of a possibly tmpfs ``/tmp``."""
        return self.cache_dir / "tmp"

    @property
    def compile_cache_dir(self) -> Path:
        """OpenVINO compile cache root, keyed per artifact/device by the caller."""
        return self.cache_dir / "compile"

    @property
    def metadata_cache_dir(self) -> Path:
        """Cache of cheap Hub metadata lookups."""
        return self.cache_dir / "metadata"

    @property
    def envs_dir(self) -> Path:
        """Managed export environments, one virtual environment per package set."""
        return self.cache_dir / "envs"

    # --- data ----------------------------------------------------------
    @property
    def artifacts_dir(self) -> Path:
        """Root of the prepared OpenVINO artifacts."""
        return self.data_dir / "artifacts"

    @property
    def staging_dir(self) -> Path:
        """Hidden staging root; incomplete builds live here until validated."""
        return self.artifacts_dir / ".staging"

    @property
    def benchmarks_dir(self) -> Path:
        """Directory holding saved benchmark result documents."""
        return self.data_dir / "benchmarks"

    @property
    def logs_dir(self) -> Path:
        """Directory holding per-job JSON Lines logs and job state files."""
        return self.data_dir / "logs"

    @property
    def chat_history_dir(self) -> Path:
        """Directory holding persisted chat transcripts and prompt history."""
        return self.data_dir / "chat-history"

    # --- per-job paths -------------------------------------------------
    def job_tmp_dir(self, job_id: str) -> Path:
        """Return the managed temporary directory for one preparation job."""
        return self.tmp_dir / _validate_job_id(job_id)

    def job_staging_dir(self, job_id: str) -> Path:
        """Return the staging directory an in-progress artifact is built in."""
        return self.staging_dir / _validate_job_id(job_id)

    def job_log_path(self, job_id: str) -> Path:
        """Return the JSON Lines log path for one job."""
        return self.logs_dir / f"{_validate_job_id(job_id)}.jsonl"

    def job_state_path(self, job_id: str) -> Path:
        """Return the small JSON job-state path used to diagnose interrupted jobs."""
        return self.logs_dir / f"{_validate_job_id(job_id)}.state.json"

    # --- lifecycle -----------------------------------------------------
    @property
    def managed_dirs(self) -> tuple[Path, ...]:
        """Every directory ``ensure()`` creates, in creation order."""
        return (
            self.config_dir,
            self.registry_overrides_dir,
            self.cache_dir,
            self.tmp_dir,
            self.compile_cache_dir,
            self.metadata_cache_dir,
            self.data_dir,
            self.artifacts_dir,
            self.staging_dir,
            self.benchmarks_dir,
            self.logs_dir,
            self.chat_history_dir,
        )

    def ensure(self) -> None:
        """Create every managed directory; never touches the Hugging Face cache."""
        for directory in self.managed_dirs:
            directory.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, str]:
        """Return the managed layout as plain strings, for diagnostics output."""
        return {
            "config_dir": str(self.config_dir),
            "cache_dir": str(self.cache_dir),
            "data_dir": str(self.data_dir),
            "config_file": str(self.config_file),
            "registry_overrides_dir": str(self.registry_overrides_dir),
            "registry_keys_dir": str(self.registry_keys_dir),
            "remote_registry_dir": str(self.remote_registry_dir),
            "tmp_dir": str(self.tmp_dir),
            "compile_cache_dir": str(self.compile_cache_dir),
            "metadata_cache_dir": str(self.metadata_cache_dir),
            "artifacts_dir": str(self.artifacts_dir),
            "staging_dir": str(self.staging_dir),
            "benchmarks_dir": str(self.benchmarks_dir),
            "logs_dir": str(self.logs_dir),
            "chat_history_dir": str(self.chat_history_dir),
        }


def default_paths(
    env: Mapping[str, str] | None = None,
    *,
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
    config_dir: Path | None = None,
) -> Paths:
    """Build :class:`Paths` from the environment, with optional explicit overrides.

    ``env`` defaults to ``os.environ``. Explicit arguments win over XDG variables,
    are expanded (``~``) and made absolute against the current directory: the
    exporter and other child processes run elsewhere, and a relative path would
    point them at a different place than yonga itself.
    """
    if env is None:
        env = os.environ

    resolved_config = (
        config_dir.expanduser().absolute()
        if config_dir is not None
        else _xdg_dir(env, XDG_CONFIG_HOME, DEFAULT_CONFIG_HOME) / APP_SLUG
    )
    resolved_cache = (
        cache_dir.expanduser().absolute()
        if cache_dir is not None
        else _xdg_dir(env, XDG_CACHE_HOME, DEFAULT_CACHE_HOME) / APP_SLUG
    )
    resolved_data = (
        data_dir.expanduser().absolute()
        if data_dir is not None
        else _xdg_dir(env, XDG_DATA_HOME, DEFAULT_DATA_HOME) / APP_SLUG
    )
    return Paths(
        config_dir=resolved_config,
        cache_dir=resolved_cache,
        data_dir=resolved_data,
    )
