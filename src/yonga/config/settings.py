"""User configuration with a deterministic precedence chain.

Precedence, highest first: explicit ``overrides`` (CLI options), ``YONGA_*``
environment variables, the user TOML configuration file, field defaults.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, PrivateAttr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from yonga import ENV_PREFIX
from yonga.config.paths import Paths, default_paths
from yonga.util.jsonlog import DEBUG_ENV_VAR
from yonga.util.redaction import SECRET_ENV_KEYS, redact_mapping

GIB: int = 2**30

CONFIG_TABLE_NAME: str = "yonga"
"""Optional TOML table name; top-level keys are accepted as well."""

NON_SETTING_ENV_KEYS: frozenset[str] = frozenset({DEBUG_ENV_VAR})
"""Prefixed environment variables that are not settings and must not warn."""

TELEMETRY_REJECTED_MESSAGE: str = (
    "Telemetry is not implemented and must stay disabled: remove 'telemetry' "
    "from the configuration or set it to false."
)


class SettingsConfigurationError(ValueError):
    """Fallback raised when ``yonga.domain.diagnostics`` cannot be imported.

    The diagnostics layer is the canonical home of ``ConfigurationError``; this
    class only exists so that configuration loading still fails loudly (and with
    the same message) in environments where the domain package is unavailable.
    """


class DevicePolicy(str, Enum):
    """Device selection policy requested by the user."""

    AUTO = "auto"
    NPU = "npu"
    GPU = "gpu"
    CPU = "cpu"


def _configuration_error(
    title: str,
    message: str,
    *,
    code: str = "config.invalid",
    remediation: Sequence[str] = (),
    context: Mapping[str, str] | None = None,
    cause: BaseException | None = None,
) -> Exception:
    """Build a ``ConfigurationError`` carrying a diagnostic, or a local fallback."""
    try:
        from yonga.domain.diagnostics import ConfigurationError, Diagnostic, FailureCategory
    except ImportError:
        return SettingsConfigurationError(message)

    diagnostic = Diagnostic(
        code=code,
        category=FailureCategory.CONFIGURATION_ERROR,
        title=title,
        message=message,
        remediation=list(remediation),
        context=dict(context or {}),
    )
    return ConfigurationError(diagnostic, cause=cause)


def _is_true(value: object) -> bool:
    """Interpret a TOML/env/CLI value as a boolean without raising."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


class Settings(BaseSettings):
    """Resolved user configuration for a single yonga invocation."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        extra="ignore",
        frozen=False,
        validate_assignment=False,
        case_sensitive=False,
    )

    # --- directories ---------------------------------------------------
    config_dir: Path | None = None
    cache_dir: Path | None = None
    data_dir: Path | None = None
    hf_home: Path | None = None

    # --- behaviour -----------------------------------------------------
    default_device: DevicePolicy = DevicePolicy.AUTO
    offline: bool = False
    download_confirm_threshold_bytes: int = Field(default=5 * GIB, ge=0)
    temp_min_bytes: int = Field(default=20 * GIB, ge=0)
    temp_size_multiplier: float = Field(default=2.5, gt=0.0)
    temp_safety_margin_bytes: int = Field(default=5 * GIB, ge=0)
    compile_cache_enabled: bool = True
    managed_export_envs: bool = True
    """Build an isolated export environment when the installed packages cannot
    satisfy an architecture's exporter requirements, instead of refusing.

    The environment lives under the cache directory and is used only by the
    export subprocess; the user's own environment is never modified. Creating
    one is confirmed in the preparation plan. ``--no-managed-env`` or
    ``managed_export_envs = false`` restores the refusal.
    """
    workspace_retention_hours: int = Field(default=72, ge=0)
    """How long a stale job working directory is kept before a sweep retires it.

    Applies to both managed working roots: ``artifacts/.staging/<job-id>`` (a
    partial export left by a failed preparation) and ``cache/tmp/<job-id>`` (the
    managed ``TMPDIR``). A directory belonging to a job that is still running is
    never swept, whatever this says. ``0`` retires every stale directory as soon
    as a job opens.
    """
    workspace_retention_max: int = Field(default=3, ge=0)
    """How many stale working directories survive the sweep, newest first.

    The count bound and :attr:`workspace_retention_hours` are both enforced, so a
    burst of failures cannot fill the disk while the window is still open. Only
    directories that actually hold a partial export count; empty husks are always
    retired.
    """
    show_reasoning: bool = False
    hash_weights: bool = True
    bench_runs: int = Field(default=3, ge=1)
    bench_warmups: int = Field(default=1, ge=0)
    max_new_tokens: int = Field(default=4096, ge=1)
    registry_update_policy: Literal["never", "manual"] = "manual"
    telemetry: bool = False
    json_output: bool = False
    verbose: bool = False
    debug: bool = False

    # --- load-time diagnostics -----------------------------------------
    config_warnings: list[str] = Field(default_factory=list, exclude=True)
    """Non-fatal problems found while loading (unknown keys, unreadable file)."""

    _env: dict[str, str] | None = PrivateAttr(default=None)
    _config_file: Path | None = PrivateAttr(default=None)

    @field_validator("default_device", mode="before")
    @classmethod
    def _normalise_device(cls, value: object) -> object:
        """Accept ``NPU``/``Npu``/``npu`` spellings for the device policy."""
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("registry_update_policy", mode="before")
    @classmethod
    def _normalise_registry_policy(cls, value: object) -> object:
        """Accept any casing for the registry update policy."""
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("config_dir", "cache_dir", "data_dir", "hf_home", mode="before")
    @classmethod
    def _blank_path_is_unset(cls, value: object) -> object:
        """Treat an empty string (a common empty env var) as 'not configured'."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _finalise(self) -> Settings:
        """Expand user paths and enforce the telemetry and consistency rules."""
        if self.telemetry:
            raise ValueError(TELEMETRY_REJECTED_MESSAGE)
        for name in ("config_dir", "cache_dir", "data_dir", "hf_home"):
            current: Path | None = getattr(self, name)
            if current is not None:
                setattr(self, name, current.expanduser())
        if self.debug and not self.verbose:
            self.verbose = True
        return self

    @property
    def paths(self) -> Paths:
        """Managed directory layout implied by this configuration."""
        return default_paths(
            self._env,
            config_dir=self.config_dir,
            cache_dir=self.cache_dir,
            data_dir=self.data_dir,
        )

    @property
    def config_file(self) -> Path | None:
        """The TOML file these settings were loaded from, when there was one."""
        return self._config_file

    def redacted_dump(self) -> dict[str, object]:
        """JSON-safe dump of the configuration with any secret-ish value masked."""
        data: dict[str, object] = self.model_dump(mode="json")
        data["config_warnings"] = list(self.config_warnings)
        data["config_file"] = str(self._config_file) if self._config_file else None
        return redact_mapping(data)


def _user_config_keys() -> frozenset[str]:
    """Names accepted in the TOML file, environment and CLI overrides."""
    return frozenset(Settings.model_fields) - {"config_warnings"}


def _read_toml(path: Path, warnings: list[str]) -> dict[str, object]:
    """Read one configuration file; a missing file yields an empty mapping."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        warnings.append(f"Configuration file {path} could not be read: {exc.strerror or exc}")
        return {}

    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise _configuration_error(
            "Invalid configuration file",
            f"{path} is not valid TOML: {exc}",
            code="config.toml.invalid",
            remediation=[f"Fix the syntax error in {path} or delete the file."],
            cause=exc,
        ) from exc

    known = _user_config_keys()
    values: dict[str, object] = {}
    table = document.get(CONFIG_TABLE_NAME)
    for key, value in document.items():
        if key == CONFIG_TABLE_NAME and isinstance(table, dict):
            continue
        if key in known:
            values[key] = value
        else:
            warnings.append(f"Unknown configuration key '{key}' in {path} (ignored).")
    if isinstance(table, dict):
        for key, value in table.items():
            if key in known:
                values[key] = value
            else:
                warnings.append(
                    f"Unknown configuration key '{CONFIG_TABLE_NAME}.{key}' in {path} (ignored)."
                )
    return values


def _read_env(env: Mapping[str, str], warnings: list[str]) -> dict[str, object]:
    """Extract ``YONGA_*`` configuration values from an environment mapping."""
    known = _user_config_keys()
    values: dict[str, object] = {}
    for raw_key, raw_value in env.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        if raw_key.upper() in SECRET_ENV_KEYS or raw_key.upper() in NON_SETTING_ENV_KEYS:
            continue
        field = raw_key[len(ENV_PREFIX) :].lower()
        if field in known:
            values[field] = raw_value
        else:
            warnings.append(f"Unknown environment variable '{raw_key}' (ignored).")
    return values


def _read_overrides(
    overrides: Mapping[str, object] | None, warnings: list[str]
) -> dict[str, object]:
    """Filter CLI overrides down to known, explicitly-set keys."""
    if not overrides:
        return {}
    known = _user_config_keys()
    values: dict[str, object] = {}
    for key, value in overrides.items():
        if value is None:
            continue
        if key in known:
            values[key] = value
        else:
            warnings.append(f"Unknown configuration override '{key}' (ignored).")
    return values


def _telemetry_error() -> Exception:
    """Build the error raised whenever telemetry is switched on."""
    return _configuration_error(
        "Telemetry is disabled by design",
        TELEMETRY_REJECTED_MESSAGE,
        code="config.telemetry.rejected",
        remediation=[
            "Remove the 'telemetry' key from the configuration file.",
            f"Unset {ENV_PREFIX}TELEMETRY if it is set in the environment.",
        ],
    )


def load_settings(
    *,
    config_file: Path | None = None,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, object] | None = None,
) -> Settings:
    """Load settings from every source and apply the documented precedence.

    Unknown keys never fail the load; they are collected on
    :attr:`Settings.config_warnings`. Enabling telemetry, or any value that does
    not validate, raises a ``ConfigurationError`` carrying a diagnostic.
    """
    env_map: dict[str, str] = dict(os.environ if env is None else env)
    warnings: list[str] = []

    env_values = _read_env(env_map, warnings)
    override_values = _read_overrides(overrides, warnings)

    resolved_config_file = config_file
    if resolved_config_file is None:
        directory = override_values.get("config_dir") or env_values.get("config_dir")
        bootstrap = default_paths(
            env_map,
            config_dir=Path(str(directory)).expanduser() if directory else None,
        )
        resolved_config_file = bootstrap.config_file
        file_values = _read_toml(resolved_config_file, warnings)
        if not file_values and not resolved_config_file.exists():
            resolved_config_file = None
    else:
        resolved_config_file = resolved_config_file.expanduser()
        file_values = _read_toml(resolved_config_file, warnings)

    merged: dict[str, Any] = {
        name: field.get_default(call_default_factory=True)
        for name, field in Settings.model_fields.items()
    }
    merged.update(file_values)
    merged.update(env_values)
    merged.update(override_values)
    merged["config_warnings"] = warnings

    if _is_true(merged.get("telemetry")):
        raise _telemetry_error()

    try:
        settings = Settings(**merged)
    except ValidationError as exc:
        raise _configuration_error(
            "Invalid configuration value",
            _describe_validation_error(exc),
            code="config.value.invalid",
            remediation=[
                "Check the configuration file, YONGA_* environment variables "
                "and command line options for the reported key."
            ],
            cause=exc,
        ) from exc

    settings._env = env_map
    settings._config_file = resolved_config_file
    return settings


def _describe_validation_error(exc: ValidationError) -> str:
    """Render a pydantic validation failure as one human-readable line."""
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"]) or "<configuration>"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts) if parts else str(exc)
