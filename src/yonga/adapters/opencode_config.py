"""OpenCode's JSON config file: read it strictly, edit it purely, write it atomically.

``yonga connect opencode`` adds an OpenAI-compatible provider entry to OpenCode's
``opencode.json``. That file belongs to the user and to OpenCode, so the rules
here are narrow on purpose:

* **Only yonga's keys change.** The edit functions work on a deep copy of the
  parsed document, touch ``provider[KEY]`` and nothing else, keep the order of
  every existing key and append new ones. Everything else the user wrote
  (theme, other providers, ``model``) comes back out exactly as it went in.
* **No JSONC rewrite.** OpenCode also accepts comments and trailing commas, but
  the standard library cannot round-trip them: rewriting would silently drop the
  user's comments. A file that is not plain JSON is refused and left as it is.
* **The write is atomic.** The new text goes to a temporary file in the same
  directory, is flushed to disk, gets the permissions of the file it replaces
  (``0600`` for a new one: the file may hold API keys) and is renamed over the
  target. A symbolic link is resolved first, so the file it points to is
  replaced and the link stays a link.
* **Compare before replace.** The bytes read earlier are remembered; right
  before the rename the target is read again, and a difference (OpenCode or the
  user saved in between) aborts the write instead of discarding their change.
* **Nothing else happens.** The file is never deleted, and there is no sudo: an
  unwritable location is reported with how to choose another one.
"""

from __future__ import annotations

import copy
import errno
import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yonga.config.paths import XDG_CONFIG_HOME, _xdg_dir
from yonga.domain.diagnostics import (
    RERUN_PLACEHOLDER,
    ConfigurationError,
    FailureCategory,
    YongaError,
    diagnostic,
)

__all__ = [
    "OPENCODE_SCHEMA_URL",
    "ConfigSnapshot",
    "check_provider",
    "default_config_path",
    "merge_provider_fields",
    "provider_fields_in",
    "read_config",
    "render",
    "with_model",
    "without_model",
    "write_config",
]

#: The ``$schema`` a config created by yonga starts with.
OPENCODE_SCHEMA_URL: str = "https://opencode.ai/config.json"

_DEFAULT_CONFIG_HOME: str = ".config"
_NEW_FILE_MODE: int = 0o600


def default_config_path(env: Mapping[str, str]) -> Path:
    """``$XDG_CONFIG_HOME/opencode/opencode.json``, else ``~/.config/opencode/opencode.json``."""
    return _xdg_dir(env, XDG_CONFIG_HOME, _DEFAULT_CONFIG_HOME) / "opencode" / "opencode.json"


# ======================================================================
# reading
# ======================================================================
@dataclass(frozen=True)
class ConfigSnapshot:
    """The config file as it was read: what to edit, and what to compare against."""

    path: Path
    """The file itself, with symbolic links resolved."""
    raw: bytes | None
    """The bytes read, ``None`` when the file does not exist."""
    document: dict[str, Any]
    """The parsed root object; ``{}`` for a missing, empty or blank file."""
    mode: int | None
    """Permission bits of the file, ``None`` when it does not exist."""


def _not_json(path: Path, problem: str) -> ConfigurationError:
    """The refusal for a file that is not a plain JSON object."""
    return ConfigurationError(
        diagnostic(
            "connect.opencode.config_not_json",
            FailureCategory.CONFIGURATION_ERROR,
            "OpenCode's config is not plain JSON",
            f"{path} {problem} yonga edits plain JSON only and left the file untouched.",
            remediation=[
                "Comments and trailing commas (JSONC) cannot be rewritten without losing them.",
                "Add the provider to the file by hand, or point yonga at another file with "
                "--config FILE.",
            ],
            context={"config": str(path)},
        )
    )


def read_config(path: Path) -> ConfigSnapshot:
    """Read and validate the config at ``path`` (see the module docstring).

    A missing file, an empty one and a whitespace-only one all read as ``{}``.
    Anything that is not UTF-8 text holding one JSON object is refused.
    """
    resolved = path.resolve()
    try:
        info = resolved.stat()
        raw = resolved.read_bytes()
    except FileNotFoundError:
        return ConfigSnapshot(resolved, None, {}, None)
    except IsADirectoryError:
        raise _not_a_file(resolved) from None
    except OSError as error:
        raise _io_failure("read", resolved, error) from error
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise _not_json(resolved, "is not UTF-8 text.") from None
    if not text.strip():
        return ConfigSnapshot(resolved, raw, {}, _mode_of(info.st_mode))
    try:
        document = json.loads(text)
    except ValueError as error:
        raise _not_json(resolved, f"does not parse as JSON ({error}).") from error
    if not isinstance(document, dict):
        raise _not_json(resolved, "does not hold a JSON object at its top level.")
    return ConfigSnapshot(resolved, raw, document, _mode_of(info.st_mode))


def _mode_of(st_mode: int) -> int:
    """The permission bits of a ``stat`` mode."""
    return st_mode & 0o7777


def _not_a_file(path: Path) -> ConfigurationError:
    """The refusal for a config path that names a directory."""
    return ConfigurationError(
        diagnostic(
            "connect.opencode.config_not_a_file",
            FailureCategory.CONFIGURATION_ERROR,
            "OpenCode's config is not a file",
            f"{path} is a directory, not a config file.",
            remediation=["Pass the config file itself with --config FILE."],
            context={"config": str(path)},
        )
    )


def _io_failure(action: str, path: Path, error: OSError) -> YongaError:
    """Turn an ``OSError`` into a typed failure with the right exit code.

    Permission and read-only-filesystem errors, and every other error the
    operating system reports, are PERMISSION_DENIED (exit 10) except a full
    disk, which is INSUFFICIENT_DISK. All name the path and the system's text.
    """
    if error.errno in (errno.ENOSPC, errno.EDQUOT):
        category = FailureCategory.INSUFFICIENT_DISK
        advice = "Free some space, or choose another location with --config FILE."
    else:
        category = FailureCategory.PERMISSION_DENIED
        advice = "Choose a location you can use with --config FILE."
    return YongaError(
        diagnostic(
            f"connect.opencode.config_not_{'readable' if action == 'read' else 'writable'}",
            category,
            f"Cannot {action} OpenCode's config",
            f"Could not {action} {path}: {error.strerror or error}.",
            remediation=[advice, "yonga never uses sudo for this, and it is not needed."],
            context={"config": str(path)},
            raw_excerpt=f"{type(error).__name__}: {error}",
        ),
        cause=error,
    )


def check_provider(document: Mapping[str, Any], key: str) -> None:
    """Refuse a document whose ``provider.KEY`` path holds something other than objects.

    ``provider``, ``provider[KEY]`` and ``provider[KEY].models`` may be absent,
    but when present must be JSON objects; the message names the offending path.
    """
    path = "provider"
    node: Any = document
    for step in ("provider", key, "models"):
        if not isinstance(node, Mapping) or step not in node:
            return
        node = node[step]
        path = path if step == "provider" else f"{path}.{step}"
        if not isinstance(node, dict):
            raise ConfigurationError(
                diagnostic(
                    "connect.opencode.config_not_json",
                    FailureCategory.CONFIGURATION_ERROR,
                    "OpenCode's config has an unexpected shape",
                    f"`{path}` in OpenCode's config is not a JSON object, so yonga cannot "
                    "add its provider there. Nothing was changed.",
                    remediation=[
                        f"Fix or remove `{path}` in the file, then run the command again."
                    ],
                    context={"key_path": path},
                )
            )


# ======================================================================
# pure edits
# ======================================================================
def with_model(
    document: Mapping[str, Any],
    key: str,
    provider_fields: Mapping[str, Any],
    model_id: str,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    """A deep copy of ``document`` with ``provider[KEY].models[MODEL_ID] = entry``.

    ``$schema`` is added first only to an empty document. A missing provider is
    created with ``provider_fields`` first and ``models`` after them; an existing
    one has them merged in by :func:`merge_provider_fields`, so a key the user
    added (``options.timeout``, ``headers``) stays. Existing key order is kept
    and new keys are appended. The input must have passed :func:`check_provider`.
    """
    result: dict[str, Any] = copy.deepcopy(dict(document))
    if not result:
        result["$schema"] = OPENCODE_SCHEMA_URL
    providers = result.setdefault("provider", {})
    provider = merge_provider_fields(providers.get(key, {}), provider_fields)
    provider.setdefault("models", {})[model_id] = copy.deepcopy(dict(entry))
    providers[key] = provider
    return result


def merge_provider_fields(provider: Mapping[str, Any], fields: Mapping[str, Any]) -> dict[str, Any]:
    """A deep copy of ``provider`` with ``fields`` written over it, one level deep.

    A field whose value is an object is merged key by key into the provider's
    object of that name (``options.baseURL`` is set, ``options.timeout`` stays);
    any other field replaces the provider's value. Nothing is ever removed.
    """
    result: dict[str, Any] = copy.deepcopy(dict(provider))
    for name, value in fields.items():
        current = result.get(name)
        if isinstance(value, Mapping) and isinstance(current, dict):
            current.update(copy.deepcopy(dict(value)))
        else:
            result[name] = copy.deepcopy(value)
    return result


def provider_fields_in(provider: Mapping[str, Any], fields: Mapping[str, Any]) -> dict[str, Any]:
    """What ``provider`` holds now at the places ``fields`` would write.

    The counterpart of :func:`merge_provider_fields`: comparing the result with
    ``fields`` tells whether the merge would change anything, and the two are
    the "before" and "after" shown to the user. Absent keys are left out.
    """
    result: dict[str, Any] = {}
    for name, value in fields.items():
        if name not in provider:
            continue
        current = provider[name]
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            result[name] = {
                inner: copy.deepcopy(current[inner]) for inner in value if inner in current
            }
        else:
            result[name] = copy.deepcopy(current)
    return result


def without_model(
    document: Mapping[str, Any],
    key: str,
    model_id: str,
    *,
    drop_provider_when: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """A deep copy of ``document`` without ``provider[KEY].models[MODEL_ID]``.

    When ``drop_provider_when`` is given, the provider ends up with no models
    and all its other fields equal ``drop_provider_when`` (so it is exactly what
    yonga wrote and nobody added to), the provider is removed as well.
    """
    result: dict[str, Any] = copy.deepcopy(dict(document))
    providers = result.get("provider")
    if not isinstance(providers, dict) or not isinstance(providers.get(key), dict):
        return result
    provider = providers[key]
    models = provider.get("models")
    if isinstance(models, dict):
        models.pop(model_id, None)
    if drop_provider_when is not None and models == {}:
        others = {name: value for name, value in provider.items() if name != "models"}
        if others == dict(drop_provider_when):
            del providers[key]
    return result


def render(document: Mapping[str, Any]) -> str:
    """The text written to disk: two-space indent, UTF-8 as is, trailing newline."""
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


# ======================================================================
# writing
# ======================================================================
def write_config(snapshot: ConfigSnapshot, document: Mapping[str, Any]) -> bool:
    """Atomically replace the file ``snapshot`` was read from; ``True`` if it was created.

    Parent directories are created. The new text is written to a temporary file
    beside the target, flushed, given ``snapshot.mode`` (``0600`` when the file
    is new) and renamed over the target after the target is read once more and
    found byte-for-byte equal to ``snapshot.raw``.
    """
    path = snapshot.path
    data = render(document).encode("utf-8")
    temporary: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".opencode-", suffix=".json.part", dir=path.parent
        )
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fchmod(output.fileno(), _NEW_FILE_MODE if snapshot.mode is None else snapshot.mode)
            os.fsync(output.fileno())
        _recheck(snapshot)
        Path(temporary).replace(path)
        temporary = None
    except OSError as error:
        raise _io_failure("write", path, error) from error
    finally:
        if temporary is not None:
            with suppress(OSError):
                Path(temporary).unlink()
    _fsync_directory(path.parent)
    return snapshot.raw is None


def _recheck(snapshot: ConfigSnapshot) -> None:
    """Refuse when the target no longer holds the bytes ``snapshot`` read."""
    try:
        now: bytes | None = snapshot.path.read_bytes()
    except FileNotFoundError:
        now = None
    if now == snapshot.raw:
        return
    raise ConfigurationError(
        diagnostic(
            "connect.opencode.config_changed",
            FailureCategory.CONFIGURATION_ERROR,
            "OpenCode's config changed",
            f"{snapshot.path} changed while yonga was working on it; nothing was replaced.",
            remediation=[f"Run the command again: {RERUN_PLACEHOLDER}"],
            context={"config": str(snapshot.path)},
        )
    )


def _fsync_directory(directory: Path) -> None:
    """Make the rename durable; best effort."""
    try:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError:
        return
    try:
        with suppress(OSError):
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
