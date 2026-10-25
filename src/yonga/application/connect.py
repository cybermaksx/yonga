"""The ``yonga connect opencode`` service: check an artifact, plan the entry, write it.

OpenCode talks to ``yonga serve`` as an OpenAI-compatible provider, and it cannot
look up a custom provider's limits, so the entry has to carry them. Before
anything is written the artifact is held against what OpenCode needs
(``docs/superpowers/specs/2026-10-24-opencode-connect-design.md`` section 4):

* **Tools.** OpenCode sends its tools with every agent request, so a model that
  ``serve`` refuses tools for cannot be used at all.
* **Prompt length.** An NPU artifact compiles with a fixed ``MAX_PROMPT_LEN``;
  below :data:`OPENCODE_MIN_PROMPT_TOKENS` the first request is refused.
* **Context.** It is read from the artifact's ``config.json``; none is invented.

The checks run on a session that is opened but never loaded, in that order, and
before the OpenCode file is read, so a refused artifact leaves everything alone.
:meth:`ConnectService.plan_connect` decides and :meth:`ConnectService.apply`
writes; the file handling itself lives in :mod:`yonga.adapters.opencode_config`.

A small record, ``<data-dir>/opencode-connections.json``, remembers what was
written so that it can be taken out again. It is advisory: failing to write it is
reported, not raised, because the OpenCode file is already correct by then.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from yonga.adapters.opencode_config import (
    ConfigSnapshot,
    check_provider,
    read_config,
    render,
    with_model,
    write_config,
)
from yonga.domain.artifacts import ArtifactManifest, ArtifactSummary
from yonga.domain.diagnostics import (
    ConfigurationError,
    FailureCategory,
    YongaError,
    diagnostic,
)
from yonga.domain.models import PipelineKind
from yonga.util.net import is_loopback, url_host

__all__ = [
    "OPENCODE_MIN_PROMPT_TOKENS",
    "RECORD_FILENAME",
    "ConnectAction",
    "ConnectPlan",
    "ConnectResult",
    "ConnectService",
    "ConnectionRecord",
    "ConnectionRecords",
    "Endpoint",
    "SessionView",
]

#: The smallest fixed prompt length OpenCode can work with. OpenCode 2.1.2,
#: measured on 2026-10-21 on the reference host with the Qwen3 chat template and
#: tokenizer: the first agent request is 5261 tokens in an empty ``HOME`` (8514
#: with skills and an ``AGENTS.md``). 8192 is the first power of two above it.
OPENCODE_MIN_PROMPT_TOKENS: int = 8192

#: The connection record, inside the data directory.
RECORD_FILENAME: str = "opencode-connections.json"

_PROVIDER_NPM = "@ai-sdk/openai-compatible"
_API_KEY_ENV = "YONGA_API_KEY"
_FIXED_PROMPT_PROPERTY = "MAX_PROMPT_LEN"
_SUGGESTIONS = 3
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8000
_RECORD_SCHEMA = 1


class ConnectAction(str, Enum):
    """What :meth:`ConnectService.apply` will do (or did)."""

    ADDED = "added"
    UPDATED = "updated"
    ALREADY_CONNECTED = "already_connected"


@dataclass(frozen=True)
class Endpoint:
    """The address ``yonga serve`` listens on, as OpenCode will reach it."""

    host: str = _DEFAULT_HOST
    port: int = _DEFAULT_PORT

    @property
    def provider_key(self) -> str:
        """``yonga-PORT`` for a loopback host, else ``yonga-HOST-PORT``."""
        if is_loopback(self.host):
            return f"yonga-{self.port}"
        return "yonga-" + re.sub(r"[^A-Za-z0-9_-]", "-", f"{self.host}-{self.port}")

    @property
    def base_url(self) -> str:
        """The OpenAI-compatible base URL."""
        return f"http://{url_host(self.host)}:{self.port}/v1"

    @property
    def label(self) -> str:
        """``HOST:PORT`` as ``serve`` prints it."""
        return f"{url_host(self.host)}:{self.port}"


class SessionView(Protocol):
    """The part of :class:`~yonga.application.chat.ChatSession` the checks read."""

    @property
    def manifest(self) -> ArtifactManifest | None:
        """The artifact manifest, when one could be read."""
        ...

    @property
    def device(self) -> str:
        """The resolved OpenVINO device."""
        ...

    @property
    def pipeline(self) -> PipelineKind:
        """Whether the artifact is an LLM or a VLM."""
        ...

    @property
    def artifact_dir(self) -> Path:
        """The prepared artifact directory."""
        ...

    @property
    def tool_calling(self) -> object | None:
        """The model's tool-call format, ``None`` when it takes no tools."""
        ...

    @property
    def tool_calling_refusal(self) -> str | None:
        """Why the model takes no tools."""
        ...

    @property
    def runtime_properties(self) -> Mapping[str, str]:
        """The properties the runtime is compiled with."""
        ...


# ======================================================================
# the record
# ======================================================================
class ConnectionRecord(BaseModel):
    """One connection yonga wrote: enough to take exactly that back out."""

    config: str
    provider: str
    artifact_id: str
    model: dict[str, Any]
    created_provider: bool
    provider_fields: dict[str, Any] | None = None
    written_at: datetime

    def same_connection(self, other: ConnectionRecord) -> bool:
        """True when both name the same config file, provider and artifact."""
        return (self.config, self.provider, self.artifact_id) == (
            other.config,
            other.provider,
            other.artifact_id,
        )


class ConnectionRecords:
    """The record file: ``{"schema": 1, "connections": [...]}``."""

    def __init__(self, path: Path) -> None:
        """Bind to the record file; nothing is read until :meth:`load`."""
        self._path = path

    def load(self) -> list[ConnectionRecord]:
        """Every recorded connection; a missing or unreadable file reads as ``[]``."""
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            return [ConnectionRecord.model_validate(item) for item in payload["connections"]]
        except (OSError, ValueError, KeyError, TypeError, ValidationError):
            return []

    def save(self, records: Sequence[ConnectionRecord]) -> bool:
        """Atomically replace the file; ``False`` instead of raising on an I/O error."""
        text = json.dumps(
            {
                "schema": _RECORD_SCHEMA,
                "connections": [record.model_dump(mode="json") for record in records],
            },
            indent=2,
            ensure_ascii=False,
        )
        temporary: str | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=".opencode-connections-", suffix=".json.part", dir=self._path.parent
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(text + "\n")
                output.flush()
                os.fsync(output.fileno())
            Path(temporary).replace(self._path)
            temporary = None
            return True
        except OSError:
            return False
        finally:
            if temporary is not None:
                with contextlib.suppress(OSError):
                    Path(temporary).unlink()


# ======================================================================
# plan and result
# ======================================================================
@dataclass(frozen=True)
class ConnectPlan:
    """Everything decided, nothing written yet."""

    action: ConnectAction
    snapshot: ConfigSnapshot
    endpoint: Endpoint
    artifact_id: str
    entry: dict[str, Any]
    previous_entry: dict[str, Any] | None
    provider_fields: dict[str, Any]
    document: dict[str, Any]
    created_provider: bool
    serve_command: str


@dataclass(frozen=True)
class ConnectResult:
    """What :meth:`ConnectService.apply` did."""

    created_config: bool
    record_written: bool


# ======================================================================
# the service
# ======================================================================
class ConnectService:
    """Check an artifact against OpenCode, plan its provider entry and write it."""

    def __init__(
        self,
        *,
        open_session: Callable[[str, str | None], SessionView],
        artifacts: Callable[[], Sequence[ArtifactSummary]],
        tool_families: Sequence[str],
        records: ConnectionRecords,
        default_max_new_tokens: int,
        env: Mapping[str, str],
    ) -> None:
        """Take the outside world as callables, so building the service opens nothing."""
        self._open_session = open_session
        self._artifacts = artifacts
        self._tool_families = tool_families
        self._records = records
        self._default_max_new_tokens = default_max_new_tokens
        self._env = env

    # --- planning ----------------------------------------------------

    def plan_connect(
        self,
        reference: str,
        *,
        endpoint: Endpoint,
        device: str | None,
        context: int | None,
        config_path: Path,
    ) -> ConnectPlan:
        """Run the checks in order, then read the config and decide the outcome."""
        session = self._open_session(reference, device)
        manifest = session.manifest
        artifact_id = manifest.artifact_id if manifest is not None else session.artifact_dir.name
        self._require_tools(session)
        fixed_prompt = self._require_prompt_length(session, manifest)
        output = self._output_limit(manifest)
        window = context
        if window is None:
            window = (
                fixed_prompt + output
                if fixed_prompt is not None
                else _context_from_config(session.artifact_dir / "config.json")
            )
        if window is None:
            raise _context_unknown(artifact_id)
        limit: dict[str, int] = {"context": window}
        if fixed_prompt is not None:
            limit["input"] = fixed_prompt
        limit["output"] = output
        entry: dict[str, Any] = {
            "name": _model_name(session, manifest, artifact_id),
            "tool_call": True,
            "limit": limit,
        }
        options: dict[str, Any] = {"baseURL": endpoint.base_url}
        if self._env.get(_API_KEY_ENV):
            options["apiKey"] = "{env:" + _API_KEY_ENV + "}"
        provider_fields: dict[str, Any] = {
            "npm": _PROVIDER_NPM,
            "name": f"yonga ({endpoint.label})",
            "options": options,
        }
        key = endpoint.provider_key
        try:
            snapshot = read_config(config_path)
            check_provider(snapshot.document, key)
        except ConfigurationError as error:
            extended = _with_snippet(error, key, provider_fields, artifact_id, entry)
            if extended is error:
                raise
            raise extended from error
        providers = snapshot.document.get("provider", {})
        provider = providers.get(key) if isinstance(providers, Mapping) else None
        models = provider.get("models") if isinstance(provider, Mapping) else None
        previous = models.get(artifact_id) if isinstance(models, Mapping) else None
        previous_entry = dict(previous) if isinstance(previous, Mapping) else None
        if previous is None:
            action = ConnectAction.ADDED
        elif previous == entry and all(
            isinstance(provider, Mapping) and provider.get(name) == value
            for name, value in provider_fields.items()
        ):
            action = ConnectAction.ALREADY_CONNECTED
        else:
            action = ConnectAction.UPDATED
        return ConnectPlan(
            action=action,
            snapshot=snapshot,
            endpoint=endpoint,
            artifact_id=artifact_id,
            entry=entry,
            previous_entry=previous_entry,
            provider_fields=provider_fields,
            document=with_model(snapshot.document, key, provider_fields, artifact_id, entry),
            created_provider=provider is None,
            serve_command=_serve_command(artifact_id, endpoint, device),
        )

    # --- applying ----------------------------------------------------

    def apply(self, plan: ConnectPlan) -> ConnectResult:
        """Write the config and remember the connection; do nothing when connected."""
        if plan.action is ConnectAction.ALREADY_CONNECTED:
            return ConnectResult(False, True)
        created_config = write_config(plan.snapshot, plan.document)
        record = ConnectionRecord(
            config=str(plan.snapshot.path),
            provider=plan.endpoint.provider_key,
            artifact_id=plan.artifact_id,
            model=dict(plan.entry),
            created_provider=plan.created_provider,
            provider_fields=dict(plan.provider_fields) if plan.created_provider else None,
            written_at=datetime.now(UTC),
        )
        records = [item for item in self._records.load() if not item.same_connection(record)]
        records.append(record)
        return ConnectResult(created_config, self._records.save(records))

    # --- the checks --------------------------------------------------

    def _require_tools(self, session: SessionView) -> None:
        """Refuse a model ``serve`` would refuse tools for, and suggest ones it would not."""
        if session.tool_calling is not None:
            return
        refusal = session.tool_calling_refusal or "The model declares no tool-call format."
        current = session.manifest.artifact_id if session.manifest is not None else None
        capable = self._tool_capable_ids(exclude=current)
        if capable:
            remediation = [
                "Connect an artifact that takes tools instead: "
                + "; ".join(f"yonga connect opencode {item}" for item in capable)
            ]
        else:
            families = ", ".join(self._tool_families) or "none"
            remediation = [
                "No prepared artifact takes tools. Prepare a model from a family whose "
                f"built-in chat profile declares tool calling: {families}."
            ]
        raise ConfigurationError(
            diagnostic(
                "connect.opencode.tools_unsupported",
                FailureCategory.CONFIGURATION_ERROR,
                "This model cannot be used from OpenCode",
                "OpenCode sends its tools with every request, and `yonga serve` refuses "
                f"tools for this model. {refusal}",
                remediation=remediation,
            )
        )

    def _tool_capable_ids(self, *, exclude: str | None) -> list[str]:
        """Up to three other ready artifacts whose session takes tools."""
        found: list[str] = []
        for summary in self._artifacts():
            if len(found) == _SUGGESTIONS:
                break
            if not summary.ready or summary.artifact_id == exclude:
                continue
            try:
                other = self._open_session(summary.artifact_id, None)
            except YongaError:
                continue
            if other.tool_calling is not None:
                found.append(summary.artifact_id)
        return found

    def _require_prompt_length(
        self, session: SessionView, manifest: ArtifactManifest | None
    ) -> int | None:
        """Return the fixed prompt length, refusing one OpenCode's first request exceeds."""
        raw = session.runtime_properties.get(_FIXED_PROMPT_PROPERTY)
        if raw is None:
            return None
        try:
            length = int(raw)
        except (TypeError, ValueError):
            return None
        if length >= OPENCODE_MIN_PROMPT_TOKENS:
            return length
        repo_id = manifest.requested_model.repo_id if manifest is not None else ""
        raise ConfigurationError(
            diagnostic(
                "connect.opencode.prompt_too_short",
                FailureCategory.CONFIGURATION_ERROR,
                "This artifact's prompt length is too short for OpenCode",
                f"The artifact on {session.device} is compiled for prompts of at most {length} "
                f"tokens. OpenCode's first request is {OPENCODE_MIN_PROMPT_TOKENS} tokens or "
                "close to it (5261 measured with OpenCode 2.1.2 in an empty home, 8514 with "
                f"skills and an AGENTS.md), so it needs at least {OPENCODE_MIN_PROMPT_TOKENS}.",
                remediation=[self._longer_prompt_advice(repo_id)],
            )
        )

    def _longer_prompt_advice(self, repo_id: str) -> str:
        """Name a prepared GPU or CPU artifact of the same model, else how to prepare one."""
        for summary in self._artifacts():
            if (
                summary.ready
                and summary.repo_id == repo_id
                and (summary.recommended_device or "").upper() in {"GPU", "CPU"}
            ):
                return f"Use the {summary.recommended_device} artifact: " + (
                    f"yonga connect opencode {summary.artifact_id}"
                )
        return f"Prepare the model for the GPU: yonga prepare {repo_id} --device gpu"

    def _output_limit(self, manifest: ArtifactManifest | None) -> int:
        """The manifest's recommended output length, else the configured one."""
        recommended = manifest.recommended_runtime if manifest is not None else None
        if recommended is not None and recommended.max_new_tokens:
            return recommended.max_new_tokens
        return self._default_max_new_tokens


# ======================================================================
# helpers
# ======================================================================
def _model_name(session: SessionView, manifest: ArtifactManifest | None, artifact_id: str) -> str:
    """``REPO · DEVICE · PROFILE`` as OpenCode lists the model."""
    if manifest is None:
        return f"{artifact_id} · {session.device}"
    return f"{manifest.requested_model.repo_id} · {session.device} · {manifest.export.profile}"


def _context_from_config(path: Path) -> int | None:
    """``max_position_embeddings`` of the model, or of its ``text_config``."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    text_config = payload.get("text_config")
    candidates = [payload.get("max_position_embeddings")]
    if isinstance(text_config, dict):
        candidates.append(text_config.get("max_position_embeddings"))
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _context_unknown(artifact_id: str) -> ConfigurationError:
    return ConfigurationError(
        diagnostic(
            "connect.opencode.context_unknown",
            FailureCategory.CONFIGURATION_ERROR,
            "The model's context length is unknown",
            f"The artifact {artifact_id} has no max_position_embeddings in its config.json, "
            "and yonga does not invent a limit.",
            remediation=[
                f"Give the context length yourself: yonga connect opencode {artifact_id} "
                "--context N"
            ],
        )
    )


def _serve_command(artifact_id: str, endpoint: Endpoint, device: str | None) -> str:
    """The ``yonga serve`` command that answers the entry just planned."""
    parts = ["yonga", "serve", artifact_id]
    if endpoint.port != _DEFAULT_PORT:
        parts += ["--port", str(endpoint.port)]
    if endpoint.host != _DEFAULT_HOST:
        parts += ["--host", endpoint.host]
    if device is not None:
        parts += ["--device", device]
    return " ".join(parts)


def _with_snippet(
    error: ConfigurationError,
    key: str,
    provider_fields: Mapping[str, Any],
    artifact_id: str,
    entry: Mapping[str, Any],
) -> ConfigurationError:
    """Add the JSON to paste by hand to a refusal; any other error passes through."""
    original = error.diagnostic
    if original.code != "connect.opencode.config_not_json":
        return error
    snippet = render({"provider": {key: {**provider_fields, "models": {artifact_id: entry}}}})
    extended = original.model_copy(
        update={
            "remediation": [
                *original.remediation,
                "The JSON to add under the top-level object by hand:\n" + snippet.rstrip(),
            ]
        }
    )
    return ConfigurationError(extended, cause=error.cause)
