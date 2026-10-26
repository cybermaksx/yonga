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
import shlex
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
    merge_provider_fields,
    model_entry_in,
    provider_fields_in,
    read_config,
    render,
    with_model,
    without_model,
    write_config,
)
from yonga.domain.artifacts import ArtifactManifest, ArtifactSummary
from yonga.domain.diagnostics import (
    ConfigurationError,
    FailureCategory,
    diagnostic,
)
from yonga.domain.hardware import normalise_device
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
    "RecordUpdate",
    "RemoveAction",
    "RemovePlan",
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
_LONG_PROMPT_DEVICES = frozenset({"GPU", "CPU"})
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8000
_RECORD_SCHEMA = 1
_EDITED = "edited since yonga wrote it"


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


@dataclass(frozen=True)
class RecordUpdate:
    """What saving the record file did."""

    written: bool
    """Whether the record file now holds what was saved."""
    set_aside: Path | None = None
    """Where a record file that did not parse was moved before it was replaced."""


class ConnectionRecords:
    """The record file: ``{"schema": 1, "connections": [...]}``."""

    def __init__(self, path: Path) -> None:
        """Bind to the record file; nothing is read until :meth:`load`."""
        self._path = path

    @property
    def backup_path(self) -> Path:
        """Where :meth:`save` moves a record file that does not parse."""
        return self._path.with_name(self._path.name + ".bak")

    def load(self) -> list[ConnectionRecord]:
        """Every recorded connection; a missing, unreadable or corrupt file reads as ``[]``."""
        try:
            return self._read() or []
        except OSError:
            return []

    def _read(self) -> list[ConnectionRecord] | None:
        """The connections, ``[]`` for a missing file, ``None`` for content that is corrupt.

        Only content that does not parse or validate is corrupt. A file that
        cannot be read at all raises its ``OSError``: it may be perfectly good,
        so :meth:`save` must neither set it aside nor replace it.
        """
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except ValueError:
            return None
        try:
            if payload["schema"] != _RECORD_SCHEMA:
                return None
            return [ConnectionRecord.model_validate(item) for item in payload["connections"]]
        except (KeyError, TypeError, ValidationError):
            return None

    def save(self, records: Sequence[ConnectionRecord]) -> RecordUpdate:
        """Atomically replace the file, never over one that did not parse.

        Such a file (broken, or written by another schema) is first moved to
        :attr:`backup_path`, replacing an older backup, so the newest bad copy
        survives. An I/O error, also reading the existing file, is reported in the
        result, not raised, and leaves that file as it is.
        """
        text = json.dumps(
            {
                "schema": _RECORD_SCHEMA,
                "connections": [record.model_dump(mode="json") for record in records],
            },
            indent=2,
            ensure_ascii=False,
        )
        temporary: str | None = None
        set_aside: Path | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=".opencode-connections-", suffix=".json.part", dir=self._path.parent
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(text + "\n")
                output.flush()
                os.fsync(output.fileno())
            if self._read() is None:
                self._path.replace(self.backup_path)
                set_aside = self.backup_path
            Path(temporary).replace(self._path)
            temporary = None
            return RecordUpdate(True, set_aside)
        except OSError:
            return RecordUpdate(False, set_aside)
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
    """The provider fields this connection writes (``name`` only into a new provider)."""
    previous_provider_fields: dict[str, Any] | None
    """What the existing provider held at those places; ``None`` for a new provider."""
    document: dict[str, Any]
    created_provider: bool
    serve_command: str

    @property
    def provider_changed(self) -> bool:
        """True when writing changes a field of an existing provider."""
        return (
            self.previous_provider_fields is not None
            and self.previous_provider_fields != self.provider_fields
        )


class RemoveAction(str, Enum):
    """What :meth:`ConnectService.apply_remove` will do (or did)."""

    REMOVED = "removed"
    KEPT = "kept"
    """Recorded entries exist, but every one was edited since and is kept."""
    NOT_CONNECTED = "not_connected"


@dataclass(frozen=True)
class RemovePlan:
    """What a removal would take out and what it leaves alone."""

    action: RemoveAction
    config: Path
    """The config file, with symbolic links resolved."""
    snapshot: ConfigSnapshot | None
    """The file as read; ``None`` when no record named it, so it was not read."""
    document: dict[str, Any]
    removed: list[tuple[str, str]]
    """``(provider, model)`` entries to delete."""
    kept: list[tuple[str, str, str]]
    """``(provider, model, reason)`` entries left in place."""
    stale: list[ConnectionRecord]
    """Records whose entry is already gone from the file."""
    elsewhere: list[str]
    """Other config files the record connects the artifact in."""


@dataclass(frozen=True)
class ConnectResult:
    """What :meth:`ConnectService.apply` did."""

    created_config: bool
    record_written: bool
    """Whether the record names this connection afterwards."""
    record_set_aside: Path | None = None
    """Where a record file that did not parse was moved."""


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
        # `serve` answers to the reference as typed plus the manifest's ids, so an
        # artifact without a manifest is keyed by the reference itself.
        artifact_id = manifest.artifact_id if manifest is not None else reference
        self._require_tools(session)
        fixed_prompt = self._require_prompt_length(session, manifest, artifact_id)
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
        key = endpoint.provider_key
        options: dict[str, Any] = {"baseURL": endpoint.base_url}
        if self._env.get(_API_KEY_ENV):
            options["apiKey"] = "{env:" + _API_KEY_ENV + "}"
        new_provider: dict[str, Any] = {
            "npm": _PROVIDER_NPM,
            "name": f"yonga ({endpoint.label})",
            "options": options,
        }
        try:
            snapshot = read_config(config_path)
            check_provider(snapshot.document, key)
        except ConfigurationError as error:
            extended = _with_snippet(error, key, new_provider, artifact_id, entry)
            if extended is error:
                raise
            raise extended from error
        providers = snapshot.document.get("provider", {})
        provider = providers.get(key) if isinstance(providers, Mapping) else None
        models = provider.get("models") if isinstance(provider, Mapping) else None
        previous = models.get(artifact_id) if isinstance(models, Mapping) else None
        # Only the keys yonga writes count; ones the user added to the entry do not.
        previous_entry = model_entry_in(previous, entry) if isinstance(previous, Mapping) else None
        if isinstance(provider, Mapping):
            # An existing provider keeps its name and every key yonga does not write.
            provider_fields = {
                field: value for field, value in new_provider.items() if field != "name"
            }
            previous_fields: dict[str, Any] | None = provider_fields_in(provider, provider_fields)
        else:
            provider_fields, previous_fields = new_provider, None
        provider_changed = previous_fields is not None and previous_fields != provider_fields
        if previous is None:
            action = ConnectAction.ADDED
        elif previous_entry == entry and not provider_changed:
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
            previous_provider_fields=previous_fields,
            document=with_model(snapshot.document, key, provider_fields, artifact_id, entry),
            created_provider=provider is None,
            serve_command=_serve_command(
                reference if _is_path(reference) else artifact_id, endpoint, device
            ),
        )

    # --- applying ----------------------------------------------------

    def apply(self, plan: ConnectPlan) -> ConnectResult:
        """Write the config unless already connected, then bring the record up to date.

        An already connected entry writes nothing to the config, and the record
        only when it has no entry for this connection (it was lost or never
        written), so that ``--remove`` can find it.
        """
        created_config = False
        if plan.action is not ConnectAction.ALREADY_CONNECTED:
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
        loaded = self._records.load()
        if plan.action is ConnectAction.ALREADY_CONNECTED and any(
            item.same_connection(record) and item.model == record.model for item in loaded
        ):
            return ConnectResult(created_config, True)
        update = self._records.save(_recorded(loaded, record, plan.provider_fields))
        return ConnectResult(created_config, update.written, update.set_aside)

    # --- removal -----------------------------------------------------

    def plan_remove(self, artifact_id: str, *, config_path: Path) -> RemovePlan:
        """Decide what to take out of the config; needs the record, never the artifact.

        The record is read first: when it names no connection of the artifact in
        this file, the answer is ``not_connected`` and the file is not read at all.
        """
        config_file = config_path.resolve()
        config = str(config_file)
        records = self._records.load()
        mine = [
            item for item in records if item.config == config and item.artifact_id == artifact_id
        ]
        elsewhere = list(
            dict.fromkeys(
                item.config
                for item in records
                if item.artifact_id == artifact_id and item.config != config
            )
        )
        if not mine:
            return RemovePlan(
                RemoveAction.NOT_CONNECTED, config_file, None, {}, [], [], [], elsewhere
            )
        try:
            snapshot = read_config(config_path)
        except ConfigurationError as error:
            extended = _for_removal(error, mine)
            if extended is error:
                raise
            raise extended from error
        document = snapshot.document
        removed: list[tuple[str, str]] = []
        kept: list[tuple[str, str, str]] = []
        stale: list[ConnectionRecord] = []
        for record in mine:
            current = _recorded_model(document, record.provider, artifact_id)
            if current is None:
                stale.append(record)
            elif model_entry_in(current, record.model) != record.model:
                kept.append((record.provider, artifact_id, _EDITED))
            else:
                removed.append((record.provider, artifact_id))
                owner = _owner(records, config, record.provider)
                document = without_model(
                    document,
                    record.provider,
                    artifact_id,
                    drop_provider_when=owner.provider_fields if owner is not None else None,
                )
        if removed:
            action = RemoveAction.REMOVED
        elif kept:
            action = RemoveAction.KEPT
        else:
            action = RemoveAction.NOT_CONNECTED
        return RemovePlan(
            action, snapshot.path, snapshot, document, removed, kept, stale, elsewhere
        )

    def apply_remove(self, plan: RemovePlan) -> RecordUpdate:
        """Write the config when something is removed, then forget the records.

        Returns what saving the record file did. Provider ownership travels with
        the provider: when the record that created it goes while the provider stays,
        it moves to a remaining record of that provider.
        """
        if plan.removed and plan.snapshot is not None:
            write_config(plan.snapshot, plan.document)
        config = str(plan.config)
        gone = {(config, provider, model) for provider, model in plan.removed}
        gone.update((item.config, item.provider, item.artifact_id) for item in plan.stale)
        loaded = self._records.load()
        remaining = [
            item for item in loaded if (item.config, item.provider, item.artifact_id) not in gone
        ]
        providers = plan.document.get("provider")
        for owner in loaded:
            if (
                (owner.config, owner.provider, owner.artifact_id) not in gone
                or not owner.created_provider
                or not isinstance(providers, Mapping)
                or owner.provider not in providers
            ):
                continue
            group = [
                index
                for index, item in enumerate(remaining)
                if item.config == owner.config and item.provider == owner.provider
            ]
            if group and not any(remaining[index].created_provider for index in group):
                remaining[group[0]] = remaining[group[0]].model_copy(
                    update={"created_provider": True, "provider_fields": owner.provider_fields}
                )
        return self._records.save(remaining)

    def is_connected(self, artifact_id: str) -> bool:
        """True when the record file lists a connection for the artifact."""
        return bool(self.connected_configs(artifact_id))

    def connected_configs(self, artifact_id: str) -> list[str]:
        """The config files the record connects the artifact in, each once."""
        return list(
            dict.fromkeys(
                item.config for item in self._records.load() if item.artifact_id == artifact_id
            )
        )

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
            except Exception:  # a sibling that cannot be opened is no suggestion
                continue
            if other.tool_calling is not None:
                found.append(summary.artifact_id)
        return found

    def _require_prompt_length(
        self, session: SessionView, manifest: ArtifactManifest | None, artifact_id: str
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
        source_id = manifest.resolved_source.repo_id if manifest is not None else ""
        raise ConfigurationError(
            diagnostic(
                "connect.opencode.prompt_too_short",
                FailureCategory.CONFIGURATION_ERROR,
                "This artifact's prompt length is too short for OpenCode",
                f"The artifact on {session.device} is compiled for prompts of at most {length} "
                "tokens. OpenCode's first request measured 5261 tokens in an empty home and 8514 "
                "with skills and an AGENTS.md (OpenCode 2.1.2), so yonga requires at least "
                f"{OPENCODE_MIN_PROMPT_TOKENS}.",
                remediation=[self._longer_prompt_advice(session, artifact_id, repo_id, source_id)],
            )
        )

    def _longer_prompt_advice(
        self, session: SessionView, artifact_id: str, repo_id: str, source_id: str
    ) -> str:
        """How to get a long enough prompt: drop ``--device``, use a sibling, or prepare one.

        When the refused artifact is itself recommended for the GPU or CPU, the
        refusal comes from a forced ``--device npu``. Otherwise a prepared GPU or
        CPU artifact of the same source model is named (never the refused one),
        else how to prepare one.
        """
        listing = list(self._artifacts())
        own = next((item for item in listing if item.artifact_id == artifact_id), None)
        own_device = normalise_device(own.recommended_device or "") if own is not None else ""
        if own_device in _LONG_PROMPT_DEVICES:
            return (
                f"This artifact is recommended for the {own_device}: drop "
                f"`--device {session.device.lower()}` (or use `--device {own_device.lower()}`)."
            )
        for summary in listing:
            if (
                summary.ready
                and summary.artifact_id != artifact_id
                and summary.repo_id == source_id
                and normalise_device(summary.recommended_device or "") in _LONG_PROMPT_DEVICES
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


def _owner(
    records: Sequence[ConnectionRecord], config: str, provider: str
) -> ConnectionRecord | None:
    """The newest record that says yonga created ``provider`` in ``config``."""
    owners = [
        item
        for item in records
        if item.config == config
        and item.provider == provider
        and item.created_provider
        and item.provider_fields is not None
    ]
    return max(owners, key=lambda item: item.written_at, default=None)


def _recorded(
    loaded: Sequence[ConnectionRecord],
    record: ConnectionRecord,
    written: Mapping[str, Any],
) -> list[ConnectionRecord]:
    """The record list after ``record`` was written with provider fields ``written``.

    ``record`` replaces the entry for the same connection. Exactly one record
    keeps owning the provider, with the fields yonga last wrote merged into what
    it wrote before, so that removing the last model drops the provider while
    those fields are still unchanged:

    * a provider ``record`` created (it was missing) is owned by ``record``;
      older owners of that provider lost what they owned and give it up;
    * otherwise the owner stays the owner: ``record`` itself when its earlier
      entry was, else the newest owner among the others.
    """
    previous = next((item for item in loaded if item.same_connection(record)), None)
    others = [item for item in loaded if not item.same_connection(record)]
    same_provider = [
        index
        for index, item in enumerate(others)
        if item.config == record.config and item.provider == record.provider
    ]
    if record.created_provider:
        for index in same_provider:
            if others[index].created_provider:
                others[index] = others[index].model_copy(
                    update={"created_provider": False, "provider_fields": None}
                )
        return [*others, record]
    if previous is not None and previous.created_provider:
        fields = merge_provider_fields(previous.provider_fields or {}, written)
        record = record.model_copy(update={"created_provider": True, "provider_fields": fields})
        return [*others, record]
    owner = _owner([others[index] for index in same_provider], record.config, record.provider)
    if owner is not None:
        index = others.index(owner)
        others[index] = owner.model_copy(
            update={"provider_fields": merge_provider_fields(owner.provider_fields or {}, written)}
        )
    return [*others, record]


def _recorded_model(
    document: Mapping[str, Any], provider: str, model_id: str
) -> Mapping[str, Any] | None:
    """``provider[PROVIDER].models[MODEL_ID]`` when the document has it."""
    providers = document.get("provider")
    entry = providers.get(provider) if isinstance(providers, Mapping) else None
    models = entry.get("models") if isinstance(entry, Mapping) else None
    found = models.get(model_id) if isinstance(models, Mapping) else None
    return found if isinstance(found, Mapping) else None


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


def _is_path(reference: str) -> bool:
    """True when ``reference`` names an artifact directory rather than an id."""
    return os.sep in reference or "/" in reference or Path(reference).is_dir()


def _serve_command(target: str, endpoint: Endpoint, device: str | None) -> str:
    """The ``yonga serve`` command that answers the entry just planned.

    ``target`` is the artifact id, or the path as the user passed it (``serve``
    resolves ids only inside the store).
    """
    parts = ["yonga", "serve", shlex.quote(target)]
    if endpoint.port != _DEFAULT_PORT:
        parts += ["--port", str(endpoint.port)]
    if endpoint.host != _DEFAULT_HOST:
        parts += ["--host", endpoint.host]
    if device is not None:
        parts += ["--device", device]
    return " ".join(parts)


def _for_removal(
    error: ConfigurationError, records: Sequence[ConnectionRecord]
) -> ConfigurationError:
    """Say which entries to delete by hand from a file yonga cannot rewrite."""
    original = error.diagnostic
    if original.code != "connect.opencode.config_not_json":
        return error
    entries = ", ".join(f"`provider.{item.provider}.models.{item.artifact_id}`" for item in records)
    extended = original.model_copy(
        update={
            "remediation": [
                "Comments and trailing commas (JSONC) cannot be rewritten without losing them.",
                f"Remove {entries} from the file by hand.",
            ]
        }
    )
    return ConfigurationError(extended, cause=error.cause)


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
