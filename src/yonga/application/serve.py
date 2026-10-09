"""Serving one prepared artifact to OpenAI-compatible clients.

``SPEC.md`` section 22 asks for the runtime and model selection of ``chat``,
not a second preparation stack, so this module owns no model logic of its own.
It holds one :class:`~yonga.application.chat.ChatSession` -- the same object
``yonga chat`` drives, with the same registry chat profile, device resolution,
compile cache and reasoning policy -- and answers each request through
:meth:`ChatSession.respond`, which reads the client's conversation and records
nothing.

Nothing here knows about HTTP. The FastAPI layer in :mod:`yonga.api` parses
requests into a :class:`CompletionRequest` and renders what comes back; that
split keeps the web framework an optional dependency and keeps every policy
decision testable without a server.

One pipeline serves every request, so generation is serialised: a second
request waits for the first rather than sharing a pipeline that is not safe to
share. That is the honest capacity of one local accelerator.
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from yonga.application.chat import (
    MAX_OUTPUT_TOKENS,
    STOP_SEQUENCE_REASON,
    TEMPERATURE_RANGE,
    TOP_K_MAX,
    TOP_P_RANGE,
    ChatHistory,
    ChatMessage,
    ChatSession,
    ChatTurn,
    TextSink,
)
from yonga.domain.benchmarks import GenerationSettings
from yonga.domain.diagnostics import ConfigurationError, FailureCategory, diagnostic

__all__ = [
    "MAX_STOP_LENGTH",
    "MAX_STOP_SEQUENCES",
    "CompletionRequest",
    "ModelServer",
    "ServedModel",
]

#: Most stop sequences one request may carry, as OpenAI allows.
MAX_STOP_SEQUENCES = 4
#: Longest stop sequence accepted, in characters. Every piece of a streamed
#: answer is held back by up to this much while a match may still complete.
MAX_STOP_LENGTH = 256


class ServedModel(BaseModel):
    """The one model a server answers for, as ``/v1/models`` describes it."""

    model_config = ConfigDict(frozen=True)

    id: str
    """The name clients send as ``model``: what the user passed to ``yonga serve``."""
    aliases: frozenset[str] = Field(default_factory=frozenset)
    """Other names accepted for it: the artifact id and the source repositories."""
    artifact_id: str | None = None
    created: int = 0
    """Unix time the artifact was prepared."""
    device: str
    pipeline: str

    def answers_to(self, name: str) -> bool:
        """Whether a request naming ``name`` is asking for this model."""
        return name == self.id or name in self.aliases


class CompletionRequest(BaseModel):
    """One chat completion, already translated out of the wire format."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    repetition_penalty: float | None = None
    stop: list[str] = Field(default_factory=list)
    """The client's stop sequences, matched on the visible answer (see ``ChatSession.respond``)."""


def _invalid(code: str, detail: str) -> ConfigurationError:
    return ConfigurationError(
        diagnostic(code, FailureCategory.CONFIGURATION_ERROR, "Invalid request", detail)
    )


class ModelServer:
    """Answers completion requests with one loaded chat session."""

    def __init__(self, session: ChatSession, *, name: str) -> None:
        """Serve ``session`` under ``name``. Nothing is loaded until :meth:`load`."""
        self._session = session
        self._lock = threading.Lock()
        manifest = session.manifest
        aliases = {name}
        created = 0
        artifact_id: str | None = None
        if manifest is not None:
            artifact_id = manifest.artifact_id
            aliases |= {
                manifest.artifact_id,
                manifest.requested_model.repo_id,
                manifest.resolved_source.repo_id,
            }
            created = _unix(manifest.created_at)
        self._model = ServedModel(
            id=name,
            aliases=frozenset(alias for alias in aliases if alias),
            artifact_id=artifact_id,
            created=created,
            device=session.device,
            pipeline=session.pipeline.value,
        )
        self._started = time.time()

    @property
    def model(self) -> ServedModel:
        """The model this server answers for."""
        return self._model

    @property
    def loaded(self) -> bool:
        """Whether the pipeline is compiled and in memory."""
        return self._session.runtime_loads > 0 and not self._session.closed

    def load(self) -> None:
        """Compile and load the model now, so the first request does not pay for it.

        Raises the session's typed error when the model cannot load, which is
        what lets ``yonga serve`` fail at startup instead of on every request.
        """
        with self._lock:
            self._session.load()

    def close(self) -> None:
        """Release the pipeline."""
        with self._lock:
            self._session.close()

    def health(self) -> dict[str, object]:
        """What ``GET /health`` reports."""
        return {
            "status": "ok" if self.loaded else "loading",
            "model": self._model.id,
            "device": self._model.device,
            "pipeline": self._model.pipeline,
            "uptime_seconds": round(time.time() - self._started, 1),
        }

    def generation_for(self, request: CompletionRequest) -> GenerationSettings:
        """The session's generation defaults with the request's overrides applied.

        Validation matches ``chat``'s ``/tokens``, ``/temp``, ``/top-p`` and
        ``/top-k``: a value chat would refuse is refused here too, as a typed
        error the API layer turns into a 400.
        """
        settings = self._session.generation_settings.model_copy(deep=True)
        if request.max_tokens is not None:
            if not 1 <= request.max_tokens <= MAX_OUTPUT_TOKENS:
                raise _invalid(
                    "serve.request.max_tokens",
                    f"max_tokens must be between 1 and {MAX_OUTPUT_TOKENS}.",
                )
            settings.max_new_tokens = request.max_tokens
        if request.temperature is not None:
            low, high = TEMPERATURE_RANGE
            if not low <= request.temperature <= high:
                raise _invalid(
                    "serve.request.temperature",
                    f"temperature must be between {low:g} and {high:g}.",
                )
            settings.temperature = request.temperature
            settings.do_sample = request.temperature > 0.0
        if request.top_p is not None:
            low, high = TOP_P_RANGE
            if not low < request.top_p <= high:
                raise _invalid("serve.request.top_p", "top_p must be above 0 and at most 1.")
            settings.top_p = request.top_p
        if request.top_k is not None:
            if not 0 <= request.top_k <= TOP_K_MAX:
                raise _invalid("serve.request.top_k", f"top_k must be between 0 and {TOP_K_MAX}.")
            settings.top_k = request.top_k
        if request.seed is not None:
            settings.seed = request.seed
        if request.repetition_penalty is not None:
            if request.repetition_penalty <= 0.0:
                raise _invalid(
                    "serve.request.repetition_penalty", "repetition_penalty must be positive."
                )
            settings.repetition_penalty = request.repetition_penalty
        return settings

    def complete(self, request: CompletionRequest, *, on_text: TextSink | None = None) -> ChatTurn:
        """Answer ``request``, streaming through ``on_text`` when one is given.

        Waits for any generation already running. Invalid parameters raise; a
        runtime failure comes back as a turn carrying its diagnostic.
        """
        settings = self.generation_for(request)
        history = ChatHistory(messages=list(request.messages))
        with self._lock:
            return self._session.respond(history, settings, on_text=on_text, stop=request.stop)


def completion_id() -> str:
    """A fresh ``chatcmpl-…`` identifier."""
    return f"chatcmpl-{uuid.uuid4().hex}"


def finish_reason(turn: ChatTurn, settings_limit: int) -> str:
    """OpenAI's ``finish_reason`` for a finished turn: ``length`` or ``stop``.

    A turn a stop sequence ended is ``stop`` even when the runtime went on to
    its token limit, as one that cannot stream early does.
    """
    result = turn.result
    if result is None:
        return "stop"
    reason = (result.stop_reason or "").lower()
    if reason == STOP_SEQUENCE_REASON:
        return "stop"
    if reason == "length":
        return "length"
    if result.output_tokens is not None and result.output_tokens >= settings_limit:
        return "length"
    return "stop"


def _unix(moment: datetime) -> int:
    try:
        return int(moment.timestamp())
    except (OverflowError, OSError, ValueError):
        return 0
