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

Images reach a VLM artifact only, and only with the newest user message:
OpenVINO GenAI 2026.4 binds images to the last user turn of a stateless
conversation and cannot place one where an earlier turn put it. An image in an
earlier message is therefore refused, unless the server was started to carry
earlier images forward, which moves them onto the newest user message with a
short note saying so (see :func:`carry_images_forward`). That trades accuracy
about *when* an image was shown for compatibility with clients that resend
every image of the conversation on each follow-up.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Sequence
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
from yonga.domain.models import PipelineKind

__all__ = [
    "EARLIER_TURN_REASON",
    "IMAGES_EARLIER_TURN",
    "IMAGES_UNSUPPORTED",
    "MAX_IMAGES",
    "MAX_IMAGE_PIXELS",
    "MAX_STOP_LENGTH",
    "MAX_STOP_SEQUENCES",
    "TOO_MANY_IMAGES",
    "CompletionRequest",
    "ImagePolicy",
    "ModelServer",
    "ServedModel",
    "carry_images_forward",
]

#: Most stop sequences one request may carry, as OpenAI allows.
MAX_STOP_SEQUENCES = 4
#: Longest stop sequence accepted, in characters. A streamed answer holds back
#: the tail that may still complete a match, at most one character less than
#: this, plus the whitespace in front of that tail.
MAX_STOP_LENGTH = 256

#: Most images one request may carry, across all of its messages.
MAX_IMAGES = 4
#: Most pixels one image may have: 4096 x 4096. A larger image is refused, not
#: scaled down, so the model never sees something other than what was sent.
MAX_IMAGE_PIXELS = 4096 * 4096

#: Diagnostic codes of the image refusals, shared with the wire layer so a
#: request gets the same code whichever layer catches it.
IMAGES_UNSUPPORTED = "serve.request.images_unsupported"
IMAGES_EARLIER_TURN = "serve.request.images_earlier_turn"
TOO_MANY_IMAGES = "serve.request.too_many_images"

#: The why of :data:`IMAGES_EARLIER_TURN`, for both layers' messages.
EARLIER_TURN_REASON = (
    "Images are accepted only in the last message, which must be the user's: OpenVINO "
    "GenAI binds images to the newest user turn of a conversation and cannot place one "
    "in an earlier turn. Send the image with the question about it, or start the server "
    "with --carry-images-forward to move earlier images onto the last message."
)


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


class ImagePolicy(BaseModel):
    """What a server accepts as image input; the wire layer checks a request against it."""

    model_config = ConfigDict(frozen=True)

    accepted: bool
    """Whether the served artifact takes images at all: only a VLM does."""
    carry_forward: bool = False
    """Move images of earlier messages onto the newest one instead of refusing them."""
    max_images: int = MAX_IMAGES
    max_pixels: int = MAX_IMAGE_PIXELS


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

    def __init__(
        self, session: ChatSession, *, name: str, carry_images_forward: bool = False
    ) -> None:
        """Serve ``session`` under ``name``. Nothing is loaded until :meth:`load`.

        ``carry_images_forward`` moves images sent with earlier messages onto
        the newest user message instead of refusing the request (see the
        module docstring for the trade-off).
        """
        self._session = session
        self._image_policy = ImagePolicy(
            accepted=session.pipeline is PipelineKind.VLM,
            carry_forward=carry_images_forward,
        )
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
    def image_policy(self) -> ImagePolicy:
        """Which images this server accepts."""
        return self._image_policy

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
        history = ChatHistory(messages=self._conversation(request))
        with self._lock:
            return self._session.respond(history, settings, on_text=on_text, stop=request.stop)

    def _conversation(self, request: CompletionRequest) -> list[ChatMessage]:
        """The request's messages, its images checked against the policy.

        The wire layer has already refused what is checked here, naming the
        parameter; this holds the same line for any other caller.
        """
        messages = list(request.messages)
        count = sum(len(message.images) for message in messages)
        if not count:
            return messages
        policy = self._image_policy
        if not policy.accepted:
            raise _invalid(IMAGES_UNSUPPORTED, "This model takes text only; it is not a VLM.")
        if count > policy.max_images:
            raise _invalid(TOO_MANY_IMAGES, f"At most {policy.max_images} images are accepted.")
        earlier = any(message.images for message in messages[:-1])
        if earlier and not policy.carry_forward:
            raise _invalid(IMAGES_EARLIER_TURN, EARLIER_TURN_REASON)
        if messages[-1].role != "user":
            raise _invalid(IMAGES_EARLIER_TURN, EARLIER_TURN_REASON)
        return carry_images_forward(messages) if earlier else messages


def carry_images_forward(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    """Move every image onto the last message, which must be the user's, with a note.

    The images keep their order, earlier ones first, and the last message's
    text gains a line saying which of them came with earlier messages, so the
    model is not told they were all sent just now. The earlier messages keep
    their text. Messages without images are returned as they are.
    """
    earlier = [image for message in messages[:-1] for image in message.images]
    if not earlier:
        return list(messages)
    last = messages[-1]
    note = (
        "[Image 1 was attached to an earlier message in this conversation.]"
        if len(earlier) == 1
        else f"[Images 1-{len(earlier)} were attached to earlier messages in this conversation.]"
    )
    content = f"{note}\n{last.content}" if last.content else note
    moved = [
        message.model_copy(update={"images": []}) if message.images else message
        for message in messages[:-1]
    ]
    moved.append(last.model_copy(update={"images": [*earlier, *last.images], "content": content}))
    return moved


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
