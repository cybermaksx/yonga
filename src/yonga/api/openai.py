"""The subset of the OpenAI chat completions wire format yonga accepts.

Unknown fields are ignored, the way OpenAI-compatible servers generally treat
them, so a client that sends ``user`` or ``stream_options`` works unchanged.
What is *not* ignored is a field whose value would change the answer in a way
yonga cannot honour -- ``n`` above 1, tools, a JSON response format, image
content, penalties. Those are refused with a 400 that names the parameter,
because silently answering a different question than the client asked is worse
than saying no.

``stop`` is honoured: a string or a list of at most
:data:`~yonga.application.serve.MAX_STOP_SEQUENCES` non-empty strings of at most
:data:`~yonga.application.serve.MAX_STOP_LENGTH` characters each. Anything else
is a 400 naming ``stop``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from yonga.application.chat import ChatMessage
from yonga.application.serve import MAX_STOP_LENGTH, MAX_STOP_SEQUENCES, CompletionRequest

__all__ = [
    "MAX_MESSAGES",
    "ChatCompletionRequest",
    "InvalidParameter",
    "ParameterError",
    "UnsupportedParameter",
]

#: Most messages one request may carry. Far beyond any real conversation; it
#: exists so a request cannot make the server build millions of objects. A
#: request over it is refused with ``400`` naming ``messages``.
MAX_MESSAGES = 4096


class ParameterError(ValueError):
    """A request parameter refused with a 400; ``param`` names it, ``code`` says why."""

    code = "invalid_request"

    def __init__(self, param: str, message: str) -> None:
        super().__init__(message)
        self.param = param


class UnsupportedParameter(ParameterError):
    """A request parameter yonga cannot honour."""

    code = "unsupported_parameter"


class InvalidParameter(ParameterError):
    """A parameter yonga honours, given a value it does not accept."""


class WireMessage(BaseModel):
    """One message as clients send it: content is a string or a list of parts."""

    model_config = ConfigDict(extra="ignore")

    role: str
    content: str | list[dict[str, Any]] | None = None


class StreamOptions(BaseModel):
    """``stream_options``; only ``include_usage`` means anything here."""

    model_config = ConfigDict(extra="ignore")

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    """``POST /v1/chat/completions``."""

    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[WireMessage] = Field(min_length=1, max_length=MAX_MESSAGES)
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    repetition_penalty: float | None = None
    n: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    tools: list[dict[str, Any]] | None = None
    functions: list[dict[str, Any]] | None = None
    response_format: dict[str, Any] | None = None
    logprobs: bool | None = None

    @property
    def include_usage(self) -> bool:
        """Whether a streamed response should end with a usage chunk."""
        return self.stream_options is not None and self.stream_options.include_usage

    def to_completion(self) -> CompletionRequest:
        """Translate into yonga's request, refusing what cannot be honoured."""
        self._refuse_unsupported()
        return CompletionRequest(
            messages=[_message(index, message) for index, message in enumerate(self.messages)],
            max_tokens=self.max_completion_tokens
            if self.max_completion_tokens is not None
            else self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=self.seed,
            repetition_penalty=self.repetition_penalty,
            stop=self._stop_sequences(),
        )

    def _refuse_unsupported(self) -> None:
        if self.n is not None and self.n != 1:
            raise UnsupportedParameter("n", "Only n=1 is supported: one pipeline, one answer.")
        for name in ("presence_penalty", "frequency_penalty"):
            if getattr(self, name):
                raise UnsupportedParameter(name, f"{name} is not supported; repetition_penalty is.")
        if self.tools or self.functions:
            raise UnsupportedParameter("tools", "Tool and function calling are not supported.")
        if self.response_format and self.response_format.get("type") not in (None, "text"):
            raise UnsupportedParameter(
                "response_format", "Only the text response format is supported."
            )
        if self.logprobs:
            raise UnsupportedParameter("logprobs", "Log probabilities are not supported.")
        last = self.messages[-1]
        if last.role != "user":
            raise UnsupportedParameter(
                "messages", "The last message must come from the user; there is nothing to answer."
            )

    def _stop_sequences(self) -> list[str]:
        """``stop`` as a list without duplicates; refused when it breaks the limits."""
        if self.stop is None:
            return []
        stops = [self.stop] if isinstance(self.stop, str) else list(self.stop)
        if len(stops) > MAX_STOP_SEQUENCES:
            raise InvalidParameter(
                "stop", f"At most {MAX_STOP_SEQUENCES} stop sequences are accepted."
            )
        if any(not text for text in stops):
            raise InvalidParameter("stop", "A stop sequence cannot be empty.")
        if any(len(text) > MAX_STOP_LENGTH for text in stops):
            raise InvalidParameter(
                "stop", f"A stop sequence can be at most {MAX_STOP_LENGTH} characters long."
            )
        return list(dict.fromkeys(stops))


_ROLES: dict[str, Literal["system", "user", "assistant"]] = {
    "system": "system",
    "developer": "system",
    "user": "user",
    "assistant": "assistant",
}


def _message(index: int, message: WireMessage) -> ChatMessage:
    role = _ROLES.get(message.role)
    if role is None:
        raise UnsupportedParameter(
            f"messages[{index}].role",
            f"Role {message.role!r} is not supported; use system, user or assistant.",
        )
    return ChatMessage(role=role, content=_text(index, message.content))


def _text(index: int, content: str | list[dict[str, Any]] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        kind = part.get("type")
        if kind != "text":
            raise UnsupportedParameter(
                f"messages[{index}].content",
                f"Content of type {kind!r} is not supported; only text is.",
            )
        parts.append(str(part.get("text", "")))
    return "".join(parts)
