"""The subset of the OpenAI chat completions wire format yonga accepts.

Unknown fields are ignored, the way OpenAI-compatible servers generally treat
them, so a client that sends ``user`` or ``stream_options`` works unchanged.
What is *not* ignored is a field whose value would change the answer in a way
yonga cannot honour -- ``n`` above 1, tools, a JSON response format, penalties,
content other than text and images. Those are refused with a 400 that names the
parameter, because silently answering a different question than the client
asked is worse than saying no.

``stop`` is honoured: a string or a list of at most
:data:`~yonga.application.serve.MAX_STOP_SEQUENCES` non-empty strings of at most
:data:`~yonga.application.serve.MAX_STOP_LENGTH` characters each. Anything else
is a 400 naming ``stop``.

``image_url`` content parts are honoured for a VLM artifact, as the server's
:class:`~yonga.application.serve.ImagePolicy` allows: only inline ``data:`` URLs
(see :mod:`yonga.api.images`, which never fetches anything), in user messages,
at most :data:`~yonga.application.serve.MAX_IMAGES` per request, and only in the
last message unless the server carries earlier images forward. Every refusal
names the content part. ``image_url.detail`` is a resolution hint and is
ignored: the image is used as sent.
"""

from __future__ import annotations

from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from yonga.api.images import ImageRefused, decode_data_url
from yonga.application.chat import ChatImage, ChatMessage
from yonga.application.serve import (
    EARLIER_TURN_REASON,
    IMAGES_EARLIER_TURN,
    IMAGES_UNSUPPORTED,
    MAX_STOP_LENGTH,
    MAX_STOP_SEQUENCES,
    TOO_MANY_IMAGES,
    CompletionRequest,
    ImagePolicy,
)

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

    def __init__(self, param: str, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.param = param
        if code is not None:
            self.code = code


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

    @property
    def has_images(self) -> bool:
        """Whether any message carries an image part, so translating it decodes images."""
        return any(
            isinstance(message.content, list)
            and any(part.get("type") == "image_url" for part in message.content)
            for message in self.messages
        )

    def to_completion(self, images: ImagePolicy | None = None) -> CompletionRequest:
        """Translate into yonga's request, refusing what cannot be honoured.

        ``images`` is what the server accepts; without one, any image is
        refused. Accepted images are decoded here, which takes CPU time, so a
        caller on an event loop runs this on a worker thread when
        :attr:`has_images` says so.
        """
        self._refuse_unsupported()
        parsed = [_parse(index, message) for index, message in enumerate(self.messages)]
        return CompletionRequest(
            messages=_with_images(parsed, images),
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


class _ImagePart(NamedTuple):
    """An image content part, located for error messages but not yet decoded."""

    param: str
    """``messages[i].content[j]``."""
    url: str


class _Parsed(NamedTuple):
    role: Literal["system", "user", "assistant"]
    text: str
    images: list[_ImagePart]


def _parse(index: int, message: WireMessage) -> _Parsed:
    role = _ROLES.get(message.role)
    if role is None:
        raise UnsupportedParameter(
            f"messages[{index}].role",
            f"Role {message.role!r} is not supported; use system, user or assistant.",
        )
    content = message.content
    if content is None:
        return _Parsed(role, "", [])
    if isinstance(content, str):
        return _Parsed(role, content, [])
    texts: list[str] = []
    images: list[_ImagePart] = []
    for position, part in enumerate(content):
        kind = part.get("type")
        if kind == "text":
            texts.append(str(part.get("text", "")))
            continue
        if kind != "image_url":
            raise UnsupportedParameter(
                f"messages[{index}].content",
                f"Content of type {kind!r} is not supported; only text and image_url are.",
            )
        param = f"messages[{index}].content[{position}]"
        if role != "user":
            raise UnsupportedParameter(param, "Images are accepted in user messages only.")
        image_url = part.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else None
        if not isinstance(url, str):
            raise InvalidParameter(
                f"{param}.image_url.url", "An image_url part needs image_url.url, a data: URL."
            )
        images.append(_ImagePart(param, url))
    return _Parsed(role, "".join(texts), images)


def _with_images(parsed: list[_Parsed], policy: ImagePolicy | None) -> list[ChatMessage]:
    """The messages, with their images checked against ``policy`` and then decoded.

    Everything that can be refused without decoding is refused first, so a
    request that cannot be served costs no decoding at all.
    """
    located = [(index, part) for index, message in enumerate(parsed) for part in message.images]
    if not located:
        return [ChatMessage(role=message.role, content=message.text) for message in parsed]
    accepted = _check_images(located, len(parsed) - 1, policy)
    return [
        ChatMessage(
            role=message.role,
            content=message.text,
            images=[_decode(part, accepted) for part in message.images],
        )
        for message in parsed
    ]


def _check_images(
    located: list[tuple[int, _ImagePart]], last: int, policy: ImagePolicy | None
) -> ImagePolicy:
    """Refuse what ``policy`` does not allow; return it when it allows these images."""
    first = located[0][1].param
    if policy is None or not policy.accepted:
        raise UnsupportedParameter(
            first,
            "This model takes text only; images need a vision-language (VLM) artifact.",
            code=IMAGES_UNSUPPORTED,
        )
    if len(located) > policy.max_images:
        raise InvalidParameter(
            "messages",
            f"At most {policy.max_images} images are accepted in one request; "
            f"this one has {len(located)}.",
            code=TOO_MANY_IMAGES,
        )
    earlier = next((part for index, part in located if index != last), None)
    if earlier is not None and not policy.carry_forward:
        raise UnsupportedParameter(earlier.param, EARLIER_TURN_REASON, code=IMAGES_EARLIER_TURN)
    return policy


def _decode(part: _ImagePart, policy: ImagePolicy) -> ChatImage:
    param = f"{part.param}.image_url.url"
    try:
        return decode_data_url(part.url, max_pixels=policy.max_pixels)
    except ImageRefused as refusal:
        if refusal.unsupported:
            raise UnsupportedParameter(param, str(refusal)) from None
        raise InvalidParameter(param, str(refusal)) from None
