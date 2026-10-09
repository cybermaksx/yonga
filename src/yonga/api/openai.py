"""The subset of the OpenAI chat completions wire format yonga accepts.

Unknown fields are ignored, the way OpenAI-compatible servers generally treat
them, so a client that sends ``user`` or ``stream_options`` works unchanged.
What is *not* ignored is a field whose value would change the answer in a way
yonga cannot honour -- ``n`` above 1, legacy ``functions``, a JSON response
format, penalties, content other than text and images, a ``tool_choice`` that
forces a call. Those are refused with a 400 that names the parameter, because
silently answering a different question than the client asked is worse than
saying no.

``tools`` are honoured for a model whose registry chat profile declares a
measured tool-call format (the server's
:class:`~yonga.application.serve.ToolPolicy`); any other model refuses them with
a 400 naming ``tools``. Each tool must be a function with a valid name and, if
given, an object of parameters; ``tool_choice`` may be ``auto`` or ``none``, and
``parallel_tool_calls: false`` keeps only the first call. A conversation may
carry assistant ``tool_calls`` and ``tool`` messages answering them, and may end
on one, but only alongside ``tools``: dropping the calls of a history would
answer a conversation the client did not send.

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
ignored: the image is used as sent. Only an image's header is read here; its
pixels are decoded once the request holds the pipeline (see
:class:`~yonga.application.serve.PendingImage`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from yonga.api.images import ImageRefused, check_data_url, decode_data_url
from yonga.application.chat import ChatImage, ChatMessage, Role, ToolCall
from yonga.application.serve import (
    EARLIER_TURN_REASON,
    IMAGES_EARLIER_TURN,
    IMAGES_UNSUPPORTED,
    MAX_STOP_LENGTH,
    MAX_STOP_SEQUENCES,
    MAX_TOOLS,
    TOO_MANY_IMAGES,
    TOOLS_UNSUPPORTED,
    CompletionRequest,
    ImagePolicy,
    PendingImage,
    ToolPolicy,
)

__all__ = [
    "MAX_MESSAGES",
    "ChatCompletionRequest",
    "InvalidParameter",
    "ParameterError",
    "UnsupportedParameter",
    "wire_tool_call",
]

#: Most messages one request may carry. Far beyond any real conversation; it
#: exists so a request cannot make the server build millions of objects. A
#: request over it is refused with ``400`` naming ``messages``.
MAX_MESSAGES = 4096

#: Deepest nesting of arrays and objects accepted in a history tool call's
#: arguments. Checked by counting brackets before parsing: whether ``json.loads``
#: raises ``RecursionError`` on deep input depends on the interpreter's stack,
#: so it cannot be the guard.
MAX_ARGUMENT_DEPTH = 64

#: A function name as OpenAI accepts one.
_FUNCTION_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")

#: Why the legacy function-calling parameters are refused.
_LEGACY_FUNCTIONS = "Legacy function calling is not supported; send tools instead."


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
    tool_calls: list[Any] | None = None
    tool_call_id: Any = None
    function_call: Any = None


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
    tools: list[Any] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    functions: Any = None
    function_call: Any = None
    response_format: dict[str, Any] | None = None
    logprobs: bool | None = None

    @property
    def include_usage(self) -> bool:
        """Whether a streamed response should end with a usage chunk."""
        return self.stream_options is not None and self.stream_options.include_usage

    @property
    def has_images(self) -> bool:
        """Whether any message carries an image part, so translating it reads image headers."""
        return any(
            isinstance(message.content, list)
            and any(part.get("type") == "image_url" for part in message.content)
            for message in self.messages
        )

    def to_completion(
        self,
        images: ImagePolicy | None = None,
        *,
        tools: ToolPolicy | None = None,
        check: Callable[[CompletionRequest], object] | None = None,
    ) -> CompletionRequest:
        """Translate into yonga's request, refusing what cannot be honoured.

        ``images`` and ``tools`` are what the server accepts; without a policy,
        any image or tool is refused. ``check`` sees the request before any
        image is read, so a request it refuses (by raising) costs no image work
        at all. Then each accepted image's header is read, which takes some CPU
        time, so a caller on an event loop runs this on a worker thread when
        :attr:`has_images` says so. The pixels are decoded later (see
        :class:`~yonga.application.serve.PendingImage`).
        """
        self._refuse_unsupported()
        choice = self._tool_choice()
        offered = self._tools(tools)
        parsed = [_parse(index, message) for index, message in enumerate(self.messages)]
        messages = _conversation(parsed, offered=bool(offered))
        located = [(index, part) for index, message in enumerate(parsed) for part in message.images]
        policy = _check_images(located, parsed, images) if located else None
        completion = CompletionRequest(
            messages=messages,
            max_tokens=self.max_completion_tokens
            if self.max_completion_tokens is not None
            else self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=self.seed,
            repetition_penalty=self.repetition_penalty,
            stop=self._stop_sequences(),
            tools=offered,
            tool_choice=choice,
            parallel_tool_calls=self.parallel_tool_calls is not False,
        )
        if check is not None:
            check(completion)
        if policy is None:
            return completion
        pending = [_pending(index, part, policy) for index, part in located]
        return completion.model_copy(update={"images": pending})

    def _refuse_unsupported(self) -> None:
        if self.n is not None and self.n != 1:
            raise UnsupportedParameter("n", "Only n=1 is supported: one pipeline, one answer.")
        for name in ("presence_penalty", "frequency_penalty"):
            if getattr(self, name):
                raise UnsupportedParameter(name, f"{name} is not supported; repetition_penalty is.")
        for name in ("functions", "function_call"):
            if getattr(self, name):
                raise UnsupportedParameter(name, _LEGACY_FUNCTIONS)
        if self.response_format and self.response_format.get("type") not in (None, "text"):
            raise UnsupportedParameter(
                "response_format", "Only the text response format is supported."
            )
        if self.logprobs:
            raise UnsupportedParameter("logprobs", "Log probabilities are not supported.")
        last = self.messages[-1]
        if last.role not in ("user", "tool"):
            raise UnsupportedParameter(
                "messages",
                "The last message must come from the user or carry a tool's result; "
                "there is nothing to answer.",
            )

    def _tool_choice(self) -> Literal["auto", "none"]:
        """``tool_choice`` as yonga honours it; forcing a call is refused."""
        choice = self.tool_choice
        if choice is None or choice == "auto":
            return "auto"
        if choice == "none":
            return "none"
        if choice == "required" or isinstance(choice, dict):
            raise UnsupportedParameter(
                "tool_choice",
                "Forcing a tool call is not supported; use tool_choice auto or none.",
            )
        raise InvalidParameter("tool_choice", "tool_choice must be auto, none or required.")

    def _tools(self, policy: ToolPolicy | None) -> list[dict[str, Any]]:
        """The offered tools, validated and normalised; refused when the model takes none."""
        if not self.tools:
            return []
        if policy is None or not policy.accepted:
            reason = policy.refusal if policy is not None and policy.refusal else ""
            raise UnsupportedParameter(
                "tools",
                f"This model does not take tools. {reason}".rstrip(),
                code=TOOLS_UNSUPPORTED,
            )
        if len(self.tools) > MAX_TOOLS:
            raise InvalidParameter("tools", f"At most {MAX_TOOLS} tools are accepted.")
        tools: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, tool in enumerate(self.tools):
            normalised = _tool(f"tools[{index}]", tool)
            name = normalised["function"]["name"]
            if name in seen:
                raise InvalidParameter(
                    f"tools[{index}].function.name", f"Two tools are named {name!r}."
                )
            seen.add(name)
            tools.append(normalised)
        return tools

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


_ROLES: dict[str, Role] = {
    "system": "system",
    "developer": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
}


def wire_tool_call(call: ToolCall, *, index: int | None = None) -> dict[str, object]:
    """``call`` as OpenAI writes one in a message, or in a stream delta with ``index``."""
    payload: dict[str, object] = {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": call.arguments},
    }
    return payload if index is None else {"index": index, **payload}


def _tool(param: str, tool: object) -> dict[str, Any]:
    """One entry of ``tools``, checked: a function with a name and, if any, object parameters."""
    if not isinstance(tool, dict) or tool.get("type") != "function":
        raise UnsupportedParameter(param, "Only tools of type function are supported.")
    function = tool.get("function")
    if not isinstance(function, dict):
        raise InvalidParameter(f"{param}.function", "A function tool needs a function object.")
    name = function.get("name")
    if not isinstance(name, str) or not _FUNCTION_NAME.fullmatch(name):
        raise InvalidParameter(
            f"{param}.function.name",
            "A function name is 1 to 64 letters, digits, underscores or dashes.",
        )
    normalised: dict[str, Any] = {"name": name}
    description = function.get("description")
    if description is not None:
        if not isinstance(description, str):
            raise InvalidParameter(f"{param}.function.description", "A description is text.")
        normalised["description"] = description
    parameters = function.get("parameters")
    if parameters is not None:
        if not isinstance(parameters, dict) or parameters.get("type", "object") != "object":
            raise InvalidParameter(
                f"{param}.function.parameters",
                "Function parameters are a JSON Schema object with type object.",
            )
        normalised["parameters"] = parameters
    return {"type": "function", "function": normalised}


class _ImagePart(NamedTuple):
    """An image content part, located for error messages but not yet decoded."""

    param: str
    """``messages[i].content[j]``."""
    url: str


class _Parsed(NamedTuple):
    role: Role
    text: str
    images: list[_ImagePart]
    tool_calls: list[ToolCall]
    tool_call_id: str | None


def _parse(index: int, message: WireMessage) -> _Parsed:
    role = _ROLES.get(message.role)
    if role is None:
        raise UnsupportedParameter(
            f"messages[{index}].role",
            f"Role {message.role!r} is not supported; use system, user, assistant or tool.",
        )
    if message.function_call:
        raise UnsupportedParameter(f"messages[{index}].function_call", _LEGACY_FUNCTIONS)
    calls = _tool_calls(index, message) if message.tool_calls else []
    call_id: str | None = None
    if role == "tool":
        call_id = message.tool_call_id
        if not isinstance(call_id, str) or not call_id:
            raise InvalidParameter(
                f"messages[{index}].tool_call_id",
                "A tool message needs the tool_call_id of the call it answers.",
            )
    text, images = _content(index, role, message.content)
    return _Parsed(role, text, images, calls, call_id)


def _tool_calls(index: int, message: WireMessage) -> list[ToolCall]:
    """An assistant message's ``tool_calls``, checked."""
    if message.role != "assistant":
        raise UnsupportedParameter(
            f"messages[{index}].tool_calls", "Only assistant messages carry tool_calls."
        )
    calls: list[ToolCall] = []
    for position, item in enumerate(message.tool_calls or []):
        param = f"messages[{index}].tool_calls[{position}]"
        function = item.get("function") if isinstance(item, dict) else None
        if (
            not isinstance(item, dict)
            or item.get("type", "function") != "function"
            or not isinstance(item.get("id"), str)
            or not item["id"]
            or not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
        ):
            raise InvalidParameter(
                param, "A tool call needs an id, type function and a function with a name."
            )
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        if not isinstance(arguments, str):
            raise InvalidParameter(
                f"{param}.function.arguments", "Tool call arguments are JSON text."
            )
        if _nested_deeper_than(arguments, MAX_ARGUMENT_DEPTH):
            raise InvalidParameter(
                f"{param}.function.arguments", "Tool call arguments are nested too deeply."
            )
        try:
            json.loads(arguments)
        except RecursionError:
            raise InvalidParameter(
                f"{param}.function.arguments", "Tool call arguments are nested too deeply."
            ) from None
        except ValueError:
            pass  # not JSON: it reaches the chat template as text
        calls.append(ToolCall(id=item["id"], name=function["name"], arguments=arguments))
    return calls


def _nested_deeper_than(text: str, limit: int) -> bool:
    """Whether JSON ``text`` opens more than ``limit`` arrays or objects at once.

    Brackets inside strings do not count. The text need not be valid JSON:
    only its depth is measured, so this runs before anything parses it.
    """
    depth = 0
    in_string = escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > limit:
                return True
        elif character in "]}":
            depth -= 1
    return False


def _content(
    index: int, role: Role, content: str | list[dict[str, Any]] | None
) -> tuple[str, list[_ImagePart]]:
    """A message's text, and its image parts located for error messages."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
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
    return "".join(texts), images


def _conversation(parsed: Sequence[_Parsed], *, offered: bool) -> list[ChatMessage]:
    """The messages in yonga's terms, with every tool turn checked against its call.

    A ``tool`` message must answer a call of the assistant message in front of
    it (other tool messages may sit between them), and it takes that call's
    function name. Tool turns need ``tools`` in the request.
    """
    messages: list[ChatMessage] = []
    calls: dict[str, ToolCall] = {}
    for index, message in enumerate(parsed):
        name: str | None = None
        if (message.tool_calls or message.role == "tool") and not offered:
            field = "tool_calls" if message.tool_calls else "role"
            raise InvalidParameter(
                f"messages[{index}].{field}",
                "A conversation with tool calls or results needs the tools parameter.",
            )
        if message.role == "tool":
            call = calls.get(message.tool_call_id or "")
            if call is None:
                raise InvalidParameter(
                    f"messages[{index}].tool_call_id",
                    "A tool message must answer a call of the assistant message before it.",
                )
            name = call.name
        else:
            calls = {call.id: call for call in message.tool_calls}
        messages.append(
            ChatMessage(
                role=message.role,
                content=message.text,
                tool_calls=message.tool_calls,
                tool_call_id=message.tool_call_id,
                name=name,
            )
        )
    return messages


def _check_images(
    located: list[tuple[int, _ImagePart]], parsed: Sequence[_Parsed], policy: ImagePolicy | None
) -> ImagePolicy:
    """Refuse what ``policy`` does not allow; return it when it allows these images."""
    last = len(parsed) - 1
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
    if earlier is not None and (not policy.carry_forward or parsed[last].role != "user"):
        raise UnsupportedParameter(earlier.param, EARLIER_TURN_REASON, code=IMAGES_EARLIER_TURN)
    return policy


def _pending(message: int, part: _ImagePart, policy: ImagePolicy) -> PendingImage:
    """``part`` with its header checked, to be decoded once the request holds the pipeline."""
    with _refused_as(part):
        check_data_url(part.url, max_pixels=policy.max_pixels)
    return PendingImage(message=message, decode=partial(_decode, part, policy))


def _decode(part: _ImagePart, policy: ImagePolicy) -> ChatImage:
    with _refused_as(part):
        return decode_data_url(part.url, max_pixels=policy.max_pixels)


@contextmanager
def _refused_as(part: _ImagePart) -> Iterator[None]:
    """Turn an :class:`ImageRefused` into the 400 naming ``part``'s URL."""
    param = f"{part.param}.image_url.url"
    try:
        yield
    except ImageRefused as refusal:
        if refusal.unsupported:
            raise UnsupportedParameter(param, str(refusal)) from None
        raise InvalidParameter(param, str(refusal)) from None
