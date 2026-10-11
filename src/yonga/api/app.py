"""The FastAPI application: ``/health``, ``/v1/models`` and ``/v1/chat/completions``.

Streaming follows OpenAI's server-sent events shape: a first chunk carrying the
assistant role, one chunk per released piece of text, one chunk per tool call
(whole, once generation has ended), a final chunk with the ``finish_reason``,
an optional usage chunk, and ``data: [DONE]``. Generation
runs on a worker thread; a client that disconnects makes the next chunk ask
the pipeline to stop, so an abandoned request does not hold the accelerator
for the rest of its token budget. The status waits until the pipeline has
accepted the prompt, so a request it refuses before generating (a prompt over
an NPU's compiled length) gets the same 400 with ``stream`` as without it.

Security posture: the server binds to loopback unless told otherwise (the CLI
decides that), keeps no request content anywhere, and when an API key is
configured compares it in constant time. The key itself is never logged or
echoed back.

Request size is bounded so one request cannot exhaust memory before it is even
validated. A body larger than ``max_body_bytes`` (by default
:data:`DEFAULT_MAX_BODY_BYTES`, far more than any model's context holds as
text, and what ``serve --max-body-mb`` sets) is refused with ``413`` and the
error code ``request_too_large``. That limit is also what bounds the encoded
size of the images a request embeds; their decoded pixels are held only by the
request that holds the pipeline (see :class:`~yonga.application.serve.PendingImage`),
so requests waiting for it cost their bodies and no more. The limit is
enforced on the bytes actually received, not only on the declared
``Content-Length``, so a chunked upload is held to it too. The number of
messages in one request is bounded separately (see
:data:`yonga.api.openai.MAX_MESSAGES`) and refused with ``400``. With an API
key configured, authentication runs first, so the server answers ``401``
without reading the body of a request that lacks the key.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import re
import threading
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from yonga import __version__
from yonga.api.openai import ChatCompletionRequest, ParameterError, wire_tool_call
from yonga.application.chat import ChatTurn
from yonga.application.serve import (
    PARAM_CONTEXT,
    PROMPT_TOO_LONG,
    CompletionRequest,
    ModelServer,
    completion_id,
    finish_reason,
)
from yonga.domain.diagnostics import Diagnostic, FailureCategory, YongaError

__all__ = ["DEFAULT_MAX_BODY_BYTES", "create_app"]

#: Largest request body accepted by default: 8 MiB. A chat history long enough
#: to fill a large context window is a small fraction of this as JSON text; an
#: image embedded as base64 takes a third more than its file, so this leaves
#: room for about 6 MB of images.
DEFAULT_MAX_BODY_BYTES = 8 * 1024 * 1024

#: Categories that describe the request rather than the server.
_CLIENT_ERRORS = frozenset({FailureCategory.CONFIGURATION_ERROR})

#: Diagnostics reported under OpenAI's own ``code`` and ``param``, so a client
#: that handles that case (trimming a history that no longer fits) recognises it.
_OPENAI_CODES = {PROMPT_TOO_LONG: ("context_length_exceeded", "messages")}

#: Where OpenVINO says which of its source lines raised: "Check '...' failed at
#: /path/file.cpp:359:" or "Exception from src/file.cpp:112:". It names the
#: vendor's build machine and nothing a client can act on.
_SOURCE_LOCATION = re.compile(r"(?:Check '[^\n]*?' failed at |Exception from )\S+?:\d+:\s*")

#: The OpenAI-style ``code`` for an HTTP error raised by routing or by a limit.
_HTTP_ERROR_CODES = {
    404: "not_found",
    405: "method_not_allowed",
    413: "request_too_large",
}


def create_app(
    server: ModelServer,
    *,
    api_key: str | None = None,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
) -> FastAPI:
    """Build the application for ``server``.

    ``api_key`` enables bearer authentication. ``max_body_bytes`` caps the size
    of a request body; a larger one is refused with ``413`` (see the module
    docstring).
    """
    app = FastAPI(
        title="yonga",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    # Added before the authentication middleware, so it runs inside it.
    app.add_middleware(_BodyLimit, max_bytes=max_body_bytes)

    @app.middleware("http")
    async def authenticate(request: Request, call_next: Any) -> Any:
        if api_key is not None and request.url.path != "/health":
            supplied = request.headers.get("authorization", "")
            scheme, _, token = supplied.partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(
                token.encode(), api_key.encode()
            ):
                return _error(401, "Invalid or missing API key.", "invalid_api_key")
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def invalid_body(_request: Request, error: RequestValidationError) -> JSONResponse:
        first = error.errors()[0] if error.errors() else {}
        message, param = _validation_failure(first)
        return _error(400, message, "invalid_request", param=param)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request: Request, error: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_ERROR_CODES.get(error.status_code, "invalid_request")
        return _error(error.status_code, str(error.detail), code)

    @app.get("/health")
    def health() -> dict[str, object]:
        return server.health()

    @app.get("/v1/models")
    def models() -> dict[str, object]:
        return {"object": "list", "data": [_model_card(server)]}

    @app.get("/v1/models/{name:path}", response_model=None)
    def model(name: str) -> dict[str, object] | JSONResponse:
        if not server.model.answers_to(name):
            return _model_not_found(name)
        return _model_card(server)

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(body: ChatCompletionRequest) -> JSONResponse | StreamingResponse:
        if not server.model.answers_to(body.model):
            return _model_not_found(body.model)
        policy, tools, check = server.image_policy, server.tool_policy, server.generation_for
        try:
            if body.has_images:
                # Reading image headers takes CPU time; it must not hold up the event loop.
                completion = await asyncio.to_thread(
                    body.to_completion, policy, tools=tools, check=check
                )
            else:
                completion = body.to_completion(policy, tools=tools, check=check)
            limit = server.generation_for(completion).max_new_tokens
        except ParameterError as error:
            return _error(400, str(error), error.code, param=error.param)
        except YongaError as error:
            return _diagnostic_error(error.diagnostic)
        if body.stream:
            generation = _Generation(server, completion)
            return _EventStream(
                generation,
                _stream(server, generation, limit=limit, include_usage=body.include_usage),
            )
        try:
            turn = await asyncio.to_thread(server.complete, completion)
        except (ParameterError, YongaError) as error:
            # A ParameterError is an image whose header was fine but whose
            # pixels are not, found only when it is decoded for the pipeline.
            return _failure_response(error)
        if turn.diagnostic is not None:
            return _failure_response(turn)
        return JSONResponse(_completion(server, turn, limit=limit))

    return app


# --- responses ------------------------------------------------------------------------


def _model_card(server: ModelServer) -> dict[str, object]:
    model = server.model
    return {
        "id": model.id,
        "object": "model",
        "created": model.created,
        "owned_by": "yonga",
        "artifact_id": model.artifact_id,
        "device": model.device,
    }


def _usage(turn: ChatTurn) -> dict[str, int] | None:
    result = turn.result
    if result is None or result.input_tokens is None or result.output_tokens is None:
        return None
    return {
        "prompt_tokens": result.input_tokens,
        "completion_tokens": result.output_tokens,
        "total_tokens": result.input_tokens + result.output_tokens,
    }


def _completion(server: ModelServer, turn: ChatTurn, *, limit: int) -> dict[str, object]:
    result = turn.result
    # Stripped exactly as a streamed answer is (see StreamingSanitizer), so the
    # same request gives the same content whether or not it streams.
    text = result.display_text.strip() if result is not None else ""
    message: dict[str, object] = {"role": "assistant", "content": text}
    calls = result.tool_calls if result is not None else []
    if calls:
        # OpenAI's shape: no text beside the calls is null, not "".
        message = {
            "role": "assistant",
            "content": text or None,
            "tool_calls": [wire_tool_call(call) for call in calls],
        }
    payload: dict[str, object] = {
        "id": completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": server.model.id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason(turn, limit),
            }
        ],
    }
    usage = _usage(turn)
    if usage is not None:
        payload["usage"] = usage
    return payload


def _chunk(
    identifier: str,
    created: int,
    model: str,
    delta: dict[str, object],
    reason: str | None = None,
) -> str:
    payload = {
        "id": identifier,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class _Generation:
    """One streamed generation, run on a worker thread and relayed through a queue.

    The worker reports ``started`` the first time the pipeline hands back a
    chunk, which is once it has accepted the prompt. What a pipeline refuses
    before that -- a prompt longer than an NPU's compiled length, an image whose
    pixels do not decode, a model that does not load -- ends the work without
    it, and :class:`_EventStream` can still answer with a status.
    """

    def __init__(self, server: ModelServer, completion: CompletionRequest) -> None:
        self._server = server
        self._completion = completion
        self._queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
        self._abandoned = threading.Event()
        self._held: tuple[str, object] | None = None

    def start(self) -> None:
        """Start generating on a worker thread; call from the event loop."""
        loop = asyncio.get_running_loop()
        abandoned, started = self._abandoned, threading.Event()

        def deliver(kind: str, value: object) -> None:
            # The event loop is gone once the client has disconnected and the
            # response was torn down; there is then nobody left to deliver to.
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(self._queue.put_nowait, (kind, value))

        def should_stop() -> bool:
            # Asked on every chunk the pipeline streams, before it is released.
            if not started.is_set():
                started.set()
                deliver("started", None)
            return abandoned.is_set()

        def on_text(text: str) -> bool | None:
            if abandoned.is_set():
                return True  # ask the pipeline to stop
            deliver("text", text)
            return None

        def work() -> None:
            try:
                turn = self._server.complete(
                    self._completion, on_text=on_text, should_stop=should_stop
                )
                deliver("done", turn)
            except BaseException as error:
                deliver("error", error)

        threading.Thread(target=work, name="yonga-generate", daemon=True).start()

    async def first(self) -> tuple[str, object]:
        """Wait for the first event; :meth:`next` returns it again, unless it is ``started``."""
        event = await self._queue.get()
        if event[0] != "started":
            self._held = event
        return event

    async def next(self) -> tuple[str, object]:
        """The next event: ``started``, ``text``, ``done`` (a turn) or ``error``."""
        if self._held is not None:
            event, self._held = self._held, None
            return event
        return await self._queue.get()

    def abandon(self) -> None:
        """Ask the pipeline to stop at its next chunk: nobody is reading any more."""
        self._abandoned.set()


class _EventStream(StreamingResponse):
    """Server-sent events whose status waits until generation has started.

    A request refused before the pipeline produces anything gets the status
    and body the same request gets without ``stream``, so a client sees a 400
    it can act on (and does not retry) rather than a 200 whose only event is
    an error. Once generation has started the status is sent, and a failure
    can only be reported as an event (see :func:`_stream`). The role chunk
    that opens the stream waits with the status; it carries nothing a client
    needs before the answer follows it.
    """

    def __init__(self, generation: _Generation, events: AsyncIterator[str]) -> None:
        super().__init__(
            events,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        self._generation = generation

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        generation = self._generation
        generation.start()
        try:
            opening = await _opening(generation, receive)
        except BaseException:
            generation.abandon()
            raise
        if opening is None:
            # The client went away while the request waited for the pipeline.
            # Nobody reads this answer, but an application must give one, or
            # the middleware reports a request that got no response.
            generation.abandon()
            gone = _error(499, "The client closed the request.", "client_closed_request")
            await gone(scope, receive, send)
            return
        kind, value = opening
        if kind == "error" or (kind == "done" and _failed(value)):
            await _failure_response(value)(scope, receive, send)
            return
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Also when the client left before the first event was read.
            generation.abandon()


async def _opening(generation: _Generation, receive: Receive) -> tuple[str, object] | None:
    """The first event of ``generation``, or ``None`` if the client disconnects first."""
    first = asyncio.create_task(generation.first())
    gone = asyncio.create_task(_disconnect(receive))
    try:
        await asyncio.wait({first, gone}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        first.cancel()
        gone.cancel()
        await asyncio.gather(first, gone, return_exceptions=True)
    return first.result() if first.done() and not first.cancelled() else None


async def _disconnect(receive: Receive) -> None:
    """Return once the client has disconnected; the body has been read already."""
    while (await receive())["type"] != "http.disconnect":
        pass


def _failed(value: object) -> bool:
    """Whether a finished generation's turn carries a failure instead of an answer."""
    return not isinstance(value, ChatTurn) or value.diagnostic is not None


async def _stream(
    server: ModelServer,
    generation: _Generation,
    *,
    limit: int,
    include_usage: bool,
) -> AsyncIterator[str]:
    """Relay a started ``generation`` as server-sent events.

    The status is already sent, so a failure from here on is an ``error``
    event carrying the body a request refused without ``stream`` gets.
    """
    identifier, created, model = completion_id(), int(time.time()), server.model.id
    try:
        yield _chunk(identifier, created, model, {"role": "assistant", "content": ""})
        while True:
            kind, value = await generation.next()
            if kind == "started":
                continue
            if kind == "text":
                yield _chunk(identifier, created, model, {"content": str(value)})
                continue
            if kind == "done" and isinstance(value, ChatTurn) and value.diagnostic is None:
                calls = value.result.tool_calls if value.result is not None else []
                for index, call in enumerate(calls):
                    delta: dict[str, object] = {"tool_calls": [wire_tool_call(call, index=index)]}
                    yield _chunk(identifier, created, model, delta)
                yield _chunk(identifier, created, model, {}, finish_reason(value, limit))
                usage = _usage(value)
                if include_usage and usage is not None:
                    payload = {
                        "id": identifier,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [],
                        "usage": usage,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
            else:
                _status, body = _failure(value)
                yield f"data: {json.dumps({'error': body})}\n\n"
            yield "data: [DONE]\n\n"
            return
    finally:
        generation.abandon()


# --- request limits -------------------------------------------------------------------


class _BodyTooLarge(HTTPException):
    """Raised from ``receive`` once a body passes the limit.

    FastAPI's own ``HTTPException`` on purpose: FastAPI turns any other error
    raised while reading a body into a generic ``400``.
    """

    def __init__(self, limit: int) -> None:
        super().__init__(413, _too_large_message(limit))


def _too_large_message(limit: int) -> str:
    return f"The request body is larger than this server accepts ({limit} bytes)."


class _BodyLimit:
    """Pure ASGI middleware that refuses request bodies larger than ``max_bytes``.

    A declared ``Content-Length`` above the limit is refused before any of the
    body is read; otherwise the bytes are counted as they arrive, which also
    covers chunked uploads that declare no length.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        declared = _content_length(scope)
        if declared is not None and declared > self._max_bytes:
            response = _error(413, _too_large_message(self._max_bytes), "request_too_large")
            await response(scope, receive, send)
            return
        received = 0

        async def limited() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_bytes:
                    raise _BodyTooLarge(self._max_bytes)
            return message

        await self._app(scope, limited, send)


def _content_length(scope: Scope) -> int | None:
    """The declared ``Content-Length``, or ``None`` when absent or unreadable."""
    for name, value in scope.get("headers", ()):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


# --- errors ---------------------------------------------------------------------------


def _error(status: int, message: str, code: str, *, param: str | None = None) -> JSONResponse:
    """A client error in OpenAI's shape; OpenAI types a bad API key this way too."""
    error = {"message": message, "type": "invalid_request_error", "param": param, "code": code}
    return JSONResponse({"error": error}, status_code=status)


def _model_not_found(name: str) -> JSONResponse:
    return _error(404, f"The model {name!r} is not served here.", "model_not_found", param="model")


def _diagnostic_error(report: Diagnostic) -> JSONResponse:
    status, body = _diagnostic_body(report)
    return JSONResponse({"error": body}, status_code=status)


def _failure_response(outcome: object) -> JSONResponse:
    status, body = _failure(outcome)
    return JSONResponse({"error": body}, status_code=status)


def _failure(outcome: object) -> tuple[int, dict[str, object]]:
    """The status and OpenAI-shaped error body for a generation that gave no answer.

    ``outcome`` is what it ended with: a :class:`ParameterError` (an image
    whose pixels do not decode), a typed error, a turn carrying a diagnostic,
    or anything else, which is the server's own failure and says no more.
    The same body is a response's, or a stream's ``error`` event once the
    stream has begun.
    """
    error: dict[str, object]
    if isinstance(outcome, ParameterError):
        error = {"message": str(outcome), "type": "invalid_request_error", "param": outcome.param}
        return 400, {**error, "code": outcome.code}
    if isinstance(outcome, YongaError | ChatTurn) and outcome.diagnostic is not None:
        return _diagnostic_body(outcome.diagnostic)
    error = {"message": "Generation failed.", "type": "server_error", "param": None}
    return 500, {**error, "code": "internal_error"}


def _diagnostic_body(report: Diagnostic) -> tuple[int, dict[str, object]]:
    """The status and OpenAI-shaped error body for a diagnostic.

    A refusal of the request is a 400 naming the parameter its diagnostic
    carries; anything else is the server's 500. Either way OpenVINO's source
    locations are left out of the message (see :data:`_SOURCE_LOCATION`).
    """
    client = report.category in _CLIENT_ERRORS
    code, param = _OPENAI_CODES.get(report.code, (report.code, None))
    if param is None and client:
        param = report.context.get(PARAM_CONTEXT)
    message = _SOURCE_LOCATION.sub("", f"{report.title}: {report.message}").strip()
    body: dict[str, object] = {
        "message": message,
        "type": "invalid_request_error" if client else "server_error",
        "param": param,
        "code": code,
    }
    return (400 if client else 500), body


def _validation_failure(error: dict[str, Any]) -> tuple[str, str | None]:
    """The message and ``param`` for the first error of a body that did not validate.

    ``param`` is the dotted path of the field, never a position in the body: a
    body that is not JSON names no field, and the decode error's offset goes
    into the message instead.
    """
    location = [part for part in error.get("loc", ()) if part != "body"]
    if error.get("type") == "json_invalid":
        offset = location[-1] if location and isinstance(location[-1], int) else None
        reason = str((error.get("ctx") or {}).get("error", "")).strip() or "it does not parse"
        where = f" at character {offset}" if offset is not None else ""
        return f"The request body is not valid JSON: {reason}{where}.", None
    if not location or not isinstance(location[0], str):
        # The body as a whole: sent without a JSON content type, or not an object.
        return (
            "The request body must be a JSON object sent with Content-Type: application/json.",
            None,
        )
    return str(error.get("msg", "Invalid request body.")), ".".join(map(str, location))
