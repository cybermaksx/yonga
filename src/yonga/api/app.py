"""The FastAPI application: ``/health``, ``/v1/models`` and ``/v1/chat/completions``.

Streaming follows OpenAI's server-sent events shape: a first chunk carrying the
assistant role, one chunk per released piece of text, a final chunk with the
``finish_reason``, an optional usage chunk, and ``data: [DONE]``. Generation
runs on a worker thread; a client that disconnects makes the next chunk ask
the pipeline to stop, so an abandoned request does not hold the accelerator
for the rest of its token budget.

Security posture: the server binds to loopback unless told otherwise (the CLI
decides that), keeps no request content anywhere, and when an API key is
configured compares it in constant time. The key itself is never logged or
echoed back.

Request size is bounded so one request cannot exhaust memory before it is even
validated. A body larger than ``max_body_bytes`` (by default
:data:`DEFAULT_MAX_BODY_BYTES`, far more than any model's context holds as
text, and what ``serve --max-body-mb`` sets) is refused with ``413`` and the
error code ``request_too_large``. That limit is also what bounds the encoded
size of the images a request embeds. The
limit is enforced on the bytes actually received, not only on the declared
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
from yonga.api.openai import ChatCompletionRequest, ParameterError
from yonga.application.chat import ChatTurn
from yonga.application.serve import (
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
                return _error(401, "Invalid or missing API key.", "invalid_api_key", kind="auth")
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def invalid_body(_request: Request, error: RequestValidationError) -> JSONResponse:
        first = error.errors()[0] if error.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        return _error(
            400, str(first.get("msg", "Invalid request body.")), "invalid_request", param=location
        )

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
        try:
            if body.has_images:
                # Decoding images takes CPU time; it must not hold up the event loop.
                completion = await asyncio.to_thread(body.to_completion, server.image_policy)
            else:
                completion = body.to_completion(server.image_policy)
            limit = server.generation_for(completion).max_new_tokens
        except ParameterError as error:
            return _error(400, str(error), error.code, param=error.param)
        except YongaError as error:
            return _diagnostic_error(error.diagnostic)
        if body.stream:
            return StreamingResponse(
                _stream(server, completion, limit=limit, include_usage=body.include_usage),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        turn = await asyncio.to_thread(server.complete, completion)
        if turn.diagnostic is not None:
            return _diagnostic_error(turn.diagnostic)
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
    payload: dict[str, object] = {
        "id": completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": server.model.id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
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
    delta: dict[str, str],
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


async def _stream(
    server: ModelServer,
    completion: CompletionRequest,
    *,
    limit: int,
    include_usage: bool,
) -> AsyncIterator[str]:
    """Run one generation on a worker thread and relay it as server-sent events."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
    abandoned = threading.Event()

    def deliver(kind: str, value: object) -> None:
        # The event loop is gone once the client has disconnected and the
        # response was torn down; there is then nobody left to deliver to.
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(queue.put_nowait, (kind, value))

    def on_text(text: str) -> bool | None:
        if abandoned.is_set():
            return True  # ask the pipeline to stop
        deliver("text", text)
        return None

    def work() -> None:
        try:
            deliver("done", server.complete(completion, on_text=on_text))
        except BaseException as error:
            deliver("error", error)

    identifier, created, model = completion_id(), int(time.time()), server.model.id
    threading.Thread(target=work, name="yonga-generate", daemon=True).start()
    try:
        yield _chunk(identifier, created, model, {"role": "assistant", "content": ""})
        while True:
            kind, value = await queue.get()
            if kind == "text":
                yield _chunk(identifier, created, model, {"content": str(value)})
                continue
            if kind == "done" and isinstance(value, ChatTurn) and value.diagnostic is None:
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
                report = (
                    value.diagnostic
                    if isinstance(value, ChatTurn) and value.diagnostic is not None
                    else value.diagnostic
                    if isinstance(value, YongaError)
                    else None
                )
                message = report.message if report is not None else "Generation failed."
                code = report.code if report is not None else "internal_error"
                yield f"data: {json.dumps({'error': {'message': message, 'code': code}})}\n\n"
            yield "data: [DONE]\n\n"
            return
    finally:
        abandoned.set()


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


def _error(
    status: int,
    message: str,
    code: str,
    *,
    param: str | None = None,
    kind: str = "invalid_request_error",
) -> JSONResponse:
    body = {"error": {"message": message, "type": kind, "param": param, "code": code}}
    return JSONResponse(body, status_code=status)


def _model_not_found(name: str) -> JSONResponse:
    return _error(404, f"The model {name!r} is not served here.", "model_not_found", param="model")


def _diagnostic_error(report: Diagnostic) -> JSONResponse:
    status = 400 if report.category in _CLIENT_ERRORS else 500
    kind = "invalid_request_error" if status == 400 else "server_error"
    return _error(status, f"{report.title}: {report.message}", report.code, kind=kind)
