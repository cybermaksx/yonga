"""``yonga serve`` with images: data URLs only, VLM artifacts only, newest turn only.

Driven through FastAPI's test client over fake runtimes, like
``test_serve_api.py``; the images are tiny PNGs made in memory. What the fake
runtime records is what the GenAI adapter would hand the pipeline.
"""

from __future__ import annotations

import asyncio
import base64
import io
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("PIL")

from fastapi.testclient import TestClient
from PIL import Image

from tests.unit.test_chat_service import build_service, make_manifest, write_artifact
from tests.unit.test_serve_api import StreamingRuntime, content_of
from yonga.adapters.runtime_genai import ChatHistory, ChatImage, ChatMessage, GenerationResult
from yonga.api.app import create_app
from yonga.api.images import decode_data_url
from yonga.application.serve import (
    MAX_IMAGES,
    CompletionRequest,
    ModelServer,
    PendingImage,
    carry_images_forward,
)
from yonga.domain.artifacts import RecommendedRuntime
from yonga.domain.benchmarks import GenerationSettings
from yonga.domain.diagnostics import ConfigurationError, FailureCategory, diagnostic
from yonga.domain.models import PipelineKind

RED, BLUE = (255, 0, 0), (0, 0, 255)
#: Where the wire layer looks up the header check and the decoder, for tests
#: that watch or forbid them.
CHECK = "yonga.api.openai.check_data_url"
DECODE = "yonga.api.openai.decode_data_url"

Seen = list[tuple[str, str, list[tuple[int, int, bytes]]]]


class SeeingRuntime(StreamingRuntime):
    """A streaming fake that also records the images each message carried."""

    def __init__(self, answers: list[str | BaseException]) -> None:
        super().__init__(answers)
        self.seen: list[Seen] = []

    def generate(self, history: ChatHistory, config: GenerationSettings) -> GenerationResult:
        self.seen.append(
            [
                (m.role, m.content, [(i.width, i.height, i.pixels[:3]) for i in m.images])
                for m in history.messages
            ]
        )
        return super().generate(history, config)


def make_server(
    tmp_path: Path,
    answers: list[str | BaseException],
    *,
    pipeline: PipelineKind = PipelineKind.VLM,
    carry_images_forward: bool = False,
) -> tuple[ModelServer, SeeingRuntime]:
    service, factory, _ = build_service(tmp_path)
    runtime = SeeingRuntime(answers)
    factory.runtime = runtime
    manifest = make_manifest(
        pipeline=pipeline,
        recommended_runtime=RecommendedRuntime(device="GPU", pipeline=pipeline),
    )
    session = service.open_artifact(write_artifact(tmp_path, manifest))
    server = ModelServer(session, name="qwen", carry_images_forward=carry_images_forward)
    server.load()
    return server, runtime


def client_for(server: ModelServer, *, max_body_bytes: int | None = None) -> TestClient:
    if max_body_bytes is None:
        return TestClient(create_app(server))
    return TestClient(create_app(server, max_body_bytes=max_body_bytes))


def png(color: tuple[int, int, int] = RED, size: tuple[int, int] = (2, 2)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def image_part(url: str) -> dict[str, object]:
    return {"type": "image_url", "image_url": {"url": url, "detail": "low"}}


def text_part(text: str) -> dict[str, object]:
    return {"type": "text", "text": text}


def ask(*messages: dict[str, object], **extra: object) -> dict[str, object]:
    return {"model": "qwen", "messages": list(messages), **extra}


def user(*parts: dict[str, object]) -> dict[str, object]:
    return {"role": "user", "content": list(parts)}


def post(client: TestClient, body: dict[str, object]) -> dict[str, object]:
    response = client.post("/v1/chat/completions", json=body)
    payload: dict[str, object] = response.json()
    payload["status"] = response.status_code
    return payload


def error_of(payload: dict[str, object]) -> dict[str, object]:
    error = payload["error"]
    assert isinstance(error, dict)
    return error


# --- what is served -------------------------------------------------------------------


def test_an_image_reaches_the_runtime_with_the_newest_user_message(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Red."])
    body = post(client_for(server), ask(user(text_part("Which colour? "), image_part(png()))))

    assert body["status"] == 200
    assert body["choices"][0]["message"]["content"] == "Red."  # type: ignore[index]
    assert runtime.seen == [[("user", "Which colour? ", [(2, 2, bytes(RED))])]]


def test_images_keep_their_order(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Red, blue."])
    post(client_for(server), ask(user(image_part(png(RED)), image_part(png(BLUE)), text_part("?"))))

    ((_, _, images),) = runtime.seen[0]
    assert [pixels for _, _, pixels in images] == [bytes(RED), bytes(BLUE)]


def test_an_image_without_any_text_is_still_a_question(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["A red square."])
    body = post(client_for(server), ask(user(image_part(png()))))

    assert body["status"] == 200
    assert runtime.seen[0][0][2], "the image went through"


def test_a_streamed_answer_about_an_image(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Red."])
    response = client_for(server).post(
        "/v1/chat/completions", json=ask(user(image_part(png()), text_part("?")), stream=True)
    )

    assert content_of(response.text) == "Red."
    assert runtime.seen[0][0][2]


def test_text_only_requests_to_a_vlm_are_unchanged(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])
    post(client_for(server), ask({"role": "user", "content": "Hello"}))

    assert runtime.seen == [[("user", "Hello", [])]]


def test_images_are_decoded_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoded_on_loop: list[bool] = []
    real = decode_data_url

    def watching(url: str, *, max_pixels: int) -> ChatImage:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            decoded_on_loop.append(False)
        else:
            decoded_on_loop.append(True)
        return real(url, max_pixels=max_pixels)

    monkeypatch.setattr(DECODE, watching)
    server, _ = make_server(tmp_path, ["Red."])
    post(client_for(server), ask(user(image_part(png()))))

    assert decoded_on_loop == [False]


# --- what is refused, and that nothing is decoded first -------------------------------


@pytest.fixture
def no_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(url: str, *, max_pixels: int) -> ChatImage:
        raise AssertionError("a request that is refused anyway must not read images")

    monkeypatch.setattr(CHECK, refuse)
    monkeypatch.setattr(DECODE, refuse)


@pytest.mark.usefixtures("no_decoding")
@pytest.mark.parametrize(
    ("extra", "code", "param"),
    [
        ({"stop": [""]}, "invalid_request", "stop"),
        ({"stop": ["a", "b", "c", "d", "e"]}, "invalid_request", "stop"),
        ({"temperature": 9.0}, "serve.request.temperature", "temperature"),
        ({"top_p": 0.0}, "serve.request.top_p", "top_p"),
        ({"max_tokens": 0}, "serve.request.max_tokens", "max_tokens"),
    ],
)
def test_a_request_refused_for_its_settings_reads_no_image(
    tmp_path: Path, extra: dict[str, object], code: str, param: str
) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])
    parts = [image_part(png()) for _ in range(MAX_IMAGES)]
    body = post(client_for(server), ask(user(*parts), **extra))

    assert body["status"] == 400
    assert error_of(body)["code"] == code
    assert error_of(body)["param"] == param
    assert runtime.seen == []


@pytest.mark.usefixtures("no_decoding")
def test_an_image_for_a_text_only_model_is_refused(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Hi."], pipeline=PipelineKind.LLM)
    body = post(client_for(server), ask(user(text_part("?"), image_part(png()))))

    assert body["status"] == 400
    error = error_of(body)
    assert error["code"] == "serve.request.images_unsupported"
    assert error["param"] == "messages[0].content[1]"
    assert runtime.seen == []


@pytest.mark.parametrize(
    "url",
    [
        "https://example.invalid/cat.png",
        "http://169.254.169.254/latest/meta-data/",
        "file:///etc/passwd",
        "data:image/svg+xml;base64,PHN2Zz4=",
    ],
)
def test_a_url_that_is_not_an_inline_image_is_refused(tmp_path: Path, url: str) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])
    body = post(client_for(server), ask(user(text_part("?"), image_part(url))))

    assert body["status"] == 400
    error = error_of(body)
    assert error["code"] == "unsupported_parameter"
    assert error["param"] == "messages[0].content[1].image_url.url"
    assert url not in str(error["message"])
    assert runtime.seen == []


@pytest.mark.usefixtures("no_decoding")
def test_an_image_in_an_earlier_message_is_refused_and_says_why(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])
    conversation = ask(
        user(text_part("What is this?"), image_part(png())),
        {"role": "assistant", "content": "A red square."},
        {"role": "user", "content": "Are you sure?"},
    )
    body = post(client_for(server), conversation)

    assert body["status"] == 400
    error = error_of(body)
    assert error["code"] == "serve.request.images_earlier_turn"
    assert error["param"] == "messages[0].content[1]"
    assert "--carry-images-forward" in str(error["message"])
    assert runtime.seen == []


@pytest.mark.usefixtures("no_decoding")
def test_too_many_images_are_refused(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, ["Hi."])
    parts = [image_part(png()) for _ in range(MAX_IMAGES + 1)]
    body = post(client_for(server), ask(user(*parts)))

    assert body["status"] == 400
    assert error_of(body)["code"] == "serve.request.too_many_images"
    assert error_of(body)["param"] == "messages"


def test_the_most_images_allowed_are_served(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])
    body = post(client_for(server), ask(user(*[image_part(png()) for _ in range(MAX_IMAGES)])))

    assert body["status"] == 200
    assert len(runtime.seen[0][0][2]) == MAX_IMAGES


@pytest.mark.usefixtures("no_decoding")
@pytest.mark.parametrize("role", ["system", "assistant"])
def test_an_image_from_anyone_but_the_user_is_refused(tmp_path: Path, role: str) -> None:
    server, _ = make_server(tmp_path, ["Hi."])
    body = post(
        client_for(server),
        ask({"role": role, "content": [image_part(png())]}, {"role": "user", "content": "?"}),
    )

    assert body["status"] == 400
    assert error_of(body)["param"] == "messages[0].content[0]"


@pytest.mark.parametrize("image_url", [{}, {"url": 7}, "data:image/png;base64,AAAA", None])
def test_an_image_part_without_a_url_is_refused(tmp_path: Path, image_url: object) -> None:
    server, _ = make_server(tmp_path, ["Hi."])
    part = {"type": "image_url", "image_url": image_url}
    body = post(client_for(server), ask(user(part)))

    assert body["status"] == 400
    assert error_of(body)["param"] == "messages[0].content[0].image_url.url"


@pytest.mark.parametrize(
    "url",
    [
        "data:image/png;base64,bm90IGFuIGltYWdl",  # "not an image"
        "data:image/png;base64,***",
    ],
)
def test_an_image_that_does_not_decode_is_a_400(tmp_path: Path, url: str) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])
    body = post(client_for(server), ask(user(image_part(url))))

    assert body["status"] == 400
    error = error_of(body)
    assert error["code"] == "invalid_request"
    assert error["param"] == "messages[0].content[0].image_url.url"
    assert runtime.seen == []


def test_an_image_over_the_pixel_limit_is_refused_not_scaled(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])
    server._image_policy = server.image_policy.model_copy(update={"max_pixels": 3})
    body = post(client_for(server), ask(user(image_part(png(size=(2, 2))))))

    assert body["status"] == 400
    assert "more than 3 pixels" in str(error_of(body)["message"])
    assert runtime.seen == []


def test_an_image_larger_than_the_body_limit_is_a_413(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])
    body = ask(user(image_part(png(size=(64, 64)))))
    response = client_for(server, max_body_bytes=200).post("/v1/chat/completions", json=body)

    assert response.status_code == 413
    assert runtime.seen == []


def test_without_pillow_an_image_request_names_the_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(url: str, *, max_pixels: int) -> None:
        raise ConfigurationError(
            diagnostic(
                "serve.images.dependency_missing",
                FailureCategory.PACKAGE_ABI_MISMATCH,
                "Image input needs Pillow",
                "Pillow is not installed; the serve extra provides it.",
                remediation=["pip install 'yonga[serve]'"],
            )
        )

    monkeypatch.setattr(CHECK, missing)
    server, _ = make_server(tmp_path, ["Hi."])
    body = post(client_for(server), ask(user(image_part(png()))))

    assert body["status"] == 500
    assert error_of(body)["code"] == "serve.images.dependency_missing"


# --- images whose pixels only fail once decoded ---------------------------------------


def truncated_png() -> str:
    """A PNG whose header is fine and whose pixel data stops halfway."""
    buffer = io.BytesIO()
    Image.effect_noise((32, 32), 50).convert("RGB").save(buffer, format="PNG")
    data = buffer.getvalue()[: len(buffer.getvalue()) // 2]
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def test_pixels_that_fail_to_decode_are_a_400_naming_the_image(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])
    body = post(client_for(server), ask(user(text_part("?"), image_part(truncated_png()))))

    assert body["status"] == 400
    error = error_of(body)
    assert error["code"] == "invalid_request"
    assert error["param"] == "messages[0].content[1].image_url.url"
    assert runtime.seen == []


def test_pixels_that_fail_to_decode_are_the_same_400_when_streamed(tmp_path: Path) -> None:
    """Found before generation starts, so the stream never opens."""
    server, runtime = make_server(tmp_path, ["Hi."])
    response = client_for(server).post(
        "/v1/chat/completions", json=ask(user(image_part(truncated_png())), stream=True)
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "The image could not be decoded as PNG, JPEG, WebP or GIF.",
            "type": "invalid_request_error",
            "param": "messages[0].content[0].image_url.url",
            "code": "invalid_request",
        }
    }
    assert runtime.seen == []


def test_an_unexpected_decoder_failure_is_a_400_not_a_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt EXIF block once escaped Pillow as ``struct.error``."""

    def failing(self: object, *args: object) -> object:
        raise struct.error("required argument is not an integer")

    monkeypatch.setattr(Image.Image, "transpose", failing)
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    Image.new("RGB", (4, 2), RED).save(buffer, format="JPEG", exif=exif)
    url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    server, runtime = make_server(tmp_path, ["Hi."])
    body = post(client_for(server), ask(user(image_part(url))))

    assert body["status"] == 400
    assert error_of(body)["code"] == "invalid_request"
    assert error_of(body)["param"] == "messages[0].content[0].image_url.url"
    assert runtime.seen == []


# --- decoded pixels are held by one request at a time ---------------------------------


class WatchedLock:
    """The server's lock, counting who waits for it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._changed = threading.Condition()
        self.waiting = 0

    def locked(self) -> bool:
        return self._lock.locked()

    def __enter__(self) -> None:
        with self._changed:
            self.waiting += 1
            self._changed.notify_all()
        self._lock.acquire()
        with self._changed:
            self.waiting -= 1

    def __exit__(self, *_: object) -> None:
        self._lock.release()

    def wait_for(self, waiting: int) -> None:
        with self._changed:
            assert self._changed.wait_for(lambda: self.waiting >= waiting, timeout=10)


def test_images_are_decoded_only_while_their_request_holds_the_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, runtime = make_server(tmp_path, ["Red.", "Blue."])
    lock = WatchedLock()
    server._lock = lock  # type: ignore[assignment]
    decoded: list[bool] = []
    real = decode_data_url

    def watching(url: str, *, max_pixels: int) -> ChatImage:
        decoded.append(lock.locked())
        return real(url, max_pixels=max_pixels)

    monkeypatch.setattr(DECODE, watching)
    client = client_for(server)
    with ThreadPoolExecutor(2) as pool, lock:
        sent = [
            pool.submit(post, client, ask(user(image_part(png(color)), text_part("?"))))
            for color in (RED, BLUE)
        ]
        lock.wait_for(2)
        assert decoded == [], "requests waiting for the pipeline hold no decoded image"
    bodies = [future.result() for future in sent]

    assert [body["status"] for body in bodies] == [200, 200]
    assert decoded == [True, True]
    assert len(runtime.seen) == 2


# --- carrying earlier images forward --------------------------------------------------


def test_carried_forward_images_move_to_the_newest_message_with_a_note(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Yes."], carry_images_forward=True)
    conversation = ask(
        user(text_part("What is this?"), image_part(png(RED))),
        {"role": "assistant", "content": "A red square."},
        user(text_part("And this one?"), image_part(png(BLUE))),
    )
    body = post(client_for(server), conversation)

    assert body["status"] == 200
    first, answer, last = runtime.seen[0]
    assert first == ("user", "What is this?", [])
    assert answer == ("assistant", "A red square.", [])
    role, content, images = last
    assert role == "user"
    assert content == (
        "[Image 1 was attached to an earlier message in this conversation.]\nAnd this one?"
    )
    assert [pixels for _, _, pixels in images] == [bytes(RED), bytes(BLUE)]


def test_carrying_forward_still_holds_the_image_limit(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, ["Yes."], carry_images_forward=True)
    conversation = ask(
        user(*[image_part(png()) for _ in range(3)]),
        {"role": "assistant", "content": "Three."},
        user(image_part(png()), image_part(png())),
    )
    body = post(client_for(server), conversation)

    assert error_of(body)["code"] == "serve.request.too_many_images"


def test_the_carry_note_counts_the_images() -> None:
    image = ChatImage(width=1, height=1, pixels=b"\x00\x00\x00")
    messages = [
        ChatMessage(role="user", content="a", images=[image, image]),
        ChatMessage(role="assistant", content="b"),
        ChatMessage(role="user", content=""),
    ]

    moved = carry_images_forward(messages)

    assert moved[-1].content == (
        "[Images 1-2 were attached to earlier messages in this conversation.]"
    )
    assert len(moved[-1].images) == 2
    assert moved[0].images == []
    assert messages[0].images, "the caller's messages are left alone"


def test_without_earlier_images_carrying_forward_changes_nothing() -> None:
    messages = [ChatMessage(role="user", content="a")]

    assert carry_images_forward(messages) == messages


# --- the service holds the same line for callers other than the API -------------------


def image_request(*messages: ChatMessage) -> CompletionRequest:
    return CompletionRequest(messages=list(messages))


def red() -> ChatImage:
    return ChatImage(width=1, height=1, pixels=bytes(RED))


def test_the_service_refuses_images_for_a_text_only_model(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Hi."], pipeline=PipelineKind.LLM)

    with pytest.raises(ConfigurationError) as refused:
        server.complete(image_request(ChatMessage(role="user", content="?", images=[red()])))

    assert refused.value.diagnostic.code == "serve.request.images_unsupported"
    assert runtime.seen == []


def test_the_service_refuses_an_earlier_image_unless_carrying_forward(tmp_path: Path) -> None:
    request = image_request(
        ChatMessage(role="user", content="?", images=[red()]),
        ChatMessage(role="assistant", content="Red."),
        ChatMessage(role="user", content="Sure?"),
    )
    refusing, _ = make_server(tmp_path / "a", ["Hi."])
    carrying, runtime = make_server(tmp_path / "b", ["Yes."], carry_images_forward=True)

    with pytest.raises(ConfigurationError) as refused:
        refusing.complete(request)
    turn = carrying.complete(request)

    assert refused.value.diagnostic.code == "serve.request.images_earlier_turn"
    assert turn.diagnostic is None
    assert runtime.seen[0][-1][2] == [(1, 1, bytes(RED))]


def test_the_service_decodes_pending_images_onto_their_messages(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Yes."], carry_images_forward=True)
    blue = ChatImage(width=1, height=1, pixels=bytes(BLUE))
    request = image_request(
        ChatMessage(role="user", content="?"),
        ChatMessage(role="assistant", content="Hm."),
        ChatMessage(role="user", content="This?", images=[red()]),
    ).model_copy(update={"images": [PendingImage(message=0, decode=lambda: blue)]})

    turn = server.complete(request)

    assert turn.diagnostic is None
    assert runtime.seen[0][-1][2] == [(1, 1, bytes(BLUE)), (1, 1, bytes(RED))]


def test_the_service_refuses_a_pending_image_of_no_message(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])
    request = image_request(ChatMessage(role="user", content="?")).model_copy(
        update={"images": [PendingImage(message=1, decode=red)]}
    )

    with pytest.raises(ConfigurationError) as refused:
        server.complete(request)

    assert refused.value.diagnostic.code == "serve.request.image_message"
    assert runtime.seen == []
