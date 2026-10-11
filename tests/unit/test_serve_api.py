"""``yonga serve``: the OpenAI-compatible API over the same session ``chat`` uses.

The server is driven through FastAPI's test client against fake runtimes and
the real built-in registry, so what is asserted about reasoning, defaults and
model naming is what the shipped chat profile produces.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from tests.unit.test_chat_service import (
    SECRET,
    FakeRuntime,
    build_service,
    make_manifest,
    write_artifact,
)
from yonga.adapters.runtime_genai import ChatHistory, GenerationResult, TextCallback
from yonga.api.app import create_app
from yonga.api.openai import MAX_MESSAGES
from yonga.application.serve import PROMPT_TOO_LONG, ModelServer
from yonga.domain.benchmarks import GenerationSettings
from yonga.domain.diagnostics import FailureCategory, GenerationError, diagnostic

ARTIFACT_ID = "qwen35-9b--0123456789ab--gpu-int4-asym-g128"
THOUGHT = f"<think>{SECRET}</think>\n\n"


class StreamingRuntime(FakeRuntime):
    """A fake runtime that also streams, a few characters at a time."""

    def __init__(self, answers: list[str | BaseException], chunk: int = 3) -> None:
        super().__init__(answers)
        self.chunk = chunk
        self.stopped_early = False

    def generate_stream(
        self, history: ChatHistory, config: GenerationSettings, on_text: TextCallback
    ) -> GenerationResult:
        result = self.generate(history, config)
        for index in range(0, len(result.text), self.chunk):
            if on_text(result.text[index : index + self.chunk]):
                self.stopped_early = True
                break
        return result


class BreakingRuntime(StreamingRuntime):
    """Streams the start of an answer, then fails: the device went away mid-answer."""

    def __init__(self, failure: BaseException) -> None:
        super().__init__(["The answer is"])
        self.failure = failure

    def generate_stream(
        self, history: ChatHistory, config: GenerationSettings, on_text: TextCallback
    ) -> GenerationResult:
        super().generate_stream(history, config, on_text)
        raise self.failure


def make_server(
    tmp_path: Path,
    answers: list[str | BaseException],
    *,
    streaming: bool = True,
    chunk: int = 3,
    show_reasoning: bool = False,
    runtime: FakeRuntime | None = None,
) -> tuple[ModelServer, FakeRuntime]:
    service, factory, _ = build_service(tmp_path)
    if runtime is None:
        runtime = StreamingRuntime(answers, chunk) if streaming else FakeRuntime(answers)
    factory.runtime = runtime
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact)
    session.set_reasoning(show_reasoning)
    server = ModelServer(session, name="qwen")
    server.load()
    return server, runtime


def client_for(
    server: ModelServer, *, api_key: str | None = None, max_body_bytes: int | None = None
) -> TestClient:
    if max_body_bytes is None:
        return TestClient(create_app(server, api_key=api_key))
    return TestClient(create_app(server, api_key=api_key, max_body_bytes=max_body_bytes))


def ask(content: str | list[dict[str, object]] = "Hello", **extra: object) -> dict[str, object]:
    return {"model": "qwen", "messages": [{"role": "user", "content": content}], **extra}


def events(response_text: str) -> list[object]:
    parsed: list[object] = []
    for line in response_text.splitlines():
        if not line.startswith("data: "):
            continue
        body = line[len("data: ") :]
        parsed.append(body if body == "[DONE]" else json.loads(body))
    return parsed


# --- models and health ----------------------------------------------------------------


def test_models_lists_the_one_served_model(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, [])
    body = client_for(server).get("/v1/models").json()

    assert body["object"] == "list"
    (card,) = body["data"]
    assert card["id"] == "qwen"
    assert card["artifact_id"] == ARTIFACT_ID
    assert card["owned_by"] == "yonga"


def test_health_reports_a_loaded_model(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, [])
    body = client_for(server).get("/health").json()

    assert body["status"] == "ok"
    assert body["model"] == "qwen"


@pytest.mark.parametrize(
    "name", ["qwen", ARTIFACT_ID, "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"]
)
def test_the_model_answers_to_its_name_its_artifact_and_its_source(
    tmp_path: Path, name: str
) -> None:
    server, _ = make_server(tmp_path, ["Hi."])
    response = client_for(server).post("/v1/chat/completions", json={**ask(), "model": name})

    assert response.status_code == 200


def test_another_model_name_is_a_404(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])
    response = client_for(server).post("/v1/chat/completions", json={**ask(), "model": "gpt-4o"})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"
    assert runtime.histories == []


# --- completions ----------------------------------------------------------------------


def test_a_completion_hides_reasoning_and_reports_usage(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, [THOUGHT + "The answer is 42."])
    body = client_for(server).post("/v1/chat/completions", json=ask()).json()

    (choice,) = body["choices"]
    assert choice["message"] == {"role": "assistant", "content": "The answer is 42."}
    assert SECRET not in json.dumps(body)
    assert body["usage"] == {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46}
    assert body["object"] == "chat.completion"
    assert runtime.histories == [[("user", "Hello")]]


def test_the_client_owns_the_conversation(tmp_path: Path) -> None:
    """Every request carries the whole history; nothing is remembered between them."""
    server, runtime = make_server(tmp_path, ["One.", "Two."])
    client = client_for(server)
    conversation = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "One."},
        {"role": "user", "content": [{"type": "text", "text": "Again"}]},
    ]

    client.post("/v1/chat/completions", json=ask())
    client.post("/v1/chat/completions", json={"model": "qwen", "messages": conversation})

    assert runtime.histories[1] == [
        ("system", "Be brief."),
        ("user", "Hi"),
        ("assistant", "One."),
        ("user", "Again"),
    ]


def test_request_parameters_override_the_profile_defaults(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])
    client_for(server).post(
        "/v1/chat/completions",
        json=ask(max_completion_tokens=17, temperature=0.0, top_p=0.5, seed=7),
    )

    (config,) = runtime.configs
    assert config.max_new_tokens == 17
    assert config.do_sample is False, "temperature 0 is greedy decoding"
    assert config.top_p == 0.5
    assert config.seed == 7


@pytest.mark.parametrize(
    ("extra", "param"),
    [
        ({"n": 2}, "n"),
        ({"tools": [{"type": "function"}]}, "tools"),
        ({"presence_penalty": 0.5}, "presence_penalty"),
        ({"response_format": {"type": "json_object"}}, "response_format"),
    ],
)
def test_parameters_that_would_change_the_answer_are_refused(
    tmp_path: Path, extra: dict[str, object], param: str
) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])
    response = client_for(server).post("/v1/chat/completions", json={**ask(), **extra})

    assert response.status_code == 400
    assert response.json()["error"]["param"] == param
    assert runtime.histories == []


def test_harmless_unknown_fields_are_ignored(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, ["Hi."])
    response = client_for(server).post(
        "/v1/chat/completions", json=ask(user="someone", stop=None, n=1, presence_penalty=0)
    )

    assert response.status_code == 200


# --- stop sequences -------------------------------------------------------------------


def content_of(response_text: str) -> str:
    return "".join(
        event["choices"][0]["delta"].get("content", "")
        for event in events(response_text)
        if isinstance(event, dict) and event.get("choices")
    )


def final_reason(response_text: str) -> str | None:
    reasons: list[str | None] = [
        event["choices"][0]["finish_reason"]
        for event in events(response_text)
        if isinstance(event, dict) and event.get("choices")
    ]
    return reasons[-1]


@pytest.mark.parametrize("stop", ["Two", ["Four", "Two"]])
def test_a_stop_sequence_ends_the_answer_before_it(tmp_path: Path, stop: object) -> None:
    server, runtime = make_server(tmp_path, ["One. Two. Three."])
    body = client_for(server).post("/v1/chat/completions", json=ask(stop=stop)).json()

    (choice,) = body["choices"]
    assert choice["message"]["content"] == "One."
    assert choice["finish_reason"] == "stop"
    assert isinstance(runtime, StreamingRuntime)
    assert runtime.stopped_early, "streamed internally so generation ends at the match"
    assert runtime.configs[0].stop_strings == [], "client stops are not native stop strings"


def test_a_streamed_answer_stops_at_the_same_place(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["One. Two. Three."])
    response = client_for(server).post("/v1/chat/completions", json=ask(stream=True, stop=["Two"]))

    assert content_of(response.text) == "One."
    assert "Two" not in response.text
    assert final_reason(response.text) == "stop"
    assert isinstance(runtime, StreamingRuntime) and runtime.stopped_early


@pytest.mark.parametrize("chunk", [1, 3, 1000])
def test_overlapping_stops_end_both_answers_where_the_first_one_completes(
    tmp_path: Path, chunk: int
) -> None:
    """``"ion"`` completes before ``"Observation:"`` does, so the answer ends there.

    The fake streams up to the stop and then returns its whole text, as GenAI
    may when its streamer still held tokens back: the text past the streamed
    point must not move the cut of the unstreamed answer.
    """
    text = "Thought: x. Observation: y"
    server, runtime = make_server(tmp_path, [text] * 2, chunk=chunk)
    client = client_for(server)
    stop = ["Observation:", "ion"]

    plain = client.post("/v1/chat/completions", json=ask(stop=stop)).json()
    streamed = client.post("/v1/chat/completions", json=ask(stream=True, stop=stop))

    assert plain["choices"][0]["message"]["content"] == "Thought: x. Observat"
    assert content_of(streamed.text) == "Thought: x. Observat"
    assert plain["choices"][0]["finish_reason"] == final_reason(streamed.text) == "stop"
    assert isinstance(runtime, StreamingRuntime) and runtime.stopped_early


def test_with_show_reasoning_a_stop_can_end_the_shown_reasoning(tmp_path: Path) -> None:
    """The block is part of the visible answer then, so a stop matches inside it."""
    server, _ = make_server(
        tmp_path, ["<think>Plan.\n\nMore plan.</think>\n\nAnswer."], show_reasoning=True
    )
    body = client_for(server).post("/v1/chat/completions", json=ask(stop="\n\n")).json()

    (choice,) = body["choices"]
    assert choice["message"]["content"] == "<think>Plan."
    assert choice["finish_reason"] == "stop"


def test_a_stop_sequence_inside_hidden_reasoning_is_ignored(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, [THOUGHT + "The answer is 42."] * 2)
    client = client_for(server)

    plain = client.post("/v1/chat/completions", json=ask(stop="never")).json()
    streamed = client.post("/v1/chat/completions", json=ask(stream=True, stop="never"))

    assert plain["choices"][0]["message"]["content"] == "The answer is 42."
    assert content_of(streamed.text) == "The answer is 42."
    assert SECRET not in json.dumps(plain) + streamed.text


@pytest.mark.parametrize("stream", [False, True])
def test_a_runtime_that_cannot_stream_is_cut_at_the_stop_too(tmp_path: Path, stream: bool) -> None:
    """It runs to its limit, but the answer still ends at the match, as ``stop``."""
    server, _ = make_server(tmp_path, ["One. Two. Three."], streaming=False)
    response = client_for(server).post(
        "/v1/chat/completions", json=ask(stop="Two", stream=stream, max_tokens=34)
    )

    if stream:
        assert content_of(response.text) == "One."
        assert final_reason(response.text) == "stop"
    else:
        (choice,) = response.json()["choices"]
        assert choice["message"]["content"] == "One."
        assert choice["finish_reason"] == "stop", "34 tokens hit the limit, but the stop came first"


def test_an_unmatched_stop_sequence_changes_nothing(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["One. Two. Three."])
    body = client_for(server).post("/v1/chat/completions", json=ask(stop=["Four"])).json()

    assert body["choices"][0]["message"]["content"] == "One. Two. Three."
    assert isinstance(runtime, StreamingRuntime) and not runtime.stopped_early


@pytest.mark.parametrize(
    "stop",
    [["a", "b", "c", "d", "e"], "", ["ok", ""], "x" * 257, [7]],
)
def test_stop_sequences_outside_the_limits_are_refused(tmp_path: Path, stop: object) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])
    response = client_for(server).post("/v1/chat/completions", json=ask(stop=stop))

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"].startswith("stop")
    assert error["code"] == "invalid_request"
    assert runtime.histories == []


def test_four_stop_sequences_of_the_longest_length_are_accepted(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, ["Hi."])
    stops = [letter * 256 for letter in "abcd"]
    response = client_for(server).post("/v1/chat/completions", json=ask(stop=stops))

    assert response.status_code == 200


def test_content_other_than_text_and_images_is_refused(tmp_path: Path) -> None:
    """Images have their own tests (test_serve_images.py); audio is not accepted at all."""
    server, runtime = make_server(tmp_path, ["Hi."])
    parts: list[dict[str, object]] = [
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}
    ]
    response = client_for(server).post("/v1/chat/completions", json=ask(parts))

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "messages[0].content"
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert runtime.histories == []


def test_out_of_range_values_are_refused_like_chat_refuses_them(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, ["Hi."])
    response = client_for(server).post("/v1/chat/completions", json=ask(temperature=5.0))

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "serve.request.temperature"


@pytest.mark.parametrize(
    ("setting", "value", "param"),
    [
        ("temperature", 2.5, "temperature"),
        ("temperature", -0.1, "temperature"),
        ("top_p", 0, "top_p"),
        ("top_p", 1.5, "top_p"),
        ("top_k", -1, "top_k"),
        ("repetition_penalty", 0, "repetition_penalty"),
        ("max_tokens", 0, "max_tokens"),
        ("max_completion_tokens", 0, "max_completion_tokens"),
    ],
)
def test_a_generation_setting_out_of_range_names_the_parameter(
    tmp_path: Path, setting: str, value: float, param: str
) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])
    for stream in (False, True):
        body = {**ask(stream=stream), setting: value}
        response = client_for(server).post("/v1/chat/completions", json=body)

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["param"] == param
        assert error["type"] == "invalid_request_error"
        assert error["code"] == f"serve.request.{param}"
        assert f"{param} must be" in error["message"]
    assert runtime.histories == []


def test_a_conversation_ending_with_the_assistant_is_refused(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, ["Hi."])
    body = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Yo"}],
    }
    response = client_for(server).post("/v1/chat/completions", json=body)

    assert response.status_code == 400


def test_a_generation_failure_is_a_500_with_its_diagnostic_code(tmp_path: Path) -> None:
    failure = GenerationError(
        diagnostic(
            "runtime.genai.generation_failed",
            FailureCategory.UNKNOWN_RUNTIME_ERROR,
            "Generation failed on GPU",
            "device lost",
        )
    )
    server, _ = make_server(tmp_path, [failure])
    response = client_for(server).post("/v1/chat/completions", json=ask())

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "runtime.genai.generation_failed"


def test_a_server_error_does_not_echo_openvinos_source_locations(tmp_path: Path) -> None:
    raw = (
        "Check 'tensor.get_size() > 0' failed at /home/jenkins/agent/workspace/ie/src/"
        "pipeline.cpp:120:\nThe device returned no logits.\n"
    )
    failure = GenerationError(
        diagnostic(
            "runtime.genai.generation_failed",
            FailureCategory.UNKNOWN_RUNTIME_ERROR,
            "Generation failed on NPU",
            raw,
            raw_excerpt=raw,
        )
    )
    server, _ = make_server(tmp_path, [failure, failure])
    client = client_for(server)

    plain = client.post("/v1/chat/completions", json=ask()).json()["error"]
    streamed = client.post("/v1/chat/completions", json=ask(stream=True))

    assert plain["message"] == "Generation failed on NPU: The device returned no logits."
    assert plain["type"] == "server_error"
    assert streamed.status_code == 500, "it failed before generating, so the status says so"
    assert streamed.json()["error"] == plain


def test_an_error_event_does_not_echo_openvinos_source_locations(tmp_path: Path) -> None:
    raw = "Check 'ok' failed at /home/jenkins/src/pipeline.cpp:120:\nThe device was lost.\n"
    failure = GenerationError(
        diagnostic(
            "runtime.genai.generation_failed",
            FailureCategory.UNKNOWN_RUNTIME_ERROR,
            "Generation failed on NPU",
            raw,
        )
    )
    server, _ = make_server(tmp_path, [], runtime=BreakingRuntime(failure))
    parsed = events(client_for(server).post("/v1/chat/completions", json=ask(stream=True)).text)

    (error,) = [e["error"] for e in parsed if isinstance(e, dict) and "error" in e]
    assert error["message"] == "Generation failed on NPU: The device was lost."


def prompt_too_long() -> GenerationError:
    """What the runtime adapter reports for GenAI's refusal of an over-long NPU prompt."""
    return GenerationError(
        diagnostic(
            PROMPT_TOO_LONG,
            FailureCategory.CONFIGURATION_ERROR,
            "The prompt is too long for this model",
            "It has 6423 tokens, and this model on NPU takes at most 1024.",
            context={"device": "NPU", "prompt_tokens": "6423", "max_prompt_tokens": "1024"},
        )
    )


def test_a_prompt_over_the_npu_length_is_openais_context_length_error(tmp_path: Path) -> None:
    """A 400, so a client does not retry it, under the code clients trim a history on."""
    server, _ = make_server(tmp_path, [prompt_too_long(), "Hi."])
    client = client_for(server)

    response = client.post("/v1/chat/completions", json=ask("word " * 6000))

    assert response.status_code == 400
    error = response.json()["error"]
    assert error == {
        "message": "The prompt is too long for this model: It has 6423 tokens, and this "
        "model on NPU takes at most 1024.",
        "type": "invalid_request_error",
        "param": "messages",
        "code": "context_length_exceeded",
    }
    assert client.post("/v1/chat/completions", json=ask()).status_code == 200


def test_a_streamed_prompt_over_the_npu_length_is_the_same_400(tmp_path: Path) -> None:
    """The pipeline refuses it before generating, so the stream never opens."""
    server, _ = make_server(tmp_path, [prompt_too_long(), prompt_too_long(), "Hi."])
    client = client_for(server)

    plain = client.post("/v1/chat/completions", json=ask())
    streamed = client.post("/v1/chat/completions", json=ask(stream=True))

    assert streamed.status_code == plain.status_code == 400
    assert streamed.headers["content-type"] == "application/json"
    assert streamed.json() == plain.json()
    assert streamed.json()["error"]["code"] == "context_length_exceeded"
    assert client.post("/v1/chat/completions", json=ask(stream=True)).status_code == 200


# --- streaming ------------------------------------------------------------------------


def test_a_streamed_completion_follows_the_openai_event_shape(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, [THOUGHT + "The answer is 42."])
    response = client_for(server).post(
        "/v1/chat/completions", json=ask(stream=True, stream_options={"include_usage": True})
    )

    assert response.headers["content-type"].startswith("text/event-stream")
    assert SECRET not in response.text
    parsed = events(response.text)
    assert parsed[-1] == "[DONE]"
    chunks = [event for event in parsed if isinstance(event, dict)]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    content = "".join(
        chunk["choices"][0]["delta"].get("content", "") for chunk in chunks if chunk["choices"]
    )
    assert content == "The answer is 42."
    finals = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert finals[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["completion_tokens"] == 34


def test_a_runtime_that_cannot_stream_still_answers_a_streaming_request(
    tmp_path: Path,
) -> None:
    server, _ = make_server(tmp_path, [THOUGHT + "Whole."], streaming=False)
    response = client_for(server).post("/v1/chat/completions", json=ask(stream=True))

    parsed = events(response.text)
    content = "".join(
        e["choices"][0]["delta"].get("content", "") for e in parsed if isinstance(e, dict)
    )
    assert content == "Whole."
    assert SECRET not in response.text


def test_a_failure_while_streaming_ends_with_an_error_event(tmp_path: Path) -> None:
    """Once text has gone out the status is sent; the event carries the body a 500 would."""
    failure = GenerationError(
        diagnostic(
            "runtime.genai.generation_failed",
            FailureCategory.UNKNOWN_RUNTIME_ERROR,
            "Generation failed on GPU",
            "device lost",
        )
    )
    server, _ = make_server(tmp_path, [], runtime=BreakingRuntime(failure))
    response = client_for(server).post("/v1/chat/completions", json=ask(stream=True))

    assert response.status_code == 200
    parsed = events(response.text)
    assert content_of(response.text) == "The answer is"
    errors = [e for e in parsed if isinstance(e, dict) and "error" in e]
    assert errors == [
        {
            "error": {
                "message": "Generation failed on GPU: device lost",
                "type": "server_error",
                "param": None,
                "code": "runtime.genai.generation_failed",
            }
        }
    ]
    assert parsed[-1] == "[DONE]"


def test_a_failure_before_the_first_chunk_is_a_status_not_an_event(tmp_path: Path) -> None:
    failure = GenerationError(
        diagnostic(
            "runtime.genai.generation_failed",
            FailureCategory.UNKNOWN_RUNTIME_ERROR,
            "Generation failed on GPU",
            "device lost",
        )
    )
    server, _ = make_server(tmp_path, [failure, failure])
    client = client_for(server)

    plain = client.post("/v1/chat/completions", json=ask())
    streamed = client.post("/v1/chat/completions", json=ask(stream=True))

    assert streamed.status_code == plain.status_code == 500
    assert streamed.json() == plain.json()


def test_a_client_gone_before_generation_starts_stops_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No status is sent while the request waits, yet a client that leaves is noticed."""
    server, runtime = make_server(tmp_path, ["One. Two. Three."], chunk=1)
    waiting, release, finished = threading.Event(), threading.Event(), threading.Event()
    complete = server.complete

    def queued(completion: Any, **kwargs: Any) -> Any:
        waiting.set()
        release.wait(5)  # another request still holds the pipeline
        try:
            return complete(completion, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(server, "complete", queued)
    sent = asyncio.run(leave_while_waiting(create_app(server), waiting))
    release.set()

    assert finished.wait(5)
    assert isinstance(runtime, StreamingRuntime) and runtime.stopped_early, "stopped at once"
    # Nobody reads it; it only ends the exchange, which the middleware insists on.
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 499


async def leave_while_waiting(app: Any, waiting: threading.Event) -> list[dict[str, Any]]:
    """Drive ``app`` over ASGI with a client that disconnects once ``waiting`` is set."""
    body = json.dumps(ask(stream=True)).encode()
    pending = [{"type": "http.request", "body": body, "more_body": False}]
    gone = asyncio.Event()
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        if pending:
            return pending.pop()
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
    }
    served = asyncio.create_task(app(scope, receive, send))
    assert await asyncio.to_thread(waiting.wait, 5)
    assert sent == [], "no status before the pipeline has accepted the prompt"
    gone.set()
    await asyncio.wait_for(served, 5)
    return sent


# --- authentication -------------------------------------------------------------------


def test_an_api_key_is_required_when_configured(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])
    client = client_for(server, api_key="s3cret-key")

    missing = client.post("/v1/chat/completions", json=ask())
    wrong = client.post(
        "/v1/chat/completions", json=ask(), headers={"Authorization": "Bearer nope"}
    )
    right = client.post(
        "/v1/chat/completions", json=ask(), headers={"Authorization": "Bearer s3cret-key"}
    )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    for refused in (missing, wrong):
        error = refused.json()["error"]
        # OpenAI's shape for a bad key, which its client raises AuthenticationError on.
        assert error["type"] == "invalid_request_error"
        assert error["code"] == "invalid_api_key"
    assert "s3cret-key" not in missing.text + wrong.text
    assert right.status_code == 200
    assert len(runtime.histories) == 1
    assert client.get("/health").status_code == 200, "health stays open for supervisors"


# --- request limits -------------------------------------------------------------------


def test_a_declared_oversized_body_is_refused_before_it_is_read(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])
    client = client_for(server, max_body_bytes=1024)

    response = client.post("/v1/chat/completions", json=ask("x" * 2048))

    assert response.status_code == 413
    error = response.json()["error"]
    assert error["code"] == "request_too_large"
    assert error["type"] == "invalid_request_error"
    assert "1024 bytes" in error["message"]
    assert runtime.histories == []


def test_a_chunked_body_is_held_to_the_same_limit(tmp_path: Path) -> None:
    """No ``Content-Length`` to check up front, so the bytes are counted as they arrive."""
    server, runtime = make_server(tmp_path, ["Hi."])
    client = client_for(server, max_body_bytes=1024)
    payload = json.dumps(ask("x" * 4096)).encode()

    def chunks() -> Iterator[bytes]:
        for index in range(0, len(payload), 256):
            yield payload[index : index + 256]

    response = client.post(
        "/v1/chat/completions", content=chunks(), headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
    assert runtime.histories == []


def test_a_body_within_the_limit_is_served(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, ["Hi."])
    response = client_for(server, max_body_bytes=4096).post("/v1/chat/completions", json=ask())
    assert response.status_code == 200


def test_the_default_limit_leaves_room_for_a_long_chat(tmp_path: Path) -> None:
    """Several hundred kilobytes of history is an ordinary long conversation."""
    server, _ = make_server(tmp_path, ["Hi."])
    history = [{"role": "user", "content": "word " * 200}] * 999 + [
        {"role": "user", "content": "Hello"}
    ]
    response = client_for(server).post(
        "/v1/chat/completions", json={"model": "qwen", "messages": history}
    )
    assert response.status_code == 200


def test_too_many_messages_are_refused_naming_the_parameter(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])
    history = [{"role": "user", "content": "x"}] * (MAX_MESSAGES + 1)

    response = client_for(server).post(
        "/v1/chat/completions", json={"model": "qwen", "messages": history}
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "messages"
    assert runtime.histories == []


@pytest.mark.parametrize("content_type", [None, "text/plain"])
def test_a_json_body_without_a_json_content_type_is_refused(
    tmp_path: Path, content_type: str | None
) -> None:
    """A well-formed request is still refused unless it says it is JSON.

    The case without a ``Content-Type`` is what the ``fastapi>=0.132`` floor
    guards: older releases read such a body as JSON, and a browser sends one
    without a CORS preflight, so any web page could have a chat request run by a
    server on localhost. The server answers the validation failure with ``400``
    and the code ``invalid_request``.
    """
    server, runtime = make_server(tmp_path, ["Hi."])
    headers = {} if content_type is None else {"Content-Type": content_type}

    response = client_for(server).post(
        "/v1/chat/completions", content=json.dumps(ask()).encode(), headers=headers
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["type"] == "invalid_request_error"
    assert error["param"] is None, "the body as a whole is refused, no field in it"
    assert "application/json" in error["message"]
    assert runtime.histories == []


def test_a_body_that_is_not_json_names_no_parameter(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])

    response = client_for(server).post(
        "/v1/chat/completions",
        content=b'{"model": "x", "messages": [',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] is None, "never the decode error's offset"
    assert error["code"] == "invalid_request"
    assert error["message"].startswith("The request body is not valid JSON")
    assert "character 28" in error["message"]
    assert runtime.histories == []


@pytest.mark.parametrize(
    ("body", "param"),
    [
        ({"messages": [{"role": "user", "content": "Hi"}]}, "model"),
        ({"model": "qwen", "messages": [{"content": "Hi"}]}, "messages.0.role"),
    ],
)
def test_a_field_that_does_not_validate_is_named(
    tmp_path: Path, body: dict[str, object], param: str
) -> None:
    server, _ = make_server(tmp_path, ["Hi."])
    response = client_for(server).post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    assert response.json()["error"]["param"] == param


def test_authentication_comes_before_the_size_limit(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, ["Hi."])
    client = client_for(server, api_key="s3cret-key", max_body_bytes=1024)

    response = client.post("/v1/chat/completions", json=ask("x" * 2048))

    assert response.status_code == 401
