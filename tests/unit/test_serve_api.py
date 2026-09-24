"""``yonga serve``: the OpenAI-compatible API over the same session ``chat`` uses.

The server is driven through FastAPI's test client against fake runtimes and
the real built-in registry, so what is asserted about reasoning, defaults and
model naming is what the shipped chat profile produces.
"""

from __future__ import annotations

import json
from pathlib import Path

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
from yonga.application.serve import ModelServer
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


def make_server(
    tmp_path: Path, answers: list[str | BaseException], *, streaming: bool = True
) -> tuple[ModelServer, FakeRuntime]:
    service, factory, _ = build_service(tmp_path)
    runtime = StreamingRuntime(answers) if streaming else FakeRuntime(answers)
    factory.runtime = runtime
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact)
    server = ModelServer(session, name="qwen")
    server.load()
    return server, runtime


def client_for(server: ModelServer, **kwargs: str) -> TestClient:
    return TestClient(create_app(server, **kwargs))


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
        ({"stop": ["\n"]}, "stop"),
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


def test_image_content_is_refused(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, ["Hi."])
    parts: list[dict[str, object]] = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    ]
    response = client_for(server).post("/v1/chat/completions", json=ask(parts))

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "messages[0].content"


def test_out_of_range_values_are_refused_like_chat_refuses_them(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, ["Hi."])
    response = client_for(server).post("/v1/chat/completions", json=ask(temperature=5.0))

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "serve.request.temperature"


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
    failure = GenerationError(
        diagnostic(
            "runtime.genai.generation_failed",
            FailureCategory.UNKNOWN_RUNTIME_ERROR,
            "Generation failed on GPU",
            "device lost",
        )
    )
    server, _ = make_server(tmp_path, [failure])
    response = client_for(server).post("/v1/chat/completions", json=ask(stream=True))

    parsed = events(response.text)
    errors = [e for e in parsed if isinstance(e, dict) and "error" in e]
    assert errors[0]["error"]["code"] == "runtime.genai.generation_failed"
    assert parsed[-1] == "[DONE]"


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
    assert "s3cret-key" not in missing.text + wrong.text
    assert right.status_code == 200
    assert len(runtime.histories) == 1
    assert client.get("/health").status_code == 200, "health stays open for supervisors"
