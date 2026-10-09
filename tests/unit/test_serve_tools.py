"""``yonga serve`` with OpenAI ``tools``: gated by the registry, read back as calls.

A Qwen3 artifact with a chat template that asks for ``<tool_call>`` blocks is
served through FastAPI's test client, against fake runtimes and the real
built-in registry, so the format being parsed is the one ``chat-qwen3``
declares. Requests are refused with a 400 naming the parameter wherever yonga
cannot honour them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from tests.unit.test_chat_service import SECRET, build_service, make_manifest, write_artifact
from tests.unit.test_serve_api import StreamingRuntime, content_of, events, final_reason
from yonga.adapters.runtime_genai import ChatHistory, GenerationResult
from yonga.api.app import create_app
from yonga.api.openai import ChatCompletionRequest, UnsupportedParameter
from yonga.application.chat import ChatMessage
from yonga.application.serve import (
    IMAGES_EARLIER_TURN,
    MAX_TOOLS,
    TOOLS_UNSUPPORTED,
    CompletionRequest,
    ImagePolicy,
    ModelServer,
    ToolPolicy,
)
from yonga.domain.benchmarks import GenerationSettings
from yonga.domain.diagnostics import ConfigurationError

QWEN3_TEMPLATE = (
    "{%- if tools %}For each function call, return a json object within "
    "<tool_call></tool_call> XML tags{%- endif %}"
)
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
TIME = {"type": "function", "function": {"name": "get_time"}}
CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Ankara"}}\n</tool_call>'
SECOND = '<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>'


class RecordingRuntime(StreamingRuntime):
    """Also records each history's tools and full message payloads."""

    def __init__(self, answers: list[str | BaseException], chunk: int = 3) -> None:
        super().__init__(answers, chunk)
        self.tools: list[list[dict[str, Any]]] = []
        self.payloads: list[list[dict[str, object]]] = []

    def generate(self, history: ChatHistory, config: GenerationSettings) -> GenerationResult:
        self.tools.append(list(history.tools))
        self.payloads.append(history.to_payload())
        return super().generate(history, config)


def make_server(
    tmp_path: Path,
    answers: list[str | BaseException],
    *,
    model_type: str = "qwen3",
    template: str | None = QWEN3_TEMPLATE,
    chunk: int = 3,
) -> tuple[ModelServer, RecordingRuntime]:
    service, factory, _ = build_service(tmp_path)
    runtime = RecordingRuntime(answers, chunk)
    factory.runtime = runtime
    manifest = make_manifest(
        model_type=model_type,
        architecture="Qwen3ForCausalLM" if model_type == "qwen3" else "Qwen3_5ForCausalLM",
    )
    artifact = write_artifact(tmp_path, manifest)
    if template is not None:
        (artifact / "chat_template.jinja").write_text(template, encoding="utf-8")
    session = service.open_artifact(artifact)
    server = ModelServer(session, name="qwen")
    server.load()
    return server, runtime


def ask(
    content: str = "Weather in Ankara?", *, tools: list[object] | None = None, **extra: object
) -> dict[str, object]:
    return {
        "model": "qwen",
        "messages": [{"role": "user", "content": content}],
        "tools": [WEATHER, TIME] if tools is None else tools,
        **extra,
    }


def post(server: ModelServer, body: dict[str, object]) -> Any:
    return TestClient(create_app(server)).post("/v1/chat/completions", json=body)


def tool_deltas(response_text: str) -> list[dict[str, object]]:
    return [
        call
        for event in events(response_text)
        if isinstance(event, dict) and event.get("choices")
        for call in event["choices"][0]["delta"].get("tool_calls", [])
    ]


# --- answers read as calls --------------------------------------------------------------


def test_a_call_comes_back_as_tool_calls(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, [CALL])
    body = post(server, ask()).json()

    (choice,) = body["choices"]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["role"] == "assistant"
    assert message["content"] is None
    (call,) = message["tool_calls"]
    assert call["id"].startswith("call_")
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Ankara"}
    assert runtime.tools == [[WEATHER, TIME]], "the tools reach the chat template"


def test_text_before_a_call_is_its_content(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, ["Let me check.\n" + CALL])
    (choice,) = post(server, ask()).json()["choices"]

    assert choice["message"]["content"] == "Let me check."
    assert len(choice["message"]["tool_calls"]) == 1


def test_a_call_drafted_in_hidden_reasoning_is_not_read_or_shown(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, [f"<think>{SECRET} {CALL}</think>\n\nIt is sunny."])
    body = post(server, ask()).json()

    (choice,) = body["choices"]
    assert choice["message"] == {"role": "assistant", "content": "It is sunny."}
    assert choice["finish_reason"] == "stop"
    assert SECRET not in json.dumps(body)


def test_several_calls_and_parallel_tool_calls_false(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, [CALL + SECOND, CALL + SECOND])

    both = post(server, ask()).json()["choices"][0]["message"]["tool_calls"]
    first = post(server, ask(parallel_tool_calls=False)).json()["choices"][0]["message"]

    assert [call["function"]["name"] for call in both] == ["get_weather", "get_time"]
    assert len({call["id"] for call in both}) == 2
    assert [call["function"]["name"] for call in first["tool_calls"]] == ["get_weather"]


@pytest.mark.parametrize(
    "answer",
    [
        "<tool_call>{not json}</tool_call>",
        '<tool_call>{"name": "format_disk", "arguments": {}}</tool_call>',
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Ankara"}}',
    ],
)
def test_an_unreadable_call_is_returned_as_text_and_logged_without_it(
    tmp_path: Path, answer: str, caplog: pytest.LogCaptureFixture
) -> None:
    server, _ = make_server(tmp_path, [answer])
    with caplog.at_level(logging.WARNING, logger="yonga.application.serve"):
        (choice,) = post(server, ask()).json()["choices"]

    assert choice["message"] == {"role": "assistant", "content": answer}
    assert choice["finish_reason"] == "stop"
    (record,) = caplog.records
    assert "could not be read" in record.getMessage()
    assert "get_weather" not in record.getMessage() and "{" not in record.getMessage()


def test_a_plain_answer_logs_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    server, _ = make_server(tmp_path, ["It is sunny."])
    with caplog.at_level(logging.WARNING, logger="yonga.application.serve"):
        (choice,) = post(server, ask()).json()["choices"]

    assert choice["message"]["content"] == "It is sunny."
    assert caplog.records == []


def test_tool_choice_none_keeps_the_tools_from_the_model(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, [CALL])
    (choice,) = post(server, ask(tool_choice="none")).json()["choices"]

    assert runtime.tools == [[]]
    assert choice["message"] == {"role": "assistant", "content": CALL}
    assert choice["finish_reason"] == "stop"


def test_tool_choice_auto_is_the_default(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, [CALL])
    (choice,) = post(server, ask(tool_choice="auto")).json()["choices"]
    assert choice["finish_reason"] == "tool_calls"


# --- streaming --------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [1, 3, 1000])
def test_a_streamed_call_arrives_as_tool_call_deltas(tmp_path: Path, chunk: int) -> None:
    server, _ = make_server(tmp_path, ["Let me check.\n" + CALL + SECOND], chunk=chunk)
    response = post(server, ask(stream=True))

    assert content_of(response.text) == "Let me check."
    assert "<tool_call" not in response.text and "<tool" not in content_of(response.text)
    deltas = tool_deltas(response.text)
    assert [delta["index"] for delta in deltas] == [0, 1]
    assert [delta["function"]["name"] for delta in deltas] == ["get_weather", "get_time"]
    assert json.loads(deltas[0]["function"]["arguments"]) == {"city": "Ankara"}
    assert all(delta["id"].startswith("call_") and delta["type"] == "function" for delta in deltas)
    assert final_reason(response.text) == "tool_calls"
    assert events(response.text)[-1] == "[DONE]"


@pytest.mark.parametrize("chunk", [1, 4, 1000])
def test_a_streamed_unreadable_call_is_released_as_text(tmp_path: Path, chunk: int) -> None:
    answer = "Sure. <tool_call>{oops</tool_call>"
    server, _ = make_server(tmp_path, [answer], chunk=chunk)
    response = post(server, ask(stream=True))

    assert content_of(response.text) == answer
    assert tool_deltas(response.text) == []
    assert final_reason(response.text) == "stop"


@pytest.mark.parametrize("stream", [False, True])
def test_a_stop_before_the_call_ends_the_answer_without_it(tmp_path: Path, stream: bool) -> None:
    server, _ = make_server(tmp_path, ["Let me check the weather.\n" + CALL])
    response = post(server, ask(stop="the weather", stream=stream))

    if stream:
        assert content_of(response.text) == "Let me check"
        assert tool_deltas(response.text) == []
        assert final_reason(response.text) == "stop"
    else:
        (choice,) = response.json()["choices"]
        assert choice["message"] == {"role": "assistant", "content": "Let me check"}
        assert choice["finish_reason"] == "stop"


def test_a_stop_sequence_does_not_match_inside_a_call(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, [CALL] * 2)
    plain = post(server, ask(stop="Ankara")).json()["choices"][0]
    streamed = post(server, ask(stop="Ankara", stream=True))

    assert plain["finish_reason"] == final_reason(streamed.text) == "tool_calls"
    assert len(plain["message"]["tool_calls"]) == len(tool_deltas(streamed.text)) == 1


# --- a conversation that carries calls and results --------------------------------------


def round_trip(**extra: object) -> dict[str, object]:
    return {
        "model": "qwen",
        "tools": [WEATHER],
        "messages": [
            {"role": "user", "content": "Weather in Ankara?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Ankara"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": '{"celsius": 21}'},
        ],
        **extra,
    }


def test_tool_calls_and_results_reach_the_chat_template(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["It is 21 C in Ankara."])
    (choice,) = post(server, round_trip()).json()["choices"]

    assert choice["message"]["content"] == "It is 21 C in Ankara."
    (payload,) = runtime.payloads
    assert payload[1]["tool_calls"] == [
        {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "get_weather", "arguments": {"city": "Ankara"}},
        }
    ]
    assert payload[2] == {
        "role": "tool",
        "content": '{"celsius": 21}',
        "tool_call_id": "call_abc",
        "name": "get_weather",
    }


def test_several_results_may_answer_one_assistant_message(tmp_path: Path) -> None:
    body = round_trip()
    messages: list[dict[str, Any]] = body["messages"]  # type: ignore[assignment]
    messages[1]["tool_calls"].append(
        {"id": "call_def", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}
    )
    messages.append({"role": "tool", "tool_call_id": "call_def", "content": "12:00"})
    body["tools"] = [WEATHER, TIME]
    server, runtime = make_server(tmp_path, ["Done."])

    assert post(server, body).status_code == 200
    assert [message.get("name") for message in runtime.payloads[0][2:]] == [
        "get_weather",
        "get_time",
    ]


# --- refusals ---------------------------------------------------------------------------


def refused(server: ModelServer, body: dict[str, object]) -> dict[str, Any]:
    response = post(server, body)
    assert response.status_code == 400, response.text
    error: dict[str, Any] = response.json()["error"]
    return error


def test_a_model_without_a_tool_call_format_refuses_tools(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Hi."], model_type="qwen3_5")
    error = refused(server, ask())

    assert error["param"] == "tools"
    assert error["code"] == TOOLS_UNSUPPORTED
    assert "chat-qwen3_5" in error["message"]
    assert runtime.histories == []


def test_an_artifact_whose_template_asks_for_another_format_refuses_tools(tmp_path: Path) -> None:
    server, runtime = make_server(tmp_path, ["Hi."], template="{{ messages }}")
    error = refused(server, ask())

    assert error["param"] == "tools"
    assert "'<tool_call>'" in error["message"] and "template" in error["message"]
    assert runtime.histories == []


@pytest.mark.parametrize(
    ("extra", "param"),
    [
        ({"tool_choice": "required"}, "tool_choice"),
        ({"tool_choice": {"type": "function", "function": {"name": "get_weather"}}}, "tool_choice"),
        ({"tool_choice": "sometimes"}, "tool_choice"),
        ({"functions": [{"name": "get_weather"}]}, "functions"),
        ({"function_call": "auto"}, "function_call"),
        ({"tools": [{"type": "retrieval"}]}, "tools[0]"),
        ({"tools": [{"type": "function"}]}, "tools[0].function"),
        ({"tools": [{"type": "function", "function": {"name": "a b"}}]}, "tools[0].function.name"),
        (
            {"tools": [{"type": "function", "function": {"name": "f", "parameters": []}}]},
            "tools[0].function.parameters",
        ),
        (
            {
                "tools": [
                    {"type": "function", "function": {"name": "f", "parameters": {"type": "array"}}}
                ]
            },
            "tools[0].function.parameters",
        ),
        ({"tools": [WEATHER, WEATHER]}, "tools[1].function.name"),
        (
            {"tools": [{"type": "function", "function": {"name": f"f{i}"}} for i in range(129)]},
            "tools",
        ),
    ],
)
def test_requests_yonga_cannot_honour_are_refused(
    tmp_path: Path, extra: dict[str, object], param: str
) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])
    error = refused(server, {**ask(), **extra})

    assert error["param"] == param
    assert runtime.histories == []
    assert MAX_TOOLS == 128


def without_tools(body: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in body.items() if key != "tools"}


@pytest.mark.parametrize(
    ("change", "param"),
    [
        (lambda body: without_tools(body), "messages[1].tool_calls"),
        (lambda body: {**body, "messages": body["messages"][:1] + body["messages"][2:]}, None),
        (
            lambda body: {
                **body,
                "messages": [*body["messages"][:2], {"role": "tool", "content": "21"}],
            },
            "messages[2].tool_call_id",
        ),
        (
            lambda body: {
                **body,
                "messages": [
                    *body["messages"][:2],
                    {"role": "tool", "tool_call_id": "call_other", "content": "21"},
                ],
            },
            "messages[2].tool_call_id",
        ),
        (
            lambda body: {
                **body,
                "messages": [
                    {
                        "role": "user",
                        "content": "x",
                        "tool_calls": body["messages"][1]["tool_calls"],
                    }
                ],
            },
            "messages[0].tool_calls",
        ),
        (
            lambda body: {
                **body,
                "messages": [
                    *body["messages"][:1],
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]},
                    *body["messages"][2:],
                ],
            },
            "messages[1].tool_calls[0]",
        ),
        (
            lambda body: {
                **body,
                "messages": [
                    {"role": "assistant", "content": "", "function_call": {"name": "f"}},
                    {"role": "user", "content": "x"},
                ],
            },
            "messages[0].function_call",
        ),
    ],
)
def test_tool_turns_that_do_not_fit_together_are_refused(
    tmp_path: Path, change: Any, param: str | None
) -> None:
    server, runtime = make_server(tmp_path, ["Hi."])
    error = refused(server, change(round_trip()))

    if param is None:
        # The tool message no longer follows the call it answers.
        assert error["param"] == "messages[1].tool_call_id"
    else:
        assert error["param"] == param
    assert runtime.histories == []


def test_the_last_message_may_not_be_the_assistant_s(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, ["Hi."])
    body = round_trip()
    body["messages"] = [*body["messages"], {"role": "assistant", "content": "Done."}]  # type: ignore[misc]

    assert refused(server, body)["param"] == "messages"


def test_the_server_holds_the_line_for_callers_other_than_the_api(tmp_path: Path) -> None:
    server, _ = make_server(tmp_path, ["Hi."], model_type="qwen3_5")
    request = CompletionRequest(
        messages=[ChatMessage(role="user", content="Weather?")], tools=[WEATHER]
    )

    with pytest.raises(ConfigurationError) as caught:
        server.complete(request)
    assert caught.value.diagnostic.code == TOOLS_UNSUPPORTED


def test_carried_images_need_a_user_message_to_land_on() -> None:
    """With tools, a conversation may end on a tool result, which takes no image."""
    body = round_trip()
    messages: list[dict[str, Any]] = body["messages"]  # type: ignore[assignment]
    messages[0]["content"] = [
        {"type": "text", "text": "Weather here?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    request = ChatCompletionRequest.model_validate(body)

    with pytest.raises(UnsupportedParameter) as caught:
        request.to_completion(
            ImagePolicy(accepted=True, carry_forward=True), tools=ToolPolicy(accepted=True)
        )
    assert caught.value.code == IMAGES_EARLIER_TURN
    assert caught.value.param == "messages[0].content[1]"
