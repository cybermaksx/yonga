"""Conversation-boundary tests for the stateful VLM runtime path.

``VLMPipeline.generate`` takes no chat history in this build, so the adapter
keeps a hidden ``start_chat()`` session and hands the pipeline only the turns it
has not seen yet. That bookkeeping cannot, on its own, tell a *new* conversation
from a continuation with nothing new in it: on real hardware,
``yonga bench --runs 2`` sent the same prompt fixture twice and the second run
died with "Add a user message to the conversation before generating", because the
new one-message history looked like a prefix of the previous one-message history.

Two defences are covered here, and each is tested with the other one removed:

* the explicit boundary — :meth:`ChatRuntime.begin_conversation`, which the
  benchmark service calls before every run and ``/reset`` calls in chat;
* the adapter's own guard — a history that only repeats what was already sent,
  or whose pending slice carries no user turn while the history does, is treated
  as a new conversation rather than as an empty one.

The genuinely empty case — a history with no user turn anywhere — must keep
raising, because that is a caller bug and not something to paper over.

``openvino_genai`` is stubbed throughout; the scriptable fake module comes from
``test_runtime_genai`` so there is one definition of the fake pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_runtime_genai import FakeGenAI
from yonga.adapters import runtime_genai as runtime_module
from yonga.adapters.runtime_genai import (
    ChatHistory,
    ChatRuntime,
    GenAIChatRuntime,
    GenerationResult,
    open_runtime,
)
from yonga.config.settings import Settings
from yonga.domain.benchmarks import GenerationSettings
from yonga.domain.diagnostics import GenerationError
from yonga.domain.models import PipelineKind

# --- stubbed runtime ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_real_genai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a test forgets to stub the lazy import."""

    def _raise() -> Any:
        raise ImportError("No module named 'openvino_genai'")

    monkeypatch.setattr(runtime_module, "_import_genai", _raise)
    monkeypatch.setattr(runtime_module, "openvino_runtime_version", lambda: "2026.4.0")


@pytest.fixture
def genai(monkeypatch: pytest.MonkeyPatch) -> FakeGenAI:
    module = FakeGenAI()
    monkeypatch.setattr(runtime_module, "_import_genai", lambda: module)
    return module


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        cache_dir=tmp_path / "cache",
        data_dir=tmp_path / "data",
        config_dir=tmp_path / "config",
    )


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    directory = tmp_path / "artifact"
    directory.mkdir()
    (directory / "openvino_model.xml").write_bytes(b"<net/>")
    (directory / "openvino_model.bin").write_bytes(b"\x00")
    return directory


def make_runtime(artifact_dir: Path, settings_obj: Settings, **kwargs: Any) -> GenAIChatRuntime:
    """Open a runtime over the stubbed GenAI module."""
    runtime = open_runtime(artifact_dir, settings=settings_obj, **kwargs)
    assert isinstance(runtime, GenAIChatRuntime)
    return runtime


def _vlm(artifact_dir: Any, settings_obj: Any) -> GenAIChatRuntime:
    """Open a VLM runtime over the stubbed GenAI module."""
    return make_runtime(artifact_dir, settings_obj, pipeline=PipelineKind.VLM)


def _prompts(pipeline: Any) -> list[str]:
    """The positional prompt of every ``generate`` call the pipeline received."""
    return [args[0] for args, _kwargs in pipeline.generate_calls]


def _one_user(text: str) -> ChatHistory:
    """A brand-new history holding exactly one user turn, as a bench run builds."""
    history = ChatHistory()
    history.add_user(text)
    return history


# --- the reproduced defect ------------------------------------------------------------


def test_repeating_the_same_first_message_is_a_new_conversation(artifact, settings, genai):
    """The bench failure: a fresh history equal to the last one must not be empty.

    No ``begin_conversation`` here on purpose — this is the adapter's own guard.
    """
    runtime = _vlm(artifact, settings)
    pipeline = genai.pipelines[0]

    runtime.generate(_one_user("Explain gravity."), GenerationSettings())
    runtime.generate(_one_user("Explain gravity."), GenerationSettings())

    assert _prompts(pipeline) == ["Explain gravity.", "Explain gravity."]
    # The second turn reopened the session rather than continuing the first.
    assert [event for event, _ in pipeline.chat_events] == ["start", "finish", "start"]
    runtime.close()


def test_three_identical_runs_all_answer(artifact, settings, genai):
    """`--runs 2 --warmups 1` against one fixture: every run must produce a prompt."""
    runtime = _vlm(artifact, settings)
    pipeline = genai.pipelines[0]

    for _ in range(3):
        runtime.begin_conversation()
        runtime.generate(_one_user("short instruction"), GenerationSettings())

    assert _prompts(pipeline) == ["short instruction"] * 3
    assert [event for event, _ in pipeline.chat_events] == [
        "start",
        "finish",
        "start",
        "finish",
        "start",
    ]
    runtime.close()


def test_a_pending_slice_without_a_user_turn_restarts_instead_of_raising(artifact, settings, genai):
    """A history whose new turns are not the user's still has a question in it."""
    runtime = _vlm(artifact, settings)
    pipeline = genai.pipelines[0]

    history = ChatHistory()
    history.add_user("First")
    runtime.generate(history, GenerationSettings())

    # Only an assistant turn was added: the pending slice carries no user text,
    # but the conversation plainly does.
    history.add_assistant("An answer.")
    runtime.generate(history, GenerationSettings())

    assert _prompts(pipeline) == ["First", "First"]
    assert [event for event, _ in pipeline.chat_events] == ["start", "finish", "start"]
    runtime.close()


def test_a_history_with_no_user_turn_still_raises(artifact, settings, genai):
    """The genuine empty case stays a typed error; it is a caller bug."""
    runtime = _vlm(artifact, settings)
    history = ChatHistory()
    history.add_system("You are terse.")

    with pytest.raises(GenerationError) as excinfo:
        runtime.generate(history, GenerationSettings())
    assert excinfo.value.diagnostic.code == "runtime.generation.empty_prompt"
    runtime.close()


def test_a_failed_empty_turn_does_not_poison_the_next_one(artifact, settings, genai):
    """Nothing was sent, so nothing may be recorded as sent."""
    runtime = _vlm(artifact, settings)
    pipeline = genai.pipelines[0]
    history = ChatHistory()
    history.add_system("You are terse.")

    with pytest.raises(GenerationError):
        runtime.generate(history, GenerationSettings())

    history.add_user("Now a question.")
    runtime.generate(history, GenerationSettings())
    assert _prompts(pipeline) == ["Now a question."]
    runtime.close()


# --- a real continuation is still incremental -----------------------------------------


def test_a_genuine_continuation_sends_only_the_new_turn(artifact, settings, genai):
    runtime = _vlm(artifact, settings)
    pipeline = genai.pipelines[0]

    history = ChatHistory()
    history.add_system("You are terse.")
    history.add_user("First")
    runtime.generate(history, GenerationSettings())

    history.add_assistant("Hi.")
    history.add_user("Second")
    runtime.generate(history, GenerationSettings())

    assert _prompts(pipeline) == ["First", "Second"]
    assert pipeline.chat_events == [("start", "You are terse.")]
    runtime.close()


def test_a_diverging_history_restarts_and_resends_everything(artifact, settings, genai):
    runtime = _vlm(artifact, settings)
    pipeline = genai.pipelines[0]

    history = ChatHistory()
    history.add_user("First")
    runtime.generate(history, GenerationSettings())

    rewritten = ChatHistory()
    rewritten.add_user("Something else")
    rewritten.add_assistant("Answered elsewhere.")
    rewritten.add_user("And now this")
    runtime.generate(rewritten, GenerationSettings())

    assert _prompts(pipeline) == ["First", "Something else\n\nAnd now this"]
    runtime.close()


# --- the explicit boundary ------------------------------------------------------------


def test_begin_conversation_discards_the_hidden_session(artifact, settings, genai):
    runtime = _vlm(artifact, settings)
    pipeline = genai.pipelines[0]

    history = ChatHistory()
    history.add_system("You are terse.")
    history.add_user("First")
    runtime.generate(history, GenerationSettings())

    runtime.begin_conversation()
    assert pipeline.chat_events[-1] == ("finish", None)

    # The continuation the caller disowned is sent in full, from a new session.
    history.add_assistant("Hi.")
    history.add_user("Second")
    runtime.generate(history, GenerationSettings())
    assert _prompts(pipeline) == ["First", "First\n\nSecond"]
    assert [event for event, _ in pipeline.chat_events] == ["start", "finish", "start"]
    runtime.close()


def test_begin_conversation_is_idempotent_and_survives_close(artifact, settings, genai):
    runtime = _vlm(artifact, settings)
    pipeline = genai.pipelines[0]

    runtime.begin_conversation()
    runtime.begin_conversation()
    assert pipeline.chat_events == []

    runtime.generate(_one_user("Hello"), GenerationSettings())
    runtime.close()
    runtime.begin_conversation()  # must not raise, must not touch a dead pipeline
    assert [event for event, _ in pipeline.chat_events] == ["start", "finish"]


def test_begin_conversation_is_a_no_op_on_the_llm_path(artifact, settings, genai):
    """The stateless path re-supplies its whole history, so there is nothing to discard."""
    runtime = make_runtime(artifact, settings, pipeline=PipelineKind.LLM)
    pipeline = genai.pipelines[0]

    runtime.begin_conversation()
    runtime.generate(_one_user("Hello"), GenerationSettings())
    runtime.begin_conversation()
    runtime.generate(_one_user("Hello"), GenerationSettings())

    assert pipeline.chat_events == []
    sent = [args[0].messages for args, _kwargs in pipeline.generate_calls]
    assert sent == [[{"role": "user", "content": "Hello"}]] * 2
    runtime.close()


def test_the_protocol_carries_the_boundary_with_a_stateless_default():
    """A stateless runtime may inherit the no-op; the domain type still declares it."""

    class Stateless(ChatRuntime):
        def generate(
            self, history: ChatHistory, config: GenerationSettings
        ) -> GenerationResult:  # pragma: no cover - never called
            raise AssertionError("not used")

        def close(self) -> None:
            return None

    runtime = Stateless()
    runtime.begin_conversation()  # inherited no-op; nothing to discard
    assert isinstance(runtime, ChatRuntime)
