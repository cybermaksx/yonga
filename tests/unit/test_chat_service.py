"""The chat service: generation policy, session lifetime and command execution.

Everything here runs against fake runtimes and the **real** built-in registry, so
the assertions about generation defaults are assertions about the shipped
``chat.yaml`` rather than about a fixture invented for the test.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from yonga.adapters.runtime_genai import (
    ChatHistory,
    ChatProfileLike,
    ChatRuntime,
    GenerationResult,
    open_runtime,
)
from yonga.application.chat import (
    ArtifactStoreLike,
    ChatAction,
    ChatOptions,
    ChatProfileView,
    ChatService,
    ChatSession,
    ParsedCommand,
    RuntimeFactory,
)
from yonga.application.progress import CollectingSink, ProgressKind
from yonga.config.settings import Settings
from yonga.domain.artifacts import (
    ArtifactManifest,
    ExportRecord,
    ManifestModelRef,
    RecommendedRuntime,
    manifest_path,
)
from yonga.domain.benchmarks import GenerationSettings
from yonga.domain.diagnostics import FailureCategory as Category
from yonga.domain.diagnostics import GenerationError, Severity, diagnostic
from yonga.domain.models import PipelineKind
from yonga.registry.loader import load_builtin_registry
from yonga.registry.resolver import RegistryResolver
from yonga.ui.chat_tui import ChatSessionInfo, ChatView, parse_command
from yonga.ui.console import YongaConsole

SECRET = "the-user-must-never-see-this"
QWEN_REPO = "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"
QWEN_SOURCE = "nDimensional/Qwen3.5-9B-Uncensored-Safetensors"

#: The real adapter entry point must satisfy the port the service depends on.
#: mypy checks this assignment; that is the actual test.
_FACTORY_CONTRACT: RuntimeFactory = open_runtime


# --- fakes ----------------------------------------------------------------------------


class FakeRuntime:
    """A chat runtime that replays canned answers and records what it was asked."""

    def __init__(self, answers: list[str | BaseException]) -> None:
        self.answers = list(answers)
        self.histories: list[list[tuple[str, str]]] = []
        self.configs: list[GenerationSettings] = []
        self.closed = False

    def generate(self, history: ChatHistory, config: GenerationSettings) -> GenerationResult:
        self.histories.append([(m.role, m.content) for m in history.messages])
        self.configs.append(config.model_copy(deep=True))
        answer = self.answers.pop(0) if self.answers else ""
        if isinstance(answer, BaseException):
            raise answer
        return GenerationResult(
            text=answer,
            display_text=answer,
            reasoning_text=None,
            input_tokens=12,
            output_tokens=34,
            ttft_ms=180.0,
            throughput_tps=24.6,
            tpot_ms=40.6,
            wall_seconds=1.5,
        )

    def close(self) -> None:
        self.closed = True


class FakeFactory:
    """Counts how many pipelines a session builds; one per session is the contract."""

    def __init__(self, runtime: FakeRuntime) -> None:
        self.runtime = runtime
        self.calls = 0
        self.kwargs: list[dict[str, Any]] = []

    def __call__(
        self,
        artifact_dir: Path,
        *,
        device: str,
        pipeline: PipelineKind,
        properties: Any,
        chat_profile: ChatProfileLike | None,
        settings: Settings | None,
        thinking: bool | None,
    ) -> ChatRuntime:
        self.calls += 1
        self.kwargs.append(
            {
                "artifact_dir": artifact_dir,
                "device": device,
                "pipeline": pipeline,
                "properties": dict(properties or {}),
                "chat_profile": chat_profile,
                "settings": settings,
                "thinking": thinking,
            }
        )
        return self.runtime


class FakeStore:
    """Maps a user-facing reference onto a prepared artifact directory."""

    def __init__(self, mapping: dict[str, Path]) -> None:
        self.mapping = mapping
        self.asked: list[str] = []

    def resolve(self, reference: str) -> Path:
        self.asked.append(reference)
        return self.mapping[reference]


@dataclass(frozen=True)
class FakeCommand:
    """A parsed command built by hand, for arguments the real parser would reject."""

    kind: str
    error: str | None = None
    int_value: int | None = None
    float_value: float | None = None
    bool_value: bool | None = None


# --- fixtures -------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Settings whose XDG directories all live under ``tmp_path``."""
    payload: dict[str, Any] = {
        "config_dir": tmp_path / "config",
        "cache_dir": tmp_path / "cache",
        "data_dir": tmp_path / "data",
    }
    payload.update(overrides)
    return Settings(**payload)


def make_manifest(**overrides: Any) -> ArtifactManifest:
    """A Qwen3.5 artifact manifest matching the built-in registry's chat profile."""
    payload: dict[str, Any] = {
        "artifact_id": "qwen35-9b--0123456789ab--gpu-int4-asym-g128",
        "created_at": datetime(2026, 4, 1, 12, 0, tzinfo=UTC),
        "tool_version": "0.1.0",
        "requested_model": ManifestModelRef(repo_id=QWEN_REPO, resolved_revision="a" * 40),
        "resolved_source": ManifestModelRef(repo_id=QWEN_SOURCE, resolved_revision="b" * 40),
        "architecture": "Qwen3_5ForConditionalGeneration",
        "model_type": "qwen3_5",
        "task": "text-generation",
        "pipeline": PipelineKind.LLM,
        "export": ExportRecord(profile="gpu-int4-asym-g128"),
        "recommended_runtime": RecommendedRuntime(
            device="GPU",
            pipeline=PipelineKind.LLM,
            properties={"ATTENTION_BACKEND": "SDPA"},
        ),
    }
    payload.update(overrides)
    return ArtifactManifest(**payload)


def write_artifact(tmp_path: Path, manifest: ArtifactManifest | None = None) -> Path:
    """Create an artifact directory, with a manifest unless one is explicitly withheld."""
    artifact = tmp_path / "artifacts" / "qwen35"
    artifact.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        target = manifest_path(artifact)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(manifest.to_json_text(), encoding="utf-8")
    return artifact


def build_service(
    tmp_path: Path,
    *,
    answers: list[str | BaseException] | None = None,
    settings: Settings | None = None,
    progress: CollectingSink | None = None,
) -> tuple[ChatService, FakeFactory, FakeRuntime]:
    """A service wired to fakes but to the real registry resolver."""
    runtime = FakeRuntime(answers if answers is not None else ["an answer"])
    factory = FakeFactory(runtime)
    store = FakeStore({})
    resolver = RegistryResolver(load_builtin_registry())
    service = ChatService(
        factory,
        store,
        resolver,
        settings if settings is not None else make_settings(tmp_path),
        progress=progress,
    )
    return service, factory, runtime


@pytest.fixture
def session_parts(tmp_path):
    service, factory, runtime = build_service(tmp_path, answers=["an answer"])
    artifact = write_artifact(tmp_path, make_manifest())
    return service, factory, runtime, artifact


# --- generation defaults come from the registry chat profile --------------------------


def test_defaults_come_from_the_registry_chat_profile(session_parts):
    service, _factory, _runtime, artifact = session_parts
    session = service.open_artifact(artifact)
    settings = session.generation_settings
    assert session.profile.id == "chat-qwen3_5"
    assert settings.temperature == pytest.approx(0.7)
    assert settings.top_p == pytest.approx(0.8)
    assert settings.top_k == 20
    assert settings.max_new_tokens == 4096
    assert settings.do_sample is True


def test_thinking_defaults_differ_from_non_thinking_defaults(session_parts):
    service, _factory, _runtime, artifact = session_parts
    standard = service.open_artifact(artifact, options=ChatOptions(thinking=False))
    thinking = service.open_artifact(artifact, options=ChatOptions(thinking=True))
    assert standard.generation_settings.temperature == pytest.approx(0.7)
    assert standard.generation_settings.top_p == pytest.approx(0.8)
    assert thinking.generation_settings.temperature == pytest.approx(0.6)
    assert thinking.generation_settings.top_p == pytest.approx(0.95)
    assert thinking.generation_settings.top_k == 20
    assert thinking.generation_settings.max_new_tokens == 4096
    assert standard.thinking is False
    assert thinking.thinking is True


def test_an_unknown_model_falls_back_to_the_conservative_chat_profile(tmp_path):
    service, _factory, _runtime = build_service(tmp_path)
    manifest = make_manifest(
        model_type="something-nobody-has-ever-heard-of",
        architecture="MysteryForCausalLM",
        recommended_runtime=None,
    )
    artifact = write_artifact(tmp_path, manifest)
    session = service.open_artifact(artifact)
    assert session.profile.id == "chat-default"
    assert session.generation_settings.temperature is None
    assert session.generation_settings.do_sample is False
    assert session.reasoning_wrappers == [("<think>", "</think>")]


def test_a_missing_manifest_still_opens_a_usable_session(tmp_path):
    service, _factory, _runtime = build_service(tmp_path)
    artifact = write_artifact(tmp_path, None)
    session = service.open_artifact(artifact)
    assert session.manifest is None
    assert session.device == "CPU"
    assert session.pipeline is PipelineKind.LLM
    assert session.generation_settings.max_new_tokens == service.settings.max_new_tokens


def test_command_line_overrides_win_over_the_profile(session_parts):
    service, _factory, _runtime, artifact = session_parts
    session = service.open_artifact(
        artifact,
        options=ChatOptions(max_new_tokens=128, temperature=0.2, top_p=0.5, top_k=7),
    )
    settings = session.generation_settings
    assert settings.max_new_tokens == 128
    assert settings.temperature == pytest.approx(0.2)
    assert settings.top_p == pytest.approx(0.5)
    assert settings.top_k == 7


def test_the_profile_is_handed_to_the_runtime_factory(session_parts):
    service, factory, _runtime, artifact = session_parts
    session = service.open_artifact(artifact)
    session.load()
    passed = factory.kwargs[0]
    assert passed["device"] == "GPU"
    assert passed["pipeline"] is PipelineKind.LLM
    assert passed["properties"] == {"ATTENTION_BACKEND": "SDPA"}
    assert passed["thinking"] is False
    profile = passed["chat_profile"]
    assert isinstance(profile, ChatProfileView)
    assert profile.supports_thinking_flag is True
    assert profile.thinking_template_variable == "enable_thinking"


def test_runtime_properties_are_not_applied_to_a_different_device(session_parts):
    service, factory, _runtime, artifact = session_parts
    session = service.open_artifact(artifact, options=ChatOptions(device="cpu"))
    session.load()
    assert factory.kwargs[0]["device"] == "CPU"
    assert factory.kwargs[0]["properties"] == {}


# --- the pipeline is built once -------------------------------------------------------


def test_the_runtime_is_constructed_once_across_several_turns(tmp_path):
    service, factory, _runtime = build_service(tmp_path, answers=["one", "two", "three"])
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact)
    assert factory.calls == 0

    for _ in range(3):
        assert session.send("hello").ok

    assert factory.calls == 1
    assert session.runtime_loads == 1


def test_reset_does_not_reload_the_pipeline(tmp_path):
    service, factory, _runtime = build_service(tmp_path, answers=["one", "two"])
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact)
    session.send("hello")
    session.reset()
    session.send("hello again")
    assert factory.calls == 1


def test_closing_a_session_closes_the_runtime(tmp_path):
    service, _factory, runtime = build_service(tmp_path)
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact)
    session.load()
    session.close()
    session.close()
    assert session.closed is True
    assert runtime.closed is True


# --- sanitised answers ----------------------------------------------------------------


def test_a_complete_think_block_never_reaches_the_display(tmp_path):
    service, _factory, _runtime = build_service(
        tmp_path, answers=[f"<think>{SECRET}</think>The answer is 42."]
    )
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact)
    turn = session.send("what is the answer?")
    assert turn.ok
    assert turn.display_text == "The answer is 42."
    assert SECRET not in turn.display_text
    assert turn.reasoning_text == SECRET


def test_an_unclosed_think_block_never_reaches_the_display(tmp_path):
    service, _factory, _runtime = build_service(tmp_path, answers=[f"<think>{SECRET}"])
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact)
    turn = session.send("hello")
    assert turn.ok
    assert SECRET not in turn.display_text
    assert turn.display_text == ""


def test_hidden_reasoning_is_not_fed_back_into_the_conversation(tmp_path):
    service, _factory, runtime = build_service(
        tmp_path, answers=[f"<think>{SECRET}</think>visible", "second"]
    )
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact)
    session.send("first question")
    session.send("second question")

    stored = [content for role, content in runtime.histories[-1] if role == "assistant"]
    assert stored == ["visible"]
    assert all(SECRET not in content for _role, content in runtime.histories[-1])


def test_show_reasoning_renders_everything(tmp_path):
    service, _factory, _runtime = build_service(
        tmp_path, answers=["<think>reasoning</think>visible"]
    )
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact, options=ChatOptions(show_reasoning=True))
    turn = session.send("hello")
    assert turn.display_text == "<think>reasoning</think>visible"
    assert turn.reasoning_text == "reasoning"


def test_show_reasoning_asks_the_template_for_thinking_mode(tmp_path):
    service, factory, _runtime = build_service(tmp_path)
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact, options=ChatOptions(show_reasoning=True))
    session.load()
    assert session.thinking is True
    assert factory.kwargs[0]["thinking"] is True


def test_an_explicit_thinking_choice_beats_the_show_reasoning_hint(tmp_path):
    service, _factory, _runtime = build_service(tmp_path)
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(
        artifact, options=ChatOptions(show_reasoning=True, thinking=False)
    )
    assert session.thinking is False


# --- metrics are best effort ----------------------------------------------------------


def test_missing_metrics_do_not_break_a_turn(tmp_path):
    class BareRuntime(FakeRuntime):
        def generate(self, history, config):
            super().generate(history, config)
            return GenerationResult(text="answer", display_text="answer")

    service, factory, _runtime = build_service(tmp_path)
    bare = BareRuntime(["answer"])
    factory.runtime = bare
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact)
    turn = session.send("hello")
    assert turn.ok
    assert turn.result is not None
    assert turn.result.ttft_ms is None
    assert turn.result.output_tokens is None
    assert turn.display_text == "answer"


def test_metrics_are_carried_through_the_sanitiser(tmp_path):
    service, _factory, _runtime = build_service(tmp_path, answers=["<think>x</think>answer"])
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact)
    turn = session.send("hello")
    assert turn.result is not None
    assert turn.result.ttft_ms == pytest.approx(180.0)
    assert turn.result.output_tokens == 34
    assert turn.result.text == "<think>x</think>answer"


# --- generation errors return to the prompt -------------------------------------------


def test_a_generation_error_returns_a_diagnostic_and_keeps_the_session(tmp_path):
    failure = GenerationError(
        diagnostic(
            "runtime.genai.generation_failed",
            Category.UNKNOWN_RUNTIME_ERROR,
            "Generation failed on GPU",
            "the device went away",
        )
    )
    sink = CollectingSink()
    service, factory, _runtime = build_service(
        tmp_path, answers=[failure, "recovered"], progress=sink
    )
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact)

    failed = session.send("first")
    assert failed.ok is False
    assert failed.diagnostic is not None
    assert failed.diagnostic.code == "runtime.genai.generation_failed"
    assert failed.display_text == ""
    assert [m.role for m in session.history.messages] == []

    recovered = session.send("second")
    assert recovered.ok
    assert recovered.display_text == "recovered"
    assert factory.calls == 1
    assert any(event.severity is Severity.ERROR for event in sink.of_kind(ProgressKind.LOG))


def test_an_untyped_runtime_failure_is_still_reported_as_a_diagnostic(tmp_path):
    service, _factory, _runtime = build_service(tmp_path, answers=[RuntimeError("boom")])
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact)
    turn = session.send("hello")
    assert turn.ok is False
    assert turn.diagnostic is not None
    assert turn.diagnostic.code == "chat.generation.unexpected_error"
    assert "boom" in turn.diagnostic.message


def test_an_empty_prompt_is_refused_without_calling_the_runtime(tmp_path):
    service, factory, _runtime = build_service(tmp_path)
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact)
    turn = session.send("   \n  ")
    assert turn.ok is False
    assert turn.diagnostic is not None
    assert turn.diagnostic.code == "chat.prompt.empty"
    assert factory.calls == 0


# --- conversation state ---------------------------------------------------------------


def test_the_whole_conversation_is_resupplied_each_turn(tmp_path):
    service, _factory, runtime = build_service(tmp_path, answers=["one", "two"])
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact, options=ChatOptions(system_prompt="Be terse."))
    session.send("first")
    session.send("second")
    assert runtime.histories[0] == [("system", "Be terse."), ("user", "first")]
    assert runtime.histories[1] == [
        ("system", "Be terse."),
        ("user", "first"),
        ("assistant", "one"),
        ("user", "second"),
    ]


def test_reset_keeps_the_system_prompt(tmp_path):
    service, _factory, _runtime = build_service(tmp_path, answers=["one"])
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact, options=ChatOptions(system_prompt="Be terse."))
    session.send("first")
    session.reset()
    assert [m.role for m in session.history.messages] == ["system"]


# --- commands -------------------------------------------------------------------------


def parsed(line: str) -> ParsedCommand:
    """Parse a command line with the real UI parser."""
    command = parse_command(line)
    assert command is not None
    return command


def test_tokens_command_changes_the_generation_settings(session_parts):
    service, _factory, _runtime, artifact = session_parts
    session = service.open_artifact(artifact)
    outcome = session.apply_command(parsed("/tokens 256"))
    assert outcome.ok
    assert outcome.action is ChatAction.NONE
    assert "256" in outcome.message
    assert session.generation_settings.max_new_tokens == 256


def test_tokens_command_is_applied_to_the_next_turn(tmp_path):
    service, _factory, runtime = build_service(tmp_path, answers=["answer"])
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact)
    session.apply_command(parsed("/tokens 64"))
    session.send("hello")
    assert runtime.configs[0].max_new_tokens == 64


def test_a_rejected_argument_reports_an_error_and_changes_nothing(session_parts):
    service, _factory, _runtime, artifact = session_parts
    session = service.open_artifact(artifact)
    before = session.generation_settings.max_new_tokens
    outcome = session.apply_command(parsed("/tokens not-a-number"))
    assert outcome.ok is False
    assert outcome.message
    assert session.generation_settings.max_new_tokens == before


def test_a_missing_argument_reports_a_usage_message(session_parts):
    service, _factory, _runtime, artifact = session_parts
    session = service.open_artifact(artifact)
    outcome = session.apply_command(FakeCommand(kind="tokens"))
    assert outcome.ok is False
    assert "/tokens N" in outcome.message


def test_an_out_of_range_argument_is_refused_by_the_service_itself(session_parts):
    service, _factory, _runtime, artifact = session_parts
    session = service.open_artifact(artifact)
    outcome = session.apply_command(FakeCommand(kind="tokens", int_value=0))
    assert outcome.ok is False
    assert "between 1 and" in outcome.message
    with pytest.raises(ValueError, match="between 1 and"):
        session.set_tokens(0)


def test_reset_command_clears_the_conversation(tmp_path):
    service, _factory, _runtime = build_service(tmp_path, answers=["one"])
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact)
    session.send("first")
    assert len(session.history.messages) == 2
    outcome = session.apply_command(parsed("/reset"))
    assert outcome.ok
    assert session.history.messages == []


def test_stats_command_toggles_the_metrics_footer(session_parts):
    service, _factory, _runtime, artifact = session_parts
    session = service.open_artifact(artifact)
    states = [session.stats]
    assert session.apply_command(parsed("/stats off")).ok
    states.append(session.stats)
    assert session.apply_command(parsed("/stats on")).ok
    states.append(session.stats)
    assert states == [True, False, True]


def test_exit_command_asks_the_loop_to_stop(session_parts):
    service, _factory, _runtime, artifact = session_parts
    session = service.open_artifact(artifact)
    outcome = session.apply_command(parsed("/exit"))
    assert outcome.action is ChatAction.EXIT
    assert outcome.exit_requested is True
    for alias in ("/quit", "/q", "/bye"):
        assert session.apply_command(parsed(alias)).exit_requested is True


@pytest.mark.parametrize(
    ("line", "action"),
    [("/help", ChatAction.HELP), ("/clear", ChatAction.CLEAR), ("/model", ChatAction.MODEL)],
)
def test_presentation_commands_are_delegated_to_the_loop(session_parts, line, action):
    service, _factory, _runtime, artifact = session_parts
    session = service.open_artifact(artifact)
    outcome = session.apply_command(parsed(line))
    assert outcome.ok
    assert outcome.action is action


def test_reasoning_command_toggles_the_sanitiser(tmp_path):
    service, _factory, _runtime = build_service(
        tmp_path, answers=["<think>hidden</think>one", "<think>hidden</think>two"]
    )
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact)
    assert session.send("a").display_text == "one"
    outcome = session.apply_command(parsed("/reasoning on"))
    assert outcome.ok
    assert session.show_reasoning is True
    assert session.send("b").display_text == "<think>hidden</think>two"


def test_sampling_commands_change_the_settings(session_parts):
    service, _factory, _runtime, artifact = session_parts
    session = service.open_artifact(artifact)
    assert session.apply_command(parsed("/temp 0.3")).ok
    assert session.apply_command(parsed("/top-p 0.4")).ok
    assert session.apply_command(parsed("/top-k 5")).ok
    settings = session.generation_settings
    assert settings.temperature == pytest.approx(0.3)
    assert settings.top_p == pytest.approx(0.4)
    assert settings.top_k == 5
    assert settings.do_sample is True


def test_a_zero_temperature_switches_to_greedy_decoding(session_parts):
    service, _factory, _runtime, artifact = session_parts
    session = service.open_artifact(artifact)
    outcome = session.apply_command(parsed("/temp 0"))
    assert outcome.ok
    assert "greedy" in outcome.message
    assert session.generation_settings.do_sample is False


def test_zero_top_k_disables_it(session_parts):
    service, _factory, _runtime, artifact = session_parts
    session = service.open_artifact(artifact)
    outcome = session.apply_command(parsed("/top-k 0"))
    assert outcome.ok
    assert session.generation_settings.top_k is None


def test_an_unknown_command_is_reported_not_executed(session_parts):
    service, _factory, _runtime, artifact = session_parts
    session = service.open_artifact(artifact)
    outcome = session.apply_command(parsed("/nope"))
    assert outcome.ok is False
    assert "/nope" in outcome.message


def test_the_ui_command_type_satisfies_the_service_port():
    command = parse_command("/tokens 10")
    assert command is not None
    assert isinstance(command, ParsedCommand)


# --- description and history path ------------------------------------------------------


def test_the_description_renders_as_a_ui_session_info(session_parts):
    service, _factory, _runtime, artifact = session_parts
    session = service.open_artifact(artifact)
    description = session.describe()
    info = ChatSessionInfo.model_validate(description.model_dump())
    assert info.display_name == "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"
    assert info.device == "GPU"
    assert info.profile == "gpu-int4-asym-g128"
    assert info.max_new_tokens == 4096
    assert info.show_reasoning is False
    assert info.source_label == f"{QWEN_REPO}@{'b' * 12}"
    assert description.extra["Chat profile"] == "chat-qwen3_5"
    assert description.extra["Thinking mode"] == "off"


def test_the_history_path_follows_the_xdg_data_policy(tmp_path):
    settings = make_settings(tmp_path)
    service, _factory, _runtime = build_service(tmp_path, settings=settings)
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact)

    path = session.history_path
    assert path.parent == settings.paths.chat_history_dir
    assert path.parent == tmp_path / "data" / "chat-history"
    assert path.name == "qwen35-9b--0123456789ab--gpu-int4-asym-g128.history"
    assert tmp_path in path.parents


def test_the_service_and_the_view_agree_on_the_history_file(tmp_path):
    settings = make_settings(tmp_path)
    service, _factory, _runtime = build_service(tmp_path, settings=settings)
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact)
    info = ChatSessionInfo.model_validate(session.describe().model_dump())
    view = ChatView(YongaConsole(), info, paths=settings.paths)
    assert view.history_path == session.history_path


def test_the_history_path_falls_back_to_the_repository_id(tmp_path):
    settings = make_settings(tmp_path)
    service, _factory, _runtime = build_service(tmp_path, settings=settings)
    artifact = write_artifact(tmp_path, make_manifest(artifact_id="x"))
    session = service.open_artifact(artifact)
    assert session.history_path.name == "x.history"


def test_the_compile_cache_path_is_reported_from_the_configuration(tmp_path):
    settings = make_settings(tmp_path)
    service, _factory, _runtime = build_service(tmp_path, settings=settings)
    artifact = write_artifact(tmp_path, make_manifest())
    session = service.open_artifact(artifact)
    description = session.describe()
    assert description.compile_cache_enabled is True
    assert description.compile_cache_path == settings.paths.compile_cache_dir


# --- the service resolves references through the store ---------------------------------


def test_open_session_asks_the_artifact_store(tmp_path):
    runtime = FakeRuntime(["answer"])
    factory = FakeFactory(runtime)
    artifact = write_artifact(tmp_path, make_manifest())
    store = FakeStore({QWEN_REPO: artifact})
    contract: ArtifactStoreLike = store
    service = ChatService(
        factory,
        contract,
        RegistryResolver(load_builtin_registry()),
        make_settings(tmp_path),
    )
    session = service.open_session(QWEN_REPO)
    assert store.asked == [QWEN_REPO]
    assert session.artifact_dir == artifact
    assert isinstance(session, ChatSession)
