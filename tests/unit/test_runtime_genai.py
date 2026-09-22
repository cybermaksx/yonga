"""Unit tests for the OpenVINO GenAI chat runtime.

``openvino_genai`` is stubbed out throughout: the lazy import hook is
monkeypatched so these tests run on a host with no GenAI wheel and no Intel
accelerator. The real pipeline is exercised in
``tests/integration/test_openvino_cpu.py``.
"""

from __future__ import annotations

import math
import types

import pytest

from yonga.adapters import runtime_genai as runtime_module
from yonga.adapters.runtime_genai import (
    ChatHistory,
    ChatMessage,
    ChatProfile,
    ChatRuntime,
    GenAIChatRuntime,
    GenerationResult,
    build_generation_config,
    collect_metrics,
    open_runtime,
    thinking_extra_context,
)
from yonga.config.settings import Settings
from yonga.domain.benchmarks import GenerationSettings
from yonga.domain.diagnostics import CompileError, FailureCategory, GenerationError
from yonga.domain.models import PipelineKind

# --- fakes ----------------------------------------------------------------------------


class MeanStd:
    """Stand-in for ``openvino_genai.MeanStdPair``."""

    def __init__(self, mean, std=0.0):
        self.mean = mean
        self.std = std


class FakePerfMetrics:
    """A perf-metrics object with one getter that raises, as real ones can."""

    def get_num_input_tokens(self):
        return 9

    def get_num_generated_tokens(self):
        return 16

    def get_ttft(self):
        return MeanStd(8.19)

    def get_tpot(self):
        raise RuntimeError("no token times recorded")

    def get_throughput(self):
        return MeanStd(float("nan"))

    def get_load_time(self):
        # Verified upstream: this one returns a plain float, not a MeanStdPair.
        return 96.0


class FakeFinishReason:
    def __init__(self, name):
        self.name = name


class FakeResults:
    def __init__(self, text="hello there", reason="LENGTH", perf_metrics=None):
        self.texts = [text]
        self.finish_reasons = [FakeFinishReason(reason)]
        self.perf_metrics = perf_metrics


class FakeGenerationConfig:
    """Only ``max_new_tokens``/``do_sample`` pre-exist, so unset fields stay absent."""

    def __init__(self):
        self.max_new_tokens = 0
        self.do_sample = False
        self.validated = 0
        self.validate_error = None

    def validate(self):
        self.validated += 1
        if self.validate_error is not None:
            raise self.validate_error


class FakeChatHistory:
    def __init__(self):
        self.messages = []
        self.extra_context = None
        self.extra_context_error = None

    def append(self, message):
        self.messages.append(message)

    def set_extra_context(self, context):
        if self.extra_context_error is not None:
            raise self.extra_context_error
        self.extra_context = context


class FakePipeline:
    def __init__(self, module, kind, path, device, properties):
        self.module = module
        self.kind = kind
        self.path = path
        self.device = device
        self.properties = properties
        self.generate_calls = []
        self.chat_events = []

    def generate(self, *args, **kwargs):
        self.generate_calls.append((args, kwargs))
        if self.module.generate_error is not None:
            raise self.module.generate_error
        return self.module.result

    def start_chat(self, system_message=None):
        self.chat_events.append(("start", system_message))

    def finish_chat(self):
        self.chat_events.append(("finish", None))


class FakeGenAI:
    """A scriptable ``openvino_genai`` module."""

    def __init__(self):
        self.ChatHistory = FakeChatHistory
        self.GenerationConfig = FakeGenerationConfig
        self.pipelines = []
        self.load_error = None
        self.generate_error = None
        self.result = FakeResults(perf_metrics=FakePerfMetrics())

    def _make(self, kind, path, device, properties):
        if self.load_error is not None:
            raise self.load_error
        pipeline = FakePipeline(self, kind, path, device, properties)
        self.pipelines.append(pipeline)
        return pipeline

    def LLMPipeline(self, path, device, **properties):
        return self._make("llm", path, device, properties)

    def VLMPipeline(self, path, device, **properties):
        return self._make("vlm", path, device, properties)


@pytest.fixture(autouse=True)
def _no_real_genai(monkeypatch):
    """Fail loudly if a test forgets to stub the lazy import."""

    def _raise():
        raise ImportError("No module named 'openvino_genai'")

    monkeypatch.setattr(runtime_module, "_import_genai", _raise)
    monkeypatch.setattr(runtime_module, "openvino_runtime_version", lambda: "2026.4.0")


@pytest.fixture
def genai(monkeypatch):
    module = FakeGenAI()
    monkeypatch.setattr(runtime_module, "_import_genai", lambda: module)
    return module


@pytest.fixture
def settings(tmp_path):
    return Settings(
        cache_dir=tmp_path / "cache",
        data_dir=tmp_path / "data",
        config_dir=tmp_path / "config",
    )


@pytest.fixture
def artifact(tmp_path):
    directory = tmp_path / "artifact"
    directory.mkdir()
    (directory / "openvino_model.xml").write_bytes(b"<net/>")
    (directory / "openvino_model.bin").write_bytes(b"\x00")
    return directory


def make_runtime(artifact, settings, **kwargs):
    runtime = open_runtime(artifact, settings=settings, **kwargs)
    assert isinstance(runtime, GenAIChatRuntime)
    return runtime


# --- conversation types ---------------------------------------------------------------


def test_chat_history_helpers():
    history = ChatHistory()
    history.add_system("You are terse.")
    history.add_user("Hello")
    history.add_assistant("Hi")
    assert len(history) == 3
    assert history.system_message == "You are terse."
    assert history.to_payload()[1] == {"role": "user", "content": "Hello"}

    history.reset()
    assert [m.role for m in history.messages] == ["system"]
    history.reset(keep_system=False)
    assert history.messages == []


def test_chat_message_rejects_an_unknown_role():
    with pytest.raises(ValueError, match="role"):
        ChatMessage(role="tool", content="x")


# --- chat profile ---------------------------------------------------------------------


def test_thinking_extra_context_is_empty_without_a_profile():
    assert thinking_extra_context(None) == {}
    assert thinking_extra_context(ChatProfile()) == {}


def test_thinking_extra_context_uses_the_profile_default_and_override():
    profile = ChatProfile(supports_thinking_flag=True, default_thinking=False)
    assert thinking_extra_context(profile) == {"enable_thinking": False}
    assert thinking_extra_context(profile, thinking=True) == {"enable_thinking": True}

    named = ChatProfile(
        supports_thinking_flag=True,
        thinking_template_variable="reasoning",
        default_thinking=True,
    )
    assert thinking_extra_context(named) == {"reasoning": True}
    assert thinking_extra_context(named, thinking=False) == {"reasoning": False}


def test_thinking_extra_context_ignores_a_blank_variable_name():
    profile = ChatProfile(supports_thinking_flag=True, thinking_template_variable="   ")
    assert thinking_extra_context(profile) == {}


# --- metrics --------------------------------------------------------------------------


def test_collect_metrics_survives_a_metric_that_raises():
    metrics = collect_metrics(FakePerfMetrics())
    assert metrics.input_tokens == 9
    assert metrics.output_tokens == 16
    assert metrics.ttft_ms == pytest.approx(8.19)
    # get_tpot() raised and get_throughput() returned NaN: both degrade to None.
    assert metrics.tpot_ms is None
    assert metrics.throughput_tps is None
    # get_load_time() returns a plain float rather than a MeanStdPair.
    assert metrics.load_ms == pytest.approx(96.0)


def test_collect_metrics_handles_absent_metrics():
    empty = collect_metrics(None)
    assert empty.model_dump() == dict.fromkeys(empty.model_dump())

    nothing = collect_metrics(types.SimpleNamespace())
    assert nothing.ttft_ms is None
    assert nothing.input_tokens is None


def test_collect_metrics_ignores_unconvertible_values():
    metrics = collect_metrics(
        types.SimpleNamespace(
            get_num_input_tokens=lambda: "many",
            get_ttft=lambda: MeanStd("fast"),
            get_throughput=lambda: MeanStd(math.inf),
        )
    )
    assert metrics.input_tokens is None
    assert metrics.ttft_ms is None
    assert metrics.throughput_tps is None


# --- generation config ----------------------------------------------------------------


def test_generation_settings_map_onto_a_greedy_config(genai):
    config = build_generation_config(GenerationSettings(max_new_tokens=128))
    assert config.max_new_tokens == 128
    assert config.do_sample is False
    assert config.validated == 1
    # Sampling knobs are inert without do_sample, so they are not set at all.
    assert not hasattr(config, "temperature")
    assert not hasattr(config, "top_p")
    assert not hasattr(config, "top_k")
    assert not hasattr(config, "rng_seed")


def test_generation_settings_map_onto_a_sampling_config(genai):
    config = build_generation_config(
        GenerationSettings(
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            seed=1234,
        )
    )
    assert config.do_sample is True
    assert config.temperature == pytest.approx(0.7)
    assert config.top_p == pytest.approx(0.9)
    assert config.top_k == 40
    assert config.rng_seed == 1234
    assert config.repetition_penalty == pytest.approx(1.1)


def test_repetition_penalty_applies_without_sampling(genai):
    config = build_generation_config(GenerationSettings(repetition_penalty=1.2))
    assert config.repetition_penalty == pytest.approx(1.2)
    assert not hasattr(config, "temperature")


def test_max_new_tokens_never_drops_below_one(genai):
    assert build_generation_config(GenerationSettings(max_new_tokens=0)).max_new_tokens == 1


def test_invalid_generation_config_raises_a_typed_error(genai, monkeypatch):
    class Rejecting(FakeGenerationConfig):
        def validate(self):
            raise RuntimeError("Either 'eos_token_id' or 'max_new_tokens' should be defined.")

    genai.GenerationConfig = Rejecting
    with pytest.raises(GenerationError) as excinfo:
        build_generation_config(GenerationSettings())
    assert excinfo.value.diagnostic.code == "runtime.generation.invalid_config"
    assert "eos_token_id" in excinfo.value.diagnostic.message


def test_generation_config_without_genai_is_a_typed_environment_error():
    with pytest.raises(CompileError) as excinfo:
        build_generation_config(GenerationSettings())
    assert excinfo.value.diagnostic.code == "runtime.openvino_genai.missing"
    assert excinfo.value.diagnostic.category is FailureCategory.PACKAGE_ABI_MISMATCH
    assert excinfo.value.diagnostic.environment_broken is True


# --- opening a runtime ----------------------------------------------------------------


def test_open_runtime_without_genai_raises_compile_error(artifact, settings):
    with pytest.raises(CompileError) as excinfo:
        open_runtime(artifact, settings=settings)
    assert excinfo.value.diagnostic.code == "runtime.openvino_genai.missing"


def test_open_runtime_rejects_a_missing_artifact(tmp_path, settings, genai):
    with pytest.raises(CompileError) as excinfo:
        open_runtime(tmp_path / "nope", settings=settings)
    assert excinfo.value.diagnostic.code == "runtime.artifact.not_found"


def test_open_runtime_preserves_the_raw_load_error(artifact, settings, genai):
    genai.load_error = RuntimeError(
        "Exception from src/inference/src/cpp/core.cpp:84:\nUnsupported element type: STRiNG"
    )
    with pytest.raises(CompileError) as excinfo:
        open_runtime(artifact, device="GPU", settings=settings)
    diagnostic = excinfo.value.diagnostic
    assert diagnostic.code == "runtime.genai.load_failed"
    assert "Unsupported element type: STRiNG" in diagnostic.message
    assert "Unsupported element type: STRiNG" in (diagnostic.raw_excerpt or "")
    assert diagnostic.context["device"] == "GPU"


def test_runtime_satisfies_the_chat_runtime_protocol(artifact, settings, genai):
    runtime = make_runtime(artifact, settings)
    assert isinstance(runtime, ChatRuntime)
    assert runtime.device == "CPU"
    assert runtime.pipeline_kind is PipelineKind.LLM
    assert runtime.artifact_dir == artifact
    runtime.close()


def test_vlm_manifest_selects_the_vlm_pipeline(artifact, settings, genai):
    runtime = make_runtime(artifact, settings, pipeline=PipelineKind.VLM)
    assert genai.pipelines[0].kind == "vlm"
    runtime.close()


def test_unknown_pipeline_kind_is_driven_as_an_llm(artifact, settings, genai):
    runtime = make_runtime(artifact, settings, pipeline=PipelineKind.UNKNOWN)
    assert genai.pipelines[0].kind == "llm"
    runtime.close()


# --- compile cache --------------------------------------------------------------------


def test_compile_cache_dir_is_injected_and_versioned(artifact, settings, genai, monkeypatch):
    runtime = make_runtime(artifact, settings, device="GPU")
    first = runtime.properties["CACHE_DIR"]
    assert first.startswith(str(settings.paths.compile_cache_dir))
    runtime.close()

    monkeypatch.setattr(runtime_module, "openvino_runtime_version", lambda: "2027.1.0")
    other = make_runtime(artifact, settings, device="GPU")
    assert other.properties["CACHE_DIR"] != first
    other.close()


def test_compile_cache_can_be_disabled(artifact, tmp_path, genai):
    settings = Settings(
        cache_dir=tmp_path / "cache",
        data_dir=tmp_path / "data",
        config_dir=tmp_path / "config",
        compile_cache_enabled=False,
    )
    runtime = make_runtime(artifact, settings, properties={"ATTENTION_BACKEND": "SDPA"})
    assert runtime.properties == {"ATTENTION_BACKEND": "SDPA"}
    runtime.close()


def test_an_explicit_cache_dir_wins(artifact, settings, genai):
    runtime = make_runtime(artifact, settings, properties={"CACHE_DIR": "/tmp/mine"})
    assert runtime.properties["CACHE_DIR"] == "/tmp/mine"
    assert genai.pipelines[0].properties["CACHE_DIR"] == "/tmp/mine"
    runtime.close()


# --- LLM generation -------------------------------------------------------------------


def test_generate_sends_the_whole_history_and_collects_metrics(artifact, settings, genai):
    profile = ChatProfile(supports_thinking_flag=True, default_thinking=False)
    runtime = make_runtime(artifact, settings, chat_profile=profile)
    history = ChatHistory()
    history.add_system("You are terse.")
    history.add_user("Hello")

    result = runtime.generate(history, GenerationSettings(max_new_tokens=16))

    (args, _kwargs) = genai.pipelines[0].generate_calls[0]
    sent = args[0]
    assert isinstance(sent, FakeChatHistory)
    assert sent.messages == [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hello"},
    ]
    # The thinking flag travels as a chat-template variable, not a template rewrite.
    assert sent.extra_context == {"enable_thinking": False}

    assert isinstance(result, GenerationResult)
    assert result.text == "hello there"
    assert result.display_text == result.text
    assert result.reasoning_text is None
    assert result.input_tokens == 9
    assert result.output_tokens == 16
    assert result.ttft_ms == pytest.approx(8.19)
    assert result.tpot_ms is None
    assert result.throughput_tps is None
    assert result.load_seconds == pytest.approx(0.096)
    assert result.wall_seconds >= 0.0
    assert result.stop_reason == "length"
    runtime.close()


def test_generate_without_a_profile_sets_no_extra_context(artifact, settings, genai):
    runtime = make_runtime(artifact, settings)
    history = ChatHistory()
    history.add_user("Hello")
    runtime.generate(history, GenerationSettings())
    (args, _kwargs) = genai.pipelines[0].generate_calls[0]
    assert args[0].extra_context is None
    runtime.close()


def test_a_failing_extra_context_is_recorded_but_not_fatal(artifact, settings, genai):
    class Rejecting(FakeChatHistory):
        def set_extra_context(self, context):
            raise RuntimeError("set_extra_context is not available")

    genai.ChatHistory = Rejecting
    profile = ChatProfile(supports_thinking_flag=True)
    runtime = make_runtime(artifact, settings, chat_profile=profile)
    history = ChatHistory()
    history.add_user("Hello")

    result = runtime.generate(history, GenerationSettings())
    assert result.text == "hello there"
    assert "set_extra_context is not available" in (runtime.extra_context_error or "")
    runtime.close()


def test_the_pipeline_is_reused_across_turns(artifact, settings, genai):
    runtime = make_runtime(artifact, settings)
    history = ChatHistory()
    history.add_user("One")
    runtime.generate(history, GenerationSettings())
    history.add_assistant("hello there")
    history.add_user("Two")
    runtime.generate(history, GenerationSettings())

    assert len(genai.pipelines) == 1
    assert len(genai.pipelines[0].generate_calls) == 2
    runtime.close()


def test_generation_failure_preserves_the_raw_error(artifact, settings, genai):
    genai.generate_error = RuntimeError("Exception from src/plugins/intel_npu: out of memory")
    runtime = make_runtime(artifact, settings, device="NPU")
    history = ChatHistory()
    history.add_user("Hello")

    with pytest.raises(GenerationError) as excinfo:
        runtime.generate(history, GenerationSettings())
    diagnostic = excinfo.value.diagnostic
    assert diagnostic.code == "runtime.genai.generation_failed"
    assert "out of memory" in diagnostic.message
    assert diagnostic.context["device"] == "NPU"
    runtime.close()


def test_generate_without_metrics_still_returns_a_result(artifact, settings, genai):
    genai.result = FakeResults(text="ok", perf_metrics=None)
    runtime = make_runtime(artifact, settings)
    history = ChatHistory()
    history.add_user("Hello")

    result = runtime.generate(history, GenerationSettings())
    assert result.text == "ok"
    assert result.input_tokens is None
    assert result.ttft_ms is None
    # Falls back to the measured pipeline load time.
    assert result.load_seconds is not None
    runtime.close()


# --- VLM generation -------------------------------------------------------------------


def test_vlm_hides_the_deprecated_chat_lifecycle(artifact, settings, genai):
    runtime = make_runtime(artifact, settings, pipeline=PipelineKind.VLM)
    pipeline = genai.pipelines[0]
    history = ChatHistory()
    history.add_system("You are terse.")
    history.add_user("First")

    runtime.generate(history, GenerationSettings())
    assert pipeline.chat_events == [("start", "You are terse.")]
    (args, kwargs) = pipeline.generate_calls[0]
    assert args[0] == "First"
    assert "generation_config" in kwargs

    history.add_assistant("hello there")
    history.add_user("Second")
    runtime.generate(history, GenerationSettings())
    # Only the new user turn is sent; the pipeline keeps its own history.
    assert pipeline.generate_calls[1][0][0] == "Second"
    assert pipeline.chat_events == [("start", "You are terse.")]

    runtime.close()
    assert pipeline.chat_events[-1] == ("finish", None)


def test_vlm_restarts_the_session_when_the_history_is_reset(artifact, settings, genai):
    runtime = make_runtime(artifact, settings, pipeline=PipelineKind.VLM)
    pipeline = genai.pipelines[0]
    history = ChatHistory()
    history.add_user("First")
    runtime.generate(history, GenerationSettings())

    history.reset(keep_system=False)
    history.add_user("Fresh start")
    runtime.generate(history, GenerationSettings())

    assert [event for event, _ in pipeline.chat_events] == ["start", "finish", "start"]
    assert pipeline.generate_calls[1][0][0] == "Fresh start"
    runtime.close()


def test_vlm_without_a_new_user_turn_is_rejected(artifact, settings, genai):
    runtime = make_runtime(artifact, settings, pipeline=PipelineKind.VLM)
    history = ChatHistory()
    history.add_system("You are terse.")

    with pytest.raises(GenerationError) as excinfo:
        runtime.generate(history, GenerationSettings())
    assert excinfo.value.diagnostic.code == "runtime.generation.empty_prompt"
    runtime.close()


# --- lifecycle ------------------------------------------------------------------------


def test_close_is_idempotent_and_blocks_further_generation(artifact, settings, genai):
    runtime = make_runtime(artifact, settings)
    runtime.close()
    runtime.close()
    assert runtime.closed is True

    history = ChatHistory()
    history.add_user("Hello")
    with pytest.raises(GenerationError) as excinfo:
        runtime.generate(history, GenerationSettings())
    assert excinfo.value.diagnostic.code == "runtime.genai.closed"


def test_runtime_works_as_a_context_manager(artifact, settings, genai):
    with GenAIChatRuntime(artifact, settings=settings) as runtime:
        assert runtime.closed is False
    assert runtime.closed is True
