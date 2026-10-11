"""Unit tests for the OpenVINO GenAI chat runtime.

``openvino_genai`` is stubbed out throughout: the lazy import hook is
monkeypatched so these tests run on a host with no GenAI wheel and no Intel
accelerator. The real pipeline is exercised in
``tests/integration/test_openvino_cpu.py``.
"""

from __future__ import annotations

import json
import math
import types
import warnings
from pathlib import Path
from typing import Any, cast

import pytest

from yonga.adapters import runtime_genai as runtime_module
from yonga.adapters.runtime_genai import (
    ChatHistory,
    ChatImage,
    ChatMessage,
    ChatProfile,
    ChatRuntime,
    GenAIChatRuntime,
    GenerationResult,
    ToolCall,
    build_generation_config,
    collect_metrics,
    open_runtime,
    resolve_starts_in_reasoning,
    template_opens_reasoning,
    thinking_extra_context,
)
from yonga.config.settings import Settings
from yonga.domain.benchmarks import GenerationSettings
from yonga.domain.diagnostics import CompileError, FailureCategory, GenerationError
from yonga.domain.models import PipelineKind

# --- fakes ----------------------------------------------------------------------------


class MeanStd:
    """Stand-in for ``openvino_genai.MeanStdPair``."""

    def __init__(self, mean: Any, std: Any = 0.0) -> None:
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
    def __init__(self, name: str) -> None:
        self.name = name


class FakeResults:
    def __init__(
        self,
        text: str = "hello there",
        reason: str = "LENGTH",
        perf_metrics: Any = None,
    ) -> None:
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
        self.tools = None

    def append(self, message):
        self.messages.append(message)

    def set_tools(self, tools):
        self.tools = tools

    def set_extra_context(self, context):
        if self.extra_context_error is not None:
            raise self.extra_context_error
        self.extra_context = context


class FakeTokenizedInputs:
    """Stand-in for ``openvino_genai.TokenizedInputs``: remembers what was encoded."""

    def __init__(self, prompt: str, add_special_tokens: bool) -> None:
        self.prompt = prompt
        self.add_special_tokens = add_special_tokens


class FakeEncodedResults:
    """What GenAI returns for tokenized inputs: token ids, not texts."""

    def __init__(self, tokens: list[int], finish_reasons: Any, perf_metrics: Any) -> None:
        self.tokens = [tokens]
        self.scores = [0.0]
        self.finish_reasons = finish_reasons
        self.perf_metrics = perf_metrics


#: Token ids the fake pipeline "generates" for tokenized inputs.
ENCODED_TOKENS = [11, 12, 13]


class FakeTokenizer:
    """Stand-in for the pipeline's ``openvino_genai.Tokenizer``."""

    RENDERED = "<|rendered chat|>"

    def __init__(self, module: Any) -> None:
        self.module = module
        self.template_calls: list[dict[str, Any]] = []
        self.decode_calls: list[tuple[list[int], bool]] = []

    def apply_chat_template(
        self, history, add_generation_prompt, chat_template="", tools=None, extra_context=None
    ):
        self.template_calls.append(
            {
                "history": history,
                "add_generation_prompt": add_generation_prompt,
                "chat_template": chat_template,
                "tools": tools,
                "extra_context": extra_context,
            }
        )
        return self.RENDERED

    def encode(self, prompt, add_special_tokens=True):
        return FakeTokenizedInputs(prompt, add_special_tokens)

    def decode(self, tokens, skip_special_tokens=True):
        self.decode_calls.append((list(tokens), skip_special_tokens))
        return self.module.result.texts[0]


class FakePipeline:
    def __init__(self, module: Any, kind: Any, path: Any, device: Any, properties: Any) -> None:
        self.module = module
        self.kind = kind
        self.path = path
        self.device = device
        self.properties = properties
        self.generate_calls: list[tuple[Any, ...]] = []
        self.chat_events: list[tuple[str, Any]] = []
        #: Every generate and lifecycle call, in order, to check what came first.
        self.events: list[str] = []
        self.tokenizer = FakeTokenizer(module)

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, *args, **kwargs):
        if (
            self.kind == "vlm"
            and args
            and isinstance(args[0], FakeChatHistory)
            and not self.module.vlm_accepts_history
        ):
            # What pybind11 raises for a missing overload, before anything runs.
            raise TypeError("generate(): incompatible function arguments.")
        self.generate_calls.append((args, kwargs))
        self.events.append("generate")
        if self.module.generate_error is not None:
            raise self.module.generate_error
        streamer = kwargs.get("streamer")
        if streamer is not None:
            for chunk in self.module.stream_chunks:
                if streamer(chunk):
                    break
        if args and isinstance(args[0], FakeTokenizedInputs):
            result = self.module.result
            return FakeEncodedResults(
                list(ENCODED_TOKENS), result.finish_reasons, result.perf_metrics
            )
        return self.module.result

    def start_chat(self, system_message=None):
        self.chat_events.append(("start", system_message))
        self.events.append("start_chat")

    def finish_chat(self):
        self.chat_events.append(("finish", None))
        self.events.append("finish_chat")
        if self.module.finish_chat_warns:
            # What the GenAI 2026.4 binding does on every call.
            warnings.warn(
                "start_chat() / finish_chat() API is deprecated", DeprecationWarning, stacklevel=2
            )


class FakeGenAI:
    """A scriptable ``openvino_genai`` module."""

    def __init__(self) -> None:
        self.ChatHistory = FakeChatHistory
        self.GenerationConfig = FakeGenerationConfig
        self.pipelines: list[FakePipeline] = []
        self.load_error: BaseException | None = None
        self.generate_error: BaseException | None = None
        self.result = FakeResults(perf_metrics=FakePerfMetrics())
        #: GenAI before 2026.4 has no ``VLMPipeline.generate(ChatHistory)`` overload;
        #: the legacy-lifecycle tests run against that shape.
        self.vlm_accepts_history = False
        self.stream_chunks: list[str] = []
        self.finish_chat_warns = False

    def _make(self, kind: Any, path: Any, device: Any, properties: Any) -> FakePipeline:
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


def make_runtime(artifact: Path, settings: Settings, **kwargs: Any) -> GenAIChatRuntime:
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
        ChatMessage(role=cast(Any, "function"), content="x")


WEATHER_TOOL = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
}


def tool_conversation() -> ChatHistory:
    call = ToolCall(id="call_1", name="get_weather", arguments='{"city": "Ankara"}')
    return ChatHistory(
        messages=[
            ChatMessage(role="user", content="Weather in Ankara?"),
            ChatMessage(role="assistant", content="", tool_calls=[call]),
            ChatMessage(role="tool", content="21 C", tool_call_id="call_1", name="get_weather"),
        ],
        tools=[WEATHER_TOOL],
    )


def test_tool_turns_take_the_chat_template_convention():
    """Arguments become an object, which templates writing ``arguments | tojson`` need."""
    payload = tool_conversation().to_payload()

    assert payload[1] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Ankara"}},
            }
        ],
    }
    assert payload[2] == {
        "role": "tool",
        "content": "21 C",
        "tool_call_id": "call_1",
        "name": "get_weather",
    }


def test_arguments_that_are_not_a_json_object_stay_text():
    call = ToolCall(id="call_1", name="f", arguments="[1, 2")
    assert call.as_payload()["function"] == {"name": "f", "arguments": "[1, 2"}


def test_tools_are_never_serialised_with_the_history():
    assert "tools" not in tool_conversation().model_dump()


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


def test_registry_stop_strings_are_applied_natively_and_left_out_of_the_answer(genai):
    config = build_generation_config(GenerationSettings(stop_strings=["<|end|>", "", "<|end|>"]))
    assert config.stop_strings == {"<|end|>"}
    assert config.include_stop_str_in_output is False
    assert config.validated == 1


def test_no_stop_strings_leave_the_genai_defaults_untouched(genai):
    config = build_generation_config(GenerationSettings())
    assert not hasattr(config, "stop_strings")
    assert not hasattr(config, "include_stop_str_in_output")


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


def test_the_compile_cache_entry_records_its_owner(artifact, settings, genai, monkeypatch):
    """Every managed entry says which artifact, device and runtime built it."""
    from yonga.adapters.openvino_probe import artifact_fingerprint, read_compile_cache_record
    from yonga.domain.compile_cache import COMPILE_CACHE_ENTRY_FILENAME

    runtime = make_runtime(
        artifact, settings, device="GPU.0", properties={"ATTENTION_BACKEND": "SDPA"}
    )
    directory = Path(runtime.properties["CACHE_DIR"])
    runtime.close()

    record = read_compile_cache_record(directory)
    assert record is not None
    assert record.key == directory.name
    assert record.fingerprint == artifact_fingerprint(artifact)
    assert record.artifact_id is None, "a directory without a manifest has no artifact id"
    assert record.artifact_path == str(artifact)
    assert record.device == "GPU"
    assert record.openvino_version == "2026.4.0"
    assert record.properties == {"ATTENTION_BACKEND": "SDPA"}
    assert record.last_used_at == record.created_at
    # The record is the only file yonga adds, and it cannot pass for a blob.
    assert [path.name for path in directory.iterdir()] == [COMPILE_CACHE_ENTRY_FILENAME]

    reopened = make_runtime(
        artifact, settings, device="GPU", properties={"ATTENTION_BACKEND": "SDPA"}
    )
    assert reopened.properties["CACHE_DIR"] == str(directory)
    reopened.close()
    refreshed = read_compile_cache_record(directory)
    assert refreshed is not None
    assert refreshed.created_at == record.created_at
    assert refreshed.last_used_at >= record.last_used_at


def test_an_unwritable_compile_cache_never_breaks_a_load(artifact, settings, genai, monkeypatch):
    """Bookkeeping that fails is skipped; the pipeline is still built."""

    def explode(*args: object, **kwargs: object) -> object:
        raise PermissionError("read-only cache")

    monkeypatch.setattr("yonga.adapters.openvino_probe.tempfile.mkstemp", explode)
    runtime = make_runtime(artifact, settings, device="CPU")
    directory = Path(runtime.properties["CACHE_DIR"])
    assert genai.pipelines, "the pipeline was built"
    assert directory.is_dir()
    assert not any(directory.iterdir())
    runtime.close()


def test_an_explicit_cache_dir_gets_no_ownership_record(artifact, settings, genai, tmp_path):
    """A directory the caller chose is not yonga's to annotate."""
    mine = tmp_path / "mine"
    mine.mkdir()
    runtime = make_runtime(artifact, settings, properties={"CACHE_DIR": str(mine)})
    runtime.close()
    assert not any(mine.iterdir())


def test_a_staged_artifact_is_never_compile_cached(settings, genai):
    """prepare's smoke test runs on staging, which no later load can ever reuse.

    The staging directory moves into the store under a new fingerprint, and the
    blobs are keyed by the path they were compiled from: caching them there only
    left an orphaned entry the size of the model after every prepare.
    """
    staged = settings.paths.job_staging_dir("82463910-98ac-41c8-9bd7-faf1a4067e6c") / "model"
    staged.mkdir(parents=True)
    (staged / "openvino_model.xml").write_bytes(b"<net/>")
    (staged / "openvino_model.bin").write_bytes(b"\x00")

    runtime = make_runtime(staged, settings, properties={"ATTENTION_BACKEND": "SDPA"})
    runtime.close()

    assert runtime.properties == {"ATTENTION_BACKEND": "SDPA"}
    assert "CACHE_DIR" not in genai.pipelines[0].properties
    assert not settings.paths.compile_cache_dir.exists() or not any(
        settings.paths.compile_cache_dir.iterdir()
    )


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


#: OpenVINO GenAI 2026.4's refusal of a prompt over the NPU's MAX_PROMPT_LEN, as
#: seen on the reference host through ``yonga serve``.
NPU_PROMPT_LIMIT_ERROR = (
    "Check 'data->input_ids.get_size() <= m_max_prompt_len' failed at /home/jenkins/agent/"
    "workspace/private-ci/ie/build-linux-manylinux_2_28/b/repos/openvino.genai/src/cpp/src/llm/"
    "pipeline_stateful.cpp:359:\nStateful LLM pipeline on NPU may only process prompts or hold "
    'chat history up to 1024 tokens. 6423 is passed.\n Set the "MAX_PROMPT_LEN" config option '
    "to increase the limit.\n"
)


@pytest.mark.parametrize(
    ("raw", "limit", "passed"),
    [
        (NPU_PROMPT_LIMIT_ERROR, "1024", "6423"),
        (
            "VLM pipeline on NPU may only process input embeddings up to 2048 tokens. "
            "2100 is passed.",
            "2048",
            "2100",
        ),
    ],
)
def test_a_prompt_over_the_compiled_length_is_the_requests_to_change(
    artifact, settings, genai, raw, limit, passed
):
    """Not an unknown runtime failure: the same pipeline answers a shorter prompt."""
    genai.generate_error = RuntimeError(raw)
    runtime = make_runtime(artifact, settings, device="NPU")
    history = ChatHistory()
    history.add_user("Hello")

    with pytest.raises(GenerationError) as excinfo:
        runtime.generate(history, GenerationSettings())
    report = excinfo.value.diagnostic
    assert report.code == runtime_module.PROMPT_TOO_LONG
    assert report.category is FailureCategory.CONFIGURATION_ERROR
    assert report.message == (
        f"It has {passed} tokens, and this model on NPU takes at most {limit}: its prompt "
        "length (MAX_PROMPT_LEN) is fixed when it is compiled. Shorten the conversation, or "
        "use a GPU or CPU artifact of this model, whose prompt length is not fixed."
    )
    assert not report.message.startswith("The prompt"), "the title already says so"
    assert "jenkins" not in report.message
    assert report.context["prompt_tokens"] == passed
    assert report.context["max_prompt_tokens"] == limit
    assert report.raw_excerpt == raw, "the runtime's own words stay for the log"
    runtime.close()


def test_a_prompt_too_long_names_its_tools_and_says_to_offer_fewer(artifact, settings, genai):
    genai.generate_error = RuntimeError(NPU_PROMPT_LIMIT_ERROR)
    runtime = make_runtime(artifact, settings, device="NPU")
    history = ChatHistory(tools=[WEATHER_TOOL] * 30)
    history.add_user("Hello")

    with pytest.raises(GenerationError) as excinfo:
        runtime.generate(history, GenerationSettings())
    message = excinfo.value.diagnostic.message
    assert message.startswith("It has 6423 tokens, counting the 30 offered tools, and this ")
    assert "Shorten the conversation or offer fewer tools, or use a GPU or CPU" in message
    runtime.close()


def test_a_prompt_too_long_names_its_images_too(artifact, settings, genai, tensors):
    genai.vlm_accepts_history = True
    genai.generate_error = RuntimeError(
        "VLM pipeline on NPU may only process input embeddings up to 2048 tokens. 2100 is passed."
    )
    runtime = make_runtime(artifact, settings, device="NPU", pipeline=PipelineKind.VLM)
    history = _asking_with_images(_image(), _image())
    history.tools = [WEATHER_TOOL]

    with pytest.raises(GenerationError) as excinfo:
        runtime.generate(history, GenerationSettings())
    message = excinfo.value.diagnostic.message
    assert message.startswith("It has 2100 tokens, counting the offered tool and the 2 images,")
    assert "Shorten the conversation, offer fewer tools or send fewer images, or use" in message
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


def test_the_conversation_boundary_is_part_of_the_domain_interface(artifact, settings, genai):
    """``begin_conversation`` reaches callers through ``ChatRuntime``, not through GenAI.

    ``start_chat``/``finish_chat`` are deprecated and stay inside the adapter
    (SPEC.md section 6.4); what the domain gets is a statement that the next
    history opens a new conversation. Conversation-boundary behaviour itself is
    covered in ``test_vlm_conversation.py``.
    """
    runtime = open_runtime(artifact, settings=settings, pipeline=PipelineKind.VLM)
    assert isinstance(runtime, ChatRuntime)
    assert not hasattr(runtime, "start_chat")
    assert not hasattr(runtime, "finish_chat")
    runtime.begin_conversation()
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


# --- the thinking variable is decided from the artifact's own chat template ------------
#
# The registry does not have to name the variable. Unless a profile states one,
# the runtime reads the artifact's template and applies a candidate only when the
# template actually references it — which is how a family nobody has written a
# rule for still gets its reasoning suppressed.

TEMPLATE_WITH_FLAG = (
    "{%- if add_generation_prompt %}\n"
    "    {%- if enable_thinking is defined and enable_thinking is false %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- endif %}\n"
    "{%- endif %}\n"
)
TEMPLATE_WITHOUT_FLAG = "{%- if add_generation_prompt %}{{- 'assistant' }}{%- endif %}\n"


def write_template(artifact: Path, template: str, *, name: str = "chat_template.jinja") -> None:
    """Give an artifact directory a chat template."""
    (artifact / name).write_text(template, encoding="utf-8")


def test_the_runtime_applies_the_variable_its_template_names(artifact, settings, genai):
    write_template(artifact, TEMPLATE_WITH_FLAG)
    runtime = make_runtime(artifact, settings, chat_profile=ChatProfile())
    assert runtime.thinking_variable == "enable_thinking"

    history = ChatHistory()
    history.add_user("Hello")
    runtime.generate(history, GenerationSettings())

    (args, _kwargs) = genai.pipelines[0].generate_calls[0]
    assert args[0].extra_context == {"enable_thinking": False}
    assert runtime.extra_context_error is None
    runtime.close()


def test_the_runtime_applies_nothing_when_the_template_names_nothing(artifact, settings, genai):
    write_template(artifact, TEMPLATE_WITHOUT_FLAG)
    runtime = make_runtime(artifact, settings, chat_profile=ChatProfile())
    assert runtime.thinking_variable is None

    history = ChatHistory()
    history.add_user("Hello")
    runtime.generate(history, GenerationSettings())

    (args, _kwargs) = genai.pipelines[0].generate_calls[0]
    assert args[0].extra_context is None
    runtime.close()


def test_an_explicit_profile_vetoes_detection(artifact, settings, genai):
    write_template(artifact, TEMPLATE_WITH_FLAG)
    profile = ChatProfile(supports_thinking_flag=False)
    runtime = make_runtime(artifact, settings, chat_profile=profile)
    assert runtime.thinking_variable is None

    history = ChatHistory()
    history.add_user("Hello")
    runtime.generate(history, GenerationSettings())
    (args, _kwargs) = genai.pipelines[0].generate_calls[0]
    assert args[0].extra_context is None
    runtime.close()


def test_an_explicit_profile_applies_without_template_evidence(artifact, settings, genai):
    write_template(artifact, TEMPLATE_WITHOUT_FLAG)
    profile = ChatProfile(supports_thinking_flag=True, thinking_template_variable="reasoning")
    runtime = make_runtime(artifact, settings, chat_profile=profile, thinking=True)
    assert runtime.thinking_variable == "reasoning"

    history = ChatHistory()
    history.add_user("Hello")
    runtime.generate(history, GenerationSettings())
    (args, _kwargs) = genai.pipelines[0].generate_calls[0]
    assert args[0].extra_context == {"reasoning": True}
    runtime.close()


def test_the_template_is_read_but_never_written(artifact, settings, genai):
    """SPEC 18.4: suppressing reasoning must not edit the model's own template."""
    write_template(artifact, TEMPLATE_WITH_FLAG)
    path = artifact / "chat_template.jinja"
    before = path.read_bytes()
    runtime = make_runtime(artifact, settings, chat_profile=ChatProfile())
    history = ChatHistory()
    history.add_user("Hello")
    runtime.generate(history, GenerationSettings())
    assert path.read_bytes() == before
    runtime.close()


def test_a_tokenizer_config_template_is_evidence_too(artifact, settings, genai):
    (artifact / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": TEMPLATE_WITH_FLAG}), encoding="utf-8"
    )
    runtime = make_runtime(artifact, settings, chat_profile=ChatProfile())
    assert runtime.thinking_variable == "enable_thinking"
    runtime.close()


def test_the_generation_result_defaults_report_nothing_hidden(artifact, settings, genai):
    """The adapter does not sanitise; the flags start false and the service sets them."""
    runtime = make_runtime(artifact, settings)
    history = ChatHistory()
    history.add_user("Hello")
    result = runtime.generate(history, GenerationSettings())
    assert result.reasoning_hidden is False
    assert result.reasoning_unterminated is False
    runtime.close()


# --- whether generation begins inside a reasoning block --------------------------------
#
# A template that opens ``<think>`` in the prompt leaves the model's reply with
# no opener of its own, which the chat service's sanitiser cannot recognise. The
# runtime read the template and knows what it could push into it, so it is the
# layer that decides — and it fails safe. The user-visible consequences are
# pinned in ``test_reasoning_leak.py``.

#: An opener emitted unconditionally, guarded by no variable at all.
TEMPLATE_ALWAYS_OPENS = "{%- if add_generation_prompt %}{{- '<think>\\n' }}{%- endif %}\n"


def test_an_opener_in_the_template_source_is_the_evidence():
    assert template_opens_reasoning(TEMPLATE_ALWAYS_OPENS) is True
    assert template_opens_reasoning(TEMPLATE_WITH_FLAG) is True
    assert template_opens_reasoning(TEMPLATE_WITHOUT_FLAG) is False
    assert template_opens_reasoning(None) is False


def test_a_profile_with_no_wrappers_has_no_opener_to_look_for():
    """Nothing to recognise means nothing to wait for, so nothing is assumed."""
    profile = ChatProfile(reasoning_wrappers=[])
    assert resolve_starts_in_reasoning(profile, template=TEMPLATE_ALWAYS_OPENS) is False


def test_the_runtime_starts_outside_the_block_when_it_pushed_the_variable_false(
    artifact, settings, genai
):
    write_template(artifact, TEMPLATE_WITH_FLAG)
    runtime = make_runtime(artifact, settings, chat_profile=ChatProfile())
    assert runtime.thinking_variable == "enable_thinking"
    assert runtime.starts_in_reasoning is False
    runtime.close()


def test_the_runtime_starts_inside_the_block_when_thinking_is_on(artifact, settings, genai):
    write_template(artifact, TEMPLATE_WITH_FLAG)
    runtime = make_runtime(artifact, settings, chat_profile=ChatProfile(), thinking=True)
    assert runtime.starts_in_reasoning is True
    runtime.close()


def test_an_unconditionally_opening_template_is_assumed_to_leave_the_block_open(
    artifact, settings, genai
):
    """The DeepSeek-R1-distill shape: no variable can switch the opener off."""
    write_template(artifact, TEMPLATE_ALWAYS_OPENS)
    runtime = make_runtime(artifact, settings, chat_profile=ChatProfile())
    assert runtime.thinking_variable is None
    assert runtime.starts_in_reasoning is True
    runtime.close()


def test_a_template_with_no_opener_never_starts_inside_a_block(artifact, settings, genai):
    write_template(artifact, TEMPLATE_WITHOUT_FLAG)
    runtime = make_runtime(artifact, settings, chat_profile=ChatProfile())
    assert runtime.starts_in_reasoning is False
    runtime.close()


def test_an_artifact_with_no_template_at_all_starts_outside_a_block(artifact, settings, genai):
    runtime = make_runtime(artifact, settings, chat_profile=ChatProfile())
    assert runtime.starts_in_reasoning is False
    runtime.close()


# --- properties are typed at the GenAI boundary --------------------------------------


def test_integer_pipeline_properties_are_passed_as_integers() -> None:
    """GenAI rejects "1024" for MAX_PROMPT_LEN ("Type mismatch: expected ... int")."""
    from yonga.adapters.runtime_genai import pipeline_kwargs

    converted = pipeline_kwargs({"MAX_PROMPT_LEN": "1024", "MIN_RESPONSE_LEN": " 128 "})

    assert converted == {"MAX_PROMPT_LEN": 1024, "MIN_RESPONSE_LEN": 128}


def test_other_pipeline_properties_pass_through_unchanged() -> None:
    from yonga.adapters.runtime_genai import pipeline_kwargs

    given = {"ATTENTION_BACKEND": "SDPA", "CACHE_DIR": "/cache/dir"}

    assert pipeline_kwargs(given) == given


def test_an_unparsable_integer_property_is_left_for_genai_to_reject() -> None:
    """yonga does not guess; GenAI reports the mismatch in its own words."""
    from yonga.adapters.runtime_genai import pipeline_kwargs

    assert pipeline_kwargs({"MAX_PROMPT_LEN": "lots"}) == {"MAX_PROMPT_LEN": "lots"}


# --- stateless VLM and streaming ---------------------------------------------------------


def test_vlm_sends_the_whole_history_when_the_pipeline_accepts_one(artifact, settings, genai):
    """GenAI 2026.4 takes a ChatHistory on the VLM pipeline too: no hidden session."""
    genai.vlm_accepts_history = True
    runtime = make_runtime(artifact, settings, pipeline=PipelineKind.VLM)
    pipeline = genai.pipelines[0]
    history = ChatHistory()
    history.add_system("You are terse.")
    history.add_user("First")
    history.add_assistant("hello there")
    history.add_user("Second")

    runtime.generate(history, GenerationSettings())

    (args, kwargs) = pipeline.generate_calls[0]
    assert [m["content"] for m in args[0].messages] == [
        "You are terse.",
        "First",
        "hello there",
        "Second",
    ]
    assert "generation_config" in kwargs
    assert pipeline.chat_events == [], "the deprecated lifecycle is never touched"


def test_tools_reach_the_genai_history_on_a_user_turn(artifact, settings, genai):
    runtime = make_runtime(artifact, settings)
    history = ChatHistory(
        messages=[ChatMessage(role="user", content="Weather in Ankara?")], tools=[WEATHER_TOOL]
    )

    runtime.generate(history, GenerationSettings())

    (args, _kwargs) = genai.pipelines[0].generate_calls[0]
    assert isinstance(args[0], FakeChatHistory)
    assert args[0].tools == [WEATHER_TOOL]


def test_tools_and_tool_turns_reach_the_chat_template(artifact, settings, genai):
    runtime = make_runtime(artifact, settings)

    runtime.generate(tool_conversation(), GenerationSettings())

    (call,) = genai.pipelines[0].tokenizer.template_calls
    sent = call["history"]
    assert sent.tools == [WEATHER_TOOL]
    assert [m["role"] for m in sent.messages] == ["user", "assistant", "tool"]
    assert sent.messages[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Ankara"}


# --- histories that do not end with a user turn ----------------------------------------
#
# OpenVINO GenAI 2026.4's stateful LLM pipeline (ATTENTION_BACKEND=SDPA, and the
# NPU pipeline) refuses generate(ChatHistory) unless the history ends with a user
# turn: "Chat doesn't support switching between input types". The follow-up of a
# tool round trip ends with the tool result, so it is rendered with the model's
# own chat template and handed over as tokens instead.


def assistant_last_conversation() -> ChatHistory:
    return ChatHistory(
        messages=[
            ChatMessage(role="user", content="Say hello."),
            ChatMessage(role="assistant", content="Hel"),
        ]
    )


@pytest.mark.parametrize(
    "conversation", [tool_conversation, assistant_last_conversation], ids=["tool", "assistant"]
)
def test_a_history_not_ending_with_a_user_turn_is_rendered_once(
    artifact, settings, genai, conversation
):
    profile = ChatProfile(supports_thinking_flag=True, default_thinking=False)
    runtime = make_runtime(artifact, settings, chat_profile=profile)
    history = conversation()
    pipeline = genai.pipelines[0]

    result = runtime.generate(history, GenerationSettings(max_new_tokens=16))

    # The model's own template renders the same GenAI history the history route
    # would have sent, tools and template variables included, opening the
    # assistant's turn...
    (call,) = pipeline.tokenizer.template_calls
    assert call["add_generation_prompt"] is True
    sent = call["history"]
    assert isinstance(sent, FakeChatHistory)
    assert sent.messages == history.to_payload()
    assert sent.tools == (list(history.tools) or None)
    assert sent.extra_context == {"enable_thinking": False}
    # ...and the pipeline gets that prompt as tokens, never a history or a string
    # it could template a second time, with no extra BOS.
    ((args, kwargs),) = pipeline.generate_calls
    (inputs, config) = args
    assert isinstance(inputs, FakeTokenizedInputs)
    assert inputs.prompt == FakeTokenizer.RENDERED
    assert inputs.add_special_tokens is False
    assert config.apply_chat_template is False
    assert "streamer" not in kwargs
    # The chat the previous history call left open is closed first, and the
    # state this call leaves behind is reset before the next history call.
    assert pipeline.events == ["finish_chat", "generate", "finish_chat"]
    # The tokens come back decoded, with metrics and finish reason intact.
    assert pipeline.tokenizer.decode_calls == [(ENCODED_TOKENS, True)]
    assert result.text == "hello there"
    assert result.input_tokens == 9
    assert result.output_tokens == 16
    assert result.stop_reason == "length"


def test_a_history_ending_with_a_user_turn_keeps_the_history_route(artifact, settings, genai):
    runtime = make_runtime(artifact, settings)
    history = tool_conversation()
    history.messages.append(ChatMessage(role="assistant", content="It is 21 C."))
    history.messages.append(ChatMessage(role="user", content="And tomorrow?"))
    pipeline = genai.pipelines[0]

    runtime.generate(history, GenerationSettings())

    ((args, _kwargs),) = pipeline.generate_calls
    assert isinstance(args[0], FakeChatHistory)
    assert args[0].tools == [WEATHER_TOOL]
    assert not hasattr(args[1], "apply_chat_template")
    assert pipeline.tokenizer.template_calls == []
    assert pipeline.events == ["generate"], "no chat is closed on the history route"


def test_a_rendered_turn_streams_like_any_other(artifact, settings, genai):
    genai.stream_chunks = ["It is ", "21 C."]
    genai.result = FakeResults(text="It is 21 C.", reason="STOP", perf_metrics=FakePerfMetrics())
    runtime = make_runtime(artifact, settings)
    seen: list[str] = []

    result = runtime.generate_stream(tool_conversation(), GenerationSettings(), seen.append)

    assert seen == ["It is ", "21 C."]
    assert result.text == "It is 21 C."
    assert result.stop_reason == "stop"
    (args, kwargs) = genai.pipelines[0].generate_calls[0]
    assert isinstance(args[0], FakeTokenizedInputs)
    assert "streamer" in kwargs


def test_the_route_follows_each_history_turn_by_turn(artifact, settings, genai):
    """A tool round trip on one pipeline: history, rendered, then history again."""
    runtime = make_runtime(artifact, settings)
    pipeline = genai.pipelines[0]
    first = ChatHistory(messages=tool_conversation().messages[:1], tools=[WEATHER_TOOL])

    runtime.generate(first, GenerationSettings())
    runtime.generate(tool_conversation(), GenerationSettings())
    runtime.generate(first, GenerationSettings())

    kinds = [type(call[0][0]) for call in pipeline.generate_calls]
    assert kinds == [FakeChatHistory, FakeTokenizedInputs, FakeChatHistory]
    assert pipeline.events == ["generate", "finish_chat", "generate", "finish_chat", "generate"]


def test_closing_the_chat_does_not_leak_genai_s_deprecation_warning(artifact, settings, genai):
    genai.finish_chat_warns = True
    runtime = make_runtime(artifact, settings)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runtime.generate(tool_conversation(), GenerationSettings())

    assert genai.pipelines[0].events == ["finish_chat", "generate", "finish_chat"]
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


def test_a_pipeline_without_finish_chat_still_renders(artifact, settings, genai):
    """A later GenAI may drop the deprecated call; then there is no chat to close."""
    runtime = make_runtime(artifact, settings)
    pipeline = genai.pipelines[0]
    pipeline.finish_chat = None

    result = runtime.generate(tool_conversation(), GenerationSettings())

    assert result.text == "hello there"
    assert pipeline.events == ["generate"]


def test_a_rendered_turn_failure_is_a_generation_failure(artifact, settings, genai):
    genai.generate_error = RuntimeError("Exception from src/plugins/intel_gpu: device lost")
    runtime = make_runtime(artifact, settings, device="GPU")

    with pytest.raises(GenerationError) as caught:
        runtime.generate(tool_conversation(), GenerationSettings())
    assert caught.value.diagnostic.code == "runtime.genai.generation_failed"
    assert "device lost" in caught.value.diagnostic.message
    # The state is reset even so, ready for the next request.
    assert genai.pipelines[0].events == ["finish_chat", "generate", "finish_chat"]


def test_a_failing_reset_after_a_rendered_turn_does_not_hide_the_answer(artifact, settings, genai):
    runtime = make_runtime(artifact, settings)
    pipeline = genai.pipelines[0]
    resets: list[str] = []

    def finish_chat() -> None:
        resets.append("finish_chat")
        if len(resets) == 2:
            raise RuntimeError("reset failed")

    pipeline.finish_chat = finish_chat

    result = runtime.generate(tool_conversation(), GenerationSettings())

    assert result.text == "hello there"
    assert resets == ["finish_chat", "finish_chat"]


def test_a_vlm_tool_turn_keeps_the_history_route(artifact, settings, genai):
    """GenAI 2026.4's VLMPipeline takes a tool-last history as it is (verified on GPU)."""
    genai.vlm_accepts_history = True
    runtime = make_runtime(artifact, settings, pipeline=PipelineKind.VLM)
    pipeline = genai.pipelines[0]

    runtime.generate(tool_conversation(), GenerationSettings())

    ((args, _kwargs),) = pipeline.generate_calls
    assert isinstance(args[0], FakeChatHistory)
    assert [m["role"] for m in args[0].messages] == ["user", "assistant", "tool"]
    assert pipeline.tokenizer.template_calls == []
    assert pipeline.events == ["generate"]


def test_a_history_without_tools_sets_none(artifact, settings, genai):
    runtime = make_runtime(artifact, settings)
    history = ChatHistory()
    history.add_user("Hello")

    runtime.generate(history, GenerationSettings())

    (args, _kwargs) = genai.pipelines[0].generate_calls[0]
    assert args[0].tools is None


def test_the_legacy_vlm_lifecycle_refuses_tools_rather_than_drop_them(artifact, settings, genai):
    runtime = make_runtime(artifact, settings, pipeline=PipelineKind.VLM)

    with pytest.raises(GenerationError) as caught:
        runtime.generate(tool_conversation(), GenerationSettings())
    assert caught.value.diagnostic.code == "runtime.genai.tools_unsupported"
    assert genai.pipelines[0].chat_events == []


def test_vlm_falls_back_to_the_lifecycle_once_and_remembers(artifact, settings, genai):
    runtime = make_runtime(artifact, settings, pipeline=PipelineKind.VLM)
    pipeline = genai.pipelines[0]
    history = ChatHistory()
    history.add_user("First")

    runtime.generate(history, GenerationSettings())
    history.add_assistant("hello there")
    history.add_user("Second")
    runtime.generate(history, GenerationSettings())

    assert [call[0][0] for call in pipeline.generate_calls] == ["First", "Second"]
    assert runtime._vlm_accepts_history is False


def test_generate_stream_hands_every_chunk_over(artifact, settings, genai):
    genai.stream_chunks = ["hel", "lo ", "there"]
    runtime = make_runtime(artifact, settings)
    seen: list[str] = []
    history = ChatHistory()
    history.add_user("Hi")

    result = runtime.generate_stream(history, GenerationSettings(), seen.append)

    assert seen == ["hel", "lo ", "there"]
    assert result.text == "hello there"


def test_a_callback_asking_to_stop_stops_generation(artifact, settings, genai):
    genai.stream_chunks = ["one", "two", "three"]
    runtime = make_runtime(artifact, settings)
    seen: list[str] = []
    history = ChatHistory()
    history.add_user("Hi")

    def stop_after_first(chunk: str) -> bool:
        seen.append(chunk)
        return True

    runtime.generate_stream(history, GenerationSettings(), stop_after_first)

    assert seen == ["one"]


def test_a_raising_callback_is_re_raised_not_reported_as_a_model_failure(artifact, settings, genai):
    genai.stream_chunks = ["one", "two"]
    runtime = make_runtime(artifact, settings)
    history = ChatHistory()
    history.add_user("Hi")

    def broken(chunk: str) -> None:
        raise BrokenPipeError("client went away")

    with pytest.raises(BrokenPipeError):
        runtime.generate_stream(history, GenerationSettings(), broken)


def test_plain_generate_passes_no_streamer(artifact, settings, genai):
    runtime = make_runtime(artifact, settings)
    history = ChatHistory()
    history.add_user("Hi")

    runtime.generate(history, GenerationSettings())

    assert "streamer" not in genai.pipelines[0].generate_calls[0][1]
    assert isinstance(runtime, runtime_module.StreamingChatRuntime)


# --- images on the VLM path -----------------------------------------------------------


@pytest.fixture
def tensors(monkeypatch: pytest.MonkeyPatch) -> list[ChatImage]:
    """Stand in for ``openvino.Tensor``: each image becomes a tagged tuple."""
    made: list[ChatImage] = []

    def tensor(image: ChatImage) -> tuple[str, int, int]:
        made.append(image)
        return ("tensor", image.width, image.height)

    monkeypatch.setattr(runtime_module, "_image_tensor", tensor)
    return made


def _image(width: int = 2, height: int = 1) -> ChatImage:
    return ChatImage(width=width, height=height, pixels=b"\x01" * (width * height * 3))


def _asking_with_images(*images: ChatImage, text: str = "What is it?") -> ChatHistory:
    history = ChatHistory()
    history.add_system("You are terse.")
    history.messages.append(ChatMessage(role="user", content=text, images=list(images)))
    return history


def test_vlm_images_go_to_the_pipeline_with_the_history(artifact, settings, genai, tensors):
    genai.vlm_accepts_history = True
    runtime = make_runtime(artifact, settings, pipeline=PipelineKind.VLM)
    pipeline = genai.pipelines[0]

    runtime.generate(_asking_with_images(_image(2, 1), _image(1, 3)), GenerationSettings())

    ((args, kwargs),) = pipeline.generate_calls
    assert kwargs["images"] == [("tensor", 2, 1), ("tensor", 1, 3)]
    assert [m["content"] for m in args[0].messages] == ["You are terse.", "What is it?"]
    assert all(set(m) == {"role", "content"} for m in args[0].messages)


def test_a_vlm_turn_without_images_passes_no_images_argument(artifact, settings, genai, tensors):
    genai.vlm_accepts_history = True
    runtime = make_runtime(artifact, settings, pipeline=PipelineKind.VLM)
    history = ChatHistory()
    history.add_user("Hi")

    runtime.generate(history, GenerationSettings())

    ((_, kwargs),) = genai.pipelines[0].generate_calls
    assert "images" not in kwargs
    assert tensors == []


def test_the_legacy_vlm_path_passes_the_images_too(artifact, settings, genai, tensors):
    runtime = make_runtime(artifact, settings, pipeline=PipelineKind.VLM)

    runtime.generate(_asking_with_images(_image(), text=""), GenerationSettings())

    ((args, kwargs),) = genai.pipelines[0].generate_calls
    assert args == ("",), "an image alone is something to answer"
    assert kwargs["images"] == [("tensor", 2, 1)]


def test_genai_media_tags_typed_by_a_client_are_removed(artifact, settings, genai, tensors):
    """GenAI reads <ov_genai_image_N> anywhere in the text; words must not claim an image."""
    genai.vlm_accepts_history = True
    runtime = make_runtime(artifact, settings, pipeline=PipelineKind.VLM)
    history = ChatHistory()
    history.add_user("Look <ov_genai_image_0> and <ov_genai_video_12>here")

    runtime.generate(history, GenerationSettings())

    ((args, _),) = genai.pipelines[0].generate_calls
    assert args[0].messages[0]["content"] == "Look  and here"


@pytest.mark.parametrize(
    "typed",
    [
        "<ov_genai_<ov_genai_image_0>image_0>",
        "<ov_genai_<ov_genai_video_1>video_2>",
        "<ov_<ov_genai_<ov_genai_image_0>image_1>genai_image_2>",
    ],
)
def test_a_media_tag_hidden_inside_another_is_removed_too(
    artifact, settings, genai, tensors, typed
):
    """Removing the inner tag must not leave a new one behind."""
    genai.vlm_accepts_history = True
    runtime = make_runtime(artifact, settings, pipeline=PipelineKind.VLM)
    history = ChatHistory()
    history.add_user(f"Look {typed}here")

    runtime.generate(history, GenerationSettings())

    ((args, _),) = genai.pipelines[0].generate_calls
    assert args[0].messages[0]["content"] == "Look here"


def test_the_legacy_vlm_prompt_loses_media_tags_too(artifact, settings, genai):
    runtime = make_runtime(artifact, settings, pipeline=PipelineKind.VLM)
    history = ChatHistory()
    history.add_user("<ov_genai_image_3>Describe")

    runtime.generate(history, GenerationSettings())

    assert genai.pipelines[0].generate_calls[0][0] == ("Describe",)


def test_an_llm_keeps_media_tag_text_as_it_is(artifact, settings, genai):
    runtime = make_runtime(artifact, settings)
    history = ChatHistory()
    history.add_user("What does <ov_genai_image_0> mean?")

    runtime.generate(history, GenerationSettings())

    ((args, _),) = genai.pipelines[0].generate_calls
    assert args[0].messages[0]["content"] == "What does <ov_genai_image_0> mean?"


def test_images_on_an_llm_pipeline_are_refused(artifact, settings, genai, tensors):
    runtime = make_runtime(artifact, settings)

    with pytest.raises(GenerationError) as refused:
        runtime.generate(_asking_with_images(_image()), GenerationSettings())

    assert refused.value.diagnostic.code == "runtime.genai.images_unsupported"
    assert genai.pipelines[0].generate_calls == []


def test_an_image_on_an_earlier_turn_is_refused(artifact, settings, genai, tensors):
    """GenAI 2026.4 cannot bind it there; the adapter does not move it on its own."""
    genai.vlm_accepts_history = True
    runtime = make_runtime(artifact, settings, pipeline=PipelineKind.VLM)
    history = _asking_with_images(_image())
    history.add_assistant("A dot.")
    history.add_user("Sure?")

    with pytest.raises(GenerationError) as refused:
        runtime.generate(history, GenerationSettings())

    assert refused.value.diagnostic.code == "runtime.genai.images_not_last_turn"
    assert refused.value.diagnostic.category is FailureCategory.CONFIGURATION_ERROR
    assert genai.pipelines[0].generate_calls == []


def test_an_image_becomes_an_rgb_tensor() -> None:
    """The real conversion, on host memory only: no device is involved."""
    pytest.importorskip("numpy")
    pytest.importorskip("openvino")
    image = ChatImage(width=3, height=2, pixels=bytes(range(18)))

    tensor = runtime_module._image_tensor(image)

    assert list(tensor.shape) == [2, 3, 3]
    assert tensor.data.dtype.name == "uint8"
    assert tensor.data[1, 2].tolist() == [15, 16, 17]
