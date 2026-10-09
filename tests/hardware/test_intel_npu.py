"""Real Intel NPU: access, deferral of raw compiles, and pipeline generation.

These guard the finding that made NPU preparation possible at all. On the
reference host a raw ``Core.compile_model`` of any stateful LLM fails on the NPU
with "to_shape was called on a dynamic shape", while OpenVINO GenAI's
``LLMPipeline`` reshapes to a static budget and generates. yonga used to treat
the raw failure as the verdict, so it could never prepare an NPU artifact.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.hardware._support import (
    isolated_settings,
    npu_artifact,
    require_device,
    tiny_artifact,
)

if TYPE_CHECKING:
    from yonga.application.probe import ProbeService

pytestmark = [pytest.mark.openvino, pytest.mark.intel_npu]


@pytest.fixture(autouse=True)
def _npu() -> None:
    require_device("NPU")


def _probe_service(tmp_path: Path) -> ProbeService:
    from yonga.adapters.openvino_probe import OpenVINOProbe
    from yonga.adapters.smoke_child import ChildProcessSmokeTester, smoke_child_runner
    from yonga.application.probe import ProbeService
    from yonga.application.wiring import GenAISmokeTester
    from yonga.registry import RegistryResolver, load_builtin_registry

    return ProbeService(
        OpenVINOProbe(),
        RegistryResolver(load_builtin_registry()),
        None,
        isolated_settings(tmp_path),
        smoke_tester=GenAISmokeTester(),
        isolated_smoke_tester=ChildProcessSmokeTester(smoke_child_runner()),
    )


def test_the_npu_node_is_accessible() -> None:
    """``docs/TESTING.md`` 6: /dev/accel is accessible and OpenVINO lists the NPU."""
    from yonga.adapters.system_linux import LinuxSystemProbe

    npu = LinuxSystemProbe().inventory().npu

    assert npu is not None
    assert npu.accessible is True


def test_a_raw_npu_compile_of_an_llm_is_deferred_not_a_verdict(tmp_path: Path) -> None:
    from yonga.adapters.openvino_probe import OpenVINOProbe
    from yonga.domain.artifacts import ProbeStatus
    from yonga.registry import RegistryResolver, load_builtin_registry

    artifact = tiny_artifact(tmp_path)
    result = OpenVINOProbe().compile(artifact / "openvino_model.xml", "NPU")

    assert result.status is ProbeStatus.FAILED
    assert "to_shape was called on a dynamic shape" in (result.message or "")
    classified = RegistryResolver(load_builtin_registry()).classify(
        raw_error=result.message or "", device="NPU"
    )
    assert classified.defer_to_pipeline is True
    assert classified.environment_broken is False


def test_an_npu_pipeline_failure_is_typed_never_an_internal_error(tmp_path: Path) -> None:
    """The tiny random Llama is rejected by the NPU pipeline: a real, fast failure."""
    from yonga.cli import _probe_exit
    from yonga.domain.artifacts import ProbeStatus, SubmodelRole
    from yonga.domain.diagnostics import ExitCode

    outcome = _probe_service(tmp_path).probe_detailed(tiny_artifact(tmp_path), device="npu")
    report = outcome.reports[0]

    assert report.status is ProbeStatus.FAILED
    language = [r for r in report.submodels if r.role is SubmodelRole.LANGUAGE]
    assert language and language[0].status is ProbeStatus.DEFERRED
    assert report.smoke_test is not None and report.smoke_test.status is ProbeStatus.FAILED
    assert report.diagnostic is not None
    assert _probe_exit(outcome) is not ExitCode.INTERNAL


def test_an_npu_llm_generates_with_manifest_shaped_properties(tmp_path: Path) -> None:
    """String-valued properties, as a manifest stores them, reach GenAI typed.

    GenAI rejects the *string* "1024" for MAX_PROMPT_LEN ("Type mismatch:
    expected types: int or int64_t"), and every property yonga records is a
    string because it round-trips through YAML and JSON. The values are passed
    as literal strings here, independent of the registry, so this fails if the
    runtime adapter ever stops converting them.
    """
    from yonga.adapters.runtime_genai import ChatHistory, open_runtime
    from yonga.domain.benchmarks import GenerationSettings
    from yonga.domain.models import PipelineKind

    runtime = open_runtime(
        npu_artifact(tmp_path),
        device="NPU",
        pipeline=PipelineKind.LLM,
        properties={"MAX_PROMPT_LEN": "1024", "MIN_RESPONSE_LEN": "128"},
        settings=isolated_settings(tmp_path),
    )
    try:
        history = ChatHistory()
        history.add_user("What is the capital of France? One word.")
        result = runtime.generate(history, GenerationSettings(max_new_tokens=16, do_sample=False))
    finally:
        runtime.close()

    assert result.output_tokens is not None and result.output_tokens > 0


def test_the_probe_service_passes_an_npu_llm_through_the_pipeline(tmp_path: Path) -> None:
    """The regression guard: yonga's own NPU probe now passes a real LLM."""
    from yonga.domain.artifacts import ProbeStatus, SubmodelRole

    report = _probe_service(tmp_path).probe(npu_artifact(tmp_path), device="npu")[0]

    assert report.status is ProbeStatus.OK
    language = [r for r in report.submodels if r.role is SubmodelRole.LANGUAGE]
    assert language and language[0].status is ProbeStatus.DEFERRED
    assert report.smoke_test is not None and report.smoke_test.status is ProbeStatus.OK


def test_an_npu_llm_streams_and_keeps_its_history_across_turns(tmp_path: Path) -> None:
    """Chunks add up to the answer, and a second turn reuses the NPU pipeline.

    ``serve`` and the chat TUI both stream, and the NPU pipeline is the one
    whose static prompt budget a second turn could overrun; two short turns on
    one runtime prove neither breaks the other.
    """
    from yonga.adapters.runtime_genai import ChatHistory, StreamingChatRuntime, open_runtime
    from yonga.domain.benchmarks import GenerationSettings
    from yonga.domain.models import PipelineKind

    runtime = open_runtime(
        npu_artifact(tmp_path),
        device="NPU",
        pipeline=PipelineKind.LLM,
        properties={"MAX_PROMPT_LEN": "1024", "MIN_RESPONSE_LEN": "128"},
        settings=isolated_settings(tmp_path),
    )
    config = GenerationSettings(max_new_tokens=16, do_sample=False)
    chunks: list[str] = []

    def collect(text: str) -> None:
        chunks.append(text)

    try:
        assert isinstance(runtime, StreamingChatRuntime)
        history = ChatHistory()
        history.add_user("What is the capital of France? One word.")
        first = runtime.generate_stream(history, config, collect)
        history.add_assistant(first.text)
        history.add_user("And of Italy? One word.")
        second = runtime.generate(history, config)
    finally:
        runtime.close()

    assert chunks and "".join(chunks) == first.text
    assert second.output_tokens is not None and second.output_tokens > 0


def test_a_quick_npu_probe_stops_at_the_raw_compile(tmp_path: Path) -> None:
    """``probe --quick`` never builds the pipeline, so it answers in seconds.

    The raw compile of an LLM defers on the NPU, so the honest quick answer is
    inconclusive -- exit 2, not a pass and not a compiler verdict. Should a
    future NPU compiler accept the raw graph, ok is the honest answer instead.
    """
    from yonga.cli import _probe_exit
    from yonga.domain.artifacts import ProbeStatus, SubmodelRole
    from yonga.domain.diagnostics import ExitCode

    artifact = npu_artifact(tmp_path)
    started = time.monotonic()
    outcome = _probe_service(tmp_path).probe_detailed(artifact, device="npu", quick=True)
    elapsed = time.monotonic() - started
    report = outcome.reports[0]

    assert report.status in (ProbeStatus.INCONCLUSIVE, ProbeStatus.OK)
    assert report.smoke_test is None, "--quick must not build a pipeline"
    assert elapsed < 120, f"a quick probe took {elapsed:.0f} s"
    if report.status is ProbeStatus.INCONCLUSIVE:
        language = [r for r in report.submodels if r.role is SubmodelRole.LANGUAGE]
        assert language and language[0].status is ProbeStatus.DEFERRED
        assert _probe_exit(outcome) is ExitCode.WARNINGS


def test_an_npu_pipeline_build_past_its_timeout_is_stopped_and_inconclusive(
    tmp_path: Path,
) -> None:
    """``--timeout 1``: no NPU pipeline builds and generates within one second.

    The child is stopped at the limit, and the device is inconclusive -- not
    failed, and not classified as a compiler rejection.
    """
    from yonga.cli import _probe_exit
    from yonga.domain.artifacts import ProbeStatus
    from yonga.domain.diagnostics import ExitCode

    artifact = npu_artifact(tmp_path)
    started = time.monotonic()
    outcome = _probe_service(tmp_path).probe_detailed(artifact, device="npu", timeout=1)
    elapsed = time.monotonic() - started
    report = outcome.reports[0]

    assert report.status is ProbeStatus.INCONCLUSIVE
    assert report.smoke_test is not None
    assert report.smoke_test.status is ProbeStatus.INCONCLUSIVE
    assert report.diagnostic is not None
    assert report.diagnostic.code == "probe.inconclusive.timeout"
    assert _probe_exit(outcome) is ExitCode.WARNINGS
    # The stop escalates SIGINT -> SIGTERM -> SIGKILL, five seconds apart; the
    # child must die on the first. Its measured run includes the stop, so one
    # that survived SIGINT would take over 1 + 5 s.
    child_seconds = report.smoke_test.duration_seconds
    assert child_seconds is not None
    assert child_seconds < 1 + 4, f"the child took {child_seconds:.1f} s to stop"
    assert elapsed < 60, f"the whole probe took {elapsed:.0f} s"


def test_a_bounded_npu_probe_passes_an_llm_through_the_child_process(tmp_path: Path) -> None:
    """With room to finish, the child builds the pipeline and the device passes."""
    from yonga.domain.artifacts import ProbeStatus

    report = _probe_service(tmp_path).probe(npu_artifact(tmp_path), device="npu", timeout=600)[0]

    assert report.status is ProbeStatus.OK
    assert report.smoke_test is not None and report.smoke_test.status is ProbeStatus.OK
