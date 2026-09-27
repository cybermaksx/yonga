"""Real Intel GPU: generation, compile-cache reuse, and the probe service.

``docs/TESTING.md`` section 6 lists the GPU host acceptance test: the device is
seen, a small compatible model compiles and generates, metrics are non-empty,
and the compile cache is reused on a second load. That is what this file does,
on the tiny generated artifact so it runs in seconds, together with the paths
built on top of generation: streaming, ``serve`` and ``eval``'s logits.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.hardware._support import isolated_settings, require_device, tiny_artifact

if TYPE_CHECKING:
    from yonga.adapters.runtime_genai import GenerationResult

pytestmark = [pytest.mark.openvino, pytest.mark.intel_gpu]


@pytest.fixture(autouse=True)
def _gpu() -> None:
    require_device("GPU")


def _generate(artifact: Path, tmp_path: Path) -> GenerationResult:
    from yonga.adapters.runtime_genai import ChatHistory, open_runtime
    from yonga.domain.benchmarks import GenerationSettings
    from yonga.domain.models import PipelineKind

    runtime = open_runtime(
        artifact, device="GPU", pipeline=PipelineKind.LLM, settings=isolated_settings(tmp_path)
    )
    try:
        history = ChatHistory()
        history.add_user("Hello")
        return runtime.generate(history, GenerationSettings(max_new_tokens=8, do_sample=False))
    finally:
        runtime.close()


def test_the_gpu_generates_with_metrics(tmp_path: Path) -> None:
    result = _generate(tiny_artifact(tmp_path), tmp_path)

    assert result.output_tokens is not None and result.output_tokens > 0
    assert result.ttft_ms is not None
    assert result.throughput_tps is not None


def test_the_compile_cache_is_written_once_and_reused(tmp_path: Path) -> None:
    artifact = tiny_artifact(tmp_path)
    cache_root = isolated_settings(tmp_path).paths.compile_cache_dir

    _generate(artifact, tmp_path)
    first = sorted(path for path in cache_root.rglob("*") if path.is_file())
    _generate(artifact, tmp_path)
    second = sorted(path for path in cache_root.rglob("*") if path.is_file())

    assert first, "the first GPU load wrote no compile-cache blob"
    assert second == first, "the second load did not reuse the first load's cache"


def test_the_probe_service_passes_on_the_gpu_with_the_tokenizer_on_cpu(tmp_path: Path) -> None:
    from yonga.adapters.openvino_probe import OpenVINOProbe
    from yonga.application.probe import ProbeService
    from yonga.domain.artifacts import ProbeStatus, SubmodelRole
    from yonga.registry import RegistryResolver, load_builtin_registry

    artifact = tiny_artifact(tmp_path)
    service = ProbeService(
        OpenVINOProbe(),
        RegistryResolver(load_builtin_registry()),
        None,
        isolated_settings(tmp_path),
    )

    report = service.probe(artifact, device="gpu")[0]

    assert report.device == "GPU"
    assert report.status is ProbeStatus.OK
    tokenizer = [r for r in report.submodels if r.role is SubmodelRole.TOKENIZER]
    assert tokenizer and tokenizer[0].device == "CPU"


def test_streamed_chunks_add_up_to_the_whole_answer(tmp_path: Path) -> None:
    from yonga.adapters.runtime_genai import ChatHistory, StreamingChatRuntime, open_runtime
    from yonga.domain.benchmarks import GenerationSettings
    from yonga.domain.models import PipelineKind

    runtime = open_runtime(
        tiny_artifact(tmp_path),
        device="GPU",
        pipeline=PipelineKind.LLM,
        settings=isolated_settings(tmp_path),
    )
    chunks: list[str] = []

    def collect(text: str) -> None:
        chunks.append(text)

    try:
        assert isinstance(runtime, StreamingChatRuntime)
        history = ChatHistory()
        history.add_user("Hello")
        result = runtime.generate_stream(
            history,
            GenerationSettings(max_new_tokens=12, do_sample=False),
            collect,
        )
    finally:
        runtime.close()

    assert chunks, "the streamer was never called"
    assert "".join(chunks) == result.text


def test_serve_answers_plain_and_streamed_requests_on_the_gpu(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    import json

    from fastapi.testclient import TestClient

    from yonga.api.app import create_app
    from yonga.application.chat import ChatOptions
    from yonga.application.serve import ModelServer
    from yonga.application.wiring import Wiring

    artifact = tiny_artifact(tmp_path)
    service = Wiring(isolated_settings(tmp_path)).chat_service()
    server = ModelServer(
        service.open_artifact(artifact, options=ChatOptions(device="gpu")), name="tiny"
    )
    server.load()
    try:
        client = TestClient(create_app(server))
        ask = {
            "model": "tiny",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8,
            "temperature": 0,
        }

        plain = client.post("/v1/chat/completions", json=ask)
        streamed = client.post("/v1/chat/completions", json={**ask, "stream": True})
    finally:
        server.close()

    assert plain.status_code == 200, plain.text
    body = plain.json()
    assert body["model"] == "tiny"
    assert body["choices"][0]["finish_reason"] in {"stop", "length"}
    assert body["usage"]["completion_tokens"] > 0

    assert streamed.status_code == 200, streamed.text
    lines = [line[6:] for line in streamed.text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(line) for line in lines[:-1]]
    assert chunks[-1]["choices"][0]["finish_reason"] in {"stop", "length"}
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert text == body["choices"][0]["message"]["content"], "greedy stream and reply differ"


def test_gpu_logits_agree_with_the_cpu_reference(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    from yonga.adapters.openvino_logits import LogitsModel
    from yonga.application.evaluation import _kl_rows

    artifact = tiny_artifact(tmp_path)
    text = "The quick brown fox jumps over the lazy dog."
    cpu = LogitsModel(artifact, device="CPU")
    try:
        ids = cpu.tokenize(text)
        reference = cpu.log_probs(ids)
    finally:
        cpu.close()
    gpu = LogitsModel(artifact, device="GPU")
    try:
        assert np.array_equal(gpu.tokenize(text), ids)
        candidate = gpu.log_probs(ids)
    finally:
        gpu.close()

    assert candidate.shape == reference.shape == (len(ids), reference.shape[1])
    assert np.allclose(np.exp(candidate).sum(axis=1), 1.0, atol=1e-3)
    kl = _kl_rows(reference, candidate)
    assert float(kl.mean()) < 0.01, f"GPU drifts from CPU f32: KL {float(kl.mean()):.4f}"
