"""Real Intel GPU: generation, compile-cache reuse, and the probe service.

``docs/TESTING.md`` section 6 lists the GPU host acceptance test: the device is
seen, a small compatible model compiles and generates, metrics are non-empty,
and the compile cache is reused on a second load. That is what this file does,
on the tiny generated artifact so it runs in seconds.
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
