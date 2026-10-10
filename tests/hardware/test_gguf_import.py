"""Real GGUF import: OpenVINO GenAI's reader, then the devices that run the result.

The import is the one path where OpenVINO GenAI, not optimum-intel, writes the
IR, so the unit tests' fakes say nothing about whether the output loads. This
file imports a small cached GGUF with the real child process, repairs it the
way ``prepare`` does, and then asks each device what it makes of it: the GPU
generates, and the NPU refuses in the way the registry's issue rule names.

The GGUF comes from the local Hugging Face cache and is never downloaded here;
``YONGA_TEST_GGUF`` points at any other supported file.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.hardware._support import isolated_settings, repaired_copy, require_device

pytestmark = [pytest.mark.openvino, pytest.mark.slow]

GGUF_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
GGUF_FILE = "qwen2.5-0.5b-instruct-q4_0.gguf"
GGUF_MISSING = (
    f"no GGUF test file is cached; fetch one with: hf download {GGUF_REPO} {GGUF_FILE} "
    "(or set YONGA_TEST_GGUF)"
)


def _cached_gguf() -> Path | None:
    override = os.environ.get("YONGA_TEST_GGUF")
    if override:
        return Path(override) if Path(override).is_file() else None
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(GGUF_REPO, GGUF_FILE)
    return Path(found) if isinstance(found, str) else None


@pytest.fixture(scope="module")
def imported(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """The cached GGUF imported once by the real child process, then repaired."""
    from yonga.adapters.gguf_import import IMPORT_OUTPUTS, GGUFImporter
    from yonga.adapters.subprocess import SubprocessRunner
    from yonga.application.jobs import open_job

    gguf = _cached_gguf()
    if gguf is None:
        pytest.skip(GGUF_MISSING)
    root = tmp_path_factory.mktemp("gguf")
    raw = root / "raw"
    with open_job(settings=isolated_settings(root), requested_model=GGUF_REPO) as ctx:
        result = GGUFImporter(SubprocessRunner()).import_file(gguf, raw, ctx)

    assert result.success, result.diagnostic
    assert all((raw / name).is_file() for name in IMPORT_OUTPUTS)
    assert not any(path.suffix == ".gguf" for path in raw.iterdir()), (
        "the working copy was left behind"
    )
    assert gguf.is_file(), "the import touched the cached source"
    yield repaired_copy(raw, root / "repaired", missing=GGUF_MISSING)


@pytest.mark.intel_gpu
def test_the_imported_model_generates_on_the_gpu(imported: Path, tmp_path: Path) -> None:
    require_device("GPU")
    from yonga.adapters.runtime_genai import ChatHistory, open_runtime
    from yonga.domain.benchmarks import GenerationSettings
    from yonga.domain.models import PipelineKind

    runtime = open_runtime(
        imported, device="GPU", pipeline=PipelineKind.LLM, settings=isolated_settings(tmp_path)
    )
    try:
        history = ChatHistory()
        history.add_user("What is the capital of France? Answer in one word.")
        result = runtime.generate(history, GenerationSettings(max_new_tokens=16, do_sample=False))
    finally:
        runtime.close()

    assert "paris" in result.text.lower(), result.text


@pytest.mark.intel_npu
def test_the_npu_refusal_is_named_by_the_registry(imported: Path, tmp_path: Path) -> None:
    require_device("NPU")
    from yonga.adapters.runtime_genai import ChatHistory, open_runtime
    from yonga.domain.benchmarks import GenerationSettings
    from yonga.domain.diagnostics import YongaError
    from yonga.domain.models import PipelineKind
    from yonga.registry import RegistryResolver, load_builtin_registry

    try:
        runtime = open_runtime(
            imported,
            device="NPU",
            pipeline=PipelineKind.LLM,
            properties={"MAX_PROMPT_LEN": "1024", "MIN_RESPONSE_LEN": "128"},
            settings=isolated_settings(tmp_path),
        )
        try:
            history = ChatHistory()
            history.add_user("Hello")
            runtime.generate(history, GenerationSettings(max_new_tokens=4, do_sample=False))
        finally:
            runtime.close()
    except YongaError as error:
        raw = error.diagnostic.raw_excerpt or error.diagnostic.message
    except Exception as error:
        raw = str(error)
    else:
        pytest.fail(
            "the NPU now runs a GGUF import: retire the npu-static-reshape-broadcast rule "
            "and record the evidence"
        )

    found = RegistryResolver(load_builtin_registry()).classify(
        raw_error=raw, device="NPU", stage="bench"
    )
    assert found is not None, raw[-2000:]
    assert found.code == "npu.pipeline.static_reshape_failed"
    assert found.matched_rule_id == "npu-static-reshape-broadcast"
    assert not found.environment_broken, "the NPU itself is fine"
    assert [device.upper() for device in found.fallback_devices] == ["GPU", "CPU"]
