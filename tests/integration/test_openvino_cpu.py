"""Integration tests that run the real OpenVINO stack on the CPU.

These are marked ``openvino`` and are deselected by the default ``addopts``; run
them with ``pytest -m openvino tests/integration``.

They work on a copy of a prepared artifact inside ``tmp_path``. The source
fixtures are never opened for writing: validation and probing must be read-only,
and the repaired/unrepaired pair is the evidence for that.

There is deliberately no module-level executable code here: OpenVINO can start a
multiprocessing forkserver whose children re-import the main module
(docs/MODEL_PIPELINE.md section 10).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from tests.support import find_broken_artifact
from yonga.adapters.openvino_probe import OpenVINOProbe, openvino_available
from yonga.adapters.runtime_genai import (
    ChatHistory,
    ChatProfile,
    GenAIChatRuntime,
    open_runtime,
)
from yonga.config.settings import Settings
from yonga.domain.artifacts import ProbeStatus, SubmodelRole
from yonga.domain.benchmarks import GenerationSettings
from yonga.domain.models import PipelineKind

pytestmark = pytest.mark.openvino

#: The vendored artifact: a tiny-random-Llama int4 export written by a real
#: ``optimum-cli export openvino`` run, kept byte-for-byte as the exporter
#: produced it -- including the ``precision="STRiNG"`` defect in the tokenizer
#: pair. ``YONGA_TEST_BROKEN_ARTIFACT`` points the suite at a larger export.
#: Resolved by ``tests/conftest.py``: the ``YONGA_TEST_BROKEN_ARTIFACT`` override,
#: then the generated cache (``python tools/make_test_artifact.py``). ``None``
#: when the artifact has not been generated on this host, in which case the
#: tests below skip.
UNREPAIRED_FIXTURE = find_broken_artifact()

#: A complete, working artifact. When it is not supplied, the ``artifact``
#: fixture derives one from the vendored export, because the two differ by
#: exactly the four bytes this substitution rewrites (two per tokenizer IR).
#: It is spelled out here rather than obtained by calling ``yonga.repairs`` so
#: that "a known-good artifact reads and compiles" stays independent evidence
#: rather than a restatement of what the repair implementation did.
_REPAIRED_ENV = os.environ.get("YONGA_TEST_ARTIFACT")
REPAIRED_FIXTURE = Path(_REPAIRED_ENV) if _REPAIRED_ENV else None

#: The exact upstream error produced by an unrepaired tokenizer IR.
STRING_PRECISION_SIGNATURE = "Unsupported element type: STRiNG"

#: The tokenizer IR files carrying the defect, and the exact textual fix.
_TOKENIZER_IRS = ("openvino_tokenizer.xml", "openvino_detokenizer.xml")
_BAD_PRECISION = 'precision="STRiNG"'
_GOOD_PRECISION = 'precision="STRING"'


def _repair_in_place(directory: Path) -> None:
    """Apply the known two-byte-per-occurrence port precision fix to a copy."""
    for name in _TOKENIZER_IRS:
        path = directory / name
        text = path.read_text(encoding="utf-8")
        assert _BAD_PRECISION in text, f"{name} does not carry the defect"
        path.write_text(text.replace(_BAD_PRECISION, _GOOD_PRECISION), encoding="utf-8")


def _require(fixture: Path | None) -> Path:
    if fixture is None or not fixture.is_dir():
        pytest.skip(
            "the tiny OpenVINO test artifact is not present; "
            "generate it with: python tools/make_test_artifact.py"
        )
    return fixture


@pytest.fixture(autouse=True)
def _require_openvino():
    available, error = openvino_available()
    if not available:
        pytest.skip(f"openvino is not importable: {error}")


@pytest.fixture
def artifact(tmp_path):
    """A writable copy of the repaired artifact; the source is never touched."""
    destination = tmp_path / "artifact"
    if REPAIRED_FIXTURE is not None:
        shutil.copytree(_require(REPAIRED_FIXTURE), destination)
        return destination
    shutil.copytree(_require(UNREPAIRED_FIXTURE), destination)
    _repair_in_place(destination)
    return destination


@pytest.fixture
def broken_artifact(tmp_path):
    """A writable copy of the artifact as the exporter produced it."""
    destination = tmp_path / "broken"
    shutil.copytree(_require(UNREPAIRED_FIXTURE), destination)
    return destination


@pytest.fixture
def settings(tmp_path):
    return Settings(
        cache_dir=tmp_path / "cache",
        data_dir=tmp_path / "data",
        config_dir=tmp_path / "config",
    )


# --- devices --------------------------------------------------------------------------


def test_cpu_is_always_enumerated():
    probe = OpenVINOProbe()
    devices = probe.available_devices()
    assert "CPU" in devices

    inventory = {device.name: device for device in probe.devices()}
    cpu = inventory["CPU"]
    assert cpu.full_name != "unknown"
    assert cpu.properties["FULL_DEVICE_NAME"] == cpu.full_name
    # Every property is readable or recorded as "unknown"; none may raise.
    assert all(isinstance(value, str) for value in cpu.properties.values())


# --- IR reading -----------------------------------------------------------------------


def test_read_ir_describes_the_language_model(artifact):
    description = OpenVINOProbe().read_ir(artifact / "openvino_model.xml")

    assert description.readable is True
    assert description.read_error is None
    assert description.role is SubmodelRole.LANGUAGE
    assert description.friendly_name
    assert description.bin_path == artifact / "openvino_model.bin"
    assert description.bytes_total > 0

    names = {port.name for port in description.inputs}
    assert {"input_ids", "attention_mask"} <= names
    assert description.outputs[0].name == "logits"
    assert description.outputs[0].shape.startswith("[")


def test_inventory_covers_every_submodel(artifact):
    roles = {
        description.path.name: description.role
        for description in OpenVINOProbe().inventory_irs(artifact)
    }
    assert roles["openvino_model.xml"] is SubmodelRole.LANGUAGE
    assert roles["openvino_tokenizer.xml"] is SubmodelRole.TOKENIZER
    assert roles["openvino_detokenizer.xml"] is SubmodelRole.DETOKENIZER


# --- compiling ------------------------------------------------------------------------


def test_cpu_compile_succeeds(artifact):
    result = OpenVINOProbe().compile(artifact / "openvino_model.xml", "CPU")

    assert result.status is ProbeStatus.OK, result.message
    assert result.device == "CPU"
    assert result.role is SubmodelRole.LANGUAGE
    assert result.duration_seconds is not None
    assert result.duration_seconds > 0.0
    assert result.diagnostic is None


def test_compile_of_a_broken_ir_fails_without_raising(broken_artifact):
    result = OpenVINOProbe().compile(broken_artifact / "openvino_tokenizer.xml", "CPU")

    assert result.status is ProbeStatus.FAILED
    assert STRING_PRECISION_SIGNATURE in (result.message or "")
    # The adapter never classifies; the registry does.
    assert result.diagnostic is None


# --- tokenizers -----------------------------------------------------------------------


def test_tokenizer_pair_reads_and_compiles_on_cpu(artifact):
    result = OpenVINOProbe().read_tokenizer(artifact)

    assert result.status is ProbeStatus.OK, result.message
    assert result.device == "CPU"
    assert result.properties["tokenizer_read"] == "ok"
    assert result.properties["tokenizer_compile"] == "ok"
    assert result.properties["detokenizer_read"] == "ok"
    assert result.properties["detokenizer_compile"] == "ok"
    assert result.properties["openvino_version"].startswith("2026.")


def test_unrepaired_tokenizer_fails_with_the_string_precision_defect(broken_artifact):
    result = OpenVINOProbe().read_tokenizer(broken_artifact)

    assert result.status is ProbeStatus.FAILED
    assert result.properties["tokenizer_read"] == "failed"
    assert result.properties["detokenizer_read"] == "failed"
    assert STRING_PRECISION_SIGNATURE in (result.message or "")
    # The source fixture must still be exactly as the exporter wrote it.
    assert 'precision="STRiNG"' in (
        (_require(UNREPAIRED_FIXTURE) / "openvino_tokenizer.xml").read_text(encoding="utf-8")
    )


# --- generation -----------------------------------------------------------------------


def test_generation_produces_text_and_metrics(artifact, settings):
    runtime = open_runtime(
        artifact,
        device="CPU",
        pipeline=PipelineKind.LLM,
        chat_profile=ChatProfile(supports_thinking_flag=True, default_thinking=False),
        settings=settings,
    )
    assert isinstance(runtime, GenAIChatRuntime)
    try:
        assert runtime.properties["CACHE_DIR"].startswith(str(settings.paths.compile_cache_dir))

        history = ChatHistory()
        history.add_user("Hello")
        result = runtime.generate(history, GenerationSettings(max_new_tokens=16))

        assert isinstance(result.text, str)
        assert result.display_text == result.text
        assert result.reasoning_text is None
        assert result.input_tokens is not None
        assert result.input_tokens > 0
        assert result.output_tokens == 16
        assert result.ttft_ms is not None
        assert result.ttft_ms > 0.0
        assert result.throughput_tps is not None
        assert result.tpot_ms is not None
        assert result.wall_seconds > 0.0
        assert result.stop_reason == "length"

        # A second turn must reuse the same pipeline, not reload the model.
        history.add_assistant(result.text)
        history.add_user("And again")
        again = runtime.generate(history, GenerationSettings(max_new_tokens=8))
        assert again.output_tokens == 8
        assert again.input_tokens is not None
        assert again.input_tokens > result.input_tokens
    finally:
        runtime.close()

    assert any(settings.paths.compile_cache_dir.iterdir())


def test_generation_on_a_broken_artifact_raises_a_typed_error(broken_artifact, settings):
    from yonga.domain.diagnostics import CompileError

    with pytest.raises(CompileError) as excinfo:
        open_runtime(broken_artifact, device="CPU", settings=settings)
    assert STRING_PRECISION_SIGNATURE in excinfo.value.diagnostic.message
    assert excinfo.value.diagnostic.code == "runtime.genai.load_failed"
