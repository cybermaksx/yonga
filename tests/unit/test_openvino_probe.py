"""Unit tests for the OpenVINO probe adapter.

Every test in this file runs with OpenVINO *stubbed out*: the lazy import hooks
are monkeypatched so the suite is meaningful on a machine with no Intel
accelerator and no openvino wheel installed. The real stack is exercised in
``tests/integration/test_openvino_cpu.py``.
"""

from __future__ import annotations

import json
import types
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import pytest

from yonga.adapters import RuntimeProbe
from yonga.adapters import openvino_probe as probe_module
from yonga.adapters.openvino_probe import (
    DETOKENIZER_IR_NAME,
    TOKENIZER_IR_NAME,
    UNKNOWN,
    OpenVINOProbe,
    artifact_fingerprint,
    compile_cache_dir,
    compile_cache_key,
    compiler_in_plugin,
    npu_compiler_version,
    npu_driver_version,
    openvino_available,
)
from yonga.config.paths import default_paths
from yonga.domain.artifacts import ProbeStatus, SubmodelRole
from yonga.domain.diagnostics import FailureCategory
from yonga.domain.hardware import DeviceKind

# --- fakes ----------------------------------------------------------------------------


class FakeCompilerType(Enum):
    """Stand-in for the ``CompilerType`` enum OpenVINO returns for the NPU."""

    PREFER_PLUGIN = 2

    def __str__(self) -> str:
        return f"CompilerType.{self.name}"


class FakePort:
    """A model port that can be made to misbehave the way real ones do."""

    def __init__(
        self,
        name: str,
        element_type: Any,
        shape: Any,
        *,
        raise_any_name: bool = False,
        names: set[str] | None = None,
    ) -> None:
        self._name = name
        self._element_type = element_type
        self._shape = shape
        self._raise_any_name = raise_any_name
        self._names = names if names is not None else {name}

    def get_any_name(self):
        if self._raise_any_name:
            raise RuntimeError("port has no unique name")
        return self._name

    def get_names(self):
        return set(self._names)

    def get_element_type(self):
        return self._element_type

    def get_partial_shape(self):
        return self._shape


class FakeModel:
    """A minimal ``openvino.Model`` stand-in."""

    def __init__(
        self,
        friendly_name: str = "Model0",
        inputs: Sequence[Any] = (),
        outputs: Sequence[Any] = (),
    ) -> None:
        self._friendly_name = friendly_name
        self.inputs = list(inputs)
        self.outputs = list(outputs)

    def get_friendly_name(self):
        return self._friendly_name


class FakeCore:
    """A scriptable ``openvino.Core``."""

    def __init__(
        self,
        *,
        devices: Sequence[str] = ("CPU",),
        properties: dict[str, dict[str, Any]] | None = None,
        models: dict[str, Any] | None = None,
        compile_errors: dict[str, BaseException] | None = None,
    ) -> None:
        self.available_devices = list(devices)
        self._properties = properties or {}
        self._models = models or {}
        self._compile_errors = compile_errors or {}
        self.compiled: list[tuple[str, dict[str, Any]]] = []

    def get_property(self, device, name):
        table = self._properties.get(device, {})
        if name not in table:
            raise RuntimeError(f"unsupported property {name}")
        value = table[name]
        if isinstance(value, Exception):
            raise value
        return value

    def read_model(self, path):
        entry = self._models.get(Path(path).name)
        if entry is None:
            raise RuntimeError(f"no model registered for {path}")
        if isinstance(entry, Exception):
            raise entry
        return entry

    def compile_model(self, model, device, config=None):
        error = self._compile_errors.get(device)
        if error is not None:
            raise error
        self.compiled.append((device, dict(config or {})))
        return object()


def install_openvino(
    monkeypatch: pytest.MonkeyPatch,
    core: Any,
    *,
    version: str = "2026.4.0-22959-releases/2026/4",
) -> types.SimpleNamespace:
    """Install a fake ``openvino`` module and return it."""
    module = types.SimpleNamespace(
        Core=lambda: core,
        get_version=lambda: version,
        __version__=version,
    )
    monkeypatch.setattr(probe_module, "_import_openvino", lambda: module)
    return module


def install_tokenizers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    version: str = "2026.4.0.0-736",
    compatible: bool = True,
) -> types.SimpleNamespace:
    """Install a fake ``openvino_tokenizers`` module and return it."""
    module = types.SimpleNamespace(
        __version__=version,
        is_openvino_tokenizers_compatible=compatible,
    )
    monkeypatch.setattr(probe_module, "_import_openvino_tokenizers", lambda: module)
    return module


def break_openvino(
    monkeypatch: pytest.MonkeyPatch, message: str = "No module named 'openvino'"
) -> None:
    """Make every OpenVINO import fail, as on a host without the runtime."""

    def _raise():
        raise ImportError(message)

    monkeypatch.setattr(probe_module, "_import_openvino", _raise)
    monkeypatch.setattr(probe_module, "_import_openvino_tokenizers", _raise)


def write_ir(directory: Path, stem: str, *, xml_bytes=b"<net/>", bin_bytes=b"\x00\x01") -> Path:
    """Write an ``<stem>.xml``/``<stem>.bin`` pair and return the XML path."""
    directory.mkdir(parents=True, exist_ok=True)
    xml = directory / f"{stem}.xml"
    xml.write_bytes(xml_bytes)
    (directory / f"{stem}.bin").write_bytes(bin_bytes)
    return xml


@pytest.fixture(autouse=True)
def _no_real_openvino(monkeypatch):
    """Fail loudly if a test forgets to stub the lazy imports."""

    def _raise():
        raise ImportError("openvino is not installed (unit test default)")

    monkeypatch.setattr(probe_module, "_import_openvino", _raise)
    monkeypatch.setattr(probe_module, "_import_openvino_tokenizers", _raise)


# --- protocol conformance -------------------------------------------------------------


def test_probe_implements_runtime_probe_protocol():
    assert isinstance(OpenVINOProbe(), RuntimeProbe)


# --- openvino missing -----------------------------------------------------------------


def test_openvino_available_reports_import_error(monkeypatch):
    break_openvino(monkeypatch, "libopenvino.so: cannot open shared object file")
    available, error = openvino_available()
    assert available is False
    assert error is not None
    assert "libopenvino.so" in error


def test_available_devices_is_empty_without_openvino(monkeypatch):
    break_openvino(monkeypatch)
    assert OpenVINOProbe().available_devices() == []
    assert OpenVINOProbe().devices() == []
    assert OpenVINOProbe().device_properties("CPU") == {}


def test_read_ir_without_openvino_is_unreadable(monkeypatch, tmp_path):
    break_openvino(monkeypatch)
    xml = write_ir(tmp_path, "openvino_model")
    description = OpenVINOProbe().read_ir(xml)
    assert description.readable is False
    assert description.read_error is not None
    assert "ImportError" in description.read_error
    assert description.role is SubmodelRole.LANGUAGE
    assert description.bin_path == tmp_path / "openvino_model.bin"
    assert description.bytes_total == 8


def test_compile_without_openvino_returns_typed_diagnostic(monkeypatch, tmp_path):
    break_openvino(monkeypatch)
    xml = write_ir(tmp_path, "openvino_model")
    result = OpenVINOProbe().compile(xml, "NPU")
    assert result.status is ProbeStatus.FAILED
    assert result.device == "NPU"
    assert result.diagnostic is not None
    assert result.diagnostic.code == "runtime.openvino.missing"
    assert result.diagnostic.category is FailureCategory.PACKAGE_ABI_MISMATCH
    assert result.diagnostic.environment_broken is True


def test_read_tokenizer_without_tokenizers_package(monkeypatch, tmp_path):
    break_openvino(monkeypatch)
    result = OpenVINOProbe().read_tokenizer(tmp_path)
    assert result.status is ProbeStatus.FAILED
    assert result.device == "CPU"
    assert result.diagnostic is not None
    assert result.diagnostic.code == "runtime.openvino_tokenizers.missing"


# --- devices and defensive property reads ---------------------------------------------


def test_device_properties_record_unknown_for_failing_reads(monkeypatch):
    core = FakeCore(
        devices=("NPU",),
        properties={
            "NPU": {
                "SUPPORTED_PROPERTIES": {
                    "SUPPORTED_PROPERTIES": "RO",
                    "FULL_DEVICE_NAME": "RO",
                    "DEVICE_ARCHITECTURE": "RO",
                    "COMPATIBILITY_CHECK": "RO",
                    "CACHE_ENCRYPTION_CALLBACKS": "WO",
                    "OPTIMIZATION_CAPABILITIES": "RO",
                    "NPU_COMPILER_TYPE": "RW",
                },
                "FULL_DEVICE_NAME": "Intel(R) AI Boost",
                "DEVICE_ARCHITECTURE": 4000,
                # Verified on the reference host: reading these raises.
                "COMPATIBILITY_CHECK": TypeError("Unsupported property type"),
                "CACHE_ENCRYPTION_CALLBACKS": TypeError("write-only property"),
                "OPTIMIZATION_CAPABILITIES": ["FP16", "INT8", "EXPORT_IMPORT"],
                "NPU_COMPILER_TYPE": FakeCompilerType.PREFER_PLUGIN,
            }
        },
    )
    install_openvino(monkeypatch, core)
    properties = OpenVINOProbe().device_properties("NPU")

    assert properties["COMPATIBILITY_CHECK"] == UNKNOWN
    assert properties["CACHE_ENCRYPTION_CALLBACKS"] == UNKNOWN
    assert properties["FULL_DEVICE_NAME"] == "Intel(R) AI Boost"
    # Non-string values are stringified, never dropped.
    assert properties["DEVICE_ARCHITECTURE"] == "4000"
    assert properties["OPTIMIZATION_CAPABILITIES"] == "FP16, INT8, EXPORT_IMPORT"
    assert properties["NPU_COMPILER_TYPE"] == "CompilerType.PREFER_PLUGIN"
    # SUPPORTED_PROPERTIES is self-referential and is not echoed back.
    assert "SUPPORTED_PROPERTIES" not in properties


def test_device_properties_fall_back_when_supported_properties_raises(monkeypatch):
    core = FakeCore(
        devices=("GPU",),
        properties={"GPU": {"FULL_DEVICE_NAME": "Intel(R) Arc(TM) Graphics (iGPU)"}},
    )
    install_openvino(monkeypatch, core)
    properties = OpenVINOProbe().device_properties("GPU")
    assert properties["FULL_DEVICE_NAME"] == "Intel(R) Arc(TM) Graphics (iGPU)"
    assert properties["DEVICE_ARCHITECTURE"] == UNKNOWN


def test_devices_builds_the_inventory(monkeypatch):
    core = FakeCore(
        devices=("CPU", "GPU"),
        properties={
            "CPU": {
                "SUPPORTED_PROPERTIES": {"FULL_DEVICE_NAME": "RO", "DEVICE_ARCHITECTURE": "RO"},
                "FULL_DEVICE_NAME": "Intel(R) Core(TM) Ultra 7 268V",
                "DEVICE_ARCHITECTURE": "intel64",
            },
            "GPU": {
                "SUPPORTED_PROPERTIES": {"FULL_DEVICE_NAME": "RO"},
                "FULL_DEVICE_NAME": "Intel(R) Arc(TM) Graphics (iGPU)",
            },
        },
    )
    install_openvino(monkeypatch, core)
    devices = OpenVINOProbe().devices()

    assert [device.name for device in devices] == ["CPU", "GPU"]
    assert devices[0].kind is DeviceKind.CPU
    assert devices[0].architecture == "intel64"
    assert devices[1].kind is DeviceKind.GPU
    assert devices[1].architecture is None
    assert devices[1].full_name == "Intel(R) Arc(TM) Graphics (iGPU)"


# --- doctor helpers -------------------------------------------------------------------


def test_npu_helpers_read_the_property_mapping():
    properties = {
        "NPU_DRIVER_VERSION": "1789049548",
        "NPU_COMPILER_VERSION": "524291",
        "NPU_COMPILER_TYPE": "CompilerType.PREFER_PLUGIN",
    }
    assert npu_driver_version(properties) == "1789049548"
    assert npu_compiler_version(properties) == "524291"
    assert compiler_in_plugin(properties) is True


def test_npu_helpers_treat_unknown_and_missing_as_none():
    assert npu_driver_version({}) is None
    assert npu_compiler_version({"NPU_COMPILER_VERSION": UNKNOWN}) is None
    assert compiler_in_plugin({}) is None
    assert compiler_in_plugin({"NPU_COMPILER_TYPE": UNKNOWN}) is None
    assert compiler_in_plugin({"NPU_COMPILER_TYPE": "CompilerType.MLIR"}) is False


# --- IR reading -----------------------------------------------------------------------


def test_read_ir_describes_ports_and_role(monkeypatch, tmp_path):
    model = FakeModel(
        friendly_name="Model0",
        inputs=[FakePort("input_ids", "<Type: 'int64_t'>", "[?,?]")],
        outputs=[FakePort("logits", "<Type: 'float32'>", "[?,?,32000]")],
    )
    core = FakeCore(models={"openvino_language_model.xml": model})
    install_openvino(monkeypatch, core)
    xml = write_ir(tmp_path, "openvino_language_model")

    description = OpenVINOProbe().read_ir(xml)
    assert description.readable is True
    assert description.role is SubmodelRole.LANGUAGE
    assert description.friendly_name == "Model0"
    assert [(p.name, p.element_type, p.shape) for p in description.inputs] == [
        ("input_ids", "<Type: 'int64_t'>", "[?,?]")
    ]
    assert description.outputs[0].name == "logits"


def test_read_ir_survives_a_port_without_a_unique_name(monkeypatch, tmp_path):
    model = FakeModel(
        inputs=[FakePort("x", "<Type: 'f32'>", "[1]", raise_any_name=True, names={"alias"})],
        outputs=[FakePort("y", "<Type: 'f32'>", "[1]", raise_any_name=True, names=set())],
    )
    core = FakeCore(models={"openvino_model.xml": model})
    install_openvino(monkeypatch, core)
    xml = write_ir(tmp_path, "openvino_model")

    description = OpenVINOProbe().read_ir(xml)
    assert description.inputs[0].name == "alias"
    assert description.outputs[0].name == UNKNOWN


def test_read_ir_reports_the_string_precision_defect(monkeypatch, tmp_path):
    failure = RuntimeError(
        "Exception from src/inference/src/cpp/core.cpp:84:\n"
        "Check 'is_valid_type_idx(type_idx)' failed at src/core/src/type/element_type.cpp:115:\n"
        "Unsupported element type: STRiNG"
    )
    core = FakeCore(models={"openvino_tokenizer.xml": failure})
    install_openvino(monkeypatch, core)
    install_tokenizers(monkeypatch)
    xml = write_ir(tmp_path, "openvino_tokenizer")

    description = OpenVINOProbe().read_ir(xml)
    assert description.readable is False
    assert description.role is SubmodelRole.TOKENIZER
    assert "Unsupported element type: STRiNG" in (description.read_error or "")


def test_inventory_irs_assigns_every_role(monkeypatch, tmp_path):
    models = {
        name: FakeModel(friendly_name=name)
        for name in (
            "openvino_language_model.xml",
            "openvino_text_embeddings_model.xml",
            "openvino_vision_embeddings_model.xml",
            "openvino_tokenizer.xml",
            "openvino_detokenizer.xml",
        )
    }
    install_openvino(monkeypatch, FakeCore(models=models))
    install_tokenizers(monkeypatch)
    for stem in models:
        write_ir(tmp_path, Path(stem).stem)

    descriptions = OpenVINOProbe().inventory_irs(tmp_path)
    roles = {description.path.name: description.role for description in descriptions}
    assert roles == {
        "openvino_language_model.xml": SubmodelRole.LANGUAGE,
        "openvino_text_embeddings_model.xml": SubmodelRole.TEXT_EMBEDDINGS,
        "openvino_vision_embeddings_model.xml": SubmodelRole.VISION_EMBEDDINGS,
        "openvino_tokenizer.xml": SubmodelRole.TOKENIZER,
        "openvino_detokenizer.xml": SubmodelRole.DETOKENIZER,
    }
    # Sorted by name so reports are reproducible.
    assert [d.path.name for d in descriptions] == sorted(roles)


def test_inventory_irs_on_a_missing_directory_is_empty(monkeypatch, tmp_path):
    install_openvino(monkeypatch, FakeCore())
    assert OpenVINOProbe().inventory_irs(tmp_path / "nope") == []


# --- compiling ------------------------------------------------------------------------


def test_compile_reports_success_with_properties(monkeypatch, tmp_path):
    core = FakeCore(models={"openvino_model.xml": FakeModel()})
    install_openvino(monkeypatch, core)
    xml = write_ir(tmp_path, "openvino_model")

    result = OpenVINOProbe().compile(
        xml, "GPU.0", {"CACHE_DIR": "/cache", "ATTENTION_BACKEND": "SDPA"}
    )
    assert result.status is ProbeStatus.OK
    assert result.ok is True
    assert result.device == "GPU"
    assert result.role is SubmodelRole.LANGUAGE
    assert result.duration_seconds is not None
    assert core.compiled == [("GPU.0", {"CACHE_DIR": "/cache", "ATTENTION_BACKEND": "SDPA"})]


def test_compile_preserves_the_raw_error_for_the_classifier(monkeypatch, tmp_path):
    raw = (
        "Exception from src/inference/src/cpp/core.cpp:117:\n"
        "Exception from src/plugins/intel_npu/src/plugin/src/plugin.cpp:552:\n"
        "[NPU_VCL] Compiler returned msg:\n"
        "Exception from src/core/src/partial_shape.cpp:261:\n"
        "to_shape was called on a dynamic shape."
    )
    core = FakeCore(
        models={"openvino_model.xml": FakeModel()},
        compile_errors={"NPU": RuntimeError(raw)},
    )
    install_openvino(monkeypatch, core)
    xml = write_ir(tmp_path, "openvino_model")

    result = OpenVINOProbe().compile(xml, "NPU")
    assert result.status is ProbeStatus.FAILED
    assert result.message is not None
    assert "to_shape was called on a dynamic shape" in result.message
    # Classification belongs to the registry, so no diagnostic is invented here.
    assert result.diagnostic is None


# --- tokenizers -----------------------------------------------------------------------


def _tokenizer_dir(tmp_path: Path) -> Path:
    write_ir(tmp_path, "openvino_tokenizer")
    write_ir(tmp_path, "openvino_detokenizer")
    return tmp_path


def test_read_tokenizer_reports_both_halves(monkeypatch, tmp_path):
    core = FakeCore(
        models={
            "openvino_tokenizer.xml": FakeModel("tokenizer"),
            "openvino_detokenizer.xml": FakeModel("detokenizer"),
        }
    )
    install_openvino(monkeypatch, core)
    install_tokenizers(monkeypatch)

    result = OpenVINOProbe().read_tokenizer(_tokenizer_dir(tmp_path))
    assert result.status is ProbeStatus.OK
    assert result.device == "CPU"
    assert result.properties["tokenizer_read"] == "ok"
    assert result.properties["tokenizer_compile"] == "ok"
    assert result.properties["detokenizer_read"] == "ok"
    assert result.properties["detokenizer_compile"] == "ok"
    # Tokenizers are CPU-hosted: nothing else may be probed.
    assert {device for device, _ in core.compiled} == {"CPU"}


def test_read_tokenizer_reports_which_half_failed(monkeypatch, tmp_path):
    core = FakeCore(
        models={
            "openvino_tokenizer.xml": RuntimeError("Unsupported element type: STRiNG"),
            "openvino_detokenizer.xml": FakeModel("detokenizer"),
        }
    )
    install_openvino(monkeypatch, core)
    install_tokenizers(monkeypatch)

    result = OpenVINOProbe().read_tokenizer(_tokenizer_dir(tmp_path))
    assert result.status is ProbeStatus.FAILED
    assert result.properties["tokenizer_read"] == "failed"
    assert result.properties["tokenizer_compile"] == "not_attempted"
    assert result.properties["detokenizer_read"] == "ok"
    assert result.properties["detokenizer_compile"] == "ok"
    assert "STRiNG" in (result.message or "")


def test_read_tokenizer_skips_when_no_tokenizer_is_present(monkeypatch, tmp_path):
    install_openvino(monkeypatch, FakeCore())
    install_tokenizers(monkeypatch)

    result = OpenVINOProbe().read_tokenizer(tmp_path)
    assert result.status is ProbeStatus.SKIPPED
    assert result.properties["tokenizer_read"] == "missing"
    assert TOKENIZER_IR_NAME in (result.message or "")
    assert DETOKENIZER_IR_NAME in (result.message or "")


def test_read_tokenizer_refuses_a_version_family_mismatch(monkeypatch, tmp_path):
    core = FakeCore(models={"openvino_tokenizer.xml": FakeModel("tokenizer")})
    install_openvino(monkeypatch, core, version="2026.4.0")
    install_tokenizers(monkeypatch, version="2025.1.0.0-1")

    result = OpenVINOProbe().read_tokenizer(_tokenizer_dir(tmp_path))
    assert result.status is ProbeStatus.FAILED
    assert result.diagnostic is not None
    assert result.diagnostic.code == "runtime.openvino_tokenizers.abi_mismatch"
    assert result.diagnostic.category is FailureCategory.PACKAGE_ABI_MISMATCH
    # Nothing was read: a mismatched extension must not be exercised.
    assert core.compiled == []


def test_read_tokenizer_honours_the_binary_compatibility_flag(monkeypatch, tmp_path):
    core = FakeCore(models={"openvino_tokenizer.xml": FakeModel("tokenizer")})
    install_openvino(monkeypatch, core, version="2026.4.0")
    install_tokenizers(monkeypatch, version="2026.4.0.0-736", compatible=False)

    result = OpenVINOProbe().read_tokenizer(_tokenizer_dir(tmp_path))
    assert result.status is ProbeStatus.FAILED
    assert result.diagnostic is not None
    assert result.diagnostic.code == "runtime.openvino_tokenizers.abi_mismatch"


# --- compile cache --------------------------------------------------------------------


def test_compile_cache_key_changes_with_the_openvino_version():
    older = compile_cache_key(
        artifact_hash="abc", device="GPU", openvino_version="2026.4.0", properties={}
    )
    newer = compile_cache_key(
        artifact_hash="abc", device="GPU", openvino_version="2026.5.0", properties={}
    )
    unknown = compile_cache_key(
        artifact_hash="abc", device="GPU", openvino_version=None, properties={}
    )
    assert older != newer
    assert unknown not in {older, newer}


def test_compile_cache_key_is_stable_and_property_sensitive():
    base = compile_cache_key(
        artifact_hash="abc", device="GPU", openvino_version="2026.4.0", properties={}
    )
    assert base == compile_cache_key(
        artifact_hash="abc", device="gpu.1", openvino_version="  2026.4.0 ", properties=None
    )
    # CACHE_DIR is derived from the key, so it must not feed back into it.
    assert base == compile_cache_key(
        artifact_hash="abc",
        device="GPU",
        openvino_version="2026.4.0",
        properties={"CACHE_DIR": "/somewhere"},
    )
    assert base != compile_cache_key(
        artifact_hash="abc",
        device="GPU",
        openvino_version="2026.4.0",
        properties={"ATTENTION_BACKEND": "SDPA"},
    )
    assert base != compile_cache_key(
        artifact_hash="def", device="GPU", openvino_version="2026.4.0", properties={}
    )
    assert base != compile_cache_key(
        artifact_hash="abc", device="NPU", openvino_version="2026.4.0", properties={}
    )


def test_compile_cache_dir_lives_under_the_managed_cache(tmp_path):
    paths = default_paths({"HOME": str(tmp_path)}, cache_dir=tmp_path / "cache")
    directory = compile_cache_dir(paths, "deadbeef")
    assert directory == tmp_path / "cache" / "compile" / "deadbeef"
    assert directory.is_dir()


def test_artifact_fingerprint_prefers_the_manifest_id(tmp_path):
    (tmp_path / ".yonga").mkdir()
    (tmp_path / ".yonga" / "manifest.json").write_text(
        json.dumps({"artifact_id": "qwen3_5--gpu-int4"}), encoding="utf-8"
    )
    assert artifact_fingerprint(tmp_path) == "qwen3_5--gpu-int4"


def test_artifact_fingerprint_falls_back_to_the_file_listing(tmp_path):
    write_ir(tmp_path, "openvino_model")
    first = artifact_fingerprint(tmp_path)
    assert first == artifact_fingerprint(tmp_path)
    (tmp_path / "openvino_model.bin").write_bytes(b"\x00" * 64)
    assert artifact_fingerprint(tmp_path) != first
