"""Failure classification tests: raw error text in, typed diagnostic out."""

from __future__ import annotations

import textwrap
from pathlib import Path

from yonga.domain.diagnostics import RAW_EXCERPT_MAX_CHARS, FailureCategory, Severity
from yonga.domain.models import ModelIdentity, ModelMetadata
from yonga.registry.loader import (
    load_builtin_registry,
    load_registry_from_documents,
    parse_document,
)
from yonga.registry.resolver import RegistryResolver

HOST_VERSIONS = {
    "openvino": "2026.4.0-22959-99c81491cc3-releases/2026/4",
    "openvino_genai": "2026.4.0.0",
    "openvino_tokenizers": "2026.4.0.0",
    "transformers": "5.5.4",
}

# Verified on the reference host (see HOST_FACTS.md).
QWEN_NPU_ERROR = "OneShotBufferizeVPU2VPUIP failed : Only supports FP16."

NPU_DYNAMIC_SHAPE_ERROR = textwrap.dedent(
    """
    Exception from src/inference/src/cpp/core.cpp:117:
    Exception from src/inference/src/dev/plugin.cpp:54:
    Exception from src/plugins/intel_npu/src/plugin/src/plugin.cpp:552:
    Exception from src/plugins/intel_npu/src/compiler_adapter/src/ze_graph_ext_wrappers.cpp:425:
    Compilation failed. Level0 pfnCreate2 result: ZE_RESULT_ERROR_INVALID_ARGUMENT, code 0x78000004
     - generic error code for invalid arguments . [NPU_DRV] Driver reports a failure from
    vclAllocatedExecutableCreate4, return code: 2013265924
    [NPU_VCL] Compiler returned msg:
    Exception from src/core/src/partial_shape.cpp:261:
    to_shape was called on a dynamic shape.
    """
).strip()

TOKENIZER_STRING_ERROR = textwrap.dedent(
    """
    Exception from src/inference/src/cpp/core.cpp:84:
    Check 'is_valid_type_idx(type_idx)' failed at src/core/src/type/element_type.cpp:115:
    Unsupported element type: STRiNG
    """
).strip()


def resolver() -> RegistryResolver:
    """Build a resolver over the shipped registry with reference-host versions."""
    return RegistryResolver(load_builtin_registry(), versions=HOST_VERSIONS)


def qwen_metadata() -> ModelMetadata:
    """Metadata for the Qwen3.5 reference fixture."""
    return ModelMetadata(
        identity=ModelIdentity(repo_id="nDimensional/Qwen3.5-9B-Uncensored-Safetensors"),
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
    )


def llama_metadata() -> ModelMetadata:
    """Metadata for an unrelated decoder-only model."""
    return ModelMetadata(
        identity=ModelIdentity(repo_id="meta-llama/Llama-3.1-8B-Instruct"),
        model_type="llama",
        architectures=["LlamaForCausalLM"],
    )


# ---------------------------------------------------------------------------
# the four required signatures
# ---------------------------------------------------------------------------


def test_qwen35_npu_signature_classifies_to_the_family_specific_code():
    result = resolver().classify(
        raw_error=QWEN_NPU_ERROR, device="NPU", meta=qwen_metadata(), stage="compile"
    )
    assert result.code == "npu.compiler.unsupported_graph.qwen35"
    assert result.category is FailureCategory.COMPILER_UNSUPPORTED_GRAPH
    assert result.matched_rule_id == "qwen35-npu-fp16-bufferize"
    assert result.environment_broken is False
    assert result.retry_same_device is False
    assert result.fallback_devices == ["GPU", "CPU"]
    assert result.severity is Severity.ERROR


def test_the_same_text_on_an_unrelated_architecture_uses_the_generic_code():
    result = resolver().classify(
        raw_error=QWEN_NPU_ERROR, device="NPU", meta=llama_metadata(), stage="compile"
    )
    assert result.code == "npu.compiler.unsupported_graph"
    assert result.matched_rule_id == "npu-compiler-unsupported-graph-generic"
    assert result.category is FailureCategory.COMPILER_UNSUPPORTED_GRAPH
    assert result.fallback_devices == ["GPU", "CPU"]


def test_the_generic_rule_also_covers_an_unknown_model():
    result = resolver().classify(raw_error=QWEN_NPU_ERROR, device="NPU")
    assert result.code == "npu.compiler.unsupported_graph"


def test_dynamic_shape_signature():
    result = resolver().classify(
        raw_error=RuntimeError(NPU_DYNAMIC_SHAPE_ERROR), device="npu", meta=llama_metadata()
    )
    assert result.code == "npu.compiler.dynamic_shape"
    assert result.category is FailureCategory.COMPILER_UNSUPPORTED_GRAPH
    assert result.environment_broken is False
    assert result.fallback_devices == ["GPU", "CPU"]
    assert "MAX_PROMPT_LEN" in result.message
    assert "MIN_RESPONSE_LEN" in result.message


def test_level_zero_invalid_argument_is_not_blamed_on_the_driver():
    result = resolver().classify(raw_error=NPU_DYNAMIC_SHAPE_ERROR, device="npu")
    assert result.category is not FailureCategory.DRIVER_MISSING
    assert result.environment_broken is False


def test_tokenizer_string_precision_signature_points_at_its_repair():
    result = resolver().classify(raw_error=TOKENIZER_STRING_ERROR, device="CPU", stage="validate")
    assert result.code == "tokenizer.ir.string_precision"
    assert result.category is FailureCategory.TOKENIZER_IR_INCOMPATIBLE
    assert result.context["repair"] == "ov-tokenizer-precision-string-casing"
    assert any("tokenizer_string_precision" in item for item in result.remediation)


# ---------------------------------------------------------------------------
# environment rules
# ---------------------------------------------------------------------------


def test_npu_permission_denied_is_environment_broken_and_offers_no_fallback():
    result = resolver().classify(
        raw_error="RuntimeError: Permission denied while opening /dev/accel/accel0",
        device="NPU",
    )
    assert result.code == "npu.device.permission_denied"
    assert result.category is FailureCategory.PERMISSION_DENIED
    assert result.environment_broken is True
    assert result.retry_same_device is False
    assert result.fallback_devices == []


def test_out_of_memory_is_classified():
    result = resolver().classify(raw_error="terminate called after throwing std::bad_alloc")
    assert result.category is FailureCategory.OUT_OF_MEMORY
    assert result.code == "runtime.out_of_memory"


def test_no_space_left_is_classified_as_insufficient_disk():
    result = resolver().classify(raw_error="OSError: [Errno 28] No space left on device")
    assert result.category is FailureCategory.INSUFFICIENT_DISK
    assert result.code == "storage.insufficient_disk"


def test_iostream_save_failure_is_classified_as_insufficient_disk():
    result = resolver().classify(
        raw_error="Exception from src/core/src/pass/serialize.cpp: basic_ios::clear: iostream error"
    )
    assert result.category is FailureCategory.INSUFFICIENT_DISK


# ---------------------------------------------------------------------------
# generic inference and unknown failures
# ---------------------------------------------------------------------------


def test_gated_repository_falls_back_to_built_in_inference():
    result = resolver().classify(
        raw_error="401 Client Error: Unauthorized for url: https://huggingface.co/api/models/x"
    )
    assert result.category is FailureCategory.GATED_MODEL_AUTH_REQUIRED
    assert result.matched_rule_id is None


def test_abi_mismatch_falls_back_to_built_in_inference():
    result = resolver().classify(
        raw_error="ImportError: libopenvino.so.2640: undefined symbol: _ZN2ov4Node4typeEv"
    )
    assert result.category is FailureCategory.PACKAGE_ABI_MISMATCH


def test_remote_code_falls_back_to_built_in_inference():
    result = resolver().classify(
        raw_error="ValueError: Loading this model requires you to execute custom code; "
        "pass trust_remote_code=True to allow it."
    )
    assert result.category is FailureCategory.REMOTE_CODE_REQUIRED


def test_unrecognised_failure_is_reported_as_unknown():
    result = resolver().classify(raw_error="something entirely new went wrong")
    assert result.code == "runtime.unknown_error"
    assert result.category is FailureCategory.UNKNOWN_RUNTIME_ERROR
    assert result.matched_rule_id is None


def test_device_scoped_rules_do_not_fire_without_a_device():
    result = resolver().classify(raw_error=QWEN_NPU_ERROR, meta=qwen_metadata())
    assert result.code == "runtime.unknown_error"


# ---------------------------------------------------------------------------
# raw text handling
# ---------------------------------------------------------------------------


def test_raw_excerpt_is_always_kept():
    for raw in (QWEN_NPU_ERROR, TOKENIZER_STRING_ERROR, "mystery failure"):
        result = resolver().classify(raw_error=raw, device="NPU")
        assert result.raw_excerpt is not None
        assert raw.splitlines()[-1] in result.raw_excerpt


def test_raw_excerpt_is_redacted():
    raw = "download failed with token hf_abcdefghijklmnopqrstuvwxyz0123456789 attached"
    result = resolver().classify(raw_error=raw)
    assert result.raw_excerpt is not None
    assert "hf_abcdefghijklmnopqrstuvwxyz0123456789" not in result.raw_excerpt
    assert "***REDACTED***" in result.raw_excerpt


def test_raw_excerpt_is_truncated_but_keeps_the_tail():
    raw = "x" * 20000 + "\nUnsupported element type: STRiNG"
    result = resolver().classify(raw_error=raw)
    assert result.code == "tokenizer.ir.string_precision"
    assert result.raw_excerpt is not None
    assert len(result.raw_excerpt) <= RAW_EXCERPT_MAX_CHARS
    assert result.raw_excerpt.endswith("Unsupported element type: STRiNG")
    assert "truncated" in result.raw_excerpt


def test_exception_type_name_is_part_of_the_matched_text():
    result = resolver().classify(raw_error=RuntimeError(""), device="NPU")
    assert result.raw_excerpt == "RuntimeError"


def test_context_carries_the_job_facts():
    result = resolver().classify(
        raw_error=QWEN_NPU_ERROR,
        device="gpu.1",
        meta=qwen_metadata(),
        stage="compile",
        job_id="job-123",
        log_path=Path("/tmp/job-123.jsonl"),
    )
    assert result.context["device"] == "GPU"
    assert result.context["stage"] == "compile"
    assert result.context["repo_id"] == "nDimensional/Qwen3.5-9B-Uncensored-Safetensors"
    assert result.context["model_type"] == "qwen3_5"
    assert result.job_id == "job-123"
    assert result.log_path == Path("/tmp/job-123.jsonl")


# ---------------------------------------------------------------------------
# robustness
# ---------------------------------------------------------------------------


class _ExplodingError(Exception):
    """An exception whose string rendering fails, as some native wrappers do."""

    def __str__(self) -> str:
        raise ValueError("cannot render this exception")


def test_classification_never_raises():
    result = resolver().classify(raw_error=_ExplodingError(), device="NPU")
    assert result.code == "runtime.unknown_error"
    assert result.category is FailureCategory.UNKNOWN_RUNTIME_ERROR


def test_classification_survives_a_broken_rule():
    registry = load_registry_from_documents(
        [
            parse_document(
                textwrap.dedent(
                    """
                    issues:
                      - id: issue-bad-category
                        match:
                          exception_contains: ["kaboom"]
                        classification:
                          code: x.broken
                          category: not_a_real_category
                    """
                ),
                origin="demo.yaml",
            )
        ]
    )
    result = RegistryResolver(registry, versions=HOST_VERSIONS).classify(
        raw_error="kaboom happened"
    )
    assert result.code == "x.broken"
    assert result.category is FailureCategory.UNKNOWN_RUNTIME_ERROR


def test_stage_scoped_rules_only_fire_for_their_stage():
    registry = load_registry_from_documents(
        [
            parse_document(
                textwrap.dedent(
                    """
                    issues:
                      - id: issue-export-only
                        match:
                          stage: export
                          exception_contains: ["kaboom"]
                        classification:
                          code: x.export
                          category: configuration_error
                    """
                ),
                origin="demo.yaml",
            )
        ]
    )
    resolver_ = RegistryResolver(registry, versions=HOST_VERSIONS)
    assert resolver_.classify(raw_error="kaboom", stage="export").code == "x.export"
    assert resolver_.classify(raw_error="kaboom", stage="compile").code == "runtime.unknown_error"


def test_classification_explains_its_candidates():
    result, explanation = resolver().classify_explained(
        raw_error=QWEN_NPU_ERROR, device="NPU", meta=qwen_metadata()
    )
    assert result.code == "npu.compiler.unsupported_graph.qwen35"
    assert explanation.winner_id == "qwen35-npu-fp16-bufferize"
    rejected = {candidate.rule_id for candidate in explanation.rejected}
    assert "tokenizer-string-precision-read" in rejected
    matched = {candidate.rule_id for candidate in explanation.matched}
    assert matched == {"qwen35-npu-fp16-bufferize", "npu-compiler-unsupported-graph-generic"}


def test_specificity_of_the_family_rule_beats_the_generic_one():
    _result, explanation = resolver().classify_explained(
        raw_error=QWEN_NPU_ERROR, device="NPU", meta=qwen_metadata()
    )
    scores = {candidate.rule_id: candidate.specificity for candidate in explanation.candidates}
    assert scores["qwen35-npu-fp16-bufferize"] > scores["npu-compiler-unsupported-graph-generic"]


def test_equally_specific_rules_are_broken_deterministically_without_raising():
    registry = load_registry_from_documents(
        [
            parse_document(
                textwrap.dedent(
                    """
                    issues:
                      - id: issue-b
                        priority: 100
                        match:
                          exception_contains: ["kaboom"]
                        classification:
                          code: x.b
                          category: configuration_error
                      - id: issue-a
                        priority: 100
                        match:
                          exception_contains: ["kaboom"]
                        classification:
                          code: x.a
                          category: configuration_error
                    """
                ),
                origin="demo.yaml",
            )
        ]
    )
    resolver_ = RegistryResolver(registry, versions=HOST_VERSIONS)
    result, explanation = resolver_.classify_explained(raw_error="kaboom")
    assert result.code == "x.a"
    assert any("equally specific" in note for note in explanation.notes)
