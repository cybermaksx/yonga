"""Failure classification tests: raw error text in, typed diagnostic out."""

from __future__ import annotations

import textwrap
from pathlib import Path

from yonga.domain.diagnostics import (
    RAW_EXCERPT_MAX_CHARS,
    ExitCode,
    FailureCategory,
    Severity,
)
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

# Verbatim tail of a real failed conversion on the reference host: `yonga prepare
# hf-internal-testing/tiny-random-LlamaForCausalLM --device cpu` with the default
# cpu-int4-asym-g128 profile (optimum-intel 2.2.0, nncf 3.4.0). Before this was
# classified it surfaced as "The Optimum Intel exporter exited with status 1".
# Written line by line because the nncf message is longer than the line limit.
NNCF_GROUP_SIZE_ERROR = "\n".join(
    [
        '  File "/venv/lib/python3.14/site-packages/nncf/quantization/algorithms'
        '/weight_compression/algorithm.py", line 679, in validate_group_size',
        "    raise nncf.InvalidGroupSizeError(msg)",
        "nncf.errors.InvalidGroupSizeError: Failed to apply group-wise quantization"
        " with group size value 128.",
        "Ensure that the group size is divisible by the channel size, or consider setting",
        "`group_size_fallback_mode` to IGNORE or ADJUST. Failed nodes:",
        '\t"__module.model.layers.0.self_attn.v_proj/aten::linear/MatMul" (channel size: 16)',
        '\t"__module.model.layers.0.self_attn.k_proj/aten::linear/MatMul" (channel size: 16)',
        '\t"__module.model.layers.0.mlp.down_proj/aten::linear/MatMul" (channel size: 64)',
    ]
)

# An exporter failure with no known signature: it must stay generic.
UNRELATED_EXPORT_ERROR = "\n".join(
    [
        '  File "/venv/lib/python3.14/site-packages/optimum/exporters/openvino'
        '/__main__.py", line 402, in main_export',
        '    raise ValueError(f"Asked to export a model for the task {task}")',
        "ValueError: Asked to export a model for the task feature-extraction,"
        " which is not supported",
    ]
)


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


# Verbatim from `yonga chat 7bd72baf64957953 --device npu` on the reference host
# (OpenVINO 2026.4, compiler-in-plugin). A different compiler pass from the
# bufferization and dynamic-shape forms above.
NPU_COMPILER_PASS_ERROR = textwrap.dedent(
    """
    [ERROR] [vpux-compiler] MultiClusterStrategyAssignment Pass failed : Cannot get per cluster
    memory shapes. Shape [1, 1, 8, 1], Unsupported distribution:
    #VPU.DistributedTensor<mode = SEGMENTED, num_tiles = [1, 1, 6, 1], num_clusters = 6>
    [ERROR] [vpux-compiler] Failed Pass MultiClusterStrategyAssignment on Operation
    Exception from src/inference/src/cpp/core.cpp:117:
    Exception from src/plugins/intel_npu/src/plugin/src/plugin.cpp:552:
    Exception from src/plugins/intel_npu/src/compiler_adapter/src/ze_graph_ext_wrappers.cpp:425:
    Compilation failed. Level0 pfnCreate2 result: ZE_RESULT_ERROR_INVALID_ARGUMENT, code 0x78000004
     - generic error code for invalid arguments . [NPU_DRV] Driver reports a failure from
    vclAllocatedExecutableCreate4, return code: 2013265924
    [NPU_VCL] Compiler returned msg:
    Compilation failed
    """
).strip()


def test_a_named_compiler_pass_failure_is_not_blamed_on_the_driver():
    """Verbatim from `yonga chat 7bd72baf64957953 --device npu` on the reference host.

    Before `npu-compiler-pass-failed` shipped, this matched no rule and fell
    through to a generic message whose remediation was "Run `yonga doctor` to
    check the device and driver state" — the one thing TROUBLESHOOTING section
    5.1 tells users not to do, because the driver is working.
    """
    result = resolver().classify(
        raw_error=NPU_COMPILER_PASS_ERROR, device="npu", meta=llama_metadata()
    )
    assert result.code == "npu.compiler.pass_failed"
    assert result.matched_rule_id == "npu-compiler-pass-failed"
    assert result.category is FailureCategory.COMPILER_UNSUPPORTED_GRAPH
    assert result.exit_code() is ExitCode.COMPILE
    assert result.environment_broken is False
    assert result.fallback_devices == ["GPU", "CPU"]
    assert not any("doctor" in step for step in result.remediation)
    assert any("--device gpu" in step for step in result.remediation)


def test_a_more_specific_npu_rule_still_wins_over_the_pass_failure_rule():
    """The dynamic-shape text also contains "[NPU_VCL] Compiler returned msg:"."""
    result = resolver().classify(raw_error=NPU_DYNAMIC_SHAPE_ERROR, device="npu")
    assert result.code == "npu.compiler.dynamic_shape"


def test_a_pass_failure_on_another_device_is_not_claimed_by_the_npu_rule():
    result = resolver().classify(raw_error=NPU_COMPILER_PASS_ERROR, device="gpu")
    assert result.code != "npu.compiler.pass_failed"


def test_an_uninitialised_driver_is_still_a_driver_problem():
    """The pass-failure rule must not swallow the genuinely broken-driver case."""
    result = resolver().classify(
        raw_error="ZE_RESULT_ERROR_UNINITIALIZED: the NPU driver could not be initialised",
        device="npu",
    )
    assert result.category is FailureCategory.DRIVER_MISSING


#: Verbatim from preparing Qwen3-4B for the NPU with Arch's intel-npu-driver,
#: which ships no compiler library.
NPU_DRIVER_COMPILER_MISSING_ERROR = (
    "Exception from src/inference/src/cpp/core.cpp:117:\n"
    "Exception from src/inference/src/dev/plugin.cpp:54:\n"
    "Exception from src/plugins/intel_npu/src/plugin/src/plugin.cpp:560:\n"
    "Exception from src/plugins/intel_npu/src/compiler_adapter/src/"
    "ze_graph_ext_wrappers.cpp:425:\n"
    "Compilation failed. Level0 pfnCreate2 result: ZE_RESULT_ERROR_UNSUPPORTED_FEATURE, "
    "code 0x78000003 - generic error code for unsupported features . \n"
)


def test_an_unsupported_feature_names_both_causes_and_keeps_the_fallback():
    """The text alone cannot tell a missing compiler from an IR it cannot take."""
    result = resolver().classify(raw_error=NPU_DRIVER_COMPILER_MISSING_ERROR, device="npu")
    assert result.code == "npu.compiler.unsupported_feature"
    assert result.matched_rule_id == "npu-unsupported-feature"
    assert result.category is FailureCategory.COMPILER_UNSUPPORTED_GRAPH
    assert result.environment_broken is False
    assert result.fallback_devices == ["GPU", "CPU"]
    remediation = " ".join(result.remediation)
    assert "libopenvino_intel_npu_compiler_loader.so" in remediation
    assert "yonga doctor" in remediation
    assert "--device gpu" in remediation


def test_a_compiler_that_answered_still_wins_over_the_missing_compiler_rule():
    raw = NPU_DRIVER_COMPILER_MISSING_ERROR + "[NPU_VCL] Compiler returned msg:\nFailed Pass"
    result = resolver().classify(raw_error=raw, device="npu")
    assert result.code == "npu.compiler.pass_failed"


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


def test_tokenizer_repair_remediation_names_a_command_that_exists():
    """The repair runs inside `prepare`; there is no `yonga repair` command."""
    result = resolver().classify(raw_error=TOKENIZER_STRING_ERROR, device="CPU", stage="validate")
    joined = " ".join(result.remediation)
    assert "yonga repair" not in joined
    assert "yonga prepare" in joined


def test_nncf_group_size_failure_is_classified_instead_of_being_a_generic_export_failure():
    result = resolver().classify(
        raw_error=NNCF_GROUP_SIZE_ERROR, device="CPU", stage="export", job_id="job-7"
    )
    assert result.code == "export.quantization.group_size_mismatch"
    assert result.matched_rule_id == "nncf-group-size-not-divisible"
    assert result.category is FailureCategory.CONFIGURATION_ERROR
    assert result.environment_broken is False
    assert result.retry_same_device is False
    # A conversion setting, not a device verdict: no fallback device is offered.
    assert result.fallback_devices == []
    assert result.exit_code() is ExitCode.REGISTRY


def test_nncf_group_size_remediation_names_the_concrete_way_out():
    result = resolver().classify(raw_error=NNCF_GROUP_SIZE_ERROR, stage="export")
    assert "group size" in result.message
    assert "channel size" in result.message
    joined = " ".join(result.remediation)
    assert "--profile npu-int4-sym-cw" in joined
    assert "group size -1" in joined
    assert "yonga logs" in joined


def test_nncf_group_size_rule_matches_the_bare_exception_name():
    result = resolver().classify(raw_error=RuntimeError("nncf.errors.InvalidGroupSizeError"))
    assert result.code == "export.quantization.group_size_mismatch"


def test_nncf_group_size_rule_is_not_scoped_to_one_device():
    for device in ("cpu", "gpu", "npu", None):
        result = resolver().classify(raw_error=NNCF_GROUP_SIZE_ERROR, device=device, stage="export")
        assert result.code == "export.quantization.group_size_mismatch"


# ---------------------------------------------------------------------------
# exporter/Transformers version mismatch
# ---------------------------------------------------------------------------

# Verbatim shape of a real `yonga prepare
# HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive --device auto` run on the
# reference host (optimum-intel 2.2.0, transformers 5.5.4). The plan had already
# printed "transformers ==5.2.* is required, but 5.5.4 is installed"; the run went
# ahead anyway, downloaded 17.5 GiB and then died here, surfacing only as "The
# Optimum Intel exporter exited with status 1".
QWEN_EXPORT_SYMBOL_ERROR = "\n".join(
    [
        '  File "/venv/lib/python3.14/site-packages/optimum/exporters/openvino'
        '/model_patcher.py", line 10049, in __init__',
        "    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache",
        "ImportError: cannot import name 'Qwen3_5DynamicCache' from "
        "'transformers.models.qwen3_5.modeling_qwen3_5' "
        "(/venv/lib/python3.14/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py)",
    ]
)

# The same failure shape with a different architecture substituted. The pin is
# per-architecture (SPEC section 5.2), so any family can reach this.
OTHER_EXPORT_SYMBOL_ERROR = "\n".join(
    [
        '  File "/venv/lib/python3.14/site-packages/optimum/exporters/openvino'
        '/model_patcher.py", line 8123, in __init__',
        "    from transformers.models.llama.modeling_llama import LlamaDynamicCache",
        "ImportError: cannot import name 'LlamaDynamicCache' from "
        "'transformers.models.llama.modeling_llama' "
        "(/venv/lib/python3.14/site-packages/transformers/models/llama/modeling_llama.py)",
    ]
)

# The attribute form of the same mismatch: the module imports, the name inside it
# has moved or been renamed.
ATTRIBUTE_EXPORT_SYMBOL_ERROR = "\n".join(
    [
        '  File "/venv/lib/python3.14/site-packages/optimum/exporters/openvino'
        '/model_patcher.py", line 7740, in patch_model',
        "    original = transformers.models.llama.modeling_llama.LlamaAttention",
        "AttributeError: module 'transformers.models.llama.modeling_llama' has no "
        "attribute 'LlamaAttention'",
    ]
)

# A patcher failure that never mentions transformers by module path: only the
# frame says where it came from.
PATCHER_FRAME_ERROR = "\n".join(
    [
        '  File "/venv/lib/python3.14/site-packages/optimum/exporters/openvino'
        '/model_patcher.py", line 4210, in __enter__',
        "    super().__enter__()",
        "AttributeError: 'MistralModel' object has no attribute '_update_causal_mask'",
    ]
)


def test_qwen35_export_symbol_mismatch_is_classified_as_a_package_combination():
    """The observed defect: this used to be a bare "exporter exited with status 1"."""
    result = resolver().classify(
        raw_error=QWEN_EXPORT_SYMBOL_ERROR,
        device="GPU",
        meta=qwen_metadata(),
        stage="export",
    )
    assert result.code == "export.transformers.symbol_missing.qwen35"
    assert result.matched_rule_id == "qwen35-exporter-transformers-symbol-mismatch"
    assert result.category is FailureCategory.PACKAGE_ABI_MISMATCH
    # The packages are each fine and every device works: nothing to fall back to,
    # nothing to declare dead, and retrying changes nothing until a package does.
    assert result.environment_broken is False
    assert result.retry_same_device is False
    assert result.fallback_devices == []
    assert result.exit_code() is ExitCode.ENVIRONMENT_UNUSABLE


def test_qwen35_export_symbol_remediation_names_the_version_and_where_to_read_it():
    result = resolver().classify(
        raw_error=QWEN_EXPORT_SYMBOL_ERROR, meta=qwen_metadata(), stage="export"
    )
    joined = " ".join(result.remediation)
    assert "transformers==5.2.*" in joined
    # The repository is known, so the step names it instead of the placeholder.
    assert "yonga inspect nDimensional/Qwen3.5-9B-Uncensored-Safetensors" in joined
    assert "MODEL" not in joined


def test_qwen35_export_symbol_message_denies_that_a_readable_config_proves_anything():
    """5.5.4 loads Qwen3_5Config and reports model_type qwen3_5, and still cannot export."""
    result = resolver().classify(
        raw_error=QWEN_EXPORT_SYMBOL_ERROR, meta=qwen_metadata(), stage="export"
    )
    assert "Qwen3_5Config" in result.message
    assert "model_type: qwen3_5" in result.message


def test_the_same_export_shape_on_another_architecture_uses_the_generic_code():
    result = resolver().classify(
        raw_error=OTHER_EXPORT_SYMBOL_ERROR,
        device="CPU",
        meta=llama_metadata(),
        stage="export",
    )
    assert result.code == "export.transformers.symbol_missing"
    assert result.matched_rule_id == "exporter-transformers-symbol-mismatch"
    assert result.category is FailureCategory.PACKAGE_ABI_MISMATCH
    assert result.environment_broken is False
    assert result.fallback_devices == []


def test_the_generic_export_rule_also_covers_an_unknown_model():
    result = resolver().classify(raw_error=OTHER_EXPORT_SYMBOL_ERROR, stage="export")
    assert result.code == "export.transformers.symbol_missing"


def test_a_qwen_traceback_without_the_named_symbol_still_reaches_the_generic_rule():
    """Specificity must not be a trap: the family rule names one symbol, not all of them."""
    result = resolver().classify(
        raw_error=OTHER_EXPORT_SYMBOL_ERROR, meta=qwen_metadata(), stage="export"
    )
    assert result.code == "export.transformers.symbol_missing"


def test_the_attribute_form_of_the_mismatch_is_classified_too():
    result = resolver().classify(raw_error=ATTRIBUTE_EXPORT_SYMBOL_ERROR, stage="export")
    assert result.code == "export.transformers.symbol_missing"
    assert result.category is FailureCategory.PACKAGE_ABI_MISMATCH


def test_a_failure_raised_inside_the_export_patcher_is_classified_by_its_frame_alone():
    """model_patcher.py exists only to rewrite transformers internals for export."""
    result = resolver().classify(raw_error=PATCHER_FRAME_ERROR, stage="export")
    assert result.code == "export.transformers.symbol_missing"


def test_the_export_symbol_rule_is_not_scoped_to_one_device():
    """Conversion runs on the host; the target device is not what failed."""
    for device in ("cpu", "gpu", "npu", None):
        result = resolver().classify(
            raw_error=QWEN_EXPORT_SYMBOL_ERROR,
            device=device,
            meta=qwen_metadata(),
            stage="export",
        )
        assert result.code == "export.transformers.symbol_missing.qwen35"


def test_specificity_of_the_export_family_rule_beats_the_generic_one():
    _result, explanation = resolver().classify_explained(
        raw_error=QWEN_EXPORT_SYMBOL_ERROR, device="GPU", meta=qwen_metadata(), stage="export"
    )
    matched = {candidate.rule_id for candidate in explanation.matched}
    assert matched == {
        "qwen35-exporter-transformers-symbol-mismatch",
        "exporter-transformers-symbol-mismatch",
    }
    scores = {candidate.rule_id: candidate.specificity for candidate in explanation.candidates}
    assert (
        scores["qwen35-exporter-transformers-symbol-mismatch"]
        > scores["exporter-transformers-symbol-mismatch"]
    )


def test_the_export_symbol_rules_do_not_claim_an_unrelated_openvino_abi_break():
    """A genuine native ABI break must keep its own built-in inference."""
    result = resolver().classify(
        raw_error="ImportError: libopenvino.so.2640: undefined symbol: _ZN2ov4Node4typeEv",
        stage="export",
    )
    assert result.matched_rule_id is None
    assert result.category is FailureCategory.PACKAGE_ABI_MISMATCH


def test_an_unrelated_exporter_failure_still_classifies_generically():
    result = resolver().classify(raw_error=UNRELATED_EXPORT_ERROR, device="CPU", stage="export")
    assert result.code == "runtime.unknown_error"
    assert result.category is FailureCategory.UNKNOWN_RUNTIME_ERROR
    assert result.matched_rule_id is None
    assert result.raw_excerpt is not None
    assert "feature-extraction" in result.raw_excerpt


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


# --- which raw failures the GenAI pipeline must settle -------------------------------


def test_only_the_dynamic_shape_rejection_is_deferred_to_the_pipeline() -> None:
    """A raw dynamic-shape failure is not evidence; a real compiler rejection is."""
    from yonga.registry import RegistryResolver, load_builtin_registry

    resolver = RegistryResolver(load_builtin_registry())
    dynamic = resolver.classify(
        raw_error="Exception ...\nto_shape was called on a dynamic shape.", device="NPU"
    )
    fp16 = resolver.classify(
        raw_error="OneShotBufferizeVPU2VPUIP failed : Only supports FP16.", device="NPU"
    )

    assert dynamic.code == "npu.compiler.dynamic_shape"
    assert dynamic.defer_to_pipeline is True
    assert fp16.defer_to_pipeline is False


def test_npu_profiles_carry_the_static_shape_budget() -> None:
    """The NPU pipeline reshapes to these; the manifest records what was probed."""
    from yonga.registry import load_builtin_registry

    profiles = {rule.id: rule.to_conversion_profile() for rule in load_builtin_registry().profiles}
    for profile_id in ("npu-int4-sym-cw", "npu-int4-sym-g128"):
        runtime = profiles[profile_id].runtime_properties
        assert runtime["MAX_PROMPT_LEN"] == "1024"
        assert runtime["MIN_RESPONSE_LEN"] == "128"
    assert "MAX_PROMPT_LEN" not in profiles["gpu-int4-asym-g128"].runtime_properties


# --- an exporter defect that only exists on one Python ----------------------------------

OPTIMUM_ALLOW_NEW_ERROR = (
    "TypeError: NormalizedConfig.__init__() got multiple values for argument 'allow_new'"
)


def test_the_optimum_allow_new_failure_is_blamed_on_python_3_14_only() -> None:
    """Verified on the reference host: mistral and phi3 configs fail on 3.14, not before."""
    registry = load_builtin_registry()
    on_314 = RegistryResolver(registry, versions={**HOST_VERSIONS, "python": "3.14.4"})
    on_313 = RegistryResolver(registry, versions={**HOST_VERSIONS, "python": "3.13.5"})

    diagnostic = on_314.classify(raw_error=OPTIMUM_ALLOW_NEW_ERROR)
    assert diagnostic.code == "export.optimum.normalized_config"
    assert "Python 3.14" in diagnostic.message
    assert any("3.13" in step for step in diagnostic.remediation)
    assert on_313.classify(raw_error=OPTIMUM_ALLOW_NEW_ERROR).code != diagnostic.code
