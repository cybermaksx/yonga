"""Every architecture family the registry names, pinned to what was measured.

Each rule below was written from a real preparation on the Lunar Lake
reference host (prepare + probe + generation on GPU and NPU, a benchmark on
the CPU). These tests make sure a model of each family resolves to its own
rule -- not to the generic fallback -- and that the rule claims exactly the
evidence that exists.
"""

from __future__ import annotations

import pytest

from yonga.domain.compatibility import DeviceStatus, EvidenceLevel
from yonga.domain.models import ModelIdentity, ModelMetadata
from yonga.registry.loader import load_builtin_registry
from yonga.registry.resolver import RegistryResolver

RESOLVER = RegistryResolver(load_builtin_registry(), versions={})

ALL_TESTED = ("CPU", "GPU", "NPU")

FAMILIES = [
    ("qwen2", "Qwen2ForCausalLM", "arch-qwen2", ALL_TESTED),
    ("qwen3", "Qwen3ForCausalLM", "arch-qwen3", ALL_TESTED),
    ("granite", "GraniteForCausalLM", "arch-granite", ALL_TESTED),
    ("llama", "LlamaForCausalLM", "arch-llama", ALL_TESTED),
    ("gemma3_text", "Gemma3ForCausalLM", "arch-gemma3-text", ALL_TESTED),
    ("gemma2", "Gemma2ForCausalLM", "arch-gemma2", ALL_TESTED),
    ("olmo2", "Olmo2ForCausalLM", "arch-olmo2", ALL_TESTED),
    ("smollm3", "SmolLM3ForCausalLM", "arch-smollm3", ALL_TESTED),
    ("lfm2", "Lfm2ForCausalLM", "arch-lfm2", ALL_TESTED),
    ("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration", "arch-qwen2_5_vl", ALL_TESTED),
]


def meta(model_type: str, architecture: str) -> ModelMetadata:
    return ModelMetadata(
        identity=ModelIdentity(repo_id=f"acme/{model_type}"),
        model_type=model_type,
        architectures=[architecture],
        config={"model_type": model_type, "architectures": [architecture]},
    )


@pytest.mark.parametrize(("model_type", "architecture", "rule_id", "tested"), FAMILIES)
def test_each_family_resolves_to_its_own_measured_rule(
    model_type: str, architecture: str, rule_id: str, tested: tuple[str, ...]
) -> None:
    rule, _ = RESOLVER.resolve_architecture(meta(model_type, architecture))

    assert rule is not None and rule.id == rule_id, "not the generic fallback"
    for device in tested:
        assert rule.devices[device].evidence is EvidenceLevel.TESTED
    assert rule.export.trust_remote_code is False
    assert any(source.kind.value == "tested" for source in rule.sources)


def test_text_models_export_with_their_kv_cache() -> None:
    for model_type, architecture, _rule, _tested in FAMILIES:
        rule, _ = RESOLVER.resolve_architecture(meta(model_type, architecture))
        assert rule is not None
        expected = (
            "image-text-to-text" if model_type == "qwen2_5_vl" else "text-generation-with-past"
        )
        assert rule.export.task == expected


def test_the_npu_refusal_of_qwen2_5_vl_is_recorded_as_tested() -> None:
    rule, _ = RESOLVER.resolve_architecture(
        meta("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration")
    )

    assert rule is not None
    assert rule.devices["NPU"].status is DeviceStatus.UNSUPPORTED
    assert rule.devices["GPU"].status is DeviceStatus.SUPPORTED


def test_an_unmeasured_family_still_falls_back_honestly() -> None:
    """mistral is only covered by the generic rule, whose evidence is inferred."""
    rule, _ = RESOLVER.resolve_architecture(meta("mistral", "MistralForCausalLM"))

    assert rule is not None and rule.id == "arch-generic-text-generation"
    assert rule.devices["GPU"].evidence is EvidenceLevel.INFERRED


def test_phi3_claims_only_the_devices_its_patched_export_ran_on() -> None:
    """Phi-4-mini ran on the CPU and NPU, exported with optimum's unreleased fix (PR 2496).

    The GPU was never tried, and the Python 3.14 export limitation stays recorded.
    """
    rule, _ = RESOLVER.resolve_architecture(meta("phi3", "Phi3ForCausalLM"))

    assert rule is not None and rule.id == "arch-phi3"
    for device in ("CPU", "NPU"):
        assert rule.devices[device].status is DeviceStatus.SUPPORTED
        assert rule.devices[device].evidence is EvidenceLevel.TESTED
    assert rule.devices["GPU"].status is DeviceStatus.UNKNOWN
    assert any("2496" in note for note in rule.devices["NPU"].notes)
    assert any("Python 3.14" in item for item in rule.known_limitations)
