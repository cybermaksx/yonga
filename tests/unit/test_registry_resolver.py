"""Resolution tests: specificity, priority, constraints and explanations."""

from __future__ import annotations

import textwrap

import pytest

from yonga.domain.compatibility import DeviceStatus, EvidenceLevel, ResolutionExplanation
from yonga.domain.diagnostics import FailureCategory, RegistryError
from yonga.domain.hardware import (
    DeviceKind,
    HardwareInventory,
    HostInfo,
    MemoryInfo,
    OpenVINODevice,
)
from yonga.domain.models import ModelIdentity, ModelMetadata
from yonga.registry.loader import (
    load_builtin_registry,
    load_registry_from_documents,
    parse_document,
)
from yonga.registry.resolver import (
    RegistryResolver,
    installed_registry_versions,
)
from yonga.registry.schema import (
    SPECIFICITY_ARCHITECTURE,
    SPECIFICITY_GENERIC,
    SPECIFICITY_REPO_ID,
    Registry,
)

HOST_VERSIONS = {
    "openvino": "2026.4.0-22959-99c81491cc3-releases/2026/4",
    "openvino_genai": "2026.4.0.0",
    "openvino_tokenizers": "2026.4.0.0",
    "optimum_intel": "2.2.0",
    "transformers": "5.5.4",
    "nncf": "3.4.0",
    "torch": "2.14.0",
    "python": "3.14.4",
}


def registry_from(text: str) -> Registry:
    """Build a registry from one inline YAML document."""
    return load_registry_from_documents([parse_document(textwrap.dedent(text), origin="demo.yaml")])


def resolver_for(text: str, **kwargs: object) -> RegistryResolver:
    """Build a resolver over an inline registry document."""
    versions = kwargs.pop("versions", HOST_VERSIONS)
    return RegistryResolver(registry_from(text), versions=versions, **kwargs)  # type: ignore[arg-type]


def builtin_resolver(**kwargs: object) -> RegistryResolver:
    """Build a resolver over the shipped registry."""
    versions = kwargs.pop("versions", HOST_VERSIONS)
    return RegistryResolver(load_builtin_registry(), versions=versions, **kwargs)  # type: ignore[arg-type]


def qwen_metadata() -> ModelMetadata:
    """Metadata for the Qwen3.5 reference fixture."""
    return ModelMetadata(
        identity=ModelIdentity(repo_id="nDimensional/Qwen3.5-9B-Uncensored-Safetensors"),
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
        pipeline_tag="image-text-to-text",
    )


def llama_metadata() -> ModelMetadata:
    """Metadata for an unrelated decoder-only model."""
    return ModelMetadata(
        identity=ModelIdentity(repo_id="meta-llama/Llama-3.1-8B-Instruct"),
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        pipeline_tag="text-generation",
    )


def inventory_with(*devices: str) -> HardwareInventory:
    """Build a minimal hardware inventory exposing the named devices."""
    return HardwareInventory(
        host=HostInfo(
            platform="Linux",
            machine="x86_64",
            kernel="7.2.6",
            python_version="3.14.4",
            distro_id="ubuntu",
            distro_version="26.04",
            distro_pretty="Ubuntu 26.04.1 LTS",
            cpu_model="Intel(R) Core(TM) Ultra 7 268V",
            cpu_count=8,
        ),
        memory=MemoryInfo(
            total_bytes=32 * 2**30,
            available_bytes=16 * 2**30,
            swap_total_bytes=0,
            swap_free_bytes=0,
        ),
        devices=[
            OpenVINODevice(name=name, kind=DeviceKind(name), full_name=name) for name in devices
        ],
        openvino_available=True,
        device_tags=["intel.cpu", "intel.gpu.xe2", "intel.npu.lunar_lake"],
    )


# ---------------------------------------------------------------------------
# architecture resolution
# ---------------------------------------------------------------------------


def test_qwen35_architecture_requires_transformers_5_2():
    resolver = builtin_resolver()
    rule, explanation = resolver.resolve_architecture(qwen_metadata())
    assert rule is not None
    assert rule.id == "arch-qwen3_5"
    assert rule.export.transformers == "==5.2.*"
    assert rule.export.pipeline.value == "vlm"
    assert rule.export.trust_remote_code is False
    assert rule.recommended_device == "GPU"
    assert explanation.winner_id == "arch-qwen3_5"
    assert isinstance(explanation, ResolutionExplanation)


def test_generic_family_rule_covers_unmeasured_families():
    """Families with a measured rule (llama, qwen2, qwen3) resolve to it; mistral does not."""
    resolver = builtin_resolver()
    llama, _ = resolver.resolve_architecture(llama_metadata())
    assert llama is not None and llama.id == "arch-llama"
    mistral = ModelMetadata(
        identity=ModelIdentity(repo_id="acme/mistral-ish"),
        model_type="mistral",
        architectures=["MistralForCausalLM"],
    )
    rule, _explanation = resolver.resolve_architecture(mistral)
    assert rule is not None
    assert rule.id == "arch-generic-text-generation"
    cpu = rule.device_support("CPU")
    gpu = rule.device_support("GPU")
    npu = rule.device_support("NPU")
    assert cpu is not None and cpu.status is DeviceStatus.SUPPORTED
    assert cpu.evidence is EvidenceLevel.INFERRED
    assert gpu is not None and gpu.status is DeviceStatus.SUPPORTED
    assert npu is not None and npu.status is DeviceStatus.UNKNOWN


def test_unknown_architecture_resolves_to_nothing_with_a_note():
    resolver = builtin_resolver()
    metadata = ModelMetadata(
        identity=ModelIdentity(repo_id="owner/exotic"), model_type="exotic_net"
    )
    rule, explanation = resolver.resolve_architecture(metadata)
    assert rule is None
    assert explanation.winner_id is None
    assert explanation.notes
    assert all(not candidate.matched for candidate in explanation.candidates)


def test_exact_repo_rule_beats_an_architecture_rule():
    resolver = resolver_for(
        """
        architectures:
          - id: arch-by-family
            priority: 100
            match:
              model_types: [llama]
          - id: arch-by-repo
            priority: 1
            match:
              repo_ids: [meta-llama/Llama-3.1-8B-Instruct]
        """
    )
    rule, explanation = resolver.resolve_architecture(llama_metadata())
    assert rule is not None
    assert rule.id == "arch-by-repo"
    winners = {candidate.rule_id: candidate for candidate in explanation.candidates}
    assert winners["arch-by-repo"].specificity == SPECIFICITY_REPO_ID
    assert winners["arch-by-family"].specificity == SPECIFICITY_ARCHITECTURE


def test_architecture_plus_device_family_beats_architecture_alone():
    resolver = resolver_for(
        """
        issues:
          - id: issue-family
            priority: 1
            match:
              model_types: [llama]
              device: npu
              exception_contains: ["boom"]
            classification:
              code: a.family
              category: compiler_unsupported_graph
          - id: issue-arch
            priority: 100
            match:
              model_types: [llama]
              exception_contains: ["boom"]
            classification:
              code: a.arch
              category: compiler_unsupported_graph
        """
    )
    result = resolver.classify(raw_error="boom", device="npu", meta=llama_metadata())
    assert result.code == "a.family"
    assert result.matched_rule_id == "issue-family"


def test_priority_breaks_a_specificity_tie():
    resolver = resolver_for(
        """
        architectures:
          - id: arch-low
            priority: 10
            match:
              model_types: [llama]
          - id: arch-high
            priority: 50
            match:
              model_types: [llama]
        """
    )
    rule, _explanation = resolver.resolve_architecture(llama_metadata())
    assert rule is not None
    assert rule.id == "arch-high"


def test_unbreakable_tie_raises_a_registry_error():
    resolver = resolver_for(
        """
        architectures:
          - id: arch-one
            priority: 50
            match:
              model_types: [llama]
          - id: arch-two
            priority: 50
            match:
              model_types: [llama]
        """
    )
    with pytest.raises(RegistryError) as excinfo:
        resolver.resolve_architecture(llama_metadata())
    diagnostic = excinfo.value.diagnostic
    assert diagnostic.code == "registry.ambiguous_rules"
    assert diagnostic.category is FailureCategory.REGISTRY_ERROR
    assert "arch-one" in diagnostic.message
    assert "arch-two" in diagnostic.message


def test_deprecated_rules_never_win():
    resolver = resolver_for(
        """
        architectures:
          - id: arch-old
            priority: 500
            status: deprecated
            summary: superseded
            match:
              model_types: [llama]
          - id: arch-new
            priority: 10
            match:
              model_types: [llama]
        """
    )
    rule, explanation = resolver.resolve_architecture(llama_metadata())
    assert rule is not None
    assert rule.id == "arch-new"
    rejected = {candidate.rule_id for candidate in explanation.rejected}
    assert "arch-old" in rejected


# ---------------------------------------------------------------------------
# version constraints
# ---------------------------------------------------------------------------


def test_version_constraints_filter_before_scoring_and_record_the_reason():
    resolver = resolver_for(
        """
        architectures:
          - id: arch-old-openvino
            priority: 100
            constraints:
              openvino: "<2026"
            match:
              model_types: [llama]
          - id: arch-current
            priority: 10
            constraints:
              openvino: ">=2026.2,<2027"
            match:
              model_types: [llama]
        """
    )
    rule, explanation = resolver.resolve_architecture(llama_metadata())
    assert rule is not None
    assert rule.id == "arch-current"
    rejected = {candidate.rule_id: candidate for candidate in explanation.rejected}
    assert "arch-old-openvino" in rejected
    assert any(
        "does not satisfy <2026" in reason for reason in rejected["arch-old-openvino"].reasons
    )


def test_openvino_build_suffix_is_tolerated_in_constraints():
    resolver = resolver_for(
        """
        architectures:
          - id: arch-current
            constraints:
              openvino: ">=2026.2,<2027"
            match:
              model_types: [llama]
        """
    )
    rule, _explanation = resolver.resolve_architecture(llama_metadata())
    assert rule is not None


def test_unknown_version_is_unverified_by_default_but_rejects_in_strict_mode():
    document = """
        architectures:
          - id: arch-needs-openvino
            constraints:
              openvino: ">=2026.2"
            match:
              model_types: [llama]
        """
    lenient = resolver_for(document, versions={"openvino": None})
    rule, explanation = lenient.resolve_architecture(llama_metadata())
    assert rule is not None
    reasons = [reason for candidate in explanation.candidates for reason in candidate.reasons]
    assert any("not verified" in reason for reason in reasons)

    strict = resolver_for(document, versions={"openvino": None}, strict_versions=True)
    rule, explanation = strict.resolve_architecture(llama_metadata())
    assert rule is None
    assert explanation.rejected


def test_installed_registry_versions_reads_distribution_metadata():
    versions = installed_registry_versions()
    assert "openvino" in versions
    assert versions["python"] is not None


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------


def test_gpu_profile_resolution_for_qwen35():
    resolver = builtin_resolver()
    arch, _ = resolver.resolve_architecture(qwen_metadata())
    profile, explanation = resolver.resolve_profile(
        device="gpu", arch_rule=arch, meta=qwen_metadata()
    )
    assert profile.id == "gpu-int4-asym-g128"
    assert profile.device == "GPU"
    assert profile.quantization.weight_format == "int4"
    assert profile.quantization.symmetric is False
    assert profile.quantization.ratio == 1.0
    assert profile.quantization.group_size == 128
    assert profile.runtime_properties == {"ATTENTION_BACKEND": "SDPA"}
    assert explanation.winner_id == "gpu-int4-asym-g128"


def test_cpu_profile_has_the_same_quantization_but_no_sdpa_property():
    resolver = builtin_resolver()
    gpu, _ = resolver.resolve_profile(device="gpu")
    cpu, _ = resolver.resolve_profile(device="cpu")
    assert cpu.id == "cpu-int4-asym-g128"
    assert cpu.id != gpu.id
    assert cpu.quantization.model_dump() == gpu.quantization.model_dump()
    assert cpu.runtime_properties == {}


def test_npu_channel_wise_profile_wins_over_the_group_wise_one_by_priority():
    resolver = builtin_resolver()
    profile, _ = resolver.resolve_profile(device="npu")
    assert profile.id == "npu-int4-sym-cw"
    assert profile.quantization.symmetric is True
    assert profile.quantization.group_size == -1


def test_architecture_rule_can_name_the_profile_for_a_device():
    resolver = resolver_for(
        """
        architectures:
          - id: arch-x
            match:
              model_types: [llama]
            profiles:
              gpu: gpu-special
        profiles:
          - id: gpu-special
            device: gpu
            priority: 1
          - id: gpu-default
            device: gpu
            priority: 999
        """
    )
    arch, _ = resolver.resolve_architecture(llama_metadata())
    profile, explanation = resolver.resolve_profile(device="gpu", arch_rule=arch)
    assert profile.id == "gpu-special"
    assert any(
        "named by architecture rule" in reason for reason in explanation.candidates[0].reasons
    )


def test_explicit_profile_id_always_wins():
    resolver = builtin_resolver()
    profile, explanation = resolver.resolve_profile(
        device="gpu", explicit_profile_id="npu-int4-sym-cw"
    )
    assert profile.id == "npu-int4-sym-cw"
    assert explanation.winner_id == "npu-int4-sym-cw"
    assert any("targets NPU" in note for note in explanation.notes)


def test_unknown_explicit_profile_id_raises():
    resolver = builtin_resolver()
    with pytest.raises(RegistryError) as excinfo:
        resolver.resolve_profile(device="gpu", explicit_profile_id="does-not-exist")
    assert excinfo.value.diagnostic.code == "registry.profile_unknown"


def test_unknown_explicit_profile_lists_the_profiles_for_the_device():
    """The remediation names the valid ids, and a command that lists them all."""
    resolver = builtin_resolver()
    with pytest.raises(RegistryError) as excinfo:
        resolver.resolve_profile(device="cpu", explicit_profile_id="cpu-int3-bogus")
    available, listing = excinfo.value.diagnostic.remediation
    cpu = [rule.id for rule in resolver.registry.profiles if rule.device == "CPU"]
    assert cpu and available == f"Profiles for CPU: {', '.join(cpu)}."
    assert not any(
        rule.id in available for rule in resolver.registry.profiles if rule.device != "CPU"
    )
    assert "yonga registry profiles" in listing


def test_unknown_explicit_profile_on_a_device_without_profiles():
    resolver = resolver_for(
        """
        profiles:
          - id: cpu-only
            device: cpu
        """
    )
    with pytest.raises(RegistryError) as excinfo:
        resolver.resolve_profile(device="npu", explicit_profile_id="npu-nothing")
    assert excinfo.value.diagnostic.remediation[0] == "No registry profile targets NPU."


def test_missing_profile_for_a_device_raises():
    resolver = resolver_for(
        """
        profiles:
          - id: cpu-only
            device: cpu
        """
    )
    with pytest.raises(RegistryError) as excinfo:
        resolver.resolve_profile(device="npu")
    assert excinfo.value.diagnostic.code == "registry.profile_unresolved"


def test_profile_rejected_by_constraints_falls_through_to_the_next_one():
    resolver = resolver_for(
        """
        profiles:
          - id: gpu-future
            device: gpu
            priority: 999
            constraints:
              openvino: ">=2030"
          - id: gpu-current
            device: gpu
            priority: 1
        """
    )
    profile, explanation = resolver.resolve_profile(device="gpu")
    assert profile.id == "gpu-current"
    assert "gpu-future" in {candidate.rule_id for candidate in explanation.rejected}


# ---------------------------------------------------------------------------
# derivatives
# ---------------------------------------------------------------------------


def test_third_party_derivative_requires_confirmation():
    resolver = builtin_resolver()
    candidates, explanation = resolver.resolve_derivatives(
        ModelIdentity(repo_id="HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive")
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.repo_id == "nDimensional/Qwen3.5-9B-Uncensored-Safetensors"
    assert candidate.relationship.value == "lossless_conversion"
    assert candidate.trust.value == "suggest"
    assert candidate.requires_confirmation is True
    assert candidate.source_evidence()[0].url == (
        "https://huggingface.co/nDimensional/Qwen3.5-9B-Uncensored-Safetensors"
    )
    assert explanation.winner_id == "hauhaucs-qwen35-safetensors"
    assert any("confirmation" in note for note in explanation.notes)


def test_repository_without_a_derivative_rule_returns_nothing():
    resolver = builtin_resolver()
    candidates, explanation = resolver.resolve_derivatives("owner/unknown")
    assert candidates == []
    assert explanation.winner_id is None
    assert explanation.notes


def test_derivative_lookup_accepts_a_plain_string():
    resolver = builtin_resolver()
    candidates, _ = resolver.resolve_derivatives(
        "hauhaucs/qwen3.5-9b-uncensored-hauhaucs-aggressive"
    )
    assert len(candidates) == 1


# ---------------------------------------------------------------------------
# repairs
# ---------------------------------------------------------------------------


def test_repair_rule_selects_only_its_exact_implementation():
    resolver = builtin_resolver()
    rules = resolver.repairs_for(
        files=["openvino_tokenizer.xml", "openvino_detokenizer.xml", "openvino_model.xml"],
        contents_probe=lambda name, needle: (
            name.endswith("tokenizer.xml") and needle == 'precision="STRiNG"'
        ),
    )
    assert [rule.id for rule in rules] == ["ov-tokenizer-precision-string-casing"]
    assert rules[0].implementation == "tokenizer_string_precision"
    assert rules[0].safety.value == "automatic"
    assert rules[0].validation.requires_tokenizer_read is True


def test_repair_is_not_selected_when_the_marker_is_absent():
    resolver = builtin_resolver()
    rules = resolver.repairs_for(
        files=["openvino_tokenizer.xml"],
        contents_probe=lambda _name, _needle: False,
    )
    assert rules == []


def test_repair_is_not_selected_when_no_matching_file_is_present():
    resolver = builtin_resolver()
    assert resolver.repairs_for(files=["openvino_model.xml"]) == []


def test_repair_is_a_candidate_when_no_content_probe_is_supplied():
    resolver = builtin_resolver()
    rules = resolver.repairs_for(files=["/tmp/artifact/openvino_tokenizer.xml"])
    assert [rule.id for rule in rules] == ["ov-tokenizer-precision-string-casing"]


def test_repair_constraints_are_honoured():
    resolver = builtin_resolver(versions={"openvino": "2020.1.0"})
    assert resolver.repairs_for(files=["openvino_tokenizer.xml"]) == []


# ---------------------------------------------------------------------------
# chat profiles
# ---------------------------------------------------------------------------


def test_chat_profile_for_qwen35():
    resolver = builtin_resolver()
    arch, _ = resolver.resolve_architecture(qwen_metadata())
    rule = resolver.chat_profile(arch_rule=arch, meta=qwen_metadata())
    assert rule is not None
    assert rule.id == "chat-qwen3_5"
    assert rule.supports_thinking_flag is True
    assert rule.thinking_template_variable == "enable_thinking"
    assert rule.default_thinking is False
    assert rule.reasoning_wrappers == [("<think>", "</think>")]
    plain = rule.defaults_for(thinking=False)
    assert (plain.temperature, plain.top_p, plain.top_k, plain.max_new_tokens) == (
        0.7,
        0.8,
        20,
        4096,
    )
    thinking = rule.defaults_for(thinking=True)
    assert (thinking.temperature, thinking.top_p, thinking.top_k) == (0.6, 0.95, 20)
    assert rule.template_context(thinking=False) == {"enable_thinking": False}


def test_generic_chat_profile_is_used_for_unknown_families():
    resolver = builtin_resolver()
    metadata = ModelMetadata(
        identity=ModelIdentity(repo_id="owner/exotic"), model_type="exotic_net"
    )
    rule = resolver.chat_profile(meta=metadata)
    assert rule is not None
    assert rule.id == "chat-default"
    # No stance: the runtime decides from the artifact's own chat template, so an
    # unknown family still gets its reasoning suppressed when its template
    # supports it.
    assert rule.supports_thinking_flag is None
    assert rule.thinking_candidates == ("enable_thinking",)
    assert rule.reasoning_wrappers == [("<think>", "</think>")]
    assert rule.defaults_for(thinking=False).max_new_tokens == 4096
    assert rule.template_context() == {}


def test_a_catch_all_architecture_rule_does_not_hide_the_family_chat_profile():
    """`registry explain` and `yonga chat` must name the same winner.

    ``arch-generic-text-generation`` names ``chat-default``. Returning on that
    name alone made ``explain`` print ``chat-default`` for a Qwen3 model while
    ``yonga chat`` used ``chat-qwen3``, which is the one command whose whole job
    is saying which rule won.
    """
    resolver = builtin_resolver()
    metadata = ModelMetadata(
        identity=ModelIdentity(repo_id="Qwen/Qwen3-0.6B"),
        model_type="qwen3",
        architectures=["Qwen3ForCausalLM"],
    )
    # Qwen3 now has its own measured rule; the catch-all is taken directly, since
    # the property under test is what happens whenever a catch-all is the winner.
    arch = resolver.registry.architecture("arch-generic-text-generation")
    assert arch is not None
    assert arch.chat_profile == "chat-default"

    rule, explanation = resolver.resolve_chat_profile(arch_rule=arch, meta=metadata)

    assert rule is not None
    assert rule.id == "chat-qwen3"
    assert explanation.winner_id == "chat-qwen3"
    # The named profile is still in the trail, with the reason it was considered.
    named = next(c for c in explanation.candidates if c.rule_id == "chat-default")
    assert any("named by architecture rule" in reason for reason in named.reasons)


def test_chat_profile_resolution_is_explained():
    resolver = builtin_resolver()
    arch, _ = resolver.resolve_architecture(qwen_metadata())
    rule, explanation = resolver.resolve_chat_profile(arch_rule=arch)
    assert rule is not None
    assert explanation.winner_id == "chat-qwen3_5"
    assert any(
        "named by architecture rule" in reason for reason in explanation.candidates[0].reasons
    )


# ---------------------------------------------------------------------------
# device compatibility
# ---------------------------------------------------------------------------


def test_device_compatibility_uses_domain_enums():
    resolver = builtin_resolver()
    arch, _ = resolver.resolve_architecture(qwen_metadata())
    entries = {
        entry.device: entry
        for entry in resolver.device_compatibility(
            arch_rule=arch, inventory=inventory_with("CPU", "GPU", "NPU")
        )
    }
    assert entries["CPU"].status is DeviceStatus.SUPPORTED
    assert entries["CPU"].evidence is EvidenceLevel.OFFICIAL
    assert entries["GPU"].status is DeviceStatus.SUPPORTED
    assert entries["NPU"].status is DeviceStatus.UNSUPPORTED
    assert entries["NPU"].evidence is EvidenceLevel.TESTED
    assert entries["NPU"].reason == "current_npu_compiler_graph_limitation"
    assert entries["NPU"].rule_id == "arch-qwen3_5"


def test_out_of_range_openvino_downgrades_a_scoped_claim_to_unknown():
    resolver = builtin_resolver(versions={"openvino": "2028.1.0"})
    arch, _ = resolver.resolve_architecture(qwen_metadata())
    entries = {
        entry.device: entry
        for entry in resolver.device_compatibility(
            arch_rule=arch, inventory=inventory_with("CPU", "GPU", "NPU")
        )
    }
    assert entries["NPU"].status is DeviceStatus.UNKNOWN
    assert entries["NPU"].evidence is EvidenceLevel.UNKNOWN
    assert any("treated as unknown" in note for note in entries["NPU"].notes)


def test_device_without_a_registry_claim_is_unknown():
    resolver = builtin_resolver()
    arch, _ = resolver.resolve_architecture(qwen_metadata())
    entries = {
        entry.device: entry
        for entry in resolver.device_compatibility(
            arch_rule=arch, inventory=inventory_with("CPU", "GPU", "NPU", "OTHER")
        )
    }
    assert entries["OTHER"].status is DeviceStatus.UNKNOWN
    assert entries["OTHER"].reason == "no_registry_rule"


def test_device_known_to_the_registry_but_absent_from_the_host_is_flagged():
    resolver = builtin_resolver()
    arch, _ = resolver.resolve_architecture(qwen_metadata())
    entries = {
        entry.device: entry
        for entry in resolver.device_compatibility(
            arch_rule=arch, inventory=inventory_with("CPU", "GPU")
        )
    }
    assert "NPU" in entries
    assert any("does not expose" in note for note in entries["NPU"].notes)


def test_device_compatibility_without_an_architecture_rule():
    resolver = builtin_resolver()
    entries = resolver.device_compatibility(inventory=inventory_with("CPU", "GPU"))
    assert {entry.device for entry in entries} == {"CPU", "GPU"}
    assert all(entry.status is DeviceStatus.UNKNOWN for entry in entries)


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


def test_explain_produces_a_full_audit_trail():
    resolver = builtin_resolver()
    explanation = resolver.explain(
        meta=qwen_metadata(), inventory=inventory_with("CPU", "GPU", "NPU"), device="gpu"
    )
    assert explanation.repo_id == "nDimensional/Qwen3.5-9B-Uncensored-Safetensors"
    assert explanation.device == "GPU"
    assert explanation.architecture_rule_id == "arch-qwen3_5"
    assert explanation.profile_id == "gpu-int4-asym-g128"
    assert explanation.chat_profile_id == "chat-qwen3_5"
    assert explanation.recommended_device == "GPU"
    assert explanation.host_tags == ["intel.cpu", "intel.gpu.xe2", "intel.npu.lunar_lake"]
    assert explanation.versions["openvino"].startswith("2026.4")
    assert explanation.architecture.candidates
    assert explanation.profile is not None
    assert {entry.device for entry in explanation.device_compatibility} == {"CPU", "GPU", "NPU"}


def test_explain_falls_back_to_the_recommended_device():
    resolver = builtin_resolver()
    explanation = resolver.explain(meta=qwen_metadata())
    assert explanation.device == "GPU"
    assert explanation.profile_id == "gpu-int4-asym-g128"


def test_explain_records_a_missing_profile_as_a_note_instead_of_raising():
    resolver = resolver_for(
        """
        architectures:
          - id: arch-x
            match:
              model_types: [llama]
            recommended_device: npu
            devices:
              npu:
                status: supported
                evidence: reported
        """
    )
    explanation = resolver.explain(meta=llama_metadata())
    assert explanation.profile_id is None
    assert explanation.notes


def test_explain_for_a_derivative_repository_lists_its_candidates():
    resolver = builtin_resolver()
    metadata = ModelMetadata(
        identity=ModelIdentity(repo_id="HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"),
    )
    explanation = resolver.explain(meta=metadata)
    assert [candidate.repo_id for candidate in explanation.derivatives] == [
        "nDimensional/Qwen3.5-9B-Uncensored-Safetensors"
    ]


def test_generic_profile_rules_keep_the_generic_specificity():
    resolver = builtin_resolver()
    _profile, explanation = resolver.resolve_profile(device="gpu")
    specificities = {candidate.specificity for candidate in explanation.candidates}
    assert specificities == {SPECIFICITY_GENERIC}


def test_a_gguf_only_repository_is_identified_through_its_alias():
    """The reference repository has no config.json; the GGUF header identifies it."""
    resolver = builtin_resolver()
    metadata = ModelMetadata(
        identity=ModelIdentity(repo_id="HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"),
        tags=["gguf", "qwen3.5", "qwen"],
        gguf_metadata={"architecture": "qwen35", "context_length": 262144},
    )
    rule, explanation = resolver.resolve_architecture(metadata)
    assert rule is not None
    assert rule.id == "arch-qwen3_5"
    assert any("GGUF metadata" in note for note in explanation.notes)


def test_an_alias_match_scores_below_a_declared_model_type():
    """A rule that recognises the declared identity always outranks an alias match."""
    resolver = resolver_for(
        """
        architectures:
          - id: arch-declared
            priority: 1
            match:
              model_types: [qwen3_5]
          - id: arch-aliased
            priority: 100
            match:
              aliases: [qwen35]
        """
    )
    metadata = ModelMetadata(
        identity=ModelIdentity(repo_id="owner/model"),
        model_type="qwen3_5",
        gguf_metadata={"architecture": "qwen35"},
    )
    rule, explanation = resolver.resolve_architecture(metadata)
    assert rule is not None
    assert rule.id == "arch-declared"
    scored = {candidate.rule_id: candidate for candidate in explanation.candidates}
    assert scored["arch-aliased"].specificity < scored["arch-declared"].specificity
