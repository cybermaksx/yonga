"""Schema-level tests for the compatibility registry records."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from yonga.domain.compatibility import DeviceStatus, EvidenceLevel
from yonga.domain.models import ModelFormat, ModelIdentity, ModelMetadata, SourceRelationship
from yonga.domain.plans import ConversionProfile, QuantizationSettings
from yonga.registry.schema import (
    SPECIFICITY_ARCHITECTURE,
    SPECIFICITY_ARCHITECTURE_FAMILY,
    SPECIFICITY_GENERIC,
    SPECIFICITY_REPO_ID,
    SPECIFICITY_TASK,
    ArchitectureRule,
    ChatProfileRule,
    DerivativeCandidate,
    DerivativeRule,
    DeviceSupport,
    GenerationDefaults,
    IssueClassification,
    IssueMatch,
    IssueRule,
    MatchSubject,
    ProfileRule,
    Registry,
    RegistryDocument,
    RepairAction,
    RepairRule,
    RuleMatch,
    RuleStatus,
    SourceKind,
    SourceRef,
    TrustPolicy,
    constraint_problems,
    subject_from_metadata,
)


def test_source_ref_renders_mapping_and_evidence():
    ref = SourceRef(kind=SourceKind.MODEL_CARD, url="https://example.invalid/card")
    assert ref.as_mapping() == {"kind": "model_card", "url": "https://example.invalid/card"}
    evidence = ref.as_evidence()
    assert evidence.kind == "model_card"
    assert evidence.note is None


def test_identity_criteria_are_an_or_group():
    match = RuleMatch(model_types=["qwen3_5"], architectures=["Qwen3_5ForCausalLM"])
    by_type = match.evaluate(MatchSubject(model_type="qwen3_5"))
    by_class = match.evaluate(MatchSubject(architectures=("Qwen3_5ForCausalLM",)))
    assert by_type.matched
    assert by_class.matched
    assert not match.evaluate(MatchSubject(model_type="llama")).matched


def test_criteria_of_different_kinds_are_combined_with_and():
    match = RuleMatch(model_types=["qwen3_5"], devices=["npu"])
    assert match.evaluate(MatchSubject(model_type="qwen3_5", device="NPU")).matched
    rejected = match.evaluate(MatchSubject(model_type="qwen3_5", device="GPU"))
    assert not rejected.matched
    assert "device GPU is not listed" in rejected.reasons


def test_specificity_ladder():
    assert RuleMatch(repo_ids=["a/b"]).static_specificity() == SPECIFICITY_REPO_ID
    family = RuleMatch(model_types=["qwen3_5"], devices=["npu"])
    assert family.static_specificity() == SPECIFICITY_ARCHITECTURE_FAMILY
    assert RuleMatch(model_types=["qwen3_5"]).static_specificity() == SPECIFICITY_ARCHITECTURE
    task = RuleMatch(tasks=["text-generation"], devices=["gpu"])
    assert task.static_specificity() == SPECIFICITY_TASK
    assert RuleMatch(devices=["gpu"]).static_specificity() == SPECIFICITY_GENERIC
    assert RuleMatch().static_specificity() == SPECIFICITY_GENERIC
    assert RuleMatch().is_generic


def test_device_shorthand_and_normalisation():
    match = RuleMatch.model_validate({"device": "npu"})
    assert match.devices == ["NPU"]
    assert RuleMatch(devices=["GPU.0", "gpu"]).devices == ["GPU"]


def test_match_subject_normalises_device():
    assert MatchSubject(device="gpu.1").device == "GPU"
    assert MatchSubject(device=None).device is None


def test_host_tag_matching():
    match = RuleMatch(model_types=["qwen3_5"], host_tags=["intel.npu.lunar_lake"])
    subject = MatchSubject(model_type="qwen3_5", host_tags=("intel.npu.lunar_lake",))
    assert match.evaluate(subject).matched
    other = MatchSubject(model_type="qwen3_5", host_tags=("intel.npu.arrow_lake",))
    assert not match.evaluate(other).matched


def test_profile_rule_converts_to_domain_conversion_profile():
    rule = ProfileRule(
        id="gpu-int4-asym-g128",
        device="gpu",
        summary="GPU profile",
        quantization=QuantizationSettings(
            weight_format="int4", symmetric=False, ratio=1.0, group_size=128
        ),
        runtime={"attention_backend": "SDPA"},
        constraints={"openvino": ">=2026.2"},
        rationale="because",
        sources=[SourceRef(kind=SourceKind.MODEL_CARD, url="https://example.invalid")],
    )
    profile = rule.to_conversion_profile()
    assert isinstance(profile, ConversionProfile)
    assert profile.id == "gpu-int4-asym-g128"
    assert profile.device == "GPU"
    assert profile.quantization.weight_format == "int4"
    assert profile.quantization.symmetric is False
    assert profile.quantization.group_size == 128
    assert profile.runtime_properties == {"ATTENTION_BACKEND": "SDPA"}
    assert profile.constraints == {"openvino": ">=2026.2"}
    assert profile.sources == [{"kind": "model_card", "url": "https://example.invalid"}]


def test_profile_rule_rejects_empty_device():
    with pytest.raises(ValidationError):
        ProfileRule(id="broken", device="   ")


def test_architecture_rule_uses_domain_enums_and_normalises_keys():
    rule = ArchitectureRule(
        id="arch-x",
        match=RuleMatch(model_types=["x"]),
        devices={
            "gpu": DeviceSupport(status=DeviceStatus.SUPPORTED, evidence=EvidenceLevel.OFFICIAL),
            "npu": DeviceSupport(
                status=DeviceStatus.UNSUPPORTED, evidence=EvidenceLevel.TESTED, reason="graph"
            ),
        },
        recommended_device="gpu",
        profiles={"gpu": "p-gpu"},
    )
    assert set(rule.devices) == {"GPU", "NPU"}
    gpu = rule.device_support("GPU.0")
    assert gpu is not None
    assert gpu.status is DeviceStatus.SUPPORTED
    assert gpu.evidence is EvidenceLevel.OFFICIAL
    npu = rule.device_support("npu")
    assert npu is not None
    assert npu.status is DeviceStatus.UNSUPPORTED
    assert rule.recommended_device == "GPU"
    assert rule.profile_id_for("gpu") == "p-gpu"
    assert rule.profile_id_for("cpu") is None


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        ArchitectureRule.model_validate({"id": "arch-x", "devicez": {}})


def test_issue_match_signature_hit_respects_case_sensitivity():
    strict = IssueMatch(
        exception_contains=["Unsupported element type: STRiNG"], case_sensitive=True
    )
    assert strict.signature_hit("... Unsupported element type: STRiNG") is not None
    assert strict.signature_hit("... unsupported element type: string") is None
    loose = IssueMatch(exception_contains=["No space left on device"])
    assert loose.signature_hit("errno 28: no space left on device") is not None


def test_issue_match_returns_the_longest_hit():
    match = IssueMatch(
        exception_contains=["OneShotBufferizeVPU2VPUIP failed", "OneShotBufferizeVPU2VPUIP"]
    )
    assert match.signature_hit("OneShotBufferizeVPU2VPUIP failed") == (
        "OneShotBufferizeVPU2VPUIP failed"
    )


def test_issue_match_stage_filter():
    match = IssueMatch(stages=["compile"])
    assert match.stage_matches("compile")
    assert not match.stage_matches("export")
    assert not match.stage_matches(None)
    assert IssueMatch().stage_matches(None)


def test_derivative_trust_policies_drive_confirmation():
    suggest = DerivativeCandidate(repo_id="a/b", trust=TrustPolicy.SUGGEST)
    never = DerivativeCandidate(repo_id="a/b", trust=TrustPolicy.NEVER_AUTO)
    auto = DerivativeCandidate(repo_id="a/b", trust=TrustPolicy.AUTO_IF_TRUSTED)
    assert suggest.requires_confirmation
    assert never.requires_confirmation
    assert not auto.requires_confirmation
    assert suggest.relationship is SourceRelationship.CONVERTED


def test_derivative_candidate_accepts_sources_alias_for_evidence():
    candidate = DerivativeCandidate.model_validate(
        {
            "repo_id": "a/b",
            "relationship": "lossless_conversion",
            "sources": [{"kind": "model_card", "url": "https://example.invalid"}],
        }
    )
    assert candidate.relationship is SourceRelationship.LOSSLESS_CONVERSION
    assert [item.kind for item in candidate.evidence] == [SourceKind.MODEL_CARD]
    assert candidate.source_evidence()[0].url == "https://example.invalid"


def test_derivative_rule_matches_repo_case_insensitively():
    rule = DerivativeRule(id="d", requested_repo="Owner/Repo")
    assert rule.matches_repo("owner/repo")
    assert not rule.matches_repo("owner/other")
    assert rule.match_block.static_specificity() == SPECIFICITY_REPO_ID


def test_chat_profile_defaults_merge_only_declared_fields():
    rule = ChatProfileRule(
        id="chat-x",
        supports_thinking_flag=True,
        thinking_template_variable="enable_thinking",
        default_thinking=False,
        reasoning_wrappers=[("<think>", "</think>")],
        defaults=GenerationDefaults(temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=4096),
        thinking_defaults=GenerationDefaults(temperature=0.6, top_p=0.95, top_k=20),
    )
    plain = rule.defaults_for(thinking=False)
    thinking = rule.defaults_for(thinking=True)
    assert plain.temperature == 0.7
    assert thinking.temperature == 0.6
    assert thinking.top_p == 0.95
    assert thinking.max_new_tokens == 4096
    assert rule.template_context(thinking=True) == {"enable_thinking": True}
    assert rule.template_context() == {"enable_thinking": False}


def test_chat_profile_without_template_variable_is_rejected():
    with pytest.raises(ValidationError):
        ChatProfileRule(id="chat-broken", supports_thinking_flag=True)


def test_chat_profile_without_thinking_returns_no_template_context():
    rule = ChatProfileRule(id="chat-plain", reasoning_wrappers=[("<think>", "</think>")])
    assert rule.template_context(thinking=True) == {}
    assert rule.reasoning_wrappers == [("<think>", "</think>")]


def test_registry_indexes_every_rule_kind():
    registry = Registry(
        architectures=[ArchitectureRule(id="a")],
        profiles=[ProfileRule(id="p", device="cpu")],
        issues=[
            IssueRule(
                id="i",
                match=IssueMatch(exception_contains=["boom"]),
                classification=IssueClassification(code="x.y", category="unknown_runtime_error"),
            )
        ],
        repairs=[
            RepairRule(id="r", repair=RepairAction(implementation="tokenizer_string_precision"))
        ],
        derivatives=[DerivativeRule(id="d", requested_repo="a/b")],
        chat=[ChatProfileRule(id="c")],
        origins={"a": "builtin:architectures.yaml"},
    )
    assert registry.architecture("a") is not None
    assert registry.profile("p") is not None
    assert registry.issue("i") is not None
    assert registry.repair("r") is not None
    assert registry.derivative("d") is not None
    assert registry.chat_profile("c") is not None
    assert registry.architecture("p") is None
    assert registry.rule("missing") is None
    assert registry.origin_of("a") == "builtin:architectures.yaml"
    assert registry.profiles_for_device("CPU")[0].id == "p"
    assert registry.derivatives_for("a/b")[0].id == "d"
    assert registry.counts()["architectures"] == 1
    assert "a" in registry.rule_ids


def test_rule_status_controls_usability():
    assert RuleStatus.ACTIVE.is_usable
    assert RuleStatus.EXPERIMENTAL.is_usable
    assert not RuleStatus.DEPRECATED.is_usable


def test_constraint_problems_reports_bad_specifiers_and_keys():
    problems = constraint_problems({"openvino": "not-a-specifier", "banana": ">=1"})
    assert any("unparsable version specifier" in item for item in problems)
    assert any("unknown constraint key" in item for item in problems)
    assert constraint_problems({"openvino": ">=2026.2,<2027"}) == []


def test_subject_from_metadata_reads_domain_metadata():
    metadata = ModelMetadata(
        identity=ModelIdentity(repo_id="owner/model"),
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
        pipeline_tag="image-text-to-text",
    )
    subject = subject_from_metadata(metadata, device="npu", host_tags=["intel.npu"])
    assert subject.repo_id == "owner/model"
    assert subject.model_type == "qwen3_5"
    assert subject.architectures == ("Qwen3_5ForConditionalGeneration",)
    assert subject.task == "image-text-to-text"
    assert subject.device == "NPU"
    assert subject.host_tags == ("intel.npu",)


def test_subject_from_metadata_tolerates_none():
    subject = subject_from_metadata(None, device="cpu")
    assert subject.repo_id is None
    assert subject.architectures == ()
    assert subject.device == "CPU"


def test_registry_document_accepts_chat_profiles_alias():
    document = RegistryDocument.model_validate({"chat_profiles": [{"id": "c"}]})
    assert [rule.id for rule in document.chat] == ["c"]
    assert [rule.id for rule in document.rules()] == ["c"]


def test_model_format_default_on_profile_rule():
    rule = ProfileRule(id="p", device="cpu")
    assert rule.format is ModelFormat.OPENVINO_IR
