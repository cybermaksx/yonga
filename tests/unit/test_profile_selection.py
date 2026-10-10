"""Conversion-profile selection by model size: bands, named profiles and explanations."""

from __future__ import annotations

import textwrap
from types import SimpleNamespace

import pytest

from yonga.domain.compatibility import EvidenceLevel, ProfileSelection, ResolutionExplanation
from yonga.domain.models import ModelIdentity, ModelMetadata
from yonga.registry.loader import (
    load_registry_from_documents,
    parse_document,
    validate_registry,
)
from yonga.registry.resolver import RegistryResolver
from yonga.registry.schema import SIZE_NOT_APPLICABLE, subject_from_metadata

HOST_VERSIONS = {
    "openvino": "2026.4.0",
    "openvino_genai": "2026.4.0.0",
    "openvino_tokenizers": "2026.4.0.0",
    "optimum_intel": "2.2.0",
    "transformers": "5.5.4",
    "nncf": "3.4.0",
    "torch": "2.14.0",
    "python": "3.14.4",
}

SAFETENSORS_BASIS = "Hugging Face Hub count from safetensors headers"
NO_COUNT_BASIS = "the Hub reports no safetensors or GGUF parameter total"

#: A GPU default plus a higher-priority profile for models below 2B. The
#: architecture rule names nothing, so the size band decides on its own.
BANDED = """
    schema_version: 2
    profiles:
      - id: gpu-int4-default
        device: gpu
        priority: 10
      - id: gpu-int8-small
        device: gpu
        priority: 20
        evidence: tested
        applies_to:
          parameter_count:
            below: 2B
"""

#: The same profiles, with an architecture rule naming the banded one and an
#: exact-model rule naming a third profile for one repository.
NAMED = """
    schema_version: 2
    architectures:
      - id: arch-llama
        match:
          model_types: [llama]
        profiles:
          gpu: gpu-int8-small
    models:
      - id: model-acme-special
        match:
          repo_ids: [acme/special-llama]
        profiles:
          gpu: gpu-model-pick
    profiles:
      - id: gpu-int4-default
        device: gpu
        priority: 10
      - id: gpu-model-pick
        device: gpu
        priority: 1
      - id: gpu-int8-small
        device: gpu
        priority: 20
        evidence: reported
        applies_to:
          parameter_count:
            below: 2B
"""

#: An architecture rule naming a deprecated profile.
DEPRECATED_NAMED = """
    schema_version: 2
    architectures:
      - id: arch-llama
        match:
          model_types: [llama]
        profiles:
          gpu: gpu-retired
    profiles:
      - id: gpu-int4-default
        device: gpu
        priority: 10
      - id: gpu-retired
        device: gpu
        priority: 50
        status: deprecated
"""


def resolver_for(text: str) -> RegistryResolver:
    """Build a resolver over one inline registry document."""
    document = parse_document(textwrap.dedent(text), origin="bands.yaml")
    return RegistryResolver(load_registry_from_documents([document]), versions=HOST_VERSIONS)


def llama(
    parameter_count: int | None,
    *,
    repo_id: str = "acme/llama",
    dtypes: dict[str, int] | None = None,
    config: dict[str, object] | None = None,
    gguf_metadata: dict[str, object] | None = None,
) -> ModelMetadata:
    """Metadata for a llama-family repository of the given size."""
    if dtypes is None:
        dtypes = {"BF16": parameter_count} if parameter_count is not None else {}
    return ModelMetadata(
        identity=ModelIdentity(repo_id=repo_id),
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        pipeline_tag="text-generation",
        config=config or {},
        parameter_count=parameter_count,
        parameter_dtypes=dtypes,
        gguf_metadata=gguf_metadata or {},
    )


def selection_of(explanation: ResolutionExplanation) -> ProfileSelection:
    """The explanation's profile selection, which every resolution must record."""
    assert explanation.profile_selection is not None
    return explanation.profile_selection


# ---------------------------------------------------------------------------
# size bands
# ---------------------------------------------------------------------------


def test_small_model_gets_the_banded_profile():
    resolver = resolver_for(BANDED)
    profile, explanation = resolver.resolve_profile(device="gpu", meta=llama(1_240_000_000))
    assert profile.id == "gpu-int8-small"
    assert explanation.winner_id == "gpu-int8-small"
    selection = selection_of(explanation)
    assert selection.basis == "size_band"
    assert selection.profile_id == "gpu-int8-small"
    assert selection.rule_id is None
    assert selection.parameter_count == 1_240_000_000
    assert selection.parameter_count_basis == SAFETENSORS_BASIS
    assert selection.band == "below 2B"
    assert selection.band_evidence == EvidenceLevel.TESTED
    assert selection.skipped == []
    assert selection.applies is True
    assert selection.sentence == (
        f"1.24B parameters ({SAFETENSORS_BASIS}) is below 2B: gpu-int8-small is the GPU "
        "profile for that size (evidence: tested)."
    )


def test_large_model_gets_the_unbanded_default_and_says_what_it_skipped():
    resolver = resolver_for(BANDED)
    profile, explanation = resolver.resolve_profile(device="gpu", meta=llama(8_000_000_000))
    assert profile.id == "gpu-int4-default"
    selection = selection_of(explanation)
    assert selection.basis == "device_default"
    assert selection.band is None
    assert selection.band_evidence is None
    assert selection.parameter_count == 8_000_000_000
    assert selection.skipped == [
        "gpu-int8-small is for models below 2B; this model has 8B parameters"
    ]
    assert selection.sentence == (
        "Highest-priority GPU profile. "
        "gpu-int8-small is for models below 2B; this model has 8B parameters."
    )
    rejected = {candidate.rule_id: candidate for candidate in explanation.rejected}
    assert rejected["gpu-int8-small"].reasons[-1] == "8B parameters is not below 2B"


def test_model_exactly_at_the_threshold_belongs_to_the_upper_band():
    """Bands are half-open, so 2,000,000,000 is not ``below 2B``."""
    resolver = resolver_for(BANDED)
    profile, explanation = resolver.resolve_profile(device="gpu", meta=llama(2_000_000_000))
    assert profile.id == "gpu-int4-default"
    assert selection_of(explanation).basis == "device_default"

    just_below, _ = resolver.resolve_profile(device="gpu", meta=llama(1_999_999_999))
    assert just_below.id == "gpu-int8-small"


@pytest.mark.parametrize(
    ("meta", "basis_fragment"),
    [
        pytest.param(llama(None), NO_COUNT_BASIS, id="no-parameter-count"),
        pytest.param(
            llama(800_000_000, dtypes={"I32": 600_000_000, "BF16": 200_000_000}),
            "packed integer weights (I32: 600,000,000)",
            id="packed-integer-weights",
        ),
        pytest.param(
            llama(1_000_000_000, config={"quantization_config": {"quant_method": "awq"}}),
            "quantization_config in config.json",
            id="quantization-config",
        ),
    ],
)
def test_unknown_size_falls_back_to_the_default(meta: ModelMetadata, basis_fragment: str):
    """An unknown size is outside every band: the safe direction for a "small model" band."""
    resolver = resolver_for(BANDED)
    profile, explanation = resolver.resolve_profile(device="gpu", meta=meta)
    assert profile.id == "gpu-int4-default"
    selection = selection_of(explanation)
    assert selection.basis == "device_default"
    assert selection.parameter_count is None
    assert selection.parameter_count_basis is not None
    assert basis_fragment in selection.parameter_count_basis
    assert len(selection.skipped) == 1
    assert selection.skipped[0].startswith(
        "gpu-int8-small is for models below 2B; this model has an unknown size ("
    )
    assert basis_fragment in selection.skipped[0]


def test_no_metadata_at_all_falls_back_to_the_default():
    resolver = resolver_for(BANDED)
    profile, explanation = resolver.resolve_profile(device="gpu")
    assert profile.id == "gpu-int4-default"
    selection = selection_of(explanation)
    assert selection.basis == "device_default"
    assert selection.parameter_count is None
    assert selection.skipped == [
        "gpu-int8-small is for models below 2B; this model has an unknown size "
        "(no parameter count is known)"
    ]


def test_size_does_not_apply_to_a_source_that_keeps_its_quantization():
    resolver = resolver_for(BANDED)
    profile, explanation = resolver.resolve_profile(
        device="gpu", meta=llama(500_000_000), size_applies=False
    )
    assert profile.id == "gpu-int4-default"
    selection = selection_of(explanation)
    assert selection.basis == "device_default"
    assert selection.parameter_count is None
    assert selection.parameter_count_basis == SIZE_NOT_APPLICABLE
    assert selection.skipped == [
        f"gpu-int8-small is for models below 2B; this model has an unknown size "
        f"({SIZE_NOT_APPLICABLE})"
    ]


def test_skipped_ignores_deprecated_banded_profiles():
    resolver = resolver_for(
        """
        schema_version: 2
        profiles:
          - id: gpu-int4-default
            device: gpu
            priority: 10
          - id: gpu-int8-small
            device: gpu
            priority: 20
            status: deprecated
            evidence: tested
            applies_to:
              parameter_count:
                below: 2B
        """
    )
    profile, explanation = resolver.resolve_profile(device="gpu", meta=llama(8_000_000_000))
    assert profile.id == "gpu-int4-default"
    selection = selection_of(explanation)
    assert selection.skipped == []
    assert selection.sentence == "Highest-priority GPU profile."


def test_skipped_ignores_other_devices_and_profiles_this_host_cannot_use():
    """Only a band that could have won here, on this host, is worth a sentence."""
    resolver = resolver_for(
        """
        schema_version: 2
        profiles:
          - id: gpu-int4-default
            device: gpu
            priority: 10
          - id: gpu-future-small
            device: gpu
            priority: 20
            evidence: tested
            constraints:
              openvino: ">=2099"
            applies_to:
              parameter_count:
                below: 2B
          - id: cpu-default
            device: cpu
            priority: 10
          - id: cpu-small
            device: cpu
            priority: 20
            evidence: tested
            applies_to:
              parameter_count:
                below: 2B
        """
    )
    profile, explanation = resolver.resolve_profile(device="gpu", meta=llama(8_000_000_000))
    assert profile.id == "gpu-int4-default"
    selection = selection_of(explanation)
    assert selection.skipped == []
    assert selection.sentence == "Highest-priority GPU profile."


def test_adjacent_bands_each_take_their_own_sizes():
    resolver = resolver_for(
        """
        schema_version: 2
        profiles:
          - id: gpu-default
            device: gpu
            priority: 10
          - id: gpu-small
            device: gpu
            priority: 20
            evidence: tested
            applies_to:
              parameter_count:
                below: 2B
          - id: gpu-medium
            device: gpu
            priority: 20
            evidence: inferred
            applies_to:
              parameter_count:
                at_least: 2B
                below: 8B
        """
    )
    # Equal priorities are not ambiguous: half-open bands that meet never overlap.
    assert validate_registry(resolver.registry) == []
    small, _ = resolver.resolve_profile(device="gpu", meta=llama(1_000_000_000))
    boundary, _ = resolver.resolve_profile(device="gpu", meta=llama(2_000_000_000))
    medium, explanation = resolver.resolve_profile(device="gpu", meta=llama(4_000_000_000))
    large, _ = resolver.resolve_profile(device="gpu", meta=llama(9_000_000_000))
    assert (small.id, boundary.id, medium.id, large.id) == (
        "gpu-small",
        "gpu-medium",
        "gpu-medium",
        "gpu-default",
    )
    selection = selection_of(explanation)
    assert selection.basis == "size_band"
    assert selection.band == "of 2B to below 8B"
    assert selection.band_evidence == EvidenceLevel.INFERRED
    assert selection.skipped == ["gpu-small is for models below 2B; this model has 4B parameters"]


def test_a_band_adds_no_specificity_so_priority_decides():
    """A banded profile below the default's priority never wins on size alone."""
    resolver = resolver_for(
        """
        schema_version: 2
        profiles:
          - id: gpu-default
            device: gpu
            priority: 30
          - id: gpu-small
            device: gpu
            priority: 20
            evidence: tested
            applies_to:
              parameter_count:
                below: 2B
        """
    )
    profile, explanation = resolver.resolve_profile(device="gpu", meta=llama(1_000_000_000))
    assert profile.id == "gpu-default"
    candidates = {candidate.rule_id: candidate for candidate in explanation.candidates}
    assert candidates["gpu-small"].matched is True
    assert candidates["gpu-small"].specificity == candidates["gpu-default"].specificity
    assert selection_of(explanation).basis == "device_default"


def test_skipped_omits_a_banded_profile_that_does_not_apply_to_the_model_anyway():
    """A qwen-only small-model profile did not lose to a llama model's size."""
    resolver = resolver_for(
        """
        schema_version: 2
        profiles:
          - id: gpu-default
            device: gpu
            priority: 10
          - id: gpu-qwen-small
            device: gpu
            priority: 20
            evidence: tested
            applies_to:
              model_types: [qwen3]
              parameter_count:
                below: 2B
        """
    )
    profile, explanation = resolver.resolve_profile(device="gpu", meta=llama(8_000_000_000))
    assert profile.id == "gpu-default"
    assert selection_of(explanation).skipped == []


def test_size_band_sentence_reads_naturally_for_a_band_with_a_lower_bound():
    resolver = resolver_for(
        """
        schema_version: 2
        profiles:
          - id: gpu-default
            device: gpu
            priority: 10
          - id: gpu-medium
            device: gpu
            priority: 20
            evidence: inferred
            applies_to:
              parameter_count:
                at_least: 2B
                below: 8B
        """
    )
    profile, explanation = resolver.resolve_profile(device="gpu", meta=llama(4_000_000_000))
    assert profile.id == "gpu-medium"
    assert " is of " not in selection_of(explanation).sentence


def test_size_band_sentence_does_not_spell_a_count_as_its_threshold():
    resolver = resolver_for(BANDED)
    profile, explanation = resolver.resolve_profile(device="gpu", meta=llama(1_996_000_000))
    assert profile.id == "gpu-int8-small"
    assert not selection_of(explanation).sentence.startswith("2B parameters ")


# ---------------------------------------------------------------------------
# named profiles
# ---------------------------------------------------------------------------


def test_architecture_rule_naming_a_banded_profile_wins_inside_the_band():
    resolver = resolver_for(NAMED)
    meta = llama(1_000_000_000)
    arch, _ = resolver.resolve_architecture(meta)
    assert arch is not None
    profile, explanation = resolver.resolve_profile(device="gpu", arch_rule=arch, meta=meta)
    assert profile.id == "gpu-int8-small"
    selection = selection_of(explanation)
    assert selection.basis == "named_by_architecture_rule"
    assert selection.rule_id == "arch-llama"
    assert selection.band == "below 2B"
    assert selection.band_evidence == EvidenceLevel.REPORTED
    assert selection.parameter_count == 1_000_000_000
    # Scoring alone picks the same profile, so there is no overridden choice to name.
    assert selection.automatic_profile_id is None
    assert selection.sentence == (
        f"Named by architecture rule arch-llama for GPU. The model has 1B parameters "
        f"({selection.parameter_count_basis})."
    )


def test_architecture_rule_naming_a_banded_profile_falls_through_outside_the_band():
    """A named profile is a preference: its own band still has to match."""
    resolver = resolver_for(NAMED)
    meta = llama(8_000_000_000)
    arch, _ = resolver.resolve_architecture(meta)
    profile, explanation = resolver.resolve_profile(device="gpu", arch_rule=arch, meta=meta)
    assert profile.id == "gpu-int4-default"
    selection = selection_of(explanation)
    assert selection.basis == "device_default"
    assert selection.rule_id is None
    assert selection.skipped == [
        "gpu-int8-small is for models below 2B; this model has 8B parameters"
    ]
    named = explanation.candidates[0]
    assert named.rule_id == "gpu-int8-small"
    assert named.matched is False
    assert named.reasons[0] == "named by architecture rule arch-llama"
    # Falling out of a band is ordinary, not an anomaly worth a note.
    assert explanation.notes == []


def test_named_deprecated_profile_falls_through_with_a_note():
    resolver = resolver_for(DEPRECATED_NAMED)
    meta = llama(1_000_000_000)
    arch, _ = resolver.resolve_architecture(meta)
    profile, explanation = resolver.resolve_profile(device="gpu", arch_rule=arch, meta=meta)
    assert profile.id == "gpu-int4-default"
    assert selection_of(explanation).basis == "device_default"
    assert explanation.notes == [
        "architecture rule arch-llama names deprecated profile gpu-retired; "
        "the GPU profiles were ranked instead."
    ]


def test_model_rule_beats_the_architecture_rule():
    resolver = resolver_for(NAMED)
    meta = llama(1_000_000_000, repo_id="acme/special-llama")
    arch, _ = resolver.resolve_architecture(meta)
    model_rule, _ = resolver.resolve_model_rule(meta)
    assert arch is not None and model_rule is not None
    profile, explanation = resolver.resolve_profile(
        device="gpu", arch_rule=arch, model_rule=model_rule, meta=meta
    )
    assert profile.id == "gpu-model-pick"
    selection = selection_of(explanation)
    assert selection.basis == "named_by_model_rule"
    assert selection.rule_id == "model-acme-special"
    assert selection.band is None
    assert selection.band_evidence is None
    # The banded profile lost to the model rule, not to the model's size.
    assert selection.skipped == []
    # Without the model rule, the architecture rule's pick would have won.
    assert selection.automatic_profile_id == "gpu-int8-small"
    assert selection.sentence == (
        f"Named by model rule model-acme-special for GPU. The model has 1B parameters "
        f"({selection.parameter_count_basis}); without the rule it would be gpu-int8-small, "
        "named by architecture rule arch-llama."
    )


def test_a_model_rule_sentence_names_the_size_band_it_bypassed():
    """CLI_SPEC section 6: the sentence carries the size and any band the rule overrode."""
    resolver = resolver_for(NAMED)
    meta = llama(1_000_000_000, repo_id="acme/special-llama")
    model_rule, _ = resolver.resolve_model_rule(meta)
    assert model_rule is not None
    # No architecture rule: without the model rule, the size band decides.
    profile, explanation = resolver.resolve_profile(device="gpu", model_rule=model_rule, meta=meta)
    assert profile.id == "gpu-model-pick"
    selection = selection_of(explanation)
    assert selection.automatic_profile_id == "gpu-int8-small"
    assert selection.sentence.endswith(
        "; without the rule it would be gpu-int8-small, the GPU profile for models below 2B."
    )
    assert "The model has 1B parameters" in selection.sentence
    # Working the alternative out leaves no trace in the explanation.
    assert [candidate.rule_id for candidate in explanation.candidates] == ["gpu-model-pick"]


def test_a_named_profile_sentence_says_when_the_size_is_unknown():
    resolver = resolver_for(NAMED)
    meta = llama(None, repo_id="acme/special-llama")
    model_rule, _ = resolver.resolve_model_rule(meta)
    _, explanation = resolver.resolve_profile(device="gpu", model_rule=model_rule, meta=meta)
    selection = selection_of(explanation)
    assert f"The model has an unknown size ({NO_COUNT_BASIS})" in selection.sentence
    assert selection.sentence.endswith("gpu-int4-default, the highest-priority GPU profile.")


# ---------------------------------------------------------------------------
# explicit profiles
# ---------------------------------------------------------------------------


def test_explicit_profile_records_the_automatic_choice_when_it_differs():
    resolver = resolver_for(BANDED)
    profile, explanation = resolver.resolve_profile(
        device="gpu", meta=llama(1_000_000_000), explicit_profile_id="gpu-int4-default"
    )
    assert profile.id == "gpu-int4-default"
    selection = selection_of(explanation)
    assert selection.basis == "explicit"
    assert selection.automatic_profile_id == "gpu-int8-small"
    assert selection.sentence == (
        "Requested with --profile. The automatic choice would be gpu-int8-small."
    )
    assert explanation.notes == []
    # Working out the automatic choice leaves no trace in the explanation.
    assert [candidate.rule_id for candidate in explanation.candidates] == ["gpu-int4-default"]
    assert explanation.winner_id == "gpu-int4-default"


def test_explicit_profile_lists_nothing_as_skipped_for_size():
    """Under --profile the banded profiles lost to the request, not to the model's size."""
    resolver = resolver_for(BANDED)
    profile, explanation = resolver.resolve_profile(
        device="gpu", meta=llama(8_000_000_000), explicit_profile_id="gpu-int4-default"
    )
    assert profile.id == "gpu-int4-default"
    selection = selection_of(explanation)
    assert selection.basis == "explicit"
    assert selection.automatic_profile_id is None
    assert selection.skipped == []
    assert selection.sentence == "Requested with --profile."


def test_explicit_profile_matching_the_automatic_choice_records_none():
    resolver = resolver_for(BANDED)
    _, explanation = resolver.resolve_profile(
        device="gpu", meta=llama(1_000_000_000), explicit_profile_id="gpu-int8-small"
    )
    selection = selection_of(explanation)
    assert selection.basis == "explicit"
    assert selection.automatic_profile_id is None
    assert selection.sentence == "Requested with --profile."
    assert selection.band == "below 2B"
    assert selection.band_evidence == EvidenceLevel.TESTED


def test_explicit_banded_profile_outside_its_band_is_used_with_a_note():
    resolver = resolver_for(BANDED)
    profile, explanation = resolver.resolve_profile(
        device="gpu", meta=llama(8_000_000_000), explicit_profile_id="gpu-int8-small"
    )
    assert profile.id == "gpu-int8-small"
    selection = selection_of(explanation)
    assert selection.basis == "explicit"
    assert selection.automatic_profile_id == "gpu-int4-default"
    assert selection.skipped == []
    assert explanation.notes == [
        "Profile gpu-int8-small is meant for models below 2B; this model has 8B parameters. "
        "It is used because it was requested explicitly."
    ]


def test_explicit_banded_profile_with_an_unknown_size_is_used_with_a_note():
    resolver = resolver_for(BANDED)
    profile, explanation = resolver.resolve_profile(
        device="gpu", meta=llama(None), explicit_profile_id="gpu-int8-small"
    )
    assert profile.id == "gpu-int8-small"
    assert explanation.notes == [
        f"Profile gpu-int8-small is meant for models below 2B; this model has an unknown size "
        f"({NO_COUNT_BASIS}). It is used because it was requested explicitly."
    ]


def test_explicit_deprecated_profile_is_used_with_a_note():
    resolver = resolver_for(DEPRECATED_NAMED)
    profile, explanation = resolver.resolve_profile(
        device="gpu", meta=llama(1_000_000_000), explicit_profile_id="gpu-retired"
    )
    assert profile.id == "gpu-retired"
    selection = selection_of(explanation)
    assert selection.basis == "explicit"
    assert selection.automatic_profile_id == "gpu-int4-default"
    assert explanation.notes == [
        "Profile gpu-retired is deprecated; it is used because it was requested explicitly."
    ]


def test_explicit_profile_without_an_automatic_alternative():
    """No profile targets the device, so there is no automatic choice to report."""
    resolver = resolver_for(BANDED)
    profile, explanation = resolver.resolve_profile(
        device="npu", meta=llama(1_000_000_000), explicit_profile_id="gpu-int4-default"
    )
    assert profile.id == "gpu-int4-default"
    selection = selection_of(explanation)
    assert selection.basis == "explicit"
    assert selection.automatic_profile_id is None
    assert any("targets GPU, not the requested NPU" in note for note in explanation.notes)


# ---------------------------------------------------------------------------
# subject_from_metadata and ModelMetadata.size_for_matching
# ---------------------------------------------------------------------------


def test_subject_takes_the_count_and_basis_from_size_for_matching():
    subject = subject_from_metadata(llama(1_500_000_000))
    assert subject.parameter_count == 1_500_000_000
    assert subject.parameter_count_basis == SAFETENSORS_BASIS


def test_subject_leaves_the_size_unknown_when_it_does_not_apply():
    subject = subject_from_metadata(llama(1_500_000_000), size_applies=False)
    assert subject.parameter_count is None
    assert subject.parameter_count_basis == SIZE_NOT_APPLICABLE


def test_subject_without_metadata_has_no_size():
    subject = subject_from_metadata(None)
    assert subject.parameter_count is None
    assert subject.parameter_count_basis is None


def test_subject_from_an_object_without_size_for_matching_has_no_size():
    """Metadata-like objects predating the method still build a subject."""
    subject = subject_from_metadata(SimpleNamespace(model_type="llama", parameter_count=10))
    assert subject.model_type == "llama"
    assert subject.parameter_count is None
    assert subject.parameter_count_basis is None


def test_subject_ignores_a_non_integer_count():
    meta = SimpleNamespace(size_for_matching=lambda: (True, "a boolean is not a count"))
    subject = subject_from_metadata(meta)
    assert subject.parameter_count is None
    assert subject.parameter_count_basis == "a boolean is not a count"


def test_subject_treats_a_zero_count_as_unknown():
    meta = SimpleNamespace(size_for_matching=lambda: (0, "an empty checkpoint"))
    subject = subject_from_metadata(meta)
    assert subject.parameter_count is None
    assert subject.parameter_count_basis == "an empty checkpoint"


def test_size_for_matching_without_a_count():
    assert llama(None).size_for_matching() == (None, NO_COUNT_BASIS)


def test_size_for_matching_treats_a_zero_count_as_no_count():
    assert llama(0, dtypes={}).size_for_matching() == (None, NO_COUNT_BASIS)


def test_size_for_matching_keeps_small_integer_buffers():
    """Position ids and similar integer buffers are far below 1% and do not mean packing."""
    meta = llama(1_000_004_096, dtypes={"BF16": 1_000_000_000, "I64": 4_096})
    assert meta.size_for_matching() == (1_000_004_096, SAFETENSORS_BASIS)


def test_size_for_matching_keeps_integers_at_exactly_one_percent():
    meta = llama(1_000_000_000, dtypes={"BF16": 990_000_000, "I64": 10_000_000})
    assert meta.size_for_matching() == (1_000_000_000, SAFETENSORS_BASIS)


def test_size_for_matching_refuses_packed_integer_weights():
    meta = llama(
        1_000_000_000,
        dtypes={"BF16": 400_000_000, "I32": 500_000_000, "U8": 100_000_000},
    )
    count, basis = meta.size_for_matching()
    assert count is None
    assert basis == (
        "the checkpoint stores packed integer weights (I32: 500,000,000), "
        "so the Hub count understates the model"
    )


def test_size_for_matching_treats_lowercase_integer_dtypes_as_packed():
    meta = llama(1_000_000_000, dtypes={"bf16": 500_000_000, "i32": 500_000_000})
    count, basis = meta.size_for_matching()
    assert count is None
    assert "i32: 500,000,000" in basis


def test_size_for_matching_refuses_a_quantization_config():
    meta = llama(1_000_000_000, config={"quantization_config": {"quant_method": "gptq"}})
    assert meta.size_for_matching() == (
        None,
        "the checkpoint is already quantized (quantization_config in config.json)",
    )


def test_size_for_matching_names_gguf_metadata_as_the_basis():
    meta = llama(1_100_000_000, dtypes={}, gguf_metadata={"architecture": "llama"})
    assert meta.size_for_matching() == (1_100_000_000, "Hugging Face Hub count from GGUF metadata")


def test_size_for_matching_prefers_safetensors_when_both_are_present():
    meta = llama(
        1_100_000_000,
        dtypes={"BF16": 1_100_000_000},
        gguf_metadata={"architecture": "llama"},
    )
    assert meta.size_for_matching() == (1_100_000_000, SAFETENSORS_BASIS)
