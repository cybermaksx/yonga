"""Unit tests for yonga.domain.models."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError as PydanticValidationError

from yonga.domain.models import (
    FormatObservation,
    ModelFormat,
    ModelIdentity,
    ModelMetadata,
    PipelineKind,
    RepoFile,
    ResolvedSource,
    SourceEvidence,
    SourceRelationship,
    infer_pipeline_kind,
)

GIB = 1024**3


def _metadata(**kwargs: Any) -> ModelMetadata:
    kwargs.setdefault("identity", ModelIdentity(repo_id="acme/demo"))
    return ModelMetadata(**kwargs)


class TestModelIdentity:
    def test_parse_repo_and_revision(self) -> None:
        identity = ModelIdentity.parse("a/b@rev")
        assert identity.repo_id == "a/b"
        assert identity.revision == "rev"

    def test_parse_without_revision(self) -> None:
        identity = ModelIdentity.parse("  HauhauCS/Qwen3.5-9B-Uncensored  ")
        assert identity.repo_id == "HauhauCS/Qwen3.5-9B-Uncensored"
        assert identity.revision is None

    def test_parse_with_empty_revision_is_unpinned(self) -> None:
        assert ModelIdentity.parse("a/b@").revision is None

    def test_parse_sha_revision(self) -> None:
        identity = ModelIdentity.parse("a/b@0123456789abcdef0123456789abcdef01234567")
        assert identity.revision == "0123456789abcdef0123456789abcdef01234567"

    def test_parse_rejects_blank(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            ModelIdentity.parse("   ")

    def test_empty_repo_id_is_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            ModelIdentity(repo_id="  ")

    def test_namespace_and_name(self) -> None:
        identity = ModelIdentity(repo_id="nDimensional/Qwen3.5-9B-Safetensors")
        assert identity.namespace == "nDimensional"
        assert identity.name == "Qwen3.5-9B-Safetensors"

    def test_bare_name_has_no_namespace(self) -> None:
        identity = ModelIdentity(repo_id="gpt2")
        assert identity.namespace is None
        assert identity.name == "gpt2"
        assert identity.slug == "gpt2"

    def test_slug_replaces_namespace_separator(self) -> None:
        assert ModelIdentity(repo_id="a/b").slug == "a--b"

    def test_slug_is_filesystem_safe(self) -> None:
        slug = ModelIdentity(repo_id="we!rd/na me:v2").slug
        assert slug == "we_rd--na_me_v2"
        assert all(character.isalnum() or character in "._-" for character in slug)

    def test_slug_keeps_dots_and_dashes(self) -> None:
        assert ModelIdentity(repo_id="OpenVINO/Qwen3.5-9B-int4-ov").slug == (
            "OpenVINO--Qwen3.5-9B-int4-ov"
        )

    def test_str_round_trips_through_parse(self) -> None:
        for raw in ("a/b", "a/b@rev"):
            assert str(ModelIdentity.parse(raw)) == raw

    def test_with_revision_returns_a_new_pinned_identity(self) -> None:
        original = ModelIdentity(repo_id="a/b")
        pinned = original.with_revision("deadbeef")
        assert pinned.revision == "deadbeef"
        assert original.revision is None
        assert pinned is not original

    def test_is_frozen(self) -> None:
        identity = ModelIdentity(repo_id="a/b")
        with pytest.raises(PydanticValidationError):
            identity.repo_id = "c/d"

    def test_is_hashable_and_compares_by_value(self) -> None:
        first = ModelIdentity(repo_id="a/b", revision="rev")
        second = ModelIdentity(repo_id="a/b", revision="rev")
        other = ModelIdentity(repo_id="a/b")
        assert first == second
        assert hash(first) == hash(second)
        assert {first, second, other} == {first, other}
        assert {first: "value"}[second] == "value"

    def test_blank_revision_normalises_to_none(self) -> None:
        assert ModelIdentity(repo_id="a/b", revision="   ").revision is None


class TestRepoFile:
    def test_suffix_and_name(self) -> None:
        entry = RepoFile(path="subdir/model-00001-of-00004.safetensors", size=10, lfs=True)
        assert entry.suffix == ".safetensors"
        assert entry.name == "model-00001-of-00004.safetensors"

    def test_file_without_suffix(self) -> None:
        assert RepoFile(path="README").suffix == ""


class TestWeightBytes:
    def test_prefers_safetensors_over_other_formats(self) -> None:
        metadata = _metadata(
            formats=[
                FormatObservation(
                    format=ModelFormat.PYTORCH_BIN, total_bytes=9 * GIB, complete=True
                ),
                FormatObservation(
                    format=ModelFormat.SAFETENSORS, total_bytes=18 * GIB, complete=True
                ),
                FormatObservation(format=ModelFormat.GGUF, total_bytes=5 * GIB, complete=True),
            ]
        )
        assert metadata.weight_bytes == 18 * GIB
        assert metadata.primary_format is ModelFormat.SAFETENSORS

    def test_falls_back_through_the_preference_order(self) -> None:
        metadata = _metadata(
            formats=[
                FormatObservation(format=ModelFormat.GGUF, total_bytes=5 * GIB, complete=True),
                FormatObservation(
                    format=ModelFormat.OPENVINO_IR, total_bytes=4 * GIB, complete=True
                ),
            ]
        )
        assert metadata.weight_bytes == 5 * GIB
        assert metadata.primary_format is ModelFormat.GGUF

    def test_complete_format_beats_a_preferred_incomplete_one(self) -> None:
        metadata = _metadata(
            formats=[
                FormatObservation(
                    format=ModelFormat.SAFETENSORS, total_bytes=2 * GIB, complete=False
                ),
                FormatObservation(
                    format=ModelFormat.PYTORCH_BIN, total_bytes=9 * GIB, complete=True
                ),
            ]
        )
        assert metadata.weight_bytes == 9 * GIB
        assert metadata.primary_format is ModelFormat.PYTORCH_BIN

    def test_incomplete_only_is_still_reported(self) -> None:
        metadata = _metadata(
            formats=[
                FormatObservation(
                    format=ModelFormat.SAFETENSORS, total_bytes=2 * GIB, complete=False
                )
            ]
        )
        assert metadata.weight_bytes == 2 * GIB

    def test_unknown_sizes_report_zero(self) -> None:
        metadata = _metadata(
            formats=[
                FormatObservation(
                    format=ModelFormat.SAFETENSORS,
                    files=["model.safetensors"],
                    total_bytes=0,
                    complete=True,
                )
            ]
        )
        assert metadata.weight_bytes == 0
        assert metadata.primary_format is ModelFormat.SAFETENSORS

    def test_no_observations_report_zero_and_unknown(self) -> None:
        metadata = _metadata()
        assert metadata.weight_bytes == 0
        assert metadata.primary_format is ModelFormat.UNKNOWN

    def test_has_format(self) -> None:
        metadata = _metadata(
            formats=[FormatObservation(format=ModelFormat.GGUF, total_bytes=1, complete=True)]
        )
        assert metadata.has_format(ModelFormat.GGUF)
        assert not metadata.has_format(ModelFormat.SAFETENSORS)


class TestPrimaryVersusRepositoryBytes:
    """``weight_bytes`` sizes one model; ``repo_weight_bytes`` sizes the repository.

    The two diverge whenever a repository publishes several interchangeable
    copies of the same model — the GGUF one-file-per-quantization convention —
    and preflight must never be handed the larger of the two.
    """

    def test_an_unset_primary_defaults_to_the_total(self) -> None:
        """One file set, one model: the honest default for every non-GGUF format."""
        observation = FormatObservation(
            format=ModelFormat.SAFETENSORS, total_bytes=18 * GIB, complete=True
        )
        assert observation.primary_bytes == 18 * GIB

    def test_an_explicit_zero_primary_means_unknown_not_the_total(self) -> None:
        """A detector that computed 0 is believed; the convenience must not fire.

        The inventory behind this: a GGUF repository where the Hub sized only
        the multimodal projector. Every candidate *model* file is sizeless, so
        the primary is genuinely unknown — and copying ``total_bytes`` over it
        would announce the projector, the one file classified as not runnable,
        as "the size of one usable model".
        """
        observation = FormatObservation(
            format=ModelFormat.GGUF,
            files=["demo-Q4_K_M.gguf", "mmproj-demo-BF16.gguf"],
            total_bytes=900,
            primary_bytes=0,
            complete=True,
        )
        assert observation.primary_bytes == 0
        assert observation.total_bytes == 900

    def test_an_unknown_primary_is_never_read_as_a_weight_figure(self) -> None:
        """An observation that cannot size one model answers "unknown" everywhere."""
        metadata = _metadata(
            formats=[
                FormatObservation(
                    format=ModelFormat.GGUF,
                    files=["demo-Q4_K_M.gguf", "mmproj-demo-BF16.gguf"],
                    total_bytes=900,
                    primary_bytes=0,
                    complete=True,
                )
            ]
        )
        assert metadata.weight_bytes == 0
        assert metadata.primary_observation is None
        # The format is still known: only its size is not.
        assert metadata.primary_format is ModelFormat.GGUF

    def test_an_unknown_primary_survives_a_serialisation_round_trip(self) -> None:
        """Dump and reload must not resurrect the fallback and invent a size."""
        observation = FormatObservation(
            format=ModelFormat.GGUF,
            files=["mmproj-demo-BF16.gguf"],
            total_bytes=900,
            primary_bytes=0,
            complete=True,
        )
        assert FormatObservation.model_validate(observation.model_dump()).primary_bytes == 0

    def test_an_explicit_primary_is_left_alone(self) -> None:
        observation = FormatObservation(
            format=ModelFormat.GGUF,
            files=["a.gguf", "b.gguf"],
            total_bytes=38 * GIB,
            primary_bytes=16 * GIB,
            complete=True,
        )
        assert observation.primary_bytes == 16 * GIB
        assert observation.total_bytes == 38 * GIB

    def test_unknown_sizes_stay_unknown_on_both_counters(self) -> None:
        observation = FormatObservation(format=ModelFormat.GGUF, files=["a.gguf"])
        assert observation.primary_bytes == 0
        assert observation.total_bytes == 0

    def test_weight_bytes_reports_one_model_and_repo_bytes_the_whole_format(self) -> None:
        metadata = _metadata(
            formats=[
                FormatObservation(
                    format=ModelFormat.GGUF,
                    files=["bf16.gguf", "q4.gguf", "q8.gguf"],
                    total_bytes=38 * GIB,
                    primary_bytes=16 * GIB,
                    complete=True,
                )
            ]
        )
        assert metadata.weight_bytes == 16 * GIB
        assert metadata.repo_weight_bytes == 38 * GIB
        assert metadata.primary_format is ModelFormat.GGUF

    def test_the_two_agree_when_every_file_is_part_of_one_model(self) -> None:
        metadata = _metadata(
            formats=[
                FormatObservation(
                    format=ModelFormat.SAFETENSORS,
                    files=[f"shard-{index}.safetensors" for index in range(4)],
                    total_bytes=18_819_721_040,
                    complete=True,
                )
            ]
        )
        assert metadata.weight_bytes == metadata.repo_weight_bytes == 18_819_721_040

    def test_a_format_preference_still_decides_which_observation_is_read(self) -> None:
        """Adding the second counter did not move the preference order."""
        metadata = _metadata(
            formats=[
                FormatObservation(
                    format=ModelFormat.GGUF,
                    total_bytes=38 * GIB,
                    primary_bytes=16 * GIB,
                    complete=True,
                ),
                FormatObservation(
                    format=ModelFormat.SAFETENSORS, total_bytes=18 * GIB, complete=True
                ),
            ]
        )
        assert metadata.primary_format is ModelFormat.SAFETENSORS
        assert metadata.weight_bytes == 18 * GIB
        assert metadata.repo_weight_bytes == 18 * GIB

    def test_a_known_primary_survives_an_unknown_repository_total(self) -> None:
        """``inspect --no-sizes`` on a GGUF repo the Hub has already measured."""
        metadata = _metadata(
            formats=[
                FormatObservation(
                    format=ModelFormat.GGUF,
                    files=["bf16.gguf", "q4.gguf"],
                    total_bytes=0,
                    primary_bytes=17_920_696_704,
                    complete=True,
                )
            ]
        )
        assert metadata.weight_bytes == 17_920_696_704
        assert metadata.repo_weight_bytes == 0

    def test_no_observations_report_zero_on_both_counters(self) -> None:
        metadata = _metadata()
        assert metadata.weight_bytes == 0
        assert metadata.repo_weight_bytes == 0
        assert metadata.primary_observation is None

    def test_primary_observation_is_the_one_the_numbers_came_from(self) -> None:
        gguf = FormatObservation(
            format=ModelFormat.GGUF,
            files=["bf16.gguf"],
            total_bytes=38 * GIB,
            primary_bytes=16 * GIB,
            complete=True,
        )
        metadata = _metadata(formats=[gguf])
        assert metadata.primary_observation is gguf


class TestPipelineKind:
    @pytest.mark.parametrize(
        "tag", ["image-text-to-text", "visual-question-answering", "image-to-text"]
    )
    def test_vlm_pipeline_tags(self, tag: str) -> None:
        assert infer_pipeline_kind(pipeline_tag=tag) is PipelineKind.VLM

    def test_vlm_tag_wins_over_causal_architecture(self) -> None:
        assert (
            infer_pipeline_kind(
                pipeline_tag="image-text-to-text", architectures=["Qwen3ForCausalLM"]
            )
            is PipelineKind.VLM
        )

    def test_conditional_generation_with_vision_config_is_vlm(self) -> None:
        assert (
            infer_pipeline_kind(
                architectures=["Qwen3VLForConditionalGeneration"],
                config={"vision_config": {"hidden_size": 1152}},
            )
            is PipelineKind.VLM
        )

    def test_conditional_generation_without_vision_config_is_not_vlm(self) -> None:
        assert (
            infer_pipeline_kind(
                architectures=["T5ForConditionalGeneration"], config={"model_type": "t5"}
            )
            is PipelineKind.UNKNOWN
        )

    def test_vision_config_alone_is_not_vlm(self) -> None:
        assert (
            infer_pipeline_kind(config={"vision_config": {}}, architectures=["SiglipVisionModel"])
            is PipelineKind.UNKNOWN
        )

    def test_text_generation_tag_is_llm(self) -> None:
        assert infer_pipeline_kind(pipeline_tag="text-generation") is PipelineKind.LLM

    def test_causal_lm_architecture_is_llm(self) -> None:
        assert infer_pipeline_kind(architectures=["Qwen3ForCausalLM"]) is PipelineKind.LLM

    def test_task_is_used_when_pipeline_tag_is_missing(self) -> None:
        assert infer_pipeline_kind(task="image-text-to-text") is PipelineKind.VLM
        assert infer_pipeline_kind(task="text-generation") is PipelineKind.LLM

    def test_tag_case_and_whitespace_are_tolerated(self) -> None:
        assert infer_pipeline_kind(pipeline_tag=" Text-Generation ") is PipelineKind.LLM

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"pipeline_tag": "feature-extraction"},
            {"architectures": ["BertModel"]},
            {"pipeline_tag": None, "architectures": [], "config": {}},
        ],
    )
    def test_unknown_cases(self, kwargs: dict[str, Any]) -> None:
        assert infer_pipeline_kind(**kwargs) is PipelineKind.UNKNOWN

    def test_metadata_property_uses_the_same_heuristic(self) -> None:
        metadata = _metadata(
            pipeline_tag="image-text-to-text",
            config={"architectures": ["Qwen3VLForConditionalGeneration"], "vision_config": {}},
        )
        assert metadata.pipeline_kind is PipelineKind.VLM

        llm = _metadata(config={"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"})
        assert llm.pipeline_kind is PipelineKind.LLM


class TestModelMetadata:
    def test_config_fills_architectures_model_type_and_task(self) -> None:
        metadata = _metadata(
            pipeline_tag="text-generation",
            config={"architectures": ["Qwen3ForCausalLM", 7], "model_type": " qwen3 "},
        )
        assert metadata.architectures == ["Qwen3ForCausalLM"]
        assert metadata.model_type == "qwen3"
        assert metadata.task == "text-generation"

    def test_explicit_values_are_not_overwritten(self) -> None:
        metadata = _metadata(
            architectures=["Explicit"],
            model_type="explicit",
            task="explicit-task",
            pipeline_tag="text-generation",
            config={"architectures": ["FromConfig"], "model_type": "from-config"},
        )
        assert metadata.architectures == ["Explicit"]
        assert metadata.model_type == "explicit"
        assert metadata.task == "explicit-task"

    def test_defaults_are_conservative(self) -> None:
        metadata = _metadata()
        assert metadata.exists is True
        assert metadata.gated is False
        assert metadata.private is False
        assert metadata.requires_remote_code is False
        assert metadata.files == []
        assert metadata.tags == []


class TestResolvedSource:
    def test_same_repository_is_not_a_derivative(self) -> None:
        identity = ModelIdentity(repo_id="a/b", revision="sha")
        resolved = ResolvedSource(
            requested=identity,
            source=identity,
            relationship=SourceRelationship.ORIGINAL,
            format=ModelFormat.SAFETENSORS,
        )
        assert resolved.is_derivative is False
        assert resolved.already_openvino is False

    def test_third_party_source_is_a_derivative(self) -> None:
        resolved = ResolvedSource(
            requested=ModelIdentity(repo_id="HauhauCS/Qwen3.5-9B-Uncensored"),
            source=ModelIdentity(repo_id="nDimensional/Qwen3.5-9B-Safetensors", revision="sha"),
            relationship=SourceRelationship.LOSSLESS_CONVERSION,
            format=ModelFormat.SAFETENSORS,
            requires_confirmation=True,
            evidence=[SourceEvidence(kind="model_card", url="https://example.invalid/card")],
        )
        assert resolved.is_derivative is True
        assert resolved.requires_confirmation is True
        assert resolved.evidence[0].kind == "model_card"

    def test_relationship_alone_marks_a_derivative(self) -> None:
        identity = ModelIdentity(repo_id="a/b")
        resolved = ResolvedSource(
            requested=identity, source=identity, relationship=SourceRelationship.QUANTIZED
        )
        assert resolved.is_derivative is True

    def test_openvino_ir_format_sets_the_flag(self) -> None:
        identity = ModelIdentity(repo_id="OpenVINO/model-int4-ov")
        resolved = ResolvedSource(
            requested=identity, source=identity, format=ModelFormat.OPENVINO_IR
        )
        assert resolved.already_openvino is True
