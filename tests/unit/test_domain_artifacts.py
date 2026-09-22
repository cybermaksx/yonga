"""Unit tests for yonga.domain.artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from yonga.domain.artifacts import (
    MANIFEST_DIRNAME,
    MANIFEST_FILENAME,
    ArtifactFileRecord,
    ArtifactManifest,
    DeviceProbeReport,
    ExportRecord,
    ManifestModelRef,
    ProbeResult,
    ProbeStatus,
    RecommendedRuntime,
    RepairRecord,
    SubmodelRole,
    ValidationReport,
    assert_no_secrets,
    manifest_path,
    role_for_filename,
)
from yonga.domain.diagnostics import ValidationError
from yonga.domain.models import PipelineKind, SourceRelationship

EXPORT_FILENAMES = [
    ("openvino_language_model.xml", SubmodelRole.LANGUAGE),
    ("openvino_text_embeddings_model.xml", SubmodelRole.TEXT_EMBEDDINGS),
    ("openvino_vision_embeddings_model.xml", SubmodelRole.VISION_EMBEDDINGS),
    ("openvino_vision_embeddings_merger_model.xml", SubmodelRole.VISION_EMBEDDINGS_MERGER),
    ("openvino_vision_embeddings_pos_model.xml", SubmodelRole.VISION_EMBEDDINGS_POS),
    ("openvino_resampler_model.xml", SubmodelRole.RESAMPLER),
    ("openvino_tokenizer.xml", SubmodelRole.TOKENIZER),
    ("openvino_detokenizer.xml", SubmodelRole.DETOKENIZER),
    ("openvino_model.xml", SubmodelRole.LANGUAGE),
]

ADVERSARIAL_FILENAMES = [
    # Case and path handling.
    ("OPENVINO_DETOKENIZER.XML", SubmodelRole.DETOKENIZER),
    ("dir/openvino_vision_embeddings_merger_model.xml", SubmodelRole.VISION_EMBEDDINGS_MERGER),
    ("./openvino_vision_embeddings_pos_model.xml", SubmodelRole.VISION_EMBEDDINGS_POS),
    ("  openvino_tokenizer.xml  ", SubmodelRole.TOKENIZER),
    # Weight files carry the same role as their IR.
    ("openvino_language_model.bin", SubmodelRole.LANGUAGE),
    ("openvino_detokenizer.bin", SubmodelRole.DETOKENIZER),
    # "detokenizer" must not be read as "tokenizer", "merger"/"pos" must beat the
    # plain vision_embeddings prefix even when the name is not an exact export name.
    ("my_openvino_detokenizer_v2.xml", SubmodelRole.DETOKENIZER),
    ("openvino_vision_embeddings_merger_model_int4.xml", SubmodelRole.VISION_EMBEDDINGS_MERGER),
    ("openvino_vision_embeddings_pos_model_int4.xml", SubmodelRole.VISION_EMBEDDINGS_POS),
    ("quantized_openvino_vision_embeddings_model.xml", SubmodelRole.VISION_EMBEDDINGS),
    # Non-IR files never become submodels, even when the name mentions a role.
    ("tokenizer_config.json", SubmodelRole.OTHER),
    ("tokenizer.json", SubmodelRole.OTHER),
    ("openvino_model.gguf", SubmodelRole.OTHER),
    ("config.json", SubmodelRole.OTHER),
    ("model.safetensors", SubmodelRole.OTHER),
    # Unknown or empty names.
    ("openvino_encoder_model.xml", SubmodelRole.OTHER),
    ("", SubmodelRole.OTHER),
    ("   ", SubmodelRole.OTHER),
]


@pytest.mark.parametrize(("name", "expected"), EXPORT_FILENAMES)
def test_role_for_exported_filenames(name: str, expected: SubmodelRole) -> None:
    assert role_for_filename(name) is expected


@pytest.mark.parametrize(("name", "expected"), ADVERSARIAL_FILENAMES)
def test_role_for_adversarial_filenames(name: str, expected: SubmodelRole) -> None:
    assert role_for_filename(name) is expected


def test_tokenizer_roles_are_flagged() -> None:
    assert SubmodelRole.TOKENIZER.is_tokenizer
    assert SubmodelRole.DETOKENIZER.is_tokenizer
    assert not SubmodelRole.LANGUAGE.is_tokenizer
    assert not SubmodelRole.VISION_EMBEDDINGS_MERGER.is_tokenizer


def test_manifest_path_layout() -> None:
    assert manifest_path(Path("/artifacts/a")) == Path("/artifacts/a/.yonga/manifest.json")
    assert MANIFEST_DIRNAME == ".yonga"
    assert MANIFEST_FILENAME == "manifest.json"


# --- secret guard -----------------------------------------------------------------------


def test_assert_no_secrets_accepts_benign_manifest_keys() -> None:
    assert_no_secrets(
        {
            "tokenizer_read_ok": True,
            "detokenizer_read_ok": True,
            "tokenizers_version": "2026.4.0.0",
            "max_new_tokens": 4096,
            "output_tokens": 128,
            "keys": ["a", "b"],
            "files": [{"path": "openvino_tokenizer.xml", "size": 1}],
        }
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"hf_token": "hf_abc"},
        {"HF_TOKEN": "hf_abc"},
        {"export": {"env": {"HUGGING_FACE_HUB_TOKEN": "hf_abc"}}},
        {"host": {"api_key": "sk-abc"}},
        {"repairs": [{"detail": {"password": "hunter2"}}]},
        {"auth": {"Authorization": "Bearer abc"}},
        {"client_secret": "abc"},
    ],
)
def test_assert_no_secrets_rejects_credential_keys(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError) as excinfo:
        assert_no_secrets(payload)
    assert excinfo.value.diagnostic.code == "artifact.manifest.secret_key"


# --- manifest ---------------------------------------------------------------------------


def make_manifest() -> ArtifactManifest:
    return ArtifactManifest(
        artifact_id="qwen35-abc123-gpu",
        created_at=datetime(2026, 9, 22, 10, 0, 0, tzinfo=UTC),
        tool_version="0.1.0",
        requested_model=ManifestModelRef(
            repo_id="HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive",
            requested_revision="main",
        ),
        resolved_source=ManifestModelRef(
            repo_id="nDimensional/Qwen3.5-9B-Uncensored-Safetensors",
            resolved_revision="abc123",
            relationship=SourceRelationship.LOSSLESS_CONVERSION,
        ),
        architecture="qwen3_5",
        model_type="qwen3_5",
        task="image-text-to-text",
        pipeline=PipelineKind.VLM,
        export=ExportRecord(
            profile="gpu-int4-asym-g128",
            arguments={"weight_format": "int4", "group_size": 128, "ratio": 1.0},
            argv=["optimum-cli", "export", "openvino"],
            package_versions={"openvino": "2026.4.0", "optimum-intel": "2.2.0"},
            duration_seconds=612.5,
        ),
        repairs=[
            RepairRecord(
                repair_id="ov-tokenizer-precision-string-casing",
                description="precision=STRiNG -> STRING",
                applied=True,
                safety="automatic",
                files=["openvino_tokenizer.xml"],
            )
        ],
        validation=ValidationReport(ok=True, openvino_version="2026.4.0", tokenizer_read_ok=True),
        device_probes={
            "GPU": DeviceProbeReport(
                device="GPU",
                status=ProbeStatus.OK,
                submodels=[
                    ProbeResult(
                        device="gpu.0",
                        role=SubmodelRole.LANGUAGE,
                        path=Path("openvino_language_model.xml"),
                        status=ProbeStatus.OK,
                        duration_seconds=3.5,
                    )
                ],
            ),
            "NPU": DeviceProbeReport(device="npu", status=ProbeStatus.FAILED),
        },
        recommended_runtime=RecommendedRuntime(
            device="gpu",
            pipeline=PipelineKind.VLM,
            properties={"CACHE_DIR": "/home/user/.cache/yonga/compile"},
            max_new_tokens=4096,
            generation_defaults={"temperature": 0.7, "do_sample": True, "top_k": 40},
        ),
        source_license="apache-2.0",
        files=[
            ArtifactFileRecord(path="openvino_tokenizer.xml", size=12, sha256="ab" * 32),
            ArtifactFileRecord(path="openvino_language_model.xml", size=34),
        ],
        host={"platform": "Linux", "kernel": "7.2.6"},
    )


def test_manifest_supported_devices_only_lists_successful_probes() -> None:
    manifest = make_manifest()
    assert manifest.supported_devices == ["GPU"]
    assert manifest.ready


def test_manifest_files_are_sorted_for_stable_output() -> None:
    manifest = make_manifest()
    assert [record.path for record in manifest.files] == [
        "openvino_language_model.xml",
        "openvino_tokenizer.xml",
    ]


def test_manifest_to_json_has_no_secrets_and_is_serialisable() -> None:
    data = make_manifest().to_json()
    assert_no_secrets(data)
    assert json.dumps(data)
    assert list(data) == sorted(data)
    # exclude_none removes empty optional fields.
    assert "source_license" in data
    assert data["pipeline"] == "vlm"


def test_manifest_json_round_trips() -> None:
    manifest = make_manifest()
    data = manifest.to_json()
    restored = ArtifactManifest.model_validate(data)
    assert restored == manifest
    assert restored.to_json() == data


def test_manifest_to_json_rejects_a_smuggled_secret() -> None:
    manifest = make_manifest()
    manifest.host["hf_token"] = "hf_secret"
    with pytest.raises(ValidationError):
        manifest.to_json()


def test_model_validate_file_round_trip(tmp_path: Path) -> None:
    manifest = make_manifest()
    path = tmp_path / MANIFEST_DIRNAME / MANIFEST_FILENAME
    path.parent.mkdir(parents=True)
    path.write_text(manifest.to_json_text(), encoding="utf-8")

    loaded = ArtifactManifest.model_validate_file(path)
    assert loaded == manifest


def test_model_validate_file_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValidationError) as excinfo:
        ArtifactManifest.model_validate_file(tmp_path / "nope.json")
    assert excinfo.value.diagnostic.code == "artifact.manifest.unreadable"


def test_model_validate_file_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValidationError) as excinfo:
        ArtifactManifest.model_validate_file(path)
    assert excinfo.value.diagnostic.code == "artifact.manifest.malformed_json"


def test_model_validate_file_schema_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(ValidationError) as excinfo:
        ArtifactManifest.model_validate_file(path)
    assert excinfo.value.diagnostic.code == "artifact.manifest.schema_mismatch"


def test_manifest_tolerates_unknown_future_fields(tmp_path: Path) -> None:
    data = make_manifest().to_json()
    data["some_future_field"] = {"added": "later"}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert ArtifactManifest.model_validate_file(path).artifact_id == "qwen35-abc123-gpu"


def test_validation_report_role_lookup() -> None:
    manifest = make_manifest()
    assert manifest.validation is not None
    assert manifest.validation.role(SubmodelRole.LANGUAGE) is None
