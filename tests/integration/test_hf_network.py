"""Live Hugging Face Hub checks against the three reference repositories.

Every test here is marked ``network`` and is therefore skipped by the default
``addopts`` filter in ``pyproject.toml``. Run them deliberately with::

    .venv/bin/pytest tests/integration/test_hf_network.py -m network

The expected values are the ones recorded in ``HOST_FACTS.md`` after querying the
Hub directly, so a failure here means either the network is unavailable or an
upstream repository actually changed.
"""

from __future__ import annotations

import pytest

from yonga.adapters.hf import CODE_REPO_NOT_FOUND, HuggingFaceAdapter
from yonga.application.progress import CollectingSink, ProgressKind
from yonga.config.settings import Settings
from yonga.domain.diagnostics import SourceResolutionError
from yonga.domain.models import ModelFormat, ModelIdentity, PipelineKind
from yonga.util.jsonlog import null_logger

pytestmark = pytest.mark.network

GGUF_REPO = "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"
GGUF_SHA = "0a41c68809d375475f954be12ba7c40efa56c2a9"

SAFETENSORS_REPO = "nDimensional/Qwen3.5-9B-Uncensored-Safetensors"
SAFETENSORS_SHA = "4623e504074fc2489f3af98dc3d68b8392bbc21f"
SAFETENSORS_WEIGHT_BYTES = 18_819_721_040

OPENVINO_REPO = "OpenVINO/Qwen3.5-9B-int4-ov"
OPENVINO_SHA = "3f31fc1609edb0cbc8b131bc4c291f1b15cb0676"


@pytest.fixture
def adapter(tmp_path):
    """An adapter whose Hub cache lives under ``tmp_path``, never the user's."""
    return HuggingFaceAdapter(Settings(hf_home=tmp_path / "hf"))


@pytest.mark.network
@pytest.mark.parametrize(
    ("repo_id", "expected_sha"),
    [
        (GGUF_REPO, GGUF_SHA),
        (SAFETENSORS_REPO, SAFETENSORS_SHA),
        (OPENVINO_REPO, OPENVINO_SHA),
    ],
)
def test_resolve_revision_pins_the_published_commit(adapter, repo_id, expected_sha):
    resolved = adapter.resolve_revision(ModelIdentity(repo_id=repo_id, revision="main"))

    assert resolved == expected_sha


@pytest.mark.network
def test_inspect_gguf_only_repository_without_config_json(adapter):
    metadata = adapter.inspect(ModelIdentity(repo_id=GGUF_REPO))

    assert metadata.resolved_revision == GGUF_SHA
    assert metadata.primary_format is ModelFormat.GGUF
    assert metadata.has_format(ModelFormat.GGUF)
    assert not metadata.has_format(ModelFormat.SAFETENSORS)
    # There genuinely is no config.json in this repository.
    assert all(entry.path != "config.json" for entry in metadata.files)
    assert metadata.model_type is None
    assert metadata.architectures == []
    assert "qwen3.5" in metadata.tags
    assert metadata.license == "apache-2.0"
    assert metadata.gated is False
    assert metadata.gguf_metadata.get("architecture") == "qwen35"
    # A multi-gigabyte chat template must never be inlined verbatim.
    template = metadata.gguf_metadata.get("chat_template")
    assert template is None or str(template).startswith("<omitted:")

    observation = next(o for o in metadata.formats if o.format is ModelFormat.GGUF)
    assert len(observation.files) == 5
    assert any("no config.json" in note for note in observation.notes)


@pytest.mark.network
def test_inspect_safetensors_derivative_with_non_standard_shard_names(adapter):
    metadata = adapter.inspect(ModelIdentity(repo_id=SAFETENSORS_REPO))

    assert metadata.resolved_revision == SAFETENSORS_SHA
    assert metadata.primary_format is ModelFormat.SAFETENSORS
    assert metadata.weight_bytes == SAFETENSORS_WEIGHT_BYTES
    assert metadata.architectures == ["Qwen3_5ForConditionalGeneration"]
    assert metadata.model_type == "qwen3_5"
    assert metadata.pipeline_tag == "image-text-to-text"
    assert metadata.library_name == "transformers"
    assert "vision_config" in metadata.config
    assert metadata.pipeline_kind is PipelineKind.VLM
    assert metadata.requires_remote_code is False
    assert metadata.parameter_count == 9_409_813_744

    observation = next(o for o in metadata.formats if o.format is ModelFormat.SAFETENSORS)
    assert observation.complete is True
    assert all(name.endswith(".safetensors") for name in observation.files)
    assert any("-of-00004.safetensors" in name for name in observation.files)


@pytest.mark.network
def test_inspect_prepared_openvino_artifact(adapter):
    metadata = adapter.inspect(ModelIdentity(repo_id=OPENVINO_REPO))

    assert metadata.resolved_revision == OPENVINO_SHA
    assert metadata.primary_format is ModelFormat.OPENVINO_IR
    assert metadata.library_name == "openvino"

    observation = next(o for o in metadata.formats if o.format is ModelFormat.OPENVINO_IR)
    assert observation.complete is True
    assert "openvino_language_model.xml" in observation.files
    assert "openvino_language_model.bin" in observation.files


@pytest.mark.network
def test_inspect_never_downloads_a_weight_shard(adapter, tmp_path):
    adapter.inspect(ModelIdentity(repo_id=SAFETENSORS_REPO))

    cache = tmp_path / "hf" / "hub"
    blobs = [path for path in cache.rglob("*") if path.is_file() and not path.is_symlink()]
    assert blobs, "the small metadata files should have been cached"
    assert max(path.stat().st_size for path in blobs) < 8 * 1024 * 1024


@pytest.mark.network
def test_download_uses_the_pinned_sha_and_emits_progress(tmp_path):
    from yonga.application.jobs import JobContext

    settings = Settings(hf_home=tmp_path / "hf", data_dir=tmp_path / "data")
    adapter = HuggingFaceAdapter(settings)
    sink = CollectingSink()
    ctx = JobContext(
        job_id="network-test",
        logger=null_logger("network-test"),
        paths=settings.paths,
        settings=settings,
        progress=sink,
    )

    snapshot = adapter.download(
        ModelIdentity(repo_id=SAFETENSORS_REPO, revision="main"),
        ctx=ctx,
        allow_patterns=["config.json"],
    )

    assert snapshot.is_dir()
    # The Hub lays snapshots out as <cache>/models--…/snapshots/<commit sha>/.
    assert snapshot.name == SAFETENSORS_SHA
    assert (snapshot / "config.json").is_file()

    statuses = [event.status for event in sink.of_kind(ProgressKind.DOWNLOAD)]
    assert statuses[0] == "started"
    assert statuses[-1] == "completed"


@pytest.mark.network
def test_missing_repository_is_a_typed_diagnostic(adapter):
    identity = ModelIdentity(repo_id="yonga-tests/definitely-not-a-real-repository-1a2b3c")

    with pytest.raises(SourceResolutionError) as excinfo:
        adapter.inspect(identity)

    assert excinfo.value.diagnostic.code == CODE_REPO_NOT_FOUND
    assert excinfo.value.diagnostic.remediation


@pytest.mark.network
def test_auth_info_reports_state_without_a_token_value(adapter, tmp_path):
    info = adapter.auth_info()

    assert info.cache_dir == tmp_path / "hf" / "hub"
    assert info.offline is False
    if info.logged_in:
        assert info.token_source is not None
        assert not info.token_source.startswith("hf_")
