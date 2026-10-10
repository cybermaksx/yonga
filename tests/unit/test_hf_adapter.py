"""Unit tests for the Hugging Face Hub adapter.

Every test here runs with no network: ``HfApi``, ``model_info``,
``hf_hub_download`` and ``snapshot_download`` are replaced by in-process fakes,
and the fixture data mirrors what the real Hub returned for the three reference
repositories (see ``HOST_FACTS.md``).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from yonga.adapters import HubClient
from yonga.adapters.hf import (
    CODE_AUTH_REQUIRED,
    CODE_FILE_NOT_FOUND,
    CODE_GATED,
    CODE_HTTP_ERROR,
    CODE_INVALID_REPO_ID,
    CODE_NETWORK_ERROR,
    CODE_OFFLINE_MISSING,
    CODE_OFFLINE_SEARCH,
    CODE_RATE_LIMITED,
    CODE_REPO_NOT_FOUND,
    CODE_REVISION_NOT_FOUND,
    CODE_REVISION_UNRESOLVED,
    CODE_UNKNOWN,
    HuggingFaceAdapter,
    detect_formats,
    detect_remote_code,
    is_commit_sha,
    translate_hub_error,
)
from yonga.application.jobs import JobContext
from yonga.application.progress import CollectingSink, ProgressKind
from yonga.config.settings import Settings
from yonga.domain.diagnostics import (
    CancelledError,
    ExitCode,
    FailureCategory,
    HubAuthError,
    SourceResolutionError,
    YongaError,
)
from yonga.domain.models import ModelFormat, ModelIdentity, PipelineKind, RepoFile
from yonga.util.jsonlog import null_logger
from yonga.util.redaction import REDACTED

# Real SHAs from the reference repositories.
GGUF_SHA = "0a41c68809d375475f954be12ba7c40efa56c2a9"
SAFETENSORS_SHA = "4623e504074fc2489f3af98dc3d68b8392bbc21f"
OPENVINO_SHA = "3f31fc1609edb0cbc8b131bc4c291f1b15cb0676"

# A token-shaped literal used to prove nothing leaks into a diagnostic.
FAKE_TOKEN = "hf_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


@dataclass
class FakeLfs:
    """Stand-in for ``huggingface_hub.hf_api.BlobLfsInfo``."""

    size: int


@dataclass
class FakeSibling:
    """Stand-in for ``huggingface_hub.hf_api.RepoSibling``."""

    rfilename: str
    size: int | None = None
    lfs: FakeLfs | None = None


@dataclass
class FakeCardData:
    """Stand-in for ``huggingface_hub.ModelCardData``."""

    license: str | None = None


@dataclass
class FakeSafeTensorsInfo:
    """Stand-in for ``huggingface_hub.hf_api.SafeTensorsInfo``."""

    total: int


@dataclass
class FakeModelInfo:
    """Stand-in for ``huggingface_hub.ModelInfo``."""

    sha: str | None = None
    siblings: list[FakeSibling] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    gated: object = False
    private: bool = False
    pipeline_tag: str | None = None
    library_name: str | None = None
    card_data: FakeCardData | None = None
    config: dict[str, object] | None = None
    gguf: dict[str, object] | None = None
    safetensors: FakeSafeTensorsInfo | None = None
    downloads: int | None = None
    likes: int | None = None
    last_modified: str | None = None


class FakeResponse:
    """Minimal ``httpx.Response`` stand-in for building Hub errors."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.request = None


class FakeApi:
    """In-process replacement for ``huggingface_hub.HfApi``.

    Records every call so the tests can assert *what was requested*, which is how
    "inspect must not download weight shards" and "download must use the resolved
    SHA" are verified.
    """

    def __init__(
        self,
        *,
        info: FakeModelInfo | None = None,
        json_files: dict[str, object] | None = None,
        root: Path | None = None,
        snapshot: Path | None = None,
        model_info_error: BaseException | None = None,
        download_error: BaseException | None = None,
        snapshot_error: BaseException | None = None,
        whoami_result: object = None,
        whoami_error: BaseException | None = None,
    ) -> None:
        self.info = info
        self.json_files = json_files or {}
        self.root = root
        self.snapshot = snapshot
        self.model_info_error = model_info_error
        self.download_error = download_error
        self.snapshot_error = snapshot_error
        self.whoami_result = whoami_result if whoami_result is not None else {"name": "octocat"}
        self.whoami_error = whoami_error

        self.model_info_calls: list[dict[str, object]] = []
        self.download_calls: list[dict[str, object]] = []
        self.snapshot_calls: list[dict[str, object]] = []

    def model_info(self, repo_id, *, revision=None, files_metadata=False, **_kwargs):
        self.model_info_calls.append(
            {"repo_id": repo_id, "revision": revision, "files_metadata": files_metadata}
        )
        if self.model_info_error is not None:
            raise self.model_info_error
        assert self.info is not None
        return self.info

    def hf_hub_download(
        self,
        repo_id,
        *,
        filename,
        revision=None,
        cache_dir=None,
        local_files_only=False,
        **_kwargs,
    ):
        self.download_calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "revision": revision,
                "cache_dir": cache_dir,
                "local_files_only": local_files_only,
            }
        )
        if self.download_error is not None:
            raise self.download_error
        if filename not in self.json_files:
            from huggingface_hub.errors import EntryNotFoundError

            raise EntryNotFoundError(f"{filename} not found")
        assert self.root is not None
        target = self.root / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.json_files[filename]), encoding="utf-8")
        return str(target)

    def snapshot_download(
        self,
        repo_id,
        *,
        revision=None,
        cache_dir=None,
        local_files_only=False,
        allow_patterns=None,
        ignore_patterns=None,
        tqdm_class=None,
        **_kwargs,
    ):
        self.snapshot_calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "cache_dir": cache_dir,
                "local_files_only": local_files_only,
                "allow_patterns": allow_patterns,
                "ignore_patterns": ignore_patterns,
                "tqdm_class": tqdm_class,
            }
        )
        if self.snapshot_error is not None:
            raise self.snapshot_error
        if tqdm_class is not None:
            bar = tqdm_class(total=0, initial=0, unit="B", unit_scale=True, disable=True)
            bar.total = 4096
            bar.update(2048)
            bar.update(2048)
            bar.close()
        assert self.snapshot is not None
        return str(self.snapshot)

    def whoami(self, *_args, **_kwargs):
        if self.whoami_error is not None:
            raise self.whoami_error
        return self.whoami_result


def make_job_context(tmp_path: Path, *, settings: Settings | None = None) -> JobContext:
    """Build a real :class:`JobContext` backed by a temporary directory."""
    resolved = settings or Settings(data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")
    return JobContext(
        job_id="test-job",
        logger=null_logger("test-job"),
        paths=resolved.paths,
        settings=resolved,
        progress=CollectingSink(),
    )


# --------------------------------------------------------------------------- #
# Fixture data mirroring the real repositories
# --------------------------------------------------------------------------- #

GGUF_SIBLINGS = [
    FakeSibling(".gitattributes", 1968),
    FakeSibling("README.md", 2952),
    FakeSibling(
        "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf",
        17920696704,
        FakeLfs(17920696704),
    ),
    FakeSibling(
        "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        5627044224,
        FakeLfs(5627044224),
    ),
    FakeSibling(
        "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q6_K.gguf", 7359259008, FakeLfs(7359259008)
    ),
    FakeSibling(
        "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf", 9527501184, FakeLfs(9527501184)
    ),
    FakeSibling(
        "mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf",
        921704448,
        FakeLfs(921704448),
    ),
]
GGUF_TOTAL_BYTES = 17920696704 + 5627044224 + 7359259008 + 9527501184 + 921704448
# What the live Hub reports as gguf["totalFileSize"]: the BF16 model alone, not
# the sum of the four quantizations and the projector beside it.
GGUF_PRIMARY_BYTES = 17920696704

SAFETENSORS_SHARDS = {
    "model.safetensors-00001-of-00004.safetensors": 5278251472,
    "model.safetensors-00002-of-00004.safetensors": 5358071808,
    "model.safetensors-00003-of-00004.safetensors": 5340098264,
    "model.safetensors-00004-of-00004.safetensors": 2843299496,
}
SAFETENSORS_WEIGHT_BYTES = 18_819_721_040

SAFETENSORS_SIBLINGS = [
    FakeSibling(".gitattributes", 1570),
    FakeSibling("LICENSE", 11544),
    FakeSibling("README.md", 3586),
    FakeSibling("chat_template.jinja", 7756),
    FakeSibling("config.json", 2660),
    FakeSibling("merges.txt", 3353259),
    *(FakeSibling(name, size, FakeLfs(size)) for name, size in SAFETENSORS_SHARDS.items()),
    FakeSibling("model.safetensors.index.json", 78336),
    FakeSibling("preprocessor_config.json", 390),
    FakeSibling("tokenizer.json", 12807982, FakeLfs(12807982)),
    FakeSibling("tokenizer_config.json", 16710),
    FakeSibling("video_preprocessor_config.json", 385),
    FakeSibling("vocab.json", 6722759, FakeLfs(6722759)),
]

SAFETENSORS_INDEX = {
    "metadata": {"total_size": SAFETENSORS_WEIGHT_BYTES},
    "weight_map": {
        f"model.layers.{index}.weight": name
        for index, name in enumerate(SAFETENSORS_SHARDS, start=1)
    },
}

VLM_CONFIG: dict[str, object] = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "vision_config": {"depth": 32},
    "text_config": {"hidden_size": 4096},
}


def gguf_info() -> FakeModelInfo:
    """The GGUF-only repository: no ``config.json`` anywhere in the listing."""
    return FakeModelInfo(
        sha=GGUF_SHA,
        siblings=list(GGUF_SIBLINGS),
        tags=["gguf", "uncensored", "qwen3.5", "qwen", "license:apache-2.0"],
        card_data=FakeCardData(license="apache-2.0"),
        gguf={
            "total": 8953803264,
            "totalFileSize": GGUF_PRIMARY_BYTES,
            "architecture": "qwen35",
            "context_length": 262144,
            "chat_template": "x" * 20000,
            "eos_token": "<|im_end|>",
        },
        downloads=714610,
        likes=2112,
        last_modified="2026-06-05T17:55:41+00:00",
    )


def safetensors_info() -> FakeModelInfo:
    """The safetensors derivative with non-standard shard names."""
    return FakeModelInfo(
        sha=SAFETENSORS_SHA,
        siblings=list(SAFETENSORS_SIBLINGS),
        tags=["transformers", "safetensors", "license:apache-2.0"],
        pipeline_tag="image-text-to-text",
        library_name="transformers",
        card_data=FakeCardData(license="apache-2.0"),
        safetensors=FakeSafeTensorsInfo(total=9409813744),
    )


def safetensors_adapter(tmp_path: Path) -> tuple[HuggingFaceAdapter, FakeApi]:
    """Adapter wired to the safetensors fixture, with its two small JSON files."""
    api = FakeApi(
        info=safetensors_info(),
        json_files={
            "config.json": VLM_CONFIG,
            "model.safetensors.index.json": SAFETENSORS_INDEX,
        },
        root=tmp_path / "hub",
    )
    return adapter_for(api), api


def adapter_for(api: FakeApi, settings: Settings | None = None) -> HuggingFaceAdapter:
    """Build the adapter around a :class:`FakeApi`.

    ``FakeApi`` is a structural stand-in that implements only the handful of
    ``HfApi`` methods the adapter actually calls, so the substitution is recorded
    here once instead of at every construction site.
    """
    return HuggingFaceAdapter(Settings() if settings is None else settings, api=cast(Any, api))


def hub_response(status_code: int) -> Any:
    """A structural ``httpx.Response`` stand-in for building Hub errors."""
    return FakeResponse(status_code)


def safetensors_bytes(tensors: dict[str, tuple[str, list[int]]]) -> bytes:
    """A real safetensors header (and zeroed data) for tensors of the given dtype and shape."""
    widths = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "U8": 1}
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        size = widths[dtype]
        for dim in shape:
            size *= dim
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header).encode()
    return len(encoded).to_bytes(8, "little") + encoded + b"\x00" * offset


def cache_snapshot(
    hf_home: Path,
    repo_id: str,
    sha: str,
    files: dict[str, bytes],
    *,
    refs: dict[str, bool] | None = None,
) -> Path:
    """Lay out one snapshot exactly as ``huggingface_hub`` caches it under ``HF_HOME/hub``.

    Files are symlinks into ``blobs/``, named by their etag: SHA-256 (64 hex
    characters) for LFS-sized files, a 40-character SHA-1 otherwise. Each name
    in *refs* gets a ``refs/<name>`` file pointing at *sha*.
    """
    repo = hf_home / "hub" / ("models--" + repo_id.replace("/", "--"))
    snapshot = repo / "snapshots" / sha
    blobs = repo / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        lfs = name.endswith((".safetensors", ".gguf", ".bin"))
        digest = hashlib.sha256(data) if lfs else hashlib.sha1(data, usedforsecurity=False)
        blob = blobs / digest.hexdigest()
        blob.write_bytes(data)
        link = snapshot / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(blob, link.parent))
    for ref in refs or {}:
        target = repo / "refs" / ref
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(sha, encoding="utf-8")
    return snapshot


# --------------------------------------------------------------------------- #
# Protocol conformance and SHA pinning
# --------------------------------------------------------------------------- #


def test_adapter_satisfies_hub_client_protocol():
    assert isinstance(adapter_for(FakeApi()), HubClient)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (GGUF_SHA, True),
        (GGUF_SHA.upper(), False),
        ("main", False),
        ("v1.0", False),
        (None, False),
        ("", False),
        (GGUF_SHA[:39], False),
    ],
)
def test_is_commit_sha(value, expected):
    assert is_commit_sha(value) is expected


def test_resolve_revision_returns_immutable_sha():
    api = FakeApi(info=gguf_info())
    adapter = adapter_for(api)

    sha = adapter.resolve_revision(ModelIdentity(repo_id="HauhauCS/x", revision="main"))

    assert sha == GGUF_SHA
    assert api.model_info_calls[0]["revision"] == "main"


def test_resolve_revision_without_sha_is_a_typed_failure():
    adapter = adapter_for(FakeApi(info=FakeModelInfo(sha=None)))

    with pytest.raises(SourceResolutionError) as excinfo:
        adapter.resolve_revision(ModelIdentity(repo_id="acme/model"))

    assert excinfo.value.diagnostic.code == CODE_REVISION_UNRESOLVED
    assert excinfo.value.exit_code() is ExitCode.SOURCE_RESOLUTION


def test_inspect_records_the_resolved_sha(tmp_path):
    adapter, _ = safetensors_adapter(tmp_path)

    metadata = adapter.inspect(ModelIdentity(repo_id="nDimensional/x", revision="main"))

    assert metadata.resolved_revision == SAFETENSORS_SHA
    assert metadata.identity.revision == "main"


def test_inspect_fetches_only_small_json_from_the_pinned_sha(tmp_path):
    adapter, api = safetensors_adapter(tmp_path)

    adapter.inspect(ModelIdentity(repo_id="nDimensional/x", revision="main"))

    requested = [call["filename"] for call in api.download_calls]
    assert set(requested) == {"config.json", "model.safetensors.index.json"}
    assert all(str(name).endswith(".json") for name in requested)
    # Never the branch name: every file comes from the immutable commit.
    assert {call["revision"] for call in api.download_calls} == {SAFETENSORS_SHA}
    assert api.model_info_calls[0]["files_metadata"] is True


def test_inspect_without_sizes_asks_the_hub_for_less(tmp_path):
    adapter, api = safetensors_adapter(tmp_path)

    adapter.inspect(ModelIdentity(repo_id="nDimensional/x"), with_sizes=False)

    assert api.model_info_calls[0]["files_metadata"] is False


# --------------------------------------------------------------------------- #
# Format detection
# --------------------------------------------------------------------------- #


def test_sizes_are_summed_per_format(tmp_path):
    adapter, _ = safetensors_adapter(tmp_path)

    metadata = adapter.inspect(ModelIdentity(repo_id="nDimensional/x"))
    observation = next(o for o in metadata.formats if o.format is ModelFormat.SAFETENSORS)

    assert observation.total_bytes == SAFETENSORS_WEIGHT_BYTES
    assert metadata.weight_bytes == SAFETENSORS_WEIGHT_BYTES
    assert metadata.primary_format is ModelFormat.SAFETENSORS
    assert metadata.parameter_count == 9409813744


def test_non_standard_safetensors_shard_names_are_detected(tmp_path):
    """The fixture uses ``model.safetensors-00001-of-00004.safetensors``."""
    adapter, _ = safetensors_adapter(tmp_path)

    metadata = adapter.inspect(ModelIdentity(repo_id="nDimensional/x"))
    observation = next(o for o in metadata.formats if o.format is ModelFormat.SAFETENSORS)

    assert sorted(observation.files) == sorted(SAFETENSORS_SHARDS)
    assert observation.complete is True
    assert any("4 shard(s), all listed" in note for note in observation.notes)


def test_safetensors_index_that_names_a_missing_shard_is_incomplete():
    files = [
        RepoFile(path=name, size=size)
        for name, size in list(SAFETENSORS_SHARDS.items())[:3]  # one shard withheld
    ]
    files.append(RepoFile(path="model.safetensors.index.json", size=78336))

    observations = detect_formats(
        files,
        shard_indexes={"model.safetensors.index.json": sorted(SAFETENSORS_SHARDS)},
    )
    observation = next(o for o in observations if o.format is ModelFormat.SAFETENSORS)

    assert observation.complete is False
    assert any("missing shard" in note for note in observation.notes)


def test_single_unsharded_safetensors_file_is_complete():
    observations = detect_formats([RepoFile(path="model.safetensors", size=1234)])

    observation = next(o for o in observations if o.format is ModelFormat.SAFETENSORS)
    assert observation.complete is True
    assert observation.total_bytes == 1234


def test_unread_index_falls_back_to_the_of_convention():
    files = [RepoFile(path=name, size=size) for name, size in SAFETENSORS_SHARDS.items()]
    files.append(RepoFile(path="model.safetensors.index.json", size=78336))

    observation = next(
        o for o in detect_formats(files, shard_indexes={}) if o.format is ModelFormat.SAFETENSORS
    )

    assert observation.complete is True
    assert any("was not read" in note for note in observation.notes)


def test_openvino_ir_needs_a_matching_bin():
    complete = detect_formats(
        [
            RepoFile(path="openvino_model.xml", size=100),
            RepoFile(path="openvino_model.bin", size=900),
        ]
    )[0]
    assert complete.format is ModelFormat.OPENVINO_IR
    assert complete.complete is True
    assert complete.total_bytes == 1000

    broken = detect_formats([RepoFile(path="openvino_model.xml", size=100)])[0]
    assert broken.complete is False
    assert any("Missing weight file" in note for note in broken.notes)


def test_openvino_bins_are_not_counted_as_pytorch_weights():
    observations = detect_formats(
        [
            RepoFile(path="openvino_language_model.xml", size=10),
            RepoFile(path="openvino_language_model.bin", size=990),
        ]
    )

    assert [o.format for o in observations] == [ModelFormat.OPENVINO_IR]


def test_multi_format_repository_reports_every_format():
    files = [
        RepoFile(path="openvino_model.xml", size=10),
        RepoFile(path="openvino_model.bin", size=90),
        RepoFile(path="model.safetensors", size=500),
        RepoFile(path="pytorch_model.bin", size=400),
        RepoFile(path="weights.gguf", size=300),
        RepoFile(path="config.json", size=7),
    ]

    observations = detect_formats(files)
    totals = {o.format: o.total_bytes for o in observations}

    assert [o.format for o in observations] == [
        ModelFormat.OPENVINO_IR,
        ModelFormat.SAFETENSORS,
        ModelFormat.PYTORCH_BIN,
        ModelFormat.GGUF,
    ]
    assert totals == {
        ModelFormat.OPENVINO_IR: 100,
        ModelFormat.SAFETENSORS: 500,
        ModelFormat.PYTORCH_BIN: 400,
        ModelFormat.GGUF: 300,
    }


def test_unknown_file_sizes_are_flagged_not_guessed():
    observation = detect_formats([RepoFile(path="a.gguf", size=None)])[0]

    assert observation.total_bytes == 0
    assert any("no size for 1 file" in note for note in observation.notes)


def test_a_repository_of_quantizations_is_not_one_giant_model(tmp_path):
    """``inspect`` hands the Hub's totalFileSize through to the observation.

    The repository publishes 38.5 GiB of GGUF across five files, but one model
    is 16.7 GiB. Sizing preflight from the sum would reserve scratch space for
    a model that does not exist.
    """
    api = FakeApi(info=gguf_info(), json_files={}, root=tmp_path / "hub")

    metadata = adapter_for(api).inspect(ModelIdentity(repo_id="HauhauCS/x"))

    assert metadata.weight_bytes == GGUF_PRIMARY_BYTES
    assert metadata.repo_weight_bytes == GGUF_TOTAL_BYTES
    observation = next(o for o in metadata.formats if o.format is ModelFormat.GGUF)
    assert any("Hub's GGUF metadata" in note for note in observation.notes)


def test_a_repository_without_gguf_metadata_estimates_from_the_file_listing(tmp_path):
    """No totalFileSize: the largest non-companion .gguf file is the model."""
    info = gguf_info()
    info.gguf = {"architecture": "qwen35"}
    api = FakeApi(info=info, json_files={}, root=tmp_path / "hub")

    metadata = adapter_for(api).inspect(ModelIdentity(repo_id="HauhauCS/x"))

    assert metadata.weight_bytes == GGUF_PRIMARY_BYTES
    assert metadata.repo_weight_bytes == GGUF_TOTAL_BYTES
    observation = next(o for o in metadata.formats if o.format is ModelFormat.GGUF)
    assert any("largest non-companion" in note for note in observation.notes)


def test_a_non_numeric_total_file_size_is_ignored(tmp_path):
    """A string where the Hub should have sent an int must not become a size."""
    info = gguf_info()
    info.gguf = {"architecture": "qwen35", "totalFileSize": "17920696704"}
    api = FakeApi(info=info, json_files={}, root=tmp_path / "hub")

    metadata = adapter_for(api).inspect(ModelIdentity(repo_id="HauhauCS/x"))

    # Falls back to the file listing rather than trusting an unusable value.
    assert metadata.weight_bytes == GGUF_PRIMARY_BYTES


def test_safetensors_shards_are_all_part_of_the_one_model(tmp_path):
    """Every shard is required, so primary and repository bytes agree."""
    adapter, _ = safetensors_adapter(tmp_path)

    metadata = adapter.inspect(ModelIdentity(repo_id="nDimensional/x"))
    observation = next(o for o in metadata.formats if o.format is ModelFormat.SAFETENSORS)

    assert observation.primary_bytes == observation.total_bytes == SAFETENSORS_WEIGHT_BYTES
    assert metadata.weight_bytes == metadata.repo_weight_bytes == SAFETENSORS_WEIGHT_BYTES


# --------------------------------------------------------------------------- #
# The GGUF-only repository with no config.json
# --------------------------------------------------------------------------- #


def test_gguf_only_repo_without_config_json_does_not_crash(tmp_path):
    api = FakeApi(info=gguf_info(), json_files={}, root=tmp_path / "hub")
    adapter = adapter_for(api)

    metadata = adapter.inspect(ModelIdentity(repo_id="HauhauCS/Qwen3.5-9B-Uncensored"))

    assert metadata.primary_format is ModelFormat.GGUF
    assert metadata.has_format(ModelFormat.GGUF)
    assert metadata.model_type is None
    assert metadata.architectures == []
    assert metadata.pipeline_kind is PipelineKind.UNKNOWN
    # Nothing was fetched: there is no config.json and no weight index.
    assert api.download_calls == []

    observation = next(o for o in metadata.formats if o.format is ModelFormat.GGUF)
    assert observation.total_bytes == GGUF_TOTAL_BYTES
    assert observation.primary_bytes == GGUF_PRIMARY_BYTES
    assert any("no config.json" in note for note in observation.notes)
    assert any("qwen3.5" in note for note in observation.notes)


def test_gguf_metadata_is_recorded_without_downloading_the_weights(tmp_path):
    api = FakeApi(info=gguf_info(), json_files={}, root=tmp_path / "hub")
    adapter = adapter_for(api)

    metadata = adapter.inspect(ModelIdentity(repo_id="HauhauCS/x"))

    assert metadata.gguf_metadata["architecture"] == "qwen35"
    assert metadata.gguf_metadata["context_length"] == 262144
    # The 20 kB chat template is recorded as elided, never inlined.
    assert metadata.gguf_metadata["chat_template"] == "<omitted: 20000 characters>"
    assert metadata.parameter_count == 8953803264
    assert not any(str(call["filename"]).endswith(".gguf") for call in api.download_calls)


def test_repository_tags_and_licence_survive_a_missing_config(tmp_path):
    api = FakeApi(info=gguf_info(), json_files={}, root=tmp_path / "hub")

    metadata = adapter_for(api).inspect(ModelIdentity(repo_id="HauhauCS/x"))

    assert "qwen3.5" in metadata.tags
    assert metadata.license == "apache-2.0"
    assert metadata.downloads == 714610
    assert metadata.likes == 2112


def test_licence_falls_back_to_a_license_tag(tmp_path):
    info = gguf_info()
    info.card_data = None
    api = FakeApi(info=info, json_files={}, root=tmp_path / "hub")

    metadata = adapter_for(api).inspect(ModelIdentity(repo_id="HauhauCS/x"))

    assert metadata.license == "apache-2.0"


OPENVINO_SIBLINGS = [
    FakeSibling("config.json", 2823),
    FakeSibling("openvino_language_model.xml", 5142667),
    FakeSibling("openvino_language_model.bin", 4614719754, FakeLfs(4614719754)),
    FakeSibling("openvino_text_embeddings_model.xml", 5936),
    FakeSibling("openvino_text_embeddings_model.bin", 1017615364, FakeLfs(1017615364)),
    FakeSibling("openvino_tokenizer.xml", 32586),
    FakeSibling("openvino_tokenizer.bin", 9631154, FakeLfs(9631154)),
    FakeSibling("openvino_detokenizer.xml", 15592),
    FakeSibling("openvino_detokenizer.bin", 3828184, FakeLfs(3828184)),
    FakeSibling("openvino_config.json", 1408),
    FakeSibling("tokenizer.json", 19989343, FakeLfs(19989343)),
]
OPENVINO_IR_BYTES = 5142667 + 4614719754 + 5936 + 1017615364 + 32586 + 9631154 + 15592 + 3828184


def test_prepared_openvino_repository_is_recognised(tmp_path):
    api = FakeApi(
        info=FakeModelInfo(
            sha=OPENVINO_SHA,
            siblings=list(OPENVINO_SIBLINGS),
            pipeline_tag="image-text-to-text",
            library_name="openvino",
            card_data=FakeCardData(license="apache-2.0"),
        ),
        json_files={"config.json": VLM_CONFIG},
        root=tmp_path / "hub",
    )

    metadata = adapter_for(api).inspect(ModelIdentity(repo_id="OpenVINO/Qwen3.5-9B-int4-ov"))

    assert metadata.resolved_revision == OPENVINO_SHA
    assert metadata.primary_format is ModelFormat.OPENVINO_IR
    observation = next(o for o in metadata.formats if o.format is ModelFormat.OPENVINO_IR)
    assert observation.complete is True
    assert observation.total_bytes == OPENVINO_IR_BYTES
    # openvino_config.json is not an IR file and tokenizer.json is not a weight.
    assert "openvino_config.json" not in observation.files
    assert metadata.library_name == "openvino"
    assert metadata.pipeline_kind is PipelineKind.VLM


# --------------------------------------------------------------------------- #
# Architecture, VLM classification and remote code
# --------------------------------------------------------------------------- #


def test_vision_config_makes_the_pipeline_a_vlm(tmp_path):
    adapter, _ = safetensors_adapter(tmp_path)

    metadata = adapter.inspect(ModelIdentity(repo_id="nDimensional/x"))

    assert metadata.architectures == ["Qwen3_5ForConditionalGeneration"]
    assert metadata.model_type == "qwen3_5"
    assert "vision_config" in metadata.config
    assert metadata.pipeline_kind is PipelineKind.VLM
    assert metadata.requires_remote_code is False
    assert metadata.remote_code_files == []


def test_auto_map_sets_requires_remote_code(tmp_path):
    info = safetensors_info()
    info.siblings = [
        *SAFETENSORS_SIBLINGS,
        FakeSibling("modeling_custom.py", 4096),
        FakeSibling("configuration_custom.py", 1024),
    ]
    api = FakeApi(
        info=info,
        json_files={
            "config.json": {
                **VLM_CONFIG,
                "auto_map": {
                    "AutoConfig": "acme/x--configuration_custom.CustomConfig",
                    "AutoModelForCausalLM": "modeling_custom.CustomForCausalLM",
                },
            },
            "model.safetensors.index.json": SAFETENSORS_INDEX,
        },
        root=tmp_path / "hub",
    )

    metadata = adapter_for(api).inspect(ModelIdentity(repo_id="acme/x"))

    assert metadata.requires_remote_code is True
    assert metadata.remote_code_files == ["configuration_custom.py", "modeling_custom.py"]


def test_detect_remote_code_handles_list_valued_auto_map():
    files = [
        RepoFile(path="tokenization_custom.py"),
        RepoFile(path="tokenization_custom_fast.py"),
    ]
    config = {
        "auto_map": {
            "AutoTokenizer": [
                "tokenization_custom.CustomTokenizer",
                "tokenization_custom_fast.CustomTokenizerFast",
            ]
        }
    }

    required, names = detect_remote_code(config, files)

    assert required is True
    assert names == ["tokenization_custom.py", "tokenization_custom_fast.py"]


def test_custom_python_without_auto_map_is_not_remote_code():
    required, names = detect_remote_code({"model_type": "llama"}, [RepoFile(path="modeling_x.py")])

    assert required is False
    assert names == []


def test_detect_remote_code_survives_a_missing_config():
    assert detect_remote_code(None, [RepoFile(path="a.gguf")]) == (False, [])


# --------------------------------------------------------------------------- #
# Error translation
# --------------------------------------------------------------------------- #


def hub_errors() -> list[Any]:
    """Every upstream error shape, paired with its expected classification."""
    from huggingface_hub.errors import (
        DisabledRepoError,
        EntryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
        HFValidationError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    message = f"denied (token {FAKE_TOKEN})"
    return [
        pytest.param(
            GatedRepoError(message, response=hub_response(403)),
            CODE_GATED,
            FailureCategory.GATED_MODEL_AUTH_REQUIRED,
            ExitCode.AUTH_REQUIRED,
            HubAuthError,
            id="gated",
        ),
        pytest.param(
            HfHubHTTPError(message, response=hub_response(401)),
            CODE_AUTH_REQUIRED,
            FailureCategory.GATED_MODEL_AUTH_REQUIRED,
            ExitCode.AUTH_REQUIRED,
            HubAuthError,
            id="http-401",
        ),
        pytest.param(
            HfHubHTTPError(message, response=hub_response(403)),
            CODE_AUTH_REQUIRED,
            FailureCategory.GATED_MODEL_AUTH_REQUIRED,
            ExitCode.AUTH_REQUIRED,
            HubAuthError,
            id="http-403",
        ),
        pytest.param(
            RepositoryNotFoundError(message, response=hub_response(404)),
            CODE_REPO_NOT_FOUND,
            FailureCategory.DOWNLOAD_FAILURE,
            ExitCode.SOURCE_RESOLUTION,
            SourceResolutionError,
            id="repo-missing",
        ),
        pytest.param(
            # VERIFIED against the live Hub: an anonymous request for a missing
            # repository is answered with 401, not 404.
            RepositoryNotFoundError(message, response=hub_response(401)),
            CODE_REPO_NOT_FOUND,
            FailureCategory.DOWNLOAD_FAILURE,
            ExitCode.SOURCE_RESOLUTION,
            SourceResolutionError,
            id="repo-missing-401-anonymous",
        ),
        pytest.param(
            DisabledRepoError(message, response=hub_response(403)),
            CODE_REPO_NOT_FOUND,
            FailureCategory.DOWNLOAD_FAILURE,
            ExitCode.SOURCE_RESOLUTION,
            SourceResolutionError,
            id="disabled-repo-403",
        ),
        pytest.param(
            RevisionNotFoundError(message, response=hub_response(404)),
            CODE_REVISION_NOT_FOUND,
            FailureCategory.DOWNLOAD_FAILURE,
            ExitCode.SOURCE_RESOLUTION,
            SourceResolutionError,
            id="revision-missing",
        ),
        pytest.param(
            LocalEntryNotFoundError(message),
            CODE_OFFLINE_MISSING,
            FailureCategory.DOWNLOAD_FAILURE,
            ExitCode.SOURCE_RESOLUTION,
            SourceResolutionError,
            id="local-entry-missing",
        ),
        pytest.param(
            OfflineModeIsEnabled(message),
            CODE_OFFLINE_MISSING,
            FailureCategory.DOWNLOAD_FAILURE,
            ExitCode.SOURCE_RESOLUTION,
            SourceResolutionError,
            id="offline-mode",
        ),
        pytest.param(
            EntryNotFoundError(message),
            CODE_FILE_NOT_FOUND,
            FailureCategory.DOWNLOAD_FAILURE,
            ExitCode.SOURCE_RESOLUTION,
            SourceResolutionError,
            id="entry-missing",
        ),
        pytest.param(
            HfHubHTTPError(message, response=hub_response(503)),
            CODE_HTTP_ERROR,
            FailureCategory.DOWNLOAD_FAILURE,
            ExitCode.SOURCE_RESOLUTION,
            SourceResolutionError,
            id="http-503",
        ),
        pytest.param(
            HfHubHTTPError(message, response=hub_response(429)),
            CODE_RATE_LIMITED,
            FailureCategory.DOWNLOAD_FAILURE,
            ExitCode.SOURCE_RESOLUTION,
            SourceResolutionError,
            id="http-429-rate-limited",
        ),
        pytest.param(
            ConnectionError(message),
            CODE_NETWORK_ERROR,
            FailureCategory.DOWNLOAD_FAILURE,
            ExitCode.SOURCE_RESOLUTION,
            SourceResolutionError,
            id="network",
        ),
        pytest.param(
            HFValidationError(message),
            CODE_INVALID_REPO_ID,
            FailureCategory.DOWNLOAD_FAILURE,
            ExitCode.SOURCE_RESOLUTION,
            SourceResolutionError,
            id="invalid-repo-id",
        ),
        pytest.param(
            RuntimeError(message),
            CODE_UNKNOWN,
            FailureCategory.UNKNOWN_RUNTIME_ERROR,
            ExitCode.INTERNAL,
            YongaError,
            id="unknown",
        ),
    ]


@pytest.mark.parametrize(("error", "code", "category", "exit_code", "error_type"), hub_errors())
def test_hub_errors_map_to_typed_diagnostics(error, code, category, exit_code, error_type):
    model = ModelIdentity(repo_id="acme/model", revision="main")

    translated = translate_hub_error(error, model=model, operation="inspection")

    assert isinstance(translated, error_type)
    assert translated.diagnostic.code == code
    assert translated.diagnostic.category is category
    assert translated.exit_code() is exit_code
    assert translated.diagnostic.context["repo_id"] == "acme/model"
    assert translated.diagnostic.remediation


@pytest.mark.parametrize(("error", "code", "_c", "_e", "_t"), hub_errors())
def test_hub_error_messages_never_contain_a_token(error, code, _c, _e, _t):
    translated = translate_hub_error(error, model=ModelIdentity(repo_id="acme/model"))
    payload = json.dumps(translated.diagnostic.to_json())

    assert FAKE_TOKEN not in payload
    assert FAKE_TOKEN not in str(translated)
    assert REDACTED in (translated.diagnostic.raw_excerpt or "")


def test_gated_remediation_mentions_hf_auth_login():
    from huggingface_hub.errors import GatedRepoError

    translated = translate_hub_error(
        GatedRepoError("gated", response=hub_response(403)),
        model=ModelIdentity(repo_id="acme/model"),
    )

    assert any("hf auth login" in line for line in translated.diagnostic.remediation)


def test_repo_not_found_remediation_covers_the_login_case():
    """The Hub conflates "missing" and "no access", so both hints must be offered."""
    from huggingface_hub.errors import RepositoryNotFoundError

    translated = translate_hub_error(
        RepositoryNotFoundError("gone", response=hub_response(401)),
        model=ModelIdentity(repo_id="acme/model"),
    )

    assert translated.diagnostic.code == CODE_REPO_NOT_FOUND
    assert any("spelling" in line for line in translated.diagnostic.remediation)
    assert any("hf auth login" in line for line in translated.diagnostic.remediation)


def test_a_bare_401_without_a_repository_error_is_an_auth_failure():
    """``whoami`` and per-file 403s still map to hub.auth.required."""
    from huggingface_hub.errors import HfHubHTTPError

    translated = translate_hub_error(
        HfHubHTTPError("nope", response=hub_response(401)),
        model=ModelIdentity(repo_id="acme/model"),
    )

    assert translated.diagnostic.code == CODE_AUTH_REQUIRED
    assert translated.exit_code() is ExitCode.AUTH_REQUIRED


def test_a_typed_error_passes_through_unchanged():
    original = SourceResolutionError("already typed")

    assert translate_hub_error(original) is original


def test_offline_turns_a_missing_entry_into_an_offline_diagnostic():
    from huggingface_hub.errors import EntryNotFoundError

    translated = translate_hub_error(
        EntryNotFoundError("nope"),
        model=ModelIdentity(repo_id="acme/model"),
        offline=True,
    )

    assert translated.diagnostic.code == CODE_OFFLINE_MISSING


def test_inspect_translates_a_hub_failure():
    from huggingface_hub.errors import GatedRepoError

    api = FakeApi(model_info_error=GatedRepoError("gated", response=hub_response(403)))
    adapter = adapter_for(api)

    with pytest.raises(HubAuthError) as excinfo:
        adapter.inspect(ModelIdentity(repo_id="acme/model"))

    assert excinfo.value.diagnostic.code == CODE_GATED
    assert excinfo.value.exit_code() is ExitCode.AUTH_REQUIRED


def test_inspect_degrades_instead_of_failing_when_config_json_is_unreadable(tmp_path):
    info = safetensors_info()
    api = FakeApi(
        info=info,
        json_files={"model.safetensors.index.json": SAFETENSORS_INDEX},
        root=tmp_path / "hub",
    )
    assert api.info is not None
    api.info.config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
    }

    metadata = adapter_for(api).inspect(ModelIdentity(repo_id="acme/x"))

    assert metadata.model_type == "qwen3_5"
    assert metadata.config == {}
    notes = [note for observation in metadata.formats for note in observation.notes]
    assert any("config.json could not be read" in note for note in notes)


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #


def test_download_uses_the_resolved_sha_not_the_branch_name(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    api = FakeApi(info=safetensors_info(), snapshot=snapshot)
    adapter = adapter_for(api)

    result = adapter.download(ModelIdentity(repo_id="nDimensional/x", revision="main"))

    assert result == snapshot
    assert api.snapshot_calls[0]["revision"] == SAFETENSORS_SHA
    assert api.snapshot_calls[0]["revision"] != "main"
    # The branch had to be resolved first.
    assert api.model_info_calls[0]["revision"] == "main"


def test_download_skips_resolution_when_already_pinned(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    api = FakeApi(snapshot=snapshot)
    adapter = adapter_for(api)

    adapter.download(ModelIdentity(repo_id="nDimensional/x", revision=SAFETENSORS_SHA))

    assert api.model_info_calls == []
    assert api.snapshot_calls[0]["revision"] == SAFETENSORS_SHA


def test_download_forwards_patterns(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    api = FakeApi(snapshot=snapshot)
    adapter = adapter_for(api)

    adapter.download(
        ModelIdentity(repo_id="acme/x", revision=SAFETENSORS_SHA),
        allow_patterns=["*.json"],
        ignore_patterns=["*.gguf"],
    )

    assert api.snapshot_calls[0]["allow_patterns"] == ["*.json"]
    assert api.snapshot_calls[0]["ignore_patterns"] == ["*.gguf"]


def test_offline_mode_passes_local_files_only(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    hf_home = tmp_path / "hf"
    cache_snapshot(hf_home, "acme/x", SAFETENSORS_SHA, {"config.json": b"{}"}, refs={"main": True})
    api = FakeApi(snapshot=snapshot, model_info_error=AssertionError("no Hub call offline"))
    adapter = adapter_for(api, Settings(offline=True, hf_home=hf_home))

    adapter.download(ModelIdentity(repo_id="acme/x", revision="main"))

    assert api.snapshot_calls[0]["local_files_only"] is True
    # The branch is pinned through the cache's own refs/ mapping, not the Hub.
    assert api.snapshot_calls[0]["revision"] == SAFETENSORS_SHA
    assert api.model_info_calls == []


def test_offline_miss_is_reported_as_hub_offline_missing(tmp_path):
    from huggingface_hub.errors import LocalEntryNotFoundError

    api = FakeApi(snapshot_error=LocalEntryNotFoundError("not cached"))
    adapter = adapter_for(api, Settings(offline=True))

    with pytest.raises(SourceResolutionError) as excinfo:
        adapter.download(ModelIdentity(repo_id="acme/x", revision=SAFETENSORS_SHA))

    assert excinfo.value.diagnostic.code == CODE_OFFLINE_MISSING
    assert excinfo.value.exit_code() is ExitCode.SOURCE_RESOLUTION


def test_download_honours_a_configured_hf_home(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    api = FakeApi(snapshot=snapshot)
    adapter = adapter_for(api, Settings(hf_home=tmp_path / "hf"))

    adapter.download(ModelIdentity(repo_id="acme/x", revision=SAFETENSORS_SHA))

    assert api.snapshot_calls[0]["cache_dir"] == tmp_path / "hf" / "hub"


def test_download_emits_progress_events(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    api = FakeApi(snapshot=snapshot)
    ctx = make_job_context(tmp_path)
    sink = ctx.progress
    assert isinstance(sink, CollectingSink)

    adapter_for(api).download(ModelIdentity(repo_id="acme/x", revision=SAFETENSORS_SHA), ctx=ctx)

    events = sink.of_kind(ProgressKind.DOWNLOAD)
    statuses = [event.status for event in events]
    assert statuses[0] == "started"
    assert statuses[-1] == "completed"
    byte_events = [event for event in events if event.status == "downloading"]
    assert byte_events, "the Hub byte counter should reach the progress sink"
    assert byte_events[-1].current == 4096
    assert byte_events[-1].total == 4096
    assert byte_events[-1].data["unit"] == "bytes"


def test_download_without_a_context_passes_no_tqdm_class(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    api = FakeApi(snapshot=snapshot)

    adapter_for(api).download(ModelIdentity(repo_id="acme/x", revision=SAFETENSORS_SHA))

    assert api.snapshot_calls[0]["tqdm_class"] is None


def test_download_stops_when_the_job_is_cancelled(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    api = FakeApi(snapshot=snapshot)
    ctx = make_job_context(tmp_path)
    ctx.cancelled.set()

    with pytest.raises(CancelledError) as excinfo:
        adapter_for(api).download(
            ModelIdentity(repo_id="acme/x", revision=SAFETENSORS_SHA), ctx=ctx
        )

    assert excinfo.value.exit_code() is ExitCode.CANCELLED


def test_download_never_copies_the_snapshot(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    api = FakeApi(snapshot=snapshot)

    result = adapter_for(api).download(ModelIdentity(repo_id="acme/x", revision=SAFETENSORS_SHA))

    assert result == snapshot
    assert list(tmp_path.iterdir()) == [snapshot]


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #


def test_auth_info_without_a_token(monkeypatch):
    monkeypatch.setattr("yonga.adapters.hf.get_token", lambda: None)
    adapter = adapter_for(FakeApi())

    info = adapter.auth_info()

    assert info.logged_in is False
    assert info.user is None
    assert info.token_source is None
    assert info.cache_dir is not None


def test_auth_info_with_a_token_reports_the_user_not_the_token(monkeypatch, tmp_path):
    monkeypatch.setattr("yonga.adapters.hf.get_token", lambda: FAKE_TOKEN)
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    adapter = adapter_for(
        FakeApi(whoami_result={"name": "octocat"}), Settings(hf_home=tmp_path / "hf")
    )

    info = adapter.auth_info()

    assert info.logged_in is True
    assert info.user == "octocat"
    assert info.token_source == "environment:HF_TOKEN"
    assert info.cache_dir == tmp_path / "hf" / "hub"
    assert FAKE_TOKEN not in json.dumps(info.model_dump(mode="json"))


def test_auth_info_reports_a_rejected_token_as_logged_out(monkeypatch):
    from huggingface_hub.errors import HfHubHTTPError

    monkeypatch.setattr("yonga.adapters.hf.get_token", lambda: FAKE_TOKEN)
    api = FakeApi(whoami_error=HfHubHTTPError("bad token", response=hub_response(401)))

    info = adapter_for(api).auth_info()

    assert info.logged_in is False
    assert info.user is None


def test_auth_info_keeps_the_token_when_the_network_is_down(monkeypatch):
    monkeypatch.setattr("yonga.adapters.hf.get_token", lambda: FAKE_TOKEN)
    api = FakeApi(whoami_error=ConnectionError("no route to host"))

    info = adapter_for(api).auth_info()

    assert info.logged_in is True
    assert info.user is None


def test_auth_info_offline_does_not_call_whoami(monkeypatch):
    monkeypatch.setattr("yonga.adapters.hf.get_token", lambda: FAKE_TOKEN)
    api = FakeApi(whoami_error=AssertionError("whoami must not run offline"))

    info = adapter_for(api, Settings(offline=True)).auth_info()

    assert info.offline is True
    assert info.logged_in is True
    assert info.user is None


def test_a_rate_limit_says_to_log_in_not_to_wait_for_a_5xx() -> None:
    from huggingface_hub.errors import HfHubHTTPError

    error = HfHubHTTPError("429 Too Many Requests", response=hub_response(429))

    translated = translate_hub_error(error, model=ModelIdentity(repo_id="acme/model"))

    assert "hf auth login" in " ".join(translated.diagnostic.remediation)
    assert "5xx" not in " ".join(translated.diagnostic.remediation)


# --------------------------------------------------------------------------- #
# Offline: everything from the local cache, nothing from the Hub
# --------------------------------------------------------------------------- #

OFFLINE_REPO = "acme/tiny-chat"
OFFLINE_CONFIG = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}
OFFLINE_CARD = (
    b"---\nlicense: apache-2.0\npipeline_tag: text-generation\n"
    b"library_name: transformers\ntags:\n- chat\n---\n# Tiny chat\n"
)
OTHER_SHA = "1" * 40


@pytest.fixture(autouse=True)
def _hub_offline_env_unset(monkeypatch):
    """``HF_HUB_OFFLINE`` is off: yonga's own offline setting has to be enough."""
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)


def offline_adapter(hf_home: Path) -> tuple[HuggingFaceAdapter, FakeApi]:
    """An offline adapter whose every Hub call would fail loudly and be recorded."""
    api = FakeApi(
        model_info_error=AssertionError("model_info must not run offline"),
        download_error=AssertionError("hf_hub_download must not run offline"),
    )
    return adapter_for(api, Settings(offline=True, hf_home=hf_home)), api


def assert_no_hub_call(api: FakeApi) -> None:
    assert api.model_info_calls == []
    assert api.download_calls == []
    assert api.snapshot_calls == []


def cached_chat_model(hf_home: Path, *, refs: dict[str, bool] | None = None) -> Path:
    return cache_snapshot(
        hf_home,
        OFFLINE_REPO,
        SAFETENSORS_SHA,
        {
            "config.json": json.dumps(OFFLINE_CONFIG).encode(),
            "README.md": OFFLINE_CARD,
            "model.safetensors": safetensors_bytes(
                {"embed.weight": ("BF16", [4, 8]), "norm.weight": ("F32", [3])}
            ),
            "tokenizer.json": b"{}",
        },
        refs=refs,
    )


def test_offline_inspect_reads_the_cached_snapshot_and_never_calls_the_hub(tmp_path):
    cached_chat_model(tmp_path / "hf", refs={"main": True})
    adapter, api = offline_adapter(tmp_path / "hf")

    metadata = adapter.inspect(ModelIdentity(repo_id=OFFLINE_REPO))

    assert_no_hub_call(api)
    assert metadata.resolved_revision == SAFETENSORS_SHA
    assert metadata.model_type == "qwen3"
    assert metadata.architectures == ["Qwen3ForCausalLM"]
    assert metadata.license == "apache-2.0"
    assert metadata.pipeline_tag == "text-generation"
    assert metadata.library_name == "transformers"
    assert "chat" in metadata.tags
    # Counted from the cached header the way the Hub counts it, so a profile's
    # size band is judged on the same number online and offline.
    assert metadata.parameter_count == 35
    assert metadata.parameter_dtypes == {"BF16": 32, "F32": 3}
    assert metadata.primary_format is ModelFormat.SAFETENSORS
    assert metadata.formats[0].complete is True
    by_path = {entry.path: entry for entry in metadata.files}
    assert by_path["model.safetensors"].lfs is True
    assert by_path["model.safetensors"].size == len(
        (tmp_path / "hf" / "hub")
        .joinpath("models--acme--tiny-chat", "snapshots", SAFETENSORS_SHA, "model.safetensors")
        .read_bytes()
    )
    assert by_path["tokenizer.json"].lfs is False
    # What only the Hub knows is absent, not invented.
    assert metadata.downloads is None
    assert metadata.likes is None
    notes = " ".join(note for observation in metadata.formats for note in observation.notes)
    assert "local Hugging Face cache" in notes


def test_offline_inspect_of_an_uncached_model_refuses_without_any_request(tmp_path):
    adapter, api = offline_adapter(tmp_path / "hf")

    with pytest.raises(SourceResolutionError) as excinfo:
        adapter.inspect(ModelIdentity(repo_id=OFFLINE_REPO))

    assert_no_hub_call(api)
    assert excinfo.value.diagnostic.code == CODE_OFFLINE_MISSING
    assert excinfo.value.exit_code() is ExitCode.SOURCE_RESOLUTION
    assert "Offline mode is active" in excinfo.value.diagnostic.message


def test_hf_hub_offline_alone_also_keeps_inspect_off_the_network(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    cached_chat_model(tmp_path / "hf", refs={"main": True})
    api = FakeApi(model_info_error=AssertionError("model_info must not run offline"))
    adapter = adapter_for(api, Settings(offline=False, hf_home=tmp_path / "hf"))

    assert adapter.inspect(ModelIdentity(repo_id=OFFLINE_REPO)).model_type == "qwen3"
    assert api.model_info_calls == []


def test_a_snapshot_downloaded_by_sha_is_found_without_a_ref(tmp_path):
    """yonga downloads by commit SHA, and the Hub records no refs/main for that."""
    cached_chat_model(tmp_path / "hf")
    adapter, api = offline_adapter(tmp_path / "hf")

    metadata = adapter.inspect(ModelIdentity(repo_id=OFFLINE_REPO))

    assert_no_hub_call(api)
    assert metadata.resolved_revision == SAFETENSORS_SHA
    notes = " ".join(note for observation in metadata.formats for note in observation.notes)
    assert "only cached snapshot" in notes


def test_an_explicit_branch_is_never_guessed_from_an_unrecorded_snapshot(tmp_path):
    cached_chat_model(tmp_path / "hf")
    adapter, _api = offline_adapter(tmp_path / "hf")

    with pytest.raises(SourceResolutionError) as excinfo:
        adapter.inspect(ModelIdentity(repo_id=OFFLINE_REPO, revision="dev"))

    assert excinfo.value.diagnostic.code == CODE_OFFLINE_MISSING
    assert "no record of which one 'dev' is" in excinfo.value.diagnostic.message


def test_several_unrecorded_snapshots_are_refused_and_listed(tmp_path):
    cached_chat_model(tmp_path / "hf")
    cache_snapshot(tmp_path / "hf", OFFLINE_REPO, OTHER_SHA, {"config.json": b"{}"})
    adapter, _api = offline_adapter(tmp_path / "hf")

    with pytest.raises(SourceResolutionError) as excinfo:
        adapter.inspect(ModelIdentity(repo_id=OFFLINE_REPO))

    message = excinfo.value.diagnostic.message
    assert SAFETENSORS_SHA in message
    assert OTHER_SHA in message
    assert any("@<sha>" in item for item in excinfo.value.diagnostic.remediation)


def test_an_explicit_commit_is_read_from_its_snapshot(tmp_path):
    cached_chat_model(tmp_path / "hf")
    cache_snapshot(tmp_path / "hf", OFFLINE_REPO, OTHER_SHA, {"config.json": b"{}"})
    adapter, api = offline_adapter(tmp_path / "hf")

    metadata = adapter.inspect(ModelIdentity(repo_id=OFFLINE_REPO, revision=SAFETENSORS_SHA))

    assert_no_hub_call(api)
    assert metadata.resolved_revision == SAFETENSORS_SHA


def test_a_revision_cannot_point_outside_the_cache(tmp_path):
    cached_chat_model(tmp_path / "hf", refs={"main": True})
    outside = tmp_path / "hf" / "hub" / "escape"
    outside.write_text(SAFETENSORS_SHA, encoding="utf-8")
    adapter, _api = offline_adapter(tmp_path / "hf")

    with pytest.raises(SourceResolutionError) as excinfo:
        adapter.inspect(ModelIdentity(repo_id=OFFLINE_REPO, revision="../../escape"))

    assert excinfo.value.diagnostic.code == CODE_OFFLINE_MISSING


def test_metadata_alone_in_the_cache_is_not_a_weightless_model(tmp_path):
    """An online inspect caches config.json; offline that must not look like a model."""
    cache_snapshot(
        tmp_path / "hf",
        OFFLINE_REPO,
        SAFETENSORS_SHA,
        {"config.json": json.dumps(OFFLINE_CONFIG).encode()},
        refs={"main": True},
    )
    adapter, _api = offline_adapter(tmp_path / "hf")

    with pytest.raises(SourceResolutionError) as excinfo:
        adapter.inspect(ModelIdentity(repo_id=OFFLINE_REPO))

    assert excinfo.value.diagnostic.code == CODE_OFFLINE_MISSING
    assert "none of its model weights" in excinfo.value.diagnostic.message


def test_offline_parameters_need_every_indexed_shard(tmp_path):
    shard = safetensors_bytes({"w": ("BF16", [2, 5])})
    index = {
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        }
    }
    files = {
        "config.json": json.dumps(OFFLINE_CONFIG).encode(),
        "model.safetensors.index.json": json.dumps(index).encode(),
        "model-00001-of-00002.safetensors": shard,
    }
    cache_snapshot(tmp_path / "hf", OFFLINE_REPO, SAFETENSORS_SHA, files, refs={"main": True})
    adapter, _api = offline_adapter(tmp_path / "hf")

    partial = adapter.inspect(ModelIdentity(repo_id=OFFLINE_REPO))

    # A partial sum would put a large model into a small model's size band.
    assert partial.parameter_count is None
    assert partial.formats[0].complete is False

    cache_snapshot(
        tmp_path / "hf",
        OFFLINE_REPO,
        SAFETENSORS_SHA,
        {"model-00002-of-00002.safetensors": safetensors_bytes({"v": ("F16", [7])})},
    )
    whole = adapter.inspect(ModelIdentity(repo_id=OFFLINE_REPO))

    assert whole.parameter_count == 17
    assert whole.parameter_dtypes == {"BF16": 10, "F16": 7}
    assert whole.formats[0].complete is True


def test_offline_gguf_identity_comes_from_the_cached_header(tmp_path):
    from tests.unit.test_gguf import gguf_bytes

    cache_snapshot(
        tmp_path / "hf",
        OFFLINE_REPO,
        GGUF_SHA,
        {
            "model-Q4_K_M.gguf": gguf_bytes("qwen2", {"Q4_K": 2}),
            "mmproj-model-f16.gguf": gguf_bytes("clip", {"F32": 1}),
        },
        refs={"main": True},
    )
    adapter, api = offline_adapter(tmp_path / "hf")

    metadata = adapter.inspect(ModelIdentity(repo_id=OFFLINE_REPO))

    assert_no_hub_call(api)
    assert metadata.gguf_metadata["architecture"] == "qwen2"
    assert metadata.primary_format is ModelFormat.GGUF


def test_offline_revision_resolution_uses_the_cache_refs(tmp_path):
    cached_chat_model(tmp_path / "hf", refs={"main": True})
    adapter, api = offline_adapter(tmp_path / "hf")

    assert adapter.resolve_revision(ModelIdentity(repo_id=OFFLINE_REPO)) == SAFETENSORS_SHA
    assert (
        adapter.resolve_revision(ModelIdentity(repo_id=OFFLINE_REPO, revision="main"))
        == SAFETENSORS_SHA
    )
    assert_no_hub_call(api)


def test_offline_search_says_search_needs_the_network(tmp_path):
    adapter, _api = offline_adapter(tmp_path / "hf")

    with pytest.raises(SourceResolutionError) as excinfo:
        adapter.search("smollm2")

    diagnostic = excinfo.value.diagnostic
    assert diagnostic.code == CODE_OFFLINE_SEARCH
    assert "network" in diagnostic.message
    assert "not fully present" not in diagnostic.message
    assert "cache" not in diagnostic.title.lower()
