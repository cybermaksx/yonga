"""Direct GGUF import: the header decides, the registry declares, nothing is guessed.

On the reference host, OpenVINO GenAI 2026.4 imported every GGUF file whose
architecture was llama, qwen2 or qwen3 and whose tensors were all F32, Q4_0,
Q4_K, Q6_K or Q8_0 -- and refused every other one, including two ``Q4_K_M``
files that carried a single unsupported tensor type. These tests pin the
parser that reads that from a header, the registry rule that records it, and
the resolver that chooses a file with it.
"""

from __future__ import annotations

import struct
import textwrap
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from yonga.adapters.gguf_header import (
    GGUFFormatError,
    GGUFHeader,
    parse_gguf_header,
    read_local_header,
)
from yonga.adapters.hf import detect_formats
from yonga.application.jobs import JobContext
from yonga.application.resolve import (
    SourcePolicy,
    SourceResolver,
    gguf_model_files,
    gguf_quant_label,
)
from yonga.domain.diagnostics import SourceResolutionError
from yonga.domain.hardware import HuggingFaceAuthInfo
from yonga.domain.models import ModelFormat, ModelIdentity, ModelMetadata, RepoFile
from yonga.registry.loader import (
    load_builtin_registry,
    load_registry_from_documents,
    parse_document,
)
from yonga.registry.resolver import RegistryResolver

TYPES = {"F32": 0, "Q4_0": 2, "Q5_0": 6, "Q8_0": 8, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14}


def gguf_bytes(architecture: str, tensors: dict[str, int], *, vocab: int = 50) -> bytes:
    """A syntactically real GGUF v3 header: metadata, a vocabulary array, tensor infos."""

    def string(text: str) -> bytes:
        data = text.encode()
        return struct.pack("<Q", len(data)) + data

    kv = [
        string("general.architecture") + struct.pack("<I", 8) + string(architecture),
        string("general.file_type") + struct.pack("<I", 4) + struct.pack("<I", 15),
        string("tokenizer.ggml.tokens")
        + struct.pack("<I", 9)
        + struct.pack("<I", 8)
        + struct.pack("<Q", vocab)
        + b"".join(string(f"tok{i}") for i in range(vocab)),
        string(f"{architecture}.block_count") + struct.pack("<I", 4) + struct.pack("<I", 2),
    ]
    infos = []
    index = 0
    for type_name, count in tensors.items():
        for _ in range(count):
            infos.append(
                string(f"blk.{index}.weight")
                + struct.pack("<I", 2)
                + struct.pack("<QQ", 64, 64)
                + struct.pack("<I", TYPES[type_name])
                + struct.pack("<Q", index * 4096)
            )
            index += 1
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", index, len(kv))
    return head + b"".join(kv) + b"".join(infos) + b"\x00" * 64


def reader_for(data: bytes) -> Callable[[int, int], bytes]:
    def read(offset: int, length: int) -> bytes:
        return data[offset : offset + length]

    return read


# --- the header -----------------------------------------------------------------------


def test_the_header_gives_the_architecture_and_every_tensor_type() -> None:
    data = gguf_bytes("qwen2", {"F32": 3, "Q4_K": 5, "Q5_0": 2})

    header = parse_gguf_header(reader_for(data), chunk=16)  # tiny chunks: many fetches

    assert header.architecture == "qwen2"
    assert header.tensor_types == {"F32": 3, "Q4_K": 5, "Q5_0": 2}
    assert header.tensor_count == 10
    assert header.metadata["tokenizer.ggml.tokens.length"] == 50
    assert "tokenizer.ggml.tokens" not in header.metadata, "array contents are skipped"
    assert header.unsupported_types({"F32", "Q4_K", "Q6_K"}) == ["Q5_0"]


def test_a_local_file_is_read_the_same_way(tmp_path: Path) -> None:
    path = tmp_path / "m.gguf"
    path.write_bytes(gguf_bytes("llama", {"Q8_0": 4}))

    assert read_local_header(path).tensor_types == {"Q8_0": 4}


@pytest.mark.parametrize(
    "data",
    [b"NOPE" + b"\x00" * 64, b"GGUF" + struct.pack("<I", 9) + b"\x00" * 64, b"GGUF\x03"],
    ids=["bad-magic", "unknown-version", "truncated"],
)
def test_malformed_headers_raise_a_typed_error(data: bytes) -> None:
    with pytest.raises(GGUFFormatError):
        parse_gguf_header(reader_for(data))


def test_an_absurd_length_is_refused_before_it_is_allocated() -> None:
    data = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 1) + struct.pack("<Q", 2**40)

    with pytest.raises(GGUFFormatError, match="implausibly long"):
        parse_gguf_header(reader_for(data + b"\x00" * 64))


# --- file names -----------------------------------------------------------------------


def test_side_cars_and_split_parts_are_not_model_files() -> None:
    names = [
        "model-Q4_K_M.gguf",
        "mmproj-model-F16.gguf",
        "big-Q8_0-00001-of-00003.gguf",
        "README.md",
    ]

    assert gguf_model_files(names) == ["model-Q4_K_M.gguf"]


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("qwen2.5-0.5b-instruct-q4_k_m.gguf", "q4_k_m"),
        ("Qwen3-0.6B-Q8_0.gguf", "q8_0"),
        ("Llama-3.2-1B-Instruct-UD-IQ1_S.gguf", "ud-iq1_s"),
        ("model-BF16.gguf", "bf16"),
    ],
)
def test_the_quantization_label_names_the_artifact(name: str, label: str) -> None:
    assert gguf_quant_label(name) == label


def test_a_name_without_a_label_still_gets_a_distinct_one() -> None:
    assert gguf_quant_label("weights.gguf") != gguf_quant_label("other.gguf")


# --- the registry rule ----------------------------------------------------------------


def test_the_shipped_rule_needs_the_genai_it_was_verified_on() -> None:
    registry = load_builtin_registry()

    verified, _ = RegistryResolver(registry, versions={"openvino_genai": "2026.4.0"}).resolve_gguf()
    older, why = RegistryResolver(registry, versions={"openvino_genai": "2026.3.0"}).resolve_gguf()
    unknown, _ = RegistryResolver(registry, versions={}).resolve_gguf()

    assert verified is not None
    assert verified.header_architectures == ["llama", "qwen2", "qwen3"]
    assert set(verified.tensor_types) == {"F32", "F16", "Q4_0", "Q4_K", "Q6_K", "Q8_0"}
    assert older is None and "does not satisfy" in why[0]
    assert unknown is None, "an unverifiable GenAI version is not support"


# --- choosing a file ------------------------------------------------------------------

REPO = "acme/model-GGUF"
SHA = "c" * 40
FILES = (
    ("model-Q4_K_M.gguf", 470),
    ("model-Q4_0.gguf", 409),
    ("model-Q8_0.gguf", 644),
    ("mmproj-model-F16.gguf", 100),
)


class HeaderHub:
    """A Hub that serves canned GGUF headers and records which it was asked for."""

    def __init__(self, headers: dict[str, bytes]) -> None:
        self.headers = headers
        self.read: list[str] = []

    def resolve_revision(self, model: ModelIdentity) -> str:
        return SHA

    def inspect(self, model: ModelIdentity, *, with_sizes: bool = True) -> ModelMetadata:
        raise AssertionError("resolution must not inspect other repositories")

    def download(
        self,
        source: ModelIdentity,
        *,
        ctx: JobContext | None = None,
        allow_patterns: Sequence[str] | None = None,
        ignore_patterns: Sequence[str] | None = None,
    ) -> Path:
        raise AssertionError("resolution must not download")

    def auth_info(self) -> HuggingFaceAuthInfo:
        return HuggingFaceAuthInfo(logged_in=False)

    def gguf_header(self, model: ModelIdentity, filename: str) -> GGUFHeader:
        self.read.append(filename)
        return parse_gguf_header(reader_for(self.headers[filename]))


def metadata(architecture: str = "qwen2") -> ModelMetadata:
    files = [RepoFile(path=path, size=size, lfs=True) for path, size in FILES]
    return ModelMetadata(
        identity=ModelIdentity(repo_id=REPO),
        resolved_revision=SHA,
        files=files,
        formats=detect_formats(files),
        gguf_metadata={"architecture": architecture},
    )


def rule_resolver() -> RegistryResolver:
    return RegistryResolver(load_builtin_registry(), versions={"openvino_genai": "2026.4.0"})


def standard_headers(q4km: dict[str, int]) -> dict[str, bytes]:
    return {
        "model-Q4_K_M.gguf": gguf_bytes("qwen2", q4km),
        "model-Q4_0.gguf": gguf_bytes("qwen2", {"F32": 2, "Q4_0": 8, "Q8_0": 1}),
        "model-Q8_0.gguf": gguf_bytes("qwen2", {"F32": 2, "Q8_0": 9}),
    }


def test_the_preferred_file_is_used_when_its_tensors_are_all_readable() -> None:
    hub = HeaderHub(standard_headers({"F32": 2, "Q4_K": 6, "Q6_K": 2}))

    resolved = SourceResolver(hub).resolve_gguf(metadata(), registry_resolver=rule_resolver())

    assert resolved.format is ModelFormat.GGUF
    assert resolved.gguf_file == "model-Q4_K_M.gguf"
    assert resolved.gguf_file_bytes == 470
    assert hub.read == ["model-Q4_K_M.gguf"]
    assert any(e.kind == "gguf_header" for e in resolved.evidence)


def test_a_file_holding_one_unreadable_tensor_type_is_passed_over() -> None:
    """The reference host's Qwen2.5 Q4_K_M: mostly Q4_K, but Q5_0 tensors too."""
    hub = HeaderHub(standard_headers({"F32": 2, "Q4_K": 2, "Q5_0": 6, "Q6_K": 2}))

    resolved = SourceResolver(hub).resolve_gguf(metadata(), registry_resolver=rule_resolver())

    assert resolved.gguf_file == "model-Q4_0.gguf"
    assert any("Passed over model-Q4_K_M.gguf: holds Q5_0 tensors" in r for r in resolved.reasons)


def test_a_named_file_is_used_or_refused_but_never_swapped() -> None:
    hub = HeaderHub(standard_headers({"F32": 2, "Q5_0": 6}))
    policy = SourcePolicy(gguf_file="model-Q8_0.gguf")

    chosen = SourceResolver(hub).resolve_gguf(
        metadata(), registry_resolver=rule_resolver(), policy=policy
    )
    assert chosen.gguf_file == "model-Q8_0.gguf"

    with pytest.raises(SourceResolutionError) as caught:
        SourceResolver(hub).resolve_gguf(
            metadata(),
            registry_resolver=rule_resolver(),
            policy=SourcePolicy(gguf_file="model-Q4_K_M.gguf"),
        )
    assert caught.value.diagnostic.code == "source.gguf_no_importable_file"
    assert "Q5_0" in caught.value.diagnostic.message


def test_an_unknown_named_file_lists_the_real_ones() -> None:
    with pytest.raises(SourceResolutionError) as caught:
        SourceResolver(HeaderHub({})).resolve_gguf(
            metadata(),
            registry_resolver=rule_resolver(),
            policy=SourcePolicy(gguf_file="nope.gguf"),
        )
    assert caught.value.diagnostic.code == "source.gguf_file_not_found"
    assert "model-Q8_0.gguf" in caught.value.diagnostic.remediation[0]
    assert "mmproj" not in caught.value.diagnostic.remediation[0]


def test_an_unimported_architecture_is_refused_without_reading_a_header() -> None:
    hub = HeaderHub({})

    with pytest.raises(SourceResolutionError) as caught:
        SourceResolver(hub).resolve_gguf(metadata("gemma3"), registry_resolver=rule_resolver())

    assert "gemma3" in caught.value.diagnostic.message
    assert hub.read == []


def test_a_header_that_contradicts_the_hub_is_believed() -> None:
    """The Hub summarises one file; each file's own header is the evidence."""
    headers = standard_headers({"F32": 2, "Q4_K": 6})
    headers["model-Q4_K_M.gguf"] = gguf_bytes("qwen35", {"F32": 2, "Q4_K": 6})

    resolved = SourceResolver(HeaderHub(headers)).resolve_gguf(
        metadata(), registry_resolver=rule_resolver()
    )

    assert resolved.gguf_file == "model-Q4_0.gguf"
    assert any("architecture qwen35" in reason for reason in resolved.reasons)


def test_without_a_verified_rule_nothing_is_imported() -> None:
    resolver = RegistryResolver(load_builtin_registry(), versions={})

    with pytest.raises(SourceResolutionError) as caught:
        SourceResolver(HeaderHub({})).resolve_gguf(metadata(), registry_resolver=resolver)
    assert "no registry gguf rule covers" in caught.value.diagnostic.message


def test_registry_data_can_extend_support_without_code() -> None:
    """A remote bundle or an override adds an architecture the moment it is verified."""
    extra = parse_document(
        textwrap.dedent(
            """\
            schema_version: 1
            gguf:
              - id: gguf-test
                schema_version: 1
                status: active
                priority: 200
                summary: test
                header_architectures: [gemma3]
                tensor_types: [F32, Q8_0]
                preferred_files: [Q8_0]
            """
        ),
        origin="test.yaml",
    )
    registry = load_registry_from_documents([extra])
    headers = {"model-Q8_0.gguf": gguf_bytes("gemma3", {"F32": 2, "Q8_0": 4})}

    resolved = SourceResolver(HeaderHub(headers)).resolve_gguf(
        metadata("gemma3"), registry_resolver=RegistryResolver(registry, versions={})
    )

    assert resolved.gguf_file == "model-Q8_0.gguf"
