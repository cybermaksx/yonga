"""Tests for the SHA-256 helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from yonga.util.hashing import (
    hash_directory,
    sha256_bytes,
    sha256_file,
    sha256_text,
    short_hash,
)


def test_sha256_bytes_matches_hashlib() -> None:
    data = b"openvino ir"

    assert sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_sha256_text_matches_hashlib_utf8() -> None:
    text = "tokenizer precision STRiNG"

    assert sha256_text(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_sha256_text_handles_non_ascii() -> None:
    text = "yonga — çekirdek"

    assert sha256_text(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_sha256_file_matches_hashlib_for_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")

    assert sha256_file(path) == hashlib.sha256(b"").hexdigest()


@pytest.mark.parametrize("chunk_size", [1, 7, 4096, 1 << 20])
def test_sha256_file_streams_identically_for_every_chunk_size(
    tmp_path: Path, chunk_size: int
) -> None:
    payload = os.urandom(65_536) + b"tail"
    path = tmp_path / "weights.bin"
    path.write_bytes(payload)

    assert sha256_file(path, chunk_size=chunk_size) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_rejects_a_non_positive_chunk_size(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(b"x")

    with pytest.raises(ValueError, match="chunk_size"):
        sha256_file(path, chunk_size=0)


def test_sha256_file_propagates_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "nope.bin")


def build_tree(root: Path) -> None:
    (root / "openvino_model.xml").write_bytes(b"<net/>")
    (root / "openvino_model.bin").write_bytes(b"\x00\x01\x02")
    nested = root / "sub"
    nested.mkdir()
    (nested / "config.json").write_bytes(b"{}")
    ignored = root / ".yonga"
    ignored.mkdir()
    (ignored / "manifest.json").write_bytes(b"{}")


def test_hash_directory_is_sorted_and_correct(tmp_path: Path) -> None:
    build_tree(tmp_path)

    entries = hash_directory(tmp_path)
    relatives = [item[0] for item in entries]

    assert relatives == sorted(relatives)
    assert relatives == [
        ".yonga/manifest.json",
        "openvino_model.bin",
        "openvino_model.xml",
        "sub/config.json",
    ]
    by_path = {item[0]: item for item in entries}
    assert by_path["openvino_model.bin"][1] == 3
    assert by_path["openvino_model.bin"][2] == hashlib.sha256(b"\x00\x01\x02").hexdigest()
    assert by_path["sub/config.json"][2] == hashlib.sha256(b"{}").hexdigest()


def test_hash_directory_excludes_patterns(tmp_path: Path) -> None:
    build_tree(tmp_path)

    entries = hash_directory(tmp_path, exclude=[".yonga", "*.bin"])
    relatives = [item[0] for item in entries]

    assert relatives == ["openvino_model.xml", "sub/config.json"]


def test_hash_directory_honours_relative_to(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    build_tree(root)

    entries = hash_directory(root, relative_to=tmp_path)
    relatives = [item[0] for item in entries]

    assert all(item.startswith("artifact/") for item in relatives)


def test_hash_directory_skips_symlinks(tmp_path: Path) -> None:
    real = tmp_path / "real.bin"
    real.write_bytes(b"abc")
    (tmp_path / "link.bin").symlink_to(real)

    entries = hash_directory(tmp_path)

    assert [item[0] for item in entries] == ["real.bin"]


def test_hash_directory_of_an_empty_tree(tmp_path: Path) -> None:
    assert hash_directory(tmp_path) == []


def test_short_hash_is_a_prefix_of_the_full_digest() -> None:
    value = "org/model@abcdef"
    full = sha256_text(value)

    assert short_hash(value) == full[:12]
    assert short_hash(value, 8) == full[:8]
    assert short_hash(value, 64) == full


def test_short_hash_is_stable() -> None:
    assert short_hash("stable") == short_hash("stable")
    assert short_hash("a") != short_hash("b")


@pytest.mark.parametrize("length", [0, -1, 65])
def test_short_hash_rejects_out_of_range_lengths(length: int) -> None:
    with pytest.raises(ValueError, match="length"):
        short_hash("x", length)
