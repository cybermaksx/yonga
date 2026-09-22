"""SHA-256 helpers used for artifact inventories, repairs and cache keys.

Every file hash is streamed so that multi-gigabyte weight shards never have to
be held in memory.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection
from fnmatch import fnmatch
from pathlib import Path

DEFAULT_CHUNK_SIZE: int = 1 << 20
"""Streaming read size (1 MiB) used by :func:`sha256_file`."""

SHA256_HEX_LENGTH: int = 64


def sha256_file(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return the hex SHA-256 of a file, read in ``chunk_size`` blocks."""
    if chunk_size <= 0:
        message = f"chunk_size must be positive, got {chunk_size}"
        raise ValueError(message)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 of a bytes object."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 of a string, encoded as UTF-8."""
    return sha256_bytes(text.encode("utf-8"))


def _is_excluded(relative: str, name: str, patterns: Collection[str]) -> bool:
    """Return whether a relative path or any of its components is excluded."""
    if not patterns:
        return False
    parts = relative.split("/")
    for pattern in patterns:
        if fnmatch(relative, pattern) or fnmatch(name, pattern):
            return True
        if any(fnmatch(part, pattern) for part in parts):
            return True
    return False


def hash_directory(
    root: Path,
    *,
    relative_to: Path | None = None,
    exclude: Collection[str] = (),
) -> list[tuple[str, int, str]]:
    """Hash every regular file under ``root``.

    Returns ``(relative_path, size_bytes, sha256)`` triples sorted by relative
    path. Paths are reported with ``/`` separators relative to ``relative_to``
    (``root`` by default). ``exclude`` holds glob patterns matched against the
    relative path, the file name and every individual path component. Symbolic
    links are skipped so that a hash never escapes the tree.
    """
    base = relative_to if relative_to is not None else root
    results: list[tuple[str, int, str]] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        try:
            relative = candidate.relative_to(base).as_posix()
        except ValueError:
            relative = candidate.as_posix()
        if _is_excluded(relative, candidate.name, exclude):
            continue
        size = candidate.stat().st_size
        results.append((relative, size, sha256_file(candidate)))
    results.sort(key=lambda item: item[0])
    return results


def short_hash(value: str, length: int = 12) -> str:
    """Return the first ``length`` hex characters of the SHA-256 of ``value``."""
    if not 1 <= length <= SHA256_HEX_LENGTH:
        message = f"length must be between 1 and {SHA256_HEX_LENGTH}, got {length}"
        raise ValueError(message)
    return sha256_text(value)[:length]
