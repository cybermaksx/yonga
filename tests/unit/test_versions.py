"""Tests for tolerant version parsing and specifier matching."""

from __future__ import annotations

import pytest

from yonga.util.versions import (
    families_aligned,
    format_family,
    installed_version,
    parse_version,
    satisfies,
    version_family,
)

OPENVINO_BUILD = "2026.4.0-22959-99c81491cc3-releases/2026/4"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026.4.0", "2026.4.0"),
        (OPENVINO_BUILD, "2026.4.0"),
        ("2026.4.0.0", "2026.4.0.0"),
        ("v2.19.0", "2.19.0"),
        ("  2.3.0  ", "2.3.0"),
        ("2.14.0+cpu", "2.14.0+cpu"),
        ("5.5.4.dev0", "5.5.4.dev0"),
        ("2026", "2026"),
    ],
)
def test_parse_version_is_tolerant(raw: str, expected: str) -> None:
    parsed = parse_version(raw)

    assert parsed is not None
    assert str(parsed) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "not-a-version", "unknown"])
def test_parse_version_returns_none_for_unusable_input(raw: str | None) -> None:
    assert parse_version(raw) is None


def test_openvino_build_suffix_compares_correctly() -> None:
    parsed = parse_version(OPENVINO_BUILD)
    baseline = parse_version("2026.2")

    assert parsed is not None
    assert baseline is not None
    assert parsed > baseline


@pytest.mark.parametrize(
    ("version", "specifier", "expected"),
    [
        (OPENVINO_BUILD, ">=2026.2", True),
        (OPENVINO_BUILD, ">=2027", False),
        (OPENVINO_BUILD, ">=2026.2,<2027", True),
        ("2.19.0", ">=2.19", True),
        ("5.5.4", "==5.2.*", False),
        ("5.2.1", "==5.2.*", True),
        ("2026.4.0", None, True),
        ("2026.4.0", "", True),
        ("2026.4.0", "   ", True),
        (None, ">=1", False),
        (None, None, True),
        ("garbage", ">=1", False),
        ("1.0.0", "not a specifier", False),
    ],
)
def test_satisfies(version: str | None, specifier: str | None, expected: bool) -> None:
    assert satisfies(version, specifier) is expected


def test_satisfies_accepts_prereleases() -> None:
    assert satisfies("2026.5.0rc1", ">=2026.4") is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (OPENVINO_BUILD, (2026, 4)),
        ("2026.4.0.0", (2026, 4)),
        ("2.19.0", (2, 19)),
        ("2026", (2026, 0)),
        (None, None),
        ("nonsense", None),
    ],
)
def test_version_family(raw: str | None, expected: tuple[int, int] | None) -> None:
    assert version_family(raw) == expected


def test_families_aligned_for_the_openvino_triple() -> None:
    assert families_aligned(
        {
            "openvino": OPENVINO_BUILD,
            "openvino-genai": "2026.4.0.0",
            "openvino-tokenizers": "2026.4.0.0",
        }
    )


def test_families_misaligned_is_detected() -> None:
    assert not families_aligned(
        {
            "openvino": "2026.4.0",
            "openvino-genai": "2026.2.0",
        }
    )


def test_families_aligned_ignores_missing_and_unparseable_entries() -> None:
    assert families_aligned({"openvino": "2026.4.0", "openvino-genai": None})
    assert families_aligned({"openvino": "2026.4.0", "openvino-genai": "unknown"})
    assert families_aligned({})
    assert families_aligned({"only": None})


def test_installed_version_reads_metadata() -> None:
    assert installed_version("pydantic") is not None
    assert installed_version("definitely-not-installed-xyz") is None


def test_format_family() -> None:
    assert format_family(version_family(OPENVINO_BUILD)) == "2026.4"
    assert format_family(None) == "unknown"
