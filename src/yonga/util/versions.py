"""Tolerant version parsing and PEP 440 specifier matching.

OpenVINO reports versions such as
``2026.4.0-22959-99c81491cc3-releases/2026/4`` which are not valid PEP 440, so
every helper here falls back to the leading numeric release segment.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

_LEADING_RELEASE = re.compile(r"^\s*[vV]?(?P<release>\d+(?:\.\d+)*)")


def parse_version(raw: str | None) -> Version | None:
    """Parse a version string, tolerating vendor build suffixes.

    Returns ``None`` when ``raw`` is empty or contains no leading numeric
    release. ``"2026.4.0-22959-99c81491cc3-releases/2026/4"`` parses as
    ``2026.4.0``.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return Version(text)
    except InvalidVersion:
        pass
    match = _LEADING_RELEASE.match(text)
    if match is None:
        return None
    try:
        return Version(match.group("release"))
    except InvalidVersion:  # pragma: no cover - the regex only yields valid releases
        return None


def satisfies(raw_version: str | None, specifier: str | None) -> bool:
    """Return whether ``raw_version`` matches a PEP 440 ``specifier``.

    An empty or ``None`` specifier means "no constraint" and returns ``True``.
    An unknown or unparseable version returns ``False``, as does an invalid
    specifier (a registry authoring bug must not silently widen support).
    """
    if specifier is None or not specifier.strip():
        return True
    version = parse_version(raw_version)
    if version is None:
        return False
    try:
        specifier_set = SpecifierSet(specifier.strip())
    except InvalidSpecifier:
        return False
    return specifier_set.contains(version, prereleases=True)


def version_family(raw: str | None) -> tuple[int, int] | None:
    """Return the ``(major, minor)`` family of a version, or ``None``."""
    version = parse_version(raw)
    if version is None:
        return None
    release = version.release
    major = release[0] if release else 0
    minor = release[1] if len(release) > 1 else 0
    return (major, minor)


def families_aligned(versions: Mapping[str, str | None]) -> bool:
    """Return whether every known version in ``versions`` shares one family.

    Entries whose value is ``None`` (package not installed) are ignored, as are
    values that cannot be parsed at all; fewer than two comparable versions is
    trivially aligned.
    """
    families = {family for raw in versions.values() if (family := version_family(raw)) is not None}
    return len(families) <= 1


def installed_version(dist_name: str) -> str | None:
    """Return the installed distribution version, or ``None`` when absent."""
    try:
        return _distribution_version(dist_name)
    except PackageNotFoundError:
        return None
    except (ValueError, OSError):  # pragma: no cover - broken metadata on disk
        return None


def format_family(family: tuple[int, int] | None) -> str:
    """Render a version family for display, or ``unknown``."""
    if family is None:
        return "unknown"
    return f"{family[0]}.{family[1]}"
