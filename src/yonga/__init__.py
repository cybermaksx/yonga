"""yonga — a Linux-first local AI model manager for Intel AI PCs.

This module intentionally imports nothing heavy: it is imported by every other
module in the package (including the CLI entry point) and must stay cheap.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

APP_NAME: str = "yonga"
"""Human-facing product name, used in help text, panels and manifests."""

APP_SLUG: str = "yonga"
"""Filesystem/identifier form of the product name (XDG directory component)."""

ENV_PREFIX: str = "YONGA_"
"""Prefix for every environment variable the tool reads for configuration."""

FALLBACK_VERSION: str = "0.2.0"
"""Version reported when the distribution metadata is unavailable.

Kept in sync by hand with ``[project].version`` in ``pyproject.toml``; it is only
used when running from a source tree that was never installed.
"""

try:
    __version__: str = _distribution_version(APP_SLUG)
except PackageNotFoundError:  # pragma: no cover - only hit in an uninstalled tree
    __version__ = FALLBACK_VERSION

__all__ = [
    "APP_NAME",
    "APP_SLUG",
    "ENV_PREFIX",
    "FALLBACK_VERSION",
    "__version__",
]
