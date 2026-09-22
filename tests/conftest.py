"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support import ARTIFACT_MISSING, find_broken_artifact


@pytest.fixture(scope="session")
def broken_artifact() -> Path:
    """Session-scoped path to an unrepaired OpenVINO artifact; skips when absent."""
    found = find_broken_artifact()
    if found is None:
        pytest.skip(ARTIFACT_MISSING)
    return found
