"""Shared pytest fixtures."""

from __future__ import annotations

from importlib.util import find_spec
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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``openvino``-marked tests, with the reason, when the runtime is not installed.

    Those tests exercise the real OpenVINO stack and mean nothing without it. On
    a base install -- ``pip install yonga`` with no extras, which CI checks too --
    they report as skipped instead of failing on a missing runtime.
    """
    del config
    if find_spec("openvino") is not None:
        return
    skip = pytest.mark.skip(reason="needs the OpenVINO runtime: pip install 'yonga[openvino]'")
    for item in items:
        if item.get_closest_marker("openvino") is not None:
            item.add_marker(skip)
