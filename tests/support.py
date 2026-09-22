"""Helpers shared across the test suite.

Locating the tiny OpenVINO artifact lives here rather than in ``conftest.py`` so
that test modules can import it directly; ``conftest`` is loaded by pytest as a
plugin, not as an ordinary importable module.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_ARTIFACT = REPO_ROOT / "tests" / "fixtures" / ".cache" / "tinyllm"
VENDORED_ARTIFACT = REPO_ROOT / "tests" / "fixtures" / "artifacts" / "tinyllm"

ARTIFACT_MISSING = (
    "the tiny OpenVINO test artifact is not present; "
    "generate it with: python tools/make_test_artifact.py"
)


def find_broken_artifact() -> Path | None:
    """Return an unrepaired OpenVINO artifact directory, or None if unavailable.

    Order: the ``YONGA_TEST_BROKEN_ARTIFACT`` override, the generated cache, then
    a vendored copy if a downstream packager chose to commit one. The artifact is
    generated rather than committed so a clone does not carry binary weights.
    """
    override = os.environ.get("YONGA_TEST_BROKEN_ARTIFACT")
    candidates = [Path(override)] if override else []
    candidates += [GENERATED_ARTIFACT, VENDORED_ARTIFACT]
    for candidate in candidates:
        if (candidate / "openvino_tokenizer.xml").is_file():
            return candidate
    return None
