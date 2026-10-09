"""Helpers shared across the test suite.

Locating the tiny OpenVINO artifact lives here rather than in ``conftest.py`` so
that test modules can import it directly; ``conftest`` is loaded by pytest as a
plugin, not as an ordinary importable module.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from yonga.adapters.subprocess import CompletedCommand

if TYPE_CHECKING:
    from yonga.application.jobs import JobContext

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


#: A small real model exported with an NPU profile. The tiny random Llama above
#: is rejected by the GenAI NPU pipeline (its 4-head x 4-dim toy shapes break the
#: NPU transformation), so proving NPU generation needs a genuine small LLM.
GENERATED_NPU_ARTIFACT = REPO_ROOT / "tests" / "fixtures" / ".cache" / "npu-llm"

NPU_ARTIFACT_MISSING = (
    "the NPU test artifact is not present; "
    "generate it with: python tools/make_test_artifact.py --npu"
)


def find_npu_artifact() -> Path | None:
    """Return an unrepaired NPU-profile LLM artifact, or None if unavailable.

    Order: the ``YONGA_TEST_NPU_ARTIFACT`` override, then the generated cache.
    """
    override = os.environ.get("YONGA_TEST_NPU_ARTIFACT")
    candidates = [Path(override)] if override else []
    candidates.append(GENERATED_NPU_ARTIFACT)
    for candidate in candidates:
        if (candidate / "openvino_tokenizer.xml").is_file():
            return candidate
    return None


class RefusingRunner:
    """A CommandRunner for unit tests: spawning any process is a test failure.

    Test wirings install it so that an adapter nobody faked -- such as the
    managed export environment builder -- fails loudly instead of quietly
    running ``pip`` against the network from a unit test.
    """

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float | None = None,
        ctx: JobContext | None = None,
    ) -> CompletedCommand:
        raise AssertionError(f"unit tests must not spawn processes: {list(argv)!r}")


class SimulatedPipRunner:
    """A CommandRunner that simulates building a managed export environment.

    ``python -m venv`` creates the interpreter file, ``pip install`` records the
    requirements it was given, and the verification script reports a version
    satisfying each one -- so the real environment manager can run end to end
    without a network or a real virtual environment.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.requirements: dict[str, str] = {}

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float | None = None,
        ctx: JobContext | None = None,
    ) -> CompletedCommand:
        command = [str(part) for part in argv]
        self.calls.append(command)
        if command[1:4] == ["-P", "-m", "venv"]:
            python = Path(command[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("#!/bin/sh\n")
        elif "pip" in command:
            for part in command[command.index("install") + 1 :]:
                match = re.match(r"^([A-Za-z0-9._-]+)(.*)$", part)
                if match and not part.startswith("-") and "://" not in part:
                    self.requirements[match.group(1)] = match.group(2)
        elif command[1:3] == ["-P", "-c"]:
            reported = {name: _satisfying(spec) for name, spec in self.requirements.items()}
            return CompletedCommand(
                argv=command,
                returncode=0,
                duration_seconds=0.0,
                stdout_lines=[json.dumps(reported)],
            )
        return CompletedCommand(argv=command, returncode=0, duration_seconds=0.0)


def _satisfying(specifier: str) -> str:
    """A version that meets ``==X.Y.*``, ``==X.Y.Z`` or an empty specifier."""
    if specifier.startswith("=="):
        return specifier[2:].replace(".*", ".0")
    return "1.0.0"


_ANSI = __import__("re").compile(r"\x1b\[[0-9;]*m")


def plain_text(text: str) -> str:
    """``text`` without ANSI styling and with whitespace collapsed.

    Typer forces coloured help output under GitHub Actions (it checks
    ``GITHUB_ACTIONS``), which splits ``--repo`` into separately styled ``-``
    and ``-repo`` runs; assertions about what help *says* must not depend on it.
    """
    return " ".join(_ANSI.sub("", text).split())
