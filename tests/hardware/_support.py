"""Shared helpers for the hardware tests.

Every test here runs against the real OpenVINO runtime and a real device. They
are deselected by default (``pyproject.toml`` addopts) and selected explicitly::

    pytest -m intel_gpu
    pytest -m intel_npu

A test skips, with the command that fixes it, when its device or artifact is
missing -- it never passes vacuously.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.support import (
    ARTIFACT_MISSING,
    NPU_ARTIFACT_MISSING,
    find_broken_artifact,
    find_npu_artifact,
)
from yonga.config.settings import Settings


def require_device(name: str) -> None:
    """Skip unless OpenVINO really exposes ``name`` on this host."""
    from yonga.adapters.openvino_probe import OpenVINOProbe, openvino_available

    available, error = openvino_available()
    if not available:
        pytest.skip(f"openvino is not importable: {error}")
    if name not in OpenVINOProbe().available_devices():
        pytest.skip(f"OpenVINO does not expose {name} on this host")


def repaired_copy(source: Path | None, destination: Path, *, missing: str) -> Path:
    """Copy an unrepaired export and apply yonga's tokenizer repair to the copy."""
    from yonga.repairs.base import RepairContext, get_repair

    if source is None:
        pytest.skip(missing)
    shutil.copytree(source, destination)
    context = RepairContext(artifact_dir=destination, owned_roots=(destination.parent,))
    repair = get_repair("tokenizer_string_precision")
    match = repair.detect(destination, context)
    if match is not None:
        repair.apply(match, destination, context)
    return destination


def tiny_artifact(tmp_path: Path) -> Path:
    """The tiny random Llama, repaired: fast, and accepted by CPU and GPU."""
    return repaired_copy(find_broken_artifact(), tmp_path / "tiny", missing=ARTIFACT_MISSING)


def npu_artifact(tmp_path: Path) -> Path:
    """A small real LLM exported with the NPU profile, repaired."""
    return repaired_copy(find_npu_artifact(), tmp_path / "npu-llm", missing=NPU_ARTIFACT_MISSING)


def isolated_settings(tmp_path: Path) -> Settings:
    """Settings whose managed directories live under ``tmp_path``."""
    return Settings(
        cache_dir=tmp_path / "cache",
        data_dir=tmp_path / "data",
        config_dir=tmp_path / "config",
    )
