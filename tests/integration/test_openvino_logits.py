"""The logits adapter against a real IR: normalised distributions, deterministic scores.

Runs on the tiny generated artifact the hardware tests use, on the CPU, so it
needs OpenVINO but no accelerator and finishes in seconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")

import numpy as np

from tests.hardware._support import tiny_artifact

pytestmark = pytest.mark.openvino


def test_log_probs_are_normalised_and_deterministic(tmp_path: Path) -> None:
    from yonga.adapters.openvino_logits import LogitsModel

    model = LogitsModel(tiny_artifact(tmp_path), device="CPU")
    try:
        ids = model.tokenize("The quick brown fox jumps over the lazy dog.")
        first = model.log_probs(ids)
        again = model.log_probs(ids)
    finally:
        model.close()

    assert ids.ndim == 1 and len(ids) > 3
    assert first.shape[0] == len(ids)
    np.testing.assert_allclose(np.exp(first.astype(np.float64)).sum(axis=-1), 1.0, rtol=1e-4)
    np.testing.assert_array_equal(first, again)  # the state is reset between passages
    assert model.properties == {"INFERENCE_PRECISION_HINT": "f32"}
