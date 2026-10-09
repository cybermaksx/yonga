"""Real VLM on the GPU: text chat keeps its history, and serve passes an image.

A VLM export has no single ``openvino_model.xml``; GenAI drives it through
``VLMPipeline``, whose chat state yonga replays turn by turn. The artifact is a
few gigabytes, so this is opt-in: point ``YONGA_TEST_VLM_ARTIFACT`` at a
prepared VLM artifact directory, for example one ``yonga prepare
Qwen/Qwen2.5-VL-3B-Instruct --device gpu`` produced.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.hardware._support import isolated_settings, require_device

pytestmark = [pytest.mark.openvino, pytest.mark.intel_gpu, pytest.mark.large_model]

VLM_MISSING = "set YONGA_TEST_VLM_ARTIFACT to a prepared VLM artifact directory"


def _vlm_artifact() -> Path:
    override = os.environ.get("YONGA_TEST_VLM_ARTIFACT")
    if not override or not (Path(override) / "openvino_language_model.xml").is_file():
        pytest.skip(VLM_MISSING)
    return Path(override)


def test_a_vlm_answers_a_follow_up_from_its_history(tmp_path: Path) -> None:
    require_device("GPU")
    from yonga.adapters.runtime_genai import ChatHistory, open_runtime
    from yonga.domain.benchmarks import GenerationSettings
    from yonga.domain.models import PipelineKind

    runtime = open_runtime(
        _vlm_artifact(),
        device="GPU",
        pipeline=PipelineKind.VLM,
        settings=isolated_settings(tmp_path),
    )
    config = GenerationSettings(max_new_tokens=24, do_sample=False)
    try:
        history = ChatHistory()
        history.add_user("My name is Deniz. Reply with just: noted.")
        first = runtime.generate(history, config)
        history.add_assistant(first.text)
        history.add_user("What is my name? Answer in one word.")
        second = runtime.generate(history, config)
    finally:
        runtime.close()

    assert first.output_tokens is not None and first.output_tokens > 0
    assert "deniz" in second.text.lower(), second.text


def test_serve_passes_an_image_to_the_vlm(tmp_path: Path) -> None:
    """A solid red PNG as a data URL, through the HTTP layer, reaches the pipeline."""
    require_device("GPU")
    pytest.importorskip("fastapi")
    pytest.importorskip("PIL")
    import base64
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    from yonga.api.app import create_app
    from yonga.application.chat import ChatOptions
    from yonga.application.serve import ModelServer
    from yonga.application.wiring import Wiring

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    service = Wiring(isolated_settings(tmp_path)).chat_service()
    session = service.open_artifact(_vlm_artifact(), options=ChatOptions(device="gpu"))
    server = ModelServer(session, name="vlm")
    server.load()
    try:
        response = TestClient(create_app(server)).post(
            "/v1/chat/completions",
            json={
                "model": "vlm",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": url}},
                            {"type": "text", "text": "What colour is this image? One word."},
                        ],
                    }
                ],
                "max_tokens": 16,
                "temperature": 0,
            },
        )
    finally:
        server.close()

    assert response.status_code == 200, response.text
    assert "red" in response.json()["choices"][0]["message"]["content"].lower()
