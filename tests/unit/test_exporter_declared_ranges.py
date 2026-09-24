"""Transformers ranges read from optimum-intel itself, and the task GenAI can run.

On the reference host optimum-intel 2.2.0 refused LFM2 ("Maximum required is
5.4.0") and Gemma-3 ("Maximum required is 5.0") with Transformers 5.5.4
installed, and exported OLMo-2 as a stateless model GenAI could not load. The
first two are now read from the exporter's own export configs before the
download and handed to a managed environment; the third is fixed by exporting
every plain text-generation model with its KV cache.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from tests.unit.test_prepare_managed_env import FakeEnvironments
from tests.unit.test_prepare_planning import (
    MISMATCHED_EXPORT_VERSIONS,
    FakeExporter,
    FakeHub,
    build_service,
    make_snapshot,
    safetensors_metadata,
    settings_for,
)
from yonga.adapters.exporter_optimum import OptimumExporter
from yonga.adapters.subprocess import CompletedCommand
from yonga.application.jobs import JobContext
from yonga.application.prepare import _runtime_task
from yonga.config.settings import Settings
from yonga.domain.plans import DeclaredRequirement

REPO = "acme/gemma-like"


class DeclaringExporter(FakeExporter):
    """A fake exporter that also reports the range its export config declares."""

    def __init__(self, declared: str | None) -> None:
        super().__init__(versions=dict(MISMATCHED_EXPORT_VERSIONS))
        self.declared = declared
        self.asked: list[tuple[str, str | None]] = []

    def declared_requirements(
        self, model_type: str, task: str | None
    ) -> DeclaredRequirement | None:
        self.asked.append((model_type, task))
        if self.declared is None:
            return None
        return DeclaredRequirement(
            transformers=self.declared, source="optimum-intel 2.2.0 Gemma3TextOpenVINOConfig"
        )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return settings_for(tmp_path)


def hub_for(tmp_path: Path, model_type: str = "gemma3_text") -> FakeHub:
    metadata = safetensors_metadata(REPO, model_type=model_type, pipeline_tag="text-generation")
    metadata = metadata.model_copy(
        update={
            "config": {"model_type": model_type, "architectures": ["Gemma3ForCausalLM"]},
            "architectures": ["Gemma3ForCausalLM"],
        }
    )
    return FakeHub(repos={REPO: metadata}, snapshots={REPO: make_snapshot(tmp_path)})


def test_a_declared_ceiling_gets_a_managed_environment(settings: Settings, tmp_path: Path) -> None:
    exporter = DeclaringExporter(">=4.57,<=5.0")
    envs = FakeEnvironments(tmp_path / "envs")
    service = build_service(
        settings=settings, hub=hub_for(tmp_path), exporter=exporter, environments=envs
    )

    plan = service.plan(REPO, device="gpu")

    assert exporter.asked == [("gemma3_text", "text-generation-with-past")]
    assert plan.exporter.transformers == ">=4.57,<=5.0"
    assert plan.export_environment is not None
    assert plan.export_environment.packages["transformers"] == ">=4.57,<=5.0"
    assert any("declared by optimum-intel 2.2.0" in p for p in plan.exporter.problems)


def test_without_managed_environments_the_declared_range_blocks_at_preflight(
    settings: Settings, tmp_path: Path
) -> None:
    exporter = DeclaringExporter(">=4.57,<=5.0")
    service = build_service(settings=settings, hub=hub_for(tmp_path), exporter=exporter)

    plan = service.plan(REPO, device="gpu", managed_env=False)

    assert plan.exporter_requirements_block, "refused before the download, not after the export"


def test_a_range_the_installed_version_meets_changes_nothing(
    settings: Settings, tmp_path: Path
) -> None:
    exporter = DeclaringExporter(">=4.57")
    service = build_service(settings=settings, hub=hub_for(tmp_path), exporter=exporter)

    plan = service.plan(REPO, device="gpu")

    assert plan.exporter.satisfied and plan.export_environment is None


def test_the_registry_pin_wins_and_the_exporter_is_not_asked(
    settings: Settings, tmp_path: Path
) -> None:
    """Qwen3.5's registry rule pins transformers; the exporter range is not consulted."""
    exporter = DeclaringExporter(">=1,<=2")
    metadata = safetensors_metadata()
    hub = FakeHub(
        repos={metadata.identity.repo_id: metadata},
        snapshots={metadata.identity.repo_id: make_snapshot(tmp_path)},
    )
    service = build_service(settings=settings, hub=hub, exporter=exporter)

    plan = service.plan(metadata.identity.repo_id, device="gpu", managed_env=False)

    assert exporter.asked == []
    assert plan.exporter.transformers == "==5.2.*"


@pytest.mark.parametrize(
    ("hub_task", "exported"),
    [
        ("text-generation", "text-generation-with-past"),
        ("text-generation-with-past", "text-generation-with-past"),
        ("image-text-to-text", "image-text-to-text"),
        (None, None),
    ],
)
def test_plain_text_generation_is_exported_with_its_kv_cache(
    hub_task: str | None, exported: str | None
) -> None:
    """GenAI's LLMPipeline needs the stateful export ("beam_idx was not found" otherwise)."""
    assert _runtime_task(hub_task) == exported


# --- the adapter ----------------------------------------------------------------------


class BoundsRunner:
    def __init__(self, payload: dict[str, str | None]) -> None:
        self.payload = payload
        self.calls = 0

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float | None = None,
        ctx: JobContext | None = None,
    ) -> CompletedCommand:
        self.calls += 1
        return CompletedCommand(
            argv=list(argv),
            returncode=0,
            duration_seconds=0.1,
            stdout_lines=["some import warning", json.dumps(self.payload)],
        )


def test_the_adapter_turns_declared_bounds_into_a_specifier_and_caches_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "yonga.adapters.exporter_optimum.installed_version",
        lambda name: "2.2.0" if name == "optimum-intel" else None,
    )
    runner = BoundsRunner({"min": "4.57", "max": "5.4.0", "config": "LFM2OpenVINOConfig"})
    exporter = OptimumExporter(runner, cache_dir=tmp_path)

    first = exporter.declared_requirements("lfm2", "text-generation-with-past")
    second = exporter.declared_requirements("lfm2", "text-generation-with-past")

    assert first is not None and first.transformers == ">=4.57,<=5.4.0"
    assert "LFM2OpenVINOConfig" in first.source
    assert second == first
    assert runner.calls == 1, "the optimum import is paid once per optimum-intel version"


def test_the_adapter_says_nothing_without_optimum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("yonga.adapters.exporter_optimum.installed_version", lambda name: None)
    runner = BoundsRunner({})

    assert OptimumExporter(runner, cache_dir=tmp_path).declared_requirements("lfm2", None) is None
    assert runner.calls == 0
