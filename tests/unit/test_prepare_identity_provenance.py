"""The preparation plan must say when its architecture came from a weight header.

``docs/CLI_SPEC.md`` section 1 requires expensive decisions to be shown before
execution, and everything expensive about a plan — the pipeline it exports, the
exporter version it pins, the device it recommends — is read off the matched
architecture rule. When that rule was reached through an identity alias rather
than through a ``config.json``, the plan preview looks exactly like one derived
from a declared ``model_type``, which overstates what was established.

The fakes and fixtures come from :mod:`tests.unit.test_prepare_planning`, the
module that owns the preparation adapters.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console, RenderableType

from tests.unit.test_prepare_planning import (
    FakeHub,
    build_service,
    make_snapshot,
    run_prepare,
    settings_for,
)
from yonga.config.settings import Settings
from yonga.domain.models import (
    FormatObservation,
    ModelFormat,
    ModelIdentity,
    ModelMetadata,
    PipelineKind,
    RepoFile,
)
from yonga.ui.tables import preparation_plan

#: A repository that publishes convertible weights but no ``config.json``.
HEADER_ONLY_REPO = "acme/qwen35-uncensored-mixed"
HEADER_ONLY_SHA = "e" * 40

#: The identifier the Hub parses out of the GGUF file itself.
GGUF_HEADER: dict[str, object] = {"architecture": "qwen35", "context_length": 262_144}

#: The sentence the registry produces for an alias match, quoted so a reworded
#: explanation cannot silently stop reaching the plan.
DISCLOSURE = (
    "Architecture identified from GGUF metadata (alias 'qwen35'); "
    "this repository has no config.json."
)


def header_only_metadata() -> ModelMetadata:
    """Metadata for a repository whose only machine-readable identity is a GGUF header.

    The safetensors shards are what an export would actually read, so the
    repository is convertible as it stands and no derivative is involved; the
    GGUF copy published beside them is where the architecture identifier lives.
    """
    shards = [
        RepoFile(path=f"model-0000{index}-of-00002.safetensors", size=size, lfs=True)
        for index, size in enumerate((5_278_251_472, 5_358_071_808), start=1)
    ]
    quantized = RepoFile(path="qwen35-uncensored-Q4_K_M.gguf", size=5_627_044_224, lfs=True)
    return ModelMetadata(
        identity=ModelIdentity(repo_id=HEADER_ONLY_REPO, revision=HEADER_ONLY_SHA),
        resolved_revision=HEADER_ONLY_SHA,
        license="apache-2.0",
        tags=["gguf", "qwen3.5", "qwen"],
        files=[*shards, quantized],
        formats=[
            FormatObservation(
                format=ModelFormat.SAFETENSORS,
                files=[shard.path for shard in shards],
                total_bytes=sum(shard.size or 0 for shard in shards),
                complete=True,
            ),
            FormatObservation(
                format=ModelFormat.GGUF,
                files=[quantized.path],
                total_bytes=quantized.size or 0,
                complete=True,
            ),
        ],
        gguf_metadata=dict(GGUF_HEADER),
    )


def render(renderable: RenderableType, *, width: int = 120) -> str:
    """Render a renderable the way a non-TTY terminal would."""
    buffer = io.StringIO()
    Console(file=buffer, width=width, force_terminal=False, no_color=True).print(renderable)
    return buffer.getvalue()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings confined to the test's temporary directory."""
    return settings_for(tmp_path)


@pytest.fixture
def hub(tmp_path: Path) -> FakeHub:
    """A hub serving the header-only repository."""
    return FakeHub(
        repos={HEADER_ONLY_REPO: header_only_metadata()},
        snapshots={HEADER_ONLY_REPO: make_snapshot(tmp_path, "snapshot-header-only")},
    )


def test_plan_carries_how_the_architecture_was_identified(settings: Settings, hub: FakeHub) -> None:
    """The alias match reaches the plan, not just the resolver's discarded explanation."""
    service = build_service(settings=settings, hub=hub)

    plan = service.plan(HEADER_ONLY_REPO, device="gpu")

    # The whole plan descends from the alias match. ``architecture`` carries the
    # identity the manifest will record — never the rule id, which is its own field.
    assert plan.architecture == "qwen3_5"
    assert plan.architecture_rule_id == "arch-qwen3_5"
    assert plan.pipeline is PipelineKind.VLM
    assert plan.exporter.transformers == "==5.2.*"
    assert DISCLOSURE in plan.known_limitations


def test_plan_panel_shows_how_the_architecture_was_identified(
    settings: Settings, hub: FakeHub
) -> None:
    """The confirmation screen prints it; a plan is reviewed before it is approved."""
    service = build_service(settings=settings, hub=hub)

    text = render(preparation_plan(service.plan(HEADER_ONLY_REPO, device="gpu")))

    assert "Architecture identified from GGUF metadata" in text
    assert "no config.json" in text


def test_the_plan_panel_names_the_model_not_the_rule(settings: Settings, hub: FakeHub) -> None:
    """The pre-flight panel used to print ``arch-qwen3_5`` under "Architecture".

    That is the id of a registry rule covering a family, not what this model is,
    and the panel is the screen a user approves the job from.
    """
    service = build_service(settings=settings, hub=hub)

    text = render(preparation_plan(service.plan(HEADER_ONLY_REPO, device="gpu")))

    architecture_line = next(line for line in text.splitlines() if "Architecture " in line)
    assert "qwen3_5" in architecture_line
    assert "rule arch-qwen3_5" in architecture_line
    assert not architecture_line.split("Architecture")[1].lstrip().startswith("arch-")


def test_manifest_records_how_the_architecture_was_identified(
    settings: Settings, hub: FakeHub
) -> None:
    """The artifact keeps the identification chain, not only its conclusion."""
    service = build_service(settings=settings, hub=hub)
    plan = service.plan(HEADER_ONLY_REPO, device="gpu")

    manifest = run_prepare(service, plan, settings).manifest

    # A repository with no config.json declares no identity of its own, so the
    # rule that identified it supplies the canonical one — and the rule id is
    # kept beside it rather than in place of it.
    assert manifest.architecture == "qwen3_5"
    assert manifest.architecture_rule_id == "arch-qwen3_5"
    assert DISCLOSURE in manifest.notes


def test_a_declared_architecture_adds_no_disclosure(settings: Settings, tmp_path: Path) -> None:
    """A repository that declares its own ``model_type`` earns no caveat."""
    declared = header_only_metadata().model_copy(
        update={
            "model_type": "qwen3_5",
            "gguf_metadata": {},
            "config": {"model_type": "qwen3_5"},
        }
    )
    hub = FakeHub(
        repos={HEADER_ONLY_REPO: declared},
        snapshots={HEADER_ONLY_REPO: make_snapshot(tmp_path, "snapshot-declared")},
    )
    service = build_service(settings=settings, hub=hub)

    plan = service.plan(HEADER_ONLY_REPO, device="gpu")

    assert plan.architecture == "qwen3_5"
    assert plan.architecture_rule_id == "arch-qwen3_5"
    assert not any("identified from" in entry for entry in plan.known_limitations)
