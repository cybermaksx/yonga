"""Size-banded conversion profiles as ``PrepareService`` plans and records them.

Every test here adds its own band, so the threshold under test does not move
with the built-in data: ``cpu-int4-asym-g128-small``, a CPU profile for models
below 2B parameters that outranks both the CPU default and the built-in
``cpu-int4-asym-g128-r08``. The Qwen3.5 fixture's architecture rule names
``cpu-int4-asym-g128`` for CPU (VLM families keep ratio 1.0), and a named
profile is preferred before any scoring happens, so :func:`banded_resolver`
drops that CPU entry; otherwise the band could never decide anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.test_gguf_import import REPO as GGUF_IMPORT_REPO
from tests.unit.test_gguf_import import FakeImporter, GGUFHub, headers
from tests.unit.test_prepare_planning import (
    MODEL_XML,
    OPENVINO_REPO,
    REGISTRY_VERSIONS,
    SAFETENSORS_REPO,
    FakeExporter,
    FakeHub,
    build_service,
    make_snapshot,
    openvino_metadata,
    run_prepare,
    safetensors_metadata,
    settings_for,
    write_tokenizer_pair,
)
from yonga.adapters.filesystem import INCOMPLETE_MARKER_NAME
from yonga.config.settings import Settings
from yonga.domain.artifacts import ArtifactManifest, manifest_path
from yonga.domain.compatibility import EvidenceLevel, ProfileSelection
from yonga.domain.models import ModelMetadata
from yonga.registry.loader import (
    load_builtin_documents,
    load_registry_from_documents,
    parse_document,
)
from yonga.registry.resolver import RegistryResolver
from yonga.registry.schema import Registry

DEFAULT_CPU = "cpu-int4-asym-g128"
BANDED_CPU = "cpu-int4-asym-g128-small"

BANDED_PROFILE_YAML = f"""
schema_version: 2
profiles:
  - id: {BANDED_CPU}
    schema_version: 2
    status: active
    priority: 120
    summary: CPU INT4 with a fifth of the layers kept at INT8, for small models
    device: cpu
    format: openvino_ir
    quantization:
      weight_format: int4
      symmetric: false
      ratio: 0.8
      group_size: 128
    constraints:
      openvino: ">=2026.2,<2027"
    evidence: tested
    applies_to:
      parameter_count:
        below: 2B
    sources:
      - kind: tested
        note: Test fixture.
"""

SMALL_COUNT = 800_000_000
LARGE_COUNT = 9_000_000_000


def banded_resolver(*, cpu_named: str | None = None) -> RegistryResolver:
    """The built-in registry plus :data:`BANDED_PROFILE_YAML`.

    Every architecture rule's CPU entry is dropped, or, with ``cpu_named``,
    replaced so that the rule names that profile for CPU.
    """
    documents = [
        *load_builtin_documents(),
        parse_document(BANDED_PROFILE_YAML, origin="<test-banded>"),
    ]
    base = load_registry_from_documents(documents)
    fields = {name: getattr(base, name) for name in Registry.model_fields}

    def cpu_entry(profiles: dict[str, str]) -> dict[str, str]:
        kept = {device: pid for device, pid in profiles.items() if device != "CPU"}
        if cpu_named is not None:
            kept["CPU"] = cpu_named
        return kept

    fields["architectures"] = [
        rule.model_copy(update={"profiles": cpu_entry(rule.profiles)})
        for rule in base.architectures
    ]
    registry = Registry.model_validate(fields)
    return RegistryResolver(
        registry,
        versions=REGISTRY_VERSIONS,
        host_tags=("intel.cpu", "intel.gpu", "intel.npu", "intel.npu.lunar_lake"),
    )


def sized(meta: ModelMetadata, count: int) -> ModelMetadata:
    """``meta`` with a Hub parameter count stored entirely as BF16."""
    return meta.model_copy(update={"parameter_count": count, "parameter_dtypes": {"BF16": count}})


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings confined to the test's temporary directory."""
    return settings_for(tmp_path)


@pytest.fixture
def hub(tmp_path: Path) -> FakeHub:
    """A small safetensors Qwen3.5 and a published OpenVINO IR of the same size."""
    return FakeHub(
        repos={
            SAFETENSORS_REPO: sized(safetensors_metadata(), SMALL_COUNT),
            OPENVINO_REPO: sized(openvino_metadata(), SMALL_COUNT),
        },
        snapshots={
            SAFETENSORS_REPO: make_snapshot(tmp_path, "snapshot-safetensors"),
            OPENVINO_REPO: make_snapshot(tmp_path, "snapshot-openvino"),
        },
    )


# --- choosing the profile ------------------------------------------------------------


def test_a_small_safetensors_model_on_cpu_gets_the_banded_profile(
    settings: Settings, hub: FakeHub
) -> None:
    service = build_service(settings=settings, hub=hub, resolver=banded_resolver())

    plan = service.plan(SAFETENSORS_REPO, device="cpu")

    assert plan.profile.id == BANDED_CPU
    assert plan.artifact_dir.name.endswith(f"--{BANDED_CPU}")
    selection = plan.profile_selection
    assert selection is not None
    assert selection.basis == "size_band"
    assert selection.applies is True
    assert selection.profile_id == BANDED_CPU
    assert selection.parameter_count == SMALL_COUNT
    assert selection.band == "below 2B"
    assert selection.band_evidence == EvidenceLevel.TESTED
    assert "800M parameters" in selection.sentence
    assert BANDED_CPU in selection.sentence


def test_a_large_model_keeps_the_default_and_says_which_band_it_missed(
    settings: Settings, tmp_path: Path
) -> None:
    hub = FakeHub(
        repos={SAFETENSORS_REPO: sized(safetensors_metadata(), LARGE_COUNT)},
        snapshots={SAFETENSORS_REPO: make_snapshot(tmp_path)},
    )
    service = build_service(settings=settings, hub=hub, resolver=banded_resolver())

    plan = service.plan(SAFETENSORS_REPO, device="cpu")

    assert plan.profile.id == DEFAULT_CPU
    assert plan.artifact_dir.name.endswith(f"--{DEFAULT_CPU}")
    selection = plan.profile_selection
    assert selection is not None
    assert selection.basis == "device_default"
    assert selection.applies is True
    assert any(BANDED_CPU in skipped for skipped in selection.skipped)
    assert BANDED_CPU in selection.sentence


@pytest.mark.parametrize(
    ("count", "expected", "basis"),
    [
        (2_000_000_000 - 1, BANDED_CPU, "size_band"),
        # Outside the test band, inside the built-in one below 5B.
        (2_000_000_000, "cpu-int4-asym-g128-r08", "size_band"),
    ],
)
def test_the_band_is_half_open_at_its_upper_bound(
    settings: Settings, tmp_path: Path, count: int, expected: str, basis: str
) -> None:
    """``below: 2B`` means below: a model of exactly 2B belongs to the next band."""
    hub = FakeHub(
        repos={SAFETENSORS_REPO: sized(safetensors_metadata(), count)},
        snapshots={SAFETENSORS_REPO: make_snapshot(tmp_path)},
    )
    service = build_service(settings=settings, hub=hub, resolver=banded_resolver())

    plan = service.plan(SAFETENSORS_REPO, device="cpu")

    assert plan.profile.id == expected
    assert plan.profile_selection is not None
    assert plan.profile_selection.basis == basis


def test_packed_integer_weights_keep_the_default_and_say_why(
    settings: Settings, tmp_path: Path
) -> None:
    """A GPTQ-style checkpoint under-reports its size, so the small band must not catch it."""
    packed = safetensors_metadata().model_copy(
        update={
            "parameter_count": SMALL_COUNT,
            "parameter_dtypes": {"I32": 600_000_000, "F16": 200_000_000},
        }
    )
    hub = FakeHub(
        repos={SAFETENSORS_REPO: packed},
        snapshots={SAFETENSORS_REPO: make_snapshot(tmp_path)},
    )
    service = build_service(settings=settings, hub=hub, resolver=banded_resolver())

    plan = service.plan(SAFETENSORS_REPO, device="cpu")

    assert plan.profile.id == DEFAULT_CPU
    selection = plan.profile_selection
    assert selection is not None
    assert selection.basis == "device_default"
    assert selection.applies is True
    assert selection.parameter_count is None
    assert selection.parameter_count_basis is not None
    assert "packed integer weights" in selection.parameter_count_basis
    assert any("unknown size" in skipped for skipped in selection.skipped)


def test_an_architecture_rule_naming_the_banded_profile_is_honoured_inside_its_band(
    settings: Settings, hub: FakeHub
) -> None:
    service = build_service(
        settings=settings, hub=hub, resolver=banded_resolver(cpu_named=BANDED_CPU)
    )

    plan = service.plan(SAFETENSORS_REPO, device="cpu")

    assert plan.profile.id == BANDED_CPU
    selection = plan.profile_selection
    assert selection is not None
    assert selection.basis == "named_by_architecture_rule"
    assert selection.rule_id == "arch-qwen3_5"


def test_a_named_profile_outside_its_band_falls_through_to_scoring(
    settings: Settings, tmp_path: Path
) -> None:
    """A rule naming a banded profile is a preference, not an override of the band."""
    hub = FakeHub(
        repos={SAFETENSORS_REPO: sized(safetensors_metadata(), LARGE_COUNT)},
        snapshots={SAFETENSORS_REPO: make_snapshot(tmp_path)},
    )
    service = build_service(
        settings=settings, hub=hub, resolver=banded_resolver(cpu_named=BANDED_CPU)
    )

    plan = service.plan(SAFETENSORS_REPO, device="cpu")

    assert plan.profile.id == DEFAULT_CPU
    assert plan.artifact_dir.name.endswith(f"--{DEFAULT_CPU}")
    selection = plan.profile_selection
    assert selection is not None
    assert selection.basis == "device_default"
    assert selection.rule_id is None


# --- the artifact the old default left behind ----------------------------------------


def test_a_ready_artifact_under_the_old_default_is_named_in_the_plan(
    settings: Settings, hub: FakeHub
) -> None:
    """A changed default is a new artifact; the user hears about the old one first."""
    service = build_service(
        settings=settings, hub=hub, resolver=banded_resolver(), exporter=FakeExporter()
    )
    old = run_prepare(
        service, service.plan(SAFETENSORS_REPO, device="cpu", profile_id=DEFAULT_CPU), settings
    )
    assert old.artifact_dir.name.endswith(f"--{DEFAULT_CPU}")

    plan = service.plan(SAFETENSORS_REPO, device="cpu")

    assert plan.profile.id == BANDED_CPU
    assert plan.reuse_existing_artifact is None
    notices = [note for note in plan.known_limitations if "already prepared as" in note]
    assert len(notices) == 1
    assert f"already prepared as {DEFAULT_CPU}" in notices[0]
    assert str(old.artifact_dir) in notices[0]
    assert f"--profile {DEFAULT_CPU}" in notices[0]


def test_an_incomplete_artifact_under_the_old_default_is_not_named(
    settings: Settings, hub: FakeHub
) -> None:
    """Only a usable artifact is worth a ``--profile`` hint; a half-written one is not."""
    service = build_service(settings=settings, hub=hub, resolver=banded_resolver())
    old = run_prepare(
        service, service.plan(SAFETENSORS_REPO, device="cpu", profile_id=DEFAULT_CPU), settings
    )
    (old.artifact_dir / INCOMPLETE_MARKER_NAME).write_text("", encoding="utf-8")

    plan = service.plan(SAFETENSORS_REPO, device="cpu")

    assert plan.profile.id == BANDED_CPU
    assert not any("already prepared as" in note for note in plan.known_limitations)


def test_a_ready_artifact_for_another_device_is_not_named(settings: Settings, hub: FakeHub) -> None:
    """A GPU artifact of the same revision is no substitute for a CPU one."""
    service = build_service(settings=settings, hub=hub, resolver=banded_resolver())
    gpu = run_prepare(service, service.plan(SAFETENSORS_REPO, device="gpu"), settings)
    assert gpu.manifest.export.profile != DEFAULT_CPU

    plan = service.plan(SAFETENSORS_REPO, device="cpu")

    assert plan.profile.id == BANDED_CPU
    assert gpu.artifact_dir.parent == plan.artifact_dir.parent, "a sibling the scan does visit"
    assert not any("already prepared as" in note for note in plan.known_limitations)


def test_an_explicit_profile_gets_no_already_prepared_notice(
    settings: Settings, hub: FakeHub
) -> None:
    """Asking for a profile by name is a decision already made; nothing to warn about."""
    service = build_service(settings=settings, hub=hub, resolver=banded_resolver())
    run_prepare(
        service, service.plan(SAFETENSORS_REPO, device="cpu", profile_id=DEFAULT_CPU), settings
    )

    plan = service.plan(SAFETENSORS_REPO, device="cpu", profile_id=BANDED_CPU)

    assert plan.profile.id == BANDED_CPU
    assert plan.profile_selection is not None
    assert plan.profile_selection.basis == "explicit"
    assert not any("already prepared as" in note for note in plan.known_limitations)


def test_an_explicit_default_profile_names_the_automatic_choice(
    settings: Settings, hub: FakeHub
) -> None:
    service = build_service(settings=settings, hub=hub, resolver=banded_resolver())

    plan = service.plan(SAFETENSORS_REPO, device="cpu", profile_id=DEFAULT_CPU)

    selection = plan.profile_selection
    assert selection is not None
    assert selection.basis == "explicit"
    assert selection.automatic_profile_id == BANDED_CPU
    assert plan.artifact_dir.name.endswith(f"--{DEFAULT_CPU}")


# --- sources whose quantization no profile decides -----------------------------------


def test_a_gguf_import_ignores_the_size_band(settings: Settings, tmp_path: Path) -> None:
    """The GGUF file keeps its own quantization, so a small model must not pick a band."""
    hub = GGUFHub(tmp_path, headers())
    hub.repos[GGUF_IMPORT_REPO] = hub.repos[GGUF_IMPORT_REPO].model_copy(
        update={"parameter_count": SMALL_COUNT}
    )
    service = build_service(
        settings=settings, hub=hub, resolver=banded_resolver(), gguf_importer=FakeImporter()
    )

    plan = service.plan(GGUF_IMPORT_REPO, device="cpu")

    assert plan.resolved_source.gguf_file is not None
    assert plan.profile.id == DEFAULT_CPU
    assert plan.artifact_dir.name.endswith("--gguf-q4_0-cpu")
    selection = plan.profile_selection
    assert selection is not None
    assert selection.applies is False
    assert selection.parameter_count is None
    assert not any("already prepared as" in note for note in plan.known_limitations)


def test_a_gguf_import_records_the_selection_but_no_profile_note(
    settings: Settings, tmp_path: Path
) -> None:
    """The record says no profile decided; the notes must not claim one did."""
    hub = GGUFHub(tmp_path, headers())
    service = build_service(
        settings=settings, hub=hub, resolver=banded_resolver(), gguf_importer=FakeImporter()
    )
    plan = service.plan(GGUF_IMPORT_REPO, device="cpu")

    outcome = run_prepare(service, plan, settings)

    on_disk = ArtifactManifest.model_validate_file(manifest_path(outcome.artifact_dir))
    assert on_disk.export.profile == "gguf-q4_0-cpu"
    assert on_disk.export.profile_selection == plan.profile_selection
    assert on_disk.export.profile_selection is not None
    assert on_disk.export.profile_selection.applies is False
    assert not any(note.startswith("Profile ") for note in on_disk.notes)


def test_an_openvino_source_ignores_the_size_band(settings: Settings, hub: FakeHub) -> None:
    """Published IR is copied as it is; its size says nothing about a quantization."""
    service = build_service(settings=settings, hub=hub, resolver=banded_resolver())

    plan = service.plan(OPENVINO_REPO, device="cpu")

    assert plan.resolved_source.already_openvino is True
    assert plan.profile.id == DEFAULT_CPU
    assert plan.artifact_dir.name.endswith(f"--{DEFAULT_CPU}")
    selection = plan.profile_selection
    assert selection is not None
    assert selection.applies is False
    assert selection.parameter_count is None


# --- the manifest --------------------------------------------------------------------


def test_the_manifest_records_why_the_profile_was_chosen(settings: Settings, hub: FakeHub) -> None:
    service = build_service(settings=settings, hub=hub, resolver=banded_resolver())
    plan = service.plan(SAFETENSORS_REPO, device="cpu")

    outcome = run_prepare(service, plan, settings)

    on_disk = ArtifactManifest.model_validate_file(manifest_path(outcome.artifact_dir))
    assert on_disk.export.profile == BANDED_CPU
    selection = on_disk.export.profile_selection
    assert selection is not None
    assert selection == plan.profile_selection
    assert selection.basis == "size_band"
    assert f"Profile {BANDED_CPU}: {selection.sentence}" in on_disk.notes


def test_an_openvino_copy_records_no_profile_note(settings: Settings, hub: FakeHub) -> None:
    """No profile decided a copied IR's quantization, so the manifest must not claim one did."""
    snapshot = hub.snapshots[OPENVINO_REPO]
    for stem in ("openvino_language_model", "openvino_text_embeddings_model"):
        (snapshot / f"{stem}.xml").write_text(MODEL_XML, encoding="utf-8")
        (snapshot / f"{stem}.bin").write_bytes(b"\x00" * 32)
    write_tokenizer_pair(snapshot, broken=False)
    service = build_service(settings=settings, hub=hub, resolver=banded_resolver())
    plan = service.plan(OPENVINO_REPO, device="cpu")

    outcome = run_prepare(service, plan, settings)

    on_disk = ArtifactManifest.model_validate_file(manifest_path(outcome.artifact_dir))
    selection = on_disk.export.profile_selection
    assert selection is not None
    assert selection.applies is False
    assert not any(note.startswith("Profile ") for note in on_disk.notes)


def test_a_manifest_without_a_profile_selection_still_loads(
    settings: Settings, hub: FakeHub
) -> None:
    """Artifacts prepared before the field existed must keep loading."""
    service = build_service(settings=settings, hub=hub, resolver=banded_resolver())
    manifest = run_prepare(service, service.plan(SAFETENSORS_REPO, device="cpu"), settings).manifest

    payload = manifest.to_json()
    export = payload["export"]
    assert isinstance(export, dict)
    del export["profile_selection"]
    restored = ArtifactManifest.model_validate(payload)

    assert restored.export.profile_selection is None
    assert restored.export.profile == BANDED_CPU


def test_a_profile_selection_survives_json() -> None:
    selection = ProfileSelection(
        basis="size_band",
        profile_id=BANDED_CPU,
        parameter_count=SMALL_COUNT,
        parameter_count_basis="Hugging Face Hub count from safetensors headers",
        band="below 2B",
        band_evidence=EvidenceLevel.TESTED,
        sentence="chosen",
    )

    assert ProfileSelection.model_validate_json(selection.model_dump_json()) == selection
