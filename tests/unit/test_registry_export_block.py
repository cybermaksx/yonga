"""An issue rule that refuses an export before the download (registry schema 4).

Phi-4-mini on Python 3.14 used to download 7.2 GiB and start converting before
optimum's ``NormalizedConfig`` defect stopped it, although the registry already
knew the outcome: the ``optimum-normalized-config-allow-new`` rule matched the
failure afterwards. ``blocks_export`` lets that rule say so up front, for the
model types it lists, when its constraints hold for the interpreter that will
run the export.
"""

from __future__ import annotations

import importlib.util
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

from tests.unit.test_exporter_preflight import render
from tests.unit.test_prepare_planning import (
    REGISTRY_VERSIONS,
    FakeExporter,
    FakeHub,
    build_service,
    make_snapshot,
    run_prepare,
    safetensors_metadata,
    settings_for,
)
from yonga.application.jobs import JobContext
from yonga.application.prepare import PrepareService
from yonga.config.settings import Settings
from yonga.domain.diagnostics import (
    ExitCode,
    FailureCategory,
    PreflightError,
    RegistryError,
)
from yonga.domain.environments import EnvironmentSpec, ExportEnvironment
from yonga.domain.invocation import CommandLine
from yonga.domain.models import ModelMetadata
from yonga.domain.plans import ExporterRequirements, PreparationPlan
from yonga.registry.loader import (
    load_builtin_registry,
    load_registry_from_documents,
    parse_document,
    validate_registry,
)
from yonga.registry.resolver import RegistryResolver
from yonga.registry.schema import (
    EXPORT_BLOCK_SCHEMA_VERSION,
    SCHEMA_MIN_TOOL_VERSION,
    SCHEMA_VERSION,
    ExportBlock,
    IssueRule,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PHI_REPO = "microsoft/Phi-4-mini-instruct"
RULE_ID = "optimum-normalized-config-allow-new"
CODE = "export.optimum.normalized_config"

#: The reference host as the AppImage and the e2e pip install see it.
PY314 = {**REGISTRY_VERSIONS, "python": "3.14.4", "optimum": "2.3.0"}

DOCUMENT = textwrap.dedent(
    """
    schema_version: 4
    issues:
      - id: demo-block
        constraints:
          python: ">=3.14"
        match:
          exception_contains:
            - "demo failure"
        classification:
          code: export.demo
          category: package_abi_mismatch
        blocks_export:
          model_types: [Phi3, mistral]
          constraints:
            optimum: "<=2.3.0"
    """
)


def phi_metadata(model_type: str = "phi3") -> ModelMetadata:
    meta = safetensors_metadata(
        repo_id=PHI_REPO, model_type=model_type, pipeline_tag="text-generation"
    )
    return meta.model_copy(
        update={
            "architectures": ["Phi3ForCausalLM"],
            "config": {"model_type": model_type, "architectures": ["Phi3ForCausalLM"]},
        }
    )


def resolver(versions: dict[str, str | None]) -> RegistryResolver:
    return RegistryResolver(load_builtin_registry(), versions=versions)


# --- the schema -------------------------------------------------------------------------


def test_schema_four_introduces_blocks_export_and_needs_yonga_0_5() -> None:
    assert SCHEMA_VERSION == EXPORT_BLOCK_SCHEMA_VERSION == 4
    assert SCHEMA_MIN_TOOL_VERSION[4] == "0.5.0"


def test_blocks_export_needs_schema_four_on_the_rule() -> None:
    with pytest.raises(ValidationError, match="blocks_export needs schema_version 4"):
        IssueRule.model_validate(
            {
                "id": "demo",
                "schema_version": 3,
                "classification": {"code": "x", "category": "package_abi_mismatch"},
                "blocks_export": {"model_types": ["phi3"]},
            }
        )


def test_a_rule_without_its_own_version_takes_the_document_version() -> None:
    with pytest.raises(RegistryError) as excinfo:
        parse_document(DOCUMENT.replace("schema_version: 4", "schema_version: 3"), origin="o.yaml")
    assert excinfo.value.diagnostic.code == "registry.schema_violation"
    assert "blocks_export needs schema_version 4" in excinfo.value.diagnostic.message


def test_a_schema_four_document_loads_and_validates() -> None:
    document = parse_document(DOCUMENT, origin="demo.yaml")
    (rule,) = document.issues
    assert rule.schema_version == 4
    assert rule.blocks_export is not None
    assert rule.blocks_export.model_types == ["phi3", "mistral"]
    assert validate_registry(load_registry_from_documents([document])) == []


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ({"model_types": []}, "at least 1"),
        ({"model_types": ["  "]}, "at least one model type"),
        ({"model_types": ["phi3"], "surprise": True}, "Extra inputs"),
    ],
)
def test_an_empty_or_unknown_block_is_refused(block: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ExportBlock.model_validate(block)


def test_validation_reports_an_unknown_block_constraint() -> None:
    text = DOCUMENT.replace('optimum: "<=2.3.0"', 'optimun: "<=2.3.0"')
    registry = load_registry_from_documents([parse_document(text, origin="demo.yaml")])
    assert any(
        "blocks_export: unknown constraint key 'optimun'" in p for p in validate_registry(registry)
    )


@pytest.fixture(scope="module")
def bundle_tool() -> ModuleType:
    path = REPO_ROOT / "tools" / "registry_bundle.py"
    spec = importlib.util.spec_from_file_location("registry_bundle_tool_v4", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_bundle_carrying_an_export_block_needs_yonga_0_5(bundle_tool: ModuleType) -> None:
    assert bundle_tool.schema_versions({"i.yaml": DOCUMENT}) == {4}
    assert bundle_tool.required_tool_version({"i.yaml": DOCUMENT}, None) == "0.5.0"


# --- the built-in rule, resolved --------------------------------------------------------


def test_phi3_on_python_3_14_with_optimum_2_3_0_is_refused_with_the_rules_advice() -> None:
    found = resolver(PY314).export_blocker(phi_metadata())

    assert found is not None
    assert found.code == CODE
    assert found.matched_rule_id == RULE_ID
    assert found.category is FailureCategory.PACKAGE_ABI_MISMATCH
    assert found.exit_code() is ExitCode.ENVIRONMENT_UNUSABLE
    assert found.context["versions"] == "optimum 2.3.0, python 3.14.4"
    assert "before any conversion" in found.message
    joined = " ".join(found.remediation)
    # The existing advice, AppImage and pip users alike, is kept as it is.
    assert "Python 3.11, 3.12 or 3.13" in joined
    assert "The AppImage cannot do that" in joined
    assert "python3.13 -m venv" in joined


@pytest.mark.parametrize("model_type", ["mistral", "mixtral", "phimoe", "gpt2", "bloom"])
def test_every_listed_model_type_is_refused(model_type: str) -> None:
    assert resolver(PY314).export_blocker(phi_metadata(model_type)) is not None


@pytest.mark.parametrize(
    "versions",
    [
        {**PY314, "python": "3.13.7"},
        {**PY314, "optimum": "2.3.1"},
        {**PY314, "optimum": "2.4.0.dev0"},
        {**PY314, "optimum": None},
        {**PY314, "python": None},
    ],
    ids=["python-3.13", "fixed-optimum", "optimum-from-main", "optimum-unknown", "python-unknown"],
)
def test_nothing_is_refused_outside_the_known_versions(versions: dict[str, str | None]) -> None:
    assert resolver(versions).export_blocker(phi_metadata()) is None


@pytest.mark.parametrize("model_type", ["llama", "qwen3", None])
def test_an_unaffected_model_type_is_never_refused(model_type: str | None) -> None:
    meta = phi_metadata().model_copy(update={"model_type": model_type})
    assert resolver(PY314).export_blocker(meta) is None


def test_the_rule_still_classifies_the_failure_with_a_fixed_optimum() -> None:
    """The block's optimum bound narrows the refusal, never the classification."""
    classified = resolver({**PY314, "optimum": "2.3.1"}).classify(
        raw_error="TypeError: NormalizedConfig.__init__() got multiple values for argument "
        "'allow_new'",
        stage="export",
    )
    assert classified.code == CODE


# --- prepare ----------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return settings_for(tmp_path)


@pytest.fixture
def hub(tmp_path: Path) -> FakeHub:
    return FakeHub(
        repos={PHI_REPO: phi_metadata()},
        snapshots={PHI_REPO: make_snapshot(tmp_path, "snapshot-phi")},
    )


def phi_service(
    settings: Settings,
    hub: FakeHub,
    versions: dict[str, str | None],
    *,
    exporter: FakeExporter | None = None,
    environments: PinnedEnvironments | None = None,
) -> PrepareService:
    return build_service(
        settings=settings,
        hub=hub,
        resolver=resolver(versions),
        exporter=exporter,
        environments=environments,
    )


def test_prepare_refuses_before_the_download_and_before_any_question(
    settings: Settings, hub: FakeHub
) -> None:
    exporter = FakeExporter()
    service = phi_service(settings, hub, PY314, exporter=exporter)
    plan = service.plan(PHI_REPO, device="cpu")
    assert plan.export_blocked is not None
    asked: list[str] = []

    def record(reason: object) -> bool:
        asked.append(str(reason))
        return True

    with pytest.raises(PreflightError) as caught:
        run_prepare(service, plan, settings, confirm=record)

    failure = caught.value.diagnostic
    assert caught.value.exit_code() is ExitCode.ENVIRONMENT_UNUSABLE
    assert failure.code == CODE
    assert "the Python environment yonga runs in, with optimum 2.3.0, python 3.14.4" in (
        failure.message
    )
    assert "Nothing has been downloaded." in failure.message
    assert any("--ignore-exporter-requirements" in line for line in failure.remediation)
    assert hub.downloaded == []
    assert exporter.requests == []
    assert asked == []


def test_the_override_step_repeats_the_users_whole_command(
    settings: Settings, hub: FakeHub
) -> None:
    """The step names --ignore-exporter-requirements on the command as typed, root options too."""
    service = phi_service(settings, hub, PY314)
    command = CommandLine(
        command=("prepare",),
        arguments=(PHI_REPO,),
        options=(("--device", "cpu"),),
        root_options=(("--data-dir", "/srv/my models"),),
    )
    plan = service.plan(PHI_REPO, device="cpu", command_line=command)

    with pytest.raises(PreflightError) as caught:
        run_prepare(service, plan, settings)

    assert caught.value.diagnostic.remediation[-1].endswith(
        f"yonga --data-dir '/srv/my models' prepare {PHI_REPO} --device cpu "
        "--ignore-exporter-requirements"
    )


def test_prepare_json_reports_the_refusal_as_its_failure_document(
    hub: FakeHub, tmp_path: Path
) -> None:
    """``docs/REGISTRY_SPEC.md`` promised an ``export_blocked`` key no document carried."""
    from typer.testing import CliRunner

    from tests.unit.test_cli import FakeWiring, invoke, one_document

    home = tmp_path / "home"
    (home / "config").mkdir(parents=True)
    result = invoke(
        CliRunner(),
        home,
        ["prepare", PHI_REPO, "--device", "cpu", "--yes", "--json"],
        factory=lambda settings: FakeWiring(settings, hub=hub, resolver=resolver(PY314)),
    )

    document = one_document(result)
    assert set(document) == {"ok", "exit_code", "diagnostic"}
    assert document["exit_code"] == int(ExitCode.ENVIRONMENT_UNUSABLE)
    found = document["diagnostic"]
    assert isinstance(found, dict)
    assert found["code"] == CODE
    assert str(found["remediation"][-1]).endswith("--ignore-exporter-requirements")
    assert hub.downloaded == []
    spec = (Path(__file__).resolve().parents[2] / "docs" / "REGISTRY_SPEC.md").read_text(
        encoding="utf-8"
    )
    assert "`export_blocked`" not in spec


def test_the_plan_panel_says_the_export_is_blocked(settings: Settings, hub: FakeHub) -> None:
    plan = phi_service(settings, hub, PY314).plan(PHI_REPO, device="cpu")
    text = render(plan)
    assert "BLOCKED — this export is known to fail here" in text
    assert "optimum 2.3.0, python 3.14.4" in text


def test_python_3_13_prepares_as_before(settings: Settings, hub: FakeHub) -> None:
    service = phi_service(settings, hub, {**PY314, "python": "3.13.7"})
    plan = service.plan(PHI_REPO, device="cpu")
    assert plan.export_blocked is None

    run_prepare(service, plan, settings)
    assert hub.downloaded == [PHI_REPO]


def test_the_override_flag_exports_anyway_and_the_manifest_says_so(
    settings: Settings, hub: FakeHub
) -> None:
    service = phi_service(settings, hub, PY314)
    plan = service.plan(PHI_REPO, device="cpu")

    outcome = run_prepare(service, plan, settings, ignore_exporter_requirements=True)

    assert hub.downloaded == [PHI_REPO]
    assert any(
        note.startswith("Known exporter failure overridden with --ignore-exporter-requirements")
        and RULE_ID in note
        for note in outcome.manifest.notes
    )


# --- the interpreter that will export decides -----------------------------------------


class NoToolchainExporter(FakeExporter):
    """yonga's own environment has no exporter at all, as in the AppImage."""

    def toolchain_available(self) -> bool:
        return False


@dataclass
class PinnedEnvironments:
    """A managed environment built from ``python`` with optimum pinned as given."""

    root: Path
    python: str
    optimum: str

    def spec_for(
        self,
        requirements: ExporterRequirements,
        *,
        installed: object = None,
        reason: str = "",
    ) -> EnvironmentSpec:
        return EnvironmentSpec(
            python=self.python, packages={"optimum": self.optimum}, reason=reason
        )

    def find(self, spec: EnvironmentSpec) -> ExportEnvironment | None:
        return None

    def ensure(self, spec: EnvironmentSpec, ctx: JobContext) -> ExportEnvironment:
        return ExportEnvironment(
            spec=spec,
            path=self.root / spec.name,
            installed={"optimum": "2.3.0"},
            created_at=datetime(2026, 10, 11, tzinfo=UTC),
        )


def managed_plan(
    settings: Settings, hub: FakeHub, tmp_path: Path, *, python: str, optimum: str
) -> PreparationPlan:
    # yonga itself runs on 3.11 here: only the environment's interpreter can block.
    service = phi_service(
        settings,
        hub,
        {**REGISTRY_VERSIONS, "optimum": None},
        exporter=NoToolchainExporter(),
        environments=PinnedEnvironments(tmp_path / "envs", python, optimum),
    )
    plan = service.plan(PHI_REPO, device="cpu")
    assert plan.export_environment is not None
    return plan


def test_a_managed_environment_on_python_3_14_is_refused(
    settings: Settings, hub: FakeHub, tmp_path: Path
) -> None:
    plan = managed_plan(settings, hub, tmp_path, python="3.14", optimum="==2.3.0")
    assert plan.export_blocked is not None
    assert plan.export_blocked.context["versions"] == "optimum 2.3.0, python 3.14"


@pytest.mark.parametrize(
    ("python", "optimum"),
    [("3.13", "==2.3.0"), ("3.14", "==2.3.1"), ("3.14", ">=2.3")],
    ids=["python-3.13", "fixed-optimum", "unpinned-optimum"],
)
def test_a_managed_environment_outside_the_known_versions_is_not(
    settings: Settings, hub: FakeHub, tmp_path: Path, python: str, optimum: str
) -> None:
    plan = managed_plan(settings, hub, tmp_path, python=python, optimum=optimum)
    assert plan.export_blocked is None
