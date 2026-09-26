"""Preflight refuses an export whose declared package requirements are unmet.

The defect this module pins down was observed end to end on the reference host:
``yonga prepare HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive`` printed

.. code-block:: text

    Exporter problems
    • transformers ==5.2.* is required, but 5.5.4 is installed

…and then resolved the derivative, downloaded 17.5 GiB and started the
conversion anyway, only to die several minutes later inside
``optimum/exporters/openvino/model_patcher.py`` with

.. code-block:: text

    ImportError: cannot import name 'Qwen3_5DynamicCache' from
    'transformers.models.qwen3_5.modeling_qwen3_5'

The registry rule is right and stays as it is: ``transformers 5.5.4`` really
does load ``Qwen3_5Config`` and report ``model_type: qwen3_5``, so the
architecture looks supported, but optimum-intel 2.2.0's export patcher needs a
symbol 5.5.4 dropped. What was wrong is that preflight printed the problem and
proceeded. ``docs/MODEL_PIPELINE.md`` section 5 counts package compatibility
among the things preflight computes and ``SPEC.md`` section 10 says preflight
exists to fail before expensive work, so the download and the quantization must
never happen.

The fakes come from ``tests/unit/test_prepare_planning.py``, which owns the
adapter doubles for this group of preparation tests.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner
from typer.testing import Result as CliResult

from tests.support import RefusingRunner, SimulatedPipRunner, plain_text
from tests.unit.test_prepare_planning import (
    INSTALLED_EXPORT_VERSIONS,
    MISMATCHED_EXPORT_VERSIONS,
    MODEL_XML,
    OPENVINO_REPO,
    SAFETENSORS_REPO,
    FakeExporter,
    FakeFilesystemProbe,
    FakeHardwareProbe,
    FakeHub,
    FakeRuntimeProbe,
    FakeSmokeTester,
    build_service,
    builtin_resolver,
    make_snapshot,
    openvino_metadata,
    reference_inventory,
    run_prepare,
    safetensors_metadata,
    settings_for,
    write_tokenizer_pair,
    yes_flag,
)
from yonga.adapters import (
    CommandRunner,
    Exporter,
    FilesystemProbe,
    HardwareProbe,
    HubClient,
    RuntimeProbe,
)
from yonga.adapters.filesystem import system_temp_dir
from yonga.application.prepare import (
    IGNORE_EXPORTER_REQUIREMENTS_FLAG,
    PrepareService,
    SmokeTester,
    requirement_state,
)
from yonga.application.wiring import Wiring
from yonga.cli import app
from yonga.config.settings import Settings
from yonga.domain.diagnostics import ExitCode, FailureCategory, PreflightError
from yonga.domain.plans import ExporterRequirements, PreparationPlan
from yonga.registry.resolver import RegistryResolver
from yonga.ui import tables

#: The token the CLI harness exports, mirrored from ``tests/unit/test_cli.py``.
FAKE_TOKEN = "hf_fake_token_for_tests"


# --- fixtures -------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings confined to the test's temporary directory."""
    return settings_for(tmp_path)


@pytest.fixture
def hub(tmp_path: Path) -> FakeHub:
    """A hub serving the safetensors Qwen3.5 acceptance repository."""
    return FakeHub(
        repos={SAFETENSORS_REPO: safetensors_metadata()},
        snapshots={SAFETENSORS_REPO: make_snapshot(tmp_path, "snapshot-safetensors")},
    )


def _log_records(path: Path) -> Iterator[dict[str, object]]:
    """Yield every parsable JSON record from a job log."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            assert isinstance(record, dict)
            yield record


def mismatched_exporter() -> FakeExporter:
    """An exporter reporting the reference host's real ``transformers`` mismatch."""
    return FakeExporter(versions=dict(MISMATCHED_EXPORT_VERSIONS))


def service_with(
    settings: Settings,
    hub: FakeHub,
    exporter: FakeExporter,
) -> PrepareService:
    """Build the preparation service around one specific exporter double."""
    return build_service(settings=settings, hub=hub, exporter=exporter)


# --- the block ------------------------------------------------------------------------


def test_unmet_requirement_stops_before_the_download_and_the_export(
    settings: Settings, hub: FakeHub
) -> None:
    """Nothing is fetched and nothing is converted: preflight is where this ends."""
    exporter = mismatched_exporter()
    service = service_with(settings, hub, exporter)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    assert plan.exporter.satisfied is False

    with pytest.raises(PreflightError) as caught:
        run_prepare(service, plan, settings)

    assert hub.downloaded == []
    assert exporter.requests == []
    assert not plan.artifact_dir.exists()
    assert caught.value.diagnostic.code == "preflight.exporter_requirement_unsatisfied"


def test_the_stop_is_recorded_against_the_preflight_stage(settings: Settings, hub: FakeHub) -> None:
    """The job state says where the pipeline died, so ``yonga logs`` can explain it."""
    service = service_with(settings, hub, mismatched_exporter())
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    with pytest.raises(PreflightError):
        run_prepare(service, plan, settings)

    state = json.loads(settings.paths.job_state_path(plan.job_id).read_text(encoding="utf-8"))
    assert state["stage"] == "failed"
    assert state["artifact_dir"] is None
    assert state["diagnostic"]["code"] == "preflight.exporter_requirement_unsatisfied"

    failures = [
        record
        for record in _log_records(settings.paths.job_log_path(plan.job_id))
        if record.get("event") == "job.failed"
    ]
    assert [record["stage"] for record in failures] == ["preflight_ok"]


def test_the_diagnostic_names_both_versions_and_the_pip_command(
    settings: Settings, hub: FakeHub
) -> None:
    """A user must be able to act on this without reading the job log."""
    service = service_with(settings, hub, mismatched_exporter())
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    with pytest.raises(PreflightError) as caught:
        run_prepare(service, plan, settings)

    report = caught.value.diagnostic
    assert "transformers ==5.2.*" in report.message
    assert "5.5.4 is installed" in report.message
    remediation = " ".join(report.remediation)
    assert "pip install 'transformers==5.2.*'" in remediation
    # AGENT_PROMPT.md 13.2: yonga diagnoses and instructs, it never installs.
    assert "never installs" in remediation
    assert "isolated environment" in remediation and "--no-managed-env" in remediation
    assert IGNORE_EXPORTER_REQUIREMENTS_FLAG in remediation


def test_the_category_is_a_package_mismatch_and_fixes_the_exit_code(
    settings: Settings, hub: FakeHub
) -> None:
    """Every package is installed; it is their combination that cannot export.

    ``PACKAGE_ABI_MISMATCH`` maps to :data:`ExitCode.ENVIRONMENT_UNUSABLE` rather
    than to :data:`ExitCode.PREFLIGHT`. That is deliberate: 10 answers *why* the
    job stopped, even though it stopped *in* preflight.
    """
    service = service_with(settings, hub, mismatched_exporter())
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    with pytest.raises(PreflightError) as caught:
        run_prepare(service, plan, settings)

    report = caught.value.diagnostic
    assert report.category is FailureCategory.PACKAGE_ABI_MISMATCH
    assert report.exit_code() is ExitCode.ENVIRONMENT_UNUSABLE
    assert caught.value.exit_code() is ExitCode.ENVIRONMENT_UNUSABLE
    assert report.environment_broken is True


def test_yes_alone_does_not_bypass_the_block(settings: Settings, hub: FakeHub) -> None:
    """``--yes`` answers ordinary questions; this is not one of them."""
    exporter = mismatched_exporter()
    service = service_with(settings, hub, exporter)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    with pytest.raises(PreflightError):
        run_prepare(service, plan, settings, confirm=yes_flag)

    assert hub.downloaded == []
    assert exporter.requests == []


# --- the escape hatch -----------------------------------------------------------------


def test_the_explicit_flag_lets_a_user_who_knows_better_proceed(
    settings: Settings, hub: FakeHub
) -> None:
    """A stale pin must not be a dead end; the export runs and the artifact lands."""
    exporter = mismatched_exporter()
    service = service_with(settings, hub, exporter)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    outcome = run_prepare(service, plan, settings, ignore_exporter_requirements=True)

    assert hub.downloaded == [SAFETENSORS_REPO]
    assert len(exporter.requests) == 1
    assert outcome.manifest.ready is True


def test_the_manifest_records_the_override_with_both_versions(
    settings: Settings, hub: FakeHub
) -> None:
    """The artifact's lineage must show that a declared requirement was overridden."""
    service = service_with(settings, hub, mismatched_exporter())
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    outcome = run_prepare(service, plan, settings, ignore_exporter_requirements=True)

    overrides = [note for note in outcome.manifest.notes if "overridden" in note]
    assert len(overrides) == 1
    note = overrides[0]
    assert IGNORE_EXPORTER_REQUIREMENTS_FLAG in note
    assert "transformers ==5.2.*" in note
    assert "5.5.4 is installed" in note
    # The same sentence reaches the caller, so the CLI prints it as a warning.
    assert note in outcome.warnings


# --- satisfied, unknown and absent ----------------------------------------------------


def test_a_satisfied_requirement_proceeds_untouched(settings: Settings, hub: FakeHub) -> None:
    """The healthy host converts as before, with no override note anywhere."""
    exporter = FakeExporter(versions=dict(INSTALLED_EXPORT_VERSIONS))
    service = service_with(settings, hub, exporter)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    assert plan.exporter.satisfied is True

    outcome = run_prepare(service, plan, settings)

    assert hub.downloaded == [SAFETENSORS_REPO]
    assert len(exporter.requests) == 1
    assert not any("Exporter requirement" in note for note in outcome.manifest.notes)


def test_an_unparsable_installed_version_does_not_block(settings: Settings, hub: FakeHub) -> None:
    """Unknown is not wrong.

    ``satisfies`` answers ``False`` for a version it cannot parse exactly as it
    does for a real contradiction, so a block inferred from that bare ``False``
    would refuse to convert on a host yonga simply cannot read.
    """
    exporter = FakeExporter(
        versions={**INSTALLED_EXPORT_VERSIONS, "transformers": "unreleased-nightly"}
    )
    service = service_with(settings, hub, exporter)

    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    assert plan.exporter.problems == []
    assert plan.exporter.satisfied is True
    assert any("could not be checked" in note for note in plan.known_limitations)

    outcome = run_prepare(service, plan, settings)

    assert hub.downloaded == [SAFETENSORS_REPO]
    assert outcome.manifest.ready is True


def test_an_absent_exporter_package_blocks_and_says_so(settings: Settings, hub: FakeHub) -> None:
    """A distribution that is not installed at all reads differently from a wrong one."""
    exporter = FakeExporter(versions={**INSTALLED_EXPORT_VERSIONS, "transformers": None})
    service = service_with(settings, hub, exporter)

    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    assert plan.exporter.satisfied is False
    assert "transformers is not installed" in " ".join(plan.exporter.problems)

    with pytest.raises(PreflightError) as caught:
        run_prepare(service, plan, settings)

    message = caught.value.diagnostic.message
    assert "transformers is not installed" in message
    assert hub.downloaded == []


def test_an_architecture_without_a_pin_is_never_blocked(settings: Settings, hub: FakeHub) -> None:
    """No declared requirement means nothing to check, not an invented failure."""
    service = service_with(settings, hub, mismatched_exporter())
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    unpinned = plan.model_copy(update={"exporter": ExporterRequirements()})
    assert unpinned.exporter.satisfied is True

    outcome = run_prepare(service, unpinned, settings)

    assert hub.downloaded == [SAFETENSORS_REPO]
    assert outcome.manifest.ready is True


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        ("5.2.1", "ok"),
        ("5.2.0", "ok"),
        ("5.5.4", "mismatch"),
        (None, "missing"),
        ("unreleased-nightly", "unverifiable"),
        ("", "unverifiable"),
    ],
)
def test_requirement_state_keeps_the_four_outcomes_apart(
    installed: str | None, expected: str
) -> None:
    """The classifier, not ``satisfies``, is what preflight is allowed to block on."""
    assert requirement_state(installed, "==5.2.*") == expected


# --- the plan panel -------------------------------------------------------------------


def render(plan: PreparationPlan) -> str:
    """Render the preparation plan panel to plain text."""
    from rich.console import Console

    console = Console(width=200, no_color=True, force_terminal=False, record=True)
    console.print(tables.preparation_plan(plan))
    return console.export_text()


def test_the_plan_panel_marks_the_problems_as_blocking(settings: Settings, hub: FakeHub) -> None:
    """The panel must not read like one more advisory bullet list."""
    service = service_with(settings, hub, mismatched_exporter())
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    text = render(plan)

    assert "BLOCKED — exporter requirements not satisfied" in text
    assert "transformers ==5.2.* is required, but 5.5.4 is installed" in text
    assert IGNORE_EXPORTER_REQUIREMENTS_FLAG in text
    assert "--yes does not" in text
    # Still visually separate from the advisory confirmation list above it.
    assert "Confirmation required" in text
    assert text.index("Confirmation required") < text.index("BLOCKED")


def test_a_satisfied_plan_shows_no_blocking_section(settings: Settings, hub: FakeHub) -> None:
    """Nothing is blocking, so nothing shouts."""
    service = service_with(settings, hub, FakeExporter())
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    text = render(plan)

    assert "BLOCKED" not in text
    assert IGNORE_EXPORTER_REQUIREMENTS_FLAG not in text


def test_the_panel_does_not_announce_a_block_the_service_will_not_perform(
    tmp_path: Path,
) -> None:
    """An already-OpenVINO source is copied, not converted, so nothing is blocked."""
    settings = settings_for(tmp_path)
    hub = FakeHub(
        repos={OPENVINO_REPO: openvino_metadata()},
        snapshots={OPENVINO_REPO: make_snapshot(tmp_path, "snapshot-openvino")},
    )
    service = service_with(settings, hub, mismatched_exporter())
    plan = service.plan(OPENVINO_REPO, device="gpu")
    assert plan.exporter.problems, "the pin is genuinely unmet; it is just irrelevant here"
    assert plan.exporter_requirements_block is False

    text = render(plan)

    assert "BLOCKED" not in text
    assert IGNORE_EXPORTER_REQUIREMENTS_FLAG not in text


def test_the_renderer_and_the_service_spell_the_flag_identically() -> None:
    """The UI states the override; a typo there would send users to a dead flag."""
    assert tables._IGNORE_EXPORTER_REQUIREMENTS_FLAG == IGNORE_EXPORTER_REQUIREMENTS_FLAG


# --- the CLI surface ------------------------------------------------------------------


class PreflightWiring(Wiring):
    """A :class:`Wiring` whose adapters are the preparation fakes.

    Only adapter factories are overridden, so ``yonga prepare`` runs the real
    service, the real preflight and the real exit-code mapping.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        hub: FakeHub,
        exporter: FakeExporter,
        runner: CommandRunner | None = None,
    ) -> None:
        """Bind the doubles this CLI test needs.

        ``runner`` defaults to one that refuses to spawn anything, so the real
        export-environment manager can never reach the network from this test.
        """
        super().__init__(settings)
        self._fake_hub = hub
        self._fake_exporter = exporter
        self._fake_runner: CommandRunner = runner if runner is not None else RefusingRunner()
        self._fake_hardware = FakeHardwareProbe(reference_inventory())
        self._fake_filesystem = FakeFilesystemProbe(tmpfs_paths=(system_temp_dir(),))
        self._fake_runtime = FakeRuntimeProbe()
        self._fake_smoke = FakeSmokeTester()
        self._fake_resolver = builtin_resolver()

    def hub_client(self) -> HubClient:
        """Return the in-memory hub."""
        return self._fake_hub

    def exporter(self) -> Exporter:
        """Return the exporter double, including its reported package versions."""
        return self._fake_exporter

    def hardware_probe(self) -> HardwareProbe:
        """Return the reference-host inventory."""
        return self._fake_hardware

    def filesystem(self) -> FilesystemProbe:
        """Return the reference-host filesystem model."""
        return self._fake_filesystem

    def runtime_probe(self) -> RuntimeProbe:
        """Return the file-driven runtime probe."""
        return self._fake_runtime

    def smoke_tester(self) -> SmokeTester | None:
        """Return the smoke test that never builds a GenAI pipeline."""
        return self._fake_smoke

    def registry_resolver(self) -> RegistryResolver:
        """Return a resolver with pinned versions, so rules match deterministically."""
        return self._fake_resolver

    def command_runner(self) -> CommandRunner:
        """Return the runner that refuses (or simulates) every process."""
        return self._fake_runner


def invoke_prepare(
    home: Path,
    args: Sequence[str],
    *,
    hub: FakeHub,
    exporter: FakeExporter,
    runner: CommandRunner | None = None,
) -> CliResult:
    """Run ``yonga prepare`` against the fakes, inside private XDG directories."""

    def build(settings: Settings) -> Wiring:
        """Return the faked adapter graph for one invocation."""
        return PreflightWiring(settings, hub=hub, exporter=exporter, runner=runner)

    return CliRunner().invoke(
        app,
        ["--data-dir", str(home / "data"), "--cache-dir", str(home / "cache"), *args],
        obj=build,
        env={
            "YONGA_CONFIG_DIR": str(home / "config"),
            "NO_COLOR": "1",
            "HF_TOKEN": FAKE_TOKEN,
            "COLUMNS": "200",
        },
    )


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """A private XDG root so no test can read or write the developer's own data."""
    root = tmp_path / "home"
    (root / "config").mkdir(parents=True)
    return root


def test_cli_prepare_with_yes_stops_with_the_environment_exit_code(
    home: Path, tmp_path: Path
) -> None:
    """With managed environments refused, ``--yes`` is still not a bypass."""
    hub = FakeHub(
        repos={SAFETENSORS_REPO: safetensors_metadata()},
        snapshots={SAFETENSORS_REPO: make_snapshot(tmp_path, "snapshot-cli")},
    )
    exporter = mismatched_exporter()

    result = invoke_prepare(
        home,
        ["prepare", SAFETENSORS_REPO, "--device", "cpu", "--yes", "--no-managed-env"],
        hub=hub,
        exporter=exporter,
    )

    assert result.exit_code == int(ExitCode.ENVIRONMENT_UNUSABLE)
    assert hub.downloaded == []
    assert exporter.requests == []
    stderr = " ".join(result.stderr.split())
    assert "pip install 'transformers==5.2.*'" in stderr


def test_cli_prepare_with_yes_builds_a_managed_environment_and_converts(
    home: Path, tmp_path: Path
) -> None:
    """The fix, at the command line: ``--yes`` approves the environment, the export runs in it.

    The real environment manager runs; only the processes it would spawn are
    simulated. The export is handed the environment's interpreter.
    """
    hub = FakeHub(
        repos={SAFETENSORS_REPO: safetensors_metadata()},
        snapshots={SAFETENSORS_REPO: make_snapshot(tmp_path, "snapshot-cli")},
    )
    exporter = mismatched_exporter()
    pip = SimulatedPipRunner()

    result = invoke_prepare(
        home,
        ["prepare", SAFETENSORS_REPO, "--device", "cpu", "--yes"],
        hub=hub,
        exporter=exporter,
        runner=pip,
    )

    assert result.exit_code in (int(ExitCode.SUCCESS), int(ExitCode.WARNINGS)), result.stderr
    assert pip.requirements["transformers"] in ("==5.2.*", "==5.2.0", "==5.2.1")
    (request,) = exporter.requests
    assert request.interpreter is not None
    assert request.interpreter.parent.parent.parent == home / "cache" / "envs"
    assert "managed export environment" in " ".join(result.stdout.split())


def test_cli_ignoring_requirements_never_builds_an_environment(home: Path, tmp_path: Path) -> None:
    """``--ignore-exporter-requirements`` means "export here"; a managed env would defeat it."""
    hub = FakeHub(
        repos={SAFETENSORS_REPO: safetensors_metadata()},
        snapshots={SAFETENSORS_REPO: make_snapshot(tmp_path, "snapshot-cli")},
    )
    exporter = mismatched_exporter()

    result = invoke_prepare(
        home,
        [
            "prepare",
            SAFETENSORS_REPO,
            "--device",
            "cpu",
            "--yes",
            IGNORE_EXPORTER_REQUIREMENTS_FLAG,
        ],
        hub=hub,
        exporter=exporter,
    )

    assert result.exit_code in (int(ExitCode.SUCCESS), int(ExitCode.WARNINGS)), result.stderr
    assert exporter.requests[0].interpreter is None


def test_cli_prepare_with_the_explicit_flag_converts(home: Path, tmp_path: Path) -> None:
    """The escape hatch is reachable from the command line and it works."""
    hub = FakeHub(
        repos={SAFETENSORS_REPO: safetensors_metadata()},
        snapshots={SAFETENSORS_REPO: make_snapshot(tmp_path, "snapshot-cli")},
    )
    exporter = mismatched_exporter()

    result = invoke_prepare(
        home,
        [
            "prepare",
            SAFETENSORS_REPO,
            "--device",
            "cpu",
            "--yes",
            IGNORE_EXPORTER_REQUIREMENTS_FLAG,
        ],
        hub=hub,
        exporter=exporter,
    )

    assert result.exit_code in (int(ExitCode.SUCCESS), int(ExitCode.WARNINGS))
    assert hub.downloaded == [SAFETENSORS_REPO]
    assert len(exporter.requests) == 1


def test_prepare_help_offers_the_flag_separately_from_yes(home: Path, tmp_path: Path) -> None:
    """The two flags are independent options, exactly like ``--trust-remote-code``."""
    hub = FakeHub()
    result = invoke_prepare(home, ["prepare", "--help"], hub=hub, exporter=FakeExporter())

    text = plain_text(result.stdout)
    assert IGNORE_EXPORTER_REQUIREMENTS_FLAG in text
    assert "--yes never enables it" in text


def test_a_published_openvino_repository_is_not_blocked_by_an_exporter_pin(
    tmp_path: Path,
) -> None:
    """A repository that already ships IR is copied, never exported, so no pin applies.

    ``PrepareService._convert`` dispatches to ``_copy_openvino_source`` when
    ``resolved_source.already_openvino`` is set: ``transformers`` is not on that
    job's path at all. Blocking it would refuse a job that has no export to
    fail, with a diagnostic promising a conversion that was never going to run.
    """
    settings = settings_for(tmp_path)
    snapshot = make_snapshot(tmp_path, "snapshot-openvino")
    for stem in ("openvino_language_model", "openvino_text_embeddings_model"):
        (snapshot / f"{stem}.xml").write_text(MODEL_XML, encoding="utf-8")
        (snapshot / f"{stem}.bin").write_bytes(b"\x00" * 32)
    write_tokenizer_pair(snapshot, broken=False)
    hub = FakeHub(
        repos={OPENVINO_REPO: openvino_metadata()},
        snapshots={OPENVINO_REPO: snapshot},
    )
    exporter = mismatched_exporter()
    service = service_with(settings, hub, exporter)

    plan = service.plan(OPENVINO_REPO, device="gpu")
    # The pin really is unmet; the point is that it is irrelevant here.
    assert plan.exporter.satisfied is False
    assert plan.resolved_source.already_openvino is True

    run_prepare(service, plan, settings)

    assert hub.downloaded == [OPENVINO_REPO]
    assert exporter.requests == []
