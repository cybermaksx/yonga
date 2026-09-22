"""Seams between the packages that were written in parallel.

Each package has its own suite; these tests only assert the things that no
single package can assert about itself, because they involve two of them:

* every concrete adapter really satisfies the Protocol the application layer
  will depend on, by name and by signature;
* the repair implementations the registry data names really exist in
  ``yonga.repairs``, and the issue rules' ``repair:`` references really resolve;
* a rule selected by ``RegistryResolver.repairs_for`` is a rule that
  ``apply_repairs`` will actually run -- the two layers evaluate the same
  version constraint, so they must judge it against the same facts;
* nothing reachable from ``yonga.cli`` imports OpenVINO, torch, transformers,
  optimum or nncf at module scope.

They are deliberately cheap: no network, no OpenVINO, no model download.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from yonga.adapters import (
    CommandRunner,
    Exporter,
    FilesystemProbe,
    HardwareProbe,
    HubClient,
    RuntimeProbe,
)
from yonga.adapters.exporter_optimum import OptimumExporter
from yonga.adapters.filesystem import FilesystemAdapter
from yonga.adapters.hf import HuggingFaceAdapter
from yonga.adapters.openvino_probe import OpenVINOProbe
from yonga.adapters.subprocess import SubprocessRunner
from yonga.adapters.system_linux import LinuxSystemProbe
from yonga.application.jobs import JobContext
from yonga.config.paths import default_paths
from yonga.config.settings import Settings
from yonga.registry import (
    KNOWN_REPAIR_IMPLEMENTATIONS,
    RegistryResolver,
    installed_registry_versions,
    load_builtin_registry,
    load_document,
    load_registry_from_documents,
    validate_registry,
)
from yonga.repairs import (
    RepairContext,
    RepairRuleView,
    apply_repairs,
    get_repair,
    installed_repair_versions,
    known_implementations,
)
from yonga.util.jsonlog import null_logger

REPO_ROOT = Path(__file__).resolve().parents[2]
BROKEN_ARTIFACT = REPO_ROOT / "tests" / "fixtures" / "artifacts" / "tinyllm"
EXAMPLE_REGISTRY = REPO_ROOT / "registry" / "examples" / "qwen3_5.yaml"

#: Every third-party package the CLI must not pull in at import time.
HEAVY_MODULES = (
    "openvino",
    "openvino_genai",
    "openvino_tokenizers",
    "optimum",
    "nncf",
    "transformers",
    "torch",
)

#: Every module an application service will import, and which therefore must
#: stay free of module-scope heavy imports.
LAZY_MODULES = (
    "yonga.cli",
    "yonga.ui",
    "yonga.registry",
    "yonga.repairs",
    "yonga.adapters.exporter_optimum",
    "yonga.adapters.filesystem",
    "yonga.adapters.hf",
    "yonga.adapters.openvino_probe",
    "yonga.adapters.runtime_genai",
    "yonga.adapters.subprocess",
    "yonga.adapters.system_linux",
)


@pytest.fixture(scope="module")
def registry():
    return load_builtin_registry()


@pytest.fixture(scope="module")
def resolver(registry):
    return RegistryResolver(registry, versions=installed_registry_versions())


# --- adapters satisfy the ports the application layer depends on ----------------------


@pytest.mark.parametrize(
    ("factory", "protocol"),
    [
        (SubprocessRunner, CommandRunner),
        (OptimumExporter, Exporter),
        (FilesystemAdapter, FilesystemProbe),
        (LinuxSystemProbe, HardwareProbe),
        (HuggingFaceAdapter, HubClient),
        (OpenVINOProbe, RuntimeProbe),
    ],
)
def test_each_adapter_satisfies_its_port(factory, protocol):
    assert isinstance(factory(), protocol)


def test_the_exporter_defaults_to_the_real_command_runner():
    """The exporter must not ship its own private subprocess handling."""
    exporter = OptimumExporter()
    request_runner = exporter._runner
    assert isinstance(request_runner, SubprocessRunner)
    assert isinstance(request_runner, CommandRunner)


# --- registry data names things that exist in yonga.repairs ---------------------------


def test_the_registry_only_names_implementations_that_exist():
    assert set(KNOWN_REPAIR_IMPLEMENTATIONS) <= set(known_implementations())


def test_every_builtin_repair_rule_resolves_to_an_implementation(registry):
    assert registry.repairs, "the built-in registry ships no repair rules"
    for rule in registry.repairs:
        view = RepairRuleView.from_rule(rule)
        assert view.implementation
        assert get_repair(view.implementation).id


def test_every_issue_repair_reference_resolves(registry):
    repair_ids = {rule.id for rule in registry.repairs}
    referenced = {
        issue.recommendation.repair
        for issue in registry.issues
        if issue.recommendation.repair is not None
    }
    assert referenced, "no issue rule points at a repair"
    assert referenced <= repair_ids


def test_every_architecture_profile_reference_resolves(registry):
    profile_ids = {profile.id for profile in registry.profiles}
    for architecture in registry.architectures:
        for profile_id in architecture.profiles.values():
            assert profile_id in profile_ids


def test_the_builtin_registry_validates_against_the_real_implementations(registry):
    assert validate_registry(registry, known_implementations=known_implementations()) == []


def test_the_documented_example_registry_still_loads():
    """`registry/examples/qwen3_5.yaml` is read as a spec; it must be loadable."""
    document = load_document(EXAMPLE_REGISTRY)
    example = load_registry_from_documents([document])
    assert validate_registry(example, known_implementations=known_implementations()) == []


# --- the registry's repair gate and the repair layer's agree --------------------------


def test_for_job_supplies_the_versions_the_version_gate_needs(tmp_path):
    """Both layers evaluate a rule's `constraints` against the same facts.

    ``repairs_for`` already filtered on the installed OpenVINO version. If the
    context reached ``apply_repairs`` without one, the second gate would read
    "unknown" and drop every version-constrained repair in silence.
    """
    paths = default_paths({"HOME": str(tmp_path), "XDG_DATA_HOME": str(tmp_path / "data")})
    ctx = JobContext(
        job_id="seam",
        logger=null_logger("seam"),
        paths=paths,
        settings=Settings(),
    )
    context = RepairContext.for_job(ctx, tmp_path / "artifact")
    detected = installed_repair_versions()
    assert context.openvino_version == detected["openvino"]
    assert context.tokenizers_version == detected["tokenizers"]


def test_an_explicit_version_still_wins_over_the_detected_one(tmp_path):
    paths = default_paths({"HOME": str(tmp_path), "XDG_DATA_HOME": str(tmp_path / "data")})
    ctx = JobContext(
        job_id="seam",
        logger=null_logger("seam"),
        paths=paths,
        settings=Settings(),
    )
    context = RepairContext.for_job(ctx, tmp_path / "artifact", openvino_version="1.2.3")
    assert context.openvino_version == "1.2.3"


def test_a_rule_the_resolver_selects_is_a_rule_the_repair_layer_runs(resolver, tmp_path):
    """The whole registry -> repairs path, on the real defective artifact."""
    if installed_repair_versions()["openvino"] is None:
        pytest.skip("the built-in repair rule is gated on an installed OpenVINO")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    # The weights come too: the repair's post-condition reads the IR back, and a
    # missing .bin would fail it for a reason that has nothing to do with the seam.
    for name in (
        "openvino_tokenizer.xml",
        "openvino_tokenizer.bin",
        "openvino_detokenizer.xml",
        "openvino_detokenizer.bin",
    ):
        shutil.copy(BROKEN_ARTIFACT / name, artifact / name)

    names = [path.name for path in artifact.iterdir()]
    rules = resolver.repairs_for(
        files=names,
        contents_probe=lambda name, needle: (
            needle in (artifact / name).read_text(encoding="utf-8", errors="ignore")
        ),
    )
    assert [rule.id for rule in rules] == ["ov-tokenizer-precision-string-casing"]

    context = RepairContext(
        artifact_dir=artifact,
        owned_roots=(tmp_path,),
        openvino_version=installed_repair_versions()["openvino"],
        tokenizers_version=installed_repair_versions()["tokenizers"],
    )
    records = apply_repairs(artifact, rules, context)

    assert [record.repair_id for record in records] == ["ov-tokenizer-precision-string-casing"]
    assert records[0].applied is True
    assert records[0].rule_id == "ov-tokenizer-precision-string-casing"
    for name in ("openvino_tokenizer.xml", "openvino_detokenizer.xml"):
        assert 'precision="STRiNG"' not in (artifact / name).read_text(encoding="utf-8")
    # The vendored fixture is the evidence; it must never be written to.
    assert 'precision="STRiNG"' in (BROKEN_ARTIFACT / "openvino_tokenizer.xml").read_text(
        encoding="utf-8"
    )
    # Running again is a no-op rather than a second record.
    assert apply_repairs(artifact, rules, context) == []


# --- nothing reachable from the CLI imports a heavy module ----------------------------


def test_no_module_reachable_from_the_cli_imports_a_heavy_package():
    """Runs in a child so it cannot be weakened by another test's imports."""
    code = (
        "import sys\n"
        f"for name in {LAZY_MODULES!r}:\n"
        "    __import__(name)\n"
        f"heavy = {HEAVY_MODULES!r}\n"
        "print(','.join(sorted(n for n in sys.modules if n.split('.')[0] in heavy)))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == ""
