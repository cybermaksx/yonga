"""Tests for the repair framework and the tokenizer port-precision repair.

The fixtures under ``tests/fixtures`` are authentic OpenVINO 2026.4 output captured on
a real Intel host: ``tokenizer_string_precision_before.xml`` is what the serialiser
writes (2 x ``precision="STRiNG"``, 1 x ``element_type="string"``), and
``tokenizer_string_precision_after.xml`` is the byte-for-byte expected result. Any
change to the repair that rewrites a single extra byte fails
:func:`test_repaired_bytes_match_the_after_fixture`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from yonga.domain.artifacts import MANIFEST_DIRNAME, RepairRecord
from yonga.domain.diagnostics import RepairError
from yonga.repairs import (
    REPAIR_ID,
    REPAIR_IMPLEMENTATIONS,
    RepairContext,
    RepairMatch,
    RepairRuleView,
    RepairSafety,
    TokenizerStringPrecisionRepair,
    apply_repairs,
    count_bad_precision,
    get_repair,
    known_implementations,
)
from yonga.util.hashing import sha256_bytes

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
BEFORE_XML = FIXTURES / "tokenizer_string_precision_before.xml"
AFTER_XML = FIXTURES / "tokenizer_string_precision_after.xml"
COMPANION_BIN = FIXTURES / "tokenizer_string_precision.bin"

TOKENIZER = "openvino_tokenizer.xml"
DETOKENIZER = "openvino_detokenizer.xml"
BACKUP_RELATIVE = f"{MANIFEST_DIRNAME}/repairs/{REPAIR_ID}"

REGISTRY_RULE: dict[str, object] = {
    "id": "tokenizer-string-precision-casing",
    "status": "active",
    "summary": "OpenVINO tokenizer XML contains invalid mixed-case legacy precision",
    "match": {
        "file_names": [TOKENIZER, DETOKENIZER],
        "contains": 'precision="STRiNG"',
        "openvino": ">=2026.4,<2027",
    },
    "repair": {"implementation": "tokenizer_string_precision", "safety": "automatic"},
    "validation": {"requires_tokenizer_read": True},
    "postconditions": ["no_precision_str_ing", "tokenizer_readable"],
}


# --- helpers ---------------------------------------------------------------------------


def make_artifact(
    tmp_path: Path, *, tokenizer: Path = BEFORE_XML, detokenizer: bool = True
) -> Path:
    """Create a tool-owned artifact directory holding tokenizer IR files."""
    artifact = tmp_path / "artifacts" / "job" / "model"
    artifact.mkdir(parents=True)
    (artifact / TOKENIZER).write_bytes(tokenizer.read_bytes())
    (artifact / "openvino_tokenizer.bin").write_bytes(COMPANION_BIN.read_bytes())
    if detokenizer:
        (artifact / DETOKENIZER).write_bytes(tokenizer.read_bytes())
        (artifact / "openvino_detokenizer.bin").write_bytes(COMPANION_BIN.read_bytes())
    return artifact


def make_context(artifact: Path, **kwargs: object) -> RepairContext:
    """Build a context whose only tool-owned root is ``artifact``."""
    defaults: dict[str, object] = {
        "artifact_dir": artifact,
        "openvino_version": "2026.4.0",
        "tokenizers_version": "2026.4.0.0",
        "env": {},
    }
    defaults.update(kwargs)
    return RepairContext(**defaults)  # type: ignore[arg-type]


def backup_dir(artifact: Path) -> Path:
    """Where the repair stores original bytes."""
    return artifact / MANIFEST_DIRNAME / "repairs" / REPAIR_ID


# --- the fixtures themselves ------------------------------------------------------------


def test_before_fixture_has_the_exact_expected_shape() -> None:
    text = BEFORE_XML.read_bytes().decode("utf-8")
    assert count_bad_precision(text) == 2
    assert text.count('element_type="string"') == 1
    assert AFTER_XML.read_bytes().decode("utf-8").count('element_type="string"') == 1
    assert count_bad_precision(AFTER_XML.read_bytes().decode("utf-8")) == 0


# --- detection -------------------------------------------------------------------------


def test_detect_finds_both_tokenizer_files(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)
    match = TokenizerStringPrecisionRepair().detect(artifact, make_context(artifact))
    assert match is not None
    assert match.repair_id == REPAIR_ID
    assert set(match.names) == {TOKENIZER, DETOKENIZER}
    assert match.counts == {TOKENIZER: 2, DETOKENIZER: 2}
    assert match.total_occurrences == 4
    assert match.details["element_type_string_counts"] == {TOKENIZER: 1, DETOKENIZER: 1}


def test_detect_returns_none_for_an_already_repaired_artifact(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path, tokenizer=AFTER_XML)
    assert TokenizerStringPrecisionRepair().detect(artifact, make_context(artifact)) is None


def test_detect_ignores_files_outside_the_openvino_xml_scope(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path, tokenizer=AFTER_XML, detokenizer=False)
    (artifact / "tokenizer.json").write_bytes(BEFORE_XML.read_bytes())
    rule = RepairRuleView.from_rule(
        {
            "id": "r",
            "match": {"files": ["tokenizer.json", "../escape.xml", TOKENIZER]},
            "repair": {"implementation": "tokenizer_string_precision"},
        }
    )
    repair = TokenizerStringPrecisionRepair()
    context = make_context(artifact, rule=rule)
    assert repair.candidate_names(context) == (TOKENIZER,)
    assert repair.detect(artifact, context) is None
    assert (artifact / "tokenizer.json").read_bytes() == BEFORE_XML.read_bytes()


def test_detect_rejects_a_non_ir_xml(tmp_path: Path) -> None:
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / TOKENIZER).write_text(
        '<?xml version="1.0"?>\n<config>\n  <port precision="STRiNG" />\n</config>\n',
        encoding="utf-8",
    )
    context = make_context(artifact)
    assert TokenizerStringPrecisionRepair().detect(artifact, context) is None


def test_detect_rejects_a_net_without_layers(tmp_path: Path) -> None:
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / TOKENIZER).write_text(
        '<?xml version="1.0"?>\n<net name="x" version="11">\n'
        '  <port precision="STRiNG" />\n</net>\n',
        encoding="utf-8",
    )
    assert TokenizerStringPrecisionRepair().detect(artifact, make_context(artifact)) is None


def test_detect_rejects_a_document_whose_pattern_hides_in_a_comment(tmp_path: Path) -> None:
    """A commented-out occurrence makes textual and structural counts disagree."""
    artifact = tmp_path / "model"
    artifact.mkdir()
    original = BEFORE_XML.read_bytes().decode("utf-8")
    poisoned = original.replace(
        "\t<layers>",
        '\t<!-- historical note: precision="STRiNG" used to be emitted here -->\n\t<layers>',
        1,
    )
    (artifact / TOKENIZER).write_text(poisoned, encoding="utf-8")
    context = make_context(artifact)
    assert count_bad_precision(poisoned) == 3
    assert TokenizerStringPrecisionRepair().detect(artifact, context) is None
    assert (artifact / TOKENIZER).read_text(encoding="utf-8") == poisoned


def test_a_layer_named_string_is_not_rewritten(tmp_path: Path) -> None:
    """``STRiNG`` inside a layer name never matches: the pattern is attribute-anchored."""
    artifact = tmp_path / "model"
    artifact.mkdir()
    document = (
        '<?xml version="1.0"?>\n'
        '<net name="STRiNG_net" version="11">\n'
        "\t<layers>\n"
        '\t\t<layer id="0" name="STRiNG_input" type="Parameter" version="opset1">\n'
        '\t\t\t<data shape="?" element_type="string" />\n'
        "\t\t\t<output>\n"
        '\t\t\t\t<port id="0" precision="STRING" names="STRiNG" />\n'
        "\t\t\t</output>\n"
        "\t\t</layer>\n"
        "\t</layers>\n"
        "</net>\n"
    )
    (artifact / TOKENIZER).write_text(document, encoding="utf-8")
    assert count_bad_precision(document) == 0
    assert TokenizerStringPrecisionRepair().detect(artifact, make_context(artifact)) is None
    assert (artifact / TOKENIZER).read_text(encoding="utf-8") == document


# --- application -----------------------------------------------------------------------


def test_apply_changes_only_the_port_precision(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)
    repair = TokenizerStringPrecisionRepair()
    context = make_context(artifact)
    match = repair.detect(artifact, context)
    assert match is not None

    before_text = (artifact / TOKENIZER).read_bytes().decode("utf-8")
    result = repair.apply(match, artifact, context)
    after_text = (artifact / TOKENIZER).read_bytes().decode("utf-8")

    assert result.applied is True
    assert count_bad_precision(before_text) == 2
    assert count_bad_precision(after_text) == 0
    assert after_text.count('precision="STRING"') == 2
    # The lower-case element type is upstream's and must survive untouched.
    assert before_text.count('element_type="string"') == 1
    assert after_text.count('element_type="string"') == 1
    assert 'element_type="STRING"' not in after_text
    assert len(after_text) == len(before_text)


def test_repaired_bytes_match_the_after_fixture(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)
    repair = TokenizerStringPrecisionRepair()
    context = make_context(artifact)
    match = repair.detect(artifact, context)
    assert match is not None
    repair.apply(match, artifact, context)

    expected = AFTER_XML.read_bytes()
    assert (artifact / TOKENIZER).read_bytes() == expected
    assert (artifact / DETOKENIZER).read_bytes() == expected


def test_backup_and_hashes_are_recorded(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path, detokenizer=False)
    repair = TokenizerStringPrecisionRepair()
    context = make_context(artifact)
    match = repair.detect(artifact, context)
    assert match is not None
    result = repair.apply(match, artifact, context)
    record = result.to_record()

    assert isinstance(record, RepairRecord)
    assert record.repair_id == REPAIR_ID
    assert record.applied is True
    assert record.safety == RepairSafety.AUTOMATIC.value
    assert record.files == [TOKENIZER]
    assert record.backup_dir == BACKUP_RELATIVE
    assert record.applied_at is not None

    assert record.before_hashes[TOKENIZER] == sha256_bytes(BEFORE_XML.read_bytes())
    assert record.after_hashes[TOKENIZER] == sha256_bytes(AFTER_XML.read_bytes())
    assert record.before_hashes[TOKENIZER] != record.after_hashes[TOKENIZER]

    backup = backup_dir(artifact) / TOKENIZER
    assert backup.read_bytes() == BEFORE_XML.read_bytes()
    assert record.detail["occurrences"] == 2
    assert record.detail["changed_files"] == [TOKENIZER]


def test_no_match_means_no_change_and_no_backup(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path, tokenizer=AFTER_XML)
    original = (artifact / TOKENIZER).read_bytes()
    records = apply_repairs(artifact, [REGISTRY_RULE], make_context(artifact))

    assert records == []
    assert (artifact / TOKENIZER).read_bytes() == original
    assert not (artifact / MANIFEST_DIRNAME).exists()


def test_a_stale_match_applied_twice_is_a_no_op(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path, detokenizer=False)
    repair = TokenizerStringPrecisionRepair()
    context = make_context(artifact)
    match = repair.detect(artifact, context)
    assert match is not None

    first = repair.apply(match, artifact, context)
    repaired_bytes = (artifact / TOKENIZER).read_bytes()
    backup_bytes = (backup_dir(artifact) / TOKENIZER).read_bytes()

    # Re-detection finds nothing: preparation can be re-run safely.
    assert repair.detect(artifact, context) is None

    # Replaying the stale match writes nothing and reports no change.
    second = repair.apply(match, artifact, context)
    assert first.applied is True
    assert second.applied is False
    assert second.detail["no_op"] is True
    assert second.backup_dir is None
    assert second.before_hashes[TOKENIZER] == second.after_hashes[TOKENIZER]
    assert (artifact / TOKENIZER).read_bytes() == repaired_bytes
    assert (backup_dir(artifact) / TOKENIZER).read_bytes() == backup_bytes


def test_dry_run_previews_without_writing(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path, detokenizer=False)
    original = (artifact / TOKENIZER).read_bytes()
    records = apply_repairs(artifact, [REGISTRY_RULE], make_context(artifact, dry_run=True))

    assert len(records) == 1
    record = records[0]
    assert record.applied is False
    assert record.detail["dry_run"] is True
    assert record.before_hashes[TOKENIZER] == sha256_bytes(BEFORE_XML.read_bytes())
    assert record.after_hashes[TOKENIZER] == sha256_bytes(AFTER_XML.read_bytes())
    assert (artifact / TOKENIZER).read_bytes() == original
    assert not (artifact / MANIFEST_DIRNAME).exists()


# --- tool-ownership boundary --------------------------------------------------------


def test_a_path_outside_the_tool_owned_root_is_refused(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)
    elsewhere = tmp_path / "somewhere-else"
    elsewhere.mkdir()
    (elsewhere / TOKENIZER).write_bytes(BEFORE_XML.read_bytes())

    context = make_context(artifact)
    with pytest.raises(RepairError) as excinfo:
        TokenizerStringPrecisionRepair().detect(elsewhere, context)
    assert excinfo.value.diagnostic.code == "repair.path_not_tool_owned"
    assert (elsewhere / TOKENIZER).read_bytes() == BEFORE_XML.read_bytes()


def test_the_hugging_face_cache_is_refused_even_when_declared_owned(tmp_path: Path) -> None:
    cache = tmp_path / "hf"
    artifact = cache / "hub" / "models--org--repo" / "snapshots" / "abc"
    artifact.mkdir(parents=True)
    (artifact / TOKENIZER).write_bytes(BEFORE_XML.read_bytes())

    context = RepairContext(artifact_dir=artifact, env={"HF_HOME": str(cache)})
    with pytest.raises(RepairError) as excinfo:
        TokenizerStringPrecisionRepair().detect(artifact, context)
    assert excinfo.value.diagnostic.code == "repair.path_in_hf_cache"
    assert (artifact / TOKENIZER).read_bytes() == BEFORE_XML.read_bytes()


# --- implementation registry ---------------------------------------------------------


def test_get_repair_accepts_both_the_id_and_the_implementation_name() -> None:
    by_id = get_repair(REPAIR_ID)
    by_name = get_repair("tokenizer_string_precision")
    assert isinstance(by_id, TokenizerStringPrecisionRepair)
    assert isinstance(by_name, TokenizerStringPrecisionRepair)
    assert by_id.safety is RepairSafety.AUTOMATIC


def test_get_repair_rejects_an_unknown_name() -> None:
    with pytest.raises(RepairError) as excinfo:
        get_repair("nope")
    diagnostic = excinfo.value.diagnostic
    assert diagnostic.code == "repair.unknown_implementation"
    assert "nope" in diagnostic.message
    assert any("tokenizer_string_precision" in line for line in diagnostic.remediation)


def test_known_implementations_covers_the_registry_mapping() -> None:
    names = known_implementations()
    assert {REPAIR_ID, "tokenizer_string_precision"} <= names
    assert REPAIR_IMPLEMENTATIONS[REPAIR_ID] is TokenizerStringPrecisionRepair
    assert REPAIR_IMPLEMENTATIONS["tokenizer_string_precision"] is TokenizerStringPrecisionRepair


def test_a_rule_without_an_implementation_is_a_typed_error() -> None:
    with pytest.raises(RepairError) as excinfo:
        RepairRuleView.from_rule({"id": "r", "match": {"files": [TOKENIZER]}})
    assert excinfo.value.diagnostic.code == "repair.rule_missing_implementation"


def test_rule_view_accepts_both_registry_spellings() -> None:
    spec_style = RepairRuleView.from_rule(
        {
            "id": "a",
            "match": {"files": [TOKENIZER], "openvino": ">=2026.4,<2027"},
            "repair": {"implementation": "tokenizer_string_precision", "safety": "automatic"},
            "validation": {"requires_tokenizer_read": True},
        }
    )
    example_style = RepairRuleView.from_rule(REGISTRY_RULE)
    assert spec_style.files == (TOKENIZER,)
    assert spec_style.safety is RepairSafety.AUTOMATIC
    assert spec_style.requires_tokenizer_read is True
    assert spec_style.openvino == ">=2026.4,<2027"
    assert example_style.files == (TOKENIZER, DETOKENIZER)
    assert example_style.id == "tokenizer-string-precision-casing"
    assert RepairRuleView.from_rule(example_style) is example_style


def test_rule_view_reads_the_registry_constraints_block() -> None:
    """``BaseRule.constraints`` is where the registry schema keeps version gates."""
    view = RepairRuleView.from_rule(
        {
            "id": "a",
            "constraints": {"openvino": ">=2026.4,<2027"},
            "repair": {"implementation": "tokenizer_string_precision", "safety": "automatic"},
        }
    )
    assert view.openvino == ">=2026.4,<2027"
    assert view.version_matches(openvino="2026.4.0", tokenizers=None) is True
    assert view.version_matches(openvino="2025.3.0", tokenizers=None) is False
    # A gated rule with no known installed version must not run.
    assert view.version_matches(openvino=None, tokenizers=None) is False


def test_rule_status_uses_a_deny_list() -> None:
    def status_of(value: str) -> bool:
        return RepairRuleView.from_rule(
            {
                "id": "a",
                "status": value,
                "repair": {"implementation": "tokenizer_string_precision"},
            }
        ).active

    assert status_of("active") is True
    assert status_of("experimental") is True
    assert status_of("deprecated") is False
    assert status_of("withdrawn") is False


def test_the_registry_spelling_of_manual_is_understood() -> None:
    """``yonga.registry.schema.RepairSafety`` spells the strictest level ``manual``."""
    view = RepairRuleView.from_rule(
        {"id": "a", "repair": {"implementation": "tokenizer_string_precision", "safety": "manual"}}
    )
    assert view.safety is RepairSafety.MANUAL_ONLY
    assert view.effective_safety(RepairSafety.AUTOMATIC) is RepairSafety.MANUAL_ONLY


def test_a_real_registry_repair_rule_is_understood() -> None:
    """The seam with the registry package, exercised against its actual schema."""
    schema = pytest.importorskip("yonga.registry.schema")
    rule = schema.RepairRule.model_validate(
        {
            "id": REPAIR_ID,
            "status": "active",
            "summary": "OpenVINO tokenizer XML contains invalid mixed-case legacy precision",
            "constraints": {"openvino": ">=2026.4,<2027"},
            "match": {"file_names": [TOKENIZER, DETOKENIZER], "contains": 'precision="STRiNG"'},
            "repair": {"implementation": "tokenizer_string_precision", "safety": "automatic"},
            "validation": {"requires_tokenizer_read": True},
        }
    )
    view = RepairRuleView.from_rule(rule)
    assert view.id == REPAIR_ID
    assert view.implementation == "tokenizer_string_precision"
    assert view.safety is RepairSafety.AUTOMATIC
    assert view.active is True
    assert view.files == (TOKENIZER, DETOKENIZER)
    assert view.openvino == ">=2026.4,<2027"
    assert view.requires_tokenizer_read is True


def test_unknown_safety_degrades_to_manual_only() -> None:
    view = RepairRuleView.from_rule(
        {
            "id": "a",
            "repair": {"implementation": "tokenizer_string_precision", "safety": "yolo"},
        }
    )
    assert view.safety is RepairSafety.MANUAL_ONLY


# --- orchestration -------------------------------------------------------------------


def test_apply_repairs_runs_the_whole_pipeline(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)
    records = apply_repairs(artifact, [REGISTRY_RULE], make_context(artifact))

    assert len(records) == 1
    record = records[0]
    assert record.applied is True
    assert record.repair_id == REPAIR_ID
    assert record.rule_id == "tokenizer-string-precision-casing"
    assert sorted(record.files) == sorted([DETOKENIZER, TOKENIZER])
    assert record.backup_dir == BACKUP_RELATIVE
    assert record.detail["runtime_validation"] in {"ok", "skipped"}
    assert (artifact / TOKENIZER).read_bytes() == AFTER_XML.read_bytes()
    assert (artifact / DETOKENIZER).read_bytes() == AFTER_XML.read_bytes()


def test_apply_repairs_skips_a_version_gated_rule(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)
    original = (artifact / TOKENIZER).read_bytes()
    records = apply_repairs(
        artifact, [REGISTRY_RULE], make_context(artifact, openvino_version="2025.3.0")
    )
    assert records == []
    assert (artifact / TOKENIZER).read_bytes() == original


def test_apply_repairs_skips_an_inactive_rule(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)
    rule = dict(REGISTRY_RULE, status="withdrawn")
    assert apply_repairs(artifact, [rule], make_context(artifact)) == []
    assert (artifact / TOKENIZER).read_bytes() == BEFORE_XML.read_bytes()


def test_apply_repairs_honours_the_confirmation_policy(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path, detokenizer=False)
    rule = dict(REGISTRY_RULE)
    rule["repair"] = {"implementation": "tokenizer_string_precision", "safety": "confirm"}

    denied = apply_repairs(artifact, [rule], make_context(artifact))
    assert len(denied) == 1
    assert denied[0].applied is False
    assert "confirmation" in str(denied[0].detail["reason"])
    assert (artifact / TOKENIZER).read_bytes() == BEFORE_XML.read_bytes()

    seen: list[RepairMatch] = []

    def approve(repair: object, match: RepairMatch) -> bool:
        assert isinstance(repair, TokenizerStringPrecisionRepair)
        seen.append(match)
        return True

    approved = apply_repairs(artifact, [rule], make_context(artifact), confirm=approve)
    assert len(seen) == 1
    assert approved[0].applied is True
    assert approved[0].safety == RepairSafety.CONFIRM.value
    assert (artifact / TOKENIZER).read_bytes() == AFTER_XML.read_bytes()


def test_apply_repairs_never_auto_applies_a_manual_only_rule(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path, detokenizer=False)
    rule = dict(REGISTRY_RULE)
    rule["repair"] = {"implementation": "tokenizer_string_precision", "safety": "manual_only"}
    records = apply_repairs(artifact, [rule], make_context(artifact))
    assert len(records) == 1
    assert records[0].applied is False
    assert "manual_only" in str(records[0].detail["reason"])
    assert (artifact / TOKENIZER).read_bytes() == BEFORE_XML.read_bytes()


def test_apply_repairs_rejects_an_unknown_implementation(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)
    rule = dict(REGISTRY_RULE)
    rule["repair"] = {"implementation": "nope", "safety": "automatic"}
    with pytest.raises(RepairError) as excinfo:
        apply_repairs(artifact, [rule], make_context(artifact))
    assert excinfo.value.diagnostic.code == "repair.unknown_implementation"


# --- validation ------------------------------------------------------------------------


def test_validation_is_skipped_but_not_failed_without_openvino(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "openvino", None)
    artifact = make_artifact(tmp_path, detokenizer=False)
    records = apply_repairs(artifact, [REGISTRY_RULE], make_context(artifact))

    assert len(records) == 1
    record = records[0]
    assert record.applied is True
    assert record.detail["runtime_validation"] == "skipped"
    assert "openvino" in str(record.detail["runtime_validation_reason"])
    assert record.detail["postcondition_checked_files"] == [TOKENIZER]


def test_validation_fails_loudly_when_a_postcondition_is_violated(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path, detokenizer=False)
    repair = TokenizerStringPrecisionRepair()
    context = make_context(artifact)
    match = repair.detect(artifact, context)
    assert match is not None
    result = repair.apply(match, artifact, context)

    # Simulate an artifact that regressed between apply and validate.
    (artifact / TOKENIZER).write_bytes(BEFORE_XML.read_bytes())
    with pytest.raises(RepairError) as excinfo:
        repair.validate(result, context)
    assert excinfo.value.diagnostic.code == "repair.postcondition_failed"


@pytest.mark.openvino
def test_repaired_fixture_is_readable_and_cpu_compilable() -> None:
    """The defect and its repair, proven against the installed OpenVINO runtime."""
    openvino = pytest.importorskip("openvino")
    core = openvino.Core()

    with pytest.raises(RuntimeError, match="STRiNG"):
        core.read_model(BEFORE_XML)

    model = core.read_model(AFTER_XML)
    compiled = core.compile_model(model, "CPU")
    assert compiled is not None


@pytest.mark.openvino
def test_repair_makes_a_real_artifact_loadable(tmp_path: Path) -> None:
    """End to end: a broken artifact goes in, a CPU-compilable one comes out."""
    openvino = pytest.importorskip("openvino")
    artifact = make_artifact(tmp_path)
    core = openvino.Core()
    with pytest.raises(RuntimeError, match="STRiNG"):
        core.read_model(artifact / TOKENIZER)

    records = apply_repairs(artifact, [REGISTRY_RULE], make_context(artifact))
    assert records[0].applied is True
    assert records[0].detail["runtime_validation"] == "ok"
    compiled = records[0].detail["cpu_compile_ok"]
    assert isinstance(compiled, list)
    assert sorted(str(name) for name in compiled) == sorted([DETOKENIZER, TOKENIZER])

    for name in (TOKENIZER, DETOKENIZER):
        model = core.read_model(artifact / name)
        assert core.compile_model(model, "CPU") is not None
