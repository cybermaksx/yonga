"""A chat profile's ``tool_calling`` (registry schema 3): its shape, its gate, the data.

Tool calling is registry data in the same way size bands are: the field needs
the schema version that introduced it, a remote bundle using it names the yonga
release that can read it, and a format is chosen from the parsers compiled
into yonga, never described.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from tests.unit.test_registry_families import meta
from tests.unit.test_remote_registry import code_of, signed, trust
from tests.unit.test_ui_tables import render
from yonga.config.settings import Settings
from yonga.domain.diagnostics import RegistryError
from yonga.registry.loader import (
    load_builtin_registry,
    load_registry_from_documents,
    parse_document,
    validate_registry,
)
from yonga.registry.remote import trusted_keys, verify_bundle
from yonga.registry.resolver import RegistryResolver
from yonga.registry.schema import (
    KNOWN_TOOL_CALL_FORMATS,
    SCHEMA_MIN_TOOL_VERSION,
    SCHEMA_VERSION,
    TOOL_CALLING_SCHEMA_VERSION,
    ChatProfileRule,
    SourceKind,
    SourceRef,
    ToolCallingRule,
)
from yonga.ui.tables import registry_explanation

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTED = [SourceRef(kind=SourceKind.TESTED, note="measured")]

DOCUMENT = """\
schema_version: 3
chat:
  - id: chat-demo
    match:
      model_types: [demo]
    tool_calling:
      format: tagged_json
      open: "<tool_call>"
      close: "</tool_call>"
      evidence: tested
      sources:
        - kind: tested
          note: measured
"""


def tool_calling(**overrides: object) -> ToolCallingRule:
    fields: dict[str, object] = {
        "format": "tagged_json",
        "open": "<tool_call>",
        "close": "</tool_call>",
        "evidence": "tested",
        "sources": TESTED,
    }
    fields.update(overrides)
    return ToolCallingRule.model_validate(fields)


# --- the schema -------------------------------------------------------------------------


def test_schema_three_introduces_tool_calling_and_names_a_tool_floor() -> None:
    assert SCHEMA_VERSION >= TOOL_CALLING_SCHEMA_VERSION == 3
    assert SCHEMA_MIN_TOOL_VERSION[3] == "0.4.0"
    assert {"tagged_json", "json_list"} <= KNOWN_TOOL_CALL_FORMATS


@pytest.mark.parametrize("evidence", ["tested", "official"])
def test_a_measured_entry_is_accepted(evidence: str) -> None:
    assert tool_calling(evidence=evidence).evidence.value == evidence


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"evidence": "inferred"}, "measured format"),
        ({"evidence": "unknown"}, "measured format"),
        ({"sources": []}, "at least 1"),
        ({"close": None}, "needs both open and close"),
        ({"open": ""}, "needs both open and close"),
        ({"format": "  "}, "must not be empty"),
        ({"name_field": ""}, "must not be empty"),
        ({"surprise": True}, "Extra inputs"),
    ],
)
def test_an_unmeasured_or_incomplete_entry_is_refused(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        tool_calling(**overrides)


def test_a_json_list_needs_no_markers() -> None:
    rule = tool_calling(format="json_list", open=None, close=None)
    assert rule.describe() == "json_list [tested]"


def test_describe_names_the_format_the_markers_and_the_template_check() -> None:
    text = tool_calling(template_contains="<tool_call>").describe()
    assert text == (
        "tagged_json (<tool_call> and </tool_call>) [tested] "
        "when the chat template contains '<tool_call>'"
    )


def test_tool_calling_needs_schema_three_on_the_rule() -> None:
    with pytest.raises(ValidationError, match="tool_calling needs schema_version 3"):
        ChatProfileRule(id="chat-demo", schema_version=2, tool_calling=tool_calling())


def test_a_rule_without_its_own_version_takes_the_document_version() -> None:
    old = DOCUMENT.replace("schema_version: 3", "schema_version: 2")
    with pytest.raises(RegistryError) as excinfo:
        parse_document(old, origin="old.yaml")
    assert excinfo.value.diagnostic.code == "registry.schema_violation"
    assert "tool_calling needs schema_version 3" in excinfo.value.diagnostic.message


def test_a_schema_three_document_loads_and_validates() -> None:
    document = parse_document(DOCUMENT, origin="demo.yaml")
    (rule,) = document.chat

    assert rule.schema_version == 3
    assert rule.tool_calling is not None and rule.tool_calling.format == "tagged_json"
    assert validate_registry(load_registry_from_documents([document])) == []


def test_an_unknown_format_loads_but_fails_validation() -> None:
    """A newer layer may name a parser this build lacks; that is reported, not a crash."""
    text = DOCUMENT.replace("format: tagged_json", "format: pythonic_list")
    registry = load_registry_from_documents([parse_document(text, origin="demo.yaml")])

    problems = validate_registry(registry)
    assert any("unknown tool-call format 'pythonic_list'" in item for item in problems)


# --- remote bundles ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def bundle_tool() -> ModuleType:
    path = REPO_ROOT / "tools" / "registry_bundle.py"
    spec = importlib.util.spec_from_file_location("registry_bundle_tool_v3", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_bundle_with_tool_calling_needs_the_yonga_that_reads_it(bundle_tool: ModuleType) -> None:
    files = {"chat.yaml": DOCUMENT}

    assert bundle_tool.schema_versions(files) == {3}
    assert bundle_tool.required_tool_version(files, None) == "0.4.0"
    with pytest.raises(SystemExit, match=r"below 0\.4\.0"):
        bundle_tool.required_tool_version(files, "0.3.0")


def test_a_signed_bundle_cannot_name_a_parser_yonga_lacks(tmp_path: Path) -> None:
    """The YAML may choose a built-in parser; it can never bring one of its own."""
    settings = Settings(
        config_dir=tmp_path / "config", cache_dir=tmp_path / "cache", data_dir=tmp_path / "data"
    )
    key = Ed25519PrivateKey.generate()
    trust(settings.paths, key)
    foreign = DOCUMENT.replace("format: tagged_json", "format: run_this_parser")
    payload, signature = signed(key, files={"chat.yaml": foreign})

    with pytest.raises(RegistryError) as caught:
        verify_bundle(payload, signature, trusted_keys(settings.paths), tool_version="0.4.0")
    assert code_of(caught) == "registry.remote.unknown_tool_call_format"


# --- the built-in data ------------------------------------------------------------------

RESOLVER = RegistryResolver(load_builtin_registry(), versions={})

#: Families the scoping study saw producing parseable calls on the reference host.
MEASURED = [
    ("qwen3", "Qwen3ForCausalLM", "chat-qwen3", "tagged_json"),
    ("qwen2", "Qwen2ForCausalLM", "chat-qwen2", "tagged_json"),
    ("smollm3", "SmolLM3ForCausalLM", "chat-smollm3", "tagged_json"),
    ("granite", "GraniteForCausalLM", "chat-granite", "json_list"),
]

#: Families measured and left without tool calling, and one never measured.
UNDECLARED = [
    ("llama", "LlamaForCausalLM"),
    ("lfm2", "Lfm2ForCausalLM"),
    ("gemma3_text", "Gemma3ForCausalLM"),
    ("gemma2", "Gemma2ForCausalLM"),
    ("olmo2", "Olmo2ForCausalLM"),
    ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
    ("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),
    ("qwen3_moe", "Qwen3MoeForCausalLM"),
]


def chat_rule(model_type: str, architecture: str) -> ChatProfileRule | None:
    architecture_rule, _ = RESOLVER.resolve_architecture(meta(model_type, architecture))
    return RESOLVER.chat_profile(arch_rule=architecture_rule, meta=meta(model_type, architecture))


@pytest.mark.parametrize(("model_type", "architecture", "rule_id", "format"), MEASURED)
def test_measured_families_declare_their_tool_call_format(
    model_type: str, architecture: str, rule_id: str, format: str
) -> None:
    rule = chat_rule(model_type, architecture)

    assert rule is not None and rule.id == rule_id
    assert rule.tool_calling is not None
    assert rule.tool_calling.format == format
    assert rule.tool_calling.evidence.value == "tested"
    assert rule.tool_calling.template_contains, "checked against the artifact's own template"
    notes = [source.note or "" for source in rule.tool_calling.sources]
    assert all(source.kind is SourceKind.TESTED for source in rule.tool_calling.sources)
    assert any("2026-10-09" in note for note in notes)


@pytest.mark.parametrize(("model_type", "architecture"), UNDECLARED)
def test_other_families_take_no_tools(model_type: str, architecture: str) -> None:
    rule = chat_rule(model_type, architecture)

    assert rule is not None
    assert rule.tool_calling is None


@pytest.mark.parametrize("rule_id", ["chat-qwen2", "chat-smollm3", "chat-granite"])
def test_the_new_family_rules_change_nothing_but_tool_calling(rule_id: str) -> None:
    """They replace chat-default for their family; chat must behave as it did."""
    registry = load_builtin_registry()
    rule, default = registry.chat_profile(rule_id), registry.chat_profile("chat-default")
    assert rule is not None and default is not None
    policy = {
        "supports_thinking_flag",
        "thinking_template_variable",
        "thinking_variable_candidates",
        "default_thinking",
        "reasoning_wrappers",
        "defaults",
        "thinking_defaults",
        "stop_strings",
    }
    assert rule.model_dump(include=policy) == default.model_dump(include=policy)


def test_qwen3_moe_keeps_the_qwen3_policy_without_tool_calling() -> None:
    """Split from chat-qwen3 only because the MoE models were not measured calling tools."""
    registry = load_builtin_registry()
    moe, dense = registry.chat_profile("chat-qwen3_moe"), registry.chat_profile("chat-qwen3")
    assert moe is not None and dense is not None
    everything_but_identity = {
        "id",
        "summary",
        "match",
        "sources",
        "tool_calling",
        "schema_version",
    }
    assert moe.model_dump(exclude=everything_but_identity) == dense.model_dump(
        exclude=everything_but_identity
    )
    assert moe.tool_calling is None and dense.tool_calling is not None


def test_registry_explain_shows_the_tool_call_format() -> None:
    declared = RESOLVER.explain(meta=meta("granite", "GraniteForCausalLM"))
    undeclared = RESOLVER.explain(meta=meta("llama", "LlamaForCausalLM"))

    assert declared.chat_tool_calling is not None
    assert declared.chat_tool_calling.startswith("json_list (<|tool_call|>) [tested]")
    assert undeclared.chat_tool_calling is None
    shown = render(registry_explanation(declared), width=200)
    assert "Tool calling  json_list (<|tool_call|>) [tested]" in shown
    refused = render(registry_explanation(undeclared), width=200)
    assert "Tool calling  not declared; serve refuses tools" in refused


def test_the_builtin_chat_document_is_schema_three() -> None:
    path = REPO_ROOT / "src" / "yonga" / "registry" / "builtin" / "chat.yaml"
    document = parse_document(path.read_text(encoding="utf-8"), origin="chat.yaml")
    assert document.schema_version == 3
    using = {rule.id for rule in document.chat if rule.tool_calling is not None}
    assert using == {"chat-qwen3", "chat-qwen2", "chat-smollm3", "chat-granite"}
    assert all(rule.schema_version == 3 for rule in document.chat if rule.id in using)
