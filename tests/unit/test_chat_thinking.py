"""Reasoning suppression end to end: which variable is sent, and what an empty answer says.

Two defects are pinned here, and they are two halves of the same user-visible
failure — a chat panel reading "(the model returned no text)" after the model
spent its whole token budget thinking.

*Deciding the thinking variable.* A registry profile does not have to know the
name. Unless it states one outright, the artifact's **own** chat template is read
and a candidate applies only when that template actually references it. That is
evidence rather than a guess, and it is what makes a family nobody has written a
rule for behave correctly. An explicit profile still overrides the decision in
either direction.

*Explaining an empty answer.* An empty answer is three different situations, and
the renderer tells them apart from typed flags rather than by re-reading the
text. The hidden reasoning itself must never appear in any rendered string; the
message reports how much was hidden, never what it said.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_chat_service import build_service, make_manifest, write_artifact
from yonga.adapters.runtime_genai import (
    ChatProfile,
    GenerationResult,
    read_chat_template,
    resolve_starts_in_reasoning,
    resolve_thinking_variable,
    template_references,
    thinking_extra_context,
)
from yonga.application.chat import ChatOptions, ChatProfileView, sanitize_answer
from yonga.domain.models import ModelIdentity, ModelMetadata
from yonga.registry.loader import load_builtin_registry
from yonga.registry.resolver import RegistryResolver
from yonga.registry.schema import ChatProfileRule
from yonga.ui.chat_tui import render_answer

#: The literal a leaked reasoning block would put on screen. No rendered string
#: in this module may contain it.
SECRET = "the-hidden-chain-of-thought-must-never-be-rendered"

#: A Qwen3-shaped template: it guards the reasoning prelude with the variable.
TEMPLATE_WITH_FLAG = (
    "{%- if add_generation_prompt %}\n"
    "    {{- '<|im_start|>assistant\\n' }}\n"
    "    {%- if enable_thinking is defined and enable_thinking is false %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- endif %}\n"
    "{%- endif %}\n"
)

#: The same template with the guard removed: nothing names a thinking variable.
TEMPLATE_WITHOUT_FLAG = (
    "{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"
)


def render(renderable: Any, *, width: int = 100) -> str:
    """Render a renderable to plain text the way a non-TTY terminal would."""
    from rich.console import Console

    buffer = io.StringIO()
    Console(file=buffer, width=width, force_terminal=False, no_color=True).print(renderable)
    return buffer.getvalue()


def artifact_with_template(tmp_path: Path, template: str | None, *, name: str = "art") -> Path:
    """An artifact directory carrying (or deliberately missing) a chat template."""
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    if template is not None:
        (directory / "chat_template.jinja").write_text(template, encoding="utf-8")
    return directory


# --------------------------------------------------------------------------------------
# reading the artifact's own chat template
# --------------------------------------------------------------------------------------


def test_the_chat_template_is_read_from_the_standalone_jinja_file(tmp_path):
    artifact = artifact_with_template(tmp_path, TEMPLATE_WITH_FLAG)
    assert read_chat_template(artifact) == TEMPLATE_WITH_FLAG


def test_the_chat_template_is_read_from_the_tokenizer_config(tmp_path):
    artifact = tmp_path / "art"
    artifact.mkdir()
    (artifact / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": TEMPLATE_WITH_FLAG}), encoding="utf-8"
    )
    assert read_chat_template(artifact) == TEMPLATE_WITH_FLAG


def test_a_list_of_named_templates_is_read_whole(tmp_path):
    """Any of the named templates could be the one that runs, so all are evidence."""
    artifact = tmp_path / "art"
    artifact.mkdir()
    (artifact / "chat_template.json").write_text(
        json.dumps(
            {
                "chat_template": [
                    {"name": "default", "template": TEMPLATE_WITHOUT_FLAG},
                    {"name": "tool_use", "template": TEMPLATE_WITH_FLAG},
                ]
            }
        ),
        encoding="utf-8",
    )
    template = read_chat_template(artifact)
    assert template is not None
    assert template_references(template, "enable_thinking")


def test_a_missing_or_unreadable_template_is_simply_no_evidence(tmp_path):
    assert read_chat_template(tmp_path / "does-not-exist") is None
    artifact = tmp_path / "art"
    artifact.mkdir()
    (artifact / "tokenizer_config.json").write_text("{not json at all", encoding="utf-8")
    assert read_chat_template(artifact) is None


# --------------------------------------------------------------------------------------
# recognising a variable in a template
# --------------------------------------------------------------------------------------


def test_a_variable_is_recognised_only_as_a_whole_identifier():
    assert template_references("{% if enable_thinking %}", "enable_thinking")
    assert not template_references("{% if enable_thinking2 %}", "enable_thinking")
    assert not template_references("{% if not_enable_thinking %}", "enable_thinking")
    assert not template_references(TEMPLATE_WITHOUT_FLAG, "enable_thinking")
    assert not template_references(None, "enable_thinking")
    assert not template_references(TEMPLATE_WITH_FLAG, "   ")


def test_an_attribute_of_another_object_is_not_a_settable_variable():
    """``message.enable_thinking`` is a property of a message, not extra context."""
    assert not template_references("{{ message.enable_thinking }}", "enable_thinking")


# --------------------------------------------------------------------------------------
# the detection path
# --------------------------------------------------------------------------------------


def test_detection_applies_the_candidate_the_template_actually_names():
    profile = ChatProfileView()
    assert profile.supports_thinking_flag is None
    assert resolve_thinking_variable(profile, template=TEMPLATE_WITH_FLAG) == "enable_thinking"
    assert thinking_extra_context(profile, template=TEMPLATE_WITH_FLAG) == {
        "enable_thinking": False
    }


def test_detection_sends_nothing_when_the_template_names_no_candidate():
    profile = ChatProfileView()
    assert resolve_thinking_variable(profile, template=TEMPLATE_WITHOUT_FLAG) is None
    assert thinking_extra_context(profile, template=TEMPLATE_WITHOUT_FLAG) == {}


def test_without_a_template_there_is_no_evidence_and_nothing_is_sent():
    """The safe direction: an unread template is left as its author wrote it."""
    assert resolve_thinking_variable(ChatProfileView(), template=None) is None
    assert resolve_thinking_variable(None, template=TEMPLATE_WITH_FLAG) is None


def test_a_profiles_own_candidate_list_is_honoured():
    profile = ChatProfileView(thinking_variable_candidates=["reasoning_mode", "enable_thinking"])
    template = "{% if reasoning_mode %}{% endif %}"
    assert resolve_thinking_variable(profile, template=template) == "reasoning_mode"
    assert resolve_thinking_variable(profile, template=TEMPLATE_WITH_FLAG) == "enable_thinking"


# --------------------------------------------------------------------------------------
# an explicit registry profile overrides detection, both ways
# --------------------------------------------------------------------------------------


def test_an_explicit_profile_turns_detection_off():
    vetoed = ChatProfileView(supports_thinking_flag=False)
    assert resolve_thinking_variable(vetoed, template=TEMPLATE_WITH_FLAG) is None
    assert thinking_extra_context(vetoed, template=TEMPLATE_WITH_FLAG) == {}

    no_candidates = ChatProfileView(thinking_variable_candidates=[])
    assert resolve_thinking_variable(no_candidates, template=TEMPLATE_WITH_FLAG) is None


def test_an_explicit_profile_turns_the_flag_on_without_asking_the_template():
    declared = ChatProfileView(
        supports_thinking_flag=True,
        thinking_template_variable="enable_thinking",
    )
    assert resolve_thinking_variable(declared, template=TEMPLATE_WITHOUT_FLAG) == "enable_thinking"
    assert resolve_thinking_variable(declared, template=None) == "enable_thinking"


def test_a_named_variable_wins_over_the_default_candidate():
    """Naming a variable is the registry saying which one this family uses."""
    named = ChatProfileView(thinking_template_variable="reasoning_mode")
    assert resolve_thinking_variable(named, template=TEMPLATE_WITH_FLAG) is None
    assert (
        resolve_thinking_variable(named, template="{% if reasoning_mode %}{% endif %}")
        == "reasoning_mode"
    )


def test_the_adapters_own_profile_type_follows_the_same_rules():
    """``ChatProfile`` is the fallback when no registry rule matched; same policy."""
    assert resolve_thinking_variable(ChatProfile(), template=TEMPLATE_WITH_FLAG) == (
        "enable_thinking"
    )
    assert resolve_thinking_variable(ChatProfile(), template=TEMPLATE_WITHOUT_FLAG) is None
    assert (
        resolve_thinking_variable(
            ChatProfile(supports_thinking_flag=False), template=TEMPLATE_WITH_FLAG
        )
        is None
    )


# --------------------------------------------------------------------------------------
# the shipped registry
# --------------------------------------------------------------------------------------


def qwen3_metadata(model_type: str = "qwen3", architecture: str = "Qwen3ForCausalLM"):
    """Metadata for a Qwen3 causal-LM repository."""
    return ModelMetadata(
        identity=ModelIdentity(repo_id="Qwen/Qwen3-0.6B"),
        model_type=model_type,
        architectures=[architecture],
        task="text-generation",
    )


@pytest.mark.parametrize(
    ("model_type", "architecture", "rule_id"),
    [
        ("qwen3", "Qwen3ForCausalLM", "chat-qwen3"),
        ("qwen3_moe", "Qwen3MoeForCausalLM", "chat-qwen3_moe"),
        ("", "Qwen3ForCausalLM", "chat-qwen3"),
        ("", "Qwen3MoeForCausalLM", "chat-qwen3_moe"),
    ],
)
def test_the_qwen3_family_resolves_to_the_qwen3_chat_profile(model_type, architecture, rule_id):
    resolver = RegistryResolver(load_builtin_registry())
    meta = qwen3_metadata(model_type=model_type, architecture=architecture)
    rule = resolver.chat_profile(meta=meta)
    assert rule is not None
    assert rule.id == rule_id


def test_the_qwen3_profile_carries_the_documented_sampling_defaults():
    resolver = RegistryResolver(load_builtin_registry())
    rule = resolver.chat_profile(meta=qwen3_metadata())
    assert rule is not None
    plain = rule.defaults_for(thinking=False)
    assert (plain.temperature, plain.top_p, plain.top_k, plain.max_new_tokens) == (
        0.7,
        0.8,
        20,
        4096,
    )
    thinking = rule.defaults_for(thinking=True)
    assert (thinking.temperature, thinking.top_p, thinking.top_k) == (0.6, 0.95, 20)
    assert rule.reasoning_wrappers == [("<think>", "</think>")]
    assert rule.sources, "a profile that states sampling numbers must cite where they came from"


def test_the_qwen3_profile_leaves_the_thinking_variable_to_the_template():
    """The sampling numbers are a family fact; the variable is the artifact's own."""
    resolver = RegistryResolver(load_builtin_registry())
    rule = resolver.chat_profile(meta=qwen3_metadata())
    assert rule is not None
    assert rule.supports_thinking_flag is None
    assert rule.thinking_candidates == ("enable_thinking",)
    assert resolve_thinking_variable(rule, template=TEMPLATE_WITH_FLAG) == "enable_thinking"
    assert resolve_thinking_variable(rule, template=TEMPLATE_WITHOUT_FLAG) is None


def test_the_shipped_default_profile_still_detects_for_an_unknown_family():
    """The bug was a catch-all profile vetoing detection for everyone it caught."""
    resolver = RegistryResolver(load_builtin_registry())
    meta = ModelMetadata(
        identity=ModelIdentity(repo_id="owner/exotic"), model_type="exotic_net_9000"
    )
    rule = resolver.chat_profile(meta=meta)
    assert rule is not None
    assert rule.id == "chat-default"
    assert resolve_thinking_variable(rule, template=TEMPLATE_WITH_FLAG) == "enable_thinking"


def test_a_declared_flag_still_needs_the_variable_that_drives_it():
    with pytest.raises(ValueError, match="thinking_template_variable"):
        ChatProfileRule(id="chat-broken", supports_thinking_flag=True)


# --------------------------------------------------------------------------------------
# the session applies it
# --------------------------------------------------------------------------------------


def qwen3_manifest(**overrides: Any):
    """A manifest for the prepared Qwen3 artifact this bug was found on."""
    payload: dict[str, Any] = {
        "model_type": "qwen3",
        "architecture": "Qwen3ForCausalLM",
        "recommended_runtime": None,
    }
    payload.update(overrides)
    return make_manifest(**payload)


def test_a_qwen3_session_sends_the_variable_its_template_names(tmp_path):
    service, factory, _runtime = build_service(tmp_path)
    artifact = write_artifact(tmp_path, qwen3_manifest())
    (artifact / "chat_template.jinja").write_text(TEMPLATE_WITH_FLAG, encoding="utf-8")

    session = service.open_artifact(artifact)
    assert session.profile.id == "chat-qwen3"
    assert session.thinking_variable == "enable_thinking"
    assert session.thinking is False
    assert session.generation_settings.temperature == pytest.approx(0.7)

    session.load()
    profile = factory.kwargs[0]["chat_profile"]
    assert thinking_extra_context(
        profile, thinking=factory.kwargs[0]["thinking"], template=TEMPLATE_WITH_FLAG
    ) == {"enable_thinking": False}


def test_a_generic_architecture_rule_does_not_hide_the_family_chat_profile(tmp_path):
    """The exact regression: the manifest named a catch-all architecture rule.

    ``arch-generic-text-generation`` names ``chat-default``, so the family profile
    that matches the recorded ``model_type`` never got a look in and no thinking
    variable was ever sent.
    """
    service, _factory, _runtime = build_service(tmp_path)
    artifact = write_artifact(tmp_path, qwen3_manifest(architecture="arch-generic-text-generation"))
    (artifact / "chat_template.jinja").write_text(TEMPLATE_WITH_FLAG, encoding="utf-8")

    session = service.open_artifact(artifact)
    assert session.profile.id == "chat-qwen3"
    assert session.thinking_variable == "enable_thinking"


def test_a_session_whose_template_names_nothing_sends_nothing(tmp_path):
    service, _factory, _runtime = build_service(tmp_path)
    artifact = write_artifact(tmp_path, qwen3_manifest())
    (artifact / "chat_template.jinja").write_text(TEMPLATE_WITHOUT_FLAG, encoding="utf-8")

    session = service.open_artifact(artifact)
    assert session.profile.id == "chat-qwen3"
    assert session.thinking_variable is None


def test_the_startup_panel_names_the_profile_and_the_thinking_state(tmp_path):
    service, _factory, _runtime = build_service(tmp_path)
    artifact = write_artifact(tmp_path, qwen3_manifest())
    (artifact / "chat_template.jinja").write_text(TEMPLATE_WITH_FLAG, encoding="utf-8")
    extras = service.open_artifact(artifact).describe().extra
    assert extras["Chat profile"] == "chat-qwen3"
    assert extras["Thinking mode"] == "off (enable_thinking=false)"

    thinking = service.open_artifact(artifact, options=ChatOptions(thinking=True))
    assert thinking.describe().extra["Thinking mode"] == "on (enable_thinking=true)"


def test_the_startup_panel_admits_when_no_thinking_variable_exists(tmp_path):
    service, _factory, _runtime = build_service(tmp_path)
    artifact = write_artifact(tmp_path, qwen3_manifest())
    (artifact / "chat_template.jinja").write_text(TEMPLATE_WITHOUT_FLAG, encoding="utf-8")
    extras = service.open_artifact(artifact).describe().extra
    assert "unavailable" in extras["Thinking mode"]


# --------------------------------------------------------------------------------------
# an empty answer explains itself
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Answer:
    """The minimum a renderer needs; deliberately without the reasoning text."""

    display_text: str = ""
    reasoning_hidden: bool = False
    reasoning_unterminated: bool = False
    output_tokens: int | None = None


def test_case_a_the_model_really_did_return_nothing():
    text = render(render_answer(Answer()))
    assert "(the model returned no text)" in text
    assert "reasoning" not in text


def test_case_b_only_reasoning_and_the_block_never_closed():
    text = render(
        render_answer(Answer(reasoning_hidden=True, reasoning_unterminated=True, output_tokens=200))
    )
    assert "(the model returned no text)" not in text
    assert "200 tokens of reasoning" in text
    assert "did not reach an answer" in text
    assert "/tokens 1024" in text
    assert "/reasoning on" in text


def test_case_c_only_reasoning_inside_a_closed_block():
    text = render(
        render_answer(Answer(reasoning_hidden=True, reasoning_unterminated=False, output_tokens=37))
    )
    assert "(the model returned no text)" not in text
    assert "37 tokens of reasoning" in text
    assert "ended its turn without an answer" in text
    assert "output limit" not in text


def test_the_suggested_limit_leaves_room_for_the_answer():
    small = render(render_answer(Answer(reasoning_hidden=True, reasoning_unterminated=True)))
    assert "/tokens 1024" in small
    large = render(
        render_answer(
            Answer(reasoning_hidden=True, reasoning_unterminated=True, output_tokens=4096)
        )
    )
    assert "/tokens 8192" in large


def test_an_answer_that_has_text_is_still_rendered_as_markdown():
    text = render(render_answer(Answer(display_text="**hello**", reasoning_hidden=True)))
    assert "hello" in text
    assert "reasoning" not in text


def test_a_plain_string_caller_still_works():
    assert "(the model returned no text)" in render(render_answer("   "))
    assert "hello" in render(render_answer("hello"))


# --------------------------------------------------------------------------------------
# the security property: shape may be reported, content never
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        f"<think>{SECRET}",
        f"<think>{SECRET}</think>",
        f"<think>{SECRET}</think>   ",
        f"{SECRET}</think>",
        f"<think>{SECRET}</think><think>{SECRET}",
    ],
)
def test_hidden_reasoning_never_reaches_a_rendered_string(raw):
    answer = sanitize_answer(raw, [("<think>", "</think>")], show=False)
    assert SECRET not in answer.display_text
    result = GenerationResult(
        text=raw,
        display_text=answer.display_text,
        reasoning_text=answer.reasoning_text,
        reasoning_hidden=answer.reasoning_hidden,
        reasoning_unterminated=answer.reasoning_unterminated,
        output_tokens=200,
    )
    rendered = render(render_answer(result))
    assert SECRET not in rendered
    assert "200 tokens of reasoning" in rendered


def test_the_generation_result_satisfies_the_renderers_protocol():
    """``GenerationResult`` is what the CLI hands the view; it must fit as-is."""
    result = GenerationResult(
        text="<think>x",
        display_text="",
        reasoning_hidden=True,
        reasoning_unterminated=True,
        output_tokens=60,
    )
    assert "60 tokens of reasoning" in render(render_answer(result))


# --------------------------------------------------------------------------------------
# the variable is also what decides where generation *starts*
# --------------------------------------------------------------------------------------
#
# Pushing the variable false is not only how thinking is switched off; it is the
# only proof that the prompt's reasoning block was closed before generation
# began. Without that proof a template carrying an opener has to be assumed to
# have left one open — see ``test_reasoning_leak.py`` for the full rule.


def test_pushing_the_variable_false_proves_the_block_was_closed():
    profile = ChatProfileView()
    assert thinking_extra_context(profile, template=TEMPLATE_WITH_FLAG) == {
        "enable_thinking": False
    }
    assert resolve_starts_in_reasoning(profile, template=TEMPLATE_WITH_FLAG) is False


def test_asking_for_thinking_means_generation_starts_inside_the_block():
    """With thinking on the template's opener is the last thing in the prompt."""
    profile = ChatProfileView()
    assert resolve_starts_in_reasoning(profile, template=TEMPLATE_WITH_FLAG, thinking=True) is True


def test_a_template_with_no_opener_is_never_assumed_to_have_left_one_open():
    profile = ChatProfileView()
    assert resolve_starts_in_reasoning(profile, template=TEMPLATE_WITHOUT_FLAG) is False
    assert resolve_starts_in_reasoning(profile, template=TEMPLATE_WITHOUT_FLAG, thinking=True) is (
        False
    )


def test_vetoing_the_variable_does_not_prove_the_block_is_closed():
    """``supports_thinking_flag: false`` means nothing is pushed, so nothing is proved."""
    vetoed = ChatProfileView(supports_thinking_flag=False)
    assert thinking_extra_context(vetoed, template=TEMPLATE_WITH_FLAG) == {}
    assert resolve_starts_in_reasoning(vetoed, template=TEMPLATE_WITH_FLAG) is True


def test_a_session_reports_the_starting_state_it_resolved(tmp_path):
    service, _factory, _runtime = build_service(tmp_path)
    artifact = write_artifact(tmp_path, qwen3_manifest())
    (artifact / "chat_template.jinja").write_text(TEMPLATE_WITH_FLAG, encoding="utf-8")

    session = service.open_artifact(artifact)
    assert session.thinking_variable == "enable_thinking"
    assert session.starts_in_reasoning is False

    thinking = service.open_artifact(artifact, options=ChatOptions(thinking=True))
    assert thinking.starts_in_reasoning is True
