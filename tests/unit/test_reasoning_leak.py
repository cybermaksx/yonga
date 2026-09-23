"""The prompt-opened reasoning block: the leak the sanitiser could not see.

``sanitize_answer`` recognises a reasoning block from an opening token in the
**model's own** output. A chat template may open the block in the **prompt**, so
generation begins already inside it and the reply carries no opener at all. Such
a reply is, byte for byte, indistinguishable from a plain answer — so the
sanitiser has to be *told*, and the thing that tells it has to fail safe.

Every hiding case here asserts that a secret substring is absent from the text
that would be rendered, rather than that the visible answer merely looks right.
The reachability cases pull their profiles out of the shipped registry, because
the question being asked is whether the **shipped** configuration leaks.
"""

from __future__ import annotations

import io
from typing import Any

import pytest

from tests.unit.test_chat_service import build_service, make_manifest, write_artifact
from yonga.adapters.runtime_genai import (
    DEFAULT_REASONING_WRAPPERS,
    ChatProfile,
    GenerationResult,
    resolve_starts_in_reasoning,
    template_opens_reasoning,
)
from yonga.application.chat import ChatOptions, ChatProfileView, sanitize_answer
from yonga.domain.models import ModelIdentity, ModelMetadata
from yonga.registry.loader import load_builtin_registry
from yonga.registry.resolver import RegistryResolver
from yonga.ui.chat_tui import render_answer

THINK = [("<think>", "</think>")]

#: What a leaked chain of thought would put on screen. No display string in this
#: module may contain it.
SECRET = "the secret is SWORDFISH"

#: The exact reply from the audit: reasoning, with no opener, because the prompt
#: opened the block and the token limit arrived before the closer did.
PROMPT_OPENED_REPLY = f"Here's my reasoning: the user wants X. Step 1. Step 2. {SECRET}"

#: The real Qwen3.5 shape: the block is closed immediately when thinking is off,
#: and left **open** at the end of the prompt when it is on.
QWEN_TEMPLATE = (
    "{%- if add_generation_prompt %}\n"
    "    {{- '<|im_start|>assistant\\n' }}\n"
    "    {%- if enable_thinking is defined and enable_thinking is false %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- else %}\n"
    "        {{- '<think>\\n' }}\n"
    "    {%- endif %}\n"
    "{%- endif %}\n"
)

#: The DeepSeek-R1-distill shape: the opener is emitted unconditionally and no
#: variable guards it, so nothing can switch it off.
DEEPSEEK_TEMPLATE = "{%- if add_generation_prompt %}{{- '<|Assistant|><think>\\n' }}{%- endif %}\n"

#: A template that never opens a reasoning block.
PLAIN_TEMPLATE = "{%- if add_generation_prompt %}{{- '<|assistant|>\\n' }}{%- endif %}\n"


def render(renderable: Any, *, width: int = 100) -> str:
    """Render a renderable to plain text the way a non-TTY terminal would."""
    from rich.console import Console

    buffer = io.StringIO()
    Console(file=buffer, width=width, force_terminal=False, no_color=True).print(renderable)
    return buffer.getvalue()


# --------------------------------------------------------------------------------------
# the four cases the audit reproduced
# --------------------------------------------------------------------------------------


def test_case_a_a_prompt_opened_block_truncated_before_its_closer():
    """The leak: reasoning with no tags at all is hidden only if we know the state."""
    answer = sanitize_answer(PROMPT_OPENED_REPLY, THINK, show=False, starts_in_reasoning=True)
    assert SECRET not in answer.display_text
    assert answer.display_text == ""
    assert answer.reasoning_hidden is True
    assert answer.reasoning_unterminated is True
    assert answer.reasoning_text is not None
    assert SECRET in answer.reasoning_text


def test_case_b_an_orphan_closer_is_still_hidden_on_its_own():
    """The existing heuristic is sound and independent of the new starting state."""
    text = f"{SECRET}</think>the answer"
    for starts_inside in (False, True):
        answer = sanitize_answer(text, THINK, show=False, starts_in_reasoning=starts_inside)
        assert SECRET not in answer.display_text
        assert answer.display_text == "the answer"
        assert answer.reasoning_hidden is True


def test_case_c_a_well_formed_block_is_hidden_in_either_state():
    text = f"<think>{SECRET}</think>the answer"
    for starts_inside in (False, True):
        answer = sanitize_answer(text, THINK, show=False, starts_in_reasoning=starts_inside)
        assert SECRET not in answer.display_text
        assert answer.display_text == "the answer"
        assert answer.reasoning_hidden is True


def test_case_d_an_opener_the_output_never_closes_is_hidden_in_either_state():
    text = f"<think>{SECRET}"
    for starts_inside in (False, True):
        answer = sanitize_answer(text, THINK, show=False, starts_in_reasoning=starts_inside)
        assert SECRET not in answer.display_text
        assert answer.display_text == ""
        assert answer.reasoning_hidden is True
        assert answer.reasoning_unterminated is True


# --------------------------------------------------------------------------------------
# what the new starting state does, and what it leaves alone
# --------------------------------------------------------------------------------------


def test_starting_inside_a_block_hides_only_up_to_the_first_closer():
    text = f"{SECRET}\n</think>\n\nThe answer is 42."
    answer = sanitize_answer(text, THINK, show=False, starts_in_reasoning=True)
    assert answer.display_text == "The answer is 42."
    assert SECRET not in answer.display_text
    assert answer.reasoning_hidden is True
    assert answer.reasoning_unterminated is False


def test_starting_inside_a_block_hides_everything_before_the_first_closer():
    """Even an opener inside the prompt-opened block does not re-expose what precedes it."""
    text = f"{SECRET} <think>nested</think> the answer"
    answer = sanitize_answer(text, THINK, show=False, starts_in_reasoning=True)
    assert SECRET not in answer.display_text
    assert answer.display_text == "the answer"


def test_show_still_reveals_everything():
    for starts_inside in (False, True):
        answer = sanitize_answer(
            PROMPT_OPENED_REPLY, THINK, show=True, starts_in_reasoning=starts_inside
        )
        assert answer.display_text == PROMPT_OPENED_REPLY
        assert answer.reasoning_hidden is False


def test_a_plain_answer_outside_a_block_is_untouched():
    """The default keeps today's behaviour for every existing caller."""
    text = "The capital of France is Paris."
    assert sanitize_answer(text, THINK, show=False).display_text == text
    assert sanitize_answer(text, THINK, show=False, starts_in_reasoning=False).display_text == text


def test_starting_inside_a_block_with_no_wrappers_configured_hides_everything():
    """Told we are inside a block with no way to recognise its end, hide it all."""
    answer = sanitize_answer(PROMPT_OPENED_REPLY, [], show=False, starts_in_reasoning=True)
    assert SECRET not in answer.display_text
    assert answer.display_text == ""
    assert answer.reasoning_hidden is True


def test_a_hidden_prompt_opened_block_explains_itself_instead_of_rendering_empty():
    """A blank panel would be a trap; the user must be told how to undo the hiding."""
    answer = sanitize_answer(PROMPT_OPENED_REPLY, THINK, show=False, starts_in_reasoning=True)
    result = GenerationResult(
        text=PROMPT_OPENED_REPLY,
        display_text=answer.display_text,
        reasoning_text=answer.reasoning_text,
        reasoning_hidden=answer.reasoning_hidden,
        reasoning_unterminated=answer.reasoning_unterminated,
        output_tokens=200,
    )
    rendered = render(render_answer(result))
    assert SECRET not in rendered
    assert "200 tokens of reasoning" in rendered
    assert "/reasoning on" in rendered


# --------------------------------------------------------------------------------------
# establishing the state from the template, failing safe
# --------------------------------------------------------------------------------------


def test_a_template_that_emits_an_opener_is_recognised():
    assert template_opens_reasoning(QWEN_TEMPLATE, DEFAULT_REASONING_WRAPPERS) is True
    assert template_opens_reasoning(DEEPSEEK_TEMPLATE, DEFAULT_REASONING_WRAPPERS) is True
    assert template_opens_reasoning(PLAIN_TEMPLATE, DEFAULT_REASONING_WRAPPERS) is False
    assert template_opens_reasoning(None, DEFAULT_REASONING_WRAPPERS) is False


def test_a_template_with_no_opener_never_starts_inside_a_block():
    for profile in (ChatProfileView(), ChatProfile(), None):
        assert resolve_starts_in_reasoning(profile, template=PLAIN_TEMPLATE) is False
        assert resolve_starts_in_reasoning(profile, template=None) is False


def test_pushing_the_variable_false_is_what_makes_the_qwen_template_safe():
    declared = ChatProfileView(
        supports_thinking_flag=True, thinking_template_variable="enable_thinking"
    )
    assert resolve_starts_in_reasoning(declared, template=QWEN_TEMPLATE, thinking=False) is False
    assert resolve_starts_in_reasoning(declared, template=QWEN_TEMPLATE, thinking=True) is True


def test_an_unestablished_variable_on_an_opening_template_fails_safe():
    """No variable to push means no proof the block is closed, so assume it is open."""
    profile = ChatProfileView()
    assert resolve_starts_in_reasoning(profile, template=DEEPSEEK_TEMPLATE) is True
    assert resolve_starts_in_reasoning(None, template=DEEPSEEK_TEMPLATE) is True


# --------------------------------------------------------------------------------------
# reachability against the shipped registry
# --------------------------------------------------------------------------------------


def metadata(model_type: str, architecture: str, repo_id: str = "owner/model") -> ModelMetadata:
    """Metadata the registry matcher can resolve a chat profile from."""
    return ModelMetadata(
        identity=ModelIdentity(repo_id=repo_id),
        model_type=model_type,
        architectures=[architecture],
        task="text-generation",
    )


@pytest.mark.parametrize(
    ("model_type", "architecture", "template", "profile_id", "expected"),
    [
        ("qwen3_5", "Qwen3_5ForConditionalGeneration", QWEN_TEMPLATE, "chat-qwen3_5", False),
        ("exotic_net_9000", "ExoticForCausalLM", DEEPSEEK_TEMPLATE, "chat-default", True),
        ("qwen3", "Qwen3ForCausalLM", DEEPSEEK_TEMPLATE, "chat-qwen3", True),
        ("qwen3", "Qwen3ForCausalLM", PLAIN_TEMPLATE, "chat-qwen3", False),
    ],
)
def test_the_shipped_registry_resolves_the_expected_starting_state(
    model_type, architecture, template, profile_id, expected
):
    resolver = RegistryResolver(load_builtin_registry())
    rule = resolver.chat_profile(meta=metadata(model_type, architecture))
    assert rule is not None
    assert rule.id == profile_id
    assert resolve_starts_in_reasoning(rule, template=template) is expected


# --------------------------------------------------------------------------------------
# the session wires it up, and says so
# --------------------------------------------------------------------------------------


def deepseek_manifest(**overrides: Any):
    """A manifest for a model whose template opens the block unconditionally."""
    payload: dict[str, Any] = {
        "model_type": "qwen3",
        "architecture": "Qwen3ForCausalLM",
        "recommended_runtime": None,
    }
    payload.update(overrides)
    return make_manifest(**payload)


def test_a_session_on_an_opening_template_hides_a_tagless_reply(tmp_path):
    service, _factory, _runtime = build_service(tmp_path, answers=[PROMPT_OPENED_REPLY])
    artifact = write_artifact(tmp_path, deepseek_manifest())
    (artifact / "chat_template.jinja").write_text(DEEPSEEK_TEMPLATE, encoding="utf-8")

    session = service.open_artifact(artifact)
    assert session.thinking_variable is None
    assert session.starts_in_reasoning is True

    turn = session.send("Hello")
    assert turn.ok
    assert turn.result is not None
    assert SECRET not in turn.display_text
    assert turn.display_text == ""
    assert turn.reasoning_hidden is True
    assert turn.reasoning_unterminated is True
    rendered = render(render_answer(turn.result))
    assert SECRET not in rendered
    assert "tokens of reasoning" in rendered


def test_a_session_on_a_closed_template_leaves_a_plain_reply_alone(tmp_path):
    answer = "The capital of France is Paris."
    service, _factory, _runtime = build_service(tmp_path, answers=[answer])
    artifact = write_artifact(tmp_path, make_manifest())
    (artifact / "chat_template.jinja").write_text(QWEN_TEMPLATE, encoding="utf-8")

    session = service.open_artifact(artifact)
    assert session.profile.id == "chat-qwen3_5"
    assert session.starts_in_reasoning is False
    assert session.send("Hello").display_text == answer


def test_enabling_thinking_on_the_qwen_template_starts_inside_the_block(tmp_path):
    """With thinking on the prompt ends on an unclosed opener, so the state flips."""
    service, _factory, _runtime = build_service(tmp_path, answers=[PROMPT_OPENED_REPLY])
    artifact = write_artifact(tmp_path, make_manifest())
    (artifact / "chat_template.jinja").write_text(QWEN_TEMPLATE, encoding="utf-8")

    session = service.open_artifact(artifact, options=ChatOptions(thinking=True))
    assert session.starts_in_reasoning is True
    assert SECRET not in session.send("Hello").display_text


def test_show_reasoning_still_gets_past_the_assumption(tmp_path):
    service, _factory, _runtime = build_service(tmp_path, answers=[PROMPT_OPENED_REPLY])
    artifact = write_artifact(tmp_path, deepseek_manifest())
    (artifact / "chat_template.jinja").write_text(DEEPSEEK_TEMPLATE, encoding="utf-8")

    session = service.open_artifact(artifact, options=ChatOptions(show_reasoning=True))
    assert session.starts_in_reasoning is True
    assert session.send("Hello").display_text == PROMPT_OPENED_REPLY


def test_the_startup_panel_admits_it_is_hiding_by_assumption(tmp_path):
    service, _factory, _runtime = build_service(tmp_path)
    artifact = write_artifact(tmp_path, deepseek_manifest())
    (artifact / "chat_template.jinja").write_text(DEEPSEEK_TEMPLATE, encoding="utf-8")

    extras = service.open_artifact(artifact).describe().extra
    summary = extras["Thinking mode"]
    assert "unavailable" in summary
    assert "hidden" in summary
