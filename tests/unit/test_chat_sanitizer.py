"""The reasoning sanitiser.

This is the function that decides whether hidden reasoning reaches a user's
screen, so the tests are written from the attacker's side: every case that hides
something asserts the secret substring is **absent** from the display string
rather than merely that the visible answer looks right.

The wrapper pairs are not hard-coded in the tests either — the "real registry"
cases pull them out of the built-in chat profiles, which is where the production
code gets them.
"""

from __future__ import annotations

from typing import Any

import pytest

from yonga.application.chat import (
    DEFAULT_REASONING_WRAPPERS,
    sanitize_answer,
    sanitize_reasoning,
)
from yonga.registry.loader import load_builtin_registry

THINK = [("<think>", "</think>")]
SECRET = "the-user-must-never-see-this"


# --- complete blocks ------------------------------------------------------------------


def test_a_complete_block_leaves_only_the_answer():
    display, reasoning = sanitize_reasoning("<think>secret</think>answer", THINK, show=False)
    assert display == "answer"
    assert reasoning == "secret"


def test_a_complete_block_is_removed_from_the_middle_of_a_sentence():
    display, reasoning = sanitize_reasoning(
        f"Hello <think>{SECRET}</think> world", THINK, show=False
    )
    assert display == "Hello world"
    assert SECRET not in display
    assert reasoning == SECRET


def test_an_empty_block_reports_no_reasoning():
    display, reasoning = sanitize_reasoning("<think></think>answer", THINK, show=False)
    assert display == "answer"
    assert reasoning is None


def test_text_with_no_block_reports_no_reasoning():
    display, reasoning = sanitize_reasoning("just an answer", THINK, show=False)
    assert display == "just an answer"
    assert reasoning is None


# --- the defence-in-depth case: an unclosed wrapper -----------------------------------


def test_an_unclosed_wrapper_hides_everything_to_the_end_of_the_text():
    text = f"<think>{SECRET}\nstill reasoning, still unclosed"
    display, reasoning = sanitize_reasoning(text, THINK, show=False)
    assert SECRET not in display
    assert "still reasoning" not in display
    assert display == ""
    assert reasoning is not None
    assert SECRET in reasoning


def test_an_unclosed_wrapper_keeps_the_prose_that_came_before_it():
    text = f"Here is the answer.\n\n<think>{SECRET}"
    display, reasoning = sanitize_reasoning(text, THINK, show=False)
    assert display == "Here is the answer."
    assert SECRET not in display
    assert reasoning == SECRET


def test_a_closing_wrapper_with_no_opener_hides_what_came_before_it():
    """The chat template can emit the opener itself, so generation starts inside the block."""
    display, reasoning = sanitize_reasoning(f"{SECRET}</think>the answer", THINK, show=False)
    assert display == "the answer"
    assert SECRET not in display
    assert reasoning == SECRET


# --- ordinary prose must survive ------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I think this is right.",
        "Thinking about it, yes.",
        "The word think, thinking and thinker all appear here.",
        "<thinking>this tag is not a configured wrapper</thinking> and this is the answer",
        "Use </thinking> in XML if you like.",
        "think",
    ],
)
def test_ordinary_prose_is_untouched(text):
    display, reasoning = sanitize_reasoning(text, THINK, show=False)
    assert display == text
    assert reasoning is None


def test_untouched_text_is_returned_byte_for_byte():
    """No block was removed, so no whitespace policy is applied either."""
    text = "  leading and trailing spaces survive when nothing was cut  "
    display, _ = sanitize_reasoning(text, THINK, show=False)
    assert display == text


# --- nested and repeated blocks -------------------------------------------------------


def test_nested_blocks_are_removed_as_one_outer_block():
    text = f"<think>outer {SECRET} <think>inner</think> tail</think>answer"
    display, reasoning = sanitize_reasoning(text, THINK, show=False)
    assert display == "answer"
    assert SECRET not in display
    assert "inner" not in display
    assert reasoning == f"outer {SECRET} <think>inner</think> tail"


def test_deeply_nested_blocks_stay_hidden():
    text = "<think>" * 12 + SECRET + "</think>" * 12 + "answer"
    display, reasoning = sanitize_reasoning(text, THINK, show=False)
    assert display == "answer"
    assert SECRET not in display
    assert reasoning is not None and SECRET in reasoning


def test_a_nested_block_left_unclosed_hides_everything():
    text = f"<think>outer<think>inner {SECRET}</think>never closed"
    display, _ = sanitize_reasoning(text, THINK, show=False)
    assert display == ""
    assert SECRET not in display


def test_multiple_blocks_are_all_removed():
    text = f"one<think>{SECRET}</think>two<think>{SECRET}-again</think>three"
    display, reasoning = sanitize_reasoning(text, THINK, show=False)
    assert display == "onetwothree"
    assert SECRET not in display
    assert reasoning == f"{SECRET}\n\n{SECRET}-again"


def test_a_second_block_left_unclosed_still_hides_its_content():
    text = f"first<think>a</think>second<think>{SECRET}"
    display, _ = sanitize_reasoning(text, THINK, show=False)
    assert display == "firstsecond"
    assert SECRET not in display


# --- configurable wrappers ------------------------------------------------------------


def test_the_wrappers_are_configurable_and_not_hard_coded():
    wrappers = [("[[reason]]", "[[/reason]]")]
    display, reasoning = sanitize_reasoning(
        f"[[reason]]{SECRET}[[/reason]]answer", wrappers, show=False
    )
    assert display == "answer"
    assert SECRET not in display
    assert reasoning == SECRET


def test_a_wrapper_that_is_not_configured_is_left_alone():
    wrappers = [("[[reason]]", "[[/reason]]")]
    text = f"<think>{SECRET}</think>answer"
    display, reasoning = sanitize_reasoning(text, wrappers, show=False)
    assert display == text
    assert reasoning is None


def test_several_wrapper_kinds_are_handled_together():
    wrappers = [("<think>", "</think>"), ("<scratch>", "</scratch>")]
    text = f"a<think>{SECRET}</think>b<scratch>{SECRET}2</scratch>c"
    display, reasoning = sanitize_reasoning(text, wrappers, show=False)
    assert display == "abc"
    assert SECRET not in display
    assert reasoning == f"{SECRET}\n\n{SECRET}2"


def test_no_configured_wrappers_means_nothing_is_stripped():
    text = f"<think>{SECRET}</think>answer"
    display, reasoning = sanitize_reasoning(text, [], show=False)
    assert display == text
    assert reasoning is None


def test_the_builtin_registry_supplies_the_wrappers_used_in_production():
    registry = load_builtin_registry()
    assert registry.chat, "the built-in registry ships no chat profiles"
    for rule in registry.chat:
        wrappers = rule.reasoning_wrappers
        assert wrappers, f"chat profile {rule.id} configures no reasoning wrappers"
        opening, closing = wrappers[0]
        display, reasoning = sanitize_reasoning(
            f"{opening}{SECRET}{closing}answer", wrappers, show=False
        )
        assert display == "answer"
        assert SECRET not in display
        assert reasoning == SECRET


def test_the_default_wrappers_match_the_registry_default():
    assert list(DEFAULT_REASONING_WRAPPERS) == THINK


# --- show mode ------------------------------------------------------------------------


def test_show_passes_everything_through_and_still_reports_the_reasoning():
    text = "<think>secret</think>answer"
    display, reasoning = sanitize_reasoning(text, THINK, show=True)
    assert display == text
    assert reasoning == "secret"


def test_show_also_reports_an_unclosed_block():
    text = f"<think>{SECRET}"
    display, reasoning = sanitize_reasoning(text, THINK, show=True)
    assert display == text
    assert reasoning == SECRET


# --- whitespace -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>x</think>\n\nanswer", "answer"),
        ("<think>x</think>\nanswer", "answer"),
        ("<think>x</think>   answer", "answer"),
        ("\n\n<think>x</think>\n\n  answer\n\n", "answer"),
        ("before\n<think>x</think>\nafter", "before\n\nafter"),
        ("before <think>x</think> after", "before after"),
        ("before<think>x</think>after", "beforeafter"),
    ],
)
def test_whitespace_around_a_removed_block_is_collapsed(text, expected):
    display, _ = sanitize_reasoning(text, THINK, show=False)
    assert display == expected
    assert not display.startswith(("\n", " "))


def test_a_block_that_is_the_whole_answer_leaves_an_empty_string():
    display, reasoning = sanitize_reasoning("<think>only reasoning</think>", THINK, show=False)
    assert display == ""
    assert reasoning == "only reasoning"


# --- totality -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrappers",
    [
        [],
        [("", "")],
        [("<think>", "")],
        [("", "</think>")],
        [("<think>", "</think>"), ("<think>", "</think>")],
        [("x", "x")],
        [("<think>", "</think>", "extra")],
        ["not-a-pair"],
        [None],
        None,
        "<think>",
        42,
    ],
)
def test_malformed_wrapper_configuration_never_raises(wrappers):
    display, reasoning = sanitize_reasoning("<think>a</think>b", wrappers, show=False)
    assert isinstance(display, str)
    assert reasoning is None or isinstance(reasoning, str)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "<",
        "</think>",
        "<think>",
        "<think></think>",
        "</think><think>",
        "<think>" * 200,
        "</think>" * 200,
        ("<think>a</think>" * 500),
        "\x00<think>\x00</think>\x00",
        "\U0001f600<think>\U0001f600</think>\U0001f600",
        "line\r\n<think>r\r\n</think>\r\nend",
    ],
)
@pytest.mark.parametrize("show", [True, False])
def test_the_sanitiser_is_total(text, show):
    display, reasoning = sanitize_reasoning(text, THINK, show=show)
    assert isinstance(display, str)
    assert reasoning is None or isinstance(reasoning, str)


def test_a_non_string_input_does_not_raise():
    not_text: Any = None
    display, reasoning = sanitize_reasoning(not_text, THINK, show=False)
    assert isinstance(display, str)
    assert reasoning is None


def test_a_pathological_alternation_still_terminates():
    text = ("<think>a</think>b" * 2000) + f"<think>{SECRET}"
    display, _ = sanitize_reasoning(text, THINK, show=False)
    assert SECRET not in display
    assert display == "b" * 2000


def test_an_identical_opening_and_closing_wrapper_pairs_up():
    wrappers = [("%%", "%%")]
    display, reasoning = sanitize_reasoning(f"a%%{SECRET}%%b", wrappers, show=False)
    assert SECRET not in display
    assert display == "ab"
    assert reasoning == SECRET


# --- the typed signals the UI renders from --------------------------------------------
#
# The renderer must be able to tell "said nothing" from "never finished thinking"
# without looking at the text again, so the sanitiser reports the *shape* of what
# it removed. These flags travel to the screen; the reasoning itself does not.


def test_a_clean_answer_reports_no_hidden_reasoning():
    answer = sanitize_answer("just an answer", THINK, show=False)
    assert answer.display_text == "just an answer"
    assert answer.reasoning_hidden is False
    assert answer.reasoning_unterminated is False
    assert answer.is_empty is False


def test_a_closed_block_reports_hidden_but_terminated_reasoning():
    answer = sanitize_answer(f"<think>{SECRET}</think>", THINK, show=False)
    assert SECRET not in answer.display_text
    assert answer.is_empty is True
    assert answer.reasoning_hidden is True
    assert answer.reasoning_unterminated is False


def test_an_unclosed_block_reports_unterminated_reasoning():
    answer = sanitize_answer(f"<think>{SECRET}", THINK, show=False)
    assert SECRET not in answer.display_text
    assert answer.is_empty is True
    assert answer.reasoning_hidden is True
    assert answer.reasoning_unterminated is True


def test_a_nested_block_left_open_is_still_unterminated():
    answer = sanitize_answer(f"<think>a<think>{SECRET}</think>", THINK, show=False)
    assert SECRET not in answer.display_text
    assert answer.reasoning_unterminated is True


def test_a_closing_wrapper_with_no_opener_is_not_unterminated():
    """The template can emit the opener, so generated text may start inside the block."""
    answer = sanitize_answer(f"{SECRET}</think>the answer", THINK, show=False)
    assert SECRET not in answer.display_text
    assert answer.display_text == "the answer"
    assert answer.reasoning_hidden is True
    assert answer.reasoning_unterminated is False


def test_showing_reasoning_hides_nothing_so_nothing_is_flagged_hidden():
    answer = sanitize_answer(f"<think>{SECRET}", THINK, show=True)
    assert SECRET in answer.display_text
    assert answer.reasoning_hidden is False
    assert answer.reasoning_unterminated is True


def test_the_two_value_form_still_agrees_with_the_typed_one():
    raw = f"<think>{SECRET}</think>answer"
    display, reasoning = sanitize_reasoning(raw, THINK, show=False)
    answer = sanitize_answer(raw, THINK, show=False)
    assert (display, reasoning) == (answer.display_text, answer.reasoning_text)


@pytest.mark.parametrize("show", [True, False])
def test_the_typed_sanitiser_is_total(show):
    for text in ("", "<think>", "</think>", "<think>" * 200, "\x00<think>\x00"):
        answer = sanitize_answer(text, THINK, show=show)
        assert isinstance(answer.display_text, str)
        assert isinstance(answer.reasoning_hidden, bool)
        assert isinstance(answer.reasoning_unterminated, bool)
