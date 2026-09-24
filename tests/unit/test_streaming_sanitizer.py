"""Streaming the reasoning policy: nothing released can ever turn out to be reasoning.

A streamed answer cannot be taken back, so the sanitiser may only release text
that no later chunk can reclassify. Every test feeds text in awkward pieces --
a character at a time, split inside the wrapper tags -- and checks both that
no reasoning escaped at any point and that the pieces add up to the answer
:func:`sanitize_answer` gives for the whole text.
"""

from __future__ import annotations

import pytest

from yonga.application.chat import StreamingSanitizer, sanitize_answer

WRAPPERS = [("<think>", "</think>")]
SECRET = "the-user-must-never-see-this"


def stream(text: str, size: int, **kwargs: bool) -> tuple[list[str], str]:
    sanitizer = StreamingSanitizer(WRAPPERS, **kwargs)
    released = [sanitizer.feed(text[i : i + size]) for i in range(0, len(text), size)]
    released.append(sanitizer.finish())
    return [piece for piece in released if piece], "".join(released)


@pytest.mark.parametrize("size", [1, 2, 3, 7, 1000])
@pytest.mark.parametrize(
    "text",
    [
        "Plain answer with no reasoning at all.",
        f"<think>{SECRET}</think>\n\nThe answer is 42.",
        f"Before. <think>{SECRET}</think> After.",
        f"<think>\n\n</think>\n\nHello <think>{SECRET}</think>world",
        "  padded answer  ",
    ],
)
def test_the_pieces_add_up_to_the_sanitised_answer(text: str, size: int) -> None:
    pieces, joined = stream(text, size)

    expected = sanitize_answer(text, WRAPPERS, show=False).display_text.strip()
    assert joined == expected
    assert all(SECRET not in piece for piece in pieces)


@pytest.mark.parametrize("size", [1, 2, 5])
def test_a_partial_opener_is_held_until_it_is_settled(size: int) -> None:
    sanitizer = StreamingSanitizer(WRAPPERS)
    released = ""
    for index in range(0, len("Hi <thi"), size):
        released += sanitizer.feed("Hi <thi"[index : index + size])
    assert "<" not in released, "a tag fragment must not be released before it is known"
    released += sanitizer.feed(f"nk>{SECRET}</think> there")
    released += sanitizer.finish()

    assert released == "Hi there"


def test_an_unterminated_block_releases_nothing_after_its_opener() -> None:
    _, joined = stream(f"Visible. <think>{SECRET} and it never ends", 3)

    assert joined == "Visible."


def test_generation_that_starts_inside_reasoning_releases_only_after_it_closes() -> None:
    _, joined = stream(f"{SECRET}</think>\n\nAnswer.", 4, starts_in_reasoning=True)

    assert joined == "Answer."


def test_generation_that_starts_inside_reasoning_and_never_closes_releases_nothing() -> None:
    pieces, joined = stream(f"{SECRET} still thinking", 4, starts_in_reasoning=True)

    assert pieces == []
    assert joined == ""


def test_show_passes_the_raw_text_through() -> None:
    _, joined = stream(f"<think>{SECRET}</think> Answer.", 3, show=True)

    assert joined == f"<think>{SECRET}</think> Answer."
