"""A client's stop sequences, matched on the visible answer as it streams.

Released text cannot be taken back, so the matcher may only release what no
later piece can turn into the start of a match. Every test feeds text in
awkward pieces and checks the pieces add up to :func:`truncate_at_stop` over
the whole text, which is what an unstreamed answer gets.
"""

from __future__ import annotations

import pytest

from yonga.application.chat import StopMatcher, StreamingSanitizer, truncate_at_stop

WRAPPERS = [("<think>", "</think>")]


def run(text: str, stops: list[str], size: int) -> tuple[list[str], str, bool]:
    matcher = StopMatcher(stops)
    released = [matcher.feed(text[i : i + size]) for i in range(0, len(text), size)]
    released.append(matcher.finish())
    return released, "".join(released), matcher.hit


@pytest.mark.parametrize("size", [1, 2, 3, 5, 1000])
@pytest.mark.parametrize(
    ("text", "stops", "expected", "hit"),
    [
        ("One. Two. Three.", ["Two"], "One.", True),
        ("One. Two. Three.", ["Four"], "One. Two. Three.", False),
        ("Line one\nLine two", ["\n"], "Line one", True),
        ("xab", ["ab", "b"], "x", True),
        ("x b ab", ["ab", "b"], "x", True),
        ("aaab", ["aab"], "a", True),
        ("STOP right away", ["STOP"], "", True),
        ("ends with a partial ST", ["STOP"], "ends with a partial ST", False),
        ("Bu çok güzel bir gün", ["güzel"], "Bu çok", True),
        ("emoji 🙂 then 🛑 stop", ["🛑"], "emoji 🙂 then", True),
        ("Answer.\n\nUser: next", ["\n\nUser:", "User:"], "Answer.", True),
    ],
)
def test_the_pieces_add_up_to_the_truncated_answer(
    text: str, stops: list[str], expected: str, hit: bool, size: int
) -> None:
    _, joined, matched = run(text, stops, size)

    assert truncate_at_stop(text, stops) == (expected, hit)
    assert joined == expected
    assert matched is hit


@pytest.mark.parametrize("size", [1, 2, 4])
def test_no_part_of_a_match_is_ever_released(size: int) -> None:
    released, joined, _ = run("Hello there, STOP and the rest", ["STOP"], size)

    assert joined == "Hello there,"
    assert all("S" not in piece for piece in released)


def test_a_possible_start_is_held_until_the_next_piece_settles_it() -> None:
    matcher = StopMatcher(["STOP"])

    assert matcher.feed("Hello ST") == "Hello", "the partial match and its space are held"
    assert matcher.feed("ILL here") == " STILL here"
    assert matcher.hit is False


def test_everything_after_a_match_is_swallowed() -> None:
    matcher = StopMatcher(["STOP"])
    matcher.feed("a STOP b")

    assert matcher.hit
    assert matcher.feed(" more text") == ""
    assert matcher.finish() == ""


def test_empty_stops_are_ignored() -> None:
    _, joined, hit = run("nothing to stop here", ["", ""], 3)

    assert (joined, hit) == ("nothing to stop here", False)
    assert truncate_at_stop("text", [""]) == ("text", False)


@pytest.mark.parametrize("size", [1, 3, 7])
def test_a_stop_inside_hidden_reasoning_does_not_end_the_answer(size: int) -> None:
    """Chained after the sanitiser, the matcher never sees reasoning text."""
    text = "<think>I will say STOP now</think>\n\nThe answer is 42. STOP. Never shown."
    sanitizer = StreamingSanitizer(WRAPPERS)
    matcher = StopMatcher(["STOP"])
    released = ""
    for index in range(0, len(text), size):
        piece = sanitizer.feed(text[index : index + size])
        released += matcher.feed(piece) if piece else ""
    released += matcher.feed(sanitizer.finish()) + matcher.finish()

    assert released == "The answer is 42."
    assert matcher.hit
