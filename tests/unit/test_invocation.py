"""The command line a remediation step repeats: changed only as asked, never a secret.

On the reference host ``yonga prepare Qwen/Qwen3-0.6B --device cpu`` without a
terminal advised ``yonga prepare Qwen/Qwen3-0.6B --yes``, which auto-selected
the GPU. The advice now repeats the command as it was typed.
"""

from __future__ import annotations

import shlex

from yonga.domain.invocation import CommandLine, is_secret_option

TYPED = CommandLine(
    command=("prepare",),
    arguments=("Qwen/Qwen3-0.6B",),
    options=(("--device", "cpu"), ("--profile", "cpu-int8-asym-ref")),
    root_options=(("--data-dir", "/srv/my models"), ("--offline",)),
)
TYPED_TEXT = (
    "yonga --data-dir '/srv/my models' --offline prepare Qwen/Qwen3-0.6B --device cpu "
    "--profile cpu-int8-asym-ref"
)


def test_the_command_is_repeated_whole_and_shell_quoted() -> None:
    assert TYPED.rerun() == TYPED_TEXT
    assert shlex.split(TYPED.rerun())[2] == "/srv/my models"


def test_a_flag_is_added_once() -> None:
    assert TYPED.rerun(add=("--yes",)) == f"{TYPED_TEXT} --yes"
    already = TYPED.model_copy(update={"options": (*TYPED.options, ("--yes",))})
    assert already.rerun(add=("--yes",)) == f"{TYPED_TEXT} --yes"


def test_an_option_is_dropped_with_its_value() -> None:
    assert TYPED.rerun(drop=("--profile",)) == (
        "yonga --data-dir '/srv/my models' --offline prepare Qwen/Qwen3-0.6B --device cpu"
    )


def test_an_option_is_replaced_where_it_stands_or_added() -> None:
    assert TYPED.rerun(replace={"--profile": "cpu-fp16-ref"}) == TYPED_TEXT.replace(
        "cpu-int8-asym-ref", "cpu-fp16-ref"
    )
    bare = CommandLine(command=("prepare",), arguments=("acme/m",))
    assert bare.rerun(replace={"--device": "cpu"}) == "yonga prepare acme/m --device cpu"


def test_a_root_option_goes_before_the_command() -> None:
    assert TYPED.rerun(add_root=("--debug",)) == TYPED_TEXT.replace(
        "--offline prepare", "--offline --debug prepare"
    )


def test_an_argument_that_looks_like_an_option_follows_a_double_dash() -> None:
    odd = CommandLine(command=("remove",), arguments=("-weird",), options=(("--json",),))
    assert odd.rerun(add=("--yes",)) == "yonga remove --json --yes -- -weird"


def test_an_option_that_carries_a_secret_is_never_kept() -> None:
    with_secret = CommandLine(
        command=("publish",),
        arguments=("abc",),
        options=(("--token", "hf_abcdefghijklmnopqrstuvwxyz0123456789"), ("--repo", "me/m")),
        root_options=(("--api-key", "sk-secret"),),
    )
    line = with_secret.rerun(add=("--yes", "--hf-token"))
    assert line == "yonga publish abc --repo me/m --yes"


def test_only_names_that_end_in_a_credential_word_are_secret() -> None:
    assert is_secret_option("--token")
    assert is_secret_option("--hf-token")
    assert is_secret_option("--api-key")
    assert is_secret_option("--password")
    assert not is_secret_option("--max-new-tokens")
    assert not is_secret_option("--top-k")
    assert not is_secret_option("--license-reviewed")


def test_a_root_option_can_be_dropped() -> None:
    assert TYPED.rerun(drop_root=("--offline",)) == TYPED_TEXT.replace(" --offline", "")


def test_a_follow_up_command_keeps_the_root_options_that_say_where_and_how() -> None:
    """``yonga chat ID`` after ``prepare --data-dir X`` looked in the default store."""
    noisy = TYPED.model_copy(
        update={"root_options": (*TYPED.root_options, ("--json",), ("--debug",), ("-v",))}
    )
    assert noisy.follow_up("chat", "d8a6bf6fea1b4356") == (
        "yonga --data-dir '/srv/my models' --offline chat d8a6bf6fea1b4356"
    )
    assert CommandLine(command=("prepare",)).follow_up("list") == "yonga list"


COMMANDS = frozenset({"chat", "list", "doctor", "prepare", "cache"})


def test_root_options_are_carried_into_the_commands_advice_names() -> None:
    carried = "yonga --data-dir '/srv/my models' --offline"
    assert TYPED.carry_root("See what is already prepared: yonga list", commands=COMMANDS) == (
        f"See what is already prepared: {carried} list"
    )
    assert TYPED.carry_root("Run 'yonga doctor' to check.", commands=COMMANDS) == (
        f"Run '{carried} doctor' to check."
    )
    assert TYPED.carry_root("yonga cache prune --all", commands=COMMANDS) == (
        f"{carried} cache prune --all"
    )


def test_prose_and_whole_commands_are_left_alone() -> None:
    for text in (
        "yonga never installs packages; that choice stays yours.",
        "yonga does not change your environment for you.",
        "an artifact deleted with yonga remove no longer has one",
        f"Re-run with --yes: {TYPED_TEXT} --yes",
    ):
        assert TYPED.carry_root(text, commands=COMMANDS) == text
    bare = CommandLine(command=("chat",), arguments=("x",))
    assert bare.carry_root("Then: yonga list", commands=COMMANDS) == "Then: yonga list"
