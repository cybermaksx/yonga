"""The command the user ran, kept so that a remediation step can repeat it.

A step that says "re-run with --yes" is only as good as the command it prints.
Rebuilt from the repository id alone it dropped every other option the user
passed: ``yonga prepare Qwen/Qwen3-0.6B --device cpu`` came back as
``yonga prepare Qwen/Qwen3-0.6B --yes``, which auto-selected the GPU. A
:class:`CommandLine` is the parsed invocation -- the root options, the command
path, its arguments and its options -- and :meth:`CommandLine.rerun` prints it
again with an option added, removed or given another value, shell-quoted so it
can be pasted as it stands.

Advice that names *another* command keeps the root options too:
:meth:`CommandLine.follow_up` builds one from words, and
:meth:`CommandLine.carry_root` adds them to the ``yonga`` commands a line of
advice already names. After ``yonga --data-dir X prepare M`` the next step is
``yonga --data-dir X chat ID``; without ``--data-dir`` it would look for the
artifact in the default store and not find it.

An option whose name says it carries a secret is never kept, so no remediation
can print one. The CLI masks secret-shaped values of the other options before
it builds the command line.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Collection, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

__all__ = ["OUTPUT_ROOT_OPTIONS", "CommandLine", "Option", "is_secret_option"]

#: One option as typed: the flag alone (``("--yes",)``) or with its value.
Option = tuple[str, ...]

#: Option names that carry a credential. ``--max-new-tokens`` is a count, not
#: one: the name has to end in the singular word.
_SECRET_OPTION = re.compile(
    r"(?:^|-)(?:token|password|passwd|secret|api-key|apikey|credentials?|auth)$", re.IGNORECASE
)


#: Root options that only change how this one invocation prints. A different
#: command suggested afterwards does not inherit them: they say nothing about
#: where the data is or whether the network may be used.
OUTPUT_ROOT_OPTIONS: frozenset[str] = frozenset({"--json", "--debug", "--verbose", "-v"})

#: ``yonga`` where a command starts: at the start of the text, after ``: ``, or
#: after an opening quote or backtick -- never in prose such as "yonga never".
_COMMAND_START = re.compile(r"(?:^|(?<=:\s)|(?<=['`\"(]))yonga\s+(?P<word>[a-z][a-z-]*)\b")


def is_secret_option(flag: str) -> bool:
    """Whether an option called ``flag`` would carry a secret as its value."""
    return bool(_SECRET_OPTION.search(flag.lstrip("-")))


def _public(options: Iterable[Option]) -> tuple[Option, ...]:
    """``options`` without any whose name marks it as a secret."""
    return tuple(tuple(option) for option in options if option and not is_secret_option(option[0]))


class CommandLine(BaseModel):
    """One ``yonga`` invocation, option by option.

    Options are kept in groups (a flag with its value) so that one can be
    removed or replaced without parsing the command again. Root options such
    as ``--data-dir`` and ``--offline`` are kept apart, because they must come
    before the command.
    """

    model_config = ConfigDict(frozen=True)

    program: str = "yonga"
    command: tuple[str, ...] = ()
    """The command path: ``("prepare",)``, ``("env", "remove")``."""

    arguments: tuple[str, ...] = ()
    options: tuple[Option, ...] = ()
    root_options: tuple[Option, ...] = ()

    @field_validator("options", "root_options", mode="after")
    @classmethod
    def _drop_secrets(cls, value: tuple[Option, ...]) -> tuple[Option, ...]:
        return _public(value)

    def rerun(
        self,
        *,
        add: Sequence[str] = (),
        drop: Sequence[str] = (),
        replace: Mapping[str, str] | None = None,
        add_root: Sequence[str] = (),
        drop_root: Sequence[str] = (),
    ) -> str:
        """The same command, changed as asked, as one shell-quoted line.

        ``add`` appends flags the command does not already carry; ``drop``
        removes options by name, with their values; ``replace`` gives an option
        another value, where it stands or at the end. ``add_root`` adds a root
        option such as ``--debug``, which only works before the command, and
        ``drop_root`` removes one, such as ``--offline``.
        """
        changes = dict(replace or {})
        removed = set(drop)
        options: list[Option] = []
        for option in self.options:
            flag = option[0]
            if flag in removed:
                continue
            if flag in changes:
                options.append((flag, changes.pop(flag)))
                continue
            options.append(option)
        options.extend((flag, value) for flag, value in changes.items())
        present = {option[0] for option in options}
        options.extend((flag,) for flag in dict.fromkeys(add) if flag not in present)
        dropped_root = set(drop_root)
        root = [option for option in self.root_options if option[0] not in dropped_root]
        present_root = {option[0] for option in root}
        root.extend((flag,) for flag in dict.fromkeys(add_root) if flag not in present_root)
        return self._render(_public(root), _public(options))

    @property
    def carried_root_options(self) -> tuple[Option, ...]:
        """The root options another suggested command needs: all but the output ones."""
        return tuple(option for option in self.root_options if option[0] not in OUTPUT_ROOT_OPTIONS)

    def follow_up(self, *words: str) -> str:
        """Another ``yonga`` command, ``words``, with this one's root options, shell-quoted.

        ``follow_up("chat", artifact_id)`` after ``yonga --data-dir X prepare M``
        is ``yonga --data-dir X chat ID``: the artifact is in X, and the plain
        ``yonga chat ID`` would not find it.
        """
        root = [word for option in self.carried_root_options for word in option]
        return shlex.join([self.program, *root, *words])

    def carry_root(self, text: str, *, commands: Collection[str] | None = None) -> str:
        """``text`` with this invocation's root options added to each ``yonga`` command in it.

        A command is ``yonga`` followed by a command name -- one of ``commands``
        when given -- where a command starts: at the start of the text, after
        ``": "``, or after an opening quote or backtick, so prose such as
        "yonga never installs" is left alone. A command that already starts
        with an option (one this class printed whole) is left as it is too.
        """
        root = [word for option in self.carried_root_options for word in option]
        if not root:
            return text
        prefix = shlex.join([self.program, *root])

        def insert(match: re.Match[str]) -> str:
            word = match.group("word")
            if commands is not None and word not in commands:
                return match.group(0)
            return f"{prefix} {word}"

        return _COMMAND_START.sub(insert, text)

    def _render(self, root: Sequence[Option], options: Sequence[Option]) -> str:
        """Join the parts in the order the parser expects them."""
        words = [self.program, *(word for option in root for word in option), *self.command]
        flat = [word for option in options for word in option]
        if any(argument.startswith("-") for argument in self.arguments):
            # An argument that looks like an option must follow ``--``.
            words.extend([*flat, "--", *self.arguments])
        else:
            words.extend([*self.arguments, *flat])
        return shlex.join(words)
