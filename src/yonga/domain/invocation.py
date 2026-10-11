"""The command the user ran, kept so that a remediation step can repeat it.

A step that says "re-run with --yes" is only as good as the command it prints.
Rebuilt from the repository id alone it dropped every other option the user
passed: ``yonga prepare Qwen/Qwen3-0.6B --device cpu`` came back as
``yonga prepare Qwen/Qwen3-0.6B --yes``, which auto-selected the GPU. A
:class:`CommandLine` is the parsed invocation -- the root options, the command
path, its arguments and its options -- and :meth:`CommandLine.rerun` prints it
again with an option added, removed or given another value, shell-quoted so it
can be pasted as it stands.

An option whose name says it carries a secret is never kept, so no remediation
can print one. The CLI masks secret-shaped values of the other options before
it builds the command line.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

__all__ = ["CommandLine", "Option", "is_secret_option"]

#: One option as typed: the flag alone (``("--yes",)``) or with its value.
Option = tuple[str, ...]

#: Option names that carry a credential. ``--max-new-tokens`` is a count, not
#: one: the name has to end in the singular word.
_SECRET_OPTION = re.compile(
    r"(?:^|-)(?:token|password|passwd|secret|api-key|apikey|credentials?|auth)$", re.IGNORECASE
)


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
    ) -> str:
        """The same command, changed as asked, as one shell-quoted line.

        ``add`` appends flags the command does not already carry; ``drop``
        removes options by name, with their values; ``replace`` gives an option
        another value, where it stands or at the end. ``add_root`` adds a root
        option such as ``--debug``, which only works before the command.
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
        root = list(self.root_options)
        present_root = {option[0] for option in root}
        root.extend((flag,) for flag in dict.fromkeys(add_root) if flag not in present_root)
        return self._render(_public(root), _public(options))

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
