"""How the commands that run a prepared artifact read the reference they are given.

``chat``, ``serve``, ``bench``, ``probe``, ``eval`` and ``publish`` all take an
artifact id, an artifact directory or a model id, and all of them must answer
the same two questions the same way: what to say when nothing matches, and
what to say when a model id matched several ready artifacts and the newest was
taken. Both answers live here so no command words them on its own.

Nothing in this module touches the disk or the Hub; the stores do the matching.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from pathlib import Path

from yonga.domain.artifacts import ArtifactSummary
from yonga.domain.diagnostics import Diagnostic, FailureCategory, diagnostic

__all__ = [
    "artifact_not_found",
    "is_model_id",
    "newest_choice_note",
]

#: A Hugging Face model id, ``owner/name``: the only reference ``yonga prepare`` takes as given.
_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
#: An artifact id as ``yonga list`` prints it (16 hex digits), or a prefix the store accepts.
_ARTIFACT_ID = re.compile(r"[0-9a-f]{6,16}")


def is_model_id(reference: str) -> bool:
    """True when ``reference`` reads as ``owner/name`` and is not a path on disk."""
    return bool(_REPO_ID.fullmatch(reference)) and not Path(reference).expanduser().exists()


def artifact_not_found(
    reference: str,
    *,
    command: str,
    code: str = "artifact.not_found",
) -> Diagnostic:
    """Explain a reference that matches no prepared artifact, for ``yonga <command>``.

    None of these commands converts a model as a side effect, so a model id
    gets the exact ``yonga prepare`` command to run first and then this one
    again (code ``artifact.not_prepared``). Anything else -- an artifact id that
    is gone, a mistyped name -- is pointed at ``yonga list``: preparing an
    artifact id would only send it to the Hub as if it were a model. ``code``
    is the one the command has always reported for a reference it cannot use;
    a command without one of its own reports a model id that is not prepared
    as ``artifact.not_prepared``, as ``chat`` always has.
    """
    quoted = shlex.quote(reference)
    if is_model_id(reference):
        return diagnostic(
            "artifact.not_prepared" if code == "artifact.not_found" else code,
            FailureCategory.CONFIGURATION_ERROR,
            "No prepared artifact for that model",
            (
                f"{reference} has not been prepared on this host. yonga {command} does not "
                "convert a model as a side effect, because that can download and quantize "
                "many gigabytes."
            ),
            remediation=[
                f"Prepare it first: yonga prepare {reference}",
                f"Then run it again: yonga {command} {quoted}",
                "See what is already prepared: yonga list",
            ],
            context={"reference": reference, "command": "prepare", "model": reference},
        )
    if _ARTIFACT_ID.fullmatch(reference):
        title = "No prepared artifact has that id"
        message = (
            f"No artifact on this host has the id {reference}. Artifact ids are the ones "
            "yonga list shows; an artifact deleted with yonga remove no longer has one."
        )
        prepare = "Prepare the model it came from again: yonga prepare OWNER/NAME"
    else:
        title = "No prepared artifact matches that reference"
        message = (
            f"{reference!r} is not the id or directory of a prepared artifact, nor a "
            "Hugging Face model id (owner/name)."
        )
        prepare = "Prepare a model first: yonga prepare OWNER/NAME"
    return diagnostic(
        code,
        FailureCategory.CONFIGURATION_ERROR,
        title,
        message,
        remediation=[
            "See what is already prepared: yonga list",
            prepare,
            "Or pass the artifact directory directly.",
        ],
        context={"reference": reference},
    )


def newest_choice_note(reference: str, matches: Sequence[ArtifactSummary]) -> str | None:
    """Say which artifact a reference chose when several ready ones answered it.

    ``matches`` is :meth:`~yonga.application.artifacts.ArtifactStore.matches`
    for ``reference``: best first, so its head is what every command runs. A
    model id usually names one artifact per profile and device, and taking the
    newest without a word left the user guessing which one ran. Returns
    ``None`` when there was nothing to choose between.
    """
    ready = [summary for summary in matches if summary.ready]
    if len(ready) < 2 or matches[0] is not ready[0]:
        return None
    chosen = ready[0]
    repo_id = reference.partition("@")[0].strip()
    named = (
        f"for {chosen.repo_id}"
        if repo_id.casefold() == chosen.repo_id.casefold()
        else f"matching {reference}"
    )
    facts = ", ".join(part for part in (chosen.profile, chosen.recommended_device) if part)
    described = f"{chosen.artifact_id} ({facts})" if facts else chosen.artifact_id
    return (
        f"Using {described}, the newest of {len(ready)} ready artifacts {named}; "
        "pass the artifact id to choose another."
    )
