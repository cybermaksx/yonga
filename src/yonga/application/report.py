"""Bug-report bundles: one archive holding everything a job left behind.

Every repair and classifier in the registry started as a failure someone had to
describe to an upstream project. A bundle makes that cheap: the job's log and
state, the artifact's manifest and validation record when there is one, and a
host report, packed into a single ``.tar.gz`` that can be attached to an issue.

A bundle is meant to be shared, possibly publicly, so it is built defensively:

* every text member is redacted again, even though job logs are redacted when
  written -- older logs predate some redaction rules, and a second pass is cheap;
* the home directory is rewritten to ``~``, the hostname to ``<host>`` and the
  login name to ``<user>``, so paths do not identify the person or machine;
* nothing outside yonga's own managed directories is read.

The archive's ``README.txt`` lists exactly what went in, so the recipient -- and
the person sharing it -- can see what they are handing over.
"""

from __future__ import annotations

import getpass
import io
import json
import re
import socket
import tarfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from yonga import APP_NAME, __version__
from yonga.config.paths import Paths
from yonga.domain.artifacts import MANIFEST_DIRNAME, MANIFEST_FILENAME
from yonga.domain.diagnostics import ConfigurationError, FailureCategory, diagnostic
from yonga.util.redaction import environment_secrets, redact_text

__all__ = [
    "BundleMember",
    "BundleResult",
    "anonymise",
    "build_bundle",
    "find_job_log",
]

#: Login names shorter than this are not rewritten: replacing a one- or
#: two-letter name as a whole word would mangle unrelated text.
_MIN_USER_LENGTH = 3

_README = """\
{app} bug-report bundle
=======================

Job        {job_id}
Created    {created}
{app} version  {version}

This archive was produced by `{app} logs {job_id} --bundle`. It is meant to be
attached to a bug report, including a public one, and was built so that doing
so is safe:

- every text file was redacted for secrets (tokens, Authorization headers, URL
  credentials, and the literal values of secret environment variables);
- the home directory was rewritten to `~`, the hostname to `<host>` and the
  login name to `<user>`.

Please still read it before sharing. Contents:

{members}
"""


class BundleMember(BaseModel):
    """One file inside a bundle."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    size: int


class BundleResult(BaseModel):
    """What :func:`build_bundle` wrote."""

    model_config = ConfigDict(frozen=True)

    path: Path
    job_id: str
    members: list[BundleMember] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def to_json(self) -> dict[str, object]:
        """A JSON-serialisable form for ``--json``."""
        return self.model_dump(mode="json")


def find_job_log(logs_dir: Path, job_id: str) -> Path:
    """Resolve a job id, or an unambiguous prefix of one, to its log file.

    This is the one resolver both ``yonga logs`` and bundling use, so a job
    reference means the same thing everywhere.
    """
    exact = logs_dir / f"{job_id}.jsonl"
    if exact.is_file():
        return exact
    matches = sorted(path for path in logs_dir.glob(f"{job_id}*.jsonl") if path.is_file())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ConfigurationError(
            diagnostic(
                "logs.not_found",
                FailureCategory.CONFIGURATION_ERROR,
                "No such job log",
                f"No job log for {job_id!r} exists in {logs_dir}.",
                remediation=[f"List the jobs that do exist: {APP_NAME} logs"],
                context={"job_id": job_id},
            )
        )
    raise ConfigurationError(
        diagnostic(
            "logs.ambiguous",
            FailureCategory.CONFIGURATION_ERROR,
            "That job id prefix is ambiguous",
            f"{len(matches)} job logs start with {job_id!r}.",
            remediation=[f"Use the full job id: {APP_NAME} logs " + matches[0].stem],
            context={"job_id": job_id},
        )
    )


def anonymise(
    text: str,
    *,
    home: str | None = None,
    user: str | None = None,
    host: str | None = None,
) -> str:
    """Rewrite the home directory to ``~``, the hostname to ``<host>``, the login to ``<user>``.

    The hostname goes first: hostnames commonly embed the login name (for
    example ``alice-laptop``), and the login rule deliberately leaves a name
    followed by a hyphen alone so it cannot mangle identifiers such as
    ``bob-builder``.
    """
    result = text
    if home and home not in ("/", ""):
        result = result.replace(home.rstrip("/"), "~")
    if host:
        for name in dict.fromkeys((host, host.split(".", 1)[0])):
            if len(name) >= _MIN_USER_LENGTH:
                result = re.sub(rf"(?<![\w.-]){re.escape(name)}(?![\w-])", "<host>", result)
    if user and len(user) >= _MIN_USER_LENGTH:
        result = re.sub(rf"(?<![\w.-]){re.escape(user)}(?![\w-])", "<user>", result)
    return result


def build_bundle(
    paths: Paths,
    job_reference: str,
    *,
    output: Path | None = None,
    host_report: Mapping[str, object] | None = None,
    env: Mapping[str, str] | None = None,
    home: str | None = None,
    user: str | None = None,
    host: str | None = None,
) -> BundleResult:
    """Write a redacted, anonymised ``.tar.gz`` for one job and describe it.

    ``host_report`` is the ``doctor`` JSON, gathered by the caller: this module
    does not probe hardware itself. ``env``, ``home`` and ``user`` default to the
    current process and exist so tests can pin them.
    """
    log_path = find_job_log(paths.logs_dir, job_reference)
    job_id = log_path.name.removesuffix(".jsonl")
    secrets = environment_secrets(env)
    home_dir = home if home is not None else str(Path.home())
    login = user if user is not None else _login_name()
    hostname = host if host is not None else _hostname()

    def clean(text: str) -> str:
        return anonymise(
            redact_text(text, extra_secrets=secrets), home=home_dir, user=login, host=hostname
        )

    members: list[tuple[str, str, bytes]] = []
    notes: list[str] = []

    members.append(("job.jsonl", "the job's structured log", _read_text(log_path, clean).encode()))

    state_path = paths.job_state_path(job_id)
    artifact_dir: Path | None = None
    if state_path.is_file():
        state_text = _read_text(state_path, clean)
        members.append(("job.state.json", "the job's final state", state_text.encode()))
        artifact_dir = _artifact_dir_from_state(state_path)
    else:
        notes.append("No job state file was found; the bundle holds the log only.")

    if artifact_dir is not None:
        for name, description in (
            (MANIFEST_FILENAME, "the artifact manifest"),
            ("validation.json", "the artifact validation record"),
        ):
            source = artifact_dir / MANIFEST_DIRNAME / name
            if _inside(source, paths.artifacts_dir) and source.is_file():
                members.append((name, description, _read_text(source, clean).encode()))

    if host_report is not None:
        host_text = json.dumps(host_report, indent=2, sort_keys=True, default=str)
        members.append(
            ("host.json", "the host report (`doctor --json`)", clean(host_text).encode())
        )

    listing = "\n".join(f"- {name}: {description}" for name, description, _ in members)
    readme = _README.format(
        app=APP_NAME,
        job_id=job_id,
        created=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        version=__version__,
        members=listing,
    )
    members.insert(0, ("README.txt", "what this archive is", readme.encode()))

    target = output if output is not None else Path.cwd() / f"{APP_NAME}-report-{job_id[:8]}.tar.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_archive(target, f"{APP_NAME}-report-{job_id[:8]}", members)

    return BundleResult(
        path=target,
        job_id=job_id,
        members=[
            BundleMember(name=name, description=description, size=len(data))
            for name, description, data in members
        ],
        notes=notes,
    )


# --- internals -------------------------------------------------------------------------


def _hostname() -> str | None:
    try:
        return socket.gethostname() or None
    except OSError:
        return None


def _login_name() -> str | None:
    try:
        return getpass.getuser()
    except Exception:  # getpass raises several unrelated types on odd hosts
        return None


def _read_text(path: Path, clean: Callable[[str], str]) -> str:
    """Read a file as text and pass it through ``clean``."""
    return clean(path.read_text(encoding="utf-8", errors="replace"))


def _artifact_dir_from_state(state_path: Path) -> Path | None:
    """The artifact directory a job recorded, if any."""
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    raw = data.get("artifact_dir") if isinstance(data, dict) else None
    return Path(raw) if isinstance(raw, str) and raw else None


def _inside(path: Path, root: Path) -> bool:
    """Whether ``path`` resolves inside ``root`` (so a bundle never reads elsewhere)."""
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _write_archive(target: Path, prefix: str, members: Sequence[tuple[str, str, bytes]]) -> None:
    """Write the members into a gzip tarball under a single top-level directory."""
    now = time.time()
    with tarfile.open(target, "w:gz") as archive:
        for name, _description, data in members:
            info = tarfile.TarInfo(name=f"{prefix}/{name}")
            info.size = len(data)
            info.mtime = int(now)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
