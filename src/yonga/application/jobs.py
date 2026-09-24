"""Job identity, shared per-job context, job state persistence and retention.

Every long-running operation (preparation, probe, benchmark) runs under a
:class:`JobContext`. The context is the single object services pass to adapters:
it carries the job id, the structured logger, resolved paths and settings, the
managed temp environment, a progress sink, and a cancellation flag.

Job state is persisted as a small JSON document so an interrupted run can be
diagnosed afterwards. This is deliberately *not* a resume mechanism — see
``docs/MODEL_PIPELINE.md``: downloads are reusable through the Hugging Face cache,
but an interrupted export restarts.

Two working roots belong to jobs rather than to artifacts: ``artifacts/.staging``
(where an export is built before it is published) and ``cache/tmp`` (the managed
``TMPDIR`` handed to conversion tooling). Both are keyed by job id, both used to
survive a failure forever, and neither was ever advertised to the user — a
successful publication renamed the model out of its staging directory and left
the empty husk behind, and a failed job left the whole tree. This module now owns
the retention policy for both (``SPEC.md`` section 14):

* a failed job keeps its staging directory **only when it holds something worth
  reading**, and the retained path is recorded in the job state, in the failure
  diagnostic and in the log, so the user can find it (``SPEC.md`` section 16
  wants raw diagnostics retained, not deleted);
* a bounded sweep runs when a job opens and removes the working directories of
  jobs that are neither live nor recent, so the roots cannot grow without limit;
* the sweep only ever touches direct children of the two managed roots, refuses
  symlinks and anything whose real parent is not the root, and never touches a
  directory belonging to a job that is still running.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from yonga.adapters.filesystem import INCOMPLETE_MARKER_NAME
from yonga.application.progress import NullSink, ProgressEvent, ProgressKind, ProgressSink
from yonga.config.paths import Paths
from yonga.config.settings import Settings
from yonga.domain.diagnostics import (
    Diagnostic,
    FailureCategory,
    Severity,
    YongaError,
    diagnostic,
)
from yonga.domain.plans import JobStage, JobState
from yonga.util.jsonlog import JobLogger, null_logger
from yonga.util.redaction import environment_secrets

__all__ = [
    "LIVE_JOB_GRACE_HOURS",
    "STATE_DOCUMENT_SUFFIX",
    "JobContext",
    "JobStateStore",
    "SweepReport",
    "WorkspaceSweep",
    "live_job_ids",
    "new_job_id",
    "open_job",
    "sweep_job_directories",
    "sweep_managed_workspaces",
]

#: Suffix of the per-job state document inside ``paths.logs_dir``.
STATE_DOCUMENT_SUFFIX: str = ".state.json"

#: How long a job whose state is not terminal is presumed to still be running.
#: A live export writes its state at every stage transition, so a full day of
#: silence means the process is gone; until then its directories are untouchable
#: whatever the retention window says.
LIVE_JOB_GRACE_HOURS: float = 24.0

#: Names a sweep refuses outright, before any filesystem check.
_UNSWEEPABLE_NAMES: frozenset[str] = frozenset({"", ".", ".."})

#: File names that do not make a staging directory worth keeping for diagnosis.
_WORTHLESS_NAMES: frozenset[str] = frozenset({INCOMPLETE_MARKER_NAME})


def new_job_id() -> str:
    """Return a fresh UUID4 job identifier."""
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


def _remove_tree(path: Path) -> None:
    """Delete a tool-owned directory tree, ignoring errors."""
    import shutil

    with suppress(OSError):
        shutil.rmtree(path, ignore_errors=True)


def _tree_size(path: Path) -> int:
    """Sum the sizes of every regular file under ``path``. Never raises."""
    total = 0
    with suppress(OSError):
        for item in path.rglob("*"):
            if item.is_symlink():
                continue
            with suppress(OSError):
                if item.is_file():
                    total += item.stat().st_size
    return total


def _has_diagnostic_content(path: Path) -> bool:
    """Whether a working directory holds anything a human could learn from.

    The tool-owned incomplete marker does not count: a staging directory that
    contains nothing but the marker is the residue of an export that produced no
    file at all, and keeping it only wastes an inode.
    """
    with suppress(OSError):
        for item in path.rglob("*"):
            if item.is_symlink():
                return True
            with suppress(OSError):
                if item.is_file() and item.name not in _WORTHLESS_NAMES:
                    return True
    return False


def _directory_mtime(path: Path) -> datetime:
    """Return a directory's modification time as an aware UTC timestamp."""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError:  # pragma: no cover - vanished between listing and stat
        return _now()


@dataclass
class JobContext:
    """Everything a service or adapter needs to do one unit of long-running work.

    Adapters receive a context rather than reaching for globals, which is what
    makes them injectable in tests.
    """

    job_id: str
    logger: JobLogger
    paths: Paths
    settings: Settings
    progress: ProgressSink = field(default_factory=NullSink)
    temp_env: dict[str, str] = field(default_factory=dict)
    cancelled: threading.Event = field(default_factory=threading.Event)
    state: JobStateStore | None = None
    staging_retained: Path | None = None
    """Staging directory kept for diagnosis, once the retention policy has run."""
    staging_resolved: bool = False
    """Whether the staging retention decision has already been taken."""

    @property
    def temp_dir(self) -> Path:
        """The managed, disk-backed temporary directory for this job."""
        return self.paths.job_tmp_dir(self.job_id)

    @property
    def staging_dir(self) -> Path:
        """The tool-owned staging directory this job exports into."""
        return self.paths.job_staging_dir(self.job_id)

    def child_env(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        """Build the environment for a subprocess, forcing managed temp directories.

        ``TMPDIR``/``TMP``/``TEMP`` are overridden so conversion tooling never
        writes multi-gigabyte intermediates into a small ``/tmp`` tmpfs.
        """
        env = dict(os.environ if base is None else base)
        env.update(self.temp_env)
        return env

    def emit(
        self,
        kind: ProgressKind,
        message: str = "",
        *,
        current: int | None = None,
        total: int | None = None,
        device: str | None = None,
        status: str | None = None,
        severity: Severity = Severity.INFO,
        diagnostic: Diagnostic | None = None,
        **data: object,
    ) -> None:
        """Emit a progress event. Never raises."""
        event = ProgressEvent(
            kind=kind,
            message=message,
            current=current,
            total=total,
            device=device,
            status=status,
            severity=severity,
            diagnostic=diagnostic,
            data=dict(data),
        )
        with suppress(Exception):
            self.progress.emit(event)

    def check_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self.cancelled.is_set()

    def advance(self, stage: JobStage) -> None:
        """Record a state-machine transition in both the log and the state file."""
        self.logger.info("job.stage", f"stage -> {stage.value}", stage=stage.value)
        if self.state is not None:
            self.state.advance(stage)

    def retain_staging_for_diagnosis(self) -> Path | None:
        """Apply the failure half of the staging retention policy. Never raises.

        A staging directory that holds a partial export is kept and returned, so
        the failure can be diagnosed from the bytes that were actually produced.
        One that holds nothing but the incomplete marker (or nothing at all) is
        removed immediately: it is the residue of an export that never wrote a
        file, and 15 of those is how ``artifacts/.staging`` grew to gigabytes.

        The decision is taken once per job and cached, so a service and
        :func:`open_job` can both call it.
        """
        if self.staging_resolved:
            return self.staging_retained
        self.staging_resolved = True
        directory = self.staging_dir
        if not directory.is_dir():
            return None
        if _has_diagnostic_content(directory):
            self.staging_retained = directory
            self.logger.info(
                "job.staging_retained",
                "staging directory retained for diagnosis",
                staging_dir=str(directory),
                size_bytes=_tree_size(directory),
            )
        else:
            _remove_tree(directory)
            self.logger.info(
                "job.staging_discarded",
                "empty staging directory removed",
                staging_dir=str(directory),
            )
        return self.staging_retained

    def discard_staging(self) -> None:
        """Remove this job's staging directory after a successful publication.

        The model is renamed out of ``.staging/<job>/model`` when it is published,
        which used to leave the parent husk behind for ever. Anything still
        holding files is left alone and logged instead: the sweep will retire it
        on policy rather than this method guessing.
        """
        self.staging_resolved = True
        directory = self.staging_dir
        if not directory.is_dir():
            return
        if _has_diagnostic_content(directory):
            self.staging_retained = directory
            self.logger.info(
                "job.staging_retained",
                "staging directory still holds files; left for the retention sweep",
                staging_dir=str(directory),
                size_bytes=_tree_size(directory),
            )
            return
        _remove_tree(directory)

    def fail(self, stage: JobStage, diagnostic: Diagnostic) -> None:
        """Record a terminal failure with the stage it happened in.

        The staging retention policy runs first so that the path of any retained
        partial export reaches both the persisted job state and the diagnostic
        the CLI is about to render.
        """
        retained = self.retain_staging_for_diagnosis()
        if retained is not None:
            diagnostic.context.setdefault("staging_dir", str(retained))
            hint = f"Partial output was kept for diagnosis at {retained}"
            if hint not in diagnostic.remediation:
                diagnostic.remediation.append(hint)
        self.logger.error(
            "job.failed",
            diagnostic.message,
            stage=stage.value,
            code=diagnostic.code,
            category=diagnostic.category.value,
        )
        if self.state is not None:
            self.state.fail(diagnostic)

    def fail_unfinished(self, error: BaseException) -> None:
        """Record ``error`` as the job's end, unless a terminal state is already recorded.

        A service records its own typed failures. This is for everything else
        that ends a job: Ctrl-C above all, but also a bug. Without it the state
        document stayed at the last stage reached, so the job looked live for
        :data:`LIVE_JOB_GRACE_HOURS`, which protected its staging directory from
        the sweep and from export reuse, for a job nobody was running. Never raises.
        """
        state = self.state
        if state is None or state.stage.is_terminal:
            return
        if isinstance(error, YongaError):
            ended = error.diagnostic
        elif isinstance(error, KeyboardInterrupt):
            ended = diagnostic(
                "job.cancelled",
                FailureCategory.USER_CANCELLED,
                "Cancelled",
                "The job was interrupted before it finished.",
            )
        else:
            ended = diagnostic(
                "job.aborted",
                FailureCategory.UNKNOWN_RUNTIME_ERROR,
                "The job ended unexpectedly",
                f"{type(error).__name__} ended the job before it finished.",
            )
        with suppress(Exception):
            self.fail(state.stage, ended)


class JobStateStore:
    """Reads and writes the small per-job state JSON document.

    Writes are atomic (temp file + ``os.replace``) so a crash mid-write cannot
    leave an unparsable state file behind.
    """

    def __init__(self, path: Path, state: JobState) -> None:
        self._path = path
        self._state = state

    @property
    def path(self) -> Path:
        """Location of the state document on disk."""
        return self._path

    @property
    def state(self) -> JobState:
        """The current in-memory state."""
        return self._state

    @classmethod
    def create(
        cls,
        *,
        paths: Paths,
        job_id: str,
        requested_model: str,
        target_device: str | None = None,
    ) -> JobStateStore:
        """Create a new state document in the ``created`` stage."""
        now = _now()
        state = JobState(
            job_id=job_id,
            stage=JobStage.CREATED,
            created_at=now,
            updated_at=now,
            requested_model=requested_model,
            target_device=target_device,
            history=[(now.isoformat(), JobStage.CREATED.value)],
        )
        store = cls(paths.job_state_path(job_id), state)
        store.save()
        return store

    @classmethod
    def load(cls, path: Path) -> JobStateStore:
        """Load an existing state document."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(path, JobState.model_validate(data))

    def save(self) -> None:
        """Persist the state atomically. Never raises."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self._state.to_json(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            tmp.replace(self._path)
        except OSError:
            # State persistence is diagnostic sugar; losing it must not fail a job.
            pass

    def advance(
        self,
        stage: JobStage,
        *,
        artifact_dir: Path | None = None,
        profile_id: str | None = None,
        target_device: str | None = None,
    ) -> None:
        """Move the job to ``stage``, optionally recording what it resolved to."""
        self._state.advance(stage)
        if artifact_dir is not None:
            self._state.artifact_dir = artifact_dir
        if profile_id is not None:
            self._state.profile_id = profile_id
        if target_device is not None:
            self._state.target_device = target_device
        self.save()

    @property
    def stage(self) -> JobStage:
        """The stage the job last reached."""
        return self._state.stage

    def fail(self, diagnostic: Diagnostic) -> None:
        """Move the job to ``failed``, recording the diagnostic, and persist."""
        self._state.advance(JobStage.FAILED)
        self._state.diagnostic = diagnostic
        self.save()


# --- retention ------------------------------------------------------------------------


class SweepReport(BaseModel):
    """What one retention sweep removed from one managed working root."""

    model_config = ConfigDict(extra="ignore")

    root: Path
    removed: list[Path] = Field(default_factory=list)
    kept: list[Path] = Field(default_factory=list)
    freed_bytes: int = 0
    remaining_bytes: int = 0
    dry_run: bool = False
    notes: list[str] = Field(default_factory=list)

    @property
    def removed_count(self) -> int:
        """How many directories the sweep removed (or would remove)."""
        return len(self.removed)


class WorkspaceSweep(BaseModel):
    """The combined result of sweeping both managed working roots."""

    model_config = ConfigDict(extra="ignore")

    staging: SweepReport
    temp: SweepReport

    @property
    def freed_bytes(self) -> int:
        """Bytes reclaimed (or reclaimable, on a dry run) across both roots."""
        return self.staging.freed_bytes + self.temp.freed_bytes

    @property
    def remaining_bytes(self) -> int:
        """Bytes still held by working directories the policy keeps."""
        return self.staging.remaining_bytes + self.temp.remaining_bytes

    @property
    def removed_count(self) -> int:
        """How many directories were removed (or would be) across both roots."""
        return self.staging.removed_count + self.temp.removed_count

    @property
    def dry_run(self) -> bool:
        """Whether this was a projection rather than an actual sweep."""
        return self.staging.dry_run and self.temp.dry_run


def live_job_ids(paths: Paths, *, now: datetime | None = None) -> frozenset[str]:
    """Return the ids of jobs that still look like they are running.

    A job counts as live when its state document exists, its stage is not
    terminal, and the document has been written within
    :data:`LIVE_JOB_GRACE_HOURS`. A job that died without reaching a terminal
    stage therefore protects its directories for a day and no longer, which is
    what keeps a crash from pinning gigabytes for ever.

    Never raises: an unreadable or malformed state document is treated as live,
    because refusing to delete is always the safe answer.
    """
    moment = now if now is not None else _now()
    cutoff = moment - timedelta(hours=LIVE_JOB_GRACE_HOURS)
    directory = paths.logs_dir
    live: set[str] = set()
    try:
        documents = sorted(directory.glob(f"*{STATE_DOCUMENT_SUFFIX}"))
    except OSError:  # pragma: no cover - unreadable log directory
        return frozenset()
    for document in documents:
        job_id = document.name[: -len(STATE_DOCUMENT_SUFFIX)]
        if not job_id:
            continue
        if _directory_mtime(document) < cutoff:
            continue
        try:
            payload = json.loads(document.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            live.add(job_id)
            continue
        if not isinstance(payload, dict):
            live.add(job_id)
            continue
        stage = payload.get("stage")
        if not isinstance(stage, str):
            live.add(job_id)
            continue
        try:
            terminal = JobStage(stage).is_terminal
        except ValueError:
            live.add(job_id)
            continue
        if not terminal:
            live.add(job_id)
    return frozenset(live)


def _sweepable_children(root: Path) -> list[Path]:
    """Return the direct child directories of ``root`` a sweep may consider.

    Everything else is refused on sight: a symlink (following one would let a
    link inside the staging root delete an unrelated tree), a name that is empty,
    ``.``, ``..`` or carries a separator, a hidden name, and any entry whose
    resolved parent is not the resolved root.
    """
    try:
        resolved_root = root.resolve()
    except OSError:  # pragma: no cover - defensive
        return []
    if not root.is_dir():
        return []
    try:
        entries = sorted(root.iterdir())
    except OSError:  # pragma: no cover - unreadable root
        return []
    children: list[Path] = []
    for entry in entries:
        name = entry.name
        if name in _UNSWEEPABLE_NAMES or name.startswith("."):
            continue
        if "/" in name or "\\" in name or "\x00" in name:
            continue
        if entry.is_symlink() or not entry.is_dir():
            continue
        try:
            resolved = entry.resolve()
        except OSError:  # pragma: no cover - vanished mid-sweep
            continue
        if resolved.parent != resolved_root:
            continue
        children.append(entry)
    return children


def sweep_job_directories(
    root: Path,
    *,
    paths: Paths,
    settings: Settings,
    protected: Iterable[str] = (),
    now: datetime | None = None,
    dry_run: bool = False,
) -> SweepReport:
    """Apply the working-directory retention policy to one managed root.

    ``root`` must be ``paths.staging_dir`` or ``paths.tmp_dir``; only direct
    children of it are ever considered, and only after the safety checks in
    :func:`_sweepable_children`. Three rules decide the rest:

    * a directory belonging to a live job (see :func:`live_job_ids`) or named in
      ``protected`` is never touched, whatever its age;
    * a directory holding no file worth reading is removed regardless of age — an
      empty husk carries no diagnosis;
    * of the directories that do hold something, the
      ``settings.workspace_retention_max`` most recent survive, and so does any
      directory younger than ``settings.workspace_retention_hours``.

    Both bounds are enforced, so the root is bounded in age *and* in count. The
    sweep never raises: a directory that cannot be removed is reported as kept.
    """
    moment = now if now is not None else _now()
    cutoff = moment - timedelta(hours=float(settings.workspace_retention_hours))
    keep_recent = max(0, int(settings.workspace_retention_max))
    report = SweepReport(root=root, dry_run=dry_run)

    reserved = set(protected) | set(live_job_ids(paths, now=moment))
    with_content: list[Path] = []
    husks: list[Path] = []
    for child in _sweepable_children(root):
        if child.name in reserved:
            report.kept.append(child)
            report.remaining_bytes += _tree_size(child)
            continue
        if _has_diagnostic_content(child):
            with_content.append(child)
        else:
            husks.append(child)

    with_content.sort(key=_directory_mtime, reverse=True)
    doomed = list(husks)
    for index, child in enumerate(with_content):
        recent_enough = index < keep_recent and _directory_mtime(child) >= cutoff
        if recent_enough:
            report.kept.append(child)
            report.remaining_bytes += _tree_size(child)
        else:
            doomed.append(child)

    for child in doomed:
        size = _tree_size(child)
        if dry_run:
            report.removed.append(child)
            report.freed_bytes += size
            continue
        _remove_tree(child)
        if child.exists():
            report.kept.append(child)
            report.remaining_bytes += _tree_size(child)
            report.notes.append(f"{child} could not be removed.")
            continue
        report.removed.append(child)
        report.freed_bytes += size
    return report


def sweep_managed_workspaces(
    settings: Settings,
    *,
    protected: Iterable[str] = (),
    now: datetime | None = None,
    dry_run: bool = False,
) -> WorkspaceSweep:
    """Sweep both managed working roots: artifact staging and the managed temp tree.

    This is the whole garbage collector. It is cheap (a stat walk of two shallow
    roots), it never raises, and it is safe to call at the start of every job.
    """
    paths = settings.paths
    reserved = tuple(protected)
    return WorkspaceSweep(
        staging=sweep_job_directories(
            paths.staging_dir,
            paths=paths,
            settings=settings,
            protected=reserved,
            now=now,
            dry_run=dry_run,
        ),
        temp=sweep_job_directories(
            paths.tmp_dir,
            paths=paths,
            settings=settings,
            protected=reserved,
            now=now,
            dry_run=dry_run,
        ),
    )


@contextmanager
def open_job(
    *,
    settings: Settings,
    requested_model: str,
    job_id: str | None = None,
    progress: ProgressSink | None = None,
    target_device: str | None = None,
    create_temp: bool = True,
) -> Iterator[JobContext]:
    """Open a job: allocate an id, logger, state file and managed temp directory.

    A bounded retention sweep of the two managed working roots runs first, with
    this job's id protected, so neither root can grow without limit however many
    jobs fail (``SPEC.md`` section 14).

    On clean exit the managed temp directory and the emptied staging husk are
    removed. On failure the temp directory is left in place so the job can be
    diagnosed, matching ``SPEC.md`` section 10, and the staging directory is kept
    only when it holds a partial export worth reading.
    """
    paths = settings.paths
    paths.ensure()
    jid = job_id or new_job_id()

    logger: JobLogger
    try:
        # The literal values of the host's secret-bearing environment variables are
        # masked on top of the pattern rules, so a token that does not *look* like
        # one cannot reach the job log through captured subprocess output.
        logger = JobLogger(jid, paths.job_log_path(jid), extra_secrets=environment_secrets())
    except Exception:
        logger = null_logger(jid)

    temp_dir = paths.job_tmp_dir(jid)
    temp_env: dict[str, str] = {}
    if create_temp:
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_env = {"TMPDIR": str(temp_dir), "TMP": str(temp_dir), "TEMP": str(temp_dir)}

    state = JobStateStore.create(
        paths=paths,
        job_id=jid,
        requested_model=requested_model,
        target_device=target_device,
    )

    ctx = JobContext(
        job_id=jid,
        logger=logger,
        paths=paths,
        settings=settings,
        progress=progress or NullSink(),
        temp_env=temp_env,
        state=state,
    )
    logger.info("job.created", "job created", model=requested_model, temp_dir=str(temp_dir))

    # Swept only once this job's state document exists, so a sweep running in a
    # second process already sees this job as live and leaves its directories be.
    swept: WorkspaceSweep | None = None
    with suppress(Exception):
        swept = sweep_managed_workspaces(settings, protected=(jid,))
    if swept is not None and swept.removed_count:
        logger.info(
            "job.workspace_swept",
            "retired stale working directories",
            removed=swept.removed_count,
            freed_bytes=swept.freed_bytes,
        )

    succeeded = False
    try:
        yield ctx
        succeeded = True
    except BaseException as error:
        ctx.fail_unfinished(error)
        raise
    finally:
        if succeeded:
            if create_temp:
                _remove_tree(temp_dir)
            ctx.discard_staging()
        else:
            if create_temp:
                logger.info(
                    "job.temp_retained",
                    "temporary directory retained for diagnosis",
                    temp_dir=str(temp_dir),
                )
            ctx.retain_staging_for_diagnosis()
        logger.close()
