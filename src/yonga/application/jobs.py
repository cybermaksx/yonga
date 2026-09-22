"""Job identity, shared per-job context, and job state persistence.

Every long-running operation (preparation, probe, benchmark) runs under a
:class:`JobContext`. The context is the single object services pass to adapters:
it carries the job id, the structured logger, resolved paths and settings, the
managed temp environment, a progress sink, and a cancellation flag.

Job state is persisted as a small JSON document so an interrupted run can be
diagnosed afterwards. This is deliberately *not* a resume mechanism — see
``docs/MODEL_PIPELINE.md``: downloads are reusable through the Hugging Face cache,
but an interrupted export restarts.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from yonga.application.progress import NullSink, ProgressEvent, ProgressKind, ProgressSink
from yonga.config.paths import Paths
from yonga.config.settings import Settings
from yonga.domain.diagnostics import Diagnostic, Severity
from yonga.domain.plans import JobStage, JobState
from yonga.util.jsonlog import JobLogger, null_logger

__all__ = [
    "JobContext",
    "JobStateStore",
    "new_job_id",
    "open_job",
]


def new_job_id() -> str:
    """Return a fresh UUID4 job identifier."""
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


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

    def fail(self, stage: JobStage, diagnostic: Diagnostic) -> None:
        """Record a terminal failure with the stage it happened in."""
        self.logger.error(
            "job.failed",
            diagnostic.message,
            stage=stage.value,
            code=diagnostic.code,
            category=diagnostic.category.value,
        )
        if self.state is not None:
            self.state.fail(diagnostic)


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

    def fail(self, diagnostic: Diagnostic) -> None:
        """Move the job to ``failed``, recording the diagnostic, and persist."""
        self._state.advance(JobStage.FAILED)
        self._state.diagnostic = diagnostic
        self.save()


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

    The managed temp directory is removed on clean exit. On failure it is left in
    place so the job can be diagnosed, matching ``SPEC.md`` section 10.
    """
    paths = settings.paths
    paths.ensure()
    jid = job_id or new_job_id()

    logger: JobLogger
    try:
        logger = JobLogger(jid, paths.job_log_path(jid))
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

    succeeded = False
    try:
        yield ctx
        succeeded = True
    finally:
        if succeeded and create_temp:
            _remove_tree(temp_dir)
        elif create_temp:
            logger.info(
                "job.temp_retained",
                "temporary directory retained for diagnosis",
                temp_dir=str(temp_dir),
            )
        logger.close()


def _remove_tree(path: Path) -> None:
    """Delete a tool-owned directory tree, ignoring errors."""
    import shutil

    with suppress(OSError):
        shutil.rmtree(path, ignore_errors=True)
