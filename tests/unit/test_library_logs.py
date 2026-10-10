"""Third-party library log records go to the job log while a job runs.

``huggingface_hub`` relays the Hub's ``X-HF-Warning`` headers through a stderr
handler of its own, which used to print "Warning: You are sending
unauthenticated requests to the HF Hub" straight through a live progress line.
"""

from __future__ import annotations

import io
import json
import logging
import threading
from pathlib import Path

import pytest

from yonga.adapters.hf import HuggingFaceAdapter
from yonga.application.jobs import open_job
from yonga.config.settings import Settings
from yonga.util.jsonlog import JobLogger
from yonga.util.library_logs import LIBRARY_LOG_EVENT, library_logs_to, route_library_logs

HUB_WARNING = "Warning: You are sending unauthenticated requests to the HF Hub."


def records(path: Path, event: str) -> list[dict[str, object]]:
    """The job-log records with ``event``."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [entry for entry in map(json.loads, lines) if entry["event"] == event]


@pytest.fixture
def library(request: pytest.FixtureRequest) -> tuple[logging.Logger, io.StringIO]:
    """A fresh library logger with its own stream handler, as libraries set up."""
    name = f"yonga-test-library-{request.node.name}"
    logger = logging.getLogger(name)
    stream = io.StringIO()
    logger.addHandler(logging.StreamHandler(stream))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    route_library_logs([name])
    return logger, stream


def test_a_library_warning_lands_in_the_job_log_not_the_terminal(
    library: tuple[logging.Logger, io.StringIO], tmp_path: Path
) -> None:
    logger, stream = library
    with JobLogger("job", tmp_path / "job.jsonl") as job_log, library_logs_to(job_log):
        logger.warning(HUB_WARNING)

    assert stream.getvalue() == ""
    [entry] = records(tmp_path / "job.jsonl", LIBRARY_LOG_EVENT)
    assert entry["level"] == "warning"
    assert entry["message"] == HUB_WARNING
    assert entry["library"] == logger.name


def test_worker_threads_are_routed_too(
    library: tuple[logging.Logger, io.StringIO], tmp_path: Path
) -> None:
    """The Hub logs from its download pool, which does not inherit the caller's context."""
    logger, stream = library
    with JobLogger("job", tmp_path / "job.jsonl") as job_log, library_logs_to(job_log):
        worker = threading.Thread(target=logger.warning, args=(HUB_WARNING,))
        worker.start()
        worker.join()

    assert stream.getvalue() == ""
    assert len(records(tmp_path / "job.jsonl", LIBRARY_LOG_EVENT)) == 1


def test_outside_a_job_the_library_handler_still_prints(
    library: tuple[logging.Logger, io.StringIO], tmp_path: Path
) -> None:
    logger, stream = library
    with JobLogger("job", tmp_path / "job.jsonl") as job_log, library_logs_to(job_log):
        pass

    logger.warning(HUB_WARNING)

    assert stream.getvalue().strip() == HUB_WARNING


def test_routing_twice_does_not_duplicate_records(
    library: tuple[logging.Logger, io.StringIO], tmp_path: Path
) -> None:
    logger, stream = library
    route_library_logs([logger.name])

    logger.warning(HUB_WARNING)

    assert stream.getvalue().count(HUB_WARNING) == 1


def test_a_hub_warning_during_a_job_goes_to_its_log(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """End to end: the real ``huggingface_hub`` logger, inside ``open_job``."""
    settings = Settings(data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")
    HuggingFaceAdapter(settings)
    hub_logger = logging.getLogger("huggingface_hub.utils._http")

    with open_job(settings=settings, requested_model="acme/x", create_temp=False) as job:
        hub_logger.warning(HUB_WARNING)
        log_path = job.logger.path

    assert HUB_WARNING not in capfd.readouterr().err
    [entry] = records(log_path, LIBRARY_LOG_EVENT)
    assert entry["message"] == HUB_WARNING
