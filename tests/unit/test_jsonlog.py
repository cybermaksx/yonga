"""Tests for the per-job JSON Lines logger."""

from __future__ import annotations

import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path

import pytest

from yonga.util.jsonlog import JobLogger, null_logger
from yonga.util.redaction import REDACTED

HF_TOKEN = "hf_QWERTYuiop1234567890asdfghjklZXCV"


class Colour(str, Enum):
    RED = "red"


def read_records(path: Path) -> list[dict[str, object]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "job.jsonl"


def test_record_shape(log_path: Path) -> None:
    logger = JobLogger("job-1", log_path)
    logger.info("stage.started", "preparing", device="NPU", attempt=1)
    logger.close()

    records = read_records(log_path)
    assert len(records) == 1
    record = records[0]
    assert set(record) >= {
        "timestamp",
        "job_id",
        "level",
        "component",
        "event",
        "message",
    }
    assert record["job_id"] == "job-1"
    assert record["level"] == "info"
    assert record["component"] == "yonga"
    assert record["event"] == "stage.started"
    assert record["message"] == "preparing"
    assert record["device"] == "NPU"
    assert record["attempt"] == 1
    assert isinstance(record["timestamp"], str)
    assert record["timestamp"].endswith("Z")
    datetime.fromisoformat(str(record["timestamp"]).replace("Z", "+00:00"))


def test_one_json_object_per_line(log_path: Path) -> None:
    logger = JobLogger("job-2", log_path)
    logger.debug("a", "first\nsecond")
    logger.info("b")
    logger.warning("c")
    logger.error("d")
    logger.close()

    raw = log_path.read_text(encoding="utf-8").splitlines()
    assert len(raw) == 4
    levels = [json.loads(line)["level"] for line in raw]
    assert levels == ["debug", "info", "warning", "error"]


def test_parent_directory_is_created(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "job.jsonl"
    logger = JobLogger("job-3", path)
    logger.info("created")
    logger.close()

    assert path.is_file()
    assert logger.path == path


def test_messages_and_fields_are_redacted(log_path: Path) -> None:
    logger = JobLogger("job-4", log_path)
    logger.info(
        "download.start",
        f"Authorization: Bearer {HF_TOKEN}",
        url=f"https://huggingface.co/api?token={HF_TOKEN}",
        env={"HF_TOKEN": HF_TOKEN},
        items=[f"see {HF_TOKEN}"],
    )
    logger.close()

    text = log_path.read_text(encoding="utf-8")
    assert HF_TOKEN not in text
    assert REDACTED in text


def test_extra_secrets_are_redacted(log_path: Path) -> None:
    secret = "private-literal-value"
    logger = JobLogger("job-5", log_path, extra_secrets=[secret])
    logger.info("event", f"the value is {secret}", field=secret)
    logger.close()

    text = log_path.read_text(encoding="utf-8")
    assert secret not in text


def test_job_id_and_event_are_redacted(log_path: Path) -> None:
    logger = JobLogger(f"job-{HF_TOKEN}", log_path, component=f"comp-{HF_TOKEN}")
    logger.info(f"event-{HF_TOKEN}")
    logger.close()

    assert HF_TOKEN not in log_path.read_text(encoding="utf-8")


def test_bind_shares_the_file_and_changes_the_component(log_path: Path) -> None:
    logger = JobLogger("job-6", log_path, component="prepare")
    child = logger.bind("exporter")
    logger.info("parent.event")
    child.info("child.event")
    logger.close()

    records = read_records(log_path)
    assert [record["component"] for record in records] == ["prepare", "exporter"]
    assert child.path == logger.path
    assert child.job_id == "job-6"


def test_with_secrets_adds_masking_to_a_child(log_path: Path) -> None:
    logger = JobLogger("job-7", log_path)
    child = logger.with_secrets(["another-private-literal"])
    child.info("event", "leaking another-private-literal")
    logger.close()

    assert "another-private-literal" not in log_path.read_text(encoding="utf-8")


def test_raw_captures_subprocess_output(log_path: Path) -> None:
    logger = JobLogger("job-8", log_path)
    logger.raw("stderr", "OneShotBufferizeVPU2VPUIP failed : Only supports FP16.\n")
    logger.close()

    record = read_records(log_path)[0]
    assert record["stream"] == "stderr"
    assert record["event"] == "subprocess.output"
    assert record["message"] == "OneShotBufferizeVPU2VPUIP failed : Only supports FP16."
    assert record["truncated"] is False


def test_raw_truncates_enormous_lines(log_path: Path) -> None:
    logger = JobLogger("job-9", log_path)
    logger.raw("stdout", "x" * 20_000)
    logger.close()

    record = read_records(log_path)[0]
    assert record["truncated"] is True
    assert len(str(record["message"])) < 20_000


def test_argv_is_masked(log_path: Path) -> None:
    logger = JobLogger("job-10", log_path)
    logger.argv("export.command", ["optimum-cli", "--token", HF_TOKEN, "--sym"])
    logger.close()

    record = read_records(log_path)[0]
    assert record["argv"] == ["optimum-cli", "--token", REDACTED, "--sym"]


def test_reserved_keys_cannot_be_overwritten(log_path: Path) -> None:
    logger = JobLogger("job-11", log_path)
    logger.info("event", "message", level="SHOULD-NOT-WIN", job_id="other")
    logger.close()

    record = read_records(log_path)[0]
    assert record["level"] == "info"
    assert record["job_id"] == "job-11"
    assert record["level_field"] == "SHOULD-NOT-WIN"
    assert record["job_id_field"] == "other"


def test_non_json_values_are_serialised(log_path: Path) -> None:
    logger = JobLogger("job-12", log_path)
    logger.info(
        "event",
        path=Path("/tmp/artifact"),
        colour=Colour.RED,
        when=datetime(2026, 9, 22, 12, 0, 0),
        nested={"a": [1, 2, {"b": Path("/x")}]},
        blob=object(),
        pair=(1, 2),
    )
    logger.close()

    record = read_records(log_path)[0]
    assert record["path"] == "/tmp/artifact"
    assert record["colour"] == "red"
    assert record["when"] == "2026-09-22T12:00:00"
    assert record["nested"] == {"a": [1, 2, {"b": "/x"}]}
    assert isinstance(record["blob"], str)
    assert record["pair"] == [1, 2]


def test_logging_never_raises_when_the_file_cannot_be_opened(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    logger = JobLogger("job-13", blocker / "job.jsonl")
    logger.info("event", "message")
    logger.error("boom")
    logger.raw("stdout", "line")
    logger.close()

    assert logger.healthy is False


def test_logging_never_raises_after_close(log_path: Path) -> None:
    logger = JobLogger("job-14", log_path)
    logger.info("before")
    logger.close()
    logger.close()
    logger.info("after")

    assert len(read_records(log_path)) == 1


def test_logging_never_raises_on_hostile_values(log_path: Path) -> None:
    class Hostile:
        def __str__(self) -> str:
            raise RuntimeError("no string for you")

        __repr__ = __str__

    logger = JobLogger("job-15", log_path)
    logger.info("event", "message", hostile=Hostile())
    logger.info("event.after", "still working")
    logger.close()

    events = [record["event"] for record in read_records(log_path)]
    assert "event.after" in events


def test_context_manager_closes_the_log(log_path: Path) -> None:
    with JobLogger("job-16", log_path) as logger:
        logger.info("inside")

    assert logger.healthy is False
    assert len(read_records(log_path)) == 1


def test_null_logger_discards_everything() -> None:
    logger = null_logger()
    logger.info("event", "message", field=1)
    logger.raw("stdout", "line")
    logger.close()

    assert logger.path == Path(os.devnull)
    assert logger.job_id == "null"
