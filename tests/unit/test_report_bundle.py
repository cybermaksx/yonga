"""Bug-report bundles: complete, redacted, anonymised, and confined to yonga's data."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from yonga.application.report import anonymise, build_bundle, find_job_log
from yonga.config.paths import Paths, default_paths
from yonga.domain.diagnostics import ConfigurationError

JOB = "0123abcd-1111-2222-3333-444455556666"
TOKEN = "hf_BundleLeakTestToken1234567890abcd"
OPAQUE_SECRET = "opaque-self-hosted-secret-value"


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    built = default_paths(
        data_dir=tmp_path / "data", cache_dir=tmp_path / "cache", config_dir=tmp_path / "config"
    )
    built.ensure()
    return built


def write_job(paths: Paths, *, artifact_dir: Path | None = None, extra: str = "") -> None:
    """A job log and state as a real run leaves them, with things worth hiding."""
    lines = [
        {"event": "job.created", "message": "job created", "temp_dir": "/home/alice/.cache/x"},
        {"event": "export.command", "message": f"token {TOKEN} and {OPAQUE_SECRET}"},
        {"event": "env", "message": "SESSION_MANAGER=local/alice-x1carbon:@/tmp/.ICE-unix/1"},
        {"event": "note", "message": f"run by alice {extra}"},
    ]
    paths.job_log_path(JOB).write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    state = {"job_id": JOB, "stage": "failed"}
    if artifact_dir is not None:
        state["artifact_dir"] = str(artifact_dir)
    paths.job_state_path(JOB).write_text(json.dumps(state))


def extract(bundle: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    with tarfile.open(bundle) as archive:
        for member in archive.getmembers():
            handle = archive.extractfile(member)
            assert handle is not None
            files[Path(member.name).name] = handle.read().decode()
    return files


def bundle(
    paths: Paths, tmp_path: Path, *, host_report: dict[str, object] | None = None
) -> dict[str, str]:
    result = build_bundle(
        paths,
        JOB[:8],
        output=tmp_path / "out" / "report.tar.gz",
        host_report=host_report,
        env={"HF_TOKEN": OPAQUE_SECRET},
        home="/home/alice",
        user="alice",
        host="alice-x1carbon",
    )
    return extract(result.path)


def test_the_bundle_holds_the_log_state_and_a_readme(paths: Paths, tmp_path: Path) -> None:
    write_job(paths)
    files = bundle(paths, tmp_path, host_report={"status": "warning"})

    assert set(files) == {"README.txt", "job.jsonl", "job.state.json", "host.json"}
    assert "job.jsonl" in files["README.txt"]
    assert "host.json" in files["README.txt"]


def test_secrets_are_redacted_by_pattern_and_by_environment_value(
    paths: Paths, tmp_path: Path
) -> None:
    write_job(paths)
    files = bundle(paths, tmp_path)

    everything = "\n".join(files.values())
    assert TOKEN not in everything
    assert OPAQUE_SECRET not in everything


def test_home_hostname_and_login_are_anonymised(paths: Paths, tmp_path: Path) -> None:
    write_job(paths)
    files = bundle(paths, tmp_path)

    log = files["job.jsonl"]
    assert "/home/alice" not in log and "~/.cache/x" in log
    assert "alice-x1carbon" not in log and "<host>" in log
    assert "alice" not in log and "run by <user>" in log


def test_the_hugging_face_account_name_never_reaches_the_bundle(
    paths: Paths, tmp_path: Path
) -> None:
    """``doctor`` asks the Hub who the token belongs to; a public bundle must not say."""
    write_job(paths, extra="as hubperson")
    host_report: dict[str, object] = {
        "hardware": {"hf_auth": {"logged_in": True, "user": "hubperson", "auth_source": "env"}},
        "sections": [
            {
                "id": "huggingface",
                "checks": [{"id": "huggingface.auth", "value": "signed in as hubperson"}],
            }
        ],
    }
    files = bundle(paths, tmp_path, host_report=host_report)

    assert all("hubperson" not in text for text in files.values())
    host = json.loads(files["host.json"])
    assert host["hardware"]["hf_auth"] == {
        "logged_in": True,
        "user": "<hf-user>",
        "auth_source": "env",
    }
    assert host["sections"][0]["checks"][0]["value"] == "signed in as <hf-user>"
    assert "as <hf-user>" in files["job.jsonl"]
    assert host_report["hardware"]["hf_auth"]["user"] == "hubperson"  # type: ignore[index]


def test_a_short_hugging_face_name_is_still_hidden_in_the_host_report(
    paths: Paths, tmp_path: Path
) -> None:
    """Too short for the free-text rule, but the structured fields are replaced regardless."""
    write_job(paths)
    host_report: dict[str, object] = {
        "hardware": {"hf_auth": {"logged_in": True, "user": "xq"}},
        "sections": [{"checks": [{"id": "huggingface.auth", "value": "signed in as xq"}]}],
    }
    host = json.loads(bundle(paths, tmp_path, host_report=host_report)["host.json"])

    assert host["hardware"]["hf_auth"]["user"] == "<hf-user>"
    assert host["sections"][0]["checks"][0]["value"] == "signed in as <hf-user>"


def test_a_full_name_starting_with_the_login_is_rewritten_whole() -> None:
    text = "signed in as alice smith, run by alice"
    assert anonymise(text, user="alice", hf_user="alice smith") == (
        "signed in as <hf-user>, run by <user>"
    )
    assert anonymise("run by alice", user="alice", hf_user="alice") == "run by <user>"


def test_a_short_login_is_not_rewritten() -> None:
    """A one- or two-letter name as a whole word would mangle unrelated text."""
    assert anonymise("al is here", user="al") == "al is here"


def test_a_login_inside_a_hyphenated_identifier_is_left_alone() -> None:
    assert anonymise("pip install bob-builder", user="bob") == "pip install bob-builder"
    assert anonymise("run by bob", user="bob") == "run by <user>"


def test_the_manifest_is_included_from_inside_the_artifact_store(
    paths: Paths, tmp_path: Path
) -> None:
    artifact = paths.artifacts_dir / "acme--model" / "abc--cpu"
    (artifact / ".yonga").mkdir(parents=True)
    (artifact / ".yonga" / "manifest.json").write_text('{"artifact_id": "x"}')
    write_job(paths, artifact_dir=artifact)

    assert "manifest.json" in bundle(paths, tmp_path)


def test_a_state_pointing_outside_the_store_is_not_followed(paths: Paths, tmp_path: Path) -> None:
    """A bundle never reads files yonga does not own, whatever the state says."""
    outside = tmp_path / "elsewhere"
    (outside / ".yonga").mkdir(parents=True)
    (outside / ".yonga" / "manifest.json").write_text('{"private": "do not ship"}')
    write_job(paths, artifact_dir=outside)

    files = bundle(paths, tmp_path)

    assert "manifest.json" not in files
    assert "do not ship" not in "\n".join(files.values())


def test_a_missing_state_is_noted_not_fatal(paths: Paths, tmp_path: Path) -> None:
    write_job(paths)
    paths.job_state_path(JOB).unlink()

    result = build_bundle(paths, JOB, output=tmp_path / "r.tar.gz", env={}, home="/h", user="u1x")

    assert [member.name for member in result.members] == ["README.txt", "job.jsonl"]
    assert result.notes


def test_job_references_resolve_by_prefix_and_fail_clearly(paths: Paths) -> None:
    write_job(paths)
    assert find_job_log(paths.logs_dir, JOB[:6]).name == f"{JOB}.jsonl"

    with pytest.raises(ConfigurationError) as missing:
        find_job_log(paths.logs_dir, "ffff")
    assert missing.value.diagnostic.code == "logs.not_found"

    (paths.logs_dir / f"{JOB[:4]}-other.jsonl").write_text("{}\n")
    with pytest.raises(ConfigurationError) as ambiguous:
        find_job_log(paths.logs_dir, JOB[:4])
    assert ambiguous.value.diagnostic.code == "logs.ambiguous"
