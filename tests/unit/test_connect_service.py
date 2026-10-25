"""The connect service: checks, entry values, plan outcomes and the record."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from yonga.adapters.opencode_config import render
from yonga.application.connect import (
    ConnectAction,
    ConnectionRecords,
    ConnectService,
    Endpoint,
)
from yonga.domain.artifacts import (
    ArtifactManifest,
    ArtifactSummary,
    ExportRecord,
    ManifestModelRef,
    RecommendedRuntime,
)
from yonga.domain.diagnostics import ConfigurationError
from yonga.domain.models import PipelineKind

REPO = "Qwen/Qwen3-4B"
ARTIFACT = "073d5d7c8ea75283"
PROFILE = "gpu-int4-asym-g128-r08"


def make_manifest(
    artifact_id: str = ARTIFACT, *, max_new_tokens: int | None = None
) -> ArtifactManifest:
    return ArtifactManifest(
        artifact_id=artifact_id,
        created_at=datetime(2026, 4, 1, tzinfo=UTC),
        tool_version="0.5.1",
        requested_model=ManifestModelRef(repo_id=REPO, resolved_revision="a" * 40),
        resolved_source=ManifestModelRef(repo_id=REPO, resolved_revision="a" * 40),
        architecture="Qwen3ForCausalLM",
        model_type="qwen3",
        task="text-generation",
        pipeline=PipelineKind.LLM,
        export=ExportRecord(profile=PROFILE),
        recommended_runtime=RecommendedRuntime(
            device="GPU", pipeline=PipelineKind.LLM, max_new_tokens=max_new_tokens
        ),
    )


@dataclass
class FakeSession:
    artifact_dir: Path
    manifest: ArtifactManifest | None = field(default_factory=make_manifest)
    device: str = "GPU"
    pipeline: PipelineKind = PipelineKind.LLM
    tool_calling: object | None = field(default_factory=object)
    tool_calling_refusal: str | None = None
    runtime_properties: dict[str, str] = field(default_factory=dict)


def summary(artifact_id: str, *, device: str = "GPU", ready: bool = True) -> ArtifactSummary:
    return ArtifactSummary(
        artifact_id=artifact_id,
        path=Path("/artifacts") / artifact_id,
        repo_id=REPO,
        profile=PROFILE,
        recommended_device=device,
        created_at=datetime(2026, 4, 1, tzinfo=UTC),
        ready=ready,
        pipeline=PipelineKind.LLM,
    )


class Harness:
    """A service over fake sessions, with a model directory whose config.json the test writes."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.sessions: dict[str, FakeSession] = {}
        self.listing: list[ArtifactSummary] = []
        self.opened: list[tuple[str, str | None]] = []
        self.model_dir = tmp_path / "model"
        self.model_dir.mkdir()
        self.config_path = tmp_path / "opencode" / "opencode.json"
        self.records = ConnectionRecords(tmp_path / "data" / "opencode-connections.json")
        self.sessions[ARTIFACT] = FakeSession(self.model_dir)
        self.write_model_config({"max_position_embeddings": 40960})

    def write_model_config(self, payload: dict[str, Any]) -> None:
        (self.model_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")

    def add_session(self, artifact_id: str, **fields: Any) -> FakeSession:
        directory = self.tmp_path / f"dir-{artifact_id}"
        directory.mkdir(exist_ok=True)
        (directory / "config.json").write_text(
            '{"max_position_embeddings": 4096}', encoding="utf-8"
        )
        fields.setdefault("manifest", make_manifest(artifact_id))
        session = FakeSession(directory, **fields)
        self.sessions[artifact_id] = session
        return session

    def open_session(self, reference: str, device: str | None) -> FakeSession:
        self.opened.append((reference, device))
        return self.sessions[reference]

    def service(
        self, *, env: dict[str, str] | None = None, families: tuple[str, ...] = ("qwen3",)
    ) -> ConnectService:
        return ConnectService(
            open_session=self.open_session,
            artifacts=lambda: self.listing,
            tool_families=list(families),
            records=self.records,
            default_max_new_tokens=4096,
            env=env or {},
        )

    def plan(
        self,
        service: ConnectService | None = None,
        *,
        reference: str = ARTIFACT,
        endpoint: Endpoint | None = None,
        device: str | None = None,
        context: int | None = None,
    ) -> Any:
        return (service or self.service()).plan_connect(
            reference,
            endpoint=endpoint or Endpoint(),
            device=device,
            context=context,
            config_path=self.config_path,
        )


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


def refusal(harness: Harness, **kwargs: Any) -> ConfigurationError:
    with pytest.raises(ConfigurationError) as caught:
        harness.plan(**kwargs)
    return caught.value


# --- tools -----------------------------------------------------------------------------


def test_tool_less_model_is_refused_before_the_config_is_read(harness: Harness) -> None:
    session = harness.sessions[ARTIFACT]
    session.tool_calling = None
    session.tool_calling_refusal = "No tool-call format is declared for qwen3_5."
    harness.add_session("other")
    harness.listing = [summary(ARTIFACT), summary("other")]
    harness.config_path.mkdir(parents=True)  # reading it would fail
    error = refusal(harness)
    assert error.diagnostic.code == "connect.opencode.tools_unsupported"
    assert "No tool-call format is declared for qwen3_5." in error.diagnostic.message
    assert "yonga connect opencode other" in " ".join(error.diagnostic.remediation)
    assert "yonga connect opencode " + ARTIFACT not in " ".join(error.diagnostic.remediation)


def test_tool_less_model_without_alternatives_names_families(harness: Harness) -> None:
    harness.sessions[ARTIFACT].tool_calling = None
    harness.listing = [summary(ARTIFACT)]
    error = refusal(harness, service=harness.service(families=("qwen2", "qwen3")))
    assert "qwen3" in " ".join(error.diagnostic.remediation)


def test_tool_less_suggestions_are_capped_at_three(harness: Harness) -> None:
    harness.sessions[ARTIFACT].tool_calling = None
    for index in range(5):
        harness.add_session(f"cap{index}")
    harness.listing = [summary(f"cap{index}") for index in range(5)]
    advice = " ".join(refusal(harness).diagnostic.remediation)
    assert advice.count("yonga connect opencode") == 3


# --- prompt length ---------------------------------------------------------------------


def test_npu_prompt_of_1024_is_refused_with_the_gpu_artifact(harness: Harness) -> None:
    session = harness.sessions[ARTIFACT]
    session.device = "NPU"
    session.runtime_properties = {"MAX_PROMPT_LEN": "1024"}
    harness.add_session("gpuid")
    harness.listing = [summary(ARTIFACT, device="NPU"), summary("gpuid")]
    error = refusal(harness)
    assert error.diagnostic.code == "connect.opencode.prompt_too_short"
    assert "1024" in error.diagnostic.message
    assert "8192" in error.diagnostic.message
    assert "yonga connect opencode gpuid" in " ".join(error.diagnostic.remediation)


def test_npu_prompt_without_gpu_artifact_suggests_prepare(harness: Harness) -> None:
    session = harness.sessions[ARTIFACT]
    session.device = "NPU"
    session.runtime_properties = {"MAX_PROMPT_LEN": "1024"}
    harness.listing = [summary(ARTIFACT, device="NPU")]
    error = refusal(harness)
    assert "yonga prepare Qwen/Qwen3-4B --device gpu" in " ".join(error.diagnostic.remediation)


def test_npu_prompt_of_8192_writes_input_limit(harness: Harness) -> None:
    session = harness.sessions[ARTIFACT]
    session.device = "NPU"
    session.runtime_properties = {"MAX_PROMPT_LEN": "8192"}
    plan = harness.plan()
    assert plan.entry["limit"] == {"context": 8192 + 4096, "input": 8192, "output": 4096}


# --- entry values ----------------------------------------------------------------------


def test_context_is_read_from_config_json(harness: Harness) -> None:
    assert harness.plan().entry["limit"]["context"] == 40960


def test_context_is_read_from_text_config(harness: Harness) -> None:
    harness.write_model_config({"text_config": {"max_position_embeddings": 32768}})
    assert harness.plan().entry["limit"]["context"] == 32768


def test_context_flag_overrides(harness: Harness) -> None:
    assert harness.plan(context=16384).entry["limit"]["context"] == 16384


def test_missing_context_is_refused(harness: Harness) -> None:
    harness.write_model_config({})
    error = refusal(harness)
    assert error.diagnostic.code == "connect.opencode.context_unknown"
    assert "--context" in " ".join(error.diagnostic.remediation)


def test_output_limit_comes_from_the_manifest(harness: Harness) -> None:
    harness.sessions[ARTIFACT].manifest = make_manifest(max_new_tokens=2048)
    assert harness.plan().entry["limit"]["output"] == 2048


def test_entry_shape(harness: Harness) -> None:
    plan = harness.plan()
    assert plan.action is ConnectAction.ADDED
    assert plan.provider_fields == {
        "npm": "@ai-sdk/openai-compatible",
        "name": "yonga (127.0.0.1:8000)",
        "options": {"baseURL": "http://127.0.0.1:8000/v1"},
    }
    assert plan.entry == {
        "name": "Qwen/Qwen3-4B · GPU · gpu-int4-asym-g128-r08",
        "tool_call": True,
        "limit": {"context": 40960, "output": 4096},
    }
    assert plan.document["$schema"] == "https://opencode.ai/config.json"
    assert plan.document["provider"]["yonga-8000"]["models"] == {ARTIFACT: plan.entry}
    assert plan.created_provider is True


def test_api_key_env_writes_placeholder_not_value(harness: Harness) -> None:
    plan = harness.plan(harness.service(env={"YONGA_API_KEY": "s3cret"}))
    assert plan.provider_fields["options"]["apiKey"] == "{env:YONGA_API_KEY}"
    assert "s3cret" not in render(plan.document)


def test_provider_key_for_other_host_and_ipv6() -> None:
    assert Endpoint("10.0.0.5", 9000).provider_key == "yonga-10-0-0-5-9000"
    assert Endpoint("::1", 8000).provider_key == "yonga-8000"
    assert Endpoint("fd00::1", 8000).base_url == "http://[fd00::1]:8000/v1"
    assert Endpoint("fd00::1", 8000).label == "[fd00::1]:8000"


def test_serve_command(harness: Harness) -> None:
    assert harness.plan().serve_command == f"yonga serve {ARTIFACT}"
    plan = harness.plan(endpoint=Endpoint("0.0.0.0", 8001), device="gpu")
    assert plan.serve_command == f"yonga serve {ARTIFACT} --port 8001 --host 0.0.0.0 --device gpu"
    assert harness.opened[-1] == (ARTIFACT, "gpu")


def test_ambiguous_reference_error_passes_through(harness: Harness) -> None:
    problem = ConfigurationError("More than one artifact matches 'abc'.")

    def failing(reference: str, device: str | None) -> FakeSession:
        raise problem

    service = ConnectService(
        open_session=failing,
        artifacts=lambda: [],
        tool_families=[],
        records=harness.records,
        default_max_new_tokens=4096,
        env={},
    )
    with pytest.raises(ConfigurationError) as caught:
        harness.plan(service, reference="abc")
    assert caught.value is problem


# --- outcomes and the file -------------------------------------------------------------


def test_apply_records_the_connection(harness: Harness) -> None:
    service = harness.service()
    plan = harness.plan(service)
    result = service.apply(plan)
    assert result.created_config is True
    assert result.record_written is True
    assert (
        json.loads(harness.config_path.read_text())["provider"]["yonga-8000"]["models"][ARTIFACT]
        == plan.entry
    )
    (record,) = harness.records.load()
    assert record.created_provider is True
    assert record.model == plan.entry
    assert record.provider == "yonga-8000"
    assert record.artifact_id == ARTIFACT
    assert record.config == str(harness.config_path.resolve())
    assert record.provider_fields == plan.provider_fields


def test_second_run_is_already_connected(harness: Harness) -> None:
    service = harness.service()
    service.apply(harness.plan(service))
    before = harness.config_path.stat().st_mtime_ns
    again = harness.plan(service)
    assert again.action is ConnectAction.ALREADY_CONNECTED
    assert service.apply(again) == service.apply(again)
    assert harness.config_path.stat().st_mtime_ns == before
    assert len(harness.records.load()) == 1


def test_changed_limits_are_updated_with_previous_entry(harness: Harness) -> None:
    service = harness.service()
    first = harness.plan(service)
    service.apply(first)
    harness.write_model_config({"max_position_embeddings": 8192})
    second = harness.plan(service)
    assert second.action is ConnectAction.UPDATED
    assert second.previous_entry == first.entry
    service.apply(second)
    (record,) = harness.records.load()
    assert record.model["limit"]["context"] == 8192


def test_second_artifact_in_provider_keeps_provider_untouched(harness: Harness) -> None:
    service = harness.service()
    service.apply(harness.plan(service))
    harness.add_session("second")
    plan = harness.plan(service, reference="second")
    assert plan.action is ConnectAction.ADDED
    assert plan.created_provider is False
    assert set(plan.document["provider"]["yonga-8000"]["models"]) == {ARTIFACT, "second"}


def test_same_artifact_on_second_port_adds_second_provider(harness: Harness) -> None:
    service = harness.service()
    service.apply(harness.plan(service))
    first = json.loads(harness.config_path.read_text())["provider"]["yonga-8000"]
    plan = harness.plan(service, endpoint=Endpoint(port=8001))
    assert plan.action is ConnectAction.ADDED
    service.apply(plan)
    providers = json.loads(harness.config_path.read_text())["provider"]
    assert set(providers) == {"yonga-8000", "yonga-8001"}
    assert providers["yonga-8000"] == first


def test_record_write_failure_is_reported_not_raised(harness: Harness, tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        if os.access(locked, os.W_OK):
            pytest.skip("directory permissions are not enforced for this user")
        records = ConnectionRecords(locked / "sub" / "opencode-connections.json")
        assert records.save([]) is False
        harness.records = records
        service = harness.service()
        result = service.apply(harness.plan(service))
        assert result.record_written is False
        assert harness.config_path.exists()
    finally:
        locked.chmod(0o700)


def test_unreadable_record_file_reads_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "records.json"
    assert ConnectionRecords(path).load() == []
    path.write_text("{ not json", encoding="utf-8")
    assert ConnectionRecords(path).load() == []


def test_jsonc_config_refusal_carries_the_snippet(harness: Harness) -> None:
    harness.config_path.parent.mkdir(parents=True)
    harness.config_path.write_text("{ // c\n}", encoding="utf-8")
    error = refusal(harness)
    assert error.diagnostic.code == "connect.opencode.config_not_json"
    advice = "\n".join(error.diagnostic.remediation)
    assert f'"{ARTIFACT}"' in advice
    assert "@ai-sdk/openai-compatible" in advice
    assert harness.config_path.read_text(encoding="utf-8") == "{ // c\n}"
