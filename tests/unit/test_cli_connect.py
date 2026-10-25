"""``yonga connect opencode``: the question, the documents and the exit codes."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.unit.test_cli import FakeWiring, as_tty, hub_with, invoke, one_document
from tests.unit.test_connect_service import ARTIFACT, FakeSession, make_manifest, summary
from yonga.application.connect import ConnectionRecords, ConnectService
from yonga.application.wiring import Wiring
from yonga.config.settings import Settings
from yonga.domain.diagnostics import ExitCode

PROVIDER = "yonga-8000"


class ConnectWiring(FakeWiring):
    """The usual fakes, with a connect service over fake sessions."""

    def __init__(self, settings: Settings, sessions: dict[str, FakeSession], **kwargs: Any) -> None:
        super().__init__(settings, **kwargs)
        self.sessions = sessions
        self.listing = [summary(name) for name in sessions]

    def connect_service(self) -> ConnectService:
        return ConnectService(
            open_session=lambda reference, device: self.sessions[reference],
            artifacts=lambda: self.listing,
            tool_families=["qwen3"],
            records=ConnectionRecords(self.paths.data_dir / "opencode-connections.json"),
            default_max_new_tokens=4096,
            env={},
        )


def session_in(tmp_path: Path, artifact_id: str = ARTIFACT, **fields: Any) -> FakeSession:
    directory = tmp_path / f"model-{artifact_id}"
    directory.mkdir(exist_ok=True)
    (directory / "config.json").write_text('{"max_position_embeddings": 40960}', encoding="utf-8")
    fields.setdefault("manifest", make_manifest(artifact_id))
    return FakeSession(directory, **fields)


def squeeze(text: str) -> str:
    """``text`` without any whitespace: rich wraps long paths mid-word."""
    return "".join(text.split())


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def home(tmp_path: Path) -> Path:
    root = tmp_path / "home"
    (root / "config").mkdir(parents=True)
    return root


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return tmp_path / "opencode" / "opencode.json"


def wiring_with(tmp_path: Path, **sessions: FakeSession) -> Callable[[Settings], Wiring]:
    found = sessions or {ARTIFACT: session_in(tmp_path)}
    return lambda settings: ConnectWiring(settings, found)


def connect(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    config: Path,
    *args: str,
    stdin: str | None = None,
    sessions: dict[str, FakeSession] | None = None,
) -> Any:
    factory = wiring_with(tmp_path, **(sessions or {}))
    return invoke(
        runner,
        home,
        ["connect", "opencode", ARTIFACT, "--config", str(config), *args],
        factory=factory,
        stdin=stdin,
    )


def test_connect_asks_then_writes(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = as_tty(monkeypatch, answer=True)
    result = connect(runner, home, tmp_path, config)
    assert result.exit_code == 0, result.output
    document = json.loads(config.read_text(encoding="utf-8"))
    assert document["$schema"] == "https://opencode.ai/config.json"
    assert ARTIFACT in document["provider"][PROVIDER]["models"]
    assert "yonga serve" in result.stdout
    assert asked == [f"Write this to {config}?"]
    assert "+ model " in result.stdout


def test_connect_declined_writes_nothing_and_exits_sixty(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_tty(monkeypatch, answer=False)
    result = connect(runner, home, tmp_path, config)
    assert result.exit_code == int(ExitCode.CANCELLED)
    assert not config.exists()


def test_connect_json_without_yes_exits_sixty(
    runner: CliRunner, home: Path, tmp_path: Path, config: Path
) -> None:
    result = connect(runner, home, tmp_path, config, "--json")
    assert result.exit_code == int(ExitCode.CANCELLED)
    assert not config.exists()


def test_connect_json_with_yes_emits_one_document(
    runner: CliRunner, home: Path, tmp_path: Path, config: Path
) -> None:
    result = connect(runner, home, tmp_path, config, "--json", "--yes")
    assert result.exit_code == 0, result.output
    payload = one_document(result)
    assert set(payload) == {
        "ok",
        "action",
        "config",
        "created_config",
        "provider",
        "created_provider",
        "model",
        "base_url",
        "limit",
        "serve_command",
        "exit_code",
    }
    assert payload["action"] == "added"
    assert payload["config"] == str(config)
    assert payload["created_config"] is True
    assert payload["provider"] == PROVIDER
    assert payload["model"] == ARTIFACT
    assert payload["base_url"] == "http://127.0.0.1:8000/v1"
    assert payload["exit_code"] == 0


def test_connect_already_connected_asks_nothing(
    runner: CliRunner, home: Path, tmp_path: Path, config: Path
) -> None:
    assert connect(runner, home, tmp_path, config, "--yes").exit_code == 0
    before = config.read_text(encoding="utf-8")
    result = connect(runner, home, tmp_path, config)
    assert result.exit_code == 0, result.output
    assert f"{ARTIFACT} is already connected in {config}." in result.stdout
    assert "Write this to" not in result.stdout
    assert config.read_text(encoding="utf-8") == before


def test_connect_updated_shows_before_and_after(
    runner: CliRunner, home: Path, tmp_path: Path, config: Path
) -> None:
    assert connect(runner, home, tmp_path, config, "--yes").exit_code == 0
    result = connect(runner, home, tmp_path, config, "--yes", "--context", "9999")
    assert result.exit_code == 0, result.output
    assert f"~ model {ARTIFACT}" in result.stdout
    assert "before" in result.stdout
    assert "after" in result.stdout
    models = json.loads(config.read_text(encoding="utf-8"))["provider"][PROVIDER]["models"]
    assert models[ARTIFACT]["limit"]["context"] == 9999


def test_connect_into_a_changed_provider_shows_the_provider_change(
    runner: CliRunner, home: Path, tmp_path: Path, config: Path
) -> None:
    config.parent.mkdir(parents=True)
    provider = {"npm": "other", "name": "Mine", "options": {"timeout": 5}, "models": {}}
    config.write_text(json.dumps({"provider": {PROVIDER: provider}}), encoding="utf-8")
    result = connect(runner, home, tmp_path, config, "--yes")
    assert result.exit_code == 0, result.output
    out = squeeze(result.stdout)
    assert squeeze(f"~ provider.{PROVIDER}") in out
    assert squeeze('before {"npm": "other", "options": {}}') in out
    assert squeeze(f"+ model {ARTIFACT}") in out
    written = json.loads(config.read_text(encoding="utf-8"))["provider"][PROVIDER]
    assert written["name"] == "Mine"
    assert written["options"] == {"timeout": 5, "baseURL": "http://127.0.0.1:8000/v1"}


def test_connect_updated_provider_only_shows_no_model_change(
    runner: CliRunner, home: Path, tmp_path: Path, config: Path
) -> None:
    assert connect(runner, home, tmp_path, config, "--yes").exit_code == 0
    document = json.loads(config.read_text(encoding="utf-8"))
    document["provider"][PROVIDER]["npm"] = "other"
    config.write_text(json.dumps(document), encoding="utf-8")
    result = connect(runner, home, tmp_path, config, "--yes", "--json")
    assert one_document(result)["action"] == "updated"
    result = connect(runner, home, tmp_path, config, "--yes")
    assert "already connected" in result.stdout
    config.write_text(json.dumps(document), encoding="utf-8")
    result = connect(runner, home, tmp_path, config, "--yes")
    out = squeeze(result.stdout)
    assert squeeze(f"~ provider.{PROVIDER}") in out
    assert "~ model" not in result.stdout
    assert json.loads(config.read_text(encoding="utf-8"))["provider"][PROVIDER]["npm"] == (
        "@ai-sdk/openai-compatible"
    )


def test_connect_tool_less_exits_seventy_with_suggestion(
    runner: CliRunner, home: Path, tmp_path: Path, config: Path
) -> None:
    other = "feedfacecafe0001"
    sessions = {
        ARTIFACT: session_in(tmp_path, tool_calling=None, tool_calling_refusal="No tools."),
        other: session_in(tmp_path, other),
    }
    result = connect(runner, home, tmp_path, config, "--yes", sessions=sessions)
    assert result.exit_code == int(ExitCode.REGISTRY)
    assert f"connect opencode {other}" in " ".join(result.stderr.split())
    assert not config.exists()


@pytest.mark.parametrize(
    "option", [["--host", "10.0.0.2"], ["--port", "9000"], ["--device", "gpu"], ["--context", "1"]]
)
def test_remove_flag_rejects_connection_options(
    runner: CliRunner, home: Path, tmp_path: Path, config: Path, option: list[str]
) -> None:
    result = connect(runner, home, tmp_path, config, "--remove", "--yes", *option)
    assert result.exit_code == int(ExitCode.REGISTRY)
    assert option[0] in result.stderr


def test_remove_flag_rejects_port(
    runner: CliRunner, home: Path, tmp_path: Path, config: Path
) -> None:
    result = connect(runner, home, tmp_path, config, "--remove", "--port", "8000")
    assert result.exit_code == int(ExitCode.REGISTRY)
    assert "--port" in result.stderr


def test_connect_remove_round_trip(
    runner: CliRunner, home: Path, tmp_path: Path, config: Path
) -> None:
    assert connect(runner, home, tmp_path, config, "--yes").exit_code == 0
    result = connect(runner, home, tmp_path, config, "--remove", "--yes", "--json")
    assert result.exit_code == 0, result.output
    payload = one_document(result)
    assert set(payload) == {"ok", "action", "config", "removed", "kept", "exit_code"}
    assert payload["removed"] == [{"provider": PROVIDER, "model": ARTIFACT}]
    assert payload["kept"] == []
    document = json.loads(config.read_text(encoding="utf-8"))
    assert PROVIDER not in document.get("provider", {})
    assert document["$schema"] == "https://opencode.ai/config.json"
    again = connect(runner, home, tmp_path, config, "--remove", "--yes")
    assert again.exit_code == 0
    assert "is not connected" in again.stdout


def test_connect_remove_declined_removes_nothing(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert connect(runner, home, tmp_path, config, "--yes").exit_code == 0
    as_tty(monkeypatch, answer=False)
    result = connect(runner, home, tmp_path, config, "--remove")
    assert result.exit_code == int(ExitCode.CANCELLED)
    assert PROVIDER in json.loads(config.read_text(encoding="utf-8"))["provider"]


def test_remove_kept_entry_exits_seventy(
    runner: CliRunner, home: Path, tmp_path: Path, config: Path
) -> None:
    assert connect(runner, home, tmp_path, config, "--yes").exit_code == 0
    document = json.loads(config.read_text(encoding="utf-8"))
    document["provider"][PROVIDER]["models"][ARTIFACT]["name"] = "mine"
    config.write_text(json.dumps(document), encoding="utf-8")
    result = connect(runner, home, tmp_path, config, "--remove", "--yes", "--json")
    assert result.exit_code == int(ExitCode.REGISTRY)
    payload = one_document(result)
    assert payload["exit_code"] == int(ExitCode.REGISTRY)
    assert payload["removed"] == []
    assert payload["kept"] == [
        {"provider": PROVIDER, "model": ARTIFACT, "reason": "edited since yonga wrote it"}
    ]
    text = connect(runner, home, tmp_path, config, "--remove", "--yes")
    assert text.exit_code == int(ExitCode.REGISTRY)
    assert "edited since yonga wrote it" in text.stdout


def test_artifact_remove_prints_the_opencode_note(
    runner: CliRunner, home: Path, tmp_path: Path, config: Path
) -> None:
    def factory(settings: Settings) -> Wiring:
        return ConnectWiring(settings, {}, hub=hub_with(tmp_path=tmp_path))

    prepared = invoke(
        runner,
        home,
        [
            "prepare",
            "nDimensional/Qwen3.5-9B-Uncensored-Safetensors",
            "--device",
            "cpu",
            "--yes",
            "--json",
        ],
        factory=factory,
    )
    document = one_document(prepared)
    artifact = Path(str(document["artifact_dir"]))
    artifact_id = str(document["artifact_id"])
    sessions = {artifact_id: session_in(tmp_path, artifact_id)}

    def connected(settings: Settings) -> Wiring:
        return ConnectWiring(settings, sessions, hub=hub_with(tmp_path=tmp_path))

    done = invoke(
        runner,
        home,
        ["connect", "opencode", artifact_id, "--config", str(config), "--yes"],
        factory=connected,
    )
    assert done.exit_code == 0, done.output
    removed = invoke(runner, home, ["remove", str(artifact), "--yes"], factory=connected)
    assert removed.exit_code == 0, removed.output
    assert "still listed in OpenCode" in removed.stdout
    assert f"connect opencode --remove {artifact_id}" in " ".join(removed.stdout.split())


def test_artifact_remove_without_a_connection_prints_no_note(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    def factory(settings: Settings) -> Wiring:
        return ConnectWiring(settings, {}, hub=hub_with(tmp_path=tmp_path))

    prepared = invoke(
        runner,
        home,
        [
            "prepare",
            "nDimensional/Qwen3.5-9B-Uncensored-Safetensors",
            "--device",
            "cpu",
            "--yes",
            "--json",
        ],
        factory=factory,
    )
    artifact = Path(str(one_document(prepared)["artifact_dir"]))
    removed = invoke(runner, home, ["remove", str(artifact), "--yes"], factory=factory)
    assert removed.exit_code == 0
    assert "OpenCode" not in removed.stdout
