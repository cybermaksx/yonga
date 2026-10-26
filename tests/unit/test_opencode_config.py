"""The OpenCode config adapter: strict reading, pure edits, atomic writing."""

from __future__ import annotations

import errno
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from yonga.adapters.opencode_config import (
    OPENCODE_SCHEMA_URL,
    check_provider,
    default_config_path,
    merge_provider_fields,
    provider_fields_in,
    read_config,
    render,
    with_model,
    without_model,
    write_config,
)
from yonga.domain.diagnostics import ConfigurationError, ExitCode, YongaError

FIELDS: dict[str, Any] = {
    "npm": "@ai-sdk/openai-compatible",
    "name": "yonga",
    "options": {"baseURL": "http://127.0.0.1:8000/v1"},
}
ENTRY: dict[str, Any] = {"name": "Qwen"}


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# --- reading ------------------------------------------------------------
def test_missing_file_reads_as_empty_document(tmp_path: Path) -> None:
    snapshot = read_config(tmp_path / "opencode.json")
    assert snapshot.raw is None
    assert snapshot.document == {}
    assert snapshot.mode is None


def test_whitespace_only_file_reads_as_empty_document(tmp_path: Path) -> None:
    snapshot = read_config(_write(tmp_path / "opencode.json", " \n\t\n"))
    assert snapshot.document == {}
    assert snapshot.raw == b" \n\t\n"


def test_jsonc_comment_is_refused_and_file_untouched(tmp_path: Path) -> None:
    path = _write(tmp_path / "opencode.json", "{ // x\n}")
    with pytest.raises(ConfigurationError) as caught:
        read_config(path)
    assert caught.value.diagnostic.code == "connect.opencode.config_not_json"
    assert path.read_bytes() == b"{ // x\n}"


def test_trailing_comma_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "opencode.json", '{"a": 1,}')
    with pytest.raises(ConfigurationError) as caught:
        read_config(path)
    assert caught.value.diagnostic.code == "connect.opencode.config_not_json"


def test_non_utf8_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "opencode.json"
    path.write_bytes(b"\xff\xfe{}")
    with pytest.raises(ConfigurationError) as caught:
        read_config(path)
    assert caught.value.diagnostic.code == "connect.opencode.config_not_json"


def test_non_object_root_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "opencode.json", "[]")
    with pytest.raises(ConfigurationError) as caught:
        read_config(path)
    assert caught.value.diagnostic.code == "connect.opencode.config_not_json"


def test_jsonc_name_with_plain_json_is_read(tmp_path: Path) -> None:
    snapshot = read_config(_write(tmp_path / "opencode.jsonc", '{"theme": "x"}'))
    assert snapshot.document == {"theme": "x"}


def test_symlink_is_resolved(tmp_path: Path) -> None:
    target = _write(tmp_path / "real.json", "{}")
    link = tmp_path / "opencode.json"
    link.symlink_to(target)
    assert read_config(link).path == target.resolve()


def test_check_provider_refuses_null_provider() -> None:
    with pytest.raises(ConfigurationError) as caught:
        check_provider({"provider": None}, "yonga-8000")
    assert "provider" in caught.value.diagnostic.message


def test_check_provider_refuses_list_models() -> None:
    with pytest.raises(ConfigurationError) as caught:
        check_provider({"provider": {"yonga-8000": {"models": []}}}, "yonga-8000")
    assert "provider.yonga-8000.models" in caught.value.diagnostic.message


def test_check_provider_refuses_non_object_entry() -> None:
    with pytest.raises(ConfigurationError) as caught:
        check_provider({"provider": {"yonga-8000": "x"}}, "yonga-8000")
    assert "provider.yonga-8000" in caught.value.diagnostic.message


def test_check_provider_accepts_missing_and_valid() -> None:
    check_provider({}, "yonga-8000")
    check_provider({"provider": {"yonga-8000": {"models": {}}}}, "yonga-8000")


# --- pure edits ---------------------------------------------------------
def test_with_model_on_empty_document_puts_schema_first() -> None:
    doc = with_model({}, "yonga-8000", FIELDS, "qwen", ENTRY)
    assert list(doc) == ["$schema", "provider"]
    assert doc["$schema"] == OPENCODE_SCHEMA_URL
    provider = doc["provider"]["yonga-8000"]
    assert list(provider) == ["npm", "name", "options", "models"]
    assert provider["models"] == {"qwen": ENTRY}


def test_with_model_keeps_other_keys_and_order() -> None:
    original: dict[str, Any] = {
        "theme": "x",
        "provider": {"other": {"name": "o", "models": {"m": {}}}},
        "model": "a/b",
    }
    doc = with_model(original, "yonga-8000", FIELDS, "qwen", ENTRY)
    assert list(doc) == ["theme", "provider", "model"]
    assert list(doc["provider"]) == ["other", "yonga-8000"]
    assert doc["provider"]["other"] == original["provider"]["other"]
    assert "yonga-8000" not in original["provider"]


def test_with_model_twice_is_identical() -> None:
    once = with_model({}, "yonga-8000", FIELDS, "qwen", ENTRY)
    twice = with_model(once, "yonga-8000", FIELDS, "qwen", ENTRY)
    assert render(once) == render(twice)


def test_with_model_replaces_entry_and_provider_fields() -> None:
    first = with_model({}, "yonga-8000", FIELDS, "qwen", ENTRY)
    fields = {**FIELDS, "options": {"baseURL": "http://127.0.0.1:9000/v1"}}
    second = with_model(first, "yonga-8000", fields, "qwen", {"name": "New"})
    provider = second["provider"]["yonga-8000"]
    assert provider["options"]["baseURL"] == "http://127.0.0.1:9000/v1"
    assert provider["models"] == {"qwen": {"name": "New"}}
    assert first["provider"]["yonga-8000"]["models"]["qwen"] == ENTRY


def test_with_model_merges_into_an_existing_provider_key_by_key() -> None:
    original: dict[str, Any] = {
        "provider": {
            "yonga-8000": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "My local",
                "options": {
                    "baseURL": "http://127.0.0.1:8000/v1",
                    "timeout": 600000,
                    "headers": {"X-A": "1"},
                    "apiKey": "{env:YONGA_API_KEY}",
                },
                "models": {"mine": {"name": "x"}},
            }
        }
    }
    fields = {"npm": "@ai-sdk/openai-compatible", "options": {"baseURL": "http://h:1/v1"}}
    provider = with_model(original, "yonga-8000", fields, "qwen", ENTRY)["provider"]["yonga-8000"]
    assert provider["name"] == "My local"
    assert provider["options"] == {
        "baseURL": "http://h:1/v1",
        "timeout": 600000,
        "headers": {"X-A": "1"},
        "apiKey": "{env:YONGA_API_KEY}",
    }
    assert list(provider["models"]) == ["mine", "qwen"]


def test_merge_and_projection_of_provider_fields() -> None:
    current: dict[str, Any] = {"npm": "x", "name": "n", "options": {"timeout": 1}}
    fields = {"npm": "y", "options": {"baseURL": "u"}}
    assert provider_fields_in(current, fields) == {"npm": "x", "options": {}}
    assert merge_provider_fields(current, fields) == {
        "npm": "y",
        "name": "n",
        "options": {"timeout": 1, "baseURL": "u"},
    }
    assert current == {"npm": "x", "name": "n", "options": {"timeout": 1}}
    assert provider_fields_in({}, fields) == {}


def test_without_model_keeps_provider_with_other_models() -> None:
    doc = with_model({}, "yonga-8000", FIELDS, "a", ENTRY)
    doc = with_model(doc, "yonga-8000", FIELDS, "b", ENTRY)
    out = without_model(doc, "yonga-8000", "a", drop_provider_when=FIELDS)
    assert list(out["provider"]["yonga-8000"]["models"]) == ["b"]


def test_without_model_drops_emptied_provider_when_fields_match() -> None:
    doc = with_model({"theme": "x"}, "yonga-8000", FIELDS, "a", ENTRY)
    out = without_model(doc, "yonga-8000", "a", drop_provider_when=FIELDS)
    assert out["provider"] == {}
    assert out["theme"] == "x"


def test_without_model_keeps_emptied_provider_when_fields_differ() -> None:
    doc = with_model({}, "yonga-8000", FIELDS, "a", ENTRY)
    doc["provider"]["yonga-8000"]["options"]["apiKey"] = "k"
    out = without_model(doc, "yonga-8000", "a", drop_provider_when=FIELDS)
    assert out["provider"]["yonga-8000"]["models"] == {}


def test_without_model_keeps_provider_without_drop_condition() -> None:
    doc = with_model({}, "yonga-8000", FIELDS, "a", ENTRY)
    out = without_model(doc, "yonga-8000", "a", drop_provider_when=None)
    assert out["provider"]["yonga-8000"]["models"] == {}


# --- writing ------------------------------------------------------------
def test_write_creates_file_with_mode_0600_and_newline(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "opencode.json"
    snapshot = read_config(path)
    document = with_model({}, "yonga-8000", FIELDS, "qwen", ENTRY)
    assert write_config(snapshot, document) is True
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text) == document
    assert list(tmp_path.joinpath("nested").iterdir()) == [path]


def test_write_keeps_existing_mode(tmp_path: Path) -> None:
    path = _write(tmp_path / "opencode.json", "{}")
    path.chmod(0o644)
    snapshot = read_config(path)
    assert write_config(snapshot, {"theme": "x"}) is False
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_write_refuses_file_changed_since_read(tmp_path: Path) -> None:
    path = _write(tmp_path / "opencode.json", "{}")
    snapshot = read_config(path)
    path.write_text('{"edited": true}', encoding="utf-8")
    with pytest.raises(ConfigurationError) as caught:
        write_config(snapshot, {"theme": "x"})
    assert caught.value.diagnostic.code == "connect.opencode.config_changed"
    assert path.read_text(encoding="utf-8") == '{"edited": true}'


def test_write_refuses_file_created_since_read(tmp_path: Path) -> None:
    path = tmp_path / "opencode.json"
    snapshot = read_config(path)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigurationError) as caught:
        write_config(snapshot, {"theme": "x"})
    assert caught.value.diagnostic.code == "connect.opencode.config_changed"
    assert path.read_text(encoding="utf-8") == "{}"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_write_in_unwritable_dir_exits_ten(tmp_path: Path) -> None:
    directory = tmp_path / "locked"
    directory.mkdir()
    snapshot = read_config(directory / "opencode.json")
    directory.chmod(0o500)
    try:
        with pytest.raises(YongaError) as caught:
            write_config(snapshot, {"theme": "x"})
    finally:
        directory.chmod(0o700)
    assert caught.value.diagnostic.code == "connect.opencode.config_not_writable"
    assert caught.value.diagnostic.exit_code() == ExitCode.ENVIRONMENT_UNUSABLE


def test_write_through_symlink_updates_target(tmp_path: Path) -> None:
    target = _write(tmp_path / "real.json", "{}")
    link = tmp_path / "opencode.json"
    link.symlink_to(target)
    write_config(read_config(link), {"theme": "x"})
    assert link.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8")) == {"theme": "x"}


# --- default path -------------------------------------------------------
def test_default_config_path_honours_xdg_config_home() -> None:
    env = {"XDG_CONFIG_HOME": "/x", "HOME": "/h"}
    assert default_config_path(env) == Path("/x/opencode/opencode.json")


def test_default_config_path_falls_back_to_home() -> None:
    assert default_config_path({"HOME": "/h"}) == Path("/h/.config/opencode/opencode.json")


# --- operating system failures -----------------------------------------
def test_directory_as_config_path_is_not_a_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError) as caught:
        read_config(tmp_path)
    assert caught.value.diagnostic.code == "connect.opencode.config_not_a_file"


def test_read_only_filesystem_on_write_exits_ten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = read_config(tmp_path / "opencode.json")

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.EROFS, "Read-only file system")

    monkeypatch.setattr(os, "replace", refuse)
    monkeypatch.setattr(Path, "replace", refuse)
    with pytest.raises(YongaError) as caught:
        write_config(snapshot, {"theme": "x"})
    diagnostic = caught.value.diagnostic
    assert diagnostic.code == "connect.opencode.config_not_writable"
    assert diagnostic.exit_code() == ExitCode.ENVIRONMENT_UNUSABLE
    assert "Read-only file system" in diagnostic.message
    assert not (tmp_path / "opencode.json").exists()
    assert list(tmp_path.iterdir()) == []


def test_full_disk_on_write_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = read_config(tmp_path / "opencode.json")

    def full(*_args: object, **_kwargs: object) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "fsync", full)
    with pytest.raises(YongaError) as caught:
        write_config(snapshot, {"theme": "x"})
    assert caught.value.diagnostic.code == "connect.opencode.config_not_writable"
    assert "No space left on device" in caught.value.diagnostic.message
    assert list(tmp_path.iterdir()) == []


def test_with_model_merges_into_an_existing_model_entry() -> None:
    original: dict[str, Any] = {
        "provider": {
            "yonga-8000": {
                "models": {
                    "qwen": {
                        "name": "old",
                        "reasoning": True,
                        "options": {"temperature": 0.2},
                        "limit": {"context": 1, "extra": 2},
                    }
                }
            }
        }
    }
    entry = {"name": "New", "tool_call": True, "limit": {"context": 9, "output": 3}}
    model = with_model(original, "yonga-8000", FIELDS, "qwen", entry)["provider"]["yonga-8000"][
        "models"
    ]["qwen"]
    assert model == {
        "name": "New",
        "reasoning": True,
        "options": {"temperature": 0.2},
        "limit": {"context": 9, "output": 3},
        "tool_call": True,
    }
    assert list(model)[:3] == ["name", "reasoning", "options"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_write_refuses_a_read_only_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "opencode.json", "{}")
    path.chmod(0o444)
    snapshot = read_config(path)
    with pytest.raises(YongaError) as caught:
        write_config(snapshot, {"theme": "x"})
    diagnostic = caught.value.diagnostic
    assert diagnostic.code == "connect.opencode.config_not_writable"
    assert diagnostic.exit_code() == ExitCode.ENVIRONMENT_UNUSABLE
    assert "read-only" in " ".join(diagnostic.remediation)
    assert path.read_text(encoding="utf-8") == "{}"
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert list(tmp_path.iterdir()) == [path]
