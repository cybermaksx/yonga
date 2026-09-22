"""Tests for secret redaction."""

from __future__ import annotations

import pytest

from yonga.util.redaction import (
    REDACTED,
    SECRET_ENV_KEYS,
    is_secret_key,
    redact_argv,
    redact_env,
    redact_mapping,
    redact_text,
)

HF_TOKEN = "hf_QWERTYuiop1234567890asdfghjklZXCV"
HF_OAUTH_TOKEN = "hf_oauth_QWERTYuiop1234567890asdfghjkl"
SK_KEY = "sk-proj-AbCdEf0123456789GhIjKlMnOpQr"
GH_TOKEN = "ghp_AbCdEf0123456789GhIjKlMnOpQrStUv"


def test_secret_env_keys_cover_the_contract() -> None:
    for name in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "HF_API_TOKEN",
        "YONGA_HF_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
    ):
        assert name in SECRET_ENV_KEYS


@pytest.mark.parametrize(
    "text",
    [
        f"downloading with token {HF_TOKEN}",
        f"HF_TOKEN={HF_TOKEN}",
        f"oauth flow returned {HF_OAUTH_TOKEN}",
    ],
)
def test_hf_tokens_are_masked(text: str) -> None:
    result = redact_text(text)

    assert HF_TOKEN not in result
    assert HF_OAUTH_TOKEN not in result
    assert REDACTED in result


def test_bearer_header_is_masked() -> None:
    text = f"Authorization: Bearer {HF_TOKEN}"
    result = redact_text(text)

    assert HF_TOKEN not in result
    assert "Authorization" in result
    assert REDACTED in result


def test_bare_bearer_scheme_is_masked() -> None:
    result = redact_text("sending Bearer abcdefghijklmnop to the hub")

    assert "abcdefghijklmnop" not in result
    assert REDACTED in result


def test_token_query_parameter_is_masked() -> None:
    url = f"https://huggingface.co/api/models/org/model?token={HF_TOKEN}&full=true"
    result = redact_text(url)

    assert HF_TOKEN not in result
    assert "full=true" in result
    assert "token=" in result


def test_url_userinfo_is_masked() -> None:
    result = redact_text("cloning https://alice:sup3rs3cret@example.invalid/repo.git")

    assert "sup3rs3cret" not in result
    assert "alice" not in result
    assert "https://" in result
    assert "example.invalid" in result


def test_sk_api_key_is_masked() -> None:
    result = redact_text(f"key {SK_KEY} rejected")

    assert SK_KEY not in result
    assert REDACTED in result


def test_github_token_is_masked() -> None:
    result = redact_text(f"remote refused {GH_TOKEN}")

    assert GH_TOKEN not in result


def test_extra_secrets_are_masked() -> None:
    secret = "a-very-private-literal"
    result = redact_text(f"value is {secret} here", extra_secrets=[secret])

    assert secret not in result
    assert REDACTED in result


def test_short_extra_secrets_are_ignored() -> None:
    result = redact_text("the model is gpu-ready", extra_secrets=["gpu"])

    assert result == "the model is gpu-ready"


def test_ordinary_text_is_untouched() -> None:
    text = "exported qwen3_5 with max_new_tokens=4096 to /home/user/.local/share/yonga"

    assert redact_text(text) == text


def test_redact_text_never_raises_on_odd_input() -> None:
    assert redact_text("") == ""
    assert redact_text("\x00￿") == "\x00￿"
    assert isinstance(redact_text("x", extra_secrets=["short", None]), str)  # type: ignore[list-item]


def test_redact_env_masks_by_key_and_by_value() -> None:
    env = {
        "HF_TOKEN": HF_TOKEN,
        "HUGGINGFACEHUB_API_TOKEN": HF_TOKEN,
        "MY_PASSWORD": "hunter2",
        "PATH": "/usr/bin",
        "SOME_URL": f"https://x.invalid?token={HF_TOKEN}",
    }
    result = redact_env(env)

    assert result["HF_TOKEN"] == REDACTED
    assert result["HUGGINGFACEHUB_API_TOKEN"] == REDACTED
    assert result["MY_PASSWORD"] == REDACTED
    assert result["PATH"] == "/usr/bin"
    assert HF_TOKEN not in result["SOME_URL"]
    assert HF_TOKEN not in repr(result)


def test_redact_argv_masks_split_and_inline_forms() -> None:
    argv = [
        "optimum-cli",
        "export",
        "openvino",
        "--token",
        HF_TOKEN,
        f"--hf-token={HF_TOKEN}",
        "--weight-format",
        "int4",
    ]
    result = redact_argv(argv)

    assert HF_TOKEN not in " ".join(result)
    assert result[3] == "--token"
    assert result[4] == REDACTED
    assert result[5] == f"--hf-token={REDACTED}"
    assert result[-2:] == ["--weight-format", "int4"]


def test_redact_argv_masks_tokens_appearing_in_free_arguments() -> None:
    result = redact_argv(["curl", f"https://x.invalid/?token={HF_TOKEN}"])

    assert HF_TOKEN not in " ".join(result)


def test_redact_mapping_is_recursive_and_key_based() -> None:
    data: dict[str, object] = {
        "hf_token": HF_TOKEN,
        "nested": {"authorization": f"Bearer {HF_TOKEN}", "safe": "value"},
        "items": [f"see {HF_TOKEN}", 3, None, True],
        "max_new_tokens": 4096,
        "path": "/home/user/model",
    }
    result = redact_mapping(data)

    assert result["hf_token"] == REDACTED
    assert result["max_new_tokens"] == 4096
    assert result["path"] == "/home/user/model"
    assert HF_TOKEN not in repr(result)
    nested = result["nested"]
    assert isinstance(nested, dict)
    assert nested["authorization"] == REDACTED
    assert nested["safe"] == "value"


def test_redact_mapping_never_raises_on_odd_keys() -> None:
    result = redact_mapping({1: "x", "ok": object()})  # type: ignore[dict-item]

    assert "1" in result
    assert "ok" in result


@pytest.mark.parametrize(
    ("key", "secret"),
    [
        ("hf_token", True),
        ("HF_TOKEN", True),
        ("password", True),
        ("Authorization", True),
        ("api_key", True),
        ("max_new_tokens", False),
        ("output_tokens", False),
        ("tokenizer", False),
        ("bench_runs", False),
    ],
)
def test_is_secret_key(key: str, secret: bool) -> None:
    assert is_secret_key(key) is secret
