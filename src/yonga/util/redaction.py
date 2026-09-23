"""Secret redaction for everything the tool prints, logs or serialises.

Every function here is defensive: redaction runs on failure paths, so it must
never raise and must fail closed (drop the value) rather than leak it.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

REDACTED: str = "***REDACTED***"
"""Replacement text written in place of any secret value."""

MIN_SECRET_LENGTH: int = 8
"""Literals shorter than this are ignored in ``extra_secrets`` to avoid masking
ordinary words such as ``true`` or a short model name."""

SECRET_ENV_KEYS: frozenset[str] = frozenset(
    {
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "HF_API_TOKEN",
        "YONGA_HF_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AZURE_CLIENT_SECRET",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GITLAB_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "NGC_API_KEY",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
    }
)
"""Environment variables whose *value* is always masked, whatever it looks like."""

SECRET_KEY_SEGMENTS: frozenset[str] = frozenset(
    {
        "token",
        "secret",
        "secrets",
        "password",
        "passwd",
        "pwd",
        "apikey",
        "credential",
        "credentials",
        "authorization",
    }
)
"""Whole word-ish key segments that make a mapping entry secret-bearing.

Matching on segments rather than substrings keeps ordinary keys such as
``max_new_tokens`` or ``output_tokens`` readable in diagnostics output.
"""

NON_SECRET_ENV_KEYS: frozenset[str] = frozenset({"PWD", "OLDPWD"})
"""Variables whose names match a secret segment but which never hold a secret.

``PWD``/``OLDPWD`` match the ``pwd`` segment and hold the working directory. They
stay masked in :func:`redact_env` (harmless), but :func:`environment_secrets`
must not turn a directory path into a literal that is blanked out everywhere.
"""

SECRET_KEY_PHRASES: tuple[str, ...] = (
    "api_key",
    "access_key",
    "private_key",
    "session_key",
    "signing_key",
)
"""Multi-word key names that are secret even though no single segment is."""

SAFE_KEY_NAMES: frozenset[str] = frozenset(
    {
        "token_source",
        "token_count",
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "max_new_tokens",
        # Model/tokenizer config fields. These are small integers or short
        # literal strings published in every public model repository, and they
        # have to stay readable in `inspect` output and artifact manifests.
        "bos_token",
        "eos_token",
        "pad_token",
        "unk_token",
        "sep_token",
        "cls_token",
        "mask_token",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "sep_token_id",
        "cls_token_id",
        "mask_token_id",
        "decoder_start_token_id",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
        "vision_token_id",
        "added_tokens_decoder",
        "special_tokens_map_file",
        "additional_special_tokens",
        "tokenizer_class",
        "clean_up_tokenization_spaces",
    }
)
"""Key names that look secret-ish but only ever carry non-secret metadata."""

_KEY_SPLIT = re.compile(r"[^a-z0-9]+")

SECRET_ARG_NAMES: frozenset[str] = frozenset(
    {
        "--token",
        "--hf-token",
        "--hf_token",
        "--auth-token",
        "--api-key",
        "--api_key",
        "--apikey",
        "--password",
        "--secret",
        "--access-token",
        "--use-auth-token",
    }
)
"""Command line options whose following value must be masked."""

_URL_USERINFO = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*://)(?P<userinfo>[^/\s@]+)@")
_AUTH_HEADER = re.compile(
    r"(?P<key>(?:authorization|proxy-authorization|x-api-key|api-key)\s*[:=]\s*)"
    r"(?P<scheme>(?:bearer|basic|token)\s+)?(?P<value>[^\s,;\"']+)",
    re.IGNORECASE,
)
_BEARER = re.compile(
    r"\b(?P<scheme>bearer|basic)\s+(?P<value>[A-Za-z0-9\-._~+/=]{8,})",
    re.IGNORECASE,
)
_QUERY_PARAM = re.compile(
    r"(?P<key>\b(?:token|access_token|api_key|apikey|auth|password|passwd|pwd|secret|sig|signature)"
    r"\s*=\s*)(?P<value>[^\s&#\"']+)",
    re.IGNORECASE,
)
_HF_TOKEN = re.compile(r"\bhf_(?:oauth_)?[A-Za-z0-9]{16,}\b")
_SK_KEY = re.compile(r"\bsk-(?:[A-Za-z0-9_\-]+-)?[A-Za-z0-9_\-]{8,}")
_GH_TOKEN = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}")

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (_URL_USERINFO, rf"\g<scheme>{REDACTED}@"),
    (_AUTH_HEADER, rf"\g<key>\g<scheme>{REDACTED}"),
    (_BEARER, rf"\g<scheme> {REDACTED}"),
    (_QUERY_PARAM, rf"\g<key>{REDACTED}"),
    (_HF_TOKEN, REDACTED),
    (_SK_KEY, REDACTED),
    (_GH_TOKEN, REDACTED),
)


def is_secret_key(key: str) -> bool:
    """Return ``True`` when a mapping/environment key name implies a secret value."""
    try:
        upper = key.upper()
        lower = key.lower()
    except Exception:  # pragma: no cover - only for exotic str subclasses
        return True
    if upper in SECRET_ENV_KEYS:
        return True
    if lower in SAFE_KEY_NAMES:
        return False
    if SECRET_KEY_SEGMENTS.intersection(_KEY_SPLIT.split(lower)):
        return True
    return any(phrase in lower for phrase in SECRET_KEY_PHRASES)


def environment_secrets(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Return the literal secret *values* present in ``env``, for ``extra_secrets``.

    Pattern-based redaction only recognises secrets that *look* like secrets, so a
    token whose value is an opaque string — a self-hosted Hub token, a password —
    survives :func:`redact_text` untouched if a child process echoes it. Collecting
    the values held under the known secret-bearing names closes that gap: whatever
    a subprocess prints, the literal is masked.

    This list is deliberately narrower than :func:`is_secret_key`, because the
    consequence of a mistake runs the other way. In :func:`redact_env` a false
    positive costs one masked value; here it would replace *every* occurrence of
    that literal in every log line and recorded argv. A value that is merely an
    existing filesystem path is therefore skipped — ``PWD`` matches the ``pwd``
    key segment, and masking the working directory would corrupt the recorded
    conversion command — as are values below :data:`MIN_SECRET_LENGTH` and the
    names in :data:`NON_SECRET_ENV_KEYS`. Never raises.
    """
    try:
        source = os.environ if env is None else env
        items = list(source.items())
    except Exception:  # pragma: no cover - hostile mapping
        return ()
    found: dict[str, None] = {}
    for key, value in items:
        name = key if isinstance(key, str) else str(key)
        if name.upper() in NON_SECRET_ENV_KEYS or not is_secret_key(name):
            continue
        text = (value if isinstance(value, str) else str(value)).strip()
        if len(text) < MIN_SECRET_LENGTH:
            continue
        if _is_existing_path(text):
            continue
        found.setdefault(text, None)
    return tuple(found)


def _is_existing_path(value: str) -> bool:
    """Whether ``value`` names something that exists on this filesystem."""
    try:
        return os.sep in value and Path(value).exists()
    except (OSError, ValueError):  # pragma: no cover - defensive
        return False


def _apply_extra_secrets(text: str, extra_secrets: Iterable[str]) -> str:
    """Mask verbatim occurrences of caller-supplied secret literals."""
    for secret in extra_secrets:
        # ``str()`` keeps the helper safe when an untyped caller passes non-strings.
        candidate = str(secret).strip()
        if len(candidate) < MIN_SECRET_LENGTH:
            continue
        text = text.replace(candidate, REDACTED)
    return text


def redact_text(text: str, extra_secrets: Iterable[str] = ()) -> str:
    """Mask every recognised secret shape in ``text``.

    Handles Hugging Face ``hf_*`` tokens, ``Authorization``/``Bearer`` headers,
    ``token=``-style query parameters, URL userinfo (``https://user:pass@host``),
    ``sk-`` API keys, plus any literal in ``extra_secrets``.
    """
    try:
        value = text if isinstance(text, str) else str(text)
        value = _apply_extra_secrets(value, extra_secrets)
        for pattern, replacement in _PATTERNS:
            value = pattern.sub(replacement, value)
    except Exception:  # pragma: no cover - fail closed rather than leak
        return REDACTED
    return value


def redact_env(env: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of ``env`` with secret variables masked and values scrubbed."""
    result: dict[str, str] = {}
    try:
        items = list(env.items())
    except Exception:  # pragma: no cover - hostile mapping
        return {}
    for key, value in items:
        name = key if isinstance(key, str) else str(key)
        if is_secret_key(name):
            result[name] = REDACTED
        else:
            result[name] = redact_text(value if isinstance(value, str) else str(value))
    return result


def _option_name(argument: str) -> str:
    """Return the option part of ``--opt=value`` or the argument itself."""
    return argument.split("=", 1)[0]


def redact_argv(argv: Sequence[str]) -> list[str]:
    """Mask secret-bearing command line options, both split and inline forms."""
    result: list[str] = []
    mask_next = False
    try:
        items = [item if isinstance(item, str) else str(item) for item in argv]
    except Exception:  # pragma: no cover - hostile sequence
        return [REDACTED]
    for argument in items:
        if mask_next:
            result.append(REDACTED)
            mask_next = False
            continue
        name = _option_name(argument)
        if name in SECRET_ARG_NAMES:
            if "=" in argument:
                result.append(f"{name}={REDACTED}")
            else:
                result.append(argument)
                mask_next = True
            continue
        result.append(redact_text(argument))
    return result


def redact_mapping(data: Mapping[str, object]) -> dict[str, object]:
    """Recursively redact a mapping by key name and by value content."""
    result: dict[str, object] = {}
    try:
        items = list(data.items())
    except Exception:  # pragma: no cover - hostile mapping
        return {}
    for key, value in items:
        name = key if isinstance(key, str) else str(key)
        result[name] = REDACTED if is_secret_key(name) else redact_value(value)
    return result


def redact_value(value: object) -> object:
    """Redact one arbitrary value, recursing into containers."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return redact_mapping(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_value(item) for item in value]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return redact_text(str(value))
