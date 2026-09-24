"""Remote registry bundles: nothing is trusted until every check passes, every time.

REGISTRY_SPEC section 12 in tests. Each attack a remote channel invites has
its own test: forgery, tampering, an untrusted signer, replaying an older or
expired bundle, a path that escapes, and -- the load-bearing invariant --
registry data trying to name repair code yonga does not already contain.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from yonga.application.wiring import Wiring
from yonga.config.paths import Paths
from yonga.config.settings import Settings
from yonga.domain.diagnostics import RegistryError
from yonga.registry.remote import (
    RemoteRegistryStore,
    fetch,
    key_id,
    sign_bundle,
    trusted_keys,
    verify_bundle,
)

#: Relative to the real clock: the Wiring tests verify with the current time.
NOW = datetime.now(UTC).replace(microsecond=0)

ISSUE = """\
schema_version: 1
issues:
  - id: remote-test-issue
    schema_version: 1
    status: active
    priority: 90
    summary: A signature delivered by a remote bundle
    match:
      device: gpu
      exception_contains: ["REMOTE_TEST_SIGNATURE"]
    classification:
      code: gpu.remote_test
      category: unknown_runtime_error
    recommendation:
      title: Remote test
      message: A test rule.
"""

FOREIGN_REPAIR = """\
schema_version: 1
repairs:
  - id: remote-repair
    schema_version: 1
    status: active
    priority: 100
    summary: Tries to bring its own code
    match:
      files: [openvino_tokenizer.xml]
      contains: "x"
    repair:
      implementation: download_and_run_this
      safety: automatic
"""


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        config_dir=tmp_path / "config", cache_dir=tmp_path / "cache", data_dir=tmp_path / "data"
    )


@pytest.fixture
def paths(settings: Settings) -> Paths:
    return settings.paths


@pytest.fixture
def key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def trust(paths: Paths, key: Ed25519PrivateKey, name: str = "publisher") -> str:
    raw = key.public_key().public_bytes_raw()
    paths.registry_keys_dir.mkdir(parents=True, exist_ok=True)
    (paths.registry_keys_dir / f"{name}.pub").write_text(
        base64.b64encode(raw).decode() + "\n", encoding="utf-8"
    )
    return key_id(raw)


def bundle(
    *,
    version: int = 1,
    files: dict[str, str] | None = None,
    expires: datetime | None = None,
    min_tool_version: str | None = None,
) -> bytes:
    document = {
        "format": 1,
        "version": version,
        "created_at": (NOW - timedelta(days=1)).isoformat(),
        "expires_at": (expires or NOW + timedelta(days=30)).isoformat(),
        "min_tool_version": min_tool_version,
        "description": "test bundle",
        "files": files if files is not None else {"issues/remote.yaml": ISSUE},
    }
    return json.dumps(document).encode()


def signed(key: Ed25519PrivateKey, **kwargs: object) -> tuple[bytes, bytes]:
    payload = bundle(**kwargs)  # type: ignore[arg-type]
    return payload, sign_bundle(payload, key)


def code_of(error: pytest.ExceptionInfo[RegistryError]) -> str:
    return error.value.diagnostic.code


# --- verification ---------------------------------------------------------------------


def test_a_bundle_signed_by_a_trusted_key_verifies(paths: Paths, key: Ed25519PrivateKey) -> None:
    signer = trust(paths, key)
    payload, signature = signed(key)

    verified = verify_bundle(payload, signature, trusted_keys(paths), now=NOW)

    assert verified.key_id == signer
    assert [rule.id for d in verified.documents for rule in d.rules()] == ["remote-test-issue"]


def test_nothing_verifies_without_a_trusted_key(paths: Paths, key: Ed25519PrivateKey) -> None:
    payload, signature = signed(key)

    with pytest.raises(RegistryError) as caught:
        verify_bundle(payload, signature, trusted_keys(paths), now=NOW)
    assert code_of(caught) == "registry.remote.no_trusted_keys"


def test_a_bundle_signed_by_another_key_is_refused(paths: Paths, key: Ed25519PrivateKey) -> None:
    trust(paths, key)
    payload, signature = signed(Ed25519PrivateKey.generate())

    with pytest.raises(RegistryError) as caught:
        verify_bundle(payload, signature, trusted_keys(paths), now=NOW)
    assert code_of(caught) == "registry.remote.untrusted_key"


def test_a_forged_signature_claiming_a_trusted_key_is_refused(
    paths: Paths, key: Ed25519PrivateKey
) -> None:
    trusted = trust(paths, key)
    payload, _ = signed(key)
    forged = sign_bundle(payload, Ed25519PrivateKey.generate())
    document = json.loads(forged)
    document["key_id"] = trusted
    with pytest.raises(RegistryError) as caught:
        verify_bundle(payload, json.dumps(document).encode(), trusted_keys(paths), now=NOW)
    assert code_of(caught) == "registry.remote.bad_signature"


def test_one_changed_byte_breaks_the_signature(paths: Paths, key: Ed25519PrivateKey) -> None:
    trust(paths, key)
    payload, signature = signed(key)
    tampered = payload.replace(b"REMOTE_TEST_SIGNATURE", b"REMOTE_TEST_SIGNATURF")

    with pytest.raises(RegistryError) as caught:
        verify_bundle(tampered, signature, trusted_keys(paths), now=NOW)
    assert code_of(caught) == "registry.remote.bad_signature"


def test_a_malformed_signature_is_refused(paths: Paths, key: Ed25519PrivateKey) -> None:
    trust(paths, key)
    payload, _ = signed(key)

    with pytest.raises(RegistryError) as caught:
        verify_bundle(payload, b"not json", trusted_keys(paths), now=NOW)
    assert code_of(caught) == "registry.remote.signature_malformed"


def test_an_expired_bundle_is_refused(paths: Paths, key: Ed25519PrivateKey) -> None:
    """A signed but stale bundle can withhold a fix as surely as a forged one injects a fault."""
    trust(paths, key)
    payload, signature = signed(key, expires=NOW - timedelta(seconds=1))

    with pytest.raises(RegistryError) as caught:
        verify_bundle(payload, signature, trusted_keys(paths), now=NOW)
    assert code_of(caught) == "registry.remote.expired"


def test_an_older_bundle_never_replaces_a_newer_one(paths: Paths, key: Ed25519PrivateKey) -> None:
    trust(paths, key)
    payload, signature = signed(key, version=3)

    with pytest.raises(RegistryError) as caught:
        verify_bundle(payload, signature, trusted_keys(paths), now=NOW, installed_version=4)
    assert code_of(caught) == "registry.remote.rollback"


def test_a_bundle_for_a_newer_yonga_is_refused(paths: Paths, key: Ed25519PrivateKey) -> None:
    trust(paths, key)
    payload, signature = signed(key, min_tool_version="99.0")

    with pytest.raises(RegistryError) as caught:
        verify_bundle(payload, signature, trusted_keys(paths), now=NOW)
    assert code_of(caught) == "registry.remote.tool_too_old"


@pytest.mark.parametrize("name", ["../escape.yaml", "/etc/passwd.yaml", "repairs/code.py"])
def test_a_signed_bundle_still_cannot_name_an_unsafe_path(
    paths: Paths, key: Ed25519PrivateKey, name: str
) -> None:
    trust(paths, key)
    payload, signature = signed(key, files={name: ISSUE})

    with pytest.raises(RegistryError) as caught:
        verify_bundle(payload, signature, trusted_keys(paths), now=NOW)
    assert code_of(caught) == "registry.remote.unsafe_path"


def test_a_signed_bundle_can_never_supply_repair_code(paths: Paths, key: Ed25519PrivateKey) -> None:
    """The YAML may choose a built-in repair; it can never bring one of its own."""
    trust(paths, key)
    payload, signature = signed(key, files={"repairs.yaml": FOREIGN_REPAIR})

    with pytest.raises(RegistryError) as caught:
        verify_bundle(payload, signature, trusted_keys(paths), now=NOW)
    assert code_of(caught) == "registry.remote.unknown_repair_implementation"


def test_yaml_is_parsed_safely_even_when_signed(paths: Paths, key: Ed25519PrivateKey) -> None:
    trust(paths, key)
    hostile = "schema_version: 1\nissues: !!python/object/apply:os.system ['true']\n"
    payload, signature = signed(key, files={"x.yaml": hostile})

    with pytest.raises(RegistryError):
        verify_bundle(payload, signature, trusted_keys(paths), now=NOW)


def test_fetch_refuses_plain_http() -> None:
    with pytest.raises(RegistryError) as caught:
        fetch("http://example.com/bundle.json")
    assert code_of(caught) == "registry.remote.insecure_url"


# --- the installed layer --------------------------------------------------------------


def install(paths: Paths, key: Ed25519PrivateKey, **kwargs: object) -> None:
    payload, signature = signed(key, **kwargs)
    store = RemoteRegistryStore(paths)
    store.install(store.verify(payload, signature, now=NOW), payload, signature, source="test")


def test_an_installed_bundle_becomes_a_registry_layer(
    settings: Settings, key: Ed25519PrivateKey
) -> None:
    trust(settings.paths, key)
    install(settings.paths, key)

    registry = Wiring(settings).registry()

    assert "remote-test-issue" in registry.rule_ids
    assert "remote" in registry.layers


def test_the_policy_never_ignores_an_installed_bundle(
    tmp_path: Path, key: Ed25519PrivateKey
) -> None:
    settings = Settings(
        config_dir=tmp_path / "config",
        cache_dir=tmp_path / "cache",
        data_dir=tmp_path / "data",
        registry_update_policy="never",
    )
    trust(settings.paths, key)
    install(settings.paths, key)

    assert "remote-test-issue" not in Wiring(settings).registry().rule_ids


def test_an_installed_bundle_edited_on_disk_is_dropped_not_trusted(
    settings: Settings, key: Ed25519PrivateKey
) -> None:
    trust(settings.paths, key)
    install(settings.paths, key)
    target = settings.paths.remote_registry_dir / "bundle.json"
    target.write_bytes(target.read_bytes().replace(b"gpu.remote_test", b"gpu.remote_tesT"))

    with pytest.raises(RegistryError):
        RemoteRegistryStore(settings.paths).load()
    registry = Wiring(settings).registry()
    assert "remote-test-issue" not in registry.rule_ids, "falls back to the built-in registry"
    assert registry.rule_ids, "and the built-in registry still loads"


def test_removing_the_key_withdraws_the_bundle(settings: Settings, key: Ed25519PrivateKey) -> None:
    trust(settings.paths, key)
    install(settings.paths, key)
    (settings.paths.registry_keys_dir / "publisher.pub").unlink()

    assert "remote-test-issue" not in Wiring(settings).registry().rule_ids


def test_a_user_override_still_wins_over_the_remote_layer(
    settings: Settings, key: Ed25519PrivateKey
) -> None:
    trust(settings.paths, key)
    install(settings.paths, key)
    overrides = settings.paths.registry_overrides_dir
    overrides.mkdir(parents=True, exist_ok=True)
    (overrides / "mine.yaml").write_text(
        ISSUE.replace("A signature delivered by a remote bundle", "My own version"),
        encoding="utf-8",
    )

    registry = Wiring(settings).registry()
    (rule,) = [r for r in registry.issues if r.id == "remote-test-issue"]
    assert rule.summary == "My own version"


def test_reset_removes_the_bundle(settings: Settings, key: Ed25519PrivateKey) -> None:
    trust(settings.paths, key)
    install(settings.paths, key)
    store = RemoteRegistryStore(settings.paths)

    assert store.reset() is True
    assert store.state() is None
    assert store.reset() is False
