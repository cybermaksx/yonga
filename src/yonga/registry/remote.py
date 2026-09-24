"""Remote registry bundles: signed, verified, and unable to carry code.

``REGISTRY_SPEC.md`` section 12: a remote registry must be authenticated before
automatic use, and must never deliver executable repair logic. Compatibility
knowledge changes faster than releases -- a new NPU compiler signature, a new
architecture's exporter pin -- so this is how it reaches a host between them.

A bundle is **one JSON document** holding registry YAML files by relative
path, plus a detached Ed25519 signature over its exact bytes. One document
rather than an archive, so there is nothing to extract and no path to escape
through. Verification, all of which must pass before a byte is trusted:

1. the signature is valid for a key the user trusts. Trusted keys are files in
   ``Paths.registry_keys_dir``; yonga ships none, and nothing downloaded can
   add one;
2. the bundle has not expired, and is not older than the one installed
   (rollback) -- a signed but stale bundle can withhold a fix as surely as a
   forged one can inject a fault;
3. the content is safe regardless of who signed it: relative ``.yaml`` paths
   only, bounded size, every document parses with ``yaml.safe_load`` into the
   strict schema, and every repair rule names an implementation that is
   already compiled into this yonga. The YAML can choose which built-in repair
   applies where; it can never supply one.

The installed bundle is verified again every time the registry is loaded, so
editing the cached file on disk disables it rather than trusting it.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from yonga import __version__
from yonga.config.paths import Paths
from yonga.domain.diagnostics import FailureCategory, RegistryError, diagnostic
from yonga.registry.schema import RegistryDocument
from yonga.util.versions import parse_version

__all__ = [
    "BUNDLE_FORMAT",
    "MAX_BUNDLE_BYTES",
    "RegistryBundle",
    "RemoteRegistryStore",
    "TrustedKey",
    "VerifiedBundle",
    "fetch",
    "key_id",
    "sign_bundle",
    "trusted_keys",
    "verify_bundle",
]

BUNDLE_FORMAT: int = 1
#: A registry is a few dozen kilobytes of YAML; anything near this is not one.
MAX_BUNDLE_BYTES: int = 4 * 2**20
MAX_FILES: int = 256
_SIGNATURE_BYTES = 64 * 1024
_KEY_SUFFIX = ".pub"


def _error(code: str, title: str, message: str, *remediation: str) -> RegistryError:
    return RegistryError(
        diagnostic(
            code,
            FailureCategory.REGISTRY_ERROR,
            title,
            message,
            remediation=list(remediation),
        )
    )


# --- models ---------------------------------------------------------------------------


class RegistryBundle(BaseModel):
    """The signed document: metadata plus registry YAML by relative path."""

    model_config = ConfigDict(extra="forbid")

    format: int
    version: int = Field(ge=1)
    """Monotonic; a host refuses a bundle older than the one it has."""
    created_at: datetime
    expires_at: datetime
    min_tool_version: str | None = None
    description: str = ""
    files: dict[str, str]


class TrustedKey(BaseModel):
    """One Ed25519 public key the user placed in the keys directory."""

    model_config = ConfigDict(frozen=True)

    key_id: str
    public_key: bytes
    source: Path


class VerifiedBundle(BaseModel):
    """A bundle that passed every check, with its parsed documents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bundle: RegistryBundle
    key_id: str
    sha256: str
    documents: list[RegistryDocument]


def key_id(public_key: bytes) -> str:
    """A short, stable name for a public key: the start of its SHA-256."""
    return hashlib.sha256(public_key).hexdigest()[:16]


# --- keys -----------------------------------------------------------------------------


def trusted_keys(paths: Paths) -> list[TrustedKey]:
    """Every ``*.pub`` in the keys directory: base64 of a raw 32-byte Ed25519 key."""
    directory = paths.registry_keys_dir
    if not directory.is_dir():
        return []
    keys: list[TrustedKey] = []
    for path in sorted(directory.glob(f"*{_KEY_SUFFIX}")):
        try:
            raw = base64.b64decode(path.read_text(encoding="utf-8").split()[0], validate=True)
        except (OSError, IndexError, ValueError, binascii.Error):
            continue
        if len(raw) == 32:
            keys.append(TrustedKey(key_id=key_id(raw), public_key=raw, source=path))
    return keys


# --- signing (maintainers) ------------------------------------------------------------


def sign_bundle(bundle_bytes: bytes, private_key: Ed25519PrivateKey) -> bytes:
    """The detached signature document for ``bundle_bytes``."""
    public = private_key.public_key().public_bytes_raw()
    signature = private_key.sign(bundle_bytes)
    payload = {
        "format": BUNDLE_FORMAT,
        "key_id": key_id(public),
        "signature": base64.b64encode(signature).decode(),
    }
    return json.dumps(payload, indent=2).encode()


# --- verification ---------------------------------------------------------------------


def verify_bundle(
    bundle_bytes: bytes,
    signature_bytes: bytes,
    keys: list[TrustedKey],
    *,
    now: datetime | None = None,
    installed_version: int | None = None,
    tool_version: str = __version__,
) -> VerifiedBundle:
    """Check a bundle completely, or raise a :class:`RegistryError` saying why."""
    if len(bundle_bytes) > MAX_BUNDLE_BYTES:
        raise _error(
            "registry.remote.too_large",
            "The registry bundle is too large",
            f"{len(bundle_bytes)} bytes, against a limit of {MAX_BUNDLE_BYTES}.",
        )
    signer = _check_signature(bundle_bytes, signature_bytes, keys)
    try:
        bundle = RegistryBundle.model_validate_json(bundle_bytes)
    except ValidationError as error:
        raise _error(
            "registry.remote.invalid",
            "The registry bundle is malformed",
            f"The signed document is not a registry bundle: {error.error_count()} problem(s).",
        ) from error
    if bundle.format != BUNDLE_FORMAT:
        raise _error(
            "registry.remote.unsupported_format",
            "This yonga cannot read that bundle format",
            f"Bundle format {bundle.format}; this yonga reads format {BUNDLE_FORMAT}.",
            "Upgrade yonga.",
        )
    moment = now if now is not None else datetime.now(UTC)
    if bundle.expires_at <= moment:
        raise _error(
            "registry.remote.expired",
            "The registry bundle has expired",
            f"It expired at {bundle.expires_at.isoformat()}. An expired bundle may be "
            "withholding newer knowledge, so it is not used.",
            "Fetch a current one: yonga registry update",
        )
    if installed_version is not None and bundle.version < installed_version:
        raise _error(
            "registry.remote.rollback",
            "The registry bundle is older than the installed one",
            f"Version {bundle.version} would replace version {installed_version}.",
            "A downgrade is never applied automatically; 'yonga registry reset' then "
            "update, if you really want the older data.",
        )
    required = parse_version(bundle.min_tool_version)
    current = parse_version(tool_version)
    if required is not None and current is not None and current < required:
        raise _error(
            "registry.remote.tool_too_old",
            "The registry bundle needs a newer yonga",
            f"It requires yonga {bundle.min_tool_version}; this is {tool_version}.",
            "Upgrade yonga.",
        )
    documents = _documents(bundle)
    return VerifiedBundle(
        bundle=bundle,
        key_id=signer,
        sha256=hashlib.sha256(bundle_bytes).hexdigest(),
        documents=documents,
    )


def _check_signature(bundle_bytes: bytes, signature_bytes: bytes, keys: list[TrustedKey]) -> str:
    if not keys:
        raise _error(
            "registry.remote.no_trusted_keys",
            "No registry signing key is trusted",
            "A remote registry is only used when it is signed by a key you trust, and no "
            "key is installed.",
            "Put the publisher's public key (base64, one line) in a *.pub file in the "
            "registry-keys directory; 'yonga registry paths' shows where.",
        )
    try:
        if len(signature_bytes) > _SIGNATURE_BYTES:
            raise ValueError("signature document too large")
        document = json.loads(signature_bytes)
        claimed = str(document["key_id"])
        signature = base64.b64decode(str(document["signature"]), validate=True)
    except (ValueError, KeyError, TypeError, binascii.Error) as error:
        raise _error(
            "registry.remote.signature_malformed",
            "The bundle signature is unreadable",
            "The signature file is not a yonga registry signature.",
        ) from error
    key = next((candidate for candidate in keys if candidate.key_id == claimed), None)
    if key is None:
        raise _error(
            "registry.remote.untrusted_key",
            "The bundle is signed by a key you do not trust",
            f"It names key {claimed}, which is not in the registry-keys directory.",
            "Only add a key after checking its fingerprint with its publisher.",
        )
    try:
        Ed25519PublicKey.from_public_bytes(key.public_key).verify(signature, bundle_bytes)
    except InvalidSignature as error:
        raise _error(
            "registry.remote.bad_signature",
            "The bundle signature does not match",
            f"The bundle was not signed by key {claimed}, or was changed after signing.",
        ) from error
    return key.key_id


def _documents(bundle: RegistryBundle) -> list[RegistryDocument]:
    from yonga.registry.loader import parse_document
    from yonga.repairs import known_implementations

    if len(bundle.files) > MAX_FILES:
        raise _error(
            "registry.remote.too_many_files",
            "The registry bundle has too many files",
            f"{len(bundle.files)} files, against a limit of {MAX_FILES}.",
        )
    built_in = known_implementations()
    documents: list[RegistryDocument] = []
    for name in sorted(bundle.files):
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.suffix not in (".yaml", ".yml")
        ):
            raise _error(
                "registry.remote.unsafe_path",
                "The registry bundle names an unsafe file",
                f"{name!r} is not a relative .yaml path.",
            )
        document = parse_document(bundle.files[name], origin=f"remote:{name}")
        for repair in document.repairs:
            if repair.implementation not in built_in:
                raise _error(
                    "registry.remote.unknown_repair_implementation",
                    "The registry bundle names a repair this yonga does not have",
                    f"Repair rule {repair.id!r} names {repair.implementation!r}. Registry data "
                    "can only select a repair compiled into yonga, never supply one.",
                    "Upgrade yonga if a newer release has that repair.",
                )
        documents.append(document)
    return documents


# --- fetching -------------------------------------------------------------------------


def fetch(url: str, *, timeout: float = 30.0, limit: int = MAX_BUNDLE_BYTES) -> bytes:
    """Download ``url`` over HTTPS, refusing anything larger than ``limit``."""
    if not url.lower().startswith("https://"):
        raise _error(
            "registry.remote.insecure_url",
            "Registry bundles are only fetched over HTTPS",
            f"{url!r} is not an https:// URL.",
            "Use --file for a bundle you already have locally.",
        )
    request = urllib.request.Request(url, headers={"User-Agent": f"yonga/{__version__}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload: bytes = response.read(limit + 1)
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise _error(
            "registry.remote.fetch_failed",
            "The registry bundle could not be downloaded",
            f"Fetching {url} failed: {error}.",
            "Check the URL and the network, then retry.",
        ) from error
    if len(payload) > limit:
        raise _error(
            "registry.remote.too_large",
            "The registry bundle is too large",
            f"{url} returned more than {limit} bytes.",
        )
    return payload


# --- the installed layer --------------------------------------------------------------


class RemoteRegistryState(BaseModel):
    """What is installed, for ``registry status``."""

    model_config = ConfigDict(extra="ignore")

    version: int
    key_id: str
    sha256: str
    source: str
    installed_at: datetime
    expires_at: datetime
    description: str = ""


class RemoteRegistryStore:
    """The verified bundle on disk, and the only code that reads or writes it."""

    BUNDLE = "bundle.json"
    SIGNATURE = "bundle.json.sig"
    STATE = "state.json"

    def __init__(self, paths: Paths) -> None:
        """Bind the store to ``paths.remote_registry_dir``."""
        self._paths = paths
        self._dir = paths.remote_registry_dir

    @property
    def directory(self) -> Path:
        """Where the bundle lives."""
        return self._dir

    def state(self) -> RemoteRegistryState | None:
        """What was installed, or ``None``. Says nothing about whether it still verifies."""
        try:
            return RemoteRegistryState.model_validate_json(
                (self._dir / self.STATE).read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError):
            return None

    def verify(
        self, bundle_bytes: bytes, signature_bytes: bytes, *, now: datetime | None = None
    ) -> VerifiedBundle:
        """Verify a candidate against the trusted keys and the installed version."""
        state = self.state()
        return verify_bundle(
            bundle_bytes,
            signature_bytes,
            trusted_keys(self._paths),
            now=now,
            installed_version=state.version if state is not None else None,
        )

    def install(
        self, verified: VerifiedBundle, bundle_bytes: bytes, signature_bytes: bytes, *, source: str
    ) -> RemoteRegistryState:
        """Write a verified bundle, the signature last-but-one and the state last."""
        if hashlib.sha256(bundle_bytes).hexdigest() != verified.sha256:
            raise _error(
                "registry.remote.mismatch",
                "The bundle changed after it was verified",
                "The bytes to install are not the bytes that were verified.",
            )
        self._dir.mkdir(parents=True, exist_ok=True)
        state = RemoteRegistryState(
            version=verified.bundle.version,
            key_id=verified.key_id,
            sha256=verified.sha256,
            source=source,
            installed_at=datetime.now(UTC),
            expires_at=verified.bundle.expires_at,
            description=verified.bundle.description,
        )
        for name, payload in (
            (self.BUNDLE, bundle_bytes),
            (self.SIGNATURE, signature_bytes),
            (self.STATE, state.model_dump_json(indent=2).encode()),
        ):
            partial = self._dir / f".{name}.partial"
            partial.write_bytes(payload)
            partial.replace(self._dir / name)
        return state

    def load(self, *, now: datetime | None = None) -> VerifiedBundle | None:
        """The installed bundle, verified again from scratch; ``None`` when none is installed.

        Raises when one is installed but no longer verifies: expired, edited on
        disk, or signed by a key that has since been removed.
        """
        bundle_path = self._dir / self.BUNDLE
        if not bundle_path.is_file():
            return None
        try:
            bundle_bytes = bundle_path.read_bytes()
            signature_bytes = (self._dir / self.SIGNATURE).read_bytes()
        except OSError as error:
            raise _error(
                "registry.remote.unreadable",
                "The installed registry bundle cannot be read",
                str(error),
                "yonga registry reset",
            ) from error
        return verify_bundle(bundle_bytes, signature_bytes, trusted_keys(self._paths), now=now)

    def reset(self) -> bool:
        """Remove the installed bundle. Returns whether there was one."""
        removed = False
        for name in (self.STATE, self.SIGNATURE, self.BUNDLE):
            path = self._dir / name
            if path.exists():
                path.unlink()
                removed = True
        return removed
