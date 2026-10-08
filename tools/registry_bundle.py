#!/usr/bin/env python3
"""Create a signing key, or build and sign a remote registry bundle. For maintainers.

    python tools/registry_bundle.py keygen --output yonga-registry
        writes yonga-registry.key (private, keep it offline) and yonga-registry.pub

    python tools/registry_bundle.py build DIR --key yonga-registry.key \\
        --version 7 --expires-days 90 --output bundle.json
        writes bundle.json and bundle.json.sig from every *.yaml under DIR

A bundle carrying registry schema 2 or later gets ``min_tool_version`` filled in
from ``SCHEMA_MIN_TOOL_VERSION``, so an older yonga refuses it at
``registry update`` ("needs a newer yonga") and keeps the bundle it has,
instead of installing data it cannot parse. Keep remote bundles at schema 1
while older releases still need remote issue and repair updates.

The bundle is verified with the same code ``yonga registry update`` uses
before it is written, so a bundle this tool produces is one yonga accepts.
Users trust the key by copying the ``.pub`` file into their registry-keys
directory (``yonga registry paths`` shows it).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from yonga.registry.loader import parse_document
from yonga.registry.remote import (
    BUNDLE_FORMAT,
    TrustedKey,
    key_id,
    sign_bundle,
    verify_bundle,
)
from yonga.registry.schema import SCHEMA_MIN_TOOL_VERSION
from yonga.util.versions import parse_version


def schema_versions(files: dict[str, str]) -> set[int]:
    """Every schema version the files use, as the loader reads them.

    Parsed with the loader itself rather than by looking for the key, because
    the loader decides what an omitted ``schema_version`` means: a file that
    declares none is read at the build's own version, and a rule inherits its
    document's. Counting only the keys written down would call such a file
    schema 1 and ship a bundle older releases cannot parse.
    """
    found: set[int] = set()
    for name, text in files.items():
        document = parse_document(text, origin=name)
        found.add(document.schema_version)
        found.update(rule.schema_version for rule in document.rules())
    return found or {1}


def required_tool_version(files: dict[str, str], requested: str | None) -> str | None:
    """The min_tool_version the bundle must carry: the schema's floor, or higher."""
    floor = SCHEMA_MIN_TOOL_VERSION.get(max(schema_versions(files)))
    if requested is None:
        return floor
    if floor is not None:
        wanted, needed = parse_version(requested), parse_version(floor)
        if wanted is None or needed is None or wanted < needed:
            sys.exit(f"--min-tool-version {requested} is below {floor}, which this schema needs")
    return requested


def keygen(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    private = Ed25519PrivateKey.generate()
    key_path = output.with_suffix(".key")
    pub_path = output.with_suffix(".pub")
    key_path.write_bytes(
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    raw = private.public_key().public_bytes_raw()
    pub_path.write_text(base64.b64encode(raw).decode() + "\n", encoding="utf-8")
    print(f"private key {key_path} (keep it offline)")
    print(f"public key  {pub_path}  id {key_id(raw)}")


def build(
    directory: Path,
    key: Path,
    version: int,
    expires_days: int,
    output: Path,
    note: str,
    min_tool_version: str | None = None,
) -> None:
    private = serialization.load_pem_private_key(key.read_bytes(), password=None)
    if not isinstance(private, Ed25519PrivateKey):
        sys.exit("the key is not an Ed25519 private key")
    files = {
        path.relative_to(directory).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(directory.rglob("*"))
        if path.suffix in (".yaml", ".yml") and path.is_file()
    }
    now = datetime.now(UTC).replace(microsecond=0)
    bundle: dict[str, object] = {
        "format": BUNDLE_FORMAT,
        "version": version,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(days=expires_days)).isoformat(),
        "description": note,
        "files": files,
    }
    needed = required_tool_version(files, min_tool_version)
    if needed is not None:
        bundle["min_tool_version"] = needed
    payload = json.dumps(bundle, indent=2, sort_keys=True).encode()
    signature = sign_bundle(payload, private)
    raw = private.public_key().public_bytes_raw()
    verified = verify_bundle(
        payload,
        signature,
        [TrustedKey(key_id=key_id(raw), public_key=raw, source=key)],
        # Check the bundle as the oldest yonga it admits will see it.
        **({"tool_version": needed} if needed is not None else {}),
    )
    output.write_bytes(payload)
    output.with_name(output.name + ".sig").write_bytes(signature)
    rules = sum(len(document.rules()) for document in verified.documents)
    floor = f", needs yonga {needed}+" if needed is not None else ""
    print(
        f"{output}: version {version}, {len(files)} file(s), {rules} rule(s), "
        f"key {key_id(raw)}{floor}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("keygen")
    generate.add_argument("--output", type=Path, required=True)
    make = commands.add_parser("build")
    make.add_argument("directory", type=Path)
    make.add_argument("--key", type=Path, required=True)
    make.add_argument("--version", type=int, required=True)
    make.add_argument("--expires-days", type=int, default=90)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--description", default="")
    make.add_argument(
        "--min-tool-version",
        default=None,
        help="oldest yonga that may install the bundle (default: what its schema needs)",
    )
    arguments = parser.parse_args()
    if arguments.command == "keygen":
        keygen(arguments.output)
    else:
        build(
            arguments.directory,
            arguments.key,
            arguments.version,
            arguments.expires_days,
            arguments.output,
            arguments.description,
            arguments.min_tool_version,
        )


if __name__ == "__main__":
    main()
