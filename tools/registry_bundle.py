#!/usr/bin/env python3
"""Create a signing key, or build and sign a remote registry bundle. For maintainers.

    python tools/registry_bundle.py keygen --output yonga-registry
        writes yonga-registry.key (private, keep it offline) and yonga-registry.pub

    python tools/registry_bundle.py build DIR --key yonga-registry.key \\
        --version 7 --expires-days 90 --output bundle.json
        writes bundle.json and bundle.json.sig from every *.yaml under DIR

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

from yonga.registry.remote import (
    BUNDLE_FORMAT,
    TrustedKey,
    key_id,
    sign_bundle,
    verify_bundle,
)


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
    directory: Path, key: Path, version: int, expires_days: int, output: Path, note: str
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
    bundle = {
        "format": BUNDLE_FORMAT,
        "version": version,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(days=expires_days)).isoformat(),
        "description": note,
        "files": files,
    }
    payload = json.dumps(bundle, indent=2, sort_keys=True).encode()
    signature = sign_bundle(payload, private)
    raw = private.public_key().public_bytes_raw()
    verified = verify_bundle(
        payload, signature, [TrustedKey(key_id=key_id(raw), public_key=raw, source=key)]
    )
    output.write_bytes(payload)
    output.with_name(output.name + ".sig").write_bytes(signature)
    rules = sum(len(document.rules()) for document in verified.documents)
    print(f"{output}: version {version}, {len(files)} file(s), {rules} rule(s), key {key_id(raw)}")


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
        )


if __name__ == "__main__":
    main()
