#!/usr/bin/env python3
"""Print one version's section of CHANGELOG.md, for the GitHub release notes.

    python tools/release_notes.py 0.2.0

Exits non-zero when the changelog has no section for that version, so a
release cannot be published without its notes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"


def section(text: str, version: str) -> str | None:
    """The body under ``## [version]``, up to the next ``## [`` heading."""
    pattern = re.compile(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)", re.M | re.S)
    found = pattern.search(text)
    return found.group(1).strip() if found else None


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: release_notes.py VERSION", file=sys.stderr)
        return 2
    notes = section(CHANGELOG.read_text(encoding="utf-8"), sys.argv[1])
    if not notes:
        print(f"CHANGELOG.md has no section for {sys.argv[1]}", file=sys.stderr)
        return 1
    print(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
