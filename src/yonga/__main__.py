"""Module entry point so that ``python -m yonga`` behaves like the console script.

The ``if __name__ == "__main__"`` guard is required: OpenVINO and the export
toolchain may start worker processes with ``multiprocessing``/``forkserver``,
which re-imports this module in the child.
"""

from __future__ import annotations

from yonga.cli import app

if __name__ == "__main__":
    app()
