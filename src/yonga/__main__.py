"""Module entry point so that ``python -m yonga`` behaves like the console script.

The ``if __name__ == "__main__"`` guard is required: OpenVINO and the export
toolchain may start worker processes with ``multiprocessing``/``forkserver``,
which re-imports this module in the child.

Help text names the program ``yonga`` either way: the AppImage runs this module,
and "Usage: python -m yonga" would describe a command its users never typed.
"""

from __future__ import annotations

from yonga import APP_NAME
from yonga.cli import app

if __name__ == "__main__":
    app(prog_name=APP_NAME)
