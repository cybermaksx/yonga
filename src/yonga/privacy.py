"""Keep third-party telemetry switched off in every Python process yonga runs.

yonga sends no telemetry, and the libraries it runs may not send any on its
behalf. OpenVINO's model conversion API, ``openvino.tools.ovc``, is imported by
``import openvino`` itself and again by Optimum Intel and NNCF, and it reports
each import through the ``openvino_telemetry`` package: without a consent file
under ``~/intel`` it sends an event to Google Analytics from a
``multiprocessing`` child and counts the import in ``~/intel/stats``. NNCF
reports through the same package.

Every one of those callers imports ``openvino_telemetry`` inside ``try`` /
``except ImportError`` and falls back to a stub that does nothing. Making that
one import fail is therefore a complete switch, and it needs neither the opt-out
file in the user's home directory (which yonga must not write) nor ``CI=true`` in
the environment (which changes how other tools behave as well).

* :func:`block_vendor_telemetry` flips the switch in this process. ``yonga``'s
  ``__init__`` calls it, so it covers the CLI, every ``python -m yonga...``
  child (the smoke test, the GGUF import) and every ``multiprocessing`` child,
  which re-imports yonga's main module before it runs anything. A program that
  imports yonga as a library gets it too, unless it imported the telemetry
  package first, which is then left alone.
* A child interpreter that never imports yonga -- an import check, or
  ``optimum-cli`` in a managed export environment where yonga is not even
  installed -- is started through :func:`python_code_argv` or
  :func:`python_module_argv`, which put the same switch in front of its code.

Only the standard library may be imported here: this module runs before
anything else in the package.
"""

from __future__ import annotations

import sys
from os import PathLike

__all__ = [
    "BLOCKED_TELEMETRY_MODULES",
    "TELEMETRY_BOOTSTRAP",
    "block_vendor_telemetry",
    "python_code_argv",
    "python_module_argv",
]

#: Packages whose only job is reporting usage to a vendor. Importing one fails.
BLOCKED_TELEMETRY_MODULES: tuple[str, ...] = ("openvino_telemetry",)

#: Python source that does what :func:`block_vendor_telemetry` does, for a child.
TELEMETRY_BOOTSTRAP: str = (
    "import sys as _yonga_sys\n"
    f"for _yonga_name in {BLOCKED_TELEMETRY_MODULES!r}:\n"
    "    _yonga_sys.modules.setdefault(_yonga_name, None)\n"
    "del _yonga_sys, _yonga_name\n"
)


def block_vendor_telemetry() -> None:
    """Make ``import openvino_telemetry`` raise ``ImportError`` in this process.

    A ``None`` entry in :data:`sys.modules` is Python's own way of saying "this
    module cannot be imported". A package that is already imported is kept:
    replacing it under a program that holds it would break that program, not
    stop a report it has already sent.
    """
    for name in BLOCKED_TELEMETRY_MODULES:
        if name not in sys.modules:
            sys.modules[name] = None  # type: ignore[assignment]


def python_code_argv(python: str | PathLike[str], code: str, *args: str) -> list[str]:
    """``python -P -c CODE ARGS...`` with vendor telemetry blocked before ``CODE`` runs.

    ``-P`` keeps the working directory off ``sys.path``, as everywhere else
    yonga starts an interpreter; ``sys.argv[1:]`` is ``args`` exactly as before.
    """
    return [str(python), "-P", "-c", TELEMETRY_BOOTSTRAP + code, *args]


def python_module_argv(python: str | PathLike[str], module: str, *args: str) -> list[str]:
    """``python -P -m MODULE ARGS...``, with vendor telemetry blocked before ``MODULE`` loads.

    :func:`runpy.run_module` with ``alter_sys=True`` is what ``-m`` does: the
    module runs as ``__main__`` and ``sys.argv[0]`` becomes its file, so the
    module sees the same command line either way.
    """
    code = f"import runpy\nrunpy.run_module({module!r}, run_name='__main__', alter_sys=True)\n"
    return python_code_argv(python, code, *args)
