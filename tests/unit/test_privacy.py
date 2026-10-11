"""No vendor telemetry: OpenVINO's usage reporting is switched off in every process.

``openvino_telemetry`` is replaced here by a stand-in that only leaves a marker
file when it is imported, so these tests prove the switch without anything ever
being sent. The real OpenVINO and NNCF fall back to their stubs when the import
fails; that is checked against the installed packages where they exist.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import yonga
from yonga.privacy import (
    BLOCKED_TELEMETRY_MODULES,
    TELEMETRY_BOOTSTRAP,
    block_vendor_telemetry,
    python_code_argv,
    python_module_argv,
)

#: ``src/``, so a child interpreter imports this checkout's yonga.
SOURCE_ROOT = Path(yonga.__file__).resolve().parent.parent


@pytest.fixture
def stand_in(tmp_path: Path) -> tuple[Path, Path]:
    """A fake ``openvino_telemetry`` that writes ``marker`` when imported."""
    site = tmp_path / "site"
    package = site / "openvino_telemetry"
    package.mkdir(parents=True)
    marker = tmp_path / "telemetry-imported"
    (package / "__init__.py").write_text(
        f"open({str(marker)!r}, 'w').close()\nclass Telemetry: pass\n", encoding="utf-8"
    )
    return site, marker


def child_env(*paths: Path) -> dict[str, str]:
    """This environment with ``paths`` first on ``PYTHONPATH``."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in paths)
    return env


def run(argv: list[str], *, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a child interpreter and capture what it prints."""
    return subprocess.run(argv, env=env, cwd=cwd, capture_output=True, text=True, timeout=120)


def test_the_stand_in_is_imported_without_the_switch(stand_in, tmp_path):
    """The control: with nothing blocking it, the telemetry package loads."""
    site, marker = stand_in
    result = run(
        [sys.executable, "-P", "-c", "import openvino_telemetry"],
        env=child_env(site),
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert marker.exists()


def test_importing_yonga_blocks_the_telemetry_package(stand_in, tmp_path):
    """The CLI, ``python -m yonga...`` children and library users all import yonga first."""
    site, marker = stand_in
    code = textwrap.dedent(
        """
        import yonga
        try:
            import openvino_telemetry
        except ImportError:
            print("blocked")
        """
    )
    result = run([sys.executable, "-P", "-c", code], env=child_env(SOURCE_ROOT, site), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "blocked"
    assert not marker.exists()


def test_a_package_already_imported_is_left_alone(monkeypatch):
    sentinel = object()
    monkeypatch.setitem(sys.modules, BLOCKED_TELEMETRY_MODULES[0], sentinel)
    block_vendor_telemetry()
    assert sys.modules[BLOCKED_TELEMETRY_MODULES[0]] is sentinel


def test_this_process_has_the_switch_on():
    """The test run imported yonga, so the telemetry package must be unimportable here."""
    assert BLOCKED_TELEMETRY_MODULES[0] in sys.modules
    assert sys.modules[BLOCKED_TELEMETRY_MODULES[0]] is None


@pytest.mark.parametrize("method", ["spawn", "forkserver"])
def test_multiprocessing_children_inherit_the_switch(stand_in, tmp_path, method):
    """A spawn or forkserver child re-imports the main module, and with it yonga."""
    site, marker = stand_in
    script = tmp_path / "main.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import multiprocessing
            import sys

            import yonga


            def child(queue):
                try:
                    import openvino_telemetry
                except ImportError:
                    queue.put("blocked")
                else:
                    queue.put("imported")


            if __name__ == "__main__":
                context = multiprocessing.get_context({method!r})
                queue = context.Queue()
                process = context.Process(target=child, args=(queue,))
                process.start()
                print(queue.get(timeout=60))
                process.join()
            """
        ),
        encoding="utf-8",
    )
    result = run(
        [sys.executable, "-P", str(script)], env=child_env(SOURCE_ROOT, site), cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "blocked"
    assert not marker.exists()


def test_a_child_without_yonga_gets_the_switch_from_its_command_line(stand_in, tmp_path):
    """An import check or a managed export environment never imports yonga."""
    site, marker = stand_in
    code = textwrap.dedent(
        """
        import json, sys
        try:
            import openvino_telemetry
        except ImportError:
            print(json.dumps(sys.argv))
        """
    )
    result = run(
        python_code_argv(sys.executable, code, "model-type", ""),
        env=child_env(site),
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["-c", "model-type", ""]
    assert not marker.exists()


def test_a_module_runs_as_it_would_under_dash_m(stand_in, tmp_path):
    """Same ``__name__``, same ``sys.argv``, same exit status as ``python -P -m``."""
    site, marker = stand_in
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "fake_cli.py").write_text(
        textwrap.dedent(
            """
            import json
            import sys

            try:
                import openvino_telemetry
                state = "imported"
            except ImportError:
                state = "blocked"

            if __name__ == "__main__":
                print(json.dumps({"argv": sys.argv, "telemetry": state}))
                sys.exit(3)
            """
        ),
        encoding="utf-8",
    )
    env = child_env(modules, site)
    args = ("export", "openvino", "--model", "m [x]", "out dir")

    wrapped = run(python_module_argv(sys.executable, "fake_cli", *args), env=env, cwd=tmp_path)
    assert not marker.exists()
    plain = run([sys.executable, "-P", "-m", "fake_cli", *args], env=env, cwd=tmp_path)

    assert wrapped.returncode == plain.returncode == 3
    wrapped_doc, plain_doc = json.loads(wrapped.stdout), json.loads(plain.stdout)
    assert wrapped_doc["argv"] == plain_doc["argv"]
    assert wrapped_doc["argv"][0].endswith("fake_cli.py")
    assert (wrapped_doc["telemetry"], plain_doc["telemetry"]) == ("blocked", "imported")


def test_the_bootstrap_leaves_no_names_behind():
    namespace: dict[str, object] = {}
    exec(TELEMETRY_BOOTSTRAP, namespace)
    assert set(namespace) == {"__builtins__"}


def test_openvino_and_nncf_fall_back_to_their_stubs(tmp_path):
    """Against the installed packages: the real reporting code is never loaded."""
    pytest.importorskip("openvino")
    code = textwrap.dedent(
        """
        import json, sys
        import yonga
        import openvino
        from openvino.tools.ovc import telemetry_utils
        result = {"ovc": telemetry_utils.tm.__name__}
        try:
            from nncf.telemetry import wrapper
        except ImportError:
            result["nncf"] = None
        else:
            result["nncf"] = type(wrapper.telemetry).__name__
        result["loaded"] = sys.modules.get("openvino_telemetry") is not None
        print(json.dumps(result))
        """
    )
    # Run outside pytest: NNCF switches its own telemetry off whenever pytest is
    # imported. CI=true is only a safety net, so a regression fails this test
    # without reporting anything: the package would still be loaded.
    env = child_env(SOURCE_ROOT)
    env["CI"] = "true"
    result = run([sys.executable, "-P", "-c", code], env=env, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout.strip().splitlines()[-1])
    assert document["ovc"] == "openvino.tools.ovc.telemetry_stub"
    # NNCF's stub is ``unittest.mock.MagicMock``; the real one is ``NNCFTelemetry``.
    assert document["nncf"] in (None, "MagicMock")
    assert document["loaded"] is False
