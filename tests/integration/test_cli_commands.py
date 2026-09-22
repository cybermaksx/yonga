"""The real ``yonga`` executable, run as a subprocess on this host.

Everything else about the CLI is tested in process with fakes. These tests exist
for what only a real process can prove: that the console entry point starts, that
it works on a host where OpenVINO cannot be imported, that ``doctor`` grades the
actual machine, and that the commands with no side effects leave the exit codes
``docs/CLI_SPEC.md`` section 15 promises.

Each subprocess gets its own XDG directories, so a test can never read or write
the developer's prepared artifacts, logs or configuration.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from importlib.util import find_spec
from pathlib import Path

import pytest

from yonga import __version__
from yonga.domain.diagnostics import ExitCode

#: Long enough for a cold import of OpenVINO and a full device enumeration.
COMMAND_TIMEOUT_SECONDS = 180

#: A ``sitecustomize`` that makes OpenVINO unimportable, whatever is installed.
BLOCK_OPENVINO = '''
"""Refuse to import the accelerator stack, to prove the CLI does not need it."""
import sys


class _Blocked:
    """Meta-path finder that fails every OpenVINO-family import."""

    BLOCKED = ("openvino", "openvino_genai", "openvino_tokenizers", "optimum", "nncf", "torch")

    def find_module(self, fullname, path=None):
        return None

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in self.BLOCKED:
            raise ImportError(f"{root} is blocked by the test harness")
        return None


sys.meta_path.insert(0, _Blocked())
'''


def run(
    args: Sequence[str],
    *,
    home: Path,
    extra_env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``python -m yonga`` with private directories and captured output."""
    env: dict[str, str] = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(home),
        "YONGA_CONFIG_DIR": str(home / "config"),
        "YONGA_DATA_DIR": str(home / "data"),
        "YONGA_CACHE_DIR": str(home / "cache"),
        "HF_HOME": str(home / "hf"),
        "NO_COLOR": "1",
        "COLUMNS": "200",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
    }
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-m", "yonga", *args],
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        env=env,
        check=False,
    )


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """A private home directory for one subprocess."""
    root = tmp_path / "home"
    root.mkdir()
    return root


# ======================================================================
# starting up
# ======================================================================
def test_help_exits_zero(home: Path) -> None:
    """The entry point starts and prints the command list."""
    result = run(["--help"], home=home)
    assert result.returncode == int(ExitCode.SUCCESS)
    assert "doctor" in result.stdout
    assert "prepare" in result.stdout


def test_version_is_printed_exactly_once(home: Path) -> None:
    """``--version`` answers once.

    A second line would mean the entry point ran again in a re-imported
    ``__main__`` — the ``multiprocessing``/``forkserver`` failure mode the guards
    in ``__main__.py`` and ``cli.py`` exist to prevent.
    """
    result = run(["--version"], home=home)
    assert result.returncode == int(ExitCode.SUCCESS)
    assert result.stdout.splitlines() == [f"yonga {__version__}"]


def test_cli_works_without_openvino(home: Path, tmp_path: Path) -> None:
    """``--help``, ``--version`` and ``list`` work when OpenVINO cannot be imported."""
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    (blocker / "sitecustomize.py").write_text(BLOCK_OPENVINO, encoding="utf-8")
    source = Path(__file__).resolve().parents[2] / "src"
    extra = {"PYTHONPATH": f"{blocker}:{source}"}

    for args in (["--help"], ["--version"], ["list"], ["registry", "validate"]):
        result = run(args, home=home, extra_env=extra)
        assert result.returncode == int(ExitCode.SUCCESS), (args, result.stderr)
        assert "Traceback" not in result.stderr


# ======================================================================
# doctor on this machine
# ======================================================================
def test_doctor_grades_this_host(home: Path) -> None:
    """``doctor`` returns 0 or 2 where OpenVINO works, and 10 where it does not."""
    result = run(["doctor"], home=home)
    assert "Traceback" not in result.stderr
    if find_spec("openvino") is None:
        assert result.returncode == int(ExitCode.ENVIRONMENT_UNUSABLE)
        return
    assert result.returncode in (int(ExitCode.SUCCESS), int(ExitCode.WARNINGS))
    assert "OpenVINO" in result.stdout
    assert "Devices" in result.stdout
    assert "Storage" in result.stdout


@pytest.mark.openvino
def test_doctor_json_is_one_parseable_document(home: Path) -> None:
    """``doctor --json`` writes exactly one document, even with a plugin loaded."""
    result = run(["doctor", "--json"], home=home)
    assert result.returncode in (int(ExitCode.SUCCESS), int(ExitCode.WARNINGS))
    assert result.stdout.count("\n") == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["exit_code"] == result.returncode
    assert isinstance(payload["sections"], list)
    names = {device["name"] for device in payload["hardware"]["devices"]}
    assert "CPU" in names


@pytest.mark.openvino
def test_doctor_sees_the_real_accelerators(home: Path) -> None:
    """On the reference host every device is reported with its real name."""
    result = run(["doctor", "--json"], home=home)
    payload = json.loads(result.stdout)
    devices = {device["name"]: device["full_name"] for device in payload["hardware"]["devices"]}
    assert devices.get("CPU")
    for name, full_name in devices.items():
        assert full_name, f"{name} was reported without a device name"


# ======================================================================
# read-only commands
# ======================================================================
def test_registry_validate_exits_zero(home: Path) -> None:
    """The shipped registry validates on a clean installation."""
    result = run(["registry", "validate"], home=home)
    assert result.returncode == int(ExitCode.SUCCESS)
    assert "Registry is valid." in result.stdout


def test_registry_paths_reports_the_private_directories(home: Path) -> None:
    """The managed directories follow the environment the process was given."""
    result = run(["registry", "paths"], home=home)
    assert result.returncode == int(ExitCode.SUCCESS)
    assert str(home / "data" / "artifacts") in result.stdout
    assert str(home / "config") in result.stdout


def test_list_is_empty_on_a_fresh_home(home: Path) -> None:
    """A new installation has no artifacts and says how to make one."""
    result = run(["list"], home=home)
    assert result.returncode == int(ExitCode.SUCCESS)
    assert "yonga prepare MODEL" in result.stdout

    as_json = run(["list", "--json"], home=home)
    assert as_json.returncode == int(ExitCode.SUCCESS)
    assert json.loads(as_json.stdout) == {
        "ok": True,
        "artifacts": [],
        "count": 0,
        "artifacts_dir": str(home / "data" / "artifacts"),
    }


def test_logs_on_a_fresh_home(home: Path) -> None:
    """``logs`` with nothing to show is not an error."""
    result = run(["logs"], home=home)
    assert result.returncode == int(ExitCode.SUCCESS)
    assert "No job logs" in result.stdout


def test_unknown_command_is_a_usage_error(home: Path) -> None:
    """An unknown command is Click's usage error, not a traceback."""
    result = run(["definitely-not-a-command"], home=home)
    assert result.returncode != int(ExitCode.SUCCESS)
    assert "Traceback" not in result.stderr


def test_probe_without_an_artifact_is_explained(home: Path) -> None:
    """A reference that matches nothing is a configuration error with advice."""
    result = run(["probe", "nobody/never-prepared"], home=home)
    assert result.returncode == int(ExitCode.REGISTRY)
    assert "yonga list" in result.stderr
    assert "Traceback" not in result.stderr


def test_chat_without_an_artifact_names_the_prepare_command(home: Path) -> None:
    """``chat`` refuses to convert anything and prints the command that would."""
    result = run(["chat", "nobody/never-prepared"], home=home)
    assert result.returncode == int(ExitCode.REGISTRY)
    assert "yonga prepare nobody/never-prepared" in " ".join(result.stderr.split())
    assert "Traceback" not in result.stderr
