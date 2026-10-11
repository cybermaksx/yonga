"""``yonga self install`` / ``self uninstall`` and the doctor's install check.

Everything happens under ``tmp_path``: a fake AppImage (an ELF header carrying
the type-2 AppImage magic, then a payload), a fake ``APPDIR`` holding the
"bundled" interpreter path, a private ``HOME`` and an explicit ``PATH``. No test
reads or writes the real ``~/.local/bin``, and no fake file is ever executed.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from collections.abc import Mapping
from pathlib import Path

import click
import pytest
from typer.main import get_command
from typer.testing import CliRunner

from tests.unit.test_cli import invoke, one_document
from yonga.application.doctor import _installation_check
from yonga.application.self_install import (
    RECEIPT_FILENAME,
    CommandRelation,
    InstallAction,
    InstallKind,
    SelfInstallService,
    TargetKind,
    detect_installation,
    inspect_target,
    is_appimage,
    on_path,
)
from yonga.cli import app
from yonga.config.paths import Paths
from yonga.domain.diagnostics import ConfigurationError, ExitCode, Severity, YongaError

#: ELF identification of an x86-64 executable with ``AI\x02`` in the padding.
APPIMAGE_HEADER = b"\x7fELF\x02\x01\x01\x00AI\x02\x00\x00\x00\x00\x00"


def fake_appimage(path: Path, payload: bytes = b"yonga 1.0") -> Path:
    """Write a file that passes the AppImage check and cannot run as anything."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(APPIMAGE_HEADER + payload)
    path.chmod(0o755)
    return path


class Host:
    """A private HOME, AppImage, APPDIR and data directory under ``tmp_path``."""

    def __init__(self, tmp_path: Path, *, payload: bytes = b"yonga 2.0") -> None:
        self.root = tmp_path
        self.home = tmp_path / "home"
        self.home.mkdir(parents=True, exist_ok=True)
        self.image = fake_appimage(tmp_path / "Downloads" / "yonga-2.0-x86_64.AppImage", payload)
        self.appdir = tmp_path / "mount"
        self.python = self.appdir / "usr" / "python" / "bin" / "python3"
        self.python.parent.mkdir(parents=True)
        self.python.write_text("")
        self.paths = Paths(
            config_dir=tmp_path / "config",
            cache_dir=tmp_path / "cache",
            data_dir=tmp_path / "data",
        )

    @property
    def default_bin(self) -> Path:
        return self.home / ".local" / "bin"

    def environ(self, *, path: str = "/usr/bin:/bin", appimage: bool = True) -> dict[str, str]:
        env = {"HOME": str(self.home), "PATH": path}
        if appimage:
            env.update({"APPIMAGE": str(self.image), "APPDIR": str(self.appdir)})
        return env

    def service(
        self, environ: Mapping[str, str] | None = None, *, version: str = "2.0"
    ) -> SelfInstallService:
        return SelfInstallService(
            self.paths,
            environ=self.environ() if environ is None else environ,
            executable=str(self.python),
            prefix=str(self.python.parent.parent),
            version=version,
        )


@pytest.fixture
def host(tmp_path: Path) -> Host:
    return Host(tmp_path)


# ======================================================================
# detection
# ======================================================================
def test_appimage_magic_is_recognised_without_running_anything(tmp_path: Path) -> None:
    image = fake_appimage(tmp_path / "a.AppImage")
    elf = tmp_path / "plain-elf"
    elf.write_bytes(b"\x7fELF\x02\x01\x01\x00\x00\x00\x00")
    script = tmp_path / "script"
    script.write_text("#!/usr/bin/python3\nprint('hi')\n")
    link = tmp_path / "link"
    link.symlink_to(image)
    assert is_appimage(image)
    assert not is_appimage(elf)
    assert not is_appimage(script)
    assert not is_appimage(link), "a link is never followed"
    assert not is_appimage(tmp_path)
    assert not is_appimage(tmp_path / "missing")


def test_targets_are_classified_by_lstat(tmp_path: Path) -> None:
    fake_appimage(tmp_path / "image")
    (tmp_path / "script").write_text("#!/bin/sh\n")
    (tmp_path / "dir").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "image")
    os.mkfifo(tmp_path / "fifo")
    kinds = {entry.name: inspect_target(entry).kind for entry in tmp_path.iterdir()}
    assert kinds == {
        "image": TargetKind.APPIMAGE,
        "script": TargetKind.FILE,
        "dir": TargetKind.DIRECTORY,
        "link": TargetKind.SYMLINK,
        "fifo": TargetKind.OTHER,
    }
    assert inspect_target(tmp_path / "absent").kind is TargetKind.MISSING


def test_an_appimage_is_detected_only_from_inside_its_mount(host: Host) -> None:
    running = detect_installation(
        host.environ(), executable=str(host.python), prefix="/p", version="2.0"
    )
    assert running.kind is InstallKind.APPIMAGE
    assert running.appimage == host.image.resolve()


def test_an_inherited_appimage_variable_is_not_believed(host: Host) -> None:
    """A pip yonga started from another AppImage's terminal sees that app's APPIMAGE."""
    running = detect_installation(
        host.environ(), executable=sys.executable, prefix=sys.prefix, version="2.0"
    )
    assert running.kind is InstallKind.PYTHON
    assert running.note is not None and "inherited" in running.note


def test_appimage_must_name_a_real_appimage_file(host: Host) -> None:
    environ = host.environ()
    environ["APPIMAGE"] = str(host.root / "missing.AppImage")
    running = detect_installation(environ, executable=str(host.python), prefix="/p")
    assert running.kind is InstallKind.PYTHON


def test_path_entries_are_compared_resolved(tmp_path: Path) -> None:
    real = tmp_path / "bin"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real)
    assert on_path(real, {"PATH": f"/usr/bin:{alias}"})
    assert not on_path(real, {"PATH": "/usr/bin:bin:."})


# ======================================================================
# install
# ======================================================================
def test_install_copies_atomically_with_mode_0755(host: Host) -> None:
    service = host.service()
    plan = service.plan_install(None)
    assert plan.action is InstallAction.INSTALL
    assert plan.create_directory
    result = service.install(plan)
    target = host.default_bin / "yonga"
    assert target.read_bytes() == host.image.read_bytes()
    assert stat.S_IMODE(target.stat().st_mode) == 0o755
    assert result.created_directory
    assert [p.name for p in host.default_bin.iterdir()] == ["yonga"], "no temp file is left"
    receipt = json.loads((host.paths.data_dir / RECEIPT_FILENAME).read_text())
    assert receipt["installs"][str(target)]["version"] == "2.0"


def test_same_content_is_already_installed(host: Host) -> None:
    service = host.service()
    service.install(service.plan_install(None))
    target = host.default_bin / "yonga"
    before = target.stat()
    plan = service.plan_install(None)
    assert plan.action is InstallAction.CURRENT
    service.install(plan)
    assert target.stat().st_ino == before.st_ino, "an identical copy is not rewritten"


def test_an_older_appimage_is_replaced_and_its_version_named(host: Host) -> None:
    older = Host(host.root / "older", payload=b"yonga 1.0, a different build")
    older.home = host.home
    older.paths = host.paths
    old_service = older.service(older.environ(), version="1.0")
    old_service.install(old_service.plan_install(host.default_bin))

    plan = host.service().plan_install(host.default_bin)
    assert plan.action is InstallAction.REPLACE
    assert plan.previous_version == "1.0"
    host.service().install(plan)
    assert (host.default_bin / "yonga").read_bytes() == host.image.read_bytes()


def test_an_unknown_appimage_has_an_unknown_version(host: Host) -> None:
    fake_appimage(host.default_bin / "yonga", b"someone else's copy")
    plan = host.service().plan_install(None)
    assert plan.action is InstallAction.REPLACE
    assert plan.previous_version is None


@pytest.mark.parametrize("kind", ["script", "symlink", "directory"])
def test_anything_but_an_appimage_is_refused_and_left_alone(host: Host, kind: str) -> None:
    target = host.default_bin / "yonga"
    host.default_bin.mkdir(parents=True)
    if kind == "script":
        target.write_text("#!/home/u/.venv/bin/python\nfrom yonga.cli import run\n")
    elif kind == "symlink":
        target.symlink_to(fake_appimage(host.root / "elsewhere.AppImage", b"linked"))
    else:
        target.mkdir()
    before = inspect_target(target)
    with pytest.raises(ConfigurationError) as caught:
        host.service().plan_install(None)
    assert caught.value.diagnostic.code == "self.target.not_appimage"
    assert inspect_target(target) == before
    if kind == "symlink":
        assert (host.root / "elsewhere.AppImage").read_bytes().endswith(b"linked")


def test_a_target_swapped_after_planning_is_not_overwritten(host: Host) -> None:
    service = host.service()
    plan = service.plan_install(None)
    host.default_bin.mkdir(parents=True)
    (host.default_bin / "yonga").write_text("#!/bin/sh\n")
    with pytest.raises(ConfigurationError):
        service.install(plan)
    assert (host.default_bin / "yonga").read_text() == "#!/bin/sh\n"
    assert [p.name for p in host.default_bin.iterdir()] == ["yonga"]


def test_outside_an_appimage_install_is_refused_with_the_pip_route(host: Host) -> None:
    service = host.service(host.environ(appimage=False))
    with pytest.raises(ConfigurationError) as caught:
        service.plan_install(None)
    remediation = " ".join(caught.value.diagnostic.remediation)
    assert caught.value.diagnostic.code == "self.not_appimage"
    assert "pipx install 'yonga[openvino,export]'" in remediation
    assert "activate" in remediation
    assert not host.default_bin.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
def test_an_unwritable_bin_dir_is_refused_without_sudo(host: Host) -> None:
    locked = host.root / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        for directory in (locked, locked / "sub"):
            with pytest.raises(YongaError) as caught:
                host.service().plan_install(directory)
            item = caught.value.diagnostic
            assert item.code == "self.bin_dir.not_writable"
            assert caught.value.exit_code() is ExitCode.ENVIRONMENT_UNUSABLE
            assert "sudo" in item.message
            assert "--bin-dir" in " ".join(item.remediation)
    finally:
        locked.chmod(0o755)


def test_a_file_as_bin_dir_is_refused(host: Host) -> None:
    (host.root / "afile").write_text("")
    with pytest.raises(ConfigurationError) as caught:
        host.service().plan_install(host.root / "afile")
    assert caught.value.diagnostic.code == "self.bin_dir.not_a_directory"


# ======================================================================
# PATH
# ======================================================================
def test_a_bin_dir_off_path_gets_the_export_line_and_no_file_is_edited(host: Host) -> None:
    profile = host.home / ".profile"
    profile.write_text("# untouched\n")
    service = host.service()
    result = service.install(service.plan_install(None))
    report = service.path_report(host.default_bin, host.default_bin / "yonga", created=True)
    assert not report.on_path
    assert not report.command_is_target
    text = "\n".join(report.advice)
    assert 'export PATH="$HOME/.local/bin:$PATH"' in text
    assert "log out and back in" in text
    assert "never edits" in text
    assert profile.read_text() == "# untouched\n"
    assert result.created_directory


def test_a_bin_dir_on_path_resolves_to_the_copy(host: Host) -> None:
    environ = host.environ(path=f"/usr/bin:{host.home}/.local/bin")
    service = host.service(environ)
    service.install(service.plan_install(None))
    report = service.path_report(host.default_bin, host.default_bin / "yonga")
    assert report.on_path and report.command_is_target
    assert report.advice == []


def test_a_shadowing_command_earlier_on_path_is_named(host: Host) -> None:
    other = host.root / "venv" / "bin"
    other.mkdir(parents=True)
    (other / "yonga").write_text("#!/bin/sh\n")
    (other / "yonga").chmod(0o755)
    environ = host.environ(path=f"{other}:{host.default_bin}")
    service = host.service(environ)
    service.install(service.plan_install(None))
    report = service.path_report(host.default_bin, host.default_bin / "yonga")
    assert report.on_path and not report.command_is_target
    assert str(other / "yonga") in " ".join(report.advice)


# ======================================================================
# uninstall
# ======================================================================
def test_uninstall_removes_only_an_appimage(host: Host) -> None:
    service = host.service()
    service.install(service.plan_install(None))
    plan = service.plan_uninstall(None)
    assert plan.version == "2.0"
    assert service.uninstall(plan)
    assert not (host.default_bin / "yonga").exists()
    receipt = json.loads((host.paths.data_dir / RECEIPT_FILENAME).read_text())
    assert receipt["installs"] == {}


def test_uninstall_refuses_a_pip_script(host: Host) -> None:
    host.default_bin.mkdir(parents=True)
    (host.default_bin / "yonga").write_text("#!/usr/bin/python3\n")
    with pytest.raises(ConfigurationError):
        host.service().plan_uninstall(None)
    assert (host.default_bin / "yonga").exists()


def test_uninstall_with_nothing_there_is_a_no_op(host: Host) -> None:
    service = host.service(host.environ(appimage=False))
    plan = service.plan_uninstall(None)
    assert plan.target.kind is TargetKind.MISSING
    assert not service.uninstall(plan)


# ======================================================================
# doctor
# ======================================================================
def test_doctor_warns_when_the_appimage_is_not_on_path(host: Host) -> None:
    check = _installation_check(host.service().status())
    assert check.status is Severity.WARNING
    assert check.value == "AppImage"
    assert str(host.image.resolve()) in (check.detail or "")
    assert any("self install" in step for step in check.remediation)


def test_doctor_is_happy_when_path_runs_a_copy_of_this_appimage(host: Host) -> None:
    service = host.service(host.environ(path=str(host.default_bin)))
    service.install(service.plan_install(None))
    status = service.status()
    assert status.relation is CommandRelation.THIS
    assert _installation_check(status).status is Severity.OK


def test_doctor_names_an_older_appimage_on_path(host: Host) -> None:
    older = fake_appimage(host.root / "old" / "yonga", b"old build")
    status = host.service(host.environ(path=str(older.parent))).status()
    assert status.relation is CommandRelation.OTHER_APPIMAGE
    check = _installation_check(status)
    assert check.status is Severity.WARNING
    assert str(older) in (check.detail or "")
    assert any("--bin-dir" in step for step in check.remediation)


def test_doctor_is_informational_for_a_python_environment(host: Host) -> None:
    status = host.service(host.environ(appimage=False, path="/nonexistent")).status()
    check = _installation_check(status)
    assert status.running.kind is InstallKind.PYTHON
    assert check.status is Severity.INFO
    assert check.value == "Python environment"


# ======================================================================
# the CLI
# ======================================================================
@pytest.fixture
def cli_host(host: Host, monkeypatch: pytest.MonkeyPatch) -> Host:
    """Make this process look like the AppImage's bundled interpreter."""
    monkeypatch.setattr(sys, "executable", str(host.python))
    return host


def cli_env(host: Host, **overrides: str) -> dict[str, str]:
    env = host.environ()
    env.update(overrides)
    return env


def test_cli_install_twice_then_uninstall(cli_host: Host) -> None:
    runner = CliRunner()
    home = cli_host.root / "yonga-home"
    (home / "config").mkdir(parents=True)
    env = cli_env(cli_host)

    first = invoke(runner, home, ["self", "install"], env=env)
    assert first.exit_code == int(ExitCode.WARNINGS), first.output
    assert "Installed yonga" in first.stdout
    assert 'export PATH="$HOME/.local/bin:$PATH"' in first.stderr

    second = invoke(runner, home, ["self", "install", "--json"], env=env)
    document = one_document(second)
    assert document["action"] == "already_installed"
    assert document["bin_dir_on_path"] is False

    on_path = cli_env(cli_host, PATH=str(cli_host.default_bin))
    third = invoke(runner, home, ["self", "install"], env=on_path)
    assert third.exit_code == int(ExitCode.SUCCESS)
    assert "already installed" in third.stdout

    keep = home / "data" / "artifacts" / "model.txt"
    keep.parent.mkdir(parents=True)
    keep.write_text("prepared model")
    declined = invoke(runner, home, ["self", "uninstall"], env=env)
    assert declined.exit_code == int(ExitCode.CANCELLED)
    assert (cli_host.default_bin / "yonga").exists()

    removed = invoke(runner, home, ["self", "uninstall", "--yes"], env=env)
    assert removed.exit_code == int(ExitCode.SUCCESS)
    assert not (cli_host.default_bin / "yonga").exists()
    assert "cache prune --all" in removed.stdout
    assert str(home / "data") in removed.stdout
    assert keep.read_text() == "prepared model"


def test_cli_replacing_needs_yes_when_nobody_can_answer(cli_host: Host) -> None:
    runner = CliRunner()
    home = cli_host.root / "yonga-home"
    (home / "config").mkdir(parents=True)
    bin_dir = cli_host.root / "bin"
    old = fake_appimage(bin_dir / "yonga", b"older")
    env = cli_env(cli_host)

    refused = invoke(runner, home, ["self", "install", "--bin-dir", str(bin_dir)], env=env)
    assert refused.exit_code == int(ExitCode.CANCELLED)
    assert old.read_bytes().endswith(b"older")
    assert "version unknown" in refused.stdout

    replaced = invoke(
        runner, home, ["self", "install", "--bin-dir", str(bin_dir), "--yes", "--json"], env=env
    )
    document = one_document(replaced)
    assert document["action"] == "replaced"
    assert old.read_bytes() == cli_host.image.read_bytes()


def test_cli_refuses_outside_an_appimage(cli_host: Host) -> None:
    runner = CliRunner()
    home = cli_host.root / "yonga-home"
    (home / "config").mkdir(parents=True)
    env = {"HOME": str(cli_host.home), "PATH": "/usr/bin", "APPIMAGE": "", "APPDIR": ""}
    result = invoke(runner, home, ["self", "install", "--json"], env=env)
    assert result.exit_code == int(ExitCode.REGISTRY)
    document = one_document(result)
    diagnostic = document["diagnostic"]
    assert isinstance(diagnostic, dict)
    assert diagnostic["code"] == "self.not_appimage"
    assert not cli_host.default_bin.exists()


def test_cli_refuses_a_symlink_target(cli_host: Host) -> None:
    runner = CliRunner()
    home = cli_host.root / "yonga-home"
    (home / "config").mkdir(parents=True)
    cli_host.default_bin.mkdir(parents=True)
    venv_script = cli_host.root / "venv-yonga"
    venv_script.write_text("#!/bin/sh\n")
    (cli_host.default_bin / "yonga").symlink_to(venv_script)
    result = invoke(runner, home, ["self", "install", "--yes"], env=cli_env(cli_host))
    assert result.exit_code == int(ExitCode.REGISTRY)
    assert "symbolic link" in result.stderr
    assert (cli_host.default_bin / "yonga").is_symlink()
    assert venv_script.read_text() == "#!/bin/sh\n"


def test_cli_doctor_reports_the_install_check(cli_host: Host) -> None:
    runner = CliRunner()
    home = cli_host.root / "yonga-home"
    (home / "config").mkdir(parents=True)
    result = invoke(runner, home, ["doctor", "--json"], env=cli_env(cli_host))
    document = one_document(result)
    installation = document["installation"]
    assert isinstance(installation, dict)
    assert installation["kind"] == "appimage"
    assert installation["path_command_relation"] == "missing"
    assert result.exit_code == int(ExitCode.WARNINGS)


def _bin_dir_help(command: str) -> str:
    group = get_command(app)
    assert isinstance(group, click.Group)
    self_group = group.commands["self"]
    assert isinstance(self_group, click.Group)
    option = next(p for p in self_group.commands[command].params if p.name == "bin_dir")
    assert isinstance(option, click.Option)
    return option.help or ""


def test_cli_uninstall_help_says_remove_not_install() -> None:
    assert "remove the 'yonga' command from" in _bin_dir_help("uninstall")
    assert "install the 'yonga' command in" in _bin_dir_help("install")


def test_cli_uninstall_from_a_python_environment_says_what_stays(cli_host: Host) -> None:
    """Run from pip, the AppImage goes and this installation's yonga stays."""
    runner = CliRunner()
    home = cli_host.root / "yonga-home"
    (home / "config").mkdir(parents=True)
    bin_dir = cli_host.root / "fakebin"
    copy = fake_appimage(bin_dir / "yonga", b"an AppImage copy")
    env = {"HOME": str(cli_host.home), "PATH": "/usr/bin", "APPIMAGE": "", "APPDIR": ""}

    removed = invoke(
        runner, home, ["self", "uninstall", "--bin-dir", str(bin_dir), "--yes"], env=env
    )

    assert removed.exit_code == int(ExitCode.SUCCESS), removed.output
    assert not copy.exists()
    text = " ".join(removed.stdout.split())
    assert "runs from a Python environment" in text
    assert "removes only the AppImage" in text
    assert "downloaded AppImage" not in text
    assert "this Python environment's yonga still runs them" in text


def test_cli_uninstall_from_an_appimage_points_at_the_download(cli_host: Host) -> None:
    runner = CliRunner()
    home = cli_host.root / "yonga-home"
    (home / "config").mkdir(parents=True)
    bin_dir = cli_host.root / "fakebin"
    fake_appimage(bin_dir / "yonga", b"an AppImage copy")

    removed = invoke(
        runner,
        home,
        ["self", "uninstall", "--bin-dir", str(bin_dir), "--yes"],
        env=cli_env(cli_host),
    )

    assert removed.exit_code == int(ExitCode.SUCCESS), removed.output
    text = " ".join(removed.stdout.split())
    assert "Python environment" not in text
    assert "run them from the downloaded AppImage" in text
