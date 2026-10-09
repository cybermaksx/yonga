"""Unit tests for the read-only ``doctor`` application service.

Every probe is a fake, so these tests describe policy only: which status a fact
earns, which remediation it carries, and which exit code the CLI must use. The
fakes deliberately raise from every mutating adapter method (``download``,
``compile``, ``read_ir``, ``read_tokenizer``), which turns "doctor is strictly
read-only" into something the suite actually enforces rather than documents.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from yonga.application.doctor import (
    AUTH_SOURCE_KEY,
    COMPILE_CACHE_WARN_BYTES,
    COMPILE_CACHE_WARN_FLOOR_BYTES,
    COMPILE_CACHE_WARN_FREE_FRACTION,
    MANAGED_TEMP_NOTE,
    DoctorReport,
    DoctorService,
    group_membership_remediation,
)
from yonga.config.settings import Settings
from yonga.domain.artifacts import IRDescription, ProbeResult, assert_no_secrets
from yonga.domain.compile_cache import CacheEntryStatus, CompileCacheEntry, CompileCacheInventory
from yonga.domain.diagnostics import ExitCode, HealthCheck, Severity
from yonga.domain.hardware import (
    DeviceKind,
    FilesystemInfo,
    HardwareInventory,
    HostInfo,
    HuggingFaceAuthInfo,
    MemoryInfo,
    NPUNodeInfo,
    OpenVINODevice,
    PackageInfo,
)
from yonga.domain.models import ModelIdentity, ModelMetadata
from yonga.registry import load_builtin_registry

GIB = 1024**3

#: Versions verified on the real target host (OpenVINO 2026.4 stack).
HOST_VERSIONS: dict[str, str] = {
    "openvino": "2026.4.0",
    "openvino-genai": "2026.4.0.0",
    "openvino-tokenizers": "2026.4.0.0",
    "optimum-intel": "2.2.0",
    "nncf": "3.4.0",
    "transformers": "5.5.4",
    "huggingface-hub": "1.21.0",
}

#: A signed-out Hub client with a known cache directory: the common default.
ANONYMOUS_AUTH = HuggingFaceAuthInfo(cache_dir=Path("/home/user/.cache/huggingface/hub"))

#: NPU device properties exactly as this machine reports them.
NPU_PROPERTIES: dict[str, str] = {
    "FULL_DEVICE_NAME": "Intel(R) AI Boost",
    "DEVICE_ARCHITECTURE": "4000",
    "NPU_DRIVER_VERSION": "1789049548",
    "NPU_COMPILER_VERSION": "524291",
    "NPU_COMPILER_TYPE": "CompilerType.PREFER_PLUGIN",
}


# --- fakes ----------------------------------------------------------------------------


class FakeHardwareProbe:
    """A :class:`~yonga.adapters.HardwareProbe` that replays a fixed inventory."""

    def __init__(self, inventory: HardwareInventory | None = None, *, error: str | None = None):
        self._inventory = inventory if inventory is not None else HardwareInventory()
        self._error = error
        self.full_flags: list[bool] = []

    def inventory(self, *, full: bool = False) -> HardwareInventory:
        """Record the ``full`` flag and return the canned inventory."""
        self.full_flags.append(full)
        if self._error is not None:
            raise RuntimeError(self._error)
        return self._inventory


class FakeRuntimeProbe:
    """A :class:`~yonga.adapters.RuntimeProbe` whose compile paths are forbidden."""

    def __init__(self, devices: Sequence[str] = (), *, error: str | None = None):
        self._devices = list(devices)
        self._error = error
        self.calls: list[str] = []

    def available_devices(self) -> list[str]:
        """Return the canned device names, or raise the canned failure."""
        self.calls.append("available_devices")
        if self._error is not None:
            raise RuntimeError(self._error)
        return list(self._devices)

    def read_ir(self, path: Path) -> IRDescription:
        """Doctor must never read an IR."""
        raise AssertionError(f"doctor must not read IR ({path})")

    def compile(
        self,
        path: Path,
        device: str,
        properties: Mapping[str, str] | None = None,
    ) -> ProbeResult:
        """Doctor must never compile anything."""
        raise AssertionError(f"doctor must not compile ({path} on {device})")

    def read_tokenizer(self, artifact: Path) -> ProbeResult:
        """Doctor must never touch an artifact."""
        raise AssertionError(f"doctor must not read a tokenizer ({artifact})")


class FakeHubClient:
    """A :class:`~yonga.adapters.HubClient` that only answers ``auth_info``."""

    def __init__(self, auth: HuggingFaceAuthInfo | None = None, *, error: str | None = None):
        self._auth = auth if auth is not None else ANONYMOUS_AUTH
        self._error = error
        self.calls: list[str] = []

    def auth_info(self) -> HuggingFaceAuthInfo:
        """Return the canned authentication state."""
        self.calls.append("auth_info")
        if self._error is not None:
            raise RuntimeError(self._error)
        return self._auth

    def resolve_revision(self, model: ModelIdentity) -> str:
        """Doctor must never reach the Hub."""
        raise AssertionError(f"doctor must not resolve a revision ({model.repo_id})")

    def inspect(self, model: ModelIdentity, *, with_sizes: bool = True) -> ModelMetadata:
        """Doctor must never reach the Hub."""
        raise AssertionError(f"doctor must not inspect a repository ({model.repo_id})")

    def download(
        self,
        source: ModelIdentity,
        *,
        ctx: object | None = None,
        allow_patterns: Sequence[str] | None = None,
        ignore_patterns: Sequence[str] | None = None,
    ) -> Path:
        """Doctor must never download anything."""
        raise AssertionError(f"doctor must not download ({source.repo_id})")


# --- builders -------------------------------------------------------------------------


def make_settings(tmp_path: Path) -> Settings:
    """Settings whose managed directories live under ``tmp_path``."""
    return Settings(
        config_dir=tmp_path / "config",
        cache_dir=tmp_path / "cache",
        data_dir=tmp_path / "data",
    )


def packages(overrides: Mapping[str, str | None] | None = None) -> list[PackageInfo]:
    """Build the seven probed packages, with per-name version overrides."""
    merged: dict[str, str | None] = dict(HOST_VERSIONS)
    merged.update(overrides or {})
    return [
        PackageInfo(name=name, version=version, importable=version is not None)
        for name, version in merged.items()
    ]


def devices(*, npu: bool = True) -> list[OpenVINODevice]:
    """The three devices this host exposes, with their real full names."""
    found = [
        OpenVINODevice(
            name="CPU",
            kind=DeviceKind.CPU,
            full_name="Intel(R) Core(TM) Ultra 7 268V",
            architecture="intel64",
        ),
        OpenVINODevice(
            name="GPU",
            kind=DeviceKind.GPU,
            full_name="Intel(R) Arc(TM) Graphics (iGPU)",
            architecture="GPU: vendor=0x8086 arch=v20.4.4",
        ),
    ]
    if npu:
        found.append(
            OpenVINODevice(
                name="NPU",
                kind=DeviceKind.NPU,
                full_name="Intel(R) AI Boost",
                architecture="4000",
                properties=dict(NPU_PROPERTIES),
            )
        )
    return found


def filesystems(
    settings: Settings,
    *,
    system_temp_fstype: str = "tmpfs",
    system_temp_bytes: int = 16 * GIB,
) -> list[FilesystemInfo]:
    """The filesystems the Linux probe measures, in the same order."""
    paths = settings.paths
    return [
        FilesystemInfo(
            path=Path("/tmp"),
            mount_point=Path("/tmp"),
            fstype=system_temp_fstype,
            total_bytes=system_temp_bytes,
            free_bytes=system_temp_bytes,
            available_bytes=system_temp_bytes,
        ),
        FilesystemInfo(
            path=paths.tmp_dir,
            mount_point=Path("/"),
            fstype="ext4",
            total_bytes=935 * GIB,
            free_bytes=750 * GIB,
            available_bytes=743 * GIB,
        ),
        FilesystemInfo(
            path=paths.artifacts_dir,
            mount_point=Path("/"),
            fstype="ext4",
            total_bytes=935 * GIB,
            free_bytes=750 * GIB,
            available_bytes=743 * GIB,
        ),
        FilesystemInfo(
            path=Path("/"),
            mount_point=Path("/"),
            fstype="ext4",
            total_bytes=935 * GIB,
            free_bytes=750 * GIB,
            available_bytes=743 * GIB,
        ),
    ]


def npu_node(
    *,
    readable: bool = True,
    writable: bool = True,
    group_name: str | None = "render",
    user_in_group: bool = True,
) -> NPUNodeInfo:
    """A ``/dev/accel/accel0`` snapshot with adjustable accessibility."""
    return NPUNodeInfo(
        directory_exists=True,
        node_path=Path("/dev/accel/accel0"),
        node_exists=True,
        readable=readable,
        writable=writable,
        group_name=group_name,
        user_in_group=user_in_group,
        user_groups=["alice", "render"] if user_in_group else ["alice"],
        driver_version=None,
        compiler_libraries=[
            "/usr/lib/x86_64-linux-gnu/libopenvino_intel_npu_compiler.so",
            "/usr/lib/x86_64-linux-gnu/libopenvino_intel_npu_compiler_loader.so",
        ],
    )


def healthy_inventory(settings: Settings, **overrides: object) -> HardwareInventory:
    """The inventory this machine really produces, with per-test overrides."""
    measured = filesystems(settings)
    base: dict[str, object] = {
        "host": HostInfo(
            platform="Linux",
            machine="x86_64",
            kernel="7.2.6-070206-generic",
            python_version="3.14.4",
            distro_id="ubuntu",
            distro_version="26.04",
            distro_pretty="Ubuntu 26.04.1 LTS",
            cpu_model="Intel(R) Core(TM) Ultra 7 268V",
            cpu_count=8,
        ),
        "memory": MemoryInfo(
            total_bytes=33 * GIB,
            available_bytes=24 * GIB,
            swap_total_bytes=8 * GIB,
            swap_free_bytes=8 * GIB,
        ),
        "filesystems": measured,
        "devices": devices(),
        "openvino_available": True,
        "npu": npu_node(),
        "packages": packages(),
        "device_tags": ["intel", "npu", "lunar_lake"],
        "system_temp": measured[0],
        "managed_temp": measured[1],
    }
    return HardwareInventory.model_validate(base).model_copy(update=dict(overrides))


def build(
    settings: Settings,
    inventory: HardwareInventory,
    *,
    device_names: Sequence[str] = ("CPU", "GPU", "NPU"),
    auth: HuggingFaceAuthInfo | None = None,
    registry: object | None = None,
) -> tuple[DoctorService, FakeHardwareProbe, FakeRuntimeProbe, FakeHubClient]:
    """Wire a service around fakes and return it together with them."""
    hardware = FakeHardwareProbe(inventory)
    runtime = FakeRuntimeProbe(device_names)
    hub = FakeHubClient(auth)
    service = DoctorService(hardware, runtime, hub, registry, settings)  # type: ignore[arg-type]
    return service, hardware, runtime, hub


def _string_values(value: object) -> list[str]:
    """Every string anywhere inside a JSON-shaped structure."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for entry in value.values() for item in _string_values(entry)]
    if isinstance(value, list):
        return [item for entry in value for item in _string_values(entry)]
    return []


def _all_keys(value: object) -> set[str]:
    """Every mapping key anywhere inside a JSON-shaped structure."""
    if isinstance(value, dict):
        found = set(value)
        for entry in value.values():
            found |= _all_keys(entry)
        return found
    if isinstance(value, list):
        found = set()
        for entry in value:
            found |= _all_keys(entry)
        return found
    return set()


def require(report: DoctorReport, check_id: str) -> HealthCheck:
    """Return one check by id, failing the test when it was not produced."""
    check = report.check(check_id)
    assert check is not None, f"{check_id} was not produced"
    return check


def remediation_text(report: DoctorReport, check_id: str) -> str:
    """Join one check's value, detail and remediation into a searchable string."""
    check = require(report, check_id)
    return " ".join([check.value or "", check.detail or "", *check.remediation])


# --- OpenVINO absent ------------------------------------------------------------------


def test_missing_openvino_still_reports_and_exits_ten(tmp_path):
    settings = make_settings(tmp_path)
    inventory = healthy_inventory(
        settings,
        devices=[],
        openvino_available=False,
        openvino_import_error="No module named 'openvino'",
        npu=None,
        packages=packages({"openvino": None, "openvino-genai": None, "openvino-tokenizers": None}),
    )
    service, _, _, _ = build(settings, inventory, device_names=())

    report = service.run()

    assert report.exit_code() is ExitCode.ENVIRONMENT_UNUSABLE
    assert report.runtime_usable is False
    runtime_check = report.check("devices.runtime")
    assert runtime_check is not None
    assert runtime_check.status is Severity.ERROR
    assert "yonga[openvino]" in " ".join(runtime_check.remediation)
    assert "No module named 'openvino'" in (runtime_check.detail or "")
    # The rest of the report is still produced: doctor reports, it does not crash.
    assert [section.id for section in report.sections] == [
        "host",
        "openvino",
        "devices",
        "npu",
        "storage",
        "huggingface",
    ]
    assert report.check("host.cpu") is not None
    assert report.check("storage.managed_temp") is not None


def test_runtime_probe_that_raises_is_reported_not_propagated(tmp_path):
    settings = make_settings(tmp_path)
    inventory = healthy_inventory(settings, devices=[], openvino_available=False)
    hardware = FakeHardwareProbe(inventory)
    runtime = FakeRuntimeProbe(error="Level0 initialisation failed")
    service = DoctorService(hardware, runtime, FakeHubClient(), None, settings)

    report = service.run()

    assert report.exit_code() is ExitCode.ENVIRONMENT_UNUSABLE
    assert "Level0 initialisation failed" in " ".join(report.notes)


def test_hardware_probe_failure_degrades_to_a_report(tmp_path):
    settings = make_settings(tmp_path)
    service = DoctorService(FakeHardwareProbe(error="psutil exploded"), None, None, None, settings)

    report = service.run()

    assert report.exit_code() is ExitCode.ENVIRONMENT_UNUSABLE
    assert any("psutil exploded" in note for note in report.notes)
    assert report.sections


# --- healthy host ---------------------------------------------------------------------


def test_all_three_devices_present_exits_zero(tmp_path):
    settings = make_settings(tmp_path)
    # A disk-backed system temp so nothing at all is flagged.
    inventory = healthy_inventory(settings)
    measured = filesystems(settings, system_temp_fstype="ext4", system_temp_bytes=900 * GIB)
    inventory = inventory.model_copy(
        update={"filesystems": measured, "system_temp": measured[0], "managed_temp": measured[1]}
    )
    service, _, _, _ = build(settings, inventory)

    report = service.run()

    assert report.runtime_usable is True
    assert report.exit_code() is ExitCode.SUCCESS
    assert report.problems == []
    for name, full_name in (
        ("cpu", "Intel(R) Core(TM) Ultra 7 268V"),
        ("gpu", "Intel(R) Arc(TM) Graphics (iGPU)"),
        ("npu", "Intel(R) AI Boost"),
    ):
        check = report.check(f"devices.{name}")
        assert check is not None
        assert check.status is Severity.OK
        assert check.value == full_name


def test_healthy_host_reports_the_real_npu_facts(tmp_path):
    settings = make_settings(tmp_path)
    service, _, _, _ = build(settings, healthy_inventory(settings))

    report = service.run()

    assert require(report, "npu.driver").value == "1789049548"
    assert require(report, "npu.compiler_version").value == "524291"
    assert require(report, "npu.compiler_in_plugin").value == "available"
    assert require(report, "npu.compiler_libraries").status is Severity.OK
    node = require(report, "npu.node")
    assert node.status is Severity.OK
    assert node.label == "/dev/accel/accel0"


def test_host_section_reports_os_kernel_python_cpu_ram_and_swap(tmp_path):
    settings = make_settings(tmp_path)
    service, _, _, _ = build(settings, healthy_inventory(settings))

    report = service.run()

    host = report.section("host")
    assert host is not None
    assert [check.id for check in host.checks] == [
        "host.os",
        "host.kernel",
        "host.python",
        "host.cpu",
        "host.ram",
        "host.swap",
    ]
    assert require(report, "host.os").value == "Ubuntu 26.04.1 LTS"
    assert require(report, "host.python").value == "3.14.4"
    assert "GiB" in (require(report, "host.ram").value or "")
    assert "GiB" in (require(report, "host.swap").value or "")


def test_no_npu_on_this_host_is_informational(tmp_path):
    settings = make_settings(tmp_path)
    inventory = healthy_inventory(
        settings,
        devices=devices(npu=False),
        npu=NPUNodeInfo(notes=["/dev/accel does not exist"]),
    )
    service, _, _, _ = build(settings, inventory, device_names=("CPU", "GPU"))

    report = service.run()

    present = report.check("npu.present")
    assert present is not None
    assert present.status is Severity.INFO
    assert report.exit_code() is ExitCode.WARNINGS  # the tmpfs warning, not the NPU


# --- ABI family (SPEC 5.3) ------------------------------------------------------------


def test_aligned_abi_family_is_ok(tmp_path):
    settings = make_settings(tmp_path)
    service, _, _, _ = build(settings, healthy_inventory(settings))

    report = service.run()

    check = report.check("openvino.abi_family")
    assert check is not None
    assert check.status is Severity.OK
    assert check.value == "aligned (2026.4)"


def test_abi_family_mismatch_warns_and_names_both_versions(tmp_path):
    settings = make_settings(tmp_path)
    inventory = healthy_inventory(settings, packages=packages({"openvino-genai": "2025.3.0.0"}))
    service, _, _, _ = build(settings, inventory)

    report = service.run()

    check = report.check("openvino.abi_family")
    assert check is not None
    assert check.status is Severity.WARNING
    text = remediation_text(report, "openvino.abi_family")
    assert "2026.4.0" in text
    assert "2025.3.0.0" in text
    assert "openvino-genai" in text
    # Doctor explains the fix, it never performs it.
    assert any("pip install" in line for line in check.remediation)
    assert any("never installs" in line for line in check.remediation)
    assert report.exit_code() is ExitCode.WARNINGS


def test_abi_family_is_unknown_when_nothing_is_installed(tmp_path):
    settings = make_settings(tmp_path)
    inventory = healthy_inventory(
        settings,
        packages=packages({"openvino": None, "openvino-genai": None, "openvino-tokenizers": None}),
    )
    service, _, _, _ = build(settings, inventory)

    report = service.run()

    check = report.check("openvino.abi_family")
    assert check is not None
    assert check.status is Severity.UNKNOWN
    assert check.remediation


def test_missing_openvino_package_is_an_error_and_missing_nncf_a_warning(tmp_path):
    settings = make_settings(tmp_path)
    inventory = healthy_inventory(settings, packages=packages({"openvino": None, "nncf": None}))
    service, _, _, _ = build(settings, inventory)

    report = service.run()

    assert require(report, "openvino.openvino").status is Severity.ERROR
    assert require(report, "openvino.nncf").status is Severity.WARNING
    assert "yonga[export]" in " ".join(require(report, "openvino.nncf").remediation)


def test_a_missing_export_package_names_the_managed_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("APPIMAGE", raising=False)
    settings = make_settings(tmp_path)
    inventory = healthy_inventory(settings, packages=packages({"optimum-intel": None}))
    service, _, _, _ = build(settings, inventory)

    remediation = " ".join(require(service.run(), "openvino.optimum_intel").remediation)

    assert "managed export environment" in remediation
    assert "yonga[export]" in remediation


def test_inside_an_appimage_no_pip_install_is_suggested_for_export(tmp_path, monkeypatch):
    """Nothing can be installed into the image; the managed environment is the answer."""
    monkeypatch.setenv("APPIMAGE", "/home/user/yonga.AppImage")
    settings = make_settings(tmp_path)
    inventory = healthy_inventory(settings, packages=packages({"optimum-intel": None}))
    service, _, _, _ = build(settings, inventory)

    remediation = " ".join(require(service.run(), "openvino.optimum_intel").remediation)

    assert "managed export environment" in remediation
    assert "pip install" not in remediation


# --- NPU permissions ------------------------------------------------------------------


def test_inaccessible_npu_node_warns_naming_the_group_and_mutates_nothing(tmp_path):
    settings = make_settings(tmp_path)
    node = npu_node(readable=False, writable=False, group_name="render", user_in_group=False)
    inventory = healthy_inventory(settings, npu=node)
    before_node = node.model_dump(mode="json")
    before_tree = sorted(path.name for path in tmp_path.iterdir())
    service, _, runtime, hub = build(settings, inventory)

    report = service.run()

    node_check = report.check("npu.node")
    assert node_check is not None
    assert node_check.status is Severity.WARNING
    joined = " ".join(node_check.remediation)
    assert "'render'" in joined
    assert "log out and back in" in joined

    group_check = report.check("npu.group")
    assert group_check is not None
    assert group_check.status is Severity.WARNING
    assert group_check.value == "render"
    assert "'render'" in " ".join(group_check.remediation)

    # Strictly read-only: the probed state is untouched, nothing was compiled or
    # downloaded, and no file appeared on disk.
    assert node.model_dump(mode="json") == before_node
    assert runtime.calls == ["available_devices"]
    assert hub.calls == ["auth_info"]
    assert sorted(path.name for path in tmp_path.iterdir()) == before_tree
    # A usable runtime with a permission warning is exit 2, never 10.
    assert report.exit_code() is ExitCode.WARNINGS


def test_group_membership_remediation_never_promises_to_act():
    lines = group_membership_remediation("render")

    assert "add your user to the 'render' group" in lines[0]
    assert "log out and back in" in lines[0]
    assert any("never runs sudo" in line for line in lines)
    # The sudo line is a suggestion the user runs, explicitly labelled as such.
    assert any(line.startswith("Suggested command, to run yourself:") for line in lines)


def test_missing_accel_node_with_an_npu_device_warns(tmp_path):
    settings = make_settings(tmp_path)
    inventory = healthy_inventory(settings, npu=NPUNodeInfo(notes=["/dev/accel does not exist"]))
    service, _, _, _ = build(settings, inventory)

    report = service.run()

    check = report.check("npu.node")
    assert check is not None
    assert check.status is Severity.WARNING
    assert "intel_vpu" in " ".join(check.remediation)


# --- storage --------------------------------------------------------------------------


def test_small_tmpfs_warns_with_the_managed_temp_note_and_is_never_fatal(tmp_path):
    settings = make_settings(tmp_path)
    service, _, _, _ = build(settings, healthy_inventory(settings))

    report = service.run()

    system_temp = require(report, "storage.system_temp")
    assert system_temp.status is Severity.WARNING
    # ...and the section as a whole never escalates past a warning.
    storage = report.section("storage")
    assert storage is not None
    assert storage.status is Severity.WARNING
    assert system_temp.value == "16.0 GiB tmpfs"
    assert MANAGED_TEMP_NOTE in system_temp.remediation
    # The remediation tells the user to do nothing, not to grow the tmpfs.
    assert any("No action is required" in line for line in system_temp.remediation)
    assert any("Do not enlarge the tmpfs" in line for line in system_temp.remediation)

    managed = report.check("storage.managed_temp")
    assert managed is not None
    assert managed.status is Severity.OK
    assert "disk-backed" in (managed.value or "")
    assert MANAGED_TEMP_NOTE in (managed.detail or "")

    # ~750 GiB free on / keeps the cache and artifact filesystems healthy.
    assert require(report, "storage.cache").status is Severity.OK
    assert require(report, "storage.artifacts").status is Severity.OK
    assert report.runtime_usable is True
    assert report.exit_code() is ExitCode.WARNINGS


def test_a_ram_backed_managed_temp_is_a_warning(tmp_path):
    settings = make_settings(tmp_path)
    measured = filesystems(settings)
    measured[1] = measured[1].model_copy(update={"fstype": "tmpfs"})
    inventory = healthy_inventory(
        settings, filesystems=measured, managed_temp=measured[1], system_temp=measured[0]
    )
    service, _, _, _ = build(settings, inventory)

    report = service.run()

    managed = report.check("storage.managed_temp")
    assert managed is not None
    assert managed.status is Severity.WARNING
    assert "cache_dir" in " ".join(managed.remediation)


def test_a_nearly_full_artifact_filesystem_is_an_error(tmp_path):
    settings = make_settings(tmp_path)
    measured = filesystems(settings)
    measured[2] = measured[2].model_copy(
        update={"free_bytes": GIB // 2, "available_bytes": GIB // 2}
    )
    inventory = healthy_inventory(settings, filesystems=measured)
    service, _, _, _ = build(settings, inventory)

    report = service.run()

    check = report.check("storage.artifacts")
    assert check is not None
    assert check.status is Severity.ERROR
    assert check.remediation
    # An out-of-space artifact directory does not make the runtime unusable.
    assert report.runtime_usable is True
    assert report.exit_code() is ExitCode.WARNINGS


# --- Hugging Face ---------------------------------------------------------------------


def test_hugging_face_section_reports_state_without_a_token(tmp_path):
    settings = make_settings(tmp_path)
    auth = HuggingFaceAuthInfo(
        logged_in=True,
        user="someone",
        token_source="environment",
        offline=False,
        cache_dir=tmp_path / "hf",
    )
    service, _, _, _ = build(settings, healthy_inventory(settings), auth=auth)

    report = service.run()

    check = report.check("huggingface.auth")
    assert check is not None
    assert check.status is Severity.OK
    assert check.value == "signed in as someone"
    assert "environment" in (check.detail or "")
    assert require(report, "huggingface.cache").value == str(tmp_path / "hf")
    assert require(report, "huggingface.offline").value == "disabled"


def test_anonymous_hugging_face_access_is_informational(tmp_path):
    settings = make_settings(tmp_path)
    service, _, _, _ = build(settings, healthy_inventory(settings))

    report = service.run()

    check = report.check("huggingface.auth")
    assert check is not None
    assert check.status is Severity.INFO
    assert check.value == "anonymous"


def test_offline_setting_is_surfaced(tmp_path):
    settings = make_settings(tmp_path).model_copy(update={"offline": True})
    service, _, _, _ = build(settings, healthy_inventory(settings))

    report = service.run()

    assert require(report, "huggingface.offline").value == "enabled"


def test_a_failing_hub_client_is_noted_not_raised(tmp_path):
    settings = make_settings(tmp_path)
    hub = FakeHubClient(error="token file unreadable")
    service = DoctorService(
        FakeHardwareProbe(healthy_inventory(settings)),
        FakeRuntimeProbe(("CPU",)),
        hub,
        None,
        settings,
    )

    report = service.run()

    assert any("token file unreadable" in note for note in report.notes)
    assert report.check("huggingface.auth") is not None


# --- serialisation --------------------------------------------------------------------


def test_report_json_is_serialisable_and_carries_no_secret(tmp_path):
    settings = make_settings(tmp_path)
    auth = HuggingFaceAuthInfo(
        logged_in=True,
        user="someone",
        token_source="environment",
        offline=False,
        cache_dir=tmp_path / "hf",
    )
    service, _, _, _ = build(settings, healthy_inventory(settings), auth=auth)

    payload = service.run().to_json()

    # The project's own gate: no key anywhere may name a credential.
    assert_no_secrets(payload)
    json.dumps(payload)  # the document really is serialisable
    assert "token_source" not in _all_keys(payload)
    # ...and no value looks like a Hugging Face or OpenAI style credential.
    leaked = [
        value for value in _string_values(payload) if value.startswith(("hf_", "api_org_", "sk-"))
    ]
    assert leaked == []
    hardware = payload["hardware"]
    assert isinstance(hardware, dict)
    hf_auth = hardware["hf_auth"]
    assert isinstance(hf_auth, dict)
    assert hf_auth[AUTH_SOURCE_KEY] == "environment"
    assert payload["exit_code"] == int(ExitCode.WARNINGS)
    assert payload["status"] == Severity.WARNING.value


def test_report_json_round_trips_through_the_model(tmp_path):
    settings = make_settings(tmp_path)
    service, _, _, _ = build(settings, healthy_inventory(settings))

    report = service.run()
    restored = DoctorReport.model_validate(report.model_dump(mode="json"))

    assert restored.status is report.status
    assert restored.exit_code() is report.exit_code()
    assert [section.id for section in restored.sections] == [
        section.id for section in report.sections
    ]


# --- invariants -----------------------------------------------------------------------


@pytest.mark.parametrize("broken", [False, True])
def test_every_warning_and_error_carries_remediation(tmp_path, broken):
    settings = make_settings(tmp_path)
    if broken:
        measured = filesystems(settings)
        measured[2] = measured[2].model_copy(
            update={"free_bytes": GIB // 4, "available_bytes": GIB // 4}
        )
        inventory = healthy_inventory(
            settings,
            devices=[],
            openvino_available=False,
            openvino_import_error="No module named 'openvino'",
            npu=npu_node(readable=False, writable=False, user_in_group=False),
            packages=packages({"openvino-genai": "2025.3.0.0", "transformers": None}),
            filesystems=measured,
            memory=MemoryInfo(total_bytes=4 * GIB, available_bytes=GIB),
        )
        service, _, _, _ = build(settings, inventory, device_names=())
    else:
        service, _, _, _ = build(settings, healthy_inventory(settings))

    report = service.run()

    offenders = [check.id for check in report.problems if not check.remediation]
    assert offenders == []


def test_full_mode_probes_deeply_and_adds_the_registry_section(tmp_path):
    settings = make_settings(tmp_path)
    registry = load_builtin_registry()
    hardware = FakeHardwareProbe(healthy_inventory(settings))
    service = DoctorService(
        hardware, FakeRuntimeProbe(("CPU",)), FakeHubClient(), registry, settings
    )

    report = service.run(full=True)

    assert hardware.full_flags == [True]
    assert report.full is True
    section = report.section("registry")
    assert section is not None
    assert require(report, "registry.validation").status is Severity.OK
    assert require(report, "registry.rules").value == f"{len(registry.all_rules())} rules"


def test_default_mode_omits_the_registry_section(tmp_path):
    settings = make_settings(tmp_path)
    hardware = FakeHardwareProbe(healthy_inventory(settings))
    service = DoctorService(
        hardware, FakeRuntimeProbe(("CPU",)), FakeHubClient(), load_builtin_registry(), settings
    )

    report = service.run()

    assert hardware.full_flags == [False]
    assert report.section("registry") is None


def test_service_works_without_any_optional_adapter(tmp_path):
    settings = make_settings(tmp_path)
    service = DoctorService(FakeHardwareProbe(healthy_inventory(settings)), settings=settings)

    report = service.run()

    # No runtime probe: the inventory alone still proves the devices are there.
    assert report.runtime_usable is True
    assert report.check("devices.npu") is not None
    # No Hub client: the authentication state is unknown, and says why.
    auth_check = require(report, "huggingface.auth")
    assert auth_check.status is Severity.UNKNOWN
    assert auth_check.remediation


@pytest.mark.parametrize(
    ("usable", "status", "expected"),
    [
        (True, Severity.OK, ExitCode.SUCCESS),
        (True, Severity.INFO, ExitCode.SUCCESS),
        (True, Severity.WARNING, ExitCode.WARNINGS),
        (True, Severity.ERROR, ExitCode.WARNINGS),
        (True, Severity.UNKNOWN, ExitCode.WARNINGS),
        (False, Severity.OK, ExitCode.ENVIRONMENT_UNUSABLE),
        (False, Severity.ERROR, ExitCode.ENVIRONMENT_UNUSABLE),
    ],
)
def test_exit_codes_follow_the_cli_spec(usable, status, expected):
    report = DoctorReport(runtime_usable=usable, status=status)

    assert report.exit_code() is expected


# --- compile cache ---------------------------------------------------------------------


def cache_inventory(settings: Settings, **sizes: int) -> CompileCacheInventory:
    """A compile cache holding one entry per status keyword, with the given sizes."""
    statuses = {
        "in_use": CacheEntryStatus.IN_USE,
        "orphaned": CacheEntryStatus.ORPHANED,
        "stale": CacheEntryStatus.STALE_RUNTIME,
        "unattributed": CacheEntryStatus.UNATTRIBUTED,
    }
    root = settings.paths.compile_cache_dir
    entries = [
        CompileCacheEntry(
            key=f"{index:016x}", path=root / f"{index:016x}", size_bytes=size, status=statuses[name]
        )
        for index, (name, size) in enumerate(sizes.items())
    ]
    return CompileCacheInventory(
        root=root,
        store_root=settings.paths.artifacts_dir,
        installed_openvino="2026.4.0",
        entries=entries,
    )


def doctor_with_cache(
    settings: Settings, cache: object, *, inventory: HardwareInventory | None = None
) -> DoctorService:
    """A healthy host's doctor whose compile cache reads ``cache`` (or raises it)."""

    def read() -> CompileCacheInventory:
        if isinstance(cache, BaseException):
            raise cache
        assert isinstance(cache, CompileCacheInventory)
        return cache

    return DoctorService(
        FakeHardwareProbe(inventory or healthy_inventory(settings)),
        FakeRuntimeProbe(("CPU", "GPU", "NPU")),
        FakeHubClient(None),
        None,
        settings,
        compile_cache=read,
    )


def test_much_reclaimable_compile_cache_warns_with_the_prune_command(tmp_path):
    settings = make_settings(tmp_path)
    cache = cache_inventory(settings, in_use=40 * GIB, orphaned=8 * GIB, stale=4 * GIB)

    report = doctor_with_cache(settings, cache).run()

    check = require(report, "storage.compile_cache")
    assert check.status is Severity.WARNING
    assert check.value == "3 entries, 52.0 GiB, 12.0 GiB reclaimable"
    assert any("yonga cache prune" in line for line in check.remediation)
    assert report.compile_cache is not None
    assert report.compile_cache["reclaimable_bytes"] == 12 * GIB
    assert report.compile_cache["warn_threshold_bytes"] == COMPILE_CACHE_WARN_BYTES
    assert report.to_json()["compile_cache"] == report.compile_cache


def test_a_little_reclaimable_cache_on_a_roomy_disk_is_fine(tmp_path):
    settings = make_settings(tmp_path)
    cache = cache_inventory(settings, in_use=40 * GIB, orphaned=2 * GIB, unattributed=30 * GIB)

    check = require(doctor_with_cache(settings, cache).run(), "storage.compile_cache")

    assert check.status is Severity.OK
    assert "2.0 GiB reclaimable" in (check.value or "")


def test_reclaimable_cache_is_a_warning_when_it_is_a_quarter_of_the_free_space(tmp_path):
    settings = make_settings(tmp_path)
    tight = FilesystemInfo(
        path=Path("/"),
        mount_point=Path("/"),
        fstype="ext4",
        total_bytes=500 * GIB,
        free_bytes=6 * GIB,
        available_bytes=6 * GIB,
    )
    inventory = healthy_inventory(settings, filesystems=[tight], managed_temp=tight)
    small = cache_inventory(settings, orphaned=2 * GIB)
    tiny = cache_inventory(settings, orphaned=GIB // 2)

    warned = doctor_with_cache(settings, small, inventory=inventory).run()
    floor = doctor_with_cache(settings, tiny, inventory=inventory).run()

    assert COMPILE_CACHE_WARN_FREE_FRACTION * 6 * GIB < 2 * GIB < COMPILE_CACHE_WARN_BYTES
    assert require(warned, "storage.compile_cache").status is Severity.WARNING
    # Half a GiB is more than a quarter of nothing much, but under the floor.
    assert GIB // 2 < COMPILE_CACHE_WARN_FLOOR_BYTES
    assert require(floor, "storage.compile_cache").status is Severity.OK


def test_the_free_space_threshold_never_drops_below_the_floor(tmp_path):
    """On a nearly full disk a quarter of the free space is tiny; 1 GiB still applies."""
    settings = make_settings(tmp_path)
    full = FilesystemInfo(
        path=Path("/"),
        mount_point=Path("/"),
        fstype="ext4",
        total_bytes=500 * GIB,
        free_bytes=2 * GIB,
        available_bytes=2 * GIB,
    )
    inventory = healthy_inventory(settings, filesystems=[full], managed_temp=full)
    assert (
        COMPILE_CACHE_WARN_FREE_FRACTION * 2 * GIB < 3 * GIB // 4 < COMPILE_CACHE_WARN_FLOOR_BYTES
    )

    under = doctor_with_cache(
        settings, cache_inventory(settings, orphaned=3 * GIB // 4), inventory=inventory
    ).run()
    at = doctor_with_cache(
        settings,
        cache_inventory(settings, orphaned=COMPILE_CACHE_WARN_FLOOR_BYTES),
        inventory=inventory,
    ).run()

    assert require(under, "storage.compile_cache").status is Severity.OK
    assert under.compile_cache is not None
    assert under.compile_cache["warn_threshold_bytes"] == COMPILE_CACHE_WARN_FLOOR_BYTES
    assert require(at, "storage.compile_cache").status is Severity.WARNING


def test_an_empty_compile_cache_is_reported_as_such(tmp_path):
    settings = make_settings(tmp_path)

    report = doctor_with_cache(settings, cache_inventory(settings)).run()

    check = require(report, "storage.compile_cache")
    assert check.status is Severity.OK
    assert check.value == "empty"


def test_a_compile_cache_that_cannot_be_measured_never_fails_doctor(tmp_path):
    settings = make_settings(tmp_path)

    report = doctor_with_cache(settings, OSError("unreadable")).run()

    assert report.check("storage.compile_cache") is None
    assert report.compile_cache is None
