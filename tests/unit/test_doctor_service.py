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
from datetime import UTC, datetime
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
from yonga.application.drivers import SYSTEM_CHANGES_NOTE, DriverComponent, install_steps
from yonga.application.self_install import (
    CommandRelation,
    InstallationStatus,
    InstallKind,
    RunningInstallation,
)
from yonga.config.settings import Settings
from yonga.domain.artifacts import IRDescription, ProbeResult, assert_no_secrets
from yonga.domain.compile_cache import CacheEntryStatus, CompileCacheEntry, CompileCacheInventory
from yonga.domain.diagnostics import ExitCode, HealthCheck, HealthSection, Severity
from yonga.domain.environments import EnvironmentSpec, ExportEnvironment
from yonga.domain.hardware import (
    DeviceKind,
    DriverLibraries,
    FilesystemInfo,
    HardwareInventory,
    HostInfo,
    HuggingFaceAuthInfo,
    MemoryInfo,
    NPUNodeInfo,
    OpenVINODevice,
    PackageInfo,
    PciAccelerator,
)
from yonga.domain.models import ModelIdentity, ModelMetadata
from yonga.registry import load_builtin_registry
from yonga.registry.resolver import RegistryResolver

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


def require_section(report: DoctorReport, section_id: str) -> HealthSection:
    """Return a section that must exist, failing the test loudly otherwise."""
    section = report.section(section_id)
    assert section is not None, f"missing section {section_id!r}"
    return section


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


#: What an AppImage build bundles: the runtime, never the conversion toolchain.
NO_TOOLCHAIN: dict[str, str | None] = {"optimum-intel": None, "nncf": None, "transformers": None}


def installation(kind: InstallKind) -> InstallationStatus:
    """yonga running as ``kind``, with 'yonga' on PATH being this installation."""
    if kind is InstallKind.APPIMAGE:
        image = Path("/home/user/.local/bin/yonga")
        running = RunningInstallation(
            kind, "0.4.0", Path("/tmp/.mount_yonga/usr/python"), appimage=image
        )
        return InstallationStatus(running, image, CommandRelation.THIS)
    running = RunningInstallation(kind, "0.4.0", Path("/home/user/venv"))
    return InstallationStatus(running, Path("/home/user/venv/bin/yonga"), CommandRelation.THIS)


def built_environment(settings: Settings, size_bytes: int) -> ExportEnvironment:
    """One managed export environment on disk, as ``yonga env list`` reports it."""
    spec = EnvironmentSpec(python="3.14", packages={"transformers": "==5.5.4"})
    return ExportEnvironment(
        spec=spec,
        path=settings.paths.envs_dir / spec.name,
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
        size_bytes=size_bytes,
    )


def build_install(
    settings: Settings,
    inventory: HardwareInventory,
    kind: InstallKind,
    environments: Sequence[ExportEnvironment] = (),
) -> DoctorService:
    """A doctor that knows how yonga runs and which export environments exist."""
    status = installation(kind)
    return DoctorService(
        FakeHardwareProbe(inventory),
        FakeRuntimeProbe(("CPU", "GPU", "NPU")),
        FakeHubClient(None),
        None,
        settings,
        export_environments=lambda: list(environments),
        installation=lambda: status,
    )


def test_an_appimage_without_the_toolchain_is_one_ok_line_and_exits_zero(tmp_path):
    """The AppImage never carries the toolchain; by design, so never a warning."""
    settings = make_settings(tmp_path)
    # A disk-backed system temp, as in the E2E run, so nothing else is flagged.
    measured = filesystems(settings, system_temp_fstype="ext4", system_temp_bytes=900 * GIB)
    inventory = healthy_inventory(
        settings,
        packages=packages(NO_TOOLCHAIN),
        filesystems=measured,
        system_temp=measured[0],
        managed_temp=measured[1],
    )
    service = build_install(settings, inventory, InstallKind.APPIMAGE)

    report = service.run()

    assert report.problems == []
    for dist_name in ("optimum_intel", "nncf", "transformers"):
        assert report.check(f"openvino.{dist_name}") is None
    toolchain = require(report, "openvino.export_toolchain")
    assert toolchain.status is Severity.OK
    assert "built on first prepare" in (toolchain.value or "")
    assert "optimum-intel, nncf and transformers are not in the AppImage" in (
        toolchain.detail or ""
    )
    assert str(settings.paths.envs_dir) in (toolchain.detail or "")
    assert "pip install" not in json.dumps(report.to_json())
    labels = [check.id for check in require_section(report, "openvino").checks]
    assert (
        labels.index("openvino.export_toolchain")
        == labels.index("openvino.openvino_tokenizers") + 1
    )
    assert require_section(report, "openvino").status is Severity.OK
    assert report.exit_code() is ExitCode.SUCCESS
    assert report.summary() is None


def test_an_appimage_names_the_export_environments_already_built(tmp_path):
    settings = make_settings(tmp_path)
    inventory = healthy_inventory(settings, packages=packages(NO_TOOLCHAIN))
    environment = built_environment(settings, 3 * GIB // 2)
    service = build_install(settings, inventory, InstallKind.APPIMAGE, [environment])

    toolchain = require(service.run(), "openvino.export_toolchain")

    assert toolchain.value == "managed environment, 1 built"
    assert "1.5 GiB" in (toolchain.detail or "")


def test_environment_disk_counts_the_base_python_like_env_list(tmp_path):
    """doctor and ``yonga env list`` total the same bytes: environments plus the base copy."""
    settings = make_settings(tmp_path)
    inventory = healthy_inventory(settings, packages=packages(NO_TOOLCHAIN))
    environment = built_environment(settings, int(1.42 * GIB))
    base = 98 * 1024**2
    service = DoctorService(
        FakeHardwareProbe(inventory),
        FakeRuntimeProbe(("CPU", "GPU", "NPU")),
        FakeHubClient(None),
        None,
        settings,
        export_environments=lambda: [environment],
        base_interpreter_bytes=lambda: base,
        installation=lambda: installation(InstallKind.APPIMAGE),
    )

    report = service.run()

    storage = require(report, "storage.export_environments")
    assert storage.value == "1 built, 1.5 GiB"
    assert "including a 98.0 MiB copy of the bundled Python" in (storage.detail or "")
    assert "(1.5 GiB now)" in (require(report, "openvino.export_toolchain").detail or "")


def test_a_base_python_alone_is_still_disk_in_use(tmp_path):
    """An unused base copy is what ``env list`` shows with no environment; doctor too."""
    settings = make_settings(tmp_path)
    inventory = healthy_inventory(settings)
    service = DoctorService(
        FakeHardwareProbe(inventory),
        FakeRuntimeProbe(("CPU", "GPU", "NPU")),
        FakeHubClient(None),
        None,
        settings,
        export_environments=list,
        base_interpreter_bytes=lambda: 98 * 1024**2,
    )

    storage = require(service.run(), "storage.export_environments")

    assert storage.value == "none built, 98.0 MiB"


def test_an_appimage_reports_a_bundled_export_package_as_usual(tmp_path):
    settings = make_settings(tmp_path)
    inventory = healthy_inventory(
        settings, packages=packages({"optimum-intel": None, "nncf": None})
    )
    service = build_install(settings, inventory, InstallKind.APPIMAGE)

    report = service.run()

    assert require(report, "openvino.transformers").status is Severity.OK
    toolchain = require(report, "openvino.export_toolchain")
    assert (toolchain.detail or "").startswith("optimum-intel and nncf are not in the AppImage")


def test_a_python_environment_still_warns_with_the_export_extra(tmp_path, monkeypatch):
    """An inherited APPIMAGE variable does not make a venv an AppImage."""
    monkeypatch.setenv("APPIMAGE", "/home/user/other.AppImage")
    settings = make_settings(tmp_path)
    inventory = healthy_inventory(settings, packages=packages(NO_TOOLCHAIN))
    service = build_install(settings, inventory, InstallKind.PYTHON)

    report = service.run()

    assert report.check("openvino.export_toolchain") is None
    nncf = require(report, "openvino.nncf")
    assert nncf.status is Severity.WARNING
    assert "yonga[export]" in " ".join(nncf.remediation)
    assert report.exit_code() is ExitCode.WARNINGS


# --- the closing line -----------------------------------------------------------------


def test_a_missing_runtime_is_named_as_the_cause_with_its_install_command(tmp_path):
    settings = make_settings(tmp_path)
    inventory = healthy_inventory(
        settings,
        devices=[],
        openvino_available=False,
        openvino_import_error="No module named 'openvino'",
        packages=packages({"openvino": None, "openvino-genai": None, "openvino-tokenizers": None}),
    )
    service, _, _, _ = build(settings, inventory, device_names=())

    summary = service.run().summary()

    assert summary == "The OpenVINO runtime is not installed: pip install 'yonga[openvino]'"


def test_a_runtime_that_does_not_import_is_not_called_missing(tmp_path):
    settings = make_settings(tmp_path)
    inventory = healthy_inventory(
        settings,
        devices=[],
        openvino_available=False,
        openvino_import_error="ImportError: libopenvino.so.2026: cannot open shared object",
    )
    service, _, _, _ = build(settings, inventory, device_names=())

    summary = service.run().summary() or ""

    assert "installed but could not be imported" in summary
    assert "pip install" not in summary


def test_a_runtime_without_devices_points_at_the_host(tmp_path):
    settings = make_settings(tmp_path)
    inventory = healthy_inventory(settings, devices=[])
    service, _, _, _ = build(settings, inventory, device_names=())

    report = service.run()

    assert report.exit_code() is ExitCode.ENVIRONMENT_UNUSABLE
    assert report.summary() == "No usable OpenVINO device was found on this host."


def test_warnings_get_the_usable_with_warnings_line(tmp_path):
    settings = make_settings(tmp_path)
    inventory = healthy_inventory(settings, packages=packages({"nncf": None}))
    service, _, _, _ = build(settings, inventory)

    assert service.run().summary() == "Usable, with warnings above."


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


def test_an_accessible_node_does_not_ask_for_group_membership(tmp_path):
    """A world-accessible node (no udev rule yet) needs no group: no sudo advice."""
    settings = make_settings(tmp_path)
    node = npu_node(readable=True, writable=True, group_name="render", user_in_group=False)
    service, _, _, _ = build(settings, healthy_inventory(settings, npu=node))

    report = service.run()

    group_check = require(report, "npu.group")
    assert group_check.status is Severity.INFO
    assert group_check.remediation == []
    assert require(report, "npu.node").status is Severity.OK


@pytest.mark.parametrize(
    # DRIVER is what the CachyOS laptop reported; PREFER_PLUGIN is OpenVINO 2026's
    # default and only a preference, so version 0 must warn under it too.
    "compiler_type",
    ["CompilerType.DRIVER", "CompilerType.PREFER_PLUGIN"],
)
def test_a_driver_reporting_compiler_version_zero_warns_that_the_compiler_is_missing(
    tmp_path, compiler_type
):
    settings = make_settings(tmp_path)
    found = devices()
    properties = dict(NPU_PROPERTIES, NPU_COMPILER_VERSION="0", NPU_COMPILER_TYPE=compiler_type)
    found[2] = found[2].model_copy(update={"properties": properties})
    node = npu_node().model_copy(update={"compiler_libraries": []})
    service, _, _, _ = build(settings, healthy_inventory(settings, devices=found, npu=node))

    report = service.run()

    check = require(report, "npu.compiler_version")
    assert check.status is Severity.WARNING
    assert check.value == "missing"
    # The test host is Ubuntu: the Debian family's package is named.
    assert check.remediation == install_steps(DriverComponent.NPU_COMPILER, inventory_host())
    assert "intel-driver-compiler-npu" in " ".join(check.remediation)
    # The libraries line gives the same advice for the same cause.
    libraries = require(report, "npu.compiler_libraries")
    assert libraries.status is Severity.WARNING
    assert libraries.remediation == check.remediation


def npu_with(settings: Settings, *, version: str, libraries: list[str]) -> HardwareInventory:
    """The healthy inventory with the NPU's compiler version and libraries replaced."""
    found = devices()
    found[2] = found[2].model_copy(
        update={"properties": dict(NPU_PROPERTIES, NPU_COMPILER_VERSION=version)}
    )
    node = npu_node().model_copy(update={"compiler_libraries": libraries})
    return healthy_inventory(settings, devices=found, npu=node)


def test_half_of_the_compiler_pair_is_not_enough(tmp_path):
    settings = make_settings(tmp_path)
    inventory = npu_with(
        settings, version="0", libraries=["/usr/lib/libopenvino_intel_npu_compiler.so"]
    )
    service, _, _, _ = build(settings, inventory)

    check = require(service.run(), "npu.compiler_libraries")

    assert check.status is Severity.WARNING
    assert "libopenvino_intel_npu_compiler_loader.so" in (check.value or "")
    assert check.remediation == install_steps(DriverComponent.NPU_COMPILER, inventory_host())


def test_a_compiler_that_answers_needs_no_library_on_the_searched_paths(tmp_path):
    settings = make_settings(tmp_path)
    service, _, _, _ = build(settings, npu_with(settings, version="524291", libraries=[]))

    check = require(service.run(), "npu.compiler_libraries")

    assert check.status is Severity.INFO
    assert "524291" in (check.detail or "")


def test_versioned_library_names_count_as_the_pair(tmp_path):
    settings = make_settings(tmp_path)
    libraries = [
        "/usr/lib64/libopenvino_intel_npu_compiler.so.1",
        "/usr/lib64/libopenvino_intel_npu_compiler_loader.so.1.38.0",
    ]
    service, _, _, _ = build(settings, npu_with(settings, version="0", libraries=libraries))

    assert require(service.run(), "npu.compiler_libraries").status is Severity.OK


def test_a_real_compiler_version_is_reported_as_a_fact(tmp_path):
    settings = make_settings(tmp_path)
    service, _, _, _ = build(settings, healthy_inventory(settings))

    check = require(service.run(), "npu.compiler_version")

    assert check.status is Severity.OK
    assert check.value == "524291"


# --- drivers --------------------------------------------------------------------------


def inventory_host(**overrides: object) -> HostInfo:
    """The Ubuntu host :func:`healthy_inventory` describes, with overrides."""
    base = HostInfo(
        platform="Linux",
        machine="x86_64",
        distro_id="ubuntu",
        distro_like=["debian"],
        distro_pretty="Ubuntu 26.04.1 LTS",
    )
    return base.model_copy(update=dict(overrides))


def lunar_lake_accelerators(
    *, gpu_driver: str | None = "xe", npu_driver: str | None = "intel_vpu"
) -> list[PciAccelerator]:
    """The reference host's iGPU and NPU on the PCI bus."""
    return [
        PciAccelerator(
            kind=DeviceKind.GPU,
            address="0000:00:02.0",
            vendor_id="0x8086",
            device_id="0x64a0",
            kernel_driver=gpu_driver,
        ),
        PciAccelerator(
            kind=DeviceKind.NPU,
            address="0000:00:0b.0",
            vendor_id="0x8086",
            device_id="0x643e",
            kernel_driver=npu_driver,
        ),
    ]


def installed_driver_libraries(**overrides: list[str]) -> DriverLibraries:
    """Every user-space driver present, unless an override empties one."""
    values: dict[str, list[str]] = {
        "gpu_opencl": ["/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so"],
        "level_zero_loader": ["/usr/lib/x86_64-linux-gnu/libze_loader.so.1"],
        "npu_level_zero": ["/usr/lib/x86_64-linux-gnu/libze_intel_npu.so.1"],
    }
    values.update(overrides)
    return DriverLibraries(**values)


def drivers_inventory(settings: Settings, **overrides: object) -> HardwareInventory:
    """:func:`healthy_inventory` with the PCI bus and the driver libraries read."""
    base: dict[str, object] = {
        "host": inventory_host(),
        "accelerators": lunar_lake_accelerators(),
        "driver_libraries": installed_driver_libraries(),
    }
    base.update(overrides)
    return healthy_inventory(settings, **base)


def test_a_host_with_every_driver_installed_has_a_clean_drivers_section(tmp_path):
    settings = make_settings(tmp_path)
    service, _, _, _ = build(settings, drivers_inventory(settings))

    report = service.run()

    section = require_section(report, "drivers")
    assert section.status is Severity.OK
    assert [check.id for check in section.checks] == [
        "drivers.gpu.kernel",
        "drivers.gpu.runtime",
        "drivers.gpu.openvino",
        "drivers.npu.kernel",
        "drivers.npu.runtime",
        "drivers.npu.openvino",
    ]
    assert require(report, "drivers.gpu.kernel").value == "xe"
    assert "8086:64a0" in (require(report, "drivers.gpu.kernel").detail or "")
    # Drivers come straight after Devices.
    ids = [s.id for s in report.sections]
    assert ids.index("drivers") == ids.index("devices") + 1


def test_a_missing_gpu_runtime_names_the_package_of_this_distribution(tmp_path):
    settings = make_settings(tmp_path)
    inventory = drivers_inventory(
        settings,
        host=inventory_host(distro_id="cachyos", distro_like=["arch"]),
        driver_libraries=installed_driver_libraries(gpu_opencl=[]),
        devices=[d for d in devices() if d.name != "GPU"],
    )
    service, _, _, _ = build(settings, inventory, device_names=("CPU", "NPU"))

    report = service.run()

    runtime = require(report, "drivers.gpu.runtime")
    assert runtime.status is Severity.WARNING
    assert runtime.value == "missing"
    assert any("sudo pacman -S intel-compute-runtime" in line for line in runtime.remediation)
    assert runtime.remediation[-1] == SYSTEM_CHANGES_NOTE
    listed = require(report, "drivers.gpu.openvino")
    assert listed.status is Severity.WARNING
    assert "missing driver" in (listed.detail or "")


def test_a_missing_npu_runtime_on_ubuntu_points_at_intels_release(tmp_path):
    settings = make_settings(tmp_path)
    inventory = drivers_inventory(
        settings,
        driver_libraries=installed_driver_libraries(level_zero_loader=[], npu_level_zero=[]),
        devices=devices(npu=False),
    )
    service, _, _, _ = build(settings, inventory, device_names=("CPU", "GPU"))

    report = service.run()

    runtime = require(report, "drivers.npu.runtime")
    assert runtime.status is Severity.WARNING
    assert "libze_loader.so.1" in (runtime.detail or "")
    assert "libze_intel_npu.so" in (runtime.detail or "")
    joined = " ".join(runtime.remediation)
    assert "intel-level-zero-npu" in joined
    assert "github.com/intel/linux-npu-driver" in joined


def test_an_accelerator_without_a_kernel_driver_says_which_module_to_load(tmp_path):
    settings = make_settings(tmp_path)
    inventory = drivers_inventory(
        settings, accelerators=lunar_lake_accelerators(npu_driver=None), devices=devices(npu=False)
    )
    service, _, _, _ = build(settings, inventory, device_names=("CPU", "GPU"))

    report = service.run()

    kernel = require(report, "drivers.npu.kernel")
    assert kernel.status is Severity.WARNING
    assert kernel.value == "no driver"
    assert "modprobe intel_vpu" in " ".join(kernel.remediation)


def test_drivers_present_but_device_unlisted_points_at_device_access(tmp_path):
    settings = make_settings(tmp_path)
    inventory = drivers_inventory(settings, devices=[d for d in devices() if d.name != "GPU"])
    service, _, _, _ = build(settings, inventory, device_names=("CPU", "NPU"))

    report = service.run()

    listed = require(report, "drivers.gpu.openvino")
    assert listed.status is Severity.WARNING
    assert "/dev/dri/renderD" in " ".join(listed.remediation)


def test_an_inaccessible_npu_node_explains_why_openvino_does_not_list_it(tmp_path):
    settings = make_settings(tmp_path)
    node = npu_node(readable=False, writable=False, user_in_group=False)
    inventory = drivers_inventory(settings, npu=node, devices=devices(npu=False))
    service, _, _, _ = build(settings, inventory, device_names=("CPU", "GPU"))

    listed = require(service.run(), "drivers.npu.openvino")

    assert listed.status is Severity.WARNING
    assert "not accessible" in (listed.detail or "")


def test_two_gpus_are_reported_separately_by_pci_address(tmp_path):
    settings = make_settings(tmp_path)
    accelerators = [
        *lunar_lake_accelerators(),
        PciAccelerator(
            kind=DeviceKind.GPU,
            address="0000:03:00.0",
            vendor_id="0x8086",
            device_id="0xe20b",
            kernel_driver="xe",
        ),
    ]
    inventory = drivers_inventory(settings, accelerators=accelerators)
    # OpenVINO's own names for two GPUs; the Devices section folds them into GPU.
    service, _, _, _ = build(settings, inventory, device_names=("CPU", "GPU.0", "GPU.1", "NPU"))

    report = service.run()

    assert require(report, "drivers.gpu0.openvino").status is Severity.OK
    assert require(report, "drivers.gpu1.openvino").status is Severity.OK
    assert require(report, "drivers.gpu1.kernel").label == "GPU 0000:03:00.0 kernel"
    assert require_section(report, "drivers").status is Severity.OK


def test_the_second_of_two_gpus_warns_when_openvino_lists_only_one(tmp_path):
    settings = make_settings(tmp_path)
    accelerators = [
        *lunar_lake_accelerators(),
        PciAccelerator(
            kind=DeviceKind.GPU,
            address="0000:03:00.0",
            vendor_id="0x8086",
            device_id="0xe20b",
            kernel_driver="xe",
        ),
    ]
    inventory = drivers_inventory(settings, accelerators=accelerators)
    service, _, _, _ = build(settings, inventory, device_names=("CPU", "GPU", "NPU"))

    second = require(service.run(), "drivers.gpu1.openvino")

    assert second.status is Severity.WARNING
    assert second.label == "GPU 0000:03:00.0 in OpenVINO"


def test_a_device_claimed_by_another_driver_is_not_told_to_modprobe(tmp_path):
    settings = make_settings(tmp_path)
    inventory = drivers_inventory(
        settings,
        accelerators=lunar_lake_accelerators(gpu_driver="vfio-pci"),
        devices=[d for d in devices() if d.name != "GPU"],
    )
    service, _, _, _ = build(settings, inventory, device_names=("CPU", "NPU"))

    kernel = require(service.run(), "drivers.gpu.kernel")

    assert kernel.status is Severity.WARNING
    assert kernel.value == "vfio-pci"
    joined = " ".join(kernel.remediation)
    assert "claimed by the 'vfio-pci'" in joined
    assert "modprobe" not in joined


def test_an_older_gpu_generation_is_informational_not_a_driver_to_install(tmp_path):
    settings = make_settings(tmp_path)
    coffee_lake = PciAccelerator(
        kind=DeviceKind.GPU,
        address="0000:00:02.0",
        vendor_id="0x8086",
        device_id="0x3e9b",
        kernel_driver="i915",
    )
    inventory = drivers_inventory(
        settings,
        accelerators=[coffee_lake],
        driver_libraries=installed_driver_libraries(gpu_opencl=[]),
        devices=[d for d in devices() if d.name == "CPU"],
    )
    service, _, _, _ = build(settings, inventory, device_names=("CPU",))

    report = service.run()

    runtime = require(report, "drivers.gpu.runtime")
    assert runtime.status is Severity.INFO
    assert runtime.remediation == []
    assert report.check("drivers.gpu.openvino") is None
    assert require_section(report, "drivers").status in (Severity.OK, Severity.INFO)


class FakeDriverProbe(FakeHardwareProbe):
    """A hardware probe that can also list the PCI bus and the driver libraries."""

    def __init__(self, inventory: HardwareInventory) -> None:
        super().__init__(inventory)
        self.calls: list[str] = []

    def pci_accelerators(self, notes: list[str] | None = None) -> list[PciAccelerator] | None:
        self.calls.append("pci_accelerators")
        return lunar_lake_accelerators()

    def driver_libraries(self) -> DriverLibraries:
        self.calls.append("driver_libraries")
        return installed_driver_libraries()


def test_doctor_asks_a_driver_probe_for_the_bus_and_the_libraries(tmp_path):
    settings = make_settings(tmp_path)
    hardware = FakeDriverProbe(healthy_inventory(settings, host=inventory_host()))
    service = DoctorService(hardware, FakeRuntimeProbe(("CPU", "GPU", "NPU")), None, None, settings)

    report = service.run()

    assert hardware.calls == ["pci_accelerators", "driver_libraries"]
    assert require_section(report, "drivers").status is Severity.OK


def test_no_intel_accelerator_is_informational(tmp_path):
    settings = make_settings(tmp_path)
    service, _, _, _ = build(settings, drivers_inventory(settings, accelerators=[]))

    check = require(service.run(), "drivers.none")

    assert check.status is Severity.INFO


def test_an_unread_pci_bus_leaves_the_drivers_section_out(tmp_path):
    settings = make_settings(tmp_path)
    service, _, _, _ = build(settings, healthy_inventory(settings))

    assert service.run().section("drivers") is None


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
    assert report.check("registry.ignored") is None


def test_full_mode_warns_about_an_ignored_remote_bundle(tmp_path):
    """An installed bundle that stopped verifying must not hide behind all-green ticks."""
    settings = make_settings(tmp_path)
    registry = load_builtin_registry()
    registry.ignored = [
        "installed remote bundle version 7 is not in use (registry.remote.bad_signature: "
        "The bundle was not signed by key abc, or was changed after signing.)"
    ]
    service = DoctorService(
        FakeHardwareProbe(healthy_inventory(settings)),
        FakeRuntimeProbe(("CPU",)),
        FakeHubClient(),
        registry,
        settings,
    )

    report = service.run(full=True)

    check = require(report, "registry.ignored")
    assert check.status is Severity.WARNING
    assert "registry.remote.bad_signature" in (check.detail or "")
    assert any("yonga registry status" in line for line in check.remediation)
    section = report.section("registry")
    assert section is not None and section.status is Severity.WARNING


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


def roomy_inventory(settings: Settings) -> HardwareInventory:
    """A host whose cache filesystem has 500 GiB free, whatever the test machine has.

    The warning threshold depends on free space, so a test that expects the
    fixed 10 GiB threshold must not read the real disk under ``tmp_path``.
    """
    roomy = FilesystemInfo(
        path=Path("/"),
        mount_point=Path("/"),
        fstype="ext4",
        total_bytes=1000 * GIB,
        free_bytes=500 * GIB,
        available_bytes=500 * GIB,
    )
    return healthy_inventory(settings, filesystems=[roomy], managed_temp=roomy)


def test_much_reclaimable_compile_cache_warns_with_the_prune_command(tmp_path):
    settings = make_settings(tmp_path)
    cache = cache_inventory(settings, in_use=40 * GIB, orphaned=8 * GIB, stale=4 * GIB)

    report = doctor_with_cache(settings, cache, inventory=roomy_inventory(settings)).run()

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

    check = require(
        doctor_with_cache(settings, cache, inventory=roomy_inventory(settings)).run(),
        "storage.compile_cache",
    )

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


# --- the exporter import ---------------------------------------------------------

#: What importing the exporter prints with a CPU torch next to PyPI's torchvision.
TORCHVISION_FAILURE = (
    "Traceback (most recent call last):\n"
    "  ...\n"
    "RuntimeError: operator torchvision::nms does not exist"
)


def import_doctor(
    tmp_path: Path, recorded: bool | str | None, checked: bool | str | None = None
) -> tuple[DoctorService, list[bool]]:
    """A doctor whose exporter recorded ``recorded`` and finds ``checked`` when asked to import.

    Also returns each ``run`` it was asked with: ``False`` reads the record,
    ``True`` imports.
    """
    settings = make_settings(tmp_path)
    asked: list[bool] = []
    resolver = RegistryResolver(load_builtin_registry())

    def exporter_import(run: bool) -> bool | str | None:
        asked.append(run)
        return checked if run else recorded

    service = DoctorService(
        FakeHardwareProbe(healthy_inventory(settings)),
        FakeRuntimeProbe(("CPU", "GPU", "NPU")),
        FakeHubClient(None),
        None,
        settings,
        exporter_import=exporter_import,
        classify_failure=lambda failure: resolver.classify(raw_error=failure, stage="export"),
    )
    return service, asked


def test_a_recorded_working_import_is_one_ok_line_after_the_packages(tmp_path):
    service, asked = import_doctor(tmp_path, True)

    report = service.run(full=True)

    check = require(report, "openvino.exporter_import")
    assert (check.status, check.value) == (Severity.OK, "imports")
    labels = [item.id for item in require_section(report, "openvino").checks]
    assert labels.index("openvino.exporter_import") == labels.index("openvino.transformers") + 1
    # A recorded success is trusted, even by --full: the seconds-long import is not repeated.
    assert asked == [False]


def test_the_first_doctor_imports_once_when_nothing_is_recorded(tmp_path):
    service, asked = import_doctor(tmp_path, False, True)

    check = require(service.run(), "openvino.exporter_import")

    assert check.status is Severity.OK
    assert asked == [False, True]


def test_a_broken_exporter_import_is_reported_with_the_registry_remedy(tmp_path):
    """CPU torch + PyPI torchvision: versions look fine, the import does not."""
    service, asked = import_doctor(tmp_path, TORCHVISION_FAILURE)

    report = service.run()

    check = require(report, "openvino.exporter_import")
    assert check.status is Severity.WARNING
    assert check.value == "fails"
    assert "torchvision cannot load against the installed torch" in (check.detail or "")
    assert "operator torchvision::nms does not exist" in (check.detail or "")
    remedy = " ".join(check.remediation)
    assert "--index-url https://download.pytorch.org/whl/cpu" in remedy
    assert "`yonga doctor --full` checks again" in remedy
    assert report.exit_code() is ExitCode.WARNINGS
    assert require(report, "openvino.optimum_intel").status is Severity.OK
    # Plain doctor reports the recorded failure without importing again.
    assert asked == [False]


def test_full_checks_a_recorded_failure_again(tmp_path):
    service, asked = import_doctor(tmp_path, TORCHVISION_FAILURE, True)

    check = require(service.run(full=True), "openvino.exporter_import")

    assert check.status is Severity.OK
    assert asked == [False, True]


def test_a_freshly_checked_failure_does_not_call_itself_recorded(tmp_path):
    service, _ = import_doctor(tmp_path, False, "ImportError: something else entirely")

    check = require(service.run(), "openvino.exporter_import")

    assert check.status is Severity.WARNING
    assert "does not import: ImportError: something else entirely" in (check.detail or "")
    assert not any("checks again" in step for step in check.remediation)
    # The way to check again is yonga's private import, never a plain interpreter's.
    assert not any("python -c" in step for step in check.remediation)
    assert any("yonga doctor --full" in step for step in check.remediation)


def test_no_exporter_to_check_adds_no_line(tmp_path):
    service, _ = import_doctor(tmp_path, None)
    assert service.run().check("openvino.exporter_import") is None
