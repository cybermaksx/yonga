"""Unit tests for the read-only Linux host probe."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from yonga.adapters.system_linux import (
    DeviceSnapshot,
    LinuxSystemProbe,
    derive_device_tags,
    parse_os_release,
)
from yonga.config.settings import load_settings
from yonga.domain.hardware import DeviceKind, FilesystemInfo, HostInfo, OpenVINODevice

OS_RELEASE = """\
PRETTY_NAME="Ubuntu 26.04.1 LTS"
NAME="Ubuntu"
VERSION_ID="26.04"
VERSION="26.04.1 LTS (Resolute Raccoon)"
ID=ubuntu
ID_LIKE=debian
# a comment
HOME_URL='https://www.ubuntu.com/'
malformed line without equals
"""

CPUINFO = """\
processor\t: 0
vendor_id\t: GenuineIntel
cpu family\t: 6
model name\t: Intel(R) Core(TM) Ultra 7 268V
flags\t\t: fpu vme de

processor\t: 1
vendor_id\t: GenuineIntel
model name\t: Intel(R) Core(TM) Ultra 7 268V
"""


def make_settings(tmp_path: Path):
    """Deterministic settings pointed at a throwaway tree."""
    return load_settings(
        config_file=tmp_path / "absent-config.toml",
        env={},
        overrides={"cache_dir": tmp_path / "cache", "data_dir": tmp_path / "data"},
    )


def build_root(tmp_path: Path, *, with_accel: bool = True, with_proc: bool = True) -> Path:
    """Create a miniature /etc, /proc and /dev tree the probe can read."""
    root = tmp_path / "root"
    (root / "etc").mkdir(parents=True)
    (root / "etc" / "os-release").write_text(OS_RELEASE, encoding="utf-8")
    if with_proc:
        (root / "proc").mkdir(parents=True)
        (root / "proc" / "cpuinfo").write_text(CPUINFO, encoding="utf-8")
    if with_accel:
        accel = root / "dev" / "accel"
        accel.mkdir(parents=True)
        (accel / "accel0").write_bytes(b"")
    return root


def real_devices() -> tuple[OpenVINODevice, ...]:
    """The devices this host actually reports, as probed for HOST_FACTS.md."""
    return (
        OpenVINODevice(
            name="CPU",
            full_name="Intel(R) Core(TM) Ultra 7 268V",
            architecture="intel64",
            properties={"FULL_DEVICE_NAME": "Intel(R) Core(TM) Ultra 7 268V"},
        ),
        OpenVINODevice(
            name="GPU",
            full_name="Intel(R) Arc(TM) Graphics (iGPU)",
            architecture="GPU: vendor=0x8086 arch=v20.4.4",
            properties={"GPU_UARCH_VERSION": "20.4.4", "GPU_DEVICE_ID": "0x64a0"},
        ),
        OpenVINODevice(
            name="NPU",
            full_name="Intel(R) AI Boost",
            architecture="4000",
            properties={
                "NPU_DRIVER_VERSION": "1789049548",
                "NPU_COMPILER_TYPE": "CompilerType.PREFER_PLUGIN",
            },
        ),
    )


class StubFilesystem:
    """A filesystem probe that always answers with the same figures."""

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def usage(self, path: Path) -> FilesystemInfo:
        self.calls.append(Path(path))
        return FilesystemInfo(
            path=Path(path),
            mount_point=Path("/"),
            fstype="ext4",
            total_bytes=1000,
            free_bytes=900,
            available_bytes=800,
        )


# --- os-release ---------------------------------------------------------------------


def test_parse_os_release_handles_quotes_and_junk():
    data = parse_os_release(OS_RELEASE)
    assert data["ID"] == "ubuntu"
    assert data["PRETTY_NAME"] == "Ubuntu 26.04.1 LTS"
    assert data["VERSION_ID"] == "26.04"
    assert data["HOME_URL"] == "https://www.ubuntu.com/"
    assert "malformed line without equals" not in data


def test_parse_os_release_of_empty_text_is_empty():
    assert parse_os_release("") == {}


# --- host facts ---------------------------------------------------------------------


def test_host_info_reads_distribution_and_cpu(tmp_path):
    probe = LinuxSystemProbe(root=build_root(tmp_path), filesystem=StubFilesystem())
    host = probe.host_info()
    assert host.distro_id == "ubuntu"
    assert host.distro_version == "26.04"
    assert host.distro_pretty == "Ubuntu 26.04.1 LTS"
    assert host.distro_like == ["debian"]
    assert host.cpu_model == "Intel(R) Core(TM) Ultra 7 268V"
    assert host.python_version == platform.python_version()
    assert host.kernel == platform.release()
    assert host.cpu_count == os.cpu_count()


def test_host_info_degrades_when_proc_and_etc_are_missing(tmp_path):
    empty = tmp_path / "empty-root"
    empty.mkdir()
    notes: list[str] = []
    probe = LinuxSystemProbe(root=empty, filesystem=StubFilesystem())
    host = probe.host_info(notes)
    assert host.distro_id is None
    assert host.cpu_model is None
    assert any("os-release" in note for note in notes)
    assert any("cpuinfo" in note for note in notes)


def test_cpu_identity_reports_the_vendor(tmp_path):
    probe = LinuxSystemProbe(root=build_root(tmp_path), filesystem=StubFilesystem())
    assert probe.cpu_identity() == ("Intel(R) Core(TM) Ultra 7 268V", "GenuineIntel")


# --- memory -------------------------------------------------------------------------


def test_memory_info_uses_psutil(tmp_path):
    probe = LinuxSystemProbe(root=build_root(tmp_path), filesystem=StubFilesystem())
    memory = probe.memory_info()
    assert memory.total_bytes > 0
    assert memory.available_bytes > 0


def test_memory_info_degrades_without_psutil(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    notes: list[str] = []
    probe = LinuxSystemProbe(root=build_root(tmp_path), filesystem=StubFilesystem())
    memory = probe.memory_info(notes)
    assert memory.total_bytes == 0
    assert memory.swap_total_bytes == 0
    assert any("psutil" in note for note in notes)


# --- packages -----------------------------------------------------------------------


def test_packages_covers_the_whole_stack(tmp_path):
    probe = LinuxSystemProbe(root=build_root(tmp_path), filesystem=StubFilesystem())
    packages = {entry.name: entry for entry in probe.packages()}
    assert set(packages) == {
        "openvino",
        "openvino-genai",
        "openvino-tokenizers",
        "optimum-intel",
        "nncf",
        "transformers",
        "huggingface-hub",
    }
    # huggingface_hub is a hard dependency, so it must always be present.
    assert packages["huggingface-hub"].present is True
    assert packages["huggingface-hub"].importable is True


def test_packages_never_import_the_heavy_modules(tmp_path):
    for name in ("openvino", "openvino_genai", "torch", "transformers", "nncf"):
        if name in sys.modules:
            pytest.skip(f"{name} was already imported by another test")
    probe = LinuxSystemProbe(root=build_root(tmp_path), filesystem=StubFilesystem())
    probe.packages()
    for name in ("openvino", "openvino_genai", "torch", "transformers", "nncf"):
        assert name not in sys.modules


# --- NPU node -----------------------------------------------------------------------


def test_npu_node_reports_the_accel_device(tmp_path):
    root = build_root(tmp_path)
    libdir = tmp_path / "libs"
    libdir.mkdir()
    (libdir / "libopenvino_intel_npu_compiler.so").write_bytes(b"")
    (libdir / "libopenvino_intel_npu_compiler_loader.so").write_bytes(b"")
    (libdir / "libopenvino_intel_npu_plugin.so").write_bytes(b"")
    (libdir / "libunrelated.so").write_bytes(b"")

    probe = LinuxSystemProbe(root=root, filesystem=StubFilesystem(), library_dirs=[libdir])
    npu = probe.npu_node(driver_version="1789049548")

    assert npu.directory_exists is True
    assert npu.node_exists is True
    assert npu.node_path == root / "dev" / "accel" / "accel0"
    assert npu.readable is True
    assert npu.writable is True
    assert npu.driver_version == "1789049548"
    assert npu.group_name is not None
    assert npu.user_groups
    assert sorted(Path(entry).name for entry in npu.compiler_libraries) == [
        "libopenvino_intel_npu_compiler.so",
        "libopenvino_intel_npu_compiler_loader.so",
    ]


def pci_device(
    root: Path, address: str, *, vendor: str, pci_class: str, device: str, driver: str | None
) -> None:
    """One ``/sys/bus/pci/devices`` entry, with its driver link when bound."""
    entry = root / "sys" / "bus" / "pci" / "devices" / address
    entry.mkdir(parents=True)
    (entry / "vendor").write_text(f"{vendor}\n")
    (entry / "class").write_text(f"{pci_class}\n")
    (entry / "device").write_text(f"{device}\n")
    if driver is not None:
        target = root / "sys" / "bus" / "pci" / "drivers" / driver
        target.mkdir(parents=True, exist_ok=True)
        (entry / "driver").symlink_to(target)


def test_pci_accelerators_lists_intel_gpus_and_npus_with_their_kernel_driver(tmp_path):
    root = build_root(tmp_path)
    # The Arrow Lake reference laptop: iGPU on i915, NPU on intel_vpu.
    pci_device(
        root, "0000:00:02.0", vendor="0x8086", pci_class="0x030000", device="0x7d41", driver="i915"
    )
    pci_device(
        root,
        "0000:00:0b.0",
        vendor="0x8086",
        pci_class="0x120000",
        device="0x7d1d",
        driver="intel_vpu",
    )
    # A discrete GPU without a driver, an NVIDIA GPU and an Intel USB controller.
    pci_device(
        root, "0000:03:00.0", vendor="0x8086", pci_class="0x030000", device="0xe20b", driver=None
    )
    pci_device(
        root,
        "0000:01:00.0",
        vendor="0x10de",
        pci_class="0x030000",
        device="0x2684",
        driver="nvidia",
    )
    pci_device(
        root,
        "0000:00:14.0",
        vendor="0x8086",
        pci_class="0x0c0330",
        device="0x7e7d",
        driver="xhci_hcd",
    )
    probe = LinuxSystemProbe(root=root, filesystem=StubFilesystem())

    found = probe.pci_accelerators()

    assert found is not None
    assert [(a.address, a.kind, a.device_id, a.kernel_driver) for a in found] == [
        ("0000:00:02.0", DeviceKind.GPU, "0x7d41", "i915"),
        ("0000:00:0b.0", DeviceKind.NPU, "0x7d1d", "intel_vpu"),
        ("0000:03:00.0", DeviceKind.GPU, "0xe20b", None),
    ]


def test_an_unreadable_pci_bus_is_none_and_a_note(tmp_path):
    probe = LinuxSystemProbe(root=build_root(tmp_path), filesystem=StubFilesystem())
    notes: list[str] = []

    assert probe.pci_accelerators(notes) is None
    assert any("PCI" in note for note in notes)


def test_driver_libraries_finds_intels_opencl_icd_and_the_level_zero_libraries(tmp_path):
    root = build_root(tmp_path)
    vendors = root / "etc" / "OpenCL" / "vendors"
    vendors.mkdir(parents=True)
    (vendors / "intel.icd").write_text("/usr/lib/intel-opencl/libigdrcl.so\n")
    (vendors / "rusticl.icd").write_text("libRusticlOpenCL.so.1\n")
    libdir = tmp_path / "libs"
    libdir.mkdir()
    for name in ("libze_loader.so.1", "libze_intel_npu.so.1", "libze_intel_gpu.so.1"):
        (libdir / name).write_bytes(b"")
    probe = LinuxSystemProbe(root=root, filesystem=StubFilesystem(), library_dirs=[libdir])

    libraries = probe.driver_libraries()

    assert libraries.gpu_opencl == ["/usr/lib/intel-opencl/libigdrcl.so"]
    assert [Path(p).name for p in libraries.level_zero_loader] == ["libze_loader.so.1"]
    assert [Path(p).name for p in libraries.npu_level_zero] == ["libze_intel_npu.so.1"]


def test_library_search_includes_fedoras_lib64(tmp_path):
    root = build_root(tmp_path)
    probe = LinuxSystemProbe(root=root, filesystem=StubFilesystem())
    assert root / "usr" / "lib64" in probe.library_search_dirs()


def test_driver_libraries_are_found_in_lib64(tmp_path):
    root = build_root(tmp_path)
    lib64 = root / "usr" / "lib64"
    lib64.mkdir(parents=True)
    (lib64 / "libze_loader.so.1").write_bytes(b"")
    (lib64 / "libze_intel_npu.so.1").write_bytes(b"")
    probe = LinuxSystemProbe(root=root, filesystem=StubFilesystem())

    libraries = probe.driver_libraries()

    assert str(lib64 / "libze_loader.so.1") in libraries.level_zero_loader
    assert str(lib64 / "libze_intel_npu.so.1") in libraries.npu_level_zero


def test_a_linked_library_directory_is_not_searched_twice(tmp_path):
    """Arch's /usr/lib64 is a link to /usr/lib."""
    root = build_root(tmp_path)
    lib = root / "usr" / "lib"
    lib.mkdir(parents=True)
    (root / "usr" / "lib64").symlink_to(lib)
    (lib / "libze_loader.so.1").write_bytes(b"")
    probe = LinuxSystemProbe(root=root, filesystem=StubFilesystem())

    assert len(probe.driver_libraries().level_zero_loader) == 1


def test_driver_libraries_are_empty_on_a_host_without_them(tmp_path):
    probe = LinuxSystemProbe(
        root=build_root(tmp_path), filesystem=StubFilesystem(), library_dirs=[]
    )
    libraries = probe.driver_libraries()
    assert libraries.gpu_opencl == []
    assert libraries.level_zero_loader == []
    assert libraries.npu_level_zero == []


def test_npu_node_without_dev_accel_is_a_note_not_a_crash(tmp_path):
    root = build_root(tmp_path, with_accel=False)
    probe = LinuxSystemProbe(root=root, filesystem=StubFilesystem(), library_dirs=[])
    npu = probe.npu_node()
    assert npu.directory_exists is False
    assert npu.node_exists is False
    assert npu.accessible is False
    assert npu.compiler_libraries == []
    assert any("does not exist" in note for note in npu.notes)


def test_npu_node_with_an_empty_accel_directory(tmp_path):
    root = build_root(tmp_path, with_accel=False)
    (root / "dev" / "accel").mkdir(parents=True)
    probe = LinuxSystemProbe(root=root, filesystem=StubFilesystem(), library_dirs=[])
    npu = probe.npu_node()
    assert npu.directory_exists is True
    assert npu.node_exists is False
    assert any("no accel* node" in note for note in npu.notes)


def test_npu_node_never_changes_permissions(tmp_path):
    root = build_root(tmp_path)
    node = root / "dev" / "accel" / "accel0"
    before = node.stat().st_mode
    probe = LinuxSystemProbe(root=root, filesystem=StubFilesystem(), library_dirs=[])
    probe.npu_node()
    assert node.stat().st_mode == before


@pytest.mark.skipif(not Path("/dev/accel/accel0").exists(), reason="no NPU node on this host")
def test_real_npu_node_matches_the_probed_host_facts():
    npu = LinuxSystemProbe().npu_node()
    assert npu.node_path == Path("/dev/accel/accel0")
    assert npu.group_name == "render"
    assert npu.user_in_group is True
    assert "render" in npu.user_groups
    assert any("libopenvino_intel_npu_compiler" in entry for entry in npu.compiler_libraries)


# --- device tags --------------------------------------------------------------------


def test_device_tags_for_this_lunar_lake_host():
    host = HostInfo(cpu_model="Intel(R) Core(TM) Ultra 7 268V")
    tags = derive_device_tags(devices=real_devices(), host=host, cpu_vendor="GenuineIntel")
    assert tags == [
        "intel.cpu",
        "intel.core_ultra.series2",
        "intel.gpu",
        "intel.gpu.xe2",
        "intel.npu",
        "intel.npu.lunar_lake",
    ]


def test_device_tags_omit_a_generation_they_cannot_justify():
    devices = (
        OpenVINODevice(
            name="GPU", full_name="Intel(R) Graphics", architecture="GPU: vendor=0x8086"
        ),
        OpenVINODevice(name="NPU", full_name="Intel(R) AI Boost", architecture="9999"),
    )
    tags = derive_device_tags(devices=devices, host=HostInfo(), cpu_vendor="GenuineIntel")
    assert "intel.gpu" in tags
    assert "intel.npu" in tags
    assert not any(tag.startswith("intel.gpu.") for tag in tags)
    assert not any(tag.startswith("intel.npu.") for tag in tags)


def test_device_tags_for_a_meteor_lake_npu():
    devices = (OpenVINODevice(name="NPU", full_name="Intel(R) AI Boost", architecture="3720"),)
    assert derive_device_tags(devices=devices) == ["intel.npu", "intel.npu.meteor_lake"]


def test_device_tags_skip_a_non_intel_cpu():
    host = HostInfo(cpu_model="AMD Ryzen 9 7950X")
    tags = derive_device_tags(devices=(), host=host, cpu_vendor="AuthenticAMD")
    assert tags == []


def test_device_tags_with_no_devices_at_all():
    assert derive_device_tags() == []


# --- inventory ----------------------------------------------------------------------


def test_inventory_with_injected_devices(tmp_path):
    settings = make_settings(tmp_path)
    filesystem = StubFilesystem()
    probe = LinuxSystemProbe(
        settings=settings,
        root=build_root(tmp_path),
        filesystem=filesystem,
        library_dirs=[],
    )

    inventory = probe.inventory(devices=real_devices(), openvino_available=True)

    assert inventory.openvino_available is True
    assert inventory.available_device_names == ["CPU", "GPU", "NPU"]
    assert inventory.host.distro_id == "ubuntu"
    assert "intel.npu.lunar_lake" in inventory.device_tags
    assert inventory.npu is not None
    assert inventory.npu.driver_version == "1789049548"
    assert inventory.system_temp is not None
    assert inventory.managed_temp is not None
    assert inventory.managed_temp.path == settings.paths.tmp_dir
    assert len(inventory.packages) == 7
    # system temp, managed tmp, artifacts, root
    assert len(inventory.filesystems) == 4
    # The PCI bus and the driver libraries are doctor's to ask for, not every prepare's.
    assert inventory.accelerators is None
    assert inventory.driver_libraries is None


def test_inventory_full_adds_more_filesystems(tmp_path):
    settings = make_settings(tmp_path)
    probe = LinuxSystemProbe(
        settings=settings,
        root=build_root(tmp_path),
        filesystem=StubFilesystem(),
        library_dirs=[],
    )
    assert len(probe.inventory(full=True).filesystems) == 6


def test_inventory_uses_the_injected_device_source(tmp_path):
    calls: list[int] = []

    def source() -> DeviceSnapshot:
        calls.append(1)
        return DeviceSnapshot(devices=real_devices(), available=True)

    probe = LinuxSystemProbe(
        settings=make_settings(tmp_path),
        root=build_root(tmp_path),
        filesystem=StubFilesystem(),
        device_source=source,
        library_dirs=[],
    )
    inventory = probe.inventory()
    assert calls == [1]
    assert inventory.openvino_available is True
    assert [device.name for device in inventory.devices] == ["CPU", "GPU", "NPU"]


def test_inventory_survives_a_failing_device_source(tmp_path):
    def source() -> DeviceSnapshot:
        raise RuntimeError("libopenvino.so: cannot open shared object file")

    probe = LinuxSystemProbe(
        settings=make_settings(tmp_path),
        root=build_root(tmp_path),
        filesystem=StubFilesystem(),
        device_source=source,
        library_dirs=[],
    )
    inventory, notes = probe.inventory_with_notes()
    assert inventory.openvino_available is False
    assert inventory.devices == []
    assert inventory.openvino_import_error is not None
    assert "cannot open shared object file" in inventory.openvino_import_error
    assert any("device probe failed" in note for note in notes)


def test_inventory_without_a_device_source_explains_itself(tmp_path):
    probe = LinuxSystemProbe(
        settings=make_settings(tmp_path),
        root=build_root(tmp_path),
        filesystem=StubFilesystem(),
        library_dirs=[],
    )
    inventory, notes = probe.inventory_with_notes()
    assert inventory.openvino_available is False
    assert inventory.device_tags == ["intel.cpu", "intel.core_ultra.series2"]
    assert any("No OpenVINO device source" in note for note in notes)


def test_inventory_on_an_unusable_host_does_not_crash(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()

    class BrokenFilesystem:
        def usage(self, path: Path) -> FilesystemInfo:
            raise OSError("statvfs failed")

    probe = LinuxSystemProbe(
        root=empty,
        filesystem=BrokenFilesystem(),
        library_dirs=[tmp_path / "absent-libdir"],
    )
    inventory, notes = probe.inventory_with_notes()
    assert inventory.host.cpu_model is None
    assert inventory.npu is not None
    assert inventory.npu.node_exists is False
    assert inventory.device_tags == []
    assert all(entry.total_bytes == 0 for entry in inventory.filesystems)
    assert notes


def test_inventory_notes_a_non_linux_platform(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    probe = LinuxSystemProbe(
        settings=make_settings(tmp_path),
        root=build_root(tmp_path),
        filesystem=StubFilesystem(),
        library_dirs=[],
    )
    inventory, notes = probe.inventory_with_notes()
    assert inventory.host.platform == "Darwin"
    assert any("not a supported platform" in note for note in notes)


def test_inventory_explicit_arguments_beat_the_device_source(tmp_path):
    def source() -> DeviceSnapshot:
        return DeviceSnapshot(devices=real_devices(), available=True)

    probe = LinuxSystemProbe(
        settings=make_settings(tmp_path),
        root=build_root(tmp_path),
        filesystem=StubFilesystem(),
        device_source=source,
        library_dirs=[],
    )
    inventory = probe.inventory(devices=[], openvino_available=False, openvino_import_error="boom")
    assert inventory.devices == []
    assert inventory.openvino_available is False
    assert inventory.openvino_import_error == "boom"


def test_importing_the_probe_does_not_import_openvino():
    """A guard against anyone re-adding a module-scope OpenVINO import."""
    code = (
        "import sys; import yonga.adapters.system_linux as m; "
        "m.LinuxSystemProbe().packages(); "
        "print(','.join(n for n in "
        "('openvino','openvino_genai','optimum','nncf','transformers','torch') "
        "if n in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == ""
    assert DeviceKind.NPU.value == "NPU"
