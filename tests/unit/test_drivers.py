"""Which driver packages ``yonga doctor`` names for each distribution."""

from __future__ import annotations

import pytest

from yonga.application.drivers import (
    SYSTEM_CHANGES_NOTE,
    DistroFamily,
    DriverComponent,
    claimed_device_steps,
    distro_family,
    install_steps,
    is_legacy_gpu,
)
from yonga.domain.hardware import HostInfo


def host(distro_id: str | None, *like: str) -> HostInfo:
    return HostInfo(distro_id=distro_id, distro_like=list(like))


@pytest.mark.parametrize(
    ("distro", "expected"),
    [
        (host("arch"), DistroFamily.ARCH),
        (host("cachyos", "arch"), DistroFamily.ARCH),
        (host("ubuntu", "debian"), DistroFamily.DEBIAN),
        (host("linuxmint", "ubuntu", "debian"), DistroFamily.DEBIAN),
        (host("fedora"), DistroFamily.FEDORA),
        # An unknown derivative is placed by its ID_LIKE.
        (host("somethingnew", "arch"), DistroFamily.ARCH),
        (host("opensuse-tumbleweed", "opensuse", "suse"), None),
        (host(None), None),
    ],
)
def test_the_distribution_family_comes_from_id_then_id_like(distro, expected):
    assert distro_family(distro) is expected


def test_each_family_gets_its_own_package_command():
    gpu = DriverComponent.GPU_RUNTIME
    assert "sudo pacman -S intel-compute-runtime" in install_steps(gpu, host("cachyos", "arch"))[0]
    assert "sudo apt install intel-opencl-icd" in install_steps(gpu, host("ubuntu"))[0]
    assert "sudo dnf install intel-compute-runtime" in install_steps(gpu, host("fedora"))[0]
    npu = DriverComponent.NPU_RUNTIME
    assert "sudo pacman -S intel-npu-driver" in install_steps(npu, host("arch"))[0]


@pytest.mark.parametrize("component", list(DriverComponent))
@pytest.mark.parametrize("distro", [host("arch"), host("ubuntu"), host("fedora"), host(None)])
def test_every_component_has_steps_everywhere_and_ends_with_the_note(component, distro):
    steps = install_steps(component, distro)
    assert len(steps) >= 2
    assert steps[-1] == SYSTEM_CHANGES_NOTE


def test_a_family_without_a_package_falls_back_to_the_upstream_release():
    steps = install_steps(DriverComponent.NPU_RUNTIME, host("fedora"))
    assert "github.com/intel/linux-npu-driver/releases" in steps[0]


def test_arch_is_told_its_driver_package_leaves_the_npu_compiler_out():
    steps = install_steps(DriverComponent.NPU_COMPILER, host("cachyos", "arch"))
    assert "libopenvino_intel_npu_compiler_loader.so" in steps[0]


def test_a_claimed_device_is_told_to_release_it_not_to_load_a_module():
    steps = claimed_device_steps("GPU", "0000:03:00.0", "vfio-pci")
    assert "0000:03:00.0" in steps[0]
    assert "'vfio-pci'" in steps[0]
    assert not any("modprobe" in step for step in steps)
    assert steps[-1] == SYSTEM_CHANGES_NOTE


@pytest.mark.parametrize(
    ("device_id", "legacy"),
    [
        ("0x1912", True),  # Skylake HD 530
        ("0x3e9b", True),  # Coffee Lake UHD 630
        ("0x8a52", True),  # Ice Lake Iris Plus
        ("0x0416", True),  # Haswell HD 4600
        ("0x9a49", False),  # Tiger Lake Iris Xe
        ("0x46a6", False),  # Alder Lake-P
        ("0x56a0", False),  # Arc A770
        ("0x7d41", False),  # Arrow Lake-U (the CachyOS laptop)
        ("0x64a0", False),  # Lunar Lake
        ("0xE20B", False),  # Battlemage, upper case
        ("0x4e55", True),  # Jasper Lake
        ("0x4571", True),  # Elkhart Lake
        ("0x3E9B", True),  # Coffee Lake, upper case
        # Ids that share a prefix with a legacy family but are not in the kernel's list.
        ("0x4e00", False),
        ("0x45ff", False),
        ("0x3eff", False),
        ("0x8a00", False),
        ("", False),
        ("not-an-id", False),
    ],
)
def test_gpu_generations_before_gen12_are_recognised(device_id, legacy):
    assert is_legacy_gpu(device_id) is legacy


def test_the_shared_note_says_yonga_runs_nothing():
    assert "never runs sudo" in SYSTEM_CHANGES_NOTE
    assert "run any command yourself" in SYSTEM_CHANGES_NOTE
