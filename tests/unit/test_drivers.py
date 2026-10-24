"""Which driver packages ``yonga doctor`` names for each distribution."""

from __future__ import annotations

import pytest

from yonga.application.drivers import (
    SYSTEM_CHANGES_NOTE,
    DistroFamily,
    DriverComponent,
    distro_family,
    install_steps,
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
