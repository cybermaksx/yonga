"""Which system drivers an Intel GPU or NPU needs, and how to install them here.

OpenVINO's GPU plugin drives the GPU through Intel's OpenCL driver
(compute-runtime); its NPU plugin goes through Level Zero: the loader, the NPU
driver library and, for compiling a model, the NPU compiler. None of those ship
with the ``openvino`` wheel, and none of them are Python packages, so ``yonga
doctor`` names the distribution package that provides each one.

The table covers the distribution families whose packages were checked: Arch
and its derivatives (CachyOS, EndeavourOS, Manjaro), Debian and Ubuntu, and
Fedora. Any other distribution, or a component a family does not package, gets
the upstream release page instead. yonga never runs any of these commands.
"""

from __future__ import annotations

from enum import Enum

from yonga.domain.hardware import HostInfo

__all__ = [
    "SYSTEM_CHANGES_NOTE",
    "DistroFamily",
    "DriverComponent",
    "claimed_device_steps",
    "distro_family",
    "install_steps",
    "is_legacy_gpu",
]

#: The one sentence, next to every system command doctor suggests, that yonga
#: runs none of them.
SYSTEM_CHANGES_NOTE = (
    "yonga never runs sudo and never changes packages, kernel modules, permissions or group "
    "membership: run any command yourself."
)

#: PCI device-id prefixes of the Intel GPU generations up to Gen11 (Ice Lake,
#: Elkhart/Jasper Lake) and their predecessors back to Sandy Bridge. Intel's
#: current compute-runtime covers Gen12 (Tiger Lake) and newer; these older
#: iGPUs are served by its legacy branch at most, so installing the current
#: runtime cannot make OpenVINO list them.
_LEGACY_GPU_ID_PREFIXES: tuple[str, ...] = (
    "0x00",  # Ironlake
    "0x01",  # Sandy Bridge, Ivy Bridge
    "0x04",  # Haswell
    "0x0a",  # Haswell ULT
    "0x0c",  # Haswell
    "0x0d",  # Haswell (Iris Pro)
    "0x0f",  # Bay Trail
    "0x16",  # Broadwell
    "0x19",  # Skylake
    "0x1a",  # Apollo Lake
    "0x22",  # Cherry Trail
    "0x31",  # Gemini Lake
    "0x3e",  # Coffee Lake
    "0x45",  # Elkhart Lake
    "0x4e",  # Jasper Lake
    "0x59",  # Kaby Lake
    "0x5a",  # Apollo Lake, Cannon Lake
    "0x87",  # Amber Lake
    "0x8a",  # Ice Lake
    "0x9b",  # Comet Lake
)


class DistroFamily(str, Enum):
    """The package ecosystems the install table knows."""

    ARCH = "arch"
    DEBIAN = "debian"
    FEDORA = "fedora"


class DriverComponent(str, Enum):
    """One piece of the system driver stack OpenVINO's plugins depend on."""

    GPU_KERNEL = "gpu_kernel"
    GPU_RUNTIME = "gpu_runtime"
    NPU_KERNEL = "npu_kernel"
    NPU_RUNTIME = "npu_runtime"
    NPU_COMPILER = "npu_compiler"


_FAMILY_IDS: dict[str, DistroFamily] = {
    "arch": DistroFamily.ARCH,
    "cachyos": DistroFamily.ARCH,
    "endeavouros": DistroFamily.ARCH,
    "manjaro": DistroFamily.ARCH,
    "debian": DistroFamily.DEBIAN,
    "ubuntu": DistroFamily.DEBIAN,
    "linuxmint": DistroFamily.DEBIAN,
    "pop": DistroFamily.DEBIAN,
    "fedora": DistroFamily.FEDORA,
}

_COMPUTE_RUNTIME_RELEASES = "https://github.com/intel/compute-runtime/releases"
_NPU_DRIVER_RELEASES = "https://github.com/intel/linux-npu-driver/releases"
_GPU_RUNTIME = "Install Intel's GPU compute runtime (its OpenCL driver)"

#: Install commands per component and family. A family missing from a
#: component's entry gets the generic steps below.
_STEPS: dict[DriverComponent, dict[DistroFamily, list[str]]] = {
    DriverComponent.GPU_RUNTIME: {
        DistroFamily.ARCH: [f"{_GPU_RUNTIME}: sudo pacman -S intel-compute-runtime"],
        DistroFamily.DEBIAN: [f"{_GPU_RUNTIME}: sudo apt install intel-opencl-icd"],
        DistroFamily.FEDORA: [f"{_GPU_RUNTIME}: sudo dnf install intel-compute-runtime"],
    },
    DriverComponent.NPU_RUNTIME: {
        DistroFamily.ARCH: [
            "Install the NPU driver and the Level Zero loader: sudo pacman -S intel-npu-driver"
        ],
        DistroFamily.DEBIAN: [
            "Install intel-level-zero-npu and libze1 from the .deb packages of the latest "
            f"release at {_NPU_DRIVER_RELEASES} (sudo apt install ./*.deb in their directory).",
        ],
    },
    DriverComponent.NPU_COMPILER: {
        DistroFamily.ARCH: [
            "Arch's intel-npu-driver package leaves the compiler out: take "
            "libopenvino_intel_npu_compiler.so and libopenvino_intel_npu_compiler_loader.so from "
            "the intel-driver-compiler-npu .deb of the release matching your driver version at "
            f"{_NPU_DRIVER_RELEASES} and copy both into /usr/lib.",
        ],
        DistroFamily.DEBIAN: [
            "Install intel-driver-compiler-npu from the release matching your NPU driver "
            f"version at {_NPU_DRIVER_RELEASES}.",
        ],
    },
}

_GENERIC_STEPS: dict[DriverComponent, list[str]] = {
    DriverComponent.GPU_KERNEL: [
        "No kernel driver is bound to the GPU: load it with sudo modprobe i915 (or xe on "
        "newer GPUs). A GPU newer than the running kernel needs a newer kernel.",
    ],
    DriverComponent.GPU_RUNTIME: [
        "Install Intel's GPU compute runtime (the OpenCL driver, libigdrcl.so) from your "
        f"distribution or from {_COMPUTE_RUNTIME_RELEASES}.",
    ],
    DriverComponent.NPU_KERNEL: [
        "No kernel driver is bound to the NPU: load it with sudo modprobe intel_vpu. It needs "
        "a recent kernel and the NPU firmware from linux-firmware.",
    ],
    DriverComponent.NPU_RUNTIME: [
        "Install the Level Zero loader and Intel's NPU driver (libze_intel_npu.so) from your "
        f"distribution or from {_NPU_DRIVER_RELEASES}.",
    ],
    DriverComponent.NPU_COMPILER: [
        "Install the NPU compiler matching your NPU driver version "
        f"(intel-driver-compiler-npu at {_NPU_DRIVER_RELEASES}).",
    ],
}


def distro_family(host: HostInfo) -> DistroFamily | None:
    """The package family of ``host``'s distribution, from ``ID`` then ``ID_LIKE``."""
    for name in [host.distro_id or "", *host.distro_like]:
        family = _FAMILY_IDS.get(name.strip().lower())
        if family is not None:
            return family
    return None


def install_steps(component: DriverComponent, host: HostInfo) -> list[str]:
    """How to provide ``component`` on ``host``, ending with :data:`SYSTEM_CHANGES_NOTE`."""
    family = distro_family(host)
    steps = _STEPS.get(component, {}).get(family) if family is not None else None
    return [*(steps or _GENERIC_STEPS[component]), SYSTEM_CHANGES_NOTE]


def claimed_device_steps(kind: str, address: str, driver: str) -> list[str]:
    """What to do about an accelerator another kernel driver has claimed.

    A device bound to, say, ``vfio-pci`` for passthrough is not missing a
    driver, and ``modprobe`` will not take it back.
    """
    return [
        f"The {kind} at {address} is claimed by the '{driver}' kernel driver (vfio-pci, for "
        "example, reserves it for a virtual machine), so this host cannot use it.",
        f"To use it here, release it from '{driver}' (undo the passthrough configuration) "
        "and reboot.",
        SYSTEM_CHANGES_NOTE,
    ]


def is_legacy_gpu(device_id: str) -> bool:
    """Whether a GPU's PCI device id belongs to a generation before Gen12."""
    normalised = device_id.strip().lower()
    return any(normalised.startswith(prefix) for prefix in _LEGACY_GPU_ID_PREFIXES)
