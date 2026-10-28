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

#: PCI device ids of the Intel GPUs before Gen12, from Gen2 through Gen11 (Ice
#: Lake, Elkhart Lake, Jasper Lake), as the Linux kernel lists them in
#: include/drm/intel/i915_pciids.h (v6.12). Intel's current compute-runtime
#: covers Gen12 (Tiger Lake) and newer; these older iGPUs are served by its
#: legacy branch at most, so installing the current runtime cannot make OpenVINO
#: list them. Whole ids, not prefixes: a newer part may share a prefix with one.
# fmt: off
_LEGACY_GPU_DEVICE_IDS: frozenset[int] = frozenset(
    {
        # Gen2 to Gen4: i810 to G45 and Pineview
        0x1132, 0x2562, 0x2572, 0x2582, 0x258a, 0x2592, 0x2772, 0x27a2, 0x27ae, 0x2972, 0x2982,
        0x2992, 0x29a2, 0x29b2, 0x29c2, 0x29d2, 0x2a02, 0x2a12, 0x2a42, 0x2e02, 0x2e12, 0x2e22,
        0x2e32, 0x2e42, 0x2e92, 0x3577, 0x3582, 0x358e, 0x7121, 0x7123, 0x7125, 0xa001, 0xa011,
        # Ironlake
        0x0042, 0x0046,
        # Sandy Bridge
        0x0102, 0x0106, 0x010a, 0x0112, 0x0116, 0x0122, 0x0126,
        # Ivy Bridge
        0x0152, 0x0156, 0x015a, 0x0162, 0x0166, 0x016a,
        # Haswell
        0x0402, 0x0406, 0x040a, 0x040b, 0x040e, 0x0412, 0x0416, 0x041a, 0x041b, 0x041e, 0x0422,
        0x0426, 0x042a, 0x042b, 0x042e, 0x0a02, 0x0a06, 0x0a0a, 0x0a0b, 0x0a0e, 0x0a12, 0x0a16,
        0x0a1a, 0x0a1b, 0x0a1e, 0x0a22, 0x0a26, 0x0a2a, 0x0a2b, 0x0a2e, 0x0c02, 0x0c06, 0x0c0a,
        0x0c0b, 0x0c0e, 0x0c12, 0x0c16, 0x0c1a, 0x0c1b, 0x0c1e, 0x0c22, 0x0c26, 0x0c2a, 0x0c2b,
        0x0c2e, 0x0d02, 0x0d06, 0x0d0a, 0x0d0b, 0x0d0e, 0x0d12, 0x0d16, 0x0d1a, 0x0d1b, 0x0d1e,
        0x0d22, 0x0d26, 0x0d2a, 0x0d2b, 0x0d2e,
        # Bay Trail
        0x0f30, 0x0f31, 0x0f32, 0x0f33,
        # Broadwell
        0x1602, 0x1606, 0x160a, 0x160b, 0x160d, 0x160e, 0x1612, 0x1616, 0x161a, 0x161b, 0x161d,
        0x161e, 0x1622, 0x1626, 0x162a, 0x162b, 0x162d, 0x162e, 0x1632, 0x1636, 0x163a, 0x163b,
        0x163d, 0x163e,
        # Cherry Trail
        0x22b0, 0x22b1, 0x22b2, 0x22b3,
        # Skylake
        0x1902, 0x1906, 0x190a, 0x190b, 0x190e, 0x1912, 0x1913, 0x1915, 0x1916, 0x1917, 0x191a,
        0x191b, 0x191d, 0x191e, 0x1921, 0x1923, 0x1926, 0x1927, 0x192a, 0x192b, 0x192d, 0x1932,
        0x193a, 0x193b, 0x193d,
        # Apollo Lake (Broxton)
        0x0a84, 0x1a84, 0x1a85, 0x5a84, 0x5a85,
        # Gemini Lake
        0x3184, 0x3185,
        # Kaby Lake and Amber Lake
        0x5902, 0x5906, 0x5908, 0x590a, 0x590b, 0x590e, 0x5912, 0x5913, 0x5915, 0x5916, 0x5917,
        0x591a, 0x591b, 0x591c, 0x591d, 0x591e, 0x5921, 0x5923, 0x5926, 0x5927, 0x593b, 0x87c0,
        0x87ca,
        # Comet Lake
        0x9b21, 0x9b41, 0x9ba2, 0x9ba4, 0x9ba5, 0x9ba8, 0x9baa, 0x9bac, 0x9bc2, 0x9bc4, 0x9bc5,
        0x9bc6, 0x9bc8, 0x9bca, 0x9bcc, 0x9be6, 0x9bf6,
        # Coffee Lake and Whiskey Lake
        0x3e90, 0x3e91, 0x3e92, 0x3e93, 0x3e94, 0x3e96, 0x3e98, 0x3e99, 0x3e9a, 0x3e9b, 0x3e9c,
        0x3ea0, 0x3ea1, 0x3ea2, 0x3ea3, 0x3ea4, 0x3ea5, 0x3ea6, 0x3ea7, 0x3ea8, 0x3ea9,
        # Cannon Lake
        0x5a40, 0x5a41, 0x5a42, 0x5a44, 0x5a49, 0x5a4a, 0x5a4c, 0x5a50, 0x5a51, 0x5a52, 0x5a54,
        0x5a59, 0x5a5a, 0x5a5c,
        # Ice Lake (Gen11)
        0x8a50, 0x8a51, 0x8a52, 0x8a53, 0x8a54, 0x8a56, 0x8a57, 0x8a58, 0x8a59, 0x8a5a, 0x8a5b,
        0x8a5c, 0x8a5d, 0x8a70, 0x8a71,
        # Elkhart Lake (Gen11)
        0x4541, 0x4551, 0x4555, 0x4557, 0x4570, 0x4571,
        # Jasper Lake (Gen11)
        0x4e51, 0x4e55, 0x4e57, 0x4e61, 0x4e71,
    }
)
# fmt: on


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
    try:
        return int(device_id.strip(), 16) in _LEGACY_GPU_DEVICE_IDS
    except ValueError:
        return False
