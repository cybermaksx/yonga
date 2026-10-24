"""Host, device and filesystem facts observed on the local machine.

This module is deliberately descriptive: it records *what is there*, never what
should be used. Recommendations live in ``yonga.domain.compatibility``.
Nothing here imports OpenVINO; adapters fill these models in.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DeviceKind(str, Enum):
    """Coarse class of an accelerator as exposed by OpenVINO."""

    CPU = "CPU"
    GPU = "GPU"
    NPU = "NPU"
    OTHER = "OTHER"


#: Canonical ordering used whenever devices are listed to a user.
DEVICE_ORDER: tuple[str, ...] = ("CPU", "GPU", "NPU")

#: Filesystem types that are backed by RAM rather than by a block device.
TMPFS_FSTYPES: frozenset[str] = frozenset({"tmpfs", "ramfs"})
RAM_BACKED_FSTYPES: frozenset[str] = frozenset({"tmpfs", "ramfs", "devtmpfs"})

_DEVICE_INDEX_SUFFIX = re.compile(r"(?:\.\d+)+$")
_PACKAGE_SEPARATORS = re.compile(r"[-_.]+")


def normalise_device(name: str) -> str:
    """Return the canonical device name for *name*.

    ``"GPU.0"`` and ``"gpu"`` both become ``"GPU"``; any trailing ``.N`` index
    suffix is removed and the remainder is upper-cased. Unknown device names are
    passed through with the same treatment so they stay comparable. An empty or
    whitespace-only name returns an empty string rather than raising.
    """
    cleaned = name.strip()
    if not cleaned:
        return ""
    return _DEVICE_INDEX_SUFFIX.sub("", cleaned).upper()


def device_kind(name: str) -> DeviceKind:
    """Classify a device name into a :class:`DeviceKind`."""
    normalised = normalise_device(name)
    if normalised == DeviceKind.CPU.value:
        return DeviceKind.CPU
    if normalised == DeviceKind.GPU.value:
        return DeviceKind.GPU
    if normalised == DeviceKind.NPU.value:
        return DeviceKind.NPU
    return DeviceKind.OTHER


def _normalise_package_name(name: str) -> str:
    """Normalise a distribution name so ``openvino-genai`` matches ``openvino_genai``."""
    return _PACKAGE_SEPARATORS.sub("-", name.strip().lower())


class OpenVINODevice(BaseModel):
    """A device reported by ``openvino.Core().available_devices``."""

    model_config = ConfigDict(extra="ignore")

    name: str
    kind: DeviceKind = DeviceKind.OTHER
    full_name: str = "unknown"
    architecture: str | None = None
    properties: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _derive_kind(self) -> OpenVINODevice:
        """Derive the kind from the device name when the caller left it unset."""
        if self.kind is DeviceKind.OTHER:
            self.kind = device_kind(self.name)
        return self

    @property
    def normalised_name(self) -> str:
        """The canonical name, e.g. ``"GPU"`` for a device called ``"GPU.0"``."""
        return normalise_device(self.name)

    @property
    def is_available(self) -> bool:
        """Always ``True``: OpenVINO only enumerates devices its plugins can open."""
        return True


class FilesystemInfo(BaseModel):
    """Capacity and type facts for the filesystem holding a given path."""

    model_config = ConfigDict(extra="ignore")

    path: Path
    mount_point: Path | None = None
    fstype: str = "unknown"
    total_bytes: int = 0
    free_bytes: int = 0
    available_bytes: int = 0

    @property
    def is_tmpfs(self) -> bool:
        """True when the filesystem is a tmpfs/ramfs (its size is RAM, not disk)."""
        return self.fstype.strip().lower() in TMPFS_FSTYPES

    @property
    def is_ram_backed(self) -> bool:
        """True when writing here consumes RAM instead of persistent storage."""
        return self.fstype.strip().lower() in RAM_BACKED_FSTYPES


class MemoryInfo(BaseModel):
    """System RAM and swap totals in bytes."""

    model_config = ConfigDict(extra="ignore")

    total_bytes: int = 0
    available_bytes: int = 0
    swap_total_bytes: int = 0
    swap_free_bytes: int = 0


class NPUNodeInfo(BaseModel):
    """Linux ``/dev/accel`` node facts used to explain NPU accessibility."""

    model_config = ConfigDict(extra="ignore")

    directory_exists: bool = False
    node_path: Path | None = None
    node_exists: bool = False
    readable: bool = False
    writable: bool = False
    group_name: str | None = None
    user_in_group: bool = False
    user_groups: list[str] = Field(default_factory=list)
    driver_version: str | None = None
    compiler_libraries: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def accessible(self) -> bool:
        """True when the accelerator node exists and the current user can use it."""
        return self.node_exists and self.readable and self.writable


class PciAccelerator(BaseModel):
    """An Intel GPU or NPU found on the PCI bus, whatever OpenVINO makes of it."""

    model_config = ConfigDict(extra="ignore")

    kind: DeviceKind
    address: str
    """PCI address, ``0000:00:02.0``."""
    vendor_id: str
    device_id: str
    kernel_driver: str | None = None
    """The kernel module bound to it (``i915``, ``xe``, ``intel_vpu``), if any."""


class DriverLibraries(BaseModel):
    """User-space driver libraries OpenVINO's GPU and NPU plugins load at run time."""

    model_config = ConfigDict(extra="ignore")

    gpu_opencl: list[str] = Field(default_factory=list)
    """Intel's OpenCL driver (compute-runtime), as named by an ICD file."""
    level_zero_loader: list[str] = Field(default_factory=list)
    npu_level_zero: list[str] = Field(default_factory=list)
    """The Level Zero NPU driver, ``libze_intel_npu.so``."""


class PackageInfo(BaseModel):
    """Installed-ness of a single Python distribution relevant to the stack."""

    model_config = ConfigDict(extra="ignore")

    name: str
    version: str | None = None
    importable: bool = False
    import_error: str | None = None

    @property
    def present(self) -> bool:
        """True when the distribution is installed or importable."""
        return self.importable or self.version is not None


class HostInfo(BaseModel):
    """Operating system and CPU identity of the host."""

    model_config = ConfigDict(extra="ignore")

    platform: str = "unknown"
    machine: str = "unknown"
    kernel: str = "unknown"
    python_version: str = "unknown"
    distro_id: str | None = None
    distro_version: str | None = None
    distro_pretty: str | None = None
    distro_like: list[str] = Field(default_factory=list)
    """``ID_LIKE`` from ``/etc/os-release``: the distributions this one derives from."""
    cpu_model: str | None = None
    cpu_count: int | None = None


class HuggingFaceAuthInfo(BaseModel):
    """Hugging Face authentication *state* — never the credential itself.

    ``extra="forbid"`` is deliberate: it makes it impossible to smuggle a token
    into this model through an unexpected keyword, and ``token_source`` only ever
    names where a credential was found (for example ``"environment"``).
    """

    model_config = ConfigDict(extra="forbid")

    logged_in: bool = False
    user: str | None = None
    token_source: str | None = None
    offline: bool = False
    cache_dir: Path | None = None

    @field_validator("token_source")
    @classmethod
    def _reject_credential_values(cls, value: str | None) -> str | None:
        """Reject anything that looks like a credential rather than a source label."""
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        if len(cleaned) > 64 or cleaned.startswith(("hf_", "api_org_", "sk-")):
            raise ValueError(
                "token_source must name where the credential came from, never the credential"
            )
        return cleaned


class HardwareInventory(BaseModel):
    """Everything ``doctor`` and the preflight stage learned about this machine."""

    model_config = ConfigDict(extra="ignore")

    host: HostInfo = Field(default_factory=HostInfo)
    memory: MemoryInfo = Field(default_factory=MemoryInfo)
    filesystems: list[FilesystemInfo] = Field(default_factory=list)
    devices: list[OpenVINODevice] = Field(default_factory=list)
    openvino_available: bool = False
    openvino_import_error: str | None = None
    npu: NPUNodeInfo | None = None
    accelerators: list[PciAccelerator] | None = None
    """Intel GPUs and NPUs on the PCI bus; ``None`` when the bus was not read."""
    driver_libraries: DriverLibraries | None = None
    packages: list[PackageInfo] = Field(default_factory=list)
    device_tags: list[str] = Field(default_factory=list)
    system_temp: FilesystemInfo | None = None
    managed_temp: FilesystemInfo | None = None
    hf_auth: HuggingFaceAuthInfo | None = None

    def device(self, name: str) -> OpenVINODevice | None:
        """Look a device up by exact name first, then by normalised name."""
        for entry in self.devices:
            if entry.name == name:
                return entry
        wanted = normalise_device(name)
        if not wanted:
            return None
        for entry in self.devices:
            if entry.normalised_name == wanted:
                return entry
        return None

    def has_device(self, name: str) -> bool:
        """True when a device matching *name* (normalised) is present."""
        return self.device(name) is not None

    def package(self, name: str) -> PackageInfo | None:
        """Look a package up, tolerating ``-``/``_``/``.`` and case differences."""
        wanted = _normalise_package_name(name)
        for entry in self.packages:
            if _normalise_package_name(entry.name) == wanted:
                return entry
        return None

    @property
    def available_device_names(self) -> list[str]:
        """Normalised, de-duplicated device names ordered CPU, GPU, NPU, then A-Z."""
        seen: list[str] = []
        for entry in self.devices:
            normalised = entry.normalised_name
            if normalised and normalised not in seen:
                seen.append(normalised)
        ordered = [name for name in DEVICE_ORDER if name in seen]
        ordered.extend(sorted(name for name in seen if name not in DEVICE_ORDER))
        return ordered
