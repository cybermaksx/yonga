"""Read-only Linux host probe backing ``yonga doctor``.

Everything this module collects is observable without root and without changing
anything: ``platform``, ``/etc/os-release``, ``/proc/cpuinfo``, ``psutil``,
``importlib.metadata``, ``/dev/accel`` and a handful of library directories.
It never runs ``sudo``, never modifies permissions or group membership, and
never installs a package.

It also never imports OpenVINO. The device list is the one thing here that
genuinely needs the runtime, so it is *injected*: either as an explicit
``devices=`` argument to :meth:`LinuxSystemProbe.inventory`, or through a
``device_source`` callable handed to the constructor. That keeps
``import yonga.cli`` working on a machine with no Intel stack installed, and
keeps the whole probe unit-testable with a fake device list.

Every field degrades to ``None``/``"unknown"`` plus a note rather than raising:
a diagnostic tool that crashes while diagnosing is worthless.
"""

from __future__ import annotations

import contextlib
import fnmatch
import importlib.util
import os
import platform
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from yonga.adapters.filesystem import FilesystemAdapter, system_temp_dir
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
from yonga.util.versions import installed_version

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yonga.adapters import FilesystemProbe
    from yonga.config.paths import Paths
    from yonga.config.settings import Settings

__all__ = [
    "ACCEL_DIRNAME",
    "INTEL_PCI_VENDOR",
    "LEVEL_ZERO_LOADER_PATTERNS",
    "NPU_LEVEL_ZERO_PATTERNS",
    "NPU_LIBRARY_PATTERNS",
    "PROBED_PACKAGES",
    "DeviceSnapshot",
    "LinuxSystemProbe",
    "derive_device_tags",
    "parse_os_release",
]

#: Directory the Linux accel subsystem exposes the NPU through.
ACCEL_DIRNAME: str = "accel"

#: Glob patterns for the NPU compiler shared libraries (the compiler and the
#: driver's loader for it). The plugin itself is not one: every openvino wheel
#: carries it, so finding it says nothing about the compiler.
NPU_LIBRARY_PATTERNS: tuple[str, ...] = ("libopenvino_intel_npu_compiler*.so*",)

#: PCI vendor id of Intel, the only vendor OpenVINO's GPU and NPU plugins drive.
INTEL_PCI_VENDOR: str = "0x8086"

#: PCI class prefixes (``/sys/bus/pci/devices/*/class``) of the accelerators:
#: display controllers (``0x03``) are GPUs, processing accelerators (``0x1200``)
#: are the NPU.
_PCI_CLASS_KINDS: tuple[tuple[str, DeviceKind], ...] = (
    ("0x03", DeviceKind.GPU),
    ("0x1200", DeviceKind.NPU),
)

#: Where OpenCL ICD loaders look for vendor drivers; Intel's names ``libigdrcl``.
_OPENCL_VENDOR_DIRS: tuple[str, ...] = ("/etc/OpenCL/vendors",)
_INTEL_OPENCL_MARKER = "libigdrcl"

#: Glob patterns for the Level Zero loader and the Level Zero NPU driver
#: (``libze_intel_vpu`` is the name before the 2024 rename).
LEVEL_ZERO_LOADER_PATTERNS: tuple[str, ...] = ("libze_loader.so*",)
NPU_LEVEL_ZERO_PATTERNS: tuple[str, ...] = ("libze_intel_npu.so*", "libze_intel_vpu.so*")

#: System directories searched for those libraries, in order. ``/usr/lib64`` is
#: where Fedora installs them.
_SYSTEM_LIBRARY_DIRS: tuple[str, ...] = (
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib",
    "/usr/lib64",
    "/usr/local/lib",
)

#: ``(distribution name, importable top-level module)`` pairs reported by doctor.
#:
#: Only top-level module names appear here on purpose: ``find_spec`` has to
#: import the *parent* of a dotted name, and importing ``optimum.intel`` or
#: ``openvino`` inside a probe would defeat the whole point of lazy imports.
PROBED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("openvino", "openvino"),
    ("openvino-genai", "openvino_genai"),
    ("openvino-tokenizers", "openvino_tokenizers"),
    ("optimum-intel", "optimum"),
    ("nncf", "nncf"),
    ("transformers", "transformers"),
    ("huggingface-hub", "huggingface_hub"),
)

_CPU_MODEL_KEYS: tuple[str, ...] = ("model name", "Model name", "Processor", "cpu model")
_CPU_VENDOR_KEYS: tuple[str, ...] = ("vendor_id", "CPU implementer")

#: ``Intel(R) Core(TM) Ultra 7 268V`` -> series 2 (the first digit of ``268``).
_CORE_ULTRA = re.compile(r"core\s*\(?tm\)?\s*ultra\s+\d+\s+(?P<series>\d)\d\d", re.IGNORECASE)

#: ``GPU: vendor=0x8086 arch=v20.4.4`` or a bare ``20.4.4`` from ``GPU_UARCH_VERSION``.
_GPU_ARCH = re.compile(r"(?:arch=v)?(?P<major>\d+)\.\d+\.\d+")

#: OpenVINO GPU architecture majors that map to a documented Xe generation.
_GPU_UARCH_TAGS: Mapping[int, str] = {12: "xe", 20: "xe2", 30: "xe3"}

#: OpenVINO NPU ``DEVICE_ARCHITECTURE`` values that name a documented platform.
_NPU_ARCH_TAGS: Mapping[str, str] = {"3720": "meteor_lake", "4000": "lunar_lake"}

_INTEL_MARKERS: tuple[str, ...] = ("intel", "genuineintel", "0x8086")


def parse_os_release(text: str) -> dict[str, str]:
    """Parse ``/etc/os-release`` content into a plain mapping.

    Values may be quoted with single or double quotes; malformed lines are
    skipped rather than raising, because this file is parsed on the failure path
    of ``doctor`` as often as on the happy path.
    """
    result: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        name = key.strip()
        if not name:
            continue
        cleaned = value.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
            cleaned = cleaned[1:-1]
        result[name] = cleaned.replace("\\$", "$").replace('\\"', '"').replace("\\\\", "\\")
    return result


@dataclass(frozen=True)
class DeviceSnapshot:
    """What an OpenVINO probe reported, as plain data.

    This is the contract between :class:`LinuxSystemProbe` and whatever knows
    how to talk to OpenVINO. The probe never constructs one itself.
    """

    devices: tuple[OpenVINODevice, ...] = ()
    available: bool = False
    import_error: str | None = None


DeviceSource = Callable[[], DeviceSnapshot]
"""Callable injected into the probe so it can obtain devices without importing OpenVINO."""


def _first_property(device: OpenVINODevice, *names: str) -> str | None:
    """Return the first non-empty property value among ``names``."""
    for name in names:
        value = device.properties.get(name)
        if value:
            return value
    return None


def _looks_intel(*values: str | None) -> bool:
    """Whether any of the given strings identifies Intel hardware."""
    for value in values:
        if not value:
            continue
        lowered = value.lower()
        if any(marker in lowered for marker in _INTEL_MARKERS):
            return True
    return False


def derive_device_tags(
    *,
    devices: Sequence[OpenVINODevice] = (),
    host: HostInfo | None = None,
    cpu_vendor: str | None = None,
) -> list[str]:
    """Derive normalised host tags such as ``intel.npu`` from observed facts.

    ``docs/REGISTRY_SPEC.md`` section 10 is explicit: "if uncertain, omit a tag
    rather than guess". So a coarse tag is emitted only when a stable property
    identifies the vendor, and a finer tag only when a stable *numeric* property
    names the generation:

    * ``intel.cpu`` — ``/proc/cpuinfo`` vendor ``GenuineIntel``, the CPU model
      string, or an OpenVINO ``CPU`` device whose full name says Intel;
    * ``intel.gpu`` / ``intel.npu`` — an OpenVINO device of that kind whose full
      name or architecture identifies Intel (PCI vendor ``0x8086`` counts);
    * ``intel.gpu.xe2`` — GPU architecture major version 20 (12 = Xe, 30 = Xe3);
    * ``intel.npu.lunar_lake`` — NPU ``DEVICE_ARCHITECTURE`` ``4000``
      (``3720`` = Meteor Lake);
    * ``intel.core_ultra.series2`` — a Core Ultra model number of the form
      ``2xx``.

    Marketing names are never inferred from anything softer than those.
    """
    tags: list[str] = []
    host_info = host or HostInfo()

    cpu_device = next((d for d in devices if d.kind is DeviceKind.CPU), None)
    if _looks_intel(cpu_vendor, host_info.cpu_model, cpu_device.full_name if cpu_device else None):
        tags.append("intel.cpu")
        series = _CORE_ULTRA.search(host_info.cpu_model or "") or _CORE_ULTRA.search(
            cpu_device.full_name if cpu_device else ""
        )
        if series is not None:
            tags.append(f"intel.core_ultra.series{series.group('series')}")

    # OpenVINO's ``GPU`` and ``NPU`` device names are produced by the Intel GPU and
    # Intel NPU plugins only, so enumerating one is itself evidence of Intel hardware;
    # the name/architecture check below is a belt-and-braces confirmation.
    gpu = next((d for d in devices if d.kind is DeviceKind.GPU), None)
    if gpu is not None:
        tags.append("intel.gpu")
        raw_arch = _first_property(gpu, "GPU_UARCH_VERSION") or gpu.architecture or ""
        match = _GPU_ARCH.search(raw_arch)
        if match is not None:
            uarch = _GPU_UARCH_TAGS.get(int(match.group("major")))
            if uarch is not None:
                tags.append(f"intel.gpu.{uarch}")

    npu = next((d for d in devices if d.kind is DeviceKind.NPU), None)
    if npu is not None:
        tags.append("intel.npu")
        arch = (npu.architecture or _first_property(npu, "DEVICE_ARCHITECTURE") or "").strip()
        platform_tag = _NPU_ARCH_TAGS.get(arch)
        if platform_tag is not None:
            tags.append(f"intel.npu.{platform_tag}")

    return list(dict.fromkeys(tags))


class LinuxSystemProbe:
    """Collects host, memory, package, NPU-node and filesystem facts.

    Implements :class:`yonga.adapters.HardwareProbe`. The protocol declares
    ``inventory(self, *, full: bool = False)``, so every additional parameter
    here has a default and the class stays a structural match; callers that know
    more (the ``doctor`` service, which has already talked to OpenVINO) pass the
    device list in, and callers that do not get an inventory with
    ``openvino_available=False`` and an explanatory note.

    Two injection points exist for the OpenVINO device list because two
    different callers need them:

    * ``device_source`` on the constructor, for wiring the real OpenVINO probe
      once and then treating this object as a plain ``HardwareProbe``;
    * ``devices=``/``openvino_available=``/``openvino_import_error=`` on
      :meth:`inventory`, for a caller that already holds a probe result.

    The explicit arguments win over ``device_source`` when both are present.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        paths: Paths | None = None,
        device_source: DeviceSource | None = None,
        filesystem: FilesystemProbe | None = None,
        root: Path | None = None,
        library_dirs: Sequence[Path] | None = None,
    ) -> None:
        """Create a probe.

        ``root`` re-bases ``/proc``, ``/etc`` and ``/dev`` onto a fake tree so
        the whole probe can be tested without privileges; ``library_dirs``
        overrides the NPU library search path for the same reason.
        """
        self._settings = settings
        self._paths = paths
        self._device_source = device_source
        self._filesystem: FilesystemProbe = filesystem or FilesystemAdapter()
        self._root = Path(root) if root is not None else Path("/")
        self._library_dirs: tuple[Path, ...] | None = (
            tuple(library_dirs) if library_dirs is not None else None
        )
        self._cpu_identity: tuple[str | None, str | None] | None = None

    # --- helpers -------------------------------------------------------
    def _under_root(self, *parts: str) -> Path:
        """Resolve an absolute system path against the configured root."""
        return self._root.joinpath(*parts)

    def _read_text(self, path: Path) -> str | None:
        """Read a small system file, returning ``None`` when it is unavailable."""
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            return None

    def _resolved_paths(self) -> Paths | None:
        """Return the managed directory layout, if one was supplied."""
        if self._paths is not None:
            return self._paths
        if self._settings is not None:
            try:
                return self._settings.paths
            except Exception:  # pragma: no cover - settings always builds paths
                return None
        return None

    # --- host ----------------------------------------------------------
    def host_info(self, notes: list[str] | None = None) -> HostInfo:
        """Collect OS, kernel, Python and CPU identity."""
        collected = notes if notes is not None else []
        os_release_path = self._under_root("etc", "os-release")
        text = self._read_text(os_release_path)
        if text is None:
            text = self._read_text(self._under_root("usr", "lib", "os-release"))
        if text is None:
            collected.append(f"{os_release_path} is unreadable; the distribution is unknown.")
        release = parse_os_release(text) if text else {}

        cpu_model, cpu_vendor = self.cpu_identity(collected)
        try:
            cpu_count = os.cpu_count()
        except (OSError, NotImplementedError):  # pragma: no cover - exotic platform
            cpu_count = None

        if cpu_vendor and not cpu_model:
            collected.append("CPU model name is not exposed by this kernel.")

        return HostInfo(
            platform=platform.system() or "unknown",
            machine=platform.machine() or "unknown",
            kernel=platform.release() or "unknown",
            python_version=platform.python_version(),
            distro_id=release.get("ID"),
            distro_version=release.get("VERSION_ID"),
            distro_pretty=release.get("PRETTY_NAME") or release.get("NAME"),
            distro_like=release.get("ID_LIKE", "").split(),
            cpu_model=cpu_model,
            cpu_count=cpu_count,
        )

    def cpu_identity(self, notes: list[str] | None = None) -> tuple[str | None, str | None]:
        """Return ``(model name, vendor id)`` from ``/proc/cpuinfo``."""
        collected = notes if notes is not None else []
        if self._cpu_identity is not None:
            return self._cpu_identity
        cpuinfo = self._under_root("proc", "cpuinfo")
        text = self._read_text(cpuinfo)
        if text is None:
            collected.append(f"{cpuinfo} is unreadable; the CPU model is unknown.")
            self._cpu_identity = (None, None)
            return self._cpu_identity

        model: str | None = None
        vendor: str | None = None
        for raw_line in text.splitlines():
            if ":" not in raw_line:
                continue
            key, _, value = raw_line.partition(":")
            name = key.strip()
            cleaned = value.strip()
            if not cleaned:
                continue
            if model is None and name in _CPU_MODEL_KEYS:
                model = cleaned
            elif vendor is None and name in _CPU_VENDOR_KEYS:
                vendor = cleaned
            if model is not None and vendor is not None:
                break
        self._cpu_identity = (model, vendor)
        return self._cpu_identity

    # --- memory --------------------------------------------------------
    def memory_info(self, notes: list[str] | None = None) -> MemoryInfo:
        """Collect RAM and swap totals through ``psutil``."""
        collected = notes if notes is not None else []
        try:
            import psutil
        except ImportError as exc:
            collected.append(f"psutil is not importable ({exc}); memory figures are unavailable.")
            return MemoryInfo()

        total = available = swap_total = swap_free = 0
        try:
            virtual = psutil.virtual_memory()
            total = int(virtual.total)
            available = int(virtual.available)
        except Exception as exc:
            collected.append(f"Physical memory could not be read: {exc}.")
        try:
            swap = psutil.swap_memory()
            swap_total = int(swap.total)
            swap_free = int(swap.free)
        except Exception as exc:
            collected.append(f"Swap could not be read: {exc}.")

        return MemoryInfo(
            total_bytes=total,
            available_bytes=available,
            swap_total_bytes=swap_total,
            swap_free_bytes=swap_free,
        )

    # --- packages ------------------------------------------------------
    def packages(self) -> list[PackageInfo]:
        """Report the installed version and importability of the stack packages.

        Importability is checked with :func:`importlib.util.find_spec`, which
        locates a module without executing it. The heavy modules are *never*
        imported here: loading OpenVINO or Torch to answer "is it installed?"
        would cost seconds and could itself crash on a broken installation.
        """
        result: list[PackageInfo] = []
        for dist_name, module_name in PROBED_PACKAGES:
            version = installed_version(dist_name)
            importable = False
            import_error: str | None = None
            try:
                importable = importlib.util.find_spec(module_name) is not None
            except (ImportError, ValueError, AttributeError) as exc:
                import_error = f"{type(exc).__name__}: {exc}"
            if not importable and import_error is None:
                import_error = f"module {module_name!r} was not found on sys.path"
            result.append(
                PackageInfo(
                    name=dist_name,
                    version=version,
                    importable=importable,
                    import_error=import_error,
                )
            )
        return result

    # --- NPU node ------------------------------------------------------
    def library_search_dirs(self) -> list[Path]:
        """Directories searched for the NPU compiler libraries."""
        if self._library_dirs is not None:
            return list(self._library_dirs)

        dirs: list[Path] = []
        machine = platform.machine() or ""
        if machine:
            dirs.append(self._under_root("usr", "lib", f"{machine}-linux-gnu"))
        dirs.extend(
            self._under_root(*entry.strip("/").split("/")) for entry in _SYSTEM_LIBRARY_DIRS
        )

        package_libs = _openvino_package_libs()
        if package_libs is not None:
            dirs.append(package_libs)

        seen: dict[Path, None] = {}
        for entry in dirs:
            seen.setdefault(entry, None)
        return list(seen)

    def npu_node(
        self,
        *,
        driver_version: str | None = None,
        notes: list[str] | None = None,
    ) -> NPUNodeInfo:
        """Inspect ``/dev/accel`` accessibility and the NPU compiler libraries.

        Strictly read-only: permissions and group membership are reported, never
        changed. The remediation for a missing ``render`` group belongs in the
        doctor report, not in this adapter.
        """
        collected = notes if notes is not None else []
        local_notes: list[str] = []

        accel_dir = self._under_root("dev", ACCEL_DIRNAME)
        directory_exists = _is_dir(accel_dir)

        node_path: Path | None = None
        if directory_exists:
            node_path = _pick_accel_node(accel_dir)
            if node_path is None:
                local_notes.append(
                    f"{accel_dir} exists but contains no accel* node; "
                    "the NPU kernel driver (intel_vpu) may not be loaded."
                )
        else:
            local_notes.append(
                f"{accel_dir} does not exist; this kernel exposes no accel device "
                "(the intel_vpu module may not be loaded, or this host has no NPU)."
            )

        node_exists = node_path is not None
        readable = writable = False
        group_name: str | None = None
        group_id: int | None = None
        if node_path is not None:
            readable = _access(node_path, os.R_OK)
            writable = _access(node_path, os.W_OK)
            try:
                group_id = node_path.stat().st_gid
            except OSError as exc:
                local_notes.append(f"{node_path} could not be stat'ed: {exc.strerror or exc}.")
            if group_id is not None:
                group_name = _group_name(group_id)
                if group_name is None:
                    local_notes.append(
                        f"The group owning {node_path} (gid {group_id}) has no name in "
                        "the group database."
                    )

        user_groups = _user_group_names()
        user_in_group = bool(group_name and group_name in user_groups)
        if node_exists and not (readable and writable) and group_name:
            local_notes.append(
                f"{node_path} is owned by group {group_name!r} and the current user "
                f"{'is' if user_in_group else 'is not'} a member of it."
            )

        libraries = self._find_npu_libraries()
        if not libraries:
            local_notes.append(
                "No libopenvino_intel_npu_compiler*.so was found in the searched library "
                "directories; NPU compilation may rely on the compiler shipped inside the "
                "OpenVINO NPU plugin."
            )

        collected.extend(local_notes)
        return NPUNodeInfo(
            directory_exists=directory_exists,
            node_path=node_path,
            node_exists=node_exists,
            readable=readable,
            writable=writable,
            group_name=group_name,
            user_in_group=user_in_group,
            user_groups=user_groups,
            driver_version=driver_version,
            compiler_libraries=libraries,
            notes=local_notes,
        )

    def _find_npu_libraries(self) -> list[str]:
        """Return the NPU compiler libraries found on this host."""
        return self._find_libraries({"compiler": NPU_LIBRARY_PATTERNS})["compiler"]

    def _find_libraries(self, groups: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
        """Return, per group, the files matching its patterns in the library directories.

        Each directory is listed once, however many patterns are matched in it,
        and a directory that is a link to another one searched (Arch's
        ``/usr/lib64`` -> ``/usr/lib``) is not listed again.
        """
        found: dict[str, dict[str, None]] = {key: {} for key in groups}
        seen: set[Path] = set()
        for directory in self.library_search_dirs():
            if not _is_dir(directory):
                continue
            try:
                real = directory.resolve()
            except OSError:  # pragma: no cover - a loop of links
                continue
            if real in seen:
                continue
            seen.add(real)
            try:
                names = sorted(entry.name for entry in directory.iterdir())
            except OSError:  # pragma: no cover - unreadable directory
                continue
            for key, patterns in groups.items():
                for name in names:
                    if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
                        found[key].setdefault(str(directory / name), None)
        return {key: list(paths) for key, paths in found.items()}

    # --- accelerators and their drivers ---------------------------------
    def pci_accelerators(self, notes: list[str] | None = None) -> list[PciAccelerator] | None:
        """Return the Intel GPUs and NPUs on the PCI bus, or ``None`` when it cannot be read.

        This is the hardware, independent of OpenVINO: a GPU or an NPU listed
        here but absent from ``available_devices`` is missing a driver, not
        missing from the machine.
        """
        bus = self._under_root("sys", "bus", "pci", "devices")
        if not _is_dir(bus):
            if notes is not None:
                notes.append(f"{bus} is not readable; PCI accelerators were not listed.")
            return None
        found: list[PciAccelerator] = []
        try:
            entries = sorted(bus.iterdir())
        except OSError:  # pragma: no cover - unreadable sysfs
            return None
        for entry in entries:
            vendor = (self._read_text(entry / "vendor") or "").strip().lower()
            if vendor != INTEL_PCI_VENDOR:
                continue
            pci_class = (self._read_text(entry / "class") or "").strip().lower()
            kind = next((k for prefix, k in _PCI_CLASS_KINDS if pci_class.startswith(prefix)), None)
            if kind is None:
                continue
            driver: str | None = None
            try:
                driver = (entry / "driver").resolve(strict=True).name
            except OSError:
                driver = None
            found.append(
                PciAccelerator(
                    kind=kind,
                    address=entry.name,
                    vendor_id=vendor,
                    device_id=(self._read_text(entry / "device") or "").strip().lower(),
                    kernel_driver=driver,
                )
            )
        return found

    def driver_libraries(self) -> DriverLibraries:
        """Find the user-space GPU and NPU drivers OpenVINO's plugins load."""
        opencl: list[str] = []
        for entry in _OPENCL_VENDOR_DIRS:
            directory = self._under_root(*entry.strip("/").split("/"))
            if not _is_dir(directory):
                continue
            for icd in sorted(directory.glob("*.icd")):
                library = (self._read_text(icd) or "").strip()
                if _INTEL_OPENCL_MARKER in library:
                    opencl.append(library)
        found = self._find_libraries(
            {"loader": LEVEL_ZERO_LOADER_PATTERNS, "npu": NPU_LEVEL_ZERO_PATTERNS}
        )
        return DriverLibraries(
            gpu_opencl=opencl,
            level_zero_loader=found["loader"],
            npu_level_zero=found["npu"],
        )

    # --- filesystems ---------------------------------------------------
    def filesystems(self, *, full: bool = False) -> list[FilesystemInfo]:
        """Report the filesystems that matter for a conversion.

        Always: the system temporary directory, the managed cache/tmp directory,
        the artifacts directory and ``/``. With ``full`` also the staging and
        compile-cache directories, which may live elsewhere when the user has
        overridden ``data_dir`` or ``cache_dir``.
        """
        wanted: list[Path] = [system_temp_dir()]
        paths = self._resolved_paths()
        if paths is not None:
            wanted.extend([paths.tmp_dir, paths.artifacts_dir])
            if full:
                wanted.extend([paths.staging_dir, paths.compile_cache_dir])
        wanted.append(self._root)

        result: list[FilesystemInfo] = []
        seen: set[str] = set()
        for entry in wanted:
            info = _coerce_filesystem(self._filesystem, entry)
            key = str(info.path)
            if key in seen:
                continue
            seen.add(key)
            result.append(info)
        return result

    # --- devices -------------------------------------------------------
    def _resolve_devices(
        self,
        devices: Sequence[OpenVINODevice] | None,
        openvino_available: bool | None,
        openvino_import_error: str | None,
        notes: list[str],
    ) -> DeviceSnapshot:
        """Merge explicit arguments with the injected device source."""
        if devices is not None:
            return DeviceSnapshot(
                devices=tuple(devices),
                available=openvino_available if openvino_available is not None else True,
                import_error=openvino_import_error,
            )

        if self._device_source is not None:
            try:
                snapshot = self._device_source()
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                notes.append(f"The OpenVINO device probe failed: {message}")
                return DeviceSnapshot(
                    available=False,
                    import_error=openvino_import_error or message,
                )
            return DeviceSnapshot(
                devices=tuple(snapshot.devices),
                available=(
                    openvino_available if openvino_available is not None else snapshot.available
                ),
                import_error=openvino_import_error or snapshot.import_error,
            )

        notes.append(
            "No OpenVINO device source was wired into the hardware probe, so no "
            "accelerators are reported; this inventory describes the host only."
        )
        return DeviceSnapshot(
            available=bool(openvino_available),
            import_error=openvino_import_error,
        )

    # --- entry point ---------------------------------------------------
    def inventory(
        self,
        *,
        full: bool = False,
        devices: Sequence[OpenVINODevice] | None = None,
        openvino_available: bool | None = None,
        openvino_import_error: str | None = None,
        hf_auth: HuggingFaceAuthInfo | None = None,
    ) -> HardwareInventory:
        """Return a complete host snapshot. Never raises."""
        return self.inventory_with_notes(
            full=full,
            devices=devices,
            openvino_available=openvino_available,
            openvino_import_error=openvino_import_error,
            hf_auth=hf_auth,
        )[0]

    def inventory_with_notes(
        self,
        *,
        full: bool = False,
        devices: Sequence[OpenVINODevice] | None = None,
        openvino_available: bool | None = None,
        openvino_import_error: str | None = None,
        hf_auth: HuggingFaceAuthInfo | None = None,
    ) -> tuple[HardwareInventory, list[str]]:
        """Return the inventory together with any degradation notes.

        ``HardwareInventory`` has no ``notes`` field, so anything that could not
        be read is reported here instead of being silently dropped; ``doctor``
        renders these as ``unknown`` checks.
        """
        notes: list[str] = []

        host = self.host_info(notes)
        _, cpu_vendor = self.cpu_identity()
        memory = self.memory_info(notes)
        packages = self.packages()
        snapshot = self._resolve_devices(devices, openvino_available, openvino_import_error, notes)

        if host.platform.lower() != "linux":
            notes.append(
                f"{host.platform} is not a supported platform for yonga; host facts that "
                "come from /proc, /dev and the group database are unavailable or "
                "approximate here."
            )

        npu_device = next((d for d in snapshot.devices if d.kind is DeviceKind.NPU), None)
        driver_version = (
            _first_property(npu_device, "NPU_DRIVER_VERSION", "DRIVER_VERSION")
            if npu_device is not None
            else None
        )
        npu = self.npu_node(driver_version=driver_version, notes=notes)

        filesystems = self.filesystems(full=full)
        system_temp = filesystems[0] if filesystems else None
        managed_temp = _find_managed_temp(self._resolved_paths(), filesystems)

        tags = derive_device_tags(
            devices=snapshot.devices,
            host=host,
            cpu_vendor=cpu_vendor,
        )

        inventory = HardwareInventory(
            host=host,
            memory=memory,
            filesystems=filesystems,
            devices=list(snapshot.devices),
            openvino_available=snapshot.available,
            openvino_import_error=snapshot.import_error,
            npu=npu,
            packages=packages,
            device_tags=tags,
            system_temp=system_temp,
            managed_temp=managed_temp,
            hf_auth=hf_auth,
        )
        return inventory, notes


# --- module level helpers -----------------------------------------------------------


def _coerce_filesystem(probe: FilesystemProbe, path: Path) -> FilesystemInfo:
    """Call ``probe.usage`` and guarantee a :class:`FilesystemInfo` comes back."""
    try:
        value = probe.usage(path)
    except Exception:
        return FilesystemInfo(path=path)
    if isinstance(value, FilesystemInfo):
        return value
    try:
        return FilesystemInfo.model_validate(value)
    except Exception:
        return FilesystemInfo(path=path)


def _find_managed_temp(
    paths: Paths | None,
    filesystems: Sequence[FilesystemInfo],
) -> FilesystemInfo | None:
    """Pick the entry describing the managed cache/tmp directory."""
    if paths is None:
        return None
    wanted = str(paths.tmp_dir)
    return next((entry for entry in filesystems if str(entry.path) == wanted), None)


def _is_dir(path: Path) -> bool:
    """``Path.is_dir`` that never raises."""
    try:
        return path.is_dir()
    except OSError:  # pragma: no cover - permission denied on a parent
        return False


def _access(path: Path, mode: int) -> bool:
    """``os.access`` that never raises."""
    try:
        return os.access(path, mode)
    except (OSError, ValueError):  # pragma: no cover - exotic path
        return False


def _pick_accel_node(accel_dir: Path) -> Path | None:
    """Return ``accel0`` if present, otherwise the first ``accel*`` node."""
    try:
        candidates = sorted(
            entry for entry in accel_dir.iterdir() if entry.name.startswith("accel")
        )
    except OSError:
        return None
    if not candidates:
        return None
    for candidate in candidates:
        if candidate.name == "accel0":
            return candidate
    return candidates[0]


def _group_name(gid: int) -> str | None:
    """Resolve a numeric gid to a group name, or ``None`` off Unix."""
    try:
        import grp
    except ImportError:  # pragma: no cover - non-Unix platform
        return None
    try:
        return str(grp.getgrgid(gid).gr_name)
    except (KeyError, OSError, OverflowError):
        return None


def _user_group_names() -> list[str]:
    """Return the names of every group the current process belongs to."""
    try:
        gids = list(os.getgroups())
    except (AttributeError, OSError):  # pragma: no cover - non-Unix platform
        return []
    with contextlib.suppress(AttributeError):  # pragma: no cover - non-Unix platform
        gids.append(os.getgid())
    names: dict[str, None] = {}
    for gid in gids:
        name = _group_name(gid)
        if name is not None:
            names.setdefault(name, None)
    return sorted(names)


def _openvino_package_libs() -> Path | None:
    """Locate ``<openvino package>/libs`` without importing OpenVINO.

    ``find_spec`` on a top-level name runs the path finders only; it does not
    execute ``openvino/__init__.py``, which is what makes this safe to call from
    a probe that must work when the runtime is broken.
    """
    try:
        spec = importlib.util.find_spec("openvino")
    except (ImportError, ValueError, AttributeError):
        return None
    if spec is None:
        return None
    locations = list(spec.submodule_search_locations or [])
    if not locations:
        return None
    return Path(locations[0]) / "libs"
