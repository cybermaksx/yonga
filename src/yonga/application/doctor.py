"""The ``yonga doctor`` application service.

``doctor`` answers one question: *can this machine convert and run a model, and
if not, what exactly is missing?* It is strictly read-only. It never installs a
package, never runs ``sudo``, never changes a permission or a group, and never
touches the Hugging Face cache. Every finding is a typed
:class:`~yonga.domain.diagnostics.HealthCheck` carrying a status and, whenever
the status is not ``ok``, the remediation the user should apply themselves.

Two rules drive the design:

* **Doctor must never crash.** A diagnostic tool that dies while diagnosing is
  worthless, so every adapter call is wrapped and degrades to an ``unknown`` or
  ``error`` check. In particular a host with no OpenVINO installed still gets a
  complete report — it just exits with
  :attr:`~yonga.domain.diagnostics.ExitCode.ENVIRONMENT_UNUSABLE`.
* **A warning the managed temp directory already solves is never fatal.** yonga
  runs conversions in its own disk-backed temporary directory, so a 16 GiB
  ``/tmp`` tmpfs on a host with hundreds of free gigabytes on ``/`` is a warning
  with an explanation, not an error (``docs/CLI_SPEC.md`` section 3).

The service takes injected adapters only: it imports neither Rich nor OpenVINO
nor ``huggingface_hub``, so the whole report can be produced on a laptop with no
Intel accelerator by handing it fake probes.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from yonga.adapters.filesystem import format_bytes
from yonga.adapters.openvino_probe import (
    compiler_in_plugin,
    npu_compiler_version,
    npu_driver_version,
)
from yonga.application.progress import ProgressEvent, ProgressKind, ProgressSink, null_sink
from yonga.config.settings import Settings
from yonga.domain.compile_cache import CompileCacheInventory
from yonga.domain.diagnostics import (
    ExitCode,
    HealthCheck,
    HealthSection,
    Severity,
    worst_severity,
)
from yonga.domain.environments import ExportEnvironment
from yonga.domain.hardware import (
    FilesystemInfo,
    HardwareInventory,
    HuggingFaceAuthInfo,
    NPUNodeInfo,
    OpenVINODevice,
    normalise_device,
)
from yonga.util.redaction import redact_text
from yonga.util.versions import families_aligned, format_family, version_family

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yonga.adapters import HardwareProbe, HubClient, RuntimeProbe
    from yonga.registry.schema import Registry

__all__ = [
    "ABI_PACKAGES",
    "AUTH_SOURCE_KEY",
    "COMPILE_CACHE_WARN_BYTES",
    "COMPILE_CACHE_WARN_FLOOR_BYTES",
    "COMPILE_CACHE_WARN_FREE_FRACTION",
    "MANAGED_TEMP_NOTE",
    "OPENVINO_INSTALL_HINT",
    "REPORTED_PACKAGES",
    "DoctorReport",
    "DoctorService",
    "group_membership_remediation",
]

_GIB = 1024**3

#: The three packages that must share one major.minor release family (SPEC 5.3).
ABI_PACKAGES: tuple[str, ...] = ("openvino", "openvino-genai", "openvino-tokenizers")

#: ``(distribution name, label, severity when missing)`` for the OpenVINO section.
REPORTED_PACKAGES: tuple[tuple[str, str, Severity], ...] = (
    ("openvino", "Runtime", Severity.ERROR),
    ("openvino-genai", "GenAI", Severity.ERROR),
    ("openvino-tokenizers", "Tokenizers", Severity.ERROR),
    ("optimum-intel", "optimum-intel", Severity.WARNING),
    ("nncf", "nncf", Severity.WARNING),
    ("transformers", "transformers", Severity.WARNING),
    ("huggingface-hub", "huggingface-hub", Severity.ERROR),
)

#: The single sentence that keeps a small ``/tmp`` from ever being fatal.
MANAGED_TEMP_NOTE: str = (
    "yonga runs conversions in a managed, disk-backed temporary directory instead of the "
    "system temp filesystem, so a small tmpfs is a warning and never fatal."
)

#: How to obtain the OpenVINO runtime. Printed, never executed.
OPENVINO_INSTALL_HINT: str = "Install the runtime extras: pip install 'yonga[openvino]'"

#: How to obtain the conversion toolchain. Printed, never executed.
EXPORT_INSTALL_HINT: str = "Install the conversion extras: pip install 'yonga[export]'"

#: What happens without the conversion toolchain: conversion still works.
MANAGED_EXPORT_NOTE: str = (
    "Without it, 'prepare' converts in a managed export environment it builds under the "
    "cache directory on first use (about 3 GiB), leaving this environment untouched."
)

#: yonga changes nothing about the Python environment; say so next to every hint.
NEVER_INSTALLS_NOTE: str = (
    "yonga never installs, upgrades or removes a package in the environment it runs "
    "from: run the command yourself."
)

#: Below this much free space a filesystem is called out as an error, not a warning.
CRITICAL_FREE_BYTES: int = 2 * _GIB

#: Reclaimable compile cache (orphaned plus stale-runtime entries) at or above
#: this many bytes is a warning, whatever the free space.
COMPILE_CACHE_WARN_BYTES: int = 10 * _GIB

#: Reclaimable compile cache at or above this share of the free space on the
#: cache filesystem is a warning too: 3 GiB matters when 8 GiB are free.
COMPILE_CACHE_WARN_FREE_FRACTION: float = 0.25

#: ... but never below this many bytes, so a nearly full disk does not turn a
#: few megabytes of dead blobs into a warning.
COMPILE_CACHE_WARN_FLOOR_BYTES: int = 1 * _GIB

#: Hosts with less RAM than this cannot hold a mid-sized model plus its export.
LOW_MEMORY_BYTES: int = 8 * _GIB

#: JSON name for ``HuggingFaceAuthInfo.token_source``.
#:
#: The field only ever names *where* a credential was found (``"environment"``),
#: never the credential, but ``yonga.domain.artifacts.assert_no_secrets`` rejects
#: any key whose words include ``token`` or ``credential``. Renaming it keeps the
#: information and lets the doctor JSON pass the project's own secret-key gate.
AUTH_SOURCE_KEY: str = "auth_source"


def group_membership_remediation(group_name: str) -> list[str]:
    """Return the remediation for an accelerator node the user cannot open.

    The first line is plain English and names the group; the second is a
    *suggested* command the user runs themselves. ``doctor`` only ever prints
    it — it never invokes ``sudo`` and never changes group membership
    (SPEC 6.1, SPEC 18.5).
    """
    return [
        f"add your user to the '{group_name}' group, then log out and back in "
        "for the new membership to take effect",
        f"Suggested command, to run yourself: sudo usermod -aG {group_name} $USER",
        "yonga never runs sudo and never changes permissions or group membership.",
    ]


class DoctorReport(BaseModel):
    """The complete result of one ``doctor`` run.

    Carries the rendered-ready :class:`~yonga.domain.diagnostics.HealthSection`
    list, the raw :class:`~yonga.domain.hardware.HardwareInventory` it was
    derived from, the overall :class:`~yonga.domain.diagnostics.Severity` and the
    process exit code the CLI should use.
    """

    model_config = ConfigDict(extra="ignore")

    sections: list[HealthSection] = Field(default_factory=list)
    hardware: HardwareInventory = Field(default_factory=HardwareInventory)
    status: Severity = Severity.UNKNOWN
    runtime_usable: bool = False
    full: bool = False
    notes: list[str] = Field(default_factory=list)
    compile_cache: dict[str, object] | None = None
    """Size and reclaimable bytes of the compile cache, when it could be measured."""

    @property
    def checks(self) -> list[HealthCheck]:
        """Every check of every section, in report order."""
        return [check for section in self.sections for check in section.checks]

    @property
    def problems(self) -> list[HealthCheck]:
        """Every check whose status is not ``ok`` or ``info``."""
        return [
            check
            for check in self.checks
            if check.status in (Severity.WARNING, Severity.ERROR, Severity.UNKNOWN)
        ]

    def section(self, section_id: str) -> HealthSection | None:
        """Look one section up by id, or ``None`` when it was not produced."""
        return next((item for item in self.sections if item.id == section_id), None)

    def check(self, check_id: str) -> HealthCheck | None:
        """Look one check up by id across every section."""
        return next((item for item in self.checks if item.id == check_id), None)

    def exit_code(self) -> ExitCode:
        """Return the process exit code (``docs/CLI_SPEC.md`` section 3).

        ``0`` when a usable OpenVINO runtime and device exist and nothing was
        flagged, ``2`` when they exist but something needs attention, and ``10``
        when there is no usable OpenVINO runtime or device at all.
        """
        if not self.runtime_usable:
            return ExitCode.ENVIRONMENT_UNUSABLE
        if self.status in (Severity.OK, Severity.INFO):
            return ExitCode.SUCCESS
        return ExitCode.WARNINGS

    def to_json(self) -> dict[str, object]:
        """Return a JSON-safe dictionary (enums as values, paths as strings).

        The payload never contains a credential: authentication is reported as
        :class:`~yonga.domain.hardware.HuggingFaceAuthInfo`, which forbids extra
        fields and only records *where* a token was found. That field is renamed
        to :data:`AUTH_SOURCE_KEY` here so the whole document also passes
        :func:`yonga.domain.artifacts.assert_no_secrets`, which rejects any key
        containing the word ``token`` or ``credential`` regardless of its value.
        """
        payload = dict(self.model_dump(mode="json"))
        hardware = payload.get("hardware")
        if isinstance(hardware, dict):
            auth = hardware.get("hf_auth")
            if isinstance(auth, dict) and "token_source" in auth:
                auth[AUTH_SOURCE_KEY] = auth.pop("token_source")
        payload["exit_code"] = int(self.exit_code())
        return payload


class DoctorService:
    """Builds a :class:`DoctorReport` from injected, read-only probes.

    ``hardware_probe`` supplies the host inventory, ``runtime_probe`` the
    authoritative OpenVINO device list, ``hub_client`` the Hugging Face
    authentication state, and ``registry`` (reported only with ``full=True``) the
    compatibility rules. All but the hardware probe may be ``None`` so that a
    partially installed host can still be diagnosed.
    """

    def __init__(
        self,
        hardware_probe: HardwareProbe,
        runtime_probe: RuntimeProbe | None = None,
        hub_client: HubClient | None = None,
        registry: Registry | None = None,
        settings: Settings | None = None,
        *,
        progress: ProgressSink | None = None,
        export_environments: Callable[[], Sequence[ExportEnvironment]] | None = None,
        compile_cache: Callable[[], CompileCacheInventory] | None = None,
    ) -> None:
        """Store the injected adapters. Nothing is probed until :meth:`run`.

        ``export_environments`` lists the managed export environments, so the
        report can say how much disk they hold. ``compile_cache`` measures the
        OpenVINO compile cache, so it can say how much of it is reclaimable.
        """
        self._hardware = hardware_probe
        self._runtime = runtime_probe
        self._hub = hub_client
        self._registry = registry
        self._settings = settings if settings is not None else Settings()
        self._progress: ProgressSink = progress if progress is not None else null_sink()
        self._export_environments = export_environments
        self._compile_cache = compile_cache
        self._compile_cache_summary: dict[str, object] | None = None

    # --- entry point ---------------------------------------------------

    def run(self, *, full: bool = False) -> DoctorReport:
        """Collect every health section and return the report. Never raises."""
        self._compile_cache_summary = None
        notes: list[str] = []
        inventory = self._inventory(full=full, notes=notes)
        auth = self._auth_info(inventory, notes)
        if auth is not None and inventory.hf_auth is None:
            inventory = inventory.model_copy(update={"hf_auth": auth})

        runtime_names, runtime_error = self._runtime_devices(notes)
        devices_section, runtime_usable = self._devices_section(
            inventory, runtime_names, runtime_error
        )

        sections = [
            self._emit(self._host_section(inventory)),
            self._emit(self._openvino_section(inventory)),
            self._emit(devices_section),
            self._emit(self._npu_section(inventory)),
            self._emit(self._storage_section(inventory, full=full)),
            self._emit(self._huggingface_section(inventory, auth)),
        ]
        if full:
            sections.append(self._emit(self._registry_section()))

        return DoctorReport(
            sections=sections,
            hardware=inventory,
            status=worst_severity(section.status for section in sections),
            runtime_usable=runtime_usable,
            full=full,
            notes=notes,
            compile_cache=self._compile_cache_summary,
        )

    # --- adapter calls -------------------------------------------------

    def _inventory(self, *, full: bool, notes: list[str]) -> HardwareInventory:
        """Call the hardware probe, degrading to an empty inventory on failure."""
        try:
            return self._hardware.inventory(full=full)
        except Exception as exc:  # pragma: no cover - a probe that raises is a bug
            notes.append(f"The hardware probe failed: {_describe(exc)}")
            return HardwareInventory()

    def _auth_info(
        self, inventory: HardwareInventory, notes: list[str]
    ) -> HuggingFaceAuthInfo | None:
        """Ask the Hub client for authentication state, falling back to the inventory."""
        if self._hub is not None:
            try:
                return self._hub.auth_info()
            except Exception as exc:
                notes.append(
                    f"The Hugging Face client could not report its state: {_describe(exc)}"
                )
        return inventory.hf_auth

    def _runtime_devices(self, notes: list[str]) -> tuple[list[str], str | None]:
        """Return the runtime's device names and the error that hid them, if any."""
        if self._runtime is None:
            return [], None
        try:
            names = [str(name) for name in self._runtime.available_devices()]
        except Exception as exc:
            message = _describe(exc)
            notes.append(f"The OpenVINO device probe failed: {message}")
            return [], message
        return names, None

    def _emit(self, section: HealthSection) -> HealthSection:
        """Publish one finished section on the progress sink and return it."""
        self._progress.emit(
            ProgressEvent(
                kind=ProgressKind.PROBE,
                message=section.title,
                status=section.status.value,
                severity=section.status,
                data={"section": section.id, "checks": len(section.checks)},
            )
        )
        return section

    # --- host ----------------------------------------------------------

    def _host_section(self, inventory: HardwareInventory) -> HealthSection:
        """OS, kernel, Python, CPU, RAM and swap."""
        host = inventory.host
        memory = inventory.memory
        checks = [
            _fact(
                "host.os",
                "OS",
                host.distro_pretty or host.platform,
                detail=_join(host.distro_id, host.distro_version),
                unknown_remediation=[
                    "/etc/os-release could not be read; distribution-specific advice "
                    "in this report may not apply.",
                ],
            ),
            _fact(
                "host.kernel",
                "Kernel",
                host.kernel,
                detail=host.machine if host.machine != "unknown" else None,
                unknown_remediation=["The kernel release could not be read from this host."],
            ),
            _fact(
                "host.python",
                "Python",
                host.python_version,
                unknown_remediation=["The running Python version could not be determined."],
            ),
            _fact(
                "host.cpu",
                "CPU",
                host.cpu_model,
                detail=(f"{host.cpu_count} logical CPUs" if host.cpu_count else None),
                unknown_remediation=[
                    "/proc/cpuinfo could not be read; CPU-specific device tags are unavailable.",
                ],
            ),
            self._memory_check(inventory),
            _check(
                "host.swap",
                "Swap",
                Severity.OK,
                value=(
                    format_bytes(memory.swap_total_bytes) if memory.swap_total_bytes else "none"
                ),
                detail=(
                    f"{format_bytes(memory.swap_free_bytes)} free"
                    if memory.swap_total_bytes
                    else "no swap configured"
                ),
            ),
        ]
        if host.platform and host.platform.lower() not in ("linux", "unknown"):
            checks.append(
                _check(
                    "host.platform",
                    "Platform",
                    Severity.WARNING,
                    value=host.platform,
                    detail="yonga targets Linux; /dev, /proc and group facts are approximate here.",
                    remediation=[
                        "Run yonga on a Linux host to get accurate NPU and filesystem facts.",
                    ],
                )
            )
        return HealthSection(id="host", title="Host", checks=_finalise(checks))

    def _memory_check(self, inventory: HardwareInventory) -> HealthCheck:
        """RAM total and availability, warning on a host too small to convert on."""
        memory = inventory.memory
        if not memory.total_bytes:
            return _check(
                "host.ram",
                "RAM",
                Severity.UNKNOWN,
                value="unknown",
                remediation=["Total system memory could not be read from this host."],
            )
        detail = f"{format_bytes(memory.available_bytes)} available"
        if memory.total_bytes < LOW_MEMORY_BYTES:
            return _check(
                "host.ram",
                "RAM",
                Severity.WARNING,
                value=format_bytes(memory.total_bytes),
                detail=detail,
                remediation=[
                    f"This host has less than {format_bytes(LOW_MEMORY_BYTES)} of RAM; "
                    "prefer small models and int4 weight compression.",
                    "Close other applications before converting a model.",
                ],
            )
        return _check(
            "host.ram",
            "RAM",
            Severity.OK,
            value=format_bytes(memory.total_bytes),
            detail=detail,
        )

    # --- openvino packages ---------------------------------------------

    def _openvino_section(self, inventory: HardwareInventory) -> HealthSection:
        """Package versions plus the OpenVINO/GenAI/Tokenizers ABI family check."""
        checks = [
            _package_check(inventory, dist_name, label, missing)
            for dist_name, label, missing in REPORTED_PACKAGES
        ]
        checks.append(_abi_family_check(inventory))
        return HealthSection(id="openvino", title="OpenVINO", checks=_finalise(checks))

    # --- devices --------------------------------------------------------

    def _devices_section(
        self,
        inventory: HardwareInventory,
        runtime_names: Sequence[str],
        runtime_error: str | None,
    ) -> tuple[HealthSection, bool]:
        """One check per OpenVINO device, or one error explaining their absence."""
        importable = inventory.openvino_available or bool(runtime_names)
        names = _merge_device_names(runtime_names, inventory)
        checks: list[HealthCheck] = []

        if not importable:
            detail = redact_text(inventory.openvino_import_error or runtime_error or "")
            checks.append(
                _check(
                    "devices.runtime",
                    "OpenVINO runtime",
                    Severity.ERROR,
                    value="not importable",
                    detail=detail or "openvino could not be imported.",
                    remediation=[
                        OPENVINO_INSTALL_HINT,
                        "Confirm the active virtual environment is the one yonga runs from.",
                        NEVER_INSTALLS_NOTE,
                    ],
                )
            )
            return HealthSection(id="devices", title="Devices", checks=_finalise(checks)), False

        if runtime_error:
            checks.append(
                _check(
                    "devices.probe",
                    "Device probe",
                    Severity.WARNING,
                    value="failed",
                    detail=redact_text(runtime_error),
                    remediation=[
                        "The device list below comes from the cached host inventory, "
                        "not from a live openvino.Core() query.",
                        OPENVINO_INSTALL_HINT,
                    ],
                )
            )

        if not names:
            checks.append(
                _check(
                    "devices.none",
                    "OpenVINO devices",
                    Severity.ERROR,
                    value="none available",
                    detail="openvino.Core().available_devices returned an empty list.",
                    remediation=[
                        OPENVINO_INSTALL_HINT,
                        "Check that the Intel GPU/NPU kernel drivers are loaded "
                        "(see the NPU section below).",
                    ],
                )
            )
            return HealthSection(id="devices", title="Devices", checks=_finalise(checks)), False

        checks.extend(_device_check(name, inventory.device(name)) for name in names)
        return HealthSection(id="devices", title="Devices", checks=_finalise(checks)), True

    # --- npu ------------------------------------------------------------

    def _npu_section(self, inventory: HardwareInventory) -> HealthSection:
        """``/dev/accel`` accessibility, driver, compiler and compiler libraries."""
        node = inventory.npu
        device = inventory.device("NPU")
        properties: Mapping[str, str] = device.properties if device is not None else {}
        present = device is not None or (node is not None and node.node_exists)

        if not present:
            detail = (
                "OpenVINO reports no NPU device and no /dev/accel node was found."
                if node is not None
                else "The NPU was not probed on this host."
            )
            return HealthSection(
                id="npu",
                title="NPU",
                checks=[
                    _check("npu.present", "NPU", Severity.INFO, value="not present", detail=detail)
                ],
            )

        checks = [_npu_node_check(node)]
        group_check = _npu_group_check(node)
        if group_check is not None:
            checks.append(group_check)
        checks.extend(_npu_compiler_checks(node, properties))
        return HealthSection(id="npu", title="NPU", checks=_finalise(checks))

    # --- storage ---------------------------------------------------------

    def _storage_section(self, inventory: HardwareInventory, *, full: bool) -> HealthSection:
        """Free space on the cache/artifact filesystems and the managed-temp note."""
        paths = self._settings.paths
        floor = max(self._settings.temp_min_bytes, CRITICAL_FREE_BYTES)
        checks = [
            _space_check(
                "storage.artifacts",
                "artifact filesystem",
                _filesystem_for(inventory, paths.artifacts_dir),
                paths.artifacts_dir,
                floor,
            ),
            _space_check(
                "storage.cache",
                "cache filesystem",
                inventory.managed_temp or _filesystem_for(inventory, paths.tmp_dir),
                paths.tmp_dir,
                floor,
            ),
            _system_temp_check(inventory.system_temp, floor),
            _managed_temp_check(inventory.managed_temp, paths.tmp_dir),
        ]
        environments = self._environment_check(paths.envs_dir)
        if environments is not None:
            checks.append(environments)
        compile_cache = self._compile_cache_check(
            _filesystem_for(inventory, paths.compile_cache_dir) or inventory.managed_temp
        )
        if compile_cache is not None:
            checks.append(compile_cache)
        if full:
            checks.extend(_extra_filesystem_checks(inventory, paths.artifacts_dir, paths.tmp_dir))
        return HealthSection(id="storage", title="Storage", checks=_finalise(checks))

    def _environment_check(self, root: Path) -> HealthCheck | None:
        """How many managed export environments exist and the disk they use."""
        if self._export_environments is None:
            return None
        try:
            environments = list(self._export_environments())
        except Exception:
            # Reporting disk use must never make doctor fail.
            return None
        total = sum(environment.size_bytes for environment in environments)
        value = (
            f"{len(environments)} built, {format_bytes(total)}" if environments else "none built"
        )
        return HealthCheck(
            id="storage.export_environments",
            label="export environments",
            status=Severity.OK,
            value=value,
            detail=f"{root} — isolated environments exports run in; `yonga env list`",
        )

    def _compile_cache_check(self, filesystem: FilesystemInfo | None) -> HealthCheck | None:
        """The compile cache's size, and a warning when much of it is dead weight.

        Reclaimable means orphaned (the artifact is gone) or stale-runtime (built
        by another OpenVINO release, so never reused). It is a warning at
        :data:`COMPILE_CACHE_WARN_BYTES` or more, or at
        :data:`COMPILE_CACHE_WARN_FREE_FRACTION` of the free space on the cache
        filesystem, never below :data:`COMPILE_CACHE_WARN_FLOOR_BYTES`.
        """
        if self._compile_cache is None:
            return None
        try:
            cache = self._compile_cache()
        except Exception:
            # Reporting disk use must never make doctor fail.
            return None
        free = None
        if filesystem is not None:
            free = filesystem.available_bytes or filesystem.free_bytes
        total = cache.total_bytes
        reclaimable = cache.reclaimable_bytes
        threshold = COMPILE_CACHE_WARN_BYTES
        if free is not None:
            threshold = min(
                threshold,
                max(COMPILE_CACHE_WARN_FLOOR_BYTES, int(free * COMPILE_CACHE_WARN_FREE_FRACTION)),
            )
        self._compile_cache_summary = {
            "path": str(cache.root),
            "entries": len(cache.entries),
            "total_bytes": total,
            "reclaimable_bytes": reclaimable,
            "warn_threshold_bytes": threshold,
            "installed_openvino": cache.installed_openvino,
        }
        if not cache.entries:
            value = "empty"
        else:
            value = f"{len(cache.entries)} entries, {format_bytes(total)}"
            if reclaimable:
                value += f", {format_bytes(reclaimable)} reclaimable"
        detail = f"{cache.root} — compiled blobs; `yonga cache list`"
        if reclaimable and reclaimable >= threshold:
            return HealthCheck(
                id="storage.compile_cache",
                label="compile cache",
                status=Severity.WARNING,
                value=value,
                detail=detail,
                remediation=[
                    f"{format_bytes(reclaimable)} of the compile cache belongs to removed "
                    "artifacts or to another OpenVINO release and will never be reused.",
                    "Reclaim it with: yonga cache prune",
                    "Preview what would go first with: yonga cache prune --dry-run",
                ],
            )
        return HealthCheck(
            id="storage.compile_cache",
            label="compile cache",
            status=Severity.OK,
            value=value,
            detail=detail,
        )

    # --- hugging face -----------------------------------------------------

    def _huggingface_section(
        self,
        inventory: HardwareInventory,
        auth: HuggingFaceAuthInfo | None,
    ) -> HealthSection:
        """Authentication state, cache directory and offline mode — never a token."""
        state = auth if auth is not None else inventory.hf_auth
        if state is None:
            return HealthSection(
                id="huggingface",
                title="Hugging Face",
                checks=_finalise(
                    [
                        _check(
                            "huggingface.auth",
                            "Authentication",
                            Severity.UNKNOWN,
                            value="unknown",
                            remediation=[
                                "No Hugging Face client was wired into doctor, so the "
                                "authentication state could not be read.",
                            ],
                        )
                    ]
                ),
            )

        offline = state.offline or self._settings.offline
        checks = [
            _hf_auth_check(state),
            _fact(
                "huggingface.cache",
                "Cache dir",
                str(state.cache_dir) if state.cache_dir is not None else None,
                unknown_remediation=[
                    "The Hugging Face cache directory could not be determined; "
                    "set HF_HOME to pin it.",
                ],
            ),
            _check(
                "huggingface.offline",
                "Offline mode",
                Severity.INFO if offline else Severity.OK,
                value="enabled" if offline else "disabled",
                detail=(
                    "Only models already in the local cache can be used."
                    if offline
                    else "The Hub is reachable for metadata and downloads."
                ),
            ),
        ]
        return HealthSection(id="huggingface", title="Hugging Face", checks=_finalise(checks))

    # --- registry ---------------------------------------------------------

    def _registry_section(self) -> HealthSection:
        """Compatibility-registry size, layers and validation problems (``--full``)."""
        registry = self._registry
        if registry is None:
            return HealthSection(
                id="registry",
                title="Registry",
                checks=[
                    _check(
                        "registry.rules",
                        "Rules",
                        Severity.INFO,
                        value="not loaded",
                        detail="No compatibility registry was wired into doctor.",
                    )
                ],
            )

        from yonga.registry.loader import validate_registry
        from yonga.repairs import known_implementations

        rules = registry.all_rules()
        checks = [
            _check(
                "registry.rules",
                "Rules",
                Severity.OK,
                value=f"{len(rules)} rules",
                detail=f"schema v{registry.schema_version}",
            ),
            _check(
                "registry.layers",
                "Layers",
                Severity.OK if registry.layers else Severity.INFO,
                value=str(len(registry.layers)),
                detail="; ".join(registry.layers) or "built-in only",
            ),
        ]
        try:
            problems = validate_registry(registry, known_implementations=known_implementations())
        except Exception as exc:  # pragma: no cover - validation is pure
            problems = [_describe(exc)]
        checks.append(
            _check(
                "registry.validation",
                "Validation",
                Severity.OK if not problems else Severity.WARNING,
                value="clean" if not problems else f"{len(problems)} problems",
                detail="; ".join(problems[:3]) or None,
                remediation=(
                    []
                    if not problems
                    else [
                        "Fix or remove the offending override in "
                        f"{self._settings.paths.registry_overrides_dir}.",
                        *problems[:5],
                    ]
                ),
            )
        )
        return HealthSection(id="registry", title="Registry", checks=_finalise(checks))


# --- check builders -------------------------------------------------------------------


def _check(
    check_id: str,
    label: str,
    status: Severity,
    *,
    value: str | None = None,
    detail: str | None = None,
    remediation: Sequence[str] = (),
) -> HealthCheck:
    """Build one :class:`~yonga.domain.diagnostics.HealthCheck`."""
    return HealthCheck(
        id=check_id,
        label=label,
        status=status,
        value=value,
        detail=detail,
        remediation=list(remediation),
    )


def _fact(
    check_id: str,
    label: str,
    value: str | None,
    *,
    detail: str | None = None,
    unknown_remediation: Sequence[str] = (),
) -> HealthCheck:
    """Build a check that is ``ok`` when the fact is known and ``unknown`` otherwise."""
    cleaned = (value or "").strip()
    if not cleaned or cleaned.lower() == "unknown":
        return _check(
            check_id,
            label,
            Severity.UNKNOWN,
            value="unknown",
            detail=detail,
            remediation=unknown_remediation,
        )
    return _check(check_id, label, Severity.OK, value=cleaned, detail=detail)


def _finalise(checks: Sequence[HealthCheck]) -> list[HealthCheck]:
    """Guarantee the report's invariant: nothing below ``ok`` lacks remediation."""
    result: list[HealthCheck] = []
    for check in checks:
        needs_help = check.status in (Severity.WARNING, Severity.ERROR, Severity.UNKNOWN)
        if needs_help and not check.remediation:
            result.append(
                check.model_copy(
                    update={
                        "remediation": [
                            f"{check.label} could not be confirmed; "
                            "run `yonga doctor --full --json` and include the output "
                            "when reporting this.",
                        ]
                    }
                )
            )
            continue
        result.append(check)
    return result


def _package_check(
    inventory: HardwareInventory,
    dist_name: str,
    label: str,
    missing_status: Severity,
) -> HealthCheck:
    """Report one distribution's installed version and importability."""
    check_id = f"openvino.{dist_name.replace('-', '_')}"
    package = inventory.package(dist_name)
    hint = OPENVINO_INSTALL_HINT if dist_name in ABI_PACKAGES else EXPORT_INSTALL_HINT
    if package is None or package.version is None:
        if dist_name in ABI_PACKAGES:
            remediation = [hint, NEVER_INSTALLS_NOTE]
        elif os.environ.get("APPIMAGE"):
            # Nothing can be installed into an AppImage; the managed environment is the way.
            remediation = [MANAGED_EXPORT_NOTE]
        else:
            remediation = [MANAGED_EXPORT_NOTE, hint, NEVER_INSTALLS_NOTE]
        return _check(
            check_id,
            label,
            missing_status,
            value="not installed",
            detail=f"{dist_name} was not found in this environment.",
            remediation=remediation,
        )
    if not package.importable:
        return _check(
            check_id,
            label,
            Severity.WARNING,
            value=package.version,
            detail=redact_text(package.import_error or f"{dist_name} could not be imported."),
            remediation=[
                f"{dist_name} is installed but not importable; reinstall it.",
                hint,
                NEVER_INSTALLS_NOTE,
            ],
        )
    return _check(check_id, label, Severity.OK, value=package.version)


def _abi_family_check(inventory: HardwareInventory) -> HealthCheck:
    """Compare the OpenVINO/GenAI/Tokenizers release families (SPEC 5.3).

    Reports the mismatch and how to fix it; it never installs anything.
    """
    versions: dict[str, str | None] = {}
    for dist_name in ABI_PACKAGES:
        package = inventory.package(dist_name)
        versions[dist_name] = package.version if package is not None else None

    known = {name: value for name, value in versions.items() if value}
    if len(known) < 2:
        return _check(
            "openvino.abi_family",
            "ABI family",
            Severity.UNKNOWN,
            value="not comparable",
            detail="Fewer than two of openvino, openvino-genai and openvino-tokenizers "
            "are installed.",
            remediation=[OPENVINO_INSTALL_HINT, NEVER_INSTALLS_NOTE],
        )

    listing = ", ".join(f"{name} {value}" for name, value in known.items())
    if families_aligned(versions):
        family = format_family(version_family(known[next(iter(known))]))
        return _check(
            "openvino.abi_family",
            "ABI family",
            Severity.OK,
            value=f"aligned ({family})",
            detail=listing,
        )

    target = format_family(version_family(versions.get("openvino") or next(iter(known.values()))))
    pins = " ".join(f"'{name}=={target}.*'" for name in ABI_PACKAGES)
    return _check(
        "openvino.abi_family",
        "ABI family",
        Severity.WARNING,
        value="mismatch",
        detail=listing,
        remediation=[
            "openvino, openvino-genai and openvino-tokenizers must come from one "
            f"release family; these do not: {listing}.",
            f"Install one family yourself, for example: pip install {pins}",
            NEVER_INSTALLS_NOTE,
        ],
    )


def _merge_device_names(runtime_names: Sequence[str], inventory: HardwareInventory) -> list[str]:
    """Merge the live device list with the inventory's, preserving report order."""
    merged: list[str] = []
    for name in runtime_names:
        normalised = normalise_device(name)
        if normalised and normalised not in merged:
            merged.append(normalised)
    for name in inventory.available_device_names:
        if name not in merged:
            merged.append(name)
    return merged


def _device_check(name: str, device: OpenVINODevice | None) -> HealthCheck:
    """Report one OpenVINO device and its ``FULL_DEVICE_NAME``."""
    check_id = f"devices.{name.lower()}"
    full_name = (device.full_name if device is not None else "").strip()
    architecture = device.architecture if device is not None else None
    if not full_name or full_name.lower() == "unknown":
        return _check(
            check_id,
            name,
            Severity.UNKNOWN,
            value="available",
            detail="FULL_DEVICE_NAME could not be read from this plugin.",
            remediation=[
                f"{name} is usable, but its full device name is unavailable; "
                "this is cosmetic and does not block conversion.",
            ],
        )
    return _check(check_id, name, Severity.OK, value=full_name, detail=architecture)


def _npu_node_check(node: NPUNodeInfo | None) -> HealthCheck:
    """Report ``/dev/accel/accel0`` existence and accessibility."""
    if node is None:
        return _check(
            "npu.node",
            "/dev/accel",
            Severity.UNKNOWN,
            value="not probed",
            remediation=["The accelerator node was not inspected on this host."],
        )
    path = str(node.node_path) if node.node_path is not None else "/dev/accel/accel0"
    if not node.node_exists:
        return _check(
            "npu.node",
            "/dev/accel",
            Severity.WARNING,
            value="missing",
            detail=node.notes[0] if node.notes else f"{path} does not exist.",
            remediation=[
                "Load the NPU kernel driver (the intel_vpu module) or install the "
                "intel-npu driver package for your distribution.",
                "yonga changes nothing about the kernel: run any module or package "
                "command yourself.",
            ],
        )
    if node.accessible:
        return _check(
            "npu.node",
            path,
            Severity.OK,
            value="accessible",
            detail=(f"group {node.group_name}" if node.group_name else None),
        )
    mode = "read" if node.readable else ("write" if node.writable else "read/write")
    if node.group_name:
        return _check(
            "npu.node",
            path,
            Severity.WARNING,
            value="not accessible",
            detail=(
                f"{path} is owned by group '{node.group_name}' and the current user "
                f"{'is' if node.user_in_group else 'is not'} a member; "
                f"{mode} access is denied."
            ),
            remediation=group_membership_remediation(node.group_name),
        )
    return _check(
        "npu.node",
        path,
        Severity.WARNING,
        value="not accessible",
        detail=f"{mode} access to {path} is denied and its owning group is unknown.",
        remediation=[
            f"Inspect the node's owner with: ls -l {path}",
            "Then add your user to the group that owns it and log out and back in.",
            "yonga never runs sudo and never changes permissions or group membership.",
        ],
    )


def _npu_group_check(node: NPUNodeInfo | None) -> HealthCheck | None:
    """Report the owning group of the accelerator node and the user's membership."""
    if node is None or not node.group_name:
        return None
    if node.user_in_group:
        return _check(
            "npu.group",
            "Group",
            Severity.OK,
            value=node.group_name,
            detail="the current user is a member",
        )
    return _check(
        "npu.group",
        "Group",
        Severity.WARNING,
        value=node.group_name,
        detail=f"the current user is not a member of '{node.group_name}'",
        remediation=group_membership_remediation(node.group_name),
    )


def _npu_compiler_checks(
    node: NPUNodeInfo | None,
    properties: Mapping[str, str],
) -> list[HealthCheck]:
    """Report the NPU driver version, compiler version and compiler libraries."""
    driver = npu_driver_version(properties) or (node.driver_version if node else None)
    compiler = npu_compiler_version(properties)
    in_plugin = compiler_in_plugin(properties)
    libraries = list(node.compiler_libraries) if node is not None else []

    checks = [
        _fact(
            "npu.driver",
            "driver",
            driver,
            unknown_remediation=[
                "NPU_DRIVER_VERSION could not be read; the NPU still works if the "
                "device appears in the Devices section.",
            ],
        ),
        _fact(
            "npu.compiler_version",
            "compiler",
            compiler,
            unknown_remediation=[
                "NPU_COMPILER_VERSION could not be read; this is informational only.",
            ],
        ),
    ]

    if in_plugin is None:
        checks.append(
            _check(
                "npu.compiler_in_plugin",
                "compiler-in-plugin",
                Severity.UNKNOWN,
                value="unknown",
                remediation=[
                    "NPU_COMPILER_TYPE could not be read; compilation may still succeed.",
                ],
            )
        )
    else:
        checks.append(
            _check(
                "npu.compiler_in_plugin",
                "compiler-in-plugin",
                Severity.OK if in_plugin else Severity.INFO,
                value="available" if in_plugin else "external",
                detail=(
                    "the NPU plugin compiles graphs itself"
                    if in_plugin
                    else "the plugin defers to a system compiler library"
                ),
            )
        )

    if libraries:
        checks.append(
            _check(
                "npu.compiler_libraries",
                "compiler libraries",
                Severity.OK,
                value=f"{len(libraries)} found",
                detail="; ".join(libraries[:3]),
            )
        )
    elif in_plugin:
        checks.append(
            _check(
                "npu.compiler_libraries",
                "compiler libraries",
                Severity.INFO,
                value="none found",
                detail="not needed: the NPU plugin ships its own compiler.",
            )
        )
    else:
        checks.append(
            _check(
                "npu.compiler_libraries",
                "compiler libraries",
                Severity.WARNING,
                value="none found",
                detail="no libopenvino_intel_npu_compiler*.so was found in the "
                "searched library directories.",
                remediation=[
                    "Install your distribution's Intel NPU driver/compiler package "
                    "(it provides libopenvino_intel_npu_compiler.so).",
                    "NPU compilation may still work if the plugin carries its own compiler.",
                ],
            )
        )
    return checks


def _space_check(
    check_id: str,
    label: str,
    info: FilesystemInfo | None,
    path: Path,
    floor: int,
) -> HealthCheck:
    """Report free space on one filesystem, warning below ``floor``."""
    if info is None:
        return _check(
            check_id,
            label,
            Severity.UNKNOWN,
            value="unknown",
            detail=str(path),
            remediation=[f"The filesystem holding {path} could not be measured."],
        )
    free = info.available_bytes or info.free_bytes
    detail = f"{info.fstype} at {info.mount_point or info.path}"
    if free < CRITICAL_FREE_BYTES:
        return _check(
            check_id,
            label,
            Severity.ERROR,
            value=f"{format_bytes(free)} free",
            detail=detail,
            remediation=[
                f"Less than {format_bytes(CRITICAL_FREE_BYTES)} is free on {info.path}; "
                "free space or point cache_dir/data_dir at a larger filesystem.",
                "Remove artifacts you no longer need with: yonga remove ARTIFACT",
            ],
        )
    if free < floor:
        return _check(
            check_id,
            label,
            Severity.WARNING,
            value=f"{format_bytes(free)} free",
            detail=detail,
            remediation=[
                f"A conversion needs roughly {format_bytes(floor)} of headroom; "
                f"only {format_bytes(free)} is free on {info.path}.",
                "Point cache_dir or data_dir at a larger filesystem in the config file.",
            ],
        )
    return _check(check_id, label, Severity.OK, value=f"{format_bytes(free)} free", detail=detail)


def _system_temp_check(info: FilesystemInfo | None, floor: int) -> HealthCheck:
    """Report the system temp filesystem *with its type*; a small tmpfs is not fatal."""
    if info is None:
        return _check(
            "storage.system_temp",
            "system temp",
            Severity.UNKNOWN,
            value="unknown",
            remediation=[
                "The system temporary filesystem could not be measured.",
                MANAGED_TEMP_NOTE,
            ],
        )
    value = f"{format_bytes(info.total_bytes)} {info.fstype}"
    detail = f"{format_bytes(info.available_bytes or info.free_bytes)} free at {info.path}"
    if info.is_ram_backed and info.total_bytes < floor:
        return _check(
            "storage.system_temp",
            "system temp",
            Severity.WARNING,
            value=value,
            detail=detail,
            remediation=[
                MANAGED_TEMP_NOTE,
                f"{info.path} is RAM-backed and smaller than the "
                f"{format_bytes(floor)} a conversion can need, so yonga does not use it.",
                "No action is required. Do not enlarge the tmpfs for yonga's sake; "
                "other tools that honour TMPDIR may still need it.",
            ],
        )
    return _check("storage.system_temp", "system temp", Severity.OK, value=value, detail=detail)


def _managed_temp_check(info: FilesystemInfo | None, path: Path) -> HealthCheck:
    """State plainly that yonga converts in a managed, disk-backed temp directory."""
    if info is None:
        return _check(
            "storage.managed_temp",
            "managed temp",
            Severity.UNKNOWN,
            value="unknown",
            detail=str(path),
            remediation=[
                MANAGED_TEMP_NOTE,
                f"The filesystem holding {path} could not be measured.",
            ],
        )
    if info.is_ram_backed:
        return _check(
            "storage.managed_temp",
            "managed temp",
            Severity.WARNING,
            value=f"RAM-backed ({info.fstype})",
            detail=f"{path} — {MANAGED_TEMP_NOTE}",
            remediation=[
                f"{path} is on a RAM-backed filesystem, which defeats the managed "
                "temp directory; set cache_dir to a path on persistent storage.",
            ],
        )
    return _check(
        "storage.managed_temp",
        "managed temp",
        Severity.OK,
        value=f"disk-backed ({info.fstype})",
        detail=f"{path} — {MANAGED_TEMP_NOTE}",
    )


def _extra_filesystem_checks(
    inventory: HardwareInventory,
    artifacts_dir: Path,
    tmp_dir: Path,
) -> list[HealthCheck]:
    """``--full`` only: every other filesystem the probe measured."""
    covered = {str(artifacts_dir), str(tmp_dir)}
    if inventory.system_temp is not None:
        covered.add(str(inventory.system_temp.path))
    checks: list[HealthCheck] = []
    for index, info in enumerate(inventory.filesystems):
        if str(info.path) in covered:
            continue
        checks.append(
            _check(
                f"storage.other.{index}",
                str(info.path),
                Severity.OK,
                value=f"{format_bytes(info.available_bytes or info.free_bytes)} free",
                detail=f"{info.fstype} at {info.mount_point or info.path}",
            )
        )
    return checks


def _hf_auth_check(state: HuggingFaceAuthInfo) -> HealthCheck:
    """Report the Hugging Face authentication state without revealing the token."""
    if not state.logged_in:
        return _check(
            "huggingface.auth",
            "Authentication",
            Severity.INFO,
            value="anonymous",
            detail="Public repositories work; gated or private ones need `hf auth login`.",
        )
    return _check(
        "huggingface.auth",
        "Authentication",
        Severity.OK,
        value=f"signed in as {state.user}" if state.user else "signed in",
        detail=(f"credential source: {state.token_source}" if state.token_source else None),
    )


# --- small helpers --------------------------------------------------------------------


def _filesystem_for(inventory: HardwareInventory, path: Path) -> FilesystemInfo | None:
    """Find the measured filesystem for ``path``: exact probe first, then mount point."""
    wanted = str(path)
    for entry in inventory.filesystems:
        if str(entry.path) == wanted:
            return entry
    best: FilesystemInfo | None = None
    for entry in inventory.filesystems:
        mount = entry.mount_point
        if mount is None or not path.is_relative_to(mount):
            continue
        if best is None or best.mount_point is None or len(str(mount)) > len(str(best.mount_point)):
            best = entry
    return best


def _describe(error: BaseException) -> str:
    """Render an exception as one redacted line."""
    return redact_text(f"{type(error).__name__}: {error}")


def _join(*parts: str | None) -> str | None:
    """Join the non-empty parts with a space, or return ``None``."""
    joined = " ".join(part for part in parts if part)
    return joined or None
