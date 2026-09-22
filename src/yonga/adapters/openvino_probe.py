"""OpenVINO runtime adapter: device inventory, IR reading, compile probes, tokenizer checks.

This module is the only place that talks to the ``openvino`` and
``openvino_tokenizers`` Python packages. Every import happens *inside* the
function that needs it so ``import yonga.cli`` keeps working on a host where
OpenVINO is not installed; a missing package becomes a typed
:class:`~yonga.domain.diagnostics.Diagnostic`, never an ``ImportError`` escaping
into the application layer.

Two deliberate policies govern the error handling here:

* **Nothing in this module raises.** ``read_ir`` returns an unreadable
  :class:`~yonga.domain.artifacts.IRDescription`, ``compile`` returns a failed
  :class:`~yonga.domain.artifacts.ProbeResult`. A validation service must be able
  to inventory a broken artifact without a try/except around every call.
* **Raw error text is preserved, classification is somebody else's job.** When
  OpenVINO itself fails, the (redacted) raw exception text is placed in
  ``ProbeResult.message`` so the registry classifier can match its signature.
  This module only attaches a ``Diagnostic`` for conditions it can classify with
  certainty on its own: a missing package or a package family mismatch.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from yonga.domain.artifacts import (
    IRDescription,
    PortDescription,
    ProbeResult,
    ProbeStatus,
    SubmodelRole,
    manifest_path,
    role_for_filename,
)
from yonga.domain.diagnostics import Diagnostic, FailureCategory, diagnostic
from yonga.domain.hardware import OpenVINODevice, device_kind, normalise_device
from yonga.util.hashing import short_hash
from yonga.util.redaction import redact_text
from yonga.util.versions import families_aligned, format_family, installed_version, version_family

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yonga.config.paths import Paths

__all__ = [
    "DETOKENIZER_IR_NAME",
    "TOKENIZER_IR_NAME",
    "UNKNOWN",
    "OpenVINOProbe",
    "artifact_fingerprint",
    "compile_cache_dir",
    "compile_cache_key",
    "compiler_in_plugin",
    "npu_compiler_version",
    "npu_driver_version",
    "openvino_available",
    "openvino_missing_diagnostic",
    "openvino_runtime_version",
    "openvino_tokenizers_version",
]

#: Placeholder recorded for any device property that could not be read.
UNKNOWN = "unknown"

#: File name of the tokenizer IR produced by the OpenVINO export.
TOKENIZER_IR_NAME = "openvino_tokenizer.xml"
#: File name of the detokenizer IR produced by the OpenVINO export.
DETOKENIZER_IR_NAME = "openvino_detokenizer.xml"

_CPU = "CPU"
_SUPPORTED_PROPERTIES = "SUPPORTED_PROPERTIES"
_FULL_DEVICE_NAME = "FULL_DEVICE_NAME"
_DEVICE_ARCHITECTURE = "DEVICE_ARCHITECTURE"
_NPU_DRIVER_VERSION = "NPU_DRIVER_VERSION"
_NPU_COMPILER_VERSION = "NPU_COMPILER_VERSION"
_NPU_COMPILER_TYPE = "NPU_COMPILER_TYPE"
_PLUGIN_MARKER = "PLUGIN"

# Used when even SUPPORTED_PROPERTIES cannot be read, so the doctor still gets
# the two identity properties every plugin implements.
_FALLBACK_PROPERTY_NAMES: tuple[str, ...] = (_FULL_DEVICE_NAME, _DEVICE_ARCHITECTURE)

# Property names that must not be folded into a compile cache key: the cache
# directory is *derived* from the key, so including it would be circular.
_CACHE_KEY_EXCLUDED_PROPERTIES = frozenset({"CACHE_DIR"})


# --- lazy imports ---------------------------------------------------------------------


def _import_openvino() -> Any:
    """Import :mod:`openvino` lazily. Raises ``ImportError`` when unavailable."""
    # Deliberately lazy: see the module docstring.
    import openvino

    return openvino


def _import_openvino_tokenizers() -> Any:
    """Import :mod:`openvino_tokenizers`, which registers the string-tensor extension.

    Importing the package is what makes ``read_model`` accept tokenizer IR, so it
    must happen *before* the tokenizer XML is read.
    """
    # Deliberately lazy: see the module docstring.
    import openvino_tokenizers

    return openvino_tokenizers


def openvino_available() -> tuple[bool, str | None]:
    """Return ``(available, error)`` for the OpenVINO runtime package.

    ``error`` is a redacted, human-readable import error when OpenVINO cannot be
    imported, and ``None`` when it can.
    """
    try:
        _import_openvino()
    except Exception as exc:  # ImportError, but a broken install can raise anything.
        return (False, redact_text(f"{type(exc).__name__}: {exc}"))
    return (True, None)


def openvino_runtime_version() -> str | None:
    """Return the full OpenVINO runtime version string, or ``None`` when unavailable.

    Prefers ``openvino.get_version()`` (which carries the build id, e.g.
    ``2026.4.0-22959-...``) and falls back to the installed distribution version.
    """
    try:
        module = _import_openvino()
    except Exception:
        return installed_version("openvino")
    with suppress(Exception):
        version = module.get_version()
        if version:
            return str(version)
    with suppress(Exception):
        version = module.__version__
        if version:
            return str(version)
    return installed_version("openvino")


def openvino_tokenizers_version() -> str | None:
    """Return the installed ``openvino-tokenizers`` version, or ``None``."""
    try:
        module = _import_openvino_tokenizers()
    except Exception:
        return installed_version("openvino-tokenizers")
    with suppress(Exception):
        version = module.__version__
        if version:
            return str(version)
    return installed_version("openvino-tokenizers")


# --- diagnostics ----------------------------------------------------------------------


def openvino_missing_diagnostic(error: object = None) -> Diagnostic:
    """Build the diagnostic used whenever the OpenVINO runtime cannot be imported."""
    detail = redact_text(str(error)) if error is not None else "openvino is not installed."
    return diagnostic(
        "runtime.openvino.missing",
        FailureCategory.PACKAGE_ABI_MISMATCH,
        "The OpenVINO runtime is not available",
        f"OpenVINO could not be imported, so no device work is possible. {detail}",
        remediation=[
            "Install the runtime extras: pip install 'yonga[openvino]'",
            "Confirm the active virtual environment is the one yonga runs from.",
        ],
        environment_broken=True,
        raw_excerpt=detail,
    )


def _tokenizers_missing_diagnostic(error: object) -> Diagnostic:
    """Build the diagnostic used when ``openvino_tokenizers`` cannot be imported."""
    detail = redact_text(str(error))
    return diagnostic(
        "runtime.openvino_tokenizers.missing",
        FailureCategory.PACKAGE_ABI_MISMATCH,
        "The OpenVINO tokenizers extension is not available",
        (f"openvino_tokenizers could not be imported, so tokenizer IR cannot be read. {detail}"),
        remediation=[
            "Install the runtime extras: pip install 'yonga[openvino]'",
            "openvino-tokenizers must come from the same release family as openvino.",
        ],
        environment_broken=True,
        raw_excerpt=detail,
    )


def _abi_mismatch_diagnostic(ov_version: str | None, tokenizers_version: str | None) -> Diagnostic:
    """Build the diagnostic for an openvino / openvino-tokenizers family mismatch."""
    ov_family = format_family(version_family(ov_version))
    tok_family = format_family(version_family(tokenizers_version))
    return diagnostic(
        "runtime.openvino_tokenizers.abi_mismatch",
        FailureCategory.PACKAGE_ABI_MISMATCH,
        "OpenVINO and openvino-tokenizers come from different releases",
        (
            f"openvino is {ov_version or UNKNOWN} (family {ov_family}) while "
            f"openvino-tokenizers is {tokenizers_version or UNKNOWN} (family {tok_family}). "
            "The tokenizer extension is ABI-coupled to the runtime, so reading tokenizer IR "
            "with mismatched packages is not safe."
        ),
        remediation=[
            "Install openvino, openvino-genai and openvino-tokenizers from one release family.",
            "For example: pip install 'openvino==2026.4.*' 'openvino-tokenizers==2026.4.*'",
        ],
        environment_broken=True,
        context={
            "openvino": ov_version or UNKNOWN,
            "openvino_tokenizers": tokenizers_version or UNKNOWN,
        },
    )


# --- device property helpers ----------------------------------------------------------


def _property_text(value: object) -> str:
    """Stringify one device property value.

    OpenVINO returns enums, tuples, dicts and ``openvino.Type`` instances as well
    as plain scalars, so every value is coerced with ``str()``.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        return ", ".join(str(item) for item in value)
    return str(value)


def npu_driver_version(props: Mapping[str, str]) -> str | None:
    """Return the NPU driver version from a device property mapping, or ``None``."""
    value = props.get(_NPU_DRIVER_VERSION)
    if value is None or value == UNKNOWN or not value.strip():
        return None
    return value


def npu_compiler_version(props: Mapping[str, str]) -> str | None:
    """Return the NPU compiler version from a device property mapping, or ``None``."""
    value = props.get(_NPU_COMPILER_VERSION)
    if value is None or value == UNKNOWN or not value.strip():
        return None
    return value


def compiler_in_plugin(props: Mapping[str, str]) -> bool | None:
    """Return whether the NPU compiler runs inside the plugin.

    ``True`` when ``NPU_COMPILER_TYPE`` mentions ``PLUGIN`` (this host reports
    ``CompilerType.PREFER_PLUGIN``), ``False`` when the property is present but
    names another compiler, and ``None`` when the property could not be read.
    """
    value = props.get(_NPU_COMPILER_TYPE)
    if value is None or value == UNKNOWN or not value.strip():
        return None
    return _PLUGIN_MARKER in value.upper()


# --- compile cache --------------------------------------------------------------------


def compile_cache_key(
    *,
    artifact_hash: str,
    device: str,
    openvino_version: str | None,
    properties: Mapping[str, str] | None = None,
) -> str:
    """Return the compile cache key for one (artifact, device, runtime, properties) tuple.

    ``openvino_version`` is part of the key on purpose: OpenVINO blobs are not
    portable across runtime versions, so a cache produced by one release must
    never be offered to another (SPEC section 7, ARCHITECTURE section 7). An
    unknown version therefore gets its own bucket rather than silently sharing
    one. ``CACHE_DIR`` is excluded from the property set because the directory is
    derived from this key.
    """
    scrubbed = {
        str(name): _property_text(value)
        for name, value in (properties or {}).items()
        if str(name).upper() not in _CACHE_KEY_EXCLUDED_PROPERTIES
    }
    payload = json.dumps(
        {
            "artifact": artifact_hash,
            "device": normalise_device(device),
            "openvino": (openvino_version or UNKNOWN).strip() or UNKNOWN,
            "properties": scrubbed,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return short_hash(payload, 16)


def compile_cache_dir(paths: Paths, key: str) -> Path:
    """Return (and create) the tool-owned compile cache directory for ``key``.

    The directory lives under ``paths.compile_cache_dir``; nothing outside the
    managed cache root is ever touched. Creation failures are swallowed because a
    missing cache directory must degrade to "no caching", never to a crash.
    """
    directory = paths.compile_cache_dir / key
    with suppress(OSError):
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def artifact_fingerprint(artifact_dir: Path) -> str:
    """Return a cheap, stable fingerprint of a prepared artifact directory.

    Weight bytes are never read (see AGENT_PROMPT section 34): the manifest's
    ``artifact_id`` is used when present, otherwise the sorted list of top-level
    file names and sizes. That is enough to notice a re-export while staying
    O(number of files).
    """
    manifest = manifest_path(artifact_dir)
    if manifest.is_file():
        with suppress(OSError, ValueError, TypeError):
            document = json.loads(manifest.read_text(encoding="utf-8"))
            if isinstance(document, dict):
                artifact_id = document.get("artifact_id")
                if isinstance(artifact_id, str) and artifact_id.strip():
                    return artifact_id.strip()
    entries: list[tuple[str, int]] = []
    with suppress(OSError):
        for item in sorted(artifact_dir.iterdir(), key=lambda path: path.name):
            if not item.is_file():
                continue
            with suppress(OSError):
                entries.append((item.name, item.stat().st_size))
    payload = json.dumps(
        {"dir": artifact_dir.name, "files": entries},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return short_hash(payload, 16)


# --- the probe ------------------------------------------------------------------------


class OpenVINOProbe:
    """Read-only OpenVINO adapter implementing the ``RuntimeProbe`` protocol.

    One instance caches one ``openvino.Core``; reuse an instance for a whole job
    so plugin registration and the compile cache are shared across probes.
    """

    def __init__(self) -> None:
        """Create a probe. No OpenVINO import happens until the first call."""
        self._core_obj: Any | None = None
        self._tokenizers_module: Any | None = None

    # --- runtime access ---------------------------------------------------

    def _core(self) -> Any:
        """Return the cached ``openvino.Core``, creating it on first use."""
        if self._core_obj is None:
            self._core_obj = _import_openvino().Core()
        return self._core_obj

    def _tokenizer_extension(self) -> Any:
        """Import ``openvino_tokenizers`` and guarantee the cached Core carries it.

        The package registers its string-tensor extension by patching
        ``openvino.Core.__init__``, so a ``Core`` built *before* the import has no
        extension and reads tokenizer IR as if it were corrupt. Importing it here
        therefore invalidates any earlier ``Core``. Raises ``ImportError`` when the
        package is not installed.
        """
        if self._tokenizers_module is None:
            self._tokenizers_module = _import_openvino_tokenizers()
            # Force a fresh Core so the patched constructor registers the extension.
            self._core_obj = None
        return self._tokenizers_module

    def _try_tokenizer_extension(self) -> None:
        """Best-effort variant of :meth:`_tokenizer_extension` that never raises.

        Used before reading arbitrary IR so tokenizer submodels are described
        accurately when the extension is installed, while an artifact without
        tokenizers still works on a runtime-only installation.
        """
        with suppress(Exception):
            self._tokenizer_extension()

    def available(self) -> tuple[bool, str | None]:
        """Return ``(available, error)`` for the OpenVINO runtime."""
        return openvino_available()

    def version(self) -> str | None:
        """Return the OpenVINO runtime version string, or ``None``."""
        return openvino_runtime_version()

    # --- devices ----------------------------------------------------------

    def available_devices(self) -> list[str]:
        """Return the device names OpenVINO reports, e.g. ``["CPU", "GPU", "NPU"]``.

        ``Core().available_devices`` is authoritative: a device missing from this
        list is not usable, whatever ``/dev`` or ``lsmod`` suggest. Returns an
        empty list when OpenVINO is unavailable.
        """
        try:
            return [str(name) for name in self._core().available_devices]
        except Exception:
            return []

    def device_properties(self, device: str) -> dict[str, str]:
        """Read every property of one device defensively.

        ``SUPPORTED_PROPERTIES`` lists the property names and their access mode,
        but reading a listed property can still raise: ``WO`` entries always do,
        and some ``RO`` entries do too (``COMPATIBILITY_CHECK`` raises on every
        plugin of this host). Any failed read is recorded as ``"unknown"`` rather
        than propagated, and every value is stringified.
        """
        try:
            core = self._core()
        except Exception:
            return {}
        names: list[str] = []
        try:
            supported = core.get_property(device, _SUPPORTED_PROPERTIES)
            if isinstance(supported, Mapping):
                names = [str(name) for name in supported]
            else:
                names = [str(name) for name in supported]
        except Exception:
            names = list(_FALLBACK_PROPERTY_NAMES)
        properties: dict[str, str] = {}
        for name in names:
            if name == _SUPPORTED_PROPERTIES:
                # Self-referential and enormous when stringified; the names are
                # already the keys of this mapping.
                continue
            try:
                properties[name] = _property_text(core.get_property(device, name))
            except Exception:
                properties[name] = UNKNOWN
        return properties

    def devices(self) -> list[OpenVINODevice]:
        """Return the full device inventory, one entry per available device."""
        inventory: list[OpenVINODevice] = []
        for name in self.available_devices():
            properties = self.device_properties(name)
            full_name = properties.get(_FULL_DEVICE_NAME) or UNKNOWN
            architecture = properties.get(_DEVICE_ARCHITECTURE)
            inventory.append(
                OpenVINODevice(
                    name=name,
                    kind=device_kind(name),
                    full_name=full_name,
                    architecture=architecture,
                    properties=properties,
                )
            )
        return inventory

    # --- IR reading -------------------------------------------------------

    def read_ir(self, path: Path) -> IRDescription:
        """Read one OpenVINO IR file and describe its ports. Never raises.

        On any failure the returned description carries ``readable=False`` and a
        redacted ``read_error``; the role, companion ``.bin`` path and byte total
        are still filled in so a broken artifact can be inventoried.
        """
        role = role_for_filename(path.name)
        bin_path = path.with_suffix(".bin")
        has_bin = False
        with suppress(OSError):
            has_bin = bin_path.is_file()
        bytes_total = _path_size(path) + (_path_size(bin_path) if has_bin else 0)
        companion = bin_path if has_bin else None
        self._try_tokenizer_extension()
        try:
            model = self._core().read_model(str(path))
        except Exception as exc:
            return IRDescription(
                path=path,
                role=role,
                bin_path=companion,
                bytes_total=bytes_total,
                readable=False,
                read_error=redact_text(f"{type(exc).__name__}: {exc}"),
            )
        friendly_name: str | None = None
        with suppress(Exception):
            friendly_name = str(model.get_friendly_name())
        return IRDescription(
            path=path,
            role=role,
            bin_path=companion,
            bytes_total=bytes_total,
            friendly_name=friendly_name,
            inputs=_describe_ports(model, "inputs"),
            outputs=_describe_ports(model, "outputs"),
            readable=True,
        )

    def inventory_irs(self, artifact: Path) -> list[IRDescription]:
        """Describe every ``*.xml`` IR in an artifact directory, sorted by name.

        Only the top level is scanned: the OpenVINO export writes a flat
        directory, and the tool-owned ``.yonga`` metadata directory must never be
        mistaken for model content.
        """
        try:
            candidates = sorted(
                (item for item in artifact.glob("*.xml") if item.is_file()),
                key=lambda item: item.name,
            )
        except OSError:
            return []
        return [self.read_ir(item) for item in candidates]

    # --- compiling --------------------------------------------------------

    def compile(
        self,
        path: Path,
        device: str,
        properties: Mapping[str, str] | None = None,
    ) -> ProbeResult:
        """Read and compile one IR on ``device``, timing the whole operation.

        Never raises. On failure the status is
        :attr:`~yonga.domain.artifacts.ProbeStatus.FAILED` and the **raw**
        exception text is preserved in ``message`` so the registry classifier can
        match its signature (for example the NPU "to_shape was called on a
        dynamic shape" compiler rejection). Classification is intentionally left
        to the caller.
        """
        role = role_for_filename(path.name)
        config = {str(name): _property_text(value) for name, value in (properties or {}).items()}
        started = time.perf_counter()
        self._try_tokenizer_extension()
        try:
            core = self._core()
        except Exception as exc:
            return ProbeResult(
                device=device,
                role=role,
                path=path,
                status=ProbeStatus.FAILED,
                duration_seconds=0.0,
                diagnostic=openvino_missing_diagnostic(exc),
                properties=config,
                message=redact_text(f"{type(exc).__name__}: {exc}"),
            )
        try:
            model = core.read_model(str(path))
            if config:
                core.compile_model(model, device, config)
            else:
                core.compile_model(model, device)
        except Exception as exc:
            return ProbeResult(
                device=device,
                role=role,
                path=path,
                status=ProbeStatus.FAILED,
                duration_seconds=time.perf_counter() - started,
                properties=config,
                message=redact_text(str(exc)),
            )
        return ProbeResult(
            device=device,
            role=role,
            path=path,
            status=ProbeStatus.OK,
            duration_seconds=time.perf_counter() - started,
            properties=config,
        )

    # --- tokenizers -------------------------------------------------------

    def read_tokenizer(self, artifact: Path) -> ProbeResult:
        """Read and CPU-compile the tokenizer/detokenizer pair of an artifact.

        The ``openvino_tokenizers`` extension is imported first (that import is
        what registers the string-tensor types), and its release family is
        checked against the runtime's before anything is read. Tokenizers are
        always CPU-hosted, so this never probes GPU or NPU.

        The returned ``properties`` mapping reports each half separately with
        ``ok``/``failed``/``missing`` under the keys ``tokenizer_read``,
        ``tokenizer_compile``, ``detokenizer_read`` and ``detokenizer_compile``.
        On failure the raw OpenVINO error text is kept in ``message`` for the
        classifier (the ``Unsupported element type: STRiNG`` signature is matched
        there, not here).
        """
        started = time.perf_counter()
        try:
            tokenizers = self._tokenizer_extension()
        except Exception as exc:
            return ProbeResult(
                device=_CPU,
                role=SubmodelRole.TOKENIZER,
                path=artifact,
                status=ProbeStatus.FAILED,
                duration_seconds=time.perf_counter() - started,
                diagnostic=_tokenizers_missing_diagnostic(exc),
                message=redact_text(f"{type(exc).__name__}: {exc}"),
            )
        try:
            core = self._core()
        except Exception as exc:
            return ProbeResult(
                device=_CPU,
                role=SubmodelRole.TOKENIZER,
                path=artifact,
                status=ProbeStatus.FAILED,
                duration_seconds=time.perf_counter() - started,
                diagnostic=openvino_missing_diagnostic(exc),
                message=redact_text(f"{type(exc).__name__}: {exc}"),
            )

        ov_version = openvino_runtime_version()
        tok_version = openvino_tokenizers_version()
        versions = {
            "openvino_version": ov_version or UNKNOWN,
            "openvino_tokenizers_version": tok_version or UNKNOWN,
        }
        # ``is_openvino_tokenizers_compatible`` is set by the extension itself when it
        # tries to load its shared library against this runtime, so it is stronger
        # evidence than the version strings; the family check is the fallback.
        binary_compatible = bool(getattr(tokenizers, "is_openvino_tokenizers_compatible", True))
        families_ok = families_aligned({"openvino": ov_version, "openvino-tokenizers": tok_version})
        if not binary_compatible or not families_ok:
            return ProbeResult(
                device=_CPU,
                role=SubmodelRole.TOKENIZER,
                path=artifact,
                status=ProbeStatus.FAILED,
                duration_seconds=time.perf_counter() - started,
                diagnostic=_abi_mismatch_diagnostic(ov_version, tok_version),
                properties=versions,
                message="openvino and openvino-tokenizers are from different release families.",
            )

        outcomes: dict[str, str] = dict(versions)
        errors: list[str] = []
        present = 0
        failures = 0
        for role, name in (
            (SubmodelRole.TOKENIZER, TOKENIZER_IR_NAME),
            (SubmodelRole.DETOKENIZER, DETOKENIZER_IR_NAME),
        ):
            target = artifact / name
            if not target.is_file():
                outcomes[f"{role.value}_read"] = "missing"
                outcomes[f"{role.value}_compile"] = "missing"
                continue
            present += 1
            try:
                model = core.read_model(str(target))
            except Exception as exc:
                outcomes[f"{role.value}_read"] = "failed"
                outcomes[f"{role.value}_compile"] = "not_attempted"
                errors.append(redact_text(str(exc)))
                failures += 1
                continue
            outcomes[f"{role.value}_read"] = "ok"
            try:
                core.compile_model(model, _CPU)
            except Exception as exc:
                outcomes[f"{role.value}_compile"] = "failed"
                errors.append(redact_text(str(exc)))
                failures += 1
                continue
            outcomes[f"{role.value}_compile"] = "ok"

        duration = time.perf_counter() - started
        if present == 0:
            return ProbeResult(
                device=_CPU,
                role=SubmodelRole.TOKENIZER,
                path=artifact,
                status=ProbeStatus.SKIPPED,
                duration_seconds=duration,
                properties=outcomes,
                message=(
                    f"Neither {TOKENIZER_IR_NAME} nor {DETOKENIZER_IR_NAME} is present in "
                    f"{artifact}."
                ),
            )
        complete = present == 2 and failures == 0
        message = "\n\n".join(errors) if errors else None
        if message is None and not complete:
            missing = [
                name
                for role, name in (
                    (SubmodelRole.TOKENIZER, TOKENIZER_IR_NAME),
                    (SubmodelRole.DETOKENIZER, DETOKENIZER_IR_NAME),
                )
                if outcomes.get(f"{role.value}_read") == "missing"
            ]
            message = f"Missing tokenizer IR: {', '.join(missing)}."
        return ProbeResult(
            device=_CPU,
            role=SubmodelRole.TOKENIZER,
            path=artifact,
            status=ProbeStatus.OK if complete else ProbeStatus.FAILED,
            duration_seconds=duration,
            properties=outcomes,
            message=message,
        )

    # --- doctor helpers ---------------------------------------------------

    npu_driver_version = staticmethod(npu_driver_version)
    npu_compiler_version = staticmethod(npu_compiler_version)
    compiler_in_plugin = staticmethod(compiler_in_plugin)


# --- private helpers ------------------------------------------------------------------


def _path_size(path: Path) -> int:
    """Return a file's size in bytes, or 0 when it cannot be stat'ed."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _describe_port(port: Any) -> PortDescription:
    """Describe one model port, falling back to ``unknown`` for unreadable pieces."""
    name = UNKNOWN
    try:
        name = str(port.get_any_name())
    except Exception:
        with suppress(Exception):
            names = sorted(str(item) for item in port.get_names())
            if names:
                name = names[0]
    element_type = UNKNOWN
    with suppress(Exception):
        element_type = str(port.get_element_type())
    shape = UNKNOWN
    with suppress(Exception):
        shape = str(port.get_partial_shape())
    return PortDescription(name=name or UNKNOWN, element_type=element_type, shape=shape)


def _describe_ports(model: Any, attribute: str) -> list[PortDescription]:
    """Describe every input or output port of a model, tolerating a hostile object."""
    try:
        ports = list(getattr(model, attribute))
    except Exception:
        return []
    return [_describe_port(port) for port in ports]


def main() -> None:
    """Print the OpenVINO device inventory as JSON (manual diagnostic entry point)."""
    probe = OpenVINOProbe()
    available, error = openvino_available()
    document = {
        "openvino_available": available,
        "openvino_version": openvino_runtime_version(),
        "openvino_import_error": error,
        "devices": [device.model_dump(mode="json") for device in probe.devices()],
    }
    print(json.dumps(document, indent=2, sort_keys=True))


# OpenVINO may start a multiprocessing forkserver on import; the child re-imports
# the main module, so every executable path in this package must stay guarded
# (docs/MODEL_PIPELINE.md section 10).
if __name__ == "__main__":  # pragma: no cover - manual diagnostic entry point
    main()
