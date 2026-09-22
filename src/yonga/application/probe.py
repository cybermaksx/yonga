"""Artifact probe service: the productised form of the manual diagnostic script.

``yonga probe ARTIFACT`` answers one question honestly: *which parts of this
prepared artifact can actually be compiled, and on which device?* It does that by

1. inventorying every ``*.xml`` at the top level of the artifact and assigning a
   role with :func:`yonga.domain.artifacts.role_for_filename`;
2. reading each IR and recording its input/output names, element types and shapes;
3. compiling the numeric submodels on the requested device, and the tokenizer and
   detokenizer on **CPU only** — tokenizers are string-tensor graphs and do not
   belong on a GPU or an NPU (``docs/MODEL_PIPELINE.md`` sections 9 and 10);
4. classifying every failure through the compatibility registry while keeping the
   raw runtime text, so the user reads "the NPU compiler rejected this graph"
   rather than a Level Zero return code.

Two rules are not negotiable. A device is never reported as working when one of
its compiles failed, and ``--device auto`` reports **every** candidate device it
tried, not just the winner.

This module deliberately has no module-scope side effects, and its command-line
entry point is guarded by ``if __name__ == "__main__"``. OpenVINO may start a
multiprocessing forkserver whose child re-imports the main module; an unguarded
entry point is re-executed in that child, which was observed for real during
development (``docs/MODEL_PIPELINE.md`` section 10).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from yonga.application.jobs import JobContext
from yonga.application.progress import ProgressKind
from yonga.config.paths import Paths
from yonga.config.settings import Settings
from yonga.domain.artifacts import (
    ArtifactManifest,
    DeviceProbeReport,
    IRDescription,
    ProbeResult,
    ProbeStatus,
    SubmodelRole,
    manifest_path,
)
from yonga.domain.compatibility import DeviceStatus
from yonga.domain.diagnostics import (
    ConfigurationError,
    Diagnostic,
    FailureCategory,
    Severity,
    diagnostic,
)
from yonga.domain.hardware import normalise_device
from yonga.domain.models import ModelIdentity, ModelMetadata

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yonga.adapters import RuntimeProbe
    from yonga.registry.resolver import RegistryResolver

__all__ = [
    "DEFAULT_DEVICE_ORDER",
    "TOKENIZER_DEVICE",
    "ArtifactStoreLike",
    "LocalArtifactStore",
    "ProbeOutcome",
    "ProbeService",
    "main",
    "resolve_artifact",
    "role_device_matrix",
]

#: Tokenizer and detokenizer IRs are always compiled here, whatever was requested.
TOKENIZER_DEVICE = "CPU"

#: Fallback probing order when the registry has no opinion, most specialised first.
DEFAULT_DEVICE_ORDER: tuple[str, ...] = ("NPU", "GPU", "CPU")

#: How confident the registry is about a device, best first.
_DEVICE_STATUS_ORDER: tuple[DeviceStatus, ...] = (
    DeviceStatus.SUPPORTED,
    DeviceStatus.EXPERIMENTAL,
    DeviceStatus.UNKNOWN,
    DeviceStatus.UNSUPPORTED,
)

#: Worst-wins ranking used when one role is probed more than once on one device.
_STATUS_SEVERITY: Mapping[ProbeStatus, int] = {
    ProbeStatus.OK: 0,
    ProbeStatus.SKIPPED: 1,
    ProbeStatus.NOT_ATTEMPTED: 2,
    ProbeStatus.FAILED: 3,
}

_AUTO = "auto"

#: Upper bound on how many directories a catalogue scan will look at.
_MAX_SCAN_DIRS = 512


def _now() -> datetime:
    """Current UTC timestamp."""
    return datetime.now(UTC)


# --- artifact store port --------------------------------------------------------------


@runtime_checkable
class ArtifactStoreLike(Protocol):
    """The slice of an artifact store the probe and benchmark services depend on.

    Declared structurally, with the method names ``yonga.application.artifacts``
    uses, so these services never import a concrete store and the real
    :class:`~yonga.application.artifacts.ArtifactStore` drops straight in. A
    reference that resolves to nothing is reported as ``None``; turning that into
    a typed failure is :func:`resolve_artifact`'s job, so every caller phrases it
    the same way.
    """

    def find(self, reference: str | Path) -> Path | None:
        """Return the prepared artifact directory named by ``reference``, or None."""
        ...

    def manifest_for(self, artifact_dir: Path) -> ArtifactManifest | None:
        """Return the artifact's manifest, or ``None`` when it has none."""
        ...


def resolve_artifact(store: ArtifactStoreLike, reference: str | Path) -> Path:
    """Resolve ``reference`` through ``store``, raising when nothing matches."""
    found = store.find(reference)
    if found is not None:
        return found
    raise ConfigurationError(
        diagnostic(
            "artifact.not_found",
            FailureCategory.CONFIGURATION_ERROR,
            "No prepared artifact matches that reference",
            f"{reference!r} is neither a directory nor a prepared artifact in the catalogue.",
            remediation=[
                "List what is prepared: yonga list",
                "Prepare the model first: yonga prepare MODEL",
                "Or pass the artifact directory directly.",
            ],
            context={"reference": str(reference)},
        )
    )


class LocalArtifactStore:
    """Resolves artifact references against the managed artifacts directory.

    Lookup order: an existing directory path, then ``<artifacts>/<reference>``,
    then ``<artifacts>/<slug>`` for a ``namespace/name`` repository id, then a
    two-level scan for a manifest whose ``artifact_id`` or recorded repository id
    matches. Nothing is ever written, and a reference that matches nothing returns
    ``None`` rather than guessing.
    """

    def __init__(self, paths: Paths | None = None) -> None:
        """Bind the store to a :class:`~yonga.config.paths.Paths` layout."""
        self._paths = paths

    @property
    def root(self) -> Path | None:
        """Root of the managed artifacts directory, when paths are known."""
        return None if self._paths is None else self._paths.artifacts_dir

    def find(self, reference: str | Path) -> Path | None:
        """Return the artifact directory for ``reference``, or ``None``.

        Unlike the catalogue store, a bare export directory with no manifest is
        accepted: ``yonga probe`` and ``yonga bench`` must work on a directory a
        user just exported by hand.
        """
        candidate = Path(reference).expanduser()
        if candidate.is_dir():
            return candidate
        root = self.root
        text = str(reference).strip()
        if root is not None and text:
            direct = root / text
            if direct.is_dir():
                return direct
            slug = text.replace("/", "__")
            by_slug = root / slug
            if by_slug.is_dir():
                return by_slug
            return self._scan(root, text)
        return None

    def manifest_for(self, artifact_dir: Path) -> ArtifactManifest | None:
        """Read the artifact manifest, returning ``None`` when it is absent."""
        path = manifest_path(artifact_dir)
        if not path.is_file():
            return None
        with suppress(Exception):
            return ArtifactManifest.model_validate_file(path)
        return None

    def _scan(self, root: Path, reference: str) -> Path | None:
        """Search the catalogue for a manifest naming ``reference``."""
        wanted = reference.casefold()
        for directory in self._candidate_dirs(root):
            manifest = self.manifest_for(directory)
            if manifest is None:
                continue
            names = {
                manifest.artifact_id.casefold(),
                manifest.requested_model.repo_id.casefold(),
                manifest.resolved_source.repo_id.casefold(),
            }
            if wanted in names:
                return directory
        return None

    @staticmethod
    def _candidate_dirs(root: Path) -> list[Path]:
        """Return the directories that may hold an artifact, at most two levels deep."""
        found: list[Path] = []
        if not root.is_dir():
            return found
        with suppress(OSError):
            for first in sorted(root.iterdir()):
                if not first.is_dir() or first.name.startswith("."):
                    continue
                found.append(first)
                with suppress(OSError):
                    for second in sorted(first.iterdir()):
                        if second.is_dir() and not second.name.startswith("."):
                            found.append(second)
        return found[:_MAX_SCAN_DIRS]


# --- results --------------------------------------------------------------------------


class ProbeOutcome(BaseModel):
    """Everything one probe run learned about an artifact.

    :meth:`ProbeService.probe` returns only :attr:`reports`; this richer object is
    what ``--all-submodels`` and the JSON output are built from, because the IR
    port descriptions have nowhere to live inside a
    :class:`~yonga.domain.artifacts.DeviceProbeReport`.
    """

    model_config = ConfigDict(extra="ignore")

    artifact_dir: Path
    requested_device: str = _AUTO
    candidates: list[str] = Field(default_factory=list)
    irs: list[IRDescription] = Field(default_factory=list)
    reports: list[DeviceProbeReport] = Field(default_factory=list)
    tokenizer: ProbeResult | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when at least one probed device compiled everything it was given."""
        return any(report.ok for report in self.reports)

    @property
    def working_devices(self) -> list[str]:
        """Devices whose probe succeeded, in the order they were probed."""
        return [report.device for report in self.reports if report.ok]

    def role_matrix(self) -> dict[str, dict[str, str]]:
        """Return the per-role, per-device map of ``docs/MODEL_PIPELINE.md`` section 10."""
        return role_device_matrix(self.reports)


def role_device_matrix(reports: Iterable[DeviceProbeReport]) -> dict[str, dict[str, str]]:
    """Collapse probe reports into ``{"language": {"GPU": "ok", "NPU": "failed"}}``.

    The device key comes from each individual :class:`ProbeResult`, not from the
    report, so a tokenizer probed on CPU while the language model was probed on
    NPU is reported under ``CPU`` — which is the truth. When one role is probed
    more than once on the same device the worse status wins, so a partially
    failing role can never be presented as working.
    """
    matrix: dict[str, dict[str, str]] = {}
    for report in reports:
        results = [*report.submodels]
        if report.smoke_test is not None:
            results.append(report.smoke_test)
        for result in results:
            role = (result.role or SubmodelRole.OTHER).value
            per_device = matrix.setdefault(role, {})
            previous = per_device.get(result.device)
            if previous is None:
                per_device[result.device] = result.status.value
                continue
            previous_rank = _STATUS_SEVERITY.get(ProbeStatus(previous), 0)
            if _STATUS_SEVERITY[result.status] > previous_rank:
                per_device[result.device] = result.status.value
    return matrix


# --- the service ----------------------------------------------------------------------


class ProbeService:
    """Reads, compiles and classifies every submodel of a prepared artifact."""

    def __init__(
        self,
        runtime_probe: RuntimeProbe,
        registry_resolver: RegistryResolver,
        artifact_store: ArtifactStoreLike | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Wire the service to its adapters.

        ``artifact_store`` defaults to a :class:`LocalArtifactStore` over the
        settings' managed artifacts directory, so the service is usable with a
        bare directory path and no catalogue.
        """
        self._runtime = runtime_probe
        self._resolver = registry_resolver
        self._settings = settings
        paths = None if settings is None else settings.paths
        self._store: ArtifactStoreLike = artifact_store or LocalArtifactStore(paths)

    # --- inventory --------------------------------------------------------

    def inventory(self, artifact_dir: Path) -> list[IRDescription]:
        """Describe every top-level ``*.xml`` IR in ``artifact_dir``, sorted by name.

        Only the top level is scanned: the OpenVINO export writes a flat
        directory and the tool-owned ``.yonga`` metadata directory is not model
        content. Roles come from
        :func:`~yonga.domain.artifacts.role_for_filename`, and an unreadable IR is
        still inventoried with ``readable=False``.
        """
        try:
            files = sorted(
                (item for item in artifact_dir.glob("*.xml") if item.is_file()),
                key=lambda item: item.name,
            )
        except OSError:
            return []
        return [self._runtime.read_ir(item) for item in files]

    # --- device selection -------------------------------------------------

    def candidate_devices(
        self,
        *,
        device: str,
        manifest: ArtifactManifest | None = None,
    ) -> list[str]:
        """Return the devices to probe, in the order they should be tried.

        An explicit device yields exactly that device: ``SPEC.md`` section 7.1
        forbids silently moving to another one. ``auto`` walks the registry's
        recommended device first, then the devices the registry rates best, then
        anything else OpenVINO exposes — intersected with what OpenVINO actually
        reports, because a device the runtime cannot see cannot be probed.
        """
        requested = (device or _AUTO).strip()
        if requested.casefold() != _AUTO:
            return [normalise_device(requested)]

        available = self._available_devices()
        ordered: list[str] = []

        meta = self._metadata(manifest)
        arch_rule, _ = self._resolver.resolve_architecture(meta)
        model_rule, _ = self._resolver.resolve_model_rule(meta)
        if arch_rule is not None and arch_rule.recommended_device:
            ordered.append(normalise_device(arch_rule.recommended_device))

        compatibility = self._resolver.device_compatibility(
            arch_rule=arch_rule,
            model_rule=model_rule,
        )
        for status in _DEVICE_STATUS_ORDER:
            for entry in compatibility:
                if entry.status is status:
                    ordered.append(normalise_device(entry.device))

        if manifest is not None and manifest.recommended_runtime is not None:
            ordered.append(normalise_device(manifest.recommended_runtime.device))
        # The fixed order comes before the runtime's enumeration order so that the
        # candidate list does not depend on which plugins happened to register first.
        ordered.extend(DEFAULT_DEVICE_ORDER)
        ordered.extend(available)

        seen: list[str] = []
        for name in ordered:
            if name and name not in seen and (not available or name in available):
                seen.append(name)
        return seen or [TOKENIZER_DEVICE]

    def _available_devices(self) -> list[str]:
        """Devices OpenVINO reports, normalised. Empty when OpenVINO is absent."""
        try:
            raw = self._runtime.available_devices()
        except Exception:
            return []
        names: list[str] = []
        for entry in raw:
            name = normalise_device(str(entry))
            if name and name not in names:
                names.append(name)
        return names

    # --- probing ----------------------------------------------------------

    def probe(
        self,
        artifact_or_model: str | Path,
        *,
        device: str = _AUTO,
        all_submodels: bool = False,
        ctx: JobContext | None = None,
    ) -> list[DeviceProbeReport]:
        """Probe an artifact and return one report per device that was tried.

        ``device`` is a device name or ``auto``. With ``all_submodels`` every
        numeric IR in the artifact is compiled; by default only the language
        model is, which is the cheap answer to "can this run here at all".
        Tokenizers are always probed, always on CPU.
        """
        return self.probe_detailed(
            artifact_or_model,
            device=device,
            all_submodels=all_submodels,
            ctx=ctx,
        ).reports

    def probe_detailed(
        self,
        artifact_or_model: str | Path,
        *,
        device: str = _AUTO,
        all_submodels: bool = False,
        ctx: JobContext | None = None,
    ) -> ProbeOutcome:
        """Probe an artifact and return the reports together with the IR inventory."""
        artifact_dir = resolve_artifact(self._store, artifact_or_model)
        manifest = self._store.manifest_for(artifact_dir)
        notes: list[str] = []

        irs = self.inventory(artifact_dir)
        if not irs:
            notes.append("No OpenVINO IR (*.xml) was found at the top level of this artifact.")
        self._emit_inventory(irs, ctx=ctx)

        numeric = self._numeric_targets(irs, all_submodels=all_submodels)
        if not numeric and irs:
            notes.append("This artifact contains no numeric submodel to compile.")

        tokenizer = self._probe_tokenizer(artifact_dir, irs, manifest=manifest, ctx=ctx)
        candidates = self.candidate_devices(device=device, manifest=manifest)
        if not self._available_devices():
            notes.append(
                "OpenVINO reported no devices on this host; the probe still ran so the "
                "failure is visible rather than hidden."
            )

        reports: list[DeviceProbeReport] = []
        for candidate in candidates:
            reports.append(
                self._probe_device(
                    candidate,
                    targets=numeric,
                    tokenizer=tokenizer,
                    manifest=manifest,
                    ctx=ctx,
                )
            )

        return ProbeOutcome(
            artifact_dir=artifact_dir,
            requested_device=(device or _AUTO).strip() or _AUTO,
            candidates=candidates,
            irs=irs,
            reports=reports,
            tokenizer=tokenizer,
            notes=notes,
        )

    # --- internals --------------------------------------------------------

    @staticmethod
    def _numeric_targets(
        irs: Sequence[IRDescription],
        *,
        all_submodels: bool,
    ) -> list[IRDescription]:
        """Pick the numeric IRs to compile on the target device.

        Tokenizer graphs are excluded here because they are CPU-only. Without
        ``all_submodels`` the language model alone is probed when the artifact has
        one; an artifact with no language role falls back to every numeric IR
        rather than probing nothing.
        """
        numeric = [ir for ir in irs if not ir.role.is_tokenizer]
        if all_submodels:
            return numeric
        language = [ir for ir in numeric if ir.role is SubmodelRole.LANGUAGE]
        return language or numeric

    def _probe_tokenizer(
        self,
        artifact_dir: Path,
        irs: Sequence[IRDescription],
        *,
        manifest: ArtifactManifest | None,
        ctx: JobContext | None,
    ) -> ProbeResult | None:
        """Read and CPU-compile the tokenizer pair, whatever device was requested."""
        if not any(ir.role.is_tokenizer for ir in irs):
            return None
        self._emit(
            ctx,
            f"Reading the tokenizer on {TOKENIZER_DEVICE}",
            device=TOKENIZER_DEVICE,
        )
        result = self._runtime.read_tokenizer(artifact_dir)
        if result.status is ProbeStatus.FAILED and result.diagnostic is None and result.message:
            result = result.model_copy(
                update={
                    "diagnostic": self._classify(
                        result.message,
                        device=TOKENIZER_DEVICE,
                        manifest=manifest,
                        ctx=ctx,
                    )
                }
            )
        self._emit_result(result, ctx=ctx)
        return result

    def _probe_device(
        self,
        device: str,
        *,
        targets: Sequence[IRDescription],
        tokenizer: ProbeResult | None,
        manifest: ArtifactManifest | None,
        ctx: JobContext | None,
    ) -> DeviceProbeReport:
        """Compile every numeric target on one device and build its report."""
        results: list[ProbeResult] = []
        total = len(targets)
        for index, ir in enumerate(targets, start=1):
            self._emit(
                ctx,
                f"Compiling {ir.path.name} on {device}",
                device=device,
                current=index,
                total=total,
            )
            results.append(self._probe_one(ir, device=device, manifest=manifest, ctx=ctx))
        if tokenizer is not None:
            # Carried into every device report on purpose: the tokenizer is part
            # of the pipeline on any device, and its own ``device`` field keeps
            # the record honest about where it was actually compiled.
            results.append(tokenizer)
        return DeviceProbeReport(
            device=device,
            status=_report_status(results),
            submodels=results,
            diagnostic=_first_diagnostic(results),
            attempted_at=_now(),
        )

    def _probe_one(
        self,
        ir: IRDescription,
        *,
        device: str,
        manifest: ArtifactManifest | None,
        ctx: JobContext | None,
    ) -> ProbeResult:
        """Compile a single IR, or report why it could not even be read.

        No runtime properties are forwarded to the compile: the manifest's
        recommended properties are OpenVINO GenAI *pipeline* keys (``CACHE_DIR``,
        ``ATTENTION_BACKEND``, ``MAX_PROMPT_LEN``) and a plugin rejects them when
        they are handed to ``Core.compile_model`` for a raw IR.
        """
        if not ir.readable:
            message = ir.read_error or "The IR could not be read."
            result = ProbeResult(
                device=device,
                role=ir.role,
                path=ir.path,
                status=ProbeStatus.FAILED,
                message=message,
                diagnostic=self._classify(message, device=device, manifest=manifest, ctx=ctx),
            )
            self._emit_result(result, ctx=ctx)
            return result
        result = self._runtime.compile(ir.path, device)
        if result.status is ProbeStatus.FAILED and result.diagnostic is None and result.message:
            result = result.model_copy(
                update={
                    "diagnostic": self._classify(
                        result.message,
                        device=device,
                        manifest=manifest,
                        ctx=ctx,
                    )
                }
            )
        if result.role is None:
            result = result.model_copy(update={"role": ir.role})
        self._emit_result(result, ctx=ctx)
        return result

    def _classify(
        self,
        text: str,
        *,
        device: str,
        manifest: ArtifactManifest | None,
        ctx: JobContext | None,
    ) -> Diagnostic:
        """Turn raw runtime text into a diagnostic without losing the raw text."""
        return self._resolver.classify(
            raw_error=text,
            device=device,
            meta=self._metadata(manifest),
            stage="probe",
            job_id=None if ctx is None else ctx.job_id,
            log_path=None if ctx is None else ctx.logger.path,
        )

    @staticmethod
    def _metadata(manifest: ArtifactManifest | None) -> ModelMetadata | None:
        """Rebuild the model facts the registry matches on from an artifact manifest."""
        if manifest is None:
            return None
        architectures = [manifest.architecture] if manifest.architecture else []
        return ModelMetadata(
            identity=ModelIdentity(
                repo_id=manifest.resolved_source.repo_id,
                revision=manifest.resolved_source.resolved_revision,
            ),
            resolved_revision=manifest.resolved_source.resolved_revision,
            architectures=architectures,
            model_type=manifest.model_type,
            task=manifest.task,
        )

    def _emit_inventory(self, irs: Sequence[IRDescription], *, ctx: JobContext | None) -> None:
        """Report the IR inventory, including the port shapes that were read."""
        if ctx is None:
            return
        for ir in irs:
            inputs = ", ".join(f"{port.name}:{port.element_type}{port.shape}" for port in ir.inputs)
            outputs = ", ".join(
                f"{port.name}:{port.element_type}{port.shape}" for port in ir.outputs
            )
            ctx.emit(
                ProgressKind.VALIDATE,
                f"{ir.path.name}: {ir.role.value}",
                severity=Severity.INFO if ir.readable else Severity.ERROR,
                status="read" if ir.readable else "unreadable",
                role=ir.role.value,
                inputs=inputs,
                outputs=outputs,
                bytes_total=ir.bytes_total,
            )

    @staticmethod
    def _emit(
        ctx: JobContext | None,
        message: str,
        *,
        device: str | None = None,
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        """Emit a probe progress event, if there is a job to emit into."""
        if ctx is None:
            return
        ctx.emit(ProgressKind.PROBE, message, device=device, current=current, total=total)

    @staticmethod
    def _emit_result(result: ProbeResult, *, ctx: JobContext | None) -> None:
        """Report one finished probe."""
        if ctx is None:
            return
        severity = Severity.OK if result.ok else Severity.ERROR
        role = (result.role or SubmodelRole.OTHER).value
        ctx.emit(
            ProgressKind.PROBE,
            f"{role} on {result.device}: {result.status.value}",
            device=result.device,
            status=result.status.value,
            severity=severity,
            diagnostic=result.diagnostic,
            role=role,
        )


def _report_status(results: Sequence[ProbeResult]) -> ProbeStatus:
    """Collapse per-submodel statuses into a device verdict, worst-case first.

    A device is only ``ok`` when every submodel it was given compiled. This is the
    rule that stops ``auto`` mode claiming a device works because most of it did.
    A mix of successes and skips is reported as ``skipped``, never as ``ok``.
    """
    if not results:
        return ProbeStatus.NOT_ATTEMPTED
    if any(result.status is ProbeStatus.FAILED for result in results):
        return ProbeStatus.FAILED
    if all(result.status is ProbeStatus.OK for result in results):
        return ProbeStatus.OK
    return ProbeStatus.SKIPPED


def _first_diagnostic(results: Sequence[ProbeResult]) -> Diagnostic | None:
    """Return the diagnostic of the first failed submodel, if any."""
    for result in results:
        if result.status is ProbeStatus.FAILED and result.diagnostic is not None:
            return result.diagnostic
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Manual diagnostic entry point: probe one artifact and print the JSON report.

    This is a command-line entry point, so — like the CLI — it is the one place in
    this module that constructs concrete adapters and writes to stdout. Nothing
    above this function imports or touches them.
    """
    import argparse

    from yonga.adapters.openvino_probe import OpenVINOProbe
    from yonga.config.settings import load_settings
    from yonga.registry.loader import load_builtin_registry
    from yonga.registry.resolver import RegistryResolver, installed_registry_versions

    parser = argparse.ArgumentParser(
        prog="python -m yonga.application.probe",
        description="Probe a prepared OpenVINO artifact and print a JSON report.",
    )
    parser.add_argument("artifact", help="artifact directory, artifact id or repository id")
    parser.add_argument("--device", default=_AUTO, help="device to probe, or 'auto'")
    parser.add_argument(
        "--all-submodels",
        action="store_true",
        help="compile every numeric submodel, not just the language model",
    )
    args = parser.parse_args(None if argv is None else list(argv))

    settings = load_settings()
    resolver = RegistryResolver(load_builtin_registry(), versions=installed_registry_versions())
    service = ProbeService(
        OpenVINOProbe(),
        resolver,
        LocalArtifactStore(settings.paths),
        settings,
    )
    outcome = service.probe_detailed(
        args.artifact,
        device=args.device,
        all_submodels=args.all_submodels,
    )
    document = outcome.model_dump(mode="json")
    document["role_matrix"] = outcome.role_matrix()
    print(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if outcome.ok else 1


# OpenVINO may start a multiprocessing forkserver whose child re-imports the main
# module; every executable path in this package therefore stays guarded
# (docs/MODEL_PIPELINE.md section 10 — observed for real during development).
if __name__ == "__main__":  # pragma: no cover - manual diagnostic entry point
    raise SystemExit(main())
