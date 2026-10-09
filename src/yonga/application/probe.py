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
import math
import shlex
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
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
    failure_text,
    manifest_path,
)
from yonga.domain.compatibility import DeviceStatus
from yonga.domain.diagnostics import (
    CancelledError,
    ConfigurationError,
    Diagnostic,
    FailureCategory,
    Severity,
    YongaError,
    diagnostic,
)
from yonga.domain.hardware import normalise_device
from yonga.domain.models import ModelIdentity, ModelMetadata, PipelineKind
from yonga.util.redaction import redact_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yonga.adapters import RuntimeProbe
    from yonga.application.prepare import SmokeTester
    from yonga.registry.resolver import RegistryResolver

__all__ = [
    "DEFAULT_DEVICE_ORDER",
    "TOKENIZER_DEVICE",
    "ArtifactStoreLike",
    "IsolatedSmokeTester",
    "LocalArtifactStore",
    "ProbeOutcome",
    "ProbeService",
    "check_probe_options",
    "guard_smoke_test",
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
    ProbeStatus.DEFERRED: 1,
    ProbeStatus.SKIPPED: 1,
    ProbeStatus.NOT_ATTEMPTED: 2,
    ProbeStatus.INCONCLUSIVE: 2,
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


@runtime_checkable
class IsolatedSmokeTester(Protocol):
    """Runs the generation smoke test in a child process it can stop.

    ``yonga probe --timeout`` depends on this port: a native NPU compile cannot
    be interrupted in-process, so a bounded pipeline build has to run where a
    signal can end it. Implemented by
    :class:`~yonga.adapters.smoke_child.ChildProcessSmokeTester`.
    """

    def smoke_test(
        self,
        artifact_dir: Path,
        *,
        device: str,
        pipeline: PipelineKind,
        properties: Mapping[str, str],
        prompt: str,
        max_new_tokens: int,
        timeout: float,
        ctx: JobContext | None = None,
    ) -> ProbeResult:
        """Build the pipeline and generate, giving up after ``timeout`` seconds.

        Returns ``ok`` or ``failed`` as :class:`~yonga.application.prepare.SmokeTester`
        does, with a failure's raw text left for the registry to classify, and
        :attr:`~yonga.domain.artifacts.ProbeStatus.INCONCLUSIVE` when the time
        limit stopped the build. Raises
        :class:`~yonga.domain.diagnostics.CancelledError` when the job is
        cancelled while the child runs.
        """
        ...


def check_probe_options(*, quick: bool, timeout: float | None) -> None:
    """Reject a ``--quick``/``--timeout`` combination that cannot mean anything.

    ``timeout`` must be a finite number of seconds above zero, and it cannot be
    combined with ``quick``: a quick probe never builds the pipeline the timeout
    would bound.
    """
    if timeout is not None and not (math.isfinite(timeout) and timeout > 0):
        raise ConfigurationError(
            diagnostic(
                "probe.timeout.invalid",
                FailureCategory.CONFIGURATION_ERROR,
                "The probe timeout must be a positive number of seconds",
                f"--timeout {timeout:g} cannot bound a pipeline build.",
                remediation=["Pass a number of seconds above zero, for example --timeout 600."],
                context={"timeout": f"{timeout:g}"},
            )
        )
    if quick and timeout is not None:
        raise ConfigurationError(
            diagnostic(
                "probe.options.conflict",
                FailureCategory.CONFIGURATION_ERROR,
                "--quick and --timeout cannot be combined",
                "A quick probe never builds the GenAI pipeline, so there is nothing for "
                "--timeout to bound.",
                remediation=[
                    "Drop --timeout for the raw compile only.",
                    "Drop --quick to settle a deferred device within the timeout.",
                ],
            )
        )


def guard_smoke_test(device: str, run: Callable[[], ProbeResult]) -> ProbeResult:
    """Run one smoke test and turn whatever it raises into a failed result.

    A typed failure keeps its diagnostic; anything else keeps its redacted
    text for the registry. Cancellation is not a result and propagates. Shared
    by the in-process probe and the child process, so both report a failure
    the same way.
    """
    try:
        return run()
    except CancelledError:
        raise
    except YongaError as error:
        return ProbeResult(
            device=device,
            status=ProbeStatus.FAILED,
            diagnostic=error.diagnostic,
            message=error.diagnostic.message,
        )
    except Exception as exc:
        return ProbeResult(
            device=device,
            status=ProbeStatus.FAILED,
            message=redact_text(f"{type(exc).__name__}: {exc}"),
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
    quick: bool = False
    """``--quick``: no GenAI pipeline was built, so a deferred device is inconclusive."""
    timeout_seconds: float | None = None
    """``--timeout``: the bound on a pipeline build, which then ran in a child process."""
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
    def inconclusive(self) -> bool:
        """True when no device passed, none failed outright, and one is inconclusive.

        That is a ``--quick`` or ``--timeout`` probe that settled nothing: a
        warning, not a failure, and never a pass.
        """
        if self.ok:
            return False
        statuses = {report.status for report in self.reports}
        return ProbeStatus.INCONCLUSIVE in statuses and ProbeStatus.FAILED not in statuses

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

    A :attr:`ProbeStatus.DEFERRED` raw compile reads as the device's verdict:
    deferral means the pipeline decides, so a role the GenAI pipeline ran is
    ``ok``, one it could not run is ``failed``, and one whose pipeline was never
    built or was stopped at ``--timeout`` is ``inconclusive``. The pipeline smoke
    test itself covers the whole model rather than one role, so it has no row of
    its own.
    """
    matrix: dict[str, dict[str, str]] = {}
    for report in reports:
        for result in report.submodels:
            status = result.status
            if status is ProbeStatus.DEFERRED:
                status = _deferred_verdict(report)
            role = (result.role or SubmodelRole.OTHER).value
            per_device = matrix.setdefault(role, {})
            previous = per_device.get(result.device)
            if previous is None:
                per_device[result.device] = status.value
                continue
            previous_rank = _STATUS_SEVERITY.get(ProbeStatus(previous), 0)
            if _STATUS_SEVERITY[status] > previous_rank:
                per_device[result.device] = status.value
    return matrix


def _deferred_verdict(report: DeviceProbeReport) -> ProbeStatus:
    """What a deferred raw compile amounts to, given its device's report."""
    if report.status in (ProbeStatus.OK, ProbeStatus.INCONCLUSIVE):
        return report.status
    smoke = report.smoke_test
    if smoke is None or smoke.status is ProbeStatus.INCONCLUSIVE:
        # The device failed on another submodel, and this one was never settled:
        # ``--quick`` built no pipeline, or ``--timeout`` stopped it.
        return ProbeStatus.INCONCLUSIVE
    return ProbeStatus.FAILED


# --- the service ----------------------------------------------------------------------


class ProbeService:
    """Reads, compiles and classifies every submodel of a prepared artifact."""

    def __init__(
        self,
        runtime_probe: RuntimeProbe,
        registry_resolver: RegistryResolver,
        artifact_store: ArtifactStoreLike | None = None,
        settings: Settings | None = None,
        smoke_tester: SmokeTester | None = None,
        isolated_smoke_tester: IsolatedSmokeTester | None = None,
    ) -> None:
        """Wire the service to its adapters.

        ``artifact_store`` defaults to a :class:`LocalArtifactStore` over the
        settings' managed artifacts directory, so the service is usable with a
        bare directory path and no catalogue.

        ``smoke_tester`` builds a real GenAI pipeline. It is consulted only to
        settle a raw compile the registry marks ``defer_to_pipeline`` -- a probe
        stays a cheap compile check otherwise. Without one, a deferred failure
        leaves the device unproven rather than passed.

        ``isolated_smoke_tester`` builds the same pipeline in a child process,
        and is used instead of ``smoke_tester`` only when a probe is given a
        ``timeout``: that is the only way to stop a native compile part-way.
        """
        self._runtime = runtime_probe
        self._resolver = registry_resolver
        self._settings = settings
        self._smoke = smoke_tester
        self._isolated = isolated_smoke_tester
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
        quick: bool = False,
        timeout: float | None = None,
        ctx: JobContext | None = None,
    ) -> list[DeviceProbeReport]:
        """Probe an artifact and return one report per device that was tried.

        ``device`` is a device name or ``auto``. With ``all_submodels`` every
        numeric IR in the artifact is compiled; by default only the language
        model is, which is the cheap answer to "can this run here at all".
        Tokenizers are always probed, always on CPU.

        A raw compile the registry defers to the GenAI pipeline is settled by
        building that pipeline, which for a large model on the NPU takes many
        minutes. ``quick`` never builds it and reports such a device
        :attr:`~yonga.domain.artifacts.ProbeStatus.INCONCLUSIVE`; ``timeout``
        builds it in a child process and stops it after that many seconds,
        which is inconclusive too. The two cannot be combined.
        """
        return self.probe_detailed(
            artifact_or_model,
            device=device,
            all_submodels=all_submodels,
            quick=quick,
            timeout=timeout,
            ctx=ctx,
        ).reports

    def probe_detailed(
        self,
        artifact_or_model: str | Path,
        *,
        device: str = _AUTO,
        all_submodels: bool = False,
        quick: bool = False,
        timeout: float | None = None,
        ctx: JobContext | None = None,
    ) -> ProbeOutcome:
        """Probe an artifact and return the reports together with the IR inventory.

        ``quick`` and ``timeout`` are described on :meth:`probe`.
        """
        check_probe_options(quick=quick, timeout=timeout)
        if timeout is not None and self._isolated is None:
            raise ConfigurationError(
                diagnostic(
                    "probe.timeout.unavailable",
                    FailureCategory.CONFIGURATION_ERROR,
                    "This probe cannot bound a pipeline build",
                    "--timeout needs a smoke tester that runs in a child process, and none "
                    "is configured here.",
                    remediation=["Run the probe without --timeout, or with --quick."],
                )
            )
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
                    artifact_dir=artifact_dir,
                    targets=numeric,
                    tokenizer=tokenizer,
                    manifest=manifest,
                    reference=str(artifact_or_model),
                    all_submodels=all_submodels,
                    quick=quick,
                    timeout=timeout,
                    ctx=ctx,
                )
            )

        return ProbeOutcome(
            artifact_dir=artifact_dir,
            requested_device=(device or _AUTO).strip() or _AUTO,
            quick=quick,
            timeout_seconds=timeout,
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
        artifact_dir: Path,
        targets: Sequence[IRDescription],
        tokenizer: ProbeResult | None,
        manifest: ArtifactManifest | None,
        reference: str,
        all_submodels: bool,
        quick: bool,
        timeout: float | None,
        ctx: JobContext | None,
    ) -> DeviceProbeReport:
        """Compile every numeric target on one device and build its report.

        A raw failure the registry marks ``defer_to_pipeline`` is recorded as
        :attr:`ProbeStatus.DEFERRED` and settled by building the GenAI pipeline:
        the device passes only if that pipeline generates. With no pipeline
        evidence the device is reported failed, never passed by default -- except
        when the probe itself chose not to wait for that evidence (``quick``, or a
        build stopped at ``timeout``): then it is
        :attr:`ProbeStatus.INCONCLUSIVE`, unless another submodel already failed.
        """
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

        deferred = [result for result in results if result.status is ProbeStatus.DEFERRED]
        if not deferred:
            return DeviceProbeReport(
                device=device,
                status=_report_status(results),
                submodels=results,
                diagnostic=_first_diagnostic(results),
                attempted_at=_now(),
            )

        others = [result for result in results if result.status is not ProbeStatus.DEFERRED]
        smoke = (
            None
            if quick
            else self._settle_on_pipeline(
                artifact_dir, device, manifest=manifest, timeout=timeout, ctx=ctx
            )
        )
        if smoke is None or smoke.status is ProbeStatus.INCONCLUSIVE:
            if _report_status(others) is ProbeStatus.FAILED:
                status = ProbeStatus.FAILED
                verdict = _first_diagnostic(others)
            else:
                status = ProbeStatus.INCONCLUSIVE
                verdict = _inconclusive(
                    device,
                    deferred[0],
                    reference=reference,
                    all_submodels=all_submodels,
                    timeout=None if smoke is None else timeout,
                )
        elif smoke.status is ProbeStatus.OK:
            status = _report_status([*others, smoke])
            verdict = _first_diagnostic(others)
        elif smoke.status is ProbeStatus.FAILED:
            status = ProbeStatus.FAILED
            verdict = smoke.diagnostic or deferred[0].diagnostic
        else:
            status = ProbeStatus.FAILED
            first = deferred[0].diagnostic
            verdict = (
                None
                if first is None
                else first.model_copy(
                    update={
                        "message": (
                            f"{first.message} The GenAI pipeline that settles this could not "
                            f"be built ({smoke.message or smoke.status.value}), so {device} "
                            "support is unproven."
                        )
                    }
                )
            )
        return DeviceProbeReport(
            device=device,
            status=status,
            submodels=results,
            smoke_test=smoke,
            diagnostic=verdict,
            attempted_at=_now(),
        )

    def _settle_on_pipeline(
        self,
        artifact_dir: Path,
        device: str,
        *,
        manifest: ArtifactManifest | None,
        timeout: float | None,
        ctx: JobContext | None,
    ) -> ProbeResult:
        """Build the GenAI pipeline on ``device`` and generate a few tokens.

        The manifest's recommended runtime properties are applied only when they
        were recorded for this same device: a GPU artifact's
        ``ATTENTION_BACKEND=SDPA`` means nothing to the NPU, whose pipeline falls
        back to GenAI's own static-shape defaults.

        With a ``timeout`` the build runs in a child process, and one stopped at
        the limit comes back :attr:`ProbeStatus.INCONCLUSIVE`. Either way a
        failure is classified by the registry from its raw text.
        """
        from yonga.application.prepare import SMOKE_MAX_NEW_TOKENS, SMOKE_PROMPT

        pipeline = PipelineKind.UNKNOWN if manifest is None else manifest.pipeline
        runtime = None if manifest is None else manifest.recommended_runtime
        properties: dict[str, str] = {}
        if runtime is not None and normalise_device(runtime.device) == normalise_device(device):
            properties = dict(runtime.properties)

        isolated, smoke = self._isolated, self._smoke
        if timeout is not None and isolated is not None:
            limit = timeout

            def run() -> ProbeResult:
                return isolated.smoke_test(
                    artifact_dir,
                    device=device,
                    pipeline=pipeline,
                    properties=properties,
                    prompt=SMOKE_PROMPT,
                    max_new_tokens=SMOKE_MAX_NEW_TOKENS,
                    timeout=limit,
                    ctx=ctx,
                )

        elif timeout is None and smoke is not None:

            def run() -> ProbeResult:
                return smoke.smoke_test(
                    artifact_dir,
                    device=device,
                    pipeline=pipeline,
                    properties=properties,
                    prompt=SMOKE_PROMPT,
                    max_new_tokens=SMOKE_MAX_NEW_TOKENS,
                )

        else:
            return ProbeResult(
                device=device,
                status=ProbeStatus.SKIPPED,
                message="no GenAI pipeline smoke tester is configured",
            )

        weights = _weight_bytes(artifact_dir)
        notice = f"Building the GenAI pipeline on {device} to settle the probe"
        if timeout is not None:
            notice += f", in a child process stopped after {timeout:g} s"
        if weights >= _SLOW_PIPELINE_BUILD_BYTES:
            # Measured on the reference host: a 9B INT4 model's NPU build ran
            # past 15 minutes at ~9 GiB RSS. Say so before the wait, not after.
            notice += (
                f" -- this compiles {weights / 2**30:.1f} GiB of weights and can take many minutes"
            )
            if timeout is None:
                notice += "; --timeout SECONDS bounds the wait and --quick skips the build"
        self._emit(ctx, notice, device=device)
        result = guard_smoke_test(device, run)
        if result.status is ProbeStatus.FAILED:
            # Registry first, even over a diagnostic the adapter attached: a
            # pipeline load failure arrives generic, and only the registry
            # recognises the compiler signature in its raw text.
            adapter = result.diagnostic
            raw = failure_text(result)
            if raw:
                classified = self._classify(raw, device=device, manifest=manifest, ctx=ctx)
                if (
                    classified.category is not FailureCategory.UNKNOWN_RUNTIME_ERROR
                    or adapter is None
                ):
                    result = result.model_copy(update={"diagnostic": classified})
        self._emit_result(result, ctx=ctx)
        return result

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
        # Re-labelled before it is reported, so a raw failure the pipeline is
        # about to overrule -- or that --quick leaves open -- is not shown as an error.
        result = _defer_if_marked(result, device)
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
        if result.ok:
            severity = Severity.OK
        elif result.status is ProbeStatus.DEFERRED:
            severity = Severity.INFO
        elif result.status is ProbeStatus.INCONCLUSIVE:
            severity = Severity.WARNING
        else:
            severity = Severity.ERROR
        role = (result.role or SubmodelRole.OTHER).value
        # A result without a role is the pipeline smoke test, which covers the whole model.
        label = role if result.role is not None else "pipeline smoke test"
        ctx.emit(
            ProgressKind.PROBE,
            f"{label} on {result.device}: {result.status.value}",
            device=result.device,
            status=result.status.value,
            severity=severity,
            diagnostic=result.diagnostic,
            role=role,
        )


#: Above this much weight data a pipeline build is slow enough to warn about.
_SLOW_PIPELINE_BUILD_BYTES: int = 2 * 2**30


def _weight_bytes(artifact_dir: Path) -> int:
    """Total size of the IR weight files at the top level of an artifact."""
    total = 0
    with suppress(OSError):
        for path in artifact_dir.glob("*.bin"):
            with suppress(OSError):
                total += path.stat().st_size
    return total


def _defer_if_marked(result: ProbeResult, device: str) -> ProbeResult:
    """Re-label a raw failure the registry says the GenAI pipeline must decide."""
    diagnosis = result.diagnostic
    if result.status is not ProbeStatus.FAILED or diagnosis is None:
        return result
    if not diagnosis.defer_to_pipeline:
        return result
    return result.model_copy(
        update={
            "status": ProbeStatus.DEFERRED,
            "message": (
                f"A raw compile on {device} is not conclusive for this graph; the GenAI "
                f"pipeline decides. Raw error: {result.message or diagnosis.message}"
            ),
        }
    )


def _inconclusive(
    device: str,
    deferred: ProbeResult,
    *,
    reference: str,
    all_submodels: bool,
    timeout: float | None,
) -> Diagnostic:
    """Explain an unsettled deferred compile, and how to settle it.

    A warning, not a failure: nothing was learned about the model on ``device``.
    ``timeout`` is the limit that stopped the pipeline build, or ``None`` when
    ``--quick`` never started one. The registry's diagnostic for the raw compile
    is deliberately not reused -- its fallback advice would read as a verdict
    the probe does not have. The commands it suggests repeat the probe as it was
    asked for, quoted so they can be pasted into a shell.
    """
    command = f"yonga probe {shlex.quote(reference)} --device {shlex.quote(device.lower())}"
    if all_submodels:
        command += " --all-submodels"
    context = {"device": device}
    if deferred.diagnostic is not None:
        context["deferred_code"] = deferred.diagnostic.code
    if timeout is None:
        return diagnostic(
            "probe.inconclusive.quick",
            FailureCategory.UNKNOWN_RUNTIME_ERROR,
            f"{device} is inconclusive: the raw compile was deferred",
            (
                f"A raw compile on {device} is not conclusive for this graph, and the GenAI "
                "pipeline that decides was not built because of --quick. Nothing is known yet "
                f"about whether this model runs on {device}."
            ),
            severity=Severity.WARNING,
            remediation=[
                f"Settle it by building the pipeline: {command}",
                f"Or bound the wait: {command} --timeout SECONDS",
            ],
            context=context,
        )
    context["timeout"] = f"{timeout:g}"
    return diagnostic(
        "probe.inconclusive.timeout",
        FailureCategory.UNKNOWN_RUNTIME_ERROR,
        f"{device} is inconclusive: the pipeline build hit --timeout",
        (
            f"A raw compile on {device} is not conclusive for this graph, and the GenAI "
            f"pipeline that decides did not finish building within {timeout:g} s, so it was "
            "stopped. This is not a compiler verdict: a large model can take many minutes "
            f"to compile for {device}."
        ),
        severity=Severity.WARNING,
        remediation=[
            f"Allow it longer: {command} --timeout {max(timeout * 4, 600):g}",
            f"Or wait for it without a limit: {command}",
        ],
        context=context,
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

    ``--quick`` and ``--timeout`` behave as they do for ``yonga probe``. Exit 0
    when a device passed, 2 when nothing failed but something is inconclusive,
    1 otherwise. An invalid ``--quick``/``--timeout`` combination is reported on
    stderr and exits 70, as ``yonga probe`` does, so it is never mistaken for an
    inconclusive probe; other argparse usage errors exit 2, as argparse's do.
    """
    import argparse

    from yonga.adapters.openvino_probe import OpenVINOProbe
    from yonga.adapters.smoke_child import ChildProcessSmokeTester, smoke_child_runner
    from yonga.application.wiring import GenAISmokeTester
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
    parser.add_argument(
        "--quick",
        action="store_true",
        help="never build the GenAI pipeline; a deferred raw compile is inconclusive",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "bound the GenAI pipeline build that settles a deferred compile: run it in a "
            "child process and stop it after SECONDS (the device is then inconclusive)"
        ),
    )
    args = parser.parse_args(None if argv is None else list(argv))
    try:
        check_probe_options(quick=args.quick, timeout=args.timeout)
    except ConfigurationError as error:
        found = error.diagnostic
        print(f"{parser.prog}: {found.code}: {found.title}. {found.message}", file=sys.stderr)
        return int(error.exit_code())

    settings = load_settings()
    resolver = RegistryResolver(load_builtin_registry(), versions=installed_registry_versions())
    service = ProbeService(
        OpenVINOProbe(),
        resolver,
        LocalArtifactStore(settings.paths),
        settings,
        smoke_tester=GenAISmokeTester(),
        isolated_smoke_tester=ChildProcessSmokeTester(smoke_child_runner()),
    )
    outcome = service.probe_detailed(
        args.artifact,
        device=args.device,
        all_submodels=args.all_submodels,
        quick=args.quick,
        timeout=args.timeout,
    )
    document = outcome.model_dump(mode="json")
    document["role_matrix"] = outcome.role_matrix()
    document["inconclusive"] = outcome.inconclusive
    print(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False))
    if outcome.ok:
        return 0
    return 2 if outcome.inconclusive else 1


# OpenVINO may start a multiprocessing forkserver whose child re-imports the main
# module; every executable path in this package therefore stays guarded
# (docs/MODEL_PIPELINE.md section 10 — observed for real during development).
if __name__ == "__main__":  # pragma: no cover - manual diagnostic entry point
    raise SystemExit(main())
