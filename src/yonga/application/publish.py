"""``yonga publish``: upload a prepared artifact with an honest model card.

Planning does every check ``SPEC.md`` section 21 requires, and all of them
before anything is uploaded:

* the destination is named explicitly and the logged-in account may write to it;
* an existing repository is only written to with ``--update``;
* the artifact is ready: validated, with at least one device that probed;
* every licence it descends from permits redistribution -- read fresh from the
  Hub for both the requested model and the converted source, because a
  derivative is bound by both;
* every file still hashes to what the manifest recorded, so the digests the
  card publishes are true.

The upload is the artifact's model files, the source's licence files, a
generated ``README.md`` and ``yonga-manifest.json``: the manifest with local
paths, the login name and the hostname anonymised, exactly as bug-report
bundles are. The artifact's own ``.yonga/`` directory -- repair backups,
validation records -- is never uploaded.
"""

from __future__ import annotations

import getpass
import re
import socket
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from yonga import __version__
from yonga.adapters import HubClient, HubPublisher
from yonga.application.artifact_refs import artifact_not_found
from yonga.application.artifacts import ArtifactStore
from yonga.application.model_card import render_model_card
from yonga.application.report import anonymise
from yonga.config.settings import Settings
from yonga.domain.artifacts import (
    MANIFEST_DIRNAME,
    ArtifactManifest,
    DeviceProbeReport,
    ProbeStatus,
)
from yonga.domain.diagnostics import (
    ConfigurationError,
    FailureCategory,
    YongaError,
    diagnostic,
)
from yonga.domain.models import ModelIdentity
from yonga.domain.publication import (
    DeviceEvidence,
    LicenseAssessment,
    LicenseClass,
    PublishFile,
    PublishPlan,
    classify_license,
)
from yonga.registry.resolver import RegistryResolver
from yonga.util.hashing import sha256_bytes, sha256_file
from yonga.util.redaction import environment_secrets, redact_text

__all__ = ["LICENSE_FILE_PATTERNS", "PublishService"]

#: Licence files copied from the source repository, so its terms travel with the weights.
LICENSE_FILE_PATTERNS: tuple[str, ...] = ("LICENSE*", "LICENCE*", "NOTICE*", "USE_POLICY*")

_CARD = "README.md"
_MANIFEST = "yonga-manifest.json"
_DEVICES = ("CPU", "GPU", "NPU")
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


def _refusal(code: str, title: str, message: str, *remediation: str) -> ConfigurationError:
    return ConfigurationError(
        diagnostic(
            code,
            FailureCategory.CONFIGURATION_ERROR,
            title,
            message,
            remediation=list(remediation),
        )
    )


class PublishService:
    """Plans and performs publications of prepared artifacts."""

    def __init__(
        self,
        *,
        store: ArtifactStore,
        hub: HubClient,
        publisher: HubPublisher,
        resolver: RegistryResolver,
        settings: Settings,
        anonymiser: Callable[[str], str] | None = None,
    ) -> None:
        """Bind the service; ``anonymiser`` defaults to this host's home, login and name."""
        self._store = store
        self._hub = hub
        self._publisher = publisher
        self._resolver = resolver
        self._settings = settings
        self._anonymise = anonymiser if anonymiser is not None else _local_anonymiser()

    # --- planning ---------------------------------------------------------------------

    def plan(
        self,
        reference: str,
        repo_id: str,
        *,
        private: bool = True,
        update: bool = False,
        license_reviewed: bool = False,
        dry_run: bool = False,
    ) -> PublishPlan:
        """Decide the whole publication, or refuse with the reason. Uploads nothing.

        A ``dry_run`` reports credential problems as warnings instead of refusing:
        previewing a card is useful before logging in, and nothing will be written.
        """
        if not _REPO_ID.match(repo_id):
            raise _refusal(
                "publish.repo_invalid",
                "Not a repository id",
                f"{repo_id!r} is not NAMESPACE/NAME.",
                "Name the destination explicitly, e.g. --repo your-name/model-openvino-int4",
            )
        if self._settings.offline:
            raise _refusal(
                "publish.offline",
                "Publishing needs the network",
                "yonga is in offline mode, and a publication is an upload.",
                "Drop --offline (or YONGA_OFFLINE) to publish.",
            )
        artifact_dir, manifest = self._ready_artifact(reference)
        warnings: list[str] = []
        try:
            self._check_credentials(repo_id)
        except YongaError as error:
            if not dry_run:
                raise
            warnings.append(f"A real upload would be refused: {error.diagnostic.message}")
        exists = self._check_destination(repo_id, update=update)
        licenses = self._licenses(manifest)
        self._check_licenses(licenses, reviewed=license_reviewed)
        devices = _device_evidence(manifest)
        files = self._model_files(artifact_dir, manifest, warnings)
        license_files = self._license_files(manifest, {f.path_in_repo for f in files}, warnings)
        card = render_model_card(
            manifest,
            repo_id=repo_id,
            licenses=licenses,
            devices=devices,
            files=[*files, *license_files],
            license_files=[f.path_in_repo for f in license_files],
            profile_device=self._profile_device(manifest.export.profile),
            license_reviewed=license_reviewed,
        )
        generated = {
            _CARD: self._anonymise(card),
            _MANIFEST: self._anonymise(manifest.to_json_text()),
        }
        generated_files = [
            PublishFile(
                path_in_repo=name,
                size=len(text.encode()),
                sha256=sha256_bytes(text.encode()),
            )
            for name, text in generated.items()
        ]
        if exists:
            warnings.append(
                f"{repo_id} already exists: files with the same names are replaced and its "
                "visibility is left as it is."
            )
        return PublishPlan(
            artifact_id=manifest.artifact_id,
            artifact_dir=artifact_dir,
            repo_id=repo_id,
            private=private,
            repo_exists=exists,
            licenses=licenses,
            license_reviewed=license_reviewed,
            devices=devices,
            repairs=[record.repair_id for record in manifest.repairs if record.applied],
            files=[*files, *license_files, *generated_files],
            generated=generated,
            warnings=warnings,
        )

    def _ready_artifact(self, reference: str) -> tuple[Path, ArtifactManifest]:
        found = self._store.find(reference)
        manifest = self._store.manifest_for(found) if found is not None else None
        if found is None or manifest is None:
            raise ConfigurationError(
                artifact_not_found(reference, command="publish", code="publish.artifact_not_found")
            )
        if not self._store.is_ready(found):
            raise _refusal(
                "publish.artifact_not_ready",
                "The artifact is not ready",
                "Only an artifact that validated and passed a device probe can be published.",
                f"Run 'yonga probe {manifest.artifact_id}' to see what fails.",
            )
        return found, manifest

    def _check_credentials(self, repo_id: str) -> None:
        identity = self._publisher.identity()
        if identity is None:
            raise YongaError(
                diagnostic(
                    "publish.hub.not_logged_in",
                    FailureCategory.GATED_MODEL_AUTH_REQUIRED,
                    "Not logged in to the Hugging Face Hub",
                    "Publishing needs a token with write access.",
                    remediation=["hf auth login", "or export HF_TOKEN with a write token"],
                )
            )
        if identity.can_write is False:
            raise YongaError(
                diagnostic(
                    "publish.hub.read_only_token",
                    FailureCategory.GATED_MODEL_AUTH_REQUIRED,
                    "The stored token is read-only",
                    f"The token for {identity.name} cannot create or change repositories.",
                    remediation=["Log in again with a write token: hf auth login"],
                )
            )
        namespace = repo_id.split("/", 1)[0]
        if not identity.may_write_to(namespace):
            raise YongaError(
                diagnostic(
                    "publish.hub.foreign_namespace",
                    FailureCategory.GATED_MODEL_AUTH_REQUIRED,
                    "That namespace is not yours",
                    f"{namespace} is neither {identity.name} nor one of its organisations.",
                    remediation=[f"Publish under your own account: --repo {identity.name}/..."],
                )
            )

    def _check_destination(self, repo_id: str, *, update: bool) -> bool:
        exists = self._publisher.repo_exists(repo_id)
        if exists and not update:
            raise _refusal(
                "publish.repo_exists",
                "The repository already exists",
                f"{repo_id} exists; yonga does not write into an existing repository by default.",
                "Pass --update to replace its files, or choose another --repo.",
            )
        return exists

    def _licenses(self, manifest: ArtifactManifest) -> list[LicenseAssessment]:
        sources = [("requested", manifest.requested_model), ("source", manifest.resolved_source)]
        seen: dict[str, str | None] = {}
        assessments: list[LicenseAssessment] = []
        for role, ref in sources:
            if ref.repo_id not in seen:
                identity = ModelIdentity(repo_id=ref.repo_id, revision=ref.resolved_revision)
                seen[ref.repo_id] = self._hub.inspect(identity, with_sizes=False).license
            name = seen[ref.repo_id]
            assessments.append(
                LicenseAssessment(
                    repo_id=ref.repo_id,
                    role=role,
                    license=name,
                    classification=classify_license(name),
                )
            )
        return assessments

    @staticmethod
    def _check_licenses(licenses: list[LicenseAssessment], *, reviewed: bool) -> None:
        for assessment in licenses:
            where = f"https://huggingface.co/{assessment.repo_id}"
            if assessment.classification is LicenseClass.RESTRICTED:
                raise _refusal(
                    "publish.license.restricted",
                    "The license forbids redistributing derivatives",
                    f"{assessment.repo_id} is licensed {assessment.license}, which does not "
                    "allow distributing a modified (quantized) version.",
                    f"Read the terms at {where}; keep this artifact local.",
                )
            if assessment.classification is LicenseClass.UNKNOWN and not reviewed:
                declared = assessment.license or "no license"
                raise _refusal(
                    "publish.license.unreviewed",
                    "The license needs a human to read it",
                    f"{assessment.repo_id} declares {declared!r}, which yonga cannot classify.",
                    f"Read it at {where}. If it permits redistributing a quantized "
                    "derivative, rerun with --license-reviewed.",
                )

    def _model_files(
        self, artifact_dir: Path, manifest: ArtifactManifest, warnings: list[str]
    ) -> list[PublishFile]:
        files: list[PublishFile] = []
        for record in manifest.files:
            relative = PurePosixPath(record.path)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise _refusal(
                    "publish.files_invalid",
                    "The manifest names a file outside the artifact",
                    f"{record.path!r} is not a path inside the artifact directory.",
                )
            if relative.parts[0] == MANIFEST_DIRNAME:
                continue
            if record.path in (_CARD, _MANIFEST):
                warnings.append(f"{record.path} is replaced by the generated one.")
                continue
            local = artifact_dir / record.path
            actual = sha256_file(local) if local.is_file() else None
            if actual is None or (record.sha256 is not None and actual != record.sha256):
                raise _refusal(
                    "publish.files_changed",
                    "The artifact changed after it was prepared",
                    f"{record.path} is missing or no longer matches the manifest's digest.",
                    "Re-prepare it with 'yonga prepare ... --force' before publishing.",
                )
            files.append(
                PublishFile(
                    path_in_repo=record.path,
                    local_path=local,
                    size=local.stat().st_size,
                    sha256=actual,
                )
            )
        if not files:
            raise _refusal(
                "publish.files_missing",
                "The manifest lists no files",
                "There is nothing to publish.",
            )
        return files

    def _license_files(
        self, manifest: ArtifactManifest, taken: set[str], warnings: list[str]
    ) -> list[PublishFile]:
        """The source repository's licence files, read from the Hub cache, never changed."""
        source = manifest.resolved_source
        identity = ModelIdentity(repo_id=source.repo_id, revision=source.resolved_revision)
        try:
            snapshot = self._hub.download(identity, allow_patterns=list(LICENSE_FILE_PATTERNS))
        except YongaError as error:
            warnings.append(
                f"The source's license files could not be fetched ({error.diagnostic.title}); "
                "the card names the license but does not include its text."
            )
            return []
        found: list[PublishFile] = []
        for path in sorted(snapshot.iterdir()) if snapshot.is_dir() else []:
            if not path.is_file() or path.name in taken:
                continue
            if not any(PurePosixPath(path.name).match(p) for p in LICENSE_FILE_PATTERNS):
                continue
            found.append(
                PublishFile(
                    path_in_repo=path.name,
                    local_path=path,
                    size=path.stat().st_size,
                    sha256=sha256_file(path),
                )
            )
        return found

    def _profile_device(self, profile_id: str) -> str | None:
        for rule in self._resolver.registry.profiles:
            if rule.id == profile_id:
                return rule.device
        return None

    # --- output -----------------------------------------------------------------------

    def write_preview(self, plan: PublishPlan, directory: Path) -> list[Path]:
        """Write the card, the manifest and the upload list to ``directory``. No weights."""
        directory.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for name, text in plan.generated.items():
            target = directory / name
            target.write_text(text, encoding="utf-8")
            written.append(target)
        listing = directory / "upload.txt"
        listing.write_text(
            "".join(f"{f.sha256}  {f.size:>14}  {f.path_in_repo}\n" for f in plan.files),
            encoding="utf-8",
        )
        written.append(listing)
        return written

    def publish(self, plan: PublishPlan) -> str:
        """Upload the plan in one commit and return the commit URL."""
        files = [(f.path_in_repo, f.local_path) for f in plan.files if f.local_path is not None]
        return self._publisher.publish(
            plan.repo_id,
            private=plan.private,
            files=files,
            generated={name: text.encode() for name, text in plan.generated.items()},
            message=f"Upload {plan.artifact_id}, prepared with yonga {__version__}",
        )


def _device_evidence(manifest: ArtifactManifest) -> list[DeviceEvidence]:
    names = list(_DEVICES)
    names += [name for name in manifest.device_probes if name not in names]
    return [_evidence(name, manifest.device_probes.get(name)) for name in names]


def _evidence(device: str, report: DeviceProbeReport | None) -> DeviceEvidence:
    if report is None or report.status.value in ("not_attempted", "skipped"):
        return DeviceEvidence(device=device, status="not_tested")
    when = report.attempted_at.date().isoformat() if report.attempted_at else None
    if report.ok:
        mine = [result for result in report.submodels if result.device == device]
        compiled = sum(1 for result in mine if result.ok)
        deferred = sum(1 for result in mine if result.status is ProbeStatus.DEFERRED)
        detail = f"{compiled} submodel(s) compiled"
        if deferred:
            detail += f", {deferred} verified through the GenAI pipeline instead"
        smoke = report.smoke_test
        if smoke is not None and smoke.ok:
            detail += f"; generation smoke test passed ({smoke.message or 'ok'})"
        return DeviceEvidence(device=device, status="ok", detail=detail, attempted_at=when)
    if report.status is ProbeStatus.INCONCLUSIVE:
        # Not a pass and not a failure: the pipeline build that decides was
        # skipped or stopped, so nothing is claimed in either direction.
        detail = "the probe stopped before the GenAI pipeline settled it"
        return DeviceEvidence(
            device=device, status="inconclusive", detail=detail, attempted_at=when
        )
    reason = report.diagnostic
    detail = f"`{reason.code}`: {reason.title}" if reason is not None else "the probe failed"
    return DeviceEvidence(device=device, status="failed", detail=detail, attempted_at=when)


def _local_anonymiser() -> Callable[[str], str]:
    """Anonymise this host's home directory, login and hostname, and redact secrets."""
    home = str(Path.home())
    try:
        user: str | None = getpass.getuser()
    except Exception:  # getpass raises several unrelated types on odd hosts
        user = None
    try:
        host: str | None = socket.gethostname() or None
    except OSError:
        host = None
    secrets = environment_secrets()

    def clean(text: str) -> str:
        return anonymise(redact_text(text, extra_secrets=secrets), home=home, user=user, host=host)

    return clean
