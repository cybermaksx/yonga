"""Planning, preflight, confirmation and reuse behaviour of ``PrepareService``.

This module also defines the adapter fakes the other preparation tests import.
They live here rather than in a helper module because every test file in this
group is owned by the same agent and pytest imports test modules as ordinary
modules (``tests/`` carries an ``__init__.py``).

The numbers in :data:`SOURCE_WEIGHT_BYTES` and friends are the real figures
measured on the reference host and recorded in ``HOST_FACTS.md``; the tmpfs
decision test asserts against them exactly rather than against round numbers.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from yonga.adapters import (
    ExportEnvironments,
    Exporter,
    FilesystemProbe,
    HardwareProbe,
    HubClient,
    RuntimeProbe,
)
from yonga.adapters.filesystem import system_temp_dir
from yonga.application.artifacts import ArtifactStore
from yonga.application.jobs import JobContext, open_job
from yonga.application.prepare import (
    ConfirmCallback,
    PreparationOutcome,
    PrepareService,
    SmokeTester,
)
from yonga.application.progress import CollectingSink, ProgressKind
from yonga.config.settings import Settings
from yonga.domain.artifacts import (
    ArtifactManifest,
    IRDescription,
    ProbeResult,
    ProbeStatus,
    SubmodelRole,
)
from yonga.domain.diagnostics import (
    CancelledError,
    ExitCode,
    ExportError,
    PreflightError,
    RemoteCodeRequiredError,
    SourceResolutionError,
    ValidationError,
)
from yonga.domain.hardware import (
    DeviceKind,
    FilesystemInfo,
    HardwareInventory,
    HostInfo,
    HuggingFaceAuthInfo,
    MemoryInfo,
    OpenVINODevice,
)
from yonga.domain.models import (
    FormatObservation,
    ModelFormat,
    ModelIdentity,
    ModelMetadata,
    PipelineKind,
    RepoFile,
    SourceRelationship,
)
from yonga.domain.plans import (
    CONFIRMATION_LARGE_DOWNLOAD,
    CONFIRMATION_REMOTE_CODE,
    CONFIRMATION_THIRD_PARTY_DERIVATIVE,
    ConfirmationReason,
    ExportRequest,
    ExportResult,
    PreparationPlan,
)
from yonga.registry.loader import load_builtin_documents, load_registry_from_documents
from yonga.registry.resolver import RegistryResolver
from yonga.registry.schema import Registry

# --- facts measured on the reference host (HOST_FACTS.md) ----------------------------

GIB = 1024**3

#: ``/tmp`` on the reference host: a 16 GiB tmpfs.
TMPFS_FREE_BYTES = 16 * GIB

#: ``/`` on the reference host: ext4 with roughly 750 GiB free.
DISK_FREE_BYTES = 750 * GIB

#: Sum of the four safetensors shards of the Qwen3.5 acceptance fixture.
SOURCE_WEIGHT_BYTES = 18_819_721_040

#: Total bytes of that repository, weights plus assets.
REPO_TOTAL_BYTES = 18_842_727_977

#: ``max(20 GiB, weights * 2.5)`` for the fixture above.
REQUIRED_TEMP_BYTES = 47_049_302_600

#: The default safety margin (5 GiB).
SAFETY_MARGIN_BYTES = 5 * GIB

GGUF_REPO = "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"
GGUF_SHA = "0a41c68809d375475f954be12ba7c40efa56c2a9"
SAFETENSORS_REPO = "nDimensional/Qwen3.5-9B-Uncensored-Safetensors"
SAFETENSORS_SHA = "4623e504074fc2489f3af98dc3d68b8392bbc21f"
OPENVINO_REPO = "OpenVINO/Qwen3.5-9B-int4-ov"
OPENVINO_SHA = "3f31fc1609edb0cbc8b131bc4c291f1b15cb0676"

#: Versions the registry resolver is judged against, pinned so a machine without
#: OpenVINO installed still exercises the same rules as the reference host.
REGISTRY_VERSIONS: dict[str, str | None] = {
    "openvino": "2026.4.0",
    "openvino_genai": "2026.4.0.0",
    "openvino_tokenizers": "2026.4.0.0",
    "optimum": "2.3.0",
    "optimum_intel": "2.2.0",
    "nncf": "3.4.0",
    "transformers": "5.5.4",
    "torch": "2.14.0",
    "huggingface_hub": "1.21.0",
    "python": "3.11.0",
}

#: Installed versions as the exporter adapter reports them on a host that can
#: actually run the Qwen3.5 export: ``transformers`` satisfies the registry's
#: ``==5.2.*`` pin, matching the ``transformers 5.2.1`` the CLI tests' healthy
#: inventory reports. This is :class:`FakeExporter`'s default because an unmet
#: pin now *stops* preparation at preflight, so every test that is about
#: something else must start from an exporter that is allowed to run.
INSTALLED_EXPORT_VERSIONS: dict[str, str | None] = {
    "openvino": "2026.4.0",
    "optimum": "2.3.0",
    "optimum-intel": "2.2.0",
    "nncf": "3.4.0",
    "transformers": "5.2.1",
    "torch": "2.14.0",
}

#: The mismatch the reference host really has: ``transformers 5.5.4`` against the
#: registry's ``==5.2.*``. Verified by hand — 5.5.4 loads ``Qwen3_5Config`` and
#: reports ``model_type: qwen3_5``, so the architecture looks supported, but
#: optimum-intel 2.2.0's export patcher imports ``Qwen3_5DynamicCache``, which
#: 5.5.4 no longer exposes, and the conversion dies minutes into the job.
MISMATCHED_EXPORT_VERSIONS: dict[str, str | None] = {
    **INSTALLED_EXPORT_VERSIONS,
    "transformers": "5.5.4",
}

#: The authentic mixed-case defect every current export writes.
BAD_PRECISION = 'precision="STRiNG"'

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

MODEL_XML = (
    '<?xml version="1.0"?>\n<net name="Model0" version="11">\n  <layers/>\n  <edges/>\n</net>\n'
)


# --- adapter fakes -------------------------------------------------------------------


@dataclass
class FakeHub:
    """In-memory :class:`~yonga.adapters.HubClient`."""

    repos: dict[str, ModelMetadata] = field(default_factory=dict)
    snapshots: dict[str, Path] = field(default_factory=dict)
    downloaded: list[str] = field(default_factory=list)
    inspected: list[str] = field(default_factory=list)

    def resolve_revision(self, model: ModelIdentity) -> str:
        """Return the pinned commit SHA recorded for this repository."""
        return self.repos[model.repo_id].resolved_revision or "main"

    def inspect(self, model: ModelIdentity, *, with_sizes: bool = True) -> ModelMetadata:
        """Return the recorded metadata, or raise the way the real adapter would."""
        self.inspected.append(model.repo_id)
        try:
            return self.repos[model.repo_id]
        except KeyError:
            raise SourceResolutionError(f"{model.repo_id} does not exist") from None

    def download(
        self,
        source: ModelIdentity,
        *,
        ctx: JobContext | None = None,
        allow_patterns: Sequence[str] | None = None,
        ignore_patterns: Sequence[str] | None = None,
    ) -> Path:
        """Record the download and hand back the prepared snapshot directory."""
        self.downloaded.append(source.repo_id)
        return self.snapshots[source.repo_id]

    def auth_info(self) -> HuggingFaceAuthInfo:
        """Report a logged-out, online session."""
        return HuggingFaceAuthInfo(logged_in=False)


@dataclass
class FakeExporter:
    """An :class:`~yonga.adapters.Exporter` that writes fixture files instead of converting."""

    versions: dict[str, str | None] = field(default_factory=lambda: dict(INSTALLED_EXPORT_VERSIONS))
    succeed: bool = True
    stderr_tail: list[str] = field(default_factory=list)
    write_tokenizer: bool = True
    broken_tokenizer: bool = True
    #: A real VLM export writes one IR per submodel; the language model and the
    #: text embeddings are the two roles validation insists on.
    numeric_models: tuple[str, ...] = (
        "openvino_language_model",
        "openvino_text_embeddings_model",
    )
    requests: list[ExportRequest] = field(default_factory=list)
    temp_env_seen: list[dict[str, str]] = field(default_factory=list)

    def requirements(self) -> dict[str, str | None]:
        """Installed versions of the packages the export depends on."""
        return dict(self.versions)

    def export(self, request: ExportRequest, ctx: JobContext) -> ExportResult:
        """Write a plausible export into ``request.output_path``."""
        self.requests.append(request)
        self.temp_env_seen.append(dict(ctx.temp_env))
        output = request.output_path
        output.mkdir(parents=True, exist_ok=True)
        argv = [
            "python",
            "-m",
            "optimum.commands.optimum_cli",
            "export",
            "openvino",
            "-m",
            str(request.source_path),
            "--weight-format",
            request.profile.quantization.weight_format,
            str(output),
        ]
        if not self.succeed:
            return ExportResult(
                success=False,
                argv=argv,
                returncode=1,
                duration_seconds=0.2,
                stderr_tail=list(self.stderr_tail),
                package_versions={k: v for k, v in self.versions.items() if v},
            )
        for stem in self.numeric_models:
            (output / f"{stem}.xml").write_text(MODEL_XML, encoding="utf-8")
            (output / f"{stem}.bin").write_bytes(b"\x00" * 32)
        (output / "config.json").write_text('{"model_type": "qwen3_5"}\n', encoding="utf-8")
        if self.write_tokenizer:
            write_tokenizer_pair(output, broken=self.broken_tokenizer)
        return ExportResult(
            success=True,
            argv=argv,
            returncode=0,
            duration_seconds=1.5,
            stdout_tail=["Export finished"],
            package_versions={k: v for k, v in self.versions.items() if v},
        )


@dataclass
class FakeRuntimeProbe:
    """A :class:`~yonga.adapters.RuntimeProbe` driven by the files on disk.

    ``read_tokenizer`` looks at the real XML text, so a test cannot pass by
    pretending a repair happened: the tokenizer only becomes valid once the
    ``STRiNG`` defect is actually gone from the staged files.
    """

    devices: list[str] = field(default_factory=lambda: ["CPU", "GPU", "NPU"])
    compile_failures: dict[str, str] = field(default_factory=dict)
    unreadable: set[str] = field(default_factory=set)
    compiled: list[tuple[str, str]] = field(default_factory=list)
    tokenizer_reads: int = 0

    def available_devices(self) -> list[str]:
        """Devices OpenVINO would report."""
        return list(self.devices)

    def read_ir(self, path: Path) -> IRDescription:
        """Describe one IR file, honouring the configured unreadable set."""
        role = SubmodelRole.OTHER
        from yonga.domain.artifacts import role_for_filename

        role = role_for_filename(path.name)
        # The real adapter records the companion weights file next to the XML;
        # so does this one, or a test could not see a staging path recorded there.
        companion = path.with_suffix(".bin")
        bin_path = companion if companion.is_file() else None
        if path.name in self.unreadable:
            return IRDescription(
                path=path,
                role=role,
                readable=False,
                read_error="Unsupported element type: STRiNG",
                bin_path=bin_path,
            )
        return IRDescription(path=path, role=role, readable=True, bytes_total=64, bin_path=bin_path)

    def compile(
        self,
        path: Path,
        device: str,
        properties: Mapping[str, str] | None = None,
    ) -> ProbeResult:
        """Compile one IR, failing on devices listed in ``compile_failures``."""
        self.compiled.append((path.name, device))
        from yonga.domain.artifacts import role_for_filename

        role = role_for_filename(path.name)
        failure = self.compile_failures.get(device)
        if failure is not None:
            return ProbeResult(
                device=device,
                role=role,
                path=path,
                status=ProbeStatus.FAILED,
                message=failure,
                properties=dict(properties or {}),
            )
        return ProbeResult(
            device=device,
            role=role,
            path=path,
            status=ProbeStatus.OK,
            duration_seconds=0.01,
            properties=dict(properties or {}),
        )

    def read_tokenizer(self, artifact: Path) -> ProbeResult:
        """Read the tokenizer pair, failing while the ``STRiNG`` defect is present."""
        self.tokenizer_reads += 1
        outcomes = {
            "openvino_version": "2026.4.0",
            "openvino_tokenizers_version": "2026.4.0.0",
        }
        present = 0
        broken: list[str] = []
        for role, name in (
            ("tokenizer", "openvino_tokenizer.xml"),
            ("detokenizer", "openvino_detokenizer.xml"),
        ):
            target = artifact / name
            if not target.is_file():
                outcomes[f"{role}_read"] = "missing"
                outcomes[f"{role}_compile"] = "missing"
                continue
            present += 1
            if BAD_PRECISION in target.read_text(encoding="utf-8"):
                outcomes[f"{role}_read"] = "failed"
                outcomes[f"{role}_compile"] = "not_attempted"
                broken.append(name)
                continue
            outcomes[f"{role}_read"] = "ok"
            outcomes[f"{role}_compile"] = "ok"
        if present == 0:
            return ProbeResult(
                device="CPU",
                role=SubmodelRole.TOKENIZER,
                path=artifact,
                status=ProbeStatus.SKIPPED,
                properties=outcomes,
                message="No tokenizer IR is present.",
            )
        if broken:
            return ProbeResult(
                device="CPU",
                role=SubmodelRole.TOKENIZER,
                path=artifact,
                status=ProbeStatus.FAILED,
                properties=outcomes,
                message=(
                    "Exception from src/inference/src/cpp/core.cpp:84:\n"
                    "Check 'is_valid_type_idx(type_idx)' failed at "
                    "src/core/src/type/element_type.cpp:115:\n"
                    "Unsupported element type: STRiNG"
                ),
            )
        return ProbeResult(
            device="CPU",
            role=SubmodelRole.TOKENIZER,
            path=artifact,
            status=ProbeStatus.OK,
            properties=outcomes,
        )


@dataclass
class FakeHardwareProbe:
    """A :class:`~yonga.adapters.HardwareProbe` returning a fixed inventory."""

    snapshot: HardwareInventory

    def inventory(self, *, full: bool = False) -> HardwareInventory:
        """Return the fixed inventory."""
        return self.snapshot


@dataclass
class FakeFilesystemProbe:
    """A :class:`~yonga.adapters.FilesystemProbe` modelling the reference host.

    Only the paths listed in ``tmpfs_paths`` are RAM-backed, which mirrors the
    reference host where ``/tmp`` is a 16 GiB tmpfs and everything else is ext4.
    Matching is by exact path so a pytest ``tmp_path`` that happens to live under
    ``/tmp`` is still treated as ordinary disk.
    """

    tmpfs_paths: tuple[Path, ...] = ()
    tmpfs_free: int = TMPFS_FREE_BYTES
    disk_free: int = DISK_FREE_BYTES
    #: Roots reported as a second, smaller filesystem, so temp and artifact space
    #: can be varied independently the way two real partitions would be.
    small_roots: tuple[Path, ...] = ()
    small_free: int = 0

    def usage(self, path: Path) -> FilesystemInfo:
        """Return capacity facts for the filesystem holding ``path``."""
        target = Path(path)
        if target in self.tmpfs_paths:
            return FilesystemInfo(
                path=target,
                mount_point=target,
                fstype="tmpfs",
                total_bytes=self.tmpfs_free,
                free_bytes=self.tmpfs_free,
                available_bytes=self.tmpfs_free,
            )
        for root in self.small_roots:
            if target == root or root in target.parents:
                return FilesystemInfo(
                    path=target,
                    mount_point=root,
                    fstype="ext4",
                    total_bytes=max(self.small_free, 1),
                    free_bytes=self.small_free,
                    available_bytes=self.small_free,
                )
        return FilesystemInfo(
            path=target,
            mount_point=Path("/"),
            fstype="ext4",
            total_bytes=935 * GIB,
            free_bytes=self.disk_free,
            available_bytes=self.disk_free,
        )


@dataclass
class FakeSmokeTester:
    """A :class:`~yonga.application.prepare.SmokeTester` that never touches GenAI."""

    failures: dict[str, str] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, str], int]] = field(default_factory=list)

    def smoke_test(
        self,
        artifact_dir: Path,
        *,
        device: str,
        pipeline: PipelineKind,
        properties: Mapping[str, str],
        prompt: str,
        max_new_tokens: int,
    ) -> ProbeResult:
        """Record the call and report success unless this device is set to fail."""
        self.calls.append((device, dict(properties), max_new_tokens))
        failure = self.failures.get(device)
        if failure is not None:
            return ProbeResult(device=device, status=ProbeStatus.FAILED, message=failure)
        return ProbeResult(
            device=device,
            status=ProbeStatus.OK,
            duration_seconds=0.05,
            message=f"generated {max_new_tokens} tokens",
            properties={key: str(value) for key, value in properties.items()},
        )


# --- fixture builders ----------------------------------------------------------------


def write_tokenizer_pair(directory: Path, *, broken: bool = True) -> None:
    """Copy the authentic OpenVINO 2026.4 tokenizer fixture into ``directory``."""
    source = FIXTURES / (
        "tokenizer_string_precision_before.xml"
        if broken
        else "tokenizer_string_precision_after.xml"
    )
    for name in ("openvino_tokenizer", "openvino_detokenizer"):
        shutil.copy(source, directory / f"{name}.xml")
        shutil.copy(FIXTURES / "tokenizer_string_precision.bin", directory / f"{name}.bin")


def reference_inventory(devices: Sequence[str] = ("CPU", "GPU", "NPU")) -> HardwareInventory:
    """Build the hardware inventory of the reference Lunar Lake host."""
    full_names = {
        "CPU": "Intel(R) Core(TM) Ultra 7 268V",
        "GPU": "Intel(R) Arc(TM) Graphics (iGPU)",
        "NPU": "Intel(R) AI Boost",
    }
    kinds = {"CPU": DeviceKind.CPU, "GPU": DeviceKind.GPU, "NPU": DeviceKind.NPU}
    return HardwareInventory(
        host=HostInfo(
            platform="Linux",
            machine="x86_64",
            kernel="7.2.6-070206-generic",
            python_version="3.14.4",
            distro_id="ubuntu",
            distro_pretty="Ubuntu 26.04.1 LTS",
            cpu_model="Intel(R) Core(TM) Ultra 7 268V",
            cpu_count=8,
        ),
        memory=MemoryInfo(total_bytes=32 * GIB, available_bytes=20 * GIB),
        devices=[
            OpenVINODevice(name=name, kind=kinds[name], full_name=full_names[name])
            for name in devices
        ],
        openvino_available=True,
        device_tags=["intel.cpu", "intel.gpu", "intel.npu", "intel.npu.lunar_lake"],
        hf_auth=HuggingFaceAuthInfo(logged_in=False),
    )


def safetensors_metadata(
    repo_id: str = SAFETENSORS_REPO,
    *,
    sha: str = SAFETENSORS_SHA,
    weight_bytes: int = SOURCE_WEIGHT_BYTES,
    total_bytes: int = REPO_TOTAL_BYTES,
    requires_remote_code: bool = False,
    model_type: str = "qwen3_5",
    pipeline_tag: str | None = "image-text-to-text",
) -> ModelMetadata:
    """Build metadata for the safetensors Qwen3.5 acceptance fixture."""
    shards = [
        RepoFile(path=f"model.safetensors-0000{index}-of-00004.safetensors", size=size, lfs=True)
        for index, size in enumerate(
            (5_278_251_472, 5_358_071_808, 5_340_098_264, 2_843_299_496), start=1
        )
    ]
    assets = [RepoFile(path="tokenizer.json", size=total_bytes - weight_bytes)]
    return ModelMetadata(
        identity=ModelIdentity(repo_id=repo_id, revision=sha),
        resolved_revision=sha,
        license="apache-2.0",
        pipeline_tag=pipeline_tag,
        library_name="transformers",
        model_type=model_type,
        config={"model_type": model_type, "architectures": ["Qwen3_5ForConditionalGeneration"]},
        files=[*shards, *assets],
        formats=[
            FormatObservation(
                format=ModelFormat.SAFETENSORS,
                files=[shard.path for shard in shards],
                total_bytes=weight_bytes,
                complete=True,
            )
        ],
        requires_remote_code=requires_remote_code,
    )


def gguf_metadata(repo_id: str = GGUF_REPO, *, sha: str = GGUF_SHA) -> ModelMetadata:
    """Build metadata for the GGUF-only repository the user actually asks for."""
    files = [
        RepoFile(path=f"{repo_id.rsplit('/', 1)[-1]}-{quant}.gguf", size=6_000_000_000, lfs=True)
        for quant in ("BF16", "Q4_K_M")
    ]
    return ModelMetadata(
        identity=ModelIdentity(repo_id=repo_id, revision=sha),
        resolved_revision=sha,
        license="apache-2.0",
        tags=["gguf", "qwen3.5", "qwen", f"base_model:{SAFETENSORS_REPO}"],
        files=files,
        formats=[
            FormatObservation(
                format=ModelFormat.GGUF,
                files=[item.path for item in files],
                total_bytes=12_000_000_000,
                complete=True,
            )
        ],
    )


def openvino_metadata(repo_id: str = OPENVINO_REPO, *, sha: str = OPENVINO_SHA) -> ModelMetadata:
    """Build metadata for a repository that already publishes OpenVINO IR."""
    files = [
        RepoFile(path="openvino_model.xml", size=2_000_000),
        RepoFile(path="openvino_model.bin", size=5_000_000_000, lfs=True),
    ]
    return ModelMetadata(
        identity=ModelIdentity(repo_id=repo_id, revision=sha),
        resolved_revision=sha,
        license="apache-2.0",
        pipeline_tag="image-text-to-text",
        library_name="openvino",
        model_type="qwen3_5",
        config={"model_type": "qwen3_5"},
        files=files,
        formats=[
            FormatObservation(
                format=ModelFormat.OPENVINO_IR,
                files=[item.path for item in files],
                total_bytes=5_002_000_000,
                complete=True,
            )
        ],
    )


def builtin_resolver(
    extra_documents: Iterable[str] = (),
    *,
    host_tags: Sequence[str] = ("intel.cpu", "intel.gpu", "intel.npu", "intel.npu.lunar_lake"),
) -> RegistryResolver:
    """Build a resolver over the built-in registry plus optional extra YAML documents."""
    from yonga.registry.loader import parse_document

    documents = list(load_builtin_documents())
    documents.extend(
        parse_document(text, origin=f"<test-{index}>") for index, text in enumerate(extra_documents)
    )
    registry: Registry = load_registry_from_documents(documents)
    return RegistryResolver(registry, versions=REGISTRY_VERSIONS, host_tags=host_tags)


def settings_for(tmp_path: Path, **overrides: object) -> Settings:
    """Build settings rooted entirely inside ``tmp_path``."""
    payload: dict[str, object] = {
        "config_dir": tmp_path / "config",
        "cache_dir": tmp_path / "cache",
        "data_dir": tmp_path / "data",
    }
    payload.update(overrides)
    return Settings.model_validate(payload)


def build_service(
    *,
    settings: Settings,
    hub: HubClient,
    exporter: Exporter | None = None,
    runtime: RuntimeProbe | None = None,
    hardware: HardwareProbe | None = None,
    filesystem: FilesystemProbe | None = None,
    resolver: RegistryResolver | None = None,
    smoke: SmokeTester | None = None,
    environments: ExportEnvironments | None = None,
) -> PrepareService:
    """Assemble a :class:`PrepareService` from fakes, with sensible defaults."""
    return PrepareService(
        hub_client=hub,
        exporter=exporter if exporter is not None else FakeExporter(),
        runtime_probe=runtime if runtime is not None else FakeRuntimeProbe(),
        hardware_probe=hardware
        if hardware is not None
        else FakeHardwareProbe(reference_inventory()),
        filesystem=filesystem
        if filesystem is not None
        else FakeFilesystemProbe(tmpfs_paths=(system_temp_dir(),)),
        registry_resolver=resolver if resolver is not None else builtin_resolver(),
        artifact_store=ArtifactStore(settings.paths, settings),
        settings=settings,
        smoke_tester=smoke if smoke is not None else FakeSmokeTester(),
        export_environments=environments,
    )


def make_snapshot(tmp_path: Path, name: str = "snapshot") -> Path:
    """Create a directory standing in for a Hugging Face snapshot."""
    snapshot = tmp_path / name
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text('{"model_type": "qwen3_5"}\n', encoding="utf-8")
    return snapshot


def approve_all(reason: ConfirmationReason) -> bool:
    """Confirmation callback that approves everything (an interactive user saying yes)."""
    assert reason.code
    return True


def yes_flag(reason: ConfirmationReason) -> bool:
    """Confirmation callback modelling ``--yes``: never approves a security reason."""
    return not reason.security_sensitive


def run_prepare(
    service: PrepareService,
    plan: PreparationPlan,
    settings: Settings,
    *,
    confirm: ConfirmCallback = approve_all,
    allow_repairs: bool = True,
    sink: CollectingSink | None = None,
    ignore_exporter_requirements: bool = False,
) -> PreparationOutcome:
    """Run a plan inside a real job context, reusing the plan's job id.

    ``ignore_exporter_requirements`` defaults to ``False``, exactly as the CLI
    does: a test that wants to convert past an unmet pin has to say so, the same
    way a user has to.
    """
    events = sink if sink is not None else CollectingSink()
    with open_job(
        settings=settings,
        requested_model=plan.requested.repo_id,
        job_id=plan.job_id,
        progress=events,
    ) as ctx:
        return service.prepare(
            plan,
            confirm=confirm,
            ctx=ctx,
            allow_repairs=allow_repairs,
            ignore_exporter_requirements=ignore_exporter_requirements,
        )


# --- fixtures ------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings confined to the test's temporary directory."""
    return settings_for(tmp_path)


@pytest.fixture
def hub(tmp_path: Path) -> FakeHub:
    """A hub carrying the three Qwen3.5 acceptance repositories."""
    return FakeHub(
        repos={
            GGUF_REPO: gguf_metadata(),
            SAFETENSORS_REPO: safetensors_metadata(),
            OPENVINO_REPO: openvino_metadata(),
        },
        snapshots={
            GGUF_REPO: make_snapshot(tmp_path, "snapshot-gguf"),
            SAFETENSORS_REPO: make_snapshot(tmp_path, "snapshot-safetensors"),
            OPENVINO_REPO: make_snapshot(tmp_path, "snapshot-openvino"),
        },
    )


# --- temp / preflight planning -------------------------------------------------------


def test_plan_prefers_managed_disk_temp_over_tmpfs(settings: Settings, hub: FakeHub) -> None:
    """The exact HOST_FACTS fixture must choose managed disk temp, not a bigger /tmp."""
    service = build_service(settings=settings, hub=hub)

    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    temp = plan.disk.temp
    assert temp.required_bytes == REQUIRED_TEMP_BYTES
    assert temp.safety_margin_bytes == SAFETY_MARGIN_BYTES
    assert temp.total_required_bytes == REQUIRED_TEMP_BYTES + SAFETY_MARGIN_BYTES
    # /tmp alone (16 GiB tmpfs) could never have held this conversion...
    assert temp.system_temp is not None
    assert temp.system_temp.is_tmpfs
    assert temp.system_temp_would_fail is True
    # ...but the managed, disk-backed directory can, so preflight passes.
    assert temp.sufficient is True
    assert temp.job_temp_dir == settings.paths.job_tmp_dir(plan.job_id)
    assert plan.disk.sufficient is True
    notes = " ".join(temp.notes).lower()
    assert "tmpdir" in notes
    assert "enlarge" not in notes and "increase the size" not in notes


def test_plan_has_no_expensive_side_effects(settings: Settings, hub: FakeHub) -> None:
    """Planning must inspect metadata only: no download, no directory, no export."""
    exporter = FakeExporter()
    service = build_service(settings=settings, hub=hub, exporter=exporter)

    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    assert hub.downloaded == []
    assert exporter.requests == []
    assert not plan.artifact_dir.exists()
    assert not plan.staging_dir.exists()
    assert not settings.paths.job_tmp_dir(plan.job_id).exists()


def test_preflight_fails_before_any_download(settings: Settings, hub: FakeHub) -> None:
    """A temp filesystem that is clearly too small stops the job with exit code 30."""
    cramped = FakeFilesystemProbe(tmpfs_paths=(system_temp_dir(),), disk_free=10 * GIB)
    exporter = FakeExporter()
    service = build_service(settings=settings, hub=hub, exporter=exporter, filesystem=cramped)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    assert plan.disk.sufficient is False

    with pytest.raises(PreflightError) as caught:
        run_prepare(service, plan, settings)

    assert caught.value.exit_code() is ExitCode.PREFLIGHT
    assert caught.value.diagnostic.code == "preflight.insufficient_disk"
    assert hub.downloaded == []
    assert exporter.requests == []


def test_export_runs_with_managed_temp_environment(settings: Settings, hub: FakeHub) -> None:
    """TMPDIR, TMP and TEMP all point at the managed job temp, never at /tmp."""
    exporter = FakeExporter()
    service = build_service(settings=settings, hub=hub, exporter=exporter)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    run_prepare(service, plan, settings)

    managed = str(settings.paths.job_tmp_dir(plan.job_id))
    assert exporter.temp_env_seen == [{"TMPDIR": managed, "TMP": managed, "TEMP": managed}]
    assert exporter.requests[0].extra_env["TMPDIR"] == managed
    # The conversion temp is yonga's own per-job directory under the configured
    # cache root, never the system temp directory itself.
    assert Path(managed).parent == settings.paths.tmp_dir
    assert Path(managed) != system_temp_dir()
    assert Path(managed).name == plan.job_id


# --- source resolution ---------------------------------------------------------------


def test_plan_resolves_gguf_only_repo_to_declared_derivative(
    settings: Settings, hub: FakeHub
) -> None:
    """A GGUF-only repository is converted through the registry's declared derivative."""
    service = build_service(settings=settings, hub=hub)

    plan = service.plan(GGUF_REPO, device="gpu")

    assert plan.resolved_source.requested.repo_id == GGUF_REPO
    assert plan.resolved_source.source.repo_id == SAFETENSORS_REPO
    assert plan.resolved_source.source.revision == SAFETENSORS_SHA
    assert plan.resolved_source.relationship is SourceRelationship.LOSSLESS_CONVERSION
    assert plan.resolved_source.requires_confirmation is True
    codes = {reason.code for reason in plan.confirmations}
    assert CONFIRMATION_THIRD_PARTY_DERIVATIVE in codes


def test_plan_refuses_to_guess_when_no_source_is_declared(
    settings: Settings, tmp_path: Path
) -> None:
    """An unconvertible repository with no declared derivative is a typed refusal."""
    orphan = "someone/unknown-gguf-model"
    hub = FakeHub(
        repos={orphan: gguf_metadata(orphan, sha="f" * 40)},
        snapshots={orphan: make_snapshot(tmp_path)},
    )
    service = build_service(settings=settings, hub=hub)

    with pytest.raises(SourceResolutionError) as caught:
        service.plan(orphan, device="gpu")

    assert caught.value.exit_code() is ExitCode.SOURCE_RESOLUTION
    assert caught.value.diagnostic.code == "source.format_unsupported"


def test_plan_uses_published_openvino_ir_without_re_exporting(
    settings: Settings, hub: FakeHub
) -> None:
    """A repository that already publishes IR is copied and validated, not converted."""
    exporter = FakeExporter()
    service = build_service(settings=settings, hub=hub, exporter=exporter)

    plan = service.plan(OPENVINO_REPO, device="gpu")

    assert plan.resolved_source.already_openvino is True
    assert plan.resolved_source.format is ModelFormat.OPENVINO_IR
    assert exporter.requests == []


# --- confirmations -------------------------------------------------------------------


def test_large_download_raises_a_confirmation(settings: Settings, hub: FakeHub) -> None:
    """A repository above the configured threshold needs an explicit yes."""
    service = build_service(settings=settings, hub=hub)

    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    reason = next(r for r in plan.confirmations if r.code == CONFIRMATION_LARGE_DOWNLOAD)
    assert reason.security_sensitive is False
    assert plan.disk.expected_download_bytes == REPO_TOTAL_BYTES


def test_declining_a_confirmation_cancels_the_job(settings: Settings, hub: FakeHub) -> None:
    """A refused confirmation stops the job before any download."""
    service = build_service(settings=settings, hub=hub)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    with pytest.raises(CancelledError) as caught:
        run_prepare(service, plan, settings, confirm=lambda reason: False)

    assert caught.value.exit_code() is ExitCode.CANCELLED
    assert hub.downloaded == []


def test_yes_cannot_enable_remote_code(settings: Settings, tmp_path: Path) -> None:
    """``--yes`` must never stand in for ``--trust-remote-code``."""
    repo = "acme/remote-code-model"
    hub = FakeHub(
        repos={repo: safetensors_metadata(repo, sha="a" * 40, requires_remote_code=True)},
        snapshots={repo: make_snapshot(tmp_path)},
    )
    exporter = FakeExporter()
    service = build_service(settings=settings, hub=hub, exporter=exporter)

    plan = service.plan(repo, device="gpu")
    assert plan.trust_remote_code_required is True
    assert plan.trust_remote_code_enabled is False
    assert plan.blocked is True
    remote = next(r for r in plan.confirmations if r.code == CONFIRMATION_REMOTE_CODE)
    assert remote.security_sensitive is True
    assert remote in plan.security_confirmations
    assert remote not in plan.auto_approvable_confirmations

    with pytest.raises(RemoteCodeRequiredError) as caught:
        run_prepare(service, plan, settings, confirm=yes_flag)

    assert caught.value.exit_code() is ExitCode.REMOTE_CODE_REQUIRED
    assert hub.downloaded == []
    assert exporter.requests == []


def test_explicit_trust_remote_code_still_needs_confirmation(
    settings: Settings, tmp_path: Path
) -> None:
    """Even with the flag, a ``--yes`` style callback cannot approve remote code."""
    repo = "acme/remote-code-model"
    hub = FakeHub(
        repos={repo: safetensors_metadata(repo, sha="a" * 40, requires_remote_code=True)},
        snapshots={repo: make_snapshot(tmp_path)},
    )
    service = build_service(settings=settings, hub=hub)
    plan = service.plan(repo, device="gpu", trust_remote_code=True)
    assert plan.blocked is False

    with pytest.raises(CancelledError):
        run_prepare(service, plan, settings, confirm=yes_flag)

    # An interactive user who actually says yes gets through, and the flag reaches
    # the exporter.
    outcome = run_prepare(service, plan, settings, confirm=approve_all)
    assert outcome.manifest.export.arguments["trust_remote_code"] is True


# --- exporter requirements -----------------------------------------------------------


def test_exporter_version_mismatch_is_reported_not_papered_over(
    settings: Settings, hub: FakeHub
) -> None:
    """Transformers 5.5.4 against the registry's ``==5.2.*`` must show up in the plan."""
    exporter = FakeExporter(versions=dict(MISMATCHED_EXPORT_VERSIONS))
    service = build_service(settings=settings, hub=hub, exporter=exporter)

    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    assert plan.exporter.transformers == "==5.2.*"
    assert plan.exporter.satisfied is False
    assert any(
        "transformers" in problem and "5.5.4" in problem for problem in plan.exporter.problems
    )
    assert any("transformers" in note for note in plan.known_limitations)


def test_exporter_requirements_pass_when_versions_match(settings: Settings, hub: FakeHub) -> None:
    """A host with the required Transformers reports a satisfied requirement set."""
    service = build_service(settings=settings, hub=hub)

    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    assert plan.exporter.satisfied is True
    assert plan.exporter.problems == []


def test_an_absent_exporter_package_reads_differently_from_a_wrong_one(
    settings: Settings, hub: FakeHub
) -> None:
    """ "Not installed" and "the wrong version" are different problems, so they say so."""
    exporter = FakeExporter(versions={**INSTALLED_EXPORT_VERSIONS, "transformers": None})
    service = build_service(settings=settings, hub=hub, exporter=exporter)

    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    assert plan.exporter.satisfied is False
    assert plan.exporter.problems == [
        "transformers ==5.2.* is required, but transformers is not installed"
    ]


def test_an_unparsable_installed_version_is_unknown_rather_than_unsatisfied(
    settings: Settings, hub: FakeHub
) -> None:
    """``satisfies`` says ``False`` for a version it cannot read; that is not a verdict.

    Treating it as one would refuse to convert on any host whose package
    metadata yonga happens not to understand, so the plan records what it could
    not check as a known limitation and leaves ``problems`` empty.
    """
    exporter = FakeExporter(
        versions={**INSTALLED_EXPORT_VERSIONS, "transformers": "unreleased-nightly"}
    )
    service = build_service(settings=settings, hub=hub, exporter=exporter)

    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    assert plan.exporter.problems == []
    assert plan.exporter.satisfied is True
    assert any(
        "could not be checked" in note and "unreleased-nightly" in note
        for note in plan.known_limitations
    )


# --- reuse ---------------------------------------------------------------------------


def test_ready_artifact_is_reused_and_force_overrides_it(settings: Settings, hub: FakeHub) -> None:
    """A second prepare reuses the ready artifact; ``--force`` converts again."""
    exporter = FakeExporter()
    service = build_service(settings=settings, hub=hub, exporter=exporter)

    first = run_prepare(service, service.plan(SAFETENSORS_REPO, device="gpu"), settings)
    assert first.reused is False
    assert len(exporter.requests) == 1

    second_plan = service.plan(SAFETENSORS_REPO, device="gpu")
    assert second_plan.reuse_existing_artifact == first.artifact_dir
    second = run_prepare(service, second_plan, settings)
    assert second.reused is True
    assert second.artifact_dir == first.artifact_dir
    assert len(exporter.requests) == 1
    assert hub.downloaded == [SAFETENSORS_REPO]

    forced_plan = service.plan(SAFETENSORS_REPO, device="gpu", force=True)
    assert forced_plan.reuse_existing_artifact is None
    forced = run_prepare(service, forced_plan, settings)
    assert forced.reused is False
    assert len(exporter.requests) == 2
    assert forced.artifact_dir == first.artifact_dir


def test_artifact_directory_layout_matches_architecture_spec(
    settings: Settings, hub: FakeHub
) -> None:
    """``<namespace>--<model>/<sha-short>--<profile>/`` is the documented layout."""
    service = build_service(settings=settings, hub=hub)

    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    assert plan.artifact_dir.parent.name == "nDimensional--Qwen3.5-9B-Uncensored-Safetensors"
    assert plan.artifact_dir.name == f"{SAFETENSORS_SHA[:12]}--gpu-int4-asym-g128"
    assert plan.artifact_dir.parent.parent == settings.paths.artifacts_dir


# --- stage stream --------------------------------------------------------------------


def test_stage_events_are_numbered_one_to_seven(settings: Settings, hub: FakeHub) -> None:
    """The stage stream carries the seven stages of ``docs/CLI_SPEC.md`` section 6, in order."""
    service = build_service(settings=settings, hub=hub)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    sink = CollectingSink()

    run_prepare(service, plan, settings, sink=sink)

    stages = sink.of_kind(ProgressKind.STAGE)[:7]
    assert [(event.current, event.total) for event in stages] == [(n, 7) for n in range(1, 8)]
    assert [event.message for event in stages] == [
        "Inspecting source",
        "Resolving compatibility",
        "Preflight",
        "Downloading",
        "Converting",
        "Validating / repairing",
        "Probing runtime",
    ]


def test_a_stage_message_never_carries_its_own_counter(settings: Settings, hub: FakeHub) -> None:
    """The position travels in ``current``/``total``; the renderer prints it once.

    A message beginning with its own ``"4/7"`` printed the counter twice, because
    the renderer adds one from the same event's position.
    """
    service = build_service(settings=settings, hub=hub)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    sink = CollectingSink()

    run_prepare(service, plan, settings, sink=sink)

    for event in sink.of_kind(ProgressKind.STAGE):
        assert not re.match(r"^\d+\s*/\s*\d+", event.message), event.message


def test_hash_events_are_emitted_during_finalization(settings: Settings, hub: FakeHub) -> None:
    """Every published file is hashed and reported through the progress stream."""
    service = build_service(settings=settings, hub=hub)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    sink = CollectingSink()

    outcome = run_prepare(service, plan, settings, sink=sink)

    hashed = {event.message for event in sink.of_kind(ProgressKind.HASH)}
    assert "openvino_language_model.xml" in hashed
    assert all(record.sha256 for record in outcome.manifest.files)


# --- guard: the fakes really are the declared protocols ------------------------------


def test_fakes_satisfy_the_adapter_protocols(settings: Settings, hub: FakeHub) -> None:
    """A structural check that the fakes match the ports they stand in for."""
    assert isinstance(hub, HubClient)
    assert isinstance(FakeExporter(), Exporter)
    assert isinstance(FakeRuntimeProbe(), RuntimeProbe)
    assert isinstance(FakeHardwareProbe(reference_inventory()), HardwareProbe)
    assert isinstance(FakeFilesystemProbe(), FilesystemProbe)
    assert isinstance(FakeSmokeTester(), SmokeTester)
    assert approve_all(ConfirmationReason(code="large_download", message="x")) is True
    assert yes_flag(ConfirmationReason(code="remote_code", message="x")) is False


def test_manifest_records_requested_and_resolved_revisions(
    settings: Settings, hub: FakeHub
) -> None:
    """``SPEC.md`` section 14 wants both what was asked for and what it resolved to."""
    service = build_service(settings=settings, hub=hub)
    plan = service.plan(GGUF_REPO, device="gpu", revision="main")

    outcome = run_prepare(service, plan, settings)

    manifest = outcome.manifest
    assert manifest.requested_model.repo_id == GGUF_REPO
    assert manifest.requested_model.requested_revision == "main"
    assert manifest.requested_model.resolved_revision == GGUF_SHA
    assert manifest.resolved_source.repo_id == SAFETENSORS_REPO
    assert manifest.resolved_source.resolved_revision == SAFETENSORS_SHA
    assert manifest.resolved_source.relationship is SourceRelationship.LOSSLESS_CONVERSION
    assert manifest.tool_version
    assert manifest.source_license == "apache-2.0"
    assert manifest.architecture == "qwen3_5"
    assert manifest.architecture_rule_id == "arch-qwen3_5"
    assert manifest.task == "image-text-to-text"
    assert manifest.pipeline is PipelineKind.VLM
    assert manifest.host["devices"] == "CPU,GPU,NPU"
    assert manifest.ready is True


def test_manifest_architecture_is_the_identity_not_the_matched_rule_id(
    settings: Settings, hub: FakeHub
) -> None:
    """``SPEC.md`` section 14 wants ``"architecture": "qwen3_5"``, not a rule id.

    An architecture rule covers a family, so its id answers a different question
    than "what is this model". Both are recorded, in separate fields.
    """
    service = build_service(settings=settings, hub=hub)

    manifest = run_prepare(service, service.plan(SAFETENSORS_REPO, device="gpu"), settings).manifest

    assert manifest.architecture == "qwen3_5"
    assert manifest.architecture_rule_id == "arch-qwen3_5"
    payload = manifest.to_json()
    assert payload["architecture"] == "qwen3_5"
    assert payload["architecture_rule_id"] == "arch-qwen3_5"
    assert not str(payload["architecture"]).startswith("arch-")


def test_manifest_written_before_the_rule_id_field_existed_still_validates(
    settings: Settings, hub: FakeHub
) -> None:
    """Artifacts prepared by an older build must keep loading, field or no field."""
    service = build_service(settings=settings, hub=hub)
    manifest = run_prepare(service, service.plan(SAFETENSORS_REPO, device="gpu"), settings).manifest

    payload = manifest.to_json()
    del payload["architecture_rule_id"]
    payload["architecture"] = "arch-generic-text-generation"

    restored = ArtifactManifest.model_validate(payload)

    assert restored.architecture_rule_id is None
    assert restored.architecture == "arch-generic-text-generation"


def test_manifest_records_paths_that_still_exist_after_finalization(
    settings: Settings, hub: FakeHub
) -> None:
    """Nothing in the manifest may point into the staging directory that was renamed away.

    In the real 9B manifest ``<data>/artifacts/.staging/<job-id>/model`` appeared
    in ``export.argv``, in every ``validation.irs[].path`` and ``bin_path``, and
    in every ``device_probes[*].submodels[].path``. The argv is a faithful record
    of the command that ran and keeps its literal path; every field that names a
    *file of this artifact* is rewritten relative to the artifact directory, the
    same form ``manifest.files`` already uses.
    """
    service = build_service(settings=settings, hub=hub)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    staging = plan.staging_dir

    outcome = run_prepare(service, plan, settings)
    manifest = outcome.manifest

    assert not staging.exists()
    assert manifest.validation is not None
    assert manifest.validation.irs
    for ir in manifest.validation.irs:
        assert not ir.path.is_absolute()
        assert (outcome.artifact_dir / ir.path).is_file()
        assert ir.bin_path is not None
        assert not ir.bin_path.is_absolute()
        assert (outcome.artifact_dir / ir.bin_path).is_file()

    probed = [
        result.path
        for report in manifest.device_probes.values()
        for result in report.submodels
        if result.path is not None
    ]
    assert probed
    for probed_path in probed:
        assert not probed_path.is_absolute()
        assert (outcome.artifact_dir / probed_path).is_file()

    # The literal staging path survives in exactly one place: the argv of the
    # command that really ran, which is covered by its own test below.
    recorded = manifest.to_json_text()
    assert recorded.count(str(staging)) == 1
    assert manifest.export.argv[-1] == str(staging)


def test_export_argv_stays_literal_and_names_where_its_output_went(
    settings: Settings, hub: FakeHub
) -> None:
    """The recorded command is what ran; the manifest says where its output ended up."""
    service = build_service(settings=settings, hub=hub)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    staging = plan.staging_dir

    manifest = run_prepare(service, plan, settings).manifest

    assert manifest.export.argv[-1] == str(staging)
    assert manifest.export.arguments["output_dir"] == "."


def test_a_cross_device_explicit_profile_is_disclosed_in_the_plan(
    settings: Settings, hub: FakeHub
) -> None:
    """``--profile npu-int4-sym-cw --device gpu`` must say so before any work starts."""
    service = build_service(settings=settings, hub=hub)

    plan = service.plan(SAFETENSORS_REPO, device="gpu", profile_id="npu-int4-sym-cw")

    assert plan.profile.id == "npu-int4-sym-cw"
    assert plan.target_device == "GPU"
    assert any(
        "npu-int4-sym-cw" in note and "NPU" in note and "GPU" in note
        for note in plan.known_limitations
    )


def test_a_matching_explicit_profile_adds_no_note(settings: Settings, hub: FakeHub) -> None:
    """A profile that targets the requested device is unremarkable and stays silent."""
    service = build_service(settings=settings, hub=hub)

    plan = service.plan(SAFETENSORS_REPO, device="gpu", profile_id="gpu-int4-asym-g128")

    assert plan.profile.id == "gpu-int4-asym-g128"
    assert not any("not the requested" in note for note in plan.known_limitations)


def test_a_manifest_full_of_old_staging_paths_still_validates(
    settings: Settings, hub: FakeHub
) -> None:
    """Artifacts prepared before the paths were rewritten must keep loading.

    Nothing in the schema changed, so an old manifest — absolute staging paths
    in ``validation.irs`` and ``device_probes``, and no ``output_dir`` among the
    export arguments — still parses and keeps exactly what it recorded.
    """
    service = build_service(settings=settings, hub=hub)
    manifest = run_prepare(service, service.plan(SAFETENSORS_REPO, device="gpu"), settings).manifest

    payload = manifest.to_json()
    stale = "/home/someone/.local/share/yonga/artifacts/.staging/old-job/model"
    validation = payload["validation"]
    assert isinstance(validation, dict)
    for ir in validation["irs"]:
        ir["path"] = f"{stale}/{Path(str(ir['path'])).name}"
        ir["bin_path"] = f"{stale}/{Path(str(ir['bin_path'])).name}"
    export = payload["export"]
    assert isinstance(export, dict)
    del export["arguments"]["output_dir"]

    restored = ArtifactManifest.model_validate(payload)

    assert restored.validation is not None
    assert all(str(ir.path).startswith(stale) for ir in restored.validation.irs)
    assert "output_dir" not in restored.export.arguments


def test_manifest_holds_no_user_paths_or_secrets(settings: Settings, hub: FakeHub) -> None:
    """The host summary is a description of the machine, not of this user's disk."""
    service = build_service(settings=settings, hub=hub)

    outcome = run_prepare(service, service.plan(SAFETENSORS_REPO, device="gpu"), settings)

    # ``to_json`` runs the secret-key guard; reaching here means it passed.
    payload = outcome.manifest.to_json()
    assert set(outcome.manifest.host) <= {
        "platform",
        "machine",
        "kernel",
        "python",
        "cpu",
        "distro",
        "devices",
        "openvino",
        "openvino_tokenizers",
    }
    assert "/" not in "".join(outcome.manifest.host.values())
    assert payload["artifact_id"] == outcome.manifest.artifact_id


def test_artifact_id_is_stable_for_the_same_source_and_profile(
    settings: Settings, hub: FakeHub
) -> None:
    """Reuse detection and the compile cache both depend on a deterministic id."""
    service = build_service(settings=settings, hub=hub)
    first = run_prepare(service, service.plan(SAFETENSORS_REPO, device="gpu"), settings)
    forced = run_prepare(
        service, service.plan(SAFETENSORS_REPO, device="gpu", force=True), settings
    )

    assert forced.manifest.artifact_id == first.manifest.artifact_id

    other_profile = run_prepare(service, service.plan(SAFETENSORS_REPO, device="cpu"), settings)
    assert other_profile.manifest.artifact_id != first.manifest.artifact_id


# --- staging retention on failure ----------------------------------------------------


def test_a_failed_export_leaves_no_staging_husk_behind(settings: Settings, hub: FakeHub) -> None:
    """An export that wrote nothing has nothing to diagnose, so nothing is kept.

    One husk per failed run is how ``artifacts/.staging`` grew to 15 directories
    on the reference host without a single line of output mentioning it.
    """
    exporter = FakeExporter(succeed=False, stderr_tail=["exporter exploded"])
    service = build_service(settings=settings, hub=hub, exporter=exporter)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")

    with pytest.raises(ExportError):
        run_prepare(service, plan, settings)

    assert not settings.paths.job_staging_dir(plan.job_id).exists()
    assert not plan.artifact_dir.exists()


def test_a_failed_validation_keeps_the_partial_export_and_says_where(
    settings: Settings, hub: FakeHub
) -> None:
    """SPEC section 16 retains raw diagnostics — and tells the user where they are."""
    exporter = FakeExporter(write_tokenizer=False, numeric_models=())
    service = build_service(settings=settings, hub=hub, exporter=exporter)
    plan = service.plan(SAFETENSORS_REPO, device="gpu")
    sink = CollectingSink()

    with pytest.raises(ValidationError) as caught:
        run_prepare(service, plan, settings, sink=sink)

    staging = settings.paths.job_staging_dir(plan.job_id)
    assert staging.is_dir()
    assert (staging / "model" / "config.json").is_file()

    failure = caught.value.diagnostic
    assert failure.context["staging_dir"] == str(staging)
    assert any(str(staging) in line for line in failure.remediation)
    assert any(str(staging) in event.message for event in sink.of_kind(ProgressKind.LOG))


def test_preparing_again_sweeps_the_staging_directories_that_piled_up(
    settings: Settings, hub: FakeHub
) -> None:
    """The bounded sweep runs when the next job opens, which is what caps the root."""
    staging_root = settings.paths.staging_dir
    staging_root.mkdir(parents=True, exist_ok=True)
    stale = staging_root / "abandoned-job"
    (stale / "model").mkdir(parents=True)
    (stale / "model" / "openvino_model.bin").write_bytes(b"\x00" * 4096)
    old = (datetime.now(tz=UTC) - timedelta(hours=500)).timestamp()
    os.utime(stale, (old, old))
    service = build_service(settings=settings, hub=hub)

    outcome = run_prepare(service, service.plan(SAFETENSORS_REPO, device="gpu"), settings)

    assert outcome.artifact_dir.is_dir()
    assert not stale.exists()
    assert not settings.paths.job_staging_dir(outcome.job_id).exists()
