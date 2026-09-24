"""The one place where concrete adapters are constructed and wired into services.

``yonga.cli`` decides *what* to do and how to print it; this module decides
*which* implementation each application service talks to. Keeping the graph here
means a test can swap the whole outside world by subclassing :class:`Wiring` and
overriding a single adapter factory, without the CLI knowing.

Two rules shape the code below:

* **every heavy import happens inside a method.** ``import yonga.cli`` pulls in
  this module, and ``yonga --help`` must work on a host with no OpenVINO, no
  ``optimum-intel`` and no GPU. Module scope therefore imports only the
  application services, the registry and the typed ports;
* **adapters are built once per invocation and cached.** One
  :class:`~yonga.adapters.openvino_probe.OpenVINOProbe` per process keeps a single
  ``openvino.Core``, so plugin registration and the compile cache are shared
  across every probe in a job.

Nothing here prints, asks a question or touches ``sys.argv``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from yonga.adapters import (
    CommandRunner,
    Exporter,
    FilesystemProbe,
    HardwareProbe,
    HubClient,
    HubPublisher,
    HubSearch,
)
from yonga.adapters import RuntimeProbe as RuntimeProbePort
from yonga.adapters.export_env import ExportEnvironmentManager
from yonga.application.artifacts import ArtifactStore
from yonga.application.benchmark import BenchmarkService
from yonga.application.benchmark import RuntimeFactory as BenchmarkRuntimeFactory
from yonga.application.chat import ChatService
from yonga.application.chat import RuntimeFactory as ChatRuntimeFactory
from yonga.application.doctor import DoctorService
from yonga.application.evaluation import EvaluationService, LogitsScorer
from yonga.application.inspect_model import InspectService
from yonga.application.prepare import PrepareService, SmokeTester
from yonga.application.probe import LocalArtifactStore, ProbeService
from yonga.application.progress import ProgressSink
from yonga.application.publish import PublishService
from yonga.application.search import SearchService
from yonga.config.paths import Paths
from yonga.config.settings import Settings
from yonga.domain.artifacts import ArtifactManifest, ProbeResult, ProbeStatus
from yonga.domain.diagnostics import ConfigurationError, FailureCategory, RegistryError, diagnostic
from yonga.domain.hardware import OpenVINODevice
from yonga.domain.models import PipelineKind
from yonga.registry.loader import load_registry
from yonga.registry.remote import RemoteRegistryStore
from yonga.registry.resolver import RegistryResolver, installed_registry_versions
from yonga.registry.schema import Registry, RegistryDocument

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    from yonga.adapters.system_linux import DeviceSnapshot

__all__ = [
    "AnyArtifactStore",
    "ArtifactResolver",
    "DeviceReporter",
    "GenAISmokeTester",
    "Wiring",
    "WiringFactory",
]


@runtime_checkable
class DeviceReporter(Protocol):
    """A runtime probe that can also enumerate OpenVINO devices.

    :class:`~yonga.adapters.openvino_probe.OpenVINOProbe` satisfies it; the
    ``RuntimeProbe`` port deliberately does not require it, because compiling and
    enumerating are different jobs.
    """

    def available(self) -> tuple[bool, str | None]:
        """Return ``(runtime importable, import error)``."""
        ...

    def devices(self) -> list[OpenVINODevice]:
        """Return every device the runtime exposes, with its properties."""
        ...


class AnyArtifactStore:
    """Artifact lookup for the read-only commands: catalogue first, then any directory.

    :class:`ArtifactStore` only recognises a directory that carries a manifest,
    which is exactly right when something is about to be published or deleted.
    ``probe``, ``bench`` and ``chat`` merely read, and must also work on an export
    directory a user produced by hand, so they fall back to
    :class:`~yonga.application.probe.LocalArtifactStore`. The catalogue is
    consulted first, so an artifact id always beats a same-named directory.
    """

    def __init__(self, catalogue: ArtifactStore, local: LocalArtifactStore) -> None:
        """Compose the managed catalogue with the permissive directory lookup."""
        self._catalogue = catalogue
        self._local = local

    def find(self, reference: str | Path) -> Path | None:
        """Return the artifact directory for ``reference``, or ``None``."""
        found = self._catalogue.find(reference)
        if found is not None:
            return found
        return self._local.find(reference)

    def manifest_for(self, artifact_dir: Path) -> ArtifactManifest | None:
        """Read the artifact manifest, or ``None`` when there is not one."""
        return self._catalogue.manifest_for(artifact_dir)


class ArtifactResolver:
    """Adapts an artifact store to the chat service's one-method port.

    ``yonga chat`` must never start a multi-gigabyte conversion as a side effect
    (``docs/CLI_SPEC.md`` section 9), so an unresolvable reference fails here with
    the exact ``yonga prepare`` command to run instead.
    """

    def __init__(self, store: AnyArtifactStore) -> None:
        """Bind the resolver to a prepared-artifact store."""
        self._store = store

    def resolve(self, reference: str) -> Path:
        """Return the prepared artifact directory for ``reference``.

        Raises:
            ConfigurationError: when no prepared artifact matches. The diagnostic
                names the ``yonga prepare`` command that would create one.
        """
        found = self._store.find(reference)
        if found is not None:
            return found
        raise ConfigurationError(
            diagnostic(
                "artifact.not_prepared",
                FailureCategory.CONFIGURATION_ERROR,
                "No prepared artifact for that model",
                (
                    f"{reference} has not been prepared on this host. yonga does not convert a "
                    "model as a side effect of starting a chat, because that can download and "
                    "quantize many gigabytes."
                ),
                remediation=[
                    f"Prepare it first: yonga prepare {reference}",
                    f"Then start the chat: yonga chat {reference}",
                    "See what is already prepared: yonga list",
                ],
                context={"reference": reference, "command": "prepare", "model": reference},
            )
        )


class GenAISmokeTester:
    """A :class:`~yonga.application.prepare.SmokeTester` over OpenVINO GenAI.

    The preparation service may not import ``openvino_genai``, so the generation
    check is a port. This is its production implementation: it lives at the edge,
    where building a real pipeline is allowed, and the CLI injects it. Without it
    no text is ever generated from a freshly converted artifact and the smoke test
    is recorded as skipped.
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
    ) -> ProbeResult:
        """Build the real pipeline, generate a few tokens and report the outcome."""
        import time

        from yonga.adapters.runtime_genai import ChatHistory, open_runtime
        from yonga.domain.benchmarks import GenerationSettings

        started = time.perf_counter()
        history = ChatHistory()
        history.add_user(prompt)
        config = GenerationSettings(max_new_tokens=max_new_tokens, do_sample=False)
        runtime = open_runtime(
            artifact_dir,
            device=device,
            pipeline=pipeline,
            properties=dict(properties),
        )
        try:
            result = runtime.generate(history, config)
        finally:
            runtime.close()
        return ProbeResult(
            device=device,
            status=ProbeStatus.OK,
            duration_seconds=time.perf_counter() - started,
            message=f"generated {result.output_tokens} token(s)",
            properties={key: str(value) for key, value in properties.items()},
        )


class Wiring:
    """The concrete adapter graph for one CLI invocation.

    Construct it from resolved :class:`~yonga.config.settings.Settings` and ask it
    for a service; every adapter it needs is built lazily, once, and reused. Tests
    subclass it and override the adapter factories — never the service factories —
    so the services under test are the real ones.
    """

    def __init__(self, settings: Settings) -> None:
        """Bind the graph to the settings resolved for this invocation."""
        self._settings = settings
        self._hardware_probe: HardwareProbe | None = None
        self._export_environments: ExportEnvironmentManager | None = None
        self._filesystem: FilesystemProbe | None = None
        self._command_runner: CommandRunner | None = None
        self._hub_client: HubClient | None = None
        self._runtime_probe: RuntimeProbePort | None = None
        self._exporter: Exporter | None = None
        self._registry: Registry | None = None
        self._registry_resolver: RegistryResolver | None = None
        self._artifact_store: ArtifactStore | None = None
        self._readable_artifacts: AnyArtifactStore | None = None

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------
    @property
    def settings(self) -> Settings:
        """The settings this graph was built from."""
        return self._settings

    @property
    def paths(self) -> Paths:
        """The managed directory layout implied by the settings."""
        return self._settings.paths

    # ------------------------------------------------------------------
    # adapters
    # ------------------------------------------------------------------
    def hardware_probe(self) -> HardwareProbe:
        """Return the host inventory probe (`/proc`, `/sys`, OpenVINO devices).

        The OpenVINO probe is injected as the inventory's device source, so a
        single ``HardwareProbe`` reports the real ``FULL_DEVICE_NAME`` of every
        accelerator instead of an inventory that describes the host only.
        """
        if self._hardware_probe is None:
            from yonga.adapters.system_linux import LinuxSystemProbe

            self._hardware_probe = LinuxSystemProbe(
                settings=self._settings,
                paths=self.paths,
                filesystem=self.filesystem(),
                device_source=self.device_source,
            )
        return self._hardware_probe

    def device_source(self) -> DeviceSnapshot:
        """Return the OpenVINO device snapshot used by the hardware inventory.

        The runtime probe is asked for the device list so that one adapter — and
        therefore one ``openvino.Core`` — serves both the inventory and every
        later compile. A probe that cannot report devices (a narrow test double,
        or a future port that only compiles) yields an empty snapshot instead of
        an error.
        """
        from yonga.adapters.system_linux import DeviceSnapshot

        probe = self.runtime_probe()
        if not isinstance(probe, DeviceReporter):
            return DeviceSnapshot(available=False, import_error=None)
        available, error = probe.available()
        return DeviceSnapshot(
            devices=tuple(probe.devices()),
            available=available,
            import_error=error,
        )

    def filesystem(self) -> FilesystemProbe:
        """Return the filesystem probe used by disk and temp preflight."""
        if self._filesystem is None:
            from yonga.adapters.filesystem import FilesystemAdapter

            self._filesystem = FilesystemAdapter()
        return self._filesystem

    def command_runner(self) -> CommandRunner:
        """Return the supervised subprocess runner used by the exporter."""
        if self._command_runner is None:
            from yonga.adapters.subprocess import SubprocessRunner, default_line_selector
            from yonga.util.redaction import environment_secrets

            # ``extra_secrets`` masks the literal values of the host's secret-bearing
            # environment variables in captured output, which the shape-based patterns
            # in ``redact_text`` cannot recognise on their own.
            self._command_runner = SubprocessRunner(
                line_selector=default_line_selector,
                extra_secrets=environment_secrets(),
            )
        return self._command_runner

    def hub_client(self) -> HubClient:
        """Return the Hugging Face Hub adapter."""
        if self._hub_client is None:
            from yonga.adapters.hf import HuggingFaceAdapter

            self._hub_client = HuggingFaceAdapter(self._settings)
        return self._hub_client

    def hub_search(self) -> HubSearch:
        """Return the Hub searcher -- the same adapter as :meth:`hub_client`.

        A Hub client that cannot search is a wiring error, reported as such
        rather than papered over by quietly building a second, networked client.
        """
        client = self.hub_client()
        if not isinstance(client, HubSearch):
            raise ConfigurationError(
                diagnostic(
                    "wiring.hub.no_search",
                    FailureCategory.CONFIGURATION_ERROR,
                    "Search is unavailable",
                    "The configured Hugging Face client does not support searching.",
                )
            )
        return client

    def search_service(self) -> SearchService:
        """Build the ``yonga search`` service. It needs the Hub, never the hardware."""
        return SearchService(self.hub_search(), self.registry_resolver())

    def evaluation_service(self, *, progress: ProgressSink | None = None) -> EvaluationService:
        """Build the ``yonga eval`` service. OpenVINO is imported only when it scores."""

        def scorer(path: Path, device: str) -> LogitsScorer:
            from yonga.adapters.openvino_logits import LogitsModel

            return LogitsModel(path, device=device)

        return EvaluationService(
            self.artifact_store(),
            scorer,
            self._settings,
            progress=progress,
            package_versions=lambda: {
                name: version for name, version in installed_registry_versions().items() if version
            },
        )

    def hub_publisher(self) -> HubPublisher:
        """Return the Hub writer. Built only by ``publish``; nothing else uploads."""
        from yonga.adapters.hf_publish import HfPublisher

        return HfPublisher()

    def publish_service(self) -> PublishService:
        """Build the ``yonga publish`` service."""
        return PublishService(
            store=self.artifact_store(),
            hub=self.hub_client(),
            publisher=self.hub_publisher(),
            resolver=self.registry_resolver(),
            settings=self._settings,
        )

    def runtime_probe(self) -> RuntimeProbePort:
        """Return the OpenVINO runtime probe.

        Constructing it imports nothing: the adapter defers every OpenVINO import
        to the first call, so this is safe on a host without the runtime.
        """
        if self._runtime_probe is None:
            from yonga.adapters.openvino_probe import OpenVINOProbe

            self._runtime_probe = OpenVINOProbe()
        return self._runtime_probe

    def exporter(self) -> Exporter:
        """Return the Optimum Intel exporter, bound to the shared command runner."""
        if self._exporter is None:
            from yonga.adapters.exporter_optimum import OptimumExporter

            self._exporter = OptimumExporter(self.command_runner())
        return self._exporter

    def chat_runtime_factory(self) -> ChatRuntimeFactory:
        """Return the GenAI runtime factory used by ``yonga chat``."""
        from yonga.adapters.runtime_genai import open_runtime

        return open_runtime

    def benchmark_runtime_factory(self) -> BenchmarkRuntimeFactory:
        """Return the GenAI runtime factory used by ``yonga bench``."""
        from yonga.adapters.runtime_genai import open_runtime

        return open_runtime

    def smoke_tester(self) -> SmokeTester | None:
        """Return the generation smoke test injected into ``prepare``."""
        return GenAISmokeTester()

    # ------------------------------------------------------------------
    # registry and artifacts
    # ------------------------------------------------------------------
    def registry(self) -> Registry:
        """Load the builtin registry merged with the user's override layers."""
        if self._registry is None:
            self._registry = load_registry(
                self.paths, remote_documents=self.remote_registry_documents()
            )
        return self._registry

    def remote_registry_documents(self) -> list[RegistryDocument]:
        """The installed remote bundle's documents, verified again; empty when unusable.

        A bundle that no longer verifies (expired, edited on disk, its key
        removed) is left out rather than failing every command: the built-in
        registry is always a safe answer. ``yonga registry status`` says why.
        """
        if self._settings.registry_update_policy == "never":
            return []
        try:
            verified = RemoteRegistryStore(self.paths).load()
        except RegistryError:
            return []
        return [] if verified is None else list(verified.documents)

    def registry_resolver(self) -> RegistryResolver:
        """Return a resolver bound to the loaded registry and the host's tags."""
        if self._registry_resolver is None:
            self._registry_resolver = RegistryResolver(
                self.registry(),
                versions=installed_registry_versions(),
                host_tags=self.host_tags(),
            )
        return self._registry_resolver

    def host_tags(self) -> Sequence[str]:
        """Return the device tags (``intel.npu.lunar_lake``, ...) of this host.

        A host whose inventory cannot be read is not a reason to fail: the
        resolver simply matches fewer host-specific rules.
        """
        try:
            return list(self.hardware_probe().inventory().device_tags)
        except Exception:  # tag discovery is advisory, never fatal
            return []

    def artifact_store(self) -> ArtifactStore:
        """Return the prepared-artifact catalogue."""
        if self._artifact_store is None:
            self._artifact_store = ArtifactStore(self.paths, self._settings)
        return self._artifact_store

    def readable_artifacts(self) -> AnyArtifactStore:
        """Return the permissive store used by ``probe``, ``bench`` and ``chat``."""
        if self._readable_artifacts is None:
            self._readable_artifacts = AnyArtifactStore(
                self.artifact_store(), LocalArtifactStore(self.paths)
            )
        return self._readable_artifacts

    def artifact_resolver(self) -> ArtifactResolver:
        """Return the chat service's reference resolver."""
        return ArtifactResolver(self.readable_artifacts())

    # ------------------------------------------------------------------
    # services
    # ------------------------------------------------------------------
    def doctor_service(self, *, progress: ProgressSink | None = None) -> DoctorService:
        """Build the ``yonga doctor`` service."""
        return DoctorService(
            self.hardware_probe(),
            self.runtime_probe(),
            self.hub_client(),
            self.registry(),
            self._settings,
            progress=progress,
            export_environments=self.export_environments().environments,
        )

    def inspect_service(self, *, progress: ProgressSink | None = None) -> InspectService:
        """Build the ``yonga inspect`` service."""
        return InspectService(
            self.hub_client(),
            self.registry_resolver(),
            self.hardware_probe(),
            self._settings,
            progress=progress,
        )

    def prepare_service(self) -> PrepareService:
        """Build the ``yonga prepare`` service, with the real generation smoke test."""
        return PrepareService(
            hub_client=self.hub_client(),
            exporter=self.exporter(),
            runtime_probe=self.runtime_probe(),
            hardware_probe=self.hardware_probe(),
            filesystem=self.filesystem(),
            registry_resolver=self.registry_resolver(),
            artifact_store=self.artifact_store(),
            settings=self._settings,
            smoke_tester=self.smoke_tester(),
            export_environments=self.export_environments(),
        )

    def export_environments(self) -> ExportEnvironmentManager:
        """Return the manager of isolated export environments."""
        if self._export_environments is None:
            self._export_environments = ExportEnvironmentManager(
                self._settings.paths, self.command_runner(), offline=self._settings.offline
            )
        return self._export_environments

    def probe_service(self) -> ProbeService:
        """Build the ``yonga probe`` service."""
        return ProbeService(
            self.runtime_probe(),
            self.registry_resolver(),
            self.readable_artifacts(),
            self._settings,
            smoke_tester=self.smoke_tester(),
        )

    def chat_service(self, *, progress: ProgressSink | None = None) -> ChatService:
        """Build the ``yonga chat`` service."""
        return ChatService(
            self.chat_runtime_factory(),
            self.artifact_resolver(),
            self.registry_resolver(),
            self._settings,
            progress=progress,
        )

    def benchmark_service(self) -> BenchmarkService:
        """Build the ``yonga bench`` service."""
        return BenchmarkService(
            self.benchmark_runtime_factory(),
            self.readable_artifacts(),
            self.hardware_probe(),
            self._settings,
        )


#: Builds an adapter graph from resolved settings. The CLI accepts one of these
#: so a test can hand it a subclass without touching a module-level global.
WiringFactory = Callable[[Settings], Wiring]
