"""Adapter ports.

Every external system — the Hugging Face Hub, the OpenVINO runtime, the Optimum
exporter, the Linux kernel, the filesystem — is reached through one of the
Protocols defined here. Application services depend on these Protocols, never on
a concrete implementation, which is what lets the whole policy layer be tested on
a machine with no Intel accelerator and no network.

Concrete implementations live in sibling modules and import their heavy
third-party dependencies *inside functions*, so that ``import yonga.cli`` stays
fast and works even when OpenVINO is not installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yonga.adapters.gguf_header import GGUFHeader
    from yonga.application.jobs import JobContext
    from yonga.domain.artifacts import IRDescription, ProbeResult
    from yonga.domain.environments import EnvironmentSpec, ExportEnvironment
    from yonga.domain.hardware import (
        DriverLibraries,
        HardwareInventory,
        HuggingFaceAuthInfo,
        PciAccelerator,
    )
    from yonga.domain.models import ModelIdentity, ModelMetadata
    from yonga.domain.plans import (
        DeclaredRequirement,
        ExporterRequirements,
        ExportRequest,
        ExportResult,
    )
    from yonga.domain.publication import PublisherIdentity

__all__ = [
    "ChecksToolchainImport",
    "CommandResult",
    "CommandRunner",
    "DeclaresRequirements",
    "DeclaresRequirementsIn",
    "DriverProbe",
    "ExportEnvironments",
    "Exporter",
    "FilesystemProbe",
    "GGUFImport",
    "GGUFInspector",
    "HardwareProbe",
    "HubClient",
    "HubPublisher",
    "HubSearch",
    "RecordsToolchainImport",
    "ReportsToolchain",
    "RuntimeProbe",
]


@runtime_checkable
class HardwareProbe(Protocol):
    """Collects host, device and package facts without changing anything."""

    def inventory(self, *, full: bool = False) -> HardwareInventory:
        """Return a snapshot of the host. Must be read-only and never need root."""
        ...


@runtime_checkable
class DriverProbe(Protocol):
    """Lists the accelerators on the bus and the system drivers present for them.

    Only ``doctor`` asks, so these are kept out of :meth:`HardwareProbe.inventory`,
    which every ``prepare`` runs.
    """

    def pci_accelerators(self, notes: list[str] | None = None) -> list[PciAccelerator] | None:
        """The Intel GPUs and NPUs on the PCI bus, or ``None`` when it cannot be read."""
        ...

    def driver_libraries(self) -> DriverLibraries:
        """The user-space GPU and NPU drivers found on this host."""
        ...


@runtime_checkable
class FilesystemProbe(Protocol):
    """Answers filesystem questions used by disk preflight."""

    def usage(self, path: Path) -> object:
        """Return a ``FilesystemInfo`` for the filesystem holding ``path``."""
        ...


@runtime_checkable
class HubClient(Protocol):
    """Reads model metadata and downloads sources from the Hugging Face Hub."""

    def resolve_revision(self, model: ModelIdentity) -> str:
        """Resolve a branch/tag/sha reference to an immutable commit SHA."""
        ...

    def inspect(self, model: ModelIdentity, *, with_sizes: bool = True) -> ModelMetadata:
        """Collect repository metadata without downloading weight shards."""
        ...

    def download(
        self,
        source: ModelIdentity,
        *,
        ctx: JobContext | None = None,
        allow_patterns: Sequence[str] | None = None,
        ignore_patterns: Sequence[str] | None = None,
    ) -> Path:
        """Materialise a pinned revision in the Hub cache and return its snapshot path."""
        ...

    def auth_info(self) -> HuggingFaceAuthInfo:
        """Report authentication state. Must never expose the token value."""
        ...


@runtime_checkable
class HubSearch(Protocol):
    """Searches the Hugging Face Hub.

    Kept separate from :class:`HubClient` so a component that only searches
    does not have to implement downloads, and so every existing ``HubClient``
    stays valid unchanged.
    """

    def search(self, query: str, *, limit: int = 20) -> list[ModelMetadata]:
        """Return metadata for the best matches, from the search response alone."""
        ...


@runtime_checkable
class GGUFInspector(Protocol):
    """Reads a GGUF file's header -- metadata and tensor types -- without its weights."""

    def gguf_header(self, model: ModelIdentity, filename: str) -> GGUFHeader:
        """The header of ``filename`` in ``model`` (pinned to a revision)."""
        ...


@runtime_checkable
class DeclaresRequirements(Protocol):
    """An exporter that can say which package versions an architecture needs."""

    def declared_requirements(
        self, model_type: str, task: str | None
    ) -> DeclaredRequirement | None:
        """The range the exporter declares for ``model_type``, or ``None``."""
        ...


@runtime_checkable
class ReportsToolchain(Protocol):
    """An exporter that can say whether it has anything to convert with at all."""

    def toolchain_available(self) -> bool:
        """Whether an export could start here, without a managed environment."""
        ...


@runtime_checkable
class ChecksToolchainImport(Protocol):
    """An exporter that can try importing itself in the environment yonga runs in."""

    def toolchain_import_failure(self) -> str | None:
        """``None`` when the in-environment exporter imports, else the failure text."""
        ...


@runtime_checkable
class RecordsToolchainImport(Protocol):
    """An exporter that remembers its last import check, so doctor can report it cheaply."""

    def toolchain_import_state(self, *, run: bool = True) -> bool | str | None:
        """``True`` imports, a ``str`` is the failure, ``None`` is nothing to check.

        With ``run=False`` only what an earlier check recorded is read and nothing
        is imported; ``False`` then means no check is recorded.
        """
        ...


@runtime_checkable
class DeclaresRequirementsIn(Protocol):
    """An exporter that can ask another environment's optimum-intel the same question.

    Used once a managed export environment exists for a host with no export
    toolchain of its own, where :class:`DeclaresRequirements` has nobody to ask.
    """

    def declared_requirements_in(
        self, python: Path, optimum_intel: str, model_type: str, task: str | None
    ) -> DeclaredRequirement | None:
        """The range optimum-intel ``optimum_intel`` under ``python`` declares, or ``None``."""
        ...


@runtime_checkable
class GGUFImport(Protocol):
    """Turns a GGUF file into OpenVINO IR in a directory yonga owns."""

    def import_file(self, gguf: Path, output_dir: Path, ctx: JobContext) -> ExportResult:
        """Import ``gguf`` into ``output_dir``; never writes next to ``gguf`` itself."""
        ...


@runtime_checkable
class HubPublisher(Protocol):
    """Writes to the Hugging Face Hub. The only port that ever uploads anything."""

    def identity(self) -> PublisherIdentity | None:
        """The logged-in account, or ``None`` without credentials. Never the token."""
        ...

    def repo_exists(self, repo_id: str) -> bool:
        """Whether the model repository already exists."""
        ...

    def publish(
        self,
        repo_id: str,
        *,
        private: bool,
        files: Sequence[tuple[str, Path]],
        generated: Mapping[str, bytes],
        message: str,
    ) -> str:
        """Create the repository if needed and upload everything in one commit; return its URL."""
        ...


@runtime_checkable
class ExportEnvironments(Protocol):
    """Builds and finds the isolated environments exports run in."""

    def spec_for(
        self,
        requirements: ExporterRequirements,
        *,
        installed: Mapping[str, str | None] | None = None,
        reason: str = "",
    ) -> EnvironmentSpec:
        """The environment that meets ``requirements`` while changing least."""
        ...

    def find(self, spec: EnvironmentSpec) -> ExportEnvironment | None:
        """The completed environment for ``spec``, if one is built."""
        ...

    def ensure(self, spec: EnvironmentSpec, ctx: JobContext) -> ExportEnvironment:
        """Return the environment for ``spec``, building and verifying it if needed."""
        ...


@runtime_checkable
class Exporter(Protocol):
    """Converts and quantizes a source checkpoint into OpenVINO IR."""

    def export(self, request: ExportRequest, ctx: JobContext) -> ExportResult:
        """Run the conversion. Must use an argv list with ``shell=False``."""
        ...

    def requirements(self) -> dict[str, str | None]:
        """Return the installed versions of the packages the export depends on."""
        ...


@runtime_checkable
class RuntimeProbe(Protocol):
    """Reads and compiles OpenVINO IR, and runs generation smoke tests."""

    def available_devices(self) -> list[str]:
        """Return the devices OpenVINO reports, e.g. ``["CPU", "GPU", "NPU"]``."""
        ...

    def read_ir(self, path: Path) -> IRDescription:
        """Read one IR file and describe its ports. Must not raise on a bad file."""
        ...

    def compile(
        self,
        path: Path,
        device: str,
        properties: Mapping[str, str] | None = None,
    ) -> ProbeResult:
        """Compile one IR on ``device`` and report the outcome as a typed result."""
        ...

    def read_tokenizer(self, artifact: Path) -> ProbeResult:
        """Read and CPU-compile the tokenizer/detokenizer pair in ``artifact``."""
        ...


class CommandResult(Protocol):
    """The outcome of one subprocess execution."""

    argv: Sequence[str]
    returncode: int
    duration_seconds: float
    stdout_lines: Sequence[str]
    stderr_lines: Sequence[str]
    cancelled: bool
    timed_out: bool


@runtime_checkable
class CommandRunner(Protocol):
    """Runs an external command as an argv list, never through a shell."""

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float | None = None,
        ctx: JobContext | None = None,
    ) -> CommandResult:
        """Execute ``argv`` with ``shell=False``, streaming output into the job log."""
        ...
