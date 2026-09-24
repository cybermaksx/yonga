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
    from yonga.application.jobs import JobContext
    from yonga.domain.artifacts import IRDescription, ProbeResult
    from yonga.domain.environments import EnvironmentSpec, ExportEnvironment
    from yonga.domain.hardware import HardwareInventory, HuggingFaceAuthInfo
    from yonga.domain.models import ModelIdentity, ModelMetadata
    from yonga.domain.plans import ExporterRequirements, ExportRequest, ExportResult

__all__ = [
    "CommandResult",
    "CommandRunner",
    "ExportEnvironments",
    "Exporter",
    "FilesystemProbe",
    "HardwareProbe",
    "HubClient",
    "HubSearch",
    "RuntimeProbe",
]


@runtime_checkable
class HardwareProbe(Protocol):
    """Collects host, device and package facts without changing anything."""

    def inventory(self, *, full: bool = False) -> HardwareInventory:
        """Return a snapshot of the host. Must be read-only and never need root."""
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
