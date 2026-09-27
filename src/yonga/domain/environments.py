"""Managed export environments: what one is, and how one is identified.

Conversion needs a specific set of packages, and the set differs by model
family: Qwen3.5 exports only with ``transformers==5.2.*``, which cannot coexist
with the newer Transformers another model needs. Rather than asking the user to
reshape their own environment -- ``SPEC.md`` section 18.5 forbids yonga
installing into it -- yonga builds an isolated virtual environment used only by
the export subprocess.

An :class:`EnvironmentSpec` names the exact packages; its :attr:`~EnvironmentSpec.key`
is a digest of them, so the same requirements always map to the same
environment and a different pin always maps to a different one.

Package *names* are fixed here, in code. Registry data can choose a version
specifier for a package this module already lists, never a package of its own:
the same rule that keeps repairs from executing instructions taken from YAML.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ENVIRONMENT_PACKAGES",
    "ESTIMATED_ENVIRONMENT_BYTES",
    "TORCH_CPU_INDEX",
    "TORCH_INDEX_PACKAGES",
    "EnvironmentSpec",
    "ExportEnvironment",
]

#: Every package a managed export environment may contain, in install order.
#: The :data:`TORCH_INDEX_PACKAGES` are installed first, together, from
#: :data:`TORCH_CPU_INDEX`.
ENVIRONMENT_PACKAGES: tuple[str, ...] = (
    "torch",
    "torchvision",
    "transformers",
    "optimum",
    "optimum-intel",
    "nncf",
    "openvino",
    "openvino-tokenizers",
)

#: PyTorch's CPU-only wheel index. Exporting runs on the CPU, and the default
#: PyPI build of torch on Linux carries CUDA libraries: on the reference host
#: that is 1.2 GB of torch plus 3.2 GB of NVIDIA packages, against a 196 MB
#: CPU wheel, on a machine with no NVIDIA GPU at all.
TORCH_CPU_INDEX: str = "https://download.pytorch.org/whl/cpu"

#: Packages built against torch, which must come from the same index as torch.
#: ``optimum-intel`` requires ``torchvision``; left to PyPI, pip fetched the
#: CUDA-built wheel, whose compiled operators cannot register against the CPU
#: torch ("operator torchvision::nms does not exist"), after which Transformers
#: could not even import ``PreTrainedModel``. Observed on the reference host.
TORCH_INDEX_PACKAGES: tuple[str, ...] = ("torch", "torchvision")

#: A deliberately generous estimate of one environment on disk (CPU torch,
#: Transformers, OpenVINO, NNCF and their dependencies), used by preflight.
ESTIMATED_ENVIRONMENT_BYTES: int = 3 * 2**30


class EnvironmentSpec(BaseModel):
    """The exact package set one managed export environment is built from."""

    model_config = ConfigDict(frozen=True)

    python: str
    """``major.minor`` of the interpreter the environment is created from."""
    packages: dict[str, str] = Field(default_factory=dict)
    """Package name to PEP 440 specifier, e.g. ``{"transformers": "==5.2.*"}``."""
    reason: str = ""
    """Why this environment exists, for the plan and ``yonga env list``."""

    @field_validator("packages")
    @classmethod
    def _known_packages_and_valid_specifiers(cls, value: dict[str, str]) -> dict[str, str]:
        for name, specifier in value.items():
            if name not in ENVIRONMENT_PACKAGES:
                raise ValueError(f"{name!r} is not a package a managed environment may contain")
            try:
                SpecifierSet(specifier)
            except InvalidSpecifier as error:
                raise ValueError(f"invalid specifier for {name}: {specifier!r}") from error
        return dict(sorted(value.items()))

    @property
    def key(self) -> str:
        """A short, stable digest of the interpreter and every pin."""
        canonical = json.dumps({"python": self.python, "packages": self.packages}, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    @property
    def name(self) -> str:
        """The environment's directory name."""
        return f"export-{self.key}"

    def requirement(self, package: str) -> str:
        """``package`` with its specifier, as pip expects it (``transformers==5.2.*``)."""
        return f"{package}{self.packages.get(package, '')}"


class ExportEnvironment(BaseModel):
    """A built, verified managed export environment on disk."""

    model_config = ConfigDict(extra="ignore")

    spec: EnvironmentSpec
    path: Path
    installed: dict[str, str] = Field(default_factory=dict)
    """Package versions actually installed, read back from the environment."""
    created_at: datetime
    size_bytes: int = 0

    @property
    def python(self) -> Path:
        """The environment's interpreter."""
        return self.path / "bin" / "python"

    @property
    def key(self) -> str:
        """The spec's digest, which is also the directory suffix."""
        return self.spec.key
