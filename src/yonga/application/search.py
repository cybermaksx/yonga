"""``yonga search``: Hub search, enriched with what the registry knows.

``docs/CLI_SPEC.md`` section 5 asks for Hugging Face search "enriched with
compatibility where architecture metadata is available", and forbids claiming
compatibility without evidence. Two things keep this honest:

* **Identity comes from the same sources ``inspect`` uses** -- the Hub's config
  summary for a repository with ``config.json``, the GGUF header otherwise --
  and goes through the same registry resolution, so search and inspect cannot
  disagree about what a repository is. Repository tags are never used as
  identity; a model with no identifiable architecture is reported ``unknown``.
* **Compatibility is per architecture, not per repository.** A search response
  carries no file list, so it cannot say whether a hit can actually be prepared:
  ``lmstudio-community/Qwen3.5-9B-MLX-4bit`` reports ``qwen3_5`` but holds MLX
  weights the exporter cannot convert. Search therefore shows a *format hint*
  from the Hub's tags beside the registry's device knowledge, and points to
  ``yonga inspect`` for the answer that depends on the files.

Search does not probe the host either, so it works without OpenVINO installed.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from yonga.adapters import HubSearch
from yonga.application.inspect_model import identified_from
from yonga.domain.compatibility import DeviceCompatibility, DeviceStatus, EvidenceLevel
from yonga.domain.hardware import DEVICE_ORDER
from yonga.domain.models import ModelMetadata
from yonga.registry.resolver import RegistryResolver

__all__ = [
    "FORMAT_HINT_TAGS",
    "MAX_SEARCH_LIMIT",
    "SearchHit",
    "SearchReport",
    "SearchService",
    "format_hints",
]

#: The Hub is asked for at most this many results at once.
MAX_SEARCH_LIMIT: int = 100

#: Tags (and library names) that say something about how a repository's weights
#: are stored. The Hub adds most of these itself from the files it sees, which is
#: why they are acceptable as a *format hint* even though tags are never used to
#: identify an architecture. Quantization formats are listed because the OpenVINO
#: exporter generally cannot convert from them.
FORMAT_HINT_TAGS: tuple[str, ...] = (
    "openvino",
    "safetensors",
    "gguf",
    "mlx",
    "awq",
    "gptq",
    "exl2",
    "bitsandbytes",
    "onnx",
)

PER_ARCHITECTURE_NOTE = (
    "Device compatibility is what the registry knows about each architecture, not a check of "
    "this host or of a repository's files. Formats are hinted from Hub tags. Run "
    "`yonga inspect MODEL` to see whether a repository can actually be prepared here."
)


class SearchHit(BaseModel):
    """One search result, with the registry's view of its architecture."""

    model_config = ConfigDict(extra="ignore")

    repo_id: str
    task: str | None = None
    architecture: str | None = None
    architecture_rule_id: str | None = None
    identified_from: str | None = None
    """Set when the architecture came from a hint (the GGUF header), not config.json."""
    formats: list[str] = Field(default_factory=list)
    """Storage formats hinted by the Hub's tags; a hint, not a file inventory."""
    compatibility: list[DeviceCompatibility] = Field(default_factory=list)
    derivative: str | None = None
    """A registry-declared conversion source, when this repository has one."""
    parameters: int | None = None
    downloads: int | None = None
    likes: int | None = None
    gated: bool = False

    def compatibility_for(self, device: str) -> DeviceCompatibility | None:
        """The entry for one device, or None."""
        wanted = device.strip().upper()
        return next((entry for entry in self.compatibility if entry.device == wanted), None)


class SearchReport(BaseModel):
    """What ``yonga search`` found."""

    model_config = ConfigDict(extra="ignore")

    query: str
    hits: list[SearchHit] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def to_json(self) -> dict[str, object]:
        """A JSON-serialisable form for ``--json``."""
        return self.model_dump(mode="json")


class SearchService:
    """Searches the Hub and describes each hit through the registry."""

    def __init__(self, hub: HubSearch, registry_resolver: RegistryResolver) -> None:
        """Wire the service to a Hub searcher and the registry."""
        self._hub = hub
        self._registry = registry_resolver

    def search(self, query: str, *, limit: int = 20) -> SearchReport:
        """Run one Hub search and enrich every hit; never fetches repository files."""
        bounded = max(1, min(limit, MAX_SEARCH_LIMIT))
        results = self._hub.search(query, limit=bounded)
        return SearchReport(
            query=query,
            hits=[self._hit(metadata) for metadata in results],
            notes=[PER_ARCHITECTURE_NOTE],
        )

    def _hit(self, metadata: ModelMetadata) -> SearchHit:
        rule, explanation = self._registry.resolve_architecture(metadata)
        architecture = metadata.model_type
        if architecture is None and rule is not None and rule.match.model_types:
            # Identified through an alias: name it the way the registry does.
            architecture = rule.match.model_types[0]
        candidates, _ = self._registry.resolve_derivatives(metadata.identity)
        return SearchHit(
            repo_id=metadata.identity.repo_id,
            task=metadata.pipeline_tag or metadata.task,
            architecture=architecture,
            architecture_rule_id=None if rule is None else rule.id,
            identified_from=identified_from(rule, explanation),
            formats=format_hints(metadata),
            compatibility=_every_device(self._registry.device_compatibility(arch_rule=rule)),
            derivative=candidates[0].repo_id if candidates else None,
            parameters=metadata.parameter_count,
            downloads=metadata.downloads,
            likes=metadata.likes,
            gated=metadata.gated,
        )


def format_hints(metadata: ModelMetadata) -> list[str]:
    """Storage formats suggested by a repository's Hub tags and library name."""
    seen = {tag.strip().lower() for tag in metadata.tags}
    if metadata.library_name:
        seen.add(metadata.library_name.strip().lower())
    return [name for name in FORMAT_HINT_TAGS if name in seen]


def _every_device(entries: list[DeviceCompatibility]) -> list[DeviceCompatibility]:
    """Report CPU, GPU and NPU even when the registry has nothing to say about one.

    A device the registry knows nothing about is ``unknown`` -- stated, not
    omitted, so the absence of a claim is never mistaken for support.
    """
    known = {entry.device: entry for entry in entries}
    ordered = [
        known.get(device)
        or DeviceCompatibility(
            device=device,
            status=DeviceStatus.UNKNOWN,
            evidence=EvidenceLevel.UNKNOWN,
            reason="no_registry_rule",
        )
        for device in DEVICE_ORDER
    ]
    extra = [entry for name, entry in known.items() if name not in DEVICE_ORDER]
    return [*ordered, *extra]
