"""Model identity, repository metadata and resolved conversion sources.

These types describe what a Hugging Face repository *is*. They never decide what
should be done with it: that is the job of the registry resolver and the
compatibility layer.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelFormat(str, Enum):
    """Weight/serialisation format observed in a repository."""

    OPENVINO_IR = "openvino_ir"
    SAFETENSORS = "safetensors"
    PYTORCH_BIN = "pytorch_bin"
    GGUF = "gguf"
    UNKNOWN = "unknown"


class PipelineKind(str, Enum):
    """The kind of GenAI pipeline a model needs."""

    LLM = "llm"
    VLM = "vlm"
    UNKNOWN = "unknown"


#: Preference order used to pick the weight total that represents a repository.
WEIGHT_FORMAT_PREFERENCE: tuple[ModelFormat, ...] = (
    ModelFormat.SAFETENSORS,
    ModelFormat.PYTORCH_BIN,
    ModelFormat.GGUF,
    ModelFormat.OPENVINO_IR,
)

#: Hugging Face pipeline tags that imply a vision-language pipeline.
VLM_PIPELINE_TAGS: frozenset[str] = frozenset(
    {"image-text-to-text", "visual-question-answering", "image-to-text"}
)

#: Hugging Face pipeline tags that imply a plain text generation pipeline.
LLM_PIPELINE_TAGS: frozenset[str] = frozenset({"text-generation"})

#: ``config.json`` keys whose presence proves a vision tower is part of the model.
VISION_CONFIG_KEYS: frozenset[str] = frozenset({"vision_config"})

_VLM_ARCHITECTURE_MARKER = "ForConditionalGeneration"
_LLM_ARCHITECTURE_SUFFIX = "ForCausalLM"
_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def infer_pipeline_kind(
    *,
    pipeline_tag: str | None = None,
    architectures: Sequence[str] = (),
    config: Mapping[str, object] | None = None,
    task: str | None = None,
) -> PipelineKind:
    """Infer the pipeline kind from repository metadata.

    The heuristic, in order:

    1. a pipeline tag (or, when absent, the task) in :data:`VLM_PIPELINE_TAGS`
       means :data:`PipelineKind.VLM`;
    2. an architecture containing ``ForConditionalGeneration`` *together with* a
       vision block in ``config.json`` (see :data:`VISION_CONFIG_KEYS`) also means
       :data:`PipelineKind.VLM` — the architecture alone is not enough, because
       plenty of text-only seq2seq models use the same suffix;
    3. the ``text-generation`` tag or an architecture ending in ``ForCausalLM``
       means :data:`PipelineKind.LLM`;
    4. anything else is :data:`PipelineKind.UNKNOWN`.

    This function is pure so that it can be unit-tested without a Hub client, and
    it is the only place the heuristic lives.
    """
    tag = (pipeline_tag or task or "").strip().lower()
    names = [name for name in architectures if name]
    payload = config or {}

    if tag in VLM_PIPELINE_TAGS:
        return PipelineKind.VLM

    has_vision_block = any(key in payload for key in VISION_CONFIG_KEYS)
    if has_vision_block and any(_VLM_ARCHITECTURE_MARKER in name for name in names):
        return PipelineKind.VLM

    if tag in LLM_PIPELINE_TAGS:
        return PipelineKind.LLM
    if any(name.endswith(_LLM_ARCHITECTURE_SUFFIX) for name in names):
        return PipelineKind.LLM

    return PipelineKind.UNKNOWN


class ModelIdentity(BaseModel):
    """An immutable, hashable reference to a repository and optional revision."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    repo_id: str
    revision: str | None = None

    @field_validator("repo_id")
    @classmethod
    def _clean_repo_id(cls, value: str) -> str:
        """Trim surrounding whitespace and slashes; a blank repo id is invalid."""
        cleaned = value.strip().strip("/")
        if not cleaned:
            raise ValueError("repo_id must not be empty")
        return cleaned

    @field_validator("revision")
    @classmethod
    def _clean_revision(cls, value: str | None) -> str | None:
        """Treat a blank revision as "unpinned" rather than as an empty string."""
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @property
    def namespace(self) -> str | None:
        """Owner part of the repository id, or ``None`` for a bare name."""
        if "/" in self.repo_id:
            return self.repo_id.split("/", 1)[0]
        return None

    @property
    def name(self) -> str:
        """Repository name without its namespace."""
        return self.repo_id.rsplit("/", 1)[-1]

    @property
    def slug(self) -> str:
        """Filesystem-safe directory name: ``a/b`` becomes ``a--b``.

        The namespace separator becomes ``--`` and every character outside
        ``[A-Za-z0-9._-]`` becomes ``_``, so the slug is safe on any filesystem
        yonga supports and still readable.
        """
        return _SLUG_UNSAFE.sub("_", self.repo_id.replace("/", "--"))

    def with_revision(self, rev: str) -> ModelIdentity:
        """Return a copy of this identity pinned to *rev*."""
        return ModelIdentity(repo_id=self.repo_id, revision=rev)

    def __str__(self) -> str:
        """Render as ``repo_id@revision``, or just ``repo_id`` when unpinned."""
        if self.revision:
            return f"{self.repo_id}@{self.revision}"
        return self.repo_id

    @classmethod
    def parse(cls, raw: str) -> ModelIdentity:
        """Parse ``"namespace/name"`` or ``"namespace/name@revision"``."""
        text = raw.strip()
        if not text:
            raise ValueError("model reference must not be empty")
        if "@" in text:
            repo_id, _, revision = text.rpartition("@")
            return cls(repo_id=repo_id, revision=revision or None)
        return cls(repo_id=text)


class RepoFile(BaseModel):
    """One file listed in a repository, with its size when the Hub reported it."""

    model_config = ConfigDict(extra="ignore")

    path: str
    size: int | None = None
    lfs: bool = False

    @property
    def suffix(self) -> str:
        """Lower-cased file extension including the dot, e.g. ``.safetensors``."""
        return PurePosixPath(self.path).suffix.lower()

    @property
    def name(self) -> str:
        """File name without its directory components."""
        return PurePosixPath(self.path).name


class FormatObservation(BaseModel):
    """Evidence that a repository contains weights in a particular format."""

    model_config = ConfigDict(extra="ignore")

    format: ModelFormat
    files: list[str] = Field(default_factory=list)
    total_bytes: int = 0
    complete: bool = False
    notes: list[str] = Field(default_factory=list)


class ModelMetadata(BaseModel):
    """Low-cost repository metadata gathered before any large download."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    identity: ModelIdentity
    resolved_revision: str | None = None
    exists: bool = True
    gated: bool = False
    private: bool = False
    license: str | None = None
    pipeline_tag: str | None = None
    tags: list[str] = Field(default_factory=list)
    library_name: str | None = None
    architectures: list[str] = Field(default_factory=list)
    model_type: str | None = None
    task: str | None = None
    config: dict[str, object] = Field(default_factory=dict)
    files: list[RepoFile] = Field(default_factory=list)
    formats: list[FormatObservation] = Field(default_factory=list)
    requires_remote_code: bool = False
    remote_code_files: list[str] = Field(default_factory=list)
    parameter_count: int | None = None
    gguf_metadata: dict[str, object] = Field(default_factory=dict)
    downloads: int | None = None
    likes: int | None = None
    last_modified: str | None = None

    @model_validator(mode="after")
    def _derive_from_config(self) -> ModelMetadata:
        """Fill architectures, model type and task from the parsed config.

        Adapters differ in how much they pre-extract; deriving here keeps the
        metadata self-consistent regardless of which one produced it.
        """
        if not self.architectures:
            raw_architectures = self.config.get("architectures")
            if isinstance(raw_architectures, list):
                self.architectures = [item for item in raw_architectures if isinstance(item, str)]
        if self.model_type is None:
            raw_type = self.config.get("model_type")
            if isinstance(raw_type, str) and raw_type.strip():
                self.model_type = raw_type.strip()
        if self.task is None and self.pipeline_tag is not None:
            self.task = self.pipeline_tag
        return self

    def _observation(self, fmt: ModelFormat) -> FormatObservation | None:
        """Return the strongest observation recorded for *fmt*, if any."""
        best: FormatObservation | None = None
        for observation in self.formats:
            if observation.format is not fmt:
                continue
            if best is None:
                best = observation
                continue
            if (observation.complete, observation.total_bytes) > (best.complete, best.total_bytes):
                best = observation
        return best

    def _weight_observation(self) -> FormatObservation | None:
        """Pick the observation that best represents this repository's weights.

        Complete observations win over incomplete ones; within each pass the
        :data:`WEIGHT_FORMAT_PREFERENCE` order decides. Observations without a
        known byte total are skipped because they cannot answer "how big is it".
        """
        for require_complete in (True, False):
            for fmt in WEIGHT_FORMAT_PREFERENCE:
                observation = self._observation(fmt)
                if observation is None or observation.total_bytes <= 0:
                    continue
                if require_complete and not observation.complete:
                    continue
                return observation
        return None

    @property
    def weight_bytes(self) -> int:
        """Total size of the most complete weight-bearing format, or 0 if unknown."""
        observation = self._weight_observation()
        return observation.total_bytes if observation is not None else 0

    def has_format(self, fmt: ModelFormat) -> bool:
        """True when *fmt* was observed in this repository.

        Contract deviation: spelled as a method, not a property, because the
        contract sketch declares it with an argument.
        """
        return any(observation.format is fmt for observation in self.formats)

    @property
    def primary_format(self) -> ModelFormat:
        """The format a conversion would read, using the same preference as ``weight_bytes``."""
        observation = self._weight_observation()
        if observation is not None:
            return observation.format
        for fmt in WEIGHT_FORMAT_PREFERENCE:
            if self.has_format(fmt):
                return fmt
        return ModelFormat.UNKNOWN

    @property
    def pipeline_kind(self) -> PipelineKind:
        """Pipeline kind inferred by :func:`infer_pipeline_kind`."""
        return infer_pipeline_kind(
            pipeline_tag=self.pipeline_tag,
            architectures=self.architectures,
            config=self.config,
            task=self.task,
        )


class SourceRelationship(str, Enum):
    """How a conversion source relates to the model the user asked for."""

    ORIGINAL = "original"
    CONVERTED = "converted"
    LOSSLESS_CONVERSION = "lossless_conversion"
    QUANTIZED = "quantized"


class SourceEvidence(BaseModel):
    """A citation supporting a lineage claim between two repositories."""

    model_config = ConfigDict(extra="ignore")

    kind: str
    url: str | None = None
    note: str | None = None


class ResolvedSource(BaseModel):
    """The repository/revision that will actually be converted, and why."""

    model_config = ConfigDict(extra="ignore")

    requested: ModelIdentity
    source: ModelIdentity
    relationship: SourceRelationship = SourceRelationship.ORIGINAL
    format: ModelFormat = ModelFormat.UNKNOWN
    already_openvino: bool = False
    evidence: list[SourceEvidence] = Field(default_factory=list)
    requires_confirmation: bool = False
    reasons: list[str] = Field(default_factory=list)
    rule_id: str | None = None
    metadata: ModelMetadata | None = None

    @model_validator(mode="after")
    def _sync_openvino_flag(self) -> ResolvedSource:
        """An OpenVINO IR source is always an OpenVINO artifact."""
        if self.format is ModelFormat.OPENVINO_IR and not self.already_openvino:
            self.already_openvino = True
        return self

    @property
    def is_derivative(self) -> bool:
        """True when the conversion input is not the requested repository itself."""
        return (
            self.relationship is not SourceRelationship.ORIGINAL
            or self.source.repo_id != self.requested.repo_id
        )
