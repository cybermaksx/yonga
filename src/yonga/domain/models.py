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
    """Evidence that a repository contains weights in a particular format.

    The two byte counters answer different questions and must not be confused:
    ``total_bytes`` describes the *repository*, ``primary_bytes`` describes the
    *model a user ends up running*. They are equal whenever every file of the
    format belongs to one model, and they differ when a repository publishes
    several interchangeable copies of the same model — the GGUF convention of
    shipping one file per quantization being the case that matters today.
    """

    model_config = ConfigDict(extra="ignore")

    format: ModelFormat
    files: list[str] = Field(default_factory=list)
    total_bytes: int = 0
    """Honest sum of every file of this format listed in the repository.

    This is a download-side number: it is what a full snapshot of the format
    costs, and it is the right input to a "how much will I fetch" estimate. It
    is *not* a statement about how much of that payload a single conversion
    reads, because a repository may carry several alternative copies of one
    model. ``0`` means the sizes were unknown, never "empty".
    """

    primary_bytes: int = 0
    """Size of ONE usable model in this format — the conversion-side number.

    A conversion reads exactly one model, so this is what preflight, temp-space
    and artifact-size estimates must be derived from. For sharded safetensors
    and PyTorch checkpoints every shard is part of that one model, so this
    equals ``total_bytes``. For a GGUF repository publishing BF16, Q4_K_M, Q6_K
    and Q8_0 copies plus a projector side-car, it is the size of the single
    file a user actually runs. ``0`` means unknown, never "empty".
    """

    complete: bool = False
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _default_primary_to_total(self) -> FormatObservation:
        """Treat an *unmentioned* ``primary_bytes`` as "one file set, one model".

        That is the truth for every format except GGUF, and it keeps callers
        that predate this field — and the many fixtures that construct an
        observation by hand — describing exactly what they meant.

        A caller that did name the field is believed, ``0`` included. ``0`` is
        the documented way to say "unknown", and a detector reaches it on a real
        inventory: a GGUF repository whose only *sized* file is the multimodal
        projector has no measurable runnable model. Copying ``total_bytes`` over
        that answer would promote the one file the detector explicitly
        classified as not runnable to "the size of one usable model", which is
        both wrong and unrecoverable downstream.
        """
        if "primary_bytes" in self.model_fields_set:
            return self
        if self.primary_bytes <= 0 and self.total_bytes > 0:
            self.primary_bytes = self.total_bytes
        return self


#: Safetensors dtypes that hold integers. Model weights stored this way are packed
#: quantized weights (several values per element), so the element count understates
#: the model; a small integer share is just buffers such as position ids.
_INTEGER_DTYPES: frozenset[str] = frozenset({"I8", "U8", "I16", "U16", "I32", "U32", "I64", "U64"})
#: Above this share of integer elements a checkpoint is treated as packed.
_PACKED_SHARE: float = 0.01


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
    parameter_dtypes: dict[str, int] = Field(default_factory=dict)
    """Parameter count per stored dtype (``BF16``, ``I32``…), as the Hub reports it."""
    parameter_count_local: bool = False
    """Whether the counts were taken here, offline, from the cached safetensors
    headers rather than reported by the Hub. The method is the Hub's, so the
    number is the same; only where it came from differs."""
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

    def size_for_matching(self) -> tuple[int | None, str]:
        """The parameter count a profile's size band is judged on, and where it came from.

        ``None`` when there is no count, or when the count would mislead: a
        checkpoint storing packed integer weights (GPTQ, AWQ, bitsandbytes) is
        reported by the Hub as far fewer parameters than the model has, which
        is the unsafe direction for a "small model" band. The rules look only
        at dtypes and ``config.json`` keys, never at a model family. There is no
        fallback from bytes to parameters: bytes depend on the dtype.
        """
        if self.parameter_count is None or self.parameter_count <= 0:
            if self.parameter_count_local:
                return None, "the cached safetensors headers give no parameter total"
            return None, "the Hub reports no safetensors or GGUF parameter total"
        packed = {
            dtype: count
            for dtype, count in self.parameter_dtypes.items()
            if dtype.upper() in _INTEGER_DTYPES
        }
        packed_total = sum(packed.values())
        if packed_total > self.parameter_count * _PACKED_SHARE:
            dtype, count = max(packed.items(), key=lambda item: item[1])
            return None, (
                f"the checkpoint stores packed integer weights ({dtype}: {count:,}), "
                f"so the {'local' if self.parameter_count_local else 'Hub'} count understates "
                "the model"
            )
        if "quantization_config" in self.config:
            return None, "the checkpoint is already quantized (quantization_config in config.json)"
        if self.parameter_count_local:
            return self.parameter_count, "counted locally from the cached safetensors headers"
        if not self.parameter_dtypes and self.gguf_metadata:
            return self.parameter_count, "Hugging Face Hub count from GGUF metadata"
        return self.parameter_count, "Hugging Face Hub count from safetensors headers"

    @property
    def parameter_count_origin(self) -> str:
        """Where :attr:`parameter_count` came from, in a few words."""
        if self.parameter_count_local:
            return "counted locally from the cached safetensors headers"
        return "Hub count"

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
        :data:`WEIGHT_FORMAT_PREFERENCE` order decides. Observations that cannot
        size a single model are skipped because they cannot answer "how big is
        it".
        """
        for require_complete in (True, False):
            for fmt in WEIGHT_FORMAT_PREFERENCE:
                observation = self._observation(fmt)
                if observation is None or observation.primary_bytes <= 0:
                    continue
                if require_complete and not observation.complete:
                    continue
                return observation
        return None

    @property
    def primary_observation(self) -> FormatObservation | None:
        """The observation :attr:`weight_bytes` and :attr:`primary_format` read from."""
        return self._weight_observation()

    @property
    def weight_bytes(self) -> int:
        """Size of one usable model in the preferred format, or 0 if unknown.

        This is the conversion-side figure that drives preflight: a GGUF
        repository shipping five quantizations of the same model still only
        needs one of them. Use :attr:`repo_weight_bytes` when the question is
        how large the repository is rather than how large the model is.
        """
        observation = self._weight_observation()
        return observation.primary_bytes if observation is not None else 0

    @property
    def repo_weight_bytes(self) -> int:
        """Sum of every file of the preferred format in the repository, or 0.

        The display counterpart of :attr:`weight_bytes`, and the honest input to
        a full-snapshot download estimate. It is never smaller than
        :attr:`weight_bytes` once the Hub has reported file sizes.
        """
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
    gguf_file: str | None = None
    """For a GGUF source, the one file that is imported."""
    gguf_architecture: str | None = None
    """Its ``general.architecture``, read from its own header."""
    gguf_tensor_types: dict[str, int] = Field(default_factory=dict)
    """Its tensor types and how many tensors have each, read from its header."""
    gguf_file_bytes: int | None = None

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
