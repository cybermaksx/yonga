"""Quality evaluation: how far a quantized artifact's predictions move from a reference's.

``yonga eval`` answers the question ``bench`` cannot: not how fast an artifact
is, but how much its quantization changed what the model computes. It is a
teacher-forced comparison. Both artifacts read the same text, token by token,
and at every position their next-token distributions are compared:

* **KL divergence** ``KL(reference || candidate)``, in nats, averaged over
  positions: how much probability mass moved. Zero means identical
  predictions. It is the most sensitive of the three and the one to rank by;
* **top-1 agreement**: how often both put the same token first, which is what
  greedy decoding would have picked;
* **perplexity** of each artifact on the text, and the candidate's change
  against the reference.

No sampling is involved, so the numbers are deterministic for a given pair of
artifacts, device and corpus. They are relative to the *reference*, which is
itself an artifact: an INT8 reference is near-lossless, not lossless, and the
result says which reference it was measured against.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "EvaluatedArtifact",
    "EvaluationResult",
    "EvaluationSummary",
    "PassageScore",
]

EVALUATION_SCHEMA_VERSION = 1


class EvaluatedArtifact(BaseModel):
    """One side of the comparison."""

    model_config = ConfigDict(extra="ignore")

    artifact_id: str
    path: Path
    profile: str | None = None
    weight_format: str | None = None
    source_repo_id: str | None = None
    source_revision: str | None = None


class PassageScore(BaseModel):
    """The comparison over one corpus passage."""

    model_config = ConfigDict(extra="ignore")

    passage_id: str
    genre: str
    positions: int
    """Scored positions: the passage's tokens minus the first, which has no prediction."""
    reference_nll: float
    """Mean negative log-likelihood of the actual next token, in nats."""
    candidate_nll: float
    kl_mean: float
    kl_max: float
    top1_agreement: float

    @property
    def reference_perplexity(self) -> float:
        """``exp(reference_nll)``."""
        return math.exp(self.reference_nll)

    @property
    def candidate_perplexity(self) -> float:
        """``exp(candidate_nll)``."""
        return math.exp(self.candidate_nll)


class EvaluationSummary(BaseModel):
    """Position-weighted totals over the whole corpus."""

    model_config = ConfigDict(extra="ignore")

    positions: int
    reference_perplexity: float
    candidate_perplexity: float
    perplexity_change_pct: float
    kl_mean: float
    top1_agreement: float


class EvaluationResult(BaseModel):
    """A complete, saved evaluation."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = EVALUATION_SCHEMA_VERSION
    evaluation_id: str
    created_at: datetime
    tool_version: str
    device: str
    runtime_properties: dict[str, str] = Field(default_factory=dict)
    corpus_id: str
    corpus_sha256: str
    reference: EvaluatedArtifact
    candidate: EvaluatedArtifact
    passages: list[PassageScore]
    summary: EvaluationSummary
    package_versions: dict[str, str] = Field(default_factory=dict)
    duration_seconds: float = 0.0
    notes: list[str] = Field(default_factory=list)


def summarise(passages: list[PassageScore]) -> EvaluationSummary:
    """Position-weighted totals; every passage counts in proportion to its length."""
    total = sum(p.positions for p in passages)
    if total == 0:
        raise ValueError("nothing was scored")

    def weighted(values: list[float]) -> float:
        return sum(v * p.positions for v, p in zip(values, passages, strict=True)) / total

    reference_nll = weighted([p.reference_nll for p in passages])
    candidate_nll = weighted([p.candidate_nll for p in passages])
    reference_ppl = math.exp(reference_nll)
    candidate_ppl = math.exp(candidate_nll)
    return EvaluationSummary(
        positions=total,
        reference_perplexity=reference_ppl,
        candidate_perplexity=candidate_ppl,
        perplexity_change_pct=100.0 * (candidate_ppl - reference_ppl) / reference_ppl,
        kl_mean=weighted([p.kl_mean for p in passages]),
        top1_agreement=weighted([p.top1_agreement for p in passages]),
    )
