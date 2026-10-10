"""``yonga eval``: measure how much a quantization changed a model's predictions.

Two passes, one model in memory at a time, because two 9B artifacts do not fit
together on the reference host. The reference reads every passage first and
its full next-token log-probabilities are spilled to a memory-mapped file in
the managed cache (never the system temp, which is RAM-backed here: the
Qwen3.5 reference is about 1.9 GB of float32). The candidate then reads the
same passages and each position is compared against the spilled row.

Refusals, all before any model is loaded:

* the two references must be different, prepared artifacts;
* they must descend from the same source repository and revision, or the
  "difference" would be between two models rather than two quantizations.

And one after tokenising: both artifacts must turn the corpus into exactly the
same token ids. Different ids mean different vocabularies, and comparing
distributions over different vocabularies means nothing.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from yonga import __version__
from yonga.application.progress import ProgressEvent, ProgressKind, ProgressSink, null_sink
from yonga.config.paths import Paths
from yonga.config.settings import Settings
from yonga.domain.artifacts import ArtifactManifest
from yonga.domain.diagnostics import ConfigurationError, FailureCategory, diagnostic
from yonga.domain.evaluation import (
    EvaluatedArtifact,
    EvaluationResult,
    PassageScore,
    summarise,
)

__all__ = [
    "CORPUS_RESOURCE",
    "Corpus",
    "EvaluationService",
    "LogitsScorer",
    "evaluation_result_path",
    "load_corpus",
]

CORPUS_RESOURCE = "data/eval_corpus.json"


class LogitsScorer(Protocol):
    """What the service needs from a loaded artifact."""

    def tokenize(self, text: str) -> npt.NDArray[np.int64]:
        """The artifact's token ids for ``text``."""
        ...

    def log_probs(self, ids: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
        """Next-token log-probabilities ``[len(ids), vocab]``."""
        ...

    def close(self) -> None:
        """Release the model."""
        ...


#: Builds a scorer for an artifact directory on a device.
ScorerFactory = Callable[[Path, str], LogitsScorer]


class Passage(BaseModel):
    """One piece of the corpus."""

    model_config = ConfigDict(extra="ignore")

    id: str
    genre: str = ""
    text: str


class Corpus(BaseModel):
    """The text both artifacts read."""

    model_config = ConfigDict(extra="ignore")

    id: str
    passages: list[Passage]
    sha256: str = ""


class _Store(Protocol):
    def find(self, reference: str | Path) -> Path | None: ...

    def manifest_for(self, artifact_dir: Path) -> ArtifactManifest | None: ...


def load_corpus(path: Path | None = None) -> Corpus:
    """The shipped corpus, or a user's: JSON with ``passages``, or plain text.

    A plain-text file is split on blank lines into passages.
    """
    if path is None:
        text = resources.files("yonga").joinpath(CORPUS_RESOURCE).read_text(encoding="utf-8")
        corpus = Corpus.model_validate_json(text)
    else:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            corpus = Corpus.model_validate_json(text)
        else:
            blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
            corpus = Corpus(
                id=path.name,
                passages=[Passage(id=f"p{i + 1}", text=block) for i, block in enumerate(blocks)],
            )
    corpus.sha256 = hashlib.sha256(text.encode()).hexdigest()
    return corpus


def evaluation_result_path(paths: Paths, result: EvaluationResult) -> Path:
    """Return the file an evaluation result is saved to, under ``<data>/evaluations/``."""
    return paths.data_dir / "evaluations" / f"{result.evaluation_id}.json"


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


class EvaluationService:
    """Runs teacher-forced comparisons and saves the results."""

    def __init__(
        self,
        store: _Store,
        scorer_factory: ScorerFactory,
        settings: Settings,
        *,
        progress: ProgressSink | None = None,
        package_versions: Callable[[], dict[str, str]] | None = None,
    ) -> None:
        """Bind the service to the artifact store and a scorer factory."""
        self._store = store
        self._factory = scorer_factory
        self._settings = settings
        self._progress = progress if progress is not None else null_sink()
        self._versions = package_versions or dict

    def evaluate(
        self,
        candidate: str,
        reference: str,
        *,
        device: str = "CPU",
        corpus: Corpus | None = None,
        save: bool = True,
        allow_different_sources: bool = False,
    ) -> EvaluationResult:
        """Compare ``candidate`` against ``reference`` over ``corpus``.

        ``allow_different_sources`` accepts artifacts converted from different
        repositories -- a GGUF release against the original weights, say --
        on the caller's word that they hold the same model. The tokenizer check
        still applies, and the result records the assertion.
        """
        started = time.perf_counter()
        target = device.upper()
        text = corpus or load_corpus()
        if not text.passages:
            raise _refusal("eval.corpus_empty", "The corpus is empty", "There is nothing to read.")
        reference_side = self._artifact(reference)
        candidate_side = self._artifact(candidate)
        if reference_side.path == candidate_side.path:
            raise _refusal(
                "eval.same_artifact",
                "The candidate is the reference",
                "Comparing an artifact with itself measures nothing.",
            )
        different = (reference_side.source_repo_id, reference_side.source_revision) != (
            candidate_side.source_repo_id,
            candidate_side.source_revision,
        )
        if different and not allow_different_sources:
            raise _refusal(
                "eval.different_sources",
                "The two artifacts were converted from different weights",
                f"{reference_side.source_repo_id}@{_short(reference_side.source_revision)} "
                f"against {candidate_side.source_repo_id}@"
                f"{_short(candidate_side.source_revision)}: the difference would be between "
                "two models, not two quantizations of one.",
                "Prepare the reference from the same source and revision.",
                "If both hold the same model in different formats (a GGUF release and the "
                "original weights), pass --allow-different-sources.",
            )

        spill_dir = self._settings.paths.tmp_dir
        spill_dir.mkdir(parents=True, exist_ok=True)
        spill_path = spill_dir / f"eval-{int(time.time() * 1000)}.f32"
        try:
            reference_ids, reference_scores, spill, properties = self._reference_pass(
                reference_side, text, target, spill_path
            )
            passages = self._candidate_pass(
                candidate_side, text, target, reference_ids, reference_scores, spill
            )
        finally:
            with contextlib.suppress(OSError):
                spill_path.unlink()

        created = datetime.now(UTC)
        result = EvaluationResult(
            evaluation_id=(
                f"{created:%Y%m%dT%H%M%SZ}-{target.lower()}-"
                f"{candidate_side.artifact_id}-vs-{reference_side.artifact_id}"
            ),
            created_at=created,
            tool_version=__version__,
            device=target,
            runtime_properties=properties,
            corpus_id=text.id,
            corpus_sha256=text.sha256,
            reference=reference_side,
            candidate=candidate_side,
            passages=passages,
            summary=summarise(passages),
            package_versions=self._versions(),
            duration_seconds=time.perf_counter() - started,
            notes=[
                f"Differences are relative to {reference_side.artifact_id} "
                f"({reference_side.weight_format or 'unknown format'}), not to the original "
                "weights.",
                *(
                    [
                        f"The two were converted from different repositories "
                        f"({candidate_side.source_repo_id} and {reference_side.source_repo_id}); "
                        "that they hold the same model was asserted with "
                        "--allow-different-sources. Their tokenizers agreed on every token."
                    ]
                    if different
                    else []
                ),
            ],
        )
        if save:
            self.save(result)
        return result

    def save(self, result: EvaluationResult) -> Path:
        """Write ``result`` under ``<data>/evaluations/`` and return the file."""
        target = evaluation_result_path(self._settings.paths, result)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return target

    # --- passes -----------------------------------------------------------------------

    def _reference_pass(
        self, side: EvaluatedArtifact, corpus: Corpus, device: str, spill_path: Path
    ) -> tuple[
        list[npt.NDArray[np.int64]], list[dict[str, Any]], np.memmap[Any, Any], dict[str, str]
    ]:
        self._emit(f"Loading the reference {side.artifact_id} on {device}")
        scorer = self._factory(side.path, device)
        try:
            ids = [scorer.tokenize(passage.text) for passage in corpus.passages]
            too_short = [p.id for p, i in zip(corpus.passages, ids, strict=True) if len(i) < 2]
            if too_short:
                raise _refusal(
                    "eval.passage_too_short",
                    "A passage is too short to score",
                    f"Passages {', '.join(too_short)} tokenise to fewer than two tokens.",
                )
            first = scorer.log_probs(ids[0])
            vocab = int(first.shape[1])
            rows = sum(len(i) - 1 for i in ids)
            spill: np.memmap[Any, Any] = np.memmap(
                spill_path, dtype=np.float32, mode="w+", shape=(rows, vocab)
            )
            scores: list[dict[str, Any]] = []
            offset = 0
            for index, (passage, tokens) in enumerate(zip(corpus.passages, ids, strict=True)):
                self._emit(f"Reference: {passage.id} ({len(tokens)} tokens)")
                lp = first if index == 0 else scorer.log_probs(tokens)
                used = lp[:-1]
                count = used.shape[0]
                spill[offset : offset + count] = used
                targets = tokens[1:]
                scores.append(
                    {
                        "offset": offset,
                        "count": count,
                        "nll": float(-used[np.arange(count), targets].mean()),
                        "argmax": used.argmax(axis=-1),
                    }
                )
                offset += count
            spill.flush()
            properties = dict(getattr(scorer, "properties", {}) or {})
        finally:
            scorer.close()
        return ids, scores, spill, properties

    def _candidate_pass(
        self,
        side: EvaluatedArtifact,
        corpus: Corpus,
        device: str,
        reference_ids: Sequence[npt.NDArray[np.int64]],
        reference_scores: Sequence[dict[str, Any]],
        spill: np.memmap[Any, Any],
    ) -> list[PassageScore]:
        self._emit(f"Loading the candidate {side.artifact_id} on {device}")
        scorer = self._factory(side.path, device)
        passages: list[PassageScore] = []
        try:
            for passage, tokens, reference in zip(
                corpus.passages, reference_ids, reference_scores, strict=True
            ):
                mine = scorer.tokenize(passage.text)
                if mine.shape != tokens.shape or not np.array_equal(mine, tokens):
                    raise _refusal(
                        "eval.tokenizer_mismatch",
                        "The two artifacts tokenise the text differently",
                        f"Passage {passage.id} becomes different token ids; their vocabularies "
                        "differ, so their predictions cannot be compared.",
                    )
                self._emit(f"Candidate: {passage.id} ({len(tokens)} tokens)")
                lp = scorer.log_probs(tokens)[:-1]
                count = int(reference["count"])
                ref = np.asarray(spill[reference["offset"] : reference["offset"] + count])
                if lp.shape != ref.shape:
                    raise _refusal(
                        "eval.vocab_mismatch",
                        "The two artifacts predict over different vocabularies",
                        f"The candidate scores {lp.shape[-1]} tokens per position and the "
                        f"reference {ref.shape[-1]}; their distributions cannot be compared.",
                    )
                kl = _kl_rows(ref, lp)
                targets = tokens[1:]
                passages.append(
                    PassageScore(
                        passage_id=passage.id,
                        genre=passage.genre,
                        positions=count,
                        reference_nll=float(reference["nll"]),
                        candidate_nll=float(-lp[np.arange(count), targets].mean()),
                        kl_mean=float(kl.mean()),
                        kl_max=float(kl.max()),
                        top1_agreement=float((lp.argmax(axis=-1) == reference["argmax"]).mean()),
                    )
                )
        finally:
            scorer.close()
        return passages

    # --- helpers ----------------------------------------------------------------------

    def _artifact(self, reference: str) -> EvaluatedArtifact:
        directory = self._store.find(reference)
        manifest = self._store.manifest_for(directory) if directory is not None else None
        if directory is None or manifest is None:
            raise _refusal(
                "eval.artifact_not_found",
                "No prepared artifact by that name",
                f"{reference!r} is not an artifact yonga prepared.",
                "Run 'yonga list' to see the prepared artifacts.",
            )
        weight = manifest.export.arguments.get("weight_format")
        return EvaluatedArtifact(
            artifact_id=manifest.artifact_id,
            path=directory,
            profile=manifest.export.profile,
            weight_format=str(weight) if weight else None,
            source_repo_id=manifest.resolved_source.repo_id,
            source_revision=manifest.resolved_source.resolved_revision,
        )

    def _emit(self, message: str) -> None:
        with contextlib.suppress(Exception):
            self._progress.emit(ProgressEvent(kind=ProgressKind.STAGE, message=message))


def _kl_rows(
    reference: npt.NDArray[np.float32],
    candidate: npt.NDArray[np.float32],
    *,
    rows: int = 32,
) -> npt.NDArray[np.float64]:
    """``KL(reference || candidate)`` per row, in nats, from log-probabilities."""
    result = np.empty(reference.shape[0], dtype=np.float64)
    for start in range(0, reference.shape[0], rows):
        ref = reference[start : start + rows].astype(np.float64)
        cand = candidate[start : start + rows].astype(np.float64)
        result[start : start + rows] = (np.exp(ref) * (ref - cand)).sum(axis=-1)
    return np.maximum(result, 0.0)


def _short(revision: str | None) -> str:
    return (revision or "unknown")[:12]


def result_payload(result: EvaluationResult) -> dict[str, Any]:
    """The saved document, as JSON-ready data."""
    payload: dict[str, Any] = json.loads(result.model_dump_json())
    return payload
