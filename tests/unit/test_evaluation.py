"""``yonga eval``: the arithmetic, the refusals, and the one-model-at-a-time passes.

Fake scorers return hand-made distributions, so every number asserted here
can be worked out on paper: identical models give zero KL and full agreement,
and a known perturbation gives a known divergence.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pytest

from yonga.adapters.openvino_logits import log_softmax
from yonga.application.evaluation import (
    Corpus,
    EvaluationService,
    Passage,
    _kl_rows,
    load_corpus,
)
from yonga.config.settings import Settings
from yonga.domain.artifacts import ArtifactManifest, ExportRecord, ManifestModelRef
from yonga.domain.diagnostics import ConfigurationError

VOCAB = 8


class FakeScorer:
    """Deterministic distributions: a base pattern, optionally perturbed."""

    loaded: ClassVar[list[str]] = []

    def __init__(self, name: str, *, perturb: float = 0.0, offset_ids: int = 0) -> None:
        self.name = name
        self.perturb = perturb
        self.offset_ids = offset_ids

    def tokenize(self, text: str) -> npt.NDArray[np.int64]:
        return np.array([(ord(c) + self.offset_ids) % VOCAB for c in text], dtype=np.int64)

    def log_probs(self, ids: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
        logits = np.zeros((len(ids), VOCAB), dtype=np.float64)
        for row, token in enumerate(ids):
            logits[row, (int(token) + 1) % VOCAB] = 3.0
            logits[row, (int(token) + 2) % VOCAB] = 1.0 + self.perturb
        return log_softmax(logits)

    def close(self) -> None:
        FakeScorer.loaded.append(f"close:{self.name}")


def manifest(artifact_id: str, *, weight: str, revision: str = "a" * 40) -> ArtifactManifest:
    return ArtifactManifest(
        artifact_id=artifact_id,
        created_at="2026-09-24T00:00:00Z",  # type: ignore[arg-type]
        tool_version="0.1.0",
        requested_model=ManifestModelRef(repo_id="org/model", resolved_revision=revision),
        resolved_source=ManifestModelRef(repo_id="org/model", resolved_revision=revision),
        export=ExportRecord(profile=f"cpu-{weight}", arguments={"weight_format": weight}),
    )


class FakeStore:
    def __init__(self, tmp_path: Path, manifests: dict[str, ArtifactManifest]) -> None:
        self.dirs = {name: tmp_path / name for name in manifests}
        self.manifests = {self.dirs[name]: m for name, m in manifests.items()}

    def find(self, reference: str | Path) -> Path | None:
        return self.dirs.get(str(reference))

    def manifest_for(self, artifact_dir: Path) -> ArtifactManifest | None:
        return self.manifests.get(artifact_dir)


CORPUS = Corpus(
    id="test",
    passages=[Passage(id="one", genre="a", text="hello world"), Passage(id="two", text="abc")],
)


def service(
    tmp_path: Path, scorers: dict[str, FakeScorer], manifests: dict[str, ArtifactManifest]
) -> EvaluationService:
    FakeScorer.loaded = []
    store = FakeStore(tmp_path, manifests)
    settings = Settings(
        config_dir=tmp_path / "config", cache_dir=tmp_path / "cache", data_dir=tmp_path / "data"
    )

    def open_scorer(path: Path, device: str) -> FakeScorer:
        FakeScorer.loaded.append(f"open:{path.name}")
        return scorers[path.name]

    return EvaluationService(store, open_scorer, settings)


def test_kl_of_identical_distributions_is_zero_and_known_otherwise() -> None:
    p = log_softmax(np.array([[0.0, math.log(3.0)]]))  # (1/4, 3/4)
    q = log_softmax(np.array([[0.0, 0.0]]))  # (1/2, 1/2)

    assert _kl_rows(p, p)[0] == pytest.approx(0.0, abs=1e-9)
    expected = 0.25 * math.log(0.25 / 0.5) + 0.75 * math.log(0.75 / 0.5)
    assert _kl_rows(p, q)[0] == pytest.approx(expected, rel=1e-6)


def test_an_identical_candidate_scores_zero_kl_and_full_agreement(tmp_path: Path) -> None:
    manifests = {"ref": manifest("ref", weight="fp16"), "cand": manifest("cand", weight="int4")}
    scorers = {"ref": FakeScorer("ref"), "cand": FakeScorer("cand")}

    result = service(tmp_path, scorers, manifests).evaluate("cand", "ref", corpus=CORPUS)

    assert result.summary.kl_mean == pytest.approx(0.0, abs=1e-9)
    assert result.summary.top1_agreement == 1.0
    assert result.summary.perplexity_change_pct == pytest.approx(0.0, abs=1e-9)
    assert result.summary.positions == len("hello world") - 1 + len("abc") - 1


def test_a_perturbed_candidate_moves_kl_and_perplexity(tmp_path: Path) -> None:
    manifests = {"ref": manifest("ref", weight="fp16"), "cand": manifest("cand", weight="int4")}
    scorers = {"ref": FakeScorer("ref"), "cand": FakeScorer("cand", perturb=1.5)}

    result = service(tmp_path, scorers, manifests).evaluate("cand", "ref", corpus=CORPUS)

    assert result.summary.kl_mean > 0.01
    assert result.summary.top1_agreement == 1.0, "the top token did not change"
    assert result.passages[0].kl_max >= result.passages[0].kl_mean
    assert "not to the original weights" in result.notes[0]


def test_only_one_model_is_loaded_at_a_time(tmp_path: Path) -> None:
    manifests = {"ref": manifest("ref", weight="fp16"), "cand": manifest("cand", weight="int4")}
    scorers = {"ref": FakeScorer("ref"), "cand": FakeScorer("cand")}
    FakeScorer.loaded = []

    service(tmp_path, scorers, manifests).evaluate("cand", "ref", corpus=CORPUS)

    assert FakeScorer.loaded == ["open:ref", "close:ref", "open:cand", "close:cand"]


def test_the_spill_file_is_removed_and_the_result_saved(tmp_path: Path) -> None:
    manifests = {"ref": manifest("ref", weight="fp16"), "cand": manifest("cand", weight="int4")}
    scorers = {"ref": FakeScorer("ref"), "cand": FakeScorer("cand")}

    service(tmp_path, scorers, manifests).evaluate("cand", "ref", corpus=CORPUS)

    assert list((tmp_path / "cache" / "tmp").glob("eval-*")) == []
    assert len(list((tmp_path / "data" / "evaluations").glob("*.json"))) == 1


def test_different_source_weights_are_refused(tmp_path: Path) -> None:
    manifests = {
        "ref": manifest("ref", weight="fp16"),
        "cand": manifest("cand", weight="int4", revision="b" * 40),
    }
    scorers = {"ref": FakeScorer("ref"), "cand": FakeScorer("cand")}

    with pytest.raises(ConfigurationError) as caught:
        service(tmp_path, scorers, manifests).evaluate("cand", "ref", corpus=CORPUS)
    assert caught.value.diagnostic.code == "eval.different_sources"


def test_a_different_vocabulary_is_refused(tmp_path: Path) -> None:
    manifests = {"ref": manifest("ref", weight="fp16"), "cand": manifest("cand", weight="int4")}
    scorers = {"ref": FakeScorer("ref"), "cand": FakeScorer("cand", offset_ids=1)}

    with pytest.raises(ConfigurationError) as caught:
        service(tmp_path, scorers, manifests).evaluate("cand", "ref", corpus=CORPUS)
    assert caught.value.diagnostic.code == "eval.tokenizer_mismatch"


def test_comparing_an_artifact_with_itself_is_refused(tmp_path: Path) -> None:
    manifests = {"ref": manifest("ref", weight="fp16")}

    with pytest.raises(ConfigurationError) as caught:
        service(tmp_path, {"ref": FakeScorer("ref")}, manifests).evaluate(
            "ref", "ref", corpus=CORPUS
        )
    assert caught.value.diagnostic.code == "eval.same_artifact"


def test_the_shipped_corpus_is_original_and_varied() -> None:
    corpus = load_corpus()

    assert corpus.id == "yonga-eval-v1"
    genres = {passage.genre for passage in corpus.passages}
    assert {"python", "prose-tr", "reasoning"} <= genres
    assert len(corpus.sha256) == 64


def test_a_plain_text_corpus_splits_on_blank_lines(tmp_path: Path) -> None:
    source = tmp_path / "mine.txt"
    source.write_text("First passage.\n\nSecond one,\nover two lines.\n", encoding="utf-8")

    corpus = load_corpus(source)

    assert [p.text for p in corpus.passages] == ["First passage.", "Second one,\nover two lines."]
