"""``yonga search``: one Hub request, identified through the registry, stated honestly.

The rules under test come from ``docs/CLI_SPEC.md`` section 5 -- enrich results
"where architecture metadata is available" and never claim compatibility without
evidence -- and from the finding that architecture is not preparability: an MLX
repository reports ``qwen3_5`` but cannot be converted.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, cast

import pytest
from rich.console import Console

from yonga.adapters.hf import HuggingFaceAdapter
from yonga.application.search import (
    PER_ARCHITECTURE_NOTE,
    SearchService,
    format_hints,
)
from yonga.config.settings import Settings
from yonga.domain.compatibility import DeviceStatus
from yonga.domain.diagnostics import YongaError
from yonga.domain.models import ModelIdentity, ModelMetadata
from yonga.registry import RegistryResolver, load_builtin_registry
from yonga.ui.tables import search_results

# --- fakes -----------------------------------------------------------------------------


@dataclass
class FakeSafetensors:
    total: int


@dataclass
class FakeSearchResult:
    """Shaped like one ``HfApi.list_models`` result."""

    id: str
    config: dict[str, object] | None = None
    gguf: dict[str, object] | None = None
    tags: list[str] = field(default_factory=list)
    pipeline_tag: str | None = None
    library_name: str | None = None
    downloads: int | None = None
    likes: int | None = None
    gated: object = False
    safetensors: FakeSafetensors | None = None
    sha: str | None = None
    last_modified: str | None = None


class FakeSearchApi:
    def __init__(self, results: list[FakeSearchResult], error: BaseException | None = None):
        self.results = results
        self.error = error
        self.calls: list[dict[str, object]] = []

    def list_models(self, **kwargs: object) -> list[FakeSearchResult]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return list(self.results)


class FakeHubSearch:
    def __init__(self, hits: list[ModelMetadata]) -> None:
        self.hits = hits
        self.limits: list[int] = []

    def search(self, query: str, *, limit: int = 20) -> list[ModelMetadata]:
        self.limits.append(limit)
        return list(self.hits)


def meta(repo_id: str, **fields: Any) -> ModelMetadata:
    return ModelMetadata(identity=ModelIdentity(repo_id=repo_id), resolved_revision=None, **fields)


def service(hits: list[ModelMetadata]) -> tuple[SearchService, FakeHubSearch]:
    hub = FakeHubSearch(hits)
    return SearchService(hub, RegistryResolver(load_builtin_registry())), hub


QWEN35_CONFIG: dict[str, object] = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
}

# --- the adapter -------------------------------------------------------------------------


def test_one_request_carries_identity_for_config_and_gguf_repos() -> None:
    api = FakeSearchApi(
        [
            FakeSearchResult("Qwen/Qwen3.5-9B", config=QWEN35_CONFIG, downloads=9_787_242),
            FakeSearchResult("unsloth/Qwen3.5-9B-GGUF", gguf={"architecture": "qwen35"}),
        ]
    )
    adapter = HuggingFaceAdapter(Settings(), api=cast(Any, api))

    hits = adapter.search("qwen3.5 9b", limit=5)

    assert len(api.calls) == 1, "search must be one request, not one per hit"
    assert api.calls[0]["search"] == "qwen3.5 9b" and api.calls[0]["limit"] == 5
    assert set(cast(list[str], api.calls[0]["expand"])) >= {"config", "gguf"}
    assert hits[0].model_type == "qwen3_5" and hits[0].downloads == 9_787_242
    assert hits[1].model_type is None
    assert hits[1].gguf_metadata.get("architecture") == "qwen35"


def test_searching_offline_is_a_typed_error_not_a_network_attempt() -> None:
    api = FakeSearchApi([])
    adapter = HuggingFaceAdapter(Settings(offline=True), api=cast(Any, api))

    with pytest.raises(YongaError):
        adapter.search("anything")
    assert api.calls == []


def test_a_hub_failure_is_translated() -> None:
    adapter = HuggingFaceAdapter(
        Settings(), api=cast(Any, FakeSearchApi([], error=ConnectionError("network down")))
    )
    with pytest.raises(YongaError) as caught:
        adapter.search("anything")
    assert caught.value.diagnostic.code


# --- the service ---------------------------------------------------------------------


def test_a_config_identified_hit_carries_the_registrys_device_knowledge() -> None:
    search, _ = service([meta("Qwen/Qwen3.5-9B", **QWEN35_CONFIG)])

    hit = search.search("q").hits[0]

    assert hit.architecture == "qwen3_5"
    assert hit.architecture_rule_id == "arch-qwen3_5"
    assert hit.identified_from is None
    statuses = {entry.device: entry.status for entry in hit.compatibility}
    assert statuses == {
        "CPU": DeviceStatus.SUPPORTED,
        "GPU": DeviceStatus.SUPPORTED,
        "NPU": DeviceStatus.UNSUPPORTED,
    }


def test_a_gguf_identified_hit_says_where_its_identity_came_from() -> None:
    search, _ = service([meta("unsloth/Qwen3.5-9B-GGUF", gguf_metadata={"architecture": "qwen35"})])

    hit = search.search("q").hits[0]

    assert hit.architecture == "qwen3_5"
    assert hit.identified_from == "GGUF metadata"


def test_an_unidentified_hit_is_unknown_on_every_device_not_omitted() -> None:
    search, _ = service([meta("someone/mystery-model")])

    hit = search.search("q").hits[0]

    assert hit.architecture is None
    assert [entry.device for entry in hit.compatibility] == ["CPU", "GPU", "NPU"]
    assert all(entry.status is DeviceStatus.UNKNOWN for entry in hit.compatibility)


def test_tags_never_identify_an_architecture() -> None:
    """A ``qwen3.5`` tag is uploader text, not evidence (SPEC.md section 8.1)."""
    search, _ = service([meta("someone/looks-like-qwen", tags=["qwen3.5", "qwen", "gguf"])])

    hit = search.search("q").hits[0]

    assert hit.architecture is None
    assert all(entry.status is DeviceStatus.UNKNOWN for entry in hit.compatibility)


def test_format_hints_expose_what_the_exporter_cannot_convert() -> None:
    mlx = meta("lmstudio-community/Qwen3.5-9B-MLX-4bit", tags=["safetensors", "mlx"])
    awq = meta("QuantTrio/Qwen3.5-9B-AWQ", tags=["safetensors", "awq", "conversational"])
    ov = meta("OpenVINO/model", library_name="openvino", tags=["image-text-to-text"])

    assert format_hints(mlx) == ["safetensors", "mlx"]
    assert format_hints(awq) == ["safetensors", "awq"]
    assert format_hints(ov) == ["openvino"]


def test_a_declared_derivative_is_surfaced() -> None:
    search, _ = service(
        [
            meta(
                "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive",
                gguf_metadata={"architecture": "qwen35"},
            )
        ]
    )

    hit = search.search("q").hits[0]

    assert hit.derivative == "nDimensional/Qwen3.5-9B-Uncensored-Safetensors"


def test_the_limit_is_bounded_and_the_report_states_its_limits() -> None:
    search, hub = service([])

    report = search.search("q", limit=10_000)

    assert hub.limits == [100]
    assert PER_ARCHITECTURE_NOTE in report.notes


# --- the renderer --------------------------------------------------------------------


def render(renderable: object) -> str:
    console = Console(file=io.StringIO(), width=100, force_terminal=False)
    console.print(renderable)
    return cast(io.StringIO, console.file).getvalue()


def test_the_table_marks_hinted_identity_and_derivatives() -> None:
    search, _ = service(
        [
            meta("Qwen/Qwen3.5-9B", **QWEN35_CONFIG, downloads=9_787_242),
            meta(
                "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive",
                gguf_metadata={"architecture": "qwen35"},
                tags=["gguf"],
            ),
        ]
    )
    text = render(search_results(search.search("qwen")))
    flat = " ".join(text.split())  # footnotes wrap at the terminal width

    assert "qwen3_5*" in text
    assert "architecture identified from GGUF metadata" in flat
    assert "converts from nDimensional/Qwen3.5-9B-Uncensored-Safetensors" in flat
    assert "9.8M" in text
    assert "yonga inspect MODEL" in flat


def test_a_device_filter_shows_only_that_column() -> None:
    search, _ = service([meta("Qwen/Qwen3.5-9B", **QWEN35_CONFIG)])
    text = render(search_results(search.search("q"), device="npu"))

    header = text.splitlines()[0]
    assert "NPU" in header and "GPU" not in header and "CPU" not in header


def test_no_hits_is_said_plainly() -> None:
    search, _ = service([])
    assert "No models on the Hub match 'nothing'" in render(
        search_results(search.search("nothing"))
    )
