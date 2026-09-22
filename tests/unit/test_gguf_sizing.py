"""How big is a model, as opposed to how big is its repository.

A GGUF repository conventionally publishes one file per quantization of the
*same* model plus, for vision models, a projector side-car. Summing that
inventory answers the wrong question: nobody downloads all of it, and feeding
the sum to the temp-space estimate
(``max(20 GiB, source_weight_bytes * 2.5)``, ``SPEC.md`` section 12) demands
scratch space for a model several times larger than the one being converted.

These tests pin the distinction down on the two real reference repositories
from ``HOST_FACTS.md`` — the five-file
``HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive`` GGUF repository and the
four-shard ``nDimensional/Qwen3.5-9B-Uncensored-Safetensors`` derivative — and
on the degenerate cases around them. Nothing here is Qwen-specific: the rule
under test is "one model, whatever the architecture", driven by the Hub's own
metadata and a documented filename convention.
"""

from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console, RenderableType

from yonga.adapters.filesystem import plan_disk, required_temp_bytes
from yonga.adapters.hf import detect_formats
from yonga.config.settings import Settings, load_settings
from yonga.domain.hardware import FilesystemInfo
from yonga.domain.models import (
    FormatObservation,
    ModelFormat,
    ModelIdentity,
    ModelMetadata,
    RepoFile,
)
from yonga.ui.tables import inspect_report

GIB = 2**30
TIB = 2**40

# --------------------------------------------------------------------------- #
# The real inventories (HOST_FACTS.md, queried against the live Hub)
# --------------------------------------------------------------------------- #

#: ``HauhauCS/…-Aggressive``: four quantizations of one 9B model, plus a
#: multimodal projector. A user runs exactly one of the four.
GGUF_PROJECTOR_NAME = "mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf"

GGUF_FILE_SIZES: dict[str, int] = {
    "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf": 17_920_696_704,
    "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf": 5_627_044_224,
    "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q6_K.gguf": 7_359_259_008,
    "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf": 9_527_501_184,
    GGUF_PROJECTOR_NAME: 921_704_448,
}

#: The unquantized copy — and what the Hub reports as ``totalFileSize``.
GGUF_PRIMARY_BYTES = 17_920_696_704
#: Every GGUF byte the repository publishes: 38.5 GiB across five files.
GGUF_REPOSITORY_BYTES = 41_356_205_568

#: ``nDimensional/…-Safetensors``: four shards that are all one model.
SAFETENSORS_SHARD_SIZES: dict[str, int] = {
    "model.safetensors-00001-of-00004.safetensors": 5_278_251_472,
    "model.safetensors-00002-of-00004.safetensors": 5_358_071_808,
    "model.safetensors-00003-of-00004.safetensors": 5_340_098_264,
    "model.safetensors-00004-of-00004.safetensors": 2_843_299_496,
}
SAFETENSORS_BYTES = 18_819_721_040


def gguf_files() -> list[RepoFile]:
    """The GGUF repository's weight inventory, largest-first order irrelevant."""
    return [RepoFile(path=name, size=size, lfs=True) for name, size in GGUF_FILE_SIZES.items()]


def safetensors_files() -> list[RepoFile]:
    """The safetensors derivative's four shards plus its index document."""
    files = [
        RepoFile(path=name, size=size, lfs=True) for name, size in SAFETENSORS_SHARD_SIZES.items()
    ]
    files.append(RepoFile(path="model.safetensors.index.json", size=78_336))
    return files


def observation_for(observations: list[FormatObservation], fmt: ModelFormat) -> FormatObservation:
    """Return the single observation of *fmt*, failing loudly when absent."""
    matches = [item for item in observations if item.format is fmt]
    assert len(matches) == 1, f"expected exactly one {fmt.value} observation, got {matches}"
    return matches[0]


def metadata_for(observations: list[FormatObservation]) -> ModelMetadata:
    """Wrap format observations in metadata so the model-level API can be read."""
    return ModelMetadata(
        identity=ModelIdentity(repo_id="HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"),
        formats=observations,
    )


def render(renderable: RenderableType, *, width: int = 120) -> str:
    """Render a renderable to plain text the way a non-TTY terminal would."""
    buffer = io.StringIO()
    Console(file=buffer, width=width, force_terminal=False, no_color=True).print(renderable)
    return buffer.getvalue()


def settings_for(tmp_path: Path) -> Settings:
    """Deterministic settings that ignore whatever config the developer has."""
    return load_settings(
        config_file=tmp_path / "absent-config.toml",
        env={},
        overrides={"cache_dir": tmp_path / "cache", "data_dir": tmp_path / "data"},
    )


class RoomyProbe:
    """A ``FilesystemProbe`` reporting one large ext4 filesystem for every path.

    Disk *capacity* is not what these tests are about; they only need a planner
    that runs to completion so the byte figure it was handed can be read back.
    """

    def usage(self, path: Path) -> FilesystemInfo:
        """Report 2 TiB free on ``/`` for *path*."""
        return FilesystemInfo(
            path=path,
            mount_point=Path("/"),
            fstype="ext4",
            total_bytes=4 * TIB,
            free_bytes=2 * TIB,
            available_bytes=2 * TIB,
        )


# --------------------------------------------------------------------------- #
# GGUF: one model out of many published copies
# --------------------------------------------------------------------------- #


def test_gguf_repository_separates_one_model_from_the_whole_inventory() -> None:
    """The Hub's ``totalFileSize`` names the primary model; the sum stays honest."""
    observations = detect_formats(gguf_files(), gguf_total_file_size=GGUF_PRIMARY_BYTES)
    observation = observation_for(observations, ModelFormat.GGUF)

    assert observation.primary_bytes == GGUF_PRIMARY_BYTES
    assert observation.total_bytes == GGUF_REPOSITORY_BYTES
    assert len(observation.files) == 5
    # The two are not accidentally equal: this repository really does publish
    # more than twice the bytes of the model a user ends up running.
    assert observation.total_bytes > 2 * observation.primary_bytes


def test_missing_total_file_size_falls_back_to_the_largest_model_file() -> None:
    """Without the Hub's figure, the largest non-companion GGUF file is the model."""
    observations = detect_formats(gguf_files())
    observation = observation_for(observations, ModelFormat.GGUF)

    assert observation.primary_bytes == GGUF_PRIMARY_BYTES
    # Neither the projector side-car ...
    assert observation.primary_bytes != GGUF_FILE_SIZES[GGUF_PROJECTOR_NAME]
    # ... nor the sum of every published quantization.
    assert observation.primary_bytes != GGUF_REPOSITORY_BYTES
    assert observation.total_bytes == GGUF_REPOSITORY_BYTES


def test_the_projector_side_car_is_never_mistaken_for_the_model() -> None:
    """A repository whose largest file is the projector still picks a real model."""
    files = [
        RepoFile(path="mmproj-demo-BF16.gguf", size=900 * GIB),
        RepoFile(path="demo-Q4_K_M.gguf", size=5 * GIB),
    ]

    observation = observation_for(detect_formats(files), ModelFormat.GGUF)

    assert observation.primary_bytes == 5 * GIB
    assert observation.total_bytes == 905 * GIB
    assert any("projector" in note for note in observation.notes)


def test_a_companion_only_repository_reports_no_usable_model() -> None:
    """Nothing but side-cars: there is no runnable model, so the size is unknown.

    Answering with the largest companion would hand callers the size of a file
    this very function classified as *not* runnable. ``0`` means unknown, and
    the temp estimate has its own floor for exactly this case.
    """
    files = [
        RepoFile(path="mmproj-demo-BF16.gguf", size=900),
        RepoFile(path="mmproj-demo-F16.gguf", size=400),
    ]

    observation = observation_for(detect_formats(files), ModelFormat.GGUF)

    assert observation.primary_bytes == 0
    # The download-side figure is untouched: those bytes really are published.
    assert observation.total_bytes == 1300
    assert any("unknown" in note for note in observation.notes)
    # Nothing downstream mistakes the companions for a model either.
    assert metadata_for([observation]).weight_bytes == 0


# --------------------------------------------------------------------------- #
# Which file names count as a companion
# --------------------------------------------------------------------------- #
#
# The rule is a filename convention (``GGUF_COMPANION_PREFIXES``), not anything
# the Hub's file listing states, so its edges are pinned here rather than left
# to the reader of a one-line ``startswith``.


def test_an_uppercase_projector_name_is_still_a_companion() -> None:
    """The comparison is case-folded: ``MMPROJ-`` is not a loophole."""
    files = [
        RepoFile(path="MMPROJ-demo-BF16.gguf", size=900 * GIB),
        RepoFile(path="demo-Q4_K_M.gguf", size=5 * GIB),
    ]

    observation = observation_for(detect_formats(files), ModelFormat.GGUF)

    assert observation.primary_bytes == 5 * GIB
    assert any("MMPROJ-demo-BF16.gguf" in note for note in observation.notes)


def test_a_projector_in_a_subdirectory_is_still_a_companion() -> None:
    """The base name decides, so a directory prefix cannot defeat the rule."""
    files = [
        RepoFile(path="projectors/mmproj-demo-BF16.gguf", size=900 * GIB),
        RepoFile(path="demo-Q4_K_M.gguf", size=5 * GIB),
    ]

    observation = observation_for(detect_formats(files), ModelFormat.GGUF)

    assert observation.primary_bytes == 5 * GIB
    assert any("mmproj-demo-BF16.gguf" in note for note in observation.notes)


def test_a_marker_later_in_the_name_is_not_a_companion() -> None:
    """Documented decision: only a *prefix* marks a side-car.

    ``llama.cpp`` writes ``mmproj-<model>.gguf``, marker first. A substring test
    would classify a legitimately named model such as ``model-mmproj-ish.gguf``
    as unrunnable and — now that a companion-only repository reports 0 — erase
    its size entirely. Prefix-only is the conservative reading.
    """
    files = [RepoFile(path="model-mmproj-ish.gguf", size=7 * GIB)]

    observation = observation_for(detect_formats(files), ModelFormat.GGUF)

    assert observation.primary_bytes == 7 * GIB
    assert not any("projector" in note for note in observation.notes)


def test_a_repository_without_any_companion_is_unaffected() -> None:
    """No side-car in sight: the largest quantization is the model, as before."""
    files = [
        RepoFile(path="demo-BF16.gguf", size=17 * GIB),
        RepoFile(path="demo-Q4_K_M.gguf", size=5 * GIB),
    ]

    observation = observation_for(detect_formats(files), ModelFormat.GGUF)

    assert observation.primary_bytes == 17 * GIB
    assert observation.total_bytes == 22 * GIB
    assert not any("projector" in note for note in observation.notes)


# --------------------------------------------------------------------------- #
# A sized companion next to an unsized model
# --------------------------------------------------------------------------- #


def test_a_sized_projector_beside_an_unsized_model_reports_unknown() -> None:
    """The one file classified as unrunnable must never become the model's size.

    The Hub reported no ``totalFileSize`` and sized only the side-car. Every
    candidate model is sizeless, so the honest answer is "unknown" — and the
    ``primary_bytes``-defaults-to-``total_bytes`` convenience must not step in
    and promote the projector.
    """
    files = [
        RepoFile(path="demo-Q4_K_M.gguf", size=None),
        RepoFile(path="mmproj-demo-BF16.gguf", size=900),
    ]

    observation = observation_for(detect_formats(files), ModelFormat.GGUF)

    assert observation.primary_bytes == 0
    assert observation.total_bytes == 900
    assert metadata_for([observation]).weight_bytes == 0


def test_a_sized_model_beside_a_projector_reports_the_model() -> None:
    """Mixed inventory: the sized model file wins, never the projector."""
    files = [
        RepoFile(path="demo-Q4_K_M.gguf", size=5 * GIB),
        RepoFile(path="demo-BF16.gguf", size=None),
        RepoFile(path="mmproj-demo-BF16.gguf", size=900 * GIB),
    ]

    observation = observation_for(detect_formats(files), ModelFormat.GGUF)

    assert observation.primary_bytes == 5 * GIB
    assert observation.primary_bytes != 900 * GIB
    assert metadata_for([observation]).weight_bytes == 5 * GIB


def test_an_unknown_primary_flows_through_the_ui_and_the_temp_estimate(
    tmp_path: Path,
) -> None:
    """Unknown stays unknown end to end: a dash on screen, the floor in preflight."""
    metadata = metadata_for(
        detect_formats(
            [
                RepoFile(path="demo-Q4_K_M.gguf", size=None),
                RepoFile(path="mmproj-demo-BF16.gguf", size=900),
            ]
        )
    )

    assert metadata.weight_bytes == 0
    assert "—" in render(inspect_report(metadata))

    settings = settings_for(tmp_path)
    assert required_temp_bytes(metadata.weight_bytes, settings) == settings.temp_min_bytes


def test_a_single_file_gguf_repository_has_nothing_to_distinguish() -> None:
    """One file is both the model and the repository, so the counters agree."""
    observation = observation_for(
        detect_formats([RepoFile(path="demo-Q4_K_M.gguf", size=5 * GIB)]),
        ModelFormat.GGUF,
    )

    assert observation.primary_bytes == observation.total_bytes == 5 * GIB
    assert metadata_for([observation]).weight_bytes == 5 * GIB


def test_a_declared_total_file_size_of_zero_is_not_trusted() -> None:
    """A non-positive Hub figure is no figure at all; fall back to the files."""
    observation = observation_for(
        detect_formats(gguf_files(), gguf_total_file_size=0),
        ModelFormat.GGUF,
    )

    assert observation.primary_bytes == GGUF_PRIMARY_BYTES


# --------------------------------------------------------------------------- #
# Every other format: all the files are the one model
# --------------------------------------------------------------------------- #


def test_safetensors_shards_are_all_required_by_the_one_model() -> None:
    """Four shards of ``nDimensional/…-Safetensors`` are not four models."""
    observations = detect_formats(
        safetensors_files(),
        shard_indexes={"model.safetensors.index.json": sorted(SAFETENSORS_SHARD_SIZES)},
    )
    observation = observation_for(observations, ModelFormat.SAFETENSORS)

    assert observation.complete is True
    assert observation.primary_bytes == observation.total_bytes == SAFETENSORS_BYTES
    assert metadata_for(observations).weight_bytes == SAFETENSORS_BYTES
    assert metadata_for(observations).repo_weight_bytes == SAFETENSORS_BYTES


def test_pytorch_and_openvino_observations_also_size_one_model() -> None:
    """The distinction is GGUF-only; no other format loses bytes to it."""
    observations = detect_formats(
        [
            RepoFile(path="openvino_model.xml", size=10),
            RepoFile(path="openvino_model.bin", size=90),
            RepoFile(path="pytorch_model-00001-of-00002.bin", size=400),
            RepoFile(path="pytorch_model-00002-of-00002.bin", size=600),
        ]
    )

    for fmt, expected in (
        (ModelFormat.OPENVINO_IR, 100),
        (ModelFormat.PYTORCH_BIN, 1000),
    ):
        observation = observation_for(observations, fmt)
        assert observation.primary_bytes == observation.total_bytes == expected


# --------------------------------------------------------------------------- #
# What the rest of the tool reads off these numbers
# --------------------------------------------------------------------------- #


def test_required_temp_bytes_differs_by_which_gguf_figure_it_is_given(
    tmp_path: Path,
) -> None:
    """Feed the temp formula both counters of a GGUF repository and compare.

    SCOPE, honestly stated: this calls ``required_temp_bytes`` directly on a
    ``ModelMetadata`` built here. It does not exercise ``prepare``, and it
    cannot — ``prepare`` only ever plans disk for a ``ResolvedSource`` whose
    metadata passed ``_USABLE_SOURCE_FORMATS`` (OpenVINO IR, safetensors,
    PyTorch), so a GGUF repository either resolves to a safetensors derivative
    or fails with ``source.format_unsupported`` long before ``plan_disk`` is
    reached. No GGUF byte count reaches preflight today.

    What the GGUF ``primary_bytes``/``total_bytes`` split really buys today is
    the ``inspect`` size row (see the rendering tests below). This test is a
    correctness guard for any FUTURE path that does convert GGUF — a
    GGUF-capable backend — and it records the size of the mistake such a path
    would inherit if the two counters were confused. See
    ``test_the_planner_prepare_actually_calls_is_sized_from_one_model`` for the
    live route.
    """
    settings = settings_for(tmp_path)
    metadata = metadata_for(detect_formats(gguf_files(), gguf_total_file_size=GGUF_PRIMARY_BYTES))

    assert metadata.weight_bytes == GGUF_PRIMARY_BYTES
    assert metadata.repo_weight_bytes == GGUF_REPOSITORY_BYTES

    required = required_temp_bytes(metadata.weight_bytes, settings)
    assert required == int(GGUF_PRIMARY_BYTES * settings.temp_size_multiplier)

    # Sizing from the repository total would demand ~96 GiB of scratch space
    # instead of ~42 GiB.
    from_the_repository = required_temp_bytes(metadata.repo_weight_bytes, settings)
    assert from_the_repository > 2 * required


def test_the_planner_prepare_actually_calls_is_sized_from_one_model(tmp_path: Path) -> None:
    """``plan_disk`` on the safetensors source, the format preflight really sees.

    ``prepare`` calls ``plan_disk(source_weight_bytes=source_meta.weight_bytes)``
    for a resolved source, and safetensors is the format that route lands on for
    the reference repositories. Here the primary/total distinction collapses —
    all four shards are the one model — so the planner receives the full
    18.8 GB and reserves working space for it.
    """
    settings = settings_for(tmp_path)
    metadata = metadata_for(
        detect_formats(
            safetensors_files(),
            shard_indexes={"model.safetensors.index.json": sorted(SAFETENSORS_SHARD_SIZES)},
        )
    )
    assert metadata.weight_bytes == metadata.repo_weight_bytes == SAFETENSORS_BYTES

    plan = plan_disk(
        job_temp_dir=tmp_path / "job",
        artifact_dir=tmp_path / "artifacts",
        source_weight_bytes=metadata.weight_bytes,
        settings=settings,
        probe=RoomyProbe(),
    )

    assert plan.temp.required_bytes == required_temp_bytes(SAFETENSORS_BYTES, settings)
    assert plan.estimated_artifact_bytes == SAFETENSORS_BYTES
    assert plan.sufficient


def test_inspect_size_row_names_both_figures_when_they_differ() -> None:
    """One line, both numbers: the download cost and what the listing sums to."""
    metadata = metadata_for(detect_formats(gguf_files(), gguf_total_file_size=GGUF_PRIMARY_BYTES))

    text = render(inspect_report(metadata))

    assert "~16.7 GiB (one model; 38.5 GiB across 5 files in the repository)" in text


def test_inspect_size_row_stays_short_when_there_is_only_one_figure() -> None:
    """A single-model repository gains no parenthetical it does not need."""
    metadata = metadata_for(
        detect_formats(
            safetensors_files(),
            shard_indexes={"model.safetensors.index.json": sorted(SAFETENSORS_SHARD_SIZES)},
        )
    )

    text = render(inspect_report(metadata))

    assert "~17.5 GiB total weights" in text
    assert "one model" not in text


# --------------------------------------------------------------------------- #
# Degenerate inventories
# --------------------------------------------------------------------------- #


def test_unknown_file_sizes_yield_zero_without_crashing() -> None:
    """The Hub omitted every size: report "unknown", never a guessed number."""
    files = [RepoFile(path=name, size=None) for name in GGUF_FILE_SIZES]

    observations = detect_formats(files)
    observation = observation_for(observations, ModelFormat.GGUF)
    metadata = metadata_for(observations)

    assert observation.primary_bytes == 0
    assert observation.total_bytes == 0
    assert metadata.weight_bytes == 0
    assert metadata.repo_weight_bytes == 0
    assert metadata.primary_format is ModelFormat.GGUF
    assert "—" in render(inspect_report(metadata))


def test_a_declared_size_survives_an_inventory_with_no_sizes() -> None:
    """``inspect --no-sizes`` still knows how big one model is, if the Hub said so."""
    files = [RepoFile(path=name, size=None) for name in GGUF_FILE_SIZES]

    observations = detect_formats(files, gguf_total_file_size=GGUF_PRIMARY_BYTES)
    metadata = metadata_for(observations)

    assert metadata.weight_bytes == GGUF_PRIMARY_BYTES
    assert metadata.repo_weight_bytes == 0


def test_an_empty_repository_reports_no_weights() -> None:
    """No weight files at all: zero, and no exception on the way there."""
    metadata = metadata_for(detect_formats([RepoFile(path="README.md", size=2952)]))

    assert metadata.weight_bytes == 0
    assert metadata.repo_weight_bytes == 0
    assert metadata.primary_observation is None
