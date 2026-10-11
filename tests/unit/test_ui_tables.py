"""Tests for the pure table/report renderers."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text

from yonga.domain.artifacts import (
    ArtifactSummary,
    DeviceProbeReport,
    ProbeResult,
    ProbeStatus,
    SubmodelRole,
)
from yonga.domain.benchmarks import (
    AggregateMetrics,
    BenchmarkResult,
    GenerationSettings,
    RunMetrics,
)
from yonga.domain.compatibility import (
    CompatibilityDecision,
    DeviceCompatibility,
    DeviceStatus,
    EvidenceLevel,
    ResolutionExplanation,
    RuleCandidate,
)
from yonga.domain.compile_cache import CacheEntryStatus, CompileCacheEntry
from yonga.domain.diagnostics import (
    FailureCategory,
    HealthCheck,
    HealthSection,
    Severity,
    diagnostic,
)
from yonga.domain.hardware import FilesystemInfo, HardwareInventory
from yonga.domain.models import (
    FormatObservation,
    ModelFormat,
    ModelIdentity,
    ModelMetadata,
    RepoFile,
    ResolvedSource,
    SourceRelationship,
)
from yonga.domain.plans import (
    CONFIRMATION_REMOTE_CODE,
    ConfirmationReason,
    ConversionProfile,
    DiskPlan,
    ExporterRequirements,
    PreparationPlan,
    QuantizationSettings,
    TempPlan,
)
from yonga.domain.publication import DeviceEvidence, LicenseAssessment, LicenseClass, PublishPlan
from yonga.ui.tables import (
    SourceOption,
    artifact_list,
    benchmark_summary,
    compile_cache_entries,
    doctor_report,
    inspect_report,
    preparation_plan,
    probe_report,
    publication_plan,
    registry_explanation,
    registry_profiles,
)

NOW = datetime(2026, 9, 22, 10, 30, tzinfo=UTC)
SHA = "0123456789abcdef0123456789abcdef01234567"


def render(renderable, *, width: int = 120) -> str:
    """Render a renderable to plain text the way a non-TTY terminal would."""
    buffer = io.StringIO()
    Console(file=buffer, width=width, force_terminal=False, no_color=True).print(renderable)
    return buffer.getvalue()


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
def health_sections() -> list[HealthSection]:
    """The five doctor sections from CLI_SPEC section 3."""
    return [
        HealthSection(
            id="host",
            title="Host",
            checks=[
                HealthCheck(id="os", label="OS", status=Severity.OK, value="Ubuntu 26.04"),
                HealthCheck(id="python", label="Python", status=Severity.OK, value="3.14.4"),
                HealthCheck(id="ram", label="RAM", status=Severity.OK, value="32 GiB"),
            ],
        ),
        HealthSection(
            id="openvino",
            title="OpenVINO",
            checks=[
                HealthCheck(id="rt", label="Runtime", status=Severity.OK, value="2026.4.0"),
                HealthCheck(
                    id="abi",
                    label="ABI family",
                    status=Severity.WARNING,
                    value="mismatched",
                    detail="genai 2025.3 against runtime 2026.4",
                    remediation=["Reinstall openvino-genai to match the runtime family."],
                ),
            ],
        ),
        HealthSection(
            id="devices",
            title="Devices",
            checks=[
                HealthCheck(id="cpu", label="CPU", status=Severity.OK, value="Intel Core Ultra 7"),
                HealthCheck(id="gpu", label="GPU", status=Severity.OK, value="Intel Arc Graphics"),
                HealthCheck(id="npu", label="NPU", status=Severity.OK, value="Intel NPU"),
            ],
        ),
        HealthSection(
            id="npu",
            title="NPU",
            checks=[
                HealthCheck(
                    id="node",
                    label="/dev/accel/accel0",
                    status=Severity.ERROR,
                    value="not writable",
                    remediation=["Add your user to the 'render' group, then log in again."],
                )
            ],
        ),
        HealthSection(
            id="storage",
            title="Storage",
            checks=[
                HealthCheck(
                    id="tmp", label="system /tmp", status=Severity.INFO, value="16 GiB tmpfs"
                ),
                HealthCheck(
                    id="managed", label="managed temp", status=Severity.OK, value="disk-backed"
                ),
            ],
        ),
    ]


def metadata() -> ModelMetadata:
    """Metadata for a gated GGUF VLM repository."""
    return ModelMetadata(
        identity=ModelIdentity(repo_id="Qwen/Qwen3.5-9B"),
        resolved_revision=SHA,
        exists=True,
        gated=True,
        license="apache-2.0",
        pipeline_tag="image-text-to-text",
        architectures=["Qwen3VLForConditionalGeneration"],
        model_type="qwen3_5",
        files=[
            RepoFile(path="config.json", size=1024),
            RepoFile(path="model.gguf", size=20 * 1024**3, lfs=True),
        ],
        formats=[
            FormatObservation(
                format=ModelFormat.GGUF, files=["model.gguf"], total_bytes=20 * 1024**3
            )
        ],
        requires_remote_code=True,
        remote_code_files=["modeling_qwen.py"],
    )


def hint_identified_metadata() -> ModelMetadata:
    """Metadata for a GGUF-only repository that declares no identity of its own.

    No ``config.json``, so no ``model_type`` and no architecture class: whatever
    the registry knows about this model, it did not learn it here.
    """
    return ModelMetadata(
        identity=ModelIdentity(repo_id="HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"),
        resolved_revision=SHA,
        exists=True,
        license="apache-2.0",
        files=[RepoFile(path="model.gguf", size=16 * 1024**3, lfs=True)],
        formats=[
            FormatObservation(
                format=ModelFormat.GGUF, files=["model.gguf"], total_bytes=16 * 1024**3
            )
        ],
        gguf_metadata={"architecture": "qwen35"},
    )


def compatibility() -> CompatibilityDecision:
    """A decision that rules the NPU out and picks the GPU."""
    return CompatibilityDecision(
        requested_device="AUTO",
        selected_device="GPU",
        status=DeviceStatus.SUPPORTED,
        evidence_level=EvidenceLevel.TESTED,
        matched_rule_ids=["arch.qwen3_5"],
        reasons=["NPU compiler cannot build this graph"],
        fallback_candidates=["CPU"],
        per_device=[
            DeviceCompatibility(
                device="NPU",
                status=DeviceStatus.UNSUPPORTED,
                evidence=EvidenceLevel.TESTED,
                reason="current NPU compiler rule",
                rule_id="issue.npu_unsupported_graph",
            ),
            DeviceCompatibility(
                device="GPU", status=DeviceStatus.SUPPORTED, evidence=EvidenceLevel.TESTED
            ),
            DeviceCompatibility(
                device="CPU", status=DeviceStatus.SUPPORTED, evidence=EvidenceLevel.INFERRED
            ),
        ],
    )


def plan(**overrides) -> PreparationPlan:
    """A full preparation plan for the Qwen3.5 fixture."""
    filesystem = FilesystemInfo(
        path=Path("/home/user/.cache/yonga/tmp"),
        fstype="ext4",
        total_bytes=800 * 1024**3,
        free_bytes=796 * 1024**3,
        available_bytes=750 * 1024**3,
    )
    system_temp = FilesystemInfo(
        path=Path("/tmp"),
        fstype="tmpfs",
        total_bytes=16 * 1024**3,
        free_bytes=15 * 1024**3,
        available_bytes=15 * 1024**3,
    )
    payload = {
        "job_id": "job-1234",
        "created_at": NOW,
        "requested": ModelIdentity(repo_id="Qwen/Qwen3.5-9B"),
        "requested_revision": "main",
        "resolved_source": ResolvedSource(
            requested=ModelIdentity(repo_id="Qwen/Qwen3.5-9B"),
            source=ModelIdentity(repo_id="nDimensional/Qwen3.5-9B-ov", revision=SHA),
            relationship=SourceRelationship.CONVERTED,
            format=ModelFormat.SAFETENSORS,
            requires_confirmation=True,
            reasons=["third-party derivative"],
        ),
        "architecture": "qwen3_5",
        "task": "image-text-to-text",
        "hardware": HardwareInventory(),
        "compatibility": compatibility(),
        "target_device": "GPU",
        "profile": ConversionProfile(
            id="gpu-int4-asym-g128",
            device="GPU",
            summary="INT4 asymmetric, group size 128",
            quantization=QuantizationSettings(
                weight_format="int4", symmetric=False, group_size=128, ratio=1.0
            ),
        ),
        "exporter": ExporterRequirements(openvino=">=2026.2"),
        "disk": DiskPlan(
            temp=TempPlan(
                job_temp_dir=Path("/home/user/.cache/yonga/tmp/job-1234"),
                temp_filesystem=filesystem,
                system_temp=system_temp,
                required_bytes=40 * 1024**3,
                safety_margin_bytes=2 * 1024**3,
                system_temp_would_fail=True,
            ),
            artifact_filesystem=filesystem,
            expected_download_bytes=18 * 1024**3,
            estimated_artifact_bytes=6 * 1024**3,
        ),
        "artifact_dir": Path("/home/user/.local/share/yonga/artifacts/qwen35-gpu"),
        "staging_dir": Path("/home/user/.local/share/yonga/artifacts/.staging/job-1234"),
        "trust_remote_code_required": True,
        "trust_remote_code_enabled": False,
        "expected_repairs": ["tokenizer_string_precision"],
        "known_limitations": ["NPU cannot compile this graph with the installed compiler."],
        "confirmations": [
            ConfirmationReason(
                code=CONFIRMATION_REMOTE_CODE,
                message="This model executes code from its repository.",
            ),
            ConfirmationReason(code="large_download", message="18 GiB will be downloaded."),
        ],
    }
    payload.update(overrides)
    return PreparationPlan.model_validate(payload)


def benchmark() -> BenchmarkResult:
    """A two-prompt benchmark result with full metrics."""
    return BenchmarkResult(
        benchmark_id="bench-1",
        created_at=NOW,
        suite="quick",
        artifact_id="qwen35-gpu",
        artifact_path=Path("/artifacts/qwen35-gpu"),
        repo_id="Qwen/Qwen3.5-9B",
        profile="gpu-int4-asym-g128",
        device="GPU",
        settings=GenerationSettings(max_new_tokens=256),
        settings_fingerprint="fp",
        warmups=1,
        runs=3,
        compile_seconds=12.5,
        run_metrics=[
            RunMetrics(
                prompt_id="short",
                prompt_sha256="a" * 64,
                run_index=0,
                warmup=False,
                input_tokens=42,
                output_tokens=256,
                ttft_ms=180.0,
                throughput_tps=24.6,
                tpot_ms=40.6,
                wall_seconds=10.8,
            )
        ],
        aggregates=[
            AggregateMetrics(
                prompt_id="short",
                runs=3,
                ttft_ms_mean=180.0,
                throughput_tps_mean=24.6,
                tpot_ms_mean=40.6,
                output_tokens_mean=256.0,
                wall_seconds_mean=10.8,
            ),
            AggregateMetrics(prompt_id="medium", runs=3),
        ],
    )


# ----------------------------------------------------------------------
# doctor
# ----------------------------------------------------------------------
def test_doctor_report_shows_every_section_and_check():
    text = render(doctor_report(health_sections()))
    for title in ("Host", "OpenVINO", "Devices", "NPU", "Storage"):
        assert title in text
    assert "Ubuntu 26.04" in text
    assert "Intel Arc Graphics" in text
    assert "16 GiB tmpfs" in text


def test_doctor_report_puts_remediation_under_warnings():
    text = render(doctor_report(health_sections()))
    assert "Reinstall openvino-genai to match the runtime family." in text
    assert "Add your user to the 'render' group, then log in again." in text
    assert text.index("ABI family") < text.index("Reinstall openvino-genai")


def test_doctor_report_uses_status_icons():
    text = render(doctor_report(health_sections()))
    assert "✓" in text
    assert "!" in text
    assert "✗" in text


def test_doctor_report_handles_empty_input():
    assert "No health checks" in render(doctor_report([]))
    assert "no checks" in render(doctor_report([HealthSection(id="x", title="Empty")]))


# ----------------------------------------------------------------------
# inspect
# ----------------------------------------------------------------------
def test_inspect_report_matches_the_spec_layout():
    text = render(
        inspect_report(
            metadata(),
            compatibility(),
            [
                SourceOption(label="Direct OpenVINO source", available=False),
                SourceOption(label="Safetensors source", available=False),
                SourceOption(
                    label="Known derivative",
                    available=True,
                    repo_id="nDimensional/Qwen3.5-9B-ov",
                    requires_confirmation=True,
                ),
            ],
        )
    )
    assert "Qwen/Qwen3.5-9B" in text
    assert "apache-2.0" in text
    assert "image-text-to-text" in text
    assert "gguf" in text
    assert "~20.0 GiB total weights" in text
    assert "0123456789ab" in text
    assert "Compatibility on this host" in text
    assert "unsupported" in text
    assert "Preparation" in text
    assert "approval required" in text
    assert "trust-remote-code" in text


def test_inspect_report_abbreviates_a_sha_but_not_a_branch():
    meta = metadata().model_copy(update={"resolved_revision": "main"})
    assert "main" in render(inspect_report(meta))
    assert SHA not in render(inspect_report(metadata()))


def test_inspect_report_lists_files_only_when_asked():
    assert "model.gguf" not in render(inspect_report(metadata()))
    assert "model.gguf" in render(inspect_report(metadata(), show_files=True))


def test_inspect_report_survives_empty_metadata():
    bare = ModelMetadata(identity=ModelIdentity(repo_id="a/b"))
    text = render(inspect_report(bare, CompatibilityDecision(), []))
    assert "a/b" in text
    assert "No registry claim" in text


def test_inspect_report_discloses_an_architecture_identified_from_a_hint():
    """Default output, no ``--verbose``: the device claims name their weakest link.

    Every row of the compatibility block descends from this identification, so a
    run that had to read the architecture out of a GGUF header says so beside
    the claims instead of burying it in a verbose-only note.
    """
    text = render(
        inspect_report(
            hint_identified_metadata(),
            compatibility(),
            identified_from="GGUF metadata",
        )
    )

    disclosure = "Architecture identified from GGUF metadata; this repository has no config.json."
    assert disclosure in text
    assert text.index("Compatibility on this host") < text.index(disclosure) < text.index("NPU")


def test_inspect_report_says_where_a_hint_disagreed_with_config_json():
    """When the repository does declare an identity, the caveat says that instead."""
    text = render(inspect_report(metadata(), compatibility(), identified_from="GGUF metadata"))

    assert (
        "Architecture identified from GGUF metadata; the repository's config.json declares a "
        "different identity." in text
    )


def test_inspect_report_stays_silent_when_config_json_declared_the_identity():
    """No caveat is invented for the ordinary case."""
    assert "identified from" not in render(inspect_report(metadata(), compatibility()))


# ----------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------
def test_preparation_plan_shows_every_decision_before_expensive_work():
    text = render(preparation_plan(plan()))
    assert "Preparation plan" in text
    assert "Qwen/Qwen3.5-9B" in text
    assert "nDimensional/Qwen3.5-9B-ov" in text
    assert "converted" in text
    assert "qwen3_5" in text
    assert "GPU" in text
    assert "gpu-int4-asym-g128" in text
    assert "int4" in text
    assert "group size 128" in text
    assert "18 GiB" in text
    assert "Estimated temp requirement" in text
    assert "42 GiB" in text
    assert "artifacts/qwen35-gpu" in text
    assert "tokenizer_string_precision" in text
    assert "NPU cannot compile this graph" in text
    assert "job-1234" in text


def test_preparation_plan_flags_remote_code_and_security_confirmations():
    text = render(preparation_plan(plan()))
    assert "NOT enabled" in text
    assert "This model executes code from its repository." in text
    assert "never auto-approved by --yes" in text
    assert "18 GiB will be downloaded." in text


def test_preparation_plan_warns_that_system_tmp_is_too_small():
    text = render(preparation_plan(plan()))
    assert "System /tmp is too small" in text


def test_preparation_plan_without_optional_sections():
    minimal = plan(
        expected_repairs=[],
        known_limitations=[],
        confirmations=[],
        trust_remote_code_required=False,
    )
    text = render(preparation_plan(minimal))
    assert "Confirmation required" not in text
    assert "Expected repairs" not in text
    assert "not required" in text


# ----------------------------------------------------------------------
# list
# ----------------------------------------------------------------------
ARTIFACT_ID = "7bd72baf64957953"
PROFILE = "cpu-int4-asym-g128"


def list_summaries() -> list[ArtifactSummary]:
    """One ready artifact and one incomplete one, with real-world field widths."""
    return [
        ArtifactSummary(
            artifact_id=ARTIFACT_ID,
            path=Path("/artifacts/qwen3"),
            repo_id="Qwen/Qwen3-0.6B",
            resolved_revision=SHA,
            profile=PROFILE,
            recommended_device="CPU",
            size_bytes=6 * 1024**3,
            created_at=NOW,
            ready=True,
            last_benchmark=NOW,
        ),
        ArtifactSummary(
            artifact_id="broken",
            path=Path("/artifacts/broken"),
            repo_id="a/b",
            created_at=NOW,
        ),
    ]


def widest_line(text: str) -> int:
    """Width of the longest rendered line, as the terminal would count it."""
    return max(Text(line).cell_len for line in text.splitlines())


def test_artifact_list_renders_rows():
    text = render(artifact_list(list_summaries()))
    assert ARTIFACT_ID in text
    assert "0123456" in text
    assert "6.0 GiB" in text
    assert "✓ yes" in text
    assert "! no" in text


def test_artifact_list_handles_empty_input():
    assert "No prepared artifacts" in render(artifact_list([]))


@pytest.mark.parametrize("width", [80, 100, 120])
def test_artifact_list_keeps_the_spec_columns_legible(width: int):
    """``docs/CLI_SPEC.md`` section 7: id, model, revision, profile, device, size, benchmark."""
    text = render(artifact_list(list_summaries()), width=width)

    assert widest_line(text) <= width
    # Nothing that identifies the artifact is abbreviated away.
    assert ARTIFACT_ID in text
    assert PROFILE in text
    assert "Qwen3-0.6B" in text
    assert "0123456" in text
    assert "CPU" in text
    assert "6.0 GiB" in text
    # Readiness is never lost: a column, or a note naming the unready artifact.
    assert "! no" in text or "Not ready: broken" in text
    if width >= 100:
        assert "2026-09-22" in text
        assert "…" not in text


def test_artifact_list_drops_the_timestamps_before_the_model():
    """A narrow terminal gives up when an artifact was made, not what it was made from."""
    summaries = list_summaries()

    narrow = render(artifact_list(summaries), width=80)
    wide = render(artifact_list(summaries), width=140)

    assert "Model" in narrow
    assert "Rev " in narrow
    assert "Created" not in narrow
    assert "Benchmark" not in narrow
    assert "Model" in wide
    assert "Qwen/Qwen3-0.6B" in wide
    assert "Created" in wide
    # The widest layout keeps the time of day; the narrower ones keep the date.
    assert NOW.astimezone().strftime("%Y-%m-%d %H:%M") in wide


def test_artifact_list_shows_the_real_row_whole_on_an_80_column_terminal():
    """The row from a real 'yonga list' fits 80 columns whole; the revision waits for 100."""
    summary = ArtifactSummary(
        artifact_id="780ff69f5ac67fb5",
        path=Path("/artifacts/qwen3"),
        repo_id="Qwen/Qwen3-0.6B",
        resolved_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        profile="cpu-int4-asym-g128-r08",
        recommended_device="CPU",
        size_bytes=430 * 1024**2,
        created_at=NOW,
        ready=True,
    )

    text = render(artifact_list([summary]), width=80)
    wide = render(artifact_list([summary]), width=100)

    assert widest_line(text) <= 80
    row = next(line for line in text.splitlines() if line.startswith("780ff69f5ac67fb5"))
    for value in ("Qwen/Qwen3-0.6B", "cpu-int4-asym-g128-r08", "CPU", "430 MiB", "✓ yes"):
        assert value in row
    assert "…" not in text
    # Every artifact is ready, so there is nothing for a note to say.
    assert "Not ready" not in text
    assert "c1899de" in wide


#: The row the E2E retest squeezed: a 35-character repository id.
SMOLLM2 = ArtifactSummary(
    artifact_id="02270ff455b13173",
    path=Path("/artifacts/smollm2"),
    repo_id="HuggingFaceTB/SmolLM2-360M-Instruct",
    resolved_revision="a10cc15e2b3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b",
    profile="cpu-int8-asym-ref",
    recommended_device="CPU",
    size_bytes=355 * 1024**2,
    created_at=NOW,
    ready=True,
)


def cells(line: str) -> list[str]:
    """Split a rendered table line into its cells (they are two spaces apart)."""
    return [cell.strip() for cell in line.split("  ") if cell.strip()]


@pytest.mark.parametrize("width", [60, 80, 100, 120])
@pytest.mark.parametrize("with_qwen", [False, True])
def test_artifact_list_gives_up_columns_instead_of_squeezing_them(width: int, with_qwen: bool):
    """Only the model may ever be shortened; every other cell is whole or gone."""
    qwen = SMOLLM2.model_copy(
        update={
            "artifact_id": "780ff69f5ac67fb5",
            "repo_id": "Qwen/Qwen3-0.6B",
            "profile": "cpu-int4-asym-g128-r08",
            "recommended_device": "GPU",
            "ready": False,
        }
    )
    summaries = [SMOLLM2, qwen] if with_qwen else [SMOLLM2]

    text = render(artifact_list(summaries), width=width)

    assert widest_line(text) <= width
    lines = text.splitlines()
    header = cells(lines[0])
    for line in lines[2:]:
        if line.startswith("! Not ready"):
            continue
        row = dict(zip(header, cells(line), strict=True))
        assert row["Artifact"] in {"02270ff455b13173", "780ff69f5ac67fb5"}
        assert row["Profile"] in {"cpu-int8-asym-ref", "cpu-int4-asym-g128-r08"}
        assert row["Dev"] in {"CPU", "GPU"}
        for key in ("Rev", "Size", "Ready"):
            assert "…" not in row.get(key, "")
        if "Model" in row and "…" in row["Model"]:
            assert len(row["Model"]) >= 12
    if width >= 80:
        assert {"Model", "Size", "Ready"} <= set(header)
    if width >= 100:
        assert "Rev" in header
        assert "SmolLM2-360M-Instruct" in text


def test_artifact_list_elides_the_organisation_before_the_model_name():
    """'…SmolLM2-360M-Instruct' beats 'SmolLM2-360M-Instruc…': the name is the point."""
    text = render(artifact_list([SMOLLM2]), width=80)

    assert "SmolLM2-360M-Instruct" in text
    assert "355 MiB" in text
    assert "✓ yes" in text


def test_artifact_list_folds_a_profile_rather_than_cut_the_id_when_very_narrow():
    """Below the narrowest layout the id still reads whole; the profile wraps."""
    text = render(artifact_list([SMOLLM2]), width=34)

    assert "02270ff455b13173" in text
    assert "…" not in text


def test_artifact_list_tells_two_models_with_one_profile_apart_when_narrow():
    """Long repository ids lose their namespace and tail, never the part that differs."""
    summaries = [
        ArtifactSummary(
            artifact_id=f"{index}" * 16,
            path=Path(f"/artifacts/{index}"),
            repo_id=f"meta-llama/Llama-3.2-{size}-Instruct",
            resolved_revision=SHA,
            profile="cpu-int4-asym-g128-r08",
            recommended_device="CPU",
            size_bytes=2 * 1024**3,
            created_at=NOW,
            ready=True,
        )
        for index, size in enumerate(("1B", "3B"), start=1)
    ]

    text = render(artifact_list(summaries), width=80)

    assert widest_line(text) <= 80
    assert "Llama-3.2-1B" in text
    assert "Llama-3.2-3B" in text
    assert "cpu-int4-asym-g128-r08" in text


def test_artifact_list_names_unready_artifacts_when_the_ready_column_goes():
    """Dropping ``Ready`` for width must not make a broken artifact look usable."""
    summaries = list_summaries()
    summaries[0] = summaries[0].model_copy(update={"profile": "cpu-int4-asym-g128-r08"})
    summaries[1] = summaries[1].model_copy(update={"profile": "npu-int4-sym-g128-cw"})

    text = render(artifact_list(summaries), width=72)

    assert widest_line(text) <= 72
    assert "Ready" not in text
    assert "Not ready: broken" in text


def test_artifact_list_elides_only_an_over_long_repository_id():
    """One 50-character repo id must not push the columns beside them out of view."""
    summaries = list_summaries()
    summaries[0] = summaries[0].model_copy(
        update={"repo_id": "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"}
    )

    text = render(artifact_list(summaries), width=120)

    assert widest_line(text) <= 120
    assert ARTIFACT_ID in text
    assert PROFILE in text
    assert "Qwen3.5-9B-Uncensored" in text
    # The short id beside it is not touched.
    assert " a/b " in text


# ----------------------------------------------------------------------
# registry profiles
# ----------------------------------------------------------------------
def profile_rows() -> list[dict[str, object]]:
    """The 'registry profiles' rows of the built-in CPU profiles, as the CLI builds them."""
    return [
        {
            "id": "cpu-int4-asym-g128-r08",
            "device": "CPU",
            "weights": "int4 asym g128 ratio 0.8",
            "band": "below 5B",
            "priority": 110,
            "status": "active",
            "summary": "CPU INT4 asymmetric group-128 with 20% of layers at INT8; "
            "the CPU default below 5B parameters",
        },
        {
            "id": "cpu-int8-asym-ref",
            "device": "CPU",
            "weights": "int8 asym",
            "band": None,
            "priority": 10,
            "status": "experimental",
            "summary": "INT8 asymmetric per-channel reference for quality evaluation",
        },
    ]


@pytest.mark.parametrize("width", [60, 80, 100, 200])
def test_registry_profiles_never_wraps_or_cuts_a_profile_id(width: int) -> None:
    """Each id sits whole on one line; the summary is the column that wraps."""
    text = render(registry_profiles(profile_rows()), width=width)

    assert widest_line(text) <= width
    lines = text.splitlines()
    for profile in ("cpu-int4-asym-g128-r08", "cpu-int8-asym-ref"):
        assert any(line.startswith(f"{profile} ") for line in lines), text
    assert "…" not in text
    if width >= 80:
        assert "experimental" in text
    if width >= 100:
        assert "int4 asym g128 ratio 0.8" in text
    if width >= 200:
        assert "the CPU default below 5B parameters" in text


# ----------------------------------------------------------------------
# probe
# ----------------------------------------------------------------------
def test_probe_report_shows_per_submodel_results():
    reports = [
        DeviceProbeReport(
            device="NPU",
            status=ProbeStatus.FAILED,
            submodels=[
                ProbeResult(
                    device="NPU",
                    role=SubmodelRole.LANGUAGE,
                    status=ProbeStatus.FAILED,
                    duration_seconds=3.2,
                    message="compiler rejected the graph",
                )
            ],
            diagnostic=diagnostic(
                "runtime.compile_failed",
                FailureCategory.COMPILER_UNSUPPORTED_GRAPH,
                "NPU compile failed",
                "The current NPU compiler cannot build this graph.",
            ),
        ),
        DeviceProbeReport(
            device="GPU",
            status=ProbeStatus.OK,
            submodels=[
                ProbeResult(device="GPU", role=SubmodelRole.LANGUAGE, status=ProbeStatus.OK)
            ],
            smoke_test=ProbeResult(
                device="GPU", status=ProbeStatus.OK, duration_seconds=1.5, message="4 tokens"
            ),
        ),
    ]
    text = render(probe_report(reports))
    assert "NPU" in text
    assert "language" in text
    assert "compiler rejected the graph" in text
    assert "smoke test" in text
    assert "NPU compile failed" in text


def test_probe_report_shows_an_inconclusive_device_as_neither_ok_nor_failed():
    reports = [
        DeviceProbeReport(
            device="NPU",
            status=ProbeStatus.INCONCLUSIVE,
            submodels=[
                ProbeResult(
                    device="NPU",
                    role=SubmodelRole.LANGUAGE,
                    status=ProbeStatus.DEFERRED,
                    message="A raw compile on NPU is not conclusive for this graph",
                )
            ],
            smoke_test=ProbeResult(
                device="NPU",
                status=ProbeStatus.INCONCLUSIVE,
                duration_seconds=30.0,
                message="the GenAI pipeline build did not finish within 30 s and was stopped",
            ),
            diagnostic=diagnostic(
                "probe.inconclusive.timeout",
                FailureCategory.UNKNOWN_RUNTIME_ERROR,
                "NPU is inconclusive: the pipeline build hit --timeout",
                "This is not a compiler verdict.",
                severity=Severity.WARNING,
            ),
        )
    ]
    text = render(probe_report(reports))
    assert "inconclusive" in text
    assert "did not finish within 30 s" in text
    assert "failed" not in text
    assert " ok" not in text


def test_every_probe_status_has_a_table_severity():
    for status in ProbeStatus:
        report = DeviceProbeReport(
            device="CPU",
            status=status,
            submodels=[ProbeResult(device="CPU", role=SubmodelRole.LANGUAGE, status=status)],
        )
        assert status.value in render(probe_report([report]))


def test_publication_plan_shows_an_inconclusive_device_as_neither_works_nor_failed():
    plan = PublishPlan(
        artifact_id="acme-model-npu-int4",
        artifact_dir=Path("/artifacts/acme-model-npu-int4"),
        repo_id="me/acme-model-npu-int4",
        private=True,
        repo_exists=False,
        licenses=[
            LicenseAssessment(
                repo_id="acme/model",
                role="source",
                license="apache-2.0",
                classification=LicenseClass.PERMISSIVE,
            )
        ],
        devices=[
            DeviceEvidence(device="CPU", status="ok"),
            DeviceEvidence(
                device="NPU", status="inconclusive", detail="probe.inconclusive.timeout"
            ),
            DeviceEvidence(device="GPU", status="not_tested"),
        ],
        files=[],
    )

    lines = render(publication_plan(plan)).splitlines()

    npu = next(line for line in lines if "Device NPU" in line)
    assert "inconclusive" in npu
    assert "works" not in npu and "failed" not in npu
    assert "works" in next(line for line in lines if "Device CPU" in line)
    assert "not tested" in next(line for line in lines if "Device GPU" in line)


def test_probe_report_handles_empty_inputs():
    assert "No device probes" in render(probe_report([]))
    assert "nothing probed" in render(probe_report([DeviceProbeReport(device="CPU")]))


# ----------------------------------------------------------------------
# bench
# ----------------------------------------------------------------------
def test_benchmark_summary_uses_the_spec_columns():
    text = render(benchmark_summary(benchmark()))
    for column in ("Prompt", "In tok", "Out tok", "TTFT", "tok/s", "TPOT", "Wall"):
        assert column in text
    assert "short" in text
    assert "42" in text
    assert "256" in text
    assert "180.0 ms" in text
    assert "24.6" in text
    assert "40.6 ms" in text
    assert "10.8 s" in text


def test_benchmark_summary_degrades_when_a_prompt_has_no_metrics():
    text = render(benchmark_summary(benchmark()))
    assert "medium" in text
    assert "—" in text


def test_benchmark_summary_handles_no_aggregates():
    result = benchmark().model_copy(update={"aggregates": [], "errors": ["pipeline crashed"]})
    text = render(benchmark_summary(result))
    assert "no measured runs" in text
    assert "pipeline crashed" in text


# ----------------------------------------------------------------------
# registry explain
# ----------------------------------------------------------------------
@pytest.fixture
def explanation():
    """A RegistryExplanation built from the real registry types."""
    resolver = pytest.importorskip("yonga.registry.resolver")
    return resolver.RegistryExplanation(
        repo_id="Qwen/Qwen3.5-9B",
        device="GPU",
        host_tags=["intel-lunarlake", "npu"],
        versions={"openvino": "2026.4.0"},
        architecture_rule_id="arch.qwen3_5",
        architecture=ResolutionExplanation(
            subject="architecture",
            winner_id="arch.qwen3_5",
            candidates=[
                RuleCandidate(
                    rule_id="arch.qwen3_5",
                    kind="architecture",
                    specificity=60,
                    priority=100,
                    matched=True,
                    reasons=["model_type matched"],
                ),
                RuleCandidate(
                    rule_id="arch.generic_causal_lm",
                    kind="architecture",
                    specificity=20,
                    priority=100,
                    matched=False,
                    reasons=["architecture did not match", "openvino constraint unsatisfied"],
                ),
            ],
            notes=["host tags: intel-lunarlake"],
        ),
        model=ResolutionExplanation(subject="model rule"),
        profile_id="gpu-int4-asym-g128",
        profile=ResolutionExplanation(
            subject="profile",
            winner_id="gpu-int4-asym-g128",
            candidates=[
                RuleCandidate(
                    rule_id="gpu-int4-asym-g128",
                    kind="profile",
                    specificity=80,
                    priority=50,
                    matched=True,
                )
            ],
        ),
        chat=ResolutionExplanation(subject="chat profile"),
        derivative_resolution=ResolutionExplanation(subject="derivative sources"),
        device_compatibility=compatibility().per_device,
        recommended_device="GPU",
        notes=["NPU excluded by issue.npu_unsupported_graph"],
    )


def test_registry_explanation_lists_candidates_winner_and_rejections(explanation):
    text = render(registry_explanation(explanation), width=140)
    assert "Qwen/Qwen3.5-9B" in text
    assert "arch.qwen3_5" in text
    assert "winner" in text
    assert "arch.generic_causal_lm" in text
    assert "rejected" in text
    assert "openvino constraint unsatisfied" in text
    assert "gpu-int4-asym-g128" in text
    assert "NPU excluded by issue.npu_unsupported_graph" in text


def test_registry_explanation_handles_empty_resolutions(explanation):
    empty = explanation.model_copy(
        update={
            "architecture": ResolutionExplanation(subject="architecture"),
            "profile": None,
            "derivatives": [],
            "device_compatibility": [],
            "notes": [],
        }
    )
    text = render(registry_explanation(empty), width=140)
    assert "no candidate rules" in text


# ----------------------------------------------------------------------
# compile cache
# ----------------------------------------------------------------------
def cache_entries() -> list[CompileCacheEntry]:
    """One entry per kind of owner, the way ``yonga cache list`` sees them."""
    root = Path("/home/user/.cache/yonga/compile")
    used = datetime(2026, 10, 8, 15, 29, tzinfo=UTC)
    return [
        CompileCacheEntry(
            key="67aa6f019e912a22",
            path=root / "67aa6f019e912a22",
            size_bytes=7823 * 2**20,
            last_used_at=used,
            device="NPU",
            openvino_version="2026.4.0-22959-99c81491cc3-releases/2026/4",
            artifact_id="201e471c1669050f",
            repo_id="nDimensional/Qwen3.5-9B-Uncensored-Safetensors",
            owner_in_store=True,
            owner_exists=True,
            status=CacheEntryStatus.IN_USE,
        ),
        CompileCacheEntry(
            key="b3da3fcc7841fcac",
            path=root / "b3da3fcc7841fcac",
            size_bytes=5 * 2**30,
            last_used_at=used,
            device="GPU",
            openvino_version="2026.3.0",
            artifact_id="bbbbbbbbbbbbbbbb",
            status=CacheEntryStatus.STALE_RUNTIME,
            owner_exists=True,
            owner_in_store=False,
        ),
        CompileCacheEntry(
            key="0123456789abcdef",
            path=root / "0123456789abcdef",
            size_bytes=2048,
            last_used_at=used,
        ),
    ]


def test_compile_cache_entries_never_cut_a_key_or_a_status_short() -> None:
    entries = cache_entries()
    for width in (80, 120):
        text = render(compile_cache_entries(entries), width=width)
        for entry in entries:
            assert entry.key in text
            assert entry.status.value in text
        assert "* the artifact exists, outside this artifact store" in text
    wide = render(compile_cache_entries(entries), width=160)
    assert "2026.4.0 " in wide
    assert "22959" not in wide, "the release is shown, not the build string"
    assert "nDimensional/Qwen3.5" in wide


def outside_owner_entry() -> CompileCacheEntry:
    """An entry owned by a staging directory: an owner known only by its long name."""
    return CompileCacheEntry(
        key="fedcba9876543210",
        path=Path("/home/user/.cache/yonga/compile/fedcba9876543210"),
        size_bytes=10 * 2**30,
        last_used_at=datetime(2026, 10, 1, tzinfo=UTC),
        device="NPU",
        openvino_version="2026.4.1",
        artifact_path=Path("/a/.staging/job/7ae557604adf--cpu-int4-asym-g128-r08"),
        repo_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        owner_exists=True,
        owner_in_store=False,
        status=CacheEntryStatus.STALE_RUNTIME,
    )


@pytest.mark.parametrize("width", [60, 80, 100])
@pytest.mark.parametrize("why", [False, True])
def test_compile_cache_entries_never_squeeze_a_key_beside_a_long_owner(
    width: int, why: bool
) -> None:
    """A long directory-name owner is cut, then dropped; keys and ids stay whole."""
    entries = [*cache_entries(), outside_owner_entry()]
    reasons = {entry.key: "older-than" for entry in entries} if why else None

    text = render(compile_cache_entries(entries, reasons=reasons), width=width)

    assert widest_line(text) <= width
    for entry in entries:
        assert entry.key in text
        assert entry.status.value in text
    if why:
        assert text.count("older-than") == len(entries)
    if "Owner" in text:
        assert "201e471c1669050f" in text
        assert "bbbbbbbbbbbbbbbb*" in text
        assert "* the artifact exists, outside this artifact store" in text
    else:
        # No owner column, so no '*' for the footnote to explain.
        assert "outside this artifact store" not in text
    if width >= 80:
        assert "Owner" in text


def test_a_prune_plan_says_why_only_when_it_is_not_the_status() -> None:
    entries = cache_entries()[1:]
    same = {entry.key: entry.status.value for entry in entries}
    aged = {**same, entries[1].key: "older-than"}

    assert "Why" not in render(compile_cache_entries(entries, reasons=same), width=160)
    assert "older-than" in render(compile_cache_entries(entries, reasons=aged), width=160)
    assert "empty" in render(compile_cache_entries([]))


#: The shape of a real deferred NPU compile: an exception chain, padding lines
#: and the native compiler output appended after them.
DEFERRED_NPU_TEXT = (
    "A raw compile on NPU is not conclusive for this graph; the GenAI pipeline decides. "
    "Raw error: Exception from src/inference/src/cpp/core.cpp:117:\n"
    "Exception from src/plugins/intel_npu/src/plugin/src/plugin.cpp:560:\n"
    "Compilation failed. Level0 pfnCreate2 result: ZE_RESULT_ERROR_INVALID_ARGUMENT\n"
    "to_shape was called on a dynamic shape.\n"
    "\n   \n\n\n\n\n"
    "--- native output ---\n"
    "[ERROR] [IE::FrontEnd::importNetwork] Upper bounds are not specified for node 'ShapeOf_1'"
)


def _deferred_npu_report() -> DeviceProbeReport:
    """An NPU probe that passed: the raw compile deferred, the pipeline generated."""
    return DeviceProbeReport(
        device="NPU",
        status=ProbeStatus.OK,
        submodels=[
            ProbeResult(
                device="NPU",
                role=SubmodelRole.LANGUAGE,
                status=ProbeStatus.DEFERRED,
                duration_seconds=0.7,
                message=DEFERRED_NPU_TEXT,
                diagnostic=diagnostic(
                    "npu.compiler.dynamic_shape",
                    FailureCategory.COMPILER_UNSUPPORTED_GRAPH,
                    "The NPU needs a statically shaped graph",
                    "The NPU compiler requires every tensor shape to be known.",
                ),
            ),
            ProbeResult(device="CPU", role=SubmodelRole.TOKENIZER, status=ProbeStatus.OK),
        ],
        smoke_test=ProbeResult(
            device="NPU", status=ProbeStatus.OK, message="generated 24 token(s)"
        ),
    )


def test_a_passing_npu_probe_shows_its_deferred_raw_compile_in_one_line():
    lines = render(probe_report([_deferred_npu_report()])).splitlines()

    language = [line for line in lines if "language" in line]
    assert len(language) == 1
    assert "deferred" in language[0]
    assert "the GenAI pipeline decides" in language[0]
    text = "\n".join(lines)
    assert "Exception from" not in text
    assert "native output" not in text
    assert "Upper bounds" not in text
    assert all(line.strip() for line in lines), "no padded empty rows"
    assert "generated 24 token(s)" in text


def test_verbose_shows_the_raw_text_of_a_deferred_compile_without_its_blank_lines():
    lines = render(probe_report([_deferred_npu_report()], verbose=True), width=200).splitlines()

    text = "\n".join(lines)
    assert "to_shape was called on a dynamic shape" in text
    assert "Upper bounds are not specified" in text
    assert all(line.strip() for line in lines), "no padded empty rows"


def test_a_recognised_failure_row_is_named_by_its_diagnostic():
    report = DeviceProbeReport(
        device="NPU",
        status=ProbeStatus.FAILED,
        submodels=[ProbeResult(device="NPU", role=SubmodelRole.LANGUAGE, status=ProbeStatus.OK)],
        smoke_test=ProbeResult(
            device="NPU",
            status=ProbeStatus.FAILED,
            message="Exception from src/inference/src/cpp/core.cpp:117:\n\n\nCompilation failed",
            diagnostic=diagnostic(
                "npu.compiler.unsupported_graph",
                FailureCategory.COMPILER_UNSUPPORTED_GRAPH,
                "The NPU compiler refused this graph",
                "One of the NPU compiler's passes could not compile this graph.",
            ),
        ),
    )

    text = render(probe_report([report]))

    smoke = next(line for line in text.splitlines() if "smoke test" in line)
    assert "The NPU compiler refused this graph" in smoke
    assert "Exception from" not in text
    assert "Exception from" in render(probe_report([report], verbose=True))


def test_benchmark_summary_counts_runs_and_warmups_in_good_english():
    result = benchmark().model_copy(update={"runs": 3, "warmups": 1})
    assert "3 runs, 1 warmup)" in render(benchmark_summary(result))
    result = benchmark().model_copy(update={"runs": 1, "warmups": 0})
    assert "1 run, 0 warmups)" in render(benchmark_summary(result))
