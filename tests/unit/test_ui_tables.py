"""Tests for the pure table/report renderers."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

import pytest
from rich.console import Console

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
from yonga.ui.tables import (
    SourceOption,
    artifact_list,
    benchmark_summary,
    doctor_report,
    inspect_report,
    preparation_plan,
    probe_report,
    registry_explanation,
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
def test_artifact_list_renders_rows():
    summaries = [
        ArtifactSummary(
            artifact_id="qwen35-gpu",
            path=Path("/artifacts/qwen35-gpu"),
            repo_id="Qwen/Qwen3.5-9B",
            resolved_revision=SHA,
            profile="gpu-int4-asym-g128",
            recommended_device="GPU",
            size_bytes=6 * 1024**3,
            created_at=NOW,
            ready=True,
        ),
        ArtifactSummary(
            artifact_id="broken",
            path=Path("/artifacts/broken"),
            repo_id="a/b",
            created_at=NOW,
        ),
    ]
    text = render(artifact_list(summaries))
    assert "qwen35-gpu" in text
    assert "0123456789ab" in text
    assert "6.0 GiB" in text
    assert "ready" in text
    assert "incomplete" in text


def test_artifact_list_handles_empty_input():
    assert "No prepared artifacts" in render(artifact_list([]))


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
