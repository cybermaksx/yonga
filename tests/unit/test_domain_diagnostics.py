"""Unit tests for yonga.domain.diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from yonga.domain.diagnostics import (
    CATEGORY_EXIT_CODES,
    RAW_EXCERPT_MAX_CHARS,
    CancelledError,
    CompileError,
    ConfigurationError,
    DeviceUnsupportedError,
    Diagnostic,
    ExitCode,
    ExportError,
    FailureCategory,
    GenerationError,
    HealthCheck,
    HealthSection,
    HubAuthError,
    PreflightError,
    RegistryError,
    RemoteCodeRequiredError,
    RepairError,
    Severity,
    SourceResolutionError,
    ValidationError,
    YongaError,
    diagnostic,
    exit_code_for_category,
    worst_severity,
)

# Written out independently of the implementation so that a change to the table
# has to be made deliberately in two places. Mirrors docs/CLI_SPEC.md section 15.
EXPECTED_EXIT_CODES = {
    FailureCategory.DEVICE_UNAVAILABLE: 50,
    FailureCategory.DRIVER_MISSING: 10,
    FailureCategory.PERMISSION_DENIED: 10,
    FailureCategory.COMPILER_MISSING: 10,
    FailureCategory.COMPILER_UNSUPPORTED_GRAPH: 51,
    FailureCategory.OUT_OF_MEMORY: 30,
    FailureCategory.INSUFFICIENT_DISK: 30,
    FailureCategory.UNSUPPORTED_IR_TYPE: 41,
    FailureCategory.TOKENIZER_IR_INCOMPATIBLE: 41,
    FailureCategory.PACKAGE_ABI_MISMATCH: 10,
    FailureCategory.SOURCE_FORMAT_UNSUPPORTED: 20,
    FailureCategory.REMOTE_CODE_REQUIRED: 22,
    FailureCategory.GATED_MODEL_AUTH_REQUIRED: 21,
    FailureCategory.DOWNLOAD_FAILURE: 20,
    FailureCategory.CONFIGURATION_ERROR: 70,
    FailureCategory.REGISTRY_ERROR: 70,
    FailureCategory.USER_CANCELLED: 60,
    FailureCategory.UNKNOWN_RUNTIME_ERROR: 99,
}


def _diag(category: FailureCategory = FailureCategory.UNKNOWN_RUNTIME_ERROR) -> Diagnostic:
    return diagnostic("test.code", category, "Title", "Message")


class TestExitCodeTable:
    def test_every_failure_category_is_mapped(self) -> None:
        assert set(CATEGORY_EXIT_CODES) == set(FailureCategory)

    def test_table_has_no_extra_entries(self) -> None:
        assert len(CATEGORY_EXIT_CODES) == len(list(FailureCategory))

    def test_all_values_are_exit_codes(self) -> None:
        assert all(isinstance(code, ExitCode) for code in CATEGORY_EXIT_CODES.values())

    def test_table_matches_cli_spec(self) -> None:
        actual = {category: int(code) for category, code in CATEGORY_EXIT_CODES.items()}
        assert actual == dict(EXPECTED_EXIT_CODES)

    @pytest.mark.parametrize("category", list(FailureCategory))
    def test_diagnostic_exit_code_uses_table(self, category: FailureCategory) -> None:
        assert _diag(category).exit_code() is CATEGORY_EXIT_CODES[category]
        assert exit_code_for_category(category) is CATEGORY_EXIT_CODES[category]

    def test_reserved_exit_code_values(self) -> None:
        assert ExitCode.SUCCESS.value == 0
        assert ExitCode.WARNINGS.value == 2
        assert ExitCode.INTERNAL.value == 99


class TestEffectiveExitCode:
    @pytest.mark.parametrize(
        ("severity", "expected"),
        [
            (Severity.OK, ExitCode.SUCCESS),
            (Severity.INFO, ExitCode.SUCCESS),
            (Severity.WARNING, ExitCode.WARNINGS),
            (Severity.ERROR, ExitCode.PREFLIGHT),
            (Severity.UNKNOWN, ExitCode.PREFLIGHT),
        ],
    )
    def test_severity_is_respected(self, severity: Severity, expected: ExitCode) -> None:
        entry = diagnostic(
            "disk.low",
            FailureCategory.INSUFFICIENT_DISK,
            "Low disk",
            "Not enough space",
            severity=severity,
        )
        assert entry.effective_exit_code() is expected

    def test_plain_exit_code_ignores_severity(self) -> None:
        entry = diagnostic(
            "disk.low",
            FailureCategory.INSUFFICIENT_DISK,
            "Low disk",
            "Not enough space",
            severity=Severity.WARNING,
        )
        assert entry.exit_code() is ExitCode.PREFLIGHT


class TestDiagnosticModel:
    def test_json_round_trip(self, tmp_path: Path) -> None:
        original = Diagnostic(
            code="npu.compiler.unsupported_graph.qwen35",
            category=FailureCategory.COMPILER_UNSUPPORTED_GRAPH,
            severity=Severity.ERROR,
            title="NPU compiler cannot compile this graph",
            message="OneShotBufferizeVPU2VPUIP failed : Only supports FP16.",
            remediation=["Try --device gpu"],
            fallback_devices=["GPU", "CPU"],
            environment_broken=False,
            retry_same_device=False,
            matched_rule_id="issue.npu.qwen35",
            raw_excerpt="raw compiler output",
            log_path=tmp_path / "job.jsonl",
            job_id="job-1234",
            context={"device": "NPU", "architecture": "qwen3_5"},
        )

        payload = original.to_json()
        encoded = json.dumps(payload)
        restored = Diagnostic.model_validate(json.loads(encoded))

        assert restored == original
        assert payload["category"] == "compiler_unsupported_graph"
        assert payload["severity"] == "error"
        assert payload["log_path"] == str(tmp_path / "job.jsonl")
        assert restored.log_path == tmp_path / "job.jsonl"
        assert restored.exit_code() is ExitCode.COMPILE

    def test_defaults_are_independent_between_instances(self) -> None:
        first = _diag()
        second = _diag()
        first.remediation.append("do something")
        assert second.remediation == []

    def test_fallback_devices_are_normalised_and_deduplicated(self) -> None:
        entry = diagnostic(
            "npu.unsupported",
            FailureCategory.COMPILER_UNSUPPORTED_GRAPH,
            "t",
            "m",
            fallback_devices=["gpu", "GPU.0", " cpu ", "GPU"],
        )
        assert entry.fallback_devices == ["GPU", "CPU"]

    def test_raw_excerpt_is_truncated(self) -> None:
        entry = diagnostic(
            "x.y",
            FailureCategory.UNKNOWN_RUNTIME_ERROR,
            "t",
            "m",
            raw_excerpt="a" * (RAW_EXCERPT_MAX_CHARS * 2),
        )
        assert entry.raw_excerpt is not None
        assert len(entry.raw_excerpt) <= RAW_EXCERPT_MAX_CHARS
        assert entry.raw_excerpt.endswith("truncated ...]")

    def test_short_raw_excerpt_is_untouched(self) -> None:
        entry = diagnostic(
            "x.y", FailureCategory.UNKNOWN_RUNTIME_ERROR, "t", "m", raw_excerpt="short"
        )
        assert entry.raw_excerpt == "short"

    def test_empty_code_is_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            Diagnostic(
                code="   ",
                category=FailureCategory.REGISTRY_ERROR,
                title="t",
                message="m",
            )

    def test_factory_sets_optional_fields(self) -> None:
        entry = diagnostic(
            "config.telemetry",
            FailureCategory.CONFIGURATION_ERROR,
            "Telemetry is not available",
            "telemetry must stay disabled",
            severity=Severity.ERROR,
            remediation=["Remove telemetry from config.toml"],
            log_path=Path("/tmp/x.jsonl"),
        )
        assert entry.category is FailureCategory.CONFIGURATION_ERROR
        assert entry.remediation == ["Remove telemetry from config.toml"]
        assert entry.log_path == Path("/tmp/x.jsonl")
        assert entry.exit_code() is ExitCode.REGISTRY


class TestTypedErrors:
    def test_primary_constructor_takes_a_diagnostic(self) -> None:
        entry = _diag(FailureCategory.REGISTRY_ERROR)
        error = RegistryError(entry)
        assert error.diagnostic is entry
        assert error.exit_code() is ExitCode.REGISTRY
        assert str(error) == "Message"

    def test_cause_is_chained(self) -> None:
        cause = RuntimeError("boom")
        error = ExportError("export died", cause=cause)
        assert error.cause is cause
        assert error.__cause__ is cause

    @pytest.mark.parametrize(
        ("error_type", "category", "exit_code"),
        [
            (YongaError, FailureCategory.UNKNOWN_RUNTIME_ERROR, ExitCode.INTERNAL),
            (ConfigurationError, FailureCategory.CONFIGURATION_ERROR, ExitCode.REGISTRY),
            (RegistryError, FailureCategory.REGISTRY_ERROR, ExitCode.REGISTRY),
            (
                SourceResolutionError,
                FailureCategory.SOURCE_FORMAT_UNSUPPORTED,
                ExitCode.SOURCE_RESOLUTION,
            ),
            (HubAuthError, FailureCategory.GATED_MODEL_AUTH_REQUIRED, ExitCode.AUTH_REQUIRED),
            (
                RemoteCodeRequiredError,
                FailureCategory.REMOTE_CODE_REQUIRED,
                ExitCode.REMOTE_CODE_REQUIRED,
            ),
            (PreflightError, FailureCategory.INSUFFICIENT_DISK, ExitCode.PREFLIGHT),
            (ExportError, FailureCategory.UNKNOWN_RUNTIME_ERROR, ExitCode.CONVERSION),
            (ValidationError, FailureCategory.UNKNOWN_RUNTIME_ERROR, ExitCode.VALIDATION),
            (RepairError, FailureCategory.UNKNOWN_RUNTIME_ERROR, ExitCode.REPAIR_DENIED),
            (
                DeviceUnsupportedError,
                FailureCategory.DEVICE_UNAVAILABLE,
                ExitCode.DEVICE_UNSUPPORTED,
            ),
            (CompileError, FailureCategory.COMPILER_UNSUPPORTED_GRAPH, ExitCode.COMPILE),
            (GenerationError, FailureCategory.UNKNOWN_RUNTIME_ERROR, ExitCode.GENERATION),
            (CancelledError, FailureCategory.USER_CANCELLED, ExitCode.CANCELLED),
        ],
    )
    def test_bare_message_uses_sensible_defaults(
        self,
        error_type: type[YongaError],
        category: FailureCategory,
        exit_code: ExitCode,
    ) -> None:
        error = error_type("something went wrong")
        assert isinstance(error, YongaError)
        assert error.diagnostic.category is category
        assert error.diagnostic.message == "something went wrong"
        assert error.diagnostic.code
        assert error.diagnostic.title
        assert error.exit_code() is exit_code

    def test_classified_diagnostic_wins_over_stage_fallback(self) -> None:
        entry = _diag(FailureCategory.INSUFFICIENT_DISK)
        assert ExportError(entry).exit_code() is ExitCode.PREFLIGHT

    def test_every_subclass_is_catchable_as_yonga_error(self) -> None:
        with pytest.raises(YongaError):
            raise CancelledError("user pressed Ctrl+C")


class TestHealth:
    def test_worst_severity_of_empty_is_ok(self) -> None:
        assert worst_severity([]) is Severity.OK

    def test_worst_severity_ranks_error_highest(self) -> None:
        assert worst_severity([Severity.OK, Severity.WARNING, Severity.ERROR]) is Severity.ERROR
        assert worst_severity([Severity.OK, Severity.UNKNOWN, Severity.INFO]) is Severity.UNKNOWN
        assert worst_severity([Severity.INFO, Severity.WARNING]) is Severity.WARNING

    def test_section_status_is_worst_check(self) -> None:
        section = HealthSection(
            id="devices",
            title="Devices",
            checks=[
                HealthCheck(id="cpu", label="CPU", status=Severity.OK, value="Intel"),
                HealthCheck(id="npu", label="NPU", status=Severity.WARNING, detail="no driver"),
            ],
        )
        assert section.status is Severity.WARNING

    def test_empty_section_is_ok(self) -> None:
        assert HealthSection(id="x", title="X").status is Severity.OK
