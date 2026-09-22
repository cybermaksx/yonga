"""Unit tests for yonga.domain.compatibility."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from yonga.domain.compatibility import (
    AUTO_DEVICE,
    CompatibilityDecision,
    DeviceCompatibility,
    DeviceStatus,
    EvidenceLevel,
    ResolutionExplanation,
    RuleCandidate,
    strongest_evidence,
)

ORDERED_LEVELS = [
    EvidenceLevel.OFFICIAL,
    EvidenceLevel.TESTED,
    EvidenceLevel.REPORTED,
    EvidenceLevel.INFERRED,
    EvidenceLevel.UNKNOWN,
]


def test_evidence_rank_values() -> None:
    assert EvidenceLevel.OFFICIAL.rank == 4
    assert EvidenceLevel.TESTED.rank == 3
    assert EvidenceLevel.REPORTED.rank == 2
    assert EvidenceLevel.INFERRED.rank == 1
    assert EvidenceLevel.UNKNOWN.rank == 0


def test_evidence_rank_is_strictly_descending() -> None:
    ranks = [level.rank for level in ORDERED_LEVELS]
    assert ranks == sorted(ranks, reverse=True)
    assert len(set(ranks)) == len(ORDERED_LEVELS)


def test_evidence_rank_sorts_levels() -> None:
    shuffled = [
        EvidenceLevel.INFERRED,
        EvidenceLevel.OFFICIAL,
        EvidenceLevel.UNKNOWN,
        EvidenceLevel.REPORTED,
        EvidenceLevel.TESTED,
    ]
    assert sorted(shuffled, key=lambda level: level.rank, reverse=True) == ORDERED_LEVELS


def test_strongest_evidence() -> None:
    assert strongest_evidence([]) is EvidenceLevel.UNKNOWN
    assert strongest_evidence([EvidenceLevel.INFERRED]) is EvidenceLevel.INFERRED
    assert (
        strongest_evidence([EvidenceLevel.INFERRED, EvidenceLevel.TESTED, EvidenceLevel.UNKNOWN])
        is EvidenceLevel.TESTED
    )


def test_device_status_usability() -> None:
    assert DeviceStatus.SUPPORTED.is_usable
    assert DeviceStatus.EXPERIMENTAL.is_usable
    assert DeviceStatus.UNKNOWN.is_usable
    assert not DeviceStatus.UNSUPPORTED.is_usable


def test_device_compatibility_normalises_device_name() -> None:
    entry = DeviceCompatibility(
        device="gpu.0",
        status=DeviceStatus.SUPPORTED,
        evidence=EvidenceLevel.TESTED,
    )
    assert entry.device == "GPU"


def test_decision_normalises_and_dedupes_devices() -> None:
    decision = CompatibilityDecision(
        requested_device="auto",
        selected_device="gpu.1",
        status=DeviceStatus.SUPPORTED,
        evidence_level=EvidenceLevel.OFFICIAL,
        fallback_candidates=["GPU.0", "gpu", "cpu"],
    )
    assert decision.requested_device == AUTO_DEVICE
    assert decision.selected_device == "GPU"
    assert decision.fallback_candidates == ["GPU", "CPU"]
    assert decision.is_auto
    assert decision.usable


def test_decision_explicit_request_is_not_auto() -> None:
    decision = CompatibilityDecision(
        requested_device="npu",
        selected_device=None,
        status=DeviceStatus.UNSUPPORTED,
        evidence_level=EvidenceLevel.TESTED,
        explicit_request=True,
    )
    assert decision.requested_device == "NPU"
    assert not decision.is_auto
    assert not decision.usable


def test_decision_compatibility_lookup() -> None:
    decision = CompatibilityDecision(
        per_device=[
            DeviceCompatibility(
                device="NPU",
                status=DeviceStatus.UNSUPPORTED,
                evidence=EvidenceLevel.TESTED,
                reason="current_npu_compiler_graph_limitation",
            ),
            DeviceCompatibility(
                device="GPU",
                status=DeviceStatus.SUPPORTED,
                evidence=EvidenceLevel.OFFICIAL,
            ),
        ]
    )
    npu = decision.compatibility_for("npu.0")
    assert npu is not None
    assert npu.status is DeviceStatus.UNSUPPORTED
    assert decision.compatibility_for("CPU") is None


def test_resolution_explanation_partitions_candidates() -> None:
    explanation = ResolutionExplanation(
        subject="architecture rule",
        candidates=[
            RuleCandidate(
                rule_id="arch-qwen3_5", kind="architecture", specificity=60, matched=True
            ),
            RuleCandidate(
                rule_id="arch-generic",
                kind="architecture",
                specificity=20,
                matched=False,
                reasons=["constraint openvino>=2027 not satisfied"],
            ),
        ],
        winner_id="arch-qwen3_5",
    )
    assert [candidate.rule_id for candidate in explanation.matched] == ["arch-qwen3_5"]
    assert [candidate.rule_id for candidate in explanation.rejected] == ["arch-generic"]


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(PydanticValidationError):
        DeviceCompatibility.model_validate(
            {
                "device": "CPU",
                "status": DeviceStatus.SUPPORTED,
                "evidence": EvidenceLevel.TESTED,
                "unexpected": "nope",
            }
        )
