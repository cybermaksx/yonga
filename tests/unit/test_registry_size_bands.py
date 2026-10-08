"""Schema and loader tests for conversion-profile size bands (registry schema 2)."""

from __future__ import annotations

import textwrap
from typing import Any

import pytest
from pydantic import ValidationError

from yonga.domain.compatibility import EvidenceLevel
from yonga.domain.diagnostics import RegistryError
from yonga.domain.models import ModelIdentity, ModelMetadata
from yonga.registry.loader import (
    load_builtin_registry,
    load_registry_from_documents,
    parse_document,
    validate_registry,
)
from yonga.registry.resolver import RegistryResolver
from yonga.registry.schema import (
    PARAMETER_BAND_SCHEMA_VERSION,
    SCHEMA_MIN_TOOL_VERSION,
    SCHEMA_VERSION,
    SPECIFICITY_ARCHITECTURE,
    SPECIFICITY_GENERIC,
    ArchitectureRule,
    ChatProfileRule,
    CountRange,
    IssueRule,
    MatchSubject,
    ModelRule,
    ProfileRule,
    Registry,
    RuleMatch,
    format_parameter_count,
)

B = 1_000_000_000
M = 1_000_000


def band(**bounds: object) -> CountRange:
    """Build a band from raw values, the way a YAML file would supply them."""
    return CountRange.model_validate(bounds)


def banded(**bounds: object) -> RuleMatch:
    """A match block carrying only a size band."""
    return RuleMatch(parameter_count=band(**bounds))


def registry_from(text: str) -> Registry:
    """Build a registry from one inline YAML document."""
    return load_registry_from_documents([parse_document(textwrap.dedent(text), origin="demo.yaml")])


# ---------------------------------------------------------------------------
# schema constants
# ---------------------------------------------------------------------------


def test_schema_version_two_introduces_bands_and_names_a_tool_floor():
    assert SCHEMA_VERSION == 2
    assert PARAMETER_BAND_SCHEMA_VERSION == 2
    assert SCHEMA_MIN_TOOL_VERSION[2]


# ---------------------------------------------------------------------------
# CountRange parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (2_000_000_000, 2 * B),
        (1, 1),
        ("2000000000", 2 * B),
        ("2_000_000_000", 2 * B),
        ("2B", 2 * B),
        ("2b", 2 * B),
        ("1.5B", 1_500_000_000),
        ("600M", 600 * M),
        ("600m", 600 * M),
        (" 600 M ", 600 * M),
        ("750K", 750_000),
        ("1T", 1_000_000_000_000),
        ("0.5B", 500 * M),
    ],
)
def test_count_range_accepts_integers_and_suffixed_decimals(raw: object, expected: int):
    assert band(at_least=raw).at_least == expected
    assert band(below=raw).below == expected


@pytest.mark.parametrize(
    "raw",
    [
        2e9,
        2.0,
        1.5,
        "2e9",
        "2E9",
        True,
        False,
        "",
        "   ",
        "two billion",
        "2 GB",
        "2BB",
        "-1B",
        "-5",
        "1.0000000001B",
        "0.5",
        0,
        "0",
        "0B",
        -3,
        [2],
    ],
)
def test_count_range_rejects_floats_booleans_and_unreadable_text(raw: object):
    """A float is refused even when whole: the threshold must not depend on the spelling."""
    with pytest.raises(ValidationError):
        band(below=raw)


def test_count_range_needs_at_least_one_bound():
    with pytest.raises(ValidationError, match="needs at_least, below, or both"):
        CountRange()
    with pytest.raises(ValidationError, match="needs at_least, below, or both"):
        band(at_least=None, below=None)


@pytest.mark.parametrize(("low", "high"), [("2B", "2B"), ("8B", "2B")])
def test_count_range_rejects_an_empty_or_inverted_band(low: str, high: str):
    with pytest.raises(ValidationError, match="smaller than below"):
        band(at_least=low, below=high)


def test_count_range_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        band(above="2B")


def test_count_range_is_frozen():
    value = band(below="2B")
    with pytest.raises(ValidationError):
        value.below = 3 * B


def test_yaml_underscored_integer_loads_as_a_count():
    """YAML 1.1 reads ``2_000_000_000`` as an integer, so it is a valid threshold."""
    document = parse_document(
        textwrap.dedent(
            """
            schema_version: 2
            profiles:
              - id: gpu-small
                device: gpu
                evidence: tested
                applies_to:
                  parameter_count:
                    below: 2_000_000_000
                    at_least: 600M
            """
        ),
        origin="demo.yaml",
    )
    parsed = document.profiles[0].applies_to.parameter_count
    assert parsed is not None
    assert parsed.below == 2 * B
    assert parsed.at_least == 600 * M


@pytest.mark.parametrize("spelling", ["2e9", "2.0e+9", "2.5"])
def test_yaml_scientific_and_float_spellings_are_refused(spelling: str):
    """``2e9`` arrives as the string "2e9", ``2.0e+9`` as a float; both are refused."""
    text = textwrap.dedent(
        f"""
        schema_version: 2
        profiles:
          - id: gpu-small
            device: gpu
            evidence: tested
            applies_to:
              parameter_count:
                below: {spelling}
        """
    )
    with pytest.raises(RegistryError) as excinfo:
        parse_document(text, origin="demo.yaml")
    diagnostic = excinfo.value.diagnostic
    assert diagnostic.code == "registry.schema_violation"
    assert "parameter count" in diagnostic.message


# ---------------------------------------------------------------------------
# CountRange behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "verdict"),
    [
        (None, "unknown"),
        (1, "outside"),
        (B - 1, "outside"),
        (B, "inside"),
        (5 * B, "inside"),
        (8 * B - 1, "inside"),
        (8 * B, "outside"),
        (70 * B, "outside"),
    ],
)
def test_verdict_is_half_open_at_both_boundaries(count: int | None, verdict: str):
    assert band(at_least="1B", below="8B").verdict(count) == verdict


def test_verdict_with_a_single_bound():
    lower_only = band(at_least="8B")
    assert lower_only.verdict(8 * B - 1) == "outside"
    assert lower_only.verdict(8 * B) == "inside"
    assert lower_only.verdict(10**13) == "inside"
    upper_only = band(below="2B")
    assert upper_only.verdict(1) == "inside"
    assert upper_only.verdict(2 * B - 1) == "inside"
    assert upper_only.verdict(2 * B) == "outside"
    assert upper_only.verdict(None) == "unknown"


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ({"below": "2B"}, {"at_least": "2B"}, False),
        ({"at_least": "1B", "below": "2B"}, {"at_least": "2B", "below": "8B"}, False),
        ({"at_least": "1B", "below": "2B"}, {"at_least": "3B", "below": "4B"}, False),
        ({"below": "2B"}, {"at_least": "1B"}, True),
        ({"below": "2B"}, {"below": "8B"}, True),
        ({"at_least": "2B"}, {"at_least": "8B"}, True),
        ({"at_least": "1B", "below": "8B"}, {"at_least": "2B", "below": "3B"}, True),
        ({"below": 2 * B + 1}, {"at_least": "2B"}, True),
    ],
)
def test_intersects_treats_touching_half_open_bands_as_disjoint(
    left: dict[str, Any], right: dict[str, Any], expected: bool
):
    first, second = band(**left), band(**right)
    assert first.intersects(second) is expected
    assert second.intersects(first) is expected


def test_intersects_none_is_unbounded():
    assert band(below="2B").intersects(None)
    assert band(at_least="8B").intersects(None)


@pytest.mark.parametrize(
    ("bounds", "text"),
    [
        ({"below": "2B"}, "below 2B"),
        ({"at_least": "8B"}, "of 8B or more"),
        ({"at_least": "600M", "below": "1.5B"}, "of 600M to below 1.5B"),
        ({"at_least": 1_240_000_000, "below": "70B"}, "of 1.24B to below 70B"),
    ],
)
def test_describe_spells_the_band(bounds: dict[str, Any], text: str):
    assert band(**bounds).describe() == text


@pytest.mark.parametrize(
    ("count", "text"),
    [
        (1, "1"),
        (999, "999"),
        (1_000, "1K"),
        (750_000, "750K"),
        (752 * M, "752M"),
        (1_240_000_000, "1.24B"),
        (1_500_000_000, "1.5B"),
        (2 * B, "2B"),
        (8 * B, "8B"),
        (8_190_000_000, "8.19B"),
        (70 * B, "70B"),
        (12_340_000_000, "12.3B"),
        (10**12, "1T"),
    ],
)
def test_format_parameter_count(count: int, text: str):
    assert format_parameter_count(count) == text


def test_a_count_just_under_a_threshold_is_not_spelled_as_the_threshold():
    outcome = banded(below="2B").evaluate(MatchSubject(parameter_count=1_996_000_000))
    assert outcome.matched
    assert outcome.reasons != ["2B parameters is below 2B"]


# ---------------------------------------------------------------------------
# RuleMatch with a band
# ---------------------------------------------------------------------------


def test_band_inside_matches_and_explains_the_size():
    outcome = banded(below="2B").evaluate(MatchSubject(parameter_count=1_240_000_000))
    assert outcome.matched
    assert outcome.specificity == SPECIFICITY_GENERIC
    assert outcome.reasons == ["1.24B parameters is below 2B"]


def test_band_outside_does_not_match_and_says_why():
    outcome = banded(below="2B").evaluate(MatchSubject(parameter_count=8 * B))
    assert not outcome.matched
    assert outcome.reasons == ["8B parameters is not below 2B"]


def test_band_lower_boundary_belongs_to_the_upper_band():
    lower = banded(below="2B").evaluate(MatchSubject(parameter_count=2 * B))
    upper = banded(at_least="2B").evaluate(MatchSubject(parameter_count=2 * B))
    assert not lower.matched
    assert upper.matched
    assert upper.reasons == ["2B parameters is 2B or more"]


def test_unknown_size_is_outside_every_band_and_reports_its_basis():
    subject = MatchSubject(
        parameter_count=None,
        parameter_count_basis="the checkpoint is already quantized",
    )
    outcome = banded(at_least="8B").evaluate(subject)
    assert not outcome.matched
    assert outcome.reasons == [
        "parameter count unknown (the checkpoint is already quantized); this applies only "
        "to models of 8B or more"
    ]


def test_unknown_size_without_a_basis_uses_a_default_sentence():
    outcome = banded(below="2B").evaluate(MatchSubject())
    assert not outcome.matched
    assert "no parameter count is known" in outcome.reasons[0]


def test_band_combines_with_identity_criteria():
    match = RuleMatch(model_types=["llama"], parameter_count=band(at_least="8B"))
    inside = match.evaluate(MatchSubject(model_type="llama", parameter_count=8 * B))
    assert inside.matched
    assert inside.specificity == SPECIFICITY_ARCHITECTURE
    assert any("8B parameters is 8B or more" in reason for reason in inside.reasons)
    assert not match.evaluate(MatchSubject(model_type="llama", parameter_count=B)).matched
    assert not match.evaluate(MatchSubject(model_type="qwen3", parameter_count=8 * B)).matched


def test_unbanded_rule_ignores_the_size():
    outcome = RuleMatch().evaluate(MatchSubject(parameter_count=None))
    assert outcome.matched
    assert outcome.reasons == ["generic rule, no constraints"]


def test_match_subject_rejects_a_non_positive_count():
    with pytest.raises(ValidationError):
        MatchSubject(parameter_count=0)


def test_band_makes_a_match_block_non_generic():
    assert RuleMatch().is_generic
    assert not banded(below="2B").is_generic


def test_band_adds_no_specificity_rung():
    """A band filters and never ranks: priority, not the band, orders matching profiles."""
    assert banded(below="2B").static_specificity() == SPECIFICITY_GENERIC
    assert banded(below="2B").resolvable_specificities() == RuleMatch().resolvable_specificities()
    plain = RuleMatch(model_types=["llama"])
    with_band = RuleMatch(model_types=["llama"], parameter_count=band(at_least="8B"))
    assert with_band.static_specificity() == plain.static_specificity() == SPECIFICITY_ARCHITECTURE
    assert with_band.resolvable_specificities() == plain.resolvable_specificities()


def test_overlaps_requires_intersecting_bands():
    small = banded(below="2B")
    large = banded(at_least="2B")
    middle = banded(at_least="1B", below="4B")
    assert not small.overlaps(large)
    assert not large.overlaps(small)
    assert small.overlaps(middle)
    assert large.overlaps(middle)
    assert small.overlaps(RuleMatch())
    assert RuleMatch().overlaps(large)


def test_overlaps_with_disjoint_bands_is_false_even_for_the_same_architecture():
    first = RuleMatch(model_types=["llama"], parameter_count=band(below="8B"))
    second = RuleMatch(model_types=["llama"], parameter_count=band(at_least="8B"))
    assert not first.overlaps(second)


# ---------------------------------------------------------------------------
# which rules may carry a band
# ---------------------------------------------------------------------------

BAND_PAYLOAD: dict[str, object] = {"parameter_count": {"below": "2B"}}


@pytest.mark.parametrize(
    ("rule_type", "payload"),
    [
        (ArchitectureRule, {"id": "arch-x", "match": BAND_PAYLOAD}),
        (
            IssueRule,
            {
                "id": "issue-x",
                "match": BAND_PAYLOAD,
                "classification": {"code": "x.y", "category": "unknown"},
            },
        ),
        (ModelRule, {"id": "model-x", "match": BAND_PAYLOAD}),
        (ChatProfileRule, {"id": "chat-x", "match": BAND_PAYLOAD}),
    ],
)
def test_only_profiles_may_carry_a_band(rule_type: type[Any], payload: dict[str, object]):
    with pytest.raises(ValidationError, match="only honoured in a conversion profile"):
        rule_type.model_validate(payload)


@pytest.mark.parametrize(
    ("section", "block"),
    [("architectures", "match"), ("models", "match"), ("chat", "match")],
)
def test_a_band_on_a_non_profile_rule_is_a_schema_violation(section: str, block: str):
    text = textwrap.dedent(
        f"""
        schema_version: 2
        {section}:
          - id: banded-elsewhere
            {block}:
              model_types: [llama]
              parameter_count:
                below: 2B
        """
    )
    with pytest.raises(RegistryError) as excinfo:
        parse_document(text, origin="demo.yaml")
    assert excinfo.value.diagnostic.code == "registry.schema_violation"


def test_banded_profile_needs_schema_version_two():
    with pytest.raises(ValidationError, match="needs schema_version 2"):
        ProfileRule(
            id="gpu-small",
            device="gpu",
            schema_version=1,
            evidence=EvidenceLevel.TESTED,
            applies_to=banded(below="2B"),
        )


def test_banded_profile_must_state_its_evidence():
    with pytest.raises(ValidationError, match="must state its evidence"):
        ProfileRule(id="gpu-small", device="gpu", applies_to=banded(below="2B"))
    with pytest.raises(ValidationError, match="must state its evidence"):
        ProfileRule(
            id="gpu-small",
            device="gpu",
            evidence=EvidenceLevel.UNKNOWN,
            applies_to=banded(below="2B"),
        )


def test_banded_profile_with_schema_two_and_evidence_is_accepted():
    rule = ProfileRule(
        id="gpu-small",
        device="gpu",
        schema_version=2,
        evidence=EvidenceLevel.REPORTED,
        applies_to=banded(below="2B"),
    )
    assert rule.evidence is EvidenceLevel.REPORTED
    assert rule.match_block.parameter_count == band(below="2B")


def test_unbanded_profile_needs_neither_schema_two_nor_evidence():
    rule = ProfileRule(id="gpu-default", device="gpu", schema_version=1)
    assert rule.evidence is EvidenceLevel.UNKNOWN
    assert rule.applies_to.parameter_count is None


# ---------------------------------------------------------------------------
# parse_document and schema versions
# ---------------------------------------------------------------------------


def test_a_newer_document_with_unknown_fields_reports_the_version_not_the_fields():
    """A schema-3 file is expected to carry fields this build does not know."""
    text = textwrap.dedent(
        """
        schema_version: 3
        future_section:
          - id: something
        profiles:
          - id: gpu-future
            device: gpu
            a_field_from_the_future: true
        """
    )
    with pytest.raises(RegistryError) as excinfo:
        parse_document(text, origin="future.yaml")
    diagnostic = excinfo.value.diagnostic
    assert diagnostic.code == "registry.schema_version_unsupported"
    assert "schema_version 3" in diagnostic.message
    assert "future.yaml" in diagnostic.message


def test_unknown_fields_in_a_current_document_are_still_a_schema_violation():
    text = textwrap.dedent(
        """
        schema_version: 2
        profiles:
          - id: gpu-typo
            device: gpu
            applies_too: {}
        """
    )
    with pytest.raises(RegistryError) as excinfo:
        parse_document(text, origin="demo.yaml")
    assert excinfo.value.diagnostic.code == "registry.schema_violation"


def test_a_version_one_document_still_loads():
    document = parse_document(
        textwrap.dedent(
            """
            schema_version: 1
            profiles:
              - id: gpu-default
                schema_version: 1
                device: gpu
                summary: demo
                quantization:
                  weight_format: int4
            architectures:
              - id: arch-demo
                schema_version: 1
                match:
                  model_types: [demo]
                profiles:
                  gpu: gpu-default
            """
        ),
        origin="v1.yaml",
    )
    assert document.schema_version == 1
    profile = document.profiles[0]
    assert profile.schema_version == 1
    assert profile.applies_to.parameter_count is None
    assert profile.evidence is EvidenceLevel.UNKNOWN
    registry = load_registry_from_documents([document])
    assert validate_registry(registry) == []


def test_a_banded_profile_inside_a_version_one_rule_is_rejected_on_load():
    text = textwrap.dedent(
        """
        schema_version: 1
        profiles:
          - id: gpu-small
            schema_version: 1
            device: gpu
            evidence: tested
            applies_to:
              parameter_count:
                below: 2B
        """
    )
    with pytest.raises(RegistryError) as excinfo:
        parse_document(text, origin="demo.yaml")
    assert excinfo.value.diagnostic.code == "registry.schema_violation"
    assert "needs schema_version 2" in excinfo.value.diagnostic.message


def test_a_banded_profile_in_a_document_declaring_version_one_is_rejected():
    text = textwrap.dedent(
        """
        schema_version: 1
        profiles:
          - id: gpu-small
            device: gpu
            evidence: tested
            applies_to:
              parameter_count:
                below: 2B
        """
    )
    with pytest.raises(RegistryError):
        parse_document(text, origin="demo.yaml")


# ---------------------------------------------------------------------------
# validate_registry
# ---------------------------------------------------------------------------


def test_a_device_with_only_banded_profiles_is_flagged():
    """An unknown size is outside every band, so such a model would get no profile."""
    registry = registry_from(
        """
        schema_version: 2
        profiles:
          - id: gpu-small
            device: gpu
            evidence: tested
            applies_to:
              parameter_count: {below: 2B}
          - id: gpu-large
            device: gpu
            evidence: tested
            applies_to:
              parameter_count: {at_least: 2B}
          - id: cpu-default
            device: cpu
        """
    )
    problems = validate_registry(registry)
    assert any(
        "no conversion profile for GPU applies to models of unknown size" in item
        for item in problems
    )
    assert not any("for CPU" in item for item in problems)


def test_a_band_beaten_by_a_higher_priority_default_is_flagged():
    registry = registry_from(
        """
        schema_version: 2
        profiles:
          - id: gpu-default
            device: gpu
            priority: 100
          - id: gpu-small
            device: gpu
            priority: 50
            evidence: tested
            applies_to:
              parameter_count: {below: 2B}
        """
    )
    problems = validate_registry(registry)
    assert any(
        "'gpu-small'" in item
        and "can never win automatically" in item
        and "gpu-default has a higher priority (100)" in item
        for item in problems
    )


def test_a_band_above_the_default_priority_is_clean():
    registry = registry_from(
        """
        schema_version: 2
        profiles:
          - id: gpu-default
            device: gpu
            priority: 100
          - id: gpu-small
            device: gpu
            priority: 110
            evidence: tested
            applies_to:
              parameter_count: {below: 2B}
        """
    )
    assert validate_registry(registry) == []


def profiles_registry(*entries: str) -> Registry:
    """A schema-2 registry whose ``profiles`` section is the given YAML list items."""
    items = "".join(
        textwrap.indent(textwrap.dedent(entry).strip("\n"), "  ") + "\n" for entry in entries
    )
    return registry_from("schema_version: 2\nprofiles:\n" + items)


SMALL_LLAMA_PROFILES = (
    """
    - id: gpu-default
      device: gpu
      priority: 10
    """,
    """
    - id: gpu-small
      device: gpu
      priority: 50
      evidence: tested
      applies_to:
        model_types: [llama]
        parameter_count: {below: 2B}
    """,
)


def llama_rival(priority: int, *extra: str, model_type: str = "llama") -> str:
    """A GPU profile on ``gpu-small``'s specificity rung; ``extra`` adds match lines."""
    lines = [
        "- id: gpu-rival",
        "  device: gpu",
        f"  priority: {priority}",
        "  applies_to:",
        f"    model_types: [{model_type}]",
        *(f"    {line}" for line in extra),
    ]
    if extra:
        lines.insert(3, "  evidence: tested")
    return "\n".join(lines)


@pytest.mark.parametrize(
    "rival",
    [
        pytest.param(llama_rival(50), id="equal-priority"),
        pytest.param(llama_rival(100, model_type="qwen3"), id="disjoint-architecture"),
        pytest.param(llama_rival(100, "parameter_count: {below: 1B}"), id="banded-rival"),
    ],
)
def test_a_rival_that_does_not_beat_the_band_for_every_size_is_not_reported(rival: str):
    """Only a strictly higher-priority, unbanded, overlapping profile makes a band dead.

    Every rival sits on the same specificity rung as ``gpu-small`` (so the
    priority-only check is sound here), yet each leaves it some model: a tie is
    reported as ambiguous instead, a llama never matches a qwen3 profile, and a
    1.5B llama is outside the rival's own band.
    """
    registry = profiles_registry(*SMALL_LLAMA_PROFILES, rival)
    assert not any("can never win" in item for item in validate_registry(registry))


def test_a_band_beaten_by_several_rivals_is_reported_once():
    """Control for the test above: the layout reaches the check, reported once per band."""
    second_rival = """
    - id: gpu-rival-too
      device: gpu
      priority: 90
      applies_to:
        model_types: [llama]
    """
    registry = profiles_registry(*SMALL_LLAMA_PROFILES, llama_rival(100), second_rival)
    dead = [item for item in validate_registry(registry) if "can never win" in item]
    assert len(dead) == 1
    assert "'gpu-small'" in dead[0]
    assert "has a higher priority" in dead[0]


@pytest.mark.parametrize(
    "fallback",
    [
        pytest.param(
            """
            - id: gpu-default
              device: gpu
              status: deprecated
            """,
            id="deprecated-generic",
        ),
        pytest.param(
            """
            - id: gpu-llama
              device: gpu
              applies_to:
                model_types: [llama]
            """,
            id="unbanded-but-narrower",
        ),
    ],
)
def test_an_unbanded_profile_that_cannot_catch_every_model_is_not_a_fallback(fallback: str):
    """A model of unknown size needs a usable profile with no condition at all."""
    banded_only = """
    - id: gpu-small
      device: gpu
      evidence: tested
      applies_to:
        parameter_count: {below: 2B}
    """
    assert any(
        "no conversion profile for GPU applies to models of unknown size" in item
        for item in validate_registry(profiles_registry(banded_only, fallback))
    )


@pytest.mark.parametrize("carrier", ["architectures", "models"])
def test_a_band_named_by_a_rule_is_not_flagged_as_unreachable(carrier: str):
    registry = registry_from(
        f"""
        schema_version: 2
        profiles:
          - id: gpu-default
            device: gpu
            priority: 100
          - id: gpu-small
            device: gpu
            priority: 50
            evidence: tested
            applies_to:
              parameter_count: {{below: 2B}}
        {carrier}:
          - id: names-small
            match:
              model_types: [demo]
            profiles:
              gpu: gpu-small
        """
    )
    assert not any("can never win" in item for item in validate_registry(registry))


def test_a_band_named_only_by_a_deprecated_rule_is_still_flagged():
    registry = registry_from(
        """
        schema_version: 2
        profiles:
          - id: gpu-default
            device: gpu
            priority: 100
          - id: gpu-small
            device: gpu
            priority: 50
            evidence: tested
            applies_to:
              parameter_count: {below: 2B}
        architectures:
          - id: arch-old
            status: deprecated
            match:
              model_types: [demo]
            profiles:
              gpu: gpu-small
        """
    )
    assert any("can never win" in item for item in validate_registry(registry))


def test_disjoint_bands_at_equal_priority_are_not_ambiguous():
    registry = registry_from(
        """
        schema_version: 2
        profiles:
          - id: gpu-default
            device: gpu
            priority: 100
          - id: gpu-small
            device: gpu
            priority: 110
            evidence: tested
            applies_to:
              parameter_count: {below: 2B}
          - id: gpu-large
            device: gpu
            priority: 110
            evidence: tested
            applies_to:
              parameter_count: {at_least: 2B}
        """
    )
    assert validate_registry(registry) == []


def test_intersecting_bands_at_equal_priority_are_ambiguous():
    registry = registry_from(
        """
        schema_version: 2
        profiles:
          - id: gpu-default
            device: gpu
            priority: 100
          - id: gpu-small
            device: gpu
            priority: 110
            evidence: tested
            applies_to:
              parameter_count: {below: 4B}
          - id: gpu-medium
            device: gpu
            priority: 110
            evidence: tested
            applies_to:
              parameter_count: {at_least: 2B, below: 8B}
        """
    )
    problems = validate_registry(registry)
    assert any(
        "'gpu-small'" in item and "'gpu-medium'" in item and "ambiguous" in item
        for item in problems
    )


def test_deprecated_banded_profiles_are_ignored_by_band_validation():
    registry = registry_from(
        """
        schema_version: 2
        profiles:
          - id: gpu-default
            device: gpu
            priority: 100
          - id: gpu-small
            device: gpu
            status: deprecated
            priority: 50
            evidence: tested
            applies_to:
              parameter_count: {below: 2B}
        """
    )
    problems = validate_registry(registry)
    assert not any("can never win" in item or "unknown size" in item for item in problems)


SPECIFIC_BAND_REGISTRY = """
schema_version: 2
profiles:
  - id: gpu-default
    device: gpu
    priority: 100
  - id: gpu-llama-large
    device: gpu
    priority: 50
    evidence: tested
    applies_to:
      model_types: [llama]
      parameter_count: {at_least: 8B}
"""


def test_a_more_specific_banded_profile_wins_over_a_higher_priority_default():
    """Specificity ranks before priority, so this band is reachable."""
    resolver = RegistryResolver(registry_from(SPECIFIC_BAND_REGISTRY), versions={})
    meta = ModelMetadata(
        identity=ModelIdentity(repo_id="demo/llama-10b"),
        model_type="llama",
        parameter_count=10 * B,
    )
    profile, explanation = resolver.resolve_profile(device="gpu", meta=meta)
    assert profile.id == "gpu-llama-large"
    assert explanation.profile_selection is not None
    assert explanation.profile_selection.basis == "size_band"


@pytest.mark.parametrize(
    "text",
    [
        SPECIFIC_BAND_REGISTRY,
        """
        schema_version: 2
        profiles:
          - id: gpu-default
            device: gpu
            priority: 10
          - id: gpu-llama
            device: gpu
            priority: 100
            applies_to:
              model_types: [llama]
          - id: gpu-small
            device: gpu
            priority: 50
            evidence: tested
            applies_to:
              parameter_count: {below: 2B}
        """,
    ],
    ids=["band-more-specific", "rival-narrower"],
)
def test_a_reachable_band_is_not_reported_as_unreachable(text: str):
    assert not any("can never win" in item for item in validate_registry(registry_from(text)))


def test_builtin_registry_still_validates_clean():
    assert validate_registry(load_builtin_registry()) == []
