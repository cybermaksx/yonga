"""Identity hints: recognising an architecture that has no ``config.json``.

``HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive`` is the reference case
(``SPEC.md`` section 20). It publishes GGUF weights and nothing else, so
``model_type`` and ``architectures`` are empty and every device came back
``unknown / no_registry_rule``. The GGUF header the Hub reports does say what
the weights are — ``architecture: qwen35`` — and the registry rule is keyed on
``qwen3_5``, so the two were never connected.

These tests pin down the mechanism that connects them and, just as importantly,
the things it must refuse to do:

* a repository *tag* is never evidence. ``qwen3.5`` and ``qwen`` are in this
  repository's tag list and must not identify anything, because tags are typed
  by an uploader while the GGUF header is read out of the weight file;
* an alias must never outrank a declared identity. A rule that recognises the
  ``model_type`` a repository actually publishes wins over one that had to fall
  back on a hint, whatever the rules' priorities say;
* an alias must never *override* a declared identity either. A header is
  evidence only where nothing was declared, so a repository declaring
  ``gemma3`` while carrying a ``qwen35`` header is unknown, not a Qwen3.5.

The mechanism also has to stay extendable: a second architecture identified by
its own alias is one registry entry, never a priority puzzle.

Everything here is registry *data*: the same mechanism identifies a future
architecture nobody has written code for, as long as someone lists its alias.
"""

from __future__ import annotations

import textwrap

from tests.unit.test_inspect_service import (
    GGUF_FILES,
    REQUESTED_REPO,
    REQUESTED_SHA,
    FakeHub,
    build_metadata,
    build_service,
)
from yonga.domain.compatibility import DeviceStatus, EvidenceLevel
from yonga.domain.models import ModelIdentity, ModelMetadata
from yonga.registry.loader import (
    load_builtin_documents,
    load_builtin_registry,
    load_registry_from_documents,
    parse_document,
    validate_registry,
)
from yonga.registry.resolver import RegistryResolver
from yonga.registry.schema import (
    SPECIFICITY_ARCHITECTURE,
    SPECIFICITY_ARCHITECTURE_ALIAS,
    MatchSubject,
    Registry,
    RuleMatch,
    subject_from_metadata,
)

#: Exactly what this machine has installed (see ``HOST_FACTS.md``).
HOST_VERSIONS: dict[str, str | None] = {
    "openvino": "2026.4.0-22959-99c81491cc3-releases/2026/4",
    "openvino_genai": "2026.4.0.0",
    "openvino_tokenizers": "2026.4.0.0",
    "optimum": "2.3.0",
    "optimum_intel": "2.2.0",
    "transformers": "5.5.4",
    "nncf": "3.4.0",
    "torch": "2.14.0",
    "huggingface_hub": "1.21.0",
    "python": "3.14.4",
}

#: The tag list the Hub really returns for the reference repository.
HAUHAUCS_TAGS: list[str] = [
    "gguf",
    "uncensored",
    "qwen3.5",
    "qwen",
    "en",
    "zh",
    "multilingual",
    "base_model:Qwen/Qwen3.5-9B",
]

#: The GGUF metadata the Hub really returns for the reference repository.
HAUHAUCS_GGUF: dict[str, object] = {
    "total": 8_953_803_264,
    "architecture": "qwen35",
    "context_length": 262_144,
    "chat_template": "<omitted: 20000 characters>",
    "eos_token": "<|im_end|>",
    "totalFileSize": 17_920_696_704,
}


def registry_from(text: str) -> Registry:
    """Build a registry from one inline YAML document."""
    return load_registry_from_documents([parse_document(textwrap.dedent(text), origin="demo.yaml")])


def resolver_for(text: str) -> RegistryResolver:
    """Build a resolver over an inline registry document."""
    return RegistryResolver(registry_from(text), versions=HOST_VERSIONS)


def builtin_resolver() -> RegistryResolver:
    """Build a resolver over the shipped registry."""
    return RegistryResolver(load_builtin_registry(), versions=HOST_VERSIONS)


def hauhaucs_metadata(**overrides: object) -> ModelMetadata:
    """Metadata in the shape the Hub returns for the GGUF-only reference repository.

    No ``config.json``, so no ``model_type`` and no ``architectures``; the only
    machine-readable identity is the GGUF header.
    """
    metadata = build_metadata(
        REQUESTED_REPO,
        GGUF_FILES,
        resolved_revision=REQUESTED_SHA,
        license="apache-2.0",
        tags=HAUHAUCS_TAGS,
        gguf_metadata=dict(HAUHAUCS_GGUF),
    )
    return metadata.model_copy(update=dict(overrides)) if overrides else metadata


# --------------------------------------------------------------------------- #
# The hint is read from the weight file, never from a tag
# --------------------------------------------------------------------------- #


def test_subject_takes_its_hint_from_gguf_metadata() -> None:
    """``subject_from_metadata`` promotes the GGUF architecture to an identity hint."""
    subject = subject_from_metadata(hauhaucs_metadata())

    assert subject.model_type is None
    assert subject.architectures == ()
    assert subject.identity_hints == ("qwen35",)


def test_repository_tags_never_become_identity_hints() -> None:
    """Tags are uploader prose; matching them would be the SPEC 8.1 mistake."""
    metadata = build_metadata(
        REQUESTED_REPO,
        GGUF_FILES,
        tags=HAUHAUCS_TAGS,
    )

    subject = subject_from_metadata(metadata)

    assert "qwen3.5" in metadata.tags
    assert subject.identity_hints == ()


def test_a_non_string_gguf_architecture_is_ignored() -> None:
    """GGUF metadata is third-party data; a surprising type is dropped, not coerced."""
    metadata = build_metadata(REQUESTED_REPO, GGUF_FILES, gguf_metadata={"architecture": 42})

    assert subject_from_metadata(metadata).identity_hints == ()


# --------------------------------------------------------------------------- #
# Hints match aliases and nothing else
# --------------------------------------------------------------------------- #


def test_a_hint_never_satisfies_model_types_or_architectures() -> None:
    """Adding an alias must stay a deliberate registry edit, not a fuzzy match."""
    by_model_type = RuleMatch(model_types=["qwen35"])
    by_class = RuleMatch(architectures=["qwen35"])
    subject = MatchSubject(identity_hints=("qwen35",))

    assert not by_model_type.evaluate(subject).matched
    assert not by_class.evaluate(subject).matched


def test_an_alias_never_widens_a_declared_identity() -> None:
    """The comparison runs one way only: aliases are matched against hints."""
    match = RuleMatch(model_types=["qwen3_5"], aliases=["qwen35"])

    assert not match.evaluate(MatchSubject(model_type="qwen35")).matched
    assert match.evaluate(MatchSubject(identity_hints=("qwen35",))).matched


def test_alias_matching_is_casefolded_like_every_other_criterion() -> None:
    """A GGUF header that shouts is still the same identifier."""
    match = RuleMatch(aliases=["QWEN35"])

    outcome = match.evaluate(MatchSubject(identity_hints=("Qwen35",)))

    assert outcome.matched
    assert outcome.alias_hint == "qwen35"


def test_a_rejection_says_which_hints_were_considered() -> None:
    """An unmatched alias rule explains itself, so ``explain`` can print why."""
    match = RuleMatch(model_types=["qwen3_5"], aliases=["qwen35"])

    outcome = match.evaluate(MatchSubject(identity_hints=("llama",)))

    assert not outcome.matched
    assert any("match no alias" in reason for reason in outcome.reasons)


# --------------------------------------------------------------------------- #
# A hint is evidence only in the absence of a declaration
# --------------------------------------------------------------------------- #


def test_a_declared_identity_closes_the_alias_path() -> None:
    """A repository that declares an identity is never re-identified from a header.

    ``SPEC.md`` section 3 forbids silently substituting a semantically different
    model and section 8.1 forbids inferring equivalence, so a declaration this
    rule does not list is a rejection — the hint is not consulted at all.
    """
    match = RuleMatch(model_types=["qwen3_5"], aliases=["qwen35"])

    outcome = match.evaluate(
        MatchSubject(
            model_type="gemma3",
            architectures=("Gemma3ForCausalLM",),
            identity_hints=("qwen35",),
        )
    )

    assert not outcome.matched
    assert outcome.alias_hint is None
    assert any("gemma3 is not listed" in reason for reason in outcome.reasons)
    assert not any("matched alias" in reason for reason in outcome.reasons)


def test_a_partially_declared_identity_still_closes_the_alias_path() -> None:
    """Architecture classes alone are a declaration too, even with no ``model_type``."""
    match = RuleMatch(aliases=["qwen35"])

    outcome = match.evaluate(
        MatchSubject(architectures=("Gemma3ForCausalLM",), identity_hints=("qwen35",))
    )

    assert not outcome.matched
    assert outcome.alias_hint is None


def test_a_gguf_header_never_overrides_a_contradicting_config_json() -> None:
    """The shipped registry must answer 'unknown', not 'Qwen3.5', for this repository.

    A repository declaring ``gemma3`` while shipping a GGUF header that says
    ``qwen35`` is a contradiction. Resolving it to ``arch-qwen3_5`` would hand
    the user Qwen3.5's exporter pin, pipeline, recommended device and full
    CPU/GPU/NPU verdict for a model that says it is a Gemma 3.
    """
    metadata = hauhaucs_metadata(model_type="gemma3", architectures=["Gemma3ForCausalLM"])

    rule, explanation = builtin_resolver().resolve_architecture(metadata)

    assert rule is None
    assert explanation.winner_id is None
    assert all(not candidate.matched for candidate in explanation.candidates)
    qwen = next(item for item in explanation.candidates if item.rule_id == "arch-qwen3_5")
    assert not any("matched alias" in reason for reason in qwen.reasons)


# --------------------------------------------------------------------------- #
# The reference case
# --------------------------------------------------------------------------- #


def test_the_gguf_only_reference_repository_resolves_to_the_qwen35_rule() -> None:
    """``docs/CLI_SPEC.md`` section 4: this repository is a Qwen3.5."""
    rule, explanation = builtin_resolver().resolve_architecture(hauhaucs_metadata())

    assert rule is not None
    assert rule.id == "arch-qwen3_5"
    assert explanation.winner_id == "arch-qwen3_5"


def test_the_explanation_records_that_an_alias_did_the_identifying() -> None:
    """``registry explain`` must not let an alias match pass for a config.json one."""
    resolver = builtin_resolver()

    _rule, explanation = resolver.resolve_architecture(hauhaucs_metadata())

    winner = next(item for item in explanation.candidates if item.rule_id == "arch-qwen3_5")
    assert any("matched alias 'qwen35'" in reason for reason in winner.reasons)
    assert any("GGUF metadata" in reason for reason in winner.reasons)
    assert any("GGUF metadata" in note and "no config.json" in note for note in explanation.notes)


def test_the_alias_match_scores_below_a_declared_identity_match() -> None:
    """The same rule scores lower when only a hint could reach it."""
    resolver = builtin_resolver()

    _hinted, hinted = resolver.resolve_architecture(hauhaucs_metadata())
    _declared, declared = resolver.resolve_architecture(
        ModelMetadata(
            identity=ModelIdentity(repo_id="nDimensional/Qwen3.5-9B-Uncensored-Safetensors"),
            model_type="qwen3_5",
            architectures=["Qwen3_5ForConditionalGeneration"],
        )
    )

    by_alias = next(item for item in hinted.candidates if item.rule_id == "arch-qwen3_5")
    by_model_type = next(item for item in declared.candidates if item.rule_id == "arch-qwen3_5")
    assert by_alias.specificity == SPECIFICITY_ARCHITECTURE_ALIAS
    assert by_model_type.specificity == SPECIFICITY_ARCHITECTURE
    assert by_alias.specificity < by_model_type.specificity


def test_npu_is_reported_unsupported_for_the_gguf_only_repository() -> None:
    """The CLI_SPEC promise: 'NPU known unsupported / current compiler rule'."""
    resolver = builtin_resolver()
    rule, _explanation = resolver.resolve_architecture(hauhaucs_metadata())

    entries = {entry.device: entry for entry in resolver.device_compatibility(arch_rule=rule)}

    npu = entries["NPU"]
    assert npu.status is DeviceStatus.UNSUPPORTED
    assert npu.evidence is EvidenceLevel.TESTED
    assert npu.rule_id == "arch-qwen3_5"
    assert npu.reason == "current_npu_compiler_graph_limitation"
    assert entries["GPU"].status is DeviceStatus.SUPPORTED
    assert entries["CPU"].status is DeviceStatus.SUPPORTED


def test_a_tag_only_repository_is_still_unidentified() -> None:
    """Strip the GGUF header and the qwen tags alone must decide nothing."""
    metadata = build_metadata(
        REQUESTED_REPO,
        GGUF_FILES,
        resolved_revision=REQUESTED_SHA,
        tags=HAUHAUCS_TAGS,
    )

    rule, explanation = builtin_resolver().resolve_architecture(metadata)

    assert rule is None
    assert explanation.winner_id is None
    assert all(not candidate.matched for candidate in explanation.candidates)


def test_an_unknown_alias_resolves_to_nothing_rather_than_a_wrong_rule() -> None:
    """An architecture the registry has never heard of stays unknown."""
    metadata = hauhaucs_metadata(gguf_metadata={"architecture": "nebula9"})

    rule, explanation = builtin_resolver().resolve_architecture(metadata)

    assert rule is None
    assert explanation.notes == ["No architecture rule matched; runtime probing decides."]


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #

TWO_WAYS_TO_IDENTIFY = """
architectures:
  - id: arch-by-model-type
    priority: 1
    match:
      model_types: [qwen3_5]
  - id: arch-by-alias
    priority: 100
    match:
      aliases: [qwen35]
"""


def test_a_declared_model_type_outscores_an_alias_even_at_lower_priority() -> None:
    """A declaration settles identity: the alias rule is not even a candidate.

    The hint is consulted only when nothing is declared, so the alias rule does
    not match at all here — and the rule that recognises the declared model type
    wins despite its far lower priority.
    """
    resolver = resolver_for(TWO_WAYS_TO_IDENTIFY)
    metadata = hauhaucs_metadata(model_type="qwen3_5")

    rule, explanation = resolver.resolve_architecture(metadata)

    assert rule is not None
    assert rule.id == "arch-by-model-type"
    scored = {item.rule_id: item for item in explanation.candidates}
    assert scored["arch-by-model-type"].matched
    assert not scored["arch-by-alias"].matched


def test_the_alias_rule_wins_when_nothing_declares_an_identity() -> None:
    """With no config.json the alias rule is the only one left, and it is used."""
    resolver = resolver_for(TWO_WAYS_TO_IDENTIFY)

    rule, _explanation = resolver.resolve_architecture(hauhaucs_metadata())

    assert rule is not None
    assert rule.id == "arch-by-alias"


# --------------------------------------------------------------------------- #
# The registry stays sound, and ``inspect`` tells the user where the name came from
# --------------------------------------------------------------------------- #


def test_the_builtin_registry_still_validates() -> None:
    """Seeding aliases must not make any rule pair ambiguous."""
    assert validate_registry(load_builtin_registry()) == []


def test_inspect_names_the_architecture_and_where_it_came_from() -> None:
    """``yonga inspect`` reports ``qwen3_5`` and admits the name came from GGUF."""
    hub = FakeHub({REQUESTED_REPO: hauhaucs_metadata()})

    report = build_service(hub).inspect(REQUESTED_REPO)

    assert report.architecture_rule_id == "arch-qwen3_5"
    assert report.architecture == "qwen3_5"
    assert any("GGUF metadata" in note for note in report.notes)
    assert not any("No registry rule matches this architecture" in note for note in report.notes)


# --------------------------------------------------------------------------- #
# The mechanism has to survive a second alias, or it is Qwen3.5-shaped
# --------------------------------------------------------------------------- #


def test_alias_rules_for_different_architectures_do_not_overlap() -> None:
    """No subject can satisfy both blocks, so validation must not claim otherwise."""
    llama = RuleMatch(model_types=["llama"], aliases=["llamacpp"])
    gemma = RuleMatch(model_types=["gemma3"], aliases=["gemma"])

    assert not llama.overlaps(gemma)
    assert not gemma.overlaps(llama)


def test_rules_sharing_an_alias_overlap() -> None:
    """One GGUF header would satisfy both blocks; that pair is genuinely ambiguous."""
    first = RuleMatch(model_types=["qwen3_5"], aliases=["qwen35"])
    second = RuleMatch(aliases=["qwen35"])

    assert first.overlaps(second)
    assert second.overlaps(first)


def test_an_alias_overlaps_a_rule_declaring_the_same_token() -> None:
    """The same token as alias and as declared identity stays a reportable pair."""
    aliased = RuleMatch(aliases=["qwen35"])
    declared = RuleMatch(model_types=["qwen35"])

    assert aliased.overlaps(declared)
    assert declared.overlaps(aliased)


def test_an_alias_rule_still_overlaps_an_unconstrained_rule() -> None:
    """A rule that constrains no identity can select any subject, alias or not."""
    assert RuleMatch(aliases=["qwen35"]).overlaps(RuleMatch(tasks=["text-generation"]))


SECOND_ALIASED_ARCHITECTURE = """
architectures:
  - id: arch-nebula9
    priority: 100
    summary: Second architecture identified from its GGUF header
    match:
      model_types:
        - nebula9
      aliases:
        - nebula
    export:
      pipeline: llm
    devices:
      cpu:
        status: supported
        evidence: inferred
"""


def test_a_second_alias_bearing_rule_needs_no_priority_tuning() -> None:
    """Teaching yonga a second GGUF spelling must be one registry line, not a rewrite.

    ``arch-nebula9`` carries an alias at exactly ``arch-qwen3_5``'s priority.
    The two cannot select the same subject, so validation stays clean and both
    rules resolve.
    """
    registry = load_registry_from_documents(
        [
            *load_builtin_documents(),
            parse_document(textwrap.dedent(SECOND_ALIASED_ARCHITECTURE), origin="extra.yaml"),
        ]
    )

    assert validate_registry(registry) == []

    resolver = RegistryResolver(registry, versions=HOST_VERSIONS)
    nebula, _ = resolver.resolve_architecture(
        hauhaucs_metadata(gguf_metadata={"architecture": "nebula"})
    )
    qwen, _ = resolver.resolve_architecture(hauhaucs_metadata())
    assert nebula is not None
    assert nebula.id == "arch-nebula9"
    assert qwen is not None
    assert qwen.id == "arch-qwen3_5"
