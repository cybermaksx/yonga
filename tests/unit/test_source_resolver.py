"""Source-resolution policy tests (``docs/TESTING.md`` "Source resolver").

The fixtures use the real repository file listings and the real commit SHAs
probed from the Hugging Face Hub on this host, so a change in the preference
order fails here rather than in front of a user with a 19 GB download queued.

Format observations are produced by the *real*
:func:`yonga.adapters.hf.detect_formats`, not by hand-written stubs: the
``nDimensional`` repository publishes its shards as
``model.safetensors-00001-of-00004.safetensors``, and a hand-written fixture
would quietly hide the fact that detection has to key on the suffix.
"""

from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path

import pytest

from yonga.adapters import HubClient
from yonga.adapters.hf import detect_formats
from yonga.application import resolve as resolve_module
from yonga.application.progress import CollectingSink
from yonga.application.resolve import (
    CODE_GGUF_UNSUPPORTED,
    CODE_NO_USABLE_SOURCE,
    CODE_REVISION_UNPINNED,
    CONFIRMATION_EXPLICIT_SOURCE,
    OPTION_DERIVATIVE,
    OPTION_GGUF,
    OPTION_OPENVINO,
    OPTION_SAFETENSORS,
    SourcePolicy,
    SourceResolver,
    describe_source_options,
)
from yonga.domain.diagnostics import (
    FailureCategory,
    SourceResolutionError,
    YongaError,
)
from yonga.domain.hardware import HuggingFaceAuthInfo
from yonga.domain.models import (
    ModelFormat,
    ModelIdentity,
    ModelMetadata,
    RepoFile,
    SourceRelationship,
)
from yonga.domain.plans import CONFIRMATION_THIRD_PARTY_DERIVATIVE
from yonga.registry.loader import (
    load_builtin_registry,
    load_registry_from_documents,
    parse_document,
)
from yonga.registry.resolver import RegistryResolver
from yonga.registry.schema import Registry

# --------------------------------------------------------------------------- #
# Verified Hub facts (probed 2026-09-22; see HOST_FACTS.md)
# --------------------------------------------------------------------------- #

REQUESTED_REPO = "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"
REQUESTED_SHA = "0a41c68809d375475f954be12ba7c40efa56c2a9"

DERIVATIVE_REPO = "nDimensional/Qwen3.5-9B-Uncensored-Safetensors"
DERIVATIVE_SHA = "4623e504074fc2489f3af98dc3d68b8392bbc21f"
DERIVATIVE_CARD_URL = "https://huggingface.co/nDimensional/Qwen3.5-9B-Uncensored-Safetensors"

OPENVINO_REPO = "OpenVINO/Qwen3.5-9B-int4-ov"
OPENVINO_SHA = "3f31fc1609edb0cbc8b131bc4c291f1b15cb0676"

OTHER_SHA = "1111111111111111111111111111111111111111"

#: The requested repository is GGUF-only and has no ``config.json``.
GGUF_FILES: tuple[tuple[str, int | None], ...] = (
    (".gitattributes", 1_500),
    ("README.md", 4_200),
    ("Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf", None),
    ("Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf", None),
    ("Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q6_K.gguf", None),
    ("Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf", None),
    ("mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf", None),
)

#: Non-standard shard names, and the exact byte sizes read from the Hub.
SAFETENSORS_FILES: tuple[tuple[str, int | None], ...] = (
    ("LICENSE", 11_000),
    ("README.md", 6_100),
    ("chat_template.jinja", 4_000),
    ("config.json", 2_100),
    ("merges.txt", 1_671_853),
    ("model.safetensors-00001-of-00004.safetensors", 5_278_251_472),
    ("model.safetensors-00002-of-00004.safetensors", 5_358_071_808),
    ("model.safetensors-00003-of-00004.safetensors", 5_340_098_264),
    ("model.safetensors-00004-of-00004.safetensors", 2_843_299_496),
    ("model.safetensors.index.json", 190_000),
    ("preprocessor_config.json", 700),
    ("tokenizer.json", 11_400_000),
    ("tokenizer_config.json", 5_000),
    ("vocab.json", 2_776_833),
)

OPENVINO_FILES: tuple[tuple[str, int | None], ...] = (
    ("chat_template.jinja", 4_000),
    ("config.json", 2_100),
    ("openvino_config.json", 800),
    ("openvino_detokenizer.bin", 1_100_000),
    ("openvino_detokenizer.xml", 60_000),
    ("openvino_language_model.bin", 5_100_000_000),
    ("openvino_language_model.xml", 3_200_000),
    ("openvino_text_embeddings_model.bin", 620_000_000),
    ("openvino_text_embeddings_model.xml", 40_000),
)

PYTORCH_FILES: tuple[tuple[str, int | None], ...] = (
    ("config.json", 2_100),
    ("pytorch_model.bin", 4_000_000_000),
    ("tokenizer.json", 1_000_000),
)


# --------------------------------------------------------------------------- #
# Fixtures and fakes
# --------------------------------------------------------------------------- #

_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin")


def build_metadata(
    repo_id: str,
    entries: tuple[tuple[str, int | None], ...],
    *,
    revision: str | None = None,
    resolved_revision: str | None = None,
    **extra: object,
) -> ModelMetadata:
    """Build repository metadata whose formats come from the real detector."""
    files = [
        RepoFile(path=path, size=size, lfs=path.endswith(_WEIGHT_SUFFIXES))
        for path, size in entries
    ]
    payload: dict[str, object] = {
        "identity": ModelIdentity(repo_id=repo_id, revision=revision),
        "resolved_revision": resolved_revision,
        "files": files,
        "formats": detect_formats(files),
    }
    payload.update(extra)
    return ModelMetadata.model_validate(payload)


class FakeHub:
    """A :class:`~yonga.adapters.HubClient` that serves canned metadata.

    Every call is recorded so a test can assert that resolution inspected only
    the repositories it had a declared reason to look at, and never downloaded.
    """

    def __init__(
        self,
        repos: dict[str, ModelMetadata] | None = None,
        *,
        revisions: dict[str, str] | None = None,
    ) -> None:
        self.repos: dict[str, ModelMetadata] = dict(repos or {})
        self.revisions: dict[str, str] = dict(revisions or {})
        self.inspected: list[str] = []
        self.resolved: list[str] = []
        self.downloaded: list[str] = []

    def resolve_revision(self, model: ModelIdentity) -> str:
        """Return the canned SHA for a repository, or fail like the real adapter."""
        self.resolved.append(model.repo_id)
        try:
            return self.revisions[model.repo_id]
        except KeyError:
            raise SourceResolutionError(f"no revision for {model.repo_id}") from None

    def inspect(self, model: ModelIdentity, *, with_sizes: bool = True) -> ModelMetadata:
        """Return the canned metadata for a repository."""
        self.inspected.append(model.repo_id)
        try:
            return self.repos[model.repo_id]
        except KeyError:
            raise SourceResolutionError(f"{model.repo_id} does not exist") from None

    def download(
        self,
        source: ModelIdentity,
        *,
        ctx: object | None = None,
        allow_patterns: object | None = None,
        ignore_patterns: object | None = None,
    ) -> Path:
        """Fail loudly: source resolution must never move weight shards."""
        self.downloaded.append(source.repo_id)
        raise AssertionError("source resolution must not download anything")

    def auth_info(self) -> HuggingFaceAuthInfo:
        """Report an anonymous session."""
        return HuggingFaceAuthInfo(logged_in=False)


def registry_from(text: str) -> Registry:
    """Build a registry from one inline YAML document."""
    return load_registry_from_documents([parse_document(textwrap.dedent(text), origin="test.yaml")])


def resolver_for(text: str) -> RegistryResolver:
    """Build a registry resolver over an inline document."""
    return RegistryResolver(registry_from(text), versions={})


def empty_resolver() -> RegistryResolver:
    """Build a registry resolver that declares nothing at all."""
    return RegistryResolver(Registry(), versions={})


def builtin_resolver() -> RegistryResolver:
    """Build a registry resolver over the shipped registry data."""
    return RegistryResolver(load_builtin_registry(), versions={})


@pytest.fixture
def gguf_only() -> ModelMetadata:
    """The requested GGUF-only repository, exactly as the Hub reports it."""
    return build_metadata(REQUESTED_REPO, GGUF_FILES, resolved_revision=REQUESTED_SHA)


@pytest.fixture
def safetensors_derivative() -> ModelMetadata:
    """The declared safetensors derivative, exactly as the Hub reports it."""
    return build_metadata(
        DERIVATIVE_REPO,
        SAFETENSORS_FILES,
        resolved_revision=DERIVATIVE_SHA,
        pipeline_tag="image-text-to-text",
        library_name="transformers",
        architectures=["Qwen3_5ForConditionalGeneration"],
        model_type="qwen3_5",
    )


# --------------------------------------------------------------------------- #
# 1. An existing OpenVINO artifact is used directly
# --------------------------------------------------------------------------- #


def test_openvino_repo_is_used_directly_without_re_export() -> None:
    """A complete IR artifact in the requested repo wins outright, unchanged."""
    metadata = build_metadata(OPENVINO_REPO, OPENVINO_FILES, resolved_revision=OPENVINO_SHA)
    hub = FakeHub({OPENVINO_REPO: metadata})

    resolved = SourceResolver(hub).resolve(metadata, registry_resolver=empty_resolver())

    assert resolved.already_openvino is True
    assert resolved.format is ModelFormat.OPENVINO_IR
    assert resolved.relationship is SourceRelationship.ORIGINAL
    assert resolved.source.repo_id == OPENVINO_REPO
    assert resolved.source.revision == OPENVINO_SHA
    assert resolved.requires_confirmation is False
    assert resolved.is_derivative is False
    assert hub.downloaded == []


def test_incomplete_openvino_ir_falls_through_to_safetensors() -> None:
    """An IR whose .bin is missing is not a usable direct artifact."""
    entries = tuple(
        (path, size) for path, size in OPENVINO_FILES if path != "openvino_language_model.bin"
    )
    entries += (("model.safetensors", 4_000_000_000),)
    metadata = build_metadata(OPENVINO_REPO, entries, resolved_revision=OPENVINO_SHA)

    resolution = SourceResolver(FakeHub()).resolve_detailed(
        metadata, registry_resolver=empty_resolver()
    )

    assert resolution.resolved.format is ModelFormat.SAFETENSORS
    assert resolution.resolved.already_openvino is False


# --------------------------------------------------------------------------- #
# 2/3. Weights in the requested repository
# --------------------------------------------------------------------------- #


def test_safetensors_are_the_preferred_conversion_source(
    safetensors_derivative: ModelMetadata,
) -> None:
    """Safetensors in the requested repository are used directly."""
    resolved = SourceResolver(FakeHub()).resolve(
        safetensors_derivative, registry_resolver=empty_resolver()
    )

    assert resolved.format is ModelFormat.SAFETENSORS
    assert resolved.relationship is SourceRelationship.ORIGINAL
    assert resolved.already_openvino is False
    assert resolved.source.revision == DERIVATIVE_SHA
    assert resolved.requires_confirmation is False
    assert resolved.metadata is not None
    assert resolved.metadata.weight_bytes == 18_819_721_040


def test_safetensors_win_over_pytorch_bin_in_the_same_repo() -> None:
    """Step 2 beats step 3 when a repository publishes both."""
    entries = (
        ("config.json", 2_100),
        ("model.safetensors", 4_000_000_000),
        ("pytorch_model.bin", 4_000_000_000),
    )
    metadata = build_metadata("acme/both", entries, resolved_revision=OTHER_SHA)

    resolved = SourceResolver(FakeHub()).resolve(metadata, registry_resolver=empty_resolver())

    assert resolved.format is ModelFormat.SAFETENSORS


def test_pytorch_bin_is_used_when_no_safetensors_exist() -> None:
    """Standard Transformers weights are step 3 of the preference order."""
    metadata = build_metadata("acme/torch-only", PYTORCH_FILES, resolved_revision=OTHER_SHA)

    resolved = SourceResolver(FakeHub()).resolve(metadata, registry_resolver=empty_resolver())

    assert resolved.format is ModelFormat.PYTORCH_BIN
    assert resolved.relationship is SourceRelationship.ORIGINAL


# --------------------------------------------------------------------------- #
# 4. The reference fixture: a declared derivative is offered, never taken
# --------------------------------------------------------------------------- #


def test_reference_fixture_offers_the_declared_derivative_for_confirmation(
    gguf_only: ModelMetadata,
    safetensors_derivative: ModelMetadata,
) -> None:
    """SPEC 20: the Qwen3.5 fixture resolves to an offer, not to a substitution.

    Every fact asserted here — both repositories, both SHAs, the declared
    ``lossless_conversion`` relationship and the model-card evidence URL — was
    verified against the real Hub and the shipped registry.
    """
    hub = FakeHub({REQUESTED_REPO: gguf_only, DERIVATIVE_REPO: safetensors_derivative})
    sink = CollectingSink()

    resolution = SourceResolver(hub, progress=sink).resolve_detailed(
        gguf_only, registry_resolver=builtin_resolver()
    )

    resolved = resolution.resolved
    assert resolved.source.repo_id == DERIVATIVE_REPO
    assert resolved.source.revision == DERIVATIVE_SHA
    assert resolved.requested.repo_id == REQUESTED_REPO
    assert resolved.relationship is SourceRelationship.LOSSLESS_CONVERSION
    assert resolved.format is ModelFormat.SAFETENSORS
    assert resolved.already_openvino is False
    assert resolved.is_derivative is True

    # It is an offer: confirmation is required and the reason says why.
    assert resolved.requires_confirmation is True
    assert resolution.confirmation_codes() == [CONFIRMATION_THIRD_PARTY_DERIVATIVE]
    reason = resolution.confirmations[0]
    assert "third-party derivative" in reason.message
    assert DERIVATIVE_REPO in reason.message
    assert reason.security_sensitive is False

    # The evidence carries the model-card URL the registry declared.
    assert [item.url for item in resolved.evidence] == [DERIVATIVE_CARD_URL]
    assert resolved.rule_id == "hauhaucs-qwen35-safetensors"
    assert resolution.lineage_warnings
    assert any("needs approval" in event.message for event in sink.events)
    assert hub.downloaded == []


def test_derivative_is_never_auto_selected_under_suggest(
    gguf_only: ModelMetadata,
    safetensors_derivative: ModelMetadata,
) -> None:
    """The plain ``resolve`` API cannot hand back a silently chosen derivative."""
    hub = FakeHub({REQUESTED_REPO: gguf_only, DERIVATIVE_REPO: safetensors_derivative})

    resolved = SourceResolver(hub).resolve(gguf_only, registry_resolver=builtin_resolver())

    assert resolved.is_derivative is True
    assert resolved.requires_confirmation is True


def test_auto_if_trusted_derivative_needs_no_confirmation(
    gguf_only: ModelMetadata,
    safetensors_derivative: ModelMetadata,
) -> None:
    """Only a registry trust policy of ``auto_if_trusted`` skips the prompt."""
    registry = resolver_for(
        f"""
        schema_version: 1
        derivatives:
          - id: trusted-mirror
            requested_repo: {REQUESTED_REPO}
            candidates:
              - repo_id: {DERIVATIVE_REPO}
                relationship: lossless_conversion
                trust: auto_if_trusted
        """
    )
    hub = FakeHub({REQUESTED_REPO: gguf_only, DERIVATIVE_REPO: safetensors_derivative})

    resolution = SourceResolver(hub).resolve_detailed(gguf_only, registry_resolver=registry)

    assert resolution.resolved.requires_confirmation is False
    assert resolution.confirmations == []
    assert resolution.lineage_warnings


def test_policy_can_withdraw_registry_auto_trust(
    gguf_only: ModelMetadata,
    safetensors_derivative: ModelMetadata,
) -> None:
    """``honour_registry_auto_trust=False`` downgrades a trusted candidate."""
    registry = resolver_for(
        f"""
        schema_version: 1
        derivatives:
          - id: trusted-mirror
            requested_repo: {REQUESTED_REPO}
            candidates:
              - repo_id: {DERIVATIVE_REPO}
                relationship: lossless_conversion
                trust: auto_if_trusted
        """
    )
    hub = FakeHub({REQUESTED_REPO: gguf_only, DERIVATIVE_REPO: safetensors_derivative})

    resolution = SourceResolver(hub).resolve_detailed(
        gguf_only,
        registry_resolver=registry,
        policy=SourcePolicy(honour_registry_auto_trust=False),
    )

    assert resolution.resolved.requires_confirmation is True
    assert resolution.confirmation_codes() == [CONFIRMATION_THIRD_PARTY_DERIVATIVE]


def test_trusted_candidate_is_preferred_over_a_suggestion(
    gguf_only: ModelMetadata,
    safetensors_derivative: ModelMetadata,
) -> None:
    """Declaration order does not beat trust order."""
    mirror = build_metadata("acme/mirror", SAFETENSORS_FILES, resolved_revision=OTHER_SHA)
    registry = resolver_for(
        f"""
        schema_version: 1
        derivatives:
          - id: mixed-trust
            requested_repo: {REQUESTED_REPO}
            candidates:
              - repo_id: {DERIVATIVE_REPO}
                relationship: lossless_conversion
                trust: suggest
              - repo_id: acme/mirror
                relationship: converted
                trust: auto_if_trusted
        """
    )
    hub = FakeHub(
        {
            REQUESTED_REPO: gguf_only,
            DERIVATIVE_REPO: safetensors_derivative,
            "acme/mirror": mirror,
        }
    )

    resolution = SourceResolver(hub).resolve_detailed(gguf_only, registry_resolver=registry)

    assert resolution.resolved.source.repo_id == "acme/mirror"
    assert resolution.resolved.requires_confirmation is False


def test_never_auto_candidate_is_informational_only(gguf_only: ModelMetadata) -> None:
    """``trust: never_auto`` is reported but never selected, so resolution stops."""
    registry = resolver_for(
        f"""
        schema_version: 1
        derivatives:
          - id: manual-only
            requested_repo: {REQUESTED_REPO}
            candidates:
              - repo_id: {DERIVATIVE_REPO}
                relationship: converted
                trust: never_auto
        """
    )
    hub = FakeHub({REQUESTED_REPO: gguf_only})

    with pytest.raises(SourceResolutionError) as caught:
        SourceResolver(hub).resolve(gguf_only, registry_resolver=registry)

    remediation = " ".join(caught.value.diagnostic.remediation)
    assert "never_auto" in remediation
    assert f"--source {DERIVATIVE_REPO}" in remediation
    # The candidate repository was never even inspected.
    assert hub.inspected == []


def test_derivative_without_readable_weights_is_rejected(gguf_only: ModelMetadata) -> None:
    """A declared derivative that is itself GGUF-only cannot stand in."""
    mirror = build_metadata("acme/gguf-mirror", GGUF_FILES, resolved_revision=OTHER_SHA)
    registry = resolver_for(
        f"""
        schema_version: 1
        derivatives:
          - id: useless-mirror
            requested_repo: {REQUESTED_REPO}
            candidates:
              - repo_id: acme/gguf-mirror
                relationship: converted
                trust: suggest
        """
    )
    hub = FakeHub({REQUESTED_REPO: gguf_only, "acme/gguf-mirror": mirror})

    with pytest.raises(SourceResolutionError) as caught:
        SourceResolver(hub).resolve(gguf_only, registry_resolver=registry)

    assert caught.value.diagnostic.code == CODE_GGUF_UNSUPPORTED
    assert any(
        "publishes no weights the OpenVINO exporter can read" in line
        for line in caught.value.diagnostic.remediation
    )


def test_unreadable_derivative_does_not_abort_resolution(
    gguf_only: ModelMetadata,
    safetensors_derivative: ModelMetadata,
) -> None:
    """A candidate the Hub cannot serve is skipped, not fatal."""
    registry = resolver_for(
        f"""
        schema_version: 1
        derivatives:
          - id: two-candidates
            requested_repo: {REQUESTED_REPO}
            candidates:
              - repo_id: acme/missing
                relationship: converted
                trust: suggest
              - repo_id: {DERIVATIVE_REPO}
                relationship: lossless_conversion
                trust: suggest
        """
    )
    hub = FakeHub({REQUESTED_REPO: gguf_only, DERIVATIVE_REPO: safetensors_derivative})

    resolution = SourceResolver(hub).resolve_detailed(gguf_only, registry_resolver=registry)

    assert resolution.resolved.source.repo_id == DERIVATIVE_REPO
    assert any("acme/missing" in note for note in resolution.notes)


def test_policy_can_disable_derivatives_entirely(
    gguf_only: ModelMetadata,
    safetensors_derivative: ModelMetadata,
) -> None:
    """``allow_derivatives=False`` stops rather than substituting."""
    hub = FakeHub({REQUESTED_REPO: gguf_only, DERIVATIVE_REPO: safetensors_derivative})

    with pytest.raises(SourceResolutionError):
        SourceResolver(hub).resolve(
            gguf_only,
            registry_resolver=builtin_resolver(),
            policy=SourcePolicy(allow_derivatives=False),
        )


# --------------------------------------------------------------------------- #
# 5. GGUF is never assumed to be readable
# --------------------------------------------------------------------------- #


def test_gguf_only_without_derivative_stops_with_source_format_unsupported(
    gguf_only: ModelMetadata,
) -> None:
    """GGUF presence is not evidence that OpenVINO GenAI can consume it."""
    hub = FakeHub({REQUESTED_REPO: gguf_only})

    with pytest.raises(SourceResolutionError) as caught:
        SourceResolver(hub).resolve(gguf_only, registry_resolver=empty_resolver())

    failure = caught.value.diagnostic
    assert failure.category is FailureCategory.SOURCE_FORMAT_UNSUPPORTED
    assert failure.code == CODE_GGUF_UNSUPPORTED
    assert "cannot be imported directly" in failure.message
    assert "not evidence" in failure.message
    assert any("--source" in line for line in failure.remediation)


# --------------------------------------------------------------------------- #
# 6. Failures name the useful choices
# --------------------------------------------------------------------------- #


def test_repository_without_weights_fails_with_choices() -> None:
    """A repository with nothing convertible produces an actionable failure."""
    metadata = build_metadata(
        "acme/card-only",
        (("README.md", 900), ("config.json", 300)),
        resolved_revision=OTHER_SHA,
    )

    with pytest.raises(SourceResolutionError) as caught:
        SourceResolver(FakeHub()).resolve(metadata, registry_resolver=empty_resolver())

    failure = caught.value.diagnostic
    assert failure.code == CODE_NO_USABLE_SOURCE
    assert failure.category is FailureCategory.SOURCE_FORMAT_UNSUPPORTED
    assert any("will not guess one from repository names" in line for line in failure.remediation)


# --------------------------------------------------------------------------- #
# Explicit --source
# --------------------------------------------------------------------------- #


def test_explicit_source_is_used_after_a_lineage_warning(
    gguf_only: ModelMetadata,
    safetensors_derivative: ModelMetadata,
) -> None:
    """``--source`` is honoured, recorded, and satisfies the derivative question."""
    hub = FakeHub({REQUESTED_REPO: gguf_only, DERIVATIVE_REPO: safetensors_derivative})

    resolution = SourceResolver(hub).resolve_detailed(
        gguf_only,
        explicit_source=DERIVATIVE_REPO,
        registry_resolver=builtin_resolver(),
    )

    assert resolution.explicit is True
    assert resolution.resolved.source.repo_id == DERIVATIVE_REPO
    assert resolution.resolved.source.revision == DERIVATIVE_SHA
    assert resolution.resolved.format is ModelFormat.SAFETENSORS
    assert resolution.lineage_warnings
    assert resolution.confirmation_codes() == [CONFIRMATION_EXPLICIT_SOURCE]
    assert CONFIRMATION_THIRD_PARTY_DERIVATIVE not in resolution.confirmation_codes()
    assert resolution.derivative_confirmation_satisfied is True
    # The registry still supplies the lineage it declared.
    assert resolution.resolved.relationship is SourceRelationship.LOSSLESS_CONVERSION


def test_explicit_source_without_declared_lineage_says_so(
    gguf_only: ModelMetadata,
) -> None:
    """An undeclared substitution is honoured but not dressed up as lineage."""
    stranger = build_metadata("acme/unrelated", SAFETENSORS_FILES, resolved_revision=OTHER_SHA)
    hub = FakeHub({REQUESTED_REPO: gguf_only, "acme/unrelated": stranger})

    resolution = SourceResolver(hub).resolve_detailed(
        gguf_only,
        explicit_source=ModelIdentity(repo_id="acme/unrelated"),
        registry_resolver=builtin_resolver(),
    )

    assert resolution.resolved.relationship is SourceRelationship.ORIGINAL
    assert resolution.resolved.is_derivative is True
    assert resolution.resolved.rule_id is None
    assert any("declares no lineage" in warning for warning in resolution.lineage_warnings)
    assert any("never infers" in warning for warning in resolution.lineage_warnings)


def test_explicit_source_equal_to_the_requested_repo_needs_no_confirmation(
    safetensors_derivative: ModelMetadata,
) -> None:
    """Naming the repository you already asked for is not a substitution."""
    hub = FakeHub({DERIVATIVE_REPO: safetensors_derivative})

    resolution = SourceResolver(hub).resolve_detailed(
        safetensors_derivative,
        explicit_source=DERIVATIVE_REPO,
        registry_resolver=empty_resolver(),
    )

    assert resolution.confirmations == []
    assert resolution.resolved.requires_confirmation is False
    assert resolution.lineage_warnings
    # The requested repository's metadata is reused rather than fetched again.
    assert hub.inspected == []


def test_explicit_source_cannot_buy_gguf_support(gguf_only: ModelMetadata) -> None:
    """``--source`` overrides the preference order, never the backend's abilities."""
    mirror = build_metadata("acme/gguf-mirror", GGUF_FILES, resolved_revision=OTHER_SHA)
    hub = FakeHub({REQUESTED_REPO: gguf_only, "acme/gguf-mirror": mirror})

    with pytest.raises(SourceResolutionError) as caught:
        SourceResolver(hub).resolve(
            gguf_only,
            explicit_source="acme/gguf-mirror",
            registry_resolver=empty_resolver(),
        )

    assert caught.value.diagnostic.code == CODE_GGUF_UNSUPPORTED


def test_explicit_source_accepts_a_repo_at_revision_string(
    gguf_only: ModelMetadata,
    safetensors_derivative: ModelMetadata,
) -> None:
    """``--source repo@sha`` is parsed and pinned to that commit."""
    pinned = safetensors_derivative.model_copy(update={"resolved_revision": None})
    hub = FakeHub({REQUESTED_REPO: gguf_only, DERIVATIVE_REPO: pinned})

    resolved = SourceResolver(hub).resolve(
        gguf_only,
        explicit_source=f"{DERIVATIVE_REPO}@{DERIVATIVE_SHA}",
        registry_resolver=empty_resolver(),
    )

    assert resolved.source.revision == DERIVATIVE_SHA
    assert hub.resolved == []


# --------------------------------------------------------------------------- #
# Revision pinning
# --------------------------------------------------------------------------- #


def test_a_branch_name_is_always_replaced_by_the_resolved_sha() -> None:
    """The resolved source never carries a mutable reference."""
    metadata = build_metadata(
        DERIVATIVE_REPO,
        SAFETENSORS_FILES,
        revision="main",
        resolved_revision="main",
    )
    hub = FakeHub({DERIVATIVE_REPO: metadata}, revisions={DERIVATIVE_REPO: DERIVATIVE_SHA})

    resolved = SourceResolver(hub).resolve(metadata, registry_resolver=empty_resolver())

    assert resolved.source.revision == DERIVATIVE_SHA
    assert resolved.requested.revision == "main"
    assert resolved.metadata is not None
    assert resolved.metadata.resolved_revision == DERIVATIVE_SHA
    assert hub.resolved == [DERIVATIVE_REPO]


def test_derivative_branch_pin_is_resolved_too(
    gguf_only: ModelMetadata,
    safetensors_derivative: ModelMetadata,
) -> None:
    """A registry candidate pinned to a branch is still resolved to a commit."""
    unpinned = safetensors_derivative.model_copy(update={"resolved_revision": None})
    registry = resolver_for(
        f"""
        schema_version: 1
        derivatives:
          - id: branch-pin
            requested_repo: {REQUESTED_REPO}
            candidates:
              - repo_id: {DERIVATIVE_REPO}
                relationship: lossless_conversion
                trust: suggest
                revision: refs/pr/1
        """
    )
    hub = FakeHub(
        {REQUESTED_REPO: gguf_only, DERIVATIVE_REPO: unpinned},
        revisions={DERIVATIVE_REPO: DERIVATIVE_SHA},
    )

    resolved = SourceResolver(hub).resolve(gguf_only, registry_resolver=registry)

    assert resolved.source.revision == DERIVATIVE_SHA


def test_an_unpinnable_revision_is_a_typed_failure() -> None:
    """Resolution refuses to continue from a reference it cannot pin."""

    class BranchOnlyHub(FakeHub):
        def resolve_revision(self, model: ModelIdentity) -> str:
            self.resolved.append(model.repo_id)
            return "main"

    metadata = build_metadata(DERIVATIVE_REPO, SAFETENSORS_FILES, resolved_revision=None)
    hub = BranchOnlyHub({DERIVATIVE_REPO: metadata})

    with pytest.raises(SourceResolutionError) as caught:
        SourceResolver(hub).resolve(metadata, registry_resolver=empty_resolver())

    assert caught.value.diagnostic.code == CODE_REVISION_UNPINNED
    assert "not be reproducible" in caught.value.diagnostic.message


# --------------------------------------------------------------------------- #
# Only declared lineage counts
# --------------------------------------------------------------------------- #


def test_a_similarly_named_repository_is_never_matched(gguf_only: ModelMetadata) -> None:
    """A near-identical repo id declared for *another* model must not be used."""
    lookalike = build_metadata(
        "nDimensional/Qwen3.5-9B-Uncensored-Safetensor",
        SAFETENSORS_FILES,
        resolved_revision=OTHER_SHA,
    )
    registry = resolver_for(
        """
        schema_version: 1
        derivatives:
          - id: for-another-model
            requested_repo: SomeoneElse/Qwen3.5-9B-Uncensored-HauhauCS-Aggressiv
            candidates:
              - repo_id: nDimensional/Qwen3.5-9B-Uncensored-Safetensor
                relationship: lossless_conversion
                trust: auto_if_trusted
        """
    )
    hub = FakeHub(
        {
            REQUESTED_REPO: gguf_only,
            "nDimensional/Qwen3.5-9B-Uncensored-Safetensor": lookalike,
        }
    )

    with pytest.raises(SourceResolutionError):
        SourceResolver(hub).resolve(gguf_only, registry_resolver=registry)

    assert hub.inspected == []


def test_resolver_contains_no_name_similarity_machinery() -> None:
    """A regression guard: fuzzy matching must never appear in this module."""
    source = Path(resolve_module.__file__).read_text(encoding="utf-8")
    for banned in (
        "difflib",
        "SequenceMatcher",
        "get_close_matches",
        "rapidfuzz",
        "fuzzywuzzy",
        "levenshtein",
    ):
        assert banned not in source


# --------------------------------------------------------------------------- #
# Source options and layering
# --------------------------------------------------------------------------- #


def test_source_options_describe_the_gguf_fixture_without_any_io(
    gguf_only: ModelMetadata,
) -> None:
    """``inspect`` can print the routes from metadata the Hub already returned."""
    candidates, _explanation = builtin_resolver().resolve_derivatives(gguf_only.identity)
    options = {option.label: option for option in describe_source_options(gguf_only, candidates)}

    assert options[OPTION_OPENVINO].available is False
    assert options[OPTION_SAFETENSORS].available is False
    assert options[OPTION_DERIVATIVE].available is True
    assert options[OPTION_DERIVATIVE].repo_id == DERIVATIVE_REPO
    assert options[OPTION_DERIVATIVE].requires_confirmation is True
    assert options[OPTION_GGUF].available is False
    detail = options[OPTION_GGUF].detail or ""
    assert "not evidence" in detail


def test_source_options_report_a_missing_derivative_declaration() -> None:
    """No declaration is said out loud, so nothing looks like a hidden option."""
    metadata = build_metadata("acme/card-only", (("README.md", 900),), resolved_revision=OTHER_SHA)

    options = {option.label: option for option in describe_source_options(metadata)}

    assert options[OPTION_DERIVATIVE].available is False
    assert options[OPTION_DERIVATIVE].detail == (
        "No derivative source is declared for this repository."
    )


def test_source_options_render_through_the_ui_view_model(gguf_only: ModelMetadata) -> None:
    """The application option maps field-for-field onto the UI's view model."""
    from yonga.ui.tables import SourceOption as UiSourceOption

    candidates, _explanation = builtin_resolver().resolve_derivatives(gguf_only.identity)
    for option in describe_source_options(gguf_only, candidates):
        rendered = UiSourceOption.model_validate(option.model_dump())
        assert rendered.label == option.label
        assert rendered.requires_confirmation == option.requires_confirmation


def test_resolver_options_helper_uses_the_registry(gguf_only: ModelMetadata) -> None:
    """``SourceResolver.options`` looks the derivatives up itself."""
    options = SourceResolver(FakeHub()).options(gguf_only, registry_resolver=builtin_resolver())

    assert any(option.repo_id == DERIVATIVE_REPO for option in options)


def test_fake_hub_satisfies_the_hub_client_protocol() -> None:
    """The fixtures exercise the same port the CLI injects."""
    assert isinstance(FakeHub(), HubClient)


def test_resolution_is_json_serialisable(
    gguf_only: ModelMetadata,
    safetensors_derivative: ModelMetadata,
) -> None:
    """``--json`` can print the whole resolution."""
    hub = FakeHub({REQUESTED_REPO: gguf_only, DERIVATIVE_REPO: safetensors_derivative})

    resolution = SourceResolver(hub).resolve_detailed(
        gguf_only, registry_resolver=builtin_resolver()
    )

    payload = json.loads(json.dumps(resolution.to_json()))
    assert payload["resolved"]["source"]["revision"] == DERIVATIVE_SHA
    assert payload["confirmations"][0]["code"] == CONFIRMATION_THIRD_PARTY_DERIVATIVE


FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "rich",
        "typer",
        "prompt_toolkit",
        "openvino",
        "openvino_genai",
        "openvino_tokenizers",
        "huggingface_hub",
        "optimum",
        "transformers",
        "torch",
        "subprocess",
    }
)


def imported_roots(path: Path) -> set[str]:
    """Return the top-level module names a source file imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_resolver_imports_no_adapter_library() -> None:
    """The application layer stays free of renderers, Hub and inference stacks."""
    roots = imported_roots(Path(resolve_module.__file__))
    assert roots & FORBIDDEN_IMPORT_ROOTS == set()


def test_cancellation_is_not_swallowed(gguf_only: ModelMetadata) -> None:
    """A user interrupt during candidate inspection propagates unchanged."""
    from yonga.domain.diagnostics import CancelledError

    class CancellingHub(FakeHub):
        def inspect(self, model: ModelIdentity, *, with_sizes: bool = True) -> ModelMetadata:
            raise CancelledError("interrupted")

    registry = resolver_for(
        f"""
        schema_version: 1
        derivatives:
          - id: one
            requested_repo: {REQUESTED_REPO}
            candidates:
              - repo_id: {DERIVATIVE_REPO}
                relationship: converted
                trust: suggest
        """
    )

    with pytest.raises(CancelledError):
        SourceResolver(CancellingHub()).resolve(gguf_only, registry_resolver=registry)


def test_hub_failures_for_a_candidate_are_typed(gguf_only: ModelMetadata) -> None:
    """A missing candidate surfaces as a source-resolution failure, not a crash."""
    registry = resolver_for(
        f"""
        schema_version: 1
        derivatives:
          - id: one
            requested_repo: {REQUESTED_REPO}
            candidates:
              - repo_id: acme/missing
                relationship: converted
                trust: suggest
        """
    )

    with pytest.raises(YongaError):
        SourceResolver(FakeHub()).resolve(gguf_only, registry_resolver=registry)
