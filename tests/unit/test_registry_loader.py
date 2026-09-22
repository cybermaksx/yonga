"""Loading, layering and validation tests for the compatibility registry."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from yonga.config.paths import Paths
from yonga.domain.diagnostics import FailureCategory, RegistryError
from yonga.registry.loader import (
    BUILTIN_FILENAMES,
    KNOWN_REPAIR_IMPLEMENTATIONS,
    builtin_registry_dir,
    load_builtin_documents,
    load_builtin_registry,
    load_registry,
    load_registry_from_documents,
    parse_document,
    registry_override_dirs,
    validate_registry,
)
from yonga.registry.schema import Registry

PROFILE_DOCUMENT = textwrap.dedent(
    """
    schema_version: 1
    profiles:
      - id: demo-profile
        device: cpu
        summary: demo
        quantization:
          weight_format: int4
    """
)


def write(directory: Path, name: str, text: str) -> Path:
    """Write a registry document into ``directory`` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def paths_for(root: Path) -> Paths:
    """Build a Paths object rooted at a temporary directory."""
    return Paths(config_dir=root / "config", cache_dir=root / "cache", data_dir=root / "data")


# ---------------------------------------------------------------------------
# built-in data
# ---------------------------------------------------------------------------


def test_builtin_registry_loads_and_is_indexed():
    registry = load_builtin_registry()
    assert isinstance(registry, Registry)
    counts = registry.counts()
    assert counts["architectures"] >= 2
    assert counts["profiles"] >= 4
    assert counts["issues"] >= 4
    assert counts["repairs"] >= 1
    assert counts["derivatives"] >= 1
    assert counts["chat"] >= 2
    assert registry.architecture("arch-qwen3_5") is not None
    assert registry.profile("gpu-int4-asym-g128") is not None
    assert registry.repair("ov-tokenizer-precision-string-casing") is not None
    assert registry.chat_profile("chat-qwen3_5") is not None
    assert registry.layers == ["builtin"]


def test_every_builtin_file_is_readable_as_package_data():
    documents = load_builtin_documents()
    assert len(documents) == len(BUILTIN_FILENAMES)
    for document, name in zip(documents, BUILTIN_FILENAMES, strict=True):
        assert document.origin == f"builtin:{name}"
    directory = builtin_registry_dir()
    assert directory is not None
    for name in BUILTIN_FILENAMES:
        assert (directory / name).is_file()


def test_shipped_registry_has_no_validation_problems():
    assert validate_registry(load_builtin_registry()) == []


def test_every_builtin_rule_records_its_origin():
    registry = load_builtin_registry()
    for rule in registry.all_rules():
        origin = registry.origin_of(rule.id)
        assert origin is not None
        assert origin.startswith("builtin:")


def test_builtin_repair_implementations_are_known():
    registry = load_builtin_registry()
    for repair in registry.repairs:
        assert repair.implementation in KNOWN_REPAIR_IMPLEMENTATIONS


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_parse_document_rejects_unparsable_yaml():
    with pytest.raises(RegistryError) as excinfo:
        parse_document("profiles: [\n  - id: broken\n", origin="demo.yaml")
    diagnostic = excinfo.value.diagnostic
    assert diagnostic.code == "registry.yaml_invalid"
    assert diagnostic.category is FailureCategory.REGISTRY_ERROR
    assert "demo.yaml" in diagnostic.message


def test_parse_document_rejects_schema_violation():
    with pytest.raises(RegistryError) as excinfo:
        parse_document("profiles:\n  - id: broken\n", origin="demo.yaml")
    assert excinfo.value.diagnostic.code == "registry.schema_violation"


def test_parse_document_rejects_non_mapping_document():
    with pytest.raises(RegistryError) as excinfo:
        parse_document("- just\n- a list\n", origin="demo.yaml")
    assert excinfo.value.diagnostic.code == "registry.document_invalid"


def test_parse_document_rejects_a_newer_schema_version():
    with pytest.raises(RegistryError) as excinfo:
        parse_document("schema_version: 99\n", origin="demo.yaml")
    assert excinfo.value.diagnostic.code == "registry.schema_version_unsupported"


def test_empty_document_is_allowed():
    document = parse_document("", origin="demo.yaml")
    assert document.rules() == []


def test_registry_data_can_never_construct_python_objects():
    payload = "profiles: !!python/object/apply:os.system ['echo unsafe']\n"
    with pytest.raises(RegistryError) as excinfo:
        parse_document(payload, origin="evil.yaml")
    assert excinfo.value.diagnostic.code == "registry.yaml_invalid"


# ---------------------------------------------------------------------------
# layering
# ---------------------------------------------------------------------------


def test_duplicate_id_within_one_layer_is_rejected(tmp_path):
    overrides = tmp_path / "config" / "registry-overrides"
    write(overrides, "a.yaml", PROFILE_DOCUMENT)
    write(overrides, "b.yaml", PROFILE_DOCUMENT)
    with pytest.raises(RegistryError) as excinfo:
        load_registry(paths_for(tmp_path))
    diagnostic = excinfo.value.diagnostic
    assert diagnostic.code == "registry.duplicate_id"
    assert diagnostic.category is FailureCategory.REGISTRY_ERROR
    assert "demo-profile" in diagnostic.message


def test_duplicate_id_inside_one_document_is_rejected():
    document = parse_document(
        textwrap.dedent(
            """
            schema_version: 1
            profiles:
              - id: demo-profile
                device: cpu
              - id: demo-profile
                device: gpu
            """
        ),
        origin="demo.yaml",
    )
    with pytest.raises(RegistryError) as excinfo:
        load_registry_from_documents([document])
    assert excinfo.value.diagnostic.code == "registry.duplicate_id"


def test_override_replaces_an_existing_rule_and_adds_new_ones(tmp_path):
    overrides = tmp_path / "config" / "registry-overrides"
    write(
        overrides,
        "local.yaml",
        """
        schema_version: 1
        profiles:
          - id: gpu-int4-asym-g128
            device: gpu
            summary: locally tuned GPU profile
            quantization:
              weight_format: int4
              group_size: 64
          - id: gpu-int8-local
            device: gpu
            summary: brand new local profile
            quantization:
              weight_format: int8
        """,
    )
    builtin = load_builtin_registry()
    merged = load_registry(paths_for(tmp_path))
    replaced = merged.profile("gpu-int4-asym-g128")
    assert replaced is not None
    assert replaced.quantization.group_size == 64
    assert replaced.summary == "locally tuned GPU profile"
    assert merged.profile("gpu-int8-local") is not None
    assert len(merged.profiles) == len(builtin.profiles) + 1
    assert merged.origin_of("gpu-int4-asym-g128") == str(overrides / "local.yaml")
    assert merged.layers == ["builtin", str(overrides)]


def test_override_may_not_change_the_kind_of_a_rule(tmp_path):
    overrides = tmp_path / "config" / "registry-overrides"
    write(
        overrides,
        "local.yaml",
        """
        schema_version: 1
        profiles:
          - id: arch-qwen3_5
            device: gpu
            quantization:
              weight_format: int4
        """,
    )
    with pytest.raises(RegistryError) as excinfo:
        load_registry(paths_for(tmp_path))
    assert excinfo.value.diagnostic.code == "registry.kind_conflict"


def test_extra_dirs_are_layered_after_the_configured_directory(tmp_path):
    first = tmp_path / "config" / "registry-overrides"
    second = tmp_path / "extra"
    write(first, "a.yaml", PROFILE_DOCUMENT)
    write(
        second,
        "b.yaml",
        """
        schema_version: 1
        profiles:
          - id: demo-profile
            device: npu
            summary: from the extra directory
            quantization:
              weight_format: int4
        """,
    )
    merged = load_registry(paths_for(tmp_path), [second])
    profile = merged.profile("demo-profile")
    assert profile is not None
    assert profile.device == "NPU"
    assert merged.layers == ["builtin", str(first), str(second)]


def test_missing_override_directory_is_ignored(tmp_path):
    merged = load_registry(paths_for(tmp_path))
    assert merged.counts() == load_builtin_registry().counts()
    assert merged.layers == ["builtin"]


def test_registry_override_dirs_deduplicates_and_keeps_order(tmp_path):
    paths = paths_for(tmp_path)
    extra = tmp_path / "extra"
    ordered = registry_override_dirs(paths, [extra, paths.registry_overrides_dir])
    assert ordered == [paths.registry_overrides_dir, extra]
    assert registry_override_dirs(None, []) == []


def test_non_yaml_files_in_an_override_directory_are_ignored(tmp_path):
    overrides = tmp_path / "config" / "registry-overrides"
    write(overrides, "notes.txt", "this is not a registry file")
    write(overrides, "a.yml", PROFILE_DOCUMENT)
    merged = load_registry(paths_for(tmp_path))
    assert merged.profile("demo-profile") is not None


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def registry_from(text: str) -> Registry:
    """Build a registry from one inline document."""
    return load_registry_from_documents([parse_document(textwrap.dedent(text), origin="demo.yaml")])


def test_validate_reports_unknown_repair_implementation():
    registry = registry_from(
        """
        repairs:
          - id: bad-repair
            repair:
              implementation: rm_minus_rf
        """
    )
    problems = validate_registry(registry)
    assert any("unknown repair implementation" in item for item in problems)


def test_validate_accepts_injected_implementation_names():
    registry = registry_from(
        """
        repairs:
          - id: future-repair
            repair:
              implementation: some_future_repair
        """
    )
    assert validate_registry(registry, known_implementations={"some_future_repair"}) == []


def test_validate_reports_unparsable_version_specifiers():
    registry = registry_from(
        """
        profiles:
          - id: broken-profile
            device: cpu
            constraints:
              openvino: "nonsense!!"
        """
    )
    assert any("unparsable version specifier" in item for item in validate_registry(registry))


def test_validate_reports_unparsable_device_constraints():
    registry = registry_from(
        """
        architectures:
          - id: arch-broken
            devices:
              npu:
                status: unsupported
                evidence: tested
                constraints:
                  openvino: "=="
        """
    )
    assert any("device NPU" in item for item in validate_registry(registry))


def test_validate_reports_dangling_profile_reference():
    registry = registry_from(
        """
        architectures:
          - id: arch-dangling
            profiles:
              gpu: does-not-exist
        """
    )
    assert any("unknown profile" in item for item in validate_registry(registry))


def test_validate_reports_profile_pointing_at_the_wrong_device():
    registry = registry_from(
        """
        architectures:
          - id: arch-mismatch
            profiles:
              gpu: cpu-only
        profiles:
          - id: cpu-only
            device: cpu
        """
    )
    assert any("which targets CPU" in item for item in validate_registry(registry))


def test_validate_reports_dangling_chat_and_repair_references():
    registry = registry_from(
        """
        architectures:
          - id: arch-chat
            chat_profile: nope
        issues:
          - id: issue-repair
            match:
              exception_contains: ["boom"]
            classification:
              code: x.y
              category: unknown_runtime_error
            recommendation:
              repair: also-nope
        """
    )
    problems = validate_registry(registry)
    assert any("unknown chat profile" in item for item in problems)
    assert any("unknown repair" in item for item in problems)


def test_validate_reports_unknown_failure_category():
    registry = registry_from(
        """
        issues:
          - id: issue-bad-category
            match:
              exception_contains: ["boom"]
            classification:
              code: x.y
              category: not_a_real_category
        """
    )
    assert any("unknown failure category" in item for item in validate_registry(registry))


def test_validate_reports_recommended_device_without_support_entry():
    registry = registry_from(
        """
        architectures:
          - id: arch-recommend
            devices:
              cpu:
                status: supported
                evidence: official
            recommended_device: npu
        """
    )
    assert any("recommended_device" in item for item in validate_registry(registry))


def test_validate_detects_same_specificity_same_priority_conflicts():
    registry = registry_from(
        """
        profiles:
          - id: gpu-a
            device: gpu
            priority: 100
          - id: gpu-b
            device: gpu
            priority: 100
        """
    )
    problems = validate_registry(registry)
    assert any("'gpu-a' and 'gpu-b'" in item for item in problems)
    assert any("resolution would be ambiguous" in item for item in problems)


def test_validate_accepts_distinct_priorities():
    registry = registry_from(
        """
        profiles:
          - id: gpu-a
            device: gpu
            priority: 100
          - id: gpu-b
            device: gpu
            priority: 90
        """
    )
    assert validate_registry(registry) == []


def test_validate_does_not_flag_rules_for_different_devices():
    registry = registry_from(
        """
        profiles:
          - id: gpu-a
            device: gpu
            priority: 100
          - id: cpu-a
            device: cpu
            priority: 100
        """
    )
    assert validate_registry(registry) == []


def test_validate_does_not_flag_issue_rules_with_disjoint_signatures():
    registry = registry_from(
        """
        issues:
          - id: issue-a
            priority: 100
            match:
              device: npu
              exception_contains: ["alpha failure"]
            classification:
              code: a.b
              category: compiler_unsupported_graph
          - id: issue-b
            priority: 100
            match:
              device: npu
              exception_contains: ["beta failure"]
            classification:
              code: c.d
              category: compiler_unsupported_graph
        """
    )
    assert validate_registry(registry) == []


def test_validate_flags_issue_rules_with_overlapping_signatures():
    registry = registry_from(
        """
        issues:
          - id: issue-a
            priority: 100
            match:
              device: npu
              exception_contains: ["shared failure text"]
            classification:
              code: a.b
              category: compiler_unsupported_graph
          - id: issue-b
            priority: 100
            match:
              device: npu
              exception_contains: ["shared failure"]
            classification:
              code: c.d
              category: compiler_unsupported_graph
        """
    )
    assert any("issue rules" in item for item in validate_registry(registry))


def test_validate_ignores_deprecated_rules_in_conflict_detection():
    registry = registry_from(
        """
        profiles:
          - id: gpu-a
            device: gpu
            priority: 100
          - id: gpu-b
            device: gpu
            priority: 100
            status: deprecated
            summary: replaced by gpu-a
        """
    )
    assert validate_registry(registry) == []
