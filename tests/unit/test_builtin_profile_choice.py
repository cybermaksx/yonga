"""What the shipped registry chooses, and the data invariants that keep that choice honest.

The size bands live in YAML, so nothing but these tests stops an edit from
silently moving every small model to another quantization, or from changing what
an existing profile id means. Reuse is keyed by revision and profile id alone
(``docs/MODEL_PIPELINE.md`` section 15): an id whose quantization changed would
hand back artifacts built with the old settings as if they were the new ones.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yonga.domain.models import ModelIdentity, ModelMetadata
from yonga.registry import RegistryResolver, load_builtin_registry
from yonga.registry.loader import validate_registry
from yonga.registry.schema import ArchitectureRule, ModelRule

BUILTIN_DIR = Path(__file__).resolve().parents[2] / "src" / "yonga" / "registry" / "builtin"


@pytest.fixture(scope="module")
def resolver() -> RegistryResolver:
    return RegistryResolver(load_builtin_registry())


def text_model(
    model_type: str, architecture: str, count: int | None, **extra: object
) -> ModelMetadata:
    return ModelMetadata(
        identity=ModelIdentity(repo_id=f"org/{model_type}-{count}"),
        model_type=model_type,
        architectures=[architecture],
        pipeline_tag="text-generation",
        parameter_count=count,
        parameter_dtypes={"BF16": count} if count else {},
        **extra,  # type: ignore[arg-type]
    )


def chosen(resolver: RegistryResolver, meta: ModelMetadata, device: str) -> str:
    arch_rule, _ = resolver.resolve_architecture(meta)
    profile, _ = resolver.resolve_profile(device=device, arch_rule=arch_rule, meta=meta)
    return profile.id


def test_the_builtin_registry_validates_clean() -> None:
    assert validate_registry(load_builtin_registry()) == []


#: Every (architecture rule, device) -> profile pin the built-in data makes. A
#: pin overrides the size bands, so adding one must be a deliberate, measured
#: decision. None is left: the VLM families, phi3 and the generic rule were
#: measured on 2026-10-09 and follow the bands like every other family
#: (``docs/PROFILE_COMPARISON.md`` sections 10.3 and 10.6).
EXPECTED_PINS: dict[tuple[str, str], str] = {}


def test_the_profile_pins_are_exactly_the_deliberate_ones() -> None:
    registry = load_builtin_registry()
    carriers: list[ArchitectureRule | ModelRule] = [*registry.architectures, *registry.models]
    pins = {
        (rule.id, device): profile_id
        for rule in carriers
        for device, profile_id in rule.profiles.items()
    }
    assert pins == EXPECTED_PINS


#: The quantization each existing id has always meant. Never edit one in place:
#: give a different quantization a new id.
FROZEN_QUANTIZATION: dict[str, dict[str, object]] = {
    "cpu-int4-asym-g128": {
        "weight_format": "int4",
        "symmetric": False,
        "ratio": 1.0,
        "group_size": 128,
    },
    "gpu-int4-asym-g128": {
        "weight_format": "int4",
        "symmetric": False,
        "ratio": 1.0,
        "group_size": 128,
    },
    "npu-int4-sym-cw": {"weight_format": "int4", "symmetric": True, "ratio": 1.0, "group_size": -1},
    "npu-int4-sym-g128": {
        "weight_format": "int4",
        "symmetric": True,
        "ratio": 1.0,
        "group_size": 128,
    },
    "cpu-int4-asym-g128-r08": {
        "weight_format": "int4",
        "symmetric": False,
        "ratio": 0.8,
        "group_size": 128,
    },
    "gpu-int4-asym-g128-r08": {
        "weight_format": "int4",
        "symmetric": False,
        "ratio": 0.8,
        "group_size": 128,
    },
}


@pytest.mark.parametrize("profile_id", sorted(FROZEN_QUANTIZATION))
def test_an_existing_profile_id_keeps_its_quantization(profile_id: str) -> None:
    profile = load_builtin_registry().profile(profile_id)
    assert profile is not None
    actual = profile.quantization.model_dump(exclude_none=True)
    for key, value in FROZEN_QUANTIZATION[profile_id].items():
        assert actual.get(key) == value, f"{profile_id}.{key}"


def test_every_r08_profile_really_keeps_a_fifth_at_int8() -> None:
    """A misspelled ``ratio`` key would be dropped silently and leave ratio 1.0."""
    r08 = [rule for rule in load_builtin_registry().profiles if rule.id.endswith("-r08")]
    assert {rule.device for rule in r08} == {"CPU", "GPU"}
    assert all(rule.quantization.ratio == 0.8 for rule in r08)


def test_every_banded_builtin_profile_carries_tested_evidence() -> None:
    banded = [
        rule
        for rule in load_builtin_registry().profiles
        if rule.applies_to.parameter_count is not None
    ]
    assert {rule.id for rule in banded} == {
        "cpu-int4-asym-g128-r08",
        "gpu-int4-asym-g128-r08",
        "npu-int4-sym-g128",
    }
    for rule in banded:
        assert rule.evidence.value == "tested"
        assert any(source.kind.value == "tested" for source in rule.sources), rule.id
        assert rule.applies_to.parameter_count is not None
        assert rule.applies_to.parameter_count.below == 5_000_000_000


def test_no_placeholder_is_left_in_the_builtin_data() -> None:
    for path in BUILTIN_DIR.glob("*.yaml"):
        assert "THRESHOLD" not in path.read_text(encoding="utf-8"), path.name


# --- what a user actually gets ---------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "cpu", "gpu", "npu"),
    [
        (751_632_384, "cpu-int4-asym-g128-r08", "gpu-int4-asym-g128-r08", "npu-int4-sym-g128"),
        (4_022_468_096, "cpu-int4-asym-g128-r08", "gpu-int4-asym-g128-r08", "npu-int4-sym-g128"),
        (5_000_000_000, "cpu-int4-asym-g128", "gpu-int4-asym-g128", "npu-int4-sym-cw"),
        (8_190_735_360, "cpu-int4-asym-g128", "gpu-int4-asym-g128", "npu-int4-sym-cw"),
    ],
)
def test_qwen3_gets_the_measured_profile_for_its_size(
    resolver: RegistryResolver, count: int, cpu: str, gpu: str, npu: str
) -> None:
    meta = text_model("qwen3", "Qwen3ForCausalLM", count)
    assert chosen(resolver, meta, "cpu") == cpu
    assert chosen(resolver, meta, "gpu") == gpu
    assert chosen(resolver, meta, "npu") == npu


@pytest.mark.parametrize(
    ("model_type", "architecture"),
    [
        ("llama", "LlamaForCausalLM"),
        ("qwen2", "Qwen2ForCausalLM"),
        ("gemma3_text", "Gemma3ForCausalLM"),
        ("gemma2", "Gemma2ForCausalLM"),
        ("olmo2", "Olmo2ForCausalLM"),
        ("granite", "GraniteForCausalLM"),
        ("smollm3", "SmolLM3ForCausalLM"),
        ("lfm2", "Lfm2ForCausalLM"),
    ],
)
def test_every_measured_family_gets_the_small_model_profiles(
    resolver: RegistryResolver, model_type: str, architecture: str
) -> None:
    meta = text_model(model_type, architecture, 1_000_000_000)
    assert chosen(resolver, meta, "cpu") == "cpu-int4-asym-g128-r08"
    assert chosen(resolver, meta, "gpu") == "gpu-int4-asym-g128-r08"
    assert chosen(resolver, meta, "npu") == "npu-int4-sym-g128"


@pytest.mark.parametrize(
    ("model_type", "architecture", "count", "cpu", "gpu"),
    [
        pytest.param(
            "qwen2_5_vl",
            "Qwen2_5_VLForConditionalGeneration",
            3_754_622_976,
            "cpu-int4-asym-g128-r08",
            "gpu-int4-asym-g128-r08",
            id="qwen2.5-vl-3b",
        ),
        pytest.param(
            "qwen2_5_vl",
            "Qwen2_5_VLForConditionalGeneration",
            8_292_166_656,
            "cpu-int4-asym-g128",
            "gpu-int4-asym-g128",
            id="qwen2.5-vl-7b",
        ),
        pytest.param(
            "qwen3_5",
            "Qwen3_5ForConditionalGeneration",
            873_438_784,
            "cpu-int4-asym-g128-r08",
            "gpu-int4-asym-g128-r08",
            id="qwen3.5-0.8b",
        ),
        pytest.param(
            "qwen3_5",
            "Qwen3_5ForConditionalGeneration",
            2_274_069_824,
            "cpu-int4-asym-g128-r08",
            "gpu-int4-asym-g128-r08",
            id="qwen3.5-2b",
        ),
        pytest.param(
            "qwen3_5",
            "Qwen3_5ForConditionalGeneration",
            9_653_104_368,
            "cpu-int4-asym-g128",
            "gpu-int4-asym-g128",
            id="qwen3.5-9b",
        ),
    ],
)
def test_a_vlm_follows_the_size_bands(
    resolver: RegistryResolver,
    model_type: str,
    architecture: str,
    count: int,
    cpu: str,
    gpu: str,
) -> None:
    """Below 5B ratio 1.0 cannot even prepare Qwen3.5 (NNCF's group-size check fails)."""
    meta = text_model(model_type, architecture, count)
    arch_rule, _ = resolver.resolve_architecture(meta)
    assert arch_rule is not None
    assert arch_rule.id == f"arch-{model_type}"
    assert chosen(resolver, meta, "cpu") == cpu
    assert chosen(resolver, meta, "gpu") == gpu


@pytest.mark.parametrize(
    "meta",
    [
        pytest.param(text_model("qwen3", "Qwen3ForCausalLM", None), id="no-count"),
        pytest.param(
            text_model("qwen3", "Qwen3ForCausalLM", 600_000_000).model_copy(
                update={"parameter_dtypes": {"I32": 500_000_000, "F16": 100_000_000}}
            ),
            id="packed",
        ),
        pytest.param(
            text_model(
                "qwen3",
                "Qwen3ForCausalLM",
                600_000_000,
                config={"model_type": "qwen3", "quantization_config": {"bits": 4}},
            ),
            id="prequantized",
        ),
    ],
)
def test_an_unusable_size_gets_the_device_defaults(
    resolver: RegistryResolver, meta: ModelMetadata
) -> None:
    assert chosen(resolver, meta, "cpu") == "cpu-int4-asym-g128"
    assert chosen(resolver, meta, "gpu") == "gpu-int4-asym-g128"
    assert chosen(resolver, meta, "npu") == "npu-int4-sym-cw"


@pytest.mark.parametrize(
    ("count", "npu"),
    [
        pytest.param(1_831_201_280, "npu-int4-sym-g128", id="danube2-1.8b"),
        pytest.param(7_248_023_552, "npu-int4-sym-cw", id="mistral-7b"),
    ],
)
def test_mistral_on_the_npu_follows_the_size_bands(
    resolver: RegistryResolver, count: int, npu: str
) -> None:
    """mistral reaches only the generic rule, which no longer pins the NPU."""
    meta = text_model("mistral", "MistralForCausalLM", count)
    arch_rule, _ = resolver.resolve_architecture(meta)
    assert arch_rule is not None
    assert arch_rule.id == "arch-generic-text-generation"
    assert chosen(resolver, meta, "npu") == npu


def test_a_small_phi3_gets_group_wise_int4_on_the_npu(resolver: RegistryResolver) -> None:
    meta = text_model("phi3", "Phi3ForCausalLM", 3_836_021_760)
    arch_rule, _ = resolver.resolve_architecture(meta)
    assert arch_rule is not None
    assert arch_rule.id == "arch-phi3"
    assert chosen(resolver, meta, "npu") == "npu-int4-sym-g128"
    assert chosen(resolver, meta, "cpu") == "cpu-int4-asym-g128-r08"
