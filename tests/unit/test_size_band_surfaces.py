"""Where a profile's size band shows up outside the resolver.

The plan panel, ``inspect``, ``registry explain``, the Hub adapter that
collects the per-dtype counts, and the maintainer tool that signs remote
bundles. The resolver itself is covered by the registry tests; here the
selection sentence only has to reach each surface, and stay away from the
ones where no profile decides the quantization.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import re
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from tests.unit.test_hf_adapter import safetensors_adapter
from tests.unit.test_ui_tables import plan, render
from yonga.adapters.hf import _parameter_dtypes, _search_hit_metadata
from yonga.domain.compatibility import ProfileSelection
from yonga.domain.diagnostics import RegistryError
from yonga.domain.models import ModelFormat, ModelIdentity, ModelMetadata, ResolvedSource
from yonga.registry.loader import load_registry_from_documents, parse_document, validate_registry
from yonga.registry.remote import TrustedKey, key_id, verify_bundle
from yonga.registry.resolver import RegistryResolver
from yonga.registry.schema import SCHEMA_MIN_TOOL_VERSION, SCHEMA_VERSION
from yonga.ui.tables import inspect_report, preparation_plan, registry_explanation

REPO_ROOT = Path(__file__).resolve().parents[2]

HOST_VERSIONS = {"openvino": "2026.4.0", "optimum_intel": "2.2.0", "transformers": "5.5.4"}

#: A CPU default plus a small-model profile that outranks it inside its band.
BANDED_REGISTRY = """\
schema_version: 2
profiles:
  - id: cpu-int8
    schema_version: 2
    device: cpu
    priority: 100
    quantization:
      weight_format: int8
  - id: cpu-small-fp16
    schema_version: 2
    device: cpu
    priority: 200
    evidence: tested
    quantization:
      weight_format: fp16
    applies_to:
      parameter_count:
        below: 2B
"""

#: The same CPU default, as a schema-1 file with no band anywhere.
PLAIN_REGISTRY = """\
schema_version: 1
profiles:
  - id: cpu-int8
    schema_version: 1
    device: cpu
    quantization:
      weight_format: int8
"""


def small_model(**overrides: Any) -> ModelMetadata:
    """A 600M-parameter BF16 checkpoint: inside the ``below 2B`` band."""
    payload: dict[str, Any] = {
        "identity": ModelIdentity(repo_id="demo/small-lm"),
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "pipeline_tag": "text-generation",
        "parameter_count": 600_000_000,
        "parameter_dtypes": {"BF16": 600_000_000},
    }
    payload.update(overrides)
    return ModelMetadata.model_validate(payload)


def banded_resolver() -> RegistryResolver:
    document = parse_document(BANDED_REGISTRY, origin="banded.yaml")
    registry = load_registry_from_documents([document])
    assert validate_registry(registry) == []
    return RegistryResolver(registry, versions=HOST_VERSIONS)


def size_band_selection() -> ProfileSelection:
    """The selection the resolver records for :func:`small_model` on the CPU."""
    _profile, explanation = banded_resolver().resolve_profile(device="cpu", meta=small_model())
    assert explanation.profile_selection is not None
    return explanation.profile_selection


# ----------------------------------------------------------------------
# plan panel
# ----------------------------------------------------------------------
def test_plan_panel_says_why_a_size_banded_profile_was_chosen():
    selection = size_band_selection()
    assert selection.basis == "size_band"
    text = render(preparation_plan(plan(profile_selection=selection)), width=240)
    assert "Chosen because" in text
    assert "600M parameters" in text
    assert "below 2B" in text


def test_plan_panel_has_no_chosen_because_row_without_a_selection():
    assert "Chosen because" not in render(preparation_plan(plan()), width=240)


def test_plan_panel_hides_a_selection_that_does_not_apply():
    """An OpenVINO copy keeps its own quantization: no reason for a profile to print."""
    selection = size_band_selection().model_copy(update={"applies": False})
    text = render(preparation_plan(plan(profile_selection=selection)), width=240)
    assert "Chosen because" not in text


def test_plan_panel_for_a_gguf_import_never_says_why_a_profile_was_chosen():
    """The GGUF file decides the weights, so even an applying selection stays out."""
    source = ResolvedSource(
        requested=ModelIdentity(repo_id="demo/small-lm-GGUF"),
        source=ModelIdentity(repo_id="demo/small-lm-GGUF", revision="0" * 40),
        format=ModelFormat.GGUF,
        gguf_file="small-lm-Q4_K_M.gguf",
        gguf_tensor_types={"Q4_K": 120, "F32": 40},
    )
    gguf_plan = plan(resolved_source=source, profile_selection=size_band_selection())
    text = render(preparation_plan(gguf_plan), width=240)
    assert "imported directly by OpenVINO GenAI" in text
    assert "Chosen because" not in text


# ----------------------------------------------------------------------
# inspect
# ----------------------------------------------------------------------
def test_inspect_parameters_row_names_the_hub_count():
    text = render(inspect_report(small_model()), width=240)
    assert "600,000,000" in text
    assert "Hub count; includes embeddings and any vision tower" in text
    assert "not used to choose a profile" not in text


def test_inspect_parameters_row_flags_a_packed_checkpoint():
    """GPTQ-style I32 packing makes the Hub count understate the model."""
    packed = small_model(
        parameter_count=1_000_000_000,
        parameter_dtypes={"I32": 600_000_000, "F16": 400_000_000},
    )
    text = render(inspect_report(packed), width=240)
    assert "1,000,000,000" in text
    assert "not used to choose a profile" in text
    assert "packed integer weights (I32: 600,000,000)" in text
    assert "Hub count; includes" not in text


def test_inspect_parameters_row_names_the_largest_integer_dtype():
    """Two packed dtypes: the row names the one holding most of the weights."""
    packed = small_model(
        parameter_count=1_000_000_000,
        parameter_dtypes={"U8": 50_000_000, "I32": 500_000_000, "F16": 450_000_000},
    )
    text = render(inspect_report(packed), width=240)
    assert "packed integer weights (I32: 500,000,000)" in text
    assert "U8: 50,000,000" not in text


def test_inspect_parameters_row_ignores_a_small_integer_buffer():
    """Position ids stored as I64 are under 1% of the count: not a packed checkpoint."""
    buffered = small_model(parameter_dtypes={"BF16": 599_000_000, "I64": 1_000_000})
    text = render(inspect_report(buffered), width=240)
    assert "Hub count; includes embeddings and any vision tower" in text
    assert "not used to choose a profile" not in text


def test_inspect_parameters_row_flags_a_quantization_config():
    quantized = small_model(config={"quantization_config": {"quant_method": "awq"}})
    text = render(inspect_report(quantized), width=240)
    assert "not used to choose a profile" in text
    assert "quantization_config" in text


def test_inspect_has_no_parameters_row_without_a_count():
    text = render(inspect_report(small_model(parameter_count=None, parameter_dtypes={})))
    assert "Parameters" not in text


# ----------------------------------------------------------------------
# registry explain
# ----------------------------------------------------------------------
def test_registry_explain_prints_why_the_profile_was_chosen():
    explanation = banded_resolver().explain(meta=small_model(), device="cpu")
    assert explanation.profile_id == "cpu-small-fp16"
    text = render(registry_explanation(explanation), width=240)
    assert "Chosen because" in text
    assert "cpu-small-fp16 is the CPU profile for that size" in text


def test_registry_explain_names_the_band_a_large_model_missed():
    large = small_model(parameter_count=8_000_000_000, parameter_dtypes={"BF16": 8_000_000_000})
    explanation = banded_resolver().explain(meta=large, device="cpu")
    assert explanation.profile_id == "cpu-int8"
    text = render(registry_explanation(explanation), width=240)
    assert "Chosen because" in text
    assert "Highest-priority CPU profile." in text
    assert "cpu-small-fp16 is for models below 2B" in text


@pytest.mark.parametrize(
    ("count", "profile_id"),
    [(1_999_999_999, "cpu-small-fp16"), (2_000_000_000, "cpu-int8")],
)
def test_registry_explain_treats_the_band_as_half_open(count: int, profile_id: str):
    """``below: 2B`` stops short of 2B: a model of exactly 2B gets the default."""
    sized = small_model(parameter_count=count, parameter_dtypes={"BF16": count})
    explanation = banded_resolver().explain(meta=sized, device="cpu")
    assert explanation.profile_id == profile_id


# ----------------------------------------------------------------------
# hf adapter
# ----------------------------------------------------------------------
def test_parameter_dtypes_from_a_safetensors_info_object():
    info = SimpleNamespace(
        safetensors=SimpleNamespace(total=7, parameters={"BF16": 5, "I64": 2}),
    )
    assert _parameter_dtypes(info) == {"BF16": 5, "I64": 2}


def test_parameter_dtypes_from_a_mapping():
    """The search endpoint can hand the safetensors block back as a plain dict."""
    info = SimpleNamespace(safetensors={"total": 9, "parameters": {"F16": 9}})
    assert _parameter_dtypes(info) == {"F16": 9}


def test_parameter_dtypes_drop_values_that_are_not_counts():
    info = SimpleNamespace(
        safetensors={"parameters": {"F16": 9, "I32": True, "U8": -1, "F32": "12", 4: 3}}
    )
    assert _parameter_dtypes(info) == {"F16": 9}


@pytest.mark.parametrize(
    "info",
    [
        SimpleNamespace(),
        SimpleNamespace(safetensors=None),
        SimpleNamespace(safetensors=SimpleNamespace(total=7)),
        SimpleNamespace(safetensors={"total": 7}),
        SimpleNamespace(safetensors={"parameters": ["F16"]}),
    ],
)
def test_parameter_dtypes_are_empty_when_the_hub_reports_none(info: object):
    assert _parameter_dtypes(info) == {}


def test_inspect_metadata_carries_the_per_dtype_counts(tmp_path: Path):
    """Without the dtypes a packed checkpoint would pass for a small model."""
    adapter, api = safetensors_adapter(tmp_path)
    assert api.info is not None
    api.info.safetensors = cast(
        Any, SimpleNamespace(total=1_000_000_000, parameters={"I32": 600_000_000, "F16": 4})
    )

    metadata = adapter.inspect(ModelIdentity(repo_id="nDimensional/x"))

    assert metadata.parameter_dtypes == {"I32": 600_000_000, "F16": 4}
    assert metadata.size_for_matching()[0] is None


def test_search_hit_metadata_carries_the_per_dtype_counts():
    info = SimpleNamespace(
        id="demo/packed-lm",
        safetensors={"total": 1_000_000_000, "parameters": {"I32": 600_000_000}},
    )
    metadata = _search_hit_metadata(info)
    assert metadata.parameter_dtypes == {"I32": 600_000_000}
    assert metadata.size_for_matching()[0] is None


# ----------------------------------------------------------------------
# tools/registry_bundle.py
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def bundle_tool() -> ModuleType:
    """The maintainer script, loaded by path: ``tools/`` is not a package."""
    path = REPO_ROOT / "tools" / "registry_bundle.py"
    spec = importlib.util.spec_from_file_location("registry_bundle_tool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_schema_versions_reads_document_and_rule_levels(bundle_tool: ModuleType):
    files = {
        "a.yaml": "schema_version: 2\nprofiles:\n  - id: x\n    device: cpu\n",
        "b.yaml": "schema_version: 1\nissues: []\n",
    }
    assert bundle_tool.schema_versions(files) == {1, 2}


def test_an_undeclared_schema_version_is_read_as_the_loader_reads_it(bundle_tool: ModuleType):
    """No ``schema_version`` means the build's own, so the bundle must need this build."""
    banded = re.sub(r"(?m)^\s*schema_version: 2\n", "", BANDED_REGISTRY)
    assert "schema_version" not in banded
    assert bundle_tool.schema_versions({"p.yaml": banded}) == {SCHEMA_VERSION}
    expected = SCHEMA_MIN_TOOL_VERSION[SCHEMA_VERSION]
    assert bundle_tool.required_tool_version({"p.yaml": banded}, None) == expected


def test_required_tool_version_for_a_schema_2_profile(bundle_tool: ModuleType):
    assert bundle_tool.required_tool_version({"p.yaml": BANDED_REGISTRY}, None) == "0.3.0"


def test_required_tool_version_follows_the_newest_schema_in_the_bundle(bundle_tool: ModuleType):
    """One schema-2 rule among schema-1 files is enough to need the newer yonga."""
    files = {"plain.yaml": PLAIN_REGISTRY, "banded.yaml": BANDED_REGISTRY}
    assert bundle_tool.required_tool_version(files, None) == "0.3.0"


def test_required_tool_version_is_none_for_schema_1(bundle_tool: ModuleType):
    assert bundle_tool.required_tool_version({"p.yaml": PLAIN_REGISTRY}, None) is None


def test_required_tool_version_keeps_an_explicit_request(bundle_tool: ModuleType):
    files = {"p.yaml": BANDED_REGISTRY}
    assert bundle_tool.required_tool_version(files, "0.4.1") == "0.4.1"
    assert bundle_tool.required_tool_version(files, "0.3.0") == "0.3.0"
    assert bundle_tool.required_tool_version({"p.yaml": PLAIN_REGISTRY}, "0.1.0") == "0.1.0"


@pytest.mark.parametrize("requested", ["0.2.9", "not-a-version"])
def test_required_tool_version_refuses_a_floor_below_the_schema(
    bundle_tool: ModuleType, requested: str
):
    with pytest.raises(SystemExit, match=r"below 0\.3\.0"):
        bundle_tool.required_tool_version({"p.yaml": BANDED_REGISTRY}, requested)


def test_a_band_without_a_rule_level_schema_version_follows_its_document(
    bundle_tool: ModuleType,
):
    """A rule inherits its document's schema version, so a schema-1 file cannot hide a band.

    Were it to default to the build's version instead, the file would load, the
    tool would see only schema 1, and the bundle would carry no
    min_tool_version for older yonga releases to refuse it by.
    """
    text = textwrap.dedent(
        """\
        schema_version: 1
        profiles:
          - id: cpu-small-fp16
            device: cpu
            evidence: tested
            applies_to:
              parameter_count:
                below: 2B
        """
    )
    with pytest.raises(RegistryError, match="needs schema_version 2"):
        parse_document(text, origin="p.yaml")
    assert bundle_tool.required_tool_version({"p.yaml": text.replace("1\n", "2\n", 1)}, None) == (
        "0.3.0"
    )


def write_registry(directory: Path, text: str) -> Path:
    directory.mkdir()
    (directory / "profiles.yaml").write_text(text, encoding="utf-8")
    return directory


def generated_key(
    bundle_tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> tuple[Path, TrustedKey]:
    """A fresh signing key from the tool's own ``keygen``, and its trusted public half."""
    bundle_tool.keygen(tmp_path / "signing")
    capsys.readouterr()
    public = base64.b64decode((tmp_path / "signing.pub").read_text(encoding="utf-8"))
    trusted = TrustedKey(key_id=key_id(public), public_key=public, source=tmp_path / "signing.pub")
    return tmp_path / "signing.key", trusted


def test_build_signs_a_schema_2_bundle_with_its_tool_floor(
    bundle_tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    key, trusted = generated_key(bundle_tool, tmp_path, capsys)
    source = write_registry(tmp_path / "registry", BANDED_REGISTRY)
    output = tmp_path / "bundle.json"

    bundle_tool.build(source, key, 7, 90, output, "size bands")

    payload = json.loads(output.read_bytes())
    assert payload["min_tool_version"] == "0.3.0"
    assert payload["version"] == 7
    assert set(payload["files"]) == {"profiles.yaml"}
    verified = verify_bundle(
        output.read_bytes(),
        output.with_name("bundle.json.sig").read_bytes(),
        [trusted],
        tool_version="0.3.0",
    )
    assert verified.bundle.min_tool_version == "0.3.0"
    assert verified.key_id == trusted.key_id
    assert "needs yonga 0.3.0+" in capsys.readouterr().out


def test_build_refuses_the_bundle_for_a_yonga_older_than_its_schema(
    bundle_tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    key, trusted = generated_key(bundle_tool, tmp_path, capsys)
    source = write_registry(tmp_path / "registry", BANDED_REGISTRY)
    output = tmp_path / "bundle.json"
    bundle_tool.build(source, key, 7, 90, output, "")

    with pytest.raises(RegistryError) as excinfo:
        verify_bundle(
            output.read_bytes(),
            output.with_name("bundle.json.sig").read_bytes(),
            [trusted],
            tool_version="0.2.0",
        )
    assert excinfo.value.diagnostic.code == "registry.remote.tool_too_old"


def test_build_leaves_a_schema_1_bundle_without_a_floor(
    bundle_tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    key, _trusted = generated_key(bundle_tool, tmp_path, capsys)
    source = write_registry(tmp_path / "registry", PLAIN_REGISTRY)
    output = tmp_path / "bundle.json"

    bundle_tool.build(source, key, 3, 30, output, "")

    assert "min_tool_version" not in json.loads(output.read_bytes())
    assert "needs yonga" not in capsys.readouterr().out


def test_build_writes_nothing_for_a_floor_below_the_schema(
    bundle_tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    key, _trusted = generated_key(bundle_tool, tmp_path, capsys)
    source = write_registry(tmp_path / "registry", BANDED_REGISTRY)
    output = tmp_path / "bundle.json"
    with pytest.raises(SystemExit, match=r"below 0\.3\.0"):
        bundle_tool.build(source, key, 7, 90, output, "", "0.2.0")

    assert not output.exists()
    assert not output.with_name("bundle.json.sig").exists()


def test_main_refuses_an_already_expired_bundle_in_one_line(
    bundle_tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """An expected refusal is a message for the maintainer, not a traceback."""
    key, _trusted = generated_key(bundle_tool, tmp_path, capsys)
    source = write_registry(tmp_path / "registry", PLAIN_REGISTRY)
    output = tmp_path / "bundle.json"
    argv = ["build", str(source), "--key", str(key), "--version", "7", "--expires-days", "0"]

    with pytest.raises(SystemExit) as excinfo:
        bundle_tool.main([*argv, "--output", str(output)])

    message = str(excinfo.value.code)
    assert message.startswith("refused: The registry bundle has expired: It expired at ")
    assert "\n" not in message
    assert not output.exists()


def test_main_reports_a_missing_key_in_one_line(bundle_tool: ModuleType, tmp_path: Path):
    source = write_registry(tmp_path / "registry", PLAIN_REGISTRY)
    argv = ["build", str(source), "--key", str(tmp_path / "absent.key"), "--version", "1"]

    with pytest.raises(SystemExit) as excinfo:
        bundle_tool.main([*argv, "--output", str(tmp_path / "bundle.json")])

    assert "absent.key" in str(excinfo.value.code)


@pytest.mark.parametrize("command", ["keygen", "build"])
def test_every_option_of_the_tool_is_described(bundle_tool: ModuleType, command: str):
    root = bundle_tool.parser()
    (subcommands,) = [
        action for action in root._actions if isinstance(action, argparse._SubParsersAction)
    ]
    sub = subcommands.choices[command]
    undocumented = [
        action.dest
        for action in sub._actions
        if not isinstance(action, argparse._HelpAction) and not action.help
    ]
    assert undocumented == []
