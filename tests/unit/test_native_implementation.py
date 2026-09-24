"""A repository's own modelling code is not "remote code required" when Transformers has its own.

Phi-4-mini ships ``modeling_phi3.py`` and an ``auto_map``; yonga refused it
(exit 22) until the registry could say that Transformers implements phi3
natively. With remote code off, that file is ignored and never run -- which is
safer than asking the user to pass --trust-remote-code to get past the refusal.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tests.unit.test_prepare_planning import (
    FakeHub,
    build_service,
    builtin_resolver,
    make_snapshot,
    safetensors_metadata,
    settings_for,
)
from yonga.application.prepare import PrepareService
from yonga.domain.diagnostics import RegistryError
from yonga.domain.models import ModelMetadata
from yonga.registry.loader import parse_document

REPO = "acme/phi-like"


def phi_metadata(model_type: str = "phi3", architecture: str = "Phi3ForCausalLM") -> ModelMetadata:
    metadata = safetensors_metadata(
        REPO, model_type=model_type, pipeline_tag="text-generation", requires_remote_code=True
    )
    return metadata.model_copy(
        update={
            "config": {"model_type": model_type, "architectures": [architecture]},
            "architectures": [architecture],
            "remote_code_files": ["modeling_phi3.py"],
        }
    )


def service_for(
    tmp_path: Path, model_type: str = "phi3", architecture: str = "Phi3ForCausalLM"
) -> PrepareService:
    settings = settings_for(tmp_path)
    metadata = phi_metadata(model_type, architecture)
    hub = FakeHub(repos={REPO: metadata}, snapshots={REPO: make_snapshot(tmp_path)})
    return build_service(settings=settings, hub=hub)


def test_a_natively_implemented_architecture_needs_no_remote_code(tmp_path: Path) -> None:
    plan = service_for(tmp_path).plan(REPO, device="gpu")

    assert plan.trust_remote_code_required is False
    assert plan.trust_remote_code_enabled is False, "nothing turns remote code on"
    assert any("modeling_phi3.py" in note and "not run" in note for note in plan.known_limitations)


def test_an_unknown_architecture_with_an_auto_map_still_needs_it(tmp_path: Path) -> None:
    service = service_for(tmp_path, model_type="mystery_arch", architecture="MysteryForCausalLM")
    plan = service.plan(REPO, device="gpu")

    assert plan.trust_remote_code_required is True


def test_a_rule_cannot_claim_both_native_code_and_remote_code() -> None:
    contradictory = textwrap.dedent(
        """\
        schema_version: 1
        architectures:
          - id: arch-bad
            schema_version: 1
            status: active
            priority: 100
            summary: contradictory
            match:
              model_types: [bad]
            export:
              trust_remote_code: true
              native_implementation: true
        """
    )

    with pytest.raises(RegistryError):
        parse_document(contradictory, origin="bad.yaml")


def test_the_shipped_phi3_rule_declares_native_support() -> None:
    rule = builtin_resolver().registry.architecture("arch-phi3")

    assert rule is not None
    assert rule.export.native_implementation is True
    assert rule.export.trust_remote_code is False
