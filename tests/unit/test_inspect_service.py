"""``InspectService`` tests: metadata-only discovery and evidence-backed compatibility.

Two properties matter more than the rest and are asserted from several angles:

* ``inspect`` never downloads a weight shard — the fake Hub raises if anything
  tries;
* compatibility is reported as *evidence*. A device that exists on the host but
  that no registry rule speaks about is ``unknown``, never ``supported``.

Host facts (device names, package versions, the Qwen3.5 repository listings and
their commit SHAs) are the ones probed on this machine, so the exporter-version
mismatch this host really has — ``transformers 5.5.4`` against the registry's
``==5.2.*`` — is asserted rather than papered over.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from yonga.adapters import HardwareProbe, HubClient
from yonga.adapters.hf import detect_formats
from yonga.application import inspect_model as inspect_module
from yonga.application.inspect_model import InspectReport, InspectService
from yonga.application.progress import CollectingSink, ProgressKind
from yonga.application.resolve import OPTION_DERIVATIVE, OPTION_GGUF, OPTION_SAFETENSORS
from yonga.config.settings import DevicePolicy, Settings
from yonga.domain.compatibility import DeviceStatus, EvidenceLevel
from yonga.domain.hardware import (
    DeviceKind,
    HardwareInventory,
    HostInfo,
    HuggingFaceAuthInfo,
    MemoryInfo,
    OpenVINODevice,
)
from yonga.domain.models import ModelIdentity, ModelMetadata, PipelineKind, RepoFile
from yonga.registry.loader import load_builtin_registry
from yonga.registry.resolver import RegistryResolver

# --------------------------------------------------------------------------- #
# Verified host and Hub facts (probed 2026-09-22; see HOST_FACTS.md)
# --------------------------------------------------------------------------- #

REQUESTED_REPO = "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"
REQUESTED_SHA = "0a41c68809d375475f954be12ba7c40efa56c2a9"
DERIVATIVE_REPO = "nDimensional/Qwen3.5-9B-Uncensored-Safetensors"
DERIVATIVE_SHA = "4623e504074fc2489f3af98dc3d68b8392bbc21f"

#: Exactly what this machine has installed.
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

GGUF_FILES: tuple[tuple[str, int | None], ...] = (
    (".gitattributes", 1_500),
    ("README.md", 4_200),
    ("Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf", None),
    ("Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf", None),
    ("mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf", None),
)

SAFETENSORS_FILES: tuple[tuple[str, int | None], ...] = (
    ("config.json", 2_100),
    ("model.safetensors-00001-of-00004.safetensors", 5_278_251_472),
    ("model.safetensors-00002-of-00004.safetensors", 5_358_071_808),
    ("model.safetensors-00003-of-00004.safetensors", 5_340_098_264),
    ("model.safetensors-00004-of-00004.safetensors", 2_843_299_496),
    ("model.safetensors.index.json", 190_000),
    ("tokenizer.json", 11_400_000),
)

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
    """A :class:`~yonga.adapters.HubClient` that serves canned metadata."""

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
        """Return the canned SHA for a repository."""
        self.resolved.append(model.repo_id)
        return self.revisions.get(model.repo_id, "main")

    def inspect(self, model: ModelIdentity, *, with_sizes: bool = True) -> ModelMetadata:
        """Return the canned metadata for a repository."""
        self.inspected.append(model.repo_id)
        return self.repos[model.repo_id]

    def download(
        self,
        source: ModelIdentity,
        *,
        ctx: object | None = None,
        allow_patterns: object | None = None,
        ignore_patterns: object | None = None,
    ) -> Path:
        """Fail loudly: ``inspect`` must never move a weight shard."""
        self.downloaded.append(source.repo_id)
        raise AssertionError("inspect must not download anything")

    def auth_info(self) -> HuggingFaceAuthInfo:
        """Report an anonymous session."""
        return HuggingFaceAuthInfo(logged_in=False)


def build_inventory(*, devices: tuple[str, ...] = ("CPU", "GPU", "NPU")) -> HardwareInventory:
    """Build an inventory that mirrors this Lunar Lake host."""
    catalogue = {
        "CPU": OpenVINODevice(
            name="CPU", kind=DeviceKind.CPU, full_name="Intel(R) Core(TM) Ultra 7 268V"
        ),
        "GPU": OpenVINODevice(
            name="GPU",
            kind=DeviceKind.GPU,
            full_name="Intel(R) Arc(TM) Graphics (iGPU)",
            architecture="GPU: vendor=0x8086 arch=v20.4.4",
        ),
        "NPU": OpenVINODevice(
            name="NPU", kind=DeviceKind.NPU, full_name="Intel(R) AI Boost", architecture="4000"
        ),
    }
    return HardwareInventory(
        host=HostInfo(
            platform="Linux",
            machine="x86_64",
            kernel="7.2.6-070206-generic",
            python_version="3.14.4",
            distro_id="ubuntu",
            cpu_model="Intel(R) Core(TM) Ultra 7 268V",
            cpu_count=8,
        ),
        memory=MemoryInfo(
            total_bytes=33_102_655_488,
            available_bytes=20_000_000_000,
            swap_total_bytes=0,
            swap_free_bytes=0,
        ),
        devices=[catalogue[name] for name in devices],
        openvino_available=True,
        device_tags=[
            "intel.cpu",
            "intel.core_ultra.series2",
            "intel.gpu",
            "intel.npu",
            "intel.npu.lunar_lake",
        ],
    )


class FakeHardware:
    """A :class:`~yonga.adapters.HardwareProbe` returning a fixed inventory."""

    def __init__(self, inventory: HardwareInventory | None = None) -> None:
        self.result = build_inventory() if inventory is None else inventory
        self.calls = 0

    def inventory(self, *, full: bool = False) -> HardwareInventory:
        """Return the canned inventory and record the call."""
        self.calls += 1
        return self.result


def registry_resolver() -> RegistryResolver:
    """Build a resolver over the shipped registry with this host's versions."""
    return RegistryResolver(load_builtin_registry(), versions=HOST_VERSIONS)


def build_service(
    hub: FakeHub,
    *,
    hardware: FakeHardware | None = None,
    settings: Settings | None = None,
    resolver: RegistryResolver | None = None,
    progress: CollectingSink | None = None,
) -> InspectService:
    """Assemble an :class:`InspectService` from fakes."""
    return InspectService(
        hub,
        resolver if resolver is not None else registry_resolver(),
        hardware if hardware is not None else FakeHardware(),
        settings if settings is not None else Settings(),
        progress=progress,
    )


@pytest.fixture
def gguf_only() -> ModelMetadata:
    """The GGUF-only requested repository, which has no ``config.json``."""
    return build_metadata(
        REQUESTED_REPO,
        GGUF_FILES,
        resolved_revision=REQUESTED_SHA,
        license="apache-2.0",
        tags=["gguf", "uncensored", "qwen3.5", "qwen", "base_model:Qwen/Qwen3.5-9B"],
    )


#: The GGUF header the Hub really reports for the reference repository.
#:
#: ``architecture`` is read out of the weight file itself, so it is a statement
#: the artifact makes about what it is — unlike the repository tags beside it,
#: which an uploader typed.
HAUHAUCS_GGUF_HEADER: dict[str, object] = {
    "architecture": "qwen35",
    "context_length": 262_144,
    "totalFileSize": 17_920_696_704,
}


@pytest.fixture
def gguf_with_header(gguf_only: ModelMetadata) -> ModelMetadata:
    """The same GGUF-only repository, carrying the header the Hub parses out of it.

    This is the repository ``SPEC.md`` section 20 is written about: no
    ``config.json``, so the registry can reach its architecture rule only
    through an identity alias.
    """
    return gguf_only.model_copy(update={"gguf_metadata": dict(HAUHAUCS_GGUF_HEADER)})


@pytest.fixture
def qwen_safetensors() -> ModelMetadata:
    """The safetensors derivative, a Qwen3.5 VLM with no remote code."""
    return build_metadata(
        DERIVATIVE_REPO,
        SAFETENSORS_FILES,
        resolved_revision=DERIVATIVE_SHA,
        license="apache-2.0",
        pipeline_tag="image-text-to-text",
        library_name="transformers",
        architectures=["Qwen3_5ForConditionalGeneration"],
        model_type="qwen3_5",
        config={"model_type": "qwen3_5", "vision_config": {}, "text_config": {}},
    )


# --------------------------------------------------------------------------- #
# Inspect stays cheap
# --------------------------------------------------------------------------- #


def test_inspect_never_downloads_weight_shards(qwen_safetensors: ModelMetadata) -> None:
    """The whole point of ``inspect`` is that an 18 GB repo costs kilobytes."""
    hub = FakeHub({DERIVATIVE_REPO: qwen_safetensors})

    report = build_service(hub).inspect(DERIVATIVE_REPO)

    assert hub.downloaded == []
    assert hub.inspected == [DERIVATIVE_REPO]
    assert report.metadata.weight_bytes == 18_819_721_040


def test_inspect_does_not_reach_out_to_declared_derivatives(gguf_only: ModelMetadata) -> None:
    """Listing a derivative as an option must not cost a second Hub round trip."""
    hub = FakeHub({REQUESTED_REPO: gguf_only})

    report = build_service(hub).inspect(REQUESTED_REPO)

    assert hub.inspected == [REQUESTED_REPO]
    derivative = report.option(OPTION_DERIVATIVE)
    assert derivative is not None
    assert derivative.repo_id == DERIVATIVE_REPO
    assert derivative.requires_confirmation is True


def test_inspect_emits_progress_stages(qwen_safetensors: ModelMetadata) -> None:
    """The service reports progress through the sink instead of printing."""
    sink = CollectingSink()

    build_service(FakeHub({DERIVATIVE_REPO: qwen_safetensors}), progress=sink).inspect(
        DERIVATIVE_REPO
    )

    assert sink.of_kind(ProgressKind.STAGE)
    assert sink.of_kind(ProgressKind.PROBE)


# --------------------------------------------------------------------------- #
# Identity and revision
# --------------------------------------------------------------------------- #


def test_report_carries_the_resolved_commit_sha(qwen_safetensors: ModelMetadata) -> None:
    """The report always states the commit the metadata came from."""
    report = build_service(FakeHub({DERIVATIVE_REPO: qwen_safetensors})).inspect(DERIVATIVE_REPO)

    assert report.resolved_revision == DERIVATIVE_SHA
    assert report.metadata.resolved_revision == DERIVATIVE_SHA
    assert report.requested_revision is None


def test_a_branch_request_is_resolved_to_a_sha(qwen_safetensors: ModelMetadata) -> None:
    """A branch is recorded as requested and reported alongside its commit."""
    unpinned = qwen_safetensors.model_copy(update={"resolved_revision": None})
    hub = FakeHub({DERIVATIVE_REPO: unpinned}, revisions={DERIVATIVE_REPO: DERIVATIVE_SHA})

    report = build_service(hub).inspect(DERIVATIVE_REPO, revision="main")

    assert report.requested.revision == "main"
    assert report.requested_revision == "main"
    assert report.resolved_revision == DERIVATIVE_SHA
    assert hub.resolved == [DERIVATIVE_REPO]


def test_an_unresolvable_revision_becomes_a_note_not_a_crash(
    qwen_safetensors: ModelMetadata,
) -> None:
    """``inspect`` reports what it can; refusing a branch is ``prepare``'s job."""
    unpinned = qwen_safetensors.model_copy(update={"resolved_revision": None})
    hub = FakeHub({DERIVATIVE_REPO: unpinned})

    report = build_service(hub).inspect(DERIVATIVE_REPO)

    assert report.resolved_revision is None
    assert any("refuse to run from a branch" in note for note in report.notes)


def test_identity_may_be_given_as_a_string_with_a_revision(
    qwen_safetensors: ModelMetadata,
) -> None:
    """``repo@revision`` is parsed, and an explicit ``revision`` overrides it."""
    hub = FakeHub({DERIVATIVE_REPO: qwen_safetensors})

    report = build_service(hub).inspect(f"{DERIVATIVE_REPO}@refs/pr/1", revision=DERIVATIVE_SHA)

    assert report.requested == ModelIdentity(repo_id=DERIVATIVE_REPO, revision=DERIVATIVE_SHA)


def test_missing_config_json_does_not_break_inspection(gguf_only: ModelMetadata) -> None:
    """The reference fixture has no ``config.json``; inspection still succeeds."""
    report = build_service(FakeHub({REQUESTED_REPO: gguf_only})).inspect(REQUESTED_REPO)

    assert report.architecture is None
    assert report.architecture_rule_id is None
    assert report.pipeline is PipelineKind.UNKNOWN
    assert any("No registry rule matches this architecture" in note for note in report.notes)


# --------------------------------------------------------------------------- #
# How the architecture was identified
# --------------------------------------------------------------------------- #


def test_an_architecture_identified_from_a_hint_says_so_in_the_report(
    gguf_with_header: ModelMetadata,
) -> None:
    """A hint-based identification is a typed field, not a sentence to grep.

    Every device claim below descends from this match, so ``--json`` consumers
    must be able to see that the chain starts at a GGUF header rather than at a
    ``config.json`` the repository never published.
    """
    report = build_service(FakeHub({REQUESTED_REPO: gguf_with_header})).inspect(REQUESTED_REPO)

    assert report.architecture_rule_id == "arch-qwen3_5"
    assert report.architecture == "qwen3_5"
    assert report.architecture_identified_from == "GGUF metadata"

    payload = json.loads(json.dumps(report.to_json()))
    assert payload["architecture_identified_from"] == "GGUF metadata"


def test_a_declared_architecture_is_never_reported_as_hint_identified(
    qwen_safetensors: ModelMetadata,
) -> None:
    """A repository that declares its own ``model_type`` carries no caveat."""
    report = build_service(FakeHub({DERIVATIVE_REPO: qwen_safetensors})).inspect(DERIVATIVE_REPO)

    assert report.architecture_rule_id == "arch-qwen3_5"
    assert report.architecture_identified_from is None


def test_an_unmatched_architecture_claims_no_identification(
    gguf_only: ModelMetadata,
) -> None:
    """Nothing was identified, so nothing is claimed about how."""
    report = build_service(FakeHub({REQUESTED_REPO: gguf_only})).inspect(REQUESTED_REPO)

    assert report.architecture_rule_id is None
    assert report.architecture_identified_from is None


# --------------------------------------------------------------------------- #
# Compatibility is evidence, not inference
# --------------------------------------------------------------------------- #


def test_compatibility_is_reported_per_device_with_its_evidence(
    qwen_safetensors: ModelMetadata,
) -> None:
    """Qwen3.5 on this host: GPU/CPU supported, NPU unsupported, each with a rule."""
    report = build_service(FakeHub({DERIVATIVE_REPO: qwen_safetensors})).inspect(DERIVATIVE_REPO)

    assert report.architecture_rule_id == "arch-qwen3_5"
    assert [entry.device for entry in report.compatibility] == ["CPU", "GPU", "NPU"]

    npu = report.compatibility_for("NPU")
    assert npu is not None
    assert npu.status is DeviceStatus.UNSUPPORTED
    assert npu.evidence is EvidenceLevel.TESTED
    assert npu.rule_id == "arch-qwen3_5"
    assert npu.reason == "current_npu_compiler_graph_limitation"

    gpu = report.compatibility_for("GPU")
    assert gpu is not None
    assert gpu.status is DeviceStatus.SUPPORTED
    assert gpu.evidence is EvidenceLevel.OFFICIAL

    assert report.recommended_device == "GPU"
    assert report.usable_devices == ["CPU", "GPU"]


def test_a_present_npu_is_never_reported_as_supported_without_a_claim() -> None:
    """ "The NPU exists" is not evidence. An unclaimed device stays unknown."""
    # mistral: covered only by the generic rule, which claims nothing about the NPU.
    metadata = build_metadata(
        "acme/llama-ish",
        (("config.json", 2_000), ("model.safetensors", 4_000_000_000)),
        resolved_revision="a" * 40,
        architectures=["MistralForCausalLM"],
        model_type="mistral",
    )

    report = build_service(FakeHub({"acme/llama-ish": metadata})).inspect("acme/llama-ish")

    assert report.architecture_rule_id == "arch-generic-text-generation"
    npu = report.compatibility_for("NPU")
    assert npu is not None
    assert npu.status is DeviceStatus.UNKNOWN
    assert npu.evidence is EvidenceLevel.UNKNOWN
    assert npu.reason == "not_verified_for_this_family"
    assert "NPU" in report.host_devices


def test_an_unknown_architecture_leaves_every_device_unknown() -> None:
    """With no matching rule, nothing is claimed about any device."""
    metadata = build_metadata(
        "acme/mystery",
        (("config.json", 2_000), ("model.safetensors", 1_000_000)),
        resolved_revision="b" * 40,
        architectures=["MysteryForCausalLM"],
        model_type="mystery",
    )

    report = build_service(FakeHub({"acme/mystery": metadata})).inspect("acme/mystery")

    assert report.architecture_rule_id is None
    assert {entry.status for entry in report.compatibility} == {DeviceStatus.UNKNOWN}
    assert all(
        "no claim about this device" in " ".join(entry.notes) for entry in report.compatibility
    )


def test_a_device_the_registry_knows_about_is_reported_even_when_absent(
    qwen_safetensors: ModelMetadata,
) -> None:
    """A host without an NPU still learns that Qwen3.5 has no NPU path."""
    hardware = FakeHardware(build_inventory(devices=("CPU", "GPU")))

    report = build_service(FakeHub({DERIVATIVE_REPO: qwen_safetensors}), hardware=hardware).inspect(
        DERIVATIVE_REPO
    )

    npu = report.compatibility_for("NPU")
    assert npu is not None
    assert npu.status is DeviceStatus.UNSUPPORTED
    assert "OpenVINO does not expose this device on this host." in npu.notes
    assert "NPU" not in report.host_devices


def test_a_configured_default_device_that_is_unsupported_is_called_out(
    qwen_safetensors: ModelMetadata,
) -> None:
    """Pinning ``device = npu`` for a model with no NPU path earns a warning."""
    report = build_service(
        FakeHub({DERIVATIVE_REPO: qwen_safetensors}),
        settings=Settings(default_device=DevicePolicy.NPU),
    ).inspect(DERIVATIVE_REPO)

    assert report.configured_device == "NPU"
    assert any("configured default device NPU" in note for note in report.notes)


def test_auto_device_policy_records_no_configured_device(
    qwen_safetensors: ModelMetadata,
) -> None:
    """``auto`` means the plan decides later, so nothing is pinned here."""
    report = build_service(FakeHub({DERIVATIVE_REPO: qwen_safetensors})).inspect(DERIVATIVE_REPO)

    assert report.configured_device is None


# --------------------------------------------------------------------------- #
# Exporter requirements
# --------------------------------------------------------------------------- #


def test_exporter_requirements_come_from_the_architecture_rule(
    qwen_safetensors: ModelMetadata,
) -> None:
    """This host really does have the wrong Transformers for a Qwen3.5 export."""
    report = build_service(FakeHub({DERIVATIVE_REPO: qwen_safetensors})).inspect(DERIVATIVE_REPO)

    assert report.exporter.transformers == "==5.2.*"
    assert report.exporter.satisfied is False
    problem = " ".join(report.exporter.problems)
    assert "transformers ==5.2.*" in problem
    assert "5.5.4 is installed" in problem
    assert any("5.5.4 is installed" in note for note in report.notes)


def test_exporter_requirements_are_satisfied_on_a_matching_host(
    qwen_safetensors: ModelMetadata,
) -> None:
    """The same rule passes once the declared version is installed."""
    versions = dict(HOST_VERSIONS)
    versions["transformers"] = "5.2.1"
    resolver = RegistryResolver(load_builtin_registry(), versions=versions)

    report = build_service(FakeHub({DERIVATIVE_REPO: qwen_safetensors}), resolver=resolver).inspect(
        DERIVATIVE_REPO
    )

    assert report.exporter.satisfied is True
    assert report.exporter.problems == []


def test_architectures_without_requirements_report_none(gguf_only: ModelMetadata) -> None:
    """No matching rule means no declared requirements, not an invented pass."""
    report = build_service(FakeHub({REQUESTED_REPO: gguf_only})).inspect(REQUESTED_REPO)

    assert report.exporter.specifiers() == {}
    assert report.exporter.satisfied is True


def test_known_limitations_are_carried_from_the_registry(
    qwen_safetensors: ModelMetadata,
) -> None:
    """The architecture rule's limitations reach the report verbatim."""
    report = build_service(FakeHub({DERIVATIVE_REPO: qwen_safetensors})).inspect(DERIVATIVE_REPO)

    assert any("Transformers 5.2" in limitation for limitation in report.known_limitations)
    assert report.pipeline is PipelineKind.VLM


# --------------------------------------------------------------------------- #
# Source options, trust and notes
# --------------------------------------------------------------------------- #


def test_source_options_follow_the_preference_order(gguf_only: ModelMetadata) -> None:
    """The GGUF fixture offers no direct route and never claims GGUF support."""
    report = build_service(FakeHub({REQUESTED_REPO: gguf_only})).inspect(REQUESTED_REPO)

    safetensors = report.option(OPTION_SAFETENSORS)
    assert safetensors is not None
    assert safetensors.available is False

    gguf = report.option(OPTION_GGUF)
    assert gguf is not None
    assert gguf.available is False
    assert "not evidence" in (gguf.detail or "")


def test_remote_code_is_flagged_without_enabling_anything() -> None:
    """A repository with custom code is reported; nothing is trusted here."""
    metadata = build_metadata(
        "acme/custom-code",
        (("config.json", 2_000), ("model.safetensors", 1_000_000)),
        resolved_revision="c" * 40,
        requires_remote_code=True,
        remote_code_files=["modeling_custom.py"],
    )

    report = build_service(FakeHub({"acme/custom-code": metadata})).inspect("acme/custom-code")

    assert report.trust_remote_code_required is True
    note = " ".join(report.notes)
    assert "--trust-remote-code" in note
    assert "--yes will not enable it" in note


def test_gated_repositories_and_offline_mode_are_noted(
    qwen_safetensors: ModelMetadata,
) -> None:
    """Access and freshness caveats belong in the report, not in a surprise later."""
    gated = qwen_safetensors.model_copy(update={"gated": True})

    report = build_service(
        FakeHub({DERIVATIVE_REPO: gated}), settings=Settings(offline=True)
    ).inspect(DERIVATIVE_REPO)

    assert report.offline is True
    notes = " ".join(report.notes)
    assert "gated or private" in notes
    assert "Offline mode is enabled" in notes


def test_missing_openvino_is_noted(qwen_safetensors: ModelMetadata) -> None:
    """Without OpenVINO the device list is registry-only, and says so."""
    inventory = build_inventory(devices=())
    hardware = FakeHardware(
        inventory.model_copy(update={"openvino_available": False, "devices": []})
    )

    report = build_service(FakeHub({DERIVATIVE_REPO: qwen_safetensors}), hardware=hardware).inspect(
        DERIVATIVE_REPO
    )

    assert any("OpenVINO is not available to yonga" in note for note in report.notes)
    assert report.host_devices == []


# --------------------------------------------------------------------------- #
# Serialisation and layering
# --------------------------------------------------------------------------- #


def test_report_is_json_serialisable(qwen_safetensors: ModelMetadata) -> None:
    """``yonga inspect --json`` prints the whole report."""
    report = build_service(FakeHub({DERIVATIVE_REPO: qwen_safetensors})).inspect(DERIVATIVE_REPO)

    payload = json.loads(json.dumps(report.to_json()))

    assert payload["resolved_revision"] == DERIVATIVE_SHA
    assert payload["metadata"]["identity"]["repo_id"] == DERIVATIVE_REPO
    assert payload["pipeline"] == "vlm"
    assert payload["compatibility"][2]["status"] == "unsupported"


def test_report_renders_through_the_ui_table(qwen_safetensors: ModelMetadata) -> None:
    """The report feeds :func:`yonga.ui.tables.inspect_report` without translation."""
    from yonga.ui.tables import SourceOption as UiSourceOption
    from yonga.ui.tables import inspect_report

    report = build_service(FakeHub({DERIVATIVE_REPO: qwen_safetensors})).inspect(DERIVATIVE_REPO)
    options = [UiSourceOption.model_validate(item.model_dump()) for item in report.source_options]

    assert inspect_report(report.metadata, None, options) is not None


def test_report_model_is_constructible_from_its_own_dump(
    qwen_safetensors: ModelMetadata,
) -> None:
    """The report round-trips, so a job can store and replay it."""
    report = build_service(FakeHub({DERIVATIVE_REPO: qwen_safetensors})).inspect(DERIVATIVE_REPO)

    assert InspectReport.model_validate(report.model_dump()) == report


def test_fakes_satisfy_the_adapter_protocols() -> None:
    """The fixtures exercise the same ports the CLI injects."""
    assert isinstance(FakeHub(), HubClient)
    assert isinstance(FakeHardware(), HardwareProbe)


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


def test_inspect_service_imports_no_adapter_library() -> None:
    """The application layer stays free of renderers, Hub and inference stacks."""
    roots = imported_roots(Path(inspect_module.__file__))
    assert roots & FORBIDDEN_IMPORT_ROOTS == set()
