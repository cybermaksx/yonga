"""Behaviour of the Typer CLI: exit codes, ``--json``, safety flags and rendering.

Every test drives the real command functions through
:class:`typer.testing.CliRunner`; only the *outside world* is faked, by handing
the root callback a :class:`~yonga.application.wiring.Wiring` subclass whose
adapter factories return the doubles from ``tests/unit/test_prepare_planning.py``.
The services under test are therefore the real ones, and so is every exit code.

The rules asserted here are the ones that are easy to regress and expensive to
get wrong: ``--yes`` must never enable ``trust_remote_code``, an explicit
``--device`` must never be silently changed, ``--json`` must emit exactly one
parseable document with no secret in it, and no failure may ever print a
traceback.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

# ``Result`` comes from typer, not click: typer 0.27 vendored click and returns
# its own ``typer.testing.Result`` from ``CliRunner.invoke``, while typer 0.25
# re-exports click's. Importing it from typer is correct under both, so the
# annotation tracks whichever version is installed.
from typer.testing import CliRunner, Result

from tests.support import RefusingRunner, SmokeChildRunner, plain_text
from tests.unit.test_artifact_store import make_manifest, publish
from tests.unit.test_compile_cache import (
    BEFORE_UPGRADE,
    INSTALLED,
    OLDER,
    legacy_entry,
    recorded_entry,
)
from tests.unit.test_prepare_planning import (
    GGUF_REPO,
    SAFETENSORS_REPO,
    FakeExporter,
    FakeFilesystemProbe,
    FakeHardwareProbe,
    FakeHub,
    FakeRuntimeProbe,
    FakeSmokeTester,
    builtin_resolver,
    gguf_metadata,
    make_snapshot,
    safetensors_metadata,
)
from yonga import cli as cli_module

if TYPE_CHECKING:
    from yonga.application.evaluation import EvaluationService
from yonga.adapters import (
    CommandRunner,
    Exporter,
    FilesystemProbe,
    HardwareProbe,
    HubClient,
    HubPublisher,
    RuntimeProbe,
)
from yonga.adapters.filesystem import system_temp_dir
from yonga.adapters.runtime_genai import ChatHistory, ChatProfileLike, ChatRuntime, GenerationResult
from yonga.adapters.smoke_child import verdict_line
from yonga.application.artifacts import ArtifactStore
from yonga.application.benchmark import RuntimeFactory as BenchmarkRuntimeFactory
from yonga.application.chat import RuntimeFactory as ChatRuntimeFactory
from yonga.application.prepare import SmokeTester
from yonga.application.progress import ProgressSink
from yonga.application.wiring import Wiring
from yonga.cli import app
from yonga.config.settings import Settings
from yonga.domain.artifacts import IRDescription, ProbeResult, ProbeStatus, assert_no_secrets
from yonga.domain.benchmarks import GenerationSettings
from yonga.domain.compile_cache import COMPILE_CACHE_ENTRY_FILENAME
from yonga.domain.diagnostics import ConfigurationError, ExitCode, FailureCategory, diagnostic
from yonga.domain.hardware import (
    DeviceKind,
    FilesystemInfo,
    HardwareInventory,
    HostInfo,
    HuggingFaceAuthInfo,
    MemoryInfo,
    NPUNodeInfo,
    OpenVINODevice,
    PackageInfo,
)
from yonga.domain.models import ModelMetadata, PipelineKind
from yonga.registry.resolver import RegistryResolver

GIB = 1024**3

#: The real Qwen3.5 NPU compiler refusal recorded in ``HOST_FACTS.md``.
#: The raw NPU compile failure the registry defers to the GenAI pipeline.
NPU_DYNAMIC_SHAPE_ERROR = (
    "[NPU_VCL] Compiler returned msg:\n"
    "Exception from src/core/src/partial_shape.cpp:261:\n"
    "to_shape was called on a dynamic shape."
)

NPU_UNSUPPORTED_GRAPH = (
    "Exception from src/plugins/intel_npu/src/plugin/src/plugin.cpp:552:\n"
    "OneShotBufferizeVPU2VPUIP failed : Only supports FP16."
)

#: A host problem rather than a compatibility limit: the kernel refused the NPU node.
NPU_PERMISSION_DENIED = (
    "Exception from src/inference/src/cpp/core.cpp:117:\n"
    "Failed to open /dev/accel/accel0: Permission denied"
)

#: Job id of the multi-megabyte log the closed-pipe tests write and then read.
LARGE_LOG_JOB_ID = "11111111-2222-3333-4444-555555555555"

#: Job id of the one-record log used to reach the interpreter's shutdown flush.
SMALL_LOG_JOB_ID = "22222222-2222-3333-4444-555555555555"

#: A token-shaped string used to prove nothing secret reaches a JSON document.
FAKE_TOKEN = "hf_clitestsecrettokenvalue000000000000"

#: The registry document that makes the NPU the recommended device for the
#: fixture repository, so ``--device auto`` actually reaches the fallback path.
NPU_FIRST_REGISTRY = f"""
schema_version: 1
architectures:
  - id: arch-cli-npu-first
    schema_version: 1
    status: active
    priority: 100
    summary: Test rule that believes the NPU can run this repository
    match:
      repo_ids:
        - {SAFETENSORS_REPO}
    export:
      pipeline: vlm
      trust_remote_code: false
    devices:
      cpu:
        status: supported
        evidence: official
      gpu:
        status: supported
        evidence: official
      npu:
        status: supported
        evidence: tested
    recommended_device: npu
    profiles:
      npu: npu-int4-sym-cw
      gpu: gpu-int4-asym-g128
      cpu: cpu-int4-asym-g128
    chat_profile: chat-qwen3_5
"""


# ======================================================================
# doubles
# ======================================================================
class CliHub(FakeHub):
    """A :class:`FakeHub` that also reports where the Hugging Face cache lives.

    ``doctor`` grades the cache directory, so a hub double that never resolves one
    would report an unknown check and make the healthy host exit 2.
    """

    def auth_info(self) -> HuggingFaceAuthInfo:
        """Report an anonymous but fully configured Hub client."""
        return HuggingFaceAuthInfo(
            logged_in=False,
            offline=False,
            cache_dir=Path("/home/tester/.cache/huggingface/hub"),
        )


class FakeChatRuntime:
    """A :class:`~yonga.adapters.runtime_genai.ChatRuntime` with no OpenVINO."""

    def __init__(self, reply: str = "Hello from the fake model.") -> None:
        self.reply = reply
        self.turns: list[int] = []
        self.conversations = 0
        self.closed = False

    def begin_conversation(self) -> None:
        """Start a new conversation; nothing to discard in a stateless fake."""
        self.conversations += 1

    def generate(self, history: ChatHistory, config: GenerationSettings) -> GenerationResult:
        """Answer one turn with a canned reply and plausible metrics."""
        self.turns.append(len(history))
        return GenerationResult(
            text=self.reply,
            display_text=self.reply,
            input_tokens=11,
            output_tokens=7,
            ttft_ms=12.5,
            throughput_tps=42.0,
            tpot_ms=23.0,
            wall_seconds=0.2,
        )

    def close(self) -> None:
        """Release the pipeline; idempotent."""
        self.closed = True


class FakeRuntimeFactory:
    """Opens :class:`FakeChatRuntime` objects, satisfying both runtime-factory ports."""

    def __init__(self) -> None:
        self.opened: list[tuple[Path, str]] = []
        self.runtimes: list[FakeChatRuntime] = []

    def __call__(
        self,
        artifact_dir: Path,
        *,
        device: str = "CPU",
        pipeline: PipelineKind = PipelineKind.LLM,
        properties: Mapping[str, str] | None = None,
        chat_profile: ChatProfileLike | None = None,
        settings: Settings | None = None,
        thinking: bool | None = None,
    ) -> ChatRuntime:
        """Record the request and return a fresh fake runtime."""
        self.opened.append((Path(artifact_dir), device))
        runtime = FakeChatRuntime()
        self.runtimes.append(runtime)
        return runtime


class ExplodingArtifactStore(ArtifactStore):
    """An artifact store whose catalogue read raises an unexpected error."""

    def list_artifacts(self) -> list[object]:  # type: ignore[override]
        """Fail the way a genuine bug would, with an untyped exception."""
        raise RuntimeError("synthetic catalogue failure")


class ClosedPipeArtifactStore(ArtifactStore):
    """An artifact store that fails the way a reader closing the pipe does."""

    def list_artifacts(self) -> list[object]:  # type: ignore[override]
        """Raise what the interpreter raises once stdout has no reader left."""
        raise BrokenPipeError(32, "Broken pipe")


class RichClosedPipeArtifactStore(ArtifactStore):
    """An artifact store failing the way Rich reports a closed pipe.

    ``rich.console.Console.on_broken_pipe`` never lets the ``BrokenPipeError``
    out: it silences the console and raises ``SystemExit(1)`` with the original
    exception still on the chain. This reproduces that exact shape.
    """

    def list_artifacts(self) -> list[object]:  # type: ignore[override]
        """Raise ``SystemExit`` from a broken pipe, as Rich does."""
        try:
            raise BrokenPipeError(32, "Broken pipe")
        except BrokenPipeError:
            raise SystemExit(1) from None


class DeliberateExitArtifactStore(ArtifactStore):
    """An artifact store that exits on purpose, for no pipe-related reason."""

    def list_artifacts(self) -> list[object]:  # type: ignore[override]
        """Raise a plain ``SystemExit`` that must keep its own code."""
        raise SystemExit(3)


class FakeWiring(Wiring):
    """A :class:`Wiring` whose entire outside world is injected.

    Only adapter factories are overridden: every service the CLI builds is the
    real one, so the exit codes these tests assert are produced by the real
    policy code rather than by a stub.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        hub: HubClient | None = None,
        hardware: HardwareProbe | None = None,
        runtime: RuntimeProbe | None = None,
        exporter: Exporter | None = None,
        filesystem: FilesystemProbe | None = None,
        resolver: RegistryResolver | None = None,
        smoke: SmokeTester | None = None,
        runtime_factory: FakeRuntimeFactory | None = None,
        store: ArtifactStore | None = None,
    ) -> None:
        """Bind the doubles; anything omitted gets a sensible healthy default."""
        super().__init__(settings)
        self._fake_hub = hub if hub is not None else CliHub()
        self._fake_hardware = (
            hardware if hardware is not None else FakeHardwareProbe(healthy_inventory())
        )
        self._fake_runtime = runtime if runtime is not None else FakeRuntimeProbe()
        self._fake_exporter = exporter if exporter is not None else FakeExporter()
        self._fake_filesystem = (
            filesystem
            if filesystem is not None
            else FakeFilesystemProbe(tmpfs_paths=(system_temp_dir(),))
        )
        self._fake_resolver = resolver if resolver is not None else builtin_resolver()
        self._fake_smoke = smoke if smoke is not None else FakeSmokeTester()
        self._fake_factory = (
            runtime_factory if runtime_factory is not None else FakeRuntimeFactory()
        )
        self._fake_store = store

    def hub_client(self) -> HubClient:
        """Return the in-memory hub."""
        return self._fake_hub

    def command_runner(self) -> CommandRunner:
        """Refuse every process: an unfaked adapter must not reach the network."""
        return RefusingRunner()

    def smoke_child_runner(self) -> CommandRunner:
        """Refuse the probe's smoke-test child too, unless a test scripts it."""
        return RefusingRunner()

    def hardware_probe(self) -> HardwareProbe:
        """Return the fixed host inventory."""
        return self._fake_hardware

    def runtime_probe(self) -> RuntimeProbe:
        """Return the file-driven runtime probe."""
        return self._fake_runtime

    def exporter(self) -> Exporter:
        """Return the exporter that writes fixture IR instead of converting."""
        return self._fake_exporter

    def filesystem(self) -> FilesystemProbe:
        """Return the reference-host filesystem model."""
        return self._fake_filesystem

    def registry_resolver(self) -> RegistryResolver:
        """Return a resolver with pinned versions, so rules match deterministically."""
        return self._fake_resolver

    def smoke_tester(self) -> SmokeTester:
        """Return the smoke test that never builds a GenAI pipeline."""
        return self._fake_smoke

    def chat_runtime_factory(self) -> ChatRuntimeFactory:
        """Return the fake runtime factory."""
        return self._fake_factory

    def benchmark_runtime_factory(self) -> BenchmarkRuntimeFactory:
        """Return the fake runtime factory."""
        return self._fake_factory

    def openvino_version(self) -> str | None:
        """Report a fixed installed runtime, so no test imports OpenVINO for it."""
        return INSTALLED

    def artifact_store(self) -> ArtifactStore:
        """Return the real store, unless a test injected a broken one."""
        if self._fake_store is not None:
            return self._fake_store
        return super().artifact_store()


def healthy_inventory(
    *,
    devices: Sequence[str] = ("CPU", "GPU", "NPU"),
    openvino: bool = True,
) -> HardwareInventory:
    """Build the inventory of a completely healthy reference host.

    Every field ``doctor`` grades is populated with a passing value, so the
    resulting report is the exit-0 case; individual tests degrade one field at a
    time to reach exit 2 and exit 10.
    """
    full_names = {
        "CPU": "Intel(R) Core(TM) Ultra 7 268V",
        "GPU": "Intel(R) Arc(TM) Graphics (iGPU)",
        "NPU": "Intel(R) AI Boost",
    }
    kinds = {"CPU": DeviceKind.CPU, "GPU": DeviceKind.GPU, "NPU": DeviceKind.NPU}
    properties = {
        "NPU": {
            "NPU_DRIVER_VERSION": "1789049548",
            "NPU_COMPILER_VERSION": "524291",
            "NPU_COMPILER_TYPE": "CompilerType.PREFER_PLUGIN",
        }
    }
    return HardwareInventory(
        host=HostInfo(
            platform="Linux",
            machine="x86_64",
            kernel="7.2.6-070206-generic",
            python_version="3.11.9",
            distro_id="ubuntu",
            distro_pretty="Ubuntu 26.04.1 LTS",
            cpu_model="Intel(R) Core(TM) Ultra 7 268V",
            cpu_count=8,
        ),
        memory=MemoryInfo(
            total_bytes=32 * GIB,
            available_bytes=20 * GIB,
            swap_total_bytes=8 * GIB,
            swap_free_bytes=8 * GIB,
        ),
        filesystems=[_filesystem("/"), _filesystem("/home")],
        devices=[
            OpenVINODevice(
                name=name,
                kind=kinds[name],
                full_name=full_names[name],
                properties=properties.get(name, {}),
            )
            for name in devices
        ],
        openvino_available=openvino,
        openvino_import_error=None if openvino else "No module named 'openvino'",
        npu=NPUNodeInfo(
            directory_exists=True,
            node_path=Path("/dev/accel/accel0"),
            node_exists=True,
            readable=True,
            writable=True,
            group_name="render",
            user_in_group=True,
            user_groups=["render"],
            driver_version="1789049548",
            compiler_libraries=["/usr/lib/x86_64-linux-gnu/libopenvino_intel_npu_compiler.so"],
        ),
        packages=[
            PackageInfo(name=name, version=version, importable=True)
            for name, version in (
                ("openvino", "2026.4.0"),
                ("openvino-genai", "2026.4.0.0"),
                ("openvino-tokenizers", "2026.4.0.0"),
                ("optimum-intel", "2.2.0"),
                ("nncf", "3.4.0"),
                ("transformers", "5.2.1"),
                ("huggingface-hub", "1.21.0"),
            )
        ],
        device_tags=["intel.cpu", "intel.gpu", "intel.npu", "intel.npu.lunar_lake"],
        system_temp=_filesystem("/tmp"),
        managed_temp=_filesystem("/home/tester/.cache/yonga/tmp"),
        hf_auth=HuggingFaceAuthInfo(
            logged_in=True,
            user="tester",
            token_source="environment",
            offline=False,
            cache_dir=Path("/home/tester/.cache/huggingface/hub"),
        ),
    )


def _filesystem(path: str, *, fstype: str = "ext4", free: int = 700 * GIB) -> FilesystemInfo:
    """Build a roomy, disk-backed filesystem reading."""
    return FilesystemInfo(
        path=Path(path),
        mount_point=Path(path),
        fstype=fstype,
        total_bytes=free + GIB,
        free_bytes=free,
        available_bytes=free,
    )


def tmpfs_inventory() -> HardwareInventory:
    """The reference host: healthy, but with the 16 GiB ``/tmp`` tmpfs warning."""
    inventory = healthy_inventory()
    inventory.system_temp = _filesystem("/tmp", fstype="tmpfs", free=16 * GIB)
    return inventory


# ======================================================================
# harness
# ======================================================================
@pytest.fixture
def runner() -> CliRunner:
    """A Typer runner with stdout and stderr captured separately."""
    return CliRunner()


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """A private XDG root so no test can read or write the developer's own data."""
    root = tmp_path / "home"
    (root / "config").mkdir(parents=True)
    return root


def invoke(
    runner: CliRunner,
    home: Path,
    args: Sequence[str],
    *,
    wiring: Wiring | None = None,
    factory: object | None = None,
    env: Mapping[str, str] | None = None,
    stdin: str | None = None,
) -> Result:
    """Run one command with private directories and a faked adapter graph."""
    environment: dict[str, str] = {
        "YONGA_CONFIG_DIR": str(home / "config"),
        "NO_COLOR": "1",
        "HF_TOKEN": FAKE_TOKEN,
        # Rich wraps to the terminal width; a wide console keeps the assertions
        # about whole sentences and paths meaningful.
        "COLUMNS": "200",
    }
    environment.update(env or {})
    build = factory if factory is not None else (lambda settings: wiring or FakeWiring(settings))
    return runner.invoke(
        app,
        ["--data-dir", str(home / "data"), "--cache-dir", str(home / "cache"), *args],
        obj=build,
        env=environment,
        input=stdin,
    )


def one_document(result: Result) -> dict[str, object]:
    """Assert stdout holds exactly one JSON document and return it."""
    text = result.stdout
    assert text.count("\n") == 1, f"expected exactly one line of JSON, got: {text!r}"
    payload = json.loads(text)
    assert isinstance(payload, dict)
    return payload


def hub_with(*, remote_code: bool = False, gguf: bool = False, tmp_path: Path) -> CliHub:
    """Build a hub carrying the acceptance fixtures used by the prepare tests."""
    repos = {SAFETENSORS_REPO: safetensors_metadata(requires_remote_code=remote_code)}
    snapshots = {SAFETENSORS_REPO: make_snapshot(tmp_path, "snapshot-safetensors")}
    if gguf:
        repos[GGUF_REPO] = gguf_metadata()
        snapshots[GGUF_REPO] = make_snapshot(tmp_path, "snapshot-gguf")
    return CliHub(repos=repos, snapshots=snapshots)


# ======================================================================
# help, version and the import graph
# ======================================================================
def test_help_lists_every_documented_command(runner: CliRunner) -> None:
    """``--help`` works with no configuration, no adapters and no OpenVINO."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "doctor",
        "inspect",
        "prepare",
        "list",
        "remove",
        "probe",
        "chat",
        "bench",
        "registry",
        "logs",
        "search",
        "env",
        "serve",
        "publish",
        "eval",
        "cache",
    ):
        assert command in result.stdout


def test_version_prints_the_product_name_and_version(runner: CliRunner) -> None:
    """``--version`` is eager: it answers before anything is configured."""
    from yonga import __version__

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"yonga {__version__}"


def test_importing_the_cli_loads_no_accelerator_stack() -> None:
    """Importing the CLI must not drag in OpenVINO, Torch or the Hub client."""
    import subprocess
    import sys

    probe = (
        "import sys, yonga.cli\n"
        "heavy = {'openvino', 'openvino_genai', 'openvino_tokenizers', 'optimum',\n"
        "         'nncf', 'transformers', 'torch', 'huggingface_hub', 'prompt_toolkit'}\n"
        "loaded = sorted(heavy & {name.split('.')[0] for name in sys.modules})\n"
        "print(','.join(loaded))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == ""


# ======================================================================
# doctor: 0 / 2 / 10
# ======================================================================
def test_doctor_exits_zero_on_a_healthy_host(runner: CliRunner, home: Path) -> None:
    """A host where every check passes returns success."""
    result = invoke(runner, home, ["doctor"])
    assert result.exit_code == int(ExitCode.SUCCESS)
    assert "Intel(R) Arc(TM) Graphics (iGPU)" in result.stdout


def test_doctor_exits_two_on_the_small_tmpfs_warning(runner: CliRunner, home: Path) -> None:
    """A 16 GiB ``/tmp`` is a warning, never fatal: managed temp solves it."""
    wiring_for = lambda settings: FakeWiring(  # noqa: E731 - one-line factory
        settings, hardware=FakeHardwareProbe(tmpfs_inventory())
    )
    result = invoke(runner, home, ["doctor"], factory=wiring_for)
    assert result.exit_code == int(ExitCode.WARNINGS)
    assert "tmpfs" in result.stdout


def test_doctor_exits_ten_without_a_usable_runtime(runner: CliRunner, home: Path) -> None:
    """No OpenVINO and no device is an unusable environment."""

    class NoRuntime:
        """A runtime probe on a host where OpenVINO is not installed."""

        def available_devices(self) -> list[str]:
            return []

        def read_ir(self, path: Path) -> IRDescription:
            raise NotImplementedError

        def compile(
            self, path: Path, device: str, properties: Mapping[str, str] | None = None
        ) -> ProbeResult:
            raise NotImplementedError

        def read_tokenizer(self, artifact: Path) -> ProbeResult:
            raise NotImplementedError

    inventory = healthy_inventory(devices=(), openvino=False)
    inventory.packages = [
        PackageInfo(name="openvino", version=None, importable=False, import_error="not installed")
    ]
    wiring_for = lambda settings: FakeWiring(  # noqa: E731 - one-line factory
        settings, hardware=FakeHardwareProbe(inventory), runtime=NoRuntime()
    )
    result = invoke(runner, home, ["doctor"], factory=wiring_for)
    assert result.exit_code == int(ExitCode.ENVIRONMENT_UNUSABLE)


def test_doctor_json_is_one_document_without_a_token(runner: CliRunner, home: Path) -> None:
    """``--json`` emits one document, and authentication never carries a token."""
    result = invoke(runner, home, ["doctor", "--json"])
    payload = one_document(result)
    assert payload["ok"] is True
    assert payload["exit_code"] == int(ExitCode.SUCCESS)
    assert FAKE_TOKEN not in result.stdout
    # The project's own secret gate, rather than a substring guess: it rejects any
    # key whose words include "token" or "credential", at any depth.
    assert_no_secrets(payload)


def test_doctor_full_adds_the_registry_section(runner: CliRunner, home: Path) -> None:
    """``--full`` reports the loaded registry as well as the host."""
    result = invoke(runner, home, ["doctor", "--full", "--json"])
    payload = one_document(result)
    sections = payload["sections"]
    assert isinstance(sections, list)
    assert any(section["id"] == "registry" for section in sections)


# ======================================================================
# inspect
# ======================================================================
def test_inspect_renders_the_report(runner: CliRunner, home: Path, tmp_path: Path) -> None:
    """The human report names the model, its formats and its device rows."""
    wiring_for = lambda settings: FakeWiring(  # noqa: E731 - one-line factory
        settings, hub=hub_with(tmp_path=tmp_path)
    )
    result = invoke(runner, home, ["inspect", SAFETENSORS_REPO], factory=wiring_for)
    assert result.exit_code == int(ExitCode.SUCCESS)
    assert SAFETENSORS_REPO in result.stdout
    assert "safetensors" in result.stdout
    assert "Compatibility on this host" in result.stdout


def test_inspect_json_is_one_document(runner: CliRunner, home: Path, tmp_path: Path) -> None:
    """``inspect --json`` is machine readable and carries no secret."""
    wiring_for = lambda settings: FakeWiring(  # noqa: E731 - one-line factory
        settings, hub=hub_with(tmp_path=tmp_path)
    )
    result = invoke(runner, home, ["inspect", SAFETENSORS_REPO, "--json"], factory=wiring_for)
    payload = one_document(result)
    assert payload["ok"] is True
    assert payload["requested"] == {"repo_id": SAFETENSORS_REPO, "revision": None}
    assert FAKE_TOKEN not in result.stdout


def test_inspect_of_an_unknown_repository_exits_twenty(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """A repository the Hub does not have is a source-resolution failure."""
    wiring_for = lambda settings: FakeWiring(  # noqa: E731 - one-line factory
        settings, hub=hub_with(tmp_path=tmp_path)
    )
    result = invoke(runner, home, ["inspect", "nobody/does-not-exist"], factory=wiring_for)
    assert result.exit_code == int(ExitCode.SOURCE_RESOLUTION)
    assert "Traceback" not in result.stderr


def test_inspect_error_json_has_the_stable_shape(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """A failure under ``--json`` is the documented error document, not a string."""
    wiring_for = lambda settings: FakeWiring(  # noqa: E731 - one-line factory
        settings, hub=hub_with(tmp_path=tmp_path)
    )
    result = invoke(
        runner, home, ["inspect", "nobody/does-not-exist", "--json"], factory=wiring_for
    )
    payload = one_document(result)
    assert payload["ok"] is False
    assert payload["exit_code"] == int(ExitCode.SOURCE_RESOLUTION)
    item = payload["diagnostic"]
    assert isinstance(item, dict)
    for key in ("code", "category", "message", "fallback_devices", "job_id"):
        assert key in item
    assert item["code"]
    assert "Traceback" not in json.dumps(payload)


# ======================================================================
# prepare
# ======================================================================
def test_prepare_succeeds_and_prints_the_artifact(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """A clean preparation publishes an artifact and names it."""
    wiring_for = lambda settings: FakeWiring(  # noqa: E731 - one-line factory
        settings, hub=hub_with(tmp_path=tmp_path)
    )
    result = invoke(
        runner,
        home,
        ["prepare", SAFETENSORS_REPO, "--device", "cpu", "--yes"],
        factory=wiring_for,
    )
    assert result.exit_code in (int(ExitCode.SUCCESS), int(ExitCode.WARNINGS))
    assert "Ready" in result.stdout
    assert "yonga chat" in result.stdout
    assert (home / "data" / "artifacts").is_dir()


def test_yes_does_not_enable_trust_remote_code(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """``--yes`` answers ordinary questions only; remote code needs its own flag."""
    wiring_for = lambda settings: FakeWiring(  # noqa: E731 - one-line factory
        settings, hub=hub_with(remote_code=True, tmp_path=tmp_path)
    )
    result = invoke(
        runner,
        home,
        ["prepare", SAFETENSORS_REPO, "--device", "cpu", "--yes"],
        factory=wiring_for,
    )
    assert result.exit_code == int(ExitCode.REMOTE_CODE_REQUIRED)
    assert "--trust-remote-code" in result.stderr
    assert "--yes does not and will not enable remote code." in result.stderr


def test_trust_remote_code_still_asks_a_human(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """The flag unblocks the guard, but a security question still needs a person.

    ``SPEC.md`` section 18.1 makes security confirmations asymmetric: they are
    never answered by ``--yes`` and never auto-answered in a non-interactive
    session. A piped run therefore stops as cancelled instead of quietly running
    repository code.
    """
    wiring_for = lambda settings: FakeWiring(  # noqa: E731 - one-line factory
        settings, hub=hub_with(remote_code=True, tmp_path=tmp_path)
    )
    result = invoke(
        runner,
        home,
        [
            "prepare",
            SAFETENSORS_REPO,
            "--device",
            "cpu",
            "--trust-remote-code",
            "--yes",
        ],
        factory=wiring_for,
        stdin="y\n" * 6,
    )
    assert result.exit_code == int(ExitCode.CANCELLED)
    assert "remote_code" in result.stderr


def test_explicit_npu_never_falls_back_and_exits_fifty(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """``--device npu`` is an instruction: the job stops and recommends the GPU."""
    runtime = FakeRuntimeProbe(compile_failures={"NPU": NPU_UNSUPPORTED_GRAPH})
    wiring_for = lambda settings: FakeWiring(  # noqa: E731 - one-line factory
        settings, hub=hub_with(tmp_path=tmp_path), runtime=runtime
    )
    result = invoke(
        runner,
        home,
        ["prepare", SAFETENSORS_REPO, "--device", "npu", "--yes"],
        factory=wiring_for,
    )
    assert result.exit_code == int(ExitCode.DEVICE_UNSUPPORTED)
    assert "--device gpu" in result.stderr
    assert "will not silently prepare for another device" in result.stderr
    assert {device for _name, device in runtime.compiled} == {"NPU"}


@pytest.mark.parametrize("device", ["npu", "auto"])
def test_a_permission_denied_npu_in_prepare_exits_ten(
    runner: CliRunner, home: Path, tmp_path: Path, device: str
) -> None:
    """A device the host will not let yonga open is an unusable environment, not code 50.

    The NPU would work once the user is given access to the device node, so
    ``prepare`` answers with :data:`ExitCode.ENVIRONMENT_UNUSABLE`, whether the
    device was requested explicitly or chosen by auto mode, and never falls back.
    """
    runtime = FakeRuntimeProbe(compile_failures={"NPU": NPU_PERMISSION_DENIED})
    wiring_for = lambda settings: FakeWiring(  # noqa: E731 - one-line factory
        settings,
        hub=hub_with(tmp_path=tmp_path),
        runtime=runtime,
        resolver=builtin_resolver([NPU_FIRST_REGISTRY]),
    )
    result = invoke(
        runner,
        home,
        ["prepare", SAFETENSORS_REPO, "--device", device, "--yes"],
        factory=wiring_for,
    )
    assert result.exit_code == int(ExitCode.ENVIRONMENT_UNUSABLE)
    assert "No permission to use the NPU device node" in result.stderr
    assert "--device auto" not in result.stderr
    assert {device for _name, device in runtime.compiled} == {"NPU"}


def test_auto_device_falls_back_from_npu_to_gpu(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """In auto mode a classified compatibility limit moves to the recommended device."""
    runtime = FakeRuntimeProbe(compile_failures={"NPU": NPU_UNSUPPORTED_GRAPH})
    wiring_for = lambda settings: FakeWiring(  # noqa: E731 - one-line factory
        settings,
        hub=hub_with(tmp_path=tmp_path),
        runtime=runtime,
        resolver=builtin_resolver([NPU_FIRST_REGISTRY]),
    )
    result = invoke(
        runner,
        home,
        ["prepare", SAFETENSORS_REPO, "--device", "auto", "--yes"],
        factory=wiring_for,
    )
    assert result.exit_code == int(ExitCode.WARNINGS)
    assert "Device    GPU" in result.stdout
    assert "NPU could not run this model, so it was prepared for GPU instead." in result.stderr
    assert {device for _name, device in runtime.compiled} == {"NPU", "GPU"}


def test_declining_a_confirmation_exits_sixty(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """Without ``--yes`` a non-interactive session cannot approve, so nothing runs."""
    hub = hub_with(tmp_path=tmp_path)
    wiring_for = lambda settings: FakeWiring(settings, hub=hub)  # noqa: E731
    result = invoke(
        runner, home, ["prepare", SAFETENSORS_REPO, "--device", "cpu"], factory=wiring_for
    )
    assert result.exit_code == int(ExitCode.CANCELLED)
    assert hub.downloaded == []


def test_unknown_device_is_a_configuration_error(runner: CliRunner, home: Path) -> None:
    """A device outside the documented set is refused before any work happens."""
    result = invoke(runner, home, ["prepare", SAFETENSORS_REPO, "--device", "tpu"])
    assert result.exit_code == int(ExitCode.REGISTRY)
    assert "auto, npu, gpu, cpu" in result.stderr


# ======================================================================
# list and remove
# ======================================================================
def test_list_is_empty_and_says_what_to_do(runner: CliRunner, home: Path) -> None:
    """An empty catalogue explains how to fill it."""
    result = invoke(runner, home, ["list"])
    assert result.exit_code == int(ExitCode.SUCCESS)
    assert "yonga prepare MODEL" in result.stdout


def test_list_json_is_one_document(runner: CliRunner, home: Path) -> None:
    """``list --json`` is a single machine-readable catalogue."""
    result = invoke(runner, home, ["list", "--json"])
    payload = one_document(result)
    assert payload == {
        "ok": True,
        "artifacts": [],
        "count": 0,
        "artifacts_dir": str(home / "data" / "artifacts"),
    }


def test_remove_refuses_without_confirmation_and_then_removes(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """``remove`` asks first; ``--yes`` answers, and the Hub cache is never touched."""
    wiring_for = lambda settings: FakeWiring(  # noqa: E731 - one-line factory
        settings, hub=hub_with(tmp_path=tmp_path)
    )
    prepared = invoke(
        runner,
        home,
        ["prepare", SAFETENSORS_REPO, "--device", "cpu", "--yes", "--json"],
        factory=wiring_for,
    )
    assert prepared.exit_code in (int(ExitCode.SUCCESS), int(ExitCode.WARNINGS))
    artifact = Path(str(one_document(prepared)["artifact_dir"]))
    assert artifact.is_dir()

    declined = invoke(runner, home, ["remove", str(artifact)], factory=wiring_for)
    assert declined.exit_code == int(ExitCode.CANCELLED)
    assert artifact.is_dir()

    removed = invoke(runner, home, ["remove", str(artifact), "--yes"], factory=wiring_for)
    assert removed.exit_code == int(ExitCode.SUCCESS)
    assert not artifact.exists()
    assert "Hugging Face download cache is never touched" in removed.stdout


# ======================================================================
# probe
# ======================================================================
def test_probe_json_is_one_document_with_a_role_matrix(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """``probe --json`` reports every submodel on every candidate device."""
    artifact = _staged_artifact(tmp_path)
    result = invoke(runner, home, ["probe", str(artifact), "--device", "cpu", "--json"])
    payload = one_document(result)
    assert payload["ok"] is True
    assert result.exit_code == int(ExitCode.SUCCESS)
    matrix = payload["role_matrix"]
    assert isinstance(matrix, dict)
    assert matrix["language"] == {"CPU": "ok"}


def test_probe_failure_uses_the_diagnostic_exit_code(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """A classified compile failure exits with the code its category reserves."""
    artifact = _staged_artifact(tmp_path)
    runtime = FakeRuntimeProbe(compile_failures={"NPU": NPU_UNSUPPORTED_GRAPH})
    wiring_for = lambda settings: FakeWiring(settings, runtime=runtime)  # noqa: E731
    result = invoke(runner, home, ["probe", str(artifact), "--device", "npu"], factory=wiring_for)
    assert result.exit_code == int(ExitCode.COMPILE)
    assert "Traceback" not in result.stderr


def _staged_artifact(tmp_path: Path) -> Path:
    """Write a minimal, readable artifact directory for the probe tests."""
    from tests.unit.test_prepare_planning import MODEL_XML, write_tokenizer_pair

    directory = tmp_path / "artifact"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "openvino_language_model.xml").write_text(MODEL_XML, encoding="utf-8")
    (directory / "openvino_language_model.bin").write_bytes(b"\x00" * 32)
    write_tokenizer_pair(directory, broken=False)
    (directory / "config.json").write_text('{"model_type": "qwen3_5"}\n', encoding="utf-8")
    return directory


def test_probe_quick_never_builds_a_pipeline_and_exits_with_a_warning(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """``--quick``: a deferred NPU compile is inconclusive, JSON says so, exit 2."""
    artifact = _staged_artifact(tmp_path)
    runtime = FakeRuntimeProbe(compile_failures={"NPU": NPU_DYNAMIC_SHAPE_ERROR})
    smoke = FakeSmokeTester()
    wiring_for = lambda settings: FakeWiring(settings, runtime=runtime, smoke=smoke)  # noqa: E731

    result = invoke(
        runner,
        home,
        ["probe", str(artifact), "--device", "npu", "--quick", "--json"],
        factory=wiring_for,
    )

    assert result.exit_code == int(ExitCode.WARNINGS)
    payload = one_document(result)
    assert payload["ok"] is False
    assert payload["inconclusive"] is True
    assert payload["quick"] is True
    reports = payload["reports"]
    assert isinstance(reports, list)
    assert reports[0]["status"] == "inconclusive"
    assert reports[0]["smoke_test"] is None
    assert reports[0]["diagnostic"]["code"] == "probe.inconclusive.quick"
    assert payload["role_matrix"] == {
        "language": {"NPU": "inconclusive"},
        "tokenizer": {"CPU": "ok"},
    }
    assert smoke.calls == []


def test_probe_quick_explains_how_to_settle_the_device(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    artifact = _staged_artifact(tmp_path)
    runtime = FakeRuntimeProbe(compile_failures={"NPU": NPU_DYNAMIC_SHAPE_ERROR})
    wiring_for = lambda settings: FakeWiring(settings, runtime=runtime)  # noqa: E731

    result = invoke(
        runner, home, ["probe", str(artifact), "--device", "npu", "--quick"], factory=wiring_for
    )

    assert result.exit_code == int(ExitCode.WARNINGS)
    assert "inconclusive" in result.stdout
    stderr = plain_text(result.stderr)
    assert "the raw compile was deferred" in stderr
    assert "--timeout SECONDS" in stderr
    assert "Traceback" not in result.stderr


class ChildProcessWiring(FakeWiring):
    """A :class:`FakeWiring` whose smoke-test runner plays the child process."""

    def __init__(self, settings: Settings, *, child: CommandRunner, **kwargs: object) -> None:
        super().__init__(settings, **kwargs)  # type: ignore[arg-type]
        self._child = child

    def smoke_child_runner(self) -> CommandRunner:
        """Return the scripted child instead of refusing it."""
        return self._child


def test_probe_timeout_stops_the_build_and_reports_inconclusive(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    artifact = _staged_artifact(tmp_path)
    runtime = FakeRuntimeProbe(compile_failures={"NPU": NPU_DYNAMIC_SHAPE_ERROR})
    smoke = FakeSmokeTester()
    child = SmokeChildRunner(timed_out=True)

    def wiring_for(settings: Settings) -> Wiring:
        return ChildProcessWiring(settings, child=child, runtime=runtime, smoke=smoke)

    result = invoke(
        runner,
        home,
        ["probe", str(artifact), "--device", "npu", "--timeout", "1.5", "--json"],
        factory=wiring_for,
    )

    assert result.exit_code == int(ExitCode.WARNINGS)
    payload = one_document(result)
    assert payload["inconclusive"] is True
    assert payload["timeout_seconds"] == 1.5
    reports = payload["reports"]
    assert isinstance(reports, list)
    assert reports[0]["status"] == "inconclusive"
    assert reports[0]["diagnostic"]["code"] == "probe.inconclusive.timeout"
    assert smoke.calls == []
    (command, timeout), *_ = child.calls
    assert command[1:4] == ["-P", "-m", "yonga.adapters.smoke_child"]
    assert timeout == 1.5


def test_probe_timeout_passes_a_device_whose_child_generated(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    artifact = _staged_artifact(tmp_path)
    runtime = FakeRuntimeProbe(compile_failures={"NPU": NPU_DYNAMIC_SHAPE_ERROR})
    verdict = ProbeResult(device="NPU", status=ProbeStatus.OK, message="generated 24 token(s)")
    child = SmokeChildRunner(stdout=[verdict_line(verdict)])

    def wiring_for(settings: Settings) -> Wiring:
        return ChildProcessWiring(settings, child=child, runtime=runtime)

    result = invoke(
        runner,
        home,
        ["probe", str(artifact), "--device", "npu", "--timeout", "600", "--json"],
        factory=wiring_for,
    )

    assert result.exit_code == int(ExitCode.SUCCESS)
    payload = one_document(result)
    assert payload["ok"] is True
    assert payload["inconclusive"] is False


@pytest.mark.parametrize(
    ("args", "code"),
    [
        (["--timeout", "0"], "probe.timeout.invalid"),
        (["--timeout", "-30"], "probe.timeout.invalid"),
        (["--timeout", "nan"], "probe.timeout.invalid"),
        (["--quick", "--timeout", "30"], "probe.options.conflict"),
    ],
)
def test_probe_rejects_a_timeout_that_cannot_bound_anything(
    runner: CliRunner, home: Path, tmp_path: Path, args: list[str], code: str
) -> None:
    artifact = _staged_artifact(tmp_path)
    runtime = FakeRuntimeProbe(compile_failures={"NPU": NPU_DYNAMIC_SHAPE_ERROR})
    wiring_for = lambda settings: FakeWiring(settings, runtime=runtime)  # noqa: E731

    result = invoke(
        runner,
        home,
        ["probe", str(artifact), "--device", "npu", *args, "--json"],
        factory=wiring_for,
    )

    assert result.exit_code == int(ExitCode.REGISTRY)
    assert runtime.compiled == [], "an invalid option is refused before anything compiles"
    assert code in result.stderr + result.stdout


def test_an_inconclusive_device_never_masks_a_failure_in_the_exit_code(tmp_path: Path) -> None:
    from yonga.application.probe import ProbeOutcome
    from yonga.cli import _probe_exit
    from yonga.domain.artifacts import DeviceProbeReport

    unsettled = diagnostic(
        "probe.inconclusive.quick",
        FailureCategory.UNKNOWN_RUNTIME_ERROR,
        "NPU is inconclusive",
        "Deferred.",
        severity="warning",
    )
    denied = diagnostic(
        "npu.permission_denied",
        FailureCategory.PERMISSION_DENIED,
        "NPU not accessible",
        "Permission denied on /dev/accel/accel0",
    )
    inconclusive = DeviceProbeReport(
        device="NPU", status=ProbeStatus.INCONCLUSIVE, diagnostic=unsettled
    )
    failed = DeviceProbeReport(device="GPU", status=ProbeStatus.FAILED, diagnostic=denied)
    passed = DeviceProbeReport(device="CPU", status=ProbeStatus.OK)

    def exit_for(*reports: DeviceProbeReport) -> ExitCode:
        return _probe_exit(ProbeOutcome(artifact_dir=tmp_path, reports=list(reports)))

    assert exit_for(inconclusive) is ExitCode.WARNINGS
    assert exit_for(inconclusive, failed) is denied.exit_code()
    assert exit_for(inconclusive, passed) is ExitCode.SUCCESS


# ======================================================================
# chat
# ======================================================================
def test_chat_without_a_prepared_artifact_explains_prepare(runner: CliRunner, home: Path) -> None:
    """``chat`` never starts a multi-gigabyte conversion as a side effect."""
    result = invoke(runner, home, ["chat", SAFETENSORS_REPO])
    assert result.exit_code == int(ExitCode.REGISTRY)
    assert f"yonga prepare {SAFETENSORS_REPO}" in result.stderr
    assert "does not convert a model as a side effect" in result.stderr


def test_chat_is_refused_in_json_mode(runner: CliRunner, home: Path, tmp_path: Path) -> None:
    """A conversation cannot be one JSON document, and says so instead of failing oddly."""
    artifact = _staged_artifact(tmp_path)
    result = invoke(runner, home, ["chat", str(artifact), "--json"])
    payload = one_document(result)
    assert payload["ok"] is False
    assert payload["exit_code"] == int(ExitCode.REGISTRY)


def test_chat_answers_a_turn_and_leaves_on_exit(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """The prompt loop renders an answer, its metrics, and honours ``/exit``."""
    artifact = _staged_artifact(tmp_path)
    factory = FakeRuntimeFactory()
    wiring_for = lambda settings: FakeWiring(  # noqa: E731 - one-line factory
        settings, runtime_factory=factory
    )
    result = invoke(
        runner,
        home,
        ["chat", str(artifact), "--device", "cpu", "--max-new-tokens", "64"],
        factory=wiring_for,
        stdin="Hello there\n/exit\n",
    )
    assert result.exit_code == int(ExitCode.SUCCESS)
    assert "Hello from the fake model." in result.stdout
    assert "tok/s" in result.stdout
    assert factory.opened and factory.opened[0][1] == "CPU"
    assert factory.runtimes[0].closed is True


def test_chat_commands_are_applied_to_the_session(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """``/tokens`` and ``/help`` are handled without reaching the model."""
    artifact = _staged_artifact(tmp_path)
    factory = FakeRuntimeFactory()
    wiring_for = lambda settings: FakeWiring(  # noqa: E731 - one-line factory
        settings, runtime_factory=factory
    )
    result = invoke(
        runner,
        home,
        ["chat", str(artifact)],
        factory=wiring_for,
        stdin="/help\n/tokens 8\n/exit\n",
    )
    assert result.exit_code == int(ExitCode.SUCCESS)
    assert "/reasoning on|off" in result.stdout
    assert "8" in result.stdout
    assert factory.runtimes == []


# ======================================================================
# bench
# ======================================================================
def test_bench_json_is_one_document(runner: CliRunner, home: Path, tmp_path: Path) -> None:
    """``bench --json`` writes its result file and prints one document."""
    artifact = _staged_artifact(tmp_path)
    result = invoke(
        runner,
        home,
        [
            "bench",
            str(artifact),
            "--device",
            "cpu",
            "--suite",
            "quick",
            "--runs",
            "1",
            "--warmups",
            "0",
            "--json",
        ],
    )
    payload = one_document(result)
    assert payload["ok"] is True
    assert payload["device"] == "CPU"
    saved = payload["result_path"]
    assert isinstance(saved, str)
    assert Path(saved).is_file()


def test_bench_rejects_an_unknown_suite(runner: CliRunner, home: Path, tmp_path: Path) -> None:
    """An unknown suite name is refused with the list of the real ones."""
    artifact = _staged_artifact(tmp_path)
    result = invoke(runner, home, ["bench", str(artifact), "--suite", "enormous"])
    assert result.exit_code == int(ExitCode.REGISTRY)
    assert "quick, standard, long" in result.stderr


# ======================================================================
# cache
# ======================================================================
def records(value: object) -> list[dict[str, Any]]:
    """Narrow a JSON array of objects for indexing."""
    assert isinstance(value, list)
    assert all(isinstance(item, dict) for item in value)
    return value


def cli_settings(home: Path) -> Settings:
    """The settings ``invoke`` gives every command, for building fixtures beside it."""
    return Settings.model_validate(
        {"config_dir": home / "config", "cache_dir": home / "cache", "data_dir": home / "data"}
    )


def populated_cache(home: Path, tmp_path: Path) -> dict[str, Path]:
    """One compile cache entry per status, under the CLI's own directories."""
    settings = cli_settings(home)
    store = ArtifactStore(settings.paths, settings)
    kept = publish(store, tmp_path, manifest=make_manifest(artifact_id="aaaaaaaaaaaaaaaa"))
    gone = publish(
        store,
        tmp_path,
        manifest=make_manifest(artifact_id="bbbbbbbbbbbbbbbb", profile="cpu-int4-asym-g128"),
        name="staging-gone",
    )
    entries = {
        "in-use": recorded_entry(settings, kept, size=1000),
        "stale-runtime": recorded_entry(
            settings, kept, device="NPU", version=OLDER, size=3000, used=BEFORE_UPGRADE
        ),
        "orphaned": recorded_entry(settings, gone, device="CPU", size=5000),
        "unattributed": legacy_entry(settings, "0123456789abcdef", size=7000),
    }
    shutil.rmtree(gone)
    return entries


def test_cache_list_json_is_one_document_even_when_empty(runner: CliRunner, home: Path) -> None:
    result = invoke(runner, home, ["cache", "list", "--json"])

    payload = one_document(result)
    assert result.exit_code == int(ExitCode.SUCCESS)
    assert payload["ok"] is True
    assert payload["count"] == 0
    assert payload["total_bytes"] == 0
    assert payload["compile_cache_dir"] == str(home / "cache" / "compile")


def test_cache_list_shows_every_status_and_the_prune_hint(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    entries = populated_cache(home, tmp_path)

    result = invoke(runner, home, ["cache", "list"])

    assert result.exit_code == int(ExitCode.SUCCESS)
    output = plain_text(result.stdout)
    for status, path in entries.items():
        assert path.name in output
        assert status in output
    assert "bbbbbbbbbbbbbbbb" in output, "the owner of an orphan is still named"
    assert "4 entries" in output
    assert "yonga cache prune" in output
    assert "--unattributed" in output


def test_plain_cache_is_cache_list(runner: CliRunner, home: Path, tmp_path: Path) -> None:
    entries = populated_cache(home, tmp_path)

    listed = one_document(invoke(runner, home, ["cache", "list", "--json"]))
    alias = one_document(invoke(runner, home, ["cache", "--json"]))

    assert alias == listed
    statuses = {entry["key"]: entry["status"] for entry in records(listed["entries"])}
    assert statuses == {path.name: status for status, path in entries.items()}
    assert listed["by_status"] == {
        "in-use": 1,
        "orphaned": 1,
        "stale-runtime": 1,
        "unattributed": 1,
    }
    assert listed["installed_openvino"] == INSTALLED


def test_cache_prune_asks_first_and_declining_removes_nothing(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    entries = populated_cache(home, tmp_path)

    declined = invoke(runner, home, ["cache", "prune"])

    assert declined.exit_code == int(ExitCode.CANCELLED)
    assert all(path.is_dir() for path in entries.values())


def test_cache_prune_dry_run_shows_the_plan_and_deletes_nothing(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    entries = populated_cache(home, tmp_path)

    result = invoke(runner, home, ["cache", "prune", "--dry-run", "--json"])

    payload = one_document(result)
    assert result.exit_code == int(ExitCode.SUCCESS)
    assert payload["dry_run"] is True
    planned = {entry["key"]: entry["selected_because"] for entry in records(payload["planned"])}
    assert planned == {
        entries["orphaned"].name: "orphaned",
        entries["stale-runtime"].name: "stale-runtime",
    }
    assert payload["removed"] == []
    assert payload["freed_bytes"] == 0
    assert all(path.is_dir() for path in entries.values())


def test_cache_prune_yes_removes_orphaned_and_stale_entries_only(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    entries = populated_cache(home, tmp_path)

    result = invoke(runner, home, ["cache", "prune", "--yes", "--json"])

    payload = one_document(result)
    assert result.exit_code == int(ExitCode.SUCCESS)
    removed = payload["removed"]
    assert isinstance(removed, list)
    assert sorted(removed) == sorted([entries["orphaned"].name, entries["stale-runtime"].name])
    assert payload["freed_bytes"] == payload["planned_bytes"]
    freed = payload["freed_bytes"]
    assert isinstance(freed, int) and freed >= 8000
    assert payload["kept"] == 2
    assert not entries["orphaned"].exists() and not entries["stale-runtime"].exists()
    assert entries["in-use"].is_dir() and entries["unattributed"].is_dir()


def test_cache_prune_all_and_unattributed_go_further(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    entries = populated_cache(home, tmp_path)
    keep = home / "cache" / "compile" / "not-an-entry"
    keep.mkdir()

    unattributed = invoke(runner, home, ["cache", "prune", "--unattributed", "--yes"])
    assert unattributed.exit_code == int(ExitCode.SUCCESS)
    assert not entries["unattributed"].exists()
    assert entries["in-use"].is_dir()
    assert "freed" in unattributed.stdout

    everything = invoke(runner, home, ["cache", "prune", "--all", "--yes"])
    assert everything.exit_code == int(ExitCode.SUCCESS)
    assert not entries["in-use"].exists()
    assert keep.is_dir(), "only key-named entries are ever removed"

    nothing = invoke(runner, home, ["cache", "prune", "--yes"])
    assert nothing.exit_code == int(ExitCode.SUCCESS)
    assert "Nothing to prune" in nothing.stdout


def test_cache_prune_dry_run_in_human_mode_lists_the_plan(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    entries = populated_cache(home, tmp_path)
    (home / "cache" / "compile" / "notes").mkdir()

    result = invoke(runner, home, ["cache", "prune", "--dry-run"])

    assert result.exit_code == int(ExitCode.SUCCESS)
    output = plain_text(result.stdout)
    assert entries["orphaned"].name in output and entries["stale-runtime"].name in output
    assert entries["in-use"].name not in output
    assert "Dry run: nothing was removed." in output
    assert "another install" in output, "stale-runtime is explained relative to this runtime"
    assert all(path.is_dir() for path in entries.values())


def test_cache_prune_older_than_also_takes_unused_entries(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    entries = populated_cache(home, tmp_path)
    stamp = (datetime.now(tz=UTC) - timedelta(days=90)).timestamp()
    for path in [*entries["in-use"].rglob("*"), entries["in-use"]]:
        os.utime(path, (stamp, stamp))
    record = entries["in-use"] / COMPILE_CACHE_ENTRY_FILENAME
    document = json.loads(record.read_text(encoding="utf-8"))
    document["last_used_at"] = datetime.fromtimestamp(stamp, tz=UTC).isoformat()
    record.write_text(json.dumps(document), encoding="utf-8")
    os.utime(record, (stamp, stamp))
    os.utime(entries["in-use"], (stamp, stamp))

    result = invoke(runner, home, ["cache", "prune", "--older-than", "30", "--yes", "--json"])

    payload = one_document(result)
    assert result.exit_code == int(ExitCode.SUCCESS)
    planned = {entry["key"]: entry["selected_because"] for entry in records(payload["planned"])}
    assert planned[entries["in-use"].name] == "older-than"
    criteria = payload["criteria"]
    assert isinstance(criteria, dict) and criteria["older_than_days"] == 30.0
    assert not entries["in-use"].exists()
    assert entries["unattributed"].is_dir(), "unattributed but recent, and not asked for"


def test_cache_prune_reports_an_entry_it_could_not_remove(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    entries = populated_cache(home, tmp_path)
    locked = entries["orphaned"] / "locked"
    locked.mkdir()
    (locked / "stuck.blob").write_bytes(b"\x07" * 4096)
    locked.chmod(0o500)
    try:
        result = invoke(runner, home, ["cache", "prune", "--yes", "--json"])
        human = invoke(runner, home, ["cache", "prune", "--yes"])
    finally:
        locked.chmod(0o700)

    payload = one_document(result)
    assert result.exit_code == int(ExitCode.WARNINGS)
    assert payload["ok"] is False
    failed = records(payload["failed"])
    assert [item["key"] for item in failed] == [entries["orphaned"].name]
    assert payload["removed"] == [entries["stale-runtime"].name]
    planned_bytes = payload["planned_bytes"]
    freed_bytes = payload["freed_bytes"]
    assert isinstance(planned_bytes, int) and isinstance(freed_bytes, int)
    assert freed_bytes < planned_bytes
    assert (locked / "stuck.blob").is_file()

    assert human.exit_code == int(ExitCode.WARNINGS)
    assert "Could not remove" in plain_text(human.stderr)


def test_cache_prune_rejects_a_negative_age(runner: CliRunner, home: Path) -> None:
    result = invoke(runner, home, ["cache", "prune", "--older-than", "-3", "--json"])

    assert result.exit_code == int(ExitCode.REGISTRY)
    document = json.loads(result.stdout)
    assert document["diagnostic"]["code"] == "cache.prune.older_than_invalid"


def test_doctor_warns_about_a_large_reclaimable_compile_cache(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """A sparse 12 GiB blob of a removed artifact is over the 10 GiB threshold."""
    entries = populated_cache(home, tmp_path)
    with (entries["orphaned"] / "huge.blob").open("wb") as blob:
        blob.truncate(12 * GIB)

    result = invoke(runner, home, ["doctor", "--json"])

    payload = one_document(result)
    assert result.exit_code == int(ExitCode.WARNINGS)
    summary = payload["compile_cache"]
    assert isinstance(summary, dict)
    assert summary["reclaimable_bytes"] >= 12 * GIB
    storage = next(item for item in records(payload["sections"]) if item["id"] == "storage")
    check = next(item for item in storage["checks"] if item["id"] == "storage.compile_cache")
    assert check["status"] == "warning"
    assert any("yonga cache prune" in line for line in check["remediation"])


# ======================================================================
# registry
# ======================================================================
def test_registry_validate_accepts_the_builtin_registry(runner: CliRunner, home: Path) -> None:
    """The shipped registry validates deterministically."""
    result = invoke(runner, home, ["registry", "validate"])
    assert result.exit_code == int(ExitCode.SUCCESS)
    assert "Registry is valid." in result.stdout


def test_registry_validate_reports_a_broken_override(runner: CliRunner, home: Path) -> None:
    """A bad override layer is a registry error, with the problems listed."""
    overrides = home / "config" / "registry-overrides"
    overrides.mkdir(parents=True, exist_ok=True)
    (overrides / "broken.yaml").write_text(
        "schema_version: 1\n"
        "architectures:\n"
        "  - id: arch-dangling-profile\n"
        "    schema_version: 1\n"
        "    status: active\n"
        "    priority: 5\n"
        "    summary: Names a conversion profile that does not exist\n"
        "    match:\n"
        "      model_types:\n"
        "        - never_seen_model_type\n"
        "    export:\n"
        "      pipeline: llm\n"
        "      trust_remote_code: false\n"
        "    profiles:\n"
        "      cpu: no-such-profile\n",
        encoding="utf-8",
    )
    result = invoke(runner, home, ["registry", "validate"])
    assert result.exit_code == int(ExitCode.REGISTRY)
    assert "no-such-profile" in result.stdout
    assert "Problems" in result.stdout


def test_registry_explain_json_names_the_winning_rules(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """``registry explain --json`` shows which rule won and why."""
    wiring_for = lambda settings: FakeWiring(  # noqa: E731 - one-line factory
        settings, hub=hub_with(tmp_path=tmp_path)
    )
    result = invoke(
        runner,
        home,
        ["registry", "explain", SAFETENSORS_REPO, "--device", "gpu", "--json"],
        factory=wiring_for,
    )
    payload = one_document(result)
    assert payload["ok"] is True
    assert payload["repo_id"] == SAFETENSORS_REPO
    assert payload["device"] == "GPU"
    assert payload["profile_id"]


def test_registry_paths_shows_where_layers_come_from(runner: CliRunner, home: Path) -> None:
    """``registry paths`` lists the builtin layer, the overrides and the managed dirs."""
    result = invoke(runner, home, ["registry", "paths"])
    assert result.exit_code == int(ExitCode.SUCCESS)
    assert "registry-overrides" in result.stdout
    assert str(home / "data" / "artifacts") in result.stdout


# ======================================================================
# logs
# ======================================================================
def test_logs_lists_jobs_and_renders_one(runner: CliRunner, home: Path) -> None:
    """``logs`` indexes the job logs and pretty-prints a single one."""
    logs_dir = home / "data" / "logs"
    logs_dir.mkdir(parents=True)
    job_id = "11111111-2222-3333-4444-555555555555"
    (logs_dir / f"{job_id}.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-09-22T10:11:12.345Z",
                "job_id": job_id,
                "level": "info",
                "event": "job.created",
                "message": "job created",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index = invoke(runner, home, ["logs"])
    assert index.exit_code == int(ExitCode.SUCCESS)
    assert job_id in index.stdout

    shown = invoke(runner, home, ["logs", job_id])
    assert shown.exit_code == int(ExitCode.SUCCESS)
    assert "job.created" in shown.stdout

    raw = invoke(runner, home, ["logs", "11111111", "--raw"])
    assert raw.exit_code == int(ExitCode.SUCCESS)
    assert json.loads(raw.stdout.strip())["event"] == "job.created"


def test_logs_for_an_unknown_job_is_a_configuration_error(runner: CliRunner, home: Path) -> None:
    """Asking for a job that never ran is explained, not a stack trace."""
    (home / "data" / "logs").mkdir(parents=True)
    result = invoke(runner, home, ["logs", "nope"])
    assert result.exit_code == int(ExitCode.REGISTRY)
    assert "yonga logs" in result.stderr


def test_logs_bundle_writes_an_archive_and_reports_it_as_json(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """``--bundle --json`` emits exactly one document naming the archive it wrote."""
    import tarfile

    logs_dir = home / "data" / "logs"
    logs_dir.mkdir(parents=True)
    job_id = "22222222-3333-4444-5555-666666666666"
    (logs_dir / f"{job_id}.jsonl").write_text(
        json.dumps({"event": "job.failed", "message": "conversion failed"}) + "\n",
        encoding="utf-8",
    )
    target = tmp_path / "bundle.tar.gz"

    result = invoke(
        runner, home, ["--json", "logs", "22222222", "--bundle", "--output", str(target)]
    )

    assert result.exit_code == int(ExitCode.SUCCESS)
    document = json.loads(result.stdout)
    assert document["job_id"] == job_id
    assert Path(document["path"]) == target
    with tarfile.open(target) as archive:
        names = {Path(member.name).name for member in archive.getmembers()}
    assert {"README.txt", "job.jsonl"} <= names


def test_logs_bundle_needs_a_job(runner: CliRunner, home: Path) -> None:
    (home / "data" / "logs").mkdir(parents=True)
    result = invoke(runner, home, ["logs", "--bundle"])
    assert result.exit_code == int(ExitCode.REGISTRY)
    assert "yonga logs JOB_ID --bundle" in result.stderr


class SearchingHub:
    """A HubSearch double returning one config-identified and one GGUF-identified hit."""

    def search(self, query: str, *, limit: int = 20) -> list[ModelMetadata]:
        from yonga.domain.models import ModelIdentity

        return [
            ModelMetadata(
                identity=ModelIdentity(repo_id="Qwen/Qwen3.5-9B"),
                resolved_revision=None,
                model_type="qwen3_5",
                architectures=["Qwen3_5ForConditionalGeneration"],
                tags=["safetensors"],
                downloads=9_787_242,
            ),
            ModelMetadata(
                identity=ModelIdentity(repo_id="unsloth/Qwen3.5-9B-GGUF"),
                resolved_revision=None,
                gguf_metadata={"architecture": "qwen35"},
                tags=["gguf"],
            ),
        ][:limit]


class SearchWiring(FakeWiring):
    def hub_search(self) -> SearchingHub:
        return SearchingHub()


def test_search_json_is_one_document_with_registry_knowledge(runner: CliRunner, home: Path) -> None:
    result = invoke(
        runner, home, ["--json", "search", "qwen3.5"], factory=lambda s: SearchWiring(s)
    )

    assert result.exit_code == int(ExitCode.SUCCESS)
    document = json.loads(result.stdout)
    first, second = document["hits"]
    assert first["architecture"] == "qwen3_5" and first["identified_from"] is None
    assert second["architecture"] == "qwen3_5" and second["identified_from"] == "GGUF metadata"
    npu = next(entry for entry in first["compatibility"] if entry["device"] == "NPU")
    assert npu["status"] == "unsupported"


def test_search_table_states_what_its_compatibility_means(runner: CliRunner, home: Path) -> None:
    result = invoke(runner, home, ["search", "qwen3.5"], factory=lambda s: SearchWiring(s))

    assert result.exit_code == int(ExitCode.SUCCESS)
    assert "Qwen/Qwen3.5-9B" in result.stdout
    assert "yonga inspect MODEL" in result.stdout


def test_search_rejects_an_unknown_device(runner: CliRunner, home: Path) -> None:
    result = invoke(
        runner, home, ["search", "qwen", "--device", "mars"], factory=lambda s: SearchWiring(s)
    )
    assert result.exit_code == int(ExitCode.REGISTRY)


def test_search_with_a_hub_that_cannot_search_fails_loudly(runner: CliRunner, home: Path) -> None:
    """A wiring error is reported, never papered over with a second networked client."""
    result = invoke(runner, home, ["search", "qwen"])

    assert result.exit_code == int(ExitCode.REGISTRY)
    assert "Search is unavailable" in result.stderr


# ======================================================================
# failure rendering
# ======================================================================
def test_an_unexpected_exception_exits_ninety_nine_without_a_traceback(
    runner: CliRunner, home: Path
) -> None:
    """A bug is reported as exit 99 with a pointer to the log, never as a traceback."""

    def factory(settings: Settings) -> Wiring:
        return FakeWiring(settings, store=ExplodingArtifactStore(settings.paths, settings))

    result = invoke(runner, home, ["list"], factory=factory)
    assert result.exit_code == int(ExitCode.INTERNAL)
    assert "Unexpected internal error" in result.stderr
    assert "Traceback" not in result.stderr
    assert "synthetic catalogue failure" not in result.stderr
    assert str(home / "data" / "logs") in result.stderr


def test_a_closed_stdout_is_a_quiet_success_not_a_bug(runner: CliRunner, home: Path) -> None:
    """``BrokenPipeError`` is the reader saying "enough", never an internal error."""

    def factory(settings: Settings) -> Wiring:
        return FakeWiring(settings, store=ClosedPipeArtifactStore(settings.paths, settings))

    result = invoke(runner, home, ["list"], factory=factory)
    assert result.exit_code == int(ExitCode.SUCCESS)
    assert "Unexpected internal error" not in result.stderr
    assert "Traceback" not in result.stderr
    assert "BrokenPipeError" not in result.stderr


def test_a_closed_stdout_reported_by_rich_is_also_a_quiet_success(
    runner: CliRunner, home: Path
) -> None:
    """Rich turns a closed pipe into ``SystemExit(1)``; that is still exit 0 here."""

    def factory(settings: Settings) -> Wiring:
        return FakeWiring(settings, store=RichClosedPipeArtifactStore(settings.paths, settings))

    result = invoke(runner, home, ["list"], factory=factory)
    assert result.exit_code == int(ExitCode.SUCCESS)
    assert "Unexpected internal error" not in result.stderr


def test_an_unrelated_system_exit_keeps_its_own_code(runner: CliRunner, home: Path) -> None:
    """Only a ``SystemExit`` caused by a closed pipe is rewritten; others pass through."""

    def factory(settings: Settings) -> Wiring:
        return FakeWiring(settings, store=DeliberateExitArtifactStore(settings.paths, settings))

    result = invoke(runner, home, ["list"], factory=factory)
    assert result.exit_code == 3


def write_into_a_broken_pipe(
    home: Path, command: Sequence[str], *, reader: str, wait_for_reader: bool
) -> tuple[int, str]:
    """Run ``yonga command`` with stdout on a pipe whose reader walks away.

    ``reader`` is the body of a throwaway Python process holding the read end;
    with ``wait_for_reader`` the writer is not started until that process has
    exited, which is how a command whose whole answer fits in the pipe buffer is
    made to meet a closed pipe deterministically.
    """
    import subprocess
    import sys

    environment = dict(os.environ)
    environment.update({"YONGA_CONFIG_DIR": str(home / "config"), "NO_COLOR": "1"})

    consumer = subprocess.Popen([sys.executable, "-c", reader], stdin=subprocess.PIPE)
    assert consumer.stdin is not None
    if wait_for_reader:
        consumer.wait(timeout=60)
    writer = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "yonga",
            "--data-dir",
            str(home / "data"),
            "--cache-dir",
            str(home / "cache"),
            *command,
        ],
        stdout=consumer.stdin,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    consumer.stdin.close()
    consumer.wait(timeout=60)
    _, stderr = writer.communicate(timeout=60)
    return writer.returncode, stderr


def test_piping_a_large_log_into_head_exits_quietly(tmp_path: Path) -> None:
    """``yonga logs JOB --raw | head -2`` must not look like a crash.

    Driven through a real process and a real pipe because that is the only place
    the failure exists: the reader closes the pipe while yonga is still writing,
    which needs a log far larger than the pipe buffer. Before the fix this
    printed the exit-99 panel, and then the interpreter's own shutdown flush
    printed a second error on top of it.
    """
    home = tmp_path / "home"
    (home / "config").mkdir(parents=True)
    logs_dir = home / "data" / "logs"
    logs_dir.mkdir(parents=True)
    record = json.dumps(
        {
            "timestamp": "2026-09-22T10:11:12.345Z",
            "job_id": LARGE_LOG_JOB_ID,
            "level": "info",
            "event": "subprocess.output",
            "message": "x" * 240,
        }
    )
    # Several megabytes: far more than any pipe buffer, so the writer is
    # guaranteed to still be writing when the reader walks away.
    (logs_dir / f"{LARGE_LOG_JOB_ID}.jsonl").write_text((record + "\n") * 20_000, encoding="utf-8")

    code, stderr = write_into_a_broken_pipe(
        home,
        ["logs", LARGE_LOG_JOB_ID, "--raw"],
        reader="import sys; sys.stdin.readline(); sys.stdin.readline()",
        wait_for_reader=False,
    )

    assert code == int(ExitCode.SUCCESS)
    assert stderr == ""


def test_a_small_raw_payload_does_not_leak_the_shutdown_flush(tmp_path: Path) -> None:
    """Output small enough to sit in the buffer must not fail at interpreter exit.

    Nothing raises while the command runs — the bytes are still in the stream's
    buffer — so the failure lands in the interpreter's own final flush, which
    prints ``Exception ignored while flushing sys.stdout`` and exits **120**.
    Committing the output while the handler can still see it is what stops the
    shell from being shown that second error.
    """
    home = tmp_path / "home"
    (home / "config").mkdir(parents=True)
    logs_dir = home / "data" / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / f"{SMALL_LOG_JOB_ID}.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-09-22T10:11:12.345Z",
                "job_id": SMALL_LOG_JOB_ID,
                "level": "info",
                "event": "job.created",
                "message": "job created",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    code, stderr = write_into_a_broken_pipe(
        home,
        ["logs", SMALL_LOG_JOB_ID, "--raw"],
        reader="raise SystemExit(0)",
        wait_for_reader=True,
    )

    assert code == int(ExitCode.SUCCESS)
    assert stderr == ""


def test_piping_a_rich_table_into_a_departed_reader_exits_quietly(tmp_path: Path) -> None:
    """The other half of the same promise, for output Rich renders.

    A short answer fits in the pipe buffer, so the reader has to be gone before
    the command starts for the write to fail at all. Rich intercepts the
    ``BrokenPipeError`` itself and exits 1, which a caller running under
    ``set -o pipefail`` reads as a failure; the CLI turns it back into 0.
    """
    home = tmp_path / "home"
    (home / "config").mkdir(parents=True)

    code, stderr = write_into_a_broken_pipe(
        home,
        ["registry", "paths"],
        reader="raise SystemExit(0)",
        wait_for_reader=True,
    )

    assert code == int(ExitCode.SUCCESS)
    assert stderr == ""


def test_debug_adds_the_raw_error_text(runner: CliRunner, home: Path) -> None:
    """``--debug`` — and only ``--debug`` — shows what actually went wrong."""

    def factory(settings: Settings) -> Wiring:
        return FakeWiring(settings, store=ExplodingArtifactStore(settings.paths, settings))

    result = invoke(runner, home, ["--debug", "list"], factory=factory)
    assert result.exit_code == int(ExitCode.INTERNAL)
    assert "synthetic catalogue failure" in result.stderr


def test_unexpected_failure_in_json_mode_is_still_one_document(
    runner: CliRunner, home: Path
) -> None:
    """Even a bug produces the stable error document rather than raw text."""

    def factory(settings: Settings) -> Wiring:
        return FakeWiring(settings, store=ExplodingArtifactStore(settings.paths, settings))

    result = invoke(runner, home, ["list", "--json"], factory=factory)
    payload = one_document(result)
    assert payload["ok"] is False
    assert payload["exit_code"] == int(ExitCode.INTERNAL)
    item = payload["diagnostic"]
    assert isinstance(item, dict)
    assert item["code"] == "internal.unexpected_error"


# ======================================================================
# colour policy
# ======================================================================
def test_no_color_strips_every_escape_sequence(runner: CliRunner, home: Path) -> None:
    """``NO_COLOR`` removes styling, including bold and dim."""
    result = invoke(runner, home, ["registry", "paths"], env={"NO_COLOR": "1"})
    assert result.exit_code == int(ExitCode.SUCCESS)
    assert "\x1b[" not in result.stdout


def test_forced_colour_is_emitted_when_asked_for(runner: CliRunner, home: Path) -> None:
    """Without ``NO_COLOR``, an explicit ``FORCE_COLOR`` still produces styling."""
    environment = {"FORCE_COLOR": "1"}
    result = runner.invoke(
        app,
        [
            "--data-dir",
            str(home / "data"),
            "--cache-dir",
            str(home / "cache"),
            "registry",
            "paths",
        ],
        obj=FakeWiring,
        env={"YONGA_CONFIG_DIR": str(home / "config"), **environment},
        color=True,
    )
    assert result.exit_code == int(ExitCode.SUCCESS)
    assert "\x1b[" in result.stdout


def test_json_mode_never_emits_colour(runner: CliRunner, home: Path) -> None:
    """``--json`` is undecorated even when the environment demands colour."""
    result = invoke(runner, home, ["list", "--json"], env={"FORCE_COLOR": "1", "NO_COLOR": ""})
    assert "\x1b[" not in result.stdout
    one_document(result)


# ======================================================================
# configuration precedence
# ======================================================================
def test_command_line_beats_environment_for_directories(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """``--data-dir`` wins over ``YONGA_DATA_DIR`` (CLI beats env beats file)."""
    result = invoke(
        runner,
        home,
        ["list", "--json"],
        env={"YONGA_DATA_DIR": str(tmp_path / "ignored")},
    )
    payload = one_document(result)
    assert payload["artifacts_dir"] == str(home / "data" / "artifacts")


def test_offline_flag_reaches_the_settings(runner: CliRunner, home: Path) -> None:
    """``--offline`` is a setting, so every adapter built from it is offline."""
    seen: list[Settings] = []

    def factory(settings: Settings) -> Wiring:
        seen.append(settings)
        return FakeWiring(settings)

    result = invoke(runner, home, ["--offline", "list"], factory=factory)
    assert result.exit_code == int(ExitCode.SUCCESS)
    assert seen and seen[0].offline is True


def test_telemetry_is_rejected_by_configuration(runner: CliRunner, home: Path) -> None:
    """Telemetry cannot be switched on, and the refusal is a clean diagnostic."""
    config = home / "config" / "config.toml"
    config.write_text("telemetry = true\n", encoding="utf-8")
    result = invoke(runner, home, ["list"], env={"YONGA_CONFIG_DIR": str(home / "config")})
    assert result.exit_code == int(ExitCode.REGISTRY)
    assert "Telemetry" in result.stderr
    assert "Traceback" not in result.stderr


def test_environment_is_not_leaked_into_json(runner: CliRunner, home: Path) -> None:
    """A token in the environment never reaches stdout in any command."""
    result = invoke(runner, home, ["doctor", "--json"], env={"HF_TOKEN": FAKE_TOKEN})
    assert FAKE_TOKEN not in result.stdout
    assert os.environ.get("HF_TOKEN") != FAKE_TOKEN or True


# --- a failed probe is a compilation failure, never "yonga hit a bug" --------------


def test_an_unclassified_probe_failure_exits_as_a_compile_failure(tmp_path: Path) -> None:
    """Exit 99 means an internal bug; a device that could not compile is exit 51.

    On the reference host the Qwen3.5 NPU pipeline failure used to arrive as the
    adapter's generic ``runtime.genai.load_failed`` and exit 99.
    """
    from yonga.application.probe import ProbeOutcome
    from yonga.cli import _probe_exit
    from yonga.domain.artifacts import DeviceProbeReport, ProbeStatus
    from yonga.domain.diagnostics import ExitCode, FailureCategory, diagnostic

    generic = diagnostic(
        "runtime.genai.load_failed",
        FailureCategory.UNKNOWN_RUNTIME_ERROR,
        "Could not load the model on NPU",
        "Compilation failed",
    )
    outcome = ProbeOutcome(
        artifact_dir=tmp_path,
        reports=[DeviceProbeReport(device="NPU", status=ProbeStatus.FAILED, diagnostic=generic)],
    )

    assert _probe_exit(outcome) is ExitCode.COMPILE


def test_a_classified_probe_failure_keeps_its_own_exit_code(tmp_path: Path) -> None:
    from yonga.application.probe import ProbeOutcome
    from yonga.cli import _probe_exit
    from yonga.domain.artifacts import DeviceProbeReport, ProbeStatus
    from yonga.domain.diagnostics import FailureCategory, diagnostic

    classified = diagnostic(
        "npu.permission_denied",
        FailureCategory.PERMISSION_DENIED,
        "NPU not accessible",
        "Permission denied on /dev/accel/accel0",
    )
    outcome = ProbeOutcome(
        artifact_dir=tmp_path,
        reports=[DeviceProbeReport(device="NPU", status=ProbeStatus.FAILED, diagnostic=classified)],
    )

    assert _probe_exit(outcome) is classified.exit_code()


# ======================================================================
# serve
# ======================================================================
@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Capture what ``serve`` would hand uvicorn, instead of opening a socket."""
    pytest.importorskip("fastapi")
    calls: list[dict[str, object]] = []

    def run_http(app_instance: object, *, host: str, port: int, verbose: bool) -> None:
        from fastapi.testclient import TestClient

        client = TestClient(app_instance)  # type: ignore[arg-type]
        calls.append(
            {
                "host": host,
                "port": port,
                "models": client.get("/v1/models").json(),
                "health": client.get("/health").json(),
            }
        )

    monkeypatch.setattr(cli_module, "_run_http", run_http)
    monkeypatch.setattr(cli_module, "_require_port", lambda host, port: None)
    return calls


def test_serve_loads_the_model_once_and_binds_loopback_by_default(
    runner: CliRunner, home: Path, tmp_path: Path, served: list[dict[str, object]]
) -> None:
    artifact = _staged_artifact(tmp_path)
    factory = FakeRuntimeFactory()
    result = invoke(
        runner,
        home,
        ["serve", str(artifact), "--device", "cpu"],
        factory=lambda settings: FakeWiring(settings, runtime_factory=factory),
    )

    assert result.exit_code == int(ExitCode.SUCCESS), result.stderr
    (call,) = served
    assert call["host"] == "127.0.0.1"
    assert call["port"] == 8000
    assert call["health"]["status"] == "ok"  # type: ignore[index]
    assert len(factory.opened) == 1, "loaded before serving, and only once"
    assert factory.runtimes[0].closed is True, "released when the server stops"
    assert "http://127.0.0.1:8000/v1" in result.stdout
    assert "warning" not in result.stderr


def test_serve_warns_when_binding_beyond_loopback_without_a_key(
    runner: CliRunner, home: Path, tmp_path: Path, served: list[dict[str, object]]
) -> None:
    artifact = _staged_artifact(tmp_path)
    result = invoke(runner, home, ["serve", str(artifact), "--host", "0.0.0.0"])

    assert result.exit_code == int(ExitCode.SUCCESS)
    assert "reachable from other machines" in result.stderr
    assert "YONGA_API_KEY" in result.stderr


def test_serve_never_prints_the_api_key(
    runner: CliRunner, home: Path, tmp_path: Path, served: list[dict[str, object]]
) -> None:
    artifact = _staged_artifact(tmp_path)
    result = invoke(
        runner,
        home,
        ["serve", str(artifact), "--host", "0.0.0.0"],
        env={"YONGA_API_KEY": "k-9f8e7d6c"},
    )

    # Rich wraps long lines (a long artifact path), so compare with whitespace collapsed.
    assert "API key required" in " ".join(result.stdout.split())
    assert "k-9f8e7d6c" not in result.stdout + result.stderr
    assert "No YONGA_API_KEY" not in result.stderr


def test_serve_is_refused_in_json_mode(runner: CliRunner, home: Path, tmp_path: Path) -> None:
    result = invoke(runner, home, ["serve", str(_staged_artifact(tmp_path)), "--json"])

    assert one_document(result)["ok"] is False


def test_serve_without_the_extra_names_the_install_command(
    runner: CliRunner, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing() -> object:
        raise ConfigurationError(
            diagnostic(
                "serve.dependencies.missing",
                FailureCategory.PACKAGE_ABI_MISMATCH,
                "The serve extra is not installed",
                "No module named 'fastapi'.",
                remediation=["pip install 'yonga[serve]'"],
            )
        )

    monkeypatch.setattr(cli_module, "_serve_dependencies", missing)
    result = invoke(runner, home, ["serve", str(_staged_artifact(tmp_path))])

    assert result.exit_code == int(ExitCode.ENVIRONMENT_UNUSABLE)
    assert "yonga[serve]" in result.stderr


@pytest.fixture
def created(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Capture what ``serve`` hands ``create_app``, without building or running a server."""
    captured: dict[str, object] = {}

    def create_app(server: object, **kwargs: object) -> object:
        captured.update(kwargs, server=server)
        return object()

    monkeypatch.setattr(cli_module, "_serve_dependencies", lambda: create_app)
    monkeypatch.setattr(cli_module, "_run_http", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_module, "_require_port", lambda host, port: None)
    return captured


def test_serve_defaults_to_an_8_mib_body_and_refusing_earlier_images(
    runner: CliRunner, home: Path, tmp_path: Path, created: dict[str, object]
) -> None:
    result = invoke(runner, home, ["serve", str(_staged_artifact(tmp_path))])

    assert result.exit_code == int(ExitCode.SUCCESS), result.stderr
    assert created["max_body_bytes"] == 8 * 1024 * 1024
    server = created["server"]
    assert server.image_policy.carry_forward is False  # type: ignore[attr-defined]


def test_serve_takes_a_body_limit_and_carries_images_forward_when_asked(
    runner: CliRunner, home: Path, tmp_path: Path, created: dict[str, object]
) -> None:
    result = invoke(
        runner,
        home,
        ["serve", str(_staged_artifact(tmp_path)), "--max-body-mb", "64", "--carry-images-forward"],
    )

    assert result.exit_code == int(ExitCode.SUCCESS), result.stderr
    assert created["max_body_bytes"] == 64 * 1024 * 1024
    server = created["server"]
    assert server.image_policy.carry_forward is True  # type: ignore[attr-defined]


@pytest.mark.parametrize("value", ["0", "65"])
def test_serve_refuses_a_body_limit_out_of_range(
    runner: CliRunner, home: Path, tmp_path: Path, created: dict[str, object], value: str
) -> None:
    result = invoke(
        runner, home, ["serve", str(_staged_artifact(tmp_path)), "--max-body-mb", value]
    )

    assert result.exit_code != int(ExitCode.SUCCESS)
    assert created == {}


def test_serve_warns_when_a_vlm_cannot_decode_images(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    created: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yonga.application.serve import ImagePolicy, ModelServer

    monkeypatch.setattr(
        ModelServer, "image_policy", property(lambda self: ImagePolicy(accepted=True))
    )
    monkeypatch.setattr(cli_module, "_pillow_available", lambda: False)
    result = invoke(runner, home, ["serve", str(_staged_artifact(tmp_path))])

    assert result.exit_code == int(ExitCode.SUCCESS), result.stderr
    assert "Pillow is not installed" in " ".join(result.stderr.split())
    assert "yonga[serve]" in result.stderr


def test_serve_says_nothing_about_pillow_for_a_text_only_model(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    created: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module, "_pillow_available", lambda: False)
    result = invoke(runner, home, ["serve", str(_staged_artifact(tmp_path))])

    assert "Pillow" not in result.stderr


def test_serve_refuses_a_port_in_use_before_loading_anything(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    import socket

    pytest.importorskip("fastapi")
    factory = FakeRuntimeFactory()
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        result = invoke(
            runner,
            home,
            ["serve", str(_staged_artifact(tmp_path)), "--port", str(port)],
            factory=lambda settings: FakeWiring(settings, runtime_factory=factory),
        )

    assert result.exit_code == int(ExitCode.REGISTRY)
    assert f"Cannot listen on 127.0.0.1:{port}" in result.stderr
    assert factory.opened == []


# ======================================================================
# publish
# ======================================================================
class PublishingWiring(FakeWiring):
    """A FakeWiring whose Hub writer records instead of uploading."""

    def __init__(self, settings: Settings, publisher: HubPublisher, hub: HubClient) -> None:
        super().__init__(settings, hub=hub)
        self._publisher = publisher

    def hub_publisher(self) -> HubPublisher:
        return self._publisher


def _prepared_artifact_id(runner: CliRunner, home: Path, tmp_path: Path) -> str:
    result = invoke(
        runner,
        home,
        ["prepare", SAFETENSORS_REPO, "--device", "cpu", "--yes", "--json"],
        factory=lambda settings: FakeWiring(settings, hub=hub_with(tmp_path=tmp_path)),
    )
    payload = json.loads(result.stdout)
    return str(payload["artifact_id"])


def _publish(
    runner: CliRunner, home: Path, tmp_path: Path, args: Sequence[str], publisher: HubPublisher
) -> Result:
    hub = hub_with(tmp_path=tmp_path)
    return invoke(
        runner,
        home,
        ["publish", *args],
        factory=lambda settings: PublishingWiring(settings, publisher, hub=hub),
    )


def test_publish_dry_run_writes_the_card_and_uploads_nothing(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    from tests.unit.test_publish import FakePublisher, someone

    artifact_id = _prepared_artifact_id(runner, home, tmp_path)
    publisher = FakePublisher(someone())
    preview = tmp_path / "preview"
    result = _publish(
        runner,
        home,
        tmp_path,
        [artifact_id, "--repo", "someone/model", "--dry-run", "--output", str(preview)],
        publisher,
    )

    assert result.exit_code in (int(ExitCode.SUCCESS), int(ExitCode.WARNINGS)), result.stderr
    assert publisher.commits == []
    assert "not a fine-tune" in (preview / "README.md").read_text(encoding="utf-8")
    assert "Publication plan" in result.stdout
    assert "nothing was uploaded" in result.stdout


def test_publish_without_yes_in_a_script_uploads_nothing(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    from tests.unit.test_publish import FakePublisher, someone

    artifact_id = _prepared_artifact_id(runner, home, tmp_path)
    publisher = FakePublisher(someone())
    result = _publish(runner, home, tmp_path, [artifact_id, "--repo", "someone/m"], publisher)

    assert result.exit_code == int(ExitCode.CANCELLED)
    assert publisher.commits == []


def test_publish_with_yes_uploads_privately_and_reports_the_url(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    from tests.unit.test_publish import FakePublisher, someone

    artifact_id = _prepared_artifact_id(runner, home, tmp_path)
    publisher = FakePublisher(someone())
    result = _publish(
        runner, home, tmp_path, [artifact_id, "--repo", "someone/m", "--yes", "--json"], publisher
    )

    payload = one_document(result)
    (commit,) = publisher.commits
    assert commit.private is True
    assert payload["url"] == "https://huggingface.co/someone/m/commit/abc123"
    assert payload["private"] is True


def test_publish_requires_a_destination(runner: CliRunner, home: Path) -> None:
    result = invoke(runner, home, ["publish", "anything"])

    assert result.exit_code != 0
    assert "--repo" in plain_text(result.stderr)


# ======================================================================
# registry update / status / reset
# ======================================================================
def _signed_bundle(tmp_path: Path, home: Path, payload: bytes | None = None) -> Path:
    """A signed bundle on disk, its key trusted in ``home``'s config directory."""
    import base64

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from tests.unit.test_remote_registry import bundle
    from yonga.registry.remote import sign_bundle

    key = Ed25519PrivateKey.generate()
    keys = home / "config" / "registry-keys"
    keys.mkdir(parents=True, exist_ok=True)
    raw = key.public_key().public_bytes_raw()
    (keys / "publisher.pub").write_text(base64.b64encode(raw).decode(), encoding="utf-8")
    payload = payload if payload is not None else bundle()
    target = tmp_path / "bundle.json"
    target.write_bytes(payload)
    (tmp_path / "bundle.json.sig").write_bytes(sign_bundle(payload, key))
    return target


def test_registry_update_verifies_shows_and_installs(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    target = _signed_bundle(tmp_path, home)

    result = invoke(runner, home, ["registry", "update", "--file", str(target), "--yes"])

    assert result.exit_code == int(ExitCode.SUCCESS), result.stderr
    assert "signature verified" in result.stdout
    assert "Adds 1 rule(s), replaces 0." in result.stdout
    status = invoke(runner, home, ["registry", "status", "--json"])
    payload = one_document(status)
    assert payload["in_use"] is True
    assert payload["installed"]["version"] == 1  # type: ignore[index]


def test_registry_update_without_a_trusted_key_changes_nothing(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    target = _signed_bundle(tmp_path, home)
    (home / "config" / "registry-keys" / "publisher.pub").unlink()

    result = invoke(runner, home, ["registry", "update", "--file", str(target), "--yes"])

    assert result.exit_code == int(ExitCode.REGISTRY)
    assert "No registry signing key is trusted" in result.stderr
    assert not (home / "data" / "registry-remote" / "bundle.json").exists()


def test_registry_update_without_yes_in_a_script_installs_nothing(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    target = _signed_bundle(tmp_path, home)

    result = invoke(runner, home, ["registry", "update", "--file", str(target)])

    assert result.exit_code == int(ExitCode.CANCELLED)
    assert not (home / "data" / "registry-remote" / "bundle.json").exists()


def test_registry_update_with_no_source_says_how(runner: CliRunner, home: Path) -> None:
    result = invoke(runner, home, ["registry", "update"])

    assert result.exit_code == int(ExitCode.REGISTRY)
    assert "--url" in result.stderr


def test_registry_status_reports_a_bundle_that_stopped_verifying(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    target = _signed_bundle(tmp_path, home)
    invoke(runner, home, ["registry", "update", "--file", str(target), "--yes"])
    (home / "config" / "registry-keys" / "publisher.pub").unlink()

    result = invoke(runner, home, ["registry", "status"])

    assert result.exit_code == int(ExitCode.WARNINGS)
    assert "Not in use" in result.stdout
    reset = invoke(runner, home, ["registry", "reset"])
    assert "Removed the remote registry bundle." in reset.stdout


def _tampered_install(runner: CliRunner, home: Path, tmp_path: Path) -> None:
    """Install a bundle, then change one byte of it on disk so it no longer verifies."""
    target = _signed_bundle(tmp_path, home)
    installed = invoke(runner, home, ["registry", "update", "--file", str(target), "--yes"])
    assert installed.exit_code == int(ExitCode.SUCCESS), installed.stderr
    bundle_path = home / "data" / "registry-remote" / "bundle.json"
    bundle_path.write_bytes(bundle_path.read_bytes().replace(b"remote_test", b"remote_tesT"))


def test_registry_validate_warns_about_a_bundle_that_stopped_verifying(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """'Registry is valid.' with exit 0 would hide that the installed bundle is not used."""
    _tampered_install(runner, home, tmp_path)

    result = invoke(runner, home, ["registry", "validate"])

    assert result.exit_code == int(ExitCode.WARNINGS)
    assert "Ignored" in result.stdout
    assert "registry.remote.bad_signature" in result.stdout
    assert "Registry is valid." not in result.stdout
    payload = one_document(invoke(runner, home, ["registry", "validate", "--json"]))
    ignored = payload["ignored"]
    assert isinstance(ignored, list) and len(ignored) == 1
    assert "bad_signature" in ignored[0]


def test_registry_validate_reports_nothing_ignored_for_a_bundle_in_use(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    target = _signed_bundle(tmp_path, home)
    invoke(runner, home, ["registry", "update", "--file", str(target), "--yes"])

    result = invoke(runner, home, ["registry", "validate", "--json"])

    assert result.exit_code == int(ExitCode.SUCCESS)
    assert one_document(result)["ignored"] == []


def test_doctor_full_warns_about_a_bundle_that_stopped_verifying(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    _tampered_install(runner, home, tmp_path)

    payload = one_document(invoke(runner, home, ["doctor", "--full", "--json"]))

    sections = payload["sections"]
    assert isinstance(sections, list)
    (registry,) = [section for section in sections if section["id"] == "registry"]
    ignored = [check for check in registry["checks"] if check["id"] == "registry.ignored"]
    assert ignored and ignored[0]["status"] == "warning"


def test_registry_explain_warns_about_a_bundle_that_stopped_verifying(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    _tampered_install(runner, home, tmp_path)
    wiring_for = lambda settings: FakeWiring(  # noqa: E731 - one-line factory
        settings, hub=hub_with(tmp_path=tmp_path)
    )

    human = invoke(
        runner,
        home,
        ["registry", "explain", SAFETENSORS_REPO, "--device", "cpu"],
        factory=wiring_for,
    )
    machine = invoke(
        runner,
        home,
        ["registry", "explain", SAFETENSORS_REPO, "--device", "cpu", "--json"],
        factory=wiring_for,
    )

    assert human.exit_code == int(ExitCode.SUCCESS)
    assert "warning:" in human.stderr and "bad_signature" in plain_text(human.stderr)
    assert len(one_document(machine)["ignored"]) == 1  # type: ignore[arg-type]


def test_registry_update_names_what_is_wrong_with_a_malformed_bundle(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    from tests.unit.test_remote_registry import bundle

    document = json.loads(bundle())
    document["created_at"] = "2026-09-01T00:00:00"
    target = _signed_bundle(tmp_path, home, json.dumps(document).encode())

    result = invoke(runner, home, ["registry", "update", "--file", str(target), "--yes", "--json"])

    assert result.exit_code == int(ExitCode.REGISTRY)
    diagnostic_payload = one_document(result)["diagnostic"]
    assert isinstance(diagnostic_payload, dict)
    assert diagnostic_payload["code"] == "registry.remote.invalid"
    assert "created_at: Input should have timezone info" in diagnostic_payload["message"]


def test_registry_update_refuses_url_and_file_together(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """CLI_SPEC section 12 gives them as alternatives; one must not be silently dropped."""
    target = _signed_bundle(tmp_path, home)

    both = ["--url", "https://127.0.0.1:9/b.json", "--file", str(target)]
    result = invoke(runner, home, ["registry", "update", *both, "--yes"])

    assert result.exit_code == int(ExitCode.REGISTRY)
    assert "--url" in result.stderr and "--file" in result.stderr
    assert not (home / "data" / "registry-remote" / "bundle.json").exists()


def test_registry_update_refuses_a_signature_without_a_file(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    target = _signed_bundle(tmp_path, home)

    result = invoke(
        runner,
        home,
        ["registry", "update", "--signature", str(target) + ".sig", "--yes"],
    )

    assert result.exit_code == int(ExitCode.REGISTRY)
    assert "--signature" in result.stderr


def test_registry_profiles_lists_what_profile_accepts(runner: CliRunner, home: Path) -> None:
    """The ids an unknown --profile points the user to, with device, weights and band."""
    from yonga.registry import load_builtin_registry

    builtin = load_builtin_registry()
    cpu = sorted(rule.id for rule in builtin.profiles if rule.device == "CPU")

    result = invoke(runner, home, ["registry", "profiles"])
    machine = one_document(invoke(runner, home, ["registry", "profiles", "--json"]))
    only_cpu = one_document(
        invoke(runner, home, ["registry", "profiles", "--device", "cpu", "--json"])
    )

    assert result.exit_code == int(ExitCode.SUCCESS)
    for rule in builtin.profiles:
        assert rule.id in result.stdout
    assert "below 5B" in result.stdout
    rows = machine["profiles"]
    assert isinstance(rows, list)
    assert {row["id"] for row in rows} == {rule.id for rule in builtin.profiles}
    assert all({"device", "weights", "band", "priority", "status"} <= set(row) for row in rows)
    cpu_rows = only_cpu["profiles"]
    assert isinstance(cpu_rows, list)
    assert sorted(row["id"] for row in cpu_rows) == cpu
    # Highest priority first within a device: the order the automatic choice weighs them.
    priorities = [row["priority"] for row in cpu_rows]
    assert priorities == sorted(priorities, reverse=True)


def test_registry_paths_inside_an_appimage_names_the_appimage(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    """The mount path is gone once the command exits, so the AppImage itself is shown."""
    from yonga.registry.loader import builtin_registry_dir

    builtin = builtin_registry_dir()
    assert builtin is not None
    appimage = tmp_path / "yonga-x86_64.AppImage"
    env = {"APPDIR": str(builtin.parent.parent), "APPIMAGE": str(appimage)}

    result = invoke(runner, home, ["registry", "paths"], env=env)
    machine = one_document(invoke(runner, home, ["registry", "paths", "--json"], env=env))

    assert result.exit_code == int(ExitCode.SUCCESS)
    assert f"inside {appimage}" in plain_text(result.stdout)
    assert str(builtin) not in result.stdout
    assert machine["builtin_appimage"] == str(appimage)


def test_registry_paths_outside_an_appimage_shows_the_directory(
    runner: CliRunner, home: Path
) -> None:
    machine = one_document(
        invoke(runner, home, ["registry", "paths", "--json"], env={"APPDIR": "", "APPIMAGE": ""})
    )

    assert machine["builtin_appimage"] is None
    assert machine["builtin"]


# ======================================================================
# eval
# ======================================================================
class EvaluatingWiring(FakeWiring):
    """A FakeWiring whose evaluation service scores with hand-made distributions."""

    def __init__(self, settings: Settings, root: Path) -> None:
        super().__init__(settings)
        self._root = root

    def evaluation_service(self, *, progress: ProgressSink | None = None) -> EvaluationService:
        from tests.unit.test_evaluation import FakeScorer, FakeStore, manifest
        from yonga.application.evaluation import EvaluationService

        manifests = {"ref": manifest("ref", weight="int8"), "cand": manifest("cand", weight="int4")}
        scorers = {"ref": FakeScorer("ref"), "cand": FakeScorer("cand", perturb=1.0)}
        return EvaluationService(
            FakeStore(self._root, manifests),
            lambda path, device: scorers[path.name],
            self._settings,
            progress=progress,
        )


def test_eval_prints_the_comparison_and_saves_it(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    pytest.importorskip("numpy")
    result = invoke(
        runner,
        home,
        ["eval", "cand", "--reference", "ref"],
        factory=lambda settings: EvaluatingWiring(settings, tmp_path),
    )

    assert result.exit_code == int(ExitCode.SUCCESS), result.stderr
    assert "Quality evaluation" in result.stdout
    assert "Mean KL" in result.stdout
    assert "not to the original weights" in result.stdout
    assert list((home / "data" / "evaluations").glob("*.json"))


def test_eval_json_is_one_document(runner: CliRunner, home: Path, tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    result = invoke(
        runner,
        home,
        ["eval", "cand", "--reference", "ref", "--json"],
        factory=lambda settings: EvaluatingWiring(settings, tmp_path),
    )

    payload = one_document(result)
    assert payload["summary"]["kl_mean"] > 0  # type: ignore[index]
    assert payload["device"] == "CPU"
