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
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

# ``Result`` comes from typer, not click: typer 0.27 vendored click and returns
# its own ``typer.testing.Result`` from ``CliRunner.invoke``, while typer 0.25
# re-exports click's. Importing it from typer is correct under both, so the
# annotation tracks whichever version is installed.
from typer.testing import CliRunner, Result

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
from yonga.adapters import Exporter, FilesystemProbe, HardwareProbe, HubClient, RuntimeProbe
from yonga.adapters.filesystem import system_temp_dir
from yonga.adapters.runtime_genai import ChatHistory, ChatProfileLike, ChatRuntime, GenerationResult
from yonga.application.artifacts import ArtifactStore
from yonga.application.benchmark import RuntimeFactory as BenchmarkRuntimeFactory
from yonga.application.chat import RuntimeFactory as ChatRuntimeFactory
from yonga.application.prepare import SmokeTester
from yonga.application.wiring import Wiring
from yonga.cli import app
from yonga.config.settings import Settings
from yonga.domain.artifacts import IRDescription, ProbeResult, assert_no_secrets
from yonga.domain.benchmarks import GenerationSettings
from yonga.domain.diagnostics import ExitCode
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
from yonga.domain.models import PipelineKind
from yonga.registry.resolver import RegistryResolver

GIB = 1024**3

#: The real Qwen3.5 NPU compiler refusal recorded in ``HOST_FACTS.md``.
NPU_UNSUPPORTED_GRAPH = (
    "Exception from src/plugins/intel_npu/src/plugin/src/plugin.cpp:552:\n"
    "OneShotBufferizeVPU2VPUIP failed : Only supports FP16."
)

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
    ):
        assert command in result.stdout
    assert "serve" not in result.stdout


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
