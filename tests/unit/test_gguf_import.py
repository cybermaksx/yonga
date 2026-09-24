"""The GGUF importer and the prepare path around it.

The importer's one hard rule: GenAI writes the IR next to the path it is
given, so it is only ever given a symlink inside a directory yonga owns --
never the file in the Hugging Face cache. The prepare tests run a whole
preparation with a fake importer that writes what GenAI writes, including
the ``STRiNG`` tokenizer defect, so validation and repair are exercised too.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pytest

from tests.unit.test_gguf import gguf_bytes, reader_for
from tests.unit.test_prepare_planning import (
    MODEL_XML,
    FakeHub,
    build_service,
    run_prepare,
    settings_for,
    write_tokenizer_pair,
)
from yonga.adapters.gguf_header import parse_gguf_header
from yonga.adapters.gguf_import import IMPORT_OUTPUTS, GGUFImporter, main
from yonga.adapters.hf import detect_formats
from yonga.adapters.subprocess import CompletedCommand
from yonga.application.jobs import JobContext, open_job
from yonga.application.prepare import PrepareService
from yonga.config.settings import Settings
from yonga.domain.diagnostics import ExportError, SourceResolutionError
from yonga.domain.models import ModelIdentity, ModelMetadata, RepoFile
from yonga.domain.plans import ExportResult
from yonga.registry.loader import load_builtin_registry
from yonga.registry.resolver import RegistryResolver

REPO = "acme/tiny-GGUF"
SHA = "d" * 40


def write_import(directory: Path) -> None:
    """What a GenAI GGUF import leaves behind, tokenizer defect included."""
    (directory / "openvino_model.xml").write_text(MODEL_XML, encoding="utf-8")
    (directory / "openvino_model.bin").write_bytes(b"\x00" * 32)
    write_tokenizer_pair(directory, broken=True)


class ImportRunner:
    """A CommandRunner that behaves like ``python -m yonga.adapters.gguf_import``."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[list[str]] = []
        self.link_seen: tuple[bool, Path] | None = None

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float | None = None,
        ctx: JobContext | None = None,
    ) -> CompletedCommand:
        command = list(argv)
        self.calls.append(command)
        link = Path(command[-1])
        self.link_seen = (link.is_symlink(), link.resolve())
        if self.fail:
            return CompletedCommand(
                argv=command,
                returncode=1,
                duration_seconds=0.1,
                stderr_lines=["[load_gguf] gguf_tensor_to_f16 failed"],
            )
        write_import(link.parent)
        return CompletedCommand(argv=command, returncode=0, duration_seconds=0.1)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return settings_for(tmp_path)


@pytest.fixture
def ctx(settings: Settings) -> Iterator[JobContext]:
    with open_job(settings=settings, requested_model=REPO) as job:
        yield job


def test_the_importer_is_given_a_link_in_staging_never_the_cached_file(
    tmp_path: Path, ctx: JobContext
) -> None:
    cached = tmp_path / "hf-cache" / "model-Q4_0.gguf"
    cached.parent.mkdir()
    cached.write_bytes(b"GGUF")
    runner = ImportRunner()

    result = GGUFImporter(runner, python="python3").import_file(cached, tmp_path / "out", ctx)

    assert result.success
    assert runner.link_seen == (True, cached.resolve())
    assert runner.calls[0][:3] == ["python3", "-m", "yonga.adapters.gguf_import"]
    assert sorted(p.name for p in cached.parent.iterdir()) == ["model-Q4_0.gguf"], (
        "nothing was written next to the cached file"
    )
    assert not (tmp_path / "out" / "model-Q4_0.gguf").exists(), "the link is removed afterwards"
    assert all((tmp_path / "out" / name).is_file() for name in IMPORT_OUTPUTS)


def test_a_failed_import_carries_the_readers_message(tmp_path: Path, ctx: JobContext) -> None:
    cached = tmp_path / "m.gguf"
    cached.write_bytes(b"GGUF")

    result = GGUFImporter(ImportRunner(fail=True)).import_file(cached, tmp_path / "out", ctx)

    assert not result.success
    assert result.diagnostic is not None
    assert "gguf_tensor_to_f16 failed" in (result.diagnostic.raw_excerpt or "")


def test_the_entry_point_refuses_a_path_that_is_not_a_link(tmp_path: Path) -> None:
    """Pointed at the real file, GenAI would write into the directory holding it."""
    real = tmp_path / "m.gguf"
    real.write_bytes(b"GGUF")

    assert main([str(real)]) == 2


# --- prepare --------------------------------------------------------------------------


class GGUFHub(FakeHub):
    """A FakeHub whose repository is GGUF-only and whose headers are canned."""

    def __init__(self, tmp_path: Path, headers: dict[str, bytes]) -> None:
        files = [RepoFile(path=name, size=1000 + i, lfs=True) for i, name in enumerate(headers)]
        files.append(RepoFile(path="README.md", size=100))
        metadata = ModelMetadata(
            identity=ModelIdentity(repo_id=REPO, revision=SHA),
            resolved_revision=SHA,
            files=files,
            formats=detect_formats(files),
            gguf_metadata={"architecture": "qwen2"},
            pipeline_tag="text-generation",
            license="apache-2.0",
        )
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        for name, data in headers.items():
            (snapshot / name).write_bytes(data)
        super().__init__(repos={REPO: metadata}, snapshots={REPO: snapshot})
        self.headers = headers
        self.allow_patterns: list[object] = []

    def gguf_header(self, model: ModelIdentity, filename: str) -> object:
        return parse_gguf_header(reader_for(self.headers[filename]))

    def download(self, source: ModelIdentity, **kwargs: object) -> Path:
        self.allow_patterns.append(kwargs.get("allow_patterns"))
        return super().download(source)


class FakeImporter:
    def __init__(self) -> None:
        self.imported: list[Path] = []

    def import_file(self, gguf: Path, output_dir: Path, ctx: JobContext) -> ExportResult:
        self.imported.append(gguf)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_import(output_dir)
        return ExportResult(success=True, argv=["python", "-m", "yonga.adapters.gguf_import"])


def headers() -> dict[str, bytes]:
    return {
        "tiny-Q4_K_M.gguf": gguf_bytes("qwen2", {"F32": 2, "Q4_K": 2, "Q5_0": 4}),
        "tiny-Q4_0.gguf": gguf_bytes("qwen2", {"F32": 2, "Q4_0": 6, "Q8_0": 1}),
    }


def gguf_service(settings: Settings, hub: GGUFHub, importer: FakeImporter) -> PrepareService:
    resolver = RegistryResolver(
        load_builtin_registry(), versions={"openvino_genai": "2026.4.0", "openvino": "2026.4.0"}
    )
    return build_service(settings=settings, hub=hub, resolver=resolver, gguf_importer=importer)


def test_a_gguf_repository_is_imported_validated_and_repaired(
    settings: Settings, tmp_path: Path
) -> None:
    hub = GGUFHub(tmp_path, headers())
    importer = FakeImporter()
    service = gguf_service(settings, hub, importer)

    plan = service.plan(REPO, device="cpu")
    outcome = run_prepare(service, plan, settings)

    assert plan.resolved_source.gguf_file == "tiny-Q4_0.gguf"
    assert plan.disk.expected_download_bytes == 1001, "only the chosen file is fetched"
    assert hub.allow_patterns == [["tiny-Q4_0.gguf"]]
    assert plan.artifact_dir.name.endswith("--gguf-q4_0-cpu")
    assert any("Passed over tiny-Q4_K_M.gguf: holds Q5_0" in n for n in plan.known_limitations)
    assert any("keeps its own quantization" in n for n in plan.known_limitations)
    assert plan.export_environment is None and not plan.exporter_requirements_block
    manifest = outcome.manifest
    assert manifest.export.backend == "openvino-genai-gguf"
    assert manifest.export.profile == "gguf-q4_0-cpu"
    assert manifest.export.arguments["tensor_types"] == {"F32": 2, "Q4_0": 6, "Q8_0": 1}
    assert [r.repair_id for r in outcome.repairs] == ["ov-tokenizer-precision-string-casing"]
    assert importer.imported[0].name == "tiny-Q4_0.gguf"


def test_two_files_of_one_repository_are_two_artifacts(settings: Settings, tmp_path: Path) -> None:
    both = headers()
    both["tiny-Q8_0.gguf"] = gguf_bytes("qwen2", {"F32": 2, "Q8_0": 7})
    hub = GGUFHub(tmp_path, both)
    service = gguf_service(settings, hub, FakeImporter())

    first = service.plan(REPO, device="cpu")
    second = service.plan(REPO, device="cpu", gguf_file="tiny-Q8_0.gguf")
    a = run_prepare(service, first, settings)
    b = run_prepare(service, second, settings)

    assert a.artifact_dir != b.artifact_dir
    assert a.manifest.artifact_id != b.manifest.artifact_id


def test_a_repository_with_no_importable_file_refuses_before_downloading(
    settings: Settings, tmp_path: Path
) -> None:
    only_bad = {"tiny-Q4_K_M.gguf": gguf_bytes("qwen2", {"F32": 2, "Q5_0": 4})}
    hub = GGUFHub(tmp_path, only_bad)
    service = gguf_service(settings, hub, FakeImporter())

    with pytest.raises(SourceResolutionError) as caught:
        service.plan(REPO, device="cpu")

    assert "Q5_0" in caught.value.diagnostic.message
    assert hub.allow_patterns == [] and hub.downloaded == []


def test_a_failed_import_is_a_conversion_failure(settings: Settings, tmp_path: Path) -> None:
    class Failing(FakeImporter):
        def import_file(self, gguf: Path, output_dir: Path, ctx: JobContext) -> ExportResult:
            return ExportResult(success=False, argv=["python"], returncode=1)

    hub = GGUFHub(tmp_path, headers())
    service = gguf_service(settings, hub, Failing())
    plan = service.plan(REPO, device="cpu")

    with pytest.raises(ExportError):
        run_prepare(service, plan, settings)


def test_the_npu_reshape_failure_of_a_gguf_import_is_not_blamed_on_the_driver() -> None:
    """The exact text the Lunar Lake NPU produced for a GGUF-built Qwen2.5 graph."""
    raw = (
        "Exception from src/core/shape_inference/include/broadcast_shape_inference.hpp:150:\n"
        "While validating node 'opset3::Broadcast Broadcast_179': Broadcast incorrect "
        "target shape. Expecting either 1 or 1024. Got 0"
    )

    found = RegistryResolver(load_builtin_registry()).classify(raw_error=raw, device="NPU")

    assert found is not None
    assert found.code == "npu.pipeline.static_reshape_failed"
    assert found.environment_broken is False
    assert "driver" not in " ".join(found.remediation).lower()
