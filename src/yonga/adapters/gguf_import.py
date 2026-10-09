"""Import a GGUF file into an OpenVINO IR artifact with OpenVINO GenAI's own reader.

OpenVINO GenAI builds a model from a GGUF file in memory, and with
``enable_save_ov_model`` it writes the result -- ``openvino_model.xml/.bin``
and the tokenizer and detokenizer IRs -- **next to the file it was given**.
Given the file in the Hugging Face cache, it would write into the cache,
which yonga never modifies. So the importer hands it a symlink placed in the
staging directory instead: the IR lands in staging, and the cache is only
ever read.

The import runs in a child process, like the optimum-intel export, so a
native crash in the reader ends that process and not yonga, its output
streams into the job log, and Ctrl-C cancels it the same way. The saved IR
keeps the GGUF's quantization (a ``Q4_K`` tensor becomes ``u4`` weights), and
its tokenizer carries the same ``STRiNG`` precision defect optimum-intel's
does, which validation's registry repair then fixes: from here on it is an
ordinary artifact.

Run directly as ``python -m yonga.adapters.gguf_import FILE`` it performs
the import and exits; nothing else in yonga imports GenAI in this way.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from yonga.adapters import CommandRunner
from yonga.application.jobs import JobContext
from yonga.application.progress import ProgressKind
from yonga.domain.diagnostics import FailureCategory, diagnostic
from yonga.domain.plans import ExportResult
from yonga.util.redaction import redact_text
from yonga.util.versions import installed_version

__all__ = ["IMPORT_OUTPUTS", "GGUFImporter", "main"]

#: What a successful import leaves in the output directory.
IMPORT_OUTPUTS: tuple[str, ...] = (
    "openvino_model.xml",
    "openvino_model.bin",
    "openvino_tokenizer.xml",
    "openvino_tokenizer.bin",
    "openvino_detokenizer.xml",
    "openvino_detokenizer.bin",
)

_TAIL = 40


class GGUFImporter:
    """Turns one GGUF file into IR in a directory yonga owns."""

    def __init__(self, runner: CommandRunner, *, python: str | None = None) -> None:
        """Run imports with ``python`` (default: the interpreter yonga runs in)."""
        self._runner = runner
        self._python = python or sys.executable

    def import_file(self, gguf: Path, output_dir: Path, ctx: JobContext) -> ExportResult:
        """Import ``gguf`` into ``output_dir`` and report what happened."""
        output_dir.mkdir(parents=True, exist_ok=True)
        link = output_dir / gguf.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(gguf.resolve())
        argv = [self._python, "-P", "-m", "yonga.adapters.gguf_import", str(link)]
        ctx.logger.argv("gguf.import.command", argv, source=str(gguf))
        ctx.emit(
            ProgressKind.EXPORT, f"Importing {gguf.name} with OpenVINO GenAI", status="started"
        )
        started = time.perf_counter()
        try:
            result = self._runner.run(argv, cwd=output_dir, ctx=ctx)
        finally:
            link.unlink(missing_ok=True)
        missing = [name for name in IMPORT_OUTPUTS if not (output_dir / name).is_file()]
        success = result.returncode == 0 and not result.cancelled and not missing
        stdout = list(result.stdout_lines)[-_TAIL:]
        stderr = list(result.stderr_lines)[-_TAIL:]
        failure = None
        if not success:
            raw = redact_text("\n".join([*stdout, *stderr]))
            failure = diagnostic(
                "export.gguf_import_failed" if not result.cancelled else "export.cancelled",
                FailureCategory.USER_CANCELLED
                if result.cancelled
                else FailureCategory.UNKNOWN_RUNTIME_ERROR,
                "The GGUF import did not produce an OpenVINO model",
                (
                    f"OpenVINO GenAI exited with {result.returncode}"
                    + (f"; missing {', '.join(missing)}" if missing else "")
                    + "."
                ),
                remediation=[
                    "Check the job log for the reader's own message.",
                    "Try another quantization of the model with --gguf-file.",
                ],
                raw_excerpt=raw[-4000:] or None,
            )
        return ExportResult(
            success=success,
            argv=list(argv),
            returncode=result.returncode,
            duration_seconds=time.perf_counter() - started,
            stdout_tail=stdout,
            stderr_tail=stderr,
            package_versions=_versions(),
            diagnostic=failure,
        )


def _versions() -> dict[str, str]:
    found: dict[str, str] = {}
    for name in ("openvino", "openvino-genai", "openvino-tokenizers"):
        version = installed_version(name)
        if version:
            found[name] = version
    return found


def main(argv: Sequence[str] | None = None) -> int:
    """Import the GGUF file named in ``argv`` next to itself. Returns an exit code."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("usage: python -m yonga.adapters.gguf_import FILE.gguf", file=sys.stderr)
        return 2
    path = Path(arguments[0])
    if not path.is_symlink():
        # GenAI writes the IR next to the path it is given. Only a link placed in
        # a directory yonga owns guarantees that is not the Hugging Face cache.
        print("refusing: FILE must be a symlink in the output directory", file=sys.stderr)
        return 2
    import openvino_genai

    started = time.perf_counter()
    openvino_genai.LLMPipeline(str(path), "CPU", enable_save_ov_model=True)
    print(f"imported {path.name} in {time.perf_counter() - started:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    os._exit(main())  # skip interpreter teardown of the native pipeline objects
