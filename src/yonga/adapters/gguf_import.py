"""Import a GGUF file into an OpenVINO IR artifact with OpenVINO GenAI's own reader.

OpenVINO GenAI builds a model from a GGUF file in memory, and with
``enable_save_ov_model`` it writes the result -- ``openvino_model.xml/.bin``
and the tokenizer and detokenizer IRs -- **next to the file it was given**.
Given the file in the Hugging Face cache, it would write into the cache,
which yonga never modifies. So the importer hands it a private copy placed in
the staging directory instead: the IR lands in staging, the copy is deleted
afterwards, and the cache is only ever read.

A copy, not a link: the reader opens the file read-write (``O_RDWR |
O_APPEND``) although it never writes to it, and huggingface-hub 1.x stores
its blobs read-only (mode 0444), so a link to the cached blob fails with
"Failed to open ... with gguf_open". The copy is a reflink where the
filesystem supports one (btrfs, XFS), costing no time and no space, and a
byte copy otherwise. It is a new inode either way, so nothing the reader does
to it can reach the cache.

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

import fcntl
import os
import shutil
import stat
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

__all__ = ["IMPORT_OUTPUTS", "GGUFImporter", "main", "working_copy"]

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

#: ``FICLONE`` from ``linux/fs.h``; the ``fcntl`` module names it from Python 3.12.
_FICLONE: int = getattr(fcntl, "FICLONE", 0x40049409)


class GGUFImporter:
    """Turns one GGUF file into IR in a directory yonga owns."""

    def __init__(self, runner: CommandRunner, *, python: str | None = None) -> None:
        """Run imports with ``python`` (default: the interpreter yonga runs in)."""
        self._runner = runner
        self._python = python or sys.executable

    def import_file(self, gguf: Path, output_dir: Path, ctx: JobContext) -> ExportResult:
        """Import ``gguf`` into ``output_dir`` and report what happened."""
        output_dir.mkdir(parents=True, exist_ok=True)
        copy = output_dir / gguf.name
        argv = [self._python, "-P", "-m", "yonga.adapters.gguf_import", str(copy)]
        started = time.perf_counter()
        ctx.emit(ProgressKind.EXPORT, f"Copying {gguf.name} into staging", status="copying")
        try:
            how = working_copy(gguf, copy)
        except OSError as error:
            copy.unlink(missing_ok=True)
            return self._copy_failed(gguf, copy, error, argv, started)
        except BaseException:
            copy.unlink(missing_ok=True)  # an interrupted copy is never imported
            raise
        ctx.logger.info(
            "gguf.import.copy", f"{how} {gguf.name} into staging", source=str(gguf), method=how
        )
        ctx.logger.argv("gguf.import.command", argv, source=str(gguf))
        ctx.emit(
            ProgressKind.EXPORT, f"Importing {gguf.name} with OpenVINO GenAI", status="started"
        )
        try:
            result = self._runner.run(argv, cwd=output_dir, ctx=ctx)
        finally:
            copy.unlink(missing_ok=True)
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

    @staticmethod
    def _copy_failed(
        gguf: Path, copy: Path, error: OSError, argv: list[str], started: float
    ) -> ExportResult:
        """The working copy could not be written: nothing was imported."""
        raw = redact_text(f"{type(error).__name__}: {error}")
        return ExportResult(
            success=False,
            argv=list(argv),
            returncode=1,
            duration_seconds=time.perf_counter() - started,
            stderr_tail=[raw],
            package_versions=_versions(),
            diagnostic=diagnostic(
                "export.gguf_copy_failed",
                FailureCategory.UNKNOWN_RUNTIME_ERROR,
                "The GGUF file could not be copied into staging",
                f"Copying {gguf.name} to {copy.parent} for the import failed: {error}.",
                remediation=["Check free space and permissions on the data filesystem."],
                raw_excerpt=raw,
            ),
        )


def working_copy(source: Path, destination: Path) -> str:
    """Write ``source`` to ``destination`` as a new file this user may write.

    Returns how: ``"reflinked"`` when the filesystem shared the extents, or
    ``"copied"``. ``destination`` is replaced, never written through: a link
    left there by an earlier run is removed first, so the cache file it may
    point at is not truncated.
    """
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    how = "copied"
    with source.open("rb") as reader, destination.open("xb") as writer:
        # A new file gets 0666 less the umask; the owner must be able to read and
        # write it whatever the umask, since the reader opens it read-write.
        mode = stat.S_IMODE(os.fstat(writer.fileno()).st_mode)
        os.fchmod(writer.fileno(), mode | stat.S_IRUSR | stat.S_IWUSR)
        try:
            fcntl.ioctl(writer.fileno(), _FICLONE, reader.fileno())
            how = "reflinked"
        except OSError:
            pass
    if how == "copied":
        shutil.copyfile(source, destination)
    return how


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
    if not _is_private_copy(path):
        # GenAI writes the IR next to the path it is given, and opens the file
        # read-write. Only a private copy placed in a directory yonga owns keeps
        # both away from the Hugging Face cache, whose snapshot entries are links
        # and whose blobs a hard link would share.
        print(
            "refusing: FILE must be a private, writable copy in the output directory",
            file=sys.stderr,
        )
        return 2
    import openvino_genai

    started = time.perf_counter()
    openvino_genai.LLMPipeline(str(path), "CPU", enable_save_ov_model=True)
    print(f"imported {path.name} in {time.perf_counter() - started:.1f} s", flush=True)
    return 0


def _is_private_copy(path: Path) -> bool:
    """True for a regular file, not a link, with no other name, that this user may write."""
    if path.is_symlink():
        return False
    try:
        info = path.stat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and os.access(path, os.W_OK)


if __name__ == "__main__":
    os._exit(main())  # skip interpreter teardown of the native pipeline objects
