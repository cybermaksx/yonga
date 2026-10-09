"""The generation smoke test in a child process, so a time limit can stop it.

``yonga probe --timeout SECONDS`` exists because a native NPU compile cannot be
interrupted from Python: once OpenVINO GenAI is building a pipeline the calling
thread does not return until the compiler does, which for a 9B model on the
reference host was more than fifteen minutes. A child process can be stopped at
any moment, so the bounded build happens in one.

:class:`ChildProcessSmokeTester` is the parent side. It starts
``python -P -m yonga.adapters.smoke_child`` through the
:class:`~yonga.adapters.CommandRunner` -- an argv list, never a shell, every
output line redacted into the job log -- and turns what comes back into a
:class:`~yonga.domain.artifacts.ProbeResult`:

* the child's own verdict, printed as one JSON line on stdout under the
  :data:`VERDICT_KEY` sentinel, when there is one: ``ok``, or ``failed`` with
  the raw runtime text for the registry. It stands even when the time limit
  ran out afterwards, while the child was still tearing down;
* :attr:`~yonga.domain.artifacts.ProbeStatus.INCONCLUSIVE` when the time limit
  stopped the build before a verdict. A stopped build says nothing about the
  graph, so it is never reported as a compiler failure;
* ``failed`` with the stderr tail when the child died without a verdict (a
  native crash, an import error), which the caller classifies.

Run directly, :func:`main` is the child: it runs the same in-process smoke test
``yonga probe`` uses without a time limit and prints the result. It never
imports OpenVINO at module scope, and its entry point is guarded because
OpenVINO may start a forkserver that re-imports the main module.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from yonga.adapters.subprocess import SubprocessRunner, default_line_selector
from yonga.application.progress import ProgressKind
from yonga.domain.artifacts import ProbeResult, ProbeStatus
from yonga.domain.diagnostics import CancelledError, FailureCategory, diagnostic
from yonga.domain.hardware import normalise_device
from yonga.domain.models import PipelineKind
from yonga.util.redaction import environment_secrets, redact_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yonga.adapters import CommandRunner
    from yonga.application.jobs import JobContext
    from yonga.config.settings import Settings

__all__ = [
    "SMOKE_CHILD_MODULE",
    "VERDICT_KEY",
    "ChildProcessSmokeTester",
    "main",
    "result_from_stdout",
    "smoke_child_runner",
    "smoke_line_selector",
    "verdict_line",
]

#: The module the child process runs; ``python -P -m`` keeps the working
#: directory off ``sys.path``, so a stray file there cannot shadow yonga.
SMOKE_CHILD_MODULE = "yonga.adapters.smoke_child"

#: The one key of the object the child prints its verdict in. Native libraries
#: write to stdout too, and a JSON object of theirs must never pass for a result.
VERDICT_KEY = "yonga_smoke_result"

#: How many trailing stderr lines a failure without a verdict keeps.
_STDERR_TAIL = 40

#: The statuses a child's verdict can carry; anything else is not a verdict.
_VERDICT_STATUSES = frozenset({ProbeStatus.OK.value, ProbeStatus.FAILED.value})


def verdict_line(result: ProbeResult) -> str:
    """Return the line the child prints its result as.

    Pure ASCII (``ensure_ascii``), so it survives a child whose locale is not
    UTF-8: the parent always decodes the pipe as UTF-8, and a message quoting a
    path or a prompt must not fail to encode after the smoke test ran.
    """
    return json.dumps({VERDICT_KEY: result.model_dump(mode="json")}, ensure_ascii=True)


def smoke_line_selector(line: str) -> ProgressKind | None:
    """Classify one line of the child's output for the probe's progress display.

    A warning or error is shown as a log line, as for any child process. The
    verdict line never is -- the probe reports the result itself -- and nothing
    is ever an ``EXPORT`` milestone: a compiler's "loading weights" is not a
    conversion step, and a probe has no export task to advance.
    """
    text = line.strip()
    if not text or text.startswith("{") or VERDICT_KEY in text:
        return None
    return ProgressKind.LOG if default_line_selector(text) is ProgressKind.LOG else None


def smoke_child_runner() -> SubprocessRunner:
    """Return a runner for the child: supervised, redacting, with its own line selector.

    The exporter's runner reports conversion milestones; this one must not, and
    it keeps the verdict line off the console (:func:`smoke_line_selector`).
    """
    return SubprocessRunner(line_selector=smoke_line_selector, extra_secrets=environment_secrets())


class ChildProcessSmokeTester:
    """Runs the generation smoke test in a child process it can stop.

    Implements :class:`yonga.application.probe.IsolatedSmokeTester`.
    """

    def __init__(
        self,
        runner: CommandRunner,
        *,
        python: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Run children with ``python`` (default: the interpreter yonga runs in).

        ``settings`` carries the parent's cache directory and compile-cache
        switch to the child, which would otherwise build its pipeline against
        the default cache directory whatever ``--cache-dir`` said.
        """
        self._runner = runner
        self._python = python or sys.executable
        self._settings = settings

    def argv(
        self,
        artifact_dir: Path,
        *,
        device: str,
        pipeline: PipelineKind,
        properties: Mapping[str, str],
        prompt: str,
        max_new_tokens: int,
    ) -> list[str]:
        """Return the child's command line: an argv list, with no shell anywhere."""
        command = [
            self._python,
            "-P",
            "-m",
            SMOKE_CHILD_MODULE,
            str(artifact_dir),
            "--device",
            device,
            "--pipeline",
            pipeline.value,
            "--max-new-tokens",
            str(max_new_tokens),
            "--prompt",
            prompt,
        ]
        for key, value in properties.items():
            command += ["--property", f"{key}={value}"]
        if self._settings is not None:
            command += ["--cache-dir", str(self._settings.paths.cache_dir)]
            if not self._settings.compile_cache_enabled:
                command.append("--no-compile-cache")
        return command

    def smoke_test(
        self,
        artifact_dir: Path,
        *,
        device: str,
        pipeline: PipelineKind,
        properties: Mapping[str, str],
        prompt: str,
        max_new_tokens: int,
        timeout: float,
        ctx: JobContext | None = None,
    ) -> ProbeResult:
        """Build the pipeline in a child, giving up after ``timeout`` seconds.

        Raises :class:`~yonga.domain.diagnostics.CancelledError` when the job was
        cancelled or interrupted while the child ran: that is the user's
        decision, not a probe result.
        """
        command = self.argv(
            artifact_dir,
            device=device,
            pipeline=pipeline,
            properties=properties,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        result = self._runner.run(command, timeout=timeout, ctx=ctx)
        if result.cancelled:
            raise CancelledError(
                diagnostic(
                    "probe.cancelled",
                    FailureCategory.USER_CANCELLED,
                    "The probe was cancelled",
                    f"The GenAI pipeline build on {device} was stopped before it finished.",
                    remediation=["Run the probe again to settle the device."],
                    context={"device": device},
                )
            )
        verdict = result_from_stdout(result.stdout_lines, device=device)
        if verdict is not None:
            # The verdict is trusted over the exit code, and over the time limit:
            # OpenVINO can crash, or take its time releasing the device, while the
            # interpreter tears down after the pipeline already generated.
            return verdict
        if result.timed_out:
            return ProbeResult(
                device=device,
                status=ProbeStatus.INCONCLUSIVE,
                duration_seconds=result.duration_seconds,
                message=(
                    f"the GenAI pipeline build did not finish within {timeout:g} s and was stopped"
                ),
            )
        tail = [line for line in result.stderr_lines if line.strip()][-_STDERR_TAIL:]
        message = f"the smoke-test process exited with {result.returncode} without a result"
        if tail:
            message += ":\n" + "\n".join(tail)
        return ProbeResult(
            device=device,
            status=ProbeStatus.FAILED,
            duration_seconds=result.duration_seconds,
            message=redact_text(message),
        )


def result_from_stdout(lines: Sequence[str], *, device: str) -> ProbeResult | None:
    """Return the child's verdict for ``device`` from its stdout, if it printed one.

    Searched from the end, line by line, because native libraries may print to
    stdout too -- even after the verdict, when their buffers are flushed at
    exit. Only an object under :data:`VERDICT_KEY` counts, and only one that
    states ``ok`` or ``failed`` for the device that was asked for; any other
    line, JSON or not, is skipped.
    """
    wanted = normalise_device(device)
    for line in reversed(lines):
        text = line.strip()
        if not text.startswith("{") or VERDICT_KEY not in text:
            continue
        try:
            payload = json.loads(text)
        except ValueError:
            continue
        if not isinstance(payload, dict) or set(payload) != {VERDICT_KEY}:
            continue
        body = payload[VERDICT_KEY]
        if not isinstance(body, dict) or body.get("status") not in _VERDICT_STATUSES:
            continue
        with suppress(ValidationError):
            verdict = ProbeResult.model_validate(body)
            if verdict.device == wanted:
                return verdict
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Run one smoke test and print its result as a single JSON line on stdout.

    Exit 0 when the pipeline generated, 1 when it failed, 2 for a usage error.
    Native output is sent to stderr while the pipeline runs, so stdout carries
    the verdict and as little else as possible.
    """
    import argparse

    from yonga.application.prepare import SmokeTester
    from yonga.application.probe import guard_smoke_test
    from yonga.application.wiring import GenAISmokeTester
    from yonga.config.settings import Settings

    parser = argparse.ArgumentParser(
        prog=f"python -m {SMOKE_CHILD_MODULE}",
        description="Build an OpenVINO GenAI pipeline, generate a few tokens, print JSON.",
    )
    parser.add_argument("artifact", type=Path, help="prepared artifact directory")
    parser.add_argument("--device", required=True, help="device to build the pipeline on")
    parser.add_argument(
        "--pipeline",
        default=PipelineKind.UNKNOWN.value,
        choices=[kind.value for kind in PipelineKind],
        help="pipeline kind recorded in the manifest",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8, help="tokens to generate")
    parser.add_argument("--prompt", default="Hello", help="prompt to generate from")
    parser.add_argument(
        "--property",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="a GenAI pipeline property (repeatable)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="the parent's cache directory, where the compile cache lives",
    )
    parser.add_argument(
        "--no-compile-cache",
        action="store_true",
        help="build the pipeline without the managed compile cache",
    )
    args = parser.parse_args(None if argv is None else list(argv))
    settings: Settings | None = None
    if args.cache_dir is not None or args.no_compile_cache:
        overrides: dict[str, object] = {"compile_cache_enabled": not args.no_compile_cache}
        if args.cache_dir is not None:
            overrides["cache_dir"] = args.cache_dir
        settings = Settings.model_validate(overrides)
    properties: dict[str, str] = {}
    for item in args.property:
        key, separator, value = str(item).partition("=")
        if not separator or not key:
            parser.error(f"--property expects KEY=VALUE, got {item!r}")
        properties[key] = value

    tester: SmokeTester = GenAISmokeTester(settings)
    with _stdout_to_stderr():
        result = guard_smoke_test(
            args.device,
            lambda: tester.smoke_test(
                args.artifact,
                device=args.device,
                pipeline=PipelineKind(args.pipeline),
                properties=properties,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
            ),
        )
    sys.stdout.write(verdict_line(result) + "\n")
    sys.stdout.flush()
    return 0 if result.ok else 1


@contextmanager
def _stdout_to_stderr() -> Iterator[None]:
    """Point file descriptor 1 at stderr for the duration of a ``with`` block."""
    try:
        sys.stdout.flush()
        saved: int | None = os.dup(1)
        os.dup2(2, 1)
    except (OSError, ValueError):
        saved = None
    try:
        yield
    finally:
        if saved is not None:
            with suppress(OSError, ValueError):
                sys.stdout.flush()
                os.dup2(saved, 1)
                os.close(saved)


# OpenVINO may start a multiprocessing forkserver whose child re-imports the main
# module, so the entry point stays guarded (docs/MODEL_PIPELINE.md section 10).
if __name__ == "__main__":  # pragma: no cover - runs only as the child process
    import signal

    # Nothing in the child needs cleaning up, so an interrupt ends it at once
    # instead of waiting for a native compile to return to the interpreter.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    raise SystemExit(main())
