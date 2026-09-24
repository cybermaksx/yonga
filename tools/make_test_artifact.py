#!/usr/bin/env python3
"""Regenerate the tiny OpenVINO artifact the integration tests use.

The artifact is a real ``optimum-cli export openvino`` run against
``hf-internal-testing/tiny-random-LlamaForCausalLM``: about 7 MB of genuine
OpenVINO IR, including the ``precision="STRiNG"`` tokenizer defect that current
OpenVINO 2026.4 emits. It is generated rather than committed so that cloning the
repository does not pull megabytes of binary weights, and regenerated rather than
hand-written so the tests keep exercising bytes an exporter actually produced.

Run it once before the ``openvino``-marked tests::

    python tools/make_test_artifact.py

It needs network access on first run (the source checkpoint is a few hundred
kilobytes) and the ``export`` extra installed. Output goes to
``tests/fixtures/.cache/tinyllm`` unless ``--output`` says otherwise; that path
is git-ignored. Set ``YONGA_TEST_BROKEN_ARTIFACT`` to point the suite elsewhere.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "fixtures" / ".cache" / "tinyllm"
SOURCE_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

#: The NPU artifact: a small *real* LLM, because the GenAI NPU pipeline rejects
#: the tiny random Llama's toy shapes. Exported with the npu-int4-sym-g128
#: profile's flags; verified on the reference host to load and generate through
#: LLMPipeline on the NPU. About 400 MB of IR from a ~1.4 GB download.
NPU_OUTPUT = REPO_ROOT / "tests" / "fixtures" / ".cache" / "npu-llm"
NPU_SOURCE_MODEL = "Qwen/Qwen3-0.6B"
TOKENIZER_IRS = ("openvino_tokenizer.xml", "openvino_detokenizer.xml")
DEFECT = 'precision="STRiNG"'


def build(output: Path, *, source: str = SOURCE_MODEL, group_size: int = -1) -> int:
    """Export a model and report whether the expected tokenizer defect is present."""
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.parent / f"{output.name}.tmp"
    argv = [
        sys.executable,
        "-m",
        "optimum.commands.optimum_cli",
        "export",
        "openvino",
        "--model",
        source,
        "--task",
        "text-generation-with-past",
        "--weight-format",
        "int4",
        "--sym",
        "--ratio",
        "1.0",
        f"--group-size={group_size}",
        str(tmp),
    ]
    print("running:", " ".join(argv))
    completed = subprocess.run(argv, check=False)
    if completed.returncode != 0:
        print(f"export failed with exit code {completed.returncode}", file=sys.stderr)
        return completed.returncode

    if output.exists():
        import shutil

        shutil.rmtree(output)
    tmp.replace(output)

    for name in TOKENIZER_IRS:
        path = output / name
        if not path.is_file():
            print(f"warning: {name} was not produced", file=sys.stderr)
            continue
        count = path.read_text(encoding="utf-8").count(DEFECT)
        print(f"{name}: {count} x {DEFECT}")
    print(f"artifact written to {output}")
    return 0


def main() -> int:
    """Parse arguments and build the artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument(
        "--npu",
        action="store_true",
        help="build the NPU-profile artifact the intel_npu tests need (a small real LLM)",
    )
    args = parser.parse_args()
    if args.npu:
        return build(
            args.output or NPU_OUTPUT, source=args.source or NPU_SOURCE_MODEL, group_size=128
        )
    return build(args.output or DEFAULT_OUTPUT, source=args.source or SOURCE_MODEL)


if __name__ == "__main__":
    # The guard matters: OpenVINO and NNCF use multiprocessing/forkserver, and an
    # unguarded entry point is re-imported by the forkserver child.
    raise SystemExit(main())
