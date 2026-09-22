"""A standalone stand-in for ``optimum-cli export openvino``.

The integration tests drive this script through the *real* subprocess runner so
the whole export path — argv list, ``shell=False``, ``TMPDIR`` propagation, log
capture, redaction, exit-code handling and cancellation — is exercised without
converting a multi-gigabyte model.

It deliberately has no yonga imports and no third-party dependencies: it must be
runnable by a bare ``python fake_optimum_cli.py ...``.

Control flags, injected through ``ConversionProfile.export_extra_args`` so they
land just before the positional output directory:

``--fail-with N``      exit with status ``N`` instead of ``0``
``--sleep S``          sleep ``S`` seconds before finishing (for cancellation)
``--emit-secret``      print a Hugging Face style token, to prove redaction
``--no-write-files``   skip writing the fixture artifact files
``--stderr-line TEXT`` print ``TEXT`` to stderr verbatim (for classification)

Everything it receives is recorded to ``<output>/argv.json`` and the environment
it saw to ``<output>/env.json``, which is what lets a test assert that no shell
ever touched the command line.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

#: Control flags that consume the following token as their value.
VALUE_FLAGS: frozenset[str] = frozenset({"--fail-with", "--sleep", "--stderr-line"})

#: Control flags that stand alone.
BARE_FLAGS: frozenset[str] = frozenset({"--emit-secret", "--no-write-files"})

#: A syntactically valid, entirely fictitious Hugging Face token shape.
FAKE_TOKEN: str = "hf_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7"

MODEL_XML: str = (
    '<?xml version="1.0"?>\n<net name="Model0" version="11">\n  <layers/>\n  <edges/>\n</net>\n'
)


def parse_controls(argv: list[str]) -> dict[str, object]:
    """Extract the fake's own control flags from a full ``optimum-cli`` style argv.

    Unknown options are ignored rather than rejected: the point of this script is
    to accept whatever the adapter generates, not to re-validate Optimum's CLI.
    """
    controls: dict[str, object] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in VALUE_FLAGS and index + 1 < len(argv):
            controls[token.lstrip("-")] = argv[index + 1]
            index += 2
            continue
        if token in BARE_FLAGS:
            controls[token.lstrip("-")] = True
        index += 1
    return controls


def find_output_dir(argv: list[str]) -> Path:
    """Return the positional output directory, which Optimum always puts last."""
    if not argv:
        message = "no output directory given"
        raise SystemExit(message)
    return Path(argv[-1])


def write_fixture_files(output: Path) -> None:
    """Write the minimal set of files a real export would leave behind."""
    (output / "openvino_model.xml").write_text(MODEL_XML, encoding="utf-8")
    (output / "openvino_model.bin").write_bytes(b"")
    (output / "config.json").write_text(
        json.dumps({"model_type": "fake", "architectures": ["FakeForCausalLM"]}, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """Emulate one export run and return the process exit status."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    controls = parse_controls(arguments)
    output = find_output_dir(arguments)
    output.mkdir(parents=True, exist_ok=True)

    (output / "argv.json").write_text(json.dumps(arguments, indent=2), encoding="utf-8")
    (output / "env.json").write_text(
        json.dumps(
            {key: os.environ.get(key, "") for key in ("TMPDIR", "TMP", "TEMP", "HF_HUB_OFFLINE")},
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "cwd.txt").write_text(str(Path.cwd()), encoding="utf-8")

    print("Framework not specified. Using pt to export the model.", flush=True)
    print(f"Loading model from {output}", flush=True)
    print("Applying Weight Compression", file=sys.stderr, flush=True)

    if controls.get("emit-secret"):
        print(f"Authorization: Bearer {FAKE_TOKEN}", flush=True)
        print(f"using token {FAKE_TOKEN}", file=sys.stderr, flush=True)

    stderr_line = controls.get("stderr-line")
    if isinstance(stderr_line, str):
        print(stderr_line, file=sys.stderr, flush=True)

    sleep_for = controls.get("sleep")
    if isinstance(sleep_for, str):
        deadline = time.monotonic() + float(sleep_for)
        while time.monotonic() < deadline:
            time.sleep(0.05)

    status = int(str(controls.get("fail-with", 0)))
    if status == 0 and not controls.get("no-write-files"):
        write_fixture_files(output)
        print("Export finished", flush=True)
    else:
        print(f"Export failed with status {status}", file=sys.stderr, flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
