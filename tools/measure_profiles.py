#!/usr/bin/env python3
"""Measure what conversion profiles cost in output quality, across many models.

For every model: prepare a reference artifact and each candidate profile with
the installed ``yonga``, then ``yonga eval`` each candidate against the
reference on the CPU. Results are appended to a JSON file as they arrive, and a
rerun skips every measurement already in it, so an interrupted campaign resumes
where it stopped. Nothing here is a test: it runs real exports and needs the
network and several gigabytes of disk per model.

    python tools/measure_profiles.py --out build/profiles.json \\
        --model Qwen/Qwen3-0.6B --model unsloth/Llama-3.2-1B-Instruct

Each profile is prepared on the device its id names (``npu-…`` on the NPU,
``gpu-…`` on the GPU, anything else on the CPU), so the artifact is probed
where it is meant to run. Evaluation always happens on the CPU at f32, where
the arithmetic does not blur the comparison.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_REFERENCE = "cpu-int8-asym-ref"
DEFAULT_CANDIDATES = ("cpu-int4-asym-g128", "cpu-int4-asym-g128-r08")


def device_for(profile: str) -> str:
    """The device a profile is prepared on: the one its id names."""
    head = profile.split("-", 1)[0]
    return head if head in {"npu", "gpu"} else "cpu"


def run_json(argv: list[str], log: Path) -> dict[str, Any]:
    """Run a yonga command with --json; stderr and the outcome go to ``log``."""
    started = time.monotonic()
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(argv)}\n")
        handle.flush()
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=handle, text=True, check=False)
        handle.write(f"exit {result.returncode} after {time.monotonic() - started:.0f}s\n")
    try:
        parsed = json.loads(result.stdout)
    except ValueError:
        parsed = {"ok": False, "raw": result.stdout[-2000:]}
    if not isinstance(parsed, dict):
        parsed = {"ok": False, "raw": parsed}
    parsed["exit_code"] = result.returncode
    return parsed


def parameter_count(repo_id: str) -> int | None:
    """The parameter count the Hub reports from the safetensors headers, if any."""
    try:
        from huggingface_hub import model_info

        info = model_info(repo_id)
    except Exception:
        return None
    total = getattr(getattr(info, "safetensors", None), "total", None)
    return int(total) if total else None


def artifact_sizes(yonga: list[str], log: Path) -> dict[str, int]:
    listing = run_json([*yonga, "list", "--json"], log)
    return {
        item["artifact_id"]: int(item.get("size_bytes") or 0)
        for item in listing.get("artifacts", [])
        if isinstance(item, dict) and "artifact_id" in item
    }


def prepare(yonga: list[str], model: str, profile: str, log: Path) -> dict[str, Any]:
    argv = [*yonga, "prepare", model, "--device", device_for(profile), "--profile", profile]
    return run_json([*argv, "--yes", "--json"], log)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", action="append", required=True, help="repeatable")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--candidate", action="append", help="repeatable; default: INT4 r1.0, r0.8")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--yonga", default="yonga", help="the yonga command to run")
    args = parser.parse_args(argv)

    yonga = args.yonga.split()
    candidates = args.candidate or list(DEFAULT_CANDIDATES)
    log = args.out.with_suffix(".log")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = (
        json.loads(args.out.read_text(encoding="utf-8")) if args.out.is_file() else {}
    )

    def save() -> None:
        args.out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    for model in args.model:
        entry = results.setdefault(model, {"candidates": {}})
        if "parameters" not in entry:
            entry["parameters"] = parameter_count(model)
        print(f"[{model}] reference {args.reference}", flush=True)
        reference = prepare(yonga, model, args.reference, log)
        if not reference.get("ok"):
            entry["reference_error"] = reference
            save()
            print(f"[{model}] reference failed; skipping the model", flush=True)
            continue
        entry["reference"] = {"profile": args.reference, "artifact_id": reference["artifact_id"]}
        for profile in candidates:
            done = entry["candidates"].get(profile, {})
            if done.get("summary"):
                print(f"[{model}] {profile}: already measured", flush=True)
                continue
            print(f"[{model}] {profile}: preparing", flush=True)
            prepared = prepare(yonga, model, profile, log)
            if not prepared.get("ok"):
                entry["candidates"][profile] = {"error": prepared}
                save()
                print(f"[{model}] {profile}: prepare failed", flush=True)
                continue
            print(f"[{model}] {profile}: evaluating", flush=True)
            evaluated = run_json(
                [
                    *yonga,
                    "eval",
                    prepared["artifact_id"],
                    "--reference",
                    reference["artifact_id"],
                    "--device",
                    "cpu",
                    "--json",
                ],
                log,
            )
            record: dict[str, Any] = {"artifact_id": prepared["artifact_id"]}
            if "summary" in evaluated:
                record["summary"] = evaluated["summary"]
                record["passages"] = [
                    {key: p.get(key) for key in ("passage_id", "genre", "kl_mean")}
                    for p in evaluated.get("passages", [])
                ]
            else:
                record["error"] = evaluated
            entry["candidates"][profile] = record
            save()
            summary = record.get("summary") or {}
            print(f"[{model}] {profile}: KL {summary.get('kl_mean')}", flush=True)
        sizes = artifact_sizes(yonga, log)
        entry["reference"]["size_bytes"] = sizes.get(entry["reference"]["artifact_id"])
        for record in entry["candidates"].values():
            if "artifact_id" in record:
                record["size_bytes"] = sizes.get(record["artifact_id"])
        save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
