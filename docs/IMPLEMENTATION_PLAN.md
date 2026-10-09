# IMPLEMENTATION_PLAN.md

> This is the milestone plan yonga's first build followed, in September 2026, kept as a record of
> the order the work was done in and what each step had to deliver. It is not a status page:
> serving, publication and the remote registry from "Phase 2" have shipped, and so has GGUF support
> in the form of a direct OpenVINO GGUF import; a llama.cpp backend and a GUI have not. What is done
> and what is still open is in [ROADMAP.md](ROADMAP.md); how the commands behave now is in
> [USAGE.md](USAGE.md) and [CLI_SPEC.md](CLI_SPEC.md).

## Milestone 0 — repository bootstrap

Deliver:

- `src/` package layout;
- `pyproject.toml`;
- Typer CLI entrypoint `yonga`;
- Rich console abstraction;
- Pydantic/domain models;
- XDG paths/config;
- ruff + mypy + pytest;
- CI baseline;
- version command.

Recommended runtime dependencies:

```text
typer
rich
prompt-toolkit
pydantic
pydantic-settings
packaging
huggingface-hub
psutil
PyYAML
```

OpenVINO/Optimum dependencies may be base dependencies for the first Linux-focused release or an `openvino` extra. Prefer a clear install path over premature dependency fragmentation.

Acceptance:

```bash
yonga --help
yonga --version
pytest
ruff check .
mypy src
```

## Milestone 1 — doctor + hardware inventory

Implement Linux system probe and OpenVINO probe.

Deliver:

- OS/kernel/Python info;
- RAM/swap/disk info;
- OpenVINO package family versions;
- available devices and names;
- NPU `/dev/accel` and permissions;
- compiler library detection;
- managed temp recommendation;
- Rich + JSON output.

Do not install/change drivers.

## Milestone 2 — registry core

Implement typed registry schemas, built-in initial rules, deterministic resolver, and `registry validate/explain`.

Seed:

- generic CPU/GPU/NPU profiles;
- Qwen3.5 export requirement (Transformers 5.2);
- Qwen3.5 NPU failure classification;
- Qwen3.5 GPU recommendation;
- HauhauCS derivative suggestion;
- tokenizer casing repair rule.

## Milestone 3 — Hugging Face inspect/source resolver

Implement:

- `inspect`;
- revision resolution;
- file format/size detection;
- license/gated/private metadata;
- architecture detection using small config files;
- derivative suggestions;
- download plan.

No conversion yet.

## Milestone 4 — preparation preflight + download

Implement:

- preparation plan;
- confirmations;
- disk-backed temp;
- HF download with pinned revision;
- job state/logging;
- source lineage record.

At the end, source is locally available but not converted.

## Milestone 5 — Optimum exporter

Implement subprocess adapter and profiles.

Support at least:

- `gpu-int4-asym-g128`;
- `npu-int4-sym-cw`;
- `npu-int4-sym-g128`;
- `cpu-int4-asym-g128`.

Capture command/package versions and errors.

## Milestone 6 — validation + repairs + probe

Implement IR inventory, submodel role detection, tokenizer CPU validation, numeric compile probes, diagnostic classifier, repair framework.

Implement exact `STRiNG -> STRING` tokenizer repair fixture.

Add `yonga probe`.

## Milestone 7 — artifact store + manifest

Finalize stable artifact layout and manifest. Add:

```bash
yonga list
yonga remove
```

Prepared artifacts become reusable across chat/bench.

## Milestone 8 — chat UX

Implement polished terminal chat using Rich + prompt_toolkit.

Requirements:

- current GenAI chat-history API abstraction;
- model-family generation defaults;
- hidden reasoning by default;
- Markdown rendering;
- commands;
- compile cache;
- metrics footer;
- graceful generation errors.

Start with non-streaming final-answer rendering to guarantee hidden reasoning does not leak. Add safe final-answer streaming only later.

## Milestone 9 — benchmark

Implement benchmark suites and JSON results.

Add machine-readable comparison foundation but do not overpromise cross-machine comparability.

## Milestone 10 — Qwen3.5 real-world acceptance

Run the full HauhauCS case on the target Intel machine.

Compare at least:

- existing NPU-oriented `INT4_SYM, group_size=-1` artifact on GPU;
- GPU-oriented `INT4_ASYM, group_size=128` artifact.

Measure:

- conversion duration;
- final size;
- compile/load time;
- TTFT;
- tok/s;
- memory;
- representative answer quality manually.

Select recommended GPU profile from evidence; update registry/docs.

## Milestone 11 — release hardening

- CLI error/exit-code audit;
- package installation docs;
- model/license attribution behavior;
- bug report bundle command or documented process;
- security review of remote code/download/repair paths;
- completion scripts if easy;
- `pipx` installation test;
- initial changelog.

## Phase 2

### OpenAI-compatible serving

Implement `serve` or integrate OVMS.

### Model publication

Generate model card from artifact manifest and optionally upload derived OpenVINO artifacts.

### Remote registry

Signed versioned registry updates independent of package releases.

### More backends

Potential llama.cpp/OpenVINO GGUF integration where architecture support is sufficient.

### GUI

A desktop/web UI can consume the same application services/progress events.

## Implementation rules

- Never skip a milestone dependency by hard-coding a one-off model path.
- Keep the Qwen3.5 case as a fixture/rule, not special-case application code.
- Prefer a complete vertical slice over broad unfinished commands.
- Commit after coherent milestones; do not rewrite published history.
- Keep docs synchronized with behavior.
