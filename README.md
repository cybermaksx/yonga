# yonga — Local Intel AI Model Manager (Working Title)

`yonga` is a Linux-first local AI model manager for Intel AI PCs. Its primary goal is to make Hugging Face models usable on Intel NPU/GPU/CPU without requiring the user to manually understand model formats, OpenVINO conversion flags, NPU compiler limitations, tokenizer IR quirks, or device-specific quantization trade-offs.

The product principle is:

> **Give me a model ID; inspect my machine; prepare the most appropriate reproducible OpenVINO artifact; validate it; run it on the best compatible Intel device; explain fallbacks instead of dumping opaque compiler tracebacks.**

The tool is **NPU-first, not NPU-only**. If a model cannot compile on the NPU, `yonga` should classify the failure, try a supported fallback according to policy, and preserve a complete diagnostic record.

## Initial target environment

The first supported platform is Linux x86_64, with Ubuntu 24.04/26.04 as the primary tested distributions and Intel Core Ultra systems with CPU, Arc iGPU, and NPU. The architecture must permit Windows support later without rewriting the core model and compatibility logic.

The initial tested software baseline is OpenVINO 2026.4.x, OpenVINO GenAI 2026.4.x, OpenVINO Tokenizers 2026.4.x, Optimum Intel 2.x, NNCF 2.x, Transformers 5.x, and `huggingface_hub` 1.x. Version compatibility must be represented as data rather than scattered hard-coded checks.

## Core user experience

Typical usage should look like:

```bash
yonga doctor
yonga inspect HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive
yonga prepare HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive --device auto
yonga chat HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive
yonga bench HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive
```

A preparation session should communicate decisions instead of merely streaming subprocess output:

```text
Model: HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive
Architecture: qwen3_5
Source format: GGUF only

Hardware:
  CPU  ✓
  GPU  ✓ Intel Arc Graphics
  NPU  ✓ Lunar Lake NPU

Compatibility:
  NPU  ✗ known Qwen3.5 compiler limitation on current stack
  GPU  ✓ supported
  CPU  ✓ supported

Plan:
  source      nDimensional/Qwen3.5-9B-Uncensored-Safetensors @ <revision>
  target      GPU
  format      OpenVINO IR
  weights     INT4 asymmetric
  group size  128
  ratio       1.0

Estimated temporary disk requirement: 40 GiB
Proceed? [Y/n]
```

The tool must never silently substitute an unrelated model. A derivative source may be suggested or used only when lineage is explicit and policy permits it; the exact repository and revision must be recorded in the artifact manifest.

## Repository documentation

Read these documents before implementation:

- [`SPEC.md`](SPEC.md) — normative product and technical specification.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — components, boundaries, state, and data flow.
- [`docs/CLI_SPEC.md`](docs/CLI_SPEC.md) — command-line behavior and UX contract.
- [`docs/REGISTRY_SPEC.md`](docs/REGISTRY_SPEC.md) — compatibility registry schema and resolution rules.
- [`docs/MODEL_PIPELINE.md`](docs/MODEL_PIPELINE.md) — inspect/download/convert/repair/probe/run pipeline.
- [`docs/TESTING.md`](docs/TESTING.md) — unit, integration, fixture, and hardware testing strategy.
- [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) — ordered milestones and acceptance criteria.
- [`AGENT_PROMPT.md`](AGENT_PROMPT.md) — detailed prompt for a coding agent implementing the project.

An example compatibility entry for the real-world Qwen3.5 case used to derive this design is in [`registry/examples/qwen3_5.yaml`](registry/examples/qwen3_5.yaml).

## Non-goals for the MVP

The MVP does **not** implement a new inference runtime, quantizer, tokenizer, model file format, or device driver. It orchestrates and validates OpenVINO, OpenVINO GenAI, Optimum Intel, NNCF, Hugging Face Hub, and the installed Intel runtime.

The MVP also does not aim to support every Hugging Face task. Focus first on text-generation LLMs and VLMs that OpenVINO GenAI can consume.

## Design values

1. **Reproducible:** pin source revisions and record every transformation.
2. **Explainable:** display why a target/quantization/fallback was selected.
3. **Safe by default:** no arbitrary remote code execution without explicit opt-in.
4. **Non-destructive:** never patch source caches in place; produce managed artifacts.
5. **Hardware-aware:** make device choices from capabilities and compatibility evidence.
6. **Graceful fallback:** distinguish unsupported model/device combinations from broken installations.
7. **Useful diagnostics:** collapse known failures into actionable messages while preserving raw logs.
8. **Upstream-friendly:** model repairs and compatibility rules should be narrow, traceable, and suitable for turning into upstream bug reports.

## Working title

`yonga` is a working CLI/package name. Treat branding and trademark review as a pre-public-release task; do not couple internal module names so tightly that renaming becomes expensive.
