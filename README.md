# yonga — Local Intel AI Model Manager (Working Title)

`yonga` is a Linux-first local AI model manager for Intel AI PCs. Its primary goal is to make Hugging Face models usable on Intel NPU/GPU/CPU without requiring the user to manually understand model formats, OpenVINO conversion flags, NPU compiler limitations, tokenizer IR quirks, or device-specific quantization trade-offs.

The product principle is:

> **Give me a model ID; inspect my machine; prepare the most appropriate reproducible OpenVINO artifact; validate it; run it on the best compatible Intel device; explain fallbacks instead of dumping opaque compiler tracebacks.**

The tool is **NPU-first, not NPU-only**. If a model cannot compile on the NPU, `yonga` should classify the failure, try a supported fallback according to policy, and preserve a complete diagnostic record.

## Status: MVP

This is a 0.1.0 MVP. It prepares, validates, runs and benchmarks OpenVINO artifacts on one machine, and it is honest about the parts that do not exist yet:

- **`serve` is not implemented.** `docs/CLI_SPEC.md` section 14 describes it as phase 2; there is no HTTP server, no OpenAI-compatible endpoint and no code for one.
- **Not published on PyPI.** `pip install yonga` does not work yet; install from a clone (see below).
- **Windows is not supported yet.** The architecture keeps the door open, but every device, driver and path probe is written and tested against Linux only.
- The registry ships knowledge for a small number of architectures. Everything else resolves through generic rules, which is why `inspect` says `inferred` rather than `official` for most models.

### What this has actually been run on

Everything in this README was executed on one machine, and nothing here is a projection:

| | |
|---|---|
| CPU / platform | Intel Core Ultra 7 268V (Lunar Lake) |
| GPU | Intel Arc iGPU (`Intel(R) Arc(TM) Graphics (iGPU)`, `GPU: vendor=0x8086 arch=v20.4.4`) |
| NPU | Intel AI Boost, `/dev/accel/accel0`, `intel_vpu` kernel module, compiler-in-plugin |
| OS | Ubuntu 26.04.1 LTS, kernel 7.2.6 |
| Python | 3.14.4 |
| OpenVINO | 2026.4.0 |
| OpenVINO GenAI | 2026.4.0.0 |
| OpenVINO Tokenizers | 2026.4.0.0 |
| optimum-intel | 2.2.0 |
| nncf | 3.4.0 |
| transformers | 5.2.0 |
| huggingface_hub | 1.21.0 |

The end-to-end path (`prepare` → `probe` → `chat` → `bench`) has been exercised on **CPU** with `Qwen/Qwen3-0.6B` quantized to INT4 asymmetric, group size 128, and on **GPU** with the Qwen3.5-9B VLM of the reference case quantized to INT4 asymmetric, group size 128. The same VLM has also been prepared, probed and benchmarked on the GPU under the channel-wise `npu-int4-sym-cw` profile, for the size-and-speed comparison in [`docs/PROFILE_COMPARISON.md`](docs/PROFILE_COMPARISON.md). On the **NPU**, `Qwen/Qwen3-0.6B` quantized to INT4 symmetric, group size 128 (`npu-int4-sym-g128`) has been prepared, chatted with and benchmarked end to end. The Qwen3.5-9B VLM does not run there: its NPU pipeline is rejected by the compiler (`OneShotBufferizeVPU2VPUIP failed : Only supports FP16.`), which yonga classifies and answers with a GPU recommendation rather than a driver complaint. Note that a *raw* compile of any LLM's language submodel fails on the NPU — that is a property of raw compiles, not of the model, and is why yonga lets the GenAI pipeline decide. No other CPU, GPU, NPU, distribution, Python version or OpenVINO release has been tested. Treat the version list as *what was verified*, not as a support matrix.

## What works today

Eleven commands, one of which (`registry`) has three subcommands:

| Command | What it does |
|---|---|
| `yonga doctor` | Host report: OS, CPU, RAM, package versions, OpenVINO ABI family, devices, NPU node/driver/compiler, storage, Hugging Face auth. Exits 0/2/10. |
| `yonga search QUERY` | Searches the Hub in one request and shows what the registry knows about each result's architecture, with a format hint and any declared derivative. Compatibility is per architecture; `inspect` answers whether a repository can actually be prepared. |
| `yonga inspect MODEL` | Reads Hub metadata (no weights), identifies the architecture, shows per-device compatibility with its evidence level, the source that would be converted, and known limitations. |
| `yonga prepare MODEL` | Plans, confirms, downloads, converts with an explicit quantization profile, validates the IR, applies known repairs, probes the runtime, and publishes an artifact with a manifest. |
| `yonga list` | The artifacts prepared on this host, with profile, size, and last benchmark. |
| `yonga remove ARTIFACT` | Deletes one artifact and its compile-cache entries, never anything outside the artifact store. |
| `yonga probe ARTIFACT --device D` | Compiles the artifact's submodels on a device and reports what worked, with timings. |
| `yonga chat ARTIFACT` | A local chat session with slash commands, hidden reasoning by default, and a metrics footer. |
| `yonga bench ARTIFACT` | Runs a prompt suite and writes a reproducible JSON result under the data directory. |
| `yonga logs [JOB_ID]` | Lists job logs, prints one as a table, or `--raw` for the JSON Lines file. |
| `yonga registry validate` | Loads every registry layer and checks it is internally consistent. |
| `yonga registry explain MODEL` | Shows which rules matched, which lost, and why. |
| `yonga registry paths` | Where registry layers, configuration and managed data live. |

Global flags: `--json`, `--verbose`, `--debug`, `--offline`, `--config`, `--data-dir`, `--cache-dir`, `--version`.

## Installation

Requires Linux x86_64 and Python 3.11-3.14. From a clone of this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[openvino,export]"
```

The extras are separate because they cost very different amounts of disk:

- **`openvino`** — `openvino`, `openvino-genai`, `openvino-tokenizers`. Needed by `probe`, `chat`, `bench` and by the device section of `doctor`.
- **`export`** — `optimum-intel[openvino]`, `nncf` (which pull in `transformers` and `torch`). Needed by `prepare` to convert and quantize a model.
- **`dev`** — pytest, ruff, mypy.

Without the extras `yonga` still runs: `doctor`, `inspect`, `list`, `logs` and `registry` work, and `doctor` exits 10 and tells you which package is missing and which extra installs it. `yonga` never installs, upgrades or removes a package on your behalf; it prints the command for you to run.

Full prerequisites, the Intel NPU driver situation on Linux, and the version rules that actually bite are in [`docs/INSTALL.md`](docs/INSTALL.md).

## Quick start

```console
$ yonga doctor
✓ Host
✓ OpenVINO
✓ Devices
✓ NPU
! Storage
· Hugging Face
...
Devices
✓  CPU  Intel(R) Core(TM) Ultra 7 268V  intel64
✓  GPU  Intel(R) Arc(TM) Graphics (iGPU)  GPU: vendor=0x8086 arch=v20.4.4
✓  NPU  Intel(R) AI Boost  4000
...
Usable, with warnings above.
```

`doctor` exits 2 here because `/tmp` is a 15 GiB tmpfs. That is a warning on purpose: `yonga` converts in its own disk-backed temporary directory and never needs `/tmp` enlarged.

Prepare a model, then use it. `prepare` prints the plan (source, revision, device, profile, quantization, disk estimate, expected repairs) and asks before doing anything expensive:

```console
$ yonga prepare Qwen/Qwen3-0.6B --device cpu
...
7/7 Artifact ready at ~/.local/share/yonga/artifacts/Qwen--Qwen3-0.6B/c1899de289a0--cpu-int4-asym-g128

$ yonga list
Artifact          Model            Revision  Profile             Dev     Size  Created           Benchmark         Ready
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
7bd72baf64957953  Qwen/Qwen3-0.6B  c1899de2  cpu-int4-asym-g128  CPU  389 MiB  2026-09-22 20:40  2026-09-22 21:11  ✓ yes

$ yonga probe 7bd72baf64957953 --device cpu
CPU  ✓ ok
✓  language   ok  1.2 s
✓  tokenizer  ok  35 ms

$ yonga chat 7bd72baf64957953 --device cpu
...
│  Hello! It seems like there's no specific task mentioned here. Could you clarify what you need
│  assistance with? Let me know how I can help!
TTFT 601 ms • 93.1 tok/s • TPOT 10.7 ms/token • 9 → 116 tokens

$ yonga bench 7bd72baf64957953 --device cpu --suite quick --runs 2 --warmups 1
Prompt              In tok   Out tok      TTFT   tok/s      TPOT    Wall
────────────────────────────────────────────────────────────────────────
short-instruction       31       256   42.8 ms    91.7   11.2 ms   2.9 s
Saved: ~/.local/share/yonga/benchmarks/20260922T174651Z-cpu-quick-8d8afeef.json
```

Those numbers are one real run of one small model on this host's CPU. They are an illustration of the output, not a performance claim.

Inside `chat`: `/help`, `/reset`, `/clear`, `/tokens N`, `/temp X`, `/top-p X`, `/top-k N`, `/reasoning on|off`, `/stats on|off`, `/model`, `/exit`.

When something fails, the diagnostic carries a stable code, remediation and a job log path. [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) maps every code the shipped registry can produce.

## Initial target environment

The first supported platform is Linux x86_64, with Ubuntu 24.04/26.04 as the primary tested distributions and Intel Core Ultra systems with CPU, Arc iGPU, and NPU. The architecture must permit Windows support later without rewriting the core model and compatibility logic.

The tested software baseline is OpenVINO 2026.4.x, OpenVINO GenAI 2026.4.x, OpenVINO Tokenizers 2026.4.x, Optimum Intel 2.2, NNCF 3.4, Transformers 5.x, and `huggingface_hub` 1.x. Version compatibility is represented as registry data rather than scattered hard-coded checks, which is why a model can declare, for example, that its export needs `transformers ==5.2.*` while the host has some other release — `yonga` stops at preflight instead of silently exporting something wrong, and only `--ignore-exporter-requirements` overrides it.

## Core user experience

Typical usage looks like this. Note that `prepare` takes a repository id, while `chat`, `bench`, `probe` and `remove` take the **artifact id** that `prepare` and `list` print:

```bash
yonga doctor
yonga inspect HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive
yonga prepare HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive --device auto
yonga list
yonga chat ARTIFACT_ID
yonga bench ARTIFACT_ID
```

A preparation session communicates decisions instead of merely streaming subprocess output. This is the header of a real run on this host, with long paths and the commit sha shortened for reading:

```text
Requested model             hf-internal-testing/tiny-random-LlamaForCausalLM@9fb191250dd5…
Resolved source             hf-internal-testing/tiny-random-LlamaForCausalLM
Revision                    9fb191250dd5
Relationship                original
Source format               safetensors
Architecture                llama  rule arch-generic-text-generation
Task                        text-generation-with-past
Device decision             CPU  supported (inferred)  CPU was requested explicitly; yonga
                            never switches device on its own.
Profile                     cpu-int4-asym-g128
                            CPU asymmetric group-wise INT4 profile
Quantization                int4, group size 128, ratio 1, asymmetric
Expected download           14 MiB
Estimated temp requirement  25 GiB  free 738 GiB in ~/.cache/yonga/tmp/<job-id>
Estimated artifact          3.9 MiB
Remote code                 not required

Expected repairs
• ov-tokenizer-precision-string-casing

Confirmation required
·  This export is expected to need a known artifact repair:
   ov-tokenizer-precision-string-casing
```

The tool never silently substitutes an unrelated model. A derivative source may be suggested or used only when lineage is explicit and policy permits it; the exact repository and revision are recorded in the artifact manifest. For a GGUF-only repository, `inspect` says so and names the declared derivative instead of guessing:

```text
Known derivative        yes  nDimensional/Qwen3.5-9B-Uncensored-Safetensors — declared
                        lossless_conversion; trust suggest; The requested repository publishes GGUF
                        weights only, which the OpenVINO exporter cannot read. — approval required
Direct GGUF conversion  no   OpenVINO GenAI cannot read GGUF for this architecture; the presence of
                        GGUF files is not evidence that it can.
```

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

User-facing documentation:

- [`docs/INSTALL.md`](docs/INSTALL.md) — Ubuntu prerequisites, extras, NPU driver and permissions, version rules.
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — every diagnostic code, exit codes, and how to read a job log.
- [`docs/PROFILE_COMPARISON.md`](docs/PROFILE_COMPARISON.md) — two INT4 quantizations of the same Qwen3.5-9B weights measured on this host: size and speed only, with the caveats spelled out.
- [`CHANGELOG.md`](CHANGELOG.md) — what changed in each release.

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
