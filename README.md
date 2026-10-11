# yonga — Local Intel AI Model Manager

`yonga` is a Linux-first local AI model manager for Intel AI PCs. Its primary goal is to make Hugging Face models usable on Intel NPU/GPU/CPU without requiring the user to manually understand model formats, OpenVINO conversion flags, NPU compiler limitations, tokenizer IR quirks, or device-specific quantization trade-offs.

The product principle is:

> **Give me a model ID; inspect my machine; prepare the most appropriate reproducible OpenVINO artifact; validate it; run it on the best compatible Intel device; explain fallbacks instead of dumping opaque compiler tracebacks.**

The tool is **NPU-first, not NPU-only**. If a model cannot compile on the NPU, `yonga` should classify the failure, try a supported fallback according to policy, and preserve a complete diagnostic record.

## Status: MVP

`yonga` prepares, validates, runs and benchmarks OpenVINO artifacts on one machine, and it is honest about the parts that do not exist yet:

- **`serve` is deliberately small.** It serves one artifact, one request at a time, and refuses what it cannot honour (`n` > 1, JSON response formats, tools on a model the registry has not seen calling them) with a 400 rather than ignoring it. Tool calls work on Qwen2.5, Qwen3 (not its MoE models), SmolLM3 and Granite, read in the format the registry records for each family. Stop sequences are honoured, matched on the visible answer so a hidden reasoning block cannot trigger one. A VLM artifact takes images, but only embedded as `data:` URLs (yonga never fetches a URL) and only with the last user message, a limit of OpenVINO GenAI's stateless chat history.
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
| Python | 3.14.4 (the test suite also passes on 3.11) |
| OpenVINO | 2026.4.0 |
| OpenVINO GenAI | 2026.4.0.0 |
| OpenVINO Tokenizers | 2026.4.0.0 |
| optimum-intel | 2.2.0 |
| nncf | 3.4.0 |
| transformers | 5.5.4 on the host; Qwen3.5 exports used 5.2.0, the version the registry pins for that architecture (the earlier ones with 5.2.0 installed on the host, the later CPU export in a managed export environment) |
| huggingface_hub | 1.21.0 |

The end-to-end path (`prepare` → `probe` → `chat` → `bench`) has been exercised on **CPU** with `Qwen/Qwen3-0.6B` quantized to INT4 asymmetric, group size 128, and on **GPU** with the Qwen3.5-9B VLM of the reference case quantized to INT4 asymmetric, group size 128. The same VLM has also been prepared, probed and benchmarked on the GPU under the channel-wise `npu-int4-sym-cw` profile, for the comparison in [`docs/PROFILE_COMPARISON.md`](https://github.com/ooguz/yonga/blob/main/docs/PROFILE_COMPARISON.md). On the **NPU**, `Qwen/Qwen3-0.6B` quantized to INT4 symmetric, group size 128 (`npu-int4-sym-g128`) has been prepared, chatted with and benchmarked end to end. The Qwen3.5-9B VLM does not run there: its NPU pipeline is rejected by the compiler (`OneShotBufferizeVPU2VPUIP failed : Only supports FP16.`), which yonga classifies and answers with a GPU recommendation rather than a driver complaint. Note that a *raw* compile of any LLM's language submodel fails on the NPU — that is a property of raw compiles, not of the model, and is why yonga lets the GenAI pipeline decide. No other CPU, GPU, NPU, distribution or OpenVINO release has been tested. Treat the version list as *what was verified*, not as a support matrix.

#### Architecture families verified on that machine

One small model per family, each prepared and probed with a generation smoke test on the GPU and
on the NPU, and benchmarked on the CPU. The registry's device claims for these families are
*tested* because of these runs.

| Family | Model | CPU | GPU | NPU |
|---|---|---|---|---|
| qwen3 | `Qwen/Qwen3-0.6B` | ✓ | ✓ | ✓ 67.3 tok/s |
| qwen2 | `Qwen/Qwen2.5-0.5B-Instruct` | ✓ 23.8 tok/s | ✓ | ✓ |
| llama | `unsloth/Llama-3.2-1B-Instruct` | ✓ 12.6 tok/s | ✓ | ✓ |
| gemma3 (text) | `unsloth/gemma-3-1b-it` | ✓ 15.3 tok/s | ✓ | ✓ |
| gemma2 | `unsloth/gemma-2-2b-it` | ✓ 7.2 tok/s | ✓ | ✓ |
| olmo2 | `allenai/OLMo-2-0425-1B-Instruct` | ✓ 13.9 tok/s | ✓ | ✓ |
| granite | `ibm-granite/granite-3.3-2b-instruct` | ✓ 7.6 tok/s | ✓ | ✓ |
| smollm3 | `HuggingFaceTB/SmolLM3-3B` | ✓ 6.2 tok/s | ✓ | ✓ |
| lfm2 | `LiquidAI/LFM2-350M` | ✓ 21.8 tok/s | ✓ | ✓ |
| qwen2.5-vl | `Qwen/Qwen2.5-VL-3B-Instruct` | ✓ 6.2 tok/s | ✓ | ✗ compiler refuses the graph |
| qwen3.5 | the reference 9B VLM | ✓ | ✓ | ✗ compiler refuses the graph |
| phi3 | `microsoft/Phi-4-mini-instruct` | ✓ prepared and evaluated | — not tried | ✓ 21.9 tok/s (group-wise) |

CPU figures are single-run INT4 throughput on the short prompt. Gemma, LFM2 and Qwen2.5-VL need an
older Transformers than the one installed; `prepare` read that range from optimum-intel and
exported them in managed environments. phi3 was exported on Python 3.14 with optimum 2.3.0 plus
the unreleased fix of optimum pull request 2496: released optimum cannot build its export config
on Python 3.14, and Python 3.11-3.13 are not affected. GGUF import is verified separately (see
[`docs/PROFILE_COMPARISON.md`](https://github.com/ooguz/yonga/blob/main/docs/PROFILE_COMPARISON.md#9-gguf-imports-against-the-original-weights) section 9).

## What works today

Sixteen commands; `registry` has seven subcommands, `env` and `cache` two each:

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
| `yonga eval ARTIFACT --reference REF` | Measures output quality: how far the artifact's next-token predictions move from a higher-precision artifact of the same weights (mean KL divergence, top-1 agreement, perplexity), teacher-forced over a shipped multi-domain corpus. |
| `yonga publish ARTIFACT --repo NS/NAME` | Uploads an artifact to the Hugging Face Hub in one commit, with a model card generated from its manifest: lineage, quantization, every repair, per-device probe evidence, file digests. Checks both source licenses first; private by default; `--dry-run` writes the card locally. |
| `yonga serve ARTIFACT` | An OpenAI-compatible HTTP API (`/v1/models`, `/v1/chat/completions` with SSE streaming, `/health`) over the same session `chat` uses, bound to `127.0.0.1` by default. |
| `yonga env list` / `env remove KEY` | The isolated environments `prepare` builds when a model's exporter needs package versions other than the installed ones, and the disk they use. |
| `yonga cache list` / `cache prune` | The OpenVINO compile cache: each entry's size, last use, owning artifact and whether it is still used, orphaned, or built by another OpenVINO release; `prune` removes the dead ones (`--older-than DAYS`, `--unattributed`, `--all`, `--dry-run`). |
| `yonga logs [JOB_ID]` | Lists job logs, prints one as a table, or `--raw` for the JSON Lines file. |
| `yonga registry validate` | Loads every registry layer and checks it is internally consistent; an installed remote bundle that no longer verifies is named, exit 2. |
| `yonga registry profiles` | The conversion profiles `--profile` accepts, with device, weights, size band and priority. |
| `yonga registry explain MODEL` | Shows which rules matched, which lost, and why. |
| `yonga registry paths` | Where registry layers, configuration and managed data live. |
| `yonga registry update` / `status` / `reset` | Installs a signed remote registry bundle after verifying it against a key you trust, reports whether the installed one still verifies, or removes it. |

Global flags: `--json`, `--verbose`, `--debug`, `--offline`, `--config`, `--data-dir`, `--cache-dir`, `--version`.

## Installation

### AppImage: one file, nothing else to install

Each [GitHub release](https://github.com/ooguz/yonga/releases) carries `yonga-X.Y.Z-x86_64.AppImage` and its `.sha256`: yonga, its own Python and the OpenVINO runtime in a single file, for x86_64 Linux with glibc 2.28 or newer. With `X.Y.Z` replaced by the release you downloaded:

```bash
sha256sum --check yonga-X.Y.Z-x86_64.AppImage.sha256
chmod +x yonga-X.Y.Z-x86_64.AppImage
./yonga-X.Y.Z-x86_64.AppImage self install
yonga doctor
```

A downloaded file is not a command yet: `self install` copies the AppImage to `~/.local/bin/yonga`, so typing `yonga` works in any shell. If `~/.local/bin` is not on your `PATH` yet, it says so and prints the line to run (on Ubuntu, `~/.profile` adds it at the next login once the directory exists); it never edits a shell startup file and never needs sudo. Running `self install` from a newer AppImage replaces the older copy after asking, and `yonga self uninstall` removes it again, leaving models and caches alone. Without FUSE, set `APPIMAGE_EXTRACT_AND_RUN=1`. The GPU and NPU still need the host's Intel drivers, which no package can carry (`doctor` says what is missing). The conversion toolchain is not in the file, because it is about 1.5 GB and depends on the model: the first `prepare` that converts a model builds a managed export environment for it under `~/.cache/yonga/envs` and reuses it afterwards.

The Python packages in the image are resolved from PyPI when it is built, so each release also carries `yonga-X.Y.Z-x86_64.AppImage.packages.txt`: the exact package versions that AppImage bundles and the sha256 of the yonga wheel inside it. The same list is in the image as `usr/share/yonga/packages.txt`.

### pip

Requires Linux x86_64 and Python 3.11-3.14. Install from PyPI into a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "yonga[openvino,export]"
```

`pip install yonga` alone installs the core without any extra. On a machine without an NVIDIA GPU, installing the CPU build of torch first saves about 4 GB of CUDA libraries that an Intel AI PC never uses; [`docs/INSTALL.md`](https://github.com/ooguz/yonga/blob/main/docs/INSTALL.md#pip) has the command.

To work on `yonga` itself, install from a clone of this repository instead: `pip install -e ".[openvino,export,serve,dev]"`.

The extras are separate because they cost very different amounts of disk:

- **`openvino`** — `openvino`, `openvino-genai`, `openvino-tokenizers`. Needed by `probe`, `chat`, `bench` and by the device section of `doctor`.
- **`export`** — `optimum-intel[openvino]`, `nncf` (which pull in `transformers` and `torch`). Used by `prepare` to convert and quantize a model. Without it, `prepare` converts in a managed export environment instead, built on first use with the toolchain yonga is tested with.
- **`serve`** — FastAPI, uvicorn and Pillow (which decodes images sent to a VLM). Needed by `serve` only.
- **`dev`** — pytest, pytest-cov, pytest-mock, pytest-timeout, ruff, mypy, build, twine. For working on `yonga` itself, not for running it.

Without the extras `yonga` still runs: `doctor`, `inspect`, `list`, `logs` and `registry` work, and `doctor` exits 10 and tells you which package is missing and which extra installs it. `yonga` never installs, upgrades or removes a package in the environment you installed it into; it prints the command for you to run. (When a model's exporter needs different package versions, `prepare` builds an isolated environment for that export instead — see [`docs/INSTALL.md`](https://github.com/ooguz/yonga/blob/main/docs/INSTALL.md#6-the-transformers-version-caveat) section 6.)

Full prerequisites, the Intel NPU driver situation on Linux, and the version rules that actually bite are in [`docs/INSTALL.md`](https://github.com/ooguz/yonga/blob/main/docs/INSTALL.md).

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
  Chosen because  751.6M parameters (Hugging Face Hub count from safetensors headers) is below 5B:
                  cpu-int4-asym-g128-r08 is the CPU profile for that size (evidence: tested).
...
7/7 Artifact ready at ~/.local/share/yonga/artifacts/Qwen--Qwen3-0.6B/c1899de289a0--cpu-int4-asym-g128-r08

$ yonga list
Artifact          Model            Revision  Profile                 Dev     Size  Benchmark   Ready
────────────────────────────────────────────────────────────────────────────────────────────────────
780ff69f5ac67fb5  Qwen/Qwen3-0.6B  c1899de2  cpu-int4-asym-g128-r08  CPU  430 MiB  2026-10-08  ✓ yes

$ yonga probe 780ff69f5ac67fb5 --device cpu
CPU  ✓ ok
✓  language   ok  1.1 s
✓  tokenizer  ok  29 ms

$ yonga chat 780ff69f5ac67fb5 --device cpu
...
│  Hello! How can I help you today? Let me know what you need! 😊
TTFT 380 ms • 101.3 tok/s • TPOT 9.9 ms/token • 13 → 19 tokens

$ yonga bench 780ff69f5ac67fb5 --device cpu --suite quick --runs 2 --warmups 1
Prompt              In tok   Out tok      TTFT   tok/s      TPOT    Wall
────────────────────────────────────────────────────────────────────────
short-instruction       31       213   41.7 ms    98.1   10.2 ms   2.2 s
Saved: ~/.local/share/yonga/benchmarks/20261008T174148Z-cpu-quick-54fb2207.json
```

Below 5B parameters each device's default profile keeps a fifth of the layers at INT8 (on the NPU,
group-wise INT4), because measured over twelve models that loses far less quality for 9-18 % more
disk. Phi-4-mini, a mistral model and Qwen2.5-VL-3B confirmed it since, and on Qwen3.5 below 5B
plain INT4 does not even prepare. Larger models and models whose size is unknown keep plain INT4
(channel-wise on the NPU), and no family is exempt from the bands. `--profile` overrides either
way, and [`docs/PROFILE_COMPARISON.md`](https://github.com/ooguz/yonga/blob/main/docs/PROFILE_COMPARISON.md#10-small-model-defaults-twelve-models-three-devices)
section 10 has the numbers.

Those numbers are one real run of one small model on this host's CPU. They are an illustration of the output, not a performance claim.

Inside `chat`: `/help`, `/reset`, `/clear`, `/tokens N`, `/temp X`, `/top-p X`, `/top-k N`, `/reasoning on|off`, `/stats on|off`, `/model`, `/exit`.

When something fails, the diagnostic carries a stable code, remediation and a job log path. [`docs/TROUBLESHOOTING.md`](https://github.com/ooguz/yonga/blob/main/docs/TROUBLESHOOTING.md) maps every code the shipped registry can produce.

## Initial target environment

The first supported platform is Linux x86_64, with Ubuntu 26.04 (verified) and 24.04 (intended, not yet tested) as the primary target distributions and Intel Core Ultra systems with CPU, Arc iGPU, and NPU. The core model and compatibility logic is kept platform-neutral so Windows support can be added later.

The tested software baseline is OpenVINO 2026.4.x, OpenVINO GenAI 2026.4.x, OpenVINO Tokenizers 2026.4.x, Optimum Intel 2.2, NNCF 3.4, Transformers 5.x, and `huggingface_hub` 1.x. Version compatibility is represented as registry data rather than scattered hard-coded checks, which is why a model can declare, for example, that its export needs `transformers ==5.2.*` while the host has some other release. When the installed packages cannot meet such a pin, `prepare` does not silently export something wrong: it shows an isolated export environment in the preparation plan and, once you confirm, builds it under the cache directory. Your own environment is never modified. `--no-managed-env` (or `managed_export_envs = false` in `config.toml`) makes preflight refuse instead (exit 10, naming the pin), and `--ignore-exporter-requirements` exports in your own environment anyway and records the override in the artifact's manifest. `--yes` approves building the environment but never implies the override. [`docs/INSTALL.md`](https://github.com/ooguz/yonga/blob/main/docs/INSTALL.md#6-the-transformers-version-caveat) section 6 has the details.

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
Architecture                llama  rule arch-llama
Task                        text-generation-with-past
Device decision             CPU  supported (tested)  CPU was requested explicitly; yonga never
                            switches device on its own.
Profile                     cpu-int4-asym-g128-r08
                            CPU INT4 asymmetric group-128 with 20% of layers at INT8; the CPU
                            default below 5B parameters
Quantization                int4, group size 128, ratio 0.8, asymmetric
Chosen because              1.03M parameters (Hugging Face Hub count from safetensors headers)
                            is below 5B: cpu-int4-asym-g128-r08 is the CPU profile for that
                            size (evidence: tested).
Expected download           6.2 MiB  only the files the conversion reads
Estimated temp requirement  25 GiB  free 230 GiB in ~/.cache/yonga/tmp/<job-id>
Estimated artifact          about 726 KiB
Remote code                 not required
Export environment          this environment  ~/.venv/bin/python  the installed packages meet the
                            exporter requirements

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
Direct GGUF conversion  no   OpenVINO GenAI cannot import qwen35 from GGUF (verified: llama, qwen2,
                        qwen3); the presence of GGUF files is not evidence that it can.
```

For llama, qwen2 and qwen3 GGUF files it can: `prepare` reads each candidate file's header, imports
the first whose every tensor type OpenVINO GenAI can read, and produces an ordinary artifact.

## Repository documentation

Design and specification documents:

- [`SPEC.md`](https://github.com/ooguz/yonga/blob/main/SPEC.md) — the design specification.
- [`docs/ARCHITECTURE.md`](https://github.com/ooguz/yonga/blob/main/docs/ARCHITECTURE.md) — components, boundaries, state, and data flow.
- [`docs/CLI_SPEC.md`](https://github.com/ooguz/yonga/blob/main/docs/CLI_SPEC.md) — command-line behavior and UX contract.
- [`docs/REGISTRY_SPEC.md`](https://github.com/ooguz/yonga/blob/main/docs/REGISTRY_SPEC.md) — compatibility registry schema and resolution rules.
- [`docs/MODEL_PIPELINE.md`](https://github.com/ooguz/yonga/blob/main/docs/MODEL_PIPELINE.md) — inspect/download/convert/repair/probe/run pipeline.
- [`docs/TESTING.md`](https://github.com/ooguz/yonga/blob/main/docs/TESTING.md) — unit, integration, fixture, and hardware testing strategy.
- [`docs/IMPLEMENTATION_PLAN.md`](https://github.com/ooguz/yonga/blob/main/docs/IMPLEMENTATION_PLAN.md) — the historical milestone plan of the first build.
- [`docs/ROADMAP.md`](https://github.com/ooguz/yonga/blob/main/docs/ROADMAP.md) — what is done, what is still open, and what comes next.

User-facing documentation:

- [`docs/USAGE.md`](https://github.com/ooguz/yonga/blob/main/docs/USAGE.md) — a hands-on tour of every command and how to check what it does.
- [`docs/INSTALL.md`](https://github.com/ooguz/yonga/blob/main/docs/INSTALL.md) — Ubuntu prerequisites, extras, NPU driver and permissions, version rules.
- [`docs/TROUBLESHOOTING.md`](https://github.com/ooguz/yonga/blob/main/docs/TROUBLESHOOTING.md) — every diagnostic code, exit codes, and how to read a job log.
- [`docs/PROFILE_COMPARISON.md`](https://github.com/ooguz/yonga/blob/main/docs/PROFILE_COMPARISON.md) — profile measurements on this host: size, speed and output quality (`yonga eval`) for two INT4 quantizations of Qwen3.5-9B, GGUF imports against the original weights, and the twelve-model, three-device measurements behind the small-model defaults, with the caveats spelled out.
- [`CHANGELOG.md`](https://github.com/ooguz/yonga/blob/main/CHANGELOG.md) — what changed in each release.

An example compatibility entry for the real-world Qwen3.5 case used to derive this design is in [`registry/examples/qwen3_5.yaml`](https://github.com/ooguz/yonga/blob/main/registry/examples/qwen3_5.yaml).

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
