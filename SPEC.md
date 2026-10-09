# SPEC.md — yonga

Status: Design specification, kept up to date  
Audience: contributors and maintainers  
Scope: Linux-first

> This is the specification yonga was originally built from. It records the requirements and the reasoning behind them, and sections are revised when the behaviour they describe changes, but words such as "MVP", "milestone" and "phase" reflect the order of the first build. The commands as they behave now are documented in [docs/USAGE.md](docs/USAGE.md) and [docs/CLI_SPEC.md](docs/CLI_SPEC.md), the registry format in [docs/REGISTRY_SPEC.md](docs/REGISTRY_SPEC.md), and what is done and what is still open in [docs/ROADMAP.md](docs/ROADMAP.md).

## 1. Product statement

`yonga` is a local model manager and execution tool for Intel AI PCs. It accepts model references (primarily Hugging Face model IDs), inspects the host hardware/software stack, determines whether the model can run on Intel NPU/GPU/CPU, builds or downloads an appropriate OpenVINO artifact, validates that artifact, records complete lineage and compatibility metadata, and provides high-quality local chat and benchmarking UX.

The product is intentionally an orchestration layer. OpenVINO Runtime/GenAI performs inference, Optimum Intel and NNCF perform conversion/quantization, Hugging Face Hub provides model discovery/download, and installed Intel drivers provide hardware access.

The tool's value is in making these parts work together reliably and explainably.

## 2. Goals

### 2.1 Primary goals

1. A user can run `yonga doctor` and understand whether CPU/GPU/NPU are usable and why.
2. A user can provide a Hugging Face model ID and receive a preparation plan before expensive work starts.
3. The tool selects a device-specific conversion/quantization profile from data-driven compatibility rules.
4. Preparation is reproducible: all sources are pinned by revision and all transformations are captured in a manifest.
5. The tool validates exported OpenVINO submodels before declaring an artifact ready.
6. Known conversion/runtime failures are classified and converted into useful diagnostics.
7. If the requested device is unsupported but another local Intel device is viable, the tool may offer or perform a policy-approved fallback and explain it.
8. The terminal chat UX is pleasant: Markdown output, hidden reasoning by default, conversation history, reset/clear/help commands, generation controls, and performance metrics.
9. The tool can benchmark prepared models using reproducible prompts and save structured results.
10. The design allows an OpenAI-compatible local serving layer (`yonga serve`, section 22) without changing model preparation semantics.

### 2.2 Secondary goals

- Make it easy to publish derived OpenVINO artifacts with accurate source lineage.
- Build a machine-readable compatibility knowledge base useful independently of the CLI.
- Generate useful bug-report bundles for upstream OpenVINO/Optimum issues.
- Make conversion cacheable and resumable where upstream tools permit it.
- Prefer official or source-author artifacts over third-party derivatives when equivalent choices exist.

## 3. Non-goals

The MVP must not:

- implement a new inference engine;
- implement a new quantization algorithm;
- implement a new tokenizer;
- implement a new device driver;
- claim support for a model/device combination solely because the device exists;
- silently run arbitrary Hugging Face remote code;
- silently substitute a semantically different model;
- silently modify the Hugging Face shared cache or an original source repository;
- promise NPU execution for every model;
- treat NPU as always superior to GPU;
- hide raw diagnostics from advanced users (raw logs must be retainable/exportable).

## 4. Terminology

**Source model**: the user-selected model identity, e.g. `HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive`.

**Resolved source**: the exact repository/revision actually used as conversion input. It may differ from the source model only when a declared lineage relationship exists and the user/policy allows it.

**Artifact**: a locally managed, prepared model directory usable by OpenVINO GenAI.

**Profile**: device-specific conversion/inference settings, e.g. `gpu-int4-asym-g128`.

**Probe**: a compile/load test against a target device without claiming full quality/performance validation.

**Repair**: a narrowly scoped deterministic transformation applied to generated artifact files to work around a known compatibility defect.

**Compatibility registry**: versioned YAML data describing architectures, devices, known issues, profiles, and repair signatures.

**Evidence level**: strength of compatibility knowledge: `official`, `tested`, `reported`, `inferred`, `unknown`.

## 5. Supported platforms and baseline

### 5.1 MVP host support

Required MVP host:

- Linux x86_64;
- Python 3.11–3.14;
- Ubuntu 24.04 and 26.04 are first-class test targets;
- Intel CPU always permitted when OpenVINO reports it;
- Intel GPU where OpenVINO reports `GPU`;
- Intel NPU where OpenVINO reports `NPU`.

Other Linux distributions may work but must be reported as unverified until tested.

### 5.2 Tested dependency family

The implementation must represent dependency compatibility centrally. Tested family:

- OpenVINO: `2026.4.x`;
- OpenVINO GenAI: matching `2026.4.x` family;
- OpenVINO Tokenizers: matching `2026.4.x` family;
- Optimum Intel: `>=2.1` (tested 2.2.0);
- NNCF: `>=2.19` (tested 3.4.0);
- Transformers: architecture-dependent; Qwen3.5 export specifically requires `5.2.x` in current OpenVINO GenAI guidance (tested 5.2.0);
- huggingface_hub: `1.x` preferred.

When yonga's own environment has no export toolchain, a managed export environment is built from a reference toolchain (`REFERENCE_TOOLCHAIN` in `src/yonga/domain/environments.py`: optimum-intel 2.2.0, NNCF 3.4.0, Transformers 5.5.4), and an architecture's declared exporter requirement overrides it.

Do not assume all architectures use the same Transformers version. The registry must support model-family-specific exporter requirements.

### 5.3 ABI rule

OpenVINO, OpenVINO GenAI, and OpenVINO Tokenizers must be treated as a compatible set. `doctor` must warn when major/minor/patch families do not align and must not auto-repair the Python environment in the MVP.

## 6. High-level workflows

### 6.1 Doctor

`yonga doctor` gathers:

- OS/distribution/kernel;
- Python and package versions;
- OpenVINO available devices;
- per-device full name and architecture if available;
- NPU driver/compiler information if available;
- `/dev/accel` accessibility on Linux;
- render-group membership when relevant;
- compiler-in-plugin availability;
- system compiler libraries if relevant;
- free disk on model/cache/temp filesystems;
- RAM and swap;
- Hugging Face authentication state without revealing secrets.

It returns a health report with statuses: `ok`, `warning`, `error`, `unknown`.

Doctor must be read-only. It may print suggested remediation commands but must not invoke `sudo` or modify system packages.

### 6.2 Inspect

`yonga inspect MODEL` performs low-cost metadata discovery before downloading large weights.

It should retrieve:

- model repo ID and pinned revision/SHA;
- license metadata when provided;
- pipeline tag;
- architecture/model type from config;
- file inventory and approximate sizes;
- presence of safetensors, GGUF, OpenVINO IR, tokenizer files;
- gated/private status;
- likely parameter count where metadata permits;
- compatibility registry matches;
- known exporter requirements;
- likely device candidates;
- whether remote code appears required.

Inspect must not automatically download multi-gigabyte weight shards.

### 6.3 Prepare

`yonga prepare MODEL [--device auto|npu|gpu|cpu]` executes the preparation state machine described in `docs/MODEL_PIPELINE.md`.

Before any large download/conversion, show a plan containing:

- requested model;
- resolved conversion source;
- source revision;
- format;
- target device;
- quantization profile;
- expected download bytes when known;
- estimated temporary storage requirement;
- expected final artifact size range when estimable;
- need for `trust_remote_code`;
- any known issue/fallback notes.

Interactive mode requires confirmation for:

- using a third-party derivative source;
- enabling `trust_remote_code`;
- downloading above a configurable threshold (default 5 GiB);
- applying a non-trivial repair;
- falling back from an explicitly requested device.

`--yes` may approve non-security confirmations, but must not implicitly enable `trust_remote_code`. Remote code requires an explicit `--trust-remote-code` flag.

### 6.4 Chat

`yonga chat MODEL_OR_ARTIFACT` resolves a prepared artifact and starts an improved terminal chat.

MVP features:

- Markdown rendering;
- prompt history using `prompt_toolkit`;
- hidden reasoning/thinking text by default;
- optional `--show-reasoning` mode only when model output supports it;
- configurable `max_new_tokens`, temperature, top-p, top-k;
- model-specific generation defaults from registry/model card profile;
- `/help`, `/reset`, `/clear`, `/tokens N`, `/stats on|off`, `/exit`;
- performance footer: TTFT, output tok/s, TPOT, input/output tokens when available;
- compile cache enabled by default for GPU and where appropriate;
- conversation represented with current OpenVINO GenAI chat-history APIs rather than depending on deprecated `start_chat()/finish_chat()` for new code.

Reasoning suppression must be defense-in-depth:

1. request non-thinking mode through the model's chat template where known;
2. do not stream untrusted hidden-reasoning blocks directly to screen;
3. strip recognized `<think>...</think>` wrappers before display;
4. if a reasoning block is left unclosed, do not expose it by accident.

### 6.5 Bench

`yonga bench MODEL_OR_ARTIFACT` produces a repeatable benchmark.

At minimum record:

- hardware/device identity;
- software versions;
- artifact manifest hash;
- prompt ID and exact prompt hash;
- warm-up count;
- measured run count;
- compile/load time;
- TTFT;
- prompt token count;
- generated token count;
- generation throughput;
- TPOT;
- wall-clock duration;
- peak process RSS if available;
- errors/timeouts.

Default benchmark set should contain a short prompt, a medium prompt, and one longer context prompt. Do not compare different devices using different prompt/generation settings.

Results are saved as JSON in the data directory and summarized as a Rich table.

## 7. Device selection and fallback

### 7.1 Requested device semantics

`--device npu|gpu|cpu` means the user explicitly requests that device. A failure must not silently move to another device. The CLI may offer fallback interactively or return a structured recommendation.

`--device auto` permits policy-based selection/fallback.

### 7.2 Auto-selection

Auto-selection is compatibility-first, not marketing-first. Recommended algorithm:

1. Exclude devices unavailable in OpenVINO.
2. Exclude device/model pairs marked `unsupported` for the active version constraints.
3. Prefer a registry `recommended_device` when evidence is `official` or `tested`.
4. Otherwise rank compatible accelerators according to architecture/profile policy.
5. Run a cheap compile probe when knowledge is `unknown` and a prepared IR exists.
6. If probe fails with a recognized unsupported signature, classify and try the next candidate.
7. If probe fails with an installation/environment error, stop instead of masking the broken environment by fallback unless the user explicitly asks to continue.

### 7.3 Failure classification

At minimum classify:

- `device_unavailable`;
- `driver_missing`;
- `permission_denied`;
- `compiler_missing`;
- `compiler_unsupported_graph`;
- `out_of_memory`;
- `insufficient_disk`;
- `unsupported_ir_type`;
- `tokenizer_ir_incompatible`;
- `package_abi_mismatch`;
- `source_format_unsupported`;
- `remote_code_required`;
- `gated_model_auth_required`;
- `download_failure`;
- `unknown_runtime_error`.

Raw exception text must be retained in logs.

## 8. Model/source resolution

### 8.1 Preferred source order

When preparing from a Hugging Face model:

1. Existing compatible OpenVINO artifact with trusted lineage, if policy permits.
2. Safetensors in the selected/source-author repository.
3. Other standard Transformers weights in that repository.
4. A declared derivative with lossless/traceable lineage, but only with user approval or a registry rule whose trust policy allows it.
5. GGUF direct loading/conversion only when the selected backend explicitly supports the architecture.

Do not infer that a similarly named repository is equivalent.

### 8.2 Derivative trust

Derivative entries must include:

- derivative repo ID;
- source repo ID;
- source revision/hash if known;
- relationship (`converted`, `lossless_conversion`, `quantized`, etc.);
- evidence URL/note;
- allowed use (`suggest`, `auto_if_trusted`, `never_auto`).

The default for third-party derivatives is `suggest`.

### 8.3 Hugging Face cache

Use `huggingface_hub.snapshot_download()` / `hf_hub_download()` and the standard Hub cache rather than reimplementing downloads. Pin a resolved revision. Respect `HF_HOME`, `HF_TOKEN`, offline mode, and local cache behavior.

## 9. Conversion profiles

Profiles are data, not branches spread throughout code.

Initial profiles:

Each device has a default for every model and, below 5B parameters, a more accurate default chosen
by a registry size band (`docs/REGISTRY_SPEC.md` section 5.1; evidence in
`docs/PROFILE_COMPARISON.md` section 10). A model of unknown size gets the unbanded default, and
`--profile` overrides either. An architecture rule can still name a profile for its family: the VLM
families keep ratio 1.0 on the CPU and GPU, and the generic rule and Phi-3 keep the NPU on
channel-wise INT4.

### `npu-int4-sym-cw`

NPU default at 5B parameters and above, and for models of unknown size:

```text
--weight-format int4
--sym
--ratio 1.0
--group-size -1
```

### `npu-int4-sym-g128`

NPU default below 5B parameters:

```text
--weight-format int4
--sym
--ratio 1.0
--group-size 128
```

### `gpu-int4-asym-g128`

GPU default at 5B parameters and above, for models of unknown size, and for the VLM families:

```text
--weight-format int4
--ratio 1.0
--group-size 128
```

`--sym` omitted => asymmetric integer weight compression in Optimum Intel semantics.

### `cpu-int4-asym-g128`

CPU counterpart of `gpu-int4-asym-g128`, with the same weights but without GPU pipeline properties.

### `cpu-int4-asym-g128-r08` and `gpu-int4-asym-g128-r08`

CPU and GPU defaults below 5B parameters (except the VLM families): the same INT4 format with a
fifth of the layers kept at INT8:

```text
--weight-format int4
--ratio 0.8
--group-size 128
```

Each profile must include version constraints, intended device classes, and a rationale.

## 10. Temporary storage and preflight

Large conversion jobs can fail late if `/tmp` is a RAM-backed tmpfs. The tool must not blindly use `/tmp`.

Preparation preflight must:

- inspect filesystem type and free bytes for system temp, artifact target, and cache;
- detect tmpfs;
- estimate working-space requirement from source weight size;
- create a managed disk-backed temp directory under `${XDG_CACHE_HOME:-~/.cache}/yonga/tmp` by default;
- pass `TMPDIR`, `TMP`, and `TEMP` to conversion subprocesses;
- clean job temp on success;
- preserve temp/logs on failure only when useful/configured.

Initial conservative estimate:

```text
required_temp = max(20 GiB, source_weight_bytes * 2.5)
```

This is a policy estimate, not a claim about upstream internals; make it configurable and record actual peak temp use where possible.

## 11. Export execution

MVP may wrap `optimum-cli export openvino` through a controlled subprocess abstraction rather than immediately depending on unstable internal Python APIs.

Requirements:

- build argv as a list; never invoke through a shell;
- stream stdout/stderr into structured job logs;
- redact tokens/secrets;
- set controlled environment (`TMPDIR`, cache vars as needed);
- capture exact package versions;
- capture command argv in manifest/logs (excluding secrets);
- return typed failures;
- support cancellation (SIGINT forwarded gracefully);
- never delete a pre-existing target directory unless it is a tool-owned incomplete job.

A future implementation may move to direct Python APIs behind the same `Exporter` interface.

## 12. Artifact validation

An artifact is not `ready` merely because export exits zero.

Validation steps:

1. inventory expected files;
2. parse manifest/config JSON;
3. load tokenizer/detokenizer with the tokenizer extension where present;
4. enumerate `*.xml` OpenVINO IR files;
5. read each IR;
6. identify model roles (language, text embeddings, vision embeddings, merger, positional, tokenizer, detokenizer);
7. compile numeric submodels on requested device where safe;
8. tokenizer/detokenizer compile on CPU;
9. run a minimal generation smoke test when pipeline construction succeeds;
10. save validation results to artifact manifest.

For VLMs, validation must not assume all submodels should execute on the same device.

## 13. Repair framework

Repairs are explicit, versioned, narrowly matched transformations.

A repair consists of:

- ID;
- description;
- affected version constraints;
- precondition matcher;
- exact transformation;
- postcondition validation;
- safety level (`automatic`, `confirm`, `manual_only`);
- upstream issue/reference metadata;
- tests with before/after fixtures.

Initial repair fixture:

`ov-tokenizer-precision-string-casing`

Precondition:

- generated OpenVINO tokenizer/detokenizer XML;
- port precision exactly `STRiNG`;
- compatible XML context;
- `element_type="string"` may be present and must not be altered.

Transformation:

```text
precision="STRiNG" -> precision="STRING"
```

Postcondition:

- no `precision="STRiNG"` remains;
- tokenizer/detokenizer can be read by current OpenVINO with OpenVINO Tokenizers extension;
- backup/hash recorded.

Never implement generic case-normalization of XML precision values.

## 14. Artifact manifest

Every prepared artifact has `.yonga/manifest.json`.

Required fields:

```json
{
  "schema_version": 1,
  "artifact_id": "...",
  "created_at": "ISO-8601",
  "tool_version": "...",
  "requested_model": {
    "repo_id": "...",
    "revision": "..."
  },
  "resolved_source": {
    "repo_id": "...",
    "revision": "...",
    "relationship": "original|converted|lossless_conversion|quantized"
  },
  "architecture": "qwen3_5",
  "task": "image-text-to-text",
  "export": {
    "backend": "optimum-intel",
    "profile": "gpu-int4-asym-g128",
    "arguments": {},
    "package_versions": {}
  },
  "repairs": [],
  "validation": {},
  "device_probes": {},
  "recommended_runtime": {
    "device": "GPU",
    "pipeline": "VLM",
    "properties": {}
  },
  "source_license": "apache-2.0",
  "files": [
    {"path": "...", "size": 0, "sha256": "..."}
  ]
}
```

Do not include HF access tokens or other secrets.

## 15. Local state layout

Respect XDG conventions.

```text
~/.config/yonga/
    config.toml
    registry-overrides/

~/.cache/yonga/
    tmp/
    compile/
    metadata/

~/.local/share/yonga/
    artifacts/
    benchmarks/
    logs/
    chat-history/
```

Allow overrides through environment variables and CLI flags.

Prefer referencing HF cache snapshots rather than duplicating source downloads.

## 16. Logging

Two channels:

1. Human UI via Rich.
2. Machine log (JSON Lines) per job.

Each log record includes timestamp, job ID, component, severity, event name, and structured fields.

`--verbose` reveals underlying commands and selected low-level logs.
`--debug` includes tracebacks and OpenVINO diagnostic properties.

Secrets must be redacted.

## 17. Configuration

Precedence:

1. CLI option;
2. environment variables (`YONGA_*`);
3. user config TOML;
4. defaults/registry.

Initial configuration keys:

- default device policy;
- data/cache/temp directories;
- HF cache override;
- confirmation threshold for downloads;
- compile cache enabled;
- reasoning visibility default;
- benchmark run/warmup counts;
- registry update policy;
- telemetry (must default to disabled in MVP unless explicitly designed and documented later).

## 18. Security and trust

### 18.1 Remote code

`trust_remote_code=False` by default.

If export requires it, preparation must stop with a clear message. The user may rerun with `--trust-remote-code`. `--yes` must not imply this flag.

### 18.2 Model files

Treat downloaded model files as untrusted data. Do not execute scripts from model repositories.

### 18.3 Credentials

Use Hugging Face's normal authentication mechanisms. Never print tokens. Do not persist tokens inside yonga manifests/logs.

### 18.4 Repairs

Automatic repairs must be pure local file transformations under tool-owned artifact directories with exact match criteria. Never apply repairs to the shared HF source cache.

### 18.5 System changes

The tool may diagnose missing NPU drivers/packages but does not use sudo or install system drivers in MVP.

## 19. Compatibility registry behavior

The built-in registry is versioned with the package. User overrides may add/override rules.

Resolution priorities:

1. exact model repo rule;
2. architecture + parameter-range rule;
3. architecture rule;
4. generic task/device rule;
5. unknown/probe.

More specific rules win only when their version constraints match.

Conflicting rules must be surfaced in debug output rather than silently chosen by load order.

See `docs/REGISTRY_SPEC.md`.

## 20. Qwen3.5 reference case

The first full integration fixture is based on the observed workflow:

Requested model:

```text
HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive
```

Observed source repo is GGUF-oriented. A lossless safetensors derivative was used for conversion:

```text
nDimensional/Qwen3.5-9B-Uncensored-Safetensors
```

NPU-oriented export used:

```text
INT4 symmetric
ratio 1.0
group_size -1
```

On the tested Lunar Lake/OpenVINO 2026.4 stack, NPU compilation failed with a compiler message containing:

```text
OneShotBufferizeVPU2VPUIP failed : Only supports FP16.
```

The same exported numeric submodels compiled on Intel Arc GPU. Tokenizer IR initially failed to read because generated XML contained:

```text
precision="STRiNG"
```

Changing only those exact port precision values to:

```text
precision="STRING"
```

allowed tokenizer/detokenizer loading, and the VLM pipeline produced generation on the GPU.

This fixture is used to test:

- derivative source disclosure;
- disk-backed temp selection;
- NPU failure classification;
- GPU fallback recommendation;
- tokenizer repair;
- VLM submodel validation;
- non-thinking chat UX.

Do not hard-code this model's behavior into generic code; encode it through registry data and fixtures.

## 21. Publication support

A prepared artifact can be published to the Hugging Face Hub with `yonga publish` ([docs/CLI_SPEC.md](docs/CLI_SPEC.md) section 14a). Each requirement below is either checked before anything is uploaded, with a refusal of its own when it does not hold, or carried by the generated model card and manifest.

Requirements before publication:

- user explicitly specifies destination repo;
- source license permits redistribution;
- generated model card includes original/resolved source IDs and revisions;
- quantization/export properties are included;
- repairs are disclosed;
- tested hardware/software matrix included;
- unsupported devices explicitly listed;
- no claim that weights were fine-tuned unless they actually were;
- artifact file hashes available.

## 22. OpenAI-compatible serving

`yonga serve` ([docs/CLI_SPEC.md](docs/CLI_SPEC.md) section 14) exposes at least:

- `GET /v1/models`;
- `POST /v1/chat/completions`;
- optional SSE streaming;
- health endpoint.

It is a lightweight FastAPI adapter over the same local runtime abstraction `chat` uses, installed with the optional `serve` extra. OpenVINO Model Server integration was the other option considered; it remains possible where it would materially reduce custom code.

The runtime/model selection and artifact manifest semantics must be shared with `chat`; do not create a second preparation stack.

## 23. Code quality requirements

- Python typing throughout public/internal boundaries;
- dataclasses/Pydantic models for persisted and external structures;
- no global mutable state for active jobs;
- dependency injection for hardware probes, HF client, exporter, runner, filesystem, clock where practical;
- deterministic registry resolution;
- typed domain exceptions;
- subprocess wrapper isolated behind an interface;
- no parsing of human-readable command output where a Python/API/property interface exists;
- tests required for every repair and failure classifier;
- ruff + mypy (strict, configured in `pyproject.toml`) + pytest;
- use `pathlib` rather than string path manipulation;
- all destructive filesystem operations scoped to tool-owned directories.

## 24. Success criteria for MVP

MVP is accepted when all of the following are true:

1. `doctor` produces a useful report on an Intel Core Ultra Linux machine with CPU/GPU/NPU.
2. `inspect` can analyze a public HF model without pulling weight shards.
3. `prepare` can export at least one safetensors LLM/VLM to OpenVINO INT4 using a profile.
4. Large conversion uses managed disk-backed temp rather than failing on a small tmpfs.
5. A prepared artifact contains a complete manifest.
6. Validation can compile model subcomponents and classify failures.
7. The Qwen3.5 `STRiNG` fixture is repaired safely.
8. The Qwen3.5 NPU compiler signature is classified as an unsupported-graph condition rather than a generic installation failure.
9. GPU fallback can be recommended in auto mode after that classification.
10. `chat` provides usable Markdown chat with hidden reasoning by default and configurable output length.
11. `bench` saves structured performance results.
12. Unit test suite can run on non-Intel CI by mocking device/runtime interfaces.
13. Hardware integration tests are separately marked and can run on a real Intel system.

## 25. Reference implementation facts (verified September 2026)

These are guidance inputs, not permanent assumptions; keep them isolated in registry/version data where appropriate:

- OpenVINO 2026.4 is current in the tested workflow.
- OpenVINO GenAI provides `LLMPipeline` and `VLMPipeline`; GPU compile caching is supported.
- OpenVINO 2026.4 release notes deprecate `start_chat()/finish_chat()` in favor of passing chat history directly in future-facing code.
- OpenVINO GenAI NPU guidance recommends INT4/NF4, symmetric weights, ratio 1.0, and group size `-1` or `128`; 128 is recommended for smaller models around 4–5B.
- NPU LLM default prompt length is 1024 and minimum response length 128; these are configurable and are relevant to memory planning.
- Qwen3.5 export requires Transformers 5.2 in current OpenVINO GenAI supported-model guidance.
- Official OpenVINO Qwen3.5-9B INT4 artifact uses INT4 asymmetric, group size 128, ratio 1.0 for its published conversion.
- Hugging Face `snapshot_download` caches and pins repositories by revision; use it instead of custom download logic.
