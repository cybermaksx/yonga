# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is 0, the CLI surface, the registry schema and the artifact
manifest may change in any minor release.

## [Unreleased]

### Fixed

- **GGUF-only repositories were described as unreadable by OpenVINO GenAI.** The code assumed the
  2026.4 build had no GGUF reader; it has one, verified above.
- **A GGUF-only repository's architecture was reported as the first model type of the matching
  family rule** (`llama` for a Qwen2.5 file). `inspect` now names the one the file declares.
- **A cancelled job looked like it was still running for 24 hours.** Ctrl-C (or any exception a
  service did not record) left the job state at its last stage, so the sweep and export reuse both
  treated its staging directory as live. The job now records `job.cancelled` or `job.aborted`.
- **yonga could not prepare an NPU artifact for any LLM.** The probe compiled each numeric
  submodel raw and took a failure as the verdict, but a raw `Core.compile_model` rejects every
  dynamically shaped LLM on the NPU ("to_shape was called on a dynamic shape"), while the OpenVINO
  GenAI pipeline reshapes first and runs. Registry issue rules can now declare
  `defer_to_pipeline`; the dynamic-shape rule does, the raw attempt is recorded as `deferred`, and
  the pipeline smoke test decides. A deferred failure never passes a device on its own. Both
  `prepare` and `probe` behave this way. Verified on the reference host: `Qwen/Qwen3-0.6B` with
  `npu-int4-sym-g128` now prepares, chats and benchmarks on the NPU (67.3 tok/s).
- **GenAI rejected the NPU's static-shape properties.** `MAX_PROMPT_LEN` and `MIN_RESPONSE_LEN`
  were passed as strings, as every recorded property is; GenAI requires integers. The runtime
  adapter now types them at the boundary.
- **A known NPU compiler rejection was reported as "yonga hit a bug" (exit 99).** The NPU compiler
  writes its reason to the native file descriptors, not into the Python exception, and the adapter's
  generic `runtime.genai.load_failed` bypassed the registry entirely. yonga now captures native
  output around every native compile, classifies pipeline failures through the registry first, and
  exits 51 for an unclassified probe failure. The Qwen3.5 NPU failure now classifies as
  `npu.compiler.unsupported_graph.qwen35`, the code the specification pins. Capturing native output
  also keeps compiler log lines out of `--json` output.

### Added

- **Hardware tests.** `pytest -m intel_gpu` (3 tests) and `pytest -m intel_npu` (5 tests) run on a
  real Intel GPU and NPU; they were empty markers before. `tools/make_test_artifact.py --npu`
  builds the small real LLM the NPU generation tests need.
- NPU profiles carry `MAX_PROMPT_LEN=1024` / `MIN_RESPONSE_LEN=128`, so the manifest records the
  static-shape budget an NPU artifact was probed with.
- `ProbeStatus.DEFERRED`, for a raw compile the pipeline was asked to settle.
- **Managed export environments.** When the installed packages cannot meet an architecture's exporter
  requirements -- Qwen3.5 needs `transformers==5.2.*` -- `prepare` builds an isolated environment
  under the cache directory and runs the export in it, instead of refusing and leaving the user to
  reshape their own environment. It mirrors the packages yonga runs with and changes only the failing
  pin; torch and torchvision come from PyTorch's CPU index (1.44 GiB against a 6.2 GB CUDA install).
  It is confirmed in the plan, built and verified at preflight before the model download, never used
  half-built, reused by later jobs, and recorded in the manifest as `export.environment_key`.
  `yonga env list|remove` manage them and `doctor` reports their disk use. `--no-managed-env` restores
  the refusal; `--ignore-exporter-requirements` exports in the current environment and never builds
  one. Registry data can choose only a version for a package yonga lists, never a package of its own.
- **Direct GGUF import.** A GGUF-only repository with no usable declared derivative is imported
  with OpenVINO GenAI's own GGUF reader when a new registry `gguf` rule verifies the file's
  architecture (llama, qwen2, qwen3) and every tensor type in it (F32, F16, Q4_0, Q4_K, Q6_K, Q8_0).
  Both are read from the file header over HTTP range requests before anything is downloaded, so a
  `Q4_K_M` file carrying `Q5_0` tensors is passed over with the reason in the plan. Only the chosen
  file is fetched. The import runs in a child process from a symlink in staging, never writing to
  the Hub cache, and the result is validated, repaired and probed like any artifact. It is named by
  the file's quantization (`gguf-q4_0-gpu`). `--gguf-file` picks the file;
  `eval --allow-different-sources` compares a GGUF import with the original weights. A new issue
  rule classifies the NPU's refusal of GGUF-built graphs (`npu.pipeline.static_reshape_failed`).
- **`yonga eval CANDIDATE --reference REF`** measures what a quantization cost. It compares the
  two artifacts' next-token distributions, teacher-forced, over a shipped corpus of original text
  in eight domains (including Python and Turkish), and reports mean KL divergence, top-1 agreement
  and perplexity, in total and per passage. It runs the IR directly through `openvino.Core`, on the
  CPU at f32 by default, with one model loaded at a time. It refuses artifacts of different weights
  or vocabularies. New `cpu-int8-asym-ref` and `cpu-fp16-ref` profiles prepare references and are
  never a device's default. On Qwen3.5-9B, `npu-int4-sym-cw` diverges from INT8 2.5 times as much as
  `gpu-int4-asym-g128` (KL 0.197 vs 0.080, perplexity +22 % vs +7 %): see `PROFILE_COMPARISON.md`
  section 8.
- **`cpu-int4-asym-g128-r08` profile**, opt-in: INT4 with 20 % of layers kept at INT8. On
  Qwen3-0.6B it cuts the divergence from FP16 by 31 % (KL 0.245 vs 0.356) for 10.5 % more disk.
  It is never selected automatically.
- **Signed remote registry bundles:** `yonga registry update [--url | --file]`, `status`, `reset`.
  A bundle is verified against Ed25519 keys the user trusts (yonga ships none) and refused when
  expired, older than the installed one, or naming a path outside it. Every repair rule must name a
  built-in implementation. The installed bundle is re-verified on every load and ignored if it
  stops verifying. `cryptography` is now a dependency. `tools/registry_bundle.py` creates keys and
  signed bundles.
- **`yonga publish ARTIFACT --repo NS/NAME`** uploads a prepared artifact to the Hugging Face Hub
  in one commit, with a model card generated from the manifest. It refuses before uploading unless
  the destination is writable, both source licenses permit redistribution (read fresh from the Hub;
  unrecognised ones need `--license-reviewed`), the artifact is ready and every file still matches
  its digest. The card says the artifact is a quantized conversion and not a fine-tune, discloses
  repairs, and claims only devices whose probe passed. Private by default; `--dry-run` writes the
  card locally; local paths, login name and hostname are anonymised.
- **`yonga serve MODEL_OR_ARTIFACT [--host] [--port] [--device]`**, an OpenAI-compatible HTTP API:
  `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` with SSE streaming. It serves the same
  session `chat` opens, loads the model at startup, binds `127.0.0.1` by default, answers one
  request at a time, stops a generation whose streaming client disconnects, hides reasoning while
  streaming too, and refuses parameters it cannot honour with a 400. `YONGA_API_KEY` enables bearer
  authentication. Needs the new `serve` extra (FastAPI, uvicorn).
- **VLM generation is stateless where OpenVINO GenAI allows it.** GenAI 2026.4's `VLMPipeline`
  accepts a `ChatHistory`, so the whole conversation now travels with every turn, as on the LLM
  path. Builds without that overload still use the hidden `start_chat()` session.
- **Reuse of a completed export after a failed job.** A successful export now writes a checkpoint
  (fingerprint of every input, plus each file's size, mtime and small-file digest) into staging. When
  a later stage fails, an identical `prepare` adopts the retained export, restoring repaired files
  from their backups and rejecting anything that changed, and skips the download and conversion.
  The plan says so and the manifest records `export.reused_from_job`; `--fresh-export` declines. An
  export interrupted part-way still restarts: optimum-intel cannot resume one.
- **`yonga search QUERY [--limit N] [--device D]`** searches the Hub in one request and runs each
  result through the registry's identification, marking GGUF-header identities. Compatibility is
  shown per architecture next to a format hint, because an MLX or AWQ repository can report a
  supported architecture it cannot be converted from. Works without OpenVINO installed.
- **`yonga logs JOB_ID --bundle`** writes a bug-report archive: job log and state, artifact manifest
  and validation record, and a host report, redacted again and with the home directory, hostname and
  login name anonymised. `yonga logs` and the bundle now share one job resolver.
- `probe` says before the wait when a pipeline build will compile more than 2 GiB of weights.

- **`docs/PROFILE_COMPARISON.md`** — the same Qwen3.5-9B weights quantized twice
  (`gpu-int4-asym-g128` and `npu-int4-sym-cw`), both converted and benchmarked on this
  host's GPU, with artifact sizes, conversion duration, cold-compile and warm-load
  times and per-prompt TTFT/tok/s/TPOT. It measures size and speed only and says so:
  output quality was not evaluated, and the two profiles differ in pipeline properties
  as well as in quantization, so the timings are not a clean comparison of the two
  quantizations.

### Fixed

- **A VLM artifact can be benchmarked with more than one run, and `/reset` really
  resets it.** `VLMPipeline` has no chat-history overload in this build, so the
  runtime adapter keeps a hidden session and hands the pipeline only the turns it
  has not seen. It decided what was new by comparing histories, and two
  conversations that open with the same words are indistinguishable that way:
  `yonga bench ARTIFACT --runs 2` re-sent the identical prompt fixture and every
  run after the first failed with "Add a user message to the conversation before
  generating", as did retyping the same first message after `/reset`. The domain
  interface now carries an explicit conversation boundary, which `bench` declares
  before every warmup and every measured run and `/reset` declares in chat, and
  the adapter additionally treats a history that only repeats what it already
  delivered as a new conversation. Multi-turn chat is unchanged: a history that
  genuinely extends the delivered turns is still continued incrementally. The LLM
  path is stateless and is not affected.

### Known issues

- `bench` does not classify a failed compile. When loading the artifact fails it
  writes a result whose metrics are all empty, prints the raw exception under
  "Errors" and exits 2. `probe` and `chat` classify the same failure through the
  registry and exit 51. Use `yonga probe ARTIFACT --device D` to get the diagnosis.

## [0.1.0] - 2026-09-22

First MVP. It prepares a Hugging Face model into a reproducible OpenVINO artifact on
an Intel AI PC, validates and repairs it, runs it, and explains failures instead of
printing compiler tracebacks. Linux x86_64 only.

### Added

- **`doctor`** — host report in six sections: OS/CPU/RAM/swap, package versions with
  an OpenVINO ABI-family check, OpenVINO device enumeration, the Intel NPU node
  (`/dev/accel/accel0`, owning group, driver and compiler versions,
  compiler-in-plugin detection, compiler libraries), storage with the managed-temp
  decision, and Hugging Face authentication. Exits 0, 2 (warnings) or 10 (unusable).
- **`inspect MODEL`** — Hub metadata only, no weights downloaded: architecture,
  formats, size, parameter count, per-device compatibility with its evidence level,
  the source that would actually be converted, declared derivatives, and known
  limitations such as an architecture's Transformers pin.
- **`prepare MODEL`** — plan, confirm, download, convert, validate, repair, probe,
  publish. The plan states the resolved source and revision, the device decision, the
  quantization profile, the disk estimate and the repairs it expects, and asks before
  spending time or bandwidth. Disk preflight refuses before downloading when the
  managed temp cannot hold the conversion, and so does an unmet exporter pin: an
  architecture whose registry entry requires a Transformers release the environment
  does not have stops at preflight with exit 10. `--ignore-exporter-requirements`
  overrides that and is recorded in the manifest; `--yes` never implies it.
- **`list`, `remove`** — the artifact catalogue, and deletion that also clears the
  artifact's compile-cache entries and refuses to touch anything outside the store.
- **`probe ARTIFACT --device D`** — compiles an artifact's submodels on a device and
  reports what worked, with timings; `--all-submodels` for every submodel.
- **`chat ARTIFACT`** — a local session over OpenVINO GenAI with slash commands
  (`/help`, `/reset`, `/clear`, `/tokens`, `/temp`, `/top-p`, `/top-k`, `/reasoning`,
  `/stats`, `/model`, `/exit`), reasoning blocks hidden by default, and a metrics
  footer (TTFT, tok/s, TPOT, token counts).
- **`bench ARTIFACT`** — quick/standard/long prompt suites with warmups, writing a
  reproducible JSON result that records the artifact, host, runtime versions and
  settings.
- **`logs [JOB_ID]`** — the job logs kept on disk, as a table, as JSON, or raw JSON
  Lines.
- **`registry validate|explain|paths`** — load and check every registry layer, show
  which rules matched a model and why, and print the managed directories.
- **Compatibility registry as data** — architectures, conversion profiles, known
  issues, repairs, derivative sources and chat profiles in YAML, with version
  constraints, a specificity ladder, deterministic tie-breaking, provenance on every
  rule, and user override layers under `~/.config/yonga/registry-overrides`.
- **Failure classification** — raw error text in, typed diagnostic out: a stable
  dotted code, a category, a process exit code, remediation and fallback devices.
  Shipped signatures cover the Qwen3.5 NPU bufferization failure, dynamically shaped
  graphs the NPU compiler rejects, any other named NPU compiler pass that refuses a
  graph, the mixed-case `STRiNG` tokenizer IR, NPU node permissions, out-of-memory,
  full disks, NNCF's group-size-versus-channel-size refusal, and the exporter patching
  a Transformers release that does not define the symbol it wants — eleven rules in
  all. `chat` classifies a
  failed compile through the same rules `probe` does, so a graph the NPU compiler
  refuses is never reported as a driver problem.
- **Tokenizer IR repair** — `ov-tokenizer-precision-string-casing` rewrites only the
  `precision="STRiNG"` port attribute that OpenVINO 2026.4 writes and then cannot read
  back, leaving `element_type="string"` alone. It is detected, confirmed in the plan,
  applied during validation and recorded in the manifest.
- **Managed temp and compile cache** — conversions run in a disk-backed directory
  under `~/.cache/yonga/tmp/<job-id>` with `TMPDIR` pointed at it, so a RAM-backed
  `/tmp` never has to be enlarged.
- **Artifact manifests** — source repository and immutable commit sha, profile,
  quantization settings, exporter argv (redacted), package versions, applied repairs
  and file hashes computed after repair.
- **JSON output** (`--json`) for every non-interactive command, and structured
  JSON Lines job logs with token redaction throughout.
- **Documentation** — `docs/INSTALL.md` and `docs/TROUBLESHOOTING.md`, plus a README
  that states what is implemented and on what hardware it was verified.

### Known limitations

- `serve` (`docs/CLI_SPEC.md` section 14) and `search` (section 5) are not
  implemented.
- Not published on PyPI; install from a clone.
- Linux only. Windows is an explicit future goal, not a current capability.
- Verified on one machine: Intel Core Ultra 7 268V (Lunar Lake) with Arc iGPU and
  NPU, Ubuntu 26.04.1, Python 3.14.4, OpenVINO 2026.4.0, GenAI 2026.4.0.0,
  Tokenizers 2026.4.0.0, optimum-intel 2.2.0, nncf 3.4.0, transformers 5.2.0,
  huggingface_hub 1.21.0. The full prepare/probe/chat/bench path was exercised on
  CPU with `Qwen/Qwen3-0.6B` (INT4 asymmetric, group size 128) and on GPU with the
  Qwen3.5-9B VLM of the reference case; the NPU was exercised for enumeration,
  inspection and IR compilation, where it compiles a tokenizer submodel and has so
  far refused every language submodel offered to it.
- The registry knows a small number of architectures in detail; everything else
  resolves through generic rules and is reported as `inferred` rather than
  `official`.
- `yonga` never installs, upgrades or removes packages, and never runs `sudo`. When
  the environment is wrong it prints the command for you to run.

[Unreleased]: https://github.com/yonga-project/yonga/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yonga-project/yonga/releases/tag/v0.1.0
