# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is 0, the CLI surface, the registry schema and the artifact
manifest may change in any minor release.

## [Unreleased]

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
  managed temp cannot hold the conversion.
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
  full disks, and NNCF's group-size-versus-channel-size refusal. `chat` classifies a
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
  Tokenizers 2026.4.0.0, optimum-intel 2.2.0, nncf 3.4.0, transformers 5.5.4,
  huggingface_hub 1.21.0. The full prepare/probe/chat/bench path was exercised on
  CPU with `Qwen/Qwen3-0.6B` (INT4 asymmetric, group size 128); GPU and NPU were
  exercised for enumeration, inspection and IR compilation.
- The registry knows a small number of architectures in detail; everything else
  resolves through generic rules and is reported as `inferred` rather than
  `official`.
- `yonga` never installs, upgrades or removes packages, and never runs `sudo`. When
  the environment is wrong it prints the command for you to run.

[Unreleased]: https://github.com/yonga-project/yonga/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yonga-project/yonga/releases/tag/v0.1.0
