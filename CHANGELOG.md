# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is 0, the CLI surface, the registry schema and the artifact
manifest may change in any minor release.

## [Unreleased]

### Added

- **`serve` honours OpenAI `stop`.** A string or up to 4 non-empty strings of at most 256
  characters; anything else is a 400 `invalid_request` naming `stop`. yonga matches them on the
  visible answer after the reasoning sanitiser, never inside a hidden reasoning block, across chunk
  boundaries, and ends the answer before the first stop sequence to complete with `finish_reason`
  `stop`. A stream holds back only the tail that could still become a match and the whitespace in
  front of it. A request with stop sequences is
  streamed internally even when the client did not ask for a stream, so generation ends at the
  match. `docs/CLI_SPEC.md` section 14.
- **`serve` takes images on a VLM artifact.** OpenAI `image_url` content parts holding a
  `data:image/(png|jpeg|webp|gif);base64,` URL. Every other URL (`http`, `https`, `file`, ...) is
  refused with a 400 naming the content part, permanently: yonga never fetches an image. At most 4
  images per request, each at most 4096 x 4096 pixels, checked from the image header before
  decoding and refused rather than scaled down. Images go with the last user message only, because
  OpenVINO GenAI 2026.4 binds images to the newest turn of a stateless conversation; an image in an
  earlier message is a 400 `serve.request.images_earlier_turn` saying so. An LLM artifact refuses
  images with `serve.request.images_unsupported`. Every other check, `stop` and the generation
  settings included, comes before any image is read. Only the headers are read, off the event
  loop, before a request waits for the pipeline; the pixels are decoded once it holds the
  pipeline, so at most one request holds decoded images however many are queued. Any failure
  inside Pillow is a refusal naming the image, never a server error. EXIF rotation is applied as a browser
  shows it (an unreadable EXIF block leaves the image as stored), and 16-bit greyscale is scaled to
  8 bits rather than clipped. `docs/CLI_SPEC.md` section 14.
- **`serve --carry-images-forward`** moves images of earlier messages onto the last user message,
  with a line telling the model they came earlier, for clients that resend every image of a
  conversation. The model then no longer knows which words each image went with.
- **`serve --max-body-mb N`** sets the request body limit, 1 to 64 MiB (8 by default), since
  embedded images count towards it.
- **`serve` takes OpenAI `tools` on models the registry has seen calling them.** A chat profile's
  new `tool_calling` entry (registry schema 3, below) says how a family writes calls; only a model
  whose profile declares one, and whose own chat template contains the entry's
  `template_contains` literal, accepts `tools`. Built in: Qwen2.5, Qwen3 and SmolLM3
  (`<tool_call>{json}</tool_call>`) and Granite (a JSON list), each seen producing a parseable call
  on the reference host on 2026-10-09 (CPU, one prompt each). Every other model refuses `tools`
  with a 400 `serve.request.tools_unsupported` naming `tools` and saying why. The tools, an
  assistant's `tool_calls` and `tool` results reach the artifact's own chat template through
  GenAI's `ChatHistory.set_tools`; a conversation may end on a tool result. The answer is read for
  calls outside reasoning, also with `--show-reasoning`, and returned as `message.tool_calls` with
  `finish_reason` `tool_calls`; streamed, text is released until a call could begin and each call
  then arrives as one complete delta, and a client that disconnects meanwhile still stops
  generation. An answer that cannot be read as calls to the offered functions is returned
  as text, never as a guessed call, and logged as such without its content. `tool_choice` `auto`
  and `none` are honoured; `required` and a named function are refused (they need constrained
  decoding). `parallel_tool_calls: false` keeps the first call. `stop` sequences match only the
  text outside the calls. `docs/CLI_SPEC.md` section 14.
- **Registry schema version 3: `tool_calling` on a chat profile.** It selects one of the tool-call
  parsers compiled into yonga (`tagged_json`, `json_list`), fills in its markers and field names,
  and must state `evidence` `tested` or `official` with at least one source. Older files load
  unchanged. A rule using it needs `schema_version: 3`, inherited from its document as size bands
  are; `tools/registry_bundle.py` gives a schema-3 bundle `min_tool_version` 0.4.0, so older
  releases refuse it at `registry update`. `registry validate` and remote bundles refuse a format
  this yonga has no parser for. `yonga registry explain` shows the chosen profile's tool-call
  format. `docs/REGISTRY_SPEC.md` section 8b.
- **`yonga probe --quick` and `--timeout SECONDS`.** Settling a deferred NPU compile means building
  the GenAI pipeline, which for a 9B model ran past 15 minutes. `--quick` never builds it;
  `--timeout` builds it in a child process (`python -P -m yonga.adapters.smoke_child`) and stops it
  at the limit, since a native compile cannot be interrupted in-process. Either way an unsettled
  device gets the new probe status `inconclusive`: never counted as working, never as a failure, and
  shown as such in the tables, the role matrix and `--json` (`"inconclusive": true`); a model card
  never counts it as working. A probe where nothing passed or failed outright exits 2. A child that
  dies without a result is a failure classified by the registry from its stderr. Without either
  option nothing changes, and `prepare` still always waits. `python -m yonga.application.probe`
  takes both options too. `docs/CLI_SPEC.md` section 11.

### Changed

- **Qwen2, SmolLM3 and Granite have their own chat profiles** (`chat-qwen2`, `chat-smollm3`,
  `chat-granite`) instead of `chat-default`. They copy its policy exactly (thinking detection,
  `<think>` stripping, 4096 new tokens, no sampling defaults) and add only `tool_calling`, so `chat`
  behaves as before; `registry explain` names the new rule. `chat-qwen2` matches the whole qwen2
  model type, but only Qwen2.5 was measured, and only an artifact whose template carries
  `<tool_call>` (the Qwen2.5 Instruct models, not the original Qwen2 Instruct) accepts tools.
- **Qwen3 MoE has its own chat profile, `chat-qwen3_moe`**, split from `chat-qwen3` with the same
  policy and no `tool_calling`: the MoE models share the template but were not measured writing
  calls, so they refuse `tools`. `chat` behaves as before.
- **`serve` accepts `tool` messages and assistant `tool_calls` in a conversation**, alongside
  `tools`. Assistant `tool_calls` used to be dropped without an error, and `tool` messages were
  refused as an unknown role; now both are accepted alongside `tools`, and without `tools` they are
  a 400 naming the message. The legacy `functions` and `function_call` are refused by their own name
  instead of as `tools`.
- **The registry chat profile's `stop_strings` is applied.** The field was parsed and never used; it
  now reaches OpenVINO GenAI's `GenerationConfig.stop_strings` as model-level end markers, left out
  of the answer. Empty and duplicate entries are dropped when a rule loads. A benchmark's settings
  fingerprint includes stop strings only when there are some, so every recorded fingerprint still
  matches. No built-in rule declares any, so nothing shipped behaves differently.
  `docs/REGISTRY_SPEC.md` section 8b.
- **The `serve` extra installs Pillow**, which decodes images. It is imported only when an image
  arrives: without it text is served as before, `serve` warns at startup for a VLM artifact, and an
  image request is a 500 `serve.images.dependency_missing` naming `pip install 'yonga[serve]'`.
  The AppImage, which bundles the extra, grows by Pillow's size.
- **`<ov_genai_image_N>` and `<ov_genai_video_N>` are removed from text sent to a VLM artifact**,
  in `chat` as in `serve`, repeatedly until none is left (removing one must not join a new one).
  GenAI reads these universal tags anywhere as "image N goes here", so typed text could claim an
  image that was not attached and fail generation. A model's own vision tokens typed as text are
  not removed; which they are is per-family knowledge the registry does not hold yet.

### Fixed

- `yonga probe` no longer prints a deferred raw NPU compile as an error with its diagnostic while
  it runs; the line now reads `deferred`, and the pipeline's result is the verdict.
- `python -m yonga.application.probe` settles a deferred device by building the GenAI pipeline, as
  `yonga probe` does. It had no smoke tester, so it reported every NPU LLM as failed.
- The export failure "NormalizedConfig.__init__() got multiple values for argument 'allow_new'"
  is now attributed to its real cause: on Python 3.14, optimum 2.3.0 cannot build the export
  config of mistral, phi3 and several other architectures (fixed upstream in optimum pull request
  2496, not yet released). The diagnostic names Python 3.11-3.13 as the way around it, and only
  fires on Python 3.14.

## [0.3.0] - 2026-10-09

### Added

- **Registry size bands (schema version 2).** A conversion profile can apply to a range of model
  sizes: `applies_to: parameter_count: {below: 5B}`, in the Hugging Face Hub's parameter count. An
  unknown size (no safetensors metadata, packed GPTQ/AWQ weights, a `quantization_config`) is
  outside every band, and a GGUF import or an OpenVINO copy never chooses a profile by size.
  Validation requires an unbanded fallback per device and reports a band that can never win.
  `docs/REGISTRY_SPEC.md` section 5.1.
- **"Chosen because".** The plan, `registry explain` and the artifact manifest
  (`export.profile_selection`) say why the profile was chosen: the size band, the device default,
  a rule naming it, or `--profile`. `inspect` shows what the parameter count includes, or why it
  is not used.
- `tools/measure_profiles.py`, which prepares and evaluates profiles across many models,
  resumably.
- `tools/registry_bundle.py` fills `min_tool_version` from the registry schema a bundle uses
  (`--min-tool-version` to raise it), so an older yonga refuses a bundle it cannot parse.
- `SECURITY.md`, for reporting a vulnerability privately through a GitHub security advisory, and
  `CONTRIBUTING.md`.
- Every AppImage release carries `yonga-X.Y.Z-x86_64.AppImage.packages.txt`, the exact Python
  packages the image bundles and the sha256 of the yonga wheel; the image holds the same list as
  `usr/share/yonga/packages.txt`.

### Changed

- **Small models get a more accurate default quantization.** Below 5B parameters, `prepare`
  now chooses `cpu-int4-asym-g128-r08` on the CPU, the new `gpu-int4-asym-g128-r08` on the GPU
  (both INT4 with a fifth of the layers at INT8) and `npu-int4-sym-g128` on the NPU. Measured
  against INT8 on twelve models from nine families, 350M to 8B: ratio 0.8 lowers the divergence on
  every one by 16-42 % for 9-18 % more disk and up to about a fifth of the GPU's tokens per second
  (Qwen3-1.7B's perplexity increase goes from +23.7 % to +8.5 %), and group-wise INT4 on the NPU
  roughly halves channel-wise's divergence or better (Qwen2.5-0.5B: +107 % perplexity to +38 %)
  for a tenth to a third of the NPU's tokens per second (`--profile npu-int4-sym-cw` is the fast
  choice). Models of 5B and above and models whose size is unknown get the unbanded defaults.
  `docs/PROFILE_COMPARISON.md` section 10 has the tables, and `tools/measure_profiles.py`
  reproduces them.
- **Exceptions, kept on purpose.** The VLM families (Qwen3.5, Qwen2.5-VL) keep their CPU and GPU
  pins to ratio 1.0: no VLM was measured. The generic rule (Mistral and other unrecognised members
  of common families) and Phi-3 keep the NPU on channel-wise INT4: group-wise was never probed on
  the NPU for them.
- **Qwen3 on the NPU at 5B and above now uses channel-wise INT4.** It was group-wise for every
  size, from a test of the 0.6B model only. At 8B channel-wise is a third faster (13.9 against
  10.6 tok/s) and much less accurate (+42.7 % against +9.4 % perplexity); the default follows the
  NPU documentation, and `--profile npu-int4-sym-g128` keeps the accurate one. A Qwen3 OpenVINO
  copy for the NPU is now stored under `npu-int4-sym-cw` too, so it is copied again once.
- A changed default is a new artifact, never a replacement. Preparing a model already prepared
  under the old default exports again under the new profile id, keeps the old artifact, and the
  plan says so beforehand with the `--profile` that would reuse it. `chat`, `bench` and `probe`
  given a model id use the newest ready artifact.
- A profile named by an architecture or model rule is now a preference, not an override: it must
  be usable, satisfy its constraints and match its own `applies_to`, or resolution falls through to
  scoring. Before, a deprecated profile named by a rule was used anyway.
- The text-model architecture rules no longer name the CPU and GPU profiles (they named the
  defaults), and the measured families no longer pin the NPU, so the size bands decide.
- **Upgrading with registry overrides.** The banded profiles sit at priority 110. An override
  profile on the same device at exactly 110 now ties with them for models below 5B, and `prepare`
  stops with `registry.ambiguous_rules` (`yonga registry validate` names the pair): raise it above
  110. One between 101 and 110 still beats the unbanded default but now loses to the band below
  5B. One above 110 wins for every family that does not name a profile.
- The `all` extra no longer installs the development tools (pytest, ruff, mypy, build, twine);
  they stay in `dev`.
- Releases publish to PyPI only after the AppImage built from the same wheel passes its smoke
  test and the `pypi` environment is approved. A manual publish runs on the tag itself
  (`gh workflow run release.yml --ref vX.Y.Z`; `docs/RELEASING.md`).
- **`prepare` exits 10 when the host keeps it from using a device.** A device failure whose
  diagnostic marks the environment as broken, such as `npu.device.permission_denied` or
  `device.driver_missing`, now exits 10 (environment unusable) instead of 50: the device would work
  once the host is fixed. Other device failures in `prepare` still exit 50, and `probe`, `chat` and
  `bench` keep the category's code.

### Fixed

- A registry file declaring a schema version newer than this build reports "too new" instead of a
  schema violation.
- A signed registry bundle whose `created_at` or `expires_at` carries no UTC offset is refused as
  `registry.remote.invalid`. Before, a naive `expires_at` crashed verification.
- `serve` gave every HTTP error the error code `not_found`, a wrong method included. A 404 now
  carries `not_found`, a 405 `method_not_allowed`, a 413 `request_too_large` and any other
  `invalid_request`.

### Security

- **The AppImage imported Python modules from the current directory.** Run inside a directory
  holding, say, a `typer.py`, it imported that file instead of its own. The image now runs
  `python -P -m yonga`, and every Python child process yonga starts (the `optimum-cli` export run as
  a module, `venv` and `pip` for managed export environments, the exporter-range lookup, the GGUF import)
  runs with `-P` too, so none of them puts the current directory on `sys.path`.
- **Registry data can no longer pass optimum-cli the options yonga controls.** A profile's
  `export_extra_args` and an architecture's or model's `export.extra_args` may not name
  `--model`/`-m`, `--task`, `--trust-remote-code`, `--model-kwargs`, `--cache_dir`, `--token`,
  `--quantization-statistics-path` or a bare `--`, nor the quantization options yonga derives from
  the profile's `quantization` section (`--weight-format`, `--sym`, `--ratio`, `--group-size`,
  `--backup-precision`, `--dataset`, `--all-layers`, `--awq`, `--scale-estimation`). Abbreviations
  argparse accepts, the `--name=value` form and `-m` with its value attached count as well. A
  registry file that does fails to load with `registry.schema_violation`; a profile built in code
  is refused at export time with `export.reserved_extra_args`. Before, a profile could enable
  remote code without `--trust-remote-code`.
- `yonga logs JOB_ID --bundle` rewrites the Hugging Face account name to `<hf-user>`, as it already
  rewrote the home directory to `~`, the hostname to `<host>` and the login name to `<user>`; the
  host report included the account name.
- `serve` read a request body of any size. A body over 8 MiB is now refused with 413 and the
  error code `request_too_large`, counted on the bytes received, so a chunked body is held to it
  too; with an API key configured, authentication runs first. A request with more than 4096
  messages is refused with 400 and `param` `messages`.
- The `serve` extra requires FastAPI 0.132 or newer, the first that does not read a body sent
  without a Content-Type as JSON. With an older one and no `YONGA_API_KEY`, any web page could send
  a chat request to `serve` on 127.0.0.1, since a browser sends such a body without a CORS
  preflight.

## [0.2.0] - 2026-09-28

### Added

- **Hardware tests.** `pytest -m intel_gpu` and `pytest -m intel_npu` run on a real Intel GPU and
  NPU; they were empty markers before. They cover generation, compile-cache reuse, probing,
  streaming, `serve` over the real runtime, GPU logits against CPU, a real GGUF import (GPU
  generation, and the NPU refusal the registry names), and, opt-in with
  `YONGA_TEST_VLM_ARTIFACT`, multi-turn VLM chat. `tools/make_test_artifact.py --npu` builds the
  small real LLM the NPU generation tests need.
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
- **Ten verified architecture families.** Registry rules with *tested* CPU, GPU and NPU claims for
  qwen3, qwen2, llama, gemma3-text, gemma2, olmo2, granite, smollm3, lfm2 and qwen2.5-vl (NPU
  refused), each measured on the reference host with a small model. phi3 is recorded as not yet
  exportable with optimum-intel 2.2.0.
- **optimum-intel's own Transformers ranges are honoured before the download.** Where the registry
  pins nothing, the range optimum-intel's export config declares for the model type (for example
  `<=5.0` for Gemma and Qwen2.5-VL) is read in a child process, cached, and satisfied with a
  managed export environment instead of failing after the download.
- **`export.native_implementation`** lets the registry declare that Transformers implements an
  architecture itself. A repository's own modelling code (Phi-4-mini's `modeling_phi3.py`) is then
  reported as not run, and remote code stays off.
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
  (`gpu-int4-asym-g128` and `npu-int4-sym-cw`), both converted and benchmarked on the
  reference host's GPU (`docs/INSTALL.md`), with artifact sizes, conversion duration,
  cold-compile and warm-load times and per-prompt TTFT/tok/s/TPOT. Section 8, added later
  in this release with `yonga eval`, measures output quality. The two profiles differ in
  pipeline properties as well as in quantization, so the timings are not a clean
  comparison of the two quantizations.
- **AppImage.** Every release carries `yonga-X.Y.Z-x86_64.AppImage`: yonga, a relocatable CPython
  3.14 and the `openvino` and `serve` extras in one 135 MB file, for x86_64 Linux with glibc 2.28 or
  newer and nothing else installed. `tools/build_appimage.sh` builds it with the interpreter and the
  AppImage tools pinned and checksummed (the Python packages are resolved at build time), chooses
  wheels for glibc 2.28 rather than the build host, and fails if a bundled library needs anything
  newer. CI builds and smoke-tests it on every push. Verified on the reference host from the image:
  `doctor`, `bench` on the GPU and the NPU, a GGUF import, a full conversion through managed
  environments, and `serve`.
- **Conversion without an export toolchain.** When no optimum-intel sits beside yonga -- a plain
  `pip install yonga`, or the AppImage -- `prepare` builds a managed export environment with the
  toolchain yonga is tested with instead of failing at the export. Once it exists, the Transformers
  range optimum-intel declares for the model type is read inside it, and an environment within the
  range replaces one outside it, still before the download (Gemma-3: `<=5.0`). Inside an AppImage,
  environments are made from a copy of the bundled interpreter kept under the environments
  directory, because a virtual environment made from the image's own mount would break when it
  exits.
- **PyPI publishing** through trusted publishing, from `release.yml`, running only while the
  repository is public, with a manual run to publish an earlier tag (`docs/RELEASING.md`).
- `.github/FUNDING.yml`, and a `Funding` project URL.

### Changed

- `python -m yonga` names itself `yonga` in help and usage text.
- `doctor` says that `prepare` builds a managed export environment when the export toolchain is
  missing, and inside an AppImage no longer suggests a `pip install` that cannot apply.

### Removed

- `AGENT_PROMPT.md`, the internal brief the MVP was built from; references to it now point at the
  matching `SPEC.md` sections.

### Fixed

- **A text-generation model without a registry rule was exported stateless**, so GenAI could not run
  it ("beam_idx was not found"). Plain `text-generation` is now exported as
  `text-generation-with-past`.
- **An HTTP 429 from the Hub was reported as a transient server error.** It now says the requests
  are rate-limited and that logging in raises the limit (`hub.rate_limited`).
- **`bench` did not classify a model that failed to load.** It wrote a result with empty metrics and
  exited 2, while `probe` and `chat` classified the same failure through the registry and exited
  51. It now does the same, and no empty result is saved.
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

[Unreleased]: https://github.com/ooguz/yonga/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/ooguz/yonga/releases/tag/v0.3.0
[0.2.0]: https://github.com/ooguz/yonga/releases/tag/v0.2.0
[0.1.0]: https://github.com/ooguz/yonga/tree/941cf2e8029624ba78a93a8a8e4eed94cc9c2dc2
