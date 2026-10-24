# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is 0, the CLI surface, the registry schema and the artifact
manifest may change in any minor release.

## [Unreleased]

### Added

- **`yonga connect opencode ARTIFACT`** adds a prepared artifact to OpenCode's config as a
  `yonga serve` provider, with the context and output limits read from the artifact, and prints the
  `serve` command to run. It refuses an artifact OpenCode cannot use (no tool calling, or a fixed
  prompt length under 8192 tokens, which every built-in NPU profile has) and says which artifact to
  use instead. It edits plain JSON only, asks first, writes atomically and never deletes the file;
  into a provider you already have it writes only `npm`, `options.baseURL` and `options.apiKey`.
  `--remove ARTIFACT` takes back only the entries it wrote and keeps any you edited since;
  `yonga remove` points to it, with `--config` when the connection is in another file.
- `yonga doctor` has a **Drivers** section: it finds the Intel GPUs and NPUs on the PCI bus,
  whatever OpenVINO makes of them, and checks for each one the kernel driver, the user-space
  driver OpenVINO's plugin loads (Intel's OpenCL compute runtime for a GPU, the Level Zero loader
  and NPU driver for an NPU) and whether OpenVINO lists it. A missing driver comes with the
  install command for the host's distribution family: Arch and derivatives such as CachyOS,
  Debian and Ubuntu, or Fedora.
- A known failure for `ZE_RESULT_ERROR_UNSUPPORTED_FEATURE` from the NPU driver without a
  compiler message (`npu.compiler.unsupported_feature`), which Arch's `intel-npu-driver` package
  causes by leaving the compiler out: it was reported as an unrecognised failure. It names both
  causes (no compiler library, or a compiler that cannot take the IR) and keeps the GPU/CPU
  fallback; `doctor` tells them apart.

### Fixed

- `doctor` reports an NPU driver that loaded no compiler (`NPU_COMPILER_VERSION` 0) as a warning
  instead of a plain version line, whatever `NPU_COMPILER_TYPE` says.
- `doctor`'s `compiler libraries` line needs both `libopenvino_intel_npu_compiler.so` and
  `libopenvino_intel_npu_compiler_loader.so`, and no longer counts
  `libopenvino_intel_npu_plugin.so`, which every `openvino` wheel ships; it gives the same advice
  as the compiler line. `/usr/lib64` is searched too, where Fedora installs the libraries.
- `doctor` no longer tells the user to join the NPU node's group, with a `sudo usermod` command,
  when the node is accessible to them anyway.

## [0.5.1] - 2026-10-11

### Changed

- **License: GNU Affero General Public License v3.0 or later (`AGPL-3.0-or-later`).** `LICENSE`,
  the package metadata and `CONTRIBUTING.md` say so. 0.5.1 is the first release on PyPI and the
  only one with downloads on GitHub; 0.5.0 was withdrawn.

### Fixed

- A remediation that repeats the command without offline mode no longer keeps `--offline` when
  it was the only option given before the command (`yonga --offline inspect …` suggested
  `yonga --offline inspect …`).

## [0.5.0] - 2026-10-11

The first public release, withdrawn the same day for 0.5.1. It fixes everything an end-to-end campaign of
fresh-user runs on real hardware found (AppImage, pip, every command, GPU, NPU and CPU), keeps
OpenVINO's usage reporting from running, and installs the AppImage on `PATH`.

### Added

- **`yonga self install` and `yonga self uninstall`.** Run from the AppImage,
  `./yonga-X.Y.Z-x86_64.AppImage self install` copies it to `~/.local/bin/yonga` (or `--bin-dir
  DIR`): atomically (temporary file in the directory, `fsync`, `0755`, rename), creating the
  directory when missing. The same file again is "already installed"; another yonga AppImage
  there is replaced after a question showing both versions where known (`--yes` answers it).
  Anything else at that name (a pip or pipx command, a symbolic link, a directory, a script) is
  refused and left alone; an AppImage is recognised by its ELF and `AI\x02` magic bytes, never by
  running it. Afterwards it checks that `yonga` on `PATH` runs the copy and, if not, says what to do
  (log out and in, or `export PATH="$HOME/.local/bin:$PATH"`) and exits 2; it never edits a shell
  startup file and never uses sudo (an unwritable directory exits 10 with the advice to pick one you
  own). Outside an AppImage it exits 70 and points pip users at their environment's `yonga` or
  `pipx install 'yonga[openvino,export]'`. `self uninstall` removes only an AppImage, after a
  question, and keeps models, logs, caches and settings, saying where they are.
  `docs/CLI_SPEC.md` section 12c.
- **`yonga registry profiles [--device DEVICE] [--json]`** lists the conversion profiles
  `--profile` accepts, with device, weights, size band, priority and status. An unknown
  `--profile` now names the profiles for the requested device and points to this command.
- The preparation plan always shows an `Export environment` row: a managed environment, or this
  environment with the interpreter path and the reason.
- Issue rule `gguf-reader-open-failed` (`export.gguf.open_failed`, exit 10). GGUF import failures
  are now classified by the registry like optimum export failures instead of always being
  `unknown_runtime_error`.
- Issue rule `torchvision-torch-build-mismatch` (`export.torchvision.build_mismatch`, exit 10) for
  `operator torchvision::nms does not exist`. `prepare` also imports the exporter in its own
  environment at preflight, so a broken torch/torchvision pair is reported before the download.
- The probe job log records every probe result with its raw runtime text (`probe.result`), as the
  diagnostics already said it did.
- `yonga eval` prints `Saved: PATH`, and `--json` includes `result_path`, as `bench` does.
- `doctor` reports whether the exporter imports (`exporter import`), reusing the check `prepare`
  recorded, so a CPU torch next to PyPI's torchvision shows as a warning with the registry's
  remedy; `--full` checks a recorded failure again.
- `chat`, `bench`, `probe`, `eval` and `publish` say which artifact a model id chose when several
  ready ones match ("Using ID (PROFILE, DEVICE), the newest of N ready artifacts for MODEL; pass the
  artifact id to choose another").
- Registry schema 4 (needs yonga 0.5.0): an issue rule's `blocks_export` lists model types whose
  export is certain to fail when the rule's constraints hold for the interpreter that will run the
  export, and `prepare` refuses such an export before the download and before any question
  (overridable with `--ignore-exporter-requirements`, recorded in the manifest).

### Changed

- **`doctor` reports how yonga is installed.** A new `Install` line in the Host section names the
  AppImage file or the Python environment, and what `yonga` on `PATH` resolves to. From an AppImage
  it is a warning with `self install` as the remedy when `yonga` is not on `PATH`, or when it is
  another installation (named with its path, for example an older AppImage). For a pip install it
  is information only. `doctor --json` carries it as `installation`. `docs/CLI_SPEC.md` section 3.
- **A command line yonga cannot parse exits 70** instead of Click's 2: an unknown command or
  option, a missing argument, a value of the wrong type (`--timeout abc`, `--older-than abc`) and
  a bare `yonga`. Exit 2 is left meaning only warnings or an inconclusive probe; `--help` still
  exits 0.
- `prepare` downloads only the files the conversion reads. ONNX copies, `original/` checkpoints,
  training runs and a second weight format are skipped (SmolLM2-360M: 693 MiB instead of
  4.7 GiB), and the plan's expected download counts the same files.
- Automatic profile selection skips a group-wise profile whose group size cannot divide the
  model's `hidden_size` (or `intermediate_size` at ratio 1.0) from `config.json`, and says why
  under "Chosen because". With no fitting profile for an explicit device, `prepare` stops before
  the download (exit 70) and names the devices that fit; `--device auto` moves on to the next
  device.
- `yonga logs` lists each job's id prefix, local start time with its zone, command, model, device,
  outcome (ok/failed/cancelled/running/unfinished), duration and log size; `--json` adds these
  fields. One job's view shows the same facts and prints record times in local time instead of
  unmarked UTC.
- `yonga list` shows the source revision as 7 characters under a `Rev` header.
- While a job runs, warnings logged by huggingface_hub (such as the Hub's "You are sending
  unauthenticated requests" advice) go to the job log as `library.log` records instead of breaking
  into the progress line.
- An explicit `--profile` whose group size cannot divide the model's `config.json` channel widths
  is refused at the plan, before any download (exit 70, `export.quantization.group_size_mismatch`).
  The message names the profiles that fit, as your command with `--profile` changed, and never
  swaps the profile itself. When no profile for that device fits (SmolLM2-360M with
  `--device gpu --profile gpu-int4-asym-g128-r08`), it offers the fitting profiles of the other
  devices, as your command with `--device` and `--profile` changed.
- The `context_length_exceeded` message no longer repeats "The prompt". It names the offered tools
  and images when they count toward the limit, and says what to do: shorten the conversation, offer
  fewer tools or send fewer images, or use a GPU or CPU artifact, whose prompt length is not fixed.
- The README quick-start `doctor` sample and its exit-2 explanation match the current output.

### Fixed

- `--offline` (and `YONGA_OFFLINE` / `HF_HUB_OFFLINE`) no longer contacts the Hugging Face Hub for
  `inspect`, `prepare` or revision resolution. A cached model is inspected and prepared from the
  local cache alone, including one downloaded by commit SHA with no `refs/main`. An uncached
  revision, or a snapshot with only metadata files, fails with `hub.offline.missing` without
  making any request.
- `yonga inspect` no longer says a model needs `--trust-remote-code` when the registry declares a
  native Transformers implementation (Phi-4-mini, phi3). `inspect` and `prepare` now share one
  remote-code decision.
- `yonga --offline search` reports `hub.offline.search_unavailable` ("Search needs the network")
  instead of a misleading cache-miss message.
- Re-running `prepare` for a model that is already prepared reuses the artifact without asking
  anything, so it works in scripts without `--yes`. The reuse plan no longer lists a download,
  temp space, repairs or an export environment.
- An explicit `--device` that the registry records as unsupported for the model (official or
  tested evidence) is refused before the download with exit 50 and the recommended fallback,
  instead of after a full download and conversion.
- Without the OpenVINO runtime installed, `prepare` stops before the download with exit 10 and
  names `pip install 'yonga[openvino]'`, instead of blaming the host for having no devices. It no
  longer offers to build a managed export environment then.
- A gated or private repository with no Hugging Face token is flagged in the plan and refused
  before the download with exit 21.
- In a session without a terminal and without `--yes`, `prepare` says the confirmation could not
  be asked and suggests `--yes`, instead of saying the user declined it.
- A `prepare` that adopts a retained export, or copies published OpenVINO IR, reports this as a
  note and exits 0 instead of 2.
- The plan's "Estimated artifact" is worked out from the profile's quantization and the model's
  parameter count (Qwen3-0.6B INT4: about 516 MiB, previously 1.4 GiB).
- The plan's device decision for `--device auto` states the actual reason, such as the registry's
  recommendation.
- NNCF's group-size refusal is classified (`export.quantization.group_size_mismatch`, exit 70) even
  when its list of refused nodes pushes the exception line out of the exporter's output tail; it
  used to end as a generic `export.failed`.
- The group-size remediation no longer offers the NPU's channel-wise profile on the CPU or GPU,
  where its `MAX_PROMPT_LEN` fails the probe; it suggests `cpu-int8-asym-ref` on the CPU instead.
  The Python 3.14 optimum diagnostic now explains what AppImage users can do.
- Remediation steps print the real repository and job id instead of `MODEL` and `JOB_ID` wherever
  they are known (`prepare`, `probe`, `chat`, `bench`, `--json`), and a step that repeats an
  earlier command is listed once.
- "Chosen because" for a profile named by a model or architecture rule gives the parameter count
  and the profile the rule overrode, such as the size-banded default (`automatic_profile_id` in the
  manifest).
- GGUF import no longer fails with "Failed to open ... with gguf_open" on a huggingface-hub 1.x
  cache (read-only blobs). OpenVINO GenAI's reader opens the file read-write, so yonga imports
  from a private writable copy in staging (a reflink where the filesystem supports one) instead of
  a link into the cache, and the disk plan reserves room for it.
- `prepare --gguf-file` naming a file the repository does not have exits 20
  (`source.gguf_file_not_found`), as `docs/CLI_SPEC.md` section 6 says, instead of 70.
- The generation smoke test no longer writes an entry to the compile cache. That entry could never
  be reused, was left orphaned after every `prepare` and grew with each re-prepare.
- `prepare` no longer redraws a finished download, or prints a stale "Downloading ... (1.4 GiB /
  1.6 GiB)" line, after the probe stage: the Hub's byte bar, closed later by the garbage
  collector, has its late reports ignored. A completed download is reported at its full size.
- Download progress no longer jumps backwards or under-counts while the Hub downloads files in
  parallel.
- huggingface_hub's own progress bars ("Fetching N files: 0%|...") no longer appear in piped
  output or compete with yonga's progress display on a terminal.
- The live conversion line shows nncf's step labels ("Applying Weight Compression") instead of
  rows of its bit-width table or the text of its progress bars.
- A conversion cancelled with Ctrl+C is announced as "Conversion cancelled", not "Conversion
  failed".
- A preparation job no longer logs `export.output_reset` ("clearing leftover staging output from
  an earlier attempt") when there were no leftovers.
- `yonga chat` loads the model before showing the startup panel. A model that cannot load on the
  chosen device ends the chat with its category's exit code (51 for an NPU compiler refusal, 10
  when OpenVINO GenAI is missing), instead of failing every message and exiting 0.
- `yonga chat` exits with the failure's code (52 when unclassified) when the user leaves right
  after a message that got no answer.
- Ctrl-C during a reply stops it at the next token and returns to the prompt with the conversation
  intact, instead of waiting for the reply to finish and then exiting 60. A second Ctrl-C quits.
- `--show-reasoning` and `/reasoning on` show the reasoning without `<think>` tags, in a dim panel
  before the answer, and only the answer is kept in the conversation. `/reasoning on` advises
  restarting with `--show-reasoning` instead of the nonexistent `--thinking` option.
- An unknown or removed artifact id no longer suggests running `yonga prepare` on that id; it
  points to `yonga list`. No step prints a literal `yonga prepare OWNER/NAME` placeholder.
- `yonga probe` without the OpenVINO runtime stops up front with `runtime.openvino.missing` and
  exit 10 (`pip install 'yonga[openvino]'`), as `bench` does, instead of an "Unrecognised failure"
  and exit 51.
- `yonga probe` shows each submodel in one line. A deferred raw compile reads "the GenAI pipeline
  decides" instead of a 25-line compiler dump with padded empty rows, and a recognised failure is
  named by its diagnostic. `--verbose` shows the raw text, and `--json` and the job log always
  carry it.
- `yonga probe --device auto` no longer prints a red refusal and a "prepare for GPU" suggestion for
  a device that failed while another passed; the failure is information about an artifact that
  runs.
- `python -m yonga.application.probe` reports a missing artifact or missing OpenVINO on stderr with
  its exit code instead of a traceback.
- `yonga eval --device npu` is refused before anything loads (`eval.device.unsupported`, exit 70).
  Native compiler output during an eval compile is captured instead of leaking onto stdout, and a
  load failure gets a one-line message.
- `yonga bench` numbers warmups and measured runs separately in progress and errors
  (`warmup 1/1`, `run 1/3`), and the summary says "1 warmup" instead of "1 warmups".
- `probe` and `bench` jobs end with `job.completed` and a terminal `completed` stage. A successful
  run used to leave a log with only `job.created` and a state that looked live for 24 hours.
- A failed job's raw error text (for example the NPU compiler's failing pass and shape) is written
  to its job log as `job.error_detail`, so `yonga logs JOB_ID --raw` and bug-report bundles contain
  what the remediation says they do.
- A cancelled job is recorded as `job.cancelled` (a warning) with a terminal `cancelled` stage
  instead of `job.failed`. The record names the last stage actually reached, plus `failed_before`
  for the step that was running.
- A job you cancel no longer keeps its managed temporary directory (often over a gigabyte) for
  diagnosis; it is removed and logged as `job.temp_discarded`.
- A prompt longer than the NPU's fixed prompt length (`MAX_PROMPT_LEN`), including one made too
  long by the offered tools, gets OpenAI's 400 `context_length_exceeded` (`param` `messages`) with
  the prompt's token count and the limit, instead of a 500 that the openai client retried;
  `yonga chat` reports the same case as `runtime.genai.prompt_too_long`.
- Error messages no longer include OpenVINO's source location ("Check '...' failed at
  /path/file.cpp:LINE"), which named the vendor's build machine.
- A generation setting refused as out of range names its field in `error.param` (`temperature`,
  `top_p`, `top_k`, `repetition_penalty`, `max_tokens` or `max_completion_tokens`, whichever was
  sent); it used to be null.
- A refused API key gets OpenAI's error type `invalid_request_error` instead of `auth`.
- A request body that is not valid JSON, or is not sent as JSON, gets `param` null instead of the
  decode error's character offset or an empty string, and the message says what was wrong.
- `yonga env remove` always asks before deleting, including when the environment is named
  `export-KEY`. With `--json` or in a non-interactive session and no `--yes`, it removes nothing,
  exits 60 and says to re-run with `--yes`.
- Inside the AppImage, the copy of the bundled Python that export environments are made from is
  shown by `yonga env list` and is deleted with the last environment that uses it or after a
  failed build.
- `yonga cache prune --older-than` with a very large number of days no longer crashes with an
  internal error; such an age selects nothing.
- `cache list`, `cache prune`, `doctor` and `env list` say "1 entry" and "1 environment" for a
  single item, as do "Nothing to prune: 1 entry (...) is kept", "1 stale working directory" and the
  `remove` note about other directories that still claim a compile cache.
- `yonga doctor` run from the AppImage no longer warns about optimum-intel, nncf and transformers,
  which the image never carries by design; one `export toolchain` line says that `prepare`
  converts in a managed export environment, so an AppImage host can reach exit 0.
- When `yonga doctor` exits 10, its last line names the cause: the OpenVINO runtime is not
  installed (with `pip install 'yonga[openvino]'`), it is installed but cannot be imported, or no
  device was found.
- `yonga self uninstall --help` describes `--bin-dir` as the directory to remove the command from,
  not to install it in. Run from a pip or pipx install, `self uninstall` says that it removes only
  the AppImage, and no longer tells you to run the cleanup commands from the downloaded AppImage
  while your pip `yonga` is still there.
- `registry update` with a malformed bundle (`registry.remote.invalid`) names each field that is
  wrong, such as a timestamp without a UTC offset, instead of only counting the problems.
- An installed remote bundle that no longer verifies is no longer ignored silently: `doctor
  --full` warns (`registry.ignored`), `registry validate` lists it under "Ignored" and exits 2,
  and `registry explain` warns that it explained without it.
- The remediation for an expired registry bundle names an update command that works as written:
  `--url`/`--file` when `registry_url` is unset, plus `registry reset`.
- `registry update` refuses `--url` together with `--file`, and `--signature` without `--file`
  (exit 70), instead of silently ignoring one of them.
- `registry paths` inside an AppImage names the AppImage file for the built-in registry instead of
  a mount path that disappears when the command exits (`builtin_appimage` in `--json`).
- `tools/registry_bundle.py` prints an expected refusal, such as an already expired bundle, as one
  line instead of a traceback, and documents every option in `--help`.
- `yonga remove` no longer deletes the newest of several artifacts when a repository id, id prefix
  or copied artifact id matches more than one; it refuses with exit 70 (even under `--yes`) and
  lists each candidate's artifact id, profile and device.
- `yonga list` no longer cuts every column short when a repository id is long (such as
  `HuggingFaceTB/SmolLM2-360M-Instruct` at 80-100 columns). It drops whole columns in order
  (creation and benchmark times, then revision, Ready, size) and lets the repository id lose its
  organisation first; the artifact id, profile and device are never shortened. If the Ready column
  has to go, unready artifacts are named under the table.
- The pip install instructions take torchvision from PyTorch's CPU index together with torch.
  With torch alone from there, pip installed PyPI's torchvision, which cannot load against a CPU
  torch, and every conversion failed. `docs/TROUBLESHOOTING.md` section 5.7 gives a repair that
  actually replaces the PyPI build.
- The README command overview lists `self` (seventeen commands). `docs/TROUBLESHOOTING.md`
  documents every `self install`/`self uninstall` diagnostic code and its exit code.
  `docs/INSTALL.md`'s quoted doctor line and its Qwen3-0.6B artifact size (430 MiB) match reality.
- Square brackets in console output are no longer read as Rich markup: `doctor`'s closing line
  again says `pip install 'yonga[openvino]'`, and messages, paths, model ids, table cells, prompts
  and progress lines print as written.
- `doctor`'s export-environment size includes the bundled-Python copy, matching `yonga env list`.
- Every "run it again with `--yes`" remediation (`prepare` without a terminal, `remove`,
  `publish`, `env remove`, `cache prune`, `self install`/`uninstall`, `registry update`), and those
  for `--trust-remote-code`, `--ignore-exporter-requirements` and dropping `--no-repair`, print your
  whole command, root options such as `--data-dir` and `--offline` included, shell-quoted, with only
  that flag changed; credential-named options are left out and secret-shaped values masked. Before,
  `prepare M --device cpu` advised `prepare M --yes`, which auto-selected the GPU.
- The `--debug` hint places `--debug` before the command (`yonga --debug list`); appended to the
  end it was a usage error.
- After a validation failure the advice names `--fresh-export`, not `--force`: a plain re-run
  adopts the export the failed job left behind.
- A system `/tmp` too small for the conversion is reported as a note, not a warning, because the
  managed directory was used; a successful `prepare` on such a host exits 0 instead of 2.
- Offline, `inspect`'s Parameters row and the profile choice's basis say the count was taken locally
  from the cached safetensors headers instead of calling it a Hub count.
- The group-size remediation's ratio-0.8 suggestion includes `--device cpu` / `--device gpu`, so the
  CPU profile is not prepared for an auto-selected GPU.
- `chat`, `bench` and `probe` answer a device failure with the same command on a device the
  artifact already passed on (for example `yonga chat ID --device gpu`) instead of telling you to
  prepare the model again.
- `chat`, `serve`, `eval` and `publish` no longer print a `yonga logs JOB_ID --raw` step with a
  literal placeholder; they open no job, and `--debug` still shows the raw error.
- `probe`, `bench`, `eval` and `publish` explain an unknown reference as `chat` does: a model id
  that is not prepared gets its `yonga prepare` command, and anything else points at `yonga list`.
  The "run it again" step prints your whole command; rebuilt from the reference it dropped
  `eval --reference` and `publish --repo`, and put an unknown reference in the candidate's place.
- `chat` no longer draws an empty reply panel for an answer such as `361.` that Markdown reads as an
  empty numbered list.
- `bench` prints where it saved the result once.
- A cancelled `prepare` is no longer logged as a failed export, and a retry that reuses its export
  says "reused from cancelled job".
- `serve` with `stream: true` refuses a request that fails before generation starts with the same
  status and body as without stream: a prompt over the NPU's `MAX_PROMPT_LEN` (400
  `context_length_exceeded`), an image whose pixels do not decode, and a failure before the first
  token. These used to come as a 200 stream whose only event was an error. A failure after the
  stream has begun is still an error event, now with the full `{message, type, param, code}` body.
- `serve` refuses an out-of-range `max_completion_tokens` with the code
  `serve.request.max_completion_tokens`, matching its `param`; it used to be
  `serve.request.max_tokens`.
- A conversion profile's runtime properties are applied only on the profile's own device:
  `--profile npu-int4-sym-cw` on the CPU or GPU, or auto mode falling back from the NPU, no longer
  passes `MAX_PROMPT_LEN`/`MIN_RESPONSE_LEN` to the smoke test (both plugins refused it, exit 50)
  or records them in the manifest's recommended runtime.
- phi3, mistral, mixtral, gpt2, bloom and the other model types optimum 2.3.0 cannot export on
  Python 3.14 are refused before the download with the existing advice (exit 10), instead of after
  a multi-gigabyte download and the start of the conversion; Python 3.13, a fixed optimum and
  unknown versions are not affected.
- The export runner keeps the first 50 and last 300 lines of each output stream plus the header
  lines of exceptions in between, so a long NNCF failure listing no longer hides the exception line
  the registry classifies.
- The torchvision repair for `export.torchvision.build_mismatch` replaces torchvision alone from the
  CPU index instead of force-reinstalling torch and every dependency, which upgraded numpy past
  nncf's bound and left `pip check` failing.
- `yonga registry profiles` no longer breaks profile ids across lines in non-TTY or 80-column
  output. The summary wraps instead, and on a narrow terminal the weights column (which the id
  spells out) and then the size band give way, so the summary is not broken mid-word.
- `yonga cache list` and `cache prune` plans no longer squeeze cache keys beside a long
  directory-name owner or the Why column: optional columns are dropped first, then the owner is cut
  or dropped. The logs index shortens a repository id from its organisation. An owner that is a
  `prepare` job's staging directory reads `staging (job PREFIX)` instead of `model`.
- Paths on `Saved:` (`bench`, `eval`), `env list`, `cache list` and `logs` lines are no longer
  folded mid-path on narrow terminals, so they can be copied whole. A long `env list` pin line
  wraps under its indent instead of at column 0.
- `yonga search` keeps repository ids whole and download counts readable at 80 columns and in
  piped output (it printed `30.` for 30.8M with no header, and wrapped ids mid-name): the format,
  download and architecture columns are dropped, in that order, when the terminal is narrow.
- Next-step hints and remediation commands repeat the root options you gave (`--data-dir`,
  `--cache-dir`, `--config`, `--offline`): after `prepare --data-dir X`, "Chat with it: yonga chat
  ID" looked in the default store and failed with exit 70. The same holds for `yonga list`, `cache
  prune`, `logs` and registry steps.
- Offline advice ("Not available offline", "Search needs the network", the export environment and
  `publish`) prints your command without `--offline` instead of only naming `YONGA_OFFLINE` and
  `HF_HUB_OFFLINE`.
- A cancelled conversion's advice prints your command, like other re-run advice, and the cancelled
  job's state records the profile and device its plan chose instead of `profile_id: null`.
- `prepare --device D` reusing an artifact of the same revision and profile that was prepared on
  another device probes it on D first (the compile and smoke test of a fresh preparation, recorded
  in the manifest) and reports D; a failure stops the job without a fallback. It used to return the
  CPU artifact as "Device CPU" without probing the requested GPU.
- `docs/REGISTRY_SPEC.md` no longer says `prepare --json` carries an `export_blocked` key: the
  refusal is the usual failure document, with the rule's diagnostic under `diagnostic`.

### Security

- **OpenVINO's usage reporting no longer loads.** `openvino_telemetry` sent every `import openvino`
  to Google Analytics and counted it in `~/intel/stats`. yonga keeps it from loading in every Python
  process it runs, including `optimum-cli` in managed export environments, without writing to
  `~/intel`.
- **yonga's advice no longer asks for a raw vendor import.** The `export.torchvision.build_mismatch`
  remedy, and the advice for an exporter import no rule recognises (`prepare` and `doctor`), check
  the import with `yonga doctor --full`, which imports with that reporting off, instead of a plain
  `python -c "import optimum.exporters.openvino"`, which reported to Google Analytics and counted in
  `~/intel/stats`. The registry and the user guides no longer advise such a command.

## [0.4.0] - 2026-10-09

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
- **`yonga cache list` and `yonga cache prune`: the OpenVINO compile cache is no longer
  write-only.** Every compile cache entry yonga uses now gets an ownership record,
  `.yonga-entry.json` (artifact id and path, repository, profile, device, OpenVINO version, runtime
  properties, created and last used); writing it never fails a load. `cache list` (or plain
  `yonga cache`) shows each entry's size, last use, device, OpenVINO version, owner and status:
  `in-use`, `orphaned` (the artifact is gone), `stale-runtime` (built by an OpenVINO release other
  than the installed one, so never reused) or `unattributed`. Entries made before the records are
  attributed by recomputing the keys of every artifact in the store and every artifact a saved
  benchmark names. Stale-runtime is relative to the OpenVINO running the command; an entry of
  another release used after this one first built an entry here belongs to another install
  sharing the cache and stays `in-use`. `cache prune` removes orphaned and stale-runtime entries,
  skipping any entry used after the plan was made (unless `--all`); `--older-than DAYS`,
  `--unattributed` and `--all` go further, `--dry-run` only shows the plan, and it asks first
  unless `--yes`. It deletes only key-named, real directories directly under the compile cache
  root, never follows a symbolic link, and reports the bytes freed. `--json` on both.
  `docs/CLI_SPEC.md` section 12b.
- **`doctor` reports the compile cache** under Storage: entries, size and the reclaimable part,
  a warning naming `yonga cache prune` from 10 GiB reclaimable or 25 % of the free space on the
  cache filesystem (never below 1 GiB), and the numbers under `compile_cache` in `--json`.

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
- **The VLM families follow the size bands.** Qwen2.5-VL and Qwen3.5 no longer pin the CPU and GPU
  to ratio 1.0, so below 5B they get `cpu-int4-asym-g128-r08` and `gpu-int4-asym-g128-r08`, and
  from 5B the ratio-1.0 profiles, like every text family. Measured on 2026-10-09, text only,
  against INT8: on `Qwen/Qwen2.5-VL-3B-Instruct` ratio 0.8 lowers the divergence by 66 % on the CPU
  (perplexity +35.2 % to +7.9 %, 2633 to 2896 MiB) and by 39 % evaluated on the GPU (+23.8 % to
  +12.7 %), for about 12 % of the GPU's tokens per second (48.4 to 42.5). Image-conditioned quality
  was not measured. A VLM's Hub count includes its vision tower, so it reaches 5B early, never
  late. `docs/PROFILE_COMPARISON.md` section 10.6.
- **phi3 and the generic rule get group-wise INT4 on the NPU below 5B.** `arch-phi3` and
  `arch-generic-text-generation` (mistral and unrecognised members of common families) no longer
  pin `npu-int4-sym-cw`, so below 5B the NPU gets `npu-int4-sym-g128` as every other family does;
  from 5B, and for an unknown size, it stays channel-wise. Against INT8, group-wise lowers the
  divergence by about 44 % on `microsoft/Phi-4-mini-instruct` (KL 0.427 to 0.240, perplexity
  +44.9 % to +20.3 %) and on `h2oai/h2o-danube2-1.8b-chat` (0.367 to 0.208, +30.0 % to +15.5 %),
  for 13 % and 26 % of the NPU's tokens per second (25.1 to 21.9 and 56.4 to 41.8 tok/s, with
  a fifth to a quarter more time to the first token); `--profile npu-int4-sym-cw` is the fast
  choice. The generic rule was measured with that one mistral model only. phi3 now claims CPU
  and NPU support as tested; the GPU was not tried. Both models were exported on Python 3.14 with
  optimum 2.3.0 plus the unreleased one-line fix of optimum pull request 2496, which changes
  descriptor binding only, not the exported graph; released optimum still cannot export them on
  Python 3.14 (Python 3.11-3.13 can). `docs/PROFILE_COMPARISON.md` sections 10.3 and 10.4.
- **Upgrading.** As in 0.3.0, a changed default is a new artifact, never a replacement: preparing
  one of these models again exports it under the new profile id next to the existing artifact,
  and the plan says so beforehand with the `--profile` that would reuse the old one.

### Fixed

- **The generation smoke test compiled into the default cache directory.** `prepare` and `probe`
  built the smoke-test pipeline with default settings, so `--cache-dir` and the configuration
  file's `cache_dir` and `compile_cache_enabled` were ignored for it, and `probe --timeout`'s child
  process did the same. Both now use the invocation's settings; the child receives them as
  `--cache-dir` and `--no-compile-cache`.
- **`yonga remove` finds compile cache entries built by any OpenVINO release** through their
  ownership records, and tries CPU, GPU and NPU keys whatever the manifest recorded; before, an
  entry built after an OpenVINO upgrade, or on a device the artifact was never probed on, was left
  behind.
- **`prepare` works again for Qwen3.5 below 5B parameters.** The 0.3.0 registry pinned Qwen3.5 to
  ratio-1.0 INT4 on the CPU and GPU, and that fails NNCF's group-size check on Qwen3.5 0.8B, 2B
  and 4B (`export.quantization.group_size_mismatch`: the one-channel weight of the rotary-embedding
  MatMul cannot be split into groups of 128), so the default `prepare` failed for every Qwen3.5
  below 5B. They now get the ratio-0.8 profiles, which prepare all three (NNCF keeps that weight at
  INT8; observed with NNCF 3.4.0, not guaranteed), with perplexity +19.9 %, +12.2 % and +8.0 %
  against INT8. Qwen3.5-9B still prepares at ratio 1.0. The limitation is recorded in
  `arch-qwen3_5`.

- **`serve` answers the request that hands a tool's result back** on an LLM artifact that runs
  OpenVINO GenAI's stateful pipeline (`ATTENTION_BACKEND=SDPA` on the GPU, and every NPU LLM).
  That pipeline refuses a `ChatHistory` whose last turn is not the user's, so the second half of
  every tool round trip failed with 500 `runtime.genai.generation_failed` ("Chat doesn't support
  switching between input types"). Such a conversation, and one ending with an assistant turn, is
  now rendered with the artifact's own chat template, tools and template variables included, and
  handed to the pipeline as tokens: once templated and without a second BOS, which on the reference
  host gives the same prompt tokens and answer as GenAI's own history route. The pipeline's chat
  state is reset on both sides of that call, because a history call after it otherwise failed
  inside the sampler ("token_idx < seq_len"). A conversation ending with the user's turn is sent
  as before. `docs/CLI_SPEC.md` section 14.
- `yonga probe` no longer prints a deferred raw NPU compile as an error with its diagnostic while
  it runs; the line now reads `deferred`, and the pipeline's result is the verdict.
- `python -m yonga.application.probe` settles a deferred device by building the GenAI pipeline, as
  `yonga probe` does. It had no smoke tester, so it reported every NPU LLM as failed.
- A relative `--data-dir` or `--cache-dir`, or a relative directory in `config.toml` or the
  environment, is made absolute against the current directory. A relative one broke `prepare`:
  the exporter wrote the model to the right place and the tokenizer to another, and validation
  refused the artifact for having no tokenizer.
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

[Unreleased]: https://github.com/ooguz/yonga/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/ooguz/yonga/releases/tag/v0.5.1
[0.5.0]: https://github.com/ooguz/yonga/tree/v0.5.0
[0.4.0]: https://github.com/ooguz/yonga/tree/v0.4.0
[0.3.0]: https://github.com/ooguz/yonga/tree/v0.3.0
[0.2.0]: https://github.com/ooguz/yonga/tree/v0.2.0
[0.1.0]: https://github.com/ooguz/yonga/tree/9bc1a6acc20720cb53656a052f04257e9e09ece9
