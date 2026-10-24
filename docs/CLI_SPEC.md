# CLI_SPEC.md

## 1. CLI principles

The CLI should feel like a package manager plus local model runner, not a thin wrapper around OpenVINO commands.

- concise by default;
- detailed with `--verbose`;
- raw traceback only with `--debug` unless unexpected fatal error requires it;
- expensive/destructive decisions shown before execution;
- JSON output available for automation on informational commands;
- stable exit codes;
- colors respect `NO_COLOR` and non-TTY output.

Use Typer/Click unless there is a compelling implementation reason otherwise.

## 2. Global options

```text
yonga [GLOBAL OPTIONS] COMMAND ...

--config PATH
--data-dir PATH
--cache-dir PATH
--offline
--json
-v, --verbose
--debug
--version
--help
```

Global options go before the command name: `yonga --offline --debug prepare MODEL`, not
`yonga prepare MODEL --offline`. Every command yonga suggests next (a hint after success, a
remediation step) repeats the global options that say where the data is and how to reach it
(`--config`, `--data-dir`, `--cache-dir`, `--offline`), so it runs against the same store;
advice that only the network can satisfy prints the command without `--offline`. `--json` is the one global option that commands also accept after
their own name (`yonga list --json`).

`--json` suppresses decorative UI and emits one JSON document for commands that support it. `chat`
and `serve` reject it (exit 70): a conversation is not one document, and `serve` reports its state
as JSON at `GET /health` instead.

## 3. `doctor`

```bash
yonga doctor [--full] [--json]
```

Default output sections:

```text
Host
  OS       Ubuntu 26.04
  Kernel   ...
  Python   3.14.x
  Install  AppImage  /home/user/.local/bin/yonga; 'yonga' on PATH is this AppImage
  RAM      32 GiB

OpenVINO
  Runtime      2026.4.0
  GenAI        2026.4.0.0
  Tokenizers   2026.4.0.0
  ABI family   ✓ aligned

Devices
  CPU  ✓ Intel ...
  GPU  ✓ Intel Arc Graphics
  NPU  ✓ Intel NPU

Drivers
  GPU kernel       ✓ xe  PCI 8086:64a0 at 0000:00:02.0
  GPU runtime      ✓ compute-runtime
  GPU in OpenVINO  ✓ listed
  NPU kernel       ✓ intel_vpu
  NPU runtime      ✓ level-zero
  NPU in OpenVINO  ✓ listed

NPU
  /dev/accel/accel0      ✓ accessible
  driver                 1.38...
  compiler-in-plugin     ✓ available

Storage
  cache filesystem       796 GiB free
  system /tmp            16 GiB tmpfs
  managed temp           ✓ disk-backed
  compile cache          42 entries, 61.0 GiB, 12.0 GiB reclaimable
```

Warnings must have remediation text.

The `Drivers` section lists every Intel GPU (PCI class `0x03`) and NPU (class `0x1200`) found in
`/sys/bus/pci/devices`, whatever OpenVINO makes of them, and for each one checks three things: the
kernel module bound to it (`i915` or `xe`, `intel_vpu`; `drivers.<kind>.kernel`), the user-space
driver OpenVINO's plugin loads (`drivers.<kind>.runtime`: Intel's OpenCL driver named by an ICD
file in `/etc/OpenCL/vendors` for a GPU, `libze_loader.so.1` and `libze_intel_npu.so` for an NPU),
and whether OpenVINO lists the device (`drivers.<kind>.openvino`). A missing piece is a warning
whose remediation names the package for the host's distribution family, from `ID` then `ID_LIKE`
in `/etc/os-release`: Arch and its derivatives, Debian and Ubuntu, or Fedora; any other gets the
upstream release page. With several GPUs the ids are `drivers.gpu0.*`, `drivers.gpu1.*`. A host
with neither is one informational line; the section is left out when the PCI bus cannot be read.
Whether OpenVINO lists a device is decided from its own names (`GPU.0`, `GPU.1`), not the folded
list the Devices section shows. A device bound to another kernel driver (`vfio-pci` for
passthrough) is told to be released, not to `modprobe`. A GPU before Gen12, which Intel's current
compute runtime does not cover, is informational when OpenVINO does not list it. Only doctor reads
the bus and the libraries (`DriverProbe`); `HardwareProbe.inventory`, which `prepare` runs, does not.
In the `NPU` section, a driver that reports compiler version 0 (`npu.compiler_version`, `missing`)
has no compiler library, whatever `NPU_COMPILER_TYPE` says; `npu.compiler_libraries` needs both
`libopenvino_intel_npu_compiler.so` and its loader unless a compiler answers anyway. The group line
is informational when the node is accessible anyway.

The `Install` line (`host.install`) says how yonga is running: an AppImage, with the image file's
path, or a Python environment, with `sys.prefix`. It also says what the command `yonga` on `PATH`
resolves to (`shutil.which`). Running from an AppImage, it is a warning when `PATH` has no `yonga`
(remediation: `<the AppImage> self install`, section 12c) or when `yonga` there is another
installation, such as an older AppImage, which is named with its path (and its version when
`self install` recorded it). A copy with the same content counts as this AppImage. For a Python
environment the line is information only. The `--json` document carries it under `installation`
(`kind`, `version`, `appimage`, `prefix`, `path_command`, `path_command_relation`: `this`,
`other_appimage`, `other` or `missing`, and `path_command_version`).

A missing `optimum-intel`, `nncf` or `transformers` is a warning in a Python environment, with
`pip install 'yonga[export]'` and the managed export environment as the alternatives. An AppImage
never carries them and nothing can be installed into it, so there the missing ones are a single
`export toolchain` line (`openvino.export_toolchain`, ok): `prepare` converts in a managed export
environment, with how many are built or that the first conversion builds one. An AppImage on a
disk-backed temp directory can therefore exit 0.

Where the toolchain is installed, versions alone do not prove it loads (a CPU torch next to PyPI's
torchvision passes every version check), so an `exporter import` line (`openvino.exporter_import`)
follows the packages. It is the import of `optimum.exporters.openvino` that `prepare` runs in
preflight, sharing its record: doctor imports once (a few seconds) when nothing is recorded for the
installed package versions and otherwise reports the record, so it stays fast. A success is ok; a
failure is a warning with the registry's diagnostic title and remedy (for example
`export.torchvision.build_mismatch`) and a note that `--full` checks again, which `--full` does.

The `export environments` line totals the disk under the environments directory exactly as
`yonga env list` does: the environments and the copy of the bundled Python they are made from.

The `compile cache` line reports the size of the OpenVINO compile cache and how much of it is
reclaimable: orphaned plus stale-runtime entries, as `yonga cache list` classifies them (section
12b). It is a warning, with `yonga cache prune` as the remediation, when the reclaimable bytes are
at least 10 GiB, or at least 25 % of the free space on the cache filesystem, but never below 1 GiB.
The `--json` document carries the same numbers under `compile_cache` (`path`, `entries`,
`total_bytes`, `reclaimable_bytes`, `warn_threshold_bytes`, `installed_openvino`).

Exit codes:

- 0: usable runtime exists;
- 2: runtime usable but warnings;
- 10: no usable OpenVINO device/runtime;

The last line names the cause of a 10: the OpenVINO runtime is not installed (with
`pip install 'yonga[openvino]'`), is installed but cannot be imported, or no device was found.

Every line is printed as plain text: square brackets in a command, a path, a model id or a message
from a library are never read as terminal styling.

Do not make warnings such as small `/tmp` fatal if managed temp solves them.

## 4. `inspect`

```bash
yonga inspect MODEL [--revision REV] [--files] [--json]
```

Output:

```text
Model       ...
Revision    <sha>
License     apache-2.0
Task        image-text-to-text
Architecture qwen3_5
Formats     GGUF
Size        ~16.7 GiB (one model; 38.5 GiB across 5 files in the repository)

Compatibility on this host
  Architecture identified from GGUF metadata; this repository has no config.json.
  NPU   known unsupported / current compiler rule
  GPU   supported/test candidate
  CPU   supported/test candidate

Preparation
  Direct OpenVINO source: no
  Safetensors source: no
  Known derivative: nDimensional/... (approval required)
```

`Size` states one model first, because a GGUF repository publishes several interchangeable quantizations of the same weights and summing them overstates the download several-fold; the repository total follows, qualified by how many files it spans.

When the architecture was identified from an identity hint instead of from `config.json` (`REGISTRY_SPEC.md` section 4), the default output says so beside the compatibility block, because every claim in that block rests on that identification. Which alias matched is a `--verbose` detail.

No multi-GB download.

## 5. `search`

```bash
yonga search QUERY [--limit 20] [--device npu|gpu|cpu|auto] [--json]
```

This command is optional in milestone 1 but desired for MVP.

Use Hugging Face API search and enrich results with compatibility where architecture metadata is available. Do not claim compatibility without evidence.

The table shows the repository id, the architecture, one column per device, the formats hinted from
Hub tags and the download count. A repository id is never wrapped or cut short, and a count is never
cut: on a narrow terminal the format, download and architecture columns are dropped, in that order.

## 6. `prepare`

```bash
yonga prepare MODEL \
  [--revision REV] \
  [--device auto|npu|gpu|cpu] \
  [--profile PROFILE] \
  [--source REPO_ID] \
  [--trust-remote-code] \
  [--ignore-exporter-requirements] \
  [--no-managed-env] \
  [--fresh-export] \
  [--gguf-file FILE] \
  [--yes] \
  [--force] \
  [--no-repair]
```

Default `--device auto`.

`--ignore-exporter-requirements` converts in the current environment even though a declared exporter
package version is unmet. Otherwise such a job runs its export in a managed export environment, or,
under `--no-managed-env`, stops in preflight. Like `--trust-remote-code` it is its own deliberate flag
and `--yes` does not imply it. It also overrides a known exporter failure: an issue rule's
`blocks_export` (`REGISTRY_SPEC.md` section 6.1) that refuses the export before any question and
before the download, for example phi3 or mistral on Python 3.14 with optimum 2.3.0 (exit 10). The
plan shows that refusal as a `BLOCKED` section, and the manifest records an override.

`--no-managed-env` refuses to build a managed export environment, so an exporter requirement the
installed packages cannot meet stops preparation at preflight instead. `--ignore-exporter-requirements`
also never builds one: it means "export in this environment regardless".

The plan's **Export environment** row always says where the export runs: a managed environment, or
this environment with the interpreter's path and why (none is needed for a GGUF import or an
OpenVINO copy). **Expected download** counts only the files the conversion reads, and the download
fetches exactly those: other runtimes' weights (`*.onnx`, Flax, TensorFlow), a native checkpoint
kept beside the Transformers one (`original/`), training leftovers (`runs/`, `training_args.bin`)
and, once the chosen weight format is complete, the other weight format's files are skipped.
Configuration, tokenizer and template files never are. With `--trust-remote-code` the whole
repository is fetched. **Estimated artifact** is computed from the profile's weight format and
ratio and the Hub parameter count (about 1.2 times parameters × bits); without a count it is the
source weight size.

When a ready artifact already exists for the revision and profile, the plan is a short reuse plan
(no download, temp, repair or environment rows) and asks no confirmation, so a re-run works
without `--yes`. `--force` prepares again. Reuse is keyed by revision and profile, not by device,
so with an explicit `--device` the artifact must already have passed a probe there; when it has
not (the CPU build of a profile, asked for with `--device gpu`), the plan says so and `prepare`
runs the same compile and smoke test on that device that a fresh preparation would, records the
probe in the manifest, and reports that device. A failure there stops the job without falling
back (exit 50), and the artifact stays ready for the devices it already runs on.

Before asking anything, `prepare` refuses a job that cannot finish, and the plan marks it
**BLOCKED**: the OpenVINO runtime is not installed (`runtime.openvino.missing`, exit 10, naming
`pip install 'yonga[openvino]'`); an explicit `--device` OpenVINO does not report
(`device.not_present`, exit 50); an explicit `--device` the registry records as unsupported for this
model with official or tested evidence (`device.known_unsupported`, exit 50, with the recommended
fallback); a gated or private repository and no Hugging Face token (`hub.auth.gated`, exit 21).
Nothing is downloaded in any of these cases.

A confirmation that needs a prompt in a session without a terminal (stdin is not a TTY) and without
`--yes` stops with `job.confirmation_unavailable` (exit 60), saying the question could not be asked
and naming `--yes`; a security confirmation says `--yes` cannot answer it. An answer of "no" at a
real prompt is `job.declined` (exit 60).

Every remediation that says to run the command again with a flag added or removed (`--yes`,
`--trust-remote-code`, `--ignore-exporter-requirements`, `--fresh-export`, without `--no-repair`,
`--debug`, and `--yes` for `remove`, `publish`, `env remove`, `cache prune`, `self install`,
`self uninstall` and `registry update`) prints the user's own command with that change: every
argument and option given, root options such as `--data-dir` and `--offline` included (and
`--debug` placed among them, where it parses), shell-quoted. An option whose name marks a credential
is left out, and a secret-shaped value is masked.

A system `/tmp` too small for the conversion is a `note:`, not a warning: the managed directory
was used, so it does not change the exit code. A managed temp directory on a RAM-backed filesystem
stays a warning.

When an earlier job with identical inputs completed its export and then failed, the plan shows an
`Export` row naming that job, and stages 4 and 5 are skipped: the retained export is adopted once
every file has been shown to be what the exporter wrote, with repaired files restored from their
backups first. Anything that does not check out is discarded and the export runs normally. The
manifest records `export.reused_from_job`, and the result prints it as a `note:`, not a warning: an
adopted export does not change the exit code. `--fresh-export` always exports again. An export that was
itself interrupted is never reused.

A GGUF-only repository with no usable declared derivative is **imported** instead of exported,
when a registry `gguf` rule covers the installed OpenVINO GenAI (`REGISTRY_SPEC.md` section 8a).
Candidate files are the rule's preferred quantization labels in order, or the one `--gguf-file`
names. Each candidate's header is read from the Hub, and the first whose architecture and every
tensor type the rule verifies is chosen; the plan lists the files passed over and why. Only that
file is downloaded. The reader opens the file read-write, so it is given a private copy in staging
(a reflink where the filesystem supports one), never the read-only file in the Hugging Face cache;
the disk plan reserves room for that copy beside the IR. The plan shows it under **GGUF file**, its tensor mix as its **Quantization**
(the profile's quantization settings do not apply), and the profile as **Runtime profile**. The
artifact is named `<revision>--gguf-<label>-<device>` and records `export.backend:
openvino-genai-gguf`. Refusals: `source.gguf_unsupported` (no verified rule, or an architecture it
does not list), `source.gguf_no_importable_file` (every candidate holds an unreadable tensor type,
named), `source.gguf_file_not_found`; all exit 20.

Under **Quantization** the plan prints **Chosen because**: one sentence saying why the profile won
(its size band, the device default, the architecture or model rule that names it, or `--profile`),
with the parameter count and where it came from, and any size-banded profile the model's size kept
out. For a profile that `--profile` or a rule names, it also names the profile that would be chosen
without it (`automatic_profile_id`), when that is a different one. The same sentence goes into the manifest (`export.profile_selection`). It is absent for a GGUF
import or an OpenVINO copy, whose quantization no profile decides. When the revision is already
prepared for the device under another profile id (a default that changed since), the plan's known
limitations name that artifact and the `--profile` that would reuse it; the new export never
replaces it.

Stages displayed:

```text
1/7 Inspecting source
2/7 Resolving compatibility
3/7 Preflight
4/7 Downloading
5/7 Converting
6/7 Validating / repairing
7/7 Probing runtime
```

On known NPU unsupported graph in auto mode:

```text
NPU probe failed: Qwen3.5 graph is not supported by the current NPU compiler.
This is a model/compiler compatibility limitation, not a missing NPU driver.

Trying recommended fallback: GPU
GPU compile probe ✓
```

On explicit `--device npu`, do not automatically move:

```text
NPU preparation cannot be completed.
Recommended fallback: --device gpu
```

Output final artifact ID/path.

## 7. `list`

```bash
yonga list [--clean] [--json]
```

Shows prepared artifacts, source revision, profile, recommended device, size, last benchmark.

On a narrow terminal whole columns are dropped instead of every column being cut short: the
creation and benchmark times first, then the revision, `Ready`, the size and the model. The
repository id may instead lose its organisation (`…/SmolLM2-360M-Instruct`) and, once the revision
is gone, the end of its name. The artifact id, the profile and the device are never shortened.
When the `Ready` column has gone, unready artifacts are named under the table.

`--clean` retires stale staging and temporary directories under the retention policy
(`workspace_retention_hours`, `workspace_retention_max`) and never touches a running job's
directories.

## 8. `remove`

```bash
yonga remove ARTIFACT [--yes] [--json]
```

Only removes tool-owned prepared artifact and compile cache references. It must not delete Hugging Face global cache unless a future explicit cache command is used. The compile cache entries it removes are the ones whose ownership record names the artifact, plus the ones whose key the artifact's manifest reproduces (section 12b); `yonga cache prune` handles everything else in the compile cache.

Interactive confirmation unless `--yes`.

`ARTIFACT` must name exactly one artifact. A repository id, artifact-id prefix or copied
artifact id that matches several is refused with exit 70 and every candidate listed, even under
`--yes`; the read-only commands (`chat`, `bench`, `probe`, `eval`, `publish`) still take the newest
ready match, and say so on stderr when there was a choice: `Using ID (PROFILE, DEVICE), the newest
of N ready artifacts for MODEL; pass the artifact id to choose another.` A reference that matches
nothing is explained the same way by all of them: a model id that is not prepared gets the
`yonga prepare MODEL` command to run first (`artifact.not_prepared`; `eval` and `publish` keep their
own `*.artifact_not_found` codes), anything else is pointed at `yonga list`.

## 9. `chat`

```bash
yonga chat MODEL_OR_ARTIFACT \
  [--device auto|npu|gpu|cpu] \
  [--max-new-tokens N] \
  [--temperature X] \
  [--top-p X] \
  [--top-k N] \
  [--show-reasoning] \
  [--system TEXT]
```

If argument is a HF model without prepared artifact, prompt to prepare or explain the command to run. Avoid surprising long conversions on `chat` unless a future `--prepare` flag explicitly permits it. Only an `owner/name` reference is told to run `yonga prepare` with it; an artifact id or any other name that matches nothing is pointed at `yonga list`, since preparing an artifact id would look it up on the Hub.

The model is loaded (compiled) before the startup panel is drawn. A model that cannot load on the chosen device ends `chat` there with its category's exit code (section 15): 51 for an NPU compiler refusal, 10 when OpenVINO GenAI is missing. A message that fails later is reported and the prompt comes back; leaving (`/exit`, Ctrl-D, end of input) right after a failed message exits with that failure's code, 52 when no rule classified it.

Startup panel:

```text
╭──────────── Local Chat ────────────╮
│ Qwen3.5-9B ...                     │
│ Device: Intel Arc Graphics (GPU)   │
│ Profile: gpu-int4-asym-g128        │
│ Reasoning display: hidden          │
│ Max output: 4096 tokens            │
╰────────────────────────────────────╯
```

Commands:

```text
/help
/reset
/clear
/tokens N
/temp X
/top-p X
/top-k N
/reasoning on|off
/stats on|off
/model
/exit
```

Default response flow:

1. show spinner `Yanıt hazırlanıyor...` / locale-neutral equivalent;
2. generate without streaming hidden reasoning to terminal;
3. sanitize recognized reasoning wrappers;
4. with `--show-reasoning` or `/reasoning on`, render the reasoning, without its wrapper tags, in a dim panel;
5. render final answer as Markdown panel;
6. render dim metrics footer.

Ctrl-C while a reply is generated stops it at the next token, drops it together with the message
that asked for it, and returns to the prompt; a second Ctrl-C before it has stopped ends `chat`
(exit 60). Ctrl-C or Ctrl-D at the prompt leaves.

Later enhancement: safe streaming after model-specific parser identifies final-answer boundary.

## 10. `bench`

```bash
yonga bench MODEL_OR_ARTIFACT \
  [--device DEVICE] \
  [--suite quick|standard|long] \
  [--runs N] \
  [--warmups N] \
  [--json]
```

Summary table:

```text
Prompt   In tok  Out tok  TTFT     tok/s   TPOT    Wall
short       42      256   180 ms   24.6    40 ms   10.8 s
medium     512      256   ...
```

Save full JSON and print its path.

## 11. `probe`

Advanced command:

```bash
yonga probe MODEL_OR_ARTIFACT [--device DEVICE] [--all-submodels] [--quick | --timeout SECONDS] [--json]
```

Shows per-submodel read/compile result. This is the productized form of the manual diagnostic script used during development.

Each row says in one line what happened: a deferred raw compile reads "the GenAI pipeline decides",
and a failure the registry recognised is named by its diagnostic. The raw runtime text, often
dozens of lines of native compiler output, is shown with `--verbose`; `--json` and the job log
always carry it. The outcome is explained once, after every device was tried: with
`--device auto`, a device that failed while another passed is reported for information, not as a
verdict on the artifact. Without the OpenVINO runtime the probe stops before reading anything, with
`runtime.openvino.missing` and exit 10, as `bench` does.

A raw compile the registry marks `defer_to_pipeline` (on the NPU, every dynamically shaped LLM) is
recorded as `deferred` and settled by building the real GenAI pipeline and generating a few
tokens. For a large model on the NPU that build is a full NPU compilation and can take many
minutes; the probe says so before it starts.

- `--quick` never builds the pipeline. A device whose only open question is a deferred raw compile
  is reported `inconclusive` (`probe.inconclusive.quick`), with the commands that settle it.
- `--timeout SECONDS` (a positive number) builds the pipeline in a child process,
  `python -P -m yonga.adapters.smoke_child`, and stops it after SECONDS. A build stopped there is
  `inconclusive` (`probe.inconclusive.timeout`): never `ok`, and never a compiler verdict. A child
  that dies without printing its JSON result is `failed`, classified through the registry from its
  stderr like an in-process failure.
- The two cannot be combined (`probe.options.conflict`, exit 70); a timeout that is not a finite
  number above zero is `probe.timeout.invalid`, exit 70. Without either option the pipeline is
  built in-process, with no time limit.

`inconclusive` never counts as working — not in the exit code, the role matrix, the manifest or a
model card — and never as a failure of the device or the model. Exit codes: 0 when any device
passed; otherwise the code of the first definite failure; otherwise, when every remaining device
is inconclusive, 2. `--json` has `"ok": false` and `"inconclusive": true` in that case, and the
`quick` and `timeout_seconds` the probe ran with; tell it from a failure by the exit code or
`inconclusive`.

`python -m yonga.application.probe ARTIFACT` accepts the same `--quick` and `--timeout`, prints the
JSON report, and exits 0, 2 (inconclusive) or 1; an invalid `--quick`/`--timeout` combination exits
70, and argparse usage errors exit 2 as usual. A refusal before anything is probed (no such
artifact, no OpenVINO) is printed on stderr and exits with its own code, 70 or 10.

## 12. `registry`

Desired advanced commands:

```bash
yonga registry explain MODEL [--device DEVICE]
yonga registry validate
yonga registry profiles [--device DEVICE] [--json]
yonga registry paths
yonga registry update [--url URL | --file BUNDLE [--signature SIG]] [--yes] [--json]
yonga registry status [--json]
yonga registry reset
```

`explain` prints matched rules and why they won, and under the chat profile its tool-call format
("Tool calling"), or that it declares none and `serve` refuses tools.

`profiles` lists the conversion profiles of the merged registry, the ids `--profile` accepts, by
device and then by priority, with their weight compression, size band, priority and status (and in
`--json` their evidence and the layer that supplied them). A profile id is never wrapped or cut
short; the summary is the column that wraps, between words, and on a narrow terminal the weights
column, which the id spells out already, and then the size band make room for it. An unknown `--profile`
(`registry.profile_unknown`, exit 70) names the profiles for the requested device and points here.

`paths` inside an AppImage names the AppImage file for the built-in registry instead of its
per-run mount, which is gone once the command exits (`builtin_appimage` in `--json`).

`update` installs a remote registry bundle (`REGISTRY_SPEC.md` section 12). The URL defaults to
`registry_url` in the configuration, which is unset: yonga contacts no registry server unless told
to. It fetches `URL` and `URL.sig` over HTTPS only, or reads `--file` and `FILE.sig`. It then
verifies the bundle completely, prints the signing key, the bundle's version and expiry and the
rules it adds or replaces, and installs it after confirmation (`--yes` approves). Refusals, exit 70:
`registry.remote.no_trusted_keys`, `untrusted_key`, `bad_signature`, `signature_malformed`,
`expired`, `rollback` (older than the installed bundle), `tool_too_old`, `unsafe_path`,
`unknown_repair_implementation` (a repair rule naming code this yonga does not contain),
`unknown_tool_call_format` (a chat profile naming a tool-call parser this yonga does not contain),
`too_large`, `insecure_url`, and `invalid` (not a bundle document; the message names each field
that is wrong, such as a timestamp without a UTC offset). `--url` and `--file` together are refused
(`registry.remote.conflicting_sources`), as is `--signature` without `--file`, exit 70.
`registry_update_policy = "never"` refuses updates and ignores an installed bundle.

The installed bundle is one layer between the built-in rules and the user override directories,
and is verified again every time the registry loads. One that no longer verifies (expired, edited
on disk, its key removed) is left out and the built-in registry is used. It is not left out
silently: `status` reports why and exits 2, `validate` lists it under "Ignored" (`ignored` in
`--json`) and exits 2, `explain` warns that it explained without it, and `doctor --full` shows a
`registry.ignored` warning. The policy `never` is a choice and reports nothing. `reset` removes the
bundle.

Trust is configuration: a key is trusted when its public key (base64 of the raw 32-byte Ed25519 key,
one line) is a `*.pub` file in `registry-keys/` under the configuration directory. yonga ships no
key, and nothing it downloads can add one. `tools/registry_bundle.py` creates keys and builds signed
bundles.

## 12a. `env`

```bash
yonga env list [--json]
yonga env remove KEY [--yes]
```

Lists and removes the managed export environments `prepare` builds when an architecture's exporter
requirements cannot be met by the installed packages. They live under the cache directory, are used
only by the export subprocess, and never modify the environment yonga runs in.

`KEY` is an environment's key, a unique prefix of it, or its directory name (`export-KEY`). `remove`
always confirms first: without `--yes`, a non-interactive session or `--json` removes nothing and
exits 60; an unknown or ambiguous `KEY` exits 70. Inside the AppImage, environments are made from a
copy of the bundled Python under the same directory; `list` shows it as `base Python` (JSON:
`base_interpreters`, counted in `total_bytes`), and it is deleted with the last environment made
from it.

## 12b. `cache`

```bash
yonga cache [--json]                  # the same as 'cache list'
yonga cache list [--json]
yonga cache prune [--older-than DAYS] [--unattributed] [--all] [--dry-run] [--yes] [--json]
```

Shows and prunes the OpenVINO compile cache, `<cache-dir>/compile/<key>/`. Every pipeline yonga
builds compiles into one entry, keyed by a digest of (artifact identity, device, OpenVINO version,
runtime properties). The exception is `prepare`'s smoke test, which runs on the staging directory
before it moves into the store: OpenVINO keys blobs by the path they were compiled from, so nothing
could reuse them, and that pipeline is built without a compile cache. The managed temp directory,
export environments (section 12a) and the Hugging Face cache are out of scope.

Whenever yonga hands an entry to OpenVINO it writes `.yonga-entry.json` into it: schema version,
key, artifact id and path, repository and profile, device, OpenVINO version, runtime properties
and their fingerprint, and when the entry was created and last used. OpenVINO only opens the blob
files it names itself, so the record never interferes; a record that cannot be written is skipped
and never fails a load.

`list` prints, per entry: key, size, last use (the record's, or the newest file time), device,
OpenVINO version, owner (artifact id; `staging (job 0745994f)` for a `prepare` job's staging
directory, by the job id prefix `yonga logs` accepts; `*` when the owner exists outside this
artifact store) and status, then the total size and count. On a narrow terminal the model, last use, OpenVINO
version and device columns are dropped, in that order, then an owner known only by its directory
name is cut and finally the owner column goes; the key, the size, the status and an owning artifact
id are never shortened. Statuses:

| Status | Meaning |
|---|---|
| `in-use` | The owning artifact exists and the installed OpenVINO would reuse the blobs. |
| `orphaned` | The owning artifact is gone from the store and from its recorded path. |
| `stale-runtime` | Built by an OpenVINO release other than the one running this command (compared on the full version string `openvino.get_version()` reports; a bare release such as `2026.4.0` and its full build string count as the same release); this runtime can never reuse it. Not assigned when `openvino.get_version()` cannot be called. |
| `unattributed` | No ownership record, and no known artifact produces the key. |

Stale-runtime is relative to the OpenVINO running the command, and two yonga installs with
different OpenVINO releases (a pipx install and a development virtualenv, say) can share one cache
directory. So an entry built by another release that was last used *after* the running release
first built an entry here, according to the ownership records, is reported as `in-use`, with a
reason naming the other release: both runtimes are active against this cache. An entry of another
release last used before that moment is stale-runtime, as after an upgrade. An install that stops
using its entries loses them to the next default prune once the other install has built anything.

An entry without a record (one made before records existed) is attributed by recomputing the
keys every artifact in the store, and every artifact a saved benchmark names, could have produced:
devices CPU, GPU and NPU plus those the manifest probed or recommends; OpenVINO versions from the
installed runtime (its full build string), manifests, benchmarks and other records; no
properties, the manifest's recommended runtime properties, a recorded smoke test's properties and a
benchmark's. `yonga remove` uses the records and the same derivation for its own artifact to find
its entries; it only asks OpenVINO for the installed version when an entry without a record
exists.

`prune` removes orphaned and stale-runtime entries. `--older-than DAYS` also removes any entry not
used for that many days (an age reaching back before the earliest representable date selects
nothing), `--unattributed` also removes unattributed entries, and `--all` empties the
compile cache. It prints the plan (entries, why each was chosen, total size) and asks for
confirmation unless `--yes`; `--dry-run` prints the plan and deletes nothing. Only direct children
of the compile cache root whose names are 16 lowercase hexadecimal characters, and that are real
directories rather than symbolic links, are ever deleted; the check is repeated immediately before
each deletion, and anything else under the root is listed (with `--verbose`) and left alone. The
ownership record is deleted last, so the remains of an entry that could not be removed completely
keep their owner and status. Unless `--all` is given, an entry whose record or modification time
changed after the inventory was taken (a `chat`, `serve` or `bench` started on it while the prompt
waited, or a compile writing into it) is skipped and reported as used since the plan. The result
reports the bytes actually freed. An entry removed while a running `chat` or `serve` uses it costs
that model a recompile on its next load, nothing more.

`--json` documents: `list` carries `compile_cache_dir`, `artifacts_dir`, `installed_openvino`,
`count`, `total_bytes`, `reclaimable_bytes`, `by_status`, `bytes_by_status`, `entries` and
`ignored`. `prune` carries `criteria`, `dry_run`, `planned` (each entry with `selected_because`:
`orphaned`, `stale-runtime`, `unattributed`, `older-than` or `all`), `planned_bytes`, `removed`,
`used_since_plan` (planned keys skipped because something used them after the inventory),
`failed`, `freed_bytes`, `kept` (the number of entries not planned) and `ignored`.

Exit codes: 0 on success, including nothing to prune and a dry run; 2 when some entries could not
be removed (`failed` lists them); 60 when the confirmation is declined, or when it cannot be asked
(a non-interactive session or `--json` without `--yes`); 70 for an invalid `--older-than`.

## 12c. `self`

```bash
yonga self install [--bin-dir DIR] [--yes] [--json]
yonga self uninstall [--bin-dir DIR] [--yes] [--json]
```

Puts the running AppImage on `PATH` as `yonga`, and takes it off again. `DIR` defaults to
`~/.local/bin`; `~` is expanded and a relative `DIR` is taken from the current directory.

`install` works only when yonga runs from an AppImage: `APPIMAGE` names an existing AppImage file
and the interpreter running yonga is inside `APPDIR`. A pip-installed yonga that inherited another
application's `APPIMAGE` does not qualify. Otherwise it exits 70 (`self.not_appimage`) and says
where the environment's own `yonga` command is, how to activate it, and
`pipx install 'yonga[openvino,export]'`.

What already exists at `DIR/yonga` is examined with `lstat` and its first bytes, never run. A
regular file starting with the ELF magic and carrying `AI\x02` at offset 8 is a yonga AppImage.

| At `DIR/yonga` | `install` | `uninstall` |
|---|---|---|
| nothing | copies the AppImage there (`installed`) | nothing to remove, exit 0 |
| an AppImage with the same size and SHA-256 | nothing (`already_installed`) | removes it after a question |
| another AppImage | shows both versions where known, asks, replaces it (`replaced`) | removes it after a question |
| a symbolic link, directory, pip/pipx command, other file | refuses, exit 70 (`self.target.not_appimage`), leaves it alone | refuses, exit 70 |

The copy is written to a temporary file in `DIR`, flushed with `fsync`, made `0755` and renamed
over `DIR/yonga` with `os.replace`; what is at `DIR/yonga` is checked again immediately before
the rename, and if it changed since the question nothing is replaced (exit 70,
`self.target.changed`). A symbolic link at `DIR/yonga` is never followed. `DIR` is created when
missing. A `DIR` (or, when it is missing, its nearest existing parent) the user cannot write to is
refused with exit 10 (`self.bin_dir.not_writable`) and the advice to choose a directory the user
owns: yonga never uses sudo. Too little free space for the copy is exit 30.

After installing, it checks whether `DIR` is on `PATH` (comparing resolved paths) and whether `yonga`
on `PATH` now runs the copy. If not, it explains what to do: for `~/.local/bin`, that Ubuntu's
`~/.profile` adds it at login only if the directory existed then (log out and in), and the
`export PATH="$HOME/.local/bin:$PATH"` line for the current shell; for another `DIR`, the same line
to add to a startup file; when an earlier `PATH` entry has another `yonga`, its path. yonga never
edits a shell startup file.

The version of each copy and its SHA-256 are recorded in `<data-dir>/self-install.json`, which is
how a later `install` (and `doctor`) can name the version it would replace without running it;
`uninstall` drops the entry. Without an entry the version is "unknown".

`uninstall` removes only the file at `DIR/yonga`. Prepared models, logs, caches and settings are
kept; it prints their directories and that `yonga remove`, `yonga cache prune --all` or deleting
the directories frees the space. Run from a Python environment (pip or pipx), it says first that
it removes only the AppImage, not the installation running it, which still runs those commands
afterwards.

`--json` documents: `install` carries `action` (`installed`, `replaced`, `already_installed`),
`source`, `target`, `bin_dir`, `version`, `size`, `sha256`, `created_bin_dir`, `previous`
(`version`, `sha256`, `size`, or null), `bin_dir_on_path`, `path_command`,
`path_command_is_target`, `advice` and `exit_code`. `uninstall` carries `target`, `removed`,
`version` and `kept` (the data, cache and config directories).

Exit codes: 0 when `yonga` on `PATH` runs the installed copy (and for `uninstall`); 2 when the copy
is in place but `yonga` does not run it yet (`DIR` not on `PATH`, or shadowed); 10 for a directory
the user cannot write to; 30 for too little space; 60 when the confirmation is declined, or cannot
be asked (a non-interactive session or `--json` without `--yes`); 70 when not running from an
AppImage or when something other than an AppImage is in the way.

## 12d. `connect`

```bash
yonga connect opencode ARTIFACT [--host 127.0.0.1] [--port 8000] [--device auto]
                                [--context N] [--config PATH] [--yes] [--json]
yonga connect opencode --remove ARTIFACT [--config PATH] [--yes] [--json]
```

Registers a prepared artifact with OpenCode as an OpenAI-compatible provider, so that
`yonga serve ARTIFACT` can be used from `opencode` without hand-written configuration.
`--host`, `--port` and `--device` default to what `serve` uses, and the device resolves as
`serve` resolves it. `--config PATH` names the OpenCode file (`~` is expanded, a relative path is
taken from the current directory); without it the target is `$XDG_CONFIG_HOME/opencode/opencode.json`,
falling back to `~/.config/opencode/opencode.json`. `--context N` replaces the context limit read
from the artifact. The command shows what it adds or changes and asks before writing (`--yes`
skips the question), then prints the `yonga serve ARTIFACT --port P` command to run (with
`--host` and `--device` when they are not the defaults). Only `--config`, `--yes` and `--json`
combine with `--remove`; anything else is exit 70 (`connect.opencode.remove_options`).

**Checks**, in this order, before the OpenCode file is read or written (the artifact is opened as
a `ChatSession` but never loaded):

1. The reference resolves through the artifact store; the store's own errors apply.
2. The device resolves as for `serve`.
3. Tools. OpenCode sends its twelve tools with every agent request, so an artifact `serve` would
   refuse tools for cannot be used: exit 70 `connect.opencode.tools_unsupported`, with up to three
   prepared artifacts that do take tools, or the families whose built-in chat profile declares
   tool calling.
4. Prompt length. A fixed `MAX_PROMPT_LEN` below `OPENCODE_MIN_PROMPT_TOKENS` (8192) is exit 70
   `connect.opencode.prompt_too_short`. When the artifact itself is recommended for the GPU or CPU
   (a forced `--device npu`), the advice is to drop `--device npu`; otherwise it names another
   prepared GPU or CPU artifact of the same source
   model, else `yonga prepare MODEL --device gpu`. The constant is the first power of two above the
   5261 tokens of OpenCode 2.1.2's first agent request in an empty `HOME` (8514 with skills and an
   `AGENTS.md`), measured on the reference host with the Qwen3 chat template and tokenizer. Every
   built-in NPU profile compiles with 1024, so none passes.

**The entry.** One provider per host and port holds every artifact connected to that address:

```json
"yonga-8000": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "yonga (127.0.0.1:8000)",
  "options": { "baseURL": "http://127.0.0.1:8000/v1" },
  "models": {
    "073d5d7c8ea75283": {
      "name": "Qwen/Qwen3-4B · GPU · gpu-int4-asym-g128-r08",
      "tool_call": true,
      "limit": { "context": 40960, "output": 4096 }
    }
  }
}
```

The provider key is `yonga-<port>` for a loopback host and `yonga-<host>-<port>` otherwise
(characters outside `[A-Za-z0-9_-]` become `-`). The model key is the artifact id, which `serve`
answers to. `limit.context` is `max_position_embeddings` from the artifact's `config.json` (or its
`text_config`), unless `--context` is given; with neither, exit 70
`connect.opencode.context_unknown`. `limit.output` is the manifest's
`recommended_runtime.max_new_tokens`, else the configured `max_new_tokens`. When the prompt length
is fixed (and passed check 4), `limit.input` is `MAX_PROMPT_LEN` and `limit.context` defaults to
that plus `limit.output`; a `--context` below that sum is exit 70
`connect.opencode.context_too_small`, naming both numbers. When `YONGA_API_KEY` is set, `options.apiKey` is `{env:YONGA_API_KEY}`;
the key itself is never written. A provider that already exists is merged into key by key: yonga
sets `npm`, `options.baseURL` and, when `YONGA_API_KEY` is set, `options.apiKey`; the provider's
`name`, an `options.apiKey` already there and every other key stay as they are (`name` is written
only into a provider yonga creates). The model entry is merged the same way: yonga sets `name`,
`tool_call` and `limit` (replaced as a whole), and any other key the user added to it
(`reasoning`, `options`) stays. Outcomes: `added`, `updated` (the model entry's keys yonga writes, or one of the
provider fields yonga writes, differ; each difference is shown as before and after lines before the
question, also for `added` into an existing provider), `already_connected` (nothing yonga writes
would change; nothing written, no question).

**The file.** Parsed with the standard `json` module. JSONC (comments, trailing commas), a broken
file, a root, `provider` or `provider.yonga-…`.`models` that is not an object is exit 70
`connect.opencode.config_not_json`; the file is left alone. A directory given as `--config` is exit
70 `connect.opencode.config_not_a_file`. A missing file is created with mode `0600` and
`"$schema": "https://opencode.ai/config.json"` first, a missing directory is created. Other keys
and their order are kept; output has two-space indentation and a trailing newline. The write goes
to a temporary file in the same directory, is flushed with `fsync`, keeps the file's mode and is
renamed over the target with `os.replace`; a symbolic link is resolved and its target edited. Just
before the rename the bytes are compared with what was read, and a file that changed is exit 70
`connect.opencode.config_changed` with nothing written. A file or directory the user cannot read
(`config_not_readable`) or write (`config_not_writable`) is exit 10, as is an existing file that is read-only for the user (`config_not_writable`, refused
before anything is written, its mode never overridden); a full disk is exit 30 with
the code `config_not_writable`; sudo is never used. The file is never deleted.

**The record**, `<data-dir>/opencode-connections.json` (`{"schema": 1, "connections": [...]}`),
keeps per connection the resolved config path, provider key, artifact id, the model entry as
written, whether yonga created the provider (and the provider fields it wrote), and the time.
One record owns each provider yonga created; every later connect to that provider merges the fields
it wrote into the owner's, and a connect that re-creates a provider the user deleted takes the
ownership over. `already_connected` adds a record entry when there is none, so `--remove` finds the
connection. A record that cannot be written is a warning on stderr (`record_written` is `false` in
`--json`): the OpenCode file is already correct. A record file that exists but does not parse
(broken, or another schema) is never written over: it is moved to `opencode-connections.json.bak`,
replacing an older backup, a new record is started, and a warning names both files. A record file that cannot be read (permissions) is left as it is and counts as not written.

**`--remove ARTIFACT`** takes the artifact id as recorded, so it works after `yonga remove`.
A model entry whose keys yonga wrote are identical to the recorded ones is removed, with any key
the user added. One whose yonga keys the user edited is kept: the command
lists it with the reason in text and in the JSON `kept` list, and exits 70 after removing the rest
(there is no separate code). The provider goes too when yonga created it, no models are left and
its other fields are still what yonga wrote. Record entries are dropped for what was removed and
for entries whose model is already gone. When every recorded entry was kept the action is `kept`
(exit 70). The record is read before the file: no record entry for this file is `not_connected`,
exit 0, without reading the file, and when the record connects the artifact in other files the
output names each with its `connect opencode --remove ID --config PATH` command. A file that is not
plain JSON is `connect.opencode.config_not_json`, naming the `provider.KEY.models.ID` entries to
remove by hand. `yonga remove ARTIFACT` does not touch OpenCode's configuration; when the record
names the artifact it prints the `connect opencode --remove ID` command, with `--config PATH` when
the recorded file is not the default one.

`--json` for a connection: `action`, `config`, `created_config`, `provider`, `created_provider`,
`model`, `base_url`, `limit`, `serve_command`, `record_written` and `exit_code`; `config` is the
resolved file (symbolic links followed), as the record holds it. For `--remove`:
`action` (`removed`, `kept`, `not_connected`), `config`, `removed` (`[{provider, model}]`), `kept`
(`[{provider, model, reason}]`), `record_written` and `exit_code`. Errors use the usual `--json` failure document and exit code (section 15).

| Code | Category | Exit | When |
|---|---|---|---|
| `connect.opencode.tools_unsupported` | `configuration_error` | 70 | The artifact takes no tools. |
| `connect.opencode.prompt_too_short` | `configuration_error` | 70 | Fixed `MAX_PROMPT_LEN` under 8192. |
| `connect.opencode.context_too_small` | `configuration_error` | 70 | `--context` is below the fixed prompt length plus the output limit. |
| `connect.opencode.context_unknown` | `configuration_error` | 70 | No context length in the artifact; pass `--context`. |
| `connect.opencode.config_not_json` | `configuration_error` | 70 | The file is not a plain JSON object of the expected shape. |
| `connect.opencode.config_not_a_file` | `configuration_error` | 70 | `--config` names a directory. |
| `connect.opencode.config_changed` | `configuration_error` | 70 | The file changed between reading and writing. |
| `connect.opencode.remove_options` | `configuration_error` | 70 | `--remove` combined with `--host`, `--port`, `--device` or `--context`. |
| `connect.opencode.config_not_readable`, `config_not_writable` | `permission_denied` | 10 | The OS refused reading or writing the file or its directory, or the file is read-only. |
| `connect.opencode.config_not_writable` | `insufficient_disk` | 30 | The disk is full. |
| `connect.opencode.declined` | `user_cancelled` | 60 | The question was declined, or cannot be asked (no terminal, or `--json` without `--yes`). |

Exit codes: 0 for `added`, `updated`, `already_connected`, `removed`, `not_connected`; 10, 30, 60
and 70 as in the table; 70 also when `--remove` kept an edited entry (action `removed` when it
removed others, `kept` when it removed nothing).

## 13. `logs`

```bash
yonga logs [JOB_ID] [--json]
yonga logs JOB_ID --raw
yonga logs JOB_ID --bundle [--output PATH]
```

Useful after failed conversion. Without a job id it lists the 20 most recent jobs: an eight-character
id prefix, the local start time (the zone is named under the table), the command, the model or
artifact, the device, the outcome (`ok`, `failed`, `cancelled`, `running`, `unfinished`), the
duration and the log size; `--json` adds the full id, the stage and the failure code. One job's
view prints the same facts above its records, with every record time in local time. `--bundle` writes a redacted, anonymised archive of the job log, job
state, artifact manifest and host report for attaching to a bug report.

## 14. `serve`

```bash
yonga serve MODEL_OR_ARTIFACT \
  [--host 127.0.0.1] \
  [--port 8000] \
  [--device auto] \
  [--max-new-tokens N] \
  [--show-reasoning] \
  [--max-body-mb 8] \
  [--carry-images-forward]
```

Default bind must be localhost, not `0.0.0.0`. Requires the `serve` extra (FastAPI, uvicorn,
Pillow); without FastAPI or uvicorn `serve` exits 10 naming `pip install 'yonga[serve]'`. Pillow
is needed only to decode images: without it a VLM artifact is still served, `serve` prints a
warning, and a request with an image is a 500 `serve.images.dependency_missing` naming the extra.

Endpoints:

| Method and path | Answer |
|---|---|
| `GET /health` | `{"status": "ok", "model", "device", "pipeline", "uptime_seconds"}`. Never requires a key. |
| `GET /v1/models`, `GET /v1/models/{id}` | The one served model, in OpenAI's model-object shape. |
| `POST /v1/chat/completions` | OpenAI chat completions; `stream: true` sends server-sent events ending in `data: [DONE]`, and `stream_options.include_usage` adds a usage chunk. |

The server opens the same `ChatSession` `chat` does (registry chat profile, device resolution,
compile cache, reasoning policy) and loads the model before it starts listening, so a model that
cannot load fails `serve` itself. A port already in use is refused before the model is loaded.
Each request carries its whole conversation and nothing is kept between requests. Generation is
serialised; a streaming client that disconnects stops its generation, checked on every generated
chunk, also while the answer is being held back (hidden reasoning, a pending tool call).

`model` must be the name given to `serve`, the artifact id, or the source repository (404
`model_not_found` otherwise). Accepted parameters: `max_tokens` / `max_completion_tokens`,
`temperature` (0 is greedy), `top_p`, `top_k`, `seed`, `repetition_penalty`, validated with the same
ranges as `chat`'s slash commands, `stop`, and `tools` with `tool_choice` and
`parallel_tool_calls` (below). Unknown fields are ignored. Parameters that would change the answer
in a way yonga cannot honour are refused with 400 and `param` set: `n` other than 1, non-zero
`presence_penalty` / `frequency_penalty`, the legacy `functions` / `function_call` (also a message's
`function_call`), a `response_format` other than text, `logprobs`, content parts other than `text`
and `image_url` (below), and a conversation whose last message is neither the user's nor a `tool`
result. A runtime failure is a 500 whose `error.code` is the diagnostic code.

`stop` is a string or a list of at most 4 strings (`MAX_STOP_SEQUENCES` in
`src/yonga/application/serve.py`), each non-empty and at most 256 characters (`MAX_STOP_LENGTH`).
Anything else is a 400 `invalid_request` with `param` `stop`; an empty list means no stop. yonga
matches the sequences itself, on the visible answer after the reasoning sanitiser, never inside a
hidden reasoning block: a common stop such as `"\n\n"` would otherwise end a reasoning model's
answer before it began. With `--show-reasoning` the reasoning block is part of the visible answer,
so a stop sequence can end it there. The answer ends before the first stop sequence to complete
(the match with the smallest end; of two ending together, the longer one), without the match and
without the whitespace in front of it, and `finish_reason` is `stop`, also when generation had
reached its token limit. That is where generation stops, and it does not depend on how the text was
chunked: with `["Observation:", "ion"]` the answer ends before `ion`. Matches across chunk
boundaries are found: a stream holds back only the tail that could still become a match and the
whitespace in front of it, so no part of a stop sequence is ever sent. With a stop sequence the
model is streamed internally even for a non-streaming request, so generation ends at the match
instead of at the token limit. A streamed and an unstreamed answer to the same request end at the
same place.

The registry chat profile's `stop_strings` (`docs/REGISTRY_SPEC.md` section 8b) are separate:
model-level end markers OpenVINO GenAI applies natively and leaves out of the answer.

Tools are accepted only by a model whose registry chat profile declares a measured
`tool_calling` format (`docs/REGISTRY_SPEC.md` section 8b), and only when this artifact's own chat
template contains the literal the profile names in `template_contains`; `yonga registry explain`
shows the format under the chat profile. In the built-in registry that is Qwen2.5, Qwen3, SmolLM3
and Granite, the families seen producing parseable calls on the reference host. `chat-qwen2`
matches the whole qwen2 model type, but only artifacts whose template carries `<tool_call>` (the
Qwen2.5 Instruct models) accept tools; the Qwen3 MoE models have their own profile,
`chat-qwen3_moe`, without tool calling, because they were not measured. Any other
model refuses a non-empty `tools` with 400 `serve.request.tools_unsupported`, `param` `tools`, and
the message says why (no format declared, a format this yonga lacks, or a template that does not
ask for it). A model never offered tools it cannot use, and an answer is never read in a format
nobody saw that model write.

- `tools` holds at most 128 entries (`MAX_TOOLS`), each `{"type": "function", "function": {"name",
  "description"?, "parameters"?}}`: a name of 1 to 64 letters, digits, `_` or `-`, unique in the
  request, and parameters that are a JSON Schema object of type `object`. Anything else is a 400
  naming the entry (`tools[i]`, `tools[i].function.name`, ...). `strict` is ignored, and arguments
  are not checked against the schema.
- `tool_choice` `auto` (the default) and `none` are honoured. `none` keeps the tools out of the
  prompt and the answer is not read for calls. `required` and a named function would need
  constrained decoding and are refused with 400 `unsupported_parameter`, `param` `tool_choice`.
- `parallel_tool_calls: false` keeps only the first call of an answer that makes several.
- The tools reach the artifact's own chat template through OpenVINO GenAI's
  `ChatHistory.set_tools`, and so do an assistant message's `tool_calls` (arguments as an object,
  the convention templates expect) and `tool` messages. A `tool` message needs the `tool_call_id`
  of a call made by the assistant message before it (other tool results may sit between them) and
  takes that call's function name; anything else is a 400 naming `messages[i].tool_call_id`.
  `tool_calls` belong to assistant messages only, and each needs an `id`, type `function` and a
  function `name`; `arguments` that are not JSON reach the template as text, and JSON nested too
  deeply to parse is a 400 naming `messages[i].tool_calls[j].function.arguments`. A conversation
  that carries calls or results needs `tools` in the same request (400 naming
  `messages[i].tool_calls` or `messages[i].role`): dropping them would answer a different
  conversation than the one sent. The last message may be a `tool` result. OpenVINO GenAI 2026.4's
  stateful LLM pipelines (`ATTENTION_BACKEND=SDPA`, and every NPU LLM) refuse a `ChatHistory`
  whose last turn is not the user's, so on an LLM artifact such a conversation is rendered with
  the artifact's own chat template, with the same tools and template variables and the assistant
  turn opened, and handed to the pipeline as tokens, once templated and without a second BOS. It
  takes this route on every device, so a tool round trip does not depend on the attention
  backend. A VLM takes it as a `ChatHistory` as it is.
- The answer is read for calls outside reasoning only, so a call drafted while the model reasons is
  never made, whether the reasoning is shown or not. With `--show-reasoning` the reasoning block
  that opens the answer is passed through as before and the content after it is read for calls as
  if the reasoning were hidden; a later reasoning block is then left out, and the two parts are
  joined by a blank line. The profile's format names one of the parsers compiled into yonga:
  `tagged_json` (each call one JSON object between the profile's `open` and `close` markers, such
  as `<tool_call>` and `</tool_call>`) or `json_list` (the whole answer one JSON array of calls,
  after an optional `open` marker). A call names an offered function and carries an object of
  arguments (or JSON text of one). The calls come back as `message.tool_calls`, each `{"id":
  "call_…", "type": "function", "function": {"name", "arguments": "<JSON text>"}}`, with the text
  around them as `content` (`null` when there is none), and `finish_reason` is `tool_calls`.
- An answer that begins like a call but cannot be read as calls to the offered functions (broken
  JSON, an unclosed block, a function that was not offered, arguments that are not an object) is
  returned whole as text, never as a guessed call, with `finish_reason` `stop` or `length`. The
  server logs a warning that this happened, without the text.
- Streamed, text is released until a call could begin; from there everything is held until the
  model finishes, so no marker or partial call is ever sent as content. Each call then arrives as
  one delta, `{"tool_calls": [{"index", "id", "type", "function": {"name", "arguments"}}]}`,
  complete rather than in fragments, before the final chunk with `finish_reason` `tool_calls`. An
  answer held this way that turns out not to be calls is released as text at the end. A
  `json_list` answer is held from its first character when that is `[` or the marker.
- `stop` sequences are matched on the text outside the calls only. One that completes before the
  first call ends the answer there, with no calls and `finish_reason` `stop`, streamed or not.

Images (`image_url` content parts) are accepted by a VLM artifact only; the checks below run in
this order, each refusal naming the content part in `param`:

- Only user messages may carry images (400 `unsupported_parameter` otherwise), and an
  `image_url` part needs `image_url.url` (400 `invalid_request`).
- On an LLM artifact any image is a 400 `serve.request.images_unsupported`, `param`
  `messages[i].content[j]`.
- At most 4 images per request, counted over all messages (`MAX_IMAGES` in
  `src/yonga/application/serve.py`); more is a 400 `serve.request.too_many_images`, `param`
  `messages`.
- Images go with the last message only. OpenVINO GenAI 2026.4 binds images to the newest user
  turn of a stateless conversation: an image in an earlier turn fails with "Missing image/video
  with index 0", and list content in an earlier turn is rejected outright. So an image in any
  earlier message is a 400 `serve.request.images_earlier_turn` that says so, unless the server
  runs with `--carry-images-forward` (below) and the last message is the user's (not a `tool`
  result, which takes no image).
- Then `stop` and the generation settings are checked, before any image is read.
- `image_url.url` must be a `data:image/(png|jpeg|webp|gif);base64,` URL. `http`, `https`, `file`
  and every other scheme are refused with 400 `unsupported_parameter`, `param`
  `messages[i].content[j].image_url.url`: permanently, with no option to allow them. yonga never
  fetches an image; a server that fetched addresses its clients chose could be made to reach
  services on its own host or network. Another media type, a data URL that is not base64, or one
  with parameters such as `charset` is refused the same way.
- Broken base64, bytes that are not a PNG, JPEG, WebP or GIF image (Pillow is offered no other
  decoder, whatever the declared type), and an image of more than 4096 x 4096 pixels
  (`MAX_IMAGE_PIXELS`) are a 400 `invalid_request` with the same `param`. The size is read from
  the image header and checked before any pixel is decoded, which also stops decompression bombs;
  a larger image is refused, never scaled down. The encoded size is bounded by the body limit.

Only the image headers are read before a request waits for the pipeline, on a worker thread, not
on the event loop. The pixels are decoded once the request holds the pipeline, so however many
requests are queued, at most one holds decoded images; a waiting request costs its body and no
more. An image whose header is fine but whose pixel data is broken (a truncated file, say) is
therefore found only then, still before generation starts: a 400 `invalid_request` with the same
`param`, with or without `stream`. Any failure inside Pillow
counts as a broken image, never as a server error. An accepted image is used as sent: the first
frame of an animation, an EXIF rotation applied as a browser shows it (an EXIF block that cannot
be read leaves the image as stored), transparency on white, 16-bit greyscale scaled to 8 bits;
`image_url.detail` is ignored. The images go to the pipeline with the last user message, in order,
ahead of its text (GenAI places untagged images there); a message holding only images is a
question too. GenAI reads its universal tags `<ov_genai_image_N>` and `<ov_genai_video_N>`
anywhere in a VLM prompt as "image N goes here", so those tags are removed from the text of every
message sent to a VLM artifact, repeatedly until none is left, and typing one cannot claim or move
an image. A model's own vision tokens typed as text (such as Qwen2-VL's
`<|vision_start|><|image_pad|><|vision_end|>`) are not removed: GenAI may read them as image
placeholders too, and which they are is per-family knowledge the registry does not hold yet.

`--carry-images-forward` trades accuracy for compatibility. Clients such as Open WebUI resend every
image of the conversation with each follow-up question, which the default refuses. With the flag,
images of earlier messages move onto the last user message, before its own images, and its text
gains a first line such as `[Image 1 was attached to an earlier message in this conversation.]`.
The model then sees every image as part of the newest turn: it is told that some came earlier, but
not with which words, so an answer that depends on *when* an image was shown may be wrong. The
4-image limit still counts all of them.

Request limits (`DEFAULT_MAX_BODY_BYTES` in `src/yonga/api/app.py`, `MAX_MESSAGES` in
`src/yonga/api/openai.py`):

- A request body over 8 MiB is refused with 413 `request_too_large`; `--max-body-mb N` sets the
  limit from 1 to 64 MiB. Base64 makes an image a third larger, so the default leaves room for
  about 6 MB of image files. A declared `Content-Length` over the limit is refused before the body
  is read; otherwise the bytes are counted as they arrive, so a chunked body without a length is
  held to the same limit.
- More than 4096 messages in one request is refused with 400, `param` `messages`.

Every error response uses OpenAI's shape, `{"error": {"message", "type", "param", "code"}}`:

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request` | The body does not validate (missing or mistyped fields, too many messages, `stop` outside its limits, an image that does not decode or is too large, a malformed tool or tool turn); `param` names the field. A body that is not JSON, or is sent without a JSON `Content-Type`, has `param` `null`. |
| 400 | `unsupported_parameter` | A parameter yonga cannot honour (above), including an image URL that is not an accepted `data:` URL and a `tool_choice` that forces a call; `param` names it. |
| 400 | the diagnostic code (`serve.request.*`) | A generation setting outside the ranges `chat` accepts, an image refused by `images_unsupported`, `too_many_images` or `images_earlier_turn`, or tools refused by `tools_unsupported` (above, with `param`). A setting ends the code as the client spelled it: `max_completion_tokens` out of range is `serve.request.max_completion_tokens`, `param` `max_completion_tokens`. |
| 400 | `context_length_exceeded` | The prompt (the conversation with its tools and images) is longer than a pipeline compiled for a fixed prompt length takes: the NPU's `MAX_PROMPT_LEN`. The message gives both token counts, names the offered tools and images when there are any, and says what to do: shorten the conversation, offer fewer tools or send fewer images, or use a GPU or CPU artifact, whose prompt length is not fixed. `param` is `messages`. |
| 401 | `invalid_api_key` | A key is configured and the request does not carry it. The `type` is `invalid_request_error`, as OpenAI's. |
| 404 | `model_not_found` | `model` does not name the served model. |
| 404 | `not_found` | No such path. |
| 405 | `method_not_allowed` | The path exists but not for this method. |
| 413 | `request_too_large` | The body passed the size limit. |
| 500 | the diagnostic code | Generation failed for any other classified reason, or an image arrived and Pillow is not installed (`serve.images.dependency_missing`). |

Other HTTP errors raised by routing carry `invalid_request`.

With `stream: true`, the status waits until the pipeline has accepted the prompt, so a request
refused before generation starts gets the same status and body as above: a prompt over the NPU's
`MAX_PROMPT_LEN`, an image whose pixels fail to decode, and a generation that fails before its
first token are a 400 or 500, never a 200 stream. The role chunk that opens a stream is sent with
the status. A generation failure after the stream has begun cannot change the status, which is
already 200: it arrives as one event, `data: {"error": {"message", "type", "param", "code"}}`,
the body the table above gives (`code` `internal_error` for an unclassified failure), followed by
`data: [DONE]`. No error message carries the OpenVINO source location
(`Check '...' failed at FILE:LINE`) of a runtime failure.

When `YONGA_API_KEY` is set in the environment, every endpoint except `/health` requires
`Authorization: Bearer <key>` (401 otherwise). Authentication runs before the size limit, so a
client without the key is refused before any body is read. The key is read from the environment
only, never from an option, and is never logged or echoed. Binding a non-loopback address prints a
warning, a stronger one without a key; set `YONGA_API_KEY` whenever `--host` is not loopback.

## 13a. `eval`

```bash
yonga eval CANDIDATE --reference REFERENCE [--device cpu|gpu] [--corpus FILE]
           [--allow-different-sources] [--json]
```

Teacher-forced quality comparison of two artifacts of the same weights. Both read the same text. At
every position the candidate's next-token distribution is compared with the reference's:

- mean `KL(reference || candidate)` in nats, the figure to rank by;
- top-1 agreement;
- each artifact's perplexity, and the candidate's change against the reference.

Nothing is sampled, so a result is deterministic for a pair of artifacts, a device and a corpus.

The logits come straight from the IR through `openvino.Core` (GenAI does not expose them),
tokenised by the artifact's own `openvino_tokenizer.xml`. `--device cpu`, the default, runs at
`INFERENCE_PRECISION_HINT=f32`, so the measured difference is the quantization's and not the
arithmetic's. `--device gpu` measures what the GPU actually computes. One model is loaded at a time;
the reference's log-probabilities are spilled to a memory-mapped file under the managed cache and
deleted afterwards.

Refusals (exit 70): `eval.device.unsupported` (`--device npu`, refused before anything loads: the
NPU compiler does not take the raw IR's dynamic shapes, and an NPU artifact is evaluated on the CPU
like any other); `eval.artifact_not_found`; `eval.same_artifact`; `eval.different_sources` (the
two were converted from different repositories or revisions); `eval.tokenizer_mismatch` (the
corpus tokenises differently, so the vocabularies differ); `eval.vocab_mismatch`.
`--allow-different-sources` accepts two repositories holding the same model in different formats,
such as a GGUF release and the original weights. It relies on your word for the lineage, which the
result records. The tokenizer and vocabulary checks still apply. The shipped corpus is `yonga-eval-v1`:
eight passages of original, CC0 text across prose, technical writing, Python, arithmetic,
dialogue, procedure, encyclopedic writing and Turkish, 1955 tokens with the Qwen3.5 tokenizer.
`--corpus` takes a `.json` of the same shape, or plain text split on blank lines. Results are
saved under `<data>/evaluations/`; the output ends with `Saved: PATH`, and `--json` carries the
same path as `result_path`, as `bench` does.

The result is relative to the reference, which is itself an artifact. The `cpu-int8-asym-ref` and
`cpu-fp16-ref` registry profiles exist to prepare one, and never become a device's default.

## 14a. `publish`

```bash
yonga publish ARTIFACT --repo NAMESPACE/NAME \
  [--public] \
  [--update] \
  [--license-reviewed] \
  [--dry-run] [--output DIR] \
  [--yes] [--json]
```

Implements `SPEC.md` section 21. The destination is always explicit, and the repository is private
unless `--public` is passed. Before anything is uploaded, planning checks the following and refuses
when a check fails:

| Check | Refusal |
|---|---|
| `--repo` is `NAMESPACE/NAME` | `publish.repo_invalid` (70) |
| not in offline mode | `publish.offline` (70) |
| the artifact is ready: validated, one device probed OK | `publish.artifact_not_ready` (70) |
| a token exists, can write, and owns the namespace (account or organisation) | `publish.hub.not_logged_in`, `…read_only_token`, `…foreign_namespace` (21) |
| an existing repository is only written with `--update` | `publish.repo_exists` (70) |
| no license involved forbids derivatives (`-nd`) | `publish.license.restricted` (70); `--license-reviewed` does not override it |
| every license is recognised, or `--license-reviewed` states it was read | `publish.license.unreviewed` (70) |
| every file still matches the manifest's SHA-256 | `publish.files_changed` (70) |

Licenses are read fresh from the Hub for the requested model **and** the converted source: a
derivative is bound by both. Recognised identifiers are permissive (`apache-2.0`, `mit`, `bsd-*`,
`cc-by-4.0`, …), conditional (`llama3*`, `gemma`, `*openrail*`, `cc-by-nc-*`; allowed, and the card
says the terms carry over), or restricted. Anything else, including no license, is unknown.
`--yes` never implies `--license-reviewed`.

The upload is one commit: the files the manifest lists, the source's `LICENSE*`/`NOTICE*`/
`USE_POLICY*` files, a generated `README.md` and `yonga-manifest.json`, the manifest with the home
directory, login name and hostname anonymised and secrets redacted. The artifact's `.yonga/`
directory (repair backups, validation records) is never uploaded.

The model card states that the artifact is a quantized conversion and not a fine-tune. It lists
lineage with revisions and licenses, every export setting and package version, and every repair with
its before and after digests. Devices appear with probe evidence: `works` only for a passing probe,
`failed` with the diagnostic code, `inconclusive` if a recorded probe is (`yonga probe` does not
write the manifest; `prepare` always settles its probes), `not tested` for a device never probed. It
also says so when the profile's target device is not among the verified ones, and lists every file
with its SHA-256.

`--dry-run` runs every check (credential problems become warnings) and writes `README.md`,
`yonga-manifest.json` and `upload.txt` (digest, size and path of every file) to `--output`, by
default `./yonga-publish-<artifact>`. Nothing is uploaded. Without `--dry-run` the upload asks for
confirmation; `--yes` approves it, and a non-interactive run without `--yes` exits 60.

## 15. Exit codes

Reserve stable classes:

```text
0   success
2   success with warnings / doctor warnings / probe inconclusive
10  environment unusable
20  model/source resolution failure
21  authentication/gated access required
22  remote code approval required
30  insufficient disk/memory preflight
40  conversion failure
41  validation failure
42  repair required but disabled/denied
50  requested device unsupported
51  runtime compilation failure
52  runtime generation failure
60  user cancelled
70  registry/configuration error
99  unexpected internal error
```

A command line that cannot be parsed exits 70, like an option value that parses but is invalid:
an unknown command or option, a missing argument, a value of the wrong type (`--timeout abc`), and
`yonga` with no command, which prints the command list. Click's own code for these is 2, which
here means warnings or an inconclusive probe, so the root command group remaps it; Click still
renders the message. `--help` exits 0.

A failure exits with its category's code (`CATEGORY_EXIT_CODES` in
`src/yonga/domain/diagnostics.py`). Codes 40, 42 and 52 are no category's code: they name the stage
a job died in (conversion, repair, generation) and appear only when the category is
`unknown_runtime_error`, except that `bench` exits 52 when none of its runs produced a measurement. Code 41 is both: the validation stage's code for an
`unknown_runtime_error`, and the category code of `unsupported_ir_type` and
`tokenizer_ir_incompatible`.

`prepare` exits 50 when it cannot use a device, whatever the diagnostic's category: an explicit
`--device` that is not present or whose probe fails, and an automatic run with no fallback left. The
exception is a failure whose diagnostic marks the environment as broken (`environment_broken`), such
as `npu.device.permission_denied` or `device.driver_missing`: no fallback is attempted, and
`prepare` exits 10, because the host has to be fixed before any device choice matters. `probe`,
`chat` and `bench` keep the category's code (a `probe` failure that stays `unknown_runtime_error`
exits 51, not 99), so an NPU compiler refusal is 51 there and 50 from
`prepare`. The diagnostic's code and category are in the message and the `--json` document either
way.

## 16. Localization

MVP UI strings may be English-first, but do not embed logic in strings. Keep a future localization boundary. Model answer language is model-driven; system prompt should default to answering in the user's language when appropriate.
