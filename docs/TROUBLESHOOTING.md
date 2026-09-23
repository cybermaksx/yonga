# TROUBLESHOOTING.md

Every failure `yonga` reports carries a stable code, a category, remediation steps and the path to a
job log. This document maps the codes to what they mean, lists the exit codes, and works through the
failures that have actually been seen on the reference host.

The code tables were produced by enumerating the shipped registry, not by hand:

```bash
python -c "
from yonga.registry import load_builtin_registry
for rule in load_builtin_registry().issues:
    print(rule.classification.code, rule.id, rule.classification.category)
"
```

## 1. How classification works

When a step fails, the raw error text goes through three stages:

1. **Registry issue rules** (`src/yonga/registry/builtin/issues.yaml`). The most specific matching
   rule wins: a rule naming a model family beats one naming only a device, which beats one naming
   only a signature. The rule that won is what produced the `code` you see, and the raw text it
   matched is kept in the job log. (`yonga registry explain MODEL` explains architecture, profile
   and chat resolution for a model; it does not replay a failure classification.)
2. **Built-in signature inference** (`GENERIC_SIGNATURES` in `src/yonga/registry/resolver.py`), for
   conditions that are not model-specific at all: a full disk, a bad allocation, a 401 from the Hub.
3. **`runtime.unknown_error`**, which keeps the raw text and asks for a bug report.

Raw error text is always kept (redacted, truncated to 4000 characters, tail preserved) in the
diagnostic and in full in the job log.

## 2. Exit codes

From `docs/CLI_SPEC.md` section 15; the mapping from category to code lives in
`CATEGORY_EXIT_CODES` in `src/yonga/domain/diagnostics.py`.

| Code | Meaning |
|---|---|
| 0 | success |
| 2 | success with warnings (this is what `doctor` returns on a tmpfs `/tmp`) |
| 10 | environment unusable (missing packages, driver, permission) |
| 20 | model/source resolution failure |
| 21 | authentication/gated access required |
| 22 | remote code approval required |
| 30 | insufficient disk/memory preflight |
| 40 | conversion failure |
| 41 | validation failure |
| 42 | repair required but disabled/denied |
| 50 | requested device unsupported |
| 51 | runtime compilation failure |
| 52 | runtime generation failure |
| 60 | user cancelled |
| 70 | registry/configuration error |
| 99 | unexpected internal error |

A closed stdout is **0**, not 99: see section 8.

Codes 40, 41, 42 and 52 say *where* a job died rather than *why*, so they appear only when the
category itself is uninformative. A conversion that fails for a classified reason exits with that
reason's code: the NNCF group-size failure in section 5.4 exits **70**, not 40.

## 3. Registry issue codes

Eleven rules ship today.

| Code | Rule id | Category | Exit | Means | Do this |
|---|---|---|---|---|---|
| `npu.compiler.unsupported_graph.qwen35` | `qwen35-npu-fp16-bufferize` | `compiler_unsupported_graph` | 51 | The NPU compiler refused a Qwen3.5 graph: an operation is implemented only for FP16 on the NPU. | Prepare for the GPU: `yonga prepare MODEL --device gpu`. The NPU is not broken. |
| `npu.compiler.dynamic_shape` | `npu-compiler-dynamic-shape` | `compiler_unsupported_graph` | 51 | The IR still has dynamic dimensions; the NPU compiler needs static shapes. | Load through OpenVINO GenAI with `MAX_PROMPT_LEN`/`MIN_RESPONSE_LEN` instead of compiling the raw IR, or use the GPU. |
| `npu.compiler.unsupported_graph` | `npu-compiler-unsupported-graph-generic` | `compiler_unsupported_graph` | 51 | The NPU compiler refused the graph, and no family-specific rule applies. | Use GPU or CPU; re-check after an OpenVINO/NPU compiler update. |
| `npu.compiler.pass_failed` | `npu-compiler-pass-failed` | `compiler_unsupported_graph` | 51 | A named NPU compiler pass failed, and the compiler answered — so the driver is working. | Use the GPU; the failing pass is in `yonga logs JOB_ID --raw`. |
| `tokenizer.ir.string_precision` | `tokenizer-string-precision-read` | `tokenizer_ir_incompatible` | 41 | The exported tokenizer XML writes `precision="STRiNG"`, which OpenVINO cannot read. | `yonga prepare MODEL --force`; the repair is applied automatically during validation. |
| `npu.device.permission_denied` | `npu-permission-denied` | `permission_denied` | 10 | The kernel refused access to the NPU node. | Join the `render` group (see `docs/INSTALL.md` section 3). No device fallback is offered on purpose. |
| `runtime.out_of_memory` | `runtime-out-of-memory` | `out_of_memory` | 30 | The host or the accelerator ran out of memory. | Close other applications; choose stronger compression or a smaller group size. |
| `storage.insufficient_disk` | `storage-no-space-left` | `insufficient_disk` | 30 | A write failed because the filesystem is full (including OpenVINO's `basic_ios::clear: iostream error`). | Free space, or point `YONGA_CACHE_DIR` at a bigger disk. |
| `export.quantization.group_size_mismatch` | `nncf-group-size-not-divisible` | `configuration_error` | 70 | The profile's group size does not divide the model's channel sizes. | `--profile npu-int4-sym-cw` (channel-wise, group size -1), or a profile with a smaller group size. |
| `export.transformers.symbol_missing.qwen35` | `qwen35-exporter-transformers-symbol-mismatch` | `package_abi_mismatch` | 10 | The exporter's Qwen3.5 patcher imports `Qwen3_5DynamicCache` and the installed Transformers does not define it. Nothing is broken; the two releases disagree about the Qwen3.5 modelling code. | Install the release the registry pins: `pip install "transformers==5.2.*"`. `yonga inspect MODEL` prints the requirement and what is installed. |
| `export.transformers.symbol_missing` | `exporter-transformers-symbol-mismatch` | `package_abi_mismatch` | 10 | Generic form of the above: `optimum/exporters/openvino/model_patcher.py` looked for something the installed Transformers does not provide, for any architecture. | Install the Transformers release the architecture requires; the failing import is in `yonga logs JOB_ID --raw`. |

The last two are a backstop, not the usual path. When the registry declares the architecture's
Transformers pin, preflight refuses before the download (section 5.6) and the export never runs.
They classify what comes back when it runs anyway — under `--ignore-exporter-requirements`, or for
an architecture whose pin the registry does not know.

## 4. Built-in signature codes

Used when no registry rule matches. These are in code rather than YAML because they are properties
of the environment, not of any model.

| Code | Category | Exit | Recognised from | Means |
|---|---|---|---|---|
| `device.permission_denied` | `permission_denied` | 10 | `permission denied`, `eacces`, `operation not permitted` | An accelerator device node cannot be opened. |
| `disk.insufficient_space` | `insufficient_disk` | 30 | `no space left on device`, `basic_ios::clear: iostream error`, `disk quota exceeded`, `enospc` | A write failed for lack of space. |
| `memory.out_of_memory` | `out_of_memory` | 30 | `std::bad_alloc`, `out of memory`, `cannot allocate memory`, `cl_out_of_resources`, `ze_result_error_out_of_*_memory` | Allocation failed. |
| `hub.auth_required` | `gated_model_auth_required` | 21 | `401 client error`, `403 client error`, `gatedrepoerror`, `access to model`, `is restricted and you are not in the authorized list` | The Hub refused the request. |
| `source.remote_code_required` | `remote_code_required` | 22 | `trust_remote_code`, `custom code`, `requires you to execute` | The repository ships Python that must run to load the model. |
| `packages.abi_mismatch` | `package_abi_mismatch` | 10 | `undefined symbol`, `glibcxx_`, `cxxabi`, `symbol not found`, ``version `glibc`` | Native libraries do not match; usually a mixed OpenVINO family. |
| `device.driver_missing` | `driver_missing` | 10 | `ze_result_error_uninitialized`, `ze_result_error_device_lost`, `no driver` | The accelerator runtime could not initialise. |
| `device.unavailable` | `device_unavailable` | 50 | `is not registered in the openvino runtime`, `was not found in the openvino runtime`, `device is not available` | OpenVINO does not expose that device here. |
| `ir.unsupported_element_type` | `unsupported_ir_type` | 41 | `unsupported element type`, `is_valid_type_idx` | The IR declares a type this OpenVINO build cannot read. |
| `runtime.unknown_error` | `unknown_runtime_error` | 99 | nothing matched | Unrecognised failure; the raw text is in the job log. |

There is no separate `validate` command: validation runs inside `prepare`. So
`ir.unsupported_element_type` tells you to run `yonga probe ARTIFACT --all-submodels`, which
compiles every submodel and shows which one is rejected, and then `yonga prepare MODEL --force`
to convert it again.

## 5. Failures we have actually hit

### 5.1 The NPU rejects a graph, and it is not a missing driver

Symptom, from a real NPU compile on this host:

```text
Exception from src/plugins/intel_npu/src/compiler_adapter/src/ze_graph_ext_wrappers.cpp:425:
Compilation failed. Level0 pfnCreate2 result: ZE_RESULT_ERROR_INVALID_ARGUMENT, code 0x78000004
 - generic error code for invalid arguments . [NPU_DRV] Driver reports a failure from
vclAllocatedExecutableCreate4, return code: 2013265924
[NPU_VCL] Compiler returned msg:
Exception from src/core/src/partial_shape.cpp:261:
to_shape was called on a dynamic shape.
```

`ZE_RESULT_ERROR_INVALID_ARGUMENT` looks like a driver failure and is not one. Read the sequence:
the Level Zero driver *answered*, the NPU compiler *ran*, and it *returned a message*. A missing or
broken driver looks different: the NPU is absent from `openvino.Core().available_devices`, or the
error is `ZE_RESULT_ERROR_UNINITIALIZED`. Only those are classified as `device.driver_missing`.

So `yonga` classifies this as `npu.compiler.dynamic_shape` (or
`npu.compiler.unsupported_graph[.qwen35]` for the bufferization form), with
`environment_broken: false`, and offers GPU and CPU as fallbacks. `yonga doctor` will still show the
NPU as healthy, because it is:

```text
NPU
✓  /dev/accel/accel0   accessible  group render
✓  driver              1789049548
✓  compiler-in-plugin  available  the NPU plugin compiles graphs itself
```

Do not "fix" this by reinstalling the driver. Either run the model on the GPU, or load it through
OpenVINO GenAI with `MAX_PROMPT_LEN` and `MIN_RESPONSE_LEN` set, which is what gives the NPU the
static shapes it needs.

The dynamic-shape message is one form of this. Another, seen on this host by loading a Qwen3-0.6B
artifact with `yonga chat ARTIFACT --device npu`, names the compiler pass instead:

```text
[ERROR] [vpux-compiler] MultiClusterStrategyAssignment Pass failed : Cannot get per cluster memory
shapes. Shape [1, 1, 8, 1], Unsupported distribution: #VPU.DistributedTensor<mode = SEGMENTED, …>
[ERROR] [vpux-compiler] Failed Pass MultiClusterStrategyAssignment on Operation
```

The reasoning is the same and so is the verdict: `npu.compiler.pass_failed`, exit 51, GPU and CPU
offered as fallbacks, and no suggestion that anything is wrong with the driver.

### 5.2 The tokenizer IR cannot be read back (`STRiNG`)

Symptom, at load or validation time:

```text
Exception from src/inference/src/cpp/core.cpp:84:
Check 'is_valid_type_idx(type_idx)' failed at src/core/src/type/element_type.cpp:115:
Unsupported element type: STRiNG
```

Cause: OpenVINO's own `save_model` writes the port precision as `STRiNG` (mixed case) while writing
the correct lower-case `element_type="string"` on the same layer, and `read_model` then refuses the
file it just wrote. This is reproducible with openvino 2026.4.0 and openvino-tokenizers 2026.4.0.0:
**every fresh export contains it**, two occurrences in `openvino_tokenizer.xml` and two in
`openvino_detokenizer.xml`.

`yonga` classifies it as `tokenizer.ir.string_precision` and fixes it with the
`ov-tokenizer-precision-string-casing` repair, which rewrites only the `precision` attribute and
leaves `element_type` alone. The repair runs inside `prepare`, and the plan says so before you
approve it:

```text
Expected repairs
• ov-tokenizer-precision-string-casing

Confirmation required
·  This export is expected to need a known artifact repair: ov-tokenizer-precision-string-casing
```

and during the run:

```text
6/7 Validating / repairing
Inventorying OpenVINO IR
Applying registry repairs: ov-tokenizer-precision-string-casing
```

If you passed `--no-repair`, validation stops instead, nothing is published, and you get
`repair.declined` (exit 42): "Validation failed and the registry has a matching repair
(ov-tokenizer-precision-string-casing), but repairs were disabled with --no-repair." Re-run without
that flag (add `--force` to rebuild an existing artifact directory).

### 5.3 `/tmp` is a tmpfs and conversion needs more space

`yonga` does not convert in `/tmp`, so a 15 GiB tmpfs is a warning and never an error. What matters
is free space on the filesystem holding `~/.cache/yonga`. Preflight refuses **before** downloading
anything (verified, exit 30):

```text
✗ Not enough temporary space

The managed temporary directory <cache>/tmp/<job-id> has 15.1 GiB free, but this conversion needs
25.0 GiB (20.0 GiB plus a 5.0 GiB safety margin).

  - Free space on the cache filesystem.
  - Or point YONGA_CACHE_DIR at a larger filesystem.
```

The fix is `YONGA_CACHE_DIR` (or `--cache-dir`) pointing at a disk-backed directory, not a bigger
`/tmp`. If a conversion does die mid-write on a full disk you get `storage.insufficient_disk` or
`disk.insufficient_space` instead, often through OpenVINO's misleading
`basic_ios::clear: iostream error`, which is a full filesystem and not a serialisation bug.

### 5.4 NNCF refuses the profile's group size

Symptom: `yonga prepare` exits 40 with nothing useful, and the job log ends with:

```text
nncf.errors.InvalidGroupSizeError: Failed to apply group-wise quantization with group size value 128.
Ensure that the group size is divisible by the channel size, or consider setting
`group_size_fallback_mode` to IGNORE or ADJUST. Failed nodes:
	"__module.model.layers.0.self_attn.v_proj/aten::linear/MatMul" (channel size: 16)
	"__module.model.layers.0.self_attn.k_proj/aten::linear/MatMul" (channel size: 16)
	...
```

Group-wise INT4 splits each weight row into fixed-size groups, so the group size has to divide the
layer's channel size. The shipped group-wise profiles use 128; a model with small channel dimensions
(the example above is `hf-internal-testing/tiny-random-LlamaForCausalLM`, channel size 16) cannot use
them. Nothing is wrong with the installation, the device or the source model.

Since `nncf-group-size-not-divisible` shipped, this is classified as
`export.quantization.group_size_mismatch` (exit 70) and says what to do:

```text
✗ This model cannot be quantized with the profile's group size

Group-wise INT4 quantization splits every weight row into groups of a fixed size, so the group size
has to divide each layer's channel size. NNCF listed the layers where it does not: their channel
size is smaller than, or not a multiple of, the group size this profile asks for. The installation,
the devices and the source model are all fine; the conversion setting is what does not fit this
model. Nothing had been compiled when this failed, so no device is implicated.

  - Re-run with the channel-wise profile, which has no divisibility requirement: yonga prepare MODEL
    --profile npu-int4-sym-cw (group size -1, one scale per output channel).
  - Or pick a profile whose group size divides every channel size in the error; the shipped
    group-wise profiles all use 128.
  - The failing nodes and their channel sizes are in the job log: yonga logs JOB_ID --raw
```

The channel-wise profile is verified to work on the same model:

```console
$ yonga prepare hf-internal-testing/tiny-random-LlamaForCausalLM --device cpu \
      --profile npu-int4-sym-cw --yes
7/7 Artifact ready at <data>/artifacts/hf-internal-testing--tiny-random-LlamaForCausalLM/…
```

`npu-int4-sym-cw` is named for the device it was written for, but a profile can be requested for any
device: `prepare` uses an explicitly requested profile whatever device it declares, and names the
mismatch in the plan before anything is downloaded.

```text
Known limitations
• Profile npu-int4-sym-cw targets NPU, not the requested GPU.
```

### 5.5 Gated or private repositories

Metadata for a gated repository is usually public, so `yonga inspect` works and reports
`Access  gated`. Downloading is what fails. The shipped diagnostics:

| Code | When | Message |
|---|---|---|
| `hub.auth.gated` | the Hub says the repo is gated | "… is gated: its owner requires you to accept conditions and authenticate before the files can be read." |
| `hub.auth.required` | HTTP 401/403 | "The Hugging Face Hub refused access to … The repository is private, gated, or the stored credential is not valid for it." |
| `hub.repo.not_found` | 404, or 401 on an unknown repo | "… does not exist on the Hugging Face Hub, is disabled, or is not visible to this account." |
| `hub.offline.missing` | `--offline` and the files are not cached | "Offline mode is active and … is not fully present in the local Hugging Face cache." |

Remediation in all the auth cases:

```text
Open the model page in a browser and accept its licence/access conditions first.
Authenticate with `hf auth login`, then re-run the command. A token exported as HF_TOKEN works too.
```

An anonymous request for a repository that does not exist answers **401, not 404** — the Hub refuses
to confirm whether a private repository exists. That is why `hub.repo.not_found` covers both
readings in one message. Check `yonga doctor`'s last section for what credential, if any, is in use:

```text
Hugging Face
·  Authentication  anonymous  Public repositories work; gated or private ones need `hf auth login`.
```

`yonga` never prints a token. Job logs and diagnostics run through a redactor that replaces anything
token-shaped with `***REDACTED***`.

### 5.6 The architecture's Transformers pin is not met

Exporting runs the architecture's modelling code from `transformers`, so the registry pins a release
per architecture. When the installed one does not satisfy the pin, `prepare` stops in preflight —
before the download, not after it — with `preflight.exporter_requirement_unsatisfied`, category
`package_abi_mismatch`, exit **10**. The plan says so before you are asked to confirm anything:

```text
Known limitations
• Exporter requirement not satisfied: transformers ==4.99.* is required, but 5.2.0 is installed

BLOCKED — exporter requirements not satisfied
x  transformers ==4.99.* is required, but 5.2.0 is installed
Preparation stops at preflight, before the download and the conversion. Override with
--ignore-exporter-requirements if you know the pin is stale; --yes does not.
```

Exit 10 rather than 30 is deliberate: nothing is short of a resource, and neither the model nor the
registry is wrong. What cannot work is the combination of installed versions, which is what
`package_abi_mismatch` means everywhere else in this document.

`--yes` does not answer this one. `--ignore-exporter-requirements` does, and the override is written
into the artifact's manifest so an artifact built past a declared pin can be explained later. The
usual answer is a dedicated virtual environment for that export — see `docs/INSTALL.md` section 6.

## 6. Reading a job log

Every `prepare` run gets a job id and a JSON Lines log under the data directory:

```
~/.local/share/yonga/logs/<job-id>.jsonl
```

The failure message always ends with its path. To read it:

```bash
yonga logs                    # list the logs, newest first
yonga logs JOB_ID             # the records as a table
yonga logs JOB_ID --raw       # the JSON Lines file verbatim
yonga logs JOB_ID --json      # the records as one JSON document
```

A table view looks like this (verified):

```text
Job 43984530-dfae-45d7-bf78-1c95727270bf
  17:50:04.818  info    job.created            job created
  17:50:05.120  info    job.stage              stage -> inspected
  17:50:05.130  info    preflight.temp         conversion temp redirected to managed storage
  17:50:05.130  info    hub.download.start     downloading hf-internal-testing/tiny-random-…
  17:50:05.299  info    download.complete      source snapshot ready
  17:50:05.303  info    export.command
  17:50:07.574  debug   subprocess.output      `torch_dtype` is deprecated! Use `dtype` instead!
  …
  17:50:13.560  error   export.failed          conversion failed
```

Useful events: `job.stage` (where the job got to), `export.command` (the exact argv, redacted),
`subprocess.output` (every line the exporter printed), `export.failed` (the tail that was
classified, in its `raw` field), `preflight.*` (disk and requirement decisions).

For a single failure, the fastest route is:

```bash
yonga logs JOB_ID --raw | grep -A5 export.failed
```

`--debug` on the original command prints the same raw text to the terminal.

## 7. What a failed job leaves behind

Two working directories belong to a job rather than to an artifact, both keyed by job id:

```
~/.local/share/yonga/artifacts/.staging/<job-id>/model   # the export under construction
~/.cache/yonga/tmp/<job-id>                              # the managed TMPDIR the tooling gets
```

Conversion happens in the staging directory; a successful run renames the finished model out of it
onto `artifacts/<repo>/<revision>--<profile>` and removes what is left. Nothing outside the staging
directory is ever half-written, which is why an interrupted `prepare` cannot corrupt an artifact you
already have.

A **failed** job keeps its staging directory when it holds a partial export worth reading, and the
failure says where, as a last remediation line:

```text
  - Partial output was kept for diagnosis at ~/.local/share/yonga/artifacts/.staging/<job-id>
```

The same path is in the diagnostic's `staging_dir` context key, in the job state document
(`logs/<job-id>.state.json`) and in the job log as `job.staging_retained`. Inside it, `model/` holds
whatever the exporter produced before it died: which IR files exist, their sizes, and whether the
tokenizer pair was written at all. The managed `TMPDIR` is kept on failure for the same reason and
logged as `job.temp_retained`.

A staging directory holding nothing but the incomplete marker is removed straight away — it is the
residue of an export that never wrote a file, and keeping those is how `.staging` used to grow to
gigabytes. Whatever survives is retired later by a bounded sweep that runs when the next job opens:
`workspace_retention_hours` (72 by default) and `workspace_retention_max` (3) both apply, and a
directory belonging to a job that is still running is never touched. Set either to `0` in the
configuration file to retire stale directories as soon as a job opens.

Neither root is inside the Hugging Face cache, and nothing here is ever deleted from it.

## 8. Normal things that look like failures

**A closed pipe.** `yonga logs JOB --raw | head -2` stops yonga's output the moment `head` has its
two lines. Every command treats that as the normal end of output: nothing is printed, no diagnostic
is rendered and the exit code is **0**, whether the output was a raw stream or a rendered table.
Quitting `less` early, or any reader that stops before yonga has finished, is the same thing, and
`set -o pipefail` is safe around it. A traceback, an `Exception ignored while flushing sys.stdout`,
or exit 99, 120 or 1 from a pipeline like this is a bug worth reporting.

**`registry validate` passing.** It checks the registry's *structure* and internal consistency:
every rule parses, schema versions are supported, version specifiers are parseable, profile,
chat-profile and repair references resolve, failure categories are known, repair implementation
names exist in the code, and no two rules could win the same match at the same priority and
specificity. That is the whole of it.

It does **not** check that the values are ones OpenVINO accepts. An override declaring

```yaml
    device: MARS
```

validates cleanly, and so would a weight format no exporter has ever heard of. This is deliberate:
device names and weight formats are upstream vocabulary, so an override targeting a device a future
OpenVINO release adds needs no change to this code. The price is that "valid" here means "this file
is well-formed and self-consistent", never "these values will work on your machine". The thing that
finds out whether they work is `yonga prepare`, which asks OpenVINO.

## 9. When nothing matches

`runtime.unknown_error` means the failure is new. The raw text is in the diagnostic and in the job
log. A good bug report contains:

- the command, with `--debug`;
- `yonga doctor` output (it carries no secrets);
- the job log for the failed job;
- the model repository id and revision.

If the failure is recognisable and repeatable, it belongs in
`src/yonga/registry/builtin/issues.yaml` as a new rule with a `sources` entry recording how it was
reproduced. That is the whole point of the registry: one narrow, traceable rule per real failure,
instead of a conditional somewhere in the pipeline.
