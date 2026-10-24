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
| 2 | success with warnings (this is what `doctor` returns on a tmpfs `/tmp`); also `probe` when no device passed or failed and one is inconclusive (section 5.1), and `self install` when the copy is in place but `yonga` on `PATH` does not run it yet (section 5.9) |
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
| 70 | registry/configuration error; also a command line yonga cannot parse (section 2) |
| 99 | unexpected internal error |

A closed stdout is **0**, not 99: see section 8.

A command line that cannot be parsed exits **70**, the code of an invalid option value: an unknown
command or option, a missing argument, a value of the wrong type (`--timeout abc`,
`--older-than abc`), or `yonga` with no command at all, which prints the command list. The parser
would exit 2, and 2 is reserved for warnings and an inconclusive probe, so a script that retries an
inconclusive `probe` with a longer `--timeout` never mistakes a typo for one. `--help` exits 0.

Codes 40, 42 and 52 say *where* a job died rather than *why* (conversion, repair, generation), so
they appear only when the category is `unknown_runtime_error`, and from `bench` when none of its
runs produced a measurement. Code 41 is the validation stage's
code in the same way, and also the category code of `unsupported_ir_type` and
`tokenizer_ir_incompatible`, so a classified tokenizer or IR-type failure exits 41 too. A conversion
that fails for a classified reason exits with that reason's code: the NNCF group-size failure in
section 5.4 exits **70**, not 40.

`prepare` is the exception for device failures. When it cannot use a device it exits **50**,
whatever the category: a `--device` that is not present or fails its probe, and an automatic run
that has tried every fallback, end the same way. A failure whose diagnostic marks the environment
as broken is different: no permission on the device node (`npu.device.permission_denied`,
`device.permission_denied`) or a driver that does not initialise (`device.driver_missing`).
`prepare` does not fall back from those, and exits **10**, because the host needs fixing before any
choice of device matters. The diagnostic still carries its own code and category, in the message
and in `--json` output, so the reason is not lost. `probe`, `chat` and `bench` exit with the
category's code, so the same NPU compiler refusal is 50 from `prepare` and 51 from `probe`, while a
denied NPU node is 10 from all of them.

## 3. Registry issue codes

The table lists every rule that ships; the command at the top of this document prints the same
set. The Exit column is the category's code, which is what `probe`, `chat` and `bench` return. A
`prepare` that fails on the device probe exits 50 instead, except for a broken environment such as
`npu.device.permission_denied`, which exits 10 there as well (section 2).

| Code | Rule id | Category | Exit | Means | Do this |
|---|---|---|---|---|---|
| `npu.compiler.unsupported_graph.qwen35` | `qwen35-npu-fp16-bufferize` | `compiler_unsupported_graph` | 51 | The NPU compiler refused a Qwen3.5 graph: an operation is implemented only for FP16 on the NPU. | Prepare for the GPU: `yonga prepare MODEL --device gpu`. The NPU is not broken. |
| `npu.compiler.dynamic_shape` | `npu-compiler-dynamic-shape` | `compiler_unsupported_graph` | 51 | A raw compile kept the IR's dynamic dimensions; the NPU compiler needs static shapes. The rule is `defer_to_pipeline`: yonga records the raw failure as `deferred` and lets the GenAI pipeline, which reshapes to `MAX_PROMPT_LEN`/`MIN_RESPONSE_LEN`, decide. You see this code only when that pipeline could not be built either. | Read the pipeline error in `yonga logs JOB_ID --raw`; if it names a compiler pass, the model is not NPU-compatible — use the GPU. |
| `npu.compiler.unsupported_graph` | `npu-compiler-unsupported-graph-generic` | `compiler_unsupported_graph` | 51 | The NPU compiler refused the graph, and no family-specific rule applies. | Use GPU or CPU; re-check after an OpenVINO/NPU compiler update. |
| `npu.pipeline.static_reshape_failed` | `npu-static-reshape-broadcast` | `compiler_unsupported_graph` | 51 | The GenAI NPU pipeline could not reshape the graph to static shapes (`Broadcast incorrect target shape`). Seen for a model GenAI built from a GGUF file. | Use the GPU or CPU; for the NPU, prepare the same model from its safetensors repository with an NPU profile and let the probe decide. |
| `npu.compiler.pass_failed` | `npu-compiler-pass-failed` | `compiler_unsupported_graph` | 51 | A named NPU compiler pass failed, and the compiler answered — so the driver is working. | Use the GPU; the failing pass is in `yonga logs JOB_ID --raw`. |
| `npu.compiler.unsupported_feature` | `npu-unsupported-feature` | `compiler_unsupported_graph` | 51 | The NPU driver refused the graph with `ZE_RESULT_ERROR_UNSUPPORTED_FEATURE` and no compiler message. Either it has no compiler library (Arch's `intel-npu-driver` ships without one; `doctor` reads `missing` on the NPU compiler line) or its compiler cannot take this IR. | Run `yonga doctor`: if the compiler is missing it names the package for your distribution (on Arch, copy `libopenvino_intel_npu_compiler.so` and `libopenvino_intel_npu_compiler_loader.so` from the matching linux-npu-driver release into `/usr/lib`); otherwise use the GPU. |
| `tokenizer.ir.string_precision` | `tokenizer-string-precision-read` | `tokenizer_ir_incompatible` | 41 | The exported tokenizer XML writes `precision="STRiNG"`, which OpenVINO cannot read. | `yonga prepare MODEL --force`; the repair is applied automatically during validation. |
| `npu.device.permission_denied` | `npu-permission-denied` | `permission_denied` | 10 | The kernel refused access to the NPU node. | Join the `render` group (see `docs/INSTALL.md` section 3). No device fallback is offered on purpose. |
| `runtime.out_of_memory` | `runtime-out-of-memory` | `out_of_memory` | 30 | The host or the accelerator ran out of memory. | Close other applications; choose stronger compression or a smaller group size. |
| `storage.insufficient_disk` | `storage-no-space-left` | `insufficient_disk` | 30 | A write failed because the filesystem is full (including OpenVINO's `basic_ios::clear: iostream error`). | Free space, or point `YONGA_CACHE_DIR` at a bigger disk. |
| `export.gguf.open_failed` | `gguf-reader-open-failed` | `permission_denied` | 10 | OpenVINO GenAI's GGUF reader could not open the file (`Failed to open ... with gguf_open`). It opens the file read-write, so a read-only file is refused, as is one that is not a GGUF file. | Check the data directory is writable (`yonga doctor`); if the download may be damaged, `hf cache verify REPO`. |
| `export.quantization.group_size_mismatch` | `nncf-group-size-not-divisible` | `configuration_error` | 70 | The profile's group size does not divide the model's channel sizes. Usually predicted from `config.json` and refused before the download (section 5.4). | A profile with no group size: `cpu-int8-asym-ref` on the CPU, `npu-int4-sym-cw` on the NPU. |
| `export.transformers.symbol_missing.qwen35` | `qwen35-exporter-transformers-symbol-mismatch` | `package_abi_mismatch` | 10 | The exporter's Qwen3.5 patcher imports `Qwen3_5DynamicCache` and the installed Transformers does not define it. Nothing is broken; the two releases disagree about the Qwen3.5 modelling code. | Install the release the registry pins: `pip install "transformers==5.2.*"`. `yonga inspect MODEL` prints the requirement and what is installed. |
| `export.transformers.out_of_range` | `exporter-transformers-version-range` | `package_abi_mismatch` | 10 | optimum-intel's export config for this architecture declares a Transformers range the installed release is outside. `prepare` normally reads that range first and builds a managed environment; you see this only when that was turned off. | Drop `--no-managed-env`, or install a release in the range the message names. |
| `export.optimum.normalized_config` | `optimum-normalized-config-allow-new` | `package_abi_mismatch` | 10 | optimum failed building the export configuration (`got multiple values for argument 'allow_new'`), before conversion began. An exporter defect on Python 3.14, seen for phi3 and mistral. For the model types the rule lists (phi3, mistral, mixtral, gpt2, bloom and others), `prepare` refuses before the download when the export would run on Python 3.14 with optimum 2.3.0 or older, in a managed environment or in yonga's own; `--ignore-exporter-requirements` overrides that for an optimum already patched. | Run yonga on Python 3.11-3.13. The AppImage bundles 3.14, so that means a pip install into a virtual environment on an older Python (`docs/INSTALL.md` section 2). |
| `export.transformers.symbol_missing` | `exporter-transformers-symbol-mismatch` | `package_abi_mismatch` | 10 | Generic form of the above: `optimum/exporters/openvino/model_patcher.py` looked for something the installed Transformers does not provide, for any architecture. | Install the Transformers release the architecture requires; the failing import is in `yonga logs JOB_ID --raw`. |
| `export.torchvision.build_mismatch` | `torchvision-torch-build-mismatch` | `package_abi_mismatch` | 10 | `operator torchvision::nms does not exist`: torchvision and torch come from different builds (PyPI's torchvision next to a CPU torch), so nothing that imports torchvision loads and no model converts. `prepare` finds this before the download (section 5.7). | `pip uninstall -y torchvision && pip install torchvision --index-url https://download.pytorch.org/whl/cpu`, then `yonga doctor --full` to check the import. |

The last two are a backstop, not the usual path. When the registry declares the architecture's
Transformers pin, `prepare` exports in a managed environment that satisfies it, or with
`--no-managed-env` refuses at preflight before the download (section 5.6). Either way the export
never runs against the wrong Transformers. They classify what comes back when it runs anyway —
under `--ignore-exporter-requirements`, or for an architecture whose pin the registry does not know.

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
broken driver looks different: the NPU is absent from `openvino.Core().available_devices` (then
`prepare --device npu` stops with `device.not_present`, exit 50), or the error is
`ZE_RESULT_ERROR_UNINITIALIZED`, classified as `device.driver_missing`. That is a broken
environment: `prepare` exits 10 and does not try another device. A graph the compiler refuses is a
compatibility failure instead: `--device auto` moves on to GPU or CPU, and an explicit
`--device npu` exits 50 and names them.

This particular message is also not evidence about the *model*. A raw `Core.compile_model` keeps
an LLM's dynamic shapes, so it fails this way on the NPU for every stateful LLM — including ones the
NPU runs perfectly well. On this host Qwen3-0.6B (INT4 symmetric, group 128) fails the raw compile
with exactly this text and then loads and generates through `LLMPipeline` on the same NPU. So
`yonga` records it as `deferred` and builds the GenAI pipeline to decide. When that pipeline fails
too, its error is the verdict: for Qwen3.5 it is the one below, classified as
`npu.compiler.unsupported_graph.qwen35` with `environment_broken: false` and GPU/CPU offered as
fallbacks.

```text
[ERROR] [vpux-compiler] OneShotBufferizeVPU2VPUIP failed : Only supports FP16.
```

That line never appears in the Python exception — the NPU compiler writes it to the process's native
standard output. `yonga` captures both descriptors around every native compile so the
registry can see it; without the capture the same failure would read only
`[NPU_VCL] Compiler returned msg: Compilation failed`.

`yonga doctor` will still show the NPU as healthy, because it is:

```text
NPU
✓  /dev/accel/accel0   accessible  group render
✓  driver              1789049548
✓  compiler-in-plugin  available  the NPU plugin compiles graphs itself
```

Do not "fix" this by reinstalling the driver. A deferred raw compile needs nothing from you; a
failed pipeline means the graph is not NPU-compatible on this compiler, so run the model on the GPU.

Probing a large model on the NPU can take many minutes, because the pipeline build is a real NPU
compilation of the whole model. An instant answer used to come from the raw compile, and it was
wrong for every LLM. When you cannot wait, choose what you give up:

```bash
yonga probe ARTIFACT --device npu --quick         # no pipeline build: NPU is inconclusive, exit 2
yonga probe ARTIFACT --device npu --timeout 600   # pipeline built in a child, stopped after 600 s
```

`--quick` always reports the NPU as `inconclusive` (`probe.inconclusive.quick`, exit 2).
`--timeout` does too, but only when the build is stopped at the limit
(`probe.inconclusive.timeout`); a build that finishes in time gives the normal ok or failed
verdict. `inconclusive` is not a compiler verdict, and it does not mean the NPU is broken: the
pipeline that decides was never finished. A native compile cannot be interrupted inside a running
process, which is why `--timeout` builds it in a child it can stop. To settle it, run the probe
again without `--quick`, or with a longer `--timeout`.

The dynamic-shape message is one form of this. Another, seen on this host by loading a Qwen3-0.6B
artifact with `yonga chat ARTIFACT --device npu`, names the compiler pass instead:

```text
[ERROR] [vpux-compiler] MultiClusterStrategyAssignment Pass failed : Cannot get per cluster memory
shapes. Shape [1, 1, 8, 1], Unsupported distribution: #VPU.DistributedTensor<mode = SEGMENTED, …>
[ERROR] [vpux-compiler] Failed Pass MultiClusterStrategyAssignment on Operation
```

The reasoning is the same and so is the verdict: `npu.compiler.pass_failed`, exit 51 from `chat`
(50 from `prepare --device npu`; `--device auto` moves on to the GPU), GPU and CPU offered as
fallbacks, and no suggestion that anything is wrong with
the driver.

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

The OpenVINO compile cache shares that filesystem and is often what filled it: one entry per
artifact, device and OpenVINO version, several GiB each for NPU and GPU blobs, kept until something
removes them. `yonga cache list` shows what it holds and which entries are dead (`orphaned`: the
artifact is gone; `stale-runtime`: built by another OpenVINO release, never reused). `yonga cache
prune --dry-run` shows what a prune would free, `yonga cache prune` frees it, and `--older-than
DAYS`, `--unattributed` or `--all` go further. Stale-runtime is judged against the OpenVINO of the
yonga you run: when two installs with different OpenVINO releases share `~/.cache/yonga`, entries
the other install used after this release first built something are kept as `in-use`, but entries
it has not touched since then are pruned and it recompiles them on its next load. Run the prune
from the install whose runtime you keep, or give the installs separate `--cache-dir`s. `yonga doctor` flags the reclaimable part under
Storage once it reaches 10 GiB, or 25 % of the free space.

### 5.4 NNCF refuses the profile's group size

Symptom in older releases: `yonga prepare` exits 40 with nothing useful, and the job log ends with:

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
`export.quantization.group_size_mismatch` (exit 70), also when NNCF lists more refused nodes than
the 40 output lines yonga keeps (the exporter keeps the exception line in front of them), and says
what to do, with the real repository and job id in place of `MODEL` and `JOB_ID`:

```text
✗ This model cannot be quantized with the profile's group size

Group-wise INT4 quantization splits every weight row into groups of a fixed size, so the group size
has to divide each layer's channel size. NNCF listed the layers where it does not: their channel
size is smaller than, or not a multiple of, the group size this profile asks for. The installation,
the devices and the source model are all fine; the conversion setting is what does not fit this
model. Nothing had been compiled when this failed, so no device is implicated.

  - Re-run with a profile that has no group size. On the CPU that is the 8-bit reference: yonga
    prepare MODEL --device cpu --profile cpu-int8-asym-ref. On the NPU it is the channel-wise
    profile (group size -1, one scale per output channel): yonga prepare MODEL --device npu
    --profile npu-int4-sym-cw. Each is for its own device: npu-int4-sym-cw carries the NPU
    pipeline's MAX_PROMPT_LEN, which the CPU and GPU plugins reject when the artifact is probed.
  - When only a few small layers fail (Qwen3.5 0.8B-4B fail on one one-channel rotary_emb weight),
    the ratio-0.8 profile for the device may do, because it keeps a fifth of the weights at INT8:
    yonga prepare MODEL --device cpu --profile cpu-int4-asym-g128-r08 (or --device gpu --profile
    gpu-int4-asym-g128-r08). It prepared those models; that it skips the failing layer is
    observed, not guaranteed. When most layers fail, as they do when the hidden size is not a
    multiple of 128, it cannot help.
  - Or pick a profile whose group size divides every channel size in the error; the shipped
    group-wise profiles all use 128.
  - The failing nodes and their channel sizes are in the job log: yonga logs JOB_ID --raw
```

Most of the time it does not get that far. When `config.json`'s `hidden_size` is not a multiple of
the group size, the refusal is certain, so the plan passes over the group-wise profiles before
anything is downloaded and says why under "Chosen because" (`docs/REGISTRY_SPEC.md` section 5.2).
`HuggingFaceTB/SmolLM2-360M-Instruct` (hidden size 960) gets `cpu-int8-asym-ref` on the CPU and
`npu-int4-sym-cw` on the NPU. No GPU profile fits it, so `--device gpu` stops at the plan with exit
70, and `--device auto` moves on to the next device:

```text
✗ No GPU profile can quantize this model

Every GPU conversion profile quantizes in groups that do not fit this model's channel sizes, so
NNCF would refuse the export after the download: gpu-int4-asym-g128 cannot quantize this model: it
quantizes in groups of 128, which does not divide this model's hidden_size of 960; … Nothing has
been downloaded.

  - Prepare it for the CPU instead, where cpu-int8-asym-ref fits: yonga prepare
    HuggingFaceTB/SmolLM2-360M-Instruct --device cpu
  - Prepare it for the NPU instead, where npu-int4-sym-cw fits: yonga prepare
    HuggingFaceTB/SmolLM2-360M-Instruct --device npu
  - Or add a profile whose group size divides the channel sizes above, in the registry override
    directory, and name it with --profile.
```

The devices offered are the ones whose own automatic choice fits the model, worked out the same
way. A profile can be requested for another device than the one it declares (`prepare` uses it and
names the mismatch in the plan, `Profile npu-int4-sym-cw targets NPU, not the requested GPU.`), but
its runtime properties come with it, and the NPU profiles' `MAX_PROMPT_LEN` fails the probe on the
CPU (`Unsupported property MAX_PROMPT_LEN by CPU plugin`) and on the GPU (`Option not found:
MAX_PROMPT_LEN`). Verified with `Qwen/Qwen3-0.6B` on the reference host, 2026-10-11; that is why
neither message offers the NPU profile for another device.

A profile named with `--profile` is checked the same way and refused at the plan, before the
download, with the same code. yonga does not swap it for another: the remediation names the
profiles that fit, for the device being prepared and for the profile's own device, and repeats
your command with the first of them. Verified with `HuggingFaceTB/SmolLM2-360M-Instruct` on the
reference host, where auto selection picked the GPU (exit 70 after one second, nothing
downloaded):

```text
✗ Profile cpu-int4-asym-g128 cannot quantize this model

cpu-int4-asym-g128 quantizes in groups of 128, which does not divide this model's hidden_size of
960, so NNCF would refuse the export after the download. The profile was named with --profile, so
yonga does not pick another one in its place. Nothing has been downloaded.

  - No GPU profile can quantize this model.
  - Prepare it for the CPU with cpu-int8-asym-ref, the one yonga chooses for this model: yonga
    prepare HuggingFaceTB/SmolLM2-360M-Instruct --profile cpu-int8-asym-ref --yes --device cpu
  - Other CPU profiles that fit this model: cpu-fp16-ref.
  - 'yonga registry profiles' lists every profile with its device, weights and size band.
```

Only the widths `config.json` declares are checked. Qwen3.5 0.8B-4B fail `cpu-int4-asym-g128` on
a one-channel `rotary_emb` weight that no `config.json` key gives away (the 9B of the same
architecture prepares with it), so naming that profile for them still fails after the download;
the plan lists it under "Known limitations".

### 5.5 Gated or private repositories

Metadata for a gated repository is usually public, so `yonga inspect` works and reports
`Access  gated`. Downloading is what fails. The shipped diagnostics:

| Code | When | Message |
|---|---|---|
| `hub.auth.gated` | the Hub says the repo is gated | "… is gated: its owner requires you to accept conditions and authenticate before the files can be read." |
| `hub.auth.required` | HTTP 401/403 | "The Hugging Face Hub refused access to … The repository is private, gated, or the stored credential is not valid for it." |
| `hub.repo.not_found` | 404, or 401 on an unknown repo | "… does not exist on the Hugging Face Hub, is disabled, or is not visible to this account." |
| `hub.offline.missing` | `--offline` and the revision, or its weights, are not cached | "Offline mode is active and … is not in the local Hugging Face cache …" (or "… holds N file(s) of … but none of its model weights") |
| `hub.offline.search_unavailable` | `yonga --offline search` | "Offline mode is active, and searching the Hugging Face Hub is a network request …" |

Offline mode never contacts the Hub. `inspect` and `prepare` read a cached model from the local
Hugging Face cache instead: the revision comes from the cache's `refs/` mapping, or, when no revision
was asked for and none is recorded (yonga downloads by commit SHA, which records none), from the
only cached snapshot. Several unrecorded snapshots are refused rather than guessed between; name one
as `MODEL@<sha>`. Download counts, likes and gated status are Hub-only facts and are not shown.

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
per architecture. When the installed one does not satisfy the pin, `prepare` runs the export in a
**managed export environment** — an isolated virtual environment it builds under
`~/.cache/yonga/envs/` with the pinned release — and says so in the plan before you confirm:

```text
Export environment          managed export-68334e6a480b
                            transformers==5.2.*  will be built
```

Nothing in your own environment changes. See `docs/INSTALL.md` section 6 for what goes into it.

With `--no-managed-env` (or `managed_export_envs = false`) it stops in preflight instead — before the
download, not after it — with `preflight.exporter_requirement_unsatisfied`, category
`package_abi_mismatch`, exit **10**:

```text
BLOCKED — exporter requirements not satisfied
x  transformers ==4.99.* is required, but 5.2.0 is installed
Preparation stops at preflight, before the download and the conversion. Override with
--ignore-exporter-requirements if you know the pin is stale; --yes does not.
```

Exit 10 rather than 30 is deliberate: nothing is short of a resource, and neither the model nor the
registry is wrong. What cannot work is the combination of installed versions, which is what
`package_abi_mismatch` means everywhere else in this document.

`--ignore-exporter-requirements` exports in your environment regardless — it never builds a managed
one — and the override is written into the artifact's manifest so an artifact built past a declared
pin can be explained later. `--yes` does not imply it.

### 5.7 A managed export environment could not be built

| Code | Category | Exit | Meaning |
|---|---|---|---|
| `export.env.install_failed` | `download_failure` | 20 | `venv` or `pip` exited non-zero. pip's full output is in `yonga logs JOB_ID --raw`. |
| `export.env.verify_failed` | `package_abi_mismatch` | 10 | It installed, but a version does not match the specification, or the exporter would not import. |
| `export.env.offline` | `download_failure` | 20 | A new environment is needed and offline mode is on. An existing one is reused offline. |
| `export.env.insufficient_disk` | `insufficient_disk` | 30 | The cache filesystem cannot hold the environment *alongside* the export's temporary files. |
| `export.env.cancelled` | `user_cancelled` | 60 | Interrupted while building. |

A failed or interrupted build removes its directory, and an environment is only used once a marker
written after verification exists, so a half-built one is never picked up. Just re-run.

A real case, from the reference host. The first build failed verification with:

```text
RuntimeError: operator torchvision::nms does not exist
ModuleNotFoundError: Could not import module 'PreTrainedModel'.
```

`optimum-intel` requires `torchvision`, and pip had fetched it from PyPI — the CUDA build — while
torch had come from the CPU index. The two cannot load together. yonga now installs every
torch-built package (`torch`, `torchvision`) from the same CPU index.

The same error in an environment of your own (a pip install that took torch alone from the CPU
index, as `INSTALL.md` once advised) passes `pip check` and every version check, and used to fail
each export after the download. `prepare` now imports `optimum.exporters.openvino` there during
preflight, as the managed environments are verified, and stops before the download with
`export.torchvision.build_mismatch` (exit 10), or `export.toolchain_import_failed` for an import
failure no rule recognises. A successful import is remembered per interpreter and package
versions, so it costs a few seconds once. `yonga doctor` reports the same check as
`exporter import` in its OpenVINO section: it imports once when nothing is recorded for the
installed versions, then shows what the last check found, a failure with this remedy as a
warning; `yonga doctor --full` checks a recorded failure again.

Installing torchvision again from the CPU index is not enough: pip sees the PyPI build already
satisfies the requirement and keeps it. Do not reach for `--force-reinstall` either: it reinstalls
every dependency from the CPU index as well, which on the reference host upgraded numpy past the
`numpy<2.5.0` that nncf 3.4.0 requires. Replace torchvision alone:

```bash
pip uninstall -y torchvision
pip install torchvision --index-url https://download.pytorch.org/whl/cpu
```

`pip list` then shows the same `+cpu` suffix on `torch` and `torchvision`, and `yonga doctor --full`
reports `exporter import` as `imports`. Check it with yonga rather than with a plain
`python -c "import ..."`: yonga imports the exporter with OpenVINO's usage reporting switched off
(section 8), and a plain interpreter reports the import.

### 5.8 A GGUF repository cannot be imported

```text
✗ No GGUF file in this repository can be imported
The GenAI GGUF importer reads F32, F16, Q4_0, Q4_K, Q6_K, Q8_0 tensors. Every candidate file holds
a type it cannot convert: qwen2.5-0.5b-instruct-q4_k_m.gguf: holds Q5_0 tensors.
```

A GGUF file's name gives its *nominal* quantization. Its header lists what every tensor actually
is, and yonga reads that before downloading anything. `Q4_K_M` files often carry `Q5_0` or `Q5_K`
tensors, which the OpenVINO GenAI 2026.4 reader rejects (`[load_gguf] gguf_tensor_to_f16 failed`).
Name a different file with `--gguf-file`: `Q8_0` files hold only `Q8_0` and `F32` tensors, and
`Q4_0` files usually add a single `Q8_0`. `source.gguf_unsupported` means the architecture itself
cannot be imported, for example gemma3 or qwen35. There, a safetensors repository of the same
model is the route, declared as a derivative or named with `--source`.

`export.gguf.open_failed` (`Failed to open ... with gguf_open`) is the reader failing to open the
file at all. It opens the file read-write although it never writes to it, and huggingface-hub 1.x
keeps its cached files read-only, so yonga imports from its own writable copy in the job's staging
directory, never from the cache. If you still see it, the data directory is not writable by you
(`yonga doctor`), or the downloaded file is damaged (`hf cache verify REPO`).

### 5.9 `self install` and `self uninstall` refuse

`yonga self install` copies the running AppImage to `~/.local/bin/yonga` (another directory with
`--bin-dir DIR`); `docs/CLI_SPEC.md` section 12c has the whole contract. Its refusals leave
everything at the target exactly as it was.

| Code | Category | Exit | Means | Do this |
|---|---|---|---|---|
| `self.not_appimage` | `configuration_error` | 70 | This yonga does not run from an AppImage: a pip or pipx install, or one that merely inherited another application's `APPIMAGE`. | Use the environment's own `yonga` (activate it), or `pipx install 'yonga[openvino,export]'`; run `self install` from the downloaded AppImage. |
| `self.target.not_appimage` | `configuration_error` | 70 | Something other than a yonga AppImage is at `DIR/yonga`: a pip or pipx command, a symbolic link, a directory or another file. Never replaced or removed. | Remove it yourself if you want the AppImage there (`pipx uninstall yonga`), or install elsewhere with `--bin-dir DIR`. |
| `self.target.changed` | `configuration_error` | 70 | What is at `DIR/yonga` changed between the question and the rename; nothing was replaced. | Run the command again. |
| `self.source.changed` | `configuration_error` | 70 | The AppImage changed size while it was copied (a download still in progress). | Run it again once the file is complete. |
| `self.bin_dir.not_a_directory` | `configuration_error` | 70 | `--bin-dir` names something that exists and is not a directory. | Pass a directory; `~/.local/bin` is the default. |
| `self.bin_dir.not_writable` | `permission_denied` | 10 | The directory, or the nearest existing parent of a missing one, is not writable by you. yonga never uses sudo. | Choose a directory you own with `--bin-dir DIR`. |
| `self.insufficient_disk` | `insufficient_disk` | 30 | The copy does not fit on that filesystem. | Free space, or choose another directory. |
| `self.io.read`, `self.io.create`, `self.io.write`, `self.io.remove` | `permission_denied`, `insufficient_disk` or `unknown_runtime_error` | 10, 30 or 99 | Reading the AppImage or the target, creating the directory, writing the copy or removing it failed; the message names the file. | The remediation follows the error: a directory you own, more space, or `--debug` for the raw error. |
| `self.install.declined`, `self.uninstall.declined` | `user_cancelled` | 60 | The confirmation to replace or remove an AppImage was declined, or could not be asked (no terminal, or `--json` without `--yes`). | Re-run with `--yes`. |

`self install` exiting **2** is not a failure: the copy is in place, but `yonga` on `PATH` does not
run it yet, because `DIR` is not on `PATH` or an earlier `PATH` entry has another `yonga`. The
message says which, and the line to run. On Ubuntu and Debian, `~/.profile` adds `~/.local/bin` to
`PATH` at login only when the directory already exists then, so after the first install log out and
back in once, or run `export PATH="$HOME/.local/bin:$PATH"` in the current shell.

### 5.10 `connect opencode` refuses

`yonga connect opencode` edits OpenCode's config; `docs/CLI_SPEC.md` section 12d has the whole
contract. Every refusal below leaves the OpenCode file exactly as it was.

| Code | Category | Exit | Means | Do this |
|---|---|---|---|---|
| `connect.opencode.tools_unsupported` | `configuration_error` | 70 | OpenCode sends its tools with every request and this artifact's chat profile takes none. | Connect an artifact that does take tools (the message lists some). |
| `connect.opencode.prompt_too_short` | `configuration_error` | 70 | The artifact's fixed prompt length (every built-in NPU profile: 1024) is under the 8192 tokens OpenCode needs. | Drop a forced `--device npu` from a GPU or CPU artifact; or connect another GPU or CPU artifact of the same model, or `yonga prepare MODEL --device gpu`. |
| `connect.opencode.context_too_small` | `configuration_error` | 70 | `--context` is below the fixed prompt length plus the output limit (for example 8192 + 4096 on an NPU artifact). | Pass `--context` of at least that sum, or leave it out. |
| `connect.opencode.context_unknown` | `configuration_error` | 70 | The artifact has no `max_position_embeddings`; yonga does not invent a limit. | Pass `--context N`. |
| `connect.opencode.config_not_json` | `configuration_error` | 70 | The file has comments, trailing commas or a broken body, or `provider` is not an object. | Add the provider by hand (the JSON is printed) or use another file with `--config`; for `--remove`, delete the model entry it names by hand. |
| `connect.opencode.config_not_a_file` | `configuration_error` | 70 | `--config` names a directory. | Pass the file itself. |
| `connect.opencode.config_changed` | `configuration_error` | 70 | Something else edited the file between reading and writing; nothing was written. | Run the command again. |
| `connect.opencode.remove_options` | `configuration_error` | 70 | `--remove` was combined with `--host`, `--port`, `--device` or `--context`. | Give only `--config`, `--yes` and `--json` with `--remove`. |
| `connect.opencode.config_not_readable`, `connect.opencode.config_not_writable` | `permission_denied`, or `insufficient_disk` (exit 30) for `config_not_writable` on a full disk | 10 or 30 | The OS refused reading or writing the file or its directory, the file is read-only, or the disk is full. yonga never uses sudo. | Make a read-only file writable, choose a location you own with `--config FILE`, or free space. |
| `connect.opencode.declined` | `user_cancelled` | 60 | The question was declined, or could not be asked (no terminal, or `--json` without `--yes`). | Re-run with `--yes`. |

`connect opencode --remove` exiting **70** without a code is not a failure to remove: an entry you
edited after yonga wrote it is kept, listed with its reason, and the rest is removed (`--json`
says `kept` when nothing was removed). `not_connected` for a connection you made means it was made
in another file: the output names that file and the `--remove ... --config PATH` command.

## 6. Reading a job log

Every `prepare`, `probe` and `bench` run gets a job id and a JSON Lines log under the data directory:

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

The list says which job is which: the first eight characters of its id (any unique prefix is
accepted), when it started in local time, the command, the model or artifact, the device, how it
ended (`ok`, `failed`, `cancelled`, `running`, or `unfinished` for a job whose process died without
recording an end) and how long it took. On a narrow terminal the log size, device, duration and
start columns are dropped, in that order. `--json` has the full ids, the failure code and the
duration in seconds.

```text
job       started           command  model                                 device  status         took      log
61926a20  2026-10-11 02:32  bench    7ae557604adf--cpu-int4-asym-g128-r08  CPU     ! cancelled   6.6 s  1.0 KiB
238a287d  2026-10-11 02:32  bench    7ae557604adf--cpu-int4-asym-g128-r08  CPU     ✓ ok          1.9 s    665 B
aba94131  2026-10-11 02:32  probe    7ae557604adf--cpu-int4-asym-g128-r08  CPU     ✓ ok         710 ms    910 B
Start times are local, UTC+03:00.
```

One job's table view starts with the same facts, and every record's time is local (verified, a
`bench` interrupted with Ctrl-C):

```text
Job 61926a20-16b1-45e4-8392-5089d9f8c25e  (~/.local/share/yonga/logs/61926a20-….jsonl)
  bench ~/…/7ae557604adf--cpu-int4-asym-g128-r08 CPU  ! cancelled (job.cancelled), took 6.6 s
  Started 2026-10-11 02:32:54 UTC+03:00; the times below are local.
  02:32:54.866  info    job.created                job created
  02:33:01.510  warning job.cancelled              The job was interrupted before it finished.
  02:33:01.511  info    job.temp_discarded         temporary directory of a cancelled job removed
```

Useful events: `job.stage` (where the job got to), `export.command` (the exact argv, redacted),
`subprocess.output` (every line the exporter printed), `export.failed` (the tail that was
classified, in its `raw` field), `preflight.*` (disk and requirement decisions), `library.log`
(warnings `huggingface_hub` logged during the job, such as the Hub's advice to set `HF_TOKEN`,
kept here instead of breaking into the progress display).

Every job ends with one terminal event. `job.completed` is a job that ran to its end, `probe` and
`bench` included. `job.failed` (an error) and `job.cancelled` (a warning: Ctrl-C is not a defect)
carry the diagnostic's `code`, the last `stage` the job reached and, when it died working towards
the next one, `failed_before`: a conversion interrupted after the download reads `"stage":
"downloaded", "failed_before": "exported"`. Right before it, `job.error_detail` holds the raw text
the failure was classified from in its `raw` field — the NPU compiler's failing pass and the shape
it rejected, for example — so `yonga logs JOB_ID --raw` has the tool's own words whichever command
failed.

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
logged as `job.temp_retained`. A job you cancelled has nothing to diagnose, so its `TMPDIR` (the
conversion's intermediates, often a gigabyte) is removed on the way out and logged as
`job.temp_discarded`; its staging directory follows the same policy as a failure's, because a
finished export kept there is what the next `prepare` reuses.

When the export itself had finished, `model/.yonga/export-complete.json` is there too, and running
the same `prepare` again adopts the directory instead of exporting (`export.reused` in the new job's
log; `export.reuse_rejected` with the reason when a file no longer matches, after which it exports
normally). Pass `--fresh-export` to keep it out of the way. A cancelled or crashed job records a
terminal state on its way out (stage `cancelled` with a `job.cancelled` event, or stage `failed`
with code `job.aborted`), so it never looks like it is still running.

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

**`~/intel/stats` and OpenVINO's usage reporting.** OpenVINO counts each import of its conversion
API in `~/intel/stats` and, without a consent file in `~/intel`, reports it to Google Analytics
through the `openvino_telemetry` package. yonga keeps that package from loading in every Python
process it starts, including `optimum-cli` in a managed export environment, so nothing is sent and
the file is not touched; it does not write the opt-out file into your home directory either. A
count that still grows comes from another program that imports OpenVINO, such as a plain Python
session that runs `import openvino`. That is why yonga's advice never asks you to check an import
that way: `yonga doctor --full` checks the exporter's import with the reporting switched off.

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
