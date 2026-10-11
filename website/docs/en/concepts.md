# How yonga works

yonga turns a Hugging Face repository id into a prepared, validated OpenVINO model for one Intel
device on your machine, and it tells you why it made each choice. This page explains the ideas
behind that: the preparation pipeline, how a device and a quantization profile are chosen, what an
artifact is, where yonga's compatibility knowledge comes from, how failures are reported, and what
yonga deliberately never does. You do not need any of it to [get started](getting-started.md), but
it makes the output of `inspect` and `prepare` much easier to read.

## The preparation pipeline { #pipeline }

`yonga prepare MODEL` first prints a preparation plan and asks before anything expensive happens.
Once you confirm, it runs seven stages, and the progress lines are numbered the same way:

| Stage | What happens |
|---|---|
| 1/7 Inspecting source | Reads Hub metadata (no weights): formats, architecture, parameter count, license. Branches and tags are resolved to an immutable commit. |
| 2/7 Resolving compatibility | Matches registry rules for the architecture, chooses the device and the conversion profile, and predicts known repairs. |
| 3/7 Preflight | Checks disk, temp space, memory, authentication, remote code and exporter package versions. A job that cannot finish is refused before anything is downloaded. |
| 4/7 Downloading | Fetches only the files the conversion reads, at the pinned commit, into the normal Hugging Face cache. |
| 5/7 Converting | Exports and quantizes with optimum-intel and NNCF using the profile's settings, into a staging directory. |
| 6/7 Validating / repairing | Inventories the OpenVINO IR, reads the tokenizers back, and applies only narrow, registry-listed repairs. |
| 7/7 Probing runtime | Compiles the model on the target device and generates a few tokens as a smoke test. |

Only when every stage passes is the staging directory moved into place as a ready artifact, with a
manifest that records what was done. A failed job never leaves a half-finished artifact behind; at
most it keeps its staging directory for diagnosis and tells you where.

This is the plan header of a real `prepare` on the reference machine (trimmed):

```text
Requested model             Qwen/Qwen3-0.6B@c1899de289a04d12100db370d814…
Resolved source             Qwen/Qwen3-0.6B
Revision                    c1899de289a0
Relationship                original
Source format               safetensors
Architecture                qwen3  rule arch-qwen3
Task                        text-generation-with-past
Device decision             NPU  supported (tested)  NPU was requested
                            explicitly; yonga never switches device on
                            its own.
Profile                     npu-int4-sym-g128
                            NPU group-wise INT4; the NPU default below
                            5B parameters
Quantization                int4, group size 128, ratio 1, symmetric
Chosen because              751.6M parameters (Hugging Face Hub count
                            from safetensors headers) is below 5B:
                            npu-int4-sym-g128 is the NPU profile for
                            that size (evidence: tested).
Expected download           1.4 GiB  only the files the conversion reads
…
Expected repairs
• ov-tokenizer-precision-string-casing
```

The [model pipeline spec](https://github.com/ooguz/yonga/blob/main/docs/MODEL_PIPELINE.md) describes
each stage in detail. For the practical side, see [Preparing models](guide/prepare.md).

## Devices { #devices }

yonga works with the three devices OpenVINO reports on an Intel AI PC: the **CPU**, the integrated
**GPU** and the **NPU**. `prepare` takes `--device auto`, `npu`, `gpu` or `cpu`, and the default is
`auto`.

### Automatic selection { #auto-selection }

With `--device auto`, yonga ranks the devices OpenVINO reports on your host:

1. Devices the registry marks **unsupported** for this model are dropped.
2. The rest are ordered by how well the registry knows they work: `supported`, then
   `experimental`, then `unknown`.
3. Devices that are equally supported are taken in the default order **GPU, NPU, CPU**. The iGPU
   takes the widest range of graphs, the NPU is the pickiest, and the CPU is the last resort.
4. If the registry recommends a device for the architecture, backed by `official` or `tested`
   evidence, that device moves to the front.

The plan's "Device decision" row says which of these rules decided. Every family in the built-in
registry recommends the GPU, so `--device auto` usually prepares for the GPU even when the NPU
would also work. To target the NPU, ask for it.

If the chosen device then fails its probe in auto mode because the model is not compatible with
it, yonga explains why and asks before it tries the next device (`--yes` answers that question).
It does not fall back when the failure means the host itself is broken, such as no permission on
the NPU device node or a driver that does not initialise: it stops, because that needs fixing
first.

### An explicit device never falls back silently { #explicit-device }

`--device npu` (or `gpu`, or `cpu`) is an instruction, not a preference. If that device cannot
run the model, `prepare` stops and tells you what to run instead; it never quietly prepares for
another device. When the registry already knows, with official or tested evidence, that the
device cannot run the model, `prepare` refuses before the download. Either way the exit code is
50 (10 when the host is broken, as above).

```bash
yonga prepare Qwen/Qwen3-0.6B --device npu
```

An artifact is labelled only for the devices it actually passed a probe on, so a "GPU" artifact
in `yonga list` is one that compiled and generated on your GPU.

## Profiles { #profiles }

A **profile** is a named set of conversion settings: the weight format, symmetric or asymmetric
quantization, the group size, the ratio of layers that are compressed to 4 bits, and any runtime
properties for OpenVINO GenAI. The names spell out the main settings:

| Part of the name | Meaning |
|---|---|
| `npu-`, `gpu-`, `cpu-` | The device the profile is written for |
| `int4`, `int8`, `fp16` | Weight format |
| `sym`, `asym` | Symmetric or asymmetric quantization |
| `g128`, `cw` | Group-wise with groups of 128, or channel-wise |
| `r08` | Ratio 0.8: about a fifth of the layers stay at INT8 |
| `ref` | A reference profile for quality evaluation, never a default |

To see every profile `--profile` accepts, run:

```bash
yonga registry profiles
```

```text
Profile                 Device  Weights                   Size band  Prio
cpu-int4-asym-g128-r08  CPU     int4 asym g128 ratio 0.8  below 5B    110
cpu-int4-asym-g128      CPU     int4 asym g128            —           100
cpu-int8-asym-ref       CPU     int8 asym                 —            10
cpu-fp16-ref            CPU     fp16                      —             5
gpu-int4-asym-g128-r08  GPU     int4 asym g128 ratio 0.8  below 5B    110
gpu-int4-asym-g128      GPU     int4 asym g128            —           100
npu-int4-sym-g128       NPU     int4 sym g128             below 5B    110
npu-int4-sym-cw         NPU     int4 sym channel-wise     —           100
```

The real output also has a summary column; add `--device npu` to list one device only.

### Size bands and how a profile is chosen { #size-bands }

Each device has a default for models **below 5B parameters** and one for everything else,
including models whose size is unknown:

| Device | Below 5B parameters | 5B and above, or unknown size |
|---|---|---|
| CPU | `cpu-int4-asym-g128-r08` | `cpu-int4-asym-g128` |
| GPU | `gpu-int4-asym-g128-r08` | `gpu-int4-asym-g128` |
| NPU | `npu-int4-sym-g128` | `npu-int4-sym-cw` |

The parameter count is the one the Hugging Face Hub reports from the safetensors headers (`yonga
inspect` shows it), so it includes embeddings and any vision tower. The small-model defaults keep
more precision because, measured over twelve models on the reference machine, that lost far less
quality for 9-18 % more disk; [Tested hardware and models](hardware.md#measurements) has the
highlights.

yonga picks a profile in this order:

1. A profile you pass with `--profile` always wins. It is never swapped for another one; if its
   group size cannot fit the model, `prepare` refuses it before the download and names the
   profiles that do.
2. A profile named by a registry rule for this exact model or for its architecture.
3. Otherwise the highest-priority profile for the device whose size band matches.

Automatic selection also skips a group-wise profile whose group size cannot divide the model's
layer widths in `config.json`. The plan's "Chosen because" row always gives the reason in one
sentence, and the artifact's manifest keeps it.

## Artifacts { #artifacts }

An **artifact** is the prepared model: the OpenVINO IR, the tokenizers, the chat template and a
`.yonga/` directory with its manifest and validation records. Artifacts live under the data
directory, by default `~/.local/share/yonga/artifacts/`, in a folder named after the repository,
the source revision and the profile:

```text
~/.local/share/yonga/artifacts/Qwen--Qwen3-0.6B/c1899de289a0--npu-int4-sym-g128
```

- **Artifact id.** Each artifact has a 16-character id, such as `cc24636a2f49356a`, derived from
  the repository, the exact source commit and the profile. The same inputs always produce the
  same id. `prepare` takes a repository id; `chat`, `probe`, `bench`, `serve` and `remove` take
  the artifact id that `prepare` and `yonga list` print (or the artifact's path).
- **Revision pinning.** A branch or tag is resolved to a commit before anything is downloaded,
  and every file comes from that commit. Pass `--revision` to choose one.
- **Manifest.** `.yonga/manifest.json` records the requested and resolved source, the revision,
  the architecture rule, the export command and settings, every repair with file hashes before
  and after, the per-device probe results, the host and package versions, and the hash of every
  file. `yonga publish` builds its model card from it.
- **Reuse.** If a ready artifact already exists for the same revision and profile, `prepare`
  reuses it without converting again and without asking. Pass `--force` to prepare it again. A
  changed default profile produces a new artifact next to the old one, never a silent reuse.

Your source model stays in the Hugging Face cache. yonga reads it there and writes only to its
own directories. [Maintenance](guide/maintain.md) covers removing artifacts and pruning caches.

## The compatibility registry { #registry }

Knowledge about which architectures work on which device, with which exporter versions, which
failures are known, and which repairs are safe changes faster than yonga's code. It therefore
lives in the **compatibility registry**: YAML data, not code. The built-in rules ship with yonga
and describe architectures, profiles, known issues, repairs, declared derivative sources, chat
behaviour and GGUF import support.

Every device claim carries an **evidence level**:

| Evidence | Meaning |
|---|---|
| `official` | Documented by OpenVINO, in its documentation or on one of its published model cards |
| `tested` | Run on real hardware; the families on [Tested hardware and models](hardware.md) are `tested` because of those runs |
| `reported` | Weaker than `tested`, stronger than `inferred` |
| `inferred` | Expected from a related family; for example, a mistral model is caught by a generic rule that rates the CPU and GPU `supported (inferred)` and the NPU `unknown` |
| `unknown` | No reliable knowledge; the probe decides |

`yonga inspect MODEL` shows the per-device verdict with its evidence. To see exactly which rules
matched a model, which lost, and why, run:

```bash
yonga registry explain Qwen/Qwen3-0.6B
```

```text
Device compatibility
✓  CPU  supported (tested)  arch-qwen3
✓  GPU  supported (tested)  arch-qwen3
✓  NPU  supported (tested)  arch-qwen3
```

(This is the last section of a much longer output that lists the candidate rules for the
architecture, profile, chat profile and derivative sources.)

### Layers and signed remote bundles { #remote-bundles }

The registry is loaded in layers, each able to replace a rule of the one before by its id:

1. the built-in rules shipped with yonga;
2. an optional **signed remote bundle**, installed with `yonga registry update`;
3. your own overrides in `~/.config/yonga/registry-overrides/`.

`yonga registry paths` shows where each layer lives. yonga contacts no registry server unless you
tell it to: `registry update` needs `--url` (or a `registry_url` in the configuration) or a bundle
on disk with `--file`. A bundle is installed only if it is signed by a key you placed in
`~/.config/yonga/registry-keys/` yourself, has not expired, is not older than the one installed,
and contains only plain data. It can select a repair or a tool-call parser that is already built
into yonga, but it can never deliver code. A bundle that stops verifying is ignored, and `yonga
registry status` says why.

The [registry spec](https://github.com/ooguz/yonga/blob/main/docs/REGISTRY_SPEC.md) documents
every rule type and the resolution order.

## Diagnostics and exit codes { #diagnostics }

When something fails, yonga does not hand you a raw compiler traceback. The error text goes
through the registry's issue rules, and you get a diagnostic with:

- a stable **code**, such as `npu.compiler.unsupported_graph.qwen35`;
- a **category**, which separates "this model does not fit this device" from "your installation is
  broken";
- **remediation** steps with real commands you can copy;
- the path of the **job log**, which keeps the raw text. `yonga logs JOB_ID` shows it, and
  `--debug` adds the raw error text to the failure itself.

Every command ends with a stable exit code, grouped by what went wrong:

| Codes | Meaning |
|---|---|
| 0, 2 | Success; 2 means success with warnings (or an inconclusive probe) |
| 10 | The environment is unusable: a missing package, driver or permission |
| 20, 21, 22 | The model source could not be resolved, needs authentication, or needs remote-code approval |
| 30 | Not enough disk or memory |
| 40, 41, 42 | Conversion, validation, or a repair that was needed but disabled |
| 50, 51, 52 | The device is unsupported, compilation failed, or generation failed |
| 60 | Cancelled, or a confirmation was declined |
| 70 | A registry or configuration error, or a command line yonga cannot parse |
| 99 | An unexpected internal error |

[Exit codes](reference/exit-codes.md) lists each one, and [Troubleshooting](troubleshooting.md)
walks through the failures that have actually been seen.

## What yonga will not do { #boundaries }

Some limits are part of the design, and no flag turns them off:

- **No telemetry.** yonga reports nothing about how you use it, and it also keeps OpenVINO's own
  usage reporting from loading in every Python process it starts. See
  [Privacy and security](privacy.md).
- **No sudo, no system changes.** yonga never runs `sudo`, `apt`, `modprobe` or `usermod`, and
  never installs, upgrades or removes packages in the Python environment you installed it into.
  When something on the host needs changing, it prints the command for you to run. The only place
  it runs `pip` is inside a managed export environment under its own cache directory, after you
  confirm it in the plan.
- **No changes to your Hugging Face cache.** Source models are read from the cache and never
  patched in place. Repairs apply to yonga's own copy of the exported files.
- **No remote code unless you allow it.** A repository that can only be converted by running its
  own Python code needs `--trust-remote-code`. `--yes` never approves it; without the flag,
  `prepare` stops with exit 22.
- **No silent substitutions.** yonga never swaps in an unrelated model, never switches an
  explicitly requested device, and never replaces a profile you named. A declared derivative
  source is suggested and needs your confirmation.
