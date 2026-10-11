# Troubleshooting

When something goes wrong, yonga tells you what failed, why, and what to run next, and it keeps the
raw error text in a job log. This page shows how to read that output and walks through the problems
users actually meet, each with the diagnostic code you will see and the fix. The full reference,
with every code the shipped registry can produce, is
[TROUBLESHOOTING.md](https://github.com/ooguz/yonga/blob/main/docs/TROUBLESHOOTING.md) on GitHub.

## Start with `yonga doctor` { #doctor }

Whatever the problem, run this first:

```bash
yonga doctor
```

`doctor` checks the host, the OpenVINO packages, the devices, NPU access, storage and Hugging Face
authentication, and changes nothing. Each line is marked `✓` (fine), `!` (warning), `✗` (error) or
`·` (information), and a problem comes with `→` lines that say what to do. This is the NPU
section of a healthy machine, shortened:

```text
NPU
✓  /dev/accel/accel0   accessible  group render
✓  Group               render  the current user is a member
✓  driver              1789049548
✓  compiler            524291
✓  compiler-in-plugin  available  the NPU plugin compiles graphs itself
…
```

Its exit code sums it up: **0** everything is usable, **2** usable with warnings, **10** something
yonga needs is missing. `yonga doctor --full` also reports the loaded registry and checks the
exporter import again if it failed before.

## Reading a diagnostic { #diagnostic }

A failure is printed as a diagnostic. This one comes from `inspect` with `--offline` for a model
that is not in the local cache:

```text
✗ Not available offline

Offline mode is active and Qwen/Qwen3-0.6B is not in the local Hugging Face cache
(~/.cache/huggingface/hub), so it cannot be used without network access.

  - Re-run without offline mode (no --offline, and YONGA_OFFLINE / HF_HUB_OFFLINE unset) to
populate the cache: …
  - Or point HF_HOME at a cache that already contains this revision.
```

Read it from the top:

- **The title** says what failed, and **the message** explains it in plain words.
- **The `-` steps** are the remediation, in order. A step that names a command gives you a real
  one: your own command with the change it suggests, ready to copy.
- **`Recommended fallback`** and **`Run:`** appear when the problem is one device, and name the
  device and the command to use instead.
- **`Job log:`** is the path of the log, for commands that run a job (`prepare`, `probe`, `bench`).
- **`Re-run with --debug to see the raw error`** appears when there is raw tool output behind the
  diagnostic, such as a compiler message.

The stable diagnostic **code**, such as `hub.offline.missing`, is in the `--json` output and in the
job log. [Exit codes](reference/exit-codes.md) maps codes to exit codes.

To look at the logs:

```bash
yonga logs                 # every job, newest first, with how it ended
yonga logs JOB_ID          # one job as a table; any unique prefix of the id works
yonga logs JOB_ID --raw    # the JSON Lines file as written
```

To see the raw error in the terminal, run the command again with `--debug`. It is a global option,
so it goes **before** the command name:

```bash
yonga --debug prepare Qwen/Qwen3-0.6B --device npu
```

## Common problems { #common }

### The NPU is missing or not accessible { #npu-access }

**What you see.** `doctor` marks `/dev/accel/accel0` as missing or `not accessible`.
`prepare --device npu` stops with `device.not_present` (exit 50) when OpenVINO reports no NPU at
all, and a command that cannot open the device fails with `npu.device.permission_denied` (exit 10).

**Why.** Three things must be true: the `intel_vpu` kernel module is loaded, the device node
`/dev/accel/accel0` exists, and your user may open it. The node belongs to the `render` group, and
a normal account is not in it by default. The userspace NPU driver is Intel's Level Zero NPU
driver, a distribution package (`intel-level-zero-npu` on the Ubuntu reference machine);
`pip install openvino` does not install it.

**Fix.** To get permission, add yourself to the group, then log out and back in:

```bash
sudo usermod -aG render $USER
```

yonga never runs `sudo` and never changes permissions itself; it only prints the command. It also
does not move the job to another device on a permission error, because that would hide a host
problem you want fixed. [Installation](installation.md) covers the NPU setup step by step.

### The accelerator driver does not initialise { #driver }

**What you see.** `device.driver_missing` (exit 10), often from an error such as
`ZE_RESULT_ERROR_UNINITIALIZED`.

**Fix.** Run `yonga doctor`: it shows the driver, the compiler and the kernel module status. Like a
permission problem, this is a broken host, so `prepare` stops instead of falling back.

!!! note
    `ZE_RESULT_ERROR_INVALID_ARGUMENT` in an NPU compile error is **not** a driver problem: the
    driver answered and the compiler ran. See [The NPU refuses the model](#npu-compiler).

### `doctor` warns about a small `/tmp` { #tmp }

**What you see.** `doctor` exits 2, with a warning under Storage:

```text
!  system temp          15.4 GiB tmpfs  7.0 GiB free at /tmp
                        → yonga runs conversions in a managed, disk-backed
                        temporary directory instead of the system temp
                        filesystem, so a small tmpfs is a warning and never
                        fatal.
…
```

**Fix.** None is needed, and you should not enlarge the tmpfs for yonga. On Ubuntu `/tmp` lives in
RAM, so yonga converts in its own directory under the cache (`~/.cache/yonga/tmp/<job-id>`)
instead. What matters is free space on the disk that holds `~/.cache/yonga`.

### Not enough disk space { #disk }

**What you see.** `prepare` checks the space it needs before downloading anything, and refuses with
`preflight.insufficient_disk` (exit 30):

```text
✗ Not enough temporary space

The managed temporary directory <cache>/tmp/<job-id> has 15.1 GiB free, but this conversion needs
25.0 GiB (20.0 GiB plus a 5.0 GiB safety margin).

  - Free space on the cache filesystem.
  - Or point YONGA_CACHE_DIR at a larger filesystem.
```

If a disk fills up during a conversion anyway, you get `storage.insufficient_disk` or
`disk.insufficient_space` (exit 30). OpenVINO often reports a full disk as
`basic_ios::clear: iostream error`, which looks like a file format bug and is not.

**Fix.** Free space, or move yonga's directories to a bigger disk, for one command with the global
options `--cache-dir` and `--data-dir`, or for every command with:

```bash
export YONGA_CACHE_DIR=/mnt/big/yonga-cache    # temporary files, compile cache, export environments
export YONGA_DATA_DIR=/mnt/big/yonga-data      # prepared models, logs, benchmarks
```

The compile cache and old export environments are often what filled the disk; see
[the compile cache keeps growing](#compile-cache) and [Maintenance](guide/maintain.md).

### A gated or private model (401) { #gated }

**What you see.** `inspect` works, but `prepare` refuses with `hub.auth.gated` or
`hub.auth.required` (exit 21). If there is no token at all, `prepare` refuses in the plan, before
the download.

**Fix.** Open the model page on the Hugging Face Hub, accept its conditions, and give yonga a
token, either by logging in with `hf auth login` or by exporting it:

```bash
export HF_TOKEN=hf_...
```

`doctor`'s Hugging Face section shows whether yonga sees a credential. Note that the Hub answers an
anonymous request for a repository that does not exist with 401, not 404, so a mistyped name can
look like an access problem: `hub.repo.not_found` (exit 20) covers both readings.

### Offline, and the model is not in the cache { #offline }

**What you see.** With `--offline` (or `YONGA_OFFLINE` or `HF_HUB_OFFLINE` set), `inspect` or
`prepare` fails with `hub.offline.missing` (exit 20), as in the example under
[Reading a diagnostic](#diagnostic).

**Fix.** Offline mode never contacts the Hub, so the model must already be in the local Hugging
Face cache. Run the command once without `--offline` to download it, or point `HF_HOME` at a cache
that has it. The first conversion of a model may also need a managed export environment, which is
built from the network; offline, a new one fails with `export.env.offline` (exit 20), and one that
already exists is reused.

### The NPU refuses the model { #npu-compiler }

**What you see.** A code that starts with `npu.compiler.` or `npu.pipeline.`, for example
`npu.compiler.unsupported_graph.qwen35` for a Qwen3.5 model. Its raw text looks like this:

```text
[ERROR] [vpux-compiler] OneShotBufferizeVPU2VPUIP failed : Only supports FP16.
```

`probe`, `chat` and `bench` exit 51; `prepare --device npu` exits 50; `prepare --device auto`
explains the failure and asks before it moves on to the GPU or the CPU.

**Why.** The NPU compiler is the strictest of the three devices, and some graphs do not compile on
it. This is a limit of the model on this compiler, not a broken driver: `doctor` still shows the
NPU as healthy, and reinstalling the driver does not help.

**Fix.** Use the GPU (or the CPU). The diagnostic prints the exact command; for a model it is:

```bash
yonga prepare MODEL --device gpu
```

If you already have an artifact of this model that passed its probe on the GPU, run that artifact
there instead, for example `yonga chat ARTIFACT_ID --device gpu`.

!!! tip
    A probe on the NPU builds the full pipeline, which can take many minutes for a large model.
    `yonga probe ARTIFACT_ID --device npu --quick` skips the build, and `--timeout 600` stops it
    after 600 seconds. Either can leave the NPU `inconclusive` (exit 2), which is not a verdict:
    run the probe again without `--quick`, or with a longer `--timeout`, to settle it.

### The profile's group size does not fit the model { #group-size }

**What you see.** `export.quantization.group_size_mismatch` (exit 70). Usually `prepare` predicts
it from the model's `config.json` and stops at the plan, before downloading, with a title such as
`No GPU profile can quantize this model`.

**Why.** Group-wise INT4 splits each weight row into groups of 128, so 128 has to divide the
model's layer widths. Some models, such as `HuggingFaceTB/SmolLM2-360M-Instruct` with a hidden size
of 960, do not fit. Nothing is wrong with the model or the device.

**Fix.** Use the device or profile the message names. The profiles with no group size are
`cpu-int8-asym-ref` and `cpu-fp16-ref` on the CPU, and `npu-int4-sym-cw` on the NPU:

```bash
yonga prepare MODEL --device cpu --profile cpu-int8-asym-ref
yonga prepare MODEL --device npu --profile npu-int4-sym-cw
```

`yonga registry profiles` lists every profile with its device and weight format.

### phi3 or mistral models on Python 3.14 { #python314 }

**What you see.** `export.optimum.normalized_config` (exit 10), before the download, for model
types such as `phi3`, `mistral`, `mixtral`, `gpt2` and `bloom`.

**Why.** optimum 2.3.0 and earlier cannot build the export configuration of these architectures on
Python 3.14. The defect is in optimum, which has fixed it in a pull request that no release carries
yet. The AppImage bundles Python 3.14, so it is affected too.

**Fix.** Run yonga from Python 3.11, 3.12 or 3.13, with a pip install into a virtual environment:

```bash
python3.13 -m venv ~/.venvs/yonga
~/.venvs/yonga/bin/pip install "yonga[openvino,export]"
```

Or choose a model of another architecture; `yonga search QUERY` shows what the registry knows.

### torch and torchvision do not match (pip installs) { #torchvision }

**What you see.** `export.torchvision.build_mismatch` (exit 10) at preflight, before the download,
and an `exporter import` warning in `doctor`. The raw error is
`operator torchvision::nms does not exist`.

**Why.** torch came from PyTorch's CPU index and torchvision from PyPI. The two builds cannot load
together, so no model can be converted, although `pip check` reports nothing wrong.

**Fix.** In the environment yonga runs in, replace torchvision alone, then check the import:

```bash
pip uninstall -y torchvision && pip install torchvision --index-url https://download.pytorch.org/whl/cpu
yonga doctor --full
```

Avoid `pip install --force-reinstall` here: it reinstalls every dependency as well and can break
other version pins. yonga does not change your environment for you.

### The AppImage does not start (FUSE) { #fuse }

**What you see.** The AppImage fails to start with an error about FUSE, typically in a container
or on a minimal install.

**Fix.** Let it unpack itself to a temporary directory instead of mounting:

```bash
APPIMAGE_EXTRACT_AND_RUN=1 yonga doctor
```

`self install` works the same way with the variable set. To make it permanent, set it in your
shell's startup file.

### The compile cache keeps growing { #compile-cache }

**What you see.** Less free disk over time. Once the space you could reclaim reaches 10 GiB, or
25 % of the free space, `doctor` flags it under Storage.

**Why.** OpenVINO keeps one compiled blob per artifact, device and OpenVINO release, and NPU and
GPU blobs of large models are several GiB each. Blobs of removed artifacts (`orphaned`) and of
another OpenVINO release (`stale-runtime`) are never used again.

**Fix.** See what is there, preview a prune, then prune:

```bash
yonga cache list
yonga cache prune --dry-run
yonga cache prune
```

`--older-than DAYS`, `--unattributed` and `--all` remove more. If two yonga installations with
different OpenVINO releases share one cache, prune from the one you keep, or give them separate
`--cache-dir`s. `yonga env list` shows the managed export environments and their size.

## When nothing matches { #unknown }

`runtime.unknown_error` (exit 99) means the failure is new to yonga. Please open an issue on
[GitHub](https://github.com/ooguz/yonga/issues) with:

- the command, run again with `--debug`;
- the output of `yonga doctor`;
- the job log, packed as a bug-report bundle;
- the model's repository id and revision.

The bundle collects the log, the job state, the manifest and a host report into one archive in
the current directory, named `yonga-report-` followed by the first eight characters of the job id
(`--output` writes it elsewhere):

```bash
yonga logs JOB_ID --bundle
```

Tokens are redacted, and the home directory, host name, login name and Hugging Face account name
are anonymised, but read the bundle before you attach it. Report a security problem privately
instead, as [Privacy and security](privacy.md#vulnerabilities) describes.
