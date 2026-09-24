# INSTALL.md

How to get `yonga` running on Ubuntu, and the four environment facts that actually decide whether it
works: the NPU device node's group, the OpenVINO ABI family, where conversions get their scratch
space, and the Transformers version a given architecture was exported with.

Everything marked **verified** below was run on the reference host: Intel Core Ultra 7 268V (Lunar
Lake) with Arc iGPU and NPU, Ubuntu 26.04.1, Python 3.14.4, OpenVINO 2026.4.0. Everything else is
marked as untested.

## 1. Prerequisites

| | |
|---|---|
| OS | Ubuntu 24.04 LTS or 26.04 LTS, x86_64. **Verified on 26.04.1 only**; 24.04 is the second intended target and has not been tested. |
| CPU | Any x86_64 for CPU-only use. Intel Core Ultra (Meteor Lake or newer) for the NPU; Intel Arc / Xe for the GPU. |
| Python | 3.11-3.14 (`requires-python = ">=3.11,<3.15"`). **Verified on 3.14.4 only.** |
| Disk | Several times the model size. A 9B model needs roughly 18 GiB of source weights plus ~44 GiB of conversion scratch; `yonga prepare` computes the number and refuses early if it does not fit. |
| RAM | 8 GiB minimum for the tool to be useful at all; `doctor` warns below that. Conversion of a mid-sized model wants 16 GiB or more. |

System packages on a fresh Ubuntu install:

```bash
sudo apt install python3-venv python3-pip
```

Nothing else is required for CPU use. For the GPU you need Intel's compute runtime
(`intel-opencl-icd` on Ubuntu, or Intel's own repository for a newer one); for the NPU see
section 3.

## 2. Install yonga

`yonga` is **not published on PyPI yet**, so `pip install yonga` does not work. Install from a clone
of this repository, into a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[openvino,export]"
```

On a machine without an NVIDIA GPU, install the CPU build of torch first. The `export` extra pulls
torch in, and the default PyPI build on Linux carries CUDA — on the reference host 1.2 GB of torch
plus 3.2 GB of NVIDIA libraries that an Intel AI PC never uses. The CPU wheel is 196 MB:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[openvino,export]"
```

The extras map to what you intend to do:

| Extra | Packages | Needed for |
|---|---|---|
| `openvino` | `openvino`, `openvino-genai`, `openvino-tokenizers` | `probe`, `chat`, `bench`, and the device/NPU sections of `doctor` |
| `export` | `optimum-intel[openvino]`, `nncf` (pulling in `transformers` and `torch`) | `prepare` (download, convert, quantize) |
| `dev` | pytest, pytest-cov, pytest-mock, pytest-timeout, ruff, mypy | running the test suite |
| `all` | all of the above | |

Without any extras the package still installs and runs. `inspect`, `list`, `logs`, `registry` and
`doctor` work; `doctor` exits **10** and names the missing package and the extra that provides it
(verified):

```text
OpenVINO
✗  Runtime          not installed  openvino was not found in this environment.
                    → Install the runtime extras: pip install 'yonga[openvino]'
                    → yonga never installs, upgrades or removes a package: run the command yourself
                    in the environment yonga runs from.
```

That last line is a rule, not a disclaimer. `yonga` never runs `apt`, `modprobe`, `sudo` or
`usermod`, and never runs `pip` in the environment you installed it into. It tells you the command;
you decide whether to run it. The one place it does run `pip` is inside a *managed export
environment* it creates under its own cache directory, and only after you confirm it in the
preparation plan (section 6).

Check the installation:

```bash
yonga doctor
```

Exit code 0 means everything is usable, 2 means usable with warnings (a small `/tmp` alone produces
this — see section 5), 10 means something is missing that `yonga` cannot work around.

## 3. The Intel NPU on Linux

Three things have to be true before OpenVINO can see the NPU. `yonga doctor` reports all three, and
none of them needs `sudo` to *diagnose*.

**a. The `intel_vpu` kernel module is loaded.** It is part of the mainline kernel tree
(`drivers/accel/ivpu`) and is shipped by the distribution kernel, not by OpenVINO. On the reference
host (kernel 7.2.6) it is present as
`/lib/modules/$(uname -r)/kernel/drivers/accel/ivpu/intel_vpu.ko.zst` and loaded:

```bash
$ lsmod | grep intel_vpu
intel_vpu             348160  1
```

The userspace side is Intel's Level Zero NPU driver, a distribution package rather than a Python
one. On the reference host:

```bash
$ dpkg -l | grep intel-level-zero-npu
ii  intel-level-zero-npu  1.38.0.20260910-34487311128~ubuntu26.04
```

`pip install openvino` does **not** install it. Without it OpenVINO enumerates no NPU, and
`yonga doctor` reports the device as absent rather than as a model problem.

If it is absent, `doctor` reports the node as missing and says:

```text
Load the NPU kernel driver (the intel_vpu module) or install the intel-npu driver package for your
distribution.
yonga changes nothing about the kernel: run any module or package command yourself.
```

**b. The device node `/dev/accel/accel0` exists.** The kernel module creates it. `/sys/class/accel/accel0`
appears with it.

**c. Your user can open it.** This is the one that bites. The node is `root:render`, mode 660
(verified):

```bash
$ ls -l /dev/accel/accel0
crw-rw----+ 1 root render 261, 0 Sep 21 20:32 /dev/accel/accel0
```

Mode 660 with group `render` means only root and members of `render` can use the NPU. A normal user
account is not in `render` by default. When you are not a member, `yonga doctor` says exactly this
(the two warnings come from the shipped diagnosis; the reference host's user *is* in `render`, so
this output was produced from the same code path with a non-member user):

```text
NPU
!  /dev/accel/accel0  not accessible  /dev/accel/accel0 is owned by group 'render' and the current
                      user is not a member; read/write access is denied.
                      → add your user to the 'render' group, then log out and back in for the new
                      membership to take effect
                      → Suggested command, to run yourself: sudo usermod -aG render $USER
                      → yonga never runs sudo and never changes permissions or group membership.
!  Group              render  the current user is not a member of 'render'
```

**The `usermod` line is a command for you to run, not something `yonga` runs.** `yonga` has no code
path that invokes `sudo`, changes a file mode or edits group membership, and it never will:

```bash
sudo usermod -aG render $USER   # you run this, once
```

Then **log out and log back in** — a new group membership does not apply to an already-running
session. `newgrp render` gives a single shell the membership if you want to test without logging
out.

A working NPU looks like this (verified):

```text
NPU
✓  /dev/accel/accel0   accessible  group render
✓  Group               render  the current user is a member
✓  driver              1789049548
✓  compiler            524291
✓  compiler-in-plugin  available  the NPU plugin compiles graphs itself
✓  compiler libraries  3 found
```

"compiler-in-plugin" means the NPU plugin compiles graphs itself instead of needing a separate
standalone compiler package, which is the normal arrangement on OpenVINO 2026.x.

Note what a healthy NPU does *not* promise: that your model will compile on it. An NPU that answers
every probe can still refuse a graph. That is a model/compiler limitation, not a driver problem, and
`yonga` classifies it as such — see `docs/TROUBLESHOOTING.md`.

## 4. The OpenVINO ABI-family rule

`openvino`, `openvino-genai` and `openvino-tokenizers` are three packages containing native
libraries that link against each other. **They must come from the same `major.minor` release
family.** Mixing them produces `undefined symbol` import errors, or a segfault, rather than a polite
message.

Verified, aligned:

```text
✓  ABI family  aligned (2026.4)  openvino 2026.4.0, openvino-genai 2026.4.0.0,
                                 openvino-tokenizers 2026.4.0.0
```

Note the differing version *strings*: `2026.4.0` and `2026.4.0.0` are the same family. What matters
is `2026.4`. When they do not match, `doctor` warns and prints the pins for you to install
yourself:

```text
!  ABI family  mismatch  openvino 2026.4.0, openvino-genai 2026.2.0.0,
                         openvino-tokenizers 2026.4.0.0
   → openvino, openvino-genai and openvino-tokenizers must come from one release family; these do
     not: openvino 2026.4.0, openvino-genai 2026.2.0.0, openvino-tokenizers 2026.4.0.0.
   → Install one family yourself, for example: pip install 'openvino==2026.4.*'
     'openvino-genai==2026.4.*' 'openvino-tokenizers==2026.4.*'
   → yonga never installs, upgrades or removes a package: run the command yourself in the
     environment yonga runs from.
```

(That block is the shipped check rendered with a deliberately mismatched set; the reference host is
aligned.)

The usual way to get into this state is upgrading one of the three on its own. Upgrade them
together, and pin all three in any requirements file you keep.

## 5. `/tmp`, tmpfs, and why yonga does not ask you to enlarge it

On Ubuntu with systemd, `/tmp` is a **tmpfs**: it lives in RAM and is limited to roughly half of it.
On the reference host that is 15.4 GiB. Model conversion writes tens of gigabytes of intermediate
tensors, so a conversion that honoured `TMPDIR` would fill RAM and die with
`basic_ios::clear: iostream error` — an I/O error that looks like a serialisation bug and is really
a full filesystem.

`yonga` does not use `/tmp` for conversions. It creates a managed, disk-backed temporary directory
under the cache directory (`~/.cache/yonga/tmp/<job-id>`), points the exporter's `TMPDIR` at it, and
removes it when the job ends. So a small `/tmp` is reported as a **warning, never an error**
(verified):

```text
Storage
✓  artifact filesystem  738.9 GiB free  ext4 at /
✓  cache filesystem     738.9 GiB free  ext4 at /
!  system temp          15.4 GiB tmpfs  15.1 GiB free at /tmp
                        → yonga runs conversions in a managed, disk-backed temporary directory
                        instead of the system temp filesystem, so a small tmpfs is a warning and
                        never fatal.
                        → /tmp is RAM-backed and smaller than the 20.0 GiB a conversion can need, so
                        yonga does not use it.
                        → No action is required. Do not enlarge the tmpfs for yonga's sake; other
                        tools that honour TMPDIR may still need it.
✓  managed temp         disk-backed (ext4)  /home/m3rcury/.cache/yonga/tmp
```

This warning is why `doctor` exits 2 rather than 0 on an otherwise perfect host.

What you *do* need is free space on the filesystem holding `~/.cache/yonga`. To put the cache
somewhere else:

```bash
export YONGA_CACHE_DIR=/mnt/big/yonga-cache     # temp + compile cache
export YONGA_DATA_DIR=/mnt/big/yonga-data       # artifacts, logs, benchmarks
```

or pass `--cache-dir` / `--data-dir` per command. `yonga registry paths` prints every managed
directory:

```text
  config_dir               ~/.config/yonga
  cache_dir                ~/.cache/yonga
  data_dir                 ~/.local/share/yonga
  config_file              ~/.config/yonga/config.toml
  registry_overrides_dir   ~/.config/yonga/registry-overrides
  tmp_dir                  ~/.cache/yonga/tmp
  compile_cache_dir        ~/.cache/yonga/compile
  metadata_cache_dir       ~/.cache/yonga/metadata
  artifacts_dir            ~/.local/share/yonga/artifacts
  staging_dir              ~/.local/share/yonga/artifacts/.staging
  benchmarks_dir           ~/.local/share/yonga/benchmarks
  logs_dir                 ~/.local/share/yonga/logs
  chat_history_dir         ~/.local/share/yonga/chat-history
```

## 6. The Transformers version caveat

Exporting a model runs that architecture's modelling code from `transformers`. When the code changes
between releases, the exported graph changes with it, so **an architecture can require a specific
Transformers version** — not merely a minimum.

The registry records that requirement per architecture. Qwen3.5 is the shipped example: it declares
`transformers ==5.2.*`, which is why the reference host has 5.2.0 installed rather than the newest
release. `yonga inspect` states the requirement whether or not it is currently met (verified):

```text
Known limitations
  - Export needs Transformers 5.2; newer Transformers releases change the modelling code this
    architecture was exported with.
  - NPU is not a target for this architecture on OpenVINO 2026.x.
```

When the installed packages cannot meet that requirement, `prepare` runs the export in a **managed
export environment** instead of stopping: an isolated virtual environment under
`~/.cache/yonga/envs/`, used only by the export subprocess. Your own environment is never changed.
The plan says so before anything happens (real output, with `transformers 5.5.4` installed):

```text
Export environment          managed export-68334e6a480b
                            transformers==5.2.*  will be built
• The installed packages do not meet an exporter requirement (transformers
  ==5.2.* is required, but 5.5.4 is installed); the export runs in managed
  environment export-68334e6a480b instead.
Confirmation required
·  A managed export environment (transformers==5.2.*) will be built under
   ~/.cache/yonga/envs: up to about 3.0 GiB of packages from PyPI and the
   PyTorch CPU index. [...]
```

What goes into it:

- the packages yonga itself runs with — the same `optimum-intel`, `nncf` and `torch` — with only the
  failing pin changed, so the environment differs from a known-good one in exactly one place;
- `openvino` and `openvino-tokenizers` pinned to the **runtime's** release, because the runtime has
  to read the IR the export writes;
- `torch` and `torchvision` from PyTorch's CPU wheel index. Exporting runs on the CPU, and the
  default PyPI build of torch on Linux carries CUDA: 1.2 GB of torch plus 3.2 GB of NVIDIA libraries,
  against a 196 MB CPU wheel. The Qwen3.5 environment on the reference host is 1.44 GiB in total.

It is built at preflight, before the model download, so a failed build costs minutes rather than a
multi-gigabyte download. It is verified by importing the exporter inside it and checking every
version against the specification, and only then marked complete; an interrupted build is never
used. Later jobs with the same requirements reuse it. The manifest of every artifact exported in one
records its key. Registry data can only choose a *version* for a package yonga already lists — never
name a package of its own — so a registry override cannot make yonga install something else.

```bash
yonga env list               # what exists, and the disk it uses (doctor shows the total too)
yonga env remove KEY         # delete one; it is rebuilt if a later export needs it
```

Opting out restores the old refusal: `--no-managed-env`, or `managed_export_envs = false` in
`config.toml`. Preparation then stops at preflight and exits **10** without downloading anything,
naming the pin and the `pip install` command that would satisfy it. `--ignore-exporter-requirements`
means "export in my environment anyway": it never builds a managed environment, and the override is
recorded in the artifact's manifest. `--yes` approves building an environment, but never implies
`--ignore-exporter-requirements`.

Building an environment needs the network once; offline, a job that needs a new one stops with
`export.env.offline`, while an environment that already exists is reused.

## 7. Verifying the installation

```bash
yonga doctor                 # 0 ok, 2 warnings, 10 unusable
yonga registry validate      # the shipped compatibility rules load and are consistent
yonga inspect Qwen/Qwen3-0.6B
yonga prepare Qwen/Qwen3-0.6B --device cpu
yonga list
yonga chat ARTIFACT_ID --device cpu
```

`Qwen/Qwen3-0.6B` is a good first model: the whole path (download, INT4 conversion, tokenizer repair,
CPU compile, generation) finishes in minutes and the artifact is around 389 MiB. If something fails,
`docs/TROUBLESHOOTING.md` maps the diagnostic code to what it means.
