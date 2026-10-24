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

There are two ways: the AppImage, one file with nothing else to install, or `pip` into a virtual
environment.

### The AppImage

Every GitHub release carries `yonga-X.Y.Z-x86_64.AppImage` and its `.sha256`, where X.Y.Z is the
release's version. The image holds yonga, a relocatable CPython
([python-build-standalone](https://github.com/astral-sh/python-build-standalone)) and the `openvino`
and `serve` extras, and runs on x86_64 Linux with glibc 2.28 or newer (Ubuntu 20.04, Debian 11,
Fedora 29 and later). The build fails if any bundled library needs a newer glibc.

```bash
sha256sum --check yonga-X.Y.Z-x86_64.AppImage.sha256
chmod +x yonga-X.Y.Z-x86_64.AppImage
./yonga-X.Y.Z-x86_64.AppImage self install
yonga doctor
```

`self install` copies the running AppImage to `~/.local/bin/yonga` (another directory with
`--bin-dir DIR`): written to a temporary file in that directory, flushed, made `0755` and renamed
into place, so a shell never sees half a file. The downloaded file stays where it was and can be
deleted afterwards. What else it does, and does not do:

- **PATH.** It checks whether the directory is on `PATH` and whether `yonga` now runs the copy. If
  not, it prints what to do and exits 2. Ubuntu's and Debian's `~/.profile` add `~/.local/bin` to
  `PATH` at login, but only when the directory already exists then, so after the first install log
  out and back in once, or run `export PATH="$HOME/.local/bin:$PATH"` in the current shell. yonga
  never edits `~/.profile`, `~/.bashrc` or any other startup file.
- **Upgrades.** Run `self install` from the new AppImage. An older yonga AppImage at the target is
  replaced after a question (`--yes` answers it), showing both versions where they are known.
  Running the same file again says it is already installed.
- **What it never touches.** Anything at `~/.local/bin/yonga` that is not an AppImage (a pip or pipx
  command, a symbolic link, a directory, a script) is refused and left as it is. A directory you
  cannot write to is refused too: it never uses sudo, so choose one you own with `--bin-dir`.
- **Removing it.** `yonga self uninstall` removes the copy after a question. Prepared models, logs,
  caches and settings stay; it prints where they are.

`doctor` reports how yonga is running (the AppImage's path, or the Python environment) and warns
when an AppImage is not what `yonga` on `PATH` runs, naming the other installation when there is
one. A pip install has no `self install`: its `yonga` command lives in the environment's `bin`
directory (activate the environment, or use `pipx install 'yonga[openvino,export]'` for a command
in `~/.local/bin`).

- **FUSE.** The image mounts itself with FUSE. Without it (containers, some minimal installs),
  `APPIMAGE_EXTRACT_AND_RUN=1 yonga ...` unpacks it to a temporary directory instead (the
  `self install` above works the same way with it).
- **Drivers.** The Intel GPU and NPU drivers belong to the host (section 3 covers the NPU); an AppImage
  cannot carry kernel drivers or the compute runtime, and `doctor` reports what is missing.
- **Conversion.** The export toolchain (`optimum-intel`, `nncf`, `transformers`, `torch`) is not in
  the image: it is about 1.5 GB and the right Transformers depends on the model. The first `prepare`
  that converts builds a managed export environment for it (section 6), from a copy of the image's
  Python kept under `~/.cache/yonga/envs`, because the image's own mount point changes every time it
  runs. Prepared OpenVINO and GGUF models need no toolchain at all.

It is built by `tools/build_appimage.sh`. The interpreter, `appimagetool` and the AppImage runtime
are pinned and checked against their SHA-256. The Python packages are not pinned: pip resolves the
ranges in `pyproject.toml` against PyPI when the image is built, so two builds of one release can
bundle different versions. Each build therefore records what it bundled, as
`usr/share/yonga/packages.txt` inside the image and, from 0.3.0 on, as the release asset
`yonga-X.Y.Z-x86_64.AppImage.packages.txt`: the interpreter, the yonga wheel with its SHA-256, and
every package version.

### pip

From PyPI, into a virtual environment. PyPI carries yonga from 0.5.1 on; earlier versions are kept only as git tags.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "yonga[openvino,export]"
```

On a machine without an NVIDIA GPU, install the CPU builds of torch and torchvision first. The
`export` extra pulls both in (`optimum-intel` requires `torchvision`), and the default PyPI build on
Linux carries CUDA — on the reference host 1.2 GB of torch plus 3.2 GB of NVIDIA libraries that an
Intel AI PC never uses. The CPU torch wheel is 196 MB:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install "yonga[openvino,export]"
```

Install the two together, from the same index. With only torch from the CPU index, pip fetches
torchvision from PyPI, and that build cannot register its operators against a CPU torch:
`import torchvision` fails with `operator torchvision::nms does not exist`, and with it the
exporter, while `pip check` reports nothing wrong. `prepare` imports the exporter before it
downloads anything, so it reports such a pair up front as `export.torchvision.build_mismatch`
(exit 10). `TROUBLESHOOTING.md` section 5.7 has the repair for an environment already in that state.

Add `serve` to the extras for the HTTP API: `pip install "yonga[openvino,export,serve]"`.

To work on `yonga` itself, install from a clone of this repository instead, in editable mode and
with the development tools:

```bash
git clone https://github.com/ooguz/yonga.git
cd yonga
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[openvino,export,serve,dev]"
```

The extras map to what you intend to do:

| Extra | Packages | Needed for |
|---|---|---|
| `openvino` | `openvino`, `openvino-genai`, `openvino-tokenizers` | `probe`, `chat`, `bench`, and the device/NPU sections of `doctor` |
| `export` | `optimum-intel[openvino]`, `nncf` (pulling in `transformers` and `torch`) | `prepare` (download, convert, quantize) |
| `serve` | `fastapi`, `uvicorn`, `pillow` | `serve` (the OpenAI-compatible HTTP API; Pillow decodes images sent to a VLM) |
| `dev` | `pytest`, `pytest-cov`, `pytest-mock`, `pytest-timeout`, `ruff`, `mypy`, `build`, `twine` | running the test suite and the linters, building and checking a release |
| `all` | `openvino`, `export` and `serve` (not `dev`) | everything a user can run |

Without any extras the package still installs and runs. `inspect`, `list`, `logs`, `registry` and
`doctor` work; `doctor` exits **10** and names the missing package and the extra that provides it
(verified):

```text
OpenVINO
✗  Runtime          not installed  openvino was not found in this environment.
                    → Install the runtime extras: pip install 'yonga[openvino]'
                    → yonga never installs, upgrades or removes a package in the environment it
                    runs from: run the command yourself.
```

That is one row of several: `GenAI` and `Tokenizers` read the same way, and `optimum-intel`, `nncf`
and `transformers` are warnings rather than errors, because without them `prepare` converts in a
managed export environment instead (section 6).

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

With yonga installed, [`USAGE.md`](USAGE.md) walks through every command, from the first `doctor`
to preparing, chatting with and serving a model. The rest of this document covers the host setup
that decides whether those commands work.

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
yonga never runs sudo and never changes packages, kernel modules, permissions or group membership: run any command yourself.
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
                      → yonga never runs sudo and never changes packages, kernel modules, permissions or group membership: run any command yourself.
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
   → yonga never installs, upgrades or removes a package in the environment it runs from: run
     the command yourself.
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
✓  managed temp         disk-backed (ext4)  /home/user/.cache/yonga/tmp
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
`transformers ==5.2.*`, while the reference host runs transformers 5.5.4; yonga meets the pin with a
managed export environment (below). `yonga inspect` states the requirement whether or not it is
currently met (verified):

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

The same happens when there is no export toolchain beside yonga at all — a `pip install` without
the `export` extra, or the AppImage. The environment then gets the toolchain yonga is tested with
(optimum-intel 2.2.0, optimum 2.3.0, nncf 3.4.0, transformers 5.5.4, torch 2.14.0 CPU), still under
any requirement the registry declares. Once it is built, the Transformers range optimum-intel itself
declares for the model type is read *inside* it; when the environment falls outside that range —
Gemma-3 allows at most 5.0 — a second environment inside the range is built instead, still before
the download.

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
CPU compile, generation) finishes in minutes and the artifact is around 430 MiB. If something fails,
`docs/TROUBLESHOOTING.md` maps the diagnostic code to what it means.
