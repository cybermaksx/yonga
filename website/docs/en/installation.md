# Installation

yonga runs on Linux x86_64 and comes in two forms: an AppImage, one file with nothing else to
install, and a Python package on PyPI. This page covers what your machine needs (including the
Intel GPU and NPU drivers, which yonga checks but never installs), both ways to install, the
managed export environments yonga builds for model conversion, where it keeps its files, and how
to upgrade and uninstall. The full specification is
[INSTALL.md](https://github.com/ooguz/yonga/blob/main/docs/INSTALL.md) on GitHub.

## Requirements { #requirements }

| | |
|---|---|
| Operating system | Linux on x86_64. Verified on Ubuntu 26.04.1; Ubuntu 24.04 is the second intended target and has not been tested yet. Windows is not supported yet. |
| Processor and devices | Any x86_64 CPU for CPU-only use. An Intel Core Ultra (Meteor Lake or newer) for the NPU, an Intel Arc / Xe GPU for the GPU. |
| AppImage | glibc 2.28 or newer (Ubuntu 20.04, Debian 11, Fedora 29 and later). |
| pip | Python 3.11 to 3.14. |
| Memory | 8 GiB at least; `yonga doctor` warns below that. Converting a mid-sized model wants 16 GiB or more. |
| Disk | Several times the model size. `yonga prepare` computes what a model needs and stops early if it does not fit. |

Everything yonga stores goes under your home directory, and yonga itself never needs root. Only
the driver and group setup below, which you do yourself, takes `sudo`.

## The Intel GPU and NPU { #the-intel-npu }

The CPU works out of the box. The GPU and NPU need drivers from your distribution, which no Python
package or AppImage can carry. `yonga doctor` tells you what is missing.

**GPU.** You need Intel's compute runtime: `intel-opencl-icd` on Ubuntu, or Intel's own package
repository for a newer one.

**NPU.** Three things have to be true before OpenVINO can see the NPU, and `yonga doctor` reports
each of them in its **NPU** section:

1. **The kernel driver is loaded.** The `intel_vpu` module ships with the distribution kernel. The
   user-space side is Intel's Level Zero NPU driver, a distribution package (on the reference
   Ubuntu machine, `intel-level-zero-npu`). `pip install openvino` does not install it.
2. **The device node exists.** The kernel module creates `/dev/accel/accel0`.
3. **Your user can open it.** The node belongs to the group `render` with mode 660, and a new
   user account is usually not in that group.

A working NPU looks like this in `yonga doctor`:

```text
NPU
✓  /dev/accel/accel0   accessible  group render
✓  Group               render  the current user is a member
✓  driver              1789049548
✓  compiler            524291
✓  compiler-in-plugin  available  the NPU plugin compiles graphs itself
…
```

When you are not in the `render` group, `doctor` warns and prints a command for you to run:

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

To fix it, run the command yourself once, then log out and back in, because a new group membership
does not apply to a session that is already running:

```bash
sudo usermod -aG render $USER
```

To try it in a single shell without logging out, run `newgrp render` there.

!!! note "What yonga does not do"
    yonga never runs `apt`, `modprobe`, `sudo` or `usermod`, and never runs `pip` in the
    environment you installed it into. It tells you the command; you decide whether to run it. The
    only place it runs `pip` is inside a [managed export environment](#managed-export-environments)
    under its own cache directory, after you confirm the plan.

A healthy NPU does not promise that every model compiles on it: the NPU compiler refuses some
graphs. yonga reports that as a model and compiler limitation, not as a driver problem, and
recommends another device. See [Tested hardware and models](hardware.md) and
[Troubleshooting](troubleshooting.md).

## Install with the AppImage { #appimage }

Every [GitHub release](https://github.com/ooguz/yonga/releases) carries
`yonga-X.Y.Z-x86_64.AppImage` and its `.sha256`. The image holds yonga, its own Python, the
OpenVINO runtime and the `serve` extra. For 0.5.1:

```bash
curl -LO https://github.com/ooguz/yonga/releases/download/v0.5.1/yonga-0.5.1-x86_64.AppImage
curl -LO https://github.com/ooguz/yonga/releases/download/v0.5.1/yonga-0.5.1-x86_64.AppImage.sha256
sha256sum --check yonga-0.5.1-x86_64.AppImage.sha256
chmod +x yonga-0.5.1-x86_64.AppImage
./yonga-0.5.1-x86_64.AppImage self install
yonga doctor
```

A downloaded file is not a command yet. `self install` copies the AppImage to `~/.local/bin/yonga`,
so `yonga` works in any shell; the downloaded file can be deleted afterwards. To install into
another directory you own, add `--bin-dir DIR`.

- **PATH.** `self install` checks that `yonga` now runs the copy. If `~/.local/bin` is not on your
  `PATH`, it says so, prints what to do and exits `2`. On Ubuntu and Debian, `~/.profile` adds
  `~/.local/bin` at login, but only if the directory already exists then, so after the first
  install log out and back in once, or run this in the current shell:

    ```bash
    export PATH="$HOME/.local/bin:$PATH"
    ```

    yonga never edits `~/.profile`, `~/.bashrc` or any other startup file.

- **What it leaves alone.** Anything at `~/.local/bin/yonga` that is not a yonga AppImage (a pip
  or pipx command, a symbolic link, a directory, a script) is refused and left as it is. A
  directory you cannot write to is refused too; choose one you own with `--bin-dir`.
- **FUSE.** The AppImage mounts itself with FUSE. Where FUSE is not available (containers, some
  minimal installs), set `APPIMAGE_EXTRACT_AND_RUN=1` and it unpacks itself to a temporary
  directory instead. This works for `self install` too:

    ```bash
    APPIMAGE_EXTRACT_AND_RUN=1 ./yonga-0.5.1-x86_64.AppImage self install
    ```

- **Conversion.** The conversion toolchain (optimum-intel, NNCF, Transformers, PyTorch) is not in
  the image: it is about 1.5 GB and the right Transformers release depends on the model. The first
  `prepare` that converts a model builds a [managed export environment](#managed-export-environments)
  for it.

The Python packages in the image are resolved from PyPI when it is built, so each release also
carries `yonga-X.Y.Z-x86_64.AppImage.packages.txt`, listing every bundled package version and the
SHA-256 of the yonga wheel inside it.

`yonga doctor` shows how yonga is running in the **Host → Install** line, and warns when an
AppImage is not what `yonga` on your `PATH` runs. The details of `self install` are in
[CLI_SPEC.md](https://github.com/ooguz/yonga/blob/main/docs/CLI_SPEC.md#12c-self).

## Install with pip { #pip }

yonga is on PyPI from 0.5.1. Install it into a virtual environment:

```bash
python3 -m venv ~/.venvs/yonga
source ~/.venvs/yonga/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install "yonga[openvino,export,serve]"
```

The `torch` line is optional but recommended on a machine without an NVIDIA GPU. The `export` extra
pulls in PyTorch, and the default PyPI build on Linux carries CUDA: about 4 GB of libraries an
Intel AI PC never uses, against a CPU wheel of about 200 MB.

!!! warning "Install torch and torchvision together"
    Take both from the CPU index, in one command, as above. With only `torch` from there, pip
    fetches `torchvision` from PyPI, the two cannot load together, and model conversion fails with
    `operator torchvision::nms does not exist`. `yonga prepare` detects this before it downloads
    anything (`export.torchvision.build_mismatch`, exit 10), and `yonga doctor` shows it under
    **OpenVINO → exporter import**. The repair for an environment already in that state is in
    [TROUBLESHOOTING.md](https://github.com/ooguz/yonga/blob/main/docs/TROUBLESHOOTING.md#57-a-managed-export-environment-could-not-be-built).

### Extras { #extras }

Pick the extras for what you want to do:

| Extra | Installs | Needed for |
|---|---|---|
| `openvino` | `openvino`, `openvino-genai`, `openvino-tokenizers` | `probe`, `chat`, `bench`, `eval`, `serve`, and the device and NPU sections of `doctor` |
| `export` | `optimum-intel[openvino]`, `nncf` (which pull in `transformers` and `torch`) | `prepare`, converting in your own environment |
| `serve` | `fastapi`, `uvicorn`, `pillow` | `serve`, the OpenAI-compatible HTTP API |
| `all` | `openvino`, `export` and `serve` | everything a user can run |

Without the `export` extra, `prepare` still works: it converts in a
[managed export environment](#managed-export-environments), as the AppImage does. Without any
extras (`pip install yonga`), `doctor`, `inspect`, `list`, `logs` and `registry` work, and
`doctor` exits `10` and names the missing package and the extra that provides it.

!!! note "OpenVINO packages come as a set"
    `openvino`, `openvino-genai` and `openvino-tokenizers` must come from the same release family
    (for example all `2026.4`). Mixing families causes import errors or crashes, so upgrade them
    together. `yonga doctor` checks this in its **ABI family** line and prints the pins to install
    when they differ. See
    [INSTALL.md](https://github.com/ooguz/yonga/blob/main/docs/INSTALL.md#4-the-openvino-abi-family-rule).

A pip install has no `self install`: the `yonga` command lives in the environment's `bin`
directory, so activate the environment to use it. If you want a `yonga` command in `~/.local/bin`
instead, `pipx install 'yonga[openvino,export]'` does that. PyTorch then comes from PyPI, which on
Linux is the larger CUDA build.

## Managed export environments { #managed-export-environments }

Converting a model runs that architecture's code from Hugging Face Transformers, and some
architectures need a specific Transformers release to export correctly. When the packages next to
yonga cannot meet a model's requirement, or there is no conversion toolchain next to yonga at all
(the AppImage, or a pip install without `export`), `prepare` runs the conversion in a **managed
export environment** instead of stopping:

- It is an isolated Python virtual environment under `~/.cache/yonga/envs`, used only by the
  export. Your own environment is never changed. The AppImage keeps a copy of its Python there too,
  because the image's own mount point changes every time it runs.
- It holds the toolchain yonga is tested with, with only the required pin changed, PyTorch from the
  CPU index, and OpenVINO pinned to the runtime's release so the runtime can read what the export
  writes.
- The preparation plan announces it in its **Export environment** row and asks before building it
  (up to about 3 GiB downloaded from PyPI and the PyTorch CPU index). `--yes` approves it.
- It is built and verified before the model download, so a failed build costs minutes, not a
  multi-gigabyte download. Later jobs with the same requirements reuse it.
- Building one needs the network once. Offline, an existing environment is reused, and a job that
  needs a new one stops with `export.env.offline`.

To see and remove them:

```bash
yonga env list
yonga env remove KEY
```

A removed environment is rebuilt if a later export needs it. To refuse managed environments
altogether, pass `--no-managed-env` to `prepare` or set `managed_export_envs = false` in
`config.toml`; preparation then stops before the download (exit 10) and names the package version
it needs. More in
[INSTALL.md](https://github.com/ooguz/yonga/blob/main/docs/INSTALL.md#6-the-transformers-version-caveat).

## Check the installation { #check }

```bash
yonga --version
yonga doctor
```

`doctor` exits `0` when everything is usable, `2` when it is usable with warnings and `10` when
something is missing that yonga cannot work around. On Ubuntu a small RAM-backed `/tmp` alone
produces a `2`: that warning is expected, because yonga converts in its own disk-backed temporary
directory. Then continue with the [Quick start](getting-started.md#prepare).

## Where files live { #where-files-live }

yonga follows the XDG base directory layout:

| Path | Contents |
|---|---|
| `~/.config/yonga/config.toml` | Settings (optional) |
| `~/.config/yonga/registry-overrides/` | Your own compatibility rules |
| `~/.local/share/yonga/artifacts/` | Prepared models (artifacts), each with a manifest |
| `~/.local/share/yonga/logs/`, `benchmarks/`, `chat-history/` | Job logs, benchmark results, chat history |
| `~/.cache/yonga/tmp/` | Managed temporary directories for conversions |
| `~/.cache/yonga/compile/` | OpenVINO compile cache (`yonga cache list`) |
| `~/.cache/yonga/envs/` | Managed export environments (`yonga env list`) |
| `~/.local/bin/yonga` | The AppImage copy made by `self install` |

`yonga registry paths` prints the real locations on your machine. `XDG_CONFIG_HOME`,
`XDG_DATA_HOME` and `XDG_CACHE_HOME` are honoured. To keep models and caches on another disk, set
the directories for one command, in the environment, or in `config.toml`:

=== "Per command"

    ```bash
    yonga --data-dir /mnt/big/yonga-data --cache-dir /mnt/big/yonga-cache list
    ```

=== "Environment"

    ```bash
    export YONGA_DATA_DIR=/mnt/big/yonga-data
    export YONGA_CACHE_DIR=/mnt/big/yonga-cache
    ```

=== "config.toml"

    ```toml
    data_dir = "/mnt/big/yonga-data"
    cache_dir = "/mnt/big/yonga-cache"
    ```

Command-line options win over `YONGA_*` variables, which win over `config.toml`. Source models stay
in the normal Hugging Face cache (`~/.cache/huggingface`) and are never copied or modified.

## Upgrading { #upgrading }

=== "AppImage"

    Download the new AppImage, verify it, and run `self install` from it. It replaces the older
    copy in `~/.local/bin` after asking, showing both versions; `--yes` answers the question.

    ```bash
    ./yonga-X.Y.Z-x86_64.AppImage self install
    ```

=== "pip"

    Upgrade inside the same environment:

    ```bash
    pip install --upgrade "yonga[openvino,export,serve]"
    ```

    Keep the three OpenVINO packages in one release family when you upgrade them.

Run `yonga doctor` afterwards. Until yonga reaches 1.0, a minor release may change the command
line, the registry format and the artifact manifest; the
[changelog](https://github.com/ooguz/yonga/blob/main/CHANGELOG.md) says what changed. An OpenVINO
upgrade leaves old compile-cache entries behind; `yonga cache prune` removes them (see
[Maintenance](guide/maintain.md)).

## Uninstalling { #uninstalling }

=== "AppImage"

    ```bash
    yonga self uninstall
    ```

    This removes the copy in `~/.local/bin` after a question. Prepared models, logs, caches and
    settings stay, and it prints where they are.

=== "pip"

    ```bash
    pip uninstall yonga
    ```

    Or delete the whole virtual environment. Prepared models, logs, caches and settings stay.

To free the disk space as well, remove artifacts with `yonga remove` and the compile cache with
`yonga cache prune --all` before uninstalling, or delete yonga's directories afterwards:

```bash
rm -rf ~/.local/share/yonga ~/.cache/yonga ~/.config/yonga
```

!!! warning
    This deletes every prepared model, log, benchmark, managed export environment and setting. If
    you moved the directories with `--data-dir`, `--cache-dir` or `YONGA_*` variables, delete
    those instead. The Hugging Face cache in `~/.cache/huggingface` belongs to every Hugging Face
    tool and is not removed; `hf cache` can clean it.
