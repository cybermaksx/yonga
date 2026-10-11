# Quick start

This page takes you from nothing to a local chat in four steps: install yonga, check your
machine, prepare a small model and talk to it. It uses `Qwen/Qwen3-0.6B`, a model small enough
that the whole path finishes in minutes. For every option and the details behind each step, see
[Installation](installation.md) and the [guides](guide/prepare.md).

## Before you start { #before-you-start }

You need:

- Linux on x86_64. Windows is not supported yet.
- An internet connection for the first preparation: yonga downloads the model from the Hugging
  Face Hub (about 1.4 GiB for this one).
- A few GiB of free disk space in your home directory. The prepared model is about 430 MiB, and
  the first conversion may also build a managed export environment of up to about 3 GiB (see
  [step 3](#prepare)).

yonga itself needs no root access: it never runs `sudo` and keeps everything under your home
directory. Only the one-time driver and `render` group setup for the GPU or NPU, if your machine
still needs it, takes `sudo`, and you run those commands yourself (see
[Installation](installation.md#the-intel-npu)).

## 1. Install yonga { #install }

=== "AppImage"

    The AppImage is one file that carries yonga, its own Python and the OpenVINO runtime. Download
    it with its checksum, check it, and install it as the `yonga` command:

    ```bash
    curl -LO https://github.com/ooguz/yonga/releases/download/v0.5.1/yonga-0.5.1-x86_64.AppImage
    curl -LO https://github.com/ooguz/yonga/releases/download/v0.5.1/yonga-0.5.1-x86_64.AppImage.sha256
    sha256sum --check yonga-0.5.1-x86_64.AppImage.sha256
    chmod +x yonga-0.5.1-x86_64.AppImage
    ./yonga-0.5.1-x86_64.AppImage self install
    ```

    `self install` copies the file to `~/.local/bin/yonga`. If that directory is not on your `PATH`
    yet, it says so and prints the line to run. After the very first install on Ubuntu or Debian,
    logging out and back in once is usually enough.

=== "pip"

    Install from PyPI into a virtual environment (Python 3.11 to 3.14). On an Intel machine without
    an NVIDIA GPU, install the CPU builds of `torch` and `torchvision` first, together: that saves
    about 4 GB of CUDA libraries you would never use.

    ```bash
    python3 -m venv ~/.venvs/yonga
    source ~/.venvs/yonga/bin/activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install "yonga[openvino,export,serve]"
    ```

    The `yonga` command is available while the environment is active.

See [Installation](installation.md) for what each extra does, the FUSE note for the AppImage, and
how to upgrade or uninstall.

## 2. Check your machine { #check }

```bash
yonga doctor
```

`doctor` reports your OS, Python, the OpenVINO packages, every device OpenVINO can see (CPU, GPU,
NPU), whether you may use the NPU, free disk space and Hugging Face authentication. It changes
nothing; when something is missing it tells you the command to run yourself.

The exit code sums it up:

| Exit code | Meaning |
|---|---|
| `0` | Everything is usable. |
| `2` | Usable, with warnings: the report ends with `Usable, with warnings above.` Read the `!` lines. |
| `10` | Something is missing that yonga cannot work around; the report says what. |

!!! note
    On Ubuntu, `/tmp` is a small RAM-backed filesystem, so `doctor` usually exits `2` with a
    warning under **Storage → system temp**. You can ignore it: yonga converts models in its own
    disk-backed directory and never needs `/tmp` enlarged.

If the **NPU** section shows a warning, see [Installation](installation.md#the-intel-npu).

## 3. Prepare a small model { #prepare }

```bash
yonga prepare Qwen/Qwen3-0.6B
```

yonga reads the model's metadata, chooses a device and a quantization profile, and prints a
**preparation plan**: the exact revision, the device and why, the profile, the expected download,
the disk it needs and anything it has to build. It asks before doing anything expensive. Answer
yes, and it downloads, converts, repairs known issues and runs a short generation test on the
device before it calls the model ready.

To choose the device yourself, add `--device npu`, `--device gpu` or `--device cpu`. With an
explicit device yonga never switches to another one on its own. This is a trimmed run with
`--device npu` on an Intel Core Ultra 7 268V:

```text
╭───────────────────────────── Preparation plan ─────────────────────────────╮
│                                                                            │
│  Requested model             Qwen/Qwen3-0.6B@c1899de289a04d12100db370d814  │
│                              85cdf75e47ca                                  │
…
│  Device decision             NPU  supported (tested)  NPU was requested    │
│                              explicitly; yonga never switches device on    │
│                              its own.                                      │
│  Profile                     npu-int4-sym-g128                             │
│                              NPU group-wise INT4; the NPU default below    │
│                              5B parameters                                 │
│  Quantization                int4, group size 128, ratio 1, symmetric      │
…
│  Expected download           1.4 GiB  only the files the conversion reads  │
…
│  Estimated artifact          about 430 MiB                                 │
…
│  Export environment          managed export-fbfac3ac91ba                   │
│                              transformers==5.5.4  will be built            │
…
╰───────────────── job de862650-85f5-44f4-9921-1a415bc541d3 ─────────────────╯
1/7 Inspecting source
2/7 Resolving compatibility
3/7 Preflight
Building managed export environment export-fbfac3ac91ba (transformers==5.5.4)
4/7 Downloading
Fetching Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca (1.4 GiB)
5/7 Converting
Converting with profile npu-int4-sym-g128
6/7 Validating / repairing
Inventorying OpenVINO IR
Applying registry repairs: ov-tokenizer-precision-string-casing
7/7 Probing runtime
· NPU  Compiling openvino_model.xml on NPU
· NPU  Generating 24 tokens on NPU
…
Ready
  Artifact  cc24636a2f49356a
…
  Device    NPU
  Profile   npu-int4-sym-g128
```

The result is an **artifact**: the converted model, with a manifest that records where it came
from and how it was made. `Artifact` is its id; `yonga list` shows every artifact on this machine.

!!! tip "The managed export environment"
    The AppImage, and a pip install without the `export` extra, do not carry the conversion
    toolchain. The first conversion therefore builds a *managed export environment* under
    `~/.cache/yonga/envs`, as the plan above says, and later preparations reuse it. Your own Python
    environment is never changed. [Installation](installation.md#managed-export-environments)
    explains why.

## 4. Chat with it { #chat }

```bash
yonga chat Qwen/Qwen3-0.6B
```

You can pass the model id or the artifact id that `prepare` printed (`yonga chat cc24636a2f49356a`).
Without `--device`, `chat` uses the device the artifact was prepared for. The session opens with a
summary, and every reply ends with a metrics line:

```text
Loading the model on NPU...
╭──────────────────────────────── Local Chat ────────────────────────────────╮
│                                                                            │
│  Model              cc24636a2f49356a                                       │
│  Source             Qwen/Qwen3-0.6B@c1899de289a0                           │
│  Profile            npu-int4-sym-g128                                      │
│  Device             NPU                                                    │
│  Max output         4096 tokens                                            │
│  Reasoning display  hidden                                                 │
…
│  Thinking mode      off (enable_thinking=false)                            │
│  Sampling           temp 0.7 • top-p 0.8 • top-k 20                        │
│  Metrics footer     on                                                     │
│                                                                            │
╰──────────────────────────── /help for commands ────────────────────────────╯
…
TTFT 391 ms • 44.3 tok/s • TPOT 22.6 ms/token • 22 → 27 tokens
```

The footer shows the time to the first token, the generation speed, the time per output token and
the prompt and reply token counts. Type `/help` for the chat commands (`/reset`, `/tokens N`,
`/temp X`, `/reasoning on|off` and more) and `/exit` or ++ctrl+d++ to leave.

## Next steps { #next-steps }

- **Serve it to other tools.** `yonga serve Qwen/Qwen3-0.6B` starts an OpenAI-compatible API on
  `http://127.0.0.1:8000/v1`. See [OpenAI-compatible server](guide/serve.md).
- **Measure it.** `yonga bench Qwen/Qwen3-0.6B` runs a prompt suite and saves the result as JSON;
  `yonga probe` checks that an artifact compiles on a device; `yonga eval` measures what the
  quantization cost. See [Chat, probe, bench and eval](guide/run.md).
- **Prepare something else.** `yonga inspect MODEL` tells you what preparing a model would involve
  before anything is downloaded. See [Preparing models](guide/prepare.md) and
  [Tested hardware and models](hardware.md).
- **When something fails,** the message carries a diagnostic code and the next step. See
  [Troubleshooting](troubleshooting.md).
