# FAQ

Short answers to the questions people ask most about yonga, with links to the pages that go into
detail. If your question is about an error message, [Troubleshooting](troubleshooting.md) is the
better place to start.

## Is yonga affiliated with Intel? { #intel }

No. yonga is an independent open-source project under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). It builds on Intel's
open-source OpenVINO toolkit (OpenVINO, OpenVINO GenAI, Optimum Intel and NNCF) and on the Hugging
Face Hub, but it is not made, endorsed or supported by Intel.

## Which models work? { #models }

yonga is for text-generation models (LLMs) and vision-language models (VLMs) that OpenVINO GenAI
can run. It can prepare:

- Hugging Face repositories with Transformers weights (safetensors), which it converts and
  quantizes;
- repositories that already contain an OpenVINO model, which it uses as they are;
- GGUF files of the llama, qwen2 and qwen3 architectures, which it imports.

The families tested on real hardware, with the results per device, are listed on
[Tested hardware and models](hardware.md). Other models of common families are handled by generic
rules and are marked `inferred` rather than `tested`. To find out before you download anything,
ask yonga:

```bash
yonga search qwen3
yonga inspect Qwen/Qwen3-0.6B
```

`inspect` reads only the model's metadata and shows, for each device on your machine, whether it
is expected to work and how sure the registry is.

## Why does yonga use INT4 by default? { #int4 }

Smaller weights take less disk and less memory, which matters on a laptop. `Qwen/Qwen3-0.6B`, for
example, is a 1.4 GiB download and 387 MiB once prepared for the NPU. Every default profile stores most
weights as 4-bit integers.

INT4 costs some output quality, and yonga's defaults are chosen from measurements of that cost.
Below 5B parameters the CPU and GPU defaults keep a fifth of the layers at INT8, and the NPU
default uses group-wise INT4. Measured over twelve models, keeping those layers at INT8 brought the
output 16-42 % closer to the INT8 reference (by KL divergence) for 9-18 % more disk. The numbers
are in
[PROFILE_COMPARISON.md](https://github.com/ooguz/yonga/blob/main/docs/PROFILE_COMPARISON.md#10-small-model-defaults-twelve-models-three-devices).

You can choose another profile with `--profile`. `cpu-int8-asym-ref` and `cpu-fp16-ref` keep more
precision on the CPU, and `yonga eval` measures the difference on your own model:

```bash
yonga registry profiles
yonga prepare Qwen/Qwen3-0.6B --device cpu --profile cpu-int8-asym-ref
```

## Why did yonga choose the GPU and not the NPU? { #why-gpu }

With `--device auto` (the default), yonga ranks the devices by how well the registry knows each
works for the model, then prefers the GPU, then the NPU, then the CPU. On an Intel AI PC the
integrated GPU accepts the widest range of models and the NPU is the most selective. Every family in
the built-in registry also recommends the GPU. The plan's "Device decision" row says which rule
decided.

To use the NPU, ask for it:

```bash
yonga prepare Qwen/Qwen3-0.6B --device npu
```

An explicit device is never switched silently: if the NPU cannot run the model, `prepare` stops and
tells you what to run instead. [How yonga works](concepts.md#devices) explains the rules.

## Does yonga work without an NPU? { #no-npu }

Yes. The NPU is the first thing yonga was built for, but not the only one. The CPU works on any
x86_64 machine, and the integrated Intel GPU works once Intel's compute runtime is installed
(`intel-opencl-icd` on Ubuntu). `yonga doctor` shows which devices OpenVINO can see on your
machine.

## Does it run on Windows or macOS? { #windows }

Not yet. yonga supports Linux on x86_64 only. Its device, driver and path checks are written and
tested for Linux; the design keeps the door open for Windows later. The AppImage needs glibc 2.28
or newer, and a pip install needs Python 3.11 to 3.14.

## Where are models stored, and how much disk do they need? { #storage }

| What | Where |
|---|---|
| Downloaded source models | The normal Hugging Face cache, `~/.cache/huggingface/hub` by default |
| Prepared models (artifacts), logs, benchmarks | `~/.local/share/yonga` |
| Temporary conversion files, compile cache, managed export environments | `~/.cache/yonga` |
| Settings and registry overrides | `~/.config/yonga` |

`yonga registry paths` prints the actual locations, and `yonga list` shows each artifact's size.
Some figures from the reference machine:

- `Qwen/Qwen3-0.6B`: a 1.4 GiB download, a 387 MiB NPU artifact (`npu-int4-sym-g128`), and an
  NPU compile cache entry of 172 MiB.
- A 9B model: about 18 GiB of source weights plus about 44 GiB of temporary space while it
  converts.
- A managed export environment: up to about 3 GiB, built once and reused.

`prepare` works out the space it needs and refuses before downloading if it does not fit. To use
another disk, set `YONGA_DATA_DIR` and `YONGA_CACHE_DIR`; see [Maintenance](guide/maintain.md) for
cleaning up.

## Can I use models that are already in OpenVINO format? { #openvino-models }

Yes, from the Hugging Face Hub. When a repository already contains an OpenVINO model, `prepare`
uses it directly instead of converting, then validates and probes it like any other artifact.
`inspect` tells you in advance:

```text
Direct OpenVINO source  yes  OpenVINO/Qwen3-0.6B-int4-ov
```

`prepare` takes a Hugging Face repository id, so a model must be on the Hub (public, or private
with your token) for yonga to prepare it.

## How does yonga compare with Ollama or llama.cpp? { #ollama }

They solve related problems in different ways. Ollama and llama.cpp are general-purpose local
runners built around the GGUF format. yonga has no inference engine of its own: it prepares models
for Intel's OpenVINO and runs them with OpenVINO GenAI on the Intel NPU, GPU or CPU, and it
explains the device and quantization decisions it makes along the way.

Its `serve` command offers an OpenAI-compatible API, so clients written for that API can use either.
It is deliberately small: one model, one request at a time. If another tool already works well for
you, there is no need to switch. yonga is meant for running models on an Intel AI PC's NPU and GPU
through OpenVINO.

## Does yonga send any data anywhere? { #data }

No. There is no telemetry, and yonga also blocks OpenVINO's own usage reporting. It contacts the
Hugging Face Hub only when a command needs it (searching, inspecting, downloading, publishing, or
`doctor` checking a token you set), Python package indexes only to build a managed export
environment you confirmed, and a registry address only when you run `registry update` with it.
`--offline` stops all Hub access. [Privacy and security](privacy.md) lists every network connection.

## Do I need a Hugging Face account? { #account }

Not for public models. You need an account and a token only for gated or private repositories:
accept the model's conditions on its Hub page, then log in with `hf auth login` or export the token
as `HF_TOKEN`.

## How do I update yonga? { #update }

=== "AppImage"

    Download the new AppImage and its `.sha256` file from the
    [releases page](https://github.com/ooguz/yonga/releases), check it, and install it over the
    old one. With `X.Y.Z` replaced by the new version:

    ```bash
    sha256sum --check yonga-X.Y.Z-x86_64.AppImage.sha256
    chmod +x yonga-X.Y.Z-x86_64.AppImage
    ./yonga-X.Y.Z-x86_64.AppImage self install
    ```

    `self install` shows both versions and asks before it replaces the old copy.

=== "pip"

    In the virtual environment you installed yonga into:

    ```bash
    pip install --upgrade "yonga[openvino,export]"
    ```

Your prepared models are kept. If the new release brings a newer OpenVINO, the old compiled blobs
become `stale-runtime`, and `yonga cache prune` removes them.

## How do I remove everything? { #uninstall }

1. Check where yonga keeps its files, in case you moved them:

    ```bash
    yonga registry paths
    ```

2. Remove the program.

    === "AppImage"

        ```bash
        yonga self uninstall
        ```

        This removes the copy in `~/.local/bin`. Delete the downloaded AppImage file too.

    === "pip"

        ```bash
        pip uninstall yonga
        ```

        Or delete the whole virtual environment.

3. Remove yonga's data, cache and settings (these are the default locations):

    ```bash
    rm -rf ~/.local/share/yonga ~/.cache/yonga ~/.config/yonga
    ```

4. The source models are in the shared Hugging Face cache, which other tools use too. Remove the
   ones you no longer need with Hugging Face's `hf` command:

    ```bash
    hf cache ls
    hf cache rm model/Qwen/Qwen3-0.6B
    ```

yonga installs nothing outside these places and changes no system settings, so nothing else is
left behind.
