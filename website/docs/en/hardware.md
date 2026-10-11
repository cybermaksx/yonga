# Tested hardware and models

Everything yonga's documentation claims about devices, models and speed was measured on one
machine. This page lists that machine, the model families verified on it, the highlights of the
profile measurements, and what is known not to work. Read it as a record of what was run, not as
a promise about other hardware: no other CPU, GPU, NPU, distribution or OpenVINO release has been
tested.

## The reference machine { #reference-machine }

| | |
|---|---|
| CPU / platform | Intel Core Ultra 7 268V (Lunar Lake) |
| GPU | Intel Arc iGPU (`Intel(R) Arc(TM) Graphics (iGPU)`, `GPU: vendor=0x8086 arch=v20.4.4`) |
| NPU | Intel AI Boost, `/dev/accel/accel0`, `intel_vpu` kernel module, compiler-in-plugin |
| OS | Ubuntu 26.04.1 LTS, kernel 7.2.6 |
| Python | 3.14.4 (the test suite also passes on 3.11) |
| OpenVINO | 2026.4.0 |
| OpenVINO GenAI | 2026.4.0.0 |
| OpenVINO Tokenizers | 2026.4.0.0 |
| optimum-intel | 2.2.0 |
| nncf | 3.4.0 |
| transformers | 5.5.4 on the host; Qwen3.5 exports used 5.2.0, the version the registry pins for that architecture |
| huggingface_hub | 1.21.0 |

Ubuntu 24.04 is an intended target but has not been tested yet. To compare your own machine with
this one, run `yonga doctor`; [Installation](installation.md) covers drivers and permissions.

## Model families verified on that machine { #families }

One small model per family was prepared and probed with a generation smoke test on the GPU and on
the NPU, and benchmarked on the CPU:

| Family | Model | CPU | GPU | NPU |
|---|---|---|---|---|
| qwen3 | `Qwen/Qwen3-0.6B` | ✓ | ✓ | ✓ 67.3 tok/s |
| qwen2 | `Qwen/Qwen2.5-0.5B-Instruct` | ✓ 23.8 tok/s | ✓ | ✓ |
| llama | `unsloth/Llama-3.2-1B-Instruct` | ✓ 12.6 tok/s | ✓ | ✓ |
| gemma3 (text) | `unsloth/gemma-3-1b-it` | ✓ 15.3 tok/s | ✓ | ✓ |
| gemma2 | `unsloth/gemma-2-2b-it` | ✓ 7.2 tok/s | ✓ | ✓ |
| olmo2 | `allenai/OLMo-2-0425-1B-Instruct` | ✓ 13.9 tok/s | ✓ | ✓ |
| granite | `ibm-granite/granite-3.3-2b-instruct` | ✓ 7.6 tok/s | ✓ | ✓ |
| smollm3 | `HuggingFaceTB/SmolLM3-3B` | ✓ 6.2 tok/s | ✓ | ✓ |
| lfm2 | `LiquidAI/LFM2-350M` | ✓ 21.8 tok/s | ✓ | ✓ |
| qwen2.5-vl | `Qwen/Qwen2.5-VL-3B-Instruct` | ✓ 6.2 tok/s | ✓ | ✗ compiler refuses the graph |
| qwen3.5 | the reference 9B VLM | ✓ | ✓ | ✗ compiler refuses the graph |
| phi3 | `microsoft/Phi-4-mini-instruct` | ✓ prepared and evaluated | — not tried | ✓ 21.9 tok/s (group-wise) |

CPU figures are single-run INT4 throughput on the short prompt. The "reference 9B VLM" is
`HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive`, a GGUF-only repository, prepared through
its declared safetensors derivative. Gemma, LFM2 and Qwen2.5-VL need an older Transformers than
the one installed; `prepare` exported them in managed export environments.

### What "tested" means { #what-tested-means }

In the registry, these families' device claims carry the evidence level `tested` because of the
runs above: on this machine, with these package versions, the model prepared, compiled and
generated on that device. It is evidence, not a guarantee:

- It is **not a support matrix**. Another model of the same family, another size, another Intel
  generation or another OpenVINO release may behave differently.
- A family that is not in the table can still work. It is then rated `inferred` or `unknown`, and
  the probe at the end of `prepare` decides. [How yonga works](concepts.md#registry) explains the
  evidence levels.
- The tok/s figures are single runs of small models, given as an illustration. They are not a
  performance claim.

For a model you care about, `yonga inspect MODEL` shows what the registry knows about it on your
host.

### A real NPU run { #npu-run }

This is a benchmark of `Qwen/Qwen3-0.6B` with the `npu-int4-sym-g128` profile, run with yonga
0.5.0 on the reference machine (trimmed):

```text
Model           Qwen/Qwen3-0.6B
Artifact        cc24636a2f49356a
Device          NPU
Profile         npu-int4-sym-g128
Suite           standard  (3 runs, 1 warmup)
Max new tokens  256
Compile         858 ms

Prompt               In tok   Out tok       TTFT   tok/s      TPOT    Wall
──────────────────────────────────────────────────────────────────────────
short-instruction        31       256   164.4 ms    68.2   14.7 ms   3.9 s
medium-explanation      104       256   164.7 ms    67.7   14.8 ms   3.9 s
long-context            623       256   165.0 ms    67.7   14.8 ms   3.9 s
```

[Chat, probe, bench and eval](guide/run.md) explains how to run your own.

## Profile measurements { #measurements }

The defaults described in [Profiles](concepts.md#profiles) rest on measurements made on the
reference machine with `yonga eval` and `yonga bench`. `eval` measures how far a quantized
artifact's next-token predictions move from a higher-precision reference of the same weights
(mean KL divergence, top-1 agreement and perplexity change). Lower divergence means less damage
from quantization; it is not a measure of answer quality on real tasks. The full tables and
caveats are in
[PROFILE_COMPARISON.md](https://github.com/ooguz/yonga/blob/main/docs/PROFILE_COMPARISON.md).

### Keeping a fifth of the layers at INT8 below 5B { #ratio-08 }

Twelve models from 0.35B to 8.19B parameters were measured with INT4 at ratio 1.0 and at ratio
0.8, against an INT8 reference:

- Ratio 0.8 is closer to the reference **on every model**, by 16-42 % in KL and 2-6 points of
  top-1 agreement, for **9-18 % more disk**.
- Below 5B, ratio 1.0 often raises perplexity by 20-33 %, and ratio 0.8 removes a fifth to two
  thirds of that increase. At 8B, ratio 1.0 costs +7.2 % and ratio 0.8 +6.8 %.
- In speed, ratio 0.8 costs 10-18 % of the GPU's tokens per second below 5B, 22 % at 8B, and up to
  12 % of the CPU's. Read that as "up to about a fifth": the host's run-to-run spread was about
  20 %.

That is why the `-r08` profiles are the CPU and GPU defaults below 5B parameters.

### The NPU's two INT4 formats { #npu-formats }

- Channel-wise INT4 (`npu-int4-sym-cw`) diverges from the reference **roughly two to five times
  as much** as group-wise (`npu-int4-sym-g128`), on every model measured and at every size. Every
  group-wise artifact compiled and generated on the NPU.
- Group-wise is slower: for Qwen3 it costs 14 % at 0.6B, nothing at 1.7B, and a third at 4B with
  twice the time to first token. For the other families it costs a tenth to a third of the NPU's
  tokens per second.
- At 8B, channel-wise is a third faster and far less accurate: perplexity +42.7 % against +9.4 %.

So the NPU default is group-wise below 5B and channel-wise from 5B up. `--profile
npu-int4-sym-cw` is the fast choice on a small model, and `--profile npu-int4-sym-g128` the
quality choice on a large one.

Qwen3 speed with `yonga bench --suite standard`, tokens per second with the time to first token
in brackets. The GPU columns use the CPU profiles' artifacts, so they run without the GPU
profiles' `ATTENTION_BACKEND=SDPA` setting:

| Qwen3 | GPU r1.0 | GPU r0.8 | CPU r1.0 | CPU r0.8 | NPU cw | NPU g128 |
|---|---|---|---|---|---|---|
| 0.6B | 152.5 (27 ms) | 125.5 (49 ms) | 85.6 (41 ms) | 82.6 (41 ms) | 77.4 (123 ms) | 66.9 (168 ms) |
| 1.7B | 74.4 (47 ms) | 67.1 (56 ms) | 44.4 (101 ms) | 39.9 (109 ms) | 35.2 (319 ms) | 36.9 (431 ms) |
| 4B | 34.7 (60 ms) | 30.4 (74 ms) | 19.2 (276 ms) | 17.0 (292 ms) | 24.6 (832 ms) | 16.5 (1651 ms) |
| 8B | 16.6 (96 ms) | 13.0 (128 ms) | 10.0 (625 ms) | 9.7 (547 ms) | 13.9 (1785 ms) | 10.6 (2842 ms) |

### A 9B VLM on the GPU { #vlm-9b }

Two INT4 quantizations of the same Qwen3.5-9B weights were compared on the GPU:
`gpu-int4-asym-g128` and the NPU's channel-wise `npu-int4-sym-cw`.

- The channel-wise artifact is 2.17 % smaller, and the speed difference between the two was
  within run-to-run noise.
- The group-wise asymmetric artifact stays much closer to an INT8 reference: mean KL 0.080
  against 0.197, top-1 agreement 86.8 % against 81.8 %, and perplexity +7.1 % against +22.4 %, on
  every one of eight passages and on both CPU and GPU.
- Both lose most on Turkish text and least on Python code, so an evaluation on English prose
  alone would understate the cost of quantization.

### GGUF imports { #gguf }

For `Qwen/Qwen2.5-0.5B-Instruct`, imported GGUF files have a divergence floor of about 0.03 nats
even unquantized, where an optimum INT8 export reaches 0.006; Q8_0 and Q6_K sit on that floor.
The GGUF Q4_0 file was closer to the reference than yonga's ratio-1.0 INT4 export (KL 0.199
against 0.290).

!!! note
    All of these numbers come from one laptop, one driver, one OpenVINO release and one thermal
    state. They explain why the defaults are what they are; they do not predict what your machine
    will measure.

## Known not to work { #not-working }

### Vision-language models on the NPU { #vlm-npu }

The NPU compiler refuses the graphs of both VLM families tested: Qwen3.5 (the 9B reference model)
and Qwen2.5-VL (`Qwen/Qwen2.5-VL-3B-Instruct`). The driver and compiler are healthy; the graph is
what this NPU compiler cannot take. The registry records the NPU as unsupported for both with
tested evidence, so:

- `--device auto` never picks the NPU for them;
- an explicit `--device npu` is refused before the download, with exit 50 and the commands that
  prepare the model for the GPU or the CPU instead.

Use the GPU or the CPU for these models.

### phi3 and mistral exports on Python 3.14 { #python-314 }

optimum 2.3.0 and earlier cannot build the export configuration of some architectures on Python
3.14, phi3 (Phi-4-mini) and mistral among them. The fix exists upstream (optimum pull request
2496) but no release carries it yet. Python 3.11, 3.12 and 3.13 are not affected.

yonga knows this. When it can tell that the export will run on Python 3.14 with an affected
optimum, it refuses before the download; otherwise it classifies the failure. Either way the code
is `export.optimum.normalized_config` and the exit code is 10. The AppImage bundles Python 3.14 and builds its
managed export environments on it, so it cannot export these models. Install yonga with pip into
a virtual environment on Python 3.11-3.13 instead:

```bash
python3.13 -m venv ~/.venvs/yonga
~/.venvs/yonga/bin/pip install "yonga[openvino,export]"
```

The phi3 results in the family table, like the phi3 and mistral rows in PROFILE_COMPARISON.md,
come from an export on Python 3.14 with optimum 2.3.0 plus that unreleased fix. A released yonga
cannot patch optimum for you.

### Granite on the NPU { #granite-npu }

`ibm-granite/granite-3.3-2b-instruct` works on the NPU but is slow there in both INT4 formats:
about 4.5 tok/s, where models of its size reach 25-40. The cause has not been isolated.

### Windows { #windows }

Windows is not supported yet. yonga runs on Linux x86_64 only: every device, driver and path
probe is written and tested against Linux.
