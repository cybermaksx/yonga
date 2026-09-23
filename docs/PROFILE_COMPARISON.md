# PROFILE_COMPARISON.md

Two INT4 quantizations of the same Qwen3.5-9B weights, converted and measured on one machine in one
session.

**This measures size and speed only. Output quality was not evaluated at all.** Nothing here says
that either artifact answers better, and neither a smaller file nor a higher tok/s is evidence that
it does. Weight quantization changes what the model computes; the only measurements below are of
bytes on disk and of how fast tokens came out. Do not read a ranking into them.

## 1. What this document is, and what asked for it

A note on provenance, because the honest answer matters more than a tidy citation: **`SPEC.md` has
no section 24.6.** Section 24 is "Success criteria for MVP", thirteen numbered acceptance items, and
none of them asks for a profile comparison. What the specification actually provides is the material
for one:

- **§20 "Qwen3.5 reference case"** names the NPU-oriented export — INT4 symmetric, ratio 1.0,
  group_size -1 — and records that it failed to compile on this host's NPU while the same submodels
  compiled on the Arc GPU.
- **§25 "Reference implementation facts"** records that the official OpenVINO Qwen3.5-9B INT4
  artifact uses INT4 asymmetric, group size 128, ratio 1.0.
- **§6** sets the only comparison rule the specification does state: *"Do not compare different
  devices using different prompt/generation settings."*
- **§21 "Publication support"** lists what a published artifact must disclose — tested
  hardware/software matrix, unsupported devices, quantization properties, repairs, and no claim that
  is not true. This document follows that list, although §21 is about a future `yonga publish`
  command and not about this file.

So the comparison is derived from the specification rather than demanded by it, and the "superiority
must not be inferred from size or tok/s" rule is asserted here on its own merits rather than quoted
from §24.6.

## 2. The two artifacts

| | **A** | **B** |
|---|---|---|
| Profile | `npu-int4-sym-cw` | `gpu-int4-asym-g128` |
| Artifact id | `197ef1cdb734bd80` | `289ede59ae155cd8` |
| Weight format | INT4 | INT4 |
| Symmetric | yes | no |
| Group size | -1 (channel-wise) | 128 |
| Ratio | 1.0 | 1.0 |
| Pipeline properties | *(none)* | `ATTENTION_BACKEND=SDPA` |
| Manifest device | GPU | GPU |

Both were converted from the same source at the same immutable revision:
`nDimensional/Qwen3.5-9B-Uncensored-Safetensors@4623e504074fc2489f3af98dc3d68b8392bbc21f`, reached
from the requested `HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive` through the registry's
declared `lossless_conversion` derivative. Both are VLM artifacts, task `image-text-to-text`,
architecture rule `arch-qwen3_5`. Both had the `ov-tokenizer-precision-string-casing` repair applied
during validation, with identical before and after hashes for `openvino_tokenizer.xml` and
`openvino_detokenizer.xml`.

### 2.1 A is not an NPU artifact

`npu-int4-sym-cw` is the name of the profile, not a claim about the device. An earlier explicit
`--device npu` run refused to finalize this export, because the NPU compile failed and `yonga` does
not label an artifact for a device it could not probe. That run is job
`de2b9194-2bce-49e4-a171-49529f396c2e`, on this same model and this same profile: the export itself
succeeded, then five submodel compiles failed on the NPU and the job ended with

```text
code     npu.compiler.dynamic_shape
category compiler_unsupported_graph
stage    probed
NPU was requested explicitly, so yonga will not silently prepare for another device.
```

so nothing was published. Artifact A was therefore produced **on the GPU**, for comparison, with the
real CLI:

```bash
yonga prepare HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive \
    --device gpu --profile npu-int4-sym-cw --yes
```

The CLI accepts this. A profile whose declared device differs from the target is used as requested —
`prepare` does not refuse it, and the resulting manifest records `device: GPU`, which is the device
it was actually probed on. That is the right behaviour: the profile is a set of conversion knobs,
the device is a fact about what compiled, and conflating the two is how an artifact ends up labelled
for hardware nobody tested. The one thing `prepare` does *not* do is say out loud that the profile
was written for another device; the profile's summary line in the plan ("NPU channel-wise INT4
profile for larger supported LLMs") is the only hint.

### 2.2 The profiles differ in more than quantization

This is the most important caveat in this document. `gpu-int4-asym-g128` declares a runtime property,
`ATTENTION_BACKEND=SDPA`; `npu-int4-sym-cw` declares none. `bench` applies each artifact's own
recommended properties, so **B was measured with SDPA attention and A with the plugin default.**

The section 5 timings are therefore not a clean measurement of asymmetric-g128 against symmetric-cw.
They are a measurement of two profiles as shipped, quantization and pipeline configuration together.
Attributing the difference to the quantization alone would be wrong.

## 3. Software and hardware identity

Recorded by `bench` in both result files; identical for both runs.

| | |
|---|---|
| CPU | Intel(R) Core(TM) Ultra 7 268V (Lunar Lake), 8 logical CPUs |
| GPU | Intel(R) Arc(TM) Graphics (iGPU), `GPU: vendor=0x8086 arch=v20.4.4` |
| NPU | Intel(R) AI Boost (present, unused here), driver `1789049548` |
| Memory | 33,102,655,488 B (30.8 GiB) |
| OS | Ubuntu 26.04.1 LTS, kernel `7.2.6-070206-generic`, x86_64 |
| Python | 3.14.4 |
| OpenVINO | 2026.4.0 (`2026.4.0-22959-99c81491cc3-releases/2026/4`) |
| OpenVINO GenAI | 2026.4.0.0 |
| OpenVINO Tokenizers | 2026.4.0.0 (`2026.4.0.0-736-a04accf6282`) |
| optimum-intel | 2.2.0 (optimum 2.3.0) |
| nncf | 3.4.0 |
| transformers | 5.2.0 |
| torch | 2.14.0 |
| huggingface_hub | 1.21.0 |
| yonga | 0.1.0 |

`transformers` is 5.2.0 because the registry pins `==5.2.*` for this architecture; `prepare` stops at
preflight otherwise.

## 4. Size and conversion

### 4.1 Artifact size

Sums of the per-file sizes recorded in each manifest.

| | A `npu-int4-sym-cw` | B `gpu-int4-asym-g128` | A − B |
|---|---|---|---|
| Whole artifact | 5,996,458,269 B (5.585 GiB) | 6,129,347,101 B (5.708 GiB) | −132,888,832 B (−2.17 %) |

Per submodel, which is where the difference actually lives:

| File | A | B | Δ |
|---|---|---|---|
| `openvino_language_model.bin` | 4274.79 MiB | 4400.94 MiB | −126.15 MiB (−2.87 %) |
| `openvino_language_model.xml` | 4.09 MiB | 4.68 MiB | −0.59 MiB |
| `openvino_text_embeddings_model.bin` | 970.47 MiB | 970.47 MiB | 0 |
| `openvino_vision_embeddings_merger_model.bin` | 431.83 MiB | 431.83 MiB | 0 |
| `openvino_vision_embeddings_pos_model.bin` | 2.54 MiB | 2.54 MiB | 0 |
| `openvino_vision_embeddings_model.bin` | 1.69 MiB | 1.69 MiB | 0 |
| `openvino_tokenizer.bin` | 9.18 MiB | 9.18 MiB | 0 |
| `openvino_detokenizer.bin` | 3.65 MiB | 3.65 MiB | 0 |

Only the language model changes. Nineteen of the twenty-two recorded files are identical in size; the
exceptions are `openvino_language_model.bin`, its `.xml`, and `openvino_config.json` (1641 against
1643 bytes, which is the recorded quantization settings). The vision tower, the text embeddings and
the tokenizer IRs are the same size in both, so the 2.17 % whole-artifact saving is the 2.87 %
saving on one file diluted by everything the profile does not touch. Channel-wise INT4 stores one
scale per output channel instead of one per 128-element group, which is where those 126 MiB go.

### 4.2 Conversion duration

Measured by `prepare` around the `optimum-cli export openvino` subprocess, recorded in the manifest.

| | A | B |
|---|---|---|
| Export subprocess | 161.10 s | 162.12 s |
| Export started | 2026-09-22T21:34:06.920892Z | 2026-09-22T18:51:24.218922Z |
| Artifact published | 2026-09-22T21:37:12.955647Z | 2026-09-22T18:54:41.918420Z |

A one-second difference over 161 s is noise, not a finding, and there is direct evidence of the
scale of that noise: the refused NPU run of section 2.1 executed the *same* `npu-int4-sym-cw` export
of the *same* source in 126.7 s. The same conversion varied by 27 % between two runs on this host,
which is more than thirty times the A-versus-B gap. Both ran with the source already in the Hugging
Face cache, so neither number includes the 17.5 GiB download. End to end, A's `prepare`
invocation took 192 s of wall clock: inspection, compatibility resolution, preflight, a cache-hit
download, the 161 s conversion, validation, the tokenizer repair, the GPU probe and hashing.

### 4.3 Compile and load

Two different things, both worth stating, because a cold compile and a cache-hit load differ by a
factor of 3.6 to 3.7 here.

Cold compile, from the `prepare` probe (first compile of this IR on this device, empty compile
cache):

| Submodel | A | B |
|---|---|---|
| `language` | 8.189 s | 13.533 s |
| `text_embeddings` | 0.478 s | 1.437 s |
| `vision_embeddings_merger` | 1.409 s | 1.757 s |
| `vision_embeddings` | 0.072 s | 0.106 s |
| `vision_embeddings_pos` | 0.038 s | 0.056 s |
| whole-pipeline smoke generation (24 tokens) | 13.882 s | 17.770 s |

Warm load, from the two benchmark runs below (compile cache populated by the probe):

| | A | B |
|---|---|---|
| `compile_seconds` | 2.240 s | 3.793 s |
| `load_seconds` per run | 2.226 s | 3.779 s |

Tokenizer and detokenizer are CPU-hosted in both artifacts and were validated on CPU, never on the
accelerator; the probe records them as `skipped` on GPU for that reason.

## 5. Benchmark

Both artifacts, same device, same suite, same settings, run back to back four minutes apart:

```bash
yonga bench 197ef1cdb734bd80 --device gpu --suite standard --runs 3 --warmups 1
yonga bench 289ede59ae155cd8 --device gpu --suite standard --runs 3 --warmups 1
```

Generation settings, identical in both result files: `do_sample=false`, `max_new_tokens=256`,
`temperature`, `top_p`, `top_k`, `repetition_penalty` and `seed` all unset. Every run produced
exactly 256 output tokens and reported no errors.

- A: `20260922T213821Z-gpu-standard-0507c4bf`
- B: `20260922T214255Z-gpu-standard-2e18d5a8`

### 5.1 Means over the three measured runs

TTFT and TPOT lower is faster; tok/s higher is faster.

| Prompt | In tok | | TTFT | tok/s | TPOT | Wall |
|---|---|---|---|---|---|---|
| short-instruction | 33 | **A** | 123.8 ms | 13.59 | 74.33 ms | 19.08 s |
| | | **B** | 185.1 ms | 10.07 | 101.40 ms | 26.04 s |
| medium-explanation | 106 | **A** | 246.5 ms | 9.28 | 108.58 ms | 27.94 s |
| | | **B** | 213.0 ms | 11.78 | 85.01 ms | 21.89 s |
| long-context | 632 | **A** | 620.7 ms | 13.87 | 72.26 ms | 19.05 s |
| | | **B** | 926.3 ms | 11.78 | 87.41 ms | 23.22 s |

### 5.2 Medians, and the spread the means hide

| Prompt | | TTFT median | TTFT stdev | tok/s median | TPOT median |
|---|---|---|---|---|---|
| short-instruction | **A** | 122.3 ms | 9.5 ms | 14.24 | 70.22 ms |
| | **B** | 190.1 ms | 17.6 ms | 10.00 | 99.99 ms |
| medium-explanation | **A** | 221.6 ms | 86.7 ms | 9.79 | 102.16 ms |
| | **B** | 213.8 ms | 6.2 ms | 11.69 | 85.51 ms |
| long-context | **A** | 609.4 ms | 48.2 ms | 13.98 | 71.51 ms |
| | **B** | 831.5 ms | 180.6 ms | 12.95 | 77.24 ms |

The individual measured runs, in order, as tok/s — the warmup is excluded from every aggregate above
and is shown here in parentheses for completeness:

| Prompt | A | B |
|---|---|---|
| short-instruction | (15.96) 14.74, 14.24, 11.77 | (10.08) 8.33, 10.00, 11.87 |
| medium-explanation | (9.41) 9.89, 8.17, 9.79 | (11.34) 11.31, 11.69, 12.32 |
| long-context | (13.33) 14.61, 13.02, 13.98 | (13.20) 12.95, 13.25, 9.13 |

**Three runs is not enough to separate these two artifacts.** B's long-context runs span 9.13 to
13.25 tok/s — a 45 % spread within one artifact on one prompt — which is larger than the 2.09 tok/s
that separates the A and B means for that prompt. A's short-instruction runs fall monotonically,
14.74 → 14.24 → 11.77, which looks like thermal or contention drift rather than a property of the
artifact. A is ahead on two prompts and behind on one, and the direction of the medium-explanation
result reverses the other two. Read the table as "these two are in the same range on this host",
not as a ranking.

Nor was the machine instrumented for exclusivity: this is a developer workstation, the two runs were
four minutes apart, and nothing here proves that background load was identical across them. That is
one more reason to treat the per-prompt gaps as within noise rather than as measurements of the
profiles.

### 5.3 Peak RSS

Recorded per run: A 1.28 GiB, B 5.60 GiB. This is host process resident memory, not GPU memory, and
for a memory-mapped OpenVINO `.bin` it mostly reflects how much of the file the page cache happened
to have resident. It is reported because `bench` records it, not because it is a meaningful
comparison between the two artifacts.

## 6. Comparability

When this session ran, `BenchmarkResult.comparable_to` reported the two results **comparable**, with
no reasons against:

```text
comparable: True
reasons: []
```

It reached that verdict by checking the benchmark schema version, the generation-settings fingerprint
(identical: `a992e1cb3810e9ab92c133b75e2201b04a5341d81047aa4b6061761a093f53a3` for both), the device
(`GPU` for both), the prompt signature — the `(prompt_id, prompt_sha256)` pairs of the measured runs,
identical for both — and the comparability-critical package versions.

**That verdict no longer stands.** The first limitation recorded below was a defect, and it has since
been fixed: `comparable_to` compares `runtime_properties` too, so on the same two files it now
reports

```text
comparable: False
reasons: ['runtime property ATTENTION_BACKEND differs (SDPA vs unset)']
```

which is the right answer — the two runs were configured differently. The numbers in sections 4 and 5
are still what the runs measured; what changed is that the tool now says out loud that they are not a
like-for-like comparison. Read them as two measurements, not as a ranking.

The two limits as they were recorded at the time:

1. **It does not compare `runtime_properties`.** Both results record them, and they differ:
   `{"ATTENTION_BACKEND": "SDPA"}` for B, `{}` for A. A "comparable" verdict does not currently mean
   the two pipelines were configured the same way. See section 2.2. *(Fixed; see above.)*
2. **It does not compare hardware.** It has no cross-machine guard, so it would call two results
   comparable that were produced on entirely different hosts. It is a same-host check being used as
   one here, which is legitimate, but it is not evidence of cross-machine comparability and no such
   claim is made anywhere in this repository. *(Still true.)*

## 7. What this does and does not establish

Established, on this host, in this session:

- Both quantizations of these weights convert, validate, repair, compile and generate on this Arc
  iGPU, in about 161 s of conversion each.
- The channel-wise symmetric artifact is 2.17 % smaller overall and 2.87 % smaller in the one file
  the profile changes, and its language submodel compiled cold in 8.189 s against 13.533 s.
- At 3 runs per prompt the generation-speed difference between them is smaller than the run-to-run
  variation within either one.

Not established, and not attempted:

- **Output quality.** Nothing here evaluated correctness, coherence, instruction-following,
  hallucination, refusal behaviour, vision-input handling or any benchmark of capability. The two
  artifacts quantize the same weights differently, which by construction changes what the model
  computes. A size or throughput number is not evidence about any of that, and no conclusion about
  which artifact is *better* follows from this document.
- **Which profile to use on the NPU.** Neither artifact compiled on the NPU; A never got that far,
  which is why it exists on the GPU at all.
- **Anything about another machine.** One CPU, one iGPU, one driver, one OpenVINO release, one
  session, one thermal state. `yonga` records the whole identity in section 3 precisely so that a
  reader can tell how narrow that is.
