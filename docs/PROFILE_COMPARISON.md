# PROFILE_COMPARISON.md

Two INT4 quantizations of the same Qwen3.5-9B weights, converted and measured on one machine in one
session.

Sections 4 to 6 measure size and speed; **section 8 measures output quality**, which was added
later with `yonga eval`. The two kinds of result must not be confused: neither a smaller file nor a
higher tok/s is evidence of better output, and in this comparison they point in opposite
directions. The speed difference is within noise. The quality difference is not: the group-wise
asymmetric artifact B stays about 2.5 times closer to an INT8 reference than the channel-wise
symmetric A, on every passage.

## 1. What this document is, and what asked for it

The comparison is derived from what `SPEC.md` provides, not demanded by any one section of it:

- **§20 "Qwen3.5 reference case"** names the NPU-oriented export — INT4 symmetric, ratio 1.0,
  group_size -1 — and records that it failed to compile on this host's NPU while the same submodels
  compiled on the Arc GPU.
- **§25 "Reference implementation facts"** records that the official OpenVINO Qwen3.5-9B INT4
  artifact uses INT4 asymmetric, group size 128, ratio 1.0.
- **§6** sets the only comparison rule the specification states: *"Do not compare different
  devices using different prompt/generation settings."*
- **§21 "Publication support"** lists what a published artifact must disclose — tested
  hardware/software matrix, unsupported devices, quantization properties, repairs, and no claim that
  is not true. This document follows that list.

The rule that superiority must not be inferred from size or tok/s is asserted here on its own merits.

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

- Measured against an INT8 reference of the same weights (section 8), B's predictions stay much
  closer: mean KL 0.080 against 0.197, 86.8 % against 81.8 % top-1 agreement, and perplexity +7.1 %
  against +22.4 %, on every one of eight passages and on both CPU and GPU.

Not established, and not attempted:

- **Task-level capability.** Section 8 measures how far each artifact's next-token predictions
  move from a reference. That is not a measurement of correctness, instruction-following,
  hallucination, refusal behaviour or vision-input handling, and no benchmark of capability was
  run. Closer to the reference is strong evidence of less damage from quantization. It does not
  prove that any particular answer is better.
- **Which profile to use on the NPU.** Neither artifact compiled on the NPU; A never got that far,
  which is why it exists on the GPU at all.
- **Anything about another machine.** One CPU, one iGPU, one driver, one OpenVINO release, one
  session, one thermal state. `yonga` records the whole identity in section 3 precisely so that a
  reader can tell how narrow that is.

## 8. Output quality

Measured after the rest of this document, with `yonga eval` (`CLI_SPEC.md` section 13a). The
measurement is teacher-forced: each artifact reads the same text and, at every position, its
next-token distribution is compared with a reference's. No text is generated, so nothing depends on
sampling, and the numbers are deterministic for a given pair, device and corpus.

### 8.1 The reference, and why it can stand in for the original weights

An FP16 IR of Qwen3.5-9B is about 18 GB and does not fit this host alongside the evaluation. The
reference is therefore `3904983a446e705e`, prepared from the same source and revision with the
`cpu-int8-asym-ref` profile: 8-bit asymmetric per-channel weights, 8.8 GiB. Every figure below is
relative to it, not to the original weights.

Whether INT8 is close enough to stand in for FP16 was measured on a model where FP16 fits,
`Qwen/Qwen3-0.6B` (CPU, f32 inference, same corpus):

| Candidate | Reference | Mean KL | Top-1 agreement | Perplexity change |
|---|---|---|---|---|
| `cpu-int8-asym-ref` | `cpu-fp16-ref` | 0.0074 | 95.42 % | +0.33 % |
| `cpu-int4-asym-g128` | `cpu-fp16-ref` | 0.3563 | 70.18 % | +32.83 % |
| `cpu-int4-asym-g128` | `cpu-int8-asym-ref` | 0.3552 | 70.13 % | +32.39 % |
| `npu-int4-sym-g128` | `cpu-fp16-ref` | 0.3884 | 69.20 % | +33.42 % |
| `npu-int4-sym-g128` | `cpu-int8-asym-ref` | 0.3849 | 69.77 % | +32.98 % |

INT8's own divergence from FP16 is about 48 times smaller than INT4's, and measuring the INT4
artifacts against INT8 instead of FP16 changes their KL by less than 1 %. On this model family,
INT8 is a sound proxy. (The same table records that INT4 at ratio 1.0 costs a 0.6B model a third of
its perplexity; small models quantize badly, see the roadmap.)

### 8.2 The corpus

`yonga-eval-v1`, shipped with yonga: eight passages of original CC0 text, 1955 tokens with this
tokenizer, 1947 scored positions (the first token of each passage has no prediction). There is one
passage per domain: narrative prose, technical explanation, Python, arithmetic reasoning, dialogue,
Turkish prose, a procedure and encyclopedic writing. Both artifacts tokenise it identically, which
`yonga eval` checks before comparing anything.

### 8.3 Results

Candidate against the INT8 reference, on the CPU at `INFERENCE_PRECISION_HINT=f32`, so the
difference is the quantization's and not the arithmetic's:

| | A `npu-int4-sym-cw` | B `gpu-int4-asym-g128` |
|---|---|---|
| Mean KL (nats) | 0.1972 | 0.0799 |
| Top-1 agreement | 81.82 % | 86.80 % |
| Perplexity (reference 4.655) | 5.700 (+22.44 %) | 4.983 (+7.05 %) |

Per passage, mean KL (lower is closer):

| Passage | A | B | A / B |
|---|---|---|---|
| narrative | 0.2030 | 0.0948 | 2.1 |
| technical | 0.2316 | 0.1103 | 2.1 |
| code | 0.0790 | 0.0247 | 3.2 |
| arithmetic | 0.1205 | 0.0541 | 2.2 |
| dialogue | 0.1921 | 0.0614 | 3.1 |
| turkish | 0.3774 | 0.1421 | 2.7 |
| instructions | 0.2362 | 0.0917 | 2.6 |
| facts | 0.1553 | 0.0739 | 2.1 |

The same comparison on the GPU, where both artifacts actually run (FP16 arithmetic on the device):

| | A `npu-int4-sym-cw` | B `gpu-int4-asym-g128` |
|---|---|---|
| Mean KL | 0.1997 | 0.0822 |
| Top-1 agreement | 81.51 % | 86.75 % |
| Perplexity (reference 4.649) | 5.710 (+22.83 %) | 4.977 (+7.07 %) |

The GPU figures agree with the CPU ones to within 3 %. The device's arithmetic adds little to what
the quantization already took. On this host a GPU evaluation took about 60 s and 3 GB of host
memory per pair, against about 900 s and 17 to 19 GB on the CPU.

### 8.4 What this establishes

- B's predictions are measurably closer to the reference than A's: 2.5 times lower mean KL, five
  points more top-1 agreement, and a perplexity increase of 7 % against 22 %. This is not an
  average hiding a mixed picture: B is closer on every passage, by a factor of 2.1 to 3.2.
- The ordering is the same on the CPU and on the GPU.
- Both artifacts lose most on Turkish and least on Python, so the cost of quantization is uneven
  across domains, and an evaluation on English prose alone would understate it.

Read together with sections 4 and 5: A saves 2.2 % of disk and its speed is within the run-to-run
noise of B's, while its output diverges from the reference 2.5 times as much. On this GPU, B is the
better artifact. The channel-wise symmetric format exists because the NPU needs it for larger
models; that is a reason to use it on the NPU, and on this evidence not a reason to use it anywhere
else.

### 8.5 The small-model cost, and what keeping some layers at INT8 buys

Section 8.1 showed that INT4 at ratio 1.0 costs `Qwen/Qwen3-0.6B` a third of its perplexity. Three
variants of the CPU profile were prepared and measured against FP16 in the same way (CPU, f32):

| Profile | Size | Mean KL | Top-1 agreement | Perplexity change |
|---|---|---|---|---|
| `cpu-int4-asym-g128` (ratio 1.0) | 389 MiB | 0.3563 | 70.18 % | +32.83 % |
| ratio 1.0, group size 64 | 397 MiB | 0.2862 | 73.57 % | +22.86 % |
| ratio 0.8 (`cpu-int4-asym-g128-r08`) | 430 MiB | 0.2447 | 74.60 % | +21.99 % |
| ratio 0.6 | 471 MiB | 0.2036 | 77.63 % | +21.44 % |
| `cpu-int8-asym-ref` | 592 MiB | 0.0074 | 95.42 % | +0.33 % |

Keeping 20 % of the layers at INT8 cuts the divergence by 31 % for 10.5 % more disk; 40 % buys a
further 17 % for twice that. The ratio-0.8 profile shipped as `cpu-int4-asym-g128-r08`, opt-in
through `--profile`, because one measured model was not evidence about all of them and the
registry had no size condition to express the rule with. Section 10 measured eleven more models,
and the profile is now the CPU default below 5B parameters.

Reproduce the ratio-1.0 and ratio-0.8 rows (the artifacts are prepared on the way; the group-size-64
and ratio-0.6 rows used profiles that do not ship):

```bash
python tools/measure_profiles.py --out build/qwen3-0.6b.json --model Qwen/Qwen3-0.6B \
  --reference cpu-fp16-ref
```

## 9. GGUF imports against the original weights

Added with direct GGUF import. `Qwen/Qwen2.5-0.5B-Instruct` was exported from its safetensors with
optimum-intel at FP16 (the reference, 962 MiB), INT8 and INT4. The same model's official GGUF release,
`Qwen/Qwen2.5-0.5B-Instruct-GGUF`, was imported at Q4_0, Q6_K, Q8_0 and FP16. Every artifact was
compared with the FP16 export on the CPU at f32, over `yonga-eval-v1`, with
`--allow-different-sources`. The two repositories hold the same model, and their tokenizers agreed
on every token of the corpus, which `yonga eval` checks.

| Artifact | Route | Size | Mean KL | Top-1 agreement | Perplexity change |
|---|---|---|---|---|---|
| `cpu-int8-asym-ref` | optimum-intel export | 493 MiB | 0.0062 | 95.42 % | +0.54 % |
| `gguf-fp16-cpu` | GGUF import, unquantized | 952 MiB | 0.0281 | 90.69 % | +3.46 % |
| `gguf-q8_0-cpu` | GGUF import | 515 MiB | 0.0308 | 90.64 % | +3.94 % |
| `gguf-q6_k-cpu` | GGUF import | 522 MiB | 0.0326 | 90.18 % | +3.80 % |
| `gguf-q4_0-cpu` | GGUF import | 419 MiB | 0.1994 | 78.20 % | +19.58 % |
| `cpu-int4-asym-g128` | optimum-intel export | 328 MiB | 0.2896 | 71.62 % | +30.01 % |

Two findings, and one thing not found:

- **GGUF imports have a divergence floor of about 0.03 nats**, even from the unquantized FP16 file,
  where an optimum INT8 export reaches 0.006. Q8_0 and Q6_K sit on that floor, so at those
  precisions the quantization adds almost nothing measurable; the floor comes from the route. Its
  cause was not isolated. It is either the published GGUF weights themselves or how OpenVINO GenAI
  builds the model from a GGUF file.
- **The GGUF Q4_0 file is closer to the reference than yonga's default INT4 export** (KL 0.199
  against 0.290). The file keeps one tensor at Q8_0 (the output layer). That is the same lesson as
  section 8.5: at this size, what stays at 8 bits matters more than the 4-bit format.
- **Not measured:** GGUF files of other architectures, and any GPU comparison of GGUF imports; the
  CPU was used so the arithmetic would not blur the comparison.

## 10. Small-model defaults: twelve models, three devices

Section 8.5 found that keeping a fifth of the layers at INT8 helps `Qwen/Qwen3-0.6B` a lot, and left
the profile opt-in because one model is not a rule. This section is the evidence the rule now rests
on. It also measured the NPU's two INT4 formats against each other, which nobody had.

### 10.1 Method

`tools/measure_profiles.py` prepared, for each model, the INT8 reference (`cpu-int8-asym-ref`) and
four INT4 artifacts, each on the device its profile names, so the NPU artifacts were also compiled
and probed on the NPU. Every INT4 artifact was then compared with the reference by `yonga eval` on
the CPU at f32 over `yonga-eval-v1`. INT8 stands in for the original weights: it is 0.006-0.007
nats from FP16 (sections 8.1 and 9). Lunar Lake reference host, OpenVINO 2026.4, optimum-intel
2.2.0, NNCF 3.4.0, October 2026. The parameter count is the Hugging Face Hub's, the number size
bands are written in.

```bash
python tools/measure_profiles.py --out build/campaign.json \
  --candidate cpu-int4-asym-g128 --candidate cpu-int4-asym-g128-r08 \
  --candidate npu-int4-sym-cw --candidate npu-int4-sym-g128 \
  --model LiquidAI/LFM2-350M --model Qwen/Qwen3-0.6B ...   # one --model per row below
```

### 10.2 INT4 ratio 1.0 against ratio 0.8

Mean KL / top-1 agreement / perplexity change against INT8, and artifact size.

| Model | Hub count | Ratio 1.0 | Ratio 0.8 | KL | Size |
|---|---|---|---|---|---|
| `LiquidAI/LFM2-350M` | 0.35B | 0.241 / 73.0 % / +16.7 % / 216 MiB | 0.178 / 76.6 % / +12.7 % / 242 MiB | -26 % | +12.3 % |
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.49B | 0.296 / 70.6 % / +29.3 % / 328 MiB | 0.196 / 76.2 % / +19.1 % / 362 MiB | -34 % | +10.3 % |
| `Qwen/Qwen3-0.6B` | 0.75B | 0.355 / 70.1 % / +32.4 % / 389 MiB | 0.245 / 74.6 % / +21.6 % / 430 MiB | -31 % | +10.6 % |
| `unsloth/gemma-3-1b-it` | 1.00B | 0.471 / 75.1 % / +32.5 % / 691 MiB | 0.361 / 78.2 % / +25.8 % / 755 MiB | -23 % | +9.3 % |
| `unsloth/Llama-3.2-1B-Instruct` | 1.24B | 0.247 / 76.3 % / +29.4 % / 761 MiB | 0.196 / 78.8 % / +23.2 % / 853 MiB | -21 % | +12.1 % |
| `allenai/OLMo-2-0425-1B-Instruct` | 1.48B | 0.162 / 78.2 % / +10.6 % / 938 MiB | 0.118 / 82.9 % / +6.9 % / 1039 MiB | -27 % | +10.7 % |
| `Qwen/Qwen3-1.7B` | 2.03B | 0.344 / 77.6 % / +23.7 % / 1017 MiB | 0.199 / 82.2 % / +8.5 % / 1148 MiB | -42 % | +12.9 % |
| `ibm-granite/granite-3.3-2b-instruct` | 2.53B | 0.184 / 80.3 % / +6.4 % / 1311 MiB | 0.128 / 83.8 % / +5.8 % / 1542 MiB | -31 % | +17.6 % |
| `unsloth/gemma-2-2b-it` | 2.61B | 0.080 / 87.3 % / +5.6 % / 1623 MiB | 0.053 / 89.2 % / +4.7 % / 1811 MiB | -34 % | +11.5 % |
| `HuggingFaceTB/SmolLM3-3B` | 3.08B | 0.252 / 80.4 % / +9.5 % / 1674 MiB | 0.212 / 83.2 % / +11.1 % / 1936 MiB | -16 % | +15.7 % |
| `Qwen/Qwen3-4B` | 4.02B | 0.243 / 81.2 % / +18.5 % / 2194 MiB | 0.177 / 84.5 % / +7.5 % / 2535 MiB | -27 % | +15.5 % |
| `Qwen/Qwen3-8B` | 8.19B | 0.136 / 85.1 % / +7.2 % / 4652 MiB | 0.096 / 87.1 % / +6.8 % / 5297 MiB | -29 % | +13.9 % |

Ratio 0.8 is closer to the reference on every model, by 16-42 % in KL and 2-6 points of top-1
agreement, for 9-18 % more disk. Perplexity improves on eleven of the twelve; on SmolLM3-3B it gets
slightly worse while KL and top-1 both improve, so the two measures disagree there. What changes
with size is not the relative gain but how much there is to gain: below 5B the perplexity increase
of ratio 1.0 is often 20-33 % and ratio 0.8 removes a fifth to two thirds of it (a fifth on
gemma-3-1b and Llama-3.2-1B) (Qwen3-1.7B +23.7 %
to +8.5 %, Qwen3-4B +18.5 % to +7.5 %); at 8B it is +7.2 % and ratio 0.8 makes it +6.8 %.

On the GPU itself, `yonga eval --device gpu` gives the same ordering: Qwen3-1.7B +24.4 % against
+10.8 %, Qwen3-4B +27.1 % against +9.8 %.

### 10.3 The NPU's two formats

| Model | Hub count | `npu-int4-sym-cw` KL / perplexity | `npu-int4-sym-g128` KL / perplexity |
|---|---|---|---|
| `LiquidAI/LFM2-350M` | 0.35B | 0.482 / +38.7 % | 0.265 / +21.4 % |
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.49B | 0.749 / +107.1 % | 0.339 / +38.2 % |
| `Qwen/Qwen3-0.6B` | 0.75B | 0.775 / +81.3 % | 0.385 / +33.0 % |
| `unsloth/gemma-3-1b-it` | 1.00B | 1.037 / +55.6 % | 0.520 / +32.4 % |
| `unsloth/Llama-3.2-1B-Instruct` | 1.24B | 0.697 / +98.2 % | 0.266 / +33.1 % |
| `allenai/OLMo-2-0425-1B-Instruct` | 1.48B | 0.391 / +33.7 % | 0.175 / +10.5 % |
| `Qwen/Qwen3-1.7B` | 2.03B | 0.835 / +108.1 % | 0.330 / +17.9 % |
| `ibm-granite/granite-3.3-2b-instruct` | 2.53B | 1.152 / +151.1 % | 0.226 / +12.5 % |
| `unsloth/gemma-2-2b-it` | 2.61B | 0.190 / +19.6 % | 0.087 / +6.8 % |
| `HuggingFaceTB/SmolLM3-3B` | 3.08B | 0.464 / +24.9 % | 0.245 / +7.2 % |
| `Qwen/Qwen3-4B` | 4.02B | 0.529 / +25.7 % | 0.264 / +12.7 % |
| `Qwen/Qwen3-8B` | 8.19B | 0.444 / +42.7 % | 0.150 / +9.4 % |

Channel-wise INT4 diverges from the reference roughly two to five times as much as group-wise, on every
model and at every size. Before this change the NPU used channel-wise for every family except
Qwen3, so a small model on the NPU was, for example, Qwen2.5-0.5B at +107 % perplexity where
+38 % was available. Every group-wise artifact also compiled and generated on the NPU.

### 10.4 Speed

`yonga bench --suite standard`, two measured runs after one warmup. Tokens per second and time to
first token, averaged over the suite's three prompts. The ratio-1.0 and ratio-0.8 columns use the
CPU-profile artifacts on both the GPU and the CPU, so the GPU columns run without the
`ATTENTION_BACKEND=SDPA` property the GPU profiles set.

| Qwen3 | GPU r1.0 | GPU r0.8 | CPU r1.0 | CPU r0.8 | NPU cw | NPU g128 |
|---|---|---|---|---|---|---|
| 0.6B | 152.5 (27 ms) | 125.5 (49 ms) | 85.6 (41 ms) | 82.6 (41 ms) | 77.4 (123 ms) | 66.9 (168 ms) |
| 1.7B | 74.4 (47 ms) | 67.1 (56 ms) | 44.4 (101 ms) | 39.9 (109 ms) | 35.2 (319 ms) | 36.9 (431 ms) |
| 4B | 34.7 (60 ms) | 30.4 (74 ms) | 19.2 (276 ms) | 17.0 (292 ms) | 24.6 (832 ms) | 16.5 (1651 ms) |
| 8B | 16.6 (96 ms) | 13.0 (128 ms) | 10.0 (625 ms) | 9.7 (547 ms) | 13.9 (1785 ms) | 10.6 (2842 ms) |

Ratio 0.8 costs 10-18 % of the GPU's tokens per second below 5B, 22 % at 8B, and up to 12 % of the
CPU's. With the GPU profiles themselves (`gpu-int4-asym-g128` against `gpu-int4-asym-g128-r08`,
SDPA on), alternated runs with cool-downs at 1.7B and 4B gave suite medians of 58-64 against 53-66
tok/s and 28-32 against 28-29 tok/s: the difference was inside the host's run-to-run spread of
about 20 %, which this laptop's thermal state (72-83 °C during the runs) makes wide. Read the speed
cost as "up to about a fifth", not as a precise figure. Group-wise INT4
on the NPU costs 14 % at 0.6B, nothing at 1.7B, and a third at 4B with twice the time to first
token, which is the size where the NPU documentation starts recommending channel-wise.

### 10.5 What the registry does with it

Below 5B parameters each device now defaults to the more accurate profile: `cpu-int4-asym-g128-r08`,
the new `gpu-int4-asym-g128-r08` and `npu-int4-sym-g128`. At 5B and above, and whenever the size is
unknown, the defaults are unchanged (`cpu-int4-asym-g128`, `gpu-int4-asym-g128`,
`npu-int4-sym-cw`). Between the 4B and 8B rows nothing was measured, and 5B follows the NPU
documentation's "roughly 4-5B". The plan says why a profile was chosen ("Chosen because"), and
`--profile` overrides it either way.

Exceptions, all deliberate: the VLM families keep their CPU and GPU pins to ratio 1.0, and the
generic rule (Mistral and other unrecognised members of common families) and Phi-3 keep the NPU on
channel-wise INT4, because group-wise was never prepared and probed on the NPU for them.

Two choices the numbers leave open, recorded here rather than hidden in the threshold:

- **At 8B on the NPU, channel-wise is a third faster and far less accurate** (+42.7 % against
  +9.4 % perplexity). The default follows the documentation and the speed;
  `--profile npu-int4-sym-g128` is the quality choice, and it compiles and runs.
- **The VLM families keep ratio 1.0.** No VLM was measured, and a VLM's Hub count includes its
  vision tower, so a band in Hub-count units says less about its language model.

