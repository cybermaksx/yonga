# Chat, probe, bench and eval

Once a model is prepared, four commands put it to work on your machine: `chat` talks to it in the terminal, `probe` checks that it compiles and runs on a device, `bench` measures its speed, and `eval` measures how much quantization changed its answers. A fifth, `publish`, uploads it to the Hugging Face Hub. None of them downloads or converts a model; if you have not prepared one yet, start with [Preparing models](prepare.md).

## Choosing the artifact {#choose-artifact}

Every command on this page takes the same reference: an artifact id as `yonga list` prints it, the artifact's directory, or the Hugging Face model id you prepared it from.

```bash
yonga list
yonga chat cc24636a2f49356a
yonga chat Qwen/Qwen3-0.6B
```

A model id often matches more than one artifact, for example one prepared for the NPU and one for the GPU. yonga then takes the newest ready one and tells you which it chose:

```text
Using ID (PROFILE, DEVICE), the newest of N ready artifacts for MODEL; pass the artifact id to choose another.
```

Pass the artifact id when you want a specific one. A model id that has not been prepared on this machine is refused with the `yonga prepare` command to run first; nothing is converted as a side effect.

## Chat in the terminal {#chat}

To chat with a prepared model, run:

```bash
yonga chat Qwen/Qwen3-0.6B
```

yonga loads (compiles) the model first and then shows a startup panel with the model, device, profile, output limit, sampling settings and compile cache. A model that cannot load on the chosen device ends `chat` right there with an [exit code](../reference/exit-codes.md) that says why, for example 51 for an NPU compiler refusal.

This is a real session with Qwen3-0.6B on the NPU of the reference machine (trimmed):

```text
Loading the model on NPU...
╭──────────────────────────────── Local Chat ────────────────────────────────╮
│  Model              cc24636a2f49356a                                       │
│  Source             Qwen/Qwen3-0.6B@c1899de289a0                           │
│  Profile            npu-int4-sym-g128                                      │
│  Device             NPU                                                    │
│  Max output         4096 tokens                                            │
│  Reasoning display  hidden                                                 │
│  Compile cache      enabled                                                │
│  …                                                                         │
│  Chat profile       chat-qwen3                                             │
│  Thinking mode      off (enable_thinking=false)                            │
│  Sampling           temp 0.7 • top-p 0.8 • top-k 20                        │
│  Metrics footer     on                                                     │
╰──────────────────────────── /help for commands ────────────────────────────╯
Generating a reply...
╭────────────────────────────────────────────────────────────────────────────╮
│  An NPU is a term used in the field of computing and networking,           │
│  referring to a type of network protocol or data format.                   │
╰────────────────────────────────────────────────────────────────────────────╯
TTFT 391 ms • 44.3 tok/s • TPOT 22.6 ms/token • 22 → 27 tokens
Goodbye.
```

The answer is wrong, which is not unusual for a 0.6B model quantized to INT4. [eval](#eval) puts a number on how much quantization costs.

### Options {#chat-options}

| Option | What it does |
|---|---|
| `--device auto\|npu\|gpu\|cpu` | Device to run on. Without it, yonga uses your configured default device, then the device the artifact was prepared and probed for. |
| `--max-new-tokens N` | Output token limit per reply. The default comes from the model's chat profile (4096 in the session above). |
| `--temperature X` | Sampling temperature. `0` means greedy decoding. |
| `--top-p X` | Nucleus sampling probability. |
| `--top-k N` | Top-k sampling cutoff. |
| `--system TEXT` | System prompt for the whole session. |
| `--show-reasoning` | Show the model's reasoning blocks instead of hiding them (see below). |

The sampling defaults come from the model's chat profile in the registry and are shown in the startup panel. `chat` does not accept `--json`: a conversation is not one document. For machine-readable output, use `yonga bench --json`.

### Commands inside the chat {#chat-commands}

Type a line starting with `/` to run a command instead of sending a message. `/help` prints this list.

| Command | What it does |
|---|---|
| `/help` | Show the command list. |
| `/reset` | Forget the conversation history (the system prompt is kept). |
| `/clear` | Clear the screen. |
| `/tokens N` | Set the maximum number of tokens in a reply. |
| `/temp X` | Set the sampling temperature, 0.0 to 2.0 (`/temperature` also works). |
| `/top-p X` | Set nucleus sampling, 0.0 to 1.0. |
| `/top-k N` | Set top-k sampling; `0` disables it. |
| `/reasoning on\|off` | Show or hide the model's reasoning block. |
| `/stats on\|off` | Show or hide the metrics footer. |
| `/model` | Show the loaded model, device and profile again. |
| `/exit` | Leave the chat (`/quit`, `/q` and `/bye` also work). |

### Reasoning and thinking mode {#reasoning}

Some models (Qwen3, for example) write a hidden reasoning block before the answer. By default yonga removes it and shows only the answer. With `--show-reasoning`, the reasoning appears in a dim panel above the answer, without its tags. On a model whose chat template has a thinking switch, `--show-reasoning` also turns thinking mode on for the session.

`/reasoning on` only changes what is displayed. If thinking mode is off for the session, the model may not produce any reasoning to show; yonga says so and suggests restarting with `--show-reasoning`.

### The metrics footer {#metrics}

After each reply, `chat` prints a dim line such as:

```text
TTFT 391 ms • 44.3 tok/s • TPOT 22.6 ms/token • 22 → 27 tokens
```

That is the time to first token, the generation speed, the time per output token, and the prompt and reply lengths in tokens. `/stats off` hides it. The terms are explained under [bench](#bench).

### Stopping a reply, leaving {#stopping}

- **Ctrl-C while a reply is being generated** stops it at the next token, drops it together with the message that asked for it, and returns you to the prompt. A second Ctrl-C before it has stopped ends `chat` (exit 60).
- **Ctrl-C or Ctrl-D at the prompt**, `/exit` or the end of piped input leaves the chat.

If the last message failed and you leave right after, `chat` exits with that failure's code instead of 0, so a script can notice it.

!!! tip
    `chat` reads standard input, so you can script a quick check:

    ```bash
    printf 'What is 2+2? Answer briefly.\n/exit\n' | yonga chat Qwen/Qwen3-0.6B
    ```

!!! note
    What you type at the prompt is kept as prompt history (for the arrow keys) under `~/.local/share/yonga/chat-history/`. Replies are not stored. See [Privacy and security](../privacy.md).

## Check a device with probe {#probe}

`probe` compiles a prepared artifact on a device and reports what worked. It reads the tokenizer, compiles the language model, and, where a raw compile cannot decide on its own, builds the real OpenVINO GenAI pipeline and generates a few tokens. On the NPU that is every LLM.

```bash
yonga probe Qwen/Qwen3-0.6B --device npu
```

```text
NPU  ✓ ok
·  language    deferred  579 ms  the raw compile is not conclusive here; the GenAI pipeline decides
✓  tokenizer   ok  28 ms
✓  smoke test  ok  2.0 s  generated 24 token(s)
```

The default is `--device auto`, which tries every candidate device. A device that fails while another passes is reported for information only. `--all-submodels` compiles every submodel, not only the language model. `--json` emits the report as JSON.

### Large models: --quick and --timeout {#probe-quick-timeout}

For a small model the pipeline build takes seconds. For a large model on the NPU it is a full NPU compilation and can take well over ten minutes and several GiB of RAM. Two options shorten that; they cannot be combined:

```bash
yonga probe ARTIFACT --device npu --quick         # never build the pipeline
yonga probe ARTIFACT --device npu --timeout 600   # build it in a child process, stop after 600 s
```

Either can leave the device **inconclusive**: neither working nor failed. When nothing passed and nothing failed outright, `probe` exits **2** (`--json` reports `"inconclusive": true`):

```text
NPU  ! inconclusive
·  language   deferred  643 ms  the raw compile is not conclusive here; the GenAI pipeline decides
✓  tokenizer  ok  27 ms
…
  - Settle it by building the pipeline: yonga probe … --device npu
  - Or bound the wait: yonga probe … --device npu --timeout SECONDS
```

Run it again without `--quick`, or with a longer `--timeout`, to settle it.

| Exit code | Meaning |
|---|---|
| 0 | At least one device passed. |
| 2 | Inconclusive: no device passed and none failed outright. |
| 51 and others | The first definite failure's code, for example 51 when the NPU compiler refuses the model. |

`probe` never switches devices for you. The [probe spec](https://github.com/ooguz/yonga/blob/main/docs/CLI_SPEC.md#11-probe) has the details.

## Measure speed with bench {#bench}

`bench` runs a fixed set of prompts, saves a reproducible JSON result and prints a summary.

```bash
yonga bench Qwen/Qwen3-0.6B --device npu
```

| Option | Default | What it does |
|---|---|---|
| `--suite` | `standard` | `quick` (one short prompt), `standard` (short, medium and long-context prompts) or `long` (adds a longer analysis prompt). |
| `--runs N` | 3 | Measured runs per prompt. |
| `--warmups N` | 1 | Warmup runs per prompt. They are recorded but left out of the averages. |
| `--device` | the artifact's device | `auto`, `npu`, `gpu` or `cpu`. |
| `--json` | | Emit the full result as JSON, with its file path in `result_path`. |

Every run uses greedy decoding and at most 256 new tokens, so results are comparable. This is the standard suite on the NPU of the reference machine (Intel Core Ultra 7 268V, yonga 0.5.0):

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
Saved: …/benchmarks/20261011T170311Z-npu-standard-96ace441.json
```

How to read the columns:

| Column | Meaning |
|---|---|
| In tok / Out tok | Prompt length and generated length, in tokens. |
| TTFT | Time to first token: from handing over the prompt to the first generated token. It covers reading the whole prompt, so it is the delay you notice before an answer starts. |
| tok/s | Generated tokens per second. |
| TPOT | Time per output token after the first one; roughly the inverse of the generation speed. |
| Wall | Total time of one run. |
| Compile | Time to compile and load the model before the first run. |

The JSON file is saved under `~/.local/share/yonga/benchmarks/`. It records the hardware, package versions, device properties, generation settings, the hash of every prompt and every individual run, so a later result can be compared with it fairly. `bench` exits 0 on success, 2 when some runs failed, and 52 when no run produced a measurement.

## Measure what quantization cost with eval {#eval}

`bench` says how fast an artifact is; `eval` says how much its quantization changed the model. It compares the candidate's next-token predictions with those of a higher-precision **reference** artifact of the same weights, reading the same text. Nothing is sampled, so the result is deterministic.

The usual workflow is to prepare a near-lossless INT8 reference on the CPU and compare your INT4 artifact with it:

```bash
yonga prepare Qwen/Qwen3-0.6B --device cpu --profile cpu-int8-asym-ref --yes
yonga list                                     # note both artifact ids
yonga eval INT4_ARTIFACT --reference REF_ARTIFACT
```

It reports three things, in total and per passage of its built-in corpus (prose, technical writing, Python, arithmetic, dialogue, Turkish and more):

| Metric | Meaning |
|---|---|
| Mean KL | Mean KL divergence from the reference, in nats. Lower is closer; 0 is identical. This is the figure to rank artifacts by. |
| Top-1 agreement | How often both pick the same most likely next token. |
| Perplexity | Each artifact's perplexity on the corpus, and the candidate's change against the reference. |

For example, on Qwen3.5-9B against an INT8 reference, the project measured ([PROFILE_COMPARISON.md](https://github.com/ooguz/yonga/blob/main/docs/PROFILE_COMPARISON.md#8-output-quality)):

| | `npu-int4-sym-cw` | `gpu-int4-asym-g128` |
|---|---|---|
| Mean KL (nats) | 0.1972 | 0.0799 |
| Top-1 agreement | 81.82 % | 86.80 % |
| Perplexity (reference 4.655) | 5.700 (+22.44 %) | 4.983 (+7.05 %) |

Options:

- `--device cpu` (the default) runs at full precision, so the difference you measure is the quantization's. `--device gpu` measures what the GPU actually computes. There is no `--device npu`: an NPU artifact is evaluated on the CPU like any other.
- `--corpus FILE` reads your own text (`.json` in the shipped shape, or plain `.txt` split on blank lines).
- `--allow-different-sources` compares artifacts converted from different repositories holding the same model, such as a GGUF release and the original weights. yonga records that you vouched for the lineage.
- `--json` emits the result as JSON.

Use `cpu-fp16-ref` as the reference instead when the model fits in memory at 16 bits. The result is saved under `~/.local/share/yonga/evaluations/` and the output ends with `Saved: PATH`. More in the [eval spec](https://github.com/ooguz/yonga/blob/main/docs/CLI_SPEC.md#13a-eval).

## Publish to the Hugging Face Hub {#publish}

`publish` uploads a prepared artifact to the Hub with a model card generated from its manifest: lineage with revisions, export settings, repairs, which devices passed a probe, and every file's SHA-256. It only ever runs when you call it; no other command uploads anything.

```bash
yonga publish ARTIFACT --repo you/model-openvino-int4 --dry-run   # card and upload list, no upload
yonga publish ARTIFACT --repo you/model-openvino-int4             # private repository, asks first
```

- It needs a Hugging Face token with write access to the namespace (`hf auth login`, or `HF_TOKEN`); without one it stops with exit 21 (`--dry-run` only warns).
- `--repo` is always required. The repository is private unless you pass `--public`; an existing repository is only written with `--update`.
- Licenses are checked on the Hub first. A license that forbids derivatives is refused; one yonga cannot classify needs `--license-reviewed`, your statement that you read it. `--yes` never implies it.
- Local paths, your login name and hostname are not uploaded.

See the [publish spec](https://github.com/ooguz/yonga/blob/main/docs/CLI_SPEC.md#14a-publish) for every check.

## Next steps {#next-steps}

- Serve the model to other tools: [OpenAI-compatible server](serve.md).
- Something failed: [Troubleshooting](../troubleshooting.md) and [Exit codes](../reference/exit-codes.md).
- Every option of every command: [Command reference](../reference/cli.md).
