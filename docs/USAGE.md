# USAGE.md — driving and testing yonga

A practical tour of what the tool does today, how to exercise each command, and how to check the
behaviours that are easy to claim and hard to verify. Everything here was run on the reference
host; where output is quoted it is real output, lightly trimmed for width.

For installation and the Intel NPU driver situation see [`INSTALL.md`](INSTALL.md). For diagnostic
codes and remediation see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

---

## 0. Setup

To use yonga, install it from PyPI with `pip install "yonga[openvino,export,serve]"` or take the
AppImage; [`INSTALL.md`](INSTALL.md) covers both. This tour also runs the test suite (section 8),
so it starts from a clone of the repository with the development tools:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[openvino,export,serve,dev]"
```

`openvino` is needed to probe, chat and benchmark. `export` (optimum-intel, nncf, transformers,
torch) is needed to *convert* a model. `serve` (FastAPI, uvicorn) is needed for `yonga serve`.
Without `openvino` and `export`, `doctor`, `inspect`, `list`, `logs` and `registry` still work and
`doctor` tells you what is missing.

Global options go before the command name, e.g. `yonga --offline --debug prepare …`: `--config`,
`--data-dir`, `--cache-dir`, `--offline`, `--verbose`/`-v`, `--debug`, `--json`. Most commands also
accept `--json` after the command name (`yonga list --json`). Two long-running commands reject
`--json`: `chat`, because a conversation is not one document, and `serve`, which reports its state
as JSON at `GET /health` instead.

---

## 1. Start here: does this machine work?

```bash
yonga doctor
```

Reports OS, CPU, RAM, package versions, the OpenVINO ABI family, every device OpenVINO enumerates,
the NPU node and driver, storage, and Hugging Face auth. Exit codes: **0** usable, **2** usable
with warnings, **10** no usable runtime.

What to look for:

- **Devices** should list CPU, GPU and NPU by their real names if you have them.
- **ABI family** must say `aligned`. OpenVINO, GenAI and Tokenizers have to share a major.minor.
- **NPU → /dev/accel/accel0** should say `accessible`. If it does not, doctor names the owning
  group (usually `render`) and prints the `usermod` command *for you to run* — yonga never uses
  sudo and never changes your system.
- **Storage → system temp** will warn if `/tmp` is a small tmpfs. That warning is deliberate and
  never fatal: conversions run in a managed disk-backed directory instead. Do not enlarge tmpfs on
  yonga's account.

```bash
yonga doctor --json | jq .status     # machine-readable, exactly one JSON document
yonga doctor --full                  # adds a registry section
```

---

## 2. Find a model, then look before you download

```bash
yonga search "qwen3.5 9b"
yonga search "qwen3.5 9b" --device npu      # one device's column only
```

One Hub request, and no repository files are fetched. Each result is identified the same way
`inspect` identifies it — from the Hub's config summary, or from the GGUF header for a GGUF-only
repository (marked `*`) — and shown with what the registry knows about that architecture on CPU,
GPU and NPU. Read the columns for what they are:

- **Compatibility is per architecture, not per repository.** `lmstudio-community/Qwen3.5-9B-MLX-4bit`
  reports `qwen3_5`, but its weights are MLX, which the exporter cannot convert. The **Format**
  column is there for exactly this, hinted from the Hub's tags; `mlx`, `awq` and `gptq` are warning
  signs.
- `?` means the registry makes no claim — never that the model works.
- `+` marks a repository the registry declares a conversion source for; the footnote names it.

Then `inspect` the one you want: it reads the file list and answers whether *this* repository can be
prepared on *this* host.


```bash
yonga inspect Qwen/Qwen3-0.6B
yonga inspect HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive
```

Reads Hub metadata and small config files only — **never weight shards**. You can prove that:

```bash
du -sb ~/.cache/huggingface           # note the number
yonga inspect Qwen/Qwen3-0.6B
du -sb ~/.cache/huggingface           # identical
```

The second example is the interesting one. That repository is GGUF-only and has **no
`config.json`**, so there is nothing to read an architecture out of. yonga identifies it from the
GGUF header instead and says so:

```
Architecture  qwen3_5
Size          ~16.7 GiB (one model; 38.5 GiB across 5 files in the repository)

Compatibility on this host
Architecture identified from GGUF metadata; this repository has no config.json.
✓  CPU  supported (official)
✓  GPU  supported (official)
✗  NPU  unsupported (tested)  current_npu_compiler_graph_limitation
```

Two things worth noticing. The size states **one usable model** first, because a GGUF repo
publishes several interchangeable quantizations and summing them overstates the download
several-fold. And the provenance line appears by default, not under `--verbose`, because every
device claim below it rests on that identification.

---

## 3. Prepare a model

```bash
yonga prepare Qwen/Qwen3-0.6B --device cpu
```

Prints a plan and stops for confirmation **before** anything expensive. The plan carries the
resolved source and pinned commit, the device decision and why, the profile and quantization, the
expected download (only the files the conversion reads, not the ONNX copies or training runs a
repository may also carry), the estimated temp requirement and artifact size, the target directory,
whether remote code is needed, which Python the export runs in, and which repairs are expected.
Then it runs seven stages: inspect → resolve → preflight → download → convert → validate/repair →
probe.

Re-running `prepare` for an artifact that is already ready reuses it without a question, so it
works in a script without `--yes`. When a question does need answering and there is no terminal,
`prepare` stops with exit 60 and says to re-run with `--yes`. A job that cannot finish — no
OpenVINO runtime, a `--device` the registry rules out for the model, a gated repository without a
token — is refused before anything is downloaded, and the plan marks it **BLOCKED**.

Useful flags: `--device auto|npu|gpu|cpu`, `--profile ID`, `--source REPO`, `--revision REV`,
`--force`, `--no-repair`, `--yes`, `--trust-remote-code`, `--ignore-exporter-requirements`.

### A GGUF-only model

```bash
yonga inspect Qwen/Qwen2.5-0.5B-Instruct-GGUF          # "Direct GGUF conversion  yes"
yonga prepare Qwen/Qwen2.5-0.5B-Instruct-GGUF --device gpu --yes
yonga prepare Qwen/Qwen2.5-0.5B-Instruct-GGUF --device gpu --gguf-file qwen2.5-0.5b-instruct-q8_0.gguf --yes
```

OpenVINO GenAI imports llama, qwen2 and qwen3 GGUF files directly. yonga reads each candidate
file's header first (about 10 MB) and picks the first whose tensor types the reader can convert.
Here the `Q4_K_M` file is passed over because it holds `Q5_0` tensors, and `Q4_0` is used. The
plan says so. Only that one file is downloaded, and the result is a normal artifact
(`gguf-q4_0-gpu`) for `chat`, `bench`, `serve` and `eval`. The weights keep the GGUF's own
quantization. On this host a GGUF import measured slightly further from the original weights than
an optimum export at the same precision (`PROFILE_COMPARISON.md` §9), and it does not run on the
NPU.

### The full reference case

```bash
yonga prepare HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive --device auto
```

This exercises nearly everything: a GGUF-only source that cannot be converted, a registry-declared
third-party derivative that must be confirmed, a 17.5 GiB download, a VLM with seven submodels, a
tokenizer defect that needs repairing, and a device the registry rules out. Expect ~30 minutes and
~50 GiB of scratch space.

> It needs `transformers==5.2.*`. On a newer release the export fails inside optimum-intel with
> `ImportError: cannot import name 'Qwen3_5DynamicCache'` — even though a newer transformers still
> *loads* the config and reports `model_type: qwen3_5`, so a successful config load is not evidence
> the export will work. yonga does not ask you to downgrade anything: the plan shows an **Export
> environment** row, and the export runs in an isolated environment it builds under
> `~/.cache/yonga/envs/` (next section).

### Managed export environments

When the installed packages cannot meet a model's exporter requirements, the export runs in an
isolated environment instead of failing. It mirrors the packages yonga runs with and changes only
the failing pin; torch comes from PyTorch's CPU index. It is built once, at preflight and before
the model download, verified, and reused by later jobs. The artifact's manifest records which
environment exported it (`export.environment_key`).

```bash
yonga env list               # environments and the disk they use
yonga env remove KEY         # delete one; rebuilt if a later export needs it
yonga prepare MODEL --no-managed-env                 # refuse instead (exit 10, names the pin)
yonga prepare MODEL --ignore-exporter-requirements   # export in *your* environment anyway
```

`--yes` approves building an environment. It never implies `--ignore-exporter-requirements`.

### Retrying after a failure

Downloads resume through the Hugging Face cache. An export interrupted part-way restarts: the
upstream exporter cannot continue from the middle. But an export that *finished* is kept when a
later stage fails (a device probe, validation, Ctrl-C during the smoke test), and retrying the same
command reuses it — the plan shows `Export  reused from failed job …` and stages 4–5 are skipped.

```bash
yonga prepare MODEL --device cpu            # Ctrl-C during 7/7 Probing runtime
yonga prepare MODEL --device cpu            # adopts the finished export, validates and probes
yonga prepare MODEL --device cpu --fresh-export   # export again regardless
```

Only an identical request qualifies: same source revision, profile, task and exporter packages (or
managed environment). Every file is checked first; anything changed is discarded and the export runs.

---

## 4. Use what you prepared

```bash
yonga list                          # what is ready, with profile, size, last benchmark
yonga list --clean                  # also reclaim stale staging/temp directories
yonga probe ARTIFACT --device gpu   # compile each submodel, report what worked
yonga chat ARTIFACT --device gpu
yonga bench ARTIFACT --device gpu --suite quick --runs 3 --warmups 1
yonga remove ARTIFACT               # artifact + its compile cache, nothing else
yonga cache list                    # compile cache entries: size, last use, owner, status
yonga cache prune                   # remove orphaned and stale-runtime entries (asks first)
```

The compile cache (`~/.cache/yonga/compile`) holds the blobs OpenVINO compiled for each artifact,
device and runtime version, so a second load is fast. It is not cleaned up on its own: an artifact
deleted by hand, one prepared in another `--data-dir`, or an OpenVINO upgrade all leave entries
behind. `yonga cache list` shows each entry's status (`in-use`, `orphaned`, `stale-runtime`,
`unattributed`) and `yonga cache prune` removes the orphaned and stale-runtime ones after showing
what it will delete. `--dry-run` only shows; `--older-than 30` also drops entries unused for 30
days, `--unattributed` also drops entries no artifact can be traced to (typically ones made by
yonga 0.3.0 and earlier, which wrote no ownership record), and `--all` empties the cache. Nothing outside the compile cache is touched,
and a pruned entry only costs a recompile on the next load. `yonga doctor` warns when the
reclaimable part reaches 10 GiB, or 25 % of the free space on the cache filesystem.

`chat` supports `/help /reset /clear /tokens N /temp X /top-p X /top-k N /reasoning on|off
/stats on|off /model /exit`, renders Markdown, and shows a metrics footer:

```
TTFT 276 ms • 18.2 tok/s • TPOT 55.0 ms/token • 20 → 159 tokens
```

### Measure what the quantization cost

`bench` says how fast an artifact is; `eval` says how much its quantization changed the model. It
needs a higher-precision artifact of the same weights to compare against:

```bash
yonga prepare MODEL --device cpu --profile cpu-int8-asym-ref --yes   # near-lossless reference
yonga eval ARTIFACT --reference REF_ARTIFACT                         # CPU, full precision
yonga eval ARTIFACT --reference REF_ARTIFACT --device gpu            # what the GPU computes
```

It reports mean KL divergence (lower is closer; 0 is identical), top-1 agreement and perplexity,
in total and per passage (prose, code, arithmetic, Turkish, and more). Use `cpu-fp16-ref` as the
reference when the model fits in memory at 16 bits. On `Qwen/Qwen3-0.6B`, INT8 against FP16 measured
KL 0.0074, so INT8 is a fair stand-in when FP16 does not fit.

### Publish it

```bash
yonga publish ARTIFACT --repo you/model-openvino-int4 --dry-run   # card + upload list, no upload
yonga publish ARTIFACT --repo you/model-openvino-int4             # private repo, asks first
yonga publish ARTIFACT --repo you/model-openvino-int4 --public --yes
```

The model card is generated from the manifest. It says the artifact is a quantized conversion,
not a fine-tune, and lists the lineage with revisions, the export settings, every repair with
digests, which devices passed a probe (and which failed or were never tested), and every file's
SHA-256. The licenses of both the requested model and the converted source are checked on the Hub
first. A license that forbids derivatives refuses. One yonga cannot classify needs
`--license-reviewed`, your statement that you read it. Local paths, your login name and hostname
never leave the machine.

### Serve it to other tools

It needs the `serve` extra (installed above; the AppImage includes it). Then:

```bash
yonga serve ARTIFACT --device gpu              # http://127.0.0.1:8000/v1, model loaded at startup
curl -s localhost:8000/v1/models
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"ARTIFACT","messages":[{"role":"user","content":"Hi"}],"stream":true}'
```

Any OpenAI client works with `base_url="http://127.0.0.1:8000/v1"`. The model answers to the name
you served it under, its artifact id, and its source repository. It is the same session `chat`
opens: same chat profile, generation defaults and hidden reasoning, including while streaming. The
client owns the conversation and sends all of it each time; nothing is kept between requests.
Requests are answered one at a time, and a streaming client that disconnects stops its generation.

`stop` takes up to 4 sequences, each at most 256 characters, as in OpenAI's API. yonga matches them
on the visible answer, so a stop such as `"\n\n"` cannot fire inside a hidden reasoning block and
leave an empty answer; with `--show-reasoning` the reasoning block is part of the visible answer, so
a stop sequence can end it there. The answer ends before the first stop sequence to complete, and
without the whitespace in front of it, with `finish_reason` `stop`, streamed or not, and generation
stops there. `n` above 1, penalties, JSON response formats and the legacy `functions` are still
refused with a 400.

Tools work on the families the registry has seen calling them on the reference host: Qwen2.5,
Qwen3 (not its MoE models), SmolLM3 and Granite (`yonga registry explain MODEL` shows "Tool
calling" under the chat profile). Send OpenAI `tools`; the answer comes back as
`message.tool_calls` with `finish_reason` `tool_calls`, and the follow-up carries the assistant's
`tool_calls` and a `tool` message with each result, exactly as with OpenAI:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
tools = [{"type": "function", "function": {"name": "get_weather",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
messages = [{"role": "user", "content": "Weather in Ankara?"}]
reply = client.chat.completions.create(model="ARTIFACT", messages=messages, tools=tools)
call = reply.choices[0].message.tool_calls[0]
messages += [reply.choices[0].message,
             {"role": "tool", "tool_call_id": call.id, "content": '{"celsius": 21}'}]
print(client.chat.completions.create(model="ARTIFACT", messages=messages, tools=tools)
      .choices[0].message.content)
```

Any other model refuses `tools` with `serve.request.tools_unsupported` rather than pretend:
Llama-3.2-1B, LFM2-350M, gemma-3-1b and OLMo-2-1B were tried and did not write usable calls.
`tool_choice` may be `auto` or `none`; `required` or a named function is refused, since forcing a
call would need constrained decoding. `parallel_tool_calls: false` keeps the first call. Streamed,
text arrives until a call could begin and each call then arrives whole, so the stream pauses while
the model writes it. An answer that looks like a call but is not a valid one (broken JSON, a
function you did not offer) comes back as plain text, never as a guessed call. A call the model
only drafted while reasoning is never made, also with `--show-reasoning`, which then shows the
opening reasoning block in front of the content. Small models call
tools unreliably; the offered schemas also lengthen every prompt, which matters on the NPU, whose
prompt length is fixed when the model is compiled.

A VLM artifact (Qwen2.5-VL, Qwen3.5) also takes images, as OpenAI `image_url` content parts holding
a `data:` URL; an LLM artifact refuses them with `serve.request.images_unsupported`:

```bash
IMG=$(base64 -w0 photo.jpg)
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d @- <<EOF
{"model":"ARTIFACT","messages":[{"role":"user","content":[
  {"type":"text","text":"What is in this picture?"},
  {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,$IMG"}}]}]}
EOF
```

Only `data:image/png`, `jpeg`, `webp` and `gif` base64 URLs are accepted. yonga never fetches an
image, so an `http(s)://` or `file://` URL is refused with a 400, and there is no option to allow
it: a server that fetched what its clients named could be made to reach anything on its host or
network. At most 4 images per request, each at most 4096 x 4096 pixels; a larger image is refused,
not scaled down, so scale it before sending. Images count towards the 8 MiB body limit, and base64
adds a third, so raise it with `--max-body-mb` (up to 64) for larger photos.

Images go with the last user message only. OpenVINO GenAI cannot attach an image to an earlier turn
of a conversation, so an image in an earlier message is refused with
`serve.request.images_earlier_turn`. Chat front ends such as Open WebUI resend the conversation's
earlier images with every follow-up question; for them, start the server with
`--carry-images-forward`. Earlier images then move onto the newest message, with a line telling
the model they were attached earlier. The model no longer knows which words each image went with,
so a question about "the first picture I sent" may be answered less well; that is the trade.

Binding anything but loopback (`--host 0.0.0.0`) prints a warning, and a stronger one when no key
is set. Set `YONGA_API_KEY` in the environment whenever you bind beyond loopback: every endpoint
except `/health` then requires `Authorization: Bearer <key>`. The key is read from the environment
only, so it never shows up in the process list.

Requests are bounded. A body over 8 MiB (or `--max-body-mb`) is refused with 413 and error code
`request_too_large`, counted on the bytes received, so a chunked upload without a `Content-Length` is held to it too.
More than 4096 messages in one request is a 400 with `param` set to `messages`. With a key set,
authentication runs first, so a client without the key is refused before its body is read. Error
responses use OpenAI's shape, `{"error": {"message", "type", "param", "code"}}`; a failure in the
middle of a stream arrives as a final `data: {"error": ...}` event instead. `docs/CLI_SPEC.md`
section 14 lists the codes.

`bench` writes a reproducible JSON result under `~/.local/share/yonga/benchmarks/` recording the
hardware, package versions, device properties, generation settings and their fingerprint, the exact
prompt hashes, and per-run metrics with warmups excluded from the aggregates.

---

## 5. Ask the registry why

```bash
yonga registry validate
yonga registry explain MODEL --device npu
yonga registry paths
```

`explain` is the one to reach for when a decision surprises you. It shows every candidate rule, its
specificity and priority, which ones were rejected **and why**, and which won. For the GGUF-only
repository from section 2, the architecture section reads:

```
    Rule                    Kind           Spec   Prio   Outcome
✓   arch-qwen3_5            architecture     50    100   winner  matched alias 'qwen35' from
                                                         GGUF metadata (no config.json)
✗   arch-generic-text-gen…  architecture     60     10   rejected  architecture/model type
                                                         <unknown> is not listed
```

Specificity is not fixed per rule: `arch-qwen3_5` scores 50 here because it matched on a GGUF alias
rather than on a `config.json` identity, which is the weaker evidence of the two. Ask the same
question about `Qwen/Qwen3-0.6B` and the picture changes: `arch-qwen3_5` is rejected at
specificity 60, `arch-qwen3` wins with `model identity qwen3 is listed`, and
`arch-generic-text-generation` matches on the same identity at the same specificity but loses on
priority (10 against 100). There are further
sections below it — model rule, profile, chat profile, derivative sources and candidates, and the
per-device verdict with the rule that produced each one.

### Newer registry knowledge between releases

```bash
cp publisher.pub ~/.config/yonga/registry-keys/      # trust a signing key: your decision, by hand
yonga registry update --url https://example.org/yonga/bundle.json
yonga registry status                                # which bundle, which key, still valid?
yonga registry reset                                 # back to the built-in rules
```

A bundle is verified completely before it is installed, and again every time it is used. It must
be signed by a trusted key, current (its timestamps carry a UTC offset), not older than the
installed one, and plain YAML naming only repairs yonga already contains. Its export arguments
cannot override what yonga decides itself: a rule that passes the model, task, remote-code, cache,
token or quantization options to the exporter fails to load. A bundle that stops verifying is
ignored, never trusted.

Note that `registry validate` checks *structure and internal consistency*, not that device names or
weight formats are ones OpenVINO actually supports — an override declaring `device: MARS` validates
cleanly, by design, so a future device needs no code change.

---

## 6. When something fails

```bash
yonga logs                      # recent jobs
yonga logs JOB_ID               # one job as a table
yonga logs JOB_ID --raw         # the JSON Lines file
yonga logs JOB_ID --bundle      # a .tar.gz to attach to a bug report
```

Every job writes `~/.local/share/yonga/logs/<job>.jsonl` with the full raw error preserved, even
when the terminal shows a short explanation. `--debug` adds the raw exception to the terminal.

`--bundle` packs the job log, its state, the artifact's manifest and validation record when there
is one, and a `doctor --json` host report into a single archive, with a `README.txt` listing what
went in. It is built to be shareable, including publicly: every file is redacted again for secrets,
and your home directory, hostname and login name are rewritten to `~`, `<host>` and `<user>`. The
Hugging Face account name `doctor` reports for your token becomes `<hf-user>` wherever it appears.
It only reads yonga's own directories. Still read it before you share it. `--output PATH` chooses
where it goes; otherwise it lands in the current directory as `yonga-report-<job>.tar.gz`.

---

## 7. Exercising the behaviours that matter

These are the properties that are easy to assert and worth checking yourself.

**Hidden reasoning stays hidden.** Chat a model that emits `<think>` blocks. The default panel
should show only the answer; `--show-reasoning` should show the chain of thought.

```bash
printf 'What is 2+2? Answer briefly.\n/exit\n' | yonga chat ARTIFACT --device gpu
printf 'What is 2+2? Answer briefly.\n/exit\n' | yonga chat ARTIFACT --device gpu --show-reasoning
```

This is defence in depth, and you can see why it needs to be: the Qwen3.5 fine-tune still emits a
reasoning block even when `enable_thinking=false` is pushed through its own chat template. The
template request did not hold; the wrapper stripper caught it. If a template opens the block in the
*prompt* and the model never closes it, yonga hides the whole reply and tells you how to see it.

**`--yes` cannot enable remote code.** Security-sensitive confirmations are never auto-approved:

```bash
yonga prepare tiiuae/falcon-rw-1b --device cpu --yes     # exit 22, tells you to pass the flag
```

**An explicit device never silently falls back.**

```bash
# An artifact the NPU cannot compile -- for example the Qwen3.5 one prepared with
# --profile npu-int4-sym-cw: exit 51, classified npu.compiler.unsupported_graph.qwen35,
# recommends GPU, and does not switch to it.
yonga probe ARTIFACT --device npu
```

On the NPU a probe builds the real GenAI pipeline whenever the raw compile is deferred to the
pipeline, which is every LLM. For a small model that takes seconds; for a 9B model it is a full NPU
compilation and can take well over ten minutes and several GiB of RAM. Two options shorten that:

```bash
yonga probe ARTIFACT --device npu --quick         # raw compile only; a deferred device is inconclusive
yonga probe ARTIFACT --device npu --timeout 600   # build in a child process, stop it after 600 s
```

`--quick` never builds the pipeline. `--timeout` builds it in a child process that can be stopped,
because a native compile cannot be interrupted from inside yonga; a build stopped there is
`inconclusive`, not a compiler verdict. An inconclusive device is neither working nor failed: when
nothing passed and nothing failed outright, the probe exits 2, and `--json` says
`"inconclusive": true`. Run it again without `--quick`, or with a longer `--timeout`, to settle it.

In `--device auto` yonga *may* fall back — but only for a compatibility failure. A broken
environment (permission denied, missing driver) stops instead of being hidden behind another device:
`prepare` exits 10 for it, where a device that cannot run the model exits 50.

**Nothing leaks secrets.**

```bash
HF_TOKEN=hf_notarealtoken12345 yonga --debug prepare Qwen/Qwen3-0.6B --device cpu --yes --force
grep -r hf_notarealtoken12345 ~/.local/share/yonga/ ; echo "exit $? (1 = not found, good)"
```

**Incomplete artifacts are never listed as ready.** Interrupt a `prepare` with Ctrl-C and run
`yonga list` — the partial build does not appear, because an interrupted export is still sitting
under the hidden `.staging` root, which `list` never walks. Anything that does survive into the
artifact tree without a parsable, `ready` manifest is shown with **Ready ✗ no** so you can clean it
up; nothing can be mistaken for a usable model. `yonga list --clean` retires the leftovers.

---

## 8. Running the test suite

```bash
pytest                       # network, hardware, slow and large_model markers excluded by default
ruff check . && ruff format --check .
mypy src tests
```

Markers: `network`, `openvino`, `intel_gpu`, `intel_npu`, `slow`, `large_model`.

```bash
pytest -m openvino           # needs OpenVINO: the CPU tests plus the hardware ones
pytest -m network            # hits the real Hub
pytest -m intel_gpu          # on a real Intel GPU
pytest -m intel_npu          # on a real Intel NPU
pytest -m slow               # the real GGUF import, from a cached GGUF file
pytest -m large_model        # VLM chat, needs a prepared VLM artifact
```

`pytest --collect-only -q -m MARKER` prints what a marker selects today.

The hardware tests run against real devices and skip, with the command that fixes it, when the
device or their input is missing; they never download anything. On the GPU they cover generation
with metrics, compile-cache reuse, the probe, streaming, `serve` (plain and SSE) and GPU logits
against CPU. On the NPU they cover node access, a raw compile being deferred rather than taken as
the verdict, a pipeline failure coming out typed instead of as exit 99, generation with
string-valued properties, yonga's own probe passing an NPU LLM, streaming with a second turn, and
`probe --quick` and `--timeout` coming back inconclusive rather than passed or failed. `slow`
selects `tests/hardware/test_gguf_import.py`, the real GGUF import from a cached GGUF file;
`large_model` selects `tests/hardware/test_vlm.py`, text chat through the VLM pipeline on the GPU and an image sent through `serve`.
`docs/TESTING.md` section 6 lists each file and the input it needs.

**Be aware of what is not covered.** CI has no Intel hardware and runs only `tests/unit` and
`tests/integration`, so the hardware, `slow` and `large_model` tests run only on a host with the
devices, when someone runs them there. No test runs the full Qwen3.5 flow in "The full reference
case" above; it is still run by hand.

Some tests need artifacts that are generated rather than committed, so a clone does not carry binary
weights:

```bash
python tools/make_test_artifact.py        # tiny random Llama, ~7 MB, needs network once
python tools/make_test_artifact.py --npu  # Qwen3-0.6B with the NPU profile, ~400 MB of IR
```

Without it those tests skip with a message telling you to run exactly that.

---

## 9. Where things live

```
~/.config/yonga/config.toml              settings, and registry-overrides/
~/.cache/yonga/{tmp,compile,metadata}    managed temp, compile cache (yonga cache list)
~/.local/share/yonga/artifacts/          prepared artifacts, each with .yonga/manifest.json
~/.local/share/yonga/{logs,benchmarks,chat-history}
```

Source models stay in the normal Hugging Face cache and are never copied or modified. Settings
follow CLI → `YONGA_*` env → config file → defaults.
