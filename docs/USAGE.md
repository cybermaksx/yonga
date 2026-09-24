# USAGE.md — driving and testing yonga

A practical tour of what the tool does today, how to exercise each command, and how to check the
behaviours that are easy to claim and hard to verify. Everything here was run on the reference
host; where output is quoted it is real output, lightly trimmed for width.

For installation and the Intel NPU driver situation see [`INSTALL.md`](INSTALL.md). For diagnostic
codes and remediation see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

---

## 0. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[openvino,export,dev]"
```

`openvino` is needed to probe, chat and benchmark. `export` (optimum-intel, nncf, transformers,
torch) is needed to *convert* a model. Without either, `doctor`, `inspect`, `list`, `logs` and
`registry` still work and `doctor` tells you what is missing.

Every command accepts `--json`, `--verbose`, `--debug`, `--offline`, `--config`, `--data-dir`,
`--cache-dir` — with one exception: `chat` rejects `--json`, because a conversation is not one
document.

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
expected download, the estimated temp requirement, the target directory, whether remote code is
needed, and which repairs are expected. Then it runs seven stages: inspect → resolve → preflight →
download → convert → validate/repair → probe.

Useful flags: `--device auto|npu|gpu|cpu`, `--profile ID`, `--source REPO`, `--revision REV`,
`--force`, `--no-repair`, `--yes`, `--trust-remote-code`, `--ignore-exporter-requirements`.

### The full reference case

```bash
yonga prepare HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive --device auto
```

This exercises nearly everything: a GGUF-only source that cannot be converted, a registry-declared
third-party derivative that must be confirmed, a 17.5 GiB download, a VLM with seven submodels, a
tokenizer defect that needs repairing, and a device the registry rules out. Expect ~30 minutes and
~50 GiB of scratch space.

> It needs `transformers==5.2.*`. On a newer release the export fails inside optimum-intel with
> `ImportError: cannot import name 'Qwen3_5DynamicCache'`. yonga now stops at preflight and tells
> you, rather than downloading 17.5 GiB first. Note that a newer transformers still *loads* the
> config and reports `model_type: qwen3_5` — a successful config load is not evidence the export
> will work.

---

## 4. Use what you prepared

```bash
yonga list                          # what is ready, with profile, size, last benchmark
yonga list --clean                  # also reclaim stale staging/temp directories
yonga probe ARTIFACT --device gpu   # compile each submodel, report what worked
yonga chat ARTIFACT --device gpu
yonga bench ARTIFACT --device gpu --suite quick --runs 3 --warmups 1
yonga remove ARTIFACT               # artifact + its compile cache, nothing else
```

`chat` supports `/help /reset /clear /tokens N /temp X /top-p X /top-k N /reasoning on|off
/stats on|off /model /exit`, renders Markdown, and shows a metrics footer:

```
TTFT 276 ms • 18.2 tok/s • TPOT 55.0 ms/token • 20 → 159 tokens
```

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
question about `Qwen/Qwen3-0.6B` and the two rows swap: `arch-qwen3_5` is rejected at specificity
60 and `arch-generic-text-generation` wins with `model identity qwen3 is listed`. There are further
sections below it — model rule, profile, chat profile, derivative sources and candidates, and the
per-device verdict with the rule that produced each one.

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
and your home directory, hostname and login name are rewritten to `~`, `<host>` and `<user>`. It only
reads yonga's own directories. Still read it before you share it. `--output PATH` chooses where it
goes; otherwise it lands in the current directory as `yonga-report-<job>.tar.gz`.

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

On the NPU a probe builds the real GenAI pipeline whenever the raw compile is inconclusive, which is
every LLM. For a small model that takes seconds; for a 9B model it is a full NPU compilation and can
take well over ten minutes and several GiB of RAM.

In `--device auto` yonga *may* fall back — but only for a compatibility failure. A broken
environment (permission denied, missing driver) stops instead of being hidden behind another device.

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
pytest                       # 1824 of 1843 tests; network and hardware markers excluded by default
ruff check . && ruff format --check .
mypy src tests
```

Markers: `network`, `openvino`, `intel_gpu`, `intel_npu`, `slow`, `large_model`.

```bash
pytest -m openvino           # 25 tests, needs OpenVINO (the 17 CPU ones plus the hardware ones)
pytest -m network            # 11 tests, hits the real Hub
pytest -m intel_gpu          # 3 tests on a real Intel GPU
pytest -m intel_npu          # 5 tests on a real Intel NPU
```

The hardware tests run against real devices and skip, with the command that fixes it, when the
device or their artifact is missing. On the GPU they cover generation with metrics, compile-cache
reuse, and the probe routing the tokenizer to CPU. On the NPU they cover node access, a raw compile
being deferred rather than taken as the verdict, a pipeline failure coming out typed instead of as
exit 99, generation with string-valued properties, and yonga's own probe passing an NPU LLM.

**Be aware of what is not covered.** `slow` and `large_model` still select zero tests, and CI has no
Intel hardware, so the hardware suites only run where someone runs them. The large-model results in
this repository — the Qwen3.5 reference case above all — are still hand-run.

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
~/.cache/yonga/{tmp,compile,metadata}    managed temp, compile cache
~/.local/share/yonga/artifacts/          prepared artifacts, each with .yonga/manifest.json
~/.local/share/yonga/{logs,benchmarks,chat-history}
```

Source models stay in the normal Hugging Face cache and are never copied or modified. Settings
follow CLI → `YONGA_*` env → config file → defaults.
