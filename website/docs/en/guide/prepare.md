# Preparing models

`yonga prepare` turns a Hugging Face repository into a prepared artifact: an OpenVINO model that has
been downloaded, converted, validated, repaired if needed, and shown to compile and generate tokens
on your device. This page covers the whole path: finding a model, reading the plan, choosing a device
and a profile, answering the questions yonga asks, special cases such as gated, GGUF and remote-code
repositories, working offline, disk space, and what happens when you re-run or cancel.

## Look before you download { #search-and-inspect }

To find candidates, search the Hub:

```bash
yonga search "qwen3 0.6b"
```

```text
Model                                  Arch        CPU   GPU   NPU   Format           DL
────────────────────────────────────────────────────────────────────────────────────────
Qwen/Qwen3-0.6B                        qwen3        ✓     ✓     ✓    safetensors   30.6M
Qwen/Qwen3-Embedding-0.6B              qwen3        ✓     ✓     ✓    safetensors    8.5M
…
✓ supported   ✗ unsupported   ! experimental   ? no registry claim
```

The device columns are what the registry knows about each *architecture*, not a check of a
repository's files. `?` means the registry makes no claim, never that the model works. Add
`--device npu` to show one device only, and `--limit N` (1 to 100, default 20) for more or fewer
results.

Then inspect the repository you want. `inspect` reads Hub metadata and small configuration files
only, never weight files, and answers whether *this* repository can be prepared on *this* machine:

```bash
yonga inspect Qwen/Qwen3-0.6B
```

```text
Model         Qwen/Qwen3-0.6B
Revision      c1899de289a0
License       apache-2.0
…
Size          ~1.4 GiB total weights
Parameters    751,632,384  Hub count; includes embeddings and any vision tower

Compatibility on this host
✓  CPU  supported (tested)
✓  GPU  supported (tested)
✓  NPU  supported (tested)

Preparation
Direct OpenVINO source  no
Safetensors source      yes  Qwen/Qwen3-0.6B
Known derivative        no  No derivative source is declared for this repository.
```

`--files` also lists the repository's files, and `--revision` inspects a branch, tag or commit.

## Prepare with the defaults { #prepare }

To prepare a model, run:

```bash
yonga prepare Qwen/Qwen3-0.6B
```

yonga first prints a **preparation plan**. If the plan lists anything under **Confirmation
required**, yonga asks about each item before anything is downloaded; if it lists nothing, the job
goes straight on. It then runs seven stages:

```text
1/7 Inspecting source
2/7 Resolving compatibility
3/7 Preflight
4/7 Downloading
5/7 Converting
6/7 Validating / repairing
7/7 Probing runtime
```

A converter that exits without an error is not the end of it: the result is inventoried, checked,
repaired when the registry knows a defect, compiled on the target device and made to generate
tokens. Only then does it become a ready artifact.

This is a real run on the reference machine (an NPU preparation of `Qwen/Qwen3-0.6B` with yonga
0.5.0, trimmed; the run used custom `--data-dir` and `--cache-dir` locations, shortened here to `…`):

```text
╭───────────────────────────── Preparation plan ─────────────────────────────╮
│  Requested model             Qwen/Qwen3-0.6B@c1899de289a04d12100db370d814  │
│                              85cdf75e47ca                                  │
│  Resolved source             Qwen/Qwen3-0.6B                               │
│  Revision                    c1899de289a0                                  │
│  Relationship                original                                      │
│  Source format               safetensors                                   │
│  Architecture                qwen3  rule arch-qwen3                        │
│  Task                        text-generation-with-past                     │
│  Device decision             NPU  supported (tested)  NPU was requested    │
│                              explicitly; yonga never switches device on    │
│                              its own.                                      │
│  Profile                     npu-int4-sym-g128                             │
│                              NPU group-wise INT4; the NPU default below    │
│                              5B parameters                                 │
│  Quantization                int4, group size 128, ratio 1, symmetric      │
│  Chosen because              751.6M parameters (Hugging Face Hub count     │
│                              from safetensors headers) is below 5B:        │
│                              npu-int4-sym-g128 is the NPU profile for      │
│                              that size (evidence: tested).                 │
│  Expected download           1.4 GiB  only the files the conversion reads  │
│  Estimated temp requirement  25 GiB  free 341 GiB in …/cache/tmp/de862650… │
│                              System /tmp is too small; the managed temp    │
│                              directory is used.                            │
│  Estimated artifact          about 430 MiB                                 │
│  Artifact directory          …/data/artifacts/Qwen--Qwen3-0.6B/c1899de289a │
│                              0--npu-int4-sym-g128                          │
│  Remote code                 not required                                  │
│  Export environment          managed export-fbfac3ac91ba                   │
│                              transformers==5.5.4  will be built            │
│                                                                            │
│  Expected repairs                                                          │
│  • ov-tokenizer-precision-string-casing                                    │
│  …                                                                         │
│  Confirmation required                                                     │
│  ·  A managed export environment (transformers==5.5.4) will be built       │
│     under …/cache/envs: up to about 3.0 GiB of packages from PyPI and the  │
│     PyTorch CPU index. It is used only by the export, is reused by later   │
│     jobs, and your own Python environment is not changed.                  │
│  ·  This export is expected to need a known artifact repair:               │
│     ov-tokenizer-precision-string-casing                                   │
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
Hashing 12 files
…

Ready
  Artifact  cc24636a2f49356a
  Path      …/data/artifacts/Qwen--Qwen3-0.6B/c1899de289a0--npu-int4-sym-g128
  Device    NPU
  Profile   npu-int4-sym-g128
  note: The system temporary directory /tmp is too small for this conversion;
the managed directory was used instead.

Chat with it: yonga … chat cc24636a2f49356a
```

The last line is the next command to run. The artifact id (`cc24636a2f49356a` here) is what you pass
to `chat`, `probe`, `bench`, `eval` and `serve`; see [Chat, probe, bench and eval](run.md) and
[OpenAI-compatible server](serve.md). `yonga list` shows every ready artifact.

## Reading the plan { #the-plan }

The rows you will look at most often:

| Row | What it tells you |
|---|---|
| **Requested model**, **Revision** | The repository and the exact commit it was pinned to. Every download in the job uses that commit. |
| **Resolved source**, **Relationship** | Which repository is actually converted. Usually the one you asked for (`original`); it differs for a declared derivative or `--source`. |
| **Device decision** | The device, the registry's verdict for it and why it was chosen. |
| **Profile**, **Quantization**, **Chosen because** | The conversion profile and, in one sentence, why it won. |
| **Expected download** | Only the files the conversion reads; other runtimes' weights and training leftovers in the repository are skipped. |
| **Estimated temp requirement** | Working space the conversion needs, and where it will be. See [Disk space](#disk-space). |
| **Estimated artifact**, **Artifact directory** | The size and location of the result. |
| **Remote code** | Whether the conversion would have to run Python code from the repository. |
| **Export environment** | Which Python runs the export: this environment, or a managed one. See [Managed export environments](#managed-environments). |
| **Expected repairs**, **Known limitations** | Defects the registry expects to fix, and anything else you should know. |
| **Confirmation required** | The questions you are about to be asked. |

When the job cannot finish as planned, the plan shows a **BLOCKED** section saying why, and nothing
is downloaded. These cases are refused before any question is asked: the OpenVINO runtime is not
installed (exit 10), an explicit `--device` the registry rules out for this model (exit 50, with the
recommended device), and a gated or private repository with no Hugging Face token (exit 21).
Exporter requirements that cannot be met under `--no-managed-env` are also marked BLOCKED; that job
stops at preflight, after the questions but still before the download (exit 10). An explicit
`--device` that OpenVINO does not report (exit 50) and remote code without `--trust-remote-code`
(exit 22) stop the job before the download too, without a BLOCKED section. See
[Exit codes](../reference/exit-codes.md).

## Choosing a device { #device }

`--device` takes `auto` (the default), `npu`, `gpu` or `cpu`.

```bash
yonga prepare Qwen/Qwen3-0.6B --device gpu
```

**An explicit device is never switched.** If it cannot run the model, the job stops, explains why,
and prints the command for a device that would work. You decide whether to run it. When the registry
or OpenVINO rules the device out before the download, the exit code is 50; a failure found later,
when the model is compiled on the device, exits with that failure's own code (for example 51 for a
compile error).

**`--device auto`** picks a device from what the registry knows and what OpenVINO reports on this
machine. Devices the registry marks unsupported for the architecture are dropped, the rest are ranked
by the registry's verdict and then in the order GPU, NPU, CPU, and an architecture's recommended
device (backed by official or tested evidence) goes first. If the chosen device then fails its probe
because the model does not fit it, auto mode may move to the next device, and asks you before it
does. A broken environment, such as no permission on the NPU node or a missing driver, is never
hidden behind another device: `prepare` stops with exit 10 so you can fix it. Run `yonga doctor` to
see what is wrong.

[How yonga works](../concepts.md) explains the device policy in more detail, and
[Tested hardware and models](../hardware.md) lists what has been verified.

## Choosing a profile { #profile }

A profile is a named conversion recipe: target device, weight format and quantization settings. By
default yonga chooses one from the model's parameter count:

| Device | Below 5B parameters | 5B and above, or size unknown |
|---|---|---|
| CPU | `cpu-int4-asym-g128-r08` | `cpu-int4-asym-g128` |
| GPU | `gpu-int4-asym-g128-r08` | `gpu-int4-asym-g128` |
| NPU | `npu-int4-sym-g128` | `npu-int4-sym-cw` |

The `-r08` profiles keep 20% of the layers at INT8. The plan's **Chosen because** row says which rule
decided. When the model's `config.json` shows that a profile's group size of 128 cannot divide its
widths, that profile is passed over before the download and the row says why; see
[TROUBLESHOOTING.md](https://github.com/ooguz/yonga/blob/main/docs/TROUBLESHOOTING.md#54-nncf-refuses-the-profiles-group-size).
To list every profile:

```bash
yonga registry profiles
yonga registry profiles --device npu
```

To use a different one, name it:

```bash
yonga prepare Qwen/Qwen3-0.6B --device npu --profile npu-int4-sym-cw
```

An explicit profile always wins over the automatic choice; the plan then also names the profile that
would have been chosen without it. An unknown profile id is refused, with the profiles for that
device and a pointer to `yonga registry profiles`. An explicit profile whose group size cannot
quantize the model (as far as `config.json` shows) is refused before the download, and yonga does
not swap in another one.

`cpu-int8-asym-ref` and `cpu-fp16-ref` are reference profiles, not size-band defaults; yonga only
picks `cpu-int8-asym-ref` on its own when no group-wise CPU profile fits the model. They exist to
prepare a near-lossless reference for measuring quantization quality with `yonga eval`; see
[Chat, probe, bench and eval](run.md). The measurements behind the defaults are in
[PROFILE_COMPARISON.md](https://github.com/ooguz/yonga/blob/main/docs/PROFILE_COMPARISON.md#105-what-the-registry-does-with-it).

## Confirmations and `--yes` { #confirmations }

yonga asks before it does something you might not expect. The plan lists the questions under
**Confirmation required**. They cover:

- building a managed export environment;
- converting a third-party derivative instead of the repository you named;
- a download above 5 GiB;
- applying a known repair to the exported model;
- replacing an existing artifact directory, for example with `--force`;
- in `--device auto`, moving to another device after a failed probe (asked when it happens);
- running remote code (see [below](#remote-code)).

`--yes` (`-y`) answers the ordinary questions with yes:

```bash
yonga prepare Qwen/Qwen3-0.6B --device cpu --yes
```

**Without a terminal** (stdin is not a TTY, as in a script or CI job) and without `--yes`, a question
cannot be asked, so `prepare` stops with exit 60 and prints your command again with `--yes` added.
Real output, trimmed:

```text
✗ Preparation needs a confirmation

Confirmation 'repair' could not be asked, because this session has no terminal to ask in (stdin is
not a terminal): This export is expected to need a known artifact repair:
ov-tokenizer-precision-string-casing

  - Re-run the same command with --yes to approve it without a prompt: yonga … --offline prepare
Qwen/Qwen3-0.6B --device cpu --yes
  - Or run it in a terminal to be asked.
```

Answering "no" at a prompt also cancels the job with exit 60.

!!! warning "What `--yes` never does"
    `--yes` never enables remote code, never answers the remote-code question, and never implies
    `--ignore-exporter-requirements`. Those are deliberate, separate decisions.

## Revisions { #revisions }

By default yonga prepares the repository's default branch, and pins it to its commit before anything
else. To prepare a specific branch, tag or commit, use either form:

```bash
yonga prepare Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca
yonga prepare Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca
```

If you give both, `--revision` wins. The artifact directory is named after the first 12 characters
of the pinned commit and the profile (`c1899de289a0--npu-int4-sym-g128`), so different revisions of
the same model live side by side.

## Gated and private models { #gated }

Some repositories require you to accept the owner's conditions and to be logged in. `yonga inspect`
usually still works, because their metadata is public; downloading is what needs the token. Without a
token, `prepare` refuses before downloading anything (exit 21).

To get access:

1. Open the model page on Hugging Face in a browser and accept its licence or access conditions.
2. Log in with Hugging Face's `hf` command-line tool, or export a token as `HF_TOKEN`:

    ```bash
    hf auth login
    ```

3. Run the same `yonga prepare` command again.

`yonga doctor` shows, in its last section, whether a credential is in use. yonga never prints a token,
and job logs pass through a redactor. More detail is in
[TROUBLESHOOTING.md](https://github.com/ooguz/yonga/blob/main/docs/TROUBLESHOOTING.md#55-gated-or-private-repositories).

## Remote code { #remote-code }

A few repositories can only be converted by running Python code they publish. yonga does not do that
unless you ask for it explicitly. Without the flag, `prepare` stops before any work with exit 22 and
tells you which flag to add.

To allow it, first review the repository's custom code, then run:

```bash
yonga prepare MODEL --trust-remote-code
```

yonga still asks you to confirm in the terminal. This question is security-sensitive: `--yes` never
answers it, and in a session without a terminal it is refused (exit 60). With remote code enabled the
whole repository is downloaded, not only the files the conversion reads.

## GGUF repositories { #gguf }

OpenVINO GenAI can import some GGUF files directly. For a GGUF-only repository, yonga reads the
header of each candidate file first (no full download), picks the first preferred quantization whose
tensor types the importer can read, and downloads only that one file. The plan shows the file under
**GGUF file** and lists the files it passed over and why.

```bash
yonga inspect Qwen/Qwen2.5-0.5B-Instruct-GGUF
yonga prepare Qwen/Qwen2.5-0.5B-Instruct-GGUF --device gpu --yes
```

To choose the file yourself, name it with `--gguf-file`:

```bash
yonga prepare Qwen/Qwen2.5-0.5B-Instruct-GGUF --device gpu --gguf-file qwen2.5-0.5b-instruct-q8_0.gguf --yes
```

A GGUF import keeps the file's own quantization, so profiles do not apply; the artifact is named
after the revision, the file's quantization and the device (for example `<revision>--gguf-q4_0-gpu`). It is a normal artifact
for `chat`, `bench`, `serve` and `eval`, but it does not run on the NPU. With yonga 0.5.0's built-in
registry, GGUF import covers the `llama`, `qwen2` and `qwen3` architectures. When a GGUF repository cannot be imported,
`--source` can point at a safetensors repository of the same model instead; see
[TROUBLESHOOTING.md](https://github.com/ooguz/yonga/blob/main/docs/TROUBLESHOOTING.md#58-a-gguf-repository-cannot-be-imported).

## Converting another repository { #source }

`--source REPO` converts another repository in place of the one you named, for example a safetensors
copy of a model whose own repository cannot be converted. The plan's **Resolved source** and
**Relationship** rows show what is used, and the artifact's manifest records the lineage. When the
registry itself declares a third-party derivative for a model, yonga can propose it in the plan, and
it asks before converting it.

## Working offline { #offline }

If the model is already in your Hugging Face cache, you can prepare it without network access.
Offline mode never contacts the Hub, for metadata either. Turn it on in any of these ways:

=== "Option"

    ```bash
    yonga --offline prepare Qwen/Qwen3-0.6B --device cpu
    ```

    `--offline` is a global option: it goes **before** the command name.

=== "yonga variable"

    ```bash
    YONGA_OFFLINE=1 yonga prepare Qwen/Qwen3-0.6B --device cpu
    ```

=== "Hugging Face variable"

    ```bash
    HF_HUB_OFFLINE=1 yonga prepare Qwen/Qwen3-0.6B --device cpu
    ```

What offline mode needs:

- The revision and its weights must be in the local Hugging Face cache. If you name no revision,
  yonga uses the one the cache records, or the only cached snapshot. If several unrecorded snapshots
  exist, it refuses rather than guess: name one as `MODEL@<sha>`.
- If the export needs a managed environment, it must already exist. Building a new one needs the
  network, so an offline job that needs one stops (exit 20); an existing one is reused.

A model that is not cached stops with exit 20 and tells you how to fill the cache (real output,
trimmed):

```text
✗ Not available offline

Offline mode is active and openai-community/gpt2-xl is not in the local Hugging Face cache
(…/.cache/huggingface/hub), so it cannot be used without network access.

  - Re-run without offline mode (no --offline, and YONGA_OFFLINE / HF_HUB_OFFLINE unset) to populate
the cache: …
  - Or point HF_HOME at a cache that already contains this revision.
```

## Managed export environments { #managed-environments }

Converting a model runs that architecture's modelling code from `transformers`, and some
architectures need a specific release. When the packages next to yonga cannot meet a model's
exporter requirements, or there is no export toolchain at all (the AppImage, or a pip install without
the `export` extra), the export runs in an isolated Python environment that yonga builds for it
instead of failing.

- It is built at preflight, **before** the model download, so a failed build costs minutes rather
  than a multi-gigabyte download.
- It lives under the cache directory (`~/.cache/yonga/envs/` by default), takes up to about 3.0 GiB,
  and is used only by the export. Your own Python environment is not changed.
- It is verified before first use and reused by later jobs with the same requirements.
- Building one is a confirmation in the plan; `--yes` approves it.

To manage them:

```bash
yonga env list          # environments and the disk they use
yonga env remove KEY    # delete one; it is rebuilt if a later export needs it
```

Two flags change the behaviour:

| Flag | Effect |
|---|---|
| `--no-managed-env` | Never build one. If the installed packages cannot meet the requirements, stop at preflight (exit 10), before the download. |
| `--ignore-exporter-requirements` | Export in your own environment anyway, even though a declared requirement is unmet or the registry knows this export fails with the installed versions. The override is recorded in the manifest. `--yes` never enables it. |

Details: [INSTALL.md, section 6](https://github.com/ooguz/yonga/blob/main/docs/INSTALL.md#6-the-transformers-version-caveat).

## Disk space and the managed temp directory { #disk-space }

A conversion needs working space well beyond the model's size. yonga never converts in `/tmp`: it
uses a managed, disk-backed directory under the cache directory (`~/.cache/yonga/tmp/<job-id>` by
default) and points `TMPDIR`, `TMP` and `TEMP` there for every conversion process. A small or
RAM-backed `/tmp` is therefore only a note, not an error. Do not enlarge it for yonga.

The plan shows three sizes:

| Row | How it is estimated | Where it goes |
|---|---|---|
| Expected download | The files the conversion reads | The Hugging Face cache (`HF_HOME`) |
| Estimated temp requirement | The larger of 20 GiB and 2.5 × the source weights, plus a 5 GiB safety margin | The cache directory |
| Estimated artifact | From the profile and the parameter count | The data directory (`~/.local/share/yonga` by default) |

If the cache filesystem (or the data directory's, for the artifact) does not have room, preflight
stops before the download with exit 30. To use a larger disk, move the cache (and, if you like, the
data) directory:

```bash
export YONGA_CACHE_DIR=/mnt/big/yonga-cache
export YONGA_DATA_DIR=/mnt/big/yonga-data
```

or pass `--cache-dir` and `--data-dir` before the command name. The source model stays in the
normal Hugging Face cache and is never modified there. The temp directory is removed when
the job ends successfully; after a failure it is kept for diagnosis until a later clean-up retires
it (see [Cancelling](#cancel)). Background:
[INSTALL.md, section 5](https://github.com/ooguz/yonga/blob/main/docs/INSTALL.md#5-tmp-tmpfs-and-why-yonga-does-not-ask-you-to-enlarge-it).

## Re-running and reusing { #reuse }

Running `prepare` again for a revision and profile that already has a ready artifact reuses it: no
download, no conversion and no question, so it is safe in a script without `--yes`. The result says
`Reused    an existing artifact at this revision and profile`.

- **Another device.** Reuse is keyed by revision and profile, not by device. If you pass a
  `--device` the artifact has never been probed on, yonga compiles and smoke-tests it there first and
  records the result. If it does not run there, the job stops without switching devices, and the
  artifact stays ready for the devices it already runs on.
- **Start over.** `--force` prepares again, and asks before replacing the existing directory.
- **Retry after a failure.** Downloads resume from the Hugging Face cache. If an earlier job with
  identical inputs finished its export and then failed (or was cancelled) in a later stage, the
  retry checks every file of that export and adopts it: the plan shows an `Export` row naming the
  earlier job, and the download and conversion stages are skipped. `--fresh-export` exports again
  regardless. An export that was itself interrupted always restarts.
- **A changed default.** If a newer yonga picks a different default profile, the new artifact is
  created next to the old one, never over it, and the plan names the `--profile` that would reuse
  the old one.

## Cancelling with Ctrl-C { #cancel }

Press ++ctrl+c++ at any time to stop a preparation. The job ends with exit 60 and is recorded as
`cancelled` in `yonga logs`.

What is kept and what is not:

- **Your existing artifacts are never touched.** A build happens in a hidden staging directory and
  only becomes an artifact by an atomic rename at the end, so a cancelled job cannot corrupt one,
  and `yonga list` never shows a partial build.
- **Downloaded files** stay in the Hugging Face cache, so the next run resumes the download.
- **A finished export** stays in the staging directory, so a retry with the same inputs can reuse it
  (see [above](#reuse)).
- **The managed temp directory** of a cancelled job is removed straight away.
- **An interrupted environment build** is removed; it is never used half-built.

Leftover staging and temp directories of failed or cancelled jobs are retired automatically when a
later job starts (by default after 72 hours, keeping at most 3). `yonga list --clean` runs the same
clean-up on demand.

## When something fails: the job log { #logs }

Every `prepare` run gets a job id (shown at the bottom of the plan) and a log file under the data
directory, `~/.local/share/yonga/logs/<job-id>.jsonl` by default. A failure message ends with its
path. To read it:

```bash
yonga logs                    # recent jobs, newest first
yonga logs JOB_ID             # one job as a table; a unique prefix is enough
yonga logs JOB_ID --raw       # the JSON Lines file as written
yonga logs JOB_ID --bundle    # a redacted .tar.gz to attach to a bug report
```

```text
job       started           command  model                     device  status        took      log
──────────────────────────────────────────────────────────────────────────────────────────────────
c1ec9272  2026-10-11 20:09  prepare  Qwen/Qwen3-0.6B           CPU     ! cancelled  2.8 s  1.8 KiB
f57e7bd4  2026-10-11 20:09  prepare  openai-community/gpt2-xl  CPU     ✗ failed     <1 ms  1.1 KiB
```

The log keeps the raw error text even when the terminal shows a short explanation. To see it in
the terminal as well, add the global `--debug` option (`yonga --debug prepare …`). For what each
diagnostic means and how to fix it, see [Troubleshooting](../troubleshooting.md).

## All `prepare` options { #options }

| Option | What it does |
|---|---|
| `MODEL` | Hugging Face repository id, optionally `MODEL@REVISION`. |
| `--revision REV` | Branch, tag or commit sha. |
| `--device auto\|npu\|gpu\|cpu` | Target device. Default `auto`. |
| `--profile ID` | Conversion profile id (`yonga registry profiles`). |
| `--source REPO` | Convert this repository instead of the requested one. |
| `--gguf-file FILE` | For a GGUF repository, the file to import. |
| `--trust-remote-code` | Allow the conversion to run Python code published in the repository. |
| `--ignore-exporter-requirements` | Export in this environment even though an exporter requirement is unmet. |
| `--no-managed-env` | Never build a managed export environment; refuse instead. |
| `--fresh-export` | Export again instead of reusing a completed export from a failed job. |
| `--yes`, `-y` | Answer ordinary questions with yes. |
| `--force` | Prepare again even if an artifact exists. |
| `--no-repair` | Never modify the exported model. If a repair is required, the job stops (exit 42). |
| `--json` | Print the outcome as one JSON document. |

The full specification is
[CLI_SPEC.md, section 6](https://github.com/ooguz/yonga/blob/main/docs/CLI_SPEC.md#6-prepare), and
every command is summarised in the [Command reference](../reference/cli.md).
