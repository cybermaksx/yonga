# Command reference

Every yonga command and subcommand, with its options and defaults as `yonga <command> --help` prints them in yonga 0.5.1, plus one example each. The guides explain when to use each command; this page is for looking things up. The full behaviour of every command, including its `--json` document, is specified in [CLI_SPEC.md](https://github.com/ooguz/yonga/blob/main/docs/CLI_SPEC.md).

In the option tables, a dash in the **Default** column means `--help` shows no default: the option is off, or yonga decides from the artifact, the registry or the configuration, as the description says. Every command and command group also accepts `--help`, which is not repeated below.

## Global options

```text
yonga [OPTIONS] COMMAND [ARGS]...
```

Local AI model manager for Intel AI PCs (OpenVINO, Optimum Intel, Hugging Face).

| Option | Value | Default | Description |
|---|---|---|---|
| `--config` | `PATH` | `~/.config/yonga/config.toml` | Configuration file to load instead of the default. |
| `--data-dir` | `PATH` | `~/.local/share/yonga` | Where prepared artifacts and logs are kept. |
| `--cache-dir` | `PATH` | `~/.cache/yonga` | Where the managed temp and compile cache live. |
| `--offline` | | off | Never contact the Hugging Face Hub. |
| `--json` | | off | Emit one JSON document on stdout and no decoration. |
| `--verbose`, `-v` | | off | Explain what is happening. |
| `--debug` | | off | Include raw error text in failures. |
| `--version`, `-V` | | | Show the version and exit. |

Global options go **before** the command name:

```bash
yonga --offline --debug prepare Qwen/Qwen3-0.6B
```

`--json` is the one global option that commands also accept after their own name (`yonga list --json`). `chat` and `serve` refuse it with exit 70. The default directories follow `XDG_CONFIG_HOME`, `XDG_DATA_HOME` and `XDG_CACHE_HOME` when those are set; `yonga registry paths` prints the ones in use. Settings are taken from the command line first, then `YONGA_*` environment variables, then the configuration file.

`yonga` with no command prints the command list and exits 70, as does any command line that cannot be parsed (an unknown option, a missing argument, a value of the wrong type). `--help` exits 0. See [Exit codes](exit-codes.md) for every code.

## Commands at a glance

| Command | Purpose |
|---|---|
| [`doctor`](#yonga-doctor) | Check this host: OpenVINO, devices, NPU access, storage and authentication. |
| [`search`](#yonga-search) | Search the Hugging Face Hub and show what the registry knows about each result. |
| [`inspect`](#yonga-inspect) | Describe a model and what preparing it on this host would involve. |
| [`prepare`](#yonga-prepare) | Download, convert, validate and probe a model into a prepared artifact. |
| [`list`](#yonga-list) | List the artifacts prepared on this host. |
| [`remove`](#yonga-remove) | Remove a prepared artifact and its compile-cache entries. |
| [`probe`](#yonga-probe) | Compile a prepared artifact on a device and report what worked. |
| [`chat`](#yonga-chat) | Chat with a prepared artifact, locally. |
| [`serve`](#yonga-serve) | Serve a prepared artifact through an OpenAI-compatible HTTP API. |
| [`eval`](#yonga-eval) | Measure how far CANDIDATE's predictions move from REFERENCE's (KL, top-1, perplexity). |
| [`publish`](#yonga-publish) | Upload a prepared artifact to the Hugging Face Hub with a generated model card. |
| [`bench`](#yonga-bench) | Benchmark a prepared artifact and save a reproducible JSON result. |
| [`logs`](#yonga-logs) | Show the job logs kept on disk, most recent first. |
| [`registry`](#yonga-registry) | Inspect, validate and update the compatibility registry. |
| [`env`](#yonga-env) | List and remove the managed environments exports run in. |
| [`cache`](#yonga-cache) | Show and prune the OpenVINO compile cache. |
| [`self`](#yonga-self) | Put the yonga AppImage on PATH as 'yonga', or take it off again. |

Where a command takes `MODEL_OR_ARTIFACT` or `ARTIFACT`, you can pass an artifact id from `yonga list` (or a unique prefix of at least six characters), the artifact's directory, or a Hugging Face repository id. For a repository id with several prepared artifacts, the read-only commands use the newest ready one; `remove` refuses and lists them.

## yonga doctor

```text
yonga doctor [OPTIONS]
```

Check this host: OpenVINO, devices, NPU access, storage and authentication.

| Option | Value | Default | Description |
|---|---|---|---|
| `--full` | | off | Also report the loaded registry. |
| `--json` | | off | Emit the report as JSON. |

Exits 0 when a runtime and device are usable and nothing needs attention, 2 when usable with warnings, 10 when no OpenVINO runtime or device is usable. [Maintenance](../guide/maintain.md#check-the-host-with-doctor) explains each section.

```bash
yonga doctor --full
```

## yonga search

```text
yonga search [OPTIONS] QUERY
```

Search the Hugging Face Hub and show what the registry knows about each result.

| Argument | Description |
|---|---|
| `QUERY` | Words to search the Hub for. Required. |

| Option | Value | Default | Description |
|---|---|---|---|
| `--limit` | integer, 1–100 | `20` | How many results to show. |
| `--device` | text | – | Show compatibility for one device: `npu`, `gpu`, `cpu` or `auto`. |
| `--json` | | off | Emit the results as JSON. |

```bash
yonga search qwen3 --device npu --limit 10
```

## yonga inspect

```text
yonga inspect [OPTIONS] MODEL
```

Describe a model and what preparing it on this host would involve. Downloads no weights.

| Argument | Description |
|---|---|
| `MODEL` | Hugging Face repository id. Required. |

| Option | Value | Default | Description |
|---|---|---|---|
| `--revision` | text | – | Branch, tag or commit sha. |
| `--files` | | off | Also list the repository files. |
| `--json` | | off | Emit the report as JSON. |

```bash
yonga inspect Qwen/Qwen3-0.6B
```

## yonga prepare

```text
yonga prepare [OPTIONS] MODEL
```

Download, convert, validate and probe a model into a prepared artifact. See [Preparing models](../guide/prepare.md).

| Argument | Description |
|---|---|
| `MODEL` | Hugging Face repository id. Required. |

| Option | Value | Default | Description |
|---|---|---|---|
| `--revision` | text | – | Branch, tag or commit sha. |
| `--device` | text | `auto` | `auto`, `npu`, `gpu` or `cpu`. |
| `--profile` | text | – | Conversion profile id (`yonga registry profiles` lists them). Without it the registry chooses. |
| `--source` | text | – | Convert this repository instead of the requested one. |
| `--trust-remote-code` | | off | Allow the conversion to execute Python published in the repository. |
| `--ignore-exporter-requirements` | | off | Convert even though a declared exporter package version is unmet, or the registry knows this export fails with the installed versions. Deliberate and separate: `--yes` never enables it. |
| `--no-managed-env` | | off | Never build a managed export environment; refuse instead when the installed packages cannot meet the exporter requirements. |
| `--gguf-file` | text | – | For a GGUF repository, the file to import. By default yonga picks the first preferred quantization whose tensor types OpenVINO GenAI can read. |
| `--fresh-export` | | off | Export again even when a failed job left a completed export with identical inputs that could be reused. |
| `--yes`, `-y` | | off | Answer ordinary questions with yes. |
| `--force` | | off | Re-prepare even if an artifact exists. |
| `--no-repair` | | off | Never modify the exported IR. |
| `--json` | | off | Emit the outcome as JSON. |

```bash
yonga prepare Qwen/Qwen3-0.6B --device npu
```

## yonga list

```text
yonga list [OPTIONS]
```

List the artifacts prepared on this host.

| Option | Value | Default | Description |
|---|---|---|---|
| `--clean` | | off | Retire stale staging and temp directories under the retention policy. |
| `--json` | | off | Emit the catalogue as JSON. |

```bash
yonga list
```

## yonga remove

```text
yonga remove [OPTIONS] ARTIFACT
```

Remove a prepared artifact and its compile-cache entries. The Hugging Face download cache is never touched.

| Argument | Description |
|---|---|
| `ARTIFACT` | Artifact id, repo id or path. Must match exactly one artifact. Required. |

| Option | Value | Default | Description |
|---|---|---|---|
| `--yes`, `-y` | | off | Do not ask for confirmation. |
| `--json` | | off | Emit the outcome as JSON. |

```bash
yonga remove cc24636a2f49356a
```

## yonga probe

```text
yonga probe [OPTIONS] MODEL_OR_ARTIFACT
```

Compile a prepared artifact on a device and report what worked.

| Argument | Description |
|---|---|
| `MODEL_OR_ARTIFACT` | Artifact id or path. Required. |

| Option | Value | Default | Description |
|---|---|---|---|
| `--device` | text | `auto` | `auto`, `npu`, `gpu` or `cpu`. |
| `--all-submodels` | | off | Compile every submodel, not only the language model. |
| `--quick` | | off | Never build the GenAI pipeline; a deferred raw compile is reported inconclusive. |
| `--timeout` | `SECONDS` | – | Bound the GenAI pipeline build that settles a deferred compile: run it in a child process and stop it after SECONDS (the device is then inconclusive). Cannot be combined with `--quick`. |
| `--json` | | off | Emit the report as JSON. |

Exits 0 when any device passed, 2 when none passed but none failed outright either (left inconclusive by `--quick` or `--timeout`), otherwise the code of the first failure.

```bash
yonga probe cc24636a2f49356a --device npu
```

## yonga chat

```text
yonga chat [OPTIONS] MODEL_OR_ARTIFACT
```

Chat with a prepared artifact, locally. See [Chat, probe, bench and eval](../guide/run.md) for the in-chat commands such as `/reset` and `/exit`.

| Argument | Description |
|---|---|
| `MODEL_OR_ARTIFACT` | Artifact id or path. Required. |

| Option | Value | Default | Description |
|---|---|---|---|
| `--device` | text | – | `auto`, `npu`, `gpu` or `cpu`. Without it, the `default_device` setting when it names a device, otherwise the artifact's recommended device. |
| `--max-new-tokens` | integer | – | Output token limit. |
| `--temperature` | number | – | Sampling temperature. |
| `--top-p` | number | – | Nucleus sampling probability. |
| `--top-k` | integer | – | Top-k sampling cutoff. |
| `--show-reasoning` | | off | Display hidden reasoning blocks. |
| `--system` | text | – | System prompt for the session. |
| `--json` | | | Not supported: a conversation is not one document (exit 70). |

```bash
yonga chat cc24636a2f49356a
```

## yonga serve

```text
yonga serve [OPTIONS] MODEL_OR_ARTIFACT
```

Serve a prepared artifact through an OpenAI-compatible HTTP API. Needs the `serve` extra in a pip install; the AppImage includes it. See [OpenAI-compatible server](../guide/serve.md).

| Argument | Description |
|---|---|
| `MODEL_OR_ARTIFACT` | Artifact id or path. Required. |

| Option | Value | Default | Description |
|---|---|---|---|
| `--host` | text | `127.0.0.1` | Address to bind. Loopback unless you say otherwise. |
| `--port` | integer, 1–65535 | `8000` | TCP port to listen on. |
| `--device` | text | – | `auto`, `npu`, `gpu` or `cpu`. Without it, the `default_device` setting when it names a device, otherwise the artifact's recommended device. |
| `--max-new-tokens` | integer | – | Default output token limit when a request sets none. |
| `--show-reasoning` | | off | Return reasoning blocks instead of hiding them. |
| `--max-body-mb` | integer, 1–64 | `8` | Largest request body in MiB; embedded images count towards it. |
| `--carry-images-forward` | | off | Move images sent with earlier messages onto the last one instead of refusing them. |
| `--json` | | | Not supported: a server is not one document (exit 70). |

```bash
yonga serve cc24636a2f49356a --port 8000
```

## yonga eval

```text
yonga eval [OPTIONS] CANDIDATE
```

Measure how far CANDIDATE's predictions move from REFERENCE's (KL, top-1, perplexity). Both must be artifacts of the same weights; the reference is usually a higher-precision one, such as one prepared with `--profile cpu-int8-asym-ref`.

| Argument | Description |
|---|---|
| `CANDIDATE` | Artifact to evaluate. Required. |

| Option | Value | Default | Description |
|---|---|---|---|
| `--reference` | text | – | Higher-precision artifact of the same weights to compare to. Required. |
| `--device` | text | `cpu` | `cpu` (full precision) or `gpu`. |
| `--corpus` | path | – | Text to read instead of the shipped corpus (`.json` or `.txt`). |
| `--allow-different-sources` | | off | Compare artifacts converted from different repositories (e.g. GGUF vs original). |
| `--json` | | off | Emit the result as JSON. |

```bash
yonga eval CANDIDATE_ID --reference REFERENCE_ID
```

## yonga publish

```text
yonga publish [OPTIONS] ARTIFACT
```

Upload a prepared artifact to the Hugging Face Hub with a generated model card. The repository is private unless you pass `--public`.

| Argument | Description |
|---|---|
| `ARTIFACT` | Prepared artifact id or path. Required. |

| Option | Value | Default | Description |
|---|---|---|---|
| `--repo` | text | – | Destination `NAMESPACE/NAME` on the Hub. Always explicit. Required. |
| `--public` | | off | Create a public repository (the default is private). |
| `--update` | | off | Write into a repository that already exists. |
| `--license-reviewed` | | off | State that you read a license yonga cannot classify and it permits redistributing a quantized derivative. `--yes` never implies it. |
| `--dry-run` | | off | Check everything and write the card locally; upload nothing. |
| `--output` | path | – | Where to write the card, manifest and upload list. With `--dry-run` and no `--output`: `./yonga-publish-<artifact>`. |
| `--yes`, `-y` | | off | Approve the upload without a prompt. |
| `--json` | | off | Emit the outcome as JSON. |

```bash
yonga publish cc24636a2f49356a --repo NAMESPACE/Qwen3-0.6B-openvino-npu --dry-run
```

## yonga bench

```text
yonga bench [OPTIONS] MODEL_OR_ARTIFACT
```

Benchmark a prepared artifact and save a reproducible JSON result.

| Argument | Description |
|---|---|
| `MODEL_OR_ARTIFACT` | Artifact id or path. Required. |

| Option | Value | Default | Description |
|---|---|---|---|
| `--device` | text | – | `auto`, `npu`, `gpu` or `cpu`. Without it, the artifact's recommended device. |
| `--suite` | text | `standard` | `quick`, `standard` or `long`. |
| `--runs` | integer | – | Measured runs per prompt. Without it, the `bench_runs` setting (3). |
| `--warmups` | integer | – | Warmup runs per prompt. Without it, the `bench_warmups` setting (1). |
| `--json` | | off | Emit the result as JSON. |

```bash
yonga bench cc24636a2f49356a --suite quick
```

## yonga logs

```text
yonga logs [OPTIONS] [JOB_ID]
```

Show the job logs kept on disk, most recent first. Without `JOB_ID`, lists the 20 most recent jobs.

| Argument | Description |
|---|---|
| `JOB_ID` | Job id, or a prefix. Optional. |

| Option | Value | Default | Description |
|---|---|---|---|
| `--raw` | | off | Print the JSON Lines file verbatim. |
| `--bundle` | | off | Write a redacted, anonymised `.tar.gz` for a bug report (log, state, manifest, host report). |
| `--output` | file | `./yonga-report-<job>.tar.gz` | Where to write the `--bundle` archive. |
| `--json` | | off | Emit the records as JSON. |

```bash
yonga logs de862650 --bundle
```

## yonga registry

```text
yonga registry [OPTIONS] COMMAND [ARGS]...
```

Inspect, validate and update the compatibility registry. See [Maintenance](../guide/maintain.md#the-compatibility-registry).

### registry validate

```text
yonga registry validate [OPTIONS]
```

Load every registry layer and check it is internally consistent. Exits 0 when valid, 2 when an installed remote bundle was ignored, 70 when problems are found.

| Option | Value | Default | Description |
|---|---|---|---|
| `--json` | | off | Emit the result as JSON. |

```bash
yonga registry validate
```

### registry profiles

```text
yonga registry profiles [OPTIONS]
```

List the conversion profiles `--profile` accepts.

| Option | Value | Default | Description |
|---|---|---|---|
| `--device` | text | – | Only profiles for this device. |
| `--json` | | off | Emit the profiles as JSON. |

```bash
yonga registry profiles --device npu
```

### registry explain

```text
yonga registry explain [OPTIONS] MODEL
```

Show which registry rules matched a model, and why they won.

| Argument | Description |
|---|---|
| `MODEL` | Hugging Face repository id. Required. |

| Option | Value | Default | Description |
|---|---|---|---|
| `--device` | text | – | Explain for this device only. |
| `--json` | | off | Emit the explanation as JSON. |

```bash
yonga registry explain Qwen/Qwen3-0.6B --device npu
```

### registry paths

```text
yonga registry paths [OPTIONS]
```

Show where registry layers, configuration and managed data live.

| Option | Value | Default | Description |
|---|---|---|---|
| `--json` | | off | Emit the paths as JSON. |

```bash
yonga registry paths
```

### registry update

```text
yonga registry update [OPTIONS]
```

Fetch, verify and install a signed remote registry bundle. Only bundles signed by a key in `~/.config/yonga/registry-keys/` are accepted.

| Option | Value | Default | Description |
|---|---|---|---|
| `--url` | text | `registry_url` setting (unset) | Bundle URL (https); the signature is fetched from URL.sig. |
| `--file` | path | – | A bundle already on disk. Cannot be combined with `--url`. |
| `--signature` | path | `FILE.sig` | Its signature. Only with `--file`. |
| `--yes`, `-y` | | off | Install without a prompt. |
| `--json` | | off | Emit the outcome as JSON. |

```bash
yonga registry update --file bundle.json
```

### registry status

```text
yonga registry status [OPTIONS]
```

Show the installed remote registry bundle and whether it still verifies. Exits 2 when an installed bundle is being ignored.

| Option | Value | Default | Description |
|---|---|---|---|
| `--json` | | off | Emit the status as JSON. |

```bash
yonga registry status
```

### registry reset

```text
yonga registry reset [OPTIONS]
```

Remove the installed remote bundle and go back to the built-in registry.

| Option | Value | Default | Description |
|---|---|---|---|
| `--json` | | off | Emit the outcome as JSON. |

```bash
yonga registry reset
```

## yonga env

```text
yonga env [OPTIONS] COMMAND [ARGS]...
```

List and remove the managed environments exports run in. See [Maintenance](../guide/maintain.md#managed-export-environments).

### env list

```text
yonga env list [OPTIONS]
```

Show the managed export environments and the disk they use.

| Option | Value | Default | Description |
|---|---|---|---|
| `--json` | | off | Emit the environments as JSON. |

```bash
yonga env list
```

### env remove

```text
yonga env remove [OPTIONS] KEY
```

Delete one managed export environment. It is rebuilt if a later export needs it.

| Argument | Description |
|---|---|
| `KEY` | Environment key, or a unique prefix. Required. |

| Option | Value | Default | Description |
|---|---|---|---|
| `--yes`, `-y` | | off | Do not ask for confirmation. |
| `--json` | | off | Emit the result as JSON. |

```bash
yonga env remove fbfac3ac91ba
```

## yonga cache

```text
yonga cache [OPTIONS] COMMAND [ARGS]...
```

Show and prune the OpenVINO compile cache. `yonga cache` without a subcommand is the same as `yonga cache list`. See [Maintenance](../guide/maintain.md#the-compile-cache).

| Option | Value | Default | Description |
|---|---|---|---|
| `--json` | | off | Emit the entries as JSON. |

### cache list

```text
yonga cache list [OPTIONS]
```

List compile cache entries: size, last use, owner, and whether they are still used.

| Option | Value | Default | Description |
|---|---|---|---|
| `--json` | | off | Emit the entries as JSON. |

```bash
yonga cache list
```

### cache prune

```text
yonga cache prune [OPTIONS]
```

Remove orphaned and stale-runtime compile cache entries (and more, when asked).

| Option | Value | Default | Description |
|---|---|---|---|
| `--older-than` | `DAYS` | – | Also remove any entry not used for this many days. |
| `--unattributed` | | off | Also remove entries that cannot be traced to any artifact. |
| `--all` | | off | Empty the whole compile cache. |
| `--dry-run` | | off | Show what would go; delete nothing. |
| `--yes`, `-y` | | off | Do not ask for confirmation. |
| `--json` | | off | Emit the outcome as JSON. |

```bash
yonga cache prune --dry-run
```

## yonga self

```text
yonga self [OPTIONS] COMMAND [ARGS]...
```

Put the yonga AppImage on PATH as 'yonga', or take it off again. Works only when yonga runs from the AppImage.

### self install

```text
yonga self install [OPTIONS]
```

Copy the running AppImage to `~/.local/bin/yonga` so 'yonga' works in any shell.

| Option | Value | Default | Description |
|---|---|---|---|
| `--bin-dir` | `DIR` | `~/.local/bin` | Directory to install the 'yonga' command in. |
| `--yes`, `-y` | | off | Replace an old AppImage without asking. |
| `--json` | | off | Emit the outcome as JSON. |

```bash
./yonga-0.5.1-x86_64.AppImage self install
```

### self uninstall

```text
yonga self uninstall [OPTIONS]
```

Remove the AppImage copy at `~/.local/bin/yonga`. Models and caches are kept.

| Option | Value | Default | Description |
|---|---|---|---|
| `--bin-dir` | `DIR` | `~/.local/bin` | Directory to remove the 'yonga' command from. |
| `--yes`, `-y` | | off | Do not ask for confirmation. |
| `--json` | | off | Emit the outcome as JSON. |

```bash
yonga self uninstall
```
