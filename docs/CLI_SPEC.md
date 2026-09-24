# CLI_SPEC.md

## 1. CLI principles

The CLI should feel like a package manager plus local model runner, not a thin wrapper around OpenVINO commands.

- concise by default;
- detailed with `--verbose`;
- raw traceback only with `--debug` unless unexpected fatal error requires it;
- expensive/destructive decisions shown before execution;
- JSON output available for automation on informational commands;
- stable exit codes;
- colors respect `NO_COLOR` and non-TTY output.

Use Typer/Click unless there is a compelling implementation reason otherwise.

## 2. Global options

```text
yonga [GLOBAL OPTIONS] COMMAND ...

--config PATH
--data-dir PATH
--cache-dir PATH
--offline
--json
-v, --verbose
--debug
--version
--help
```

`--json` suppresses decorative UI and emits one JSON document for commands that support it.

## 3. `doctor`

```bash
yonga doctor [--full] [--json]
```

Default output sections:

```text
Host
  OS       Ubuntu 26.04
  Kernel   ...
  Python   3.14.x
  RAM      32 GiB

OpenVINO
  Runtime      2026.4.0
  GenAI        2026.4.0.0
  Tokenizers   2026.4.0.0
  ABI family   ✓ aligned

Devices
  CPU  ✓ Intel ...
  GPU  ✓ Intel Arc Graphics
  NPU  ✓ Intel NPU

NPU
  /dev/accel/accel0      ✓ accessible
  driver                 1.38...
  compiler-in-plugin     ✓ available

Storage
  cache filesystem       796 GiB free
  system /tmp            16 GiB tmpfs
  managed temp           ✓ disk-backed
```

Warnings must have remediation text.

Exit codes:

- 0: usable runtime exists;
- 2: runtime usable but warnings;
- 10: no usable OpenVINO device/runtime;

Do not make warnings such as small `/tmp` fatal if managed temp solves them.

## 4. `inspect`

```bash
yonga inspect MODEL [--revision REV] [--files] [--json]
```

Output:

```text
Model       ...
Revision    <sha>
License     apache-2.0
Task        image-text-to-text
Architecture qwen3_5
Formats     GGUF
Size        ~16.7 GiB (one model; 38.5 GiB across 5 files in the repository)

Compatibility on this host
  Architecture identified from GGUF metadata; this repository has no config.json.
  NPU   known unsupported / current compiler rule
  GPU   supported/test candidate
  CPU   supported/test candidate

Preparation
  Direct OpenVINO source: no
  Safetensors source: no
  Known derivative: nDimensional/... (approval required)
```

`Size` states one model first, because a GGUF repository publishes several interchangeable quantizations of the same weights and summing them overstates the download several-fold; the repository total follows, qualified by how many files it spans.

When the architecture was identified from an identity hint instead of from `config.json` (`REGISTRY_SPEC.md` section 4), the default output says so beside the compatibility block, because every claim in that block rests on that identification. Which alias matched is a `--verbose` detail.

No multi-GB download.

## 5. `search`

```bash
yonga search QUERY [--limit 20] [--device npu|gpu|cpu|auto]
```

This command is optional in milestone 1 but desired for MVP.

Use Hugging Face API search and enrich results with compatibility where architecture metadata is available. Do not claim compatibility without evidence.

## 6. `prepare`

```bash
yonga prepare MODEL \
  [--revision REV] \
  [--device auto|npu|gpu|cpu] \
  [--profile PROFILE] \
  [--source REPO_ID] \
  [--trust-remote-code] \
  [--ignore-exporter-requirements] \
  [--no-managed-env] \
  [--fresh-export] \
  [--yes] \
  [--force] \
  [--no-repair]
```

Default `--device auto`.

`--ignore-exporter-requirements` converts in the current environment even though a declared exporter
package version is unmet. Otherwise such a job runs its export in a managed export environment, or,
under `--no-managed-env`, stops in preflight. Like `--trust-remote-code` it is its own deliberate flag
and `--yes` does not imply it.

`--no-managed-env` refuses to build a managed export environment, so an exporter requirement the
installed packages cannot meet stops preparation at preflight instead. `--ignore-exporter-requirements`
also never builds one: it means "export in this environment regardless".

When an earlier job with identical inputs completed its export and then failed, the plan shows an
`Export` row naming that job, and stages 4 and 5 are skipped: the retained export is adopted once
every file has been shown to be what the exporter wrote, with repaired files restored from their
backups first. Anything that does not check out is discarded and the export runs normally. The
manifest records `export.reused_from_job`. `--fresh-export` always exports again. An export that was
itself interrupted is never reused.

Stages displayed:

```text
1/7 Inspecting source
2/7 Resolving compatibility
3/7 Preflight
4/7 Downloading
5/7 Converting
6/7 Validating / repairing
7/7 Probing runtime
```

On known NPU unsupported graph in auto mode:

```text
NPU probe failed: Qwen3.5 graph is not supported by the current NPU compiler.
This is a model/compiler compatibility limitation, not a missing NPU driver.

Trying recommended fallback: GPU
GPU compile probe ✓
```

On explicit `--device npu`, do not automatically move:

```text
NPU preparation cannot be completed.
Recommended fallback: --device gpu
```

Output final artifact ID/path.

## 7. `list`

```bash
yonga list [--json]
```

Shows prepared artifacts, source revision, profile, recommended device, size, last benchmark.

## 8. `remove`

```bash
yonga remove ARTIFACT
```

Only removes tool-owned prepared artifact and compile cache references. It must not delete Hugging Face global cache unless a future explicit cache command is used.

Interactive confirmation unless `--yes`.

## 9. `chat`

```bash
yonga chat MODEL_OR_ARTIFACT \
  [--device auto|npu|gpu|cpu] \
  [--max-new-tokens N] \
  [--temperature X] \
  [--top-p X] \
  [--top-k N] \
  [--show-reasoning] \
  [--system TEXT]
```

If argument is a HF model without prepared artifact, prompt to prepare or explain the command to run. Avoid surprising long conversions on `chat` unless a future `--prepare` flag explicitly permits it.

Startup panel:

```text
╭──────────── Local Chat ────────────╮
│ Qwen3.5-9B ...                     │
│ Device: Intel Arc Graphics (GPU)   │
│ Profile: gpu-int4-asym-g128        │
│ Reasoning display: hidden          │
│ Max output: 4096 tokens            │
╰────────────────────────────────────╯
```

Commands:

```text
/help
/reset
/clear
/tokens N
/temp X
/top-p X
/top-k N
/reasoning on|off
/stats on|off
/model
/exit
```

Default response flow:

1. show spinner `Yanıt hazırlanıyor...` / locale-neutral equivalent;
2. generate without streaming hidden reasoning to terminal;
3. sanitize recognized reasoning wrappers;
4. render final answer as Markdown panel;
5. render dim metrics footer.

Later enhancement: safe streaming after model-specific parser identifies final-answer boundary.

## 10. `bench`

```bash
yonga bench MODEL_OR_ARTIFACT \
  [--device DEVICE] \
  [--suite quick|standard|long] \
  [--runs N] \
  [--warmups N] \
  [--json]
```

Summary table:

```text
Prompt   In tok  Out tok  TTFT     tok/s   TPOT    Wall
short       42      256   180 ms   24.6    40 ms   10.8 s
medium     512      256   ...
```

Save full JSON and print its path.

## 11. `probe`

Advanced command:

```bash
yonga probe MODEL_OR_ARTIFACT [--device DEVICE] [--all-submodels]
```

Shows per-submodel read/compile result. This is the productized form of the manual diagnostic script used during development.

## 12. `registry`

Desired advanced commands:

```bash
yonga registry explain MODEL [--device DEVICE]
yonga registry validate
yonga registry paths
yonga registry update [--url URL | --file BUNDLE [--signature SIG]] [--yes] [--json]
yonga registry status [--json]
yonga registry reset
```

`explain` prints matched rules and why they won.

`update` installs a remote registry bundle (`REGISTRY_SPEC.md` section 12). The URL defaults to
`registry_url` in the configuration, which is unset: yonga contacts no registry server unless told
to. It fetches `URL` and `URL.sig` over HTTPS only, or reads `--file` and `FILE.sig`. It then
verifies the bundle completely, prints the signing key, the bundle's version and expiry and the
rules it adds or replaces, and installs it after confirmation (`--yes` approves). Refusals, exit 70:
`registry.remote.no_trusted_keys`, `untrusted_key`, `bad_signature`, `signature_malformed`,
`expired`, `rollback` (older than the installed bundle), `tool_too_old`, `unsafe_path`,
`unknown_repair_implementation` (a repair rule naming code this yonga does not contain),
`too_large`, `insecure_url`. `registry_update_policy = "never"` refuses updates and ignores an
installed bundle.

The installed bundle is one layer between the built-in rules and the user override directories,
and is verified again every time the registry loads. One that no longer verifies (expired, edited
on disk, its key removed) is left out and the built-in registry is used; `status` reports why and
exits 2. `reset` removes the bundle.

Trust is configuration: a key is trusted when its public key (base64 of the raw 32-byte Ed25519 key,
one line) is a `*.pub` file in `registry-keys/` under the configuration directory. yonga ships no
key, and nothing it downloads can add one. `tools/registry_bundle.py` creates keys and builds signed
bundles.

## 12a. `env`

```bash
yonga env list [--json]
yonga env remove KEY [--yes]
```

Lists and removes the managed export environments `prepare` builds when an architecture's exporter
requirements cannot be met by the installed packages. They live under the cache directory, are used
only by the export subprocess, and never modify the environment yonga runs in.

## 13. `logs`

```bash
yonga logs [JOB_ID]
yonga logs JOB_ID --raw
yonga logs JOB_ID --bundle [--output PATH]
```

Useful after failed conversion. `--bundle` writes a redacted, anonymised archive of the job log, job
state, artifact manifest and host report for attaching to a bug report.

## 14. `serve` (phase 2)

```bash
yonga serve MODEL_OR_ARTIFACT \
  [--host 127.0.0.1] \
  [--port 8000] \
  [--device auto] \
  [--max-new-tokens N] \
  [--show-reasoning]
```

Default bind must be localhost, not `0.0.0.0`. Requires the `serve` extra (FastAPI, uvicorn);
without it `serve` exits 10 naming `pip install 'yonga[serve]'`.

Endpoints:

| Method and path | Answer |
|---|---|
| `GET /health` | `{"status": "ok", "model", "device", "pipeline", "uptime_seconds"}`. Never requires a key. |
| `GET /v1/models`, `GET /v1/models/{id}` | The one served model, in OpenAI's model-object shape. |
| `POST /v1/chat/completions` | OpenAI chat completions; `stream: true` sends server-sent events ending in `data: [DONE]`, and `stream_options.include_usage` adds a usage chunk. |

The server opens the same `ChatSession` `chat` does (registry chat profile, device resolution,
compile cache, reasoning policy) and loads the model before it starts listening, so a model that
cannot load fails `serve` itself. A port already in use is refused before the model is loaded.
Each request carries its whole conversation and nothing is kept between requests. Generation is
serialised; a streaming client that disconnects stops its generation.

`model` must be the name given to `serve`, the artifact id, or the source repository (404
`model_not_found` otherwise). Accepted parameters: `max_tokens` / `max_completion_tokens`,
`temperature` (0 is greedy), `top_p`, `top_k`, `seed`, `repetition_penalty`, validated with the same
ranges as `chat`'s slash commands. Unknown fields are ignored. Parameters that would change the
answer in a way yonga cannot honour are refused with 400 and `param` set: `n` other than 1,
non-empty `stop`, non-zero `presence_penalty` / `frequency_penalty`, `tools` / `functions`, a
`response_format` other than text, `logprobs`, non-text content parts, and a conversation whose
last message is not from the user. A runtime failure is a 500 whose `error.code` is the diagnostic
code.

When `YONGA_API_KEY` is set in the environment, every endpoint except `/health` requires
`Authorization: Bearer <key>` (401 otherwise). The key is never logged or echoed. Binding a
non-loopback address prints a warning, a stronger one without a key.

## 14a. `publish`

```bash
yonga publish ARTIFACT --repo NAMESPACE/NAME \
  [--public] \
  [--update] \
  [--license-reviewed] \
  [--dry-run] [--output DIR] \
  [--yes] [--json]
```

Implements `SPEC.md` section 21. The destination is always explicit, and the repository is private
unless `--public` is passed. Before anything is uploaded, planning checks the following and refuses
when a check fails:

| Check | Refusal |
|---|---|
| `--repo` is `NAMESPACE/NAME` | `publish.repo_invalid` (70) |
| not in offline mode | `publish.offline` (70) |
| the artifact is ready: validated, one device probed OK | `publish.artifact_not_ready` (70) |
| a token exists, can write, and owns the namespace (account or organisation) | `publish.hub.not_logged_in`, `…read_only_token`, `…foreign_namespace` (21) |
| an existing repository is only written with `--update` | `publish.repo_exists` (70) |
| no license involved forbids derivatives (`-nd`) | `publish.license.restricted` (70); `--license-reviewed` does not override it |
| every license is recognised, or `--license-reviewed` states it was read | `publish.license.unreviewed` (70) |
| every file still matches the manifest's SHA-256 | `publish.files_changed` (70) |

Licenses are read fresh from the Hub for the requested model **and** the converted source: a
derivative is bound by both. Recognised identifiers are permissive (`apache-2.0`, `mit`, `bsd-*`,
`cc-by-4.0`, …), conditional (`llama3*`, `gemma`, `*openrail*`, `cc-by-nc-*`; allowed, and the card
says the terms carry over), or restricted. Anything else, including no license, is unknown.
`--yes` never implies `--license-reviewed`.

The upload is one commit: the files the manifest lists, the source's `LICENSE*`/`NOTICE*`/
`USE_POLICY*` files, a generated `README.md` and `yonga-manifest.json`, the manifest with the home
directory, login name and hostname anonymised and secrets redacted. The artifact's `.yonga/`
directory (repair backups, validation records) is never uploaded.

The model card states that the artifact is a quantized conversion and not a fine-tune. It lists
lineage with revisions and licenses, every export setting and package version, and every repair
with its before and after digests. Devices appear with probe evidence: `works` only for a passing
probe, `failed` with the diagnostic code, `not tested` for a device never probed. It also says so
when the profile's target device is not among the verified ones, and lists every file with its
SHA-256.

`--dry-run` runs every check (credential problems become warnings) and writes `README.md`,
`yonga-manifest.json` and `upload.txt` (digest, size and path of every file) to `--output`, by
default `./yonga-publish-<artifact>`. Nothing is uploaded. Without `--dry-run` the upload asks for
confirmation; `--yes` approves it, and a non-interactive run without `--yes` exits 60.

## 15. Exit codes

Reserve stable classes:

```text
0   success
2   success with warnings / doctor warnings
10  environment unusable
20  model/source resolution failure
21  authentication/gated access required
22  remote code approval required
30  insufficient disk/memory preflight
40  conversion failure
41  validation failure
42  repair required but disabled/denied
50  requested device unsupported
51  runtime compilation failure
52  runtime generation failure
60  user cancelled
70  registry/configuration error
99  unexpected internal error
```

## 16. Localization

MVP UI strings may be English-first, but do not embed logic in strings. Keep a future localization boundary. Model answer language is model-driven; system prompt should default to answering in the user's language when appropriate.
