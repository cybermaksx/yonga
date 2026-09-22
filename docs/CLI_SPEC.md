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
Size        ~18.8 GiB total weights

Compatibility on this host
  NPU   known unsupported / current compiler rule
  GPU   supported/test candidate
  CPU   supported/test candidate

Preparation
  Direct OpenVINO source: no
  Safetensors source: no
  Known derivative: nDimensional/... (approval required)
```

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
  [--yes] \
  [--force] \
  [--no-repair]
```

Default `--device auto`.

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
```

`explain` prints matched rules and why they won.

## 13. `logs`

```bash
yonga logs [JOB_ID]
yonga logs JOB_ID --raw
```

Useful after failed conversion.

## 14. `serve` (phase 2)

```bash
yonga serve MODEL_OR_ARTIFACT \
  [--host 127.0.0.1] \
  [--port 8000] \
  [--device auto]
```

Default bind must be localhost, not `0.0.0.0`.

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
