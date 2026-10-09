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

Global options go before the command name: `yonga --offline --debug prepare MODEL`, not
`yonga prepare MODEL --offline`. `--json` is the one global option that commands also accept after
their own name (`yonga list --json`).

`--json` suppresses decorative UI and emits one JSON document for commands that support it. `chat`
and `serve` reject it (exit 70): a conversation is not one document, and `serve` reports its state
as JSON at `GET /health` instead.

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
yonga search QUERY [--limit 20] [--device npu|gpu|cpu|auto] [--json]
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
  [--gguf-file FILE] \
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

A GGUF-only repository with no usable declared derivative is **imported** instead of exported,
when a registry `gguf` rule covers the installed OpenVINO GenAI (`REGISTRY_SPEC.md` section 8a).
Candidate files are the rule's preferred quantization labels in order, or the one `--gguf-file`
names. Each candidate's header is read from the Hub, and the first whose architecture and every
tensor type the rule verifies is chosen; the plan lists the files passed over and why. Only that
file is downloaded. The plan shows it under **GGUF file**, its tensor mix as its **Quantization**
(the profile's quantization settings do not apply), and the profile as **Runtime profile**. The
artifact is named `<revision>--gguf-<label>-<device>` and records `export.backend:
openvino-genai-gguf`. Refusals: `source.gguf_unsupported` (no verified rule, or an architecture it
does not list), `source.gguf_no_importable_file` (every candidate holds an unreadable tensor type,
named), `source.gguf_file_not_found`; all exit 20.

Under **Quantization** the plan prints **Chosen because**: one sentence saying why the profile won
(its size band, the device default, the architecture or model rule that names it, or `--profile`),
with the parameter count and where it came from, and any size-banded profile the model's size kept
out. The same sentence goes into the manifest (`export.profile_selection`). It is absent for a GGUF
import or an OpenVINO copy, whose quantization no profile decides. When the revision is already
prepared for the device under another profile id (a default that changed since), the plan's known
limitations name that artifact and the `--profile` that would reuse it; the new export never
replaces it.

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
yonga list [--clean] [--json]
```

Shows prepared artifacts, source revision, profile, recommended device, size, last benchmark.

`--clean` retires stale staging and temporary directories under the retention policy
(`workspace_retention_hours`, `workspace_retention_max`) and never touches a running job's
directories.

## 8. `remove`

```bash
yonga remove ARTIFACT [--yes] [--json]
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
yonga probe MODEL_OR_ARTIFACT [--device DEVICE] [--all-submodels] [--quick | --timeout SECONDS] [--json]
```

Shows per-submodel read/compile result. This is the productized form of the manual diagnostic script used during development.

A raw compile the registry marks `defer_to_pipeline` (on the NPU, every dynamically shaped LLM) is
recorded as `deferred` and settled by building the real GenAI pipeline and generating a few
tokens. For a large model on the NPU that build is a full NPU compilation and can take many
minutes; the probe says so before it starts.

- `--quick` never builds the pipeline. A device whose only open question is a deferred raw compile
  is reported `inconclusive` (`probe.inconclusive.quick`), with the commands that settle it.
- `--timeout SECONDS` (a positive number) builds the pipeline in a child process,
  `python -P -m yonga.adapters.smoke_child`, and stops it after SECONDS. A build stopped there is
  `inconclusive` (`probe.inconclusive.timeout`): never `ok`, and never a compiler verdict. A child
  that dies without printing its JSON result is `failed`, classified through the registry from its
  stderr like an in-process failure.
- The two cannot be combined (`probe.options.conflict`, exit 70); a timeout that is not a finite
  number above zero is `probe.timeout.invalid`, exit 70. Without either option the pipeline is
  built in-process, with no time limit.

`inconclusive` never counts as working — not in the exit code, the role matrix, the manifest or a
model card — and never as a failure of the device or the model. Exit codes: 0 when any device
passed; otherwise the code of the first definite failure; otherwise, when every remaining device
is inconclusive, 2. `--json` has `"ok": false` and `"inconclusive": true` in that case, and the
`quick` and `timeout_seconds` the probe ran with; tell it from a failure by the exit code or
`inconclusive`.

`python -m yonga.application.probe ARTIFACT` accepts the same `--quick` and `--timeout`, prints the
JSON report, and exits 0, 2 (inconclusive) or 1; an invalid `--quick`/`--timeout` combination exits
70, and argparse usage errors exit 2 as usual.

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
yonga logs [JOB_ID] [--json]
yonga logs JOB_ID --raw
yonga logs JOB_ID --bundle [--output PATH]
```

Useful after failed conversion. `--bundle` writes a redacted, anonymised archive of the job log, job
state, artifact manifest and host report for attaching to a bug report.

## 14. `serve`

```bash
yonga serve MODEL_OR_ARTIFACT \
  [--host 127.0.0.1] \
  [--port 8000] \
  [--device auto] \
  [--max-new-tokens N] \
  [--show-reasoning] \
  [--max-body-mb 8] \
  [--carry-images-forward]
```

Default bind must be localhost, not `0.0.0.0`. Requires the `serve` extra (FastAPI, uvicorn,
Pillow); without FastAPI or uvicorn `serve` exits 10 naming `pip install 'yonga[serve]'`. Pillow
is needed only to decode images: without it a VLM artifact is still served, `serve` prints a
warning, and a request with an image is a 500 `serve.images.dependency_missing` naming the extra.

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
ranges as `chat`'s slash commands, and `stop` (below). Unknown fields are ignored. Parameters that
would change the answer in a way yonga cannot honour are refused with 400 and `param` set: `n`
other than 1, non-zero `presence_penalty` / `frequency_penalty`, `tools` / `functions`, a
`response_format` other than text, `logprobs`, content parts other than `text` and `image_url`
(below), and a conversation whose last message is not from the user. A runtime failure is a 500
whose `error.code` is the diagnostic code.

`stop` is a string or a list of at most 4 strings (`MAX_STOP_SEQUENCES` in
`src/yonga/application/serve.py`), each non-empty and at most 256 characters (`MAX_STOP_LENGTH`).
Anything else is a 400 `invalid_request` with `param` `stop`; an empty list means no stop. yonga
matches the sequences itself, on the visible answer after the reasoning sanitiser, never inside a
hidden reasoning block: a common stop such as `"\n\n"` would otherwise end a reasoning model's
answer before it began. With `--show-reasoning` the reasoning block is part of the visible answer,
so a stop sequence can end it there. The answer ends before the first stop sequence to complete
(the match with the smallest end; of two ending together, the longer one), without the match and
without the whitespace in front of it, and `finish_reason` is `stop`, also when generation had
reached its token limit. That is where generation stops, and it does not depend on how the text was
chunked: with `["Observation:", "ion"]` the answer ends before `ion`. Matches across chunk
boundaries are found: a stream holds back only the tail that could still become a match and the
whitespace in front of it, so no part of a stop sequence is ever sent. With a stop sequence the
model is streamed internally even for a non-streaming request, so generation ends at the match
instead of at the token limit. A streamed and an unstreamed answer to the same request end at the
same place.

The registry chat profile's `stop_strings` (`docs/REGISTRY_SPEC.md` section 8b) are separate:
model-level end markers OpenVINO GenAI applies natively and leaves out of the answer.

Images (`image_url` content parts) are accepted by a VLM artifact only; the checks below run in
this order, each refusal naming the content part in `param`:

- Only user messages may carry images (400 `unsupported_parameter` otherwise), and an
  `image_url` part needs `image_url.url` (400 `invalid_request`).
- On an LLM artifact any image is a 400 `serve.request.images_unsupported`, `param`
  `messages[i].content[j]`.
- At most 4 images per request, counted over all messages (`MAX_IMAGES` in
  `src/yonga/application/serve.py`); more is a 400 `serve.request.too_many_images`, `param`
  `messages`.
- Images go with the last message only. OpenVINO GenAI 2026.4 binds images to the newest user
  turn of a stateless conversation: an image in an earlier turn fails with "Missing image/video
  with index 0", and list content in an earlier turn is rejected outright. So an image in any
  earlier message is a 400 `serve.request.images_earlier_turn` that says so, unless the server
  runs with `--carry-images-forward` (below).
- Then `stop` and the generation settings are checked, before any image is read.
- `image_url.url` must be a `data:image/(png|jpeg|webp|gif);base64,` URL. `http`, `https`, `file`
  and every other scheme are refused with 400 `unsupported_parameter`, `param`
  `messages[i].content[j].image_url.url`: permanently, with no option to allow them. yonga never
  fetches an image; a server that fetched addresses its clients chose could be made to reach
  services on its own host or network. Another media type, a data URL that is not base64, or one
  with parameters such as `charset` is refused the same way.
- Broken base64, bytes that are not a PNG, JPEG, WebP or GIF image (Pillow is offered no other
  decoder, whatever the declared type), and an image of more than 4096 x 4096 pixels
  (`MAX_IMAGE_PIXELS`) are a 400 `invalid_request` with the same `param`. The size is read from
  the image header and checked before any pixel is decoded, which also stops decompression bombs;
  a larger image is refused, never scaled down. The encoded size is bounded by the body limit.

Only the image headers are read before a request waits for the pipeline, on a worker thread, not
on the event loop. The pixels are decoded once the request holds the pipeline, so however many
requests are queued, at most one holds decoded images; a waiting request costs its body and no
more. An image whose header is fine but whose pixel data is broken (a truncated file, say) is
therefore found only then: a 400 `invalid_request` with the same `param` without `stream`, and
with `stream: true` an error event (below) that also carries `param`. Any failure inside Pillow
counts as a broken image, never as a server error. An accepted image is used as sent: the first
frame of an animation, an EXIF rotation applied as a browser shows it (an EXIF block that cannot
be read leaves the image as stored), transparency on white, 16-bit greyscale scaled to 8 bits;
`image_url.detail` is ignored. The images go to the pipeline with the last user message, in order,
ahead of its text (GenAI places untagged images there); a message holding only images is a
question too. GenAI reads its universal tags `<ov_genai_image_N>` and `<ov_genai_video_N>`
anywhere in a VLM prompt as "image N goes here", so those tags are removed from the text of every
message sent to a VLM artifact, repeatedly until none is left, and typing one cannot claim or move
an image. A model's own vision tokens typed as text (such as Qwen2-VL's
`<|vision_start|><|image_pad|><|vision_end|>`) are not removed: GenAI may read them as image
placeholders too, and which they are is per-family knowledge the registry does not hold yet.

`--carry-images-forward` trades accuracy for compatibility. Clients such as Open WebUI resend every
image of the conversation with each follow-up question, which the default refuses. With the flag,
images of earlier messages move onto the last user message, before its own images, and its text
gains a first line such as `[Image 1 was attached to an earlier message in this conversation.]`.
The model then sees every image as part of the newest turn: it is told that some came earlier, but
not with which words, so an answer that depends on *when* an image was shown may be wrong. The
4-image limit still counts all of them.

Request limits (`DEFAULT_MAX_BODY_BYTES` in `src/yonga/api/app.py`, `MAX_MESSAGES` in
`src/yonga/api/openai.py`):

- A request body over 8 MiB is refused with 413 `request_too_large`; `--max-body-mb N` sets the
  limit from 1 to 64 MiB. Base64 makes an image a third larger, so the default leaves room for
  about 6 MB of image files. A declared `Content-Length` over the limit is refused before the body
  is read; otherwise the bytes are counted as they arrive, so a chunked body without a length is
  held to the same limit.
- More than 4096 messages in one request is refused with 400, `param` `messages`.

Every error response uses OpenAI's shape, `{"error": {"message", "type", "param", "code"}}`:

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request` | The body does not validate (missing or mistyped fields, too many messages, `stop` outside its limits, an image that does not decode or is too large); `param` names the field. |
| 400 | `unsupported_parameter` | A parameter yonga cannot honour (above), including an image URL that is not an accepted `data:` URL; `param` names it. |
| 400 | the diagnostic code (`serve.request.*`) | A generation setting outside the ranges `chat` accepts, or an image refused by `images_unsupported`, `too_many_images` or `images_earlier_turn` (above, with `param`). |
| 401 | `invalid_api_key` | A key is configured and the request does not carry it. |
| 404 | `model_not_found` | `model` does not name the served model. |
| 404 | `not_found` | No such path. |
| 405 | `method_not_allowed` | The path exists but not for this method. |
| 413 | `request_too_large` | The body passed the size limit. |
| 500 | the diagnostic code | Generation failed for any other classified reason, or an image arrived and Pillow is not installed (`serve.images.dependency_missing`). |

Other HTTP errors raised by routing carry `invalid_request`.

With `stream: true`, a request refused before generation starts gets the same status and body as
above. A generation failure after the stream has begun cannot change the status, which is already
200: it arrives as one event, `data: {"error": {"message", "code"}}`, where `code` is the
diagnostic code or `internal_error`, followed by `data: [DONE]`. An image whose pixels fail to
decode is reported the same way, with `code` `invalid_request` and the image's `param`.

When `YONGA_API_KEY` is set in the environment, every endpoint except `/health` requires
`Authorization: Bearer <key>` (401 otherwise). Authentication runs before the size limit, so a
client without the key is refused before any body is read. The key is read from the environment
only, never from an option, and is never logged or echoed. Binding a non-loopback address prints a
warning, a stronger one without a key; set `YONGA_API_KEY` whenever `--host` is not loopback.

## 13a. `eval`

```bash
yonga eval CANDIDATE --reference REFERENCE [--device cpu|gpu] [--corpus FILE]
           [--allow-different-sources] [--json]
```

Teacher-forced quality comparison of two artifacts of the same weights. Both read the same text. At
every position the candidate's next-token distribution is compared with the reference's:

- mean `KL(reference || candidate)` in nats, the figure to rank by;
- top-1 agreement;
- each artifact's perplexity, and the candidate's change against the reference.

Nothing is sampled, so a result is deterministic for a pair of artifacts, a device and a corpus.

The logits come straight from the IR through `openvino.Core` (GenAI does not expose them),
tokenised by the artifact's own `openvino_tokenizer.xml`. `--device cpu`, the default, runs at
`INFERENCE_PRECISION_HINT=f32`, so the measured difference is the quantization's and not the
arithmetic's. `--device gpu` measures what the GPU actually computes. One model is loaded at a time;
the reference's log-probabilities are spilled to a memory-mapped file under the managed cache and
deleted afterwards.

Refusals (exit 70): `eval.artifact_not_found`; `eval.same_artifact`; `eval.different_sources` (the
two were converted from different repositories or revisions); `eval.tokenizer_mismatch` (the
corpus tokenises differently, so the vocabularies differ); `eval.vocab_mismatch`.
`--allow-different-sources` accepts two repositories holding the same model in different formats,
such as a GGUF release and the original weights. It relies on your word for the lineage, which the
result records. The tokenizer and vocabulary checks still apply. The shipped corpus is `yonga-eval-v1`:
eight passages of original, CC0 text across prose, technical writing, Python, arithmetic,
dialogue, procedure, encyclopedic writing and Turkish, 1955 tokens with the Qwen3.5 tokenizer.
`--corpus` takes a `.json` of the same shape, or plain text split on blank lines. Results are
saved under `<data>/evaluations/`.

The result is relative to the reference, which is itself an artifact. The `cpu-int8-asym-ref` and
`cpu-fp16-ref` registry profiles exist to prepare one, and never become a device's default.

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
lineage with revisions and licenses, every export setting and package version, and every repair with
its before and after digests. Devices appear with probe evidence: `works` only for a passing probe,
`failed` with the diagnostic code, `inconclusive` if a recorded probe is (`yonga probe` does not
write the manifest; `prepare` always settles its probes), `not tested` for a device never probed. It
also says so when the profile's target device is not among the verified ones, and lists every file
with its SHA-256.

`--dry-run` runs every check (credential problems become warnings) and writes `README.md`,
`yonga-manifest.json` and `upload.txt` (digest, size and path of every file) to `--output`, by
default `./yonga-publish-<artifact>`. Nothing is uploaded. Without `--dry-run` the upload asks for
confirmation; `--yes` approves it, and a non-interactive run without `--yes` exits 60.

## 15. Exit codes

Reserve stable classes:

```text
0   success
2   success with warnings / doctor warnings / probe inconclusive
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

A failure exits with its category's code (`CATEGORY_EXIT_CODES` in
`src/yonga/domain/diagnostics.py`). Codes 40, 42 and 52 are no category's code: they name the stage
a job died in (conversion, repair, generation) and appear only when the category is
`unknown_runtime_error`, except that `bench` exits 52 when none of its runs produced a measurement. Code 41 is both: the validation stage's code for an
`unknown_runtime_error`, and the category code of `unsupported_ir_type` and
`tokenizer_ir_incompatible`.

`prepare` exits 50 when it cannot use a device, whatever the diagnostic's category: an explicit
`--device` that is not present or whose probe fails, and an automatic run with no fallback left. The
exception is a failure whose diagnostic marks the environment as broken (`environment_broken`), such
as `npu.device.permission_denied` or `device.driver_missing`: no fallback is attempted, and
`prepare` exits 10, because the host has to be fixed before any device choice matters. `probe`,
`chat` and `bench` keep the category's code (a `probe` failure that stays `unknown_runtime_error`
exits 51, not 99), so an NPU compiler refusal is 51 there and 50 from
`prepare`. The diagnostic's code and category are in the message and the `--json` document either
way.

## 16. Localization

MVP UI strings may be English-first, but do not embed logic in strings. Keep a future localization boundary. Model answer language is model-driven; system prompt should default to answering in the user's language when appropriate.
