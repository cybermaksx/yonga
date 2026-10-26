# `yonga connect opencode`: register an artifact with OpenCode

Status: approved design, not implemented. Date: 2026-10-24.

## 1. Goal

A user who has prepared an artifact should be able to use it from OpenCode without writing
OpenCode configuration by hand. Today that takes knowing the artifact id (the name `serve`
answers to), the base URL, the context and output limits OpenCode cannot look up for a custom
provider, and, most of all, whether the model can work with OpenCode at all. The command writes
the provider entry, refuses an artifact that is known not to work, and says which `serve`
command to run.

Success: `yonga connect opencode ARTIFACT` followed by `yonga serve ARTIFACT` lets `opencode`
run a task that calls tools, with no hand-written configuration; an artifact that cannot work
is refused with the reason and an alternative, and nothing is written.

Out of scope: starting or supervising `serve`, other clients (Continue, Aider, Open WebUI),
serving several artifacts from one process, editing JSONC while keeping its comments.

## 2. Facts this design rests on

Measured on the reference host on 2026-10-21 with OpenCode 2.1.2 and yonga 0.5.1:

- OpenCode configures an OpenAI-compatible server as a provider with `npm`
  `@ai-sdk/openai-compatible`, `options.baseURL` and a `models` map whose keys are the names sent
  as `model`. Each model takes `limit.context`, `limit.output`, optional `limit.input`, and
  `tool_call`. Files are merged: the global `~/.config/opencode/opencode.json`, `OPENCODE_CONFIG`,
  then a project `opencode.json`. JSON and JSONC are both read. `{env:NAME}` is substituted.
- OpenCode sends its tools with every agent request, also when the model sets
  `"tool_call": false`: 12 tools (`edit`, `glob`, `grep`, `question`, `read`, `shell`, `skill`,
  `subagent`, `webfetch`, `websearch`, `write`, `execute`). Only the title request carries none.
  A model whose `serve` refuses tools (400 `serve.request.tools_unsupported`) therefore cannot
  be used by OpenCode at all.
- The first agent request, counted with Qwen3's chat template and tokenizer, is 5261 tokens in an
  empty `HOME` (system prompt 1965 tokens, the rest tool schemas) and 8514 tokens in a home with
  skills and an `AGENTS.md`. The title request is 525 tokens.
- The built-in NPU profiles compile with `MAX_PROMPT_LEN` 1024, so every built-in NPU artifact
  answers OpenCode with 400 `context_length_exceeded`; with `limit.context` 1024 OpenCode does not
  send at all ("The compaction request cannot be reduced further").
- `yonga serve 073d5d7c8ea75283` (Qwen/Qwen3-4B, GPU) with a hand-written entry ran
  `opencode run "Read README.txt …"`: one `read` tool call, a correct answer, 35 s in total.
- OpenCode sends `max_completion_tokens` equal to `limit.output`, `stream: true` and
  `stream_options.include_usage`; all are accepted by `serve`.

## 3. Command

```bash
yonga connect opencode ARTIFACT [--host 127.0.0.1] [--port 8000] [--device auto]
                                [--context N] [--config PATH] [--yes] [--json]
yonga connect opencode --remove ARTIFACT [--config PATH] [--yes] [--json]
```

`connect` is a new Typer sub-application with one command, `opencode`, wired like `self`.

- `--host`, `--port` and `--device` default to what `serve` uses, and the device is resolved
  exactly as `serve` resolves it: the fixed prompt length depends on it.
- `--config PATH` names the OpenCode file to edit. Without it the target is
  `$XDG_CONFIG_HOME/opencode/opencode.json` (`~/.config/opencode/opencode.json`). `~` is expanded
  and a relative path is taken from the current directory.
- `--context N` overrides the context limit read from the artifact (section 5).
- The command shows what it will add or change and asks before writing; `--yes` skips the
  question. After writing it prints the `yonga serve ARTIFACT --port P` command to run (with
  `--host` and `--device` when they are not the defaults).

`--remove ARTIFACT` takes an artifact id as recorded (section 6), so it works after
`yonga remove` deleted the artifact. Only `--config`, `--yes` and `--json` combine with it.

## 4. Compatibility checks

Run in this order, before anything is read from or written to the OpenCode file. The artifact
is opened as a `ChatSession` (as `serve` does) but never loaded.

1. The reference resolves through the artifact store; an unknown or ambiguous one fails with the
   store's existing error.
2. The device resolves as for `serve`.
3. Tools. When `session.tool_calling` is `None`, exit 70 `connect.opencode.tools_unsupported`.
   The message says that OpenCode sends its tools with every request and gives
   `session.tool_calling_refusal`. The remediation lists up to three prepared artifacts that do
   take tools, each as a `yonga connect opencode ID` command; when there are none it names the
   families whose built-in chat profile declares tool calling (read from the registry, not
   hard-coded).
4. Fixed prompt length. When the resolved device's runtime properties carry `MAX_PROMPT_LEN`
   below `OPENCODE_MIN_PROMPT_TOKENS` (8192), exit 70 `connect.opencode.prompt_too_short`. The
   message gives both numbers and the measurement behind the constant. The remediation names a
   prepared GPU or CPU artifact of the same source model (`yonga connect opencode ID`) when one
   exists, else `yonga prepare MODEL --device gpu`.

`OPENCODE_MIN_PROMPT_TOKENS` is the first power of two above the 5261-token minimum of section
2, leaving about 3000 tokens for the conversation; the constant's comment records the
measurement (OpenCode version, date, host, tokenizer).

## 5. The entry written

One provider per host and port, holding every artifact connected to that address:

```json
"yonga-8000": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "yonga (127.0.0.1:8000)",
  "options": { "baseURL": "http://127.0.0.1:8000/v1" },
  "models": {
    "073d5d7c8ea75283": {
      "name": "Qwen/Qwen3-4B · GPU · gpu-int4-asym-g128-r08",
      "tool_call": true,
      "limit": { "context": 40960, "output": 4096 }
    }
  }
}
```

- The provider key is `yonga-<port>` for a loopback host and `yonga-<host>-<port>` otherwise,
  with characters outside `[A-Za-z0-9_-]` replaced by `-`. The base URL formats the host as
  `serve` prints it (IPv6 in brackets).
- The model key is the artifact id: `serve` answers to it, and unlike the repository id it cannot
  collide between two artifacts on one port. The name is the requested repository, the device
  and the profile.
- Whichever artifact `serve` runs on the port answers; picking another one in OpenCode gets the
  existing 404 `The model '…' is not served here.`
- `limit.context` is `max_position_embeddings` from the artifact's `config.json`, or from its
  `text_config` for a VLM, unless `--context` is given. When neither exists, exit 70
  `connect.opencode.context_unknown`, asking for `--context`; no value is invented.
- `limit.output` is the manifest's `recommended_runtime.max_new_tokens`, else the configured
  `max_new_tokens` (4096 by default).
- `limit.input` is written only when the prompt length is fixed (and passed check 4): it is
  `MAX_PROMPT_LEN`, so OpenCode compacts before `serve` would refuse. `limit.context` is then
  `MAX_PROMPT_LEN` plus `limit.output` unless `--context` is given.
- When `YONGA_API_KEY` is set in the environment, `options.apiKey` is `{env:YONGA_API_KEY}`;
  the key itself is never written.
- `reasoning` and `modalities` are not written: what OpenCode sends with them is unverified, and
  no artifact with tool calling takes images today.

Outcomes: `added` (new model entry), `updated` (the artifact's entry or the provider's
`options`/`npm`/`name` differ from what would be written; the difference is shown before the
question), `already_connected` (identical; nothing is written, exit 0, no question).

## 6. Editing the file

`src/yonga/adapters/opencode_config.py` holds pure functions over the parsed document plus the
file I/O; `src/yonga/application/connect.py` plans, checks, applies and records.

- The file is read once and parsed with `json.loads`. Anything that does not parse (comments,
  trailing commas, a broken file), a root that is not an object, a `provider` that is not an
  object, or a `provider.yonga-…` entry whose `models` is not an object is exit 70
  `connect.opencode.config_not_json`. The file is left alone and the JSON to paste is printed.
- A missing file is created with `$schema` (`https://opencode.ai/config.json`) first; a missing
  directory is created.
- Every other key and the existing key order are kept; new keys are appended. Output is indented
  with two spaces and ends with a newline.
- The write goes to a temporary file in the same directory, is flushed with `fsync`, keeps the
  existing file's mode (0600 for a new one) and replaces the target with `os.replace`. Just before
  the replace the file's bytes are compared with what was read; if they changed, nothing is
  written: exit 70 `connect.opencode.config_changed`. A symbolic link as the target is resolved
  and its target edited, as OpenCode reads through it.
- A directory the user cannot write to is exit 10 `connect.opencode.config_not_writable`; sudo
  is never used.

The record, `<data-dir>/opencode-connections.json`, keeps one entry per connection: the resolved
config path, the provider key, the artifact id, the model entry exactly as written, whether yonga
created the provider and, if so, the provider fields it wrote, and the time. `connect` adds or
replaces the entry after a successful write; a record that cannot be written is a warning, not a
failure (the OpenCode file is already correct).

`--remove ARTIFACT` looks up the record entries for that artifact and config file:

- A model entry identical to the recorded one is removed. One that differs was edited by the
  user and is kept: exit 70 `connect.opencode.entry_modified`, naming it.
- The provider is removed too when yonga created it, it has no models left and its other fields
  are still what yonga wrote.
- The file itself is never deleted. The record entries are dropped for what was removed, and for
  entries whose model is already gone from the file.
- No record entry: `not_connected`, exit 0.

`yonga remove ARTIFACT` does not edit OpenCode configuration; when the record has an entry for
the removed artifact it prints one line with the `yonga connect opencode --remove ID` command.

## 7. Exit codes and JSON

| Code | When |
|---|---|
| 0 | `added`, `updated`, `already_connected`, `removed`, `not_connected` |
| 10 | `connect.opencode.config_not_writable` |
| 60 | The question was declined, or cannot be asked (non-interactive, or `--json` without `--yes`) |
| 70 | `tools_unsupported`, `prompt_too_short`, `context_unknown`, `config_not_json`, `config_changed`, `entry_modified` (all `connect.opencode.*`) |

`--json` for `connect`: `action`, `config`, `created_config`, `provider`, `created_provider`,
`model`, `base_url`, `limit`, `serve_command`, `exit_code`. For `--remove`: `action`, `config`,
`removed` (`[{provider, model}]`), `kept` (`[{provider, model, reason}]`), `exit_code`. Errors use
the existing error document.

## 8. Testing

Written test first.

- Adapter unit tests: create a new file; add to an existing file keeping other keys and their
  order; a second identical run changes nothing; update; refuse JSONC, a broken file, a non-object
  root, `provider` or `models`; detect a file changed between read and write; keep the file mode;
  remove only a matching entry; remove an emptied provider yonga created and keep one it did not.
- Service unit tests with fake sessions and manifests: no tool calling (with and without other
  tool-capable artifacts); NPU with `MAX_PROMPT_LEN` 1024 (with and without a GPU artifact of the
  same model); NPU at 8192 (writes `limit.input`); context from `config.json` and from
  `text_config`; `--context`; missing context; `YONGA_API_KEY`; provider keys for loopback,
  another host and IPv6; the record written, replaced and dropped.
- CLI tests in the style of `tests/unit/test_cli.py`: the question, `--yes`, `--json`,
  non-interactive without `--yes` (60), `--remove`, and the note printed by `remove`.
- By hand on the reference host: `connect` into a scratch `--config`, `serve`, then an
  `opencode run` that calls a tool; and an NPU artifact refused. Reported with the change.
- `ruff`, `mypy src tests`, and the suite.

## 9. Documentation

- `docs/CLI_SPEC.md`: a `connect` section (this design, condensed), and the new codes.
- `README.md`: the command table and count (eighteen commands; `connect` has one subcommand).
- `docs/USAGE.md`: an OpenCode part under "Serve it to other tools".
- `website/docs/en/guide/serve.md` and `website/docs/tr/guide/serve.md`: the same, in both
  languages.
- `CHANGELOG.md`: under Unreleased.
