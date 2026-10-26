# `yonga connect opencode` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `yonga connect opencode ARTIFACT` command that writes a working OpenCode provider entry for a prepared artifact, refuses artifacts that cannot work with OpenCode, and can remove what it wrote.

**Architecture:** A file adapter (`adapters/opencode_config.py`) reads, edits and atomically writes OpenCode's JSON config with pure document functions. An application service (`application/connect.py`) opens the artifact as an unloaded `ChatSession`, runs the compatibility checks, builds the entry, and keeps a record of what it wrote under the data directory. `cli.py` gets a `connect` Typer sub-app with one `opencode` command, wired like `self`.

**Tech Stack:** Python 3.11-3.14, Typer, Pydantic, pytest (`typer.testing.CliRunner`), ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-10-24-opencode-connect-design.md`

## Global Constraints

- No new dependency. The standard library `json` module parses and writes the config; JSONC is refused, never rewritten.
- `OPENCODE_MIN_PROMPT_TOKENS = 8192`; its comment records: OpenCode 2.1.2, 2026-10-21, reference host, Qwen3 chat template and tokenizer, 5261 tokens for the first agent request in an empty `HOME` (8514 with skills and an `AGENTS.md`).
- Provider entry fields: `"npm": "@ai-sdk/openai-compatible"`, `"name": "yonga (HOST:PORT)"`, `"options": {"baseURL": "http://HOST:PORT/v1"}`, plus `"apiKey": "{env:YONGA_API_KEY}"` in `options` only when `YONGA_API_KEY` is set. The key value itself is never written.
- A new config file starts with `"$schema": "https://opencode.ai/config.json"`, mode `0600`. JSON is written with `indent=2`, `ensure_ascii=False`, and a trailing newline.
- Default config: `$XDG_CONFIG_HOME/opencode/opencode.json`, falling back to `~/.config/opencode/opencode.json` under the same XDG rules `yonga.config.paths` applies.
- Record file: `<data-dir>/opencode-connections.json`, `{"schema": 1, "connections": [...]}`.
- Exit codes come from the existing categories: `ConfigurationError`/`CONFIGURATION_ERROR` → 70, `YongaError`/`PERMISSION_DENIED` → 10, `CancelledError`/`USER_CANCELLED` → 60.
- Diagnostic codes: `connect.opencode.tools_unsupported`, `.prompt_too_short`, `.context_unknown`, `.config_not_json`, `.config_changed`, `.config_not_writable`, `.entry_modified`, `.declined`.
- Never sudo, never delete the config file, never touch keys yonga did not write.
- Checks: `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest tests/unit --basetemp=$PWD/build/pt-connect` (delete `build/pt-connect` afterwards; `/tmp` has a quota).

## Review Focus

1. **An empty or whitespace-only `opencode.json`** (a user ran `touch`). Expect it to be treated as an empty document and written, not refused. Test in Task 1.
2. **`provider` present but `null`, or a list.** Expect `config_not_json` naming `provider`, with the file untouched. Test in Task 1.
3. **`--config` pointing at an `opencode.jsonc` whose content is plain JSON.** Expect it to be edited normally. Only content that does not parse is refused. Test in Task 1.
4. **The same artifact connected on a second port.** Expect a second provider (`yonga-8001`) and no change to `yonga-8000`. `--remove` then removes both. Tests in Tasks 2 and 3.
5. **A repository id that matches several artifacts.** Expect the artifact store's existing ambiguity error, unchanged and before any file is read. Test in Task 2.

---

### Task 1: OpenCode config file adapter

**Files:**
- Create: `src/yonga/adapters/opencode_config.py`
- Test: `tests/unit/test_opencode_config.py`

**Interfaces:**
- Produces:
  - `OPENCODE_SCHEMA_URL: str = "https://opencode.ai/config.json"`
  - `default_config_path(env: Mapping[str, str]) -> Path`, reusing `_xdg_dir` from `yonga.config.paths` with `XDG_CONFIG_HOME` and the `.config` default.
  - `@dataclass(frozen=True) class ConfigSnapshot`: `path: Path` (symlinks resolved), `raw: bytes | None` (`None` when missing), `document: dict[str, Any]`, `mode: int | None`.
  - `read_config(path: Path) -> ConfigSnapshot`. Raises `ConfigurationError` `connect.opencode.config_not_json` when the bytes are not UTF-8, do not parse, or the root is not an object. Empty or whitespace-only content counts as `{}`.
  - `check_provider(document: Mapping[str, Any], key: str) -> None`. Raises `config_not_json` when `provider` exists but is not an object, or `provider[key]` exists but is not an object, or its `models` exists but is not an object. The message names the offending key path.
  - `with_model(document: Mapping[str, Any], key: str, provider_fields: Mapping[str, Any], model_id: str, entry: Mapping[str, Any]) -> dict[str, Any]`. Returns a deep copy. It adds `$schema` first only when the document is empty, creates `provider`/`provider[key]` when missing (fields first, then `models`), updates `provider_fields` on an existing provider, and sets `models[model_id] = entry`. Existing key order is kept; new keys are appended.
  - `without_model(document: Mapping[str, Any], key: str, model_id: str, *, drop_provider_when: Mapping[str, Any] | None) -> dict[str, Any]`. Deep copy without `models[model_id]`. When `drop_provider_when` is given, the provider's `models` is now empty, and its other fields equal `drop_provider_when`, the provider key goes too.
  - `render(document: Mapping[str, Any]) -> str`
  - `write_config(snapshot: ConfigSnapshot, document: Mapping[str, Any]) -> bool`. Returns whether the file was created. Creates parent directories. Writes a temporary file in the same directory, `fsync`s it, `chmod`s it to `snapshot.mode` (or `0o600`), and `os.replace`s the target. Right before the replace, it re-reads the target and raises `config_changed` (`ConfigurationError`) if its bytes differ from `snapshot.raw` (or if it now exists when `raw is None`). `PermissionError` on a write becomes a `YongaError` `config_not_writable` with `PERMISSION_DENIED`, whose remediation says to choose a writable `--config` and that yonga never uses sudo.

- [ ] **Step 1: Write the failing tests**

`test_missing_file_reads_as_empty_document` (`raw is None`, `document == {}`); `test_whitespace_only_file_reads_as_empty_document`; `test_jsonc_comment_is_refused_and_file_untouched` (content `{ // x\n}` → `ConfigurationError` with `diagnostic.code == "connect.opencode.config_not_json"`, file bytes unchanged); `test_trailing_comma_is_refused`; `test_non_object_root_is_refused` (`[]`); `test_jsonc_name_with_plain_json_is_read`; `test_symlink_is_resolved` (`snapshot.path` is the link target); `test_check_provider_refuses_null_provider` and `..._list_models` (message contains `provider` / `provider.yonga-8000.models`).

`test_with_model_on_empty_document_puts_schema_first` (`list(doc) == ["$schema", "provider"]`). `test_with_model_keeps_other_keys_and_order`: input `{"theme": "x", "provider": {"other": {...}}, "model": "a/b"}`; output has the keys in the same order, `other` unchanged, and `yonga-8000` appended last. `test_with_model_twice_is_identical` (`render` equal). `test_with_model_replaces_entry_and_provider_fields`.

`test_without_model_keeps_provider_with_other_models`; `test_without_model_drops_emptied_provider_when_fields_match`; `test_without_model_keeps_emptied_provider_when_fields_differ`.

`test_write_creates_file_with_mode_0600_and_newline` (returns `True`, `stat.S_IMODE == 0o600`, ends with `"\n"`); `test_write_keeps_existing_mode` (0o644 stays); `test_write_refuses_file_changed_since_read` (modify after `read_config` → `config_changed`, content is the modified one); `test_write_refuses_file_created_since_read`; `test_write_in_unwritable_dir_exits_ten` (chmod 0o500 dir, skip as root; `error.diagnostic.exit_code() == ExitCode.ENVIRONMENT_UNUSABLE`).

`test_default_config_path_honours_xdg_config_home` (`{"XDG_CONFIG_HOME": "/x", "HOME": "/h"}` → `/x/opencode/opencode.json`) and `..._falls_back_to_home` (`{"HOME": "/h"}` → `/h/.config/opencode/opencode.json`).

- [ ] **Step 2: Run them and see them fail**

Run: `pytest tests/unit/test_opencode_config.py -q --basetemp=$PWD/build/pt-connect`
Expected: collection error, `ModuleNotFoundError: yonga.adapters.opencode_config`

- [ ] **Step 3: Implement `src/yonga/adapters/opencode_config.py`** with the interfaces above. The module docstring explains the rules (only yonga's keys, no JSONC rewrite, atomic replace, compare before replace), matching the docstring density of `application/self_install.py`.

- [ ] **Step 4: Run the tests and see them pass**

Run: `pytest tests/unit/test_opencode_config.py -q --basetemp=$PWD/build/pt-connect`
Expected: all pass

- [ ] **Step 5: Lint, type-check, commit**

```bash
ruff check src/yonga/adapters/opencode_config.py tests/unit/test_opencode_config.py && mypy src tests
git add src/yonga/adapters/opencode_config.py tests/unit/test_opencode_config.py
git commit -m "feat(connect): read, edit and atomically write OpenCode's config"
```

### Task 2: Connect service: checks, entry, plan and apply

**Files:**
- Create: `src/yonga/util/net.py`
- Create: `src/yonga/application/connect.py`
- Modify: `src/yonga/cli.py:1890-1902`. Delete `_is_loopback` and `_url_host` and import `is_loopback`/`url_host` from `yonga.util.net` at their call sites.
- Modify: `src/yonga/application/chat.py`. Add a `ChatSession.runtime_properties` property next to `device`.
- Modify: `src/yonga/application/wiring.py`. Add `connect_service()`.
- Test: `tests/unit/test_connect_service.py`

**Interfaces:**
- Consumes: Task 1's `ConfigSnapshot`, `read_config`, `check_provider`, `with_model`, `write_config`.
- Produces:
  - `yonga.util.net.is_loopback(host: str) -> bool` and `url_host(host: str) -> str`, moved verbatim.
  - `ChatSession.runtime_properties -> Mapping[str, str]` (a copy of `self._properties`).
  - In `connect.py`:
    - `OPENCODE_MIN_PROMPT_TOKENS: int = 8192` and `RECORD_FILENAME: str = "opencode-connections.json"`
    - `class ConnectAction(str, Enum)` with values `added`, `updated`, `already_connected`
    - `@dataclass(frozen=True) class Endpoint(host: str = "127.0.0.1", port: int = 8000)` with properties:
      - `provider_key`: `yonga-<port>` when `is_loopback(host)`, else `yonga-<host>-<port>` with `[^A-Za-z0-9_-]` replaced by `-`
      - `base_url`: `http://{url_host(host)}:{port}/v1`
      - `label`: `{url_host(host)}:{port}`
    - `class SessionView(Protocol)`: `manifest`, `device`, `pipeline`, `artifact_dir`, `tool_calling`, `tool_calling_refusal`, `runtime_properties`, all as on `ChatSession`.
    - `class ConnectionRecord(BaseModel)`: `config: str`, `provider: str`, `artifact_id: str`, `model: dict[str, Any]`, `created_provider: bool`, `provider_fields: dict[str, Any] | None`, `written_at: datetime`
    - `class ConnectionRecords`, built as `ConnectionRecords(path: Path)`:
      - `load() -> list[ConnectionRecord]`: a missing or unreadable file reads as `[]`
      - `save(records: Sequence[ConnectionRecord]) -> bool`: atomic; returns `False` instead of raising on `OSError`
    - `@dataclass(frozen=True) class ConnectPlan`: `action: ConnectAction`, `snapshot: ConfigSnapshot`, `endpoint: Endpoint`, `artifact_id: str`, `entry: dict[str, Any]`, `previous_entry: dict[str, Any] | None`, `provider_fields: dict[str, Any]`, `document: dict[str, Any]`, `created_provider: bool`, `serve_command: str`
    - `@dataclass(frozen=True) class ConnectResult`: `created_config: bool`, `record_written: bool`
    - `class ConnectService`, built as `ConnectService(*, open_session: Callable[[str, str | None], SessionView], artifacts: Callable[[], Sequence[ArtifactSummary]], tool_families: Sequence[str], records: ConnectionRecords, default_max_new_tokens: int, env: Mapping[str, str])`. Methods:
      - `plan_connect(reference: str, *, endpoint: Endpoint, device: str | None, context: int | None, config_path: Path) -> ConnectPlan`
      - `apply(plan: ConnectPlan) -> ConnectResult`: for `already_connected` it writes nothing and returns `ConnectResult(False, True)`
  - `Wiring.connect_service() -> ConnectService`. Wire `open_session` to `chat_service(command="serve").open_session(ref, options=ChatOptions(device=device, command="serve"))`, `artifacts` to `artifact_store().list_artifacts`, `tool_families` to the sorted, de-duplicated `model_types` of `registry().chat` rules whose `tool_calling` is set, `records` to `ConnectionRecords(paths.data_dir / RECORD_FILENAME)`, `default_max_new_tokens` to `settings.max_new_tokens`, and `env` to `os.environ`.

Check order inside `plan_connect` (spec section 4): open the session (which resolves the artifact and device), then tools, then fixed prompt length, then context, and only then read the config. The suggestions work like this:
- **`tools_unsupported`:** opens each other `ready` artifact from `artifacts()` with `device=None` and lists up to 3 whose session has `tool_calling`, as `yonga connect opencode <id>`. With none, the remediation names `tool_families`.
- **`prompt_too_short`:** names the first `ready` artifact with the same `repo_id` whose `recommended_device` is `GPU` or `CPU`, else `yonga prepare <repo_id> --device gpu`.

Entry values follow spec section 5:
- **`name`:** `f"{repo_id} · {device} · {profile}"`.
- **`limit.context`:** read from `artifact_dir/"config.json"`, using `max_position_embeddings` or else `text_config.max_position_embeddings`.
- **`limit.output`:** `manifest.recommended_runtime.max_new_tokens` or `default_max_new_tokens`.
- **Fixed prompt length:** `limit.input = int(MAX_PROMPT_LEN)` and `context = input + output`.
- **`serve_command`:** `yonga serve <id>`, plus ` --port P` when not 8000, ` --host H` when not `127.0.0.1`, and ` --device d` when a device was given.

A `config_not_json` from `read_config`/`check_provider` is re-raised with one more remediation line: the JSON to paste by hand, `render({"provider": {key: {**provider_fields, "models": {id: entry}}}})`.

`action` is `added` when the model id is absent from the provider, `already_connected` when entry and provider fields are equal, and `updated` otherwise. `apply` calls `write_config`, then replaces any record with the same `(config, provider, artifact_id)` and saves.

- [ ] **Step 1: Write the failing tests**

Use a `FakeSession` dataclass that satisfies `SessionView` and a temporary `artifact_dir` whose `config.json` is written by the test. Build the service with `env={}` unless a test says otherwise.

- `test_tool_less_model_is_refused_before_the_config_is_read`: `tool_calling=None`, `tool_calling_refusal="No tool-call format is declared for qwen3_5."`, and the config path is a directory that would fail to read. Expect `ConfigurationError` with code `connect.opencode.tools_unsupported`, a message containing the refusal text, and a remediation containing `yonga connect opencode <other id>` for a second, tool-capable fake artifact.
- `test_tool_less_model_without_alternatives_names_families`: remediation contains `qwen3` from `tool_families=["qwen2", "qwen3"]`.
- `test_npu_prompt_of_1024_is_refused_with_the_gpu_artifact`: `runtime_properties={"MAX_PROMPT_LEN": "1024"}`, device `NPU`, and the same repo prepared for GPU as `gpuid`. Expect code `prompt_too_short`, `1024` and `8192` in the message, and `yonga connect opencode gpuid` in the remediation.
- `test_npu_prompt_without_gpu_artifact_suggests_prepare`: `yonga prepare Qwen/Qwen3-4B --device gpu`.
- `test_npu_prompt_of_8192_writes_input_limit`: `entry["limit"] == {"context": 8192 + 4096, "input": 8192, "output": 4096}`.
- `test_context_is_read_from_config_json` (40960) and `..._from_text_config`. `test_context_flag_overrides`. `test_missing_context_is_refused` (code `context_unknown`, `--context` in remediation).
- `test_entry_shape` asserts the exact provider fields and entry from spec section 5 for Qwen3-4B on GPU (`yonga-8000`, `http://127.0.0.1:8000/v1`, name `Qwen/Qwen3-4B · GPU · gpu-int4-asym-g128-r08`, `tool_call: True`).
- `test_api_key_env_writes_placeholder_not_value`: env `{"YONGA_API_KEY": "s3cret"}` gives `options["apiKey"] == "{env:YONGA_API_KEY}"`, and `"s3cret"` is not in `render(plan.document)`.
- `test_provider_key_for_other_host_and_ipv6`: `Endpoint("10.0.0.5", 9000).provider_key == "yonga-10-0-0-5-9000"`, `Endpoint("::1", 8000).provider_key == "yonga-8000"`, and `Endpoint("fd00::1", 8000).base_url == "http://[fd00::1]:8000/v1"`.
- `test_second_run_is_already_connected` (apply, plan again; the second action is `already_connected` and the file mtime is unchanged).
- `test_changed_limits_are_updated_with_previous_entry`.
- `test_same_artifact_on_second_port_adds_second_provider` (Review Focus 4: both `yonga-8000` and `yonga-8001` present, the first unchanged).
- `test_ambiguous_reference_error_passes_through` (Review Focus 5): `open_session` raises the store's `ConfigurationError`, and the same object propagates.
- `test_apply_records_the_connection` (record has `created_provider=True` and the exact `model`). `test_record_write_failure_is_reported_not_raised` (`ConnectionRecords.save` returns `False` for a record path inside a read-only directory, and `apply` returns `record_written=False`).
- `test_jsonc_config_refusal_carries_the_snippet`: config `{ // c\n}`; the remediation contains `"073d5d7c8ea75283"` and `"@ai-sdk/openai-compatible"`, and the file is unchanged.
- `test_serve_command`: `yonga serve id`, and `yonga serve id --port 8001 --host 0.0.0.0 --device gpu`.

- [ ] **Step 2: Run them and see them fail**

Run: `pytest tests/unit/test_connect_service.py -q --basetemp=$PWD/build/pt-connect`
Expected: `ModuleNotFoundError: yonga.application.connect`

- [ ] **Step 3: Move the two host helpers to `src/yonga/util/net.py`, add `ChatSession.runtime_properties`, and implement `connect.py` and `Wiring.connect_service()`** as specified above.

- [ ] **Step 4: Run the new tests and the serve and CLI tests that use the moved helpers**

Run: `pytest tests/unit/test_connect_service.py tests/unit/test_cli.py tests/unit/test_chat_service.py -q --basetemp=$PWD/build/pt-connect`
Expected: all pass

- [ ] **Step 5: Lint, type-check, commit**

```bash
ruff check . && mypy src tests
git add src/yonga/util/net.py src/yonga/application/connect.py src/yonga/application/chat.py src/yonga/application/wiring.py src/yonga/cli.py tests/unit/test_connect_service.py
git commit -m "feat(connect): check an artifact against OpenCode and plan its provider entry"
```

### Task 3: Connect service: removal

**Files:**
- Modify: `src/yonga/application/connect.py`
- Test: `tests/unit/test_connect_service.py`

**Interfaces:**
- Consumes: Task 1's `without_model`, `read_config`, `write_config`, and Task 2's `ConnectionRecords` and `ConnectService`.
- Produces:
  - `class RemoveAction(str, Enum)` with values `removed`, `not_connected`
  - `@dataclass(frozen=True) class RemovePlan`: `action: RemoveAction`, `snapshot: ConfigSnapshot`, `document: dict[str, Any]`, `removed: list[tuple[str, str]]`, `kept: list[tuple[str, str, str]]` (provider, model, reason), `stale: list[ConnectionRecord]`
  - `ConnectService.plan_remove(artifact_id: str, *, config_path: Path) -> RemovePlan`: needs only the record, never the artifact store
  - `ConnectService.apply_remove(plan: RemovePlan) -> bool`: writes when `removed` is non-empty, drops the removed and stale records, and returns whether the record was saved
  - `ConnectService.is_connected(artifact_id: str) -> bool`

Records match on `artifact_id` and the resolved `config_path`.
- **Model entry equal to the record:** removed, using `drop_provider_when=provider_fields` when `created_provider`.
- **Model entry differs:** kept with reason `"edited since yonga wrote it"`.
- **Model entry already gone:** goes to `stale`.

No matching record means `not_connected`.

- [ ] **Step 1: Write the failing tests**

- `test_remove_deletes_only_the_recorded_entry` (another model in the same provider survives).
- `test_remove_drops_provider_it_created_once_empty`.
- `test_remove_keeps_provider_the_user_extended` (the user added `"headers"` to the provider: the model goes, the provider stays).
- `test_remove_keeps_an_edited_entry` (the user changed `limit.context`: `kept == [("yonga-8000", id, "edited since yonga wrote it")]`, the entry still in the file).
- `test_remove_without_record_is_not_connected_and_writes_nothing`.
- `test_remove_after_artifact_deleted_works` (`artifacts()` returns `[]` and `open_session` raises; the removal still succeeds).
- `test_remove_both_ports` (Review Focus 4).
- `test_remove_clears_stale_records` (the entry was deleted by hand; the record is dropped, `action == not_connected`, nothing written).
- `test_is_connected`.

- [ ] **Step 2: Run them and see the new ones fail**

Run: `pytest tests/unit/test_connect_service.py -q --basetemp=$PWD/build/pt-connect`
Expected: the new tests fail with `AttributeError: ... 'plan_remove'`

- [ ] **Step 3: Implement the three methods in `connect.py`**

- [ ] **Step 4: Run the tests and see them pass**

Run: `pytest tests/unit/test_connect_service.py -q --basetemp=$PWD/build/pt-connect`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
ruff check . && mypy src tests
git add src/yonga/application/connect.py tests/unit/test_connect_service.py
git commit -m "feat(connect): remove only the OpenCode entries yonga wrote"
```

### Task 4: CLI: `connect opencode`, `--remove`, and the note in `remove`

**Files:**
- Modify: `src/yonga/cli.py`. Add a `connect_app` Typer sub-app after `self_app` (`help="Register a prepared artifact with a client that talks to 'yonga serve'."`, `no_args_is_help=True`), with a `# connect` section holding `connect_opencode` and `_connect_opencode`/`_disconnect_opencode`, and a note in `_remove`.
- Test: `tests/unit/test_cli_connect.py`, plus `"connect"` added to the list in `tests/unit/test_cli.py::test_help_lists_every_documented_command`.

**Interfaces:**
- Consumes: Task 2/3 `ConnectService`, `Endpoint`, `ConnectAction`, `RemoveAction`, and `default_config_path`.
- Produces: the command line from spec section 3. `ARTIFACT` is required; `--remove` is a flag that changes its meaning. With `--remove`, any of `--host`, `--port`, `--device` or `--context` is a usage error (exit 70).

Behaviour:
- **Default path:** `--config` defaults to `default_config_path(os.environ)`; a given path gets `expanduser()` and `resolve()`.
- **Before the question:**
  - For `added`, print the target file, `+ provider.<key> → <base_url>` (only when the provider is created) and `+ model <id> (context C, output O[, input I])`.
  - For `updated`, print `~ model <id>` followed by `before`/`after` JSON lines.
- **Question:** `console.confirm("Write this to <path>?", default=False, assume_yes=yes)`. A refusal raises `CancelledError` `connect.opencode.declined` with the `state.rerun('--yes')` remediation, as `_self_install` does.
- **`already_connected`:** prints `<id> is already connected in <path>.` and asks nothing.
- **After writing:** print `Connected <id> to OpenCode as <key>/<id>.` and `Start the server with: <serve_command>`. When `record_written` is false, add a `console.detail` line naming the record path.
- **`--json` documents:** keys exactly as in spec section 7. `connect` emits `action`, `config`, `created_config`, `provider`, `created_provider`, `model`, `base_url`, `limit`, `serve_command` and `exit_code`; `--remove` emits `action`, `config`, `removed`, `kept` and `exit_code`.
- **Removal with entries kept:** print each one with its reason and exit 70 (`ExitCode.REGISTRY`); the JSON `exit_code` is 70 as well.
- **`_remove`:** after a successful removal, when `state.graph.connect_service().is_connected(report.artifact_id)`, print `This artifact is still listed in OpenCode; remove it there with: yonga connect opencode --remove <id>` (text mode only).

- [ ] **Step 1: Write the failing tests**

Use `invoke` from `tests/unit/test_cli.py`, with a `ConnectWiring(FakeWiring)` subclass whose `connect_service()` returns a `ConnectService` built over fake sessions (reuse Task 2's `FakeSession`) and `--config` under `tmp_path`.
- `test_connect_asks_then_writes`: stdin `"y\n"`, exit 0, the file contains the model, stdout contains `yonga serve`.
- `test_connect_declined_writes_nothing_and_exits_sixty`: stdin `"n\n"`.
- `test_connect_json_without_yes_exits_sixty` and `test_connect_json_with_yes_emits_one_document` (`one_document` keys equal the set from the spec).
- `test_connect_already_connected_asks_nothing`.
- `test_connect_tool_less_exits_seventy_with_suggestion`.
- `test_remove_flag_rejects_port`.
- `test_connect_remove_round_trip` (connect `--yes`, then `--remove --yes`; the provider is gone and the file keeps `$schema`).
- `test_remove_kept_entry_exits_seventy`.
- `test_artifact_remove_prints_the_opencode_note`.
- The help test includes `connect`.

- [ ] **Step 2: Run them and see them fail**

Run: `pytest tests/unit/test_cli_connect.py tests/unit/test_cli.py::test_help_lists_every_documented_command -q --basetemp=$PWD/build/pt-connect`
Expected: `No such command 'connect'`

- [ ] **Step 3: Implement the CLI changes described above**

- [ ] **Step 4: Run the whole unit suite**

Run: `pytest tests/unit -q --basetemp=$PWD/build/pt-connect`
Expected: all pass (existing skips unchanged)

- [ ] **Step 5: Commit**

```bash
ruff check . && ruff format --check . && mypy src tests
git add src/yonga/cli.py tests/unit/test_cli_connect.py tests/unit/test_cli.py
git commit -m "feat(cli): yonga connect opencode, with --remove"
```

### Task 5: Documentation and end-to-end check

**Files:**
- Modify: `docs/CLI_SPEC.md`. Add a `## 12d. connect` section after `12c. self`, condensing spec sections 3-7, and add the new diagnostic codes wherever the spec lists codes.
- Modify: `README.md`. In the command table, add a `yonga connect opencode ARTIFACT` row; change "Seventeen commands; `registry` has seven subcommands, `env`, `cache` and `self` two each" to "Eighteen commands; `registry` has seven subcommands, `env`, `cache` and `self` two each, `connect` one".
- Modify: `docs/USAGE.md`. Add "### Use it from OpenCode" after "Serve it to other tools": the two commands, what is refused and why, and `--remove`.
- Modify: `website/docs/en/guide/serve.md` and `website/docs/tr/guide/serve.md`. Add the same section in English and Turkish.
- Modify: `CHANGELOG.md`. Add an entry under Unreleased.

- [ ] **Step 1: Write the documentation changes above**

- [ ] **Step 2: Build the site in strict mode**

Run: `pip install -q -r website/requirements.txt && mkdocs build --strict -f website/mkdocs.yml -d $PWD/build/site`
Expected: `Documentation built`, no warnings

- [ ] **Step 3: Check end to end on the reference host, with a scratch config, without touching `~/.config/opencode`**

```bash
C=$PWD/build/oc/opencode.json
.venv/bin/yonga connect opencode 073d5d7c8ea75283 --port 8765 --config $C --yes
.venv/bin/yonga connect opencode 6faca50f61838aa5 --config $C --yes; echo "exit $?"   # expect 70, prompt_too_short, suggests 073d5d7c8ea75283
.venv/bin/yonga serve 073d5d7c8ea75283 --port 8765 &   # record the PID; stop it with kill PID afterwards
cd build/oc && echo hello > README.txt && OPENCODE_CONFIG=$C opencode run --standalone -m yonga-8765/073d5d7c8ea75283 "Read README.txt and say what it says."
```

Expected:
- the first command exits 0
- the second exits 70
- `opencode run` shows a `Read README.txt` tool call and an answer quoting "hello"

Then run `connect opencode --remove 073d5d7c8ea75283 --config $C --yes` and check that the file keeps only `$schema`. Report the commands and their output with the change.

- [ ] **Step 4: Clean up and commit**

```bash
rm -rf build/oc build/site build/pt-connect
git add docs/CLI_SPEC.md README.md docs/USAGE.md website/docs/en/guide/serve.md website/docs/tr/guide/serve.md CHANGELOG.md
git commit -m "docs: yonga connect opencode"
```
