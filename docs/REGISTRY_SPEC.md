# REGISTRY_SPEC.md

## 1. Purpose

The compatibility registry turns fast-changing knowledge about model architectures, OpenVINO versions, Intel devices, exporter requirements, quantization profiles, known failures, workarounds, and trusted derivatives into data rather than hard-coded conditionals.

The registry is advisory + policy input. Runtime probes remain important when evidence is unknown.

## 2. Files

The built-in rules ship inside the package, in `src/yonga/registry/builtin/`, and load in this order
as one layer:

```text
src/yonga/registry/builtin/
  architectures.yaml
  profiles.yaml
  issues.yaml
  derivatives.yaml
  repairs.yaml
  chat.yaml
  gguf.yaml
```

Each file is one registry document: a top-level `schema_version` and one or more lists of rules
under the section names `architectures`, `profiles`, `issues`, `repairs`, `derivatives`, `models`,
`chat` (also accepted as `chat_profiles`) and `gguf`. The file name does not decide what a document
may contain, so a document can mix sections; there is no built-in `models.yaml`, and exact model
rules (section 9) can go in any document. A remote bundle (section 12.1) and the user's
`registry-overrides/` directory hold documents of the same shape. `registry/examples/qwen3_5.yaml`
in the repository is a complete example document for the Qwen3.5 reference case. The loader
normalizes every document into typed records.

The examples in sections 3 to 7 (except 5.1) and 9 show single rules; in a document each sits as a
list item under its section, as the examples in sections 5.1, 8, 8a and 8b do.

## 3. Common fields

All rules have:

```yaml
id: unique-stable-id
schema_version: 3        # 1 and 2 still load; 2 adds profile size bands (section 5.1),
                         # 3 adds a chat profile's tool_calling (section 8b)
status: active           # active | deprecated | experimental
summary: short text
sources:
  - kind: official_docs  # official_docs | upstream_issue | tested | model_card | community
    url: https://...
    note: optional
```

Version constraints use PEP 440 semantics where possible.

## 4. Architecture rule

Example:

```yaml
id: arch-qwen3_5
schema_version: 1
match:
  architectures: [Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM]
  model_types: [qwen3_5]
  aliases: [qwen35]
export:
  transformers: "==5.2.*"
  pipeline: vlm
  trust_remote_code: false
devices:
  cpu:
    status: supported
    evidence: official
  gpu:
    status: supported
    evidence: official
  npu:
    status: unsupported
    evidence: tested
    constraints:
      openvino: ">=2026.2,<2027"
    reason: current_npu_compiler_graph_limitation
recommended_device: gpu
```

Status values:

- `supported` — documented/tested;
- `unsupported` — known incompatible;
- `experimental` — works under restricted conditions;
- `unknown` — no reliable knowledge.

`aliases` lists spellings the architecture is known by outside `config.json`. It exists for repositories that declare no identity of their own: a GGUF-only repository ships no `config.json`, so `model_types` and `architectures` can never match it, and without an alias every device is reported `unknown`.

An alias is compared only against *identity hints*, and the only hint is the `architecture` string in the GGUF header the Hub reports — a value read out of the weight file itself. It is never compared against `model_types` or `architectures`, so an alias cannot widen what a declared identity matches. Hints are consulted only when the repository declares no identity at all — an empty `model_type` and an empty `architectures` list. A header is evidence in the *absence* of a `config.json`, never a correction of one, so a repository that declares an identity no rule lists resolves to nothing rather than to whatever its header names.

Repository tags are deliberately never used as hints. A tag such as `qwen3.5` or `qwen` is free-form text an uploader typed; identifying a model from it would be exactly the "do not infer that a similarly named repository is equivalent" inference `SPEC.md` section 8.1 forbids. A tag may well be true, but it is not evidence.

A rule reached through an alias scores below every rule that matched a declared identity; see section 11.

## 5. Profile rule

```yaml
id: gpu-int4-asym-g128
schema_version: 1
device: gpu
format: openvino_ir
quantization:
  weight_format: int4
  symmetric: false
  ratio: 1.0
  group_size: 128
constraints:
  openvino: ">=2026.2"
priority: 100
```

Profiles may contain runtime properties separately from export properties: `runtime` holds
OpenVINO GenAI pipeline properties (`ATTENTION_BACKEND`, `MAX_PROMPT_LEN`), and
`export_extra_args` extra `optimum-cli` arguments. `rationale` says why the profile exists.

`export_extra_args` tunes an export; it never decides what is exported, where it is written or
whether a repository's code runs. It may not name:

- `--model` or `-m` (also with its value attached to `-m`), `--task`,
  `--trust-remote-code`, `--model-kwargs`, `--cache_dir` (or `--cache-dir`), `--token` or
  `--quantization-statistics-path`, which yonga either sets itself or keeps out of registry data;
- the quantization options yonga derives from the profile's `quantization` section:
  `--weight-format`, `--sym`, `--ratio`, `--group-size`, `--backup-precision`, `--dataset`,
  `--all-layers`, `--awq` and `--scale-estimation`;
- a bare `--`, which would turn the arguments after it into positionals.

The check reads an argument the way `argparse` does: `--name=value` counts as `--name`, and a long
option that is a prefix of a reserved one (`--trust-remote`, `--weight`) counts as that option.
Arguments without a leading dash are values and are not checked. The same restriction applies to
`export.extra_args` in an architecture rule (section 4). A file that breaks it fails to load with
`registry.schema_violation`; a profile built in code rather than loaded from a file is refused when
the export command is built, with `export.reserved_extra_args`.

### 5.1 Model-size bands (schema version 2)

A profile can apply to a range of model sizes only. The band is part of `applies_to` and filters,
it never ranks: among the profiles that match, priority decides, so a banded profile is given a
higher priority than the device's default to win inside its band.

```yaml
schema_version: 2
profiles:
  - id: cpu-int4-asym-g128-r08
    schema_version: 2
    device: cpu
    priority: 110          # above the unbanded default (100), so it wins inside the band
    evidence: tested       # required on a banded profile
    applies_to:
      parameter_count:
        below: 5B          # also: at_least: 600M; both: [at_least, below)
    quantization: {weight_format: int4, symmetric: false, ratio: 0.8, group_size: 128}
```

- **Unit.** Parameters as the Hugging Face Hub counts them from the safetensors headers
  (`yonga inspect` shows the number): embeddings, an untied output layer and any vision tower
  included.
- **Spelling.** An integer (`2000000000`, `2_000_000_000`) or a decimal with a `K`, `M`, `B` or `T`
  suffix (`600M`, `1.5B`). Floats are refused: YAML reads `2e9` as text and `2.0e+9` as a float, and
  a threshold must not depend on which one was typed.
- **Interval.** Half-open, `at_least <= count < below`, so a model exactly at a threshold belongs to
  the upper band and adjacent bands never overlap.
- **Unknown size.** A model whose size is unknown is outside every band. That includes a repository
  with no safetensors metadata, a checkpoint that stores packed integer weights (more than 1 % of its
  elements in an integer dtype: GPTQ, AWQ, bitsandbytes, whose element count understates the model)
  and one whose `config.json` declares a `quantization_config`. A GGUF import and an OpenVINO copy
  keep their own quantization, so their size never chooses a profile either.
- **Only profiles.** Every other rule kind refuses `parameter_count`.
- **Evidence.** `evidence` (`official`, `tested`, `reported`, `inferred`) states how well the band
  is supported; a banded profile must set it.
- **Version.** A banded profile needs `schema_version: 2`. A rule without its own
  `schema_version` takes its document's, and no rule may declare a newer version than its document.

Validation adds two checks. Every device that has a banded profile must also have a usable profile
with no condition at all, for models of unknown size. And a band that can never win is reported:
when an unbanded profile on the same specificity rung matches everything the band's other criteria
match, at a higher priority, and no architecture or model rule names the banded profile.

### 5.2 Group sizes the model cannot take

Group-wise 4-bit quantization splits each weight row along its input channels into groups of
`group_size`, and NNCF refuses any 4-bit layer whose channel count the group size does not divide
(`InvalidGroupSizeError`). The channel counts are in `config.json` (`hidden_size`,
`intermediate_size`, or the same keys under `text_config`), so automatic selection passes over a
profile whose refusal is certain and says so in "Chosen because":

- `hidden_size` feeds the q/k/v and gate/up projections, well over half of a decoder's weights, so
  a group size that does not divide it fails at any `ratio` of 0.5 or more;
- `intermediate_size` feeds only the down projections, which a mixed ratio may leave at 8 bits, so
  it rules a profile out only at `ratio: 1.0`.

A named profile (`profiles:` in an architecture or model rule) is passed over the same way. When
no profile for the device is left, `prepare` stops before the download with
`export.quantization.group_size_mismatch` (exit 70), unless the device was chosen automatically:
then the next device in the ranking is tried. A profile requested with `--profile` is refused the
same way, before the download and with the same code, naming the profiles that fit, for the device
being prepared and for the profile's own device (the automatic choice first, as the user's own
command with `--profile` and `--device` changed); it is never swapped for one of them. Example:
SmolLM2-360M has a hidden size of 960, so the CPU gets `cpu-int8-asym-ref`, the NPU
`npu-int4-sym-cw`, and no GPU profile fits.

Only the widths `config.json` declares are checked. A refusal on another layer is not predicted:
Qwen3.5 0.8B-4B fail `cpu-int4-asym-g128` on a one-channel `rotary_emb` weight while the 9B of the
same architecture prepares with it, so that is recorded as a known limitation, not a refusal.

## 6. Issue/signature rule

```yaml
id: qwen35-npu-fp16-bufferize
schema_version: 1
match:
  architectures: [qwen3_5]
  device: npu
  exception_contains:
    - "OneShotBufferizeVPU2VPUIP failed : Only supports FP16"
classification:
  code: npu.compiler.unsupported_graph.qwen35
  category: compiler_unsupported_graph
  environment_broken: false
  retry_same_device: false
recommendation:
  fallback_devices: [gpu, cpu]
  message: >-
    The current NPU compiler cannot compile this Qwen3.5 graph.
```

Exact signatures should be preferred over broad regexes. Regexes require tests preventing false positives.

**Placeholders in remediation.** A rule is written before any job exists, so a remediation step
names what it cannot know with the upper-case words `MODEL` (the repository) and `JOB_ID` (the
job): `yonga prepare MODEL --device gpu`, `yonga logs JOB_ID --raw`. Wherever the failure knows the
value, it is filled in before the step is shown, in the terminal and in `--json` alike, so the step
can be copied as it stands; an unknown value leaves the word, which still reads as a placeholder.
Only the whole word is replaced (`MODEL_DIR` is left alone). A step that then repeats an earlier one,
or ends in the same `yonga …` command, is dropped, so a rule's "run this model on the GPU instead:
yonga prepare MODEL --device gpu" and `prepare`'s own "Prepare for GPU instead: …" print once.

The words are filled per command. `chat`, `bench` and `probe` run an artifact that already exists:
a step ending in `yonga prepare MODEL --device D` for a device that artifact passed its probe on
becomes "Run this artifact on the D instead: yonga chat ID --device d" (the command that failed,
with the artifact id and the other device), and so does the `Run:` line under the fallback devices;
a device the artifact was never probed on keeps the `prepare` advice. A command that opens no job
(`chat`, `serve`, `eval`, `publish`) has no id for `JOB_ID`, so a rule's step naming it is not
shown there; `--debug` shows the raw text the step points at.

## 7. Repair rule

```yaml
id: ov-tokenizer-precision-string-casing
schema_version: 1
match:
  files:
    - openvino_tokenizer.xml
    - openvino_detokenizer.xml
  contains: 'precision="STRiNG"'
  openvino: ">=2026.4,<2027"
repair:
  implementation: tokenizer_string_precision
  safety: automatic
validation:
  requires_tokenizer_read: true
sources:
  - kind: tested
    note: Reproduced with a Qwen3.5 OpenVINO export.
```

The YAML selects a code implementation; arbitrary find/replace instructions must not be executed from remote registry data.

## 8. Derivative source rule

```yaml
derivatives:
  - id: hauhaucs-qwen35-safetensors
    schema_version: 1
    requested_repo: HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive
    candidates:
      - repo_id: nDimensional/Qwen3.5-9B-Uncensored-Safetensors
        relationship: lossless_conversion
        trust: suggest
        evidence:
          - kind: model_card
            url: https://huggingface.co/nDimensional/Qwen3.5-9B-Uncensored-Safetensors
```

`candidates` lists the repositories that may stand in for `requested_repo`, each with its own
`evidence`. The rule's own `sources` (section 3) remain available for the provenance of the rule
itself.

`trust` values:

- `suggest`: display and require confirmation;
- `auto_if_trusted`: may be selected in auto mode only when registry source itself is trusted and lineage constraints pass;
- `never_auto`: informational only.

The built-in registry uses `suggest` for every third-party derivative.

## 8a. GGUF rule

What the installed OpenVINO GenAI can import directly from a GGUF file. Both conditions are read
from the file's own header, never from its name:

```yaml
gguf:
  - id: gguf-openvino-genai-2026
    header_architectures: [llama, qwen2, qwen3]    # general.architecture values
    tensor_types: [F32, F16, Q4_0, Q4_K, Q6_K, Q8_0]
    preferred_files: [Q4_K_M, Q4_0, Q6_K, Q8_0]    # labels tried, in order, when no file is named
    constraints:
      openvino_genai: ">=2026.4,<2027"
```

A file qualifies only when its architecture is listed **and** every one of its tensors is of a
listed type. A `Q4_K_M` label is not enough: one such file holds only `Q4_K` and `Q6_K`, another
also holds `Q5_0`, which the reader cannot convert. List only what has been verified by importing
and running a file on real hardware. A rule whose constraint cannot be checked (GenAI not
installed) does not count, because unknown is not support. The highest-priority usable rule wins;
a remote bundle or a user override can extend support without a code change.

## 8b. Chat profile rule

How a model family expects a conversation to be driven: its thinking variable, reasoning wrappers,
generation defaults, end markers and, when measured, how it writes tool calls. `chat` and `serve` use the matching rule; without one, the
copy `prepare` recorded in the artifact's manifest (which includes `stop_strings`), and otherwise a
conservative built-in default.

```yaml
chat:
  - id: chat-qwen3_5
    match:
      model_types: [qwen3_5]
    supports_thinking_flag: true          # true | false (veto) | unset (detect from template)
    thinking_template_variable: enable_thinking
    default_thinking: false
    reasoning_wrappers:
      - ["<think>", "</think>"]
    defaults: {temperature: 0.7, top_p: 0.8, top_k: 20, max_new_tokens: 4096}
    thinking_defaults: {temperature: 0.6, top_p: 0.95}
    stop_strings: []                      # model-level end markers, applied natively
```

`stop_strings` lists raw-text end markers, such as an end-of-turn marker a template writes as text
rather than as a special token. They are passed to OpenVINO GenAI's `GenerationConfig.stop_strings`,
which stops at the first one and leaves it out of the answer. Empty and duplicate entries are
dropped when the rule loads. GenAI matches them in the raw output, inside a reasoning block too, so
list only markers that end the model's turn wherever they appear. A client's `stop` sequences are a
different thing, matched by yonga on the visible answer (`docs/CLI_SPEC.md` section 14). An empty
list leaves generation and a benchmark's settings fingerprint exactly as they were; no built-in rule
declares any yet.

### 8b.1 Tool calling (schema version 3)

`tool_calling` says how a family writes tool calls. `yonga serve` accepts OpenAI `tools` only for a
model whose chat profile carries one (`docs/CLI_SPEC.md` section 14), so the entry is a claim that
the format was seen, not a template's promise:

```yaml
schema_version: 3
chat:
  - id: chat-qwen3
    schema_version: 3
    match:
      model_types: [qwen3]
    tool_calling:
      format: tagged_json              # a parser compiled into yonga: tagged_json | json_list
      open: "<tool_call>"              # the marker in front of a call (or of the call list)
      close: "</tool_call>"            # the marker after it
      name_field: name                 # default; the call object's key for the function name
      arguments_field: arguments       # default; its key for the arguments
      template_contains: "<tool_call>" # optional; the artifact's chat template must contain it
      evidence: tested                 # tested | official, required
      sources:                         # at least one, required
        - kind: tested
          note: 2026-10-09, reference host, ... answered <tool_call>{...}</tool_call>
```

- **Formats.** `format` names one of the parsers compiled into yonga (`KNOWN_TOOL_CALL_FORMATS` in
  `src/yonga/registry/schema.py`); registry data selects and parameterises a parser and can never
  describe one. `tagged_json`: each call is one JSON object between `open` and `close` (both
  required), repeatable, with text allowed around the blocks. `json_list`: the whole answer is one
  JSON array of call objects, after `open` when it is present (a marker that is a special token is
  removed by the detokenizer, so it may be absent) and before `close` when that is set. A call
  object names the function under `name_field` and carries an object of arguments, or JSON text of
  one, under `arguments_field`; missing arguments mean none.
- **Reading.** Calls are read from the visible answer, after the reasoning sanitiser, and only calls
  to the functions the request offered count. Anything that cannot be read that way is the answer's
  text, never a guessed call.
- **Template check.** With `template_contains`, tool calling is enabled for an artifact only when
  its own chat template contains that literal, as the thinking variable is detected: a family rule
  then cannot switch tools on for a repackaged model whose template asks for another format.
- **Evidence.** `evidence` must be `tested` or `official`, and `sources` must say where the format
  was seen; an unmeasured family gets no entry and refuses `tools`.
- **Unknown format.** A rule naming a format this yonga has no parser for still loads (a layer may
  be written for a newer release), its model refuses `tools`, and `yonga registry validate` reports
  it. A remote bundle naming one is refused (`registry.remote.unknown_tool_call_format`), as a
  bundle naming an unknown repair is.
- **Version.** A chat profile with `tool_calling` needs `schema_version: 3`. A rule without its own
  `schema_version` takes its document's, and no rule may declare a newer version than its document,
  as for size bands (section 5.1). A schema-2 yonga refuses a schema-3 file as too new rather than
  as a schema violation.

`yonga registry explain` prints the winning chat profile's format under "Tool calling", or says
that none is declared. The built-in registry declares `tagged_json` for `chat-qwen3`, `chat-qwen2`
and `chat-smollm3` and `json_list` for `chat-granite`; `chat.yaml` records which families were
measured and left out, and why. `chat-qwen3_moe` keeps `chat-qwen3`'s policy without
`tool_calling`, because the Qwen3 MoE models were not measured.

## 9. Exact model rules

Use exact model rules only for behavior that genuinely differs from its architecture. Avoid filling the registry with model-name-specific duplicates.

```yaml
id: model-openvino-qwen35-9b-int4
match:
  repo_ids: [OpenVINO/Qwen3.5-9B-int4-ov]
artifact:
  format: openvino_ir
  already_prepared: true
```

## 10. Device families

The registry should support device-family predicates without hard-coding marketing names into application logic.

Example normalized host tags:

```text
intel.cpu
intel.gpu
intel.gpu.xe2
intel.npu
intel.npu.lunar_lake
intel.core_ultra.series2
```

The Linux hardware adapter is responsible for deriving tags from stable runtime/system properties. If uncertain, omit a tag rather than guess.

## 11. Rule resolution

Specificity score order:

1. exact repo ID;
2. architecture + host family;
3. architecture;
4. architecture reached through an identity alias (section 4);
5. task + host family;
6. generic device profile.

Rung 4 sits below every declared-identity rung on purpose: a rule that recognises the `model_type` a repository actually publishes must always beat one that had to fall back on a hint found outside `config.json`, whatever the two rules' priorities say.

Then apply version constraints and explicit `priority`.

Profiles are resolved in this order: an explicit `--profile` always wins; then a profile named by
the exact-model rule, then one named by the architecture rule; then scoring among the device's
profiles. A named profile is a preference, not an override: it must itself be usable, satisfy its
version constraints and match its own `applies_to`, size band included, or resolution falls
through to scoring. Name a profile in an architecture rule only when the family must differ from
what scoring would choose. The decision is recorded as a `profile_selection` (basis
`explicit`, `named_by_model_rule`, `named_by_architecture_rule`, `size_band` or `device_default`)
whose one sentence the preparation plan shows as "Chosen because" and the manifest keeps in its
notes.

If two same-specificity rules conflict and priority does not resolve them, registry validation fails.

`registry explain` must output:

- candidates;
- rejected rules and reason (version mismatch, architecture mismatch, etc.);
- winning rule(s);
- final decision.

## 12. Registry provenance

Built-in rules are shipped with the package and covered by tests. Every compatibility claim should have at least one provenance entry when practical.

A remote registry is used only when it is signed by a trusted key (section 12.1). yonga never fetches executable repair logic from the network: a repair rule can only name an implementation compiled into yonga.

### 12.1 Remote bundles (implemented)

A remote registry is distributed as one JSON document plus a detached signature:

```json
{
  "format": 1,
  "version": 7,
  "created_at": "2026-09-24T12:00:00+00:00",
  "expires_at": "2026-12-23T12:00:00+00:00",
  "min_tool_version": "0.1.0",
  "description": "NPU driver 1.30 signatures",
  "files": {"issues/npu-1.30.yaml": "schema_version: 1\nissues:\n  - ..."}
}
```

```json
{"format": 1, "key_id": "40eb0328d070643d", "signature": "<base64 Ed25519 signature of the bundle bytes>"}
```

`created_at` and `expires_at` must carry a UTC offset, such as the `+00:00` that
`tools/registry_bundle.py build` writes, or `Z`. A naive timestamp is ambiguous between time zones,
so a bundle with one is refused as malformed (`registry.remote.invalid`).

A bundle is used only when all of the following hold, and they are checked again on every load:

- the signature verifies against a key in `registry-keys/`;
- it has not expired, and its `version` is not lower than the installed one;
- `min_tool_version`, when set, is not newer than the running yonga;
- every file is a relative `.yaml`/`.yml` path and parses with `yaml.safe_load` into the strict
  schema;
- every repair rule names an implementation already compiled into yonga, and every chat profile's
  `tool_calling` names a tool-call parser compiled into it. Registry data selects a built-in repair
  or parser; it can never deliver one.

The bundle forms one layer after the built-in rules and before user overrides: it may replace a
shipped rule by id, and a user override replaces either.

`tools/registry_bundle.py build` sets `min_tool_version` from the newest `schema_version` in the
bundle (schema 2 needs yonga 0.3.0, schema 3 needs 0.4.0), and refuses a lower
`--min-tool-version`. An older yonga then refuses the bundle at `registry update` and keeps the one
it has, instead of installing data it cannot parse. A schema-2 or schema-3 bundle therefore cuts
older releases off from every remote update, including issue signatures; keep bundles at schema 1
until a band or a tool-call format itself must change remotely. A file that declares no
`schema_version` is read at the building yonga's own version, so declare one.

## 13. Registry tests

Required tests:

- schema validation;
- unique IDs;
- version constraint parsing;
- deterministic conflict resolution;
- fixture model -> expected profile;
- known exception -> expected classification;
- derivative never selected silently when `trust=suggest`;
- repair rule selects only its exact repair implementation.
