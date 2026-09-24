# REGISTRY_SPEC.md

## 1. Purpose

The compatibility registry turns fast-changing knowledge about model architectures, OpenVINO versions, Intel devices, exporter requirements, quantization profiles, known failures, workarounds, and trusted derivatives into data rather than hard-coded conditionals.

The registry is advisory + policy input. Runtime probes remain important when evidence is unknown.

## 2. Files

Suggested built-in layout:

```text
registry/
  architectures.yaml
  profiles.yaml
  issues.yaml
  derivatives.yaml
  models.yaml
```

For development it is acceptable to combine them, but the loader must normalize them into typed records.

## 3. Common fields

All rules have:

```yaml
id: unique-stable-id
schema_version: 1
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

Profiles may contain runtime properties separately from export properties.

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
id: hauhaucs-qwen35-safetensors-derivative
schema_version: 1
requested_repo: HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive
sources:
  - repo_id: nDimensional/Qwen3.5-9B-Uncensored-Safetensors
    relationship: lossless_conversion
    trust: suggest
    evidence:
      - kind: model_card
        url: https://huggingface.co/nDimensional/Qwen3.5-9B-Uncensored-Safetensors
```

`trust` values:

- `suggest`: display and require confirmation;
- `auto_if_trusted`: may be selected in auto mode only when registry source itself is trusted and lineage constraints pass;
- `never_auto`: informational only.

MVP should use `suggest` for all third-party derivatives.

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

If two same-specificity rules conflict and priority does not resolve them, registry validation fails.

`registry explain` must output:

- candidates;
- rejected rules and reason (version mismatch, architecture mismatch, etc.);
- winning rule(s);
- final decision.

## 12. Registry provenance

Built-in rules are shipped with the package and covered by tests. Every compatibility claim should have at least one provenance entry when practical.

A future remote registry must be authenticated/signed before automatic use. MVP should not fetch executable repair logic from the network.

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

A bundle is used only when all of the following hold, and they are checked again on every load:

- the signature verifies against a key in `registry-keys/`;
- it has not expired, and its `version` is not lower than the installed one;
- `min_tool_version`, when set, is not newer than the running yonga;
- every file is a relative `.yaml`/`.yml` path and parses with `yaml.safe_load` into the strict
  schema;
- every repair rule names an implementation already compiled into yonga. Registry data selects a
  built-in repair; it can never deliver one.

The bundle forms one layer after the built-in rules and before user overrides: it may replace a
shipped rule by id, and a user override replaces either.

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
