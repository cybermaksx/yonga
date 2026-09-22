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
4. task + host family;
5. generic device profile.

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
