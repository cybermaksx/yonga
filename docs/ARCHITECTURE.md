# ARCHITECTURE.md

## 1. Overview

The architecture separates policy from mechanism. Policy decides *what should be done* (device, source, profile, repair, fallback). Mechanism executes it (Hub download, Optimum export, OpenVINO compile, chat generation).

```text
CLI / TUI
   |
Application services
   |-- DoctorService
   |-- InspectService
   |-- PrepareService
   |-- ChatService
   |-- BenchmarkService
   |
Domain
   |-- ModelIdentity / ModelMetadata
   |-- HardwareInventory
   |-- CompatibilityDecision
   |-- PreparationPlan
   |-- ArtifactManifest
   |-- Diagnostic / FailureClass
   |
Adapters
   |-- HuggingFaceClient
   |-- OpenVINOProbe
   |-- OptimumExporter
   |-- GenAIRuntime
   |-- SystemProbeLinux
   |-- RegistryRepository
   |-- ArtifactStore
   |-- JobLogger
```

No CLI command should directly invoke OpenVINO, Hugging Face, or subprocess APIs. It calls an application service.

## 2. Suggested package layout

```text
src/yonga/
  __init__.py
  __main__.py
  cli.py

  domain/
    models.py
    hardware.py
    compatibility.py
    plans.py
    artifacts.py
    diagnostics.py
    benchmarks.py

  application/
    doctor.py
    inspect_model.py
    prepare.py
    chat.py
    benchmark.py
    resolve.py

  registry/
    loader.py
    schema.py
    resolver.py
    builtin/
      architectures.yaml
      profiles.yaml
      issues.yaml
      derivatives.yaml

  adapters/
    hf.py
    system_linux.py
    openvino_probe.py
    exporter_optimum.py
    runtime_genai.py
    filesystem.py
    subprocess.py

  repairs/
    base.py
    tokenizer_string_precision.py

  ui/
    console.py
    chat_tui.py
    tables.py
    progress.py

  config/
    settings.py
    paths.py

  util/
    hashing.py
    versions.py
    redaction.py
    jsonlog.py

tests/
  unit/
  integration/
  fixtures/
  hardware/
```

## 3. Domain objects

Use immutable dataclasses or Pydantic models where practical.

### ModelIdentity

```python
class ModelIdentity(BaseModel):
    repo_id: str
    revision: str | None = None
```

### ResolvedSource

```python
class ResolvedSource(BaseModel):
    requested: ModelIdentity
    source: ModelIdentity
    relationship: Literal[
        "original", "converted", "lossless_conversion", "quantized"
    ]
    evidence: list[str] = []
    requires_confirmation: bool = False
```

### HardwareInventory

Contains host facts, not recommendations:

- CPU model;
- RAM/swap;
- `OpenVINODevice` entries;
- NPU node/permission facts;
- OS/kernel;
- disk/temp facts;
- package/runtime facts.

### CompatibilityDecision

Contains recommendation and evidence:

```python
class CompatibilityDecision(BaseModel):
    requested_device: str
    selected_device: str | None
    status: Literal["supported", "unsupported", "unknown", "broken_environment"]
    evidence_level: Literal["official", "tested", "reported", "inferred", "unknown"]
    matched_rule_ids: list[str]
    reasons: list[str]
    fallback_candidates: list[str]
```

### PreparationPlan

The plan is generated before side effects. It should be serializable and printable. A plan must contain enough information to replay the preparation.

### ArtifactManifest

Persisted in each artifact. Schema versioned from day one.

### Diagnostic

A structured diagnostic contains:

- `code` stable machine-readable string;
- severity;
- human title;
- explanation;
- remediation hints;
- raw error/log reference;
- matched signature/rule ID.

## 4. Ports/interfaces

Define Protocols/ABCs to make core behavior testable without Intel hardware.

### HardwareProbe

```python
class HardwareProbe(Protocol):
    def inventory(self) -> HardwareInventory: ...
```

### HubClient

```python
class HubClient(Protocol):
    def inspect(self, model: ModelIdentity) -> ModelMetadata: ...
    def resolve_revision(self, model: ModelIdentity) -> str: ...
    def download(self, source: ResolvedSource, policy: DownloadPolicy) -> Path: ...
```

### Exporter

```python
class Exporter(Protocol):
    def export(self, request: ExportRequest, context: JobContext) -> ExportResult: ...
```

### RuntimeProbe

```python
class RuntimeProbe(Protocol):
    def read_ir(self, path: Path) -> IRDescription: ...
    def compile(self, ir: Path, device: str, properties: dict[str, object]) -> ProbeResult: ...
    def pipeline_smoke_test(self, artifact: Path, device: str, config: RuntimeConfig) -> ProbeResult: ...
```

### Repair

```python
class Repair(Protocol):
    id: str
    def detect(self, artifact: Path, context: RepairContext) -> RepairMatch | None: ...
    def apply(self, match: RepairMatch, artifact: Path) -> RepairResult: ...
    def validate(self, result: RepairResult, context: RepairContext) -> None: ...
```

## 5. Preparation state machine

```text
CREATED
  -> INSPECTED
  -> SOURCE_RESOLVED
  -> PREFLIGHT_OK
  -> DOWNLOADED
  -> EXPORTED
  -> VALIDATING
       -> REPAIR_NEEDED -> REPAIRED -> VALIDATING
  -> PROBED
  -> READY
```

Failure at any stage transitions to `FAILED` and records stage, diagnostic, log path, and resumability. A user cancellation transitions to `CANCELLED` instead. Jobs that make no artifact (`probe`, `bench`) end in `COMPLETED` when they run to their end.

Preparation jobs use a UUID job ID. Persist a small job state JSON so an interrupted conversion can be diagnosed. Do not promise generic resumability of upstream export unless verified; initial implementation may restart export while reusing cached downloads.

## 6. Device/submodel routing

Do not assume one device for every model component.

Represent roles:

```text
language
text_embeddings
vision_embeddings
vision_embeddings_merger
vision_embeddings_pos
resampler
tokenizer
detokenizer
```

Tokenizers are CPU-hosted by default. Numeric VLM submodels may be probed on GPU/NPU. The runtime adapter owns role-to-device configuration where supported by GenAI.

## 7. OpenVINO integration

### Discovery

Use `openvino.Core().available_devices` as authoritative runtime availability. Platform probes may add explanatory details but must not override OpenVINO availability.

Collect properties defensively because properties differ by plugin/version. Failed optional property reads become `unknown`, not fatal errors.

### GenAI runtime

Expose a `GenAIRuntime` abstraction with LLM and VLM implementations selected from artifact metadata.

New code should favor chat history objects/current GenAI APIs rather than coupling the domain to deprecated start/finish-chat lifecycle calls.

### Compile cache

Use a tool-managed compile cache path keyed by:

```text
artifact hash + device + runtime version + relevant compile properties
```

Never reuse cache blindly across incompatible OpenVINO versions.

The key is a one-way digest, so every entry also carries a tool-owned ownership record,
`.yonga-entry.json` (artifact, device, OpenVINO version, properties, last use), written whenever the
runtime hands the directory to OpenVINO. `yonga cache list` and `yonga cache prune` read it to tell
live entries from orphaned and stale-runtime ones (`CLI_SPEC.md` section 12b).

## 8. Optimum export adapter

Initial implementation wraps CLI execution. Example logical request:

```python
ExportRequest(
    source_path=...,
    output_path=...,
    weight_format="int4",
    symmetric=False,
    ratio=1.0,
    group_size=128,
    trust_remote_code=False,
)
```

Adapter translates request to argv. The domain must not know CLI flag spelling.

The adapter captures:

- argv;
- sanitized environment changes;
- stdout/stderr log;
- exit status;
- elapsed duration;
- package versions.

## 9. Artifact store

Artifact identity should include the resolved source revision and profile. Suggested human-readable path:

```text
~/.local/share/yonga/artifacts/
  <namespace>--<model>/
    <source-sha-short>--<profile>/
```

The manifest is the canonical identity; directory names are convenience only.

Incomplete builds live under a job staging directory and are atomically renamed into the final artifact directory after validation.

## 10. Registry architecture

Registry data should be loaded into typed schema objects, normalized, validated, then indexed. Never have application logic grep YAML directly.

Support:

- built-in package registry;
- signed remote registry bundles, layered between the built-in rules and user overrides (`yonga registry update`, `REGISTRY_SPEC.md` section 12.1);
- user override directory;
- debug command to explain rule matching.

Rule matching is explained by:

```bash
yonga registry explain MODEL --device NPU
```

## 11. Diagnostics pipeline

Raw exceptions flow through classifiers in this order:

1. strongly typed adapter errors;
2. exact known signatures from registry;
3. regex signatures from registry;
4. generic category inference;
5. unknown.

A classifier returns a diagnostic but retains the raw exception/log.

Example:

```text
Raw:
OneShotBufferizeVPU2VPUIP failed : Only supports FP16.

Diagnostic:
code: npu.compiler.unsupported_graph.qwen35
status: unsupported
message: Current NPU compiler cannot compile this Qwen3.5 graph.
fallback: GPU
```

## 12. UI architecture

Use Rich for panels, tables, status indicators and Markdown. Use `prompt_toolkit` for line editing/history in chat.

Application services should emit structured progress events, not Rich calls. UI subscribes/renders:

```python
ProgressEvent(kind="download", current=..., total=...)
ProgressEvent(kind="export_stage", message=...)
ProgressEvent(kind="probe", device="NPU", status="failed", diagnostic=...)
```

This separation enables a future GUI/web frontend.

## 13. Configuration architecture

Use `pydantic-settings` or a small equivalent. Paths derive from XDG environment variables.

Configuration must be serializable for diagnostics, with secrets removed.

## 14. Concurrency

MVP can run one preparation job per process. Downloads may use Hugging Face's native concurrency. Avoid adding an async framework without need.

Chat generation remains synchronous initially. Ctrl+C should cancel the current generation when upstream permits and return to prompt rather than terminating the entire process where practical.

## 15. Future plugin points

Some of the extension points this section first listed have been filled: the remote registry updater (`yonga registry update`), the model publication adapter (`yonga publish`) and an OpenVINO GGUF import for architectures and tensor types a registry `gguf` rule verifies (`REGISTRY_SPEC.md` section 8a).

Keep interfaces open for:

- Windows hardware probe;
- llama.cpp backend;
- OpenVINO Model Server adapter;
- GUI frontend;
- additional accelerator vendors only if the project intentionally expands scope later.
