# MODEL_PIPELINE.md

## 1. Pipeline overview

```text
input model reference
      |
      v
inspect metadata
      |
      v
resolve source + revision
      |
      v
resolve hardware/compatibility/profile
      |
      v
preflight (disk/RAM/temp/auth/trust)
      |
      v
download/cache source
      |
      v
export/quantize to staging artifact
      |
      v
validate IR and tokenizers
      |
      +---- known issue? ----> narrowly repair -> revalidate
      |
      v
probe target device
      |
      +---- unsupported + auto? --> candidate fallback probe
      |
      v
smoke generation
      |
      v
write final manifest + atomic finalize
```

## 2. Inspect phase

Use Hugging Face Hub metadata APIs before large downloads. Request file metadata when size estimates are needed.

Fetch only small configuration/tokenizer metadata required to identify architecture when possible.

Identify formats:

- OpenVINO IR: `openvino_*.xml` + `.bin`;
- safetensors: `*.safetensors` and index;
- PyTorch: `*.bin` / indexes;
- GGUF: `*.gguf`;
- tokenizer/config assets.

Do not equate `GGUF exists` with `OpenVINO GenAI can consume GGUF architecture`.

## 3. Source resolution

Return a `ResolvedSource` object and explain it.

Case A — source contains safetensors: use source directly.

Case B — source is already a compatible OpenVINO IR artifact: download/use directly, validate locally, and create an yonga wrapper manifest without re-exporting.

Case C — source is GGUF-only:

1. check the registry for a declared derivative source; the trust policy applies, so a third-party derivative under `suggest` requires confirmation and a `never_auto` one is only informational;
2. if no derivative is usable, import the GGUF file directly, but only when a registry `gguf` rule (`REGISTRY_SPEC.md` section 8a) covers the installed OpenVINO GenAI and the chosen file's own header shows an architecture and tensor types that rule lists as verified;
3. otherwise stop with `source_format_unsupported` and suggest options.

Never scrape model names and auto-pick search results.

## 4. Revision pinning

Resolve branch/tag to immutable commit SHA before preparation. Record both requested revision and resolved SHA.

All downloads for the job use that SHA.

## 5. Preflight

Compute:

- expected download size;
- source weight bytes;
- target filesystem free bytes;
- managed temp filesystem free bytes;
- RAM/swap;
- device presence;
- package compatibility;
- authentication/gated state;
- trust-remote-code requirement if known.

Disk policy:

```python
estimated_temp = max(20 * GiB, source_weight_bytes * 2.5)
```

Require `estimated_temp + safety_margin` free on managed temp filesystem. Safety margin default 5 GiB.

If `/tmp` is tmpfs, report it but use managed disk temp automatically.

## 6. Download

Use `snapshot_download` with pinned `revision`.

Avoid duplicate source copies. Export may point at the cached snapshot.

For known file selection, use `allow_patterns`/`ignore_patterns` conservatively. Do not accidentally omit custom model/config/tokenizer files needed by Transformers.

yonga passes `ignore_patterns` only (never an allow-list, except the one GGUF file of an import), built by `download_ignore_patterns` in `yonga.domain.plans`: other runtimes' weights (`*.onnx`, `onnx/`, `*.msgpack`, `*.h5`, `*.tflite`, Core ML), a native checkpoint kept beside the Transformers one (`original/`, `consolidated*.pth`), training leftovers (`runs/`, `*tfevents*`, `training_args.bin`, optimizer and scheduler states) and stray `*.gguf`. Once the format being converted is complete, the other weight formats' files are excluded by their exact paths, so a repository publishing both `.safetensors` and `.bin` is fetched once and a `.bin`-only one keeps its weights. With remote code enabled nothing is excluded. The plan's expected download is computed with the same patterns.

Offline mode uses `local_files_only=True` and fails clearly if missing. It never contacts the Hub,
for metadata either: inspection and revision resolution read the cached snapshot (file list and
sizes, `config.json`, the model card's front matter, safetensors and GGUF headers) and fail with
`hub.offline.missing` when the revision or its weights are not cached.

## 7. Export

Create staging directory:

```text
<data>/artifacts/.staging/<job-id>/model/
```

Create temp directory:

```text
<cache>/tmp/<job-id>/
```

Environment:

```text
TMPDIR=<managed temp>
TMP=<managed temp>
TEMP=<managed temp>
```

Build Optimum argv from profile.

`optimum-cli` in the examples stands for the launcher. In a managed export environment yonga runs `optimum.commands.optimum_cli` as a module of `<env python>`; in its own environment it runs it with `<python>` when that module is importable and falls back to the `optimum-cli` executable on `PATH` otherwise, through the interpreter its `#!` line names when that is a plain path. The module is started as `<python> -P -c "<bootstrap>" ARGS`, where the bootstrap makes `import openvino_telemetry` fail and then runs the module with `runpy.run_module(..., run_name="__main__", alter_sys=True)`, which is what `-m` does: the same `sys.argv`, the same exit status, and no usage report from OpenVINO (`src/yonga/privacy.py`). `-P` keeps the current directory off `sys.path`, so a module of the same name there cannot stand in for a real one.

Example GPU profile:

```text
optimum-cli export openvino
  --model <snapshot>
  --weight-format int4
  --ratio 1.0
  --group-size 128
  [--trust-remote-code only if explicitly enabled]
  <staging output>
```

Example NPU profile:

```text
--weight-format int4 --sym --ratio 1.0 --group-size -1
```

Record duration and command.

## 8. Validation inventory

After export, list all files and calculate hashes only after repairs complete (optionally pre-repair hashes in repair record).

Expected VLM roles may include:

```text
openvino_language_model.xml
openvino_text_embeddings_model.xml
openvino_vision_embeddings_model.xml
openvino_vision_embeddings_merger_model.xml
openvino_vision_embeddings_pos_model.xml
openvino_tokenizer.xml
openvino_detokenizer.xml
```

Do not require every role universally. Determine requirements from pipeline/config.

## 9. Tokenizer validation

Load OpenVINO Tokenizers extension. Verify package compatibility first.

Tokenizer/detokenizer are read and compiled on CPU.

If `read_model` fails with unsupported string element type, run repair detection before failing.

The known `STRiNG` repair must only modify `precision="STRiNG"`, preserving `element_type="string"`.

## 10. Numeric submodel validation

Read each numeric IR and collect input/output types/shapes.

Probe requested device. For a VLM, store result per role:

```json
{
  "language": {"GPU": "ok", "NPU": "failed"},
  "text_embeddings": {"GPU": "ok"},
  "vision_embeddings": {"GPU": "ok"}
}
```

If OpenVINO uses subprocess/forkserver internally, executable scripts and tests must guard entrypoints with:

```python
if __name__ == "__main__":
    main()
```

## 11. Failure classification

Feed exception + architecture + device + versions to classifier.

For the Qwen3.5 NPU signature:

```text
OneShotBufferizeVPU2VPUIP failed : Only supports FP16.
```

classification is `compiler_unsupported_graph`, not `driver_missing`.

In auto mode, this permits GPU fallback. In explicit NPU mode, fail with suggestion.

## 12. Pipeline smoke test

After successful compile probe, instantiate `LLMPipeline` or `VLMPipeline` depending on metadata.

Use a tiny deterministic smoke prompt and low token count (e.g. 16–32) solely to confirm generation. This is not a quality test.

Use appropriate attention/backend properties from profile. GPU profile may select SDPA when required by compatibility rules.

## 13. Chat-template handling

Store chat behavior separately from model weights.

A runtime profile may specify:

```yaml
chat:
  supports_thinking_flag: true
  thinking_template_variable: enable_thinking
  default_thinking: false
  reasoning_wrappers:
    - ["<think>", "</think>"]
```

Do not permanently rewrite the source model's chat template just to disable reasoning. Apply a runtime override to the managed artifact/runtime object where possible.

## 14. Finalization

On success:

1. write `.yonga/manifest.json`;
2. write `.yonga/validation.json`;
3. write `.yonga/source.json` if helpful;
4. fsync important metadata when practical;
5. atomically rename staging artifact to final path;
6. update local artifact index.

On failure:

- source HF cache remains intact;
- final artifact path is not created;
- job log/state remains available;
- staging cleanup follows retention policy.

## 15. Re-preparation

If the exact source revision already has a ready artifact under the chosen profile id, `prepare` returns it without re-exporting unless `--force`. Reuse is keyed by revision and profile id only, which is why a profile id never changes its quantization: a different quantization is a new id, so a changed default (a new size band, a re-prioritised profile) produces a new artifact next to the old one instead of silently reusing it. The plan says so when the revision is already prepared for the device under another profile, and names the `--profile` that would reuse it.

If OpenVINO runtime changes, existing IR may remain usable but compile cache must be invalidated/re-keyed. Validation state should record which runtime version last validated the artifact.
