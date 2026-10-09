# REFERENCE_NOTES.md

This file records upstream references used while drafting the initial specification. These are not substitutes for runtime capability probing; upstream behavior can change.

## OpenVINO 2026.4

- Documentation: https://docs.openvino.ai/2026/
- Release notes: https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html
- GenAI inference: https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai.html
- NPU GenAI guidance: https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html

Relevant initial facts:

- NPU LLM export guidance uses symmetric INT4/NF4, ratio 1.0 and group size `-1` or `128`.
- Group size 128 is recommended for smaller models around 4–5B; channel-wise can be appropriate for larger models.
- NPU LLM prompt/response static context settings include `MAX_PROMPT_LEN` and `MIN_RESPONSE_LEN`.
- `start_chat()` / `finish_chat()` are deprecated in OpenVINO 2026.4 release notes; new application code should isolate chat lifecycle so current history APIs can be used.

## OpenVINO GenAI supported models

- Supported model notes: https://github.com/openvinotoolkit/openvino.genai/blob/master/site/docs/supported-models/index.mdx

Relevant initial fact:

- Qwen3.5 export requires `transformers==5.2` in current guidance.

Checked on the reference host, 2026-09-22 (one host, one stack; not a general law):

- With `optimum-intel` 2.2.0 and `transformers` 5.5.4 the OpenVINO export fails in
  `optimum/exporters/openvino/model_patcher.py` with `ImportError: cannot import name
  'Qwen3_5DynamicCache' from 'transformers.models.qwen3_5.modeling_qwen3_5'`. With
  `transformers` 5.2.0 that symbol imports and the export proceeds.
- `transformers` 5.5.4 still loads `Qwen3_5Config` and still reports `model_type: qwen3_5`, so a
  successful config load says nothing about whether the export will work.

Checked on the reference host, 2026-09-24 (same caveat):

- A raw `Core.compile_model` of a stateful LLM on the NPU fails with
  `to_shape was called on a dynamic shape` for every model tried, including Qwen3-0.6B, which then
  loads and generates through `LLMPipeline(..., "NPU", MAX_PROMPT_LEN=1024, MIN_RESPONSE_LEN=128)`.
  A raw compile is therefore not evidence about NPU support for an LLM; the pipeline is.
- Through `VLMPipeline` on the NPU, Qwen3.5 (INT4 symmetric, group -1) is rejected with
  `OneShotBufferizeVPU2VPUIP failed : Only supports FP16.` — the signature this specification pins.
  It is written by native code to standard output (yonga's `--json` mode redirects descriptor 1 to
  descriptor 2, which is why it can surface on stderr there) and is absent from the Python
  exception, which reads only
  `[NPU_VCL] Compiler returned msg: Compilation failed`.
- `MAX_PROMPT_LEN` / `MIN_RESPONSE_LEN` must be integers; the string `"1024"` raises
  `Failed to extract MAX_PROMPT_LEN. Type mismatch: expected types: int or int64_t`.
- The tiny random Llama test model is rejected by the NPU pipeline
  (`Model references undeclared parameters`), so NPU generation needs a small real model.

## OpenVINO GenAI API

- VLMPipeline: https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.VLMPipeline.html
- LLMPipeline: https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.LLMPipeline.html

## Optimum Intel

- Export: https://huggingface.co/docs/optimum-intel/openvino/export
- Optimization: https://huggingface.co/docs/optimum-intel/openvino/optimization

Relevant facts:

- OpenVINO export supports `int4`, `--sym`, `--ratio`, and `--group-size`.
- `--group-size -1` means per-column quantization.
- Default integer weight-only behavior without `--sym` is asymmetric.

## Official OpenVINO Qwen3.5-9B artifact

- https://huggingface.co/OpenVINO/Qwen3.5-9B-int4-ov

Published conversion metadata at drafting time:

```text
mode: INT4_ASYM
group_size: 128
ratio: 1.0
```

This is evidence for a GPU/CPU-oriented comparison profile, not proof that the same profile is optimal for every derivative/model/device.

## Hugging Face Hub

- Download guide: https://huggingface.co/docs/huggingface_hub/guides/download
- HfApi: https://huggingface.co/docs/huggingface_hub/package_reference/hf_api

Use `snapshot_download()` and immutable revisions instead of custom HTTP download logic.

## OpenVINO package compatibility

- GenAI dependencies: https://docs.openvino.ai/2026/get-started/install-openvino/configurations/genai-dependencies.html

OpenVINO GenAI, OpenVINO Tokenizers and OpenVINO package versions must be kept ABI-compatible. The CLI should diagnose mismatches instead of automatically mutating the user's environment.

## Serving

- OpenVINO Model Server generative endpoints: https://docs.openvino.ai/2026/model-server/ovms_docs_clients_genai.html

OVMS provides OpenAI-compatible generative endpoints. It was considered for `yonga serve`, which shipped instead as a FastAPI layer over the chat runtime (the optional `serve` extra; see ROADMAP.md item 2.1). OVMS remains a possible alternative backend.
