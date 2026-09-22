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

OVMS provides OpenAI-compatible generative endpoints and is a candidate for the future `yonga serve` implementation.
