# ROADMAP.md — what to do next

Written at the end of the MVP build, 2026-09-23. Ordered by what would most change the tool's
usefulness or trustworthiness, not by what is easiest. Sizes are rough: **S** a session, **M** a
few days, **L** a substantial piece of work.

The MVP works end to end on one machine: ten commands (`registry` carries three subcommands of its
own — `validate`, `explain`, `paths`), the full Qwen3.5 reference case (GGUF-only source → declared derivative → 17.5 GiB download → INT4 conversion → seven VLM submodels
validated → tokenizer repaired → GPU probed → artifact with a complete manifest), and — since
2026-09-24 — NPU preparation, chat and benchmarking of a small LLM. 1824 default tests plus eight
hardware tests, ruff and mypy clean. What follows is what it does *not* yet have.

---

## Phase 0 — close the honesty gaps

These were places where the project looked better covered than it was. Items 0.1–0.3 closed on
2026-09-24; working them turned up a defect that made NPU preparation impossible, described under
0.3.

**0.1 — Hardware test coverage actually exists. [done]**
`intel_gpu` and `intel_npu` now select real tests in `tests/hardware/` — three on the GPU
(generation with metrics, compile-cache reuse, the probe service with the tokenizer routed to CPU)
and five on the NPU (node access, raw-compile deferral, a typed pipeline failure, generation with
manifest-shaped properties, and the probe service passing an NPU LLM). All eight pass on the
reference host. The NPU generation tests need `python tools/make_test_artifact.py --npu` once; they
skip with that instruction otherwise. `slow` and `large_model` still select nothing.

**0.2 — Reconcile the NPU acceptance signature. [done]**
The signature `AGENT_PROMPT.md` §18 pins does reproduce — on the pipeline path, not on a raw
compile. A raw `Core.compile_model` of Qwen3.5 on the NPU fails first on dynamic shapes; through
`VLMPipeline`, which reshapes to a static budget, the compiler reaches the real rejection:
`OneShotBufferizeVPU2VPUIP failed : Only supports FP16.` Two further things were needed before yonga
could *classify* it: the NPU compiler writes that line to the native file descriptors, never into
the Python exception, so yonga now captures them around every native compile; and a pipeline
failure used to arrive as a generic `runtime.genai.load_failed` that bypassed the registry and
exited 99 ("yonga hit a bug"). It now classifies as `npu.compiler.unsupported_graph.qwen35`, exit 51.

**0.3 — Prove the NPU path with a model that should work. [done]**
`yonga prepare Qwen/Qwen3-0.6B --device npu --profile npu-int4-sym-g128` produces a ready NPU
artifact; `chat` answers on the NPU and `bench` measures 67.3 tok/s, TTFT 165 ms over 256 tokens.
Getting there exposed the real defect: the probe compiled each numeric submodel *raw* and treated a
failure as the verdict, but a raw compile rejects every dynamically shaped LLM on the NPU, so no
NPU artifact could ever be finalized. Registry issue rules can now declare `defer_to_pipeline`; the
dynamic-shape rule does, and the GenAI pipeline smoke test decides. A deferred failure never passes
a device by itself — without pipeline evidence it stays failed. A second bug sat behind it: GenAI
needs `MAX_PROMPT_LEN` as an integer, and every property yonga records is a string.

**0.4 — Squash the 7 MB test fixture out of git history before the first push. [done]**
An early commit added `tests/fixtures/artifacts/tinyllm/` — twelve blobs, 7,406,017 bytes — and a
later one removed it in favour of generating the artifact, but the blobs stayed reachable in
history. On 2026-09-24 the directory was filtered out of every commit: all thirteen commits and
their messages are kept, the final tree is byte-identical, and `.git` went from 7.0 MB to 964 KB.
Every commit from the one that added the fixture onward has a new SHA.

**0.5 — NPU probes of large models are slow now, honestly. [partly done]**
Because deferral hands the verdict to a real pipeline build, `yonga probe` of a large model on the
NPU no longer fails instantly on a raw compile — it performs the NPU compilation the answer actually
depends on. Measured on the reference host: the GPU-quantized Qwen3.5-9B artifact was still
compiling for the NPU after 15 minutes at ~9 GiB RSS. The instant answer was false, so this is the
right trade. The probe now says before the wait when it is about to compile more than 2 GiB of
weights. Still open: a `--quick` mode that stops at the raw compile and labels the result
inconclusive, and a bounded timeout for pipeline builds.

---

## Phase 1 — make it useful to a second person

**1.1 — `yonga search`. [done]**
`CLI_SPEC.md` §5. One `list_models` request returns, per hit, the Hub's config summary and the GGUF
header, so every result goes through the same registry identification as `inspect` — alias rung
included — with no per-repository download. The honesty problem turned out to be preparability,
not identity: an MLX or AWQ repository reports a supported architecture it cannot be converted
from. Search therefore shows device knowledge per architecture beside a format hint from Hub tags,
and defers to `inspect` for whether a repository can be prepared.

**1.2 — Managed export environments. [done]**
When the installed packages cannot meet an exporter requirement, `prepare` builds an isolated venv
under `~/.cache/yonga/envs/export-<key>/` and runs only the export subprocess in it. The spec mirrors
the installed packages and changes only the failing pin; the key digests every pin, so the same
requirements always reuse the same environment. It is built at preflight, before the download,
verified by importing the exporter, and marked complete last, so a half-built one is never used.
`yonga env list|remove`, `doctor` storage report, `--no-managed-env` to refuse instead. Verified on
the reference host: the Qwen3.5-9B CPU export ran in `export-d1aedd10dc08` (torch 2.14.0+cpu,
transformers 5.2.0, 1.44 GiB) and the manifest names it. Building it found one trap worth knowing:
`optimum-intel` pulls `torchvision`, and PyPI's CUDA build of it cannot register its operators
against a CPU torch, so both now come from PyTorch's CPU index.

**1.3 — Reuse of a completed export after a failed job. [done]**
What optimum-intel can resume: nothing. `main_export` runs `export_from_model` and then weight
compression in one process, marks no stage as done, and a run killed part-way leaves files that look
like a finished run's. So an interrupted *export* still restarts, and yonga says so. What it can do
is keep an export that *finished* when a later stage fails — a probe failure, a validation defect,
Ctrl-C during repair or the smoke test. The exporter's success writes `.yonga/export-complete.json`
into staging: a fingerprint of every input (source revision, profile, task, exporter packages or
managed environment, yonga version) plus the size, mtime and, for files up to 16 MiB, the digest of
each file. An identical `prepare` finds it at plan time ("Export: reused from failed job …"), claims
it by rename, restores repaired files from the repair's own backups, rejects anything that changed,
and skips the download and the conversion. `--fresh-export` declines. On the reference host an
interrupted Qwen3-0.6B CPU preparation retried in 9.4 s instead of 34 s. Building it found that
Ctrl-C never recorded a terminal job state, so a cancelled job looked live for 24 hours; `open_job`
now records `job.cancelled`.

**1.4 — Bug-report bundles. [done]**
`IMPLEMENTATION_PLAN.md` milestone 11. `yonga logs JOB_ID --bundle` writes a `.tar.gz` with the job
log and state, the artifact manifest and validation record, and a `doctor --json` host report, plus
a README listing the contents. Every file is redacted again and the home directory, hostname and
login name are anonymised, because a bundle is meant to be attached to public issues. Building it
against the reference host's oldest logs found that a hostname embedding the login name
(`<user>-x1carbon`) leaked both, so the hostname is anonymised first.

---

## Phase 2 — serving and publication

**2.1 — `yonga serve`. [L]**
`SPEC.md` §22, `CLI_SPEC.md` §14. `GET /v1/models`, `POST /v1/chat/completions`, optional SSE, a
health endpoint, bound to localhost by default. The runtime abstraction already exists and chat
already goes through it, so this should be an adapter over the same `ChatRuntime` rather than a
second preparation stack. Decide early between a small FastAPI layer and integrating OpenVINO Model
Server; the second is less custom code but a much heavier dependency.

**2.2 — `yonga publish`. [M]**
`SPEC.md` §21. The manifest already carries everything a generated model card needs — source
lineage and revisions, quantization, repairs applied, device probe evidence, software versions.
The discipline in that section's requirement list matters more than the plumbing: never describe a
quantized conversion as a fine-tune, never label an artifact for a device whose probe failed, and
always disclose repairs.

**2.3 — Remote registry. [M]**
`REGISTRY_SPEC.md` §12. Compatibility knowledge moves faster than releases. This must be signed and
verified before automatic use, and must never be able to deliver executable repair logic — the
YAML names a built-in implementation and that invariant is load-bearing.

---

## Phase 3 — breadth

- **Windows support. [L]** The adapter boundary was built for it; `system_linux.py` has a sibling's
  worth of work in it and every path assumption needs auditing.
- **GGUF execution. [M]** Today GGUF is detected and explicitly refused as a conversion source.
  If OpenVINO GenAI gains architecture support, the resolver already has the branch for it.
- **More architectures in the registry. [ongoing]** Two architecture rules ship today. Each new one
  is data plus a test, not code — that was the point of the design.
- **A GUI or web frontend. [L]** Application services already emit structured `ProgressEvent`s and
  never import Rich, so a second frontend does not require touching the core.

---

## Known defects still open

Small, real, and worth clearing before the next feature.

| Where | What |
|---|---|
| `docs/PROFILE_COMPARISON.md` | Measures size and speed only. Output quality was never evaluated, and that gap is the interesting one — a symmetric channel-wise INT4 artifact and an asymmetric group-128 one are not expected to be equally accurate. |

---

## A note on maintenance

Two rules kept this codebase honest and are worth keeping:

1. **Compatibility knowledge is data.** Adding an architecture, a failure signature, a repair or a
   chat profile should mean editing YAML and adding a test — never a branch in application logic.
   Every time that rule was bent during the build it produced a defect.
2. **Claims need evidence.** The registry's `evidence` levels, the manifest's probe records and the
   provenance line in `inspect` all exist so the tool can say *how well it knows* something. An
   adversarial review of this project found three defects that were all the same mistake in
   different clothes: presenting an inference as a fact.
