# ROADMAP.md — what to do next

Written at the end of the MVP build, 2026-09-23. Ordered by what would most change the tool's
usefulness or trustworthiness, not by what is easiest. Sizes are rough: **S** a session, **M** a
few days, **L** a substantial piece of work.

The MVP works end to end on one machine: ten commands (`registry` carries three subcommands of its
own — `validate`, `explain`, `paths`), the full Qwen3.5 reference case (GGUF-only source → declared derivative → 17.5 GiB download → INT4 conversion → seven VLM submodels
validated → tokenizer repaired → GPU probed → artifact with a complete manifest), 1779 tests, ruff
and mypy clean. What follows is what it does *not* yet have.

---

## Phase 0 — close the honesty gaps (do this first)

These are places where the project currently looks better covered than it is. None is a bug; each
is a claim without evidence behind it.

**0.1 — Hardware test coverage actually exists. [M]**
`intel_gpu`, `intel_npu`, `slow` and `large_model` select **zero tests**. Every GPU and NPU result
in this repository came from a command someone typed. A reader seeing those markers in
`pyproject.toml` will assume otherwise. Write real marked tests: GPU compile of a small artifact,
NPU compile producing a *classified* failure, tokenizer-on-CPU-while-numeric-on-GPU routing, and
the compile-cache reuse path. They will not run in CI — that is fine, `docs/TESTING.md` §6 specifies
exactly these hardware acceptance tests and §11 always expected them on a self-hosted runner — but
they must exist and pass on a real Core Ultra host.

**0.2 — Reconcile the NPU acceptance signature. [S]**
`AGENT_PROMPT.md` §18 pins `OneShotBufferizeVPU2VPUIP failed : Only supports FP16.` as *the*
Qwen3.5 NPU signature. The registry rule exists and its unit test passes against a fixture string,
but that error **did not reproduce** on this stack — the real artifact fails with
`to_shape was called on a dynamic shape` and classifies as `npu.compiler.dynamic_shape`. Either the
original observation came from a different code path (likely the GenAI NPU pipeline with static
reshaping, not a direct `compile_model`) or the stack has moved. Find out which, and record the
answer in `REFERENCE_NOTES.md` rather than leaving a rule whose only evidence is a fixture.

**0.3 — Prove the NPU path with a model that should work. [M]**
Everything we know about NPU behaviour comes from a model the registry already rules out. Prepare a
small LLM the NPU *is* expected to handle (the OpenVINO GenAI NPU guidance suggests ~4-5B with
`group_size 128`), through `LLMPipeline` with `MAX_PROMPT_LEN`/`MIN_RESPONSE_LEN` set. Until that
succeeds, "NPU-first" is a design stance rather than a demonstrated capability.

**0.4 — Squash the 7 MB test fixture out of git history before the first push. [S]**
Commit `65f4e4a` added `tests/fixtures/artifacts/tinyllm/` — twelve blobs, 7,406,017 bytes in total,
the largest being a 3.6 MB `tokenizer.json`. `ff2dd19` removed it from the tree in favour of
generating the tiny artifact, but every blob is still reachable in history and every clone will pay
for it.

---

## Phase 1 — make it useful to a second person

**1.1 — `yonga search`. [M]**
`CLI_SPEC.md` §5. Hub search enriched with compatibility, so the loop starts inside the tool
instead of on the website. The hard part is honesty: do not claim compatibility for a repository
whose architecture you have not established, which for GGUF-only repos means reading the header.
The registry and inspect paths already do this work.

**1.2 — Managed export environments. [L]**
The single sharpest edge today. Qwen3.5 needs `transformers==5.2.*`; something else will need a
different pin; they cannot coexist. yonga currently diagnoses the mismatch and stops, which is
correct but leaves the user to hand-manage a venv. `AGENT_PROMPT.md` §13.2 anticipates this: "a
future managed environment feature can solve this properly." Build it — a per-architecture
throwaway environment created on demand, used for the export subprocess only, cached and
garbage-collected. This is what turns "diagnoses the problem" into "handles it".

**1.3 — Real resume for interrupted preparation. [M]**
Downloads already resume through the Hub cache; export restarts from scratch, which on a 17.5 GiB
model is expensive. Investigate what optimum-intel can actually resume, and only promise what it
delivers — `ARCHITECTURE.md` §5 is explicit about not claiming generic resumability.

**1.4 — Bug-report bundles. [S]**
`IMPLEMENTATION_PLAN.md` milestone 11. One command that collects the job log, the manifest, the
hardware inventory and the redacted argv into an attachable archive. Every repair and classifier in
the registry is a candidate upstream bug; this makes reporting them cheap.

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
| job logs | Four pre-fix logs on the reference host still carry a 71-variable environment dump. No token-shaped string survives in them, but they should be deleted. |
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
