# ROADMAP.md — status and what to do next

Started at the end of the MVP build, 2026-09-23, and updated as items closed. Ordered by what would
most change the tool's usefulness or trustworthiness, not by what is easiest. Sizes are rough: **S**
a session, **M** a few days, **L** a substantial piece of work.

## Where things stand

Phases 0, 1 and 2 are done, within the limits each item records (2.1, for example, lists what
`serve` does not do yet). What remains is in Phase 3: Windows support, a GUI or web frontend, and
more architectures in the registry. Each item keeps the note written when it closed, with what
was measured and what building it found; the numbers in those notes are as of that date. For the
current command set run `yonga --help`, and for the current test count `pytest --collect-only -q`.

At the end of the MVP build the tool worked end to end on one machine: the full Qwen3.5 reference
case (GGUF-only source → declared derivative → 17.5 GiB download → INT4 conversion → seven VLM
submodels validated → tokenizer repaired → GPU probed → artifact with a complete manifest) and, from
2026-09-24, NPU preparation, chat and benchmarking of a small LLM. The items below were the plan
from that point; each is marked **[done]**, **[partly done]** or **[ongoing]**, or left open with
a size.

---

## Phase 0 — close the honesty gaps

These were places where the project looked better covered than it was. Items 0.1–0.3 closed on
2026-09-24; working them turned up a defect that made NPU preparation impossible, described under
0.3.

**0.1 — Hardware test coverage actually exists. [done]**
`intel_gpu` and `intel_npu` now select real tests in `tests/hardware/`. When this item closed on
2026-09-24 there were three on the GPU (generation with metrics, compile-cache reuse, the probe
service with the tokenizer routed to CPU) and five on the NPU (node access, raw-compile deferral, a
typed pipeline failure, generation with manifest-shaped properties, and the probe service passing
an NPU LLM), all passing on the reference host. Streaming, `serve` and GPU-against-CPU logits
followed, and `slow` and `large_model` now select the GGUF import and VLM chat tests;
`TESTING.md` §6, "What the hardware suites run", lists the current set. The NPU generation tests
need `python tools/make_test_artifact.py --npu` once; they skip with that instruction otherwise.

**0.2 — Reconcile the NPU acceptance signature. [done]**
The signature `SPEC.md` §20 records does reproduce — on the pipeline path, not on a raw
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

**0.5 — NPU probes of large models are slow now, honestly. [done]**
Because deferral hands the verdict to a real pipeline build, `yonga probe` of a large model on the
NPU no longer fails instantly on a raw compile — it performs the NPU compilation the answer actually
depends on. Measured on the reference host: the GPU-quantized Qwen3.5-9B artifact was still
compiling for the NPU after 15 minutes at ~9 GiB RSS. The instant answer was false, so this is the
right trade. The probe says before the wait when it is about to compile more than 2 GiB of weights,
and names the two ways out. `--quick` stops at the raw compile and reports a deferred device with
the new status `inconclusive`. `--timeout SECONDS` builds the pipeline in a child process
(`python -P -m yonga.adapters.smoke_child`, through the supervised subprocess runner) and stops it
at the limit, which is `inconclusive` too — a native compile cannot be interrupted in-process. An
inconclusive device never counts as working or as failed: exit 2 when nothing else decided, and a
model card would never list it as working. `prepare` still waits for its pipeline build. On the
reference host a quick NPU probe of the 0.6B test artifact answers in about a second, and
`--timeout 1` stops its pipeline build and reports inconclusive; both are hardware tests.

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
a README listing the contents. Every file is redacted again and the home directory, hostname,
login name and the Hugging Face account name the `doctor` report carries are anonymised (to `~`,
`<host>`, `<user>` and `<hf-user>`), because a bundle is meant to be attached to public issues.
Building it against the reference host's oldest logs found that a hostname embedding the login name
(`<user>-x1carbon`) leaked both, so the hostname is anonymised first.

---

## Phase 2 — serving and publication

**2.1 — `yonga serve`. [done]**
A FastAPI layer (optional `serve` extra) over the same `ChatSession` `chat` uses, through a new
stateless `ChatSession.respond`: `/health`, `/v1/models`, `/v1/chat/completions` with SSE, bound to
loopback, optional bearer key from `YONGA_API_KEY`. Streaming goes through a `StreamingSanitizer`
that only releases text no later chunk can turn into reasoning. Building it found that OpenVINO
GenAI 2026.4's `VLMPipeline.generate` *does* accept a `ChatHistory`, contrary to the adapter's note;
the VLM path is now stateless where the build allows it and falls back to the hidden
`start_chat()` session otherwise. Verified on the reference host with the Qwen3.5-9B GPU artifact
and the official `openai` Python client: plain, streamed and multi-turn completions, 404/400
mapping, and a disconnected 1500-token stream releasing the GPU (the next request answered in
0.43 s). Not done: images, tool calls, stop sequences, concurrent generation.

**2.2 — `yonga publish`. [done]**
One-commit upload of the manifest's files, the source's license files, a generated model card and
an anonymised `yonga-manifest.json`. Every §21 requirement is a refusal with its own code, checked
before upload: explicit destination, write access to the namespace, `--update` for an existing
repo, both source licenses read fresh from the Hub (derivative-forbidding ones refuse, unrecognised
ones need `--license-reviewed`), a ready artifact, and file digests that still match. The card
calls the artifact a quantized conversion, not a fine-tune. It shows a device as working only on a
passing probe, lists failed and untested devices explicitly, flags a profile whose target device
was not verified, and discloses every repair with digests. Verified on the reference host by dry
runs of the Qwen3.5-9B artifacts: the card and manifest contain no home path or login name, the
Apache `LICENSE` is included, and the `npu-int4-sym-cw` artifact is described as verified on GPU
only. An unauthenticated real upload is refused at planning (exit 21). **Nothing has been uploaded
to the Hub yet**; the upload path is covered by a fake publisher and adapter tests.

**2.3 — Remote registry. [done]**
`yonga registry update | status | reset`, `REGISTRY_SPEC.md` §12.1. A bundle is one JSON document of
registry YAML with a detached Ed25519 signature (`cryptography` is now a core dependency; the check
is never optional). It is verified against keys the user placed in `registry-keys/` (yonga ships
none), refused when expired or older than the installed one, and re-verified on every load, so an
edit on disk withdraws it. Signed or not, it may only contain relative YAML paths, and every repair
rule must name an implementation compiled into yonga. It is layered between built-in rules and user
overrides. `tools/registry_bundle.py` makes keys and bundles. Verified end to end in scratch
directories: untrusted refused, trusted installed and used, a one-byte edit on disk reported as
`bad_signature` and dropped. There is no hosted registry or project signing key yet; that is
infrastructure, not code.

---

## Phase 3 — breadth

- **Windows support. [L]** The adapter boundary was built for it; `system_linux.py` has a sibling's
  worth of work in it and every path assumption needs auditing.
- **GGUF execution. [done]** OpenVINO GenAI 2026.4 turned out to carry a GGUF reader, contrary
  to what the code assumed. `prepare` now imports a GGUF file directly when a registry `gguf` rule
  verifies both its `general.architecture` and every tensor type in it. Both are read from the
  file's own header over HTTP range requests (about 10 MB, even for a 9 GB file) before anything is
  downloaded. The import runs in a child process from a symlink in staging, so the Hub cache is
  never written, although GenAI saves its IR next to the path it is given by default. The result
  is an ordinary artifact: validated, repaired (it carries the same `STRiNG` tokenizer defect),
  probed, and usable by `chat`, `bench`, `serve`, `eval` and `publish`. Verified on the reference
  host: llama, qwen2 and qwen3 with F32, F16, Q4_0, Q4_K, Q6_K and Q8_0 tensors. It fails, as
  predicted by the header, on every file holding Q5_0, Q5_1, Q5_K or IQ4_NL tensors (which is most
  `Q4_K_M` Qwen files), and on the gemma3 and qwen35 architectures. `eval` found a divergence floor
  of about 0.03 nats against an optimum FP16 export even from an F16 GGUF
  (`PROFILE_COMPARISON.md` §9), whose cause was not isolated. The NPU cannot reshape a GGUF-built
  graph (`npu.pipeline.static_reshape_failed`).
- **More architectures in the registry. [ongoing]** Ten families have device claims measured on
  the reference host with one small model each, prepared and generating on GPU and NPU and
  benchmarked on the CPU: qwen3, qwen2, llama, gemma3-text, gemma2, olmo2, granite, smollm3, lfm2
  and qwen2.5-vl (whose NPU compile is refused). Beside them the registry has rules for qwen3.5,
  phi3 and a generic text-generation fallback. Measuring the ten found four defects, all fixed:
  - models were exported stateless when no rule named a task (OLMo-2: "beam_idx was not found");
  - optimum-intel's own per-architecture Transformers ranges were ignored until the export failed
    (Gemma, LFM2, Qwen2.5-VL); they are now read and satisfied by a managed environment;
  - a repository's own modelling code was treated as required even for architectures Transformers
    implements natively (Phi-4-mini);
  - an HTTP 429 from the Hub was reported as a transient 5xx.
  phi3 and mistral cannot be exported on Python 3.14: optimum 2.3.0, the release optimum-intel
  2.2.0 requires, binds `NormalizedConfig.with_args` through a `functools.partial`, which Python
  3.14 turned into a method descriptor (`export.optimum.normalized_config`). optimum fixed it in
  pull request 2496, which no release carries yet; Python 3.11-3.13 are not affected. mistral
  remains covered only by the generic rule.
- **A GUI or web frontend. [L]** Application services already emit structured `ProgressEvent`s and
  never import Rich, so a second frontend does not require touching the core.

---

**Resolved: output quality was never evaluated.** `PROFILE_COMPARISON.md` measured only size and
speed. `yonga eval` now measures the rest: teacher-forced KL divergence, top-1 agreement and
perplexity against a higher-precision artifact of the same weights, over a shipped multi-domain
corpus. It shows that the channel-wise symmetric Qwen3.5 artifact diverges 2.5 times as much as the
group-wise asymmetric one, on every passage (`PROFILE_COMPARISON.md` section 8). INT8 was validated
as a reference against FP16 on Qwen3-0.6B.

## Known defects still open

- **Granite is slow on the NPU.** `ibm-granite/granite-3.3-2b-instruct` generates about 4.5 tokens
  per second on the NPU in both INT4 formats, where other models of its size reach 25-40
  (`PROFILE_COMPARISON.md` section 10.4). It runs to completion; the cause was not isolated.

The previous defect, the default INT4 profiles on small models, is resolved: below 5B parameters
every device now defaults to the more accurate profile, from a registry size condition and
measurements on twelve models (`PROFILE_COMPARISON.md` section 10). What it leaves open is listed
there: no VLM was measured, so the VLM families keep ratio 1.0, and nothing was measured
between 4B and 8B.

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
