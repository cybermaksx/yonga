# TESTING.md

## 1. Testing goals

The project must be testable on ordinary CI machines without Intel NPU/GPU while still having a clear hardware test suite for real Intel hosts.

Split tests into:

- unit tests: no network, no OpenVINO hardware requirement;
- integration tests: temporary filesystem, mocked/fake subprocesses, optional Hub network marker;
- runtime tests: OpenVINO CPU where available;
- hardware tests: Intel GPU/NPU, opt-in markers;
- acceptance fixtures: known real-world workflows and failures.

## 2. Tooling

Use:

- pytest;
- pytest-cov;
- pytest-mock or unittest.mock;
- pytest-timeout for hardware/runtime tests;
- respx only if HTTP mocking is truly needed (prefer mocking Hub adapter);
- ruff;
- mypy in strict mode (configured under `[tool.mypy]` in `pyproject.toml`).

Markers:

```ini
network
openvino
intel_gpu
intel_npu
slow
large_model
```

Default CI excludes network/hardware/large-model tests.

## 3. Unit test requirements

### Registry

Test:

- schema loads;
- duplicate rule ID fails;
- version constraints;
- specificity ordering;
- priority tie behavior;
- conflict detection;
- Qwen3.5 architecture selects Transformers 5.2 requirement;
- NPU failure signature classifies correctly;
- GPU profile resolution;
- third-party derivative returns `requires_confirmation=True`.

### Disk/temp planner

Fixtures:

- `/tmp` tmpfs 16 GiB, root 796 GiB -> choose managed disk temp;
- insufficient cache disk -> preflight failure;
- large source estimate math;
- filesystem statistics unavailable -> warning, not crash.

### Repair: tokenizer casing

Before fixture contains:

```xml
<port precision="STRiNG" />
<data element_type="string" />
```

After repair:

```xml
<port precision="STRING" />
<data element_type="string" />
```

Tests:

- only exact port precision modified;
- lower-case `element_type="string"` untouched;
- no match => no change;
- backup/hash recorded;
- second application is idempotent;
- unrelated `STRiNG` text is not blindly rewritten if parser/context does not match.

### Failure classifier

Input snippets -> codes:

- `OneShotBufferizeVPU2VPUIP failed : Only supports FP16.` -> `npu.compiler.unsupported_graph.qwen35` when architecture qwen3_5;
- same text on unrelated architecture -> generic NPU compiler graph failure, not qwen35-specific;
- `Unsupported element type: STRiNG` while reading tokenizer -> tokenizer IR incompatible;
- no space / iostream save failure plus disk evidence -> insufficient disk/temp classification;
- HF 401/403/gated -> authentication/gated;
- package ABI mismatch signature -> ABI diagnostic.

### Source resolver

Cases:

- OpenVINO repo -> direct artifact;
- safetensors source -> direct conversion source;
- GGUF-only + trusted registry derivative -> suggestion requiring confirmation;
- GGUF-only + no derivative/support -> stop;
- explicit `--source` -> use only after lineage warning/confirmation policy.

### Manifest

- deterministic schema;
- no secrets;
- paths relative where appropriate;
- hashes computed after repair;
- source revision immutable SHA recorded.

## 4. Integration tests

### Fake exporter

Implement a fake exporter script/process that emits progress, writes minimal fixture files, and exits with chosen codes. Verify subprocess adapter:

- argv list, no shell;
- TMPDIR variables;
- log capture;
- cancellation;
- secret redaction;
- non-zero exit classification.

### Artifact staging

Verify:

- incomplete staging never appears as ready artifact;
- successful finalization is atomic at directory level;
- `--force` behavior;
- existing ready artifact reused.

### Hub adapter

Mock `HfApi`/`snapshot_download` for most tests. Optional network tests against a tiny public model should be separate and resilient.

## 5. OpenVINO CPU tests

When OpenVINO is installed in CI, use tiny hand-built or fixture IRs to verify:

- device enumeration;
- model read/compile CPU;
- tokenizer extension registration when package installed;
- compile-cache key generation.

Avoid downloading multi-GB models in normal CI.

## 6. Hardware tests

### Intel GPU host

Acceptance test:

1. `doctor` sees GPU;
2. prepared tiny compatible model compiles;
3. generation succeeds;
4. metrics non-empty;
5. compile cache reused on second load.

### Intel NPU host

Acceptance test:

1. `/dev/accel` accessible;
2. OpenVINO lists NPU;
3. tiny known-supported model compiles/generates;
4. profile properties accepted;
5. explicit unsupported fixture returns diagnostic rather than false driver error.

### What the hardware suites run

The tests live in `tests/hardware/` and never download anything; a missing
input skips the test with the command that produces it.

| Marker | File | Covers | Input |
|---|---|---|---|
| `intel_gpu` | `test_intel_gpu.py` | generation and metrics, compile-cache reuse, probe, streaming, `serve` (plain and SSE, real runtime), GPU logits against CPU f32 | tiny artifact (`tools/make_test_artifact.py`) |
| `intel_npu` | `test_intel_npu.py` | device access, deferred raw compiles, typed pipeline failures, generation with manifest-shaped properties, probe, `probe --quick` and `--timeout` (inconclusive, or a pass through the child process), streaming and a second turn | NPU artifact (`tools/make_test_artifact.py --npu`) |
| `intel_gpu`, `intel_npu`, `slow` | `test_gguf_import.py` | the real GGUF import child process, GPU generation from its IR, and the NPU refusal named by `npu-static-reshape-broadcast` | a cached `Qwen/Qwen2.5-0.5B-Instruct-GGUF` Q4_0 file, or `YONGA_TEST_GGUF` |
| `intel_gpu`, `large_model` | `test_vlm.py` | text chat through the VLM pipeline, with history | `YONGA_TEST_VLM_ARTIFACT` |

On a host with both devices:

```bash
pytest -m "openvino or intel_gpu or intel_npu or slow or large_model"
```

`YONGA_TEST_BROKEN_ARTIFACT` and `YONGA_TEST_NPU_ARTIFACT` override the
generated artifacts' locations.

## 7. Qwen3.5 manual acceptance procedure

A hand-run procedure on a real machine; there is no automated test that performs the whole flow.
Its parts are covered separately: offline, by the resolver, planning, repair, classifier and
fallback unit tests (`tests/unit/test_source_resolver.py`, `test_prepare_planning.py`,
`test_repair_tokenizer_precision.py`, `test_registry_classifier.py`, `test_prepare_fallback.py`),
and on hardware by `tests/hardware/test_vlm.py`, which chats through a prebuilt VLM artifact. The
commands are in `USAGE.md`, "The full reference case".

Environment expected to resemble:

- Intel Core Ultra Series 2 / Lunar Lake;
- Intel Arc iGPU;
- Intel NPU;
- OpenVINO 2026.4 family.

Flow:

1. inspect HauhauCS repo;
2. resolve third-party safetensors derivative only after approval;
3. export artifact;
4. if fixture contains `precision="STRiNG"`, repair;
5. tokenizer CPU read/compile succeeds;
6. language/text/vision numeric models compile on GPU;
7. NPU compilation hits known unsupported signature and is classified;
8. auto mode recommends/selects GPU;
9. VLM pipeline generates a response;
10. hidden reasoning does not render in default chat display.

The procedure is not part of CI: it downloads about 17.5 GiB and needs the hardware above.

## 8. Chat tests

Use fake runtime outputs.

Cases:

- Markdown rendering receives cleaned answer;
- complete `<think>secret</think>answer` -> only answer rendered;
- unclosed `<think>` -> hidden content not rendered;
- ordinary word `think` unaffected;
- `/tokens`, `/reset`, `/stats`, `/exit` behavior;
- history file path uses XDG data/cache policy;
- generation error returns to prompt.

## 9. Benchmark tests

Fake metrics validate:

- mean/median calculations if implemented;
- exact prompt hashes;
- warmups excluded;
- JSON schema;
- artifact/hardware/runtime identity present;
- comparison refuses incompatible benchmark settings by default.

## 10. Snapshot/golden tests

Use sparingly for CLI tables because terminal width can make them brittle. Prefer testing structured render models, plus a few fixed-width snapshots for important UX.

## 11. CI matrix

`.github/workflows/ci.yml`, on ubuntu-latest:

```text
lint           ruff check and ruff format --check over the whole tree
typing         mypy src tests, Python 3.11-3.14, with the serve extra and no OpenVINO
tests          unit and integration, Python 3.11-3.14, full toolchain (OpenVINO CPU plugin)
base-install   unit and integration with no OpenVINO or export extras, Python 3.11 and 3.14
package        build, twine check --strict, and a smoke test of the wheel in a clean venv
appimage       the AppImage built from that wheel, then --version, registry validate and a
               doctor whose OpenVINO runtime must be usable, run outside the checkout from a
               directory holding a decoy typer.py the image must not import
```

No job passes `-m`: the default `addopts` deselect `network`, `intel_gpu`, `intel_npu`,
`large_model` and `slow`, and overriding that is how a CI run would start depending on hardware or
a network it does not have. The `appimage` job is the exception to "no network in CI": building
the image downloads the interpreter and the AppImage tools, which are pinned and checked against
their SHA-256, and installs yonga's Python dependencies from PyPI. Those are resolved at build time
within the ranges in `pyproject.toml`, not pinned, so the build records what it bundled in
`usr/share/yonga/packages.txt` inside the image and in `yonga-X.Y.Z-x86_64.AppImage.packages.txt`
beside it, which the job uploads. `.github/workflows/release.yml` publishes a tagged version
(`docs/RELEASING.md`); it builds the image with the same script from the released wheel, smoke-tests
`--version` and `registry validate` beside the same decoy `typer.py`, and attaches the package list
to the release.

Hardware CI can be self-hosted later and must not block ordinary contributor PRs unless specifically required.

## 12. Definition of done for a feature

A feature is complete when:

- typed API exists;
- user-visible errors are classified;
- unit tests cover success + principal failures;
- CLI help updated;
- relevant spec/registry docs updated;
- no secrets/raw tokens logged;
- path handling works with spaces;
- cancellation/cleanup considered for long operations.
