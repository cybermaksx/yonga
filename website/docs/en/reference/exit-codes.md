# Exit codes

Every yonga command ends with a stable exit code, so a script can tell "the NPU cannot compile this
model" from "the disk is full" without parsing any text. This page lists every code with the
failures that typically cause it, explains how a code is chosen, and describes the JSON document a
failed command prints with `--json`. The codes are defined by `ExitCode` in
[`src/yonga/domain/diagnostics.py`](https://github.com/ooguz/yonga/blob/main/src/yonga/domain/diagnostics.py)
and specified in
[CLI_SPEC.md section 15](https://github.com/ooguz/yonga/blob/main/docs/CLI_SPEC.md#15-exit-codes).

## All codes { #codes }

The diagnostic codes in the last column are examples, not a complete list.
[Troubleshooting](../troubleshooting.md) explains the common ones, and
[TROUBLESHOOTING.md](https://github.com/ooguz/yonga/blob/main/docs/TROUBLESHOOTING.md#3-registry-issue-codes)
lists every code the shipped registry can produce.

| Code | Meaning | Typical causes |
|---|---|---|
| 0 | Success | The command did what you asked. `--help` exits 0, and so does a command whose output you cut short, such as `yonga logs JOB_ID --raw \| head`. |
| 2 | Success with warnings | `doctor` found warnings only, for example a small RAM-backed `/tmp`. `prepare` finished, but on a fallback device or with another warning. `bench` measured some runs while others failed. `probe` found no device that passed or failed, and at least one inconclusive (`--quick`, or `--timeout` reached). `self install` put the copy in place, but `yonga` on your `PATH` does not run it yet. `registry validate` found an installed remote bundle that no longer verifies. |
| 10 | Environment unusable | A required package is missing (`runtime.openvino.missing`). No permission on the NPU device node (`npu.device.permission_denied`). A driver that does not initialise (`device.driver_missing`). Native packages from different releases (`packages.abi_mismatch`). An exporter that cannot import (`export.torchvision.build_mismatch`) or cannot run on this Python (`export.optimum.normalized_config`). An unmet exporter version pin with `--no-managed-env` (`preflight.exporter_requirement_unsatisfied`). `doctor` exits 10 when something yonga needs is missing. |
| 20 | Model or source could not be resolved | The repository does not exist or is not visible to you (`hub.repo.not_found`). `--offline` and the model is not in the local cache (`hub.offline.missing`). `search` with `--offline` (`hub.offline.search_unavailable`). A GGUF repository whose architecture cannot be imported (`source.gguf_unsupported`). A managed export environment that could not be installed, or that needs the network while offline (`export.env.install_failed`, `export.env.offline`). |
| 21 | Authentication required | A gated or private repository and no valid Hugging Face token (`hub.auth.gated`, `hub.auth.required`). |
| 22 | Remote code approval required | The model can only be converted by running Python code from its repository, and you did not pass `--trust-remote-code` (`source.remote_code_required`). |
| 30 | Not enough disk or memory | Preflight refused before downloading anything (`preflight.insufficient_disk`). A write hit a full filesystem (`storage.insufficient_disk`, `disk.insufficient_space`). Memory ran out (`runtime.out_of_memory`, `memory.out_of_memory`). |
| 40 | Conversion failed | The export failed and no rule recognised why (`export.failed`). |
| 41 | Validation failed | The exported IR cannot be read back (`tokenizer.ir.string_precision`, `ir.unsupported_element_type`), or the artifact failed validation for another reason. |
| 42 | Repair needed but disabled | Validation found a problem that a known repair fixes, and you passed `--no-repair` (`repair.declined`). |
| 50 | Requested device unsupported | `prepare` cannot use the device: OpenVINO does not report it (`device.not_present`), the registry knows it cannot run this model (`device.known_unsupported`), or it failed its probe, including when `--device auto` has tried every fallback. |
| 51 | Compilation failed | The device's compiler refused the model in `probe`, `chat` or `bench` (`npu.compiler.unsupported_graph`, `npu.compiler.pass_failed` and the other `npu.compiler.*` codes). |
| 52 | Generation failed | Generation failed after the model had loaded. `bench` also exits 52 when none of its runs produced a measurement. |
| 60 | Cancelled | You pressed Ctrl-C (`job.cancelled`), declined a confirmation (`job.declined`), or a confirmation could not be asked because there is no terminal and no `--yes` (`job.confirmation_unavailable`). |
| 70 | Registry or configuration error | An invalid option value or configuration file, a registry problem, a profile whose group size does not fit the model (`export.quantization.group_size_mismatch`), `--json` given to `chat` or `serve`, and any command line yonga cannot parse: an unknown command or option, a missing argument, a value of the wrong type, or `yonga` with no command. |
| 99 | Unexpected internal error | A bug (`internal.unexpected_error`), or a failure that nothing recognised (`runtime.unknown_error`). Please [report it](../troubleshooting.md#unknown). |

## How a code is chosen { #how }

Every failure is a diagnostic with a **category**, and the category decides the exit code. A few
rules refine that:

- **40, 42 and 52 name a stage, not a cause.** They mean "conversion", "repair" and "generation",
  and appear only when the failure's cause is unknown. A conversion that fails for a recognised
  reason exits with that reason's code: the group-size failure exits 70, not 40. Code 41 is both
  the validation stage's code and the code of the IR and tokenizer categories.
- **`prepare` reports device failures as 50.** Whatever the category, a device that `prepare`
  cannot use ends with 50. `probe`, `chat` and `bench` keep the category's code, so the same NPU
  compiler refusal is 50 from `prepare` and 51 from `probe`.
- **A broken host is 10 everywhere.** No permission on the device node, or a driver that does not
  initialise, exits 10 from every command, and `prepare` does not fall back to another device for
  it: the host needs fixing before any choice of device matters.
- **Usage errors are 70, not 2.** Most command-line tools exit 2 for a typo. yonga keeps 2 for
  warnings and inconclusive probes, so a script that retries an inconclusive `probe` never mistakes
  a typo for one.

The diagnostic's own code and category are kept in the `--json` document and in the job log, so the
reason is not lost when the exit code is a broader one.

## The `--json` failure document { #json }

With `--json`, a failed command writes exactly one JSON document to standard output and the
human-readable diagnostic, without decoration, to standard error. This is real output from
`yonga --offline --json search qwen`, formatted for reading:

```json
{
  "ok": false,
  "exit_code": 20,
  "diagnostic": {
    "code": "hub.offline.search_unavailable",
    "category": "download_failure",
    "message": "Offline mode is active, and searching the Hugging Face Hub is a network request: there is no local copy of the Hub to search.",
    "fallback_devices": [],
    "job_id": null,
    "title": "Search needs the network",
    "severity": "error",
    "remediation": [
      "Run the search without offline mode (no --offline, and YONGA_OFFLINE / HF_HUB_OFFLINE unset): yonga --json search qwen",
      "A model already in the local cache can still be inspected offline: yonga --offline inspect MODEL"
    ],
    "log_path": null
  }
}
```

| Field | What it holds |
|---|---|
| `ok` | `false` for a failure. A successful command's document also starts with `ok`, set to `true`. |
| `exit_code` | The process exit code, as an integer. |
| `diagnostic.code` | The stable diagnostic code, such as `hub.offline.search_unavailable`. |
| `diagnostic.category` | The failure category the exit code was derived from, such as `download_failure`. |
| `diagnostic.title`, `diagnostic.message` | The same text the terminal shows. |
| `diagnostic.severity` | `error` for a failure. |
| `diagnostic.remediation` | The suggested steps, in order. A step that names a command repeats your own options, ready to copy. |
| `diagnostic.fallback_devices` | Devices the failure suggests instead, when it is about one device. |
| `diagnostic.job_id`, `diagnostic.log_path` | The job and its log file, for commands that run a job (`prepare`, `probe`, `bench`). `null` otherwise. |

`--json` is a global option, so it goes before the command name, and most commands also accept it
after their own name (`yonga list --json`). `chat` and `serve` reject it with exit 70.

## Using exit codes in scripts { #scripts }

Check whether a machine is ready, treating warnings as success:

```bash
yonga doctor > /dev/null
case $? in
  0) echo "ready" ;;
  2) echo "ready, with warnings" ;;
  *) echo "not usable: run 'yonga doctor' to see why" ;;
esac
```

Read the diagnostic code of a failure (this example uses `jq`):

```bash
yonga --json --offline inspect Qwen/Qwen3-0.6B 2>/dev/null | jq -r '.diagnostic.code // "ok"'
```
