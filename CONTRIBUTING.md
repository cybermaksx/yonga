# Contributing to yonga

Bug reports, fixes and registry data are welcome. Report bugs in the
[issue tracker](https://github.com/ooguz/yonga/issues); `yonga logs JOB_ID --bundle` writes an
archive with the job's log, manifest and a host report, redacted and anonymised, that you can
attach. A security problem goes through [`SECURITY.md`](SECURITY.md), not a public issue.

## Development setup

Linux x86_64 and Python 3.11-3.14, from a clone:

```bash
git clone https://github.com/ooguz/yonga.git
cd yonga
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[openvino,export,serve,dev]"
```

These are the extras CI's test job installs. On a machine without an NVIDIA GPU, install the CPU
build of torch first: the default PyPI build on Linux carries CUDA, 1.2 GB of torch plus 3.2 GB of
NVIDIA libraries on the reference host, against a 196 MB CPU wheel. `docs/INSTALL.md` has the
command.

## Checks

CI runs these, and a change should pass them locally first (`docs/TESTING.md` section 11):

```bash
ruff check .
ruff format --check .
mypy src tests
pytest tests/unit
pytest tests/integration
```

The default `addopts` in `pyproject.toml` deselect the `network`, `intel_gpu`, `intel_npu`,
`large_model` and `slow` markers, so the default run needs neither the network nor Intel hardware.
CI never passes `-m`; a test that needs a device or the Hub must carry its marker.

On an Intel host with a GPU and an NPU, the hardware tests run against generated artifacts:

```bash
python tools/make_test_artifact.py        # tiny random Llama, needs network once
python tools/make_test_artifact.py --npu  # Qwen3-0.6B with the NPU profile
pytest -m "openvino or intel_gpu or intel_npu or slow or large_model"
```

A test whose device or input is missing skips with the command that provides it.
`docs/TESTING.md` section 6 lists each hardware test and its input.

## Changes

- Knowledge about models and devices (which architecture exports how, which profile a device
  gets, known failures and their repairs) is registry data in `src/yonga/registry/builtin/*.yaml`,
  not code. `yonga registry validate` checks it; `docs/REGISTRY_SPEC.md` describes it.
- New behaviour comes with unit tests, including its principal failures.
- A user-visible change gets an entry under `## [Unreleased]` in `CHANGELOG.md`.
- A few rules are not negotiable: tokens and other secrets are never logged or written out;
  remote code stays off unless the user passes `--trust-remote-code`, and `--yes` never implies
  it; subprocesses take argument lists, never a shell string; the Hugging Face cache is never
  modified; and yonga sends no telemetry. That includes the libraries it runs: OpenVINO's own
  usage reporting (the `openvino_telemetry` package, which Optimum Intel and NNCF reach too) is
  kept from loading in every Python process yonga starts, without writing anything under `~/intel`
  (`src/yonga/privacy.py`). A new child interpreter is started through
  `python_code_argv`/`python_module_argv` there, or imports `yonga` first.

By contributing you agree that your contribution is licensed under the GNU Affero General Public License v3.0 or later, as
the rest of the project is (`LICENSE`).
