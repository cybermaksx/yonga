# Privacy and security

yonga runs your models on your own machine and sends nothing about how you use it. This page lists
every case in which it does use the network, how it handles your Hugging Face token, when it will
run code from a model repository, how to check what you downloaded, how to keep the local server
private, and how to report a vulnerability. The security policy itself is
[SECURITY.md](https://github.com/ooguz/yonga/blob/main/SECURITY.md) on GitHub.

## No telemetry { #telemetry }

yonga has no telemetry: no usage statistics, no crash reports, no update checks. A `telemetry`
setting exists only so that turning it on can be refused. `telemetry = true` in the configuration
file, or `YONGA_TELEMETRY` set to true, stops yonga with `config.telemetry.rejected`.

yonga also stops OpenVINO from reporting on its behalf. OpenVINO's conversion API counts each time
it is imported in `~/intel/stats` and, unless a consent file exists in `~/intel`, reports it to
Google Analytics through the `openvino_telemetry` package. yonga makes that package impossible to
import in every Python process it starts, including the exporter that runs in a managed export
environment. Nothing is sent, the count is not touched, and yonga does this without writing an
opt-out file into your home directory.

!!! note
    If `~/intel/stats` still changes, another program imported OpenVINO, for example a plain
    `python -c "import openvino"`. That is why yonga's own advice checks imports with
    `yonga doctor --full`, which keeps the reporting switched off.

## When yonga uses the network { #network }

yonga makes a network request only when a command needs one:

| What you run | Where it connects | Why |
|---|---|---|
| `search`, `inspect`, `prepare` | Hugging Face Hub | Model metadata, and for `prepare` the model files. `prepare` downloads only the files the conversion reads. |
| `doctor`, when a token is configured | Hugging Face Hub | To check that the token is valid and show the account it belongs to. |
| `prepare`, when a managed export environment has to be built | PyPI and PyTorch's CPU index (`download.pytorch.org`) | To install the conversion toolchain into `~/.cache/yonga/envs`. The plan asks for your confirmation first. |
| `registry update` | The `https://` address you give | To fetch a signed registry bundle and its signature. yonga has no default address. |
| `publish` | Hugging Face Hub | To upload an artifact you chose to publish. |

`chat`, `probe`, `bench`, `eval` and `serve` run from the prepared artifact on your disk and contact
nothing. `serve` listens for requests but never fetches anything a request names (see
[The local server](#serve)).

### Offline mode { #offline-mode }

To make sure yonga never contacts the Hub, use the global `--offline` option, or set
`YONGA_OFFLINE` or `HF_HUB_OFFLINE`:

```bash
yonga --offline prepare Qwen/Qwen3-0.6B --device cpu
```

`inspect` and `prepare` then work from the local Hugging Face cache only. `search` refuses, because
a search is a network request, and a model that is not cached fails with `hub.offline.missing`.
An existing managed export environment is reused offline, but a new one cannot be built.

## Your Hugging Face token { #tokens }

You need a token only for gated or private models. yonga never asks for one and never stores one:
it uses the token the `huggingface_hub` library finds, from the `HF_TOKEN` environment variable or
from a login made with `hf auth login`.

The token never appears in yonga's output:

- Job logs, diagnostics and `--debug` output pass through a redactor that replaces token-shaped
  text, and the value of variables such as `HF_TOKEN`, with `***REDACTED***`.
- When a suggested command repeats your command line, options that carry a credential are left out
  and secret-looking values are masked.
- `doctor` shows only whether you are signed in, the account name and where the credential was
  found (`credential source: environment:HF_TOKEN`, for example), never the token.
- A bug-report bundle (`yonga logs JOB_ID --bundle`) is redacted the same way and also anonymises
  your home directory, host name, login name and Hugging Face account name. Read it before you
  attach it anywhere.

## Code from model repositories { #remote-code }

Some repositories ship their own Python modelling code, which Transformers runs only with
`trust_remote_code`. yonga never enables it on its own:

- If a model can only be converted by running its repository's code, `prepare` stops before doing
  anything, with `source.remote_code_required` (exit 22).
- You allow it with `--trust-remote-code`. Even then yonga asks in the terminal, and **`--yes`
  never answers that question**. In a session without a terminal, the answer is no.
- When the registry knows that Transformers implements the architecture itself (Phi-4-mini is an
  example), the repository's code is ignored and never run, and the plan says so.

Review the repository's code before you pass the flag. Its code then runs with your user's
permissions.

## The compatibility registry { #registry }

The built-in registry ships inside yonga. You can add a newer **remote bundle** between releases,
and yonga treats it as untrusted until it verifies:

- It is fetched only when you run `yonga registry update`, from an `https://` address or a file on
  disk.
- It must carry an Ed25519 signature that verifies against a public key **you** placed in
  `~/.config/yonga/registry-keys/`. No key is trusted out of the box.
- It is checked again on every load: not expired, not older than the installed one, not made for a
  newer yonga, and plain YAML data in the registry's strict schema.
- It can only select repairs and tool-call parsers already compiled into yonga. Registry data can
  never deliver code.

`yonga registry status` shows the installed bundle and whether it still verifies, and
`yonga registry reset` removes it. The
[registry spec](https://github.com/ooguz/yonga/blob/main/docs/REGISTRY_SPEC.md#121-remote-bundles-implemented)
has the details.

## Checking what you download { #downloads }

Each [GitHub release](https://github.com/ooguz/yonga/releases) carries the AppImage with a
`.sha256` file. Check it before you run the AppImage:

```bash
sha256sum --check yonga-0.5.1-x86_64.AppImage.sha256
```

The release also carries `yonga-0.5.1-x86_64.AppImage.packages.txt`. It lists the bundled Python
interpreter, the yonga wheel with its SHA-256, and the exact version of every Python package in the
image. The same list is inside the image as `usr/share/yonga/packages.txt`. The build pins the
interpreter and the AppImage tools by SHA-256; the Python packages are resolved from PyPI when the
image is built, which is why each release records them.

## The local server { #serve }

`yonga serve` gives other programs an OpenAI-compatible API to your model. It is private by
default:

- It listens on `127.0.0.1` (port 8000), so only programs on your machine can reach it. `--host`
  changes the address. When it is not a loopback address, yonga warns that the model is reachable
  from other machines and, when no key is set, that anyone who can reach it can use it.
- To require a key, set `YONGA_API_KEY` before you start the server. Every endpoint except
  `/health` then needs `Authorization: Bearer <key>` and answers 401 otherwise. The key is read
  from the environment only, never from a command-line option, and is never logged or printed.
- It never opens a URL a request contains. Images are accepted only embedded in the request as
  `data:image/png`, `jpeg`, `webp` or `gif` base64 URLs, and any other URL is refused.
- Tool calls the model writes are returned to your client, which decides whether to run them. The
  server never runs them.

```bash
export YONGA_API_KEY="$(openssl rand -hex 32)"
yonga serve ARTIFACT_ID --host 0.0.0.0
```

Set a key whenever `--host` is not a loopback address. See
[OpenAI-compatible server](guide/serve.md) for using the API.

## What yonga keeps on your machine { #local-data }

Everything yonga writes stays under its own directories (see `yonga registry paths`):

- prepared artifacts, job logs and benchmark results under `~/.local/share/yonga`;
- the prompts you type in `chat`, as a line-editing history with one file per artifact, under
  `~/.local/share/yonga/chat-history`;
- temporary files, the compile cache and managed export environments under `~/.cache/yonga`;
- your settings and registry overrides under `~/.config/yonga`.

Source models stay in the normal Hugging Face cache and are never modified. yonga never runs
`sudo`, never changes system settings, and never installs packages into the Python environment you
installed it into. [FAQ](faq.md#uninstall) explains how to remove everything.

## Reporting a vulnerability { #vulnerabilities }

Report a vulnerability privately through a GitHub security advisory:
<https://github.com/ooguz/yonga/security/advisories/new>. Please do not open a public issue, pull
request or discussion for it.

Say which version and install method (pip or AppImage) are affected, how to reproduce the problem,
and what an attacker gains. The answer comes in the advisory, and a confirmed problem is fixed in a
new patch release. Only the newest release receives fixes.

In scope are yonga's own code and what it ships, including the `serve` API, token handling, the
remote-code confirmation, signed registry bundles, the bug-report bundle's redaction and the
release files. Problems in OpenVINO, Transformers, PyTorch and the other packages yonga installs,
or in the models themselves, are best reported upstream.
