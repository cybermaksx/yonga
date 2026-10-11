# Maintenance

Prepared models, compiled blobs, export environments and job logs all take disk space, and yonga never deletes any of them on its own. This page covers the commands that show what is kept, free the space safely, check the health of the host, keep the AppImage on your `PATH`, look after the compatibility registry and upgrade yonga. `remove`, `cache prune`, `env remove` and `self uninstall` show what they will delete and ask first.

## See what is prepared

To list the artifacts on this machine, run:

```bash
yonga list
```

```text
Artifact          Model            Rev      Profile            Dev     Size  Benchmark   Ready
──────────────────────────────────────────────────────────────────────────────────────────────
cc24636a2f49356a  Qwen/Qwen3-0.6B  c1899de  npu-int4-sym-g128  NPU  387 MiB  2026-10-11  ✓ yes
```

The first column is the **artifact id**, the 16-character name you pass to `chat`, `bench`, `probe`, `serve`, `eval`, `publish` and `remove`. A unique prefix of at least six characters also works.

A failed or interrupted `prepare` can leave a working directory behind (a partial export in the staging area, or files in the managed temp directory). yonga keeps the most recent ones so that a retry can reuse a finished export, and `yonga list` prints one line when there is space to reclaim. To retire the stale ones, run:

```bash
yonga list --clean
```

The same sweep also runs on its own whenever a new `prepare`, `probe` or `bench` job starts. It never touches the directories of a job that is still running. How long and how many are kept is set by `workspace_retention_hours` (72) and `workspace_retention_max` (3) in the configuration file.

## Remove an artifact

To delete a prepared artifact, pass its artifact id:

```bash
yonga remove cc24636a2f49356a
```

yonga shows the plan and asks before it deletes anything:

```text
Remove cc24636a2f49356a
  Path   …/artifacts/Qwen--Qwen3-0.6B/c1899de289a0--npu-int4-sym-g128
  Frees  559 MiB
  Cache  …/compile/1bccad8a2629cc22
The Hugging Face download cache is never touched.
```

It removes the artifact directory and the compile cache entries that belong to it, and nothing else. The source model you downloaded stays in the Hugging Face cache.

You can also name the artifact by its repository id (`yonga remove Qwen/Qwen3-0.6B`) or by its directory path. Because deleting cannot be undone, the reference must match **exactly one** artifact: if a repository id or id prefix matches several (for example the same model prepared for the NPU and the GPU), yonga removes nothing, lists every candidate with its profile and device, and exits 70, even with `--yes`. Pass the artifact id of the one you mean.

| Situation | What happens | Exit code |
|---|---|---|
| You answer yes, or pass `--yes` | The artifact and its compile cache entries are deleted | 0 |
| You answer no | Nothing is removed | 60 |
| No terminal to ask (a script, a pipe) and no `--yes` | Nothing is removed | 60 |
| The reference matches several artifacts, or none | Nothing is removed; the candidates or `yonga list` are named | 70 |

!!! tip
    `chat`, `bench`, `probe`, `eval` and `publish` only read an artifact, so when a repository id matches several they take the newest ready one and say so. Only `remove` insists on one exact match.

## The compile cache

When OpenVINO compiles a model for a device, yonga keeps the compiled blobs in the compile cache (`~/.cache/yonga/compile/` by default), so the next `chat`, `serve` or `bench` on that artifact and device loads quickly. There is one entry per artifact, device, OpenVINO version and set of runtime properties.

To see the entries, run:

```bash
yonga cache list
```

```text
Compile cache  …/cache/compile
Key                  Size  Last used   Dev  OpenVINO  Owner             Status    Model
─────────────────────────────────────────────────────────────────────────────────────────────────
1bccad8a2629cc22  172 MiB  2026-10-11  NPU  2026.4.1  cc24636a2f49356a  ✓ in-use  Qwen/Qwen3-0.6B
1 entry, 172 MiB in total (1 in-use).
```

`yonga cache` on its own does the same. Each entry has one of four statuses:

| Status | Meaning |
|---|---|
| `in-use` | The owning artifact exists and the installed OpenVINO would reuse the blobs. |
| `orphaned` | The owning artifact is gone: it was removed by hand, or the entry was made with another `--data-dir`. |
| `stale-runtime` | Built by a different OpenVINO release than the one running now, so it can never be reused. This is what an OpenVINO upgrade leaves behind. |
| `unattributed` | No ownership record, and no known artifact produces its key. Typically an entry made by yonga 0.3.0 or earlier, which wrote no ownership record. |

### Prune the cache

To remove the orphaned and stale-runtime entries, run:

```bash
yonga cache prune
```

It prints the plan (each entry, why it was chosen and the total size) and asks before deleting. These options widen or control the selection:

| Option | Effect |
|---|---|
| `--older-than DAYS` | Also remove any entry not used for this many days. |
| `--unattributed` | Also remove entries that cannot be traced to any artifact. |
| `--all` | Empty the whole compile cache. |
| `--dry-run` | Show the plan; delete nothing. |
| `--yes`, `-y` | Do not ask for confirmation. |
| `--json` | Emit the outcome as JSON. |

To preview what a monthly clean-up would free, combine them:

```bash
yonga cache prune --older-than 30 --dry-run
```

Pruning only ever touches the compile cache directory. Removing an entry that a running `chat` or `serve` uses costs that model a recompile the next time it loads, nothing more. Unless you pass `--all`, an entry that something starts using while the prompt waits is skipped and reported.

`cache prune` exits 0 on success (including when there is nothing to prune, and for a dry run), 2 when some entries could not be removed, 60 when you decline or there is no terminal to ask and no `--yes`, and 70 for an invalid `--older-than`.

!!! note
    `yonga doctor` reports the compile cache size and how much of it is reclaimable, and turns that line into a warning when the reclaimable part reaches 10 GiB, or a quarter of the free space on the cache filesystem (never less than 1 GiB).

Details: [CLI_SPEC.md, section 12b](https://github.com/ooguz/yonga/blob/main/docs/CLI_SPEC.md#12b-cache).

## Managed export environments

When the packages yonga runs with cannot export a model (for example, the architecture needs a different Transformers version), `prepare` builds a separate, isolated environment for the export, after you confirm it in the preparation plan. The AppImage, which carries no export toolchain, builds one for its first conversion. These environments live under the cache directory (`~/.cache/yonga/envs/`) and are reused by later jobs. See [Preparing models](prepare.md) for when they are built.

To see them and the disk they use, run:

```bash
yonga env list
```

```text
fbfac3ac91ba    1.4 GiB  2026-10-11  …/cache/envs/export-fbfac3ac91ba
    nncf==3.4.0, openvino==2026.4.1, openvino-tokenizers==2026.4.1.0, optimum==2.3.0,
    optimum-intel==2.2.0, torch==2.14.0, torchvision==0.29.0, transformers==5.5.4
base Python      98 MiB  …/cache/envs/python-3.14.7-3d38950fe1ce  (the environments' base)
1 environment, 1.5 GiB in total.
The base Python is a copy of the AppImage's interpreter; it is removed with the last environment
made from it.
```

To delete one, pass its key or a unique prefix of it:

```bash
yonga env remove fbfac3ac91ba
```

It asks first (`--yes` answers for you; without a terminal and without `--yes` it removes nothing and exits 60). Removing an environment does not affect the artifacts it exported. If a later export needs it again, yonga rebuilds it.

## Job logs

Every `prepare`, `probe` and `bench` run is a job, and each job writes a log in JSON Lines format under the data directory (`~/.local/share/yonga/logs/`). To list the 20 most recent jobs, run:

```bash
yonga logs
```

```text
Job logs in …/logs
job       started           command  model             device  status    took     log
─────────────────────────────────────────────────────────────────────────────────────
9dabb616  2026-10-11 20:03  bench    cc24636a2f49356a  —       ✓ ok    48.5 s   888 B
de862650  2026-10-11 20:01  prepare  Qwen/Qwen3-0.6B   NPU     ✓ ok    1m 09s  88 KiB
Start times are local, UTC+03:00.
```

A job's outcome is one of `ok`, `failed`, `cancelled`, `running` or `unfinished`. To read one job, pass its id or a unique prefix:

```bash
yonga logs 9dab
```

```text
Job 9dabb616-2dda-4418-9c9d-16e7f3c42a34
(…/logs/9dabb616-2dda-4418-9c9d-16e7f3c42a34.jsonl)
  bench cc24636a2f49356a  ✓ ok, took 48.5 s
  Started 2026-10-11 20:03:11 UTC+03:00; the times below are local.
  20:03:11.697  info    job.created                job created
  20:04:00.231  info    bench.saved                benchmark result saved
  20:04:00.231  info    job.completed              job completed
```

Add `--raw` to print the JSON Lines file exactly as it is on disk, one JSON object per line, for `jq` or a text editor:

```bash
yonga logs 9dab --raw
```

To attach a log to a bug report, `--bundle` writes a redacted, anonymised `.tar.gz` with the job log, the job state, the artifact manifest and a host report. It goes to `yonga-report-<job>.tar.gz` in the current directory unless you pass `--output FILE`:

```bash
yonga logs 9dab --bundle
```

[Troubleshooting](../troubleshooting.md) explains how to read a failed job's log.

## Check the host with doctor

`yonga doctor` checks everything yonga depends on and suggests a fix for every warning. Run it after installing, after a driver or kernel update, and whenever something stops working:

```bash
yonga doctor
```

The report has these sections:

| Section | What it checks |
|---|---|
| Host | Operating system, kernel, Python, how yonga is installed (AppImage or Python environment, and what `yonga` on `PATH` runs), CPU, RAM and swap. |
| OpenVINO | The OpenVINO runtime, GenAI and Tokenizers versions and whether they belong to the same release (the ABI family), and the export toolchain (`optimum-intel`, `nncf`, `transformers`) with a test import of the exporter. In the AppImage the toolchain is a single line saying exports run in a managed environment. |
| Devices | Which OpenVINO devices (CPU, GPU, NPU) are present and usable. |
| NPU | The NPU device node (`/dev/accel/accel0`) and your access to it, the driver and compiler versions and the compiler libraries. |
| Storage | Free space where artifacts and caches go, the system temp directory, the managed temp directory, the export environments and the compile cache. |
| Hugging Face | Whether you are logged in (needed only for gated or private models), the Hugging Face cache directory and offline mode. |
| Registry | Only with `--full`: the number of rules, the loaded layers and the validation result. |

`--full` also repeats an exporter import check that failed last time instead of reporting the recorded failure, and lists every other filesystem it measured. `--json` emits the whole report as one JSON document.

An excerpt from the reference machine:

```text
Devices
✓  CPU  Intel(R) Core(TM) Ultra 7 268V  intel64
✓  GPU  Intel(R) Arc(TM) Graphics (iGPU)  GPU: vendor=0x8086 arch=v20.4.4
✓  NPU  Intel(R) AI Boost  4000

NPU
✓  /dev/accel/accel0   accessible  group render
✓  Group               render  the current user is a member
…
Usable, with warnings above.
```

The exit code tells a script whether the host can run models:

| Exit code | Meaning |
|---|---|
| 0 | A usable OpenVINO runtime and device exist, and nothing needs attention. |
| 2 | Usable, with warnings. A small RAM-backed `/tmp` alone causes this; yonga works around it with its own disk-backed temp directory, so no action is needed. |
| 10 | No usable OpenVINO runtime or device. The last line says why: the runtime is not installed (with the `pip install` command), cannot be imported, or no device was found. |

[Troubleshooting](../troubleshooting.md) covers the common warnings, and [Installation](../installation.md) the NPU driver and permission setup.

## Put the AppImage on your PATH

The AppImage can install itself as the `yonga` command. From the downloaded file, run:

```bash
chmod +x yonga-0.5.1-x86_64.AppImage
./yonga-0.5.1-x86_64.AppImage self install
```

It copies the AppImage to `~/.local/bin/yonga` (choose another directory with `--bin-dir DIR`) and checks that `yonga` on your `PATH` now runs that copy. If it does not, it tells you what to do and exits 2: on Ubuntu and Debian, log out and back in once after the first install, or run `export PATH="$HOME/.local/bin:$PATH"` in the current shell. yonga never edits your shell startup files and never uses `sudo`.

It replaces an older yonga AppImage at that path after asking (`--yes` answers for you), and refuses to overwrite anything that is not an AppImage, such as a `pip` or `pipx` command, a symbolic link or a directory (exit 70). The downloaded file can be deleted afterwards.

To take it off your `PATH` again, run:

```bash
yonga self uninstall
```

This removes only the file at `~/.local/bin/yonga`. Your prepared models, logs, caches and settings stay where they are; the command prints their directories. To free that space too, use `yonga remove`, `yonga cache prune --all`, or delete the directories.

!!! note
    `self install` works only when yonga runs from the AppImage. A `pip` install already has its own `yonga` command in the environment's `bin` directory; `self install` there exits 70 and says where it is.

## The compatibility registry

The registry is the data yonga uses to decide which architectures, devices and conversion profiles work, and how to recognise known failures. It ships inside yonga. These commands look into it:

| Command | Use it to |
|---|---|
| `yonga registry explain MODEL` | See which rules matched a model, which were rejected and why, and which won. Add `--device npu` to explain for one device. Reach for this when a decision surprises you. |
| `yonga registry profiles` | List the conversion profiles that `prepare --profile` accepts. Add `--device` to show one device. |
| `yonga registry validate` | Load every registry layer and check it is consistent. Useful after you write your own override rules. |
| `yonga registry paths` | Show where the registry layers, the configuration and the managed data live. |

For example, to see which NPU profile Qwen3-0.6B gets:

```bash
yonga registry explain Qwen/Qwen3-0.6B --device npu
```

```text
Profile
    Rule                Kind      Spec   Prio   Outcome
────────────────────────────────────────────────────────────────────────────────────────────────────
·   npu-int4-sym-cw     profile     20    100   matched  openvino 2026.4.1 satisfies >=2026.2,<2027;
                                                generic rule, no constraints
✓   npu-int4-sym-g128   profile     20    110   winner  openvino 2026.4.1 satisfies >=2026.2,<2027;
                                                751.6M parameters is below 5B
```

`registry validate` exits 0 when the registry is valid, 2 when it is valid but an installed remote bundle was ignored, and 70 when it finds problems.

### Signed registry updates

New registry knowledge can arrive between releases as a signed bundle. yonga contacts no registry server unless you tell it to, ships no signing key, and installs nothing it cannot verify. Trusting a publisher is your decision: copy their public key (a `*.pub` file) into `~/.config/yonga/registry-keys/` yourself. Then:

```bash
yonga registry update --url https://example.org/yonga/bundle.json
```

The signature is fetched from the same URL with `.sig` appended (HTTPS only); `--file BUNDLE` installs a bundle already on disk, with `--signature` when its signature is not `BUNDLE.sig`. yonga verifies the bundle completely, shows the signing key, the version, the expiry date and the rules it adds or replaces, and installs it after you confirm. A bundle with a bad signature, signed by a key you do not trust, expired, or older than the installed one is refused with exit 70.

```bash
yonga registry status    # which bundle is installed, and does it still verify?
yonga registry reset     # remove it and go back to the built-in registry
```

The installed bundle is verified again every time the registry loads. If it stops verifying (it expired, or was edited on disk), yonga ignores it and uses the built-in rules, and `registry status` says why and exits 2. Setting `registry_update_policy = "never"` in the configuration file refuses updates altogether.

Details: [USAGE.md, section 5](https://github.com/ooguz/yonga/blob/main/docs/USAGE.md#5-ask-the-registry-why) and [REGISTRY_SPEC.md, section 12.1](https://github.com/ooguz/yonga/blob/main/docs/REGISTRY_SPEC.md#121-remote-bundles-implemented).

## Upgrade yonga

Read the [changelog](https://github.com/ooguz/yonga/blob/main/CHANGELOG.md) before you upgrade. While yonga's major version is 0, the command line, the registry schema and the artifact manifest may change in any minor release.

=== "AppImage"

    Download the new AppImage from the [releases page](https://github.com/ooguz/yonga/releases), then run `self install` from it, replacing `X.Y.Z` with the new version. It asks before it replaces the old copy at `~/.local/bin/yonga`, and shows both versions where it knows them:

    ```bash
    chmod +x yonga-X.Y.Z-x86_64.AppImage
    ./yonga-X.Y.Z-x86_64.AppImage self install
    ```

=== "pip"

    Upgrade the package in the environment you installed it into:

    ```bash
    pip install --upgrade "yonga[openvino,export]"
    ```

    Keep the same extras you installed with (add `serve` if you use the HTTP server).

An upgrade does not touch your data. This is what happens to it:

- **Prepared artifacts** stay where they are. When a new release changes the default profile for a model, preparing that model again creates a **new** artifact next to the old one instead of replacing it; the preparation plan says so beforehand and names the `--profile` that would reuse the old one. Commands given a repository id then use the newest ready artifact. Remove the old one with `yonga remove` when you no longer need it.
- **The compile cache** keeps its entries. If the upgrade brings a new OpenVINO release, the entries built by the old one become `stale-runtime`: the new runtime compiles fresh blobs on first load, and `yonga cache prune` frees the old ones.
- **Managed export environments** stay and are listed by `yonga env list`. Remove the ones you no longer need with `yonga env remove`; yonga rebuilds an environment if a later export needs it.
- **Configuration and job logs** are kept unchanged.

`yonga doctor` after an upgrade confirms that the runtime and devices still work and reports how much of the compile cache is reclaimable.

## Where things live

| Path | Contents | Managed with |
|---|---|---|
| `~/.local/share/yonga/artifacts/` | Prepared artifacts | `yonga list`, `yonga remove` |
| `~/.local/share/yonga/logs/` | Job logs | `yonga logs` |
| `~/.local/share/yonga/benchmarks/` | Saved benchmark results | `yonga bench` |
| `~/.cache/yonga/compile/` | Compile cache | `yonga cache list`, `yonga cache prune` |
| `~/.cache/yonga/envs/` | Managed export environments | `yonga env list`, `yonga env remove` |
| `~/.cache/yonga/tmp/` | Managed temp directory | `yonga list --clean` |
| `~/.config/yonga/` | `config.toml`, `registry-overrides/`, `registry-keys/` | by hand |
| `~/.local/bin/yonga` | The AppImage copy | `yonga self install`, `yonga self uninstall` |

`yonga registry paths` prints the actual locations, including any you changed with `--data-dir`, `--cache-dir` or the configuration file. Downloaded source models stay in the normal Hugging Face cache; yonga never copies, modifies or deletes them.
