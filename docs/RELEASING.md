# Releasing yonga

A release is a version tag. Pushing `vX.Y.Z` runs `.github/workflows/release.yml`, which

1. checks that the tag matches `pyproject.toml` and that `CHANGELOG.md` has a section for the
   version, then builds the sdist and wheel;
2. builds `yonga-X.Y.Z-x86_64.AppImage` from that wheel, checked against the sha256 the build job
   recorded, with `tools/build_appimage.sh`, and smoke-tests it from a directory holding a decoy
   `typer.py`;
3. once the AppImage has passed, publishes the sdist and wheel to PyPI through trusted publishing,
   after a reviewer approves the `pypi` environment — **only while the repository is public**,
   because the sdist is the source code;
4. creates a GitHub release whose notes are that version's `CHANGELOG.md` section, with the sdist,
   the wheel, the AppImage, its `.sha256` and its `.packages.txt` attached.

The AppImage's Python packages are resolved from PyPI when it is built, not pinned. The
`.packages.txt` asset (also inside the image, as `usr/share/yonga/packages.txt`) lists the exact
versions it bundles and the sha256 of the yonga wheel; the build log prints the same list.

A tag pushed while the repository is private gets its GitHub release and skips PyPI. After making
the repository public, publish that tag by running the workflow on the tag itself:
`gh workflow run release.yml --ref vX.Y.Z`, or from the Actions tab: *Release* → *Run workflow* →
*Use workflow from* → *Tags* → `vX.Y.Z`. That run rebuilds the sdist and wheel from the tag and
publishes them once approved; it builds no AppImage and does not touch the existing GitHub release.
A run started from a branch fails at once, and the `pypi` environment would refuse it anyway.

A manual run uses the `release.yml` stored in the tag, so this path works only for tags cut after
the workflow took its current form. `v0.2.0` carries the earlier workflow, which takes the tag as
an input and publishes without the AppImage gate. None of 0.1.0 to 0.4.0 is published to PyPI:
those were private releases, and 0.5.0 is the first version released there.

## One-time setup

1. **PyPI trusted publisher.** Signed in to pypi.org as `ooguz`, open *Your projects* →
   *Publishing* → *Add a new pending publisher* → GitHub, and enter:

   | Field | Value |
   |---|---|
   | PyPI project name | `yonga` |
   | Owner | `ooguz` |
   | Repository name | `yonga` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   The first successful publish creates the project and turns the pending publisher into a normal
   one. No API token is stored anywhere.
2. **GitHub environment `pypi`.** Required before the first publish. PyPI trusts any run of
   `release.yml` that names this environment, so the environment is what limits which runs those
   are: it must admit only `v*` tags and wait for a reviewer. As the only maintainer, leave
   *Prevent self-review* off, or no publish could ever be approved. After the repository is public:

   ```bash
   id=$(gh api users/ooguz --jq .id)
   gh api -X PUT repos/ooguz/yonga/environments/pypi --input - <<EOF
   {"wait_timer": 0, "prevent_self_review": false, "can_admins_bypass": false,
    "reviewers": [{"type": "User", "id": $id}],
    "deployment_branch_policy": {"protected_branches": false, "custom_branch_policies": true}}
   EOF
   gh api -X POST repos/ooguz/yonga/environments/pypi/deployment-branch-policies \
     -f name='v*' -f type=tag
   ```

   `gh api repos/ooguz/yonga/environments/pypi` should then show one `required_reviewers` rule and
   `custom_branch_policies: true`, and the second command's policy should be listed by
   `gh api repos/ooguz/yonga/environments/pypi/deployment-branch-policies`.
3. **Private vulnerability reporting**, which `SECURITY.md` points reporters at. After the
   repository is public: `gh api -X PUT repos/ooguz/yonga/private-vulnerability-reporting`.
4. **Project URLs.** `[project.urls]` in `pyproject.toml` point at `github.com/ooguz/yonga`; PyPI
   displays them on the project page.

The actions in `release.yml` are pinned to commit shas, with the release tag in a comment. To move
one, resolve the new tag to its commit (`gh api repos/OWNER/ACTION/git/ref/tags/TAG`, then
`gh api repos/OWNER/ACTION/git/tags/SHA` when the result is an annotated tag) and replace both.
Dependabot (`.github/dependabot.yml`) checks the workflows' actions weekly and opens one pull
request that moves the sha and the comment together; the manual steps are for moving one sooner.

## Every release

CI (`docs/TESTING.md` section 11) runs ruff, mypy on Python 3.11–3.14 without OpenVINO, the unit
and integration suites on 3.11–3.14 with the OpenVINO CPU plugin, the same suites without the
OpenVINO or export extras on 3.11 and 3.14, a package build with `twine check` and a clean-install
smoke test, and an AppImage build with its smoke test. It cannot cover the accelerators, because
hosted runners have neither an Intel GPU nor an NPU. Run those on real hardware first:

```bash
python tools/make_test_artifact.py && python tools/make_test_artifact.py --npu
pytest -m "intel_gpu"
pytest -m "intel_npu"
pytest -m "not network and not slow and not large_model"     # the full local suite
```

Then:

1. In `CHANGELOG.md`, move the `## [Unreleased]` entries under a new `## [X.Y.Z] - YYYY-MM-DD`
   heading and leave an empty `## [Unreleased]` above it. In the link block at the bottom, change
   `[Unreleased]:` to `https://github.com/ooguz/yonga/compare/vX.Y.Z...HEAD` and add
   `[X.Y.Z]: https://github.com/ooguz/yonga/releases/tag/vX.Y.Z` above the previous version's
   line. `python tools/release_notes.py X.Y.Z` must print the section; the release fails before
   anything is published without it.
2. Set `version` in `pyproject.toml` and `FALLBACK_VERSION` in `src/yonga/__init__.py`. Then look
   for version strings left behind: `git grep -nF 'A.B.C' -- README.md docs/`, with `A.B.C` the
   previous release. An install command, a download name or a sentence about the current version
   that still names it is stale; a measurement record, such as the environment table in
   `docs/PROFILE_COMPARISON.md`, keeps the version it was measured with. `README.md` is the PyPI
   long description (`readme` in `pyproject.toml`): PyPI shows it as it is in the release commit,
   and only a new release changes it there.
3. Commit the changelog cut, the version bump and any stale strings step 2 found. Everything
   below builds, tests and tags this commit.
4. Build and inspect locally from a clean worktree of that commit, so the files match the ones CI
   builds: `git worktree add ../yonga-release HEAD`, then in `../yonga-release` run
   `python -m build && twine check --strict dist/*`, install the wheel into a fresh venv and run
   `yonga --version` and `yonga registry validate`. Build the AppImage from the same wheel with
   `tools/build_appimage.sh --wheel dist/yonga-X.Y.Z-*.whl` and run its `doctor` and a `bench` on
   the GPU and the NPU. These local files are for inspection only: never upload them, because the
   release workflow is the only path to PyPI. If something needs fixing, commit the fix on `main`,
   remove the worktree and start this step again.
5. `git push origin main`, and wait for CI to pass on that commit (`gh run watch`). The release
   workflow runs no tests of its own.
6. `git tag vX.Y.Z && git push origin vX.Y.Z`, then approve the `pypi` deployment when the run
   asks for it.
7. Remove the worktree: `git worktree remove ../yonga-release`. The build outputs in it are
   git-ignored and do not block the removal.

A tag that does not match the package version, or a version with no changelog section, fails the
build job before anything is published. PyPI never accepts the same version twice, so a mistake
after publishing needs a new patch release.
