# Releasing yonga

A release is a version tag. Pushing `vX.Y.Z` runs `.github/workflows/release.yml`, which

1. builds the sdist and wheel and checks that the tag matches `pyproject.toml`;
2. builds `yonga-X.Y.Z-x86_64.AppImage` from that wheel with `tools/build_appimage.sh` and
   smoke-tests it;
3. publishes the sdist and wheel to PyPI through trusted publishing — **only while the repository
   is public**, because the sdist is the source code;
4. creates a GitHub release whose notes are that version's `CHANGELOG.md` section, with the sdist,
   the wheel, the AppImage and its `.sha256` attached.

A tag pushed while the repository is private gets its GitHub release and skips PyPI. After making
the repository public, publish that tag from the Actions tab: *Release* → *Run workflow* → tag
`vX.Y.Z` (or `gh workflow run release.yml -f tag=vX.Y.Z`). That run rebuilds the sdist and wheel
from the tag and publishes them; it does not touch the existing GitHub release.

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
2. **GitHub environment.** The repository has an environment named `pypi`. Adding yourself as a
   required reviewer there makes every publish wait for an explicit approval.
3. **Project URLs.** `[project.urls]` in `pyproject.toml` point at `github.com/ooguz/yonga`; PyPI
   displays them on the project page.

## Every release

CI covers lint, types, the unit suite on Python 3.11–3.14 without OpenVINO, the OpenVINO CPU suite
and a clean-install smoke test. It cannot cover the accelerators, because hosted runners have
neither an Intel GPU nor an NPU. Run those on real hardware first:

```bash
python tools/make_test_artifact.py && python tools/make_test_artifact.py --npu
pytest -m "intel_gpu"
pytest -m "intel_npu"
pytest -m "not network and not slow and not large_model"     # the full local suite
```

Then:

1. Move the `## [Unreleased]` entries in `CHANGELOG.md` under `## [X.Y.Z] - YYYY-MM-DD`.
   `python tools/release_notes.py X.Y.Z` must print them; the release fails without notes.
2. Set `version` in `pyproject.toml` and `FALLBACK_VERSION` in `src/yonga/__init__.py`.
3. Build and inspect locally: `python -m build && twine check --strict dist/*`, then install the
   wheel into a fresh venv and run `yonga --version` and `yonga registry validate`. Build the
   AppImage from the same wheel with `tools/build_appimage.sh --wheel dist/yonga-X.Y.Z-*.whl` and
   run its `doctor` and a `bench` on the GPU and the NPU.
4. Commit, then `git tag vX.Y.Z && git push origin main vX.Y.Z`.

A tag that does not match the package version fails the build job before anything is published.
PyPI never accepts the same version twice, so a mistake after publishing needs a new patch release.
