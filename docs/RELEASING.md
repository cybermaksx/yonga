# Releasing yonga

A release is a version tag. Pushing `vX.Y.Z` runs `.github/workflows/release.yml`, which builds the
sdist and wheel, checks that the tag matches `pyproject.toml`, publishes to PyPI through trusted
publishing, and creates a GitHub release whose notes are that version's `CHANGELOG.md` section.

## One-time setup

1. **PyPI project and trusted publisher.** On pypi.org, add a *pending* trusted publisher for the
   project name `yonga`: owner and repository of the GitHub repo, workflow `release.yml`,
   environment `pypi`. No API token is stored anywhere.
2. **GitHub environment.** Create an environment named `pypi` in the repository settings.
   Requiring a reviewer there makes every publish wait for an explicit approval.
3. **Project URLs.** `[project.urls]` in `pyproject.toml` must point at the real repository;
   PyPI displays them on the project page.

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
   wheel into a fresh venv and run `yonga --version` and `yonga registry validate`.
4. Commit, then `git tag vX.Y.Z && git push origin main vX.Y.Z`.

A tag that does not match the package version fails the build job before anything is published.
PyPI never accepts the same version twice, so a mistake after publishing needs a new patch release.
