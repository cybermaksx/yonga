# Security policy

## Reporting a vulnerability

Report it privately through a GitHub security advisory:
<https://github.com/ooguz/yonga/security/advisories/new>. Please do not open a public issue, pull
request or discussion for it.

Say what is affected (version, install method: pip or AppImage), how to reproduce it, and what an
attacker gains. yonga redacts tokens in its logs, and a bug-report bundle also anonymises the home
directory, hostname, login name and Hugging Face account name, but read a log or bundle before
attaching it all the same.

The answer comes in the advisory. A confirmed problem is fixed in a new patch release, and the
advisory is published with it, crediting the reporter unless they prefer otherwise.

## Supported versions

Only the newest release receives fixes. While the major version is 0, a fix is not backported to
an earlier minor release.

## Scope

In scope is yonga's own code and what it ships: the CLI, the `serve` HTTP API, the handling of
Hugging Face tokens, the remote-code confirmation (`--trust-remote-code`), registry data and the
verification of signed registry bundles, the bug-report bundle's redaction, and the release
artifacts (the wheel, the sdist and the AppImage).

Out of scope, and best reported upstream: vulnerabilities in OpenVINO, OpenVINO GenAI,
optimum-intel, Transformers, PyTorch or the other packages yonga installs, and in the models
themselves. A model repository's own code runs only after `--trust-remote-code`, which is the
user's decision to trust it.

`yonga serve` binds `127.0.0.1` and needs no key there by default. Exposing it on another
address without setting `YONGA_API_KEY` is a configuration choice that yonga warns about, not a
vulnerability.
