"""Writing to the Hugging Face Hub: the only adapter that ever uploads anything.

Everything is one commit. A publication either lands whole or not at all, so
a failure part-way can never leave a repository with weights but no model
card, or a card describing files that are not there.

The token is resolved by ``huggingface_hub`` itself, from ``HF_TOKEN`` or the
stored login; this module never reads, logs or returns it. Hub error text is
redacted before it becomes part of a diagnostic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError

from yonga.domain.diagnostics import FailureCategory, YongaError, diagnostic
from yonga.domain.publication import PublisherIdentity
from yonga.util.redaction import redact_text

__all__ = ["HfPublisher"]


class HfPublisher:
    """Creates repositories and commits files through ``huggingface_hub``."""

    def __init__(self, api: HfApi | None = None) -> None:
        """Use ``api``, or a default :class:`HfApi` that finds the token itself."""
        self._api = api if api is not None else HfApi()

    def identity(self) -> PublisherIdentity | None:
        """The logged-in account, or ``None`` when there is no token at all."""
        try:
            info: dict[str, Any] = self._api.whoami()
        except LocalTokenNotFoundError:
            return None  # an OSError subclass, but "not logged in", not "offline"
        except HfHubHTTPError as error:
            if _status(error) == 401:
                return None
            raise _translate(error, "checking who you are logged in as") from error
        except OSError as error:
            raise _translate(error, "checking who you are logged in as") from error
        role = (((info.get("auth") or {}).get("accessToken") or {}).get("role") or "").lower()
        return PublisherIdentity(
            name=str(info.get("name") or ""),
            organizations=[
                str(org.get("name")) for org in info.get("orgs") or [] if org.get("name")
            ],
            can_write=True if role == "write" else False if role == "read" else None,
        )

    def repo_exists(self, repo_id: str) -> bool:
        """Whether the model repository already exists."""
        try:
            return bool(self._api.repo_exists(repo_id, repo_type="model"))
        except (HfHubHTTPError, OSError) as error:
            raise _translate(error, f"checking whether {repo_id} exists") from error

    def publish(
        self,
        repo_id: str,
        *,
        private: bool,
        files: Sequence[tuple[str, Path]],
        generated: Mapping[str, bytes],
        message: str,
    ) -> str:
        """Create the repository if needed and upload everything in one commit.

        Returns the commit's URL.
        """
        operations = [
            CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(local))
            for path_in_repo, local in files
        ]
        operations += [
            CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=payload)
            for path_in_repo, payload in generated.items()
        ]
        try:
            self._api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
            commit = self._api.create_commit(
                repo_id, operations, commit_message=message, repo_type="model"
            )
        except (HfHubHTTPError, OSError) as error:
            raise _translate(error, f"uploading to {repo_id}") from error
        return str(getattr(commit, "commit_url", "") or f"https://huggingface.co/{repo_id}")


def _status(error: BaseException) -> int | None:
    response = getattr(error, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def _translate(error: BaseException, operation: str) -> YongaError:
    status = _status(error)
    raw = redact_text(str(error))
    if status in (401, 403):
        return YongaError(
            diagnostic(
                "publish.hub.not_authorized",
                FailureCategory.GATED_MODEL_AUTH_REQUIRED,
                "The Hub refused the request",
                f"HTTP {status} while {operation}: the credentials cannot write there.",
                remediation=[
                    "Log in with a token that has write access: hf auth login",
                    "Check that the namespace is your account or an organisation you can write to.",
                ],
                raw_excerpt=raw,
            )
        )
    if status is None:
        return YongaError(
            diagnostic(
                "publish.hub.unreachable",
                FailureCategory.DOWNLOAD_FAILURE,
                "Cannot reach the Hugging Face Hub",
                f"The network request failed while {operation}.",
                remediation=["Check the network connection and any proxy settings, then retry."],
                raw_excerpt=raw,
            )
        )
    return YongaError(
        diagnostic(
            "publish.hub.http_error",
            FailureCategory.DOWNLOAD_FAILURE,
            "The Hub returned an error",
            f"HTTP {status} while {operation}.",
            remediation=["Retry in a moment; 5xx errors are usually transient."],
            raw_excerpt=raw,
        )
    )
