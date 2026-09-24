"""The Hub writer: one commit, and credentials problems reported as what they are."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError

from yonga.adapters.hf_publish import HfPublisher
from yonga.domain.diagnostics import YongaError


class FakeApi:
    def __init__(self, *, whoami: Any = None, error: BaseException | None = None) -> None:
        self._whoami = whoami
        self.error = error
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def whoami(self) -> Any:
        if isinstance(self._whoami, BaseException):
            raise self._whoami
        return self._whoami

    def repo_exists(self, repo_id: str, **kwargs: Any) -> bool:
        return False

    def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        self.calls.append(("create_repo", (repo_id,), kwargs))

    def create_commit(self, repo_id: str, operations: Any, **kwargs: Any) -> Any:
        if self.error is not None:
            raise self.error
        self.calls.append(("create_commit", (repo_id, list(operations)), kwargs))
        return type("Commit", (), {"commit_url": f"https://huggingface.co/{repo_id}/commit/1"})()


def http_error(status: int) -> HfHubHTTPError:
    request = httpx.Request("POST", "https://huggingface.co/api")
    return HfHubHTTPError("denied", response=httpx.Response(status, request=request))


def publisher(api: FakeApi) -> HfPublisher:
    return HfPublisher(api)  # type: ignore[arg-type]


def test_no_token_means_not_logged_in_not_offline() -> None:
    api = FakeApi(whoami=LocalTokenNotFoundError("no token"))

    assert publisher(api).identity() is None


def test_the_token_role_decides_whether_it_can_write() -> None:
    def identity(role: str) -> bool | None:
        info = {"name": "me", "orgs": [{"name": "team"}], "auth": {"accessToken": {"role": role}}}
        found = publisher(FakeApi(whoami=info)).identity()
        assert found is not None
        assert found.may_write_to("team")
        return found.can_write

    assert identity("write") is True
    assert identity("read") is False
    assert identity("fineGrained") is None


def test_everything_goes_up_in_one_commit(tmp_path: Path) -> None:
    weights = tmp_path / "model.bin"
    weights.write_bytes(b"\x00")
    api = FakeApi()

    url = publisher(api).publish(
        "me/model",
        private=True,
        files=[("model.bin", weights)],
        generated={"README.md": b"# card"},
        message="Upload",
    )

    (create, commit) = api.calls
    assert create[2]["private"] is True
    assert [op.path_in_repo for op in commit[1][1]] == ["model.bin", "README.md"]
    assert url == "https://huggingface.co/me/model/commit/1"


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (http_error(403), "publish.hub.not_authorized"),
        (http_error(500), "publish.hub.http_error"),
        (OSError("connection reset"), "publish.hub.unreachable"),
    ],
)
def test_upload_failures_are_typed(tmp_path: Path, error: BaseException, code: str) -> None:
    with pytest.raises(YongaError) as caught:
        publisher(FakeApi(error=error)).publish(
            "me/model", private=True, files=[], generated={"README.md": b"x"}, message="m"
        )
    assert caught.value.diagnostic.code == code
