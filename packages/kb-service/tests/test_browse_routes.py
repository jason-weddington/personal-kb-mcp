"""Hermetic tests for browse_routes: kb-core browse functions are monkeypatched."""

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from kb_core.browse import (
    BrowsePage,
    BrowseParamError,
    BrowseRow,
    ProjectNotFound,
    ProjectPage,
    ProjectSummary,
)

import kb_service.routes.browse_routes as browse_routes
from kb_service.auth import get_current_user
from kb_service.main import app
from tests.conftest import FakeKnowledgeBase, fake_user

LOGGER = "kb_service.routes.browse_routes"

ENTRY_KEYS = {
    "id",
    "short_title",
    "long_title",
    "entry_type",
    "project_ref",
    "created_at",
    "updated_at",
    "last_accessed",
    "expires_at",
    "confidence_level",
    "effective_confidence",
    "half_life_days",
    "anchor_age_days",
    "expired",
    "signal_value",
}


def _row(entry_id: str = "kb-90001") -> BrowseRow:
    ts = datetime(2026, 6, 30, tzinfo=UTC)
    return BrowseRow(
        id=entry_id,
        short_title="Alpha fact",
        long_title="Alpha fact (long)",
        entry_type="factual_reference",
        project_ref="p-alpha",
        created_at=ts,
        updated_at=ts,
        last_accessed=None,
        expires_at=None,
        confidence_level=1.0,
        effective_confidence=0.99,
        half_life_days=90.0,
        anchor_age_days=1.0,
        expired=False,
    )


def _page() -> BrowsePage:
    return BrowsePage(items=[_row()], next_cursor="25", total=30, scanned=31, skipped=1)


class Recorder:
    """Async fake for a kb-core browse function that records its calls."""

    def __init__(self, result: Any = None, raises: Exception | None = None) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.result = result if result is not None else _page()
        self.raises = raises

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        if self.raises is not None:
            raise self.raises
        return self.result


@pytest.fixture
def authed(client: TestClient) -> Iterator[TestClient]:
    """The shared client with the fake user authenticated."""
    app.dependency_overrides[get_current_user] = fake_user
    yield client


def _patch(monkeypatch: pytest.MonkeyPatch, name: str, fake: Recorder) -> Recorder:
    monkeypatch.setattr(browse_routes, name, fake)
    return fake


PATHS = [
    "/api/kb/browse/recent",
    "/api/kb/browse/top",
    "/api/kb/browse/stale",
    "/api/kb/browse/projects",
    "/api/kb/browse/project/p-alpha",
]


@pytest.mark.parametrize("path", PATHS)
def test_requires_auth(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 401


def test_recent_defaults_and_serialization(
    authed: TestClient, fake_kb: FakeKnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _patch(monkeypatch, "browse_recent", Recorder())
    resp = authed.get("/api/kb/browse/recent")
    assert resp.status_code == 200
    args, kwargs = fake.calls[0]
    assert args == (fake_kb.db,)
    assert kwargs == {
        "sort": "created",
        "entry_type": None,
        "limit": 25,
        "cursor": None,
    }
    body = resp.json()
    assert set(body["items"][0]) == ENTRY_KEYS
    assert body["items"][0]["created_at"] == "2026-06-30T00:00:00+00:00"
    assert body["next_cursor"] == "25"
    assert body["total"] == 30
    assert body["skipped"] == 1
    assert body["signal"] is None


def test_recent_params_pass_through(
    authed: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _patch(monkeypatch, "browse_recent", Recorder())
    resp = authed.get(
        "/api/kb/browse/recent?sort=updated&entry_type=decision&limit=10&cursor=10"
    )
    assert resp.status_code == 200
    assert fake.calls[0][1] == {
        "sort": "updated",
        "entry_type": "decision",
        "limit": 10,
        "cursor": "10",
    }


@pytest.mark.parametrize(
    "query", ["sort=bogus", "entry_type=bogus", "limit=0", "limit=101"]
)
def test_recent_validation_422(
    authed: TestClient, monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    fake = _patch(monkeypatch, "browse_recent", Recorder())
    assert authed.get(f"/api/kb/browse/recent?{query}").status_code == 422
    assert fake.calls == []


@pytest.mark.parametrize(
    ("name", "path"),
    [
        ("browse_recent", "/api/kb/browse/recent"),
        ("browse_projects", "/api/kb/browse/projects"),
    ],
)
def test_param_error_is_422(
    authed: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    name: str,
    path: str,
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    _patch(monkeypatch, name, Recorder(raises=BrowseParamError("invalid cursor")))
    resp = authed.get(f"{path}?cursor=abc")
    assert resp.status_code == 422
    assert resp.json()["detail"] == "invalid cursor"
    assert any("outcome=bad_param" in r.getMessage() for r in caplog.records)


def test_plain_value_error_is_not_422(
    authed: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    _patch(monkeypatch, "browse_recent", Recorder(raises=ValueError("boom")))
    with pytest.raises(ValueError, match="boom"):
        authed.get("/api/kb/browse/recent")
    errors = [r for r in caplog.records if "outcome=error" in r.getMessage()]
    assert len(errors) == 1
    assert errors[0].levelno == logging.WARNING


def test_top_echoes_signal(authed: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _patch(monkeypatch, "browse_top", Recorder())
    assert authed.get("/api/kb/browse/top").json()["signal"] == "degree"
    assert authed.get("/api/kb/browse/top?signal=read").json()["signal"] == "read"
    assert fake.calls[1][1]["signal"] == "read"


def test_stale_kind(authed: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _patch(monkeypatch, "browse_stale", Recorder())
    assert authed.get("/api/kb/browse/stale?kind=expired").status_code == 200
    assert fake.calls[0][1]["kind"] == "expired"


def test_projects(authed: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    page = ProjectPage(
        items=[
            ProjectSummary(
                project_ref="p-gamma",
                entry_count=2,
                counts_by_type={"factual_reference": 1, "decision": 1},
                last_activity=datetime(2026, 6, 10, tzinfo=UTC),
            )
        ],
        next_cursor=None,
        total=1,
    )
    fake = _patch(monkeypatch, "browse_projects", Recorder(result=page))
    resp = authed.get("/api/kb/browse/projects?q=ga")
    assert resp.status_code == 200
    assert fake.calls[0][1] == {"q": "ga", "limit": 24, "cursor": None}
    item = resp.json()["items"][0]
    assert item["counts_by_type"] == {"factual_reference": 1, "decision": 1}
    assert item["last_activity"] == "2026-06-10T00:00:00+00:00"


def test_project_params(authed: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _patch(monkeypatch, "browse_project", Recorder())
    resp = authed.get("/api/kb/browse/project/p-alpha?sort=confidence&q=x")
    assert resp.status_code == 200
    args, kwargs = fake.calls[0]
    assert args[1] == "p-alpha"
    assert kwargs["sort"] == "confidence"
    assert kwargs["q"] == "x"


def test_project_ref_with_slash(
    authed: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _patch(monkeypatch, "browse_project", Recorder())
    assert authed.get("/api/kb/browse/project/team%2Fsub").status_code == 200
    assert fake.calls[0][0][1] == "team/sub"


def test_project_not_found(
    authed: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    _patch(monkeypatch, "browse_project", Recorder(raises=ProjectNotFound("p-none")))
    resp = authed.get("/api/kb/browse/project/p-none")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Unknown project: p-none"
    assert any("outcome=not_found" in r.getMessage() for r in caplog.records)


def test_success_log_line(
    authed: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    _patch(monkeypatch, "browse_recent", Recorder())
    authed.get("/api/kb/browse/recent")
    lines = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert len(lines) == 1
    for part in (
        "browse-route",
        "view=recent",
        "mode=created",
        "outcome=ok",
        "returned=",
        "duration_ms=",
    ):
        assert part in lines[0]
