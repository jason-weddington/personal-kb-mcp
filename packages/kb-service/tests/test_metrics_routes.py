"""GET /api/kb/metrics/repeat-rate."""

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import kb_service.database as database
import kb_service.repeat_rate
from kb_service.main import app
from kb_service.routes import metrics_routes
from tests.repeat_rate_fixtures import insert_rows_sqlite3

_SCRUB_ENV = (
    "KB_SERVICE_DATABASE_URL",
    "KB_DATABASE_URL",
    "ANTHROPIC_API_KEY",
    "KB_AWS_PROFILE",
)
URL = "/api/kb/metrics/repeat-rate"


@pytest.fixture
def local_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """The real app in local no-auth mode, seeded with the fixture rows."""
    for var in _SCRUB_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.setenv("KB_OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("KB_OLLAMA_TIMEOUT", "0.5")
    monkeypatch.setenv("KB_EXTRACTION_PROVIDER", "ollama")
    monkeypatch.setenv("KB_QUERY_PROVIDER", "ollama")
    monkeypatch.setattr(database, "_pool", None)
    with TestClient(app) as client:
        insert_rows_sqlite3(database.sqlite_service_db_path())
        monkeypatch.setattr(
            metrics_routes, "_utcnow", lambda: datetime(2026, 10, 7, 12, tzinfo=UTC)
        )
        yield client
    app.dependency_overrides.clear()


def test_requires_auth(client: TestClient) -> None:
    assert client.get(URL).status_code == 401


def test_repeat_rate(
    local_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="kb_service.repeat_rate"):
        r = local_client.get(URL, params={"weeks": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["failure_rows"] == 11 and body["resolutions_loaded"] == 0
    assert body["diagnostics"]["resolutions_scanned"] == 0
    assert body["diagnostics"]["tripwires"] == []
    assert body["weeks"][0]["week"] == "2026-W40"
    assert body["weeks"][0]["sessions"] == 2
    assert body["weeks"][1] == {
        "week": "2026-W41",
        "week_start": "2026-10-05",
        "sessions": 6,
        "repeat_sessions": 2,
        "repeat_rate": 0.3333,
        "distinct_cue_keys": 6,
        "pairs": 6,
        "repeat_pairs": 2,
        "aggregate_rate": 0.3333,
        "covered_sessions": 0,
        "covered_repeat_sessions": 0,
        "covered_repeat_rate": None,
    }
    lines = [
        rec.getMessage()
        for rec in caplog.records
        if all(
            s in rec.getMessage()
            for s in (
                "repeat_rate weeks=2",
                "failure_rows_scanned=11",
                "sessions=8",
                "tripwires=none",
                "fetch_ms=",
            )
        )
    ]
    assert len(lines) == 1


def test_project_filter(local_client: TestClient) -> None:
    body = local_client.get(URL, params={"weeks": 2, "project": "p"}).json()
    assert body["project"] == "p" and body["failure_rows"] == 9
    assert body["weeks"][1]["sessions"] == 4
    empty = local_client.get(URL, params={"weeks": 2, "project": ""}).json()
    assert empty["project"] is None and empty["failure_rows"] == 11


@pytest.mark.parametrize(
    "params",
    [{"weeks": 0}, {"weeks": 105}, {"min_gap_hours": -1}, {"min_gap_hours": 721}],
)
def test_bad_params(local_client: TestClient, params: dict[str, Any]) -> None:
    assert local_client.get(URL, params=params).status_code == 422


def test_failure_is_500_not_zeros(
    local_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(kb_service.repeat_rate, "fetch_failure_rows", boom)
    with pytest.raises(RuntimeError, match="boom"):
        local_client.get(URL, params={"weeks": 2})
