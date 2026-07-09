"""Hermetic tests for POST /api/kb/telemetry/whispers (whisper telemetry sink).

Test matrix:
  (a) 401 without Bearer in jwt mode
  (b) flush of N rows returns upserted==N and records N execute() calls each
      carrying ON CONFLICT SQL, an int-bound consumed param, and a json.dumps'd
      trigger_context TEXT.
  (c) empty rows list returns upserted==0 with zero execute() calls.
  (d) idempotent re-flush of the same (session_id, surface, map_id) — first
      with consumed=false, then with consumed=true — leaves ONE row in the
      in-memory store with consumed==1 (no duplicate; ON CONFLICT DO UPDATE).

The conftest StatefulFakeDbPool.execute() (conftest.py lines 606-644) returns
'OK' for unrecognized SQL and records nothing — so this module installs a
local recording fake pool whose execute() records (sql, args) AND emulates
the composite-key upsert.
"""

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

import kb_service.database as database
import kb_service.main as main_module
import kb_service.routes.telemetry_routes as telemetry_routes
from kb_service.auth import get_current_user
from kb_service.main import app
from tests.conftest import FakeKnowledgeBase, fake_user

# ─── recording fake pool ─────────────────────────────────────────────────────


class _RecordingTelemetryPool:
    """A minimal pool that records execute() calls AND emulates the upsert.

    The conftest StatefulFakeDbPool inherits from FakeDbPool whose ``acquire``
    raises NotImplementedError, so the telemetry route (which uses
    ``async with pool.acquire() as conn``) needs its own fake.  This pool's
    ``acquire()`` returns an async context manager yielding ``self`` so the
    route's ``await conn.execute(...)`` lands on the same recorder.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        # Emulated table: keyed by (session_id, surface, map_id) → row dict.
        self.rows: dict[tuple[str, str, str], dict[str, Any]] = {}

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator["_RecordingTelemetryPool"]:
        yield self

    async def execute(self, sql: str, *args: Any) -> str:
        self.calls.append((sql, args))
        # Emulate the telemetry upsert (NOT a generic SQL emulator).
        if "INSERT INTO whisper_telemetry" in sql and "ON CONFLICT" in sql:
            (
                session_id,
                host,
                surface,
                map_id,
                source_kb,
                cwd_project,
                trigger_context,
                emitted_ts,
                consumed,
                consumed_ts,
                build_engine,
                flushed_at,
            ) = args
            key = (str(session_id), str(surface), str(map_id))
            existing = self.rows.get(key)
            if existing is None:
                self.rows[key] = {
                    "session_id": session_id,
                    "host": host,
                    "surface": surface,
                    "map_id": map_id,
                    "source_kb": source_kb,
                    "cwd_project": cwd_project,
                    "trigger_context": trigger_context,
                    "emitted_ts": emitted_ts,
                    "consumed": consumed,
                    "consumed_ts": consumed_ts,
                    "build_engine": build_engine,
                    "flushed_at": flushed_at,
                }
            else:
                # ON CONFLICT DO UPDATE — only the five mutable columns win;
                # emitted_ts / source_kb / host / cwd_project are NOT overwritten.
                existing["consumed"] = consumed
                existing["consumed_ts"] = consumed_ts
                existing["build_engine"] = build_engine
                existing["flushed_at"] = flushed_at
                existing["trigger_context"] = trigger_context
        return "OK"


# ─── fixture: dedicated telemetry client ─────────────────────────────────────


@pytest.fixture
def telemetry_pool() -> _RecordingTelemetryPool:
    """A fresh recording pool per test (so test counts are independent)."""
    return _RecordingTelemetryPool()


@pytest.fixture
def telemetry_client(
    monkeypatch: pytest.MonkeyPatch,
    telemetry_pool: _RecordingTelemetryPool,
) -> Iterator[TestClient]:
    """TestClient whose service-auth pool is the recording telemetry pool.

    Mirrors conftest.client's lifespan patches but swaps the pool for
    ``telemetry_pool`` so the telemetry route's ``pool.acquire()`` lands on
    the recorder.  ``init_db`` / ``close_db`` / ``create_postgres`` are
    no-ops here just like in conftest.client.
    """
    fake_kb = FakeKnowledgeBase(results=[], filtered_count=0)

    async def _fake_init_db() -> None:
        return None

    async def _fake_close_db() -> None:
        return None

    async def _fake_create_postgres(*args: Any, **kwargs: Any) -> FakeKnowledgeBase:
        return fake_kb

    async def _fake_get_db() -> _RecordingTelemetryPool:
        return telemetry_pool

    monkeypatch.setenv("KB_DATABASE_URL", "postgresql://test/test")
    monkeypatch.setattr(main_module, "init_db", _fake_init_db)
    monkeypatch.setattr(main_module, "close_db", _fake_close_db)
    monkeypatch.setattr(main_module, "create_postgres", _fake_create_postgres)
    # telemetry_routes binds get_db at import time — must patch its module too.
    monkeypatch.setattr(database, "get_db", _fake_get_db)
    monkeypatch.setattr(telemetry_routes, "get_db", _fake_get_db)

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


# ─── helpers ─────────────────────────────────────────────────────────────────


def _row(
    *,
    session_id: str = "s1",
    surface: str = "roster",
    map_id: str = "kb-00001",
    source_kb: str = "personal",
    cwd_project: str | None = "my-project",
    trigger_context: dict[str, Any] | None = None,
    emitted_ts: str = "2026-06-17T08:00:00+00:00",
    consumed: bool = False,
    consumed_ts: str | None = None,
    build_engine: str | None = None,
    host: str = "test-host",
) -> dict[str, Any]:
    """Build a telemetry row JSON dict suitable for the request body."""
    return {
        "session_id": session_id,
        "host": host,
        "surface": surface,
        "map_id": map_id,
        "source_kb": source_kb,
        "cwd_project": cwd_project,
        "trigger_context": trigger_context if trigger_context is not None else {},
        "emitted_ts": emitted_ts,
        "consumed": consumed,
        "consumed_ts": consumed_ts,
        "build_engine": build_engine,
    }


# ─── (a) 401 without Bearer in jwt mode ──────────────────────────────────────


def test_flush_requires_auth(telemetry_client: TestClient) -> None:
    """POST without Bearer credential in jwt mode returns 401."""
    resp = telemetry_client.post("/api/kb/telemetry/whispers", json={"rows": []})
    assert resp.status_code == 401


# ─── (b) flush of N rows ─────────────────────────────────────────────────────


def test_flush_n_rows_returns_n_and_records_executes(
    telemetry_client: TestClient,
    telemetry_pool: _RecordingTelemetryPool,
) -> None:
    """N rows → upserted==N, N execute() calls, each ON CONFLICT + int + json."""
    app.dependency_overrides[get_current_user] = fake_user

    rows = [
        _row(
            map_id="kb-00001",
            surface="roster",
            trigger_context={"cwd_project": "alpha"},
            consumed=False,
        ),
        _row(
            map_id="kb-00002",
            surface="listener",
            trigger_context={
                "operating": ["mcp:personal-kb"],
                "cwd_project": "alpha",
                "excerpt_hash": "abc123",
            },
            consumed=True,
            consumed_ts="2026-06-17T08:01:00+00:00",
            build_engine="claude-code",
        ),
        _row(
            map_id="kb-00003",
            surface="roster",
            trigger_context={"cwd_project": "alpha"},
            consumed=False,
        ),
    ]

    resp = telemetry_client.post("/api/kb/telemetry/whispers", json={"rows": rows})

    assert resp.status_code == 200
    assert resp.json() == {"upserted": 3}

    assert len(telemetry_pool.calls) == 3

    # Every call must carry the ON CONFLICT SQL and the int-bound consumed param
    # and the json.dumps'd trigger_context TEXT.
    for i, (sql, args) in enumerate(telemetry_pool.calls):
        assert "ON CONFLICT" in sql, f"call {i}: missing ON CONFLICT"
        assert "whisper_telemetry" in sql

        # 12 positional args matching the INSERT column order.
        assert len(args) == 12

        # Per the route: $7 trigger_context, $9 consumed.
        trigger_context_text = args[6]
        consumed_arg = args[8]

        assert isinstance(trigger_context_text, str), (
            f"call {i}: trigger_context must be a TEXT json string"
        )
        # And must round-trip through json.loads to the original dict.
        parsed = json.loads(trigger_context_text)
        assert parsed == rows[i]["trigger_context"]

        type_name = type(consumed_arg).__name__
        assert isinstance(consumed_arg, int), (
            f"call {i}: consumed must be bound as int (got {type_name})"
        )
        assert consumed_arg == int(bool(rows[i]["consumed"]))


# ─── (c) empty rows list ─────────────────────────────────────────────────────


def test_flush_empty_rows_returns_zero_no_executes(
    telemetry_client: TestClient,
    telemetry_pool: _RecordingTelemetryPool,
) -> None:
    """Empty batch → upserted==0, zero execute() calls (no connection acquired)."""
    app.dependency_overrides[get_current_user] = fake_user
    resp = telemetry_client.post("/api/kb/telemetry/whispers", json={"rows": []})
    assert resp.status_code == 200
    assert resp.json() == {"upserted": 0}
    assert telemetry_pool.calls == []


# ─── (d) idempotent re-flush ────────────────────────────────────────────────


def test_idempotent_reflush_collapses_via_on_conflict(
    telemetry_client: TestClient,
    telemetry_pool: _RecordingTelemetryPool,
) -> None:
    """Re-flush of the same (session_id, surface, map_id) collapses to one row.

    First with consumed=false, then consumed=true.  The recording pool's
    composite-key emulation proves the route is using ON CONFLICT DO UPDATE
    rather than blindly inserting duplicates.
    """
    app.dependency_overrides[get_current_user] = fake_user

    # First flush: consumed=false.
    resp1 = telemetry_client.post(
        "/api/kb/telemetry/whispers",
        json={
            "rows": [
                _row(
                    session_id="sX",
                    surface="roster",
                    map_id="kb-00099",
                    consumed=False,
                )
            ]
        },
    )
    assert resp1.status_code == 200
    assert resp1.json() == {"upserted": 1}

    # Second flush: same (session_id, surface, map_id), consumed=true.
    resp2 = telemetry_client.post(
        "/api/kb/telemetry/whispers",
        json={
            "rows": [
                _row(
                    session_id="sX",
                    surface="roster",
                    map_id="kb-00099",
                    consumed=True,
                    consumed_ts="2026-06-17T09:00:00+00:00",
                    build_engine="claude-code",
                )
            ]
        },
    )
    assert resp2.status_code == 200
    # The route reports rows-accepted, NOT DB-affected rowcount.
    assert resp2.json() == {"upserted": 1}

    # The pool emulates the upsert into rows[(session_id, surface, map_id)].
    key = ("sX", "roster", "kb-00099")
    assert list(telemetry_pool.rows.keys()) == [key], (
        "ON CONFLICT must collapse the two flushes onto a single row"
    )
    final = telemetry_pool.rows[key]
    assert final["consumed"] == 1, "second flush's consumed=true must win via DO UPDATE"
    assert final["consumed_ts"] == "2026-06-17T09:00:00+00:00"
    assert final["build_engine"] == "claude-code"


# ─── rows carrying the new hook-side ``pointers`` field don't 500 ───────────


def test_flush_tolerates_extra_pointers_field_on_row(
    telemetry_client: TestClient,
    telemetry_pool: _RecordingTelemetryPool,
) -> None:
    """A row with an unknown ``pointers`` field must NOT 500 the flush.

    Rollout order: an upgraded personal-kb-hook now appends a top-level
    ``pointers`` list to every roster jsonl row so :func:`mark_consumed`
    can chain-credit map → detail fetches (GTD 88441f9c). The service
    ``WhisperTelemetryRow`` model has no ``pointers`` field, but its
    default pydantic-v2 config (``extra='ignore'``) silently drops
    unknown keys — so a pre-pointer-model service accepting a pointer-
    carrying flush is a no-op-safe schema evolution. This test locks
    that in and doubles as the AC's explicit verification.
    """
    app.dependency_overrides[get_current_user] = fake_user

    row = _row(map_id="kb-00042", surface="roster", consumed=False)
    # Hook-side row-shape addition: a top-level ``pointers`` list.
    row["pointers"] = ["kb-00043", "kb-00044"]
    # Also exercise the ``consumed_via`` piggyback path — it lives inside
    # ``trigger_context`` (a ``dict[str, Any]`` on the wire), so it flows
    # through to the DB inside the json.dumps'd TEXT with zero schema change.
    row["trigger_context"] = {"cwd_project": "alpha", "consumed_via": "pointer"}

    resp = telemetry_client.post("/api/kb/telemetry/whispers", json={"rows": [row]})
    assert resp.status_code == 200
    assert resp.json() == {"upserted": 1}

    # The row was upserted; the trigger_context TEXT round-trips including
    # the piggybacked ``consumed_via`` key.
    assert len(telemetry_pool.calls) == 1
    _sql, args = telemetry_pool.calls[0]
    trigger_context_text = args[6]
    assert isinstance(trigger_context_text, str)
    parsed = json.loads(trigger_context_text)
    assert parsed == {"cwd_project": "alpha", "consumed_via": "pointer"}


# ─── endpoint is mounted ─────────────────────────────────────────────────────


def test_telemetry_endpoint_mounted() -> None:
    """POST /api/kb/telemetry/whispers is registered in the FastAPI app."""
    paths = {route.path for route in app.routes}  # type: ignore[attr-defined]
    assert "/api/kb/telemetry/whispers" in paths
