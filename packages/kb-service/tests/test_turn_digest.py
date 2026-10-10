"""turn_digest helpers: mode, redaction, anomalies, storage, prune."""

import json
import logging
import typing
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import kb_service.database as database
from kb_service import turn_digest
from kb_service.db_sqlite import SqlitePool
from kb_service.models import (
    StoredTurnDigest,
    TurnAssistantTextItem,
    TurnDigestRequest,
    TurnToolCallItem,
    TurnToolResultItem,
)
from kb_service.turn_digest import (
    get_session_turn_digests,
    insert_turn_digest,
    list_pending_turn_digests,
    mark_turn_digests_processed,
    prune_turn_events,
    redact_turn_digest,
    surprise_capture_mode,
    turn_digest_anomalies,
)

_SCRUB_ENV = (
    "KB_SERVICE_DATABASE_URL",
    "KB_DATABASE_URL",
    "ANTHROPIC_API_KEY",
    "KB_AWS_PROFILE",
)
_RECEIVED = "2026-10-09T12:00:00+00:00"


@pytest.fixture
def local_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    for var in _SCRUB_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("KB_SURPRISE_CAPTURE", raising=False)
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.setattr(database, "_pool", None)
    yield tmp_path / "service.db"


@pytest.fixture
async def pool(local_env: Path) -> AsyncIterator[SqlitePool]:
    await database.init_db()
    db = await database.get_db()
    assert isinstance(db, SqlitePool)
    yield db
    await database.close_db()


def _body(session: str = "s1", turn: int = 0, **kw: Any) -> TurnDigestRequest:
    data: dict[str, Any] = {
        "event_id": f"{session}:{turn}",
        "session_id": session,
        "turn_index": turn,
        "user_prompt": "hi",
        "items": [{"kind": "assistant_text", "text": "ok"}],
        "final_message": "done",
    }
    data.update(kw)
    return TurnDigestRequest.model_validate(data)


async def _insert(pool: SqlitePool, body: TurnDigestRequest, received: str = _RECEIVED):
    return await insert_turn_digest(
        pool, body, capture_mode="shadow", redactions=[], received_ts=received
    )


async def test_get_session_ordering(pool: SqlitePool) -> None:
    for t in (2, 0, 1):
        await _insert(pool, _body("s1", t))
    await _insert(pool, _body("s2", 0))
    rows = await get_session_turn_digests(pool, "s1")
    assert [r.turn_index for r in rows] == [0, 1, 2]
    assert all(isinstance(r, StoredTurnDigest) for r in rows)
    assert isinstance(rows[0].items[0], TurnAssistantTextItem)


async def test_pending_mark_and_claim(pool: SqlitePool) -> None:
    for t in (2, 0, 1):
        await _insert(pool, _body("s1", t))
    await _insert(pool, _body("s2", 0))
    assert len(await list_pending_turn_digests(pool)) == 4
    first = await list_pending_turn_digests(pool, limit=1)
    assert [r.event_id for r in first] == ["s1:2"]
    assert await mark_turn_digests_processed(pool, ["s1:0", "s1:1"], "t") == 2
    assert len(await list_pending_turn_digests(pool)) == 2
    assert await mark_turn_digests_processed(pool, ["s1:0", "s1:1"], "t") == 0
    assert await mark_turn_digests_processed(pool, [], "t") == 0


async def test_prune_strict_cutoff(pool: SqlitePool) -> None:
    now = datetime(2026, 10, 9, 12, 0, 0, tzinfo=UTC)
    for turn, days in enumerate((31, 30, 1)):
        received = (now - timedelta(days=days)).isoformat(timespec="seconds")
        await _insert(pool, _body("s1", turn), received)
    assert await prune_turn_events(pool, now=now) == 1
    remaining = await get_session_turn_digests(pool, "s1")
    assert [r.turn_index for r in remaining] == [1, 2]


async def test_retruncation_after_redaction(
    pool: SqlitePool, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    excerpt = 'pwd="hunter2hunter2"\n' + "x" * 1479
    assert len(excerpt) == 1500
    body = _body(
        items=[
            {
                "kind": "tool_call",
                "tool_use_id": "t1",
                "tool": "Bash",
                "target_class": "ls",
            },
            {
                "kind": "tool_result",
                "tool_use_id": "t1",
                "is_error": False,
                "excerpt": excerpt,
            },
        ]
    )
    result = redact_turn_digest(body)
    assert result is not None
    red, types = result
    item = red.items[1]
    assert isinstance(item, TurnToolResultItem)
    assert item.excerpt == "[REDACTED:Secret Keyword]\n" + "x" * 1474
    assert types == ["Secret Keyword"]
    assert (
        "turn_event retruncated event_id=s1:0 fields=1 chars_dropped=5" in caplog.text
    )
    await _insert(pool, red)
    assert len(await get_session_turn_digests(pool, "s1")) == 1


async def test_malformed_row_skipped(
    pool: SqlitePool, caplog: pytest.LogCaptureFixture
) -> None:
    await _insert(pool, _body("s1", 0))
    await pool.execute(
        "UPDATE turn_events SET items = 'not json' WHERE event_id = $1", "s1:0"
    )
    caplog.set_level(logging.WARNING)
    assert await get_session_turn_digests(pool, "s1") == []
    msgs = [
        r
        for r in caplog.records
        if "turn_event malformed event_id=s1:0" in r.getMessage()
    ]
    assert len(msgs) == 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "off"), ("", "off"), ("yes", "off"), (" ON ", "on"), ("shadow", "shadow")],
)
def test_surprise_capture_mode(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: str
) -> None:
    if value is None:
        monkeypatch.delenv("KB_SURPRISE_CAPTURE", raising=False)
    else:
        monkeypatch.setenv("KB_SURPRISE_CAPTURE", value)
    assert surprise_capture_mode() == expected


async def test_project_default_empty(pool: SqlitePool) -> None:
    await _insert(pool, _body())
    assert (await get_session_turn_digests(pool, "s1"))[0].project == ""


def test_redaction_field_partition() -> None:
    result = redact_turn_digest(
        _body(
            user_prompt="curl https://user:s3cretpass@example.com/x",
            items=[
                {
                    "kind": "tool_call",
                    "tool_use_id": "toolu_1",
                    "tool": "Bash",
                    "target": "ls",
                    "target_class": 'password = "hunter2hunter2"',
                },
                {
                    "kind": "tool_result",
                    "tool_use_id": "toolu_1",
                    "is_error": True,
                    "excerpt": 'password = "hunter2hunter2"',
                },
            ],
        )
    )
    assert result is not None
    red, types = result
    assert types == ["Basic Auth Credentials", "Secret Keyword"]
    call = red.items[0]
    assert isinstance(call, TurnToolCallItem)
    assert call.tool == "Bash"
    assert call.target_class == 'password = "hunter2hunter2"'


def test_redaction_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(turn_digest, "redact_secrets", lambda _c: None)
    assert redact_turn_digest(_body()) is None


def _call(tool: str = "Bash", cls: str = "ls") -> dict[str, Any]:
    return {"kind": "tool_call", "tool_use_id": "a", "tool": tool, "target_class": cls}


def _res(uid: str = "a") -> dict[str, Any]:
    return {"kind": "tool_result", "tool_use_id": uid, "is_error": False}


def test_anomalies() -> None:
    assert turn_digest_anomalies(_body(items=[_call(cls="")])) == [
        "empty_bash_target_class"
    ]
    assert turn_digest_anomalies(_body(items=[_res("zz")])) == ["orphan_tool_result"]
    assert turn_digest_anomalies(_body(items=[_res("zz")], truncated=True)) == []
    assert turn_digest_anomalies(
        _body(user_prompt=None, items=[], final_message=None)
    ) == ["empty_turn"]
    assert turn_digest_anomalies(_body(items=[_call(cls=""), _res("zz")])) == [
        "empty_bash_target_class",
        "orphan_tool_result",
    ]
    assert turn_digest_anomalies(_body(items=[_call(), _res()])) == []


async def test_clean_digest_stores_null_anomaly(pool: SqlitePool) -> None:
    await _insert(pool, _body(items=[_call(), _res()]))
    assert (await get_session_turn_digests(pool, "s1"))[0].anomaly is None
    row = await pool.fetchrow("SELECT items FROM turn_events")
    assert row is not None
    assert json.loads(row["items"])[0]["kind"] == "tool_call"


def test_str_field_partition_drift_guard() -> None:
    """Only user_prompt, final_message, text, target and excerpt are redacted.

    A newly added str field fails here until it is classified.
    """

    def str_fields(model: type) -> set[str]:
        out = set()
        for name, info in model.model_fields.items():  # type: ignore[attr-defined]
            ann = info.annotation
            if ann is str or (
                typing.get_origin(ann) is not None
                and set(typing.get_args(ann)) == {str, type(None)}
            ):
                out.add(name)
        return out

    assert str_fields(TurnDigestRequest) == {
        "event_id",
        "session_id",
        "harness",
        "engine",
        "host",
        "hook_version",
        "project",
        "ts",
        "user_prompt",
        "final_message",
    }
    assert str_fields(TurnAssistantTextItem) == {"text"}
    assert str_fields(TurnToolCallItem) == {
        "tool_use_id",
        "tool",
        "target",
        "target_class",
    }
    assert str_fields(TurnToolResultItem) == {"tool_use_id", "excerpt"}
