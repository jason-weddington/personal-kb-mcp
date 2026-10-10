"""Shared seed rows for the repeat-rate tests (not collected as a test module)."""

import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kb_service.repeat_rate import FailureRow, ResolutionCue

_A: dict[str, Any] = {
    "cue_key": "cue-a",
    "tool": "Bash",
    "target": "git push github main",
    "target_class": "git push",
    "normalized_error": "exit 1 | fatal: remote github rejected",
}
_B: dict[str, Any] = {
    "cue_key": "cue-b",
    "tool": "Bash",
    "target": "pytest -q",
    "target_class": "pytest",
    "normalized_error": "exit 1 | <n> failed",
}


def _row(
    row_id: int,
    session: str,
    base: dict[str, Any],
    project: str,
    ts: str,
    mode: str,
    engine: str | None,
    host: str,
    host_class: str,
    harness: str,
    is_interrupt: int,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "session_id": session,
        **base,
        "project": project,
        "ts": ts,
        "mode": mode,
        "engine": engine,
        "host": host,
        "host_class": host_class,
        "harness": harness,
        "is_interrupt": is_interrupt,
        **extra,
    }


_H = "claude-code"
_SONNET = "claude-code-sonnet"
FIXTURE_ROWS: list[dict[str, Any]] = [
    _row(
        1,
        "S0",
        _A,
        "p",
        "2026-09-20T10:00:00+00:00",
        "interactive",
        None,
        "h1",
        "linux",
        _H,
        0,
    ),
    _row(
        2,
        "S1",
        _A,
        "p",
        "2026-09-29T10:00:00+00:00",
        "interactive",
        None,
        "h1",
        "linux",
        _H,
        0,
    ),
    _row(
        3,
        "S1",
        _B,
        "p",
        "2026-09-29T11:00:00+00:00",
        "interactive",
        None,
        "h1",
        "linux",
        _H,
        0,
    ),
    _row(
        4,
        "S2",
        _B,
        "p",
        "2026-09-29T20:00:00+00:00",
        "headless",
        _SONNET,
        None,
        "darwin",
        _H,
        0,
    ),
    _row(
        5,
        "S3",
        _B,
        "p",
        "2026-10-06T09:00:00+00:00",
        "headless",
        _SONNET,
        "h2",
        "darwin",
        _H,
        0,
    ),
    _row(
        6,
        "S3",
        _A,
        "p",
        "2026-10-06T09:30:00+00:00",
        "headless",
        _SONNET,
        "h2",
        "darwin",
        _H,
        1,
    ),
    _row(
        7,
        "S4",
        {
            "cue_key": "cue-c",
            "tool": "Bash",
            "target": "git show HEAD; git push github main",
            "target_class": "git show",
            "normalized_error": "exit 128 | fatal: bad object",
        },
        "p",
        "2026-10-06T12:00:00+00:00",
        "interactive",
        None,
        "h1",
        "linux",
        _H,
        0,
    ),
    _row(
        8,
        "S5",
        {
            "cue_key": "cue-d",
            "tool": "Bash",
            "target": "git push github main",
            "target_class": "git push",
            "normalized_error": "exit 1 | fatal: remote github rejected",
        },
        "q",
        "2026-10-07T08:00:00+00:00",
        "interactive",
        None,
        "h1",
        "linux",
        _H,
        0,
    ),
    # Harness 'talos' is SYNTHETIC: a forward-compat value for GTD 506d0b53;
    # no producer writes it today.
    _row(
        9,
        "S6",
        _A,
        "p",
        "2026-10-07T09:00:00+00:00",
        "headless",
        "talos-glm",
        "h3",
        "linux",
        "talos",
        0,
    ),
    _row(
        10,
        "S7",
        {
            "cue_key": "cue-e",
            "tool": "Bash",
            "target": "git push origin main",
            "target_class": "git push",
            "normalized_error": "exit 1 | rejected (fetch first)",
        },
        "p",
        "2026-10-07T10:00:00+00:00",
        "interactive",
        None,
        "h2",
        "darwin",
        _H,
        0,
    ),
    _row(
        11,
        "S8",
        {
            "cue_key": "cue-f",
            "tool": "Read",
            "target": "/x/README.md",
            "target_class": "ext:md",
            "normalized_error": "file does not exist",
        },
        "q",
        "2026-10-07T11:00:00+00:00",
        "interactive",
        None,
        "h1",
        "linux",
        _H,
        0,
    ),
    _row(
        12,
        "S1",
        _A,
        "p",
        "2026-09-29T13:00:00+00:00",
        "interactive",
        None,
        "h1",
        "linux",
        _H,
        0,
    ),
]

EXTRA_ROW_BASE: dict[str, Any] = {
    "id": 13,
    "session_id": "S9",
    "cue_key": "cue-z",
    "tool": "Bash",
    "target": "ls /nope",
    "target_class": "ls",
    "project": "p",
    "mode": "interactive",
    "engine": None,
    "host": "h1",
    "host_class": "linux",
    "harness": "claude-code",
    "is_interrupt": 0,
    "normalized_error": "exit 2 | no such file",
}

R1 = ResolutionCue(
    "r1",
    "p",
    "project",
    "Bash",
    "git push",
    "github",
    datetime(2026, 9, 29, 12, tzinfo=UTC),
)
R2 = ResolutionCue(
    "r2", "other", "global", "Read", "ext:md", "", datetime(2026, 9, 1, tzinfo=UTC)
)


def make_row(**overrides: Any) -> FailureRow:
    """A FailureRow with sensible defaults; override any field."""
    fields: dict[str, Any] = {
        "id": 1,
        "session_id": "S1",
        "cue_key": "cue-a",
        "normalizer_version": 1,
        "harness": "claude-code",
        "mode": "interactive",
        "engine": None,
        "host": "h1",
        "host_class": "linux",
        "project": "p",
        "tool": "Bash",
        "target": "git push github main",
        "target_class": "git push",
        "normalized_error": "exit 1 | fatal: remote github rejected",
        "ts": datetime(2026, 10, 6, 12, tzinfo=UTC),
    }
    fields.update(overrides)
    return FailureRow(**fields)


_COLUMNS = (
    "event_id, cue_key, normalizer_version, session_id, harness, mode, engine, "
    "host, hook_version, host_class, project, project_source, tool, target, "
    "target_class, normalized_error, error_rule, anomaly, raw_error_excerpt, "
    "is_interrupt, ts, received_ts"
)


def _values(r: dict[str, Any]) -> list[Any]:
    return [
        f"e{r['id']}",
        r["cue_key"],
        1,
        r["session_id"],
        r["harness"],
        r["mode"],
        r["engine"],
        r["host"],
        None,
        r["host_class"],
        r["project"],
        "kb_project",
        r["tool"],
        r["target"],
        r["target_class"],
        r["normalized_error"],
        "test",
        None,
        "raw",
        r["is_interrupt"],
        r["ts"],
        r["ts"],
    ]


async def insert_rows_pool(
    pool: Any, rows: Sequence[dict[str, Any]] = FIXTURE_ROWS
) -> None:
    """Insert *rows* through a service-DB pool ($N placeholders)."""
    marks = ", ".join(f"${i}" for i in range(1, 23))
    sql = f"INSERT INTO failure_events ({_COLUMNS}) VALUES ({marks})"  # noqa: S608
    for r in rows:
        await pool.execute(sql, *_values(r))


def insert_rows_sqlite3(
    path: Path, rows: Sequence[dict[str, Any]] = FIXTURE_ROWS
) -> None:
    """Insert *rows* with plain sqlite3 into the service DB file at *path*."""
    marks = ", ".join("?" * 22)
    sql = f"INSERT INTO failure_events ({_COLUMNS}) VALUES ({marks})"  # noqa: S608
    conn = sqlite3.connect(path)
    try:
        for r in rows:
            conn.execute(sql, _values(r))
        conn.commit()
    finally:
        conn.close()
