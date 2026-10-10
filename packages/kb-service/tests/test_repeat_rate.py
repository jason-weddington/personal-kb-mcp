"""Weekly cross-session repeat rate: fetch, load, match, compute, render."""

import logging
import sys
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from kb_core import create_sqlite
from kb_core.cues import FailureCue

import kb_service.database as database
from kb_service import repeat_rate
from kb_service.db_sqlite import SqlitePool
from kb_service.repeat_rate import (
    FailureFetchStats,
    ResolutionCue,
    ResolutionCueLoadStats,
    audit_repeat_rate,
    build_repeat_rate,
    compute_repeat_rate,
    cue_matches,
    fetch_failure_rows,
    load_resolution_cues,
    render_repeat_rate_markdown,
    resolution_matches,
    window_bounds,
)
from tests.repeat_rate_fixtures import (
    EXTRA_ROW_BASE,
    FIXTURE_ROWS,
    R1,
    R2,
    insert_rows_pool,
    make_row,
)

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from failure_cue_baseline import Failure, compute_report  # noqa: E402

NOW = datetime(2026, 10, 7, 12, tzinfo=UTC)
END = datetime(2026, 10, 12, tzinfo=UTC)

_SCRUB_ENV = (
    "KB_SERVICE_DATABASE_URL",
    "KB_DATABASE_URL",
    "ANTHROPIC_API_KEY",
    "KB_AWS_PROFILE",
)


@pytest.fixture
def local_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Local no-auth mode with the data DB and service DB in *tmp_path*."""
    for var in _SCRUB_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.setenv("KB_OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("KB_OLLAMA_TIMEOUT", "0.5")
    monkeypatch.setenv("KB_EXTRACTION_PROVIDER", "ollama")
    monkeypatch.setenv("KB_QUERY_PROVIDER", "ollama")
    monkeypatch.setattr(database, "_pool", None)
    yield tmp_path / "service.db"


@pytest.fixture
async def pool(local_env: Path) -> AsyncIterator[SqlitePool]:
    """An initialised SQLite service DB, closed afterwards."""
    await database.init_db()
    db = await database.get_db()
    assert isinstance(db, SqlitePool)
    yield db
    await database.close_db()


@pytest.fixture
async def kb(tmp_path: Path) -> AsyncIterator[Any]:
    kb = await create_sqlite(
        tmp_path / "kb.db",
        embedder=None,
        extraction_llm=None,
        query_llm=None,
        synthesis_llm=None,
    )
    try:
        yield kb
    finally:
        await kb.close()


async def _store(kb: Any, project: str, **resolution: Any) -> str:
    entry = await kb.store(
        short_title="Resolution",
        long_title="Resolution long",
        knowledge_details="details",
        project_ref=project,
        hints={"resolution": resolution},
        enrich=False,
    )
    return str(entry.id)


async def _seeded(pool: SqlitePool, project: str | None = None) -> Any:
    await insert_rows_pool(pool)
    return await fetch_failure_rows(pool, END, project)


def _wk(w: Any) -> dict[str, Any]:
    return w.model_dump()


# ─── window ──────────────────────────────────────────────────────────────────


def test_window_bounds() -> None:
    want = (datetime(2026, 9, 28, tzinfo=UTC), datetime(2026, 10, 12, tzinfo=UTC))
    assert window_bounds(datetime(2026, 10, 7, 12, tzinfo=UTC), 2) == want
    assert window_bounds(datetime(2026, 10, 7, 12), 2) == want
    est = timezone(timedelta(hours=-5))
    assert window_bounds(datetime(2026, 10, 4, 23, tzinfo=est), 1) == (
        datetime(2026, 10, 5, tzinfo=UTC),
        datetime(2026, 10, 12, tzinfo=UTC),
    )
    assert window_bounds(datetime(2027, 1, 1, tzinfo=UTC), 1) == (
        datetime(2026, 12, 28, tzinfo=UTC),
        datetime(2027, 1, 4, tzinfo=UTC),
    )
    resp = compute_repeat_rate([], [], weeks=1, now=datetime(2027, 1, 1, tzinfo=UTC))
    assert (resp.weeks[0].week, resp.weeks[0].week_start) == ("2026-W53", "2026-12-28")
    for bad in (0, 105):
        with pytest.raises(ValueError, match="weeks must be between 1 and 104"):
            window_bounds(NOW, bad)


# ─── fetch ───────────────────────────────────────────────────────────────────


async def test_fetch_all_and_project(pool: SqlitePool) -> None:
    rows, stats = await _seeded(pool)
    assert [r.id for r in rows] == [1, 2, 3, 12, 4, 5, 7, 8, 9, 10, 11]
    assert stats.scanned == 11
    rows_p, _ = await fetch_failure_rows(pool, END, "p")
    assert [r.id for r in rows_p] == [1, 2, 3, 12, 4, 5, 7, 9, 10]


async def test_fetch_bad_ts(pool: SqlitePool, caplog: pytest.LogCaptureFixture) -> None:
    extra = {**EXTRA_ROW_BASE, "ts": "2026-02-30T00:00:00+00:00"}
    await insert_rows_pool(pool, [*FIXTURE_ROWS, extra])
    with caplog.at_level(logging.WARNING):
        rows, stats = await fetch_failure_rows(pool, END, None)
    assert len(rows) == 11
    assert (stats.scanned, stats.skipped_bad_ts) == (12, 1)
    assert "repeat_rate skipped_row id=13 reason=bad_ts" in caplog.text


async def test_fetch_noncanonical_ts(
    pool: SqlitePool, caplog: pytest.LogCaptureFixture
) -> None:
    extra = {**EXTRA_ROW_BASE, "ts": "2026-10-07T12:00:00Z"}
    await insert_rows_pool(pool, [*FIXTURE_ROWS, extra])
    with caplog.at_level(logging.WARNING):
        rows, stats = await fetch_failure_rows(pool, END, None)
    assert len(rows) == 12 and rows[-1].id == 13
    assert stats.noncanonical_ts == 1 and stats.first_noncanonical_id == 13
    assert "repeat_rate tripwire=noncanonical_ts count=1 first_id=13" in caplog.text
    resp = compute_repeat_rate(rows, [], weeks=2, now=NOW, fetch_stats=stats)
    assert resp.diagnostics.tripwires == ["noncanonical_ts"]
    assert render_repeat_rate_markdown(resp).split("\n")[2] == (
        "**TRIPWIRE: noncanonical_ts**"
    )


async def test_fetch_naive_window_end(pool: SqlitePool) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        await fetch_failure_rows(pool, datetime(2026, 10, 12), None)


async def test_fetch_naive_row_ts(pool: SqlitePool) -> None:
    extra = {**EXTRA_ROW_BASE, "ts": "2026-10-07T12:00:00"}
    await insert_rows_pool(pool, [extra])
    rows, stats = await fetch_failure_rows(pool, END, None)
    assert rows[0].ts == datetime(2026, 10, 7, 12, tzinfo=UTC)
    assert stats.noncanonical_ts == 1


# ─── resolution loader ───────────────────────────────────────────────────────


async def test_load_resolution_cues(kb: Any, caplog: pytest.LogCaptureFixture) -> None:
    r1 = await _store(
        kb,
        "p",
        corrected_fact="x",
        cue={"tool": "Bash", "target_class": "git push", "args_prefix": "github"},
    )
    r2 = await _store(
        kb,
        "other",
        corrected_fact="x",
        scope="global",
        cue={"tool": "Read", "target_class": "ext:md"},
    )
    await _store(kb, "p", corrected_fact="x")
    r4 = await _store(kb, "p", corrected_fact=5)
    r5 = await _store(
        kb, "p", corrected_fact="x", cue={"tool": "Bash", "target_class": "git push"}
    )
    r6 = await _store(
        kb, "p", corrected_fact="x", cue={"tool": "Bash", "target_class": "git pull"}
    )
    r7 = await _store(
        kb, "p", corrected_fact="x", cue={"tool": "Bash", "target_class": "git fetch"}
    )
    for sql, params in (
        ("UPDATE knowledge_entries SET is_active = 0 WHERE id = ?", (r5,)),
        (
            "UPDATE knowledge_entries SET created_at = ? WHERE id = ?",
            ("2026-09-29T12:00:00+00:00", r1),
        ),
        ("UPDATE knowledge_entries SET superseded_by = ? WHERE id = ?", (r1, r6)),
        (
            "UPDATE knowledge_entries SET created_at = ? WHERE id = ?",
            ("not-a-date", r7),
        ),
    ):
        await kb.db.execute(sql, params)
    await kb.db.commit()
    with caplog.at_level(logging.WARNING):
        cues, stats = await load_resolution_cues(kb.db)
    assert [c.entry_id for c in cues] == [r1, r2, r5, r6]
    assert cues[0].args_prefix == "github"
    assert cues[0].created_at == datetime(2026, 9, 29, 12, tzinfo=UTC)
    assert cues[0].project == "p"
    assert (cues[1].scope, cues[1].project) == ("global", "other")
    assert cues[2].args_prefix == ""
    assert stats == ResolutionCueLoadStats(
        scanned=7, skipped_malformed=1, skipped_no_cue=1, skipped_bad_created_at=1
    )
    assert (
        f"repeat_rate skipped_resolutions first_entry_id={r4} malformed=1"
        " bad_created_at=1" in caplog.text
    )


# ─── matching ────────────────────────────────────────────────────────────────


def test_matching() -> None:
    h = make_row(
        target="git show HEAD; git push github main",
        target_class="git show",
        ts=datetime(2026, 10, 6, 12, tzinfo=UTC),
    )
    assert resolution_matches(R1, h)
    origin = make_row(
        target="git push origin main", ts=datetime(2026, 10, 7, 10, tzinfo=UTC)
    )
    assert not resolution_matches(R1, origin)
    other_project = make_row(project="q", ts=datetime(2026, 10, 7, 8, tzinfo=UTC))
    assert not resolution_matches(R1, other_project)
    read = make_row(
        tool="Read",
        target="/x/README.md",
        target_class="ext:md",
        project="q",
        ts=datetime(2026, 10, 7, 11, tzinfo=UTC),
    )
    assert resolution_matches(R2, read)
    early = make_row(ts=datetime(2026, 9, 29, 11, 59, 59, tzinfo=UTC))
    assert not resolution_matches(R1, early) and cue_matches(R1, early)
    assert resolution_matches(R1, make_row(ts=datetime(2026, 9, 29, 12, tzinfo=UTC)))
    r9 = ResolutionCue(
        "r9",
        "other",
        "global",
        "Read",
        "ext:md",
        "foo",
        datetime(2026, 9, 1, tzinfo=UTC),
    )
    assert resolution_matches(r9, read)
    assert not resolution_matches(R1, read)


# ─── compute ─────────────────────────────────────────────────────────────────

W40 = {
    "week": "2026-W40",
    "week_start": "2026-09-28",
    "sessions": 2,
    "repeat_sessions": 1,
    "repeat_rate": 0.5,
    "distinct_cue_keys": 2,
    "pairs": 3,
    "repeat_pairs": 1,
    "aggregate_rate": 0.3333,
    "covered_sessions": 0,
    "covered_repeat_sessions": 0,
    "covered_repeat_rate": None,
}
W41 = {
    "week": "2026-W41",
    "week_start": "2026-10-05",
    "sessions": 6,
    "repeat_sessions": 2,
    "repeat_rate": 0.3333,
    "distinct_cue_keys": 6,
    "pairs": 6,
    "repeat_pairs": 2,
    "aggregate_rate": 0.3333,
    "covered_sessions": 3,
    "covered_repeat_sessions": 1,
    "covered_repeat_rate": 0.3333,
}


async def _main(pool: SqlitePool) -> Any:
    rows, fstats = await _seeded(pool)
    return compute_repeat_rate(
        rows, [R1, R2], weeks=2, now=NOW, min_gap_hours=24.0, fetch_stats=fstats
    )


async def test_main_case(pool: SqlitePool) -> None:
    resp = await _main(pool)
    assert resp.window_start == "2026-09-28T00:00:00+00:00"
    assert resp.window_end == "2026-10-12T00:00:00+00:00"
    assert (resp.failure_rows, resp.resolutions_loaded, resp.project) == (11, 2, None)
    assert resp.min_gap_hours == 24.0
    assert _wk(resp.weeks[0]) == W40
    assert _wk(resp.weeks[1]) == W41
    assert _wk(resp.total) == {
        "sessions": 8,
        "repeat_sessions": 3,
        "repeat_rate": 0.375,
        "distinct_cue_keys": 6,
        "pairs": 9,
        "repeat_pairs": 3,
        "aggregate_rate": 0.3333,
        "covered_sessions": 3,
        "covered_repeat_sessions": 1,
        "covered_repeat_rate": 0.3333,
    }
    assert _wk(resp.diagnostics) == {
        "failure_rows_scanned": 11,
        "failure_rows_skipped_bad_ts": 0,
        "failure_rows_noncanonical_ts": 0,
        "failure_rows_before_window": 1,
        "resolutions_scanned": 2,
        "resolutions_skipped_malformed": 0,
        "resolutions_skipped_no_cue": 0,
        "resolutions_skipped_bad_created_at": 0,
        "coverage_near_miss_created_after": 1,
        "normalizer_versions": [1],
        "singleton_cue_fraction": 0.6667,
        "top_cue_share": 0.3,
        "tripwires": [],
    }
    rows, _ = await fetch_failure_rows(pool, END, None)
    base = compute_repeat_rate(rows, [R1, R2], weeks=2, now=NOW)
    assert compute_repeat_rate(list(reversed(rows)), [R1, R2], weeks=2, now=NOW) == base
    future = make_row(id=99, session_id="S99", ts=datetime(2026, 10, 12, tzinfo=UTC))
    more = compute_repeat_rate([*rows, future], [R1, R2], weeks=2, now=NOW)
    assert more.weeks == base.weeks and more.total == base.total
    assert more.failure_rows == 11


async def test_project_and_gap_variants(pool: SqlitePool) -> None:
    rows, _ = await _seeded(pool, "p")
    resp = compute_repeat_rate(rows, [R1, R2], weeks=2, now=NOW, project="p")
    assert (resp.failure_rows, resp.project) == (9, "p")
    assert _wk(resp.weeks[0]) == W40
    assert _wk(resp.weeks[1]) == {
        **W41,
        "sessions": 4,
        "repeat_rate": 0.5,
        "distinct_cue_keys": 4,
        "pairs": 4,
        "aggregate_rate": 0.5,
        "covered_sessions": 2,
        "covered_repeat_rate": 0.5,
    }
    assert resp.total.sessions == 6 and resp.total.aggregate_rate == 0.4286
    rows, _ = await _seeded_again(pool)
    z = compute_repeat_rate(rows, [R1, R2], weeks=2, now=NOW, min_gap_hours=0.0)
    assert z.weeks[0].repeat_sessions == 2 and z.weeks[0].repeat_rate == 1.0
    assert z.weeks[0].aggregate_rate == 0.6667
    assert _wk(z.weeks[1]) == W41
    assert (z.total.repeat_sessions, z.total.aggregate_rate) == (4, 0.4444)
    assert [c.cue_key for c in z.top_cues] == ["cue-b", "cue-a"]
    assert z.top_cues[0].repeat_sessions == 2
    for bad in (-1, 720.5):
        with pytest.raises(ValueError, match="min_gap_hours must be between 0 and"):
            compute_repeat_rate(rows, [], weeks=2, now=NOW, min_gap_hours=bad)


async def _seeded_again(pool: SqlitePool) -> Any:
    return await fetch_failure_rows(pool, END, None)


async def test_cuts_and_top_cues(pool: SqlitePool) -> None:
    resp = await _main(pool)

    def t(rows: Any) -> list[tuple[Any, ...]]:
        return [
            (
                r.week,
                r.key,
                r.sessions,
                r.repeat_sessions,
                r.repeat_rate,
                r.distinct_cue_keys,
                r.pairs,
                r.repeat_pairs,
                r.aggregate_rate,
                r.covered_sessions,
                r.covered_repeat_sessions,
                r.covered_repeat_rate,
            )
            for r in rows
        ]

    assert t(resp.by_harness) == [
        ("2026-W40", "claude-code", 2, 1, 0.5, 2, 3, 1, 0.3333, 0, 0, None),
        ("2026-W41", "claude-code", 5, 1, 0.2, 5, 5, 1, 0.2, 2, 0, 0.0),
        ("2026-W41", "talos", 1, 1, 1.0, 1, 1, 1, 1.0, 1, 1, 1.0),
    ]
    assert t(resp.by_mode) == [
        ("2026-W40", "headless", 1, 0, 0.0, 1, 1, 0, 0.0, 0, 0, None),
        ("2026-W40", "interactive", 1, 1, 1.0, 2, 2, 1, 0.5, 0, 0, None),
        ("2026-W41", "headless", 2, 2, 1.0, 2, 2, 2, 1.0, 1, 1, 1.0),
        ("2026-W41", "interactive", 4, 0, 0.0, 4, 4, 0, 0.0, 2, 0, 0.0),
    ]
    assert t(resp.by_engine) == [
        ("2026-W40", "claude-code-sonnet", 1, 0, 0.0, 1, 1, 0, 0.0, 0, 0, None),
        ("2026-W40", "none", 1, 1, 1.0, 2, 2, 1, 0.5, 0, 0, None),
        ("2026-W41", "claude-code-sonnet", 1, 1, 1.0, 1, 1, 1, 1.0, 0, 0, None),
        ("2026-W41", "none", 4, 0, 0.0, 4, 4, 0, 0.0, 2, 0, 0.0),
        ("2026-W41", "talos-glm", 1, 1, 1.0, 1, 1, 1, 1.0, 1, 1, 1.0),
    ]
    assert t(resp.by_host_class) == [
        ("2026-W40", "darwin", 1, 0, 0.0, 1, 1, 0, 0.0, 0, 0, None),
        ("2026-W40", "linux", 1, 1, 1.0, 2, 2, 1, 0.5, 0, 0, None),
        ("2026-W41", "darwin", 2, 1, 0.5, 2, 2, 1, 0.5, 0, 0, None),
        ("2026-W41", "linux", 4, 1, 0.25, 4, 4, 1, 0.25, 3, 1, 0.3333),
    ]
    assert t(resp.by_host) == [
        ("2026-W40", "h1", 1, 1, 1.0, 2, 2, 1, 0.5, 0, 0, None),
        ("2026-W40", "unknown", 1, 0, 0.0, 1, 1, 0, 0.0, 0, 0, None),
        ("2026-W41", "h1", 3, 0, 0.0, 3, 3, 0, 0.0, 2, 0, 0.0),
        ("2026-W41", "h2", 2, 1, 0.5, 2, 2, 1, 0.5, 0, 0, None),
        ("2026-W41", "h3", 1, 1, 1.0, 1, 1, 1, 1.0, 1, 1, 1.0),
    ]
    assert [c.model_dump() for c in resp.top_cues] == [
        {
            "cue_key": "cue-a",
            "tool": "Bash",
            "target_class": "git push",
            "project": "p",
            "normalized_error": "exit 1 | fatal: remote github rejected",
            "sessions": 2,
            "repeat_sessions": 2,
            "covered_sessions": 1,
            "covering_resolution_ids": ["r1"],
        },
        {
            "cue_key": "cue-b",
            "tool": "Bash",
            "target_class": "pytest",
            "project": "p",
            "normalized_error": "exit 1 | <n> failed",
            "sessions": 3,
            "repeat_sessions": 1,
            "covered_sessions": 0,
            "covering_resolution_ids": [],
        },
    ]


async def test_parity_with_backfill(pool: SqlitePool) -> None:
    rows, fstats = await _seeded(pool)
    resp = compute_repeat_rate(rows, [R1, R2], weeks=4, now=NOW, fetch_stats=fstats)
    assert resp.window_start == "2026-09-14T00:00:00+00:00"
    assert _wk(resp.weeks[0]) == {
        "week": "2026-W38",
        "week_start": "2026-09-14",
        "sessions": 1,
        "repeat_sessions": 0,
        "repeat_rate": 0.0,
        "distinct_cue_keys": 1,
        "pairs": 1,
        "repeat_pairs": 0,
        "aggregate_rate": 0.0,
        "covered_sessions": 0,
        "covered_repeat_sessions": 0,
        "covered_repeat_rate": None,
    }
    z = resp.weeks[1]
    assert (z.week, z.week_start, z.sessions, z.repeat_rate) == (
        "2026-W39",
        "2026-09-21",
        0,
        None,
    )
    assert (z.aggregate_rate, z.covered_repeat_rate) == (None, None)
    assert _wk(resp.weeks[2]) == W40 and _wk(resp.weeks[3]) == W41
    tot = resp.total
    assert (tot.sessions, tot.pairs, tot.repeat_pairs) == (9, 10, 3)
    assert tot.aggregate_rate == 0.3 and tot.repeat_rate == 0.3333
    d = resp.diagnostics
    assert d.failure_rows_before_window == 0
    assert d.coverage_near_miss_created_after == 2
    assert (d.singleton_cue_fraction, d.top_cue_share) == (0.6667, 0.3636)

    failures = [
        Failure(
            cue=FailureCue(
                tool=r["tool"],
                target=r["target"],
                target_class=r["target_class"],
                normalized_error=r["normalized_error"],
                error_rule="test",
                project=r["project"],
                project_source="kb_project",
                host_class=r["host_class"],
                normalizer_version=1,
                cue_key=r["cue_key"],
            ),
            session_id=r["session_id"],
            ts=datetime.fromisoformat(r["ts"]),
            mode=r["mode"],
            raw="raw",
            tool_use_id=f"t{r['id']}",
        )
        for r in FIXTURE_ROWS
        if r["is_interrupt"] == 0
    ]
    rep = compute_report(failures, 24.0)
    assert sum(c["sessions"] for c in rep["cues"]) == 10 == tot.pairs
    assert sum(c["repeat_sessions"] for c in rep["cues"]) == tot.repeat_pairs == 3
    assert rep["totals"]["aggregate_rate"] == 0.3 == tot.aggregate_rate
    assert rep["totals"]["singleton_cue_fraction"] == d.singleton_cue_fraction
    assert rep["totals"]["top_cue_share"] == d.top_cue_share
    assert (
        {x["week"]: x["repeat_sessions"] for x in rep["weeks"]}
        == {
            "2026-W40": 1,
            "2026-W41": 2,
        }
        == {w.week: w.repeat_pairs for w in resp.weeks if w.repeat_pairs}
    )


def test_per_pair_booking_and_empty_keys() -> None:
    rows = [
        make_row(
            id=1,
            session_id="S10",
            cue_key="cue-y",
            ts=datetime(2026, 10, 1, tzinfo=UTC),
        ),
        make_row(
            id=2,
            session_id="S9",
            cue_key="cue-x",
            ts=datetime(2026, 10, 4, 23, tzinfo=UTC),
        ),
        make_row(
            id=3,
            session_id="S9",
            cue_key="cue-y",
            ts=datetime(2026, 10, 5, 1, tzinfo=UTC),
        ),
    ]
    resp = compute_repeat_rate(rows, [], weeks=2, now=NOW)
    w0, w1 = resp.weeks
    assert (w0.sessions, w0.repeat_sessions, w0.pairs, w0.repeat_pairs) == (2, 0, 2, 0)
    assert w0.aggregate_rate == 0.0
    assert (w1.sessions, w1.repeat_sessions, w1.pairs, w1.repeat_pairs) == (1, 1, 1, 1)
    assert w1.aggregate_rate == 1.0
    t = resp.total
    assert (t.sessions, t.repeat_sessions, t.pairs, t.repeat_pairs) == (2, 1, 3, 1)
    assert t.aggregate_rate == 0.3333

    one = compute_repeat_rate(
        [make_row(host="", engine="", ts=datetime(2026, 10, 6, tzinfo=UTC))],
        [],
        weeks=1,
        now=NOW,
    )
    assert [r.key for r in one.by_host] == ["unknown"]
    assert [r.key for r in one.by_engine] == ["none"]


async def test_audit_tripwires(pool: SqlitePool) -> None:
    resp = await _main(pool)
    assert audit_repeat_rate(resp) == []
    c = resp.model_copy(deep=True)
    c.by_harness[0].sessions += 1
    assert audit_repeat_rate(c) == ["cut_sum_mismatch"]
    c2 = resp.model_copy(deep=True)
    c2.total.pairs += 1
    assert audit_repeat_rate(c2) == ["pairs_total_mismatch"]
    c3 = resp.model_copy(deep=True)
    c3.total.repeat_sessions = c3.total.sessions + 1
    assert audit_repeat_rate(c3) == ["count_bounds"]


def test_mixed_normalizer_versions(caplog: pytest.LogCaptureFixture) -> None:
    rows = [
        make_row(id=1, normalizer_version=1),
        make_row(id=2, session_id="S2", normalizer_version=2),
    ]
    with caplog.at_level(logging.WARNING):
        resp = compute_repeat_rate(rows, [], weeks=1, now=NOW)
    assert resp.diagnostics.normalizer_versions == [1, 2]
    assert "mixed_normalizer_versions" in caplog.text


# ─── orchestrator ────────────────────────────────────────────────────────────


async def test_build_repeat_rate(
    pool: SqlitePool, kb: Any, caplog: pytest.LogCaptureFixture
) -> None:
    await insert_rows_pool(pool)
    with caplog.at_level(logging.INFO, logger="kb_service.repeat_rate"):
        resp = await build_repeat_rate(
            pool, kb.db, weeks=2, project=None, min_gap_hours=24.0, now=NOW
        )
    assert resp.failure_rows == 11 and resp.resolutions_loaded == 0
    lines = [
        r.getMessage() for r in caplog.records if "repeat_rate weeks=2" in r.message
    ]
    assert len(lines) == 1 and "tripwires=none" in lines[0]
    with pytest.raises(ValueError, match="min_gap_hours"):
        await build_repeat_rate(
            pool, kb.db, weeks=2, project=None, min_gap_hours=-1, now=NOW
        )
    with pytest.raises(ValueError, match="weeks"):
        await build_repeat_rate(
            pool, kb.db, weeks=0, project=None, min_gap_hours=1, now=NOW
        )


# ─── markdown ────────────────────────────────────────────────────────────────


async def test_markdown(pool: SqlitePool) -> None:
    out = render_repeat_rate_markdown(await _main(pool))
    lines = out.split("\n")
    for want in (
        "# Cross-session repeat rate: 2026-W40 to 2026-W41"
        " (project=all, min_gap_hours=24.0)",
        "| 2026-W40 | 2 | 1 | 0.5000 | 3 | 1 | 0.3333 | 2 | 0 | 0 | - |",
        "| 2026-W41 | 6 | 2 | 0.3333 | 6 | 2 | 0.3333 | 6 | 3 | 1 | 0.3333 |",
        "| total | 8 | 3 | 0.3750 | 9 | 3 | 0.3333 | 6 | 3 | 1 | 0.3333 |",
        "| 2026-W41 | talos | 1 | 1 | 1.0000 | 1.0000 | 1 | 1 | 1.0000 |",
        "| 2026-W41 | talos-glm | 1 | 1 | 1.0000 | 1.0000 | 1 | 1 | 1.0000 |",
        "| cue-a | Bash | git push | p | 2 | 2 | 1 | r1 |"
        " exit 1 \\| fatal: remote github rejected |",
        "| cue-b | Bash | pytest | p | 3 | 1 | 0 | - | exit 1 \\| <n> failed |",
    ):
        assert want in lines
    heads = [ln for ln in lines if ln.startswith("## ")]
    assert heads == [
        "## weeks",
        "## by_harness",
        "## by_mode",
        "## by_engine",
        "## by_host_class",
        "## by_host",
        "## top_cues",
    ]
    assert not any(ln.startswith("**TRIPWIRE") for ln in lines)
    assert lines[-1] == (
        "failure_rows=11 failure_rows_scanned=11 skipped_bad_ts=0 noncanonical_ts=0"
        " before_window=1 resolutions_loaded=2 resolutions_scanned=2"
        " resolutions_skipped=0 near_miss_created_after=1 normalizer_versions=1"
        " tripwires=none"
    )


def test_markdown_empty_and_project() -> None:
    resp = compute_repeat_rate([], [], weeks=1, now=datetime(2027, 1, 1, tzinfo=UTC))
    lines = render_repeat_rate_markdown(resp).split("\n")
    assert "| 2026-W53 | 0 | 0 | - | 0 | 0 | - | 0 | 0 | 0 | - |" in lines
    assert "| total | 0 | 0 | - | 0 | 0 | - | 0 | 0 | 0 | - |" in lines
    i = lines.index("## top_cues")
    assert lines[i + 2] == "|---|---|---|---|---|---|---|---|---|"
    assert lines[i + 3] == ""
    assert lines[-1].endswith("normalizer_versions=- tripwires=none")
    proj = compute_repeat_rate([], [], weeks=1, now=NOW, project="p")
    assert "project=p," in render_repeat_rate_markdown(proj).split("\n")[0]


def test_unused_stats_defaults() -> None:
    assert FailureFetchStats().first_noncanonical_id is None
    assert repeat_rate.TOP_CUES_CAP == 20
