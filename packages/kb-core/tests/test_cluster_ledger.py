"""Hermetic SQLite suite for the cluster/decline ledger.

No Postgres, no Ollama, no network: everything runs against a real SQLite
KB created via ``create_sqlite(tmp_path / "kb.db")``. Ledger rows are
seeded via direct ``kb.db.execute`` INSERTs, never ``kb.store()`` — the
store touches the embedder, and these tests are about SQL, Jaccard
arithmetic and the decline/reopen boundary.

The worked fixture (``_seed_proposed_row`` and the A-G candidates) is the
one pinned in the item spec: one ``proposed`` row of ``kb-00001..kb-00010``
in project ``p-net``, seven candidates whose Jaccard values span the match
threshold, the near-miss band, the marginal band and zero.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from kb_core import create_sqlite
from kb_core.cluster_ledger import (
    CLUSTER_KEY_HEX_CHARS,
    CLUSTER_LEDGER_MARKER,
    CLUSTER_MATCH_MAX_CANDIDATES,
    CLUSTER_MEMBER_MAX,
    clear_cluster,
    cluster_key_for,
    decline_cluster,
    jaccard,
    match_clusters,
    normalize_members,
    record_cluster,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from kb_core.db.backend import Database
    from kb_core.knowledge_base import KnowledgeBase

LOGGER = "kb_core.cluster_ledger"

_LEDGER_INSERT_SQL = (
    "INSERT INTO map_cluster_ledger"
    " (cluster_key, project_ref, member_entry_ids, last_label, status, sightings,"
    " first_seen_at, last_seen_at, declined_member_count, declined_reason,"
    " declined_by, declined_at)"
    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def _ids(lo: int, hi: int) -> list[str]:
    """The contiguous ascending id block ``kb-<lo:05d>..kb-<hi:05d>``."""
    return [f"kb-{i:05d}" for i in range(lo, hi + 1)]


async def _insert_row(
    db: Database,
    cluster_key: str = "aaaaaaaaaaaaaaaa",
    project_ref: str = "p-net",
    members: list[str] | None = None,
    last_label: str = "VPN and DNS",
    status: str = "proposed",
    sightings: int = 1,
    first_seen_at: str = "2026-03-01T12:00:00+00:00",
    last_seen_at: str = "2026-03-01T12:00:00+00:00",
    declined_member_count: int | None = None,
    declined_reason: str | None = None,
    declined_by: str | None = None,
    declined_at: str | None = None,
) -> None:
    """Hand-insert one raw ledger row (tests seed SQL, never the facade)."""
    raw_members = members if members is not None else _ids(1, 10)
    await db.execute(
        _LEDGER_INSERT_SQL,
        (
            cluster_key,
            project_ref,
            json.dumps(raw_members),
            last_label,
            status,
            sightings,
            first_seen_at,
            last_seen_at,
            declined_member_count,
            declined_reason,
            declined_by,
            declined_at,
        ),
    )
    await db.commit()


async def _seed_proposed_row(db: Database) -> None:
    """The worked fixture: proposed row R (kb-00001..kb-00010, p-net)."""
    await _insert_row(db)


async def _audit_details(db: Database, event_type: str) -> list[str | None]:
    """Every audit detail for *event_type*, in insertion order."""
    cursor = await db.execute(
        "SELECT detail FROM audit_events WHERE event_type = ? ORDER BY id", (event_type,)
    )
    rows = await cursor.fetchall()
    return [row["detail"] for row in rows]


async def _ledger_rows(db: Database, project_ref: str) -> list[tuple]:
    """Raw (cluster_key, sightings, member_entry_ids) rows for one project."""
    cursor = await db.execute(
        "SELECT cluster_key, sightings, member_entry_ids FROM map_cluster_ledger"
        " WHERE project_ref = ?",
        (project_ref,),
    )
    return [tuple(r) for r in await cursor.fetchall()]


@pytest.fixture
async def kb(tmp_path: Path) -> AsyncIterator[KnowledgeBase]:
    """A real SQLite KB with no ledger rows."""
    kb = await create_sqlite(tmp_path / "kb.db")
    try:
        yield kb
    finally:
        await kb.close()


# ---------------------------------------------------------------------------
# Pure helpers (AC-5)
# ---------------------------------------------------------------------------


def test_normalize_members_dedupes_and_sorts() -> None:
    """Dedupe then sort ascending: the stored value and hash input are canonical."""
    assert normalize_members(["b", "a", "b"]) == ("a", "b")
    assert normalize_members([]) == ()


def test_jaccard_empty_sides_return_zero() -> None:
    """An empty candidate matches nothing rather than raising."""
    assert jaccard([], []) == 0.0
    assert jaccard(["a"], []) == 0.0
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)


def test_cluster_key_order_and_duplicate_insensitive() -> None:
    """The key is the hash of the NORMALIZED member set, so order/dupes cannot move it."""
    assert cluster_key_for("p", ["kb-00002", "kb-00001"]) == cluster_key_for(
        "p", ["kb-00001", "kb-00002", "kb-00001"]
    )


def test_cluster_key_contains_project_ref() -> None:
    """The project_ref is inside the hash input — same ids, different project, different key."""
    ids = ["kb-00001", "kb-00002"]
    assert cluster_key_for("p", ids) != cluster_key_for("q", ids)


def test_cluster_key_length_and_hex() -> None:
    """First CLUSTER_KEY_HEX_CHARS chars of a sha256 hexdigest."""
    import string

    key = cluster_key_for("p", ["kb-00001"])
    assert len(key) == CLUSTER_KEY_HEX_CHARS == 16
    assert all(c in string.hexdigits for c in key)


# ---------------------------------------------------------------------------
# AC-13: the seven-candidate worked fixture against a proposed row
# ---------------------------------------------------------------------------


async def test_worked_fixture_proposed_row(kb: KnowledgeBase) -> None:
    """Seven candidates against proposed row R: exact Jaccards and flags."""
    db = kb.db
    await _seed_proposed_row(db)
    candidates = {
        "A": _ids(1, 10),
        "B": _ids(1, 5) + _ids(91, 95),
        "C": _ids(1, 8),
        "D": [*_ids(1, 7), "kb-00091", "kb-00092"],
        "E": _ids(95, 99),
        "F": _ids(1, 20),
        "G": _ids(1, 19),
    }
    result = await match_clusters(db, "p-net", list(candidates.values()))
    assert result.rows_considered == 1
    assert len(result.verdicts) == 7
    v = dict(zip(candidates, result.verdicts, strict=True))

    for key, verdict in v.items():
        assert verdict.suppressed is False, key
        assert verdict.reopened is False, key

    assert v["A"].jaccard == pytest.approx(1.0)
    assert v["A"].matched is not None
    assert v["A"].near_miss is False
    assert v["A"].marginal_match is False

    assert v["B"].jaccard == pytest.approx(5 / 15)
    assert v["B"].matched is None
    assert v["B"].near_miss is True
    assert v["B"].marginal_match is False

    assert v["C"].jaccard == pytest.approx(0.8)
    assert v["C"].matched is not None
    assert v["C"].near_miss is False
    assert v["C"].marginal_match is False

    assert v["D"].jaccard == pytest.approx(7 / 12)
    assert v["D"].matched is not None
    assert v["D"].marginal_match is True

    assert v["E"].jaccard == 0.0
    assert v["E"].matched is None
    assert v["E"].near_miss is False
    assert v["E"].marginal_match is False

    # F sits on the inclusive match boundary AND inside the marginal band.
    assert v["F"].jaccard == 0.5
    assert v["F"].matched is not None
    assert v["F"].marginal_match is True
    assert v["F"].near_miss is False

    assert v["G"].jaccard == pytest.approx(10 / 19)
    assert v["G"].matched is not None
    assert v["G"].marginal_match is True


async def test_proposed_row_does_not_suppress(kb: KnowledgeBase) -> None:
    """AC-10: a proposed row at Jaccard 1.0 matches but never suppresses."""
    db = kb.db
    await _seed_proposed_row(db)
    result = await match_clusters(db, "p-net", [_ids(1, 10)])
    verdict = result.verdicts[0]
    assert verdict.matched is not None
    assert verdict.matched.status == "proposed"
    assert verdict.suppressed is False
    assert verdict.reopened is False


async def test_verdict_echoes_normalized_candidate(kb: KnowledgeBase) -> None:
    """The verdict's member_entry_ids is the NORMALIZED candidate (AC-8)."""
    db = kb.db
    await _seed_proposed_row(db)
    result = await match_clusters(db, "p-net", [["kb-00003", "kb-00001", "kb-00003"]])
    assert result.verdicts[0].member_entry_ids == ("kb-00001", "kb-00003")


async def test_constants_travel_with_result(kb: KnowledgeBase) -> None:
    """The four pinned constants travel with the match result (AC-8)."""
    db = kb.db
    result = await match_clusters(db, "p-net", [_ids(1, 3)])
    assert result.jaccard_threshold == 0.5
    assert result.near_miss_floor == 0.3
    assert result.marginal_match_ceiling == 0.7
    assert result.reopen_growth_factor == 2


# ---------------------------------------------------------------------------
# AC-14: decline / reopen boundaries
# ---------------------------------------------------------------------------


async def _seed_declined_row(db: Database, **kwargs: object) -> None:
    """Row R re-seeded as declined with declined_member_count=10 (AC-14)."""
    await _insert_row(
        db,
        status="declined",
        declined_member_count=10,
        declined_reason="not a subject area",
        declined_by="jason",
        declined_at="2026-03-03T12:00:00+00:00",
        **kwargs,  # type: ignore[arg-type]
    )


async def test_declined_reopen_boundaries(kb: KnowledgeBase) -> None:
    """A, G, F and B against the declined row: the correctness proof for the one escape."""
    db = kb.db
    await _seed_declined_row(db)
    candidates = [_ids(1, 10), _ids(1, 19), _ids(1, 20), _ids(1, 5) + _ids(91, 95)]
    result = await match_clusters(db, "p-net", candidates)
    a, g, f, b = result.verdicts

    assert a.suppressed is True  # 10 >= 20 is false
    assert a.reopened is False
    assert g.suppressed is True  # 19 >= 20 is false — the just-under case
    assert g.reopened is False
    # F sits on BOTH inclusive boundaries at once.
    assert f.reopened is True  # 20 >= 2 * 10
    assert f.suppressed is False
    # A declined cluster does not suppress a candidate that merely resembles it.
    assert b.matched is None
    assert b.suppressed is False
    assert b.near_miss is True


async def test_declined_null_member_count_is_conservative(kb: KnowledgeBase) -> None:
    """A hand-written declined row with NULL declined_member_count never reopens."""
    db = kb.db
    await _insert_row(db, status="declined", declined_member_count=None)
    result = await match_clusters(db, "p-net", [_ids(1, 10)])
    verdict = result.verdicts[0]
    assert verdict.reopened is False
    assert verdict.suppressed is True


async def test_identity_decay_boundary(kb: KnowledgeBase) -> None:
    """A declined 20-member row vs a 5-member subset: Jaccard 0.25, no suppression."""
    db = kb.db
    await _insert_row(
        db,
        members=_ids(1, 20),
        status="declined",
        declined_member_count=20,
        declined_reason="not a subject area",
        declined_by="jason",
        declined_at="2026-03-03T12:00:00+00:00",
    )
    result = await match_clusters(db, "p-net", [_ids(1, 5)])
    verdict = result.verdicts[0]
    assert verdict.jaccard == pytest.approx(0.25)
    assert verdict.matched is None
    assert verdict.suppressed is False


async def test_multi_row_declined_over_proposed_precedence(kb: KnowledgeBase) -> None:
    """AC-9: a decline must never be out-scored by a proposed row (AC-14 group 4)."""
    db = kb.db
    await _insert_row(
        db,
        cluster_key="aaaaaaaaaaaaaaaa",
        status="declined",
        declined_member_count=10,
        declined_reason="not a subject area",
        declined_by="jason",
        declined_at="2026-03-03T12:00:00+00:00",
    )
    await _insert_row(
        db,
        cluster_key="bbbbbbbbbbbbbbbb",
        members=[*_ids(1, 11), "kb-00091"],
        status="proposed",
        last_label="Networking",
        sightings=1,
    )
    result = await match_clusters(db, "p-net", [_ids(1, 10)])
    verdict = result.verdicts[0]
    assert verdict.matched is not None
    assert verdict.matched.status == "declined"
    assert verdict.matched.cluster_key == "aaaaaaaaaaaaaaaa"
    assert verdict.suppressed is True
    assert verdict.jaccard == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AC-9: the exact-Jaccard tie-break — match and record select the SAME row
# ---------------------------------------------------------------------------


async def test_exact_tie_break_and_record_selects_same_row(kb: KnowledgeBase) -> None:
    """Two same-status rows at identical Jaccard: the smaller cluster_key wins twice."""
    db = kb.db
    await _insert_row(db, cluster_key="aaaaaaaaaaaaaaaa", members=_ids(1, 10))
    await _insert_row(db, cluster_key="bbbbbbbbbbbbbbbb", members=_ids(1, 5) + _ids(91, 95))
    candidate = _ids(1, 5)  # 5/10 = 0.5 against BOTH rows

    matched = await match_clusters(db, "p-net", [candidate])
    assert matched.verdicts[0].matched is not None
    assert matched.verdicts[0].matched.cluster_key == "aaaaaaaaaaaaaaaa"

    row = await record_cluster(
        db, "p-net", candidate, "Tie", now=datetime(2026, 3, 2, 12, 0, tzinfo=UTC)
    )
    assert row.cluster_key == "aaaaaaaaaaaaaaaa"
    keys = await _ledger_rows(db, "p-net")
    sightings_by_key = {k: s for k, s, _ in keys}
    assert sightings_by_key["aaaaaaaaaaaaaaaa"] == 2
    assert sightings_by_key["bbbbbbbbbbbbbbbb"] == 1


# ---------------------------------------------------------------------------
# AC-11: Python-side ordering
# ---------------------------------------------------------------------------


async def test_list_clusters_sorts_by_cluster_key(kb: KnowledgeBase) -> None:
    """Three rows inserted in descending key order come back ascending."""
    from kb_core.cluster_ledger import list_clusters

    db = kb.db
    for key in ("cccccccccccccccc", "aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"):
        await _insert_row(db, cluster_key=key)
    rows = await list_clusters(db, "p-net")
    assert [r.cluster_key for r in rows] == [
        "aaaaaaaaaaaaaaaa",
        "bbbbbbbbbbbbbbbb",
        "cccccccccccccccc",
    ]


# ---------------------------------------------------------------------------
# AC-15: the four-step key-drift regression + its corrupt-row variant
# ---------------------------------------------------------------------------


async def test_four_step_key_drift_regression(kb: KnowledgeBase) -> None:
    """The INSERT branch is key-safe: the old 'unreachable' claim is false."""
    db = kb.db
    now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    k = cluster_key_for("p-drift", _ids(1, 10))

    await record_cluster(db, "p-drift", _ids(1, 10), "L1", now=now)
    await record_cluster(db, "p-drift", _ids(1, 20), "L2", now=now)
    await record_cluster(db, "p-drift", _ids(6, 25), "L3", now=now)
    row = await record_cluster(db, "p-drift", _ids(1, 10), "L4", now=now)

    rows = await _ledger_rows(db, "p-drift")
    assert len(rows) == 1
    assert rows[0][0] == k
    assert rows[0][1] == 4
    assert row.cluster_key == k
    assert row.sightings == 4


async def test_key_drift_reachable_via_corrupt_row(kb: KnowledgeBase) -> None:
    """AC-12's fail-open _parse_members reaches the same key-collision branch in one step."""
    db = kb.db
    members = _ids(1, 10)
    k = cluster_key_for("p-x", members)
    await _insert_row(db, project_ref="p-x", cluster_key=k, members=None)
    await db.execute(
        "UPDATE map_cluster_ledger SET member_entry_ids = 'not json' WHERE cluster_key = ?",
        (k,),
    )
    await db.commit()

    row = await record_cluster(
        db, "p-x", members, "Fixed", now=datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    )
    assert row.cluster_key == k
    assert row.sightings == 2
    assert list(row.member_entry_ids) == members
    assert len(await _ledger_rows(db, "p-x")) == 1


# ---------------------------------------------------------------------------
# AC-16: the record / decline / clear lifecycle
# ---------------------------------------------------------------------------


async def test_lifecycle(kb: KnowledgeBase) -> None:
    """Seven steps pinning: identity is overlap, a sighting never clears a decline."""
    db = kb.db
    step1 = await record_cluster(
        db, "p-net", _ids(1, 10), "VPN and DNS", now=datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    )
    assert step1.status == "proposed"
    assert step1.sightings == 1
    assert step1.first_seen_at == step1.last_seen_at
    assert step1.declined_member_count is None
    assert step1.declined_reason is None
    assert step1.declined_by is None
    assert step1.declined_at is None
    key1 = step1.cluster_key

    step2 = await record_cluster(
        db,
        "p-net",
        [*_ids(1, 8), "kb-00091", "kb-00092"],
        "Networking",
        now=datetime(2026, 3, 2, 12, 0, tzinfo=UTC),
    )
    assert step2.cluster_key == key1  # identity is member-set overlap, not the exact set
    assert list(step2.member_entry_ids) == [*_ids(1, 8), "kb-00091", "kb-00092"]
    assert step2.last_label == "Networking"
    assert step2.sightings == 2
    assert step2.first_seen_at.startswith("2026-03-01")
    assert step2.last_seen_at.startswith("2026-03-02")
    assert cluster_key_for("p-net", step2.member_entry_ids) != step2.cluster_key

    step3 = await decline_cluster(
        db,
        key1,
        reason="not a subject area",
        declined_by="jason",
        now=datetime(2026, 3, 3, 12, 0, tzinfo=UTC),
    )
    assert step3 is not None
    assert step3.status == "declined"
    assert step3.declined_member_count == 10
    assert step3.declined_by == "jason"
    assert step3.declined_at == "2026-03-03T12:00:00+00:00"

    step4 = await record_cluster(
        db,
        "p-net",
        [*_ids(1, 8), "kb-00091", "kb-00092"],
        "Networking",
        now=datetime(2026, 3, 4, 12, 0, tzinfo=UTC),
    )
    assert step4.sightings == 3
    assert step4.last_seen_at.startswith("2026-03-04")
    assert step4.status == "declined"
    assert step4.declined_member_count == 10
    assert step4.declined_reason == "not a subject area"
    assert step4.declined_by == "jason"
    assert step4.declined_at == "2026-03-03T12:00:00+00:00"

    assert await clear_cluster(db, key1) is True
    assert await clear_cluster(db, key1) is False

    assert len(await _audit_details(db, "map_cluster_ledger_recorded")) == 3
    declined = await _audit_details(db, "map_cluster_ledger_declined")
    assert len(declined) == 1
    assert declined[0] is not None and "was=proposed" in declined[0]
    assert len(await _audit_details(db, "map_cluster_ledger_cleared")) == 1

    fresh = await record_cluster(
        db, "p-net", _ids(301, 310), "Fresh", now=datetime(2026, 3, 5, 12, 0, tzinfo=UTC)
    )
    assert fresh.sightings == 1
    fresh_again = await record_cluster(
        db, "p-net", _ids(301, 310), "Fresh", now=datetime(2026, 3, 6, 12, 0, tzinfo=UTC)
    )
    assert fresh_again.sightings == 2
    assert fresh_again.status == "proposed"


# ---------------------------------------------------------------------------
# AC-17: instrumentation
# ---------------------------------------------------------------------------


async def test_near_miss_only_candidate_writes_one_band_row(kb: KnowledgeBase) -> None:
    """A batch whose only candidate is B writes exactly one near-miss audit row."""
    db = kb.db
    await _seed_proposed_row(db)
    await match_clusters(db, "p-net", [_ids(1, 5) + _ids(91, 95)])
    near = await _audit_details(db, "map_cluster_near_miss")
    assert len(near) == 1
    assert near[0] is not None and "jaccard=0.3333" in near[0]
    assert await _audit_details(db, "map_cluster_marginal_match") == []


async def test_marginal_only_candidate_writes_one_band_row(kb: KnowledgeBase) -> None:
    """A batch whose only candidate is D writes exactly one marginal-match audit row."""
    db = kb.db
    await _seed_proposed_row(db)
    await match_clusters(db, "p-net", [[*_ids(1, 7), "kb-00091", "kb-00092"]])
    marginal = await _audit_details(db, "map_cluster_marginal_match")
    assert len(marginal) == 1
    assert marginal[0] is not None and "jaccard=0.5833" in marginal[0]
    assert await _audit_details(db, "map_cluster_near_miss") == []


@pytest.mark.parametrize("candidate", [_ids(1, 10), _ids(95, 99)])
async def test_far_candidates_write_no_band_rows(kb: KnowledgeBase, candidate: list[str]) -> None:
    """Jaccard 1.0 and 0.0 against a proposed row write zero band rows."""
    db = kb.db
    await _seed_proposed_row(db)
    await match_clusters(db, "p-net", [candidate])
    assert await _audit_details(db, "map_cluster_near_miss") == []
    assert await _audit_details(db, "map_cluster_marginal_match") == []


async def test_full_batch_band_row_counts(kb: KnowledgeBase) -> None:
    """The seven-candidate batch writes exactly 1 near-miss and 3 marginal rows."""
    db = kb.db
    await _seed_proposed_row(db)
    await match_clusters(
        db,
        "p-net",
        [
            _ids(1, 10),
            _ids(1, 5) + _ids(91, 95),
            _ids(1, 8),
            [*_ids(1, 7), "kb-00091", "kb-00092"],
            _ids(95, 99),
            _ids(1, 20),
            _ids(1, 19),
        ],
    )
    assert len(await _audit_details(db, "map_cluster_near_miss")) == 1
    assert len(await _audit_details(db, "map_cluster_marginal_match")) == 3


async def test_suppressed_candidate_writes_one_row(kb: KnowledgeBase) -> None:
    """A suppressed candidate writes exactly one map_cluster_suppressed row."""
    db = kb.db
    await _seed_declined_row(db)
    await match_clusters(db, "p-net", [_ids(1, 10)])
    suppressed = await _audit_details(db, "map_cluster_suppressed")
    assert len(suppressed) == 1
    assert suppressed[0] is not None
    assert "declined_member_count=10" in suppressed[0]
    assert "reopen_at=20" in suppressed[0]
    assert await _audit_details(db, "map_cluster_reopened") == []


async def test_reopened_candidate_writes_one_row(kb: KnowledgeBase) -> None:
    """A reopened candidate writes exactly one map_cluster_reopened row."""
    db = kb.db
    await _seed_declined_row(db)
    await match_clusters(db, "p-net", [_ids(1, 20)])
    reopened = await _audit_details(db, "map_cluster_reopened")
    assert len(reopened) == 1
    assert reopened[0] is not None and "declined_reason=" in reopened[0]
    assert await _audit_details(db, "map_cluster_suppressed") == []


async def test_proposed_stalled_warns_on_third_sighting(kb: KnowledgeBase) -> None:
    """Three successive records on one member set: the warning lands on the third only."""
    db = kb.db
    now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    for _ in range(3):
        await record_cluster(db, "p-net", _ids(1, 10), "VPN and DNS", now=now)
    stalled = await _audit_details(db, "map_cluster_proposed_stalled")
    assert len(stalled) == 1
    assert stalled[0] is not None and "sightings=3" in stalled[0]


async def test_declined_row_never_emits_proposed_stalled(kb: KnowledgeBase) -> None:
    """A declined row at the same sightings count emits no proposed-stalled warning."""
    db = kb.db
    now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    for _ in range(2):
        await record_cluster(db, "p-net", _ids(1, 10), "VPN and DNS", now=now)
    await decline_cluster(db, (await _ledger_rows(db, "p-net"))[0][0], reason="r", now=now)
    for _ in range(2):
        await record_cluster(db, "p-net", _ids(1, 10), "VPN and DNS", now=now)
    assert await _audit_details(db, "map_cluster_proposed_stalled") == []


async def test_two_records_before_warn_threshold(kb: KnowledgeBase) -> None:
    """The first two sightings of a proposed cluster write zero stalled rows."""
    db = kb.db
    now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    await record_cluster(db, "p-net", _ids(1, 10), "VPN and DNS", now=now)
    assert await _audit_details(db, "map_cluster_proposed_stalled") == []
    await record_cluster(db, "p-net", _ids(1, 10), "VPN and DNS", now=now)
    assert await _audit_details(db, "map_cluster_proposed_stalled") == []


async def test_match_clusters_never_writes_ledger_rows(
    kb: KnowledgeBase,
) -> None:
    """match_clusters writes audit_events rows and NEVER a map_cluster_ledger row."""
    db = kb.db
    await _seed_proposed_row(db)
    before = await _ledger_rows(db, "p-net")
    await match_clusters(
        db,
        "p-net",
        [_ids(1, 5) + _ids(91, 95), [*_ids(1, 7), "kb-00091", "kb-00092"]],
    )
    assert await _ledger_rows(db, "p-net") == before


# ---------------------------------------------------------------------------
# AC-18: the decline-clobber tripwire
# ---------------------------------------------------------------------------


async def test_decline_clobber_tripwire_fires(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A patched UPDATE that un-declines the row trips the tripwire exactly once."""
    db = kb.db
    now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    await record_cluster(db, "p-net", _ids(1, 10), "VPN and DNS", now=now)
    key = (await _ledger_rows(db, "p-net"))[0][0]
    await decline_cluster(db, key, reason="not a subject area", now=now)

    original_execute = db.execute

    async def _clobbering_execute(sql: str, params: object = ()) -> object:
        if sql.startswith("UPDATE map_cluster_ledger"):
            sql = sql.replace("SET member_entry_ids", "SET status = 'proposed', member_entry_ids")
        return await original_execute(sql, params)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(db, "execute", _clobbering_execute)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        row = await record_cluster(db, "p-net", _ids(1, 8), "X", now=now)
    monkeypatch.undo()

    errors = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and CLUSTER_LEDGER_MARKER in r.getMessage()
    ]
    assert len(errors) == 1
    assert "invariant=decline_clobbered" in errors[0].getMessage()
    clobbered = await _audit_details(db, "map_cluster_decline_clobbered")
    assert len(clobbered) == 1
    assert row.status == "proposed"  # the OBSERVED row, returned unchanged


async def test_no_clobber_on_the_healthy_path(
    kb: KnowledgeBase, caplog: pytest.LogCaptureFixture
) -> None:
    """A healthy update writes zero clobber audit rows."""
    db = kb.db
    now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    await record_cluster(db, "p-net", _ids(1, 10), "VPN and DNS", now=now)
    key = (await _ledger_rows(db, "p-net"))[0][0]
    await decline_cluster(db, key, reason="r", now=now)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        await record_cluster(db, "p-net", _ids(1, 8), "X", now=now)
    assert await _audit_details(db, "map_cluster_decline_clobbered") == []
    assert [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and CLUSTER_LEDGER_MARKER in r.getMessage()
    ] == []


# ---------------------------------------------------------------------------
# AC-12: corrupt-row parsing (fail-open)
# ---------------------------------------------------------------------------


async def test_garbage_status_parses_to_proposed_and_never_suppresses(
    kb: KnowledgeBase,
) -> None:
    """An unrecognised status parses to proposed — the fail direction is deliberate."""
    db = kb.db
    await _insert_row(db, status="garbage")
    from kb_core.cluster_ledger import list_clusters

    rows = await list_clusters(db, "p-net")
    assert rows[0].status == "proposed"
    result = await match_clusters(db, "p-net", [_ids(1, 10)])
    verdict = result.verdicts[0]
    assert verdict.matched is not None
    assert verdict.suppressed is False
    assert verdict.reopened is False


async def test_unparseable_members_yield_jaccard_zero(
    kb: KnowledgeBase, caplog: pytest.LogCaptureFixture
) -> None:
    """A corrupt member set parses to (), yields Jaccard 0.0 and warns once — no raise."""
    db = kb.db
    await _insert_row(db, cluster_key="dddddddddddddddd")
    await db.execute(
        "UPDATE map_cluster_ledger SET member_entry_ids = 'not json'"
        " WHERE cluster_key = 'dddddddddddddddd'"
    )
    await db.commit()

    from kb_core.cluster_ledger import list_clusters

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        rows = await list_clusters(db, "p-net")
    assert rows[0].member_entry_ids == ()
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and CLUSTER_LEDGER_MARKER in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "dddddddddddddddd" in warnings[0].getMessage()

    result = await match_clusters(db, "p-net", [_ids(1, 10)])
    assert result.verdicts[0].jaccard == 0.0
    assert result.verdicts[0].matched is None
    assert result.verdicts[0].suppressed is False


# ---------------------------------------------------------------------------
# Cross-project isolation
# ---------------------------------------------------------------------------


async def test_row_in_project_a_never_matches_project_b(kb: KnowledgeBase) -> None:
    """Only project B's rows are loaded, so identical members do not cross-match."""
    db = kb.db
    await _seed_proposed_row(db)  # in p-net
    result = await match_clusters(db, "p-other", [_ids(1, 10)])
    assert result.rows_considered == 0
    verdict = result.verdicts[0]
    assert verdict.matched is None
    assert verdict.jaccard == 0.0
    assert verdict.suppressed is False


# ---------------------------------------------------------------------------
# Size / batch boundaries and ValueError guards
# ---------------------------------------------------------------------------


async def test_exactly_member_max_accepted(kb: KnowledgeBase) -> None:
    """1200 normalized members: record inserts, match returns oversized=False."""
    db = kb.db
    members = [f"kb-{i:05d}" for i in range(1, CLUSTER_MEMBER_MAX + 1)]
    row = await record_cluster(db, "p-big", members, "Big", now=datetime(2026, 3, 1, tzinfo=UTC))
    assert len(row.member_entry_ids) == CLUSTER_MEMBER_MAX
    result = await match_clusters(db, "p-big", [members])
    assert result.verdicts[0].oversized is False
    assert result.verdicts[0].matched is not None


async def test_member_max_plus_one(kb: KnowledgeBase) -> None:
    """1201 distinct members: oversized verdict from match, ValueError from record."""
    db = kb.db
    members = [f"kb-{i:05d}" for i in range(1, CLUSTER_MEMBER_MAX + 2)]
    result = await match_clusters(db, "p-big", [members])
    verdict = result.verdicts[0]
    assert verdict.oversized is True
    assert verdict.jaccard == 0.0
    assert verdict.matched is None
    assert verdict.suppressed is False
    assert verdict.reopened is False
    assert verdict.near_miss is False
    assert verdict.marginal_match is False

    with pytest.raises(ValueError, match="1200"):
        await record_cluster(db, "p-big", members, "Big", now=datetime(2026, 3, 1, tzinfo=UTC))


async def test_duplicates_do_not_count_toward_the_max(kb: KnowledgeBase) -> None:
    """1205 raw ids of which 10 are duplicates (1195 distinct) are accepted by both."""
    db = kb.db
    raw = [f"kb-{i:05d}" for i in range(1, CLUSTER_MEMBER_MAX - 4)]
    raw = raw + raw[:10]
    assert len(raw) == 1205
    assert len(set(raw)) == 1195
    row = await record_cluster(db, "p-dup", raw, "Dupes", now=datetime(2026, 3, 1, tzinfo=UTC))
    assert len(row.member_entry_ids) == 1195
    result = await match_clusters(db, "p-dup", [raw])
    assert result.verdicts[0].oversized is False


async def test_candidate_batch_boundaries(kb: KnowledgeBase) -> None:
    """Exactly 200 candidates accepted; 201 raises ValueError before any read."""
    db = kb.db
    batch = [[f"kb-{i:05d}", f"kb-{i + 1:05d}"] for i in range(1, CLUSTER_MATCH_MAX_CANDIDATES + 1)]
    assert len(batch) == 200
    result = await match_clusters(db, "p-net", batch)
    assert len(result.verdicts) == 200
    with pytest.raises(ValueError, match="200"):
        await match_clusters(db, "p-net", [*batch, ["kb-99999"]])


async def test_empty_candidate_raises(kb: KnowledgeBase) -> None:
    """An empty candidate is a caller bug, not a shape."""
    db = kb.db
    with pytest.raises(ValueError, match="empty"):
        await match_clusters(db, "p-net", [[]])


async def test_empty_member_set_raises_on_record(kb: KnowledgeBase) -> None:
    """record_cluster raises ValueError on an empty member set (422 is the honest answer)."""
    db = kb.db
    with pytest.raises(ValueError, match="empty"):
        await record_cluster(db, "p-net", [], "L", now=datetime(2026, 3, 1, tzinfo=UTC))


async def test_oversized_candidate_logs_warning(
    kb: KnowledgeBase, caplog: pytest.LogCaptureFixture
) -> None:
    """An oversized candidate is never batch-fatal — one warning, a normal verdict."""
    db = kb.db
    members = [f"kb-{i:05d}" for i in range(1, CLUSTER_MEMBER_MAX + 2)]
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        await match_clusters(db, "p-net", [members, _ids(1, 3)])
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "oversized-candidate" in r.getMessage()
    ]
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# Facade pass-through (AC-19 / AC-21)
# ---------------------------------------------------------------------------


async def test_facade_pass_through(kb: KnowledgeBase) -> None:
    """The facade supplies its own now and threads declined_by verbatim."""
    row = await kb.record_cluster("p-f", _ids(1, 10), "VPN and DNS")
    assert row.status == "proposed"
    assert row.sightings == 1
    parsed = datetime.fromisoformat(row.first_seen_at)
    assert parsed.tzinfo is not None  # the facade supplied an aware `now`

    result = await kb.match_clusters("p-f", [_ids(1, 5) + _ids(91, 95)])
    assert result.verdicts[0].near_miss is True

    rows = await kb.cluster_ledger("p-f")
    assert [r.cluster_key for r in rows] == [row.cluster_key]

    declined = await kb.decline_cluster(row.cluster_key, reason="not a subject area")
    assert declined is not None
    assert declined.declined_by is None  # no config.contributor fallback, by design

    explicit = await kb.record_cluster("p-g", _ids(50, 59), "G")
    declined_g = await kb.decline_cluster(explicit.cluster_key, reason="r", declined_by="jason")
    assert declined_g is not None
    assert declined_g.declined_by == "jason"  # threaded VERBATIM

    assert await kb.clear_cluster(explicit.cluster_key) is True
    assert await kb.clear_cluster(explicit.cluster_key) is False
    assert await kb.decline_cluster("no-such-key", reason="r") is None
