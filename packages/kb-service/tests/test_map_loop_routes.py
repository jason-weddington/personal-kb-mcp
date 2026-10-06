"""Hermetic tests for GET /api/kb/map-loop-input (map_loop_routes.py).

No live Postgres, no Ollama, no network, and no change to ``conftest.py``:
every case here is testable with the existing fakes. All three statements
MUST be registered in ``fake_kb.db.rows_for`` before the request —
``FakeKbDb.execute`` checks ``rows_for`` first and otherwise falls through to
a ``graph_edges`` branch, then a ``knowledge_vec`` branch, then a
``mental_map`` branch, and ALL THREE of this route's statements contain at
least one of those substrings (the entries statement contains both
``graph_edges`` via the anti-join and ``mental_map`` via the mappable
predicate; the pair statement contains ``knowledge_vec``; the map-bodies
statement contains ``mental_map``), so an unregistered statement silently
returns the wrong arity and breaks the unpack. The three needles are
deliberately disjoint.
"""

import logging
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from kb_core.map_eligibility import MapEligibilityOverride
from kb_core.map_lint import lint_map_body

import kb_service.routes.map_loop_routes as map_loop_routes
from kb_service.auth import get_current_user
from kb_service.main import app
from tests.conftest import (
    FakeKbDb,
    FakeKnowledgeBase,
    fake_user,
    make_map_eligibility_verdict,
)

MARKER = "map-loop-input"
LOGGER = "kb_service.routes.map_loop_routes"

ENTRIES_NEEDLE = "substr("
BODIES_NEEDLE = "knowledge_details, contributor"
PAIRS_NEEDLE = "<=>"

PROJ = "proj"
URL = "/api/kb/map-loop-input"

# The AC-13 worked fixture, verbatim: ten in-band pairs plus the (2,5)=0.45
# pair sitting in the observation band below POCKET_MIN_SIMILARITY.
WORKED_PAIR_ROWS: list[tuple[Any, ...]] = [
    ("kb-00001", "kb-00002", 0.90),
    ("kb-00001", "kb-00003", 0.88),
    ("kb-00002", "kb-00003", 0.86),
    ("kb-00004", "kb-00005", 0.92),
    ("kb-00004", "kb-00006", 0.91),
    ("kb-00004", "kb-00007", 0.90),
    ("kb-00005", "kb-00006", 0.89),
    ("kb-00005", "kb-00007", 0.88),
    ("kb-00006", "kb-00007", 0.87),
    ("kb-00003", "kb-00004", 0.58),
    ("kb-00002", "kb-00005", 0.45),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _marker_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return every captured record whose message carries the route marker."""
    return [r for r in caplog.records if MARKER in r.getMessage()]


def _entry_row(
    entry_id: str,
    *,
    tags: Any = "net",
    excerpt: str = "body",
    details_length: int = 4000,
    unpointed: int = 1,
    short_title: str = "T1",
    details: str | None = None,
) -> tuple[Any, ...]:
    """One entries-statement row in the pinned SELECT order.

    Column 9 is the entry's FULL knowledge_details — the source the
    directory_tokens half extracts from — stubbed independently of the
    excerpt/details_length columns so a test can pin that extraction reads
    past the 600-char excerpt.
    """
    return (
        entry_id,
        short_title,
        f"{entry_id} long",
        "factual_reference",
        tags,
        excerpt,
        details_length,
        unpointed,
        details,
    )


def _eligible_verdict(ref: str = PROJ) -> Any:
    """A verdict whose computed evidence is eligible (no override needed)."""
    return make_map_eligibility_verdict(
        ref, mappable=50, ingested=0, top_prefix_count=5, top_prefix="Run"
    )


def _wire(
    fake_kb: FakeKnowledgeBase,
    *,
    verdicts: list[Any],
    entry_rows: list[tuple[Any, ...]] | None = None,
    body_rows: list[tuple[Any, ...]] | None = None,
    pair_rows: list[tuple[Any, ...]] | None = None,
    maps: dict[str, list[dict[str, Any]]] | None = None,
) -> None:
    """Point the route's dependencies at the fakes and register all three
    statements' row sets (insertion-order first-match-wins, disjoint needles).
    """
    app.dependency_overrides[get_current_user] = fake_user
    fake_kb.map_eligibility_verdicts = verdicts
    fake_kb.maps_projects = maps or {}
    fake_kb.db.rows_for[ENTRIES_NEEDLE] = entry_rows or []
    fake_kb.db.rows_for[BODIES_NEEDLE] = body_rows or []
    fake_kb.db.rows_for[PAIRS_NEEDLE] = pair_rows or []


# ---------------------------------------------------------------------------
# Auth (kb-01745: HTTPBearer auto_error=False -> 401, not 403)
# ---------------------------------------------------------------------------


def test_map_loop_input_requires_auth_401(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No Authorization header -> 401, NOT 403 (plain non-admin route)."""
    monkeypatch.delenv("KB_AUTH_MODE", raising=False)
    assert client.get(URL, params={"project_ref": PROJ}).status_code == 401


# ---------------------------------------------------------------------------
# Admission gate: 404 / 409, and NO route-issued SQL on either branch
# ---------------------------------------------------------------------------


def test_unknown_project_ref_404_and_zero_sql(
    client: TestClient, caplog: pytest.LogCaptureFixture, fake_kb: FakeKnowledgeBase
) -> None:
    """A ref with no verdict -> 404 and the route issued NO SQL at all."""
    _wire(fake_kb, verdicts=[_eligible_verdict()])
    with caplog.at_level(logging.INFO, logger=LOGGER):
        resp = client.get(URL, params={"project_ref": "ghost"})
    assert resp.status_code == 404
    assert resp.json() == {"detail": "project_ref not found"}
    assert fake_kb.db.calls == []
    records = _marker_records(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert "outcome=not_found" in records[0].getMessage()


def test_ineligible_ref_409_and_zero_sql(
    client: TestClient, caplog: pytest.LogCaptureFixture, fake_kb: FakeKnowledgeBase
) -> None:
    """A computed-ineligible verdict (journal shape) -> 409, no SQL issued."""
    verdict = make_map_eligibility_verdict(
        "journal-proj", mappable=624, ingested=0, top_prefix_count=611, top_prefix="Run"
    )
    assert verdict.evidence.is_journal is True
    assert verdict.effective_eligible is False
    _wire(fake_kb, verdicts=[verdict])
    with caplog.at_level(logging.INFO, logger=LOGGER):
        resp = client.get(URL, params={"project_ref": "journal-proj"})
    assert resp.status_code == 409
    assert resp.json() == {"detail": "project_ref not map-eligible"}
    assert fake_kb.db.calls == []
    records = _marker_records(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    msg = records[0].getMessage()
    assert "outcome=ineligible" in msg
    assert "decided_by=computed" in msg
    assert "computed_eligible=False" in msg
    assert "mappable=624" in msg
    assert "hand_authored=624" in msg
    assert "is_journal=True" in msg


def test_human_override_forces_ineligible_ref_eligible(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """An override with eligible=True on an ineligible ref is honoured."""
    override = MapEligibilityOverride(
        "journal-proj", True, "human verdict", "jason", "2026-09-20T12:00:00+00:00"
    )
    verdict = make_map_eligibility_verdict(
        "journal-proj",
        mappable=624,
        ingested=0,
        top_prefix_count=611,
        top_prefix="Run",
        override=override,
    )
    assert verdict.effective_eligible is True
    _wire(
        fake_kb,
        verdicts=[verdict],
        entry_rows=[_entry_row("kb-00001"), _entry_row("kb-00002")],
        pair_rows=[],
    )
    resp = client.get(URL, params={"project_ref": "journal-proj"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["pockets_omitted_reason"] is None


def test_orphaned_override_forced_eligible_is_200_not_error(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """Zero mappable entries + eligible override -> 200, empty halves, too-few."""
    override = MapEligibilityOverride(
        PROJ, True, "r", None, "2026-09-20T12:00:00+00:00"
    )
    verdict = make_map_eligibility_verdict(
        PROJ,
        mappable=0,
        ingested=0,
        top_prefix_count=0,
        top_prefix="",
        maps=1,
        override=override,
        orphaned=True,
    )
    assert verdict.effective_eligible is True
    _wire(
        fake_kb,
        verdicts=[verdict],
        body_rows=[("map-1", "Lives in proj.", "jason", None)],
        maps={
            PROJ: [
                {
                    "id": "map-1",
                    "short_title": "M",
                    "long_title": "M long",
                    "pointers": [],
                }
            ]
        },
    )
    resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    body = resp.json()
    assert body["entries"] == []
    assert body["maps"] != []
    assert body["pockets"] == []
    assert body["pockets_omitted_reason"] == "too-few-unpointed-entries"
    assert len(fake_kb.db.calls) == 2


# ---------------------------------------------------------------------------
# AC-13: the worked fixture — the correctness proof for the pocket builder
# ---------------------------------------------------------------------------


def _wire_worked_fixture(fake_kb: FakeKnowledgeBase) -> None:
    """Seven unpointed entries kb-00001..kb-00007 plus the stubbed pair rows."""
    _wire(
        fake_kb,
        verdicts=[_eligible_verdict()],
        entry_rows=[
            _entry_row(f"kb-{i:05d}", tags="net dns", excerpt=f"excerpt-{i}")
            for i in range(1, 8)
        ],
        pair_rows=WORKED_PAIR_ROWS,
    )


def test_worked_fixture_exactly_two_pockets(
    client: TestClient, caplog: pytest.LogCaptureFixture, fake_kb: FakeKnowledgeBase
) -> None:
    """Mutual-top-K separates the two subject areas; the 0.58 bridge is not
    mutual (kb-00003 is only kb-00004's 4th-best neighbour) and the 0.45 pair
    is sub-threshold — neither appears in any pocket."""
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _wire_worked_fixture(fake_kb)
        resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    body = resp.json()
    assert body["pockets_omitted_reason"] is None
    assert body["excerpt_chars"] == map_loop_routes.LOOP_INPUT_EXCERPT_CHARS
    pockets = body["pockets"]
    assert len(pockets) == 2

    first = pockets[0]
    assert first["member_entry_ids"] == [
        "kb-00004",
        "kb-00005",
        "kb-00006",
        "kb-00007",
    ]
    assert first["mean_similarity"] == pytest.approx(5.37 / 6)
    assert first["min_similarity"] == pytest.approx(0.87)
    assert first["max_similarity"] == pytest.approx(0.92)
    assert len(first["edges"]) == 6

    second = pockets[1]
    assert second["member_entry_ids"] == ["kb-00001", "kb-00002", "kb-00003"]
    assert second["mean_similarity"] == pytest.approx(2.64 / 3)
    assert second["min_similarity"] == pytest.approx(0.86)
    assert second["max_similarity"] == pytest.approx(0.90)
    assert len(second["edges"]) == 3

    # The bridge: 0.58 clears POCKET_MIN_SIMILARITY but is NOT mutual, and
    # that non-mutuality is exactly what keeps the two subject areas apart.
    for pocket in pockets:
        for edge in pocket["edges"]:
            assert {edge["a"], edge["b"]} != {"kb-00003", "kb-00004"}

    # The (2,5)=0.45 pair is below POCKET_MIN_SIMILARITY: no pocket, no
    # neighbour list, and it IS counted in the log's below-threshold counter.
    for pocket in pockets:
        for edge in pocket["edges"]:
            assert {edge["a"], edge["b"]} != {"kb-00002", "kb-00005"}
    infos = [r for r in _marker_records(caplog) if r.levelno == logging.INFO]
    assert len(infos) == 1
    msg = infos[0].getMessage()
    assert "pairs_below_threshold=1" in msg
    assert "max_similarity_below_threshold=0.4500" in msg

    # Bind parameters: statement 1 carries exactly (project_ref,); statement 3
    # carries exactly the unpointed id list, and its SQL has one ? per id.
    assert fake_kb.db.calls[0][1] == (PROJ,)
    assert fake_kb.db.calls[1][1] == (PROJ,)
    pair_sql, pair_params = fake_kb.db.calls[2]
    assert pair_params == tuple(f"kb-{i:05d}" for i in range(1, 8))
    assert pair_sql.count("?") == 7


def test_pocket_object_field_names_exact(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """AC-6: the pocket key set is the contract of record, no label field."""
    _wire_worked_fixture(fake_kb)
    resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {
        "project_ref",
        "excerpt_chars",
        "entries",
        "maps",
        "pockets",
        "pockets_omitted_reason",
    }
    assert set(body["pockets"][0].keys()) == {
        "member_entry_ids",
        "mean_similarity",
        "min_similarity",
        "max_similarity",
        "edges",
    }
    assert set(body["pockets"][0]["edges"][0].keys()) == {"a", "b", "similarity"}


# ---------------------------------------------------------------------------
# Deterministic ordering (prompt-prefix stability)
# ---------------------------------------------------------------------------


def test_deterministic_ordering_byte_identical_repeat(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """Two successive calls with the same stubs serialize BYTE-identically."""
    maps = {
        PROJ: [
            {
                "id": "map-2",
                "short_title": "M2",
                "long_title": "M2 long",
                "pointers": ["kb-00009"],
            },
            {
                "id": "map-1",
                "short_title": "M1",
                "long_title": "M1 long",
                "pointers": [],
            },
        ]
    }
    rows = [
        _entry_row("kb-00002", tags="a b"),
        _entry_row("kb-00001", tags="a"),
        _entry_row("kb-00003", tags=""),
    ]
    _wire(
        fake_kb,
        verdicts=[_eligible_verdict()],
        entry_rows=rows,
        body_rows=[("map-2", "two", None, None)],
        pair_rows=[
            ("kb-00001", "kb-00002", 0.90),
            ("kb-00001", "kb-00003", 0.88),
            ("kb-00002", "kb-00003", 0.86),
        ],
        maps=maps,
    )
    resp1 = client.get(URL, params={"project_ref": PROJ})
    resp2 = client.get(URL, params={"project_ref": PROJ})
    assert resp1.status_code == 200
    # BYTE-level comparison: response.json() dict equality is insensitive to
    # key order and would pass a response whose serialization order moved.
    assert resp1.content == resp2.content

    body = resp1.json()
    assert [e["id"] for e in body["entries"]] == [
        "kb-00001",
        "kb-00002",
        "kb-00003",
    ]
    assert [m["id"] for m in body["maps"]] == ["map-1", "map-2"]
    pockets = body["pockets"]
    assert [p["member_entry_ids"] for p in pockets] == [
        ["kb-00001", "kb-00002", "kb-00003"],
    ]
    assert [(e["a"], e["b"]) for e in pockets[0]["edges"]] == [
        ("kb-00001", "kb-00002"),
        ("kb-00001", "kb-00003"),
        ("kb-00002", "kb-00003"),
    ]


# ---------------------------------------------------------------------------
# pockets_omitted_reason — all four values
# ---------------------------------------------------------------------------


def test_too_few_unpointed_entries_skips_pairs(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """Fewer than POCKET_MIN_SIZE unpointed entries -> no pair statement."""
    _wire(fake_kb, verdicts=[_eligible_verdict()], entry_rows=[_entry_row("kb-00001")])
    resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    body = resp.json()
    assert body["pockets"] == []
    assert body["pockets_omitted_reason"] == "too-few-unpointed-entries"
    assert len(fake_kb.db.calls) == 2
    assert all(PAIRS_NEEDLE not in sql for sql, _ in fake_kb.db.calls)


def test_unpointed_set_too_large_skips_pairs(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """More than POCKET_MAX_UNPOINTED unpointed entries -> no pair statement."""
    rows = [_entry_row(f"kb-{i:05d}") for i in range(1, 302)]
    _wire(fake_kb, verdicts=[_eligible_verdict()], entry_rows=rows)
    resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["entries"]) == 301
    assert body["pockets"] == []
    assert body["pockets_omitted_reason"] == "unpointed-set-too-large"
    assert len(fake_kb.db.calls) == 2
    assert all(PAIRS_NEEDLE not in sql for sql, _ in fake_kb.db.calls)


def test_non_postgres_backend_skips_pairs(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """SQLite data DB -> entries/maps still returned, pockets skipped with a
    WARNING naming KB_DATABASE_URL."""
    monkeypatch.setattr(map_loop_routes, "SQLiteBackend", FakeKbDb)
    _wire(
        fake_kb,
        verdicts=[_eligible_verdict()],
        entry_rows=[_entry_row("kb-00001"), _entry_row("kb-00002")],
        pair_rows=[("kb-00001", "kb-00002", 0.9)],
    )
    with caplog.at_level(logging.INFO, logger=LOGGER):
        resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    body = resp.json()
    assert body["pockets_omitted_reason"] == "non-postgres-backend"
    assert body["pockets"] == []
    assert body["entries"]
    assert all(PAIRS_NEEDLE not in sql for sql, _ in fake_kb.db.calls)
    assert len(fake_kb.db.calls) == 2
    warnings = [r for r in _marker_records(caplog) if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "KB_DATABASE_URL" in warnings[0].getMessage()


# ---------------------------------------------------------------------------
# Field derivations (AC-21)
# ---------------------------------------------------------------------------


def test_tags_excerpt_details_length_derivations(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """tags splits on whitespace (never json.loads); excerpt/dlength pass."""
    _wire(
        fake_kb,
        verdicts=[_eligible_verdict()],
        entry_rows=[
            _entry_row("kb-00001", tags="", excerpt="abcde", details_length=5),
            _entry_row("kb-00002", tags="   ", excerpt="x" * 600, details_length=4000),
            _entry_row("kb-00003", tags="a  b"),
            _entry_row("kb-00004", tags=None),
        ],
        pair_rows=[],
    )
    resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    by_id = {e["id"]: e for e in resp.json()["entries"]}
    assert by_id["kb-00001"]["tags"] == []
    assert by_id["kb-00002"]["tags"] == []
    assert by_id["kb-00003"]["tags"] == ["a", "b"]
    assert by_id["kb-00004"]["tags"] == []

    assert by_id["kb-00001"]["excerpt"] == "abcde"
    assert by_id["kb-00001"]["details_length"] == 5
    assert by_id["kb-00002"]["excerpt"] == "x" * 600
    assert not by_id["kb-00002"]["excerpt"].endswith("…")
    assert by_id["kb-00002"]["details_length"] == 4000
    assert by_id["kb-00001"]["entry_type"] == "factual_reference"
    assert by_id["kb-00001"]["unpointed"] is True


# ---------------------------------------------------------------------------
# Maps half (AC-19)
# ---------------------------------------------------------------------------


def test_map_body_round_trips_verbatim_and_missing_body_is_empty(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """body round-trips knowledge_details verbatim (no truncation); a map
    present in one result and not the other yields body="" / pointers=[]"""
    full_body = "Lives in proj.\n\nDetail entries:\n- kb-00001 gloss\n" * 40
    _wire(
        fake_kb,
        verdicts=[_eligible_verdict()],
        entry_rows=[_entry_row("kb-00001"), _entry_row("kb-00002")],
        body_rows=[("map-1", full_body, "kb-machine", "kb-machine")],
        pair_rows=[],
        maps={
            PROJ: [
                {
                    "id": "map-1",
                    "short_title": "M1",
                    "long_title": "M1 long",
                    "pointers": ["kb-00001", "kb-00002"],
                },
                {
                    "id": "map-2",
                    "short_title": "M2",
                    "long_title": "M2 long",
                    "pointers": [],
                },
            ]
        },
    )
    resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    maps = {m["id"]: m for m in resp.json()["maps"]}
    assert maps["map-1"]["body"] == full_body
    assert maps["map-1"]["contributor"] == "kb-machine"
    assert maps["map-1"]["updated_by"] == "kb-machine"
    assert maps["map-2"]["body"] == ""
    assert maps["map-2"]["contributor"] is None
    assert maps["map-2"]["updated_by"] is None


def test_map_pointer_naming_id_absent_from_entries_survives(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """maps[].pointers MAY name ids absent from entries; they survive."""
    _wire(
        fake_kb,
        verdicts=[_eligible_verdict()],
        entry_rows=[_entry_row("kb-00001")],
        body_rows=[("map-1", "pointing", None, None)],
        maps={
            PROJ: [
                {
                    "id": "map-1",
                    "short_title": "M",
                    "long_title": "M long",
                    "pointers": ["kb-99999", "kb-00001"],
                }
            ]
        },
    )
    resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    assert resp.json()["maps"][0]["pointers"] == ["kb-99999", "kb-00001"]


# ---------------------------------------------------------------------------
# Runtime audits (AC-20)
# ---------------------------------------------------------------------------


def test_edge_regex_divergence_warning(
    client: TestClient, caplog: pytest.LogCaptureFixture, fake_kb: FakeKnowledgeBase
) -> None:
    """A map pointing at an entry statement 1 called unpointed -> ONE warning."""
    _wire(
        fake_kb,
        verdicts=[_eligible_verdict()],
        entry_rows=[_entry_row("kb-00001"), _entry_row("kb-00002")],
        body_rows=[("map-1", "pointing at kb-00001", None, None)],
        pair_rows=[],
        maps={
            PROJ: [
                {
                    "id": "map-1",
                    "short_title": "M",
                    "long_title": "M long",
                    "pointers": ["kb-00001"],
                }
            ]
        },
    )
    with caplog.at_level(logging.INFO, logger=LOGGER):
        resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    warnings = [r for r in _marker_records(caplog) if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "edge_regex_divergence=kb-00001" in msg
    assert f"project_ref={PROJ}" in msg
    assert "count=1" in msg


def test_pocket_member_not_unpointed_pocket_dropped(
    client: TestClient, caplog: pytest.LogCaptureFixture, fake_kb: FakeKnowledgeBase
) -> None:
    """A pair row naming an id absent from entries -> pocket dropped, warned."""
    _wire(
        fake_kb,
        verdicts=[_eligible_verdict()],
        entry_rows=[_entry_row("kb-00001"), _entry_row("kb-00002")],
        pair_rows=[("kb-00001", "kb-99999", 0.99), ("kb-00001", "kb-00002", 0.90)],
    )
    with caplog.at_level(logging.INFO, logger=LOGGER):
        resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    body = resp.json()
    assert body["pockets"] == []
    warnings = [r for r in _marker_records(caplog) if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "pocket_member_not_unpointed=kb-99999" in warnings[0].getMessage()


def test_slow_pair_query_warning(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """pair_query_ms > 5000 -> one WARNING with the ms and the unpointed count."""

    class _FakeClock:
        """Advances 6000 'seconds' per perf_counter call."""

        def __init__(self) -> None:
            self.now = 0.0

        def perf_counter(self) -> float:
            self.now += 6000.0
            return self.now

    monkeypatch.setattr(map_loop_routes, "time", _FakeClock())
    _wire(
        fake_kb,
        verdicts=[_eligible_verdict()],
        entry_rows=[_entry_row("kb-00001"), _entry_row("kb-00002")],
        pair_rows=[("kb-00001", "kb-00002", 0.9)],
    )
    with caplog.at_level(logging.INFO, logger=LOGGER):
        resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    warnings = [r for r in _marker_records(caplog) if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "pair_query_ms=6000000" in msg
    assert "unpointed_count=2" in msg


# ---------------------------------------------------------------------------
# Read-only invariants (AC-18) + reachability
# ---------------------------------------------------------------------------


def test_reachable_through_real_app_returns_200(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """The route is registered on the real app object — 200, not router-404."""
    _wire(fake_kb, verdicts=[_eligible_verdict()], entry_rows=[_entry_row("kb-00001")])
    resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200


def test_read_only_mechanically_asserted(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """No commits, no write SQL, at most three statements (two when omitted)."""
    _wire(
        fake_kb,
        verdicts=[_eligible_verdict()],
        entry_rows=[_entry_row("kb-00001"), _entry_row("kb-00002")],
        body_rows=[("map-1", "body", None, None)],
        pair_rows=[("kb-00001", "kb-00002", 0.9)],
        maps={
            PROJ: [
                {
                    "id": "map-1",
                    "short_title": "M",
                    "long_title": "M long",
                    "pointers": [],
                }
            ]
        },
    )
    resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    assert fake_kb.db.committed == 0
    for sql, _params in fake_kb.db.calls:
        # Word-boundary match: the map-bodies statement legitimately names the
        # updated_by COLUMN, which a naive case-insensitive substring check
        # would misread as a write statement.
        assert re.search(r"\b(INSERT|UPDATE|DELETE)\b", sql, re.IGNORECASE) is None
    assert len(fake_kb.db.calls) <= 3
    assert len(fake_kb.db.calls) == 3


def test_read_only_invariants_in_docstring_source() -> None:
    """Source greps: no decay-anchor touch, no per-entry getter, ONE anti-join.

    The full edge clause (not the bare quoted word) so prose in the module
    docstring cannot break the count.
    """
    src = Path(map_loop_routes.__file__).read_text(encoding="utf-8")
    assert "touch_accessed" not in src
    assert "kb.get(" not in src
    assert src.count("edge_type = 'references'") == 1


def test_skip_case_issues_exactly_two_statements(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """When pockets_omitted_reason is non-null the statement count is 2."""
    _wire(
        fake_kb,
        verdicts=[_eligible_verdict()],
        entry_rows=[_entry_row("kb-00001", unpointed=0)],
    )
    resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    assert resp.json()["pockets_omitted_reason"] == "too-few-unpointed-entries"
    assert len(fake_kb.db.calls) == 2


# ---------------------------------------------------------------------------
# Telemetry (AC-25)
# ---------------------------------------------------------------------------


def test_successful_call_emits_one_info_telemetry_line(
    client: TestClient, caplog: pytest.LogCaptureFixture, fake_kb: FakeKnowledgeBase
) -> None:
    """Exactly ONE INFO marker record carrying every AC-25 key."""
    _wire_worked_fixture(fake_kb)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    infos = [r for r in _marker_records(caplog) if r.levelno == logging.INFO]
    assert len(infos) == 1
    msg = infos[0].getMessage()
    keys = [
        "project_ref=",
        "entries=",
        "unpointed=",
        "pockets=",
        "pairs_returned=",
        "pairs_below_threshold=",
        "max_similarity_below_threshold=",
        "pairs_mutual=",
        "components_total=",
        "components_below_min_size=",
        "pockets_truncated=",
        "pairs_truncated=",
        "entries_query_ms=",
        "pair_query_ms=",
        "total_ms=",
        "payload_bytes=",
        "payload_digest=",
        "pockets_omitted_reason=",
        "min_similarity=",
        "observation_floor=",
        "top_k=",
        "min_size=",
        "max_count=",
        "max_unpointed=",
        "max_pairs=",
    ]
    for key in keys:
        assert key in msg, key
    assert "entries_query_ms=" in msg
    assert f"project_ref={PROJ}" in msg
    assert "entries=7" in msg
    assert "unpointed=7" in msg
    assert "pockets=2" in msg
    assert "pairs_returned=11" in msg
    assert "pairs_mutual=9" in msg
    assert "components_total=2" in msg
    assert "components_below_min_size=0" in msg
    assert "pockets_truncated=0" in msg
    assert "pockets_omitted_reason=none" in msg
    assert "observation_floor=0.4000" in msg
    assert "min_similarity=0.5500" in msg
    assert "top_k=3" in msg
    assert "min_size=2" in msg
    assert "max_count=10" in msg
    assert "max_unpointed=300" in msg
    assert "max_pairs=20000" in msg
    # payload_digest is the first 12 hex chars of sha256 over the model dump.
    assert f"payload_digest={_expected_digest(resp)}" in msg
    assert f"payload_bytes={len(resp.content)}" in msg


def _expected_digest(resp: Any) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(resp.json(), separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Pocket-builder unit tests (AC-12 order of operations, truncation)
# ---------------------------------------------------------------------------


def test_build_pockets_truncates_to_max_count() -> None:
    """More than POCKET_MAX_COUNT pockets -> keep the top, count the overflow."""
    pairs = [(f"kb-{i:05d}", f"kb-{i + 100:05d}", 0.9) for i in range(1, 12)]
    pockets, stats = map_loop_routes._build_pockets(pairs)
    assert len(pockets) == map_loop_routes.POCKET_MAX_COUNT
    assert stats.pockets_truncated == 1
    # Highest-mean tie is broken by member_entry_ids[0] ascending.
    assert pockets[0].member_entry_ids == ["kb-00001", "kb-00101"]


def test_build_pockets_single_member_component_below_min_size() -> None:
    """A degenerate self-pair forms a 1-member component: below min size."""
    pockets, stats = map_loop_routes._build_pockets([("kb-00001", "kb-00001", 0.9)])
    assert pockets == []
    assert stats.components_total == 1
    assert stats.components_below_min_size == 1
    assert stats.pairs_mutual == 1


def test_build_pockets_sub_threshold_pair_never_influences_topk() -> None:
    """The threshold filter runs FIRST: a sub-threshold pair is invisible."""
    pairs = [
        ("kb-00001", "kb-00002", 0.90),
        ("kb-00002", "kb-00003", 0.54),
        ("kb-00002", "kb-00004", 0.54),
        ("kb-00002", "kb-00005", 0.54),
    ]
    pockets, stats = map_loop_routes._build_pockets(pairs)
    # 0.54 < 0.55: dropped before neighbour lists are built, so kb-00002's
    # top-3 is just kb-00001 and the mutual edge (1,2) forms one pocket.
    assert len(pockets) == 1
    assert pockets[0].member_entry_ids == ["kb-00001", "kb-00002"]
    assert stats.pairs_below_threshold == 3
    assert stats.max_similarity_below_threshold == pytest.approx(0.54)


# ---------------------------------------------------------------------------
# directory_tokens — Rung 3's per-ENTRY "Lives in" source (somnus spec)
# ---------------------------------------------------------------------------

# The real-world SHAPES, not clean fixtures. The dropped literals are what an
# unanchored extractor actually mined out of the 122 photoqueue entry bodies;
# the kept literals are the anchored candidate shapes that must survive.
DROPPED_LITERALS = [
    "creating/updating",
    "task/write",
    "publish/export",
    "opportunistic/deferred",
    "faved/commented",
    "sectionsForTopOfDialog/sectionsForBottomOfDialog",
    "and/or",
    "1800-2400/hr",
]

KEPT_LITERALS = [
    "src/photoqueue/api/foo.py",
    "api/discovery/handler.py",
    "tests/test_thing.py",
    "~/git/personal_kb/packages/kb-core/x.py",
    "/srv/talos/bin/y",
]


def _dir_tokens(entry: dict[str, Any]) -> list[tuple[str, int]]:
    """Flatten one entry's directory_tokens to (token, hits) pairs."""
    return [(t["token"], t["hits"]) for t in entry["directory_tokens"]]


def test_directory_tokens_key_set_and_shape(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """Each token row is exactly {"token": ..., "hits": ...}."""
    _wire(
        fake_kb,
        verdicts=[_eligible_verdict()],
        entry_rows=[
            _entry_row("kb-00001", details="See src/photoqueue/api/foo.py for it.")
        ],
        pair_rows=[],
    )
    resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    entry = resp.json()["entries"][0]
    assert set(entry["directory_tokens"][0].keys()) == {"token", "hits"}
    assert _dir_tokens(entry) == [("src/photoqueue", 1)]


def test_directory_tokens_aggregate_rank_and_tiebreak(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """hits aggregate every candidate normalising to one token; rank is hits
    descending, tie-broken by token ascending."""
    details = (
        "Handlers live in src/photoqueue/api/foo.py and src/photoqueue/ui/bar.py,"
        " plus `src/photoqueue` itself, api/discovery/handler.py,"
        " api/queues/mod.rs and api/queues/list.py, and tests/test_thing.py."
    )
    _wire(
        fake_kb,
        verdicts=[_eligible_verdict()],
        entry_rows=[_entry_row("kb-00001", details=details)],
        pair_rows=[],
    )
    resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    entry = resp.json()["entries"][0]
    assert _dir_tokens(entry) == [
        ("src/photoqueue", 3),
        ("api/queues", 2),
        ("api/discovery", 1),
        ("tests", 1),
    ]


def test_directory_tokens_empty_case_is_allowed_and_meaningful(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """Orientation prose with no anchored path, and a null details column, both
    render directory_tokens: [] — never an error, never a fabricated token."""
    _wire(
        fake_kb,
        verdicts=[_eligible_verdict()],
        entry_rows=[
            _entry_row("kb-00001", details="Plain orientation prose, no paths."),
            _entry_row("kb-00002", details=None),
        ],
        pair_rows=[],
    )
    resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    for entry in resp.json()["entries"]:
        assert entry["directory_tokens"] == []


def test_directory_tokens_dropped_real_corpus_prose_false_positives(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """The mined-out-of-photoqueue prose slash-constructions emit NOTHING.

    A bare two-word slash pair in running prose is not a path; the extractor
    must refuse every member of the class, not just the literal 'and/or'.
    The backticked 1800-2400/hr is anchored yet still dropped — a
    digits-and-dashes segment is not a directory, ever.
    """
    paragraph = (
        "Per-run knobs: creating/updating tasks, task/write splitting,"
        " publish/export ordering, opportunistic/deferred flushing,"
        " faved/commented sync, and/or batching,"
        " sectionsForTopOfDialog/sectionsForBottomOfDialog wiring,"
        " throughput 1800-2400/hr, plus `1800-2400/hr` backticked."
    )
    # The reportable AC, pinned directly: a prose paragraph containing
    # creating/updating and publish/export produces an EMPTY token list.
    assert (
        map_loop_routes._directory_tokens(
            "We trade off creating/updating and publish/export on every run."
        )
        == []
    )
    for literal in DROPPED_LITERALS:
        assert map_loop_routes._directory_tokens(literal) == []
    _wire(
        fake_kb,
        verdicts=[_eligible_verdict()],
        entry_rows=[_entry_row("kb-00001", details=paragraph)],
        pair_rows=[],
    )
    resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    assert resp.json()["entries"][0]["directory_tokens"] == []


def test_directory_tokens_cover_every_verified_lint_table_row(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """Every row of the spec's verified token-shape table, as extracted tokens.

    Three-segment rows normalise to two segments, dotted second segments keep
    only the first, leading roots strip, and every emitted token composes a
    lint-clean 'Lives in' line — including the table's two-token join row.
    """
    details = (
        "Table rows: `packages/kb-core`, `src/kb_service`,"
        " `src/kb_service/routes`, `frontend/src/pages`,"
        " `lightroom-plugin/PhotoQueue.lrdevplugin`, app/main.py,"
        " /srv/talos, ~/git/personal_kb."
    )
    _wire(
        fake_kb,
        verdicts=[_eligible_verdict()],
        entry_rows=[_entry_row("kb-00001", details=details)],
        pair_rows=[],
    )
    resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    entry = resp.json()["entries"][0]
    assert _dir_tokens(entry) == [
        ("src/kb_service", 2),
        ("app", 1),
        ("frontend/src", 1),
        ("git/personal_kb", 1),
        ("lightroom-plugin", 1),
        ("packages/kb-core", 1),
        ("srv/talos", 1),
    ]
    for token, _hits in _dir_tokens(entry):
        assert lint_map_body(f"Lives in {token}.") == []
    # The table's joined row: two tokens joined by ' and ' stay lint-clean.
    assert lint_map_body("Lives in packages/kb-core and src/kb_service.") == []


def test_directory_tokens_lint_round_trip_over_real_world_shapes(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """The strengthened binding AC: the lint round-trip runs over a corpus
    containing the real-world shapes — every dropped literal emits nothing,
    every kept literal emits exactly one token, and 'Lives in <token>' for
    every emitted token has ZERO lint findings."""
    for literal in DROPPED_LITERALS:
        assert map_loop_routes._directory_tokens(literal) == []
    emitted: list[str] = []
    for literal in KEPT_LITERALS:
        tokens = map_loop_routes._directory_tokens(literal)
        assert len(tokens) == 1
        emitted.append(tokens[0].token)
    for token in emitted:
        assert lint_map_body(f"Lives in {token}.") == []

    details = (
        "Kept: "
        + ", ".join(KEPT_LITERALS)
        + ". Dropped prose: creating/updating, task/write, publish/export,"
        " opportunistic/deferred, faved/commented,"
        " sectionsForTopOfDialog/sectionsForBottomOfDialog, and/or,"
        " 1800-2400/hr."
    )
    _wire(
        fake_kb,
        verdicts=[_eligible_verdict()],
        entry_rows=[_entry_row("kb-00001", details=details)],
        pair_rows=[],
    )
    resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    entry = resp.json()["entries"][0]
    assert _dir_tokens(entry) == [
        ("api/discovery", 1),
        ("git/personal_kb", 1),
        ("src/photoqueue", 1),
        ("srv/talos", 1),
        ("tests", 1),
    ]
    for token, _hits in _dir_tokens(entry):
        assert lint_map_body(f"Lives in {token}.") == []


def test_directory_tokens_read_full_details_not_the_excerpt(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """Extraction reads the FULL knowledge_details — a path past char 600, far
    outside the excerpt, is still mined (the excerpt would miss most paths)."""
    details = "filler orientation sentence. " * 40 + "src/photoqueue/api/foo.py"
    assert len(details) > 600
    assert "src/photoqueue" not in details[:600]
    _wire(
        fake_kb,
        verdicts=[_eligible_verdict()],
        entry_rows=[
            _entry_row(
                "kb-00001",
                excerpt="x" * 600,
                details_length=len(details),
                details=details,
            )
        ],
        pair_rows=[],
    )
    resp = client.get(URL, params={"project_ref": PROJ})
    assert resp.status_code == 200
    entry = resp.json()["entries"][0]
    assert entry["excerpt"] == "x" * 600
    assert entry["details_length"] == len(details)
    assert _dir_tokens(entry) == [("src/photoqueue", 1)]


# ---------------------------------------------------------------------------
# GET /api/kb/map-worklist (Rung 0b of docs/somnus-functional-spec.md)
# ---------------------------------------------------------------------------

WORKLIST_URL = "/api/kb/map-worklist"
# The one statement kb_core.map_caps.map_write_summary issues; disjoint from
# the three loop-input needles above.
SUMMARY_NEEDLE = "GROUP BY project_ref"


def _summary_row(project_ref: str, map_count: int, latest: str) -> dict[str, Any]:
    """One map_write_summary row, in kb-core's own column-name-indexed shape.

    The real ``map_write_summary`` reads rows by COLUMN NAME
    (``row["project_ref"]``), so the fake DB hands it mappings, not tuples.
    """
    return {
        "project_ref": project_ref,
        "map_count": map_count,
        "latest_map_written_at": latest,
    }


def _ineligible_verdict(ref: str) -> Any:
    """A computed-ineligible verdict (journal shape), for the drop filter."""
    verdict = make_map_eligibility_verdict(
        ref, mappable=624, ingested=0, top_prefix_count=611, top_prefix="Run"
    )
    assert verdict.effective_eligible is False
    return verdict


def _wire_worklist(
    fake_kb: FakeKnowledgeBase,
    *,
    verdicts: list[Any],
    summary_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Point the worklist's dependencies at the fakes, non-admin user."""
    app.dependency_overrides[get_current_user] = fake_user
    fake_kb.map_eligibility_verdicts = verdicts
    fake_kb.db.rows_for[SUMMARY_NEEDLE] = summary_rows or []


def test_worklist_ordering_never_mapped_then_oldest_write(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """Ranked: never-mapped first, then oldest latest_map_written_at first.

    Do NOT "improve" this into most-unpointed-first — it starves. A project
    whose unpointed tail sits entirely in declined clusters would top that
    list every night forever, accomplish nothing each time, and block every
    other project from ever being picked; declining a cluster does not make
    its members pointed, so the decline ledger cannot rescue it. Staleness
    ordering is starvation-free by construction: a project worked last
    night sinks to the bottom whether or not the night accomplished
    anything.

    The never-mapped ref is alphabetically LAST on purpose, so any
    ordering that ignores the never-mapped-first rule (or ranks on ref
    alone) fails here, not just in theory.
    """
    _wire_worklist(
        fake_kb,
        verdicts=[
            _eligible_verdict("mid-recent"),
            _eligible_verdict("zeta-never"),
            _eligible_verdict("alpha-old"),
        ],
        summary_rows=[
            _summary_row("mid-recent", 1, "2026-09-20T03:00:00+00:00"),
            _summary_row("alpha-old", 3, "2024-03-01T00:00:00+00:00"),
        ],
    )
    resp = client.get(WORKLIST_URL)
    assert resp.status_code == 200
    assert [p["project_ref"] for p in resp.json()["projects"]] == [
        "zeta-never",
        "alpha-old",
        "mid-recent",
    ]


def test_worklist_tiebreak_project_ref_ascending(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """Identical latest_map_written_at -> project_ref ascending, so two runs
    against identical state pick the same three."""
    _wire_worklist(
        fake_kb,
        verdicts=[_eligible_verdict("b-second"), _eligible_verdict("a-first")],
        summary_rows=[
            _summary_row("b-second", 4, "2025-06-01T00:00:00+00:00"),
            _summary_row("a-first", 2, "2025-06-01T00:00:00+00:00"),
        ],
    )
    resp = client.get(WORKLIST_URL)
    assert resp.status_code == 200
    assert [p["project_ref"] for p in resp.json()["projects"]] == [
        "a-first",
        "b-second",
    ]


def test_worklist_non_admin_non_machine_user_gets_200(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """Any authenticated user — the worklist is deliberately NOT admin-gated.

    ``map-eligibility`` is ``require_admin`` and the machine principal is
    deliberately non-admin, so an admin gate here would make the endpoint
    unreachable by exactly the one caller it exists for.
    """
    assert fake_user().is_admin is False
    # The gate pin, not a prose grep: the route module never imports or
    # wires require_admin, so no endpoint here can regress into admin-only.
    src = Path(map_loop_routes.__file__).read_text(encoding="utf-8")
    assert "Depends(require_admin)" not in src
    assert "import require_admin" not in src
    _wire_worklist(
        fake_kb,
        verdicts=[_eligible_verdict(PROJ)],
        summary_rows=[_summary_row(PROJ, 1, "2026-09-20T03:00:00+00:00")],
    )
    resp = client.get(WORKLIST_URL)
    assert resp.status_code == 200
    assert resp.json()["projects"][0]["project_ref"] == PROJ


def test_worklist_no_eligible_projects_is_200_empty(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """No eligible projects is a legitimate quiet night: 200, {projects: []}.

    Never a 404 — somnus's exit-code contract maps an empty worklist to 0.
    """
    _wire_worklist(fake_kb, verdicts=[], summary_rows=[])
    resp = client.get(WORKLIST_URL)
    assert resp.status_code == 200
    assert resp.json() == {"projects": []}


def test_worklist_never_mapped_project_renders_zero_and_null(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """A project with no active maps is ABSENT from map_write_summary, so the
    join is a left-join over the verdicts: map_count 0, null timestamp."""
    _wire_worklist(fake_kb, verdicts=[_eligible_verdict(PROJ)], summary_rows=[])
    resp = client.get(WORKLIST_URL)
    assert resp.status_code == 200
    projects = resp.json()["projects"]
    assert len(projects) == 1
    row = projects[0]
    assert set(row.keys()) == {
        "project_ref",
        "mappable_entries",
        "map_count",
        "latest_map_written_at",
    }
    assert row["project_ref"] == PROJ
    assert row["mappable_entries"] == 50
    assert row["map_count"] == 0
    assert row["latest_map_written_at"] is None


def test_worklist_ineligible_project_absent(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """Only eligible verdicts appear; the override table is respected by
    dropping every verdict whose effective_eligible is False."""
    override = MapEligibilityOverride(
        "forced-in", True, "human verdict", "jason", "2026-09-20T12:00:00+00:00"
    )
    forced_in = make_map_eligibility_verdict(
        "journal-proj",
        mappable=624,
        ingested=0,
        top_prefix_count=611,
        top_prefix="Run",
        override=override,
    )
    assert forced_in.effective_eligible is True
    _wire_worklist(
        fake_kb,
        verdicts=[
            forced_in,
            _ineligible_verdict("computed-out"),
            _eligible_verdict(PROJ),
        ],
        summary_rows=[
            _summary_row("journal-proj", 1, "2026-09-20T03:00:00+00:00"),
            _summary_row("computed-out", 5, "2020-01-01T00:00:00+00:00"),
        ],
    )
    resp = client.get(WORKLIST_URL)
    assert resp.status_code == 200
    refs = [p["project_ref"] for p in resp.json()["projects"]]
    assert "computed-out" not in refs
    assert set(refs) == {"journal-proj", PROJ}
    # The forced-in journal project renders with its summary row; the
    # never-mapped eligible project renders with map_count 0 / null.
    by_ref = {p["project_ref"]: p for p in resp.json()["projects"]}
    assert by_ref["journal-proj"]["map_count"] == 1
    assert by_ref[PROJ]["map_count"] == 0
    assert by_ref[PROJ]["latest_map_written_at"] is None
