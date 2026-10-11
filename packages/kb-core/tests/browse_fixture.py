"""Shared seed data + expectations for the kb-core browse tests.

The SQLite suite (``test_browse.py``) and the Postgres test
(``test_postgres_backend.py::test_browse_cases_on_postgres``) both seed the
SAME graph nodes, graph edges and entries and assert the SAME cases, so a
dialect difference in what the browse SELECT returns fails wherever
Postgres runs.

This module is NOT a test module (no ``test_`` functions) — pure data,
safely importable from both suites.

Seed shapes (positional, matching the INSERT column lists both suites use):

* ``BROWSE_SEED_NODES``: ``(node_id, node_type, properties, created_at)``.
* ``BROWSE_SEED_EDGES``: ``(source, target, edge_type, properties,
  created_at)``. Insert nodes first: Postgres graph_edges has a foreign key
  to graph_nodes.
* ``BROWSE_SEED_ENTRIES``: ``(id, project_ref, short_title, long_title,
  knowledge_details, entry_type, confidence_level, created_at, updated_at,
  last_accessed, expires_at, superseded_by, is_active)``.

Confidences are float32-exact, so Postgres REAL matches SQLite.
"""

from __future__ import annotations

from datetime import UTC, datetime

BROWSE_NOW = datetime(2026, 7, 1, tzinfo=UTC)

_GRAPH_TS = "2026-01-01T00:00:00+00:00"

BROWSE_SEED_NODES: list[tuple[str, str, str, str]] = [
    *[(f"kb-{n}", "entry", "{}", _GRAPH_TS) for n in range(90001, 90012)],
    ("tag:x", "tag", "{}", _GRAPH_TS),
    ("project:p-alpha", "project", "{}", _GRAPH_TS),
]

BROWSE_SEED_EDGES: list[tuple[str, str, str, str, str]] = [
    ("kb-90001", "tag:x", "has_tag", "{}", _GRAPH_TS),
    ("kb-90001", "project:p-alpha", "in_project", "{}", _GRAPH_TS),
    ("kb-90002", "kb-90001", "related_to", "{}", _GRAPH_TS),
    ("kb-90005", "kb-90001", "references", "{}", _GRAPH_TS),
    ("kb-90005", "kb-90002", "references", "{}", _GRAPH_TS),
    ("kb-90007", "kb-90001", "supersedes", "{}", _GRAPH_TS),
    ("kb-90003", "kb-90003", "related_to", "{}", _GRAPH_TS),
]

BrowseSeedEntry = tuple[
    str, str | None, str, str, str, str, float, str, str, str | None, str | None, str | None, int
]


def _entry(
    entry_id: str,
    project_ref: str | None,
    short_title: str,
    entry_type: str,
    confidence: float,
    created_at: str,
    updated_at: str | None = None,
    *,
    last_accessed: str | None = None,
    expires_at: str | None = None,
    superseded_by: str | None = None,
    is_active: int = 1,
) -> BrowseSeedEntry:
    return (
        entry_id,
        project_ref,
        short_title,
        short_title + " (long)",
        "Details for " + short_title,
        entry_type,
        confidence,
        created_at,
        updated_at if updated_at is not None else created_at,
        last_accessed,
        expires_at,
        superseded_by,
        is_active,
    )


BROWSE_SEED_ENTRIES: list[BrowseSeedEntry] = [
    _entry(
        "kb-90001",
        "p-alpha",
        "Alpha fact",
        "factual_reference",
        1.0,
        "2026-06-30T00:00:00+00:00",
        last_accessed="2026-06-30T12:00:00+00:00",
    ),
    _entry(
        "kb-90002",
        "p-alpha",
        "Alpha decision",
        "decision",
        1.0,
        "2026-01-01T00:00:00+00:00",
        "2026-06-20T00:00:00+00:00",
    ),
    _entry(
        "kb-90003",
        "p-beta",
        "Beta stale fact",
        "factual_reference",
        0.5,
        "2025-07-01T00:00:00+00:00",
    ),
    _entry(
        "kb-90004",
        "p-beta",
        "Beta read fact",
        "factual_reference",
        1.0,
        "2025-07-01T00:00:00+00:00",
        last_accessed="2026-06-25T00:00:00+00:00",
    ),
    _entry("kb-90005", "p-alpha", "Alpha map", "mental_map", 1.0, "2024-01-01T00:00:00+00:00"),
    _entry(
        "kb-90006",
        "p-beta",
        "Beta weak lesson",
        "lesson_learned",
        0.25,
        "2026-06-01T00:00:00+00:00",
    ),
    _entry(
        "kb-90007",
        "p-alpha",
        "Alpha superseded",
        "factual_reference",
        1.0,
        "2026-06-29T00:00:00+00:00",
        "2026-06-30T06:00:00+00:00",
        superseded_by="kb-90001",
    ),
    _entry(
        "kb-90008",
        "p-alpha",
        "Alpha inactive",
        "decision",
        1.0,
        "2026-06-28T00:00:00+00:00",
        is_active=0,
    ),
    _entry(
        "kb-90009",
        None,
        "Unscoped expired pattern",
        "pattern_convention",
        0.75,
        "2026-06-15T10:15:16",
        expires_at="2026-06-20T00:00:00+00:00",
    ),
    _entry(
        "kb-90010",
        "p-gamma",
        "Gamma decision",
        "decision",
        0.75,
        "2026-06-10T00:00:00+00:00",
        expires_at="2026-12-31T00:00:00+00:00",
    ),
    _entry(
        "kb-90011",
        "p-gamma",
        "Gamma offset fact",
        "factual_reference",
        1.0,
        "2026-06-10T01:00:00+02:00",
    ),
]

BROWSE_ENTRY_INSERT_SQL = (
    "INSERT INTO knowledge_entries"
    " (id, project_ref, short_title, long_title, knowledge_details, entry_type,"
    " confidence_level, created_at, updated_at, last_accessed, expires_at,"
    " superseded_by, is_active)"
    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
BROWSE_NODE_INSERT_SQL = (
    "INSERT INTO graph_nodes (node_id, node_type, properties, created_at) VALUES (?, ?, ?, ?)"
)
BROWSE_EDGE_INSERT_SQL = (
    "INSERT INTO graph_edges (source, target, edge_type, properties, created_at)"
    " VALUES (?, ?, ?, ?, ?)"
)

BROWSE_CASES: list[tuple[str, str, dict[str, object], list[str]]] = [
    (
        "recent_created",
        "browse_recent",
        {},
        [
            "kb-90001",
            "kb-90009",
            "kb-90010",
            "kb-90011",
            "kb-90006",
            "kb-90002",
            "kb-90003",
            "kb-90004",
            "kb-90005",
        ],
    ),
    ("recent_updated", "browse_recent", {"sort": "updated"}, ["kb-90002"]),
    (
        "recent_factual",
        "browse_recent",
        {"entry_type": "factual_reference"},
        ["kb-90001", "kb-90011", "kb-90003", "kb-90004"],
    ),
    ("top_degree", "browse_top", {}, ["kb-90001", "kb-90002", "kb-90005", "kb-90003"]),
    ("top_read", "browse_top", {"signal": "read"}, ["kb-90001", "kb-90004"]),
    ("stale_decayed", "browse_stale", {}, ["kb-90003", "kb-90006"]),
    ("stale_lesson", "browse_stale", {"entry_type": "lesson_learned"}, ["kb-90006"]),
    ("stale_expired", "browse_stale", {"kind": "expired"}, ["kb-90009"]),
    ("projects", "browse_projects", {}, ["p-alpha", "p-gamma", "p-beta"]),
    ("projects_q", "browse_projects", {"q": "GA"}, ["p-gamma"]),
    (
        "project_alpha_newest",
        "browse_project",
        {"project_ref": "p-alpha"},
        ["kb-90005", "kb-90001", "kb-90002"],
    ),
    (
        "project_beta_confidence",
        "browse_project",
        {"project_ref": "p-beta", "sort": "confidence"},
        ["kb-90004", "kb-90006", "kb-90003"],
    ),
    (
        "project_beta_newest",
        "browse_project",
        {"project_ref": "p-beta"},
        ["kb-90006", "kb-90003", "kb-90004"],
    ),
    (
        "project_gamma_newest",
        "browse_project",
        {"project_ref": "p-gamma"},
        ["kb-90010", "kb-90011"],
    ),
    (
        "project_alpha_q",
        "browse_project",
        {"project_ref": "p-alpha", "q": "DECISION"},
        ["kb-90002"],
    ),
    (
        "project_alpha_q_id",
        "browse_project",
        {"project_ref": "p-alpha", "q": "kb-90005"},
        ["kb-90005"],
    ),
    (
        "project_alpha_type",
        "browse_project",
        {"project_ref": "p-alpha", "entry_type": "decision"},
        ["kb-90002"],
    ),
]

BROWSE_EXPECTED_DEGREES = [5, 2, 2, 1]

BROWSE_EXPECTED_PROJECTS: list[tuple[str, int, dict[str, int], str]] = [
    (
        "p-alpha",
        3,
        {"factual_reference": 1, "decision": 1, "mental_map": 1},
        "2026-06-30T00:00:00+00:00",
    ),
    ("p-gamma", 2, {"factual_reference": 1, "decision": 1}, "2026-06-10T00:00:00+00:00"),
    ("p-beta", 3, {"factual_reference": 2, "lesson_learned": 1}, "2026-06-01T00:00:00+00:00"),
]
