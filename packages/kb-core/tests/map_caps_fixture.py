"""Shared seed data + expected counts for the map-caps tests.

``kb_core.map_caps.count_maps_created_since`` is ONE SQL statement that runs on BOTH backends.

The two dialects differ in exactly one token.

The Postgres side must carry ``COLLATE "C"`` on the ``created_at >=`` comparison.

That is because the column is TEXT, not a real timestamp type.

The SQLite suite and the Postgres test seed the SAME corpus from ``SEED_ENTRIES``.

The SQLite suite is ``test_map_caps.py``; the Postgres one is ``test_postgres_backend.py``.

Both assert the SAME ``EXPECTED_COUNTS``.

One expectation per backend means a dialect drift fails on whichever side runs.

On a host with no ``KB_TEST_DATABASE_URL`` the Postgres leg ships unexercised.

This module is what makes the drift detectable wherever Postgres DOES run.

The corpus is built around one boundary instant, ``SINCE``.

One map is created exactly at it (the INCLUSIVE ``>=`` under test), one strictly before it.

The other rows are the neighbours the WHERE clause must exclude, one per exclusion term.

This module is NOT a test module itself and is not matched by pytest's collection glob.

It carries no test logic — pure data, safely importable from both suites.

Row shape: the ten columns both suites' INSERT lists name, in the order they bind them.

The builder below is that order's single definition; see ``_entry``.

``created_at`` is the ``.isoformat()`` of an aware UTC datetime.

That is the exact normalization ``count_maps_created_since`` applies to ``since``.

So the boundary case compares two byte-identical strings, not two that differ in formatting.
"""

from __future__ import annotations

from datetime import UTC, datetime

# The single boundary instant every count in EXPECTED_COUNTS is taken at.
SINCE = datetime(2026, 3, 2, 0, 0, 0, tzinfo=UTC)

# The machine principal whose per-night map creation the caps count.
CONTRIBUTOR = "somnus"

_AT = SINCE + datetime.resolution  # one microsecond — "strictly after"
_BEFORE = SINCE - datetime.resolution  # one microsecond — "strictly before"

_SOMNUS = CONTRIBUTOR
_HUMAN = "jason"


def _iso(dt: datetime) -> str:
    """House normalization: aware datetime -> UTC ISO-8601, as the store writes it."""
    return dt.astimezone(UTC).isoformat()


def _entry(
    entry_id: str,
    project_ref: str | None,
    *,
    entry_type: str = "mental_map",
    contributor: str = _SOMNUS,
    created_at: datetime = SINCE,
    is_active: int = 1,
) -> tuple[str, str | None, str, str, str, str, str, str, str, int]:
    """Build one SEED_ENTRIES row."""
    return (
        entry_id,
        project_ref,
        f"Map {entry_id}",
        f"Map {entry_id}",
        f"Orientation prose for {entry_id}, pointing at kb-00001.",
        entry_type,
        contributor,
        _iso(created_at),
        _iso(created_at),
        is_active,
    )


SEED_ENTRIES: tuple[tuple[str, str | None, str, str, str, str, str, str, str, int], ...] = (
    # p-alpha: the inclusive boundary (exactly at SINCE) plus its strictly-
    # before neighbour that the >= must exclude.
    _entry("kb-90001", "p-alpha", created_at=SINCE),
    _entry("kb-90002", "p-alpha", created_at=_BEFORE),
    # p-beta: a map strictly after SINCE in a second project.
    _entry("kb-90003", "p-beta", created_at=_AT),
    # p-beta: same contributor, same instant, WRONG entry_type — excluded.
    _entry("kb-90004", "p-beta", entry_type="factual_reference"),
    # p-beta: right type and instant, DIFFERENT contributor — excluded.
    _entry("kb-90005", "p-beta", contributor=_HUMAN),
    # p-gamma: a DEACTIVATED map created after SINCE — excluded.
    _entry("kb-90006", "p-gamma", created_at=_AT, is_active=0),
    # project_ref NULL: counts only in the unscoped call, never in a scoped one.
    _entry("kb-90007", None, created_at=_AT),
)

# The exact ints ``count_maps_created_since`` must return for the corpus
# above. Keyed by the call's scoping so one table covers both the
# project_ref=None sweep and the per-project scope.
EXPECTED_COUNTS = {
    ("somnus", None): 3,  # kb-90001, kb-90003, kb-90007
    ("somnus", "p-alpha"): 1,  # kb-90001 (the boundary row) — kb-90002 is before SINCE
    ("somnus", "p-beta"): 1,  # kb-90003 — the factual_reference and the foreign map do not count
    ("somnus", "p-gamma"): 0,  # kb-90006 is deactivated
    ("somnus", "p-absent"): 0,  # a project that simply has no maps
    ("jason", None): 1,  # kb-90005 — the human's own map, absent from every somnus count
}
