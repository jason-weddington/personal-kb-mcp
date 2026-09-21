"""Shared seed data + expected counts for the map-caps tests.

``kb_core.map_caps.count_maps_created_since`` is ONE SQL statement that runs on BOTH backends.

The two dialects differ in exactly one token.

The Postgres side must carry ``COLLATE "C"`` on the ``created_at >=`` comparison.

That is because the column is TEXT, not a real timestamp type.

The SQLite suite and the Postgres test seed the SAME corpus from ``SEED_ENTRIES``.

The SQLite suite is ``test_map_caps.py``; the Postgres one is ``test_postgres_backend.py``.

Both assert the SAME ``EXPECTED_COUNTS`` and the SAME ``EXPECTED_SUMMARY``.

One expectation per backend means a dialect drift fails on whichever side runs.

On a host with no ``KB_TEST_DATABASE_URL`` the Postgres leg ships unexercised.

This module is what makes the drift detectable wherever Postgres DOES run.

The corpus also feeds ``map_caps.map_write_summary`` — the same rows, no second seed.

That query's two facts are per PROJECT, so it has no contributor and no ``since``.

Every row therefore plays two roles: one for the count's WHERE, one for the summary's GROUP BY.

The human's map (``kb-90005``) is the clearest example: the count excludes it, the summary keeps it.

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

``updated_at`` defaults to ``created_at`` — a row nobody has touched since is written exactly once.

The two rows that override it carry the instants ``LATEST_MAP_WRITE`` and ``LATER_THAN_ANY_MAP``.

They are what keeps ``map_write_summary`` honest about WHICH column it reads.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kb_core.map_caps import MapWriteSummary

# The single boundary instant every count in EXPECTED_COUNTS is taken at.
SINCE = datetime(2026, 3, 2, 0, 0, 0, tzinfo=UTC)

# The machine principal whose per-night map creation the caps count.
CONTRIBUTOR = "somnus"

_AT = SINCE + datetime.resolution  # one microsecond — "strictly after"
_BEFORE = SINCE - datetime.resolution  # one microsecond — "strictly before"

# The newest last-write of any ACTIVE mental_map in the corpus.
# kb-90002 carries it and was also created FIRST (a microsecond before SINCE),
# so a summary that read created_at would report SINCE for p-alpha instead.
LATEST_MAP_WRITE = SINCE + timedelta(hours=3)

# A write newer than every map's, carried by a factual_reference.
# p-beta's latest must stay at P_BETA_LATEST_WRITE, because the newest
# write in that project belongs to an entry the summary must not read at all.
LATER_THAN_ANY_MAP = SINCE + timedelta(hours=4)

# p-beta's expected latest write — kb-90003's, one microsecond after SINCE.
# Nothing newer in that project is a mental_map, which is the point of the row.
P_BETA_LATEST_WRITE = _AT

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
    updated_at: datetime | None = None,
    is_active: int = 1,
) -> tuple[str, str | None, str, str, str, str, str, str, str, int]:
    """Build one SEED_ENTRIES row.

    ``updated_at`` defaults to ``created_at`` — the shape a row has until
    something writes to it again.
    """
    return (
        entry_id,
        project_ref,
        f"Map {entry_id}",
        f"Map {entry_id}",
        f"Orientation prose for {entry_id}, pointing at kb-00001.",
        entry_type,
        contributor,
        _iso(created_at),
        _iso(created_at if updated_at is None else updated_at),
        is_active,
    )


SEED_ENTRIES: tuple[tuple[str, str | None, str, str, str, str, str, str, str, int], ...] = (
    # p-alpha: the inclusive boundary (exactly at SINCE) plus its strictly-
    # before neighbour that the >= must exclude.
    _entry("kb-90001", "p-alpha", created_at=SINCE),
    # p-alpha: created FIRST (a microsecond before SINCE), written LAST
    # (LATEST_MAP_WRITE) — the row that separates updated_at from created_at.
    _entry("kb-90002", "p-alpha", created_at=_BEFORE, updated_at=LATEST_MAP_WRITE),
    # p-beta: a map strictly after SINCE in a second project.
    _entry("kb-90003", "p-beta", created_at=_AT),
    # p-beta: same contributor, same instant, WRONG entry_type — excluded
    # from the count, and from BOTH summary fields even though its
    # LATER_THAN_ANY_MAP write is the newest in the whole corpus.
    _entry("kb-90004", "p-beta", entry_type="factual_reference", updated_at=LATER_THAN_ANY_MAP),
    # p-beta: right type and instant, DIFFERENT contributor — excluded from
    # every somnus count, but still an ACTIVE map in its project, so the
    # summary counts it.
    _entry("kb-90005", "p-beta", contributor=_HUMAN),
    # p-gamma: a DEACTIVATED map created after SINCE — excluded, and the
    # only map its project has, so p-gamma is absent from the summary.
    _entry("kb-90006", "p-gamma", created_at=_AT, is_active=0),
    # project_ref NULL: counts only in the unscoped call, never in a scoped
    # one, and belongs to no project the summary can rank.
    _entry("kb-90007", None, created_at=_AT),
    # project_ref "": an empty ref is no project either — the summary must
    # drop it, while the unscoped count still sees it.
    _entry("kb-90008", "", contributor=_HUMAN, created_at=_AT),
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
    ("jason", None): 2,  # kb-90005, kb-90008 — the human's own maps, absent from every somnus count
}

# The exact rows ``map_write_summary`` must return for the corpus above.
# Ordered by project_ref ascending, like the query.
# p-gamma is ABSENT, not present with map_count = 0: its only map is deactivated.
# The NULL- and empty-ref rows are absent for the same reason — they are no project.
# The consumer reads absence as never-mapped and sorts those projects first.
EXPECTED_SUMMARY: tuple[MapWriteSummary, ...] = (
    MapWriteSummary(project_ref="p-alpha", map_count=2, latest_map_written_at=LATEST_MAP_WRITE),
    MapWriteSummary(project_ref="p-beta", map_count=2, latest_map_written_at=P_BETA_LATEST_WRITE),
)
