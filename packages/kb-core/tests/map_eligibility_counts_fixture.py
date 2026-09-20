"""Shared seed data + expected counts for the map-eligibility tests.

The SQLite behaviour suite (``test_map_eligibility.py``) and the Postgres
counts test (``test_postgres_backend.py``) both seed the SAME eight-project
corpus from ``SEED_ENTRIES`` / ``SEED_INGESTED_FILES`` and both assert the
SAME ``EXPECTED_COUNTS``. That shared expectation is the point: the two
backend implementations of ``Database.map_eligibility_counts`` diverge far
more than most dialect seams (``instr``/``substr`` vs ``split_part``,
``json_each`` vs ``jsonb_array_elements_text(entry_ids::jsonb)``, the
``COLLATE "C"`` tie-break, the required derived-table alias), so a fixture
duplicated per backend would let either side drift silently. On a host with
no ``KB_TEST_DATABASE_URL`` the Postgres query ships unexercised — this
module is what makes the drift detectable wherever Postgres DOES run.

The magnitudes are invented to be minimal and fully enumerable; they mirror
the live classifications (corpus / thin / journal / leading-colon) but not
the live sizes.

This module is NOT a test module itself (no ``test_`` functions), is not
matched by pytest's collection glob, and carries no test logic — pure data,
safely importable from both suites.

Seed shapes (positional, matching the INSERT column lists both suites use):

* ``SEED_ENTRIES``: ``(id, project_ref, short_title, long_title,
  knowledge_details, entry_type, created_at, updated_at, is_active)`` —
  every remaining ``knowledge_entries`` column has a default, and the
  FTS trigger reads ``tags`` which defaults to ``'[]'``.
* ``SEED_INGESTED_FILES``: ``(relative_path, content_hash, note_node_id,
  entry_ids, summary, file_size, file_extension, ingested_at, updated_at,
  is_active)`` — ``entry_ids`` is the ready-to-bind JSON array string;
  ``relative_path`` is UNIQUE so every row is distinct (``p-dup-ingest``
  needs two rows listing the same entry id).
"""

from __future__ import annotations

import json

from kb_core.db.backend import MapEligibilityCounts

_TS = "2026-01-01T00:00:00+00:00"


def _entry(
    entry_id: str,
    project_ref: str | None,
    short_title: str,
    *,
    entry_type: str = "factual_reference",
    is_active: int = 1,
) -> tuple[str, str | None, str, str, str, str, str, str, int]:
    """Build one SEED_ENTRIES row."""
    return (
        entry_id,
        project_ref,
        short_title,
        short_title,
        f"Details for {short_title}",
        entry_type,
        _TS,
        _TS,
        is_active,
    )


def _file(
    relative_path: str,
    entry_ids: list[str],
    *,
    is_active: int = 1,
) -> tuple[str, str, str, str, str, int, str, str, str, int]:
    """Build one SEED_INGESTED_FILES row (entry_ids serialized to JSON)."""
    return (
        relative_path,
        f"hash-{relative_path}",
        f"node-{relative_path}",
        json.dumps(entry_ids),
        f"Summary of {relative_path}",
        len(relative_path) * 10,
        ".md",
        _TS,
        _TS,
        is_active,
    )


SEED_ENTRIES: tuple[tuple[str, str | None, str, str, str, str, str, str, int], ...] = (
    # p-corpus: 6 entries, ALL 6 ingested -> ingest corpus, hand_authored 0.
    *(_entry(f"e-corpus-{i}", "p-corpus", f"Corpus note {i}") for i in range(1, 7)),
    # p-dup-ingest: 5 entries, ONE id listed in TWO active ingested_files
    # rows -> mappable 5, ingested 1 (the DISTINCT under test).
    *(_entry(f"e-dup-{i}", "p-dup-ingest", f"Dup note {i}") for i in range(1, 6)),
    # p-eligible-mixed: 10 mappable with distinct colon-free titles, 5
    # ingested, PLUS one mental_map entry (counts in `maps`, not in
    # `mappable`) and one is_active=0 entry (counts nowhere).
    *(_entry(f"e-mix-{i:02d}", "p-eligible-mixed", f"Mixed note {i:02d}") for i in range(1, 11)),
    _entry("e-mix-map", "p-eligible-mixed", "Mixed map", entry_type="mental_map"),
    _entry("e-mix-dead", "p-eligible-mixed", "Mixed dead", is_active=0),
    # p-inactive-file: 5 mappable, 2 of them listed ONLY in an is_active=0
    # ingested_files row -> ingested 0 (the f.is_active filter under test).
    *(_entry(f"e-inact-{i}", "p-inactive-file", f"Inactive note {i}") for i in range(1, 6)),
    # p-journal: 20 entries all titled "Run: <i>" -> journal (>= 20, share 1.0).
    *(_entry(f"e-journal-{i}", "p-journal", f"Run: {i}") for i in range(1, 21)),
    # p-journal-sub20: 19 entries all titled "Run: <i>" -> NOT a journal
    # (pins the mappable >= 20 guard), so computed-eligible.
    *(_entry(f"e-journal-sub-{i}", "p-journal-sub20", f"Run: {i}") for i in range(1, 20)),
    # p-leading-colon: the leading-colon title's prefix is the EMPTY STRING
    # and the byte-ordered tie-break selects it deterministically over
    # "zzz other" (1 entry each).
    _entry("e-colon-1", "p-leading-colon", ": leading colon"),
    _entry("e-colon-2", "p-leading-colon", "zzz other"),
    # p-thin: 9 mappable, 5 ingested -> hand_authored 4 (pins `< 5`).
    *(_entry(f"e-thin-{i}", "p-thin", f"Thin note {i}") for i in range(1, 10)),
    # Three project_ref-NULL entries: absent from every returned row.
    *(_entry(f"e-null-{i}", None, f"Null note {i}") for i in range(1, 4)),
)

SEED_INGESTED_FILES: tuple[tuple[str, str, str, str, str, int, str, str, str, int], ...] = (
    _file("corpus/all.md", [f"e-corpus-{i}" for i in range(1, 7)]),
    _file("dup/a.md", ["e-dup-3"]),
    _file("dup/b.md", ["e-dup-3"]),
    _file("mixed/half.md", [f"e-mix-{i:02d}" for i in range(1, 6)]),
    _file("thin/half.md", [f"e-thin-{i}" for i in range(1, 6)]),
    _file("inactive/file.md", ["e-inact-1", "e-inact-2"], is_active=0),
)

# The exact rows `Database.map_eligibility_counts()` must return for the
# corpus above, in ascending project_ref order (byte/code-point order —
# identical on SQLite BINARY, Postgres COLLATE "C", and Python).
EXPECTED_COUNTS: tuple[MapEligibilityCounts, ...] = (
    MapEligibilityCounts("p-corpus", 6, 6, 1, "Corpus note 1", 0),
    MapEligibilityCounts("p-dup-ingest", 5, 1, 1, "Dup note 1", 0),
    MapEligibilityCounts("p-eligible-mixed", 10, 5, 1, "Mixed note 01", 1),
    MapEligibilityCounts("p-inactive-file", 5, 0, 1, "Inactive note 1", 0),
    MapEligibilityCounts("p-journal", 20, 0, 20, "Run", 0),
    MapEligibilityCounts("p-journal-sub20", 19, 0, 19, "Run", 0),
    MapEligibilityCounts("p-leading-colon", 2, 0, 1, "", 0),
    MapEligibilityCounts("p-thin", 9, 5, 1, "Thin note 1", 0),
)
