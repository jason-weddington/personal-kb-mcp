"""SQLite run of the shared FTS partial-match coverage cases."""

from __future__ import annotations

from typing import Any

from fts_coverage_fixture import (
    FTS_COVERAGE_CASES,
    FTS_COVERAGE_ENTRY_INSERT_SQL,
    FTS_COVERAGE_FILTER_CASE,
    FTS_COVERAGE_SEED,
)
from kb_core import create_sqlite


async def test_fts_coverage_cases_on_sqlite(tmp_path: Any) -> None:
    kb = await create_sqlite(tmp_path / "kb.db")
    try:
        for row in FTS_COVERAGE_SEED:
            await kb.db.execute(FTS_COVERAGE_ENTRY_INSERT_SQL, row)
        await kb.db.commit()

        for query, expected, ordered in FTS_COVERAGE_CASES:
            got = [eid for eid, _ in await kb.db.fts_search(query, limit=10)]
            if ordered:
                assert got == expected, query
            else:
                assert sorted(got) == sorted(expected), query

        query, entry_type, expected = FTS_COVERAGE_FILTER_CASE
        got = [eid for eid, _ in await kb.db.fts_search(query, entry_type=entry_type, limit=10)]
        assert got == expected
    finally:
        await kb.close()
