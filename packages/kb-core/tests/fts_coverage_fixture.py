"""Shared seed data + expectations for the FTS partial-match coverage tests.

The SQLite suite (``test_fts_coverage.py``) and the Postgres test
(``test_postgres_backend.py::test_fts_coverage_cases_on_postgres``) both seed
the SAME entries and assert the SAME cases, so a backend divergence in which
entries a query reaches, or in their order, fails wherever Postgres runs.

This module is NOT a test module (no ``test_`` functions) — pure data,
safely importable from both suites.

``FTS_COVERAGE_CASES`` rows are ``(query, expected_ids, ordered)``; an
unordered case compares sorted ids.
"""

from __future__ import annotations

FTS_COVERAGE_ENTRY_INSERT_SQL = (
    "INSERT INTO knowledge_entries (id, short_title, long_title, knowledge_details, "
    "entry_type, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
)

_TS = "2026-01-01T00:00:00+00:00"

FTS_COVERAGE_SEED: list[tuple[str, str, str, str, str, str, str]] = [
    (
        "kb-91001",
        "whisper-consumption-dashboard-plan",
        "Proposal: a dashboard to measure whisper telemetry consumption",
        "Proposal, not built: a dashboard that would measure whisper telemetry consumption "
        "and the read rate of each whisper hint, per project and per session. Deferred "
        "until the lab asks for it.",
        "decision",
        _TS,
        _TS,
    ),
    (
        "kb-91002",
        "whisper-uptake-metric",
        "Whisper uptake metric: share of fired whispers an agent opened",
        "Uptake is the share of fired whispers that the agent opened with kb_get within "
        "the same session. It is computed per session from the turn digests; a whisper "
        "the agent never opened counts against uptake.",
        "factual_reference",
        _TS,
        _TS,
    ),
    (
        "kb-91003",
        "whisper-cooldown-window",
        "Whisper cooldown: the same entry is not whispered again within an hour",
        "After an entry is whispered, the hook suppresses it for sixty minutes so a long "
        "session is not nagged by repeats.",
        "pattern_convention",
        _TS,
        _TS,
    ),
    (
        "kb-91004",
        "whisper-consume-signal",
        "Whisper telemetry: consume signal valid after the July cutover",
        "The whisper telemetry consume signal records when an agent opens a whispered "
        "entry with kb_get after the whisper fired. Rows before the cutover predate the "
        "requested-ids field and are invalid. Count consumed whispers per session from "
        "the observe API.",
        "factual_reference",
        _TS,
        _TS,
    ),
]

FTS_COVERAGE_CASES: list[tuple[str, list[str], bool]] = [
    ("measure whisper telemetry consumption", ["kb-91001", "kb-91004"], True),
    ("whisper hint read rate", ["kb-91001"], True),
    ("the and of", [], True),
    ("whisper cooldown", ["kb-91003"], True),
    ("cooldown hour dashboard", ["kb-91003"], True),
    ("the whisper and the cooldown", ["kb-91003"], True),
    ("Whisper whisper WHISPER", ["kb-91001", "kb-91002", "kb-91003", "kb-91004"], False),
    ("kb_get opened", ["kb-91002", "kb-91004"], False),
    ("cooldown's hour", ["kb-91003"], True),
    ("cooldown NEAR * ^ : ( ) {hour}", ["kb-91003"], True),
]

FTS_COVERAGE_FILTER_CASE: tuple[str, str, list[str]] = (
    "whisper telemetry",
    "factual_reference",
    ["kb-91004"],
)
