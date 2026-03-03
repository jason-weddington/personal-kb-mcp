"""Regression tests for schema migration on pre-existing databases.

Creates a database with the old schema (before multi-user columns),
then runs the full startup path and verifies everything works.
"""

import pytest

from personal_kb.db.connection import create_connection

# Old schema: tables exist but WITHOUT contributor/team/updated_by/sensitivity columns.
# This simulates a database created before the multi-user feature.
_OLD_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
INSERT INTO schema_version (version) VALUES (1);

CREATE TABLE IF NOT EXISTS knowledge_entries (
    id TEXT PRIMARY KEY,
    project_ref TEXT,
    short_title TEXT NOT NULL,
    long_title TEXT NOT NULL,
    knowledge_details TEXT NOT NULL,
    entry_type TEXT NOT NULL,
    source_context TEXT,
    confidence_level REAL NOT NULL DEFAULT 0.9,
    tags TEXT NOT NULL DEFAULT '[]',
    hints TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed TEXT,
    superseded_by TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    has_embedding INTEGER NOT NULL DEFAULT 0,
    version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_entries_project ON knowledge_entries(project_ref);
CREATE INDEX IF NOT EXISTS idx_entries_type ON knowledge_entries(entry_type);
CREATE INDEX IF NOT EXISTS idx_entries_active ON knowledge_entries(is_active);

CREATE TABLE IF NOT EXISTS entry_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id TEXT NOT NULL REFERENCES knowledge_entries(id),
    version_number INTEGER NOT NULL,
    knowledge_details TEXT NOT NULL,
    change_reason TEXT,
    confidence_level REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(entry_id, version_number)
);

CREATE TABLE IF NOT EXISTS entry_id_seq (next_id INTEGER NOT NULL DEFAULT 1);
INSERT INTO entry_id_seq (next_id) VALUES (1);

CREATE TABLE IF NOT EXISTS graph_nodes (
    node_id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    properties TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL REFERENCES graph_nodes(node_id),
    target TEXT NOT NULL REFERENCES graph_nodes(node_id),
    edge_type TEXT NOT NULL,
    properties TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(source, target, edge_type)
);

CREATE TABLE IF NOT EXISTS ingested_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    relative_path TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    note_node_id TEXT NOT NULL,
    entry_ids TEXT NOT NULL DEFAULT '[]',
    summary TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    file_extension TEXT NOT NULL,
    project_ref TEXT,
    redactions TEXT NOT NULL DEFAULT '[]',
    ingested_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS search_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT NOT NULL,
    result_count INTEGER NOT NULL,
    top_score REAL,
    match_source TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_type TEXT NOT NULL CHECK(feedback_type IN ('missing', 'unhelpful', 'friction')),
    tool_name TEXT,
    query_or_params TEXT,
    detail TEXT,
    created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    short_title, long_title, knowledge_details, tags,
    content='knowledge_entries', content_rowid='rowid',
    tokenize='porter unicode61'
);
"""

_MIGRATED_COLUMNS = {
    "knowledge_entries": ["contributor", "team", "updated_by", "sensitivity"],
    "entry_versions": ["contributor"],
    "search_events": ["contributor"],
    "agent_feedback": ["contributor"],
    "ingested_files": ["contributor"],
}


@pytest.fixture
def old_db_path(tmp_path, monkeypatch):
    """Create a SQLite DB file with the old (pre-migration) schema."""
    import sqlite3

    # Force SQLite path even if KB_DATABASE_URL is set
    monkeypatch.delenv("KB_DATABASE_URL", raising=False)

    db_file = tmp_path / "old.db"
    conn = sqlite3.connect(str(db_file))
    conn.executescript(_OLD_SCHEMA)
    conn.close()
    return str(db_file)


@pytest.mark.asyncio
async def test_migration_adds_columns(old_db_path):
    """create_connection on a pre-migration DB should add all multi-user columns."""
    db = await create_connection(old_db_path)
    try:
        for table, expected_cols in _MIGRATED_COLUMNS.items():
            cursor = await db.execute(f"PRAGMA table_info({table})")
            columns = {row[1] for row in await cursor.fetchall()}
            for col in expected_cols:
                assert col in columns, f"{table}.{col} missing after migration"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_migration_creates_indexes(old_db_path):
    """Migration should create indexes on contributor and team."""
    db = await create_connection(old_db_path)
    try:
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name")
        indexes = {row[0] for row in await cursor.fetchall()}
        assert "idx_entries_contributor" in indexes
        assert "idx_entries_team" in indexes
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_migration_insert_with_contributor(old_db_path):
    """After migration, INSERT with contributor column should work."""
    db = await create_connection(old_db_path)
    try:
        await db.execute(
            "INSERT INTO knowledge_entries"
            " (id, short_title, long_title, knowledge_details, entry_type,"
            "  confidence_level, tags, hints, created_at, updated_at,"
            "  is_active, has_embedding, version, contributor, team)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "kb-00001",
                "test",
                "test entry",
                "details",
                "factual_reference",
                0.9,
                "tag1",
                "{}",
                "2025-01-01T00:00:00",
                "2025-01-01T00:00:00",
                1,
                0,
                1,
                "jason",
                "platform",
            ),
        )
        await db.commit()

        cursor = await db.execute(
            "SELECT contributor, team FROM knowledge_entries WHERE id = ?",
            ("kb-00001",),
        )
        row = await cursor.fetchone()
        assert row[0] == "jason"
        assert row[1] == "platform"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_migration_search_event_with_contributor(old_db_path):
    """After migration, INSERT into search_events with contributor should work."""
    db = await create_connection(old_db_path)
    try:
        await db.execute(
            "INSERT INTO search_events"
            " (query_text, result_count, top_score, match_source, contributor, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            ("test query", 5, 0.8, "fts", "jason", "2025-01-01T00:00:00"),
        )
        await db.commit()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_migration_idempotent(old_db_path):
    """Running create_connection twice on the same DB should not fail."""
    db = await create_connection(old_db_path)
    await db.close()

    db = await create_connection(old_db_path)
    try:
        cursor = await db.execute("PRAGMA table_info(knowledge_entries)")
        columns = {row[1] for row in await cursor.fetchall()}
        assert "contributor" in columns
    finally:
        await db.close()
