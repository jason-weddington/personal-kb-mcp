"""DDL and migrations for the knowledge database."""

from personal_kb.db.backend import Database

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

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

CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    short_title,
    long_title,
    knowledge_details,
    tags,
    content='knowledge_entries',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Triggers to keep FTS in sync with the content table
CREATE TRIGGER IF NOT EXISTS knowledge_fts_ai AFTER INSERT ON knowledge_entries BEGIN
    INSERT INTO knowledge_fts(rowid, short_title, long_title, knowledge_details, tags)
    VALUES (new.rowid, new.short_title, new.long_title, new.knowledge_details, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS knowledge_fts_ad AFTER DELETE ON knowledge_entries
BEGIN
    INSERT INTO knowledge_fts(
        knowledge_fts, rowid, short_title, long_title, knowledge_details, tags
    ) VALUES (
        'delete', old.rowid, old.short_title, old.long_title,
        old.knowledge_details, old.tags
    );
END;

CREATE TRIGGER IF NOT EXISTS knowledge_fts_au AFTER UPDATE ON knowledge_entries
BEGIN
    INSERT INTO knowledge_fts(
        knowledge_fts, rowid, short_title, long_title, knowledge_details, tags
    ) VALUES (
        'delete', old.rowid, old.short_title, old.long_title,
        old.knowledge_details, old.tags
    );
    INSERT INTO knowledge_fts(rowid, short_title, long_title, knowledge_details, tags)
    VALUES (new.rowid, new.short_title, new.long_title, new.knowledge_details, new.tags);
END;

CREATE TABLE IF NOT EXISTS entry_id_seq (
    next_id INTEGER NOT NULL DEFAULT 1
);
"""


def _vec_table_sql(dim: int) -> str:
    return f"""
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_vec USING vec0(
    entry_id TEXT PRIMARY KEY,
    embedding FLOAT[{dim}] distance_metric=cosine
);
"""


INIT_SEQ_SQL = """
INSERT INTO entry_id_seq (next_id)
SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM entry_id_seq);
"""


async def apply_schema(db: Database) -> None:
    """Apply the database schema."""
    await db.executescript(SCHEMA_SQL)
    await db.execute(INIT_SEQ_SQL)

    # Check schema version
    cursor = await db.execute("SELECT version FROM schema_version")
    row = await cursor.fetchone()
    if row is None:
        await db.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))

    # Migration: add last_accessed column (nullable, default NULL)
    await _migrate_add_last_accessed(db)

    await db.commit()


async def apply_vec_schema(db: Database, dim: int = 1024) -> None:
    """Create the vec0 virtual table. Requires sqlite-vec extension loaded."""
    await db.executescript(_vec_table_sql(dim))
    await db.commit()


GRAPH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS graph_nodes (
    node_id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    properties TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON graph_nodes(node_type);

CREATE TABLE IF NOT EXISTS graph_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL REFERENCES graph_nodes(node_id),
    target TEXT NOT NULL REFERENCES graph_nodes(node_id),
    edge_type TEXT NOT NULL,
    properties TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(source, target, edge_type)
);
CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target);
CREATE INDEX IF NOT EXISTS idx_edges_type ON graph_edges(edge_type);
"""


async def apply_graph_schema(db: Database) -> None:
    """Create graph_nodes and graph_edges tables."""
    await db.executescript(GRAPH_SCHEMA_SQL)
    await db.commit()


INGEST_SCHEMA_SQL = """
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
"""


async def apply_ingest_schema(db: Database) -> None:
    """Create ingested_files table."""
    await db.executescript(INGEST_SCHEMA_SQL)
    await db.commit()


async def _migrate_add_last_accessed(db: Database) -> None:
    """Add last_accessed column to knowledge_entries if it doesn't exist."""
    cursor = await db.execute("PRAGMA table_info(knowledge_entries)")
    columns = {row[1] for row in await cursor.fetchall()}
    if "last_accessed" not in columns:
        await db.execute("ALTER TABLE knowledge_entries ADD COLUMN last_accessed TEXT")
