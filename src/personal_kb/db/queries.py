"""Query helpers for common database operations."""

import json
from datetime import UTC, datetime
from typing import Any

from personal_kb.db.backend import Database, Row
from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.models.version import EntryVersion


async def next_entry_id(db: Database) -> str:
    """Get and increment the next entry ID."""
    cursor = await db.execute("SELECT next_id FROM entry_id_seq")
    row = await cursor.fetchone()
    if row is None:
        raise RuntimeError("entry_id_seq table is empty")
    next_id = row[0]
    await db.execute("UPDATE entry_id_seq SET next_id = ?", (next_id + 1,))
    return f"kb-{next_id:05d}"


def row_to_entry(row: Row) -> KnowledgeEntry:
    """Convert a database row to a KnowledgeEntry."""
    col_names = row.keys()
    last_accessed_raw = row["last_accessed"] if "last_accessed" in col_names else None
    return KnowledgeEntry(
        id=row["id"],
        project_ref=row["project_ref"],
        short_title=row["short_title"],
        long_title=row["long_title"],
        knowledge_details=row["knowledge_details"],
        entry_type=EntryType(row["entry_type"]),
        source_context=row["source_context"],
        confidence_level=row["confidence_level"],
        tags=_parse_tags(row["tags"]),
        hints=json.loads(row["hints"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        last_accessed=datetime.fromisoformat(last_accessed_raw) if last_accessed_raw else None,
        superseded_by=row["superseded_by"],
        is_active=bool(row["is_active"]),
        has_embedding=bool(row["has_embedding"]),
        version=row["version"],
    )


async def insert_entry(db: Database, entry: KnowledgeEntry) -> None:
    """Insert a new knowledge entry. FTS is auto-synced via triggers."""
    tags_text = " ".join(entry.tags)
    await db.execute(
        """INSERT INTO knowledge_entries
        (id, project_ref, short_title, long_title, knowledge_details, entry_type,
         source_context, confidence_level, tags, hints, created_at, updated_at,
         superseded_by, is_active, has_embedding, version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            entry.id,
            entry.project_ref,
            entry.short_title,
            entry.long_title,
            entry.knowledge_details,
            entry.entry_type.value,
            entry.source_context,
            entry.confidence_level,
            tags_text,
            json.dumps(entry.hints),
            entry.created_at.isoformat() if entry.created_at else _now_iso(),
            entry.updated_at.isoformat() if entry.updated_at else _now_iso(),
            entry.superseded_by,
            int(entry.is_active),
            int(entry.has_embedding),
            entry.version,
        ),
    )
    await db.commit()


async def update_entry(db: Database, entry: KnowledgeEntry) -> None:
    """Update an existing knowledge entry. FTS is auto-synced via triggers."""
    tags_text = " ".join(entry.tags)
    await db.execute(
        """UPDATE knowledge_entries SET
        project_ref=?, short_title=?, long_title=?, knowledge_details=?, entry_type=?,
        source_context=?, confidence_level=?, tags=?, hints=?, updated_at=?,
        superseded_by=?, is_active=?, has_embedding=?, version=?
        WHERE id=?""",
        (
            entry.project_ref,
            entry.short_title,
            entry.long_title,
            entry.knowledge_details,
            entry.entry_type.value,
            entry.source_context,
            entry.confidence_level,
            tags_text,
            json.dumps(entry.hints),
            _now_iso(),
            entry.superseded_by,
            int(entry.is_active),
            int(entry.has_embedding),
            entry.version,
            entry.id,
        ),
    )
    await db.commit()


async def get_entry(db: Database, entry_id: str) -> KnowledgeEntry | None:
    """Get a single entry by ID."""
    cursor = await db.execute("SELECT * FROM knowledge_entries WHERE id = ?", (entry_id,))
    row = await cursor.fetchone()
    return row_to_entry(row) if row else None


async def insert_version(db: Database, version: EntryVersion) -> None:
    """Insert an entry version record."""
    await db.execute(
        """INSERT INTO entry_versions (entry_id, version_number, knowledge_details,
        change_reason, confidence_level, created_at) VALUES (?, ?, ?, ?, ?, ?)""",
        (
            version.entry_id,
            version.version_number,
            version.knowledge_details,
            version.change_reason,
            version.confidence_level,
            version.created_at.isoformat() if version.created_at else _now_iso(),
        ),
    )
    await db.commit()


async def deactivate_entry_db(db: Database, entry_id: str) -> None:
    """Set is_active=0 and update timestamp."""
    await db.execute(
        "UPDATE knowledge_entries SET is_active = 0, updated_at = ? WHERE id = ?",
        (_now_iso(), entry_id),
    )
    await db.commit()


async def reactivate_entry_db(db: Database, entry_id: str) -> None:
    """Set is_active=1 and update timestamp."""
    await db.execute(
        "UPDATE knowledge_entries SET is_active = 1, updated_at = ? WHERE id = ?",
        (_now_iso(), entry_id),
    )
    await db.commit()


async def delete_entry_cascade(db: Database, entry_id: str) -> None:
    """Hard-delete an entry and all related data."""
    await db.execute("DELETE FROM knowledge_vec WHERE entry_id = ?", (entry_id,))
    await db.execute("DELETE FROM entry_versions WHERE entry_id = ?", (entry_id,))
    await db.execute("DELETE FROM graph_edges WHERE source = ? OR target = ?", (entry_id, entry_id))
    await db.execute("DELETE FROM graph_nodes WHERE node_id = ?", (entry_id,))
    await db.execute("DELETE FROM knowledge_entries WHERE id = ?", (entry_id,))
    await db.commit()


async def touch_accessed(db: Database, entry_ids: list[str]) -> None:
    """Batch-update last_accessed to now for the given entry IDs."""
    if not entry_ids:
        return
    now = _now_iso()
    placeholders = ",".join("?" for _ in entry_ids)
    await db.execute(
        "UPDATE knowledge_entries SET last_accessed = ? WHERE id IN ("  # noqa: S608
        + placeholders
        + ")",
        [now, *entry_ids],
    )
    await db.commit()


async def get_all_active_entry_ids(db: Database) -> list[str]:
    """Get all active entry IDs."""
    cursor = await db.execute("SELECT id FROM knowledge_entries WHERE is_active = 1 ORDER BY id")
    rows = await cursor.fetchall()
    return [row["id"] for row in rows]


async def get_db_stats(db: Database) -> dict[str, Any]:
    """Return database statistics: entry counts, graph counts, embedding counts."""
    stats: dict[str, Any] = {}

    # Entry counts
    cursor = await db.execute(
        "SELECT COUNT(*) as total,"
        " SUM(is_active) as active,"
        " COUNT(*) - SUM(is_active) as inactive"
        " FROM knowledge_entries"
    )
    row = await cursor.fetchone()
    if row is None:
        raise RuntimeError("COUNT query returned no rows")
    stats["total_entries"] = row["total"]
    stats["active_entries"] = row["active"] or 0
    stats["inactive_entries"] = row["inactive"] or 0

    # By type
    cursor = await db.execute(
        "SELECT entry_type, COUNT(*) as cnt"
        " FROM knowledge_entries WHERE is_active = 1"
        " GROUP BY entry_type ORDER BY entry_type"
    )
    stats["by_type"] = {row["entry_type"]: row["cnt"] for row in await cursor.fetchall()}

    # By project
    cursor = await db.execute(
        "SELECT COALESCE(project_ref, '(none)') as proj, COUNT(*) as cnt"
        " FROM knowledge_entries WHERE is_active = 1"
        " GROUP BY project_ref ORDER BY cnt DESC"
    )
    stats["by_project"] = {row["proj"]: row["cnt"] for row in await cursor.fetchall()}

    # Embeddings
    cursor = await db.execute(
        "SELECT SUM(has_embedding) as with_emb,"
        " COUNT(*) - SUM(has_embedding) as without_emb"
        " FROM knowledge_entries WHERE is_active = 1"
    )
    row = await cursor.fetchone()
    if row is None:
        raise RuntimeError("COUNT query returned no rows")
    stats["with_embeddings"] = row["with_emb"] or 0
    stats["without_embeddings"] = row["without_emb"] or 0

    # Graph nodes by type
    cursor = await db.execute(
        "SELECT node_type, COUNT(*) as cnt FROM graph_nodes GROUP BY node_type ORDER BY node_type"
    )
    stats["graph_nodes_by_type"] = {row["node_type"]: row["cnt"] for row in await cursor.fetchall()}

    # Graph edges by type
    cursor = await db.execute(
        "SELECT edge_type, COUNT(*) as cnt FROM graph_edges GROUP BY edge_type ORDER BY edge_type"
    )
    stats["graph_edges_by_type"] = {row["edge_type"]: row["cnt"] for row in await cursor.fetchall()}

    return stats


def _parse_tags(raw: str) -> list[str]:
    """Parse tags from storage format. Tags are stored as space-separated text."""
    if not raw or not raw.strip():
        return []
    return raw.split()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
