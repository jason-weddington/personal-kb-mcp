"""Query helpers for common database operations."""

import json
from datetime import UTC, datetime

import aiosqlite

from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.models.version import EntryVersion


async def next_entry_id(db: aiosqlite.Connection) -> str:
    """Get and increment the next entry ID."""
    cursor = await db.execute("SELECT next_id FROM entry_id_seq")
    row = await cursor.fetchone()
    if row is None:
        raise RuntimeError("entry_id_seq table is empty")
    next_id = row[0]
    await db.execute("UPDATE entry_id_seq SET next_id = ?", (next_id + 1,))
    return f"kb-{next_id:05d}"


def row_to_entry(row: aiosqlite.Row) -> KnowledgeEntry:
    """Convert a database row to a KnowledgeEntry."""
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
        superseded_by=row["superseded_by"],
        is_active=bool(row["is_active"]),
        has_embedding=bool(row["has_embedding"]),
        version=row["version"],
    )


async def insert_entry(db: aiosqlite.Connection, entry: KnowledgeEntry) -> None:
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


async def update_entry(db: aiosqlite.Connection, entry: KnowledgeEntry) -> None:
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


async def get_entry(db: aiosqlite.Connection, entry_id: str) -> KnowledgeEntry | None:
    """Get a single entry by ID."""
    cursor = await db.execute("SELECT * FROM knowledge_entries WHERE id = ?", (entry_id,))
    row = await cursor.fetchone()
    return row_to_entry(row) if row else None


async def insert_version(db: aiosqlite.Connection, version: EntryVersion) -> None:
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


def _parse_tags(raw: str) -> list[str]:
    """Parse tags from storage format. Tags are stored as space-separated text."""
    if not raw or not raw.strip():
        return []
    return raw.split()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
