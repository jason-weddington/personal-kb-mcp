"""CRUD operations for knowledge entries."""

import logging
from datetime import UTC, datetime
from typing import Literal

from personal_kb.db.backend import Database
from personal_kb.db.queries import (
    deactivate_entry_db,
    get_entry,
    insert_entry,
    insert_version,
    next_entry_id,
    reactivate_entry_db,
    update_entry,
)
from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.models.version import EntryVersion

logger = logging.getLogger(__name__)


async def _record_audit_event(
    db: Database,
    event_type: str,
    entry_id: str,
    contributor: str | None = None,
    detail: str | None = None,
) -> None:
    """Record an audit event. Fire-and-forget — failures never break the operation."""
    try:
        now = datetime.now(UTC).isoformat()
        await db.execute(
            "INSERT INTO audit_events (event_type, entry_id, contributor, detail, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (event_type, entry_id, contributor, detail, now),
        )
        await db.commit()
    except Exception:
        logger.warning("Failed to record audit event", exc_info=True)


class KnowledgeStore:
    """CRUD operations for knowledge entries with versioning."""

    def __init__(self, db: Database):
        """Initialize with a database connection."""
        self.db = db

    async def create_entry(
        self,
        short_title: str,
        long_title: str,
        knowledge_details: str,
        entry_type: EntryType,
        project_ref: str | None = None,
        source_context: str | None = None,
        confidence_level: float = 0.9,
        tags: list[str] | None = None,
        hints: dict[str, object] | None = None,
        contributor: str | None = None,
        team: str | None = None,
        sensitivity: Literal["internal", "restricted", "public"] | None = None,
    ) -> KnowledgeEntry:
        """Create a new knowledge entry with initial version."""
        entry_id = await next_entry_id(self.db)
        now = datetime.now(UTC)

        entry = KnowledgeEntry(
            id=entry_id,
            project_ref=project_ref,
            short_title=short_title,
            long_title=long_title,
            knowledge_details=knowledge_details,
            entry_type=entry_type,
            source_context=source_context,
            contributor=contributor,
            team=team,
            sensitivity=sensitivity,
            confidence_level=confidence_level,
            tags=tags or [],
            hints=hints or {},
            created_at=now,
            updated_at=now,
            version=1,
        )
        await insert_entry(self.db, entry)

        # Create initial version record
        version = EntryVersion(
            entry_id=entry_id,
            version_number=1,
            knowledge_details=knowledge_details,
            change_reason="Initial creation",
            contributor=contributor,
            confidence_level=confidence_level,
            created_at=now,
        )
        await insert_version(self.db, version)

        await _record_audit_event(self.db, "entry_created", entry_id, contributor, short_title)

        logger.info("Created entry %s: %s", entry_id, short_title)
        return entry

    async def update_entry(
        self,
        entry_id: str,
        knowledge_details: str,
        change_reason: str | None = None,
        confidence_level: float | None = None,
        tags: list[str] | None = None,
        hints: dict[str, object] | None = None,
        updated_by: str | None = None,
        sensitivity: Literal["internal", "restricted", "public"] | None = None,
        short_title: str | None = None,
        long_title: str | None = None,
        entry_type: EntryType | None = None,
        project_ref: str | None = None,
        source_context: str | None = None,
    ) -> KnowledgeEntry:
        """Update an existing entry, creating a new version."""
        existing = await get_entry(self.db, entry_id)
        if existing is None:
            raise ValueError(f"Entry {entry_id} not found")
        if not existing.is_active:
            raise ValueError(f"Entry {entry_id} is inactive and cannot be updated")

        now = datetime.now(UTC)
        new_version = existing.version + 1
        new_confidence = (
            confidence_level if confidence_level is not None else existing.confidence_level
        )

        # Merge hints
        merged_hints = dict(existing.hints)
        if hints:
            merged_hints.update(hints)

        update_fields: dict[str, object] = {
            "knowledge_details": knowledge_details,
            "confidence_level": new_confidence,
            "tags": tags if tags is not None else existing.tags,
            "hints": merged_hints,
            "updated_at": now,
            "version": new_version,
            "has_embedding": False,  # Reset — needs re-embedding
            "updated_by": updated_by,
        }
        if sensitivity is not None:
            update_fields["sensitivity"] = sensitivity
        if short_title is not None:
            update_fields["short_title"] = short_title
        if long_title is not None:
            update_fields["long_title"] = long_title
        if entry_type is not None:
            update_fields["entry_type"] = entry_type
        if project_ref is not None:
            update_fields["project_ref"] = project_ref
        if source_context is not None:
            update_fields["source_context"] = source_context

        updated = existing.model_copy(update=update_fields)
        await update_entry(self.db, updated)

        # Create version record
        version = EntryVersion(
            entry_id=entry_id,
            version_number=new_version,
            knowledge_details=knowledge_details,
            change_reason=change_reason,
            contributor=updated_by,
            confidence_level=new_confidence,
            created_at=now,
        )
        await insert_version(self.db, version)

        detail = change_reason or f"Updated to v{new_version}"
        await _record_audit_event(self.db, "entry_updated", entry_id, updated_by, detail)

        logger.info("Updated entry %s to v%d", entry_id, new_version)
        return updated

    async def get_entry(self, entry_id: str) -> KnowledgeEntry | None:
        """Get a single entry by ID."""
        return await get_entry(self.db, entry_id)

    async def mark_embedding(self, entry_id: str, has_embedding: bool = True) -> None:
        """Mark an entry as having (or not having) an embedding."""
        await self.db.execute(
            "UPDATE knowledge_entries SET has_embedding = ? WHERE id = ?",
            (int(has_embedding), entry_id),
        )
        await self.db.commit()

    async def deactivate_entry(
        self, entry_id: str, contributor: str | None = None
    ) -> KnowledgeEntry:
        """Deactivate an entry (soft-delete). Entry must exist and be active."""
        existing = await get_entry(self.db, entry_id)
        if existing is None:
            raise ValueError(f"Entry {entry_id} not found")
        if not existing.is_active:
            raise ValueError(f"Entry {entry_id} is already inactive")

        await deactivate_entry_db(self.db, entry_id)
        entry = await get_entry(self.db, entry_id)
        await _record_audit_event(
            self.db, "entry_deactivated", entry_id, contributor, existing.short_title
        )
        logger.info("Deactivated entry %s", entry_id)
        return entry  # type: ignore[return-value]

    async def reactivate_entry(
        self, entry_id: str, contributor: str | None = None
    ) -> KnowledgeEntry:
        """Reactivate a previously deactivated entry. Entry must exist and be inactive."""
        existing = await get_entry(self.db, entry_id)
        if existing is None:
            raise ValueError(f"Entry {entry_id} not found")
        if existing.is_active:
            raise ValueError(f"Entry {entry_id} is already active")

        await reactivate_entry_db(self.db, entry_id)
        entry = await get_entry(self.db, entry_id)
        await _record_audit_event(
            self.db, "entry_reactivated", entry_id, contributor, existing.short_title
        )
        logger.info("Reactivated entry %s", entry_id)
        return entry  # type: ignore[return-value]

    async def get_entries_without_embeddings(self, limit: int = 100) -> list[str]:
        """Get entry IDs that need embeddings."""
        cursor = await self.db.execute(
            "SELECT id FROM knowledge_entries WHERE has_embedding = 0 AND is_active = 1 LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [row["id"] for row in rows]
