"""CRUD operations for knowledge entries."""

import logging
from datetime import UTC, datetime
from typing import Literal, cast

from personal_kb.db.backend import Database
from personal_kb.db.queries import (
    deactivate_entry_db,
    get_entry,
    insert_entry,
    insert_version,
    next_entry_id,
    reactivate_entry_db,
    row_to_entry,
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
    """CRUD operations for knowledge entries with versioning.

    CONVENTION: All write paths that touch multiple tables (entry + version +
    audit) MUST be wrapped in ``async with self.db.transaction()``. This
    ensures atomicity — a failure mid-write rolls back cleanly instead of
    leaving partial updates. See create_entry, update_entry, bulk_update.
    """

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
        expires_at: datetime | None = None,
    ) -> KnowledgeEntry:
        """Create a new knowledge entry with initial version."""
        async with self.db.transaction():
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
                expires_at=expires_at,
                version=1,
            )
            await insert_entry(self.db, entry)

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
        knowledge_details: str | None = None,
        change_reason: str | None = None,
        confidence_level: float | None = None,
        tags: list[str] | None = None,
        hints: dict[str, object] | None = None,
        updated_by: str | None = None,
        sensitivity: Literal["internal", "restricted", "public"] | None = None,
        expires_at: datetime | None = None,
        short_title: str | None = None,
        long_title: str | None = None,
        entry_type: EntryType | None = None,
        project_ref: str | None = None,
        source_context: str | None = None,
    ) -> KnowledgeEntry:
        """Update an existing entry, creating a new version."""
        async with self.db.transaction():
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

            merged_hints = dict(existing.hints)
            if hints:
                merged_hints.update(hints)

            content_changed = knowledge_details is not None
            effective_details = (
                knowledge_details
                if knowledge_details is not None
                else (existing.knowledge_details or "")
            )

            update_fields: dict[str, object] = {
                "knowledge_details": effective_details,
                "confidence_level": new_confidence,
                "tags": tags if tags is not None else existing.tags,
                "hints": merged_hints,
                "updated_at": now,
                "version": new_version,
                "updated_by": updated_by,
            }
            if content_changed:
                update_fields["has_embedding"] = False
            if sensitivity is not None:
                update_fields["sensitivity"] = sensitivity
            if expires_at is not None:
                update_fields["expires_at"] = expires_at
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

            version = EntryVersion(
                entry_id=entry_id,
                version_number=new_version,
                knowledge_details=effective_details,
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
        async with self.db.transaction():
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
        async with self.db.transaction():
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

    async def bulk_update(
        self,
        filters: dict[str, object],
        updates: dict[str, object],
        *,
        contributor: str | None = None,
        dry_run: bool = False,
    ) -> list[tuple[KnowledgeEntry, KnowledgeEntry]]:
        """Find entries matching filters and apply metadata updates.

        Returns list of (before, after) pairs. In dry_run mode, 'after' is a
        preview copy — nothing is persisted.

        Supported filters: contributor, team, project_ref (use None value to
        match unset), tags, entry_ids, entry_type.

        Supported updates: project_ref, entry_type, confidence_level,
        tags_add, tags_remove.
        """
        # Build WHERE clause
        sql = "SELECT * FROM knowledge_entries WHERE is_active = 1"
        params: list[object] = []

        if "contributor" in filters:
            sql += " AND contributor = ?"
            params.append(filters["contributor"])
        if "team" in filters:
            sql += " AND team = ?"
            params.append(filters["team"])
        if "project_ref" in filters:
            val = filters["project_ref"]
            if val is None:
                sql += " AND project_ref IS NULL"
            else:
                sql += " AND project_ref = ?"
                params.append(val)
        if "entry_type" in filters:
            sql += " AND entry_type = ?"
            params.append(str(filters["entry_type"]))
        if "tags" in filters:
            filter_tags = cast("list[str]", filters["tags"])
            for tag in filter_tags:
                sql += " AND (' ' || tags || ' ') LIKE ?"
                params.append(f"% {tag} %")
        if "entry_ids" in filters:
            ids = filters["entry_ids"]
            if isinstance(ids, list) and ids:
                placeholders = ",".join("?" for _ in ids)
                sql += f" AND id IN ({placeholders})"
                params.extend(ids)

        sql += " ORDER BY id"
        cursor = await self.db.execute(sql, tuple(params))
        rows = await cursor.fetchall()

        results: list[tuple[KnowledgeEntry, KnowledgeEntry]] = []
        for row in rows:
            entry = row_to_entry(row)

            # Compute updated fields
            update_fields: dict[str, object] = {}
            if "project_ref" in updates and updates["project_ref"] != entry.project_ref:
                update_fields["project_ref"] = updates["project_ref"]
            if "entry_type" in updates:
                new_type = EntryType(str(updates["entry_type"]))
                if new_type != entry.entry_type:
                    update_fields["entry_type"] = new_type
            if "confidence_level" in updates:
                new_conf = float(str(updates["confidence_level"]))
                if new_conf != entry.confidence_level:
                    update_fields["confidence_level"] = new_conf

            # Tag operations
            new_tags = list(entry.tags)
            if "tags_add" in updates:
                add_tags = cast("list[str]", updates["tags_add"])
                for t in add_tags:
                    if t not in new_tags:
                        new_tags.append(t)
            if "tags_remove" in updates:
                remove_set = set(cast("list[str]", updates["tags_remove"]))
                new_tags = [t for t in new_tags if t not in remove_set]
            if new_tags != list(entry.tags):
                update_fields["tags"] = new_tags

            if not update_fields:
                continue

            now = datetime.now(UTC)
            new_version = entry.version + 1
            update_fields["version"] = new_version
            update_fields["updated_at"] = now
            update_fields["updated_by"] = contributor

            after = entry.model_copy(update=update_fields)

            if not dry_run:
                async with self.db.transaction():
                    await update_entry(self.db, after)
                    version = EntryVersion(
                        entry_id=entry.id,
                        version_number=new_version,
                        knowledge_details=entry.knowledge_details,
                        change_reason="Bulk metadata update",
                        contributor=contributor,
                        confidence_level=after.confidence_level,
                        created_at=now,
                    )
                    await insert_version(self.db, version)
                    await _record_audit_event(
                        self.db, "entry_updated", entry.id, contributor, "Bulk metadata update"
                    )

            results.append((entry, after))

        return results

    async def get_entries_without_embeddings(self, limit: int = 100) -> list[str]:
        """Get entry IDs that need embeddings."""
        cursor = await self.db.execute(
            "SELECT id FROM knowledge_entries WHERE has_embedding = 0 AND is_active = 1 LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [row["id"] for row in rows]
