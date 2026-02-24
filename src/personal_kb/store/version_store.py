"""Version history operations."""

from datetime import datetime

import aiosqlite

from personal_kb.models.version import EntryVersion


class VersionStore:
    """Read access to entry version history."""

    def __init__(self, db: aiosqlite.Connection):
        """Initialize with a database connection."""
        self.db = db

    async def get_versions(self, entry_id: str) -> list[EntryVersion]:
        """Get all versions of an entry, ordered by version number."""
        cursor = await self.db.execute(
            """SELECT entry_id, version_number, knowledge_details, change_reason,
            confidence_level, created_at
            FROM entry_versions WHERE entry_id = ? ORDER BY version_number""",
            (entry_id,),
        )
        rows = await cursor.fetchall()
        return [
            EntryVersion(
                entry_id=row["entry_id"],
                version_number=row["version_number"],
                knowledge_details=row["knowledge_details"],
                change_reason=row["change_reason"],
                confidence_level=row["confidence_level"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    async def get_latest_version(self, entry_id: str) -> EntryVersion | None:
        """Get the latest version of an entry."""
        cursor = await self.db.execute(
            """SELECT entry_id, version_number, knowledge_details, change_reason,
            confidence_level, created_at
            FROM entry_versions WHERE entry_id = ? ORDER BY version_number DESC LIMIT 1""",
            (entry_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return EntryVersion(
            entry_id=row["entry_id"],
            version_number=row["version_number"],
            knowledge_details=row["knowledge_details"],
            change_reason=row["change_reason"],
            confidence_level=row["confidence_level"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
