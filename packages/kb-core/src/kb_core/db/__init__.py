"""Database connection and schema management."""

from kb_core.db.backend import Cursor, Database, Row
from kb_core.db.sqlite_backend import SQLiteBackend

try:
    from kb_core.db.postgres_backend import PostgresBackend
except ImportError:
    PostgresBackend = None  # type: ignore[assignment,misc]

__all__ = ["Cursor", "Database", "PostgresBackend", "Row", "SQLiteBackend"]
