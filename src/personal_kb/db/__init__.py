"""Database connection and schema management."""

from personal_kb.db.backend import Cursor, Database, Row
from personal_kb.db.sqlite_backend import SQLiteBackend

try:
    from personal_kb.db.postgres_backend import PostgresBackend
except ImportError:
    PostgresBackend = None  # type: ignore[assignment,misc]

__all__ = ["Cursor", "Database", "PostgresBackend", "Row", "SQLiteBackend"]
