#!/usr/bin/env python3
"""Migrate a personal-kb SQLite database to PostgreSQL.

Supports two modes:
- FRESH: Target Postgres is empty -> IDs copied as-is
- MERGE: Target has entries -> source IDs remapped to avoid collisions

Usage:
    uv run python scripts/migrate_sqlite_to_pg.py ~/.local/share/personal_kb/knowledge.db
    uv run python scripts/migrate_sqlite_to_pg.py --dry-run ~/kb/knowledge.db
    uv run python scripts/migrate_sqlite_to_pg.py --contributor jason --team infra ~/kb/knowledge.db

Target Postgres connection comes from environment variables:
    KB_DATABASE_URL     PostgreSQL connection URL (required)
    KB_PG_IAM_AUTH      Set TRUE for Aurora IAM authentication
    KB_PG_REGION        AWS region for IAM token signing (default: us-east-1)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from personal_kb.db.backend import Database

logger = logging.getLogger(__name__)

# Column lists for each migrated table (after v2 multi-user migration).
_KE_COLS = [
    "id",
    "project_ref",
    "short_title",
    "long_title",
    "knowledge_details",
    "entry_type",
    "source_context",
    "confidence_level",
    "tags",
    "hints",
    "created_at",
    "updated_at",
    "last_accessed",
    "superseded_by",
    "is_active",
    "has_embedding",
    "version",
    "contributor",
    "team",
    "updated_by",
    "sensitivity",
]

_EV_COLS = [
    "entry_id",
    "version_number",
    "knowledge_details",
    "change_reason",
    "confidence_level",
    "created_at",
    "contributor",
]

_GN_COLS = ["node_id", "node_type", "properties", "created_at"]

_GE_COLS = ["source", "target", "edge_type", "properties", "created_at"]

_IF_COLS = [
    "relative_path",
    "content_hash",
    "note_node_id",
    "entry_ids",
    "summary",
    "file_size",
    "file_extension",
    "project_ref",
    "redactions",
    "ingested_at",
    "updated_at",
    "is_active",
    "contributor",
]

_AE_COLS = ["event_type", "entry_id", "contributor", "detail", "created_at"]


# ---------------------------------------------------------------------------
# ID remapping
# ---------------------------------------------------------------------------


def _remap_entry_id(old_id: str, offset: int) -> str:
    """kb-00042 with offset=100 -> kb-00142."""
    num = int(old_id.split("-")[1])
    return f"kb-{num + offset:05d}"


def _remap_node_id(node_id: str, offset: int) -> str:
    """entry:kb-00042 -> entry:kb-00142; tag:python -> tag:python."""
    if node_id.startswith("entry:"):
        return "entry:" + _remap_entry_id(node_id[6:], offset)
    return node_id


def _remap_json_entry_ids(json_str: str | None, offset: int) -> str | None:
    """Remap entry IDs inside a JSON array string."""
    if not json_str:
        return json_str
    try:
        ids = json.loads(json_str)
        if isinstance(ids, list):
            return json.dumps([_remap_entry_id(eid, offset) for eid in ids])
    except (json.JSONDecodeError, ValueError):
        pass
    return json_str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _count(db: Database, table: str) -> int:
    """Count rows in a table, returning 0 if it doesn't exist."""
    try:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM " + table  # noqa: S608
        )
        row = await cursor.fetchone()
        return row[0] if row else 0
    except Exception:
        return 0


async def _max_entry_number(db: Database) -> int:
    """Find the highest numeric part of any entry ID in the DB."""
    cursor = await db.execute("SELECT id FROM knowledge_entries")
    rows = await cursor.fetchall()
    max_num = 0
    for r in rows:
        try:
            num = int(r[0].split("-")[1])
            max_num = max(max_num, num)
        except (IndexError, ValueError):
            pass
    return max_num


def _insert_sql(table: str, cols: list[str], *, conflict: str = "") -> str:
    """Build an INSERT statement with ? placeholders."""
    col_names = ", ".join(cols)
    placeholders = ", ".join("?" for _ in cols)
    sql = "INSERT INTO " + table + " (" + col_names + ") VALUES (" + placeholders + ")"  # noqa: S608
    if conflict:
        sql += " ON CONFLICT " + conflict + " DO NOTHING"
    return sql


# ---------------------------------------------------------------------------
# Table copiers — each returns (rows_copied, errors)
# ---------------------------------------------------------------------------


async def _copy_knowledge_entries(
    source: Database,
    target: Database,
    offset: int,
    contributor: str | None,
    team: str | None,
) -> tuple[int, list[str]]:
    select_sql = "SELECT " + ", ".join(_KE_COLS) + " FROM knowledge_entries"  # noqa: S608
    cursor = await source.execute(select_sql)
    rows = await cursor.fetchall()

    insert = _insert_sql("knowledge_entries", _KE_COLS, conflict="(id)")
    copied = 0
    errors: list[str] = []

    for row in rows:
        try:
            vals = {col: row[i] for i, col in enumerate(_KE_COLS)}
            vals["id"] = _remap_entry_id(vals["id"], offset)
            if vals["superseded_by"]:
                vals["superseded_by"] = _remap_entry_id(vals["superseded_by"], offset)
            vals["has_embedding"] = 0
            if contributor and not vals.get("contributor"):
                vals["contributor"] = contributor
            if team and not vals.get("team"):
                vals["team"] = team
            await target.execute(insert, tuple(vals[c] for c in _KE_COLS))
            copied += 1
        except Exception as e:
            errors.append(f"  knowledge_entries: {e}")

    await target.commit()
    return copied, errors


async def _copy_entry_versions(
    source: Database,
    target: Database,
    offset: int,
    contributor: str | None,
) -> tuple[int, list[str]]:
    select_sql = "SELECT " + ", ".join(_EV_COLS) + " FROM entry_versions"  # noqa: S608
    cursor = await source.execute(select_sql)
    rows = await cursor.fetchall()

    insert = _insert_sql("entry_versions", _EV_COLS, conflict="(entry_id, version_number)")
    copied = 0
    errors: list[str] = []

    for row in rows:
        try:
            vals = {col: row[i] for i, col in enumerate(_EV_COLS)}
            vals["entry_id"] = _remap_entry_id(vals["entry_id"], offset)
            if contributor and not vals.get("contributor"):
                vals["contributor"] = contributor
            await target.execute(insert, tuple(vals[c] for c in _EV_COLS))
            copied += 1
        except Exception as e:
            errors.append(f"  entry_versions: {e}")

    await target.commit()
    return copied, errors


async def _copy_graph_nodes(
    source: Database,
    target: Database,
    offset: int,
) -> tuple[int, list[str]]:
    select_sql = "SELECT " + ", ".join(_GN_COLS) + " FROM graph_nodes"  # noqa: S608
    cursor = await source.execute(select_sql)
    rows = await cursor.fetchall()

    insert = _insert_sql("graph_nodes", _GN_COLS, conflict="(node_id)")
    copied = 0
    errors: list[str] = []

    for row in rows:
        try:
            vals = [row[i] for i in range(len(_GN_COLS))]
            vals[0] = _remap_node_id(vals[0], offset)  # node_id
            await target.execute(insert, tuple(vals))
            copied += 1
        except Exception as e:
            errors.append(f"  graph_nodes: {e}")

    await target.commit()
    return copied, errors


async def _copy_graph_edges(
    source: Database,
    target: Database,
    offset: int,
) -> tuple[int, list[str]]:
    select_sql = "SELECT " + ", ".join(_GE_COLS) + " FROM graph_edges"  # noqa: S608
    cursor = await source.execute(select_sql)
    rows = await cursor.fetchall()

    insert = _insert_sql("graph_edges", _GE_COLS, conflict="(source, target, edge_type)")
    copied = 0
    errors: list[str] = []

    for row in rows:
        try:
            vals = [row[i] for i in range(len(_GE_COLS))]
            vals[0] = _remap_node_id(vals[0], offset)  # source
            vals[1] = _remap_node_id(vals[1], offset)  # target
            await target.execute(insert, tuple(vals))
            copied += 1
        except Exception as e:
            errors.append(f"  graph_edges: {e}")

    await target.commit()
    return copied, errors


async def _copy_ingested_files(
    source: Database,
    target: Database,
    offset: int,
    contributor: str | None,
) -> tuple[int, list[str]]:
    select_sql = "SELECT " + ", ".join(_IF_COLS) + " FROM ingested_files"  # noqa: S608
    try:
        cursor = await source.execute(select_sql)
    except Exception:
        return 0, []
    rows = await cursor.fetchall()

    insert = _insert_sql("ingested_files", _IF_COLS, conflict="(relative_path)")
    copied = 0
    errors: list[str] = []

    for row in rows:
        try:
            vals = {col: row[i] for i, col in enumerate(_IF_COLS)}
            vals["entry_ids"] = _remap_json_entry_ids(vals["entry_ids"], offset)
            if contributor and not vals.get("contributor"):
                vals["contributor"] = contributor
            await target.execute(insert, tuple(vals[c] for c in _IF_COLS))
            copied += 1
        except Exception as e:
            errors.append(f"  ingested_files: {e}")

    await target.commit()
    return copied, errors


async def _copy_audit_events(
    source: Database,
    target: Database,
    offset: int,
) -> tuple[int, list[str]]:
    select_sql = "SELECT " + ", ".join(_AE_COLS) + " FROM audit_events"  # noqa: S608
    try:
        cursor = await source.execute(select_sql)
    except Exception:
        return 0, []
    rows = await cursor.fetchall()

    insert = _insert_sql("audit_events", _AE_COLS)
    copied = 0
    errors: list[str] = []

    for row in rows:
        try:
            vals = [row[i] for i in range(len(_AE_COLS))]
            if vals[1]:  # entry_id
                vals[1] = _remap_entry_id(vals[1], offset)
            await target.execute(insert, tuple(vals))
            copied += 1
        except Exception as e:
            errors.append(f"  audit_events: {e}")

    await target.commit()
    return copied, errors


# ---------------------------------------------------------------------------
# Embedding rebuild
# ---------------------------------------------------------------------------


async def _rebuild_embeddings(target: Database) -> tuple[int, int]:
    """Embed all active entries via Ollama into the target backend."""
    from personal_kb.db.queries import get_entry
    from personal_kb.search.embeddings import EmbeddingClient

    embedder = EmbeddingClient(target)
    if not await embedder.is_available():
        print("\n  Ollama not reachable — skipping embedding rebuild.")
        print("  Rebuild later with: kb_maintain rebuild_embeddings (force=True)")
        await embedder.close()
        return 0, 0

    cursor = await target.execute(
        "SELECT id FROM knowledge_entries WHERE is_active = 1 ORDER BY id"
    )
    rows = await cursor.fetchall()
    total = len(rows)
    print(f"\nRebuilding embeddings for {total} active entries...")

    succeeded = 0
    failed = 0
    for i, row in enumerate(rows, 1):
        entry_id = row[0]
        entry = await get_entry(target, entry_id)
        if entry is None:
            failed += 1
            continue
        try:
            embedding = await embedder.embed(entry.embedding_text)
            if embedding is not None:
                await target.vector_store(entry_id, embedding)
                await target.commit()
                await target.execute(
                    "UPDATE knowledge_entries SET has_embedding = 1 WHERE id = ?",
                    (entry_id,),
                )
                await target.commit()
                succeeded += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"    Failed to embed {entry_id}: {e}")

        if i % 25 == 0 or i == total:
            print(f"    {i}/{total} entries processed...")

    await embedder.close()
    return succeeded, failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Run the SQLite -> PostgreSQL migration."""
    parser = argparse.ArgumentParser(
        description="Migrate personal-kb from SQLite to PostgreSQL",
        epilog="Target Postgres comes from KB_DATABASE_URL env var. "
        "Set KB_PG_IAM_AUTH=TRUE for Aurora IAM auth.",
    )
    parser.add_argument("source", help="Path to source SQLite database file")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without writing")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding rebuild")
    parser.add_argument("--contributor", default=None, help="Attribution: contributor name")
    parser.add_argument("--team", default=None, help="Attribution: team name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    # --- Validate inputs ---
    source_path = Path(args.source).expanduser()
    if not source_path.exists():
        print(f"Error: SQLite database not found at {source_path}")
        return 1

    from personal_kb.config import get_contributor, get_database_url, get_embedding_dim, get_team

    target_url = get_database_url()
    if not target_url or not target_url.startswith("postgresql"):
        print("Error: KB_DATABASE_URL must be set to a postgresql:// URL")
        return 1

    try:
        import asyncpg  # noqa: F401
    except ImportError:
        print("Error: asyncpg required. Install with: uv sync --extra postgres")
        return 1

    contributor = args.contributor or get_contributor()
    team = args.team or get_team()
    embedding_dim = get_embedding_dim()

    # --- Open source SQLite (runs migrations -> schema is current) ---
    from personal_kb.db.connection import _create_postgres, _create_sqlite

    print(f"Source: {source_path}")
    source_db = await _create_sqlite(str(source_path), embedding_dim=embedding_dim)

    target_db: Database | None = None
    try:
        # Count source tables
        counts: dict[str, int] = {}
        for table in [
            "knowledge_entries",
            "entry_versions",
            "graph_nodes",
            "graph_edges",
            "ingested_files",
            "audit_events",
        ]:
            counts[table] = await _count(source_db, table)

        print("\nSource tables:")
        for table, count in counts.items():
            print(f"  {table}: {count} rows")

        active_cursor = await source_db.execute(
            "SELECT COUNT(*) FROM knowledge_entries WHERE is_active = 1"
        )
        active_row = await active_cursor.fetchone()
        active_count = active_row[0] if active_row else 0

        total_source = sum(counts.values())
        if total_source == 0:
            print("\nNothing to migrate.")
            return 0

        # --- Open target Postgres ---
        target_db = await _create_postgres(target_url, embedding_dim=embedding_dim)
        display_url = target_url.split("@")[-1] if "@" in target_url else target_url
        print(f"\nTarget: {display_url}")
        print("Schema applied.")

        # --- Detect mode ---
        target_entry_count = await _count(target_db, "knowledge_entries")
        if target_entry_count == 0:
            mode = "FRESH"
            offset = 0
        else:
            mode = "MERGE"
            offset = await _max_entry_number(target_db)

        print(f"\nMode: {mode}" + (f" (offset={offset})" if offset else ""))
        if contributor:
            print(f"Attribution: {contributor}" + (f"/{team}" if team else ""))

        # --- Dry run ---
        if args.dry_run:
            print(f"\nDry run: {total_source} rows would be migrated.")
            if offset:
                print(f"  Entry IDs remapped with offset +{offset}")
            if not args.skip_embeddings:
                print(f"  {active_count} active entries would be re-embedded via Ollama.")
            return 0

        # --- Copy tables ---
        all_errors: list[str] = []

        print("\nCopying tables...")
        n, errs = await _copy_knowledge_entries(source_db, target_db, offset, contributor, team)
        all_errors.extend(errs)
        print(f"  knowledge_entries: {n} rows")

        n, errs = await _copy_entry_versions(source_db, target_db, offset, contributor)
        all_errors.extend(errs)
        print(f"  entry_versions: {n} rows")

        n, errs = await _copy_graph_nodes(source_db, target_db, offset)
        all_errors.extend(errs)
        print(f"  graph_nodes: {n} rows")

        n, errs = await _copy_graph_edges(source_db, target_db, offset)
        all_errors.extend(errs)
        print(f"  graph_edges: {n} rows")

        n, errs = await _copy_ingested_files(source_db, target_db, offset, contributor)
        all_errors.extend(errs)
        print(f"  ingested_files: {n} rows")

        n, errs = await _copy_audit_events(source_db, target_db, offset)
        all_errors.extend(errs)
        print(f"  audit_events: {n} rows")

        # --- Update entry_id_seq to max(all IDs) + 1 ---
        max_num = await _max_entry_number(target_db)
        next_id = max_num + 1
        await target_db.execute("UPDATE entry_id_seq SET next_id = ?", (next_id,))
        await target_db.commit()
        print(f"\n  entry_id_seq: next_id = {next_id}")

        # --- Record migration audit events ---
        now_iso = datetime.now(UTC).isoformat()
        src_cursor = await source_db.execute("SELECT id FROM knowledge_entries")
        src_rows = await src_cursor.fetchall()
        audit_n = 0
        for r in src_rows:
            entry_id = _remap_entry_id(r[0], offset)
            try:
                await target_db.execute(
                    "INSERT INTO audit_events"
                    " (event_type, entry_id, contributor, detail, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (
                        "migrated",
                        entry_id,
                        contributor,
                        "Migrated from " + source_path.name,
                        now_iso,
                    ),
                )
                audit_n += 1
            except Exception:
                logger.debug("Failed to record audit event for %s", entry_id)
        await target_db.commit()
        print(f"  audit: {audit_n} 'migrated' events recorded")

        # --- Re-embed ---
        if not args.skip_embeddings and active_count > 0:
            succeeded, failed = await _rebuild_embeddings(target_db)
            print(f"  Embeddings: {succeeded} succeeded, {failed} failed")
        elif args.skip_embeddings:
            print(
                "\nEmbeddings skipped. Rebuild later: kb_maintain rebuild_embeddings (force=True)"
            )

        # --- Summary ---
        if all_errors:
            print(f"\n{len(all_errors)} errors:")
            for err in all_errors:
                print(err)

        print("\nDone.")
        return 1 if all_errors else 0

    finally:
        await source_db.close()
        if target_db is not None:
            await target_db.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
