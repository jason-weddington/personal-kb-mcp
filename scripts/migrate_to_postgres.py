#!/usr/bin/env python3
"""Migrate a personal-kb SQLite database to PostgreSQL.

Usage:
    python scripts/migrate_to_postgres.py <postgres_url>
    python scripts/migrate_to_postgres.py postgresql://user:pass@localhost/kb

Options:
    --sqlite PATH   SQLite database path (default: KB_DB_PATH or
                    ~/.local/share/personal_kb/knowledge.db)
    --dry-run       Show what would be migrated without writing

The script:
1. Opens the SQLite database (read-only)
2. Connects to PostgreSQL and applies the schema
3. Copies all data tables in dependency order
4. Skips embeddings (rebuild with `kb_maintain rebuild_embeddings`)
5. Skips FTS data (Postgres tsvector trigger populates on insert)
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import aiosqlite


async def main() -> int:
    """Run the migration."""
    parser = argparse.ArgumentParser(description="Migrate personal-kb from SQLite to PostgreSQL")
    parser.add_argument("postgres_url", help="PostgreSQL connection URL")
    parser.add_argument(
        "--sqlite",
        default=None,
        help="SQLite database path (default: KB_DB_PATH env or "
        "~/.local/share/personal_kb/knowledge.db)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show counts without writing")
    args = parser.parse_args()

    # Resolve SQLite path
    sqlite_path = args.sqlite or os.environ.get(
        "KB_DB_PATH", str(Path("~/.local/share/personal_kb/knowledge.db").expanduser())
    )
    if not Path(sqlite_path).exists():
        print(f"Error: SQLite database not found at {sqlite_path}")
        return 1

    # Lazy import asyncpg — it's an optional dependency
    try:
        import asyncpg
    except ImportError:
        print("Error: asyncpg is required. Install with: uv sync --extra postgres")
        return 1

    # Open SQLite (read-only)
    sqlite = await aiosqlite.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    sqlite.row_factory = aiosqlite.Row

    # Count what we're migrating
    tables = [
        "knowledge_entries",
        "entry_versions",
        "entry_id_seq",
        "graph_nodes",
        "graph_edges",
        "ingested_files",
        "schema_version",
    ]

    counts: dict[str, int] = {}
    for table in tables:
        try:
            cursor = await sqlite.execute(f"SELECT COUNT(*) FROM [{table}]")  # noqa: S608
            row = await cursor.fetchone()
            counts[table] = row[0] if row else 0
        except aiosqlite.OperationalError:
            counts[table] = 0

    print("SQLite database:", sqlite_path)
    print()
    for table, count in counts.items():
        print(f"  {table}: {count} rows")

    total = sum(counts.values())
    if total == 0:
        print("\nNothing to migrate.")
        await sqlite.close()
        return 0

    if args.dry_run:
        print(f"\nDry run: {total} total rows would be migrated.")
        print("Embeddings will need to be rebuilt: kb_maintain rebuild_embeddings (force=True)")
        await sqlite.close()
        return 0

    # Connect to Postgres and apply schema
    print("\nConnecting to PostgreSQL...")
    pool = await asyncpg.create_pool(args.postgres_url, min_size=1, max_size=5)

    # Apply schema using the backend
    from personal_kb.db.postgres_backend import PostgresBackend

    pg = PostgresBackend(pool)

    # Get embedding dim from env
    embedding_dim = int(os.environ.get("KB_EMBEDDING_DIM", "1024"))
    await pg.apply_schema(embedding_dim=embedding_dim)
    print("Schema applied.")

    # Migrate in dependency order
    migrated = 0

    # 1. schema_version
    if counts["schema_version"] > 0:
        cursor = await sqlite.execute("SELECT version FROM schema_version")
        rows = await cursor.fetchall()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM schema_version")
            for row in rows:
                await conn.execute("INSERT INTO schema_version (version) VALUES ($1)", row[0])
        migrated += len(rows)
        print(f"  schema_version: {len(rows)} rows")

    # 2. entry_id_seq
    if counts["entry_id_seq"] > 0:
        cursor = await sqlite.execute("SELECT next_id FROM entry_id_seq")
        row = await cursor.fetchone()
        if row:
            async with pool.acquire() as conn:
                await conn.execute("UPDATE entry_id_seq SET next_id = $1", row[0])
            migrated += 1
            print(f"  entry_id_seq: next_id = {row[0]}")

    # 3. knowledge_entries (needed before entry_versions and graph_edges)
    if counts["knowledge_entries"] > 0:
        cursor = await sqlite.execute("SELECT * FROM knowledge_entries")
        rows = await cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]

        async with pool.acquire() as conn:
            for row in rows:
                values = [row[col] for col in cols]
                placeholders = ", ".join(f"${i + 1}" for i in range(len(cols)))
                col_names = ", ".join(cols)
                await conn.execute(
                    f"INSERT INTO knowledge_entries ({col_names}) VALUES ({placeholders})"  # noqa: S608
                    " ON CONFLICT (id) DO NOTHING",
                    *values,
                )
        migrated += len(rows)
        print(f"  knowledge_entries: {len(rows)} rows")

    # 4. entry_versions
    if counts["entry_versions"] > 0:
        cursor = await sqlite.execute(
            "SELECT entry_id, version_number, knowledge_details, "
            "change_reason, confidence_level, created_at FROM entry_versions"
        )
        rows = await cursor.fetchall()
        async with pool.acquire() as conn:
            for row in rows:
                await conn.execute(
                    "INSERT INTO entry_versions "
                    "(entry_id, version_number, knowledge_details, "
                    "change_reason, confidence_level, created_at) "
                    "VALUES ($1, $2, $3, $4, $5, $6) "
                    "ON CONFLICT (entry_id, version_number) DO NOTHING",
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                )
        migrated += len(rows)
        print(f"  entry_versions: {len(rows)} rows")

    # 5. graph_nodes (needed before graph_edges)
    if counts["graph_nodes"] > 0:
        cursor = await sqlite.execute(
            "SELECT node_id, node_type, properties, created_at FROM graph_nodes"
        )
        rows = await cursor.fetchall()
        async with pool.acquire() as conn:
            for row in rows:
                await conn.execute(
                    "INSERT INTO graph_nodes (node_id, node_type, properties, created_at) "
                    "VALUES ($1, $2, $3, $4) ON CONFLICT (node_id) DO NOTHING",
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                )
        migrated += len(rows)
        print(f"  graph_nodes: {len(rows)} rows")

    # 6. graph_edges
    if counts["graph_edges"] > 0:
        cursor = await sqlite.execute(
            "SELECT source, target, edge_type, properties, created_at FROM graph_edges"
        )
        rows = await cursor.fetchall()
        async with pool.acquire() as conn:
            for row in rows:
                await conn.execute(
                    "INSERT INTO graph_edges (source, target, edge_type, properties, created_at) "
                    "VALUES ($1, $2, $3, $4, $5) "
                    "ON CONFLICT (source, target, edge_type) DO NOTHING",
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                )
        migrated += len(rows)
        print(f"  graph_edges: {len(rows)} rows")

    # 7. ingested_files
    if counts["ingested_files"] > 0:
        cursor = await sqlite.execute(
            "SELECT relative_path, content_hash, note_node_id, entry_ids, summary, "
            "file_size, file_extension, project_ref, redactions, ingested_at, "
            "updated_at, is_active FROM ingested_files"
        )
        rows = await cursor.fetchall()
        async with pool.acquire() as conn:
            for row in rows:
                await conn.execute(
                    "INSERT INTO ingested_files "
                    "(relative_path, content_hash, note_node_id, entry_ids, summary, "
                    "file_size, file_extension, project_ref, redactions, ingested_at, "
                    "updated_at, is_active) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12) "
                    "ON CONFLICT (relative_path) DO NOTHING",
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    row[7],
                    row[8],
                    row[9],
                    row[10],
                    row[11],
                )
        migrated += len(rows)
        print(f"  ingested_files: {len(rows)} rows")

    print(f"\nMigrated {migrated} rows total.")
    print()
    print("Next steps:")
    print("  1. Set KB_DATABASE_URL in your MCP config to point to PostgreSQL")
    print("  2. Rebuild embeddings: kb_maintain rebuild_embeddings (force=True)")
    print("     (Embeddings are binary blobs in a different format — they can't be copied)")

    await sqlite.close()
    await pool.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
