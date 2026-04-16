"""kb_maintain MCP tool — database maintenance operations."""

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

from personal_kb.db.backend import Database
from personal_kb.db.queries import (
    delete_entry_cascade,
    get_all_active_entry_ids,
    get_db_stats,
    get_entry,
)
from personal_kb.graph.builder import GraphBuilder
from personal_kb.graph.enricher import GraphEnricher
from personal_kb.llm.provider import LLMProvider
from personal_kb.models.entry import KnowledgeEntry  # noqa: TC001
from personal_kb.search.embeddings import EmbeddingClient
from personal_kb.store.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)

_ACTIONS = {
    "stats",
    "deactivate",
    "reactivate",
    "rebuild_embeddings",
    "rebuild_graph",
    "purge_inactive",
    "vacuum",
    "entry_versions",
    "list_feedback",
    "summarize_feedback",
    "search_stats",
    "list_contributors",
    "list_audit",
}


def register_kb_maintain(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_maintain tool with the MCP server."""

    @mcp.tool(name=f"{prefix}maintain")
    async def kb_maintain(
        action: Annotated[
            str,
            Field(
                description=(
                    "Maintenance action: stats, deactivate, reactivate, "
                    "rebuild_embeddings, rebuild_graph, purge_inactive, vacuum, "
                    "entry_versions, list_feedback, summarize_feedback, search_stats, "
                    "list_contributors, list_audit"
                ),
            ),
        ],
        entry_id: Annotated[
            str | None,
            Field(description="Required for deactivate, reactivate, entry_versions"),
        ] = None,
        days_inactive: Annotated[
            int,
            Field(description="For purge_inactive: min days since deactivation", ge=1),
        ] = 90,
        force: Annotated[
            bool,
            Field(description="For rebuild_embeddings: re-embed ALL (not just missing)"),
        ] = False,
        confirm: Annotated[
            bool,
            Field(description="Required True for purge_inactive"),
        ] = False,
        feedback_type: Annotated[
            str | None,
            Field(description="For list_feedback: filter by type (missing, unhelpful, friction)"),
        ] = None,
        since: Annotated[
            str | None,
            Field(description="ISO date for list_feedback/summarize_feedback/search_stats"),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Administrative maintenance operations for the knowledge base.

        Requires KB_MANAGER=TRUE environment variable.

        Actions:
        - stats: Database overview (entry counts, graph stats, embeddings)
        - deactivate: Soft-delete an entry (requires entry_id)
        - reactivate: Undo deactivation (requires entry_id)
        - rebuild_embeddings: Re-embed entries (force=True for all)
        - rebuild_graph: Full graph reconstruction from all active entries
        - purge_inactive: Hard-delete entries inactive for N+ days (requires confirm=True)
        - vacuum: Optimize database (PRAGMA optimize + VACUUM)
        - entry_versions: Show version history (requires entry_id)
        - list_feedback: List recent agent feedback (optional: feedback_type, since)
        - summarize_feedback: LLM-clustered summary of feedback themes (optional: since)
        - search_stats: Search telemetry overview (optional: since)
        - list_contributors: Show contributor/team stats for active entries
        - list_audit: Recent audit events (optional: entry_id, since)
        """
        if ctx is None:
            raise RuntimeError("Context not injected")

        if action not in _ACTIONS:
            return f"Unknown action '{action}'. Use: {', '.join(sorted(_ACTIONS))}"

        lifespan = ctx.lifespan_context
        db: Database = lifespan["db"]
        store: KnowledgeStore = lifespan["store"]
        embedder: EmbeddingClient = lifespan["embedder"]
        graph_builder: GraphBuilder = lifespan["graph_builder"]
        graph_enricher: GraphEnricher | None = lifespan.get("graph_enricher")
        query_llm: LLMProvider | None = lifespan.get("query_llm")
        contributor: str | None = lifespan.get("contributor")

        if action == "stats":
            return await _action_stats(db)
        elif action == "deactivate":
            return await _action_deactivate(db, store, entry_id, contributor)
        elif action == "reactivate":
            return await _action_reactivate(
                db, store, graph_builder, graph_enricher, entry_id, contributor
            )
        elif action == "rebuild_embeddings":
            return await _action_rebuild_embeddings(db, store, embedder, force)
        elif action == "rebuild_graph":
            return await _action_rebuild_graph(db, graph_builder, graph_enricher)
        elif action == "purge_inactive":
            return await _action_purge_inactive(db, days_inactive, confirm)
        elif action == "vacuum":
            return await _action_vacuum(db)
        elif action == "entry_versions":
            return await _action_entry_versions(db, entry_id)
        elif action == "list_feedback":
            return await _action_list_feedback(db, feedback_type, since)
        elif action == "summarize_feedback":
            return await _action_summarize_feedback(db, query_llm, since)
        elif action == "search_stats":
            return await _action_search_stats(db, since)
        elif action == "list_contributors":
            return await _action_list_contributors(db)
        elif action == "list_audit":
            return await _action_list_audit(db, entry_id, since)

        return "Action not implemented."


async def _action_stats(db: Database) -> str:
    """Database overview with counts."""
    stats = await get_db_stats(db)

    lines = ["Knowledge Base Statistics\n"]
    lines.append(
        f"Entries: {stats['total_entries']} total"
        f" ({stats['active_entries']} active, {stats['inactive_entries']} inactive)"
    )

    by_type = stats.get("by_type", {})
    if by_type:
        lines.append("\nActive entries by type:")
        for entry_type, count in by_type.items():
            lines.append(f"  {entry_type}: {count}")

    by_project = stats.get("by_project", {})
    if by_project:
        lines.append("\nActive entries by project:")
        for project, count in by_project.items():
            lines.append(f"  {project}: {count}")

    lines.append(
        f"\nEmbeddings: {stats['with_embeddings']} with, {stats['without_embeddings']} without"
    )

    nodes_by_type = stats.get("graph_nodes_by_type", {})
    if nodes_by_type:
        total_nodes = sum(nodes_by_type.values())
        lines.append(f"\nGraph nodes: {total_nodes}")
        for node_type, count in nodes_by_type.items():
            lines.append(f"  {node_type}: {count}")

    edges_by_type = stats.get("graph_edges_by_type", {})
    if edges_by_type:
        total_edges = sum(edges_by_type.values())
        lines.append(f"\nGraph edges: {total_edges}")
        for edge_type, count in edges_by_type.items():
            lines.append(f"  {edge_type}: {count}")

    return "\n".join(lines)


async def _action_deactivate(
    db: Database,
    store: KnowledgeStore,
    entry_id: str | None,
    contributor: str | None = None,
) -> str:
    """Soft-delete an entry and clean up graph edges."""
    if not entry_id:
        return "Error: entry_id is required for deactivate action."

    try:
        entry = await store.deactivate_entry(entry_id, contributor=contributor)
    except ValueError as e:
        return f"Error: {e}"

    # Remove outgoing graph edges
    await db.execute("DELETE FROM graph_edges WHERE source = ?", (entry_id,))
    await db.commit()

    return f"Deactivated entry {entry.id}: {entry.short_title}"


async def _action_reactivate(
    db: Database,
    store: KnowledgeStore,
    graph_builder: GraphBuilder,
    graph_enricher: GraphEnricher | None,
    entry_id: str | None,
    contributor: str | None = None,
) -> str:
    """Reactivate a deactivated entry and rebuild graph edges."""
    if not entry_id:
        return "Error: entry_id is required for reactivate action."

    try:
        entry = await store.reactivate_entry(entry_id, contributor=contributor)
    except ValueError as e:
        return f"Error: {e}"

    # Rebuild graph edges
    try:
        await graph_builder.build_for_entry(entry)
    except Exception:
        logger.warning("Failed to rebuild graph for %s", entry_id, exc_info=True)

    # Enrich graph via LLM
    if graph_enricher:
        try:
            await graph_enricher.enrich_entry(entry)
        except Exception:
            logger.warning("Failed to enrich graph for %s", entry_id, exc_info=True)

    return f"Reactivated entry {entry.id}: {entry.short_title}"


async def _action_rebuild_embeddings(
    db: Database,
    store: KnowledgeStore,
    embedder: EmbeddingClient,
    force: bool,
) -> str:
    """Re-embed entries. force=True re-embeds all, otherwise only missing."""
    if not await embedder.is_available():
        return "Ollama is not available. Cannot rebuild embeddings."

    if force:
        entry_ids = await get_all_active_entry_ids(db)
    else:
        entry_ids = await store.get_entries_without_embeddings(limit=10000)

    if not entry_ids:
        return "No entries need embedding."

    succeeded = 0
    failed = 0

    for eid in entry_ids:
        entry = await get_entry(db, eid)
        if entry is None:
            failed += 1
            continue
        try:
            embedding = await embedder.embed(entry.embedding_text)
            if embedding is not None:
                await embedder.store_embedding(eid, embedding)
                await store.mark_embedding(eid, True)
                succeeded += 1
            else:
                failed += 1
        except Exception:
            logger.warning("Failed to embed %s", eid, exc_info=True)
            failed += 1

    mode = "all entries" if force else "entries without embeddings"
    return (
        f"Rebuild embeddings ({mode}): {len(entry_ids)} processed,"
        f" {succeeded} succeeded, {failed} failed"
    )


async def _action_rebuild_graph(
    db: Database,
    graph_builder: GraphBuilder,
    graph_enricher: GraphEnricher | None,
) -> str:
    """Full graph reconstruction from all active entries."""
    # Clear entire graph
    await db.execute("DELETE FROM graph_edges")
    await db.execute("DELETE FROM graph_nodes")
    await db.commit()

    entry_ids = await get_all_active_entry_ids(db)

    entries: list[KnowledgeEntry] = []
    processed = 0
    for eid in entry_ids:
        entry = await get_entry(db, eid)
        if entry is None:
            continue
        try:
            await graph_builder.build_for_entry(entry)
            entries.append(entry)
            processed += 1
        except Exception:
            logger.warning("Failed to build graph for %s", eid, exc_info=True)

    # Run LLM enrichment after deterministic rebuild
    enriched = 0
    if graph_enricher:
        for entry in entries:
            try:
                await graph_enricher.enrich_entry(entry)
                enriched += 1
            except Exception:
                logger.warning("Failed to enrich graph for %s", entry.id, exc_info=True)

    # Count resulting nodes and edges
    cursor = await db.execute("SELECT COUNT(*) FROM graph_nodes")
    row = await cursor.fetchone()
    node_count = row[0] if row else 0
    cursor = await db.execute("SELECT COUNT(*) FROM graph_edges")
    row = await cursor.fetchone()
    edge_count = row[0] if row else 0

    result = f"Graph rebuilt: {processed} entries processed, {node_count} nodes, {edge_count} edges"
    if enriched:
        result += f" ({enriched} enriched via LLM)"
    return result


async def _action_purge_inactive(
    db: Database,
    days_inactive: int,
    confirm: bool,
) -> str:
    """Hard-delete entries inactive for N+ days."""
    if not confirm:
        return "Error: purge_inactive requires confirm=True. This permanently deletes data."

    cutoff = (datetime.now(UTC) - timedelta(days=days_inactive)).isoformat()

    cursor = await db.execute(
        "SELECT id FROM knowledge_entries WHERE is_active = 0 AND updated_at < ?",
        (cutoff,),
    )
    rows = await cursor.fetchall()
    entry_ids = [row["id"] for row in rows]

    if not entry_ids:
        return f"No inactive entries older than {days_inactive} days to purge."

    for eid in entry_ids:
        await delete_entry_cascade(db, eid)

    return f"Purged {len(entry_ids)} inactive entries (older than {days_inactive} days)."


async def _action_vacuum(db: Database) -> str:
    """Optimize database — delegates to backend-specific implementation."""
    return await db.vacuum()


async def _action_entry_versions(
    db: Database,
    entry_id: str | None,
) -> str:
    """Show version history for an entry."""
    if not entry_id:
        return "Error: entry_id is required for entry_versions action."

    entry = await get_entry(db, entry_id)
    if entry is None:
        return f"Error: Entry {entry_id} not found."

    cursor = await db.execute(
        "SELECT * FROM entry_versions WHERE entry_id = ? ORDER BY version_number",
        (entry_id,),
    )
    rows = await cursor.fetchall()

    status = "active" if entry.is_active else "inactive"
    lines = [
        f"Version history for {entry_id}: {entry.short_title}",
        f"Status: {status} | Current version: {entry.version}"
        f" | Confidence: {entry.confidence_level:.0%}\n",
    ]

    if not rows:
        lines.append("No version records found.")
    else:
        for row in rows:
            date_str = row["created_at"][:19] if row["created_at"] else "unknown"
            reason = row["change_reason"] or "(no reason)"
            confidence = row["confidence_level"]
            lines.append(f"  v{row['version_number']} ({date_str}) — {reason} [{confidence:.0%}]")

    return "\n".join(lines)


async def _action_list_feedback(
    db: Database,
    feedback_type: str | None,
    since: str | None,
) -> str:
    """List recent agent feedback with optional filters."""
    sql = "SELECT * FROM agent_feedback WHERE 1=1"
    params: list[str] = []

    if feedback_type:
        sql += " AND feedback_type = ?"
        params.append(feedback_type)
    if since:
        sql += " AND created_at >= ?"
        params.append(since)

    sql += " ORDER BY created_at DESC LIMIT 50"

    cursor = await db.execute(sql, tuple(params))
    rows = await cursor.fetchall()

    if not rows:
        return "No feedback entries found."

    lines = [f"Agent Feedback ({len(rows)} entries)\n"]
    for row in rows:
        date_str = row["created_at"][:19] if row["created_at"] else "unknown"
        line = f"[{row['feedback_type']}] {date_str}"
        if row["tool_name"]:
            line += f" via {row['tool_name']}"
        if row["query_or_params"]:
            line += f" | query: {row['query_or_params']}"
        if row["contributor"]:
            badge = f"@{row['contributor']}"
            if row["team"]:
                badge += f"/{row['team']}"
            line += f" {badge}"
        if row["detail"]:
            line += f"\n  {row['detail']}"
        lines.append(line)

    return "\n".join(lines)


_FEEDBACK_SUMMARY_SYSTEM = """\
You are analyzing agent feedback about a knowledge base. \
Given a list of feedback entries, identify the top themes and patterns. \
Group related feedback together and rank themes by frequency/impact. \
Be concise — bullet points, no preamble.\
"""


async def _action_summarize_feedback(
    db: Database,
    query_llm: LLMProvider | None,
    since: str | None,
) -> str:
    """Cluster feedback into themes using LLM, fallback to raw list."""
    sql = "SELECT * FROM agent_feedback WHERE 1=1"
    params: list[str] = []

    if since:
        sql += " AND created_at >= ?"
        params.append(since)

    sql += " ORDER BY created_at DESC LIMIT 100"

    cursor = await db.execute(sql, tuple(params))
    rows = await cursor.fetchall()

    if not rows:
        return "No feedback entries to summarize."

    # Build text representation of feedback
    feedback_lines = []
    for row in rows:
        line = f"[{row['feedback_type']}]"
        if row["tool_name"]:
            line += f" tool={row['tool_name']}"
        if row["query_or_params"]:
            line += f" query={row['query_or_params']}"
        if row["detail"]:
            line += f" — {row['detail']}"
        if row["contributor"]:
            by = f"@{row['contributor']}"
            if row["team"]:
                by += f"/{row['team']}"
            line += f" by={by}"
        feedback_lines.append(line)
    feedback_text = "\n".join(feedback_lines)

    # Try LLM synthesis
    if query_llm is not None:
        try:
            summary = await query_llm.generate(
                f"Feedback entries ({len(rows)} total):\n\n{feedback_text}",
                system=_FEEDBACK_SUMMARY_SYSTEM,
            )
            if summary:
                return f"Feedback Summary ({len(rows)} entries)\n\n{summary}"
        except Exception:
            logger.warning("LLM summarization failed, falling back to raw list", exc_info=True)

    # Fallback: raw list
    return await _action_list_feedback(db, None, since)


async def _action_search_stats(
    db: Database,
    since: str | None,
) -> str:
    """Aggregate search telemetry: totals, zero-result rate, top missed queries."""
    where = ""
    params: list[str] = []
    if since:
        where = " WHERE created_at >= ?"
        params.append(since)

    # Total queries
    total_sql = "SELECT COUNT(*) FROM search_events" + where  # noqa: S608
    cursor = await db.execute(total_sql, tuple(params))
    row = await cursor.fetchone()
    total = row[0] if row else 0

    if total == 0:
        return "No search events recorded."

    # Zero-result count
    zero_where = " WHERE result_count = 0"
    if since:
        zero_where += " AND created_at >= ?"
    zero_sql = "SELECT COUNT(*) FROM search_events" + zero_where  # noqa: S608
    cursor = await db.execute(zero_sql, tuple(params))
    row = await cursor.fetchone()
    zero_count = row[0] if row else 0

    # Average top score (non-null only)
    avg_where = " WHERE top_score IS NOT NULL"
    if since:
        avg_where += " AND created_at >= ?"
    avg_sql = "SELECT AVG(top_score) FROM search_events" + avg_where  # noqa: S608
    cursor = await db.execute(avg_sql, tuple(params))
    row = await cursor.fetchone()
    avg_score = row[0] if row and row[0] is not None else 0.0

    # Top missed queries (zero results, grouped)
    missed_where = " WHERE result_count = 0"
    if since:
        missed_where += " AND created_at >= ?"
    missed_sql = (
        "SELECT query_text, COUNT(*) as cnt FROM search_events"  # noqa: S608
        + missed_where
        + " GROUP BY query_text ORDER BY cnt DESC LIMIT 10"
    )
    cursor = await db.execute(missed_sql, tuple(params))
    missed_rows = await cursor.fetchall()

    lines = [
        "Search Telemetry\n",
        f"Total queries: {total}",
        f"Zero-result queries: {zero_count} ({zero_count * 100 // total}%)",
        f"Average top score: {avg_score:.4f}",
    ]

    if missed_rows:
        lines.append("\nTop missed queries (zero results):")
        for mrow in missed_rows:
            lines.append(f"  [{mrow[1]}x] {mrow[0]}")

    return "\n".join(lines)


async def _action_list_contributors(db: Database) -> str:
    """Show contributor/team stats for active entries."""
    cursor = await db.execute(
        "SELECT contributor, team, COUNT(*) as entry_count"
        " FROM knowledge_entries"
        " WHERE contributor IS NOT NULL AND is_active = 1"
        " GROUP BY contributor, team"
        " ORDER BY entry_count DESC"
    )
    rows = await cursor.fetchall()

    if not rows:
        return "No attributed entries found."

    lines = ["Contributors\n"]
    for row in rows:
        contributor = row["contributor"]
        team = row["team"]
        count = row["entry_count"]
        badge = f"@{contributor}"
        if team:
            badge += f"/{team}"
        lines.append(f"  {badge} — {count} entries")

    return "\n".join(lines)


async def _action_list_audit(
    db: Database,
    entry_id: str | None,
    since: str | None,
) -> str:
    """List recent audit events with optional filters."""
    sql = "SELECT * FROM audit_events WHERE 1=1"
    params: list[str] = []

    if entry_id:
        sql += " AND entry_id = ?"
        params.append(entry_id)
    if since:
        sql += " AND created_at >= ?"
        params.append(since)

    sql += " ORDER BY created_at DESC LIMIT 50"

    cursor = await db.execute(sql, tuple(params))
    rows = await cursor.fetchall()

    if not rows:
        return "No audit events found."

    lines = [f"Audit Events ({len(rows)} entries)\n"]
    for row in rows:
        date_str = row["created_at"][:19] if row["created_at"] else "unknown"
        line = f"[{row['event_type']}] {date_str}"
        if row["entry_id"]:
            line += f" {row['entry_id']}"
        if row["contributor"]:
            line += f" by @{row['contributor']}"
        if row["detail"]:
            line += f" — {row['detail']}"
        lines.append(line)

    return "\n".join(lines)
