"""Regression tests for kb-017b7606: metadata-only updates must not delete
LLM-enriched graph edges.

``KnowledgeBase.update`` always calls ``_build_graph`` (unconditionally,
by design — tags/project_ref changes depend on it) but only calls
``_enrich_one`` when ``knowledge_details`` was part of the update. Before
the fix, ``GraphBuilder._clear_edges_for_source`` ran an unscoped
``DELETE FROM graph_edges WHERE source = ?`` ahead of every rebuild, so
any metadata-only update (tags, hints, title, ...) silently and
permanently destroyed every edge the enricher had ever created for that
entry, because re-enrichment never ran to replace them.

These tests exercise the real SQLite backend via :func:`kb_core.create_sqlite`
(no mocks) and stand in for enrichment output by inserting an edge with the
same ``properties`` shape :meth:`GraphEnricher._add_enrichment_edge` writes
(``{"source": "llm"}``) directly into ``graph_edges``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from kb_core import create_sqlite
from kb_core.models.entry import EntryType

pytestmark = pytest.mark.asyncio


async def _insert_llm_edge(kb: Any, source: str, target: str, edge_type: str) -> None:
    """Insert an edge shaped like enricher output (``properties.source == "llm"``)."""
    await kb.db.execute(
        """INSERT INTO graph_nodes (node_id, node_type, properties, created_at)
           VALUES (?, 'concept', '{}', datetime('now'))
           ON CONFLICT(node_id) DO NOTHING""",
        (target,),
    )
    await kb.db.execute(
        """INSERT INTO graph_edges (source, target, edge_type, properties, created_at)
           VALUES (?, ?, ?, '{"source": "llm"}', datetime('now'))
           ON CONFLICT (source, target, edge_type) DO NOTHING""",
        (source, target, edge_type),
    )
    await kb.db.commit()


async def _edges_for(kb: Any, source: str) -> list[dict[str, Any]]:
    cursor = await kb.db.execute(
        "SELECT source, target, edge_type, properties FROM graph_edges WHERE source = ?",
        (source,),
    )
    rows = await cursor.fetchall()
    return [
        {
            "source": r[0],
            "target": r[1],
            "edge_type": r[2],
            "properties": json.loads(r[3]),
        }
        for r in rows
    ]


@pytest.fixture
async def kb_seeded(tmp_path: Any) -> Any:
    """A real SQLite KB with one entry carrying deterministic AND LLM edges.

    The entry has:
    - a tag (-> has_tag edge)
    - a project_ref (-> in_project edge)
    - a kb-XXXXX reference in its body (-> references edge to another entry)
    - one LLM-derived edge inserted directly to stand in for enrichment
      output (concept:async-io, edge_type "discusses")
    """
    db_path = tmp_path / "kb.db"
    kb = await create_sqlite(db_path)

    # The referenced entry must exist as a node for the FK-free graph_edges
    # table to make sense, though the builder itself doesn't enforce that.
    referenced = await kb.store(
        short_title="referenced entry",
        long_title="Referenced Entry",
        knowledge_details="Some other entry.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        enrich=False,
    )

    entry = await kb.store(
        short_title="orig title",
        long_title="Original Long Title",
        knowledge_details=f"Some details mentioning {referenced.id}.",
        entry_type=EntryType.PATTERN_CONVENTION,
        project_ref="personal-kb",
        tags=["python", "sqlite"],
        enrich=False,
    )

    await _insert_llm_edge(kb, entry.id, "concept:async-io", "discusses")

    try:
        yield kb, entry, referenced
    finally:
        await kb.close()


async def test_tags_only_update_preserves_llm_edge_and_refreshes_has_tag(
    kb_seeded: Any,
) -> None:
    """A tags-only update must not wipe the LLM edge, and must refresh has_tag."""
    kb, entry, _referenced = kb_seeded

    await kb.update(entry.id, tags=["python", "async"], enrich=False)

    edges = await _edges_for(kb, entry.id)
    edge_types = {e["edge_type"] for e in edges}

    # LLM edge survived.
    llm_edges = [e for e in edges if e["properties"].get("source") == "llm"]
    assert len(llm_edges) == 1
    assert llm_edges[0]["target"] == "concept:async-io"

    # Deterministic has_tag edges reflect the NEW tag set.
    tag_targets = {e["target"] for e in edges if e["edge_type"] == "has_tag"}
    assert tag_targets == {"tag:python", "tag:async"}
    assert "has_tag" in edge_types


async def test_hints_only_update_preserves_llm_edge(kb_seeded: Any) -> None:
    """A hints-only update (mirrors scripts/set_operated_via_hints.py) preserves the LLM edge."""
    kb, entry, _referenced = kb_seeded

    await kb.update(entry.id, hints={"person": ["jason"]}, enrich=False)

    edges = await _edges_for(kb, entry.id)
    llm_edges = [e for e in edges if e["properties"].get("source") == "llm"]
    assert len(llm_edges) == 1
    assert llm_edges[0]["target"] == "concept:async-io"

    # The new hint-derived deterministic edge was built.
    person_edges = [e for e in edges if e["edge_type"] == "mentions_person"]
    assert any(e["target"] == "person:jason" for e in person_edges)


async def test_title_only_update_preserves_llm_edge(kb_seeded: Any) -> None:
    """A title-only update (no tags/hints/content change) preserves the LLM edge."""
    kb, entry, _referenced = kb_seeded

    await kb.update(entry.id, short_title="new title", enrich=False)

    edges = await _edges_for(kb, entry.id)
    llm_edges = [e for e in edges if e["properties"].get("source") == "llm"]
    assert len(llm_edges) == 1
    assert llm_edges[0]["target"] == "concept:async-io"

    # Deterministic edges (tags, project) are still intact — the rebuild ran.
    edge_types = {e["edge_type"] for e in edges}
    assert "has_tag" in edge_types
    assert "in_project" in edge_types


async def test_content_update_still_rebuilds_deterministic_edges(kb_seeded: Any) -> None:
    """A content update (knowledge_details provided) still rebuilds deterministic
    edges correctly — same end state as before the fix for the content-change path.
    """
    kb, entry, _referenced = kb_seeded

    other = await kb.store(
        short_title="second reference",
        long_title="Second Reference",
        knowledge_details="Another entry to reference.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        enrich=False,
    )

    await kb.update(
        entry.id,
        knowledge_details=f"Updated body now mentions {other.id} and nothing else.",
        enrich=False,
    )

    edges = await _edges_for(kb, entry.id)

    # references edge now points at the NEW mentioned entry.
    reference_targets = {e["target"] for e in edges if e["edge_type"] == "references"}
    assert reference_targets == {other.id}

    # Deterministic tag/project edges are still present (unchanged fields).
    edge_types = {e["edge_type"] for e in edges}
    assert "has_tag" in edge_types
    assert "in_project" in edge_types

    # The stale LLM edge from before the content change is still present —
    # acceptable per spec (enrich=False here, so no re-enrichment ran to
    # replace it; the UNIQUE(source, target, edge_type) constraint prevents
    # duplicates if enrichment does run later).
    llm_edges = [e for e in edges if e["properties"].get("source") == "llm"]
    assert len(llm_edges) == 1
