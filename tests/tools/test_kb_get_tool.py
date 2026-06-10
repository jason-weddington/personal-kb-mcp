"""Tests for the kb_get MCP tool."""

import pytest

from personal_kb.db.queries import (
    deactivate_entry_db,
    get_entry,
    update_entry,
)
from personal_kb.graph.builder import GraphBuilder
from personal_kb.models.entry import EntryType
from personal_kb.tools.formatters import format_entry_full, format_result_list
from personal_kb.tools.kb_get import _MAX_IDS, _render_pointer_rot


async def _kb_get_logic(db, ids: list[str]) -> str:
    """Replicate kb_get logic for testing without MCP context.

    Uses ``backend_from_lifespan`` so this helper stays in lockstep with
    the real kb_get tool body — both go through the same
    LocalBackend.get_entries() path.
    """
    if len(ids) > _MAX_IDS:
        return f"Error: Maximum {_MAX_IDS} IDs per request (got {len(ids)})."

    from personal_kb.tools._lifespan import backend_from_lifespan

    backend = backend_from_lifespan({"db": db})
    entries_data = await backend.get_entries(ids)

    formatted: list[str] = []
    for eid, entry, rot_pairs in entries_data:
        if entry is None:
            formatted.append(f"[{eid}] not found")
        else:
            rendered = format_entry_full(entry)
            note = _render_pointer_rot(rot_pairs)
            if note is not None:
                rendered = f"{rendered}\n{note}"
            formatted.append(rendered)

    return format_result_list(formatted)


@pytest.mark.asyncio
async def test_get_single_entry(db, store):
    """Retrieve a single entry by ID."""
    entry = await store.create_entry(
        short_title="Test",
        long_title="Test entry",
        knowledge_details="Full details here",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["python"],
        project_ref="my-proj",
    )

    result = await _kb_get_logic(db, [entry.id])
    assert entry.id in result
    assert "Full details here" in result
    assert "#python" in result
    assert "my-proj" in result


@pytest.mark.asyncio
async def test_get_multiple_entries(db, store):
    """Retrieve multiple entries at once."""
    e1 = await store.create_entry(
        short_title="First",
        long_title="First entry",
        knowledge_details="First details",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    e2 = await store.create_entry(
        short_title="Second",
        long_title="Second entry",
        knowledge_details="Second details",
        entry_type=EntryType.DECISION,
    )

    result = await _kb_get_logic(db, [e1.id, e2.id])
    assert e1.id in result
    assert e2.id in result
    assert "First details" in result
    assert "Second details" in result
    assert "2 result(s)" in result


@pytest.mark.asyncio
async def test_get_missing_entry(db):
    """Missing IDs show 'not found'."""
    result = await _kb_get_logic(db, ["kb-99999"])
    assert "kb-99999" in result
    assert "not found" in result


@pytest.mark.asyncio
async def test_get_mixed_found_and_missing(db, store):
    """Mix of found and missing entries."""
    entry = await store.create_entry(
        short_title="Exists",
        long_title="Existing entry",
        knowledge_details="Real content",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    result = await _kb_get_logic(db, [entry.id, "kb-99999"])
    assert entry.id in result
    assert "Real content" in result
    assert "kb-99999" in result
    assert "not found" in result
    assert "2 result(s)" in result


@pytest.mark.asyncio
async def test_get_inactive_entry_skipped(db, store):
    """Inactive entries are treated as not found."""
    entry = await store.create_entry(
        short_title="Soon gone",
        long_title="Will be deactivated",
        knowledge_details="Should not appear after deactivation",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    await store.deactivate_entry(entry.id)

    result = await _kb_get_logic(db, [entry.id])
    assert "not found" in result
    assert "Should not appear" not in result


@pytest.mark.asyncio
async def test_get_cap_at_20(db):
    """Exceeding 20 IDs returns an error."""
    ids = [f"kb-{i:05d}" for i in range(1, 22)]
    result = await _kb_get_logic(db, ids)
    assert "Maximum 20" in result


@pytest.mark.asyncio
async def test_get_updates_last_accessed(db, store):
    """kb_get should update last_accessed — explicit retrieval resets decay clock."""
    entry = await store.create_entry(
        short_title="Decay test",
        long_title="Access-aware decay",
        knowledge_details="Explicit retrieval should reset the decay clock.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    # Initially NULL
    fetched = await get_entry(db, entry.id)
    assert fetched is not None
    assert fetched.last_accessed is None

    # kb_get should touch last_accessed
    await _kb_get_logic(db, [entry.id])

    fetched = await get_entry(db, entry.id)
    assert fetched is not None
    assert fetched.last_accessed is not None


@pytest.mark.asyncio
async def test_get_does_not_touch_missing_entries(db, store):
    """kb_get should not touch last_accessed for missing/inactive entries."""
    entry = await store.create_entry(
        short_title="Will deactivate",
        long_title="Inactive entry",
        knowledge_details="Should not get accessed.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    await store.deactivate_entry(entry.id)

    await _kb_get_logic(db, [entry.id, "kb-99999"])

    fetched = await get_entry(db, entry.id)
    assert fetched is not None
    assert fetched.last_accessed is None


# --- Pointer-rot tests for mental_map (§7.4) ---------------------------------


async def _make_superseded(db, target_id: str, replacement_id: str) -> None:
    """Mark ``target_id`` as superseded by ``replacement_id``.

    KnowledgeStore.create_entry has no superseded_by parameter, so we re-fetch
    the row, copy with superseded_by set, and persist via the DB-level
    update_entry. Caller is responsible for the replacement existing or not —
    we only mutate the target.
    """
    target = await get_entry(db, target_id)
    assert target is not None
    mutated = target.model_copy(update={"superseded_by": replacement_id})
    await update_entry(db, mutated)


async def _seed_graph(builder: GraphBuilder, *entries) -> None:
    """Populate graph_nodes for the given entries via the builder.

    graph_edges has a FOREIGN KEY on graph_nodes(node_id), so both source and
    target nodes must exist before ``_add_edge`` can run. KnowledgeStore.
    create_entry does not author graph nodes — that's normally the enricher's
    job — so tests have to seed the graph explicitly.
    """
    for entry in entries:
        await builder.build_for_entry(entry)


@pytest.mark.asyncio
async def test_mental_map_renders_superseded_pointer(db, store, graph_builder):
    """(a) Map with one superseded pointer target renders 'superseded by' line."""
    target = await store.create_entry(
        short_title="Old fact",
        long_title="Old factual entry",
        knowledge_details="The old way.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    replacement = await store.create_entry(
        short_title="New fact",
        long_title="New factual entry",
        knowledge_details="The new way.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    await _make_superseded(db, target.id, replacement.id)

    mmap = await store.create_entry(
        short_title="Map",
        long_title="Mental map",
        knowledge_details="Orientation map.",
        entry_type=EntryType.MENTAL_MAP,
    )
    await _seed_graph(graph_builder, target, replacement, mmap)
    await graph_builder._add_edge(mmap.id, target.id, "references")

    refetched = await get_entry(db, target.id)
    assert refetched is not None and refetched.superseded_by == replacement.id

    result = await _kb_get_logic(db, [mmap.id])
    assert "  Pointer-rot:" in result
    assert f"    [{target.id}] superseded by [{replacement.id}]" in result
    assert "deactivated" not in result


@pytest.mark.asyncio
async def test_mental_map_renders_deactivated_pointer(db, store, graph_builder):
    """(b) Map with one deactivated (not-superseded) target renders 'deactivated'."""
    target = await store.create_entry(
        short_title="Dead",
        long_title="Deactivated target",
        knowledge_details="Gone.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    await deactivate_entry_db(db, target.id)

    mmap = await store.create_entry(
        short_title="Map",
        long_title="Mental map",
        knowledge_details="Orientation map.",
        entry_type=EntryType.MENTAL_MAP,
    )
    await _seed_graph(graph_builder, target, mmap)
    await graph_builder._add_edge(mmap.id, target.id, "references")

    refetched = await get_entry(db, target.id)
    assert refetched is not None
    assert refetched.is_active is False
    assert refetched.superseded_by is None

    result = await _kb_get_logic(db, [mmap.id])
    assert "  Pointer-rot:" in result
    assert f"    [{target.id}] deactivated" in result
    assert "superseded by" not in result


@pytest.mark.asyncio
async def test_mental_map_all_healthy_targets_silent(db, store, graph_builder):
    """(c) Map with all-healthy targets renders NO rot note."""
    t1 = await store.create_entry(
        short_title="Healthy 1",
        long_title="First healthy",
        knowledge_details="Active and current.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    t2 = await store.create_entry(
        short_title="Healthy 2",
        long_title="Second healthy",
        knowledge_details="Active and current.",
        entry_type=EntryType.DECISION,
    )

    mmap = await store.create_entry(
        short_title="Map",
        long_title="Mental map",
        knowledge_details="Orientation map.",
        entry_type=EntryType.MENTAL_MAP,
    )
    await _seed_graph(graph_builder, t1, t2, mmap)
    await graph_builder._add_edge(mmap.id, t1.id, "references")
    await graph_builder._add_edge(mmap.id, t2.id, "contains")

    result = await _kb_get_logic(db, [mmap.id])
    assert "Pointer-rot" not in result
    assert "superseded by" not in result
    assert "deactivated" not in result


@pytest.mark.asyncio
async def test_non_map_pointing_at_superseded_target_silent(db, store, graph_builder):
    """(d) Non-mental_map entry pointing at a superseded target renders NO rot note.

    Also asserts byte-identical output to the no-edge baseline: the map-only
    check must not change rendering for the 4 pre-existing entry types.
    """
    target = await store.create_entry(
        short_title="Old",
        long_title="Old fact",
        knowledge_details="The old way.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    replacement = await store.create_entry(
        short_title="New",
        long_title="New fact",
        knowledge_details="The new way.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    await _make_superseded(db, target.id, replacement.id)

    non_map = await store.create_entry(
        short_title="A decision",
        long_title="A decision pointing somewhere",
        knowledge_details="Points at the old fact.",
        entry_type=EntryType.DECISION,
    )
    await _seed_graph(graph_builder, target, replacement, non_map)
    await graph_builder._add_edge(non_map.id, target.id, "references")

    result = await _kb_get_logic(db, [non_map.id])
    assert "Pointer-rot" not in result
    assert "superseded by" not in result
    assert "deactivated" not in result

    # Byte-identical guarantee: the helper returns None for non-map entries,
    # so the rendered output equals format_entry_full + the format_result_list
    # wrapper — no extra bytes appended.
    fetched = await get_entry(db, non_map.id)
    assert fetched is not None
    expected = format_result_list([format_entry_full(fetched)])
    # Drop the last_accessed-touched timestamp from comparison by re-fetching
    # the rendered baseline through the same code path on a fresh row state.
    assert result == expected


@pytest.mark.asyncio
async def test_mental_map_superseded_and_deactivated_renders_superseded_form(
    db, store, graph_builder
):
    """(e) Target both superseded AND deactivated → SUPERSEDED form (precedence)."""
    target = await store.create_entry(
        short_title="Both",
        long_title="Both rotted",
        knowledge_details="Superseded and deactivated.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    replacement = await store.create_entry(
        short_title="Replacement",
        long_title="The replacement",
        knowledge_details="Live.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    await _make_superseded(db, target.id, replacement.id)
    await deactivate_entry_db(db, target.id)

    refetched = await get_entry(db, target.id)
    assert refetched is not None
    assert refetched.superseded_by == replacement.id
    assert refetched.is_active is False

    mmap = await store.create_entry(
        short_title="Map",
        long_title="Mental map",
        knowledge_details="Orientation map.",
        entry_type=EntryType.MENTAL_MAP,
    )
    await _seed_graph(graph_builder, target, replacement, mmap)
    await graph_builder._add_edge(mmap.id, target.id, "references")

    result = await _kb_get_logic(db, [mmap.id])
    assert "  Pointer-rot:" in result
    # Precedence: superseded form wins; the deactivated form must NOT appear.
    assert f"    [{target.id}] superseded by [{replacement.id}]" in result
    assert f"    [{target.id}] deactivated" not in result
