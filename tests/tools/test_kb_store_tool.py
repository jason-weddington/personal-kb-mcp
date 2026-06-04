"""Tests for the kb_store MCP tool logic."""

from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from personal_kb.db.connection import create_connection
from personal_kb.graph.builder import GraphBuilder
from personal_kb.graph.enricher import GraphEnricher
from personal_kb.models.entry import EntryType
from personal_kb.store.knowledge_store import KnowledgeStore
from personal_kb.tools.kb_store import (
    ORPHAN_MAP_ERROR,
    _mental_map_has_pointer,
    _validate_sensitivity,
    format_store_result,
    register_kb_store,
)
from tests.conftest import FakeEmbedder, FakeLLM


@pytest.mark.asyncio
async def test_format_store_result_create(store):
    entry = await store.create_entry(
        short_title="Test",
        long_title="Test entry",
        knowledge_details="Details",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="my-project",
        tags=["tag1", "tag2"],
    )
    result = format_store_result(entry, is_update=False)
    assert "Created kb-00001" in result
    assert "my-project" in result
    assert "#tag1 #tag2" in result


@pytest.mark.asyncio
async def test_format_store_result_update(store):
    entry = await store.create_entry(
        short_title="Test",
        long_title="Test entry",
        knowledge_details="Details",
        entry_type=EntryType.DECISION,
    )
    updated = await store.update_entry(
        entry_id=entry.id,
        knowledge_details="New details",
        change_reason="Updated",
    )
    result = format_store_result(updated, is_update=True)
    assert "Updated kb-00001 (v2)" in result


@pytest.mark.asyncio
async def test_deactivate_entry(store):
    """Deactivate removes entry from active set."""
    entry = await store.create_entry(
        short_title="Wrong fact",
        long_title="An incorrect fact",
        knowledge_details="This is wrong",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    deactivated = await store.deactivate_entry(entry.id)
    assert deactivated.is_active is False

    reloaded = await store.get_entry(entry.id)
    assert reloaded is not None
    assert reloaded.is_active is False


@pytest.mark.asyncio
async def test_deactivate_nonexistent_entry(store):
    """Deactivating a nonexistent entry raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        await store.deactivate_entry("kb-99999")


@pytest.mark.asyncio
async def test_deactivate_already_inactive(store):
    """Deactivating an already-inactive entry raises ValueError."""
    entry = await store.create_entry(
        short_title="Test",
        long_title="Test entry",
        knowledge_details="Details",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    await store.deactivate_entry(entry.id)
    with pytest.raises(ValueError, match="already inactive"):
        await store.deactivate_entry(entry.id)


# --- Sensitivity validation ---


def test_validate_sensitivity_valid_values():
    """Valid sensitivity values should pass."""
    assert _validate_sensitivity(None) is None
    assert _validate_sensitivity("internal") is None
    assert _validate_sensitivity("restricted") is None
    assert _validate_sensitivity("public") is None


def test_validate_sensitivity_invalid_value():
    """Invalid sensitivity should return an error string."""
    result = _validate_sensitivity("banana")
    assert result is not None
    assert "Invalid sensitivity" in result
    assert '"banana"' in result
    assert "internal" in result
    assert "public" in result
    assert "restricted" in result


# --- mental_map pointer-detection helper ---


def test_pointer_helper_body_reference():
    assert _mental_map_has_pointer("see kb-00050 for details", None) is True


def test_pointer_helper_supersedes_hint():
    assert _mental_map_has_pointer("no refs", {"supersedes": "kb-00042"}) is True
    # invalid supersedes target is not a pointer
    assert _mental_map_has_pointer("no refs", {"supersedes": "not-an-id"}) is False


def test_pointer_helper_superseded_by():
    assert _mental_map_has_pointer("no refs", None, superseded_by="kb-00099") is True
    assert _mental_map_has_pointer("no refs", None, superseded_by="") is False


def test_pointer_helper_related_entities_dict():
    assert _mental_map_has_pointer("no refs", {"related_entities": [{"id": "kb-00042"}]}) is True
    assert (
        _mental_map_has_pointer("no refs", {"related_entities": [{"target": "kb-00043"}]}) is True
    )


def test_pointer_helper_related_entities_bare_string():
    """A bare-string related_entity produces a real related_to edge → it is a pointer."""
    assert _mental_map_has_pointer("no refs", {"related_entities": ["kb-00044"]}) is True


def test_pointer_helper_non_pointer_hints_dont_count():
    hints = {
        "has_tag": ["python"],
        "in_project": ["kb"],
        "mentions_person": ["jason"],
        "uses_tool": ["sqlite"],
    }
    assert _mental_map_has_pointer("no refs at all", hints) is False


def test_pointer_helper_zero_pointers():
    assert _mental_map_has_pointer("just plain prose, no ids", None) is False
    assert _mental_map_has_pointer("just plain prose", {}) is False


# --- mental_map create path via the registered tool ---


@pytest_asyncio.fixture
async def tool_context():
    """Create a mock MCP context with all lifespan dependencies."""
    db = await create_connection(":memory:")
    store = KnowledgeStore(db)
    embedder = FakeEmbedder(db)
    graph_builder = GraphBuilder(db)
    enricher = GraphEnricher(db, FakeLLM())

    lifespan = {
        "db": db,
        "store": store,
        "embedder": embedder,
        "graph_builder": graph_builder,
        "graph_enricher": enricher,
        "contributor": None,
        "team": None,
    }

    ctx = MagicMock()
    ctx.lifespan_context = lifespan

    yield ctx, lifespan

    await db.close()


def _register_and_capture():
    """Register kb_store on a mock MCP and return the captured tool callable."""
    tools = {}

    def capture_tool(**_kwargs):
        def decorator(func):
            tools[func.__name__] = func
            return func

        return decorator

    mcp_mock = MagicMock()
    mcp_mock.tool = capture_tool
    register_kb_store(mcp_mock)
    return tools["kb_store"]


async def _entry_count(db) -> int:
    cursor = await db.execute("SELECT COUNT(*) FROM knowledge_entries")
    row = await cursor.fetchone()
    return row[0]


@pytest.mark.asyncio
async def test_mental_map_with_body_reference_succeeds(tool_context):
    """(a) mental_map with in-body kb-XXXXX ref stores and shows un-decayed confidence."""
    ctx, lifespan = tool_context
    kb_store = _register_and_capture()
    result = await kb_store(
        short_title="Auth map",
        long_title="Auth subsystem orientation",
        knowledge_details="Start at kb-00050 then follow the edges.",
        entry_type=EntryType.MENTAL_MAP,
        ctx=ctx,
    )
    assert "Created" in result
    # un-decayed: effective == base (~90%), and no staleness badge
    assert "(90%)" in result
    assert "[STALE]" not in result
    assert await _entry_count(lifespan["db"]) == 1


@pytest.mark.asyncio
async def test_mental_map_with_related_entity_hint_succeeds(tool_context):
    """(b) mental_map with a related_entities id hint and no in-body ref stores."""
    ctx, lifespan = tool_context
    kb_store = _register_and_capture()
    result = await kb_store(
        short_title="Map",
        long_title="A map",
        knowledge_details="No inline references here.",
        entry_type=EntryType.MENTAL_MAP,
        hints={"related_entities": [{"id": "kb-00042"}]},
        ctx=ctx,
    )
    assert "Created" in result
    assert await _entry_count(lifespan["db"]) == 1


@pytest.mark.asyncio
async def test_mental_map_zero_pointers_rejected(tool_context):
    """(c) zero-pointer mental_map returns the orphan error and creates nothing."""
    ctx, lifespan = tool_context
    kb_store = _register_and_capture()
    result = await kb_store(
        short_title="Orphan",
        long_title="Orphan map",
        knowledge_details="Just prose, no pointers at all.",
        entry_type=EntryType.MENTAL_MAP,
        ctx=ctx,
    )
    assert result == ORPHAN_MAP_ERROR
    assert await _entry_count(lifespan["db"]) == 0
    # A subsequent get finds nothing.
    assert await lifespan["store"].get_entry("kb-00001") is None


@pytest.mark.asyncio
async def test_non_mental_map_zero_pointers_still_succeeds(tool_context):
    """(d) regression guard: a factual_reference with zero pointers still stores."""
    ctx, lifespan = tool_context
    kb_store = _register_and_capture()
    result = await kb_store(
        short_title="Fact",
        long_title="A fact",
        knowledge_details="Plain fact with no pointers.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        ctx=ctx,
    )
    assert "Created" in result
    assert await _entry_count(lifespan["db"]) == 1


# --- advisory mental_map lint at the store call site ---


@pytest.mark.asyncio
async def test_mental_map_create_with_value_surfaces_advisory(tool_context):
    """A mental_map create with a config value in the body returns Created + advisory."""
    ctx, lifespan = tool_context
    kb_store = _register_and_capture()
    result = await kb_store(
        short_title="Net map",
        long_title="Network orientation",
        knowledge_details="orients kb-00050; the explorer runs on port 8767",
        entry_type=EntryType.MENTAL_MAP,
        ctx=ctx,
    )
    assert "Created kb-" in result
    assert "Map lint (advisory):" in result
    # Store still succeeds — entry was created.
    assert await _entry_count(lifespan["db"]) == 1


@pytest.mark.asyncio
async def test_mental_map_backend_warning_precedes_lint(tool_context):
    """When both a backend warning and a lint advisory are present, backend comes first."""
    ctx, _lifespan = tool_context
    kb_store = _register_and_capture()
    with patch(
        "personal_kb.config.get_backend_warning",
        return_value="Backend fallback: using degraded mode",
    ):
        result = await kb_store(
            short_title="Net map",
            long_title="Network orientation",
            knowledge_details="orients kb-00050; the explorer runs on port 8767",
            entry_type=EntryType.MENTAL_MAP,
            ctx=ctx,
        )
    assert "Backend fallback" in result
    assert "Map lint (advisory):" in result
    assert result.index("Backend fallback") < result.index("Map lint (advisory):")


@pytest.mark.asyncio
async def test_mental_map_update_with_body_surfaces_advisory(tool_context):
    """Update with a new body and NO entry_type param still lints (gate reads persisted type)."""
    ctx, _lifespan = tool_context
    kb_store = _register_and_capture()
    created = await kb_store(
        short_title="Net map",
        long_title="Network orientation",
        knowledge_details="orients kb-00050; clean orientation prose",
        entry_type=EntryType.MENTAL_MAP,
        ctx=ctx,
    )
    assert "Map lint (advisory):" not in created  # clean body, no advisory
    result = await kb_store(
        update_entry_id="kb-00001",
        knowledge_details="orients kb-00050; the explorer runs on port 8767",
        ctx=ctx,  # NOTE: no entry_type argument
    )
    assert "Updated kb-00001" in result
    assert "Map lint (advisory):" in result


@pytest.mark.asyncio
async def test_mental_map_metadata_only_update_skips_lint(tool_context):
    """A metadata-only update (no knowledge_details) returns Updated with NO advisory."""
    ctx, _lifespan = tool_context
    kb_store = _register_and_capture()
    await kb_store(
        short_title="Net map",
        long_title="Network orientation",
        knowledge_details="orients kb-00050; the explorer runs on port 8767",
        entry_type=EntryType.MENTAL_MAP,
        ctx=ctx,
    )
    result = await kb_store(
        update_entry_id="kb-00001",
        tags=["new-tag"],
        ctx=ctx,  # no knowledge_details — metadata-only
    )
    assert "Updated kb-00001" in result
    assert "Map lint (advisory):" not in result


@pytest.mark.asyncio
async def test_non_map_create_is_not_linted(tool_context):
    """A factual_reference with a numeral body is never linted (call-site gating)."""
    ctx, _lifespan = tool_context
    kb_store = _register_and_capture()
    result = await kb_store(
        short_title="Fact",
        long_title="A fact",
        knowledge_details="the explorer runs on port 8767",
        entry_type=EntryType.FACTUAL_REFERENCE,
        ctx=ctx,
    )
    assert "Created kb-" in result
    assert "Map lint (advisory):" not in result


@pytest.mark.asyncio
async def test_mental_map_contains_edge(tool_context):
    """(e) related_entities edge_type='contains' produces a graph edge + get_neighbors tuple."""
    from personal_kb.graph.queries import get_neighbors

    ctx, lifespan = tool_context
    db = lifespan["db"]
    kb_store = _register_and_capture()
    result = await kb_store(
        short_title="Container map",
        long_title="Container map",
        knowledge_details="No inline refs.",
        entry_type=EntryType.MENTAL_MAP,
        hints={"related_entities": [{"id": "kb-00050", "edge_type": "contains"}]},
        ctx=ctx,
    )
    assert "Created" in result
    map_id = "kb-00001"

    # (1) direct DB query finds exactly one contains edge
    cursor = await db.execute(
        "SELECT source, target, edge_type FROM graph_edges "
        "WHERE source = ? AND target = ? AND edge_type = ?",
        (map_id, "kb-00050", "contains"),
    )
    rows = await cursor.fetchall()
    assert len(rows) == 1

    # (2) get_neighbors returns the exact 3-tuple (neighbor_id, edge_type, direction)
    neighbors = await get_neighbors(db, map_id, edge_types=["contains"])
    assert ("kb-00050", "contains", "outgoing") in neighbors
