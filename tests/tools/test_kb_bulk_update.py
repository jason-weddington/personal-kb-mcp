"""Tests for the bulk_update feature (store + tool).

The store-level tests drive ``KnowledgeStore.bulk_update`` directly. The
tool-level tests either exercise the pure ``_format_result`` formatter or
drive the *registered* ``kb_bulk_update`` tool over an HTTP backend (the
only backend that remains — the in-process LocalBackend was deleted). HTTP
tests inject ``ctx.lifespan_context = {"backend": <HttpBackend>}`` over an
``httpx.MockTransport`` and assert on the tool's returned string.
"""

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
import pytest_asyncio

from personal_kb.backend.http import HttpBackend
from personal_kb.models.entry import EntryType
from personal_kb.store.knowledge_store import KnowledgeStore


@pytest_asyncio.fixture
async def populated_store(store: KnowledgeStore):
    """Store with a few entries for bulk update testing."""
    await store.create_entry(
        short_title="Entry A",
        long_title="Entry A long",
        knowledge_details="Details A",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref=None,
        contributor="alice",
        team="alpha",
        tags=["python", "api"],
    )
    await store.create_entry(
        short_title="Entry B",
        long_title="Entry B long",
        knowledge_details="Details B",
        entry_type=EntryType.DECISION,
        project_ref=None,
        contributor="alice",
        team="alpha",
        tags=["python"],
    )
    await store.create_entry(
        short_title="Entry C",
        long_title="Entry C long",
        knowledge_details="Details C",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="other-project",
        contributor="bob",
        team="beta",
        tags=["rust"],
    )
    return store


@pytest.mark.asyncio
async def test_bulk_update_project_ref(populated_store: KnowledgeStore):
    """Filter by contributor, set project_ref."""
    results = await populated_store.bulk_update(
        filters={"contributor": "alice"},
        updates={"project_ref": "new-project"},
    )
    assert len(results) == 2
    for before, after in results:
        assert before.project_ref is None
        assert after.project_ref == "new-project"
        assert after.version == before.version + 1


@pytest.mark.asyncio
async def test_bulk_update_dry_run(populated_store: KnowledgeStore):
    """Dry run should return previews but not persist changes."""
    results = await populated_store.bulk_update(
        filters={"contributor": "alice"},
        updates={"project_ref": "preview-project"},
        dry_run=True,
    )
    assert len(results) == 2
    # Verify nothing was persisted
    entry = await populated_store.get_entry(results[0][0].id)
    assert entry is not None
    assert entry.project_ref is None
    assert entry.version == 1


@pytest.mark.asyncio
async def test_bulk_update_tags_add(populated_store: KnowledgeStore):
    """Add tags without removing existing ones."""
    results = await populated_store.bulk_update(
        filters={"contributor": "alice"},
        updates={"tags_add": ["new-tag"]},
    )
    assert len(results) == 2
    for _before, after in results:
        assert "new-tag" in after.tags
        assert "python" in after.tags  # original preserved


@pytest.mark.asyncio
async def test_bulk_update_tags_remove(populated_store: KnowledgeStore):
    """Remove specific tags."""
    results = await populated_store.bulk_update(
        filters={"contributor": "alice"},
        updates={"tags_remove": ["python"]},
    )
    assert len(results) == 2
    for _before, after in results:
        assert "python" not in after.tags


@pytest.mark.asyncio
async def test_bulk_update_tags_add_and_remove(populated_store: KnowledgeStore):
    """Add and remove tags in the same operation."""
    results = await populated_store.bulk_update(
        filters={"entry_ids": ["kb-00001"]},
        updates={"tags_add": ["new"], "tags_remove": ["api"]},
    )
    assert len(results) == 1
    _before, after = results[0]
    assert "new" in after.tags
    assert "api" not in after.tags
    assert "python" in after.tags


@pytest.mark.asyncio
async def test_bulk_update_filter_null_project_ref(populated_store: KnowledgeStore):
    """Filter by project_ref=None to find unscoped entries."""
    results = await populated_store.bulk_update(
        filters={"project_ref": None},
        updates={"project_ref": "scoped"},
    )
    assert len(results) == 2  # entries A and B have no project_ref
    for _before, after in results:
        assert after.project_ref == "scoped"


@pytest.mark.asyncio
async def test_bulk_update_filter_by_entry_ids(populated_store: KnowledgeStore):
    """Filter by specific entry IDs."""
    results = await populated_store.bulk_update(
        filters={"entry_ids": ["kb-00003"]},
        updates={"project_ref": "targeted"},
    )
    assert len(results) == 1
    assert results[0][1].project_ref == "targeted"


@pytest.mark.asyncio
async def test_bulk_update_filter_by_entry_type(populated_store: KnowledgeStore):
    """Filter by entry_type."""
    results = await populated_store.bulk_update(
        filters={"entry_type": "decision"},
        updates={"confidence_level": 0.5},
    )
    assert len(results) == 1
    assert results[0][1].confidence_level == 0.5


@pytest.mark.asyncio
async def test_bulk_update_no_matching_entries(populated_store: KnowledgeStore):
    """No matches returns empty list."""
    results = await populated_store.bulk_update(
        filters={"contributor": "nobody"},
        updates={"project_ref": "x"},
    )
    assert results == []


@pytest.mark.asyncio
async def test_bulk_update_no_effective_changes(populated_store: KnowledgeStore):
    """Entries where updates produce no change are skipped."""
    results = await populated_store.bulk_update(
        filters={"entry_ids": ["kb-00003"]},
        updates={"project_ref": "other-project"},  # already this value
    )
    assert results == []


@pytest.mark.asyncio
async def test_bulk_update_version_bump(populated_store: KnowledgeStore, db):
    """Each updated entry gets a version bump and version record."""
    await populated_store.bulk_update(
        filters={"entry_ids": ["kb-00001"]},
        updates={"project_ref": "versioned"},
    )
    entry = await populated_store.get_entry("kb-00001")
    assert entry is not None
    assert entry.version == 2

    cursor = await db.execute(
        "SELECT * FROM entry_versions WHERE entry_id = ? ORDER BY version_number",
        ("kb-00001",),
    )
    versions = await cursor.fetchall()
    assert len(versions) == 2  # initial + bulk update


@pytest.mark.asyncio
async def test_bulk_update_audit_event(populated_store: KnowledgeStore, db):
    """Bulk update records audit events."""
    await populated_store.bulk_update(
        filters={"entry_ids": ["kb-00001"]},
        updates={"project_ref": "audited"},
        contributor="tester",
    )
    cursor = await db.execute(
        "SELECT * FROM audit_events WHERE entry_id = ? AND event_type = 'entry_updated'",
        ("kb-00001",),
    )
    events = await cursor.fetchall()
    # At least one audit event from the bulk update
    assert any("Bulk" in (e["detail"] or "") for e in events)


@pytest.mark.asyncio
async def test_bulk_update_skips_inactive(populated_store: KnowledgeStore):
    """Inactive entries are not included in bulk updates."""
    await populated_store.deactivate_entry("kb-00001")
    results = await populated_store.bulk_update(
        filters={"contributor": "alice"},
        updates={"project_ref": "x"},
    )
    assert len(results) == 1  # only kb-00002
    assert results[0][0].id == "kb-00002"


# --- Tool-level tests ---
#
# The registered tool only has an HTTP path now (LocalBackend is gone), so the
# tool-driven tests below inject an HttpBackend over httpx.MockTransport. The
# pure ``_format_result`` formatter is tested directly without any backend.


def _make_http_backend(handler) -> HttpBackend:
    """Build an HttpBackend backed by a MockTransport sync handler."""
    transport = httpx.MockTransport(handler)
    backend = HttpBackend(base_url="http://kb.test", api_key="testkey")
    backend._client = httpx.AsyncClient(
        base_url="http://kb.test",
        headers={"Authorization": "Bearer testkey"},
        transport=transport,
    )
    return backend


def _make_ctx(handler) -> MagicMock:
    """Return a MagicMock Context whose lifespan injects an HttpBackend."""
    ctx = MagicMock()
    ctx.lifespan_context = {"backend": _make_http_backend(handler)}
    return ctx


def _register_tool() -> Any:
    """Register kb_bulk_update on a mock MCP and return the captured callable."""
    from personal_kb.tools.kb_bulk_update import register_kb_bulk_update

    tools: dict[str, Any] = {}

    def capture(**_kw):
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn

        return decorator

    mcp = MagicMock()
    mcp.tool = capture
    register_kb_bulk_update(mcp)
    return next(iter(tools.values()))


_ENTRY_BEFORE: dict[str, Any] = {
    "id": "kb-00001",
    "short_title": "Test Entry",
    "long_title": "Test long title",
    "knowledge_details": "Some details",
    "entry_type": "factual_reference",
    "confidence_level": 0.9,
    "is_active": True,
    "version": 1,
    "has_embedding": False,
    "tags": ["python"],
    "project_ref": None,
    "source_context": None,
    "created_at": "2026-01-01T00:00:00+00:00",
    "updated_at": "2026-01-01T00:00:00+00:00",
    "last_accessed": None,
    "expires_at": None,
    "superseded_by": None,
    "sensitivity": None,
    "contributor": None,
    "team": None,
}
_ENTRY_AFTER: dict[str, Any] = {**_ENTRY_BEFORE, "project_ref": "new-project", "version": 2}


@pytest.mark.asyncio
async def test_tool_requires_filters():
    """Tool rejects calls with empty filters before any backend call."""
    called: list[bool] = []

    def handler(req: httpx.Request) -> httpx.Response:
        called.append(True)
        return httpx.Response(200, json={"results": []})

    kb_bulk_update = _register_tool()
    ctx = _make_ctx(handler)
    result = await kb_bulk_update(filters={}, updates={"project_ref": "x"}, ctx=ctx)
    assert "Error" in result
    assert not called  # validation happens before the backend is hit


@pytest.mark.asyncio
async def test_tool_requires_updates():
    """Tool rejects calls with empty updates before any backend call."""
    called: list[bool] = []

    def handler(req: httpx.Request) -> httpx.Response:
        called.append(True)
        return httpx.Response(200, json={"results": []})

    kb_bulk_update = _register_tool()
    ctx = _make_ctx(handler)
    result = await kb_bulk_update(filters={"contributor": "alice"}, updates={}, ctx=ctx)
    assert "Error" in result
    assert not called


@pytest.mark.asyncio
async def test_format_result_dry_run():
    """Dry run output includes DRY RUN prefix."""
    from datetime import UTC, datetime

    from personal_kb.models.entry import EntryType, KnowledgeEntry
    from personal_kb.tools.kb_bulk_update import _format_result

    now = datetime.now(UTC)
    before = KnowledgeEntry(
        id="kb-00001",
        short_title="T",
        long_title="T",
        knowledge_details="D",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref=None,
        version=1,
        created_at=now,
        updated_at=now,
    )
    after = before.model_copy(update={"project_ref": "new", "version": 2})
    result = _format_result([(before, after)], dry_run=True)
    assert "DRY RUN" in result
    assert "kb-00001" in result
    assert "project_ref" in result


@pytest.mark.asyncio
async def test_format_result_committed():
    """Committed output does not include DRY RUN prefix."""
    from datetime import UTC, datetime

    from personal_kb.models.entry import EntryType, KnowledgeEntry
    from personal_kb.tools.kb_bulk_update import _format_result

    now = datetime.now(UTC)
    before = KnowledgeEntry(
        id="kb-00001",
        short_title="T",
        long_title="T",
        knowledge_details="D",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref=None,
        version=1,
        created_at=now,
        updated_at=now,
    )
    after = before.model_copy(update={"project_ref": "new", "version": 2})
    result = _format_result([(before, after)], dry_run=False)
    assert "DRY RUN" not in result
    assert "1 entries updated" in result


# --- Tool-driven HTTP-contract tests ---


@pytest.mark.asyncio
async def test_tool_http_committed_renders_diff():
    """Committed apply formats the before/after diff from the backend pairs."""
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(
            200, json={"results": [{"before": _ENTRY_BEFORE, "after": _ENTRY_AFTER}]}
        )

    kb_bulk_update = _register_tool()
    ctx = _make_ctx(handler)
    result = await kb_bulk_update(
        filters={"contributor": "alice"},
        updates={"project_ref": "new-project"},
        dry_run=False,
        ctx=ctx,
    )
    assert "DRY RUN" not in result
    assert "1 entries updated" in result
    assert "kb-00001" in result
    assert "project_ref" in result
    assert "new-project" in result
    assert captured[0]["dry_run"] is False


@pytest.mark.asyncio
async def test_tool_http_dry_run_preview():
    """Dry run passes dry_run=True and renders the DRY RUN preview output."""
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(
            200, json={"results": [{"before": _ENTRY_BEFORE, "after": _ENTRY_AFTER}]}
        )

    kb_bulk_update = _register_tool()
    ctx = _make_ctx(handler)
    result = await kb_bulk_update(
        filters={"contributor": "alice"},
        updates={"project_ref": "new-project"},
        dry_run=True,
        ctx=ctx,
    )
    assert "DRY RUN" in result
    assert "would be" in result
    assert captured[0]["dry_run"] is True


@pytest.mark.asyncio
async def test_tool_http_no_matching_entries():
    """An empty results list maps to the 'No entries matched' message."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    kb_bulk_update = _register_tool()
    ctx = _make_ctx(handler)
    result = await kb_bulk_update(
        filters={"contributor": "nobody"},
        updates={"project_ref": "x"},
        dry_run=False,
        ctx=ctx,
    )
    assert "No entries matched" in result


@pytest.mark.asyncio
async def test_tool_http_admin_required_error():
    """A 403 from the service maps to an admin-required error string."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": "Admin only"})

    kb_bulk_update = _register_tool()
    ctx = _make_ctx(handler)
    result = await kb_bulk_update(
        filters={"project_ref": "x"},
        updates={"project_ref": "y"},
        dry_run=False,
        ctx=ctx,
    )
    assert "Error" in result
    assert "admin" in result.lower()


@pytest.mark.asyncio
async def test_bulk_update_team(populated_store: KnowledgeStore):
    """Filter by team, reassign team field to a canonical name."""
    results = await populated_store.bulk_update(
        filters={"team": "alpha"},
        updates={"team": "platform-eng"},
        dry_run=False,
    )
    assert len(results) == 2
    for before, after in results:
        assert after.team == "platform-eng"
        assert after.version == before.version + 1

    # Verify the DB was actually updated
    entry = await populated_store.get_entry(results[0][0].id)
    assert entry is not None
    assert entry.team == "platform-eng"


@pytest.mark.asyncio
async def test_bulk_update_team_dry_run(populated_store: KnowledgeStore):
    """Dry run team change returns previews but does not persist."""
    results = await populated_store.bulk_update(
        filters={"team": "alpha"},
        updates={"team": "platform-eng"},
        dry_run=True,
    )
    assert len(results) == 2
    for _before, after in results:
        assert after.team == "platform-eng"

    # DB must still show the original team
    entry = await populated_store.get_entry(results[0][0].id)
    assert entry is not None
    assert entry.team == "alpha"


@pytest.mark.asyncio
async def test_bulk_update_team_no_change(populated_store: KnowledgeStore):
    """Updating team to its current value is a no-op."""
    results = await populated_store.bulk_update(
        filters={"team": "alpha"},
        updates={"team": "alpha"},
    )
    assert results == []
