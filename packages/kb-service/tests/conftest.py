"""Hermetic test fixtures: no live Postgres, Ollama, or network.

The app's lifespan normally calls ``init_db()`` and ``create_postgres()``. Both
are monkeypatched here so the suite never touches a real database or embedder.
``app.state.kb`` is replaced with a ``FakeKnowledgeBase`` whose ``search`` is
parametrizable, and ``database.get_db`` is replaced with a minimal fake pool.
"""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from kb_core.cluster_ledger import (
    CLUSTER_MARGINAL_MATCH_CEILING,
    CLUSTER_MATCH_JACCARD,
    CLUSTER_NEAR_MISS_FLOOR,
    REOPEN_GROWTH_FACTOR,
    ClusterLedgerMatchResult,
    ClusterLedgerRow,
)
from kb_core.ingest.ingester import FileResult
from kb_core.map_delete import DeletedMapRecord
from kb_core.map_eligibility import (
    MapEligibilityOverride,
    MapEligibilityVerdict,
    classify,
)
from kb_core.models.entry import EntryType, KnowledgeEntry
from kb_core.models.search import SearchResult
from kb_core.near_duplicates import NearDuplicateCheck
from kb_core.supersession import SupersessionReconcileReport

import kb_service.attribution as attribution_module
import kb_service.auth as auth_module
import kb_service.chat_history as chat_history_module
import kb_service.database as database
import kb_service.main as main_module
from kb_service.main import app
from kb_service.models import User

# ─── kb-core config stub ─────────────────────────────────────────────────────


class FakeIngestConfig:
    """Stub for kb_core IngestConfig — exposes only skip_safety."""

    def __init__(self, skip_safety: bool = False) -> None:
        self.skip_safety = skip_safety


class FakeKbConfig:
    """Stub for kb_core KbConfig — exposes only .ingest."""

    def __init__(self, skip_safety: bool = False) -> None:
        self.ingest = FakeIngestConfig(skip_safety=skip_safety)


# ─── graph-builder stub ──────────────────────────────────────────────────────


class FakeGraphBuilder:
    """Records build_for_entry calls."""

    def __init__(self) -> None:
        self.build_calls: list[KnowledgeEntry] = []
        self._raise_on_build: Exception | None = None

    async def build_for_entry(self, entry: KnowledgeEntry) -> None:
        """Record the call; raise if configured to do so."""
        self.build_calls.append(entry)
        if self._raise_on_build is not None:
            raise self._raise_on_build


# ─── main fake KB ─────────────────────────────────────────────────────────────


def make_entry(entry_id: str = "kb-00001") -> KnowledgeEntry:
    """Build a minimal ``KnowledgeEntry`` for use in fake responses."""
    return KnowledgeEntry(
        id=entry_id,
        short_title="Fake entry",
        long_title="A fake knowledge entry",
        knowledge_details="Fake details.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )


class FakeCursor:
    """Fake DB cursor returned by FakeKbDb.execute."""

    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class FakeKbDb:
    """Fake kb-core Database handle (NOT the asyncpg service-auth pool).

    Implements: execute -> FakeCursor (recording calls), commit -> no-op.
    Do NOT reuse or extend FakeDbPool — that is asyncpg-pool-shaped.

    Dual row source so the SAME fake serves both maps-index and read/meta
    routes:

    * If the SQL targets the maps-index discovery (entry_type='mental_map'),
      the cursor rows come from ``kb.maps_rows`` if set, otherwise are
      synthesised from the keys of ``kb.maps_projects``.
    * Otherwise the cursor returns ``self.rows`` verbatim — read/meta tests
      assign ``kb.db.rows`` directly per case.
    """

    def __init__(self, kb: "FakeKnowledgeBase | None" = None) -> None:
        self._kb = kb
        self.rows: list[tuple[Any, ...]] = []
        self.rows_for: dict[str, list[tuple[Any, ...]]] = {}
        self.calls: list[tuple[str, Any]] = []
        self.committed: int = 0

        # ── pointer-candidates route (nudge_routes.py) ──────────────────────
        # Settable per-test; vector_search_result defaults to [] (no hits).
        self.has_owning_map: bool = False
        self.own_embedding: list[float] | None = None
        self.vector_search_result: list[tuple[str, float]] = []
        self.vector_search_calls: list[tuple[list[float], dict[str, Any]]] = []

    async def execute(self, sql: str, params: Any = ()) -> FakeCursor:
        self.calls.append((sql, params))
        # rows_for: first-match-wins in insertion order (substring of SQL)
        for key, rows in self.rows_for.items():
            if key in sql:
                return FakeCursor(rows)
        if "graph_edges" in sql:
            return FakeCursor([(1,)] if self.has_owning_map else [])
        if "knowledge_vec" in sql:
            if self.own_embedding is None:
                return FakeCursor([])
            vec_str = "[" + ",".join(str(v) for v in self.own_embedding) + "]"
            return FakeCursor([(vec_str,)])
        if self._kb is not None and "mental_map" in sql:
            maps_rows = (
                self._kb.maps_rows
                if self._kb.maps_rows is not None
                else [(ref,) for ref in self._kb.maps_projects]
            )
            return FakeCursor(maps_rows)
        return FakeCursor(self.rows)

    async def commit(self) -> None:
        self.committed += 1

    async def vector_search(
        self,
        embedding: list[float],
        limit: int = 20,
        *,
        project_ref: str | None = None,
        entry_type: str | None = None,
        tags: list[str] | None = None,
        contributor: str | None = None,
        team: str | None = None,
    ) -> list[tuple[str, float]]:
        """Fake of the kb-core ``Database.vector_search`` protocol method."""
        self.vector_search_calls.append(
            (
                embedding,
                {
                    "limit": limit,
                    "project_ref": project_ref,
                    "entry_type": entry_type,
                    "tags": tags,
                    "contributor": contributor,
                    "team": team,
                },
            )
        )
        return self.vector_search_result


class FakeGraph:
    """Fake kb-core _GraphAccessor with settable return-value attributes."""

    def __init__(self) -> None:
        self.neighbors_result: list[tuple[str, str, str]] = []
        self.bfs_result: list[tuple[str, int, list[str]]] = []
        self.find_path_result: list[tuple[str, str, str]] | None = None
        self.supersedes_chain_result: list[str] = []
        self.entries_for_scope_result: list[str] = []
        self.vocabulary_result: dict[str, list[str]] = {}
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def neighbors(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        direction: str = "both",
        limit: int = 50,
    ) -> list[tuple[str, str, str]]:
        self.calls.append(
            (
                "neighbors",
                (node_id,),
                {"edge_types": edge_types, "direction": direction, "limit": limit},
            )
        )
        return self.neighbors_result

    async def bfs_entries(
        self,
        start_node: str,
        max_depth: int = 2,
        edge_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[tuple[str, int, list[str]]]:
        self.calls.append(
            (
                "bfs_entries",
                (start_node,),
                {"max_depth": max_depth, "edge_types": edge_types, "limit": limit},
            )
        )
        return self.bfs_result

    async def find_path(
        self,
        source: str,
        target: str,
        max_depth: int = 4,
    ) -> list[tuple[str, str, str]] | None:
        self.calls.append(("find_path", (source, target), {"max_depth": max_depth}))
        return self.find_path_result

    async def supersedes_chain(self, entry_id: str) -> list[str]:
        self.calls.append(("supersedes_chain", (entry_id,), {}))
        return self.supersedes_chain_result

    async def entries_for_scope(
        self,
        scope: str,
        entry_type: str | None = None,
        order_by: str = "created_at",
    ) -> list[str]:
        self.calls.append(
            (
                "entries_for_scope",
                (scope,),
                {"entry_type": entry_type, "order_by": order_by},
            )
        )
        return self.entries_for_scope_result

    async def vocabulary(self, max_nodes: int = 200) -> dict[str, list[str]]:
        self.calls.append(("vocabulary", (), {"max_nodes": max_nodes}))
        return self.vocabulary_result


class FakeKnowledgeBase:
    """Stand-in for the kb-core ``KnowledgeBase`` singleton.

    Covers both the existing ``search``/``close`` surface and the full write
    surface added in P2: store, store_batch, update, deactivate, reactivate,
    bulk_update, get.  All write methods record their kwargs so tests can
    assert on them.

    Set ``_update_raises``, ``_deactivate_raises``, etc. to inject controlled
    ``ValueError`` failures.
    """

    def __init__(
        self,
        results: list[SearchResult],
        filtered_count: int,
        *,
        maps_projects: dict[str, list[dict[str, str]]] | None = None,
    ) -> None:
        self.results = results
        self.filtered_count = filtered_count
        self.maps_projects: dict[str, list[dict[str, str]]] = (
            maps_projects if maps_projects is not None else {}
        )
        self.maps_rows: list[tuple[Any, ...]] | None = None

        # call-recording lists
        self.search_calls: list[tuple[Any, str | None]] = []
        self.store_calls: list[dict[str, Any]] = []
        self.store_batch_calls: list[tuple[list[dict[str, Any]], bool]] = []
        self.update_calls: list[tuple[str, dict[str, Any]]] = []
        self.deactivate_calls: list[tuple[str, str, str | None, str | None]] = []
        # supersession facade (kb.check_supersedes / kb.reconcile_supersession)
        self.supersedes_problems: list[str] = []
        self.check_supersedes_calls: list[tuple[list[str], str | None, EntryType]] = []
        self.reconcile_calls = 0
        # near-duplicate guard facade
        self.near_duplicate_check = NearDuplicateCheck(
            status="checked", candidates=(), top_similarity=None
        )
        self.find_near_duplicates_calls: list[dict[str, Any]] = []
        self.audit_events: list[tuple[str, str | None, str | None, dict[str, Any]]] = []
        self.reactivate_calls: list[tuple[str, str]] = []
        self.bulk_update_calls: list[dict[str, Any]] = []
        self.bulk_update_result: list[tuple[KnowledgeEntry, KnowledgeEntry]] | None = (
            None
        )
        self.recompute_calls: list[list[str]] = []
        self.reconcile_report: SupersessionReconcileReport | None = None
        self.map_eligibility_verdicts: list[MapEligibilityVerdict] = []
        self.set_map_eligibility_override_calls: list[tuple[str, dict[str, Any]]] = []
        self.clear_map_eligibility_override_calls: list[str] = []
        self.clear_map_eligibility_override_result: bool = True

        # cluster/decline ledger (HTTP half; kb-core 246891c facade surface).
        # Canned returns only — the routes delegate ALL matching to
        # kb.match_clusters, so the fake never touches self.db.
        self.cluster_ledger_match_result: ClusterLedgerMatchResult | None = None
        self.match_clusters_calls: list[tuple[str, list[list[str]]]] = []
        self.record_cluster_calls: list[tuple[str, list[str], str]] = []
        self.record_cluster_result: ClusterLedgerRow | None = None
        self.decline_cluster_calls: list[dict[str, Any]] = []
        self.decline_cluster_result: ClusterLedgerRow | None = None

        # injectable errors (write surface)
        self._update_raises: ValueError | None = None
        self._deactivate_raises: ValueError | None = None
        self._reactivate_raises: ValueError | None = None
        self._bulk_update_raises: ValueError | None = None

        # map delete (kb_core.map_delete via the facade): canned record only,
        # so the route under test keeps owning the audit INSERT.
        self.delete_map_calls: list[str] = []
        self.delete_map_record: DeletedMapRecord = DeletedMapRecord(
            entry_id="kb-00001",
            project_ref=None,
            short_title="Fake map",
            long_title="A fake mental map",
            knowledge_details="Fake map body.\n\nDetail entries:\n- kb-00001 one\n",
            pointer_ids=["kb-00001"],
            outbound_edges_deleted=1,
            inbound_edges_deleted=0,
            inbound_referrer_ids=[],
        )
        self._delete_map_raises: ValueError | None = None

        # sub-objects expected by read/meta + write endpoints
        self.db = FakeKbDb(self)
        self.graph = FakeGraph()
        self.config = FakeKbConfig()
        self.graph_builder = FakeGraphBuilder()
        self.graph_enricher: None = None

        # P3 SSE query stream — None means classifier is skipped (explore fallback)
        self.query_llm: Any | None = None

        # P3 chat stream — all three default None (means unavailable)
        self.synthesis_llm: Any | None = None
        self.extraction_llm: Any | None = None
        self.embedder: Any | None = None

        # P2 read/meta + query state
        self.entries: dict[str, KnowledgeEntry] = {}
        self.preflight_result: str = "preflight context"
        self.preflight_calls: list[tuple[str, Any]] = []
        self.ask_calls: list[tuple[str, dict[str, Any]]] = []
        self.summarize_calls: list[tuple[str, dict[str, Any]]] = []
        self.ask_return: tuple[list[tuple[KnowledgeEntry, str]], int] = (
            [(make_search_result().entry, "fake ask context")],
            3,
        )
        self.summarize_return: str = "fake synthesized answer"

        # P2 ingest state — mutable file_result returned by all four fake
        # ingest methods; tests override fake_kb.file_result directly.
        self.file_result: FileResult = FileResult(
            path="doc.md",
            action="ingested",
            reason=None,
            entry_count=1,
            entry_ids=["kb-00002"],
            summary=None,
            chunks_processed=1,
            chunks_skipped=0,
            chunks_flagged=0,
        )
        self.ingest_text_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.ingest_url_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.ingest_url_content_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.ingest_file_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self._ingest_raises: RuntimeError | None = None

        # lifespan shutdown tracking (P: embedding queue / try-finally hardening)
        self.close_calls = 0
        self.start_embedding_worker_calls = 0
        self.stop_embedding_worker_calls = 0

    async def search(
        self, query: Any, *, contributor: str | None = None
    ) -> tuple[list[SearchResult], int]:
        """Record the call and return the configured results."""
        self.search_calls.append((query, contributor))
        return self.results, self.filtered_count

    async def maps_for_project(
        self, project_ref: str, *, team: str | None = None
    ) -> list[dict[str, str]]:
        """Return configured maps for the given project_ref."""
        return self.maps_projects.get(project_ref, [])

    async def get(self, entry_id: str) -> KnowledgeEntry | None:
        """Return the configured entry for entry_id, or None on miss."""
        return self.entries.get(entry_id)

    async def preflight(self, project_ref: str, *, since: Any = None) -> str:
        """Record the call and return the configured preflight_result."""
        self.preflight_calls.append((project_ref, since))
        return self.preflight_result

    async def ask(
        self,
        question: str,
        *,
        scope: str | None = None,
        agentic: bool | None = None,
        max_tool_calls: int | None = None,
        limit: int = 20,
        include_graph_context: bool = True,
        event_callback: Any | None = None,
    ) -> tuple[list[tuple[KnowledgeEntry, str]], int]:
        """Record the call and return the configured ask results."""
        self.ask_calls.append(
            (
                question,
                {
                    "scope": scope,
                    "agentic": agentic,
                    "max_tool_calls": max_tool_calls,
                    "limit": limit,
                    "include_graph_context": include_graph_context,
                    "event_callback": event_callback,
                },
            )
        )
        return self.ask_return

    async def summarize(
        self,
        question: str,
        *,
        scope: str | None = None,
        agentic: bool | None = None,
        agentic_synthesis: bool | None = None,
        max_tool_calls: int | None = None,
        limit: int = 20,
        event_callback: Any | None = None,
    ) -> str:
        """Record the call and return the configured summarize result."""
        self.summarize_calls.append(
            (
                question,
                {
                    "scope": scope,
                    "agentic": agentic,
                    "agentic_synthesis": agentic_synthesis,
                    "max_tool_calls": max_tool_calls,
                    "limit": limit,
                    "event_callback": event_callback,
                },
            )
        )
        return self.summarize_return

    async def close(self) -> None:
        """No-op close; records the call so lifespan-shutdown tests can assert on it."""
        self.close_calls += 1

    # ── embedding retry queue / background worker ───────────────────────────

    async def start_embedding_worker(self) -> None:
        """No-op — lifespan-driven TestClient suites don't run a real worker."""
        self.start_embedding_worker_calls += 1

    async def stop_embedding_worker(self) -> None:
        """No-op — see start_embedding_worker."""
        self.stop_embedding_worker_calls += 1

    async def embedding_queue_stats(self) -> dict[str, Any]:
        """Return a canned five-key stats dict (mirrors EmbeddingQueueStats)."""
        return {
            "pending": 3,
            "exhausted": 1,
            "oldest_pending_age_seconds": 42.5,
            "next_due_at": "2026-09-16T12:00:00+00:00",
            "vectorless_unqueued": 0,
        }

    embedding_worker_running: bool = True

    # ── map eligibility (override review surface) ────────────────────────────

    async def map_eligibility(self) -> list[MapEligibilityVerdict]:
        """Return the configured verdict list (tests set it directly)."""
        return self.map_eligibility_verdicts

    async def set_map_eligibility_override(
        self,
        project_ref: str,
        *,
        eligible: bool,
        reason: str,
        set_by: str | None = None,
    ) -> MapEligibilityOverride:
        """Record the call and return the fake override row."""
        self.set_map_eligibility_override_calls.append(
            (project_ref, {"eligible": eligible, "reason": reason, "set_by": set_by})
        )
        return MapEligibilityOverride(
            project_ref, eligible, reason, set_by, "2026-09-19T12:00:00+00:00"
        )

    async def clear_map_eligibility_override(self, project_ref: str) -> bool:
        """Record the call and return the configured clear result."""
        self.clear_map_eligibility_override_calls.append(project_ref)
        return self.clear_map_eligibility_override_result

    # ── cluster/decline ledger ───────────────────────────────────────────────

    async def match_clusters(
        self, project_ref: str, candidates: list[list[str]]
    ) -> ClusterLedgerMatchResult:
        """Record the batch and return the canned result.

        With no canned result set, a zero-row batch result is built from
        kb-core's own pinned constants (the route echoes them, so the fake
        must too).
        """
        self.match_clusters_calls.append((project_ref, [list(c) for c in candidates]))
        if self.cluster_ledger_match_result is not None:
            return self.cluster_ledger_match_result
        return ClusterLedgerMatchResult(
            project_ref=project_ref,
            jaccard_threshold=CLUSTER_MATCH_JACCARD,
            near_miss_floor=CLUSTER_NEAR_MISS_FLOOR,
            marginal_match_ceiling=CLUSTER_MARGINAL_MATCH_CEILING,
            reopen_growth_factor=REOPEN_GROWTH_FACTOR,
            rows_considered=0,
            verdicts=(),
        )

    async def record_cluster(
        self, project_ref: str, member_entry_ids: list[str], label: str
    ) -> ClusterLedgerRow:
        """Record the call and return the canned row (a proposed row by default)."""
        self.record_cluster_calls.append((project_ref, list(member_entry_ids), label))
        if self.record_cluster_result is not None:
            return self.record_cluster_result
        return make_cluster_ledger_row()

    async def decline_cluster(
        self, cluster_key: str, *, reason: str, declined_by: str | None = None
    ) -> ClusterLedgerRow | None:
        """Record the call and return the canned declined row (declined by default)."""
        self.decline_cluster_calls.append(
            {"cluster_key": cluster_key, "reason": reason, "declined_by": declined_by}
        )
        if self.decline_cluster_result is not None:
            return self.decline_cluster_result
        return make_cluster_ledger_row(status="declined", declined_member_count=2)

    # ── write surface (P2) ───────────────────────────────────────────────────

    async def store(self, **kwargs: Any) -> KnowledgeEntry:
        """Record kwargs and return a fake entry."""
        self.store_calls.append(kwargs)
        return make_entry()

    async def store_batch(
        self, entries: list[dict[str, Any]], *, enrich: bool = True
    ) -> list[KnowledgeEntry]:
        """Record the call and return one fake entry per input dict."""
        self.store_batch_calls.append((entries, enrich))
        return [make_entry() for _ in entries]

    async def update(self, entry_id: str, **kwargs: Any) -> KnowledgeEntry:
        """Record kwargs; raise configured error if set."""
        self.update_calls.append((entry_id, kwargs))
        if self._update_raises is not None:
            raise self._update_raises
        return make_entry()

    async def deactivate(
        self,
        entry_id: str,
        *,
        contributor: str,
        change_reason: str | None = None,
        superseded_by: str | None = None,
    ) -> KnowledgeEntry:
        """Record the call as a 4-tuple; raise configured error if set."""
        self.deactivate_calls.append(
            (entry_id, contributor, change_reason, superseded_by)
        )
        if self._deactivate_raises is not None:
            raise self._deactivate_raises
        return make_entry()

    async def check_supersedes(
        self,
        target_ids: list[str],
        *,
        writer_id: str | None,
        writer_entry_type: EntryType,
    ) -> list[str]:
        """Record the call and return the configured ``supersedes_problems``."""
        self.check_supersedes_calls.append(
            (list(target_ids), writer_id, writer_entry_type)
        )
        return list(self.supersedes_problems)

    async def find_near_duplicates(
        self,
        *,
        short_title: str,
        long_title: str,
        knowledge_details: str,
        project_ref: str,
        floor: float,
        limit: int = 5,
    ) -> NearDuplicateCheck:
        """Record the call and return the configured ``near_duplicate_check``."""
        self.find_near_duplicates_calls.append(
            {
                "short_title": short_title,
                "long_title": long_title,
                "knowledge_details": knowledge_details,
                "project_ref": project_ref,
                "floor": floor,
                "limit": limit,
            }
        )
        return self.near_duplicate_check

    async def record_audit_event(
        self,
        event_type: str,
        *,
        entry_id: str | None,
        contributor: str | None,
        detail: str,
    ) -> None:
        """Record the audit row with its JSON detail decoded."""
        self.audit_events.append(
            (event_type, entry_id, contributor, json.loads(detail))
        )

    async def recompute_supersession(self, entry_ids: Any) -> None:
        """Record the ids the route asked to recompute."""
        self.recompute_calls.append(sorted(entry_ids))

    async def reconcile_supersession(self) -> SupersessionReconcileReport:
        """Count the call and return the canned (default all-zero) report."""
        self.reconcile_calls += 1
        if self.reconcile_report is not None:
            return self.reconcile_report
        return SupersessionReconcileReport(
            edges_added=0, set_count=0, cleared_count=0, changed=(), edges_added_ids=()
        )

    async def reactivate(self, entry_id: str, *, contributor: str) -> KnowledgeEntry:
        """Record the call; raise configured error if set."""
        self.reactivate_calls.append((entry_id, contributor))
        if self._reactivate_raises is not None:
            raise self._reactivate_raises
        return make_entry()

    async def delete_map(self, entry_id: str) -> DeletedMapRecord:
        """Record the call and return the canned record, or raise the injected error.

        The real facade method hard-deletes one mental_map and returns the
        full ``DeletedMapRecord``; the fake never touches ``self.db`` so the
        route's own audit INSERT is the only row the tests see recorded.
        """

        self.delete_map_calls.append(entry_id)
        if self._delete_map_raises is not None:
            raise self._delete_map_raises
        return self.delete_map_record

    async def bulk_update(
        self,
        filters: dict[str, Any],
        updates: dict[str, Any],
        *,
        contributor: str,
        dry_run: bool = True,
    ) -> list[tuple[KnowledgeEntry, KnowledgeEntry]]:
        """Record the call; raise configured error if set."""
        self.bulk_update_calls.append(
            {
                "filters": filters,
                "updates": updates,
                "contributor": contributor,
                "dry_run": dry_run,
            }
        )
        if self._bulk_update_raises is not None:
            raise self._bulk_update_raises
        if self.bulk_update_result is not None:
            return self.bulk_update_result
        return [(make_entry(), make_entry())]

    # ── ingest surface (P2) ──────────────────────────────────────────────────

    async def ingest_text(
        self,
        content: str,
        source_name: str,
        *,
        project_ref: str | None = None,
        dry_run: bool = False,
        contributor: str | None = None,
        team: str | None = None,
    ) -> FileResult:
        """Record call args/kwargs; raise configured error; return file_result."""
        self.ingest_text_calls.append(
            (
                (content, source_name),
                {
                    "project_ref": project_ref,
                    "dry_run": dry_run,
                    "contributor": contributor,
                    "team": team,
                },
            )
        )
        if self._ingest_raises is not None:
            raise self._ingest_raises
        return self.file_result

    async def ingest_url(
        self,
        url: str,
        *,
        project_ref: str | None = None,
        dry_run: bool = False,
        contributor: str | None = None,
        team: str | None = None,
    ) -> FileResult:
        """Record call args/kwargs; raise configured error; return file_result."""
        self.ingest_url_calls.append(
            (
                (url,),
                {
                    "project_ref": project_ref,
                    "dry_run": dry_run,
                    "contributor": contributor,
                    "team": team,
                },
            )
        )
        if self._ingest_raises is not None:
            raise self._ingest_raises
        return self.file_result

    async def ingest_url_content(
        self,
        content: str,
        source_url: str,
        *,
        project_ref: str | None = None,
        dry_run: bool = False,
        contributor: str | None = None,
        team: str | None = None,
    ) -> FileResult:
        """Record call args/kwargs; raise configured error; return file_result."""
        self.ingest_url_content_calls.append(
            (
                (content, source_url),
                {
                    "project_ref": project_ref,
                    "dry_run": dry_run,
                    "contributor": contributor,
                    "team": team,
                },
            )
        )
        if self._ingest_raises is not None:
            raise self._ingest_raises
        return self.file_result

    async def ingest_file(
        self,
        path: Path | str,
        *,
        project_ref: str | None = None,
        dry_run: bool = False,
        contributor: str | None = None,
        team: str | None = None,
    ) -> FileResult:
        """Record call args/kwargs; raise configured error; return file_result."""
        self.ingest_file_calls.append(
            (
                (path,),
                {
                    "project_ref": project_ref,
                    "dry_run": dry_run,
                    "contributor": contributor,
                    "team": team,
                },
            )
        )
        if self._ingest_raises is not None:
            raise self._ingest_raises
        return self.file_result


class FakeDbPool:
    """Minimal DbPool whose row lookups return None by default."""

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        return []

    async def fetchrow(self, sql: str, *args: Any) -> Any | None:
        return None

    async def execute(self, sql: str, *args: Any) -> str:
        return "OK"

    def acquire(self) -> Any:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class StatefulFakeDbPool(FakeDbPool):
    """FakeDbPool extended with a stateful app_config store and optional user rows.

    Pass in a shared dict so that multiple calls to get_db() within the same
    test all see the same in-memory state — required for round-trip tests that
    PUT a setting then GET it back.

    Pass ``users`` to pre-seed user rows for JWT authentication tests —
    ``get_current_user_from_token`` does a real ``SELECT * FROM users WHERE id``
    that would 401 on an empty pool.
    """

    def __init__(
        self,
        app_config: dict[str, str],
        users: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._app_config = app_config
        self._users: dict[str, dict[str, Any]] = users or {}
        # in-memory chat store: {chat_id: {id, user_id, title, mode, updated_at}}
        self._chats: dict[str, dict[str, Any]] = {}
        # {chat_id: [{role, content}]}
        self._chat_messages: dict[str, list[dict[str, str]]] = {}

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        if "FROM chats" in sql and "user_id = $1" in sql:
            user_id = args[0]
            limit = args[1] if len(args) > 1 else 50
            rows = [c for c in self._chats.values() if c["user_id"] == user_id]
            rows.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
            return rows[:limit]
        if "FROM chat_messages" in sql and "chat_id = $1" in sql:
            chat_id = args[0]
            return list(self._chat_messages.get(str(chat_id), []))
        return []

    async def fetchrow(self, sql: str, *args: Any) -> Any | None:
        if "app_config" in sql and "SELECT value" in sql:
            key = args[0]
            val = self._app_config.get(key)
            if val is None:
                return None
            return {"value": val}
        if "FROM users WHERE id" in sql:
            user_id = str(args[0])
            return self._users.get(user_id)
        if "FROM chats WHERE id = $1 AND user_id = $2" in sql:
            chat_id = str(args[0])
            user_id = str(args[1])
            chat = self._chats.get(chat_id)
            if chat is not None and chat["user_id"] == user_id:
                return {"exists": 1}
            return None
        return await super().fetchrow(sql, *args)

    async def execute(self, sql: str, *args: Any) -> str:
        if "INSERT INTO app_config" in sql:
            # args: key, value, updated_at, updated_by
            self._app_config[args[0]] = args[1]
        elif "DELETE FROM app_config" in sql:
            self._app_config.pop(args[0], None)
        elif "INSERT INTO chats" in sql:
            # args: id, user_id, title, mode, created_at, updated_at
            chat_id = str(args[0])
            self._chats[chat_id] = {
                "id": chat_id,
                "user_id": str(args[1]),
                "title": str(args[2]),
                "mode": str(args[3]),
                "created_at": str(args[4]),
                "updated_at": str(args[5]),
            }
            self._chat_messages.setdefault(chat_id, [])
        elif "INSERT INTO chat_messages" in sql:
            # args: chat_id, role, content, created_at
            chat_id = str(args[0])
            self._chat_messages.setdefault(chat_id, []).append(
                {"role": str(args[1]), "content": str(args[2])}
            )
        elif "UPDATE chats SET updated_at" in sql:
            # args: updated_at, chat_id
            chat_id = str(args[1])
            if chat_id in self._chats:
                self._chats[chat_id]["updated_at"] = str(args[0])
        elif "DELETE FROM chats WHERE id = $1 AND user_id = $2" in sql:
            chat_id = str(args[0])
            user_id = str(args[1])
            chat = self._chats.get(chat_id)
            if chat is not None and chat["user_id"] == user_id:
                del self._chats[chat_id]
                self._chat_messages.pop(chat_id, None)
                return "DELETE 1"
            return "DELETE 0"
        return "OK"


class FakeLLM:
    """Scriptable LLM stub — returns responses in the order they were enqueued.

    Call ``fake_llm.enqueue("...")`` before each expected ``generate_chat`` or
    ``generate`` call.  When the queue is exhausted, subsequent calls return
    ``None``.

    Both ``generate_chat`` and ``generate`` share the same ``_responses`` queue
    so test scripts are interchangeable between the two calling conventions.
    """

    def __init__(self) -> None:
        self._responses: list[str | None] = []
        # MANDATORY call recording (mirrors FakeKnowledgeBase.search_calls).
        # Each entry is (prompt, system) for generate(); tests assert on this.
        self.generate_calls: list[tuple[Any, Any]] = []

    def enqueue(self, response: str | None) -> None:
        """Add *response* to the tail of the response queue."""
        self._responses.append(response)

    async def generate_chat(self, messages: Any, *, system: Any = None) -> str | None:
        """Pop and return the next scripted response (None when exhausted)."""
        if self._responses:
            return self._responses.pop(0)
        return None

    async def generate(self, prompt: Any, *, system: Any = None) -> str | None:
        """Record (prompt, system) then pop and return the next scripted response.

        Mirrors the ``AnthropicLLMClient.generate`` signature (anthropic.py:51-76).
        Appends to ``self.generate_calls`` — tests assert ``len(generate_calls)``
        and inspect the prompt text.
        """
        self.generate_calls.append((prompt, system))
        if self._responses:
            return self._responses.pop(0)
        return None


def make_search_result() -> SearchResult:
    """Build one SearchResult-shaped object for the non-empty search case."""
    entry = KnowledgeEntry(
        id="kb-00001",
        short_title="Example entry",
        long_title="An example knowledge entry",
        knowledge_details="Some details.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    return SearchResult(
        entry=entry,
        score=0.9,
        effective_confidence=0.9,
        staleness_warning=None,
        match_source="hybrid",
    )


def make_cluster_ledger_row(
    cluster_key: str = "0123456789abcdef",
    *,
    project_ref: str = "proj",
    member_entry_ids: tuple[str, ...] = ("kb-00001", "kb-00002"),
    last_label: str = "",
    status: str = "proposed",
    sightings: int = 1,
    first_seen_at: str = "2026-03-01T12:00:00+00:00",
    last_seen_at: str = "2026-03-01T12:00:00+00:00",
    declined_member_count: int | None = None,
    declined_reason: str | None = None,
    declined_by: str | None = None,
    declined_at: str | None = None,
) -> ClusterLedgerRow:
    """Build one canned ``ClusterLedgerRow`` for the ledger fakes."""
    return ClusterLedgerRow(
        cluster_key=cluster_key,
        project_ref=project_ref,
        member_entry_ids=member_entry_ids,
        last_label=last_label,
        status=status,  # type: ignore[arg-type]
        sightings=sightings,
        first_seen_at=first_seen_at,
        last_seen_at=last_seen_at,
        declined_member_count=declined_member_count,
        declined_reason=declined_reason,
        declined_by=declined_by,
        declined_at=declined_at,
    )


def make_map_eligibility_verdict(
    project_ref: str,
    *,
    mappable: int,
    ingested: int,
    top_prefix_count: int,
    top_prefix: str,
    maps: int = 0,
    override: MapEligibilityOverride | None = None,
    orphaned: bool = False,
) -> MapEligibilityVerdict:
    """Build one MapEligibilityVerdict via kb-core's own ``classify``.

    The evidence fields are NEVER hand-populated — kb-core's
    ``classify()`` is the single source of truth for the flag logic, so
    the fixtures track it instead of re-encoding it.
    """
    evidence = classify(
        project_ref,
        mappable=mappable,
        ingested=ingested,
        top_prefix_count=top_prefix_count,
        top_prefix=top_prefix,
        maps=maps,
    )
    if override is not None:
        effective_eligible = override.eligible
        decided_by = "override"
    else:
        effective_eligible = evidence.computed_eligible
        decided_by = "computed"
    return MapEligibilityVerdict(
        evidence=evidence,
        override=override,
        effective_eligible=effective_eligible,
        decided_by=decided_by,
        orphaned=orphaned,
    )


def fake_user() -> User:
    """A fake authenticated user for dependency-override cases."""
    return User(
        id="00000000-0000-0000-0000-000000000001",
        email="tester@example.com",
        hashed_password="x",
        is_admin=False,
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
    )


def fake_admin_user() -> User:
    """A fake admin user for dependency-override cases requiring admin privileges."""
    return User(
        id="00000000-0000-0000-0000-000000000002",
        email="admin@example.com",
        hashed_password="x",
        is_admin=True,
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
def fake_kb(request: pytest.FixtureRequest) -> FakeKnowledgeBase:
    """Fake KB. Returns no results by default; one result when indirectly
    parametrized with ``one_result``.
    """
    if getattr(request, "param", None) == "one_result":
        return FakeKnowledgeBase(results=[make_search_result()], filtered_count=1)
    return FakeKnowledgeBase(results=[], filtered_count=0)


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> Iterator[TestClient]:
    """A TestClient with a hermetic lifespan and mocked kb + service-auth DB."""

    async def _fake_init_db() -> None:
        return None

    async def _fake_close_db() -> None:
        return None

    async def _fake_create_postgres(*args: Any, **kwargs: Any) -> FakeKnowledgeBase:
        return fake_kb

    # A single stateful pool instance shared across all get_db() calls in this
    # test, so round-trip PUT→GET tests see the same in-memory app_config state.
    _shared_pool = StatefulFakeDbPool({})

    async def _fake_get_db() -> StatefulFakeDbPool:
        return _shared_pool

    # Force the Postgres branch of _open_kb so the un-patched create_sqlite
    # is never reached. KB_DATABASE_URL is set NOWHERE in the suite today;
    # without this, _open_kb would fall through to create_sqlite and open the
    # user's real ~/.local/share/personal_kb/knowledge.db.
    monkeypatch.setenv("KB_DATABASE_URL", "postgresql://test/test")
    monkeypatch.setattr(main_module, "init_db", _fake_init_db)
    monkeypatch.setattr(main_module, "close_db", _fake_close_db)
    monkeypatch.setattr(main_module, "create_postgres", _fake_create_postgres)
    monkeypatch.setattr(database, "get_db", _fake_get_db)
    # auth.py and attribution.py each bind get_db at import time; patch both.
    monkeypatch.setattr(auth_module, "get_db", _fake_get_db)
    monkeypatch.setattr(attribution_module, "get_db", _fake_get_db)

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def chat_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, str]]:
    """TestClient pre-wired for chat route tests.

    Provides:
    - A ``StatefulFakeDbPool`` seeded with ``fake_user()``'s row so JWT
      ``?token=`` auth on ``/api/chat/stream`` resolves correctly without a
      real Postgres connection.
    - All ``kb_service.chat_history`` module functions replaced with
      in-memory dict-backed fakes so persistence is hermetic and inspectable.
    - A ``FakeKnowledgeBase`` with one search result attached as
      ``app.state.kb`` (tests set ``app.state.kb.synthesis_llm`` directly).
    - A valid JWT ``token`` (string) for the seeded user.

    Yields:
        ``(TestClient, token_string)``
    """
    # ── in-memory chat store ─────────────────────────────────────────────────
    _chats: dict[str, dict[str, Any]] = {}
    _msgs: dict[str, list[dict[str, str]]] = {}

    async def _create_chat(
        chat_id: str, user_id: str, title: str, mode: str = ""
    ) -> dict[str, str]:
        now = datetime.now(UTC).isoformat()
        _chats[chat_id] = {
            "id": chat_id,
            "user_id": user_id,
            "title": title,
            "mode": mode,
            "updated_at": now,
        }
        _msgs.setdefault(chat_id, [])
        return {"id": chat_id, "title": title, "mode": mode, "updated_at": now}

    async def _save_message(chat_id: str, role: str, content: str) -> None:
        _msgs.setdefault(chat_id, []).append({"role": role, "content": content})

    async def _save_messages_bulk(chat_id: str, messages: list[dict[str, str]]) -> None:
        _msgs.setdefault(chat_id, []).extend(messages)

    async def _list_chats(user_id: str, limit: int = 50) -> list[dict[str, str]]:
        return [
            {k: v for k, v in c.items() if k != "user_id"}  # type: ignore[misc]
            | {"id": c["id"]}
            for c in _chats.values()
            if c["user_id"] == user_id
        ][:limit]

    async def _get_messages(chat_id: str) -> list[dict[str, str]]:
        return list(_msgs.get(chat_id, []))

    async def _delete_chat(chat_id: str, user_id: str) -> bool:
        c = _chats.get(chat_id)
        if c is not None and c["user_id"] == user_id:
            del _chats[chat_id]
            _msgs.pop(chat_id, None)
            return True
        return False

    async def _chat_exists(chat_id: str, user_id: str) -> bool:
        c = _chats.get(chat_id)
        return c is not None and c["user_id"] == user_id

    monkeypatch.setattr(chat_history_module, "create_chat", _create_chat)
    monkeypatch.setattr(chat_history_module, "save_message", _save_message)
    monkeypatch.setattr(chat_history_module, "save_messages_bulk", _save_messages_bulk)
    monkeypatch.setattr(chat_history_module, "list_chats", _list_chats)
    monkeypatch.setattr(chat_history_module, "get_messages", _get_messages)
    monkeypatch.setattr(chat_history_module, "delete_chat", _delete_chat)
    monkeypatch.setattr(chat_history_module, "chat_exists", _chat_exists)

    # ── KB + DB fakes ────────────────────────────────────────────────────────
    fake_kb = FakeKnowledgeBase(results=[make_search_result()], filtered_count=1)
    user = fake_user()
    user_row: dict[str, Any] = {
        "id": user.id,
        "email": user.email,
        "hashed_password": user.hashed_password,
        "is_admin": 0,
        "created_at": user.created_at.isoformat(),
    }
    _shared_pool = StatefulFakeDbPool({}, users={user.id: user_row})

    async def _fake_init_db() -> None:
        return None

    async def _fake_close_db() -> None:
        return None

    async def _fake_create_postgres(*args: Any, **kwargs: Any) -> FakeKnowledgeBase:
        return fake_kb

    async def _fake_get_db() -> StatefulFakeDbPool:
        return _shared_pool

    # Force the Postgres branch of _open_kb so the un-patched create_sqlite
    # is never reached. See the `client` fixture for the full rationale.
    monkeypatch.setenv("KB_DATABASE_URL", "postgresql://test/test")
    monkeypatch.setattr(main_module, "init_db", _fake_init_db)
    monkeypatch.setattr(main_module, "close_db", _fake_close_db)
    monkeypatch.setattr(main_module, "create_postgres", _fake_create_postgres)
    monkeypatch.setattr(database, "get_db", _fake_get_db)
    monkeypatch.setattr(auth_module, "get_db", _fake_get_db)
    monkeypatch.setattr(attribution_module, "get_db", _fake_get_db)

    token = auth_module.create_token(user.id)

    with TestClient(app) as test_client:
        yield test_client, token

    app.dependency_overrides.clear()


# Production-DB URLs must never reach a test. A dispatch host or a developer
# shell can carry a real KB_DATABASE_URL (it did: a daemon smoke test opened
# the live KB and ran the startup reconcile, 2026-10-07). Tests that need one
# set it explicitly with monkeypatch after this runs.
_AMBIENT_DB_VARS = ("KB_DATABASE_URL", "KB_SERVICE_DATABASE_URL")
# Surprise capture runs with mode off and default models unless a test opts in.
_AMBIENT_SURPRISE_VARS = (
    "KB_SURPRISE_CAPTURE",
    "KB_SURPRISE_DETECTOR_MODEL",
    "KB_SURPRISE_MIN_CONFIDENCE",
    "KB_SURPRISE_MIN_CONFIDENCE_SHAPE2",
    "KB_SURPRISE_MIN_CONFIDENCE_SHAPE3",
    "KB_SURPRISE_DISTILL_MODEL",
)


@pytest.fixture(autouse=True)
def _no_ambient_production_db(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _AMBIENT_DB_VARS + _AMBIENT_SURPRISE_VARS:
        monkeypatch.delenv(var, raising=False)
