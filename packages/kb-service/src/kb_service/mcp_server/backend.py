"""InProcessBackend — the /mcp tools' backend, calling route handlers directly.

Each method mirrors the same-named ``personal_kb.backend.http.HttpBackend``
method (name, signature, return type, request-body construction and response
parsing), but instead of an HTTP round trip it awaits the FastAPI route
handler in-process with a shim ``Request`` that carries the caller's
``AuthPrincipal`` and raw headers. Handlers are looked up as module
attributes at call time so tests can monkeypatch them.

FastAPI dependencies do NOT run on this path, so the admin gate
(``require_admin``) is replicated in ``ADMIN_ONLY_METHODS``; drift guards in
``tests/test_mcp_backend.py`` fail the build if a handler grows a dependency
this class does not replicate.
"""

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypeVar, cast

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from kb_core.ingest.ingester import FileResult
from kb_core.models.entry import EntryType, KnowledgeEntry
from kb_core.models.search import SearchQuery, SearchResult
from pydantic import ValidationError
from starlette.exceptions import HTTPException
from starlette.requests import Request

from kb_service.auth import AuthPrincipal
from kb_service.mcp_server.errors import BackendHttpError
from kb_service.mcp_server.observability import (
    MCP_BACKEND_MARKER,
    record_backend_status,
)
from kb_service.models import (
    AskRequest,
    GetRequest,
    IngestUrlRequest,
    MapEligibilityOverrideClearRequest,
    MapEligibilityOverrideSetRequest,
    SearchRequest,
    SummarizeRequest,
)
from kb_service.models_kb import (
    BulkUpdateRequest,
    DeactivateRequest,
    FeedbackRequest,
    StoreBatchRequest,
    StoreRequest,
)
from kb_service.routes import (
    ingest_routes,
    kb_read_routes,
    kb_routes,
    kb_write_routes,
    map_eligibility_routes,
    query_routes,
)

logger = logging.getLogger(__name__)

ADMIN_ONLY_METHODS: frozenset[str] = frozenset(
    {
        "reactivate",
        "bulk_update",
        "reconcile_supersession",
        "map_eligibility",
        "set_map_eligibility_override",
        "clear_map_eligibility_override",
    }
)

_T = TypeVar("_T")


def _parse_file_result(data: dict[str, Any]) -> FileResult:
    """Reconstruct a :class:`~kb_core.ingest.ingester.FileResult` from JSON."""
    return FileResult(
        path=data.get("path", ""),
        action=data.get("action", "error"),
        reason=data.get("reason"),
        entry_count=data.get("entry_count", 0),
        entry_ids=data.get("entry_ids", []),
        summary=data.get("summary"),
        chunks_processed=data.get("chunks_processed", 0),
        chunks_skipped=data.get("chunks_skipped", 0),
        chunks_flagged=data.get("chunks_flagged", 0),
    )


def _parse_entry(data: dict[str, Any]) -> KnowledgeEntry:
    """Reconstruct a :class:`~kb_core.models.entry.KnowledgeEntry` from JSON."""
    return KnowledgeEntry.model_validate(data)


class InProcessBackend:
    """Backend that calls kb-service route handlers in-process."""

    def __init__(
        self,
        app: FastAPI,
        principal: AuthPrincipal,
        raw_headers: list[tuple[bytes, bytes]],
    ) -> None:
        """Bind the app, the authenticated caller and the client's raw headers."""
        self._app = app
        self._principal = principal
        self._raw_headers = raw_headers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(self) -> Request:
        """A fresh shim request: principal in state, client headers, the app."""
        return Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/mcp",
                "query_string": b"",
                "headers": self._raw_headers,
                "app": self._app,
                "state": {"kb_principal": self._principal},
            }
        )

    def _log_ids(self) -> tuple[str, str]:
        key_id = self._principal.api_key_id
        return self._principal.user.id, key_id if key_id is not None else "none"

    async def _call(
        self,
        op: str,
        build: Callable[[], Awaitable[Any]],
        parse: Callable[[Any], _T],
    ) -> _T:
        """Run one handler call with the admin gate, error mapping and status.

        ``build`` constructs the request models and awaits the handler (so a
        model ``ValidationError`` is mapped like the handler's own errors);
        ``parse`` receives the ``jsonable_encoder`` form of its result.
        """
        try:
            if op in ADMIN_ONLY_METHODS and not self._principal.user.is_admin:
                user_id, key_id = self._log_ids()
                logger.warning(
                    MCP_BACKEND_MARKER
                    + " op=%s status=403 reason=admin_gate user_id=%s key_id=%s",
                    op,
                    user_id,
                    key_id,
                )
                raise BackendHttpError(403, "Admin only")
            try:
                result = await build()
            except BackendHttpError:
                raise
            except HTTPException as exc:
                detail = (
                    exc.detail
                    if isinstance(exc.detail, str)
                    else json.dumps(jsonable_encoder(exc.detail))
                )
                raise BackendHttpError(exc.status_code, detail) from exc
            except ValidationError as exc:
                raise BackendHttpError(
                    422, json.dumps(exc.errors(include_url=False), default=str)
                ) from exc
            except Exception as exc:
                user_id, key_id = self._log_ids()
                logger.exception(
                    MCP_BACKEND_MARKER + " op=%s status=500 user_id=%s key_id=%s",
                    op,
                    user_id,
                    key_id,
                )
                raise BackendHttpError(500, "Internal Server Error") from exc
        except BackendHttpError as exc:
            record_backend_status(exc.status)
            raise
        record_backend_status(200)
        return parse(jsonable_encoder(result))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: SearchQuery,
        contributor: str | None = None,
    ) -> tuple[list[SearchResult], int]:
        """Hybrid search via ``kb_routes.search``."""
        body: dict[str, Any] = {
            "query": query.query or "",
            "limit": query.limit,
            "include_stale": query.include_stale,
            "include_expired": query.include_expired,
            "include_superseded": query.include_superseded,
        }
        if query.project_ref is not None:
            body["project_ref"] = query.project_ref
        if query.entry_type is not None:
            et = query.entry_type
            body["entry_type"] = str(et.value) if hasattr(et, "value") else str(et)
        if query.tags is not None:
            body["tags"] = query.tags
        # Note: contributor/team deliberately NOT sent — the service has no such
        # fields

        async def build() -> Any:
            return await kb_routes.search(
                body=SearchRequest.model_validate(body),
                request=self._request(),
                user=self._principal.user,
            )

        def parse(data: dict[str, Any]) -> tuple[list[SearchResult], int]:
            results = [
                SearchResult(
                    entry=_parse_entry(item["entry"]),
                    score=float(item.get("score", 0.0)),
                    effective_confidence=float(item.get("effective_confidence", 0.0)),
                    staleness_warning=item.get("staleness_warning"),
                    match_source=item.get("match_source", ""),
                )
                for item in data.get("results", [])
            ]
            filtered_count = data.get("filtered_count", 0)
            return results, filtered_count

        return await self._call("search", build, parse)

    async def vector_search_available(self) -> bool:
        """The service always exposes vector search; return True."""
        record_backend_status(200)
        return True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def get_entries(
        self,
        ids: list[str],
    ) -> list[tuple[str, KnowledgeEntry | None, list[tuple[str, str | None]]]]:
        """Fetch entries in batches of 20 (GetRequest max_length)."""
        batch_size = 20
        all_results: list[
            tuple[str, KnowledgeEntry | None, list[tuple[str, str | None]]]
        ] = []

        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size]

            async def build(batch: list[str] = batch) -> Any:
                return await kb_read_routes.get_entries(
                    body=GetRequest.model_validate({"ids": batch}),
                    request=self._request(),
                    user=self._principal.user,
                )

            def parse(
                data: dict[str, Any],
            ) -> list[tuple[str, KnowledgeEntry | None, list[tuple[str, str | None]]]]:
                out: list[
                    tuple[str, KnowledgeEntry | None, list[tuple[str, str | None]]]
                ] = []
                for item in data.get("results", []):
                    eid = item["id"]
                    if not item.get("found", False) or item.get("entry") is None:
                        out.append((eid, None, []))
                    else:
                        entry = _parse_entry(item["entry"])
                        rot_raw = item.get("pointer_rot", [])
                        rot = [
                            (r["target_id"], r.get("superseded_by")) for r in rot_raw
                        ]
                        out.append((eid, entry, rot))
                return out

            all_results.extend(await self._call("get_entries", build, parse))

        return all_results

    async def neighbors(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        direction: str = "both",
        limit: int = 10,
    ) -> list[tuple[str, str, str]]:
        """1-hop graph neighbours via ``kb_read_routes.graph_neighbors``."""

        async def build() -> Any:
            return await kb_read_routes.graph_neighbors(
                node_id=node_id,
                request=self._request(),
                user=self._principal.user,
                edge_types=edge_types or None,
                direction=cast("Literal['outgoing', 'incoming', 'both']", direction),
                limit=limit,
            )

        def parse(data: dict[str, Any]) -> list[tuple[str, str, str]]:
            return [
                (n["neighbor_id"], n["edge_type"], n["direction"])
                for n in data.get("neighbors", [])
            ]

        return await self._call("neighbors", build, parse)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def store(
        self,
        *,
        short_title: str = "",
        long_title: str = "",
        knowledge_details: str = "",
        entry_type: EntryType | None = None,
        project_ref: str | None = None,
        source_context: str | None = None,
        confidence_level: float = 0.9,
        tags: list[str] | None = None,
        hints: dict[str, object] | None = None,
        sensitivity: str | None = None,
        ttl: str | None = None,
        update_entry_id: str | None = None,
        change_reason: str | None = None,
        supersedes: list[str] | Literal["none"] | None = None,
        distinct_from: list[str] | None = None,
    ) -> tuple[Literal["created", "updated"], KnowledgeEntry, list[str] | None]:
        """Create or update via ``kb_write_routes.store``.

        Returns ``(action, entry, superseded_ids)``.
        """
        body: dict[str, Any] = {
            "short_title": short_title,
            "long_title": long_title,
            "knowledge_details": knowledge_details,
            "confidence_level": confidence_level,
        }
        if entry_type is not None:
            body["entry_type"] = (
                str(entry_type.value)
                if hasattr(entry_type, "value")
                else str(entry_type)
            )
        if project_ref is not None:
            body["project_ref"] = project_ref
        if source_context is not None:
            body["source_context"] = source_context
        if tags is not None:
            body["tags"] = tags
        if hints is not None:
            body["hints"] = hints
        if sensitivity is not None:
            body["sensitivity"] = sensitivity
        if ttl is not None:
            body["ttl"] = ttl
        if update_entry_id is not None:
            body["update_entry_id"] = update_entry_id
        if change_reason is not None:
            body["change_reason"] = change_reason
        if supersedes is not None:
            body["supersedes"] = supersedes
        if distinct_from is not None:
            body["distinct_from"] = distinct_from

        async def build() -> Any:
            return await kb_write_routes.store(
                body=StoreRequest.model_validate(body),
                request=self._request(),
                user=self._principal.user,
            )

        def parse(
            data: dict[str, Any],
        ) -> tuple[Literal["created", "updated"], KnowledgeEntry, list[str] | None]:
            action: Literal["created", "updated"] = data["action"]
            entry = _parse_entry(data["entry"])
            raw_ids = data.get("superseded_ids")
            superseded_ids = list(raw_ids) if raw_ids is not None else None
            return action, entry, superseded_ids

        return await self._call("store", build, parse)

    async def deactivate(
        self,
        entry_id: str,
        *,
        change_reason: str | None = None,
        superseded_by: str | None = None,
    ) -> KnowledgeEntry:
        """Soft-delete via ``kb_write_routes.deactivate``."""
        body: dict[str, Any] = {}
        if change_reason is not None:
            body["change_reason"] = change_reason
        if superseded_by is not None:
            body["superseded_by"] = superseded_by

        async def build() -> Any:
            return await kb_write_routes.deactivate(
                entry_id=entry_id,
                request=self._request(),
                user=self._principal.user,
                body=DeactivateRequest.model_validate(body),
            )

        def parse(data: dict[str, Any]) -> KnowledgeEntry:
            return _parse_entry(data["entry"])

        return await self._call("deactivate", build, parse)

    async def reactivate(self, entry_id: str) -> KnowledgeEntry:
        """Undo a deactivation via ``kb_write_routes.reactivate`` (admin)."""

        async def build() -> Any:
            return await kb_write_routes.reactivate(
                entry_id=entry_id,
                request=self._request(),
                user=self._principal.user,
            )

        def parse(data: dict[str, Any]) -> KnowledgeEntry:
            return _parse_entry(data["entry"])

        return await self._call("reactivate", build, parse)

    async def store_batch(
        self,
        entries: list[dict[str, Any]],
    ) -> tuple[list[KnowledgeEntry], list[tuple[int, str, str]], list[list[str]]]:
        """Send all pre-validated entries in one ``kb_write_routes.store_batch``.

        The third element is the per-created-entry ``superseded_ids``.
        """
        if not entries:
            return [], [], []

        # Build the request body — no contributor/team (attributed from the key)
        batch_entries = []
        for e in entries:
            item: dict[str, Any] = {
                "short_title": e["short_title"],
                "long_title": e["long_title"],
                "knowledge_details": e["knowledge_details"],
            }
            if e.get("entry_type"):
                item["entry_type"] = e["entry_type"]
            if e.get("project_ref"):
                item["project_ref"] = e["project_ref"]
            if e.get("source_context"):
                item["source_context"] = e["source_context"]
            if e.get("confidence_level") is not None:
                item["confidence_level"] = float(e["confidence_level"])
            if e.get("tags"):
                item["tags"] = e["tags"]
            if e.get("hints"):
                item["hints"] = e["hints"]
            if e.get("sensitivity"):
                item["sensitivity"] = e["sensitivity"]
            if e.get("ttl"):
                item["ttl"] = e["ttl"]
            if "supersedes" in e:
                item["supersedes"] = e["supersedes"]
            if e.get("distinct_from"):
                item["distinct_from"] = e["distinct_from"]
            batch_entries.append(item)

        async def build() -> Any:
            return await kb_write_routes.store_batch(
                body=StoreBatchRequest.model_validate({"entries": batch_entries}),
                request=self._request(),
                user=self._principal.user,
            )

        def parse(
            data: dict[str, Any],
        ) -> tuple[list[KnowledgeEntry], list[tuple[int, str, str]], list[list[str]]]:
            created = [_parse_entry(e) for e in data.get("created", [])]
            superseded = [
                [str(i) for i in ids] for ids in data.get("superseded_ids") or []
            ]
            # No per-entry detail for server-side failures
            return created, [], superseded

        return await self._call("store_batch", build, parse)

    async def bulk_update(
        self,
        filters: dict[str, Any],
        updates: dict[str, Any],
        dry_run: bool,
    ) -> list[tuple[KnowledgeEntry, KnowledgeEntry]]:
        """Bulk metadata update via ``kb_write_routes.bulk_update`` (admin)."""
        body = {"filters": filters, "updates": updates, "dry_run": dry_run}

        async def build() -> Any:
            return await kb_write_routes.bulk_update(
                body=BulkUpdateRequest.model_validate(body),
                request=self._request(),
                user=self._principal.user,
            )

        def parse(data: dict[str, Any]) -> list[tuple[KnowledgeEntry, KnowledgeEntry]]:
            return [
                (_parse_entry(r["before"]), _parse_entry(r["after"]))
                for r in data.get("results", [])
            ]

        return await self._call("bulk_update", build, parse)

    async def reconcile_supersession(self) -> dict[str, Any]:
        """Heal superseded_by drift via the admin reconcile handler."""

        async def build() -> Any:
            return await kb_write_routes.reconcile_supersession(
                request=self._request(),
                user=self._principal.user,
            )

        def parse(data: Any) -> dict[str, Any]:
            return cast("dict[str, Any]", data)

        return await self._call("reconcile_supersession", build, parse)

    async def feedback(
        self,
        feedback_type: str,
        tool_name: str | None = None,
        query_or_params: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Record agent feedback via ``kb_write_routes.feedback``."""
        body: dict[str, Any] = {"feedback_type": feedback_type}
        if tool_name is not None:
            body["tool_name"] = tool_name
        if query_or_params is not None:
            body["query_or_params"] = query_or_params
        if detail is not None:
            body["detail"] = detail

        async def build() -> Any:
            return await kb_write_routes.feedback(
                body=FeedbackRequest.model_validate(body),
                request=self._request(),
                user=self._principal.user,
            )

        def parse(_data: Any) -> None:
            return None

        await self._call("feedback", build, parse)

    # ------------------------------------------------------------------
    # Ask / Summarize
    # ------------------------------------------------------------------

    async def ask_auto(
        self,
        question: str,
        scope: str | None,
        include_graph_context: bool,
        limit: int,
    ) -> tuple[list[tuple[KnowledgeEntry, str]], int]:
        """Agentic retrieval via ``query_routes.ask``."""
        body: dict[str, Any] = {
            "question": question,
            "include_graph_context": include_graph_context,
            "limit": limit,
        }
        if scope is not None:
            body["scope"] = scope

        async def build() -> Any:
            return await query_routes.ask(
                body=AskRequest.model_validate(body),
                request=self._request(),
                _user=self._principal.user,
            )

        def parse(data: dict[str, Any]) -> tuple[list[tuple[KnowledgeEntry, str]], int]:
            entries_with_context = [
                (_parse_entry(item["entry"]), item.get("context", ""))
                for item in data.get("entries", [])
            ]
            agent_turns_used = data.get("agent_turns_used", 0)
            return entries_with_context, agent_turns_used

        return await self._call("ask_auto", build, parse)

    async def summarize(
        self,
        question: str,
        scope: str | None,
        limit: int,
    ) -> str:
        """Synthesised answer via ``query_routes.summarize``."""
        body: dict[str, Any] = {"question": question, "limit": limit}
        if scope is not None:
            body["scope"] = scope

        async def build() -> Any:
            return await query_routes.summarize(
                body=SummarizeRequest.model_validate(body),
                request=self._request(),
                _user=self._principal.user,
            )

        def parse(data: dict[str, Any]) -> str:
            return cast("str", data.get("answer", ""))

        return await self._call("summarize", build, parse)

    # ------------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------------

    async def preflight(self, project_ref: str, since: str | None) -> str:
        """Project context primer via ``kb_read_routes.preflight``."""

        async def build() -> Any:
            return await kb_read_routes.preflight(
                project_ref=project_ref,
                request=self._request(),
                user=self._principal.user,
                since=since,
            )

        def parse(data: dict[str, Any]) -> str:
            return cast("str", data.get("context", ""))

        return await self._call("preflight", build, parse)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    async def ingest_url(
        self,
        url: str,
        content: str | None,
        project_ref: str | None,
        dry_run: bool,
    ) -> FileResult:
        """URL (or pre-fetched content) ingestion via ``ingest_routes.ingest_url``."""
        body: dict[str, Any] = {"url": url, "dry_run": dry_run}
        if content is not None:
            body["content"] = content
        if project_ref is not None:
            body["project_ref"] = project_ref

        async def build() -> Any:
            return await ingest_routes.ingest_url(
                body=IngestUrlRequest.model_validate(body),
                request=self._request(),
                user=self._principal.user,
            )

        return await self._call("ingest_url", build, _parse_file_result)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_counts(data: dict[str, Any]) -> list[tuple[str, int]]:
        return [(item["name"], item["entry_count"]) for item in data.get("items", [])]

    async def list_projects(self) -> list[tuple[str, int]]:
        """``(name, entry_count)`` pairs via ``kb_read_routes.list_projects``."""

        async def build() -> Any:
            return await kb_read_routes.list_projects(
                request=self._request(), user=self._principal.user
            )

        return await self._call("list_projects", build, self._parse_counts)

    async def list_contributors(self) -> list[tuple[str, int]]:
        """``(name, entry_count)`` pairs via ``kb_read_routes.list_contributors``."""

        async def build() -> Any:
            return await kb_read_routes.list_contributors(
                request=self._request(), user=self._principal.user
            )

        return await self._call("list_contributors", build, self._parse_counts)

    async def list_teams(self) -> list[tuple[str, int]]:
        """``(name, entry_count)`` pairs via ``kb_read_routes.list_teams``."""

        async def build() -> Any:
            return await kb_read_routes.list_teams(
                request=self._request(), user=self._principal.user
            )

        return await self._call("list_teams", build, self._parse_counts)

    # ------------------------------------------------------------------
    # Map eligibility — verdict dicts cross this layer UNPARSED.
    # ------------------------------------------------------------------

    async def map_eligibility(self) -> list[dict[str, Any]]:
        """The ``projects`` verdict list via the admin map-eligibility handler."""

        async def build() -> Any:
            return await map_eligibility_routes.map_eligibility(
                request=self._request(), _admin=self._principal.user
            )

        def parse(data: dict[str, Any]) -> list[dict[str, Any]]:
            return cast("list[dict[str, Any]]", data["projects"])

        return await self._call("map_eligibility", build, parse)

    async def set_map_eligibility_override(
        self,
        project_ref: str,
        *,
        eligible: bool,
        reason: str,
    ) -> dict[str, Any]:
        """Set a human override.  Returns ``{changed, verdict}``."""
        body = {"project_ref": project_ref, "eligible": eligible, "reason": reason}

        async def build() -> Any:
            return await map_eligibility_routes.set_override(
                body=MapEligibilityOverrideSetRequest.model_validate(body),
                request=self._request(),
                user=self._principal.user,
            )

        def parse(data: dict[str, Any]) -> dict[str, Any]:
            return {"changed": bool(data["changed"]), "verdict": data["verdict"]}

        return await self._call("set_map_eligibility_override", build, parse)

    async def clear_map_eligibility_override(self, project_ref: str) -> bool:
        """Clear a human override.  Returns the ``changed`` flag."""
        body = {"project_ref": project_ref}

        async def build() -> Any:
            return await map_eligibility_routes.clear_override(
                body=MapEligibilityOverrideClearRequest.model_validate(body),
                request=self._request(),
                user=self._principal.user,
            )

        def parse(data: dict[str, Any]) -> bool:
            return bool(data["changed"])

        return await self._call("clear_map_eligibility_override", build, parse)
