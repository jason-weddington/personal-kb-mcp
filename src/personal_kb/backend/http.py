"""HttpBackend — thin MCP client over the KB HTTP service.

All configuration (base URL, API key) is read from environment variables at
construction time via :func:`personal_kb.config.get_personal_kb_url` and
:func:`personal_kb.config.get_personal_kb_api_key`.  The backend holds a
single shared :class:`httpx.AsyncClient` that is opened in
:meth:`__aenter__` / closed in :meth:`__aexit__`.

Timeouts (pinned by the spec):
- Default: connect=10s, read=60s, write=60s, pool=10s
- Long operations (ask_auto, summarize, ingest_file, ingest_url): read=300s
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Literal, cast

import httpx

if TYPE_CHECKING:
    from pathlib import Path

    from kb_core.ingest.ingester import FileResult
    from kb_core.models.entry import EntryType, KnowledgeEntry
    from kb_core.models.search import SearchQuery, SearchResult

logger = logging.getLogger(__name__)

# Default timeout applied to every request.
_DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0)

# Extended read timeout for long-running operations.
_LONG_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=10.0)


class BackendHttpError(Exception):
    """Non-2xx response from the KB service."""

    def __init__(self, status: int, detail: str) -> None:
        """Initialise with HTTP *status* code and error *detail* string."""
        self.status = status
        self.detail = detail
        super().__init__(f"KB service returned {status}: {detail}")


def _extract_detail(response: httpx.Response) -> str:
    """Parse ``detail`` from a FastAPI error response body."""
    try:
        body = response.json()
        detail = body.get("detail", response.text)
        if isinstance(detail, str):
            return detail
        # FastAPI RequestValidationError returns a list
        return json.dumps(detail)
    except Exception:
        return response.text


def _raise_for_status(response: httpx.Response, base_url: str) -> None:
    """Raise :class:`BackendHttpError` for non-2xx responses."""
    if response.is_success:
        return
    detail = _extract_detail(response)
    raise BackendHttpError(response.status_code, detail)


def _map_error(exc: BackendHttpError, base_url: str) -> str:
    """Convert a BackendHttpError to a human-readable tool-layer error string."""
    if exc.status == 401:
        return "Error: KB service authentication failed (401). Check PERSONAL_KB_API_KEY."
    if exc.status == 403:
        return f"Error: admin privileges required (403): {exc.detail}"
    if exc.status in (404, 409):
        return f"Error: {exc.detail}"
    return f"Error: KB service returned {exc.status}: {exc.detail}"


def _parse_file_result(data: dict[str, Any]) -> FileResult:
    """Reconstruct a :class:`~kb_core.ingest.ingester.FileResult` from JSON."""
    from kb_core.ingest.ingester import FileResult

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
    from kb_core.models.entry import KnowledgeEntry

    return KnowledgeEntry.model_validate(data)


class HttpBackend:
    """Backend that talks to a remote KB service over HTTP."""

    def __init__(self, base_url: str, api_key: str) -> None:
        """Initialise with service *base_url* and bearer *api_key*."""
        # Strip trailing slash so all paths can start with '/'
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle (used by server.py lifespan)
    # ------------------------------------------------------------------

    async def open(self) -> None:
        """Open the shared httpx client."""
        headers = {"Authorization": f"Bearer {self._api_key}"}
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=_DEFAULT_TIMEOUT,
        )

    async def close(self) -> None:
        """Close the shared httpx client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> HttpBackend:
        """Open the client and return self."""
        await self.open()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the client on context exit."""
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _c(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("HttpBackend not opened — call await backend.open() first")
        return self._client

    async def _post(
        self,
        path: str,
        body: dict[str, Any],
        *,
        timeout: httpx.Timeout | None = None,
    ) -> dict[str, Any]:
        try:
            kwargs: dict[str, Any] = {"json": body}
            if timeout is not None:
                kwargs["timeout"] = timeout
            resp = await self._c().post(path, **kwargs)
            _raise_for_status(resp, self._base_url)
            return cast("dict[str, Any]", resp.json())
        except httpx.ConnectError as exc:
            msg = f"cannot reach KB service at {self._base_url}: {exc}"
            raise BackendHttpError(0, msg) from exc
        except httpx.TimeoutException as exc:
            msg = f"cannot reach KB service at {self._base_url}: {exc}"
            raise BackendHttpError(0, msg) from exc

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: httpx.Timeout | None = None,
    ) -> dict[str, Any]:
        try:
            kwargs: dict[str, Any] = {"params": params or {}}
            if timeout is not None:
                kwargs["timeout"] = timeout
            resp = await self._c().get(path, **kwargs)
            _raise_for_status(resp, self._base_url)
            return cast("dict[str, Any]", resp.json())
        except httpx.ConnectError as exc:
            msg = f"cannot reach KB service at {self._base_url}: {exc}"
            raise BackendHttpError(0, msg) from exc
        except httpx.TimeoutException as exc:
            msg = f"cannot reach KB service at {self._base_url}: {exc}"
            raise BackendHttpError(0, msg) from exc

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    @property
    def is_remote(self) -> bool:
        """Always True — this backend calls the remote KB service."""
        return True

    # -- Search ------------------------------------------------------------

    async def search(
        self,
        query: SearchQuery,
        contributor: str | None = None,
    ) -> tuple[list[SearchResult], int]:
        """POST /api/kb/search and parse results.  Returns (results, filtered_count)."""
        from kb_core.models.search import SearchResult

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
        # Note: contributor/team deliberately NOT sent — the service has no such fields

        data = await self._post("/api/kb/search", body)
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

    async def vector_search_available(self) -> bool:
        """The remote service always exposes vector search; return True."""
        # The service exposes no availability endpoint; assume True.
        return True

    # -- Retrieval ---------------------------------------------------------

    async def get_entries(
        self,
        ids: list[str],
    ) -> list[tuple[str, KnowledgeEntry | None, list[tuple[str, str | None]]]]:
        """Fetch entries in batches of 20 (service GetRequest max_length)."""
        batch_size = 20
        all_results: list[tuple[str, KnowledgeEntry | None, list[tuple[str, str | None]]]] = []

        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size]
            data = await self._post("/api/kb/get", {"ids": batch})
            for item in data.get("results", []):
                eid = item["id"]
                if not item.get("found", False) or item.get("entry") is None:
                    all_results.append((eid, None, []))
                else:
                    entry = _parse_entry(item["entry"])
                    rot_raw = item.get("pointer_rot", [])
                    rot = [(r["target_id"], r.get("superseded_by")) for r in rot_raw]
                    all_results.append((eid, entry, rot))

        return all_results

    # -- Write -------------------------------------------------------------

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
        """POST /api/kb/store.  Returns (action, entry, superseded_ids).

        ``superseded_ids`` is None when the response lacks the key (old server).
        """
        body: dict[str, Any] = {
            "short_title": short_title,
            "long_title": long_title,
            "knowledge_details": knowledge_details,
            "confidence_level": confidence_level,
        }
        if entry_type is not None:
            body["entry_type"] = (
                str(entry_type.value) if hasattr(entry_type, "value") else str(entry_type)
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

        data = await self._post("/api/kb/store", body)
        action: Literal["created", "updated"] = data["action"]
        entry = _parse_entry(data["entry"])
        raw_ids = data.get("superseded_ids")
        superseded_ids = list(raw_ids) if raw_ids is not None else None
        return action, entry, superseded_ids

    async def deactivate(
        self,
        entry_id: str,
        *,
        change_reason: str | None = None,
        superseded_by: str | None = None,
    ) -> KnowledgeEntry:
        """POST /api/kb/entries/{id}/deactivate.  Returns the deactivated entry.

        The body carries ``change_reason`` (required by the server) and the
        optional ``superseded_by``; keys whose value is ``None`` are omitted.
        """
        body: dict[str, Any] = {}
        if change_reason is not None:
            body["change_reason"] = change_reason
        if superseded_by is not None:
            body["superseded_by"] = superseded_by
        data = await self._post(f"/api/kb/entries/{entry_id}/deactivate", body)
        return _parse_entry(data["entry"])

    async def reactivate(self, entry_id: str) -> KnowledgeEntry:
        """POST /api/kb/entries/{id}/reactivate.  Returns the reactivated entry."""
        data = await self._post(f"/api/kb/entries/{entry_id}/reactivate", {})
        return _parse_entry(data["entry"])

    async def store_batch(
        self,
        entries: list[dict[str, Any]],
    ) -> tuple[list[KnowledgeEntry], list[tuple[int, str, str]]]:
        """Send all pre-validated entries to the service in one request."""
        if not entries:
            return [], []

        # Build the request body — no contributor/team (service attributes from API key)
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

        data = await self._post("/api/kb/store_batch", {"entries": batch_entries})
        created = [_parse_entry(e) for e in data.get("created", [])]
        # HTTP backend: no per-entry detail for server-side failures
        return created, []

    async def bulk_update(
        self,
        filters: dict[str, Any],
        updates: dict[str, Any],
        dry_run: bool,
    ) -> list[tuple[KnowledgeEntry, KnowledgeEntry]]:
        """POST /api/kb/bulk_update.  Returns (before, after) entry pairs."""
        data = await self._post(
            "/api/kb/bulk_update",
            {"filters": filters, "updates": updates, "dry_run": dry_run},
        )
        return [
            (_parse_entry(r["before"]), _parse_entry(r["after"])) for r in data.get("results", [])
        ]

    async def feedback(
        self,
        feedback_type: str,
        tool_name: str | None = None,
        query_or_params: str | None = None,
        detail: str | None = None,
    ) -> None:
        """POST /api/kb/feedback.  Records agent feedback on the service side."""
        body: dict[str, Any] = {"feedback_type": feedback_type}
        if tool_name is not None:
            body["tool_name"] = tool_name
        if query_or_params is not None:
            body["query_or_params"] = query_or_params
        if detail is not None:
            body["detail"] = detail
        await self._post("/api/kb/feedback", body)

    # -- Ask / Summarize ---------------------------------------------------

    async def ask_auto(
        self,
        question: str,
        scope: str | None,
        include_graph_context: bool,
        limit: int,
    ) -> tuple[list[tuple[KnowledgeEntry, str]], int]:
        """POST /api/kb/ask.  Returns (entries_with_context, agent_turns_used)."""
        body: dict[str, Any] = {
            "question": question,
            "include_graph_context": include_graph_context,
            "limit": limit,
        }
        if scope is not None:
            body["scope"] = scope

        data = await self._post("/api/kb/ask", body, timeout=_LONG_TIMEOUT)
        entries_with_context = [
            (_parse_entry(item["entry"]), item.get("context", ""))
            for item in data.get("entries", [])
        ]
        agent_turns_used = data.get("agent_turns_used", 0)
        return entries_with_context, agent_turns_used

    async def summarize(
        self,
        question: str,
        scope: str | None,
        limit: int,
    ) -> str:
        """POST /api/kb/summarize.  Returns a synthesised natural-language answer."""
        body: dict[str, Any] = {"question": question, "limit": limit}
        if scope is not None:
            body["scope"] = scope
        data = await self._post("/api/kb/summarize", body, timeout=_LONG_TIMEOUT)
        return cast("str", data.get("answer", ""))

    # -- Graph traversal ---------------------------------------------------

    async def supersedes_chain(self, entry_id: str) -> list[str]:
        """GET /api/kb/graph/supersedes-chain.  Returns chain oldest-first."""
        data = await self._get("/api/kb/graph/supersedes-chain", {"entry_id": entry_id})
        return cast("list[str]", data.get("chain", []))

    async def bfs_entries(
        self,
        start: str,
        max_depth: int,
        limit: int,
    ) -> list[tuple[str, int, list[str]]]:
        """GET /api/kb/graph/bfs.  Returns (entry_id, depth, path) tuples."""
        params: dict[str, Any] = {"start_node": start, "max_depth": max_depth, "limit": limit}
        data = await self._get("/api/kb/graph/bfs", params)
        return [
            (item["entry_id"], item["depth"], item.get("path", []))
            for item in data.get("entries", [])
        ]

    async def find_path(
        self,
        source: str,
        target: str,
        max_depth: int,
    ) -> list[tuple[str, str, str]] | None:
        """GET /api/kb/graph/path.  Returns hop list or None when no path exists."""
        params: dict[str, Any] = {"source": source, "target": target, "max_depth": max_depth}
        data = await self._get("/api/kb/graph/path", params)
        if not data.get("found", False) and data.get("hops") == []:
            # found=false,hops=[] means no path
            return None
        hops = data.get("hops", [])
        return [(h["source"], h["edge_type"], h["target"]) for h in hops]

    async def entries_for_scope(
        self,
        scope: str,
        entry_type: str | None = None,
        order_by: str = "updated_at",
    ) -> list[str]:
        """GET /api/kb/graph/scope-entries.  Returns entry IDs for the given scope."""
        params: dict[str, Any] = {"scope": scope}
        if entry_type is not None:
            params["entry_type"] = entry_type
        if order_by != "updated_at":
            params["order_by"] = order_by
        data = await self._get("/api/kb/graph/scope-entries", params)
        return cast("list[str]", data.get("entry_ids", []))

    async def decision_search(self, query: str, limit: int) -> list[str]:
        """POST /api/kb/search filtered to entry_type='decision'.  Returns IDs."""
        body: dict[str, Any] = {
            "query": query,
            "entry_type": "decision",
            "limit": limit,
        }
        data = await self._post("/api/kb/search", body)
        return [item["entry"]["id"] for item in data.get("results", [])]

    async def neighbors(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        direction: str = "both",
        limit: int = 10,
    ) -> list[tuple[str, str, str]]:
        """GET /api/kb/graph/neighbors.  Returns (neighbor_id, edge_type, direction) tuples."""
        params: dict[str, Any] = {
            "node_id": node_id,
            "direction": direction,
            "limit": limit,
        }
        if edge_types:
            # httpx sends repeated params as a list
            params["edge_types"] = edge_types
        data = await self._get("/api/kb/graph/neighbors", params)
        return [
            (n["neighbor_id"], n["edge_type"], n["direction"]) for n in data.get("neighbors", [])
        ]

    # -- Preflight ---------------------------------------------------------

    async def preflight(self, project_ref: str, since: str | None) -> str:
        """GET /api/kb/preflight.  Returns the context primer string."""
        params: dict[str, Any] = {"project_ref": project_ref}
        if since is not None:
            params["since"] = since
        data = await self._get("/api/kb/preflight", params)
        return cast("str", data.get("context", ""))

    # -- Ingest ------------------------------------------------------------

    async def ingest_file(
        self,
        path: Path,
        project_ref: str | None,
        dry_run: bool,
    ) -> FileResult:
        """POST /api/kb/ingest/file via multipart upload."""
        files = {"file": (path.name, path.read_bytes())}
        form_data: dict[str, Any] = {"dry_run": str(dry_run).lower()}
        if project_ref is not None:
            form_data["project_ref"] = project_ref

        try:
            resp = await self._c().post(
                "/api/kb/ingest/file",
                files=files,
                data=form_data,
                timeout=_LONG_TIMEOUT,
            )
            _raise_for_status(resp, self._base_url)
            return _parse_file_result(resp.json())
        except httpx.ConnectError as exc:
            msg = f"cannot reach KB service at {self._base_url}: {exc}"
            raise BackendHttpError(0, msg) from exc
        except httpx.TimeoutException as exc:
            msg = f"cannot reach KB service at {self._base_url}: {exc}"
            raise BackendHttpError(0, msg) from exc

    async def ingest_url(
        self,
        url: str,
        content: str | None,
        project_ref: str | None,
        dry_run: bool,
    ) -> FileResult:
        """POST /api/kb/ingest/url with optional pre-fetched content."""
        body: dict[str, Any] = {"url": url, "dry_run": dry_run}
        if content is not None:
            body["content"] = content
        if project_ref is not None:
            body["project_ref"] = project_ref
        data = await self._post("/api/kb/ingest/url", body, timeout=_LONG_TIMEOUT)
        return _parse_file_result(data)

    # -- Discovery ---------------------------------------------------------

    async def list_projects(self) -> list[tuple[str, int]]:
        """GET /api/kb/projects.  Returns ``(name, entry_count)`` pairs."""
        data = await self._get("/api/kb/projects")
        return [(item["name"], item["entry_count"]) for item in data.get("items", [])]

    async def list_contributors(self) -> list[tuple[str, int]]:
        """GET /api/kb/contributors.  Returns ``(name, entry_count)`` pairs."""
        data = await self._get("/api/kb/contributors")
        return [(item["name"], item["entry_count"]) for item in data.get("items", [])]

    async def list_teams(self) -> list[tuple[str, int]]:
        """GET /api/kb/teams.  Returns ``(name, entry_count)`` pairs."""
        data = await self._get("/api/kb/teams")
        return [(item["name"], item["entry_count"]) for item in data.get("items", [])]

    # -- Map eligibility ---------------------------------------------------
    #
    # Verdict dicts cross this layer UNPARSED — no _parse_entry-style
    # reconstruction, no kb_core.map_eligibility import, no TypedDict.

    async def map_eligibility(self) -> list[dict[str, Any]]:
        """GET /api/kb/map-eligibility.  Returns the ``projects`` verdict list."""
        data = await self._get("/api/kb/map-eligibility")
        return cast("list[dict[str, Any]]", data["projects"])

    async def set_map_eligibility_override(
        self,
        project_ref: str,
        *,
        eligible: bool,
        reason: str,
    ) -> dict[str, Any]:
        """POST /api/kb/map-eligibility/override.  Returns ``{changed, verdict}``."""
        data = await self._post(
            "/api/kb/map-eligibility/override",
            {"project_ref": project_ref, "eligible": eligible, "reason": reason},
        )
        return {"changed": bool(data["changed"]), "verdict": data["verdict"]}

    async def clear_map_eligibility_override(self, project_ref: str) -> bool:
        """POST /api/kb/map-eligibility/override/clear.  Returns the ``changed`` flag.

        The clear is a POST with the ``project_ref`` in the JSON body, so no
        project_ref is ever interpolated into a URL path.  Because neither
        write endpoint ever 404s for an absent row, a 404 reaching the caller
        can only mean the endpoint does not exist.
        """
        data = await self._post(
            "/api/kb/map-eligibility/override/clear",
            {"project_ref": project_ref},
        )
        return bool(data["changed"])
