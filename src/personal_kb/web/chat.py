"""Multi-turn chat session grounded in KB data."""

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from personal_kb.db.backend import Database
from personal_kb.llm.json_parser import parse_json_object
from personal_kb.llm.provider import LLMProvider, Message
from personal_kb.search.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)

# Approximate token budget: keep total conversation under this char count.
# ~25K tokens ≈ 100K chars. Leaves headroom for the LLM's context window.
_MAX_CONVERSATION_CHARS = 100_000

_CHAT_SYSTEM_PROMPT = """\
You are a knowledge base assistant. You answer questions grounded in KB entries.

Rules:
- Answer ONLY from the provided KB entries. Do not use outside knowledge.
- Cite entry IDs in [kb-XXXXX] format when referencing specific entries.
- If entries contain conflicting information, note the conflict and cite both.
- Be concise. Prefer bullet points for multi-part answers.
- On follow-up questions, use context from the conversation history.

You have a read tool. To fetch a specific entry by ID, output a JSON block:
```json
{"tool": "get_entry", "args": {"entry_id": "kb-00042"}}
```

Available read tools:
- get_entry: Fetch a KB entry by ID and add it to context. Args: entry_id (required)

Use get_entry when a user references a specific entry ID that is not in your context.\
"""

_WRITE_TOOLS_PROMPT = """

You also have write tools. To use one, output a JSON block:
```json
{"tool": "update_entry", "args": {"entry_id": "kb-00042", "tags": ["postgres", "migration"]}}
```

Available tools:
- update_entry: Update an existing KB entry. Args: entry_id (required), knowledge_details, \
tags, project_ref, sensitivity, ttl, change_reason
- ingest_url: Ingest a URL into the KB. Args: url (required), project_ref

Rules:
- Only use a tool when the user explicitly asks for a write operation
- Never create new entries — only update existing ones
- Always confirm what you did after the tool completes"""

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass
class WriteDeps:
    """Dependencies needed for write operations in the chat session."""

    store: Any  # KnowledgeStore
    graph_builder: Any  # GraphBuilder
    graph_enricher: Any | None  # GraphEnricher | None
    extraction_llm: Any | None  # LLMProvider | None
    contributor: str | None = None
    team: str | None = None


@dataclass
class _ToolResult:
    tool: str
    success: bool
    message: str
    entry_ids: list[str] = field(default_factory=list)


def _parse_tool_call(text: str) -> dict[str, Any] | None:
    """Extract a tool call JSON from LLM response text.

    Returns dict with 'tool' and 'args' keys, or None if no tool call found.
    """
    # Try fenced code blocks first — may contain non-tool JSON, so iterate
    for m in _FENCE_RE.finditer(text):
        parsed = parse_json_object(m.group(1))
        if parsed is not None and "tool" in parsed:
            return parsed

    # Fallback: bare JSON object in text (only if it has "tool" key)
    parsed = parse_json_object(text)
    if parsed is not None and "tool" in parsed:
        return parsed

    return None


_KB_ID_RE = re.compile(r"kb-\d{5}")


class ChatSession:
    """Manages a multi-turn conversation grounded in KB data."""

    def __init__(  # noqa: D107
        self,
        db: Database,
        embedder: EmbeddingClient | None,
        llm: LLMProvider,
        write_deps: WriteDeps | None = None,
        session_id: str | None = None,
    ) -> None:
        self.id = session_id or str(uuid.uuid4())
        self.db = db
        self.embedder = embedder
        self.llm = llm
        self.write_deps = write_deps
        self.messages: list[Message] = []
        # Entry IDs surfaced so far — available as context
        self.entry_ids: list[str] = []

    @classmethod
    def from_saved(
        cls,
        chat_id: str,
        messages: list[dict[str, str]],
        db: Database,
        embedder: EmbeddingClient | None,
        llm: LLMProvider,
        write_deps: WriteDeps | None = None,
    ) -> "ChatSession":
        """Reconstruct a session from persisted messages."""
        session = cls(db, embedder, llm, write_deps=write_deps, session_id=chat_id)
        session.messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        # Extract entry IDs mentioned in assistant messages
        for m in session.messages:
            if m["role"] == "assistant":
                for match in _KB_ID_RE.finditer(m["content"]):
                    eid = match.group(0)
                    if eid not in session.entry_ids:
                        session.entry_ids.append(eid)
        return session

    def seed(self, question: str, answer: str, entry_ids: list[str]) -> None:
        """Seed the conversation with the initial Q+A from summarize."""
        self.messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        self.entry_ids = list(entry_ids)

    async def reply(
        self,
        user_message: str,
        event_callback: Any | None = None,
    ) -> str:
        """Process a user follow-up and return the assistant's response."""
        self.messages.append({"role": "user", "content": user_message})
        self._trim_history()

        # Search for additional context relevant to the follow-up
        new_entries = await self._retrieve_context(user_message)

        # Build the system prompt with grounding entries
        system = _CHAT_SYSTEM_PROMPT
        if self.write_deps is not None:
            system += _WRITE_TOOLS_PROMPT
        if self.entry_ids:
            entries_context = await self._format_entries()
            system += "\n\nAvailable KB entries:\n" + entries_context

        if event_callback:
            await event_callback({"type": "chat_thinking"})

        response = await self.llm.generate_chat(
            self.messages,
            system=system,
        )

        if response is None:
            response = "Sorry, I couldn't generate a response. The LLM may be unavailable."
            self.messages.append({"role": "assistant", "content": response})
            if event_callback:
                await event_callback({"type": "chat_done", "new_entries": new_entries})
            return response

        # Check for tool call in the response
        tool_call = _parse_tool_call(response)

        if tool_call is not None:
            # Execute the tool
            self.messages.append({"role": "assistant", "content": response})
            result = await self._dispatch_tool(tool_call)

            if event_callback:
                event_type = "chat_tool_result"
                await event_callback(
                    {
                        "type": event_type,
                        "tool": tool_call.get("tool", ""),
                        "success": result.success,
                        "entry_ids": result.entry_ids,
                    }
                )
                new_entries.extend(result.entry_ids)

            # Inject result and get final response
            self.messages.append({"role": "user", "content": f"Tool result: {result.message}"})
            final_response = await self.llm.generate_chat(
                self.messages,
                system=system,
            )
            if final_response is None:
                final_response = result.message
            self.messages.append({"role": "assistant", "content": final_response})

            if event_callback:
                await event_callback({"type": "chat_done", "new_entries": new_entries})
            return final_response

        # No tool call — standard passthrough
        self.messages.append({"role": "assistant", "content": response})

        if event_callback:
            await event_callback({"type": "chat_done", "new_entries": new_entries})

        return response

    async def _dispatch_tool(self, tool_call: dict[str, Any]) -> _ToolResult:
        """Dispatch a tool call and return the result."""
        tool_name = tool_call.get("tool", "")
        args = tool_call.get("args", {})

        if tool_name == "get_entry":
            return await self._tool_get_entry(args)
        if tool_name == "update_entry":
            return await self._tool_update_entry(args)
        if tool_name == "ingest_url":
            return await self._tool_ingest_url(args)

        return _ToolResult(
            tool=tool_name,
            success=False,
            message=f"Unknown tool: {tool_name}",
        )

    async def _tool_get_entry(self, args: dict[str, Any]) -> _ToolResult:
        """Fetch a KB entry by ID and add it to context."""
        from personal_kb.db.queries import get_entry

        entry_id = args.get("entry_id")
        if not entry_id:
            return _ToolResult(tool="get_entry", success=False, message="entry_id is required")

        entry = await get_entry(self.db, entry_id)
        if entry is None:
            return _ToolResult(
                tool="get_entry", success=False, message=f"Entry {entry_id} not found"
            )

        # Add to context so it appears in system prompt on next LLM call
        if entry_id not in self.entry_ids:
            self.entry_ids.append(entry_id)

        tags = " ".join(f"#{t}" for t in entry.tags) if entry.tags else ""
        message = (
            f"[{entry.id}] {entry.short_title} {tags}\n"
            f"Type: {entry.entry_type.value if entry.entry_type else 'unknown'}\n"
            f"{entry.knowledge_details}"
        )
        return _ToolResult(tool="get_entry", success=True, message=message, entry_ids=[entry_id])

    async def _tool_update_entry(self, args: dict[str, Any]) -> _ToolResult:
        """Execute the update_entry tool."""
        if self.write_deps is None:
            return _ToolResult(tool="update_entry", success=False, message="Write not available")

        entry_id = args.get("entry_id")
        if not entry_id:
            return _ToolResult(tool="update_entry", success=False, message="entry_id is required")

        from personal_kb.tools.kb_store import (
            _build_graph,
            _check_secrets,
            _embed_entry,
            _enrich_graph,
            _validate_sensitivity,
        )
        from personal_kb.tools.ttl import compute_expires_at

        # Validate sensitivity
        sensitivity = args.get("sensitivity")
        sens_err = _validate_sensitivity(sensitivity)
        if sens_err:
            return _ToolResult(tool="update_entry", success=False, message=sens_err)

        # Secret scanning on content if provided
        knowledge_details = args.get("knowledge_details")
        if knowledge_details:
            secret_err = _check_secrets(knowledge_details)
            if secret_err:
                return _ToolResult(tool="update_entry", success=False, message=secret_err)

        # Compute expires_at from TTL
        expires_at = None
        ttl = args.get("ttl")
        if ttl:
            try:
                expires_at = compute_expires_at(ttl)
            except ValueError as e:
                return _ToolResult(tool="update_entry", success=False, message=str(e))

        # Normalize tags
        tags = args.get("tags")
        if tags is not None and not isinstance(tags, list):
            tags = [str(tags)]

        try:
            store = self.write_deps.store
            entry = await store.update_entry(
                entry_id=entry_id,
                knowledge_details=knowledge_details,
                change_reason=args.get("change_reason"),
                tags=tags,
                updated_by=self.write_deps.contributor,
                sensitivity=sensitivity,
                expires_at=expires_at,
                project_ref=args.get("project_ref"),
            )
        except (ValueError, KeyError) as e:
            return _ToolResult(tool="update_entry", success=False, message=str(e))

        # Re-embed and rebuild graph
        if self.embedder:
            await _embed_entry(self.embedder, self.write_deps.store, entry)
        await _build_graph(self.write_deps.graph_builder, entry)
        await _enrich_graph(self.write_deps.graph_enricher, entry)

        return _ToolResult(
            tool="update_entry",
            success=True,
            message=f"Updated {entry.id}: {entry.short_title}",
            entry_ids=[entry.id],
        )

    async def _tool_ingest_url(self, args: dict[str, Any]) -> _ToolResult:
        """Execute the ingest_url tool."""
        if self.write_deps is None:
            return _ToolResult(tool="ingest_url", success=False, message="Write not available")

        url = args.get("url")
        if not url:
            return _ToolResult(tool="ingest_url", success=False, message="url is required")

        from personal_kb.ingest.ingester import FileIngester

        extraction_llm = self.write_deps.extraction_llm
        if extraction_llm is None:
            return _ToolResult(
                tool="ingest_url", success=False, message="Extraction LLM not available"
            )
        if self.embedder is None:
            return _ToolResult(tool="ingest_url", success=False, message="Embedder not available")

        ingester = FileIngester(
            db=self.db,
            store=self.write_deps.store,
            embedder=self.embedder,
            llm=extraction_llm,
            graph_builder=self.write_deps.graph_builder,
            graph_enricher=self.write_deps.graph_enricher,
            contributor=self.write_deps.contributor,
            team=self.write_deps.team,
        )

        result = await ingester.ingest_url(
            url,
            project_ref=args.get("project_ref"),
        )

        if result.action in ("skipped", "error"):
            return _ToolResult(
                tool="ingest_url",
                success=False,
                message=f"{result.action}: {result.reason}",
            )

        entry_ids = result.entry_ids or []
        count = len(entry_ids)
        return _ToolResult(
            tool="ingest_url",
            success=True,
            message=f"Ingested {url}: {count} entries created ({', '.join(entry_ids)})",
            entry_ids=entry_ids,
        )

    async def _retrieve_context(self, query: str) -> list[str]:
        """Search for entries relevant to the follow-up question."""
        from personal_kb.models.search import SearchQuery
        from personal_kb.search.hybrid import hybrid_search

        sq = SearchQuery(query=query, limit=5, include_stale=False)
        results, _ = await hybrid_search(self.db, self.embedder, sq)

        new_ids = []
        for r in results:
            if r.entry.id not in self.entry_ids:
                self.entry_ids.append(r.entry.id)
                new_ids.append(r.entry.id)
        return new_ids

    async def _format_entries(self) -> str:
        """Format all known entries as context for the system prompt."""
        from personal_kb.db.queries import get_entry
        from personal_kb.tools.formatters import format_entry_full

        blocks = []
        for eid in self.entry_ids:
            entry = await get_entry(self.db, eid)
            if entry and entry.is_active:
                blocks.append(format_entry_full(entry))
        return "\n\n".join(blocks)

    def _trim_history(self) -> None:
        """Trim conversation history if it exceeds the token budget.

        Strategy: keep first turn (original Q+A for grounding) + last N turns.
        Drop middle turns.
        """
        total_chars = sum(len(m["content"]) for m in self.messages)
        if total_chars <= _MAX_CONVERSATION_CHARS:
            return

        # Always keep first 2 messages (seed Q+A) and the latest user message
        if len(self.messages) <= 3:
            return

        first_pair = self.messages[:2]
        rest = self.messages[2:]

        # Remove from the front of 'rest' until we fit
        while rest and total_chars > _MAX_CONVERSATION_CHARS:
            removed = rest.pop(0)
            total_chars -= len(removed["content"])

        self.messages = first_pair + rest


# In-memory session store (keyed by session ID)
_sessions: dict[str, ChatSession] = {}


def get_or_create_session(
    session_id: str | None,
    db: Database,
    embedder: EmbeddingClient | None,
    llm: LLMProvider,
    write_deps: WriteDeps | None = None,
) -> ChatSession:
    """Get an existing session or create a new one."""
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    session = ChatSession(db, embedder, llm, write_deps=write_deps)
    _sessions[session.id] = session
    return session


def cache_session(session: ChatSession) -> None:
    """Add a restored session to the in-memory cache."""
    _sessions[session.id] = session


def get_session(session_id: str) -> ChatSession | None:
    """Get a session by ID, or None if not found."""
    return _sessions.get(session_id)
