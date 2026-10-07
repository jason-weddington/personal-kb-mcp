"""Chat session: per-user conversation state and tool dispatch.

Ports the old explorer's ``ChatSession`` (web/chat.py) into the service,
wiring it to the singleton ``KnowledgeBase`` via per-request ``Attribution``
objects so write-backs carry the correct contributor/team.

Module-level session registry (``_sessions``) keeps recently-used ChatSession
objects in memory keyed by ``(user_id, session_id)`` — no eviction or TTL
(home-lab MVP).  Use ``get_session`` and ``cache_session`` rather than
accessing ``_sessions`` directly.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from kb_core import Attribution

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

_MAX_CONVERSATION_CHARS: int = 100_000
_VALID_SENSITIVITY: set[str] = {"internal", "restricted", "public"}
_KB_ID_RE: re.Pattern[str] = re.compile(r"kb-\d{5}")
_FENCE_RE: re.Pattern[str] = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# ── system prompts (verbatim from old explorer web/chat.py) ───────────────────

_CHAT_SYSTEM_PROMPT: str = (
    "You are a knowledge base assistant. You answer questions grounded in KB entries.\n"
    "\n"
    "Rules:\n"
    "- Answer ONLY from the provided KB entries. Do not use outside knowledge.\n"
    "- Cite entry IDs in [kb-XXXXX] format when referencing specific entries.\n"
    "- If entries contain conflicting information, note the conflict and cite both.\n"
    "- Be concise. Prefer bullet points for multi-part answers.\n"
    "- On follow-up questions, use context from the conversation history.\n"
    "\n"
    "You have a read tool. To fetch a specific entry by ID, output a JSON block:\n"
    "```json\n"
    '{"tool": "get_entry", "args": {"entry_id": "kb-00042"}}\n'
    "```\n"
    "\n"
    "Available read tools:\n"
    "- get_entry: Fetch a KB entry by ID and add it to context."
    " Args: entry_id (required)\n"
    "\n"
    "Use get_entry when a user references a specific entry ID that is not in"
    " your context."
)

_WRITE_TOOLS_PROMPT: str = (
    "\n"
    "\n"
    "You also have write tools. To use one, output a JSON block:\n"
    "```json\n"
    '{"tool": "update_entry", "args": {"entry_id": "kb-00042",'
    ' "tags": ["postgres", "migration"]}}\n'
    "```\n"
    "\n"
    "Available tools:\n"
    "- update_entry: Update an existing KB entry."
    " Args: entry_id (required), knowledge_details, tags, project_ref,"
    " sensitivity, ttl; change_reason (required: why the entry is changed)\n"
    "- ingest_url: Ingest a URL into the KB. Args: url (required), project_ref\n"
    "\n"
    "Rules:\n"
    "- Only use a tool when the user explicitly asks for a write operation\n"
    "- Never create new entries — only update existing ones\n"
    "- Always confirm what you did after the tool completes"
)


# ── helpers ───────────────────────────────────────────────────────────────────


@dataclass
class _ToolResult:
    """Result of a single tool dispatch."""

    tool: str
    success: bool
    message: str
    entry_ids: list[str] = field(default_factory=list)


def _parse_tool_call(text: str) -> dict[str, Any] | None:
    """Extract a tool-call JSON object from *text*.

    Iterates fenced code blocks (```json ... ```) via
    ``kb_core.llm.json_parser.parse_json_object``, keeping the first dict
    containing the key ``'tool'``.  Falls back to bare-JSON parse of the whole
    text.  Returns ``None`` when no valid tool call is found.
    """
    from kb_core.llm.json_parser import parse_json_object

    for match in _FENCE_RE.finditer(text):
        block = match.group(1)
        parsed = parse_json_object(block)
        if isinstance(parsed, dict) and "tool" in parsed:
            return parsed

    # Fallback: try the whole text as bare JSON
    parsed_bare = parse_json_object(text)
    if isinstance(parsed_bare, dict) and "tool" in parsed_bare:
        return parsed_bare

    return None


def _trim_history(messages: list[dict[str, Any]]) -> None:
    """Trim conversation history in-place to stay under the char budget.

    No-op when ``total content chars <= 100_000`` OR ``len(messages) <= 3``.
    Otherwise keeps ``messages[:2]`` (seed Q+A) and removes messages from the
    front of the remainder until the total drops to ``<= 100_000``.
    """
    total = sum(len(str(m.get("content", ""))) for m in messages)
    if total <= _MAX_CONVERSATION_CHARS or len(messages) <= 3:
        return
    seed = messages[:2]
    rest = list(messages[2:])
    while rest:
        total = sum(len(str(m.get("content", ""))) for m in seed) + sum(
            len(str(m.get("content", ""))) for m in rest
        )
        if total <= _MAX_CONVERSATION_CHARS:
            break
        rest.pop(0)
    messages.clear()
    messages.extend(seed)
    messages.extend(rest)


# ── per-user session registry ─────────────────────────────────────────────────

_sessions: dict[tuple[str, str], ChatSession] = {}


def get_session(user_id: str, session_id: str) -> ChatSession | None:
    """Look up a cached ``ChatSession`` by *(user_id, session_id)*."""
    return _sessions.get((user_id, session_id))


def cache_session(session: ChatSession) -> None:
    """Store *session* in the registry keyed by *(session.user_id, session.id)*."""
    _sessions[(session.user_id, session.id)] = session


# ── ChatSession ───────────────────────────────────────────────────────────────


class ChatSession:
    """A single user's conversation session with the knowledge base.

    Holds the in-memory message history and the set of KB entry IDs that have
    been pulled into context.  Write-backs (update_entry, ingest_url) carry
    per-request ``Attribution`` so contributor/team are never stale.
    """

    def __init__(
        self,
        kb: Any,
        llm: Any,
        attribution: Attribution,
        user_id: str,
        session_id: str | None = None,
    ) -> None:
        """Initialise a new or restored chat session.

        Args:
            kb: The ``KnowledgeBase`` singleton from ``app.state.kb``.
            llm: The LLM provider (``kb.synthesis_llm`` or ``kb.query_llm``).
            attribution: Per-request attribution (contributor + team).
            user_id: ID of the owning user.
            session_id: Optional existing session ID; a UUID is generated if absent.
        """
        self.kb = kb
        self.llm = llm
        self.attribution = attribution
        self.user_id = user_id
        self.id = session_id or str(uuid.uuid4())
        self.messages: list[dict[str, Any]] = []
        self.entry_ids: list[str] = []

    @classmethod
    def from_saved(
        cls,
        chat_id: str,
        messages: list[dict[str, str]],
        kb: Any,
        llm: Any,
        attribution: Attribution,
        user_id: str,
    ) -> ChatSession:
        """Restore a ``ChatSession`` from persisted messages.

        Rebuilds the message list and re-extracts unique entry IDs from
        assistant messages via ``_KB_ID_RE``.

        Args:
            chat_id: The persisted chat session ID.
            messages: Saved ``[{'role': ..., 'content': ...}]`` rows.
            kb: The ``KnowledgeBase`` singleton.
            llm: The LLM provider.
            attribution: Per-request attribution.
            user_id: ID of the owning user.
        """
        session = cls(kb, llm, attribution, user_id, session_id=chat_id)
        session.messages = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        seen: set[str] = set()
        for m in session.messages:
            if m["role"] == "assistant":
                for eid in _KB_ID_RE.findall(str(m["content"])):
                    if eid not in seen:
                        seen.add(eid)
                        session.entry_ids.append(eid)
        return session

    def seed(self, question: str, answer: str, entry_ids: list[str]) -> None:
        """Seed the conversation with a pre-existing Q&A pair.

        Sets ``messages`` to a single user/assistant turn and initialises
        ``entry_ids`` from the provided list.

        Args:
            question: The seed user question.
            answer: The seed assistant answer.
            entry_ids: Entry IDs that appeared in the seed answer.
        """
        self.messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        self.entry_ids = list(entry_ids)

    async def _retrieve_context(self, user_message: str) -> list[str]:
        """Search for relevant entries and return the newly-added IDs.

        Args:
            user_message: The user's latest message (used as the search query).

        Returns:
            List of entry IDs that were not previously in ``self.entry_ids``.
        """
        from kb_core.models.search import SearchQuery

        results, _ = await self.kb.search(
            SearchQuery(query=user_message, limit=5, include_stale=False)
        )
        new_ids = [r.entry.id for r in results if r.entry.id not in self.entry_ids]
        self.entry_ids.extend(new_ids)
        return new_ids

    async def _dispatch_tool(self, tool_call: dict[str, Any]) -> _ToolResult:
        """Dispatch a parsed tool call and return its result.

        Supports exactly three tools: ``get_entry``, ``update_entry``,
        ``ingest_url``.  Any other ``tool`` name returns a failure result.
        """
        tool_name: str = str(tool_call.get("tool", ""))
        args: dict[str, Any] = dict(tool_call.get("args", {}))

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
        entry_id = args.get("entry_id")
        if not entry_id:
            return _ToolResult(
                tool="get_entry",
                success=False,
                message="entry_id is required",
            )
        entry = await self.kb.get(str(entry_id))
        if entry is None:
            return _ToolResult(
                tool="get_entry",
                success=False,
                message=f"Entry {entry_id} not found",
            )
        if str(entry_id) not in self.entry_ids:
            self.entry_ids.append(str(entry_id))
        tags = " ".join(f"#{t}" for t in entry.tags) if entry.tags else ""
        entry_type_val = entry.entry_type.value if entry.entry_type else "unknown"
        message = (
            f"[{entry.id}] {entry.short_title} {tags}\n"
            f"Type: {entry_type_val}\n"
            f"{entry.knowledge_details}"
        )
        return _ToolResult(
            tool="get_entry",
            success=True,
            message=message,
            entry_ids=[entry.id],
        )

    async def _tool_update_entry(self, args: dict[str, Any]) -> _ToolResult:
        """Update an existing KB entry."""
        entry_id = args.get("entry_id")
        if not entry_id:
            return _ToolResult(
                tool="update_entry",
                success=False,
                message="entry_id is required",
            )

        change_reason = args.get("change_reason")
        if not isinstance(change_reason, str) or not change_reason.strip():
            return _ToolResult(
                tool="update_entry",
                success=False,
                message="change_reason is required for update_entry",
            )

        sensitivity = args.get("sensitivity")
        if sensitivity is not None and sensitivity not in _VALID_SENSITIVITY:
            sorted_valid = ", ".join(sorted(_VALID_SENSITIVITY))
            return _ToolResult(
                tool="update_entry",
                success=False,
                message=(
                    f'Error: Invalid sensitivity "{sensitivity}".'
                    f" Must be one of: {sorted_valid}"
                ),
            )

        knowledge_details = args.get("knowledge_details")
        if knowledge_details is not None and not self.kb.config.ingest.skip_safety:
            from kb_core.ingest.safety import detect_secrets_in_content

            secrets = detect_secrets_in_content(str(knowledge_details))
            if secrets:
                return _ToolResult(
                    tool="update_entry",
                    success=False,
                    message=(
                        f"Error: Potential secrets detected ({', '.join(secrets)})."
                        " Remove sensitive values before storing."
                        " Set KB_SKIP_SAFETY=TRUE to override."
                    ),
                )

        expires_at: Any = None
        ttl = args.get("ttl")
        if ttl is not None:
            try:
                from kb_core.ttl import compute_expires_at

                expires_at = compute_expires_at(ttl)
            except ValueError as exc:
                return _ToolResult(tool="update_entry", success=False, message=str(exc))

        tags = args.get("tags")
        if tags is not None and not isinstance(tags, list):
            tags = [str(tags)]

        try:
            entry = await self.kb.update(
                str(entry_id),
                knowledge_details=knowledge_details,
                change_reason=change_reason,
                tags=tags,
                updated_by=self.attribution.contributor,
                sensitivity=sensitivity,
                expires_at=expires_at,
                project_ref=args.get("project_ref"),
            )
        except (ValueError, KeyError) as exc:
            return _ToolResult(tool="update_entry", success=False, message=str(exc))

        return _ToolResult(
            tool="update_entry",
            success=True,
            message=f"Updated {entry.id}: {entry.short_title}",
            entry_ids=[entry.id],
        )

    async def _tool_ingest_url(self, args: dict[str, Any]) -> _ToolResult:
        """Ingest a URL into the KB."""
        url = args.get("url")
        if not url:
            return _ToolResult(
                tool="ingest_url", success=False, message="url is required"
            )
        if self.kb.extraction_llm is None:
            return _ToolResult(
                tool="ingest_url",
                success=False,
                message="Extraction LLM not available",
            )
        if self.kb.embedder is None:
            return _ToolResult(
                tool="ingest_url", success=False, message="Embedder not available"
            )
        try:
            result = await self.kb.ingest_url(
                str(url),
                project_ref=args.get("project_ref"),
                contributor=self.attribution.contributor,
                team=self.attribution.team,
            )
        except RuntimeError as exc:
            return _ToolResult(tool="ingest_url", success=False, message=str(exc))
        if result.action in ("skipped", "error"):
            return _ToolResult(
                tool="ingest_url",
                success=False,
                message=f"{result.action}: {result.reason}",
            )
        entry_ids: list[str] = result.entry_ids or []
        message = (
            f"Ingested {url}: {len(entry_ids)} entries created ({', '.join(entry_ids)})"
        )
        return _ToolResult(
            tool="ingest_url",
            success=True,
            message=message,
            entry_ids=entry_ids,
        )

    async def reply(
        self,
        user_message: str,
        event_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> str:
        """Generate a reply to *user_message*, optionally streaming events.

        Appends the user message, retrieves relevant context, calls the LLM,
        and dispatches any tool calls before returning the final answer.

        Args:
            user_message: The user's latest message text.
            event_callback: Optional async callable that receives progress event
                dicts.  Called with ``chat_thinking``, ``chat_tool_result``, and
                ``chat_done`` events.

        Returns:
            The assistant's reply text.
        """
        import kb_core.formatting

        # 1. Append user message
        self.messages.append({"role": "user", "content": user_message})

        # 2. Trim history
        _trim_history(self.messages)

        # 3. Retrieve context
        new_entries = await self._retrieve_context(user_message)

        # 4. Build system prompt
        entries_text_parts: list[str] = []
        for eid in self.entry_ids:
            e = await self.kb.get(eid)
            if e is not None and e.is_active:
                entries_text_parts.append(kb_core.formatting.format_entry_full(e))

        system = _CHAT_SYSTEM_PROMPT + _WRITE_TOOLS_PROMPT
        if entries_text_parts:
            system += "\n\nAvailable KB entries:\n" + "\n\n".join(entries_text_parts)

        # 5. Emit thinking event
        if event_callback is not None:
            await event_callback({"type": "chat_thinking"})

        # 6. First LLM call
        response: str | None = await self.llm.generate_chat(
            self.messages, system=system
        )

        if response is None:
            response = (
                "Sorry, I couldn't generate a response. The LLM may be unavailable."
            )
            self.messages.append({"role": "assistant", "content": response})
            if event_callback is not None:
                await event_callback({"type": "chat_done", "new_entries": new_entries})
            return response

        # 7. Parse tool call
        tool_call = _parse_tool_call(response)

        if tool_call is not None:
            # Append the tool-call message as assistant turn
            self.messages.append({"role": "assistant", "content": response})

            # Dispatch the tool
            result = await self._dispatch_tool(tool_call)

            # Emit tool result event
            if event_callback is not None:
                await event_callback(
                    {
                        "type": "chat_tool_result",
                        "tool": tool_call.get("tool", ""),
                        "success": result.success,
                        "entry_ids": result.entry_ids,
                    }
                )

            # Collect newly-referenced entry IDs
            new_entries.extend(result.entry_ids)

            # Append tool result as a user message
            self.messages.append(
                {"role": "user", "content": f"Tool result: {result.message}"}
            )

            # Second LLM call for the final answer
            response2: str | None = await self.llm.generate_chat(
                self.messages, system=system
            )
            final = response2 if response2 is not None else result.message
            self.messages.append({"role": "assistant", "content": final})
            if event_callback is not None:
                await event_callback({"type": "chat_done", "new_entries": new_entries})
            return final

        # No tool call — regular response
        self.messages.append({"role": "assistant", "content": response})
        if event_callback is not None:
            await event_callback({"type": "chat_done", "new_entries": new_entries})
        return response
