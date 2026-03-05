"""Multi-turn chat session grounded in KB data."""

import logging
import uuid
from typing import Any

from personal_kb.db.backend import Database
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
- You can use the retrieve tool to search for more entries when needed.
- On follow-up questions, use context from the conversation history.\
"""


class ChatSession:
    """Manages a multi-turn conversation grounded in KB data."""

    def __init__(  # noqa: D107
        self,
        db: Database,
        embedder: EmbeddingClient | None,
        llm: LLMProvider,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.db = db
        self.embedder = embedder
        self.llm = llm
        self.messages: list[Message] = []
        # Entry IDs surfaced so far — available as context
        self.entry_ids: list[str] = []

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
            await event_callback(
                {
                    "type": "chat_done",
                    "new_entries": new_entries,
                }
            )

        return response

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

        blocks = []
        for eid in self.entry_ids:
            entry = await get_entry(self.db, eid)
            if entry and entry.is_active:
                tags = " ".join(f"#{t}" for t in entry.tags) if entry.tags else ""
                block = f"[{entry.id}] {entry.short_title} {tags}"
                block += f"\n  {entry.knowledge_details}"
                blocks.append(block)
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
) -> ChatSession:
    """Get an existing session or create a new one."""
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    session = ChatSession(db, embedder, llm)
    _sessions[session.id] = session
    return session


def get_session(session_id: str) -> ChatSession | None:
    """Get a session by ID, or None if not found."""
    return _sessions.get(session_id)
