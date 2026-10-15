"""LLM provider protocol for pluggable language model backends."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

# Message dict: {"role": "user"|"assistant", "content": "..."}
Message = dict[str, str]

LLMPurpose = Literal[
    "agent",  # graph/agent.py ReAct loop
    "chat",  # kb_service/chat.py ChatSession
    "classifier",  # kb_service/classifier.py
    "coverage",  # kb_core/coverage.py
    "dedup_agent",  # ingest/dedup_agent.py
    "enricher",  # graph/enricher.py single entry, _SYSTEM_PROMPT
    "enricher_batch",  # graph/enricher.py batch, _BATCH_SYSTEM_PROMPT
    "feedback_summary",  # src/personal_kb/tools/kb_maintain.py
    "ingest_extract",  # ingest/extractor.py extract_entries
    "ingest_summarize",  # ingest/extractor.py summarize_file
    "listener_vote",  # routes/listener_routes.py three-vote gate
    "planner",  # graph/planner.py
    "query_synthesis",  # kb_core/query.py
    "seed_resolutions",  # packages/kb-service/scripts/seed_resolutions.py
    "store_critic",  # surprise_worker.py STORE_CRITIC_SYSTEM
    "store_distiller",  # surprise_worker.py STORE_DISTILLER_SYSTEM
    "surprise_critic",  # surprise_worker.py SURPRISE_CRITIC_SYSTEM
    "surprise_detector_shape2",  # surprise_worker._model_record, shape 2
    "surprise_detector_shape3",  # surprise_worker._model_record, shape 3
    "surprise_distiller",  # surprise_worker.py SURPRISE_DISTILLER_SYSTEM
]
"""Why an LLM call was made; a required keyword on every generate call."""

LLMCallOutcome = Literal["ok", "no_text", "error"]
"""ok: at least one text block; no_text: none (generate returns None); error: create raised."""

DiagnosticsState = Literal["sent", "off_config", "off_disabled", "fallback"]
"""Whether the cache-diagnostics object was sent on a recorded call."""


@dataclass(frozen=True)
class LLMCallRecord:
    """One recorded Claude API call: usage and cache diagnostics, by purpose."""

    ts: str
    purpose: LLMPurpose
    model: str
    outcome: LLMCallOutcome
    error_type: str | None
    response_id: str | None
    stop_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    cache_creation_input_tokens: int | None
    cache_creation_1h_input_tokens: int | None
    cache_read_input_tokens: int | None
    diagnostics_state: DiagnosticsState
    previous_message_id: str | None
    cache_miss_reason: str | None
    cache_missed_input_tokens: int | None
    diagnostics_raw: str | None
    system_sha256: str | None
    latency_ms: int
    thinking: str | None = None
    """Thinking mode the request carried (``disabled``/``adaptive``); None when none sent."""
    effort: str | None = None
    """Effort level the request carried (``output_config.effort``); None when none sent."""
    candidate_id: int | None = None
    """the surprise candidate a capture call served; set by kb-service, None elsewhere."""


LLMCallRecorder = Callable[[LLMCallRecord], None]
"""Synchronous sink for call records; it must not block the caller."""


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for language model providers with graceful degradation."""

    async def is_available(self) -> bool:
        """Check if the LLM backend is reachable."""
        ...

    async def generate(
        self,
        prompt: str,
        *,
        purpose: LLMPurpose,
        system: str | None = None,
    ) -> str | None:
        """Generate text from a prompt. Returns None if unavailable."""
        ...

    async def generate_chat(
        self,
        messages: list[Message],
        *,
        purpose: LLMPurpose,
        system: str | None = None,
    ) -> str | None:
        """Generate text from a conversation history. Returns None if unavailable."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...
