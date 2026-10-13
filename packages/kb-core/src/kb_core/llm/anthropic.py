"""Anthropic LLM client with graceful degradation.

Channel-agnostic: takes an :class:`~kb_core.config.AnthropicProviderConfig`
at construction time. No environment reads — every tunable is on the
config. If ``config.api_key`` is ``None`` the underlying SDK is allowed
to resolve credentials on its own (env, keychain, etc.). The kb_core
class itself never touches ``os.environ``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from kb_core.llm.provider import LLMCallRecord

if TYPE_CHECKING:
    from kb_core.config import AnthropicProviderConfig
    from kb_core.llm.provider import (
        DiagnosticsState,
        LLMCallOutcome,
        LLMCallRecorder,
        LLMPurpose,
        Message,
    )

logger = logging.getLogger(__name__)


_SONNET_MODEL = "claude-sonnet-5-5"
"""Default model identifier for the human-facing synthesis role."""

_PREVIOUS_IDS: dict[tuple[str, str], str] = {}
"""Last response id per ``(purpose, model)``, sent as the diagnostics ``previous_message_id``.

Process-wide and unlocked: concurrent calls on one key each send the value
present when they start, and the last to complete wins.
"""

_UNPARSED_WARNED: set[str] = set()
"""Models already warned about an unparsed diagnostics shape (once per model per process)."""

_DIAGNOSTICS_RAW_MAX = 2000
"""Longest diagnostics_raw text stored per record."""


def _has_text(response: Any) -> bool:
    """Whether a response has at least one text block (``_extract_text``'s predicate, silent)."""
    try:
        blocks = list(response.content)
    except Exception:
        return False
    return any(
        getattr(b, "type", None) == "text" and isinstance(getattr(b, "text", None), str)
        for b in blocks
    )


def _int_or_none(value: Any) -> int | None:
    """Keep ``value`` only when it is a real int (not a bool)."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _str_or_none(value: Any) -> str | None:
    """Keep ``value`` only when it is a str."""
    return value if isinstance(value, str) else None


def _is_diagnostics_400(exc: BaseException) -> bool:
    """Whether ``exc`` is a 400 that names the diagnostics field."""
    return getattr(exc, "status_code", None) == 400 and "diagnostics" in str(exc).lower()


def _parse_diagnostics(
    response: Any, *, model: str, purpose: str
) -> tuple[str | None, int | None, str | None]:
    """Return ``(cache_miss_reason, cache_missed_input_tokens, diagnostics_raw)``."""
    extra = getattr(response, "model_extra", None)
    if not isinstance(extra, dict) or "diagnostics" not in extra:
        return None, None, None
    diag = extra["diagnostics"]
    raw = json.dumps(diag, separators=(",", ":"), default=str)[:_DIAGNOSTICS_RAW_MAX]
    if diag is None:
        return None, None, raw
    if isinstance(diag, dict) and "cache_miss_reason" in diag:
        reason = diag["cache_miss_reason"]
        if reason is None:
            return "pending", None, raw
        if isinstance(reason, dict) and isinstance(reason.get("type"), str):
            return reason["type"], _int_or_none(reason.get("cache_missed_input_tokens")), raw
    if model not in _UNPARSED_WARNED:
        _UNPARSED_WARNED.add(model)
        logger.warning("llm-diagnostics unparsed shape model=%s purpose=%s", model, purpose)
    return "unparsed", None, raw


def _system_sha256(kwargs: dict[str, Any]) -> str | None:
    """Short stable hash of the request's system blocks, or None without a system."""
    if "system" not in kwargs:
        return None
    payload = json.dumps(kwargs["system"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _extract_text(response: Any) -> str | None:
    """Join the text blocks of a response; None (with a warning) if there are none.

    Responses may lead with thinking/redacted_thinking blocks, so ``content[0]``
    is not safe to read.
    """
    blocks = list(response.content)
    texts = [
        b.text
        for b in blocks
        if getattr(b, "type", None) == "text" and isinstance(getattr(b, "text", None), str)
    ]
    if not texts:
        logger.warning(
            "Anthropic response had no text block (stop_reason=%s, block types: %s)",
            getattr(response, "stop_reason", None),
            [getattr(b, "type", type(b).__name__) for b in blocks],
        )
        return None
    return "".join(texts)


def _build_record(
    model: str,
    kwargs: dict[str, Any],
    *,
    purpose: LLMPurpose,
    start: float,
    state: DiagnosticsState,
    prev: str | None,
    response: Any,
    error: BaseException | None,
) -> LLMCallRecord:
    """Build the :class:`LLMCallRecord` for one ``_create`` call."""
    latency_ms = int((time.monotonic() - start) * 1000)
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    outcome: LLMCallOutcome
    usage_fields: list[int | None] = [None] * 5
    response_id = stop_reason = None
    miss_reason: str | None = None
    missed_tokens: int | None = None
    raw: str | None = None
    if error is not None:
        outcome = "error"
    else:
        outcome = "ok" if _has_text(response) else "no_text"
        response_id = _str_or_none(getattr(response, "id", None))
        stop_reason = _str_or_none(getattr(response, "stop_reason", None))
        usage = getattr(response, "usage", None)
        usage_fields = [
            _int_or_none(getattr(usage, "input_tokens", None)),
            _int_or_none(getattr(usage, "output_tokens", None)),
            _int_or_none(getattr(usage, "cache_creation_input_tokens", None)),
            _int_or_none(
                getattr(
                    getattr(usage, "cache_creation", None),
                    "ephemeral_1h_input_tokens",
                    None,
                )
            ),
            _int_or_none(getattr(usage, "cache_read_input_tokens", None)),
        ]
        if state == "sent":
            miss_reason, missed_tokens, raw = _parse_diagnostics(
                response, model=model, purpose=purpose
            )
    return LLMCallRecord(
        ts=ts,
        purpose=purpose,
        model=model,
        outcome=outcome,
        error_type=type(error).__name__ if error is not None else None,
        response_id=response_id,
        stop_reason=stop_reason,
        input_tokens=usage_fields[0],
        output_tokens=usage_fields[1],
        cache_creation_input_tokens=usage_fields[2],
        cache_creation_1h_input_tokens=usage_fields[3],
        cache_read_input_tokens=usage_fields[4],
        diagnostics_state=state,
        previous_message_id=prev if state == "sent" else None,
        cache_miss_reason=miss_reason,
        cache_missed_input_tokens=missed_tokens,
        diagnostics_raw=raw,
        system_sha256=_system_sha256(kwargs),
        latency_ms=latency_ms,
        thinking=kwargs.get("thinking", {}).get("type"),
        effort=kwargs.get("output_config", {}).get("effort"),
    )


class AnthropicLLMClient:
    """Generates text via the Anthropic Messages API."""

    def __init__(
        self,
        config: AnthropicProviderConfig,
        *,
        recorder: LLMCallRecorder | None = None,
        system_cache_1h_purposes: frozenset[LLMPurpose] = frozenset(),
    ) -> None:
        """Initialize with an explicit provider config and an optional call recorder.

        ``system_cache_1h_purposes`` lists the purposes whose system block gets
        the 1-hour cache TTL instead of the 5-minute default.
        """
        self._config = config
        self._recorder = recorder
        self._system_cache_1h_purposes = system_cache_1h_purposes
        self._client: Any = None
        self._available: bool | None = None
        self._diagnostics_disabled: bool = False

    async def is_available(self) -> bool:
        """Check availability. Only caches success — retries on failure.

        Returns True if the SDK is importable. We no longer pre-check for
        an explicit API key — when ``config.api_key`` is ``None`` we
        defer credential resolution to the SDK. A failed ``generate()``
        will clear the cached availability so the next call retries.
        """
        if self._available is True:
            return True
        try:
            client = self._get_client()
            return client is not None
        except Exception:
            return False

    def _system_block(self, system: str, purpose: LLMPurpose) -> list[dict[str, Any]]:
        """Build the system content-block array with an explicit cache breakpoint."""
        # Explicit breakpoint on the system block. Never send top-level
        # cache_control: the API treats it as automatic caching, puts the
        # breakpoint on the last cacheable block and writes the whole prompt,
        # including a one-shot user message, at the cache-write premium.
        cache_control = {"type": "ephemeral"}
        if purpose in self._system_cache_1h_purposes:
            cache_control["ttl"] = "1h"
        return [{"type": "text", "text": system, "cache_control": cache_control}]

    def _apply_effort_thinking(self, kwargs: dict[str, Any]) -> None:
        """Add the optional effort and thinking request fields when configured.

        ``effort`` goes out as ``output_config.effort`` and ``thinking`` as
        ``thinking = {"type": mode}``. Either being ``None`` (the default) adds
        nothing, so the kwargs dict is byte-identical to before.
        """
        if self._config.effort is not None:
            kwargs["output_config"] = {"effort": self._config.effort}
        if self._config.thinking is not None:
            kwargs["thinking"] = {"type": self._config.thinking}

    async def generate(
        self,
        prompt: str,
        *,
        purpose: LLMPurpose,
        system: str | None = None,
    ) -> str | None:
        """Generate text from a prompt. Returns None if unavailable."""
        try:
            client = self._get_client()
            if client is None:
                return None

            kwargs: dict[str, Any] = {
                "model": self._config.model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system is not None:
                kwargs["system"] = self._system_block(system, purpose)
            self._apply_effort_thinking(kwargs)

            response = await self._create(client, kwargs, purpose=purpose)
            result = _extract_text(response)
            if result is None:
                self._available = None
                return None
            self._available = True
            return result
        except Exception:
            logger.warning("Anthropic generation failed", exc_info=True)
            self._available = None
            return None

    async def generate_chat(
        self,
        messages: list[Message],
        *,
        purpose: LLMPurpose,
        system: str | None = None,
    ) -> str | None:
        """Generate text from a conversation history.

        Prompt caching uses explicit breakpoints on content blocks. The client
        never sends top-level ``cache_control``: the API treats that as
        automatic caching, which places a breakpoint on the last cacheable
        block and charges the cache-write premium (1.25x base input for the
        5-minute TTL, 2x for 1 hour) on a prefix a one-shot call never reads
        back.  Two breakpoints are set per request:

        1. **System block** (when ``system`` is provided): the system prompt is
           sent as a content-block array with ``cache_control`` attached, with
           the 5-minute TTL, or 1 hour when ``purpose`` is in
           ``system_cache_1h_purposes``, so the stable system prefix is cached
           across turns.
        2. **Rolling message breakpoint**: the last content block of the last
           message carries ``cache_control`` so each turn reuses the prefix
           built by the previous turn (5-min TTL, refreshed on read).  If the
           last message's ``content`` is a bare string it is first converted to
           a single text content block before attaching the breakpoint.

        Requests below the model's minimum cacheable token count silently
        no-op (``cache_creation_input_tokens=0``), which is harmless.
        """
        try:
            client = self._get_client()
            if client is None:
                return None

            # Build message list; attach rolling cache breakpoint to the last
            # content block of the last message.
            api_messages: list[dict[str, Any]] = []
            for i, m in enumerate(messages):
                if i < len(messages) - 1:
                    api_messages.append({"role": m["role"], "content": m["content"]})
                else:
                    # Normalise the final message content to a list of blocks.
                    content = m["content"]
                    if isinstance(content, str):
                        blocks: list[dict[str, Any]] = [{"type": "text", "text": content}]
                    else:
                        blocks = [dict(b) for b in content]
                    # Attach rolling breakpoint to the last block.
                    blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
                    api_messages.append({"role": m["role"], "content": blocks})

            kwargs: dict[str, Any] = {
                "model": self._config.model,
                "max_tokens": 4096,
                "messages": api_messages,
            }
            if system is not None:
                kwargs["system"] = self._system_block(system, purpose)
            self._apply_effort_thinking(kwargs)

            response = await self._create(client, kwargs, purpose=purpose)
            result = _extract_text(response)
            if result is None:
                self._available = None
                return None
            self._available = True
            return result
        except Exception:
            logger.warning("Anthropic chat generation failed", exc_info=True)
            self._available = None
            return None

    async def _create(self, client: Any, kwargs: dict[str, Any], *, purpose: LLMPurpose) -> Any:
        """Call ``messages.create``, recording usage and cache diagnostics when a recorder is set.

        Never mutates ``kwargs``. Returns the SDK response or re-raises the
        final exception unchanged. With diagnostics on, a 400 naming the
        diagnostics field disables it for this client and retries once
        without it.
        """
        timeout = self._config.timeout
        if self._recorder is None:
            return await client.messages.create(**kwargs, timeout=timeout)

        recorder = self._recorder
        model = self._config.model
        key = (purpose, model)
        state: DiagnosticsState
        prev: str | None = None
        if self._config.diagnostics is False:
            state = "off_config"
        elif self._diagnostics_disabled:
            state = "off_disabled"
        else:
            state = "sent"
            prev = _PREVIOUS_IDS.get(key)

        def emit(response: Any, error: BaseException | None) -> None:
            try:
                recorder(
                    _build_record(
                        model,
                        kwargs,
                        purpose=purpose,
                        start=start,
                        state=state,
                        prev=prev,
                        response=response,
                        error=error,
                    )
                )
            except Exception:
                logger.warning("llm-call recorder failed purpose=%s", purpose, exc_info=True)

        start = time.monotonic()
        try:
            if state == "sent":
                try:
                    response = await client.messages.create(
                        **kwargs,
                        extra_body={"diagnostics": {"previous_message_id": prev}},
                        timeout=timeout,
                    )
                except Exception as exc:
                    if not _is_diagnostics_400(exc):
                        raise
                    if not self._diagnostics_disabled:
                        self._diagnostics_disabled = True
                        logger.warning("llm-diagnostics disabled model=%s reason=400", model)
                    state = "fallback"
                    response = await client.messages.create(**kwargs, timeout=timeout)
            else:
                response = await client.messages.create(**kwargs, timeout=timeout)
        except (Exception, asyncio.CancelledError) as exc:
            emit(None, exc)
            raise

        if state == "sent":
            response_id = getattr(response, "id", None)
            if isinstance(response_id, str):
                _PREVIOUS_IDS[key] = response_id
        emit(response, None)
        return response

    def _get_client(self) -> Any:
        """Lazily create the AsyncAnthropic client. Returns None if SDK missing."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic

                kwargs: dict[str, Any] = {}
                if self._config.api_key is not None:
                    kwargs["api_key"] = self._config.api_key
                self._client = AsyncAnthropic(**kwargs)
            except ImportError:
                logger.warning("anthropic package not installed — Anthropic LLM disabled")
                return None
        return self._client

    async def close(self) -> None:
        """Close the Anthropic client if open."""
        if self._client is not None:
            await self._client.close()
            self._client = None
