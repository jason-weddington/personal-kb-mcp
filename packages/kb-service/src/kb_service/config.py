"""Engine-config adapters: snapshot ``KB_*`` env vars into kb_core dataclasses.

Ported (not imported — separate repo) from
``personal_kb/src/personal_kb/config.py``. These builders are the single
channel-side adapter layer between the service's environment and the env-free
``kb_core.config`` dataclasses that ``create_postgres`` consumes. The DSN for
the kb-core data DB (``KB_DATABASE_URL``) and per-request attribution are NOT
handled here — the lifespan passes the DSN directly to ``create_postgres`` and
attribution defaults to ``Attribution()``.
"""

import logging
import os
from typing import TYPE_CHECKING

from starlette.requests import Request

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from kb_core.config import (
        AgenticConfig,
        AnthropicProviderConfig,
        BedrockProviderConfig,
        EmbeddingConfig,
        EmbeddingRetryConfig,
        IngestConfig,
        OllamaProviderConfig,
        ProviderConfig,
    )

_VALID_PROVIDERS = {"anthropic", "bedrock", "ollama"}


def _parse_int(env_var: str, default: str) -> int:
    """Parse an integer env var with a clear error on bad values."""
    raw = os.environ.get(env_var, default)
    try:
        return int(raw)
    except ValueError:
        msg = f"{env_var}={raw!r} is not a valid integer"
        raise ValueError(msg) from None


def _parse_float(env_var: str, default: str) -> float:
    """Parse a float env var with a clear error on bad values."""
    raw = os.environ.get(env_var, default)
    try:
        return float(raw)
    except ValueError:
        msg = f"{env_var}={raw!r} is not a valid number"
        raise ValueError(msg) from None


def _parse_provider(env_var: str, default: str) -> str:
    """Parse and validate a provider env var."""
    raw = os.environ.get(env_var, default).lower()
    if raw not in _VALID_PROVIDERS:
        choices = ", ".join(sorted(_VALID_PROVIDERS))
        msg = f"{env_var}={raw!r} is not valid. Choose from: {choices}"
        raise ValueError(msg)
    return raw


# --- env getters -----------------------------------------------------------

DEFAULT_CLIENT_INSTALL_SPEC = (
    "personal-kb @ git+https://github.com/jason-weddington/personal-kb-mcp"
)


def get_client_install_spec() -> str:
    """The ``uvx --from`` spec shown in the Settings MCP snippet.

    Env ``KB_SERVICE_CLIENT_INSTALL_SPEC``; defaults to the public GitHub URL.
    """
    return (
        os.environ.get("KB_SERVICE_CLIENT_INSTALL_SPEC") or DEFAULT_CLIENT_INSTALL_SPEC
    )


def get_ollama_url() -> str:
    """Return the Ollama API URL from KB_OLLAMA_URL."""
    return os.environ.get("KB_OLLAMA_URL", "http://localhost:11434")


def get_embedding_model() -> str:
    """Return the embedding model name from KB_EMBEDDING_MODEL."""
    return os.environ.get("KB_EMBEDDING_MODEL", "qwen3-embedding:0.6b")


def get_ollama_timeout() -> float:
    """Return the Ollama timeout in seconds from KB_OLLAMA_TIMEOUT."""
    return _parse_float("KB_OLLAMA_TIMEOUT", "10.0")


def get_embedding_keep_alive() -> str:
    """Return the per-request Ollama keep_alive duration from KB_OLLAMA_KEEP_ALIVE.

    Passed through verbatim to the embed request body so only the
    embedding model is pinned in VRAM (never a host-global default).
    """
    return os.environ.get("KB_OLLAMA_KEEP_ALIVE", "30m")


def get_embedding_dim() -> int:
    """Return the embedding vector dimensions from KB_EMBEDDING_DIM."""
    return _parse_int("KB_EMBEDDING_DIM", "1024")


def get_llm_model() -> str:
    """Return the Ollama LLM model name from KB_OLLAMA_MODEL."""
    return os.environ.get("KB_OLLAMA_MODEL", "qwen3:4b")


def get_llm_timeout() -> float:
    """Return the Ollama LLM timeout in seconds from KB_OLLAMA_LLM_TIMEOUT."""
    return _parse_float("KB_OLLAMA_LLM_TIMEOUT", "120.0")


def get_anthropic_model() -> str:
    """Return the Anthropic model name from KB_ANTHROPIC_MODEL."""
    return os.environ.get("KB_ANTHROPIC_MODEL", "claude-haiku-5-5")


def get_anthropic_thinking() -> str | None:
    """Return the thinking mode for the extraction and query roles.

    ``KB_ANTHROPIC_THINKING`` (case-insensitive): ``disabled`` (the default,
    also when unset or empty) sends ``thinking = {"type": "disabled"}``;
    ``adaptive`` sends adaptive thinking; ``default`` maps to ``None`` so no
    thinking field is sent at all (the model's own default). Any other value
    warns once per call and falls back to ``disabled``. Never applied to the
    synthesis role or the surprise capture clients.
    """
    raw = os.environ.get("KB_ANTHROPIC_THINKING", "").strip().lower()
    if raw in ("", "disabled"):
        return "disabled"
    if raw == "adaptive":
        return "adaptive"
    if raw == "default":
        return None
    logger.warning(
        "KB_ANTHROPIC_THINKING=%r is not one of disabled/adaptive/default;"
        " falling back to disabled",
        raw,
    )
    return "disabled"


def get_anthropic_timeout() -> float:
    """Return the Anthropic timeout in seconds from KB_ANTHROPIC_TIMEOUT."""
    return _parse_float("KB_ANTHROPIC_TIMEOUT", "30.0")


def get_synthesis_model() -> str:
    """Return the synthesis-role Anthropic model from KB_SYNTHESIS_MODEL.

    Unset or empty means the kb-core ``_SONNET_MODEL`` default
    (``claude-sonnet-5-5``); any other non-empty value is used verbatim
    (whitespace stripped, no validation). Applies only to the synthesis role.
    """
    raw = os.environ.get("KB_SYNTHESIS_MODEL", "").strip()
    if raw:
        return raw
    from kb_core.llm.anthropic import _SONNET_MODEL

    return _SONNET_MODEL


_SYNTHESIS_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


def get_synthesis_effort() -> str | None:
    """Return the synthesis-role reasoning effort from KB_SYNTHESIS_EFFORT.

    Case-insensitive, whitespace stripped. Unset or empty means ``low`` (the
    measured default for ``claude-sonnet-5-5``); one of ``low``, ``medium``,
    ``high``, ``xhigh`` or ``max`` means that effort; ``default`` maps to
    ``None`` so no effort field is sent at all (the model's own default).
    Any other value warns once per call and falls back to ``low``. Applies
    only to the synthesis role.
    """
    raw = os.environ.get("KB_SYNTHESIS_EFFORT", "").strip()
    value = raw.lower()
    if value in ("", "low"):
        return "low"
    if value == "default":
        return None
    if value in _SYNTHESIS_EFFORTS:
        return value
    logger.warning(
        "KB_SYNTHESIS_EFFORT=%r is not one of low/medium/high/xhigh/max/default;"
        " falling back to low",
        raw,
    )
    return "low"


def get_extraction_provider() -> str:
    """Return the LLM provider for graph extraction from KB_EXTRACTION_PROVIDER."""
    return _parse_provider("KB_EXTRACTION_PROVIDER", "anthropic")


def get_query_provider() -> str:
    """Return the LLM provider for query planning from KB_QUERY_PROVIDER."""
    return _parse_provider("KB_QUERY_PROVIDER", "anthropic")


def get_bedrock_model() -> str:
    """Return the Bedrock model ID from KB_BEDROCK_MODEL."""
    return os.environ.get("KB_BEDROCK_MODEL", "us.anthropic.claude-haiku-5-5")


def get_bedrock_synthesis_model() -> str:
    """Return the synthesis-role Bedrock model from KB_BEDROCK_SYNTHESIS_MODEL.

    Unset or empty means the kb-core Bedrock ``_SONNET_MODEL`` default
    (``us.anthropic.claude-sonnet-5-5``); any other non-empty value is used
    verbatim (whitespace stripped, no validation). Applies only to the
    synthesis role on Bedrock.
    """
    raw = os.environ.get("KB_BEDROCK_SYNTHESIS_MODEL", "").strip()
    if raw:
        return raw
    from kb_core.llm.bedrock import _SONNET_MODEL

    return _SONNET_MODEL


def get_bedrock_region() -> str:
    """Return the AWS region for Bedrock from KB_BEDROCK_REGION."""
    return os.environ.get("KB_BEDROCK_REGION", "us-east-1")


def get_bedrock_timeout() -> float:
    """Return the Bedrock timeout in seconds from KB_BEDROCK_TIMEOUT."""
    return _parse_float("KB_BEDROCK_TIMEOUT", "60.0")


def get_aws_profile() -> str | None:
    """Return the AWS profile name for Bedrock credentials, or None."""
    return os.environ.get("KB_AWS_PROFILE") or None


def get_pg_pool_min() -> int:
    """Return the Postgres connection pool minimum size from KB_PG_POOL_MIN."""
    return _parse_int("KB_PG_POOL_MIN", "1")


def get_pg_pool_max() -> int:
    """Return the Postgres connection pool maximum size from KB_PG_POOL_MAX."""
    return _parse_int("KB_PG_POOL_MAX", "5")


def get_ingest_max_file_size() -> int:
    """Return max file size in bytes for ingestion from KB_INGEST_MAX_FILE_SIZE."""
    return _parse_int("KB_INGEST_MAX_FILE_SIZE", str(10 * 1024 * 1024))


def get_ingest_chunk_size() -> int:
    """Return chunk size in chars for ingestion from KB_INGEST_CHUNK_SIZE."""
    return _parse_int("KB_INGEST_CHUNK_SIZE", "16000")


def get_ingest_chunk_overlap() -> int:
    """Return chunk overlap in chars for ingestion from KB_INGEST_CHUNK_OVERLAP."""
    return _parse_int("KB_INGEST_CHUNK_OVERLAP", "600")


def is_agentic_ingest() -> bool:
    """Return True if agentic ingestion dedup is enabled (default: TRUE)."""
    return os.environ.get("KB_AGENTIC_INGEST", "TRUE").upper() == "TRUE"


def get_ingest_dedup_threshold() -> float:
    """Return the hybrid search score threshold for dedup."""
    return _parse_float("KB_INGEST_DEDUP_THRESHOLD", "0.06")


NEAR_DUPLICATE_FLOOR_DEFAULT = 0.88


def get_near_duplicate_floor() -> float:
    """Return the cosine floor at/above which a create is a near-duplicate.

    Read on every request (not cached). A malformed value raises ValueError.
    """
    return _parse_float("KB_NEAR_DUPLICATE_FLOOR", str(NEAR_DUPLICATE_FLOOR_DEFAULT))


def is_safety_skip() -> bool:
    """Return True if KB_SKIP_SAFETY is set to TRUE."""
    return os.environ.get("KB_SKIP_SAFETY", "").upper() == "TRUE"


def is_manager_mode() -> bool:
    """Return True if KB_MANAGER is TRUE (registers maintenance MCP tools)."""
    return os.environ.get("KB_MANAGER", "").upper() == "TRUE"


def get_contributor() -> str | None:
    """Return KB_CONTRIBUTOR, or None when unset/empty."""
    return os.environ.get("KB_CONTRIBUTOR") or None


def get_instance_role() -> str:
    """Return KB_INSTANCE_ROLE lower-cased ('' when unset)."""
    return os.environ.get("KB_INSTANCE_ROLE", "").lower()


def public_base_url(request: Request) -> str:
    """Public-facing base URL for issued links (invites, password resets).

    Behind a reverse proxy that doesn't forward the original Host header,
    ``request.url.netloc`` reports the inner bind address (e.g. ``localhost:8000``)
    which is useless for sharing. Set ``KB_SERVICE_PUBLIC_URL`` (e.g.
    ``https://kb-service``) to override.
    """
    override = os.environ.get("KB_SERVICE_PUBLIC_URL", "").rstrip("/")
    if override:
        return override
    return f"{request.url.scheme}://{request.url.netloc}"


def is_agentic_query() -> bool:
    """Return True if agentic query planning is enabled (default: TRUE)."""
    return os.environ.get("KB_AGENTIC_QUERY", "TRUE").upper() == "TRUE"


def get_agentic_max_tool_calls() -> int:
    """Return max tool calls for agentic query loop from KB_AGENTIC_MAX_CALLS."""
    return _parse_int("KB_AGENTIC_MAX_CALLS", "4")


def is_agentic_synthesis() -> bool:
    """Return True if agentic synthesis is enabled (default: TRUE)."""
    return os.environ.get("KB_AGENTIC_SYNTHESIS", "TRUE").upper() == "TRUE"


def is_embed_worker_enabled() -> bool:
    """Return True if the embedding retry worker is enabled (default: TRUE)."""
    return os.environ.get("KB_EMBED_WORKER_ENABLED", "TRUE").upper() == "TRUE"


def get_embed_worker_batch_size() -> int:
    """Return the embedding retry worker's per-drain batch size."""
    return _parse_int("KB_EMBED_WORKER_BATCH_SIZE", "16")


def get_embed_worker_poll_seconds() -> float:
    """Return the embedding retry worker's poll interval in seconds."""
    return _parse_float("KB_EMBED_WORKER_POLL_SECONDS", "60.0")


def get_embed_worker_timeout() -> float:
    """Return the embedding retry worker's own embedder request timeout in seconds."""
    return _parse_float("KB_EMBED_WORKER_TIMEOUT", "180.0")


# --- kb_core config builders -----------------------------------------------


def build_ingest_config() -> "IngestConfig":
    """Build a kb_core ``IngestConfig`` from this module's env getters."""
    from kb_core.config import IngestConfig

    return IngestConfig(
        max_file_size=get_ingest_max_file_size(),
        chunk_size=get_ingest_chunk_size(),
        chunk_overlap=get_ingest_chunk_overlap(),
        dedup_threshold=get_ingest_dedup_threshold(),
        agentic_ingest=is_agentic_ingest(),
        skip_safety=is_safety_skip(),
    )


def build_embedding_config() -> "EmbeddingConfig":
    """Build a kb_core ``EmbeddingConfig`` from this module's env getters.

    NOTE: ``keep_alive`` is deliberately NOT passed here yet.
    ``get_embedding_keep_alive()`` above reads ``KB_OLLAMA_KEEP_ALIVE`` and is
    ready to wire in, but the ``EmbeddingConfig`` dataclass this service
    currently gets via the ``uv.lock``-pinned kb-core git rev predates the
    ``keep_alive`` field (GTD e6c01c04) — passing it raises ``TypeError:
    unexpected keyword argument 'keep_alive'`` and breaks the service
    lifespan. Add ``keep_alive=get_embedding_keep_alive(),`` to the call
    below in the same change that bumps kb-core
    (``uv lock --upgrade-package kb-core``), matching the established
    pattern (e.g. f4cdca7's EmbeddingRetryConfig wiring, which bundled its
    own lock bump).
    """
    from kb_core.config import EmbeddingConfig

    return EmbeddingConfig(
        ollama_url=get_ollama_url(),
        timeout=get_ollama_timeout(),
        model=get_embedding_model(),
        dim=get_embedding_dim(),
    )


def build_embedding_retry_config() -> "EmbeddingRetryConfig":
    """Build a kb_core ``EmbeddingRetryConfig`` from this module's env getters."""
    from kb_core.config import EmbeddingRetryConfig

    return EmbeddingRetryConfig(
        enabled=is_embed_worker_enabled(),
        batch_size=get_embed_worker_batch_size(),
        poll_interval_seconds=get_embed_worker_poll_seconds(),
        request_timeout=get_embed_worker_timeout(),
    )


_LLM_DIAGNOSTICS_OFF = frozenset({"0", "false", "off", "no"})


def get_llm_diagnostics() -> bool:
    """Whether recorded Anthropic calls send the cache-diagnostics object.

    ``KB_LLM_DIAGNOSTICS`` set to 0, false, off or no (any case) turns it off;
    anything else, including unset or empty, leaves it on.
    """
    raw = os.environ.get("KB_LLM_DIAGNOSTICS", "").strip().lower()
    return raw not in _LLM_DIAGNOSTICS_OFF


def build_anthropic_config(
    *,
    model: str | None = None,
    effort: str | None = None,
    thinking: str | None = None,
) -> "AnthropicProviderConfig":
    """Snapshot env-driven Anthropic config into the explicit dataclass.

    ``effort`` and ``thinking`` are passed through to
    ``AnthropicProviderConfig`` unchanged (``None`` sends neither field to the
    API). This function deliberately does NOT read ``KB_ANTHROPIC_THINKING``
    itself: the surprise detector, distiller and critic clients are built
    through it with only ``model=``, so they must stay thinking-free; the role
    builder (:func:`build_provider_config`) passes the env-resolved value in
    explicitly. It also never reads ``KB_SYNTHESIS_MODEL`` or
    ``KB_SYNTHESIS_EFFORT``: the synthesis role is the only caller that passes
    ``effort``, so the capture clients stay effort-free; other ``effort``
    callers are eval scripts.
    """
    from kb_core.config import AnthropicProviderConfig

    return AnthropicProviderConfig(
        model=model or get_anthropic_model(),
        timeout=get_anthropic_timeout(),
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        diagnostics=get_llm_diagnostics(),
        effort=effort,
        thinking=thinking,
    )


def build_bedrock_config(
    *, model: str | None = None, thinking: str | None = None, effort: str | None = None
) -> "BedrockProviderConfig":
    """Snapshot env-driven Bedrock config into the explicit dataclass."""
    from kb_core.config import BedrockProviderConfig

    return BedrockProviderConfig(
        model=model or get_bedrock_model(),
        timeout=get_bedrock_timeout(),
        region=get_bedrock_region(),
        profile=get_aws_profile(),
        bearer_token=os.environ.get("AWS_BEARER_TOKEN_BEDROCK"),
        has_env_credentials=bool(os.environ.get("AWS_ACCESS_KEY_ID")),
        thinking=thinking,
        effort=effort,
    )


def build_ollama_provider_config() -> "OllamaProviderConfig":
    """Snapshot env-driven Ollama LLM config into the explicit dataclass."""
    from kb_core.config import OllamaProviderConfig

    return OllamaProviderConfig(
        model=get_llm_model(),
        timeout=get_llm_timeout(),
        url=get_ollama_url(),
    )


def build_provider_config() -> "ProviderConfig":
    """Build a per-role provider config for the engine.

    Each role (extraction / query / synthesis) is a ``ProviderRoleConfig`` that
    bundles which provider is active + credentials for all three providers. The
    synthesis role points at the Sonnet model (``KB_SYNTHESIS_MODEL``, default
    ``claude-sonnet-5-5``) at effort ``KB_SYNTHESIS_EFFORT`` (default ``low``),
    and never sends a thinking field. The extraction and query roles get their
    thinking mode from ``KB_ANTHROPIC_THINKING`` (disabled by default).
    """
    from kb_core.config import ProviderConfig, ProviderRoleConfig

    extraction_provider = get_extraction_provider()
    query_provider = get_query_provider()

    # One read so an invalid value warns once per build. The same mode reaches
    # Claude on Bedrock: same models, same measured setting.
    thinking = get_anthropic_thinking()
    anthropic = build_anthropic_config(thinking=thinking)
    bedrock = build_bedrock_config(thinking=thinking)
    ollama = build_ollama_provider_config()

    synthesis_anthropic = build_anthropic_config(
        model=get_synthesis_model(), effort=get_synthesis_effort()
    )
    synthesis_bedrock = build_bedrock_config(
        model=get_bedrock_synthesis_model(), effort=get_synthesis_effort()
    )

    extraction_role = ProviderRoleConfig(
        provider=extraction_provider,  # type: ignore[arg-type]
        anthropic=anthropic,
        bedrock=bedrock,
        ollama=ollama,
    )
    query_role = ProviderRoleConfig(
        provider=query_provider,  # type: ignore[arg-type]
        anthropic=anthropic,
        bedrock=bedrock,
        ollama=ollama,
    )
    synthesis_role = ProviderRoleConfig(
        provider=query_provider,  # type: ignore[arg-type]
        anthropic=synthesis_anthropic,
        bedrock=synthesis_bedrock,
        ollama=ollama,
    )

    logger.info(
        "llm roles extraction_model=%s extraction_thinking=%s query_model=%s"
        " query_thinking=%s synthesis_model=%s synthesis_thinking=%s"
        " synthesis_effort=%s extraction_provider=%s query_provider=%s"
        " bedrock_model=%s bedrock_synthesis_model=%s",
        anthropic.model,
        anthropic.thinking,
        anthropic.model,
        anthropic.thinking,
        synthesis_anthropic.model,
        synthesis_anthropic.thinking,
        synthesis_anthropic.effort,
        extraction_provider,
        query_provider,
        bedrock.model,
        synthesis_bedrock.model,
    )
    return ProviderConfig(
        extraction=extraction_role,
        query=query_role,
        synthesis=synthesis_role,
    )


def build_agentic_config() -> "AgenticConfig":
    """Snapshot env-driven agentic flags into the explicit dataclass."""
    from kb_core.config import AgenticConfig

    return AgenticConfig(
        agentic_query=is_agentic_query(),
        agentic_synthesis=is_agentic_synthesis(),
        max_tool_calls=get_agentic_max_tool_calls(),
    )
