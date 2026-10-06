"""Environment-variable-based configuration."""

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kb_core.config import (
        AgenticConfig,
        AnthropicProviderConfig,
        Attribution,
        BedrockProviderConfig,
        DatabaseConfig,
        EmbeddingConfig,
        IngestConfig,
        KbConfig,
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
        msg = f"{env_var}={raw!r} is not valid. Choose from: {', '.join(sorted(_VALID_PROVIDERS))}"
        raise ValueError(msg)
    return raw


def get_db_path() -> Path:
    """Return the database file path from KB_DB_PATH."""
    raw = os.environ.get("KB_DB_PATH", "~/.local/share/personal_kb/knowledge.db")
    return Path(raw).expanduser()


def get_hook_scratch_path(session_id: str) -> Path:
    """Return the per-session hook scratch file path.

    Used by the CLI hook to suppress re-injection of the same map directory in
    a session. Stored under ``~/.cache/personal_kb/`` so it survives across the
    pair of ``SessionStart`` / ``UserPromptSubmit`` hook invocations within one
    session, but is naturally torn down with the cache.
    """
    return Path(f"~/.cache/personal_kb/injected-{session_id}.json").expanduser()


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


def is_manager_mode() -> bool:
    """Return True if KB_MANAGER is set to TRUE."""
    return os.environ.get("KB_MANAGER", "").upper() == "TRUE"


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
    return os.environ.get("KB_ANTHROPIC_MODEL", "claude-haiku-4-5")


def get_anthropic_timeout() -> float:
    """Return the Anthropic timeout in seconds from KB_ANTHROPIC_TIMEOUT."""
    return _parse_float("KB_ANTHROPIC_TIMEOUT", "30.0")


def get_extraction_provider() -> str:
    """Return the LLM provider for graph extraction from KB_EXTRACTION_PROVIDER."""
    return _parse_provider("KB_EXTRACTION_PROVIDER", "anthropic")


def get_query_provider() -> str:
    """Return the LLM provider for query planning from KB_QUERY_PROVIDER."""
    return _parse_provider("KB_QUERY_PROVIDER", "anthropic")


def get_log_level() -> str:
    """Return the logging level from KB_LOG_LEVEL."""
    return os.environ.get("KB_LOG_LEVEL", "WARNING")


def get_bedrock_model() -> str:
    """Return the Bedrock model ID from KB_BEDROCK_MODEL."""
    return os.environ.get("KB_BEDROCK_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")


def get_bedrock_region() -> str:
    """Return the AWS region for Bedrock from KB_BEDROCK_REGION."""
    return os.environ.get("KB_BEDROCK_REGION", "us-east-1")


def get_bedrock_timeout() -> float:
    """Return the Bedrock timeout in seconds from KB_BEDROCK_TIMEOUT."""
    return _parse_float("KB_BEDROCK_TIMEOUT", "60.0")


def get_database_url() -> str | None:
    """Return database URL if set, None for SQLite file-based.

    When set to a ``postgresql://`` URL, the server will create a
    PostgresBackend instead of SQLiteBackend.
    """
    return os.environ.get("KB_DATABASE_URL")


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
    """Return the hybrid search score threshold for dedup from KB_INGEST_DEDUP_THRESHOLD."""
    return _parse_float("KB_INGEST_DEDUP_THRESHOLD", "0.06")


def build_ingest_config() -> "IngestConfig":
    """Build a kb_core ``IngestConfig`` from this module's env getters.

    The kb_core ingest pipeline is env-free; the engine reads its
    tunables from a typed dataclass instead of ``os.environ``. This
    helper is the single channel-side adapter that snapshots the
    relevant ``KB_*`` env vars and hands them to the engine. Every
    ``FileIngester(...)`` construction site uses it so behavior stays
    identical across the move and the env surface stays centralized
    here.
    """
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

    The kb_core ``EmbeddingClient`` is env-free; it reads its tunables
    from a typed dataclass instead of ``os.environ``. This helper is the
    single channel-side adapter that snapshots ``KB_OLLAMA_URL``,
    ``KB_EMBEDDING_MODEL``, ``KB_OLLAMA_TIMEOUT``, ``KB_EMBEDDING_DIM``, and
    ``KB_OLLAMA_KEEP_ALIVE`` into an :class:`~kb_core.config.EmbeddingConfig`. Every
    ``EmbeddingClient(...)`` construction site uses it so behavior stays
    identical across the move and the env surface stays centralized here.
    """
    from kb_core.config import EmbeddingConfig

    return EmbeddingConfig(
        ollama_url=get_ollama_url(),
        timeout=get_ollama_timeout(),
        model=get_embedding_model(),
        dim=get_embedding_dim(),
        keep_alive=get_embedding_keep_alive(),
    )


def is_agentic_query() -> bool:
    """Return True if agentic query planning is enabled (default: TRUE)."""
    return os.environ.get("KB_AGENTIC_QUERY", "TRUE").upper() == "TRUE"


def get_agentic_max_tool_calls() -> int:
    """Return max tool calls for agentic query loop from KB_AGENTIC_MAX_CALLS."""
    return _parse_int("KB_AGENTIC_MAX_CALLS", "4")


def is_agentic_synthesis() -> bool:
    """Return True if agentic synthesis is enabled (default: TRUE)."""
    return os.environ.get("KB_AGENTIC_SYNTHESIS", "TRUE").upper() == "TRUE"


def get_contributor() -> str | None:
    """Return the contributor name from KB_CONTRIBUTOR, or None."""
    return os.environ.get("KB_CONTRIBUTOR") or None


def get_team() -> str | None:
    """Return the team name from KB_TEAM, or None."""
    return os.environ.get("KB_TEAM") or None


def get_pg_pool_min() -> int:
    """Return the Postgres connection pool minimum size from KB_PG_POOL_MIN."""
    return _parse_int("KB_PG_POOL_MIN", "1")


def get_pg_pool_max() -> int:
    """Return the Postgres connection pool maximum size from KB_PG_POOL_MAX."""
    return _parse_int("KB_PG_POOL_MAX", "5")


def is_safety_skip() -> bool:
    """Return True if KB_SKIP_SAFETY is set to TRUE."""
    return os.environ.get("KB_SKIP_SAFETY", "").upper() == "TRUE"


def is_pg_iam_auth() -> bool:
    """Return True if KB_PG_IAM_AUTH is set to TRUE (RDS/Aurora IAM auth)."""
    return os.environ.get("KB_PG_IAM_AUTH", "").upper() == "TRUE"


def get_pg_region() -> str:
    """Return the AWS region for RDS IAM token signing from KB_PG_REGION."""
    return os.environ.get("KB_PG_REGION", "us-east-1")


LOCAL_KB_URL = "http://127.0.0.1:8765"
LOCAL_KB_API_KEY = "local-no-auth"


def get_personal_kb_url() -> str:
    """Return the personal-kb service URL: ``PERSONAL_KB_URL`` or the local default.

    An unset or empty variable means local mode: :data:`LOCAL_KB_URL`, served
    by the local kb-service daemon. Read at **call time**.
    """
    return os.environ.get("PERSONAL_KB_URL") or LOCAL_KB_URL


def get_personal_kb_api_key() -> str | None:
    """Return the personal-kb API key, or None.

    ``PERSONAL_KB_API_KEY`` wins when set. When unset/empty and the resolved
    URL is loopback, returns :data:`LOCAL_KB_API_KEY`; a remote URL with no
    key returns None so callers can fail closed. Read at **call time**.
    """
    key = os.environ.get("PERSONAL_KB_API_KEY")
    if key:
        return key
    from urllib.parse import urlparse

    host = (urlparse(get_personal_kb_url()).hostname or "").lower()
    if host in {"127.0.0.1", "localhost", "::1"}:
        return LOCAL_KB_API_KEY
    return None


def get_aws_profile() -> str | None:
    """Return the AWS profile name for Bedrock credentials.

    Resolution order:
    1. KB_AWS_PROFILE env var (explicit override)
    2. 'personal_kb_bedrock' if it exists in ~/.aws/credentials or config
    3. None (fall back to other auth methods)
    """
    explicit = os.environ.get("KB_AWS_PROFILE")
    if explicit:
        return explicit
    return None


_CONVENTION_PROFILE = "personal_kb_bedrock"


_BACKEND_STATE_FILE = Path("~/.local/share/personal_kb/backend_state.json").expanduser()

_backend_fallback_warning: str | None = None


def check_backend_fallback() -> None:
    """Check if the current backend differs from the last-known state.

    Reads/writes ``~/.local/share/personal_kb/backend_state.json``,
    keyed by instance role. Sets a module-level warning string if this
    instance was previously on Postgres but is now on SQLite.
    """
    global _backend_fallback_warning

    role = os.environ.get("KB_INSTANCE_ROLE", "").lower() or "default"
    current = "postgres" if get_database_url() else "sqlite"

    state: dict[str, str] = {}
    try:
        if _BACKEND_STATE_FILE.exists():
            state = json.loads(_BACKEND_STATE_FILE.read_text())
    except Exception:
        logging.getLogger(__name__).debug("Could not read backend state", exc_info=True)

    previous = state.get(role)
    if previous == "postgres" and current == "sqlite":
        logger = logging.getLogger(__name__)
        label = f"[{role}] " if role != "default" else ""
        _backend_fallback_warning = (
            f"WARNING: {label}KB fell back to local SQLite "
            f"(was Postgres). Check that KB_DATABASE_URL is set — "
            f"entries stored now will NOT appear in your Postgres KB."
        )
        logger.warning(_backend_fallback_warning)

    # Update state
    state[role] = current
    try:
        _BACKEND_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _BACKEND_STATE_FILE.write_text(json.dumps(state))
    except Exception:
        logging.getLogger(__name__).debug("Could not write backend state", exc_info=True)


def get_backend_warning() -> str | None:
    """Return the backend fallback warning, if any."""
    return _backend_fallback_warning


# ---------------------------------------------------------------------------
# kb_core config builders (channel-side env → dataclass adapters)
# ---------------------------------------------------------------------------
#
# These snapshots ``KB_*`` env vars into the explicit, env-free
# ``kb_core.config`` dataclasses. Every ``KnowledgeBase.create(...)`` call from
# the MCP channel flows through :func:`build_kb_config` so behavior across the
# move stays identical and the env surface stays centralized here.


def build_anthropic_config(*, model: str | None = None) -> "AnthropicProviderConfig":
    """Snapshot env-driven Anthropic config into the explicit dataclass."""
    from kb_core.config import AnthropicProviderConfig

    return AnthropicProviderConfig(
        model=model or get_anthropic_model(),
        timeout=get_anthropic_timeout(),
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )


def build_bedrock_config(*, model: str | None = None) -> "BedrockProviderConfig":
    """Snapshot env-driven Bedrock config into the explicit dataclass.

    Reads ``AWS_BEARER_TOKEN_BEDROCK`` and ``AWS_ACCESS_KEY_ID`` HERE (the
    channel) so kb_core never touches ``os.environ``. The bearer token
    string is captured; for env-credential auth we only pass a boolean
    flag — the smithy resolver reads the actual creds at use time.
    """
    from kb_core.config import BedrockProviderConfig

    return BedrockProviderConfig(
        model=model or get_bedrock_model(),
        timeout=get_bedrock_timeout(),
        region=get_bedrock_region(),
        profile=get_aws_profile(),
        bearer_token=os.environ.get("AWS_BEARER_TOKEN_BEDROCK"),
        has_env_credentials=bool(os.environ.get("AWS_ACCESS_KEY_ID")),
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

    Each role (extraction / query / synthesis) is a :class:`ProviderRoleConfig`
    that bundles which provider is active + credentials for all three providers
    (so switching providers requires no extra plumbing). The synthesis role
    points at the Sonnet model for Anthropic/Bedrock (mirroring the legacy
    ``_create_synthesis_llm`` behavior). For Ollama, the synthesis role keeps
    the default Ollama config; the channel passes ``synthesis_llm=None`` to
    ``KnowledgeBase.create`` so the engine doesn't build a separate synthesis
    client (matching today's "no Sonnet equivalent" fallback).
    """
    from kb_core.config import ProviderConfig, ProviderRoleConfig
    from kb_core.llm.anthropic import _SONNET_MODEL as _ANTHROPIC_SONNET
    from kb_core.llm.bedrock import _SONNET_MODEL as _BEDROCK_SONNET

    extraction_provider = get_extraction_provider()
    query_provider = get_query_provider()

    anthropic = build_anthropic_config()
    bedrock = build_bedrock_config()
    ollama = build_ollama_provider_config()

    # Synthesis role uses the same provider as query (today's behavior), but
    # with the Sonnet model for Anthropic/Bedrock. For Ollama the role config
    # stays as the default Ollama; the channel overrides ``synthesis_llm=None``
    # at create() time to preserve the "no Sonnet equivalent" behavior.
    synthesis_anthropic = build_anthropic_config(model=_ANTHROPIC_SONNET)
    synthesis_bedrock = build_bedrock_config(model=_BEDROCK_SONNET)

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


def build_attribution() -> "Attribution":
    """Snapshot ``KB_CONTRIBUTOR`` / ``KB_TEAM`` into the explicit dataclass."""
    from kb_core.config import Attribution

    return Attribution(contributor=get_contributor(), team=get_team())


def build_database_config(*, embedding_dim: int | None = None) -> "DatabaseConfig":
    """Build :class:`SqliteConfig` or :class:`PostgresConfig` from env.

    Dispatches on ``KB_DATABASE_URL`` — when set, returns a
    :class:`PostgresConfig`; otherwise a :class:`SqliteConfig` over
    ``KB_DB_PATH``.
    """
    from kb_core.config import PostgresConfig, SqliteConfig

    dim = embedding_dim if embedding_dim is not None else get_embedding_dim()
    db_url = get_database_url()
    if db_url:
        return PostgresConfig(
            dsn=db_url,
            embedding_dim=dim,
            pool_min=get_pg_pool_min(),
            pool_max=get_pg_pool_max(),
            iam_auth=is_pg_iam_auth(),
            region=get_pg_region(),
        )
    return SqliteConfig(path=get_db_path(), embedding_dim=dim)


def build_kb_config() -> "KbConfig":
    """Build a complete :class:`KbConfig` from this module's env getters.

    This is the single channel-side adapter that snapshots every ``KB_*`` env
    var the engine cares about into the explicit ``kb_core.config`` dataclasses
    so the MCP server can hand the engine a fully-typed configuration object.
    """
    from kb_core.config import KbConfig

    return KbConfig(
        database=build_database_config(),
        embedding=build_embedding_config(),
        providers=build_provider_config(),
        ingest=build_ingest_config(),
        agentic=build_agentic_config(),
        attribution=build_attribution(),
    )
