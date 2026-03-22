"""Environment-variable-based configuration."""

import json
import logging
import os
from pathlib import Path

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


def get_ollama_url() -> str:
    """Return the Ollama API URL from KB_OLLAMA_URL."""
    return os.environ.get("KB_OLLAMA_URL", "http://localhost:11434")


def get_embedding_model() -> str:
    """Return the embedding model name from KB_EMBEDDING_MODEL."""
    return os.environ.get("KB_EMBEDDING_MODEL", "qwen3-embedding:0.6b")


def get_ollama_timeout() -> float:
    """Return the Ollama timeout in seconds from KB_OLLAMA_TIMEOUT."""
    return _parse_float("KB_OLLAMA_TIMEOUT", "10.0")


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


def is_auto_explore() -> bool:
    """Return True if KB_AUTO_EXPLORE is set to TRUE (default: TRUE)."""
    return os.environ.get("KB_AUTO_EXPLORE", "TRUE").upper() == "TRUE"


def get_explore_port() -> int:
    """Return the explorer web server port from KB_EXPLORE_PORT (default: 8765)."""
    return _parse_int("KB_EXPLORE_PORT", "8765")


def is_pg_iam_auth() -> bool:
    """Return True if KB_PG_IAM_AUTH is set to TRUE (RDS/Aurora IAM auth)."""
    return os.environ.get("KB_PG_IAM_AUTH", "").upper() == "TRUE"


def get_pg_region() -> str:
    """Return the AWS region for RDS IAM token signing from KB_PG_REGION."""
    return os.environ.get("KB_PG_REGION", "us-east-1")


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
