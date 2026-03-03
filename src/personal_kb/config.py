"""Environment-variable-based configuration."""

import os
from pathlib import Path


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
    return float(os.environ.get("KB_OLLAMA_TIMEOUT", "10.0"))


def is_manager_mode() -> bool:
    """Return True if KB_MANAGER is set to TRUE."""
    return os.environ.get("KB_MANAGER", "").upper() == "TRUE"


def get_embedding_dim() -> int:
    """Return the embedding vector dimensions from KB_EMBEDDING_DIM."""
    return int(os.environ.get("KB_EMBEDDING_DIM", "1024"))


def get_llm_model() -> str:
    """Return the Ollama LLM model name from KB_OLLAMA_MODEL."""
    return os.environ.get("KB_OLLAMA_MODEL", "qwen3:4b")


def get_llm_timeout() -> float:
    """Return the Ollama LLM timeout in seconds from KB_OLLAMA_LLM_TIMEOUT."""
    return float(os.environ.get("KB_OLLAMA_LLM_TIMEOUT", "120.0"))


def get_anthropic_model() -> str:
    """Return the Anthropic model name from KB_ANTHROPIC_MODEL."""
    return os.environ.get("KB_ANTHROPIC_MODEL", "claude-haiku-4-5")


def get_anthropic_timeout() -> float:
    """Return the Anthropic timeout in seconds from KB_ANTHROPIC_TIMEOUT."""
    return float(os.environ.get("KB_ANTHROPIC_TIMEOUT", "30.0"))


def get_extraction_provider() -> str:
    """Return the LLM provider for graph extraction from KB_EXTRACTION_PROVIDER."""
    return os.environ.get("KB_EXTRACTION_PROVIDER", "anthropic")


def get_query_provider() -> str:
    """Return the LLM provider for query planning from KB_QUERY_PROVIDER."""
    return os.environ.get("KB_QUERY_PROVIDER", "anthropic")


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
    return float(os.environ.get("KB_BEDROCK_TIMEOUT", "30.0"))


def get_database_url() -> str | None:
    """Return database URL if set, None for SQLite file-based.

    When set to a ``postgresql://`` URL, the server will create a
    PostgresBackend instead of SQLiteBackend.
    """
    return os.environ.get("KB_DATABASE_URL")


def get_ingest_max_file_size() -> int:
    """Return max file size in bytes for ingestion from KB_INGEST_MAX_FILE_SIZE."""
    return int(os.environ.get("KB_INGEST_MAX_FILE_SIZE", str(5 * 1024 * 1024)))


def get_ingest_chunk_size() -> int:
    """Return chunk size in chars for ingestion from KB_INGEST_CHUNK_SIZE."""
    return int(os.environ.get("KB_INGEST_CHUNK_SIZE", "16000"))


def get_ingest_chunk_overlap() -> int:
    """Return chunk overlap in chars for ingestion from KB_INGEST_CHUNK_OVERLAP."""
    return int(os.environ.get("KB_INGEST_CHUNK_OVERLAP", "600"))


def is_agentic_ingest() -> bool:
    """Return True if agentic ingestion dedup is enabled (default: TRUE)."""
    return os.environ.get("KB_AGENTIC_INGEST", "TRUE").upper() == "TRUE"


def get_ingest_dedup_threshold() -> float:
    """Return the hybrid search score threshold for dedup from KB_INGEST_DEDUP_THRESHOLD."""
    return float(os.environ.get("KB_INGEST_DEDUP_THRESHOLD", "0.06"))


def is_agentic_query() -> bool:
    """Return True if agentic query planning is enabled (default: TRUE)."""
    return os.environ.get("KB_AGENTIC_QUERY", "TRUE").upper() == "TRUE"


def get_agentic_max_tool_calls() -> int:
    """Return max tool calls for agentic query loop from KB_AGENTIC_MAX_CALLS."""
    return int(os.environ.get("KB_AGENTIC_MAX_CALLS", "4"))


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
    return int(os.environ.get("KB_PG_POOL_MIN", "1"))


def get_pg_pool_max() -> int:
    """Return the Postgres connection pool maximum size from KB_PG_POOL_MAX."""
    return int(os.environ.get("KB_PG_POOL_MAX", "5"))


def is_safety_skip() -> bool:
    """Return True if KB_SKIP_SAFETY is set to TRUE."""
    return os.environ.get("KB_SKIP_SAFETY", "").upper() == "TRUE"


def is_pg_iam_auth() -> bool:
    """Return True if KB_PG_IAM_AUTH is set to TRUE (RDS/Aurora IAM auth)."""
    return os.environ.get("KB_PG_IAM_AUTH", "").upper() == "TRUE"


def get_pg_region() -> str:
    """Return the AWS region for RDS IAM token signing from KB_PG_REGION."""
    return os.environ.get("KB_PG_REGION", "us-east-1")
