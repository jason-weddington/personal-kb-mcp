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


def get_ingest_max_file_size() -> int:
    """Return max file size in bytes for ingestion from KB_INGEST_MAX_FILE_SIZE."""
    return int(os.environ.get("KB_INGEST_MAX_FILE_SIZE", str(500 * 1024)))
