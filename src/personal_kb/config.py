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


def get_log_level() -> str:
    """Return the logging level from KB_LOG_LEVEL."""
    return os.environ.get("KB_LOG_LEVEL", "WARNING")
