"""Tests for backend fallback warning."""

import json

from personal_kb.config import (
    check_backend_fallback,
    get_backend_warning,
)


class TestBackendFallbackWarning:
    def test_no_warning_on_first_start(self, tmp_path, monkeypatch):
        """First start — no state file, no warning."""
        state_file = tmp_path / "backend_state.json"
        monkeypatch.setattr("personal_kb.config._BACKEND_STATE_FILE", state_file)
        monkeypatch.setattr("personal_kb.config._backend_fallback_warning", None)
        monkeypatch.delenv("KB_DATABASE_URL", raising=False)

        check_backend_fallback()
        assert get_backend_warning() is None
        assert state_file.exists()
        assert json.loads(state_file.read_text()) == {"default": "sqlite"}

    def test_no_warning_postgres_to_postgres(self, tmp_path, monkeypatch):
        """Staying on Postgres — no warning."""
        state_file = tmp_path / "backend_state.json"
        state_file.write_text(json.dumps({"default": "postgres"}))
        monkeypatch.setattr("personal_kb.config._BACKEND_STATE_FILE", state_file)
        monkeypatch.setattr("personal_kb.config._backend_fallback_warning", None)
        monkeypatch.setenv("KB_DATABASE_URL", "postgresql://localhost/kb")

        check_backend_fallback()
        assert get_backend_warning() is None

    def test_warning_on_postgres_to_sqlite(self, tmp_path, monkeypatch):
        """Falling back from Postgres to SQLite — warning set."""
        state_file = tmp_path / "backend_state.json"
        state_file.write_text(json.dumps({"default": "postgres"}))
        monkeypatch.setattr("personal_kb.config._BACKEND_STATE_FILE", state_file)
        monkeypatch.setattr("personal_kb.config._backend_fallback_warning", None)
        monkeypatch.delenv("KB_DATABASE_URL", raising=False)

        check_backend_fallback()
        warning = get_backend_warning()
        assert warning is not None
        assert "fell back to local SQLite" in warning
        assert "KB_DATABASE_URL" in warning

    def test_no_warning_sqlite_to_sqlite(self, tmp_path, monkeypatch):
        """Staying on SQLite — no warning."""
        state_file = tmp_path / "backend_state.json"
        state_file.write_text(json.dumps({"default": "sqlite"}))
        monkeypatch.setattr("personal_kb.config._BACKEND_STATE_FILE", state_file)
        monkeypatch.setattr("personal_kb.config._backend_fallback_warning", None)
        monkeypatch.delenv("KB_DATABASE_URL", raising=False)

        check_backend_fallback()
        assert get_backend_warning() is None

    def test_per_instance_tracking(self, tmp_path, monkeypatch):
        """Each instance role tracked independently."""
        state_file = tmp_path / "backend_state.json"
        state_file.write_text(json.dumps({"personal": "postgres", "team": "postgres"}))
        monkeypatch.setattr("personal_kb.config._BACKEND_STATE_FILE", state_file)
        monkeypatch.setattr("personal_kb.config._backend_fallback_warning", None)
        monkeypatch.setenv("KB_INSTANCE_ROLE", "personal")
        monkeypatch.delenv("KB_DATABASE_URL", raising=False)

        check_backend_fallback()
        warning = get_backend_warning()
        assert warning is not None
        assert "[personal]" in warning

        # team should still be recorded as postgres
        state = json.loads(state_file.read_text())
        assert state["personal"] == "sqlite"
        assert state["team"] == "postgres"

    def test_no_warning_sqlite_to_postgres(self, tmp_path, monkeypatch):
        """Upgrading from SQLite to Postgres — no warning."""
        state_file = tmp_path / "backend_state.json"
        state_file.write_text(json.dumps({"default": "sqlite"}))
        monkeypatch.setattr("personal_kb.config._BACKEND_STATE_FILE", state_file)
        monkeypatch.setattr("personal_kb.config._backend_fallback_warning", None)
        monkeypatch.setenv("KB_DATABASE_URL", "postgresql://localhost/kb")

        check_backend_fallback()
        assert get_backend_warning() is None
        state = json.loads(state_file.read_text())
        assert state["default"] == "postgres"
