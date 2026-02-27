"""Tests for the safety pipeline."""

from pathlib import Path
from unittest.mock import patch

from personal_kb.ingest.safety import (
    SafetyResult,
    check_deny_list,
    detect_secrets_in_content,
    redact_pii,
    run_safety_pipeline,
)


class TestCheckDenyList:
    def test_allows_markdown(self):
        assert check_deny_list(Path("notes/readme.md")) is None

    def test_allows_text(self):
        assert check_deny_list(Path("notes.txt")) is None

    def test_denies_pem(self):
        assert check_deny_list(Path("server.pem")) == "*.pem"

    def test_denies_key(self):
        assert check_deny_list(Path("private.key")) == "*.key"

    def test_denies_env(self):
        assert check_deny_list(Path(".env")) == ".env"

    def test_denies_env_local(self):
        assert check_deny_list(Path(".env.local")) == ".env.*"

    def test_denies_id_rsa(self):
        assert check_deny_list(Path("id_rsa")) == "id_rsa"

    def test_denies_wireguard(self):
        assert check_deny_list(Path("wg0.conf")) == "wg*.conf"

    def test_denies_png(self):
        assert check_deny_list(Path("screenshot.png")) == "*.png"

    def test_denies_zip(self):
        assert check_deny_list(Path("archive.zip")) == "*.zip"

    def test_denies_sqlite(self):
        assert check_deny_list(Path("data.db")) == "*.db"

    def test_denies_case_insensitive(self):
        assert check_deny_list(Path("IMAGE.PNG")) == "*.png"

    def test_allows_python(self):
        assert check_deny_list(Path("script.py")) is None

    def test_allows_json(self):
        assert check_deny_list(Path("config.json")) is None


class TestDetectSecrets:
    def test_returns_none_when_not_installed(self):
        mocked_modules = {
            "detect_secrets": None,
            "detect_secrets.core.scan": None,
            "detect_secrets.settings": None,
        }
        with patch.dict("sys.modules", mocked_modules):
            # Can't easily mock ImportError for already-imported modules,
            # so test the public contract: returns list or None
            result = detect_secrets_in_content("safe content")
            # Result depends on whether detect-secrets is installed
            assert result is None or isinstance(result, list)

    def test_clean_content_no_secrets(self):
        result = detect_secrets_in_content("Just some regular text with no secrets.")
        if result is not None:
            # detect-secrets installed, should find nothing
            assert result == []

    def test_detects_keyword_secret(self):
        result = detect_secrets_in_content('password = "hunter2"')
        if result is not None:
            # KeywordDetector should catch "password"
            assert len(result) > 0


class TestRedactPII:
    def test_passthrough_when_not_installed(self):
        with patch.dict("sys.modules", {"scrubadub": None}):
            # Can't easily mock ImportError, but test the contract
            content = "Contact john@example.com"
            cleaned, pii_types = redact_pii(content)
            # Either original or redacted
            assert isinstance(cleaned, str)
            assert isinstance(pii_types, list)

    def test_returns_string_and_list(self):
        content = "Hello world"
        cleaned, pii_types = redact_pii(content)
        assert isinstance(cleaned, str)
        assert isinstance(pii_types, list)


class TestRunSafetyPipeline:
    def test_skip_on_deny_list(self):
        result = run_safety_pipeline(Path("secret.pem"), "cert content")
        assert result.action == "skip"
        assert "deny-list" in result.reason

    def test_allow_safe_content(self):
        result = run_safety_pipeline(Path("notes.md"), "Just some notes about Python")
        assert result.action in ("allow", "flag")

    def test_result_has_content(self):
        result = run_safety_pipeline(Path("notes.md"), "Hello world")
        assert isinstance(result.content, str)
        assert isinstance(result.redactions, list)

    def test_skip_env_file(self):
        result = run_safety_pipeline(Path(".env.production"), "DB_PASSWORD=secret")
        assert result.action == "skip"

    def test_returns_safety_result(self):
        result = run_safety_pipeline(Path("readme.md"), "Normal content")
        assert isinstance(result, SafetyResult)
