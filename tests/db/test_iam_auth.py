"""Tests for RDS/Aurora IAM authentication helpers."""

import ssl
import sys
from unittest.mock import MagicMock, patch

import pytest

from personal_kb.db.iam_auth import DSNComponents, make_ssl_context, make_token_factory, parse_dsn


class TestParseDSN:
    """Test DSN parsing from PostgreSQL connection URLs."""

    def test_full_dsn(self):
        url = "postgresql://myuser:pass@myhost.rds.amazonaws.com:5432/mydb"
        result = parse_dsn(url)
        assert result == DSNComponents(
            host="myhost.rds.amazonaws.com", port=5432, username="myuser"
        )

    def test_default_port(self):
        url = "postgresql://myuser@myhost.rds.amazonaws.com/mydb"
        result = parse_dsn(url)
        assert result.port == 5432

    def test_default_database(self):
        url = "postgresql://myuser@myhost.rds.amazonaws.com:5432"
        result = parse_dsn(url)
        assert result.host == "myhost.rds.amazonaws.com"
        assert result.username == "myuser"

    def test_missing_hostname_raises(self):
        with pytest.raises(ValueError, match="missing hostname"):
            parse_dsn("postgresql:///mydb")

    def test_missing_username_raises(self):
        with pytest.raises(ValueError, match="missing username"):
            parse_dsn("postgresql://myhost.rds.amazonaws.com/mydb")

    def test_dsn_with_password_still_parses(self):
        url = "postgresql://admin:secretpass@aurora.cluster-xxx.us-east-1.rds.amazonaws.com:5432/kb"
        result = parse_dsn(url)
        assert result.username == "admin"
        assert result.host == "aurora.cluster-xxx.us-east-1.rds.amazonaws.com"

    def test_query_params_ignored(self):
        url = "postgresql://myuser@myhost:5432/mydb?sslmode=require&connect_timeout=10"
        result = parse_dsn(url)
        assert result.host == "myhost"
        assert result.username == "myuser"


def _make_mock_boto3(return_values=None):
    """Create a mock boto3 module with RDS client."""
    if return_values is None:
        return_values = "token-abc"
    mock_client = MagicMock()
    if isinstance(return_values, list):
        mock_client.generate_db_auth_token.side_effect = return_values
    else:
        mock_client.generate_db_auth_token.return_value = return_values
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_client
    return mock_boto3, mock_client


class TestMakeTokenFactory:
    """Test IAM token factory creation."""

    def test_creates_callable(self):
        mock_boto3, _mock_client = _make_mock_boto3()
        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            factory = make_token_factory("myhost", 5432, "myuser", "us-east-1")
        assert callable(factory)
        mock_boto3.client.assert_called_once_with("rds", region_name="us-east-1")

    def test_reuses_client_across_calls(self):
        mock_boto3, mock_client = _make_mock_boto3()
        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            factory = make_token_factory("myhost", 5432, "myuser", "us-east-1")
        factory()
        factory()

        # Client created once, token generated twice
        mock_boto3.client.assert_called_once()
        assert mock_client.generate_db_auth_token.call_count == 2

    def test_returns_fresh_token_each_call(self):
        mock_boto3, _mock_client = _make_mock_boto3(["token-1", "token-2"])
        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            factory = make_token_factory("myhost", 5432, "myuser", "us-east-1")
        assert factory() == "token-1"
        assert factory() == "token-2"

    def test_passes_correct_params(self):
        mock_boto3, mock_client = _make_mock_boto3("token")
        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            factory = make_token_factory(
                "aurora.cluster-xxx.rds.amazonaws.com", 5433, "admin", "eu-west-1"
            )
        factory()

        mock_client.generate_db_auth_token.assert_called_once_with(
            DBHostname="aurora.cluster-xxx.rds.amazonaws.com",
            Port=5433,
            DBUsername="admin",
            Region="eu-west-1",
        )


class TestMakeSSLContext:
    """Test SSL context creation for IAM auth."""

    def test_returns_ssl_context(self):
        ctx = make_ssl_context()
        assert isinstance(ctx, ssl.SSLContext)

    def test_verify_mode_cert_required(self):
        ctx = make_ssl_context()
        assert ctx.verify_mode == ssl.CERT_REQUIRED

    def test_check_hostname_enabled(self):
        ctx = make_ssl_context()
        assert ctx.check_hostname is True
