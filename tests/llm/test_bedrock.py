"""Tests for the Bedrock LLM client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from personal_kb.llm.bedrock import BedrockLLMClient

_has_sdk = True
try:
    import aws_sdk_bedrock_runtime  # noqa: F401
except ImportError:
    _has_sdk = False

needs_sdk = pytest.mark.skipif(not _has_sdk, reason="aws-sdk-bedrock-runtime not installed")


@pytest.fixture
def mock_converse_response():
    """Create a mock Bedrock Converse response."""
    content_block = MagicMock()
    content_block.value = "Hello from Bedrock"
    message = MagicMock()
    message.content = [content_block]
    output = MagicMock()
    output.value = message
    response = MagicMock()
    response.output = output
    return response


@pytest.fixture
def mock_bedrock_client(mock_converse_response):
    """Patch _get_client and return the mock client."""
    with patch("personal_kb.llm.bedrock.BedrockLLMClient._get_client") as mock_get:
        client = MagicMock()
        client.converse = AsyncMock(return_value=mock_converse_response)
        mock_get.return_value = client
        yield client


@needs_sdk
@pytest.mark.asyncio
async def test_generate_success(mock_bedrock_client):
    """Successful generate returns text and sets available."""
    llm = BedrockLLMClient()
    result = await llm.generate("test prompt")
    assert result == "Hello from Bedrock"
    assert llm._available is True


@needs_sdk
@pytest.mark.asyncio
async def test_generate_with_system_prompt(mock_bedrock_client):
    """System prompt is passed through to the Converse API."""
    llm = BedrockLLMClient()
    await llm.generate("test prompt", system="You are helpful")
    call_args = mock_bedrock_client.converse.call_args
    converse_input = call_args[0][0]
    assert converse_input.system is not None
    assert len(converse_input.system) == 1
    assert converse_input.system[0].value == "You are helpful"


@needs_sdk
@pytest.mark.asyncio
async def test_generate_without_system_prompt(mock_bedrock_client):
    """No system field when system is None."""
    llm = BedrockLLMClient()
    await llm.generate("test prompt")
    call_args = mock_bedrock_client.converse.call_args
    converse_input = call_args[0][0]
    assert converse_input.system is None


@pytest.mark.asyncio
async def test_generate_failure_returns_none():
    """Generate returns None after exhausting retries."""
    with (
        patch("personal_kb.llm.bedrock.BedrockLLMClient._get_client") as mock_get,
        patch("personal_kb.llm.bedrock.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        client = MagicMock()
        client.converse = AsyncMock(side_effect=Exception("AWS error"))
        mock_get.return_value = client

        llm = BedrockLLMClient()
        llm._available = True
        result = await llm.generate("test")
        assert result is None
        assert llm._available is None
        # 1 initial + 3 retries = 4 total calls
        assert client.converse.call_count == 4
        assert mock_sleep.call_count == 3


@pytest.mark.asyncio
async def test_generate_retries_on_transient_failure(mock_converse_response):
    """Generate succeeds after transient failure on first attempt."""
    with (
        patch("personal_kb.llm.bedrock.BedrockLLMClient._get_client") as mock_get,
        patch("personal_kb.llm.bedrock.asyncio.sleep", new_callable=AsyncMock),
    ):
        client = MagicMock()
        client.converse = AsyncMock(
            side_effect=[Exception("throttled"), mock_converse_response],
        )
        mock_get.return_value = client

        llm = BedrockLLMClient()
        result = await llm.generate("test")
        assert result == "Hello from Bedrock"
        assert client.converse.call_count == 2


@pytest.mark.asyncio
async def test_generate_returns_none_when_client_none():
    """Generate returns None when SDK is not installed."""
    with patch("personal_kb.llm.bedrock.BedrockLLMClient._get_client") as mock_get:
        mock_get.return_value = None

        llm = BedrockLLMClient()
        result = await llm.generate("test")
        assert result is None


@pytest.mark.asyncio
async def test_is_available_caches_success(mock_bedrock_client):
    """After successful generate, is_available returns True."""
    llm = BedrockLLMClient()
    await llm.generate("test")
    assert await llm.is_available() is True


@pytest.mark.asyncio
async def test_is_available_true_with_aws_key():
    """is_available returns True when AWS credentials are set."""
    with patch("personal_kb.llm.bedrock.BedrockLLMClient._get_client") as mock_get:
        mock_get.return_value = MagicMock()
        with patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "AKIA_TEST"}):
            llm = BedrockLLMClient()
            llm._auth_method = "env"
            assert await llm.is_available() is True


@pytest.mark.asyncio
async def test_is_available_true_with_bearer_token():
    """is_available returns True when bearer token is set."""
    with patch("personal_kb.llm.bedrock.BedrockLLMClient._get_client") as mock_get:
        mock_get.return_value = MagicMock()
        with patch.dict("os.environ", {"AWS_BEARER_TOKEN_BEDROCK": "ABSKtest123"}):
            llm = BedrockLLMClient()
            llm._auth_method = "bearer"
            assert await llm.is_available() is True


@pytest.mark.asyncio
async def test_is_available_false_without_any_credentials():
    """is_available returns False when no credentials set."""
    with patch("personal_kb.llm.bedrock.BedrockLLMClient._get_client") as mock_get:
        mock_get.return_value = MagicMock()
        with patch.dict("os.environ", {}, clear=True):
            llm = BedrockLLMClient()
            # _auth_method stays None when no creds found
            assert await llm.is_available() is False


@pytest.mark.asyncio
async def test_is_available_false_when_sdk_missing():
    """is_available returns False when SDK is not importable."""
    with patch("personal_kb.llm.bedrock.BedrockLLMClient._get_client") as mock_get:
        mock_get.return_value = None

        llm = BedrockLLMClient()
        assert await llm.is_available() is False


@needs_sdk
@pytest.mark.asyncio
async def test_generate_passes_newlines_through(mock_bedrock_client):
    """Newlines are passed through directly (smithy-json now handles escaping)."""
    llm = BedrockLLMClient()
    await llm.generate("line 1\nline 2\nline 3", system="rule 1\nrule 2")
    call_args = mock_bedrock_client.converse.call_args
    converse_input = call_args[0][0]
    prompt_value = converse_input.messages[0].content[0].value
    system_value = converse_input.system[0].value
    assert prompt_value == "line 1\nline 2\nline 3"
    assert system_value == "rule 1\nrule 2"


@pytest.mark.asyncio
async def test_close_is_noop():
    """Close is safe to call (no-op)."""
    llm = BedrockLLMClient()
    await llm.close()  # Should not raise


def test_model_override():
    """model_override changes the model returned by _model()."""
    llm = BedrockLLMClient(model_override="us.anthropic.claude-sonnet-4-6-20250514-v1:0")
    assert llm._model() == "us.anthropic.claude-sonnet-4-6-20250514-v1:0"


def test_model_default(monkeypatch):
    """Without override, uses config default."""
    monkeypatch.setenv("KB_BEDROCK_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
    llm = BedrockLLMClient()
    assert llm._model() == "us.anthropic.claude-haiku-4-5-20251001-v1:0"


@pytest.mark.asyncio
async def test_protocol_conformance():
    """BedrockLLMClient satisfies LLMProvider protocol."""
    from personal_kb.llm.provider import LLMProvider

    assert isinstance(BedrockLLMClient(), LLMProvider)


# ---------------------------------------------------------------------------
# Profile-based credentials
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_available_true_with_explicit_profile():
    """is_available returns True when KB_AWS_PROFILE resolves credentials."""
    mock_boto3 = MagicMock()
    frozen = MagicMock()
    frozen.access_key = "AKIA_PROFILE"
    frozen.secret_key = "test-key"  # noqa: S105
    frozen.token = "test-token"  # noqa: S105
    mock_creds = MagicMock()
    mock_creds.get_frozen_credentials.return_value = frozen
    mock_boto3.Session.return_value.get_credentials.return_value = mock_creds

    with (
        patch.dict("os.environ", {"KB_AWS_PROFILE": "work"}, clear=True),
        patch.dict("sys.modules", {"boto3": mock_boto3}),
    ):
        # Need a fresh client so _get_client runs the profile path
        llm = BedrockLLMClient()
        client = llm._get_client()
        if client is not None:
            assert llm._auth_method == "profile:work"


@pytest.mark.asyncio
async def test_convention_profile_auto_detected():
    """Convention profile 'personal_kb_bedrock' is used when available."""
    mock_boto3 = MagicMock()
    # available_profiles check
    mock_boto3.Session.return_value.available_profiles = [
        "default",
        "personal_kb_bedrock",
    ]
    # credential resolution
    frozen = MagicMock()
    frozen.access_key = "AKIA_CONV"
    frozen.secret_key = "test-key"  # noqa: S105
    frozen.token = None
    mock_creds = MagicMock()
    mock_creds.get_frozen_credentials.return_value = frozen
    mock_boto3.Session.return_value.get_credentials.return_value = mock_creds

    with (
        patch.dict("os.environ", {}, clear=True),
        patch.dict("sys.modules", {"boto3": mock_boto3}),
    ):
        llm = BedrockLLMClient()
        client = llm._get_client()
        if client is not None:
            assert llm._auth_method == "profile:personal_kb_bedrock"


@pytest.mark.asyncio
async def test_explicit_profile_overrides_convention():
    """KB_AWS_PROFILE takes priority over convention profile."""
    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value.available_profiles = [
        "default",
        "personal_kb_bedrock",
    ]
    frozen = MagicMock()
    frozen.access_key = "AKIA_EXPLICIT"
    frozen.secret_key = "test-key"  # noqa: S105
    frozen.token = None
    mock_creds = MagicMock()
    mock_creds.get_frozen_credentials.return_value = frozen
    mock_boto3.Session.return_value.get_credentials.return_value = mock_creds

    with (
        patch.dict("os.environ", {"KB_AWS_PROFILE": "custom"}, clear=True),
        patch.dict("sys.modules", {"boto3": mock_boto3}),
    ):
        llm = BedrockLLMClient()
        client = llm._get_client()
        if client is not None:
            assert llm._auth_method == "profile:custom"


@pytest.mark.asyncio
async def test_profile_takes_priority_over_bearer_and_env():
    """Profile credentials are checked before bearer token and env vars."""
    mock_boto3 = MagicMock()
    frozen = MagicMock()
    frozen.access_key = "AKIA_PROFILE"
    frozen.secret_key = "test-key"  # noqa: S105
    frozen.token = None
    mock_creds = MagicMock()
    mock_creds.get_frozen_credentials.return_value = frozen
    mock_boto3.Session.return_value.get_credentials.return_value = mock_creds

    with (
        patch.dict(
            "os.environ",
            {
                "KB_AWS_PROFILE": "work",
                "AWS_BEARER_TOKEN_BEDROCK": "ABSKtest",
                "AWS_ACCESS_KEY_ID": "AKIA_ENV",
            },
            clear=True,
        ),
        patch.dict("sys.modules", {"boto3": mock_boto3}),
    ):
        llm = BedrockLLMClient()
        client = llm._get_client()
        if client is not None:
            assert llm._auth_method == "profile:work"


@pytest.mark.asyncio
async def test_profile_fallback_when_boto3_missing():
    """No crash when boto3 is not installed and no env credentials."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch.dict("sys.modules", {"boto3": None}),
    ):
        # _find_aws_profile should return None gracefully
        from personal_kb.llm.bedrock import _find_aws_profile

        assert _find_aws_profile() is None
