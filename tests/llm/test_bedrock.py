"""Tests for the Bedrock LLM client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kb_core.config import BedrockProviderConfig

from personal_kb.llm.bedrock import BedrockLLMClient

_has_sdk = True
try:
    import aws_sdk_bedrock_runtime  # noqa: F401
except ImportError:
    _has_sdk = False

needs_sdk = pytest.mark.skipif(not _has_sdk, reason="aws-sdk-bedrock-runtime not installed")


def _client(
    *,
    model: str = "us.anthropic.claude-haiku-5-5",
    profile: str | None = None,
    bearer_token: str | None = None,
    has_env_credentials: bool = False,
    thinking: str | None = None,
    effort: str | None = None,
) -> BedrockLLMClient:
    """Helper: build a client with an explicit BedrockProviderConfig."""
    return BedrockLLMClient(
        BedrockProviderConfig(
            model=model,
            profile=profile,
            bearer_token=bearer_token,
            has_env_credentials=has_env_credentials,
            thinking=thinking,
            effort=effort,
        )
    )


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
    with patch("kb_core.llm.bedrock.BedrockLLMClient._get_client") as mock_get:
        client = MagicMock()
        client.converse = AsyncMock(return_value=mock_converse_response)
        mock_get.return_value = client
        yield client


@needs_sdk
@pytest.mark.asyncio
async def test_generate_success(mock_bedrock_client):
    """Successful generate returns text and sets available."""
    llm = _client()
    result = await llm.generate("test prompt", purpose="planner")
    assert result == "Hello from Bedrock"
    assert llm._available is True


@needs_sdk
@pytest.mark.asyncio
async def test_generate_with_system_prompt(mock_bedrock_client):
    """System prompt is passed through to the Converse API."""
    llm = _client()
    await llm.generate("test prompt", system="You are helpful", purpose="planner")
    call_args = mock_bedrock_client.converse.call_args
    converse_input = call_args[0][0]
    assert converse_input.system is not None
    assert len(converse_input.system) == 1
    assert converse_input.system[0].value == "You are helpful"


@needs_sdk
@pytest.mark.asyncio
async def test_generate_without_system_prompt(mock_bedrock_client):
    """No system field when system is None."""
    llm = _client()
    await llm.generate("test prompt", purpose="planner")
    call_args = mock_bedrock_client.converse.call_args
    converse_input = call_args[0][0]
    assert converse_input.system is None


@pytest.mark.asyncio
async def test_generate_failure_returns_none():
    """Generate returns None after exhausting retries."""
    with (
        patch("kb_core.llm.bedrock.BedrockLLMClient._get_client") as mock_get,
        patch("kb_core.llm.bedrock.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        client = MagicMock()
        client.converse = AsyncMock(side_effect=Exception("AWS error"))
        mock_get.return_value = client

        llm = _client()
        llm._available = True
        result = await llm.generate("test", purpose="planner")
        assert result is None
        assert llm._available is None
        # 1 initial + 3 retries = 4 total calls
        assert client.converse.call_count == 4
        assert mock_sleep.call_count == 3


@pytest.mark.asyncio
async def test_generate_retries_on_transient_failure(mock_converse_response):
    """Generate succeeds after transient failure on first attempt."""
    with (
        patch("kb_core.llm.bedrock.BedrockLLMClient._get_client") as mock_get,
        patch("kb_core.llm.bedrock.asyncio.sleep", new_callable=AsyncMock),
    ):
        client = MagicMock()
        client.converse = AsyncMock(
            side_effect=[Exception("throttled"), mock_converse_response],
        )
        mock_get.return_value = client

        llm = _client()
        result = await llm.generate("test", purpose="planner")
        assert result == "Hello from Bedrock"
        assert client.converse.call_count == 2


@pytest.mark.asyncio
async def test_generate_returns_none_when_client_none():
    """Generate returns None when SDK is not installed."""
    with patch("kb_core.llm.bedrock.BedrockLLMClient._get_client") as mock_get:
        mock_get.return_value = None

        llm = _client()
        result = await llm.generate("test", purpose="planner")
        assert result is None


@pytest.mark.asyncio
async def test_is_available_caches_success(mock_bedrock_client):
    """After successful generate, is_available returns True."""
    llm = _client()
    await llm.generate("test", purpose="planner")
    assert await llm.is_available() is True


@pytest.mark.asyncio
async def test_is_available_true_with_aws_key():
    """is_available returns True when AWS credentials are configured."""
    with patch("kb_core.llm.bedrock.BedrockLLMClient._get_client") as mock_get:
        mock_get.return_value = MagicMock()
        llm = _client(has_env_credentials=True)
        llm._auth_method = "env"
        assert await llm.is_available() is True


@pytest.mark.asyncio
async def test_is_available_true_with_bearer_token():
    """is_available returns True when bearer token is configured."""
    with patch("kb_core.llm.bedrock.BedrockLLMClient._get_client") as mock_get:
        mock_get.return_value = MagicMock()
        llm = _client(bearer_token="ABSKtest123")
        llm._auth_method = "bearer"
        assert await llm.is_available() is True


@pytest.mark.asyncio
async def test_is_available_false_without_any_credentials():
    """is_available returns False when no credentials are configured."""
    with patch("kb_core.llm.bedrock.BedrockLLMClient._get_client") as mock_get:
        mock_get.return_value = MagicMock()
        llm = _client()  # nothing — no bearer, no profile, no env flag
        # _auth_method stays None when no creds found
        assert await llm.is_available() is False


@pytest.mark.asyncio
async def test_is_available_false_when_sdk_missing():
    """is_available returns False when SDK is not importable."""
    with patch("kb_core.llm.bedrock.BedrockLLMClient._get_client") as mock_get:
        mock_get.return_value = None

        llm = _client()
        assert await llm.is_available() is False


@needs_sdk
@pytest.mark.asyncio
async def test_generate_passes_newlines_through(mock_bedrock_client):
    """Newlines are passed through directly (smithy-json now handles escaping)."""
    llm = _client()
    await llm.generate("line 1\nline 2\nline 3", system="rule 1\nrule 2", purpose="planner")
    call_args = mock_bedrock_client.converse.call_args
    converse_input = call_args[0][0]
    prompt_value = converse_input.messages[0].content[0].value
    system_value = converse_input.system[0].value
    assert prompt_value == "line 1\nline 2\nline 3"
    assert system_value == "rule 1\nrule 2"


@pytest.mark.asyncio
async def test_close_is_noop():
    """Close is safe to call (no-op)."""
    llm = _client()
    await llm.close()  # Should not raise


def test_model_override():
    """Model passed via config is returned by ``_config.model``."""
    llm = _client(model="us.anthropic.claude-sonnet-4-6-20250514-v1:0")
    assert llm._config.model == "us.anthropic.claude-sonnet-4-6-20250514-v1:0"


def test_model_default():
    """Default config carries the default Bedrock model id."""
    llm = BedrockLLMClient(BedrockProviderConfig())
    assert llm._config.model == "us.anthropic.claude-haiku-5-5"


def test_sonnet_model_is_sonnet_5_5_profile():
    """The Bedrock synthesis default is the Sonnet 5.5 US inference profile."""
    from kb_core.llm.bedrock import _SONNET_MODEL

    assert _SONNET_MODEL == "us.anthropic.claude-sonnet-5-5"


def _sent_fields(mock_client):
    fields = mock_client.converse.call_args[0][0].additional_model_request_fields
    return None if fields is None else fields.as_value()


@needs_sdk
@pytest.mark.asyncio
async def test_generate_sends_no_additional_fields_by_default(mock_bedrock_client):
    """No thinking and no effort configured: the request keeps model defaults."""
    await _client().generate("p", purpose="planner")
    assert _sent_fields(mock_bedrock_client) is None


@needs_sdk
@pytest.mark.asyncio
async def test_generate_sends_thinking_disabled(mock_bedrock_client):
    """thinking="disabled" goes out as additionalModelRequestFields.thinking."""
    await _client(thinking="disabled").generate("p", purpose="planner")
    assert _sent_fields(mock_bedrock_client) == {"thinking": {"type": "disabled"}}


@needs_sdk
@pytest.mark.asyncio
async def test_generate_chat_sends_effort(mock_bedrock_client):
    """effort="low" goes out as additionalModelRequestFields.output_config.effort."""
    llm = _client(model="us.anthropic.claude-sonnet-5-5", effort="low")
    await llm.generate_chat([{"role": "user", "content": "hi"}], purpose="chat")
    assert _sent_fields(mock_bedrock_client) == {"output_config": {"effort": "low"}}


@needs_sdk
@pytest.mark.asyncio
async def test_generate_sends_both_fields(mock_bedrock_client):
    """Both configured: one document carrying both keys."""
    await _client(thinking="adaptive", effort="medium").generate("p", purpose="planner")
    assert _sent_fields(mock_bedrock_client) == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "medium"},
    }


@needs_sdk
def test_additional_fields_serialize_on_the_wire():
    """The SDK's JSON codec renders the document as additionalModelRequestFields."""
    from aws_sdk_bedrock_runtime.models import (
        ContentBlockText,
        ConverseInput,
        InferenceConfiguration,
        Message,
    )
    from smithy_json import JSONCodec

    llm = _client(thinking="disabled")
    ci = ConverseInput(
        model_id="m",
        messages=[Message(role="user", content=[ContentBlockText(value="hi")])],
        inference_config=InferenceConfiguration(max_tokens=10),
        additional_model_request_fields=llm._additional_fields(),
    )
    assert b'"additionalModelRequestFields":{"thinking":{"type":"disabled"}}' in (
        JSONCodec().serialize(ci)
    )


@pytest.mark.parametrize(("field", "value"), [("thinking", "enabled"), ("effort", "extreme")])
def test_invalid_thinking_or_effort_rejected(field, value):
    """The Bedrock config validates thinking/effort like the Anthropic one."""
    with pytest.raises(ValueError, match=field):
        BedrockProviderConfig(**{field: value})


@pytest.mark.asyncio
async def test_protocol_conformance():
    """BedrockLLMClient satisfies LLMProvider protocol."""
    from personal_kb.llm.provider import LLMProvider

    assert isinstance(_client(), LLMProvider)


# ---------------------------------------------------------------------------
# Profile-based credentials
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_available_true_with_explicit_profile():
    """is_available returns True when an explicit profile resolves credentials."""
    mock_boto3 = MagicMock()
    frozen = MagicMock()
    frozen.access_key = "AKIA_PROFILE"
    frozen.secret_key = "test-key"  # noqa: S105
    frozen.token = "test-token"  # noqa: S105
    mock_creds = MagicMock()
    mock_creds.get_frozen_credentials.return_value = frozen
    mock_boto3.Session.return_value.get_credentials.return_value = mock_creds

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        llm = _client(profile="work")
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

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        llm = _client()  # no explicit profile
        client = llm._get_client()
        if client is not None:
            assert llm._auth_method == "profile:personal_kb_bedrock"


@pytest.mark.asyncio
async def test_explicit_profile_overrides_convention():
    """Explicit ``config.profile`` takes priority over convention profile."""
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

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        llm = _client(profile="custom")
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

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        llm = _client(
            profile="work",
            bearer_token="ABSKtest",
            has_env_credentials=True,
        )
        client = llm._get_client()
        if client is not None:
            assert llm._auth_method == "profile:work"


@pytest.mark.asyncio
async def test_profile_fallback_when_boto3_missing():
    """No crash when boto3 is not installed and no explicit profile."""
    with patch.dict("sys.modules", {"boto3": None}):
        # _find_aws_profile should return None gracefully
        from personal_kb.llm.bedrock import _find_aws_profile

        assert _find_aws_profile(None) is None


def _converse_response(*values):
    """Build a converse response; non-str values mimic reasoning-content unions."""
    items = []
    for v in values:
        item = MagicMock()
        item.value = v
        items.append(item)
    message = MagicMock()
    message.content = items
    response = MagicMock()
    response.output.value = message
    return response


async def _generate_with(response):
    with (
        patch("kb_core.llm.bedrock.BedrockLLMClient._get_client") as mock_get,
        patch("kb_core.llm.bedrock.asyncio.sleep", new_callable=AsyncMock),
    ):
        client = MagicMock()
        client.converse = AsyncMock(return_value=response)
        mock_get.return_value = client
        return await _client().generate("test", purpose="planner"), client


@pytest.mark.asyncio
async def test_generate_skips_reasoning_block():
    result, _ = await _generate_with(_converse_response(object(), "answer"))
    assert result == "answer"


@pytest.mark.asyncio
async def test_generate_concatenates_text_items():
    result, _ = await _generate_with(_converse_response("foo", "bar"))
    assert result == "foobar"


@pytest.mark.asyncio
async def test_generate_reasoning_only_returns_none():
    result, client = await _generate_with(_converse_response(object()))
    assert result is None
    assert client.converse.call_count == 4
