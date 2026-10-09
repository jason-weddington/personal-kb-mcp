"""AWS Bedrock LLM client with graceful degradation.

Channel-agnostic: takes an :class:`~kb_core.config.BedrockProviderConfig`
at construction time. No ``os.environ`` reads inside this module — the
channel snapshots ``AWS_BEARER_TOKEN_BEDROCK`` and the presence of
``AWS_ACCESS_KEY_ID`` into the config. The smithy ``Environment-
CredentialsResolver`` may still read AWS env vars at use time; that
read happens inside smithy_aws_core, not in kb_core.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kb_core.config import BedrockProviderConfig
    from kb_core.llm.provider import Message

logger = logging.getLogger(__name__)


_CONVENTION_PROFILE = "personal_kb_bedrock"
"""Auto-detected AWS profile name when ``config.profile`` is unset."""


def _resolve_profile_credentials(profile_name: str) -> dict[str, str] | None:
    """Resolve AWS credentials from a named profile via boto3.

    Returns a dict with ``access_key``, ``secret_key``, and optionally
    ``token``, or ``None`` if the profile doesn't exist or credentials
    can't be resolved.
    """
    try:
        import boto3
    except ImportError:
        logger.debug("boto3 not installed — cannot resolve profile '%s'", profile_name)
        return None

    try:
        session = boto3.Session(profile_name=profile_name)
        creds = session.get_credentials()
        if creds is None:
            return None
        frozen = creds.get_frozen_credentials()
        result: dict[str, str] = {
            "access_key": frozen.access_key,
            "secret_key": frozen.secret_key,
        }
        if frozen.token:
            result["token"] = frozen.token
        return result
    except Exception:
        logger.debug("Failed to resolve credentials for profile '%s'", profile_name, exc_info=True)
        return None


def _find_aws_profile(explicit_profile: str | None) -> str | None:
    """Find the AWS profile to use for Bedrock credentials.

    Resolution order:
    1. ``explicit_profile`` (from :class:`BedrockProviderConfig`)
    2. :data:`_CONVENTION_PROFILE` if it exists in available profiles
    3. ``None``
    """
    if explicit_profile:
        return explicit_profile

    # Check if convention profile exists
    try:
        import boto3

        if _CONVENTION_PROFILE in boto3.Session().available_profiles:
            logger.info("Bedrock: found convention profile '%s'", _CONVENTION_PROFILE)
            return _CONVENTION_PROFILE
    except ImportError:
        pass
    except Exception:
        logger.debug("Failed to check available profiles", exc_info=True)

    return None


def _configure_bearer_auth(config: Any, bearer_token: str) -> None:
    """Add bearer token auth scheme to a Bedrock Config object.

    Monkey-patches the generated Config to support ``httpBearerAuth``,
    which the service model declares but the codegen doesn't wire up
    yet. Uses the existing smithy_http APIKeyAuthScheme plumbing with
    ``Authorization: Bearer``. The bearer token is captured here from
    the caller (the channel's snapshot of ``AWS_BEARER_TOKEN_BEDROCK``);
    we do not read env inside this module.
    """
    from smithy_core.auth import AuthOption
    from smithy_core.shapes import ShapeID
    from smithy_core.traits import APIKeyLocation
    from smithy_http.aio.auth.apikey import APIKeyAuthScheme
    from smithy_http.aio.identity.apikey import (
        APIKeyIdentity,
        APIKeyIdentityProperties,
        APIKeyIdentityResolver,
    )

    bearer_scheme_id = ShapeID("smithy.api#httpBearerAuth")

    # --- Auth scheme: signs requests with Authorization: Bearer <token> ---
    class BearerAuthScheme(APIKeyAuthScheme):
        scheme_id = bearer_scheme_id

        def __init__(self) -> None:
            super().__init__(
                name="Authorization",
                location=APIKeyLocation.HEADER,
                scheme="Bearer",
            )

        def identity_properties(self, *, context: Any) -> APIKeyIdentityProperties:
            return {"api_key": bearer_token}

        def identity_resolver(self, *, context: Any) -> APIKeyIdentityResolver:
            return _StaticBearerTokenResolver()

    # --- Identity resolver: returns the captured token ---
    class _StaticBearerTokenResolver(APIKeyIdentityResolver):
        async def get_identity(self, *, properties: APIKeyIdentityProperties) -> APIKeyIdentity:
            token = properties.get("api_key") or bearer_token
            if not token:
                from smithy_core.exceptions import SmithyIdentityError

                raise SmithyIdentityError("Bedrock bearer token not configured")
            return APIKeyIdentity(api_key=token)

    # --- Inject into Config ---
    config.auth_schemes[bearer_scheme_id] = BearerAuthScheme()

    # Patch the resolver to prefer bearer auth when token is available
    original_resolve = config.auth_scheme_resolver.resolve_auth_scheme

    def patched_resolve(auth_parameters: Any) -> list[Any]:
        options: list[Any] = original_resolve(auth_parameters)
        # Prepend bearer option so it's tried first
        bearer_option = AuthOption(
            scheme_id=bearer_scheme_id,
            identity_properties={},  # type: ignore[arg-type]
            signer_properties={},  # type: ignore[arg-type]
        )
        options.insert(0, bearer_option)
        return options

    config.auth_scheme_resolver.resolve_auth_scheme = patched_resolve


_SONNET_MODEL = "us.anthropic.claude-sonnet-4-6"
"""Default Bedrock model identifier for the human-facing synthesis role."""

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds — exponential: 1s, 2s, 4s


class BedrockLLMClient:
    """Generates text via the AWS Bedrock Converse API."""

    def __init__(self, config: BedrockProviderConfig) -> None:
        """Initialize with an explicit provider config."""
        self._config = config
        self._client: Any = None
        self._available: bool | None = None
        self._auth_method: str | None = None

    async def is_available(self) -> bool:
        """Check availability. Only caches success — retries on failure."""
        if self._available is True:
            return True
        try:
            client = self._get_client()
            if client is None:
                return False
            if self._auth_method is None:
                logger.warning(
                    "No AWS credentials found (checked profile, "
                    "personal_kb_bedrock profile, bearer token, env credentials)"
                    " — Bedrock LLM disabled",
                )
                return False
            return True
        except Exception:
            return False

    async def generate(self, prompt: str, *, system: str | None = None) -> str | None:
        """Generate text from a prompt. Returns None if unavailable."""
        try:
            client = self._get_client()
            if client is None:
                return None

            from aws_sdk_bedrock_runtime.models import (
                ContentBlockText,
                ConverseInput,
                InferenceConfiguration,
                Message,
                SystemContentBlockText,
            )

            converse_input = ConverseInput(
                model_id=self._config.model,
                messages=[
                    Message(
                        role="user",
                        content=[ContentBlockText(value=prompt)],
                    ),
                ],
                inference_config=InferenceConfiguration(max_tokens=4096),
            )
            if system is not None:
                converse_input.system = [SystemContentBlockText(value=system)]

            return await self._converse_with_retry(client, converse_input)
        except Exception:
            logger.warning("Bedrock generation failed", exc_info=True)
            self._available = None
            return None

    async def generate_chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> str | None:
        """Generate text from a conversation history."""
        try:
            client = self._get_client()
            if client is None:
                return None

            from aws_sdk_bedrock_runtime.models import (
                ContentBlockText,
                ConverseInput,
                InferenceConfiguration,
                SystemContentBlockText,
            )
            from aws_sdk_bedrock_runtime.models import Message as BRMessage

            br_messages = [
                BRMessage(
                    role=m["role"],
                    content=[ContentBlockText(value=m["content"])],
                )
                for m in messages
            ]
            converse_input = ConverseInput(
                model_id=self._config.model,
                messages=br_messages,
                inference_config=InferenceConfiguration(max_tokens=4096),
            )
            if system is not None:
                converse_input.system = [SystemContentBlockText(value=system)]

            return await self._converse_with_retry(client, converse_input)
        except Exception:
            logger.warning("Bedrock chat generation failed", exc_info=True)
            self._available = None
            return None

    async def _converse_with_retry(self, client: Any, converse_input: Any) -> str:
        """Call client.converse with exponential-backoff retries."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await client.converse(converse_input)
                items = response.output.value.content
                result = "".join(
                    v for v in (getattr(i, "value", None) for i in items) if isinstance(v, str)
                )
                if not result:
                    # Reasoning-only (or empty) output: treat as a failed call.
                    raise ValueError("Bedrock response contained no text content")
                self._available = True
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        "Bedrock converse attempt %d/%d failed, retrying in %.1fs: %s",
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    def _get_client(self) -> Any:
        """Lazily create the BedrockRuntimeClient. Returns None if SDK missing."""
        if self._client is None:
            try:
                from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient
                from aws_sdk_bedrock_runtime.config import Config

                config = Config(region=self._config.region)

                # Wire up auth — priority: profile > bearer token > env vars
                profile = _find_aws_profile(self._config.profile)
                if profile:
                    creds = _resolve_profile_credentials(profile)
                    if creds:
                        from smithy_aws_core.identity.static import StaticCredentialsResolver

                        config.aws_access_key_id = creds["access_key"]
                        config.aws_secret_access_key = creds["secret_key"]
                        if "token" in creds:
                            config.aws_session_token = creds["token"]
                        config.aws_credentials_identity_resolver = StaticCredentialsResolver()  # type: ignore[no-untyped-call]
                        self._auth_method = f"profile:{profile}"
                        logger.info("Bedrock: using SigV4 auth (profile '%s')", profile)
                if self._auth_method is None and self._config.bearer_token:
                    _configure_bearer_auth(config, self._config.bearer_token)
                    self._auth_method = "bearer"
                    logger.info("Bedrock: using bearer token auth")
                elif self._auth_method is None and self._config.has_env_credentials:
                    from smithy_aws_core.identity import EnvironmentCredentialsResolver

                    config.aws_credentials_identity_resolver = EnvironmentCredentialsResolver()  # type: ignore[no-untyped-call]
                    self._auth_method = "env"
                    logger.info("Bedrock: using SigV4 auth (env vars)")

                self._client = BedrockRuntimeClient(config)
            except ImportError:
                logger.warning(
                    "aws-sdk-bedrock-runtime package not installed — Bedrock LLM disabled"
                )
                return None
        return self._client

    async def close(self) -> None:
        """No-op — SDK client doesn't need explicit cleanup."""
