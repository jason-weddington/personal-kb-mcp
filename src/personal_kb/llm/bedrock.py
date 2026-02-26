"""AWS Bedrock LLM client with graceful degradation."""

from __future__ import annotations

import logging
import os
from typing import Any

from personal_kb.config import get_bedrock_model, get_bedrock_region

logger = logging.getLogger(__name__)


class BedrockLLMClient:
    """Generates text via the AWS Bedrock Converse API."""

    def __init__(self) -> None:
        """Initialize with lazy client creation."""
        self._client: Any = None
        self._available: bool | None = None

    async def is_available(self) -> bool:
        """Check availability. Only caches success — retries on failure."""
        if self._available is True:
            return True
        try:
            client = self._get_client()
            if client is None:
                return False
            if not os.environ.get("AWS_ACCESS_KEY_ID"):
                logger.warning("AWS_ACCESS_KEY_ID not set — Bedrock LLM disabled")
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

            # Workaround: smithy-json doesn't escape newlines in string
            # values, causing SerializationException on the Bedrock API.
            # Replace \n with \\n so the JSON payload is valid.
            safe_prompt = prompt.replace("\n", "\\n")

            converse_input = ConverseInput(
                model_id=get_bedrock_model(),
                messages=[
                    Message(
                        role="user",
                        content=[ContentBlockText(value=safe_prompt)],
                    ),
                ],
                inference_config=InferenceConfiguration(max_tokens=4096),
            )
            if system is not None:
                safe_system = system.replace("\n", "\\n")
                converse_input.system = [SystemContentBlockText(value=safe_system)]

            response = await client.converse(converse_input)
            result: str = response.output.value.content[0].value
            self._available = True
            return result
        except Exception:
            logger.warning("Bedrock generation failed", exc_info=True)
            self._available = None
            return None

    def _get_client(self) -> Any:
        """Lazily create the BedrockRuntimeClient. Returns None if SDK missing."""
        if self._client is None:
            try:
                from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient
                from aws_sdk_bedrock_runtime.config import Config
                from smithy_aws_core.identity import EnvironmentCredentialsResolver

                self._client = BedrockRuntimeClient(
                    Config(
                        region=get_bedrock_region(),
                        aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),  # type: ignore[no-untyped-call]
                    )
                )
            except ImportError:
                logger.warning(
                    "aws-sdk-bedrock-runtime package not installed — Bedrock LLM disabled"
                )
                return None
        return self._client

    async def close(self) -> None:
        """No-op — SDK client doesn't need explicit cleanup."""
