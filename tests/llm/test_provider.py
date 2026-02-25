"""Tests for LLM provider protocol conformance."""

from personal_kb.llm.ollama import OllamaLLMClient
from personal_kb.llm.provider import LLMProvider
from tests.conftest import FakeLLM


def test_fake_llm_conforms_to_protocol():
    assert isinstance(FakeLLM(), LLMProvider)


def test_ollama_client_conforms_to_protocol():
    assert isinstance(OllamaLLMClient(), LLMProvider)
