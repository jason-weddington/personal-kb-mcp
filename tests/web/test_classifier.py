"""Tests for the query classifier."""

import pytest

from personal_kb.web.classifier import classify_query
from tests.conftest import FakeLLM


@pytest.mark.asyncio
async def test_classify_explore():
    llm = FakeLLM(response="explore")
    result = await classify_query(llm, "what connects to Python?")
    assert result == "explore"


@pytest.mark.asyncio
async def test_classify_summarize():
    llm = FakeLLM(response="summarize")
    result = await classify_query(llm, "why did we choose FastAPI?")
    assert result == "summarize"


@pytest.mark.asyncio
async def test_classify_with_whitespace():
    llm = FakeLLM(response="  summarize  \n")
    result = await classify_query(llm, "explain the pipeline")
    assert result == "summarize"


@pytest.mark.asyncio
async def test_classify_defaults_to_explore_on_garbage():
    llm = FakeLLM(response="I think you should explore the graph")
    result = await classify_query(llm, "test query")
    assert result == "explore"


@pytest.mark.asyncio
async def test_classify_defaults_to_explore_on_none():
    llm = FakeLLM(response=None, available=False)
    result = await classify_query(llm, "test query")
    assert result == "explore"


@pytest.mark.asyncio
async def test_classify_extracts_summarize_from_sentence():
    llm = FakeLLM(response="The classification is: summarize")
    result = await classify_query(llm, "what is our auth pattern?")
    assert result == "summarize"
