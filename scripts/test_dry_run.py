#!/usr/bin/env python3
"""Test dry-run ingestion of a real file to evaluate extraction quality.

Usage:
    uv run python scripts/test_dry_run.py [FILE_PATH]

Respects KB_EXTRACTION_PROVIDER (default: anthropic) to select the LLM.
Set KB_EXTRACTION_PROVIDER=ollama and KB_OLLAMA_MODEL=qwen3.5:35b-a3b
to test local models.
"""

import asyncio
import sys
import time
from pathlib import Path

from personal_kb.config import get_extraction_provider
from personal_kb.db.connection import create_connection
from personal_kb.graph.builder import GraphBuilder
from personal_kb.graph.enricher import GraphEnricher
from personal_kb.ingest.ingester import FileIngester
from personal_kb.llm.provider import LLMProvider
from personal_kb.search.embeddings import EmbeddingClient
from personal_kb.store.knowledge_store import KnowledgeStore


def _create_llm(provider: str) -> LLMProvider | None:
    """Create an LLM client for the given provider name."""
    if provider == "anthropic":
        try:
            from personal_kb.llm.anthropic import AnthropicLLMClient
        except Exception:
            return None
        return AnthropicLLMClient()
    if provider == "bedrock":
        try:
            from personal_kb.llm.bedrock import BedrockLLMClient
        except Exception:
            return None
        return BedrockLLMClient()
    if provider == "ollama":
        from personal_kb.llm.ollama import OllamaLLMClient

        return OllamaLLMClient()
    return None


async def main() -> None:
    """Run dry-run ingestion on a file and print extraction results."""
    default_path = "/Users/jason/Documents/my_notes/Cleanr Project Notes.md"
    file_path = Path(sys.argv[1] if len(sys.argv) > 1 else default_path)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    provider = get_extraction_provider()

    print(f"File: {file_path}")
    print(f"Size: {file_path.stat().st_size:,} bytes")
    print(f"Provider: {provider}")
    print()

    # Set up in-memory DB + LLM
    db = await create_connection(":memory:")
    store = KnowledgeStore(db)
    embedder = EmbeddingClient(db)
    graph_builder = GraphBuilder(db)

    llm = _create_llm(provider)
    if llm is None or not await llm.is_available():
        print(f"ERROR: {provider} LLM not available. Check credentials/config.")
        await db.close()
        sys.exit(1)

    enricher = GraphEnricher(db, llm)
    ingester = FileIngester(
        db=db,
        store=store,
        embedder=embedder,
        graph_builder=graph_builder,
        graph_enricher=enricher,
        llm=llm,
    )

    print(f"Running dry-run ingestion via {provider}...")
    print("=" * 70)

    t0 = time.monotonic()
    result = await ingester.ingest_file(
        file_path,
        base_dir=file_path.parent,
        dry_run=True,
        project_ref=None,
    )
    t_ingest = time.monotonic() - t0

    print(f"\nAction: {result.action}")
    if result.reason:
        print(f"Reason: {result.reason}")

    print(f"\n{'─' * 70}")
    print("SUMMARY")
    print(f"{'─' * 70}")
    print(result.summary or "(no summary)")

    print(f"\n{'─' * 70}")
    print(f"EXTRACTED ENTRIES ({result.entry_count})")
    print(f"{'─' * 70}")

    from personal_kb.ingest.extractor import extract_entries

    content = file_path.read_text(encoding="utf-8", errors="replace")
    rel_path = file_path.name
    t1 = time.monotonic()
    entries = await extract_entries(llm, rel_path, content)
    t_extract = time.monotonic() - t1

    for i, entry in enumerate(entries, 1):
        print(f"\n  [{i}] {entry.short_title}")
        print(f"      Title: {entry.long_title}")
        print(f"      Type:  {entry.entry_type}")
        print(f"      Tags:  {', '.join(entry.tags)}")
        print("      Content:")
        for line in entry.knowledge_details.splitlines():
            print(f"        {line}")

    t_total = t_ingest + t_extract

    print(f"\n{'=' * 70}")
    print(f"Total: {len(entries)} entries extracted")
    print(f"Time:  {t_ingest:.1f}s summarize + {t_extract:.1f}s extract = {t_total:.1f}s total")

    await llm.close()
    await embedder.close()
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
