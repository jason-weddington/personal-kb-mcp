"""Tests for markdown-aware content chunking."""

from personal_kb.ingest.chunker import chunk_content


class TestSmallContent:
    def test_small_file_returns_single_chunk(self):
        content = "# Hello\n\nSmall file content."
        chunks = chunk_content(content, chunk_size=1000, overlap=100)
        assert len(chunks) == 1
        assert chunks[0].text == content
        assert chunks[0].index == 0
        assert chunks[0].start_char == 0
        assert chunks[0].heading == "Hello"

    def test_no_heading_returns_none_heading(self):
        content = "Just plain text, no headings."
        chunks = chunk_content(content, chunk_size=1000, overlap=100)
        assert len(chunks) == 1
        assert chunks[0].heading is None

    def test_exact_chunk_size_no_split(self):
        content = "x" * 100
        chunks = chunk_content(content, chunk_size=100, overlap=10)
        assert len(chunks) == 1


class TestHeadingSplits:
    def test_splits_at_h1_headings(self):
        content = "# Section 1\n\nContent 1.\n\n# Section 2\n\nContent 2."
        chunks = chunk_content(content, chunk_size=30, overlap=0)
        assert len(chunks) >= 2
        assert any("Section 1" in c.text for c in chunks)
        assert any("Section 2" in c.text for c in chunks)

    def test_splits_at_h2_headings(self):
        content = "## Part A\n\nText A.\n\n## Part B\n\nText B."
        chunks = chunk_content(content, chunk_size=25, overlap=0)
        assert len(chunks) >= 2

    def test_does_not_split_at_h3(self):
        # H3 headings should NOT trigger splits — only H1/H2
        content = "### Sub A\n\nText A.\n\n### Sub B\n\nText B."
        chunks = chunk_content(content, chunk_size=1000, overlap=0)
        assert len(chunks) == 1

    def test_heading_extracted_per_chunk(self):
        content = "# Intro\n\nIntro text.\n\n# Methods\n\nMethod details."
        chunks = chunk_content(content, chunk_size=30, overlap=0)
        headings = [c.heading for c in chunks]
        assert "Intro" in headings
        assert "Methods" in headings


class TestMergeSmallSections:
    def test_merges_small_adjacent_sections(self):
        # Three tiny sections that together fit in one chunk
        content = "# A\n\na\n\n# B\n\nb\n\n# C\n\nc"
        chunks = chunk_content(content, chunk_size=1000, overlap=0)
        assert len(chunks) == 1
        assert "# A" in chunks[0].text
        assert "# C" in chunks[0].text

    def test_does_not_merge_beyond_limit(self):
        # Each section is ~20 chars, chunk_size=35 should prevent merging all three
        content = ""
        for i in range(3):
            content += f"# S{i}\n\n" + "x" * 15 + "\n\n"
        chunks = chunk_content(content, chunk_size=35, overlap=0)
        assert len(chunks) >= 2


class TestParagraphFallback:
    def test_splits_long_section_at_paragraphs(self):
        # One section that's too long for chunk_size
        paragraphs = ["Paragraph " + str(i) + ". " + "x" * 40 for i in range(5)]
        content = "\n\n".join(paragraphs)
        chunks = chunk_content(content, chunk_size=120, overlap=0)
        assert len(chunks) >= 2


class TestNewlineFallback:
    def test_splits_at_newlines_when_no_paragraphs(self):
        # Single giant paragraph with only single newlines
        lines = ["Line " + str(i) + " " + "x" * 30 for i in range(10)]
        content = "\n".join(lines)
        chunks = chunk_content(content, chunk_size=100, overlap=0)
        assert len(chunks) >= 2


class TestOverlap:
    def test_overlap_prepends_previous_text(self):
        # Two sections, each fits in one chunk, but too big together
        sec1 = "# Section 1\n\n" + "a" * 50
        sec2 = "\n\n# Section 2\n\n" + "b" * 50
        content = sec1 + sec2
        chunks_with = chunk_content(content, chunk_size=80, overlap=20)
        chunks_without = chunk_content(content, chunk_size=80, overlap=0)
        assert len(chunks_with) >= 2
        assert len(chunks_without) >= 2
        # With overlap, the second chunk is longer than without
        assert len(chunks_with[1].text) > len(chunks_without[1].text)

    def test_no_overlap_when_zero(self):
        sec1 = "# A\n\n" + "a" * 30
        sec2 = "\n\n# B\n\n" + "b" * 30
        content = sec1 + sec2
        chunks = chunk_content(content, chunk_size=40, overlap=0)
        if len(chunks) >= 2:
            # Second chunk should NOT contain 'a' chars from first section
            assert "aaa" not in chunks[1].text


class TestIndexSequencing:
    def test_indices_are_sequential(self):
        content = "# A\n\nText.\n\n# B\n\nText.\n\n# C\n\nText."
        chunks = chunk_content(content, chunk_size=20, overlap=0)
        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_start_chars_are_non_decreasing(self):
        content = "# A\n\nSome text.\n\n# B\n\nMore text.\n\n# C\n\nFinal text."
        chunks = chunk_content(content, chunk_size=25, overlap=0)
        starts = [c.start_char for c in chunks]
        for i in range(1, len(starts)):
            assert starts[i] >= starts[i - 1]


class TestEdgeCases:
    def test_empty_content(self):
        chunks = chunk_content("", chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0].text == ""

    def test_content_before_first_heading(self):
        content = "Preamble text.\n\n# First Heading\n\nContent."
        chunks = chunk_content(content, chunk_size=25, overlap=0)
        assert any("Preamble" in c.text for c in chunks)

    def test_uses_config_defaults(self):
        # Just verify it doesn't crash when using defaults
        chunks = chunk_content("Short content.")
        assert len(chunks) == 1
