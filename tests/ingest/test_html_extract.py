"""Tests for HTML content extraction."""

from personal_kb.ingest.html_extract import extract_content


def test_extracts_article_content():
    """Extracts main content from a typical article page."""
    html = """
    <html>
    <head><title>Test Article</title></head>
    <body>
        <nav><a href="/">Home</a> <a href="/about">About</a></nav>
        <main>
            <article>
                <h1>Important Technical Decision</h1>
                <p>We decided to use PostgreSQL because it supports JSONB
                and has excellent full-text search capabilities.</p>
                <p>The migration from SQLite was straightforward thanks to
                the abstraction layer we built.</p>
            </article>
        </main>
        <footer>Copyright 2024</footer>
    </body>
    </html>
    """
    result = extract_content(html, url="https://example.com/article")
    assert result is not None
    assert "PostgreSQL" in result
    assert "JSONB" in result


def test_returns_none_for_empty_html():
    """Returns None when no content can be extracted."""
    result = extract_content("")
    assert result is None


def test_returns_none_for_script_only():
    """Returns None for pages with only scripts and no text."""
    html = "<html><body><script>var x = 1;</script></body></html>"
    result = extract_content(html)
    assert result is None
