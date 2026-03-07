"""HTML-to-markdown extraction for URL ingestion."""

import trafilatura


def extract_content(html: str, url: str | None = None) -> str | None:
    """Extract main content from HTML, returning clean text.

    Uses trafilatura for article extraction. Returns None if
    extraction fails.
    """
    result = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        output_format="txt",
        favor_precision=True,
    )
    return result
