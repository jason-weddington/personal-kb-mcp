"""HTML-to-markdown extraction for URL ingestion."""

import logging

logger = logging.getLogger(__name__)


def extract_content(html: str, url: str | None = None) -> str | None:
    """Extract main content from HTML, returning clean text.

    Uses trafilatura for article extraction. Returns None if
    trafilatura is not installed or extraction fails.
    """
    try:
        import trafilatura
    except ImportError:
        logger.warning("trafilatura not installed — cannot extract HTML content")
        return None

    result = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        output_format="txt",
        favor_precision=True,
    )
    return result
