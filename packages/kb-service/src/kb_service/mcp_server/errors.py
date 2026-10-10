"""Tool-layer error type and rendering for the /mcp tools.

Byte-identical to ``personal_kb.backend.http.BackendHttpError`` /
``_map_error`` so the HTTP-served tools render exactly what the stdio tools
render during the overlap.
"""


class BackendHttpError(Exception):
    """Non-2xx outcome of a route handler call."""

    def __init__(self, status: int, detail: str) -> None:
        """Initialise with HTTP *status* code and error *detail* string."""
        self.status = status
        self.detail = detail
        super().__init__(f"KB service returned {status}: {detail}")


def map_error(exc: BackendHttpError) -> str:
    """Convert a BackendHttpError to a human-readable tool-layer error string."""
    if exc.status == 401:
        return (
            "Error: KB service authentication failed (401). Check PERSONAL_KB_API_KEY."
        )
    if exc.status == 403:
        if exc.detail.startswith("write policy: "):
            return f"Error: {exc.detail}"
        return f"Error: admin privileges required (403): {exc.detail}"
    if exc.status in (404, 409):
        return f"Error: {exc.detail}"
    return f"Error: KB service returned {exc.status}: {exc.detail}"
