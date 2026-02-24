"""Entry point for the personal-kb MCP server."""

from personal_kb.server import create_server


def main() -> None:
    """Run the personal-kb MCP server."""
    server = create_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
