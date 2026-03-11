"""Vulture allowlist — false positives that should not be flagged as dead code.

Vulture can't see dynamic usage (MCP tool registration via decorators, FastAPI
route handlers, Protocol method implementations, SDK attribute assignments).
Add entries here when vulture flags something that IS used at runtime.

Format: just reference the name so vulture knows it's intentional.
"""
