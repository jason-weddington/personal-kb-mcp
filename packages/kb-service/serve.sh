#!/usr/bin/env bash
# Production entry point: uvicorn only.
set -e
exec uv run uvicorn kb_service.main:app --host 127.0.0.1 --port 8000
