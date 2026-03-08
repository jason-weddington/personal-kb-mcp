#!/usr/bin/env bash
set -euo pipefail

# Personal Knowledge MCP Server — Setup
# Works on macOS and Linux.
# Usage: curl -fsSL https://raw.githubusercontent.com/jason-weddington/personal-kb-mcp/main/setup.sh | bash

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; exit 1; }
step() { echo -e "\n${BOLD}$1${NC}"; }

echo -e "\n${BOLD}Personal Knowledge MCP Server — Setup${NC}"

# ── uv ─────────────────────────────────────────────────────────────

step "Checking uv..."
if command -v uv &>/dev/null; then
    ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"
else
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the env so uv is available in this session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if command -v uv &>/dev/null; then
        ok "uv installed"
    else
        fail "uv installation failed. Install manually: https://docs.astral.sh/uv/"
    fi
fi

# ── Python 3.13 ────────────────────────────────────────────────────

step "Checking Python 3.13..."
if uv python find 3.13 &>/dev/null; then
    ok "Python 3.13 ($(uv python find 3.13))"
else
    echo "  Installing Python 3.13 via uv..."
    uv python install 3.13
    if uv python find 3.13 &>/dev/null; then
        ok "Python 3.13 installed"
    else
        fail "Python 3.13 installation failed."
    fi
fi

# ── Ollama ─────────────────────────────────────────────────────────

step "Checking Ollama..."
if command -v ollama &>/dev/null; then
    ok "Ollama $(ollama --version 2>/dev/null | awk '{print $NF}')"
else
    echo "  Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    if command -v ollama &>/dev/null; then
        ok "Ollama installed"
    else
        fail "Ollama installation failed. Install manually: https://ollama.com/download"
    fi
fi

# Start Ollama if not running
if ! curl -s --max-time 2 http://localhost:11434/api/tags &>/dev/null; then
    echo "  Starting Ollama..."
    ollama serve &>/dev/null &
    sleep 2
    if curl -s --max-time 2 http://localhost:11434/api/tags &>/dev/null; then
        ok "Ollama started"
    else
        warn "Could not start Ollama. Start it manually: ollama serve"
    fi
else
    ok "Ollama is running"
fi

# ── Embedding model ────────────────────────────────────────────────

EMBED_MODEL="qwen3-embedding:0.6b"
step "Checking embedding model ($EMBED_MODEL)..."
if ollama list 2>/dev/null | grep -q "$EMBED_MODEL"; then
    ok "$EMBED_MODEL (already pulled)"
else
    echo "  Pulling $EMBED_MODEL (this may take a minute)..."
    if ollama pull "$EMBED_MODEL"; then
        ok "$EMBED_MODEL pulled"
    else
        warn "Failed to pull $EMBED_MODEL — vector search will be disabled (FTS-only mode)"
    fi
fi

# ── Anthropic API key ──────────────────────────────────────────────

step "Anthropic API key..."
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    ok "ANTHROPIC_API_KEY is set"
else
    echo "  An Anthropic API key enables graph enrichment, query planning,"
    echo "  and AI-powered synthesis. The KB works without it (Ollama-only mode)"
    echo "  but these features significantly improve quality."
    echo ""
    read -rp "  Enter your Anthropic API key (or press Enter to skip): " api_key
    if [ -n "$api_key" ]; then
        export ANTHROPIC_API_KEY="$api_key"
        ok "ANTHROPIC_API_KEY set for this session"
        echo ""
        echo "  To make it permanent, add to your shell profile:"
        echo "    export ANTHROPIC_API_KEY='$api_key'"
    else
        warn "Skipped — running in Ollama-only mode"
    fi
fi

# ── Launch explorer ────────────────────────────────────────────────

step "Launching the Knowledge Explorer..."
echo ""
echo "  The explorer will open at http://127.0.0.1:8765"
echo "  Press Ctrl+C to stop."
echo ""

EXPLORER_CMD="uvx --from 'git+https://github.com/jason-weddington/personal-kb-mcp.git[safety]' personal-kb-web"

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" uvx --from "git+https://github.com/jason-weddington/personal-kb-mcp.git[safety]" personal-kb-web
else
    uvx --from "git+https://github.com/jason-weddington/personal-kb-mcp.git[safety]" personal-kb-web
fi
