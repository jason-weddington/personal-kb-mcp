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

PKG="git+https://github.com/jason-weddington/personal-kb-mcp.git[safety]"

echo -e "\n${BOLD}Personal Knowledge MCP Server — Setup${NC}"

# ── Xcode Command Line Tools (macOS only) ─────────────────────────

if [ "$(uname)" = "Darwin" ]; then
    step "Checking Xcode Command Line Tools..."
    if xcode-select -p &>/dev/null; then
        ok "Xcode CLT installed"
    else
        echo "  Installing Xcode Command Line Tools (required for git)..."
        echo "  A system dialog may appear — click Install and wait for it to finish."
        xcode-select --install 2>/dev/null || true
        # Wait for the install to complete
        until xcode-select -p &>/dev/null; do
            sleep 5
        done
        ok "Xcode CLT installed"
    fi
fi

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
    read -rp "  Enter your Anthropic API key (or press Enter to skip): " api_key < /dev/tty
    if [ -n "$api_key" ]; then
        export ANTHROPIC_API_KEY="$api_key"
        ok "ANTHROPIC_API_KEY set for this session"
    else
        warn "Skipped — running in Ollama-only mode"
    fi
fi

# ── MCP config ─────────────────────────────────────────────────────

step "Setup complete!"

# Build the MCP config JSON
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    MCP_JSON=$(cat <<ENDJSON
{
  "mcpServers": {
    "personal-kb": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "$PKG", "personal-kb"],
      "env": {
        "ANTHROPIC_API_KEY": "$ANTHROPIC_API_KEY"
      }
    }
  }
}
ENDJSON
)
else
    MCP_JSON=$(cat <<ENDJSON
{
  "mcpServers": {
    "personal-kb": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "$PKG", "personal-kb"],
      "env": {
        "KB_EXTRACTION_PROVIDER": "ollama",
        "KB_QUERY_PROVIDER": "ollama"
      }
    }
  }
}
ENDJSON
)
fi

echo ""
echo "  Add this to your MCP client config:"
echo "    Claude Code  → ~/.claude.json"
echo "    Claude Desktop → claude_desktop_config.json"
echo ""
echo -e "${BOLD}$MCP_JSON${NC}"
echo ""

# Try to copy to clipboard
if command -v pbcopy &>/dev/null; then
    echo "$MCP_JSON" | pbcopy
    ok "Copied to clipboard"
elif command -v xclip &>/dev/null; then
    echo "$MCP_JSON" | xclip -selection clipboard
    ok "Copied to clipboard"
elif command -v xsel &>/dev/null; then
    echo "$MCP_JSON" | xsel --clipboard
    ok "Copied to clipboard"
else
    echo "  (Copy the JSON above manually)"
fi

# ── Optional: launch explorer ─────────────────────────────────────

echo ""
read -rp "  Launch the knowledge explorer in your browser? [Y/n] " launch < /dev/tty
if [ "${launch,,}" != "n" ]; then
    echo ""
    echo "  Opening http://127.0.0.1:8767 ..."
    echo "  Press Ctrl+C to stop."
    echo ""
    if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
        ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
            uvx --from "$PKG" personal-kb-web
    else
        KB_EXTRACTION_PROVIDER=ollama KB_QUERY_PROVIDER=ollama \
            uvx --from "$PKG" personal-kb-web
    fi
fi
