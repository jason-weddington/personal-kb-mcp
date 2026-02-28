#!/usr/bin/env bash
set -euo pipefail

# Personal Knowledge MCP Server — setup script
# Checks prerequisites, installs dependencies, and pulls Ollama models.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }

echo -e "\n${BOLD}Personal Knowledge MCP Server — Setup${NC}\n"

errors=0

# ── Python ──────────────────────────────────────────────────────────

echo -e "${BOLD}Checking Python...${NC}"
if command -v python3 &>/dev/null; then
    py_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    py_major=$(echo "$py_version" | cut -d. -f1)
    py_minor=$(echo "$py_version" | cut -d. -f2)
    if [ "$py_major" -ge 3 ] && [ "$py_minor" -ge 13 ]; then
        ok "Python $py_version"
    else
        fail "Python $py_version found, but 3.13+ is required"
        errors=$((errors + 1))
    fi
else
    fail "Python not found — install Python 3.13+ from https://python.org"
    errors=$((errors + 1))
fi

# ── uv ──────────────────────────────────────────────────────────────

echo -e "${BOLD}Checking uv...${NC}"
if command -v uv &>/dev/null; then
    ok "uv $(uv --version 2>/dev/null | head -1)"
else
    fail "uv not found — install from https://docs.astral.sh/uv/"
    errors=$((errors + 1))
fi

# ── Dependencies ────────────────────────────────────────────────────

echo -e "${BOLD}Installing Python dependencies...${NC}"
if command -v uv &>/dev/null; then
    uv sync --quiet 2>/dev/null && ok "Dependencies installed" || { fail "uv sync failed"; errors=$((errors + 1)); }
else
    warn "Skipped (uv not available)"
fi

# ── Ollama ──────────────────────────────────────────────────────────

echo -e "${BOLD}Checking Ollama...${NC}"
if command -v ollama &>/dev/null; then
    ok "Ollama installed"

    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        ok "Ollama is running"

        # Pull embedding model
        echo -e "${BOLD}Pulling embedding model...${NC}"
        embed_model="${KB_EMBEDDING_MODEL:-qwen3-embedding:0.6b}"
        pulled_models=$(ollama list 2>/dev/null || true)
        if echo "$pulled_models" | grep -q "$embed_model"; then
            ok "$embed_model (already pulled)"
        else
            echo "  Pulling $embed_model (this may take a minute)..."
            if ollama pull "$embed_model"; then
                ok "$embed_model pulled"
            else
                fail "Failed to pull $embed_model"
                errors=$((errors + 1))
            fi
        fi

        # Optionally pull LLM model
        echo -e "${BOLD}Checking Ollama LLM model (optional)...${NC}"
        llm_model="${KB_OLLAMA_MODEL:-qwen3:4b}"
        if echo "$pulled_models" | grep -q "$llm_model"; then
            ok "$llm_model (already pulled)"
        else
            warn "$llm_model not pulled — needed only for fully-local operation (no Anthropic API key)"
            echo -e "     Run ${BOLD}ollama pull $llm_model${NC} to enable"
        fi
    else
        warn "Ollama is installed but not running — start it with: ollama serve"
        warn "Vector search will be disabled until Ollama is running"
    fi
else
    warn "Ollama not found — vector search will be disabled (FTS-only mode)"
    echo -e "     Install from ${BOLD}https://ollama.com/download${NC}"
fi

# ── Anthropic ───────────────────────────────────────────────────────

echo -e "${BOLD}Checking Anthropic API key...${NC}"
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    ok "ANTHROPIC_API_KEY is set"
else
    warn "ANTHROPIC_API_KEY not set — graph enrichment, query planning, and synthesis will be disabled"
    echo "     Set it in your MCP client config or shell environment"
    echo "     Alternatively, use Ollama for all LLM features (see README)"
fi

# ── Tests ───────────────────────────────────────────────────────────

echo -e "${BOLD}Running tests...${NC}"
if command -v uv &>/dev/null; then
    test_output=$(uv run pytest --tb=no -q 2>&1)
    if echo "$test_output" | grep -q "passed"; then
        passed=$(echo "$test_output" | grep -oE '[0-9]+ passed')
        ok "Tests passed ($passed)"
    else
        fail "Some tests failed"
        echo "$test_output" | tail -5
        errors=$((errors + 1))
    fi
else
    warn "Skipped (uv not available)"
fi

# ── Summary ─────────────────────────────────────────────────────────

echo ""
if [ "$errors" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}Setup complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Add the server to your MCP client config (see README.md)"
    echo "  2. Set ANTHROPIC_API_KEY in the config for full features"
    echo "  3. Start using kb_store, kb_search, kb_ask, and kb_summarize"
else
    echo -e "${RED}${BOLD}Setup completed with $errors error(s).${NC} See above for details."
fi
echo ""
