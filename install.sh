#!/usr/bin/env bash
set -euo pipefail

# Personal Knowledge MCP Server — one-liner installer
# Usage: curl -fsSL https://raw.githubusercontent.com/jason-weddington/personal-kb-mcp/main/install.sh | bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m'

REPO="https://github.com/jason-weddington/personal-kb-mcp.git"
INSTALL_DIR="${PERSONAL_KB_DIR:-$HOME/personal-kb-mcp}"

echo -e "\n${BOLD}Personal Knowledge MCP Server — Installer${NC}\n"

# Check prerequisites
for cmd in git uv; do
    if ! command -v "$cmd" &>/dev/null; then
        echo -e "${RED}Error:${NC} $cmd is required but not found."
        case "$cmd" in
            git) echo "  Install git from https://git-scm.com/" ;;
            uv)  echo "  Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh" ;;
        esac
        exit 1
    fi
done

# Clone or update
if [ -d "$INSTALL_DIR" ]; then
    echo "Updating existing installation at $INSTALL_DIR..."
    git -C "$INSTALL_DIR" pull --quiet
else
    echo "Cloning to $INSTALL_DIR..."
    git clone --quiet "$REPO" "$INSTALL_DIR"
fi

# Run setup
cd "$INSTALL_DIR"
./setup.sh

echo -e "${BOLD}Installed to:${NC} $INSTALL_DIR"
echo ""
echo "Add to your MCP client config:"
echo ""
echo "  {\"mcpServers\": {\"personal-kb\": {"
echo "    \"type\": \"stdio\","
echo "    \"command\": \"uv\","
echo "    \"args\": [\"run\", \"--directory\", \"$INSTALL_DIR\", \"personal-kb\"],"
echo "    \"env\": {\"ANTHROPIC_API_KEY\": \"sk-ant-...\"}"
echo "  }}}"
echo ""
