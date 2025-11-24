#!/bin/bash
set -euo pipefail

# Scrape a single documentation source into the docs-mcp-server database
# Usage: scrape-doc.sh <name> <url> <version> <max_pages> <max_depth>

SOURCE_NAME="$1"
SOURCE_URL="$2"
SOURCE_VERSION="$3"
MAX_PAGES="$4"
MAX_DEPTH="$5"

# Configuration (provided by environment or defaults for local use)
export DOCS_MCP_STORE_PATH="${DOCS_MCP_STORE_PATH:-./docs-data}"
export DOCS_MCP_EMBEDDING_MODEL="${DOCS_MCP_EMBEDDING_MODEL:-openai/text-embedding-3-small}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export HF_HUB_DISABLE_SYMLINKS_WARNING="${HF_HUB_DISABLE_SYMLINKS_WARNING:-1}"
export DOCS_MCP_SERVER_VERSION="${DOCS_MCP_SERVER_VERSION:-1.29.0}"

# Create docs-data directory if it doesn't exist
mkdir -p "$DOCS_MCP_STORE_PATH"

echo "Scraping $SOURCE_NAME..."
echo "  URL: $SOURCE_URL"
echo "  Version: $SOURCE_VERSION"
echo "  Store: $DOCS_MCP_STORE_PATH"

# Scrape using docs-mcp-server
EXCLUDE_PATTERN="${6:-}"
ARGS=(scrape "$SOURCE_NAME" "$SOURCE_URL" \
    --version "$SOURCE_VERSION" \
    --max-pages "$MAX_PAGES" \
    --max-depth "$MAX_DEPTH" \
    --ignore-errors)

if [ -n "$EXCLUDE_PATTERN" ]; then
    ARGS+=(--exclude-pattern "$EXCLUDE_PATTERN")
fi

npx -y @arabold/docs-mcp-server@${DOCS_MCP_SERVER_VERSION} "${ARGS[@]}" || true

echo "✓ Completed scraping $SOURCE_NAME"

