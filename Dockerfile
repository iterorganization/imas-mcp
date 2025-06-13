# Single stage build - simplified Dockerfile
FROM python:3.12-slim

# Install system dependencies including git for git dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.4.30 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Add build args for IDS filter and transport
ARG IDS_FILTER=""
ARG TRANSPORT="streamable-http"

# Set environment variables
ENV PYTHONPATH="/app" \
    IDS_FILTER=${IDS_FILTER} \
    TRANSPORT=${TRANSPORT} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy dependency files and git metadata 
COPY .git/ ./.git/
COPY pyproject.toml ./
COPY README.md ./

# Ensure git repository is properly initialized for version detection
RUN git config --global --add safe.directory /app

# Install dependencies in a virtual environment
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --no-dev

# Copy source code (separate layer for better caching)
COPY imas_mcp_server/ ./imas_mcp_server/
COPY scripts/ ./scripts/

# Copy index files if present in build context (optional)
COPY index/ ./index/

# Build/verify index in single layer
RUN set -e && \
    echo "Get index name..." && \
    INDEX_NAME=$(uv run index-name ${IDS_FILTER:+--ids-filter "${IDS_FILTER}"}) && \
    echo "Pre-filtering index files to keep: $INDEX_NAME" && \
    if [ -d "./index/" ]; then \
    find ./index/ -not -name "${INDEX_NAME}*" -not -name "." -not -name ".." -type f -delete 2>/dev/null || true; \
    echo "Pre-filtered index files:" && \
    ls -la ./index/ 2>/dev/null || echo "No index directory"; \
    fi && \
    \
    echo "Building/verifying index..." && \
    FINAL_INDEX_NAME=$(uv run build-index ${IDS_FILTER:+--ids-filter "${IDS_FILTER}"} | tail -n 1) && \
    echo "Index built/verified: $FINAL_INDEX_NAME" && \
    \
    echo "Asserting index names match..." && \
    if [ "$INDEX_NAME" != "$FINAL_INDEX_NAME" ]; then \
    echo "ERROR: Index name mismatch!" && \
    echo "  Expected: $INDEX_NAME" && \
    echo "  Actual:   $FINAL_INDEX_NAME" && \
    exit 1; \
    fi && \
    echo "✓ Index names match: $INDEX_NAME" && \
    echo "Final index files:" && \
    ls -la ./index/ 2>/dev/null || echo "No index files"

# Expose port (only needed for streamable-http transport)
EXPOSE 8000

# Run the application (host and port only needed for streamable-http transport)
CMD ["sh", "-c", "\
    exec uv run run-server \
    --transport ${TRANSPORT} \
    --host 0.0.0.0 \
    --port 8000 \
    ${IDS_FILTER:+--ids-filter \"${IDS_FILTER}\"} \
    --no-auto-build\
    "]