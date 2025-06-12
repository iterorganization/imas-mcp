# Start with a Python 3.12 slim image
FROM python:3.12-slim

# Install system dependencies including git for git dependencies
# Use --mount=cache for apt cache to speed up builds
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager - use specific version for reproducibility
COPY --from=ghcr.io/astral-sh/uv:0.4.30 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files and git metadata 
COPY .git/ ./.git/
COPY pyproject.toml ./
COPY README.md ./

# Install dependencies using uv with cache mount for faster builds
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --no-install-project --link-mode=copy

# Copy source code and scripts after dependency installation
COPY imas_mcp_server/ ./imas_mcp_server/
COPY scripts/ ./scripts/

# Install the project itself (non-editable) after copying source code
RUN uv pip install --no-deps --link-mode=copy .

# Copy index files if present in build context
COPY index/ ./index/

# Add build arg for IDS filter - supports space-separated string (e.g., "core_profiles equilibrium") or empty for all IDS
ARG IDS_FILTER=""

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    IDS_FILTER=${IDS_FILTER} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build/verify index in single layer
RUN set -e && \
    echo "Get index name..." && \
    INDEX_NAME=$(uv run index-name ${IDS_FILTER:+--ids-filter "${IDS_FILTER}"}) && \
    echo "Pre-filtering index files to keep: $INDEX_NAME" && \
    find ./index/ -not -name "${INDEX_NAME}*" -not -name "." -not -name ".." -type f -delete && \
    echo "Pre-filtered index files:" && \
    ls -la ./index/ && \
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
    ls -la ./index/

# Expose port
EXPOSE 8000

# Run the application using the script entrypoint with streamable-http transport
CMD ["sh", "-c", "\
    exec uv run run-server \
    --transport streamable-http \
    --host 0.0.0.0 \
    --port 8000 \
    ${IDS_FILTER:+--ids-filter \"${IDS_FILTER}\"} \
    --no-auto-build\
    "]