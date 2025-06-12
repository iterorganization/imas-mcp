# Build stage - handles git dependencies and creates virtual environment
FROM python:3.12-slim AS builder

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

# Copy dependency and git metadata files first (for better caching)
COPY pyproject.toml ./
COPY README.md ./
COPY .git/ ./.git/

# Install dependencies only (without the project itself)
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --no-install-project --link-mode=copy

# Copy source code and git metadata for dynamic versioning
COPY imas_mcp_server/ ./imas_mcp_server/
COPY scripts/ ./scripts/

# Install the project itself with no-deps (dependencies already installed)
RUN uv pip install --no-deps --link-mode=copy .

# Runtime stage - minimal image with only installed packages
FROM python:3.12-slim AS runtime

# Install minimal runtime dependencies (git might be needed for some packages at runtime)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager for runtime
COPY --from=ghcr.io/astral-sh/uv:0.4.30 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy the virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy source code (needed for runtime)
COPY --from=builder /app/imas_mcp_server ./imas_mcp_server/
COPY --from=builder /app/scripts ./scripts/

# Copy git metadata for runtime version detection (smaller impact on cache)
COPY .git/ ./.git/

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