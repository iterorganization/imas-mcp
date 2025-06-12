# Start with a Python 3.12 slim image
FROM python:3.12-slim

# Install system dependencies including git for git dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy git directory and essential files for versioning with hatch-vcs
COPY .git/ ./.git/
COPY pyproject.toml uv.lock* ./
COPY README.md ./

# Install dependencies using uv (git context is now available for hatch-vcs)
RUN uv sync

# Copy source code and scripts after dependency installation
COPY imas_mcp_server/ ./imas_mcp_server/
COPY scripts/ ./scripts/

# Add build arg for IDS filter - supports space-separated list or empty for all IDS
ARG IDS_FILTER=""

# Set environment variables before running scripts
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV IDS_FILTER=${IDS_FILTER}

# Copy index files if present in build context, using a pattern that won't fail
COPY index/ ./index/

# Build IDS arguments, filter existing index files, and build/verify index in single layer
RUN set -e && \
    echo "Processing IDS filter: '${IDS_FILTER}'" && \
    if [ -n "${IDS_FILTER}" ]; then \
    IDS_ARGS=""; \
    for ids in ${IDS_FILTER}; do \
    IDS_ARGS="$IDS_ARGS --ids-filter $ids"; \
    done; \
    else \
    IDS_ARGS=""; \
    echo "No IDS filter - using all IDS"; \
    fi && \
    echo "IDS arguments: $IDS_ARGS" && \
    \
    echo "Getting expected index name..." && \
    INDEX_NAME=$(uv run index-name $IDS_ARGS) && \
    echo "Pre-filtering index files to keep: $INDEX_NAME" && \
    find ./index/ -not -name "${INDEX_NAME}*" -not -name "." -not -name ".." -type f -delete && \
    echo "Pre-filtered index files:" && \
    ls -la ./index/ && \
    \
    echo "Building/verifying index..." && \
    FINAL_INDEX_NAME=$(uv run build-index $IDS_ARGS | tail -n 1) && \
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
CMD ["uv", "run", "run-server", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000"]