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
    PYTHONDONTWRITEBYTECODE=1 \
    HATCH_BUILD_NO_HOOKS=true

# Copy dependency files and git metadata 
COPY .git/ ./.git/
COPY pyproject.toml ./
COPY README.md ./
COPY hatch_build_hooks.py ./

# Ensure git repository is properly initialized for version detection
RUN git config --global --add safe.directory /app

# Install only dependencies without the local project to avoid build hooks
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --no-dev --no-install-project --extra http --extra build

# Copy source code (separate layer for better caching)
COPY imas_mcp/ ./imas_mcp/
COPY scripts/ ./scripts/

# Install project with HTTP and build support for container deployment
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --no-dev --extra http --extra build

# Install imas-data-dictionary manually from git (needed for index building)
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install "imas-data-dictionary @ git+https://github.com/iterorganization/imas-data-dictionary.git@develop"

# Build JSON data
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "Building JSON data..." && \
    uv run --no-dev build-json-data && \
    echo "✓ JSON data ready"

# Expose port (only needed for streamable-http transport)
EXPOSE 8000

# Run the application (host and port only needed for streamable-http transport)
CMD ["sh", "-c", "\
    uv run --no-dev run-server \
    --transport ${TRANSPORT} \
    --host 0.0.0.0 \
    --port 8000 \
    "] 