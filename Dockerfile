## Stage 1: acquire uv binary (kept minimal)
ARG UV_VERSION=0.7.13
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

## Stage 2: Build complete project
FROM python:3.12-slim AS builder

# Install system dependencies including git for git dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary
COPY --from=uv /uv /bin/

# Set working directory
WORKDIR /app

# Add build args for IDS filter and transport
ARG IDS_FILTER=""
ARG TRANSPORT="streamable-http"
ARG IMAS_DD_VERSION="4.0.0"

# Additional build-time metadata for cache busting & traceability
ARG GIT_SHA=""
ARG GIT_TAG=""
ARG GIT_REF=""

# Set environment variables
ENV PYTHONPATH="/app" \
    IDS_FILTER=${IDS_FILTER} \
    TRANSPORT=${TRANSPORT} \
    IMAS_DD_VERSION=${IMAS_DD_VERSION} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HATCH_BUILD_NO_HOOKS=true \
    # OpenRouter configuration for embeddings
    OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
    IMAS_MCP_EMBEDDING_MODEL=qwen/qwen3-embedding-4b

# Labels for image provenance
LABEL imas_mcp.git_sha=${GIT_SHA} \
      imas_mcp.git_tag=${GIT_TAG} \
      imas_mcp.git_ref=${GIT_REF}

## Copy git metadata first so hatch-vcs sees repository state exactly as on tag
COPY .git/ ./.git/
RUN git config --global --add safe.directory /app

# Sparse checkout phase 1: only dependency definition artifacts (non-cone to allow root files)
# We intentionally exclude source so code changes do not invalidate dependency layer.
RUN git config core.sparseCheckout true \
    && git sparse-checkout init --no-cone \
    && git sparse-checkout set pyproject.toml uv.lock \
    && git reset --hard HEAD \
    && echo "Sparse checkout (phase 1) paths:" \
    && git sparse-checkout list

## Install only dependencies without installing the local project (frozen = must match committed lock)
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --no-dev --no-install-project --frozen || \
    (echo "Dependency sync failed (lock mismatch). Run 'uv lock' locally and commit changes." >&2; exit 1) && \
    if [ -n "$(git status --porcelain uv.lock)" ]; then echo "uv.lock changed during dep sync (unexpected)." >&2; exit 1; fi

# Expand sparse checkout to include project sources and scripts (phase 2)
RUN git sparse-checkout set pyproject.toml uv.lock README.md imas_mcp scripts \
    && git reset --hard HEAD \
    && echo "Sparse checkout (phase 2) paths:" \
    && git sparse-checkout list \
    && echo "Git status after expanding sparse set (should be clean):" \
    && git status --porcelain

## Install project. Using --reinstall-package to ensure wheel build picks up version.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "Pre-install (project) git status (should be clean):" && git status --porcelain && \
    uv sync --no-dev --reinstall-package imas-mcp --no-editable --frozen && \
    if [ -n "$(git status --porcelain uv.lock)" ]; then echo "uv.lock changed during project install (lock out of date). Run 'uv lock' and recommit." >&2; exit 1; fi && \
    echo "Post-install git status (should still be clean):" && git status --porcelain && \
    if [ -n "$(git status --porcelain)" ]; then \
        echo "Git tree became dirty during project install (will cause dev version)" >&2; exit 1; \
    else \
        echo "Repository clean; hatch-vcs should emit tag version"; \
    fi

# Build schema data
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "Building schema data..." && \
    if [ -n "${IDS_FILTER}" ]; then \
    echo "Building schema data for IDS: ${IDS_FILTER}" && \
    uv run --no-dev build-schemas --ids-filter "${IDS_FILTER}" --no-rich; \
    else \
    echo "Building schema data for all IDS" && \
    uv run --no-dev build-schemas --no-rich; \
    fi && \
    echo "✓ Schema data ready"

# Build embeddings (conditional on IDS_FILTER)
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=secret,id=OPENAI_API_KEY \
    export OPENAI_API_KEY=$(cat /run/secrets/OPENAI_API_KEY) && \
    echo "Building embeddings..." && \
    if [ -n "${IDS_FILTER}" ]; then \
    echo "Building embeddings for IDS: ${IDS_FILTER}" && \
    uv run --no-dev build-embeddings --ids-filter "${IDS_FILTER}" --no-rich; \
    else \
    echo "Building embeddings for all IDS" && \
    uv run --no-dev build-embeddings --no-rich; \
    fi && \
    echo "✓ Embeddings ready"

# Build relationships (requires embeddings)
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=secret,id=OPENAI_API_KEY \
    export OPENAI_API_KEY=$(cat /run/secrets/OPENAI_API_KEY 2>/dev/null || echo "") && \
    echo "Building relationships..." && \
    if [ -n "${IDS_FILTER}" ]; then \
    echo "Building relationships for IDS: ${IDS_FILTER}" && \
    uv run --no-dev build-relationships --ids-filter "${IDS_FILTER}" --quiet; \
    else \
    echo "Building relationships for all IDS" && \
    uv run --no-dev build-relationships --quiet; \
    fi && \
    echo "✓ Relationships ready"

# Build mermaid graphs (requires schemas)
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "Building mermaid graphs..." && \
    if [ -n "${IDS_FILTER}" ]; then \
    echo "Building mermaid graphs for IDS: ${IDS_FILTER}" && \
    uv run --no-dev build-mermaid --ids-filter "${IDS_FILTER}" --quiet; \
    else \
    echo "Building mermaid graphs for all IDS" && \
    uv run --no-dev build-mermaid --quiet; \
    fi && \
    echo "✓ Mermaid graphs ready"

## Stage 3: Use pre-scraped documentation (from CI cache)
FROM python:3.12-slim AS docs-provider

# Documentation already available in build context from CI
# No copying needed - will be copied directly to final stage

## Stage 4: Final runtime image (assemble from builder + direct docs copy)
FROM python:3.12-slim

# Install Node.js for runtime docs server support
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Install docs-mcp-server for runtime
ARG DOCS_MCP_SERVER_VERSION=1.29.0
RUN npm install -g @arabold/docs-mcp-server@${DOCS_MCP_SERVER_VERSION}

# Copy Python app from builder stage
COPY --from=builder /bin/uv /bin/
COPY --from=builder /app /app

# Copy scraped documentation directly from build context (CI artifacts)
COPY docs-data/ /app/data

# Set working directory
WORKDIR /app

# Set runtime environment variables
# Note: OPENAI_API_KEY should be passed at runtime via docker run -e or docker-compose
ENV PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DOCS_SERVER_URL=http://localhost:6280 \
    DOCS_MCP_TELEMETRY=false \
    DOCS_MCP_STORE_PATH=/app/data \
    OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
    IMAS_MCP_EMBEDDING_MODEL=qwen/qwen3-embedding-4b \
    DOCS_MCP_EMBEDDING_MODEL=qwen/qwen3-embedding-4b

# Expose port (only needed for streamable-http transport)
EXPOSE 8000

## Run via uv to ensure the synced environment is activated; additional args appended after CMD
ENTRYPOINT ["uv", "run", "--no-dev", "imas-mcp"]
CMD ["--no-rich", "--host", "0.0.0.0", "--port", "8000"]