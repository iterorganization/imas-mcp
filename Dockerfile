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

# Additional build-time metadata for cache busting & traceability
ARG GIT_SHA=""
ARG GIT_TAG=""
ARG GIT_REF=""

# Set environment variables
ENV PYTHONPATH="/app" \
    IDS_FILTER=${IDS_FILTER} \
    TRANSPORT=${TRANSPORT} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HATCH_BUILD_NO_HOOKS=true \
    IMAS_MCP_COMMIT=${GIT_SHA} \
    IMAS_MCP_TAG=${GIT_TAG} \
    IMAS_MCP_REF=${GIT_REF}

# Labels for image provenance
LABEL imas_mcp.git_sha=${GIT_SHA} \
      imas_mcp.git_tag=${GIT_TAG} \
      imas_mcp.git_ref=${GIT_REF}

## Copy git metadata first so hatch-vcs sees repository state exactly as on tag
COPY .git/ ./.git/
RUN git config --global --add safe.directory /app

## Copy project metadata (tracked files)
COPY pyproject.toml ./
COPY README.md ./
COPY uv.lock ./

## Install only dependencies without installing the local project. Use --frozen so uv.lock isn't modified.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --no-dev --no-install-project --extra http --extra build --frozen || \
    (echo "Dependency sync failed (lock mismatch). Run 'uv lock' locally and commit changes." >&2; exit 1)

# Copy source code (separate layer for better caching)
COPY imas_mcp/ ./imas_mcp/
COPY scripts/ ./scripts/

## Install project with HTTP/build extras. Using --reinstall-package to ensure wheel build picks up version.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "Pre-install git status (should be clean):" && git status --porcelain && \
    uv sync --no-dev --reinstall-package imas-mcp --extra http --extra build --frozen && \
    echo "Post-install git status (should still be clean):" && git status --porcelain && \
    if [ -n "$(git status --porcelain)" ]; then \
        echo "Git tree became dirty during project install (will cause dev version)" >&2; exit 1; \
    else \
        echo "Repository clean; hatch-vcs should emit tag version"; \
    fi

# Install imas-data-dictionary manually from git (needed for index building)
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install "imas-data-dictionary @ git+https://github.com/iterorganization/imas-data-dictionary.git@c1342e2514ba36d007937425b2df522cd1b213df"

# Build schema data
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "Building schema data..." && \
    uv run --no-dev build-schemas --no-rich && \
    echo "✓ Schema data ready"

# Build embeddings (conditional on IDS_FILTER)
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
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

# Expose port (only needed for streamable-http transport)
EXPOSE 8000

## Run via uv to ensure the synced environment is activated; additional args appended after CMD
ENTRYPOINT ["uv", "run", "--no-dev", "imas-mcp"]
CMD ["--no-rich", "--host", "0.0.0.0", "--port", "8000"]