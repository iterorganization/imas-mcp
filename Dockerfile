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
    HATCH_BUILD_NO_HOOKS=true \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/root/.cache/sentence-transformers

# Copy dependency files and git metadata 
COPY .git/ ./.git/
RUN echo '=== GIT DIAG: after-copy-git ===' && git describe --tags --always --dirty && git status --porcelain || true
COPY pyproject.toml ./
COPY README.md ./
COPY hatch_build_hooks.py ./
RUN echo '=== GIT DIAG: after-meta-files ===' && git describe --tags --always --dirty && git status --porcelain || true

# Ensure git repository is properly initialized for version detection
RUN git config --global --add safe.directory /app

# Install only dependencies without the local project to avoid build hooks
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --no-dev --no-install-project --extra http --extra build && \
    echo '=== GIT DIAG: after-deps-sync ===' && git describe --tags --always --dirty && git status --porcelain || true

# Copy source code (separate layer for better caching)
COPY imas_mcp/ ./imas_mcp/
COPY scripts/ ./scripts/
RUN echo '=== GIT DIAG: after-source-copy ===' && git describe --tags --always --dirty && git status --porcelain || true

# Install project with HTTP and build support for container deployment
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --no-dev --extra http --extra build && \
    echo '=== GIT DIAG: after-project-install ===' && git describe --tags --always --dirty && git status --porcelain || true

# Install imas-data-dictionary manually from git (needed for index building)
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install "imas-data-dictionary @ git+https://github.com/iterorganization/imas-data-dictionary.git@c1342e2514ba36d007937425b2df522cd1b213df" && \
    echo '=== GIT DIAG: after-dd-install ===' && git describe --tags --always --dirty && git status --porcelain || true

# Build schema data
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "Building schema data..." && \
    uv run --no-dev build-schemas --no-rich && \
    echo "✓ Schema data ready" && \
    echo '=== GIT DIAG: after-build-schemas ===' && git describe --tags --always --dirty && git status --porcelain || true

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
    echo "✓ Embeddings ready" && \
    echo '=== GIT DIAG: after-build-embeddings ===' && git describe --tags --always --dirty && git status --porcelain || true

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
    echo "✓ Relationships ready" && \
    echo '=== GIT DIAG: after-build-relationships ===' && git describe --tags --always --dirty && git status --porcelain || true

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
    echo "✓ Mermaid graphs ready" && \
    echo '=== GIT DIAG: after-build-mermaid ===' && git describe --tags --always --dirty && git status --porcelain || true

# Expose port (only needed for streamable-http transport)
EXPOSE 8000

RUN echo '=== GIT DIAG: final-before-entrypoint ===' && git describe --tags --always --dirty && git status --porcelain || true

ENTRYPOINT ["python", "-m", "imas_mcp.cli"]
CMD ["--no-rich", "--host", "0.0.0.0", "--port", "8000"]