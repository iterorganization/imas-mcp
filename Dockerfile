## Stage 1: acquire uv binary (kept minimal)
ARG UV_VERSION=0.7.13
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

## Stage 2: Build complete project
FROM python:3.12-slim AS builder

# Install system dependencies including git for git dependencies
# and gcc/build tools for packages with C extensions (hdbscan, etc.)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary
COPY --from=uv /uv /bin/

# Set working directory
WORKDIR /app

# Add build args for IDS filter and transport
ARG IDS_FILTER=""
ARG TRANSPORT="streamable-http"
# IMAS_DD_VERSION: Optional. If empty, Python code reads default from pyproject.toml [tool.imas-codex].default-dd-version
# This ensures a single source of truth. Pass explicitly only to override.
ARG IMAS_DD_VERSION=""

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
    OPENAI_BASE_URL=https://openrouter.ai/api/v1

# Labels for image provenance
LABEL imas_codex.git_sha=${GIT_SHA} \
      imas_codex.git_tag=${GIT_TAG} \
      imas_codex.git_ref=${GIT_REF}

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
# Lock file already specifies CPU-only PyTorch (no nvidia-* CUDA deps)
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --no-dev --no-install-project --frozen || \
    (echo "Dependency sync failed (lock mismatch). Run 'uv lock' locally and commit changes." >&2; exit 1) && \
    if [ -n "$(git status --porcelain uv.lock)" ]; then echo "uv.lock changed during dep sync (unexpected)." >&2; exit 1; fi

# Expand sparse checkout to include project sources, scripts, and build hooks (phase 2)
# Include hatch_build_hooks.py even though HATCH_BUILD_NO_HOOKS=true, because
# hatchling validates file existence before checking the env var
RUN git sparse-checkout set pyproject.toml uv.lock README.md imas_codex scripts hatch_build_hooks.py \
    && git reset --hard HEAD \
    && echo "Sparse checkout (phase 2) paths:" \
    && git sparse-checkout list \
    && echo "Git status after expanding sparse set (should be clean):" \
    && git status --porcelain

## Install project. Using --reinstall-package to ensure wheel build picks up version.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "Pre-install (project) git status (should be clean):" && git status --porcelain && \
    uv sync --no-dev --reinstall-package imas-codex --no-editable --frozen && \
    if [ -n "$(git status --porcelain uv.lock)" ]; then echo "uv.lock changed during project install (lock out of date). Run 'uv lock' and recommit." >&2; exit 1; fi && \
    echo "Post-install git status (should still be clean):" && git status --porcelain && \
    if [ -n "$(git status --porcelain)" ]; then \
        echo "Git tree became dirty during project install (will cause dev version)" >&2; exit 1; \
    else \
        echo "Repository clean; hatch-vcs should emit tag version"; \
    fi

# Note: We do NOT copy pre-built resources from local filesystem.
# The build scripts below generate all resources for the specific DD version.
# This ensures consistency and avoids copying stale or multi-version data.
# In CI, resources may be pre-generated as artifacts and copied separately.

# Build generated Python models (graph models, physics domains)
# These are normally built by hatch build hook, but HATCH_BUILD_NO_HOOKS=true
# Must run BEFORE build-schemas which imports physics_domain.py
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "Building generated models..." && \
    uv run --no-dev build-models --force && \
    echo "✓ Generated models ready"

# Cache bust: Explicit ARG ensures rebuild when DD version changes.
# IDS_FILTER is tracked automatically via its use in RUN commands.
# IMAS_DD_VERSION needs explicit tracking since it may be empty (default from pyproject.toml).
ARG CACHE_BUST_DD="${IMAS_DD_VERSION}"

# Build schema data (will skip if already exists from CI artifacts)
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "DD version cache key: ${CACHE_BUST_DD:-default}" && \
    echo "Building schema data..." && \
    if [ -n "${IDS_FILTER}" ]; then \
    echo "Building schema data for IDS: ${IDS_FILTER}" && \
    uv run --no-dev build-schemas --ids-filter "${IDS_FILTER}" --no-rich; \
    else \
    echo "Building schema data for all IDS" && \
    uv run --no-dev build-schemas --no-rich; \
    fi && \
    echo "✓ Schema data ready"

# Build path map for version upgrade mappings (will skip if already exists from CI artifacts)
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "Building path map..." && \
    if [ -n "${IDS_FILTER}" ]; then \
    echo "Building path map for IDS: ${IDS_FILTER}" && \
    uv run --no-dev build-path-map --ids-filter "${IDS_FILTER}" --no-rich; \
    else \
    echo "Building path map for all IDS" && \
    uv run --no-dev build-path-map --no-rich; \
    fi && \
    echo "✓ Path map ready"

# Build embeddings using local sentence-transformers model (all-MiniLM-L6-v2)
# NO API key required - runs entirely locally
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

# Build clusters using HDBSCAN (local algorithm)
# OPENAI_API_KEY is OPTIONAL - used only for LLM-generated cluster labels
# If no key is provided, falls back to:
#   1. Cached labels from definitions/clusters/labels.json (version-controlled)
#   2. Auto-generated fallback labels from path names
# Note: Secret id=openai_api_key for docker-compose, id=OPENAI_API_KEY for manual builds
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=secret,id=openai_api_key \
    --mount=type=secret,id=OPENAI_API_KEY \
    export OPENAI_API_KEY=$(cat /run/secrets/openai_api_key 2>/dev/null || cat /run/secrets/OPENAI_API_KEY 2>/dev/null || echo "") && \
    echo "Building clusters..." && \
    if [ -n "${IDS_FILTER}" ]; then \
    echo "Building clusters for IDS: ${IDS_FILTER}" && \
    uv run --no-dev build-clusters --ids-filter "${IDS_FILTER}" --quiet; \
    else \
    echo "Building clusters for all IDS" && \
    uv run --no-dev build-clusters --quiet; \
    fi && \
    echo "✓ Clusters ready"

## Stage 3: Final runtime image (assemble from builder)
FROM python:3.12-slim

# Copy Python app from builder stage
COPY --from=builder /bin/uv /bin/
COPY --from=builder /app /app

# Set working directory
WORKDIR /app

# Re-declare build args for runtime stage
ARG IDS_FILTER=""

# Set runtime environment variables
# Note: OPENAI_API_KEY should be passed at runtime via docker run -e or docker-compose
# IDS_FILTER is persisted from build time to ensure runtime uses same subset
ENV PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
    PATH="/app/.venv/bin:$PATH" \
    IDS_FILTER=${IDS_FILTER}

# Expose port (only needed for streamable-http transport)
EXPOSE 8000

## Run the installed CLI directly from venv (avoids uv sync at runtime)
ENTRYPOINT ["imas-codex"]
CMD ["--no-rich", "--host", "0.0.0.0", "--port", "8000"]