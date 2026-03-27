## Stage 1: acquire uv binary (kept minimal)
ARG UV_VERSION=0.7.13
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

## Stage 2: Neo4j + JRE binaries (copied into final image)
FROM neo4j:2026.01.4-community AS neo4j-src

## Stage 3: Build complete project
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
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary
COPY --from=uv /uv /bin/

# Set working directory
WORKDIR /app

# Add build args for IDS filter
ARG IDS_FILTER=""

# Additional build-time metadata for cache busting & traceability
ARG GIT_SHA=""
ARG GIT_TAG=""
ARG GIT_REF=""

# Set environment variables
ENV PYTHONPATH="/app" \
    IDS_FILTER=${IDS_FILTER} \
    TRANSPORT="streamable-http" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HATCH_BUILD_NO_HOOKS=true \
    OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

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

# Configure PyTorch CPU-only to minimize image size
ENV UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"

## Install only dependencies without installing the local project (frozen = must match committed lock)
# Lock file already specifies CPU-only PyTorch (no nvidia-* CUDA deps)
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --no-dev --no-install-project --frozen --extra cpu || \
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
    uv sync --no-dev --reinstall-package imas-codex --no-editable --frozen --extra cpu && \
    if [ -n "$(git status --porcelain uv.lock)" ]; then echo "uv.lock changed during project install (lock out of date). Run 'uv lock' and recommit." >&2; exit 1; fi && \
    echo "Post-install git status (should still be clean):" && git status --porcelain && \
    if [ -n "$(git status --porcelain)" ]; then \
        echo "Git tree became dirty during project install (will cause dev version)" >&2; exit 1; \
    else \
        echo "Repository clean; hatch-vcs should emit tag version"; \
    fi

# Build generated Python models (graph models, physics domains)
# These are normally built by hatch build hook, but HATCH_BUILD_NO_HOOKS=true.
# linkml/linkml-runtime are build-system requires — install temporarily for gen-pydantic.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "Installing build-requires for model generation..." && \
    uv pip install linkml linkml-runtime && \
    echo "Building generated models..." && \
    uv run --no-dev build-models --force && \
    echo "✓ Generated models ready"

# Pre-download embedding model for offline operation
ENV HF_HOME=/app/.cache/huggingface
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    for i in 1 2 3; do \
      uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True)" && \
      echo "✓ Embedding model cached" && break || \
      echo "⚠ Attempt $i failed, retrying in 10s..." && sleep 10; \
    done && \
    test -d /app/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B

# ── Graph-native data: pull IMAS-only graph from GHCR ──────────────────
# Replaces build-schemas, build-path-map, build-embeddings, clusters build.
# The graph contains all DD structure, embeddings, clusters, and version history.
# Builds gracefully without graph data if the package is unavailable.

# Install oras CLI for OCI artifact pull
ARG ORAS_VERSION=1.2.0
RUN curl -sLO "https://github.com/oras-project/oras/releases/download/v${ORAS_VERSION}/oras_${ORAS_VERSION}_linux_amd64.tar.gz" \
    && tar -xzf "oras_${ORAS_VERSION}_linux_amd64.tar.gz" -C /usr/local/bin oras \
    && rm -f "oras_${ORAS_VERSION}_linux_amd64.tar.gz"

# Pull graph from GHCR (optional — builds without graph if unavailable)
ARG GHCR_REGISTRY="ghcr.io/iterorganization"
ARG GRAPH_TAG="latest"
ARG GRAPH_PACKAGE="imas-codex-graph-dd"
RUN --mount=type=secret,id=GHCR_TOKEN \
    mkdir -p /tmp/graph-pull && \
    GHCR_TOKEN=$(cat /run/secrets/GHCR_TOKEN 2>/dev/null || echo "") && \
    if [ -z "$GHCR_TOKEN" ] || [ "$GRAPH_TAG" = "none" ]; then \
        echo "⚠ Skipping graph pull (token=${GHCR_TOKEN:+set}${GHCR_TOKEN:-missing}, tag=${GRAPH_TAG})" >&2; \
        touch /tmp/graph-pull/.no-graph; \
    elif echo "$GHCR_TOKEN" | oras login ghcr.io --username __token__ --password-stdin && \
         oras pull --allow-path-traversal \
           "${GHCR_REGISTRY}/${GRAPH_PACKAGE}:${GRAPH_TAG}" \
           -o /tmp/graph-pull; then \
        echo "✓ Graph downloaded"; \
        # Handle artifacts pushed with absolute paths (oras writes to original path)
        if ! ls /tmp/graph-pull/*.tar.gz >/dev/null 2>&1; then \
            FOUND=$(find /tmp -name "*.tar.gz" -not -path "/tmp/graph-pull/*" 2>/dev/null | head -1); \
            if [ -n "$FOUND" ]; then \
                echo "Moving archive from $FOUND to /tmp/graph-pull/"; \
                mv "$FOUND" /tmp/graph-pull/; \
            fi; \
        fi; \
        ls -la /tmp/graph-pull/; \
    else \
        echo "⚠ Graph pull failed — building without pre-loaded graph" >&2; \
        touch /tmp/graph-pull/.no-graph; \
    fi

## Stage 4: Load graph dump into Neo4j data directory
# Uses neo4j-admin from the Neo4j image to load the dump
FROM neo4j:2026.01.4-community AS graph-loader

# Propagate GRAPH_TAG to bust cache when graph version changes
ARG GRAPH_TAG="latest"
RUN echo "Graph tag: ${GRAPH_TAG}" > /dev/null

# Copy graph archive from builder
COPY --from=builder /tmp/graph-pull/ /tmp/graph-pull/

# Extract and load the graph dump (or create empty database)
# Handles both raw .dump files (oras pull) and .tar.gz archives.
# CRITICAL: clean up intermediate files progressively to minimize peak disk usage.
# The graph dump is ~5 GB; without cleanup we'd have archive + extracted + copy + loaded
# data all on disk simultaneously (~15+ GB), exceeding CI runner capacity.
RUN set -ex && \
    if [ -f /tmp/graph-pull/.no-graph ]; then \
        echo "⚠ No graph data — creating empty Neo4j database"; \
        mkdir -p /data/databases/neo4j /data/transactions/neo4j; \
    else \
        cd /tmp/graph-pull && \
        DUMP=$(ls *.dump 2>/dev/null | head -1) && \
        ARCHIVE=$(ls *.tar.gz 2>/dev/null | head -1) && \
        mkdir -p /tmp/dumps && \
        if [ -n "$DUMP" ]; then \
            echo "Loading dump directly: $DUMP" && \
            mv "$DUMP" /tmp/dumps/neo4j.dump && \
            rm -rf /tmp/graph-pull; \
        elif [ -n "$ARCHIVE" ]; then \
            echo "Extracting: $ARCHIVE" && \
            mkdir -p /tmp/graph-extracted && \
            tar -xzf "$ARCHIVE" -C /tmp/graph-extracted && \
            rm -rf /tmp/graph-pull && \
            DUMP=$(find /tmp/graph-extracted -name "*.dump" -type f | head -1) && \
            if [ -z "$DUMP" ]; then \
                echo "ERROR: No .dump file found in archive" >&2; \
                find /tmp/graph-extracted -type f >&2; \
                exit 1; \
            fi && \
            echo "Found dump: $DUMP ($(du -sh "$DUMP" | cut -f1))" && \
            mv "$DUMP" /tmp/dumps/neo4j.dump && \
            rm -rf /tmp/graph-extracted; \
        else \
            echo "ERROR: No .dump or .tar.gz found in /tmp/graph-pull/" >&2; \
            ls -la /tmp/graph-pull/ >&2; \
            exit 1; \
        fi && \
        echo "Loading dump into Neo4j ($(du -sh /tmp/dumps/neo4j.dump | cut -f1))..." && \
        df -h / && \
        cd / && \
        neo4j-admin database load neo4j --from-path=/tmp/dumps --overwrite-destination 2>&1 && \
        rm -rf /tmp/dumps && \
        echo "✓ Graph loaded into Neo4j data directory"; \
    fi

# NOTE: Do NOT remove transaction logs (/data/transactions/neo4j/*).
# Neo4j 2026 requires valid WAL state to open the database after
# neo4j-admin load.  Deleting tx logs causes Neo4j HTTP to start but
# the bolt database remains offline — all Cypher queries fail silently.
# The ~2.3 GB cost is acceptable for a working container.

## Stage 5: Final runtime image (assemble from builder + Neo4j + graph data)
FROM python:3.12-slim

# Install runtime dependencies (curl for health checks, procps for process mgmt)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy JRE from Neo4j image (Neo4j requires Java 21+)
COPY --from=neo4j-src /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Copy Neo4j binaries (without the Docker entrypoint — we use our own)
COPY --from=neo4j-src /var/lib/neo4j /opt/neo4j
COPY --from=neo4j-src /var/lib/neo4j/conf /opt/neo4j/conf

# Copy pre-loaded graph data from graph-loader stage
COPY --from=graph-loader /data /opt/neo4j/data

# Configure Neo4j for embedded use (read-only, internal bolt only)
RUN rm -rf /opt/neo4j/logs && mkdir -p /opt/neo4j/logs && \
    echo "server.bolt.listen_address=127.0.0.1:7687" >> /opt/neo4j/conf/neo4j.conf && \
    echo "server.http.listen_address=127.0.0.1:7474" >> /opt/neo4j/conf/neo4j.conf && \
    echo "server.default_listen_address=127.0.0.1" >> /opt/neo4j/conf/neo4j.conf && \
    echo "server.memory.heap.initial_size=256m" >> /opt/neo4j/conf/neo4j.conf && \
    echo "server.memory.heap.max_size=512m" >> /opt/neo4j/conf/neo4j.conf && \
    echo "dbms.security.auth_enabled=false" >> /opt/neo4j/conf/neo4j.conf

ENV NEO4J_HOME=/opt/neo4j

# Copy Python app from builder stage (exclude .git to save ~21 MB)
COPY --from=builder /bin/uv /bin/
COPY --from=builder /app /app
RUN rm -rf /app/.git

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Set working directory
WORKDIR /app

# Re-declare build args for runtime stage
ARG IDS_FILTER=""

# Set runtime environment variables
# NEO4J_URI points to embedded Neo4j (internal, not exposed externally)
ENV PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENROUTER_BASE_URL=https://openrouter.ai/api/v1 \
    PATH="/app/.venv/bin:${PATH}" \
    IDS_FILTER=${IDS_FILTER} \
    TRANSPORT="streamable-http" \
    NEO4J_URI="bolt://127.0.0.1:7687" \
    NEO4J_USERNAME="neo4j" \
    NEO4J_PASSWORD="neo4j" \
    IMAS_CODEX_GRAPH_LOCATION=local \
    IMAS_CODEX_EMBEDDING_LOCATION=local \
    HF_HOME=/app/.cache/huggingface

# Expose MCP server port (Neo4j ports are internal only)
# PORT env var tells Azure App Service / Cloud Run which port to probe
EXPOSE 8000
ENV PORT=8000

# Health check verifies both Neo4j and MCP server
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD curl -sf http://127.0.0.1:7474/ > /dev/null && \
        curl -sf http://127.0.0.1:8000/health > /dev/null

## Entrypoint starts Neo4j then MCP server
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["serve", "--read-only", "--host", "0.0.0.0", "--port", "8000"]