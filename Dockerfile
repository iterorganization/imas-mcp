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

# Build generated Python models (graph models, physics domains)
# These are normally built by hatch build hook, but HATCH_BUILD_NO_HOOKS=true
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "Building generated models..." && \
    uv run --no-dev build-models --force && \
    echo "✓ Generated models ready"

# ── Graph-native data: pull IMAS-only graph from GHCR ──────────────────
# Replaces build-schemas, build-path-map, build-embeddings, clusters build.
# The graph contains all DD structure, embeddings, clusters, and version history.
# Build FAILS if graph cannot be downloaded — no fallback to file-based data.

# Install oras CLI for OCI artifact pull
ARG ORAS_VERSION=1.2.0
RUN curl -sLO "https://github.com/oras-project/oras/releases/download/v${ORAS_VERSION}/oras_${ORAS_VERSION}_linux_amd64.tar.gz" \
    && tar -xzf "oras_${ORAS_VERSION}_linux_amd64.tar.gz" -C /usr/local/bin oras \
    && rm -f "oras_${ORAS_VERSION}_linux_amd64.tar.gz"

# Pull IMAS-only graph from GHCR (requires GHCR_TOKEN build secret)
ARG GHCR_REGISTRY="ghcr.io/simon-mcintosh"
ARG GRAPH_TAG="latest"
RUN --mount=type=secret,id=GHCR_TOKEN \
    GHCR_TOKEN=$(cat /run/secrets/GHCR_TOKEN 2>/dev/null || echo "") && \
    if [ -z "$GHCR_TOKEN" ]; then \
        echo "ERROR: GHCR_TOKEN build secret is required to download the graph." >&2; \
        echo "Pass it with: --secret id=GHCR_TOKEN,env=GHCR_TOKEN" >&2; \
        exit 1; \
    fi && \
    echo "$GHCR_TOKEN" | oras login ghcr.io --username __token__ --password-stdin && \
    mkdir -p /tmp/graph-pull && \
    echo "Pulling IMAS-only graph: ${GHCR_REGISTRY}/imas-codex-graph-imas:${GRAPH_TAG}" && \
    oras pull "${GHCR_REGISTRY}/imas-codex-graph-imas:${GRAPH_TAG}" -o /tmp/graph-pull && \
    echo "✓ Graph downloaded" && \
    ls -la /tmp/graph-pull/

## Stage 4: Load graph dump into Neo4j data directory
# Uses neo4j-admin from the Neo4j image to load the dump
FROM neo4j:2026.01.4-community AS graph-loader

# Copy graph archive from builder
COPY --from=builder /tmp/graph-pull/ /tmp/graph-pull/

# Extract and load the graph dump
RUN set -e && \
    cd /tmp/graph-pull && \
    ARCHIVE=$(ls *.tar.gz 2>/dev/null | head -1) && \
    if [ -z "$ARCHIVE" ]; then \
        echo "ERROR: No graph archive found in /tmp/graph-pull/" >&2; \
        ls -la /tmp/graph-pull/ >&2; \
        exit 1; \
    fi && \
    echo "Extracting: $ARCHIVE" && \
    mkdir -p /tmp/graph-extracted && \
    tar -xzf "$ARCHIVE" -C /tmp/graph-extracted && \
    DUMP=$(find /tmp/graph-extracted -name "*.dump" -type f | head -1) && \
    if [ -z "$DUMP" ]; then \
        echo "ERROR: No .dump file found in archive" >&2; \
        find /tmp/graph-extracted -type f >&2; \
        exit 1; \
    fi && \
    echo "Loading dump: $DUMP" && \
    neo4j-admin database load neo4j --from-path="$(dirname $DUMP)" --overwrite-destination && \
    echo "✓ Graph loaded into Neo4j data directory" && \
    rm -rf /tmp/graph-pull /tmp/graph-extracted

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
RUN mkdir -p /opt/neo4j/logs && \
    echo "server.bolt.listen_address=127.0.0.1:7687" >> /opt/neo4j/conf/neo4j.conf && \
    echo "server.http.listen_address=127.0.0.1:7474" >> /opt/neo4j/conf/neo4j.conf && \
    echo "server.default_listen_address=127.0.0.1" >> /opt/neo4j/conf/neo4j.conf && \
    echo "server.memory.heap.initial_size=256m" >> /opt/neo4j/conf/neo4j.conf && \
    echo "server.memory.heap.max_size=512m" >> /opt/neo4j/conf/neo4j.conf && \
    echo "dbms.security.auth_enabled=false" >> /opt/neo4j/conf/neo4j.conf

ENV NEO4J_HOME=/opt/neo4j

# Copy Python app from builder stage
COPY --from=builder /bin/uv /bin/
COPY --from=builder /app /app

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Set working directory
WORKDIR /app

# Re-declare build args for runtime stage
ARG IDS_FILTER=""

# Set runtime environment variables
# NEO4J_URI points to embedded Neo4j (internal, not exposed externally)
# IMAS_CODEX_GRAPH_NATIVE=1 activates graph-native mode — all data from Neo4j
ENV PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
    PATH="/app/.venv/bin:${PATH}" \
    IDS_FILTER=${IDS_FILTER} \
    NEO4J_URI="bolt://127.0.0.1:7687" \
    NEO4J_USERNAME="neo4j" \
    NEO4J_PASSWORD="neo4j" \
    IMAS_CODEX_GRAPH_NATIVE="1"

# Expose MCP server port (Neo4j ports are internal only)
EXPOSE 8000

# Health check verifies both Neo4j and MCP server
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD curl -sf http://127.0.0.1:7474/ > /dev/null && \
        curl -sf http://127.0.0.1:8000/health > /dev/null

## Entrypoint starts Neo4j then MCP server
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["--host", "0.0.0.0", "--port", "8000"]