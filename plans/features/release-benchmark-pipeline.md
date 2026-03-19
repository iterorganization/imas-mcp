# Release Benchmark Pipeline

**Status**: Plan  
**Created**: 2026-03-19  
**Scope**: CI pipeline, benchmark framework, performance tracking  

## Problem Statement

The current benchmark workflow is fundamentally broken:

1. **Empty database**: Benchmarks run against an empty Neo4j instance — every tool call that touches the graph returns nothing or errors. Results are meaningless.
2. **Every push**: Benchmarks run on every push to `main`, but graph data only changes on tagged releases (via `graph push` → GHCR). Per-commit noise obscures real trends.
3. **Missing artifacts**: The `schema_context_data.py` generated file isn't included in the wheel, causing `ModuleNotFoundError` at runtime.
4. **Narrow coverage**: Only 7 timing tests across 2 ASV classes (`SearchBenchmarks`, `ClusterSearchBenchmarks`), all calling just `search_imas` and `search_imas_clusters`. The project has 30 MCP tools and ~10 performance-critical subsystems.
5. **No regression gates**: Results are tracked on gh-pages but no CI step fails on performance regressions.

## Design Goals

- Benchmark against the **real GHCR graph dump** (same data deployed to production)
- Track performance **across tagged releases only** (when data actually changes)
- Cover all **user-facing read tools** (22 MCP tools) plus key subsystems
- **Do not benchmark**: LLM calls, discovery pipelines, write tools, dev-only scripts
- Enable **parallel implementation** by independent agents via clear module boundaries
- Maintain the existing ASV infrastructure for trend visualization on gh-pages

## Architecture

### Workflow Trigger

```yaml
on:
  push:
    tags: ["v*"]          # Tagged releases only
  workflow_dispatch:       # Manual trigger for debugging
    inputs:
      graph_tag:
        description: "Graph version tag (default: latest)"
        default: "latest"
      benchmark_filter:
        description: "ASV benchmark filter (e.g., SearchToolBenchmarks)"
        default: ""
```

### Workflow Steps

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Checkout + restore gh-pages benchmark history               │
│  2. Setup Python 3.12 + UV + ASV                                │
│  3. Start Neo4j service container                               │
│  4. Pull graph dump from GHCR (oras pull)                       │
│  5. Stop Neo4j → neo4j-admin database load → restart Neo4j      │
│  6. Verify graph loaded (sanity query)                          │
│  7. Build wheel (triggers hatch hooks → schema_context_data.py) │
│  8. Run ASV benchmarks                                          │
│  9. Compare against previous release (regression detection)     │
│ 10. Generate HTML report + deploy to gh-pages                   │
└─────────────────────────────────────────────────────────────────┘
```

Steps 4–6 are adapted directly from `graph-quality.yml` which already does this reliably.

### Benchmark Module Structure

```
benchmarks/
├── __init__.py                      # BenchmarkRunner utility (existing)
├── benchmark_runner.py              # ASV runner utility (existing)
├── conftest_bench.py                # Shared fixtures: graph client, encoder, server
│
├── bench_mcp_search.py              # Phase 1A: Search tool benchmarks
├── bench_mcp_imas_tools.py          # Phase 1B: IMAS DD tool benchmarks
├── bench_mcp_facility_tools.py      # Phase 1C: Facility tool benchmarks (signals, docs, code)
│
├── bench_graph_queries.py           # Phase 2A: Raw Cypher query patterns
├── bench_embeddings.py              # Phase 2B: Encoder performance (no graph)
├── bench_query_builder.py           # Phase 2C: graph_search() Cypher generation (no graph)
│
├── bench_subsystems.py              # Phase 3:  COCOS, units, schema_for()
├── bench_server_startup.py          # Phase 3:  Cold start time
└── bench_memory.py                  # Phase 3:  Peak memory for key operations
```

ASV discovers benchmark classes by scanning `benchmark_dir` for files. Each file
is an independent module with no cross-imports (except shared fixtures), enabling
parallel development by separate agents.

---

## Phase 1 — MCP Tool Benchmarks (Graph-Dependent)

### Goal

Measure end-to-end latency of all 22 read-only MCP tools through the FastMCP client,
against a real GHCR graph dump. These are the numbers that matter to users.

### Shared Fixture (`conftest_bench.py`)

```python
# Re-used across all MCP benchmark modules
class MCPFixture:
    """Lazy-loaded MCP server + client for benchmarks."""

    @cached_property
    def server(self):
        from imas_codex.llm.server import AgentsServer
        return AgentsServer()

    @cached_property
    def client(self):
        from fastmcp import Client
        return Client(self.server.mcp)

    @cached_property
    def graph_client(self):
        from imas_codex.graph.client import GraphClient
        return GraphClient.from_profile()

_fixture = MCPFixture()
```

### Phase 1A — Search Tool Benchmarks (`bench_mcp_search.py`)

**Tools**: `search_imas`, `search_imas_clusters`, `get_imas_path_context`

| Benchmark | Tool | Parameters | What it measures |
|-----------|------|------------|-----------------|
| `time_search_imas_basic` | `search_imas` | `query="electron temperature", k=10` | Baseline hybrid search |
| `time_search_imas_filtered` | `search_imas` | `+ ids_filter="core_profiles"` | IDS-filtered search |
| `time_search_imas_with_version` | `search_imas` | `+ include_version_context=True` | Version enrichment overhead |
| `time_search_imas_large_k` | `search_imas` | `k=100` | Scaling with result count |
| `time_search_imas_complex_query` | `search_imas` | `query="magnetic field topology near X-point"` | Multi-word semantic query |
| `time_search_clusters_semantic` | `search_imas_clusters` | `query="temperature profiles"` | Cluster centroid search |
| `time_search_clusters_filtered` | `search_imas_clusters` | `+ ids_filter="equilibrium"` | Filtered cluster search |
| `time_search_clusters_by_path` | `search_imas_clusters` | `query="core_profiles/profiles_1d/electrons/temperature"` | Path-based cluster lookup |
| `time_path_context_all` | `get_imas_path_context` | `path="…/temperature", relationship_types="all"` | Full cross-IDS analysis |
| `time_path_context_cluster` | `get_imas_path_context` | `relationship_types="cluster"` | Cluster-only relationships |
| `peakmem_search_imas_basic` | `search_imas` | `k=10` | Memory footprint |
| `peakmem_search_imas_large_k` | `search_imas` | `k=100` | Memory scaling |

**Agent assignment**: 1 agent, self-contained module.

### Phase 1B — IMAS DD Tool Benchmarks (`bench_mcp_imas_tools.py`)

**Tools**: `check_imas_paths`, `fetch_imas_paths`, `fetch_error_fields`, `list_imas_paths`, `get_imas_overview`, `get_imas_identifiers`, `analyze_imas_structure`, `export_imas_ids`, `export_imas_domain`, `get_dd_version_context`, `get_dd_versions`

| Benchmark | Tool | Parameters | What it measures |
|-----------|------|------------|-----------------|
| `time_check_paths_single` | `check_imas_paths` | 1 path | Single path validation |
| `time_check_paths_batch` | `check_imas_paths` | 10 comma-separated paths | Batch validation (N+1 cost) |
| `time_fetch_paths_single` | `fetch_imas_paths` | 1 path | Full path documentation |
| `time_fetch_paths_with_history` | `fetch_imas_paths` | `+ include_version_history=True` | Version history enrichment |
| `time_fetch_error_fields` | `fetch_error_fields` | 1 path with error data | Error field traversal |
| `time_list_paths_ids` | `list_imas_paths` | `paths="equilibrium"` (full IDS) | IDS enumeration |
| `time_list_paths_subtree` | `list_imas_paths` | `paths="equilibrium/time_slice/profiles_1d"` | Subtree enumeration |
| `time_list_paths_leaf_only` | `list_imas_paths` | `+ leaf_only=True` | Leaf filtering |
| `time_overview_all` | `get_imas_overview` | No filter | Full IDS summary scan |
| `time_overview_filtered` | `get_imas_overview` | `query="magnetics"` | Filtered overview with vector search |
| `time_identifiers` | `get_imas_identifiers` | `query="coordinate"` | Identifier schema search |
| `time_structure_analysis` | `analyze_imas_structure` | `ids_name="equilibrium"` | Hierarchical analysis |
| `time_export_ids` | `export_imas_ids` | `ids_name="core_profiles"` | Full IDS export |
| `time_export_domain` | `export_imas_domain` | `domain="equilibrium"` | Domain export |
| `time_dd_version_context` | `get_dd_version_context` | 3 paths | Version change history |
| `time_dd_versions` | `get_dd_versions` | (none) | Version metadata |
| `peakmem_export_ids` | `export_imas_ids` | Large IDS | Export memory |

**Agent assignment**: 1 agent, self-contained module.

### Phase 1C — Facility Tool Benchmarks (`bench_mcp_facility_tools.py`)

**Tools**: `search_signals`, `signal_analytics`, `search_docs`, `search_code`, `fetch`, `get_discovery_context`, `get_graph_schema`

These tools require **facility data** in the graph. The GHCR IMAS-only graph (`imas-codex-graph-imas`)
may not contain facility nodes. Two strategies:

1. **If facility data exists in dump**: Benchmark directly
2. **If IMAS-only dump**: Skip facility tools, benchmark `get_graph_schema` only

The benchmark should detect available data and skip gracefully:

```python
def setup(self):
    # Check if facility data exists in the loaded graph
    result = self.fixture.graph_client.query(
        "MATCH (f:Facility) RETURN count(f) AS n"
    )
    self.has_facility_data = result[0]["n"] > 0
    if not self.has_facility_data:
        raise NotImplementedError  # ASV skips benchmarks that raise this
```

| Benchmark | Tool | Requires Facility? | What it measures |
|-----------|------|--------------------|-----------------|
| `time_search_signals` | `search_signals` | Yes | Hybrid signal search with enrichment |
| `time_signal_analytics` | `signal_analytics` | Yes | Aggregate signal counts |
| `time_search_docs` | `search_docs` | Yes | Wiki/doc hybrid search |
| `time_search_code` | `search_code` | Yes | Code hybrid search |
| `time_fetch_by_id` | `fetch` | Yes | Content retrieval by ID |
| `time_fetch_by_title` | `fetch` | Yes | Fuzzy title matching |
| `time_discovery_context` | `get_discovery_context` | Yes | Coverage/gap analysis |
| `time_graph_schema_overview` | `get_graph_schema` | No | Schema overview (cached) |
| `time_graph_schema_signals` | `get_graph_schema` | No | Signals task scope |
| `time_graph_schema_imas` | `get_graph_schema` | No | IMAS task scope |

**Agent assignment**: 1 agent. Must handle the conditional skip pattern.

---

## Phase 2 — Subsystem Benchmarks

### Phase 2A — Raw Graph Query Benchmarks (`bench_graph_queries.py`)

**Dependency**: Neo4j with loaded graph  
**Purpose**: Isolate Neo4j query performance from MCP overhead. Track Cypher query efficiency independent of Python formatting/serialization.

| Benchmark | Query Pattern | What it measures |
|-----------|--------------|-----------------|
| `time_vector_search_imas` | `db.index.vector.queryNodes('imas_node_embedding', k, embedding)` | Vector index latency |
| `time_fulltext_search` | `db.index.fulltext.queryNodes('imas_node_text', query)` | BM25 text search |
| `time_path_traversal_enrichment` | `UNWIND paths + OPTIONAL MATCH` to Unit, DDVersion, Cluster | Multi-hop enrichment |
| `time_ids_aggregation` | `MATCH (i:IDS) RETURN i.id, i.path_count ORDER BY …` | Full IDS scan + sort |
| `time_version_chain_traversal` | `MATCH path = (v:DDVersion)-[:HAS_PREDECESSOR*]->()` | Recursive version chain |
| `time_prefix_scan` | `MATCH (n:IMASNode) WHERE n.path STARTS WITH $prefix` | B-tree index on `path` |
| `time_cluster_expansion` | `MATCH (c:IMASSemanticCluster)<-[:IN_CLUSTER]-(n:IMASNode)` | Cluster member expansion |
| `time_cross_ids_relationships` | Complex multi-OPTIONAL MATCH | Cross-IDS join cost |

**Agent assignment**: 1 agent. Uses `GraphClient` directly, no MCP layer.

### Phase 2B — Embedding Performance (`bench_embeddings.py`)

**Dependency**: None (no graph required)  
**Purpose**: Track embedding encode latency. This is the gating factor for all semantic search tools.

| Benchmark | Operation | What it measures |
|-----------|----------|-----------------|
| `time_encode_single_query` | `encoder.embed_texts(["electron temperature"])` | Single query latency |
| `time_encode_batch_10` | `encoder.embed_texts([…] * 10)` | Batch encode (10 texts) |
| `time_encode_batch_100` | `encoder.embed_texts([…] * 100)` | Batch encode (100 texts) |
| `time_encode_long_text` | `encoder.embed_texts(["<200 word description>"])` | Long text latency |
| `time_model_load` | Fresh `Encoder()` construction | Cold start model load |
| `peakmem_encode_batch_100` | `encoder.embed_texts([…] * 100)` | Batch memory footprint |

**Implementation note**: Force `IMAS_CODEX_EMBEDDING_LOCATION=local` (CPU) for reproducible CI results. The ASV benchmark environment already installs CPU-only PyTorch.

**Agent assignment**: 1 agent, completely independent of Neo4j.

### Phase 2C — Query Builder Benchmarks (`bench_query_builder.py`)

**Dependency**: None (pure Python, no graph execution)  
**Purpose**: Track the overhead of Cypher query generation and schema validation in `graph_search()`.

| Benchmark | Operation | What it measures |
|-----------|----------|-----------------|
| `time_basic_query_generation` | `graph_search("IMASNode", k=10)` | Minimal query gen |
| `time_filtered_query_generation` | `+ filters, traversals` | Complex query gen |
| `time_schema_validation` | Query with invalid label (expect error) | Validation overhead |
| `time_traversal_expansion` | `traverse=["HAS_UNIT>Unit", "IN_IDS>IDS"]` | Multi-hop expansion |

**Agent assignment**: 1 agent, completely independent.

---

## Phase 3 — Offline Subsystem & System Benchmarks

### Phase 3A — Offline Subsystems (`bench_subsystems.py`)

**Dependency**: None  
**Purpose**: Track performance of pure-Python subsystems that are part of the critical path.

| Benchmark | Subsystem | Operation | What it measures |
|-----------|-----------|----------|-----------------|
| `time_cocos_determine` | COCOS | `determine_cocos(Bt, Ip, q, psi_sign, theta_sign)` | COCOS computation |
| `time_cocos_to_params` | COCOS | `cocos_to_parameters(11)` | Lookup table |
| `time_cocos_transform_check` | COCOS | `path_needs_cocos_transform(path)` | Path classification |
| `time_unit_normalize_cached` | Units | `normalize_unit_symbol("m.s^-1")` (warm cache) | LRU hit |
| `time_unit_normalize_cold` | Units | `normalize_unit_symbol()` with cleared cache | LRU miss + pint parse |
| `time_unit_normalize_batch` | Units | 50 different unit strings | Throughput |
| `time_schema_for_overview` | Schema | `schema_for(task="overview")` | Schema slice (overview) |
| `time_schema_for_signals` | Schema | `schema_for(task="signals")` | Schema slice (task-scoped) |
| `time_schema_for_labels` | Schema | `schema_for("Facility", "DataSource")` | Schema slice (label-scoped) |

**Agent assignment**: 1 agent, completely independent.

### Phase 3B — Server Startup (`bench_server_startup.py`)

**Dependency**: Neo4j with loaded graph (for REPL init), embeddings (for encoder)  
**Purpose**: Track cold-start time — critical for container boot and MCP server initialization.

| Benchmark | Operation | What it measures |
|-----------|----------|-----------------|
| `time_server_cold_start` | `AgentsServer()` construction | Full server init |
| `time_mcp_tool_registration` | Tool registration via `_register_tools()` | MCP setup cost |
| `time_first_tool_call` | First `search_imas` after cold start | Lazy init triggers |

**Implementation note**: Each timing run must construct a fresh `AgentsServer()` instance to avoid cached state. Use `setup_cache` pattern or ensure module-level singletons are cleared.

**Agent assignment**: 1 agent.

### Phase 3C — Memory Profiling (`bench_memory.py`)

**Dependency**: Neo4j with loaded graph  
**Purpose**: Track peak memory for key operations to detect memory leaks or regressions.

| Benchmark | Operation | What it measures |
|-----------|----------|-----------------|
| `peakmem_server_idle` | `AgentsServer()` at rest | Base memory footprint |
| `peakmem_search_burst` | 20 sequential `search_imas` calls | Memory under load |
| `peakmem_export_large_ids` | `export_imas_ids("equilibrium")` | Export serialization |
| `peakmem_encoder_loaded` | `Encoder()` with model loaded | Model memory |

**Agent assignment**: Combined with Phase 3B agent.

---

## CI Workflow Implementation

### File: `.github/workflows/benchmark.yml` (rewrite)

```yaml
name: Benchmark

on:
  push:
    tags: ["v*"]
  workflow_dispatch:
    inputs:
      graph_tag:
        description: "Graph version tag to test"
        required: false
        default: "latest"
      benchmark_filter:
        description: "Benchmark filter (e.g., SearchToolBenchmarks)"
        required: false
        default: ""

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  run-benchmark:
    runs-on: ubuntu-22.04
    permissions:
      contents: write
      packages: read

    services:
      neo4j:
        image: neo4j:2026.01.4-community
        ports:
          - 7474:7474
          - 7687:7687
        env:
          NEO4J_AUTH: neo4j/imas-codex
          NEO4J_PLUGINS: '["apoc"]'
          NEO4J_server_memory_heap_initial__size: 512m
          NEO4J_server_memory_heap_max__size: 1G
        options: >-
          --health-cmd "wget -q --spider http://localhost:7474/ || exit 1"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 10
          --health-start-period 30s

    steps:
      # ── Setup ──
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Checkout gh-pages for benchmark history
        run: |
          if git ls-remote --exit-code origin gh-pages >/dev/null 2>&1; then
            git fetch origin gh-pages:gh-pages
            git worktree add gh-pages-data gh-pages
          else
            mkdir -p gh-pages-data
          fi
        continue-on-error: true

      - name: Restore previous benchmark results
        run: |
          if [ -d "gh-pages-data/.asv" ]; then
            mkdir -p .asv
            cp -r gh-pages-data/.asv/* .asv/ || true
          fi

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.12"

      - name: Free disk space
        run: |
          sudo rm -rf /usr/share/dotnet /opt/ghc /usr/local/share/boost
          df -h

      - name: Install UV + ASV
        uses: astral-sh/setup-uv@v5
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}

      - name: Install tools
        run: |
          uv tool install asv
          # Install oras for graph pull
          ORAS_VERSION="1.2.0"
          curl -sLO "https://github.com/oras-project/oras/releases/download/v${ORAS_VERSION}/oras_${ORAS_VERSION}_linux_amd64.tar.gz"
          tar -xzf "oras_${ORAS_VERSION}_linux_amd64.tar.gz" oras
          sudo mv oras /usr/local/bin/

      - name: Configure pip for CPU-only PyTorch
        run: |
          mkdir -p ~/.config/pip
          cat > ~/.config/pip/pip.conf << 'EOF'
          [global]
          extra-index-url = https://download.pytorch.org/whl/cpu
          EOF

      # ── Load Graph Data ──
      - name: Resolve graph tag
        id: graph-tag
        run: |
          TAG="${{ github.event.inputs.graph_tag || 'latest' }}"
          echo "tag=${TAG}" >> $GITHUB_OUTPUT

      - name: Login to GHCR
        run: echo "${{ secrets.GHCR_TOKEN }}" | oras login ghcr.io -u token --password-stdin

      - name: Pull graph dump from GHCR
        run: |
          REGISTRY="ghcr.io/iterorganization"
          PACKAGE="imas-codex-graph-imas"
          TAG="${{ steps.graph-tag.outputs.tag }}"
          echo "Pulling: ${REGISTRY}/${PACKAGE}:${TAG}"
          mkdir -p /tmp/graph-dump
          oras pull "${REGISTRY}/${PACKAGE}:${TAG}" -o /tmp/graph-dump
          ls -la /tmp/graph-dump/

      - name: Stop Neo4j for dump load
        run: |
          docker stop $(docker ps -q --filter "ancestor=neo4j:2026.01.4-community") || true
          sleep 3

      - name: Load graph dump
        run: |
          ARCHIVE=$(ls /tmp/graph-dump/*.tar.gz | head -1)
          mkdir -p /tmp/graph-extracted
          tar -xzf "${ARCHIVE}" -C /tmp/graph-extracted
          DUMP_FILE=$(find /tmp/graph-extracted -name "*.dump" -o -name "graph.dump" | head -1)
          NEO4J_CONTAINER=$(docker ps -aq --filter "ancestor=neo4j:2026.01.4-community" | head -1)
          NEO4J_DATA_VOLUME=$(docker inspect "${NEO4J_CONTAINER}" --format '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Name}}{{end}}{{end}}')
          docker run --rm \
            -v "${NEO4J_DATA_VOLUME}:/data" \
            -v "$(dirname ${DUMP_FILE}):/dump" \
            neo4j:2026.01.4-community \
            neo4j-admin database load neo4j --from-path=/dump --overwrite-destination=true

      - name: Restart Neo4j with loaded data
        run: |
          NEO4J_CONTAINER=$(docker ps -aq --filter "ancestor=neo4j:2026.01.4-community" | head -1)
          docker start "${NEO4J_CONTAINER}"
          for i in $(seq 1 60); do
            if curl -sf http://localhost:7474/ > /dev/null 2>&1; then
              echo "Neo4j ready with loaded graph"
              break
            fi
            sleep 2
          done

      - name: Reset Neo4j password
        run: |
          NEO4J_CONTAINER=$(docker ps -q --filter "ancestor=neo4j:2026.01.4-community" | head -1)
          docker exec "${NEO4J_CONTAINER}" neo4j-admin dbms set-initial-password imas-codex 2>/dev/null || true

      - name: Verify graph loaded
        run: |
          NEO4J_CONTAINER=$(docker ps -q --filter "ancestor=neo4j:2026.01.4-community" | head -1)
          docker exec "${NEO4J_CONTAINER}" cypher-shell \
            -u neo4j -p imas-codex \
            "MATCH (n) RETURN count(n) AS nodes, labels(n)[0] AS label ORDER BY nodes DESC LIMIT 10"

      # ── Run Benchmarks ──
      - name: Setup ASV machine
        run: |
          CPU_INFO=$(lscpu | grep "Model name" | cut -d: -f2 | sed 's/^[ \t]*//' | head -1)
          CPU_COUNT=$(nproc)
          RAM_GB=$(($(free -m | grep "Mem:" | awk '{print $2}') / 1024))
          ARCH=$(uname -m)
          OS_VERSION=$(lsb_release -ds 2>/dev/null | tr -d '"' || echo "Unknown")
          MACHINE_NAME="ghactions-${ARCH}-${CPU_COUNT}c-${RAM_GB}gb"
          asv machine --machine "$MACHINE_NAME" --os "$OS_VERSION" --arch "$ARCH" \
            --cpu "$CPU_INFO" --num_cpu "$CPU_COUNT" --ram "${RAM_GB}GB" --yes
          echo "MACHINE_NAME=$MACHINE_NAME" >> $GITHUB_ENV

      - name: Run benchmarks
        env:
          NEO4J_URI: bolt://localhost:7687
          NEO4J_USERNAME: neo4j
          NEO4J_PASSWORD: imas-codex
          IMAS_CODEX_EMBEDDING_LOCATION: local
        run: |
          FILTER="${{ github.event.inputs.benchmark_filter }}"
          if [ -n "$FILTER" ]; then
            asv run --python=3.12 --machine "$MACHINE_NAME" -b "$FILTER" --verbose --show-stderr
          else
            asv run --python=3.12 --machine "$MACHINE_NAME" HEAD^! --verbose --show-stderr
          fi

      # ── Regression Detection ──
      - name: Check for regressions
        if: always()
        run: |
          # Compare current results against the previous tag
          PREV_TAG=$(git describe --tags --abbrev=0 HEAD^ 2>/dev/null || echo "")
          if [ -n "$PREV_TAG" ]; then
            echo "Comparing against previous release: $PREV_TAG"
            asv compare "$PREV_TAG" HEAD --factor 1.5 --split || true
          else
            echo "No previous tag found — skipping comparison"
          fi

      # ── Publish ──
      - name: Generate HTML report
        run: asv publish

      - name: Deploy to GitHub Pages
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
          mkdir -p gh-pages-deploy
          cp -r .asv/html/* gh-pages-deploy/
          mkdir -p gh-pages-deploy/.asv/results
          cp -r .asv/results/* gh-pages-deploy/.asv/results/ || true
          cp .asv/machine.json gh-pages-deploy/.asv/ || true
          touch gh-pages-deploy/.nojekyll
          cd gh-pages-deploy
          git init && git checkout -b gh-pages
          git add -f .asv/ && git add -A
          git commit -m "benchmark: ${{ github.ref_name }}"
          git remote add origin https://x-access-token:${{ secrets.GITHUB_TOKEN }}@github.com/${{ github.repository }}.git
          git push -f origin gh-pages
```

---

## Benchmark Queries — Reference Corpus

All benchmark modules should use a shared set of realistic queries. This avoids
each module inventing ad-hoc test data and ensures cross-module consistency.

```python
# benchmarks/conftest_bench.py — shared constants

SEARCH_QUERIES = {
    "simple": "electron temperature",
    "multi_term": "magnetic field equilibrium boundary",
    "physics_specific": "poloidal flux gradient",
    "cross_ids": "temperature AND pressure profiles",
    "complex": "magnetic field topology near X-point separatrix",
}

IMAS_PATHS = {
    "leaf": "core_profiles/profiles_1d/electrons/temperature",
    "branch": "equilibrium/time_slice/profiles_1d",
    "ids": "equilibrium",
    "short": "core_profiles",
}

IDS_NAMES = {
    "small": "core_profiles",
    "large": "equilibrium",
    "domain": "magnetics",
}

UNIT_STRINGS = [
    "eV", "m", "m^-3", "T", "Pa", "s", "A", "V", "W",
    "m.s^-1", "T.m^2", "keV", "m^-2.s^-1", "ohm.m",
    # Sentinel / edge cases
    "-", "mixed", "dimensionless",
]
```

---

## Implementation Phases & Agent Assignment

### Phase 1: MCP Tool Benchmarks (3 parallel agents)

| Work Item | Agent | File | Dependencies | Estimated Effort |
|-----------|-------|------|-------------|-----------------|
| 1A: Search tool benchmarks | Agent A | `bench_mcp_search.py` | conftest_bench.py | Medium |
| 1B: IMAS DD tool benchmarks | Agent B | `bench_mcp_imas_tools.py` | conftest_bench.py | Medium |
| 1C: Facility tool benchmarks | Agent C | `bench_mcp_facility_tools.py` | conftest_bench.py | Medium |
| Shared fixture + constants | Any agent (first) | `conftest_bench.py` | None | Small |

**Pre-requisite**: One agent (any) writes `conftest_bench.py` first with the shared fixture and constants. Then all three modules can be developed in parallel.

**Acceptance criteria**:
- Each benchmark class has `setup()` that does a warmup call
- All tool calls go through FastMCP `Client` (not direct function calls)
- Facility benchmarks gracefully skip if no facility data in graph
- At least `time_` and `peakmem_` variants for high-traffic tools

### Phase 2: Subsystem Benchmarks (3 parallel agents)

| Work Item | Agent | File | Dependencies | Estimated Effort |
|-----------|-------|------|-------------|-----------------|
| 2A: Raw graph queries | Agent D | `bench_graph_queries.py` | Neo4j loaded | Medium |
| 2B: Embedding performance | Agent E | `bench_embeddings.py` | None (standalone) | Small |
| 2C: Query builder | Agent F | `bench_query_builder.py` | None (standalone) | Small |

**Parallelization**: All three are completely independent. Phase 2B and 2C need no Neo4j at all.

**Acceptance criteria**:
- Graph query benchmarks use `GraphClient` directly, not MCP
- Embedding benchmarks force `IMAS_CODEX_EMBEDDING_LOCATION=local`
- Query builder benchmarks test Cypher generation only, never execute against Neo4j

### Phase 3: System Benchmarks + CI Workflow (2 parallel agents)

| Work Item | Agent | File | Dependencies | Estimated Effort |
|-----------|-------|------|-------------|-----------------|
| 3A: Offline subsystems + startup + memory | Agent G | `bench_subsystems.py`, `bench_server_startup.py`, `bench_memory.py` | Neo4j for startup/memory | Medium |
| 3B: CI workflow rewrite | Agent H | `.github/workflows/benchmark.yml` | All benchmark files | Medium |

**Parallelization**: Agent G writes benchmark modules. Agent H rewrites the workflow YAML. Both can work simultaneously — the workflow just needs to know the file naming convention.

### Phase 4: Cleanup & Validation (1 agent)

| Work Item | Agent | File | Dependencies | Estimated Effort |
|-----------|-------|------|-------------|-----------------|
| Remove old benchmarks.py | Agent I | `benchmarks/benchmarks.py` | Phases 1–3 complete | Small |
| Update asv.conf.json | Agent I | `asv.conf.json` | — | Small |
| Update README/docs | Agent I | `benchmarks/README.md` | — | Small |
| End-to-end validation | Agent I | Run full benchmark locally | All phases | Medium |

---

## Dependency Graph (What Can Run In Parallel)

```
                    ┌──────────────────────┐
                    │ conftest_bench.py     │
                    │ (shared fixture)      │
                    └──────┬───────────────┘
                           │
              ┌────────────┼────────────────┐
              ▼            ▼                ▼
     ┌────────────┐ ┌──────────────┐ ┌──────────────┐
     │ Phase 1A   │ │ Phase 1B     │ │ Phase 1C     │
     │ Search     │ │ IMAS tools   │ │ Facility     │
     │ Agent A    │ │ Agent B      │ │ Agent C      │
     └────────────┘ └──────────────┘ └──────────────┘

     ┌────────────┐ ┌──────────────┐ ┌──────────────┐
     │ Phase 2A   │ │ Phase 2B     │ │ Phase 2C     │
     │ Graph qry  │ │ Embeddings   │ │ Query builder│
     │ Agent D    │ │ Agent E      │ │ Agent F      │
     └────────────┘ └──────────────┘ └──────────────┘
              │            │                │
              │  ┌─────────┼────────────────┤
              │  │         │                │
              ▼  ▼         ▼                ▼
     ┌────────────┐                 ┌──────────────┐
     │ Phase 3A   │                 │ Phase 3B     │
     │ Subsystems │                 │ CI workflow  │
     │ Agent G    │                 │ Agent H      │
     └────────────┘                 └──────────────┘
              │                            │
              └──────────┬─────────────────┘
                         ▼
                  ┌──────────────┐
                  │ Phase 4      │
                  │ Cleanup      │
                  │ Agent I      │
                  └──────────────┘
```

**Maximum parallelism**: 6 agents (Phases 1A–1C + 2A–2C can all run simultaneously after `conftest_bench.py` is written).

---

## ASV Configuration Changes (`asv.conf.json`)

The existing config mostly works but needs the `benchmark_dir` to pick up the new
module files. No schema changes needed — ASV auto-discovers benchmark classes from
Python files in the `benchmark_dir`.

Ensure `build_command` remains the same (triggers hatch hooks for `schema_context_data.py`).

---

## What We Explicitly Do NOT Benchmark

| Category | Reason |
|----------|--------|
| LLM API calls | External service, variable latency, costs money |
| Discovery pipelines (`discover paths/code/wiki/signals`) | Multi-minute workflows, involve external SSH/HTTP, LLM scoring |
| Write tools (`add_to_graph`, `update_facility_config`, etc.) | Side effects, would pollute test graph |
| CLI commands (`graph start/stop`, `embed start`, etc.) | System administration, not performance-critical computation |
| `python` REPL tool | Arbitrary code execution, not meaningfully benchmarkable |
| Graph build (`imas dd build`) | Multi-minute build process, tested by graph-quality workflow |
| Remote embedding server | External HTTP service, tested by `benchmark_embedding.py` script |
| SSH tunnels, HPC/SLURM operations | Infrastructure management, not computation |

---

## Regression Detection Strategy

ASV has built-in regression detection via `asv compare`. The workflow runs:

```bash
asv compare "$PREV_TAG" HEAD --factor 1.5 --split
```

This flags any benchmark that regressed by >50%. Currently informational only (no CI failure).

### Future Enhancement (Not In Scope)

A subsequent iteration could add a hard gate:

```bash
asv compare "$PREV_TAG" HEAD --factor 2.0 --only-changed --machine "$MACHINE_NAME"
# Exit non-zero if any benchmark regressed by >2x
```

This requires sufficient benchmark history to establish stable baselines first.

---

## Success Criteria

1. **Benchmark workflow passes on tagged releases** with real graph data
2. **All 22 read-only MCP tools** have at least one timing benchmark
3. **Key subsystems** (embeddings, query builder, COCOS, units, schema_for) are covered
4. **gh-pages** shows performance trends across releases
5. **No empty-graph benchmarks** — every Neo4j-dependent benchmark uses GHCR data
6. **Regression comparison** runs against previous tag on each release
