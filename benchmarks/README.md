# IMAS Codex Performance Benchmarks

Performance benchmarking for IMAS Codex using [ASV (airspeed velocity)](https://asv.readthedocs.io/). Benchmarks run against a real GHCR graph dump on tagged releases.

## Benchmark Modules

| Module | Class | Benchmarks | Dependencies |
|--------|-------|-----------|-------------|
| `bench_mcp_search.py` | `SearchToolBenchmarks` | 12 | Neo4j + graph dump |
| `bench_mcp_imas_tools.py` | `IMASToolBenchmarks` | 17 | Neo4j + graph dump |
| `bench_mcp_facility_tools.py` | `FacilityToolBenchmarks` | 7 | Neo4j + facility data |
| `bench_mcp_facility_tools.py` | `GraphSchemaToolBenchmarks` | 3 | Neo4j + graph dump |
| `bench_graph_queries.py` | `GraphQueryBenchmarks` | 8 | Neo4j + graph dump |
| `bench_embeddings.py` | `EmbeddingBenchmarks` | 6 | None (CPU-only) |
| `bench_query_builder.py` | `QueryBuilderBenchmarks` | 4 | None (pure Python) |
| `bench_subsystems.py` | `SubsystemBenchmarks` | 9 | None (pure Python) |
| `bench_server_startup.py` | `ServerStartupBenchmarks` | 3 | Neo4j for first call |
| `bench_memory.py` | `MemoryBenchmarks` | 4 | Neo4j + graph dump |

**Total: 72 benchmarks** across 10 classes.

## Shared Infrastructure

- `conftest_bench.py` — `MCPFixture` (lazy server/client/graph_client), shared constants (`SEARCH_QUERIES`, `IMAS_PATHS`, `IDS_NAMES`, `UNIT_STRINGS`), `run_tool()` async helper
- `benchmark_runner.py` — `BenchmarkRunner` utility class for programmatic ASV usage

## Setup

```bash
uv sync --extra bench
asv machine --yes
```

## Usage

```bash
# Run all benchmarks
asv run --python=3.12

# Run specific suite
asv run --python=3.12 -b SearchToolBenchmarks

# Run only offline benchmarks (no Neo4j needed)
asv run --python=3.12 -b "EmbeddingBenchmarks|QueryBuilderBenchmarks|SubsystemBenchmarks"

# Compare against previous commit
asv compare HEAD~1 HEAD

# Generate HTML report
asv publish
```

## CI Pipeline

Benchmarks run automatically on tagged releases (`v*`) via `.github/workflows/benchmark.yml`:

1. Pull graph dump from GHCR (`imas-codex-graph:latest`)
2. Load into Neo4j via `neo4j-admin database load`
3. Run all ASV benchmarks with CPU-only embeddings
4. Compare against previous tag for regression detection
5. Deploy results to GitHub Pages

Manual trigger via `workflow_dispatch` supports `graph_tag` and `benchmark_filter` inputs.

## What We Benchmark

- **MCP tools**: All 22 read-only tools through the FastMCP client
- **Graph queries**: Raw Cypher patterns (vector, fulltext, traversal, aggregation)
- **Embeddings**: Encode latency (single, batch, long text, cold start)
- **Query builder**: Cypher generation overhead (no Neo4j execution)
- **Subsystems**: COCOS computation, unit normalization, schema context
- **Server startup**: Cold start time, tool registration, first call latency
- **Memory**: Peak memory for server, search bursts, exports, encoder

## What We Do NOT Benchmark

- LLM API calls (external, variable, costs money)
- Discovery pipelines (multi-minute, involve SSH/HTTP/LLM)
- Write tools (side effects, pollutes test graph)
- CLI commands (system admin, not performance-critical)
- Remote embedding server (external HTTP service)
