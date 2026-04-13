# Plans

> Strategic vision and active feature plans for imas-codex.
>
> **Rule:** Delete plans when implemented. The code is the documentation.
> Completed features belong in `docs/architecture/`, not here.

## Vision & Strategy

| Plan | Scope |
|------|-------|
| [STRATEGY.md](STRATEGY.md) | The Federated Fusion Knowledge Graph — four-zone architecture, discovery engine, agile LinkML schema workflow |
| [gpu-cluster-scoping.md](gpu-cluster-scoping.md) | Executive proposal for ITER GPU infrastructure (4x H200) |

## Active Feature Plans

Gap documents consolidate remaining work from completed implementation phases. These are the canonical work items.

| Priority | Plan | Scope | Status |
|----------|------|-------|--------|
| **P2** | [features/search-recall-and-enrichment.md](features/search-recall-and-enrichment.md) | Search recall, scoring, output enrichment | Implemented |
| **P2** | [features/dd-server-capability-gaps.md](features/dd-server-capability-gaps.md) | DD server capability gaps from A/B testing — COCOS fields, lifecycle filter, migration summary | Active |
| **P2** | [features/dd-opportunity-gap-tools.md](features/dd-opportunity-gap-tools.md) | New DD analytics tools: changelog, coverage, unit consistency, change impact | Active |
| **P3** | [features/search-cluster-output-and-evaluation.md](features/search-cluster-output-and-evaluation.md) | Search cluster output, auto-eval, dim comparison | Implemented |
| **P3** | ~~embedding-upgrade-and-search-migration~~ | Fix embed bug, dim eval, SEARCH clause, quantization | **Done** — dim stays 256, SEARCH migrated, quantization enabled |
| **P4** | [gaps-compute-orchestration.md](features/gaps-compute-orchestration.md) | Compute session orchestration remaining gaps | Low priority — Python CLI covers basics |

### Standard Names

| Plan | Scope | Status | Depends On |
|------|-------|--------|------------|
| [standard-names/09-sn-generate.md](features/standard-names/09-sn-generate.md) | Core pipeline: EXTRACT→COMPOSE→VALIDATE→PERSIST | ✅ Done | — |
| [standard-names/11-rich-compose.md](features/standard-names/11-rich-compose.md) | Rich compose: full catalog fields, schema extension | ✅ Done | 09 |
| [standard-names/12-catalog-import.md](features/standard-names/12-catalog-import.md) | Catalog import & bootstrap (309 entries, feedback loop) | ✅ Done | 11 P1 |
| [standard-names/13-publish-pipeline.md](features/standard-names/13-publish-pipeline.md) | Lossless publish, round-trip, batched PRs | ✅ Done | 11, 12 P1 |
| [standard-names/14-mcp-tools-benchmark.md](features/standard-names/14-mcp-tools-benchmark.md) | SN MCP tools + benchmark quality tiers | ✅ Done | 11, 12 |
| [standard-names/21-architecture-boundary.md](features/standard-names/21-architecture-boundary.md) | Architecture boundary, ISN validation, 0-1 scoring, prompt infrastructure | ✅ Done | 14 |

### Pending plans (partially implemented)

These plans are reference material for the gap documents above — not direct work items. Gaps were extracted into the active gap docs.

| Plan | Consolidated into |
|------|-------------------|
| [compute-session-orchestration.md](features/pending/compute-session-orchestration.md) | gaps-compute-orchestration |

## Documentation Gaps

These implemented systems need architecture docs in `docs/`:

1. **Discovery pipeline** — 6-domain supervised worker system with shared engine skeleton
2. **Signal enrichment** — multi-phase LLM enrichment with deterministic context injection
3. **IMAS mapping pipeline** — signal mapping, assembly discovery, programmatic validation
4. **MCP search tools** — multi-index semantic search + graph enrichment pattern
5. **Schema context system** — LinkML-driven `schema_for()` with task groups
6. **Graph profiles & merge** — multi-graph management, GHCR push/pull

## Related Projects

- **[imas-ambix](https://github.com/iterorganization/imas-ambix)** — Universal Fusion Data Client
- **[tree-sitter-gdl](https://github.com/iterorganization/tree-sitter-gdl)** — Tree-sitter grammar for GDL/IDL
