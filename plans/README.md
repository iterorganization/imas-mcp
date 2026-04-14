# Plans

> Strategic vision and active feature plans for imas-codex.
>
> **Rule:** Delete plans when implemented. The code is the documentation.
> Completed features belong in `completed/` for reference, then eventually `docs/architecture/`.

## Vision & Strategy

| Plan | Scope |
|------|-------|
| [STRATEGY.md](STRATEGY.md) | The Federated Fusion Knowledge Graph — four-zone architecture, discovery engine, agile LinkML schema workflow |
| [gpu-cluster-scoping.md](gpu-cluster-scoping.md) | Executive proposal for ITER GPU infrastructure (4x H200) |
| [gpu-cluster-internet-justification.md](gpu-cluster-internet-justification.md) | Network topology analysis and DNS workarounds |

## Active Feature Plans

Gap documents consolidate remaining work from completed implementation phases.

| Priority | Plan | Scope | Status |
|----------|------|-------|--------|
| **P2** | [features/imas-dd-server-improvements.md](features/imas-dd-server-improvements.md) | DD server gaps: explain_concept, total_paths, Phase 4 migration redesign | ~80% done |
| **P2** | [features/gap-sn-quality-and-review.md](features/gap-sn-quality-and-review.md) | SN quality parity + standalone review CLI | New gap plan |
| **P3** | [features/search-cluster-output-and-evaluation.md](features/search-cluster-output-and-evaluation.md) | Search cluster labels, See Also siblings, DoE optimization | ~40% done |
| **P4** | [features/gaps-compute-orchestration.md](features/gaps-compute-orchestration.md) | Compute session orchestration remaining gaps | Low priority |

### Standard Names (remaining work)

| Plan | Scope | Status |
|------|-------|--------|
| [standard-names/gap-sn-enrichment-gaps.md](features/standard-names/gap-sn-enrichment-gaps.md) | DD enrichment remaining: regenerate CLI, benchmark profiles, concept registry | New gap plan |

### Completed plans

Fully implemented plans archived in `features/completed/` and `features/standard-names/completed/` for reference.

### Pending plans (partially implemented)

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

## Research

Historical analysis documents in `research/` — findings incorporated into implementation plans.

## Related Projects

- **[imas-ambix](https://github.com/iterorganization/imas-ambix)** — Universal Fusion Data Client
- **[tree-sitter-gdl](https://github.com/iterorganization/tree-sitter-gdl)** — Tree-sitter grammar for GDL/IDL
