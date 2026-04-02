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
| **P3** | [ids-properties-population.md](features/ids-properties-population.md) | IDS properties and code metadata population | Ready — staged-mapping prerequisite complete |
| **P4** | [gaps-compute-orchestration.md](features/gaps-compute-orchestration.md) | Compute session orchestration remaining gaps | Low priority — Python CLI covers basics |

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
