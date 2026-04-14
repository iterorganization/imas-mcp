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

Refactored after rubber-duck review with independent code investigation (2026-04-14).
Plans validated against live MCP tool output and current codebase.

| Priority | Plan | Scope | Est. Agents |
|----------|------|-------|-------------|
| **P1** | [features/sn-bootstrap-loop.md](features/sn-bootstrap-loop.md) | Wire `sn review` CLI, run generate→review→regenerate loop | 1 agent |
| **P2** | [features/dd-server-cleanup.md](features/dd-server-cleanup.md) | 3 surgical fixes: truncation count, migration API, fuzzy matcher | 1-3 agents |
| **P3** | [features/search-quality-improvements.md](features/search-quality-improvements.md) | Ranking fixes, evaluation alignment, fuzzy search | 2-3 agents |

### Wave Implementation Order

```
Wave 1: sn-bootstrap-loop (Phase 1 only — review CLI)
Wave 2: dd-server-cleanup (all 3 fixes, parallel)  ← can run parallel with Wave 1
Wave 3: search-quality-improvements (Phases 1-2)
Wave 4: sn-bootstrap-loop Phases 2-3 (operational bootstrap + prompt improvements)
```

### Explicitly Removed / Deferred

| Plan | Reason |
|------|--------|
| Compute orchestration | Removed — compute nodes cannot access internet |
| `explain_concept` tool | Removed — frontier LLMs already do this, zero value |
| SN benchmarking | Deferred — insufficient quality content in graph |
| SN enrichment gaps | Deferred — low priority refinements, workarounds exist |
| Search dimension upgrade | Deferred — fix ranking first, then investigate |
| DoE weight optimization | Deferred — harness doesn't match production scoring |

### Completed plans

Fully implemented plans archived in `features/completed/` and `features/standard-names/completed/`.

### Reference plans (pending/)

Partially implemented plans kept as reference for gap documents.

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
