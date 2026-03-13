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

### Infrastructure

| Plan | Scope | Status |
|------|-------|--------|
| [signal-scanner-diagnostics.md](features/signal-scanner-diagnostics.md) | Scanner progress streaming, worker health indicators, MCP log tools | ~75% — infrastructure built, needs wiring to production workers |

### JET Machine Description

| Plan | Scope | Status |
|------|-------|--------|
| [jet-machine-description-completion.md](features/jet-machine-description-completion.md) | Remaining enrichments: historical sensor versions, MCFG calibration epochs, PF circuit JPF addresses | Low priority — core pipeline operational |

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
