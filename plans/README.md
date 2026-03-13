# Plans

> Strategic vision and active feature plans for imas-codex.
>
> **Rule:** Delete plans when implemented. The code is the documentation.
> Completed features belong in `docs/architecture/`, not here.

## Vision & Strategy

### [STRATEGY.md](STRATEGY.md)
The Federated Fusion Knowledge Graph — four-zone architecture, discovery engine,
agile LinkML schema workflow, MCP server design. Start here.

### [gpu-cluster-scoping.md](gpu-cluster-scoping.md)
Executive proposal for ITER GPU infrastructure. Three use cases: multilingual
embedding, local agentic LLM deployment, Fusion World Model. Business case
for 4x H200 server.

### [graph-agent-interaction.md](graph-agent-interaction.md)
Architecture for efficient agent-graph interaction. LinkML-driven schema context
generation, tiered `schema_for()` API, codegen-first REPL philosophy. Partially
implemented (`schema_context_data.py`, `schema_context.py` exist).

## Active Feature Plans

### IMAS Mapping Pipeline (core deliverable)

| Plan | Scope |
|------|-------|
| [imas-mapping-combined.md](features/imas-mapping-combined.md) | **Master plan** — schema redesign + LLM pipeline. Six phases |
| [imas-map-context-v2.md](features/imas-map-context-v2.md) | Context enrichment for the mapping pipeline |
| [mapping-quality.md](features/mapping-quality.md) | Output quality: naming, validation, coverage reporting |
| [signal-enrichment-v3.md](features/signal-enrichment-v3.md) | Signal enrichment redesign. Blocks mapping quality |

### MCP & Search Quality

| Plan | Scope |
|------|-------|
| [unified-mcp-tools.md](features/unified-mcp-tools.md) | Consolidated design for `search_signals`/`search_docs`/`search_code`/`search_imas` |
| [mcp-cypher-gap-rectification.md](features/mcp-cypher-gap-rectification.md) | 8 gaps between MCP tools and raw Cypher quality |
| [ingestion-schema-alignment.md](features/ingestion-schema-alignment.md) | Schema-graph drift: missing relationships, label mismatches |
| [schema-id-normalization.md](features/schema-id-normalization.md) | Normalize all identifier fields to `id`. Fixes 24 silent relationship failures |

### Infrastructure

| Plan | Scope |
|------|-------|
| [discovery-unification.md](features/discovery-unification.md) | Deduplicate shared infrastructure across 5 discovery domains |
| [signal-scanner-diagnostics.md](features/signal-scanner-diagnostics.md) | Scanner progress streaming, worker health indicators |

### JET Machine Description

| Plan | Scope |
|------|-------|
| [jet-machine-description-ingestion.md](features/jet-machine-description-ingestion.md) | Full JET geometry ingestion: device XML, limiter, JEC2020, MCFG, PPF |
| [jet-legacy-machine-description.md](features/jet-legacy-machine-description.md) | Pre-EFIT++ era (shots 1–68612): magnetics config files, parsed formats |

## Documentation Gaps

These implemented systems need architecture docs in `docs/`:

1. **Discovery pipeline** — 5-domain supervised worker system (paths, wiki, signals, code, documents)
2. **Signal enrichment** — multi-phase LLM enrichment with context injection
3. **MCP search tools** — multi-index semantic search + graph traversal pattern
4. **Graph profiles & merge** — multi-graph management, GHCR push/pull

## Related Projects

- **[imas-ambix](https://github.com/iterorganization/imas-ambix)** — Universal Fusion Data Client
- **[tree-sitter-gdl](https://github.com/iterorganization/tree-sitter-gdl)** — Tree-sitter grammar for GDL/IDL
