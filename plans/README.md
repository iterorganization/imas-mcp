# Plans Index

> Strategic and tactical planning documents for imas-codex development.

## Document Classification

| Type | Purpose | Naming |
|------|---------|--------|
| **Strategy** | High-level vision, architecture, multi-phase roadmaps | `*_STRATEGY.md` |
| **Feature** | Tactical implementation plans for specific capabilities | `features/*.md` |

## Strategic Plans

### [CODEX_PLAN.md](CODEX_PLAN.md)
**The Master Document** - Overall vision, four-zone architecture, and 7-phase roadmap for the Federated Fusion Knowledge Graph. Start here to understand the project.

### [DISCOVERY_STRATEGY.md](DISCOVERY_STRATEGY.md)
**Discovery & Ingestion** - Three-phase approach (Map → Score → Ingest) for code discovery across fusion facilities. Includes multi-facility onboarding workflow and autonomous agent architecture.

## Feature Plans

Tactical implementation plans for specific capabilities:

### [features/wiki-ingestion.md](features/wiki-ingestion.md)
Wiki content ingestion pipeline with ReAct agent evaluation, semantic chunking, and facility-agnostic design.

### [features/enrichment.md](features/enrichment.md)
TreeNode LLM enrichment using Gemini Flash for metadata generation and graph relationship discovery.

### [features/mcp-tool-fixes.md](features/mcp-tool-fixes.md)
IMAS MCP tool improvements in three phases: input handling, error recovery, and description updates.

### [features/path-consistency.md](features/path-consistency.md)
TreeNode path normalization and TDI function integration for deduplication and matching.

## Related Projects

- **[imas-ambix](https://github.com/iterorganization/imas-ambix)** - Universal Fusion Data Client (runtime Recipe execution)
- **[tree-sitter-gdl](https://github.com/iterorganization/tree-sitter-gdl)** - Tree-sitter grammar for GDL/IDL parsing

## Document Lifecycle

1. **Draft** → Initial ideas and proposals
2. **Active** → Under implementation, updated as work progresses  
3. **Complete** → Feature shipped, document archived to `docs/`
4. **Superseded** → Replaced by newer plan

When a plan is fully implemented, move relevant content to `docs/architecture/` or `docs/workflows/` and archive the plan.
