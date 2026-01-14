# Plans Index

> Strategic and tactical planning documents for imas-codex development.

## Document Classification

| Type | Purpose | Location |
|------|---------|----------|
| **Strategy** | High-level vision, architecture, multi-phase roadmaps | `STRATEGY.md` |
| **Feature** | Tactical implementation plans for specific capabilities | `features/*.md` |

## Strategic Plan

### [STRATEGY.md](STRATEGY.md)
**The Master Document** - Overall vision, four-zone architecture, and 7-phase roadmap for the Federated Fusion Knowledge Graph. Start here to understand the project.

## Feature Plans

Tactical implementation plans for specific capabilities:

### [features/cocos-enrichment.md](features/cocos-enrichment.md)
COCOS graph enrichment and imas-python integration. Extends the implemented `imas_codex.cocos` module.

### [features/discovery-agents.md](features/discovery-agents.md)
Discovery & ingestion pipeline with Map → Score → Ingest phases. Includes multi-facility onboarding workflow and autonomous ReAct agent architecture.

### [features/wiki-ingestion.md](features/wiki-ingestion.md)
Wiki content ingestion pipeline with ReAct agent evaluation, semantic chunking, and facility-agnostic design.

### [features/mcp-tool-fixes.md](features/mcp-tool-fixes.md)
IMAS MCP tool improvements in three phases: input handling, error recovery, and description updates.

### [features/path-normalization.md](features/path-normalization.md)
TreeNode path normalization and TDI function integration for deduplication and matching.

### [features/enrichment.md](features/enrichment.md)
TreeNode LLM enrichment for metadata generation and graph relationship discovery.

## Related Projects

- **[imas-ambix](https://github.com/iterorganization/imas-ambix)** - Universal Fusion Data Client (runtime Recipe execution)
- **[tree-sitter-gdl](https://github.com/iterorganization/tree-sitter-gdl)** - Tree-sitter grammar for GDL/IDL parsing

## Related Documentation

- **[docs/architecture/](../docs/architecture/)** - How things work (implemented systems)

## Document Lifecycle

1. **Draft** → Initial ideas and proposals
2. **Active** → Under implementation, updated as work progresses  
3. **Complete** → Feature shipped, document archived to `docs/`
4. **Superseded** → Replaced by newer plan

When a plan is fully implemented, move relevant content to `docs/architecture/` or `docs/workflows/` and archive the plan.
