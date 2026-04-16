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

Refactored after rubber-duck review, docs audit, and live MCP tool testing (2026-04-14).
Plans validated against 62 existing standard names, current scoring architecture, and docs/ state.

| Priority | Plan | Scope | Est. Agents |
|----------|------|-------|-------------|
| ~~**P1**~~ | [features/sn-bootstrap-loop.md](features/sn-bootstrap-loop.md) | ✅ Phase 1 (review CLI) complete, ✅ Phase 2 (bootstrap loop) complete, ⏳ Phase 3 blocked (needs ≥500 reviewed names) | 1 agent |
| **P1.5** | [features/dd-unified-classification.md](features/dd-unified-classification.md) | 8-value NodeCategory, quantity→physical_quantity+geometry split, classifier bug fixes, sonnet enrichment model, MCP tool augmentation. **Prerequisite for P1.6.** | 2-3 agents |
| **P1.6** | [features/standard-names/28-sn-greenfield-pipeline.md](features/standard-names/28-sn-greenfield-pipeline.md) | Greenfield SN pipeline: NAME→ENRICH split, PRELINK/POSTLINK context retrieval, vector hierarchy (GROUP), documentation inheritance. Clears all existing SNs. | 2-3 agents |
| **P2** | [features/dd-server-cleanup.md](features/dd-server-cleanup.md) | 3 surgical fixes: truncation count, migration API, fuzzy matcher | 1-3 agents |
| **P3** | [features/search-quality-improvements.md](features/search-quality-improvements.md) | Careful ranking fixes (accessor de-ranking, IDS preference), evaluation alignment | 2 agents |
| **P4** | [features/docs-refresh.md](features/docs-refresh.md) | Fix 7 stale docs, rewrite docs/README.md (17+ missing entries) | 1 agent |
| **P5** | ~~features/sn-extraction-coverage-gaps.md~~ | ✅ **Completed** — StandardNameSource graph-primary architecture, naming standardization, extraction coverage gap fixes | — |

### Wave Implementation Order

```
Wave 1: sn-bootstrap-loop Phase 1 (review CLI)  ✅ complete
Wave 2: dd-unified-classification (schema + classifier + enrichment model)
Wave 3: sn-greenfield-pipeline (clear all SNs, run new multi-pass pipeline)
Wave 4: dd-server-cleanup (all 3 fixes, parallel)
Wave 5: search-quality-improvements (Phases 1-2)
Wave 6: docs-refresh (fix stale docs after code changes land)
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
| Leaf concept boost | Deferred — accessor de-ranking may be sufficient, needs investigation |

### Completed plans

Fully implemented plans archived in `features/completed/` and `features/standard-names/completed/`.

### Reference plans (pending/)

Partially implemented plans kept as reference for gap documents.

## Documentation Gaps

Docs review (2026-04-14) found these gaps — addressed in P4 docs-refresh plan:

1. **docs/README.md** — only lists 6 of 17+ architecture docs
2. **graph.md** — documents non-existent `backup`/`restore` CLI commands
3. **services.md** — wrong LLM proxy port (18790 vs 18400)
4. **standard-names.md** — references non-existent `boundary.md`
5. **llamaindex-agents.md** — references removed `create_enrichment_agent()`

## Research

Historical analysis documents in `research/` — findings incorporated into implementation plans.

## Related Projects

- **[imas-ambix](https://github.com/iterorganization/imas-ambix)** — Universal Fusion Data Client
- **[tree-sitter-gdl](https://github.com/iterorganization/tree-sitter-gdl)** — Tree-sitter grammar for GDL/IDL
