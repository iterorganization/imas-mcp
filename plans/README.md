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

| Priority | Plan | Scope | Status |
|----------|------|-------|--------|
| **P1** | [features/standard-names/29-architectural-pivot.md](features/standard-names/29-architectural-pivot.md) | Authoritative SN roadmap: generate/enrich split, ISN grammar upgrade, schema evolution. Supersedes plan 28. | Living — most structural items shipped |
| **P1a** | [features/standard-names/31-quality-bootstrap-v2.md](features/standard-names/31-quality-bootstrap-v2.md) | 7-workstream bootstrap loop driving rc14/rc15; quarantine-rate + description-coverage + reviewer-score targets. | Active rotation |
| **P1b** | [features/standard-names/32-extraction-prompt-overhaul.md](features/standard-names/32-extraction-prompt-overhaul.md) | Extraction filter audit + compose-prompt A/B bake-off. | Phase 2 done (see `research/standard-names/prompt-ab-results.md`); Phase 4 deferred |
| **P1c** | [research/standard-names/33-state-of-the-nation-2026-04-20.md](research/standard-names/33-state-of-the-nation-2026-04-20.md) | **Next-cycle direction:** grammar persistence, cross-family review, benchmark-set v1, ISN rc16 packaging. | New |
| **P1d** | [features/standard-names/34-benchmark-v1.md](features/standard-names/34-benchmark-v1.md) | SN quality benchmark v1: 50 positives × 10 domains + ≥20 anti-pattern negatives; two-reviewer consensus; gating at pass@1 ≥ 0.80, mean score ≥ 0.75, reject rate ≥ 0.90. | Scaffolded (20 + 10 filled; runner + mock dry-run) |
| ~~P1e~~ | ~~features/standard-names/39-graph-relationship-completeness.md~~ | ✅ Completed — structural edges (HAS_ARGUMENT, HAS_ERROR, HAS_PREDECESSOR, HAS_SUCCESSOR, IN_CLUSTER, HAS_PHYSICS_DOMAIN) wired into both write_standard_names + catalog import paths via shared `_write_standard_name_edges` helper; D1–D16 + G1–G10 tests | — |
| **P1f** | [features/standard-names/40-catalog-layout-hierarchy.md](features/standard-names/40-catalog-layout-hierarchy.md) | ISNC one-file-per-domain migration; graph-traversal ordering over shipped `HAS_ARGUMENT` / `HAS_ERROR` edges; `COMPUTED_FIELDS` = `{arguments, error_variants}`; `CANONICAL_KEY_ORDER` byte-stable round-trip; `export_scope` manifest + `edge_model_version: plan_39_v1` pin + publish-safety `FileLock`. Rewritten for shipped plan-39 edge model. | Ready — plan 39 shipped |
| **P1g** | [features/standard-names/41-catalog-graph-consumer.md](features/standard-names/41-catalog-graph-consumer.md) | ISN-side NetworkX `MultiDiGraph` emitting `HAS_ARGUMENT` / `HAS_ERROR` / `HAS_PREDECESSOR` / `HAS_SUCCESSOR` / `REFERENCES` with full edge-property dicts; MCP tools (get_neighbours, get_ancestors, get_descendants, shortest_path); catalog-site Mermaid labels from edge properties; structured nav. Rewritten for shipped plan-39 edge model. Depends on P1f. | Ready after P1f |
| **P2** | [features/dd-server-cleanup.md](features/dd-server-cleanup.md) | 3 surgical fixes: truncation count, migration API, fuzzy matcher | 1-3 agents |
| **P3** | [features/search-quality-improvements.md](features/search-quality-improvements.md) | Careful ranking fixes (accessor de-ranking, IDS preference), evaluation alignment | 2 agents |
| **P4** | [features/docs-refresh.md](features/docs-refresh.md) | Fix 7 stale docs, rewrite docs/README.md (17+ missing entries) | 1 agent |
| ~~P5~~ | ~~features/sn-extraction-coverage-gaps.md~~ | ✅ Completed — StandardNameSource graph-primary architecture, naming standardization, extraction coverage gap fixes | — |
| ~~P6~~ | ~~features/standard-names/30-dd-semantic-categories.md~~ | ✅ Completed — `fit_artifact` & `representation` NodeCategory enums, DD classifier rules (F1/F2/R1/R2/R4), SN classifier simplified to S0+S1+S2, graph migration (269 fit_artifact, 2209 representation) | — |

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

Notable recent completions:
- `dd-unified-classification.md` — superseded by dd-rebuild.md (merged with multi-pass enrichment)
- `dd-multi-pass-enrichment.md` — superseded by dd-rebuild.md (merged with classification)
- `isn-standard-name-kind.md` — ISN StandardNameKind assessed: no changes needed (scalar/vector/metadata sufficient)
- `26-sn-pipeline-quality-iteration.md` — superseded by #28 greenfield pipeline

### Reference plans (pending/)

Partially implemented plans kept as reference for gap documents.

| Plan | Scope | Status |
|------|-------|--------|
| [35-catalog-schema-redesign.md](features/standard-names/pending/35-catalog-schema-redesign.md) | Catalog schema v2: export/preview/publish/import round-trip, protection model, origin tracking | Phases 0-7 done; Phase 8 (ISNC clean-break regeneration) pending |

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
