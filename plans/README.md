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
> **See also**: [`features/README.md`](features/README.md) for active-plan
> sequencing, layout conventions, and the plan lifecycle.

| Priority | Plan | Scope | Status |
|----------|------|-------|--------|
| **P1** | [features/standard-names/40-sn-search-facility.md](features/standard-names/40-sn-search-facility.md) | **NEW** Grammar-aware SN search & fetch facility: 7 MCP tools mirroring DD palette; tiered grammar streams (T1/T2/T3) preventing `x_component_of_*` floods; backing-function unification in `standard_names/search.py`; `_sn`/`_sns` → `_standard_names` rename audit. `include_standard_names=True` default already shipped. | RD-cleared at v3.2 (commit `31ec17ee`) — dispatch-ready |
| **P1·b** | [features/standard-names/39-structured-fanout.md](features/standard-names/39-structured-fanout.md) | **NEW** Structured multi-pass LLM fan-out: searcher LLM emits typed search queries → backing functions execute → composer consumes targeted context. Depends on plan 40's `search_existing_names` runner. | RD-cleared (commit `5e4d3bbe`) — dispatch after plan 40 lands |
| **P1·c** | [features/standard-names/29-architectural-pivot.md](features/standard-names/29-architectural-pivot.md) | Authoritative SN roadmap: generate/enrich split, ISN grammar upgrade, schema evolution. Supersedes plan 28. | Living — most structural items shipped |
| **P1·d** | [features/standard-names/31-quality-bootstrap-v2.md](features/standard-names/31-quality-bootstrap-v2.md) | 7-workstream bootstrap loop; quarantine-rate + description-coverage + reviewer-score targets. | Active rotation |
| **P1·e** | [features/standard-names/32-extraction-prompt-overhaul.md](features/standard-names/32-extraction-prompt-overhaul.md) | Extraction filter audit + compose-prompt A/B bake-off. | Phase 2 done (see `research/standard-names/prompt-ab-results.md`); Phase 4 deferred |
| **P1·f** | [features/standard-names/34-benchmark-v1.md](features/standard-names/34-benchmark-v1.md) | SN quality benchmark v1: 50 positives × 10 domains + ≥20 negatives; two-reviewer consensus; gating at pass@1 ≥ 0.80. | Scaffolded (20 + 10 filled; runner + mock dry-run) |
| **P1·g** | [features/standard-names/36-catalog-quality-refactor.md](features/standard-names/36-catalog-quality-refactor.md) | Catalog quality refactor v4+round4: graph-backed DD context amplification (Deltas A–J); target-anchored dynamic example library (Delta K); rubric unification (publish threshold = `good` floor = 0.65). | RD round-4 cleared — dispatch-ready |
| **P1·h** | [features/standard-names/33-benchmark-evolution.md](features/standard-names/33-benchmark-evolution.md) | SN benchmark evolution strategy. | Design / research |
| ~~P1e~~ | ~~features/standard-names/39-graph-relationship-completeness.md~~ | ✅ Completed (DELETED — number reused) — structural edges wired into both write_standard_names + catalog import paths; D1–D16 + G1–G10 tests | — |
| ~~P1f~~ | ~~features/standard-names/40-catalog-layout-hierarchy.md~~ | ✅ Completed (DELETED — number reused) — ISNC one-file-per-domain migration; `COMPUTED_FIELDS` round-trip; `edge_model_version: plan_39_v1`. | — |
| ~~P1g~~ | ~~features/standard-names/41-catalog-graph-consumer.md~~ | ✅ Completed (DELETED) — ISN-side NetworkX `DiGraph`; 4 MCP tools; catalog-site Mermaid hierarchy blocks. | — |
| ~~P1h~~ | features/standard-names/completed/42-polling-workers.md | ✅ Completed (2026-04-30 → `completed/`) — budget-managed polling workers; `BudgetManager.pool_admit`; async `_orphan_sweep_tick`. | — |
| ~~P1i~~ | features/standard-names/completed/43-pipeline-rd-fix.md | ✅ Completed (2026-04-30 → `completed/`) — compose-prompt reduction 39K→≤8K tokens; budget split; reviewer-pilot profile; error-gate fixes. | — |
| ~~P1j~~ | features/standard-names/completed/44-sn-graph-renames-batched-embed.md | ✅ Completed (2026-04-30 → `completed/`) — `StandardNameReview` rename; multi-valued `HAS_PHYSICS_DOMAIN`; batch embedding; `sn clear`. | — |
| ~~P8.1~~ | features/standard-names/completed/38-grammar-vnext.md | ✅ Completed — six-pool generate/review/refine architecture; `REFINED_FROM`; `DocsRevision`; `--min-score`/`--rotation-cap`/`--escalation-model`; commit `51fb80dd`. | — |
| ~~—~~ | features/standard-names/completed/superseded/37-grammar-identity-prefix.md | ❌ Superseded (2026-04-30) by plan 38 — preposition-stripping rejected by RD + self-review. | — |
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
