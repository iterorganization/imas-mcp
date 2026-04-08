# Implementation Order — Post-Pivot

**Status:** Approved  
**Date:** 2025-07-18 (revised 2026-04-08)  
**Decision:** Generation pipeline moves to imas-codex; imas-standard-names retains grammar, validation, catalog, website

See `plans/research/standard-names/09-codex-pivot-analysis.md` for the full strategic analysis.

---

## Pivot Resolution (unambiguous)

**Where does pipeline development happen?** All generation/review/minting pipeline work happens in **this repo (imas-codex)**. The imas-standard-names project does not build pipeline infrastructure. It exports grammar and validation for this project to consume.

**Why not in imas-standard-names?** This project already has ~8K lines of proven pipeline infrastructure: graph-as-ledger, supervised workers, LLM structured output, cost tracking, claim-based concurrency, Jinja2 prompt loading. Rebuilding this in standard-names would be duplication.

**Dependency direction:** One-way. Codex `pip install`s standard-names for grammar + validation. Standard-names never imports codex.

**Scope:** Standard names label physics and geometric quantities from **multiple sources**, not just the IMAS Data Dictionary. Sources include:
- IMAS DD paths (IMASNode) — physics quantities defined in the data dictionary
- Facility signals (FacilitySignal) — experimentally measured quantities
- Potentially other graph entities in the future

**CLI placement in codex:** `imas-codex sn` — a **top-level command group**, not nested under `imas` or `discover`. Standard names are broader than IMAS; minting from facility signals has nothing to do with the DD.

---

## Feature Summary

| ID | Feature | Complexity | Repository | Wave |
|----|---------|------------|------------|------|
| 01 | Grammar API Exports | Low | imas-standard-names | 1 |
| 03 | Grammar Extensions (Maarten) | Medium | imas-standard-names | 1 |
| 04 | JSON Schema Contract | Low | imas-standard-names | 1 (after 01) |
| 05 | SN Build Pipeline (multi-source) | High | imas-codex | 2 |
| 06 | Cross-Model Review | Medium | imas-codex | 3 |
| 07 | Benchmarking | Medium | imas-codex | 3 |
| 08 | Publish (YAML + PR) | Medium | imas-codex | 3 |

**Dropped:** Feature 02 (DD Path Linking for existing entries) — existing catalog entries will be archived and regenerated. Path linking happens during codex generation.

## Dependency Graph

```
imas-standard-names work:                 imas-codex work:
                                          
┌──────────────────┐                      
│ 01: Grammar API  │──────────────────────┐
│    Exports       │                      │
└────────┬─────────┘                      │ pip install
         │                                │
         ▼                                ▼
┌──────────────────┐              ┌──────────────────────┐
│ 04: JSON Schema  │─ ─ ─ ─ ─ ─ →│ 05: SN Build Pipeline │
│    Contract      │  validates   │ (sn/ module)          │
└──────────────────┘  against     │                       │
                                  │  Sources:             │
┌──────────────────┐              │  ├─ dd.py    (DD)     │
│ 03: Grammar      │─ ─ ─ ─ ─ ─ →│  └─ signals.py (Fac)  │
│    Extensions    │  extended    └──────────┬────────────┘
│ (Maarten)        │  grammar               │
└──────────────────┘              ┌──────────┼──────────┐
                                  │          │          │
                                  ▼          ▼          ▼
                         ┌────────────┐ ┌────────┐ ┌────────┐
                         │06: Cross   │ │07:     │ │08:     │
                         │  Model     │ │Bench-  │ │Publish │
                         │  Review    │ │mark    │ │(YAML+  │
                         └────────────┘ └────────┘ │ PR)    │
                                                   └────────┘
```

## Wave 1: imas-standard-names (2 parallel agents)

| Agent | Feature | Deliverables |
|-------|---------|-------------|
| A | 01: Grammar API Exports → 04: JSON Schema Contract | Clean public API, `__all__` coverage, `py.typed`, then JSON schema export |
| B | 03: Grammar Extensions | Unary transformations, binary operators, validation rules, missing entries |

Agent A chains Feature 01 → 04 sequentially (04 depends on 01). Agent B works independently on Feature 03.

**Feature 01 details:** Ensure grammar module is importable as a clean library:
- `compose_standard_name()`, `parse_standard_name()` in `__all__`
- All enums (Component, Position, Subject, etc.) exported
- `validate_models()` from `services.py` exported as public API
- No side effects on import
- Verify with `python -c "from imas_standard_names.grammar import compose_standard_name"`

**Feature 03 details:** Grammar changes per Maarten's feedback:
- `square_of_X`, `change_over_time_in_X` unary operators
- `product_of_X_and_Y`, `ratio_of_X_to_Y` binary operators
- `flux_loop_name`, `coil_current`, `passive_current` entries
- Units `None` ≠ `dimensionless` enforcement

**Feature 04 details:** JSON schema contract at the project boundary:
- Export `StandardNameEntry` Pydantic JSON schema as versioned static file
- Store at `imas_standard_names/schemas/entry_schema.json`
- Include in package distribution
- Validation utility callable without full catalog

## Wave 2: imas-codex (parallel, 2-3 agents)

| Agent | Feature | Deliverables |
|-------|---------|-------------|
| A | 05: SN Build Pipeline (core) | `sn/pipeline.py`, `sn/workers.py`, `sn/state.py`, `sn/graph_ops.py` |
| B | 05: Source Plugins + Prompts | `sn/sources/dd.py`, `sn/sources/signals.py`, `llm/prompts/sn/` Jinja2 templates |
| C | 05: CLI + Schema + Progress | `cli/sn.py` (top-level group), graph schema updates, `sn/progress.py` display |

**Pipeline architecture** (in codex, using existing infrastructure):
```
imas_codex/sn/ module (NEW top-level, peer of discovery/ and graph/):
  EXTRACT → COMPOSE → REVIEW → VALIDATE → PUBLISH

  Uses: run_discovery_engine(), WorkerSpec, PipelinePhase
  State: SNBuildState(DiscoveryStateBase) with source, ids_filter, domain_filter, facility_filter
  Graph: StandardName nodes populated on publish; transient candidates in-memory
  
CLI:
  imas-codex sn build --source dd --ids equilibrium --cost-limit 5
  imas-codex sn build --source signals --facility tcv
  imas-codex sn status
  imas-codex sn benchmark
```

**Progress monitoring** — leverage `discovery/base/progress.py` infrastructure:

The SN build uses `DataDrivenProgressDisplay` with `StageDisplaySpec` for each pipeline phase. This gives the same rich terminal UI as `discover paths` and `imas dd build`:

```python
# sn/progress.py — extends DataDrivenProgressDisplay
stages = [
    StageDisplaySpec(key="extract",  label="Extract",  unit="paths"),
    StageDisplaySpec(key="compose",  label="Compose",  unit="names"),
    StageDisplaySpec(key="review",   label="Review",   unit="names"),
    StageDisplaySpec(key="validate", label="Validate", unit="names"),
    StageDisplaySpec(key="publish",  label="Publish",  unit="entries"),
]

# Tracks per-phase: pending/done/error counts, rate, ETA
# Tracks overall: cost, elapsed, worker status, source being processed
# ResourceConfig for LLM cost gauge and token budget
```

Key progress components to reuse from `discovery/base/progress.py`:
- `PipelineRowConfig` — per-phase progress bar with done/pending/error counts and rate
- `WorkerStats` — EMA rate calculation, error rate tracking, scanner status
- `build_resource_section()` — cost gauge, token budget, embed queue
- `build_worker_status_section()` — live worker state (idle/running/crashed)
- `compute_parallel_eta()` — ETA for parallel worker groups
- `ProgressConfig` — source label, cost limit, deadline
- `StreamQueue` — rate-limited event stream for smooth display updates

## Wave 3: imas-codex refinement (parallel, 3 agents)

| Agent | Feature | Deliverables |
|-------|---------|-------------|
| A | 06: Cross-Model Review | Review phase using different LLM family, scoring, accept/reject |
| B | 07: Benchmarking | `imas-codex sn benchmark` command, model comparison tables |
| C | 08: Publish | YAML generation, batched GitHub PRs with confidence tiers to catalog repo |

## Full Summary Table

| # | Feature | Repo | Depends On | Enables | Wave | Agents |
|---|---------|------|-----------|---------|------|--------|
| 01 | Grammar API Exports | SN | — | 04, 05 | 1 | A |
| 03 | Grammar Extensions | SN | — | 05 | 1 | B |
| 04 | JSON Schema Contract | SN | 01 | 05 | 1 | A (after 01) |
| 05 | SN Build Pipeline | Codex | 01, 04 | 06, 07, 08 | 2 | A, B, C |
| 06 | Cross-Model Review | Codex | 05 | 08 | 3 | A |
| 07 | Benchmarking | Codex | 05 | — | 3 | B |
| 08 | Publish (YAML + PR) | Codex | 05, 06 | — | 3 | C |

**Dropped:** Feature 02 (DD Path Linking) — existing entries archived, regenerated by pipeline.

## Superseded Plans

| Old ID | Old Feature | Superseded By |
|--------|-------------|---------------|
| 01 | Prompt System | Codex `llm/prompt_loader.py` + Jinja2 templates |
| 02 | LLM Pipeline Infrastructure | Codex `discovery/base/llm.py` + cost tracking |
| 03 | Batch Generation Pipeline | Codex `discovery/base/engine.py` + Feature 05 |
| 04 | CLI Dispatch Commands | Codex CLI framework + `cli/sn.py` |
| 05 (Phases 2-4) | DD Integration (generation parts) | Feature 05 in codex |

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Grammar version lock between repos | High | JSON schema contract at boundary; CI tests against SN main |
| Codex infrastructure requirement | Low | Generation only needs codex; SN catalog consumers never need Neo4j |
| Cross-model review quality variance | Medium | Benchmarking phase (Feature 07) before production runs |
| PR review avalanche | Medium | Batched PRs by IDS/domain with confidence tiers |
| Name collisions with existing catalog | Medium | Catalog-aware compose phase with dedup checks |
| Binary operator grammar complexity | Medium | Design review in Feature 03 before implementation |

## Definition of Done

A feature is complete when:
1. All deliverables implemented with passing tests
2. 100% test coverage on new code
3. Cross-project interface verified (for features spanning repos)
4. Documentation updated (docstrings + relevant docs/)
5. Code passes `ruff check` and `ruff format`
