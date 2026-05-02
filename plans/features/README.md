# Active Feature Plans

This directory holds **active, ready-to-implement** feature plans. The
canonical roadmap and completed-plan ledger live one level up in
[`../README.md`](../README.md).

## Layout

```
features/
├── README.md                    ← this file (sequencing & scope)
├── <topic>.md                   ← top-level cross-cutting plans
├── completed/                   ← shipped plans kept for reference
└── standard-names/              ← SN-specific plan family
    ├── *.md                     ← active SN plans
    ├── pending/                 ← partially-implemented SN plans
    └── completed/
        └── superseded/          ← rejected / replaced SN plans
```

Per the project rule: **delete plans when fully implemented**. The
`completed/` folders exist for plans whose work is complete enough to keep
as reference (architecture documentation candidates) but not yet promoted
to `docs/`.

## Plan Lifecycle

```
features/<name>.md                       active
   ↓
features/pending/<name>.md               partial — gaps documented
   ↓
features/completed/<name>.md             reference — work shipped
   ↓
DELETE                                   absorbed into code + docs
```

## Active plans (SN family)

The standard-names family is the dominant active workstream. Two **new**
plans landed at HEAD on 2026-04-30 and are ready for implementation
dispatch:

| Order | Plan | Scope | Status |
|-------|------|-------|--------|
| 1 | [standard-names/40-sn-search-facility.md](standard-names/40-sn-search-facility.md) | Grammar-aware SN search & fetch facility: 7 MCP tools mirroring the DD palette (`search_standard_names`, `fetch_standard_names`, `list_standard_names`, `list_grammar_vocabulary`, `find_related_standard_names`, `check_standard_names`, `get_standard_name_summary`); tiered grammar streams (T1/T2/T3) preventing `x_component_of_*` floods; backing-function unification in `standard_names/search.py` mirroring `graph/dd_search.py`; `_sn`/`_sns` → `_standard_names` rename audit; `include_standard_names=True` default already shipped (commit `33514f2a`). | RD-cleared at v3.2 (commit `31ec17ee`, 943 lines) — **dispatch-ready** |
| 2 | [standard-names/39-structured-fanout.md](standard-names/39-structured-fanout.md) | Structured multi-pass LLM fan-out: searcher LLM emits typed search queries → backing functions execute → composer LLM consumes targeted context. Underwritten by plan 40's unified search surface (`search_existing_names` runner). Replaces unbounded agentic-loop alternative with deterministic chains. | RD-cleared (commit `5e4d3bbe`, 1102 lines) — **dispatch after plan 40 lands** |

### Sequencing rationale

**Plan 40 must land first.** Plan 39's `search_existing_names` runner is a
direct caller of plan 40's redesigned `search_similar_standard_names` and
`find_related_standard_names`. Dispatching them in parallel risks two
concurrent rewrites of the same backing-function surface in
`standard_names/search.py`.

**Phase-3 vs Phase-4 of plan 40**: per the v3.2 plan §15.1 ("alias-bridge
window"), the 1-release deprecated aliases (`search_similar_names`,
`search_similar_sns_with_full_docs`) bridge from Phase-3 (MCP tool ship) to
Phase-4 (pipeline-caller migration). If plan 39 is dispatched between
Phase-3 and Phase-4 of plan 40, plan 39 should consume the **new**
canonical names directly — not the aliases — to avoid widening the
alias-bridge window.

### Other active SN plans

These are tracked in [`../README.md`](../README.md) under the P1a–P1d
priority list. Status as of HEAD (2026-04-30):

| Plan | Status |
|------|--------|
| [standard-names/29-architectural-pivot.md](standard-names/29-architectural-pivot.md) | Living roadmap — most structural items shipped |
| [standard-names/31-quality-bootstrap-v2.md](standard-names/31-quality-bootstrap-v2.md) | Active rotation harness |
| [standard-names/32-extraction-prompt-overhaul.md](standard-names/32-extraction-prompt-overhaul.md) | Phase 2 done; later phases deferred |
| [standard-names/33-benchmark-evolution.md](standard-names/33-benchmark-evolution.md) | Design / research |
| [standard-names/34-benchmark-v1.md](standard-names/34-benchmark-v1.md) | Scaffolded; runner + mock dry-run in place |
| [standard-names/36-catalog-quality-refactor.md](standard-names/36-catalog-quality-refactor.md) | RD round-4 cleared — dispatch-ready |

## Active plans (DD / search / docs)

These are independent of the SN workstream and sequenceable in any order.

| Priority | Plan | Scope |
|----------|------|-------|
| P2 | [docs-refresh.md](docs-refresh.md) | Fix stale docs: graph backup/restore CLI, LLM proxy port, llamaindex-agents references, docs/README.md (17+ missing entries) |

### Recently completed (w3 audit, 2026-05-xx)

| Plan | Commits | Notes |
|------|---------|-------|
| `dd-rebuild.md` | `a975c7f9`, `5541de01`, `49a0b850`, `d79bb9ef` | geometry NodeCategory, Pass 2 refine_worker, bug fixes |
| `dd-search-quality-ab.md` | `fddfecd0` | find_related_dd_paths fix, cocos_transformation_type filter |
| `dd-server-cleanup.md` | `fb21ef4e` | list_dd_paths COUNT, migration recipes, fuzzy check_dd_paths |
| `search-quality-improvements.md` | `64abde14`, `1d264f23`, `ebd311b6` | accessor de-ranking, IDS preference, evaluation alignment, Lucene fuzzy |
| `sn-unit-integrity-tests.md` | `9d8207e9` | test_sn_unit_integrity.py + backfill |
| `sn-bootstrap-loop.md` | `eefd8506` | three-layer review pipeline |
| `sn-enrichment-rotation.md` | `a0bc4a9f` | absorbed into `sn run --target docs` |
| `sn-coverage-closure.md` | — | infrastructure shipped; `sn generate` → `sn run` |

## Pending plans (partial implementations)

Plans with gaps that are worth tracking:

| Plan | Gaps |
|------|------|
| [pending/33-cli-harmonization.md](pending/33-cli-harmonization.md) | Phase 4 subcommand consolidation; `-f` freeing on facility-aware commands; `--limit` harmonisation on discover subcommands |

## Recently relocated (2026-04-30 → 2026-05-xx)

| Plan | Destination | Reason |
|------|-------------|--------|
| `42-polling-workers.md` | `standard-names/completed/` | 6-pool polling architecture shipped (commit `650caab2`) |
| `43-pipeline-rd-fix.md` | `standard-names/completed/` | Compose-prompt reduction + budget split + reviewer-pilot all shipped (commit `a8a5b89d` + downstream) |
| `44-sn-graph-renames-batched-embed.md` | `standard-names/completed/` | `StandardNameReview` rename + multi-valued `HAS_PHYSICS_DOMAIN` + batch embedding + `sn clear` shipped (commit `9ef134c9`) |
| `37-grammar-identity-prefix.md` | `standard-names/completed/superseded/` | Self-marked superseded by plan 38 |
| `38-grammar-vnext.md` | `standard-names/completed/` (earlier) | Six-pool refactor shipped in `51fb80dd` |
| `33-cli-harmonization.md` | `pending/` | Phases 1–3 shipped; Phase 4 + partial Phase 3 items deferred |
| `dd-rebuild.md` | DELETED | All code steps shipped (commits `a975c7f9`, `d79bb9ef`, `49a0b850`) |
| `dd-search-quality-ab.md` | DELETED | All features shipped in `fddfecd0` |
| `dd-server-cleanup.md` | DELETED | All 3 fixes shipped in `fb21ef4e` |
| `search-quality-improvements.md` | DELETED | All non-deferred phases shipped |
| `sn-unit-integrity-tests.md` | DELETED | All 3 phases shipped in `9d8207e9` |
| `sn-bootstrap-loop.md` | DELETED | Three-layer review shipped in `eefd8506` |
| `sn-enrichment-rotation.md` | DELETED | Enrichment absorbed into `sn run --target docs` (`a0bc4a9f`) |
| `sn-coverage-closure.md` | DELETED | Expansion infrastructure shipped; protocol was operational only |

## Conventions

- **One number per plan family.** SN plans share the `NN-name.md` numbering
  in the `standard-names/` subfamily; cross-cutting plans use a different
  numbering or a topical filename.
- **Header status field.** Every active plan should carry a `Status:` line
  (e.g. `proposed`, `RD-cleared`, `implementing`, `superseded`).
- **No co-authorship trailers in commits** that touch plan files.
- **Acceptance criteria** must map 1:1 to tests in the implementation
  phase. RD review enforces this.
