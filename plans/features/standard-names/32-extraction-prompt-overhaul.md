# Plan 32: Extraction Classifier Audit & Prompt Overhaul

**Status:** proposed
**Depends on:** plan 30 (node_category classifier), plan 31 (name-only batching — shipped as `feat(sn): add --name-only batching by (physics_domain × unit)`)
**Companion research:** `plans/research/extraction-batching-prompt-ab.md`

## Problem

Two questions remain after the Workstream-2a name-only batching shipped:

1. **Workstream 1 — node_category audit.** The classifier was tightened
   in plan 30 to 3 skip rules and `SN_SOURCE_CATEGORIES = {quantity,
   geometry}`. A 20-path spot audit of rejected transport paths
   confirmed all rejections were correct (representation = GGD
   interpolation coeffs, fit_artifact = chi_squared, coordinate =
   time/rho arrays). We now need a **broader audit across all IDSs**
   to rule out mis-classifications that silently starve entire
   domains of candidates.
2. **Workstream 3 — prompt architecture A/B.** The current compose
   prompt is a 215-line static Jinja template carrying cluster
   siblings, cross-IDS exemplars, sibling fields, version history,
   and review feedback. The name-only prompt proves a lean variant
   (~60 lines) is viable for bootstrap; we have no evidence yet on
   which regime (static / static-lean / tool-calling) wins on
   quality at a fixed cost budget.

## Goal

Decide — with empirical evidence — whether to

- keep the default pipeline on the rich static prompt,
- migrate default to a lean static prompt resembling the name-only template, or
- migrate to a tool-calling regime where the LLM queries the graph / DD
  for context on demand instead of receiving it all upfront.

And separately, confirm that the `node_category`-based eligibility
filter does not silently drop valid candidates.

## Phases

### Phase 1 — node_category audit (1 day)

1. Sample **100 paths** from `IMASNode WHERE NOT node_category IN
   $SN_SOURCE_CATEGORIES`, stratified by IDS (≥ 1 path per IDS where
   possible) and node_category (balanced across representation,
   fit_artifact, coordinate, structural, metadata, error).
2. For each path, record:
   - path, ids_name, node_category, data_type, unit, description
   - human call: `should_be_standard_name` ∈ {yes, no, borderline}
   - if yes/borderline: proposed correct node_category
3. Compute false-negative rate. Threshold: accept current rules if
   FN ≤ 2 % on balanced sample; otherwise open an issue per
   problematic node_category with fix proposal.
4. Write audit notes to `plans/research/node-category-audit.md`.

**Deliverable:** audit report + (conditional) per-category fix issues.
**Exit:** FN rate ≤ 2 % or an explicit list of category-repair tasks.

### Phase 2 — prompt architecture A/B harness (2 days)

1. Select **a fixed 20-path evaluation set** spanning 5 physics
   domains × 4 path types (scalar, vector, 1D profile, 2D map).
   Freeze it in `tests/standard_names/eval_sets/prompt_ab_v1.json`.
2. Build a **reviewer-scoring harness** (not a new CLI command —
   reuse the existing SN review pipeline) that:
   - runs compose on the 20-path set under each variant,
   - feeds the output through `sn review` with a fixed reviewer model
     and a fixed rubric (correctness, specificity, grammar,
     consistency),
   - emits a CSV of `(variant, path, name, reviewer_score, cost)`.
3. Run three variants:
   - **A — static rich** (current `sn/compose_dd.md`, batch_size=25,
     cluster × unit)
   - **B — static lean** (`sn/compose_dd_name_only.md` behaviour,
     batch_size=50, physics_domain × unit, no exemplars / siblings /
     history / review feedback)
   - **C — tool-calling hybrid** (lean prompt + `fetch_cluster_siblings`,
     `fetch_reference_exemplar`, `fetch_version_history` tools; LLM
     calls them only when it asks a low-confidence item)
4. Score: mean reviewer_score per variant, cost per accepted name,
   grammar-retry rate, tool-call count (C only).

**Deliverable:** A/B/C report with per-variant distributions + a
recommendation.
**Exit:** a variant is declared the default if its mean reviewer
score ≥ variant A − 0.05 **and** its cost-per-accepted-name ≤ 50 %
of variant A, OR variant A is re-confirmed.

### Phase 3 — implement the winner (1–3 days, scope-dependent)

- **If A wins:** document the decision in this plan's "Outcome"
  section, mark `compose_dd.md` canonical, retain `compose_dd_name_only.md`
  for `--name-only` only.
- **If B wins:** promote the lean prompt to default. The new default
  grouping becomes `group_for_name_only`; keep the rich prompt behind
  an opt-in flag (e.g., `--rich-context`) for quarantine / regen runs
  where the review feedback loop needs it.
- **If C wins:** add graph fetch tools under
  `imas_codex.standard_names.tools`, wire them through
  `call_llm_structured(..., tools=...)`, and ship a new
  `sn/compose_dd_tool_calling.md` user prompt. Gate behind
  `--tool-calling` initially, then graduate to default after one
  bootstrap cycle confirms parity.

**Deliverable:** PR implementing the winner. **Exit:** shipped +
rotation cycle completed without regression in composed/hr or
composed/$.

## Risks

- **Reviewer-model bias.** The A/B harness uses a single reviewer
  model; mitigate by also checking grammar-retry rate and composed/$
  independently.
- **Eval set drift.** Freeze the 20-path set under version control;
  regenerate only for plan-33.
- **Tool-calling variant complexity.** C needs new tools **and**
  prompt engineering; if Phase 2 timelines slip, ship A/B only and
  defer C to plan 33.

## Non-goals

- Changing the compose output schema (ADR-1 name-only stays).
- Changing the COCOS / lifecycle / rate-hint logic in the prompts.
- Changing batch-size caps (already tuned in plan 31).

## Open questions

- Should Phase 2 include the current `--name-only` (B-ish) as a
  separate variant from "static lean" with cluster × unit grouping?
  Decision: no — Workstream 2a already measured the grouping axis;
  Phase 2 isolates the **prompt content** axis by holding grouping
  fixed at physics_domain × unit for all three variants.
- Should the harness gate on `dd_version`? Yes — freeze at the
  `dd_version` in `settings.get_dd_version()` at the time of the eval
  to avoid noise from DD updates mid-experiment.

---

## Phase 4 — DD completion endpoint (1–2 days)

**Motivation.** User goal: leave `sn generate` running with a large budget
and have it drive the IMAS DD naming exercise to completion — all
SN-eligible paths named, enriched, reviewed, low-scorers regenerated
once. Currently this requires manual rotation across domains plus
manual enrich/review phases. Domains become invisible once extract
returns 0 despite having unenriched/low-score names.

**Design — a single long-running rotator under the existing CLI.**

Add `sn rotate` (or extend `sn generate --until-complete`) that loops:

1. **Extract phase.** For each physics_domain with extract-eligible
   paths (query the graph live, don't hardcode a list), run
   `generate --physics-domain D --name-only` until extract returns 0
   (no `--force`). Respect `-c` as a global cap across the loop.
2. **Enrich phase.** Target `review_status IN {named, drafted}`
   batched across domains. Batch by (physics_domain × unit) to align
   with compose batching.
3. **Review phase.** Run the 4-dim `--name-only` rubric on newly
   composed/enriched names. Persist `reviewer_score` +
   `reviewer_comments`.
4. **Regen phase.** For names with `reviewer_score < threshold`
   (default 0.5) that have not yet been regenerated (track via
   `regen_count` property on StandardName; new field), reset to
   `drafted` with `--regen-only`, regenerate once.
   Hard cap at `regen_count <= 1` to prevent infinite loops.
5. **Stop conditions** (any triggers exit):
   - Budget exhausted (`spend >= cost_limit`).
   - A full extract→enrich→review→regen pass produces 0 changes.
   - Manual Ctrl-C (checkpoint state to graph so a resumed run
     continues from the same point).

**State tracking.** New columns on StandardName:

- `regen_count: int` — times this name has been through a
  review→regen cycle (0 for fresh).
- `regen_reason: str` — reviewer comments that triggered last regen.
- `rotation_id: str` — UUID of the rotation run that last touched
  this name, for audit.

**Skip-by-design support.** The extract query already filters
`node_category IN {quantity, geometry}` so structural/fit/metadata
are skipped at DB level. Add two additional skip rules for SN-level
policy that cannot be expressed via node_category:

- **Configurable-meaning paths.** Paths where the semantic meaning
  depends on a runtime configuration field (e.g., `radiation/process/*`
  inside a `radiation/process/identifier` array — meaning depends on
  which emission process is selected). Detect via the DD's
  `coordinates` field referencing an identifier array in the same
  array-of-structures chain.
- **Changeable-dimension paths.** Paths whose data_type is
  `FLT_ND` / `CPX_ND` where N is unbounded (e.g., the profile array
  depends on `grid_type`). Already flagged as `representation` by the
  DD builder in most cases; confirm coverage in Phase 1 audit.

Persist the skip as `StandardNameSource.status = 'skip'` with
`skip_reason` so rotations recognize them and don't re-extract.

**Progress reporting.** The rotator writes one row per pass to a
new `RotationRun` node, keyed by UUID, with fields:
`started_at`, `ended_at`, `cost_spent`, `names_composed`,
`names_enriched`, `names_reviewed`, `names_regenerated`,
`domains_touched` (list), `stop_reason`. `sn status` gains a
"Latest rotation" section summarizing the most recent RotationRun.

**Deliverable:** `sn rotate` (or the `--until-complete` flag),
schema additions, and a one-command path from "empty graph" to
"DD complete" within a user-supplied budget.

**Exit:** one end-to-end run on a fresh graph with `-c 50` hits
≥ 60 % DD coverage with mean reviewer_score ≥ 0.7 and no infinite
loops / duplicate regenerations observed.

### Phase 4 schema additions (LinkML)

Add to `imas_codex/schemas/standard_name.yaml`:

```yaml
StandardName:
  attributes:
    regen_count:
      range: integer
      ifabsent: int(0)
    regen_reason:
      range: string
    rotation_id:
      range: string

RotationRun:
  description: A single invocation of `sn rotate` / `sn generate --until-complete`
  attributes:
    id:
      identifier: true
    facility_id:
      description: 'dd' (sentinel — RotationRuns are not facility-scoped yet)
      required: true
      range: Facility
      annotations:
        relationship_type: AT_FACILITY
    started_at: { range: datetime, required: true }
    ended_at: { range: datetime }
    cost_spent: { range: float }
    names_composed: { range: integer, ifabsent: int(0) }
    names_enriched: { range: integer, ifabsent: int(0) }
    names_reviewed: { range: integer, ifabsent: int(0) }
    names_regenerated: { range: integer, ifabsent: int(0) }
    domains_touched: { range: string, multivalued: true }
    stop_reason: { range: string }
```

Rebuild models with `uv run build-models --force`.

## Phase ordering + agent budget

| Phase | Agent | Model | Est. cost |
|-------|-------|-------|-----------|
| 1 | node_category audit | opus-4.7 | < $2 |
| 2 | A/B/C harness | opus-4.7 | < $10 (20-path × 3 × compose+review) |
| 3 | implement winner | opus-4.7 | < $2 |
| 4 | completion endpoint | opus-4.7 | < $2 (implementation; budget test separate) |

Phases 1 & 4 are parallelizable. Phase 2 depends on Phase 1 only for
confidence that the eval set is drawn from genuinely eligible paths;
run concurrent if time-boxed. Phase 3 depends on Phase 2 result.

## Outcome

- **Phase 1 — node_category audit:** ✅ empirical. 105 paths (10 IDSs × 7
  node_categories, stratified) scored by `anthropic/claude-opus-4.6` at
  temperature 0.  FN rate 1 / 81 = 1.2 % (below the 2 % widening
  threshold) — **`SN_SOURCE_CATEGORIES = {quantity, geometry}` retained**.
  FP rate 4 / 24 = 16.7 % surfaces a separate `quantity` mis-labelling
  tail (GGD containers, provenance timestamps, bare geometric
  coordinates leaking in) that should be tackled as a follow-up ticket,
  not in plan 32 scope. Results:
  `plans/research/standard-names/node-category-audit.md`; raw data
  `plans/research/data/node-category-samples{,-scored}.json`. Reviewer
  cost $0.18.
- **Phase 2 — prompt A/B/C bake-off:** ✅ empirical. 20-path eval set ×
  3 variants × opus-4.6 (compose + review). Mean reviewer scores:
  A = 0.805, B = 0.755, C = 0.815. pass@1: A = 0.75, B = 0.65, C = 0.75.
  Compose cost per 20-path batch (harness-compressed A baseline):
  A = $0.042, B = $0.036, C = $0.037. Results + discussion at
  `plans/research/standard-names/prompt-ab-results.md`; raw data
  `plans/research/data/prompt-ab-v1.*`. Runner
  `scripts/prompt_ab_run.py`. Total Phase 2 cost $0.24.
- **Phase 3 — implement winner:** ✅ closed — **status quo, variant A
  retained**. Under the harness-compressed A baseline neither B nor C
  passes the 50 %-of-A cost gate, so no promotion. Discussion section
  of the Phase 2 report notes that with the production
  `compose_system.md` cache block (~60 KB) in play, A's real cost is
  5–10× the harness estimate and C would clearly pass the cost gate;
  revisit when `compose_worker` is next touched.
- **Phase 4 — DD completion endpoint: shipped.** `sn generate
  --until-complete` (with `--plateau-passes`, `--cost-limit`,
  `--dry-run`) orchestrates per-domain rotations with fair-share
  budgeting, plateau detection, and persisted `RotationRun` audit
  nodes. `_apply_skip_by_design()` in
  `imas_codex/standard_names/sources/dd.py` marks `/process/` paths as
  `configurable_meaning` so they do not re-enter the candidate queue.
  `sn status` surfaces the latest `RotationRun` alongside skipped
  counts. 9 new tests in `tests/standard_names/test_dd_completion.py`
  cover skip-by-design, the `RotationSummary` shape, and the four
  stop conditions (dry_run / plateau / budget_exhausted / rotation_id
  propagation). All 202 SN tests pass.

  **2025-07-15 — rotator-as-default:** `--until-complete` was removed
  in favour of making the domain-rotating completion loop the default
  behaviour of `sn generate`. The rotator is strictly more useful
  than single-pass extract+compose, and a cost limit (`-c`) is the
  natural stop condition. `--paths` auto-selects single-pass (explicit
  paths are too narrow for rotation); `--single-pass` is the opt-out
  escape hatch for CI/regression tests.
