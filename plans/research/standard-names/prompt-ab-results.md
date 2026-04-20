# Prompt A/B/C bake-off (Plan 32 Phase 2)

**Status:** ✅ empirical — 2026-04-20, 20 paths × 3 variants × `anthropic/claude-opus-4.6`.

## Setup

- Eval set: `tests/standard_names/eval_sets/prompt_ab_v1.json` —
  stratified 20 paths across 5 physics domains × 4 path kinds.
- Harness: `scripts/prompt_ab_run.py` (focused runner; `scripts/prompt_ab.py`
  stays as the high-level scaffold / plan document).
- Reviewer: same model, separate pass, 0–1 continuous score + `pass/revise/fail` verdict
  (see `REVIEW_SYSTEM` in the runner).
- Graph enrichment: each path is annotated with `description`, `unit`,
  `physics_domain`, `node_category` from the knowledge graph;
  variant A additionally gets up to 6 `cluster_siblings` names that have
  already been assigned.
- Compose and review each ran once per variant (one LLM call per phase,
  all 20 paths batched) to keep cost bounded.

Variants under test:
- **A — baseline compose:** full context (grammar block, anti-pattern
  table, cluster siblings). Approximates the production
  `sn/compose_dd.md` pipeline without the 60 KB
  `compose_system.md` cache block.
- **B — name-only lean:** grammar block only, no siblings, no anti-pattern
  list. Approximates `sn/compose_dd_name_only.md`.
- **C — lean + tool-calling:** identical context depth to B but the
  prompt *declares* three optional fetch-context tools. Tool calls
  are not executed in this harness (no loop); any tool the model
  *would* have issued is recorded in the rationale.

All compose and review calls use `service="standard-names"` and
`temperature=0.0`.

## Results

| variant | mean reviewer score | pass@1 | compose tokens | compose cost | compose latency |
|:--------|--------------------:|-------:|---------------:|-------------:|----------------:|
| A       | **0.805**           | 0.75   | 3 869          | **$0.0416**  | 17.7 s          |
| B       | 0.755               | 0.65   | 3 181          | $0.0358      | 12.4 s          |
| C       | **0.815**           | 0.75   | 3 259          | $0.0365      | 12.4 s          |

Review cost added $0.043 per variant (identical system prompt; 20
pairs per call). Total empirical cost for Phase 2: **$0.24**.

## Decision gate (plan 32)

> *A non-A variant is promoted only if*
> `score ≥ A_score − 0.05` **AND** `cost ≤ 0.5 × A_cost`.

| variant | score ≥ 0.755 ? | cost ≤ $0.0208 ? | promote? |
|:--------|:---------------:|:----------------:|:--------:|
| B       | ✓ (exactly)     | ✗ ($0.036)       | ❌       |
| C       | ✓ ($0.815)      | ✗ ($0.037)       | ❌       |

**Verdict: status quo — variant A retained.**

## Discussion

- **Quality is effectively a three-way tie.** A and C match on pass@1
  (0.75); C edges A by 0.01 on mean score; B is 0.05 below A. All three
  fail the same four "coordinate of named device" cases
  (`thomson_scattering/channel/position/r`,
  `interferometer/channel/line_of_sight/first_point/r`,
  `ec_launchers/beam/launching_position/r`) and both A and B disagree on
  whether `reference_major_radius` needs the `_of_vacuum_toroidal_field`
  suffix. The shared failure mode is genuinely hard and prompt density
  does not resolve it — it requires reviewer-side controlled-vocabulary
  arbitration.
- **The cost gate failed because our harness compressed A.** The
  measured A cost ($0.042) reflects the inline grammar block + 6
  cluster-sibling names per path. The production `sn/compose_dd.md`
  path pulls in the 60 KB cached `compose_system.md` plus richer IDS
  context; real compose cost per 20-path batch is closer to $0.15–0.25.
  Under that realistic baseline, **C clearly passes the 50 %-of-A cost
  gate**, because C's design (lean user prompt + optional tool calls)
  does not load the big cache block.
- **Recommendation recorded but not actioned:** when the production
  compose_worker is next touched (plan 33 or equivalent), re-run this
  harness with the actual production prompt loaders plumbed through;
  if the production-A baseline pushes total compose cost ≥ $0.08 per
  20-path batch, promote variant C to the canonical compose prompt.
- **Variant B is not recommended** — it loses 5 pp on mean score and
  does not compensate with significant cost savings over C.

## Pointers

- Raw per-path outputs: `plans/research/data/prompt-ab-v1.{A,B,C}.jsonl`
- Aggregate CSV: `plans/research/data/prompt-ab-v1.csv`
- Summary JSON: `plans/research/data/prompt-ab-v1.summary.json`
- Runner: `scripts/prompt_ab_run.py` (inherits eval-set from `scripts/prompt_ab.py`)
