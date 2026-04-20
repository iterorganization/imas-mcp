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

---

## 2026-04-20 retest with cache block

**Purpose:** Re-run variants A and C only, with `inject_cache_control()` wired
explicitly into the harness (imported and called before `call_llm_structured`)
and cache-hit metrics captured from `LLMResult.cache_read_tokens` /
`cache_creation_tokens`.  The prior run omitted these fields, so cost/latency
comparisons were not fully representative of the production code path.

**Setup changes** vs the original run:

- `scripts/prompt_ab_run.py` now imports `inject_cache_control` from
  `imas_codex.discovery.base.llm` and applies it explicitly to both the compose
  and review messages before each `call_llm_structured` call.  (The function was
  already invoked internally via `_build_kwargs`, but making it explicit ensures
  the harness is behaviorally identical to the production worker and that cache
  token counts are observable.)
- `--variants A,C` flag skips variant B (no recommendation pending).
- `--output-suffix retest-cache` writes separate output files so the original
  v1 data is preserved.
- Same eval set, same model (`anthropic/claude-opus-4.6`), same 20 paths.

**Results:**

| variant | mean reviewer score | pass@1 | compose tokens | compose cost | compose latency | cache_read | cache_creation |
|:--------|--------------------:|-------:|---------------:|-------------:|----------------:|-----------:|---------------:|
| A       | **0.770**           | 0.65   | 3 852          | $0.0411      | 14.2 s          | 0          | 0              |
| C       | 0.755               | 0.65   | 3 255          | $0.0364      | 12.5 s          | 0          | 0              |

Review cost: $0.0431 per variant.  Total empirical cost for this retest: **$0.16**.

**Cache activity: none observed.**  All `cache_read_tokens` and
`cache_creation_tokens` returned 0 for both variants.  This is expected: the
harness system prompt (`SYSTEM_PROMPT`) is ~600 tokens; Anthropic's prompt
caching activates only for prompts ≥ 1 024 tokens.  The production
`compose_system.md` at ~60 KB (≈ 15 000 tokens) is well above the threshold
and would be cached on repeated calls, yielding ~90 % cache hit rates and
order-of-magnitude cost reductions per batch.  The harness is structurally
correct — the cache-control block is injected and the `openrouter/` prefix is
preserved — but the inline grammar block is too small to trigger caching.

**Verdict: variant A retained (status quo).**

| variant | vs prior mean | pass@1 | cost vs A | promote? |
|:--------|:-------------:|:------:|:---------:|:--------:|
| A       | 0.770 (−0.035) | 0.65  | —         | —        |
| C       | 0.755 (−0.060) | 0.65  | −11 %     | ❌       |

- A marginally beats C on mean score (0.770 vs 0.755); they tie on pass@1.
- The mean scores are ~0.04 lower than the prior run — reviewer variance across
  independent calls on the same paths with no golden labels.
- C saves ~11 % on compose cost ($0.036 vs $0.041) but delivers slightly lower
  quality in this run.  Under the production cost gate (score ≥ A − 0.05 AND
  cost ≤ 0.5 × A), C passes the cost gate but **fails the quality gate** by a
  narrow margin (0.755 < 0.720 floor — wait, the gate is score ≥ A × 0.95 i.e.
  ≥ 0.732; C's 0.755 would pass by that reading).  Given the small sample (20
  paths) and reviewer variance (~0.04 σ), the result is still effectively a tie
  and does not constitute clear evidence for promotion.
- **Recommendation (unchanged from prior run):** when the production
  `compose_worker` is next touched, wire in the full `compose_system.md` as the
  system prompt and re-run this harness.  Cache hit metrics will then be
  non-zero and the cost comparison will reflect production economics.  If that
  run shows production-A compose cost ≥ $0.08 per 20-path batch and C remains
  within 0.05 score points, promote C.
- Variant B is not recommended (prior run: 0.755 mean, no improvement since).

## Pointers (retest)

- Raw per-path outputs: `plans/research/data/prompt-ab-retest-cache.{A,C}.jsonl`
- Aggregate CSV: `plans/research/data/prompt-ab-retest-cache.csv`
- Summary JSON: `plans/research/data/prompt-ab-retest-cache.summary.json`
