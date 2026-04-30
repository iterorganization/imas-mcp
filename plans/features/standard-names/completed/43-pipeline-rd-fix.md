# Plan 43 — Pipeline R&D Fix: Prompt Cost, Budget Split, Error Gates & Reviewer Pilot

> **Status**: READY — dispatch in wave order  
> **Parent plan**: 38-grammar-vnext  
> **R&D sources**: rd1 (budget-allocation-audit), rd2 (reviewer-pattern-audit), rd3 (uncertainty-index-leak), rd4 (cycle-trace-2usd)  
> **Branch**: `main`  
> **Commit baseline**: `650caab2`  

---

## Executive Summary

Round-1 standard-name generation spent $18.87 across 8 physics domains but
achieved only 21.6% mean review coverage.  The root cause is a 39K-token
compose prompt that costs ~$0.15/batch regardless of content size — with 90%
singleton batches, a $2 domain cycle exhausts at batch 9 of 40, starving all
downstream review.  Four R&D audits identified four independent fixes:

1. **Compose prompt reduction** (rd4) — trim static context from 39K→≤8K
   tokens for a ~4× per-batch cost drop ($0.15→$0.04).  This is the
   dominant lever.
2. **Budget split rebalance** (rd1) — shift `TURN_SPLIT` from
   `(0.30, 0.25, 0.15, 0.15, 0.15)` to `(0.15, 0.10, 0.30, 0.30, 0.15)`
   so review phases get 60% of budget instead of 30%.
3. **Error-sibling semantic gate** (rd3) — add
   `_parent_supports_uncertainty_index()` to prevent minting
   `uncertainty_index_of_<process_term>` names that waste review budget.
4. **Reviewer cost pilot** (rd2) — trial Haiku-primary + Opus-escalator
   (Pattern D) for 75% review cost reduction.

After applying fixes A–D, a single-domain probe loop (Phase E) iterates
compose→review→prompt-tune cycles on `magnetic_field_diagnostics` until the
mean review score stabilises at ≥0.85.  Only then does the $50 all-domain
final run (Phase G) execute.

---

## Phase Status

| Phase | Title | Wave | Status | Depends On |
|-------|-------|------|--------|------------|
| A | Compose prompt reduction | 1 | ☐ Not started | — |
| B | Budget split rebalance | 1 | ☐ Not started | — |
| C | Error-sibling semantic gate | 1 | ☐ Not started | — |
| D | Reviewer Haiku pilot | 2 | ☐ Not started | B |
| F | Anti-pattern feedback loop | 2 | ☐ Not started | — |
| E | Single-domain probe loop | 3 | ☐ Not started | A, B, C, D, F |
| G | All-domain $50 final | 4 | ☐ Not started | E (exit criterion met) |

**Wave 1** — Phases A, B, C dispatched in parallel (3 engineer agents).  
**Wave 2** — Phases D and F dispatched in parallel (2 engineer agents).  
**Wave 3** — Phase E dispatched (1 architect agent).  
**Wave 4** — Phase G dispatched after E exit criterion (1 architect agent).  
**Fleet cap**: 3 agents max in any wave.

---

## Phase A — Compose Prompt Reduction

**Goal**: Reduce the per-batch compose cost from ~$0.15 to ≤$0.04 by trimming
the 39K-token system prompt to ≤8K tokens.

**Root cause (rd4)**: The system prompt `sn/compose_system.md` (74 KB, 1152
lines) is rendered once and reused for all batches. It includes four
{% include %} partials plus extensive inline sections. The user prompt
additionally injects up to 200 `existing_names`.  Output averages 38–639
tokens.  Cost is 98% input-driven.

### Token budget analysis

| Section | Est. tokens | Action |
|---------|------------|--------|
| `_grammar_reference.md` (4.3 KB) | ~1,100 | KEEP — core grammar rules |
| `_exemplars.md` (4.1 KB) | ~1,000 | TRIM — keep ≤3 exemplars, cut from 77 lines |
| `_exemplars_name_only.md` (9.8 KB) | ~2,500 | TRIM aggressively — keep ≤5 name-only exemplars |
| `_compose_scored_examples.md` (0.9 KB, dynamic) | ~200–2,000 | CAP at 5 examples |
| Hard pre-emit checks (lines 22–68) | ~1,200 | KEEP — prevents name errors |
| REJECT / FORBIDDEN blocks (lines 70–155) | ~2,000 | CONDENSE — merge into single compact table |
| vNext composition guidance (lines 182–221) | ~800 | KEEP — essential for grammar |
| BANNED PREFIXES table (lines 198–221) | ~400 | KEEP — compact already |
| Anti-pattern gallery (entries 1–5, lines 251–298) | ~1,200 | TRIM to 2 entries |
| Naming Guidance (Jinja template, lines 300–319) | ~2,000 | MOVE to prompt cache — render once at ISN init |
| Documentation Quality Guidance (lines 321–339) | ~1,500 | REMOVE for `--target names` mode |
| Curated Examples (lines 341–351) | ~3,000 | TRIM to 3 examples max |
| Tokamak Ranges (lines 352–366) | ~1,500 | REMOVE for `--target names` — only needed for docs |
| Applicability (lines 368–382) | ~500 | CONDENSE to 3 lines |
| Quick Start / Common Patterns / Critical Distinctions / Anti-Patterns (lines 384–412) | ~2,000 | REMOVE — redundant with pre-emit checks |
| Peer-Review Quality Rules NC-1 through NC-29 (lines 414–1002) | ~12,000 | TRIM to top-10 most-violated rules |
| Output Format / Schema (lines 1004–end) | ~2,000 | KEEP — structural requirement |
| `existing_names` (user prompt, ≤200 entries) | ~3,000 | CAP at 50 names |
| `nearby_existing_names` (user prompt) | ~500 | KEEP |
| `reference_exemplars` (user prompt, ≤5) | ~1,000 | CAP at 3 |
| `domain_vocabulary` preseed (user prompt) | ~500 | KEEP |
| `reviewer_themes` (user prompt) | ~200 | KEEP |

**Target token budget**: ≤8,000 tokens system + ≤2,000 tokens user = ≤10K total
(~4× reduction from current ~39K).

### Implementation strategy

Create a new prompt template `sn/compose_system_lean.md` that:

1. Keeps: grammar reference, pre-emit checks, vNext guidance, REJECT table
   (condensed), output schema, BANNED PREFIXES.
2. Trims: exemplars to 3+5 entries, anti-pattern gallery to 2 entries,
   NC rules to top-10, scored examples capped at 5.
3. Removes: Documentation Quality Guidance, Tokamak Ranges, Applicability
   (verbose form), Quick Start, Common Patterns, Critical Distinctions,
   Anti-Patterns (template section).
4. Caps user-prompt injections: `existing_names[:50]`,
   `reference_exemplars[:3]`.

Use a feature flag (`--lean-prompt` / config key `compose_lean = true`) so
the full prompt is still available for `--target docs` mode.  For
`--target names`, the lean prompt is the default.

### Files to modify

| File | Change |
|------|--------|
| `imas_codex/llm/prompts/sn/compose_system_lean.md` | **NEW** — lean system prompt |
| `imas_codex/llm/prompts/shared/sn/_exemplars_name_only.md` | TRIM to ≤5 exemplars |
| `imas_codex/standard_names/workers.py` ~L1213 | In `compose_worker()`, select `compose_system_lean` template when target is names |
| `imas_codex/standard_names/workers.py` ~L1328 | Render lean system prompt instead of full |
| `imas_codex/standard_names/workers.py` ~L1423 | Cap `existing_names[:50]`, `reference_exemplars[:3]` |
| `imas_codex/standard_names/context.py` ~L35 | Add `build_lean_compose_context()` that skips tokamak_ranges, documentation_guidance |
| `imas_codex/standard_names/example_loader.py` | Cap scored examples at 5 |

### Tests to add

| Test | File |
|------|------|
| `test_lean_prompt_renders_under_8k_tokens` | `tests/standard_names/test_prompt_size.py` (NEW) — use chars/3 heuristic (24K chars ≈ 8K tokens, conservative for mixed prose+JSON) |
| `test_lean_prompt_contains_grammar_reference` | same |
| `test_lean_prompt_omits_tokamak_ranges` | same |
| `test_existing_names_capped_at_50` | same |

### Acceptance criteria

1. Render the lean system prompt for a 1-item batch → token count ≤ 8,000
   (measured via `tiktoken` cl100k_base or Anthropic tokenizer).
2. Render the lean user prompt for the same batch → token count ≤ 2,000.
3. Re-run a 4-name `divertor_physics` batch with `--cost-limit 0.50`:
   compose cost ≤ $0.04/batch (was $0.15/batch).
4. No grammar validation regressions: all pre-emit checks still present.

### Rollback strategy

Revert `compose_system_lean.md` creation and restore original template
selection in `workers.py`.  One-commit revert via `git revert <sha>`.

### Agent type: engineer

---

## Phase B — Budget Split Rebalance

**Goal**: Shift budget from compose (over-provisioned at 30%) and enrich
(under-used at 25%) toward review phases (starved at 15% each).

**Root cause (rd1)**: `TURN_SPLIT = (0.30, 0.25, 0.15, 0.15, 0.15)` gives
review_names only $0.75 on a $5 budget.  A 15-name batch requires $1.125
reservation → instant exhaustion.  Round-1 achieved 0% review coverage in
4 of 6 domains.

### Implementation

1. **Update `TURN_SPLIT`** in `turn.py` line 31:
   ```python
   TURN_SPLIT: tuple[float, float, float, float, float] = (0.15, 0.10, 0.30, 0.30, 0.15)
   ```

2. **Add per-phase hard caps** — phases must not overspend their allocation
   even if adaptive budget offers more.  Add a `max_phase_budget` parameter
   to `phase_budget()` that returns `min(adaptive, split_allocation * 1.5)`
   (50% overshoot allowed, not unbounded).  This prevents compose from
   burning 2.26× its share.

3. **Update adaptive shares** — currently `_ADAPTIVE_SHARES` gives
   `review_names` 0.45 and `review_docs` 0.35.  After Phase A reduces
   compose cost, the adaptive budget should give more to names when docs
   are deferred: when `--target names`, set `review_names: 0.80,
   regen: 0.20` (no docs budget needed).

4. **Transitional guard** — add `compose_lean: bool = False` field to
   `TurnConfig`.  In `__post_init__`, if `compose_lean` is False, override
   `self.split` to the legacy `(0.30, 0.25, 0.15, 0.15, 0.15)`.  The
   constructor in `turn.py` reads `get_compose_lean()` from settings to
   set this flag.  When Phase A lands and enables lean prompts, the new
   split activates.  This prevents B from breaking existing compose
   behavior if it merges before A.

### Files to modify

| File | Change |
|------|--------|
| `imas_codex/standard_names/turn.py` L31 | Update `TURN_SPLIT` constant |
| `imas_codex/standard_names/turn.py` L113–120 | Add 1.5× hard cap to `phase_budget()` |
| `imas_codex/standard_names/turn.py` L563–589 | Update `_ADAPTIVE_SHARES` for `--target names` |

### Tests to add

| Test | File |
|------|------|
| `test_turn_split_sums_to_one` | `tests/standard_names/test_turn.py` (NEW or append) |
| `test_turn_split_values` | same — assert exact `(0.15, 0.10, 0.30, 0.30, 0.15)` |
| `test_phase_budget_hard_cap` | same — verify phase can't exceed 1.5× its share |
| `test_adaptive_shares_names_only` | same — verify 80/20 split when `--target names` |
| `test_legacy_split_when_lean_disabled` | same — verify old (0.30, 0.25, ...) when compose_lean=False |

### Acceptance criteria

1. `TURN_SPLIT` sums to 1.0.
2. With $5 budget: `phase_budget(2)` returns $1.50 (review_names = 30%).
3. With $2 budget and `compose_lean=True`: compose phase hard-capped at
   $0.45 (0.15 × $2 × 1.5).  With `compose_lean=False`: legacy split
   active, compose gets $0.90 (0.30 × $2 × 1.5).
4. No regression in existing turn orchestration tests.

### Rollback strategy

Single constant change — `git revert <sha>` reverts the line.

### Agent type: engineer

---

## Phase C — Error-Sibling Semantic Gate

**Goal**: Prevent `uncertainty_index_of_<P>` siblings from being minted when
the parent `P` is a process term, identifier, or dimensionless integer.

**Root cause (rd3)**: `mint_error_siblings()` generates
`uncertainty_index_of_<P>` for every parent with `HAS_ERROR` relationships,
regardless of semantic suitability.  Process terms like
`power_due_to_thermalization` produce nonsensical uncertainty names that cost
review cycles to score low.

### Implementation

Add `_parent_supports_uncertainty_index(parent_name: str, unit: str | None) -> bool`
to `error_siblings.py` before `mint_error_siblings()`.

**Signature note**: The function needs `unit` because two deny rules are
unit-dependent.  The call site in `mint_error_siblings()` already has `unit`
as a parameter — pass it through.

**Deny rules** (name-based, no unit needed):
- Parent contains `due_to_` → process attribution
- Parent contains `caused_by_` → process attribution
- Parent starts with `constant_` or `generic_` → data-type descriptor
- Parent ends in `_type`, `_flag`, `_identifier`, `_status`,
  `_code`, `_name` → metadata/categorical

**Deny rules** (unit-dependent):
- Parent contains `_count` AND unit is dimensionless/None → counter
- Parent ends in `_ratio` or `_fraction` AND unit is dimensionless/None →
  dimensionless ratio (uncertainty index on a ratio is meaningless)

**Deliberately NOT denied**: Parent ending in `_index` (too aggressive —
legitimate physical indices like `safety_factor_q_index` can have
measurement uncertainty).

**Gate insertion** in `mint_error_siblings()` around line 95:
```python
if suffix == "_error_index" and not _parent_supports_uncertainty_index(parent_name, unit):
    logger.info("Skipped uncertainty_index for unsuitable parent %r", parent_name)
    continue
```

Note: Only gate `_error_index` siblings.  `_error_upper` and `_error_lower`
are always valid (they represent measurement bounds on any quantity).

### Files to modify

| File | Change |
|------|--------|
| `imas_codex/standard_names/error_siblings.py` | Add `_parent_supports_uncertainty_index()` + gate |

### Tests to add

| Test | File |
|------|------|
| `test_uncertainty_index_denied_for_process_term` | `tests/standard_names/test_error_siblings.py` |
| `test_uncertainty_index_denied_for_metadata_field` | same |
| `test_uncertainty_index_denied_for_constant_prefix` | same |
| `test_uncertainty_index_denied_for_dimensionless_ratio` | same — `_ratio` + unit=None |
| `test_uncertainty_index_allowed_for_physical_quantity` | same |
| `test_uncertainty_index_allowed_for_physical_index` | same — e.g. parent ending `_index` with real unit → allowed |
| `test_error_upper_lower_not_affected_by_gate` | same — pass mixed `[_error_upper, _error_lower, _error_index]` for process term → assert exactly 2 returned |

### Acceptance criteria

1. `mint_error_siblings("power_due_to_thermalization", ...)` with
   `_error_index` in node IDs → returns 0 `uncertainty_index_*` siblings.
2. `mint_error_siblings("power_due_to_thermalization", ...)` with
   `_error_upper` and `_error_lower` → still returns 2 siblings.
3. `mint_error_siblings("plasma_current", ...)` with all 3 error suffixes →
   returns 3 siblings (including `uncertainty_index_of_plasma_current`).
4. All existing `test_error_siblings.py` tests pass unchanged.

### Rollback strategy

Revert single function addition — `git revert <sha>`.

### Agent type: engineer

---

## Phase D — Reviewer Haiku Pilot

**Goal**: Validate Pattern D (Haiku primary + Opus escalator) for 75% review
cost reduction.  If Haiku quality holds, flip the default configuration.

**Root cause (rd2)**: The 3-model RD-quorum (Opus→GPT→Sonnet) costs ~$0.70
per reviewed name.  Pattern D projects $0.0075/name if Haiku-Opus dispute
rate stays below 20%.

**Dependency**: Requires Phase B (budget split) to be merged so the
reviewer can actually run within budget.

### Implementation

The current 2-model semantics in `pyproject.toml` (lines 340–342) define
"blind primary + blind secondary, no escalator" — BOTH models always run.
This means Pattern D ("Haiku primary, Opus escalator only on disagreement")
is **NOT achievable** with the 2-model config path.

Two implementation options:

**Option 1 (recommended)**: Use 3-model config with Haiku as primary and
secondary, Opus as escalator.  This leverages existing RD-quorum logic:
```toml
[tool.imas-codex.sn.review.names.pilot]
models = [
  "openrouter/anthropic/claude-haiku-4.5",     # cycle 0 primary (blind, cheap)
  "openrouter/anthropic/claude-haiku-4.5",     # cycle 1 secondary (blind, cheap)
  "openrouter/anthropic/claude-opus-4.6",      # cycle 2 escalator (disputes only)
]
```
Cost: 2× Haiku always ($0.0006/name) + Opus on disputes (~20% × $0.015) =
~$0.004/name.  Still an 85% reduction.

**Option 2**: Add a new `escalation_only` mode to `review/pipeline.py` where
the 2nd model only runs on items where the 1st model's confidence is below a
threshold.  More complex, deferred to a future plan.

Recommend Option 1: zero new review-pipeline code, just a config change.

**Config profile in `pyproject.toml`** — add the pilot block.  The review
pipeline reads an environment variable `IMAS_CODEX_SN_REVIEW_PROFILE=pilot`
to select the model chain.  Default remains the current Opus→GPT→Sonnet.

### Pilot protocol

1. Select 4 names from `plasma_initiation` (round-1 reviewed, mean 0.795).
2. Run Haiku-pilot review with `$0.50` cap.
3. Run default (Opus) review on same 4 names with same cap.
4. Compare: per-dimension scores, per-name composite, cost.
5. **Go/NoGo**: If Haiku-Opus dispute rate >50% on these 4 names, fall back
   to Pattern B (Opus+Sonnet, 2-model) instead.

### Files to modify

| File | Change |
|------|--------|
| `pyproject.toml` after L357 | Add `[tool.imas-codex.sn.review.names.pilot]` block |
| `imas_codex/standard_names/review/pipeline.py` | Read `IMAS_CODEX_SN_REVIEW_PROFILE` env var, select model list |
| `imas_codex/settings.py` | Add `get_review_profile()` helper |

### Tests to add

| Test | File |
|------|------|
| `test_pilot_profile_selects_haiku_opus` | `tests/standard_names/test_review_config.py` (NEW) |
| `test_default_profile_selects_opus_gpt_sonnet` | same |
| `test_env_override_review_profile` | same |

### Acceptance criteria

1. `IMAS_CODEX_SN_REVIEW_PROFILE=pilot` selects Haiku+Haiku+Opus chain.
2. Pilot run on 4 names completes within $0.50 budget.
3. Per-dimension score delta vs default ≤ 0.15 (median across names).
4. Cost per name: ≤ $0.02 (pilot) vs ~$0.70 (default).
5. If pilot passes, update `[tool.imas-codex.sn.review.names]` default to
   pilot chain.  If not, update to Pattern B (Opus+Sonnet, 2-model).

### Rollback strategy

Remove pilot config block and env-var reader.  Single-commit revert.

### Agent type: engineer

---

## Phase E — Single-Domain Probe Loop

**Goal**: Iterate compose→review→prompt-tune cycles on one domain until
mean review score reaches ≥0.85 across two consecutive cycles.

**Probe domain**: `magnetic_field_diagnostics` — 11 names, 0% reviewed in
round-1 (worst performer), best stress test for review-starvation fix.

**Depends on**: All of A, B, C, D merged.

### Cycle protocol

```
For each cycle N:
  1. sn clear --force --physics-domain magnetic_field_diagnostics
  2. sn run --physics-domain magnetic_field_diagnostics \
          --target names --cost-limit 2.00 --turn-number N
  3. Query graph: mean(reviewer_score_name) WHERE physics_domain = 'magnetic_field_diagnostics'
  4. If mean < 0.85:
       a. Extract reviewer comments → research/comments-mfd-cycle-N.jsonl
       b. Analyze top-3 recurring criticisms
       c. Update prompts/vocab to address criticisms
       d. Repeat from step 1
  5. If mean ≥ 0.85 for two consecutive cycles → EXIT
```

**Budget**: $2.00 per cycle, max 5 cycles = $10.00 total probe budget.

### Exit criterion

Mean `reviewer_score_name` ≥ 0.85 across all `magnetic_field_diagnostics`
names with `validation_status = 'valid'`, achieved in two consecutive cycles
(i.e., prompt changes in cycle N-1 did not regress in cycle N).

### Files to modify

| File | Change |
|------|--------|
| Prompt templates (various) | Iterative edits based on reviewer feedback |
| ISN vocabulary files (upstream) | Potential vocab additions surfaced by compose |

### Acceptance criteria

1. Mean score ≥ 0.85 on `magnetic_field_diagnostics`.
2. Zero `uncertainty_index_of_<process_term>` names in graph (Phase C gate).
3. Compose cost ≤ $0.04/batch (Phase A effect verified in production).
4. Review coverage ≥ 80% of valid names (Phase B effect verified).

### Agent type: architect (iterative prompt tuning requires judgment)

---

## Phase F — Anti-Pattern Feedback Loop

**Goal**: Persist reviewer comments across `sn clear` cycles and provide
tooling to mine recurring criticisms.

**Key finding**: `sn clear` deletes Review nodes (Step A in
`clear_standard_names`, line 2627–2648).  Reviewer comments stored as
`comments_per_dim_json` on Review nodes are destroyed.  This breaks the
feedback loop.

### Implementation

1. **Pre-clear comment dump** — before each `sn clear`, export Review
   comments to `research/comments-<domain>-cycle-<N>.jsonl`:
   ```cypher
   MATCH (r:Review)-[:REVIEWS]->(sn:StandardName)
   WHERE sn.physics_domain = $domain
   RETURN sn.id AS name, r.score AS score,
          r.comments_per_dim_json AS comments,
          r.llm_model AS model, r.reviewed_at AS at
   ```

2. **`sn analyse-comments` CLI command** — reads JSONL files, groups by
   dimension, counts recurring criticism keywords, surfaces top-N as
   candidate prompt amendments.

   ```bash
   uv run imas-codex sn analyse-comments --domain magnetic_field_diagnostics
   # Output:
   # Top criticisms (grammar): "instrument as prefix" (4 occurrences)
   # Top criticisms (semantic): "bare outline_point" (3 occurrences)
   # Suggested prompt additions: ...
   ```

3. **Alternative (simpler)**: REPL recipe function
   `analyse_review_comments(domain)` that queries graph directly before
   clear.  No new CLI command needed.  Recommend this for Phase F v1.

### Files to modify

| File | Change |
|------|--------|
| `imas_codex/standard_names/graph_ops.py` | Add `export_review_comments(domain, output_path)` |
| `imas_codex/cli/sn.py` | Add `analyse-comments` subcommand (or defer to REPL recipe) |

### Tests to add

| Test | File |
|------|------|
| `test_export_review_comments_writes_jsonl` | `tests/standard_names/test_graph_ops.py` |
| `test_export_review_comments_empty_domain` | same |

### Acceptance criteria

1. Before every `sn clear` in Phase E, comments are dumped to JSONL.
2. JSONL files contain: name, score, model, per-dimension comments.
3. `analyse_review_comments()` returns top-3 criticisms per dimension.

### Rollback strategy

Delete JSONL export function.  No schema changes.

### Agent type: engineer

---

## Phase G — All-Domain $50 Final

**Goal**: Run the full pipeline across all physics domains with the tuned
prompts, budget split, and reviewer configuration established in Phases A–F.

**Depends on**: Phase E exit criterion met.

### Protocol

1. Use `scripts/run_domain_cycles.py` in single-domain-at-a-time mode.
2. Total budget: $50.00, fair-share split across all domains with eligible
   work (domains with 0 unprocessed items are skipped).
3. Target: `--target names` only.
4. Quality gate: abort and roll back if any domain's mean score regresses
   below 0.70 after review completes.

### Budget arithmetic (post Phase A + D)

With compose at $0.04/batch and Haiku-pilot review (Option 1: Haiku×2 +
Opus escalator at ~20% dispute rate):
- Per name: ~$0.04 (compose) + $0.01 (enrich) + $0.004 (review, 2× Haiku
  + 20% Opus escalation) = ~$0.054
- 500 names × $0.054 = $27.00 — within $50 budget.
- Residual ~$23 covers retries (~10% rate observed in round-1), re-reviews,
  and domains with higher escalation rates.
- If dispute rate is higher (40%): ~$0.008/name review → 500 × $0.058 =
  $29, still within budget.

**Worst-case guard**: If Pattern D pilot failed and Pattern B (Opus+Sonnet,
2-model always-run) is active: review ≈ $0.024/name → 500 × $0.074 = $37,
still within budget.

### Abort mechanism

After each domain completes review, the turn orchestrator checks:
```python
# In run_domain_cycles.py or turn.py post-review hook
mean_score = query_mean_score(domain)
if mean_score < 0.70:
    logger.error("Domain %s regressed below 0.70 (mean=%.3f), aborting", domain, mean_score)
    raise QualityGateError(domain, mean_score)
```

**Files to modify**: `scripts/run_domain_cycles.py` — add post-domain score
check.  This is a manual abort: the script stops, operator reviews, and
either resumes or rolls back.  No auto-rollback (too risky on a $50 run).

### Acceptance criteria

1. All domains with >0 eligible names have review coverage ≥ 80%.
2. No domain has mean score < 0.70.
3. Total spend ≤ $50.00.
4. Graph state: all names `pipeline_status IN ['enriched', 'drafted']` with
   `reviewer_score_name IS NOT NULL`.

### Rollback strategy

`sn clear --force` removes all drafted names.  Round-1 names at `accepted`
status are preserved.

### Agent type: architect

---

## Cost Tracking Audit

**Constraint 5**: All cost sites must be tracked in graph (per-phase
`llm_cost` fields).

Current state from code inspection:

| Phase | `llm_cost` tracked? | Location |
|-------|-------------------|----------|
| compose | ✅ Yes | `workers.py` L1451–1455, written to StandardName node |
| enrich | ✅ Yes | `enrich_workers.py` |
| review_names | ✅ Yes | `review/pipeline.py` L143, written to Review node |
| review_docs | ✅ Yes | Same pipeline |
| regen | ❓ Verify | Check `regen` worker — may inherit from compose |

**Action**: Phase F engineer should verify regen cost tracking as part of
the graph_ops work (both involve auditing cost/comment persistence).  If
missing, add `llm_cost` write in the regen worker.  This is a sub-task of
Phase F, not a separate phase.

---

## Worker Async + Polling-Claim Verification

**Constraint 4**: Workers async + polling-claim pattern (commit 650caab2)
must not regress.

All phases must verify that `compose_worker`, `enrich_worker`, and
`review_worker` use `asyncio` with `budget_manager.reserve()` and
`@retry_on_deadlock()` claim patterns.  No phase in this plan changes the
worker concurrency model — this is a hold-the-line constraint, not new work.

---

## Rubber-Duck Critique Findings

### Adopted

1. **Phase A: token counting test should use conservative heuristic** —
   chars/3 (not chars/4) because compose prompts contain JSON schemas and
   dense identifier lists that tokenize at ~3 chars/token.  Threshold:
   24,000 chars ≈ 8K tokens.
   *Adopted: updated test spec.*

2. **Phase B: 1.5× hard cap on phase_budget breaks compose if A hasn't
   landed** — added a transitional guard: `compose_lean: bool` field on
   `TurnConfig`, with legacy split `(0.30, 0.25, ...)` when disabled.
   The new split activates only when lean prompt is enabled.
   *Adopted: documented `__post_init__` override + new test.*

3. **Phase C: deny list needs unit parameter** — gate function signature
   changed to `(parent_name, unit)`.  Added `_ratio`/`_fraction` + unit
   deny rule.  Dropped `_index` suffix deny (too aggressive — legitimate
   physical indices have real uncertainty).
   *Adopted: signature, deny rules, and tests updated.*

4. **Phase D: 2-model config doesn't produce escalator semantics** — the
   existing 2-model path runs both models always (blind+blind), not
   primary+escalator.  Switched to 3-model config (Haiku×2 + Opus
   escalator) which leverages existing RD-quorum logic.
   *Adopted: rewrote Phase D implementation.*

5. **Phase E depends on Phase F functionality** — Phase E step 4a
   (export reviewer comments) IS Phase F.  Moved F to Wave 2 so it's
   complete before E starts.  Phase E now declares dependency on F.
   *Adopted: updated wave table.*

6. **Phase F: JSONL export should include `run_id` and `turn_number`** for
   cycle traceability.
   *Adopted: add both fields to the export query.*

7. **Phase G budget arithmetic used wrong review cost** — recomputed with
   realistic Haiku×2 + 20% Opus escalation.  $27 for 500 names, $37
   worst-case with Pattern B.  Both within $50.
   *Adopted: updated arithmetic section.*

8. **Phase G abort mechanism was undefined** — added post-domain score
   check in `run_domain_cycles.py` with `QualityGateError`.  Manual
   abort, not auto-rollback.
   *Adopted: added implementation spec.*

9. **Phase A line-number references pointed to wrong site** — system
   prompt is rendered in `compose_worker()` at ~L1213/L1328, not in
   `_compose_batch_body`.  Updated file modification list.
   *Adopted: corrected file table.*

10. **Phase A missing partial template files** — `_exemplars_name_only.md`
    and `example_loader.py` were not in the modification list.
    *Adopted: added both.*

### Deliberately set aside

1. **"Phase D should also pilot Pattern B (Opus+Sonnet) as a comparison
   point"** — rd2 already analysed Pattern B theoretically.  Running both
   pilots doubles cost.  We'll run D only and fall back to B if D fails.
   *Reason: budget conservation.*

2. **"Phase E should use a larger probe domain for statistical
   significance"** — `magnetic_field_diagnostics` has only 11 names, which
   is a small sample.  However, it's the worst-performing domain and the
   user explicitly recommended it.  Statistical power is less important than
   exposing the review-starvation bug.
   *Reason: user directive overrides statistical concern.*

3. **"Consider adding a `--dry-run` mode to the probe loop"** — the probe
   loop already has `sn clear` as the reset mechanism.  A dry-run would
   require mocking the entire LLM call chain.
   *Reason: complexity not justified for a probe.*

4. **"Phase D pilot sample too small (n=4) for statistical conclusions"** —
   true, but this is a cost-constrained pilot ($0.50 cap).  We use median
   per-dimension delta ≤0.15 as the gate, which tolerates single outliers.
   If quality is borderline, we'll expand to 10 names before flipping
   the default.
   *Reason: $0.50 cap constraint; expandable if results are ambiguous.*

5. **"Phase E exit criterion (2 consecutive cycles) may ping-pong with LLM
   nondeterminism"** — accepted risk.  If cycle 3 regresses, we continue
   iterating.  The 5-cycle budget cap ($10 total) prevents runaway.
   *Reason: bounded by budget cap; acceptable variance.*

---

## Sequencing Summary

```
Wave 1 (parallel, 3 engineers):
  ├─ Phase A: compose prompt reduction
  ├─ Phase B: budget split rebalance  
  └─ Phase C: error-sibling gate

Wave 2 (parallel, 2 engineers):
  ├─ Phase D: reviewer Haiku pilot (after B)
  └─ Phase F: anti-pattern feedback loop

Wave 3 (1 architect):
  └─ Phase E: single-domain probe loop (after A+B+C+D+F)

Wave 4 (gated, 1 architect):
  └─ Phase G: all-domain $50 final (after E exit)
```
