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
