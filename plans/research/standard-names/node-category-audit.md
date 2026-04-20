# Node-category false-negative audit (Plan 32 Phase 1)

**Status:** scaffolded (empirical review pending).

## Goal

Test whether `SN_SOURCE_CATEGORIES = {quantity, geometry}` in
`imas_codex/core/node_categories.py` is too restrictive, by sampling
DD paths that are currently filtered out and asking a reviewer whether
they *should* have become Standard Names.

**Exit criterion from plan 32:** if > 2 % of paths in the non-SN
categories are flagged `keep`, widen `SN_SOURCE_CATEGORIES` accordingly
and re-run the DD extract.

## Harness

- `scripts/one_offs/sample_node_category_audit.py` — emits a stratified
  random sample to
  `plans/research/standard-names/node-category-audit-samples.json`.
- Strata: 10 IDSs × 5 categories × 10 samples = up to 500 paths.
- Reviewer fills `reviewer_verdict ∈ {keep, drop, borderline}` per row.

## Run

```bash
uv run python scripts/one_offs/sample_node_category_audit.py \
    --samples-per-stratum 10 \
    --output plans/research/standard-names/node-category-audit-samples.json
```

## Review protocol

1. Open the JSON file; sort by `node_category`.
2. For each row in `coordinate` / `identifier` / `meta`:
   - If the path surfaces a physical quantity with a definable unit →
     `keep`.
   - If it is array index / enum / metadata → `drop`.
   - If context-dependent → `borderline`.
3. Compute the keep-rate per non-SN category. If any exceeds 2 %, update
   `SN_SOURCE_CATEGORIES` and document the change here.

## Findings

_Pending reviewer input._ The harness was scaffolded in the same commit
set as plan 32 Phase 4; empirical review is deferred to a dedicated LLM
sweep or human review pass. When results are in, replace this section
with the per-category keep-rate table, the list of representative
false-negative paths, and the SN_SOURCE_CATEGORIES decision.
