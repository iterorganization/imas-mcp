# Node-category false-negative audit (Plan 32 Phase 1)

**Status:** ✅ empirical — 2026-04-20, 105 paths reviewed by `anthropic/claude-opus-4.6`.

## Goal

Test whether `SN_SOURCE_CATEGORIES = {quantity, geometry}` in
`imas_codex/core/node_categories.py` is too restrictive, by sampling
DD paths that are currently filtered out and asking a reviewer
whether they *should* become Standard Name candidates.

**Exit criterion (plan 32):** if > 2 % of paths in non-SN categories
are flagged `keep`, widen `SN_SOURCE_CATEGORIES` and re-run the DD
extract. Otherwise retain status quo.

## Harness

- `scripts/one_offs/sample_node_category_audit.py` — stratified random
  sample across 10 IDSs × 7 categories.
- `scripts/one_offs/score_node_category_audit.py` — LLM reviewer pass
  (`anthropic/claude-opus-4.6`, `service="standard-names"`,
  temperature 0, batch of 20 paths per call).
- Samples: `plans/research/data/node-category-samples.json`
- Scored:  `plans/research/data/node-category-samples-scored.json`
- Reviewer cost: **$0.18**.

Identifier & metadata strata were added cross-IDS after the initial
stratified run (identifier has no `dynamic`/`constant` node-types and
metadata fields carry no description, so both were under-sampled by
the generic SELECT).

## Results — confusion matrix

| category       | in SN set | N  | keep | drop | borderline |
|:---------------|:---------:|---:|-----:|-----:|-----------:|
| quantity       | ✅        | 23 |   18 |    3 |          2 |
| geometry       | ✅        |  1 |    0 |    1 |          0 |
| coordinate     | ❌        | 30 |    1 |   29 |          0 |
| identifier     | ❌        | 15 |    0 |   15 |          0 |
| metadata       | ❌        | 15 |    0 |   15 |          0 |
| representation | ❌        | 12 |    0 |    8 |          4 |
| fit_artifact   | ❌        |  9 |    0 |    9 |          0 |

### Headline rates

- **False-negative rate** (non-SN paths the reviewer flagged `keep`):
  **1 / 81 = 1.2 %** — below the 2 % widening threshold.
- **False-positive rate** (SN paths the reviewer flagged `drop`):
  **4 / 24 = 16.7 %** — independent signal on over-inclusion within
  `quantity`/`geometry`.
- **Borderline rate** on non-SN side: 4 / 81 = 4.9 %, all representation
  (metric-tensor components on GGD grids).

## Recommendation

### ✳️ Do not widen `SN_SOURCE_CATEGORIES`

1.2 % FN is well below the 2 % trigger.  The single `keep` call — the
plasma-boundary outline `equilibrium/time_slice/boundary_separatrix/outline/r`
— is already a well-understood geometric curve that *does* appear in
the Standard Name draft set under other paths (e.g. LCFS / separatrix
outline coordinates live in the `geometry` category for other IDSs).
Its `coordinate` classification looks like an isolated mis-categorisation
rather than a systematic gap.

### ✳️ Open follow-up: false-positive cleanup inside `quantity`

4 of 24 sampled `quantity`/`geometry` paths do **not** look like true
SN candidates:

1. `equilibrium/.../ggd/grid/space/objects_per_dimension/object/geometry_2d`
   — GGD container node (mislabelled `quantity`).
2. `interferometer/channel/wavelength/fringe_jump_correction_times`
   — processing timestamps (provenance).
3. `thomson_scattering/equilibrium_id/data_entry/pulse`
   — shot number (provenance/identifier).
4. `wall/description_2d/mobile/unit/outline/z`
   — bare geometric coordinate (like `coordinate` paths).

These are a separate problem — the DD `node_category` classifier has
some mis-labels that let non-measurables into the SN candidate set.
They are dropped downstream by the composition filter (or by the
reviewer), but they waste LLM budget.  Tracking: **spawn plan-33-style
item for `quantity` node-category clean-up, not in scope for plan 32**.

### ✳️ Representation / metric-tensor borderlines

Four GGD metric-tensor entries were flagged `borderline`.  These are
physically meaningful but computationally derived; excluding them is
consistent with current DD conventions and they are recoverable at
analysis time from the grid geometry.  No action.

## Exit status

Plan 32 Phase 1 **closed** — retain existing `SN_SOURCE_CATEGORIES`.
Follow-up ticket recommended for `quantity` false-positive cleanup.
