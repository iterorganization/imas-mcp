# Phase 0 — StandardName Field Audit Report

**Plan**: `plans/features/standard-names/35-catalog-schema-redesign.md`
**Date**: 2025-07-15
**Total StandardName nodes**: 925
**Machine-readable companion**: `field-audit.json`

---

## 1. Potentially-Drop Fields (Legacy Grammar)

These are the six legacy grammar fields that predate the `grammar_*` slot
system. The plan says: *"Drop from schema ONLY if 0/925 populated."*

| Field | Populated | Superseded by | Recommendation |
|---|---|---|---|
| `physical_base` | 0/925 (0%) | `grammar_physical_base` (907/925) | **DROP** |
| `subject` | 0/925 (0%) | `grammar_subject` (280/925) | **DROP** |
| `component` | 0/925 (0%) | `grammar_component` (151/925) | **DROP** |
| `coordinate` | 0/925 (0%) | `grammar_coordinate` (83/925) | **DROP** |
| `position` | 0/925 (0%) | `grammar_position` (51/925) | **DROP** |
| `process` | 0/925 (0%) | `grammar_process` (19/925) | **DROP** |

**Verdict: All six legacy grammar fields are 0/925 populated and fully
superseded by their `grammar_*` counterparts. Safe to drop from schema in
Phase 2.**

---

## 2. Dead-but-Retained Fields (Schema Contract)

The plan explicitly says these must be kept in the graph schema even though
they are unpopulated today. They represent future extensibility for the
standard-name grammar.

| Field | Populated | Recommendation |
|---|---|---|
| `species` | 0/925 (0%) | **KEEP** (plan mandate) |
| `population` | 0/925 (0%) | **KEEP** (plan mandate) |
| `toroidal_mode` | 0/925 (0%) | **KEEP** (plan mandate) |
| `flux_surface_average` | 0/925 (0%) | **KEEP** (plan mandate) |

---

## 3. Deprecated Secondary Review Fields

These are already marked deprecated in the current schema. The plan lists
them for drop.

| Field | Populated | Recommendation |
|---|---|---|
| `reviewer_model_secondary` | 31/925 (3.4%) | **DROP** |
| `reviewer_score_secondary` | 31/925 (3.4%) | **DROP** |
| `reviewer_scores_secondary` | 31/925 (3.4%) | **DROP** |
| `reviewer_disagreement` | 31/925 (3.4%) | **DROP** |

All four fields are present on exactly the same 31 nodes (100% overlap
confirmed). The data is expendable secondary-review output from an early
review pass. The primary review fields (`reviewer_score`, `reviewer_model`,
`reviewer_scores`, `reviewer_comments`, `review_disagreement` — all at
702/925) are unaffected.

**Phase 2 migration Cypher** should `REMOVE` these four properties from all
nodes.

---

## 4. Status / Lifecycle Fields

| Field | Populated | Notes |
|---|---|---|
| `review_status` | 925/925 (100%) | **Rename → `pipeline_status`** in Phase 2. Values: enriched (688), named (236), drafted (1). |
| `status` | 0/925 (0%) | **New field** — to be added in Phase 2, backfilled to `'draft'`. |
| `deprecates` | 0/925 (0%) | **New field** — to be added in Phase 2. |
| `superseded_by` | 0/925 (0%) | **New field** — to be added in Phase 2. |
| `origin` | 0/925 (0%) | **New field** — to be added in Phase 2, backfilled to `'pipeline'`. |
| `cocos_transformation_type` | 521/925 (56%) | **Keep**. Values: one_like (511), psi_like (4), b0_like (3), q_like (1), grid_type_tensor_contravariant_like (1), ip_like (1). |

---

## 5. Graph-Only Provenance (Sanity Check)

The plan designates these as graph-only fields that should never appear in
catalog YAML. Confirming they are populated:

| Field / Relationship | Populated | Status |
|---|---|---|
| `source_paths` (property) | 924/925 (99.9%) | ✅ PASS — 1 node missing (`toroidal_plasma_current_due_to_non_inductive_current_drive`) |
| `source_types` (property) | 925/925 (100%) | ✅ PASS — all values are `['dd']` |
| `HAS_COCOS` (relationship) | 867/925 (93.7%) | ✅ PASS |
| `HAS_UNIT` (relationship) | 925/925 (100%) | ✅ PASS (note: `atomic_number` has 2 HAS_UNIT rels) |
| `SOURCE_DD_PATH` (relationship) | 0 rels | ℹ️ INFO — not yet created; plan mentions as future provenance mechanism |
| `SOURCE_SIGNAL` (relationship) | 0 rels | ℹ️ INFO — not yet created; no signal-sourced names yet |

---

## 6. Relationship Census

| Direction | Relationship | Count |
|---|---|---|
| Outgoing | `HAS_UNIT` | 926 |
| Outgoing | `HAS_COCOS` | 867 |
| Outgoing | `HAS_SEGMENT` | 214 |
| Incoming | `PRODUCED_NAME` (from StandardNameSource) | 1,812 |
| Incoming | `HAS_STANDARD_NAME` (from IMASNode) | 1,738 |
| Incoming | `REVIEWS` (from Review) | 738 |
| Incoming | `EVIDENCED_BY` (from PromotionCandidate) | 57 |

---

## 7. Surprising Findings

1. **`atomic_number` has 2 `HAS_UNIT` relationships** (926 rels for 925
   nodes). This is a data quality issue to investigate separately — may be
   a duplicate unit assignment.

2. **`review_tier` (702/925)** appears in the graph but is not listed in
   the plan's disposition table. It should receive an explicit keep/drop
   decision before Phase 2 finalises the schema.

3. **`reviewer_disagreement` vs `review_disagreement`** — two similarly
   named fields with very different semantics. `reviewer_disagreement`
   (31/925) is secondary-review data slated for drop.
   `review_disagreement` (702/925) is the primary QA metric and should be
   kept. Phase 2 migration must be careful not to confuse these.

4. **All `source_types` values are `['dd']`** — the graph contains no
   signal-sourced names yet. The `SOURCE_DD_PATH` and `SOURCE_SIGNAL`
   relationships mentioned in the plan do not exist yet; `source_paths`
   (string list property) currently carries this provenance.

5. **`links` is 0/925** — no standard names have link data yet. This is
   expected (link resolution hasn't run) but worth noting since the plan
   includes pipeline protection for the `links` field.

---

## 8. Summary for Phase 2 Executor

### Fields to DROP from schema (and REMOVE from graph)

```
physical_base
subject
component
coordinate
position
process
reviewer_model_secondary
reviewer_score_secondary
reviewer_scores_secondary
reviewer_disagreement
```

**Total: 10 fields.** The first 6 are legacy grammar (0/925). The last 4
are deprecated secondary review (31/925 — expendable).

### Fields to KEEP in schema despite 0/925 population

```
species
population
toroidal_mode
flux_surface_average
```

These are retained per explicit plan mandate as schema contract for future
grammar extensibility.

### Phase 2 migration Cypher (recommended)

```cypher
// Drop legacy grammar fields (0/925 populated — no data loss)
MATCH (sn:StandardName)
REMOVE sn.physical_base, sn.subject, sn.component,
       sn.coordinate, sn.position, sn.process

// Drop deprecated secondary review fields (31/925 — expendable)
MATCH (sn:StandardName)
REMOVE sn.reviewer_model_secondary, sn.reviewer_score_secondary,
       sn.reviewer_scores_secondary, sn.reviewer_disagreement
```

### Open questions for Phase 2

1. Should `review_tier` (702/925) be kept or dropped? Not in disposition
   table.
2. Should the duplicate `HAS_UNIT` on `atomic_number` be cleaned up?
3. Should `enrich_batch_id`, `enrich_tokens`, `vocab_gap_detail` (all
   0/925) be pruned from schema as never-used pipeline metadata?
