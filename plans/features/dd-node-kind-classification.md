# DD Node Kind Classification

## Problem

The current `node_category` field on `IMASNode` has three values: `data`, `error`, `metadata`. The `data` bucket is too coarse — it lumps together 20,037 nodes including:

- **Physics quantities** (temperature, current, flux) — should be embedded + SN-extracted
- **Coordinates** (time arrays, rho_tor_norm) — should NOT be SN-extracted
- **Structural artifacts** (data arrays under STRUCTURE parents, validity flags) — should NOT be embedded or SN-extracted
- **Identifier fields** (coordinate_identifier, grid_type) — should NOT be embedded or SN-extracted

This causes two problems:
1. **Vector space pollution**: ~13,000 non-quantity nodes are embedded alongside ~7,000 real physics quantities, degrading semantic search precision
2. **SN extraction gaps**: The `node_type='dynamic'` filter (a proxy for "is physics data") blocks legitimate quantities in magnetics and other IDSs where STRUCTURE parents have `node_type='none'` or `'static'`

The SN classifier (13 rules) partially compensates, but classification must happen at the DD level, be deterministic, and be directly queryable.

## Design Decisions

### Expand NodeCategory (not add node_kind)

Backwards compatibility is not a concern. One field with clear semantics is better than two overlapping fields. The expanded enum replaces the monolithic `data` with more specific values.

### Cannot Share Enum with StandardName Kind

They classify different things:
- **StandardNameKind** (scalar/vector/metadata): Mathematical nature of a named quantity
- **NodeCategory** (proposed 6 values): Structural role in the data dictionary

The `metadata` label appears in both but means different things. Orthogonal taxonomies — sharing an enum would be misleading.

## Proposed NodeCategory Expansion

```yaml
NodeCategory:
  permissible_values:
    quantity:
      description: Independent physics quantity. Embedded + SN-extracted.
    coordinate:
      description: Independent variable (time, space, flux coords).
        NOT embedded, NOT SN-extracted. Searchable via properties.
    structural:
      description: Storage artifact (/data arrays, /validity flags,
        containers without unit). Neither embedded nor SN-extracted.
    identifier:
      description: Typed descriptor with HAS_IDENTIFIER_SCHEMA
        relationship. Neither embedded nor SN-extracted.
    error:
      description: Uncertainty bounds (_error_*). Unchanged.
    metadata:
      description: Bookkeeping subtrees (ids_properties/*, code/*). Unchanged.
```

### Pipeline Participation Matrix

| NodeCategory | Enriched | Embedded | SN Extracted | Searchable |
|---|---|---|---|---|
| `quantity` | ✓ | ✓ | ✓ | ✓ |
| `coordinate` | ✓ | ✗ | ✗ | ✓ (property) |
| `structural` | ✗ | ✗ | ✗ | ✗ |
| `identifier` | ✗ | ✗ | ✗ | ✗ |
| `error` | ✗ | ✗ | ✗ | ✗ |
| `metadata` | ✗ | ✗ | ✗ | ✗ |

**Coordinates NOT embedded** (RD consensus): Only `quantity` nodes in the vector space. Coordinates searchable via property queries. If search quality demands coordinate embeddings later, add to a separate vector index.

## Two-Pass Classification Architecture

Classification happens in two passes because some signals require graph relationships created after node creation.

### Pass 1 — Build-Time (XML signals only)

Applied in `_classify_node()` with expanded signature: `_classify_node(path_id, name, *, data_type, unit, parent_data_type)`.

```
1.  Error suffix (_error_upper, _error_lower, _error_index)     → error
2.  Metadata subtree (ids_properties/*, code/*)                   → metadata
3.  Generic metadata leaf (description, name, comment, source,
    provider at depth ≥ 3)                                        → metadata
4.  data_type = STR_0D or STR_1D                                   → metadata
5.  last_segment = 'time' AND data_type in FLT_*                  → coordinate
6.  last_segment in (validity, validity_timed)                     → structural
7.  last_segment = 'data' AND parent_type in STRUCTURE types       → structural
8.  data_type in (INT_0D, INT_1D) AND structural keywords
    in segment or description                                      → structural
9.  data_type in PHYSICS_LEAF_TYPES AND unit is physical           → quantity
10. data_type in PHYSICS_LEAF_TYPES AND unit is None/'-'
    - Known coordinate fallback set → coordinate
    - Default: quantity (dimensionless physics quantity)
11. STRUCTURE/STRUCT_ARRAY with unit                               → quantity (provisional)
12. STRUCTURE/STRUCT_ARRAY without unit                            → structural
13. INT_0D without unit, no structural keywords                   → quantity
14. Everything else                                                → structural
```

### Pass 2 — Post-Build Relational (graph relationships)

Runs after all nodes and relationships exist. Overrides Pass 1 where relationships provide stronger signal.

```
R1. Node has HAS_IDENTIFIER_SCHEMA relationship                   → identifier
R2. Node is a HAS_COORDINATE target (referenced by other nodes)   → coordinate
R3. STRUCTURE with unit: must have data/time/validity children
    to confirm signal_value pattern                                → quantity (confirmed)
    If no children evidence                                        → structural (demoted)
```

### Coordinate Detection (Three Layers)

1. **Relational (Pass 2, R2):** Node is target of `HAS_COORDINATE` → definitive
2. **Lexical (Pass 1, rule 5):** `last_segment == 'time'` with FLT type → definitive
3. **Fallback set (Pass 1, rule 10):** Known coordinate segment names — ONLY for unitless FLT nodes:
   ```python
   COORDINATE_SEGMENTS = frozenset({
       "rho_tor_norm", "rho_pol_norm", "psi_norm", "psi",
       "phi", "theta", "r", "z", "rho_tor",
       "zeta", "chi", "s", "rho",
   })
   ```

## Centralized Category Constants

**File**: `imas_codex/core/node_categories.py` (new)

```python
EMBEDDABLE_CATEGORIES: frozenset[str] = frozenset({"quantity"})
SEARCHABLE_CATEGORIES: frozenset[str] = frozenset({"quantity", "coordinate"})
SN_SOURCE_CATEGORIES: frozenset[str] = frozenset({"quantity"})
ENRICHABLE_CATEGORIES: frozenset[str] = frozenset({"quantity", "coordinate"})
```

All consumers import from this module — no inline `'data'` literals anywhere.

## Actual Graph Counts (DD v4.1.1)

| Signal | Count |
|---|---|
| Total data nodes | 20,037 |
| — dynamic | 11,441 |
| — none | 4,646 |
| — static | 2,295 |
| — constant | 1,655 |
| HAS_IDENTIFIER_SCHEMA | 244 |
| HAS_COORDINATE targets | 1,467 |
| STR_0D/STR_1D | 689 |
| /time leaves | 695 |
| /validity + /validity_timed | 254 |
| /data under STRUCTURE | 530 |
| INT structural keywords | 2,514 |
| FLT/CPX with physical unit | 9,956 |
| FLT/CPX without unit | 1,339 |
| STRUCTURE with unit | 1,560 |
| STRUCTURE without unit | 3,204 |
| INT_0D no unit, no structural kw | 431 |

**Predicted distribution after migration:**
- `quantity`: ~11,947 (FLT/CPX+unit 9,956 + STRUCTURE+unit ~1,560 + INT_0D dimensionless 431)
- `coordinate`: ~2,162 (time 695 + HAS_COORDINATE targets 1,467 — overlap ~)
- `structural`: ~4,998 (STRUCT no unit 3,204 + /data 530 + /validity 254 + INT structural 2,514 — overlaps with identifier/coord)
- `identifier`: ~244 (HAS_IDENTIFIER_SCHEMA)
- `metadata (reclassified)`: ~689 (STR_0D/STR_1D)

## Implementation Plan

### Rollout Ordering (RD consensus)

The rollout must avoid a state where consumer code expects new categories but graph data still has `data`. The ordering is:

1. **Phase A**: Schema + build pipeline + shared classifier module (new builds produce new categories)
2. **Phase B**: Migration (existing graph data converted, `data` value eliminated)
3. **Phase C**: Consumer update (safe — all nodes now have new categories)
4. **Phase D**: Schema cleanup (remove deprecated `data` from enum)
5. **Phase E**: Validation + tests
6. **Phase F**: Documentation

**Key constraint**: Schema initially KEEPS `data` as a deprecated value so that pre-migration queries don't fail schema validation. `data` is removed only after migration confirms zero remaining nodes.

### Shared Classifier Module (Single Source of Truth)

**File**: `imas_codex/core/node_classifier.py` (new)

Contains the classification logic used by BOTH the build pipeline and the migration:
- `classify_node_pass1(path, name, *, data_type, unit, parent_data_type) -> str`
- `classify_node_pass2(node_id, *, has_identifier_schema, is_coordinate_target, children_types) -> str | None`
- `COORDINATE_SEGMENTS` fallback set

Both `_classify_node()` in build_dd.py and the migration Cypher generator call this same module, ensuring rule consistency.

### Phase A: Schema + Build Pipeline

**Task 1: Schema Change**
- `imas_codex/schemas/imas_dd.yaml`: Add 4 new values (quantity, coordinate, structural, identifier) to NodeCategory. **Keep `data` temporarily** as deprecated.
- `uv run build-models --force`

**Task 2: Category Constants Module**
- Create `imas_codex/core/node_categories.py`
- Constants initially include `'data'` as transitional alias:
  ```python
  EMBEDDABLE_CATEGORIES = frozenset({"quantity", "data"})  # data is transitional
  ```

**Task 3: Shared Classifier Module**
- Create `imas_codex/core/node_classifier.py`
- Pass 1 rules (14 rules) + Pass 2 rules (3 rules)
- Used by both build pipeline and migration

**Task 4: Update `_classify_node()` in Build Pipeline**
- `imas_codex/graph/build_dd.py`: Delegate to shared classifier
- Update `_batch_create_path_nodes()` call site with full context

**Task 5: Add Pass 2 Relational Classification**
- `imas_codex/graph/build_dd.py`: New `_reclassify_relational()` function
- Runs after all nodes + relationships created
- Uses HAS_IDENTIFIER_SCHEMA, HAS_COORDINATE, children evidence

### Phase B: Migration

**Task 6: Read-Only Classification Preview**
Run full classification as read-only query using shared classifier logic. Compare predicted counts with actual graph counts above.
Add post-preview audit: `FLT_* + no unit + predicted=quantity` → verify these are real dimensionless quantities, not missed coordinates.

**Task 7: Shadow-Property Migration**
1. Write classification to `node_category_new` using shared classifier logic
2. Validate distribution matches preview
3. Verify zero `'data'` values in `node_category_new`
4. Atomic swap: `SET n.node_category = n.node_category_new REMOVE n.node_category_new`

**Task 8: Remove Stale Embeddings**
Batch remove `embedding`, `embedded_at`, `embedding_hash`, `embedding_text` from non-embeddable nodes. Reset status from `embedded` → `built` for reclassified nodes.

**Task 9: CLI Command**
`imas-codex dd migrate-categories --preview` / `--apply`
- New subcommand (not folded into `dd build`)
- Preview shows predicted distribution without mutation
- Apply runs shadow-property migration

### Phase C: Consumer Update (after migration)

**Task 10: Audit + Update All `node_category='data'` Consumers**
Full grep-driven audit. Known locations: `sources/dd.py`, `build_dd.py`, `dd_graph_ops.py`, `dd_workers.py`, `dd_ids_enrichment.py`, `tools/graph_search.py`, `cli/imas_dd.py`, `ids/tools.py`, all tests.
Every consumer → import from `node_categories.py`.

**Task 11: Update Embedding Pipeline**
- `phase_embed()`: `node_category IN $embeddable` using `EMBEDDABLE_CATEGORIES`

**Task 12: Update Enrichment Pipeline**
- Enrichment claim queries in `dd_graph_ops.py`/`dd_workers.py`: update from `node_category='data'` to `node_category IN $enrichable` using `ENRICHABLE_CATEGORIES`
- Define terminal status for `coordinate` nodes: enrichable but never embedded → terminal status is `enriched` (not `embedded`)

**Task 13: Update SN Extraction Query**
- `sources/dd.py`: `node_category IN $sn_categories` using `SN_SOURCE_CATEGORIES`
- **Keep** `node_type = 'dynamic'` for v1 (relax after non-dynamic quantity audit)

**Task 14: Update Search/MCP Tools**
- All tools → use `SEARCHABLE_CATEGORIES`

### Phase D: Schema Cleanup

**Task 15: Remove `data` from Schema**
- After verifying zero `data` nodes remain, remove `data` from NodeCategory enum
- Remove `'data'` from transitional category constant sets
- `uv run build-models --force`

### Phase E: Validation + Tests

**Task 16: Post-Migration Verification**
```cypher
MATCH (n:IMASNode) RETURN n.node_category, count(*) ORDER BY count(*) DESC
MATCH (n:IMASNode {node_category: 'data'}) RETURN count(n) -- must be 0
MATCH (n:IMASNode) WHERE n.node_category = 'quantity' AND n.node_type <> 'dynamic'
  RETURN n.node_type, count(*) ORDER BY count(*) DESC
MATCH (n:IMASNode) WHERE NOT (n.node_category IN ['quantity']) AND n.embedding IS NOT NULL
  RETURN count(n) -- must be 0 after stale embedding cleanup
```

**Task 17: Update Tests**
- Unit tests for shared classifier (all 17 rules)
- `test_node_category.py`: New categories
- `test_classifier.py`: Verify on quantity-filtered nodes
- Migration idempotence test
- Update all tests asserting old enum values

### Phase F: Documentation

**Task 18: Documentation Updates**
- `AGENTS.md`: NodeCategory expansion, pipeline matrix, centralized constants
- `docs/architecture/standard-names.md`: Extraction filter changes
- `plans/README.md`: Plan entry
- Schema reference: auto-generated

## SN node_type Filter Strategy

The `node_type='dynamic'` filter is **kept for v1**. Removing requires:
1. Post-migration audit: `WHERE node_category='quantity' AND node_type <> 'dynamic'`
2. Manual inspection of non-dynamic quantity candidates
3. If safe, relax in follow-up task

Magnetics STRUCTURE parents (node_type='none') are unblocked only after this audit confirms safety.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Partial migration state breaks queries | Shadow-property migration with atomic swap |
| Next DD build reverts changes | Schema + build updated BEFORE migration (Phase A) |
| Missing consumers of `'data'` | Full repo-wide grep audit (Task 5) |
| Coordinate detection misses | Three-layer detection (relational + lexical + fallback) |
| STRUCTURE+unit too broad for quantity | Pass 2 R3: require children evidence |
| Non-dynamic quantities in SN | Keep node_type filter for v1, audit before relaxing |
| Vector space degradation | Embed quantity-only initially |

## Estimated Impact

| Metric | Before | After |
|---|---|---|
| Embedded nodes | 20,037 | ~11,947 (quantity only) |
| Vector space noise | ~8,090 non-quantity | 0 |
| SN extractable (v1) | ~11,441 (data+dynamic) | ~11,441 (quantity+dynamic) |
| SN extractable (v2, filter relaxed) | — | ~11,947 (all quantity) |
| SN classifier FP rate | ~50% → skip/metadata | <10% |
