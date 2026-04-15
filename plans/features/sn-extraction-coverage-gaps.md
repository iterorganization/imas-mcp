# SN Extraction Coverage Gaps

## Problem Statement

The `sn generate` pipeline has several extraction-layer bugs that prevent it from
achieving full coverage of the IMAS Data Dictionary. Empirical analysis of the
live graph reveals:

- **11,441** dynamic IMASNode paths with descriptions are eligible for standard names
- **342** StandardName nodes exist (3% coverage)
- **1,042** unclustered quantity-type paths lack standard names
- Each run with default settings processes only **~251 unique paths** (2.2% of pool)
- Repeated runs **stall on the same front slice** — they re-query the same rows, filter
  them out as already named, and never advance

Root causes are a LIMIT-on-rows bug, a missing Cypher field (`cluster_scope`), and
absent pre-LIMIT filtering of already-named nodes.

## Gap Inventory

### Gap 1: LIMIT applies to expanded rows, not distinct paths — CRITICAL

**Location:** `imas_codex/standard_names/sources/dd.py`, `_ENRICHED_QUERY`

The extraction query uses multiple `OPTIONAL MATCH` clauses (clusters, coordinates,
parents) that fan out rows. A path in 3 clusters produces 3 rows. `LIMIT $limit`
(default 500) is applied to expanded rows, not distinct paths.

**Measured impact:**
- Row fan-out factor: **1.88×** (21,497 rows for 11,441 paths)
- `LIMIT 500` → **251 unique paths** (2.2% of pool)
- To process all paths would require `--limit 21500`+

**Fix:** Move LIMIT to a CTE that selects distinct paths first, then join enrichment:

```cypher
MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
WHERE <filters>
WITH DISTINCT n, ids
ORDER BY ids.id, n.id
LIMIT $limit
// THEN join clusters, units, etc.
OPTIONAL MATCH (n)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
...
```

### Gap 2: Repeated runs stall — unnamed filter is post-LIMIT — CRITICAL

**Location:** `imas_codex/standard_names/workers.py:76-80`, `sources/dd.py:124-142`

The extraction query's WHERE clause does not exclude already-named nodes. The
unnamed filter (`get_named_source_ids()`) runs in the extract worker and filters
batch items AFTER the query returns. With deterministic `ORDER BY ids.id, n.id`:

1. Run 1 gets rows for paths A₁..A₂₅₁, composes names
2. Run 2 queries the **same** rows A₁..A₂₅₁ again (deterministic ORDER BY)
3. Worker filters them out as already named → 0 items to compose
4. Paths A₂₅₂+ never enter the extraction window

**Fix:** Add a pre-LIMIT unnamed exclusion to the Cypher when `--force` is not set:

```cypher
AND NOT EXISTS { MATCH (n)-[:HAS_STANDARD_NAME]->(:StandardName) }
```

### Gap 3: `cluster_scope` not returned in extraction query — MODERATE-HIGH

**Location:** `imas_codex/standard_names/sources/dd.py:19-53`

The enriched query returns `c.label`, `c.id`, `c.description` but NOT `c.scope`.
`enrich_paths()` builds cluster dicts with `"scope": row.get("cluster_scope") or ""`
which always resolves to `""`.

`select_primary_cluster()` uses a scope priority map (IDS=0, domain=1, global=2)
but since scope is always empty, all clusters get `_DEFAULT_SCOPE_RANK=3` and
tie-break degrades to similarity_score/label order.

**Impact:**
- Primary cluster selection is non-deterministic for multi-cluster paths
- Batching coherence suffers — paths may land in wrong conceptual group
- Runtime behavior diverges from test expectations

**Fix:** Add `c.scope AS cluster_scope` to both `_ENRICHED_QUERY` and
`_TARGETED_PATH_QUERY` RETURN clauses.

### Gap 4: Grouping cluster policy contradicts global batching goal — MODERATE

**Location:** `imas_codex/standard_names/enrichment.py:78-93`

`select_grouping_cluster()` just delegates to `select_primary_cluster()` which
uses IDS-first priority. Once Gap 3 is fixed and scope works correctly, grouping
will prefer IDS-scope clusters — fragmenting cross-IDS paths that share a
global/domain cluster into separate per-IDS batches.

The docstring says "global/domain preferred for batch formation" but the code
does the opposite. Two distinct policies are needed:

- **Primary cluster** (per-item context): IDS > domain > global (most specific)
- **Grouping cluster** (batch formation): global > domain > IDS (widest scope)

**Fix:** Implement reversed scope priority in `select_grouping_cluster()`.

### Gap 5: Unclustered paths — functional but low quality — LOW-MODERATE

**Status:** 1,154 unclustered paths (10.1% of pool), 1,042 quantity-type without names.

The fallback `unclustered/{parent_path}/{unit}` grouping works but:
- 21 paths have no `HAS_PARENT` → `parent_path="root"`, creating an incoherent bag
- Grouping by parent is structural, not semantic — siblings may be unrelated concepts
- Batch context lacks cluster description, cross-IDS siblings, concept summary

**Fix:** For unclustered paths, use IDS + parent_path grouping. For rootless paths,
use IDS as the grouping key. Consider enriching batch context with parent description.

### Gap 6: Observability reports rows as paths — LOW

**Location:** `imas_codex/standard_names/sources/dd.py:151`

`_status(f"found {len(results)} paths, resolving units…")` reports row count, not
unique path count, since `results` may contain multiple rows per path from cluster
fan-out.

**Fix:** Report `len(set(r['path'] for r in results))` as the path count.

## Implementation Plan

### Phase 1: Fix extraction query (Gaps 1, 2, 3, 6)

**Files to modify:**
- `imas_codex/standard_names/sources/dd.py`

**Changes:**

1. **Restructure `_ENRICHED_QUERY`** — apply LIMIT to distinct paths FIRST, then
   expand with OPTIONAL MATCH for clusters, units, parents, coordinates:

   ```python
   _ENRICHED_QUERY = """
   MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
   WHERE {where_clause}
   WITH DISTINCT n, ids
   ORDER BY ids.id, n.id
   LIMIT $limit
   OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
   OPTIONAL MATCH (n)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
   OPTIONAL MATCH (n)-[:HAS_PARENT]->(parent:IMASNode)
   OPTIONAL MATCH (n)-[:HAS_COORDINATE]->(coord:IMASNode)
   OPTIONAL MATCH (coord)-[:HAS_UNIT]->(cu:Unit)
   RETURN n.id AS path, ...
          c.scope AS cluster_scope,
          ...
   """
   ```

2. **Add unnamed exclusion to where_parts** in `extract_dd_candidates()` when
   force is not set. Pass a `force` parameter to the function:

   ```python
   if not force:
       where_parts.append(
           "NOT EXISTS { MATCH (n)-[:HAS_STANDARD_NAME]->(:StandardName) }"
       )
   ```

3. **Add `c.scope AS cluster_scope`** to `_ENRICHED_QUERY` RETURN clause.

4. **Add `c.scope AS cluster_scope`** to `_TARGETED_PATH_QUERY` RETURN clause.

5. **Fix observability** — compute unique path count for status messages.

### Phase 2: Fix grouping cluster policy (Gap 4)

**Files to modify:**
- `imas_codex/standard_names/enrichment.py`

**Changes:**

1. **Implement `select_grouping_cluster()`** with reversed scope priority
   (global=0, domain=1, IDS=2) so cross-IDS paths sharing a global cluster
   land in the same batch.

2. **Update docstrings** to document the two-policy design.

### Phase 3: Improve unclustered path handling (Gap 5)

**Files to modify:**
- `imas_codex/standard_names/enrichment.py`

**Changes:**

1. **Enhance unclustered grouping key** — use `unclustered/{ids_name}/{parent_path}/{unit}`
   to keep IDS context for LLM prompt coherence.

2. **Enrich unclustered batch context** — include parent description, IDS description,
   and a note about the path being unclustered for the LLM.

3. **Handle rootless paths** — use IDS name as fallback group key when `parent_path`
   is None.

### Phase 4: Propagate `force` flag to extraction (Gap 2 support)

**Files to modify:**
- `imas_codex/standard_names/sources/dd.py` (`extract_dd_candidates` signature)
- `imas_codex/standard_names/workers.py` (pass `force` from state)

**Changes:**

1. Add `force: bool = False` parameter to `extract_dd_candidates()`.
2. Pass `state.force` when calling `extract_dd_candidates()` from extract worker.
3. Only add the unnamed exclusion when `force=False`.

### Phase 5: Tests

**Files to modify/create:**
- `tests/standard_names/test_dd_extraction.py` (or existing test file)
- `tests/standard_names/test_enrichment.py` (update existing)

**Test cases:**

1. **LIMIT-on-paths test**: Mock graph query returning fan-out rows, verify
   the number of unique paths matches the limit, not the row count.

2. **Stall prevention test**: Verify that already-named paths are excluded
   from the query when force=False, allowing subsequent runs to advance.

3. **cluster_scope propagation test**: Verify that cluster scope flows through
   enrichment and affects primary/grouping cluster selection correctly.

4. **Grouping cluster policy test**: Verify global-scope clusters are preferred
   for batch grouping while IDS-scope clusters are preferred for per-item context.

5. **Unclustered IDS grouping test**: Verify unclustered paths are grouped by
   IDS + parent + unit, not just parent + unit.

6. **Observability test**: Verify status messages report unique path counts.

### Phase 6: Documentation Updates

| Target | Update |
|--------|--------|
| `AGENTS.md` | Update SN pipeline section to document the fix and the two cluster-selection policies |
| `plans/README.md` | Add this plan |

## Dependencies

- Phase 2 depends on Phase 1 (scope must be returned before grouping policy can use it)
- Phase 3 is independent
- Phase 4 depends on Phase 1 (query structure must be restructured first)
- Phase 5 depends on Phases 1-4
- Phase 6 depends on all phases

## Risk Assessment

- **Phase 1** is the highest-value fix — resolves both critical gaps with a single
  query restructure. Low risk since it's a Cypher-level change with no Python logic change.
- **Phase 2** has moderate risk — changing grouping policy may shift batch boundaries,
  producing different standard names for the same paths. This is acceptable since
  current behavior is already non-deterministic due to the scope bug.
- **Phase 4** changes the extraction query behavior (excluding named paths) which
  could interact with `--force` and `--from-model` modes. Careful testing needed.
