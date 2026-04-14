# Search Quality: Ranking and Evaluation Alignment

> Priority: P3 — Search works but ranking quality has known gaps.

## Problem

The DD search pipeline functions correctly and already surfaces cluster output
via the MCP wrapper's parallel `search_dd_clusters` call. However, rubber-duck
review with live tool testing found that **ranking quality** and **evaluation
alignment** are the real gaps — not output formatting.

Key finding: canonical concept paths (e.g., `core_profiles/profiles_1d/electrons/temperature`)
sometimes rank below diagnostic structures and containers for plain physics queries
like "electron temperature".

## Verified Current State (rubber-duck confirmed)

### Already Done (do NOT reimplement)
- `SearchHit.cluster_labels` field exists in `search_strategy.py`
- `SearchHit.see_also` field exists
- Formatter renders both `Clusters:` and `See also:` sections
- Tests exist for formatter/model fields (`test_cluster_output.py`)
- Public `search_dd_paths` MCP tool appends full `## IMAS Clusters` section via parallel cluster search
- `generate_expected_paths.py` exists
- Expected-path cache fixture exists in `conftest.py`
- Dimension comparison harness exists
- DoE harness skeleton exists

### NOT Done in Core Search
- `GraphSearchTool.search_dd_paths()` does NOT populate `cluster_labels` or `see_also` per hit
- The MCP wrapper compensates by running a parallel cluster search at query level
- Evaluation harness does NOT use expanded expected paths in actual benchmarks
- Cache freshness validation is incomplete (no graph version key comparison)
- DoE harness does NOT match production scoring pipeline

### Explicitly Deferred
- **Dimension upgrade (Phase 3 of old plan)**: Not justified until ranking and evaluation are fixed. Investigation only after this plan is complete.
- **DoE weight optimization**: Blocked until harness matches production scoring
- **Benchmark corpus expansion to 200+ queries**: After evaluation alignment

## Phase 1: Canonical Concept Reranking

**Goal**: Physics concept paths rank above diagnostic/container/metadata paths for concept-level queries.

### 1A: Leaf concept preference
Add a reranking boost for paths that represent leaf physics concepts (temperature, density, current) over container structures and diagnostic metadata (validity, fit, coefficients).

### 1B: Canonical IDS preference
For concept queries without IDS qualification, boost paths in canonical IDS:
- `core_profiles` for Te, Ti, ne, q, etc.
- `equilibrium` for psi, ip, boundary, magnetic axis
- `magnetics` for B-field measurements
- `summary` for scalar summary values

### 1C: Accessor de-ranking
Paths ending in `/data`, `/value`, `/time`, `/validity`, `/fit` should rank below their parent concept path unless the query explicitly targets them.

**Files**: `imas_codex/tools/graph_search.py` (scoring section)

## Phase 2: Evaluation Alignment

**Goal**: Benchmark metrics reflect actual search quality by using graph-derived expected paths.

### 2A: Integrate expected path generator into benchmarks
Wire `generate_expected_paths.py` output into `evaluate_config()` and `test_search_benchmarks.py`.

### 2B: Fix cache freshness validation
Add graph version key (DD version + node count hash) comparison before reusing cached expected paths.

### 2C: Ground truth regression tests
Add the 20 A/B test queries from the old plan to guard against regressions while improving ranking.

**Files**: `tests/search/` (benchmark data, evaluation, conftest)

## Phase 3: Core Search Cluster Enrichment (Optional)

**Decision point**: The MCP wrapper already provides cluster output at query level. Per-hit cluster labels in core `GraphSearchTool` results are only needed if:
- Downstream consumers use `GraphSearchTool` directly (not via MCP)
- Structured per-hit output is needed for agent pipelines

### If needed:
Add `IN_CLUSTER` traversal to the enrichment query in `graph_search.py:571-597`:
```cypher
OPTIONAL MATCH (path)-[:IN_CLUSTER]->(cl:IMASSemanticCluster)
  WHERE cl.scope = 'global'
```
Return `collect(DISTINCT cl.label) AS cluster_labels` per hit.

### If not needed:
Remove `cluster_labels` and `see_also` from `SearchHit` model to avoid confusion about unpopulated fields, or document them as "populated by MCP wrapper only".

## Phase 4: Fuzzy Search / Typo Handling

**Goal**: Misspelled queries return useful results instead of nothing.

### 4A: Lucene fuzzy matching
Add `~2` fuzzy operator support for short queries in the BM25 channel:
```
temperture~2 → finds "temperature"
```

### 4B: Integration with query analysis
Detect likely misspellings (short queries with no exact BM25 hits) and auto-retry with fuzzy matching.

**Files**: `imas_codex/tools/graph_search.py` (BM25 query construction)

## Phase Dependencies

```
Phase 1 (reranking)          → independent, highest impact
Phase 2 (evaluation)         → independent, can parallel with Phase 1
Phase 3 (core enrichment)    → independent, decision point
Phase 4 (fuzzy)              → independent, can parallel
```

## Documentation Updates

| Target | Update |
|--------|--------|
| `AGENTS.md` | Update search quality baselines after improvements |
| `plans/README.md` | Update status |
