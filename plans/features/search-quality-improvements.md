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
- Metadata paths already excluded (`node_category = 'data'` filter in all queries)
- Path segment tiebreaker exists (1.5% multiplicative boost per exact segment match)

### NOT Done in Core Search
- `GraphSearchTool.search_dd_paths()` does NOT populate `cluster_labels` or `see_also` per hit
- The MCP wrapper compensates by running a parallel cluster search at query level
- Evaluation harness does NOT use expanded expected paths in actual benchmarks
- Cache freshness validation is incomplete (no graph version key comparison)
- DoE harness does NOT match production scoring pipeline

### Explicitly Deferred
- **Dimension upgrade**: Not justified until ranking and evaluation are fixed
- **DoE weight optimization**: Blocked until harness matches production scoring
- **Benchmark corpus expansion to 200+ queries**: After evaluation alignment

## Phase 1: Careful Reranking

> ⚠️ **CAUTION**: Metadata paths are already excluded via `node_category='data'`.
> The existing path segment tiebreaker (1.5% per match) is conservative and correct.
> Any new boosts must be carefully tested to avoid distorting results.

**Goal**: Leaf physics concept paths rank above container/accessor paths for
concept-level queries, without over-boosting or distorting non-concept queries.

### 1A: Accessor de-ranking (safest, highest signal)
Paths ending in `/data`, `/value`, `/time`, `/validity`, `/fit`, `/coefficients`
should rank below their parent concept path unless the query explicitly targets them.

Implementation: small multiplicative penalty (e.g., 0.95×) for terminal accessor
segments. This is safe because accessor paths are almost never the user's intent
for concept queries, and the penalty is small enough to be overridden by strong
text/vector matches.

**Files**: `imas_codex/tools/graph_search.py` (scoring section, after line 362)

### 1B: IDS preference for unqualified queries (careful, medium impact)
For concept queries without IDS qualification, apply a small boost for canonical IDS:
- `core_profiles` for Te, Ti, ne, q, etc.
- `equilibrium` for psi, ip, boundary, magnetic axis
- `magnetics` for B-field measurements
- `summary` for scalar summary values

> ⚠️ This must be a **soft preference** (e.g., 1.03× boost), NOT a hard filter.
> The boost should only apply when the query has no explicit IDS scope.
> Over-boosting risks hiding valid results in non-canonical IDS (edge_profiles,
> transport_solver_numerics, etc.).

Implementation: maintain a small dict of `{concept_keyword: preferred_ids}`.
When query words match a concept keyword AND a result is from the preferred IDS,
apply a 3% boost. This is enough to break ties without distorting strong matches.

**Files**: `imas_codex/tools/graph_search.py` (scoring section)

### 1C: Leaf concept preference (defer — needs investigation)
Boosting "leaf" paths over "structure" paths sounds appealing but is complex:
- What counts as a "leaf concept" vs a "container"?
- Some containers ARE the user's intent (e.g., `profiles_1d` for a profile query)
- The `data_type` filter already excludes STRUCTURE/STRUCT_ARRAY nodes

**Recommendation**: Defer until 1A and 1B are evaluated. The accessor de-ranking
may be sufficient.

## Phase 2: Evaluation Alignment

**Goal**: Benchmark metrics reflect actual search quality by using graph-derived
expected paths. This is prerequisite for any further ranking work.

### 2A: Integrate expected path generator into benchmarks
Wire `generate_expected_paths.py` output into `evaluate_config()` and
`test_search_benchmarks.py`.

### 2B: Fix cache freshness validation
Add graph version key (DD version + node count hash) comparison before reusing
cached expected paths.

### 2C: Ground truth regression tests
Add 20 A/B test queries to guard against regressions while improving ranking:
- 10 concept queries (electron temperature, plasma current, safety factor, etc.)
- 5 IDS-qualified queries (equilibrium/psi, core_profiles/te)
- 5 edge cases (short terms like "ip", "ne", misspellings)

**Files**: `tests/search/` (benchmark data, evaluation, conftest)

## Phase 3: Core Search Cluster Enrichment (Decision Point)

The MCP wrapper already provides cluster output at query level. Per-hit cluster
labels in core `GraphSearchTool` are only needed if downstream consumers use
`GraphSearchTool` directly (not via MCP).

**Decision**: If MCP is the only consumer, remove `cluster_labels` and `see_also`
from `SearchHit` to avoid confusion about unpopulated fields. If direct consumers
exist, add the `IN_CLUSTER` traversal to enrichment.

## Phase 4: Fuzzy Search / Typo Handling

Add `~2` fuzzy operator support for short queries in the BM25 channel when no
exact BM25 hits are found. Auto-retry with fuzzy matching for likely misspellings.

**Files**: `imas_codex/tools/graph_search.py` (BM25 query construction)

## Phase Dependencies

```
Phase 1A (accessor de-ranking)  → highest impact, safest, do first
Phase 1B (IDS preference)       → after 1A evaluation
Phase 2 (evaluation alignment)  → independent, can parallel with Phase 1
Phase 3 (core enrichment)       → decision point after Phase 1+2
Phase 4 (fuzzy)                 → independent, can parallel
```

## Documentation Updates

| Target | Update |
|--------|--------|
| `AGENTS.md` | Update search quality baselines after improvements |
| `plans/README.md` | Update status |
