# Search Quality: Cluster Output, Auto-Evaluation, and Dim Upgrade

## Problem Statement

Post-embedding-optimization assessment reveals three categories of gaps:

1. **Output gap**: `search_imas_paths()` uses clusters for ranking but doesn't
   surface cluster membership or cross-IDS siblings in results. Users must make
   separate `find_related_imas_paths` calls to discover that their result exists
   in 10+ other IDSs.

2. **Evaluation gap**: Benchmark expected paths are hand-curated (2-5 per query)
   but concepts have 10-36 valid paths. MRR is deflated by false negatives.
   The attempted fix (cluster expansion in benchmarks) patched the metric instead
   of the product — it has been removed.

3. **Vector quality ceiling**: At Matryoshka dim 256, vector MRR is ~0.15 and
   P@1 is ~0.02. BM25 carries the hybrid pipeline. Dim 512/1024 would likely
   recover the vector channel.

### Current Baselines (65-query benchmark, 2026-04)

| Metric | Value | Gate |
|--------|-------|------|
| Overall MRR | 0.390 | 0.35 |
| Abbreviation MRR | 0.232 | 0.20 |
| Vector MRR | ~0.15 | 0.15 |
| BM25 MRR | ~0.40 | 0.15 |
| Vector P@1 | 0.023 | 0.02 |

### Relationship to Existing Plan

`search-recall-and-enrichment.md` covers Phases 1-5 of the search pipeline.
Phases 1.1-1.3, 2.2-2.4, 3.1, 3.3, 4.1, 5.2, 5.3 are implemented.
This plan covers gaps NOT in that plan, plus unfinished items that need
re-scoping based on current measurements.

---

## Phase 1: Cluster Labels in Search Results

**Goal**: Every search hit includes its cluster labels so LLM agents and users
can see which physics concepts a path belongs to without a separate call.

### 1.1 Add IN_CLUSTER to search enrichment query

**File**: `imas_codex/tools/graph_search.py`, lines 571-597

Add to the enrichment query after the existing OPTIONAL MATCHes:

```cypher
OPTIONAL MATCH (path)-[:IN_CLUSTER]->(cl:IMASSemanticCluster)
  WHERE cl.scope = 'global'
```

Return `collect(DISTINCT cl.label) AS cluster_labels` and pass to SearchHit.

### 1.2 Add cluster_labels to SearchHit

**File**: `imas_codex/search/search_strategy.py`

Add `cluster_labels: list[str] = []` field to `SearchHit`.

### 1.3 Format cluster labels in output

**File**: `imas_codex/llm/search_formatters.py`, line ~1187

After keywords, add:
```python
if hit.cluster_labels:
    parts.append(f"  Clusters: {', '.join(hit.cluster_labels)}")
```

### 1.4 Add "See Also" for top results

For the top 3 search hits, append a compact cross-IDS line sourced from
cluster siblings:

```
  See also: ece/channel/t_e, edge_profiles/.../temperature, summary/.../t_e (+7 more)
```

Implementation: After enrichment, for top 3 hits, run a single Cypher query:
```cypher
UNWIND $top_ids AS tid
MATCH (t:IMASNode {id: tid})-[:IN_CLUSTER]->(c:IMASSemanticCluster {scope: 'global'})
  <-[:IN_CLUSTER]-(sibling:IMASNode)
WHERE sibling.ids <> t.ids AND sibling.id <> tid
RETURN tid AS source, collect(DISTINCT sibling.id)[..10] AS siblings
```

### Tests

- Verify `search_imas_paths` returns `cluster_labels` per hit
- Verify "See Also" appears for top 3 results with cluster siblings
- Verify no performance regression (cluster traversal is O(cluster_size))

---

## Phase 2: Auto-Generated Expected Paths

**Goal**: Replace hand-curated expected paths with graph-derived ground truth
that self-maintains across DD rebuilds.

### 2.1 Build expected path generator

Create `tests/search/generate_expected_paths.py` with a function:

```python
def generate_expected_paths(
    query: BenchmarkQuery,
    gc: GraphClient,
) -> set[str]:
    """Query the graph for all valid paths matching a benchmark concept.
    
    Strategy:
    1. Find global clusters matching query text (via cluster label search)
    2. Collect all cluster members
    3. Find paths whose terminal segment matches query abbreviation
    4. Union with hand-curated expected_paths
    5. Filter to concept-level paths (exclude /data, /time, /validity, /value)
    """
```

### 2.2 Cache expected paths as JSON

Add a `conftest.py` fixture that:
1. On first run: generates expected paths for all benchmark queries
2. Caches result to `tests/search/.expected_paths_cache.json` (gitignored)
3. On subsequent runs: loads from cache unless `--regenerate-expected` flag
4. Cache key: DD version + cluster count hash (auto-invalidates on rebuild)

### 2.3 Use cached paths in MRR computation

`QueryResult._compute_rr()` uses the cached expanded set instead of
just `query.expected_paths`. No hand-curated cluster references needed.

### 2.4 Validate cache freshness

Add a test that checks cache staleness:
- Compare cache DD version vs graph DD version
- Warn if cache is >7 days old
- Fail if cache DD version doesn't match graph

### Tests

- Unit test: `generate_expected_paths` for "electron temperature" returns ≥10 paths
- Unit test: cache roundtrip (generate → save → load → compare)
- Integration test: MRR with auto-generated paths vs hand-curated (should be ≥)

---

## Phase 3: Embedding Dimension Evaluation

**Goal**: Measure vector MRR at dim 512 and 1024 to determine if upgrading
from dim 256 is worth the 2-4x storage increase.

### 3.1 Build dimension comparison harness

Add to `tests/search/test_search_evaluation.py`:

```python
class TestDimensionComparison:
    """Compare vector search quality across Matryoshka dimensions."""
    
    @pytest.mark.parametrize("dim", [256, 512, 1024])
    def test_vector_mrr_by_dimension(self, graph_client, dim):
        """Measure raw vector MRR at different truncation dimensions."""
```

This requires embedding queries at each dimension and comparing against
the graph's stored embeddings truncated to the same dimension. Since the
graph stores dim-256 embeddings, we'd need to:
1. Re-embed a sample of ~100 nodes at dim 512 and 1024 (temporary)
2. Embed benchmark queries at each dimension
3. Compute cosine similarity directly (no index needed for 100 nodes)
4. Report MRR at each dimension

### 3.2 If dim upgrade justified, update pipeline

- Update `pyproject.toml` `[tool.imas-codex.embedding]` dimension
- Run `--reset-to enriched` to re-embed all 20K nodes
- Recreate vector indexes at new dimension
- Update benchmark thresholds

### Decision criteria

Upgrade to dim 512 if vector MRR improves by ≥0.10 (from ~0.15 to ≥0.25).
Upgrade to dim 1024 only if the additional gain over 512 is ≥0.05.

---

## Phase 4: Remaining Items from search-recall-and-enrichment.md

These items from the existing plan are still relevant:

### 4.1 Lucene fuzzy search (plan item 2.1)

Add fuzzy matching for short queries and misspellings:
```
temperture~2 → finds "temperature"
```
Currently BM25 misses misspellings entirely. Vector catches some via
embedding similarity but not reliably.

### 4.2 DoE weight optimization (plan items 4.2-4.4)

The evaluation harness exists in `test_search_evaluation.py`. Run the grid
search to optimize the 7 tunable boost weights:
- RRF k (currently 60)
- Path segment match boost (currently 0.03)
- Terminal segment boost (currently 0.08)
- IDS name boost (currently 0.05)
- Abbreviation boost (currently 0.15)
- Cluster boost (currently 0.02)
- Hierarchy boost (currently 0.02)
- Coordinate boost (currently 0.01)

### 4.3 Expand benchmark corpus (plan item 5.1)

Current: 65 queries across 7 categories.
Target: 200+ queries with stratified coverage:
- 40 exact concept (currently 10)
- 20 disambiguating (currently 5)
- 20 structural (currently 5)
- 40 abbreviation (currently 13)
- 20 accessor (currently 5)
- 40 cross-domain (currently 10)
- 20 edge-case (currently 5)

---

## Phase Dependencies

```
Phase 1 (cluster output)     → independent, do first
Phase 2 (auto-eval)          → after Phase 1 (needs cluster labels to validate)
Phase 3 (dim evaluation)     → independent, can parallel with Phase 1
Phase 4.1 (fuzzy search)     → independent
Phase 4.2 (DoE weights)      → after Phases 1, 3 (tune with final pipeline)
Phase 4.3 (expand corpus)    → after Phase 2 (auto-generation makes this easy)
```

## Documentation Updates

- `AGENTS.md`: Update search tool description to mention cluster labels
- `plans/README.md`: Add this plan, update search-recall-and-enrichment status
- `plans/features/search-recall-and-enrichment.md`: Mark completed items
