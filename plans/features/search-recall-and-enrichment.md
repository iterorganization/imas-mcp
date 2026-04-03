# Search Recall, Scoring, and Output Enrichment

## Problem Statement

The graph-backed IMAS DD search pipeline (codex-*) shows lower recall than the
legacy file-backed server (imas-*) on expanded MRR benchmarks (0.660 vs 0.872),
despite having richer data, 3,071 semantic clusters, 17 vector indexes, and full
graph traversal capabilities. The root cause is **compounding filtering losses**
across 6 pipeline stages, a broken score-mixing strategy, and under-exploitation
of Neo4j's native capabilities. The codex server already wins on precision for
semantic queries (P@10: 0.743 vs 0.686) — the goal is to close the recall gap
while preserving precision, then go far beyond the old server by leveraging
graph-native features that have no in-memory equivalent.

### Measured Baselines (13-query benchmark)

| Metric             | imas-*  | codex-* | Target  |
|--------------------|---------|---------|---------|
| Expanded MRR       | 0.872   | 0.660   | ≥ 0.90  |
| Precision@10       | 0.652   | 0.669   | ≥ 0.75  |
| IDS Recall@10      | 0.354   | 0.361   | ≥ 0.50  |
| Acronym MRR        | 0.750   | 0.271   | ≥ 0.80  |

### Architectural Insight

The old server computes `numpy.dot(query, ALL_60K_embeddings)` — exact NN with
zero filtering. The graph server uses HNSW approximate NN → LIMIT 150 → 4 WHERE
clauses → BM25 with score floor → weak additive merge. Each stage loses
candidates. But the graph gives us capabilities the old server can never have:
cluster-aware ranking, hierarchical boosting, coordinate-based discovery, Lucene
advanced queries (fuzzy, field-boost, phrase proximity), and children metadata
previews for agents.

---

## Phase 1: Fix Scoring Foundation

**Goal:** Remove score distortion and broken mixing. This is prerequisite for all
later phases — no point tuning anything built on a broken scoring signal.

### 1.1 Remove BM25 Score Floor

**File:** `imas_codex/tools/graph_search.py:1905`

```python
# Before:
normalized.append({"id": pid, "score": max(raw, 0.7)})
# After:
normalized.append({"id": pid, "score": raw})
```

The floor at 0.7 compresses BM25's discriminative range from [0, 1] to
[0.7, 1.0]. This causes weak text matches to outrank strong vector matches
during merge. Verified empirically: for "electron temperature", BM25 raw
normalized scores already range [0.74, 1.00] — the floor is redundant for
good matches and harmful for bad ones.

### 1.2 Implement Reciprocal Rank Fusion (RRF)

**File:** `imas_codex/tools/graph_search.py:191-226`

Replace the current `max(vector, text) + 0.05` merge with standard RRF:

```python
def _reciprocal_rank_fusion(
    vector_results: list[dict],  # [{id, score}] sorted by score desc
    text_results: list[dict],    # [{id, score}] sorted by score desc
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion — rank-based, score-normalization-free."""
    scores: dict[str, float] = {}
    for rank, r in enumerate(vector_results):
        scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (k + rank + 1)
    for rank, r in enumerate(text_results):
        scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (k + rank + 1)
    return scores
```

RRF is rank-based, not score-based. It's immune to score normalization
issues and is the industry standard for hybrid search (Pinecone, Weaviate,
Qdrant all default to RRF). The `k=60` constant controls how quickly
relevance decays with rank — standard value from the original Cormack et al.
(2009) paper.

### 1.3 Increase Vector Candidate Limit

**File:** `imas_codex/tools/graph_search.py:164`

```python
# Before:
"vector_limit": min(max_results * 3, 150),
# After:
"vector_limit": min(max_results * 5, 500),
```

For broad queries like "electron temperature" (495 valid paths), 150
candidates drops ~70% of valid matches before scoring even begins. 500
candidates is still fast (HNSW is O(log n)) and matches the over-fetch
factor recommended by vector DB vendors (5-10x).

### 1.4 Relax `documentation > 10` Filter

**File:** `imas_codex/tools/graph_search.py:1892, 1921`

```python
# Before:
WHERE size(coalesce(p.documentation, '')) > 10
# After:
WHERE (p.description IS NOT NULL OR size(coalesce(p.documentation, '')) > 0)
```

The 10-char minimum was added to reduce noise from sparse nodes. But 100%
of data nodes now have LLM-enriched descriptions (verified: 20,037/20,037).
The `description` field is the primary text — `documentation` is the raw
DD text. Check for either.

---

## Phase 2: Design of Experiments (DoE) — Score Mixing

**Goal:** Empirically discover optimal mixing parameters for the three search
channels (vector, BM25, graph-structural) using a systematic DoE.

### 2.1 Build Evaluation Harness

Create `tests/llm/test_search_evaluation.py` with:
- 13 benchmark queries from our existing ground truth corpus
- Automated MRR, P@10, IDS-Recall@10 computation
- Parameterized test runner accepting mixing configuration
- Results logged to JSON for analysis

The harness should call the search pipeline directly (not through MCP) to
avoid network overhead. It should accept configuration like:

```python
@dataclass
class MixConfig:
    rrf_k: int = 60                    # RRF rank decay constant
    vector_limit: int = 500            # HNSW candidate cap
    text_limit: int = 200              # BM25 candidate cap
    vector_weight: float = 1.0         # RRF channel weight multiplier
    text_weight: float = 1.0           # RRF channel weight multiplier
    path_boost: float = 0.03           # per-word path segment boost
    abbreviation_boost: float = 0.35   # abbreviation exact-match boost
    cluster_boost: float = 0.0         # cluster membership boost (Phase 3)
    hierarchy_boost: float = 0.0       # parent proximity boost (Phase 3)
```

### 2.2 Run DoE Grid Search

Test a factorial grid across the key parameters:

| Parameter       | Values to test           |
|-----------------|--------------------------|
| `rrf_k`         | 20, 60, 100              |
| `vector_limit`  | 150, 300, 500            |
| `text_limit`    | 100, 200, 300            |
| `vector_weight` | 0.5, 1.0, 1.5            |
| `text_weight`   | 0.5, 1.0, 1.5            |

This is 3^5 = 243 configurations. Each runs 13 queries. Total: ~3,200 search
calls. At ~50ms per call, this completes in ~3 minutes. Log all results.

### 2.3 Analyze DoE Results

Identify the Pareto frontier of configurations that maximize MRR while
maintaining P@10 ≥ 0.70. Select the configuration that maximizes the
composite metric: `0.5 * MRR + 0.3 * P@10 + 0.2 * IDS_Recall@10`.

### 2.4 Consider Separate Channel Results

Investigate whether returning vector results and text results as separate
sections in the output (rather than fused) would be more useful for agents.
This would let the consumer decide which channel to trust.

**Design decision:** Probably not — agents expect a single ranked list. But
we could add a `search_channel` field to each result indicating whether it
came from vector, text, or both. This preserves signal provenance without
polluting the ranking.

---

## Phase 3: Graph-Native Search Augmentation

**Goal:** Exploit graph structure for ranking signals that are impossible in
flat/file-backed search. This is where we go beyond the old server.

### 3.1 Exploit Lucene Advanced Queries

The Neo4j fulltext index is powered by Apache Lucene, which supports features
we are completely ignoring. Verified working on our graph:

| Feature | Lucene Syntax | Use Case |
|---------|---------------|----------|
| **Field boosting** | `description:psi^3 OR name:psi^2` | Weight description matches higher than path ID matches |
| **Fuzzy matching** | `temperture~2` | Catch misspellings (Levenshtein distance ≤ 2) |
| **Phrase proximity** | `"electron temperature"~3` | Match words within 3 positions of each other |
| **Prefix queries** | `name:elonga*` | Match partial terms |
| **Boolean operators** | `AND`, `OR`, `NOT` | Compose complex queries |

**Implementation:** Modify `_text_search_imas_paths()` to construct Lucene
queries with field boosting:

```python
def _build_lucene_query(query: str, intent: QueryIntent) -> str:
    """Build a Lucene query with field boosting and fuzzy matching."""
    terms = query.split()
    parts = []
    for term in terms:
        # Boost description matches highest (LLM-enriched, physics-aware)
        parts.append(f"(description:{term}^3 OR name:{term}^2 "
                     f"OR documentation:{term} OR keywords:{term}^2)")
    base = " AND ".join(parts)
    # Add fuzzy variant for each term (catches misspellings)
    fuzzy = " OR ".join(f"{term}~1" for term in terms if len(term) > 3)
    if fuzzy:
        return f"({base}) OR ({fuzzy})"
    return base
```

This gives us **typo tolerance** (a feature neither server has today),
**field-weighted relevance** (description > name > documentation), and
**phrase proximity** for multi-word queries.

### 3.2 Cluster-Aware Ranking Boost

**Relationship:** `(n:IMASNode)-[:IN_CLUSTER]->(c:IMASSemanticCluster)`
**Stats:** 3,071 clusters, avg 6.4 members, 452 global clusters

When search results include multiple nodes from the same cluster, this is
a strong signal of thematic relevance. Add a cluster consensus boost:

```cypher
// After initial scored results, check cluster membership overlap
UNWIND $path_ids AS pid
MATCH (p:IMASNode {id: pid})-[:IN_CLUSTER]->(c:IMASSemanticCluster)
WITH pid, collect(c.id) AS clusters
RETURN pid, clusters,
       size([cl IN clusters WHERE cl IN $top_clusters]) AS cluster_overlap
```

Boost paths that share clusters with other top-ranked results. This creates
a self-reinforcing relevance signal unique to graph-backed search.

### 3.3 Hierarchical Proximity Boost

**Relationship:** `(child:IMASNode)-[:HAS_PARENT]->(parent:IMASNode)`
**Verified:** `shortestPath` works — psi↔q distance = 2 (share parent
`equilibrium/time_slice/profiles_1d`)

Nodes that are siblings in the DD hierarchy are structurally related. If
the query matches one sibling strongly, boost nearby siblings:

```cypher
// Find graph-proximate nodes for top results
MATCH (top:IMASNode {id: $top_result_id})-[:HAS_PARENT]->(parent)<-[:HAS_PARENT]-(sibling)
WHERE sibling.id IN $candidate_ids
RETURN sibling.id AS id, 0.02 AS hierarchy_boost
```

This helps surface related quantities (e.g., searching for "psi" should
also surface "q", "phi", "rho_tor_norm" from the same profiles_1d parent).

### 3.4 Coordinate-Based Cross-IDS Discovery

**Relationship:** `(n:IMASNode)-[:HAS_COORDINATE]->(coord:IMASNode)`
**Stats:** 17,755 paths have IMASNode coordinate targets

Paths sharing the same coordinate basis are physically related. This is a
powerful cross-IDS discovery signal:

```cypher
// Find paths sharing coordinates with top results
MATCH (top:IMASNode {id: $top_result_id})-[:HAS_COORDINATE]->(coord:IMASNode)
    <-[:HAS_COORDINATE]-(related:IMASNode)
WHERE related.id <> top.id
RETURN DISTINCT related.id AS id, related.ids AS ids,
       collect(DISTINCT coord.name) AS shared_coords
```

This enables queries like "show me everything on the same radial grid as
this profile" — impossible in flat search.

### 3.5 Activate QueryAnalyzer for Intent-Based Routing

**File:** `imas_codex/tools/query_analysis.py` — fully implemented but NOT
connected to the search pipeline (confirmed: 0 references in graph_search.py).

Connect it to route queries to different search strategies:

| Intent | Strategy |
|--------|----------|
| `path_exact` | Direct Cypher MATCH on path ID, skip vector/text |
| `abbreviation` | Expand terms + abbreviation boost + Lucene fuzzy |
| `concept` | Full hybrid: vector + Lucene + cluster boost |
| `hybrid` | Both abbreviation expansion and concept search |

This avoids running expensive vector search for exact path lookups and
ensures abbreviation expansion always happens for short queries.

---

## Phase 4: Output Enrichment for Agents

**Goal:** Transform search results from flat path lists into rich, actionable
context that helps LLM agents understand IMAS data without follow-up queries.

### 4.1 Children Metadata Preview

When a search result is a structure node (STRUCTURE, STRUCT_ARRAY) or a data
node with children, show immediate children with their types, units, and
descriptions. This eliminates the need for agents to call `list_imas_paths`
as a follow-up.

**Schema support:** `SearchHit.children` field already defined in
`search_strategy.py:186-189` but not populated.

**Enrichment Cypher:**
```cypher
UNWIND $path_ids AS pid
MATCH (p:IMASNode {id: pid})
OPTIONAL MATCH (child:IMASNode)-[:HAS_PARENT]->(p)
WHERE child.data_type IS NOT NULL
    AND NOT (child.name ENDS WITH '_error_upper'
         OR child.name ENDS WITH '_error_lower'
         OR child.name ENDS WITH '_error_index')
OPTIONAL MATCH (child)-[:HAS_UNIT]->(u:Unit)
WITH pid, child, u
ORDER BY child.name
WITH pid, collect(DISTINCT {
    name: child.name,
    data_type: child.data_type,
    unit: u.id,
    description: left(coalesce(child.description, ''), 80)
})[..10] AS children
RETURN pid AS id, children
```

**Formatting in output:**
```
### equilibrium/time_slice/boundary (score: 0.92)
  "Plasma boundary description from equilibrium reconstruction."
  IDS: equilibrium | Type: STRUCT_ARRAY
  Children (15 total, showing 10):
    - elongation (FLT_0D) — Global plasma elongation parameter
    - outline (STRUCTURE) — RZ outline of the plasma boundary
    - psi (FLT_0D, Wb) — Poloidal flux at the boundary
    - strike_point (STRUCT_ARRAY) — Strike point positions
    - triangularity_lower (FLT_0D) — Lower triangularity
    ...
```

This is transformative for agents — they immediately see what data lives
under a structure node without a second tool call.

### 4.2 Templated Description Injection

For nodes with enriched descriptions, the graph already stores high-quality
physics text. Currently, the output formatter shows either `description` or
`documentation`. Enhance to show both when the description adds context
beyond the raw documentation:

```python
if hit.description and hit.documentation:
    if hit.description != hit.documentation:
        parts.append(f'  "{hit.description}"')
        parts.append(f'  Raw DD: "{hit.documentation[:100]}"')
    else:
        parts.append(f'  "{hit.description}"')
```

### 4.3 Search Channel Provenance

Add a `source` field to each result indicating which search channels found
it: `vector`, `text`, `both`, `cluster`, or `graph`. This helps agents
and users understand why a result was returned without polluting the ranking.

```python
@dataclass
class SearchHitSource:
    vector_rank: int | None = None    # rank in vector results (None = not found)
    text_rank: int | None = None      # rank in text results (None = not found)
    cluster_match: bool = False       # found via cluster search
    graph_boost: str | None = None    # "sibling", "coordinate", "cluster_consensus"
```

### 4.4 Parent Context in Results

For leaf data nodes deep in the hierarchy, show the parent chain up to IDS
level as breadcrumbs:

```
### core_profiles/profiles_1d/electrons/temperature (score: 0.95)
  Path: core_profiles > profiles_1d > electrons > temperature
  "Electron temperature radial profile from core profile analysis."
```

This helps agents understand WHERE in the IDS hierarchy a result lives
without having to parse the path string.

---

## Phase 5: Evaluation Infrastructure

**Goal:** Build permanent evaluation infrastructure so we can measure the
impact of every change and catch regressions.

### 5.1 Expand Ground Truth Corpus

Current: 13 queries. Target: 50+ queries covering:
- 10 acronyms (ip, q, b0, te, ti, ne, psi, bt, jt, zeff)
- 15 semantic concepts (electron temperature, safety factor, plasma boundary,
  toroidal field, bootstrap current, resistivity, etc.)
- 10 keyword/path-like queries (flux_loop, elongation, rho_tor_norm, etc.)
- 10 cross-IDS queries (quantities appearing in 5+ IDSs)
- 5 edge cases (misspellings, partial paths, mixed notation)

Ground truth built from graph: `IN_CLUSTER` membership + path pattern + manual
curation. Store in `tests/llm/ground_truth_corpus.json`.

### 5.2 Automated Regression Gate

Add to CI: run the evaluation harness after any change to `graph_search.py`,
`query_analysis.py`, or `search_formatters.py`. Fail if MRR drops below
threshold or P@10 drops below 0.70.

### 5.3 Per-Query Diagnostics

For each benchmark query, log:
- Vector results count and top-5 scores
- BM25 results count and top-5 scores
- RRF merged results count
- Filters applied and candidates removed at each stage
- Final ranking vs ground truth

This enables rapid debugging when a change helps one query but hurts another.

---

## Documentation Updates

| Target | Update |
|--------|--------|
| `AGENTS.md` | Add search tuning parameters section, DoE workflow |
| `plans/README.md` | Add this plan |
| `tests/llm/README.md` | Document evaluation harness usage |
| Schema reference | Auto-updated by `build-models` if SearchHit model changes |

---

## Phase Dependencies

```
Phase 1 (Scoring Foundation)
  ├── 1.1 Remove BM25 floor
  ├── 1.2 Implement RRF
  ├── 1.3 Increase vector limit
  └── 1.4 Relax doc filter
        │
Phase 2 (DoE)
  ├── 2.1 Build eval harness ← depends on Phase 1
  ├── 2.2 Grid search
  ├── 2.3 Analyze results
  └── 2.4 Channel separation decision
        │
Phase 3 (Graph-Native) ← can start 3.1, 3.5 in parallel with Phase 2
  ├── 3.1 Lucene advanced queries
  ├── 3.2 Cluster-aware ranking
  ├── 3.3 Hierarchical proximity
  ├── 3.4 Coordinate cross-IDS
  └── 3.5 QueryAnalyzer activation
        │
Phase 4 (Output Enrichment) ← independent, can start any time
  ├── 4.1 Children preview
  ├── 4.2 Description injection
  ├── 4.3 Channel provenance
  └── 4.4 Parent breadcrumbs
        │
Phase 5 (Evaluation) ← 5.1 should start with Phase 2
  ├── 5.1 Expand corpus
  ├── 5.2 CI gate
  └── 5.3 Diagnostics
```

## Verified Capabilities

All graph-native features in this plan have been verified working on our
Neo4j 2026.01.4 instance:

- Lucene field boosting: `description:psi^3 OR name:psi^2` ✓
- Lucene fuzzy matching: `temperture~2` catches misspellings ✓
- Lucene phrase proximity: `"electron temperature"~3` ✓
- Lucene prefix: `name:elonga*` ✓
- `HAS_PARENT` traversal: sibling discovery in 2 hops ✓
- `IN_CLUSTER` membership: 19,649 cluster edges, 452 global clusters ✓
- `HAS_COORDINATE` cross-IDS: 17,755 coordinate edges ✓
- `shortestPath`: works for hierarchical distance ✓
- Children via `HAS_PARENT` reverse: verified on boundary (15 children) ✓
- Vector cosine scores range: [0.90, 1.00] for related terms ✓
- BM25 raw scores range: [11.8, 16.0] for "electron temperature" ✓
- APOC enabled but unused (available for fuzzy matching) ✓
- GDS NOT available (community edition) — not required for this plan ✓
- `db.index.fulltext.queryRelationships` available but not yet needed ✓
