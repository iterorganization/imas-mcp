# MCP Tool Quality Refactor Plan

> **Date**: 2026-03-08
> **Status**: Proposed
> **Severity**: Critical — multiple tools are functionally broken

## Executive Summary

A systematic audit of all codex MCP search tools reveals **5 critical bugs** that cause
complete tool failure, **8 quality issues** that degrade results, and **significant gaps**
compared to the older non-graph-backed IMAS MCP server. The graph contains rich data
(230K facility paths, 121K wiki chunks, 66K signals, 61K IMAS paths, 40K documents,
25K code chunks) but the tools fail to surface it.

---

## 1. Critical Bugs (Must Fix)

### 1.1 Code Search Always Returns Empty — Schema Mismatch

**File**: `imas_codex/agentic/search_tools.py` `_vector_search_code_chunks()`

**Bug**: The facility filter uses `MATCH (cf:CodeFile)-[:HAS_CHUNK]->(cc)` but this
relationship does **not exist** in the graph.

**Actual relationship chain**:
```
FacilityPath -[:PRODUCED]-> CodeExample -[:HAS_CHUNK]-> CodeChunk
```

There are **0** `CodeFile-[:HAS_CHUNK]->CodeChunk` relationships in the graph. There are
also 0 `CodeFile-[:PRODUCED]->CodeExample` relationships.

**Impact**: 100% failure rate for `search_code()` with any facility filter.
Both TCV (17,809 code chunks) and JET (7,426 code chunks) return "No code examples found."

**Evidence** (direct graph query):
```
CodeFile-[:HAS_CHUNK]->CodeChunk: 0 rows
CodeFile-[:PRODUCED]->CodeExample-[:HAS_CHUNK]->CodeChunk: 0 rows
FacilityPath-[:PRODUCED]->CodeExample-[:HAS_CHUNK]->CodeChunk: 20,797 rows
```

**Fix**: Replace the facility filter with a direct property filter:
```python
# Current (broken)
facility_filter = (
    "MATCH (cf:CodeFile)-[:HAS_CHUNK]->(cc) "
    "WHERE cf.facility_id = $facility "
    "WITH cc, score "
)
# Fixed
facility_filter = "WHERE cc.facility_id = $facility "
```
CodeChunk has `facility_id` stored directly — 17,809 TCV and 7,426 JET chunks have it set.

Also fix `_enrich_code_chunks()` which traverses the same broken chain:
```cypher
-- Current (broken):
OPTIONAL MATCH (ce:CodeExample)-[:HAS_CHUNK]->(cc)
OPTIONAL MATCH (cf:CodeFile {id: ce.source_file})
OPTIONAL MATCH (cf)-[:CONTAINS_REF]->(dr:DataReference)

-- Fixed:
OPTIONAL MATCH (ce:CodeExample)-[:HAS_CHUNK]->(cc)
OPTIONAL MATCH (fp:FacilityPath)-[:PRODUCED]->(ce)
OPTIONAL MATCH (cc)-[:CONTAINS_REF]->(dr:DataReference)  -- CONTAINS_REF is on CodeChunk, not CodeFile
```

### 1.2 Document Search Index Name Wrong

**File**: `imas_codex/agentic/search_tools.py` `_vector_search_documents()`

**Bug**: Uses index name `"wiki_document_desc_embedding"` but the actual index is
`"document_desc_embedding"`.

**Impact**: Document search silently fails for ALL queries. The exception is caught and
logged at DEBUG level, so it's invisible. JET has 19,159 documents (11,011 with embeddings)
that are never surfaced.

**Evidence**: Direct query on the correct index finds excellent results:
```
document_desc_embedding for "EFIT equilibrium":
  jet:T17-15_equilibrium_status.pdf (score: 0.902)
  jet:AMG_JDN-ITM_EFIT_(09-02).doc (score: 0.889)
  jet:T17-15_internal_constraints_V2_ok.pdf (score: 0.887)
```
But `search_docs()` returns "No documentation found for 'EFIT equilibrium' at jet."

**Fix**: Change `"wiki_document_desc_embedding"` to `"document_desc_embedding"`.

### 1.3 Post-Filtering Causes Facility Starvation

**File**: `imas_codex/agentic/search_tools.py` — all `_vector_search_*` functions

**Bug**: Vector searches use `CALL db.index.vector.queryNodes(index, $k, $embedding)` to
get the global top-k results, then **post-filter** by facility using WHERE clauses like
`WHERE (s)-[:AT_FACILITY]->(:Facility {id: $facility})`.

When one facility dominates the index (TCV has 32,832 embedded signals vs JET's ~1,200),
the smaller facility gets zero results because all k slots are taken by the dominant facility.

**Impact**: JET signal search returns "No signals found" for ANY query. Even with k=200,
JET signals don't appear. With k=5000, JET gets 512 results (vs TCV's 4,488).

Confirmed by unfiltered vector search for "plasma current":
```
k=20 unfiltered results: TCV: 20, JET: 0
k=5000 results: TCV: 4488, JET: 512
```

**Fix options** (in order of preference):
1. **Property pre-filter** — Use `WHERE node.facility_id = $facility` directly after YIELD.
   While this is still technically post-filtering in Neo4j, adding a property-based check
   before expensive relationship traversal is significantly faster:
   ```cypher
   CALL db.index.vector.queryNodes("facility_signal_desc_embedding", $k_multiplied, $embedding)
   YIELD node AS s, score
   WHERE s.facility_id = $facility
   RETURN s.id, score ORDER BY score DESC LIMIT $k
   ```
2. **Over-fetch then filter** — Set internal k to `k * 20` or a minimum of 200, then
   limit to requested k after filtering:
   ```python
   internal_k = max(k * 20, 200)
   ```
3. **Separate indexes per facility** — Most complex but guarantees results. Only worthwhile
   if we have 5+ facilities.

The property-based filter (option 1) with over-fetching (option 2) should be combined.
This applies to ALL vector search functions:
- `_vector_search_signals`
- `_vector_search_wiki_chunks`
- `_vector_search_documents`
- `_vector_search_code_chunks`
- `_vector_search_tree_nodes`

### 1.4 Fetch Tool CodeFile Traversal Broken

**File**: `imas_codex/agentic/search_tools.py` `_fetch_code_file()`

**Bug**: Uses `MATCH (cf:CodeFile)-[:PRODUCED]->(ce:CodeExample)-[:HAS_CHUNK]->(cc:CodeChunk)`
which has the same relationship chain issue — `CodeFile-[:PRODUCED]->CodeExample` does not exist.

**Impact**: `fetch()` for any code file resource always returns "No resource found."

**Fix**: Use `FacilityPath` instead of `CodeFile`:
```cypher
MATCH (fp:FacilityPath)-[:PRODUCED]->(ce:CodeExample)-[:HAS_CHUNK]->(cc:CodeChunk)
WHERE fp.id = $resource OR fp.path = $resource
```

### 1.5 IMAS Search Missing Canonical Paths

**Bug**: The `search_imas("plasma current")` returns 9 results with NO canonical plasma
current paths. Missing:
- `equilibrium/time_slice/global_quantities/ip` (doc: "Plasma current")
- `summary/global_quantities/ip` (doc: "Total plasma current")
- `equilibrium/time_slice/constraints/ip` (doc: "Plasma current")

All three exist in the graph WITH embeddings. Instead, the tool returns:
- 5 "Verbose description" paths (generic metadata)
- Edge profiles current paths (tangentially related)

**Root cause**: The embedding similarity for short docs like "Verbose description" or
"Plasma current" is noisy — the vector model ranks them similarly because both are
semantically thin. The old IMAS MCP server uses hybrid (BM25 + semantic) search, which
catches keyword matches like `ip` and `current` via text scoring.

**Fix**: Add hybrid search:
1. Run vector search as today
2. Also run a text-match query:
   ```cypher
   MATCH (p:IMASPath)
   WHERE toLower(p.documentation) CONTAINS $query_lower
      OR toLower(p.id) CONTAINS $normalized_query
   RETURN p.id, 0.5 AS score LIMIT $k
   ```
3. Merge and deduplicate results, boosting paths that appear in both
4. Filter out "Verbose description" paths explicitly (or penalize paths where
   `documentation` length < 20 characters as generic metadata)

---

## 2. Quality Issues (Should Fix)

### 2.1 Signal Duplicates from Multi-Access DataAccess

**Symptom**: `tcv:general/tcv_cp_norm` appears twice in results — once with `tdi` access,
once with `local` access. Same signal, different access methods.

**Root cause**: The enrichment query `_enrich_signals()` does OPTIONAL MATCH on DataAccess,
creating a cartesian product when a signal has 2+ access methods.

**Fix**: Collect access methods into an array:
```cypher
OPTIONAL MATCH (s)-[:DATA_ACCESS]->(da:DataAccess)
...
RETURN s.id AS id, ...,
       collect(DISTINCT {template: da.data_template, type: da.access_type,
               imports: da.imports_template, connection: da.connection_template}) AS access_methods
```
Then format multiple access methods under one signal heading.

### 2.2 Template Placeholders Not Interpolated

**Symptom**: Data access shows `tree.getNode('{node_path}').data()` with literal
`{node_path}` placeholder. The signal has `node_path: \MAGNETICS::IPLASMA` which should
be substituted.

**Fix**: In the formatter or enrichment, interpolate known values:
```python
template = sig.get("access_template", "")
substitutions = {
    "node_path": sig.get("sig_node_path") or sig.get("tree_path"),
    "accessor": sig.get("sig_accessor"),
    "data_source": sig.get("sig_tree_name"),
    "shot": "{shot}",  # Keep as placeholder — user-specific
}
for key, val in substitutions.items():
    if val:
        template = template.replace(f"{{{key}}}", val)
```
Return the enrichment query needs to also fetch `s.node_path`, `s.accessor`, `s.tree_name`.

### 2.3 "Related Tree Nodes" Section Is Noise

**Symptom**: For "plasma current", tree nodes section shows STATIC calibration data
(`\STATIC::PHI_P`, `\STATIC::DMUTDZI_SL_M`) with scores of 0.90+. These are completely
irrelevant.

**Root cause**: DataNode descriptions for STATIC tree nodes are generic calibration/geometry
text. The vector similarity between "plasma current" and calibration data is misleadingly
high at 256 dimensions.

**Fix options**:
1. Remove the tree node section entirely (it adds noise)
2. Apply a minimum score threshold (e.g., 0.92+)
3. Filter out `STATIC::` nodes
4. Only show tree nodes that have a relationship to the matched signals

Option 4 is best — show tree nodes that are connected to results, not independently searched.

### 2.4 IMAS Cluster Results Triplicated

**Symptom**: Same cluster appears 3 times with different `scope` values (ids, global, domain).
Example: "Measured Current Density Constraints" × 3.

**Fix**: Deduplicate clusters by label, displaying all scopes in one entry:
```python
# Group by label
seen = {}
for cl in cluster_results:
    label = cl["label"]
    if label not in seen:
        seen[label] = {"label": label, "scopes": [], "path_count": cl["path_count"]}
    seen[label]["scopes"].append(cl["scope"])
```

### 2.5 No Hybrid Search (Text + Semantic)

**Symptom**: Queries like "electron temperature" miss `core_profiles/profiles_1d/electrons/temperature`
because the embedding for "Temperature" (the documentation) doesn't rank highest for
"electron temperature" — instead `core_instant_changes` paths rank higher.

The old IMAS MCP server uses hybrid search (BM25 + vector) and finds better results because
text matching catches keyword-relevant paths.

**Fix**: Add a text match stage to all search functions:
1. `search_signals`: Match on `s.name CONTAINS $query OR s.node_path CONTAINS $query`
2. `search_docs`: Match on `c.text CONTAINS $query` (full-text)
3. `search_imas`: Match on `p.documentation CONTAINS $query OR p.id CONTAINS $query_normalized`
4. `search_code`: Match on `cc.text CONTAINS $query`

Merge text results with vector results, boosting dual-match entries.

Consider creating Neo4j full-text indexes for key text fields:
```cypher
CREATE FULLTEXT INDEX wiki_chunk_text IF NOT EXISTS
FOR (n:WikiChunk) ON EACH [n.text]
```

### 2.6 Low Default k Values

**Symptom**: `search_code` defaults to k=5, `search_signals`/`search_docs`/`search_imas`
default to k=10. Combined with post-filtering, this guarantees failure for minority facilities.

**Fix**: Increase defaults and use over-fetching:
- Internal k = `max(k * 20, 200)` for vector search
- User-facing k remains the same (5/10)
- Limit to requested k after filtering

### 2.7 Code Chunk `function_name` Always Null

**Symptom**: All code chunks have `function_name: None`, reducing the usefulness of code
search results.

**Root cause**: The `function_name` property may not be populated during ingestion, or it's
stored on a different property name.

**Action**: Audit the code ingestion pipeline to verify `function_name` is being set. Check
if it's stored as a different property on CodeChunk or CodeExample nodes.

### 2.8 IMAS Search Result Richness Gap

**Comparison with old IMAS MCP server**:

| Feature | Old IMAS MCP | Codex MCP |
|---------|-------------|-----------|
| Results per query | 50 paths | 8-10 paths |
| Search mode | Hybrid (BM25 + vector) | Vector only |
| Per-result metadata | Rich (docs, units, type, coordinates, validation_rules, lifecycle, structure_reference) | Basic (docs, units, type, clusters) |
| Result quality | Better (focs/current for "plasma current") | Worse (edge_profiles, "Verbose description") |
| Path coverage | Covers all major IDS | Misses canonical paths |
| Response size | 70KB structured JSON | 2-3KB markdown |

**Fix**: The graph HAS all this data on IMASPath nodes. The enrichment query should return:
- `p.structure_reference` (e.g., "signal_flt_1d_validity")
- Coordinate details via `[:HAS_COORDINATE]`
- Validation info
- More documentation context via cluster traversal

Also increase default k from 10 to 20-30 for IMAS search.

---

## 3. Verified Index Configuration

All vector indexes are 256-dimensional (matching the current encoder), ONLINE, using cosine
similarity and vector-3.0 provider. No dimension mismatches detected.

| Index | Dims | Data Volume |
|-------|------|------------|
| wiki_chunk_embedding | 256 | 118K chunks embedded |
| facility_signal_desc_embedding | 256 | 34K signals embedded |
| code_chunk_embedding | 256 | 25K chunks embedded |
| imas_path_embedding | 256 | 61K paths |
| document_desc_embedding | 256 | 11K docs embedded |
| image_desc_embedding | 256 | 9K images embedded |
| data_node_desc_embedding | 256 | 4K nodes |
| facility_path_desc_embedding | 256 | 230K paths |
| cluster_description_embedding | 256 | 8K clusters |
| cluster_embedding | 256 | 8K clusters |
| cluster_label_embedding | 256 | 8K clusters |
| tree_node_desc_embedding | 256 | 0 nodes (empty) |

**Note**: `tree_node_desc_embedding` is empty — all tree node data is in `data_node_desc_embedding`.

---

## 4. Graph Data Audit

### Per-Facility Content Summary

| Metric | TCV | JET | ITER |
|--------|-----|-----|------|
| FacilityPath | 55,375 | 175,327 | 87 |
| FacilitySignal | 64,950 (32,832 embedded) | 1,207 true signals | 0 |
| WikiPages | 7,782 | 15,033 | 951 |
| WikiChunks | 72,816 (all embedded) | 45,556 (all embedded) | 2,901 |
| CodeChunks | 17,809 (all embedded) | 7,426 (7,416 embedded) | 0 |
| Documents | - | 19,159 (11,011 embedded) | - |
| Images | - | 9,480 (9,454 embedded) | - |
| DataNodes | 82,717 | 4,045 | 0 |

### Neo4j Label Anomaly (JET)

`MATCH (n:FacilitySignal {facility_id: 'jet'}) RETURN count(n)` returns 4,028 but
`MATCH (n) WHERE 'FacilitySignal' IN labels(n) AND n.facility_id = 'jet'` returns 1,207.

This 3.3x discrepancy suggests a corrupted property index. The extra matches include
CodeFile (1,308), FacilityPath (725), WikiChunk (370), CodeChunk (249) nodes that don't
actually carry the FacilitySignal label. This should be investigated — may require
dropping and recreating the `facility_id` range index on FacilitySignal.

**Action**: Run `CALL db.index.fulltext.drop()` or recreate indexes after investigating.

---

## 5. Refactor Implementation Plan

### Phase 1: Critical Bug Fixes (Immediate)

1. **Fix code search schema** (`search_tools.py`)
   - Replace `CodeFile` traversal with `cc.facility_id` property filter
   - Fix enrichment query to use `FacilityPath->CodeExample->CodeChunk` chain
   - Fix `CONTAINS_REF` source: it's on CodeChunk, not CodeFile

2. **Fix document index name** (`search_tools.py`)
   - Change `"wiki_document_desc_embedding"` → `"document_desc_embedding"`

3. **Fix facility starvation** (`search_tools.py`)
   - All `_vector_search_*` functions: over-fetch with `internal_k = max(k * 20, 200)`
   - Use property-based facility filter `WHERE node.facility_id = $facility` instead of
     relationship traversal for initial filtering
   - Only use AT_FACILITY relationship check if property is missing

4. **Fix fetch CodeFile traversal** (`search_tools.py`)
   - Use `FacilityPath` instead of `CodeFile` in `_fetch_code_file()`

### Phase 2: Quality Improvements (Next Sprint)

5. **Deduplicate signal DataAccess results**
   - Collect access methods into array, show all under one signal heading

6. **Interpolate template placeholders**
   - Substitute `{node_path}`, `{accessor}`, `{data_source}` from signal properties
   - Keep `{shot}` as user-parameterized

7. **Add hybrid search to IMAS tool**
   - Text match on `documentation CONTAINS` and `id CONTAINS`
   - Merge with vector results, boost dual-matches
   - Filter out "Verbose description" paths (doc length < 20 chars)

8. **Deduplicate IMAS clusters**
   - Group by label, combine scope values

9. **Improve tree node relevance**
   - Only show tree nodes connected to matched signals via graph traversal
   - Remove independent tree node vector search (or make it optional)

### Phase 3: Feature Parity with Old IMAS Server (Planned)

10. **Enrich IMAS results**
    - Add `structure_reference`, coordinate details, comprehensive documentation
    - Increase default k to 20-30

11. **Full-text indexes**
    - Create Neo4j full-text indexes on `WikiChunk.text`, `IMASPath.documentation`,
      `FacilitySignal.name`, `CodeChunk.text`
    - Use in hybrid search alongside vector search

12. **Investigate Neo4j label anomaly**
    - Drop and recreate range indexes on `facility_id`
    - Verify no cross-label contamination

### Phase 4: Unit Tests (Ongoing)

13. **Create comprehensive test suite** for search tools:

```python
# tests/agentic/test_search_tools.py

class TestSearchSignals:
    """Test signal search quality and correctness."""

    def test_tcv_plasma_current_returns_results(self, graph_client, encoder):
        """Regression: TCV plasma current should always return results."""
        result = _search_signals("plasma current", "tcv", gc=graph_client, encoder=encoder)
        assert "No signals found" not in result
        assert "iplasma" in result.lower() or "tcv_cp" in result.lower()

    def test_jet_plasma_current_returns_results(self, graph_client, encoder):
        """Regression: JET signals must not be starved by TCV dominance."""
        result = _search_signals("plasma current", "jet", gc=graph_client, encoder=encoder)
        assert "No signals found" not in result

    def test_no_duplicate_signals(self, graph_client, encoder):
        """Regression: Same signal should not appear multiple times."""
        result = _search_signals("plasma current", "tcv", gc=graph_client, encoder=encoder)
        lines = [l for l in result.split("\n") if l.startswith("### ")]
        ids = [l.split("(")[0].strip("# ") for l in lines]
        assert len(ids) == len(set(ids)), f"Duplicates: {ids}"

    def test_template_placeholders_interpolated(self, graph_client, encoder):
        """Regression: {node_path} should be filled with actual path."""
        result = _search_signals("plasma current", "tcv", gc=graph_client, encoder=encoder)
        assert "{node_path}" not in result or "\\MAGNETICS" in result


class TestSearchCode:
    """Test code search quality and correctness."""

    def test_tcv_returns_results(self, graph_client, encoder):
        """Regression: TCV code search must not return empty."""
        result = _search_code("equilibrium reconstruction", facility="tcv",
                             gc=graph_client, encoder=encoder)
        assert "No code examples found" not in result

    def test_jet_returns_results(self, graph_client, encoder):
        """Regression: JET code search must not return empty."""
        result = _search_code("equilibrium reconstruction", facility="jet",
                             gc=graph_client, encoder=encoder)
        assert "No code examples found" not in result


class TestSearchDocs:
    """Test documentation search quality and correctness."""

    def test_jet_efit_returns_results(self, graph_client, encoder):
        """Regression: JET EFIT docs must include PDF/DOC documents."""
        result = _search_docs("EFIT equilibrium", "jet",
                             gc=graph_client, encoder=encoder)
        assert "No documentation found" not in result

    def test_includes_documents(self, graph_client, encoder):
        """Documents from document_desc_embedding index must be included."""
        result = _search_docs("equilibrium reconstruction", "jet",
                             gc=graph_client, encoder=encoder)
        assert "Related Documents" in result or ".pdf" in result.lower()


class TestSearchImas:
    """Test IMAS search quality and correctness."""

    def test_electron_temperature_finds_core_profiles(self, graph_client, encoder):
        """Canonical path must appear in results."""
        result = _search_imas("electron temperature", gc=graph_client, encoder=encoder)
        assert "core_profiles/profiles_1d/electrons/temperature" in result

    def test_plasma_current_finds_equilibrium_ip(self, graph_client, encoder):
        """Canonical plasma current path must appear."""
        result = _search_imas("plasma current", gc=graph_client, encoder=encoder)
        assert "equilibrium/time_slice/global_quantities/ip" in result

    def test_no_verbose_description_noise(self, graph_client, encoder):
        """Generic 'Verbose description' paths should not dominate results."""
        result = _search_imas("plasma current", gc=graph_client, encoder=encoder)
        verbose_count = result.count('"Verbose description"')
        assert verbose_count <= 2, f"Too many 'Verbose description' results: {verbose_count}"

    def test_clusters_deduplicated(self, graph_client, encoder):
        """Same cluster should not appear 3 times with different scopes."""
        result = _search_imas("electron temperature", gc=graph_client, encoder=encoder)
        cluster_lines = [l for l in result.split("\n") if "score:" in l and "Scope:" in l]
        labels = [l.split('"')[1] for l in cluster_lines if '"' in l]
        # Each label should appear at most once
        assert len(labels) == len(set(labels)), f"Duplicate clusters: {labels}"


class TestFetch:
    """Test fetch tool correctness."""

    def test_fetch_code_file(self, graph_client):
        """Fetch should resolve code files via FacilityPath."""
        # Use a known FacilityPath that has PRODUCED a CodeExample
        result = _fetch("tcv:some_known_path", gc=graph_client)
        assert "No resource found" not in result


class TestVectorSearchInternals:
    """Test vector search helper functions."""

    def test_facility_filter_uses_property(self, graph_client, encoder):
        """Facility filter should use cc.facility_id, not relationship."""
        embedding = encoder.embed_texts(["test"])[0].tolist()
        ids, scores = _vector_search_code_chunks(graph_client, embedding, "tcv", 5)
        # Should return results (was returning 0 before fix)
        assert len(ids) > 0

    def test_over_fetch_prevents_starvation(self, graph_client, encoder):
        """Minority facility should get results with over-fetching."""
        embedding = encoder.embed_texts(["magnetic probe"])[0].tolist()
        ids, scores = _vector_search_signals(
            graph_client, embedding, "jet", k=10
        )
        assert len(ids) > 0
```

---

## 6. Relationship Between Graph Schema and Tool Queries

### Verified Correct Relationships
- `FacilitySignal -[:DATA_ACCESS]-> DataAccess` ✓
- `FacilitySignal -[:BELONGS_TO_DIAGNOSTIC]-> Diagnostic` ✓
- `FacilitySignal -[:AT_FACILITY]-> Facility` ✓
- `WikiPage -[:HAS_CHUNK]-> WikiChunk` ✓
- `WikiChunk -[:AT_FACILITY]-> Facility` ✓
- `IMASPath -[:HAS_UNIT]-> Unit` ✓
- `IMASPath -[:IN_CLUSTER]-> IMASSemanticCluster` ✓
- `IMASPath -[:INTRODUCED_IN]-> DDVersion` ✓
- `DataAccess -[:MAPS_TO_IMAS]-> IMASPath` ✓

### Broken Relationships (Used in Tools but Don't Exist)
- `CodeFile -[:HAS_CHUNK]-> CodeChunk` ✗ (should be: `FacilityPath -[:PRODUCED]-> CodeExample -[:HAS_CHUNK]-> CodeChunk`)
- `CodeFile -[:PRODUCED]-> CodeExample` ✗ (should be: `FacilityPath -[:PRODUCED]-> CodeExample`)
- `CodeFile -[:CONTAINS_REF]-> DataReference` ✗ (should be: `CodeChunk -[:CONTAINS_REF]-> DataReference`)
- `FacilitySignal -[:HAS_DATA_SOURCE_NODE]-> DataNode` — exists but 0 matches found

### Relationship Query Compatibility Matrix

| Tool Function | Relationship Used | Exists? | Fix Needed? |
|---------------|-------------------|---------|-------------|
| `_vector_search_signals` | `AT_FACILITY` | ✓ | Over-fetch k |
| `_enrich_signals` | `DATA_ACCESS`, `BELONGS_TO_DIAGNOSTIC`, `HAS_DATA_SOURCE_NODE`, `MAPS_TO_IMAS` | Mostly ✓ | Add signal properties |
| `_vector_search_wiki_chunks` | `AT_FACILITY` | ✓ | Over-fetch k |
| `_enrich_wiki_chunks` | `HAS_CHUNK`, `DOCUMENTS`, `MENTIONS_IMAS` | ✓ | None |
| `_vector_search_documents` | `AT_FACILITY` | ✓ | Fix index name |
| `_vector_search_code_chunks` | `CodeFile-[:HAS_CHUNK]` | ✗ | Use property filter |
| `_enrich_code_chunks` | `CodeExample-[:HAS_CHUNK]`, `CodeFile.PRODUCED`, `CodeFile-[:CONTAINS_REF]` | All ✗ | Major rewrite |
| `_fetch_code_file` | `CodeFile-[:PRODUCED]->CodeExample-[:HAS_CHUNK]` | ✗ | Use FacilityPath |
| `_vector_search_imas_paths` | `DEPRECATED_IN` | ✓ | Add text search |
| `_enrich_imas_paths` | `HAS_UNIT`, `IN_CLUSTER`, `HAS_COORDINATE`, `INTRODUCED_IN` | ✓ | Add more properties |

---

## 7. Files to Modify

| File | Changes |
|------|---------|
| `imas_codex/agentic/search_tools.py` | Fix all vector search functions, fix index name, fix code enrichment, fix fetch |
| `imas_codex/agentic/search_formatters.py` | Deduplicate signals, interpolate templates, deduplicate clusters |
| `tests/agentic/test_search_tools.py` | New — regression tests for all fixes |
| `tests/agentic/test_search_formatters.py` | New — unit tests for formatting logic |

---

## 8. Priority Order

1. **Fix document index name** — 1 line change, immediately doubles JET docs coverage
2. **Fix code search schema** — unblocks all code search
3. **Fix facility starvation** — unblocks minority facility search
4. **Fix fetch CodeFile** — unblocks content retrieval
5. **Deduplicate signals** — improve output quality
6. **Interpolate templates** — make results actionable
7. **Add hybrid search to IMAS** — match old server quality
8. **Create unit tests** — prevent regressions
9. **Enrich IMAS results** — feature parity with old server
10. **Full-text indexes** — advanced search capability
