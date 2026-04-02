# IMAS-DD MCP Server Improvements

## Context

A/B testing of the `imas` and `imas-dd` MCP servers (59+ test cases, 20 ground-truth
search queries) revealed the imas-dd server wins on metadata richness, completeness,
and unique features but has critical gaps in search recall (MRR@10 = 0.195),
broken DD version filtering, and an outdated migration guide design.

This plan addresses every identified gap and designs improvements suitable
for parallel agent implementation.

---

## Phase 1 — Critical Bug Fixes

Independent items. All can be implemented in parallel.

### 1a. Fix `dd_version` integer handling

**Problem:** `search_imas`, `check_imas_paths`, and other tools crash with
`'int' object has no attribute 'split'` when `dd_version` is passed as an integer.

**Root cause:** The deployed container at `imas-dd.iter.org` runs an older
version. The local `_dd_version_clause()` in `imas_codex/tools/graph_search.py:74`
correctly generates Cypher that uses `toInteger(split(iv.id, '.')[0]) <= $dd_major_version`
where `$dd_major_version` is the integer param. However, commit `25b7e6ab` added
`resolve_dd_version()` which accepts flexible input (`int | str | "latest"`) but
is NOT yet integrated into main.

**Implementation:**
1. Add `resolve_dd_version()` to `imas_codex/tools/graph_search.py` (or a shared utils module)
2. Call it at the entry point of every tool that accepts `dd_version` BEFORE
   passing to `_dd_version_clause()`
3. Accept `int` (major), `str` ("3.39.0", "4", "latest"), or `None`
4. Resolution: `4` → latest 4.x.y semver, `"3.39.0"` → exact, `"latest"` → `is_current=true`
5. Deploy updated container

**Files:**
- `imas_codex/tools/graph_search.py` — add resolver, call in all tool entry points
- `tests/graph_mcp/test_graph_search.py` — add `dd_version=3` and `dd_version=4` tests
- `tests/tools/test_dd_version_filtering.py` — expand parametrized tests

**Test cases:**
```python
@pytest.mark.parametrize("dd_version", [3, 4, "3.39.0", "4.0.0", "latest", None])
async def test_search_with_dd_version(self, graph_client, dd_version):
    tool = GraphSearchTool(graph_client)
    result = await tool.search_imas_paths("temperature", dd_version=dd_version)
    assert isinstance(result, SearchPathsResult)
```

### 1b. Fix `_get_version_cocos()` property name bug

**Problem:** `imas_codex/tools/migration_guide.py:220` queries `v.cocos_id` but
the DDVersion schema uses `v.cocos`. The fallback logic masks this — COCOS factors
show `?` for intra-3.x migrations.

**Implementation:**
1. Fix the Cypher query: `v.cocos_id` → `v.cocos`
2. Verify DDVersion nodes for 3.35.0+ have `cocos=11` and 4.x have `cocos=17`
3. Add regression test

**Files:**
- `imas_codex/tools/migration_guide.py:220`

### 1c. Wire `PathFuzzyMatcher` into `check_imas_paths`

**Problem:** `check_imas_paths` only checks `RENAMED_TO` edges for not-found paths.
The `PathFuzzyMatcher` class exists (`imas_codex/search/fuzzy_matcher.py`) with
`rapidfuzz`-based typo correction but is **never called** from `check_imas_paths`.

**Implementation:**
1. In `check_imas_paths()` (`graph_search.py:440`), when a path is not found
   AND no `RENAMED_TO` edge exists, call `PathFuzzyMatcher.suggest_paths()`
2. Populate `CheckPathsResultItem.suggestion` with the fuzzy match
3. Validate IDS name before prepending (call `suggest_ids()` for invalid IDS)
4. Add IDS existence validation for the `ids` parameter

**Files:**
- `imas_codex/tools/graph_search.py` — integrate fuzzy matcher
- `tests/graph_mcp/test_graph_search.py` — add typo correction tests

**Test cases:**
```python
@pytest.mark.parametrize("typo,expected_suggestion", [
    ("equilibrium/time_slice/profiles_1d/psi_boundary", "equilibrium/time_slice/profiles_1d/psi"),
    ("core_profiles/profiles_1d/electons/temperature", "core_profiles/profiles_1d/electrons/temperature"),
    ("magnetics/flux_lop/flux/data", "magnetics/flux_loop/flux/data"),
])
async def test_check_suggests_correction(self, graph_client, typo, expected_suggestion):
    tool = GraphPathTool(graph_client)
    result = await tool.check_imas_paths(typo)
    assert result.results[0].exists is False
    assert result.results[0].suggestion is not None
```

### 1d. Fix `include_version_history` in `fetch_imas_paths`

**Problem:** The parameter is fully implemented in code (queries `IMASNodeChange`
nodes), but the A/B test found no version history in output. This is likely
because the deployed container has sparse `IMASNodeChange` data — only
COCOS-affected paths have change records.

**Investigation needed:**
1. Verify `IMASNodeChange` node count in production graph
2. Check if change records exist for non-COCOS paths
3. If data is sparse, this is a data population issue, not a code bug

**Files:**
- `imas_codex/tools/graph_search.py:493-508` — verify code path
- `tests/graph_mcp/test_graph_search.py` — existing test `test_fetch_version_history_enabled`
  passes with fixture data

**Action:** Verify deployed data; if sparse, add to Phase 5 (data population).

---

## Phase 2 — Search Quality (MRR Improvement)

This is the critical phase. Current MRR@10 = 0.195 is unacceptable.
The IMAS DD is finite (~5000 data nodes); good search is entirely within reach.

### Architecture: Accessor Node Surfacing

**Core design decision:** Metadata child nodes (`data`, `value`, `error_upper`,
`r`, `z`, `phi`, `time`, etc.) are deliberately excluded from search indexes
to prevent dilution. This is correct — there are thousands of these nodes and
they would swamp real results.

However, when a user explicitly searches for a `*/data` or `*/value` path,
we must still surface it. The strategy:

1. **Index only "concept" nodes** (parent nodes with physics meaning) — current behavior, keep it
2. **Pattern-match accessor suffixes** in the query at search time
3. **Traverse to children** via graph relationships to surface the actual leaf node
4. **Return the leaf path** in results with the parent's description enriched

### 2a. Query analysis and accessor resolution

**Problem:** Queries like "plasma current" should find `equilibrium/time_slice/
global_quantities/ip` but also know that the actual data is at `ip/data` or `ip/value`.
Queries like "flux_loop/flux/data" explicitly target a `*/data` leaf.

**Implementation:**
1. Add a `QueryAnalyzer` class in `imas_codex/tools/query_analysis.py`:
   ```python
   class QueryAnalyzer:
       ACCESSOR_TERMINALS = {"data", "value", "time", "r", "z", "phi",
                             "coefficients", "label", "grid_index",
                             "measured", "reconstructed", "parallel",
                             "toroidal", "perpendicular"}
       
       def analyze(self, query: str) -> QueryIntent:
           """Classify query and extract accessor hints."""
           # Detect if query ends with accessor terminal
           # Detect path-like queries (contains "/")
           # Detect abbreviations (ip, te, ti, ne, q)
           # Return QueryIntent with:
           #   - query_type: "path_exact", "path_partial", "concept", "hybrid"
           #   - accessor_hint: Optional[str] (e.g., "data", "value")
           #   - expanded_terms: list[str] (e.g., "ip" → ["plasma current", "ip"])
           #   - path_segments: list[str] (extracted from path-like queries)
   ```

2. In `search_imas_paths()`, call `QueryAnalyzer.analyze()` first
3. Use the result to adjust search strategy:
   - **path_exact**: Do exact match + ancestor match (strip trailing accessor)
   - **path_partial**: Do CONTAINS match on `p.id`
   - **concept**: Standard hybrid search (current)
   - **hybrid**: Combine concept search with path filtering

**Files:**
- `imas_codex/tools/query_analysis.py` — new module
- `imas_codex/tools/graph_search.py` — integrate into `search_imas_paths()`
- `tests/tools/test_query_analysis.py` — comprehensive unit tests

### 2b. Path-mode search (exact and partial matching)

**Problem:** Path queries like `time_slice/profiles_1d/psi` and
`equilibrium/time_slice/global_quantities/ip` should be handled as
structural lookups, not semantic searches.

**Implementation:**
1. When `QueryAnalyzer` detects a path-like query, route to a dedicated
   path-matching function:
   ```python
   def _path_search(gc, query, max_results, ids_filter, dd_version):
       """Search by path structure — exact, prefix, and substring matching."""
       params = {"query": query, "query_lower": query.lower(), "limit": max_results}
       
       # Strategy 1: Exact match (highest score)
       exact = gc.query("""
           MATCH (p:IMASNode {id: $query})
           WHERE p.node_category = 'data'
           RETURN p.id AS id, 1.0 AS score
       """, **params)
       
       # Strategy 2: Suffix match — query is a partial path
       suffix = gc.query("""
           MATCH (p:IMASNode)
           WHERE p.node_category = 'data'
             AND p.id ENDS WITH $query_lower
           RETURN p.id AS id, 0.95 AS score
           LIMIT $limit
       """, **params)
       
       # Strategy 3: Contains match
       contains = gc.query("""
           MATCH (p:IMASNode)
           WHERE p.node_category = 'data'
             AND p.id CONTAINS $query_lower
           RETURN p.id AS id, 0.85 AS score
           LIMIT $limit
       """, **params)
       
       # Strategy 4: If query ends with accessor terminal,
       # strip it and search for parent, then re-attach
       stripped = strip_accessor_suffix(query)
       if stripped != query:
           parent_results = _path_search(gc, stripped, ...)
           # For each parent, check if child with accessor name exists
           # Return child path with parent's score
   ```

2. Merge path-search results with vector+text results using score-based ranking

**Files:**
- `imas_codex/tools/graph_search.py` — add `_path_search()`, integrate
- `tests/tools/test_path_search.py` — parametrized tests

**Test cases (from ground truth):**
```python
PATH_SEARCH_TESTS = [
    # (query, expected_top1)
    ("time_slice/profiles_1d/psi", "equilibrium/time_slice/profiles_1d/psi"),
    ("profiles_1d/electrons/temperature", "core_profiles/profiles_1d/electrons/temperature"),
    ("flux_loop/flux/data", "magnetics/flux_loop/flux/data"),
    ("equilibrium/time_slice/global_quantities/ip", "equilibrium/time_slice/global_quantities/ip"),
    ("core_profiles/profiles_1d/electrons/density", "core_profiles/profiles_1d/electrons/density"),
    ("nbi/unit/power_launched/data", "nbi/unit/power_launched/data"),
]
```

### 2c. Physics abbreviation expansion

**Problem:** Common fusion abbreviations (Ip, Te, Ti, ne, q, psi, B_tor) are
not expanded to their full physics terms, limiting semantic and keyword search.

**Implementation:**
1. Add abbreviation map to `QueryAnalyzer`:
   ```python
   PHYSICS_ABBREVIATIONS = {
       "ip": ["plasma current", "ip"],
       "te": ["electron temperature", "te"],
       "ti": ["ion temperature", "ti"],
       "ne": ["electron density", "ne"],
       "ni": ["ion density", "ni"],
       "bt": ["toroidal magnetic field", "b_field_tor", "bt", "b0"],
       "bp": ["poloidal magnetic field", "b_field_pol", "bp"],
       "q": ["safety factor", "q"],
       "psi": ["poloidal flux", "psi"],
       "beta": ["plasma beta", "beta_pol", "beta_tor", "beta_normal"],
       "li": ["internal inductance", "li"],
       "wmhd": ["stored energy", "w_mhd"],
   }
   ```

2. When a single-word query matches an abbreviation, expand it
3. Run parallel searches: one with original term, one with expanded terms
4. Merge results with bonus for hits matching both

**Files:**
- `imas_codex/tools/query_analysis.py` — abbreviation map
- `imas_codex/tools/graph_search.py` — integrate expansion
- `tests/tools/test_query_analysis.py` — expansion tests

### 2d. Improved score combination

**Problem:** Current scoring is `max(vector, text) + 0.05` with a weak
path-segment boost of `+0.03 * count`. This underweights strong lexical
matches and path structure.

**Implementation:**
1. Increase path-segment boost: `0.03` → `0.08` per matching segment
2. Add exact-name boost: if query word matches terminal node name exactly, +0.15
3. Add IDS-name boost: if query contains the IDS name, +0.10 for paths in that IDS
4. Use weighted combination instead of max:
   ```python
   if pid in vector_scores and pid in text_scores:
       combined = 0.6 * vector_scores[pid] + 0.4 * text_scores[pid] + 0.05
   ```
5. Fix BM25 normalization: current `max(raw, 0.7)` floor compresses scores.
   Change to: `max(0.3 + 0.7 * raw, 0.0)` for better discrimination

**Files:**
- `imas_codex/tools/graph_search.py:207-225` — scoring section

### 2e. Ground truth regression test suite

**Problem:** Need to guard against regressions while improving MRR.

**Implementation:**
1. Add 20 A/B test queries to `tests/search/benchmark_data.py`
2. These supplement the existing 30 queries for a total of 50
3. Tests assert minimum MRR thresholds per category

**New test data (add to `benchmark_data.py`):**
```python
AB_TEST_QUERIES = [
    # Keyword queries
    BenchmarkQuery("electron temperature", ["core_profiles/profiles_1d/electrons/temperature"], "keyword"),
    BenchmarkQuery("plasma current", ["equilibrium/time_slice/global_quantities/ip"], "keyword"),
    BenchmarkQuery("magnetic field", ["magnetics/b_field_tor_probe/field/data"], "keyword"),
    BenchmarkQuery("ion density", ["core_profiles/profiles_1d/ion/density"], "keyword"),
    BenchmarkQuery("safety factor", ["core_profiles/profiles_1d/q"], "keyword"),
    # Path queries (no prefix)
    BenchmarkQuery("time_slice/profiles_1d/psi", ["equilibrium/time_slice/profiles_1d/psi"], "path_no_prefix"),
    BenchmarkQuery("profiles_1d/electrons/temperature", ["core_profiles/profiles_1d/electrons/temperature"], "path_no_prefix"),
    BenchmarkQuery("flux_loop/flux/data", ["magnetics/flux_loop/flux/data"], "path_no_prefix"),
    # Path queries (with prefix)
    BenchmarkQuery("equilibrium/time_slice/global_quantities/ip", ["equilibrium/time_slice/global_quantities/ip"], "path_with_prefix"),
    BenchmarkQuery("core_profiles/profiles_1d/electrons/density", ["core_profiles/profiles_1d/electrons/density"], "path_with_prefix"),
    BenchmarkQuery("nbi/unit/power_launched/data", ["nbi/unit/power_launched/data"], "path_with_prefix"),
    # Semantic queries
    BenchmarkQuery("where is the plasma boundary shape stored", ["equilibrium/time_slice/boundary/outline/r"], "semantic"),
    BenchmarkQuery("toroidal magnetic field strength", ["equilibrium/time_slice/global_quantities/magnetic_axis/b_field_tor"], "semantic"),
    BenchmarkQuery("neutral beam injection power", ["nbi/unit/power_launched/data"], "semantic"),
    BenchmarkQuery("plasma elongation", ["equilibrium/time_slice/boundary/elongation"], "semantic"),
    BenchmarkQuery("poloidal flux profile", ["equilibrium/time_slice/profiles_1d/psi"], "semantic"),
    # Hybrid queries
    BenchmarkQuery("electron temperature core_profiles", ["core_profiles/profiles_1d/electrons/temperature"], "hybrid"),
    BenchmarkQuery("psi equilibrium profiles", ["equilibrium/time_slice/profiles_1d/psi"], "hybrid"),
    BenchmarkQuery("beta_pol global", ["core_profiles/global_quantities/beta_pol"], "hybrid"),
    BenchmarkQuery("wall limiter outline geometry", ["wall/description_2d/limiter/unit/outline/r"], "hybrid"),
]
```

**Thresholds (post-improvement targets):**
```python
assert overall_mrr >= 0.60  # Up from 0.195
assert keyword_mrr >= 0.70
assert path_with_prefix_mrr >= 0.95  # Exact matches must work
assert path_no_prefix_mrr >= 0.80
assert semantic_mrr >= 0.40
assert hybrid_mrr >= 0.50
```

**Files:**
- `tests/search/benchmark_data.py` — add 20 queries
- `tests/search/test_search_benchmarks.py` — add threshold assertions

---

## Phase 3 — Relationship & Cluster Improvements

### 3a. Add semantic vector search to `find_related_imas_paths`

**Problem:** `get_imas_path_context()` (`graph_search.py:1452`) only uses
graph relationships (cluster siblings, coordinate partners, unit companions,
identifier links). It has NO vector similarity search despite the docstring
promising it. Paths without cluster membership return 0 results.

**Implementation:**
1. Add a semantic relationship type that uses vector search:
   ```python
   # In get_imas_path_context(), add as first relationship type:
   if "semantic" in requested_types or "all" in requested_types:
       # Get the path's embedding
       path_embedding = gc.query("""
           MATCH (p:IMASNode {id: $path})
           RETURN p.embedding AS embedding
       """, path=path)
       
       if path_embedding and path_embedding[0]["embedding"]:
           semantic_results = gc.query("""
               CALL db.index.vector.queryNodes(
                   'imas_node_embedding', $k, $embedding
               ) YIELD node AS similar, score
               WHERE similar.id <> $path
                 AND similar.ids <> $source_ids
                 AND similar.node_category = 'data'
                 AND score > 0.5
               RETURN similar.id AS path, similar.ids AS ids,
                      score, 'semantic' AS relationship_type
               LIMIT $limit
           """, embedding=path_embedding[0]["embedding"],
                path=path, source_ids=source_ids, k=50, limit=max_results)
   ```

2. This ensures every path gets related results even without cluster membership

**Files:**
- `imas_codex/tools/graph_search.py:1452+` — add semantic search branch
- `tests/graph_mcp/test_graph_search.py` — test vector-based relationships

### 3b. Filter generic coordinate matches

**Problem:** Coordinate partner matching is polluted by generic specs like `1...N`.
The docstring promises filtering but it's not implemented.

**Implementation:**
1. Add exclusion list for generic coordinate specs:
   ```python
   GENERIC_COORDINATES = {"1...N", "1...3", "-1", "1", "1...2"}
   ```

2. Add WHERE clause to coordinate query:
   ```cypher
   MATCH (p:IMASNode {id: $path})-[:HAS_COORDINATE]->(coord:IMASCoordinateSpec)
         <-[:HAS_COORDINATE]-(sibling:IMASNode)
   WHERE sibling.ids <> p.ids
     AND NOT (coord.id IN $generic_coords)
   ```

**Files:**
- `imas_codex/tools/graph_search.py:1484-1500` — add coordinate filter

### 3c. Improve cluster granularity

**Problem:** "plasma current" returns a single mega-cluster ("Global Plasma Physics Data"
with 50+ paths). Clusters are too broad for specific concepts.

**Investigation:** The HDBSCAN clustering uses `method="eom"` (Excess of Mass)
which produces broader clusters. The `min_cluster_size=2` is appropriate.

**Implementation:**
1. After retrieving clusters, add a relevance re-ranking step:
   - For each cluster, compute overlap between query terms and cluster label/paths
   - Penalize clusters with path_count > 30 (too broad)
   - Boost clusters where label closely matches query
2. Consider adding a `max_cluster_size` filter parameter

**Files:**
- `imas_codex/tools/graph_search.py:1167-1190` — add post-retrieval re-ranking

---

## Phase 4 — Migration Guide Redesign (Code Migration)

This is a significant redesign. The current tool produces Python data-migration
snippets. The new tool should produce **language-agnostic code migration advice**
that enables any agent to update source code in any language.

### Design Principles

1. **Code migration, not data migration** — imas-python already handles data migration.
   This tool advises how to update *codes* that read/write IMAS data.
2. **Required vs optional updates** — distinguish between changes that WILL break
   a code and changes that are best-practice improvements.
3. **No NBC reliance** — when updating a code fully to a new DD version, update
   ALL paths, not just breaking changes. The non-breaking-change (NBC) compatibility
   layer in the IMAS access layer should not carry the burden.
4. **Search patterns** — provide grep/search patterns that locate affected code
5. **Language agnostic** — output structured data, not Python snippets

### 4a. New output format: `CodeMigrationGuide`

Replace the current markdown string output with a structured Pydantic model:

```python
class CodeUpdateAction:
    """A single code update action."""
    path: str                          # IMAS path affected
    ids: str                           # IDS name
    change_type: str                   # "cocos_sign_flip", "type_change", "path_rename",
                                       # "path_removed", "unit_change", "new_path"
    severity: str                      # "required" or "optional"
    
    # Search patterns — language-agnostic patterns to find affected code
    search_patterns: list[str]         # e.g., ["profiles_1d.*psi", "get.*psi", "put.*psi"]
    path_fragments: list[str]          # Path segments to search for: ["profiles_1d", "psi"]
    
    # What to change
    description: str                   # Human-readable description
    before: str                        # What the code does now
    after: str                         # What the code should do
    
    # For COCOS sign flips
    cocos_label: str | None            # e.g., "psi_like"
    cocos_factor: float | None         # e.g., -1.0
    
    # For path renames
    old_path: str | None
    new_path: str | None
    
    # For type changes
    old_type: str | None
    new_type: str | None

class CodeMigrationGuide:
    """Complete code migration guide between DD versions."""
    from_version: str
    to_version: str
    cocos_change: str | None           # e.g., "11 → 17"
    
    # Update mode
    full_update: bool                  # True = update ALL paths (no NBC reliance)
    
    # Grouped actions
    required_actions: list[CodeUpdateAction]   # MUST change or code breaks
    optional_actions: list[CodeUpdateAction]   # SHOULD change for full compliance
    
    # Summary statistics
    total_actions: int
    required_count: int
    optional_count: int
    ids_affected: list[str]
    
    # Search strategy — how to find affected code sections
    global_search_patterns: dict[str, list[str]]  # Grouped by IDS
    
    # Structured advice sections
    cocos_advice: CocosMigrationAdvice | None
    path_update_advice: PathUpdateAdvice | None
    type_update_advice: TypeUpdateAdvice | None

class CocosMigrationAdvice:
    """COCOS-specific migration advice."""
    from_cocos: int
    to_cocos: int
    
    # Categorized by action needed
    sign_flips: list[dict]             # Paths needing * -1
    no_change: list[dict]              # Paths with factor 1 (verify only)
    
    # Search patterns for COCOS-affected code
    access_patterns: dict[str, str]    # Language → pattern template
    # e.g., {"fortran": "CALL ids_get(ids, '{path}'",
    #        "c++":     "ids.{cpp_accessor}",
    #        "python":  "ids.{python_accessor}"}

class PathUpdateAdvice:
    """Path-level update advice with search patterns."""
    renamed_paths: list[dict]          # old_path → new_path
    removed_paths: list[dict]          # path + suggested_replacement
    new_paths: list[dict]              # Newly available paths
    
    # For each rename/removal, provide search patterns:
    # - Exact path string: "equilibrium/time_slice/profiles_1d/old_name"
    # - Path segments: ["profiles_1d", "old_name"]
    # - Common accessor patterns per language
```

### 4b. Search pattern generation

For each affected path, generate language-agnostic search patterns:

```python
def generate_search_patterns(path: str, change_type: str) -> list[str]:
    """Generate search patterns to find code that accesses this path."""
    segments = path.split("/")
    ids_name = segments[0]
    leaf_name = segments[-1]
    parent_name = segments[-2] if len(segments) > 1 else ""
    
    patterns = [
        # Exact path string (any language)
        f"'{'/'.join(segments[1:])}'",
        f'"{"/".join(segments[1:])}"',
        
        # Leaf name in accessor context
        f"{parent_name}.*{leaf_name}",
        f"{leaf_name}.*{parent_name}",
        
        # IDS-level patterns
        f"{ids_name}.*{leaf_name}",
        
        # Common IMAS access patterns
        f"get.*{leaf_name}",
        f"put.*{leaf_name}",
        f"ids_get.*{leaf_name}",
        f"ids_put.*{leaf_name}",
    ]
    return patterns
```

### 4c. Full-update mode (no NBC reliance)

When `full_update=True`:
1. Enumerate ALL paths in the target DD version
2. For each path, determine if its semantics, units, type, or COCOS
   convention differ from the source version
3. Generate update actions even for "non-breaking" changes:
   - Path documentation clarifications → optional (verify comments)
   - COCOS factor=1 paths → optional (verify sign handling)
   - New paths → optional (consider adoption)
   - All path renames → required (even if NBC layer handles it)

### 4d. Rendering pipeline

The structured `CodeMigrationGuide` is rendered to markdown by the MCP
tool's format function. The structured data is also available via the
`include_structured=True` parameter for programmatic consumption.

Rendering sections:
1. **Executive Summary** — versions, COCOS change, action counts
2. **Required Updates** — grouped by IDS, sorted by severity
3. **Search Strategy** — how to find affected code (patterns per IDS)
4. **COCOS Migration** — sign-flip table with search patterns
5. **Path Updates** — renames, removals, type changes with patterns
6. **Optional Improvements** — new paths, documentation changes
7. **Verification Checklist** — how to verify each change was applied

**Files:**
- `imas_codex/tools/migration_guide.py` — complete rewrite
- `imas_codex/models/migration_models.py` — new Pydantic models
- `imas_codex/llm/search_formatters.py` — new `format_migration_report()`
- `tests/tools/test_migration_guide.py` — comprehensive tests

---

## Phase 5 — Polish & Missing Features

### 5a. Populate `IMASNodeChange` data for non-COCOS paths

**Problem:** Version history is sparse — only COCOS-affected paths have change
records. Unit changes, coordinate changes, and documentation updates are untracked.

**Implementation:**
1. Extend DD build pipeline (`build_dd.py`) to diff consecutive DD versions
2. For each path, detect: added, removed, type changed, units changed,
   documentation changed, coordinate changed
3. Store as `IMASNodeChange` nodes with appropriate `semantic_change_type`

**Files:**
- `imas_codex/graph/build_dd.py` — extend version diffing

### 5b. Populate path rename tracking

**Problem:** All migration guides show 0 renames. Either there are genuinely no
renames in the DD history, or the detection pipeline is empty.

**Investigation:** Check DD XML changelogs for actual renames between versions.

### 5c. Add total path counts to `list_imas_paths`

**Problem:** Server B doesn't report total IDS size when truncating results.

**Implementation:** Add `total_paths` to the return dict:
```python
total = gc.query("MATCH (p:IMASNode) WHERE p.ids = $ids RETURN count(p) AS cnt", ids=ids)
return {"paths": [...], "total_paths": total[0]["cnt"], "truncated": len(paths) < total}
```

**Files:**
- `imas_codex/tools/graph_search.py` — add count query to list function

### 5d. Port `explain_concept` from imas server

**Problem:** The imas server's `explain_concept` produced exceptionally good
COCOS explanations but timed out 75% of the time. The imas-dd server has no
equivalent.

**Implementation:**
1. Add a `explain_concept` tool backed by graph data (not LLM):
   - Query cluster descriptions for the concept
   - Query COCOS metadata for convention explanations
   - Query identifier schemas for enumeration explanations
   - Compose a structured explanation from graph data
2. This avoids the LLM timeout issue while providing domain-specific context

**Files:**
- `imas_codex/tools/graph_search.py` — add `explain_concept` method
- `imas_codex/llm/server.py` — register MCP tool

---

## Implementation Order & Parallelism

```
Phase 1 (all parallel):
  ├── 1a: dd_version fix ─────────────┐
  ├── 1b: cocos property fix ─────────┤
  ├── 1c: fuzzy matcher wiring ───────┤
  └── 1d: version_history investigation┘
                                       │
Phase 2 (sequential within, parallel with Phase 3):
  ├── 2a: QueryAnalyzer ──────────────┐ (parallel with Phase 3)
  ├── 2b: path-mode search ───────────┤ (depends on 2a)
  ├── 2c: abbreviation expansion ─────┤ (depends on 2a)
  ├── 2d: score combination ──────────┤ (independent)
  └── 2e: ground truth tests ─────────┘ (independent, run first as TDD)
                                       │
Phase 3 (parallel with Phase 2):       │
  ├── 3a: semantic search in related ──┤
  ├── 3b: coordinate filtering ────────┤
  └── 3c: cluster re-ranking ──────────┘
                                       │
Phase 4 (after Phase 1):              │
  ├── 4a: migration models ───────────┐
  ├── 4b: search pattern generation ──┤ (depends on 4a)
  ├── 4c: full-update mode ───────────┤ (depends on 4a)
  └── 4d: rendering pipeline ─────────┘ (depends on 4a-4c)
                                       │
Phase 5 (after Phase 2):              │
  ├── 5a: IMASNodeChange population ───┤
  ├── 5b: rename tracking ────────────┤
  ├── 5c: total path counts ──────────┤
  └── 5d: explain_concept ────────────┘
```

## Agent Assignment Strategy

| Phase | Parallelizable Items | Estimated Complexity |
|-------|---------------------|---------------------|
| 1a | Independent | Small (10-20 lines + tests) |
| 1b | Independent | Tiny (1 line fix + test) |
| 1c | Independent | Medium (30 lines + tests) |
| 1d | Investigation | Small |
| 2a | Independent | Medium (new module) |
| 2b | Depends on 2a | Medium (new function) |
| 2c | Depends on 2a | Small (data + integration) |
| 2d | Independent | Small (scoring tweaks) |
| 2e | Independent, do first (TDD) | Medium (test data) |
| 3a | Independent | Medium (new search branch) |
| 3b | Independent | Small (WHERE clause) |
| 3c | Independent | Small (re-ranking logic) |
| 4a-4d | Sequential | Large (new module + models) |
| 5a-5d | Independent | Medium each |

**Maximum parallelism:** 4 agents in Phase 1, then 2-3 agents across Phases 2+3,
then 1 agent for Phase 4, then 2-3 agents for Phase 5.
