# IMAS Tool Quality — Closing the Gap

**Goal:** Bring Codex MCP's IMAS DD tools to feature-parity with (and beyond) the old standalone IMAS MCP server, with fixes applied in the backing `imas_codex/tools/` functions so all clients benefit.

**Status:** Planning  
**Date:** 2026-03-14

---

## Diagnostic Summary

Comparison of 12 IMAS tool categories across Codex MCP and old IMAS MCP with direct graph validation revealed 5 significant gaps:

| # | Gap | Severity | Root Cause |
|---|-----|----------|------------|
| 1 | Cluster semantic search returns 0 results | **Critical** | `_search_by_text()` queries `cluster_description_embedding` index, but 0/3635 clusters have `description_embedding` populated. DD build pipeline creates centroid `embedding` (step 8) but LLM labeling (step 9) requires `label IS NOT NULL` — and labels are never generated because the `ClusterLabeler` is only wired in the old file-based `extractor.py`, not the graph-native `_import_clusters()` in `build_dd.py`. |
| 2 | `export_imas_domain` returns 0 for common names | **High** | Tool queries `WHERE p.physics_domain = $domain` using exact match. Graph stores canonical domain names like `magnetic_field_diagnostics`, but users pass friendly names like `magnetics`. No alias resolution or fuzzy matching. |
| 3 | `_normalize_paths` fragile with list inputs | **Medium** | `_normalize_paths()` only handles `str` (space/comma split) or `list[str]`. When MCP clients serialize JSON arrays as strings (e.g., `'["path1","path2"]'`), the brackets and quotes become part of the path strings. |
| 4 | `fetch_imas_paths` missing metadata vs old server | **Medium** | Old server returns coordinates, lifecycle, structure_reference, physics_context, validation_rules. Codex fetches most of this from graph but doesn't return coordinates in leaf-level format, no lifecycle info, no introduced_after_version. |
| 5 | `get_imas_identifiers` keyword matching limited | **Low** | Query filtering matches against `name`, `description`, and `options` text, but `description` is always empty in the graph (62 schemas, 0 with descriptions). Search for "cocos" finds nothing because identifier names like `cocos_identifier` don't fuzzy-match. |

### Graph Data Inventory (validated via Cypher)

| Entity | Count | Notes |
|--------|-------|-------|
| IMASNode | 61,366 | All with physics_domain, 22,181 with embeddings |
| IMASSemanticCluster | 3,635 | All with centroid `embedding` (256-dim), **0 with label/description/label_embedding/description_embedding** |
| IN_CLUSTER relationships | 22,141 | Avg 6.1 members/cluster, min 2, max 168 |
| DDVersion | 35 | 3.22.0 → 4.1.0, current = 4.1.0 |
| INTRODUCED_IN rels | 61,366 | 100% coverage |
| HAS_UNIT rels | 44,923 | 144 distinct Units |
| IdentifierSchema | 62 | Options stored as inline JSON, **0 with description** |
| IMASCoordinateSpec | 8 | |
| IDS nodes | 81 | Each with physics_domain |
| Vector indexes | 3 cluster indexes | `cluster_embedding` (centroid), `cluster_label_embedding` (empty), `cluster_description_embedding` (empty) |

### Cluster Search Failure Chain

```
User query "safety factor profile"
  → _search_by_text() embeds query → 256-dim vector
  → Queries 'cluster_description_embedding' index
  → Index is on IMASSemanticCluster.description_embedding property
  → 0/3635 clusters have description_embedding populated
  → Returns 0 results
```

**Fix chain:** The `_import_clusters()` pipeline in `build_dd.py`:
- Step 5: MERGE cluster nodes — sets `path_count`, `cross_ids`, `similarity_score`, `scope`, `ids_names`. Does NOT set `label` or `description`.
- Step 8: Computes centroid `embedding` from member IMASNode embeddings. This populates the `cluster_embedding` index.
- Step 9: Calls `_embed_cluster_text()` which does `WHERE c.label IS NOT NULL` — finds 0 clusters → skips entirely.

The `ClusterLabeler` (LLM-based) exists in `imas_codex/clusters/labeler.py` with `label_clusters()` but is only called from `imas_codex/clusters/extractor.py` (the old file-based pipeline). It is **never invoked** from the graph-native `_import_clusters()`.

---

## Phase 1: Fix Cluster Search (Critical)

**Files:** `imas_codex/tools/graph_search.py`, `imas_codex/graph/build_dd.py`

### 1.1 Immediate fix: Use `cluster_embedding` index for search

The `cluster_embedding` vector index (on `IMASSemanticCluster.embedding`) has 3,635 populated vectors and works correctly (verified via Cypher test query). The `_search_by_text()` method should use this index instead of the empty `cluster_description_embedding`.

**Change in** `GraphClustersTool._search_by_text()` (`graph_search.py`):
```python
# Before:
CALL db.index.vector.queryNodes('cluster_description_embedding', $k, $embedding)

# After:
CALL db.index.vector.queryNodes('cluster_embedding', $k, $embedding)
```

This is the centroid embedding (mean of member IMASNode embeddings, L2-normalized). It represents the semantic center of the cluster's member paths. While not as interpretable as a label embedding, it will return semantically meaningful results because the centroid naturally represents the physics concept that groups the paths.

### 1.2 Enrich cluster search results with member-derived context

Since clusters lack labels and descriptions, the search results need to derive human-readable context from member paths. After the vector query, fetch the top member paths and their documentation to provide meaningful results.

**Change in** `GraphClustersTool._search_by_text()`:
- After getting cluster matches, query member IMASNodes to collect representative paths
- Include IDS names and sample documentation from the top 3-5 members
- Synthesize a summary line from the common path prefix + IDS coverage

### 1.3 Wire LLM cluster labeling into `_import_clusters()`

Add a new step between step 8 (centroid embedding) and step 9 (text embedding) that:
1. Reads clusters from graph with their member paths
2. Calls `ClusterLabeler.label_clusters()` in batches
3. Updates `IMASSemanticCluster` nodes with `label` and `description`
4. Then step 9 (`_embed_cluster_text()`) will find labeled clusters and generate `label_embedding` + `description_embedding`

**Change in** `_import_clusters()` (`build_dd.py`):
```python
# After Step 8 (centroid embedding), before Step 9:
# Step 8.5: Generate labels for unlabeled clusters
if not skip_labels:
    _label_unlabeled_clusters(client, batch_size=50)
```

New function `_label_unlabeled_clusters()`:
- Query clusters WHERE `label IS NULL` with their member paths
- Format clusters for `ClusterLabeler` input
- Call `label_clusters()` in batches
- SET `label`, `description`, `physics_concepts`, `tags` on cluster nodes
- Guard with `skip_labels` flag and error handling (labeling is not required for basic functionality)

### 1.4 Add CLI flag for label-only rebuild

Add `--labels-only` flag to `imas-codex imas dd build` that skips all phases except cluster labeling + text embedding. This allows re-running just the labeling step without a full rebuild.

### 1.5 Tests

- Test `_search_by_text()` returns results when using `cluster_embedding` index
- Test that `_label_unlabeled_clusters()` correctly updates cluster properties
- Test `_embed_cluster_text()` generates embeddings for labeled clusters

---

## Phase 2: Domain Name Resolution

**Files:** `imas_codex/tools/graph_search.py`

### 2.1 Add domain name resolution to `export_imas_domain()`

The physics_domains.yaml schema defines the canonical domain names. Users may pass:
- IDS names: `magnetics` → should resolve to IDS physics_domain `magnetic_field_diagnostics`
- Category names: `diagnostics` → should resolve to all domains in that category
- Partial matches: `magnetic` → should match `magnetic_field_diagnostics` and `magnetic_field_systems`
- Exact domain names: `magnetic_field_diagnostics` → direct match

**Add** `_resolve_physics_domain()` function to `graph_search.py`:
```python
def _resolve_physics_domain(gc: GraphClient, domain_query: str) -> list[str]:
    """Resolve a domain query to canonical physics domain names.

    Resolution order:
    1. Exact match on physics_domain property
    2. IDS name → its physics_domain
    3. Substring match on domain names
    4. Category-level expansion (if domain_query matches a DomainCategory)
    """
```

**Implementation:**
1. First try exact match: `MATCH (n:IMASNode {physics_domain: $domain}) RETURN DISTINCT n.physics_domain LIMIT 1`
2. If no match, check IDS name: `MATCH (i:IDS {name: $domain}) RETURN i.physics_domain`
3. If no match, substring: `MATCH (n:IMASNode) WHERE n.physics_domain CONTAINS $domain RETURN DISTINCT n.physics_domain`
4. Report resolved domain(s) in the result for transparency

**Change in** `export_imas_domain()`:
- Replace direct `WHERE p.physics_domain = $domain` with resolved domain list
- Use `WHERE p.physics_domain IN $resolved_domains`
- Return `resolved_from` field showing what the input resolved to

### 2.2 Apply same resolution to overview and search tools

The `get_imas_overview()` already does fuzzy IDS-level filtering (name, description, domain substring match) — this is adequate.

The `search_imas_paths()` does vector search so domain filtering is secondary — no change needed.

### 2.3 Tests

- Test resolution: `magnetics` → `['magnetic_field_diagnostics']`
- Test resolution: `magnetic` → `['magnetic_field_diagnostics', 'magnetic_field_systems']`  
- Test resolution: `diagnostics` → all diagnostic domains
- Test exact pass-through: `transport` → `['transport']`

---

## Phase 3: Input Normalization Hardening

**Files:** `imas_codex/tools/graph_search.py`, `imas_codex/tools/utils.py`, `imas_codex/graph/build_dd.py`, `imas_codex/core/paths.py`

### 3.0 Unified path annotation stripping (DONE)

**Problem:** IMAS paths carry index annotations — `(itime)`, `(i1)`, `(:)` from DD docs and `[1]`, `[:]` from code — but graph `IMASNode.id` values are clean. Users pasting annotated paths from documentation or code get 0 results.

**Fix (implemented):**

1. Created `imas_codex/core/paths.py` with shared `strip_path_annotations()` that strips both `(...)` and `[...]` patterns.
2. `build_dd.py` now imports and aliases this as `_strip_dd_indices` (replaces old private function that only handled parens).
3. `_normalize_paths()` in `graph_search.py` applies `strip_path_annotations()` to every path after splitting.
4. `normalize_paths_input()` in `tools/utils.py` applies `strip_path_annotations()` to every path after splitting.
5. Tests added in `tests/core/test_paths.py` and `tests/tools/test_utils.py`.

### 3.1 Harden `_normalize_paths()` for JSON array strings

When MCP clients (VS Code Copilot, Claude Desktop) serialize a `list[str]` parameter but the tool declares `str`, the transport may deliver it as `'["path1","path2"]'`. The current `_normalize_paths()` treats this as a single string and splits by spaces/commas, leaving brackets and quotes as part of path strings.

**Fix:**
```python
def _normalize_paths(paths: str | list[str]) -> list[str]:
    """Normalize paths input to a flat list."""
    if isinstance(paths, list):
        return [strip_path_annotations(p.strip()) for p in paths if isinstance(p, str) and p.strip()]
    # Handle JSON array strings from MCP transport
    s = paths.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [strip_path_annotations(str(p).strip()) for p in parsed if str(p).strip()]
        except (json.JSONDecodeError, TypeError):
            pass
    return [strip_path_annotations(p.strip()) for p in s.replace(",", " ").split() if p.strip()]
```

### 3.2 Apply to `normalize_ids_filter()` too

Check `imas_codex/tools/utils.py` — apply the same JSON array handling to `normalize_ids_filter()` since MCP clients may send list parameters the same way.

### 3.3 Tests

- Test `_normalize_paths('["magnetics/flux_loop", "core_profiles/profiles_1d"]')` → correctly parsed
- Test `_normalize_paths("path1 path2")` → unchanged behavior
- Test `_normalize_paths(["path1", "path2"])` → unchanged behavior
- Test edge case: `_normalize_paths('["path with spaces"]')` → handled
- Test annotation stripping: `_normalize_paths("flux_loop(i1)/flux/data(:)")` → `["flux_loop/flux/data"]` (DONE)
- Test bracket stripping: `normalize_paths_input(["time_slice[1]/profiles_1d[:]/psi"])` → `["time_slice/profiles_1d/psi"]` (DONE)

---

## Phase 4: Fetch Path Metadata Parity

**Files:** `imas_codex/tools/graph_search.py`

### 4.1 Add `introduced_after_version` to fetch results

The graph has `INTRODUCED_IN` relationships for every IMASNode. Add this to the fetch query:

```cypher
OPTIONAL MATCH (p)-[:INTRODUCED_IN]->(iv:DDVersion)
WITH ..., iv.id AS introduced_in_version
```

Map to `IdsNode.introduced_after_version`.

### 4.2 Add coordinate detail to fetch results

Currently `fetch_imas_paths` returns `coordinates` as a list of `IMASCoordinateSpec.id`. The old server returns coordinate paths like `profiles_1d(itime)/grid/rho_tor_norm`. The Codex graph stores this info on IMASNode properties.

**Change:** Also return `p.path_doc` (which contains the structure reference with coordinate annotations like `profiles_1d(itime)/electrons/temperature(:)`). Parse coordinate references from `path_doc` or add them from the `HAS_COORDINATE` relationship.

### 4.3 Add `node_type` categorization

The graph already stores `p.node_type` (static/dynamic). Surface this in the fetch result as a `type` field for parity with the old server's `dynamic`/`static` classification.

### 4.4 Add `path_doc` (structure path with indices) to fetch results

The `path_doc` property (e.g., `profiles_1d(itime)/electrons/temperature(:)`) provides the structural path with array index annotations that tells users how to navigate the IDS hierarchy. This is present in the graph but not returned by fetch.

**Change in** `fetch_imas_paths()` Cypher RETURN clause:
- Add `p.path_doc AS structure_path`
- Map to new `IdsNode` field or include in the documentation

### 4.5 Tests

- Test fetch returns `introduced_in_version` for paths
- Test fetch returns coordinate paths
- Test fetch returns node_type

---

## Phase 5: Identifier Search Improvement

**Files:** `imas_codex/tools/graph_search.py`, `imas_codex/graph/build_dd.py`

### 5.1 Populate IdentifierSchema descriptions during DD build

The graph has 62 `IdentifierSchema` nodes with `description = null`. The source XML files for these schemas contain descriptions that should be extracted during the DD build.

**Check** `build_dd.py` identifier extraction code — if descriptions exist in source XML, extract and store them. If not, generate short descriptions from the schema name + option names during the build.

### 5.2 Improve identifier search with fuzzy matching

Current `get_imas_identifiers()` does exact substring match on `name`, `description`, and `options` text. For "cocos", this misses `cocos_identifier` because the search string `cocos` is a substring of the name but somehow fails to match.

**Debug:** The query does `query_lower in (r["name"] or "").lower()` — this IS a substring check. If the query is "cocos" and the name is "cocos_identifier", this should match. Need to verify if there's actually a `cocos_identifier` schema in the graph.

**Verify via query:**
```cypher
MATCH (s:IdentifierSchema)
WHERE s.name CONTAINS 'cocos'
RETURN s.name, s.option_count
```

If the identifier exists, the bug is likely in null handling or that the search loads all schemas first then filters client-side. If it doesn't exist, the issue is that COCOS identifiers aren't being extracted during DD build.

### 5.3 Add option to search by IDS usage

Allow filtering identifiers by which IDS uses them:
```python
# Find identifiers used in equilibrium
get_imas_identifiers(ids_filter="equilibrium")
```

Query paths that have `HAS_IDENTIFIER_SCHEMA` relationships within the specified IDS.

### 5.4 Tests

- Test identifier search returns results for "cocos"
- Test identifier descriptions are populated
- Test ids_filter works

---

## Implementation Order

| Phase | Effort | Impact | Dependencies |
|-------|--------|--------|-------------|
| Phase 1.1 | 15 min | Critical — unblocks cluster search | None |
| Phase 3.1 | 15 min | Medium — fixes MCP input robustness | None |
| Phase 2.1 | 30 min | High — domain export usability | None |
| Phase 4 | 45 min | Medium — metadata parity | None |
| Phase 5 | 30 min | Low — identifier discoverability | None |
| Phase 1.2 | 30 min | High — cluster result quality | Phase 1.1 |
| Phase 1.3-1.4 | 2 hr | High — full cluster labeling | Phase 1.1, requires LLM access |

Phases 1.1, 2.1, 3.1 can be done immediately. Phase 1.3 (LLM labeling) requires an LLM call budget and should be run after the immediate fixes are validated.

## Validation Protocol

After implementation, re-run the comparison protocol from the evaluation session:

1. `search_imas_clusters("plasma current measurement")` → should return relevant clusters
2. `export_imas_domain("magnetics")` → should return magnetic_field_diagnostics paths
3. `check_imas_paths('["magnetics/flux_loop/flux/data"]')` → should parse and validate
4. `fetch_imas_paths("magnetics/flux_loop/flux/data")` → should include coordinates, version info
5. `get_imas_identifiers("cocos")` → should return cocos_identifier schema
6. Direct graph queries to verify all content is being served

## Out of Scope

- **`explain_concept`** — LLM-generated physics explanations. The calling agent can produce these from the metadata we return. Not implementing.
- **AI-augmented search responses** — The old server's AI insights (physics context, recommended follow-ups, etc.) add token cost without proportional value. The calling agent can synthesize this from the raw results.
- **Old server's `explore_relationships`** — Our `get_imas_path_context` already serves this use case better via graph traversal (cluster siblings, coordinate partners, unit companions, identifier links). The old server's `explore_relationships` relies on a static JSON file.
