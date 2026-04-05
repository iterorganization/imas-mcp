# IMAS-DD MCP Server: Opportunity Gap Tools

## Context

A/B testing of the imas-test and imas-prod MCP servers (17 tools tested on imas-test,
11 on imas-prod) identified 12 opportunity gaps — graph-backed analytics possible via
Cypher but not exposed as tools. After critique and filtering, 5 genuine opportunities
remain that provide information a frontier LLM cannot derive independently.

**Prerequisite:** The embedding fix (commit `642b1397`) must be deployed to restore
semantic search in the Docker container before these tools can use vector search.

**Bug fixes already shipped:** 5 bugs fixed in commit `9effc98b` (scope docstring,
unit stats rendering, version history rendering, fuzzy typo suggestions, coordinate
noise filtering).

---

## Naming Convention

All DD tools use the `imas` prefix — consistent with the 17 existing tools and the
shared API contract with imas-prod. The `dd` prefix is reserved for DD *versioning*
tools (`get_dd_versions`, `get_dd_version_context`, `get_dd_migration_guide`,
`get_dd_changelog`). A full `imas` → `dd` rename is deferred to a future major
version break.

| Prefix | Verb semantics | Scope |
|--------|---------------|-------|
| `search_imas` | Semantic/keyword discovery | IMAS-wide |
| `check_imas_*` | Validate paths/consistency | Specific paths or cross-IDS |
| `fetch_imas_*` | Retrieve detailed data | Specific paths |
| `list_imas_*` | Enumerate/browse | IDS subtree or cross-IDS |
| `get_imas_*` | Retrieve catalogs/metadata | IMAS-wide |
| `get_dd_*` | DD versioning/metadata | DD versions |
| `analyze_imas_*` | Compute derived analytics | Single IDS or cross-IDS |
| `export_imas_*` | Bulk export | Single IDS or domain |

---

## Phase 1 — Surface existing data + extend version context

Independent items. Can be implemented in parallel.

### 1a. Fix lifecycle rendering in `analyze_imas_structure`

**Problem:** The backend already queries `lifecycle_distribution` (graph_search.py
lines 1837-1845, stored in result dict line 1870-1872) but `format_structure_report()`
in search_formatters.py (lines 1247-1312) never renders it. The tool docstring in
server.py also doesn't mention lifecycle data.

**Implementation:**
1. In `search_formatters.py` `format_structure_report()`, add a "Lifecycle Status"
   section after the COCOS section that renders `lifecycle_distribution` from the
   result dict
2. In `server.py` line 2802, update the docstring to include "lifecycle status
   distribution" in the Returns description

**Files:** `imas_codex/llm/search_formatters.py`, `imas_codex/llm/server.py`
**Complexity:** Low
**Test:** Call `analyze_imas_structure(ids_name="distributions")` — should show
obsolescent paths in output

---

### 1b. Add rename chain following to `get_dd_version_context`

**Problem:** When `change_type_filter='path_renamed'` is used, only individual
rename records are returned. The graph has `(:IMASNode)-[:RENAMED_TO]->(:IMASNode)`
chains that can be multi-hop, but the tool doesn't follow them.

**Implementation:**
1. In `version_tool.py`, add `follow_rename_chains: bool = False` parameter to
   `get_dd_version_context()`
2. When `follow_rename_chains=True`, run a Cypher query that traverses
   `[:RENAMED_TO*1..10]` chains and returns the full lineage per path
3. Add a formatter section in `search_formatters.py` that renders rename chains
   as `original → intermediate → ... → current`
4. Register the new parameter in `server.py`

**Cypher pattern:**
```cypher
MATCH chain = (old:IMASNode)-[:RENAMED_TO*1..10]->(current:IMASNode)
WHERE old.ids = $ids_filter
  AND NOT EXISTS { ()-[:RENAMED_TO]->(old) }  // start of chain only
RETURN [n IN nodes(chain) | n.id] AS rename_chain,
       length(chain) AS hops
ORDER BY hops DESC
```

**Files:** `imas_codex/tools/version_tool.py`, `imas_codex/llm/search_formatters.py`,
`imas_codex/llm/server.py`
**Complexity:** Low-medium
**Test:** Query with `ids_filter='equilibrium'` — verify multi-hop chains appear

---

## Phase 2 — New DD analytics tools

Items 2a and 2b-2c can be implemented in parallel (2a goes in version_tool.py,
2b-2c go in new dd_analytics_tool.py).

### 2a. New tool: `get_dd_changelog`

**Purpose:** "Which paths change the most across DD versions?" Ranks paths by
change count, change type diversity, and rename history within an optional
version range.

**Parameters:**
- `ids_filter: str | None` — restrict to one IDS
- `from_version: str | None` — start of version range (exclusive)
- `to_version: str | None` — end of version range (inclusive)
- `limit: int = 50` — max results

**Returns:** Ranked list with volatility scores, change type breakdown, version
range summary

**Implementation:**
1. Add `get_dd_changelog()` method to `VersionTool` in `version_tool.py`
2. Key Cypher: aggregate `IMASNodeChange` counts per path, join `RENAMED_TO`
   presence, compute volatility score:
   ```cypher
   MATCH (change:IMASNodeChange)-[:FOR_IMAS_PATH]->(p:IMASNode)
   WHERE ($ids_filter IS NULL OR p.ids = $ids_filter)
     AND ($from_version IS NULL OR change.version > $from_version)
     AND ($to_version IS NULL OR change.version <= $to_version)
   WITH p, count(change) AS change_count,
        count(DISTINCT change.semantic_change_type) AS type_variety,
        collect(DISTINCT change.semantic_change_type) AS change_types
   OPTIONAL MATCH (p)-[:RENAMED_TO]->()
   WITH p, change_count, type_variety, change_types,
        CASE WHEN EXISTS { (p)-[:RENAMED_TO]->() } THEN 1 ELSE 0 END AS was_renamed
   RETURN p.id AS path, p.ids AS ids,
          change_count, type_variety, change_types, was_renamed,
          change_count + (type_variety * 2) + (was_renamed * 3) AS volatility_score
   ORDER BY volatility_score DESC
   LIMIT $limit
   ```
3. Add `format_dd_changelog_report()` in `search_formatters.py`
4. Register tool in `server.py`

**Files:** `imas_codex/tools/version_tool.py`, `imas_codex/llm/search_formatters.py`,
`imas_codex/llm/server.py`
**Complexity:** Medium
**Test:** Call without filters — verify ec_launchers paths rank highly (known
from A/B testing to have up to 12 version changes)

---

### 2b. New tool: `analyze_imas_coverage`

**Purpose:** "Which physical quantities span the most IDS?" Uses
`IMASSemanticCluster` membership to find concepts appearing across many IDS.

**Parameters:**
- `physics_domain: str | None` — optional filter
- `min_ids_count: int = 3` — minimum IDS to qualify
- `dd_version: int | None` — DD major version filter
- `limit: int = 30` — max results

**Returns:** Ranked clusters with IDS counts, representative paths, scope labels

**Implementation:**
1. Create new file `imas_codex/tools/dd_analytics_tool.py` with class
   `DDAnalyticsTool`
2. Add `analyze_imas_coverage()` method
3. Key Cypher:
   ```cypher
   MATCH (p:IMASNode)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
   WHERE c.scope = 'global' AND c.label IS NOT NULL
   WITH c, collect(DISTINCT p.ids) AS ids_list,
        count(DISTINCT p.ids) AS ids_count
   WHERE ids_count >= $min_ids_count
   OPTIONAL MATCH (c)-[:REPRESENTATIVE_PATH]->(rep:IMASNode)
   RETURN c.id AS cluster_id, c.label AS label,
          c.description AS description,
          ids_count, ids_list,
          rep.id AS representative_path
   ORDER BY ids_count DESC
   LIMIT $limit
   ```
4. Add physics_domain post-filter if specified (check member paths' domains)
5. Add `format_imas_coverage_report()` in `search_formatters.py`
6. Register tool in `server.py`

**Files:** `imas_codex/tools/dd_analytics_tool.py` (new),
`imas_codex/llm/search_formatters.py`, `imas_codex/llm/server.py`,
`imas_codex/tools/__init__.py`
**Complexity:** Medium
**Test:** Verify toroidal field cluster spans 28 IDS (known from A/B testing)

---

### 2c. New tool: `check_imas_units`

**Purpose:** "Are units consistent for the same physical concept across IDS?"
Finds paths in the same cluster with different units, distinguishing
dimension-incompatible (real issues) from symbol-different (advisory).

**Parameters:**
- `ids_filter: str | None` — restrict to clusters containing this IDS
- `physics_domain: str | None` — restrict to domain
- `dd_version: int | None` — DD major version filter
- `severity: str = 'all'` — 'all', 'incompatible', or 'advisory'

**Returns:** Inconsistencies grouped by cluster with unit details, dimension
comparison, severity level

**Implementation:**
1. Add `check_imas_units()` method to `DDAnalyticsTool`
2. Key Cypher:
   ```cypher
   MATCH (p1:IMASNode)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
         <-[:IN_CLUSTER]-(p2:IMASNode),
         (p1)-[:HAS_UNIT]->(u1:Unit),
         (p2)-[:HAS_UNIT]->(u2:Unit)
   WHERE p1.ids < p2.ids  // avoid duplicates
     AND u1.id <> u2.id
   RETURN c.label AS cluster, c.id AS cluster_id,
          p1.id AS path1, u1.id AS unit1, u1.dimension AS dim1,
          p2.id AS path2, u2.id AS unit2, u2.dimension AS dim2,
          CASE WHEN u1.dimension = u2.dimension THEN 'advisory'
               ELSE 'incompatible' END AS severity
   ORDER BY severity DESC, c.label
   ```
3. Post-filter by `severity` parameter
4. Add `format_imas_units_report()` in `search_formatters.py`
5. Register tool in `server.py`

**Files:** `imas_codex/tools/dd_analytics_tool.py`,
`imas_codex/llm/search_formatters.py`, `imas_codex/llm/server.py`
**Complexity:** Medium
**Test:** Verify some known unit inconsistencies appear. Check that
dimension-compatible pairs (e.g., eV vs keV) get 'advisory' severity

---

## Phase 3 — Impact analysis

Depends on Phase 2 completion (reuses `DDAnalyticsTool` class and
`find_related_imas_paths` machinery from graph_search.py).

### 3a. New tool: `analyze_imas_changes`

**Purpose:** "If path X changed, what else should I check?" Given a path and
optional version range, finds co-changing cluster siblings and related paths
with deterministic risk scores.

**Parameters:**
- `path: str` — IMAS path to analyze
- `from_version: str | None` — start of version range
- `to_version: str | None` — end of version range
- `dd_version: int | None` — DD major version filter

**Returns:** Impact report with:
1. The path's own change history
2. Cluster siblings that also changed in the same version range
3. Related paths via coordinates, units, identifier schemas
4. Deterministic risk score per related path

**Implementation:**
1. Add `analyze_imas_changes()` method to `DDAnalyticsTool`
2. Step 1 — Own changes:
   ```cypher
   MATCH (change:IMASNodeChange)-[:FOR_IMAS_PATH]->(p:IMASNode {id: $path})
   WHERE ($from_version IS NULL OR change.version > $from_version)
     AND ($to_version IS NULL OR change.version <= $to_version)
   RETURN change.version, change.semantic_change_type,
          change.summary, change.breaking_level
   ORDER BY change.version
   ```
3. Step 2 — Co-changing cluster siblings:
   ```cypher
   MATCH (p:IMASNode {id: $path})-[:IN_CLUSTER]->(c:IMASSemanticCluster)
         <-[:IN_CLUSTER]-(sibling:IMASNode)
   WHERE sibling.ids <> p.ids
   OPTIONAL MATCH (sc:IMASNodeChange)-[:FOR_IMAS_PATH]->(sibling)
   WHERE ($from_version IS NULL OR sc.version > $from_version)
     AND ($to_version IS NULL OR sc.version <= $to_version)
   WITH sibling, c, count(sc) AS sibling_changes,
        collect(DISTINCT sc.semantic_change_type) AS sibling_change_types
   WHERE sibling_changes > 0
   RETURN sibling.id, sibling.ids, c.label,
          sibling_changes, sibling_change_types
   ORDER BY sibling_changes DESC
   ```
4. Step 3 — Related paths via coordinates/units/identifiers (reuse patterns from
   `GraphPathContextTool.find_related_imas_paths` in graph_search.py lines 1670-1741)
5. Step 4 — Compute deterministic risk score:
   - `risk = co_change_count * 3 + cluster_overlap * 2 + coordinate_shared * 1`
   - `breaking_level` of changes adds multiplier
6. Add `format_imas_changes_report()` in `search_formatters.py`
7. Register tool in `server.py`

**Files:** `imas_codex/tools/dd_analytics_tool.py`,
`imas_codex/llm/search_formatters.py`, `imas_codex/llm/server.py`
**Complexity:** High
**Test:** Query `equilibrium/time_slice/profiles_1d/psi` — should find cluster
siblings in `core_profiles` and `core_transport` that co-change

---

## Phase 4 — Tests and documentation

Depends on all previous phases.

### 4a. Tests

Add test file `tests/tools/test_dd_analytics.py` covering:
- `get_dd_changelog`: empty result, single IDS filter, version range filter,
  volatility score ordering
- `analyze_imas_coverage`: min_ids_count filtering, physics_domain filter,
  cluster with NULL label excluded
- `check_imas_units`: dimension-incompatible vs advisory severity, ids_filter,
  empty results
- `analyze_imas_changes`: path with no changes, co-changing siblings found,
  risk score calculation, version range filtering

Add test file `tests/tools/test_version_tool_rename_chains.py` covering:
- Single-hop rename chain
- Multi-hop chain (3+ hops)
- No renames found
- IDS filter

Add assertions to existing `tests/llm/test_imas_dd_formatters.py` for:
- `format_structure_report` lifecycle section rendering
- New formatter functions

### 4b. Documentation

Update `server.py` docstrings for all new/modified tools.
Update `AGENTS.md` quick reference table to include new tools.

**Files:** `tests/tools/test_dd_analytics.py` (new),
`tests/tools/test_version_tool_rename_chains.py` (new),
`tests/llm/test_imas_dd_formatters.py`, `AGENTS.md`

---

## Parallel Implementation Guide

```
Phase 1a ──────────────────────────┐
(formatter fix)                     │
                                    ├──► Phase 2a (get_dd_changelog)
Phase 1b ──────────────────────────┤    in version_tool.py
(rename chains in version_tool.py)  │
                                    │
                                    ├──► Phase 2b + 2c (coverage + units)
                                    │    in dd_analytics_tool.py
                                    │
                                    └──► Phase 3a (changes)
                                         in dd_analytics_tool.py
                                         (after 2b-2c merge)
                                              │
                                              ▼
                                         Phase 4 (tests + docs)
```

**Agent 1:** Phase 1a (formatter) + Phase 2a (changelog in version_tool.py)
**Agent 2:** Phase 1b (rename chains) — also in version_tool.py but different method
**Agent 3:** Phase 2b + 2c (coverage + units in new dd_analytics_tool.py)
**Agent 4:** Phase 3a (changes — after Agent 3 merges, extends same file)
**Agent 5:** Phase 4 (tests + docs — after all above merge)

---

## Dropped Gaps (with reasoning)

| Gap | Reason dropped |
|-----|---------------|
| compare_dd_versions | Overlaps with `get_dd_migration_guide(summary_only=True)` |
| lifecycle_dashboard | Already exists — just needs formatter fix (Phase 1a) |
| deprecation_report | Covered by `list_imas_paths(lifecycle_filter='obsolescent')` |
| data_type_profile | Low value — `analyze_imas_structure` returns type distribution |
| unit_catalog | Low value — `get_imas_identifiers` covers this |
| identifier_usage | Low value — `get_imas_identifiers` shows field_count |
| orphan_analysis | Low value — rare edge case, ad-hoc REPL query |

## Naming Decision: imas prefix retained

All DD tools keep the `imas` prefix for consistency with the 17 existing tools
and shared API contract with imas-prod. A full `imas` → `dd` rename is deferred
to a future major version break. The `dd` prefix is reserved for DD *versioning*
tools only.
