# DD MCP Server: Tool Rename + Opportunity Gap Tools

## Context

A/B testing of the imas-test and imas-prod MCP servers (17 tools tested on imas-test,
11 on imas-prod) identified 12 opportunity gaps — graph-backed analytics possible via
Cypher but not exposed as tools. After critique and filtering, 5 genuine opportunities
remain that provide information a frontier LLM cannot derive independently.

This plan also includes a **breaking rename** of all DD tool names from `imas` → `dd`
prefix. This is appropriate for the current major release RC because:
- IMAS = Integrated Modelling & Analysis Suite (the whole platform)
- DD = Data Dictionary (what these tools actually query)
- Tools named `search_imas_*` misleadingly suggest they search all of IMAS
- The `dd` prefix is already used by version tools (`get_dd_versions`, etc.)
- Unifying all DD tools under `dd` creates a clean, consistent namespace

**Prerequisite:** The embedding fix (commit `642b1397`) must be deployed to restore
semantic search in the Docker container before these tools can use vector search.

**Bug fixes already shipped:** 5 bugs fixed in commit `9effc98b` (scope docstring,
unit stats rendering, version history rendering, fuzzy typo suggestions, coordinate
noise filtering).

---

## Naming Convention (Updated)

**All DD tools use the `dd` prefix.** This replaces the previous `imas` prefix.
The rename is a breaking change shipped with the major release.

| Prefix | Verb semantics | Scope |
|--------|---------------|-------|
| `search_dd_*` | Semantic/keyword discovery | DD-wide |
| `check_dd_*` | Validate paths/consistency | Specific paths or cross-IDS |
| `fetch_dd_*` | Retrieve detailed data | Specific paths |
| `list_dd_*` | Enumerate/browse | IDS subtree or cross-IDS |
| `get_dd_*` | Retrieve catalogs/metadata/versions | DD-wide |
| `analyze_dd_*` | Compute derived analytics | Single IDS or cross-IDS |
| `export_dd_*` | Bulk export | Single IDS or domain |

**NOT renamed** (facility tools, infrastructure):
`search_signals`, `signal_analytics`, `search_docs`, `search_code`, `fetch_content`,
`get_facility_coverage`, `get_graph_schema`, `repl`, `add_to_graph`,
`update_facility_config`, `list_logs`, `get_logs`, `tail_logs`

### Scope Boundary: DD Tools vs IMAS Model

The rename applies ONLY to **DD query tools** — functions whose purpose is to
search, browse, validate, or analyze the Data Dictionary itself. It does NOT
apply to code that references the IMAS data model as a target for mapping or
data assembly.

**The distinction:**
- **DD tool** = "query the Data Dictionary" → rename to `dd`
- **IMAS model reference** = "map/assemble data into the IMAS standard" → keep `imas`

| Function | Purpose | Verdict |
|----------|---------|---------|
| `search_imas` (MCP tool) | Search DD paths by concept | **Rename** → `search_dd_paths` |
| `map_signals_to_imas` (REPL) | Map facility signals to IMAS model paths | **Keep** `imas` |
| `find_imas` (REPL domain query) | Find DD paths by concept | **Rename** → `find_dd` |
| `ids/tools.py` functions | IDS assembly helpers that call DD tools internally | **Keep** `imas` |
| `ids/mapping.py` | IMAS mapping pipeline | **Keep** `imas` |
| `format_imas_report` (formatter) | Format imas-prod search results | **Keep** `imas` |
| `format_search_imas_report` | Format DD search tool output | **Rename** → `format_search_dd_report` |
| `IMASNode`, `IMASMapping` | Graph schema labels for the IMAS data model | **Keep** `imas` |
| `imas_codex` | Python package / CLI brand | **Keep** `imas` |

**Why `ids/tools.py` keeps `imas`:** Functions like `fetch_imas_subtree()`,
`fetch_imas_fields()`, `search_imas_semantic()`, and `check_imas_paths()` are
internal helpers for the IDS mapping pipeline (`ids/mapping.py`). They call
DD tool methods internally, but their purpose is data assembly — fetching parts
of the IMAS data model to build IDS instances. They are not MCP tools. Renaming
their internal method calls to match the new DD tool names is sufficient; the
function names themselves stay `imas` because they describe operations on the
IMAS model, not DD queries.

**Why `map_signals_to_imas` keeps `imas`:** This REPL function maps facility
signals to their IMAS-standard equivalents. The destination is "the IMAS data
model" — a conceptual standard. Calling it `map_signals_to_dd` would wrongly
imply mapping signals into the dictionary itself, rather than into the IMAS
standard that the DD defines.

**Why `format_imas_report` keeps `imas`:** This formats results from the
imas-prod-style search service (the `SearchService` pipeline), not from the
DD graph tools. It's a different code path.

**NOT renamed** (out of scope):
- Python package name: `imas_codex` (too deep, different concern)
- Graph node labels: `IMASNode`, `IMASSemanticCluster` (schema-level)
- Schema files: `imas_dd.yaml` (internal)
- Class names: `GraphSearchTool`, `GraphPathTool` (no imas prefix)
- CLI commands: `imas-codex` (brand name)
- IDS mapping pipeline: `ids/tools.py`, `ids/mapping.py`, `ids/validation.py`
- REPL function: `map_signals_to_imas` (maps to IMAS model, not DD)
- Formatter: `format_imas_report` (imas-prod search formatter)

---

## Phase 0 — The Big Rename: `imas` → `dd`

**This phase must complete before any other phase starts.** It is the atomic
contract change that all subsequent work depends on.

### Canonical Rename Mapping — MCP Tools

| # | Current Name | New Name | Notes |
|---|-------------|----------|-------|
| 1 | `search_imas` | `search_dd_paths` | Added `_paths` suffix for consistency |
| 2 | `check_imas_paths` | `check_dd_paths` | |
| 3 | `fetch_imas_paths` | `fetch_dd_paths` | |
| 4 | `list_imas_paths` | `list_dd_paths` | |
| 5 | `get_imas_overview` | `get_dd_overview` | |
| 6 | `get_imas_identifiers` | `get_dd_identifiers` | |
| 7 | `search_imas_clusters` | `search_dd_clusters` | |
| 8 | `find_related_imas_paths` | `find_related_dd_paths` | |
| 9 | `analyze_imas_structure` | `analyze_dd_structure` | |
| 10 | `export_imas_ids` | `export_dd_ids` | |
| 11 | `export_imas_domain` | `export_dd_domain` | |
| 12 | `fetch_error_fields` | `fetch_dd_error_fields` | Added `dd_` prefix |
| 13 | `get_cocos_fields` | `get_dd_cocos_fields` | Added `dd_` prefix |
| — | `get_dd_versions` | `get_dd_versions` | Already `dd` — no change |
| — | `get_dd_version_context` | `get_dd_version_context` | Already `dd` — no change |
| — | `get_dd_migration_guide` | `get_dd_migration_guide` | Already `dd` — no change |

### Canonical Rename Mapping — REPL Helpers

| Current | New | Notes |
|---------|-----|-------|
| `search_imas` | `search_dd` | REPL shorthand (no `_paths` suffix) |
| `fetch_imas` | `fetch_dd` | |
| `list_imas` | `list_dd` | |
| `check_imas` | `check_dd` | |
| `get_imas_overview` | `get_dd_overview` | |
| `get_imas_path_context` | `get_dd_path_context` | |
| `analyze_imas_structure` | `analyze_dd_structure` | |
| `export_imas_ids` | `export_dd_ids` | |
| `export_imas_domain` | `export_dd_domain` | |
| `find_imas` | `find_dd` | Domain query — searches DD by concept |
| `map_signals_to_imas` | **NO CHANGE** | Maps to IMAS model, not a DD query |

### Canonical Rename Mapping — Internal Methods

These are the tool class methods that implement the MCP tools above.
`ids/tools.py` functions keep their `imas` names but update their
internal calls to use the renamed tool methods.

| Current | New | File |
|---------|-----|------|
| `GraphSearchTool.search_imas_paths` | `.search_dd_paths` | graph_search.py |
| `GraphPathTool.check_imas_paths` | `.check_dd_paths` | graph_search.py |
| `GraphPathTool.fetch_imas_paths` | `.fetch_dd_paths` | graph_search.py |
| `GraphPathTool.fetch_error_fields` | `.fetch_dd_error_fields` | graph_search.py |
| `GraphListTool.list_imas_paths` | `.list_dd_paths` | graph_search.py |
| `GraphOverviewTool.get_imas_overview` | `.get_dd_overview` | graph_search.py / overview_tool.py |
| `GraphClustersTool.search_imas_clusters` | `.search_dd_clusters` | graph_search.py |
| `GraphIdentifiersTool.get_imas_identifiers` | `.get_dd_identifiers` | graph_search.py / identifiers_tool.py |
| `GraphPathContextTool.get_imas_path_context` | `.get_dd_path_context` | graph_search.py |
| `GraphPathContextTool.find_related_imas_paths` | `.find_related_dd_paths` | graph_search.py |
| `GraphStructureTool.analyze_imas_structure` | `.analyze_dd_structure` | graph_search.py |
| `GraphStructureTool.export_imas_ids` | `.export_dd_ids` | graph_search.py |
| `GraphStructureTool.export_imas_domain` | `.export_dd_domain` | graph_search.py |
| `GraphStructureTool.get_cocos_fields` | `.get_dd_cocos_fields` | graph_search.py |
| `_text_search_imas_paths` | `_text_search_dd_paths` | graph_search.py |
| `IMASTools` delegation (13 methods) | Updated names | __init__.py |
| `format_search_imas_report` | `format_search_dd_report` | search_formatters.py |
| `DomainQueries.find_imas` | `.find_dd` | domain_queries.py |

**NOT renamed** (internal IMAS model references):
| Function | File | Reason |
|----------|------|--------|
| `fetch_imas_subtree` | ids/tools.py | IDS assembly helper |
| `fetch_imas_fields` | ids/tools.py | IDS assembly helper |
| `search_imas_semantic` | ids/tools.py | IDS mapping search |
| `check_imas_paths` | ids/tools.py | IDS validation helper |
| `map_signals_to_imas` | domain_queries.py | Maps to IMAS model |
| `format_imas_report` | search_formatters.py | imas-prod formatter |

### Files in Scope (Phase 0)

**Layer 1 — Tool implementations** (rename method names):
- `imas_codex/tools/graph_search.py` — 15 method renames + 1 helper
- `imas_codex/tools/__init__.py` — 13 delegation methods
- `imas_codex/tools/identifiers_tool.py` — `get_imas_identifiers` → `get_dd_identifiers`
- `imas_codex/tools/list_tool.py` — `list_imas_paths` → `list_dd_paths`
- `imas_codex/tools/overview_tool.py` — `get_imas_overview` → `get_dd_overview`

**Layer 2 — MCP server** (rename function defs + REPL bindings):
- `imas_codex/llm/server.py` — 13 MCP tool functions + 10 REPL functions + docstrings

**Layer 3 — Search/recommendation infrastructure** (hardcoded tool name strings):
- `imas_codex/search/tool_suggestions.py` — ~12 tool name string references
- `imas_codex/search/decorators/tool_recommendations.py` — ~18 tool name strings
- `imas_codex/search/decorators/error_handling.py` — ~6 tool name strings
- `imas_codex/models/result_models.py` — 5 `tool_name` property returns

**Layer 4 — Domain queries** (REPL backend):
- `imas_codex/graph/domain_queries.py` — `find_imas` → `find_dd` only

**Layer 5 — Formatters** (one function name):
- `imas_codex/llm/search_formatters.py` — `format_search_imas_report` → `format_search_dd_report`

**Layer 6 — IDS tools call-site updates** (update method calls, NOT function names):
- `imas_codex/ids/tools.py` — internal calls like `tool.list_imas_paths()` →
  `tool.list_dd_paths()`, `tool.fetch_imas_paths()` → `tool.fetch_dd_paths()`,
  `tool.search_imas_paths()` → `tool.search_dd_paths()`

**Layer 7 — Prompts**:
- `imas_codex/llm/prompts/shared/tools.md` — 3 references

**Layer 8 — Benchmarks**:
- `benchmarks/bench_memory.py` — 3 tool name strings
- `benchmarks/bench_mcp_search.py` — ~5 tool name strings

### Phase 0 Agent Assignment

This phase is **NOT parallelizable** at the file level because layers are tightly
coupled (server.py imports tools/, models reference tool names). Use a single agent
or serialize strictly:

**Agent 0a** (Core rename — one agent, one commit):
- Layers 1-6 above (all Python source files)
- Must be atomic — all method renames + all callers in one commit
- Run `uv run ruff check --fix . && uv run ruff format .` after rename
- Run `uv run pytest tests/graph_mcp/test_tool_registration.py -x` to verify tools register

**Agent 0b** (References — after 0a merges):
- Layer 7 (prompts) + Layer 8 (benchmarks)
- Documentation updates (see Phase 4)

### Verification

After Phase 0, these must pass:
```bash
uv run pytest tests/graph_mcp/test_tool_registration.py -x
uv run pytest tests/llm/test_tool_schemas.py -x
uv run pytest tests/tools/ -x
```

---

## Phase 1 — Surface existing data + extend version context

Independent items. Can be implemented in parallel. **Use `dd` names throughout.**

### 1a. Fix lifecycle rendering in `analyze_dd_structure`

**Problem:** The backend already queries `lifecycle_distribution` (graph_search.py
lines 1837-1845, stored in result dict line 1870-1872) but `format_structure_report()`
in search_formatters.py (lines 1247-1312) never renders it. The tool docstring in
server.py also doesn't mention lifecycle data.

**Implementation:**
1. In `search_formatters.py` `format_structure_report()`, add a "Lifecycle Status"
   section after the COCOS section that renders `lifecycle_distribution` from the
   result dict
2. In `server.py`, update the docstring to include "lifecycle status
   distribution" in the Returns description

**Files:** `imas_codex/llm/search_formatters.py`, `imas_codex/llm/server.py`
**Complexity:** Low
**Test:** Call `analyze_dd_structure(ids_name="distributions")` — should show
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

### 2b. New tool: `analyze_dd_coverage`

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
2. Add `analyze_dd_coverage()` method
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
5. Add `format_dd_coverage_report()` in `search_formatters.py`
6. Register tool in `server.py`

**Files:** `imas_codex/tools/dd_analytics_tool.py` (new),
`imas_codex/llm/search_formatters.py`, `imas_codex/llm/server.py`,
`imas_codex/tools/__init__.py`
**Complexity:** Medium
**Test:** Verify toroidal field cluster spans 28 IDS (known from A/B testing)

---

### 2c. New tool: `check_dd_units`

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
1. Add `check_dd_units()` method to `DDAnalyticsTool`
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
4. Add `format_dd_units_report()` in `search_formatters.py`
5. Register tool in `server.py`

**Files:** `imas_codex/tools/dd_analytics_tool.py`,
`imas_codex/llm/search_formatters.py`, `imas_codex/llm/server.py`
**Complexity:** Medium
**Test:** Verify some known unit inconsistencies appear. Check that
dimension-compatible pairs (e.g., eV vs keV) get 'advisory' severity

---

## Phase 3 — Impact analysis

Depends on Phase 2 completion (reuses `DDAnalyticsTool` class and
`find_related_dd_paths` machinery from graph_search.py).

### 3a. New tool: `analyze_dd_changes`

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
1. Add `analyze_dd_changes()` method to `DDAnalyticsTool`
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
   `GraphPathContextTool.find_related_dd_paths` in graph_search.py)
5. Step 4 — Compute deterministic risk score:
   - `risk = co_change_count * 3 + cluster_overlap * 2 + coordinate_shared * 1`
   - `breaking_level` of changes adds multiplier
6. Add `format_dd_changes_report()` in `search_formatters.py`
7. Register tool in `server.py`

**Files:** `imas_codex/tools/dd_analytics_tool.py`,
`imas_codex/llm/search_formatters.py`, `imas_codex/llm/server.py`
**Complexity:** High
**Test:** Query `equilibrium/time_slice/profiles_1d/psi` — should find cluster
siblings in `core_profiles` and `core_transport` that co-change

---

## Phase 4 — Tests and documentation

Depends on all previous phases.

### 4a. Test Updates for Rename

Update all test files referencing old tool names. Key files (~34 total):

**High priority** (tool registration + integration):
- `tests/graph_mcp/test_tool_registration.py`
- `tests/graph_mcp/test_dd_tool_features.py`
- `tests/graph_mcp/test_graph_search.py`
- `tests/llm/test_tool_schemas.py`
- `tests/llm/test_mcp_bug_regressions.py`
- `tests/llm/test_search_tools.py`
- `tests/llm/test_imas_dd_formatters.py` (rename file → `test_dd_formatters.py`)

**Medium priority** (tools unit tests):
- `tests/tools/test_tools.py`
- `tests/tools/test_clusters.py`
- `tests/tools/test_identifiers_tool.py`
- `tests/tools/test_list_tool.py`
- `tests/tools/test_overview_tool.py`
- `tests/tools/test_dd_version_and_error_fields.py`
- `tests/tools/test_dd_version_filtering.py`
- `tests/tools/test_graph_search_structure_semantics.py`
- `tests/tools/test_query_analysis.py`
- `tests/tools/test_utils.py`
- `tests/tools/test_migration_guide.py`

**Lower priority** (search/IDS/integration):
- `tests/search/test_*.py` (multiple files)
- `tests/ids/test_*.py` (multiple files)
- `tests/integration/test_*.py`
- `tests/features/test_*.py`
- `tests/test_imas_search_remediation.py`
- `tests/conftest.py`

### 4b. New Tool Tests

Add test file `tests/tools/test_dd_analytics.py` covering:
- `get_dd_changelog`: empty result, single IDS filter, version range filter,
  volatility score ordering
- `analyze_dd_coverage`: min_ids_count filtering, physics_domain filter,
  cluster with NULL label excluded
- `check_dd_units`: dimension-incompatible vs advisory severity, ids_filter,
  empty results
- `analyze_dd_changes`: path with no changes, co-changing siblings found,
  risk score calculation, version range filtering

Add test file `tests/tools/test_version_tool_rename_chains.py` covering:
- Single-hop rename chain
- Multi-hop chain (3+ hops)
- No renames found
- IDS filter

Add assertions to existing formatter tests for:
- `format_structure_report` lifecycle section rendering
- New formatter functions

### 4c. Documentation Updates

| File | Changes |
|------|---------|
| `AGENTS.md` | Update quick reference table, REPL examples, all tool name references |
| `README.md` | Update any tool name examples |
| `docs/api/REPL_API.md` | Update all REPL function names |
| `imas_codex/llm/README.md` | Update tool references |
| `imas_codex/llm/prompts/shared/tools.md` | Update tool name examples |
| `plans/features/*.md` | Update all plan references to use new names |
| `plans/STRATEGY.md` | Update tool references |
| `.github/prompts/*.prompt.md` | Update any tool references |

**Files:** `tests/tools/test_dd_analytics.py` (new),
`tests/tools/test_version_tool_rename_chains.py` (new),
formatter test files, ~34 test files for rename, ~10 doc files

---

## Parallel Implementation Guide

```
Phase 0 (rename) ─────────────────────────────────────────────┐
  Agent 0a: Core rename (tools/ + server.py + search/ + ids/) │
  Agent 0b: References (prompts, benchmarks) [after 0a]       │
                                                               │
Phase 1a ──────────────────────────┐                           │
(formatter fix)                     │                           │
                                    ├──► Phase 2a (changelog)  │
Phase 1b ──────────────────────────┤    in version_tool.py     │
(rename chains in version_tool.py)  │                           │
                                    │                           │
                                    ├──► Phase 2b + 2c          │
                                    │    in dd_analytics_tool.py │
                                    │                           │
                                    └──► Phase 3a (changes)     │
                                         in dd_analytics_tool.py│
                                         (after 2b-2c merge)   │
                                              │                 │
                                              ▼                 │
                                         Phase 4 (tests + docs)│
                                         [after ALL above]      │
```

**Agent 0a:** Phase 0 core rename (ALL Python source — one atomic commit)
**Agent 0b:** Phase 0 references (prompts + benchmarks — after 0a merges)
**Agent 1:** Phase 1a (lifecycle formatter) + Phase 2a (changelog)
**Agent 2:** Phase 1b (rename chains in version_tool.py)
**Agent 3:** Phase 2b + 2c (coverage + units in new dd_analytics_tool.py)
**Agent 4:** Phase 3a (change impact — after Agent 3 merges)
**Agent 5:** Phase 4 (all tests + all docs — after ALL above merge)

---

## Dropped Gaps (with reasoning)

| Gap | Reason dropped |
|-----|---------------|
| compare_dd_versions | Overlaps with `get_dd_migration_guide(summary_only=True)` |
| lifecycle_dashboard | Already exists — just needs formatter fix (Phase 1a) |
| deprecation_report | Covered by `list_dd_paths(lifecycle_filter='obsolescent')` |
| data_type_profile | Low value — `analyze_dd_structure` returns type distribution |
| unit_catalog | Low value — `get_dd_identifiers` covers this |
| identifier_usage | Low value — `get_dd_identifiers` shows field_count |
| orphan_analysis | Low value — rare edge case, ad-hoc REPL query |
