# Plan: Efficient Agent-Graph Interaction

## Problem Statement

Agents interacting with the imas-codex Neo4j graph are inefficient, generating 10s-100s of Python REPL calls for tasks that should take 1-3 calls. Root causes:

1. **Schema context flooding**: `get_graph_schema()` returns ~38,000 tokens (53 node types × 957 properties + 71 relationship types + 11 vector indexes). This either overwhelms the context window or agents skip calling it and write blind Cypher.

2. **No domain-aware query layer**: The REPL exposes `query(cypher)` — raw Cypher — forcing agents to compose queries from scratch each time. Agents don't know which properties exist, which indexes to use, or how to traverse relationships, so they probe incrementally (list labels → get properties → try a query → fix errors → retry).

3. **No composed operations**: Common multi-step patterns (semantic search → traverse → format) require 3-5 REPL calls that should be a single function call returning actionable results.

4. **Missing codegen scaffolding**: The REPL philosophy is "agents write code to solve problems" but there are no reusable building blocks for agents to compose from — no query builders, no traversal helpers, no typed result formatters.

5. **No auto-generation from LinkML**: Schema context, task groups, enum mappings, and query functions are hand-maintained, meaning they drift from the source-of-truth LinkML schemas. Changes to the schema silently break downstream code.

## Design Principles

1. **Codegen-first**: Agents generate Python code, not individual tool calls. REPL functions return rich, structured results that chain together. The `python()` REPL is the primary interface.
2. **Minimal MCP tools for context + gated writes**: MCP tools serve two specific roles: (a) providing context/schema that agents need *before* writing code, and (b) gating write operations through Pydantic validators. Everything else goes through the REPL.
3. **LinkML as single source of truth**: All schema context, task groups, enum values, and query scaffolding are auto-generated from LinkML schemas at build time (`uv sync`), ensuring they never drift.
4. **Tiered schema context**: Instead of one 38K-token dump, provide gated, task-specific schema slices via `schema_for()`. The MCP `get_graph_schema()` tool delegates to `schema_for()` internally.
5. **Pre-composed queries for hot paths**: The 80% of common operations should be one function call. The 20% of novel queries use schema-aware Cypher helpers.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Agent (Claude, etc.)                       │
│  Generates Python code using REPL functions                  │
├──────────┬───────────────────────────────────────────────────┤
│  MCP     │  REPL (python() tool)                             │
│  Tools   │                                                   │
│  ┌──────┐│  Layer 3: Domain Operations                       │
│  │schema││  find_signals(), find_wiki(), find_code()          │
│  │ (R)  ││  map_to_imas(), facility_overview()               │
│  ├──────┤│                                                   │
│  │add_to││  Layer 2: Schema-Aware Helpers                    │
│  │graph ││  schema_for(), graph_search()                     │
│  │ (W)  ││  as_table(), pick()                               │
│  ├──────┤│                                                   │
│  │config││  Layer 1: Core (EXISTS)                            │
│  │tools ││  query(), embed(), semantic_search()              │
│  │ (W)  ││  get_facility(), run()                            │
│  └──────┘│                                                   │
├──────────┴───────────────────────────────────────────────────┤
│  GraphClient + Encoder + GraphSchema (LinkML)                │
│  Auto-generated: models.py, schema_context.py, enums         │
└──────────────────────────────────────────────────────────────┘
```

**MCP vs REPL boundary**: MCP tools are for (R) providing read-only context that agents need before writing code, and (W) gating writes through Pydantic validation. All composition, control flow, and multi-step logic happens in the REPL.

## Two MCP Server Contexts

The project has two distinct MCP servers serving different audiences:

### 1. AgentsServer (`imas_codex/agentic/server.py`) — Internal agents

9 tools for agent-driven facility exploration and graph building:

| Tool | Role | Assessment |
|------|------|------------|
| `python()` | Primary REPL interface | **Core** — keep as-is |
| `get_graph_schema()` | Schema context for query generation | **Refactor** — delegate to `schema_for()` with `scope` parameter |
| `add_to_graph()` | Gated graph writes with Pydantic validation | **Core** — keep as-is, exemplary gated write pattern |
| `update_facility_config()` | Read/update facility config | **Keep** — consolidates public/private config access |
| `update_facility_infrastructure()` | Deep-merge private YAML | **Keep** — gated write for infrastructure |
| `get_facility_infrastructure()` | Read private infra data | **Keep** — context retrieval |
| `add_exploration_note()` | Append timestamped notes | **Keep** — convenience, well-scoped |
| `update_facility_paths()` | Update path mappings | **Consolidate** — thin wrapper around `update_facility_infrastructure`, could be REPL-only |
| `update_facility_tools()` | Update tool availability | **Consolidate** — thin wrapper around `update_facility_infrastructure`, could be REPL-only |

**Action**: Keep 7 core tools. The two thin wrappers (`update_facility_paths`, `update_facility_tools`) can remain as REPL functions (already available as `update_infrastructure()`) but don't need dedicated MCP tool status. The key refactor is `get_graph_schema()` gaining a `scope` parameter powered by `schema_for()` internally.

### 2. DD Server (`imas_codex/server.py`) — External MCP clients

10 tools for IMAS Data Dictionary querying via the `Tools` class:

| Tool | Role | Assessment |
|------|------|------------|
| `search_imas_paths` | Semantic DD search | **Good** — core DD tool |
| `check_imas_paths` | Path validation | **Good** — essential |
| `fetch_imas_paths` | Full path docs | **Good** — essential |
| `list_imas_paths` | Path listing | **Good** — essential |
| `get_imas_overview` | DD summary | **Good** — overview context |
| `search_imas_clusters` | Cluster search | **Good** — semantic grouping |
| `get_imas_identifiers` | Identifier listing | **Good** — reference data |
| `query_imas_graph` | Read-only Cypher | **Good** — read-gated with mutation blocking |
| `get_dd_graph_schema` | DD-specific schema | **Refactor** — should also use `schema_for()` under the hood |
| `get_dd_version` | Version info | **Good** — metadata |

**Action**: These external tools are well-designed and serve their audience (MCP clients without REPL access). The only change is `get_dd_graph_schema` delegating to `schema_for(task='imas')` internally.

## Phase 1: Auto-Generated Schema Context from LinkML (High Impact, Medium Effort)

### Problem

The current `get_graph_schema()` MCP tool returns a live-built dict from `GraphSchema` at runtime. This works but: (a) returns everything (~38K tokens), (b) the task group definitions and example Cypher patterns would be hand-maintained Python code that drifts from the LinkML schemas.

### Solution: Build-Time Schema Context Generation

Extend the existing `uv sync` build pipeline (`hatch_build_hooks.py` → `scripts/build_models.py`) to also generate a **schema context module** from LinkML. This module provides `schema_for()` with all the data it needs, auto-generated and always in sync.

#### What gets generated at build time

| Generated Artifact | Source | Destination | Purpose |
|-------------------|--------|-------------|---------|
| Pydantic models | `schemas/facility.yaml`, `imas_dd.yaml` | `graph/models.py`, `graph/dd_models.py` | **Exists** — node validation |
| Physics domain enum | `definitions/physics/domains.yaml` | `core/physics_domain.py` | **Exists** — enum code |
| Schema reference | `schemas/*.yaml` | `agents/schema-reference.md` | **Exists** — agent docs |
| Config models | `schemas/facility_config.yaml` | `config/models.py` | **Exists** — config validation |
| **Schema context data** | `schemas/*.yaml` | `graph/schema_context_data.py` | **NEW** — task groups, compact formats, index maps |

#### What `schema_context_data.py` contains (auto-generated)

```python
"""Auto-generated schema context data from LinkML schemas.

DO NOT EDIT — regenerated by: uv run build-models --force
Source: imas_codex/schemas/facility.yaml, imas_dd.yaml, common.yaml
"""

# All node labels with their key properties (excludes embeddings, internal fields)
NODE_LABEL_PROPS: dict[str, dict[str, str]] = {
    "FacilitySignal": {
        "id": "string (ID, format: '{facility}:{diagnostic}/{name}')",
        "name": "string (required)",
        "facility_id": "string (required)",
        "diagnostic": "string",
        "description": "string",
        "physics_domain": "PhysicsDomain",
        "status": "FacilitySignalStatus",
        "unit": "string",
        "has_data": "boolean",
        "data_type": "string",
        # ... all non-private, non-embedding properties
    },
    # ... all 53 node types
}

# Enum values (complete, from LinkML)
ENUM_VALUES: dict[str, list[str]] = {
    "PhysicsDomain": ["magnetics", "kinetic", "equilibrium", ...],
    "FacilitySignalStatus": ["discovered", "validated", "stale"],
    # ... all enums
}

# Relationship types with directionality
RELATIONSHIPS: list[tuple[str, str, str, str]] = [
    # (from_label, rel_type, to_label, cardinality)
    ("FacilitySignal", "DATA_ACCESS", "DataAccess", "many"),
    ("FacilitySignal", "BELONGS_TO_DIAGNOSTIC", "Diagnostic", "one"),
    ("FacilitySignal", "AT_FACILITY", "Facility", "one"),
    # ... all relationships
]

# Vector indexes
VECTOR_INDEXES: dict[str, tuple[str, str]] = {
    "facility_signal_desc_embedding": ("FacilitySignal", "embedding"),
    "wiki_chunk_embedding": ("WikiChunk", "embedding"),
    "imas_path_embedding": ("IMASPath", "embedding"),
    # ... all 11 indexes
}

# Task groups — which labels are relevant for each domain task
TASK_GROUPS: dict[str, list[str]] = {
    "signals": ["FacilitySignal", "DataAccess", "Diagnostic", "AccessCheck"],
    "wiki": ["WikiPage", "WikiChunk", "WikiArtifact", "Image"],
    "imas": ["IMASPath", "IDS", "IMASSemanticCluster", "DDVersion", "Unit",
             "IMASPathChange", "IMASCoordinateSpec"],
    "code": ["CodeFile", "CodeChunk", "CodeExample"],
    "facility": ["Facility", "FacilityPath", "FacilitySignal", "DataNode", "Diagnostic"],
    "trees": ["DataNode", "DataSource", "DataNodePattern"],
}
```

#### Build script: `scripts/gen_schema_context.py`

This script reads LinkML schemas and generates the above Python module. It follows
the same pattern as `gen_schema_reference.py` — freshness checking, called during
`uv sync`, gitignored output.

**Key**: Task group definitions live in a small YAML file (`schemas/task_groups.yaml`)
so they can be edited without touching Python code. The build script reads this YAML
and validates that all referenced labels exist in the LinkML schemas.

```yaml
# schemas/task_groups.yaml — defines schema slices for schema_for()
signals:
  labels: [FacilitySignal, DataAccess, Diagnostic, AccessCheck]
  description: Signal discovery, data access patterns, and diagnostics

wiki:
  labels: [WikiPage, WikiChunk, WikiArtifact, Image]
  description: Wiki content search and page navigation

imas:
  labels: [IMASPath, IDS, IMASSemanticCluster, DDVersion, Unit, IMASPathChange, IMASCoordinateSpec]
  description: IMAS Data Dictionary paths, versions, and semantic clusters

code:
  labels: [CodeFile, CodeChunk, CodeExample]
  description: Ingested source code and code examples

facility:
  labels: [Facility, FacilityPath, FacilitySignal, DataNode, Diagnostic]
  description: Facility infrastructure and discovery paths

trees:
  labels: [DataNode, DataSource, DataNodePattern]
  description: MDSplus tree structure and node patterns
```

The build script validates at generation time that:
- All labels in task groups exist in the LinkML schemas
- All relationships referenced connect labels that exist
- Enum values match what's in the schemas
- Vector indexes reference valid label/property pairs

This catches schema drift at build time, not runtime.

### `schema_for()` — Runtime function using generated data

```python
# imas_codex/graph/schema_context.py
from imas_codex.graph.schema_context_data import (
    NODE_LABEL_PROPS,
    ENUM_VALUES,
    RELATIONSHIPS,
    VECTOR_INDEXES,
    TASK_GROUPS,
)

def schema_for(
    *labels: str,
    task: str | None = None,
    include_relationships: bool = True,
    include_examples: bool = True,
) -> str:
    """Get compact, task-relevant schema context for Cypher query generation.

    Uses auto-generated data from LinkML schemas — always in sync with
    the graph structure.

    Args:
        labels: Specific node labels (e.g., 'FacilitySignal', 'DataNode')
        task: Predefined task name for curated schema slices:
              'signals', 'wiki', 'imas', 'code', 'facility', 'trees'
        include_relationships: Include relevant relationship types
        include_examples: Include example Cypher patterns

    Returns:
        Compact schema text (~500-2000 tokens per task)
    """
```

Example Cypher patterns remain hand-maintained in the `schema_context.py` module
(not auto-generated) since they encode domain knowledge about query patterns.
The auto-generated data handles the schema structure; the human-maintained code
handles the Cypher recipes.

### `get_graph_schema()` MCP tool refactor

The MCP tool delegates to `schema_for()` internally:

```python
@self.mcp.tool()
def get_graph_schema(
    scope: str = "overview",
) -> str:
    """Get graph schema context for Cypher query generation.

    Args:
        scope: What to return:
            'overview'  - Node labels with counts, vector indexes,
                         key relationships (< 2000 tokens)
            'signals'   - Signal domain schema slice
            'wiki'      - Wiki domain schema slice
            'imas'      - IMAS DD schema slice
            'trees'     - MDSplus tree schema slice
            'code'      - Code/ingestion schema slice
            'facility'  - Facility infrastructure schema slice
            'full'      - Full schema (legacy, 38K tokens)
            'label:X'   - Schema for specific label X

    Returns:
        Compact schema text formatted for Cypher query generation
    """
    if scope == "full":
        return _build_full_schema()  # Legacy behavior
    if scope == "overview":
        return schema_for()  # No labels, no task = overview
    if scope.startswith("label:"):
        label = scope.removeprefix("label:")
        return schema_for(label)
    # Treat scope as a task name
    return schema_for(task=scope)
```

**Key change**: Return type is `str` (compact text), not `dict`. This is more
token-efficient and agents immediately see the schema in a readable format.

Similarly, `get_dd_graph_schema` in the DD server's `SchemaTool` should also
delegate to `schema_for()` for the `imas` task group, ensuring both MCP servers
use the same auto-generated schema context.

## Phase 2: Domain Query Functions (High Impact, Medium Effort)

### Problem
Agents repeatedly compose the same multi-hop patterns: "find signals matching X at facility Y", "get wiki content about Z", "find IMAS paths for physics domain Q". Each time they write 5-15 lines of Cypher from scratch, often with errors.

### Solution: Pre-composed domain functions in the REPL

These are **REPL functions** that agents call from generated Python code. This maintains the codegen approach while eliminating boilerplate. They are NOT MCP tools because:
1. **Composability**: Agent chains `results = find_signals(...); imas = find_imas(results[0]['description'])` in one `python()` call
2. **Control flow**: Agent filters, transforms, aggregates in Python between calls
3. **Fewer tool calls**: One `python()` REPL call can invoke 3-5 domain functions
4. **Introspection**: Agent can `help(find_signals)` to discover parameters

```python
# ---- Signal discovery ----
def find_signals(
    query: str | None = None,
    facility: str | None = None,
    diagnostic: str | None = None,
    physics_domain: str | None = None,
    has_data: bool | None = None,
    limit: int = 20,
    include_access: bool = True,
) -> list[dict]:
    """Find facility signals by semantic search and/or filters.

    Combines vector search (when query given) with property filters.
    Automatically traverses to DataAccess for access templates.

    Returns flat dicts with: name, diagnostic, description,
    physics_domain, template_python, template_mdsplus, score
    """

# ---- Wiki search ----
def find_wiki(
    query: str,
    facility: str | None = None,
    k: int = 10,
    include_page_context: bool = True,
) -> list[dict]:
    """Semantic search over wiki content with page context.

    Returns: text, section, page_title, page_url, score
    """

# ---- IMAS path search ----
def find_imas(
    query: str,
    ids_filter: str | None = None,
    include_deprecated: bool = False,
    include_clusters: bool = True,
    limit: int = 20,
) -> list[dict]:
    """Find IMAS paths by semantic similarity.

    Returns: id, name, ids, documentation, data_type, units,
    physics_domain, cluster_labels, score
    """

# ---- Code search ----
def find_code(
    query: str,
    facility: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Semantic search over ingested code chunks.

    Returns: text (truncated), function_name, source_file,
    facility_id, score
    """

# ---- Tree exploration ----
def find_data_nodes(
    query: str | None = None,
    tree_name: str | None = None,
    facility: str | None = None,
    path_prefix: str | None = None,
    physics_domain: str | None = None,
    limit: int = 30,
) -> list[dict]:
    """Explore MDSplus tree nodes by search or filter.

    Returns: path, tree_name, description, unit, physics_domain,
    data_type, score
    """

# ---- Cross-domain: signal-to-IMAS mapping ----
def map_signals_to_imas(
    facility: str,
    diagnostic: str | None = None,
    physics_domain: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Find facility signals and their IMAS path mappings.

    Combines FacilitySignal -> DataAccess -> MAPS_TO_IMAS -> IMASPath
    traversal with enriched context from both sides.

    Returns: signal_name, diagnostic, imas_path, imas_documentation,
    template_python, match_type
    """

# ---- Facility overview ----
def facility_overview(facility: str) -> dict:
    """Single-call facility summary with counts and key entities.

    Returns: {
        counts: {signals: N, tree_nodes: N, wiki_pages: N, ...},
        diagnostics: [{name, signal_count, has_data_pct}],
        trees: [{name, node_count}],
        physics_domains: [{domain, signal_count}],
        recent_wiki: [{title, url}],
    }
    """
```

### Implementation: `imas_codex/graph/domain_queries.py`

Each function:
- Builds optimized Cypher internally (parameterized, projection-only — no full node returns)
- Combines vector search with property filters when both are provided
- Handles the `embed()` call internally (agent doesn't need to call embed separately)
- Returns flat dicts (no nested Neo4j objects)
- Includes appropriate LIMIT clauses
- Projects only relevant properties (token-efficient results)

### Relationship to existing REPL functions

Several REPL functions in `server.py` already do similar things (`search_imas`, `search_code`, `get_tree_structure`). The domain query functions should **replace** these where there is overlap, not sit alongside them:

| Existing | Replacement | Change |
|----------|-------------|--------|
| `search_imas()` | `find_imas()` | Richer: includes clusters, units, physics domain |
| `search_code()` | `find_code()` | Richer: direct graph query instead of ChunkSearch wrapper |
| `get_tree_structure()` | `find_data_nodes()` | Richer: adds semantic search + physics domain filter |
| `semantic_search()` | Kept | Low-level, still useful for ad-hoc index queries |
| `get_facility()` | Kept + `facility_overview()` | Overview adds aggregated graph stats |

## Phase 3: Graph Query Builder (Medium Impact, Medium Effort)

### Problem
For the 20% of queries that don't fit a pre-composed function, agents still struggle with Cypher. They need a thin helper for ad-hoc queries that provides guardrails.

### Solution: `graph_search()` — a schema-aware query builder

```python
def graph_search(
    label: str,
    *,
    where: dict | None = None,
    semantic: str | None = None,
    traverse: list[str] | None = None,
    return_props: list[str] | None = None,
    limit: int = 25,
    order_by: str | None = None,
) -> list[dict]:
    """Flexible graph query builder with schema validation.

    Uses auto-generated schema data to validate labels, properties,
    and automatically resolve vector indexes.

    Args:
        label: Node label to search (validated against schema)
        where: Property filters (validated against schema)
        semantic: Text for vector similarity (auto-resolves index)
        traverse: Relationship paths to follow
        return_props: Properties to project (default: key props from schema)
        limit: Maximum results
        order_by: Property to order by

    Examples:
        # Property filter
        graph_search('FacilitySignal',
                     where={'facility_id': 'tcv', 'physics_domain': 'magnetics'})

        # Semantic + traversal
        graph_search('WikiChunk',
                     semantic='plasma equilibrium reconstruction',
                     traverse=['-[:HAS_CHUNK]<-WikiPage'],
                     return_props=['text', 'section', 'page_title', 'page_url'])
    """
```

**Key design choice**: This is NOT Text2Cypher (no LLM in the loop). It's a deterministic query builder that translates structured parameters into Cypher. Fast, predictable, no hallucination risk. It uses the auto-generated `NODE_LABEL_PROPS` and `VECTOR_INDEXES` from `schema_context_data.py` to validate inputs and resolve indexes.

## Phase 4: Result Formatters (Low Effort, Quality of Life)

```python
def as_table(results: list[dict], columns: list[str] | None = None) -> str:
    """Format results as a compact markdown table."""

def as_summary(results: list[dict], group_by: str | None = None) -> str:
    """Summarize results with counts, showing top values per group."""

def pick(results: list[dict], *fields: str) -> list[dict]:
    """Project results to only the specified fields."""
```

Trivial but eliminate formatting code from every REPL call.

## Phase 5: MCP Tool Consolidation

### AgentsServer tools

Reduce from 9 to 7 MCP tools:

| Keep | Reason |
|------|--------|
| `python()` | Core REPL — primary agent interface |
| `get_graph_schema(scope)` | Context retrieval — delegates to `schema_for()` |
| `add_to_graph()` | Gated write — Pydantic validation, dedup, privacy filtering |
| `update_facility_config()` | Gated write — public/private config with YAML round-trip |
| `update_facility_infrastructure()` | Gated write — deep merge with comment preservation |
| `get_facility_infrastructure()` | Context retrieval — private infra data |
| `add_exploration_note()` | Gated write — timestamped, append-only |

| Remove as MCP tool | Why | Alternative |
|--------------------|-----|-------------|
| `update_facility_paths()` | Thin wrapper around `update_facility_infrastructure()` | Use `update_infrastructure()` in REPL |
| `update_facility_tools()` | Thin wrapper around `update_facility_infrastructure()` | Use `update_infrastructure()` in REPL |

### DD Server tools

No changes. The 10 tools serve external MCP clients without REPL access. The only refactor is `get_dd_graph_schema` delegating to `schema_for(task='imas')` internally.

## Auto-Generation Pipeline Summary

### What the build pipeline generates

```
uv sync
  └── hatch_build_hooks.py → build_models.py
       ├── gen_physics_domains.py   → core/physics_domain.py          (EXISTS)
       ├── LinkML gen-pydantic       → graph/models.py                (EXISTS)
       ├── LinkML gen-pydantic       → graph/dd_models.py             (EXISTS)
       ├── LinkML gen-pydantic       → config/models.py               (EXISTS)
       ├── gen_schema_reference.py   → agents/schema-reference.md     (EXISTS)
       └── gen_schema_context.py     → graph/schema_context_data.py   (NEW)
```

### What stays in sync automatically

| Component | Generated From | Benefit |
|-----------|---------------|---------|
| Node labels, properties, types | `schemas/facility.yaml` + `imas_dd.yaml` | Schema slices always match graph structure |
| Enum values | `schemas/common.yaml` + facility/DD schemas | No hardcoded enum lists to maintain |
| Relationship types + directionality | `schemas/facility.yaml` + `imas_dd.yaml` | Cypher patterns use correct rel types |
| Vector index names | `vector_index_name` annotations in schemas | `schema_for()` returns correct index names |
| Task group membership | `schemas/task_groups.yaml` | Validated at build time — referenced labels must exist |
| Config model fields | `schemas/facility_config.yaml` | Config validation stays current |

### What remains hand-maintained (and why)

| Component | Why Not Auto-Generated |
|-----------|----------------------|
| Example Cypher patterns | Encode domain-specific query recipes — require human knowledge of what agents commonly need |
| Domain query functions (`find_signals`, etc.) | Business logic — which traversals to compose, how to format results |
| Task group definitions (`schemas/task_groups.yaml`) | Semantic grouping decision — which labels belong together for which task |

## Implementation Order

### Sprint 1: Auto-Gen + Schema Context (Foundation) ✅ COMPLETE

1. **`schemas/task_groups.yaml`** — Define task groups
2. **`scripts/gen_schema_context.py`** — Build-time context generator
   - Reads facility.yaml, imas_dd.yaml, common.yaml, task_groups.yaml
   - Generates `graph/schema_context_data.py`
   - Validates task group labels exist in schemas
   - Gitignore the generated file
3. **`imas_codex/graph/schema_context.py`** — Runtime `schema_for()` function
   - Uses auto-generated data from `schema_context_data.py`
   - Hand-maintained example Cypher patterns per task
   - Compact text formatter
4. **Integrate into build pipeline** — Add to `build_models.py` and `hatch_build_hooks.py`
5. **Refactor `get_graph_schema()` MCP tool** — Delegate to `schema_for()` with `scope` parameter
6. **Refactor `get_dd_graph_schema` in SchemaTool** — Delegate to `schema_for(task='imas')`

### Sprint 2: Domain Queries (Biggest Productivity Win) ✅ COMPLETE

7. **`imas_codex/graph/domain_queries.py`** — Domain query functions
   - `find_signals()`, `find_wiki()`, `find_imas()`, `find_code()`
   - `find_data_nodes()`, `map_signals_to_imas()`, `facility_overview()`
8. **Update REPL registration** in `server.py`
   - Add domain query functions to `_repl_globals`
   - Replace overlapping functions (`search_imas` → `find_imas`, etc.)
   - Add `schema_for` to REPL
9. **MCP tool consolidation** — Remove `update_facility_paths/tools` as MCP tools

### Sprint 3: Query Builder + Polish ✅ COMPLETE

10. **`graph_search()`** query builder (uses auto-generated schema data)
11. **Result formatters** (`as_table`, `as_summary`, `pick`)
12. **Tests** for schema context generation, domain queries, query builder

### Sprint 4: Agent Instructions

13. **Update AGENTS.md** — Document new functions in Quick Reference, update recommended patterns
14. **Update agent prompts** — Prefer domain functions over raw Cypher
15. **Update `.instructions.md` files** — New interaction patterns

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| REPL calls for "find signals about electron density at TCV" | 8-15 | 1-2 |
| REPL calls for "what wiki content about equilibrium?" | 5-10 | 1 |
| Schema context tokens consumed | 38,000 | 500-2,000 |
| Schema drift from LinkML | Silent failures | Build-time validation |
| Cypher syntax errors | Frequent | Rare |
| Agent time to first useful result | 30-60s | 5-10s |
| MCP tool calls per exploration session | 30-50 | 10-15 |

## What NOT to Do

1. **Don't adopt neo4j-graphrag-python**: Its retriever/RAG pipeline is designed for chatbot QA. Our agents need programmatic access to graph data. Our `query()` + domain functions give agents the raw data they need to reason.

2. **Don't add Text2Cypher**: Adding an LLM to generate Cypher is slow, unreliable, and adds latency. Deterministic query builders are faster and more predictable.

3. **Don't replace the REPL with more MCP tools**: The codegen approach — composing multiple functions in a single `python()` call — is inherently more efficient than adding more MCP tools. MCP tools are one-shot; REPL functions chain.

4. **Don't make a "generic" schema summarizer using an LLM**: The schema is static — auto-generated data with hand-crafted examples will always be higher quality and faster than LLM-summarized schema.

5. **Don't hand-maintain schema data**: All schema structure data (labels, properties, enums, relationships, indexes) must be auto-generated from LinkML. Only domain-specific Cypher recipes and query logic stay hand-maintained. If you find yourself hardcoding a property name or enum value, it should come from the generated module instead.

## Key Design Decisions

### Why auto-generate from LinkML?

The existing build pipeline already generates Pydantic models, physics domain enums, and schema reference docs from LinkML. Extending it to generate schema context data is a natural evolution:

- **Single source of truth**: LinkML schemas define the graph structure. Everything downstream derives from them.
- **Build-time validation**: If a task group references a label that was removed from the schema, the build fails — not the runtime.
- **No drift**: When a new property is added to a schema, it automatically appears in `schema_for()` output on the next `uv sync`.
- **Existing pattern**: `gen_schema_reference.py` already does this for the markdown reference. `gen_schema_context.py` follows the same pattern for Python code.

### Why not third-party libraries (LangChain, LlamaIndex)?

Our agents don't need a RAG pipeline — they need **programmatic graph access with schema awareness**. Third-party libraries:
- Add a heavy dependency chain (LangChain alone pulls 50+ deps)
- Are designed around the "question → retrieved docs → LLM answer" loop, which isn't our pattern
- Don't understand our LinkML-derived schema
- Would need extensive customization to work with our embedding server

Our equivalent of `VectorCypherRetriever` is a 20-line domain function that uses our existing `embed()` + `query()` with a pre-built Cypher template.

### MCP tools: context retrieval + gated writes

The right MCP tools are those that either:
1. **Provide context** agents need *before* writing code (`get_graph_schema`, `get_facility_infrastructure`) — these return schema, configuration, or overview data that agents use to generate correct code
2. **Gate write operations** through validation (`add_to_graph`, `update_facility_infrastructure`, `add_exploration_note`) — these enforce Pydantic validation, privacy filtering, and deep-merge semantics that would be error-prone if agents wrote the code themselves

Everything else — composition, control flow, multi-step queries, formatting — happens in the REPL via `python()`.

### REPL functions vs MCP tools trade-offs

| Aspect | MCP Tools | REPL Functions |
|--------|-----------|----------------|
| Composability | None — each call is isolated | Full — chain in Python |
| Round trips | 1 per operation | 1 per `python()` call (N operations) |
| Error handling | Agent must retry via new tool call | try/except in same code block |
| Discovery | Tool descriptions in MCP schema | `help()`, `dir()`, `schema_for()` |
| Flexibility | Fixed parameters | Full Python expressiveness |
| Validation | Schema-enforced parameters | Developer must validate |
| Best for | Context retrieval, gated writes | Composition, exploration, analysis |

### Why hand-crafted task groups over automatic clustering?

Task groups (`schemas/task_groups.yaml`) are intentionally human-curated:
- They represent **agent workflows**, not graph topology — an agent exploring signals needs `DataAccess` and `Diagnostic` context even though those are separate node types
- Automatic clustering (e.g., by graph connectivity) would group mechanically but miss the semantic intent
- They are small (6 groups, ~30 labels total) and stable — new groups are rare
- They are validated at build time against the actual schema, catching any staleness