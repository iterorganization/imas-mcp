# Plan: Efficient Agent-Graph Interaction

## Problem Statement

Agents interacting with the imas-codex Neo4j graph are inefficient, generating 10s-100s of Python REPL calls for tasks that should take 1-3 calls. Root causes:

1. **Schema context flooding**: `get_graph_schema()` returns ~38,000 tokens (53 node types × 957 properties + 71 relationship types + 11 vector indexes). This either overwhelms the context window or agents skip calling it and write blind Cypher.

2. **No domain-aware query layer**: The REPL exposes `query(cypher)` — raw Cypher — forcing agents to compose queries from scratch each time. Agents don't know which properties exist, which indexes to use, or how to traverse relationships, so they probe incrementally (list labels → get properties → try a query → fix errors → retry).

3. **No composed operations**: Common multi-step patterns (semantic search → traverse → format) require 3-5 REPL calls that should be a single function call returning actionable results.

4. **Missing codegen scaffolding**: The REPL philosophy is "agents write code to solve problems" but there are no reusable building blocks for agents to compose from — no query builders, no traversal helpers, no typed result formatters.

## Design Principles

1. **Codegen-first**: Agents generate Python code, not individual tool calls. REPL functions return rich, structured results that chain together.
2. **Tiered schema context**: Instead of one 38K-token dump, provide gated, task-specific schema slices.
3. **Pre-composed queries for hot paths**: The 80% of common operations should be one function call. The 20% of novel queries use thin Cypher helpers with schema assistance.
4. **VectorCypher pattern**: Follow Neo4j's proven `VectorCypherRetriever` pattern — combine vector similarity with graph traversal in a single operation.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Agent (Claude, etc.)                       │
│  Generates Python code using REPL functions                  │
├──────────────────────────────────────────────────────────────┤
│  Layer 3: Domain Operations (NEW)                            │
│  find_signals(), find_code(), find_wiki(), map_to_imas()     │
│  explore_facility(), get_diagnostics(), ...                  │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: Graph Query Composers (NEW)                        │
│  graph_search(), graph_traverse(), graph_aggregate()         │
│  schema_for(), examples_for()                                │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: Core (EXISTS)                                      │
│  query(), embed(), semantic_search()                         │
├──────────────────────────────────────────────────────────────┤
│  Neo4j GraphClient                                           │
└──────────────────────────────────────────────────────────────┘
```

## Phase 1: Tiered Schema Context (High Impact, Low Effort)

### Problem
The `get_graph_schema()` MCP tool returns ALL 53 node types with ALL 957 properties — 38K tokens. This is too large for the agent context and too unfocused for any specific task.

### Solution: `schema_for()` — Contextual Schema Slices

Add a REPL function that returns **only the schema relevant to a specific task or node type**, formatted as compact Cypher-ready reference:

```python
def schema_for(
    *labels: str,
    task: str | None = None,
    include_relationships: bool = True,
    include_examples: bool = True,
) -> str:
    """Get compact, task-relevant schema context.
    
    Args:
        labels: Specific node labels (e.g., 'FacilitySignal', 'TreeNode')
        task: Predefined task name for curated schema slices:
              'signals' - FacilitySignal, DataAccess, Diagnostic, AccessCheck
              'wiki'    - WikiPage, WikiChunk, WikiArtifact, Image
              'imas'    - IMASPath, IDS, IMASSemanticCluster, DDVersion
              'code'    - CodeFile, CodeChunk, CodeExample, SourceFile
              'facility'- Facility, FacilityPath, FacilitySignal, TreeNode
              'trees'   - TreeNode, MDSplusTree, TreeNodePattern
        include_relationships: Include relevant relationship types
        include_examples: Include example Cypher patterns
    
    Returns:
        Compact schema text (~500-2000 tokens per task)
    """
```

**Task-based schema groups** (predefined slices covering 95% of use cases):

| Task | Node Types | Relationships | Size |
|------|-----------|---------------|------|
| `signals` | FacilitySignal, DataAccess, Diagnostic, AccessCheck | DATA_ACCESS, BELONGS_TO_DIAGNOSTIC, CHECKED_WITH, AT_FACILITY | ~1500 tokens |
| `wiki` | WikiPage, WikiChunk, WikiArtifact, Image | HAS_CHUNK, NEXT_CHUNK, HAS_IMAGE, HAS_ARTIFACT | ~1200 tokens |
| `imas` | IMASPath, IDS, IMASSemanticCluster, DDVersion, Unit | IN_IDS, IN_CLUSTER, INTRODUCED_IN, DEPRECATED_IN, HAS_UNIT | ~2000 tokens |
| `code` | CodeFile, CodeChunk, SourceFile, CodeExample | HAS_CHUNK, AT_FACILITY | ~800 tokens |
| `facility` | Facility, FacilityPath, FacilitySignal, TreeNode, Diagnostic | AT_FACILITY, IN_DIRECTORY, DATA_ACCESS | ~2000 tokens |
| `trees` | TreeNode, MDSplusTree, TreeNodePattern | HAS_NODE, FOLLOWS_PATTERN, SOURCE_NODE, AT_FACILITY | ~1000 tokens |

Each slice includes:
- Properties with types and whether they're indexed
- Key enum values (not all — just the ones agents need)
- 2-3 example Cypher patterns for common queries
- The correct vector index name for that domain

**Compact format** (not JSON — structured text for token efficiency):

```
## FacilitySignal (63,459 nodes)
ID: id (string, format: "{facility}:{diagnostic}/{name}")
Key props: name, diagnostic, description, physics_domain, status, unit
Data: has_data (bool), data_type, shape, shot_range_min/max
Vector: facility_signal_desc_embedding (on embedding)
Relationships:
  -[:DATA_ACCESS]-> DataAccess (template_python, template_mdsplus)
  -[:BELONGS_TO_DIAGNOSTIC]-> Diagnostic
  -[:AT_FACILITY]-> Facility
  -[:CHECKED_WITH]-> AccessCheck

## Example Cypher
# Find signals by physics domain at a facility
MATCH (s:FacilitySignal)-[:AT_FACILITY]->(f:Facility {id: $facility})
WHERE s.physics_domain = $domain
RETURN s.name, s.description, s.diagnostic LIMIT 20

# Semantic search + data access
CALL db.index.vector.queryNodes('facility_signal_desc_embedding', 10, $embedding)
YIELD node AS s, score
MATCH (s)-[:DATA_ACCESS]->(da:DataAccess)
RETURN s.name, s.description, da.template_python, score
```

### Implementation: `imas_codex/graph/schema_context.py`

A new module with:
- Task group definitions (which labels/rels per task)
- Compact text formatter (not JSON — optimized for token count)
- Node count injection (queries graph for cardinality to help agents estimate result sizes)
- Example Cypher patterns per task group baked in

Registered in the REPL as `schema_for()`.

## Phase 2: Domain Query Functions (High Impact, Medium Effort)

### Problem
Agents repeatedly compose the same multi-hop patterns: "find signals matching X at facility Y", "get wiki content about Z", "find IMAS paths for physics domain Q". Each time they write 5-15 lines of Cypher from scratch, often with errors.

### Solution: Pre-composed domain functions

These are **not MCP tools** — they are **REPL functions** that agents call from generated Python code. This maintains the codegen approach while eliminating boilerplate.

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
def find_tree_nodes(
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
- Handles the embed() call internally (agent doesn't need to call embed separately)
- Returns flat dicts (no nested Neo4j objects)
- Includes appropriate LIMIT clauses
- Projects only relevant properties (token-efficient results)

### Why REPL functions, not MCP tools?

1. **Composability**: Agent can chain `results = find_signals(...); imas = find_imas(results[0]['description'])`
2. **Control flow**: Agent can filter, transform, aggregate in Python between calls
3. **Fewer tool calls**: One python() REPL call can invoke 3-5 domain functions
4. **Type introspection**: Agent can `help(find_signals)` to discover parameters

## Phase 3: Graph Query Builder (Medium Impact, Medium Effort)

### Problem
For the 20% of queries that don't fit a pre-composed function, agents still struggle with Cypher. They need a thin helper for ad-hoc queries that provides guardrails.

### Solution: `graph_search()` — a query builder function

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
    """Flexible graph query builder with guardrails.
    
    Args:
        label: Node label to search (e.g., 'FacilitySignal')
        where: Property filters as {prop: value} or {prop: {'$gt': value}}
        semantic: Optional text for vector similarity search
        traverse: Relationship paths to traverse, e.g.:
                  ['-[:DATA_ACCESS]->DataAccess', 
                   '-[:AT_FACILITY]->Facility']
        return_props: Properties to include in results (default: key props)
        limit: Maximum results
        order_by: Property to order by
    
    Returns:
        List of flat result dicts
    
    Examples:
        # Property filter
        graph_search('FacilitySignal', 
                     where={'facility_id': 'tcv', 'physics_domain': 'magnetics'})
        
        # Semantic + traversal
        graph_search('WikiChunk', 
                     semantic='plasma equilibrium reconstruction',
                     traverse=['-[:HAS_CHUNK]<-WikiPage'],
                     return_props=['text', 'section', 'page_title', 'page_url'])
        
        # Aggregation
        graph_search('TreeNode',
                     where={'facility_id': 'tcv'},
                     return_props=['tree_name', 'count(*)'],
                     order_by='count(*) DESC')
    """
```

### Implementation

Internally builds parameterized Cypher from the declarative spec:
1. Resolves `label` against schema → validates it exists
2. Resolves `where` keys against schema → validates properties exist
3. If `semantic` provided, finds the correct vector index for that label automatically
4. Builds Cypher with proper traversal patterns
5. Projects only requested properties (or defaults for that label)

**Key design choice**: This is NOT Text2Cypher (no LLM in the loop). It's a deterministic query builder that translates structured parameters into Cypher. Fast, predictable, no hallucination risk.

## Phase 4: Result Formatters (Low Effort, Quality of Life)

After retrieval, agents often need to format results for their own reasoning or for user presentation. Add thin formatters:

```python
def as_table(results: list[dict], columns: list[str] | None = None) -> str:
    """Format results as a compact markdown table."""

def as_summary(results: list[dict], group_by: str | None = None) -> str:
    """Summarize results with counts, showing top values per group."""

def pick(results: list[dict], *fields: str) -> list[dict]:
    """Project results to only the specified fields."""
```

These are trivial but save agents from writing formatting code in every REPL call.

## Phase 5: MCP Schema Tool Refactor (Medium Impact)

### Problem
The MCP `get_graph_schema()` tool returns the full 38K-token schema. Agents that call it flood their context. Agents that skip it write broken Cypher.

### Solution
Add a `scope` parameter to the existing MCP tool:

```python
@self.mcp.tool()
def get_graph_schema(
    scope: str = "overview",
) -> dict[str, Any]:
    """Get graph schema context for query generation.
    
    Args:
        scope: What to return:
            'overview' - Node labels with counts, vector indexes, 
                        key relationships (< 2000 tokens)
            'signals' - Signal domain schema slice
            'wiki'    - Wiki domain schema slice
            'imas'    - IMAS DD schema slice
            'trees'   - MDSplus tree schema slice
            'code'    - Code/ingestion schema slice
            'facility'- Facility infrastructure schema slice
            'full'    - Full schema (legacy, 38K tokens)
            'label:X' - Schema for specific label X
    """
```

The `overview` scope (new default) returns:
- Node labels with approximate counts
- Vector index names with their labels
- Top relationship types
- ~1500 tokens total

This means agents can always call `get_graph_schema()` cheaply to orient themselves, then request a specific scope if they need details.

## Implementation Order

### Sprint 1: Schema Context + Domain Queries (Biggest ROI)

1. **`imas_codex/graph/schema_context.py`** — Task-based schema slices
   - Define task groups with labels, relationships, examples
   - Compact text formatter
   - Register `schema_for()` in REPL

2. **`imas_codex/graph/domain_queries.py`** — Domain query functions
   - `find_signals()`, `find_wiki()`, `find_imas()`, `find_code()`
   - `find_tree_nodes()`, `map_signals_to_imas()`, `facility_overview()`
   - All use parameterized Cypher, embed internally, return flat dicts

3. **Update REPL registration** in `server.py`
   - Add new functions to `_repl_globals`
   - Update docstring listing

4. **Update `get_graph_schema()` MCP tool** with `scope` parameter

5. **Update AGENTS.md** — Document new functions in Quick Reference

### Sprint 2: Query Builder + Polish

6. **`graph_search()`** query builder
7. **Result formatters** (`as_table`, `as_summary`, `pick`)
8. **Example Cypher patterns** embedded in schema_context per task
9. **Tests** for domain queries and schema context

### Sprint 3: Agent Instructions

10. **Update agent prompts** to prefer domain functions over raw Cypher
11. **Update `.instructions.md` files** with new interaction patterns
12. **Add usage examples** showing 1-call vs N-call patterns

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| REPL calls for "find signals about electron density at TCV" | 8-15 | 1-2 |
| REPL calls for "what wiki content about equilibrium?" | 5-10 | 1 |
| Schema context tokens consumed | 38,000 | 1,000-2,000 |
| Cypher syntax errors | Frequent | Rare |
| Agent time to first useful result | 30-60s | 5-10s |

## What NOT to Do

1. **Don't adopt neo4j-graphrag-python**: Its retriever/RAG pipeline is designed for a different use case (chatbot QA). Our agents need programmatic access to graph data, not LLM-mediated answers. Our `query()` + domain functions give agents the raw data they need to reason.

2. **Don't add Text2Cypher**: Adding an LLM to generate Cypher is slow, unreliable, and adds latency per query. Deterministic query builders (Phase 3) are faster and more predictable. Agents can always write raw Cypher when they need to.

3. **Don't replace the REPL with more MCP tools**: MCP tools are one-shot; REPL functions chain. The codegen approach of composing multiple functions in a single `python()` call is inherently more efficient. More tools = more round trips.

4. **Don't make a "generic" schema summarizer using an LLM**: The schema is static — hand-crafted task groups with example Cypher will always be higher quality and faster than LLM-summarized schema.

## Key Design Decisions

### Why not third-party libraries (LangChain, LlamaIndex)?

Our agents don't need a RAG pipeline — they need **programmatic graph access with schema awareness**. Third-party libraries:
- Add a heavy dependency chain (LangChain alone pulls 50+ deps)
- Are designed around the "question → retrieved docs → LLM answer" loop, which isn't our pattern
- Don't understand our LinkML-derived schema
- Would need extensive customization to work with our embedding server

Our equivalent of `VectorCypherRetriever` is a 20-line function that uses our existing `embed()` + `query()` with a pre-built Cypher template. No need for an abstraction layer.

### Why REPL functions over MCP tools?

| Aspect | MCP Tools | REPL Functions |
|--------|-----------|----------------|
| Composability | None — each call is isolated | Full — chain in Python |
| Round trips | 1 per operation | 1 per `python()` call (N operations) |
| Error handling | Agent must retry via new tool call | Try/except in same code block |
| Discovery | Tool descriptions in schema | `help()`, `dir()`, `schema_for()` |
| Flexibility | Fixed parameters | Full Python expressiveness |

### Why hand-crafted schema groups over LLM-generated summaries?

- Schema is static (changes only on `uv sync`)
- We know exactly which properties agents need per task
- Example Cypher patterns are higher quality when hand-written with domain knowledge
- Zero latency (no LLM call needed)
- Deterministic — same context every time
