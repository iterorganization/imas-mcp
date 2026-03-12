# Crafted MCP Tools — Strategy & Design

## Problem

Agents answering physics research questions make 13-24 separate `python()` REPL calls because each domain query function (find_wiki, find_signals, find_code, find_imas, etc.) returns raw data that the agent must format, correlate, and follow up on. Instruction-based nudges ("combine queries in one call") don't reliably change LLM tool-call planning behavior. The problem is tool *shape*, not instructions.

## Strategy

Add 3 crafted MCP tools that each perform comprehensive research within a domain. Each tool:

1. Searches across multiple vector indexes relevant to the domain
2. Follows graph relationships from results to gather context
3. Returns a formatted, self-contained report

The tools are **composable, not monolithic**: the agent calls 1–3 tools depending on what it needs. A topic question might use `search_docs` + `search_signals`. A data access question might use only `search_signals`. A code review might use `search_code` + `search_signals`.

`python()` REPL remains for custom queries, raw Cypher, IMAS DD operations, COCOS validation, and chained processing that doesn't fit the crafted tools.

### Design Principles

- **One call, one domain, complete answer** — each tool returns everything the agent needs within that domain. No follow-up queries required for the 80% case.
- **Semantic search first, traversal second** — vector search finds the entry points, graph traversal enriches them with related context. This exploits the graph's depth.
- **Formatted for consumption** — output is structured text (sections with headers), not raw dicts. The agent can pass it directly to the user without reformatting.
- **Facility-scoped** — all tools require a facility parameter. Cross-facility queries are the 20% case handled by python().

## Proposed Tools

### 1. `search_docs`

**Purpose**: "What does the knowledge base say about [topic] at [facility]?"

Answers the documentation research question in one call. Replaces `find_wiki()` → `wiki_page_chunks()` → repeat pattern.

**Parameters**:
- `query: str` — natural language search text
- `facility: str` — facility id
- `k: int = 10` — number of results per index

**Indexes searched**:
| Index | Why |
|-------|-----|
| `wiki_chunk_embedding` | Core documentation content |
| `wiki_artifact_desc_embedding` | Linked PDFs, presentations, technical documents |
| `image_desc_embedding` | Diagrams, schematics referenced in docs |

**Graph traversals from results**:
```
WikiChunk ←[:HAS_CHUNK]- WikiPage          → page title, URL
WikiChunk -[:DOCUMENTS]→ FacilitySignal     → cross-linked signals
WikiChunk -[:DOCUMENTS]→ DataNode           → cross-linked tree nodes
WikiChunk -[:MENTIONS_IMAS]→ IMASPath       → referenced IMAS paths
WikiChunk -[:NEXT_CHUNK]→ WikiChunk         → surrounding context
WikiArtifact ←[:HAS_ARTIFACT]- WikiPage     → parent page context
Image ←[:HAS_IMAGE]- WikiPage               → parent page context
```

**Output format** (formatted text sections):
```
## Wiki Documentation (N chunks from M pages)

### Page: "Fishbone instabilities" (https://wiki.jet.efda.org/...)
Section: Overview
  [chunk text...]
  Linked signals: jet:mhd/fishbone_amplitude, jet:mhd/n1_mode
  IMAS refs: mhd_linear.time_slice[:].toroidal_mode[:].n_tor

Section: Detection Methods
  [chunk text...]

### Page: "MHD diagnostics at JET" (https://...)
  [chunk text...]

## Related Artifacts (K items)
  - "Fishbone analysis report.pdf" (from page "MHD diagnostics at JET")
  - "Mirnov coil layout.png" (from page "Magnetics diagnostic")
```

### 2. `search_signals`

**Purpose**: "How do I access [quantity] at [facility]? What signals exist and how do they map to IMAS?"

Answers the data access question in one call. Replaces `find_signals()` → inspect result → `python("query('MATCH (s:FacilitySignal)...')")` → format pattern.

**Parameters**:
- `query: str` — natural language search text (e.g., "electron density", "plasma current")
- `facility: str` — facility id
- `diagnostic: str | None = None` — optional diagnostic filter
- `physics_domain: str | None = None` — optional physics domain filter
- `k: int = 10` — number of results

**Indexes searched**:
| Index | Why |
|-------|-----|
| `facility_signal_desc_embedding` | Core signal search |
| `data_node_desc_embedding` | MDSplus tree nodes matching the concept |

**Graph traversals from results**:
```
FacilitySignal -[:DATA_ACCESS]→ DataAccess       → access template (code snippet)
FacilitySignal -[:BELONGS_TO_DIAGNOSTIC]→ Diagnostic → diagnostic name, category
FacilitySignal -[:HAS_DATA_SOURCE_NODE]→ DataNode          → MDSplus tree path
DataAccess -[:MAPS_TO_IMAS]→ IMASPath             → IMAS standard path + docs
DataNode (from separate index search)              → additional tree context
```

**Output format**:
```
## Signals (N matches)

### tcv:magnetics/ip (score: 0.92)
  Description: Plasma current from magnetic measurements
  Diagnostic: magnetics (category: magnetics)
  Physics domain: magnetics
  Unit: A

  Access template (MDSplus):
    import MDSplus
    t = MDSplus.Tree('tcv_shot', shot)
    node = t.getNode('\\results::i_p')
    data = node.data()
    time = node.dim_of().data()

  IMAS mapping: magnetics.ip.0d[:].value
    Documentation: "Plasma current. Positive sign means anti-clockwise..."

  MDSplus tree node: \RESULTS::I_P (tree: tcv_shot)

### tcv:magnetics/ip_marte (score: 0.78)
  [...]

## Related Tree Nodes (M matches from tree_node index)
  \RESULTS::I_P — "Plasma current" (tree: tcv_shot, unit: A)
  \RESULTS::I_P:FOO — ...
```

### 3. `search_code`

**Purpose**: "Show me code that does [task] at [facility]."

Answers the code example question in one call. Replaces `find_code()` → `python("query('MATCH (cc:CodeChunk)...')")` pattern.

**Parameters**:
- `query: str` — natural language search text
- `facility: str | None = None` — optional facility filter
- `k: int = 5` — number of results

**Indexes searched**:
| Index | Why |
|-------|-----|
| `code_chunk_embedding` | Core code search |
| `facility_path_desc_embedding` | Find relevant directories/analysis codes |

**Graph traversals from results**:
```
CodeChunk ←[:HAS_CHUNK]- CodeFile               → source file path, facility
CodeFile -[:CONTAINS_REF]→ DataReference         → what data it accesses
DataReference -[:RESOLVES_TO_NODE]→ DataNode → MDSplus paths used
DataReference -[:RESOLVES_TO_IMAS_PATH]→ IMASPath → IMAS paths used
DataReference -[:CALLS_TDI_FUNCTION]→ TDIFunction → TDI functions called
CodeFile -[:IN_DIRECTORY]→ FacilityPath           → parent directory context
```

**Output format**:
```
## Code Examples (N matches)

### read_equilibrium() — /home/codes/liuqe/liuqe_reader.py (score: 0.89)
  ```python
  def read_equilibrium(shot, time=None):
      tree = MDSplus.Tree('tcv_shot', shot)
      psi = tree.getNode('\\results::psi').data()
      ...
  ```
  Data references:
    - MDSplus: \RESULTS::PSI → maps to IMAS equilibrium.time_slice[:].profiles_2d[:].psi
    - MDSplus: \RESULTS::R_AXIS
    - TDI: TCV_EQ('r_axis', shot)
  Directory: /home/codes/liuqe (interest_score: 0.85, purpose: analysis_code)

### compute_density() — /home/codes/thomson/fit.py (score: 0.72)
  [...]
```

## What Stays in python() REPL

The REPL retains its role for operations that don't fit the crafted tool pattern:

| Use Case | REPL Function |
|----------|---------------|
| IMAS DD lookup | `search_imas()`, `fetch_imas()`, `list_imas()`, `check_imas()` |
| COCOS validation | `validate_cocos()`, `determine_cocos()`, `cocos_info()` |
| Signal-to-IMAS mapping table | `map_signals_to_imas()` |
| Facility overview (counts) | `facility_overview()` |
| Flexible graph queries | `graph_search()`, `query()` |
| Custom traversals | Raw Cypher via `query()` |
| Remote commands | `run()`, `check_tools()` |
| Schema introspection | `schema_for()`, `get_schema()` |
| Chained processing | Multi-step logic with intermediate variables |

The REPL functions (`find_wiki`, `find_signals`, `find_code`, `find_data_nodes`, `find_imas`) remain available for custom compositions — they're the building blocks that the MCP tools orchestrate internally.

## Architecture

### Implementation Pattern

Follow the existing domain query architecture. Each MCP tool:

1. Calls `_get_encoder()` once to get the embedding
2. Runs parallel vector searches via `gc.query()` against multiple indexes
3. Collects unique node IDs from all vector results
4. Runs a single enrichment Cypher query that traverses all relevant relationships from those IDs
5. Formats the combined results into sections

The MCP tool functions are registered in `_register_tools()` alongside the existing tools. They use the same lazy REPL initialization for `GraphClient` and `Encoder` access.

### Formatter

Each tool has a dedicated `_format_*` function (e.g., `_format_docs_report`, `_format_signals_report`, `_format_code_report`) that takes raw query results and produces the formatted text output. These live alongside the tool definitions in `server.py` or in a new `imas_codex/llm/formatters.py` module if they grow large.

The formatters should truncate text fields to keep output under ~4000 tokens per tool call. Agents can use `python()` for full text retrieval when needed.

### Error Handling

If the embedding service is unavailable, return a clear message suggesting the agent use `python()` with `graph_search()` (property-only, no embeddings) as a fallback. Don't raise — the agent can't catch exceptions from MCP tools.

If Neo4j is unavailable, return the standard `NEO4J_NOT_RUNNING_MSG`.

## Naming Convention

MCP tools: `search_docs`, `search_signals`, `search_code`. Short, verb-first, matches the domain. The `search_` prefix signals "I do comprehensive research", distinguishing from the REPL's `find_` prefix which returns raw data.

| Layer | Prefix | Returns | Example |
|-------|--------|---------|---------|
| MCP tool | `search_` | Formatted report (text) | `search_docs("fishbone", facility="jet")` |
| REPL function | `find_` | Raw dicts (data) | `find_wiki("fishbone", facility="jet")` |

## Example: "Fishbone Instabilities at JET"

**Before** (13+ python() calls):
```
python("find_wiki('fishbone instabilities', facility='jet')")
python("wiki_page_chunks('fishbone', facility='jet')")
python("find_signals('fishbone', facility='jet')")
python("find_code('fishbone', facility='jet')")
python("find_imas('fishbone instabilities')")
python("as_table(pick(...))")
# ... more formatting, follow-ups
```

**After** (2 MCP calls):
```
search_docs("fishbone instabilities", facility="jet")
search_signals("fishbone", facility="jet")
```

Each returns a complete, formatted report. If the agent needs IMAS DD details, one more call:
```
python("print(fetch_imas('mhd_linear.time_slice[:].toroidal_mode[:].n_tor'))")
```

Total: 2-3 calls instead of 13+.

## Considerations

### Token Budget

Each tool's output should be capped around 4000 tokens. For `search_docs`, this means truncating chunk text to ~200 chars each and limiting to the top pages. The full text is always available via `python("wiki_page_chunks(...)")`.

### Index Availability

Not all facilities will have all indexes populated. The tools should gracefully handle empty results from any index — return the sections that have data and omit empty sections.

### Deduplication

When the same concept appears in both wiki and signal results (e.g., a WikiChunk that `-[:DOCUMENTS]→ FacilitySignal`), present it in both sections with a cross-reference rather than deduplicating. The agent benefits from seeing the connection from both perspectives.

### REPL Cleanup

After implementing the MCP tools, consider:
- Removing `_CHAIN_NUDGE` — the crafted tools make it unnecessary
- Simplifying the python() docstring — less pressure to educate about chaining
- Keeping `_generate_api_reference()` but marking REPL functions as "for custom queries within python()"
