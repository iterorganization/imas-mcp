# Unified MCP Tools Plan

## Background

Two prior plans proposed similar architectural patterns for different servers:

1. **Graph-Native MCP Phase 7** ([features/graph-native-mcp.md](features/graph-native-mcp.md)) proposed `search_imas_deep`, `compare_dd_versions`, `get_imas_path_context`, and `search_imas_by_unit` for the IMAS MCP server (Docker, DD-only graph). Phases 0-6 are complete — the IMAS server is fully graph-native. Phase 7 was never implemented.

2. **Crafted MCP Tools** ([crafted-mcp-tools.md](crafted-mcp-tools.md)) proposed `search_docs`, `search_signals`, `search_code` for the agentic server (local, full graph). Motivated by the observation that agents make 13-24 separate `python()` REPL calls per research question. The problem is tool *shape*, not instructions.

Both plans converge on the same architecture: **multi-index semantic search → graph traversal enrichment → formatted text output**. This plan combines them into a single implementation targeting the agentic server, which has access to the full knowledge graph (facility + IMAS DD data) and can therefore provide cross-domain answers that neither plan achieves independently.

## Assessment of Prior Plans

### Graph-Native Phase 7

| Aspect | Assessment |
|--------|------------|
| Multi-index fan-out pattern | Strong — concrete Cypher examples for coordinate cross-referencing, cross-IDS discovery, version evolution, unit-aware search |
| `search_imas_deep` design | Good architecture, but limited to DD-only data. Cannot show facility signals, wiki docs, or code that reference IMAS paths |
| `compare_dd_versions` | Useful but niche — version comparison is important for codegen but rare for general research questions |
| `get_imas_path_context` | Good — full context for a single path is a common follow-up query |
| `search_imas_by_unit` | Too narrow for a dedicated tool — better as a parameter on `search_imas` |
| Target server | Wrong target for the tool-call problem. The IMAS MCP server is consumed by external clients (Cursor, Cline), not by our agents. The excessive-call problem is on the agentic server |

### Crafted MCP Tools

| Aspect | Assessment |
|--------|------------|
| 3-tool composable design | Strong — right granularity. An agent picks 1-3 tools per question |
| `search_docs` | Good — wiki + artifacts + images covers the documentation domain |
| `search_signals` | Good — signals + access templates + IMAS mappings + tree nodes covers data access |
| `search_code` | Good — code chunks + data references covers code examples |
| IMAS DD coverage | Weak — relegated to python() REPL. Misses the opportunity to exploit the full graph for cross-domain IMAS queries |
| Output format examples | Strong — concrete, shows what agents receive |
| Implementation pattern | Underspecified — says "parallel vector searches" but no Cypher examples |

### Synthesis

The crafted plan has the right framing (3 composable tools for the agentic server) and the graph-native Phase 7 has the right implementation detail (Cypher patterns for multi-index queries). The gap in both plans is cross-domain enrichment: when an agent asks about IMAS paths, the answer should include which facility signals map there, which wiki pages mention them, and which code files write to them. Only the agentic server (with the full graph) can provide this.

## Design

Four MCP tools on the agentic server. Each performs multi-index semantic search, traverses graph relationships for enrichment, and returns formatted text.

### Tool Overview

| Tool | Domain | Primary Question |
|------|--------|-----------------|
| `search_docs` | Documentation | "What does the knowledge base say about [topic] at [facility]?" |
| `search_signals` | Data access | "How do I access [quantity] at [facility]?" |
| `search_code` | Code examples | "Show me code that does [task] at [facility]" |
| `search_imas` | IMAS DD | "What IMAS paths represent [concept]?" |

### Naming Convention

| Layer | Prefix | Returns | Use |
|-------|--------|---------|-----|
| MCP tool | `search_` | Formatted text report | Agent calls directly, passes to user |
| REPL function | `find_` | Raw dicts | Custom compositions in python() |

### 1. `search_docs`

Searches documentation (wiki, artifacts, images) and enriches with cross-links to signals, tree nodes, and IMAS paths.

**Parameters**:
```
query: str              — natural language search text
facility: str           — facility id (required)
k: int = 10            — results per index
```

**Indexes**:
- `wiki_chunk_embedding` — core documentation content
- `wiki_artifact_desc_embedding` — linked PDFs, presentations
- `image_desc_embedding` — diagrams, schematics

**Enrichment query** (single Cypher after collecting IDs from vector search):
```cypher
UNWIND $chunk_ids AS cid
MATCH (c:WikiChunk {id: cid})
OPTIONAL MATCH (p:WikiPage)-[:HAS_CHUNK]->(c)
OPTIONAL MATCH (c)-[:DOCUMENTS]->(sig:FacilitySignal)
OPTIONAL MATCH (c)-[:DOCUMENTS]->(tn:TreeNode)
OPTIONAL MATCH (c)-[:MENTIONS_IMAS]->(ip:IMASPath)
OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(next:WikiChunk)
RETURN c.id, c.text, c.section,
       p.title AS page_title, p.url AS page_url,
       collect(DISTINCT sig.id) AS linked_signals,
       collect(DISTINCT tn.path) AS linked_tree_nodes,
       collect(DISTINCT ip.id) AS imas_refs,
       next.text AS next_chunk_text
```

**Output format**:
```
## Wiki Documentation (N chunks from M pages)

### Page: "Fishbone instabilities" (https://wiki.jet.efda.org/...)
**Section: Overview**
  [chunk text, truncated to ~200 chars]
  Signals: jet:mhd/fishbone_amplitude, jet:mhd/n1_mode
  IMAS: mhd_linear.time_slice[:].toroidal_mode[:].n_tor

**Section: Detection Methods**
  [chunk text...]

## Related Documents (K items)
  - "Fishbone analysis report.pdf" — from "MHD diagnostics at JET"
  - "Mirnov coil layout.png" — from "Magnetics diagnostic"
```

### 2. `search_signals`

Searches facility signals and enriches with data access templates, IMAS mappings, diagnostic context, and related tree nodes.

**Parameters**:
```
query: str                       — natural language search text
facility: str                    — facility id (required)
diagnostic: str | None = None    — optional diagnostic filter
physics_domain: str | None = None — optional physics domain filter
k: int = 10                      — results per index
```

**Indexes**:
- `facility_signal_desc_embedding` — signal descriptions
- `tree_node_desc_embedding` — MDSplus tree node descriptions

**Enrichment query**:
```cypher
UNWIND $signal_ids AS sid
MATCH (s:FacilitySignal {id: sid})
OPTIONAL MATCH (s)-[:DATA_ACCESS]->(da:DataAccess)
OPTIONAL MATCH (s)-[:BELONGS_TO_DIAGNOSTIC]->(diag:Diagnostic)
OPTIONAL MATCH (s)-[:SOURCE_NODE]->(tn:TreeNode)
OPTIONAL MATCH (da)-[:MAPS_TO_IMAS]->(ip:IMASPath)
OPTIONAL MATCH (ip)-[:HAS_UNIT]->(u:Unit)
RETURN s.id, s.name, s.description, s.physics_domain, s.unit,
       s.checked, s.example_shot,
       diag.name AS diagnostic_name, diag.category AS diagnostic_category,
       da.data_template AS access_template, da.access_type,
       da.imports_template, da.connection_template,
       tn.path AS tree_path, tn.tree_name,
       ip.id AS imas_path, ip.documentation AS imas_docs, u.symbol AS imas_unit
```

**Output format**:
```
## Signals (N matches)

### tcv:magnetics/ip (score: 0.92)
  Plasma current from magnetic measurements
  Diagnostic: magnetics (magnetics)
  Unit: A | Checked: shot 84000

  **Data access** (mdsplus):
    import MDSplus
    t = MDSplus.Tree('tcv_shot', shot)
    node = t.getNode('\\results::i_p')
    data = node.data(); time = node.dim_of().data()

  **IMAS mapping**: magnetics.ip.0d[:].value
    "Plasma current. Positive sign means anti-clockwise..."
    Unit: A

  **Tree node**: \RESULTS::I_P (tree: tcv_shot)

## Related Tree Nodes (M matches)
  \RESULTS::I_P — Plasma current (tree: tcv_shot, unit: A)
```

### 3. `search_code`

Searches ingested code and enriches with data references (MDSplus paths, TDI functions, IMAS paths) and directory context.

**Parameters**:
```
query: str                    — natural language search text
facility: str | None = None   — optional facility filter
k: int = 5                    — number of results
```

**Indexes**:
- `code_chunk_embedding` — code content
- `facility_path_desc_embedding` — directory descriptions

**Enrichment query**:
```cypher
UNWIND $chunk_ids AS cid
MATCH (cc:CodeChunk {id: cid})
OPTIONAL MATCH (ce:CodeExample)-[:HAS_CHUNK]->(cc)
OPTIONAL MATCH (cf:CodeFile {id: ce.source_file})
OPTIONAL MATCH (cf)-[:CONTAINS_REF]->(dr:DataReference)
OPTIONAL MATCH (dr)-[:RESOLVES_TO_TREE_NODE]->(tn:TreeNode)
OPTIONAL MATCH (dr)-[:RESOLVES_TO_IMAS_PATH]->(ip:IMASPath)
OPTIONAL MATCH (dr)-[:CALLS_TDI_FUNCTION]->(tdi:TDIFunction)
OPTIONAL MATCH (cf)-[:IN_DIRECTORY]->(fp:FacilityPath)
RETURN cc.id, substring(cc.text, 0, 500) AS text,
       cc.function_name, ce.source_file,
       cf.facility_id,
       collect(DISTINCT {type: dr.ref_type, raw: dr.raw_string,
               tree: tn.path, imas: ip.id, tdi: tdi.id}) AS data_refs,
       fp.path AS directory, fp.description AS dir_description
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
  **Data references**:
    MDSplus: \RESULTS::PSI → IMAS equilibrium.time_slice[:].profiles_2d[:].psi
    MDSplus: \RESULTS::R_AXIS
    TDI: TCV_EQ('r_axis', shot)
  **Directory**: /home/codes/liuqe — "LIUQE equilibrium reconstruction code"
```

### 4. `search_imas`

Searches IMAS Data Dictionary paths across multiple indexes, enriched with cluster membership, coordinate context, version history, and — uniquely — facility cross-references showing which signals, wiki pages, and code files reference each IMAS path. This is the tool that benefits most from the agentic server having the full graph.

**Parameters**:
```
query: str                             — natural language search text
ids_filter: str | None = None          — optional IDS name filter
facility: str | None = None            — optional facility for cross-references
include_version_context: bool = False  — include version history for top hits
k: int = 10                            — number of results
```

**Indexes**:
- `imas_path_embedding` — path documentation (physics meaning)
- `cluster_description_embedding` — thematic cluster descriptions

**Enrichment query** (path results):
```cypher
UNWIND $path_ids AS pid
MATCH (p:IMASPath {id: pid})
OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
OPTIONAL MATCH (p)-[:IN_CLUSTER]->(cl:IMASSemanticCluster)
OPTIONAL MATCH (p)-[:HAS_COORDINATE]->(coord)
OPTIONAL MATCH (p)-[:INTRODUCED_IN]->(intro:DDVersion)
RETURN p.id, p.name, p.ids, p.documentation, p.data_type,
       p.physics_domain, p.cocos_label_transformation,
       u.symbol AS unit,
       collect(DISTINCT cl.label) AS clusters,
       collect(DISTINCT coord.id) AS coordinates,
       intro.id AS introduced_in
```

**Facility cross-reference query** (when `facility` is provided):
```cypher
UNWIND $path_ids AS pid
MATCH (ip:IMASPath {id: pid})
OPTIONAL MATCH (da:DataAccess)-[:MAPS_TO_IMAS]->(ip)
OPTIONAL MATCH (sig:FacilitySignal)-[:DATA_ACCESS]->(da)
WHERE sig.facility_id = $facility
OPTIONAL MATCH (wc:WikiChunk)-[:MENTIONS_IMAS]->(ip)
WHERE (wc)-[:AT_FACILITY]->(:Facility {id: $facility})
OPTIONAL MATCH (dr:DataReference)-[:RESOLVES_TO_IMAS_PATH]->(ip)
OPTIONAL MATCH (cf:CodeFile)-[:CONTAINS_REF]->(dr)
WHERE cf.facility_id = $facility
RETURN ip.id,
       collect(DISTINCT sig.id) AS facility_signals,
       collect(DISTINCT wc.section) AS wiki_mentions,
       collect(DISTINCT cf.path) AS code_files
```

**Version context query** (when `include_version_context=True`, top 5 hits only):
```cypher
UNWIND $path_ids AS pid
MATCH (p:IMASPath {id: pid})
OPTIONAL MATCH (change:IMASPathChange)-[:FOR_IMAS_PATH]->(p)
WHERE change.semantic_change_type IN
      ['sign_convention', 'coordinate_convention', 'units', 'definition_clarification']
RETURN p.id,
       count(change) AS change_count,
       collect({version: change.version,
                type: change.semantic_change_type,
                summary: change.summary})[..5] AS notable_changes
```

**Output format**:
```
## IMAS Paths (N matches)

### equilibrium.time_slice[:].profiles_2d[:].psi (score: 0.95)
  "2D map of the poloidal magnetic flux on the (R,Z) grid"
  IDS: equilibrium | Type: FLT_2D | Unit: Wb
  Physics domain: equilibrium
  Clusters: "Equilibrium 2D profiles", "Poloidal flux quantities"
  Coordinates: grid.dim1, grid.dim2
  Introduced: DD 3.22.0
  COCOS: psi_like

  **At TCV**:
    Signals: tcv:equilibrium/psi_2d
    Wiki: mentioned in "Equilibrium reconstruction" (section: LIUQE)
    Code: /home/codes/liuqe/liuqe_reader.py

### equilibrium.time_slice[:].profiles_1d.psi (score: 0.88)
  [...]

## Related Clusters (M matches)
  "Equilibrium 2D profiles" (score: 0.82)
    Scope: Single-IDS | 12 paths
    Sample: profiles_2d.psi, profiles_2d.j_tor, profiles_2d.b_field_r
```

## What Stays in python() REPL

| Use Case | Function |
|----------|----------|
| IMAS path validation | `check_imas(paths)`, `fetch_imas(paths)`, `list_imas(paths)` |
| COCOS validation | `validate_cocos()`, `determine_cocos()`, `cocos_info()` |
| Signal-to-IMAS mapping table | `map_signals_to_imas(facility, ...)` |
| Facility overview (counts) | `facility_overview(facility)` |
| Flexible graph queries | `graph_search(label, where={})`, `query(cypher)` |
| Remote commands | `run()`, `check_tools()` |
| Schema introspection | `schema_for()`, `get_schema()` |
| Multi-step processing | Chained logic with intermediate variables |

The REPL `find_*` functions remain as building blocks for custom compositions. The `search_*` MCP tools orchestrate them internally.

## Architecture

### Implementation Pattern

Each tool follows the same 4-step pattern:

1. **Embed** — call `_get_encoder().embed_texts([query])` once
2. **Fan out** — run vector searches across N indexes, collecting scored node IDs
3. **Enrich** — run a single Cypher query that traverses relationships from those IDs
4. **Format** — produce structured text sections, truncated to ~4000 tokens

The embed step is shared. The fan-out queries can be issued sequentially (they're fast — vector search is sub-millisecond in Neo4j). The enrichment is a single Cypher query per domain. The formatter truncates text fields and omits empty sections.

### Where Code Lives

```
imas_codex/agentic/
  server.py          — tool registration in _register_tools()
  search_tools.py    — tool implementations (search_docs, search_signals, etc.)
  search_formatters.py — _format_docs_report(), _format_signals_report(), etc.
```

Tool functions are defined in `search_tools.py` and imported into `server.py` for registration. This keeps `server.py` from growing further. Formatters are in a separate module because they're pure functions (text in, text out) and straightforward to test.

### Shared Infrastructure

All four tools use these existing components:
- `GraphClient` — via `_get_repl()` lazy init (already exists)
- `Encoder` — via `_get_encoder()` lazy init (already exists)
- `_neo4j_error_message()` — for error formatting (already exists)

No new base classes needed. Each tool is a standalone function registered with `@self.mcp.tool()`.

### Error Handling

Errors return descriptive text (not exceptions — agents can't catch MCP tool errors):
- Embedding unavailable → "Embedding service unavailable. Use python() with graph_search() for property-based queries."
- Neo4j unavailable → standard `NEO4J_NOT_RUNNING_MSG`
- Empty results → "No [docs|signals|code|IMAS paths] found for '[query]' at [facility]"
- Partial results (some indexes empty) → return available sections, omit empty ones

### Token Budget

Each tool caps output at ~4000 tokens:
- `search_docs` — chunk text truncated to 200 chars, max 10 chunks
- `search_signals` — access templates truncated to 300 chars, max 10 signals
- `search_code` — code text truncated to 500 chars, max 5 chunks
- `search_imas` — documentation truncated to 150 chars, max 10 paths + 3 clusters

Agents use `python()` with `wiki_page_chunks()`, `find_signals()`, etc. for full untruncated content.

## Phased Implementation

### Phase 1: search_signals

Start here — signals have the richest graph connectivity (DataAccess, Diagnostic, TreeNode, IMASPath) and will validate the multi-index + traversal + format pattern.

**Tests first**:
- Mock GraphClient returning canned vector search + enrichment results
- Verify formatted output structure (sections, headers, truncation)
- Verify empty index handling (no tree nodes → section omitted)
- Verify error messages (no embedding, no Neo4j)

**Deliverable**: `search_signals` MCP tool registered and working.

### Phase 2: search_docs

Wiki domain — WikiChunk → WikiPage traversal, cross-links to signals/tree nodes/IMAS paths. Artifacts and images from secondary indexes.

**Tests first**:
- Mock vector search across 3 indexes (wiki_chunk, wiki_artifact, image)
- Verify page grouping (chunks from same page grouped under one header)
- Verify cross-link extraction (signals, IMAS refs from chunk relationships)

**Deliverable**: `search_docs` MCP tool registered and working.

### Phase 3: search_code

Code domain — CodeChunk → CodeFile → DataReference traversal.

**Tests first**:
- Mock code chunk vector search + enrichment
- Verify data reference grouping (MDSplus, TDI, IMAS refs from same file)
- Verify facility filter works

**Deliverable**: `search_code` MCP tool registered and working.

### Phase 4: search_imas

IMAS DD domain — the cross-domain tool. Uses multi-index search (path + cluster embeddings) and optionally enriches with facility cross-references and version context.

**Tests first**:
- Mock path vector search + cluster vector search
- Verify cluster section formatting
- Verify facility cross-reference query (signals, wiki mentions, code files)
- Verify version context query (changes, introduced_in)
- Verify `ids_filter` works

**Deliverable**: `search_imas` MCP tool registered and working.

### Phase 5: Cleanup

- Remove `_CHAIN_NUDGE` from python() output
- Simplify python() docstring — mark REPL functions as "for custom queries in python()"
- Update AGENTS.md Quick Reference to list the 4 MCP tools as the primary interface
- Consider removing `_generate_api_reference()` verbose function list — the MCP tools cover 80% of queries
- Run E2E test with a research question to verify tool-call count

## Relationship to Graph-Native MCP Phase 7

Phase 7 of the graph-native plan proposed tools for the IMAS MCP server (Docker, DD-only graph). Those tools remain relevant for external clients that only need DD data. This plan does not replace Phase 7 — it *supersedes* it for the agentic server context, where the full graph enables richer answers.

If Phase 7 is later implemented on the IMAS MCP server, its `search_imas_deep` can reuse the Cypher patterns from this plan's `search_imas` tool (minus the facility cross-reference queries, which are unavailable in the DD-only graph).

## Example: "Fishbone Instabilities at JET"

**Before** (13+ python() calls):
```
python("find_wiki('fishbone', facility='jet')")
python("wiki_page_chunks('fishbone', facility='jet')")
python("find_signals('fishbone', facility='jet')")
python("find_code('fishbone', facility='jet')")
python("find_imas('fishbone instabilities')")
python("as_table(pick(...))")
# ... more formatting, follow-ups
```

**After** (2-3 MCP calls):
```
search_docs("fishbone instabilities", facility="jet")
search_signals("fishbone", facility="jet")
search_imas("fishbone instabilities", facility="jet")
```

Each returns a complete, formatted report. If the agent needs to validate a specific IMAS path or check COCOS:
```
python("print(fetch_imas('mhd_linear.time_slice[:].toroidal_mode[:].n_tor'))")
python("validate_cocos(11, psi_axis=0.5, psi_edge=-0.2, ip=-1e6, b0=-5.0)")
```

Total: 2-5 calls instead of 13+.
