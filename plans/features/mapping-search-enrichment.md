# Mapping Pipeline Search Enrichment

**Status**: Plan  
**Created**: 2026-03-15  
**Scope**: `imas_codex/ids/`, `imas_codex/llm/search_tools.py`, `imas_codex/tools/graph_search.py`

## Executive Summary

The IMAS mapping pipeline (`imas map run`) generates signal-level mappings from
facility signal sources to IMAS IDS fields. While the pipeline already uses
physics-domain filtering and per-source semantic search, it leaves significant
value on the table by not leveraging:

1. **Scored wiki and code content** — 982 wiki pages and 569 code files in
   pf_active-relevant domains are never consulted during mapping
2. **Score dimension filtering** — 6 content dimensions and 11 code/path
   dimensions exist on graph nodes but are unused in search queries
3. **Cross-domain semantic bridging** — source embeddings can be cosine-compared
   directly to IMAS target embeddings for high-precision candidate ranking
4. **MCP tool discoverability** — agents calling `search_docs`, `search_code`,
   `search_signals` have no visibility into available physics domains or score
   dimensions at call time

## Current State

### Pipeline Architecture

```
gather_context → assign_sections → map_signals → discover_assembly → validate → persist
     │                │                 │
     │                │                 ├─ per-source semantic search (IMAS only)
     │                │                 ├─ code references (SignalSource→CodeChunk)
     │                │                 └─ unit analysis + COCOS context
     │                │
     │                └─ ALL sources + full IDS subtree + semantic hits + clusters
     │
     ├─ physics domain scoping (target IDS domains → filter sources)
     ├─ signal source query (enriched sources with metadata)
     ├─ IDS subtree fetch
     ├─ per-source semantic search against imas_node_embedding
     ├─ cross-facility mapping precedent
     └─ section cluster listing
```

### What Works Well

- **Physics domain scoping** reduces candidate sources dramatically (e.g., JET
  pf_active: 2288 → 197 sources, 91% reduction)
- **Per-source semantic search** in `gather_context()` runs each source's
  description against `imas_node_embedding` to find candidate IMAS fields
- **Hybrid vector + text search** on signals, wiki, and code via the MCP tools
- **Rich enrichment metadata** — signals have descriptions, keywords, units,
  COCOS, diagnostics; IMAS nodes have enriched descriptions, keywords, physics
  domains

### Gap Analysis

#### Gap 1: Wiki and Code Content Not Used in Mapping

The mapping pipeline calls `fetch_source_code_refs()` which traverses
`SignalSource → representative FacilitySignal → SignalNode → CodeChunk`.
This narrow traversal only finds code that directly reads the specific signal.

**Missing content** (verified via live graph queries):
- **Wiki pages**: 982 pages in `magnetic_field_systems`/`plasma_control` domains
  with `score_composite > 0.3`, including pages like "Magnetostatic Ontology"
  (imas_relevance=0.80), "EFCC" (data_access=0.60), "PIW:RTPS:Handbook"
  (data_access=0.85) — all highly relevant to pf_active mapping
- **Code files**: 569 facility paths in relevant domains with scored dimensions,
  including CREATE-NL simulation suites (imas=0.95), magnetic field analysis
  tools (imas=0.92), ripple analysis environments
- **Code chunks**: Semantic search against `code_chunk_embedding` using source
  descriptions yields relevant code like pfcoil geometry parsers (score=0.877),
  EFIT XML handlers (0.868), coil geometry extractors (0.862)

**Impact**: The LLM makes mapping decisions without access to domain expertise
documented in wiki pages or data access patterns demonstrated in code files.

#### Gap 2: Score Dimension Filtering Unused

Every wiki page has 6 content scoring dimensions and every facility path has
11 code/path scoring dimensions. These are stored on graph nodes but **never
used in any search query**.

**Content scoring dimensions** (WikiPage, Document):
```
score_data_documentation  — Signal tables, node lists, shot databases
score_physics_content     — Physics explanations, methodology, theory
score_code_documentation  — Software docs, API references, usage guides
score_data_access         — MDSplus paths, TDI expressions, access methods
score_calibration         — Calibration info, conversion factors, sensor specs
score_imas_relevance      — IMAS integration, IDS references, mapping hints
```

**Code/path scoring dimensions** (FacilityPath, SourceFile):
```
score_modeling_code       — Forward modeling/simulation code
score_analysis_code       — Experimental analysis code
score_operations_code     — Real-time operations code
score_data_access         — Data access tools (shared with content)
score_workflow            — Workflow/orchestration
score_visualization       — Visualization tools
score_documentation       — Documentation
score_imas                — IMAS relevance
score_convention          — Convention handling (COCOS, units)
score_modeling_data        — (path-only) Modeling data
score_experimental_data    — (path-only) Experimental data
```

**Mapping-relevant dimensions**:
- `score_data_access` — Most critical. When building signal mappings, we need
  code and docs that show _how_ data is accessed. Filtering on this dimension
  surfaces TDI expressions, MDSplus patterns, and API examples.
- `score_imas_relevance` / `score_imas` — Directly relevant. Content scored
  high on IMAS relevance explicitly discusses IDS structure, mapping patterns,
  or IMAS integration.
- `score_convention` — For COCOS/sign convention handling. Code scored high on
  this dimension contains sign flip logic, coordinate transformations, and unit
  conversion patterns.
- `score_calibration` — For unit conversion context. Calibration docs contain
  conversion factors and sensor specifications.

#### Gap 3: No Cross-Domain Semantic Bridge

The pipeline currently performs unidirectional semantic search: source
description → IMAS node embeddings. A richer approach would compute
**bidirectional embedding matches** across all content types:

```
Source content (signals, wiki, code) ←→ Target content (IMAS nodes)
```

**Verified via live testing**: Embedding a signal source description
("turns per element for Poloidal Field coil 10") and searching against
`imas_node_embedding` yields `pf_active/coil/element/turns_with_sign`
at cosine=0.893. The same embedding against `code_chunk_embedding` yields
pfcoil geometry code at 0.877. Against `wiki_chunk_embedding` yields
disruption database entries at 0.865.

This means we can build a **match matrix** where rows are source signals
and columns are IMAS target paths, with each cell being a cosine similarity.
High-scoring cells are strong mapping candidates. Adding wiki and code chunks
to the source side provides additional bridging evidence.

#### Gap 4: MCP Tool Docstrings Lack Dynamic Content

Agents calling `search_signals`, `search_docs`, `search_code` have no way to
know what `physics_domain` values are valid or what score dimensions exist.
The `physics_domain` parameter on `search_signals` accepts a string but
doesn't list the 22 valid enum values. The `search_docs` and `search_code`
tools don't even accept a `physics_domain` parameter.

## Phased Implementation

### Phase 1: Score-Dimension Filtering in Search Tools

**Goal**: Enable filtering search results by score dimensions across all three
content search tools.

**Targets**:
- `imas_codex/llm/search_tools.py` — `_search_docs()`, `_search_code()`
- `imas_codex/llm/search_tools.py` — `_vector_search_wiki_chunks()`,
  `_vector_search_code_chunks()`
- `imas_codex/llm/server.py` — MCP tool registration docstrings

**Changes**:

1. **Add `physics_domain` parameter to `_search_docs()` and `_search_code()`**

   Both functions currently lack physics domain filtering. Add a `physics_domain`
   parameter that filters WikiPage/FacilityPath nodes by their `physics_domain`
   property, mirroring the existing pattern in `_vector_search_signals()`.

   For wiki: filter via `WikiChunk → WikiPage` traversal where
   `WikiPage.physics_domain = $domain`.

   For code: filter via `CodeChunk → SourceFile → FacilityPath` traversal where
   `FacilityPath.physics_domain = $domain`, or directly on
   `CodeChunk.facility_id` combined with path-level domain.

2. **Add `min_score` parameter with optional `score_dimension` selector**

   ```python
   def _search_docs(
       query: str,
       facility: str,
       *,
       physics_domain: str | None = None,   # NEW
       min_score: float | None = None,       # NEW
       score_dimension: str | None = None,   # NEW — defaults to score_composite
       ...
   )
   ```

   When `min_score` is set, filter results to only those where
   `node.{score_dimension} >= min_score`. When `score_dimension` is None,
   use `score_composite`. Valid dimensions come from
   `CONTENT_SCORE_DIMENSIONS` (docs) or `CODE_SCORE_DIMENSIONS` (code).

3. **Add the same parameters to `_search_signals()`**

   Signals don't have content scoring dimensions but do have a
   `score_composite`-equivalent via the SignalSource `status` and enrichment.
   The `physics_domain` filter already exists. Consider adding a `min_score`
   based on the vector similarity threshold.

**Validation**:
- Unit tests: parametrize across all 22 physics domains confirming filter works
- Integration test: `search_docs("coil current", "jet", physics_domain="magnetic_field_systems")`
  should return fewer, more relevant results than unfiltered

### Phase 2: Wiki and Code Context in Mapping Pipeline

**Goal**: Enrich the LLM prompt context in `gather_context()` and
`map_signals()` with relevant wiki and code content, filtered by physics
domain and score dimensions.

**Targets**:
- `imas_codex/ids/tools.py` — new functions
- `imas_codex/ids/mapping.py` — `gather_context()`, `map_signals()`
- `imas_codex/llm/prompts/mapping/signal_mapping.md` — prompt template

**Changes**:

1. **New tool function: `fetch_wiki_context()`**

   ```python
   def fetch_wiki_context(
       facility: str,
       physics_domains: list[str],
       *,
       query: str | None = None,
       min_imas_relevance: float = 0.5,
       k: int = 10,
       gc: GraphClient | None = None,
   ) -> list[dict[str, Any]]:
       """Fetch wiki chunks relevant to the mapping task.

       Uses physics_domain + score_imas_relevance filtering to find
       high-value documentation, then optionally narrows with semantic search.
       """
   ```

   Implementation: Query WikiChunks where parent WikiPage has
   `physics_domain IN $domains AND score_imas_relevance >= $min_score`,
   optionally run vector search on `wiki_chunk_embedding` to rank by
   query relevance. Return chunk text + page title + scores.

2. **New tool function: `fetch_code_context()`**

   ```python
   def fetch_code_context(
       facility: str,
       physics_domains: list[str],
       *,
       query: str | None = None,
       score_dimension: str = "score_data_access",
       min_score: float = 0.5,
       k: int = 10,
       gc: GraphClient | None = None,
   ) -> list[dict[str, Any]]:
       """Fetch code chunks demonstrating data access patterns.

       Filters FacilityPaths by physics_domain and score_data_access,
       then retrieves their CodeChunks. This surfaces code that shows
       HOW signals are read, complementing the narrow
       fetch_source_code_refs() which only finds code for a single signal.
       """
   ```

3. **Integrate into `gather_context()`**

   After querying signal sources and before returning, call both new functions
   using `target_domains` from the physics domain scoping step:

   ```python
   wiki_context = fetch_wiki_context(
       facility, target_domains,
       min_imas_relevance=0.5, k=15, gc=gc,
   )
   code_context = fetch_code_context(
       facility, target_domains,
       score_dimension="score_data_access",
       min_score=0.5, k=15, gc=gc,
   )
   ```

   Add to the returned context dict as `"wiki_context"` and `"code_context"`.

4. **Add to `map_signals()` prompt**

   Pass wiki and code context to the `signal_mapping` prompt template:
   ```python
   prompt = _render_prompt(
       "signal_mapping",
       ...
       wiki_context=_format_wiki_context(context.get("wiki_context", [])),
       code_data_access=_format_code_context(context.get("code_context", [])),
   )
   ```

   Add corresponding sections to `signal_mapping.md`:
   ```markdown
   ### Domain Documentation

   Wiki documentation relevant to this physics domain (filtered by
   IMAS relevance score):

   {{ wiki_context }}

   ### Data Access Code Patterns

   Code examples showing how similar signals are accessed at this facility
   (filtered by data_access score):

   {{ code_data_access }}
   ```

5. **Per-source wiki/code narrowing** (optional for Phase 2)

   In `map_signals()` per-section loop, optionally run semantic search on wiki
   and code using the source description for per-source context, similar to
   the existing `source_candidates` pattern for IMAS paths. This is lower
   priority as domain-level context may suffice initially.

**Validation**:
- Run `imas map run --no-persist jet pf_active` before and after, compare
  prompt lengths and binding quality
- Check cost increase is acceptable (wiki/code context adds tokens)
- Verify no regression on existing binding quality

### Phase 3: Semantic Match Matrix (Cross-Domain Bridge)

**Goal**: Implement a semantic bridge that computes cosine similarity between
source content embeddings and target IMAS node embeddings, returning a ranked
match matrix that the LLM uses as strong candidate signals.

**Targets**:
- `imas_codex/ids/tools.py` — new function `compute_semantic_matches()`
- `imas_codex/ids/mapping.py` — `gather_context()`, `map_signals()`
- `imas_codex/llm/prompts/mapping/signal_mapping.md`

**Changes**:

1. **New function: `compute_semantic_matches()`**

   ```python
   def compute_semantic_matches(
       source_descriptions: list[tuple[str, str]],  # (source_id, text)
       target_ids_name: str,
       *,
       gc: GraphClient | None = None,
       k_per_source: int = 5,
       include_wiki: bool = True,
       include_code: bool = True,
   ) -> dict[str, list[dict[str, Any]]]:
       """Compute semantic match vectors between sources and targets.

       For each source, embeds its description and searches against:
       1. imas_node_embedding (primary: target IMAS fields)
       2. wiki_chunk_embedding (bridging: domain documentation)
       3. code_chunk_embedding (bridging: data access patterns)

       Returns a dict mapping source_id → ranked match list, where each
       match has {target_id, score, content_type, excerpt}.
       """
   ```

   This function extends the existing per-source semantic search in
   `gather_context()` which currently only searches `imas_node_embedding`.
   The new version searches all three indexes and returns a unified match
   set per source.

2. **Batch embedding optimization**

   Instead of encoding one source at a time, batch all source descriptions
   and compute all embeddings in a single encoder call:

   ```python
   texts = [desc for _, desc in source_descriptions]
   embeddings = encoder.embed_texts(texts)
   ```

   Then run each embedding against the three vector indexes. This is
   significantly faster than the current one-at-a-time approach.

3. **Match matrix formatting for prompt**

   Format the top-k matches per source as a structured section:

   ```markdown
   ### Semantic Match Matrix

   For each source, the top matching IMAS fields, wiki excerpts, and code
   patterns are listed by cosine similarity:

   **Source: jet:magnetic_field_systems/pf_coil_current**
   - IMAS: pf_active/coil/current (0.922) — "Time-dependent electrical current..."
   - Wiki: "Magnetostatic Ontology" (0.865) — "PF coil current mapping to IMAS..."
   - Code: pfcoil.py:chunk_2 (0.877) — "def read_pf_current(...)"

   **Source: jet:magnetic_field_diagnostics/pf_v5-ipla**
   - IMAS: pf_active/supply/current (0.912) — "Output current of a PF supply..."
   ```

4. **Replace existing per-source semantic search**

   The current `source_candidates` computation in `gather_context()` loops
   over sources one at a time and only searches IMAS nodes. Replace this with
   `compute_semantic_matches()` which is batched and searches all indexes.

**Validation**:
- Compare cosine scores from the match matrix against actual binding targets
  in existing validated mappings
- Measure latency increase (batch embedding should be faster than one-at-a-time)
- Verify wiki/code bridge matches add value vs noise in LLM output

### Phase 4: Dynamic MCP Tool Docstrings

**Goal**: Update MCP tool docstrings to include available physics domains and
score dimensions at registration time, so calling agents know what values are
valid without consulting external documentation.

**Targets**:
- `imas_codex/llm/server.py` — tool registrations for `search_signals`,
  `search_docs`, `search_code`
- `imas_codex/tools/graph_search.py` — `@mcp_tool` description strings

**Changes**:

1. **Dynamic docstring generation from schema enums**

   At server startup / tool registration time, import the `PhysicsDomain` enum
   and score dimension constants, and inject them into the tool descriptions:

   ```python
   from imas_codex.core.physics_domain import PhysicsDomain
   from imas_codex.discovery.base.scoring import (
       CODE_SCORE_DIMENSIONS, CONTENT_SCORE_DIMENSIONS
   )

   _PHYSICS_DOMAINS_DOC = ", ".join(sorted(d.value for d in PhysicsDomain))
   _CONTENT_SCORES_DOC = ", ".join(CONTENT_SCORE_DIMENSIONS)
   _CODE_SCORES_DOC = ", ".join(CODE_SCORE_DIMENSIONS)
   ```

2. **Update `search_signals` docstring** (codex server — `server.py`)

   ```python
   @self.mcp.tool()
   def search_signals(
       query: str,
       facility: str,
       physics_domain: str | None = None,
       ...
   ) -> str:
       f"""Search facility signals with full graph enrichment.
       ...
       Args:
           physics_domain: Filter by physics domain. Valid values:
               {_PHYSICS_DOMAINS_DOC}
       ...
       """
   ```

3. **Update `search_docs` docstring** with new `physics_domain` and
   `score_dimension` parameters

   ```python
   @self.mcp.tool()
   def search_docs(
       query: str,
       facility: str,
       physics_domain: str | None = None,     # NEW
       min_score: float | None = None,         # NEW
       score_dimension: str | None = None,     # NEW
       ...
   ) -> str:
       f"""Search documentation with physics domain and score filtering.
       ...
       Args:
           physics_domain: Filter by physics domain. Valid values:
               {_PHYSICS_DOMAINS_DOC}
           min_score: Minimum score threshold (0.0-1.0)
           score_dimension: Score dimension to filter on. Valid values:
               {_CONTENT_SCORES_DOC}
               Defaults to score_composite.
       ...
       """
   ```

4. **Update `search_code` docstring** similarly

   Valid score dimensions for code are different from content:
   `{_CODE_SCORES_DOC}`.

5. **Update graph-native MCP `@mcp_tool` descriptions**

   In `imas_codex/tools/graph_search.py`, the `@mcp_tool` decorator takes a
   static description string. These should also enumerate valid physics domains.
   Options:
   - **Dynamic string at import time**: Construct the description string using
     f-strings at module load time (imports `PhysicsDomain` from generated
     models)
   - **Template pattern**: Use a helper that formats the docstring from enum
     values:
     ```python
     from imas_codex.tools.utils import physics_domain_doc, score_dimensions_doc

     @mcp_tool(
         "Find IMAS IDS entries using semantic and lexical search. "
         f"physics_domain: Filter results by physics domain. {physics_domain_doc()} "
         f"score_dimension: Filter by score. {score_dimensions_doc('content')} "
     )
     ```

**Validation**:
- Call `get_graph_schema()` or inspect tool descriptions to verify domains and
  dimensions are listed
- Verify docstrings update when new domains are added to the schema (rebuild
  models → reimport)

## Score Dimension Usage Recommendations

Based on analysis of live graph data, the following score dimensions are
recommended for specific mapping pipeline contexts:

### For `map_signals()` — Signal-Level Mapping

| Context Need | Primary Dimension | Threshold | Rationale |
|-------------|-------------------|-----------|-----------|
| Data access patterns | `score_data_access` | ≥0.5 | Surfaces MDSplus paths, TDI expressions, API patterns |
| IMAS integration | `score_imas` / `score_imas_relevance` | ≥0.5 | Content that explicitly discusses IMAS/IDS mapping |
| Convention handling | `score_convention` | ≥0.3 | COCOS, sign flips, unit conversions |
| Calibration context | `score_calibration` | ≥0.3 | Conversion factors, sensor specifications |

### For `assign_sections()` — Section Assignment

| Context Need | Primary Dimension | Threshold | Rationale |
|-------------|-------------------|-----------|-----------|
| Physics explanations | `score_physics_content` | ≥0.5 | Understanding what each section represents |
| Data documentation | `score_data_documentation` | ≥0.5 | Signal tables mapping signals to IDS |

### For `discover_assembly()` — Assembly Pattern Discovery

| Context Need | Primary Dimension | Threshold | Rationale |
|-------------|-------------------|-----------|-----------|
| Data access code | `score_data_access` | ≥0.6 | How data is structured in source systems |
| Modeling code | `score_modeling_code` | ≥0.5 | How simulations assemble coil/circuit data |

## PhysicsDomain Enum Values (for reference)

The `PhysicsDomain` enum (22 values) from `physics_domains.yaml`:

**Core Plasma Physics**: `equilibrium`, `transport`, `magnetohydrodynamics`, `turbulence`  
**Heating & Current Drive**: `auxiliary_heating`, `current_drive`  
**Plasma-Material Interactions**: `plasma_wall_interactions`, `divertor_physics`, `edge_plasma_physics`  
**Diagnostics**: `particle_measurement_diagnostics`, `electromagnetic_wave_diagnostics`, `radiation_measurement_diagnostics`, `magnetic_field_diagnostics`, `mechanical_measurement_diagnostics`  
**Control & Operations**: `plasma_control`, `machine_operations`  
**Engineering Systems**: `magnetic_field_systems`, `structural_components`, `plant_systems`  
**Data & Workflow**: `data_management`, `computational_workflow`  
**Fallback**: `general`

## Implementation Order & Dependencies

```
Phase 1 (search tool filters)
    ↓
Phase 2 (pipeline wiki/code context)  ←— depends on Phase 1 tools
    ↓
Phase 3 (semantic match matrix)       ←— depends on Phase 2 context model
    ↓
Phase 4 (dynamic docstrings)          ←— can run in parallel with 2-3
```

Phase 4 is independent and can be implemented alongside any other phase.

## Risk Assessment

- **Token budget**: Adding wiki + code context to prompts increases token
  count. Mitigated by: physics domain filtering (reduces to ~5-10% of total),
  score dimension thresholds (further reduces), and k-limit parameters.
- **Latency**: Semantic bridge requires N × 3 vector index queries (one per
  source × 3 indexes). Mitigated by: batch embedding, parallelized queries.
- **False positives**: Low-quality wiki/code matches could mislead the LLM.
  Mitigated by: score thresholds, cosine similarity cutoffs, and clear prompt
  framing that marks these as "supporting evidence, not authoritative".

## Files Modified Per Phase

### Phase 1
- `imas_codex/llm/search_tools.py` — add physics_domain and score_dimension params
- `imas_codex/llm/server.py` — update tool registration signatures
- `tests/` — new parametrized tests for filtered search

### Phase 2
- `imas_codex/ids/tools.py` — new `fetch_wiki_context()`, `fetch_code_context()`
- `imas_codex/ids/mapping.py` — `gather_context()`, `map_signals()` integration
- `imas_codex/llm/prompts/mapping/signal_mapping.md` — new template sections
- `tests/ids/` — tests for new tool functions

### Phase 3
- `imas_codex/ids/tools.py` — new `compute_semantic_matches()`
- `imas_codex/ids/mapping.py` — replace `source_candidates` with match matrix
- `imas_codex/llm/prompts/mapping/signal_mapping.md` — match matrix section
- `tests/ids/` — tests for semantic matching

### Phase 4
- `imas_codex/tools/utils.py` — `physics_domain_doc()`, `score_dimensions_doc()`
- `imas_codex/llm/server.py` — dynamic docstrings in tool registration
- `imas_codex/tools/graph_search.py` — dynamic `@mcp_tool` descriptions
- `tests/tools/` — docstring content validation
