# IMAS Search & MCP Quality Remediation Plan

> **Date**: 2026-03-09 (updated 2026-03-10)
> **Status**: Active — Steps 1-5 implemented (commit `9aa0e28`), deeper issues identified
> **Depends on**: mcp-tool-quality-refactor.md (Phase 1 bugs fixed), ingestion-schema-alignment.md (schema drift)

## Executive Summary

The graph-backed MCP tools must achieve feature parity with — and ultimately surpass — the original file-backed IMAS MCP server before that infrastructure can be removed. This plan tracks all remaining gaps between the two systems and the work required to close them.

**What's been done**: Steps 1-5 from the original plan are implemented (text scoring raised to 0.85-0.95 range, generic metadata filtering, IMAS path fetch, unparsed document metadata fallback, dead stub removed). These were committed as `9aa0e28`.

**What's been discovered since**: Five deeper issues that affect overall MCP tool quality and need resolution before claiming full parity with the old server.

---

## Part 1: Old vs New Server Architecture

### The Old File-Backed IMAS Server

The original IMAS MCP server lived in `imas_codex/server.py` and was backed by:

- **`imas_codex/search/document_store.py`** — In-memory JSON store + SQLite FTS5 index, loaded from `detailed/{ids_name}.json` files
- **`imas_codex/search/engines/hybrid_engine.py`** — Combined semantic + lexical results with score boosting
- **`imas_codex/search/engines/lexical_engine.py`** — SQLite FTS5 full-text search with IDS extraction and path intelligence
- **`imas_codex/search/engines/semantic_engine.py`** — Sentence transformer embeddings (local model, numpy cosine)
- **`imas_codex/search/search_strategy.py`** — Rich `SearchHit` model (18+ fields per result)

**Key transition commits**:
- `3c5644f` — "feat: graph-native search tools with Neo4j backend" — introduced `GraphSearchTool` etc.
- `fe6856d` — "refactor: clean break to graph-only MCP server" — removed file-backed mode (-747 lines)
- `ac6a6e5` — "remove whoosh index"

The file-backed search infrastructure still exists in `imas_codex/search/` but is no longer used by any server.

### Old Hybrid Scoring Formula

```python
# Semantic results get 1.2x boost
result.score *= 1.2
# Items found by BOTH engines:
combined_results[path_id].score = (existing_score * 0.7) + (new_score * 0.3) + 0.1
# Lexical-only results: passed through unmodified
```

The old lexical engine (SQLite FTS5) had features the current text search lacks:
- **IDS extraction from path queries**: Auto-detected `equilibrium/time_slice/...` patterns and filtered by IDS
- **Path intelligence**: `_enhance_config_with_path_intelligence()` — recognized path-like queries and routed to exact search
- **Exact path search**: `_try_exact_path_search()` — special handling for exact IMAS paths

### Old SearchHit Model (18+ fields)

The old server returned rich per-result metadata:

| Field | Description | Graph Equivalent |
|-------|-------------|------------------|
| `path` | Full IMAS path | `IMASPath.id` |
| `documentation` | Path documentation | `IMASPath.documentation` |
| `units` | Physical units | `Unit` node via `HAS_UNIT` |
| `data_type` | Data type | `IMASPath.data_type` |
| `ids_name` | IDS name | `IMASPath.ids` |
| `physics_domain` | Physics domain | `IMASPath.physics_domain` |
| `coordinates` | Coordinate labels | `IMASCoordinateSpec` via `HAS_COORDINATE` |
| `lifecycle` | Lifecycle designation | `IMASPath.lifecycle_status` |
| `node_type` | Data node type | `IMASPath.node_type` |
| `timebase` | Reference timebase path | Not yet queried |
| `coordinate1` | Primary coordinate | `IMASCoordinateSpec` (not separated) |
| `coordinate2` | Secondary coordinate | `IMASCoordinateSpec` (not separated) |
| `structure_reference` | Shared structure ref | `IMASPath.structure_reference` |
| `has_identifier_schema` | Has identifier schema | `IdentifierSchema` node exists |
| `validation_rules` | Validation rules | Not migrated |
| `identifier_schema` | Full identifier schema | `IdentifierSchema` node |
| `introduced_after_version` | Version introduced | `DDVersion` via `INTRODUCED_IN` |
| `lifecycle_status` | alpha/obsolescent/etc | `IMASPath.lifecycle_status` |
| `lifecycle_version` | Version for lifecycle | `IMASPath.lifecycle_version` |
| `highlights` | FTS5 highlighted snippets | Not available |

**Default `max_results`**: 50 (old) vs 20 (current codex server)

---

## Part 2: Tool Inventory — Feature Parity

### Tool Coverage (Old → Current)

| Old Tool (file-backed) | IMAS DD Server (graph) | Codex Server (graph) | Status |
|---|---|---|---|
| `search_imas_paths` (hybrid) | `search_imas_paths` (vector-only) | `search_imas` (vector+text) | Partial — see Issues |
| `check_imas_paths` | `check_imas_paths` + renamed detection | Via `python()` only | Gap — no direct codex tool |
| `fetch_imas_paths` | `fetch_imas_paths` + clusters/coords | `fetch` (IMAS path) | Parity |
| `list_imas_paths` | `list_imas_paths` + leaf filtering | Via `python()` only | Gap — no direct codex tool |
| `get_imas_overview` | `get_imas_overview` (dynamic) | Via `python()` only | Gap — no direct codex tool |
| `search_imas_clusters` | `search_imas_clusters` (vector) | Embedded in `search_imas` | Parity |
| `get_imas_identifiers` | `get_imas_identifiers` | Via `python()` only | Gap — no direct codex tool |
| `query_imas_graph` | `query_imas_graph` | Via `python()` | Parity |
| `get_dd_graph_schema` | `get_dd_graph_schema` | `get_graph_schema` | Parity |
| `get_dd_versions` | `get_dd_versions` | Via `python()` only | Gap — no direct codex tool |
| — | — | `search_signals` | New capability |
| — | — | `search_docs` | New capability |
| — | — | `search_code` | New capability |
| — | — | `fetch` | New capability |
| — | — | `python` (REPL) | New capability |
| — | — | Admin tools (6) | New capability |

### IMAS DD Server — `GraphSearchTool` Missing Fields

The `GraphSearchTool.search_imas_paths()` returns the same `SearchHit` model but populates only 8 of 18+ fields. Missing:

| Missing Field | Data Available in Graph? | Fix |
|---|---|---|
| `coordinates` | Yes — `IMASCoordinateSpec` via `HAS_COORDINATE` | Extend Cypher |
| `lifecycle` | Yes — `IMASPath.lifecycle_status` | Already a property, just not queried |
| `timebase` | Uncertain — may need `IMASPath.timebase` property | Check schema |
| `coordinate1`/`coordinate2` | Yes — derivable from `IMASCoordinateSpec` | Extend Cypher |
| `structure_reference` | Yes — `IMASPath.structure_reference` or `path_doc` | Extend Cypher |
| `has_identifier_schema` | Yes — check `IdentifierSchema` node existence | Extend Cypher |
| `validation_rules` | Unknown — may not be in graph | Check schema |
| `identifier_schema` | Yes — `IdentifierSchema` nodes | Extend Cypher |
| `introduced_after_version` | Yes — `DDVersion` via `INTRODUCED_IN` | Extend Cypher |
| `lifecycle_version` | Yes — `IMASPath.lifecycle_version` | Already a property |
| `highlights` | No — FTS5 snippets not applicable to graph | Not fixable |

---

## Part 3: Remaining Issues

### Issue 1: Text Score Antipattern (Medium-High)

**What**: `_text_search_imas_paths_by_query()` uses hardcoded scores (0.95 for full query doc match, 0.90 for full query path match, 0.85 for word match). These produce stable results but are fundamentally an antipattern — they don't reflect actual relevance.

**Why it matters**: A path that matches one query word gets the same 0.85 score as a path that matches five query words. The scoring doesn't distinguish between a perfect keyword match and a marginal one.

**Current state**: The hardcoded scores work acceptably because:
- They're competitive with vector scores (0.85-0.95 vs 0.88+ vector)
- The merge formula `max(vector, text) + 0.05` produces reasonable rankings
- Generic metadata filtering prevents the worst results

**What should replace them**: Normalized BM25 or TF-IDF scoring using the existing Neo4j fulltext index. The index `imas_path_text` supports `db.index.fulltext.queryNodes()` which returns relevance scores. The fix:
1. Use the fulltext index query with BM25 scoring
2. Normalize the raw BM25 scores to the 0.0-1.0 range (divide by max score in the result set)
3. Apply a floor (0.7) to prevent very low text scores from being discarded

**Priority**: Medium-high. The current hardcoded scores are a temporary fix that works, not a permanent solution.

### Issue 2: CodeExample Has No Embedding Field (Medium)

**What**: The `CodeExample` class in `imas_codex/schemas/facility.yaml` has no `embedding` field, no `embedded_at` field, and no vector index. There are 10,065 CodeExample nodes in the graph, all without embeddings.

**Impact**: CodeExamples cannot be found via semantic search. Only their child CodeChunks are searchable. This means you can't search for "equilibrium reconstruction code" and find a CodeExample directly — you have to find CodeChunks and traverse up to CodeExample.

**Investigation**: Git history shows no commits have added an embedding field to CodeExample. This was NOT fixed by another agent contrary to expectations.

**Fix**:
1. Add `embedding` (multivalued float) and `embedded_at` (datetime) slots to `CodeExample` in `facility.yaml`
2. Run `uv run build-models --force` to regenerate models
3. Create vector index `code_example_desc_embedding` (auto-derived from schema)
4. Embed all 10,065 CodeExamples using their `description` field
5. Add CodeExample vector search to `search_code()` in the codex server

**Priority**: Medium. CodeChunk search works as a workaround, but direct CodeExample search would improve code discovery.

### Issue 3: Orphaned CodeChunks — Missing Relationships (Medium)

**What**: 4,419 CodeChunks (3,626 TCV, 793 JET) have a `code_example_id` property set but are missing:
- `HAS_CHUNK` incoming relationship from CodeExample (created by raw Cypher in pipeline, not schema-defined)
- `CODE_EXAMPLE_ID` outgoing relationship to CodeExample (schema-defined via `code_example_id` slot with `range: CodeExample`)

**Root cause**: Historical pipeline used `SET c += chunk` raw Cypher instead of `create_nodes()`, so auto-relationship creation for `CODE_EXAMPLE_ID` didn't happen. The separate `HAS_CHUNK` Cypher step either failed silently or was interrupted.

**Impact**: These chunks are invisible to `search_code()` enrichment which traverses `(CodeExample)-[:HAS_CHUNK]->(CodeChunk)`. They also break `link_chunks_to_data_nodes()` which relies on `(c)<-[:HAS_CHUNK]-(e:CodeExample)`.

**Fix** (direct graph repair, no re-ingestion needed):
```cypher
-- Step 1: Check for truly orphaned chunks (CodeExample doesn't exist)
MATCH (cc:CodeChunk)
WHERE cc.code_example_id IS NOT NULL
  AND NOT EXISTS { MATCH (ce:CodeExample {id: cc.code_example_id}) }
RETURN count(cc) AS truly_orphaned

-- Step 2: Create missing HAS_CHUNK relationships (where CodeExample exists)
MATCH (cc:CodeChunk)
WHERE cc.code_example_id IS NOT NULL
  AND NOT (cc)<-[:HAS_CHUNK]-(:CodeExample)
MATCH (ce:CodeExample {id: cc.code_example_id})
MERGE (ce)-[:HAS_CHUNK]->(cc)

-- Step 3: Create missing CODE_EXAMPLE_ID relationships
MATCH (cc:CodeChunk)
WHERE cc.code_example_id IS NOT NULL
  AND NOT (cc)-[:CODE_EXAMPLE_ID]->(:CodeExample)
MATCH (ce:CodeExample {id: cc.code_example_id})
MERGE (cc)-[:CODE_EXAMPLE_ID]->(ce)
```

If Step 1 reveals truly orphaned chunks (CodeExample doesn't exist), those either need re-ingestion of the parent CodeFile or deletion.

**Schema fix**: Add `has_chunks` slot to `CodeExample` pointing to `CodeChunk` so `create_nodes()` handles this automatically in future. Currently `HAS_CHUNK` is only created by raw Cypher in `pipeline.py:549`.

**Priority**: Medium. Blocks search enrichment for ~17% of code chunks.

### Issue 4: .ppt Documents Deferred by Size Limit (Low-Medium)

**What**: PowerPoint files (.ppt, .pptx) ARE supported by the document processing pipeline — `python-pptx` for modern OOXML format, binary ASCII-run extraction for legacy OLE2 format. The pipeline maps both to `document_type: "presentation"` which is in the `INGESTABLE_DOCUMENT_TYPES` set.

**The real issue**: `jet:Eq_recon.ppt` (7.6 MB) is `deferred` because it exceeds the 5.0 MB `max_size_mb` limit in `docs_worker`. This size check runs *before* the type check, so any document >5 MB gets deferred regardless of type.

**Graph-wide document status**:
| Status | Count | Notes |
|---|---|---|
| `scored` | 735 | Queued for ingestion, not yet processed |
| `failed` | 104 | OLE2 extraction failures |
| `deferred` | 12 | Oversized (>5 MB) |
| `ingested` | 4 | Successfully parsed and chunked |

**Fix options**:
1. Increase `max_size_mb` (via `--max-size-mb` CLI option) to 10 MB for large presentations
2. Add a separate deferred document pipeline that handles oversized files with streaming parsing
3. Accept the deferral — 12 files is a tiny fraction

**Priority**: Low-medium. Only 12 files affected. The 735 `scored` documents awaiting processing are the bigger opportunity — running `imas-codex discover wiki <facility>` would process them.

### Issue 5: IMAS DD Server Search Quality Gap (Medium)

**What**: The IMAS DD server (`imas_codex/tools/graph_search.py`) uses pure vector-only search with no text/lexical fallback. It suffers from the same "Verbose description" pollution that was fixed in the codex server.

**Comparison**:
| Capability | Old File-Backed | IMAS DD Server | Codex Server |
|---|---|---|---|
| Hybrid search (semantic+text) | Yes (FTS5+numpy) | No (vector-only) | Yes (Neo4j vector+fulltext) |
| Generic metadata filtering | N/A (not in JSON data) | No | Yes |
| IDS extraction from path queries | Yes | No | No |
| Exact path search routing | Yes | No | No |
| Score formula | 0.7/0.3/0.1 dual-match | Pure cosine | max(v,t)+0.05 |
| Default max results | 50 | 50 | 20 |

**Fix**: Port `_text_search_imas_paths_by_query()` and `_is_generic_metadata_path()` from the codex server's `search_tools.py` into the IMAS DD server's `GraphSearchTool`. Alternatively, deprecate the IMAS DD server in favor of the codex server which is a superset.

**Priority**: Medium. Depends on whether the IMAS DD server will continue to be used independently.

---

## Part 4: Cross-Cutting Issues (from ingestion-schema-alignment.md)

These issues from the schema alignment plan directly affect MCP tool quality:

### Missing Graph Relationships

| Relationship | Expected | Actual | Impact on MCP Tools |
|---|---|---|---|
| `(CodeChunk)-[:CODE_EXAMPLE_ID]->(CodeExample)` | ~21,107 | 0 | Code search enrichment partially broken |
| `(CodeExample)-[:FROM_FILE]->(CodeFile)` | ~1,500+ | 0 | Cannot traverse CodeExample → CodeFile |
| `(CodeFile)-[:PRODUCED]->(CodeExample)` | ~1,500+ | 0 | Fetch code file broken |
| `(DataReference)-[:RESOLVES_TO_IMAS_PATH]->(IMASPath)` | TBD | 0 | IMAS facility cross-references empty |
| `(WikiChunk)-[:MENTIONS_IMAS]->(IMASPath)` | TBD | 0 | Wiki-to-IMAS cross-references broken |

These gaps are tracked in detail in `ingestion-schema-alignment.md` and should be fixed there. They are listed here because they affect MCP tool output quality.

---

## Part 5: Implementation Plan

### Phase 1: IMAS DD Server Parity (addresses Issue 5)

**Goal**: Match the codex server's search quality in the IMAS DD server.

1. Port hybrid text+vector scoring to `GraphSearchTool.search_imas_paths()`
2. Port `_is_generic_metadata_path()` filtering
3. Extend the Cypher query to populate missing `SearchHit` fields (coordinates, lifecycle, structure_reference, etc.)

### Phase 2: Replace Hardcoded Text Scores (addresses Issue 1)

**Goal**: Use normalized BM25 scoring from the Neo4j fulltext index instead of hardcoded values.

1. Replace `_text_search_imas_paths_by_query()` implementation:
   - Use `db.index.fulltext.queryNodes('imas_path_text', $query)` for BM25 scoring
   - Normalize scores to 0.0-1.0 range
   - Keep `_is_generic_metadata_path()` filtering
   - Keep the `max(vector, text) + 0.05` merge formula
2. Validate with the same test queries (plasma current, electron temperature, etc.)

### Phase 3: CodeExample Embeddings (addresses Issue 2)

**Goal**: Make CodeExample nodes searchable via semantic search.

1. Add `embedding` + `embedded_at` to `CodeExample` schema
2. Run `uv run build-models --force`
3. Embed all 10,065 CodeExample descriptions
4. Optionally add CodeExample vector search to `search_code()`

### Phase 4: Graph Relationship Repair (addresses Issue 3)

**Goal**: Fix orphaned CodeChunks and missing relationships.

1. Run diagnostic query to check for truly orphaned chunks
2. Execute repair queries (HAS_CHUNK + CODE_EXAMPLE_ID)
3. Add `has_chunks` slot to CodeExample schema for future-proofing
4. Re-run `link_chunks_to_data_nodes()` for repaired chunks

### Phase 5: Document Processing (addresses Issue 4)

**Goal**: Process remaining scored documents and handle oversized files.

1. Run `imas-codex discover wiki jet` to process the 735 scored documents
2. Optionally increase `max_size_mb` for the 12 deferred documents

---

## Part 6: Validation Plan

### Search Quality Tests

```python
# IMAS search: canonical paths must appear in top 5
result = search_imas("plasma current", k=10)
assert any("ip" in r for r in result[:5])
assert sum(1 for r in result if "Verbose description" in r) <= 1

# IMAS search: electron temperature
result = search_imas("electron temperature profile", k=10)
assert any("core_profiles" in r for r in result)
assert any("electrons/temperature" in r for r in result)

# Code search: enrichment works for previously-orphaned chunks
result = search_code("equilibrium reconstruction", facility="tcv")
assert len(result) > 0
assert all(r.get("source_file") for r in result)  # has source linkage

# Doc search: documents are findable
result = search_docs("equilibrium", facility="jet")
assert len(result) > 0
```

### Feature Parity Regression Tests

```python
# Fetch IMAS path returns rich metadata
result = fetch("equilibrium/time_slice/global_quantities/ip")
assert "documentation" in result  # has description
assert "unit" in result.lower()    # has units
assert "coordinate" in result.lower() or "lifecycle" in result.lower()  # has metadata

# Fetch document returns metadata even when unparsed
result = fetch("jet:Eq_recon.ppt")
assert "No resource found" not in result

# Text search for exact paths works
result = search_imas("magnetics/ip", k=5)
assert result[0]["id"] == "magnetics/ip"  # exact match ranks first
```

### Performance Targets

| Metric | Old Server | Target | Current |
|---|---|---|---|
| Search latency (p50) | ~200ms (local numpy) | <500ms (graph) | ~300ms |
| Results relevance (top-5) | Good (hybrid) | Equal or better | Improving |
| Field completeness | 18/18 fields | 16/18 fields | 8/18 fields |
| Max results default | 50 | 50 | 20 (should increase) |

---

## Part 7: Priority & Sequencing

| Phase | Priority | Effort | Impact |
|---|---|---|---|
| Phase 2: BM25 scoring | High | Medium | Eliminates antipattern, improves relevance |
| Phase 4: Relationship repair | High | Low | Fixes 17% of code chunks |
| Phase 1: IMAS DD server parity | Medium | Medium | Consistent quality across servers |
| Phase 3: CodeExample embeddings | Medium | Low | Enables direct code search |
| Phase 5: Document processing | Low | Low | 735+ documents processable |

---

## Part 8: Can the Graph Server Outperform the Old?

**Yes**, once the issues above are resolved. The graph-backed server has structural advantages:

1. **Cross-domain linking**: Old server searched IMAS paths in isolation. Graph server can show which facility signals map to an IMAS path, which wiki pages discuss it, and which code files reference it.
2. **Dynamic data**: Old server loaded static JSON files. Graph server reflects the latest discovery results.
3. **Multi-facility**: Old server was IMAS DD only. Graph server covers TCV, JET, and ITER data simultaneously.
4. **Cluster context**: Semantic clusters provide physics-domain grouping that the old server approximated with JSON files.
5. **Version tracking**: Graph stores IMAS path changes across DD versions, enabling "what changed" queries.
6. **Scalability**: Adding new facilities or data sources doesn't require regenerating JSON files.

**Where the old server was better** (gaps to close):
- Hybrid search quality (partially addressed, need BM25)
- Rich per-result metadata (need to extend Cypher queries)
- Path intelligence / IDS extraction (nice-to-have, not critical)
- FTS5 highlighting (not reproducible in graph, acceptable loss)
