# Ingestion Pipeline Schema Alignment

**Created:** 2026-03-08
**Status:** Draft
**Priority:** High — these gaps break MCP search tool traversals

## Problem Statement

The LinkML schemas define the authoritative graph structure, but the ingestion pipelines bypass `create_nodes()` and use raw Cypher that doesn't create all schema-defined relationships. This creates drift between what the schema declares and what the graph contains, breaking search tools that rely on schema-correct traversals.

## Schema-Graph Drift Inventory

### Missing Relationships (schema defines, graph lacks)

| Relationship | Schema Source | Expected Count | Actual Count | Impact |
|---|---|---|---|---|
| `(CodeChunk)-[:CODE_EXAMPLE_ID]->(CodeExample)` | `CodeChunk.code_example_id` (range: CodeExample) | ~21,107 | **0** | Code search facility filter broken, enrichment broken |
| `(CodeExample)-[:FROM_FILE]->(CodeFile)` | `CodeExample.from_file` (range: CodeFile) | ~1,500+ | **0** | Cannot traverse from CodeExample to CodeFile |
| `(CodeFile)-[:PRODUCED]->(CodeExample)` | `CodeFile.code_example_id` (range: CodeExample) | ~1,500+ | **0** | Fetch code file completely broken |
| `(DataReference)-[:RESOLVES_TO_IMAS_PATH]->(IMASPath)` | `DataReference.resolves_to_imas_path` (range: IMASPath) | TBD | **0** | IMAS facility cross-references return empty |
| `(DataReference)-[:CALLS_TDI_FUNCTION]->(TDIFunction)` | `DataReference.calls_tdi_function` (range: TDIFunction) | TBD | **0** | TDI function references not linked |
| `(CodeChunk)-[:REFERENCES_IMAS]->(IMASPath)` | `CodeChunk.references_imas` (range: IMASPath) | TBD | **0** | IDS references in code not resolvable |
| `(WikiChunk)-[:MENTIONS_IMAS]->(IMASPath)` | `WikiChunk.mentions_imas_paths` (range: IMASPath) | TBD | **0** | Wiki-to-IMAS cross-references broken |

### Relationships That Exist Only In Graph (not in schema)

| Relationship | Graph Count | Notes |
|---|---|---|
| `(CodeChunk)-[:AT_FACILITY]->(Facility)` | ~21,107 | NOT in schema but set by pipeline; useful convenience |
| `(CodeChunk).facility_id` property | ~21,107 | NOT in schema but set by pipeline; enables direct filtering |
| `(CodeChunk).source_file` property | ~21,107 | NOT in schema; duplicates `CodeExample.source_file` |
| `(CodeChunk).language` property | ~21,107 | NOT in schema; duplicates `CodeExample.language` |
| `(DataReference)-[:RESOLVES_TO_NODE]->(TreeNode)` | 5,317 | Schema says `DataNode` target, graph has `TreeNode` target |
| `(DataReference)-[:RESOLVES_TO_TREE_NODE]->(TreeNode)` | 566 | NOT in schema at all |

### Aspirational Relationships (mentioned but not formalized)

| Relationship | Where Mentioned | Schema Slot Defined? | Notes |
|---|---|---|---|
| `MAPS_TO_IMAS` | FacilitySignal description comment | **No** | Aspirational; `IMASMapping -[:TARGET_PATH]-> IMASPath` is the proper vehicle |

### Entity Label Mismatch

| Issue | Schema Says | Graph Has | Count |
|---|---|---|---|
| `DataReference -[:RESOLVES_TO_NODE]->` target | `DataNode` | `TreeNode` | 5,317 |

`TreeNode` was likely a pre-schema name for `DataNode`. This needs investigation — check if `DataNode` nodes exist separately with different data vs `TreeNode`:

```cypher
MATCH (n:DataNode) RETURN count(n)   -- Check if DataNode label exists
MATCH (n:TreeNode) RETURN count(n)   -- Check if TreeNode label exists
MATCH (n) WHERE 'TreeNode' IN labels(n) AND 'DataNode' IN labels(n) RETURN count(n)  -- Dual labeled?
```

## Root Cause Analysis

### Code Ingestion Pipeline (`imas_codex/ingestion/pipeline.py`)

The pipeline uses raw Cypher instead of `create_nodes()` for all graph writes:

**Line 506–515: CodeChunk creation**
```python
graph_client.query("""
    UNWIND $chunks AS chunk
    MERGE (c:CodeChunk {id: chunk.id})
    SET c += chunk
    WITH c, chunk
    MATCH (f:Facility {id: chunk.facility_id})
    MERGE (c)-[:AT_FACILITY]->(f)
""", chunks=all_chunks)
```
- ✅ Creates `AT_FACILITY` relationship (not in schema but useful)
- ✅ Sets `code_example_id` as property on CodeChunk
- ❌ Does NOT create `(CodeChunk)-[:CODE_EXAMPLE_ID]->(CodeExample)` relationship
- ❌ Does NOT create `(CodeChunk)-[:REFERENCES_IMAS]->(IMASPath)` relationship

**Line 543–549: CodeExample→CodeChunk linking**
```python
graph_client.query("""
    MATCH (c:CodeChunk)
    WHERE c.code_example_id IN $example_ids
    MATCH (e:CodeExample {id: c.code_example_id})
    MERGE (e)-[:HAS_CHUNK]->(c)
""", example_ids=chunk_example_ids)
```
- ✅ Creates `(CodeExample)-[:HAS_CHUNK]->(CodeChunk)` — but this is the INVERSE direction from what the schema defines. The schema has `CodeChunk.code_example_id` → relationship points FROM CodeChunk TO CodeExample. The pipeline creates FROM CodeExample TO CodeChunk. Both directions should exist for the dual model.

**Line 527–533: CodeExample creation**
```python
graph_client.query("""
    MERGE (e:CodeExample {id: $id})
    SET e += $props
""", id=example_id, props=meta)
```
- ❌ Does NOT create `(CodeExample)-[:FROM_FILE]->(CodeFile)` relationship
- ❌ Does NOT create `(CodeExample)-[:AT_FACILITY]->(Facility)` from the MERGE

**Line (in `_update_facility_path_status`):**
```python
MERGE (p)-[:PRODUCED]->(e)
```
- ✅ Creates `(FacilityPath)-[:PRODUCED]->(CodeExample)` — but the schema also says `(CodeFile)-[:PRODUCED]->(CodeExample)`, which is NOT created

### Ingestion Graph Module (`imas_codex/ingestion/graph.py`)

**`link_chunks_to_imas_paths()`** — Creates `REFERENCES_IMAS` from CodeChunk to IMASPath based on `related_ids` property. This function IS called at the end of ingestion, but:
- Only matches `IMASPath` where `p.id = ids_name AND p.ids = ids_name` (IDS-level only)
- Result: 0 relationships in graph — means either no CodeChunks had `related_ids` set, or the IMASPath matching condition is too strict

**`link_chunks_to_tree_nodes()`** — Creates DataReference nodes and RESOLVES_TO_NODE relationships. This works but:
- Creates `RESOLVES_TO_NODE` to `DataNode` — but graph shows them pointing to `TreeNode`
- Does NOT create `RESOLVES_TO_IMAS_PATH` relationships at all
- Does NOT create `CALLS_TDI_FUNCTION` relationships

**`link_example_mdsplus_paths()`** — Same as above but per-example.

**`link_examples_to_facility()`** — Creates `AT_FACILITY` from CodeExample. This works. ✅

### File Discovery Pipeline (`imas_codex/discovery/code/scanner.py`)

**Line 320:** Creates CodeFile nodes with `MERGE (sf:CodeFile {id: item.id})` and `AT_FACILITY` and `IN_DIRECTORY` relationships.
- ❌ Does NOT create `PRODUCED -> CodeExample` (because CodeExamples don't exist yet at scan time — they're created during ingestion)

### Queue Module (`imas_codex/ingestion/queue.py`)

**Line 119:** Creates CodeFile nodes with `MERGE (sf:CodeFile {id: item.id})` and `AT_FACILITY` and `IN_DIRECTORY`.
- Same gap — no PRODUCED relationship (expected — CodeExamples don't exist at queue time)

### Wiki Pipeline (`imas_codex/discovery/wiki/graph_ops.py`)

Not yet audited in detail, but graph shows 0 MENTIONS_IMAS relationships.
Schema defines `WikiChunk.mentions_imas_paths` with `relationship_type: MENTIONS_IMAS`.
The wiki workers likely don't create this relationship. Needs separate investigation.

## Fix Plan

### Phase 1: Schema Formalization (prerequisites)

1. **Add `facility_id`, `source_file`, `language` as formal properties to CodeChunk schema**
   These are set by the pipeline and are useful for direct filtering. Make them official rather than fighting their existence.

   ```yaml
   # In CodeChunk:
   facility_id:
     description: Parent facility ID (denormalized for query performance)
     required: true
     range: Facility
     annotations:
       relationship_type: AT_FACILITY
   source_file:
     description: Source file path (denormalized from CodeExample.source_file)
   language:
     description: Programming language (denormalized from CodeExample.language)
   ```

2. **Add `MAPS_TO_IMAS` as a formal slot on FacilitySignal or verify IMASMapping is the intended vehicle**
   Currently mentioned only in a description comment. If `IMASMapping -[:TARGET_PATH]-> IMASPath` is the correct pattern (it appears to be), then remove the `MAPS_TO_IMAS` mention from the FacilitySignal description and update the search tools to use `IMASMapping` traversals.

3. **Verify TreeNode vs DataNode label usage**
   If TreeNode is a legacy label, add DataNode as a secondary label to existing TreeNode nodes, or rename all TreeNode nodes to DataNode.

### Phase 2: Ingestion Pipeline Refactor

#### 2a. Use `create_nodes()` Instead of Raw Cypher

The `create_nodes()` method on GraphClient automatically creates all schema-defined relationships. Refactor `ingest_files()` in `pipeline.py` to use it:

```python
# Instead of raw MERGE + SET:
graph_client.create_nodes("CodeChunk", all_chunks)
graph_client.create_nodes("CodeExample", [meta])
```

This will automatically create:
- `(CodeChunk)-[:CODE_EXAMPLE_ID]->(CodeExample)` ← from `code_example_id` slot
- `(CodeChunk)-[:AT_FACILITY]->(Facility)` ← from `facility_id` slot (after Phase 1)
- `(CodeExample)-[:AT_FACILITY]->(Facility)` ← from `facility_id` slot
- `(CodeExample)-[:FROM_FILE]->(CodeFile)` ← from `from_file` slot (if CodeFile exists)

**Risk:** `create_nodes()` requires target nodes to exist. CodeExample must be created BEFORE CodeChunk (for CODE_EXAMPLE_ID), and CodeFile must exist (for FROM_FILE). Currently pipeline creates CodeChunk first, then CodeExample. Reverse the order.

**Proposed write order:**
1. Create/update `CodeExample` nodes (need facility_id, source_file)
2. Create `CodeChunk` nodes via `create_nodes()` (automatically links to CodeExample via CODE_EXAMPLE_ID)
3. Create `HAS_CHUNK` from CodeExample to CodeChunk (existing pattern is fine — this is the inverse traversal convenience)
4. Link DataReferences

#### 2b. Add Missing Relationship Creation to `link_chunks_to_tree_nodes()`

Add these steps:
- **RESOLVES_TO_IMAS_PATH**: After resolving to DataNode/TreeNode, traverse any existing IMAS mappings to create RESOLVES_TO_IMAS_PATH links
- **CALLS_TDI_FUNCTION**: For refs with `ref_type = 'tdi_call'`, match to TDIFunction nodes

```python
# After RESOLVES_TO_NODE:
# Resolve IMAS paths via DataNode -> IMASMapping -> IMASPath chain
client.query("""
    MATCH (dr:DataReference)-[:RESOLVES_TO_NODE]->(dn:DataNode)
    WHERE NOT (dr)-[:RESOLVES_TO_IMAS_PATH]->()
    MATCH (m:IMASMapping)-[:SOURCE_PATH]->(dn)
    MATCH (m)-[:TARGET_PATH]->(ip:IMASPath)
    MERGE (dr)-[:RESOLVES_TO_IMAS_PATH]->(ip)
""")

# Resolve TDI function references
client.query("""
    MATCH (dr:DataReference {ref_type: 'tdi_call'})
    WHERE NOT (dr)-[:CALLS_TDI_FUNCTION]->()
    MATCH (tdi:TDIFunction)
    WHERE tdi.id CONTAINS dr.raw_string OR dr.raw_string CONTAINS tdi.name
    MERGE (dr)-[:CALLS_TDI_FUNCTION]->(tdi)
""")
```

**Note:** RESOLVES_TO_IMAS_PATH depends on IMASMapping nodes existing, which requires the signals-to-IMAS mapping pipeline (see `plans/features/imas-mappings.md`). This is a future dependency.

#### 2c. Fix CodeFile→CodeExample Linkage

After CodeExamples are created during ingestion, create the `PRODUCED` relationship from CodeFile:

```python
# In pipeline.py, after creating CodeExample:
graph_client.query("""
    MATCH (ce:CodeExample {id: $example_id})
    MATCH (cf:CodeFile {path: ce.source_file, facility_id: ce.facility_id})
    MERGE (cf)-[:PRODUCED]->(ce)
    MERGE (ce)-[:FROM_FILE]->(cf)
""", example_id=example_id)
```

#### 2d. Fix `link_chunks_to_imas_paths()` matching

Current matching is too strict: `p.id = ids_name AND p.ids = ids_name`. This only matches IDS roots.
Relax to match any IMASPath where the IDS name matches:

```python
# Current (too strict):
MATCH (p:IMASPath) WHERE p.id = ids_name AND p.ids = ids_name

# Proposed (broader):
MATCH (p:IMASPath) WHERE p.ids = ids_name AND p.node_type = 'IDS_root'
```

### Phase 3: Graph Migration (Fix Existing Data)

Run one-time migration queries to create missing relationships for already-ingested data:

```cypher
-- 1. Create CODE_EXAMPLE_ID relationships from existing property
MATCH (cc:CodeChunk)
WHERE cc.code_example_id IS NOT NULL
MATCH (ce:CodeExample {id: cc.code_example_id})
MERGE (cc)-[:CODE_EXAMPLE_ID]->(ce)

-- 2. Create FROM_FILE relationships
MATCH (ce:CodeExample)
WHERE ce.source_file IS NOT NULL AND ce.facility_id IS NOT NULL
MATCH (cf:CodeFile {path: ce.source_file})
WHERE cf.facility_id = ce.facility_id
MERGE (ce)-[:FROM_FILE]->(cf)

-- 3. Create PRODUCED from CodeFile to CodeExample
MATCH (ce:CodeExample)-[:FROM_FILE]->(cf:CodeFile)
MERGE (cf)-[:PRODUCED]->(ce)

-- 4. Fix RESOLVES_TO_NODE target label (TreeNode → DataNode)
-- First check if nodes are dual-labeled or need renaming
MATCH (n:TreeNode) WHERE NOT n:DataNode
SET n:DataNode
-- OR if TreeNode and DataNode are different concepts, skip this
```

**Migration should be idempotent** — use MERGE, not CREATE.

### Phase 4: Wiki MENTIONS_IMAS Pipeline

The WikiChunk schema defines `mentions_imas_paths` with relationship type `MENTIONS_IMAS`.
The wiki ingestion pipeline does not currently:
1. Extract IMAS path references from wiki chunk text
2. Create MENTIONS_IMAS relationships

This requires:
- Adding IMAS path extraction to the wiki workers (reuse `extract_ids_references()` from ingestion)
- Running a post-processing pass similar to `link_chunks_to_imas_paths()` for WikiChunks

### Phase 5: Search Tool Adaptation

While Phases 2-4 fix the graph, the search tools need to be resilient to BOTH the current graph state AND the future correct state:

1. **`_vector_search_code_chunks`**: Use `cc.facility_id` property directly (works now, will work after fix)
2. **`_text_search_code_chunks`**: Same — use `cc.facility_id` directly
3. **`_enrich_code_chunks`**: Use `(ce:CodeExample)-[:HAS_CHUNK]->(cc)` for CE→CC traversal (current graph), NOT `(cc)-[:CODE_EXAMPLE_ID]->(ce)` (schema but missing). After migration, both exist.
4. **`_fetch_code_file`**: Use `(ce:CodeExample {source_file: $resource})-[:HAS_CHUNK]->(cc)` rather than `(cf:CodeFile)-[:PRODUCED]->(ce)`. After migration, both work.
5. **`_get_facility_crossrefs`**: Remove `RESOLVES_TO_IMAS_PATH` (0 in graph). Remove `MAPS_TO_IMAS` (0 in graph, not in schema). Keep DataAccess traversals that work. After Phase 2b+3, add back.

## Execution Order

1. **Immediate (this PR)**: Fix search tools to work with CURRENT graph state (Phase 5)
2. **Schema PR**: Formalize CodeChunk.facility_id etc. (Phase 1)
3. **Migration PR**: Run migration queries to create missing relationships (Phase 3)
4. **Pipeline PR**: Refactor ingestion to use `create_nodes()` (Phase 2)
5. **Wiki PR**: Add MENTIONS_IMAS pipeline (Phase 4)
6. **Verification**: Re-run search tools, verify all traversals work

## Validation Queries

After migration, verify all relationships exist:

```cypher
// All CODE_EXAMPLE_ID relationships created
MATCH (cc:CodeChunk) WHERE cc.code_example_id IS NOT NULL
OPTIONAL MATCH (cc)-[:CODE_EXAMPLE_ID]->(ce:CodeExample)
WITH cc, ce
WHERE ce IS NULL
RETURN count(cc) AS orphaned_chunks  // Should be 0

// All FROM_FILE relationships created
MATCH (ce:CodeExample) WHERE ce.source_file IS NOT NULL
OPTIONAL MATCH (ce)-[:FROM_FILE]->(cf:CodeFile)
WITH ce, cf
WHERE cf IS NULL
RETURN count(ce) AS unlinked_examples  // Should be 0 (or count of files not in CodeFile)

// All CodeFile PRODUCED relationships created
MATCH (cf:CodeFile)-[:PRODUCED]->(ce:CodeExample)
RETURN count(*) AS code_file_produced  // Should be > 0
```

## Impact Assessment

| Fix | Search Tools Affected | User-Visible Impact |
|---|---|---|
| CODE_EXAMPLE_ID | `search_code` (facility filter), `_enrich_code_chunks` | Code search with facility filter returns 0 results → returns correct results |
| FROM_FILE | `_fetch_code_file` | Fetching code by file path completely broken → works |
| PRODUCED (CodeFile) | `_fetch_code_file` | Same as above |
| RESOLVES_TO_IMAS_PATH | `_get_facility_crossrefs` | IMAS cross-refs always empty → shows code refs |
| MENTIONS_IMAS | `_get_facility_crossrefs` | Wiki mentions always empty → shows wiki refs |
| MAPS_TO_IMAS | `_get_facility_crossrefs`, `_enrich_signals` | Signal-to-IMAS mapping always empty → shows IMAS paths |
