# MCP–Cypher Gap Rectification Plan

**Created:** 2026-03-14
**Status:** Draft
**Prerequisite:** [ingestion-schema-alignment.md](ingestion-schema-alignment.md)
**Scope:** Schema design, CLI discovery tools, graph data migration, MCP search tools

## Executive Summary

The MCP search tools (`search_signals`, `search_docs`, `search_code`, `search_imas`) return substantially less useful results than equivalent raw Cypher queries. Eight gaps were identified in the Part 2 analysis. This plan addresses all eight across four stages: multi-facility schema design, CLI discovery tooling, graph data migration, and MCP tool fixes.

### Gap Inventory (from Part 2 Analysis)

| # | Gap | Severity | Root Cause |
|---|-----|----------|------------|
| 1 | Wiki chunk deduplication | Critical | 5,576 duplicate chunks (12% waste), no dedup in formatters |
| 2 | Relevance drift | High | Vector search returns semantically similar but topically wrong results |
| 3 | Empty signal search for JET | Critical | JET FacilitySignals are device config probes (1,560), not PPF/JPF measurement signals |
| 4 | DataAccess nodes unpopulated | High | 7 nodes exist, all static geometry — no general PPF/JPF/MDSplus access patterns |
| 5 | Code search structural disconnect | Medium | 5,469 orphaned CodeChunks, 1,414 orphaned CodeExamples |
| 6 | FacilityPath underutilization | Medium | 181K paths, only 18 enriched |
| 7 | Fixed text search scores | Low | Hardcoded 0.5/0.6, no fulltext indexes |
| 8 | Over-fetch strategy | Low | k×20 Python post-filter instead of pre-filter |

### Recent Schema Context

Key commits inform what's already been fixed:
- `6833f91` — Legacy label migration (TreeNode→DataNode, MDSplusTree→DataSource). 18 migration steps.
- `985bff9` — Relationship renames (PRODUCED→HAS_EXAMPLE, SAME_SENSOR→MATCHES_SENSOR).
- `c004d2f` — Search tool Cypher aligned with actual graph relationships.
- `ba0bcb7` — Hybrid vector+keyword search (0.7/0.3 weighted merge + cross-method boost).
- `70b7076` — Initial ingestion-schema alignment (CODE_EXAMPLE_ID relationships created: 7,158).

---

## Stage 1: Multi-Facility Schema Design

### 1.1 DataAccess Template Expansion (Gaps #3, #4)

**Problem:** JET has 7 DataAccess nodes, all for static geometry (device_xml, magnetics_config, JEC2020). No general-purpose DataAccess nodes for PPF, JPF, MDSplus, or UDA — the primary data access methods at JET.

**Current TCV comparison:** TCV has 10+ DataAccess nodes covering MDSplus tree TDI, TDI functions, static trees, VSYSTEM, MATLAB, Fortran, IDL, tcvpy, and python_mdsplus access patterns. Each has `connection_template` and `data_template` with `{accessor}`, `{shot}`, `{data_source}` placeholders.

**Schema changes:** None required. The `DataAccess` schema (facility.yaml lines 2146-2318) already supports all needed `method_type` values via the `DataAccessMethodType` enum, which includes `ppf`, `jpf`, `mdsplus`, `uda`, `hdf5`, etc. The issue is purely that the JET DataAccess nodes haven't been created.

**Design for JET DataAccess nodes:**

| DataAccess ID | method_type | connection_template | data_template |
|---|---|---|---|
| `jet:ppf:standard` | `ppf` | `ppfgo(pulse={shot})` | `ppfdata({shot}, '{dda}', '{dtype}')` |
| `jet:ppf:python` | `ppf` | `import ppf; ppf.ppfgo({shot})` | `ppf.ppfdata({shot}, '{dda}', '{dtype}')` |
| `jet:jpf:standard` | `jpf` | `jpfgo({shot})` | `jpf.jpfget({shot}, '{signal}')` |
| `jet:mdsplus:standard` | `mdsplus` | `mds.Connection('{server}')` | `conn.get('\\\\{tree}::{node}')` |
| `jet:uda:standard` | `uda` | `import pyuda; client = pyuda.Client()` | `client.get('{signal}', {shot})` |
| `jet:sal:standard` | `sal` | `import sal; client = sal.Client('{server}')` | `client.get('{signal}', {shot})` |

**Action:** Create these via a JET-specific signal scanner plugin or a one-time migration. The PPF scanner (`scanners/ppf.py`) already creates `jet:ppf:standard` — it just hasn't been run against prod.

### 1.2 FacilitySignal Scope Clarification (Gap #3)

**Problem:** JET's 1,560 FacilitySignals are all device configuration probes (magnetic probe positions from device_xml). These are geometric metadata, not the measurement signals users search for (PPF DDAs/DTypes, JPF signals, MDSplus nodes).

**Design decision:** Device config signals are valid FacilitySignals — they represent sensor positions and orientations. But they should coexist with measurement signals, not replace them. The PPF scanner (`scanners/ppf.py`) already produces the correct FacilitySignal format:
- ID: `jet:magnetics/bpme_100_r` (device config) vs `jet:magnetics/hrts_te` (measurement)
- Accessor: `device_xml:magprobes/100/r` (device config) vs `ppfdata(pulse, 'HRTS', 'TE')` (measurement)
- Status: device config signals are `discovered`, measurement signals would go through the `discovered→enriched→checked` lifecycle

**Schema changes:** None. FacilitySignal already supports both use cases. The `accessor` field and `data_access` relationship distinguish access patterns.

**Signal contamination fix:** 289 JET FacilitySignals have wrong `facility_id` (278 with `tcv:` prefix, 9 `git:`, 2 `roo:`). These need deletion or re-attribution in migration.

### 1.3 WikiChunk Cross-Reference Formalization (Gap #1, #2)

**Problem:** WikiChunk nodes have metadata properties populated during ingestion (`ppf_paths_mentioned`: 578, `imas_paths_mentioned`: 655, `mdsplus_paths_mentioned`: 29, `tool_mentions`: 1,876) but zero DOCUMENTS or MENTIONS_IMAS relationships exist. The function `link_chunks_to_entities()` exists in `imas_codex/discovery/wiki/pipeline.py` line 251 but is never called from any CLI command.

**Schema status:** Already correct. WikiChunk defines:
- `documents_nodes` → DataNode via DOCUMENTS
- `documents_signals` → FacilitySignal via DOCUMENTS  
- `mentions_imas_paths` → IMASPath via MENTIONS_IMAS

**What's needed:** Wire `link_chunks_to_entities()` into the wiki CLI and/or run it as a post-processing step.

### 1.4 Fulltext Index Schema (Gap #7)

**Problem:** Zero fulltext indexes exist. All text search uses `toLower(n.text) CONTAINS toLower($query)`, forcing sequential scans. Hardcoded scores (0.5 for code, 0.6 for signals) destroy ranking.

**Proposed fulltext indexes:**

```cypher
CREATE FULLTEXT INDEX wiki_chunk_text IF NOT EXISTS
FOR (n:WikiChunk) ON EACH [n.text]
OPTIONS { indexConfig: { `fulltext.analyzer`: 'standard-no-stop-words' } }

CREATE FULLTEXT INDEX code_chunk_text IF NOT EXISTS
FOR (n:CodeChunk) ON EACH [n.text, n.function_name]
OPTIONS { indexConfig: { `fulltext.analyzer`: 'standard-no-stop-words' } }

CREATE FULLTEXT INDEX facility_signal_text IF NOT EXISTS
FOR (n:FacilitySignal) ON EACH [n.name, n.description, n.node_path]
OPTIONS { indexConfig: { `fulltext.analyzer`: 'standard-no-stop-words' } }

CREATE FULLTEXT INDEX data_node_text IF NOT EXISTS
FOR (n:DataNode) ON EACH [n.description, n.canonical_path]
OPTIONS { indexConfig: { `fulltext.analyzer`: 'standard-no-stop-words' } }
```

**Schema integration:** Add a `fulltext_indexes` section to the LinkML schema or a `fulltext_index_name` annotation (mirroring `vector_index_name`). This keeps indexes declarative and auto-generated.

### 1.5 Content Deduplication Strategy (Gap #1)

**Problem:** 5,576 duplicate WikiChunks from 3,352 unique duplicate texts. Top offender: 88 copies of a proposal template. 12% of JET wiki chunks are wasted storage and pollute search results.

**Schema addition:** Add `content_hash` property to WikiChunk:

```yaml
content_hash:
  description: SHA-256 hash of text content for deduplication
  range: string
```

**Dedup model:** Keep one canonical chunk per unique text, link duplicates via a `SAME_CONTENT` relationship or simply delete duplicates and update parent page relationships. Prefer deletion — duplicate chunks provide zero additional signal and inflate vector indexes.

---

## Stage 2: CLI Discovery Tools

### 2.1 Run PPF Signal Scanner (Gap #3)

The PPF scanner already exists at `imas_codex/discovery/signals/scanners/ppf.py`. It:
- SSH connects to JET compute nodes
- Calls `ppfdda()` and `ppfdti()` to enumerate DDAs and DTypes
- Creates `jet:ppf:standard` DataAccess node with templates
- Creates FacilitySignals with ID format `jet:{domain}/{dda_lower}_{dtype_lower}`
- Has domain hints for 20 DDAs (EFIT→equilibrium, HRTS→electron_temperature, etc.)

**Command:** `uv run imas-codex signals scan jet --scanner ppf`

**Prerequisites:**
- JET SSH access configured in facility YAML
- PPF library available on remote host
- Discovery roots set for JET

**Expected output:** ~500-2000 FacilitySignals covering major PPF DDAs and their DTypes, each linked to `jet:ppf:standard` DataAccess node.

### 2.2 Add JPF Signal Scanner (Gap #3)

JPF (JET Processing Facility) is a separate data system. No scanner exists.

**Design:** Create `scanners/jpf.py` following the PPF scanner pattern:
- SSH to JET, enumerate JPF signal groups
- Create `jet:jpf:standard` DataAccess node
- Create FacilitySignals with accessor like `jpf.jpfget({shot}, '{group}/{signal}')`
- Map signal groups to physics domains

**Estimated effort:** Small — mostly reuses the PPF scanner architecture.

### 2.3 Wire Wiki Cross-Reference Linking (Gap #1, #2)

`link_chunks_to_entities()` exists but is never called. Two options:

**Option A: Add CLI command**
```bash
uv run imas-codex wiki link-entities jet
```

Add to `imas_codex/cli/wiki.py`:
```python
@wiki.command("link-entities")
@click.argument("facility")
def wiki_link_entities(facility: str):
    """Create DOCUMENTS and MENTIONS_IMAS relationships from chunk metadata."""
    from imas_codex.discovery.wiki import link_chunks_to_entities
    stats = link_chunks_to_entities(facility)
    console.print(f"Linked: {stats}")
```

**Option B: Call automatically after wiki ingestion** — add to the wiki ingestion pipeline's post-processing step.

**Recommendation:** Both. Add the CLI command for manual runs, and call it automatically at the end of wiki ingestion.

### 2.4 Add DataNode Embedding Generation (related to Gap #2)

**Problem:** 6,623 JET DataNodes have descriptions but 0 have embeddings. The `data_node_desc_embedding` vector index is empty, so any vector search against DataNodes for JET returns nothing.

**Command:** `uv run imas-codex embed nodes jet --label DataNode`

If this command doesn't exist, add it to the embed CLI. The embedding pipeline already handles other node types — DataNode just needs to be added to the batch.

### 2.5 Add Content-Hash Generation CLI (Gap #1)

**Command:** Add a `wiki dedup` subcommand:
```bash
uv run imas-codex wiki dedup jet --dry-run    # Preview duplicates
uv run imas-codex wiki dedup jet              # Remove duplicates
```

Algorithm:
1. Compute SHA-256 of each WikiChunk.text
2. Group by hash
3. For each group with >1 chunk: keep the one with the lowest chunk_index (earliest in document), delete others
4. Update parent WikiPage/Document HAS_CHUNK relationships

---

## Stage 3: Graph Data Migration

### Migration Ordering

Migrations must run in dependency order. Each step is idempotent (MERGE-based).

#### Step 1: Fix Signal Contamination (Gap #3)

```cypher
-- Delete 289 contaminated JET FacilitySignals with wrong prefixes
MATCH (fs:FacilitySignal {facility_id: 'jet'})
WHERE NOT fs.id STARTS WITH 'jet:'
DETACH DELETE fs
```

**Validation:**
```cypher
MATCH (fs:FacilitySignal {facility_id: 'jet'})
WHERE NOT fs.id STARTS WITH 'jet:'
RETURN count(fs)  -- Should be 0
```

#### Step 2: Create General-Purpose DataAccess Nodes (Gap #4)

```cypher
-- JET PPF standard access
MERGE (da:DataAccess {id: 'jet:ppf:standard'})
SET da.name = 'PPF Standard Access',
    da.facility_id = 'jet',
    da.method_type = 'ppf',
    da.library = 'ppf',
    da.connection_template = 'ppfgo(pulse={shot})',
    da.data_template = 'ppfdata({shot}, ''{dda}'', ''{dtype}'')',
    da.time_template = 'ppftim({shot}, ''{dda}'', ''{dtype}'')',
    da.description = 'Standard PPF data access via ppfdata/ppfget functions'
WITH da MATCH (f:Facility {id: 'jet'}) MERGE (da)-[:AT_FACILITY]->(f);

-- JET PPF Python access
MERGE (da:DataAccess {id: 'jet:ppf:python'})
SET da.name = 'PPF Python Access',
    da.facility_id = 'jet',
    da.method_type = 'ppf',
    da.library = 'ppf',
    da.connection_template = 'import ppf',
    da.data_template = 'ppf.ppfdata({shot}, ''{dda}'', ''{dtype}'')',
    da.imports_template = 'import ppf',
    da.description = 'Python PPF module access for JET processed data'
WITH da MATCH (f:Facility {id: 'jet'}) MERGE (da)-[:AT_FACILITY]->(f);

-- JET JPF access
MERGE (da:DataAccess {id: 'jet:jpf:standard'})
SET da.name = 'JPF Standard Access',
    da.facility_id = 'jet',
    da.method_type = 'jpf',
    da.library = 'jpf',
    da.data_template = 'jpfget({shot}, ''{signal_path}'')',
    da.description = 'JET Primary Facility data access for raw diagnostic signals'
WITH da MATCH (f:Facility {id: 'jet'}) MERGE (da)-[:AT_FACILITY]->(f);

-- JET MDSplus access
MERGE (da:DataAccess {id: 'jet:mdsplus:standard'})
SET da.name = 'MDSplus Standard Access',
    da.facility_id = 'jet',
    da.method_type = 'mdsplus',
    da.connection_template = 'import MDSplus; conn = MDSplus.Connection(''{server}'')',
    da.data_template = 'conn.openTree(''{tree}'', {shot}); conn.get(''{node_path}'')',
    da.description = 'MDSplus tree access for JET experimental data'
WITH da MATCH (f:Facility {id: 'jet'}) MERGE (da)-[:AT_FACILITY]->(f);

-- JET UDA access
MERGE (da:DataAccess {id: 'jet:uda:standard'})
SET da.name = 'UDA Standard Access',
    da.facility_id = 'jet',
    da.method_type = 'uda',
    da.library = 'pyuda',
    da.connection_template = 'import pyuda; client = pyuda.Client()',
    da.data_template = 'client.get(''{signal}'', {shot})',
    da.description = 'Universal Data Access for JET data via IDAM/UDA client'
WITH da MATCH (f:Facility {id: 'jet'}) MERGE (da)-[:AT_FACILITY]->(f);
```

#### Step 3: Create Missing Code Relationships (Gap #5)

Already partially addressed by commit `70b7076`. Verify and fix remaining gaps:

```cypher
-- 3a. Fix remaining orphaned CodeChunks (no CODE_EXAMPLE_ID relationship)
MATCH (cc:CodeChunk)
WHERE cc.code_example_id IS NOT NULL
  AND NOT (cc)-[:CODE_EXAMPLE_ID]->(:CodeExample)
MATCH (ce:CodeExample {id: cc.code_example_id})
MERGE (cc)-[:CODE_EXAMPLE_ID]->(ce)

-- 3b. Create FROM_FILE relationships
MATCH (ce:CodeExample)
WHERE ce.source_file IS NOT NULL AND ce.facility_id IS NOT NULL
  AND NOT (ce)-[:FROM_FILE]->(:CodeFile)
MATCH (cf:CodeFile)
WHERE cf.path = ce.source_file AND cf.facility_id = ce.facility_id
MERGE (ce)-[:FROM_FILE]->(cf)

-- 3c. Create HAS_EXAMPLE from CodeFile to CodeExample
MATCH (ce:CodeExample)-[:FROM_FILE]->(cf:CodeFile)
WHERE NOT (cf)-[:HAS_EXAMPLE]->(ce)
MERGE (cf)-[:HAS_EXAMPLE]->(ce)
```

**Validation:**
```cypher
-- Should return 0
MATCH (cc:CodeChunk)
WHERE cc.code_example_id IS NOT NULL
  AND NOT (cc)-[:CODE_EXAMPLE_ID]->(:CodeExample)
RETURN count(cc) AS orphaned_chunks

-- Should return 0 (or count of files not discoverable as CodeFile)
MATCH (ce:CodeExample)
WHERE ce.source_file IS NOT NULL
  AND NOT (ce)-[:FROM_FILE]->(:CodeFile)
RETURN count(ce) AS unlinked_examples
```

#### Step 4: Create Wiki Cross-Reference Relationships (Gaps #1, #2)

Run `link_chunks_to_entities('jet')` — the function exists, just needs to be called:

```python
from imas_codex.discovery.wiki.pipeline import link_chunks_to_entities
stats = link_chunks_to_entities('jet')
# Expected: {data_nodes_linked: N, imas_paths_linked: N, ppf_signals_linked: N}
```

**Dependency:** Step 2 must run first (PPF signal scanner) for `ppf_signals_linked` to match any FacilitySignals.

**Known issue:** The PPF path matching in `link_chunks_to_entities` uses `fs.id ENDS WITH ppf_path OR fs.name = ppf_path`. After PPF scanner runs, signal IDs will be like `jet:magnetics/hrts_te` while wiki chunks have PPF paths like `HRTS/TE`. The matching logic may need adjustment to handle case-insensitive DDA/DTYPE matching.

**Fix for matching:**
```cypher
-- Better PPF matching: normalize to upper and compare components
MATCH (c:WikiChunk {facility_id: $facility_id})
WHERE c.ppf_paths_mentioned IS NOT NULL AND size(c.ppf_paths_mentioned) > 0
UNWIND c.ppf_paths_mentioned AS ppf_path
WITH c, ppf_path,
     toLower(split(ppf_path, '/')[0]) AS dda,
     CASE WHEN size(split(ppf_path, '/')) > 1
          THEN toLower(split(ppf_path, '/')[1])
          ELSE NULL END AS dtype
MATCH (fs:FacilitySignal {facility_id: $facility_id})
WHERE fs.id ENDS WITH dda + '_' + dtype
   OR fs.name = ppf_path
MERGE (c)-[:DOCUMENTS]->(fs)
```

#### Step 5: Deduplicate Wiki Chunks (Gap #1)

```cypher
-- Find duplicates, keep lowest chunk_index per text
MATCH (c:WikiChunk {facility_id: 'jet'})
WITH c.text AS text, collect(c) AS chunks
WHERE size(chunks) > 1
WITH text, chunks,
     [x IN chunks | x.chunk_index] AS indices,
     head([x IN chunks WHERE x.chunk_index = 
           apoc.coll.min([y IN chunks | y.chunk_index])]) AS keeper
UNWIND [x IN chunks WHERE x <> keeper] AS dup
DETACH DELETE dup
```

**Note:** This requires APOC. A simpler alternative without APOC:

```cypher
-- Step 5a: Mark duplicates with a dedup pass
MATCH (c:WikiChunk {facility_id: 'jet'})
WITH c.text AS text, collect(c) AS chunks
WHERE size(chunks) > 1
WITH text, head(chunks) AS keeper, tail(chunks) AS dupes
UNWIND dupes AS dup
DETACH DELETE dup
```

**Validation:**
```cypher
MATCH (c:WikiChunk {facility_id: 'jet'})
WITH c.text AS text, count(*) AS cnt
WHERE cnt > 1
RETURN sum(cnt - 1) AS remaining_duplicates  -- Should be 0
```

#### Step 6: Create Fulltext Indexes (Gap #7)

```cypher
CREATE FULLTEXT INDEX wiki_chunk_text IF NOT EXISTS
FOR (n:WikiChunk) ON EACH [n.text]
OPTIONS { indexConfig: { `fulltext.analyzer`: 'standard-no-stop-words' } };

CREATE FULLTEXT INDEX code_chunk_text IF NOT EXISTS
FOR (n:CodeChunk) ON EACH [n.text]
OPTIONS { indexConfig: { `fulltext.analyzer`: 'standard-no-stop-words' } };

CREATE FULLTEXT INDEX facility_signal_text IF NOT EXISTS
FOR (n:FacilitySignal) ON EACH [n.name, n.description]
OPTIONS { indexConfig: { `fulltext.analyzer`: 'standard-no-stop-words' } };

CREATE FULLTEXT INDEX data_node_text IF NOT EXISTS
FOR (n:DataNode) ON EACH [n.description, n.canonical_path]
OPTIONS { indexConfig: { `fulltext.analyzer`: 'standard-no-stop-words' } };
```

#### Step 7: Generate DataNode Embeddings (Related to Gap #2)

```python
# Run embedding generation for all 6,623 JET DataNodes with descriptions
# This enables vector search via data_node_desc_embedding index
uv run imas-codex embed batch jet --label DataNode
```

#### Step 8: Clean Up Legacy Vector Index

```cypher
-- Remove legacy index if it's an alias for data_node_desc_embedding
DROP INDEX tree_node_desc_embedding IF EXISTS
```

---

## Stage 4: MCP Tool Fixes

### 4.1 Wiki Chunk Deduplication in Formatters (Gap #1)

**File:** `imas_codex/llm/search_formatters.py`, `format_docs_report()` (line 226)

**Current behavior:** Groups chunks by page but shows all chunks including duplicates. A query about "equilibrium" may return 5 identical proposal-template chunks.

**Fix:** Add content-hash dedup before formatting:

```python
def format_docs_report(results: list[dict], query: str, ...) -> str:
    # Deduplicate by text content hash
    seen_texts: set[str] = set()
    unique_results = []
    for r in results:
        text = r.get("text", "")
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        if text_hash not in seen_texts:
            seen_texts.add(text_hash)
            unique_results.append(r)
    results = unique_results
    # ... rest of formatter
```

**Impact:** Eliminates ~5,576 duplicate chunks from JET search results immediately, without waiting for graph migration.

### 4.2 Title-Match Relevance Boosting (Gap #2)

**File:** `imas_codex/llm/search_tools.py`, `_search_docs()` (line 295)

**Problem:** Searching for "PPF data access" returns wiki chunks about semantically similar but topically irrelevant content because vector similarity alone can't distinguish between "mentions PPF" and "is about PPF".

**Fix:** Add title/page-name boosting to the hybrid merge:

```python
def _merge_results(vector_results, text_results, query, ...):
    # Existing weighted merge...
    merged = ...
    
    # Title-match boost: if query terms appear in page title, boost score
    query_terms = set(query.lower().split())
    for r in merged:
        page_title = (r.get("page_title") or r.get("wiki_page_id") or "").lower()
        title_terms = set(page_title.replace("_", " ").split())
        overlap = len(query_terms & title_terms)
        if overlap > 0:
            r["score"] = min(1.0, r["score"] + 0.1 * overlap)
    
    # Re-sort after boosting
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged
```

### 4.3 Replace Hardcoded Text Scores with Fulltext BM25 (Gap #7)

**Files:** `search_tools.py` — `_text_search_wiki_chunks()` (line 404), `_text_search_code_chunks()` (line 870), `_text_search_signals()` (line 185)

**Current:** All text search functions use `toLower(n.text) CONTAINS toLower($query)` and return hardcoded scores (0.5, 0.6).

**Fix (after Stage 3, Step 6 creates fulltext indexes):**

```python
def _text_search_wiki_chunks(query: str, facility: str | None, k: int) -> list[dict]:
    cypher = """
        CALL db.index.fulltext.queryNodes('wiki_chunk_text', $query)
        YIELD node, score
        WHERE ($facility IS NULL OR node.facility_id = $facility)
        RETURN node.id AS id, node.text AS text, node.facility_id AS facility_id,
               node.wiki_page_id AS wiki_page_id, score
        LIMIT $k
    """
    with GraphClient() as gc:
        return list(gc.query(cypher, query=query, facility=facility, k=k))
```

**Benefit:** BM25 scores reflect actual term frequency and inverse document frequency. Results are pre-filtered and pre-ranked by the index engine, eliminating sequential scans.

**Fallback:** Keep the CONTAINS fallback for queries with special characters that Lucene doesn't handle well. Check if fulltext index exists at startup and route accordingly.

### 4.4 Wiki Category/Site Filtering (Gap #2)

**File:** `search_tools.py`, `_search_docs()` (line 295)

**Current:** No way to filter wiki results by source site or category. JET has multiple wiki sources (Confluence, MediaWiki) with different content types.

**Fix:** Add optional `site` parameter and filter by WikiPage.wiki_site:

```python
async def _search_docs(query: str, facility: str | None = None,
                       site: str | None = None, k: int = 20) -> str:
    # In vector search enrichment, filter by site:
    # MATCH (c:WikiChunk)-[:HAS_CHUNK]-(p:WikiPage)
    # WHERE $site IS NULL OR p.wiki_site CONTAINS $site
```

### 4.5 Vector Pre-Filter Strategy (Gap #8)

**File:** `search_tools.py`, `_vector_search_signals()` (line 140), `_vector_search_wiki_chunks()` (line 360)

**Current:** Over-fetches k×20 (min 200) from vector index, then filters by `facility_id` in Python. For facilities with few nodes in a large index, most fetched results are discarded.

**Fix:** Neo4j 5.x supports `db.index.vector.queryNodes()` with pre-filter via Cypher. But the pre-filter approach has limitations — Neo4j's vector index doesn't natively support property pre-filtering in the index scan itself. The actual fix is:

1. **Reduce over-fetch factor** from k×20 to k×5 for well-populated indexes
2. **Use facility-specific vector indexes** for high-cardinality cases (create per-facility indexes if a facility has >10K embeddings)
3. **Short-term:** Accept the over-fetch as a reasonable tradeoff — it adds latency but doesn't affect correctness

**Recommendation:** Low priority. The k×20 strategy works correctly. Only optimize if latency becomes a user-reported issue.

### 4.6 Enrich Wiki Chunks with Cross-References (Gap #2)

**File:** `search_tools.py`, `_enrich_wiki_chunks()` (line 380)

**Current enrichment traversals:**
```cypher
MATCH (c:WikiChunk)-[:HAS_CHUNK]-(p:WikiPage)
OPTIONAL MATCH (c)-[:DOCUMENTS]->(fs:FacilitySignal)
OPTIONAL MATCH (c)-[:DOCUMENTS]->(dn:DataNode)
OPTIONAL MATCH (c)-[:MENTIONS_IMAS]->(ip:IMASPath)
```

These traversals are already in the search tools but return empty because the relationships don't exist (0 DOCUMENTS, 0 MENTIONS_IMAS). After Stage 3 Step 4 runs `link_chunks_to_entities()`, these traversals will start returning data.

**No code change needed** — the search tools are already correct. The gap is purely in the graph data.

**Additional enrichment to add:** Surface the metadata properties as a fallback until relationships are created:

```python
# In _enrich_wiki_chunks, add fallback for metadata properties
OPTIONAL MATCH (c)-[:DOCUMENTS]->(fs:FacilitySignal)
OPTIONAL MATCH (c)-[:DOCUMENTS]->(dn:DataNode)
OPTIONAL MATCH (c)-[:MENTIONS_IMAS]->(ip:IMASPath)
RETURN ...,
    CASE WHEN fs IS NOT NULL THEN collect(DISTINCT fs.id)
         ELSE c.ppf_paths_mentioned END AS ppf_refs,
    CASE WHEN ip IS NOT NULL THEN collect(DISTINCT ip.id)
         ELSE c.imas_paths_mentioned END AS imas_refs,
    c.tool_mentions AS tool_mentions
```

### 4.7 Signal Enrichment for New DataAccess Patterns (Gap #4)

**File:** `search_tools.py`, `_enrich_signals()` (line 210)

**Current:** Traverses `(fs)-[:DATA_ACCESS]->(da:DataAccess)` and interpolates templates. This already works — once Stage 3 Step 2 creates the general DataAccess nodes and the PPF scanner links FacilitySignals to them, the enrichment will surface access patterns.

**Addition:** Format PPF vs MDSplus vs UDA access differently in the signal report:

```python
# In format_signals_report, add method_type-aware formatting:
if method.get("method_type") == "ppf":
    code_block = f"ppfdata({shot}, '{dda}', '{dtype}')"
elif method.get("method_type") == "uda":
    code_block = f"client.get('{signal}', {shot})"
```

### 4.8 Code Search Facility Filter Fix (Gap #5)

**File:** `search_tools.py`, `_vector_search_code_chunks()` (approx line 810)

**Current:** Uses `cc.facility_id = $facility` property filter. This works because commit `70b7076` added `facility_id` as a denormalized property on CodeChunks. No change needed.

**Validation:** Verify that all CodeChunks have `facility_id` set:
```cypher
MATCH (cc:CodeChunk) WHERE cc.facility_id IS NULL RETURN count(cc)
-- Should be 0
```

---

## Implementation Priority Matrix

### Phase A: Quick Wins (immediate, no schema changes)

| Task | Stage | Gaps Fixed | Effort | Impact |
|------|-------|-----------|--------|--------|
| Run `link_chunks_to_entities('jet')` | 3.4 | #1, #2 | 5 min | Creates 578+ PPF and 655+ IMAS cross-references |
| Dedup in `format_docs_report()` | 4.1 | #1 | 30 min | Eliminates 5,576 duplicates from search results |
| Delete contaminated signals | 3.1 | #3 | 5 min | Removes 289 garbage signals |
| Create general DataAccess nodes | 3.2 | #4 | 15 min | Enables PPF/JPF/MDSplus/UDA template interpolation |

### Phase B: Infrastructure (days, requires dev work)

| Task | Stage | Gaps Fixed | Effort | Impact |
|------|-------|-----------|--------|--------|
| Run PPF signal scanner | 2.1 | #3 | 1 session | Populates ~500-2000 JET measurement signals |
| Fix code relationships | 3.3 | #5 | 30 min | Links 5,469 orphaned CodeChunks |
| Create fulltext indexes | 3.6 | #7 | 15 min | Enables BM25-scored text search |
| Update text search functions | 4.3 | #7 | 2 hours | Real relevance scores instead of hardcoded values |
| Add title-match boosting | 4.2 | #2 | 1 hour | Reduces vector relevance drift |

### Phase C: Deep Fixes (multi-session, broader infrastructure)

| Task | Stage | Gaps Fixed | Effort | Impact |
|------|-------|-----------|--------|--------|
| Deduplicate wiki chunk graph | 3.5 | #1 | 1 hour | Removes 5,576 nodes, shrinks vector index |
| Generate DataNode embeddings | 3.7 | #2 | 1 session | Enables 6,623 DataNodes in vector search |
| Add JPF signal scanner | 2.2 | #3 | 1 session | Covers JET's second major data system |
| Add `wiki link-entities` CLI | 2.3 | #1, #2 | 1 hour | Makes cross-ref generation repeatable |
| Add `content_hash` to schema | 1.5 | #1 | 30 min | Prevents future duplicate accumulation |
| Add wiki metadata fallback enrichment | 4.6 | #2 | 1 hour | Surfaces metadata until relationships fully populated |

### Phase D: Optimization (low priority)

| Task | Stage | Gaps Fixed | Effort | Impact |
|------|-------|-----------|--------|--------|
| Reduce over-fetch factor | 4.5 | #8 | 30 min | Minor latency improvement |
| Clean up legacy vector index | 3.8 | — | 5 min | Housekeeping |
| Per-facility vector indexes | 4.5 | #8 | 2 hours | Only if latency is a reported problem |

---

## Validation Plan

After each phase, run these diagnostic queries to verify progress:

```cypher
-- Gap #1: Deduplication
MATCH (c:WikiChunk {facility_id: 'jet'})
WITH c.text AS text, count(*) AS cnt WHERE cnt > 1
RETURN sum(cnt - 1) AS duplicates  -- Target: 0

-- Gap #2: Cross-references populated
MATCH (c:WikiChunk {facility_id: 'jet'})-[:DOCUMENTS]->(t)
RETURN labels(t)[0] AS type, count(*) AS cnt
-- Target: DataNode > 0, FacilitySignal > 0

MATCH (c:WikiChunk {facility_id: 'jet'})-[:MENTIONS_IMAS]->(ip)
RETURN count(*) AS imas_links  -- Target: > 0

-- Gap #3: PPF signals exist
MATCH (fs:FacilitySignal {facility_id: 'jet'})
WHERE fs.accessor STARTS WITH 'ppfdata'
RETURN count(fs) AS ppf_signals  -- Target: > 500

-- Gap #4: DataAccess coverage
MATCH (da:DataAccess {facility_id: 'jet'})
RETURN da.id, da.method_type  -- Target: ppf, jpf, mdsplus, uda

-- Gap #5: Code relationships
MATCH (cc:CodeChunk {facility_id: 'jet'})
WHERE cc.code_example_id IS NOT NULL
  AND NOT (cc)-[:CODE_EXAMPLE_ID]->(:CodeExample)
RETURN count(cc) AS orphaned  -- Target: 0

-- Gap #7: Fulltext indexes
SHOW FULLTEXT INDEXES
-- Target: wiki_chunk_text, code_chunk_text, facility_signal_text, data_node_text

-- Overall: search tool E2E test
-- Run: search_signals("plasma current", facility="jet")
-- Expect: PPF signals with access templates
-- Run: search_docs("equilibrium reconstruction", facility="jet")
-- Expect: Relevant wiki pages with cross-references, no duplicates
```

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| PPF scanner SSH timeout | Medium | Blocks signal population | Use chunked enumeration, persist progress |
| `link_chunks_to_entities` matching logic too loose | Medium | False positive DOCUMENTS relationships | Validate sample matches manually before bulk run |
| Fulltext index creation OOM | Low | Index build fails | Create indexes one at a time, monitor heap |
| Wiki dedup deletes chunks with different context | Low | Lose page-specific context | Check chunk_index and wiki_page_id before delete |
| DataNode embedding batch overwhelms GPU | Low | Embedding server crash | Use batch size limits, monitor VRAM |

## Dependencies Between Steps

```
Stage 1 (Schema) ─── independent, can proceed in parallel ──→ Stage 4 (MCP)
                                                                  ↑
Stage 2 (CLI) ───── feeds ──→ Stage 3 (Migration) ──── feeds ──→ │
                                                                  │
  2.1 PPF scanner ──→ 3.2 DataAccess ──→ 3.4 Wiki cross-refs    │
  2.3 Wiki CLI ──→ (enables repeatable 3.4)                       │
  2.4 DataNode embeds ──→ (enables 4.6 enrichment)                │
                                                                  │
  3.6 Fulltext indexes ──→ 4.3 BM25 text search ─────────────────┘
  3.5 Dedup ──→ 4.1 Formatter dedup (complementary) ─────────────┘
```

Phase A items have zero dependencies and can run immediately.
