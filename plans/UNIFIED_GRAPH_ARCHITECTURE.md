# Unified Graph Architecture for IMAS DD + Agents

> **Status**: In Progress (Phase 1)
> **Created**: 2026-01-05
> **Goal**: Unify IMAS Data Dictionary and facility knowledge in Neo4j for fast agent startup (~2s)

## Problem Statement

The current architecture has a critical startup time issue:

| Component | Import Time | Cause |
|-----------|-------------|-------|
| `imas_codex.tools.Tools` | ~57s | pint, pydantic, scipy, linkml |
| `imas_codex.graph.GraphClient` | ~28s | linkml_runtime schema introspection |
| `imas_codex.search.DocumentStore` | ~12s | pint, pydantic |

LlamaIndex agents require fast startup for interactive use. The current ~60s import time makes the agentic workflow impractical.

### Root Cause

Two separate data stores with different access patterns:

1. **IMAS DD data**: JSON files → DocumentStore → pickle embeddings
   - Rich data (documentation, units, coordinates, embeddings)
   - Heavy imports at module load (pint, pydantic, linkml)
   - Used by MCP tools

2. **Facility knowledge**: Neo4j graph
   - TreeNodes, CodeChunks, Facilities
   - Vector search for code examples
   - Used by agents

The agent tools bridge these by importing both, paying the full startup cost.

## Solution: Neo4j as Single Source of Truth

### Core Principle

**Build-time complexity, query-time simplicity.**

All heavy processing (XML parsing, text enrichment, embedding generation) happens at build time. Query time uses only the lightweight Neo4j driver.

### Target Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BUILD TIME (slow, run once)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   IMAS DD    │    │  Enrich &    │    │   Generate   │    │  Populate │ │
│  │  XML/JSON    │───▶│  Concatenate │───▶│  Embeddings  │───▶│   Neo4j   │ │
│  │   Schemas    │    │    Context   │    │  (MiniLM)    │    │   Graph   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│                                                                             │
│  Uses: pint, pydantic, linkml, sentence-transformers (~60s)                │
│  Runs: Once per DD version release                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           QUERY TIME (fast, every request)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐  │
│  │   Agent /    │    │    Graph     │    │           Neo4j              │  │
│  │  MCP Tool    │───▶│    Client    │───▶│  ┌────────────────────────┐  │  │
│  │              │    │   (neo4j     │    │  │ IMASPath nodes         │  │  │
│  └──────────────┘    │    driver)   │    │  │  - path, documentation │  │  │
│                      └──────────────┘    │  │  - units, data_type    │  │  │
│                                          │  │  - enriched_text       │  │  │
│  Uses: neo4j driver only (~2s import)   │  │  - embedding [384]     │  │  │
│  Runs: Every query                       │  │                        │  │  │
│                                          │  │ Vector Index           │  │  │
│                                          │  │  - imas_path_embedding │  │  │
│                                          │  └────────────────────────┘  │  │
│                                          └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Startup Time Target

| Component | Current | Target | How |
|-----------|---------|--------|-----|
| Agent tools import | 57s | <3s | Only import neo4j driver |
| GraphClient import | 28s | <2s | Remove linkml runtime dependency |
| First query | 0.3s | 0.3s | Same (Neo4j warm) |

## Implementation Plan

### Phase 1: Simplify GraphClient (Day 1) ✅ COMPLETE

**Goal**: Remove linkml_runtime import from query path.

**Status**: Completed 2026-01-05

**Changes Made:**
1. `graph/client.py`: Made `GraphSchema` import lazy (TYPE_CHECKING + lazy property)
2. `graph/__init__.py`: Added `__getattr__` for lazy schema utility imports
3. Removed eager `get_schema()` call from `__post_init__`

**Results:**
| Metric | Before | After |
|--------|--------|-------|
| `from imas_codex.graph import GraphClient` | 28s | 1.8s |
| Schema utilities still work when imported | N/A | ✅ |

**Validation:**
```bash
uv run python3 -c "
import time; t=time.time()
from imas_codex.graph import GraphClient
print(f'Import: {time.time()-t:.2f}s')  # Result: 1.82s ✅
"
```

### Phase 2: Enrich IMASPath Nodes (Day 2-3)

**Goal**: Store full IMAS DD data in Neo4j with context-enriched text.

**Current IMASPath nodes:**
```cypher
(:IMASPath {path: "equilibrium", ids: "equilibrium"})
-- Missing: documentation, units, data_type, coordinates, enriched_text, embedding
```

**Target IMASPath nodes:**
```cypher
(:IMASPath {
  path: "equilibrium/time_slice/profiles_1d/psi",
  ids: "equilibrium",
  documentation: "Poloidal flux profile...",
  units: "Wb",
  data_type: "FLT_1D",
  coordinates: ["1...N"],
  
  // Enriched text for embedding (concatenated ancestor context)
  enriched_text: "IDS: equilibrium. Parent: profiles_1d - 1D profiles. 
                  Path: psi. Poloidal flux profile as function of...",
  
  // Pre-computed embedding vector
  embedding: [0.123, -0.456, ...]  // 384 floats
})
```

**Text Enrichment Strategy:**

Preserve the current DocumentStore enrichment logic:
1. Start with the path's own documentation
2. Prepend IDS name and description
3. Prepend parent path descriptions (2 levels up)
4. Add units and data type context
5. This enriched text is what gets embedded

**Build Script: `build-imas-graph`**

```python
# scripts/build_imas_graph.py

def build_imas_graph(dd_version: str, neo4j_uri: str):
    """Populate Neo4j with enriched IMAS DD data."""
    
    # 1. Load existing DocumentStore (pays heavy import cost once)
    from imas_codex.search.document_store import DocumentStore
    from imas_codex.embeddings import Encoder
    
    store = DocumentStore()
    store.load_all_documents()
    
    # 2. Get or generate embeddings
    encoder = Encoder()
    embeddings = encoder.get_or_build_embeddings(store)
    
    # 3. Connect to Neo4j
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(neo4j_uri, auth=("neo4j", "imas-codex"))
    
    # 4. Batch upsert nodes with embeddings
    with driver.session() as session:
        for doc, embedding in zip(store.get_all_documents(), embeddings):
            session.run("""
                MERGE (p:IMASPath {path: $path})
                SET p.ids = $ids,
                    p.documentation = $documentation,
                    p.units = $units,
                    p.data_type = $data_type,
                    p.coordinates = $coordinates,
                    p.enriched_text = $enriched_text,
                    p.embedding = $embedding,
                    p.dd_version = $dd_version
            """, {
                "path": doc.metadata.path_id,
                "ids": doc.metadata.ids_name,
                "documentation": doc.documentation,
                "units": doc.metadata.units,
                "data_type": doc.metadata.data_type,
                "coordinates": doc.metadata.coordinates,
                "enriched_text": doc.enriched_text,  # The concatenated context
                "embedding": embedding.tolist(),
                "dd_version": dd_version,
            })
    
    # 5. Create vector index
    session.run("""
        CREATE VECTOR INDEX imas_path_embedding IF NOT EXISTS
        FOR (p:IMASPath) ON (p.embedding)
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }
        }
    """)
```

**CLI Integration:**
```bash
# Add to imas-codex CLI
imas-codex build imas-graph --dd-version 4.1.0

# Include in release workflow
imas-codex release v4.1.0 --include-imas-graph
```

### Phase 3: Update Agent Tools (Day 3-4)

**Goal**: Agent tools use Neo4j for IMAS DD queries, not DocumentStore.

**Current agent tools (`agents/tools.py`):**
```python
def _search_imas_paths(query, ids_filter, max_results):
    tools = _get_imas_tools()  # Heavy import!
    result = _run_async(tools.search_imas_paths(...))
    return format_result(result)
```

**New agent tools:**
```python
from imas_codex.graph import GraphClient

def _search_imas_paths(query: str, ids_filter: str | None, max_results: int = 10) -> str:
    """Semantic search for IMAS paths via Neo4j vector index."""
    
    # Get query embedding (lazy load encoder only when needed)
    embedding = _get_query_embedding(query)
    
    with GraphClient() as client:
        # Vector search with optional IDS filter
        cypher = """
            CALL db.index.vector.queryNodes('imas_path_embedding', $k, $embedding)
            YIELD node, score
            WHERE $ids_filter IS NULL OR node.ids IN $ids_list
            RETURN node.path AS path,
                   node.documentation AS documentation,
                   node.units AS units,
                   score
            ORDER BY score DESC
        """
        results = client.query(cypher, {
            "k": max_results * 2,  # Over-fetch for filtering
            "embedding": embedding,
            "ids_filter": ids_filter,
            "ids_list": ids_filter.split() if ids_filter else None,
        })
    
    return _format_search_results(results[:max_results])


def _get_query_embedding(query: str) -> list[float]:
    """Get embedding for query string (lazy load encoder)."""
    global _encoder
    if _encoder is None:
        # This is the only heavy import, and it's deferred
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    return _encoder.encode(query, normalize_embeddings=True).tolist()

_encoder = None
```

**Trade-off**: The first semantic search still loads sentence-transformers (~3s). Options:
1. Accept 3s first-query latency (subsequent queries instant)
2. Use OpenAI/Anthropic embeddings via API (no local model)
3. Pre-warm encoder in background thread

### Phase 4: Update MCP Tools (Day 4-5)

**Goal**: MCP tools query Neo4j instead of DocumentStore.

The MCP server (`imas-codex serve imas`) can still pay the full startup cost since it runs continuously. But for consistency, update it to use the same graph-based approach.

**Changes to `tools/search_tool.py`:**
```python
class SearchTool(BaseTool):
    async def search_imas_paths(self, query, ids_filter, max_results):
        # Option A: Keep DocumentStore for MCP (backward compatible)
        # Option B: Use Neo4j (unified approach)
        
        # Recommend Option B for consistency
        embedding = await self._get_embedding(query)
        results = await self._vector_search(embedding, ids_filter, max_results)
        return self._format_results(results)
```

**Consideration**: The MCP server currently builds SemanticSearch at startup. With Neo4j:
- No embedding cache to load
- No DocumentStore to initialize
- Just connect to Neo4j
- MCP server startup drops from ~60s to ~10s

### Phase 5: Graph Versioning (Day 5)

**Goal**: Version IMAS DD data in graph, support multiple DD versions.

**Approach:**
1. Add `dd_version` property to all IMASPath nodes
2. Include DD version in graph dumps
3. Support querying specific version:

```cypher
-- Query specific DD version
MATCH (p:IMASPath {dd_version: "4.1.0"})
WHERE p.path CONTAINS "temperature"
RETURN p.path, p.documentation
```

**Release workflow update:**
```bash
# Build graph with IMAS DD data
imas-codex build imas-graph --dd-version 4.1.0

# Dump and push
imas-codex neo4j dump
imas-codex neo4j push v4.1.0

# The graph artifact now includes:
# - Facility data (TreeNodes, CodeChunks, etc.)
# - IMAS DD data (IMASPath with embeddings)
# - Cluster data (if applicable)
```

## Data Model Changes

### IMASPath Node (Enhanced)

```yaml
# schemas/facility.yaml additions

IMASPath:
  description: IMAS Data Dictionary path with full metadata
  attributes:
    path:
      identifier: true
      description: Full path (e.g., equilibrium/time_slice/profiles_1d/psi)
    ids:
      description: IDS name (e.g., equilibrium)
    documentation:
      description: Path documentation from DD
    units:
      description: Physical units (e.g., Wb, m, eV)
    data_type:
      description: IMAS data type (e.g., FLT_1D, STR_0D)
    coordinates:
      multivalued: true
      description: Coordinate references
    enriched_text:
      description: Concatenated context for embedding
    embedding:
      description: 384-dim embedding vector (all-MiniLM-L6-v2)
    dd_version:
      description: Data dictionary version
    lifecycle:
      description: active, deprecated, obsolete
```

### New Indexes

```cypher
-- Vector index for semantic search
CREATE VECTOR INDEX imas_path_embedding IF NOT EXISTS
FOR (p:IMASPath) ON (p.embedding)
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 384,
        `vector.similarity_function`: 'cosine',
        `vector.quantization.enabled`: true
    }
}

-- Full-text index for keyword search
CREATE FULLTEXT INDEX imas_path_text IF NOT EXISTS
FOR (p:IMASPath)
ON EACH [p.path, p.documentation, p.enriched_text]

-- Composite index for filtered queries
CREATE INDEX imas_path_ids IF NOT EXISTS
FOR (p:IMASPath) ON (p.ids)
```

## Migration Strategy

### For Existing Graphs

```python
# scripts/migrate_imas_paths.py

def migrate_imas_paths():
    """Add full metadata to existing IMASPath stubs."""
    
    # Load source data
    store = DocumentStore()
    store.load_all_documents()
    encoder = Encoder()
    embeddings_cache = encoder.load_or_build_embeddings(store)
    
    # Update existing nodes
    with GraphClient() as client:
        for doc, embedding in zip(store.get_all_documents(), embeddings_cache):
            client.query("""
                MATCH (p:IMASPath {path: $path})
                SET p += $properties
            """, {
                "path": doc.metadata.path_id,
                "properties": {
                    "documentation": doc.documentation,
                    "units": doc.metadata.units,
                    # ... all properties
                    "embedding": embedding.tolist(),
                }
            })
        
        # Create any missing nodes
        # Create vector index
```

### For New Graphs

New graphs built with `imas-codex build imas-graph` will have full data from the start.

## Validation Checklist

### Performance
- [ ] `from imas_codex.graph import GraphClient` imports in <3s
- [ ] Agent tools import in <3s (without using sentence-transformers)
- [ ] First semantic search <5s (includes model load)
- [ ] Subsequent semantic searches <0.5s

### Functionality
- [ ] Semantic search returns same results as current MCP tool
- [ ] Path lookup returns full documentation
- [ ] IDS filtering works correctly
- [ ] Hybrid search (vector + keyword) works

### Data Integrity
- [ ] All 19,136 paths have embeddings
- [ ] Enriched text matches current DocumentStore logic
- [ ] DD version tracked correctly
- [ ] Deprecated paths marked appropriately

## Decisions Made (2026-01-05)

1. **Query embedding loading**: Option A - Lazy-load sentence-transformers on first query (~3s penalty once). Consider pre-warmed background solution in future if needed.

2. **MCP backward compatibility**: Option B - Keep existing MCP tools using DocumentStore. Don't break until graph-based replacement is proven.

3. **Multiple DD versions**: Option B - Support all versions from outset. Use epoch-based architecture already in graph. Must be compatible with imas-data-dictionaries package.

## Open Questions

1. **Embedding model updates?**
   - Model is fixed (all-MiniLM-L6-v2)
   - If model changes, full re-embed required
   - Store model name in graph metadata

2. **Cluster data in graph?**
   - Currently in JSON + NPZ files
   - Could migrate to graph (ClusterInfo nodes)
   - Lower priority than IMASPath migration

## Timeline

| Day | Task | Status | Notes |
|-----|------|--------|-------|
| 1 | Simplify GraphClient | ✅ Complete | 28s → 1.8s import time |
| 2 | Build script for IMAS graph | 🔲 Pending | `build-imas-graph` CLI |
| 3 | Populate test graph | 🔲 Pending | 19,136 IMASPath nodes with embeddings |
| 4 | Update agent tools | 🔲 Pending | Agents use Neo4j for IMAS search |
| 5 | Update MCP tools | 🔲 Pending | Unified query path |
| 6 | Testing & validation | 🔲 Pending | Performance + functionality tests |
| 7 | Documentation | 🔲 Pending | Update AGENTS.md, README |

## Progress Log

### 2026-01-05: Phase 1 Complete

**GraphClient import optimized from 28s to 1.8s**

Changes:
- Made `GraphSchema` import lazy in `graph/client.py`
- Added `__getattr__` for lazy schema imports in `graph/__init__.py`
- Schema utilities still work when explicitly imported

Remaining agent tools import time (~22s) is from:
- LlamaIndex core (~4s) - Necessary framework
- LlamaIndex LLM integrations (~16s) - Transitive dependency from llama-index-llms-openai-like

This will be addressed in Phase 3 when agent tools are updated to use GraphClient directly for IMAS queries, avoiding the need to import the full Tools class.

## Success Criteria

1. **Agent startup**: `from imas_codex.agents.tools import get_imas_tools` in <3s
2. **Search parity**: Same semantic search quality as current implementation
3. **Single source**: Neo4j is the only data store for IMAS DD queries
4. **Maintainability**: Clear build → query separation

## References

- [Neo4j Vector Search](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)
- [Current DocumentStore](../imas_codex/search/document_store.py)
- [Current Embeddings](../imas_codex/embeddings/)
- [LlamaIndex Agents Plan](./LLAMAINDEX_AGENTS.md)
