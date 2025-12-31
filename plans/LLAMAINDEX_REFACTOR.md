# LlamaIndex Code Embedding Refactor Plan

> **Goal**: Replace ~1000 lines of custom orchestration code with LlamaIndex's native `IngestionPipeline` and `Neo4jVectorStore`.

## Current Status

### What We Have (Custom Code)

| File | LOC | Purpose |
|------|-----|---------|
| `code_examples/ingester.py` | ~587 | Full orchestration: fetch files, chunk with CodeSplitter, batch embed, write to Neo4j |
| `code_examples/processor.py` | ~170 | Async queue processing with `anyio.to_thread.run_sync()` |
| `code_examples/queue.py` | ~200 | File staging queue with JSON manifest |
| `code_examples/search.py` | ~200 | Vector search with raw Cypher + fallback |

**Total: ~1150 lines of custom code**

### What LlamaIndex Provides (That We're Not Using)

| LlamaIndex Feature | Our Custom Replacement |
|-------------------|----------------------|
| `IngestionPipeline` | `CodeExampleIngester` class (400+ lines) |
| `Neo4jVectorStore` | Raw Cypher queries + manual vector index |
| `VectorStoreIndex` | `CodeExampleSearch` with raw Cypher |
| Transformation caching | None (reprocess every time) |
| Document management | Custom queue.py |
| `pipeline.arun()` async | Custom `anyio.to_thread.run_sync()` |

### Known Issues

1. **Line numbers broken**: CodeSplitter doesn't provide `start_line`/`end_line` metadata - our code assumes it does
2. **No caching**: Re-embedding identical files wastes compute
3. **Maintenance burden**: We maintain chunking, embedding, storage orchestration ourselves

## What We Must Keep Custom

1. **IDS Detection** - IMAS-specific regex patterns to detect IDS references in code
2. **SSH/SCP File Transfer** - Remote facility access via Fabric
3. **Graph Relationships** - `CodeExample -[:HAS_CHUNK]-> CodeChunk -[:RELATED_PATHS]-> IMASPath`

---

## Implementation Plan

### Phase 1: Create Custom IDSExtractor Transformation

LlamaIndex supports custom transformations via `TransformComponent`. Create one for IDS extraction:

```python
# code_examples/ids_extractor.py
import re
from llama_index.core.schema import TransformComponent, TextNode

# IDS detection patterns (from current ingester.py)
IDS_PATTERNS = [
    r"ids\.(\w+)",
    r'ids_name\s*=\s*["\'](\w+)["\']',
    r"get_ids\(['\"](\w+)['\"]",
    r"put_ids\(['\"](\w+)['\"]",
    r"imas\.(\w+)\(",
    r"imasdb\.get\(['\"](\w+)['\"]",
]

VALID_IDS_NAMES = {
    "equilibrium", "core_profiles", "core_sources", "summary",
    "wall", "pf_active", "magnetics", "thomson_scattering",
    "interferometer", "ece", "nbi", "ic_antennas", "ec_launchers",
    # ... full list from IMAS DD
}

class IDSExtractor(TransformComponent):
    """Extract IMAS IDS references from code chunks."""
    
    def __call__(self, nodes: list[TextNode], **kwargs) -> list[TextNode]:
        for node in nodes:
            ids_refs = self._extract_ids(node.text)
            if ids_refs:
                node.metadata["related_ids"] = list(ids_refs)
        return nodes
    
    def _extract_ids(self, text: str) -> set[str]:
        found = set()
        for pattern in IDS_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                ids_name = match.group(1).lower()
                if ids_name in VALID_IDS_NAMES:
                    found.add(ids_name)
        return found
```

### Phase 2: Create IngestionPipeline Configuration

Replace `CodeExampleIngester` with LlamaIndex's `IngestionPipeline`:

```python
# code_examples/pipeline.py
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import CodeSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore

from .ids_extractor import IDSExtractor

def create_pipeline(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j", 
    neo4j_password: str = "imas-codex",
) -> IngestionPipeline:
    """Create LlamaIndex ingestion pipeline for code examples."""
    
    vector_store = Neo4jVectorStore(
        username=neo4j_user,
        password=neo4j_password,
        url=neo4j_uri,
        embed_dim=384,  # all-MiniLM-L6-v2
        index_name="code_chunk_embedding",
        node_label="CodeChunk",
        text_node_property="content",
        embedding_node_property="embedding",
    )
    
    return IngestionPipeline(
        transformations=[
            CodeSplitter(language="python", chunk_lines=40, chunk_lines_overlap=10),
            IDSExtractor(),
            HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        ],
        vector_store=vector_store,
    )

async def ingest_code_files(
    pipeline: IngestionPipeline,
    files: list[tuple[str, str, str]],  # (path, content, language)
    facility_id: str,
) -> list:
    """Ingest code files into the pipeline.
    
    Args:
        pipeline: Configured IngestionPipeline
        files: List of (remote_path, content, language) tuples
        facility_id: Facility identifier (e.g., "epfl")
    
    Returns:
        List of ingested nodes
    """
    documents = [
        Document(
            text=content,
            metadata={
                "source_file": path,
                "facility_id": facility_id,
                "language": language,
            }
        )
        for path, content, language in files
    ]
    
    # Use async ingestion with caching
    nodes = await pipeline.arun(documents=documents)
    return nodes
```

### Phase 3: Create Graph Relationship Post-Processor

After ingestion, create `RELATED_PATHS` relationships to IMASPath nodes:

```python
# code_examples/graph_linker.py
from imas_codex.graph import GraphClient

def link_chunks_to_imas_paths(graph_client: GraphClient) -> int:
    """Create RELATED_PATHS relationships for chunks with IDS references.
    
    Returns:
        Number of relationships created
    """
    cypher = """
        MATCH (c:CodeChunk)
        WHERE c.related_ids IS NOT NULL
        UNWIND c.related_ids AS ids_name
        MATCH (p:IMASPath {ids: ids_name})
        MERGE (c)-[:RELATED_PATHS]->(p)
        RETURN count(*) AS created
    """
    with graph_client:
        result = graph_client.query(cypher)
        return result[0]["created"] if result else 0
```

### Phase 4: Simplify Search with VectorStoreIndex

Replace custom `CodeExampleSearch` with LlamaIndex's `VectorStoreIndex`:

```python
# code_examples/search.py (simplified)
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore

def create_search_index(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "imas-codex",
) -> VectorStoreIndex:
    """Create search index from existing Neo4j vector store."""
    
    vector_store = Neo4jVectorStore(
        username=neo4j_user,
        password=neo4j_password,
        url=neo4j_uri,
        embed_dim=384,
        index_name="code_chunk_embedding",
        node_label="CodeChunk",
        text_node_property="content",
        embedding_node_property="embedding",
    )
    
    return VectorStoreIndex.from_vector_store(vector_store)

def search_code_examples(
    index: VectorStoreIndex,
    query: str,
    top_k: int = 10,
    facility: str | None = None,
    ids_filter: list[str] | None = None,
):
    """Search for code examples using semantic similarity."""
    
    # Build metadata filters
    filters = {}
    if facility:
        filters["facility_id"] = facility
    if ids_filter:
        # Neo4jVectorStore supports metadata filtering
        filters["related_ids"] = {"$in": ids_filter}
    
    retriever = index.as_retriever(
        similarity_top_k=top_k,
        filters=filters if filters else None,
    )
    
    return retriever.retrieve(query)
```

### Phase 5: Keep SSH/SCP File Transfer

Keep the facility file fetching logic (it's genuinely custom):

```python
# code_examples/facility_reader.py
from pathlib import Path
from fabric import Connection

def fetch_remote_files(
    facility_host: str,
    remote_paths: list[str],
    local_staging: Path,
) -> list[tuple[str, str, str]]:
    """Fetch files from remote facility via SCP.
    
    Returns:
        List of (remote_path, content, language) tuples
    """
    files = []
    
    with Connection(facility_host) as conn:
        for remote_path in remote_paths:
            local_path = local_staging / Path(remote_path).name
            conn.get(remote_path, str(local_path))
            
            content = local_path.read_text()
            language = _detect_language(remote_path)
            files.append((remote_path, content, language))
            
            local_path.unlink()  # Clean up
    
    return files

def _detect_language(path: str) -> str:
    """Detect programming language from file extension."""
    ext_map = {
        ".py": "python",
        ".m": "matlab",
        ".f90": "fortran",
        ".f": "fortran",
        ".c": "c",
        ".cpp": "cpp",
        ".jl": "julia",
        ".pro": "idl",
    }
    ext = Path(path).suffix.lower()
    return ext_map.get(ext, "python")
```

---

## Final Architecture

```
code_examples/
├── __init__.py           # Export public API
├── ids_extractor.py      # ~50 LOC - Custom TransformComponent for IDS detection
├── facility_reader.py    # ~60 LOC - SSH/SCP file fetching
├── graph_linker.py       # ~30 LOC - Create RELATED_PATHS relationships
├── pipeline.py           # ~60 LOC - Configure IngestionPipeline
└── search.py             # ~50 LOC - VectorStoreIndex wrapper
```

**Total: ~250 lines instead of ~1150 lines (78% reduction)**

---

## Dependencies to Add

```toml
[project.dependencies]
llama-index-vector-stores-neo4jvector = ">=0.3"
```

---

## Migration Steps

1. [ ] Add `llama-index-vector-stores-neo4jvector` dependency
2. [ ] Create `ids_extractor.py` with `IDSExtractor` transformation
3. [ ] Create `pipeline.py` with `IngestionPipeline` configuration
4. [ ] Create `graph_linker.py` for post-ingestion relationship creation
5. [ ] Create `facility_reader.py` (extract from current ingester.py)
6. [ ] Simplify `search.py` to use `VectorStoreIndex`
7. [ ] Update MCP tools in `agents/server.py` to use new API
8. [ ] Delete `processor.py` and `queue.py` (replaced by IngestionPipeline caching)
9. [ ] Test full workflow: fetch → ingest → search
10. [ ] Update tests

---

## Key Benefits

| Benefit | Description |
|---------|-------------|
| **Caching** | IngestionPipeline caches node+transformation pairs |
| **Async Native** | `pipeline.arun()` instead of manual `anyio.to_thread` |
| **Parallel Processing** | `pipeline.run(num_workers=4)` |
| **Hybrid Search** | Neo4jVectorStore supports keyword + vector |
| **Metadata Filtering** | Built-in for IDS and facility filtering |
| **Less Maintenance** | LlamaIndex maintains the complex parts |
| **Line Numbers** | Neo4jVectorStore may handle node metadata better |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Neo4jVectorStore schema conflicts with existing graph | Test with fresh Neo4j instance first; may need to customize node labels |
| Loss of custom graph relationships | Post-processing step to create `CodeExample` nodes and relationships |
| Different embedding storage format | May need migration script for existing data |
| Line number metadata still missing | Investigate CodeSplitter metadata or use custom splitter |

---

## References

- [LlamaIndex IngestionPipeline](https://developers.llamaindex.ai/python/framework/module_guides/loading/ingestion_pipeline/)
- [LlamaIndex Neo4jVectorStore](https://developers.llamaindex.ai/python/examples/vector_stores/neo4jvectordemo/)
- [LlamaIndex Custom Transformations](https://developers.llamaindex.ai/python/framework/module_guides/loading/ingestion_pipeline/transformations/)
- Current implementation: `imas_codex/code_examples/`
