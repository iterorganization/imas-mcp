# Wiki Ingestion Architecture

The wiki ingestion system extracts and indexes content from facility MediaWiki documentation, enabling semantic search over authoritative signal descriptions, conventions, and diagnostic documentation.

## Overview

The system uses a three-phase pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Wiki Ingestion Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1: SCAN (No LLM)                                        │
│  ─────────────────────                                          │
│  - Start from portal page (e.g., Portal:TCV)                    │
│  - Extract links via SSH + curl                                 │
│  - Create WikiPage nodes with status='discovered'               │
│  - Track link_depth, in_degree, out_degree metrics              │
│                                                                  │
│  Phase 2: SCORE (ReAct Agent)                                   │
│  ──────────────────────────                                     │
│  - Agent evaluates pages using graph metrics                    │
│  - Assigns interest_score (0.0-1.0)                             │
│  - High-value: diagnostics, signals, codes, conventions         │
│  - Low-value: meetings, events, stubs                           │
│  - Sets status='scored' or 'skipped'                            │
│                                                                  │
│  Phase 3: INGEST (Deterministic)                                │
│  ─────────────────────────────                                  │
│  - Fetch full HTML content via SSH                              │
│  - Extract text with MediaWiki-aware parser                     │
│  - Chunk using LlamaIndex SentenceSplitter                      │
│  - Generate embeddings (Qwen3-Embedding-8B)                    │
│  - Create WikiChunk nodes with vector index                     │
│  - Link to TreeNodes and IMASPaths                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## CLI Commands

```bash
# Full discovery pipeline: scan + score
uv run imas-codex wiki discover tcv

# Individual phases
uv run imas-codex wiki scan tcv       # Link extraction only
uv run imas-codex wiki score tcv       # Agent evaluation
uv run imas-codex wiki ingest tcv      # Fetch, chunk, embed

# Check status
uv run imas-codex wiki status tcv
```

## Graph Schema

### WikiPage

Represents a wiki page with discovery metadata.

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Canonical ID: `facility:PageName` |
| `title` | string | Page title (decoded) |
| `url` | string | Full wiki URL |
| `status` | enum | `discovered`, `scored`, `ingested`, `failed` |
| `interest_score` | float | Agent-assigned value (0.0-1.0) |
| `link_depth` | int | Distance from portal page |
| `in_degree` | int | Number of pages linking to this |
| `out_degree` | int | Number of outgoing links |

### WikiChunk

Searchable text chunk with vector embedding.

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | `page_id:chunk_N` |
| `content` | string | Text content |
| `embedding` | float[] | 256-dim vector |
| `mdsplus_paths_mentioned` | string[] | Extracted MDSplus paths |
| `imas_paths_mentioned` | string[] | Extracted IMAS paths |

### Relationships

- `WikiPage` -[:FACILITY_ID]-> `Facility`
- `WikiPage` -[:HAS_CHUNK]-> `WikiChunk`
- `WikiChunk` -[:DOCUMENTS]-> `TreeNode`
- `WikiChunk` -[:MENTIONS_IMAS]-> `IMASPath`

## Semantic Search

Wiki content is indexed in the `wiki_chunk_embedding` vector index:

```python
from imas_codex.graph import GraphClient
from imas_codex.wiki.pipeline import get_embed_model

# Generate query embedding
embed_model = get_embed_model()
embedding = embed_model.get_text_embedding("Thomson scattering calibration")

# Search wiki content
with GraphClient() as gc:
    result = gc.query("""
        CALL db.index.vector.queryNodes("wiki_chunk_embedding", 5, $embedding)
        YIELD node, score
        MATCH (wp:WikiPage)-[:HAS_CHUNK]->(node)
        RETURN wp.title AS page, wp.url AS url, 
               node.content AS content, score
        ORDER BY score DESC
    """, embedding=embedding)
```

## Entity Extraction

The scraper extracts entities from wiki HTML using regex patterns:

| Entity | Pattern | Example |
|--------|---------|---------|
| MDSplus paths | `\\TREE::NODE:PATH` | `\RESULTS::THOMSON:NE` |
| IMAS paths | `ids/path/to/field` | `equilibrium/time_slice/psi` |
| Units | Common physics units | `eV`, `Tesla`, `m^-3` |
| COCOS | `COCOS N` patterns | `COCOS 11` |
| Sign conventions | Direction + quantity | `positive clockwise` |

## Artifact Support

The system also handles wiki artifacts (PDFs, presentations):

```bash
# Ingest only artifacts
uv run imas-codex wiki ingest tcv --type artifacts

# Control max file size
uv run imas-codex wiki ingest tcv --max-size-mb 10
```

Supported artifact types:
- PDF documents (full text extraction via pypdf)
- Presentations (deferred - requires OCR)
- Spreadsheets (deferred - requires specialized parsing)

## MCP Integration

Wiki content is accessible via the MCP `python()` REPL:

```python
# Search wiki content
from imas_codex.wiki.pipeline import get_embed_model
embed_model = get_embed_model()
embedding = embed_model.get_text_embedding("Thomson calibration")

hits = query("""
    CALL db.index.vector.queryNodes("wiki_chunk_embedding", 5, $embedding)
    YIELD node, score
    RETURN node.content, score
""", embedding=embedding)

# Ingest a specific page
from imas_codex.wiki.scraper import fetch_wiki_page
from imas_codex.wiki.pipeline import WikiIngestionPipeline

page = fetch_wiki_page("Thomson", facility="tcv")
pipeline = WikiIngestionPipeline(facility_id="tcv")
stats = await pipeline.ingest_page(page)
```

## Configuration

Wiki access is configured per facility. For EPFL:

- **Base URL**: `https://spcwiki.epfl.ch/wiki`
- **Portal**: `Portal:TCV`
- **Access**: SSH to EPFL host (no authentication from internal network)
- **SSL**: Self-signed cert, use `-k` flag

## Statistics

Query current wiki ingestion state:

```bash
uv run imas-codex wiki status tcv
```

Or via Cypher:

```cypher
MATCH (wp:WikiPage {facility_id: 'tcv'})
RETURN wp.status AS status, count(*) AS count
ORDER BY count DESC
```
