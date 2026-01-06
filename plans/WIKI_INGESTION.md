# Wiki Ingestion Strategy

## Problem

The EPFL TCV wiki contains authoritative documentation about diagnostics, signals, and conventions that would significantly improve TreeNode enrichment. Currently the agent relies on:
1. Code examples (good for usage patterns)
2. Graph context (good for structure)
3. SSH queries (slow, limited metadata)

Wiki content would provide:
- Official signal descriptions
- Sign conventions and COCOS definitions
- Diagnostic specifications
- Data quality notes
- Historical context

## Proposed Approach

### Phase 1: Wiki Scraping (One-time)

Create a scraper that extracts structured content from the TCV wiki:

```python
# imas_codex/wiki/scraper.py
@dataclass
class WikiPage:
    url: str
    title: str
    content: str  # Cleaned markdown
    sections: dict[str, str]  # Section name -> content
    related_paths: list[str]  # MDSplus paths mentioned
    last_updated: datetime

def scrape_tcv_wiki(base_url: str, pages: list[str]) -> list[WikiPage]:
    """Scrape specified wiki pages and extract structured content."""
```

**Key pages to target:**
- Diagnostic pages (CXRS, Thomson, bolometry, magnetics)
- Signal convention pages (COCOS, sign conventions)
- Code documentation (LIUQE, ASTRA, analysis routines)
- Tree structure documentation

### Phase 2: Content Chunking & Embedding

Similar to code ingestion, chunk wiki content and embed:

```python
# imas_codex/wiki/pipeline.py
@dataclass
class WikiChunk:
    id: str
    page_url: str
    section: str
    content: str
    embedding: list[float]
    related_paths: list[str]  # Extracted MDSplus paths
    physics_domains: list[str]  # Detected domains
```

Store in Neo4j with vector index for semantic search.

### Phase 3: Graph Integration

Create relationships:
```cypher
// WikiChunk documents a TreeNode
(w:WikiChunk)-[:DOCUMENTS]->(t:TreeNode)

// WikiChunk mentions IMAS paths
(w:WikiChunk)-[:MENTIONS_IMAS]->(i:IMASPath)

// WikiChunk belongs to wiki page
(p:WikiPage)-[:HAS_CHUNK]->(w:WikiChunk)
```

### Phase 4: Agent Tool

Add wiki search tool to enrichment agent:

```python
def _search_wiki(query: str, limit: int = 5) -> str:
    """Semantic search over wiki documentation.
    
    Use this for official signal descriptions, conventions, and specs.
    """
```

## Implementation Steps

1. **Identify wiki structure** - SSH to EPFL, explore wiki format
2. **Build scraper** - Handle authentication, extract content
3. **Create pipeline** - Chunk, embed, ingest to graph
4. **Add agent tool** - Enable wiki search in enrichment
5. **Run enrichment** - Re-enrich nodes with wiki context

## Authentication Considerations

The TCV wiki may require EPFL authentication:
- Store credentials in private infrastructure file
- Use SSH tunnel if needed
- Consider caching scraped content locally

## Alternative: RAG over PDF Documentation

If wiki scraping is problematic, consider:
1. Export wiki pages to PDF/HTML
2. Transfer to local machine
3. Process with standard document loaders
4. Embed and ingest

## Priority

Medium - The code examples already provide good context, but wiki would add:
- Sign conventions (critical for IMAS)
- Official descriptions (higher authority)
- Diagnostic specifications

## Notes

- Wiki content should be marked with `enrichment_source: "wiki"`
- Include `wiki_source` URL on TreeNodes enriched from wiki
- Re-run enrichment for high-value nodes (equilibrium, profiles) after wiki ingestion
