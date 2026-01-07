# Wiki Ingestion Strategy

## Problem

The EPFL TCV wiki (https://spcwiki.epfl.ch/wiki/Main_Page) contains authoritative documentation about diagnostics, signals, and conventions that would significantly improve TreeNode enrichment. Currently the agent relies on:

1. Code examples (good for usage patterns)
2. Graph context (good for structure)
3. SSH queries (slow, limited metadata)

Wiki content would provide:

- Official signal descriptions
- Sign conventions and COCOS definitions
- Diagnostic specifications
- Data quality notes
- Historical context

## Wiki Access (Verified 2026-01-07)

### Access Method

**No authentication required from EPFL network.** The wiki is directly accessible via HTTPS from any EPFL host.

```bash
# Access wiki from EPFL host (use -k to skip self-signed cert verification)
ssh epfl "curl -sk 'https://spcwiki.epfl.ch/wiki/Main_Page' | head -100"

# List all wiki links on a page
ssh epfl "curl -sk 'https://spcwiki.epfl.ch/wiki/Diagnostics' | grep -oP 'href=\"/wiki/[^\"]+\"' | sort -u"

# Get page content
ssh epfl "curl -sk 'https://spcwiki.epfl.ch/wiki/Thomson'"
```

### External Access (requires Tequila)

From outside EPFL network, the wiki redirects to Tequila SSO. Options:
- **SSH tunnel**: `ssh -L 8080:spcwiki.epfl.ch:443 epfl` then access `https://localhost:8080`
- **Run scraper via SSH**: Execute Python scraper on EPFL host, pipe results back

### Recommended Approach

Run the scraper directly on EPFL host via SSH and stream JSON results back:

```bash
ssh epfl 'python3 -c "
import json, urllib.request, ssl
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
url = \"https://spcwiki.epfl.ch/wiki/Thomson\"
html = urllib.request.urlopen(url, context=ctx).read().decode()
print(json.dumps({\"url\": url, \"size\": len(html)}))
"'
```

### Phase 1: Schema Extension

Add wiki-related classes to `imas_codex/schemas/facility.yaml`:

```yaml
WikiPage:
  description: >-
    A wiki page from facility documentation (e.g., TCV wiki).
    Used for authoritative signal descriptions and conventions.
  class_uri: facility:WikiPage
  attributes:
    id:
      identifier: true
      description: Composite key as facility:page_path (e.g., epfl:Diagnostics/Thomson)
      required: true
    facility_id:
      description: Parent facility ID
      required: true
      range: Facility
    url:
      description: Original wiki URL
      required: true
    title:
      description: Page title
      required: true
    content_hash:
      description: Hash of content for change detection
    last_scraped:
      description: When this page was last scraped
      range: datetime
    last_modified:
      description: When wiki page was last modified (if available)
      range: datetime

WikiChunk:
  description: >-
    A searchable chunk from a wiki page with vector embeddings.
    Links to TreeNodes it documents.
  class_uri: facility:WikiChunk
  attributes:
    id:
      identifier: true
      description: Unique chunk identifier
      required: true
    wiki_page_id:
      description: Parent WikiPage
      required: true
      range: WikiPage
    section:
      description: Section heading this chunk belongs to
    content:
      description: The actual text content
      required: true
    embedding:
      description: Vector embedding for semantic search
      multivalued: true
      range: float
    mdsplus_paths_mentioned:
      description: MDSplus paths mentioned in this chunk
      multivalued: true
    imas_paths_mentioned:
      description: IMAS paths mentioned in this chunk
      multivalued: true
```

### Phase 2: Scraper Implementation

Create a scraper that runs via SSH on EPFL host:

```python
# imas_codex/wiki/scraper.py
"""Wiki scraper for TCV documentation via SSH."""

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field


@dataclass
class WikiPage:
    """Scraped wiki page with structured content."""
    
    url: str
    title: str
    content_html: str
    content_text: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    mdsplus_paths: list[str] = field(default_factory=list)
    
    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content_html.encode()).hexdigest()[:16]


MDSPLUS_PATH_PATTERN = re.compile(r'\\\\?[A-Z_]+::[A-Z_:]+', re.IGNORECASE)


def fetch_wiki_page(page_name: str, facility: str = "epfl") -> WikiPage:
    """Fetch a wiki page via SSH.
    
    Args:
        page_name: Wiki page name (e.g., "Thomson", "Diagnostics")
        facility: SSH host alias
        
    Returns:
        WikiPage with HTML content and extracted MDSplus paths
    """
    url = f"https://spcwiki.epfl.ch/wiki/{page_name}"
    
    # Fetch via SSH with SSL verification disabled
    cmd = f'''python3 -c "
import urllib.request, ssl, json
ctx = ssl.create_default_context()
ctx.check_hostname = False  
ctx.verify_mode = ssl.CERT_NONE
html = urllib.request.urlopen('{url}', context=ctx).read().decode('utf-8', errors='ignore')
print(html)
"'''
    
    result = subprocess.run(
        ["ssh", facility, cmd],
        capture_output=True,
        text=True,
        timeout=60,
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"SSH failed: {result.stderr}")
    
    html = result.stdout
    
    # Extract title (simple regex)
    title_match = re.search(r'<title>([^<]+)</title>', html)
    title = title_match.group(1).replace(" - SPCwiki", "") if title_match else page_name
    
    # Extract MDSplus paths
    mdsplus_paths = list(set(MDSPLUS_PATH_PATTERN.findall(html)))
    
    return WikiPage(
        url=url,
        title=title,
        content_html=html,
        mdsplus_paths=mdsplus_paths,
    )


def discover_wiki_pages(start_page: str = "Main_Page", facility: str = "epfl") -> list[str]:
    """Discover wiki pages by crawling links.
    
    Args:
        start_page: Page to start crawling from
        facility: SSH host alias
        
    Returns:
        List of discovered page names
    """
    cmd = f'''curl -sk 'https://spcwiki.epfl.ch/wiki/{start_page}' | grep -oP 'href="/wiki/[^"]+\"' | sed 's|href="/wiki/||;s|"||g' | sort -u'''
    
    result = subprocess.run(
        ["ssh", facility, cmd],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    pages = [p for p in result.stdout.strip().split('\n') if p and ':' not in p]
    return pages
```

### Phase 3: LlamaIndex Integration for Chunking & Embedding

```python
# imas_codex/wiki/pipeline.py
"""Wiki ingestion pipeline using LlamaIndex."""

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from imas_codex.graph import GraphClient
from imas_codex.wiki.scraper import WikiPage


class WikiIngestionPipeline:
    """Ingest wiki pages into the knowledge graph."""
    
    def __init__(
        self,
        facility_id: str = "epfl",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.facility_id = facility_id
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
    
    def ingest_page(self, page: WikiPage) -> int:
        """Ingest a single wiki page into the graph."""
        # Convert HTML to text for chunking
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(page.content_html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        
        # Create LlamaIndex document
        doc = Document(
            text=text,
            metadata={
                "url": page.url,
                "title": page.title,
                "facility_id": self.facility_id,
            },
        )
        
        # Split into chunks
        nodes = self.splitter.get_nodes_from_documents([doc])
        
        # Generate embeddings
        for node in nodes:
            embedding = self.embed_model.get_text_embedding(node.text)
            node.embedding = embedding
        
        # Store in Neo4j
        with GraphClient() as gc:
            # Create WikiPage node
            page_id = f"{self.facility_id}:{page.url.split('/wiki/')[-1]}"
            gc.query("""
                MERGE (p:WikiPage {id: $id})
                SET p.url = $url,
                    p.title = $title,
                    p.facility_id = $facility_id,
                    p.content_hash = $hash,
                    p.last_scraped = datetime()
                WITH p
                MATCH (f:Facility {id: $facility_id})
                MERGE (p)-[:FACILITY_ID]->(f)
            """, id=page_id, url=page.url, title=page.title,
                facility_id=self.facility_id, hash=page.content_hash)
            
            # Create WikiChunk nodes with embeddings
            for i, node in enumerate(nodes):
                chunk_id = f"{page_id}:chunk_{i}"
                gc.query("""
                    MERGE (c:WikiChunk {id: $id})
                    SET c.content = $content,
                        c.embedding = $embedding,
                        c.mdsplus_paths_mentioned = $mdsplus_paths
                    WITH c
                    MATCH (p:WikiPage {id: $page_id})
                    MERGE (p)-[:HAS_CHUNK]->(c)
                """, id=chunk_id, content=node.text,
                    embedding=node.embedding,
                    mdsplus_paths=page.mdsplus_paths,
                    page_id=page_id)
                
                # Link to mentioned TreeNodes
                for path in page.mdsplus_paths:
                    gc.query("""
                        MATCH (c:WikiChunk {id: $chunk_id})
                        MATCH (t:TreeNode) WHERE t.path CONTAINS $path
                        MERGE (c)-[:DOCUMENTS]->(t)
                    """, chunk_id=chunk_id, path=path)
        
        return len(nodes)
```

### Phase 4: Agent Tool for Wiki Search

Add to `imas_codex/agents/tools.py`:

```python
def _search_wiki(query: str, limit: int = 5) -> str:
    """
    Search wiki documentation using semantic vector search.

    Use this for official signal descriptions, sign conventions, and specs.
    Wiki content is authoritative documentation from facility experts.

    Args:
        query: Natural language query (e.g., 'Thomson scattering calibration')
        limit: Maximum number of results (default: 5)

    Returns:
        Matching wiki chunks with page titles, sections, and content
    """
    try:
        with GraphClient() as gc:
            # Vector similarity search on WikiChunk embeddings
            result = gc.query("""
                CALL db.index.vector.queryNodes('wiki_chunk_embedding', $limit, $embedding)
                YIELD node, score
                MATCH (p:WikiPage)-[:HAS_CHUNK]->(node)
                RETURN p.title AS page_title, p.url AS url, 
                       node.section AS section, node.content AS content,
                       score
                ORDER BY score DESC
            """, limit=limit, embedding=_get_query_embedding(query))
            
            if not result:
                return f"No wiki content found for '{query}'"
            
            output = []
            for r in result:
                output.append(f"=== {r['page_title']} ({r['section']}) ===")
                output.append(f"URL: {r['url']}")
                content = r['content'][:500] + "..." if len(r['content']) > 500 else r['content']
                output.append(content)
                output.append(f"Score: {r['score']:.3f}\n")
            
            return "\n".join(output)
    except Exception as e:
        return f"Wiki search error: {e}"
```

### Phase 5: CLI Commands

```bash
# Discover wiki pages
uv run imas-codex wiki discover epfl

# Scrape and ingest wiki content
uv run imas-codex wiki ingest epfl --max-pages 100

# Check ingestion status
uv run imas-codex wiki status epfl
```

## Wiki Structure Discovery (2026-01-07)

**Access confirmed**: Wiki accessible from EPFL hosts via HTTPS with `-k` flag (self-signed cert).
No Tequila authentication required from internal network.

### Key Portals

| Portal | URL | Content |
|--------|-----|---------|
| TCV | `/wiki/Portal:TCV` | Main TCV documentation hub |
| Science | `/wiki/Portal:Science` | Physics and analysis |
| Diagnostics | `/wiki/Diagnostics` | All diagnostic systems |

### Diagnostic Categories (from wiki)

| Category | Pages |
|----------|-------|
| **Electron diagnostics** | Thomson, ECE, HXRS, FIR, DBS, Edge_thomson, Vertical_ECE, XTOMO, RADCAM, PMTX |
| **Ion diagnostics** | CXRS, CNPA, NPA, DNBI, FIDA, INPA, Ion_Temperature_Nodes |
| **Magnetic diagnostics** | Magnetics, Flux_loops, Magnetic_probes, Coil_currents, IMSE, DML |
| **Edge diagnostics** | (in category) |
| **X-ray diagnostics** | (in category) |
| **FIR diagnostics** | 1mm_interferometer, FIR |

### Key Pages with Signal Documentation

These pages contain **structured MDS node tables** with paths, units, and descriptions:

| Page | Content Type | Value |
|------|--------------|-------|
| `MDS` | MDSplus overview, MATLAB commands | High |
| `MDS_commands` | mdsconnect, mdsvalue, mdsput examples | High |
| `Thomson` | TE/NE profiles, spatial resolution, MDS nodes | Very High |
| `Ion_Temperature_Nodes` | `\RESULTS::TI:TI0` tree, units, confidence | Very High |
| `CXRS` | Ti/Vi profiles, rotation measurements | High |
| `Magnetics` | Flux loops, probes, calibration | High |
| `Personal_RESULTS_trees` | Tree structure conventions | Medium |
| `TCV_all_diagnostics` | Complete diagnostic list | Medium |
| `TCV_data_acquisition` | DAQ system details | Medium |

### Analysis Code Documentation

| Page | Code | Value |
|------|------|-------|
| `LIUQE` (if exists) | Equilibrium reconstruction | Very High |
| `ASTRA` (if exists) | Transport code | High |
| `NBI` | Neutral beam injection | High |
| `DNBI` | Diagnostic NBI | High |

### Sample Content Quality

**Ion_Temperature_Nodes page** contains:
```
MDS path                      | Tree      | Units | Description
\RESULTS::TI:TI0              | TCV_SHOT  | eV    | On-axis Ion Temperature
\RESULTS::TI:TI0:ERROR_BAR    | TCV_SHOT  | eV    | errorbars (dTo)
\RESULTS::TI:TI0:CONF_ID      | TCV_SHOT  |       | confidence ID
\RESULTS::TI:TI0:NPA:FOO      | TCV_SHOT  | eV    | maximal Ti along NPA view-line
```

**Thomson page** contains:
- 117 spectrometer channels documented
- Spatial resolution specifications
- R,Z coordinates of measurements
- Data analysis tools (`ts_getdata.m`, `ts_autofits.m`)
- Profile fitting details (rho, boundary conditions)

## Priority Pages for Initial Ingestion

Phase 1 - High Value (50 pages):
1. All diagnostic category pages
2. Ion_Temperature_Nodes, Thomson, CXRS, ECE, Magnetics
3. MDS, MDS_commands
4. Analysis code pages

Phase 2 - Medium Value (100 pages):
1. Individual diagnostic hardware pages
2. Calibration documentation
3. TCV port/configuration pages

## Implementation Checklist

1. [x] ~~Add dependencies~~ - `llama-index-readers-web beautifulsoup4 html2text` installed
2. [x] ~~Verify wiki access~~ - Confirmed accessible from EPFL without auth
3. [x] ~~Document wiki structure~~ - Key pages and categories identified
4. [ ] Extend schema with `WikiPage` and `WikiChunk` classes
5. [ ] Run `uv run build-models --force` to regenerate Pydantic models  
6. [ ] Implement scraper module `imas_codex/wiki/scraper.py`
7. [ ] Implement pipeline `imas_codex/wiki/pipeline.py`
8. [ ] Create Neo4j vector index: `CREATE VECTOR INDEX wiki_chunk_embedding FOR (c:WikiChunk) ON (c.embedding)`
9. [ ] Add `_search_wiki` tool to agents
10. [ ] Add CLI commands for wiki management
11. [ ] Initial ingestion of priority pages

## Next Steps

1. **Implement scraper** - Create `imas_codex/wiki/` module with SSH-based scraper
2. **Extend schema** - Add `WikiPage` and `WikiChunk` to `facility.yaml`
3. **Build pipeline** - Chunk, embed, and ingest to Neo4j
4. **Add agent tool** - Enable wiki search in enrichment workflow

## Notes

- Wiki content should be marked with `enrichment_source: "wiki"`
- Include `wiki_source` URL on TreeNodes enriched from wiki
- Re-run enrichment for high-value nodes (equilibrium, profiles) after wiki ingestion
- Consider rate limiting to avoid overloading the wiki server
