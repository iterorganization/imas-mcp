# Wiki Ingestion Strategy

> **Status**: Core pipeline complete, **CRITICAL BUG** blocking ingestion of scored pages
> Applicable to any facility with MediaWiki-based documentation.
> Last audit: 2026-01-14

## Current State (2026-01-14 Audit)

### Graph Statistics

| Status | Count | Notes |
|--------|-------|-------|
| `scored` | 2,240 | Awaiting ingestion |
| `ingested` | 606 | Successfully processed |
| `failed` | 127 | HTTP 404 errors (mostly subpages with `/`) |

**45 pages** have `status='scored'` AND `interest_score >= 0.5` (ingest threshold), but **are not being processed** by `wiki ingest`.

### Root Cause: WikiPage ID Mismatch Bug 🐛

**Critical Bug**: Discovery and ingestion create duplicate WikiPage nodes with different IDs for the same page.

```
Discovery Phase:
  Creates: WikiPage {id: "epfl:Portal:TCV", title: "Portal:TCV", status: "scored"}

Ingestion Phase:
  Creates: WikiPage {id: "epfl:TCV", title: "Portal:TCV", status: "ingested"}
  OR
  Creates: WikiPage {id: "epfl:Portal%3ATCV", title: "Portal:TCV", status: "ingested"}
```

**Impact**: 55 titles have duplicate nodes with different IDs and statuses. The `scored` pages are never updated because ingestion creates new nodes instead of matching existing ones.

**Cause**: 
1. `discovery.py` uses `page_name` from crawl link as ID: `id = f"{facility}:{page}"`
2. `pipeline.py` uses `page.page_name` from fetched URL: `page_id = f"{facility_id}:{page.page_name}"`
3. URL encoding differs: `Portal:TCV` vs `Portal%3ATCV` vs just `TCV` (redirect)

### 404 Failures (127 pages)

Pages with `/` in their titles fail because `urllib.parse.quote(page_name, safe="")` encodes `/` as `%2F`:
- `Thomson/DDJ` → `Thomson%2FDDJ` → 404
- Wiki expects literal `/` for subpages: `Thomson/DDJ`

**Fix**: Change to `urllib.parse.quote(page_name, safe="/")` to preserve slashes.

### Truly Pending Pages (No Duplicate)

After filtering out pages that have ingested duplicates, **~20 truly pending** pages remain:
- `Portal:TCV_Ports` (score=0.95)
- `Portal:TCV_PdJ` (score=0.90)
- `Service_Mécanique` (score=0.85)
- Various other Portal pages

## Overview

Many fusion facilities maintain internal wikis with authoritative documentation about diagnostics, signals, and conventions. This strategy describes how to:

1. **Discover** wiki pages via crawling or portal enumeration
2. **Evaluate** page value using ReAct agent assessment
3. **Ingest** high-value content with semantic chunking and embedding
4. **Link** wiki content to TreeNodes and IMAS paths

## Multi-Facility Applicability

| Facility | Wiki System | Access Method |
|----------|-------------|---------------|
| EPFL/TCV | MediaWiki | HTTPS from internal network |
| JET | Confluence | API or SSH tunnel |
| DIII-D | MediaWiki | TBD |
| ITER | Confluence | API |

The pipeline is designed to be wiki-system agnostic with facility-specific scrapers.

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

## Strategic Pivot: ReAct Agent Evaluation (2026-01-08)

### Problem with Current Approach

The existing `wiki discover` command uses a dumb breadth-first crawler that queues ALL internal wiki links. Observations:

- Many low-value pages discovered (e.g., "Missions 2025", meeting notes)
- No intelligence about page value before queuing
- Wastes ingestion time and graph space
- Dilutes semantic search quality

### New Architecture: Wiki Scout Agent

Replace the crawler with a ReAct agent that evaluates pages before queuing:

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Wiki Scout Agent                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Phase 1: Link Discovery (lightweight, unchanged)                    │
│  ────────────────────────────────────────────────                   │
│  - Crawl from portal page to find all internal links                │
│  - Return list of page names (NO WikiPage nodes yet)                │
│                                                                       │
│  Phase 2: Agent Evaluation (NEW - ReAct Agent)                       │
│  ────────────────────────────────────────────────                   │
│  For batches of discovered pages:                                    │
│                                                                       │
│  ┌─────────────────────────┐                                         │
│  │  Wiki Scout Agent       │                                         │
│  │  (LlamaIndex ReAct)     │                                         │
│  │                         │                                         │
│  │  Tools:                 │                                         │
│  │  - fetch_wiki_preview() │  Title + first 500 chars               │
│  │  - check_categories()   │  MediaWiki categories                  │
│  │  - update_wiki_page()   │  Queue or skip in graph                │
│  └─────────────────────────┘                                         │
│                                                                       │
│  Agent Decision Output (per page):                                   │
│  - interest_score: 0.0-1.0                                           │
│  - skip_reason: null | "administrative" | "event" | "stub" | ...    │
│  - recommended_status: "discovered" | "skipped"                      │
│                                                                       │
│  Phase 3: Ingestion (unchanged)                                      │
│  ────────────────────────────────                                   │
│  - Process pages with status='discovered'                           │
│  - Full fetch, chunk, embed, link                                   │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Skip Patterns the Agent Should Recognize

| Pattern | Example Pages | Skip Reason |
|---------|--------------|-------------|
| Events/Missions | "Missions 2025", "Workshop 2024" | `event_or_mission` |
| Personal pages | "User:Simon", "John's Notes" | `personal_page` |
| Administrative | "Help:Editing", "Template:Box" | `administrative` |
| Stubs | Pages with <100 chars content | `stub_or_empty` |
| Categories | "Category:Diagnostics" (meta pages) | `category_page` |

### High-Value Page Indicators

| Indicator | Weight | Example |
|-----------|--------|---------|
| MDSplus paths in preview | High | `\RESULTS::THOMSON:NE` |
| Signal tables | High | Tables with "Path", "Units" columns |
| Diagnostic names | Medium | "Thomson", "CXRS", "ECE" |
| COCOS/convention mentions | Medium | "COCOS 11", "positive clockwise" |
| IMAS path mentions | Medium | `equilibrium/time_slice` |

### Batch Evaluation Strategy

Process pages in batches of 30-50 for cost efficiency:

| Approach | Pages | LLM Calls | Est. Cost |
|----------|-------|-----------|-----------|
| Per-page agent | 500 | 500 | ~$5-10 |
| Batch 30 | 500 | 17 | ~$0.50-1.00 |
| Batch 50 | 500 | 10 | ~$0.30-0.60 |

### Implementation Plan

1. **Create wiki scout module** (`imas_codex/wiki/scout.py`)
   - `fetch_wiki_preview(page_name, facility)` - lightweight SSH fetch
   - `PageEvaluation` dataclass for agent output
   - `evaluate_wiki_pages(page_names, batch_size)` - main entry point

2. **Create scout prompt** (`imas_codex/agents/prompts/wiki-scout.md`)
   - System prompt for batch evaluation
   - JSON output schema
   - Examples of high/low value pages

3. **Update CLI** (`imas-codex wiki discover`)
   - Add `--evaluate` flag to run agent evaluation after discovery
   - Keep existing `--priority-only` for known high-value pages

4. **Update WikiPageStatus enum** (already simplified to: discovered, ingested, failed, stale)

### CLI Changes

```bash
# Current (unchanged for backward compat)
imas-codex wiki discover epfl

# New: with agent evaluation
imas-codex wiki discover epfl --evaluate

# Evaluate already-discovered pages (e.g., after schema change)
imas-codex wiki evaluate epfl [--limit 100] [--batch-size 30]
```

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `imas_codex/wiki/scout.py` | Create | Wiki scout agent with preview fetching |
| `imas_codex/agents/prompts/wiki-scout.md` | Create | System prompt for batch evaluation |
| `imas_codex/cli.py` | Modify | Add `wiki evaluate` command and `--evaluate` flag |
| `imas_codex/wiki/pipeline.py` | Minor | Remove duplicate return statement (line 348) |

### Cost and Time Estimates

For initial EPFL wiki (~500 pages):
- Discovery: ~10 min (existing crawler)
- Evaluation: ~10 min (17 batch calls × 30s each)
- Total cost: ~$0.50-1.00

For ongoing maintenance:
- Re-evaluate stale pages weekly
- New pages discovered incrementally

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

### Phase 1: Core Infrastructure ✅
- [x] Schema with `WikiPage` and `WikiChunk` classes
- [x] Scraper module with facility-agnostic design
- [x] LlamaIndex chunking and embedding pipeline
- [x] CLI commands (`wiki discover/crawl/score/ingest/status`)
- [x] Vector index for semantic search
- [x] Three-phase pipeline (crawl → score → ingest)
- [x] WikiArtifact support (PDFs, presentations)
- [x] ReAct agent scoring with graph metrics

### Phase 1.5: Bug Fixes 🔧 ✅ (COMPLETED 2026-01-14)
- [x] **ID normalization**: Added `canonical_page_id()` function used by discovery and ingestion
- [x] **URL encoding fix**: Changed to `safe="/"` to preserve subpage slashes
- [x] **Duplicate cleanup**: Merged 58 duplicate WikiPage nodes by title
- [x] **Ingest updates**: Pipeline now matches by (facility_id, title) instead of creating new nodes

### Phase 2: Quality Filtering ✅ (Implemented but needs fixes)
- [x] Wiki scout module with preview fetching (`scout.py`)
- [x] ReAct agent for batch page evaluation (`discovery.py`)
- [x] Skip patterns (administrative, stubs, events)
- [x] High-value indicators (signal tables, paths)
- [ ] Fix agent to properly handle special characters in page names

### Phase 3: MCP Integration 📡 ✅ (Use python() REPL - No New Tools)

**Decision (2026-01-14)**: Use existing `python()` REPL tool instead of adding wiki-specific MCP tools.

The MCP server's `python()` REPL provides everything needed:
- `query()` for Neo4j operations
- `ssh()` for remote wiki fetching
- `semantic_search()` with `wiki_chunk_embedding` index
- Full Python import capability for wiki modules

**Example Usage in python() REPL:**
```python
# Search wiki content
hits = semantic_search("Thomson calibration", "wiki_chunk_embedding", k=5)

# Ingest a page interactively
from imas_codex.wiki.scraper import fetch_wiki_page
from imas_codex.wiki.pipeline import WikiIngestionPipeline
page = fetch_wiki_page("Thomson", facility="epfl")
pipeline = WikiIngestionPipeline(facility_id="epfl")
stats = await pipeline.ingest_page(page)
```

**No new MCP tools needed.** This avoids tool proliferation and leverages the REPL's flexibility.

### Phase 4: Multi-Facility ⬜
- [ ] Abstract scraper interface
- [ ] Confluence adapter (for JET, ITER)
- [ ] Per-facility portal configuration

## Immediate Fix Plan (Priority Order)

### Fix 1: ID Normalization (scraper.py + pipeline.py)

```python
# imas_codex/wiki/scraper.py - Add canonical ID function
def canonical_page_id(page_name: str, facility_id: str) -> str:
    """Generate canonical WikiPage ID.
    
    Uses decoded page name with colons preserved (not URL-encoded).
    Example: "Portal:TCV" → "epfl:Portal:TCV"
    """
    import urllib.parse
    decoded = urllib.parse.unquote(page_name)
    return f"{facility_id}:{decoded}"


# imas_codex/wiki/pipeline.py - Use canonical ID
def ingest_page(self, page: WikiPage) -> PageIngestionStats:
    from .scraper import canonical_page_id
    page_id = canonical_page_id(page.page_name, self.facility_id)
    # ... rest of method
```

### Fix 2: URL Encoding for Subpages (scraper.py)

```diff
# imas_codex/wiki/scraper.py
def fetch_wiki_page(page_name: str, ...):
-    encoded_page_name = urllib.parse.quote(page_name, safe="")
+    encoded_page_name = urllib.parse.quote(page_name, safe="/")
```

### Fix 3: Duplicate Cleanup (one-time Cypher)

```cypher
-- Find duplicates and merge to canonical ID
MATCH (w:WikiPage {facility_id: 'epfl'})
WITH w.title AS title, collect(w) AS pages, count(*) AS cnt
WHERE cnt > 1
WITH title, pages, [p IN pages WHERE p.status = 'ingested'][0] AS keeper
WHERE keeper IS NOT NULL
UNWIND [p IN pages WHERE p <> keeper] AS dup
// Move relationships from dup to keeper
MATCH (dup)-[r:HAS_CHUNK]->(c)
MERGE (keeper)-[:HAS_CHUNK]->(c)
DELETE r
// Delete duplicate
DETACH DELETE dup
```

### Fix 4: Ingest Updates Existing Nodes (pipeline.py)

```diff
# imas_codex/wiki/pipeline.py
-    gc.query("""
-        MERGE (p:WikiPage {id: $id})
-        SET p.url = $url, ...
+    gc.query("""
+        MATCH (p:WikiPage {facility_id: $facility_id, title: $title})
+        SET p.id = $id,  -- Normalize to canonical ID
+            p.url = $url,
+            p.status = 'ingested', ...
```

## MCP Python REPL Integration Strategy

With the Codex MCP server providing a persistent `python()` REPL, wiki operations can be more interactive:

### Current Approach (CLI-driven)
```bash
imas-codex wiki discover epfl   # Separate process
imas-codex wiki score epfl      # Separate agent process  
imas-codex wiki ingest epfl     # Separate process
```

### Proposed Approach (MCP-driven)
```python
# In MCP python() REPL - persistent state, immediate feedback

# 1. Check current wiki state
result = query("""
    MATCH (w:WikiPage {facility_id: 'epfl'})
    RETURN w.status, count(*) AS count
    ORDER BY count DESC
""")
print(result)

# 2. Fetch and preview a specific page
from imas_codex.wiki.scraper import fetch_wiki_page
page = fetch_wiki_page("Thomson", facility="epfl")
print(f"Title: {page.title}")
print(f"MDSplus paths: {page.mdsplus_paths[:5]}")

# 3. Interactive ingestion with preview
from imas_codex.wiki import WikiIngestionPipeline
pipeline = WikiIngestionPipeline(facility_id="epfl")
stats = await pipeline.ingest_page(page)
print(f"Created {stats['chunks']} chunks")

# 4. Semantic search across wiki content
hits = semantic_search("Thomson scattering calibration", "wiki_chunk_embedding", k=5)
for h in hits:
    print(f"{h['score']:.3f}: {h['content'][:100]}")
```

### Benefits of MCP Integration
1. **Persistent state**: Variables survive between calls
2. **Interactive debugging**: Inspect page content before ingesting
3. **Immediate feedback**: See graph changes in real-time
4. **Unified interface**: Same `query()` for wiki and other graph data
5. **No subprocess overhead**: Faster iteration

**No new MCP tools needed** - the existing `python()` REPL provides all functionality:
- `query()` for graph operations
- `ssh()` for remote wiki fetching
- `semantic_search()` with `wiki_chunk_embedding` index
- Direct module imports for wiki pipeline

## Notes

- Wiki content should be marked with `enrichment_source: "wiki"`
- Include `wiki_source` URL on TreeNodes enriched from wiki
- Re-run enrichment for high-value nodes (equilibrium, profiles) after wiki ingestion
- Consider rate limiting to avoid overloading the wiki server
- Neo4j now runs as persistent systemd user service: `systemctl --user start neo4j`

## Audit History

### 2026-01-14: Comprehensive Audit
- **Found**: ID mismatch bug causing duplicate WikiPage nodes
- **Found**: 127 failed pages due to `/` encoding in URLs
- **Found**: 45 scored pages stuck, not being ingested
- **Analyzed**: MCP `python()` REPL as alternative to CLI commands
- **Proposed**: 4 immediate fixes + MCP integration strategy

### 2026-01-08: Strategic Pivot
- Replaced dumb BFS crawler with ReAct agent evaluation
- Added three-phase pipeline (crawl → score → ingest)
- Added WikiArtifact support for PDFs

### 2026-01-07: Initial Implementation
- Verified wiki access from EPFL network
- Implemented scraper, pipeline, CLI commands
- Created WikiPage and WikiChunk schema
