"""Wiki ingestion pipeline using LlamaIndex.

Processes wiki pages and artifacts through a deterministic pipeline:
1. Crawl: Discover pages/artifacts via link traversal (creates nodes with status='crawled')
2. Score: Agent evaluates interest_score (sets status='scored' or 'skipped')
3. Ingest: Fetch content, chunk, embed, and link to graph (sets status='ingested')

The pipeline is graph-driven and fully deterministic (no LLM calls during ingestion).
Entity extraction uses regex patterns for MDSplus paths, IMAS paths, units, conventions.

Example:
    # Step 1: Crawl wiki (link extraction, no content fetch)
    # imas-codex wiki crawl epfl

    # Step 2: Score pages with LLM agent
    # imas-codex wiki score epfl

    # Step 3: Ingest high-score pages (deterministic)
    pipeline = WikiIngestionPipeline(facility_id="epfl")
    stats = await pipeline.ingest_from_graph(min_interest_score=0.7)
    print(f"Created {stats['chunks']} chunks")
"""

import hashlib
import logging
import re
from collections.abc import Callable
from html.parser import HTMLParser
from typing import TypedDict

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from imas_codex.graph import GraphClient
from imas_codex.settings import get_imas_embedding_model

from .progress import WikiProgressMonitor, set_current_monitor
from .scraper import WikiPage, fetch_wiki_page

logger = logging.getLogger(__name__)

# Batch size for UNWIND operations
BATCH_SIZE = 50

# Progress callback type: (current, total, message) -> None
ProgressCallback = Callable[[int, int, str], None]


class PageIngestionStats(TypedDict):
    """Statistics from ingesting a single wiki page."""

    chunks: int
    tree_nodes_linked: int
    imas_paths_linked: int
    conventions: int
    units: int
    content_preview: str
    mdsplus_paths: list[str]


# =============================================================================
# Graph Query Functions
# =============================================================================


def get_pending_wiki_pages(
    facility_id: str,
    limit: int = 100,
    min_interest_score: float = 0.0,
) -> list[dict]:
    """Get wiki pages pending ingestion from the graph.

    Returns WikiPage nodes with status='scored' (passed agent evaluation),
    sorted by interest_score descending.

    Args:
        facility_id: Facility ID
        limit: Maximum pages to return
        min_interest_score: Minimum interest score threshold

    Returns:
        List of page dicts with id, url, title, interest_score
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility_id, status: 'scored'})
            WHERE wp.interest_score >= $min_score
            RETURN wp.id AS id, wp.url AS url, wp.title AS title,
                   wp.interest_score AS interest_score
            ORDER BY wp.interest_score DESC, wp.discovered_at ASC
            LIMIT $limit
            """,
            facility_id=facility_id,
            min_score=min_interest_score,
            limit=limit,
        )
        return [dict(r) for r in result] if result else []


def get_wiki_queue_stats(facility_id: str) -> dict:
    """Get wiki page queue statistics by status.

    Args:
        facility_id: Facility ID

    Returns:
        Dict with status counts for pages and artifacts
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility_id})
            RETURN wp.status AS status, count(*) AS count
            ORDER BY count DESC
            """,
            facility_id=facility_id,
        )

        # Initialize with zero counts for all expected statuses
        stats = {
            "crawled": 0,
            "scored": 0,
            "skipped": 0,
            "ingested": 0,
            "failed": 0,
            "stale": 0,
            "total": 0,
        }

        if result:
            for r in result:
                status = r["status"]
                count = r["count"]
                if status in stats:
                    stats[status] = count
                stats["total"] += count

        return stats


def get_pending_wiki_artifacts(
    facility_id: str,
    limit: int = 100,
    min_interest_score: float = 0.0,
) -> list[dict]:
    """Get wiki artifacts pending ingestion from the graph.

    Returns WikiArtifact nodes with status='scored' (passed agent evaluation),
    sorted by interest_score descending.

    Args:
        facility_id: Facility ID
        limit: Maximum artifacts to return
        min_interest_score: Minimum interest score threshold

    Returns:
        List of artifact dicts with id, url, filename, artifact_type, interest_score
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wa:WikiArtifact {facility_id: $facility_id, status: 'scored'})
            WHERE wa.interest_score >= $min_score
            RETURN wa.id AS id, wa.url AS url, wa.filename AS filename,
                   wa.artifact_type AS artifact_type,
                   wa.interest_score AS interest_score
            ORDER BY wa.interest_score DESC
            LIMIT $limit
            """,
            facility_id=facility_id,
            min_score=min_interest_score,
            limit=limit,
        )
        return [dict(r) for r in result] if result else []


def mark_wiki_page_status(
    page_id: str,
    status: str,
    error: str | None = None,
) -> None:
    """Update the status of a wiki page.

    Args:
        page_id: WikiPage ID
        status: New status (crawled, scored, skipped, ingested, failed, stale)
        error: Error message if status is 'failed'
    """
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (wp:WikiPage {id: $id})
            SET wp.status = $status,
                wp.error = $error
            """,
            id=page_id,
            status=status,
            error=error,
        )


def persist_chunks_batch(chunks: list[dict]) -> int:
    """Persist a batch of WikiChunk nodes using UNWIND for efficiency.

    Args:
        chunks: List of chunk dicts with required fields:
            - id, wiki_page_id, facility_id, content, embedding
            - mdsplus_paths, imas_paths, units, conventions (optional)

    Returns:
        Number of chunks persisted
    """
    if not chunks:
        return 0

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $chunks AS chunk
            MERGE (c:WikiChunk {id: chunk.id})
            SET c.wiki_page_id = chunk.wiki_page_id,
                c.facility_id = chunk.facility_id,
                c.content = chunk.content,
                c.embedding = chunk.embedding,
                c.mdsplus_paths_mentioned = chunk.mdsplus_paths,
                c.imas_paths_mentioned = chunk.imas_paths,
                c.units_mentioned = chunk.units,
                c.conventions_mentioned = chunk.conventions
            WITH c, chunk
            MATCH (p:WikiPage {id: chunk.wiki_page_id})
            MERGE (p)-[:HAS_CHUNK]->(c)
            """,
            chunks=chunks,
        )
        return len(chunks)


def link_chunks_to_entities(facility_id: str) -> dict[str, int]:
    """Create DOCUMENTS and MENTIONS relationships from chunk metadata.

    Uses batched queries to link WikiChunks to TreeNodes and IMASPaths
    based on the paths stored in chunk properties during ingestion.

    Args:
        facility_id: Facility to process

    Returns:
        Dict with counts: {tree_nodes_linked, imas_paths_linked}
    """
    stats = {"tree_nodes_linked": 0, "imas_paths_linked": 0}

    with GraphClient() as gc:
        # Link to TreeNodes via MDSplus paths
        result = gc.query(
            """
            MATCH (c:WikiChunk {facility_id: $facility_id})
            WHERE c.mdsplus_paths_mentioned IS NOT NULL
            UNWIND c.mdsplus_paths_mentioned AS mds_path
            MATCH (t:TreeNode)
            WHERE t.path = mds_path
               OR t.path ENDS WITH replace(mds_path, '\\\\', '')
               OR t.canonical_path = toUpper(mds_path)
            MERGE (c)-[:DOCUMENTS]->(t)
            RETURN count(*) AS linked
            """,
            facility_id=facility_id,
        )
        if result:
            stats["tree_nodes_linked"] = result[0]["linked"]

        # Link to IMASPaths
        result = gc.query(
            """
            MATCH (c:WikiChunk {facility_id: $facility_id})
            WHERE c.imas_paths_mentioned IS NOT NULL
            UNWIND c.imas_paths_mentioned AS imas_path
            MATCH (ip:IMASPath)
            WHERE ip.full_path CONTAINS imas_path
            MERGE (c)-[:MENTIONS_IMAS]->(ip)
            RETURN count(*) AS linked
            """,
            facility_id=facility_id,
        )
        if result:
            stats["imas_paths_linked"] = result[0]["linked"]

    return stats


class MediaWikiExtractor(HTMLParser):
    """MediaWiki-aware HTML to text converter.

    Targets the main content area (mw-parser-output) and skips
    navigation, sidebar, and footer elements.
    """

    # Tags that should be completely skipped
    SKIP_TAGS = {"script", "style", "nav", "footer", "header", "noscript"}

    # CSS classes that indicate navigation/sidebar content to skip
    SKIP_CLASSES = {
        "mw-navigation",
        "mw-sidebar",
        "mw-footer",
        "printfooter",
        "catlinks",
        "mw-editsection",
        "noprint",
        "toc",
        "navbox",
        "metadata",
        "mw-jump-link",
        "portal",
        "portlet",
        "p-personal",
        "p-navigation",
        "p-search",
        "p-tb",
        "p-coll-print_export",
    }

    # CSS classes that indicate main content
    CONTENT_CLASSES = {"mw-parser-output", "mw-body-content", "mw-content-text"}

    def __init__(self):
        super().__init__()
        self.text_parts: list[str] = []
        self.current_section: str = ""
        self.sections: dict[str, str] = {}
        self._skip_depth = 0
        self._content_depth = 0
        self._in_heading = False
        self._heading_text = ""
        self._in_table = False
        self._table_row: list[str] = []
        self._in_pre = False
        self._skipping_div = 0  # Track divs we're skipping

    def _get_class(self, attrs: list[tuple[str, str | None]]) -> set[str]:
        """Extract CSS classes from tag attributes."""
        for name, value in attrs:
            if name == "class" and value:
                return set(value.split())
        return set()

    def _get_id(self, attrs: list[tuple[str, str | None]]) -> str | None:
        """Extract id from tag attributes."""
        for name, value in attrs:
            if name == "id":
                return value
        return None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = self._get_class(attrs)
        element_id = self._get_id(attrs)

        # Check if this element should be skipped
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return

        if classes & self.SKIP_CLASSES:
            self._skip_depth += 1
            if tag == "div":
                self._skipping_div += 1
            return

        # Skip by ID patterns (common MediaWiki navigation)
        if element_id:
            # Skip navigation/sidebar divs by ID
            skip_ids = ("jump-to-nav", "siteSub", "contentSub")
            if element_id in skip_ids or element_id.startswith(
                ("mw-", "p-", "ca-", "n-")
            ):
                if element_id not in ("mw-content-text", "mw-body-content"):
                    self._skip_depth += 1
                    if tag == "div":
                        self._skipping_div += 1
                    return

        # Track when we enter content area
        if classes & self.CONTENT_CLASSES:
            self._content_depth += 1

        # Handle specific content tags
        if tag in ("h1", "h2", "h3", "h4", "h5"):
            self._in_heading = True
            self._heading_text = ""
        elif tag == "table":
            self._in_table = True
        elif tag == "tr":
            self._table_row = []
        elif tag == "pre":
            self._in_pre = True
            self.text_parts.append("\n```\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return

        # Track end of skipped divs
        if tag == "div" and self._skipping_div > 0:
            self._skipping_div -= 1
            self._skip_depth = max(0, self._skip_depth - 1)
            return

        # Check for skip class divs - we don't track which specific div
        # so we just decrement if we're in a skip state
        if tag == "div" and self._skip_depth > 0:
            self._skip_depth = max(0, self._skip_depth - 1)
            return

        if tag in ("h1", "h2", "h3", "h4", "h5"):
            self._in_heading = False
            if self._heading_text.strip():
                # Save previous section
                if self.current_section and self.text_parts:
                    self.sections[self.current_section] = " ".join(self.text_parts)
                self.current_section = self._heading_text.strip()
                self.text_parts = []
                self.text_parts.append(f"\n## {self._heading_text.strip()}\n")
        elif tag == "table":
            self._in_table = False
            self.text_parts.append("\n")
        elif tag == "tr":
            if self._table_row:
                self.text_parts.append(" | ".join(self._table_row) + "\n")
            self._table_row = []
        elif tag in ("td", "th"):
            pass  # Cell content already added
        elif tag in ("p", "div", "li"):
            self.text_parts.append("\n")
        elif tag == "br":
            self.text_parts.append("\n")
        elif tag == "pre":
            self._in_pre = False
            self.text_parts.append("\n```\n")

    def handle_data(self, data: str) -> None:
        # Skip if we're in a skipped section
        if self._skip_depth > 0:
            return

        text = data.strip() if not self._in_pre else data

        if not text:
            return

        if self._in_heading:
            self._heading_text += text
        elif self._in_table and self._table_row is not None:
            # Collect table cell content
            self._table_row.append(text)
            self.text_parts.append(text + " ")
        else:
            self.text_parts.append(text)

    def get_result(self) -> tuple[str, dict[str, str]]:
        """Get extracted text and sections dict."""
        # Save final section
        if self.current_section and self.text_parts:
            self.sections[self.current_section] = " ".join(self.text_parts)

        full_text = " ".join(self.text_parts)
        # Clean up whitespace (but preserve code blocks)
        full_text = re.sub(r"[ \t]+", " ", full_text)
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)
        return full_text.strip(), self.sections


def html_to_text(html: str) -> tuple[str, dict[str, str]]:
    """Convert MediaWiki HTML to plain text with section extraction.

    Specifically designed for MediaWiki HTML structure, targeting the
    bodyContent or mw-parser-output div and skipping navigation/sidebar content.

    Supports both modern MediaWiki (mw-parser-output class) and older versions
    (bodyContent id).

    Args:
        html: Raw HTML content from MediaWiki page

    Returns:
        Tuple of (full_text, sections_dict)
    """
    content_html = html

    # Try older MediaWiki structure first (id="bodyContent") - common on many wikis
    body_start = html.find('<div id="bodyContent">')
    if body_start >= 0:
        # End at printfooter (before footer/categories)
        footer_start = html.find('<div class="printfooter">', body_start)
        if footer_start > body_start:
            content_html = html[body_start:footer_start]
        else:
            # No printfooter, try to find end of bodyContent div
            # This is harder, so just take everything after bodyContent start
            content_html = html[body_start:]
    else:
        # Try modern MediaWiki structure (class="mw-parser-output")
        content_match = re.search(
            r'<div[^>]*class="[^"]*mw-parser-output[^"]*"[^>]*>(.*?)</div>\s*'
            r'(?:<div[^>]*class="[^"]*printfooter|$)',
            html,
            re.DOTALL | re.IGNORECASE,
        )
        if content_match:
            content_html = content_match.group(1)

    # Use simple tag stripping instead of complex parser
    # Remove script and style tags with their content
    content_html = re.sub(
        r"<script[^>]*>.*?</script>", " ", content_html, flags=re.DOTALL | re.IGNORECASE
    )
    content_html = re.sub(
        r"<style[^>]*>.*?</style>", " ", content_html, flags=re.DOTALL | re.IGNORECASE
    )

    # Remove HTML comments
    content_html = re.sub(r"<!--.*?-->", " ", content_html, flags=re.DOTALL)

    # Remove all HTML tags
    text = re.sub(r"<[^>]+>", " ", content_html)

    # Decode HTML entities
    import html as html_module

    text = html_module.unescape(text)

    # Clean up whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # Extract sections (simple heading detection)
    sections: dict[str, str] = {}
    lines = text.split("\n")
    current_section = ""
    current_text: list[str] = []

    for line in lines:
        # Detect headings (lines that look like section titles)
        if len(line) < 100 and line.strip() and not line.strip().startswith("("):
            # Could be a heading - save previous section
            if current_section and current_text:
                sections[current_section] = "\n".join(current_text).strip()
            current_section = line.strip()
            current_text = []
        else:
            current_text.append(line)

    # Save final section
    if current_section and current_text:
        sections[current_section] = "\n".join(current_text).strip()

    return text, sections


def get_embed_model() -> HuggingFaceEmbedding:
    """Get the project's standard embedding model (all-MiniLM-L6-v2)."""
    model_name = get_imas_embedding_model()
    return HuggingFaceEmbedding(
        model_name=model_name,
        trust_remote_code=False,
    )


class WikiIngestionPipeline:
    """Pipeline for ingesting wiki pages into the knowledge graph.

    Handles the full lifecycle: fetch → chunk → embed → link.
    Uses same embedding model as code examples for unified search.
    """

    def __init__(
        self,
        facility_id: str = "epfl",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_rich: bool = True,
    ):
        """Initialize the pipeline.

        Args:
            facility_id: Facility ID (e.g., "epfl")
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            use_rich: Use Rich progress display
        """
        self.facility_id = facility_id
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_rich = use_rich

        # Initialize text splitter
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n",
        )

        # Embedding model is loaded lazily
        self._embed_model: HuggingFaceEmbedding | None = None

    @property
    def embed_model(self) -> HuggingFaceEmbedding:
        """Lazy-load embedding model."""
        if self._embed_model is None:
            self._embed_model = get_embed_model()
        return self._embed_model

    def _generate_page_id(self, page_name: str) -> str:
        """Generate unique ID for a WikiPage."""
        return f"{self.facility_id}:{page_name}"

    def _generate_chunk_id(self, page_id: str, chunk_idx: int) -> str:
        """Generate unique ID for a WikiChunk."""
        return f"{page_id}:chunk_{chunk_idx}"

    async def ingest_page(self, page: WikiPage) -> PageIngestionStats:
        """Ingest a single wiki page into the graph.

        Deterministic pipeline: fetch → extract → chunk → embed → persist.
        No LLM calls - all entity extraction uses regex patterns.

        Args:
            page: Scraped WikiPage instance

        Returns:
            Stats dict: {chunks, tree_nodes_linked, imas_paths_linked, conventions,
                        content_preview, mdsplus_paths}
        """
        from .scraper import (
            extract_conventions,
            extract_imas_paths,
            extract_mdsplus_paths,
            extract_units,
        )

        stats: PageIngestionStats = {
            "chunks": 0,
            "tree_nodes_linked": 0,
            "imas_paths_linked": 0,
            "conventions": 0,
            "units": 0,
            "content_preview": "",
            "mdsplus_paths": [],
        }

        page_id = self._generate_page_id(page.page_name)

        # Extract text content
        text_content, sections = html_to_text(page.content_html)
        if not text_content or len(text_content) < 50:
            logger.warning("Skipping %s: insufficient content", page.page_name)
            return stats

        # Add content preview and MDSplus paths for progress display
        stats["content_preview"] = text_content[:300]
        stats["mdsplus_paths"] = page.mdsplus_paths[:10] if page.mdsplus_paths else []

        # Create LlamaIndex document for chunking
        doc = Document(
            text=text_content,
            metadata={
                "page_id": page_id,
                "title": page.title,
                "url": page.url,
                "facility_id": self.facility_id,
            },
        )

        # Split into chunks
        nodes = self.splitter.get_nodes_from_documents([doc])
        stats["chunks"] = len(nodes)

        # Generate embeddings and prepare batch data
        chunk_batch: list[dict] = []
        all_mdsplus: set[str] = set()
        all_imas: set[str] = set()

        for i, node in enumerate(nodes):
            chunk_text: str = node.text  # type: ignore[attr-defined]
            node.embedding = self.embed_model.get_text_embedding(chunk_text)

            # Extract entities from this chunk
            chunk_mdsplus = extract_mdsplus_paths(chunk_text)
            chunk_imas = extract_imas_paths(chunk_text)
            chunk_units = extract_units(chunk_text)
            chunk_conventions = extract_conventions(chunk_text)

            # Track for stats
            all_mdsplus.update(chunk_mdsplus)
            all_imas.update(chunk_imas)
            stats["units"] += len(chunk_units)
            stats["conventions"] += len(chunk_conventions)

            chunk_batch.append(
                {
                    "id": self._generate_chunk_id(page_id, i),
                    "wiki_page_id": page_id,
                    "facility_id": self.facility_id,
                    "content": chunk_text,
                    "embedding": node.embedding,
                    "mdsplus_paths": chunk_mdsplus,
                    "imas_paths": chunk_imas,
                    "units": chunk_units,
                    "conventions": [c.get("name", "") for c in chunk_conventions],
                }
            )

        # Persist all chunks in batches
        with GraphClient() as gc:
            # Update WikiPage status to ingested
            gc.query(
                """
                MERGE (p:WikiPage {id: $id})
                SET p.url = $url,
                    p.title = $title,
                    p.facility_id = $facility_id,
                    p.status = 'ingested',
                    p.content_hash = $hash,
                    p.last_scraped = datetime(),
                    p.chunk_count = $chunk_count,
                    p.mdsplus_paths_found = $mdsplus_paths,
                    p.imas_paths_found = $imas_paths,
                    p.conventions_found = $conventions
                WITH p
                MATCH (f:Facility {id: $facility_id})
                MERGE (p)-[:FACILITY_ID]->(f)
                """,
                id=page_id,
                url=page.url,
                title=page.title,
                facility_id=self.facility_id,
                hash=page.content_hash,
                chunk_count=len(nodes),
                mdsplus_paths=page.mdsplus_paths,
                imas_paths=page.imas_paths,
                conventions=[c.get("name", "") for c in page.conventions],
            )

            # Batch persist chunks using UNWIND
            for i in range(0, len(chunk_batch), BATCH_SIZE):
                batch = chunk_batch[i : i + BATCH_SIZE]
                gc.query(
                    """
                    UNWIND $chunks AS chunk
                    MERGE (c:WikiChunk {id: chunk.id})
                    SET c.wiki_page_id = chunk.wiki_page_id,
                        c.facility_id = chunk.facility_id,
                        c.content = chunk.content,
                        c.embedding = chunk.embedding,
                        c.mdsplus_paths_mentioned = chunk.mdsplus_paths,
                        c.imas_paths_mentioned = chunk.imas_paths,
                        c.units_mentioned = chunk.units,
                        c.conventions_mentioned = chunk.conventions
                    WITH c, chunk
                    MATCH (p:WikiPage {id: chunk.wiki_page_id})
                    MERGE (p)-[:HAS_CHUNK]->(c)
                    """,
                    chunks=batch,
                )

            # Batch link chunks to TreeNodes
            if all_mdsplus:
                result = gc.query(
                    """
                    MATCH (c:WikiChunk {wiki_page_id: $page_id})
                    WHERE c.mdsplus_paths_mentioned IS NOT NULL
                    UNWIND c.mdsplus_paths_mentioned AS mds_path
                    MATCH (t:TreeNode)
                    WHERE t.path = mds_path
                       OR t.path ENDS WITH replace(mds_path, '\\\\', '')
                       OR t.canonical_path = toUpper(mds_path)
                    MERGE (c)-[:DOCUMENTS]->(t)
                    RETURN count(*) AS linked
                    """,
                    page_id=page_id,
                )
                if result and result[0]["linked"]:
                    stats["tree_nodes_linked"] = result[0]["linked"]

            # Batch link chunks to IMASPaths
            if all_imas:
                result = gc.query(
                    """
                    MATCH (c:WikiChunk {wiki_page_id: $page_id})
                    WHERE c.imas_paths_mentioned IS NOT NULL
                    UNWIND c.imas_paths_mentioned AS imas_path
                    MATCH (ip:IMASPath)
                    WHERE ip.full_path CONTAINS imas_path
                    MERGE (c)-[:MENTIONS_IMAS]->(ip)
                    RETURN count(*) AS linked
                    """,
                    page_id=page_id,
                )
                if result and result[0]["linked"]:
                    stats["imas_paths_linked"] = result[0]["linked"]

            # Create SignConvention nodes from page conventions (batch)
            if page.conventions:
                conv_batch = []
                for conv in page.conventions:
                    conv_id = f"{self.facility_id}:{conv.get('type', 'unknown')}:{hashlib.md5(conv.get('name', '').encode()).hexdigest()[:8]}"
                    conv_batch.append(
                        {
                            "id": conv_id,
                            "facility_id": self.facility_id,
                            "type": conv.get("type", "sign"),
                            "name": conv.get("name", ""),
                            "context": conv.get("context", ""),
                            "url": page.url,
                            "cocos_index": conv.get("cocos_index"),
                        }
                    )

                gc.query(
                    """
                    UNWIND $convs AS conv
                    MERGE (sc:SignConvention {id: conv.id})
                    SET sc.facility_id = conv.facility_id,
                        sc.convention_type = conv.type,
                        sc.name = conv.name,
                        sc.description = conv.context,
                        sc.wiki_source = conv.url,
                        sc.cocos_index = conv.cocos_index
                    WITH sc, conv
                    MATCH (f:Facility {id: conv.facility_id})
                    MERGE (sc)-[:FACILITY_ID]->(f)
                    """,
                    convs=conv_batch,
                )

            # Update link counts on WikiPage
            gc.query(
                """
                MATCH (p:WikiPage {id: $page_id})
                SET p.link_count = $links
                """,
                page_id=page_id,
                links=stats["tree_nodes_linked"] + stats["imas_paths_linked"],
            )

        return stats

    async def ingest_pages(
        self,
        page_names: list[str],
        progress_callback: ProgressCallback | None = None,
        rate_limit: float = 0.5,
    ) -> dict[str, int]:
        """Ingest multiple wiki pages by name (legacy method).

        For graph-driven workflow, use ingest_from_graph() instead.

        Args:
            page_names: List of wiki page names to ingest
            progress_callback: Optional callback for progress updates
            rate_limit: Minimum seconds between requests

        Returns:
            Aggregate stats dict
        """
        import asyncio

        total_stats = {
            "pages": 0,
            "pages_failed": 0,
            "chunks": 0,
            "tree_nodes_linked": 0,
            "imas_paths_linked": 0,
            "conventions": 0,
            "units": 0,
        }

        # Set up progress monitoring
        monitor = WikiProgressMonitor(use_rich=self.use_rich)
        set_current_monitor(monitor)
        monitor.start(total_pages=len(page_names))

        def report(current: int, total: int, message: str) -> None:
            if progress_callback:
                progress_callback(current, total, message)

        try:
            for i, page_name in enumerate(page_names):
                report(i, len(page_names), f"Processing {page_name}")
                page_id = self._generate_page_id(page_name)

                try:
                    # Fetch the page
                    page = fetch_wiki_page(page_name, facility=self.facility_id)

                    # Ingest it
                    page_stats = await self.ingest_page(page)

                    # Aggregate stats
                    total_stats["pages"] += 1
                    total_stats["chunks"] += page_stats["chunks"]
                    total_stats["tree_nodes_linked"] += page_stats["tree_nodes_linked"]
                    total_stats["imas_paths_linked"] += page_stats["imas_paths_linked"]
                    total_stats["conventions"] += page_stats["conventions"]
                    total_stats["units"] += page_stats["units"]

                    # Update progress monitor with content preview
                    monitor.update_scrape(
                        page_name,
                        chunks=page_stats["chunks"],
                        tree_nodes=page_stats["tree_nodes_linked"],
                        imas_paths=page_stats["imas_paths_linked"],
                        conventions=page_stats["conventions"],
                        units=page_stats["units"],
                        content_preview=str(page_stats.get("content_preview", "")),
                        mdsplus_paths=page_stats.get("mdsplus_paths"),
                    )

                except Exception as e:
                    logger.error("Failed to ingest %s: %s", page_name, e)
                    total_stats["pages_failed"] += 1
                    monitor.update_scrape(page_name, failed=True)
                    # Mark as failed in graph
                    mark_wiki_page_status(page_id, "failed", str(e))

                # Rate limiting
                await asyncio.sleep(rate_limit)

        finally:
            monitor.finish()
            set_current_monitor(None)

        report(len(page_names), len(page_names), "Ingestion complete")
        return total_stats

    async def ingest_from_graph(
        self,
        limit: int = 50,
        min_interest_score: float = 0.0,
        progress_callback: ProgressCallback | None = None,
        rate_limit: float = 0.5,
    ) -> dict[str, int]:
        """Ingest wiki pages from the graph queue (graph-driven workflow).

        Reads WikiPage nodes with status='discovered', fetches content,
        generates embeddings, and creates chunks with graph links.

        This is the preferred method - discover creates the queue,
        ingest processes it.

        Args:
            limit: Maximum pages to process
            min_interest_score: Minimum interest score threshold
            progress_callback: Optional callback for progress updates
            rate_limit: Minimum seconds between requests

        Returns:
            Aggregate stats dict
        """
        # Get pending pages from graph
        pending = get_pending_wiki_pages(
            self.facility_id,
            limit=limit,
            min_interest_score=min_interest_score,
        )

        if not pending:
            logger.info("No pending wiki pages for %s", self.facility_id)
            return {
                "pages": 0,
                "pages_failed": 0,
                "chunks": 0,
                "tree_nodes_linked": 0,
                "imas_paths_linked": 0,
                "conventions": 0,
                "units": 0,
            }

        # Extract page names from the graph results
        page_names = [p["title"] for p in pending]

        logger.info(
            "Processing %d wiki pages from graph queue for %s",
            len(page_names),
            self.facility_id,
        )

        # Use the existing ingest_pages method
        return await self.ingest_pages(
            page_names,
            progress_callback=progress_callback,
            rate_limit=rate_limit,
        )


def create_wiki_vector_index() -> None:
    """Create Neo4j vector index for WikiChunk embeddings.

    Call this once after first ingestion to enable semantic search.
    """
    with GraphClient() as gc:
        gc.query(
            """
            CREATE VECTOR INDEX wiki_chunk_embedding IF NOT EXISTS
            FOR (c:WikiChunk) ON c.embedding
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """
        )
        logger.info("Created wiki_chunk_embedding vector index")


def get_wiki_stats(facility_id: str) -> dict:
    """Get wiki ingestion statistics for a facility.

    Args:
        facility_id: Facility ID

    Returns:
        Stats dict with page/chunk counts and link statistics
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility_id})
            OPTIONAL MATCH (wp)-[:HAS_CHUNK]->(wc:WikiChunk)
            OPTIONAL MATCH (wc)-[:DOCUMENTS]->(t:TreeNode)
            OPTIONAL MATCH (wc)-[:MENTIONS_IMAS]->(ip:IMASPath)
            WITH wp, count(DISTINCT wc) AS chunks,
                 count(DISTINCT t) AS tree_nodes,
                 count(DISTINCT ip) AS imas_paths
            RETURN count(wp) AS pages,
                   sum(chunks) AS total_chunks,
                   sum(tree_nodes) AS tree_nodes_linked,
                   sum(imas_paths) AS imas_paths_linked
            """,
            facility_id=facility_id,
        )

        if result:
            return {
                "pages": result[0]["pages"],
                "chunks": result[0]["total_chunks"],
                "tree_nodes_linked": result[0]["tree_nodes_linked"],
                "imas_paths_linked": result[0]["imas_paths_linked"],
            }
        return {"pages": 0, "chunks": 0, "tree_nodes_linked": 0, "imas_paths_linked": 0}


# =============================================================================
# Artifact Ingestion
# =============================================================================


class ArtifactIngestionStats(TypedDict):
    """Statistics from ingesting a single artifact."""

    chunks: int
    content_preview: str
    artifact_type: str


# Default size limits for artifact ingestion
DEFAULT_MAX_ARTIFACT_SIZE_MB = 5.0  # 5 MB default
MAX_ARTIFACT_SIZE_BYTES = int(DEFAULT_MAX_ARTIFACT_SIZE_MB * 1024 * 1024)


def _decode_url(url: str) -> str:
    """Decode URL-encoded path components.

    Wiki artifact URLs may have encoded slashes (%2F) that need decoding.
    """
    from urllib.parse import unquote

    return unquote(url)


def fetch_artifact_size(
    url: str,
    facility: str = "epfl",
    timeout: int = 30,
) -> int | None:
    """Fetch artifact file size via HTTP HEAD request.

    Uses SSH to fetch Content-Length header without downloading content.

    Args:
        url: Full URL to artifact (will be URL-decoded)
        facility: SSH host alias
        timeout: Timeout in seconds

    Returns:
        File size in bytes, or None if size cannot be determined
    """
    import subprocess

    decoded_url = _decode_url(url)
    cmd = f'curl -skI "{decoded_url}" 2>/dev/null | grep -i content-length | head -1'
    try:
        result = subprocess.run(
            ["ssh", facility, cmd],
            capture_output=True,
            timeout=timeout,
            text=True,
        )

        if result.returncode == 0 and result.stdout:
            # Parse "Content-Length: 12345"
            for line in result.stdout.strip().split("\n"):
                if "content-length" in line.lower():
                    parts = line.split(":")
                    if len(parts) == 2:
                        try:
                            return int(parts[1].strip())
                        except ValueError:
                            pass
    except subprocess.TimeoutExpired:
        logger.warning("Timeout fetching size for %s", url)

    return None


async def fetch_artifact_content(
    url: str,
    facility: str = "epfl",
    timeout: int = 120,
) -> tuple[str, bytes]:
    """Fetch artifact content via SSH.

    Args:
        url: Full URL to artifact (will be URL-decoded)
        facility: SSH host alias
        timeout: Timeout in seconds

    Returns:
        Tuple of (content_type, raw_bytes)
    """
    import subprocess

    # Decode URL-encoded paths
    decoded_url = _decode_url(url)

    # Fetch via SSH with SSL verification disabled
    cmd = f'curl -skL -o /dev/stdout "{decoded_url}"'
    result = subprocess.run(
        ["ssh", facility, cmd],
        capture_output=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to fetch {url}: {result.stderr.decode()}")

    # Determine content type from URL extension
    ext = url.rsplit(".", 1)[-1].lower()
    content_types = {
        "pdf": "application/pdf",
        "ppt": "application/vnd.ms-powerpoint",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "doc": "application/msword",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xls": "application/vnd.ms-excel",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
    }

    return content_types.get(ext, "application/octet-stream"), result.stdout


class WikiArtifactPipeline:
    """Pipeline for ingesting wiki artifacts (PDFs, presentations, etc.).

    Uses LlamaIndex readers for PDF extraction. Other artifact types
    are currently marked as deferred (require OCR or specialized parsers).

    Supported types:
    - PDF: Full text extraction via pypdf (up to max_size_mb limit)
    - Others: Deferred for future implementation

    Size limits:
    - Artifacts larger than max_size_mb are marked as 'deferred' with size_bytes stored
    - This prevents flooding the graph with oversized content
    - Deferred artifacts can still be searched via metadata (filename, linked pages)
    """

    def __init__(
        self,
        facility_id: str = "epfl",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_rich: bool = True,
        max_size_mb: float = DEFAULT_MAX_ARTIFACT_SIZE_MB,
    ):
        """Initialize the artifact pipeline.

        Args:
            facility_id: Facility ID
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            use_rich: Use Rich progress display
            max_size_mb: Maximum artifact size in MB (default 5.0)
        """
        self.facility_id = facility_id
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_rich = use_rich
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)

        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n",
        )

        self._embed_model: HuggingFaceEmbedding | None = None

    @property
    def embed_model(self) -> HuggingFaceEmbedding:
        """Lazy-load embedding model."""
        if self._embed_model is None:
            self._embed_model = get_embed_model()
        return self._embed_model

    async def ingest_pdf(
        self,
        artifact_id: str,
        pdf_bytes: bytes,
    ) -> ArtifactIngestionStats:
        """Ingest a PDF artifact.

        Args:
            artifact_id: WikiArtifact node ID
            pdf_bytes: Raw PDF content

        Returns:
            Ingestion stats
        """
        import tempfile
        from pathlib import Path

        from llama_index.readers.file import PDFReader

        from .scraper import (
            extract_conventions,
            extract_imas_paths,
            extract_mdsplus_paths,
            extract_units,
        )

        stats: ArtifactIngestionStats = {
            "chunks": 0,
            "content_preview": "",
            "artifact_type": "pdf",
        }

        # Write to temp file for PDFReader
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            temp_path = Path(f.name)

        try:
            reader = PDFReader()
            documents = reader.load_data(temp_path)

            if not documents:
                logger.warning("No content extracted from PDF: %s", artifact_id)
                return stats

            # Combine all pages
            full_text = "\n\n".join(doc.text for doc in documents if doc.text)
            stats["content_preview"] = full_text[:300]

            # Split into chunks
            combined_doc = Document(
                text=full_text,
                metadata={
                    "artifact_id": artifact_id,
                    "facility_id": self.facility_id,
                },
            )
            nodes = self.splitter.get_nodes_from_documents([combined_doc])
            stats["chunks"] = len(nodes)

            # Generate embeddings and prepare batch
            chunk_batch: list[dict] = []
            for i, node in enumerate(nodes):
                chunk_text: str = node.text  # type: ignore[attr-defined]
                node.embedding = self.embed_model.get_text_embedding(chunk_text)

                chunk_mdsplus = extract_mdsplus_paths(chunk_text)
                chunk_imas = extract_imas_paths(chunk_text)
                chunk_units = extract_units(chunk_text)
                chunk_conventions = extract_conventions(chunk_text)

                chunk_batch.append(
                    {
                        "id": f"{artifact_id}:chunk_{i}",
                        "artifact_id": artifact_id,
                        "facility_id": self.facility_id,
                        "content": chunk_text,
                        "embedding": node.embedding,
                        "mdsplus_paths": chunk_mdsplus,
                        "imas_paths": chunk_imas,
                        "units": chunk_units,
                        "conventions": [c.get("name", "") for c in chunk_conventions],
                    }
                )

            # Persist chunks
            with GraphClient() as gc:
                # Update artifact status
                gc.query(
                    """
                    MATCH (wa:WikiArtifact {id: $id})
                    SET wa.status = 'ingested',
                        wa.chunk_count = $chunks,
                        wa.ingested_at = datetime()
                    """,
                    id=artifact_id,
                    chunks=len(nodes),
                )

                # Batch persist chunks (WikiChunk for artifacts too)
                for i in range(0, len(chunk_batch), BATCH_SIZE):
                    batch = chunk_batch[i : i + BATCH_SIZE]
                    gc.query(
                        """
                        UNWIND $chunks AS chunk
                        MERGE (c:WikiChunk {id: chunk.id})
                        SET c.artifact_id = chunk.artifact_id,
                            c.facility_id = chunk.facility_id,
                            c.content = chunk.content,
                            c.embedding = chunk.embedding,
                            c.mdsplus_paths_mentioned = chunk.mdsplus_paths,
                            c.imas_paths_mentioned = chunk.imas_paths,
                            c.units_mentioned = chunk.units,
                            c.conventions_mentioned = chunk.conventions
                        WITH c, chunk
                        MATCH (wa:WikiArtifact {id: chunk.artifact_id})
                        MERGE (wa)-[:HAS_CHUNK]->(c)
                        """,
                        chunks=batch,
                    )

            return stats

        finally:
            temp_path.unlink(missing_ok=True)

    async def ingest_from_graph(
        self,
        limit: int = 20,
        min_interest_score: float = 0.5,
    ) -> dict[str, int]:
        """Ingest artifacts from the graph queue.

        Pre-checks file size via HTTP HEAD before downloading. Artifacts
        exceeding max_size_bytes are marked as 'deferred' with size stored.
        Currently only processes PDFs. Other types are marked as deferred.

        Args:
            limit: Maximum artifacts to process
            min_interest_score: Minimum score threshold

        Returns:
            Stats dict
        """
        total_stats = {
            "artifacts": 0,
            "artifacts_failed": 0,
            "artifacts_deferred": 0,
            "artifacts_oversized": 0,
            "chunks": 0,
        }

        pending = get_pending_wiki_artifacts(
            self.facility_id,
            limit=limit,
            min_interest_score=min_interest_score,
        )

        if not pending:
            logger.info("No pending artifacts for %s", self.facility_id)
            return total_stats

        for artifact in pending:
            artifact_id = artifact["id"]
            artifact_type = artifact.get("artifact_type", "unknown")
            url = artifact["url"]
            filename = artifact.get("filename", "unknown")

            try:
                # Check size before downloading
                size_bytes = fetch_artifact_size(url, facility=self.facility_id)

                if size_bytes is not None:
                    # Update size in graph
                    with GraphClient() as gc:
                        gc.query(
                            """
                            MATCH (wa:WikiArtifact {id: $id})
                            SET wa.size_bytes = $size
                            """,
                            id=artifact_id,
                            size=size_bytes,
                        )

                    # Check against limit
                    if size_bytes > self.max_size_bytes:
                        size_mb = size_bytes / (1024 * 1024)
                        max_mb = self.max_size_bytes / (1024 * 1024)
                        defer_reason = (
                            f"File size {size_mb:.1f} MB exceeds limit {max_mb:.1f} MB"
                        )
                        logger.info(
                            "Deferring oversized artifact %s: %s",
                            filename,
                            defer_reason,
                        )
                        with GraphClient() as gc:
                            gc.query(
                                """
                                MATCH (wa:WikiArtifact {id: $id})
                                SET wa.status = 'deferred',
                                    wa.defer_reason = $reason
                                """,
                                id=artifact_id,
                                reason=defer_reason,
                            )
                        total_stats["artifacts_deferred"] += 1
                        total_stats["artifacts_oversized"] += 1
                        continue

                if artifact_type == "pdf":
                    _, content = await fetch_artifact_content(
                        url, facility=self.facility_id
                    )
                    stats = await self.ingest_pdf(artifact_id, content)
                    total_stats["artifacts"] += 1
                    total_stats["chunks"] += stats["chunks"]
                else:
                    # Mark non-PDF as deferred
                    with GraphClient() as gc:
                        gc.query(
                            """
                            MATCH (wa:WikiArtifact {id: $id})
                            SET wa.status = 'deferred',
                                wa.defer_reason = $reason
                            """,
                            id=artifact_id,
                            reason=f"Artifact type '{artifact_type}' not yet supported",
                        )
                    total_stats["artifacts_deferred"] += 1

            except Exception as e:
                logger.error("Failed to ingest artifact %s: %s", artifact_id, e)
                with GraphClient() as gc:
                    gc.query(
                        """
                        MATCH (wa:WikiArtifact {id: $id})
                        SET wa.status = 'failed', wa.error = $error
                        """,
                        id=artifact_id,
                        error=str(e),
                    )
                total_stats["artifacts_failed"] += 1

        return total_stats


__all__ = [
    "ArtifactIngestionStats",
    "DEFAULT_MAX_ARTIFACT_SIZE_MB",
    "ProgressCallback",
    "WikiArtifactPipeline",
    "WikiIngestionPipeline",
    "create_wiki_vector_index",
    "fetch_artifact_content",
    "fetch_artifact_size",
    "get_embed_model",
    "get_pending_wiki_artifacts",
    "get_pending_wiki_pages",
    "get_wiki_queue_stats",
    "get_wiki_stats",
    "html_to_text",
    "link_chunks_to_entities",
    "mark_wiki_page_status",
    "persist_chunks_batch",
]
