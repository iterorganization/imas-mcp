"""Wiki ingestion pipeline using LlamaIndex.

Processes wiki pages through a multi-stage pipeline:
1. Discover: Find wiki pages and queue them in Neo4j (status='discovered')
2. Fetch: SSH-based HTML retrieval from wiki (status='scraped')
3. Chunk: Split into semantic chunks with SentenceSplitter
4. Embed: Generate vector embeddings with all-MiniLM-L6-v2
5. Link: Connect to TreeNodes, IMASPaths, and SignConventions (status='linked')

The pipeline is graph-driven: discover creates WikiPage nodes, ingest processes them.
This follows the same pattern as SourceFile ingestion for code examples.

Example:
    # Step 1: Discover and queue pages
    queue_wiki_pages("epfl", ["Thomson", "Ion_Temperature_Nodes"])

    # Step 2: Process the queue
    pipeline = WikiIngestionPipeline(facility_id="epfl")
    stats = await pipeline.ingest_from_graph(limit=20)
    print(f"Created {stats['chunks']} chunks with {stats['links']} links")
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
# Graph-Driven Queue Functions
# =============================================================================


def queue_wiki_pages(
    facility_id: str,
    page_names: list[str],
    interest_score: float = 0.5,
    is_priority: bool = False,
) -> dict[str, int]:
    """Queue wiki pages for ingestion by creating WikiPage nodes.

    Creates WikiPage nodes with status='discovered'. Already-discovered
    or ingested pages are skipped (idempotent).

    Args:
        facility_id: Facility ID (e.g., "epfl")
        page_names: List of page names to queue
        interest_score: Priority score (0.0-1.0), higher = more interesting
        is_priority: If True, use higher interest score (0.9)

    Returns:
        Dict with counts: {queued, skipped, total}
    """
    if is_priority:
        interest_score = 0.9

    stats = {"queued": 0, "skipped": 0, "total": len(page_names)}

    with GraphClient() as gc:
        for page_name in page_names:
            page_id = f"{facility_id}:{page_name}"
            url = f"https://spcwiki.epfl.ch/wiki/{page_name}"

            # Use MERGE to avoid duplicates, only set properties if creating
            result = gc.query(
                """
                MERGE (wp:WikiPage {id: $id})
                ON CREATE SET
                    wp.facility_id = $facility_id,
                    wp.url = $url,
                    wp.title = $page_name,
                    wp.status = 'discovered',
                    wp.discovered_at = datetime(),
                    wp.interest_score = $interest_score
                WITH wp
                MATCH (f:Facility {id: $facility_id})
                MERGE (wp)-[:FACILITY_ID]->(f)
                RETURN wp.status AS status
                """,
                id=page_id,
                facility_id=facility_id,
                url=url,
                page_name=page_name,
                interest_score=interest_score,
            )

            if result and result[0]["status"] == "discovered":
                stats["queued"] += 1
            else:
                stats["skipped"] += 1

    logger.info(
        "Queued %d wiki pages for %s (%d skipped)",
        stats["queued"],
        facility_id,
        stats["skipped"],
    )
    return stats


def get_pending_wiki_pages(
    facility_id: str,
    limit: int = 100,
    min_interest_score: float = 0.0,
) -> list[dict]:
    """Get wiki pages pending ingestion from the graph.

    Returns WikiPage nodes with status='discovered', sorted by interest_score.

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
            MATCH (wp:WikiPage {facility_id: $facility_id, status: 'discovered'})
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
        Dict with status counts and total:
        {discovered, scraped, chunked, linked, failed, stale, total}
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
            "discovered": 0,
            "scraped": 0,
            "chunked": 0,
            "linked": 0,
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


def mark_wiki_page_status(
    page_id: str,
    status: str,
    error: str | None = None,
) -> None:
    """Update the status of a wiki page.

    Args:
        page_id: WikiPage ID
        status: New status (discovered, scraped, chunked, linked, failed, stale)
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

        Args:
            page: Scraped WikiPage instance

        Returns:
            Stats dict: {chunks, tree_nodes_linked, imas_paths_linked, conventions,
                        content_preview, mdsplus_paths}
        """
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

        # Generate embeddings
        for node in nodes:
            # LlamaIndex types BaseNode but actual nodes have .text
            node.embedding = self.embed_model.get_text_embedding(node.text)  # type: ignore[attr-defined]

        # Store in Neo4j
        with GraphClient() as gc:
            # Create WikiPage node
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

            # Create WikiChunk nodes with embeddings
            for i, node in enumerate(nodes):
                chunk_id = self._generate_chunk_id(page_id, i)
                chunk_text: str = node.text  # type: ignore[attr-defined]

                # Extract entities from this specific chunk
                from .scraper import (
                    extract_conventions,
                    extract_imas_paths,
                    extract_mdsplus_paths,
                    extract_units,
                )

                chunk_mdsplus = extract_mdsplus_paths(chunk_text)
                chunk_imas = extract_imas_paths(chunk_text)
                chunk_units = extract_units(chunk_text)
                chunk_conventions = extract_conventions(chunk_text)

                gc.query(
                    """
                    MERGE (c:WikiChunk {id: $id})
                    SET c.wiki_page_id = $page_id,
                        c.facility_id = $facility_id,
                        c.content = $content,
                        c.embedding = $embedding,
                        c.mdsplus_paths_mentioned = $mdsplus_paths,
                        c.imas_paths_mentioned = $imas_paths,
                        c.units_mentioned = $units,
                        c.conventions_mentioned = $conventions
                    WITH c
                    MATCH (p:WikiPage {id: $page_id})
                    MERGE (p)-[:HAS_CHUNK]->(c)
                    """,
                    id=chunk_id,
                    page_id=page_id,
                    facility_id=self.facility_id,
                    content=chunk_text,
                    embedding=node.embedding,
                    mdsplus_paths=chunk_mdsplus,
                    imas_paths=chunk_imas,
                    units=chunk_units,
                    conventions=[c.get("name", "") for c in chunk_conventions],
                )

                # Link to TreeNodes (DOCUMENTS relationship)
                for mds_path in chunk_mdsplus:
                    result = gc.query(
                        """
                        MATCH (c:WikiChunk {id: $chunk_id})
                        MATCH (t:TreeNode)
                        WHERE t.path = $path
                           OR t.path ENDS WITH $path_suffix
                           OR t.canonical_path = $canonical
                        MERGE (c)-[:DOCUMENTS]->(t)
                        RETURN count(*) AS linked
                        """,
                        chunk_id=chunk_id,
                        path=mds_path,
                        path_suffix=mds_path.lstrip("\\"),
                        canonical=mds_path.upper(),
                    )
                    if result and result[0]["linked"] > 0:
                        stats["tree_nodes_linked"] += result[0]["linked"]

                # Link to IMASPaths (MENTIONS_IMAS relationship)
                for imas_path in chunk_imas:
                    result = gc.query(
                        """
                        MATCH (c:WikiChunk {id: $chunk_id})
                        MATCH (ip:IMASPath)
                        WHERE ip.full_path CONTAINS $path
                        MERGE (c)-[:MENTIONS_IMAS]->(ip)
                        RETURN count(*) AS linked
                        """,
                        chunk_id=chunk_id,
                        path=imas_path,
                    )
                    if result and result[0]["linked"] > 0:
                        stats["imas_paths_linked"] += result[0]["linked"]

                # Track units and conventions found
                stats["units"] += len(chunk_units)
                stats["conventions"] += len(chunk_conventions)

            # Create SignConvention nodes from page conventions
            for conv in page.conventions:
                conv_id = f"{self.facility_id}:{conv.get('type', 'unknown')}:{hashlib.md5(conv.get('name', '').encode()).hexdigest()[:8]}"
                gc.query(
                    """
                    MERGE (sc:SignConvention {id: $id})
                    SET sc.facility_id = $facility_id,
                        sc.convention_type = $type,
                        sc.name = $name,
                        sc.description = $context,
                        sc.wiki_source = $url,
                        sc.cocos_index = $cocos_index
                    WITH sc
                    MATCH (f:Facility {id: $facility_id})
                    MERGE (sc)-[:FACILITY_ID]->(f)
                    """,
                    id=conv_id,
                    facility_id=self.facility_id,
                    type=conv.get("type", "sign"),
                    name=conv.get("name", ""),
                    context=conv.get("context", ""),
                    url=page.url,
                    cocos_index=conv.get("cocos_index"),
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


__all__ = [
    "ProgressCallback",
    "WikiIngestionPipeline",
    "create_wiki_vector_index",
    "get_embed_model",
    "get_pending_wiki_pages",
    "get_wiki_queue_stats",
    "get_wiki_stats",
    "html_to_text",
    "mark_wiki_page_status",
    "queue_wiki_pages",
]
