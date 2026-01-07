"""Wiki ingestion module for TCV documentation.

Provides tools for scraping, chunking, embedding, and linking
wiki content to the knowledge graph.

Graph-Driven Workflow:
    1. discover: Find pages and create WikiPage nodes (status='discovered')
    2. ingest: Process queued pages via ingest_from_graph()
    3. status: Check queue statistics

Components:
    scraper: SSH-based wiki page fetching and entity extraction
    pipeline: LlamaIndex ingestion with chunking and embeddings
    progress: Rich progress monitoring for CLI and MCP tools

Example:
    from imas_codex.wiki import (
        WikiIngestionPipeline,
        queue_wiki_pages,
        get_pending_wiki_pages,
    )

    # Step 1: Queue pages for ingestion
    queue_wiki_pages("epfl", ["Thomson", "Ion_Temperature_Nodes"])

    # Step 2: Process the queue
    pipeline = WikiIngestionPipeline(facility_id="epfl")
    stats = await pipeline.ingest_from_graph(limit=20)
"""

from .pipeline import (
    WikiIngestionPipeline,
    get_pending_wiki_pages,
    get_wiki_queue_stats,
    mark_wiki_page_status,
    queue_wiki_pages,
)
from .progress import WikiProgressMonitor
from .scraper import (
    WikiPage,
    discover_wiki_pages,
    extract_conventions,
    extract_imas_paths,
    extract_mdsplus_paths,
    extract_units,
    fetch_wiki_page,
)

__all__ = [
    "WikiIngestionPipeline",
    "WikiPage",
    "WikiProgressMonitor",
    "discover_wiki_pages",
    "extract_conventions",
    "extract_imas_paths",
    "extract_mdsplus_paths",
    "extract_units",
    "fetch_wiki_page",
    "get_pending_wiki_pages",
    "get_wiki_queue_stats",
    "mark_wiki_page_status",
    "queue_wiki_pages",
]
