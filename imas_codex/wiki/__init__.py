"""Wiki ingestion module for TCV documentation.

Provides tools for scraping, chunking, embedding, and linking
wiki content to the knowledge graph.

Components:
    scraper: SSH-based wiki page fetching and entity extraction
    pipeline: LlamaIndex ingestion with chunking and embeddings
    progress: Rich progress monitoring for CLI and MCP tools

Example:
    from imas_codex.wiki import WikiIngestionPipeline

    pipeline = WikiIngestionPipeline(facility_id="epfl")
    stats = await pipeline.ingest_pages(["Thomson", "Ion_Temperature_Nodes"])
"""

from .pipeline import WikiIngestionPipeline
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
]
