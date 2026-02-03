"""Wiki ingestion module for facility documentation.

Provides a four-phase parallel pipeline for discovering and ingesting wiki content:

Phase 1 - SCAN: Fast link extraction, builds wiki graph structure
Phase 2 - PREFETCH: Fetch content and generate summaries
Phase 3 - SCORE: LLM evaluation with cost tracking (stops at budget limit)
Phase 4 - INGEST: Chunk and embed high-score pages

Facility-agnostic design - wiki configuration comes from facility YAML.

Example:
    from imas_codex.wiki.parallel import run_parallel_wiki_discovery

    # Run via CLI (recommended):
    # imas-codex discover wiki tcv --cost-limit 10.0
"""

from .auth import CredentialManager, WikiSiteConfig, require_credentials
from .config import WikiConfig
from .confluence import (
    ConfluenceClient,
    ConfluencePage,
    ConfluenceSpace,
    detect_site_type,
)
from .pipeline import (
    WikiArtifactPipeline,
    WikiIngestionPipeline,
    get_pending_wiki_artifacts,
    get_pending_wiki_pages,
    get_wiki_queue_stats,
    link_chunks_to_entities,
    mark_wiki_page_status,
    persist_chunks_batch,
)
from .progress import ScanProgressMonitor, WikiProgressMonitor
from .scraper import (
    WikiPage,
    extract_conventions,
    extract_imas_paths,
    extract_mdsplus_paths,
    extract_units,
    fetch_wiki_page,
)

__all__ = [
    # Authentication
    "CredentialManager",
    "WikiSiteConfig",
    "require_credentials",
    # Confluence client
    "ConfluenceClient",
    "ConfluencePage",
    "ConfluenceSpace",
    "detect_site_type",
    # Configuration
    "WikiConfig",
    # Ingestion
    "WikiArtifactPipeline",
    "WikiIngestionPipeline",
    "WikiPage",
    "WikiProgressMonitor",
    "ScanProgressMonitor",
    # Extraction utilities
    "extract_conventions",
    "extract_imas_paths",
    "extract_mdsplus_paths",
    "extract_units",
    "fetch_wiki_page",
    # Queue management
    "get_pending_wiki_artifacts",
    "get_pending_wiki_pages",
    "get_wiki_queue_stats",
    "mark_wiki_page_status",
    # Batch operations
    "link_chunks_to_entities",
    "persist_chunks_batch",
]
