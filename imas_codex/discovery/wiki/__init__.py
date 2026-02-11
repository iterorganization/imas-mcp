"""Wiki discovery for facility documentation.

Provides a parallel pipeline for discovering and ingesting wiki content:

Phase 1 - SCAN: Enumerate all wiki pages per site
Phase 2 - SCORE: LLM relevance evaluation with content fetch
Phase 3 - INGEST: Chunk and embed high-score pages
Phase 4 - ARTIFACTS: Score and embed wiki attachments (PDFs, images, etc.)

Facility-agnostic design - wiki configuration comes from facility YAML.
"""

from imas_codex.discovery.wiki.auth import (
    CredentialManager,
    WikiSiteConfig,
    require_credentials,
)
from imas_codex.discovery.wiki.config import WikiConfig
from imas_codex.discovery.wiki.confluence import (
    ConfluenceClient,
    ConfluencePage,
    ConfluenceSpace,
    detect_site_type,
)
from imas_codex.discovery.wiki.mediawiki import (
    MediaWikiClient,
    MediaWikiPage,
    TequilaAuthError,
    get_mediawiki_client,
)
from imas_codex.discovery.wiki.monitor import (
    ScanProgressMonitor,
    WikiIngestionStats,
    WikiProgressMonitor,
)
from imas_codex.discovery.wiki.parallel import (
    bulk_discover_pages,
    get_wiki_discovery_stats,
    release_orphaned_claims,
    run_parallel_wiki_discovery,
)
from imas_codex.discovery.wiki.pipeline import (
    WikiArtifactPipeline,
    WikiIngestionPipeline,
    clear_facility_wiki,
    get_pending_wiki_artifacts,
    get_pending_wiki_pages,
    get_wiki_queue_stats,
    get_wiki_stats,
    link_chunks_to_entities,
    mark_wiki_page_status,
    persist_chunks_batch,
)
from imas_codex.discovery.wiki.progress import WikiProgressDisplay
from imas_codex.discovery.wiki.scraper import (
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
    # MediaWiki client
    "MediaWikiClient",
    "MediaWikiPage",
    "TequilaAuthError",
    "get_mediawiki_client",
    # Configuration
    "WikiConfig",
    # Progress monitoring
    "WikiProgressMonitor",
    "WikiIngestionStats",
    "ScanProgressMonitor",
    "WikiProgressDisplay",
    # Parallel discovery
    "bulk_discover_pages",
    "get_wiki_discovery_stats",
    "release_orphaned_claims",
    "run_parallel_wiki_discovery",
    # Ingestion pipeline
    "WikiArtifactPipeline",
    "WikiIngestionPipeline",
    "WikiPage",
    # Extraction utilities
    "extract_conventions",
    "extract_imas_paths",
    "extract_mdsplus_paths",
    "extract_units",
    "fetch_wiki_page",
    # Queue management
    "clear_facility_wiki",
    "get_pending_wiki_artifacts",
    "get_pending_wiki_pages",
    "get_wiki_queue_stats",
    "get_wiki_stats",
    "mark_wiki_page_status",
    # Batch operations
    "link_chunks_to_entities",
    "persist_chunks_batch",
]
