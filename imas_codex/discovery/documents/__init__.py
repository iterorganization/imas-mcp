"""Document discovery module for non-code file scanning and image processing.

Discovers documents (PDF, Markdown, notebooks, etc.) and images from
scored FacilityPaths, creating Document nodes in the graph. Images
are fetched, downsampled, and optionally captioned with a VLM.

Pipeline:
    discover documents <facility>
      1. SCAN: SSH enumerate document + image files
      2. FETCH: Download images, extract metadata
      3. CAPTION: VLM captioning + relevance scoring (optional)

Uses claim coordination via ``claimed_at`` timestamps for parallel-safe
execution across CLI instances.
"""

from .pipeline import run_document_discovery
from .scanner import (
    clear_facility_documents,
    get_document_discovery_stats,
    scan_facility_documents,
)

__all__ = [
    "clear_facility_documents",
    "get_document_discovery_stats",
    "run_document_discovery",
    "scan_facility_documents",
]
