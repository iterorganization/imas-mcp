"""File discovery module for source file scanning, scoring, and ingestion.

Bridges from scored FacilityPaths to source file ingestion. Parallel
worker architecture (scan → score → code/docs) following the
same pattern as wiki discovery.

Pipeline:
    discover files <facility>
      1. SCAN: SSH enumerate files in high-scoring FacilityPaths
      2. SCORE: LLM batch scores files for relevance + interest
      3. CODE: Fetch, chunk, embed high-scoring code files
      4. DOCS: Ingest documents, notebooks, configs

Uses claim coordination via ``claimed_at`` and ``files_claimed_at``
for parallel-safe execution across CLI instances.
"""

from .graph_ops import reset_orphaned_file_claims
from .parallel import get_file_discovery_stats, run_parallel_file_discovery
from .scanner import scan_facility_files
from .scorer import FileScoreBatch, FileScoreResult, score_facility_files

__all__ = [
    "FileScoreBatch",
    "FileScoreResult",
    "get_file_discovery_stats",
    "reset_orphaned_file_claims",
    "run_parallel_file_discovery",
    "scan_facility_files",
    "score_facility_files",
]
