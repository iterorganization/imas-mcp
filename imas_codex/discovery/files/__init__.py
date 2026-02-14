"""File discovery module for source file scanning and scoring.

Bridges from scored FacilityPaths to source file ingestion.

Pipeline:
    discover files <facility>
      1. SCAN: SSH enumerate files in high-scoring FacilityPaths
      2. SCORE: LLM batch scores files for relevance + interest
      3. (Files become SourceFile nodes ready for `ingest run`)

Uses claim coordination via ``claimed_at`` and ``files_claimed_at``
for parallel-safe execution across CLI instances.
"""

from .graph_ops import reset_orphaned_file_claims
from .scanner import scan_facility_files
from .scorer import score_facility_files

__all__ = [
    "reset_orphaned_file_claims",
    "scan_facility_files",
    "score_facility_files",
]
