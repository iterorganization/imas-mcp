"""File discovery module for source file scanning and scoring.

Bridges from scored FacilityPaths to source file ingestion.

Pipeline:
    discover files <facility>
      1. SCAN: SSH enumerate files in high-scoring FacilityPaths
      2. SCORE: LLM batch scores files for relevance + interest
      3. (Files become SourceFile nodes ready for `ingest run`)

Uses the same graph state machine as paths/wiki/signals.
"""

from .scanner import scan_facility_files
from .scorer import score_facility_files

__all__ = [
    "scan_facility_files",
    "score_facility_files",
]
