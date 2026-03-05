"""Code discovery module for source code scanning, scoring, and ingestion.

Bridges from scored FacilityPaths to code file ingestion. Parallel
worker architecture (scan → triage → score → code) following the
same pattern as wiki discovery.

Pipeline:
    discover code <facility>
      1. SCAN: SSH enumerate code files from scored FacilityPaths
      2. TRIAGE: Per-dimension LLM scoring (discovered → triaged | skipped)
      3. ENRICH: rg pattern matching + preview extraction (triaged → enriched)
      4. SCORE: Full LLM scoring with enrichment evidence (enriched → scored)
      5. CODE: Fetch, tree-sitter chunk, embed (scored → ingested)
      6. LINK: Propagate code evidence to FacilitySignals

Uses claim coordination via ``claimed_at`` and ``files_claimed_at``
for parallel-safe execution across CLI instances.
"""

from .graph_ops import reset_orphaned_file_claims
from .parallel import get_code_discovery_stats, run_parallel_code_discovery
from .scanner import scan_facility_files
from .scorer import (
    FileScoreBatch,
    FileScoreResult,
    FileTriageBatch,
    FileTriageResult,
    apply_file_scores,
    apply_triage_results,
)

__all__ = [
    "FileScoreBatch",
    "FileScoreResult",
    "FileTriageBatch",
    "FileTriageResult",
    "apply_file_scores",
    "apply_triage_results",
    "get_code_discovery_stats",
    "reset_orphaned_file_claims",
    "run_parallel_code_discovery",
    "scan_facility_files",
]
