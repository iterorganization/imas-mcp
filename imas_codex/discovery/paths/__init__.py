"""Path discovery for remote facility filesystems.

Provides parallel discovery workers for scanning, triaging, enriching,
and scoring directory structures at fusion facilities.

Pipeline:
    seed → scan → triage → enrich → score

Workers:
    - scan_worker: SSH directory enumeration
    - expand_worker: Create child paths from scan results
    - triage_worker: LLM classification and initial scoring
    - enrich_worker: Deep analysis (du, tokei, patterns)
    - score_worker: Independent scoring with enrichment evidence
"""

from imas_codex.discovery.paths.frontier import (
    cleanup_orphaned_software_repos,
    clear_facility_paths,
    get_discovery_stats,
    get_frontier,
    get_high_value_paths,
    get_purpose_distribution,
    get_scorable_paths,
    seed_facility_roots,
    seed_missing_roots,
)
from imas_codex.discovery.paths.models import (
    DirectoryEvidence,
    ResourcePurpose,
    TerminalReason,
    TriageBatch,
    TriagedBatch,
    TriagedDirectory,
    TriageResult,
)
from imas_codex.discovery.paths.parallel import run_parallel_discovery
from imas_codex.discovery.paths.progress import ParallelProgressDisplay
from imas_codex.discovery.paths.scorer import DirectoryTriager, combined_score

__all__ = [
    # Frontier management
    "get_discovery_stats",
    "get_purpose_distribution",
    "get_frontier",
    "get_scorable_paths",
    "get_high_value_paths",
    "seed_facility_roots",
    "seed_missing_roots",
    "clear_facility_paths",
    "cleanup_orphaned_software_repos",
    # Models
    "ResourcePurpose",
    "TerminalReason",
    "DirectoryEvidence",
    "TriagedDirectory",
    "TriagedBatch",
    "TriageResult",
    "TriageBatch",
    # Triage
    "DirectoryTriager",
    "combined_score",
    # Parallel discovery
    "run_parallel_discovery",
    # Progress
    "ParallelProgressDisplay",
]
