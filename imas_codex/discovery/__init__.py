"""
Discovery Engine for remote facility exploration.

Provides three discovery domains:

1. **base** - Shared infrastructure
   - Facility configuration (get_facility, update_infrastructure, etc.)
   - Parallel command execution
   - Progress display utilities

2. **paths** - Filesystem discovery
   - SSH directory scanning
   - LLM-based path scoring
   - Deep enrichment (du, tokei, patterns)

3. **wiki** - Documentation discovery
   - MediaWiki/Confluence support
   - Parallel scan/score/ingest workers
   - Content chunking and embedding
"""

# Re-export base infrastructure at discovery level for convenience
from imas_codex.discovery.base import (
    add_exploration_note,
    filter_private_fields,
    get_facilities_dir,
    get_facility,
    get_facility_infrastructure,
    get_facility_metadata,
    get_facility_validated,
    list_facilities,
    update_infrastructure,
    update_metadata,
    validate_no_private_fields,
)

# Re-export paths discovery at discovery level for convenience
from imas_codex.discovery.paths import (
    DirectoryScorer,
    cleanup_orphaned_software_repos,
    clear_facility_paths,
    get_discovery_stats,
    get_frontier,
    get_high_value_paths,
    get_purpose_distribution,
    get_scorable_paths,
    grounded_score,
    run_parallel_discovery,
    seed_facility_roots,
    seed_missing_roots,
)

__all__ = [
    # Base infrastructure
    "get_facility",
    "get_facility_metadata",
    "get_facility_validated",
    "get_facility_infrastructure",
    "update_infrastructure",
    "update_metadata",
    "add_exploration_note",
    "list_facilities",
    "get_facilities_dir",
    "filter_private_fields",
    "validate_no_private_fields",
    # Paths discovery
    "DirectoryScorer",
    "grounded_score",
    "get_discovery_stats",
    "get_purpose_distribution",
    "get_frontier",
    "get_scorable_paths",
    "get_high_value_paths",
    "seed_facility_roots",
    "seed_missing_roots",
    "clear_facility_paths",
    "cleanup_orphaned_software_repos",
    "run_parallel_discovery",
]
