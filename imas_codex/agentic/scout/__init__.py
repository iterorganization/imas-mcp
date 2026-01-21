"""Scout module for graph-first stateless facility exploration.

This module implements a "moving frontier" approach to facility discovery:
- Graph-first: Each step queries graph for frontier, persists results
- Stateless: No context carried between steps - graph IS the state
- Dead-end detection to skip irrelevant paths (git repos, site-packages, etc.)
- Multi-dimensional interest scoring (IMAS, physics codes, data sources)

Uses existing node types:
- FacilityPath: Discovered paths with status and interest_score
- SourceFile: Files queued for ingestion

Primary interface:
- StatelessScout: Main exploration class
- ScoutConfig: Configuration for exploration runs

Legacy (deprecated):
- ScoutSession: Windowed session (use StatelessScout instead)
"""

# Primary interface - graph-first stateless exploration
# Legacy - kept for backward compatibility
from .frontier import FrontierManager, FrontierStats
from .session import ScoutConfig as LegacyScoutConfig, ScoutSession
from .stateless import (
    InterestScore,
    ScoutConfig,
    StatelessScout,
    advance_path_status,
    compute_interest_score,
    discover_path,
    get_exploration_summary,
    get_frontier,
    is_dead_end,
    is_dry_run,
    queue_source_file,
    set_dry_run,
    skip_path,
)
from .tools import get_scout_tools

__all__ = [
    # Primary
    "StatelessScout",
    "ScoutConfig",
    "InterestScore",
    "get_scout_tools",
    # Graph operations
    "discover_path",
    "skip_path",
    "advance_path_status",
    "queue_source_file",
    "get_frontier",
    "get_exploration_summary",
    # Utilities
    "compute_interest_score",
    "is_dead_end",
    "is_dry_run",
    "set_dry_run",
    # Legacy
    "FrontierManager",
    "FrontierStats",
    "ScoutSession",
]
