"""Unified MDSplus tree discovery pipeline.

Handles both versioned (machine-description) and shot-scoped (dynamic)
MDSplus trees through a shared infrastructure: epoch detection, tree
extraction, unit extraction, pattern detection, and signal promotion.
"""

from .pipeline import run_parallel_static_discovery, run_tree_discovery
from .state import StaticDiscoveryState, TreeDiscoveryState

__all__ = [
    "StaticDiscoveryState",
    "TreeDiscoveryState",
    "run_parallel_static_discovery",
    "run_tree_discovery",
]
