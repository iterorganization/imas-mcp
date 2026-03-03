"""Backward-compatible re-exports from ``extraction.py``.

This module is a shim — all functionality has moved to
``imas_codex.mdsplus.extraction``. Import from there directly
for new code.
"""

from imas_codex.mdsplus.extraction import (
    _compute_diff,
    _compute_parent_path,
    _load_mdsplus_config,
    _resolve_shots,
    async_extract_tree_version,
    async_extract_units_for_version,
    discover_tree,
    extract_tree_version,
    extract_units_for_version,
    get_static_tree_config,
    get_static_tree_graph_state,
    ingest_static_tree,
    merge_units_into_data,
    merge_version_results,
)

# Backward-compatible aliases
discover_static_tree = discover_tree
discover_static_tree_version = extract_tree_version
async_discover_static_tree_version = async_extract_tree_version
_resolve_versions = _resolve_shots

__all__ = [
    "_compute_diff",
    "_compute_parent_path",
    "_load_mdsplus_config",
    "_resolve_shots",
    "_resolve_versions",
    "async_discover_static_tree_version",
    "async_extract_tree_version",
    "async_extract_units_for_version",
    "discover_static_tree",
    "discover_static_tree_version",
    "discover_tree",
    "extract_tree_version",
    "extract_units_for_version",
    "get_static_tree_config",
    "get_static_tree_graph_state",
    "ingest_static_tree",
    "merge_units_into_data",
    "merge_version_results",
]
