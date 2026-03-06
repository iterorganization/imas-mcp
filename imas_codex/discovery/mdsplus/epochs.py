"""Epoch detection integration for the unified MDSplus pipeline.

Wraps BatchDiscovery to detect structural epoch boundaries for trees
with ``detect_epochs: true`` in their TreeConfig. Discovered epochs
become StructuralEpoch nodes, just like pre-configured versions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from imas_codex.mdsplus.batch_discovery import (
    EpochProgress,
    discover_epochs_optimized,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)


def detect_epochs_for_tree(
    facility: str,
    data_source_name: str,
    tree_config: dict[str, Any],
    client: GraphClient | None = None,
    on_progress: Callable[[EpochProgress], None] | None = None,
) -> list[dict[str, Any]]:
    """Detect structural epochs for a tree with ``detect_epochs: true``.

    Uses the batched SSH fingerprinting + binary search algorithm from
    ``BatchDiscovery`` to find shot boundaries where tree structure
    changes. Returns epoch records ready for StructuralEpoch ingestion.

    Args:
        facility: Facility identifier / SSH host alias
        data_source_name: MDSplus tree name (e.g., "results")
        tree_config: TreeConfig dict from facility YAML
        client: Optional GraphClient for incremental mode
        on_progress: Optional callback for progress updates

    Returns:
        List of epoch dicts with keys: id, data_source_name, facility_id,
        version, first_shot, last_shot, representative_shot, node_count,
        is_new. Empty list if no epochs detected.
    """
    epoch_config = tree_config.get("epoch_config", {}) or {}
    start_shot = epoch_config.get("start_shot")
    step_size = epoch_config.get("step_size", 1000)

    logger.info(
        "Detecting epochs for %s:%s (step=%d)",
        facility,
        data_source_name,
        step_size,
    )

    epochs, _structures = discover_epochs_optimized(
        facility=facility,
        data_source_name=data_source_name,
        start_shot=start_shot,
        coarse_step=step_size,
        client=client,
        on_progress=on_progress,
    )

    if not epochs:
        logger.info(
            "No structural epochs detected for %s:%s", facility, data_source_name
        )
        return []

    new_count = sum(1 for e in epochs if e.get("is_new", True))
    logger.info(
        "Detected %d epochs for %s:%s (%d new)",
        len(epochs),
        facility,
        data_source_name,
        new_count,
    )

    return epochs


def detect_epochs_for_subtrees(
    facility: str,
    parent_tree: str,
    subtree_configs: list[dict[str, Any]],
    client: GraphClient | None = None,
    on_progress: Callable[[EpochProgress], None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Detect epochs for multiple subtrees of a parent tree.

    Each subtree is independently epoched — structure changes in one
    subtree don't affect others.

    Args:
        facility: Facility identifier / SSH host alias
        parent_tree: Parent tree name (e.g., "tcv_shot")
        subtree_configs: List of TreeConfig dicts for subtrees
        client: Optional GraphClient for incremental mode
        on_progress: Optional callback for progress updates

    Returns:
        Dict mapping subtree name to list of epoch records.
    """
    results: dict[str, list[dict[str, Any]]] = {}

    for sub_config in subtree_configs:
        sub_name = sub_config.get("source_name", "")
        if not sub_config.get("detect_epochs"):
            continue

        logger.info(
            "Detecting epochs for %s:%s (subtree of %s)",
            facility,
            sub_name,
            parent_tree,
        )

        epochs = detect_epochs_for_tree(
            facility=facility,
            data_source_name=sub_name,
            tree_config=sub_config,
            client=client,
            on_progress=on_progress,
        )
        results[sub_name] = epochs

    return results


def epochs_to_versions(epochs: list[dict[str, Any]]) -> list[int]:
    """Extract version numbers from epoch records for seeding.

    Args:
        epochs: List of epoch dicts from detect_epochs_for_tree

    Returns:
        Sorted list of version numbers
    """
    return sorted(e["version"] for e in epochs if "version" in e)
