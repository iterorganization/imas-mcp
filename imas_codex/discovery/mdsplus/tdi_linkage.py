"""TDI-to-TreeNode linkage for bidirectional context.

Links TDI function build_path references to TreeNode paths in the graph.
This is a graph-only operation — no SSH needed. Idempotent: safe to
rerun after either TDI or tree discovery completes.
"""

from __future__ import annotations

import logging
import re

from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)


def link_tdi_to_tree_nodes(
    facility: str,
) -> int:
    """Match TDI build_path references to TreeNode paths.

    For each TDIFunction with mdsplus_trees references:
    1. Look up build_path patterns in TDIFunction.source_code
    2. Match against existing TreeNode nodes by path suffix
    3. Create RESOLVES_TO_TREE_NODE edges from TDIFunction to TreeNode
    4. Set accessor_function on matching TreeNodes

    Args:
        facility: Facility identifier (e.g., "tcv")

    Returns:
        Number of RESOLVES_TO_TREE_NODE edges created.
    """
    with GraphClient() as gc:
        # Get all TDI functions with source code and tree references
        tdi_funcs = gc.query(
            """
            MATCH (tf:TDIFunction {facility_id: $facility})
            WHERE tf.source_code IS NOT NULL
              AND tf.mdsplus_trees IS NOT NULL
              AND size(tf.mdsplus_trees) > 0
            RETURN tf.id AS id, tf.name AS name,
                   tf.source_code AS source_code,
                   tf.mdsplus_trees AS trees
            """,
            facility=facility,
        )

        if not tdi_funcs:
            logger.info("No TDI functions with tree references for %s", facility)
            return 0

        logger.info(
            "Processing %d TDI functions for tree linkage in %s",
            len(tdi_funcs),
            facility,
        )

        total_linked = 0
        for func in tdi_funcs:
            paths = _extract_build_paths(func["source_code"])
            if not paths:
                continue

            # Create RESOLVES_TO_TREE_NODE edges for matching paths
            result = gc.query(
                """
                UNWIND $paths AS bp
                WITH bp, $func_id AS fid
                MATCH (tn:TreeNode {facility_id: $facility})
                WHERE tn.path ENDS WITH bp
                WITH tn, fid
                MATCH (tf:TDIFunction {id: fid})
                MERGE (tf)-[:RESOLVES_TO_TREE_NODE]->(tn)
                ON CREATE SET tn.accessor_function = tf.name
                RETURN count(*) AS linked
                """,
                paths=paths,
                func_id=func["id"],
                facility=facility,
            )
            linked = result[0]["linked"] if result else 0
            total_linked += linked

            if linked:
                logger.debug(
                    "Linked %s to %d TreeNodes (from %d build_paths)",
                    func["name"],
                    linked,
                    len(paths),
                )

        logger.info(
            "Created %d TDI→TreeNode links for %s",
            total_linked,
            facility,
        )
        return total_linked


def update_signal_accessors(
    facility: str,
) -> int:
    """Set preferred TDI accessor on FacilitySignals with matching TreeNodes.

    For FacilitySignals that have a SOURCE_NODE TreeNode with a linked
    TDIFunction, set the preferred_accessor to the TDI expression.

    Args:
        facility: Facility identifier

    Returns:
        Number of FacilitySignals updated.
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (s:FacilitySignal {facility_id: $facility})
                  -[:SOURCE_NODE]->(tn:TreeNode)
                  <-[:RESOLVES_TO_TREE_NODE]-(tf:TDIFunction)
            WHERE s.preferred_accessor IS NULL
            SET s.preferred_accessor = tf.name + '("' + s.accessor + '")'
            RETURN count(s) AS updated
            """,
            facility=facility,
        )
        updated = result[0]["updated"] if result else 0

        if updated:
            logger.info(
                "Updated %d FacilitySignal preferred_accessors for %s",
                updated,
                facility,
            )
        return updated


def _extract_build_paths(source_code: str) -> list[str]:
    """Extract MDSplus path suffixes from build_path() calls in TDI source.

    Extracts the path portion after the tree qualifier (\\TREE::).
    Returns canonical uppercase path segments for matching against
    TreeNode.path.

    Args:
        source_code: TDI .fun file content

    Returns:
        List of path suffixes (e.g., ["TOP.RESULTS:I_P", "TOP:PSI_AXIS"])
    """
    paths = []
    # Match build_path("\\TREE::PATH") or build_path('\\TREE::PATH')
    matches = re.findall(
        r'build_path\s*\(\s*["\']([^"\']+)["\']',
        source_code,
        re.IGNORECASE,
    )
    for match in matches:
        # Strip tree qualifier: \\RESULTS::TOP.FOO -> TOP.FOO
        path_part = re.sub(r"^\\\\?\w+::", "", match)
        if path_part:
            paths.append(path_part.upper())

    return list(set(paths))  # Deduplicate
