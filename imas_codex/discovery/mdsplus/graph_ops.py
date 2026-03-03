"""Graph operations for tree discovery.

Re-exports all operations from the static discovery module and adds
new operations for the unified tree discovery pipeline.
"""

from __future__ import annotations

import logging

from imas_codex.discovery.static.graph_ops import *  # noqa: F401, F403
from imas_codex.discovery.static.graph_ops import (
    CLAIM_TIMEOUT_SECONDS,  # noqa: F401
    seed_versions,  # noqa: F401
)
from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)


def promote_leaf_nodes_to_signals(
    facility: str,
    tree_name: str,
    batch_size: int = 500,
) -> int:
    """Create FacilitySignal nodes from leaf TreeNodes (NUMERIC/SIGNAL).

    Promotes leaf TreeNodes to FacilitySignals with status=discovered.
    Signals get no description at this stage — that comes from the
    signals enrichment pipeline later.

    Args:
        facility: Facility identifier
        tree_name: MDSplus tree name
        batch_size: Number of nodes to process per transaction

    Returns:
        Number of FacilitySignal nodes created or updated.
    """
    da_id = f"{facility}:mdsplus:tree_tdi"

    with GraphClient() as gc:
        # Ensure DataAccess node exists
        gc.query(
            """
            MERGE (da:DataAccess {id: $id})
            ON CREATE SET
                da.facility_id = $facility,
                da.method_type = 'mdsplus',
                da.library = 'MDSplus',
                da.access_type = 'local',
                da.data_source = 'mdsplus',
                da.name = 'MDSplus tree TDI'
            WITH da
            MATCH (f:Facility {id: $facility})
            MERGE (da)-[:AT_FACILITY]->(f)
            """,
            id=da_id,
            facility=facility,
        )

        # Count promotable leaf nodes
        count_result = gc.query(
            """
            MATCH (n:TreeNode {facility_id: $facility, tree_name: $tree})
            WHERE n.node_type IN ['NUMERIC', 'SIGNAL']
              AND NOT EXISTS {
                  MATCH (n)<-[:SOURCE_NODE]-(:FacilitySignal)
              }
            RETURN count(n) AS total
            """,
            facility=facility,
            tree=tree_name,
        )
        total = count_result[0]["total"] if count_result else 0
        if total == 0:
            logger.info("No new leaf nodes to promote for %s:%s", facility, tree_name)
            return 0

        logger.info(
            "Promoting %d leaf TreeNodes to FacilitySignals for %s:%s",
            total,
            facility,
            tree_name,
        )

        promoted = 0
        for offset in range(0, total, batch_size):
            batch_limit = min(batch_size, total - offset)
            result = gc.query(
                """
                MATCH (n:TreeNode {facility_id: $facility,
                                   tree_name: $tree})
                WHERE n.node_type IN ['NUMERIC', 'SIGNAL']
                  AND NOT EXISTS {
                      MATCH (n)<-[:SOURCE_NODE]-(:FacilitySignal)
                  }
                WITH n ORDER BY n.id LIMIT $batch_limit
                WITH n,
                     split(n.path, '::')[-1] AS node_path_raw
                WITH n, node_path_raw,
                     $facility + ':' +
                     $tree + '/' +
                     toLower(replace(node_path_raw, ':', '/'))
                     AS sig_id
                MERGE (s:FacilitySignal {id: sig_id})
                ON CREATE SET
                    s.facility_id = $facility,
                    s.status = 'discovered',
                    s.accessor = node_path_raw,
                    s.data_access = $da_id,
                    s.tree_name = $tree,
                    s.node_path = n.path,
                    s.unit = n.unit,
                    s.discovery_source = 'tree_traversal',
                    s.source_node = n.id,
                    s.discovered_at = datetime()
                WITH s, n
                MERGE (s)-[:SOURCE_NODE]->(n)
                WITH s
                MATCH (f:Facility {id: $facility})
                MERGE (s)-[:AT_FACILITY]->(f)
                WITH s
                MATCH (da:DataAccess {id: $da_id})
                MERGE (s)-[:DATA_ACCESS]->(da)
                RETURN count(s) AS promoted
                """,
                facility=facility,
                tree=tree_name,
                da_id=da_id,
                batch_limit=batch_limit,
            )
            batch_count = result[0]["promoted"] if result else 0
            promoted += batch_count
            logger.debug(
                "Promote batch %d: %d signals (%d total)",
                offset // batch_size + 1,
                batch_count,
                promoted,
            )

        logger.info(
            "Promoted %d FacilitySignals for %s:%s",
            promoted,
            facility,
            tree_name,
        )
        return promoted
