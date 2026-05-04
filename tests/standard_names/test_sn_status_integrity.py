"""Regression test for orphan-integrity false-positives on superseded SNs.

Bug: ``uv run imas-codex sn status`` reported 197 orphan StandardNames after
a $10 smoke.  Investigation showed ALL 197 had ``name_stage='superseded'`` —
they were predecessors in REFINED_FROM chains.  By design,
``persist_refined_name_batch`` migrates the PRODUCED_NAME edge from the
predecessor to the successor, so superseded SNs legitimately have no
PRODUCED_NAME edge and must not be counted as orphans.

Marked ``graph`` — only runs when a live Neo4j is available (auto-skipped
otherwise via the top-level conftest hook).
"""

from __future__ import annotations

import uuid

import pytest

from imas_codex.graph.client import GraphClient

pytestmark = pytest.mark.graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ORPHAN_SN_QUERY = """
MATCH (sn:StandardName)
WHERE sn._test_run_id = $run_id
RETURN
  count(CASE WHEN NOT (sn)<-[:PRODUCED_NAME]-()
              AND sn.model <> 'deterministic:dd_error_modifier'
              AND sn.name_stage <> 'superseded'
        THEN 1 END) AS orphan_sn
"""

_OLD_ORPHAN_SN_QUERY = """
MATCH (sn:StandardName)
WHERE sn._test_run_id = $run_id
RETURN
  count(CASE WHEN NOT (sn)<-[:PRODUCED_NAME]-()
              AND sn.model <> 'deterministic:dd_error_modifier'
        THEN 1 END) AS orphan_sn_old
"""


def _has_graph() -> bool:
    try:
        with GraphClient() as gc:
            gc.get_stats()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_graph(), reason="Neo4j not available")
def test_superseded_sn_not_counted_as_orphan():
    """A superseded predecessor in a REFINED_FROM chain must not be an orphan.

    Synthetic fixture:
        (:StandardNameSource {status:'composed'})
          -[:PRODUCED_NAME]->
        (sn_b:StandardName {name_stage:'accepted'})
          -[:REFINED_FROM]->
        (sn_a:StandardName {name_stage:'superseded'})

    sn_a has no outgoing PRODUCED_NAME (the edge was migrated to sn_b).
    The fixed query must report 0 orphans; the old query would report 1.
    """
    run_id = str(uuid.uuid4())

    create_cypher = """
    CREATE (src:StandardNameSource {
        id:            $src_id,
        name:          'test_quantity',
        status:        'composed',
        _test_run_id:  $run_id
    })
    CREATE (sn_a:StandardName {
        id:            $sn_a_id,
        name:          'test_quantity_v1',
        name_stage:    'superseded',
        model:         'gpt-4o',
        _test_run_id:  $run_id
    })
    CREATE (sn_b:StandardName {
        id:            $sn_b_id,
        name:          'test_quantity_v2',
        name_stage:    'accepted',
        model:         'gpt-4o',
        _test_run_id:  $run_id
    })
    CREATE (src)-[:PRODUCED_NAME]->(sn_b)
    CREATE (sn_b)-[:REFINED_FROM]->(sn_a)
    """

    cleanup_cypher = """
    MATCH (n)
    WHERE n._test_run_id = $run_id
    DETACH DELETE n
    """

    with GraphClient() as gc:
        try:
            gc.query(
                create_cypher,
                src_id=f"test_src_{run_id}",
                sn_a_id=f"test_sn_a_{run_id}",
                sn_b_id=f"test_sn_b_{run_id}",
                run_id=run_id,
            )

            # Fixed query: superseded sn_a must NOT be counted
            rows = list(gc.query(_ORPHAN_SN_QUERY, run_id=run_id))
            orphan_count = rows[0]["orphan_sn"] if rows else -1

            assert orphan_count == 0, (
                f"Expected 0 orphans (superseded SN excluded), got {orphan_count}. "
                "The orphan query must exclude name_stage='superseded' nodes."
            )

            # Sanity: old query (without the fix) would count sn_a as orphan
            old_rows = list(gc.query(_OLD_ORPHAN_SN_QUERY, run_id=run_id))
            old_orphan_count = old_rows[0]["orphan_sn_old"] if old_rows else -1

            assert old_orphan_count == 1, (
                f"Expected old query to count 1 orphan (sn_a has no PRODUCED_NAME), "
                f"got {old_orphan_count}. Fixture may not be set up correctly."
            )

        finally:
            gc.query(cleanup_cypher, run_id=run_id)
