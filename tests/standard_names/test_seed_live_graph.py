"""Live-graph regression test for ``_list_physics_domains_with_extractable_paths``.

Caught a real production bug (smoke-3, 2026-05-04): the Cypher used
``(IDS)-[:HAS_PATH*]->(IMASNode)`` but the actual schema relationship is
``(IMASNode)-[:IN_IDS]->(IDS)``. Result: 0 domains seeded after ``sn clear``,
$10 smoke exited in 31s with 0 SNs created.

Mocked unit tests in ``test_seed_all_domains.py`` did not catch this because
they stub the function; this test hits the real graph.

Marked ``requires_graph`` — only runs when a Neo4j with DD content is available.
"""

from __future__ import annotations

import pytest

from imas_codex.graph.client import GraphClient

pytestmark = pytest.mark.requires_graph


def _has_dd_content() -> bool:
    try:
        with GraphClient() as gc:
            rows = list(
                gc.query(
                    "MATCH (n:IMASNode) WHERE n.physics_domain IS NOT NULL "
                    "RETURN count(n) AS c"
                )
            )
        return bool(rows) and rows[0].get("c", 0) > 0
    except Exception:
        return False


@pytest.mark.skipif(not _has_dd_content(), reason="DD-loaded graph not available")
def test_seed_query_finds_real_domains():
    """The Cypher used by ``_list_physics_domains_with_extractable_paths`` must
    return at least 10 domains against a DD-loaded graph.

    A previous bug used ``HAS_PATH*`` and returned 0 rows, silently breaking
    auto-seed.
    """
    from imas_codex.standard_names.loop import (
        _list_physics_domains_with_extractable_paths,
    )

    domains = _list_physics_domains_with_extractable_paths("dd")

    # DD has ~30 physics domains; a working query must return many of them.
    assert len(domains) >= 10, (
        f"Expected ≥10 domains from DD-loaded graph, got {len(domains)}: {domains}. "
        "Likely the IDS↔IMASNode relationship pattern is wrong "
        "(canonical: (n:IMASNode)-[:IN_IDS]->(ids:IDS))."
    )

    # Sanity: well-known domains must be present.
    expected_subset = {"equilibrium", "transport", "magnetohydrodynamics"}
    missing = expected_subset - set(domains)
    assert not missing, f"Expected domains missing: {missing} (got {len(domains)})"


@pytest.mark.skipif(not _has_dd_content(), reason="DD-loaded graph not available")
def test_seed_query_returns_empty_for_non_dd_source():
    from imas_codex.standard_names.loop import (
        _list_physics_domains_with_extractable_paths,
    )

    assert _list_physics_domains_with_extractable_paths("signals") == []
    assert _list_physics_domains_with_extractable_paths("manual") == []
