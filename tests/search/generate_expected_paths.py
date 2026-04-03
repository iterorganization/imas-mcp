"""Auto-generate expanded expected paths from the IMAS DD graph.

Queries semantic clusters and path name matching to discover all valid
paths matching a benchmark concept, replacing hand-curated lists of 2-5
paths with comprehensive sets of 10-36 paths.

Usage:
    # From conftest fixture (auto-cached):
    expanded = expanded_expected_paths[query.query_text]

    # Standalone:
    from tests.search.generate_expected_paths import generate_expected_paths
    paths = generate_expected_paths(query, gc)
"""

from __future__ import annotations

import logging

from imas_codex.graph.client import GraphClient
from tests.search.benchmark_data import BenchmarkQuery

logger = logging.getLogger(__name__)


def generate_expected_paths(
    query: BenchmarkQuery,
    gc: GraphClient,
) -> set[str]:
    """Query the graph for all valid paths matching a benchmark concept.

    Strategy:
    1. Find global clusters whose labels match query text (fuzzy)
    2. Collect all cluster member paths
    3. Find paths whose terminal segment matches any expected path terminal
    4. Union with hand-curated expected_paths
    5. Filter to data-category paths (exclude /data, /time subpaths unless
       they appear in the hand-curated set)
    """
    expanded = set(query.expected_paths)

    # --- Strategy 1: Cluster member expansion ---
    # Find clusters matching query terms
    query_terms = [t.lower() for t in query.query_text.split() if len(t) > 2]
    if query_terms:
        # Use CONTAINS matching on cluster labels
        where_clauses = " OR ".join(
            f"toLower(cl.label) CONTAINS ${f'term{i}'}" for i in range(len(query_terms))
        )
        params = {f"term{i}": t for i, t in enumerate(query_terms)}

        cluster_members = gc.query(
            f"""
            MATCH (cl:IMASSemanticCluster)
            WHERE cl.scope = 'global' AND ({where_clauses})
            MATCH (cl)<-[:IN_CLUSTER]-(member:IMASNode)
            WHERE member.node_category = 'data'
              AND NOT (member)-[:DEPRECATED_IN]->(:DDVersion)
            RETURN DISTINCT member.id AS id
            """,
            **params,
        )
        for r in cluster_members or []:
            expanded.add(r["id"])

    # --- Strategy 2: Terminal segment matching ---
    # Find paths whose terminal segment matches any expected path terminal
    expected_terminals = set()
    for ep in query.expected_paths:
        terminal = ep.rsplit("/", 1)[-1]
        if terminal and len(terminal) > 1:
            expected_terminals.add(terminal)

    if expected_terminals:
        terminal_matches = gc.query(
            """
            UNWIND $terminals AS term
            MATCH (n:IMASNode)
            WHERE n.name = term
              AND n.node_category = 'data'
              AND NOT (n)-[:DEPRECATED_IN]->(:DDVersion)
            RETURN DISTINCT n.id AS id
            """,
            terminals=list(expected_terminals),
        )
        for r in terminal_matches or []:
            expanded.add(r["id"])

    # --- Strategy 3: Abbreviation expansion for short queries ---
    # If query looks like an abbreviation (1-4 chars, or matches known patterns)
    query_stripped = query.query_text.strip().lower()
    if len(query_stripped) <= 5 or "_" in query_stripped:
        abbrev_matches = gc.query(
            """
            MATCH (n:IMASNode)
            WHERE n.name = $abbrev
              AND n.node_category = 'data'
              AND NOT (n)-[:DEPRECATED_IN]->(:DDVersion)
            RETURN DISTINCT n.id AS id
            """,
            abbrev=query_stripped,
        )
        for r in abbrev_matches or []:
            expanded.add(r["id"])

    # --- Filter: remove generic sub-paths unless hand-curated ---
    _GENERIC_SUFFIXES = {"/time", "/validity/time", "/validity/value"}
    filtered = set()
    for path in expanded:
        # Keep hand-curated paths unconditionally
        if path in query.expected_paths:
            filtered.add(path)
            continue
        # Skip generic leaf paths
        if any(path.endswith(s) for s in _GENERIC_SUFFIXES):
            continue
        filtered.add(path)

    return filtered


def generate_all_expected_paths(
    queries: list[BenchmarkQuery],
    gc: GraphClient,
) -> dict[str, set[str]]:
    """Generate expanded expected paths for all benchmark queries.

    Returns dict mapping query_text -> expanded expected path set.
    """
    result = {}
    for q in queries:
        try:
            result[q.query_text] = generate_expected_paths(q, gc)
        except Exception:
            logger.warning("Failed to expand paths for %r", q.query_text, exc_info=True)
            result[q.query_text] = set(q.expected_paths)
    return result
