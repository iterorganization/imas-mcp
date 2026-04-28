"""Physics-domain priority derived from cluster mapping relevance.

A StandardName may span multiple physics domains. When choosing a
single "primary" domain (e.g. for catalog YAML file path), we want
to surface the most important domain rather than picking
alphabetically.

The existing notion of importance lives at the cluster level via
``Cluster.mapping_relevance`` (``high`` / ``medium`` / ``low``).
``Cluster.physics_domain`` carries the primary physics domain of the
cluster's members. Domain importance therefore lifts naturally from
cluster importance: a domain whose clusters are mostly HIGH-relevance
scores higher than one whose clusters are mostly LOW.

Weights ``high=100, medium=10, low=1`` make HIGH dominate by a factor
of 10× over MEDIUM and 100× over LOW, so a single HIGH cluster
outranks ten MEDIUM clusters. This matches the semantic intent of
the relevance tiers.

The result is cached for the lifetime of the process — the DD does
not change during a single ``sn run`` invocation.
"""

from __future__ import annotations

import functools
import logging

from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def get_domain_priority_index() -> dict[str, int]:
    """Return rank (0-based, lower = more important) per physics domain.

    Computed once per process from weighted ``Cluster.mapping_relevance``
    counts. Domains absent from the index get rank ``999``.

    If the graph is unreachable or no clusters carry the required
    fields, return an empty dict — callers fall back to alphabetical.
    """
    cypher = """
        MATCH (c:Cluster)
        WHERE c.physics_domain IS NOT NULL
          AND c.mapping_relevance IS NOT NULL
        WITH c.physics_domain AS domain,
             CASE c.mapping_relevance
               WHEN 'high'   THEN 100
               WHEN 'medium' THEN 10
               WHEN 'low'    THEN 1
               ELSE 0
             END AS weight
        RETURN domain, sum(weight) AS importance
        ORDER BY importance DESC, domain ASC
    """
    try:
        with GraphClient() as gc:
            rows = list(gc.query(cypher))
    except Exception as exc:  # pragma: no cover — graph unreachable
        logger.warning(
            "Domain priority index unavailable; falling back to "
            "alphabetical (reason: %s)",
            exc,
        )
        return {}

    return {row["domain"]: rank for rank, row in enumerate(rows)}


def pick_primary_domain(domains: list[str]) -> str:
    """Pick the highest-priority domain from a list.

    Sort key is ``(rank, domain)`` so unranked domains land at the end
    and ties break alphabetically for determinism.
    """
    if not domains:
        raise ValueError("pick_primary_domain requires a non-empty list")
    ranks = get_domain_priority_index()
    return sorted(domains, key=lambda d: (ranks.get(d, 999), d))[0]


def reset_cache() -> None:
    """Clear the cached priority index. Used by tests."""
    get_domain_priority_index.cache_clear()


def domain_key(value, fallback: str = "unknown") -> str:
    """Coerce a possibly-list ``physics_domain`` value to a single grouping key.

    Used by consolidation/audit/grouping code that pre-dates the
    multivalued ``physics_domain`` schema. Returns ``fallback`` for
    None/empty inputs. For lists, picks the highest-priority domain via
    :func:`pick_primary_domain`, falling back to alphabetical-first
    when the priority index is unavailable.
    """
    if value is None:
        return fallback
    if isinstance(value, str):
        s = value.strip()
        return s if s else fallback
    if isinstance(value, list | tuple):
        valid = [d.strip() for d in value if isinstance(d, str) and d.strip()]
        if not valid:
            return fallback
        try:
            return pick_primary_domain(valid)
        except Exception:  # pragma: no cover — defensive
            return sorted(valid)[0]
    return fallback


def domain_list(value) -> list[str]:
    """Coerce a possibly-scalar ``physics_domain`` value to a list of strings.

    Returns ``[]`` for None/empty. Trims whitespace and drops empty items.
    """
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, list | tuple):
        return [d.strip() for d in value if isinstance(d, str) and d.strip()]
    return []
