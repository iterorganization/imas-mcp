"""Pre-run coverage report for the standard name pipeline.

``compute_coverage`` queries the graph and returns a :class:`CoverageReport`
that answers "how many names do we expect to mint?" before any LLM spend.

Typical use::

    report = compute_coverage()
    print(report.to_json())

or via the CLI::

    imas-codex sn coverage [--physics-domain equilibrium] [--json]
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from imas_codex.core.node_categories import SN_SOURCE_CATEGORIES
from imas_codex.graph.client import GraphClient

# ---------------------------------------------------------------------------
# Cypher queries
# ---------------------------------------------------------------------------

# Total eligible leaf count + breakdown by node_category, physics_domain, node_type
_ELIGIBLE_TOTAL_QUERY = """
MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
WHERE n.node_category IN $sn_categories
  AND n.description IS NOT NULL
  AND n.description <> ''
  AND ids.id <> 'core_instant_changes'
  AND NOT (n.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
{domain_clause}
RETURN count(n) AS total
"""

_ELIGIBLE_BY_CATEGORY_QUERY = """
MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
WHERE n.node_category IN $sn_categories
  AND n.description IS NOT NULL
  AND n.description <> ''
  AND ids.id <> 'core_instant_changes'
  AND NOT (n.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
{domain_clause}
RETURN coalesce(n.node_category, 'none') AS bucket, count(n) AS cnt
ORDER BY cnt DESC
"""

_ELIGIBLE_BY_DOMAIN_QUERY = """
MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
WHERE n.node_category IN $sn_categories
  AND n.description IS NOT NULL
  AND n.description <> ''
  AND ids.id <> 'core_instant_changes'
  AND NOT (n.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
{domain_clause}
RETURN coalesce(n.physics_domain, 'none') AS bucket, count(n) AS cnt
ORDER BY cnt DESC
"""

_ELIGIBLE_BY_NODE_TYPE_QUERY = """
MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
WHERE n.node_category IN $sn_categories
  AND n.description IS NOT NULL
  AND n.description <> ''
  AND ids.id <> 'core_instant_changes'
  AND NOT (n.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
{domain_clause}
RETURN coalesce(n.node_type, 'none') AS bucket, count(n) AS cnt
ORDER BY cnt DESC
"""

_ELIGIBLE_WITH_ERRORS_QUERY = """
MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
WHERE n.node_category IN $sn_categories
  AND n.description IS NOT NULL
  AND n.description <> ''
  AND ids.id <> 'core_instant_changes'
  AND NOT (n.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
  AND EXISTS {{ (n)-[:HAS_ERROR]->(:IMASNode) }}
{domain_clause}
RETURN count(n) AS cnt
"""

# StandardName node coverage queries
_SN_TOTAL_QUERY = """
MATCH (sn:StandardName)
RETURN count(sn) AS total
"""

_SN_BY_PIPELINE_STATUS_QUERY = """
MATCH (sn:StandardName)
RETURN coalesce(sn.pipeline_status, 'none') AS bucket, count(sn) AS cnt
ORDER BY cnt DESC
"""

_SN_BY_VALIDATION_STATUS_QUERY = """
MATCH (sn:StandardName)
RETURN coalesce(sn.validation_status, 'none') AS bucket, count(sn) AS cnt
ORDER BY cnt DESC
"""

_SN_COVERED_PARENTS_QUERY = """
MATCH (n:IMASNode)-[:HAS_STANDARD_NAME]->(:StandardName)
RETURN count(DISTINCT n) AS cnt
"""

_SN_ERROR_SIBLINGS_QUERY = """
MATCH (sn:StandardName)
WHERE sn.model = 'deterministic:dd_error_modifier'
RETURN count(sn) AS cnt
"""

# Work remaining: eligible leaves without any HAS_STANDARD_NAME link
_UNCOVERED_TOTAL_QUERY = """
MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
WHERE n.node_category IN $sn_categories
  AND n.description IS NOT NULL
  AND n.description <> ''
  AND ids.id <> 'core_instant_changes'
  AND NOT (n.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
  AND NOT EXISTS {{ MATCH (n)-[:HAS_STANDARD_NAME]->(:StandardName) }}
{domain_clause}
RETURN count(n) AS cnt
"""

_UNCOVERED_WITH_ERRORS_QUERY = """
MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
WHERE n.node_category IN $sn_categories
  AND n.description IS NOT NULL
  AND n.description <> ''
  AND ids.id <> 'core_instant_changes'
  AND NOT (n.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
  AND NOT EXISTS {{ MATCH (n)-[:HAS_STANDARD_NAME]->(:StandardName) }}
  AND EXISTS {{ (n)-[:HAS_ERROR]->(:IMASNode) }}
{domain_clause}
RETURN count(n) AS cnt
"""

# SNRun telemetry: avg cost_per_name from completed/budget_exhausted runs
_SNRUN_COST_QUERY = """
MATCH (rr:SNRun)
WHERE rr.names_composed IS NOT NULL
  AND rr.names_composed > 0
  AND rr.cost_spent IS NOT NULL
  AND rr.cost_spent > 0
  AND rr.stop_reason IN ['budget_exhausted', 'completed', 'budget_reached']
RETURN rr.cost_spent AS cost_spent, rr.names_composed AS names_composed
LIMIT 20
"""


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CoverageReport:
    """Coverage report for the standard name pipeline.

    All counts reflect the B3' leaf invariant:
    ``node_category IN SN_SOURCE_CATEGORIES AND
    data_type NOT IN ['STRUCTURE','STRUCT_ARRAY']``.
    """

    # -- DD eligibility -------------------------------------------------------
    eligible_total: int
    """Total eligible leaf nodes (all physics domains)."""

    eligible_by_category: dict[str, int]
    """Eligible count keyed by node_category (quantity/geometry/coordinate)."""

    eligible_by_domain: dict[str, int]
    """Eligible count keyed by physics_domain."""

    eligible_by_node_type: dict[str, int]
    """Eligible count keyed by node_type (dynamic/static/constant/none)."""

    eligible_with_errors: int
    """Eligible leaves that have at least one HAS_ERROR edge (B9 parents)."""

    # -- Already-minted -------------------------------------------------------
    sn_total: int
    """Total StandardName nodes in the graph."""

    sn_by_pipeline_status: dict[str, int]
    """StandardName count keyed by pipeline_status."""

    sn_by_validation_status: dict[str, int]
    """StandardName count keyed by validation_status."""

    covered_parents: int
    """IMASNodes already linked via HAS_STANDARD_NAME."""

    error_siblings_minted: int
    """Error-sibling names with model='deterministic:dd_error_modifier'."""

    # -- Work remaining -------------------------------------------------------
    to_compose: int
    """Eligible leaves without any HAS_STANDARD_NAME link."""

    to_compose_with_errors: int
    """Subset of ``to_compose`` that have HAS_ERROR edges (B9 eligible)."""

    expected_error_siblings: int
    """Estimated new error-sibling names: 3 × ``to_compose_with_errors``."""

    cost_per_name: float | None
    """Avg USD per name from prior SNRun telemetry; None if no data."""

    estimated_compose_cost: float | None
    """Rough estimate: ``cost_per_name × to_compose``; None if no telemetry."""

    # -- Filter context -------------------------------------------------------
    physics_domain_filter: str | None
    """Physics domain filter applied, or None for all domains."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Core compute function
# ---------------------------------------------------------------------------


def compute_coverage(physics_domain: str | None = None) -> CoverageReport:
    """Query the graph and return a :class:`CoverageReport`.

    Args:
        physics_domain: Restrict eligibility counts to this physics domain.
            When None, all domains are included.

    Returns:
        Populated :class:`CoverageReport` dataclass.  All graph queries are
        executed inside a single :class:`~imas_codex.graph.client.GraphClient`
        context.
    """
    params: dict[str, Any] = {"sn_categories": list(SN_SOURCE_CATEGORIES)}
    domain_clause = ""
    if physics_domain:
        domain_clause = "  AND n.physics_domain = $physics_domain"
        params["physics_domain"] = physics_domain

    def _inject_domain(query: str) -> str:
        return query.format(domain_clause=domain_clause)

    with GraphClient() as gc:
        # -- Eligible totals -------------------------------------------------
        eligible_total = _one(
            gc.query(_inject_domain(_ELIGIBLE_TOTAL_QUERY), **params), "total"
        )
        eligible_by_category = _bucket(
            gc.query(_inject_domain(_ELIGIBLE_BY_CATEGORY_QUERY), **params)
        )
        eligible_by_domain = _bucket(
            gc.query(_inject_domain(_ELIGIBLE_BY_DOMAIN_QUERY), **params)
        )
        eligible_by_node_type = _bucket(
            gc.query(_inject_domain(_ELIGIBLE_BY_NODE_TYPE_QUERY), **params)
        )
        eligible_with_errors = _one(
            gc.query(_inject_domain(_ELIGIBLE_WITH_ERRORS_QUERY), **params), "cnt"
        )

        # -- Already-minted (not domain-filtered: always total catalog) ------
        sn_total = _one(gc.query(_SN_TOTAL_QUERY), "total")
        sn_by_pipeline_status = _bucket(gc.query(_SN_BY_PIPELINE_STATUS_QUERY))
        sn_by_validation_status = _bucket(gc.query(_SN_BY_VALIDATION_STATUS_QUERY))
        covered_parents = _one(gc.query(_SN_COVERED_PARENTS_QUERY), "cnt")
        error_siblings_minted = _one(gc.query(_SN_ERROR_SIBLINGS_QUERY), "cnt")

        # -- Work remaining --------------------------------------------------
        to_compose = _one(
            gc.query(_inject_domain(_UNCOVERED_TOTAL_QUERY), **params), "cnt"
        )
        to_compose_with_errors = _one(
            gc.query(_inject_domain(_UNCOVERED_WITH_ERRORS_QUERY), **params), "cnt"
        )

        # -- Cost telemetry --------------------------------------------------
        run_rows = list(gc.query(_SNRUN_COST_QUERY))

    cost_per_name: float | None = None
    if run_rows:
        total_cost = sum(float(r["cost_spent"] or 0) for r in run_rows)
        total_names = sum(int(r["names_composed"] or 0) for r in run_rows)
        if total_names > 0:
            cost_per_name = total_cost / total_names

    estimated_compose_cost: float | None = None
    if cost_per_name is not None and to_compose > 0:
        estimated_compose_cost = cost_per_name * to_compose

    return CoverageReport(
        eligible_total=eligible_total,
        eligible_by_category=eligible_by_category,
        eligible_by_domain=eligible_by_domain,
        eligible_by_node_type=eligible_by_node_type,
        eligible_with_errors=eligible_with_errors,
        sn_total=sn_total,
        sn_by_pipeline_status=sn_by_pipeline_status,
        sn_by_validation_status=sn_by_validation_status,
        covered_parents=covered_parents,
        error_siblings_minted=error_siblings_minted,
        to_compose=to_compose,
        to_compose_with_errors=to_compose_with_errors,
        expected_error_siblings=3 * to_compose_with_errors,
        cost_per_name=cost_per_name,
        estimated_compose_cost=estimated_compose_cost,
        physics_domain_filter=physics_domain,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _one(rows: Any, key: str) -> int:
    """Return the first row's ``key`` as int, defaulting to 0."""
    row = next(iter(rows), None)
    if row is None:
        return 0
    val = row.get(key)
    return int(val) if val is not None else 0


def _bucket(rows: Any) -> dict[str, int]:
    """Convert ``(bucket, cnt)`` rows to a plain dict."""
    result: dict[str, int] = {}
    for row in rows:
        bucket = str(row.get("bucket") or "none")
        result[bucket] = int(row.get("cnt") or 0)
    return result
