"""Standard name health gate tests for post-rotation quality checks.

Runs against a live, populated Neo4j graph. NOT part of default CI;
invoke with:

    uv run pytest -m sn_health -v

All metrics are scoped to terminal-state names only:
- name_stage='accepted' for name-axis metrics
- docs_stage='accepted' for documentation-axis metrics
- name_stage IN ('accepted', 'exhausted') for quarantine/exclusion checks

If the graph is unreachable or contains fewer than 10 accepted
StandardName nodes, all tests are skipped.
"""

from __future__ import annotations

import statistics

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _connect_and_count_accepted():
    """Return (GraphClient, accepted_count) or raise on connection failure."""
    from imas_codex.graph.client import GraphClient

    gc = GraphClient()
    rows = gc.query(
        "MATCH (sn:StandardName {name_stage: 'accepted'}) RETURN count(sn) AS n"
    )
    return gc, rows[0]["n"]


@pytest.fixture(scope="module")
def gc():
    """Module-scoped GraphClient; skip the entire module if unavailable."""
    try:
        client, count = _connect_and_count_accepted()
    except Exception as exc:
        pytest.skip(f"Neo4j graph unreachable: {exc}")
    if count < 10:
        pytest.skip(
            f"Graph has only {count} accepted StandardName nodes (<10); "
            "populate via `sn run` before running sn_health tests."
        )
    yield client


# ---------------------------------------------------------------------------
# Queries — all scoped to terminal states
# ---------------------------------------------------------------------------

_QUARANTINE_RATE_QUERY = """
MATCH (sn:StandardName)
WHERE sn.name_stage IN ['accepted', 'exhausted']
WITH count(sn) AS total,
     sum(CASE WHEN sn.validation_status = 'quarantined' THEN 1 ELSE 0 END) AS q
RETURN total, q,
       CASE WHEN total = 0 THEN 0.0
            ELSE toFloat(q) / toFloat(total)
       END AS rate
"""

_DESCRIPTION_COVERAGE_QUERY = """
MATCH (sn:StandardName {name_stage: 'accepted'})
WITH count(sn) AS total,
     sum(CASE WHEN nullIf(coalesce(sn.description, ''), '') IS NOT NULL
              THEN 1 ELSE 0 END) AS has_desc
RETURN total, has_desc,
       CASE WHEN total = 0 THEN 0.0
            ELSE toFloat(has_desc) / toFloat(total)
       END AS coverage
"""

_DOCUMENTATION_LENGTHS_QUERY = """
MATCH (sn:StandardName {docs_stage: 'accepted'})
WHERE nullIf(coalesce(sn.documentation, ''), '') IS NOT NULL
RETURN size(sn.documentation) AS len
ORDER BY len
"""

_NAME_REVIEWER_SCORES_QUERY = """
MATCH (sn:StandardName {name_stage: 'accepted'})
WHERE sn.reviewer_score_name IS NOT NULL
RETURN sn.reviewer_score_name AS score
"""

_DOCS_REVIEWER_SCORES_QUERY = """
MATCH (sn:StandardName {docs_stage: 'accepted'})
WHERE sn.reviewer_score_docs IS NOT NULL
RETURN sn.reviewer_score_docs AS score
"""

_PULSE_SCHEDULE_REF_QUERY = """
MATCH (sn:StandardName {name_stage: 'accepted'})<-[:PRODUCED_NAME]-(src:StandardNameSource)
WHERE src.source_type = 'dd'
  AND src.source_id STARTS WITH 'pulse_schedule/'
  AND src.source_id CONTAINS '/reference'
RETURN sn.id AS name, src.source_id AS path
LIMIT 10
"""

_BAD_COEFFICIENT_NAMES_QUERY = """
MATCH (sn:StandardName {name_stage: 'accepted'})
WHERE sn.id CONTAINS '_ggd_coefficients'
   OR sn.id CONTAINS '_finite_element_coefficients'
RETURN sn.id AS name
LIMIT 10
"""

_DIAMAGNETIC_NAMES_QUERY = """
MATCH (sn:StandardName {name_stage: 'accepted'})
WHERE sn.id STARTS WITH 'diamagnetic_component_of_'
RETURN sn.id AS name
LIMIT 10
"""

_COORD_PREFIX_ERROR_QUERY = """
MATCH (sn:StandardName {name_stage: 'accepted'})
WHERE any(issue IN coalesce(sn.validation_issues, [])
          WHERE issue CONTAINS 'coordinate_prefix')
RETURN sn.id AS name
LIMIT 10
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.sn_health
class TestStandardNameHealth:
    """Quantitative gates for a post-rotation standard name corpus.

    All metrics are scoped to terminal-state names — names still in
    drafted/reviewed/refining are in-flight and excluded.
    """

    def test_quarantine_rate(self, gc):
        """Quarantine rate among terminal names ≤ 5%."""
        rows = gc.query(_QUARANTINE_RATE_QUERY)
        row = rows[0]
        rate = row["rate"]
        assert rate <= 0.05, (
            f"Quarantine rate {rate:.1%} exceeds 5% threshold "
            f"({row['q']}/{row['total']} quarantined among terminal names)"
        )

    def test_description_coverage(self, gc):
        """All accepted names have a description."""
        rows = gc.query(_DESCRIPTION_COVERAGE_QUERY)
        row = rows[0]
        coverage = row["coverage"]
        assert coverage >= 0.95, (
            f"Description coverage {coverage:.1%} below 95% threshold "
            f"({row['has_desc']}/{row['total']} accepted names have descriptions)"
        )

    def test_documentation_length_median(self, gc):
        """Documentation (not description) length median ≥ 800 chars among docs-accepted names."""
        rows = gc.query(_DOCUMENTATION_LENGTHS_QUERY)
        if not rows:
            pytest.skip("No docs-accepted names with documentation found")
        lengths = [r["len"] for r in rows]
        median = statistics.median(lengths)
        assert median >= 800, (
            f"Documentation length median {median:.0f} chars "
            f"below 800 threshold (n={len(lengths)})"
        )

    def test_name_reviewer_score_mean(self, gc):
        """Name-axis reviewer score mean ≥ 0.75 among accepted names."""
        rows = gc.query(_NAME_REVIEWER_SCORES_QUERY)
        if not rows:
            pytest.skip("No name-axis reviewer scores on accepted names")
        scores = [r["score"] for r in rows]
        mean = statistics.mean(scores)
        assert mean >= 0.75, (
            f"Name reviewer score mean {mean:.3f} below 0.75 threshold "
            f"(n={len(scores)})"
        )

    def test_docs_reviewer_score_mean(self, gc):
        """Docs-axis reviewer score mean ≥ 0.75 among docs-accepted names."""
        rows = gc.query(_DOCS_REVIEWER_SCORES_QUERY)
        if not rows:
            pytest.skip("No docs-axis reviewer scores on docs-accepted names")
        scores = [r["score"] for r in rows]
        mean = statistics.mean(scores)
        assert mean >= 0.75, (
            f"Docs reviewer score mean {mean:.3f} below 0.75 threshold "
            f"(n={len(scores)})"
        )

    def test_no_pulse_schedule_reference_sources(self, gc):
        """No pulse_schedule/.*/reference.* paths in source edges."""
        rows = gc.query(_PULSE_SCHEDULE_REF_QUERY)
        assert len(rows) == 0, (
            f"Found {len(rows)} accepted names sourced from "
            "pulse_schedule/*/reference* paths: " + ", ".join(r["name"] for r in rows)
        )

    def test_no_coefficient_names(self, gc):
        """No *_ggd_coefficients / _finite_element_coefficients_* names."""
        rows = gc.query(_BAD_COEFFICIENT_NAMES_QUERY)
        assert len(rows) == 0, (
            f"Found {len(rows)} coefficient names that should be excluded: "
            + ", ".join(r["name"] for r in rows)
        )

    def test_no_diamagnetic_component_names(self, gc):
        """No diamagnetic_component_of_* names."""
        rows = gc.query(_DIAMAGNETIC_NAMES_QUERY)
        assert len(rows) == 0, (
            f"Found {len(rows)} diamagnetic_component_of_* names: "
            + ", ".join(r["name"] for r in rows)
        )

    def test_no_coordinate_prefix_errors(self, gc):
        """No coordinate_prefix validation errors on accepted names."""
        rows = gc.query(_COORD_PREFIX_ERROR_QUERY)
        assert len(rows) == 0, (
            f"Found {len(rows)} names with coordinate_prefix validation errors: "
            + ", ".join(r["name"] for r in rows)
        )
