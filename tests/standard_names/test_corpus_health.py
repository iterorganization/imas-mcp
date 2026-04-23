"""Corpus-health gate test suite for standard name rotations.

Plan 31 §G.1 — runs against a live, populated Neo4j graph.
NOT part of default CI; invoke with:

    uv run pytest -m corpus_health -v

If the graph is unreachable or contains fewer than 50 StandardName
nodes, all tests are skipped with a clear diagnostic message.
"""

from __future__ import annotations

import statistics

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _connect_and_count():
    """Return (GraphClient instance, sn_count) or raise on connection failure."""
    from imas_codex.graph.client import GraphClient

    gc = GraphClient()
    rows = gc.query("MATCH (sn:StandardName) RETURN count(sn) AS n")
    return gc, rows[0]["n"]


@pytest.fixture(scope="module")
def gc():
    """Module-scoped GraphClient; skip the entire module if unavailable."""
    try:
        client, count = _connect_and_count()
    except Exception as exc:
        pytest.skip(f"Neo4j graph unreachable: {exc}")
    if count < 50:
        pytest.skip(
            f"Graph has only {count} StandardName nodes (<50); "
            "populate via a bootstrap rotation before running corpus_health tests."
        )
    yield client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QUARANTINE_QUERY = """
MATCH (sn:StandardName)
WITH count(sn) AS total,
     sum(CASE WHEN sn.validation_status = 'quarantined' THEN 1 ELSE 0 END) AS q
RETURN total, q,
       CASE WHEN total = 0 THEN 0.0
            ELSE toFloat(q) / toFloat(total)
       END AS rate
"""

_DESCRIPTION_COVERAGE_QUERY = """
MATCH (sn:StandardName)
WITH count(sn) AS total,
     sum(CASE WHEN nullIf(coalesce(sn.description, ''), '') IS NOT NULL
              THEN 1 ELSE 0 END) AS has_desc
RETURN total, has_desc,
       CASE WHEN total = 0 THEN 0.0
            ELSE toFloat(has_desc) / toFloat(total)
       END AS coverage
"""

_DESCRIPTION_LENGTHS_QUERY = """
MATCH (sn:StandardName)
WHERE nullIf(coalesce(sn.description, ''), '') IS NOT NULL
RETURN size(sn.description) AS len
ORDER BY len
"""

_REVIEWER_SCORES_QUERY = """
MATCH (sn:StandardName)
WHERE sn.reviewer_score_name IS NOT NULL
RETURN sn.reviewer_score_name AS score
"""

_PULSE_SCHEDULE_REF_QUERY = """
MATCH (sn:StandardName)<-[:PRODUCED_NAME]-(src:StandardNameSource)
WHERE src.source_type = 'dd'
  AND src.source_id STARTS WITH 'pulse_schedule/'
  AND src.source_id CONTAINS '/reference'
RETURN sn.id AS name, src.source_id AS path
LIMIT 10
"""

_BAD_COEFFICIENT_NAMES_QUERY = """
MATCH (sn:StandardName)
WHERE sn.id CONTAINS '_ggd_coefficients'
   OR sn.id CONTAINS '_finite_element_coefficients'
RETURN sn.id AS name
LIMIT 10
"""

_DIAMAGNETIC_NAMES_QUERY = """
MATCH (sn:StandardName)
WHERE sn.id STARTS WITH 'diamagnetic_component_of_'
RETURN sn.id AS name
LIMIT 10
"""

_COORD_PREFIX_ERROR_QUERY = """
MATCH (sn:StandardName)
WHERE any(issue IN coalesce(sn.validation_issues, [])
          WHERE issue CONTAINS 'coordinate_prefix')
RETURN sn.id AS name
LIMIT 10
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.corpus_health
class TestCorpusHealth:
    """Quantitative gates for a post-rotation standard name corpus."""

    def test_quarantine_rate(self, gc):
        """Quarantine rate ≤ 5%."""
        rows = gc.query(_QUARANTINE_QUERY)
        row = rows[0]
        rate = row["rate"]
        assert rate <= 0.05, (
            f"Quarantine rate {rate:.1%} exceeds 5% threshold "
            f"({row['q']}/{row['total']} quarantined)"
        )

    def test_description_coverage(self, gc):
        """Description coverage ≥ 90%."""
        rows = gc.query(_DESCRIPTION_COVERAGE_QUERY)
        row = rows[0]
        coverage = row["coverage"]
        assert coverage >= 0.90, (
            f"Description coverage {coverage:.1%} below 90% threshold "
            f"({row['has_desc']}/{row['total']} have descriptions)"
        )

    def test_description_length_median(self, gc):
        """Description length median ≥ 1200 chars."""
        rows = gc.query(_DESCRIPTION_LENGTHS_QUERY)
        if not rows:
            pytest.skip("No descriptions found — coverage test will catch this")
        lengths = [r["len"] for r in rows]
        median = statistics.median(lengths)
        assert median >= 1200, (
            f"Description length median {median:.0f} chars "
            f"below 1200 threshold (n={len(lengths)})"
        )

    def test_reviewer_score_mean(self, gc):
        """Reviewer score mean ≥ 0.80."""
        rows = gc.query(_REVIEWER_SCORES_QUERY)
        if not rows:
            pytest.skip("No reviewer scores found — run `sn review` first")
        scores = [r["score"] for r in rows]
        mean = statistics.mean(scores)
        assert mean >= 0.80, (
            f"Reviewer score mean {mean:.3f} below 0.80 threshold (n={len(scores)})"
        )

    def test_no_pulse_schedule_reference_sources(self, gc):
        """No pulse_schedule/.*/reference.* paths in SOURCED_FROM edges."""
        rows = gc.query(_PULSE_SCHEDULE_REF_QUERY)
        assert len(rows) == 0, (
            f"Found {len(rows)} names sourced from pulse_schedule/*/reference* paths: "
            + ", ".join(r["name"] for r in rows)
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
        """No pydantic coordinate_prefix validation errors."""
        rows = gc.query(_COORD_PREFIX_ERROR_QUERY)
        assert len(rows) == 0, (
            f"Found {len(rows)} names with coordinate_prefix validation errors: "
            + ", ".join(r["name"] for r in rows)
        )
