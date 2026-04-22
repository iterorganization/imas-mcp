"""Tests for the sn coverage pre-run report (rc22 B4).

All tests mock GraphClient.query so no live graph is required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest
from click.testing import CliRunner

from imas_codex.standard_names.coverage import (
    CoverageReport,
    _bucket,
    _one,
    compute_coverage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gc_side_effects(overrides: dict[int, list] | None = None) -> list:
    """Return a list of 12 empty query results (one per query in compute_coverage).

    Pass ``overrides={idx: rows}`` to substitute specific queries.
    """
    defaults: list = [[] for _ in range(12)]
    # query 0: eligible total -> [{total: 0}]
    defaults[0] = [{"total": 0}]
    # query 1: by_category -> []
    # query 2: by_domain -> []
    # query 3: by_node_type -> []
    # query 4: with_errors total -> [{cnt: 0}]
    defaults[4] = [{"cnt": 0}]
    # query 5: sn total -> [{total: 0}]
    defaults[5] = [{"total": 0}]
    # query 6: sn_by_pipeline_status -> []
    # query 7: sn_by_validation_status -> []
    # query 8: covered_parents -> [{cnt: 0}]
    defaults[8] = [{"cnt": 0}]
    # query 9: error_siblings_minted -> [{cnt: 0}]
    defaults[9] = [{"cnt": 0}]
    # query 10: to_compose -> [{cnt: 0}]
    defaults[10] = [{"cnt": 0}]
    # query 11: to_compose_with_errors -> [{cnt: 0}]
    defaults[11] = [{"cnt": 0}]
    # query 12: SNRun cost telemetry -> []
    defaults.append([])

    if overrides:
        for idx, rows in overrides.items():
            defaults[idx] = rows

    return defaults


def _mock_gc(side_effects: list):
    """Return a patched GraphClient that serves the given side_effects list."""
    gc = MagicMock()
    gc.query = MagicMock(side_effect=side_effects)
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    return gc


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_one_returns_int(self):
        assert _one([{"total": 42}], "total") == 42

    def test_one_empty_returns_zero(self):
        assert _one([], "total") == 0

    def test_one_none_value_returns_zero(self):
        assert _one([{"total": None}], "total") == 0

    def test_bucket_converts_rows(self):
        rows = [{"bucket": "quantity", "cnt": 5}, {"bucket": "geometry", "cnt": 3}]
        result = _bucket(rows)
        assert result == {"quantity": 5, "geometry": 3}

    def test_bucket_handles_none_bucket(self):
        rows = [{"bucket": None, "cnt": 2}]
        result = _bucket(rows)
        assert result == {"none": 2}


# ---------------------------------------------------------------------------
# compute_coverage tests
# ---------------------------------------------------------------------------


class TestComputeCoverageEmpty:
    """No nodes, no names, no runs → structured zeros."""

    def test_returns_coverage_report(self):
        effects = _make_gc_side_effects()
        gc = _mock_gc(effects)

        with patch("imas_codex.standard_names.coverage.GraphClient", return_value=gc):
            report = compute_coverage()

        assert isinstance(report, CoverageReport)

    def test_all_zeros(self):
        effects = _make_gc_side_effects()
        gc = _mock_gc(effects)

        with patch("imas_codex.standard_names.coverage.GraphClient", return_value=gc):
            report = compute_coverage()

        assert report.eligible_total == 0
        assert report.sn_total == 0
        assert report.to_compose == 0
        assert report.covered_parents == 0
        assert report.error_siblings_minted == 0

    def test_cost_per_name_is_none(self):
        effects = _make_gc_side_effects()
        gc = _mock_gc(effects)

        with patch("imas_codex.standard_names.coverage.GraphClient", return_value=gc):
            report = compute_coverage()

        assert report.cost_per_name is None
        assert report.estimated_compose_cost is None

    def test_physics_domain_filter_stored(self):
        effects = _make_gc_side_effects()
        gc = _mock_gc(effects)

        with patch("imas_codex.standard_names.coverage.GraphClient", return_value=gc):
            report = compute_coverage(physics_domain="magnetics")

        assert report.physics_domain_filter == "magnetics"

    def test_no_domain_filter_is_none(self):
        effects = _make_gc_side_effects()
        gc = _mock_gc(effects)

        with patch("imas_codex.standard_names.coverage.GraphClient", return_value=gc):
            report = compute_coverage()

        assert report.physics_domain_filter is None


class TestComputeCoverageCountsEligibleLeaves:
    """Seed minimal graph: 5 leaves, 2 with errors, 1 already covered."""

    def _build_effects(self):
        return _make_gc_side_effects(
            overrides={
                0: [{"total": 5}],  # eligible total
                1: [  # by_category
                    {"bucket": "quantity", "cnt": 3},
                    {"bucket": "geometry", "cnt": 1},
                    {"bucket": "coordinate", "cnt": 1},
                ],
                2: [  # by_domain
                    {"bucket": "equilibrium", "cnt": 4},
                    {"bucket": "magnetics", "cnt": 1},
                ],
                3: [  # by_node_type
                    {"bucket": "dynamic", "cnt": 4},
                    {"bucket": "static", "cnt": 1},
                ],
                4: [{"cnt": 2}],  # with_errors
                5: [{"total": 1}],  # sn total
                8: [{"cnt": 1}],  # covered_parents
                9: [{"cnt": 0}],  # error_siblings_minted
                10: [{"cnt": 4}],  # to_compose (5 - 1)
                11: [{"cnt": 1}],  # to_compose_with_errors
            }
        )

    def test_eligible_total(self):
        gc = _mock_gc(self._build_effects())
        with patch("imas_codex.standard_names.coverage.GraphClient", return_value=gc):
            report = compute_coverage()
        assert report.eligible_total == 5

    def test_eligible_with_errors(self):
        gc = _mock_gc(self._build_effects())
        with patch("imas_codex.standard_names.coverage.GraphClient", return_value=gc):
            report = compute_coverage()
        assert report.eligible_with_errors == 2

    def test_covered_parents(self):
        gc = _mock_gc(self._build_effects())
        with patch("imas_codex.standard_names.coverage.GraphClient", return_value=gc):
            report = compute_coverage()
        assert report.covered_parents == 1

    def test_to_compose(self):
        gc = _mock_gc(self._build_effects())
        with patch("imas_codex.standard_names.coverage.GraphClient", return_value=gc):
            report = compute_coverage()
        assert report.to_compose == 4

    def test_to_compose_with_errors(self):
        gc = _mock_gc(self._build_effects())
        with patch("imas_codex.standard_names.coverage.GraphClient", return_value=gc):
            report = compute_coverage()
        assert report.to_compose_with_errors == 1

    def test_expected_error_siblings(self):
        gc = _mock_gc(self._build_effects())
        with patch("imas_codex.standard_names.coverage.GraphClient", return_value=gc):
            report = compute_coverage()
        assert report.expected_error_siblings == 3  # 3 × 1

    def test_eligible_by_category(self):
        gc = _mock_gc(self._build_effects())
        with patch("imas_codex.standard_names.coverage.GraphClient", return_value=gc):
            report = compute_coverage()
        assert report.eligible_by_category["quantity"] == 3
        assert report.eligible_by_category["geometry"] == 1
        assert report.eligible_by_category["coordinate"] == 1

    def test_eligible_by_node_type(self):
        gc = _mock_gc(self._build_effects())
        with patch("imas_codex.standard_names.coverage.GraphClient", return_value=gc):
            report = compute_coverage()
        assert report.eligible_by_node_type["dynamic"] == 4
        assert report.eligible_by_node_type["static"] == 1


class TestComputeCoveragePhysicsDomainFilter:
    """--physics-domain filter propagates domain_clause to queries."""

    def test_domain_clause_injected(self):
        effects = _make_gc_side_effects()
        gc = _mock_gc(effects)

        with patch("imas_codex.standard_names.coverage.GraphClient", return_value=gc):
            report = compute_coverage(physics_domain="equilibrium")

        assert report.physics_domain_filter == "equilibrium"
        # The param $physics_domain must have been passed to at least one query
        all_kwargs = [c[1] for c in gc.query.call_args_list]
        assert any("physics_domain" in kw for kw in all_kwargs)

    def test_domain_filtered_totals(self):
        effects = _make_gc_side_effects(
            overrides={
                0: [{"total": 20}],
                10: [{"cnt": 15}],
            }
        )
        gc = _mock_gc(effects)

        with patch("imas_codex.standard_names.coverage.GraphClient", return_value=gc):
            report = compute_coverage(physics_domain="equilibrium")

        assert report.eligible_total == 20
        assert report.to_compose == 15


class TestComputeCoverageCostEstimate:
    """Cost estimate derived from SNRun telemetry."""

    def test_cost_per_name_computed(self):
        # Two runs: $10 for 100 names, $5 for 50 names → avg $0.10/name
        effects = _make_gc_side_effects(
            overrides={
                10: [{"cnt": 500}],
            }
        )
        # SNRun rows are the last query (index 12)
        effects[12] = [
            {"cost_spent": 10.0, "names_composed": 100},
            {"cost_spent": 5.0, "names_composed": 50},
        ]
        gc = _mock_gc(effects)

        with patch("imas_codex.standard_names.coverage.GraphClient", return_value=gc):
            report = compute_coverage()

        expected_cost_per_name = 15.0 / 150  # 0.1
        assert report.cost_per_name == pytest.approx(expected_cost_per_name)
        assert report.estimated_compose_cost == pytest.approx(
            expected_cost_per_name * 500
        )

    def test_no_runs_cost_is_none(self):
        effects = _make_gc_side_effects()
        gc = _mock_gc(effects)

        with patch("imas_codex.standard_names.coverage.GraphClient", return_value=gc):
            report = compute_coverage()

        assert report.cost_per_name is None
        assert report.estimated_compose_cost is None


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestSnCoverageCLI:
    """Test the `sn coverage` CLI command via Click's test runner."""

    def _invoke(self, args: list[str], side_effects: list | None = None):
        from imas_codex.cli.sn import sn

        runner = CliRunner()

        if side_effects is None:
            side_effects = _make_gc_side_effects()

        gc = _mock_gc(side_effects)
        with patch("imas_codex.standard_names.coverage.GraphClient", return_value=gc):
            result = runner.invoke(sn, ["coverage"] + args, catch_exceptions=False)
        return result

    def test_json_output_is_valid_json(self):
        result = self._invoke(["--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "eligible_total" in data
        assert "to_compose" in data
        assert "cost_per_name" in data

    def test_json_output_with_zeros(self):
        result = self._invoke(["--json"])
        data = json.loads(result.output)
        assert data["eligible_total"] == 0
        assert data["to_compose"] == 0
        assert data["cost_per_name"] is None

    def test_json_output_with_data(self):
        effects = _make_gc_side_effects(
            overrides={
                0: [{"total": 9213}],
                10: [{"cnt": 7000}],
            }
        )
        result = self._invoke(["--json"], side_effects=effects)
        data = json.loads(result.output)
        assert data["eligible_total"] == 9213
        assert data["to_compose"] == 7000

    def test_rich_output_contains_sections(self):
        result = self._invoke([])
        assert result.exit_code == 0
        assert "DD Extract Eligibility" in result.output
        assert "Already-Minted" in result.output
        assert "Work Remaining" in result.output

    def test_physics_domain_flag_accepted(self):
        result = self._invoke(["--physics-domain", "equilibrium"])
        assert result.exit_code == 0

    def test_json_with_physics_domain(self):
        effects = _make_gc_side_effects(overrides={0: [{"total": 42}]})
        result = self._invoke(
            ["--physics-domain", "equilibrium", "--json"], side_effects=effects
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["physics_domain_filter"] == "equilibrium"
        assert data["eligible_total"] == 42

    def test_rich_output_no_domain_table_when_filtered(self):
        """When --physics-domain is set, the domain breakdown table is omitted."""
        result = self._invoke(["--physics-domain", "equilibrium"])
        assert "By physics_domain" not in result.output

    def test_rich_output_has_domain_table_without_filter(self):
        effects = _make_gc_side_effects(
            overrides={
                2: [
                    {"bucket": "equilibrium", "cnt": 100},
                    {"bucket": "magnetics", "cnt": 50},
                ]
            }
        )
        result = self._invoke([], side_effects=effects)
        assert "By physics_domain" in result.output

    def test_cost_estimate_shown_when_telemetry_available(self):
        effects = _make_gc_side_effects(overrides={10: [{"cnt": 100}]})
        effects[12] = [{"cost_spent": 1.0, "names_composed": 10}]
        result = self._invoke([], side_effects=effects)
        assert "0.1" in result.output or "cost" in result.output.lower()

    def test_cost_unknown_when_no_telemetry(self):
        result = self._invoke([])
        assert "unknown" in result.output.lower() or "no prior" in result.output.lower()
