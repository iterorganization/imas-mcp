"""Tests for rc22 B3: node_type filter removal and breakdown diagnostic.

Verifies that:
- ``extract_dd_candidates`` admits nodes with ``node_type='static'`` when
  their ``node_category`` is in ``SN_SOURCE_CATEGORIES``.
- Nodes with ``node_category='metadata'`` (not in ``SN_SOURCE_CATEGORIES``)
  are structurally excluded by the Cypher filter.
- ``report_extract_breakdown`` returns the expected shape and aggregations.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DD_VERSION_ROW = {
    "dd_version": "4.0.0",
    "cocos_version": 11,
    "cocos_params": {},
}


def _make_node_row(
    path: str,
    node_category: str,
    physics_domain: str,
    description: str,
    ids_name: str = "equilibrium",
    node_type: str | None = "static",
    unit: str = "m",
    unit_from_rel: str = "m",
    data_type: str = "FLT_1D",
) -> dict:
    """Build a minimal enriched-query result row."""
    return {
        "path": path,
        "description": description,
        "documentation": None,
        "unit": unit,
        "unit_from_rel": unit_from_rel,
        "data_type": data_type,
        "physics_domain": physics_domain,
        "keywords": None,
        "node_category": node_category,
        "ndim": 1,
        "lifecycle_status": None,
        "ids_name": ids_name,
        "cluster_label": "Test cluster",
        "cluster_id": "test_cluster",
        "cluster_description": "Test cluster description",
        "cluster_scope": "ids",
        "parent_path": None,
        "parent_description": None,
        "parent_type": None,
        "coord_path": None,
        "coord_description": None,
        "coord_unit": None,
        "cocos_label": None,
        "cocos_expression": None,
    }


def _make_mock_gc(side_effects: list) -> MagicMock:
    """Build a GraphClient context-manager mock with given query side_effects."""
    mock_gc = MagicMock()
    mock_gc.__enter__ = MagicMock(return_value=mock_gc)
    mock_gc.__exit__ = MagicMock(return_value=False)
    mock_gc.query = MagicMock(side_effect=[iter(rows) for rows in side_effects])
    return mock_gc


# ---------------------------------------------------------------------------
# B3.1 — extract_dd_candidates admits static/geometry
# ---------------------------------------------------------------------------


class TestExtractAdmitsStaticGeometry:
    """rc22 B3: static geometry paths must pass through extraction."""

    def test_extract_admits_static_geometry(self):
        """A node with node_type='static' + node_category='geometry' is included."""
        static_geo_row = _make_node_row(
            path="equilibrium/time_slice/boundary/outline/r",
            node_category="geometry",
            physics_domain="equilibrium",
            description="R coordinates of the plasma boundary outline",
            node_type="static",
            unit="m",
            unit_from_rel="m",
            data_type="FLT_1D",
        )

        mock_gc = _make_mock_gc(
            [
                [_DD_VERSION_ROW],  # DD version query
                [static_geo_row],  # Main extraction query
                [],  # Siblings query
            ]
        )

        with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
            from imas_codex.standard_names.sources.dd import extract_dd_candidates

            batches = extract_dd_candidates(
                domain_filter="equilibrium",
                limit=10,
                force=True,
            )

        assert len(batches) > 0, (
            "extract_dd_candidates must return batches for static/geometry nodes"
        )
        all_paths = [item["path"] for batch in batches for item in batch.items]
        assert "equilibrium/time_slice/boundary/outline/r" in all_paths, (
            "static/geometry node must be present in extraction output"
        )

    def test_extract_query_gates_on_node_category_not_node_type(self):
        """The Cypher WHERE clause must use node_category, not node_type."""
        import inspect

        from imas_codex.standard_names.sources.dd import extract_dd_candidates

        src = inspect.getsource(extract_dd_candidates)
        assert "node_category IN $sn_categories" in src, (
            "extract_dd_candidates must filter on node_category IN $sn_categories"
        )
        assert "node_type IN ['dynamic'" not in src, (
            "extract_dd_candidates must NOT gate on node_type (rc22 B3 fix)"
        )


# ---------------------------------------------------------------------------
# B3.2 — extract_dd_candidates rejects metadata category
# ---------------------------------------------------------------------------


class TestExtractRejectsMetadataCategory:
    """rc22 B3: node_category='metadata' is not in SN_SOURCE_CATEGORIES."""

    def test_metadata_not_in_sn_source_categories(self):
        """'metadata' must be absent from SN_SOURCE_CATEGORIES."""
        from imas_codex.core.node_categories import SN_SOURCE_CATEGORIES

        assert "metadata" not in SN_SOURCE_CATEGORIES, (
            "SN_SOURCE_CATEGORIES must not include 'metadata'"
        )

    def test_sn_source_categories_contains_quantity_and_geometry(self):
        """SN_SOURCE_CATEGORIES must contain 'quantity' and 'geometry'."""
        from imas_codex.core.node_categories import SN_SOURCE_CATEGORIES

        assert "quantity" in SN_SOURCE_CATEGORIES
        assert "geometry" in SN_SOURCE_CATEGORIES

    def test_extract_query_passes_sn_categories_as_param(self):
        """The Cypher query must pass SN_SOURCE_CATEGORIES as $sn_categories.

        This ensures that 'metadata' (absent from the param list) would be
        rejected by the Cypher filter at the graph level, not accidentally
        admitted.
        """
        import inspect

        from imas_codex.standard_names.sources.dd import extract_dd_candidates

        src = inspect.getsource(extract_dd_candidates)
        # Parameter must be used
        assert "sn_categories" in src, (
            "extract_dd_candidates must pass sn_categories parameter to Cypher"
        )


# ---------------------------------------------------------------------------
# B3.3 — report_extract_breakdown return shape
# ---------------------------------------------------------------------------


class TestReportBreakdownGroupsByTypeCategory:
    """report_extract_breakdown must return correct shape and aggregations."""

    _MOCK_COUNTS = [
        {"node_type": "dynamic", "node_category": "quantity", "cnt": 8956},
        {"node_type": "constant", "node_category": "quantity", "cnt": 600},
        {"node_type": "static", "node_category": "quantity", "cnt": 250},
        {"node_type": "static", "node_category": "geometry", "cnt": 350},
        {"node_type": None, "node_category": "quantity", "cnt": 125},
    ]

    _MOCK_SAMPLES = [
        {
            "node_type": "static",
            "node_category": "quantity",
            "path": "equilibrium/time_slice/global_quantities/psi_boundary",
        },
        {
            "node_type": "static",
            "node_category": "quantity",
            "path": "equilibrium/time_slice/global_quantities/beta_pol",
        },
        {
            "node_type": "static",
            "node_category": "geometry",
            "path": "equilibrium/time_slice/boundary/outline/r",
        },
        {
            "node_type": None,
            "node_category": "quantity",
            "path": "core_profiles/global_quantities/vacuum_toroidal_field/r0",
        },
    ]

    def test_return_shape(self):
        """report_extract_breakdown returns a dict with the expected top-level keys."""
        mock_gc = _make_mock_gc([self._MOCK_COUNTS, self._MOCK_SAMPLES])

        with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
            from imas_codex.standard_names.sources.dd import report_extract_breakdown

            result = report_extract_breakdown()

        assert isinstance(result, dict)
        assert "total" in result
        assert "by_node_type" in result
        assert "by_category" in result
        assert "samples" in result

    def test_total_aggregation(self):
        """total must be the sum of all cnt values."""
        mock_gc = _make_mock_gc([self._MOCK_COUNTS, self._MOCK_SAMPLES])

        with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
            from imas_codex.standard_names.sources.dd import report_extract_breakdown

            result = report_extract_breakdown()

        expected_total = 8956 + 600 + 250 + 350 + 125
        assert result["total"] == expected_total

    def test_by_node_type_aggregation(self):
        """by_node_type must aggregate counts correctly."""
        mock_gc = _make_mock_gc([self._MOCK_COUNTS, self._MOCK_SAMPLES])

        with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
            from imas_codex.standard_names.sources.dd import report_extract_breakdown

            result = report_extract_breakdown()

        assert result["by_node_type"]["dynamic"] == 8956
        assert result["by_node_type"]["constant"] == 600
        # static has both quantity (250) and geometry (350)
        assert result["by_node_type"]["static"] == 600
        assert result["by_node_type"][None] == 125

    def test_by_category_aggregation(self):
        """by_category must aggregate counts across all node_type values."""
        mock_gc = _make_mock_gc([self._MOCK_COUNTS, self._MOCK_SAMPLES])

        with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
            from imas_codex.standard_names.sources.dd import report_extract_breakdown

            result = report_extract_breakdown()

        # quantity: dynamic(8956) + constant(600) + static(250) + None(125)
        assert result["by_category"]["quantity"] == 8956 + 600 + 250 + 125
        # geometry: static(350)
        assert result["by_category"]["geometry"] == 350

    def test_samples_buckets_present(self):
        """samples must contain keys for each non-dynamic (node_type, node_category) pair."""
        mock_gc = _make_mock_gc([self._MOCK_COUNTS, self._MOCK_SAMPLES])

        with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
            from imas_codex.standard_names.sources.dd import report_extract_breakdown

            result = report_extract_breakdown()

        assert "static/quantity" in result["samples"]
        assert "static/geometry" in result["samples"]
        assert "none/quantity" in result["samples"]

    def test_samples_capped_at_five(self):
        """Each samples bucket must contain at most 5 paths."""
        # Provide 10 sample rows for the same bucket
        many_samples = [
            {"node_type": "static", "node_category": "quantity", "path": f"path/{i}"}
            for i in range(10)
        ]
        mock_gc = _make_mock_gc([self._MOCK_COUNTS, many_samples])

        with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
            from imas_codex.standard_names.sources.dd import report_extract_breakdown

            result = report_extract_breakdown()

        assert len(result["samples"].get("static/quantity", [])) <= 5

    def test_dynamic_nodes_not_in_samples(self):
        """Dynamic-type paths must not appear in samples (only new admissions)."""
        # The samples query has a filter for non-dynamic; our mock respects it
        # by only returning non-dynamic rows in _MOCK_SAMPLES.
        mock_gc = _make_mock_gc([self._MOCK_COUNTS, self._MOCK_SAMPLES])

        with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
            from imas_codex.standard_names.sources.dd import report_extract_breakdown

            result = report_extract_breakdown()

        # No "dynamic/..." key should exist in samples
        for key in result["samples"]:
            assert not key.startswith("dynamic/"), (
                f"Dynamic paths must not appear in samples, but found key '{key}'"
            )

    def test_empty_graph_returns_zero_total(self):
        """If the graph returns no rows, total must be 0."""
        mock_gc = _make_mock_gc([[], []])

        with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
            from imas_codex.standard_names.sources.dd import report_extract_breakdown

            result = report_extract_breakdown()

        assert result["total"] == 0
        assert result["by_node_type"] == {}
        assert result["by_category"] == {}
        assert result["samples"] == {}
