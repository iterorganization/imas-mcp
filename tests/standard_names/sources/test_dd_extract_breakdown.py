"""Tests for rc22 B3': leaf-invariant, coordinate category, node_type context,
and error-field surface.

Verifies that:
- ``extract_dd_candidates`` admits nodes with ``node_type='static'`` when
  their ``node_category`` is in ``SN_SOURCE_CATEGORIES``.
- Nodes with ``data_type='STRUCTURE'`` are excluded by the leaf invariant
  even when ``node_category='quantity'``.
- Nodes with ``node_category='coordinate'`` are now admitted (B3' addition).
- Nodes with ``node_category='metadata'`` (not in ``SN_SOURCE_CATEGORIES``)
  are excluded.
- ``report_extract_breakdown`` returns the expected shape and aggregations,
  including ``by_data_type`` and ``has_errors_count``.
- Candidates expose ``node_type`` (LLM context) and ``has_errors`` /
  ``error_node_ids`` (B9 prep).
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
    error_node_ids: list[str] | None = None,
) -> dict:
    """Build a minimal enriched-query result row."""
    row = {
        "path": path,
        "description": description,
        "documentation": None,
        "unit": unit,
        "unit_from_rel": unit_from_rel,
        "data_type": data_type,
        "node_type": node_type,
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
        "error_node_ids": error_node_ids if error_node_ids is not None else [],
    }
    return row


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
# B3' Part 1 — leaf invariant: STRUCTURE/STRUCT_ARRAY excluded
# ---------------------------------------------------------------------------


class TestExtractLeafInvariant:
    """Leaf invariant: STRUCTURE/STRUCT_ARRAY containers must not be admitted."""

    def test_extract_excludes_STRUCTURE_containers(self):
        """A node with node_category='quantity' but data_type='STRUCTURE' is NOT admitted.

        Pre-fix, the extraction admitted ~1 599 such containers (e.g.
        summary/global_quantities/q_95 which wraps {value, source}).
        The leaf invariant (data_type NOT IN STRUCTURE/STRUCT_ARRAY) blocks them.
        """
        # The leaf invariant is applied at the Cypher WHERE level.  Verify
        # the source string contains the guard clause.
        import inspect

        from imas_codex.standard_names.sources.dd import _ENRICHED_QUERY

        assert "STRUCTURE" in _ENRICHED_QUERY or "STRUCTURE" in inspect.getsource(
            __import__(
                "imas_codex.standard_names.sources.dd",
                fromlist=["extract_dd_candidates"],
            ).extract_dd_candidates
        ), (
            "leaf invariant (data_type NOT IN STRUCTURE/STRUCT_ARRAY) must be in "
            "the extraction query or where_parts"
        )

    def test_leaf_invariant_in_where_parts(self):
        """The where_parts list must include the STRUCTURE/STRUCT_ARRAY guard."""
        import inspect

        from imas_codex.standard_names.sources.dd import extract_dd_candidates

        src = inspect.getsource(extract_dd_candidates)
        assert "STRUCTURE" in src, (
            "extract_dd_candidates must contain leaf invariant for STRUCTURE types"
        )
        assert "STRUCT_ARRAY" in src, (
            "extract_dd_candidates must contain leaf invariant for STRUCT_ARRAY types"
        )

    def test_breakdown_queries_exclude_structure(self):
        """Breakdown queries must also enforce the leaf invariant."""
        from imas_codex.standard_names.sources.dd import (
            _BREAKDOWN_QUERY,
            _BREAKDOWN_SAMPLES_QUERY,
        )

        for qname, q in [
            ("_BREAKDOWN_QUERY", _BREAKDOWN_QUERY),
            ("_BREAKDOWN_SAMPLES_QUERY", _BREAKDOWN_SAMPLES_QUERY),
        ]:
            assert "STRUCTURE" in q, f"{qname} must exclude STRUCTURE data_type"
            assert "STRUCT_ARRAY" in q, f"{qname} must exclude STRUCT_ARRAY data_type"


# ---------------------------------------------------------------------------
# B3' Part 2 — coordinate category admitted
# ---------------------------------------------------------------------------


class TestCoordinateCategoryAdmitted:
    """'coordinate' is now in SN_SOURCE_CATEGORIES."""

    def test_coordinate_in_sn_source_categories(self):
        """SN_SOURCE_CATEGORIES must include 'coordinate'."""
        from imas_codex.core.node_categories import SN_SOURCE_CATEGORIES

        assert "coordinate" in SN_SOURCE_CATEGORIES, (
            "SN_SOURCE_CATEGORIES must include 'coordinate' (rc22 B3')"
        )

    def test_extract_admits_coordinate_leaves(self):
        """A node with node_category='coordinate' and a leaf data_type is admitted."""
        coord_row = _make_node_row(
            path="core_profiles/profiles_1d/grid/rho_tor_norm",
            node_category="coordinate",
            physics_domain="transport",
            description="Normalised toroidal flux coordinate",
            node_type="dynamic",
            unit="-",
            unit_from_rel="-",
            data_type="FLT_1D",
        )

        mock_gc = _make_mock_gc(
            [
                [_DD_VERSION_ROW],
                [coord_row],
                [],  # siblings
            ]
        )

        with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
            from imas_codex.standard_names.sources.dd import extract_dd_candidates

            batches = extract_dd_candidates(limit=10, force=True)

        all_paths = [item["path"] for batch in batches for item in batch.items]
        assert "core_profiles/profiles_1d/grid/rho_tor_norm" in all_paths, (
            "coordinate-category leaf must be admitted"
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
# B3' Part 3 — node_type in candidate payload
# ---------------------------------------------------------------------------


class TestCandidateSurfacesNodeType:
    """node_type must be returned by the query and preserved in the item dict."""

    def test_candidate_surfaces_node_type(self):
        """An admitted candidate exposes 'node_type' with the original graph value."""
        for nt in ("dynamic", "static", "constant"):
            row = _make_node_row(
                path="core_profiles/profiles_1d/electrons/temperature",
                node_category="quantity",
                physics_domain="transport",
                description="Electron temperature",
                node_type=nt,
                data_type="FLT_1D",
            )

            mock_gc = _make_mock_gc(
                [
                    [_DD_VERSION_ROW],
                    [row],
                    [],
                ]
            )

            with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
                from imas_codex.standard_names.sources.dd import extract_dd_candidates

                batches = extract_dd_candidates(limit=10, force=True)

            assert len(batches) > 0, f"node_type={nt!r} must produce batches"
            items = [i for b in batches for i in b.items]
            assert len(items) > 0
            assert items[0]["node_type"] == nt, f"node_type must be preserved as {nt!r}"

    def test_node_type_returned_in_enriched_query(self):
        """_ENRICHED_QUERY must include node_type in its RETURN clause."""
        from imas_codex.standard_names.sources.dd import _ENRICHED_QUERY

        assert "n.node_type AS node_type" in _ENRICHED_QUERY, (
            "_ENRICHED_QUERY must return n.node_type AS node_type"
        )


# ---------------------------------------------------------------------------
# B3' Part 4 — error-field surface (B9 prep)
# ---------------------------------------------------------------------------


class TestCandidateSurfacesErrorLinks:
    """Candidates must surface has_errors / error_node_ids for B9 prep."""

    def test_candidate_surfaces_error_links(self):
        """A candidate with HAS_ERROR siblings exposes has_errors=True and error_node_ids."""
        row = _make_node_row(
            path="equilibrium/time_slice/profiles_1d/psi",
            node_category="quantity",
            physics_domain="equilibrium",
            description="Poloidal magnetic flux",
            node_type="dynamic",
            data_type="FLT_1D",
            error_node_ids=[
                "equilibrium/time_slice/profiles_1d/psi_error_upper",
                "equilibrium/time_slice/profiles_1d/psi_error_lower",
            ],
        )

        mock_gc = _make_mock_gc(
            [
                [_DD_VERSION_ROW],
                [row],
                [],
            ]
        )

        with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
            from imas_codex.standard_names.sources.dd import extract_dd_candidates

            batches = extract_dd_candidates(limit=10, force=True)

        items = [i for b in batches for i in b.items]
        assert len(items) > 0
        item = items[0]
        assert item["has_errors"] is True, (
            "has_errors must be True when error_node_ids non-empty"
        )
        assert len(item["error_node_ids"]) == 2, (
            "error_node_ids must contain both error companion IDs"
        )

    def test_candidate_no_errors_defaults_false(self):
        """A candidate without HAS_ERROR siblings gets has_errors=False."""
        row = _make_node_row(
            path="equilibrium/time_slice/global_quantities/beta_pol",
            node_category="quantity",
            physics_domain="equilibrium",
            description="Poloidal beta",
            node_type="dynamic",
            data_type="FLT_0D",
            error_node_ids=[],
        )

        mock_gc = _make_mock_gc(
            [
                [_DD_VERSION_ROW],
                [row],
                [],
            ]
        )

        with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
            from imas_codex.standard_names.sources.dd import extract_dd_candidates

            batches = extract_dd_candidates(limit=10, force=True)

        items = [i for b in batches for i in b.items]
        assert len(items) > 0
        item = items[0]
        assert item["has_errors"] is False
        assert item["error_node_ids"] == []

    def test_enriched_query_returns_error_node_ids(self):
        """_ENRICHED_QUERY must include error_node_ids in its RETURN."""
        from imas_codex.standard_names.sources.dd import _ENRICHED_QUERY

        assert "error_node_ids" in _ENRICHED_QUERY, (
            "_ENRICHED_QUERY must surface error_node_ids via COLLECT or pattern comprehension"
        )
        assert "HAS_ERROR" in _ENRICHED_QUERY, (
            "_ENRICHED_QUERY must OPTIONAL MATCH HAS_ERROR edges"
        )


# ---------------------------------------------------------------------------
# B3.3 — report_extract_breakdown return shape
# ---------------------------------------------------------------------------


class TestReportBreakdownGroupsByTypeCategory:
    """report_extract_breakdown must return correct shape and aggregations."""

    _MOCK_COUNTS = [
        {
            "node_type": "dynamic",
            "node_category": "quantity",
            "data_type": "FLT_1D",
            "cnt": 8956,
        },
        {
            "node_type": "constant",
            "node_category": "quantity",
            "data_type": "FLT_0D",
            "cnt": 600,
        },
        {
            "node_type": "static",
            "node_category": "quantity",
            "data_type": "FLT_1D",
            "cnt": 250,
        },
        {
            "node_type": "static",
            "node_category": "geometry",
            "data_type": "FLT_1D",
            "cnt": 350,
        },
        {
            "node_type": None,
            "node_category": "quantity",
            "data_type": "INT_1D",
            "cnt": 125,
        },
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

    _MOCK_ERRORS = [{"has_errors_count": 8203}]

    def test_return_shape(self):
        """report_extract_breakdown returns a dict with the expected top-level keys."""
        mock_gc = _make_mock_gc(
            [self._MOCK_COUNTS, self._MOCK_SAMPLES, self._MOCK_ERRORS]
        )

        with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
            from imas_codex.standard_names.sources.dd import report_extract_breakdown

            result = report_extract_breakdown()

        assert isinstance(result, dict)
        assert "total" in result
        assert "by_node_type" in result
        assert "by_category" in result
        assert "by_data_type" in result
        assert "has_errors_count" in result
        assert "samples" in result

    def test_total_aggregation(self):
        """total must be the sum of all cnt values."""
        mock_gc = _make_mock_gc(
            [self._MOCK_COUNTS, self._MOCK_SAMPLES, self._MOCK_ERRORS]
        )

        with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
            from imas_codex.standard_names.sources.dd import report_extract_breakdown

            result = report_extract_breakdown()

        expected_total = 8956 + 600 + 250 + 350 + 125
        assert result["total"] == expected_total

    def test_by_node_type_aggregation(self):
        """by_node_type must aggregate counts correctly."""
        mock_gc = _make_mock_gc(
            [self._MOCK_COUNTS, self._MOCK_SAMPLES, self._MOCK_ERRORS]
        )

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
        mock_gc = _make_mock_gc(
            [self._MOCK_COUNTS, self._MOCK_SAMPLES, self._MOCK_ERRORS]
        )

        with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
            from imas_codex.standard_names.sources.dd import report_extract_breakdown

            result = report_extract_breakdown()

        # quantity: dynamic(8956) + constant(600) + static(250) + None(125)
        assert result["by_category"]["quantity"] == 8956 + 600 + 250 + 125
        # geometry: static(350)
        assert result["by_category"]["geometry"] == 350

    def test_by_data_type_aggregation(self):
        """by_data_type must aggregate counts and must not include STRUCTURE."""
        mock_gc = _make_mock_gc(
            [self._MOCK_COUNTS, self._MOCK_SAMPLES, self._MOCK_ERRORS]
        )

        with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
            from imas_codex.standard_names.sources.dd import report_extract_breakdown

            result = report_extract_breakdown()

        by_dt = result["by_data_type"]
        # FLT_1D appears in dynamic/quantity (8956) + static/quantity (250) + static/geometry (350)
        assert by_dt.get("FLT_1D") == 8956 + 250 + 350
        # No STRUCTURE or STRUCT_ARRAY admitted
        assert "STRUCTURE" not in by_dt, "STRUCTURE must not appear in by_data_type"
        assert "STRUCT_ARRAY" not in by_dt, (
            "STRUCT_ARRAY must not appear in by_data_type"
        )

    def test_has_errors_count(self):
        """has_errors_count must reflect the errors-query result."""
        mock_gc = _make_mock_gc(
            [self._MOCK_COUNTS, self._MOCK_SAMPLES, self._MOCK_ERRORS]
        )

        with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
            from imas_codex.standard_names.sources.dd import report_extract_breakdown

            result = report_extract_breakdown()

        assert result["has_errors_count"] == 8203

    def test_samples_buckets_present(self):
        """samples must contain keys for each non-dynamic (node_type, node_category) pair."""
        mock_gc = _make_mock_gc(
            [self._MOCK_COUNTS, self._MOCK_SAMPLES, self._MOCK_ERRORS]
        )

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
        mock_gc = _make_mock_gc([self._MOCK_COUNTS, many_samples, self._MOCK_ERRORS])

        with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
            from imas_codex.standard_names.sources.dd import report_extract_breakdown

            result = report_extract_breakdown()

        assert len(result["samples"].get("static/quantity", [])) <= 5

    def test_dynamic_nodes_not_in_samples(self):
        """Dynamic-type paths must not appear in samples (only new admissions)."""
        mock_gc = _make_mock_gc(
            [self._MOCK_COUNTS, self._MOCK_SAMPLES, self._MOCK_ERRORS]
        )

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
        mock_gc = _make_mock_gc([[], [], []])

        with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
            from imas_codex.standard_names.sources.dd import report_extract_breakdown

            result = report_extract_breakdown()

        assert result["total"] == 0
        assert result["by_node_type"] == {}
        assert result["by_category"] == {}
        assert result["by_data_type"] == {}
        assert result["has_errors_count"] == 0
        assert result["samples"] == {}
