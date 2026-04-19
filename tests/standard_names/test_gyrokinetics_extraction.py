"""Regression tests: gyrokinetics (constant node_type) extraction.

The gyrokinetics IDS has ONLY ``node_type='constant'`` quantity paths.
Before the fix, the extraction query filtered on ``node_type = 'dynamic'``,
silently dropping all gyrokinetics paths.

See: plans/research/standard-names/gyrokinetics-extraction-debug.md
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestConstantNodeTypeExtraction:
    """Verify that constant-type DD nodes pass extraction filters."""

    def test_extraction_query_includes_constant(self):
        """The enriched extraction query must accept node_type='constant'."""
        # The query template has a {where_clause} placeholder, but
        # extract_dd_candidates builds the WHERE clause with node_type filter.
        # Verify the function constructs the correct filter.
        import inspect

        from imas_codex.standard_names.sources.dd import (
            _ENRICHED_QUERY,
            extract_dd_candidates,
        )

        src = inspect.getsource(extract_dd_candidates)
        assert "constant" in src, (
            "extract_dd_candidates must include 'constant' in node_type filter"
        )
        assert "'dynamic'" in src, (
            "extract_dd_candidates must still include 'dynamic' in node_type filter"
        )

    def test_graph_ops_query_includes_constant(self):
        """Legacy graph_ops extraction also accepts constant node_type."""
        import inspect

        from imas_codex.standard_names.graph_ops import get_extraction_candidates_dd

        src = inspect.getsource(get_extraction_candidates_dd)
        assert "constant" in src, (
            "get_extraction_candidates_dd must include 'constant' in node_type filter"
        )

    def test_gyrokinetics_constant_paths_classified_as_quantity(self):
        """Gyrokinetics paths (constant node_type) pass the classifier."""
        from imas_codex.standard_names.classifier import classify_path

        gyro_paths = [
            {
                "path": "gyrokinetics/model/time_interval_norm",
                "data_type": "FLT_0D",
            },
            {
                "path": "gyrokinetics/species_all/beta_reference",
                "data_type": "FLT_0D",
            },
            {
                "path": "gyrokinetics/wavevector/eigenmode/frequency_norm",
                "data_type": "FLT_0D",
            },
        ]
        for node in gyro_paths:
            assert classify_path(node) == "quantity", (
                f"Gyrokinetics path {node['path']} should classify as 'quantity'"
            )

    def test_mock_extraction_returns_gyrokinetics(self):
        """Full mock: extract_dd_candidates returns batches for gyrokinetics data."""
        # Simulate graph returning gyrokinetics rows (constant node_type)
        mock_dd_version_row = {
            "dd_version": "4.0.0",
            "cocos_version": 11,
            "cocos_params": {},
        }
        mock_gyro_rows = [
            {
                "path": "gyrokinetics/model/time_interval_norm",
                "description": "Normalised time interval for averaging fluxes",
                "documentation": None,
                "unit": "-",
                "unit_from_rel": "-",
                "data_type": "FLT_0D",
                "physics_domain": "gyrokinetics",
                "keywords": None,
                "node_category": "quantity",
                "ndim": 0,
                "lifecycle_status": None,
                "ids_name": "gyrokinetics",
                "cluster_label": "Gyrokinetic simulation parameters",
                "cluster_id": "gk_sim_params",
                "cluster_description": "Parameters for gyrokinetic simulations",
                "cluster_scope": "ids",
                "parent_path": "gyrokinetics/model",
                "parent_description": "Model parameters",
                "parent_type": None,
                "coord_path": None,
                "coord_description": None,
                "coord_unit": None,
                "cocos_label": None,
                "cocos_expression": None,
            },
            {
                "path": "gyrokinetics/species_all/beta_reference",
                "description": "Reference plasma beta for gyrokinetic simulations",
                "documentation": None,
                "unit": "-",
                "unit_from_rel": "-",
                "data_type": "FLT_0D",
                "physics_domain": "gyrokinetics",
                "keywords": None,
                "node_category": "quantity",
                "ndim": 0,
                "lifecycle_status": None,
                "ids_name": "gyrokinetics",
                "cluster_label": "Gyrokinetic simulation parameters",
                "cluster_id": "gk_sim_params",
                "cluster_description": "Parameters for gyrokinetic simulations",
                "cluster_scope": "ids",
                "parent_path": "gyrokinetics/species_all",
                "parent_description": "Species parameters",
                "parent_type": None,
                "coord_path": None,
                "coord_description": None,
                "coord_unit": None,
                "cocos_label": None,
                "cocos_expression": None,
            },
        ]

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)
        mock_gc.query = MagicMock(
            side_effect=[
                iter([mock_dd_version_row]),  # DD version query
                iter(mock_gyro_rows),  # Main extraction query
                iter([]),  # Siblings query
            ]
        )

        with patch(
            "imas_codex.graph.client.GraphClient",
            return_value=mock_gc,
        ):
            from imas_codex.standard_names.sources.dd import extract_dd_candidates

            batches = extract_dd_candidates(
                domain_filter="gyrokinetics",
                limit=10,
                force=True,
            )

        assert len(batches) > 0, (
            "extract_dd_candidates must return >0 batches for gyrokinetics"
        )
        # Verify all items are from gyrokinetics
        all_paths = [item["path"] for batch in batches for item in batch.items]
        assert all(p.startswith("gyrokinetics/") for p in all_paths)
