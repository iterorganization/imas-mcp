"""Tests for B9 error-sibling minting.

Verifies that the compose pipeline deterministically mints
error-sibling StandardNames for DD paths with HAS_ERROR edges,
using ISN uncertainty modifier grammar.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# -----------------------------------------------------------------------
# Unit tests for mint_error_siblings
# -----------------------------------------------------------------------


class TestMintErrorSiblings:
    """Unit tests for the mint_error_siblings helper."""

    def test_error_siblings_minted_for_parent_with_errors(self):
        """A parent with 3 error node IDs produces 2 sibling candidates (W24 policy).

        W24 policy gate (Rule 6 in _parent_supports_uncertainty_index) blocks
        uncertainty_index_of_* for all parents.  Only upper and lower uncertainty
        siblings are produced.
        """
        from imas_codex.standard_names.error_siblings import mint_error_siblings

        siblings = mint_error_siblings(
            "plasma_current",
            error_node_ids=[
                "magnetics/ip/plasma_current_error_upper",
                "magnetics/ip/plasma_current_error_lower",
                "magnetics/ip/plasma_current_error_index",
            ],
            unit="A",
            physics_domain="magnetics",
            cocos_type="ip_like",
            cocos_version=11,
            dd_version="4.0.0",
        )

        assert len(siblings) == 2

        ids = {s["id"] for s in siblings}
        assert "upper_uncertainty_of_plasma_current" in ids
        assert "lower_uncertainty_of_plasma_current" in ids
        assert "uncertainty_index_of_plasma_current" not in ids

    def test_error_sibling_shares_parent_unit(self):
        """Each sibling inherits the parent's unit."""
        from imas_codex.standard_names.error_siblings import mint_error_siblings

        siblings = mint_error_siblings(
            "pressure",
            error_node_ids=[
                "core_profiles/profiles_1d/pressure_error_upper",
            ],
            unit="Pa",
            physics_domain="transport",
            cocos_type=None,
            cocos_version=None,
            dd_version="4.0.0",
        )

        assert len(siblings) == 1
        assert siblings[0]["unit"] == "Pa"

    def test_error_sibling_has_from_dd_path_to_error_node(self):
        """Each sibling's source_id points to the error IMASNode, not the parent."""
        from imas_codex.standard_names.error_siblings import mint_error_siblings

        error_id = "core_profiles/profiles_1d/electrons/temperature_error_upper"
        siblings = mint_error_siblings(
            "temperature",
            error_node_ids=[error_id],
            unit="eV",
            physics_domain="transport",
            cocos_type=None,
            cocos_version=None,
            dd_version="4.0.0",
        )

        assert len(siblings) == 1
        assert siblings[0]["source_id"] == error_id
        assert siblings[0]["source_types"] == ["dd"]

    def test_parent_compose_fails_skips_siblings_cleanly(self):
        """If parent name is empty or None, no siblings are minted (silent skip)."""
        from imas_codex.standard_names.error_siblings import mint_error_siblings

        # Parent name that will fail ISN validation (gibberish)
        siblings = mint_error_siblings(
            "zzz_not_a_valid_base_xyz",
            error_node_ids=[
                "some/path_error_upper",
            ],
            unit="eV",
            physics_domain=None,
            cocos_type=None,
            cocos_version=None,
            dd_version="4.0.0",
        )

        # Should fail validation and return empty (no exception)
        assert len(siblings) == 0

    def test_error_sibling_naming(self):
        """Verify exact naming pattern for upper and lower suffixes (W24 policy).

        W24 policy gate blocks uncertainty_index_of_* for all parents.
        Only upper and lower uncertainty siblings are produced.
        """
        from imas_codex.standard_names.error_siblings import mint_error_siblings

        siblings = mint_error_siblings(
            "plasma_current",
            error_node_ids=[
                "magnetics/ip_error_upper",
                "magnetics/ip_error_lower",
                "magnetics/ip_error_index",
            ],
            unit="A",
            physics_domain="magnetics",
            cocos_type=None,
            cocos_version=None,
            dd_version="4.0.0",
        )

        # Build a map for easy assertion
        name_map = {s["id"]: s for s in siblings}

        assert "upper_uncertainty_of_plasma_current" in name_map
        assert "lower_uncertainty_of_plasma_current" in name_map
        assert "uncertainty_index_of_plasma_current" not in name_map

    def test_error_sibling_provenance_fields(self):
        """Error siblings have deterministic provenance markers."""
        from imas_codex.standard_names.error_siblings import mint_error_siblings

        siblings = mint_error_siblings(
            "pressure",
            error_node_ids=["x/pressure_error_upper"],
            unit="Pa",
            physics_domain=None,
            cocos_type=None,
            cocos_version=None,
            dd_version="4.0.0",
        )

        assert len(siblings) == 1
        s = siblings[0]
        assert s["model"] == "deterministic:dd_error_modifier"
        assert s["pipeline_status"] == "named"
        assert s["validation_status"] == "valid"
        assert s["confidence"] == 1.0
        assert s["reviewer_score_name"] == 1.0
        assert s["reviewed_name_at"] is not None
        assert s["reviewer_score_docs"] == 1.0
        assert s["reviewed_docs_at"] is not None

    def test_error_sibling_inherits_physics_domain(self):
        """Error siblings inherit the parent's physics_domain."""
        from imas_codex.standard_names.error_siblings import mint_error_siblings

        siblings = mint_error_siblings(
            "pressure",
            error_node_ids=["x/pressure_error_lower"],
            unit="Pa",
            physics_domain="equilibrium",
            cocos_type=None,
            cocos_version=None,
            dd_version="4.0.0",
        )

        assert siblings[0]["physics_domain"] == "equilibrium"

    def test_error_sibling_inherits_cocos(self):
        """Error siblings inherit the parent's COCOS metadata."""
        from imas_codex.standard_names.error_siblings import mint_error_siblings

        siblings = mint_error_siblings(
            "plasma_current",
            error_node_ids=["x/ip_error_upper"],
            unit="A",
            physics_domain="magnetics",
            cocos_type="ip_like",
            cocos_version=11,
            dd_version="4.0.0",
        )

        assert siblings[0]["cocos_transformation_type"] == "ip_like"
        assert siblings[0]["cocos"] == 11

    def test_unknown_suffix_is_skipped(self):
        """An error_node_id with an unexpected suffix is silently skipped."""
        from imas_codex.standard_names.error_siblings import mint_error_siblings

        siblings = mint_error_siblings(
            "pressure",
            error_node_ids=["x/pressure_error_unknown"],
            unit="Pa",
            physics_domain=None,
            cocos_type=None,
            cocos_version=None,
            dd_version="4.0.0",
        )

        assert len(siblings) == 0

    def test_uncertainty_index_kind_is_scalar(self):
        """W24 policy: uncertainty_index siblings are never produced (blocked by Rule 6).

        Prior to W24, _error_index siblings had kind='scalar'.  Rule 6 now
        blocks all uncertainty_index_of_* creation, so passing only
        _error_index returns an empty list.
        """
        from imas_codex.standard_names.error_siblings import mint_error_siblings

        siblings = mint_error_siblings(
            "pressure",
            error_node_ids=["x/pressure_error_index"],
            unit="Pa",
            physics_domain=None,
            cocos_type=None,
            cocos_version=None,
            dd_version="4.0.0",
        )

        assert len(siblings) == 0, (
            "W24 policy: uncertainty_index_of_* must not be generated; "
            f"got {[s['id'] for s in siblings]}"
        )


# -----------------------------------------------------------------------
# Tests for _detect_error_suffix
# -----------------------------------------------------------------------


class TestDetectErrorSuffix:
    """Tests for the suffix detection helper."""

    def test_error_upper(self):
        from imas_codex.standard_names.error_siblings import _detect_error_suffix

        assert _detect_error_suffix("foo/bar_error_upper") == "_error_upper"

    def test_error_lower(self):
        from imas_codex.standard_names.error_siblings import _detect_error_suffix

        assert _detect_error_suffix("foo/bar_error_lower") == "_error_lower"

    def test_error_index(self):
        from imas_codex.standard_names.error_siblings import _detect_error_suffix

        assert _detect_error_suffix("foo/bar_error_index") == "_error_index"

    def test_no_error_suffix(self):
        from imas_codex.standard_names.error_siblings import _detect_error_suffix

        assert _detect_error_suffix("foo/bar") is None


# -----------------------------------------------------------------------
# Reconcile orphan error siblings
# -----------------------------------------------------------------------


class TestReconcileErrorSiblings:
    """Tests for reconcile_error_siblings orphan detection."""

    def test_reconcile_orphans_error_siblings(self):
        """When the parent StandardName is deleted, error siblings are marked skipped."""
        from imas_codex.standard_names.graph_ops import reconcile_error_siblings

        mock_gc = MagicMock()

        # Simulate: one error sibling exists, parent does NOT
        mock_gc.query = MagicMock(
            side_effect=[
                # First call: find all error siblings
                [{"id": "upper_uncertainty_of_plasma_current"}],
                # Second call: check if parent "plasma_current" exists → empty
                [],
                # Third call: SET pipeline_status = 'skipped'
                None,
            ]
        )

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            result = reconcile_error_siblings()

        assert result["stale_marked"] == 1

        # Verify the SET query was called with the orphan ID
        set_call = mock_gc.query.call_args_list[2]
        assert "SET sn.pipeline_status = 'skipped'" in set_call[0][0]
        assert set_call[1]["ids"] == ["upper_uncertainty_of_plasma_current"]

    def test_reconcile_no_orphans(self):
        """When all parents exist, no siblings are marked stale."""
        from imas_codex.standard_names.graph_ops import reconcile_error_siblings

        mock_gc = MagicMock()

        mock_gc.query = MagicMock(
            side_effect=[
                # Find error siblings
                [{"id": "upper_uncertainty_of_plasma_current"}],
                # Parent exists
                [{"p.id": "plasma_current"}],
            ]
        )

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            result = reconcile_error_siblings()

        assert result["stale_marked"] == 0

    def test_reconcile_empty_graph(self):
        """When no error siblings exist, reconcile is a no-op."""
        from imas_codex.standard_names.graph_ops import reconcile_error_siblings

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            result = reconcile_error_siblings()

        assert result["stale_marked"] == 0
