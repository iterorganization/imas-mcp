"""Tests for DD migration guide generator."""

from unittest.mock import MagicMock

import pytest


class TestMigrationGuideRendering:
    """Unit tests for migration guide rendering (no graph needed)."""

    def test_render_empty_guide(self):
        from imas_codex.tools.migration_guide import _render_guide

        result = _render_guide(
            from_ver="3.39.0",
            to_ver="4.0.0",
            version_range=["3.40.0", "4.0.0"],
            summary=[],
            cocos_table=[],
            renames=[],
            unit_changes=[],
            type_changes=[],
            removals=[],
            additions=[],
            from_cocos=11,
            to_cocos=17,
            ids_filter=None,
            include_recipes=True,
        )
        assert "3.39.0" in result and "4.0.0" in result
        assert "COCOS convention change:" in result
        assert "11" in result and "17" in result

    def test_render_with_renames(self):
        from imas_codex.tools.migration_guide import _render_guide

        renames = [
            {"ids": "equilibrium", "old_path": "eq/label", "new_path": "eq/name"}
        ]
        result = _render_guide(
            from_ver="3.39.0",
            to_ver="4.0.0",
            version_range=["4.0.0"],
            summary=[],
            cocos_table=[],
            renames=renames,
            unit_changes=[],
            type_changes=[],
            removals=[],
            additions=[],
            from_cocos=11,
            to_cocos=17,
            ids_filter=None,
            include_recipes=True,
        )
        assert "Path Renames (1)" in result
        assert "eq/label" in result
        assert "eq/name" in result

    def test_render_with_unit_changes(self):
        from imas_codex.tools.migration_guide import _render_guide

        units = [
            {
                "ids": "core_profiles",
                "path": "cp/pressure",
                "old_unit": "J.m^-3",
                "new_unit": "Pa",
                "subtype": "dim_equivalent",
                "level": "advisory",
            }
        ]
        result = _render_guide(
            from_ver="3.39.0",
            to_ver="4.0.0",
            version_range=["4.0.0"],
            summary=[],
            cocos_table=[],
            renames=[],
            unit_changes=units,
            type_changes=[],
            removals=[],
            additions=[],
            from_cocos=None,
            to_cocos=None,
            ids_filter=None,
            include_recipes=True,
        )
        assert "Unit Changes" in result
        assert "J.m^-3" in result

    def test_render_with_type_changes(self):
        from imas_codex.tools.migration_guide import _render_guide

        type_changes = [
            {
                "ids": "equilibrium",
                "path": "eq/flag",
                "old_type": "INT_0D",
                "new_type": "STR_0D",
            }
        ]
        result = _render_guide(
            from_ver="3.39.0",
            to_ver="4.0.0",
            version_range=["4.0.0"],
            summary=[],
            cocos_table=[],
            renames=[],
            unit_changes=[],
            type_changes=type_changes,
            removals=[],
            additions=[],
            from_cocos=None,
            to_cocos=None,
            ids_filter=None,
            include_recipes=True,
        )
        assert "Type Changes (1)" in result
        assert "INT_0D" in result
        assert "STR_0D" in result

    def test_render_with_removals(self):
        from imas_codex.tools.migration_guide import _render_guide

        removals = [
            {
                "ids": "equilibrium",
                "path": "eq/old_field",
                "replacement": "eq/new_field",
            },
            {"ids": "equilibrium", "path": "eq/dead_field", "replacement": None},
        ]
        result = _render_guide(
            from_ver="3.39.0",
            to_ver="4.0.0",
            version_range=["4.0.0"],
            summary=[],
            cocos_table=[],
            renames=[],
            unit_changes=[],
            type_changes=[],
            removals=removals,
            additions=[],
            from_cocos=None,
            to_cocos=None,
            ids_filter=None,
            include_recipes=True,
        )
        assert "Removed Paths (2)" in result
        assert "eq/new_field" in result
        assert "—" in result

    def test_render_summary_counts(self):
        from imas_codex.tools.migration_guide import _render_guide

        summary = [
            {"type": "path_renamed", "level": "advisory", "cnt": 5},
            {"type": "units", "level": "advisory", "cnt": 3},
            {"type": "path_added", "level": "informational", "cnt": 10},
        ]
        result = _render_guide(
            from_ver="3.39.0",
            to_ver="4.0.0",
            version_range=["3.40.0", "4.0.0"],
            summary=summary,
            cocos_table=[],
            renames=[],
            unit_changes=[],
            type_changes=[],
            removals=[],
            additions=[],
            from_cocos=None,
            to_cocos=None,
            ids_filter=None,
            include_recipes=True,
        )
        assert "Total changes:** 18" in result
        assert "Breaking changes:** 0" in result
        assert "Advisory changes:** 8" in result
        assert "Informational:** 10" in result
        assert "Versions spanned:** 2" in result

    def test_ids_filter(self):
        from imas_codex.tools.migration_guide import _render_guide

        renames = [
            {"ids": "equilibrium", "old_path": "eq/a", "new_path": "eq/b"},
            {"ids": "core_profiles", "old_path": "cp/a", "new_path": "cp/b"},
        ]
        result = _render_guide(
            from_ver="3.39.0",
            to_ver="4.0.0",
            version_range=["4.0.0"],
            summary=[],
            cocos_table=[],
            renames=renames,
            unit_changes=[],
            type_changes=[],
            removals=[],
            additions=[],
            from_cocos=None,
            to_cocos=None,
            ids_filter="equilibrium",
            include_recipes=True,
        )
        assert "eq/a" in result
        assert "cp/a" not in result

    def test_cocos_sign_flip_table(self):
        from imas_codex.tools.migration_guide import _render_guide

        cocos_table = [
            {
                "ids": "equilibrium",
                "path": "equilibrium/time_slice/profiles_1d/psi",
                "label": "psi_like",
                "source": "xml",
                "factor": -1,
                "action": "Multiply by -1",
            }
        ]
        result = _render_guide(
            from_ver="3.39.0",
            to_ver="4.0.0",
            version_range=["4.0.0"],
            summary=[],
            cocos_table=cocos_table,
            renames=[],
            unit_changes=[],
            type_changes=[],
            removals=[],
            additions=[],
            from_cocos=11,
            to_cocos=17,
            ids_filter=None,
            include_recipes=True,
        )
        assert "COCOS Sign-Flip Table" in result
        assert "psi_like" in result
        assert "Multiply by -1" in result

    def test_code_recipes_with_renames(self):
        from imas_codex.tools.migration_guide import _render_guide

        renames = [{"ids": "eq", "old_path": "eq/a", "new_path": "eq/b"}]
        result = _render_guide(
            from_ver="3.39.0",
            to_ver="4.0.0",
            version_range=["4.0.0"],
            summary=[],
            cocos_table=[],
            renames=renames,
            unit_changes=[],
            type_changes=[],
            removals=[],
            additions=[],
            from_cocos=None,
            to_cocos=None,
            ids_filter=None,
            include_recipes=True,
        )
        assert "Code Update Recipes" in result
        assert "Path Renames" in result

    def test_no_recipes_when_disabled(self):
        from imas_codex.tools.migration_guide import _render_guide

        renames = [{"ids": "eq", "old_path": "eq/a", "new_path": "eq/b"}]
        result = _render_guide(
            from_ver="3.39.0",
            to_ver="4.0.0",
            version_range=["4.0.0"],
            summary=[],
            cocos_table=[],
            renames=renames,
            unit_changes=[],
            type_changes=[],
            removals=[],
            additions=[],
            from_cocos=None,
            to_cocos=None,
            ids_filter=None,
            include_recipes=False,
        )
        assert "Code Update Recipes" not in result


class TestCocosFactors:
    """Tests for COCOS factor computation."""

    def test_cocos_factors_computed(self):
        from imas_codex.tools.migration_guide import _compute_cocos_factors

        paths = [{"ids": "eq", "path": "eq/psi", "label": "psi_like", "source": "xml"}]
        result = _compute_cocos_factors(paths, from_cocos=11, to_cocos=17)
        assert len(result) == 1
        assert result[0]["factor"] == -1
        assert "Multiply" in result[0]["action"]

    def test_cocos_factors_same_convention(self):
        from imas_codex.tools.migration_guide import _compute_cocos_factors

        paths = [{"ids": "eq", "path": "eq/psi", "label": "psi_like", "source": "xml"}]
        result = _compute_cocos_factors(paths, from_cocos=11, to_cocos=11)
        assert len(result) == 1
        assert result[0]["factor"] is None
        assert "Unknown" in result[0]["action"]

    def test_cocos_factors_none_convention(self):
        from imas_codex.tools.migration_guide import _compute_cocos_factors

        paths = [{"ids": "eq", "path": "eq/psi", "label": "psi_like", "source": "xml"}]
        result = _compute_cocos_factors(paths, from_cocos=None, to_cocos=17)
        assert len(result) == 1
        assert result[0]["factor"] is None
        assert "Unknown" in result[0]["action"]

    def test_cocos_factors_no_change(self):
        from imas_codex.tools.migration_guide import _compute_cocos_factors

        paths = [{"ids": "eq", "path": "eq/ip", "label": "one_like", "source": "xml"}]
        result = _compute_cocos_factors(paths, from_cocos=11, to_cocos=17)
        assert len(result) == 1
        assert result[0]["factor"] == 1
        assert "No change needed" in result[0]["action"]

    def test_cocos_factors_unknown_label_returns_unity(self):
        from imas_codex.tools.migration_guide import _compute_cocos_factors

        paths = [
            {
                "ids": "eq",
                "path": "eq/unknown",
                "label": "nonexistent_label",
                "source": "xml",
            }
        ]
        result = _compute_cocos_factors(paths, from_cocos=11, to_cocos=17)
        assert len(result) == 1
        # cocos_sign returns 1 for unknown labels (with a warning)
        assert result[0]["factor"] == 1
        assert "No change needed" in result[0]["action"]


class TestVersionCocos:
    """Tests for _get_version_cocos fallback logic."""

    def test_dd3_fallback_to_cocos_11(self):
        from imas_codex.tools.migration_guide import _get_version_cocos

        gc = MagicMock()
        gc.query.return_value = [{"cocos": None}]
        assert _get_version_cocos(gc, "3.39.0") == 11

    def test_dd4_fallback_to_cocos_17(self):
        from imas_codex.tools.migration_guide import _get_version_cocos

        gc = MagicMock()
        gc.query.return_value = [{"cocos": None}]
        assert _get_version_cocos(gc, "4.0.0") == 17

    def test_explicit_cocos_from_graph(self):
        from imas_codex.tools.migration_guide import _get_version_cocos

        gc = MagicMock()
        gc.query.return_value = [{"cocos": 13}]
        assert _get_version_cocos(gc, "3.39.0") == 13

    def test_unknown_version_returns_none(self):
        from imas_codex.tools.migration_guide import _get_version_cocos

        gc = MagicMock()
        gc.query.return_value = []
        assert _get_version_cocos(gc, "5.0.0") is None


class TestVersionRange:
    """Tests for version range resolution."""

    def test_resolve_version_range_returns_sorted(self):
        gc = MagicMock()
        gc.query.return_value = [
            {"version": "3.40.0"},
            {"version": "3.41.0"},
            {"version": "4.0.0"},
        ]
        from imas_codex.tools.migration_guide import _resolve_version_range

        result = _resolve_version_range(gc, "3.39.0", "4.0.0")
        assert result == ["3.40.0", "3.41.0", "4.0.0"]

    def test_resolve_version_range_empty(self):
        gc = MagicMock()
        gc.query.return_value = []
        from imas_codex.tools.migration_guide import _resolve_version_range

        result = _resolve_version_range(gc, "3.39.0", "3.39.0")
        assert result == []


class TestGenerateMigrationGuide:
    """Tests for the top-level generate function with mocked graph."""

    def test_version_not_found(self):
        gc = MagicMock()
        gc.query.return_value = []
        from imas_codex.tools.migration_guide import generate_migration_guide

        result = generate_migration_guide(gc, "99.0.0", "100.0.0")
        assert "Error" in result
        assert "99.0.0" in result

    def test_no_versions_in_range(self):
        gc = MagicMock()
        # First two calls: version existence checks return results
        # Third call: version range returns empty
        gc.query.side_effect = [
            [{"v.id": "3.39.0"}],  # from_version exists
            [{"v.id": "3.39.0"}],  # to_version exists (same)
            [],  # version range is empty
        ]
        from imas_codex.tools.migration_guide import generate_migration_guide

        result = generate_migration_guide(gc, "3.39.0", "3.39.0")
        assert "Error" in result
        assert "No versions found" in result
