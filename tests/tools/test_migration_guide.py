"""Tests for DD migration guide generator."""

from unittest.mock import MagicMock

import pytest

from imas_codex.models.migration_models import (
    CocosMigrationAdvice,
    CodeMigrationGuide,
    CodeUpdateAction,
    PathUpdateAdvice,
    TypeUpdateAdvice,
    generate_search_patterns,
)
from imas_codex.tools.migration_guide import (
    _compute_cocos_factors,
    _get_rename_lineages,
    _get_version_cocos,
    _render_guide,
    _render_rename_lineages_section,
    _resolve_version_range,
    build_migration_guide,
    format_migration_guide,
    generate_migration_guide,
)

# ---------------------------------------------------------------------------
# New model-based tests
# ---------------------------------------------------------------------------


class TestCodeMigrationModels:
    """Tests for new migration guide models."""

    def test_search_patterns_generation(self):
        patterns = generate_search_patterns(
            "equilibrium/time_slice/profiles_1d/psi", "cocos_sign_flip"
        )
        assert any("psi" in p for p in patterns)
        assert any("profiles_1d" in p for p in patterns)

    def test_search_patterns_short_path(self):
        patterns = generate_search_patterns("equilibrium", "path_rename")
        assert patterns == ["equilibrium"]

    def test_search_patterns_two_segments(self):
        patterns = generate_search_patterns("equilibrium/time", "unit_change")
        assert any("time" in p for p in patterns)
        # No parent_name for two-segment paths
        assert not any(".*time" in p and "equilibrium" not in p for p in patterns)

    def test_code_update_action(self):
        action = CodeUpdateAction(
            path="equilibrium/time_slice/profiles_1d/psi",
            ids="equilibrium",
            change_type="cocos_sign_flip",
            severity="required",
            description="COCOS 11\u219217: multiply by -1",
            cocos_factor=-1.0,
        )
        assert action.severity == "required"
        assert action.cocos_factor == -1.0
        assert action.change_type == "cocos_sign_flip"

    def test_code_update_action_defaults(self):
        action = CodeUpdateAction(
            path="eq/psi",
            ids="equilibrium",
            change_type="cocos_sign_flip",
            severity="optional",
            description="Test",
        )
        assert action.search_patterns == []
        assert action.path_fragments == []
        assert action.before == ""
        assert action.after == ""
        assert action.cocos_label is None
        assert action.old_path is None
        assert action.old_type is None
        assert action.old_units is None

    def test_migration_guide_model(self):
        guide = CodeMigrationGuide(
            from_version="3.39.0",
            to_version="4.0.0",
            cocos_change="11 \u2192 17",
            total_actions=5,
            required_count=3,
            optional_count=2,
        )
        assert guide.total_actions == 5
        assert guide.required_count == 3
        assert guide.optional_count == 2
        assert guide.cocos_change is not None

    def test_migration_guide_defaults(self):
        guide = CodeMigrationGuide(
            from_version="3.39.0",
            to_version="4.0.0",
        )
        assert guide.total_actions == 0
        assert guide.required_actions == []
        assert guide.optional_actions == []
        assert guide.ids_affected == []
        assert guide.cocos_advice is None
        assert guide.path_update_advice is None
        assert guide.type_update_advice is None

    def test_cocos_advice(self):
        advice = CocosMigrationAdvice(
            from_cocos=11,
            to_cocos=17,
            sign_flips=[{"path": "eq/ts/p1d/psi", "factor": -1}],
            no_change=[{"path": "eq/ts/p1d/rho", "factor": 1}],
        )
        assert len(advice.sign_flips) == 1
        assert len(advice.no_change) == 1
        assert advice.from_cocos == 11
        assert advice.to_cocos == 17

    def test_path_update_advice(self):
        advice = PathUpdateAdvice(
            renamed_paths=[{"old_path": "a/b", "new_path": "a/c"}],
            removed_paths=[{"path": "a/d"}],
            new_paths=[{"path": "a/e"}],
        )
        assert len(advice.renamed_paths) == 1
        assert len(advice.removed_paths) == 1
        assert len(advice.new_paths) == 1

    def test_type_update_advice(self):
        advice = TypeUpdateAdvice(
            type_changes=[
                {"path": "eq/flag", "old_type": "INT_0D", "new_type": "STR_0D"}
            ]
        )
        assert len(advice.type_changes) == 1


# ---------------------------------------------------------------------------
# New format/build tests
# ---------------------------------------------------------------------------


class TestFormatMigrationGuide:
    """Tests for the new format_migration_guide renderer."""

    def test_empty_guide(self):
        guide = CodeMigrationGuide(
            from_version="3.39.0",
            to_version="4.0.0",
        )
        result = format_migration_guide(guide)
        assert "3.39.0" in result
        assert "4.0.0" in result
        assert "## Summary" in result
        assert "## Verification Checklist" in result

    def test_guide_with_cocos(self):
        guide = CodeMigrationGuide(
            from_version="3.39.0",
            to_version="4.0.0",
            cocos_change="11 \u2192 17",
            total_actions=1,
            required_count=1,
            required_actions=[
                CodeUpdateAction(
                    path="equilibrium/time_slice/profiles_1d/psi",
                    ids="equilibrium",
                    change_type="cocos_sign_flip",
                    severity="required",
                    description="COCOS 11\u219217: multiply by -1",
                    search_patterns=["psi", "profiles_1d.*psi"],
                    cocos_factor=-1.0,
                    cocos_label="psi_like",
                )
            ],
            ids_affected=["equilibrium"],
            cocos_advice=CocosMigrationAdvice(
                from_cocos=11,
                to_cocos=17,
                sign_flips=[
                    {
                        "path": "equilibrium/time_slice/profiles_1d/psi",
                        "factor": -1,
                        "label": "psi_like",
                    }
                ],
            ),
        )
        result = format_migration_guide(guide)
        assert "COCOS change:" in result
        assert "Required Updates" in result
        assert "cocos_sign_flip" in result
        assert "Sign Flips Required" in result
        assert "psi_like" in result
        assert "COCOS" in result
        assert "Verify sign conventions" in result

    def test_guide_with_renames(self):
        guide = CodeMigrationGuide(
            from_version="3.39.0",
            to_version="4.0.0",
            total_actions=1,
            required_count=1,
            required_actions=[
                CodeUpdateAction(
                    path="eq/label",
                    ids="equilibrium",
                    change_type="path_rename",
                    severity="required",
                    description="Path renamed: eq/label \u2192 eq/name",
                    search_patterns=["label"],
                    old_path="eq/label",
                    new_path="eq/name",
                )
            ],
            ids_affected=["equilibrium"],
            global_search_patterns={"equilibrium": ["label"]},
        )
        result = format_migration_guide(guide)
        assert "Required Updates" in result
        assert "path_rename" in result
        assert "Search Strategy" in result
        assert "grep" in result

    def test_guide_with_optional_actions(self):
        guide = CodeMigrationGuide(
            from_version="3.39.0",
            to_version="4.0.0",
            total_actions=1,
            optional_count=1,
            optional_actions=[
                CodeUpdateAction(
                    path="eq/new_field",
                    ids="equilibrium",
                    change_type="new_path",
                    severity="optional",
                    description="New path available: eq/new_field",
                )
            ],
        )
        result = format_migration_guide(guide)
        assert "Optional Updates" in result
        assert "eq/new_field" in result

    def test_guide_optional_truncation(self):
        actions = [
            CodeUpdateAction(
                path=f"eq/field_{i}",
                ids="equilibrium",
                change_type="new_path",
                severity="optional",
                description=f"New path: eq/field_{i}",
            )
            for i in range(25)
        ]
        guide = CodeMigrationGuide(
            from_version="3.39.0",
            to_version="4.0.0",
            optional_actions=actions,
            optional_count=25,
            total_actions=25,
        )
        result = format_migration_guide(guide)
        assert "... and 5 more" in result


class TestBuildMigrationGuide:
    """Tests for build_migration_guide with mocked graph."""

    def _make_gc(self, query_returns=None):
        gc = MagicMock()
        gc.query.side_effect = query_returns or []
        return gc

    def test_build_empty_guide(self):
        gc = self._make_gc(
            [
                # _resolve_version_range
                [{"version": "4.0.0"}],
                # _get_version_cocos (from)
                [{"cocos": None}],
                # _get_version_cocos (to)
                [{"cocos": None}],
                # _get_cocos_table (skipped since no cocos_change when both 11)
                # Actually cocos_change = "11 \u2192 17" since 3.x=11, 4.x=17
                # Let me reconsider: from=3.39.0 -> fallback 11, to=4.0.0 -> fallback 17
                # So cocos_change WILL be set. Need cocos_table query:
                [],  # _get_cocos_table
                # _get_renames
                [],
                # _get_removals
                [],
                # _get_additions
                [],
                # _get_unit_changes
                [],
                # _get_type_changes
                [],
                # _get_semantic_doc_changes
                [],
            ]
        )
        guide = build_migration_guide(gc, "3.39.0", "4.0.0")
        assert isinstance(guide, CodeMigrationGuide)
        assert guide.from_version == "3.39.0"
        assert guide.to_version == "4.0.0"

    def test_build_with_cocos_change(self):
        gc = self._make_gc(
            [
                # _resolve_version_range
                [{"version": "4.0.0"}],
                # _get_version_cocos (from) - returns None, fallback to 11
                [{"cocos": None}],
                # _get_version_cocos (to) - returns None, fallback to 17
                [{"cocos": None}],
                # _get_cocos_table
                [
                    {
                        "ids": "equilibrium",
                        "path": "equilibrium/time_slice/profiles_1d/psi",
                        "label": "psi_like",
                        "source": "xml",
                    }
                ],
                # _get_renames
                [],
                # _get_removals
                [],
                # _get_additions
                [],
                # _get_unit_changes
                [],
                # _get_type_changes
                [],
                # _get_semantic_doc_changes
                [],
            ]
        )
        guide = build_migration_guide(gc, "3.39.0", "4.0.0")
        assert guide.cocos_change is not None
        assert guide.cocos_advice is not None
        # psi_like with COCOS 11->17 = factor -1 -> required
        assert guide.required_count >= 1 or guide.optional_count >= 1


# ---------------------------------------------------------------------------
# Legacy rendering tests (kept for backward compatibility)
# ---------------------------------------------------------------------------


class TestMigrationGuideRendering:
    """Unit tests for migration guide rendering (no graph needed)."""

    def test_render_empty_guide(self):
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
        removals = [
            {"ids": "equilibrium", "path": "eq/old_field"},
            {"ids": "equilibrium", "path": "eq/dead_field"},
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
        assert "eq/old_field" in result
        assert "eq/dead_field" in result

    def test_render_summary_counts(self):
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
        paths = [{"ids": "eq", "path": "eq/psi", "label": "psi_like", "source": "xml"}]
        result = _compute_cocos_factors(paths, from_cocos=11, to_cocos=17)
        assert len(result) == 1
        assert result[0]["factor"] == -1
        assert "Multiply" in result[0]["action"]

    def test_cocos_factors_same_convention(self):
        paths = [{"ids": "eq", "path": "eq/psi", "label": "psi_like", "source": "xml"}]
        result = _compute_cocos_factors(paths, from_cocos=11, to_cocos=11)
        assert len(result) == 1
        assert result[0]["factor"] is None
        assert "Unknown" in result[0]["action"]

    def test_cocos_factors_none_convention(self):
        paths = [{"ids": "eq", "path": "eq/psi", "label": "psi_like", "source": "xml"}]
        result = _compute_cocos_factors(paths, from_cocos=None, to_cocos=17)
        assert len(result) == 1
        assert result[0]["factor"] is None
        assert "Unknown" in result[0]["action"]

    def test_cocos_factors_no_change(self):
        paths = [{"ids": "eq", "path": "eq/ip", "label": "one_like", "source": "xml"}]
        result = _compute_cocos_factors(paths, from_cocos=11, to_cocos=17)
        assert len(result) == 1
        assert result[0]["factor"] == 1
        assert "No change needed" in result[0]["action"]

    def test_cocos_factors_unknown_label_returns_unity(self):
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
        gc = MagicMock()
        gc.query.return_value = [{"cocos": None}]
        assert _get_version_cocos(gc, "3.39.0") == 11

    def test_dd4_fallback_to_cocos_17(self):
        gc = MagicMock()
        gc.query.return_value = [{"cocos": None}]
        assert _get_version_cocos(gc, "4.0.0") == 17

    def test_explicit_cocos_from_graph(self):
        gc = MagicMock()
        gc.query.return_value = [{"cocos": 13}]
        assert _get_version_cocos(gc, "3.39.0") == 13

    def test_unknown_version_returns_none(self):
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
        result = _resolve_version_range(gc, "3.39.0", "4.0.0")
        assert result == ["3.40.0", "3.41.0", "4.0.0"]

    def test_resolve_version_range_empty(self):
        gc = MagicMock()
        gc.query.return_value = []
        result = _resolve_version_range(gc, "3.39.0", "3.39.0")
        assert result == []


class TestGenerateMigrationGuide:
    """Tests for the top-level generate function with mocked graph."""

    def test_version_not_found(self):
        gc = MagicMock()
        gc.query.return_value = []
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
        result = generate_migration_guide(gc, "3.39.0", "3.39.0")
        assert "Error" in result
        assert "No versions found" in result


class TestConventionChanges:
    """Tests for convention change handling in migration guide."""

    def test_format_includes_convention_changes_section(self):
        """Formatted guide should include Convention Changes section when definition_change actions exist."""
        guide = CodeMigrationGuide(
            from_version="3.42.0",
            to_version="4.0.0",
            required_actions=[
                CodeUpdateAction(
                    path="pf_active/circuit/connections",
                    ids="pf_active",
                    change_type="definition_change",
                    severity="required",
                    description="Convention change (sign_convention): pf_active/circuit/connections",
                    before="Matrix elements are 1 or 0.",
                    after="Matrix elements are 1 if positive side, -1 if negative side, or 0.",
                    search_patterns=["circuit/connections", "connections"],
                    path_fragments=["circuit", "connections"],
                ),
            ],
            optional_actions=[],
            total_actions=1,
            required_count=1,
            optional_count=0,
            ids_affected=["pf_active"],
        )
        output = format_migration_guide(guide)
        assert "Convention Changes" in output
        assert "pf_active/circuit/connections" in output
        assert "**BREAKING**" in output
        assert "silently incorrect results" in output

    def test_format_no_convention_section_when_no_definition_changes(self):
        """Convention Changes section should not appear when there are no definition_change actions."""
        guide = CodeMigrationGuide(
            from_version="3.42.0",
            to_version="4.0.0",
            required_actions=[
                CodeUpdateAction(
                    path="some/path",
                    ids="some",
                    change_type="type_change",
                    severity="required",
                    description="Type change",
                    search_patterns=[],
                    path_fragments=["path"],
                ),
            ],
            optional_actions=[],
            total_actions=1,
            required_count=1,
            optional_count=0,
            ids_affected=["some"],
        )
        output = format_migration_guide(guide)
        assert "Convention Changes" not in output

    def test_advisory_convention_change_shows_advisory_badge(self):
        """Optional convention changes should show 'advisory' badge."""
        guide = CodeMigrationGuide(
            from_version="3.42.0",
            to_version="4.0.0",
            required_actions=[],
            optional_actions=[
                CodeUpdateAction(
                    path="some/phi",
                    ids="some",
                    change_type="definition_change",
                    severity="optional",
                    description="Convention change (coordinate_convention): some/phi",
                    before="Toroidal angle",
                    after="Toroidal angle (right-handed)",
                    search_patterns=["phi"],
                    path_fragments=["phi"],
                ),
            ],
            total_actions=1,
            required_count=0,
            optional_count=1,
            ids_affected=["some"],
        )
        output = format_migration_guide(guide)
        assert "Convention Changes" in output
        assert "advisory" in output


# ---------------------------------------------------------------------------
# Rename Lineages tests
# ---------------------------------------------------------------------------


class TestRenderRenameLineages:
    """Unit tests for _render_rename_lineages_section (no graph needed)."""

    def _make_lineage(
        self,
        chain_ids: list[str],
        versions: list[str | None],
        ids_name: str = "summary",
    ) -> dict:
        chain = [
            {"path": path, "introduced": ver}
            for path, ver in zip(chain_ids, versions, strict=True)
        ]
        return {
            "ids": ids_name,
            "start_path": chain_ids[0],
            "end_path": chain_ids[-1],
            "hops": len(chain_ids) - 1,
            "chain": chain,
        }

    def test_empty_returns_empty_string(self):
        result = _render_rename_lineages_section([], "3.22.0", "4.1.0")
        assert result == ""

    def test_section_header_present(self):
        lineage = self._make_lineage(
            ["summary/a", "summary/b", "summary/c"],
            [None, "4.0.0", "4.1.0"],
        )
        result = _render_rename_lineages_section([lineage], "3.22.0", "4.1.0")
        assert "Rename Lineages (Multi-Hop)" in result

    def test_version_columns_show_from_to(self):
        lineage = self._make_lineage(
            ["summary/a", "summary/b", "summary/c"],
            [None, "4.0.0", "4.1.0"],
        )
        result = _render_rename_lineages_section([lineage], "3.22.0", "4.1.0")
        assert "3.22.0" in result
        assert "4.1.0" in result

    def test_original_and_final_paths_present(self):
        lineage = self._make_lineage(
            ["summary/a", "summary/b", "summary/c"],
            [None, "4.0.0", "4.1.0"],
        )
        result = _render_rename_lineages_section([lineage], "3.22.0", "4.1.0")
        assert "summary/a" in result
        assert "summary/c" in result

    def test_intermediate_path_in_chain_column(self):
        lineage = self._make_lineage(
            ["summary/a", "summary/b", "summary/c"],
            [None, "4.0.0", "4.1.0"],
        )
        result = _render_rename_lineages_section([lineage], "3.22.0", "4.1.0")
        assert "summary/b" in result

    def test_version_annotation_on_hops(self):
        lineage = self._make_lineage(
            ["summary/a", "summary/b", "summary/c"],
            [None, "4.0.0", "4.1.0"],
        )
        result = _render_rename_lineages_section([lineage], "3.22.0", "4.1.0")
        assert "v4.0.0" in result

    def test_recipe_included_when_requested(self):
        lineage = self._make_lineage(
            ["summary/a", "summary/b", "summary/c"],
            [None, "4.0.0", "4.1.0"],
        )
        result = _render_rename_lineages_section(
            [lineage], "3.22.0", "4.1.0", include_recipes=True
        )
        assert "Multi-Hop Rename Recipe" in result

    def test_recipe_absent_when_disabled(self):
        lineage = self._make_lineage(
            ["summary/a", "summary/b", "summary/c"],
            [None, "4.0.0", "4.1.0"],
        )
        result = _render_rename_lineages_section(
            [lineage], "3.22.0", "4.1.0", include_recipes=False
        )
        assert "Recipe" not in result

    def test_multiple_lineages_all_present(self):
        l1 = self._make_lineage(
            ["a/x", "a/y", "a/z"], [None, "4.0.0", "4.1.0"], ids_name="a"
        )
        l2 = self._make_lineage(
            ["b/x", "b/y", "b/z"], [None, "4.0.0", "4.1.0"], ids_name="b"
        )
        result = _render_rename_lineages_section([l1, l2], "3.22.0", "4.1.0")
        assert "Rename Lineages (Multi-Hop) (2)" in result
        assert "a/z" in result
        assert "b/z" in result

    def test_hop_without_version_renders_gracefully(self):
        lineage = self._make_lineage(
            ["summary/a", "summary/b", "summary/c"],
            [None, None, None],  # No version info
        )
        result = _render_rename_lineages_section([lineage], "3.22.0", "4.1.0")
        assert "summary/b" in result
        # Should not error, and chain column should have arrows
        assert "→" in result


class TestGetRenameLineagesFiltering:
    """Unit tests for _get_rename_lineages version filtering logic."""

    def _make_gc(self, chain_rows: list[dict], version_rows: list[dict]) -> MagicMock:
        gc = MagicMock()
        gc.query.side_effect = [chain_rows, version_rows]
        return gc

    def _chain_row(
        self,
        chain_ids: list[str],
        ids: str = "summary",
        start_introduced: str | None = "3.22.0",
        start_deprecated: str | None = None,
        end_introduced: str | None = "4.1.0",
        end_deprecated: str | None = None,
    ) -> dict:
        return {
            "start_path": chain_ids[0],
            "end_path": chain_ids[-1],
            "ids": ids,
            "chain_ids": chain_ids,
            "hops": len(chain_ids) - 1,
            "start_introduced": start_introduced,
            "start_deprecated": start_deprecated,
            "end_introduced": end_introduced,
            "end_deprecated": end_deprecated,
        }

    def test_valid_chain_included(self):
        chain = ["summary/a", "summary/b", "summary/c"]
        gc = self._make_gc(
            [self._chain_row(chain)],
            [{"id": "summary/b", "introduced_in": "4.0.0"}],
        )
        result = _get_rename_lineages(gc, "3.22.0", "4.1.0")
        assert len(result) == 1
        assert result[0]["start_path"] == "summary/a"
        assert result[0]["end_path"] == "summary/c"
        assert result[0]["hops"] == 2

    def test_start_not_introduced_by_from_version_excluded(self):
        chain = ["summary/a", "summary/b", "summary/c"]
        gc = self._make_gc(
            # start introduced AFTER from_version
            [self._chain_row(chain, start_introduced="4.0.0")],
            [],
        )
        result = _get_rename_lineages(gc, "3.22.0", "4.1.0")
        assert result == []

    def test_start_deprecated_before_from_version_excluded(self):
        chain = ["summary/a", "summary/b", "summary/c"]
        gc = self._make_gc(
            # start deprecated before or at from_version
            [self._chain_row(chain, start_deprecated="3.20.0")],
            [],
        )
        result = _get_rename_lineages(gc, "3.22.0", "4.1.0")
        assert result == []

    def test_end_not_introduced_by_to_version_excluded(self):
        chain = ["summary/a", "summary/b", "summary/c"]
        gc = self._make_gc(
            # end introduced AFTER to_version
            [self._chain_row(chain, end_introduced="5.0.0")],
            [],
        )
        result = _get_rename_lineages(gc, "3.22.0", "4.1.0")
        assert result == []

    def test_chain_per_hop_versions_attached(self):
        chain = ["summary/a", "summary/b", "summary/c"]
        gc = self._make_gc(
            [self._chain_row(chain)],
            [
                {"id": "summary/b", "introduced_in": "4.0.0"},
                {"id": "summary/c", "introduced_in": "4.1.0"},
            ],
        )
        result = _get_rename_lineages(gc, "3.22.0", "4.1.0")
        assert len(result) == 1
        chain_out = result[0]["chain"]
        assert chain_out[0]["path"] == "summary/a"
        assert chain_out[0]["introduced"] == "3.22.0"  # start_introduced
        # intermediate node should have queried version
        b_entry = next(h for h in chain_out if h["path"] == "summary/b")
        assert b_entry["introduced"] == "4.0.0"

    def test_empty_graph_returns_empty_list(self):
        gc = MagicMock()
        gc.query.return_value = []
        result = _get_rename_lineages(gc, "3.22.0", "4.1.0")
        assert result == []

    def test_ids_filter_passed_to_query(self):
        gc = MagicMock()
        gc.query.return_value = []
        _get_rename_lineages(gc, "3.22.0", "4.1.0", ids_filter="summary")
        call_args = gc.query.call_args
        assert "ids_filter" in call_args.kwargs

    def test_query_exception_returns_empty_list(self):
        gc = MagicMock()
        gc.query.side_effect = RuntimeError("DB error")
        result = _get_rename_lineages(gc, "3.22.0", "4.1.0")
        assert result == []


class TestFormatMigrationGuideWithLineages:
    """Tests for format_migration_guide including rename_lineages section."""

    def test_lineages_section_present_when_populated(self):
        guide = CodeMigrationGuide(
            from_version="3.22.0",
            to_version="4.1.0",
            rename_lineages=[
                {
                    "ids": "summary",
                    "start_path": "summary/local/magnetic_axis/b_field",
                    "end_path": "summary/local/magnetic_axis/b_field_phi",
                    "hops": 2,
                    "chain": [
                        {
                            "path": "summary/local/magnetic_axis/b_field",
                            "introduced": "3.22.0",
                        },
                        {
                            "path": "summary/local/magnetic_axis/b_field_tor",
                            "introduced": "4.0.0",
                        },
                        {
                            "path": "summary/local/magnetic_axis/b_field_phi",
                            "introduced": "4.1.0",
                        },
                    ],
                }
            ],
        )
        result = format_migration_guide(guide)
        assert "Rename Lineages" in result
        assert "b_field_phi" in result
        assert "b_field_tor" in result

    def test_lineages_section_absent_when_empty(self):
        guide = CodeMigrationGuide(
            from_version="3.22.0",
            to_version="4.1.0",
            rename_lineages=[],
        )
        result = format_migration_guide(guide)
        assert "Rename Lineages" not in result

    def test_lineages_section_absent_when_field_default(self):
        guide = CodeMigrationGuide(
            from_version="3.22.0",
            to_version="4.1.0",
        )
        result = format_migration_guide(guide)
        assert "Rename Lineages" not in result


class TestBuildMigrationGuideWithLineages:
    """Tests for build_migration_guide including rename_lineages field."""

    def _make_gc(self, query_returns: list) -> MagicMock:
        gc = MagicMock()
        gc.query.side_effect = query_returns
        return gc

    def test_lineages_populated_from_graph(self):
        """build_migration_guide should populate rename_lineages from _get_rename_lineages."""
        chain = ["summary/a", "summary/b", "summary/c"]
        gc = self._make_gc(
            [
                # _resolve_version_range
                [{"version": "4.0.0"}, {"version": "4.1.0"}],
                # _get_version_cocos (from) — None, fallback to 11
                [{"cocos": None}],
                # _get_version_cocos (to) — None, fallback to 17
                [{"cocos": None}],
                # _get_cocos_table
                [],
                # _get_renames
                [],
                # _get_removals
                [],
                # _get_additions
                [],
                # _get_unit_changes
                [],
                # _get_type_changes
                [],
                # _get_semantic_doc_changes
                [],
                # _get_rename_lineages: chain query
                [
                    {
                        "start_path": chain[0],
                        "end_path": chain[-1],
                        "ids": "summary",
                        "chain_ids": chain,
                        "hops": 2,
                        "start_introduced": "3.22.0",
                        "start_deprecated": None,
                        "end_introduced": "4.1.0",
                        "end_deprecated": None,
                    }
                ],
                # _get_rename_lineages: intermediate node versions
                [{"id": "summary/b", "introduced_in": "4.0.0"}],
            ]
        )
        guide = build_migration_guide(gc, "3.22.0", "4.1.0")
        assert isinstance(guide.rename_lineages, list)
        assert len(guide.rename_lineages) == 1
        assert guide.rename_lineages[0]["start_path"] == "summary/a"
        assert guide.rename_lineages[0]["hops"] == 2

    def test_lineages_empty_when_no_multi_hop(self):
        gc = self._make_gc(
            [
                [{"version": "4.0.0"}],  # _resolve_version_range
                [{"cocos": None}],  # from cocos
                [{"cocos": None}],  # to cocos
                [],  # _get_cocos_table
                [],  # _get_renames
                [],  # _get_removals
                [],  # _get_additions
                [],  # _get_unit_changes
                [],  # _get_type_changes
                [],  # _get_semantic_doc_changes
                [],  # _get_rename_lineages: chain query (empty)
            ]
        )
        guide = build_migration_guide(gc, "3.39.0", "4.0.0")
        assert guide.rename_lineages == []


class TestRenameLineagesLiveGraph:
    """Integration tests against the live Neo4j graph.

    Skipped automatically if no multi-hop chains exist in the graph.
    """

    @pytest.fixture(autouse=True)
    def _skip_without_neo4j(self):
        """Skip if the graph client is unavailable."""
        try:
            from imas_codex.graph.client import get_graph_client

            gc = get_graph_client()
            gc.query("RETURN 1 AS ok")
            self._gc = gc
        except Exception:
            pytest.skip("Neo4j graph not available")

    def test_live_lineages_structure(self):
        """Live query: multi-hop chains returned should have expected structure."""
        lineages = _get_rename_lineages(self._gc, "3.22.0", "4.1.0")
        if not lineages:
            pytest.skip("no multi-hop chains in 3.22.0→4.1.0 range")

        entry = lineages[0]
        assert "ids" in entry
        assert "start_path" in entry
        assert "end_path" in entry
        assert "hops" in entry
        assert entry["hops"] >= 2
        assert "chain" in entry
        assert len(entry["chain"]) >= 3

    def test_live_known_chain_summary_b_field(self):
        """summary/local/magnetic_axis/b_field is a known 2-hop chain."""
        lineages = _get_rename_lineages(self._gc, "3.22.0", "4.1.0")
        if not lineages:
            pytest.skip("no multi-hop chains in 3.22.0→4.1.0 range")

        starts = {e["start_path"] for e in lineages}
        assert "summary/local/magnetic_axis/b_field" in starts, (
            "Expected the known summary/local/magnetic_axis/b_field→tor→phi chain"
        )

    def test_live_render_no_crash(self):
        """Rendering live lineages should not raise."""
        lineages = _get_rename_lineages(self._gc, "3.22.0", "4.1.0")
        if not lineages:
            pytest.skip("no multi-hop chains in 3.22.0→4.1.0 range")

        result = _render_rename_lineages_section(lineages, "3.22.0", "4.1.0")
        assert "Rename Lineages (Multi-Hop)" in result
        assert len(result) > 100
