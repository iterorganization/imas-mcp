"""Tests for static tree extraction."""

import pytest

from imas_codex.mdsplus.static import _compute_parent_path, get_static_tree_config


class TestComputeParentPath:
    def test_subtree_path(self):
        assert _compute_parent_path("\\STATIC::TOP.C.R") == "\\STATIC::TOP.C"

    def test_top_level(self):
        assert _compute_parent_path("\\STATIC::TOP") is None

    def test_deep_path(self):
        assert _compute_parent_path("\\STATIC::TOP.C.R.XSECT") == "\\STATIC::TOP.C.R"

    def test_no_subtree(self):
        assert _compute_parent_path("\\TOP.SUB") == "\\TOP"

    def test_single_node(self):
        assert _compute_parent_path("\\TOP") is None


class TestGetStaticTreeConfig:
    def test_tcv_has_static_trees(self):
        configs = get_static_tree_config("tcv")
        assert len(configs) >= 1
        assert configs[0]["tree_name"] == "static"

    def test_tcv_versions(self):
        configs = get_static_tree_config("tcv")
        versions = configs[0].get("versions", [])
        assert len(versions) == 8
        assert versions[0]["version"] == 1
        assert versions[0]["first_shot"] == 1
        assert versions[-1]["version"] == 8

    def test_tcv_systems(self):
        configs = get_static_tree_config("tcv")
        systems = configs[0].get("systems", [])
        symbols = {s["symbol"] for s in systems}
        assert "C" in symbols  # Coils
        assert "V" in symbols  # Vessel
        assert "F" in symbols  # Flux loops
        assert "M" in symbols  # Magnetic probes
        assert "T" in symbols  # Tiles

    def test_nonexistent_facility(self):
        with pytest.raises(ValueError):
            get_static_tree_config("nonexistent_facility_xyz")
