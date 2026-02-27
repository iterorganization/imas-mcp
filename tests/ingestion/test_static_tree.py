"""Tests for static tree extraction."""

import pytest

from imas_codex.mdsplus.static import (
    _compute_parent_path,
    get_static_tree_config,
    merge_units_into_data,
)


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


class TestMergeUnitsIntoData:
    def test_merges_units_by_path(self):
        data = {
            "versions": {
                "1": {
                    "nodes": [
                        {"path": "\\STATIC::TOP.C.R", "node_type": "NUMERIC"},
                        {"path": "\\STATIC::TOP.C.Z", "node_type": "NUMERIC"},
                        {"path": "\\STATIC::TOP", "node_type": "STRUCTURE"},
                    ],
                    "node_count": 3,
                }
            }
        }
        units = {
            "\\STATIC::TOP.C.R": "m",
            "\\STATIC::TOP.C.Z": "m",
        }
        updated = merge_units_into_data(data, units)
        assert updated == 2
        nodes = data["versions"]["1"]["nodes"]
        assert nodes[0]["units"] == "m"
        assert nodes[1]["units"] == "m"
        assert "units" not in nodes[2]  # STRUCTURE node — no units

    def test_units_across_versions(self):
        data = {
            "versions": {
                "1": {
                    "nodes": [
                        {"path": "\\STATIC::TOP.C.R", "node_type": "NUMERIC"},
                    ],
                    "node_count": 1,
                },
                "2": {
                    "nodes": [
                        {"path": "\\STATIC::TOP.C.R", "node_type": "NUMERIC"},
                    ],
                    "node_count": 1,
                },
            }
        }
        units = {"\\STATIC::TOP.C.R": "m"}
        updated = merge_units_into_data(data, units)
        # Same path in both versions gets units
        assert updated == 2
        assert data["versions"]["1"]["nodes"][0]["units"] == "m"
        assert data["versions"]["2"]["nodes"][0]["units"] == "m"

    def test_empty_units(self):
        data = {
            "versions": {
                "1": {
                    "nodes": [
                        {"path": "\\STATIC::TOP.C.R", "node_type": "NUMERIC"},
                    ],
                    "node_count": 1,
                }
            }
        }
        updated = merge_units_into_data(data, {})
        assert updated == 0
        assert "units" not in data["versions"]["1"]["nodes"][0]

    def test_skips_error_versions(self):
        data = {
            "versions": {
                "1": {"error": "tree not found"},
            }
        }
        updated = merge_units_into_data(data, {"\\STATIC::TOP.C.R": "m"})
        assert updated == 0
