"""Tests for static tree extraction and IMAS mapping."""

import pytest

from imas_codex.mdsplus.static import _compute_parent_path, get_static_tree_config
from imas_codex.mdsplus.static_mapping import (
    TCV_STATIC_MAPPINGS,
    MappingResult,
    StaticMapping,
    build_mapping_proposals,
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


class TestMappingTable:
    def test_table_has_entries(self):
        assert len(TCV_STATIC_MAPPINGS) > 20

    def test_coil_mappings(self):
        coil_maps = [m for m in TCV_STATIC_MAPPINGS if m.system_symbol == "C"]
        assert len(coil_maps) >= 4  # R, Z, W, H at minimum
        assert all("pf_active" in m.imas_ids for m in coil_maps)

    def test_flux_loop_mappings(self):
        fl_maps = [m for m in TCV_STATIC_MAPPINGS if m.system_symbol == "F"]
        assert len(fl_maps) >= 2  # R, Z
        assert all("magnetics" in m.imas_ids for m in fl_maps)

    def test_probe_mappings(self):
        probe_maps = [m for m in TCV_STATIC_MAPPINGS if m.system_symbol == "M"]
        assert len(probe_maps) >= 2
        assert all("magnetics" in m.imas_ids for m in probe_maps)

    def test_wall_mappings(self):
        wall_maps = [m for m in TCV_STATIC_MAPPINGS if m.system_symbol == "T"]
        assert len(wall_maps) >= 2
        assert all("wall" in m.imas_ids for m in wall_maps)

    def test_confidence_range(self):
        for m in TCV_STATIC_MAPPINGS:
            assert 0.0 <= m.confidence <= 1.0, (
                f"Bad confidence for {m.system_symbol}/{m.parameter}"
            )


class TestBuildMappingProposals:
    """Test mapping proposal generation with synthetic tree data."""

    @pytest.fixture
    def sample_tree_data(self) -> dict:
        return {
            "tree_name": "static",
            "versions": {
                "1": {
                    "version": 1,
                    "node_count": 5,
                    "nodes": [
                        {
                            "path": "\\STATIC::TOP.C.R",
                            "name": "R",
                            "node_type": "NUMERIC",
                            "tags": ["\\R_C"],
                            "shape": [29],
                            "dtype": "float64",
                        },
                        {
                            "path": "\\STATIC::TOP.C.Z",
                            "name": "Z",
                            "node_type": "NUMERIC",
                            "tags": ["\\Z_C"],
                            "shape": [29],
                            "dtype": "float64",
                        },
                        {
                            "path": "\\STATIC::TOP.F.R",
                            "name": "R",
                            "node_type": "NUMERIC",
                            "tags": ["\\R_F"],
                            "shape": [61],
                            "dtype": "float64",
                        },
                        {
                            "path": "\\STATIC::TOP.M.R",
                            "name": "R",
                            "node_type": "NUMERIC",
                            "tags": ["\\R_M"],
                            "shape": [38],
                            "dtype": "float64",
                        },
                        {
                            "path": "\\STATIC::TOP.MUT_F_A",
                            "name": "MUT_F_A",
                            "node_type": "NUMERIC",
                            "tags": ["\\MUT_F_A"],
                            "shape": [61, 19],
                            "dtype": "float64",
                        },
                    ],
                    "tags": {},
                },
            },
            "diff": {"added": {}, "removed": {}},
        }

    def test_builds_proposals(self, sample_tree_data):
        result = build_mapping_proposals("tcv", sample_tree_data)
        assert isinstance(result, MappingResult)
        assert len(result.proposals) > 0

    def test_maps_coil_r(self, sample_tree_data):
        result = build_mapping_proposals("tcv", sample_tree_data)
        coil_r = [p for p in result.proposals if p["source_tag"] == "R_C"]
        assert len(coil_r) == 1
        assert "pf_active" in coil_r[0]["target_path"]

    def test_maps_flux_loop(self, sample_tree_data):
        result = build_mapping_proposals("tcv", sample_tree_data)
        fl_r = [p for p in result.proposals if p["source_tag"] == "R_F"]
        assert len(fl_r) == 1
        assert "flux_loop" in fl_r[0]["target_path"]

    def test_maps_probe(self, sample_tree_data):
        result = build_mapping_proposals("tcv", sample_tree_data)
        m_r = [p for p in result.proposals if p["source_tag"] == "R_M"]
        assert len(m_r) == 1
        assert "bpol_probe" in m_r[0]["target_path"]

    def test_unmapped_greens_function(self, sample_tree_data):
        result = build_mapping_proposals("tcv", sample_tree_data)
        # MUT_F_A (mutual inductance Green's function) should be unmapped
        # because it's \MUT_F_A pattern, not \<PARAM>_<SYSTEM>
        assert any("MUT_F_A" in p for p in result.unmapped_nodes)

    def test_stats(self, sample_tree_data):
        result = build_mapping_proposals("tcv", sample_tree_data)
        assert "mapped" in result.stats
        assert "unmapped" in result.stats
        assert result.stats["mapped"] > 0
