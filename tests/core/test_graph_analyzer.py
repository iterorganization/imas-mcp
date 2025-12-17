"""
Tests for graph_analyzer module.

Tests graph construction, structural analysis, and cross-IDS pattern detection.
"""

import pytest

from imas_codex.graph_analyzer import IMASGraphAnalyzer, analyze_imas_graphs


class TestIMASGraphAnalyzer:
    """Tests for the IMASGraphAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create an analyzer instance."""
        return IMASGraphAnalyzer()

    @pytest.fixture
    def sample_paths(self):
        """Create sample path data for testing."""
        return {
            "profiles_1d/electrons/temperature": {
                "data_type": "float",
                "units": "eV",
                "coordinates": ["rho_tor_norm"],
            },
            "profiles_1d/electrons/density": {
                "data_type": "float",
                "units": "m^-3",
                "coordinates": ["rho_tor_norm"],
            },
            "profiles_1d/ions/temperature": {
                "data_type": "float",
                "units": "eV",
                "coordinates": ["rho_tor_norm"],
            },
            "time": {
                "data_type": "float",
                "units": "s",
            },
            "global_quantities/ip": {
                "data_type": "float",
                "units": "A",
            },
        }

    @pytest.fixture
    def sample_paths_with_slashes(self):
        """Create sample path data using slash separator."""
        return {
            "core_profiles/profiles_1d/electrons/temperature": {
                "data_type": "float",
                "coordinates": ["time"],
            },
            "core_profiles/profiles_1d/electrons/density": {
                "data_type": "float",
                "coordinates": ["time"],
            },
            "core_profiles/time": {
                "data_type": "float",
            },
        }

    def test_build_ids_graph_basic(self, analyzer, sample_paths):
        """Test basic graph construction."""
        graph = analyzer.build_ids_graph("core_profiles", sample_paths)

        # Should have nodes for all path segments
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

        # Root nodes should exist
        assert "profiles_1d" in graph.nodes()
        assert "time" in graph.nodes()
        assert "global_quantities" in graph.nodes()

    def test_build_ids_graph_with_slashes(self, analyzer, sample_paths_with_slashes):
        """Test graph construction with slash-separated paths."""
        graph = analyzer.build_ids_graph("test", sample_paths_with_slashes)

        # Should handle slash separators
        assert graph.number_of_nodes() > 0
        assert "core_profiles" in graph.nodes()
        assert "core_profiles/profiles_1d" in graph.nodes()

    def test_build_ids_graph_hierarchy(self, analyzer, sample_paths):
        """Test that graph maintains hierarchical structure."""
        graph = analyzer.build_ids_graph("core_profiles", sample_paths)

        # Check parent-child relationships using dot separator
        # Graph should have edges from parent to child nodes
        nodes = list(graph.nodes())
        edges = list(graph.edges())

        # Should have edges connecting path segments
        assert len(edges) > 0

        # profiles_1d should be a parent of profiles_1d.electrons
        # (since sample_paths uses dot separator)
        if "profiles_1d" in nodes:
            # Find child edges
            children = [e[1] for e in edges if e[0] == "profiles_1d"]
            assert len(children) > 0

    def test_build_ids_graph_node_attributes(self, analyzer, sample_paths):
        """Test that nodes have correct attributes."""
        graph = analyzer.build_ids_graph("core_profiles", sample_paths)

        # Check that nodes have level and name attributes
        for _node, data in graph.nodes(data=True):
            assert "level" in data
            assert "name" in data
            assert isinstance(data["level"], int)
            assert data["level"] >= 0

    def test_analyze_ids_structure_basic_metrics(self, analyzer, sample_paths):
        """Test basic metrics in structure analysis."""
        result = analyzer.analyze_ids_structure("core_profiles", sample_paths)

        # Check basic metrics exist
        assert "basic_metrics" in result
        assert "total_nodes" in result["basic_metrics"]
        assert "total_edges" in result["basic_metrics"]
        assert "density" in result["basic_metrics"]
        assert "avg_clustering" in result["basic_metrics"]

        # Values should be reasonable
        assert result["basic_metrics"]["total_nodes"] > 0
        assert result["basic_metrics"]["density"] >= 0
        assert result["basic_metrics"]["density"] <= 1

    def test_analyze_ids_structure_hierarchy_metrics(self, analyzer, sample_paths):
        """Test hierarchy metrics in structure analysis."""
        result = analyzer.analyze_ids_structure("core_profiles", sample_paths)

        # Check hierarchy metrics
        assert "hierarchy_metrics" in result
        assert "max_depth" in result["hierarchy_metrics"]
        assert "avg_depth" in result["hierarchy_metrics"]
        assert "levels" in result["hierarchy_metrics"]
        assert "leaf_count" in result["hierarchy_metrics"]
        assert "root_count" in result["hierarchy_metrics"]

        # Depth should be reasonable for test data
        assert result["hierarchy_metrics"]["max_depth"] >= 2

    def test_analyze_ids_structure_branching_metrics(self, analyzer, sample_paths):
        """Test branching metrics in structure analysis."""
        result = analyzer.analyze_ids_structure("core_profiles", sample_paths)

        # Check branching metrics
        assert "branching_metrics" in result
        assert "avg_branching_factor" in result["branching_metrics"]
        assert "max_branching_factor" in result["branching_metrics"]
        assert "non_leaf_nodes" in result["branching_metrics"]
        assert "leaf_nodes" in result["branching_metrics"]

    def test_analyze_ids_structure_complexity_indicators(self, analyzer, sample_paths):
        """Test complexity indicators in structure analysis."""
        result = analyzer.analyze_ids_structure("core_profiles", sample_paths)

        # Check complexity indicators
        assert "complexity_indicators" in result
        assert "array_paths" in result["complexity_indicators"]
        assert "array_ratio" in result["complexity_indicators"]
        assert "time_dependent_paths" in result["complexity_indicators"]
        assert "time_dependent_ratio" in result["complexity_indicators"]

        # Should detect time-dependent path
        assert result["complexity_indicators"]["time_dependent_paths"] >= 1

    def test_analyze_ids_structure_key_nodes(self, analyzer, sample_paths):
        """Test key nodes identification."""
        result = analyzer.analyze_ids_structure("core_profiles", sample_paths)

        # Check key nodes
        assert "key_nodes" in result
        assert "most_connected" in result["key_nodes"]
        assert "deepest_paths" in result["key_nodes"]
        assert "root_categories" in result["key_nodes"]

        # Should have identified most connected nodes
        assert len(result["key_nodes"]["most_connected"]) > 0

    def test_get_most_connected_nodes(self, analyzer, sample_paths):
        """Test most connected nodes identification."""
        graph = analyzer.build_ids_graph("core_profiles", sample_paths)
        result = analyzer._get_most_connected_nodes(graph, top_n=3)

        # Should return list of node info
        assert isinstance(result, list)
        assert len(result) <= 3

        # Each result should have expected fields
        for node_info in result:
            assert "node" in node_info
            assert "centrality" in node_info
            assert "degree" in node_info
            assert "level" in node_info

    def test_analyze_cross_ids_patterns(self, analyzer, sample_paths):
        """Test cross-IDS pattern analysis."""
        all_ids_data = {
            "core_profiles": sample_paths,
            "equilibrium": {
                "time_slice/profiles_1d/psi": {"data_type": "float"},
                "time_slice/boundary/psi": {"data_type": "float"},
                "time": {"data_type": "float"},
            },
        }

        result = analyzer.analyze_cross_ids_patterns(all_ids_data)

        # Check overview section
        assert "overview" in result
        assert result["overview"]["total_ids"] == 2
        assert "total_nodes_all_ids" in result["overview"]
        assert "complexity_range" in result["overview"]

        # Check complexity rankings
        assert "complexity_rankings" in result
        assert "most_complex" in result["complexity_rankings"]
        assert "least_complex" in result["complexity_rankings"]
        assert "complexity_scores" in result["complexity_rankings"]

        # Check structural patterns
        assert "structural_patterns" in result
        assert "deepest_ids" in result["structural_patterns"]
        assert "most_branched" in result["structural_patterns"]
        assert "array_heavy" in result["structural_patterns"]

    def test_analyze_cross_ids_patterns_empty(self, analyzer):
        """Test cross-IDS analysis with empty data."""
        result = analyzer.analyze_cross_ids_patterns({})

        assert result["overview"]["total_ids"] == 0
        assert result["overview"]["avg_depth_across_ids"] == 0.0

    def test_graphs_are_cached(self, analyzer, sample_paths):
        """Test that analyzed graphs are cached."""
        analyzer.analyze_ids_structure("test_ids", sample_paths)

        assert "test_ids" in analyzer.graphs
        assert analyzer.graphs["test_ids"].number_of_nodes() > 0


class TestAnalyzeImasGraphs:
    """Tests for the analyze_imas_graphs function."""

    def test_analyze_imas_graphs_basic(self):
        """Test the main analysis function."""
        data_dict = {
            "ids_catalog": {
                "core_profiles": {
                    "paths": {
                        "profiles_1d/electrons/temperature": {"data_type": "float"},
                        "time": {"data_type": "float"},
                    }
                },
                "equilibrium": {
                    "paths": {
                        "time_slice/psi": {"data_type": "float"},
                    }
                },
            },
            "metadata": {"build_time": "2024-01-01T00:00:00"},
        }

        result = analyze_imas_graphs(data_dict)

        # Check main sections
        assert "graph_statistics" in result
        assert "structural_insights" in result
        assert "analysis_metadata" in result

        # Check graph statistics per IDS
        assert "core_profiles" in result["graph_statistics"]
        assert "equilibrium" in result["graph_statistics"]

        # Check metadata
        assert result["analysis_metadata"]["total_ids_analyzed"] == 2

    def test_analyze_imas_graphs_empty_catalog(self):
        """Test with empty IDS catalog."""
        data_dict = {"ids_catalog": {}, "metadata": {}}

        result = analyze_imas_graphs(data_dict)

        assert result["graph_statistics"] == {}
        assert result["analysis_metadata"]["total_ids_analyzed"] == 0

    def test_analyze_imas_graphs_missing_paths(self):
        """Test with IDS entries missing paths."""
        data_dict = {
            "ids_catalog": {
                "core_profiles": {
                    "paths": {"time": {"data_type": "float"}},
                },
                "no_paths": {
                    "description": "IDS without paths",
                },
            },
            "metadata": {},
        }

        result = analyze_imas_graphs(data_dict)

        # Only core_profiles should be analyzed
        assert "core_profiles" in result["graph_statistics"]
        assert "no_paths" not in result["graph_statistics"]
        assert result["analysis_metadata"]["total_ids_analyzed"] == 1
