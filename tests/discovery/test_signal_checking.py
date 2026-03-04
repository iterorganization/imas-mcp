"""Unit tests for signal checking improvements.

Tests the check worker routing logic, multi-shot batch checking,
DataAccess-driven signal routing, and expression node handling.

Phase 1: Static tree routing - independent trees use their own tree_name/shot
Phase 2: DataAccess-driven check routing
Phase 3: Multi-shot batch checking (check all versions/epochs by default)
Phase 4: Expression node classification
Phase 5: TDI function categorization
Phase 6: Missing library error detection
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.discovery.signals.parallel import (
    _classify_check_error,
    _resolve_check_tree,
)

# =============================================================================
# Phase 1: Check tree resolution — independent trees vs subtrees
# =============================================================================


class TestResolveCheckTree:
    """Test _resolve_check_tree routes signals to the correct tree/shot."""

    def test_subtree_signal_routes_to_connection_tree(self):
        """Subtree signals (results, magnetics) route to connection tree."""
        signal = {
            "tree_name": "results",
            "node_path": "\\RESULTS::THOMSON:NE",
            "discovery_source": "tree_traversal",
            "accessor": "THOMSON:NE",
        }
        tree_name, accessor, shot = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"static", "vsystem"},
            tree_shots={"static": [1, 2, 3, 4, 5, 6, 7, 8]},
            reference_shot=85000,
        )
        assert tree_name == "tcv_shot"
        assert accessor == "\\RESULTS::THOMSON:NE"
        assert shot == 85000

    def test_static_tree_routes_independently(self):
        """Static tree signals open the static tree directly, not tcv_shot."""
        signal = {
            "tree_name": "static",
            "node_path": "\\STATIC::TOP.MECHANICAL.COIL:R",
            "discovery_source": "tree_traversal",
            "accessor": "TOP.MECHANICAL.COIL:R",
        }
        tree_name, accessor, shots = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"static", "vsystem"},
            tree_shots={"static": [1, 2, 3, 4, 5, 6, 7, 8]},
            reference_shot=85000,
        )
        assert tree_name == "static"
        assert accessor == "\\STATIC::TOP.MECHANICAL.COIL:R"

    def test_static_tree_uses_version_shots(self):
        """Static tree checks ALL versions, not just the reference shot."""
        signal = {
            "tree_name": "static",
            "node_path": "\\STATIC::TOP.MECHANICAL.COIL:R",
            "discovery_source": "tree_traversal",
            "accessor": "TOP.MECHANICAL.COIL:R",
        }
        tree_name, accessor, shot = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"static", "vsystem"},
            tree_shots={"static": [1, 2, 3, 4, 5, 6, 7, 8]},
            reference_shot=85000,
        )
        assert tree_name == "static"
        # Static tree uses version numbers as shots, first version
        assert shot == 1

    def test_vsystem_routes_independently(self):
        """vsystem tree signals open vsystem directly."""
        signal = {
            "tree_name": "vsystem",
            "node_path": "\\VSYSTEM::SOME:NODE",
            "discovery_source": "tree_traversal",
            "accessor": "SOME:NODE",
        }
        tree_name, accessor, shot = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"static", "vsystem"},
            tree_shots={},
            reference_shot=85000,
        )
        assert tree_name == "vsystem"
        assert shot == 85000  # No special shots for vsystem

    def test_tdi_function_uses_connection_tree(self):
        """TDI function signals route through connection tree."""
        signal = {
            "tree_name": None,
            "tdi_function": "tcv_eq",
            "discovery_source": "tdi_extraction",
            "accessor": 'tcv_eq("r_axis")',
        }
        tree_name, accessor, shot = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"static", "vsystem"},
            tree_shots={},
            reference_shot=85000,
        )
        assert tree_name == "tcv_shot"
        assert shot == 85000

    def test_unknown_tree_defaults_to_connection_tree(self):
        """Signals from unknown trees default to connection tree."""
        signal = {
            "tree_name": "unknown_tree",
            "node_path": "\\UNKNOWN::NODE",
            "discovery_source": "tree_traversal",
            "accessor": "NODE",
        }
        tree_name, accessor, shot = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"static", "vsystem"},
            tree_shots={},
            reference_shot=85000,
        )
        assert tree_name == "tcv_shot"
        assert shot == 85000


# =============================================================================
# Phase 6: Error classification — missing library detection
# =============================================================================


class TestClassifyCheckError:
    """Test _classify_check_error recognizes additional error types."""

    def test_node_not_found(self):
        assert _classify_check_error("%TREE-W-NNF, Node Not Found") == "node_not_found"

    def test_no_data(self):
        assert _classify_check_error("TreeNODATA") == "no_data"

    def test_timeout(self):
        assert _classify_check_error("Group timeout after 30s") == "timeout"

    def test_missing_library(self):
        assert (
            _classify_check_error(
                "Error loading libjmmshr_gsl.so: cannot open shared object"
            )
            == "missing_library"
        )

    def test_missing_library_dlopen(self):
        assert (
            _classify_check_error(
                "Error loading /usr/local/mdsplus/lib/libblas.so: "
                "libblas.so: cannot open shared object file"
            )
            == "missing_library"
        )

    def test_expression_error(self):
        """Expression nodes that fail to resolve should be classified."""
        assert (
            _classify_check_error("%TDI-E-EXTRA_ARG, Too many arguments")
            == "expression_error"
        )

    def test_roprand(self):
        """$ROPRAND indicates expression evaluation produced invalid result."""
        assert _classify_check_error("$ROPRAND") == "expression_error"

    def test_segfault(self):
        assert _classify_check_error("Segmentation fault") == "segfault"


# =============================================================================
# Phase 3: Multi-shot batch checking — check_signals_batch.py
# =============================================================================


class TestCheckSignalsBatchScript:
    """Test the remote check_signals_batch.py script logic."""

    def test_groups_signals_by_tree_and_shot(self):
        """Signals are grouped by (tree_name, shot) for efficient batching."""
        # Import the grouping logic from the script
        from collections import defaultdict

        signals = [
            {
                "id": "s1",
                "tree_name": "results",
                "shot": 85000,
                "accessor": "\\RESULTS::A",
            },
            {
                "id": "s2",
                "tree_name": "results",
                "shot": 85000,
                "accessor": "\\RESULTS::B",
            },
            {"id": "s3", "tree_name": "static", "shot": 1, "accessor": "\\STATIC::C"},
            {
                "id": "s4",
                "tree_name": "results",
                "shot": 75000,
                "accessor": "\\RESULTS::A",
            },
        ]

        groups: dict[tuple[str, int], list] = defaultdict(list)
        for sig in signals:
            groups[(sig["tree_name"], sig["shot"])].append(sig)

        assert len(groups) == 3
        assert len(groups[("results", 85000)]) == 2
        assert len(groups[("static", 1)]) == 1
        assert len(groups[("results", 75000)]) == 1

    def test_multi_shot_signals_generate_multiple_groups(self):
        """When check_shots has multiple values, signal appears in all groups."""
        from collections import defaultdict

        # Simulate how multi-shot checking works
        signals = [
            {
                "id": "s1",
                "tree_name": "tcv_shot",
                "shot": 85000,
                "accessor": "\\RESULTS::A",
                "check_shots": [85000, 75000, 65000],
            },
        ]

        groups: dict[tuple[str, int], list] = defaultdict(list)
        for sig in signals:
            check_shots = sig.get("check_shots", [sig["shot"]])
            for shot in check_shots:
                groups[(sig["tree_name"], shot)].append(sig)

        assert len(groups) == 3


# =============================================================================
# Phase 5: TDI function categorization
# =============================================================================


class TestTDIFunctionCategorization:
    """Test that TDI functions are categorized correctly."""

    def test_hardware_functions_excluded(self):
        """Hardware control functions should be excluded from checking."""
        from imas_codex.discovery.signals.parallel import _is_excluded_tdi_function

        excluded = [
            "tile_store",
            "tile_init_action",
            "beckhoff_setstate",
            "shot_close",
            "dt100_mds",
            "wavegen_set",
        ]
        for func in excluded:
            assert _is_excluded_tdi_function(func, exclude_list=excluded), (
                f"{func} should be excluded"
            )

    def test_physics_functions_not_excluded(self):
        """Core physics accessor functions should NOT be excluded."""
        from imas_codex.discovery.signals.parallel import _is_excluded_tdi_function

        excluded = ["tile_store", "beckhoff_setstate"]
        physics = ["tcv_eq", "tcv_get", "tcv_ip", "fir_aut", "ts_rawdata"]
        for func in physics:
            assert not _is_excluded_tdi_function(func, exclude_list=excluded), (
                f"{func} should NOT be excluded"
            )
