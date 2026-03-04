"""Unit tests for signal checking improvements.

Tests the check worker routing logic, multi-shot batch checking,
DataAccess-driven signal routing, and expression node handling.

Phase 1: Static tree routing - independent trees use their own tree_name/shot
Phase 2: Multi-version batch checking as primary (not fallback)
Phase 3: Expression node classification
Phase 4: TDI function categorization
Phase 5: Missing library error detection
"""

from __future__ import annotations

import json
import subprocess
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
    """Test _resolve_check_tree routes signals to the correct tree/check_shots."""

    def test_subtree_signal_routes_to_connection_tree(self):
        """Subtree signals (results, magnetics) route to connection tree."""
        signal = {
            "tree_name": "results",
            "node_path": "\\RESULTS::THOMSON:NE",
            "discovery_source": "tree_traversal",
            "accessor": "THOMSON:NE",
        }
        tree_name, accessor, check_shots = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"static", "vsystem"},
            tree_shots={"static": [1, 2, 3, 4, 5, 6, 7, 8]},
            reference_shot=85000,
        )
        assert tree_name == "tcv_shot"
        assert accessor == "\\RESULTS::THOMSON:NE"
        assert check_shots == [85000]

    def test_static_tree_routes_independently(self):
        """Static tree signals open the static tree directly, not tcv_shot."""
        signal = {
            "tree_name": "static",
            "node_path": "\\STATIC::TOP.MECHANICAL.COIL:R",
            "discovery_source": "tree_traversal",
            "accessor": "TOP.MECHANICAL.COIL:R",
        }
        tree_name, accessor, check_shots = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"static", "vsystem"},
            tree_shots={"static": [1, 2, 3, 4, 5, 6, 7, 8]},
            reference_shot=85000,
        )
        assert tree_name == "static"
        assert accessor == "\\STATIC::TOP.MECHANICAL.COIL:R"

    def test_static_tree_returns_all_version_shots(self):
        """Static tree returns ALL version shots as check_shots."""
        signal = {
            "tree_name": "static",
            "node_path": "\\STATIC::TOP.MECHANICAL.COIL:R",
            "discovery_source": "tree_traversal",
            "accessor": "TOP.MECHANICAL.COIL:R",
        }
        tree_name, accessor, check_shots = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"static", "vsystem"},
            tree_shots={"static": [1, 2, 3, 4, 5, 6, 7, 8]},
            reference_shot=85000,
        )
        assert tree_name == "static"
        # All versions returned, not just the first
        assert check_shots == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_vsystem_routes_independently(self):
        """vsystem tree signals open vsystem directly."""
        signal = {
            "tree_name": "vsystem",
            "node_path": "\\VSYSTEM::SOME:NODE",
            "discovery_source": "tree_traversal",
            "accessor": "SOME:NODE",
        }
        tree_name, accessor, check_shots = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"static", "vsystem"},
            tree_shots={},
            reference_shot=85000,
        )
        assert tree_name == "vsystem"
        # No version shots for vsystem, falls back to reference
        assert check_shots == [85000]

    def test_tdi_function_uses_connection_tree(self):
        """TDI function signals route through connection tree."""
        signal = {
            "tree_name": None,
            "tdi_function": "tcv_eq",
            "discovery_source": "tdi_extraction",
            "accessor": 'tcv_eq("r_axis")',
        }
        tree_name, accessor, check_shots = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"static", "vsystem"},
            tree_shots={},
            reference_shot=85000,
        )
        assert tree_name == "tcv_shot"
        assert check_shots == [85000]

    def test_unknown_tree_defaults_to_connection_tree(self):
        """Signals from unknown trees default to connection tree."""
        signal = {
            "tree_name": "unknown_tree",
            "node_path": "\\UNKNOWN::NODE",
            "discovery_source": "tree_traversal",
            "accessor": "NODE",
        }
        tree_name, accessor, check_shots = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"static", "vsystem"},
            tree_shots={},
            reference_shot=85000,
        )
        assert tree_name == "tcv_shot"
        assert check_shots == [85000]

    def test_subtree_with_fallback_shots(self):
        """Subtree signals include fallback shots from tree versions."""
        signal = {
            "tree_name": "results",
            "node_path": "\\RESULTS::THOMSON:NE",
            "discovery_source": "tree_traversal",
            "accessor": "THOMSON:NE",
        }
        tree_name, accessor, check_shots = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"static"},
            tree_shots={"tcv_shot": [85000, 75000, 65000]},
            reference_shot=85000,
        )
        assert tree_name == "tcv_shot"
        # Reference shot first, then fallbacks from tcv_shot versions
        assert check_shots[0] == 85000
        assert 75000 in check_shots
        assert 65000 in check_shots

    def test_independent_tree_no_versions_uses_reference(self):
        """Independent tree with no configured versions uses reference shot."""
        signal = {
            "tree_name": "vsystem",
            "node_path": "\\VSYSTEM::NODE",
            "discovery_source": "tree_traversal",
            "accessor": "NODE",
        }
        tree_name, accessor, check_shots = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"vsystem"},
            tree_shots={},
            reference_shot=85000,
        )
        assert tree_name == "vsystem"
        assert check_shots == [85000]


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
# Phase 2: Multi-shot batch checking — check_signals_batch.py
# =============================================================================


class TestCheckSignalsBatchScript:
    """Test the remote check_signals_batch.py script logic."""

    def test_groups_signals_by_tree_and_shot(self):
        """Signals are grouped by (tree_name, shot) for efficient batching."""
        from collections import defaultdict

        signals = [
            {
                "id": "s1",
                "tree_name": "results",
                "check_shots": [85000],
                "accessor": "\\RESULTS::A",
            },
            {
                "id": "s2",
                "tree_name": "results",
                "check_shots": [85000],
                "accessor": "\\RESULTS::B",
            },
            {
                "id": "s3",
                "tree_name": "static",
                "check_shots": [1],
                "accessor": "\\STATIC::C",
            },
            {
                "id": "s4",
                "tree_name": "results",
                "check_shots": [75000],
                "accessor": "\\RESULTS::A",
            },
        ]

        # First check_shot is the primary group key
        groups: dict[tuple[str, int], list] = defaultdict(list)
        for sig in signals:
            primary = sig["check_shots"][0]
            groups[(sig["tree_name"], primary)].append(sig)

        assert len(groups) == 3
        assert len(groups[("results", 85000)]) == 2
        assert len(groups[("static", 1)]) == 1
        assert len(groups[("results", 75000)]) == 1

    def test_check_shots_drives_multi_version_checking(self):
        """check_shots list drives systematic multi-version checking."""
        from collections import defaultdict

        signals = [
            {
                "id": "s1",
                "tree_name": "static",
                "check_shots": [1, 2, 3, 4, 5, 6, 7, 8],
                "accessor": "\\STATIC::A",
            },
        ]

        # Primary attempt uses first check_shot
        primary_groups: dict[tuple[str, int], list] = defaultdict(list)
        for sig in signals:
            primary_groups[(sig["tree_name"], sig["check_shots"][0])].append(sig)

        assert len(primary_groups) == 1
        assert ("static", 1) in primary_groups

        # Remaining check_shots are for retry
        sig = signals[0]
        retry_shots = sig["check_shots"][1:]
        assert retry_shots == [2, 3, 4, 5, 6, 7, 8]


class TestCheckSignalsBatchInputFormat:
    """Test the input/output format of the batch check script."""

    def test_check_shots_replaces_shot_and_fallback(self):
        """check_shots is the single field for all shots to try."""
        # New format — check_shots is primary
        batch_input = {
            "signals": [
                {
                    "id": "tcv:static:/r_c",
                    "accessor": "\\STATIC::R_C",
                    "tree_name": "static",
                    "check_shots": [1, 2, 3, 4, 5, 6, 7, 8],
                },
                {
                    "id": "tcv:results:/ip",
                    "accessor": "\\ip",
                    "tree_name": "tcv_shot",
                    "check_shots": [85000],
                },
            ],
            "timeout_per_group": 30,
        }
        # Verify structure
        for sig in batch_input["signals"]:
            assert "check_shots" in sig
            assert isinstance(sig["check_shots"], list)
            assert len(sig["check_shots"]) >= 1
            # Old fields should not be present
            assert "shot" not in sig
            assert "fallback_shots" not in sig

    def test_backward_compat_shot_field(self):
        """Script should handle legacy shot+fallback_shots format."""
        from imas_codex.remote.scripts.check_signals_batch import (
            _normalize_signal_shots,
        )

        # Legacy format
        sig = {"id": "s1", "shot": 85000, "fallback_shots": [75000, 65000]}
        check_shots = _normalize_signal_shots(sig)
        assert check_shots == [85000, 75000, 65000]

        # New format
        sig2 = {"id": "s2", "check_shots": [1, 2, 3]}
        check_shots2 = _normalize_signal_shots(sig2)
        assert check_shots2 == [1, 2, 3]

        # Minimal format — just shot, no fallbacks
        sig3 = {"id": "s3", "shot": 85000}
        check_shots3 = _normalize_signal_shots(sig3)
        assert check_shots3 == [85000]


class TestBatchRetryLogic:
    """Test the multi-shot retry logic in check_signals_batch.py."""

    def test_failed_shots_tracks_all_failures(self):
        """When a signal fails on shots 1,2,3 and succeeds on 4,
        failed_shots should list all failed shots."""
        from imas_codex.remote.scripts.check_signals_batch import (
            _is_shot_dependent_error,
        )

        # Simulate the retry tracking
        failed_shots_tracker: dict[str, list[int]] = {}
        sig_id = "tcv:static:/r_c"

        # Shot 1 fails with NNF (shot-dependent)
        error1 = "%TREE-W-NNF, Node Not Found"
        assert _is_shot_dependent_error(error1)
        failed_shots_tracker.setdefault(sig_id, []).append(1)

        # Shot 2 also fails
        failed_shots_tracker[sig_id].append(2)

        # Shot 3 also fails
        failed_shots_tracker[sig_id].append(3)

        # Shot 4 succeeds — final result should have all failed shots
        assert failed_shots_tracker[sig_id] == [1, 2, 3]

    def test_structural_error_stops_retries(self):
        """Structural errors (SYNTAX, MISS_ARG) should not trigger retries."""
        from imas_codex.remote.scripts.check_signals_batch import (
            _is_shot_dependent_error,
        )

        assert not _is_shot_dependent_error("%TDI-E-SYNTAX, Unexpected token")
        assert not _is_shot_dependent_error("MISS_ARG: Function requires 2 arguments")
        assert not _is_shot_dependent_error("INVCLADSC: Invalid class descriptor")

    def test_shot_dependent_errors_trigger_retries(self):
        """Shot-dependent errors should trigger retry on next check_shot."""
        from imas_codex.remote.scripts.check_signals_batch import (
            _is_shot_dependent_error,
        )

        assert _is_shot_dependent_error("%TREE-W-NNF, Node Not Found")
        assert _is_shot_dependent_error("TreeNODATA: No data for this segment")
        assert _is_shot_dependent_error("KEYNOTFOU: Key not found")
        assert _is_shot_dependent_error("TreeNOT_OPEN: Tree not available")

    def test_normalize_preserves_shot_order(self):
        """check_shots order must be preserved — first shot tried first."""
        from imas_codex.remote.scripts.check_signals_batch import (
            _normalize_signal_shots,
        )

        sig = {"id": "s1", "check_shots": [8, 7, 6, 5, 4, 3, 2, 1]}
        shots = _normalize_signal_shots(sig)
        assert shots == [8, 7, 6, 5, 4, 3, 2, 1]


# =============================================================================
# Phase 3: Expression node error handling
# =============================================================================


class TestExpressionNodeErrors:
    """Test that expression node errors are classified correctly for retry."""

    def test_roprand_is_shot_dependent(self):
        """$ROPRAND may resolve at different versions — should be retried."""
        from imas_codex.remote.scripts.check_signals_batch import (
            _is_shot_dependent_error,
        )

        assert _is_shot_dependent_error("$ROPRAND")

    def test_extra_arg_is_structural(self):
        """EXTRA_ARG is a function signature error — no retry."""
        from imas_codex.remote.scripts.check_signals_batch import (
            _is_shot_dependent_error,
        )

        assert not _is_shot_dependent_error("%TDI-E-EXTRA_ARG, Too many arguments")

    def test_missing_library_is_structural(self):
        """Missing shared library errors should never be retried."""
        from imas_codex.remote.scripts.check_signals_batch import (
            _is_shot_dependent_error,
        )

        assert not _is_shot_dependent_error(
            "Error loading libjmmshr_gsl.so: cannot open shared object"
        )

    def test_missing_library_is_classified_by_parallel(self):
        """parallel.py classifies missing library consistently."""
        assert (
            _classify_check_error(
                "Error loading libjmmshr_gsl.so: cannot open shared object"
            )
            == "missing_library"
        )

    def test_roprand_classified_as_expression_error(self):
        """parallel.py classifies $ROPRAND as expression error."""
        assert _classify_check_error("$ROPRAND") == "expression_error"

    def test_extra_arg_classified_as_expression_error(self):
        """parallel.py classifies EXTRA_ARG as expression error."""
        assert (
            _classify_check_error("%TDI-E-EXTRA_ARG, Too many arguments")
            == "expression_error"
        )


# =============================================================================
# Phase 4: TDI function categorization
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
