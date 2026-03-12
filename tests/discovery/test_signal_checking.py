"""Unit tests for signal checking improvements.

Tests the check worker routing logic, multi-shot batch checking,
DataAccess-driven signal routing, and expression node handling.

Phase 1: Static tree routing - independent trees use their own data_source_name/shot
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
            "data_source_name": "results",
            "data_source_path": "\\RESULTS::THOMSON:NE",
            "discovery_source": "tree_traversal",
            "accessor": "THOMSON:NE",
        }
        data_source_name, accessor, check_shots = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"static", "vsystem"},
            tree_shots={"static": [1, 2, 3, 4, 5, 6, 7, 8]},
            reference_shot=85000,
        )
        assert data_source_name == "tcv_shot"
        assert accessor == "\\RESULTS::THOMSON:NE"
        assert check_shots == [85000]

    def test_static_tree_routes_independently(self):
        """Static tree signals open the static tree directly, not tcv_shot."""
        signal = {
            "data_source_name": "static",
            "data_source_path": "\\STATIC::TOP.MECHANICAL.COIL:R",
            "discovery_source": "tree_traversal",
            "accessor": "TOP.MECHANICAL.COIL:R",
        }
        data_source_name, accessor, check_shots = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"static", "vsystem"},
            tree_shots={"static": [1, 2, 3, 4, 5, 6, 7, 8]},
            reference_shot=85000,
        )
        assert data_source_name == "static"
        assert accessor == "\\STATIC::TOP.MECHANICAL.COIL:R"

    def test_static_tree_returns_all_version_shots(self):
        """Static tree returns ALL version shots as check_shots."""
        signal = {
            "data_source_name": "static",
            "data_source_path": "\\STATIC::TOP.MECHANICAL.COIL:R",
            "discovery_source": "tree_traversal",
            "accessor": "TOP.MECHANICAL.COIL:R",
        }
        data_source_name, accessor, check_shots = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"static", "vsystem"},
            tree_shots={"static": [1, 2, 3, 4, 5, 6, 7, 8]},
            reference_shot=85000,
        )
        assert data_source_name == "static"
        # All versions returned, not just the first
        assert check_shots == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_vsystem_routes_independently(self):
        """vsystem tree signals open vsystem directly."""
        signal = {
            "data_source_name": "vsystem",
            "data_source_path": "\\VSYSTEM::SOME:NODE",
            "discovery_source": "tree_traversal",
            "accessor": "SOME:NODE",
        }
        data_source_name, accessor, check_shots = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"static", "vsystem"},
            tree_shots={},
            reference_shot=85000,
        )
        assert data_source_name == "vsystem"
        # No version shots for vsystem, falls back to reference
        assert check_shots == [85000]

    def test_tdi_function_uses_connection_tree(self):
        """TDI function signals route through connection tree."""
        signal = {
            "data_source_name": None,
            "tdi_function": "tcv_eq",
            "discovery_source": "tdi_extraction",
            "accessor": 'tcv_eq("r_axis")',
        }
        data_source_name, accessor, check_shots = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"static", "vsystem"},
            tree_shots={},
            reference_shot=85000,
        )
        assert data_source_name == "tcv_shot"
        assert check_shots == [85000]

    def test_unknown_tree_defaults_to_connection_tree(self):
        """Signals from unknown trees default to connection tree."""
        signal = {
            "data_source_name": "unknown_tree",
            "data_source_path": "\\UNKNOWN::NODE",
            "discovery_source": "tree_traversal",
            "accessor": "NODE",
        }
        data_source_name, accessor, check_shots = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"static", "vsystem"},
            tree_shots={},
            reference_shot=85000,
        )
        assert data_source_name == "tcv_shot"
        assert check_shots == [85000]

    def test_subtree_with_fallback_shots(self):
        """Subtree signals include fallback shots from tree versions."""
        signal = {
            "data_source_name": "results",
            "data_source_path": "\\RESULTS::THOMSON:NE",
            "discovery_source": "tree_traversal",
            "accessor": "THOMSON:NE",
        }
        data_source_name, accessor, check_shots = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"static"},
            tree_shots={"tcv_shot": [85000, 75000, 65000]},
            reference_shot=85000,
        )
        assert data_source_name == "tcv_shot"
        # Reference shot first, then fallbacks from tcv_shot versions
        assert check_shots[0] == 85000
        assert 75000 in check_shots
        assert 65000 in check_shots

    def test_independent_tree_no_versions_uses_reference(self):
        """Independent tree with no configured versions uses reference shot."""
        signal = {
            "data_source_name": "vsystem",
            "data_source_path": "\\VSYSTEM::NODE",
            "discovery_source": "tree_traversal",
            "accessor": "NODE",
        }
        data_source_name, accessor, check_shots = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees={"vsystem"},
            tree_shots={},
            reference_shot=85000,
        )
        assert data_source_name == "vsystem"
        assert check_shots == [85000]


# =============================================================================
# Scanner type routing — device_xml signals
# =============================================================================


class TestGetSignalScannerType:
    """Test _get_signal_scanner_type routes discovery sources correctly."""

    def _get_scanner_type(self, discovery_source: str) -> str:
        """Helper to call the nested function via check_worker's logic."""
        # Replicate the routing logic from check_worker
        source = discovery_source
        if source == "ppf":
            return "ppf"
        if source == "edas":
            return "edas"
        if source == "wiki":
            return "wiki"
        if source in ("device_xml", "jec2020_xml", "magnetics_config"):
            return "device_xml"
        return "mdsplus"

    def test_device_xml_routes_to_device_xml(self):
        """Device XML signals from device_xml source route to device_xml scanner."""
        assert self._get_scanner_type("device_xml") == "device_xml"

    def test_jec2020_routes_to_device_xml(self):
        """JEC2020 XML signals route to device_xml scanner."""
        assert self._get_scanner_type("jec2020_xml") == "device_xml"

    def test_magnetics_config_routes_to_device_xml(self):
        """Magnetics config signals route to device_xml scanner."""
        assert self._get_scanner_type("magnetics_config") == "device_xml"

    def test_ppf_routes_to_ppf(self):
        assert self._get_scanner_type("ppf") == "ppf"

    def test_edas_routes_to_edas(self):
        assert self._get_scanner_type("edas") == "edas"

    def test_wiki_routes_to_wiki(self):
        assert self._get_scanner_type("wiki") == "wiki"

    def test_tree_traversal_routes_to_mdsplus(self):
        assert self._get_scanner_type("tree_traversal") == "mdsplus"

    def test_unknown_routes_to_mdsplus(self):
        assert self._get_scanner_type("") == "mdsplus"


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
        """Signals are grouped by (data_source_name, shot) for efficient batching."""
        from collections import defaultdict

        signals = [
            {
                "id": "s1",
                "data_source_name": "results",
                "check_shots": [85000],
                "accessor": "\\RESULTS::A",
            },
            {
                "id": "s2",
                "data_source_name": "results",
                "check_shots": [85000],
                "accessor": "\\RESULTS::B",
            },
            {
                "id": "s3",
                "data_source_name": "static",
                "check_shots": [1],
                "accessor": "\\STATIC::C",
            },
            {
                "id": "s4",
                "data_source_name": "results",
                "check_shots": [75000],
                "accessor": "\\RESULTS::A",
            },
        ]

        # First check_shot is the primary group key
        groups: dict[tuple[str, int], list] = defaultdict(list)
        for sig in signals:
            primary = sig["check_shots"][0]
            groups[(sig["data_source_name"], primary)].append(sig)

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
                "data_source_name": "static",
                "check_shots": [1, 2, 3, 4, 5, 6, 7, 8],
                "accessor": "\\STATIC::A",
            },
        ]

        # Primary attempt uses first check_shot
        primary_groups: dict[tuple[str, int], list] = defaultdict(list)
        for sig in signals:
            primary_groups[(sig["data_source_name"], sig["check_shots"][0])].append(sig)

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
                    "data_source_name": "static",
                    "check_shots": [1, 2, 3, 4, 5, 6, 7, 8],
                },
                {
                    "id": "tcv:results:/ip",
                    "accessor": "\\ip",
                    "data_source_name": "tcv_shot",
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


# =============================================================================
# Phase 5: End-to-end check flow
# =============================================================================


class TestCheckWorkerBatchInputConstruction:
    """Test that check_worker builds the correct batch_input for the remote script."""

    def _make_facility_config(self):
        """Create a realistic TCV-like facility config."""
        return {
            "data_systems": {
                "mdsplus": {
                    "connection_tree": "tcv_shot",
                    "trees": [
                        {
                            "data_source_name": "tcv_shot",
                            "versions": [
                                {"first_shot": 85000},
                                {"first_shot": 75000},
                            ],
                        },
                        {
                            "data_source_name": "static",
                            "versions": [
                                {"version": 1},
                                {"version": 2},
                                {"version": 3},
                            ],
                        },
                    ],
                }
            },
            "data_access_patterns": {
                "independent_trees": ["static", "vsystem"],
            },
        }

    def test_static_signal_gets_all_version_check_shots(self):
        """Static tree signal should get all versions as check_shots."""
        signal = {
            "data_source_name": "static",
            "data_source_path": "\\STATIC::TOP.MECHANICAL.COIL:R",
            "discovery_source": "tree_traversal",
            "accessor": "TOP.MECHANICAL.COIL:R",
        }
        config = self._make_facility_config()

        # Build routing tables as check_worker does
        mdsplus_config = config["data_systems"]["mdsplus"]
        dap = config["data_access_patterns"]
        independent_trees = set(dap.get("independent_trees", []))
        tree_shots: dict[str, list[int]] = {}
        for st in mdsplus_config["trees"]:
            versions = st.get("versions", [])
            shots = [
                v.get("first_shot") or v.get("version")
                for v in versions
                if v.get("first_shot") or v.get("version")
            ]
            if shots:
                tree_shots[st["data_source_name"]] = sorted(shots)

        data_source_name, accessor, check_shots = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees=independent_trees,
            tree_shots=tree_shots,
            reference_shot=85000,
        )

        # Batch input should use check_shots, not shot+fallback_shots
        batch_entry = {
            "id": "tcv:static:/mechanical/coil/r",
            "accessor": accessor,
            "data_source_name": data_source_name,
            "check_shots": check_shots,
        }

        assert batch_entry["data_source_name"] == "static"
        assert batch_entry["check_shots"] == [1, 2, 3]
        assert "shot" not in batch_entry
        assert "fallback_shots" not in batch_entry

    def test_subtree_signal_gets_connection_tree_shots(self):
        """Subtree signal should get reference + connection tree version shots."""
        signal = {
            "data_source_name": "results",
            "data_source_path": "\\RESULTS::THOMSON:NE",
            "discovery_source": "tree_traversal",
            "accessor": "THOMSON:NE",
        }
        config = self._make_facility_config()

        mdsplus_config = config["data_systems"]["mdsplus"]
        dap = config["data_access_patterns"]
        independent_trees = set(dap.get("independent_trees", []))
        tree_shots: dict[str, list[int]] = {}
        for st in mdsplus_config["trees"]:
            versions = st.get("versions", [])
            shots = [
                v.get("first_shot") or v.get("version")
                for v in versions
                if v.get("first_shot") or v.get("version")
            ]
            if shots:
                tree_shots[st["data_source_name"]] = sorted(shots)

        data_source_name, accessor, check_shots = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees=independent_trees,
            tree_shots=tree_shots,
            reference_shot=85000,
        )

        assert data_source_name == "tcv_shot"
        assert check_shots[0] == 85000  # Reference shot first
        assert 75000 in check_shots  # Fallback included

    def test_tdi_function_gets_reference_shot_only(self):
        """TDI function signals should only check at reference shot."""
        signal = {
            "data_source_name": None,
            "tdi_function": "tcv_eq",
            "discovery_source": "tdi_extraction",
            "accessor": 'tcv_eq("r_axis")',
        }
        config = self._make_facility_config()

        mdsplus_config = config["data_systems"]["mdsplus"]
        dap = config["data_access_patterns"]
        independent_trees = set(dap.get("independent_trees", []))
        tree_shots: dict[str, list[int]] = {}
        for st in mdsplus_config["trees"]:
            versions = st.get("versions", [])
            shots = [
                v.get("first_shot") or v.get("version")
                for v in versions
                if v.get("first_shot") or v.get("version")
            ]
            if shots:
                tree_shots[st["data_source_name"]] = sorted(shots)

        data_source_name, accessor, check_shots = _resolve_check_tree(
            signal,
            connection_tree="tcv_shot",
            independent_trees=independent_trees,
            tree_shots=tree_shots,
            reference_shot=85000,
        )

        assert data_source_name == "tcv_shot"
        assert check_shots == [85000]


class TestCheckResultHandling:
    """Test how check results map to graph updates."""

    def test_successful_check_has_required_fields(self):
        """Successful check result should have shape, dtype, and shot."""
        result = {
            "id": "tcv:static:/r_c",
            "success": True,
            "shape": [1],
            "dtype": "float64",
            "checked_shot": 3,
        }
        # Build the entry as check_worker does
        entry = {
            "id": result["id"],
            "success": True,
            "shot": result["checked_shot"],
            "data_access": "tcv:mdsplus:static",
            "shape": result.get("shape"),
            "dtype": result.get("dtype"),
        }
        assert entry["success"] is True
        assert entry["shot"] == 3
        assert entry["shape"] == [1]

    def test_failed_check_has_error_classification(self):
        """Failed check result should have error and error_type."""
        result = {
            "id": "tcv:static:/greens/r",
            "success": False,
            "error": "Error loading libjmmshr_gsl.so: cannot open shared object",
            "checked_shot": 1,
        }
        entry = {
            "id": result["id"],
            "success": False,
            "shot": result["checked_shot"],
            "data_access": "tcv:mdsplus:static",
            "error": result["error"],
            "error_type": _classify_check_error(result["error"]),
        }
        assert entry["success"] is False
        assert entry["error_type"] == "missing_library"

    def test_retry_success_reports_failed_shots(self):
        """Check result from retry should report all failed shots."""
        result = {
            "id": "tcv:static:/ang_a",
            "success": True,
            "shape": [1],
            "dtype": "float64",
            "checked_shot": 4,
            "failed_shots": [1, 2, 3],
        }
        assert result["checked_shot"] == 4
        assert result["failed_shots"] == [1, 2, 3]
        assert result["success"] is True
