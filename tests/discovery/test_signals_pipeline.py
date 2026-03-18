"""Unit tests for the redesigned signals discovery pipeline.

Tests the async worker pipeline (seed → epoch → extract → units →
promote → enrich → check → embed) with mocked infrastructure.

Verifies:
- Static trees: versions seeded from config, extracted, promoted
- Dynamic trees: epochs detected independently, seeded, extracted, promoted
- All workers start simultaneously and coordinate via graph state machine
- Pipeline phases complete correctly and workers exit cleanly
- 50+ simulated signal paths flow through the full pipeline
- CLI integration: help, scanner validation, scan-only routing

No live infrastructure needed — all graph/SSH/LLM calls are mocked.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.discovery.signals.parallel import (
    DataDiscoveryState,
    check_worker,
    run_parallel_data_discovery,
)

# ── Fixtures ─────────────────────────────────────────────────────────────

FACILITY = "test_facility"
SSH_HOST = "test_host"

# Static tree: 3 versions defined in config
STATIC_TREE = {
    "source_name": "magnetics",
    "versions": [
        {"version": 1, "first_shot": 1000, "description": "v1"},
        {"version": 2, "first_shot": 2000, "description": "v2"},
        {"version": 3, "first_shot": 3000, "description": "v3"},
    ],
}

# Dynamic tree: no versions in config — epochs detected at runtime
DYNAMIC_TREE = {
    "source_name": "results",
}

FACILITY_CONFIG = {
    "ssh_host": SSH_HOST,
    "data_systems": {
        "mdsplus": {
            "setup_commands": ["module load mdsplus"],
            "trees": [STATIC_TREE, DYNAMIC_TREE],
            "connection_tree": "magnetics",
        },
    },
}

# 25 leaf nodes per tree version simulating signal paths
LEAF_NODES_PER_VERSION = 25


def _make_data_nodes(data_source_name: str, version: int) -> dict:
    """Build a mock extraction result with realistic SignalNode structure."""
    nodes = {}
    for i in range(LEAF_NODES_PER_VERSION):
        path = f"\\{data_source_name.upper()}::TOP.NODE_{i:03d}"
        nodes[path] = {
            "path": path,
            "node_type": "SIGNAL" if i % 3 else "NUMERIC",
            "usage": "signal" if i % 3 else "numeric",
        }
    return {
        "data_source_name": data_source_name,
        "versions": {
            str(version): {
                "node_count": len(nodes),
                "nodes": nodes,
            }
        },
    }


# Track which versions were claimed/extracted
_extracted_versions: dict[str, set[int]] = {}
_promoted_trees: set[str] = set()
_units_extracted_trees: set[str] = set()
_call_counts: dict[str, int] = {}


@pytest.fixture(autouse=True)
def _reset_tracking():
    """Reset tracking state between tests."""
    _extracted_versions.clear()
    _promoted_trees.clear()
    _units_extracted_trees.clear()
    _call_counts.clear()


def _mock_claim_version(facility: str) -> dict | None:
    """Simulate claiming a version for extraction.

    Tracks internally which versions have been consumed.
    Returns static tree versions first, then dynamic epoch versions.
    """
    all_versions = [
        ("magnetics", 1),
        ("magnetics", 2),
        ("magnetics", 3),
        ("results", 5000),
        ("results", 8000),
    ]
    claimed = _extracted_versions.setdefault(facility, set())
    for tree, ver in all_versions:
        if (tree, ver) not in claimed:
            claimed.add((tree, ver))
            return {
                "id": f"{facility}:{tree}:v{ver}",
                "version": ver,
                "data_source_name": tree,
                "first_shot": ver,
            }
    return None


def _mock_claim_tree_for_units(facility: str) -> dict | None:
    """Simulate claiming a tree for unit extraction."""
    for data_source_name in ["magnetics", "results"]:
        if data_source_name not in _units_extracted_trees:
            _units_extracted_trees.add(data_source_name)
            all_versions = _extracted_versions.get(facility, set())
            latest = max(
                (v for t, v in all_versions if t == data_source_name),
                default=1,
            )
            return {"data_source_name": data_source_name, "latest_version": latest}
    return None


def _mock_claim_tree_for_promote(facility: str) -> str | None:
    """Simulate claiming a tree for promotion."""
    for data_source_name in ["magnetics", "results"]:
        if data_source_name not in _promoted_trees:
            _promoted_trees.add(data_source_name)
            return data_source_name
    return None


# ── DataDiscoveryState Tests ─────────────────────────────────────────────


class TestDataDiscoveryState:
    """Tests for the redesigned DataDiscoveryState with sub-phases."""

    def _make_state(self, **kwargs) -> DataDiscoveryState:
        defaults = {
            "facility": FACILITY,
            "ssh_host": SSH_HOST,
            "scanner_types": ["mdsplus"],
            "cost_limit": 10.0,
        }
        defaults.update(kwargs)
        return DataDiscoveryState(**defaults)

    def test_initial_state(self):
        """All sub-phases start as not done."""
        state = self._make_state()
        assert not state.seed_phase.done
        assert not state.epoch_phase.done
        assert not state.extract_phase.done
        assert not state.units_phase.done
        assert not state.promote_phase.done
        assert not state.scan_phase.done

    def test_scan_phase_composite(self):
        """scan_phase is done only when ALL sub-phases are done."""
        state = self._make_state()
        state.seed_phase.mark_done()
        state.epoch_phase.mark_done()
        state.extract_phase.mark_done()
        state.units_phase.mark_done()
        state._update_scan_phase()
        assert not state.scan_phase.done  # promote not done

        state.promote_phase.mark_done()
        state._update_scan_phase()
        assert state.scan_phase.done

    def test_should_stop_when_requested(self):
        """stop_requested immediately stops everything."""
        state = self._make_state()
        state.stop_requested = True
        assert state.should_stop()
        assert state.should_stop_discovering()
        assert state.should_stop_enriching()
        assert state.should_stop_checking()

    def test_budget_exhausted(self):
        """Budget exhaustion stops enrichment."""
        state = self._make_state(cost_limit=5.0)
        state.enrich_stats.cost = 5.5
        assert state.budget_exhausted
        assert state.should_stop_enriching()

    def test_signal_limit_reached(self):
        """Signal limit stops enrichment."""
        state = self._make_state(signal_limit=50)
        state.enrich_stats.processed = 50
        assert state.signal_limit_reached
        assert state.should_stop_enriching()

    def test_discover_only_stops_after_scan(self):
        """In discover_only mode, should_stop_discovering checks all sub-phases."""
        state = self._make_state()
        assert not state.should_stop_discovering()

        state.seed_phase.mark_done()
        state.epoch_phase.mark_done()
        state.extract_phase.mark_done()
        state.units_phase.mark_done()
        state.promote_phase.mark_done()
        assert state.should_stop_discovering()

    def test_enrich_only_preset(self):
        """enrich_only flag can be tested in state."""
        state = self._make_state(enrich_only=True)
        assert state.enrich_only

    def test_deadline_expired(self):
        """Deadline expiration stops workers."""
        import time

        state = self._make_state()
        state.deadline = time.time() - 1  # Already expired
        assert state.deadline_expired
        assert state.should_stop()
        assert state.should_stop_enriching()
        assert state.should_stop_checking()

    def test_separate_stats_per_worker_group(self):
        """Each worker group has independent stats."""
        state = self._make_state()
        state.discover_stats.processed = 10
        state.extract_stats.processed = 5
        state.units_stats.processed = 2
        state.promote_stats.processed = 50
        state.enrich_stats.processed = 30
        state.check_stats.processed = 20

        assert state.discover_stats.processed == 10
        assert state.extract_stats.processed == 5
        assert state.promote_stats.processed == 50
        assert state.enrich_stats.processed == 30


# ── Worker Unit Tests ────────────────────────────────────────────────────


class TestSeedWorker:
    """Tests for the seed_worker function."""

    @pytest.mark.anyio
    async def test_seed_static_versions(self):
        """seed_worker creates SignalEpochs from config."""
        from imas_codex.discovery.signals.parallel import seed_worker

        state = DataDiscoveryState(
            facility=FACILITY,
            ssh_host=SSH_HOST,
            scanner_types=["mdsplus"],
            facility_config=FACILITY_CONFIG,
            initial_version_counts={
                "total": 0,
                "discovered": 0,
                "ingested": 0,
                "failed": 0,
            },
            initial_signal_counts={
                "total": 0,
                "discovered": 0,
                "enriched": 0,
                "checked": 0,
            },
            cost_limit=10.0,
        )

        progress_calls = []

        def on_progress(msg, stats, results=None):
            progress_calls.append(msg)

        with (
            patch(
                "imas_codex.discovery.mdsplus.graph_ops.seed_versions",
                return_value=3,
            ) as mock_seed,
            patch("imas_codex.graph.GraphClient") as mock_gc_class,
        ):
            mock_gc = MagicMock()
            mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
            mock_gc_class.return_value.__exit__ = MagicMock(return_value=None)

            await seed_worker(state, on_progress=on_progress)

        assert state.seed_phase.done
        # Should have seeded magnetics (3 versions) - dynamic tree has no versions
        assert mock_seed.call_count == 1
        assert mock_seed.call_args[0][1] == "magnetics"
        assert mock_seed.call_args[0][2] == [1, 2, 3]
        assert any("seeded" in msg for msg in progress_calls)

    @pytest.mark.anyio
    async def test_seed_dynamic_tree_with_reference_shot(self):
        """seed_worker uses reference_shot for trees without config versions."""
        from imas_codex.discovery.signals.parallel import seed_worker

        state = DataDiscoveryState(
            facility=FACILITY,
            ssh_host=SSH_HOST,
            scanner_types=["mdsplus"],
            facility_config=FACILITY_CONFIG,
            initial_version_counts={
                "total": 0,
                "discovered": 0,
                "ingested": 0,
                "failed": 0,
            },
            initial_signal_counts={
                "total": 0,
                "discovered": 0,
                "enriched": 0,
                "checked": 0,
            },
            cost_limit=10.0,
            reference_shot=5000,  # Provide reference shot
        )

        with (
            patch(
                "imas_codex.discovery.mdsplus.graph_ops.seed_versions",
                return_value=1,
            ) as mock_seed,
            patch("imas_codex.graph.GraphClient") as mock_gc_class,
        ):
            mock_gc = MagicMock()
            mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
            mock_gc_class.return_value.__exit__ = MagicMock(return_value=None)

            await seed_worker(state)

        # Both trees should be seeded: magnetics(3 versions) + results(reference_shot)
        assert mock_seed.call_count == 2
        calls = {c[0][1]: c[0][2] for c in mock_seed.call_args_list}
        assert calls["magnetics"] == [1, 2, 3]
        assert calls["results"] == [5000]


class TestEpochWorker:
    """Tests for the epoch_worker function."""

    @pytest.mark.anyio
    async def test_epoch_detects_dynamic_tree(self):
        """epoch_worker runs detection on trees without config versions."""
        from imas_codex.discovery.signals.parallel import epoch_worker

        state = DataDiscoveryState(
            facility=FACILITY,
            ssh_host=SSH_HOST,
            scanner_types=["mdsplus"],
            cost_limit=10.0,
            reference_shot=5000,
        )

        mock_epochs = [
            {"version": 5000, "first_shot": 5000},
            {"version": 8000, "first_shot": 8000},
        ]

        progress_calls = []

        def on_progress(msg, stats, results=None):
            progress_calls.append(msg)

        state.facility_config = FACILITY_CONFIG

        with (
            patch(
                "imas_codex.discovery.mdsplus.epochs.detect_epochs_for_tree",
                return_value=mock_epochs,
            ) as mock_detect,
            patch(
                "imas_codex.discovery.signals.parallel.ingest_epochs",
            ) as mock_ingest,
            patch(
                "imas_codex.discovery.mdsplus.graph_ops.seed_versions",
                return_value=2,
            ),
        ):
            await epoch_worker(state, on_progress=on_progress)

        assert state.epoch_phase.done
        mock_detect.assert_called_once()
        mock_ingest.assert_called_once()
        assert any("epochs detected" in msg for msg in progress_calls)

    @pytest.mark.anyio
    async def test_epoch_seeds_per_subtree(self):
        """epoch_worker seeds epoch versions per-subtree, not parent tree.

        When a tree has subtrees (e.g., tcv_shot with results, magnetics),
        epochs detected on the parent should be seeded for each subtree
        so extraction opens the small subtree instead of the full parent.
        """
        from imas_codex.discovery.signals.parallel import epoch_worker

        # Config with parent tree containing subtrees (like TCV)
        config_with_subtrees = {
            "ssh_host": SSH_HOST,
            "data_systems": {
                "mdsplus": {
                    "trees": [
                        {
                            "source_name": "parent_tree",
                            "subtrees": [
                                {"source_name": "results"},
                                {"source_name": "magnetics"},
                            ],
                        },
                    ],
                },
            },
        }

        state = DataDiscoveryState(
            facility=FACILITY,
            ssh_host=SSH_HOST,
            scanner_types=["mdsplus"],
            cost_limit=10.0,
            reference_shot=5000,
        )
        state.facility_config = config_with_subtrees

        mock_epochs = [
            {"version": 5000, "first_shot": 5000},
            {"version": 8000, "first_shot": 8000},
        ]

        seed_calls = []

        def mock_seed(facility, data_source_name, ver_list, version_configs):
            seed_calls.append((facility, data_source_name, ver_list, version_configs))
            return len(ver_list)

        with (
            patch(
                "imas_codex.discovery.mdsplus.epochs.detect_epochs_for_tree",
                return_value=mock_epochs,
            ),
            patch(
                "imas_codex.discovery.signals.parallel.ingest_epochs",
            ),
            patch(
                "imas_codex.discovery.mdsplus.graph_ops.seed_versions",
                side_effect=mock_seed,
            ),
        ):
            await epoch_worker(state, on_progress=lambda *a, **kw: None)

        assert state.epoch_phase.done
        # Should seed per-subtree, NOT for parent_tree
        seeded_trees = [call[1] for call in seed_calls]
        assert "results" in seeded_trees
        assert "magnetics" in seeded_trees
        assert "parent_tree" not in seeded_trees
        # Each subtree should get both epoch versions
        for call in seed_calls:
            assert sorted(call[2]) == [5000, 8000]
            assert call[3] == mock_epochs

    @pytest.mark.anyio
    async def test_epoch_skips_static_trees(self):
        """epoch_worker skips trees that have versions in config."""
        from imas_codex.discovery.signals.parallel import epoch_worker

        # Config where ALL trees have versions — no epochs needed
        config_all_static = {
            "ssh_host": SSH_HOST,
            "data_systems": {
                "mdsplus": {
                    "trees": [STATIC_TREE],  # Only static
                },
            },
        }

        state = DataDiscoveryState(
            facility=FACILITY,
            ssh_host=SSH_HOST,
            scanner_types=["mdsplus"],
            cost_limit=10.0,
            reference_shot=5000,
        )
        state.facility_config = config_all_static

        await epoch_worker(state)

        assert state.epoch_phase.done

    @pytest.mark.anyio
    async def test_epoch_skips_non_mdsplus(self):
        """epoch_worker exits immediately for non-MDSplus scanners."""
        from imas_codex.discovery.signals.parallel import epoch_worker

        state = DataDiscoveryState(
            facility=FACILITY,
            ssh_host=SSH_HOST,
            scanner_types=["tdi"],  # No mdsplus
            cost_limit=10.0,
        )
        state.facility_config = FACILITY_CONFIG

        await epoch_worker(state)

        assert state.epoch_phase.done


class TestExtractWorker:
    """Tests for the mdsplus_extract_worker function."""

    @pytest.mark.anyio
    async def test_extract_claims_and_processes(self):
        """extract_worker claims versions and runs SSH extraction."""
        from imas_codex.discovery.signals.parallel import mdsplus_extract_worker

        _extracted_versions.clear()

        state = DataDiscoveryState(
            facility=FACILITY,
            ssh_host=SSH_HOST,
            scanner_types=["mdsplus"],
            cost_limit=10.0,
        )

        claim_call_count = 0

        def mock_claim(facility):
            nonlocal claim_call_count
            claim_call_count += 1
            # Return first 2 versions, then None
            versions = [
                {
                    "id": f"{facility}:magnetics:v1",
                    "version": 1,
                    "data_source_name": "magnetics",
                    "first_shot": 1000,
                },
                {
                    "id": f"{facility}:magnetics:v2",
                    "version": 2,
                    "data_source_name": "magnetics",
                    "first_shot": 2000,
                },
            ]
            if claim_call_count <= len(versions):
                return versions[claim_call_count - 1]
            # After returning None, mark phase done so worker exits
            state.extract_phase.mark_done()
            return None

        async def mock_extract(
            *, facility, data_source_name, shot, timeout=600, node_usages=None
        ):
            return _make_data_nodes(data_source_name, shot)

        state.facility_config = FACILITY_CONFIG
        state.initial_version_counts = {
            "total": 0,
            "discovered": 0,
            "ingested": 0,
            "failed": 0,
        }
        state.initial_signal_counts = {
            "total": 0,
            "discovered": 0,
            "enriched": 0,
            "checked": 0,
        }

        with (
            patch(
                "imas_codex.discovery.mdsplus.graph_ops.claim_version_for_extraction_facility",
                side_effect=mock_claim,
            ),
            patch(
                "imas_codex.mdsplus.extraction.async_extract_tree_version",
                side_effect=mock_extract,
            ),
            patch(
                "imas_codex.mdsplus.extraction.merge_version_results",
                return_value={},
            ),
            patch(
                "imas_codex.mdsplus.extraction.ingest_static_tree",
            ),
            patch(
                "imas_codex.discovery.mdsplus.graph_ops.mark_version_extracted",
            ),
        ):
            await mdsplus_extract_worker(state)

        assert state.extract_stats.processed == 2


class TestPromoteWorker:
    """Tests for the mdsplus_promote_worker function."""

    @pytest.mark.anyio
    async def test_promote_creates_signals(self):
        """promote_worker creates FacilitySignals from leaf DataNodes."""
        from imas_codex.discovery.signals.parallel import mdsplus_promote_worker

        state = DataDiscoveryState(
            facility=FACILITY,
            ssh_host=SSH_HOST,
            scanner_types=["mdsplus"],
            cost_limit=10.0,
        )
        # Pretend extract already finished one version
        state.extract_stats.processed = 1
        state.extract_phase.mark_done()
        state.units_phase.mark_done()

        promote_call_count = 0

        def mock_claim_promote(facility):
            nonlocal promote_call_count
            promote_call_count += 1
            if promote_call_count == 1:
                return "magnetics"
            state.promote_phase.mark_done()
            return None

        state.initial_signal_counts = {
            "total": 0,
            "discovered": 0,
            "enriched": 0,
            "checked": 0,
        }

        with (
            patch(
                "imas_codex.discovery.mdsplus.graph_ops.claim_tree_for_promote",
                side_effect=mock_claim_promote,
            ),
            patch(
                "imas_codex.discovery.mdsplus.graph_ops.promote_leaf_nodes_to_signals",
                return_value=25,
            ),
        ):
            # Suppress TDI linkage import
            with patch(
                "imas_codex.discovery.mdsplus.tdi_linkage.link_tdi_to_data_nodes",
                return_value=0,
                create=True,
            ):
                await mdsplus_promote_worker(state)

        assert state.promote_stats.processed == 25


# ── Full Pipeline E2E Tests ──────────────────────────────────────────────


class TestPipelineE2E:
    """Full pipeline E2E tests with mocked infrastructure."""

    def _build_patches(self):
        """Build the set of patches for a full pipeline run.

        Workers use lazy imports, so patches target origin modules.
        Module-level imports in parallel.py also need parallel.py targets.
        """
        claim_version_idx = {"value": 0}
        versions_to_claim = [
            {
                "id": f"{FACILITY}:magnetics:v{v}",
                "version": v,
                "data_source_name": "magnetics",
                "first_shot": v * 1000,
            }
            for v in [1, 2, 3]
        ] + [
            {
                "id": f"{FACILITY}:results:v{v}",
                "version": v,
                "data_source_name": "results",
                "first_shot": v,
            }
            for v in [5000, 8000]
        ]

        def mock_claim_version_facility(facility):
            idx = claim_version_idx["value"]
            if idx < len(versions_to_claim):
                claim_version_idx["value"] += 1
                return versions_to_claim[idx]
            return None

        units_tree_idx = {"value": 0}
        trees_for_units = [
            {"data_source_name": "magnetics", "latest_version": 3, "latest_shot": 3000},
            {
                "data_source_name": "results",
                "latest_version": 8000,
                "latest_shot": 8000,
            },
        ]

        def mock_claim_tree_units(facility):
            idx = units_tree_idx["value"]
            if idx < len(trees_for_units):
                units_tree_idx["value"] += 1
                return trees_for_units[idx]
            return None

        promote_tree_idx = {"value": 0}
        trees_for_promote = ["magnetics", "results"]

        def mock_claim_tree_promote(facility):
            idx = promote_tree_idx["value"]
            if idx < len(trees_for_promote):
                promote_tree_idx["value"] += 1
                return trees_for_promote[idx]
            return None

        async def mock_extract(
            *, facility, data_source_name, shot, timeout=600, node_usages=None
        ):
            return _make_data_nodes(data_source_name, shot)

        async def mock_extract_units(
            facility, data_source_name, version, timeout=600, batch_size=500
        ):
            return [
                {"path": f"\\{data_source_name.upper()}::TOP.NODE_{i:03d}", "unit": "V"}
                for i in range(10)
            ]

        mock_epochs = [
            {"version": 5000, "first_shot": 5000},
            {"version": 8000, "first_shot": 8000},
        ]

        mock_gc = MagicMock()
        mock_gc.ensure_facility = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        def make_gc_cm():
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=mock_gc)
            cm.__exit__ = MagicMock(return_value=None)
            return cm

        patches = {
            # --- Lazy imports: patch at origin module ---
            "get_facility": patch(
                "imas_codex.discovery.base.facility.get_facility",
                return_value=FACILITY_CONFIG,
            ),
            "get_scanners": patch(
                "imas_codex.discovery.signals.scanners.base.get_scanners_for_facility",
                return_value=[MagicMock(scanner_type="mdsplus")],
            ),
            "seed_versions": patch(
                "imas_codex.discovery.mdsplus.graph_ops.seed_versions",
                return_value=3,
            ),
            "detect_epochs": patch(
                "imas_codex.discovery.mdsplus.epochs.detect_epochs_for_tree",
                return_value=mock_epochs,
            ),
            "claim_version": patch(
                "imas_codex.discovery.mdsplus.graph_ops.claim_version_for_extraction_facility",
                side_effect=mock_claim_version_facility,
            ),
            "mark_extracted": patch(
                "imas_codex.discovery.mdsplus.graph_ops.mark_version_extracted",
            ),
            "release_version": patch(
                "imas_codex.discovery.mdsplus.graph_ops.release_version_claim",
            ),
            "claim_units": patch(
                "imas_codex.discovery.mdsplus.graph_ops.claim_tree_for_units",
                side_effect=mock_claim_tree_units,
            ),
            "merge_units": patch(
                "imas_codex.discovery.mdsplus.graph_ops.merge_units_to_graph",
            ),
            "mark_units": patch(
                "imas_codex.discovery.mdsplus.graph_ops.mark_all_versions_units_extracted",
            ),
            "claim_promote": patch(
                "imas_codex.discovery.mdsplus.graph_ops.claim_tree_for_promote",
                side_effect=mock_claim_tree_promote,
            ),
            "promote": patch(
                "imas_codex.discovery.mdsplus.graph_ops.promote_leaf_nodes_to_signals",
                return_value=LEAF_NODES_PER_VERSION,
            ),
            "extract_tree": patch(
                "imas_codex.mdsplus.extraction.async_extract_tree_version",
                side_effect=mock_extract,
            ),
            "merge_results": patch(
                "imas_codex.mdsplus.extraction.merge_version_results",
                return_value={},
            ),
            "ingest_static": patch(
                "imas_codex.mdsplus.extraction.ingest_static_tree",
            ),
            "extract_units": patch(
                "imas_codex.mdsplus.extraction.async_extract_units_for_version",
                side_effect=mock_extract_units,
            ),
            "link_tdi": patch(
                "imas_codex.discovery.mdsplus.tdi_linkage.link_tdi_to_data_nodes",
                return_value=0,
                create=True,
            ),
            # --- Module-level graph_ops has_work_fn patches ---
            "has_pending_extract": patch(
                "imas_codex.discovery.mdsplus.graph_ops.has_pending_extract_work_facility",
                return_value=False,
            ),
            "has_pending_units": patch(
                "imas_codex.discovery.mdsplus.graph_ops.has_pending_units_work_facility",
                return_value=False,
            ),
            "has_pending_promote": patch(
                "imas_codex.discovery.mdsplus.graph_ops.has_pending_promote_work_facility",
                return_value=False,
            ),
            "get_version_counts": patch(
                "imas_codex.discovery.mdsplus.graph_ops.get_version_counts",
                return_value={"total": 0, "discovered": 0, "ingested": 0, "failed": 0},
            ),
            "get_signal_counts": patch(
                "imas_codex.discovery.mdsplus.graph_ops.get_signal_counts",
                return_value={"total": 0, "discovered": 0, "enriched": 0, "checked": 0},
            ),
            # --- Module-level imports in parallel.py ---
            "gc_module": patch(
                "imas_codex.discovery.signals.parallel.GraphClient",
                side_effect=make_gc_cm,
            ),
            "gc_origin": patch(
                "imas_codex.graph.GraphClient",
                side_effect=make_gc_cm,
            ),
            "reset_transient": patch(
                "imas_codex.discovery.signals.parallel.reset_transient_signals",
                return_value={},
            ),
            "ingest_epochs": patch(
                "imas_codex.discovery.signals.parallel.ingest_epochs",
            ),
            "orphan_recovery": patch(
                "imas_codex.discovery.base.engine.make_orphan_recovery_tick",
                return_value=AsyncMock(),
            ),
        }
        return patches

    @pytest.mark.anyio
    @pytest.mark.timeout(30)
    async def test_discover_only_static_and_dynamic(self):
        """Full pipeline run in discover_only mode processes both tree types.

        Static tree (magnetics): 3 versions from config
        Dynamic tree (results): 2 epochs detected at runtime
        Total: 5 versions × 25 nodes = 125 DataNodes, 50 FacilitySignals
        """
        patches = self._build_patches()
        mocks = {}
        for name, p in patches.items():
            mocks[name] = p.start()

        try:
            result = await run_parallel_data_discovery(
                facility=FACILITY,
                ssh_host=SSH_HOST,
                scanner_types=["mdsplus"],
                reference_shot=5000,
                cost_limit=10.0,
                discover_only=True,
                num_enrich_workers=1,
                num_check_workers=1,
            )
        finally:
            for p in patches.values():
                p.stop()

        # Seed worker should have seeded static versions
        assert mocks["seed_versions"].call_count >= 1

        # Epoch detection ran for dynamic tree
        mocks["detect_epochs"].assert_called_once()

        # Extract worker processed 5 versions (3 static + 2 dynamic)
        assert result["extract_count"] == 5

        # Units worker processed 2 trees
        assert result["units_count"] == 2

        # Promote worker created signals for 2 trees
        # 2 trees × 25 nodes = 50 signals
        assert result["promote_count"] == 50

        # No enrichment/checking in discover_only
        assert result["enriched"] == 0
        assert result["checked"] == 0
        assert result["cost"] == 0.0

        # Total discovered includes seeded + promoted
        assert result["discovered"] >= 50
        assert result["elapsed_seconds"] > 0

    @pytest.mark.anyio
    @pytest.mark.timeout(30)
    async def test_discover_only_static_trees_only(self):
        """Pipeline with only static trees (no epoch detection needed).

        3 versions × 25 nodes = 75 DataNodes, 25 FacilitySignals (1 tree)
        """
        config_static_only = {
            "ssh_host": SSH_HOST,
            "data_systems": {
                "mdsplus": {
                    "setup_commands": [],
                    "trees": [STATIC_TREE],
                    "connection_tree": "magnetics",
                },
            },
        }

        patches = self._build_patches()
        # Override config to static-only
        patches["get_facility"] = patch(
            "imas_codex.discovery.base.facility.get_facility",
            return_value=config_static_only,
        )

        # Override claim functions for single-tree scenario
        claim_idx = {"value": 0}
        versions = [
            {
                "id": f"{FACILITY}:magnetics:v{v}",
                "version": v,
                "data_source_name": "magnetics",
                "first_shot": v * 1000,
            }
            for v in [1, 2, 3]
        ]

        def mock_claim(facility):
            idx = claim_idx["value"]
            if idx < len(versions):
                claim_idx["value"] += 1
                return versions[idx]
            return None

        units_idx = {"value": 0}

        def mock_claim_units(facility):
            if units_idx["value"] == 0:
                units_idx["value"] += 1
                return {"data_source_name": "magnetics", "latest_version": 3}
            return None

        promote_idx = {"value": 0}

        def mock_claim_promote(facility):
            if promote_idx["value"] == 0:
                promote_idx["value"] += 1
                return "magnetics"
            return None

        patches["claim_version"] = patch(
            "imas_codex.discovery.mdsplus.graph_ops.claim_version_for_extraction_facility",
            side_effect=mock_claim,
        )
        patches["claim_units"] = patch(
            "imas_codex.discovery.mdsplus.graph_ops.claim_tree_for_units",
            side_effect=mock_claim_units,
        )
        patches["claim_promote"] = patch(
            "imas_codex.discovery.mdsplus.graph_ops.claim_tree_for_promote",
            side_effect=mock_claim_promote,
        )

        mocks = {}
        for name, p in patches.items():
            mocks[name] = p.start()

        try:
            result = await run_parallel_data_discovery(
                facility=FACILITY,
                ssh_host=SSH_HOST,
                scanner_types=["mdsplus"],
                cost_limit=10.0,
                discover_only=True,
                num_enrich_workers=1,
                num_check_workers=1,
            )
        finally:
            for p in patches.values():
                p.stop()

        # No epochs detected for static-only config (static tree has versions)
        mocks["detect_epochs"].assert_not_called()

        # 3 static versions extracted
        assert result["extract_count"] == 3

        # 1 tree with units
        assert result["units_count"] == 1

        # 1 tree promoted with 25 signals
        assert result["promote_count"] == 25

    @pytest.mark.anyio
    @pytest.mark.timeout(30)
    async def test_enrich_only_skips_scan(self):
        """enrich_only mode pre-marks scan phases done and only enriches.

        This tests that when enrich_only=True, no scan/extract/epoch workers
        are started, wiki context is preloaded, and only enrich/embed workers
        are wired into the engine.
        """
        patches = self._build_patches()
        patches["get_facility"] = patch(
            "imas_codex.discovery.base.facility.get_facility",
            return_value={
                **FACILITY_CONFIG,
                "wiki_sites": ["https://example.test/wiki"],
            },
        )
        mocks = {}
        for name, p in patches.items():
            mocks[name] = p.start()

        engine_calls = []

        async def mock_run_engine(state, workers, **kwargs):
            engine_calls.append(
                {
                    "scanner_types": state.scanner_types,
                    "wiki_context": state.wiki_context,
                    "worker_names": [worker.name for worker in workers],
                }
            )
            state.enrich_phase.mark_done()
            state.check_phase.mark_done()

        with (
            patch(
                "imas_codex.discovery.signals.parallel.run_discovery_engine",
                side_effect=mock_run_engine,
            ),
            patch(
                "imas_codex.discovery.signals.scanners.wiki.load_wiki_context",
                return_value={"\\MAGNETICS::TOP.NODE_000": {"description": "wiki"}},
            ) as load_wiki_context,
            patch(
                "imas_codex.discovery.signals.parallel.individualize_source_descriptions",
                new=AsyncMock(return_value=0),
            ),
            patch(
                "imas_codex.discovery.base.embed_worker.embed_description_worker",
                new_callable=AsyncMock,
            ),
        ):
            result = await run_parallel_data_discovery(
                facility=FACILITY,
                ssh_host=SSH_HOST,
                scanner_types=["mdsplus"],
                cost_limit=10.0,
                enrich_only=True,
                num_enrich_workers=1,
                num_check_workers=0,
            )

        for p in patches.values():
            p.stop()

        # No scan work should have happened
        assert result["extract_count"] == 0
        assert result["units_count"] == 0
        assert result["promote_count"] == 0
        assert engine_calls == [
            {
                "scanner_types": ["mdsplus"],
                "wiki_context": {"\\MAGNETICS::TOP.NODE_000": {"description": "wiki"}},
                "worker_names": ["enrich", "embed"],
            }
        ]
        load_wiki_context.assert_called_once_with(FACILITY)

    @pytest.mark.anyio
    @pytest.mark.timeout(30)
    async def test_check_worker_passes_scanner_scope_to_claims(self):
        """Check worker claims are scoped to the requested scanners."""
        state = DataDiscoveryState(
            facility=FACILITY,
            ssh_host=SSH_HOST,
            facility_config=FACILITY_CONFIG,
            scanner_types=["ppf"],
            reference_shot=5000,
            cost_limit=10.0,
        )

        claim_calls = []

        def mock_claim_check(
            facility,
            batch_size=5,
            reference_shot=None,
            scanner_types=None,
        ):
            claim_calls.append(
                {
                    "facility": facility,
                    "batch_size": batch_size,
                    "reference_shot": reference_shot,
                    "scanner_types": scanner_types,
                }
            )
            state.stop_requested = True
            return []

        with (
            patch(
                "imas_codex.discovery.signals.parallel.claim_signals_for_check",
                side_effect=mock_claim_check,
            ),
            patch("asyncio.sleep", new=AsyncMock()),
        ):
            await check_worker(state)

        assert claim_calls == [
            {
                "facility": FACILITY,
                "batch_size": 20,
                "reference_shot": 5000,
                "scanner_types": ["ppf"],
            }
        ]

    @pytest.mark.anyio
    @pytest.mark.timeout(30)
    async def test_signal_counts_50_paths(self):
        """Pipeline produces 50 FacilitySignals from 2 trees × 25 nodes.

        Verifies the "50 paths" requirement from the task spec, testing
        that the promote step correctly accounts for both trees producing
        LEAF_NODES_PER_VERSION (25) signals each.
        """
        patches = self._build_patches()
        mocks = {}
        for name, p in patches.items():
            mocks[name] = p.start()

        try:
            result = await run_parallel_data_discovery(
                facility=FACILITY,
                ssh_host=SSH_HOST,
                scanner_types=["mdsplus"],
                reference_shot=5000,
                cost_limit=10.0,
                discover_only=True,
                num_enrich_workers=1,
                num_check_workers=1,
            )
        finally:
            for p in patches.values():
                p.stop()

        # 2 trees × 25 = 50 promoted signals
        assert result["promote_count"] == 50, (
            f"Expected 50 promoted signals, got {result['promote_count']}"
        )

    @pytest.mark.anyio
    @pytest.mark.timeout(15)
    async def test_deadline_stops_pipeline(self):
        """Pipeline respects deadline and stops all workers."""
        import time

        patches = self._build_patches()
        mocks = {}
        for name, p in patches.items():
            mocks[name] = p.start()

        try:
            result = await run_parallel_data_discovery(
                facility=FACILITY,
                ssh_host=SSH_HOST,
                scanner_types=["mdsplus"],
                reference_shot=5000,
                cost_limit=10.0,
                discover_only=True,
                deadline=time.time() + 0.1,  # 100ms deadline
                num_enrich_workers=1,
                num_check_workers=1,
            )
        finally:
            for p in patches.values():
                p.stop()

        # Pipeline should have stopped quickly
        assert result["elapsed_seconds"] < 10

    @pytest.mark.anyio
    @pytest.mark.timeout(30)
    async def test_stop_event_triggers_shutdown(self):
        """External stop_event triggers clean pipeline shutdown."""
        patches = self._build_patches()
        mocks = {}
        for name, p in patches.items():
            mocks[name] = p.start()

        stop_event = asyncio.Event()

        async def fake_watch_stop_event(event, state):
            await asyncio.sleep(0.2)
            state.stop_requested = True
            event.set()

        with patch(
            "imas_codex.cli.shutdown.watch_stop_event",
            side_effect=fake_watch_stop_event,
        ):
            try:
                result = await run_parallel_data_discovery(
                    facility=FACILITY,
                    ssh_host=SSH_HOST,
                    scanner_types=["mdsplus"],
                    reference_shot=5000,
                    cost_limit=10.0,
                    discover_only=True,
                    stop_event=stop_event,
                    num_enrich_workers=1,
                    num_check_workers=1,
                )
            finally:
                for p in patches.values():
                    p.stop()

        # Pipeline ran and returned
        assert "elapsed_seconds" in result


# ── CLI E2E Tests ────────────────────────────────────────────────────────


class TestCLISignals:
    """Tests for the signals CLI command with mocked pipeline."""

    def test_cli_help(self):
        """CLI signals command responds to --help."""
        from click.testing import CliRunner

        from imas_codex.cli.discover.signals import signals

        runner = CliRunner()
        result = runner.invoke(signals, ["--help"])
        assert result.exit_code == 0
        assert "Discover signals" in result.output
        assert "--scan-only" in result.output
        assert "--enrich-only" in result.output
        assert "--scanners" in result.output
        assert "--cost-limit" in result.output

    def test_cli_unknown_scanner_exits(self):
        """CLI rejects unknown scanner types."""
        from click.testing import CliRunner

        from imas_codex.cli.discover.signals import signals

        runner = CliRunner()
        with (
            patch(
                "imas_codex.cli.rich_output.should_use_rich",
                return_value=False,
            ),
            patch(
                "imas_codex.cli.logging.configure_cli_logging",
            ),
            patch(
                "imas_codex.discovery.base.facility.get_facility",
                return_value=FACILITY_CONFIG,
            ),
            patch(
                "imas_codex.discovery.signals.scanners.base.list_scanners",
                return_value=["mdsplus", "tdi"],
            ),
        ):
            result = runner.invoke(
                signals,
                [FACILITY, "-s", "nonexistent_scanner"],
                catch_exceptions=False,
            )
        assert result.exit_code != 0

    def test_cli_scan_only_invokes_pipeline(self):
        """CLI --scan-only routes to run_parallel_data_discovery with discover_only."""
        from click.testing import CliRunner

        from imas_codex.cli.discover.signals import signals

        runner = CliRunner()

        captured_kwargs = {}

        async def mock_pipeline(**kwargs):
            captured_kwargs.update(kwargs)
            return {
                "scanned": 10,
                "discovered": 50,
                "enriched": 0,
                "checked": 0,
                "cost": 0.0,
                "elapsed_seconds": 1.5,
                "extract_count": 5,
                "units_count": 2,
                "promote_count": 50,
            }

        with (
            patch(
                "imas_codex.cli.rich_output.should_use_rich",
                return_value=False,
            ),
            patch(
                "imas_codex.cli.logging.configure_cli_logging",
            ),
            patch(
                "imas_codex.discovery.base.facility.get_facility",
                return_value=FACILITY_CONFIG,
            ),
            patch(
                "imas_codex.discovery.signals.scanners.base.get_scanners_for_facility",
                return_value=[MagicMock(scanner_type="mdsplus")],
            ),
            patch(
                "imas_codex.discovery.signals.scanners.base.list_scanners",
                return_value=["mdsplus", "tdi"],
            ),
            patch(
                "imas_codex.discovery.signals.parallel.run_parallel_data_discovery",
                side_effect=mock_pipeline,
            ),
            patch(
                "imas_codex.cli.shutdown.safe_asyncio_run",
                side_effect=lambda coro: asyncio.run(coro),
            ),
            patch(
                "imas_codex.cli.shutdown.install_shutdown_handlers",
            ),
        ):
            result = runner.invoke(
                signals,
                [FACILITY, "--scan-only", "-s", "mdsplus"],
                catch_exceptions=False,
            )

        # CLI should complete successfully
        assert result.exit_code == 0, f"CLI failed: {result.output}"


# ── Signal Pattern Detection Tests ───────────────────────────────────────


class TestSignalPatternDetection:
    """Tests for indexed signal pattern detection and propagation."""

    def test_accessor_to_source_key(self):
        """Numeric segments are replaced with NNN."""
        from imas_codex.discovery.signals.parallel import _accessor_to_source_key

        assert (
            _accessor_to_source_key("CALIB_GAS_010:PROPERTIES:PARAM_048:LIM")
            == "CALIB_GAS_NNN:PROPERTIES:PARAM_NNN:LIM"
        )
        assert (
            _accessor_to_source_key("WAVE_GEN_A:OUTPUT_064:OFFSET")
            == "WAVE_GEN_A:OUTPUT_NNN:OFFSET"
        )
        assert (
            _accessor_to_source_key("TOP.INPUTS.S_DSP_003:ADC_GAIN")
            == "TOP.INPUTS.S_DSP_NNN:ADC_GAIN"
        )
        # Single digit should NOT be replaced (min 2 digits)
        assert _accessor_to_source_key("COIL_R:S1") == "COIL_R:S1"
        # No numbers should be unchanged
        assert _accessor_to_source_key("IP:VALUE") == "IP:VALUE"

    def test_detect_signal_sources(self):
        """Pattern detection groups indexed signals and creates SignalSource nodes."""
        from imas_codex.discovery.signals.parallel import detect_signal_sources

        # Mock GraphClient to return indexed signals
        mock_results = [
            {"id": f"tcv:sig_{i:03d}_param_a", "accessor": f"GAS_{i:03d}:PARAM:A"}
            for i in range(10)
        ] + [
            {"id": "tcv:unique_signal", "accessor": "UNIQUE:SIGNAL"},
        ]

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)
        mock_gc.query = MagicMock(return_value=mock_results)

        with patch(
            "imas_codex.discovery.signals.parallel.GraphClient",
            return_value=mock_gc,
        ):
            patterns, followers = detect_signal_sources("tcv", min_instances=3)

        # Should detect 1 pattern (GAS_NNN:PARAM:A) with 10 signals
        assert patterns == 1
        assert followers == 10  # All 10 signals linked as members

        # First query: fetch signals, second: ensure_facility, third: create group+members
        assert mock_gc.query.call_count >= 2
        # Verify member_ids were passed (all 10)
        group_call_kwargs = mock_gc.query.call_args_list[1]
        member_ids = group_call_kwargs.kwargs.get("member_ids") or group_call_kwargs[
            1
        ].get("member_ids")
        assert len(member_ids) == 10

    def test_detect_groups_below_threshold(self):
        """Groups below min_instances threshold are not detected."""
        from imas_codex.discovery.signals.parallel import detect_signal_sources

        # Only 2 signals in the group (below default min_instances=3)
        mock_results = [
            {"id": "tcv:sig_01_a", "accessor": "GAS_01:A"},
            {"id": "tcv:sig_02_a", "accessor": "GAS_02:A"},
            {"id": "tcv:unique", "accessor": "UNIQUE:SIGNAL"},
        ]

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=mock_gc)
        mock_gc.query = MagicMock(return_value=mock_results)

        with patch(
            "imas_codex.discovery.signals.parallel.GraphClient",
            return_value=mock_gc,
        ):
            patterns, followers = detect_signal_sources("tcv", min_instances=3)

        assert patterns == 0
        assert followers == 0

    def test_propagate_source_enrichment(self):
        """Enrichment is propagated from representative to group members."""
        from imas_codex.discovery.signals.parallel import (
            propagate_source_enrichment,
        )

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)
        # First call: count discovered followers -> 5
        # Second call: update SignalSource node -> void
        # Third call: propagate discovered → enriched -> 5 updated
        mock_gc.query = MagicMock(
            side_effect=[
                [{"cnt": 5}],
                [],  # SignalSource update
                [{"updated": 5}],
                [],  # diagnostic creation
            ]
        )

        enrichment = {
            "physics_domain": "plasma_control",
            "description": "Gas injection parameter limit",
            "name": "Gas Calibration Limit",
            "diagnostic": "gas_injection",
            "analysis_code": "",
            "keywords": ["gas", "calibration", "limit"],
            "sign_convention": "",
        }

        with patch(
            "imas_codex.discovery.signals.parallel.GraphClient",
            return_value=mock_gc,
        ):
            result = propagate_source_enrichment(
                "tcv:rep_signal", enrichment, batch_cost=0.01
            )

        assert result == 5
