"""Tests for unified tree discovery pipeline (discovery/mdsplus/)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.discovery.mdsplus.state import TreeDiscoveryState


class TestTreeDiscoveryState:
    """Test TreeDiscoveryState dataclass."""

    def test_has_required_fields(self):
        state = TreeDiscoveryState(
            facility="tcv",
            ssh_host="tcv",
            tree_name="static",
            tree_config={"versions": [{"version": 1}]},
        )
        assert state.facility == "tcv"
        assert state.tree_name == "static"
        assert state.stop_requested is False

    def test_should_stop_on_request(self):
        state = TreeDiscoveryState(
            facility="tcv",
            ssh_host="tcv",
            tree_name="static",
            tree_config={},
        )
        assert state.should_stop() is False
        state.stop_requested = True
        assert state.should_stop() is True

    def test_should_stop_on_deadline(self):
        import time

        state = TreeDiscoveryState(
            facility="tcv",
            ssh_host="tcv",
            tree_name="static",
            tree_config={},
            deadline=time.time() - 1,  # Already past
        )
        assert state.should_stop() is True

    def test_has_promote_phase(self):
        state = TreeDiscoveryState(
            facility="tcv",
            ssh_host="tcv",
            tree_name="static",
            tree_config={},
        )
        assert hasattr(state, "promote_phase")
        assert hasattr(state, "promote_stats")
        assert state.promote_phase.name == "promote"

    def test_no_enrich_fields(self):
        """TreeDiscoveryState should not have enrich-related fields."""
        state = TreeDiscoveryState(
            facility="tcv",
            ssh_host="tcv",
            tree_name="static",
            tree_config={},
        )
        assert not hasattr(state, "enrich_stats")
        assert not hasattr(state, "enrich_phase")
        assert not hasattr(state, "enrich")
        # budget_exhausted is inherited from DiscoveryStateBase but always False
        # for tree discovery (no cost tracking)
        assert not state.budget_exhausted


class TestBackwardCompatState:
    """Test backward-compatible alias."""

    def test_alias_exists_in_mdsplus_state(self):
        from imas_codex.discovery.mdsplus.state import StaticDiscoveryState

        assert StaticDiscoveryState is TreeDiscoveryState

    def test_old_state_still_works(self):
        from imas_codex.discovery.static.state import StaticDiscoveryState

        state = StaticDiscoveryState(
            facility="tcv",
            ssh_host="tcv",
            tree_name="static",
            tree_config={},
        )
        # Old state still has enrich fields
        assert hasattr(state, "enrich_stats")
        assert hasattr(state, "enrich_phase")

    def test_old_and_new_are_different_classes(self):
        """Old StaticDiscoveryState and new TreeDiscoveryState are different."""
        from imas_codex.discovery.static.state import StaticDiscoveryState

        assert StaticDiscoveryState is not TreeDiscoveryState


class TestPipelineExports:
    """Test pipeline module exports."""

    def test_run_tree_discovery_importable(self):
        from imas_codex.discovery.mdsplus.pipeline import run_tree_discovery

        assert callable(run_tree_discovery)

    def test_backward_compat_alias(self):
        from imas_codex.discovery.mdsplus.pipeline import (
            run_parallel_static_discovery,
            run_tree_discovery,
        )

        assert run_parallel_static_discovery is run_tree_discovery

    def test_init_exports(self):
        from imas_codex.discovery.mdsplus import (
            TreeDiscoveryState,
            run_tree_discovery,
        )

        assert callable(run_tree_discovery)


class TestWorkerExports:
    """Test worker module exports."""

    def test_workers_importable(self):
        from imas_codex.discovery.mdsplus.workers import (
            extract_worker,
            promote_worker,
            units_worker,
        )

        assert callable(extract_worker)
        assert callable(units_worker)
        assert callable(promote_worker)

    def test_no_enrich_worker(self):
        """New workers module should not export enrich_worker."""
        import imas_codex.discovery.mdsplus.workers as w

        assert not hasattr(w, "enrich_worker")


class TestGraphOps:
    """Test graph_ops module."""

    def test_has_core_ops(self):
        from imas_codex.discovery.mdsplus.graph_ops import (
            claim_version_for_extraction,
            mark_version_extracted,
            seed_versions,
        )

        assert callable(seed_versions)
        assert callable(claim_version_for_extraction)
        assert callable(mark_version_extracted)

    def test_has_promote_function(self):
        from imas_codex.discovery.mdsplus.graph_ops import (
            promote_leaf_nodes_to_signals,
        )

        assert callable(promote_leaf_nodes_to_signals)

    def test_static_shim_re_exports(self):
        """Static graph_ops shim re-exports from mdsplus."""
        from imas_codex.discovery.mdsplus.graph_ops import seed_versions as sv1
        from imas_codex.discovery.static.graph_ops import seed_versions as sv2

        assert sv1 is sv2


class TestPromoteWorker:
    """Test promote_worker behavior."""

    @pytest.mark.asyncio
    @patch("imas_codex.discovery.mdsplus.graph_ops.promote_leaf_nodes_to_signals")
    async def test_waits_for_extract_and_units(self, mock_promote):
        """Promote worker waits for extract+units to complete."""
        from imas_codex.discovery.mdsplus.workers import promote_worker

        mock_promote.return_value = 42
        progress_calls = []

        state = TreeDiscoveryState(
            facility="tcv",
            ssh_host="tcv",
            tree_name="static",
            tree_config={},
        )
        # Mark extract and units as done immediately
        state.extract_phase.mark_done()
        state.units_phase.mark_done()

        await promote_worker(
            state, on_progress=lambda msg, stats, items: progress_calls.append(msg)
        )

        mock_promote.assert_called_once_with("tcv", "static")
        assert state.promote_stats.processed == 42
        assert state.promote_phase.done
        assert any("42 signals" in c for c in progress_calls)
