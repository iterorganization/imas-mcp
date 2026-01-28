"""Tests for parallel discovery engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.discovery.parallel import (
    DiscoveryState,
    WorkerStats,
)


class TestWorkerStats:
    """Tests for WorkerStats dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        stats = WorkerStats()
        assert stats.processed == 0
        assert stats.errors == 0
        assert stats.cost == 0.0

    def test_rate_with_no_progress(self):
        """Test rate is None when no work done."""
        stats = WorkerStats()
        stats.processed = 0
        assert stats.rate is None

    def test_rate_calculation(self):
        """Test rate is calculated correctly."""
        import time

        stats = WorkerStats()
        stats.start_time = time.time() - 10  # 10 seconds ago
        stats.processed = 50
        rate = stats.rate
        assert rate is not None
        assert 4.5 < rate < 5.5  # ~5 per second


class TestDiscoveryState:
    """Tests for DiscoveryState dataclass."""

    def test_budget_not_exhausted_initially(self):
        """Test budget is not exhausted at start."""
        state = DiscoveryState(facility="test", cost_limit=10.0)
        assert not state.budget_exhausted

    def test_budget_exhausted_when_over_limit(self):
        """Test budget is exhausted when cost exceeds limit."""
        state = DiscoveryState(facility="test", cost_limit=10.0)
        state.score_stats.cost = 10.5
        assert state.budget_exhausted

    def test_should_stop_when_budget_exhausted(self):
        """Test should_stop returns True when budget exhausted."""
        state = DiscoveryState(facility="test", cost_limit=5.0)
        state.score_stats.cost = 5.5
        assert state.should_stop()

    @patch("imas_codex.discovery.parallel.has_pending_work", return_value=False)
    def test_should_stop_when_both_idle(self, mock_has_pending):
        """Test should_stop returns True when all workers idle."""
        state = DiscoveryState(facility="test", cost_limit=10.0)
        state.scan_idle_count = 3
        state.expand_idle_count = 3
        state.score_idle_count = 3
        state.enrich_idle_count = 3
        state.rescore_idle_count = 3
        assert state.should_stop()

    def test_should_not_stop_when_one_active(self):
        """Test should_stop returns False when one worker active."""
        state = DiscoveryState(facility="test", cost_limit=10.0)
        state.scan_idle_count = 5
        state.score_idle_count = 1  # Still active
        assert not state.should_stop()

    def test_should_stop_when_requested(self):
        """Test should_stop returns True when stop requested."""
        state = DiscoveryState(facility="test", cost_limit=10.0)
        state.stop_requested = True
        assert state.should_stop()


class TestClaimPaths:
    """Tests for atomic path claiming functions."""

    @patch("imas_codex.graph.GraphClient")
    def test_claim_paths_for_scanning_calls_graph(self, mock_gc_class):
        """Test claim_paths_for_scanning calls graph correctly."""
        from imas_codex.discovery.parallel import claim_paths_for_scanning

        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=None)
        # Single query returns unscored paths (expansion now handled by expand_worker)
        mock_gc.query.return_value = [
            {
                "id": "test:/path1",
                "path": "/path1",
                "depth": 1,
                "is_expanding": False,
            }
        ]

        result = claim_paths_for_scanning("test", limit=50)

        assert mock_gc.query.call_count == 1  # Only unscored query
        assert len(result) == 1
        assert result[0]["path"] == "/path1"
        assert result[0]["is_expanding"] is False

    @patch("imas_codex.graph.GraphClient")
    def test_claim_paths_for_expanding_calls_graph(self, mock_gc_class):
        """Test claim_paths_for_expanding claims expansion paths."""
        from imas_codex.discovery.parallel import claim_paths_for_expanding

        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=None)
        mock_gc.query.return_value = [
            {
                "id": "test:/path2",
                "path": "/path2",
                "depth": 2,
                "is_expanding": True,
            }
        ]

        result = claim_paths_for_expanding("test", limit=50)

        assert mock_gc.query.call_count == 1
        assert len(result) == 1
        assert result[0]["path"] == "/path2"
        assert result[0]["is_expanding"] is True

    @patch("imas_codex.graph.GraphClient")
    def test_claim_paths_for_scoring_calls_graph(self, mock_gc_class):
        """Test claim_paths_for_scoring calls graph correctly."""
        from imas_codex.discovery.parallel import claim_paths_for_scoring

        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=None)
        mock_gc.query.return_value = [
            {
                "id": "test:/path1",
                "path": "/path1",
                "depth": 1,
                "total_files": 10,
                "total_dirs": 2,
            }
        ]

        result = claim_paths_for_scoring("test", limit=25)

        assert mock_gc.query.called
        assert len(result) == 1
        assert result[0]["total_files"] == 10
