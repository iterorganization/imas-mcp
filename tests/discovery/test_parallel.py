"""Tests for parallel discovery engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.discovery.paths.parallel import (
    DiscoveryState,
    WorkerStats,
)


class TestGetCheckpointDir:
    """Tests for checkpoint directory utility."""

    def test_get_checkpoint_dir_creates_directory(self, tmp_path, monkeypatch):
        """Test checkpoint directory is created if it doesn't exist."""
        from pathlib import Path

        # Patch home to use tmp_path
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        from imas_codex.discovery.signals.parallel import get_checkpoint_dir

        checkpoint_dir = get_checkpoint_dir()

        expected_path = tmp_path / ".local/share/imas-codex/checkpoints/data"
        assert checkpoint_dir == expected_path
        assert checkpoint_dir.exists()
        assert checkpoint_dir.is_dir()

    def test_get_checkpoint_dir_idempotent(self, tmp_path, monkeypatch):
        """Test get_checkpoint_dir can be called multiple times safely."""
        from pathlib import Path

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        from imas_codex.discovery.signals.parallel import get_checkpoint_dir

        dir1 = get_checkpoint_dir()
        dir2 = get_checkpoint_dir()

        assert dir1 == dir2
        assert dir1.exists()


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

    @patch("imas_codex.discovery.paths.parallel.has_pending_work", return_value=False)
    def test_should_stop_when_both_idle(self, mock_has_pending):
        """Test should_stop returns True when all workers idle."""
        state = DiscoveryState(facility="test", cost_limit=10.0)
        state.scan_phase._idle_count = 3
        state.expand_phase._idle_count = 3
        state.score_phase._idle_count = 3
        state.enrich_phase._idle_count = 3
        state.refine_phase._idle_count = 3
        assert state.should_stop()

    @patch("imas_codex.discovery.paths.parallel.has_pending_work", return_value=True)
    def test_should_not_stop_when_pending_work(self, mock_has_pending):
        """Test should_stop returns False when has_pending_work reports work.

        Even when all workers are idle, should_stop must return False if
        has_pending_work returns True. This covers the case where a worker
        has claimed paths and is mid-task (e.g., LLM scoring call).
        has_pending_work counts claimed paths as in-progress work.
        """
        state = DiscoveryState(facility="test", cost_limit=10.0)
        state.scan_phase._idle_count = 3
        state.expand_phase._idle_count = 3
        state.score_phase._idle_count = 3
        state.enrich_phase._idle_count = 3
        state.refine_phase._idle_count = 3
        assert not state.should_stop()
        # Idle counts should have been reset
        assert state.scan_phase.idle_count == 0
        assert state.score_phase.idle_count == 0

    def test_should_not_stop_when_one_active(self):
        """Test should_stop returns False when one worker active."""
        state = DiscoveryState(facility="test", cost_limit=10.0)
        state.scan_phase._idle_count = 5
        state.score_phase._idle_count = 1  # Still active
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
        from imas_codex.discovery.paths.parallel import claim_paths_for_scanning

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
        from imas_codex.discovery.paths.parallel import claim_paths_for_expanding

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
        from imas_codex.discovery.paths.parallel import claim_paths_for_scoring

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


class TestFindCloneGroups:
    """Tests for _find_clone_groups graph query."""

    @patch("imas_codex.graph.GraphClient")
    def test_returns_sorted_groups(self, mock_gc_class):
        """Clone groups are sorted: accessible > has_remote > shallowest > earliest."""
        from imas_codex.discovery.paths.parallel import _find_clone_groups

        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=None)
        mock_gc.query.return_value = [
            {
                "repo_id": "github.com/user/MEQ",
                "repo_name": "MEQ",
                "active_paths": [
                    {
                        "id": "tcv:/deep/clone",
                        "path": "/deep/clone",
                        "depth": 5,
                        "accessible": False,
                        "has_remote": 1,
                        "discovered_at": "2024-01-02",
                    },
                    {
                        "id": "tcv:/shallow/meq",
                        "path": "/shallow/meq",
                        "depth": 2,
                        "accessible": True,
                        "has_remote": 1,
                        "discovered_at": "2024-01-01",
                    },
                    {
                        "id": "tcv:/mid/meq",
                        "path": "/mid/meq",
                        "depth": 3,
                        "accessible": False,
                        "has_remote": 0,
                        "discovered_at": "2024-01-03",
                    },
                ],
            },
        ]

        groups = _find_clone_groups("tcv", limit=10)

        assert len(groups) == 1
        repo_id, repo_name, paths = groups[0]
        assert repo_id == "github.com/user/MEQ"
        assert repo_name == "MEQ"
        # Canonical (first) = accessible remote
        assert paths[0]["id"] == "tcv:/shallow/meq"
        # Second = has remote but not accessible, shallower
        assert paths[1]["id"] == "tcv:/deep/clone"
        # Last = no remote
        assert paths[2]["id"] == "tcv:/mid/meq"

    @patch("imas_codex.graph.GraphClient")
    def test_empty_result(self, mock_gc_class):
        """Returns empty list when no clone groups found."""
        from imas_codex.discovery.paths.parallel import _find_clone_groups

        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=None)
        mock_gc.query.return_value = []

        groups = _find_clone_groups("tcv")
        assert groups == []

    @patch("imas_codex.graph.GraphClient")
    def test_none_result(self, mock_gc_class):
        """Handles None result from graph client."""
        from imas_codex.discovery.paths.parallel import _find_clone_groups

        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=None)
        mock_gc.query.return_value = None

        groups = _find_clone_groups("tcv")
        assert groups == []


class TestMarkClonesTerminal:
    """Tests for _mark_clones_terminal graph mutation."""

    @patch("imas_codex.graph.GraphClient")
    def test_marks_clones_and_returns_count(self, mock_gc_class):
        """Marks clone paths terminal and returns count."""
        from imas_codex.discovery.paths.parallel import _mark_clones_terminal

        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=None)
        mock_gc.query.return_value = [{"marked": 3}]

        result = _mark_clones_terminal(
            ["tcv:/a", "tcv:/b", "tcv:/c"],
            canonical_path="/canonical/meq",
            repo_id="github.com/user/MEQ",
        )

        assert result == 3
        assert mock_gc.query.called
        # Verify the query uses UNWIND for batch operation
        query_str = mock_gc.query.call_args[0][0]
        assert "UNWIND" in query_str
        assert "terminal_reason" in query_str

    @patch("imas_codex.graph.GraphClient")
    def test_returns_zero_on_empty_result(self, mock_gc_class):
        """Returns 0 when graph returns empty result."""
        from imas_codex.discovery.paths.parallel import _mark_clones_terminal

        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=None)
        mock_gc.query.return_value = []

        result = _mark_clones_terminal(
            ["tcv:/a"],
            canonical_path="/canonical",
            repo_id="repo",
        )
        assert result == 0


class TestDedupWorker:
    """Tests for the dedup_worker async coroutine."""

    @pytest.mark.anyio
    @patch(
        "imas_codex.discovery.paths.parallel._mark_accessible_elsewhere",
        return_value=False,
    )
    @patch("imas_codex.discovery.paths.parallel._mark_clones_terminal", return_value=2)
    @patch("imas_codex.discovery.paths.parallel._find_clone_groups")
    async def test_processes_clone_groups(self, mock_find, mock_mark, mock_mark_ae):
        """Dedup worker processes clone groups and reports progress."""
        from imas_codex.discovery.paths.parallel import dedup_worker

        mock_find.side_effect = [
            [
                (
                    "github.com/user/MEQ",
                    "MEQ",
                    [
                        {
                            "id": "tcv:/canonical",
                            "path": "/canonical",
                            "accessible": False,
                        },
                        {"id": "tcv:/clone1", "path": "/clone1", "accessible": False},
                        {"id": "tcv:/clone2", "path": "/clone2", "accessible": False},
                    ],
                ),
            ],
            [],  # No more groups — triggers idle
        ]

        state = DiscoveryState(facility="tcv", cost_limit=10.0)
        progress_calls = []

        def on_progress(msg, stats, results=None):
            progress_calls.append((msg, stats.processed, results))
            # Stop after processing
            state.stop_requested = True

        await dedup_worker(state, on_progress=on_progress, batch_size=10)

        assert state.dedup_stats.processed == 2
        assert len(progress_calls) >= 1
        # First progress call should report deduped results
        msg, processed, results = progress_calls[0]
        assert "deduped" in msg
        assert results is not None
        assert results[0]["repo_name"] == "MEQ"
        assert results[0]["clones_marked"] == 2
        assert results[0]["canonical_skipped"] is False
        # Canonical is not accessible, so _mark_accessible_elsewhere not called
        mock_mark_ae.assert_not_called()

    @pytest.mark.anyio
    @patch(
        "imas_codex.discovery.paths.parallel._mark_accessible_elsewhere",
        return_value=True,
    )
    @patch("imas_codex.discovery.paths.parallel._mark_clones_terminal", return_value=1)
    @patch("imas_codex.discovery.paths.parallel._find_clone_groups")
    async def test_marks_canonical_accessible_elsewhere(
        self, mock_find, mock_mark, mock_mark_ae
    ):
        """When canonical is externally accessible, it's also marked terminal."""
        from imas_codex.discovery.paths.parallel import dedup_worker

        mock_find.side_effect = [
            [
                (
                    "github.com/user/MEQ",
                    "MEQ",
                    [
                        {
                            "id": "tcv:/canonical",
                            "path": "/canonical",
                            "accessible": True,
                        },
                        {"id": "tcv:/clone1", "path": "/clone1", "accessible": False},
                    ],
                ),
            ],
            [],
        ]

        state = DiscoveryState(facility="tcv", cost_limit=10.0)
        progress_calls = []

        def on_progress(msg, stats, results=None):
            progress_calls.append((msg, stats.processed, results))
            state.stop_requested = True

        await dedup_worker(state, on_progress=on_progress, batch_size=10)

        # 1 clone + 1 canonical = 2 total
        assert state.dedup_stats.processed == 2
        mock_mark_ae.assert_called_once_with("tcv:/canonical", "github.com/user/MEQ")
        msg, processed, results = progress_calls[0]
        assert results[0]["canonical_skipped"] is True

    @pytest.mark.anyio
    @patch("imas_codex.discovery.paths.parallel._find_clone_groups", return_value=[])
    async def test_idles_when_no_groups(self, mock_find):
        """Dedup worker idles and reports waiting when no clone groups."""
        from imas_codex.discovery.paths.parallel import dedup_worker

        state = DiscoveryState(facility="tcv", cost_limit=10.0)
        progress_calls = []

        def on_progress(msg, stats, results=None):
            progress_calls.append(msg)
            state.stop_requested = True

        await dedup_worker(state, on_progress=on_progress)

        assert len(progress_calls) >= 1
        assert "waiting" in progress_calls[0]
