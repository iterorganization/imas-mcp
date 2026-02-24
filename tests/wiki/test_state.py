"""Tests for wiki discovery state management.

Focuses on should_stop() termination logic, particularly the interaction
between budget exhaustion and worker idle counts.
"""

from unittest.mock import patch

import pytest

from imas_codex.discovery.wiki.state import WikiDiscoveryState


@pytest.fixture
def state():
    """Create a WikiDiscoveryState with defaults for testing."""
    return WikiDiscoveryState(
        facility="test",
        site_type="mediawiki",
        base_url="https://example.com/wiki",
        portal_page="Main_Page",
        cost_limit=2.0,
    )


def _set_all_idle(state: WikiDiscoveryState, count: int = 3) -> None:
    """Set all worker idle counts to the given value."""
    state.scan_idle_count = count
    state.score_idle_count = count
    state.ingest_idle_count = count
    state.docs_idle_count = count
    state.artifact_score_idle_count = count
    state.image_idle_count = count


def _no_pending_work(*args, **kwargs):
    return False


class TestShouldStop:
    """Tests for WikiDiscoveryState.should_stop()."""

    @patch("imas_codex.discovery.wiki.state._get_graph_ops")
    def test_stop_when_all_idle_no_pending(self, mock_graph_ops, state):
        """All idle + no pending work -> should stop."""
        _set_all_idle(state)
        mock_ops = mock_graph_ops.return_value
        mock_ops.has_pending_work.return_value = False
        mock_ops.has_pending_artifact_work.return_value = False
        mock_ops.has_pending_image_work.return_value = False

        assert state.should_stop() is True

    @patch("imas_codex.discovery.wiki.state._get_graph_ops")
    def test_no_stop_when_score_not_idle(self, mock_graph_ops, state):
        """Score worker not idle -> should not stop."""
        _set_all_idle(state)
        state.score_idle_count = 0  # Score still active

        assert state.should_stop() is False

    def test_stop_requested(self, state):
        """stop_requested flag immediately terminates."""
        state.stop_requested = True
        assert state.should_stop() is True

    @patch("imas_codex.discovery.wiki.state._get_graph_ops")
    def test_budget_exhausted_llm_idle_bypass(self, mock_graph_ops, state):
        """When budget exhausted, LLM workers with idle_count=0 count as done.

        This is the core bug fix: LLM workers exit their loops when budget
        hits, leaving idle_count at 0. should_stop() must still return True
        once I/O workers are also done.
        """
        # Simulate: budget exhausted, LLM workers exited (idle=0),
        # I/O workers drained queues (idle=3), scan idle (3)
        state.score_stats.cost = 2.50  # Exceeds cost_limit=2.0
        state.score_idle_count = 0  # Exited due to budget, never incremented
        state.artifact_score_idle_count = 0  # Same
        state.image_idle_count = 0  # Same

        state.scan_idle_count = 3
        state.ingest_idle_count = 3
        state.docs_idle_count = 3

        assert state.budget_exhausted is True

        # No pending I/O work
        mock_ops = mock_graph_ops.return_value
        mock_ops.has_pending_ingest_work.return_value = False
        mock_ops.has_pending_artifact_ingest_work.return_value = False

        assert state.should_stop() is True

    @patch("imas_codex.discovery.wiki.state._get_graph_ops")
    def test_budget_exhausted_io_work_pending(self, mock_graph_ops, state):
        """Budget exhausted but I/O ingest work pending -> don't stop yet."""
        state.score_stats.cost = 2.50
        state.score_idle_count = 0
        state.artifact_score_idle_count = 0
        state.image_idle_count = 0

        state.scan_idle_count = 3
        state.ingest_idle_count = 3
        state.docs_idle_count = 3

        # Pending ingest work (scored pages waiting for embedding)
        mock_ops = mock_graph_ops.return_value
        mock_ops.has_pending_ingest_work.return_value = True
        mock_ops.has_pending_artifact_ingest_work.return_value = False

        assert state.should_stop() is False
        # I/O idle counts should be reset
        assert state.ingest_idle_count == 0
        assert state.docs_idle_count == 0
        # LLM idle counts should NOT be reset (they've exited)
        assert state.score_idle_count == 0
        assert state.artifact_score_idle_count == 0
        assert state.image_idle_count == 0

    @patch("imas_codex.discovery.wiki.state._get_graph_ops")
    def test_budget_exhausted_ignores_llm_pending_work(self, mock_graph_ops, state):
        """Budget exhausted: pending scoring work should be ignored.

        12K scanned pages awaiting LLM scoring shouldn't prevent termination
        when the cost limit means scoring can't happen.
        """
        state.score_stats.cost = 2.50
        state.score_idle_count = 0
        state.artifact_score_idle_count = 0
        state.image_idle_count = 0

        state.scan_idle_count = 3
        state.ingest_idle_count = 3
        state.docs_idle_count = 3

        # No I/O work pending — only LLM work pending
        mock_ops = mock_graph_ops.return_value
        mock_ops.has_pending_ingest_work.return_value = False
        mock_ops.has_pending_artifact_ingest_work.return_value = False
        # These should NOT be called when budget is exhausted:
        mock_ops.has_pending_work.return_value = True  # 12K scanned pages
        mock_ops.has_pending_image_work.return_value = True  # VLM pending

        assert state.should_stop() is True
        # Verify LLM pending work checks were NOT called
        mock_ops.has_pending_work.assert_not_called()
        mock_ops.has_pending_image_work.assert_not_called()
        mock_ops.has_pending_artifact_work.assert_not_called()

    @patch("imas_codex.discovery.wiki.state._get_graph_ops")
    def test_no_budget_resets_all_idle_on_pending(self, mock_graph_ops, state):
        """Without budget exhaustion, all idle counts reset on pending work."""
        _set_all_idle(state)
        # Budget NOT exhausted
        assert state.budget_exhausted is False

        mock_ops = mock_graph_ops.return_value
        mock_ops.has_pending_work.return_value = True
        mock_ops.has_pending_artifact_work.return_value = False
        mock_ops.has_pending_image_work.return_value = False

        assert state.should_stop() is False
        # ALL idle counts should be reset
        assert state.scan_idle_count == 0
        assert state.score_idle_count == 0
        assert state.ingest_idle_count == 0
        assert state.docs_idle_count == 0
        assert state.artifact_score_idle_count == 0
        assert state.image_idle_count == 0


class TestShouldStopScoring:
    """Tests for score worker stop conditions."""

    def test_stops_on_budget(self, state):
        state.score_stats.cost = 2.50  # Over limit
        assert state.should_stop_scoring() is True

    def test_stops_on_page_limit(self, state):
        state.page_limit = 100
        state.score_stats.processed = 100
        assert state.should_stop_scoring() is True

    def test_continues_under_budget(self, state):
        state.score_stats.cost = 1.0  # Under limit
        assert state.should_stop_scoring() is False

    def test_stops_on_request(self, state):
        state.stop_requested = True
        assert state.should_stop_scoring() is True


class TestShouldStopIngesting:
    """Tests for ingest worker stop conditions."""

    @patch("imas_codex.discovery.wiki.state._get_graph_ops")
    def test_continues_after_budget(self, mock_graph_ops, state):
        """Ingest workers should continue even after budget exhaustion."""
        state.score_stats.cost = 2.50  # Budget exhausted
        state.ingest_idle_count = 0  # Still working
        assert state.should_stop_ingesting() is False

    @patch("imas_codex.discovery.wiki.state._get_graph_ops")
    def test_stops_when_scoring_done_and_no_work(self, mock_graph_ops, state):
        """Ingest stops when scoring done + no pending ingest work."""
        state.score_stats.cost = 2.50  # Budget exhausted
        state.score_idle_count = 0
        state.ingest_idle_count = 3

        mock_ops = mock_graph_ops.return_value
        mock_ops.has_pending_ingest_work.return_value = False

        assert state.should_stop_ingesting() is True


class TestShouldStopImageScoring:
    """Tests for image score worker stop conditions."""

    def test_stops_on_budget(self, state):
        state.score_stats.cost = 2.50  # Budget exhausted
        assert state.should_stop_image_scoring() is True


class TestBudgetExhausted:
    """Tests for the budget_exhausted property."""

    def test_under_budget(self, state):
        state.score_stats.cost = 1.0
        assert state.budget_exhausted is False

    def test_over_budget(self, state):
        state.score_stats.cost = 2.50
        assert state.budget_exhausted is True

    def test_at_exact_limit(self, state):
        state.score_stats.cost = 2.0
        assert state.budget_exhausted is True

    def test_includes_all_cost_sources(self, state):
        """Budget accounts for score + ingest + image + artifact_score costs."""
        state.score_stats.cost = 0.5
        state.ingest_stats.cost = 0.5
        state.image_stats.cost = 0.5
        state.artifact_score_stats.cost = 0.5
        assert state.total_cost == 2.0
        assert state.budget_exhausted is True
