"""Tests for the LLM session management module."""

import os
from unittest.mock import patch

import pytest

from imas_codex.agentic.session import (
    BudgetExhaustedError,
    CostTracker,
    LLMSession,
    create_session,
    estimate_cost,
)


class TestCostTracker:
    """Tests for CostTracker class."""

    def test_init_defaults(self):
        """Test default initialization."""
        tracker = CostTracker()
        assert tracker.limit_usd is None
        assert tracker.total_cost_usd == 0.0
        assert tracker.input_tokens == 0
        assert tracker.output_tokens == 0
        assert tracker.request_count == 0

    def test_init_with_limit(self):
        """Test initialization with cost limit."""
        tracker = CostTracker(limit_usd=10.0)
        assert tracker.limit_usd == 10.0
        assert not tracker.is_exhausted()

    def test_record_updates_counters(self):
        """Test that record() updates all counters."""
        tracker = CostTracker()
        cost = tracker.record(
            input_tokens=1000,
            output_tokens=500,
            model="google/gemini-3-flash-preview",
        )

        assert tracker.input_tokens == 1000
        assert tracker.output_tokens == 500
        assert tracker.request_count == 1
        assert tracker.total_cost_usd > 0
        assert cost > 0

    def test_is_exhausted_no_limit(self):
        """Test that unlimited tracker is never exhausted."""
        tracker = CostTracker()
        tracker.record(1_000_000, 1_000_000, "google/gemini-3-pro-preview")
        assert not tracker.is_exhausted()

    def test_is_exhausted_with_limit(self):
        """Test budget exhaustion detection."""
        tracker = CostTracker(limit_usd=0.001)  # Very low limit
        tracker.record(100_000, 100_000, "google/gemini-3-pro-preview")
        assert tracker.is_exhausted()

    def test_remaining_usd(self):
        """Test remaining budget calculation."""
        tracker = CostTracker(limit_usd=10.0)
        assert tracker.remaining_usd() == 10.0

        tracker.record(1000, 500, "google/gemini-3-flash-preview")
        assert tracker.remaining_usd() < 10.0
        assert tracker.remaining_usd() >= 0.0

    def test_remaining_usd_no_limit(self):
        """Test remaining is None when no limit."""
        tracker = CostTracker()
        assert tracker.remaining_usd() is None

    def test_summary(self):
        """Test summary output."""
        tracker = CostTracker(limit_usd=10.0)
        tracker.record(1000, 500, "google/gemini-3-flash-preview")

        summary = tracker.summary()
        assert "$" in summary
        assert "1 requests" in summary
        assert "1,000 in" in summary
        assert "limit: $10.00" in summary


class TestEstimateCost:
    """Tests for estimate_cost function."""

    def test_known_model(self):
        """Test cost estimation for known models."""
        # gemini-flash: $0.10/$0.40 per 1M tokens
        cost = estimate_cost(1_000_000, 1_000_000, "google/gemini-3-flash-preview")
        assert cost == pytest.approx(0.50, rel=0.01)  # $0.10 + $0.40

    def test_unknown_model_uses_default(self):
        """Test that unknown models use default pricing."""
        cost = estimate_cost(1_000_000, 1_000_000, "unknown/model-xyz")
        # Default is $1.00/$5.00 per 1M tokens
        assert cost == pytest.approx(6.00, rel=0.01)


class TestLLMSession:
    """Tests for LLMSession class."""

    def test_init_defaults(self):
        """Test default initialization."""
        session = LLMSession()
        assert session.task == "agent"
        assert session.model is None
        assert session.cost_limit_usd is None
        assert session.temperature == 0.3
        assert session.dry_run is False

    def test_resolved_model_uses_section_config(self):
        """Test that resolved_model returns section-based model from config."""
        session = LLMSession(task="language")
        # Should get model from pyproject.toml config
        assert session.resolved_model is not None
        assert len(session.resolved_model) > 0

    def test_resolved_model_explicit_override(self):
        """Test that explicit model overrides section config."""
        session = LLMSession(task="language", model="my-custom-model")
        assert session.resolved_model == "my-custom-model"

    def test_env_model_override(self):
        """Test that IMAS_CODEX_MODEL env var overrides config."""
        with patch.dict(os.environ, {"IMAS_CODEX_MODEL": "env-override-model"}):
            session = LLMSession(task="language")
            assert session.resolved_model == "env-override-model"

    def test_explicit_model_beats_env(self):
        """Test that explicit model parameter beats env var."""
        with patch.dict(os.environ, {"IMAS_CODEX_MODEL": "env-model"}):
            session = LLMSession(model="explicit-model")
            assert session.resolved_model == "explicit-model"

    def test_env_cost_limit(self):
        """Test that IMAS_CODEX_COST_LIMIT env var sets limit."""
        with patch.dict(os.environ, {"IMAS_CODEX_COST_LIMIT": "25.50"}):
            session = LLMSession()
            assert session.cost_limit_usd == 25.50
            assert session.cost_tracker.limit_usd == 25.50

    def test_explicit_cost_limit_beats_env(self):
        """Test that explicit cost_limit parameter beats env var."""
        with patch.dict(os.environ, {"IMAS_CODEX_COST_LIMIT": "100.0"}):
            session = LLMSession(cost_limit_usd=10.0)
            assert session.cost_limit_usd == 10.0

    def test_budget_exhausted(self):
        """Test budget_exhausted property."""
        session = LLMSession(cost_limit_usd=0.001)
        assert not session.budget_exhausted

        session.record_usage(100_000, 100_000)
        assert session.budget_exhausted

    def test_check_budget_raises(self):
        """Test check_budget raises when exhausted."""
        session = LLMSession(cost_limit_usd=0.001)
        session.record_usage(100_000, 100_000)

        with pytest.raises(BudgetExhaustedError) as exc_info:
            session.check_budget()

        assert "Budget exhausted" in str(exc_info.value)

    def test_record_usage(self):
        """Test recording usage updates tracker."""
        session = LLMSession()
        cost = session.record_usage(1000, 500)

        assert cost > 0
        assert session.cost_tracker.request_count == 1
        assert session.cost_tracker.input_tokens == 1000

    def test_estimate_cost(self):
        """Test cost estimation without recording."""
        session = LLMSession()
        initial_cost = session.cost_tracker.total_cost_usd

        estimate = session.estimate_cost(1000, 500)

        assert estimate > 0
        assert session.cost_tracker.total_cost_usd == initial_cost  # Not recorded

    def test_summary(self):
        """Test summary output."""
        session = LLMSession(task="language", cost_limit_usd=10.0)
        summary = session.summary()

        assert "Model:" in summary
        assert "Section: language" in summary
        assert "$" in summary

    def test_summary_dry_run(self):
        """Test summary includes DRY RUN indicator."""
        session = LLMSession(dry_run=True)
        summary = session.summary()

        assert "[DRY RUN]" in summary


class TestCreateSession:
    """Tests for create_session factory function."""

    def test_creates_session(self):
        """Test that factory creates session with correct parameters."""
        session = create_session(
            task="agent",
            model="test-model",
            cost_limit_usd=5.0,
            temperature=0.5,
            dry_run=True,
        )

        assert isinstance(session, LLMSession)
        assert session.task == "agent"
        assert session.model == "test-model"
        assert session.cost_limit_usd == 5.0
        assert session.temperature == 0.5
        assert session.dry_run is True

    def test_default_values(self):
        """Test factory with default values."""
        session = create_session()

        assert session.task == "agent"
        assert session.model is None
        assert session.cost_limit_usd is None
        assert session.temperature == 0.3
        assert session.dry_run is False


class TestBudgetExhaustedError:
    """Tests for BudgetExhaustedError exception."""

    def test_contains_summary(self):
        """Test that error contains cost summary."""
        error = BudgetExhaustedError("$5.00 spent (limit: $5.00)")
        assert "$5.00" in str(error)
        assert "$5.00" in error.summary
