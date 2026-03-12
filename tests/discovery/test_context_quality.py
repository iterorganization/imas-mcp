"""Tests for context quality assessment and underspecified signal handling.

Verifies:
- ContextQuality enum in enrichment models
- context_quality field on SignalEnrichmentResult
- Underspecified signal routing in enrich_worker result handler
- mark_signals_underspecified graph persistence
- claim_signals_for_enrichment picks up underspecified signals
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.discovery.signals.models import (
    ContextQuality,
    SignalEnrichmentBatch,
    SignalEnrichmentResult,
)
from imas_codex.graph.models import FacilitySignalStatus

# =============================================================================
# ContextQuality enum
# =============================================================================


class TestContextQualityEnum:
    """Test ContextQuality enum values and behaviour."""

    def test_enum_values(self):
        assert ContextQuality.low == "low"
        assert ContextQuality.medium == "medium"
        assert ContextQuality.high == "high"

    def test_all_values(self):
        assert set(ContextQuality) == {
            ContextQuality.low,
            ContextQuality.medium,
            ContextQuality.high,
        }


# =============================================================================
# SignalEnrichmentResult context_quality field
# =============================================================================


class TestSignalEnrichmentResultContextQuality:
    """Test context_quality field on SignalEnrichmentResult."""

    def test_default_is_medium(self):
        result = SignalEnrichmentResult(
            signal_index=1,
            physics_domain="magnetic_field_diagnostics",
            name="Test Signal",
            description="A test signal.",
        )
        assert result.context_quality == ContextQuality.medium

    def test_can_set_low(self):
        result = SignalEnrichmentResult(
            signal_index=1,
            physics_domain="magnetic_field_diagnostics",
            name="Test Signal",
            description="A test signal.",
            context_quality="low",
        )
        assert result.context_quality == ContextQuality.low

    def test_can_set_high(self):
        result = SignalEnrichmentResult(
            signal_index=1,
            physics_domain="magnetic_field_diagnostics",
            name="Test Signal",
            description="A test signal with rich context.",
            context_quality="high",
        )
        assert result.context_quality == ContextQuality.high

    def test_serialization_roundtrip(self):
        """context_quality survives JSON serialization."""
        result = SignalEnrichmentResult(
            signal_index=1,
            physics_domain="magnetic_field_diagnostics",
            name="Test Signal",
            description="A test signal.",
            context_quality="low",
        )
        data = result.model_dump()
        assert data["context_quality"] == "low"
        restored = SignalEnrichmentResult(**data)
        assert restored.context_quality == ContextQuality.low

    def test_batch_with_mixed_quality(self):
        """A batch can contain mixed context quality levels."""
        batch = SignalEnrichmentBatch(
            results=[
                SignalEnrichmentResult(
                    signal_index=1,
                    physics_domain="magnetic_field_diagnostics",
                    name="IP",
                    description="Plasma current",
                    context_quality="high",
                ),
                SignalEnrichmentResult(
                    signal_index=2,
                    physics_domain="general",
                    name="Unknown",
                    description="Static parameter node.",
                    context_quality="low",
                ),
            ]
        )
        assert batch.results[0].context_quality == ContextQuality.high
        assert batch.results[1].context_quality == ContextQuality.low


# =============================================================================
# FacilitySignalStatus.underspecified
# =============================================================================


class TestUnderspecifiedStatus:
    """Test the underspecified status enum value exists and works."""

    def test_underspecified_exists(self):
        assert hasattr(FacilitySignalStatus, "underspecified")
        assert FacilitySignalStatus.underspecified.value == "underspecified"

    def test_underspecified_in_status_values(self):
        values = [s.value for s in FacilitySignalStatus]
        assert "underspecified" in values


# =============================================================================
# mark_signals_underspecified
# =============================================================================


class TestMarkSignalsUnderspecified:
    """Test mark_signals_underspecified graph persistence."""

    def test_empty_list_returns_zero(self):
        from imas_codex.discovery.signals.parallel import mark_signals_underspecified

        assert mark_signals_underspecified([], 0.0) == 0

    def test_calls_graph_with_underspecified_status(self):
        from imas_codex.discovery.signals.parallel import mark_signals_underspecified

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        signals = [
            {
                "id": "tcv:static/dbrdr_ex_r:val",
                "physics_domain": "general",
                "description": "Static parameter node.",
                "name": "DBRDR_EX_R:VAL",
                "diagnostic": "",
                "analysis_code": "",
                "keywords": [],
                "sign_convention": "",
            }
        ]

        with patch(
            "imas_codex.discovery.signals.parallel.GraphClient",
            return_value=mock_gc,
        ):
            count = mark_signals_underspecified(signals, 0.01)

        assert count == 1
        # Verify the query used underspecified status
        call_args = mock_gc.query.call_args
        assert call_args.kwargs["status"] == "underspecified"

    def test_graph_error_returns_zero(self):
        from imas_codex.discovery.signals.parallel import mark_signals_underspecified

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)
        mock_gc.query.side_effect = Exception("connection lost")

        signals = [
            {
                "id": "tcv:test",
                "physics_domain": "general",
                "description": "Test.",
                "name": "Test",
                "diagnostic": "",
                "analysis_code": "",
                "keywords": [],
                "sign_convention": "",
            }
        ]

        with patch(
            "imas_codex.discovery.signals.parallel.GraphClient",
            return_value=mock_gc,
        ):
            count = mark_signals_underspecified(signals, 0.01)

        assert count == 0


# =============================================================================
# Claim query includes underspecified
# =============================================================================


class TestClaimIncludesUnderspecified:
    """Test that claim_signals_for_enrichment picks up underspecified signals."""

    def test_claim_query_includes_underspecified_status(self):
        """The claim query should match both discovered and underspecified signals."""
        from imas_codex.discovery.signals.parallel import claim_signals_for_enrichment

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = []

        with patch(
            "imas_codex.discovery.signals.parallel.GraphClient",
            return_value=mock_gc,
        ):
            claim_signals_for_enrichment("tcv", batch_size=5)

        # The first query is the channel skip, the second is the claim
        claim_call = mock_gc.query.call_args_list[1]
        # Verify underspecified parameter is passed
        assert claim_call.kwargs.get("underspecified") == "underspecified"
        # Verify discovered parameter is also passed
        assert claim_call.kwargs.get("discovered") == "discovered"
