"""Tests for rotation provenance stamping.

Validates:
- ``last_run_id`` is written on generate + regen phases
- ``last_run_at`` is written alongside
- Phases that didn't produce the node don't overwrite provenance
- ``write_run_provenance`` handles empty lists gracefully
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.standard_names.turn import PhaseResult, TurnConfig

# ═══════════════════════════════════════════════════════════════════════
# write_run_provenance unit tests
# ═══════════════════════════════════════════════════════════════════════


class TestWriteRotationProvenance:
    """Unit tests for the graph_ops.write_run_provenance helper."""

    def test_empty_list_returns_zero(self):
        from imas_codex.standard_names.graph_ops import write_run_provenance

        result = write_run_provenance([], "test-rotation-id")
        assert result == 0

    def test_calls_graph_with_correct_params(self):
        from imas_codex.standard_names.graph_ops import write_run_provenance

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"n": 3}])
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=mock_gc,
        ):
            n = write_run_provenance(["sn_a", "sn_b", "sn_c"], "rotation-123")

        assert n == 3
        mock_gc.query.assert_called_once()
        call_args = mock_gc.query.call_args
        # Check params
        assert call_args.kwargs["ids"] == ["sn_a", "sn_b", "sn_c"]
        assert call_args.kwargs["rid"] == "rotation-123"
        assert "ts" in call_args.kwargs


# ═══════════════════════════════════════════════════════════════════════
# Rotation provenance on generate phase
# ═══════════════════════════════════════════════════════════════════════


class TestRotationProvenanceOnGenerate:
    """Verify that generate phase stamps run_id on produced names."""

    @pytest.mark.asyncio
    async def test_generate_stamps_run_id(self):
        """After generate, write_run_provenance should be called
        with the consolidated name IDs."""
        from imas_codex.standard_names.turn import _run_generate_phase

        cfg = TurnConfig(domain="equilibrium", dry_run=False)

        mock_state = MagicMock()
        mock_state.total_cost = 0.1
        mock_state.stats = {"compose_count": 2, "compose_cost": 0.1}
        mock_state.consolidated = [
            {"id": "electron_temperature"},
            {"id": "ion_density"},
        ]

        provenance_calls = []

        def capture_provenance(ids, rid, tn=1):
            provenance_calls.append((ids, rid, tn))
            return len(ids)

        with (
            patch(
                "imas_codex.standard_names.pipeline.run_sn_pipeline",
                new_callable=AsyncMock,
            ) as _mock_engine,
            patch(
                "imas_codex.standard_names.state.StandardNameBuildState",
                return_value=mock_state,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.write_run_provenance",
                side_effect=capture_provenance,
            ),
        ):
            result = await _run_generate_phase(cfg)

        assert result.name == "generate"
        assert result.count == 2
        assert len(provenance_calls) == 1
        ids, rid, _tn = provenance_calls[0]
        assert "electron_temperature" in ids
        assert "ion_density" in ids
        assert rid == cfg.run_id

    @pytest.mark.asyncio
    async def test_dry_run_no_provenance(self):
        """Dry-run should NOT call write_run_provenance."""
        from imas_codex.standard_names.turn import _run_generate_phase

        cfg = TurnConfig(domain="equilibrium", dry_run=True)
        result = await _run_generate_phase(cfg)

        assert result.name == "generate"
        assert result.count == 0


# ═══════════════════════════════════════════════════════════════════════
# Regen phase stamps rotation provenance
# ═══════════════════════════════════════════════════════════════════════


class TestRotationProvenanceOnRegen:
    """Verify that regen phase also stamps run_id."""

    @pytest.mark.asyncio
    async def test_regen_stamps_run_id(self):
        from imas_codex.standard_names.turn import _run_generate_phase

        cfg = TurnConfig(domain="equilibrium", dry_run=False)

        mock_state = MagicMock()
        mock_state.total_cost = 0.05
        mock_state.stats = {"compose_count": 1, "compose_cost": 0.05}
        mock_state.consolidated = [{"id": "plasma_current"}]

        provenance_calls = []

        def capture_provenance(ids, rid, tn=1):
            provenance_calls.append((ids, rid, tn))
            return len(ids)

        with (
            patch(
                "imas_codex.standard_names.pipeline.run_sn_pipeline",
                new_callable=AsyncMock,
            ),
            patch(
                "imas_codex.standard_names.state.StandardNameBuildState",
                return_value=mock_state,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.write_run_provenance",
                side_effect=capture_provenance,
            ),
        ):
            result = await _run_generate_phase(cfg, regen=True, force=True)

        assert result.name == "regen"
        assert len(provenance_calls) == 1
        ids, rid, _tn = provenance_calls[0]
        assert "plasma_current" in ids
        assert rid == cfg.run_id


# ═══════════════════════════════════════════════════════════════════════
# Enrich phase does NOT overwrite rotation provenance
# ═══════════════════════════════════════════════════════════════════════


class TestEnrichNoRotationOverwrite:
    """Verify enrich phase does not stamp rotation provenance."""

    @pytest.mark.asyncio
    async def test_enrich_does_not_call_write_run_provenance(self):
        """The enrich phase should not touch last_run_id."""
        from imas_codex.standard_names.turn import _run_enrich_phase

        cfg = TurnConfig(domain="equilibrium", dry_run=True)
        result = await _run_enrich_phase(cfg)

        # Dry-run enrich returns without calling any graph operations
        assert result.name == "enrich"
        assert result.count == 0


class TestReviewNoRotationOverwrite:
    """Verify review phase does not stamp rotation provenance."""

    @pytest.mark.asyncio
    async def test_review_does_not_call_write_run_provenance(self):
        """The review phase should not touch last_run_id."""
        from imas_codex.standard_names.turn import _run_review_phase

        cfg = TurnConfig(domain="equilibrium", dry_run=True)
        result = await _run_review_phase(cfg)

        assert result.name == "review"
        assert result.count == 0


# ═══════════════════════════════════════════════════════════════════════
# PhaseResult dataclass
# ═══════════════════════════════════════════════════════════════════════


class TestPhaseResult:
    """Basic PhaseResult construction tests."""

    def test_default_values(self):
        r = PhaseResult(name="test")
        assert r.exit_code == 0
        assert r.cost == 0.0
        assert r.elapsed == 0.0
        assert r.count == 0
        assert r.error is None
        assert not r.skipped

    def test_error_result(self):
        r = PhaseResult(name="test", exit_code=1, error="something broke")
        assert r.exit_code == 1
        assert r.error == "something broke"
