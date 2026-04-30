"""Smoke tests for the enrich pipeline skeleton (C.1).

Verifies that:
1. Extract worker queries graph and produces batches from named SNs.
2. Stub workers (contextualise, document, validate, persist) log TODO
   but don't crash.
3. No graph writes happen in dry_run mode.
4. Pipeline wires up end-to-end through run_discovery_engine.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Fixtures
# =============================================================================


def _make_mock_sn(name: str, **overrides: Any) -> dict[str, Any]:
    """Build a mock StandardName dict as returned by the claim query."""
    base = {
        "id": name,
        "description": f"Description of {name}",
        "documentation": None,
        "kind": "scalar",
        "unit": "eV",
        "tags": ["equilibrium"],
        "links": None,
        "source_paths": [f"equilibrium/time_slice/profiles_1d/{name}"],
        "physical_base": "temperature",
        "subject": "electron",
        "component": None,
        "coordinate": None,
        "position": None,
        "process": None,
        "physics_domain": "equilibrium",
        "model": "test-model",
    }
    base.update(overrides)
    return base


MOCK_SNS = [
    _make_mock_sn("electron_temperature"),
    _make_mock_sn("ion_temperature"),
    _make_mock_sn("electron_density", unit="m^-3", physical_base="density"),
]


# =============================================================================
# State construction
# =============================================================================


class TestEnrichState:
    """Test StandardNameEnrichState dataclass."""

    def test_state_construction(self) -> None:
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState

        state = StandardNameEnrichState(
            facility="dd",
            domain="equilibrium",
            cost_limit=2.0,
            dry_run=True,
        )
        assert state.domain == "equilibrium"
        assert state.dry_run is True
        assert state.cost_limit == 2.0
        assert state.total_cost == 0.0
        assert not state.should_stop()

    def test_state_phases_exist(self) -> None:
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState

        state = StandardNameEnrichState(facility="dd")
        assert hasattr(state, "extract_phase")
        assert hasattr(state, "contextualise_phase")
        assert hasattr(state, "document_phase")
        assert hasattr(state, "validate_phase")
        assert hasattr(state, "persist_phase")

    def test_budget_exhaustion(self) -> None:
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState

        state = StandardNameEnrichState(facility="dd", cost_limit=1.0)
        state.cost = 1.5
        assert state.budget_exhausted
        assert state.should_stop()


# =============================================================================
# Extract worker
# =============================================================================


class TestExtractWorker:
    """Test the extract worker claim + batch logic."""

    def test_extract_builds_batches(self) -> None:
        """Extract worker should batch claimed SNs into groups."""
        from imas_codex.standard_names.enrich_workers import _build_batches

        batches = _build_batches(MOCK_SNS, batch_size=10, token="test-token")
        assert len(batches) == 1
        assert len(batches[0]["items"]) == 3
        assert batches[0]["claim_token"] == "test-token"

    def test_extract_splits_large_batches(self) -> None:
        """Extract worker should split items exceeding batch_size."""
        from imas_codex.standard_names.enrich_workers import _build_batches

        items = [_make_mock_sn(f"name_{i}") for i in range(25)]
        batches = _build_batches(items, batch_size=10, token="tok")
        assert len(batches) == 3
        assert len(batches[0]["items"]) == 10
        assert len(batches[1]["items"]) == 10
        assert len(batches[2]["items"]) == 5

    def test_extract_empty(self) -> None:
        """Empty item list → no batches."""
        from imas_codex.standard_names.enrich_workers import _build_batches

        assert _build_batches([]) == []

    @pytest.mark.asyncio
    async def test_extract_dry_run(self) -> None:
        """Dry-run: claims, releases immediately, populates batches."""
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState
        from imas_codex.standard_names.enrich_workers import enrich_extract_worker

        state = StandardNameEnrichState(facility="dd", dry_run=True, cost_limit=2.0)

        token = "mock-token-123"
        with (
            patch(
                "imas_codex.standard_names.enrich_workers.claim_names_for_enrichment",
                return_value=(token, list(MOCK_SNS)),
            ) as mock_claim,
            patch(
                "imas_codex.standard_names.enrich_workers.release_enrichment_claims",
                return_value=3,
            ) as mock_release,
        ):
            await enrich_extract_worker(state)

        # Claim was called
        mock_claim.assert_called_once()
        # Claims released in dry_run
        mock_release.assert_called_once_with(token)

        # Batches populated
        assert len(state.batches) == 1
        assert len(state.batches[0]["items"]) == 3
        assert state.stats["extract_count"] == 3
        assert state.stats["extract_batches"] == 1
        assert state.extract_phase.done

    @pytest.mark.asyncio
    async def test_extract_live_mode(self) -> None:
        """Live mode: claims but does NOT release."""
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState
        from imas_codex.standard_names.enrich_workers import enrich_extract_worker

        state = StandardNameEnrichState(facility="dd", dry_run=False, cost_limit=2.0)

        with (
            patch(
                "imas_codex.standard_names.enrich_workers.claim_names_for_enrichment",
                return_value=("live-token", list(MOCK_SNS)),
            ),
            patch(
                "imas_codex.standard_names.enrich_workers.release_enrichment_claims",
            ) as mock_release,
        ):
            await enrich_extract_worker(state)

        # Claims NOT released in live mode
        mock_release.assert_not_called()
        assert len(state.batches) == 1
        assert state.extract_phase.done

    @pytest.mark.asyncio
    async def test_extract_no_items(self) -> None:
        """No named SNs → empty batches, phase still done."""
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState
        from imas_codex.standard_names.enrich_workers import enrich_extract_worker

        state = StandardNameEnrichState(facility="dd", dry_run=False, cost_limit=2.0)

        with patch(
            "imas_codex.standard_names.enrich_workers.claim_names_for_enrichment",
            return_value=("empty-token", []),
        ):
            await enrich_extract_worker(state)

        assert state.batches == []
        assert state.extract_phase.done


# =============================================================================
# Stub workers
# =============================================================================


class TestStubWorkers:
    """Verify stub workers log TODO and complete without error."""

    @pytest.mark.asyncio
    async def test_contextualise_stub(self, caplog) -> None:
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState
        from imas_codex.standard_names.enrich_workers import (
            enrich_contextualise_worker,
        )

        state = StandardNameEnrichState(facility="dd")
        state.batches = [{"items": MOCK_SNS, "claim_token": None, "batch_index": 0}]

        await enrich_contextualise_worker(state)

        assert state.contextualise_phase.done
        assert state.contextualise_stats.processed == 3

    @pytest.mark.asyncio
    async def test_document_stub(self, caplog) -> None:
        pytest.skip(
            "enrich_document_worker is now fully implemented and requires a mocked "
            "acall_llm_structured; add mock_llm fixture to cover this worker properly"
        )

    @pytest.mark.asyncio
    async def test_validate_stub(self, caplog) -> None:
        pytest.skip(
            "enrich_validate_worker skips items without enriched_description; "
            "this stub test needs items pre-populated by enrich_document_worker — "
            "covered by TestRoundTripMocked in test_enrich_integration.py"
        )

    @pytest.mark.asyncio
    async def test_persist_stub_dry_run(self) -> None:
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState
        from imas_codex.standard_names.enrich_workers import enrich_persist_worker

        state = StandardNameEnrichState(facility="dd", dry_run=True)
        state.batches = [{"items": MOCK_SNS, "claim_token": None, "batch_index": 0}]

        await enrich_persist_worker(state)

        assert state.persist_phase.done

    @pytest.mark.asyncio
    async def test_persist_stub_releases_claims(self) -> None:
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState
        from imas_codex.standard_names.enrich_workers import enrich_persist_worker

        state = StandardNameEnrichState(facility="dd", dry_run=False)
        state.batches = [
            {"items": MOCK_SNS, "claim_token": "test-tok", "batch_index": 0}
        ]

        with patch(
            "imas_codex.standard_names.enrich_workers.release_enrichment_claims",
            return_value=3,
        ) as mock_release:
            await enrich_persist_worker(state)

        mock_release.assert_called_once_with("test-tok")
        assert state.persist_phase.done

    @pytest.mark.asyncio
    async def test_stubs_empty_batches(self) -> None:
        """All stubs handle empty batches gracefully."""
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState
        from imas_codex.standard_names.enrich_workers import (
            enrich_contextualise_worker,
            enrich_document_worker,
            enrich_persist_worker,
            enrich_validate_worker,
        )

        state = StandardNameEnrichState(facility="dd")
        state.batches = []

        await enrich_contextualise_worker(state)
        await enrich_document_worker(state)
        await enrich_validate_worker(state)

        with patch(
            "imas_codex.standard_names.enrich_workers.release_enrichment_claims"
        ):
            await enrich_persist_worker(state)

        assert state.contextualise_phase.done
        assert state.document_phase.done
        assert state.validate_phase.done
        assert state.persist_phase.done


# =============================================================================
# End-to-end pipeline smoke test
# =============================================================================


class TestEnrichPipelineE2E:
    """End-to-end smoke test for the full enrich pipeline."""

    @pytest.mark.asyncio
    async def test_dry_run_pipeline(self) -> None:
        """Run the full pipeline in dry_run mode: no graph writes."""
        from imas_codex.standard_names.enrich_pipeline import run_sn_enrich_engine
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState

        state = StandardNameEnrichState(
            facility="dd",
            domain="equilibrium",
            cost_limit=2.0,
            dry_run=True,
        )

        with (
            patch(
                "imas_codex.standard_names.enrich_workers.claim_names_for_enrichment",
                return_value=("e2e-token", list(MOCK_SNS)),
            ),
            patch(
                "imas_codex.standard_names.enrich_workers.release_enrichment_claims",
                return_value=3,
            ) as mock_release,
        ):
            await run_sn_enrich_engine(state)

        # Extract produced batches
        assert len(state.batches) == 1
        assert state.stats["extract_count"] == 3

        # All phases completed
        assert state.extract_phase.done
        assert state.contextualise_phase.done
        assert state.document_phase.done
        assert state.validate_phase.done
        assert state.persist_phase.done

        # Dry run releases claims (from extract) but persist also skips
        mock_release.assert_called()

    @pytest.mark.asyncio
    async def test_pipeline_no_items(self) -> None:
        """Pipeline with no named SNs completes gracefully."""
        from imas_codex.standard_names.enrich_pipeline import run_sn_enrich_engine
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState

        state = StandardNameEnrichState(
            facility="dd",
            cost_limit=2.0,
            dry_run=False,
        )

        with patch(
            "imas_codex.standard_names.enrich_workers.claim_names_for_enrichment",
            return_value=("empty-token", []),
        ):
            await run_sn_enrich_engine(state)

        assert state.batches == []
        assert state.extract_phase.done
        # Downstream phases also complete (no work to do)
        assert state.contextualise_phase.done
        assert state.document_phase.done
        assert state.validate_phase.done
        assert state.persist_phase.done
