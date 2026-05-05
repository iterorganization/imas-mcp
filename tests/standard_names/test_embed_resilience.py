"""Tests for embed-failure resilience in the SN pipeline.

Verifies that embedding failures do NOT quarantine names — they should
set ``embed_failed_at`` and allow the name to continue advancing through
the pipeline (review does not require vectors; only MCP search does).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(name: str, **overrides: Any) -> dict[str, Any]:
    """Build a minimal SN item as produced by the validate worker."""
    base: dict[str, Any] = {
        "id": name,
        "description": f"Description of {name}",
        "documentation": None,
        "kind": "scalar",
        "unit": "eV",
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
        # Enriched fields
        "enriched_description": f"The {name.replace('_', ' ')} in plasma.",
        "enriched_documentation": f"Detailed docs for {name}.",
        "enriched_links": [],
        "enriched_tags": ["spatial-profile"],
        # Validation results
        "validation_status": "valid",
        "validation_issues": [],
        # Enrichment provenance
        "llm_model": "test-enrich-model",
        "llm_cost": 0.005,
        "enrich_tokens": 500,
        # Embedding (populated by embed step)
        "embedding": [0.1] * 384,
    }
    base.update(overrides)
    return base


def _make_batch(items: list[dict], batch_index: int = 0) -> dict[str, Any]:
    return {
        "items": items,
        "claim_token": "test-token",
        "batch_index": batch_index,
    }


def _make_state(batches: list[dict], *, dry_run: bool = False) -> MagicMock:
    state = MagicMock()
    state.batches = batches
    state.stop_requested = False
    state.dry_run = dry_run
    state.persist_stats = MagicMock()
    state.persist_stats.total = 0
    state.persist_stats.processed = 0
    state.persist_stats.errors = 0
    state.persist_phase = MagicMock()
    state.stats = {}
    return state


# ---------------------------------------------------------------------------
# Tests: graph_ops.persist_generated_name_batch — embed failure handling
# ---------------------------------------------------------------------------


class TestEmbedFailureDoesNotQuarantine:
    """Embed failure sets embed_failed_at; never sets quarantine status."""

    def test_embed_failure_does_not_quarantine(self) -> None:
        """When embed fails, validation_status is not 'quarantined'."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        candidate = {
            "id": "electron_temperature",
            "model": "test-model",
            "validation_status": "valid",
            "validation_issues": [],
            "source_paths": [],
            "unit": "eV",
            "physics_domain": "equilibrium",
        }

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        def _failing_embed(*args, **kwargs) -> None:
            raise ConnectionError("Embedding server down")

        with (
            patch(
                "imas_codex.standard_names.graph_ops.GraphClient",
                return_value=mock_gc,
            ),
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                side_effect=_failing_embed,
            ),
        ):
            persist_generated_name_batch(
                [candidate],
                compose_model="test-model",
            )

        # Should NOT be quarantined
        assert candidate.get("validation_status") != "quarantined"
        # embed_failed_at should be set
        assert candidate.get("embed_failed_at") is not None

    def test_embed_success_sets_embedded_at(self) -> None:
        """When embed succeeds, embedded_at is set and embed_failed_at is absent."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        candidate = {
            "id": "electron_temperature",
            "model": "test-model",
            "validation_status": "valid",
            "validation_issues": [],
            "source_paths": [],
            "unit": "eV",
            "physics_domain": "equilibrium",
        }

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        def _successful_embed(*args, **kwargs) -> None:
            # embed_descriptions_batch signature: (candidates, text_field, ...)
            # It writes the embedding back into items in-place
            for arg in args:
                if isinstance(arg, list):
                    for item in arg:
                        if isinstance(item, dict):
                            item["embedding"] = [0.1] * 384

        with (
            patch(
                "imas_codex.standard_names.graph_ops.GraphClient",
                return_value=mock_gc,
            ),
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                side_effect=_successful_embed,
            ),
        ):
            persist_generated_name_batch(
                [candidate],
                compose_model="test-model",
            )

        # embedded_at should be set on success
        assert candidate.get("embedded_at") is not None
        # embed_failed_at should NOT be set
        assert candidate.get("embed_failed_at") is None


# ---------------------------------------------------------------------------
# Tests: enrich_workers.enrich_persist_worker — embed failure handling
# ---------------------------------------------------------------------------


class TestEnrichEmbedFailurePersistsAll:
    """Enrich persist worker persists all candidates even when embedding fails."""

    @pytest.mark.asyncio
    async def test_enrich_embed_failure_persists_all(self) -> None:
        """When embedding fails, ALL candidates are persisted (not just embeddable)."""
        from imas_codex.standard_names.enrich_workers import enrich_persist_worker

        items = [
            _make_item("electron_temperature", embedding=None),
            _make_item("ion_temperature", embedding=None),
        ]
        state = _make_state([_make_batch(items)])

        def _failing_embed(*args, **kwargs) -> None:
            raise ConnectionError("Embedding server down")

        persist_calls: list[list] = []

        def _mock_persist(candidates: list, **kwargs) -> int:
            persist_calls.append(list(candidates))
            return len(candidates)

        with (
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                side_effect=_failing_embed,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_enriched_batch",
                side_effect=_mock_persist,
            ),
            patch(
                "imas_codex.standard_names.enrich_workers.release_enrichment_claims",
            ),
        ):
            await enrich_persist_worker(state)

        # persist_enriched_batch must have been called
        assert len(persist_calls) == 1, "persist_enriched_batch should be called once"
        persisted_ids = {c["id"] for c in persist_calls[0]}
        assert "electron_temperature" in persisted_ids
        assert "ion_temperature" in persisted_ids
        assert len(persist_calls[0]) == 2, "All candidates should be persisted"

    @pytest.mark.asyncio
    async def test_embed_failure_sets_embed_failed_at_not_quarantine(self) -> None:
        """Embed failure sets embed_failed_at but does NOT set quarantine status."""
        from imas_codex.standard_names.enrich_workers import enrich_persist_worker

        item = _make_item("electron_temperature", embedding=None)
        state = _make_state([_make_batch([item])])

        def _failing_embed(*args, **kwargs) -> None:
            raise ConnectionError("Embedding server down")

        with (
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                side_effect=_failing_embed,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_enriched_batch",
                return_value=1,
            ),
            patch(
                "imas_codex.standard_names.enrich_workers.release_enrichment_claims",
            ),
        ):
            await enrich_persist_worker(state)

        # Must NOT be quarantined
        assert item.get("validation_status") != "quarantined"
        assert "embedding_failed" not in (item.get("validation_issues") or [])
        # Must have embed_failed_at set for retry tracking
        assert item.get("embed_failed_at") is not None

    @pytest.mark.asyncio
    async def test_embed_success_no_embed_failed_at(self) -> None:
        """Successful embedding does not set embed_failed_at."""
        from imas_codex.standard_names.enrich_workers import enrich_persist_worker

        item = _make_item("electron_temperature")  # has embedding=[0.1]*384
        state = _make_state([_make_batch([item])])

        def _noop_embed(candidates, src_field, dst_field) -> None:
            pass  # embedding already set on item

        with (
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                side_effect=_noop_embed,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_enriched_batch",
                return_value=1,
            ),
            patch(
                "imas_codex.standard_names.enrich_workers.release_enrichment_claims",
            ),
        ):
            await enrich_persist_worker(state)

        # embed_failed_at must NOT be set on success
        assert item.get("embed_failed_at") is None
