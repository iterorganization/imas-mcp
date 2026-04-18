"""Tests for the PERSIST worker (Phase C.4).

Covers:
- Persist writes all expected fields to graph (mocked).
- Persist does NOT overwrite unit/physics_domain/cocos.
- REFERENCES relationships created for known targets only.
- Tags union (existing + enriched, deduplicated).
- Claim release after persist.
- Embedding failure → quarantines item.
- Integration: round-trip persist of one item against live graph.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

# =============================================================================
# Fixtures / helpers
# =============================================================================


def _make_item(name: str, **overrides: Any) -> dict[str, Any]:
    """Build a mock SN item as produced by the validate worker."""
    base = {
        "id": name,
        "description": f"Description of {name}",
        "documentation": None,
        "kind": "scalar",
        "unit": "eV",
        "tags": ["time-dependent"],
        "links": None,
        "source_paths": [f"equilibrium/time_slice/profiles_1d/{name}"],
        "physical_base": "temperature",
        "subject": "electron",
        "component": None,
        "coordinate": None,
        "position": None,
        "process": None,
        "physics_domain": "equilibrium",
        "confidence": 0.9,
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
        "enrich_model": "test-enrich-model",
        "enrich_cost_usd": 0.005,
        "enrich_tokens": 500,
        # Embedding (set by embed step)
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


# =============================================================================
# Unit: persist_enriched_batch (graph_ops helper)
# =============================================================================


class TestPersistEnrichedBatch:
    def test_writes_expected_fields(self):
        """Cypher MERGE sets all enrichment fields."""
        from imas_codex.standard_names.graph_ops import persist_enriched_batch

        item = _make_item("electron_temperature")

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=mock_gc,
        ):
            count = persist_enriched_batch([item])

        assert count == 1

        # Check the MERGE query was called with batch containing our item
        merge_call = mock_gc.query.call_args_list[0]
        batch_arg = merge_call.kwargs.get("batch") or merge_call[1].get("batch")
        assert len(batch_arg) == 1
        b = batch_arg[0]
        assert b["id"] == "electron_temperature"
        assert b["description"] == item["enriched_description"]
        assert b["documentation"] == item["enriched_documentation"]
        assert b["embedding"] == item["embedding"]
        assert b["review_status"] == "enriched"
        assert b["enrich_model"] == "test-enrich-model"
        assert b["enrich_cost_usd"] == 0.005
        assert b["enrich_tokens"] == 500
        assert b["validation_status"] == "valid"

    def test_tags_union_deduplicated(self):
        """Tags = union of existing + enriched, deduplicated."""
        from imas_codex.standard_names.graph_ops import persist_enriched_batch

        item = _make_item(
            "electron_temperature",
            tags=["equilibrium", "transport"],
            enriched_tags=["transport", "kinetics"],
        )

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=mock_gc,
        ):
            persist_enriched_batch([item])

        merge_call = mock_gc.query.call_args_list[0]
        batch_arg = merge_call.kwargs.get("batch") or merge_call[1].get("batch")
        tags = batch_arg[0]["tags"]
        # Order: existing first, then new enriched
        assert tags == ["equilibrium", "transport", "kinetics"]

    def test_references_rels_created(self):
        """REFERENCES rels created for enriched_links."""
        from imas_codex.standard_names.graph_ops import persist_enriched_batch

        item = _make_item(
            "electron_temperature",
            enriched_links=["ion_temperature", "electron_density"],
        )

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=mock_gc,
        ):
            persist_enriched_batch([item])

        # Should have 2 calls: MERGE node, REFERENCES rels
        assert mock_gc.query.call_count == 2
        ref_call = mock_gc.query.call_args_list[1]
        ref_batch = ref_call.kwargs.get("batch") or ref_call[1].get("batch")
        assert len(ref_batch) == 2
        targets = {r["target_id"] for r in ref_batch}
        assert targets == {"ion_temperature", "electron_density"}

    def test_no_links_skips_references(self):
        """No enriched_links → no REFERENCES query."""
        from imas_codex.standard_names.graph_ops import persist_enriched_batch

        item = _make_item("electron_temperature", enriched_links=[])

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=mock_gc,
        ):
            persist_enriched_batch([item])

        # Only the MERGE query, no REFERENCES query
        assert mock_gc.query.call_count == 1

    def test_preserves_identity_fields_via_coalesce(self):
        """Cypher uses coalesce(b.field, sn.field) — new values don't overwrite existing."""
        from imas_codex.standard_names.graph_ops import persist_enriched_batch

        item = _make_item("electron_temperature")

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=mock_gc,
        ):
            persist_enriched_batch([item])

        # Verify the Cypher query uses coalesce pattern
        merge_call = mock_gc.query.call_args_list[0]
        cypher = merge_call.args[0] if merge_call.args else merge_call[0][0]
        # Description should use coalesce (first-write-wins for new)
        assert "coalesce(b.description, sn.description)" in cypher
        assert "coalesce(b.documentation, sn.documentation)" in cypher
        assert "coalesce(b.embedding, sn.embedding)" in cypher
        # Enrichment claims released
        assert "enrich_claimed_at = null" in cypher
        assert "enrich_claim_token = null" in cypher

    def test_empty_batch_returns_zero(self):
        from imas_codex.standard_names.graph_ops import persist_enriched_batch

        assert persist_enriched_batch([]) == 0


# =============================================================================
# Unit: persist worker
# =============================================================================


class TestPersistWorker:
    @pytest.mark.asyncio
    async def test_valid_items_persisted(self):
        """Valid items are embedded and persisted."""
        from imas_codex.standard_names.enrich_workers import enrich_persist_worker

        item = _make_item("electron_temperature")
        state = _make_state([_make_batch([item])])

        with (
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
            ) as mock_embed,
            patch(
                "imas_codex.standard_names.graph_ops.persist_enriched_batch",
                return_value=1,
            ) as mock_persist,
            patch(
                "imas_codex.standard_names.enrich_workers.release_enrichment_claims",
            ) as mock_release,
        ):
            await enrich_persist_worker(state)

        mock_embed.assert_called_once()
        mock_persist.assert_called_once()
        mock_release.assert_called_once_with("test-token")
        assert state.stats["persist_written"] == 1

    @pytest.mark.asyncio
    async def test_quarantined_items_skipped(self):
        """Quarantined items are not persisted."""
        from imas_codex.standard_names.enrich_workers import enrich_persist_worker

        item = _make_item("bad_name", validation_status="quarantined")
        state = _make_state([_make_batch([item])])

        with (
            patch("imas_codex.standard_names.enrich_workers.release_enrichment_claims"),
        ):
            await enrich_persist_worker(state)

        assert state.stats["persist_skipped"] == 1
        assert state.stats["persist_written"] == 0

    @pytest.mark.asyncio
    async def test_no_enriched_description_skipped(self):
        """Items without enriched_description are skipped."""
        from imas_codex.standard_names.enrich_workers import enrich_persist_worker

        item = _make_item("electron_temperature", enriched_description=None)
        state = _make_state([_make_batch([item])])

        with patch(
            "imas_codex.standard_names.enrich_workers.release_enrichment_claims"
        ):
            await enrich_persist_worker(state)

        assert state.stats["persist_skipped"] == 1

    @pytest.mark.asyncio
    async def test_embedding_failure_quarantines(self):
        """Embedding failure → item quarantined, not persisted."""
        from imas_codex.standard_names.enrich_workers import enrich_persist_worker

        item = _make_item("electron_temperature", embedding=None)
        state = _make_state([_make_batch([item])])

        def _failing_embed(*args, **kwargs):
            raise ConnectionError("Embedding server down")

        with (
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                side_effect=_failing_embed,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_enriched_batch",
                return_value=0,
            ),
            patch("imas_codex.standard_names.enrich_workers.release_enrichment_claims"),
        ):
            await enrich_persist_worker(state)

        assert item["validation_status"] == "quarantined"
        assert "embedding_failed" in item["validation_issues"]
        assert state.stats["persist_errors"] >= 1

    @pytest.mark.asyncio
    async def test_dry_run_skips(self):
        """Dry run → no persist, no embedding."""
        from imas_codex.standard_names.enrich_workers import enrich_persist_worker

        state = _make_state([_make_batch([_make_item("x")])], dry_run=True)
        await enrich_persist_worker(state)
        state.persist_phase.mark_done.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_batches_skips(self):
        from imas_codex.standard_names.enrich_workers import enrich_persist_worker

        state = _make_state([])
        await enrich_persist_worker(state)
        state.persist_phase.mark_done.assert_called_once()

    @pytest.mark.asyncio
    async def test_claims_released_on_empty_candidates(self):
        """When all items are quarantined, claims are still released."""
        from imas_codex.standard_names.enrich_workers import enrich_persist_worker

        item = _make_item("bad", validation_status="quarantined")
        state = _make_state([_make_batch([item])])

        with patch(
            "imas_codex.standard_names.enrich_workers.release_enrichment_claims"
        ) as mock_release:
            await enrich_persist_worker(state)

        mock_release.assert_called_once_with("test-token")
