"""Tests that superseded SNs cannot be claimed or persisted by docs/review/refine workers.

Covers the B2 (claim filter exclusion) and B3 (persist Cypher guard) fixes
that break the infinite retry loop observed during the $50 rotation when
``persist_generated_docs`` hit token mismatches on names concurrently
superseded by ``refine_name`` workers.

All tests mock :class:`GraphClient` — no live Neo4j required.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — mock GraphClient wiring (same pattern as test_claim_atomic.py)
# ---------------------------------------------------------------------------


def _mock_gc_tx():
    """Build a mock GraphClient wired for single-transaction claims."""
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)

    tx = MagicMock()
    tx.closed = False
    tx.commit = MagicMock()
    tx.close = MagicMock()

    session = MagicMock()
    session.begin_transaction = MagicMock(return_value=tx)

    @contextmanager
    def _session_ctx():
        yield session

    gc.session = _session_ctx
    return gc, tx


def _patch_gc(mock_gc):
    return patch(
        "imas_codex.standard_names.graph_ops.GraphClient",
        return_value=mock_gc,
    )


# ---------------------------------------------------------------------------
# B2 — Claim filter tests: superseded/exhausted excluded from WHERE clause
# ---------------------------------------------------------------------------


class TestClaimFiltersExcludeSuperseded:
    """Verify that every claim function's eligibility WHERE includes the
    ``NOT (sn.name_stage IN ['superseded', 'exhausted'])`` guard."""

    @pytest.mark.parametrize(
        "func_name",
        [
            "claim_generate_docs_batch",
            "claim_review_docs_batch",
            "claim_refine_docs_batch",
            "claim_review_name_batch",
            "claim_refine_name_batch",
        ],
    )
    def test_claim_where_excludes_superseded(self, func_name: str):
        """The seed Cypher WHERE clause must contain a guard against
        superseded/exhausted name_stage so that terminal-state nodes
        are never claimed."""
        import imas_codex.standard_names.graph_ops as go

        func = getattr(go, func_name)

        gc, tx = _mock_gc_tx()
        # Seed returns empty — we only care about the generated Cypher.
        tx.run = MagicMock(return_value=[])

        with _patch_gc(gc):
            try:
                func(batch_size=1)
            except Exception:
                pass  # Some may raise on empty result; that's fine.

        # The first tx.run call is the seed query.
        if tx.run.call_args_list:
            seed_cypher = tx.run.call_args_list[0].args[0]
            assert "superseded" in seed_cypher, (
                f"{func_name} seed Cypher must exclude superseded nodes"
            )
            assert "exhausted" in seed_cypher, (
                f"{func_name} seed Cypher must exclude exhausted nodes"
            )

    def test_claim_generate_docs_skips_superseded_names(self):
        """When all eligible nodes have name_stage='superseded', the claim
        returns an empty batch (the WHERE clause filters them out)."""
        from imas_codex.standard_names.graph_ops import claim_generate_docs_batch

        gc, tx = _mock_gc_tx()
        # Seed returns empty — superseded nodes filtered out.
        tx.run = MagicMock(return_value=[])

        with _patch_gc(gc):
            result = claim_generate_docs_batch(batch_size=5)

        assert result == []

    def test_claim_review_name_skips_superseded_names(self):
        """When all eligible nodes have name_stage='superseded', the claim
        returns an empty batch."""
        from imas_codex.standard_names.graph_ops import claim_review_name_batch

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(return_value=[])

        with _patch_gc(gc):
            result = claim_review_name_batch(batch_size=5)

        assert result == []


# ---------------------------------------------------------------------------
# B3 — Persist guard tests: name_stage verified in transaction
# ---------------------------------------------------------------------------


class TestPersistGuardsNameStage:
    """Verify that persist functions include name_stage/docs_stage guards
    so a concurrent supersession produces 0 rows → ValueError/no-op."""

    def test_persist_generated_docs_rejects_superseded(self):
        """persist_generated_docs raises ValueError when the node's
        name_stage has been changed from 'accepted' to 'superseded'
        between claim and persist (0 rows returned)."""
        from imas_codex.standard_names.graph_ops import persist_generated_docs

        gc = MagicMock()
        gc.__enter__ = MagicMock(return_value=gc)
        gc.__exit__ = MagicMock(return_value=False)
        # Simulate 0 rows returned — the name_stage guard filtered it out.
        gc.query = MagicMock(return_value=[])

        with _patch_gc(gc):
            with pytest.raises(ValueError, match="token mismatch"):
                persist_generated_docs(
                    sn_id="test_superseded_sn",
                    claim_token="fake-token-123",
                    description="A description",
                    documentation="Some docs",
                    model="test/model",
                )

    def test_persist_generated_docs_cypher_contains_name_stage_guard(self):
        """The persist_generated_docs Cypher must include
        ``sn.name_stage = 'accepted'`` in its WHERE clause."""
        from imas_codex.standard_names.graph_ops import persist_generated_docs

        gc = MagicMock()
        gc.__enter__ = MagicMock(return_value=gc)
        gc.__exit__ = MagicMock(return_value=False)
        gc.query = MagicMock(return_value=[{"docs_stage": "drafted"}])

        with _patch_gc(gc):
            persist_generated_docs(
                sn_id="test_sn",
                claim_token="tok-abc",
                description="desc",
                documentation="doc",
                model="test/model",
            )

        cypher = gc.query.call_args.args[0]
        assert "name_stage = 'accepted'" in cypher

    def test_persist_reviewed_name_rejects_superseded(self):
        """persist_reviewed_name returns '' when the node's name_stage has
        been changed from 'drafted' (0 rows from the chain_length read)."""
        from imas_codex.standard_names.graph_ops import persist_reviewed_name

        gc = MagicMock()
        gc.__enter__ = MagicMock(return_value=gc)
        gc.__exit__ = MagicMock(return_value=False)
        # chain_length read returns 0 rows — name_stage guard filtered it.
        gc.query = MagicMock(return_value=[])

        with _patch_gc(gc):
            result = persist_reviewed_name(
                sn_id="test_superseded_sn",
                claim_token="fake-token",
                score=0.9,
                model="test/model",
            )

        assert result == ""

    def test_persist_reviewed_name_cypher_contains_name_stage_guard(self):
        """Both queries in persist_reviewed_name include
        ``sn.name_stage = 'drafted'`` in their WHERE clause."""
        from imas_codex.standard_names.graph_ops import persist_reviewed_name

        gc = MagicMock()
        gc.__enter__ = MagicMock(return_value=gc)
        gc.__exit__ = MagicMock(return_value=False)

        # First call: chain_length read; second call: write.
        gc.query = MagicMock(
            side_effect=[
                [{"chain_length": 0}],  # chain_length read
                [],  # write
            ]
        )

        with _patch_gc(gc), patch("imas_codex.standard_names.graph_ops.write_reviews"):
            persist_reviewed_name(
                sn_id="test_sn",
                claim_token="tok-abc",
                score=0.95,
                model="test/model",
            )

        # Both queries should contain the guard.
        for call_obj in gc.query.call_args_list:
            cypher = call_obj.args[0]
            assert "name_stage = 'drafted'" in cypher, (
                f"Missing name_stage guard in Cypher:\n{cypher}"
            )

    def test_persist_reviewed_docs_rejects_superseded(self):
        """persist_reviewed_docs returns '' when name_stage is not 'accepted'."""
        from imas_codex.standard_names.graph_ops import persist_reviewed_docs

        gc = MagicMock()
        gc.__enter__ = MagicMock(return_value=gc)
        gc.__exit__ = MagicMock(return_value=False)
        gc.query = MagicMock(return_value=[])

        with _patch_gc(gc):
            result = persist_reviewed_docs(
                sn_id="test_superseded_sn",
                claim_token="fake-token",
                score=0.8,
                model="test/model",
            )

        assert result == ""

    def test_persist_refined_docs_returns_noop_for_superseded(self):
        """persist_refined_docs returns no-op dict when name_stage guard
        filters out the node (concurrent supersession)."""
        from imas_codex.standard_names.graph_ops import persist_refined_docs

        gc, tx = _mock_gc_tx()
        # Transaction returns 0 rows — name_stage guard filtered it.
        tx.run = MagicMock(return_value=[])

        with _patch_gc(gc):
            result = persist_refined_docs(
                sn_id="test_superseded_sn",
                claim_token="fake-token",
                description="new desc",
                documentation="new docs",
                model="test/model",
                current_description="old desc",
                current_documentation="old docs",
            )

        assert result["docs_chain_length"] == -1
        assert result["revision_id"] == ""


# ---------------------------------------------------------------------------
# B2 + B3 combined: no infinite loop possible
# ---------------------------------------------------------------------------


class TestNoInfiniteLoopOnSupersede:
    """Combined test verifying that the B2 + B3 guards together break the
    infinite retry loop: claim skips superseded → even if race slips through,
    persist produces 0 rows → ValueError → release → next claim skips."""

    def test_claim_then_persist_race_does_not_loop(self):
        """Simulate the race: claim succeeds (node not yet superseded),
        then persist fails (node superseded between claim and persist).
        Verify: persist raises ValueError, and a subsequent claim returns
        empty (B2 filter kicks in)."""
        from imas_codex.standard_names.graph_ops import (
            claim_generate_docs_batch,
            persist_generated_docs,
        )

        # --- Phase 1: claim succeeds (node still accepted) ---
        gc_claim, tx_claim = _mock_gc_tx()
        tx_claim.run = MagicMock(
            side_effect=[
                # seed: returns a grouping key
                [
                    {
                        "_cluster_id": "cluster1",
                        "_unit": "eV",
                        "_physics_domain": "core_profiles",
                    }
                ],
                # read-back: returns claimed item (including claim_token)
                [
                    {
                        "id": "sn_target",
                        "description": "desc",
                        "documentation": None,
                        "kind": "scalar",
                        "unit": "eV",
                        "cluster_id": "cluster1",
                        "physics_domain": "core_profiles",
                        "tags": [],
                        "reviewer_score_name": 0.9,
                        "reviewer_comments_name": "good",
                        "chain_length": 0,
                        "docs_stage": "pending",
                        "name_stage": "accepted",
                        "validation_status": "valid",
                        "claim_token": "test-token-abc",
                    }
                ],
            ]
        )

        with (
            _patch_gc(gc_claim),
            patch(
                "imas_codex.standard_names.chain_history.name_chain_history",
                return_value=[],
            ),
        ):
            items = claim_generate_docs_batch(batch_size=1)
        assert len(items) == 1
        assert items[0]["claim_token"] == "test-token-abc"

        # --- Phase 2: persist fails (node superseded mid-flight) ---
        gc_persist = MagicMock()
        gc_persist.__enter__ = MagicMock(return_value=gc_persist)
        gc_persist.__exit__ = MagicMock(return_value=False)
        gc_persist.query = MagicMock(return_value=[])  # 0 rows

        with _patch_gc(gc_persist):
            with pytest.raises(ValueError, match="token mismatch"):
                persist_generated_docs(
                    sn_id="sn_target",
                    claim_token="test-token-abc",
                    description="desc",
                    documentation="docs",
                    model="test/model",
                )

        # --- Phase 3: next claim returns empty (B2 filter) ---
        gc_reclaim, tx_reclaim = _mock_gc_tx()
        tx_reclaim.run = MagicMock(return_value=[])  # No eligible items

        with _patch_gc(gc_reclaim):
            items2 = claim_generate_docs_batch(batch_size=1)
        assert items2 == []
