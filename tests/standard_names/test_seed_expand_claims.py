"""Tests for Phase 8 seed-and-expand claim queries.

Verifies the five ``claim_*_seed_and_expand`` functions in
``imas_codex.standard_names.graph_ops`` that will be used by the
``run_sn_pools`` worker-pool orchestrator.

Tests mock :class:`GraphClient` — no live Neo4j required.

All five claim functions now execute seed+expand+read-back inside a
**single Neo4j transaction** (via ``session.begin_transaction()``).
The mock helper ``_mock_gc_tx()`` wires up the nested context
managers so ``tx.run()`` receives the same side-effect sequences
that previously drove ``gc.query()``.
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_gc_tx():
    """Build a mock GraphClient that supports transactional claim queries.

    Returns ``(gc, tx)`` where *gc* is the mock GraphClient and *tx* is
    the mock Transaction whose ``.run()`` should be configured with
    ``side_effect`` by each test.
    """
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


def _mock_gc():
    """Build a mock GraphClient that supports ``with`` blocks.

    Legacy helper — kept for tests that only need ``gc.query()``.
    For transaction-based tests, use ``_mock_gc_tx()``.
    """
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    return gc


def _patch_gc(mock_gc):
    """Return a ``patch`` context manager for GraphClient."""
    return patch(
        "imas_codex.standard_names.graph_ops.GraphClient",
        return_value=mock_gc,
    )


# ---------------------------------------------------------------------------
# 1. test_seed_is_random
# ---------------------------------------------------------------------------


class TestSeedIsRandom:
    """Populate 100 candidate rows; run claim 10×; seeds must not be constant."""

    def test_compose_seed_is_random(self):
        """claim_generate_name_seed_and_expand sends ORDER BY rand() and passes
        through whichever seed the graph returns."""
        from imas_codex.standard_names.graph_ops import (
            claim_generate_name_seed_and_expand,
        )

        seen_ids: set[str] = set()
        for _i in range(10):
            gc, tx = _mock_gc_tx()
            # Seed returns a different row each time (simulating rand())
            seed_id = f"src-{random.randint(1, 100)}"
            tx.run = MagicMock(
                side_effect=[
                    # Step 1 (seed) — with batch_size=1, expand is skipped
                    [
                        {
                            "_cluster_id": "c1",
                            "_unit": "eV",
                            "_physics_domain": "core_profiles",
                            "_batch_key": "core_profiles",
                        }
                    ],
                    # Step 2 (read-back — no expand when batch_size=1)
                    [
                        {
                            "id": seed_id,
                            "source_id": f"cp/{seed_id}",
                            "source_type": "dd",
                            "batch_key": "core_profiles",
                            "description": "test",
                        }
                    ],
                ]
            )
            with _patch_gc(gc):
                result = claim_generate_name_seed_and_expand(batch_size=1)
                if result:
                    seen_ids.add(result[0]["id"])

            # Verify the Cypher seed query uses ORDER BY rand()
            seed_query = tx.run.call_args_list[0]
            assert "ORDER BY rand()" in seed_query.args[0]

        # With random seeds from 1..100, 10 draws should NOT all be the same
        assert len(seen_ids) > 1, f"all seeds identical: {seen_ids}"

    def test_sn_seed_is_random(self):
        """StandardName-based claims use ORDER BY rand()."""
        from imas_codex.standard_names.graph_ops import (
            claim_enrich_seed_and_expand,
        )

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
                # seed
                [{"_cluster_id": "c1", "_unit": "eV", "_physics_domain": "eq"}],
                # expand
                None,
                # read-back
                [
                    {
                        "id": "n1",
                        "description": "d",
                        "documentation": None,
                        "kind": None,
                        "unit": "eV",
                        "cluster_id": "c1",
                        "physics_domain": ["eq"],
                        "validation_status": "valid",
                        "enriched_at": None,
                    }
                ],
            ]
        )
        with _patch_gc(gc):
            claim_enrich_seed_and_expand(batch_size=5)

        seed_query = tx.run.call_args_list[0].args[0]
        assert "ORDER BY rand()" in seed_query


# ---------------------------------------------------------------------------
# 2. test_batch_is_coherent
# ---------------------------------------------------------------------------


class TestBatchIsCoherent:
    """Returned batch items all share the seed's (cluster_id, unit)."""

    def test_enrich_batch_coherent(self):
        from imas_codex.standard_names.graph_ops import (
            claim_enrich_seed_and_expand,
        )

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
                # seed — cluster c1, unit eV
                [{"_cluster_id": "c1", "_unit": "eV", "_physics_domain": "cp"}],
                # expand (SET, no return)
                None,
                # read-back: 3 items, all same cluster/unit
                [
                    {
                        "id": f"n{i}",
                        "description": f"d{i}",
                        "documentation": None,
                        "kind": None,
                        "unit": "eV",
                        "cluster_id": "c1",
                        "physics_domain": ["cp"],
                        "validation_status": "valid",
                        "enriched_at": None,
                    }
                    for i in range(3)
                ],
            ]
        )

        with _patch_gc(gc):
            items = claim_enrich_seed_and_expand(batch_size=5)

        assert len(items) == 3
        assert all(it["cluster_id"] == "c1" for it in items)
        assert all(it["unit"] == "eV" for it in items)

        # Verify expand query targets the same cluster × unit
        expand_query = tx.run.call_args_list[1].args[0]
        assert "IMASSemanticCluster" in expand_query
        assert "$cluster_id" in expand_query
        assert "$unit" in expand_query


# ---------------------------------------------------------------------------
# 3. test_fallback_to_physics_domain
# ---------------------------------------------------------------------------


class TestFallbackToPhysicsDomain:
    """When seed has no cluster_id, expand uses (physics_domain, unit)."""

    def test_fallback_domain(self):
        from imas_codex.standard_names.graph_ops import (
            claim_review_names_seed_and_expand,
        )

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
                # seed — NO cluster_id, has physics_domain + unit
                [
                    {
                        "_cluster_id": None,
                        "_unit": "T",
                        "_physics_domain": "magnetics",
                    }
                ],
                # expand (fallback path)
                None,
                # read-back
                [
                    {
                        "id": "n1",
                        "description": "d",
                        "documentation": None,
                        "kind": None,
                        "unit": "T",
                        "cluster_id": None,
                        "physics_domain": ["magnetics"],
                        "validation_status": "valid",
                        "reviewer_score_name": None,
                        "reviewed_name_at": None,
                    }
                ],
            ]
        )

        with _patch_gc(gc):
            items = claim_review_names_seed_and_expand(batch_size=5)

        assert len(items) == 1

        # The expand query should reference $fallback_domain, NOT $cluster_id
        expand_query = tx.run.call_args_list[1].args[0]
        assert "$fallback_domain" in expand_query
        assert "physics_domain" in expand_query
        assert "$cluster_id" not in expand_query


# ---------------------------------------------------------------------------
# 4. test_review_excludes_low_score
# ---------------------------------------------------------------------------


class TestReviewExcludesLowScore:
    """review_names claim with min_score=0.5 excludes reviewer_score<0.5."""

    def test_review_names_eligibility(self):
        from imas_codex.standard_names.graph_ops import (
            claim_review_names_seed_and_expand,
        )

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
                # seed
                [{"_cluster_id": "c1", "_unit": "eV", "_physics_domain": "cp"}],
                # expand
                None,
                # read-back — only names with high scores
                [
                    {
                        "id": "high_score_name",
                        "description": "d",
                        "documentation": None,
                        "kind": None,
                        "unit": "eV",
                        "cluster_id": "c1",
                        "physics_domain": ["cp"],
                        "validation_status": "valid",
                        "reviewer_score_name": 0.9,
                        "reviewed_name_at": None,
                    }
                ],
            ]
        )

        with _patch_gc(gc):
            items = claim_review_names_seed_and_expand(batch_size=5, min_score=0.5)

        # Verify the WHERE clause contains the B3 exclusivity filter
        seed_query = tx.run.call_args_list[0].args[0]
        assert "coalesce(sn.reviewer_score_name, 1.0) >= $min_score" in seed_query

        # Should only return the high-score name
        assert len(items) == 1
        assert items[0]["id"] == "high_score_name"

    def test_review_docs_excludes_low_score(self):
        from imas_codex.standard_names.graph_ops import (
            claim_review_docs_seed_and_expand,
        )

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
                [{"_cluster_id": None, "_unit": None, "_physics_domain": None}],
                # no expand (no grouping keys)
                # read-back
                [
                    {
                        "id": "n1",
                        "description": "d",
                        "documentation": "doc",
                        "kind": None,
                        "unit": None,
                        "cluster_id": None,
                        "physics_domain": None,
                        "validation_status": "valid",
                        "reviewer_score_docs": None,
                        "reviewed_docs_at": None,
                        "enriched_at": "2024-01-01",
                    }
                ],
            ]
        )

        with _patch_gc(gc):
            items = claim_review_docs_seed_and_expand(batch_size=1, min_score=0.5)

        seed_query = tx.run.call_args_list[0].args[0]
        assert "coalesce(sn.reviewer_score_name, 1.0) >= $min_score" in seed_query
        assert len(items) == 1


# ---------------------------------------------------------------------------
# 5. test_regen_excludes_unreviewed
# ---------------------------------------------------------------------------


class TestRegenExcludesUnreviewed:
    """Regen claim requires reviewed_name_at IS NOT NULL."""

    def test_regen_eligibility(self):
        from imas_codex.standard_names.graph_ops import (
            claim_regen_seed_and_expand,
        )

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
                # seed — a reviewed name with low score
                [{"_cluster_id": "c1", "_unit": "m", "_physics_domain": "eq"}],
                # expand
                None,
                # read-back
                [
                    {
                        "id": "reviewed_low",
                        "description": "d",
                        "documentation": None,
                        "kind": None,
                        "unit": "m",
                        "cluster_id": "c1",
                        "physics_domain": ["eq"],
                        "validation_status": "valid",
                        "reviewer_score_name": 0.3,
                        "reviewed_name_at": "2024-01-01T00:00:00Z",
                        "regen_count": 0,
                    }
                ],
            ]
        )

        with _patch_gc(gc):
            items = claim_regen_seed_and_expand(min_score=0.5, batch_size=5)

        # Verify the WHERE clause requires reviewed_name_at IS NOT NULL
        seed_query = tx.run.call_args_list[0].args[0]
        assert "sn.reviewed_name_at IS NOT NULL" in seed_query
        assert "sn.reviewer_score_name < $min_score" in seed_query

        assert len(items) == 1
        assert items[0]["id"] == "reviewed_low"


# ---------------------------------------------------------------------------
# 6. test_review_and_regen_disjoint
# ---------------------------------------------------------------------------


class TestReviewAndRegenDisjoint:
    """Review and regen claim predicates are mutually exclusive."""

    def test_predicates_are_disjoint(self):
        """review_names requires reviewed_name_at IS NULL;
        regen requires reviewed_name_at IS NOT NULL.
        Same-name overlap is structurally impossible.
        """
        from imas_codex.standard_names.graph_ops import (
            claim_regen_seed_and_expand,
            claim_review_names_seed_and_expand,
        )

        # ── review_names claim ──
        gc_review, tx_review = _mock_gc_tx()
        tx_review.run = MagicMock(
            side_effect=[
                [{"_cluster_id": None, "_unit": None, "_physics_domain": None}],
                # read-back
                [
                    {
                        "id": "unreviewed_name",
                        "description": "d",
                        "documentation": None,
                        "kind": None,
                        "unit": None,
                        "cluster_id": None,
                        "physics_domain": None,
                        "validation_status": "valid",
                        "reviewer_score_name": None,
                        "reviewed_name_at": None,
                    }
                ],
            ]
        )

        with _patch_gc(gc_review):
            review_items = claim_review_names_seed_and_expand(batch_size=1)

        # ── regen claim ──
        gc_regen, tx_regen = _mock_gc_tx()
        tx_regen.run = MagicMock(
            side_effect=[
                [{"_cluster_id": None, "_unit": None, "_physics_domain": None}],
                # read-back
                [
                    {
                        "id": "reviewed_low_score",
                        "description": "d",
                        "documentation": None,
                        "kind": None,
                        "unit": None,
                        "cluster_id": None,
                        "physics_domain": None,
                        "validation_status": "valid",
                        "reviewer_score_name": 0.3,
                        "reviewed_name_at": "2024-01-01T00:00:00Z",
                        "regen_count": 0,
                    }
                ],
            ]
        )

        with _patch_gc(gc_regen):
            regen_items = claim_regen_seed_and_expand(min_score=0.5, batch_size=1)

        review_ids = {it["id"] for it in review_items}
        regen_ids = {it["id"] for it in regen_items}
        assert review_ids.isdisjoint(regen_ids), f"overlap: {review_ids & regen_ids}"

        # Structural check: predicates are mutually exclusive
        review_seed_q = tx_review.run.call_args_list[0].args[0]
        regen_seed_q = tx_regen.run.call_args_list[0].args[0]
        assert "sn.reviewed_name_at IS NULL" in review_seed_q
        assert "sn.reviewed_name_at IS NOT NULL" in regen_seed_q


# ---------------------------------------------------------------------------
# 7. test_claim_token_two_step
# ---------------------------------------------------------------------------


class TestClaimTokenTwoStep:
    """Read-back uses claim_token match, not just claimed_at."""

    def _run_and_check_readback(self, fn, side_effects, **kwargs):
        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(side_effect=side_effects)
        with _patch_gc(gc):
            fn(**kwargs)

        # The final query is the read-back (last call).
        readback_query = tx.run.call_args_list[-1].args[0]
        assert "claim_token" in readback_query, (
            f"read-back must use claim_token match, got:\n{readback_query}"
        )
        return gc, tx

    def test_compose_two_step(self):
        from imas_codex.standard_names.graph_ops import (
            claim_generate_name_seed_and_expand,
        )

        self._run_and_check_readback(
            claim_generate_name_seed_and_expand,
            [
                [
                    {
                        "_cluster_id": "c",
                        "_unit": "V",
                        "_physics_domain": "eq",
                        "_batch_key": "eq",
                    }
                ],
                None,
                [
                    {
                        "id": "s1",
                        "source_id": "x",
                        "source_type": "dd",
                        "batch_key": "eq",
                        "description": "d",
                    }
                ],
            ],
            batch_size=5,
        )

    def test_enrich_two_step(self):
        from imas_codex.standard_names.graph_ops import (
            claim_enrich_seed_and_expand,
        )

        self._run_and_check_readback(
            claim_enrich_seed_and_expand,
            [
                [{"_cluster_id": "c", "_unit": "eV", "_physics_domain": "cp"}],
                None,
                [
                    {
                        "id": "n1",
                        "description": "d",
                        "documentation": None,
                        "kind": None,
                        "unit": "eV",
                        "cluster_id": "c",
                        "physics_domain": ["cp"],
                        "validation_status": "valid",
                        "enriched_at": None,
                    }
                ],
            ],
            batch_size=5,
        )

    def test_review_names_two_step(self):
        from imas_codex.standard_names.graph_ops import (
            claim_review_names_seed_and_expand,
        )

        self._run_and_check_readback(
            claim_review_names_seed_and_expand,
            [
                [{"_cluster_id": "c", "_unit": "eV", "_physics_domain": "cp"}],
                None,
                [
                    {
                        "id": "n1",
                        "description": "d",
                        "documentation": None,
                        "kind": None,
                        "unit": "eV",
                        "cluster_id": "c",
                        "physics_domain": ["cp"],
                        "validation_status": "valid",
                        "reviewer_score_name": None,
                        "reviewed_name_at": None,
                    }
                ],
            ],
            batch_size=5,
        )

    def test_regen_two_step(self):
        from imas_codex.standard_names.graph_ops import (
            claim_regen_seed_and_expand,
        )

        self._run_and_check_readback(
            claim_regen_seed_and_expand,
            [
                [{"_cluster_id": None, "_unit": None, "_physics_domain": None}],
                [
                    {
                        "id": "n1",
                        "description": "d",
                        "documentation": None,
                        "kind": None,
                        "unit": None,
                        "cluster_id": None,
                        "physics_domain": None,
                        "validation_status": "valid",
                        "reviewer_score_name": 0.3,
                        "reviewed_name_at": "2024-01-01",
                        "regen_count": 0,
                    }
                ],
            ],
            batch_size=1,
        )


# ---------------------------------------------------------------------------
# 8. test_retry_on_deadlock_decorator_applied
# ---------------------------------------------------------------------------


class TestRetryOnDeadlockApplied:
    """All five claim functions have @retry_on_deadlock (→ __wrapped__)."""

    @pytest.mark.parametrize(
        "fn_name",
        [
            "claim_generate_name_seed_and_expand",
            "claim_enrich_seed_and_expand",
            "claim_review_names_seed_and_expand",
            "claim_review_docs_seed_and_expand",
            "claim_regen_seed_and_expand",
        ],
    )
    def test_has_wrapped(self, fn_name: str):
        import imas_codex.standard_names.graph_ops as ops

        fn = getattr(ops, fn_name)
        assert hasattr(fn, "__wrapped__"), (
            f"{fn_name} missing __wrapped__ — @retry_on_deadlock not applied"
        )


# ---------------------------------------------------------------------------
# Extra edge-case tests
# ---------------------------------------------------------------------------


class TestEmptyPoolReturnsEmpty:
    """When no eligible seed exists, all claim functions return []."""

    @pytest.mark.parametrize(
        "fn_name",
        [
            "claim_generate_name_seed_and_expand",
            "claim_enrich_seed_and_expand",
            "claim_review_names_seed_and_expand",
            "claim_review_docs_seed_and_expand",
            "claim_regen_seed_and_expand",
        ],
    )
    def test_empty(self, fn_name: str):
        import imas_codex.standard_names.graph_ops as ops

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(return_value=[])  # seed returns nothing

        with _patch_gc(gc):
            fn = getattr(ops, fn_name)
            result = fn(batch_size=5)

        assert result == []
