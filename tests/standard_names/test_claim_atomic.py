"""Tests for the single-transaction claim primitive ``_claim_sn_atomic``.

Verifies atomicity, stage parametrisation, token handling, and
randomised ordering of the new claim primitive that replaces the
old 3-query ``_seed_and_expand_sn`` helper.

All tests mock :class:`GraphClient` — no live Neo4j required.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, call, patch

import pytest
from neo4j.exceptions import TransientError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_gc_tx():
    """Build a mock GraphClient wired for single-transaction claims.

    Returns ``(gc, tx)`` where *gc* is the mock ``GraphClient`` and *tx*
    is the mock ``Transaction`` whose ``.run()`` should be configured
    via ``side_effect`` per test.
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


def _patch_gc(mock_gc):
    return patch(
        "imas_codex.standard_names.graph_ops.GraphClient",
        return_value=mock_gc,
    )


# ---------------------------------------------------------------------------
# 1. test_atomic_claim_sets_token_and_stage
# ---------------------------------------------------------------------------


class TestAtomicClaimSetsTokenAndStage:
    """Verify that stage_field+to_stage produce SET clauses in Cypher."""

    def test_stage_transition_in_seed_query(self):
        """When stage_field='name_stage' and to_stage='refining', the
        seed SET clause includes ``sn.name_stage = $to_stage``."""
        from imas_codex.standard_names.graph_ops import _claim_sn_atomic

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
                # seed
                [
                    {
                        "_cluster_id": None,
                        "_unit": None,
                        "_physics_domain": None,
                    }
                ],
                # read-back (no expand — seed has no grouping keys)
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
                    }
                ],
            ]
        )

        with _patch_gc(gc):
            items = _claim_sn_atomic(
                eligibility_where="sn.validation_status = 'valid'",
                query_params={},
                batch_size=1,
                stage_field="name_stage",
                to_stage="refining",
            )

        assert len(items) == 1
        seed_cypher = tx.run.call_args_list[0].args[0]
        assert "sn.name_stage = $to_stage" in seed_cypher
        # Verify the parameter was passed
        seed_kwargs = tx.run.call_args_list[0].kwargs
        assert seed_kwargs.get("to_stage") == "refining"

    def test_no_stage_transition_when_to_stage_none(self):
        """When to_stage is None, the SET clause omits stage writes."""
        from imas_codex.standard_names.graph_ops import _claim_sn_atomic

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
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
                    }
                ],
            ]
        )

        with _patch_gc(gc):
            _claim_sn_atomic(
                eligibility_where="sn.validation_status = 'valid'",
                query_params={},
                batch_size=1,
            )

        seed_cypher = tx.run.call_args_list[0].args[0]
        assert "name_stage" not in seed_cypher
        assert "docs_stage" not in seed_cypher


# ---------------------------------------------------------------------------
# 2. test_atomic_claim_filters_by_from_stage
# ---------------------------------------------------------------------------


class TestAtomicClaimFiltersByFromStage:
    """The eligibility_where clause is applied correctly."""

    def test_eligibility_in_seed_query(self):
        from imas_codex.standard_names.graph_ops import _claim_sn_atomic

        gc, tx = _mock_gc_tx()
        # Seed returns nothing — no eligible items
        tx.run = MagicMock(return_value=[])

        with _patch_gc(gc):
            items = _claim_sn_atomic(
                eligibility_where=(
                    "sn.name_stage = $from_stage"
                    " AND sn.reviewer_score_name < $min_score"
                ),
                query_params={"from_stage": "reviewed", "min_score": 0.75},
                batch_size=5,
            )

        assert items == []

        # Check the generated Cypher
        seed_cypher = tx.run.call_args_list[0].args[0]
        assert "sn.name_stage = $from_stage" in seed_cypher
        assert "sn.reviewer_score_name < $min_score" in seed_cypher

        # Check parameters
        seed_kwargs = tx.run.call_args_list[0].kwargs
        assert seed_kwargs.get("from_stage") == "reviewed"
        assert seed_kwargs.get("min_score") == 0.75


# ---------------------------------------------------------------------------
# 3. test_atomic_claim_returns_expanded_payload
# ---------------------------------------------------------------------------


class TestAtomicClaimReturnsExpandedPayload:
    """Returned dicts have all expected keys (same as old _seed_and_expand_sn)."""

    def test_payload_keys_enrich(self):
        """claim_enrich_seed_and_expand returns standard SN payload + enriched_at."""
        from imas_codex.standard_names.graph_ops import (
            claim_enrich_seed_and_expand,
        )

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
                [{"_cluster_id": "c1", "_unit": "eV", "_physics_domain": "cp"}],
                None,  # expand
                [
                    {
                        "id": "electron_temperature",
                        "description": "Temperature of electrons",
                        "documentation": None,
                        "kind": "scalar",
                        "unit": "eV",
                        "cluster_id": "c1",
                        "physics_domain": ["cp"],
                        "validation_status": "valid",
                        "enriched_at": None,
                    }
                ],
            ]
        )

        with _patch_gc(gc):
            items = claim_enrich_seed_and_expand(batch_size=5)

        assert len(items) == 1
        item = items[0]
        # All standard keys present
        for key in [
            "id",
            "description",
            "documentation",
            "kind",
            "unit",
            "cluster_id",
            "physics_domain",
            "validation_status",
            "enriched_at",
        ]:
            assert key in item, f"missing key: {key}"

    def test_payload_keys_refine_name(self):
        """claim_refine_name_batch returns SN payload + refine-specific fields."""
        from imas_codex.standard_names.graph_ops import (
            claim_refine_name_batch,
        )

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
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
                        "reviewer_comments_per_dim_name": None,
                        "chain_length": 1,
                        "name_stage": "reviewed",
                        "source_paths": [],
                        "chain_history": [],
                        "reviewed_name_at": "2024-01-01",
                        "regen_count": 1,
                    }
                ],
            ]
        )

        with (
            _patch_gc(gc),
            patch(
                "imas_codex.standard_names.chain_history.name_chain_history",
                return_value=[],
            ),
        ):
            items = claim_refine_name_batch(min_score=0.5, batch_size=1)

        assert len(items) == 1
        for key in [
            "reviewer_score_name",
            "reviewer_comments_per_dim_name",
            "chain_length",
            "name_stage",
            "source_paths",
            "chain_history",
        ]:
            assert key in items[0], f"missing key: {key}"


# ---------------------------------------------------------------------------
# 4. test_atomic_claim_random_order
# ---------------------------------------------------------------------------


class TestAtomicClaimRandomOrder:
    """Seed query contains ORDER BY rand()."""

    def test_order_by_rand_in_atomic_primitive(self):
        from imas_codex.standard_names.graph_ops import _claim_sn_atomic

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
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
                    }
                ],
            ]
        )

        with _patch_gc(gc):
            _claim_sn_atomic(
                eligibility_where="sn.validation_status = 'valid'",
                query_params={},
                batch_size=1,
            )

        seed_cypher = tx.run.call_args_list[0].args[0]
        assert "ORDER BY rand()" in seed_cypher

    def test_all_five_claim_functions_use_rand(self):
        """All 5 public claim functions emit ORDER BY rand() via
        the atomic primitive."""
        pytest.skip(
            "removed: claim_compose_seed_and_expand and claim_regen_seed_and_expand"
            " are linear-path symbols that never landed (W2 audit)"
        )
        import imas_codex.standard_names.graph_ops as ops

        fns = [
            ("claim_compose_seed_and_expand", {}),
            ("claim_enrich_seed_and_expand", {}),
            ("claim_review_names_seed_and_expand", {}),
            ("claim_review_docs_batch", {}),
            ("claim_regen_seed_and_expand", {"min_score": 0.5}),
        ]

        for fn_name, extra_kwargs in fns:
            gc, tx = _mock_gc_tx()
            tx.run = MagicMock(return_value=[])  # empty seed

            with _patch_gc(gc):
                getattr(ops, fn_name)(batch_size=1, **extra_kwargs)

            seed_cypher = tx.run.call_args_list[0].args[0]
            assert "ORDER BY rand()" in seed_cypher, (
                f"{fn_name} missing ORDER BY rand()"
            )


# ---------------------------------------------------------------------------
# 5. test_atomic_claim_no_partial_state
# ---------------------------------------------------------------------------


class TestAtomicClaimNoPartialState:
    """On TransientError, the transaction rolls back — no partial commit."""

    def test_transient_error_rolls_back(self):
        """Simulate TransientError after seed succeeds but before read-back.
        The transaction must NOT commit."""
        from imas_codex.standard_names.graph_ops import _claim_sn_atomic

        gc, tx = _mock_gc_tx()

        # Seed succeeds, expand deadlocks
        tx.run = MagicMock(
            side_effect=[
                # seed succeeds
                [
                    {
                        "_cluster_id": "c1",
                        "_unit": "eV",
                        "_physics_domain": "eq",
                    }
                ],
                # expand raises TransientError (deadlock)
                TransientError("Deadlock detected"),
            ]
        )

        with _patch_gc(gc):
            with pytest.raises(TransientError):
                _claim_sn_atomic(
                    eligibility_where="sn.validation_status = 'valid'",
                    query_params={},
                    batch_size=5,
                )

        # tx.commit() must NOT have been called
        tx.commit.assert_not_called()
        # Transaction should be closed (rollback)
        tx.close.assert_called()

    def test_retry_decorator_creates_new_token(self):
        """When @retry_on_deadlock retries a caller, a new token is generated
        for the next attempt."""
        from imas_codex.standard_names.graph_ops import (
            claim_enrich_seed_and_expand,
        )

        call_count = 0
        tokens_seen: list[str] = []

        original_mock_gc_tx = _mock_gc_tx

        def _capture_token_gc():
            gc, tx = original_mock_gc_tx()

            def run_capture(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    # First attempt: seed succeeds then deadlock on expand
                    if call_count == 1:
                        if "token" in kwargs:
                            tokens_seen.append(kwargs["token"])
                        return [
                            {
                                "_cluster_id": "c1",
                                "_unit": "eV",
                                "_physics_domain": "cp",
                            }
                        ]
                    raise TransientError("Deadlock detected")
                else:
                    # Second attempt: succeeds
                    if call_count == 3:
                        if "token" in kwargs:
                            tokens_seen.append(kwargs["token"])
                        return [
                            {
                                "_cluster_id": None,
                                "_unit": None,
                                "_physics_domain": None,
                            }
                        ]
                    return [
                        {
                            "id": "n1",
                            "description": "d",
                            "documentation": None,
                            "kind": None,
                            "unit": None,
                            "cluster_id": None,
                            "physics_domain": None,
                            "validation_status": "valid",
                            "enriched_at": None,
                        }
                    ]

            tx.run = MagicMock(side_effect=run_capture)
            return gc, tx

        # Patch GraphClient to return a new mock each time
        gc1, tx1 = _capture_token_gc()
        gc2, tx2 = _capture_token_gc()

        call_count = 0
        tokens_seen.clear()

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            side_effect=[gc1, gc2],
        ):
            result = claim_enrich_seed_and_expand(batch_size=1)

        assert len(result) == 1
        # Two different tokens were used (one per attempt)
        assert len(tokens_seen) == 2
        assert tokens_seen[0] != tokens_seen[1]


# ---------------------------------------------------------------------------
# 6. test_atomic_claim_commit_called_on_success
# ---------------------------------------------------------------------------


class TestAtomicClaimCommitCalledOnSuccess:
    """The transaction is committed when all steps succeed."""

    def test_commit_after_successful_claim(self):
        from imas_codex.standard_names.graph_ops import _claim_sn_atomic

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
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
                    }
                ],
            ]
        )

        with _patch_gc(gc):
            _claim_sn_atomic(
                eligibility_where="sn.validation_status = 'valid'",
                query_params={},
                batch_size=1,
            )

        tx.commit.assert_called_once()

    def test_no_commit_on_empty_seed(self):
        """Empty seed → tx.close() without commit."""
        from imas_codex.standard_names.graph_ops import _claim_sn_atomic

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(return_value=[])

        with _patch_gc(gc):
            result = _claim_sn_atomic(
                eligibility_where="sn.validation_status = 'valid'",
                query_params={},
                batch_size=1,
            )

        assert result == []
        tx.commit.assert_not_called()
        tx.close.assert_called()


# ---------------------------------------------------------------------------
# 7. test_stage_transition_in_expand_queries
# ---------------------------------------------------------------------------


class TestStageTransitionInExpandQueries:
    """Stage SET clause appears in expand queries too, not just seed."""

    def test_expand_includes_stage_set(self):
        from imas_codex.standard_names.graph_ops import _claim_sn_atomic

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
                # seed with cluster + unit for expand path
                [
                    {
                        "_cluster_id": "c1",
                        "_unit": "eV",
                        "_physics_domain": "cp",
                    }
                ],
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
                        "physics_domain": ["cp"],
                        "validation_status": "valid",
                    }
                ],
            ]
        )

        with _patch_gc(gc):
            _claim_sn_atomic(
                eligibility_where="sn.validation_status = 'valid'",
                query_params={},
                batch_size=5,
                stage_field="docs_stage",
                to_stage="drafted",
            )

        # Expand is the second tx.run call
        expand_cypher = tx.run.call_args_list[1].args[0]
        assert "sn.docs_stage = $to_stage" in expand_cypher


# ---------------------------------------------------------------------------
# 8. test_compose_uses_single_transaction
# ---------------------------------------------------------------------------


class TestComposeUsesSingleTransaction:
    """claim_compose_seed_and_expand uses a single Neo4j transaction."""

    def test_compose_commit_once(self):
        pytest.skip(
            "removed: claim_compose_seed_and_expand is a linear-path symbol"
            " that never landed (W2 audit)"
        )

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
                [
                    {
                        "_cluster_id": "c1",
                        "_unit": "eV",
                        "_physics_domain": "cp",
                        "_batch_key": "core_profiles",
                    }
                ],
                None,  # expand
                [
                    {
                        "id": "src-1",
                        "source_id": "cp/psi",
                        "source_type": "dd",
                        "batch_key": "core_profiles",
                        "description": "Poloidal flux",
                    }
                ],
            ]
        )

        with _patch_gc(gc):
            items = claim_compose_seed_and_expand(batch_size=5)  # noqa: F821

        assert len(items) == 1
        # Verify single commit
        tx.commit.assert_called_once()
        # Verify session.begin_transaction was used (not gc.query)
        gc.query = MagicMock()  # ensure gc.query was NOT called
