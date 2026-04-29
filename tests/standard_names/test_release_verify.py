"""Tests for token + stage verification on claim-release helpers.

All tests mock GraphClient — no live Neo4j required.

Test matrix
-----------
Every test is parametrized across all four in-scope release helpers
(generate_name, enrich, review_names, review_docs) to confirm the
verification pattern is uniform.

- test_release_with_correct_token        — correct token → count=1, claim cleared
- test_release_wrong_token_noop          — wrong token   → count=0, no-op (race-safe)
- test_release_after_orphan_sweep_noop   — simulate sweep (token=null) → count=0
- test_release_stage_mismatch_noop       — stage changed → count=0
- test_failed_release_reverts_stage      — failed release with stage params → stage reverted
- test_batch_partial_release             — 3 items, 2 with correct token → count=2
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.graph_ops import (
    release_enrich_claims,
    release_enrich_failed_claims,
    release_generate_name_claims,
    release_generate_name_failed_claims,
    release_review_docs_claims,
    release_review_docs_failed_claims,
    release_review_names_claims,
    release_review_names_failed_claims,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOKEN_A = "aaaaaaaa-0000-0000-0000-000000000001"
_TOKEN_B = "bbbbbbbb-0000-0000-0000-000000000002"


def _make_gc(released: int) -> MagicMock:
    """Return a mock GraphClient that reports *released* nodes cleared."""
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    gc.query = MagicMock(return_value=[{"released": released}])
    return gc


@contextmanager
def _patch_gc(gc: MagicMock):
    with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=gc):
        yield gc


# ---------------------------------------------------------------------------
# Parametrize helpers — SN helpers share the same kwargs shape
# ---------------------------------------------------------------------------

_SN_HELPERS = [
    pytest.param(release_enrich_claims, id="enrich"),
    pytest.param(release_review_names_claims, id="review_names"),
    pytest.param(release_review_docs_claims, id="review_docs"),
]

_SN_FAILED_HELPERS = [
    pytest.param(release_enrich_failed_claims, id="enrich_failed"),
    pytest.param(release_review_names_failed_claims, id="review_names_failed"),
    pytest.param(release_review_docs_failed_claims, id="review_docs_failed"),
]


def _call_sn_release(fn: Any, sn_ids: list[str], token: str, **kw: Any) -> int:
    return fn(sn_ids=sn_ids, claim_token=token, **kw)


# ---------------------------------------------------------------------------
# 1. Correct token → claim cleared, count == len(ids)
# ---------------------------------------------------------------------------


def test_generate_name_correct_token():
    gc = _make_gc(released=2)
    with _patch_gc(gc):
        count = release_generate_name_claims(
            source_ids=["sns1", "sns2"],
            claim_token=_TOKEN_A,
        )
    assert count == 2
    # WHERE clause must include token verification
    cypher: str = gc.query.call_args[0][0]
    assert "$token" in cypher
    assert "claim_token" in cypher


@pytest.mark.parametrize("release_fn", _SN_HELPERS)
def test_sn_correct_token(release_fn):
    gc = _make_gc(released=1)
    with _patch_gc(gc):
        count = _call_sn_release(release_fn, ["sn1"], _TOKEN_A)
    assert count == 1
    cypher: str = gc.query.call_args[0][0]
    assert "$token" in cypher
    assert "claim_token" in cypher


# ---------------------------------------------------------------------------
# 2. Wrong token → no-op, count == 0
# ---------------------------------------------------------------------------


def test_generate_name_wrong_token_noop():
    """release with T2 when graph has T1 → count=0, no clobber."""
    gc = _make_gc(released=0)  # DB sees token mismatch → 0 released
    with _patch_gc(gc):
        count = release_generate_name_claims(
            source_ids=["sns1"],
            claim_token=_TOKEN_B,
        )
    assert count == 0


@pytest.mark.parametrize("release_fn", _SN_HELPERS)
def test_sn_wrong_token_noop(release_fn):
    gc = _make_gc(released=0)
    with _patch_gc(gc):
        count = _call_sn_release(release_fn, ["sn1"], _TOKEN_B)
    assert count == 0


# ---------------------------------------------------------------------------
# 3. After orphan sweep (token already null) → release with original token → count=0
# ---------------------------------------------------------------------------


def test_generate_name_after_orphan_sweep_noop():
    """Orphan sweep cleared token → late release must not clobber fresh re-claim."""
    gc = _make_gc(released=0)
    with _patch_gc(gc):
        count = release_generate_name_claims(
            source_ids=["sns1"],
            claim_token=_TOKEN_A,
        )
    assert count == 0


@pytest.mark.parametrize("release_fn", _SN_HELPERS)
def test_sn_after_orphan_sweep_noop(release_fn):
    gc = _make_gc(released=0)
    with _patch_gc(gc):
        count = _call_sn_release(release_fn, ["sn1"], _TOKEN_A)
    assert count == 0


# ---------------------------------------------------------------------------
# 4. Stage mismatch → count=0 (node advanced to a different stage)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("release_fn", _SN_HELPERS)
def test_sn_stage_mismatch_noop(release_fn):
    """SN advanced beyond expected_stage → release must be a no-op."""
    gc = _make_gc(released=0)  # DB sees stage mismatch → 0 released
    with _patch_gc(gc):
        count = _call_sn_release(
            release_fn,
            ["sn1"],
            _TOKEN_A,
            expected_stage="drafted",
        )
    assert count == 0
    # Verify the Cypher contains stage filtering
    cypher: str = gc.query.call_args[0][0]
    assert "name_stage" in cypher
    assert "$expected_stage" in cypher


# ---------------------------------------------------------------------------
# 5. Failed release reverts stage + clears claim
# ---------------------------------------------------------------------------


def test_generate_name_failed_release():
    gc = _make_gc(released=1)
    with _patch_gc(gc):
        count = release_generate_name_failed_claims(
            source_ids=["sns1"],
            claim_token=_TOKEN_A,
        )
    assert count == 1


@pytest.mark.parametrize("failed_fn", _SN_FAILED_HELPERS)
def test_sn_failed_release_reverts_stage(failed_fn):
    """Failed release with from_stage + to_stage should include SET for stage."""
    gc = _make_gc(released=1)
    with _patch_gc(gc):
        count = failed_fn(
            sn_ids=["sn1"],
            claim_token=_TOKEN_A,
            from_stage="drafted",
            to_stage="drafted",
        )
    assert count == 1
    cypher: str = gc.query.call_args[0][0]
    assert "$from_stage" in cypher
    assert "$to_stage" in cypher
    assert "name_stage" in cypher


@pytest.mark.parametrize("failed_fn", _SN_FAILED_HELPERS)
def test_sn_failed_release_wrong_token_noop(failed_fn):
    """Failed release with wrong token must be a no-op."""
    gc = _make_gc(released=0)
    with _patch_gc(gc):
        count = failed_fn(
            sn_ids=["sn1"],
            claim_token=_TOKEN_B,
            from_stage="drafted",
            to_stage="drafted",
        )
    assert count == 0


# ---------------------------------------------------------------------------
# 6. Batch partial release — 3 SNs, DB reports 2 released (1 swept)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("release_fn", _SN_HELPERS)
def test_sn_batch_partial_release(release_fn):
    """3 SNs claimed; 1 was swept by orphan sweep → only 2 released."""
    gc = _make_gc(released=2)
    with _patch_gc(gc):
        count = _call_sn_release(release_fn, ["sn1", "sn2", "sn3"], _TOKEN_A)
    assert count == 2  # caller observes partial release without error


def test_generate_name_batch_partial_release():
    gc = _make_gc(released=2)
    with _patch_gc(gc):
        count = release_generate_name_claims(
            source_ids=["sns1", "sns2", "sns3"],
            claim_token=_TOKEN_A,
        )
    assert count == 2


# ---------------------------------------------------------------------------
# 7. Empty id-list fast-path — no DB call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("release_fn", _SN_HELPERS)
def test_sn_empty_ids_no_db_call(release_fn):
    gc = _make_gc(released=0)
    with _patch_gc(gc):
        count = _call_sn_release(release_fn, [], _TOKEN_A)
    assert count == 0
    gc.query.assert_not_called()


def test_generate_name_empty_ids_no_db_call():
    gc = _make_gc(released=0)
    with _patch_gc(gc):
        count = release_generate_name_claims(source_ids=[], claim_token=_TOKEN_A)
    assert count == 0
    gc.query.assert_not_called()
