"""Tests for the generate_docs pipeline (P4.1).

Covers:
- Claim eligibility (name_stage='accepted' + docs_stage='pending')
- Claim skips already-drafted docs
- Claim payload includes reviewer feedback fields
- Persist writes docs fields and transitions docs_stage → 'drafted'
- Persist does NOT modify name identity fields
- Persist token mismatch is a no-op
- Failed release clears claim_token, leaves docs_stage at 'pending'
- Worker renders chain history in prompt context
- Worker renders reviewer feedback in prompt context
- Worker uses get_model("language") not reasoning model
- Worker streams per-item progress
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# =============================================================================
# Path constants
# =============================================================================

_GC_PATH = "imas_codex.standard_names.graph_ops.GraphClient"
_CHAIN_HISTORY_PATH = "imas_codex.standard_names.chain_history.name_chain_history"
_TOKEN_A = "aaaaaaaa-0000-0000-0000-000000000001"
_TOKEN_B = "bbbbbbbb-0000-0000-0000-000000000002"

# =============================================================================
# Mock helpers
# =============================================================================


def _make_gc_tx(seed_rows=None, readback_rows=None):
    """Build a mock GraphClient whose session returns a controllable transaction.

    Used for ``_claim_sn_atomic``-based functions. With batch_size=1,
    expand_limit=0 so only seed + readback tx.run calls are made.
    Using cluster_id=None AND unit=None AND physics_domain=None ensures the
    expand step is always skipped (falls through all elif branches).
    """
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)

    tx = MagicMock()
    tx.closed = False
    tx.commit = MagicMock()
    tx.close = MagicMock()

    # cluster_id=None, unit=None, physics_domain=None → expand step skipped
    _seed = (
        seed_rows
        if seed_rows is not None
        else [{"_cluster_id": None, "_unit": None, "_physics_domain": None}]
    )
    _readback = (
        readback_rows
        if readback_rows is not None
        else [
            {
                "id": "electron_temperature",
                "name": "electron_temperature",
                "unit": "eV",
                "kind": "scalar",
                "physics_domain": ["core_profiles"],
                "validation_status": "valid",
                "cluster_id": None,
                "claim_token": _TOKEN_A,
                "description": "Electron temperature",
                "tags": None,
                "reviewer_score_name": 0.85,
                "reviewer_comments_name": "Well formed name.",
                "reviewer_verdict_name": "accept",
                "chain_length": 0,
                "docs_stage": "pending",
                "name_stage": "accepted",
            }
        ]
    )
    # With expand skipped: 2 tx.run calls — [seed, readback]
    tx.run = MagicMock(side_effect=[iter(_seed), iter(_readback)])

    session = MagicMock()
    session.begin_transaction = MagicMock(return_value=tx)

    @contextmanager
    def _session_ctx():
        yield session

    gc.session = _session_ctx
    return gc, tx


def _make_gc_query(return_value=None):
    """Build a mock GraphClient with a .query() method."""
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    _rv = [{"released": 1}] if return_value is None else return_value
    gc.query = MagicMock(return_value=_rv)
    return gc


@contextmanager
def _patch_gc(gc):
    with patch(_GC_PATH, return_value=gc):
        yield gc


@contextmanager
def _patch_chain_history(return_value=None):
    with patch(_CHAIN_HISTORY_PATH, return_value=return_value or []):
        yield


def _make_docs_item(
    sn_id: str = "electron_temperature",
    name_stage: str = "accepted",
    docs_stage: str = "pending",
    reviewer_score: float = 0.85,
    reviewer_comments: str = "Well formed name.",
    reviewer_verdict: str = "accept",
    chain_length: int = 0,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a claimed generate_docs item dict."""
    item: dict[str, Any] = {
        "id": sn_id,
        "name": sn_id,
        "description": "Electron kinetic temperature",
        "kind": "scalar",
        "unit": "eV",
        "tags": ["electron", "temperature"],
        "physics_domain": ["core_profiles"],
        "cluster_id": None,
        "validation_status": "valid",
        "name_stage": name_stage,
        "docs_stage": docs_stage,
        "reviewer_score_name": reviewer_score,
        "reviewer_comments_name": reviewer_comments,
        "reviewer_verdict_name": reviewer_verdict,
        "chain_length": chain_length,
        "chain_history": [],
        "claim_token": _TOKEN_A,
    }
    item.update(overrides)
    return item


# =============================================================================
# 1. test_claim_only_when_name_accepted
# =============================================================================


def test_claim_only_when_name_accepted():
    """SN with name_stage='reviewed' is NOT claimed; accepted+pending IS claimed."""
    from imas_codex.standard_names.graph_ops import (
        claim_generate_docs_seed_and_expand,
    )

    # Simulate empty seed result (not eligible)
    gc_empty, tx_empty = _make_gc_tx(seed_rows=[])
    with _patch_gc(gc_empty):
        with _patch_chain_history():
            result = claim_generate_docs_seed_and_expand()

    assert result == [], "SN with name_stage='reviewed' must NOT be claimed"

    # Verify the eligibility WHERE clause contains name_stage='accepted'
    cypher_calls = tx_empty.run.call_args_list
    assert cypher_calls, "At least one tx.run call expected"
    first_call_cypher = cypher_calls[0][0][0]
    assert "name_stage" in first_call_cypher
    assert "accepted" in first_call_cypher
    assert "docs_stage" in first_call_cypher
    assert "pending" in first_call_cypher


def test_claim_eligible_accepted_pending():
    """SN with name_stage='accepted', docs_stage='pending' IS claimed."""
    from imas_codex.standard_names.graph_ops import (
        claim_generate_docs_seed_and_expand,
    )

    gc, tx = _make_gc_tx()
    with _patch_gc(gc):
        with _patch_chain_history():
            items = claim_generate_docs_seed_and_expand(batch_size=1)

    assert len(items) == 1
    assert items[0]["id"] == "electron_temperature"


# =============================================================================
# 2. test_claim_skips_already_drafted_docs
# =============================================================================


def test_claim_skips_already_drafted_docs():
    """SN with docs_stage='drafted' is NOT returned even if name_stage='accepted'."""
    from imas_codex.standard_names.graph_ops import (
        claim_generate_docs_seed_and_expand,
    )

    # Simulate the seed query filtering out docs_stage='drafted' → empty
    gc_empty, tx_empty = _make_gc_tx(seed_rows=[])
    with _patch_gc(gc_empty):
        with _patch_chain_history():
            result = claim_generate_docs_seed_and_expand()

    assert result == []

    # Verify the eligibility WHERE includes docs_stage='pending'
    first_cypher = tx_empty.run.call_args_list[0][0][0]
    assert "docs_stage" in first_cypher
    assert "pending" in first_cypher


# =============================================================================
# 3. test_claim_includes_reviewer_feedback
# =============================================================================


def test_claim_includes_reviewer_feedback():
    """Claimed items include reviewer_score_name, reviewer_comments_name, reviewer_verdict_name."""
    from imas_codex.standard_names.graph_ops import (
        claim_generate_docs_seed_and_expand,
    )

    gc, tx = _make_gc_tx()
    with _patch_gc(gc):
        with _patch_chain_history():
            items = claim_generate_docs_seed_and_expand(batch_size=1)

    assert len(items) == 1
    item = items[0]
    assert "reviewer_score_name" in item
    assert "reviewer_comments_name" in item
    assert "reviewer_verdict_name" in item
    assert item["reviewer_score_name"] == 0.85
    assert item["reviewer_verdict_name"] == "accept"

    # Verify the RETURN clause in the expand query includes these fields
    # Check the readback query (3rd tx.run call) or via extra_return_fields
    cypher_args = [c[0][0] for c in tx.run.call_args_list]
    # At least one query should reference reviewer_score_name
    assert any("reviewer_score_name" in c for c in cypher_args)


# =============================================================================
# 4. test_persist_writes_docs_fields
# =============================================================================


def test_persist_writes_docs_fields():
    """persist_generated_docs writes description, documentation, docs_stage='drafted', etc."""
    from imas_codex.standard_names.graph_ops import persist_generated_docs

    gc = _make_gc_query(return_value=[{"docs_stage": "drafted"}])
    with _patch_gc(gc):
        stage = persist_generated_docs(
            sn_id="electron_temperature",
            claim_token=_TOKEN_A,
            description="The electron kinetic temperature.",
            documentation="Full documentation text here.",
            model="test-model",
        )

    assert stage == "drafted"

    cypher: str = gc.query.call_args[0][0]
    assert "docs_stage" in cypher
    assert "'drafted'" in cypher
    assert "docs_chain_length" in cypher
    assert "docs_model" in cypher
    assert "docs_generated_at" in cypher
    assert "claim_token" in cypher
    assert "claimed_at" in cypher

    kwargs = gc.query.call_args[1]
    assert kwargs["description"] == "The electron kinetic temperature."
    assert kwargs["documentation"] == "Full documentation text here."
    assert kwargs["model"] == "test-model"


# =============================================================================
# 5. test_persist_does_not_change_name_fields
# =============================================================================


def test_persist_does_not_change_name_fields():
    """persist_generated_docs Cypher must NOT SET name, kind, unit, name_stage, or tags."""
    from imas_codex.standard_names.graph_ops import persist_generated_docs

    gc = _make_gc_query(return_value=[{"docs_stage": "drafted"}])
    with _patch_gc(gc):
        persist_generated_docs(
            sn_id="electron_temperature",
            claim_token=_TOKEN_A,
            description="Desc.",
            documentation="Docs.",
            model="test-model",
        )

    cypher: str = gc.query.call_args[0][0]
    # The SET clause must not include identity fields
    set_start = cypher.find("SET")
    set_block = cypher[set_start:] if set_start >= 0 else cypher

    for forbidden in ("sn.name", "sn.kind", "sn.unit", "sn.name_stage", "sn.tags"):
        assert forbidden not in set_block, (
            f"persist_generated_docs must not SET {forbidden}"
        )


# =============================================================================
# 6. test_persist_token_mismatch_no_op
# =============================================================================


def test_persist_token_mismatch_no_op():
    """Wrong token → ValueError raised (no node matched)."""
    from imas_codex.standard_names.graph_ops import persist_generated_docs

    # Simulate empty result from Cypher (token didn't match)
    gc = _make_gc_query(return_value=[])
    with _patch_gc(gc):
        with pytest.raises(ValueError, match="token mismatch"):
            persist_generated_docs(
                sn_id="electron_temperature",
                claim_token=_TOKEN_B,  # wrong token
                description="Desc.",
                documentation="Docs.",
                model="test-model",
            )


# =============================================================================
# 7. test_failed_release_clears_token
# =============================================================================


def test_failed_release_clears_token():
    """release_generate_docs_failed_claims clears claim_token; docs_stage stays 'pending'."""
    from imas_codex.standard_names.graph_ops import (
        release_generate_docs_failed_claims,
    )

    gc = _make_gc_query(return_value=[{"released": 1}])
    with _patch_gc(gc):
        count = release_generate_docs_failed_claims(
            sn_ids=["electron_temperature"],
            claim_token=_TOKEN_A,
        )

    assert count == 1

    cypher: str = gc.query.call_args[0][0]
    # Must clear claim fields
    assert "claim_token" in cypher
    assert "claimed_at" in cypher

    # Must NOT change docs_stage (no SET docs_stage in cypher)
    set_start = cypher.find("SET")
    set_block = cypher[set_start:] if set_start >= 0 else cypher
    assert "docs_stage" not in set_block, (
        "release_generate_docs_failed_claims must NOT change docs_stage"
    )


def test_failed_release_token_verification():
    """release_generate_docs_failed_claims only releases matching token."""
    from imas_codex.standard_names.graph_ops import (
        release_generate_docs_failed_claims,
    )

    gc = _make_gc_query(return_value=[{"released": 0}])
    with _patch_gc(gc):
        count = release_generate_docs_failed_claims(
            sn_ids=["electron_temperature"],
            claim_token=_TOKEN_B,  # different token
        )

    assert count == 0
    cypher: str = gc.query.call_args[0][0]
    assert "$token" in cypher


# =============================================================================
# 8. test_worker_renders_chain_history
# =============================================================================


def test_worker_renders_chain_history():
    """prompt context includes chain_history when chain is non-empty."""
    from imas_codex.llm.prompt_loader import render_prompt

    chain_history = [
        {
            "name": "old_electron_temp",
            "description": "Old description",
            "reviewer_score_name": 0.6,
            "reviewer_comments_per_dim_name": "Name too abbreviated",
        }
    ]
    item = _make_docs_item(chain_length=1, chain_history=chain_history)
    context = {"item": item, "chain_history": chain_history}

    rendered = render_prompt("sn/generate_docs_user", context)
    assert "old_electron_temp" in rendered
    assert "chain" in rendered.lower() or "predecessor" in rendered.lower()


# =============================================================================
# 9. test_worker_renders_reviewer_feedback
# =============================================================================


def test_worker_renders_reviewer_feedback():
    """prompt context includes reviewer feedback fields."""
    from imas_codex.llm.prompt_loader import render_prompt

    item = _make_docs_item(
        reviewer_score=0.91,
        reviewer_comments="Excellent name — very precise.",
        reviewer_verdict="accept",
    )
    context = {"item": item, "chain_history": []}

    rendered = render_prompt("sn/generate_docs_user", context)
    assert "0.91" in rendered
    assert "accept" in rendered


# =============================================================================
# 10. test_worker_uses_language_model
# =============================================================================


@pytest.mark.asyncio
async def test_worker_uses_language_model():
    """process_generate_docs_batch uses get_model('language'), not a reasoning model."""
    from imas_codex.standard_names.models import GeneratedDocs
    from imas_codex.standard_names.workers import process_generate_docs_batch

    mgr = MagicMock()
    mgr.reserve = MagicMock(return_value=MagicMock())

    language_model_used: list[str] = []

    async def _fake_acall(model, messages, response_model, service):
        language_model_used.append(model)
        return (
            GeneratedDocs(
                description="Electron kinetic temperature in the plasma.",
                documentation="The electron temperature $T_e$ is the thermal energy of electrons.",
            ),
            0.001,
            100,
        )

    item = _make_docs_item()
    stop = asyncio.Event()

    with (
        patch(
            "imas_codex.settings.get_model",
            return_value="test-language-model",
        ) as mock_get_model,
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            side_effect=_fake_acall,
        ),
        patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
        ),
        patch(
            "asyncio.to_thread",
            new=AsyncMock(return_value="drafted"),
        ),
    ):
        processed = await process_generate_docs_batch([item], mgr, stop)

    mock_get_model.assert_called_with("language")
    assert processed == 1


# =============================================================================
# 11. test_worker_streams_per_item
# =============================================================================


@pytest.mark.asyncio
async def test_worker_streams_per_item():
    """process_generate_docs_batch logs per-item progress (name + desc preview)."""
    from imas_codex.standard_names.models import GeneratedDocs
    from imas_codex.standard_names.workers import process_generate_docs_batch

    expected_desc = "Electron kinetic temperature in the plasma."

    async def _fake_acall(model, messages, response_model, service):
        return (
            GeneratedDocs(
                description=expected_desc,
                documentation="Full documentation text for electron temperature.",
            ),
            0.001,
            100,
        )

    mgr = MagicMock()
    mgr.reserve = MagicMock(return_value=MagicMock())

    item = _make_docs_item()
    stop = asyncio.Event()

    log_messages: list[str] = []

    import logging

    class _Capture(logging.Handler):
        def emit(self, record):
            log_messages.append(record.getMessage())

    handler = _Capture()
    import imas_codex.standard_names.workers as _workers_mod

    _workers_mod.logger.addHandler(handler)
    _workers_mod.logger.setLevel(logging.DEBUG)

    try:
        with (
            patch(
                "imas_codex.settings.get_model",
                return_value="test-language-model",
            ),
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                side_effect=_fake_acall,
            ),
            patch(
                "asyncio.to_thread",
                new=AsyncMock(return_value="drafted"),
            ),
        ):
            processed = await process_generate_docs_batch([item], mgr, stop)
    finally:
        _workers_mod.logger.removeHandler(handler)

    assert processed == 1
    # At least one log message should reference the item name and description preview
    all_messages = " ".join(log_messages)
    assert "electron_temperature" in all_messages
    assert expected_desc[:50] in all_messages or "generate_docs" in all_messages


# =============================================================================
# Extra: release_generate_docs_claims (normal path)
# =============================================================================


def test_release_generate_docs_claims_correct_token():
    """release_generate_docs_claims with correct token returns count."""
    from imas_codex.standard_names.graph_ops import release_generate_docs_claims

    gc = _make_gc_query(return_value=[{"released": 2}])
    with _patch_gc(gc):
        count = release_generate_docs_claims(
            sn_ids=["sn1", "sn2"],
            claim_token=_TOKEN_A,
        )

    assert count == 2
    cypher: str = gc.query.call_args[0][0]
    assert "$token" in cypher
    assert "claim_token" in cypher


def test_release_generate_docs_claims_empty_ids():
    """release_generate_docs_claims with empty list returns 0 without querying."""
    from imas_codex.standard_names.graph_ops import release_generate_docs_claims

    gc = _make_gc_query()
    with _patch_gc(gc):
        count = release_generate_docs_claims(sn_ids=[], claim_token=_TOKEN_A)

    assert count == 0
    gc.query.assert_not_called()
