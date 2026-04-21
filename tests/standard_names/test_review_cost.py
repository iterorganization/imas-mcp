"""Guardrail: Review records carry cost_usd / tokens_in / tokens_out.

The regression fixed here: review_review_worker built Review dicts via
``_build_review_record()`` without forwarding the batch-level cost and
token usage from ``acall_llm_structured()``. As a result every persisted
``Review`` node had ``cost_usd=NULL``, ``tokens_in=NULL``,
``tokens_out=NULL`` — making budget analytics impossible.

Verify at the Python boundary:

* ``_build_review_record`` writes whatever cost/tokens the caller passes.
* ``write_reviews`` forwards those fields into the Cypher UNWIND batch.
"""

from __future__ import annotations

from unittest.mock import patch

from imas_codex.standard_names.graph_ops import write_reviews
from imas_codex.standard_names.review.pipeline import _build_review_record


def test_build_review_record_preserves_cost_and_tokens() -> None:
    rec = _build_review_record(
        {"id": "electron_temperature"},
        model="anthropic/claude-opus-4.6",
        is_canonical=True,
        reviewed_at="2026-04-21T12:00:00+00:00",
        score=0.83,
        tier="good",
        cost_usd=0.0123,
        tokens_in=4567,
        tokens_out=891,
    )
    assert rec, "expected a non-empty record"
    assert rec["cost_usd"] == 0.0123
    assert rec["tokens_in"] == 4567
    assert rec["tokens_out"] == 891
    assert rec["model"] == "anthropic/claude-opus-4.6"


def test_write_reviews_forwards_cost_and_tokens() -> None:
    records = [
        {
            "id": "electron_temperature:anthropic-claude-opus-4-6:2026-04-21T12:00:00+00:00",
            "standard_name_id": "electron_temperature",
            "model": "anthropic/claude-opus-4.6",
            "model_family": "anthropic",
            "is_canonical": True,
            "score": 0.83,
            "scores_json": "{}",
            "tier": "good",
            "comments": "",
            "reviewed_at": "2026-04-21T12:00:00+00:00",
            "cost_usd": 0.0123,
            "tokens_in": 4567,
            "tokens_out": 891,
        }
    ]

    with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:  # noqa: N806
        MockGC.return_value.__enter__.return_value.query.return_value = []
        n = write_reviews(records)

    assert n == 1
    call = MockGC.return_value.__enter__.return_value.query.call_args
    cypher = call.args[0] if call.args else call.kwargs["query"]
    # Cypher must still SET the three fields
    assert "r.cost_usd" in cypher
    assert "r.tokens_in" in cypher
    assert "r.tokens_out" in cypher
    # And the UNWIND batch passes them through
    batch = call.kwargs.get("batch") or call.args[1]
    assert batch[0]["cost_usd"] == 0.0123
    assert batch[0]["tokens_in"] == 4567
    assert batch[0]["tokens_out"] == 891


def test_build_review_record_defaults_are_none() -> None:
    """When no cost/tokens are supplied, the record stores None — not 0."""
    rec = _build_review_record(
        {"id": "plasma_current"},
        model="x/y",
        is_canonical=False,
        reviewed_at="2026-04-21T00:00:00+00:00",
    )
    assert rec["cost_usd"] is None
    assert rec["tokens_in"] is None
    assert rec["tokens_out"] is None
