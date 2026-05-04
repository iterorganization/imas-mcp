"""Tests for RD-quorum logic in the pool review workers (workers.py).

Covers the cycle 0/1/2 behaviour wired by ``_run_rd_quorum_cycles`` in
``process_review_name_batch`` and ``process_review_docs_batch``:

* Cycle 0 + 1 always run when ≥ 2 models configured.
* Cycle 2 runs only when per-dim disagreement exceeds threshold AND
  ≥ 3 models configured.
* Single-model chain → ``single_review`` resolution method.
* Quarantined names skip LLM and persist a single zero-score Review.
* Per-cycle Review records share a single ``review_group_id`` and
  carry distinct ``cycle_index`` / ``resolution_role`` values.
* Aggregate cost is the sum across all cycles.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_budget_manager() -> MagicMock:
    mgr = MagicMock()
    lease = MagicMock()
    lease.charge_event = MagicMock(return_value=SimpleNamespace(overspend=0.0))
    lease.release_unused = MagicMock(return_value=0.0)
    mgr.reserve = MagicMock(return_value=lease)
    mgr.pool_admit = MagicMock(return_value=True)
    mgr.exhausted = MagicMock(return_value=False)
    mgr.run_id = "test-run"
    return mgr, lease


def _names_result(scores: dict[str, int], reasoning: str = "ok") -> Any:
    """Build a fake LLM result for the names rubric.

    ``scores`` is the int 0-20 dict returned by ``model_dump()``. The
    normalised ``score`` property is computed as the mean.
    """
    norm = sum(scores.values()) / (20.0 * len(scores)) if scores else 0.0
    return SimpleNamespace(
        scores=SimpleNamespace(score=norm, model_dump=lambda: scores),
        comments=None,
        reasoning=reasoning,
    )


def _docs_result(scores: dict[str, int], reasoning: str = "ok") -> Any:
    norm = sum(scores.values()) / (20.0 * len(scores)) if scores else 0.0
    return SimpleNamespace(
        scores=SimpleNamespace(score=norm, model_dump=lambda: scores),
        comments=None,
        reasoning=reasoning,
    )


def _llm_returns(values: list[Any]) -> AsyncMock:
    """AsyncMock that returns successive ``(result, cost, tokens)`` tuples."""
    side_effect = [(v, 0.01, 100) for v in values]
    return AsyncMock(side_effect=side_effect)


def _patch_common(
    *,
    captured: dict[str, Any],
    llm_mock: AsyncMock,
    models: list[str],
    review_axis: str,
):
    """Yield a single ``patch.multiple``-style context for review-pool tests."""

    persist_target = (
        "imas_codex.standard_names.graph_ops.persist_reviewed_name"
        if review_axis == "names"
        else "imas_codex.standard_names.graph_ops.persist_reviewed_docs"
    )
    release_target = (
        "imas_codex.standard_names.graph_ops.release_review_names_failed_claims"
        if review_axis == "names"
        else "imas_codex.standard_names.graph_ops.release_review_docs_failed_claims"
    )
    models_getter = (
        "imas_codex.settings.get_sn_review_names_models"
        if review_axis == "names"
        else "imas_codex.settings.get_sn_review_docs_models"
    )

    def _capture_persist(**kwargs):
        captured["persist"] = kwargs
        return "accepted"

    def _capture_write_reviews(records, **_):
        captured.setdefault("write_reviews_calls", []).append(list(records))
        return len(records)

    return [
        patch("imas_codex.discovery.base.llm.acall_llm_structured", new=llm_mock),
        patch(models_getter, return_value=models),
        patch(
            "imas_codex.settings.get_sn_review_disagreement_threshold",
            return_value=0.20,
        ),
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            return_value="mock prompt",
        ),
        patch(
            "imas_codex.standard_names.context.fetch_review_neighbours",
            return_value={
                "vector_neighbours": [],
                "same_base_neighbours": [],
                "same_path_neighbours": [],
            },
        ),
        patch(
            "imas_codex.standard_names.context._build_enum_lists",
            return_value={},
        ),
        patch(persist_target, side_effect=_capture_persist),
        patch(release_target),
        patch(
            "imas_codex.standard_names.graph_ops.write_reviews",
            side_effect=_capture_write_reviews,
        ),
        patch(
            "imas_codex.standard_names.graph_ops.update_review_aggregates",
        ),
    ]


# ---------------------------------------------------------------------------
# Names axis
# ---------------------------------------------------------------------------


def _names_item() -> dict[str, Any]:
    return {
        "id": "test_field",
        "description": "Test description.",
        "claim_token": "tok",
        "pipeline_status": "drafted",
        "name_stage": "claimed_review_name",
    }


@pytest.mark.asyncio
async def test_names_two_models_agreement_writes_two_cycles() -> None:
    """2 models, agreement → 2 cycle records, ``quorum_consensus`` on c1."""
    from imas_codex.standard_names.workers import process_review_name_batch

    captured: dict[str, Any] = {}
    mgr, lease = _mock_budget_manager()
    # Both reviewers agree perfectly.
    scores = {"grammar": 17, "semantic": 17, "convention": 17, "completeness": 17}
    llm = _llm_returns([_names_result(scores), _names_result(scores)])

    patches = _patch_common(
        captured=captured,
        llm_mock=llm,
        models=["model-a", "model-b"],
        review_axis="names",
    )
    for p in patches:
        p.start()
    try:
        await process_review_name_batch(
            [_names_item()], mgr, asyncio.Event(), on_event=None
        )
    finally:
        for p in patches:
            p.stop()

    assert llm.await_count == 2, "expected exactly 2 LLM calls (cycle 0 + 1)"
    records = captured["write_reviews_calls"][0]
    assert len(records) == 2
    assert {r["cycle_index"] for r in records} == {0, 1}
    assert len({r["review_group_id"] for r in records}) == 1
    assert records[0]["resolution_role"] == "primary"
    assert records[1]["resolution_role"] == "secondary"
    assert records[1]["resolution_method"] == "quorum_consensus"
    assert records[0]["resolution_method"] is None
    # Cost charged once per cycle.
    assert lease.charge_event.call_count == 2


@pytest.mark.asyncio
async def test_names_two_models_disagreement_no_escalator_marks_max_cycles() -> None:
    """2 models, big per-dim disagreement → ``max_cycles_reached`` on c1."""
    from imas_codex.standard_names.workers import process_review_name_batch

    captured: dict[str, Any] = {}
    mgr, _ = _mock_budget_manager()
    # 17/20 vs 8/20 on grammar → diff = 0.45, well over 0.20 threshold.
    s_a = {"grammar": 17, "semantic": 17, "convention": 17, "completeness": 17}
    s_b = {"grammar": 8, "semantic": 17, "convention": 17, "completeness": 17}
    llm = _llm_returns([_names_result(s_a), _names_result(s_b)])

    patches = _patch_common(
        captured=captured,
        llm_mock=llm,
        models=["model-a", "model-b"],
        review_axis="names",
    )
    for p in patches:
        p.start()
    try:
        await process_review_name_batch(
            [_names_item()], mgr, asyncio.Event(), on_event=None
        )
    finally:
        for p in patches:
            p.stop()

    assert llm.await_count == 2, "no escalator available → still only 2 calls"
    records = captured["write_reviews_calls"][0]
    assert len(records) == 2
    assert records[1]["resolution_method"] == "max_cycles_reached"


@pytest.mark.asyncio
async def test_names_three_models_disagreement_runs_escalator() -> None:
    """3 models, big disagreement c0/c1 → cycle 2 runs, ``authoritative_escalation``."""
    from imas_codex.standard_names.workers import process_review_name_batch

    captured: dict[str, Any] = {}
    mgr, _ = _mock_budget_manager()
    s_a = {"grammar": 17, "semantic": 17, "convention": 17, "completeness": 17}
    s_b = {"grammar": 8, "semantic": 17, "convention": 17, "completeness": 17}
    s_c = {"grammar": 13, "semantic": 17, "convention": 17, "completeness": 17}
    llm = _llm_returns(
        [_names_result(s_a), _names_result(s_b), _names_result(s_c, "escalator")]
    )

    patches = _patch_common(
        captured=captured,
        llm_mock=llm,
        models=["model-a", "model-b", "model-c"],
        review_axis="names",
    )
    for p in patches:
        p.start()
    try:
        await process_review_name_batch(
            [_names_item()], mgr, asyncio.Event(), on_event=None
        )
    finally:
        for p in patches:
            p.stop()

    assert llm.await_count == 3
    records = captured["write_reviews_calls"][0]
    assert len(records) == 3
    assert {r["cycle_index"] for r in records} == {0, 1, 2}
    c2 = next(r for r in records if r["cycle_index"] == 2)
    assert c2["resolution_role"] == "escalator"
    assert c2["resolution_method"] == "authoritative_escalation"
    # SN persist receives the escalator's score, not the mean.
    persisted = captured["persist"]
    expected = sum(s_c.values()) / (20.0 * len(s_c))
    assert persisted["score"] == pytest.approx(expected)
    # canonical_model attribution remains cycle-0 model.
    assert persisted["model"] == "model-a"
    assert persisted["skip_review_node"] is True


@pytest.mark.asyncio
async def test_names_three_models_agreement_skips_cycle_two() -> None:
    """3 models, agreement → only 2 LLM calls; no escalator; ``quorum_consensus``."""
    from imas_codex.standard_names.workers import process_review_name_batch

    captured: dict[str, Any] = {}
    mgr, _ = _mock_budget_manager()
    scores = {"grammar": 17, "semantic": 17, "convention": 17, "completeness": 17}
    llm = _llm_returns([_names_result(scores), _names_result(scores)])

    patches = _patch_common(
        captured=captured,
        llm_mock=llm,
        models=["model-a", "model-b", "model-c"],
        review_axis="names",
    )
    for p in patches:
        p.start()
    try:
        await process_review_name_batch(
            [_names_item()], mgr, asyncio.Event(), on_event=None
        )
    finally:
        for p in patches:
            p.stop()

    assert llm.await_count == 2, "agreement → escalator not invoked"
    records = captured["write_reviews_calls"][0]
    assert len(records) == 2
    assert records[1]["resolution_method"] == "quorum_consensus"


@pytest.mark.asyncio
async def test_names_single_model_chain_uses_single_review() -> None:
    """1-model chain → 1 cycle, ``single_review`` resolution method."""
    from imas_codex.standard_names.workers import process_review_name_batch

    captured: dict[str, Any] = {}
    mgr, _ = _mock_budget_manager()
    scores = {"grammar": 17, "semantic": 17, "convention": 17, "completeness": 17}
    llm = _llm_returns([_names_result(scores)])

    patches = _patch_common(
        captured=captured,
        llm_mock=llm,
        models=["only-model"],
        review_axis="names",
    )
    for p in patches:
        p.start()
    try:
        await process_review_name_batch(
            [_names_item()], mgr, asyncio.Event(), on_event=None
        )
    finally:
        for p in patches:
            p.stop()

    assert llm.await_count == 1
    records = captured["write_reviews_calls"][0]
    assert len(records) == 1
    assert records[0]["cycle_index"] == 0
    assert records[0]["resolution_role"] == "primary"
    assert records[0]["resolution_method"] == "single_review"


@pytest.mark.asyncio
async def test_names_quarantined_item_skips_llm() -> None:
    """Quarantined name → no LLM calls, persist via fast-path with score=0."""
    from imas_codex.standard_names.workers import process_review_name_batch

    captured: dict[str, Any] = {}
    mgr, _ = _mock_budget_manager()
    llm = AsyncMock()  # should never be awaited

    item = _names_item()
    item["validation_status"] = "quarantined"

    patches = _patch_common(
        captured=captured,
        llm_mock=llm,
        models=["model-a", "model-b", "model-c"],
        review_axis="names",
    )
    for p in patches:
        p.start()
    try:
        await process_review_name_batch([item], mgr, asyncio.Event(), on_event=None)
    finally:
        for p in patches:
            p.stop()

    assert llm.await_count == 0
    # No write_reviews invocation in the quarantine fast-path.
    assert "write_reviews_calls" not in captured
    # Persist called with score=0 and the quarantine model marker.
    assert captured["persist"]["score"] == 0.0
    assert "quarantined" in captured["persist"]["model"]


# ---------------------------------------------------------------------------
# Docs axis (mirrors names)
# ---------------------------------------------------------------------------


def _docs_item() -> dict[str, Any]:
    return {
        "id": "test_field",
        "description": "Test description.",
        "documentation": "Test documentation.",
        "claim_token": "tok",
        "pipeline_status": "drafted",
        "docs_stage": "claimed_review_docs",
        "reviewer_score_name": 75,
        "reviewer_comments_name": "OK",
    }


@pytest.mark.asyncio
async def test_docs_two_models_agreement_writes_two_cycles() -> None:
    from imas_codex.standard_names.workers import process_review_docs_batch

    captured: dict[str, Any] = {}
    mgr, lease = _mock_budget_manager()
    scores = {
        "description_quality": 17,
        "documentation_quality": 17,
        "completeness": 17,
        "physics_accuracy": 17,
    }
    llm = _llm_returns([_docs_result(scores), _docs_result(scores)])

    patches = _patch_common(
        captured=captured,
        llm_mock=llm,
        models=["model-a", "model-b"],
        review_axis="docs",
    )
    for p in patches:
        p.start()
    try:
        await process_review_docs_batch(
            [_docs_item()], mgr, asyncio.Event(), on_event=None
        )
    finally:
        for p in patches:
            p.stop()

    assert llm.await_count == 2
    records = captured["write_reviews_calls"][0]
    assert len(records) == 2
    assert {r["cycle_index"] for r in records} == {0, 1}
    assert {r["review_axis"] for r in records} == {"docs"}
    assert records[1]["resolution_method"] == "quorum_consensus"
    assert lease.charge_event.call_count == 2


@pytest.mark.asyncio
async def test_docs_three_models_disagreement_runs_escalator() -> None:
    from imas_codex.standard_names.workers import process_review_docs_batch

    captured: dict[str, Any] = {}
    mgr, _ = _mock_budget_manager()
    s_a = {
        "description_quality": 17,
        "documentation_quality": 17,
        "completeness": 17,
        "physics_accuracy": 17,
    }
    s_b = {
        "description_quality": 8,
        "documentation_quality": 17,
        "completeness": 17,
        "physics_accuracy": 17,
    }
    s_c = {
        "description_quality": 13,
        "documentation_quality": 17,
        "completeness": 17,
        "physics_accuracy": 17,
    }
    llm = _llm_returns([_docs_result(s_a), _docs_result(s_b), _docs_result(s_c)])

    patches = _patch_common(
        captured=captured,
        llm_mock=llm,
        models=["model-a", "model-b", "model-c"],
        review_axis="docs",
    )
    for p in patches:
        p.start()
    try:
        await process_review_docs_batch(
            [_docs_item()], mgr, asyncio.Event(), on_event=None
        )
    finally:
        for p in patches:
            p.stop()

    assert llm.await_count == 3
    records = captured["write_reviews_calls"][0]
    assert len(records) == 3
    c2 = next(r for r in records if r["cycle_index"] == 2)
    assert c2["resolution_method"] == "authoritative_escalation"
    expected = sum(s_c.values()) / (20.0 * len(s_c))
    assert captured["persist"]["score"] == pytest.approx(expected)
    assert captured["persist"]["skip_review_node"] is True
