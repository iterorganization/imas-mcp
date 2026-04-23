"""Regression test for rc22 docs-target filter bug.

Bug (caught during C3 stabilization, 2026-04-23):
    When ``target="docs"`` and ``unreviewed_only=True``, the extract filter
    used ``has_score = reviewer_score is not None`` to decide whether a name
    was already reviewed. After name-review persists, the canonical
    ``reviewer_score`` is bootstrapped from the name score, so the filter
    treated every name as already reviewed and produced zero docs targets.

Fix (pipeline.py):
    The freshness check is now target-aware — for ``target="docs"`` it
    consults ``reviewed_docs_at``, for ``target="names"`` it consults
    ``reviewed_name_at``, and otherwise falls back to ``reviewer_score``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from imas_codex.standard_names.review.pipeline import extract_review_worker


class _FakeGraphClient:
    """Minimal GraphClient stub returning canned name rows."""

    _rows: list[dict] = []

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def query(self, cypher, *args, **kwargs):
        # Only the initial StandardName catalog load should receive the rows.
        # Cluster reconstruction and any follow-on queries return empty.
        if "StandardName" in cypher and "validation_status" in cypher:
            return list(self._rows)
        return []


def _make_state(*, target: str, unreviewed_only: bool = True) -> SimpleNamespace:
    """Build a minimal StandardNameReviewState-compatible object."""

    class _FakeExtractStats:
        status_text = ""
        total = 0
        processed = 0
        stream_queue = SimpleNamespace(add=lambda _items: None)

        def freeze_rate(self):
            pass

        def record_batch(self, _n):
            pass

    class _FakeExtractPhase:
        def mark_done(self):
            pass

    return SimpleNamespace(
        target=target,
        unreviewed_only=unreviewed_only,
        force_review=False,
        ids_filter=None,
        domain_filter=None,
        status_filter=None,
        target_names=[],
        all_names=[],
        stats={},
        extract_stats=_FakeExtractStats(),
        extract_phase=_FakeExtractPhase(),
        review_fidelity_rank=None,
        review_batches=[],
        batch_size=10,
    )


_NAME_WITH_NAME_REVIEW_ONLY = {
    "id": "electron_temperature",
    "pipeline_status": "enriched",
    "validation_status": "valid",
    "physics_domain": "equilibrium",
    "reviewer_score": 0.85,
    "reviewer_score_name": 0.85,
    "reviewer_score_docs": None,
    "reviewed_name_at": "2026-04-23T05:00:00Z",
    "reviewed_docs_at": None,
    "review_mode": "names",
    "review_input_hash": None,
}

_NAME_WITH_BOTH_REVIEWS = {
    "id": "ion_temperature",
    "pipeline_status": "enriched",
    "validation_status": "valid",
    "physics_domain": "equilibrium",
    "reviewer_score": 0.85,
    "reviewer_score_name": 0.85,
    "reviewer_score_docs": 0.80,
    "reviewed_name_at": "2026-04-23T05:00:00Z",
    "reviewed_docs_at": "2026-04-23T05:05:00Z",
    "review_mode": "full",
    "review_input_hash": None,
}

_NAME_UNREVIEWED = {
    "id": "plasma_current",
    "pipeline_status": "enriched",
    "validation_status": "valid",
    "physics_domain": "equilibrium",
    "reviewer_score": None,
    "reviewer_score_name": None,
    "reviewer_score_docs": None,
    "reviewed_name_at": None,
    "reviewed_docs_at": None,
    "review_mode": None,
    "review_input_hash": None,
}


async def _run_extract_with_rows(state, rows):
    _FakeGraphClient._rows = rows
    with patch(
        "imas_codex.graph.client.GraphClient",
        _FakeGraphClient,
    ):
        await extract_review_worker(state)


@pytest.mark.asyncio
async def test_docs_target_picks_up_name_reviewed_names():
    """target=docs must include names with reviewed_name_at but no reviewed_docs_at.

    Before the fix, all of them were filtered out because ``reviewer_score``
    was populated by the name-review bootstrap.
    """
    state = _make_state(target="docs")
    await _run_extract_with_rows(state, [_NAME_WITH_NAME_REVIEW_ONLY])
    target_ids = {n["id"] for n in state.target_names}
    assert "electron_temperature" in target_ids, (
        f"docs target must include name-reviewed names; got {target_ids}"
    )


@pytest.mark.asyncio
async def test_docs_target_skips_already_docs_reviewed_names():
    """target=docs must skip names that already have reviewed_docs_at set."""
    state = _make_state(target="docs")
    await _run_extract_with_rows(state, [_NAME_WITH_BOTH_REVIEWS])
    target_ids = {n["id"] for n in state.target_names}
    assert "ion_temperature" not in target_ids, (
        f"docs target must skip already-docs-reviewed names; got {target_ids}"
    )


@pytest.mark.asyncio
async def test_docs_target_skips_name_without_name_review():
    """target=docs gate: skip names without prior name review."""
    state = _make_state(target="docs")
    await _run_extract_with_rows(state, [_NAME_UNREVIEWED])
    target_ids = {n["id"] for n in state.target_names}
    assert "plasma_current" not in target_ids
    assert state.stats.get("docs_skipped_missing_name", 0) == 1


@pytest.mark.asyncio
async def test_name_only_target_uses_reviewed_name_at():
    """target=names must freshness-check against reviewed_name_at."""
    state = _make_state(target="names")
    await _run_extract_with_rows(state, [_NAME_WITH_NAME_REVIEW_ONLY, _NAME_UNREVIEWED])
    target_ids = {n["id"] for n in state.target_names}
    assert "electron_temperature" not in target_ids, (
        "target=names should skip names already name-reviewed"
    )
    assert "plasma_current" in target_ids, (
        "target=names should include unreviewed names"
    )


@pytest.mark.asyncio
async def test_full_target_uses_reviewer_score():
    """target=full (default) still uses canonical reviewer_score for freshness."""
    state = _make_state(target="full")
    await _run_extract_with_rows(state, [_NAME_WITH_NAME_REVIEW_ONLY, _NAME_UNREVIEWED])
    target_ids = {n["id"] for n in state.target_names}
    assert "electron_temperature" not in target_ids
    assert "plasma_current" in target_ids
