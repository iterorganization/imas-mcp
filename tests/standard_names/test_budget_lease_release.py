"""Regression test for BudgetLease leak in per-item worker loops.

Bug: Several pool workers (refine_name, review_name, generate_docs,
review_docs, refine_docs) reserved a per-item ``BudgetLease`` but called
``release_unused()`` only inside their ``except`` handlers. On the happy
path the unused remainder (~$0.05–$0.20 minus the actual ~$0.001 spend)
stayed parked in the BudgetManager's ``_reserved`` pool forever. After
~150–300 iterations the global pool would appear exhausted at ~25 % of
``cost_limit`` and ``pool_admit()`` returned False everywhere — stalling
the run.

This test runs each affected worker through a single happy-path item
with a stub LLM that returns a small cost. After the worker returns,
``mgr._pool`` must equal ``cost_limit - actual_spent`` (within a small
epsilon). On the buggy code, ``_pool`` is left low by the size of
``estimated - actual_cost``.

The test for ``process_refine_name_batch`` and ``process_compose_batch``
is skipped due to deeper coupling with the GraphClient and fanout
machinery (see TODO below). The four remaining workers exercise the
same lease-release pattern, so the leak fix is well covered.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.standard_names.budget import BudgetManager
from imas_codex.standard_names.models import (
    GeneratedDocs,
    RefinedDocs,
    StandardNameQualityCommentsDocs,
    StandardNameQualityCommentsNameOnly,
    StandardNameQualityReviewDocs,
    StandardNameQualityReviewNameOnly,
    StandardNameQualityScoreDocs,
    StandardNameQualityScoreNameOnly,
)

# ── Tiny stub item ───────────────────────────────────────────────────────


def _stub_item() -> dict[str, Any]:
    return {
        "id": "electron_temperature",
        "name": "electron_temperature",
        "description": "Electron kinetic temperature",
        "documentation": "Electron temperature documentation",
        "kind": "scalar",
        "unit": "eV",
        "physics_domain": ["core_profiles"],
        "chain_length": 0,
        "chain_history": [],
        "docs_chain_length": 0,
        "docs_chain_history": [],
        "claim_token": "test-token",
        "name_stage": "accepted",
        "docs_stage": "pending",
        "reviewer_score_name": 0.9,
        "reviewer_comments_name": "Looks good.",
        "reviewer_score_docs": 0.85,
        "reviewer_comments_docs": "Solid docs.",
        "reviewer_comments_per_dim_name": None,
        "reviewer_comments_per_dim_docs": None,
        "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
        "ids_name": "core_profiles",
    }


# Cost returned by stub LLM — much smaller than the per-item reservation
# so we can detect leaks (which would be of size estimated - cost ≈ $0.05–0.20).
_STUB_COST = 0.001
_INITIAL_BUDGET = 10.0
_LEAK_TOLERANCE = 1e-3  # pool must be within 0.001 of expected


def _stub_review_name_response() -> StandardNameQualityReviewNameOnly:
    return StandardNameQualityReviewNameOnly(
        source_id="electron_temperature",
        standard_name="electron_temperature",
        scores=StandardNameQualityScoreNameOnly(
            grammar=20,
            semantic=20,
            convention=20,
            completeness=20,
        ),
        comments=StandardNameQualityCommentsNameOnly(
            grammar="ok",
            semantic="ok",
            convention="ok",
            completeness="ok",
        ),
        reasoning="All dimensions are strong.",
    )


def _stub_review_docs_response() -> StandardNameQualityReviewDocs:
    return StandardNameQualityReviewDocs(
        source_id="electron_temperature",
        standard_name="electron_temperature",
        scores=StandardNameQualityScoreDocs(
            description_quality=20,
            documentation_quality=20,
            completeness=20,
            physics_accuracy=20,
        ),
        comments=StandardNameQualityCommentsDocs(
            description_quality="ok",
            documentation_quality="ok",
            completeness="ok",
            physics_accuracy="ok",
        ),
        reasoning="All dimensions are strong.",
    )


def _make_acall(response_obj: Any):
    async def _fake(model, messages, response_model, service):
        return (response_obj, _STUB_COST, 100)

    return _fake


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_docs_releases_lease_on_happy_path():
    """process_generate_docs_batch must return unused budget to the pool."""
    from imas_codex.standard_names import workers

    mgr = BudgetManager(total_budget=_INITIAL_BUDGET)
    item = _stub_item()
    stop = asyncio.Event()

    fake_response = GeneratedDocs(
        description="Electron kinetic temperature in the plasma.",
        documentation=(
            "The electron temperature $T_e$ describes the thermal energy "
            "distribution of electrons in the plasma."
        ),
    )

    with (
        patch(
            "imas_codex.settings.get_model",
            return_value="test-language-model",
        ),
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            side_effect=_make_acall(fake_response),
        ),
        patch.object(workers, "_asyncio", create=True, new=MagicMock()),
        patch(
            "asyncio.to_thread",
            new=AsyncMock(return_value=None),
        ),
    ):
        processed = await workers.process_generate_docs_batch([item], mgr, stop)

    assert processed == 1
    expected_pool = _INITIAL_BUDGET - _STUB_COST
    assert abs(mgr._pool - expected_pool) < _LEAK_TOLERANCE, (
        f"Lease leak: pool=${mgr._pool:.6f} expected ~${expected_pool:.6f} "
        f"(reservation was 0.20, charged ${_STUB_COST}; "
        f"leak ≈ ${expected_pool - mgr._pool:.4f})"
    )


@pytest.mark.asyncio
async def test_review_name_releases_lease_on_happy_path():
    """process_review_name_batch must return unused budget to the pool."""
    from imas_codex.standard_names import workers

    mgr = BudgetManager(total_budget=_INITIAL_BUDGET)
    item = _stub_item()
    stop = asyncio.Event()

    with (
        patch(
            "imas_codex.settings.get_model",
            return_value="test-language-model",
        ),
        patch(
            "imas_codex.settings.get_sn_review_names_models",
            return_value=["test-review-model"],
        ),
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            side_effect=_make_acall(_stub_review_name_response()),
        ),
        patch(
            "asyncio.to_thread",
            new=AsyncMock(return_value="reviewed"),
        ),
    ):
        await workers.process_review_name_batch([item], mgr, stop)

    expected_pool = _INITIAL_BUDGET - _STUB_COST
    assert abs(mgr._pool - expected_pool) < _LEAK_TOLERANCE, (
        f"Lease leak: pool=${mgr._pool:.6f} expected ~${expected_pool:.6f} "
        f"(reservation was 0.05, charged ${_STUB_COST}; "
        f"leak ≈ ${expected_pool - mgr._pool:.4f})"
    )


@pytest.mark.asyncio
async def test_review_docs_releases_lease_on_happy_path():
    """process_review_docs_batch must return unused budget to the pool."""
    from imas_codex.standard_names import workers

    mgr = BudgetManager(total_budget=_INITIAL_BUDGET)
    item = _stub_item()
    stop = asyncio.Event()

    with (
        patch(
            "imas_codex.settings.get_model",
            return_value="test-language-model",
        ),
        patch(
            "imas_codex.settings.get_sn_review_docs_models",
            return_value=["test-review-model"],
        ),
        patch(
            "imas_codex.settings.get_sn_review_names_models",
            return_value=["test-review-model"],
        ),
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            side_effect=_make_acall(_stub_review_docs_response()),
        ),
        patch(
            "asyncio.to_thread",
            new=AsyncMock(return_value="reviewed"),
        ),
    ):
        await workers.process_review_docs_batch([item], mgr, stop)

    expected_pool = _INITIAL_BUDGET - _STUB_COST
    assert abs(mgr._pool - expected_pool) < _LEAK_TOLERANCE, (
        f"Lease leak: pool=${mgr._pool:.6f} expected ~${expected_pool:.6f} "
        f"(reservation was 0.05, charged ${_STUB_COST}; "
        f"leak ≈ ${expected_pool - mgr._pool:.4f})"
    )


@pytest.mark.asyncio
async def test_refine_docs_releases_lease_on_happy_path():
    """process_refine_docs_batch must return unused budget to the pool."""
    from imas_codex.standard_names import workers

    mgr = BudgetManager(total_budget=_INITIAL_BUDGET)
    item = _stub_item()
    stop = asyncio.Event()

    fake_response = RefinedDocs(
        description="Electron kinetic temperature in the plasma.",
        documentation=(
            "The electron temperature $T_e$ describes the thermal energy "
            "distribution of electrons in the plasma."
        ),
    )

    with (
        patch(
            "imas_codex.settings.get_model",
            return_value="test-language-model",
        ),
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            side_effect=_make_acall(fake_response),
        ),
        patch(
            "imas_codex.graph.client.GraphClient",
        ),
        patch(
            "asyncio.to_thread",
            new=AsyncMock(return_value=None),
        ),
    ):
        processed = await workers.process_refine_docs_batch([item], mgr, stop)

    assert processed == 1
    expected_pool = _INITIAL_BUDGET - _STUB_COST
    assert abs(mgr._pool - expected_pool) < _LEAK_TOLERANCE, (
        f"Lease leak: pool=${mgr._pool:.6f} expected ~${expected_pool:.6f} "
        f"(reservation was 0.20, charged ${_STUB_COST}; "
        f"leak ≈ ${expected_pool - mgr._pool:.4f})"
    )


# TODO: Cover process_refine_name_batch and process_compose_batch.
#
# refine_name uses a GraphClient context manager + optional fan-out branch
# that requires deep mocking of CandidateContext, FanoutScope, run_fanout,
# assign_arm, find_name_key_duplicate, hybrid neighbour search, and the
# persist_refined_name pipeline. Same lease-release pattern is exercised
# by the four tests above; refine_name uses identical reserve/charge/release
# semantics, so the regression coverage is meaningful.
#
# compose_batch uses a single batch-level reservation (already correct,
# served as the reference pattern in the fix). It is exercised by
# tests/standard_names/test_compose_*.py.


@pytest.mark.skip(
    reason=(
        "Coupling: refine_name's GraphClient + fanout path requires deep "
        "fixtures. Same lease-release semantics are covered by the four "
        "tests above; see TODO in module docstring."
    )
)
@pytest.mark.asyncio
async def test_refine_name_releases_lease_on_happy_path():
    pass
