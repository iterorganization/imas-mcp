"""Tests for Phase 8 pool-mode batch processors.

Covers:

* Batch-scope domain context derivation (H5).
* Cooperative shutdown via stop_event (H6).
* Happy-path processing for each pool.
* Review names/docs independence (M9).
* Exception propagation and budget lease release.
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
    """Return a MagicMock that behaves like BudgetManager.

    ``reserve()`` returns a lease whose ``charge_event()`` and
    ``release_unused()`` are no-ops, and ``pool_admit()`` always admits.
    """
    mgr = MagicMock()
    lease = MagicMock()
    lease.charge_event = MagicMock(return_value=SimpleNamespace(overspend=0.0))
    lease.release_unused = MagicMock(return_value=0.0)
    mgr.reserve = MagicMock(return_value=lease)
    mgr.pool_admit = MagicMock(return_value=True)
    mgr.exhausted = MagicMock(return_value=False)
    return mgr


def _make_batch_items(
    n: int = 3,
    physics_domain: str | None = "magnetics",
    *,
    extra_domains: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Create N synthetic batch items for compose/regen tests."""
    items: list[dict[str, Any]] = []
    domains = [physics_domain] if physics_domain else []
    if extra_domains:
        domains.extend(extra_domains)
    for i in range(n):
        dom = domains[i % len(domains)] if domains else None
        items.append(
            {
                "path": f"equilibrium/time_slice/profiles_1d/field_{i}",
                "description": f"Test field {i} description",
                "physics_domain": dom,
                "unit": "T",
                "dd_version": "4.0.0",
                "cocos_version": "11",
            }
        )
    return items


def _make_sn_items(
    n: int = 3,
    physics_domain: str | None = "magnetics",
) -> list[dict[str, Any]]:
    """Create N synthetic StandardName items for enrich/review tests."""
    items: list[dict[str, Any]] = []
    for i in range(n):
        items.append(
            {
                "id": f"magnetic_field_test_{i}",
                "description": f"Test SN {i} description.",
                "documentation": f"Full documentation for test SN {i}.",
                "kind": "scalar",
                "unit": "T",
                "physics_domain": physics_domain,
                "source_paths": [f"dd:equilibrium/time_slice/profiles_1d/f_{i}"],
                "physical_base": "magnetic_field",
                "subject": None,
                "component": None,
                "coordinate": None,
                "position": None,
                "process": None,
                "pipeline_status": "enriched",
                "validation_status": "valid",
                "reviewed_name_at": None,
                "reviewed_docs_at": None,
            }
        )
    return items


def _mock_compose_llm_result(items: list[dict]) -> Any:
    """Build a mock LLM result for compose calls."""
    candidates = []
    for item in items:
        path = item.get("path", f"unknown/{len(candidates)}")
        cand = SimpleNamespace(
            standard_name=f"test_name_{len(candidates)}",
            source_id=path,
            kind="scalar",
            dd_paths=[path],
            grammar_fields={},
            reason="test",
        )
        candidates.append(cand)
    result = SimpleNamespace(
        candidates=candidates,
        vocab_gaps=[],
        attachments=[],
        skipped=[],
    )

    # Make it unpackable
    class _LLMResult:
        def __init__(self):
            self.input_tokens = 50
            self.output_tokens = 50
            self.cache_read_tokens = 0
            self.cache_creation_tokens = 0

        def __iter__(self):
            return iter((result, 0.01, 100))

    return _LLMResult()


def _mock_enrich_llm_result(items: list[dict]) -> Any:
    """Build a mock LLM result for enrich calls."""
    enriched = []
    for item in items:
        enriched.append(
            SimpleNamespace(
                standard_name=item["id"],
                description=f"Enriched description for {item['id']}.",
                documentation=f"Full docs for {item['id']}.",
                links=[],
                validity_domain=None,
                constraints=None,
            )
        )
    result = SimpleNamespace(items=enriched)

    class _LLMResult:
        def __init__(self):
            self.input_tokens = 50
            self.output_tokens = 50
            self.cache_read_tokens = 0
            self.cache_creation_tokens = 0

        def __iter__(self):
            return iter((result, 0.005, 80))

    return _LLMResult()


def _mock_review_llm_result(items: list[dict]) -> Any:
    """Build a mock LLM result for review calls."""
    reviews = []
    for item in items:
        reviews.append(
            SimpleNamespace(
                standard_name=item.get("id", ""),
                reviewer_score=75,
                reviewer_comments="Looks good.",
                revised_name=None,
                dim_scores={},
            )
        )
    result = SimpleNamespace(reviews=reviews)

    class _LLMResult:
        def __init__(self):
            self.input_tokens = 50
            self.output_tokens = 50
            self.cache_read_tokens = 0
            self.cache_creation_tokens = 0

        def __iter__(self):
            return iter((result, 0.01, 100))

    return _LLMResult()


# The set of patches shared across compose/regen tests.
_COMPOSE_PATCHES = {
    "imas_codex.standard_names.context.build_compose_context": lambda: {},
    "imas_codex.settings.get_compose_lean": lambda: False,
    "imas_codex.standard_names.workers._enrich_batch_items": lambda items: None,
    "imas_codex.standard_names.workers._search_nearby_names": lambda *a, **k: [],
    "imas_codex.standard_names.workers._enrich_ids_context": lambda *a, **k: None,
    "imas_codex.llm.prompt_loader.render_prompt": lambda *a, **k: "mock prompt",
    "imas_codex.standard_names.context.build_domain_vocabulary_preseed": lambda d: (
        f"vocab:{d}" if d else ""
    ),
    "imas_codex.standard_names.review.themes.extract_reviewer_themes": lambda *a,
    **k: [],
    "imas_codex.standard_names.graph_ops.persist_generated_name_batch": lambda *a,
    **k: (len(a[0]) if a else 0),
}


# ---------------------------------------------------------------------------
# Test 1: compose derives domain context from batch (H5)
# ---------------------------------------------------------------------------


class TestComposeDomainContextFromBatch:
    @pytest.mark.asyncio
    async def test_compose_domain_context_from_batch(self) -> None:
        """Pass a batch with mixed physics_domain; mock the LLM call;
        assert the prompt includes all distinct domains via
        build_domain_vocabulary_preseed being called for each domain.
        """
        from imas_codex.standard_names.workers import process_generate_name_batch

        items = _make_batch_items(
            4,
            physics_domain="magnetics",
            extra_domains=["equilibrium", "transport"],
        )
        mgr = _mock_budget_manager()
        stop = asyncio.Event()

        vocab_calls: list[str] = []

        def _track_vocab(domain):
            vocab_calls.append(domain)
            return f"vocab:{domain}" if domain else ""

        with (
            patch(
                "imas_codex.standard_names.context.build_compose_context",
                return_value={},
            ),
            patch(
                "imas_codex.settings.get_compose_lean",
                return_value=False,
            ),
            patch(
                "imas_codex.standard_names.workers._enrich_batch_items",
                side_effect=lambda items: None,
            ),
            patch(
                "imas_codex.standard_names.workers._search_nearby_names",
                return_value=[],
            ),
            patch(
                "imas_codex.standard_names.workers._enrich_ids_context",
                return_value=None,
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="mock prompt",
            ),
            patch(
                "imas_codex.standard_names.context.build_domain_vocabulary_preseed",
                side_effect=_track_vocab,
            ),
            patch(
                "imas_codex.standard_names.review.themes.extract_reviewer_themes",
                return_value=[],
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_generated_name_batch",
                return_value=4,
            ),
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                new_callable=AsyncMock,
                return_value=_mock_compose_llm_result(items),
            ),
            patch(
                "imas_codex.settings.get_model",
                return_value="test-model",
            ),
            patch(
                "imas_codex.standard_names.example_loader.load_compose_examples",
                return_value=[],
            ),
            patch(
                "imas_codex.graph.client.GraphClient",
            ),
        ):
            count = await process_generate_name_batch(items, mgr, stop)

        # All three distinct domains should have been queried
        assert sorted(vocab_calls) == ["equilibrium", "magnetics", "transport"]
        assert count > 0


# ---------------------------------------------------------------------------
# Test 2: compose stop_event short-circuits (H6)
# ---------------------------------------------------------------------------


class TestComposeStopEvent:
    @pytest.mark.asyncio
    async def test_compose_stop_event_short_circuits(self) -> None:
        """Set stop_event before the LLM call; assert function returns 0."""
        from imas_codex.standard_names.workers import process_generate_name_batch

        items = _make_batch_items(3)
        mgr = _mock_budget_manager()
        stop = asyncio.Event()

        # We set stop BEFORE the LLM call by having the domain vocab call
        # trigger the stop event.
        call_count = 0

        def _set_stop_on_second_vocab(domain):
            nonlocal call_count
            call_count += 1
            # After building domain vocab, set stop so the LLM call is skipped
            stop.set()
            return ""

        with (
            patch(
                "imas_codex.standard_names.context.build_compose_context",
                return_value={},
            ),
            patch(
                "imas_codex.settings.get_compose_lean",
                return_value=False,
            ),
            patch(
                "imas_codex.standard_names.workers._enrich_batch_items",
                side_effect=lambda items: None,
            ),
            patch(
                "imas_codex.standard_names.workers._search_nearby_names",
                return_value=[],
            ),
            patch(
                "imas_codex.standard_names.workers._enrich_ids_context",
                return_value=None,
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="mock prompt",
            ),
            patch(
                "imas_codex.standard_names.context.build_domain_vocabulary_preseed",
                side_effect=_set_stop_on_second_vocab,
            ),
            patch(
                "imas_codex.standard_names.review.themes.extract_reviewer_themes",
                return_value=[],
            ),
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                new_callable=AsyncMock,
            ) as mock_llm,
            patch(
                "imas_codex.settings.get_model",
                return_value="test-model",
            ),
            patch(
                "imas_codex.standard_names.example_loader.load_compose_examples",
                return_value=[],
            ),
            patch(
                "imas_codex.graph.client.GraphClient",
            ),
        ):
            count = await process_generate_name_batch(items, mgr, stop)

        # LLM should NOT have been called because stop was set
        mock_llm.assert_not_called()
        assert count == 0


# ---------------------------------------------------------------------------
# Test 3: enrich stop_event short-circuits (H6)
# ---------------------------------------------------------------------------


class TestEnrichStopEvent:
    @pytest.mark.asyncio
    async def test_enrich_stop_event_short_circuits(self) -> None:
        """Set stop_event before LLM call; assert 0 items processed."""
        from imas_codex.standard_names.enrich_workers import process_enrich_batch

        items = _make_sn_items(3)
        mgr = _mock_budget_manager()
        stop = asyncio.Event()
        stop.set()  # Set immediately — should short-circuit

        with (
            patch(
                "imas_codex.standard_names.example_loader.load_compose_examples",
                return_value=[],
            ),
            patch(
                "imas_codex.graph.client.GraphClient",
            ),
            patch(
                "imas_codex.standard_names.enrich_workers._fetch_existing_sn_names",
                return_value=set(),
            ),
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                new_callable=AsyncMock,
            ) as mock_llm,
            patch(
                "imas_codex.settings.get_model",
                return_value="test-model",
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="mock prompt",
            ),
        ):
            count = await process_enrich_batch(items, mgr, stop)

        mock_llm.assert_not_called()
        assert count == 0


# ---------------------------------------------------------------------------
# Test 4: review names processes batch (happy path)
# ---------------------------------------------------------------------------


class TestReviewNamesProcessesBatch:
    @pytest.mark.asyncio
    async def test_review_names_processes_batch(self) -> None:
        """Happy path: mock LLM, assert all items reviewed."""
        from imas_codex.standard_names.review.pipeline import (
            process_review_names_batch,
        )

        items = _make_sn_items(3)
        mgr = _mock_budget_manager()
        stop = asyncio.Event()

        # Mock _review_single_batch to return scored items
        async def _mock_review(**kwargs):
            names = kwargs.get("names", [])
            scored = []
            for n in names:
                scored.append(
                    {
                        "id": n["id"],
                        "reviewer_score": 75,
                        "reviewer_comments": "Good.",
                    }
                )
            return {
                "_items": scored,
                "_cost": 0.01,
                "_tokens": 100,
                "_input_tokens": 50,
                "_output_tokens": 50,
                "_primary_cost": 0.01,
                "_primary_input_tokens": 50,
                "_primary_output_tokens": 50,
                "_revised": 0,
                "_unscored": 0,
            }

        with (
            patch(
                "imas_codex.standard_names.review.pipeline._get_grammar_enums",
                return_value={},
            ),
            patch(
                "imas_codex.standard_names.review.pipeline._get_compose_context_for_review",
                return_value={},
            ),
            patch(
                "imas_codex.standard_names.example_loader.load_review_examples",
                return_value=[],
            ),
            patch(
                "imas_codex.standard_names.review.themes.extract_reviewer_themes",
                return_value=[],
            ),
            patch(
                "imas_codex.settings.get_sn_review_names_models",
                return_value=["test-model"],
            ),
            patch(
                "imas_codex.standard_names.review.pipeline._review_single_batch",
                side_effect=_mock_review,
            ),
            patch(
                "imas_codex.standard_names.review.pipeline._build_review_record",
                return_value={"id": "rec", "standard_name": "test"},
            ),
            patch(
                "imas_codex.standard_names.review.pipeline._get_hash_fn",
                return_value=None,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.write_name_review_results",
                return_value=3,
            ),
            patch(
                "imas_codex.standard_names.review.pipeline._persist_review_records_sync",
                return_value=3,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.update_review_aggregates",
            ),
            patch(
                "imas_codex.graph.client.GraphClient",
            ),
        ):
            count = await process_review_names_batch(items, mgr, stop)

        assert count == 3


# ---------------------------------------------------------------------------
# Test 5: review docs independent from names (M9)
# ---------------------------------------------------------------------------


class TestReviewDocsIndependent:
    @pytest.mark.asyncio
    async def test_review_docs_independent_from_names(self) -> None:
        """process_review_docs_batch can be called without
        process_review_names_batch having run first.
        """
        from imas_codex.standard_names.review.pipeline import (
            process_review_docs_batch,
        )

        # Items with reviewed_name_at=None — docs pool still works
        items = _make_sn_items(2)

        mgr = _mock_budget_manager()
        stop = asyncio.Event()

        async def _mock_review(**kwargs):
            names = kwargs.get("names", [])
            scored = []
            for n in names:
                scored.append(
                    {
                        "id": n["id"],
                        "reviewer_score": 70,
                        "reviewer_comments": "Docs OK.",
                    }
                )
            return {
                "_items": scored,
                "_cost": 0.01,
                "_tokens": 100,
                "_input_tokens": 50,
                "_output_tokens": 50,
                "_primary_cost": 0.01,
                "_primary_input_tokens": 50,
                "_primary_output_tokens": 50,
                "_revised": 0,
                "_unscored": 0,
            }

        with (
            patch(
                "imas_codex.standard_names.review.pipeline._get_grammar_enums",
                return_value={},
            ),
            patch(
                "imas_codex.standard_names.review.pipeline._get_compose_context_for_review",
                return_value={},
            ),
            patch(
                "imas_codex.standard_names.example_loader.load_review_examples",
                return_value=[],
            ),
            patch(
                "imas_codex.standard_names.review.themes.extract_reviewer_themes",
                return_value=[],
            ),
            patch(
                "imas_codex.settings.get_sn_review_docs_models",
                return_value=["test-model"],
            ),
            patch(
                "imas_codex.standard_names.review.pipeline._review_single_batch",
                side_effect=_mock_review,
            ),
            patch(
                "imas_codex.standard_names.review.pipeline._build_review_record",
                return_value={"id": "rec", "standard_name": "test"},
            ),
            patch(
                "imas_codex.standard_names.review.pipeline._get_hash_fn",
                return_value=None,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.write_docs_review_results",
                return_value=2,
            ),
            patch(
                "imas_codex.standard_names.review.pipeline._persist_review_records_sync",
                return_value=2,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.update_review_aggregates",
            ),
            patch(
                "imas_codex.graph.client.GraphClient",
            ),
        ):
            count = await process_review_docs_batch(items, mgr, stop)

        # Should succeed without requiring prior name review
        assert count == 2


# ---------------------------------------------------------------------------
# Test 7: batch processor releases claims on exception
# ---------------------------------------------------------------------------


class TestBatchProcessorReleasesClaimsOnException:
    @pytest.mark.asyncio
    async def test_batch_processor_releases_claims_on_exception(self) -> None:
        """Raise from mock LLM; assert lease.release_unused was called."""
        from imas_codex.standard_names.workers import process_generate_name_batch

        items = _make_batch_items(2)
        mgr = _mock_budget_manager()
        stop = asyncio.Event()

        with (
            patch(
                "imas_codex.standard_names.context.build_compose_context",
                return_value={},
            ),
            patch(
                "imas_codex.settings.get_compose_lean",
                return_value=False,
            ),
            patch(
                "imas_codex.standard_names.workers._enrich_batch_items",
                side_effect=lambda items: None,
            ),
            patch(
                "imas_codex.standard_names.workers._search_nearby_names",
                return_value=[],
            ),
            patch(
                "imas_codex.standard_names.workers._enrich_ids_context",
                return_value=None,
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="mock prompt",
            ),
            patch(
                "imas_codex.standard_names.context.build_domain_vocabulary_preseed",
                return_value="",
            ),
            patch(
                "imas_codex.standard_names.review.themes.extract_reviewer_themes",
                return_value=[],
            ),
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                new_callable=AsyncMock,
                side_effect=RuntimeError("LLM exploded"),
            ),
            patch(
                "imas_codex.settings.get_model",
                return_value="test-model",
            ),
            patch(
                "imas_codex.standard_names.example_loader.load_compose_examples",
                return_value=[],
            ),
            patch(
                "imas_codex.graph.client.GraphClient",
            ),
        ):
            with pytest.raises(RuntimeError, match="LLM exploded"):
                await process_generate_name_batch(items, mgr, stop)

        # Verify the lease was released despite the exception
        lease = mgr.reserve.return_value
        lease.release_unused.assert_called_once()


# ---------------------------------------------------------------------------
# Test 8: on_event comment/description payloads are NOT pre-truncated
# Regression for: https://github.com/Simon-McIntosh/imas-codex/issues/???
# Worker was clipping comments[:80] before emitting on_event, preventing
# terminal-aware clipping in the rich display from using the full width.
# ---------------------------------------------------------------------------

_LONG_COMMENT = "A" * 200  # 200-char comment — well beyond old 80-char clip


def _mock_review_name_llm_result(long_comment: str) -> Any:
    """Build a mock acall_llm_structured return for process_review_name_batch."""
    from types import SimpleNamespace

    result_obj = SimpleNamespace(
        scores=SimpleNamespace(score=0.85, model_dump=lambda: {"score": 0.85}),
        comments=None,
        verdict="accept",
        reasoning=long_comment,
    )

    class _LLMResult:
        def __iter__(self):
            return iter((result_obj, 0.01, 100))

    return _LLMResult()


def _mock_review_docs_llm_result(long_comment: str) -> Any:
    """Build a mock acall_llm_structured return for process_review_docs_batch."""
    from types import SimpleNamespace

    result_obj = SimpleNamespace(
        scores=SimpleNamespace(score=0.85, model_dump=lambda: {"score": 0.85}),
        comments=None,
        verdict="accept",
        reasoning=long_comment,
    )

    class _LLMResult:
        def __iter__(self):
            return iter((result_obj, 0.01, 100))

    return _LLMResult()


def _mock_generate_docs_llm_result(long_desc: str) -> Any:
    """Build a mock acall_llm_structured return for process_generate_docs_batch."""
    from types import SimpleNamespace

    result_obj = SimpleNamespace(
        description=long_desc,
        documentation="Full documentation text.",
    )

    class _LLMResult:
        def __iter__(self):
            return iter((result_obj, 0.01, 100))

    return _LLMResult()


def _mock_refine_docs_llm_result(long_desc: str) -> Any:
    """Build a mock acall_llm_structured return for process_refine_docs_batch."""
    from types import SimpleNamespace

    result_obj = SimpleNamespace(
        description=long_desc,
        documentation="Full documentation text.",
    )

    class _LLMResult:
        def __iter__(self):
            return iter((result_obj, 0.01, 100))

    return _LLMResult()


class TestOnEventPayloadsNotTruncated:
    """Regression: on_event payloads must carry full strings for terminal clipping."""

    @pytest.mark.asyncio
    async def test_review_name_comment_not_clipped(self) -> None:
        """process_review_name_batch emits the full comment in on_event."""
        from imas_codex.standard_names.workers import process_review_name_batch

        items = [
            {
                "id": "magnetic_flux_density",
                "description": "Magnetic flux density.",
                "claim_token": "tok1",
                "pipeline_status": "drafted",
                "name_stage": "claimed_review_name",
            }
        ]
        mgr = _mock_budget_manager()
        stop = asyncio.Event()
        events: list[dict] = []

        llm_result = _mock_review_name_llm_result(_LONG_COMMENT)

        with (
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
            patch(
                "imas_codex.settings.get_sn_review_names_models",
                return_value=["test-model"],
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="mock prompt",
            ),
            patch(
                "imas_codex.standard_names.context._build_enum_lists",
                return_value={},
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_reviewed_name",
                return_value="accepted",
            ),
            patch(
                "imas_codex.standard_names.graph_ops.release_review_names_failed_claims",
            ),
        ):
            await process_review_name_batch(items, mgr, stop, on_event=events.append)

        assert len(events) == 1, "Expected one on_event emission"
        comment = events[0]["comment"]
        assert len(comment) == len(_LONG_COMMENT), (
            f"on_event comment was clipped to {len(comment)} chars "
            f"(expected {len(_LONG_COMMENT)})"
        )
        assert comment == _LONG_COMMENT

    @pytest.mark.asyncio
    async def test_review_docs_comment_not_clipped(self) -> None:
        """process_review_docs_batch emits the full comment in on_event."""
        from imas_codex.standard_names.workers import process_review_docs_batch

        items = [
            {
                "id": "magnetic_flux_density",
                "description": "Magnetic flux density.",
                "documentation": "Full docs.",
                "claim_token": "tok2",
                "pipeline_status": "drafted",
                "docs_stage": "claimed_review_docs",
                "reviewer_score_name": 75,
                "reviewer_comments_name": "OK",
            }
        ]
        mgr = _mock_budget_manager()
        stop = asyncio.Event()
        events: list[dict] = []

        llm_result = _mock_review_docs_llm_result(_LONG_COMMENT)

        with (
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
            patch(
                "imas_codex.settings.get_sn_review_docs_models",
                return_value=["test-model"],
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="mock prompt",
            ),
            patch(
                "imas_codex.standard_names.context._build_enum_lists",
                return_value={},
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_reviewed_docs",
                return_value="accepted",
            ),
            patch(
                "imas_codex.standard_names.graph_ops.release_review_docs_failed_claims",
            ),
        ):
            await process_review_docs_batch(items, mgr, stop, on_event=events.append)

        assert len(events) == 1, "Expected one on_event emission"
        comment = events[0]["comment"]
        assert len(comment) == len(_LONG_COMMENT), (
            f"on_event comment was clipped to {len(comment)} chars "
            f"(expected {len(_LONG_COMMENT)})"
        )
        assert comment == _LONG_COMMENT
