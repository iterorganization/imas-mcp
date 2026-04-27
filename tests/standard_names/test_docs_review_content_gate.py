"""Test the docs-review content gate (P0 fix for docs-axis score collapse).

Background: docs review was running on names without docs, and the LLM was
scoring empty `(missing)` content at 0/80 → mean 0.327 across the catalog.
The gate skips names with insufficient documentation/description.

Also covers the model-provenance fix: persist now uses
``state.canonical_review_model`` first (was ``"unknown"``).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_name(
    nid: str,
    *,
    documentation: str = "",
    description: str = "",
    reviewed_name_at: str | None = "2024-04-27T00:00:00Z",
    physics_domain: str = "transport",
) -> dict:
    return {
        "id": nid,
        "description": description,
        "documentation": documentation,
        "kind": "scalar",
        "unit": "m^-3",
        "tags": [],
        "links": [],
        "source_paths": [],
        "physical_base": "",
        "subject": "",
        "component": "",
        "coordinate": "",
        "position": "",
        "process": "",
        "cocos_transformation_type": None,
        "physics_domain": physics_domain,
        "pipeline_status": "drafted",
        "reviewer_scores_name": None,
        "reviewer_scores_docs": None,
        "review_input_hash": None,
        "embedding": None,
        "review_tier": None,
        "source_types": [],
        "source_id": None,
        "generated_at": "2024-04-27T00:00:00Z",
        "reviewed_name_at": reviewed_name_at,
        "reviewed_docs_at": None,
        "link_status": None,
    }


def _make_state(target: str = "docs"):
    from imas_codex.standard_names.review.state import StandardNameReviewState

    state = StandardNameReviewState(facility="dd")
    state.target = target
    state.unreviewed_only = False
    state.force_review = False
    state.status_filter = ""  # Disable status filter to focus on content gate
    return state


async def _run_extract(state, names: list[dict]):
    """Invoke extract_review_worker with patched graph & enrichment.

    GraphClient and the enrichment helpers are imported lazily inside the
    function body, so patches must target the source modules, not pipeline.
    """
    from imas_codex.standard_names.review import pipeline as review_pipeline

    mock_gc = MagicMock()
    mock_gc.query.return_value = names

    fake_client_ctx = MagicMock(__enter__=lambda s: mock_gc, __exit__=lambda *a: False)
    with (
        patch("imas_codex.graph.client.GraphClient", return_value=fake_client_ctx),
        patch(
            "imas_codex.standard_names.review.enrichment.reconstruct_clusters_batch",
            return_value={n["id"]: [] for n in names},
        ),
        patch(
            "imas_codex.standard_names.review.enrichment.group_into_review_batches",
            side_effect=lambda targets, *args, **kw: (
                [{"names": targets, "cluster_id": "c1", "unit": "m^-3"}]
                if targets
                else []
            ),
        ),
    ):
        await review_pipeline.extract_review_worker(state)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_docs_gate_skips_empty_content() -> None:
    """Names with empty documentation AND description must be skipped."""
    state = _make_state(target="docs")
    names = [
        _make_name("empty_one", documentation="", description=""),
        _make_name("empty_two", documentation="   ", description="\n"),
    ]
    asyncio.run(_run_extract(state, names))
    assert state.stats.get("docs_skipped_empty_content") == 2
    assert len(state.target_names) == 0


def test_docs_gate_passes_names_with_documentation() -> None:
    """Names with substantial documentation must pass the gate."""
    state = _make_state(target="docs")
    names = [
        _make_name(
            "good_one",
            documentation="A" * 100,  # >= 50 chars
            description="short",
        ),
    ]
    asyncio.run(_run_extract(state, names))
    assert state.stats.get("docs_skipped_empty_content", 0) == 0
    assert len(state.target_names) == 1


def test_docs_gate_passes_names_with_description_only() -> None:
    """A long enough description (>=20 chars) is sufficient even if doc is short."""
    state = _make_state(target="docs")
    names = [
        _make_name(
            "desc_only",
            documentation="",
            description="x" * 25,
        ),
    ]
    asyncio.run(_run_extract(state, names))
    assert state.stats.get("docs_skipped_empty_content", 0) == 0
    assert len(state.target_names) == 1


def test_docs_gate_inactive_for_names_target() -> None:
    """target=names must NOT apply the docs content gate."""
    state = _make_state(target="names")
    names = [
        # Use reviewed_name_at=None so this name is genuinely unreviewed and
        # passes the "skip already-reviewed" filter for the names target.
        _make_name(
            "empty_for_names",
            documentation="",
            description="",
            reviewed_name_at=None,
        ),
    ]
    asyncio.run(_run_extract(state, names))
    assert "docs_skipped_empty_content" not in state.stats
    assert len(state.target_names) == 1


def test_docs_gate_combines_with_missing_name_review_gate() -> None:
    """Both the missing-name gate and the empty-content gate must independently fire."""
    state = _make_state(target="docs")
    names = [
        # Missing name review → skipped by old gate
        _make_name(
            "no_name_review",
            documentation="A" * 100,
            description="",
            reviewed_name_at=None,
        ),
        # Has name review but empty docs → skipped by new gate
        _make_name(
            "name_reviewed_empty",
            documentation="",
            description="",
        ),
    ]
    asyncio.run(_run_extract(state, names))
    assert state.stats.get("docs_skipped_missing_name") == 1
    assert state.stats.get("docs_skipped_empty_content") == 1
    assert len(state.target_names) == 0


def test_persist_uses_canonical_review_model() -> None:
    """state.canonical_review_model must take precedence over review_model."""
    from imas_codex.standard_names.review import pipeline as review_pipeline

    state = _make_state(target="names")
    state.canonical_review_model = "openrouter/anthropic/claude-opus-4.6"
    state.review_model = "fallback-model"
    state.review_results = [{"name": "test_name", "score": 0.8}]
    state.persist_stats.total = 0

    captured: dict[str, list] = {"merge_calls": []}

    mock_gc = MagicMock()

    def _record_query(cypher, **kw):
        captured["merge_calls"].append((cypher, kw))
        return []

    mock_gc.query.side_effect = _record_query

    async def _run():
        with patch(
            "imas_codex.graph.client.GraphClient",
            return_value=MagicMock(
                __enter__=lambda s: mock_gc, __exit__=lambda *a: False
            ),
        ):
            await review_pipeline.persist_review_worker(state)

    try:
        asyncio.run(_run())
    except Exception:
        pass  # We only care about the model stamping, not full persist success

    # The reviewer_model field must come from canonical_review_model
    assert state.review_results[0].get("reviewer_model") == (
        "openrouter/anthropic/claude-opus-4.6"
    )


def test_persist_falls_back_to_review_model() -> None:
    """When canonical is None, review_model is used."""
    from imas_codex.standard_names.review import pipeline as review_pipeline

    state = _make_state(target="names")
    state.canonical_review_model = None
    state.review_model = "openrouter/openai/gpt-5.4"
    state.review_results = [{"name": "test_name", "score": 0.8}]

    mock_gc = MagicMock()
    mock_gc.query.return_value = []

    async def _run():
        with patch(
            "imas_codex.graph.client.GraphClient",
            return_value=MagicMock(
                __enter__=lambda s: mock_gc, __exit__=lambda *a: False
            ),
        ):
            await review_pipeline.persist_review_worker(state)

    try:
        asyncio.run(_run())
    except Exception:
        pass

    assert state.review_results[0].get("reviewer_model") == "openrouter/openai/gpt-5.4"


def test_persist_uses_unknown_when_both_none() -> None:
    """Final fallback is the literal 'unknown' string."""
    from imas_codex.standard_names.review import pipeline as review_pipeline

    state = _make_state(target="names")
    state.canonical_review_model = None
    state.review_model = None
    state.review_results = [{"name": "test_name", "score": 0.8}]

    mock_gc = MagicMock()
    mock_gc.query.return_value = []

    async def _run():
        with patch(
            "imas_codex.graph.client.GraphClient",
            return_value=MagicMock(
                __enter__=lambda s: mock_gc, __exit__=lambda *a: False
            ),
        ):
            await review_pipeline.persist_review_worker(state)

    try:
        asyncio.run(_run())
    except Exception:
        pass

    assert state.review_results[0].get("reviewer_model") == "unknown"
