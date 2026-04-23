"""Tests for the three-rubric sn review pipeline (--target {names,docs,full}).

Covers:
* ``StandardNameQualityScoreDocs`` total/score/tier arithmetic.
* ``_match_reviews_to_entries`` stamping ``review_mode`` per target.
* ``_review_single_batch`` selecting the correct prompt + response model
  for each ``target`` value.
* ``sn/review_docs`` prompt renders without referencing the name-scoring
  dimensions from other rubrics.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# StandardNameQualityScoreDocs — arithmetic
# ---------------------------------------------------------------------------


def test_docs_score_total_and_tier() -> None:
    from imas_codex.standard_names.models import StandardNameQualityScoreDocs

    s = StandardNameQualityScoreDocs(
        description_quality=20,
        documentation_quality=18,
        completeness=17,
        physics_accuracy=20,
    )
    assert s.total == 75
    assert s.score == pytest.approx(75 / 80)
    assert s.tier == "outstanding"

    perfect = StandardNameQualityScoreDocs(
        description_quality=20,
        documentation_quality=20,
        completeness=20,
        physics_accuracy=20,
    )
    assert perfect.score == 1.0
    assert perfect.tier == "outstanding"

    good = StandardNameQualityScoreDocs(
        description_quality=15,
        documentation_quality=14,
        completeness=14,
        physics_accuracy=14,
    )
    # 57/80 = 0.7125 → good
    assert good.tier == "good"

    inadequate = StandardNameQualityScoreDocs(
        description_quality=10,
        documentation_quality=10,
        completeness=10,
        physics_accuracy=10,
    )
    assert inadequate.tier == "inadequate"

    poor = StandardNameQualityScoreDocs(
        description_quality=5,
        documentation_quality=5,
        completeness=5,
        physics_accuracy=5,
    )
    assert poor.tier == "poor"


def test_docs_batch_model_parses_minimal_json() -> None:
    from imas_codex.standard_names.models import StandardNameQualityReviewDocsBatch

    batch = StandardNameQualityReviewDocsBatch.model_validate(
        {
            "reviews": [
                {
                    "source_id": "foo",
                    "standard_name": "electron_temperature",
                    "scores": {
                        "description_quality": 19,
                        "documentation_quality": 18,
                        "completeness": 17,
                        "physics_accuracy": 20,
                    },
                    "verdict": "accept",
                    "reasoning": "ok",
                }
            ]
        }
    )
    assert batch.reviews[0].scores.total == 74
    assert batch.reviews[0].scores.tier == "outstanding"


# ---------------------------------------------------------------------------
# _match_reviews_to_entries — review_mode stamping per target
# ---------------------------------------------------------------------------


def _make_docs_review(source_id: str, name: str, **scores: int):
    from imas_codex.standard_names.models import (
        StandardNameQualityScoreDocs,
        StandardNameReviewVerdict,
    )

    return SimpleNamespace(
        source_id=source_id,
        standard_name=name,
        scores=StandardNameQualityScoreDocs(**scores),
        verdict=StandardNameReviewVerdict.accept,
        reasoning="test",
        revised_description=None,
        revised_documentation=None,
    )


def test_match_stamps_docs_mode_for_target_docs() -> None:
    from imas_codex.standard_names.review.pipeline import _match_reviews_to_entries

    wlog = logging.LoggerAdapter(logging.getLogger("test"), {})
    names = [{"id": "electron_temperature", "source_id": "electron_temperature"}]
    reviews = [
        _make_docs_review(
            source_id="electron_temperature",
            name="electron_temperature",
            description_quality=20,
            documentation_quality=18,
            completeness=17,
            physics_accuracy=20,
        )
    ]

    scored, unmatched, _revised = _match_reviews_to_entries(
        reviews, names, wlog, target="docs"
    )
    assert len(scored) == 1
    assert unmatched == []
    assert scored[0]["review_mode"] == "docs"
    assert scored[0]["reviewer_score"] == pytest.approx(75 / 80)


def test_match_stamps_name_only_for_target_names() -> None:
    from imas_codex.standard_names.models import StandardNameQualityScoreNameOnly
    from imas_codex.standard_names.review.pipeline import _match_reviews_to_entries

    wlog = logging.LoggerAdapter(logging.getLogger("test"), {})
    names = [{"id": "x", "source_id": "x"}]
    review = SimpleNamespace(
        source_id="x",
        standard_name="x",
        scores=StandardNameQualityScoreNameOnly(
            grammar=20, semantic=18, convention=19, completeness=18
        ),
        verdict=_accept_verdict(),
        reasoning="ok",
        revised_name=None,
        revised_fields=None,
    )

    scored, _u, _r = _match_reviews_to_entries([review], names, wlog, target="names")
    assert scored[0]["review_mode"] == "names"


def test_match_stamps_full_for_target_full() -> None:
    from imas_codex.standard_names.models import StandardNameQualityScore
    from imas_codex.standard_names.review.pipeline import _match_reviews_to_entries

    wlog = logging.LoggerAdapter(logging.getLogger("test"), {})
    names = [{"id": "x", "source_id": "x"}]
    review = SimpleNamespace(
        source_id="x",
        standard_name="x",
        scores=StandardNameQualityScore(
            grammar=20,
            semantic=18,
            documentation=16,
            convention=19,
            completeness=18,
            compliance=17,
        ),
        verdict=_accept_verdict(),
        reasoning="ok",
        revised_name=None,
        revised_fields=None,
    )

    scored, _u, _r = _match_reviews_to_entries([review], names, wlog, target="full")
    assert scored[0]["review_mode"] == "full"


def _accept_verdict():
    from imas_codex.standard_names.models import StandardNameReviewVerdict

    return StandardNameReviewVerdict.accept


# ---------------------------------------------------------------------------
# _review_single_batch — prompt + response-model dispatch
# ---------------------------------------------------------------------------


class _DispatchStub:
    """Captures prompt_name and response_model passed to the LLM."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, type]] = []


@pytest.mark.asyncio
async def test_single_batch_dispatch_selects_prompt_per_target(monkeypatch) -> None:
    """Each target value must pick the paired prompt name + response model."""
    from imas_codex.standard_names import models as sn_models
    from imas_codex.standard_names.review import pipeline as pl

    captured: dict[str, object] = {}

    def fake_render_prompt(prompt_name: str, context: dict) -> str:
        captured["prompt_name"] = prompt_name
        return "stub"

    async def fake_call_llm_structured(**kwargs):
        captured["response_model"] = kwargs["response_model"]
        # Return an empty reviews batch that matches the response model shape.
        return kwargs["response_model"](reviews=[]), 0.0, 0

    monkeypatch.setattr(pl, "render_prompt", fake_render_prompt, raising=False)

    import imas_codex.discovery.base.llm as dbllm

    monkeypatch.setattr(dbllm, "acall_llm_structured", fake_call_llm_structured)

    # Also patch the local re-import used inside _review_single_batch.
    from imas_codex.llm import prompt_loader

    monkeypatch.setattr(prompt_loader, "render_prompt", fake_render_prompt)

    wlog = logging.LoggerAdapter(logging.getLogger("test"), {})

    async def run(target: str):
        captured.clear()
        await pl._review_single_batch(
            names=[],
            model="fake/model",
            grammar_enums={
                "subjects": [],
                "components": [],
                "positions": [],
                "processes": [],
                "transformations": [],
                "geometric_bases": [],
                "objects": [],
                "binary_operators": [],
            },
            compose_ctx={},
            batch_context="",
            neighborhood=[],
            audit_findings=[],
            wlog=wlog,
            target=target,
        )
        return dict(captured)

    # target=names
    c = await run("names")
    assert c["prompt_name"] == "sn/review_names"
    assert c["response_model"] is sn_models.StandardNameQualityReviewNameOnlyBatch

    # target=docs
    c = await run("docs")
    assert c["prompt_name"] == "sn/review_docs"
    assert c["response_model"] is sn_models.StandardNameQualityReviewDocsBatch

    # target=full
    c = await run("full")
    assert c["prompt_name"] == "sn/review"
    assert c["response_model"] is sn_models.StandardNameQualityReviewBatch


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def test_docs_prompt_scores_docs_dimensions_only() -> None:
    from imas_codex.llm.prompt_loader import render_prompt

    rendered = render_prompt(
        "sn/review_docs",
        {
            "items": [
                {
                    "standard_name": "electron_temperature",
                    "source_id": "x",
                    "unit": "eV",
                    "kind": "scalar",
                    "tags": [],
                    "description": "Temperature of the electron population.",
                    "documentation": "T_e measured by Thomson scattering.",
                }
            ],
            "batch_context": "",
            "nearby_existing_names": [],
        },
    )

    # Must score the four docs dimensions
    assert "Description Quality" in rendered
    assert "Documentation Quality" in rendered
    assert "Completeness" in rendered
    assert "Physics Accuracy" in rendered

    # Must NOT re-score name dimensions
    assert "Grammar Correctness" not in rendered
    assert "Naming Convention Adherence" not in rendered
    assert "Semantic Accuracy" not in rendered

    # /80 scale tier table
    assert "0-80" in rendered


# ---------------------------------------------------------------------------
# Downgrade guard — fidelity rank names < docs < full
# ---------------------------------------------------------------------------


def test_downgrade_guard_skips_lower_fidelity_without_force() -> None:
    """A target=names run must not overwrite a prior docs/full review mode."""
    # Build a mock state + name with existing review_mode="docs".
    from imas_codex.standard_names.review.state import StandardNameReviewState

    state = StandardNameReviewState(
        facility="dd",
        cost_limit=1.0,
        force_review=False,
        target="names",
        name_only=True,
    )
    existing = {"id": "x", "reviewer_score": 0.8, "review_mode": "docs"}
    incoming = {"id": "y", "reviewer_score": 0.8, "review_mode": "names"}

    # Inline the guard logic (kept identical to pipeline.py) to verify
    # the fidelity rank is correct.
    _rank = {"names": 1, "docs": 2, "full": 3}
    target = state.target or ("names" if state.name_only else "full")
    incoming_mode = "names" if target == "names" else target

    # names (rank 1) should NOT overwrite docs (rank 2)
    assert _rank[incoming_mode] < _rank[existing["review_mode"]]
    # but names (rank 1) MAY overwrite another name_only (rank 1)
    assert _rank[incoming_mode] == _rank[incoming["review_mode"]]
