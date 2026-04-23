"""Tests for sn review --target=names mode.

Covers:
* StandardNameQualityScoreNameOnly total/score/tier arithmetic.
* ``_match_reviews_to_entries`` stamping ``review_mode`` correctly.
* Downgrade guard in the extract phase filter chain.
* Prompt rendering: name-only prompt omits documentation/compliance
  scoring sections and has a /80 tier table.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# StandardNameQualityScoreNameOnly — arithmetic
# ---------------------------------------------------------------------------


def test_name_only_score_total_and_tier() -> None:
    from imas_codex.standard_names.models import StandardNameQualityScoreNameOnly

    s = StandardNameQualityScoreNameOnly(
        grammar=20, semantic=18, convention=19, completeness=18
    )
    assert s.total == 75
    assert s.score == pytest.approx(75 / 80)
    assert s.tier == "outstanding"

    good = StandardNameQualityScoreNameOnly(
        grammar=15, semantic=14, convention=14, completeness=14
    )
    # 57/80 = 0.7125 → good
    assert good.tier == "good"

    outstanding = StandardNameQualityScoreNameOnly(
        grammar=20, semantic=20, convention=20, completeness=20
    )
    assert outstanding.score == 1.0
    assert outstanding.tier == "outstanding"

    poor = StandardNameQualityScoreNameOnly(
        grammar=5, semantic=5, convention=5, completeness=5
    )
    assert poor.tier == "poor"

    inadequate = StandardNameQualityScoreNameOnly(
        grammar=10, semantic=10, convention=10, completeness=10
    )
    assert inadequate.tier == "inadequate"


def test_name_only_batch_model_parses_minimal_json() -> None:
    from imas_codex.standard_names.models import (
        StandardNameQualityReviewNameOnlyBatch,
    )

    batch = StandardNameQualityReviewNameOnlyBatch.model_validate(
        {
            "reviews": [
                {
                    "source_id": "foo",
                    "standard_name": "electron_temperature",
                    "scores": {
                        "grammar": 20,
                        "semantic": 18,
                        "convention": 19,
                        "completeness": 18,
                    },
                    "verdict": "accept",
                    "reasoning": "ok",
                }
            ]
        }
    )
    assert batch.reviews[0].scores.total == 75
    assert batch.reviews[0].scores.tier == "outstanding"


# ---------------------------------------------------------------------------
# _match_reviews_to_entries — review_mode stamping
# ---------------------------------------------------------------------------


def _make_review(score_cls, source_id: str, name: str, **scores: int):
    from imas_codex.standard_names.models import StandardNameReviewVerdict

    return SimpleNamespace(
        source_id=source_id,
        standard_name=name,
        scores=score_cls(**scores),
        verdict=StandardNameReviewVerdict.accept,
        reasoning="test",
        revised_name=None,
        revised_fields=None,
    )


def test_match_stamps_name_only_mode() -> None:
    from imas_codex.standard_names.models import StandardNameQualityScoreNameOnly
    from imas_codex.standard_names.review.pipeline import _match_reviews_to_entries

    wlog = logging.LoggerAdapter(logging.getLogger("test"), {})
    names = [{"id": "electron_temperature", "source_id": "electron_temperature"}]
    reviews = [
        _make_review(
            StandardNameQualityScoreNameOnly,
            source_id="electron_temperature",
            name="electron_temperature",
            grammar=20,
            semantic=18,
            convention=19,
            completeness=18,
        )
    ]

    scored, unmatched, revised = _match_reviews_to_entries(
        reviews, names, wlog, name_only=True
    )
    assert len(scored) == 1
    assert unmatched == []
    assert scored[0]["review_mode"] == "names"
    assert scored[0]["reviewer_score"] == pytest.approx(75 / 80)


def test_match_stamps_full_mode_by_default() -> None:
    from imas_codex.standard_names.models import StandardNameQualityScore
    from imas_codex.standard_names.review.pipeline import _match_reviews_to_entries

    wlog = logging.LoggerAdapter(logging.getLogger("test"), {})
    names = [{"id": "electron_temperature", "source_id": "electron_temperature"}]
    reviews = [
        _make_review(
            StandardNameQualityScore,
            source_id="electron_temperature",
            name="electron_temperature",
            grammar=20,
            semantic=18,
            documentation=16,
            convention=19,
            completeness=18,
            compliance=17,
        )
    ]

    scored, _unmatched, _revised = _match_reviews_to_entries(reviews, names, wlog)
    assert scored[0]["review_mode"] == "full"


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def test_name_only_prompt_omits_documentation_and_compliance() -> None:
    from imas_codex.llm.prompt_loader import render_prompt

    rendered = render_prompt(
        "sn/review_names",
        {
            "items": [
                {
                    "standard_name": "electron_temperature",
                    "source_id": "x",
                    "unit": "eV",
                    "kind": "scalar",
                    "tags": [],
                    "grammar_fields": {},
                }
            ],
            "batch_context": "",
            "nearby_existing_names": [],
        },
    )

    # Headline rubric must not mention documentation/compliance scoring.
    assert "Documentation Quality" not in rendered
    assert "Prompt Compliance" not in rendered

    # Must use the /80 scale in the tier table.
    assert "0-80" in rendered

    # Must list only the four retained dimensions.
    assert "Grammar Correctness" in rendered
    assert "Semantic Accuracy" in rendered
    assert "Naming Convention Adherence" in rendered
    assert "Completeness" in rendered


def test_full_prompt_still_renders_six_dimensions() -> None:
    from imas_codex.llm.prompt_loader import render_prompt

    rendered = render_prompt(
        "sn/review",
        {
            "items": [],
            "review_scored_examples": [],
            "existing_names": [],
            "batch_context": "",
            "nearby_existing_names": [],
            "audit_findings": [],
        },
    )
    assert "Documentation Quality" in rendered
    assert "0-120" in rendered
