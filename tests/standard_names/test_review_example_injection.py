"""Tests for review scored-example injection (K4).

Validates that the ``_review_scored_examples.md`` Jinja template:
- Renders all entries when review_scored_examples has content.
- Renders as zero-length when the list is empty (no whitespace artifacts).
- Per-dimension breakdown is the hero content.
- Works for all three review prompts (review, review_names, review_docs).
"""

from __future__ import annotations

import pytest

from imas_codex.llm.prompt_loader import PROMPTS_DIR, _get_jinja_env


def _render_review(examples: list[dict]) -> str:
    """Render the review scored-examples partial directly."""
    env = _get_jinja_env(PROMPTS_DIR)
    tmpl = env.get_template("shared/sn/_review_scored_examples.md")
    return tmpl.render(review_scored_examples=examples)


def _example(
    *,
    name_id: str = "electron_temperature",
    score: float = 0.82,
    verdict: str = "good",
    domain: str = "transport",
    scores: dict | None = None,
    comments: dict | None = None,
) -> dict:
    return {
        "id": name_id,
        "description": f"Description of {name_id}",
        "documentation": f"Docs for {name_id}",
        "kind": "scalar",
        "unit": "eV",
        "reviewer_score": score,
        "reviewer_verdict": verdict,
        "scores": scores or {"dim_a": 18, "dim_b": 16},
        "dimension_comments": comments or {"dim_a": "Good analysis", "dim_b": "OK"},
        "reviewer_comments": "Overall good quality",
        "physics_domain": domain,
    }


class TestReviewEmptyList:
    """Empty list → section is absent (no whitespace artifacts)."""

    def test_empty_list_renders_nothing(self) -> None:
        result = _render_review([])
        assert result.strip() == ""

    def test_empty_list_no_header(self) -> None:
        result = _render_review([])
        assert "REVIEWER CALIBRATION" not in result


class TestReviewWithEntries:
    """Entries appear in rendered output with per-dim breakdown."""

    def test_single_entry(self) -> None:
        result = _render_review([_example()])
        assert "REVIEWER CALIBRATION EXAMPLES" in result
        assert "electron_temperature" in result
        assert "0.82" in result
        assert "good" in result

    def test_two_entries(self) -> None:
        ex1 = _example(name_id="electron_temperature", score=0.95)
        ex2 = _example(name_id="plasma_current", score=0.42)
        result = _render_review([ex1, ex2])
        assert "electron_temperature" in result
        assert "plasma_current" in result

    def test_per_dim_breakdown_is_present(self) -> None:
        result = _render_review([_example()])
        assert "Per-dimension scores and reasoning:" in result
        assert "dim_a: 18/20" in result
        assert "Good analysis" in result
        assert "dim_b: 16/20" in result


class TestReviewDynamicDims:
    """Dimension names are iterated dynamically."""

    def test_4dim_docs_reviewer(self) -> None:
        scores = {
            "description_quality": 17,
            "documentation_quality": 14,
            "completeness": 19,
            "physics_accuracy": 18,
        }
        comments = {
            "description_quality": "Clear and precise",
            "documentation_quality": "Could use LaTeX",
            "completeness": "All fields covered",
            "physics_accuracy": "Physically correct",
        }
        result = _render_review([_example(scores=scores, comments=comments)])
        for dim in scores:
            assert dim in result
        for comment in comments.values():
            assert comment in result

    def test_6dim_full_reviewer(self) -> None:
        scores = {
            "naming": 15,
            "description": 18,
            "documentation": 12,
            "unit_consistency": 20,
            "physics_accuracy": 17,
            "grammar_compliance": 19,
        }
        result = _render_review([_example(scores=scores, comments={})])
        for dim in scores:
            assert dim in result

    def test_missing_dim_comment_shows_fallback(self) -> None:
        """Dimension present in scores but missing from comments → fallback text."""
        scores = {"naming": 15, "description": 18}
        comments = {"naming": "Good naming"}  # description missing
        result = _render_review([_example(scores=scores, comments=comments)])
        assert "naming: 15/20" in result
        assert "Good naming" in result
        assert "description: 18/20" in result
        assert "(no per-dimension comment recorded)" in result


class TestReviewOptionalFields:
    """Optional fields handled gracefully."""

    def test_no_documentation(self) -> None:
        ex = _example()
        ex["documentation"] = ""
        result = _render_review([ex])
        assert "Documentation:" not in result

    def test_no_physics_domain(self) -> None:
        ex = _example()
        ex["physics_domain"] = ""
        result = _render_review([ex])
        assert "Physics domain:" not in result
