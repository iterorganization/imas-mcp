"""Tests for compose scored-example injection (K4).

Validates that the ``_compose_scored_examples.md`` Jinja template:
- Renders all entries when compose_scored_examples has content.
- Renders as zero-length when the list is empty (no whitespace artifacts).
- Dynamically iterates dimension names without hardcoding them.
"""

from __future__ import annotations

import pytest

from imas_codex.llm.prompt_loader import PROMPTS_DIR, _get_jinja_env


def _render_compose(examples: list[dict]) -> str:
    """Render the compose scored-examples partial directly."""
    env = _get_jinja_env(PROMPTS_DIR)
    tmpl = env.get_template("shared/sn/_compose_scored_examples.md")
    return tmpl.render(compose_scored_examples=examples)


def _example(
    *,
    name_id: str = "electron_temperature",
    score: float = 0.82,
    verdict: str = "good",
    domain: str = "transport",
    target_score: float = 0.80,
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
        "target_score": target_score,
        "scores": scores or {"dim_a": 18, "dim_b": 16},
        "dimension_comments": comments or {"dim_a": "Good", "dim_b": "OK"},
        "reviewer_comments": "Overall good quality",
        "physics_domain": domain,
    }


class TestComposeEmptyList:
    """Empty list → section is absent (no whitespace artifacts)."""

    def test_empty_list_renders_nothing(self) -> None:
        result = _render_compose([])
        assert result.strip() == ""

    def test_empty_list_no_header(self) -> None:
        result = _render_compose([])
        assert "SCORED EXAMPLES" not in result


class TestComposeWithEntries:
    """Entries appear in rendered output."""

    def test_single_entry(self) -> None:
        result = _render_compose([_example()])
        assert "SCORED EXAMPLES" in result
        assert "electron_temperature" in result
        assert "0.82" in result
        assert "good" in result

    def test_two_entries(self) -> None:
        ex1 = _example(name_id="electron_temperature", score=0.95, target_score=0.90)
        ex2 = _example(name_id="plasma_current", score=0.42, target_score=0.40)
        result = _render_compose([ex1, ex2])
        assert "electron_temperature" in result
        assert "plasma_current" in result
        assert "0.95" in result
        assert "0.42" in result
        # High target_score → ✅ EMULATE; low target_score → ⚠️ AVOID
        assert "EMULATE" in result
        assert "AVOID" in result


class TestComposeDynamicDims:
    """Dimension names are iterated dynamically, not hardcoded."""

    def test_4dim_reviewer(self) -> None:
        scores = {
            "description_quality": 17,
            "documentation_quality": 14,
            "completeness": 19,
            "physics_accuracy": 18,
        }
        comments = {
            "description_quality": "Clear",
            "documentation_quality": "Could improve",
            "completeness": "All covered",
            "physics_accuracy": "Correct",
        }
        result = _render_compose([_example(scores=scores, comments=comments)])
        for dim in scores:
            assert dim in result
        for comment in comments.values():
            assert comment in result

    def test_6dim_reviewer(self) -> None:
        scores = {
            "naming": 15,
            "description": 18,
            "documentation": 12,
            "unit_consistency": 20,
            "physics_accuracy": 17,
            "grammar_compliance": 19,
        }
        result = _render_compose([_example(scores=scores, comments={})])
        for dim in scores:
            assert dim in result


class TestComposeOptionalFields:
    """Optional fields are handled gracefully."""

    def test_no_documentation(self) -> None:
        ex = _example()
        ex["documentation"] = ""
        result = _render_compose([ex])
        assert "Documentation:" not in result

    def test_no_reviewer_comments(self) -> None:
        ex = _example()
        ex["reviewer_comments"] = ""
        result = _render_compose([ex])
        assert "Reviewer summary:" not in result

    def test_no_physics_domain(self) -> None:
        ex = _example()
        ex["physics_domain"] = ""
        result = _render_compose([ex])
        assert "Physics domain:" not in result
