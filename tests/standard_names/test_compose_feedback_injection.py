"""Phase C: compose_dd.md renders the ``item.review_feedback`` block.

These tests don't exercise the LLM — they just render the Jinja template
through ``imas_codex.llm.prompt_loader.render_prompt`` and assert that the
feedback payload is surfaced when present, and that the block is absent
otherwise.
"""

from __future__ import annotations

import pytest

from imas_codex.llm.prompt_loader import render_prompt


def _min_item(**overrides) -> dict:
    base = {
        "path": "equilibrium/time_slice/profiles_1d/psi",
        "description": "Poloidal flux",
        "data_type": "FLT_1D",
        "units": "Wb",
    }
    base.update(overrides)
    return base


def _min_context(items: list[dict]) -> dict:
    return {
        "items": items,
        "ids_name": "equilibrium",
        "ids_contexts": {},
        "existing_names": [],
        "cluster_context": None,
        "nearby_existing_names": [],
        "reference_exemplars": [],
        "cocos_version": 11,
        "dd_version": "4.0.0",
    }


class TestReviewFeedbackInjection:
    """The feedback block should appear only when item.review_feedback is set."""

    def test_block_absent_without_feedback(self) -> None:
        rendered = render_prompt("sn/compose_dd", _min_context([_min_item()]))
        assert "Prior reviewer feedback" not in rendered
        assert "Reviewer critique" not in rendered

    def test_block_present_with_feedback(self) -> None:
        fb = {
            "previous_name": "equilibrium_plasma_boundary_outline_radial_coordinate",
            "previous_description": "A very long and redundant name.",
            "reviewer_score": 0.32,
            "review_tier": "poor",
            "reviewer_comments": (
                "Name is far too long. Drop the `equilibrium_` prefix — "
                "it's implied by physics_domain. Use `_r` instead of "
                "`_radial_coordinate`."
            ),
            "reviewer_scores": {
                "grammar": 15,
                "semantic": 14,
                "documentation": 10,
                "convention": 5,
                "completeness": 12,
                "compliance": 6,
            },
            "validation_status": "valid",
        }
        item = _min_item(review_feedback=fb)
        rendered = render_prompt("sn/compose_dd", _min_context([item]))

        # Header marker
        assert "Prior reviewer feedback" in rendered
        # Previous name echoed
        assert "equilibrium_plasma_boundary_outline_radial_coordinate" in rendered
        # Tier + score
        assert "poor" in rendered
        assert "0.32" in rendered
        # Per-dimension scores (at least some)
        assert "convention" in rendered and "5" in rendered
        # Reviewer critique surfaced
        assert "Drop the `equilibrium_` prefix" in rendered
        # Instruction to fix issues present
        assert "directly fixes" in rendered or "address" in rendered

    def test_block_handles_missing_reviewer_scores(self) -> None:
        """Feedback without a reviewer_scores dict should still render."""
        fb = {
            "previous_name": "old_name",
            "reviewer_score": 0.45,
            "review_tier": "adequate",
            "reviewer_comments": "Needs sharper definition.",
            "reviewer_scores": None,
            "validation_status": "valid",
        }
        item = _min_item(review_feedback=fb)
        rendered = render_prompt("sn/compose_dd", _min_context([item]))
        assert "Prior reviewer feedback" in rendered
        assert "old_name" in rendered
        assert "Needs sharper definition." in rendered

    def test_multiple_items_only_affected_one_gets_block(self) -> None:
        clean_item = _min_item(path="equilibrium/time_slice/profiles_1d/q")
        flagged_item = _min_item(
            path="equilibrium/time_slice/profiles_1d/psi",
            review_feedback={
                "previous_name": "verbose_flux_name",
                "reviewer_score": 0.3,
                "review_tier": "poor",
                "reviewer_comments": "Too verbose.",
                "reviewer_scores": None,
                "validation_status": "valid",
            },
        )
        rendered = render_prompt(
            "sn/compose_dd", _min_context([clean_item, flagged_item])
        )
        # Feedback block appears exactly once
        assert rendered.count("Prior reviewer feedback") == 1
        assert "verbose_flux_name" in rendered


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
