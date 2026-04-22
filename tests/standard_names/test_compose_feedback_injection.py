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
            "review_tier": "inadequate",
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


class TestRelatedNeighboursInjection:
    """The related-neighbours block renders when item.related_neighbours is set."""

    def test_block_absent_without_related(self) -> None:
        rendered = render_prompt("sn/compose_dd", _min_context([_min_item()]))
        assert "Graph-relationship neighbours" not in rendered

    def test_block_present_with_related(self) -> None:
        related = [
            {
                "path": "core_profiles/profiles_1d/grid/psi",
                "ids": "core_profiles",
                "relationship_type": "cluster",
                "via": "Poloidal magnetic flux profile",
            },
            {
                "path": "mhd_linear/time_slice/toroidal_mode/psi",
                "ids": "mhd_linear",
                "relationship_type": "coordinate",
                "via": "psi",
            },
        ]
        item = _min_item(related_neighbours=related)
        rendered = render_prompt("sn/compose_dd", _min_context([item]))

        assert "Graph-relationship neighbours" in rendered
        assert "core_profiles/profiles_1d/grid/psi" in rendered
        assert "core_profiles" in rendered
        assert "cluster" in rendered
        assert "Poloidal magnetic flux profile" in rendered
        assert "mhd_linear" in rendered
        assert "coordinate" in rendered

    def test_both_hybrid_and_related_render(self) -> None:
        """Hybrid and related are parallel channels — both should appear."""
        hybrid = [
            {
                "tag": "name:electron_temperature",
                "unit": "eV",
                "physics_domain": "transport",
                "doc_short": "Electron temperature",
                "cocos_label": "",
            }
        ]
        related = [
            {
                "path": "core_profiles/profiles_1d/electrons/temperature",
                "ids": "core_profiles",
                "relationship_type": "unit",
                "via": "eV",
            }
        ]
        item = _min_item(hybrid_neighbours=hybrid, related_neighbours=related)
        rendered = render_prompt("sn/compose_dd", _min_context([item]))

        assert "Hybrid-search neighbours" in rendered
        assert "Graph-relationship neighbours" in rendered


class TestErrorFieldsInjection:
    """The error-companions block renders when item.error_fields is set."""

    def test_block_absent_without_error_fields(self) -> None:
        rendered = render_prompt("sn/compose_dd", _min_context([_min_item()]))
        assert "DD error companions" not in rendered

    def test_block_present_with_error_fields(self) -> None:
        error_fields = [
            "equilibrium/time_slice/profiles_1d/psi_error_upper",
            "equilibrium/time_slice/profiles_1d/psi_error_lower",
        ]
        item = _min_item(error_fields=error_fields)
        rendered = render_prompt("sn/compose_dd", _min_context([item]))

        assert "DD error companions" in rendered
        assert "psi_error_upper" in rendered
        assert "psi_error_lower" in rendered
        assert "uncertainty" in rendered.lower()

    def test_block_absent_with_empty_error_fields(self) -> None:
        item = _min_item(error_fields=[])
        rendered = render_prompt("sn/compose_dd", _min_context([item]))
        assert "DD error companions" not in rendered


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
