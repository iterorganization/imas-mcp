"""Verify reviewer feedback fields are rendered into compose prompts.

Regression suite for two graph→template orphans found alongside the
``reviewer_suggested_name`` canary (commit a5ce32da):

1. ``previous_documentation`` is fetched by ``fetch_review_feedback_for_sources``
   and packed into the per-item ``review_feedback`` mapping (graph_ops.py),
   but the compose templates omitted it — regen LLM never saw the prior doc.
2. ``compose_dd_names.md`` did not render the per-dimension reviewer score
   breakdown, even though ``compose_dd.md`` did and the data was already in
   the item dict. The lean naming prompt was missing dimension-targeted
   feedback.
"""

from __future__ import annotations

from imas_codex.llm.prompt_loader import render_prompt

_PRIOR_DOC_MARKER = "PRIOR_DOC_MARKER_FOR_REGRESSION"


def _make_item() -> dict:
    return {
        "path": "equilibrium/time_slice/profiles_1d/psi",
        "ids_name": "equilibrium",
        "description": "Poloidal flux",
        "unit": "Wb",
        "data_type": "FLT_1D",
        "physics_domain": "equilibrium",
        "parent_path": "equilibrium/time_slice/profiles_1d",
        "parent_description": "1D profiles",
        "parent_type": "STRUCTURE",
        "review_feedback": {
            "previous_name": "poloidal_flux_old",
            "previous_description": "Old description text.",
            "previous_documentation": _PRIOR_DOC_MARKER
            + " — multi-paragraph prior doc",
            "reviewer_score": 0.45,
            "review_tier": "inadequate",
            "reviewer_comments": "Name lacks locus distinguisher.",
            "reviewer_scores": {
                "grammar": 12,
                "semantic": 10,
                "convention": 14,
                "completeness": 9,
            },
            "reviewer_suggested_name": "poloidal_magnetic_flux",
            "reviewer_suggestion_justification": "Cluster siblings use _magnetic_.",
        },
    }


def _render(template_path: str) -> str:
    ctx = {
        "items": [_make_item()],
        "ids_contexts": [],
        "reference_exemplars": [],
        "nearby_existing_names": [],
    }
    return render_prompt(template_path, ctx)


def test_compose_dd_renders_previous_documentation():
    """compose_dd.md must surface review_feedback.previous_documentation."""
    rendered = _render("sn/generate_name_dd")
    assert _PRIOR_DOC_MARKER in rendered, (
        "previous_documentation absent from compose_dd.md — regen composer "
        "cannot see what was wrong with the prior documentation"
    )
    # Sanity: existing fields still render
    assert "poloidal_flux_old" in rendered
    assert "poloidal_magnetic_flux" in rendered


def test_compose_dd_names_renders_previous_documentation():
    """compose_dd_names.md (lean naming prompt) must also surface prior doc."""
    rendered = _render("sn/generate_name_dd_names")
    assert _PRIOR_DOC_MARKER in rendered, (
        "previous_documentation absent from compose_dd_names.md"
    )


def test_compose_dd_names_renders_per_dimension_scores():
    """compose_dd_names.md must render reviewer_scores per-dimension breakdown.

    Without this, the lean naming prompt only sees the aggregate score and
    free-form comments — it cannot target the specific failing rubric
    dimension (grammar/semantic/convention/completeness).
    """
    rendered = _render("sn/generate_name_dd_names")
    # All four dimensions from the test fixture must appear with their scores
    for dim, score in (
        ("grammar", 12),
        ("semantic", 10),
        ("convention", 14),
        ("completeness", 9),
    ):
        assert f"`{dim}`" in rendered, (
            f"per-dim label `{dim}` missing from compose_dd_names.md render"
        )
        assert str(score) in rendered, (
            f"per-dim score {score} for `{dim}` missing from compose_dd_names.md"
        )


def test_compose_dd_per_dim_scores_still_render():
    """Regression guard: the existing compose_dd.md per-dim block must keep working."""
    rendered = _render("sn/generate_name_dd")
    assert "grammar" in rendered and "12" in rendered
    assert "semantic" in rendered and "10" in rendered
