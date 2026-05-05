"""W37: review_names.md user prompt MUST render compose-parity per-item context.

Background: Until W37, the review prompt only showed ``dd_source_docs``,
``nearest_peers``, and (filtered) ``related_neighbours``. The composer received
11+ context channels per item but the reviewer did not — leading to "revise"
verdicts that flagged issues already addressed by context the reviewer never
saw. W37 brings the reviewer to compose parity AND adds suggested_name +
suggestion_justification to the response model.

This test guards against regression by asserting:

1. Every per-item context block from the compose prompt also renders in the
   review prompt when populated.
2. Empty context channels do NOT inject orphan headers.
3. The Pydantic response model accepts both ``suggested_name=None`` (accept
   verdict) and a populated ``suggested_name`` + ``suggestion_justification``.
"""

from __future__ import annotations

import pytest

from imas_codex.llm.prompt_loader import render_prompt

REVIEW_TEMPLATE = "sn/review_names"


def _full_review_item() -> dict:
    """Build a review item with every compose context channel populated."""
    return {
        "id": "core_profiles__electron_temperature",
        "source_id": "core_profiles/profiles_1d/electrons/temperature",
        "standard_name": "electron_temperature",
        "unit": "eV",
        "kind": "scalar",
        "grammar_fields": {"physical_base": "temperature", "subject": "electron"},
        "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
        "validation_issues": [],
        # Compose-parity context channels
        "dd_source_docs": [
            {
                "id": "core_profiles/profiles_1d/electrons/temperature",
                "unit": "eV",
                "documentation": "Electron temperature profile",
                "description": "Electron temperature",
            }
        ],
        "data_type": "FLT_1D",
        "node_type": "dynamic",
        "physics_domain": "core_plasma_physics",
        "ndim": 1,
        "lifecycle_status": "active",
        "cocos_label": None,
        "parent_path": "core_profiles/profiles_1d/electrons",
        "parent_description": "Per-electron-species profiles",
        "previous_name": {
            "name": "electron_temperature_old",
            "pipeline_status": "drafted",
        },
        "identifier_schema": "magnetics_probe_type_identifier",
        "identifier_schema_doc": "MagDiag probe taxonomy",
        "identifier_values": [
            {"name": "flux_loop", "index": 1, "description": "FL"},
        ],
        "clusters": [
            {
                "label": "core_kinetic_temperature",
                "scope": "global",
                "description": "Core kinetic temperatures across IDSs",
            }
        ],
        "cross_ids_paths": [
            "edge_profiles/profiles_1d/electrons/temperature",
        ],
        "hybrid_neighbours": [
            {
                "tag": "ion_temperature",
                "unit": "eV",
                "physics_domain": "core_plasma_physics",
                "doc_short": "Ion temperature profile",
                "cocos_label": None,
            }
        ],
        "related_neighbours": [
            {
                "path": "edge_profiles/profiles_1d/electrons/temperature",
                "ids": "edge_profiles",
                "relationship_type": "cluster",
                "via": "core_kinetic_temperature",
            }
        ],
        "error_fields": [
            "core_profiles/profiles_1d/electrons/temperature_error_upper",
        ],
        "sibling_fields": [
            {
                "path": "core_profiles/profiles_1d/electrons/density",
                "description": "Electron density profile",
                "data_type": "FLT_1D",
            }
        ],
        "version_history": [
            {"version": "3.39.0", "change_type": "definition_clarification"},
        ],
        "review_feedback": {
            "previous_name": "electron_temperature_old",
            "reviewer_score": 0.42,
            "review_tier": "inadequate",
            "reviewer_comments": "Too generic; missing locus distinguisher.",
        },
    }


def _render_review_prompt(items: list[dict] | None = None, **extra) -> str:
    context = {
        "items": items if items is not None else [_full_review_item()],
        "facility": "iter",
        "physics_domain": "core_plasma_physics",
        "nearby_existing_names": [],
        "review_scored_examples": [],
        "batch_context": "",
        "audit_findings": [],
        "prior_reviews": [],
        # grammar enums (defaulted empty for template safety)
        "subjects": [],
        "components": [],
        "positions": [],
        "operators": [],
        "statistics": [],
        "processes": [],
        "coordinates": [],
        "references": [],
        **extra,
    }
    return render_prompt(REVIEW_TEMPLATE, context)


# ---------------------------------------------------------------------------
# Per-item context channels MUST appear when populated (compose parity)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "marker",
    [
        # previous_name
        "Previous generation",
        "electron_temperature_old",
        # identifier
        "Identifier schema",
        "magnetics_probe_type_identifier",
        "flux_loop",
        # clusters
        "Semantic clusters",
        "core_kinetic_temperature",
        # cross-IDS
        "Cross-IDS equivalents",
        "edge_profiles/profiles_1d/electrons/temperature",
        # hybrid neighbours
        "Hybrid-search neighbours",
        "ion_temperature",
        # related neighbours
        "Graph-relationship neighbours",
        # error fields
        "DD error companions",
        "temperature_error_upper",
        # sibling fields
        "Sibling fields",
        "electrons/density",
        # version history
        "DD version history",
        "definition_clarification",
        # parent description
        "Per-electron-species profiles",
        # review feedback (regen scenario)
        "Prior reviewer feedback",
        "Too generic; missing locus distinguisher.",
    ],
)
def test_review_prompt_renders_compose_parity_context(marker: str) -> None:
    rendered = _render_review_prompt()
    assert marker in rendered, (
        f"Review prompt missing context marker '{marker}'. "
        f"W37 requires compose-parity per-item context in review_names.md."
    )


# ---------------------------------------------------------------------------
# Suggested-name policy must appear in the rendered prompt
# ---------------------------------------------------------------------------


def test_review_prompt_includes_suggestion_policy() -> None:
    rendered = _render_review_prompt()
    assert "Suggested-Name Policy" in rendered
    assert "suggested_name" in rendered
    assert "suggestion_justification" in rendered


# ---------------------------------------------------------------------------
# Empty context channels MUST NOT leak orphan headers
# ---------------------------------------------------------------------------


def test_review_prompt_minimal_item_no_orphan_headers() -> None:
    minimal = {
        "id": "x",
        "source_id": "core_profiles/time",
        "standard_name": "time",
        "unit": "s",
        "kind": "scalar",
        "grammar_fields": {},
        "source_paths": [],
        "validation_issues": [],
    }
    rendered = _render_review_prompt(items=[minimal])
    for header in (
        "Hybrid-search neighbours",
        "Semantic clusters",
        "DD error companions",
        "Graph-relationship neighbours",
        "Identifier schema",
        "Sibling fields",
        "Prior reviewer feedback",
        "Cross-IDS equivalents",
        "DD version history",
    ):
        assert header not in rendered, (
            f"Empty-context render leaked review header '{header}'."
        )


# ---------------------------------------------------------------------------
# Pydantic model: suggested_name + suggestion_justification accept None
# ---------------------------------------------------------------------------


def test_review_pydantic_accepts_null_suggestion() -> None:
    from imas_codex.standard_names.models import StandardNameQualityReviewNameOnly

    payload = {
        "source_id": "core_profiles/profiles_1d/electrons/temperature",
        "standard_name": "electron_temperature",
        "scores": {"grammar": 20, "semantic": 18, "convention": 19, "completeness": 18},
        "reasoning": "Clean grammar, accurate physics.",
        "revised_name": None,
        "revised_fields": None,
        "suggested_name": None,
        "suggestion_justification": None,
        "issues": [],
    }
    review = StandardNameQualityReviewNameOnly(**payload)
    assert review.suggested_name is None
    assert review.suggestion_justification is None
    assert review.scores.tier in {"outstanding", "good", "inadequate", "poor"}


def test_review_pydantic_accepts_populated_suggestion() -> None:
    from imas_codex.standard_names.models import StandardNameQualityReviewNameOnly

    payload = {
        "source_id": "core_profiles/profiles_1d/electrons/temperature",
        "standard_name": "Te_core",
        "scores": {"grammar": 8, "semantic": 14, "convention": 6, "completeness": 12},
        "reasoning": "Symbol abbreviation; missing locus distinguisher.",
        "revised_name": "electron_temperature_core",
        "revised_fields": None,
        "suggested_name": "electron_temperature_core",
        "suggestion_justification": (
            "Original used the symbol abbreviation 'Te'; cluster siblings "
            "show all related quantities use full 'electron_temperature' with "
            "a '_core' locus suffix for inner-flux-surface variants."
        ),
        "issues": ["abbreviation"],
    }
    review = StandardNameQualityReviewNameOnly(**payload)
    assert review.suggested_name == "electron_temperature_core"
    assert (
        review.suggestion_justification
        and "cluster siblings" in review.suggestion_justification
    )


def test_review_batch_parses_full_sample() -> None:
    """Reviewer-style batch JSON parses cleanly with new fields."""
    from imas_codex.standard_names.models import (
        StandardNameQualityReviewNameOnlyBatch,
    )

    sample = {
        "reviews": [
            {
                "source_id": "p1",
                "standard_name": "good_name",
                "scores": {
                    "grammar": 20,
                    "semantic": 18,
                    "convention": 19,
                    "completeness": 18,
                },
                "reasoning": "All four dims clear.",
                "suggested_name": None,
                "suggestion_justification": None,
            },
            {
                "source_id": "p2",
                "standard_name": "bad_name",
                "scores": {
                    "grammar": 8,
                    "semantic": 12,
                    "convention": 6,
                    "completeness": 10,
                },
                "reasoning": "Multiple defects.",
                "revised_name": "better_name",
                "suggested_name": "better_name",
                "suggestion_justification": "Cluster siblings use that form.",
            },
        ]
    }
    batch = StandardNameQualityReviewNameOnlyBatch(**sample)
    assert len(batch.reviews) == 2
    assert batch.reviews[0].suggested_name is None
    assert batch.reviews[1].suggested_name == "better_name"


def test_prior_reviews_iteration_uses_dict_index() -> None:
    """W37: prior_reviews[i].items must use dict indexing, not Jinja attribute access.

    Jinja2 resolves ``pr.items`` on a dict to the bound ``dict.items`` method
    (a builtin_function_or_method), not the value at key ``'items'``. This
    triggers ``TypeError: 'builtin_function_or_method' object is not iterable``
    when Jinja tries ``{% for x in pr.items %}``.

    The escalator (RD-quorum cycle 2) populates ``prior_reviews`` with dicts
    that have an ``items`` key, so the bug fired silently across docs review
    in the W37 Set B rotation, suppressing layer-2 docs scores. Both
    ``review_names.md`` and ``review_docs.md`` must use ``pr['items']``.
    """
    base_ctx = {
        "items": [],
        "existing_names": [],
        "review_scored_examples": [],
        "batch_context": "",
        "nearby_existing_names": [],
        "audit_findings": [],
        "grammar_segments": [],
        "grammar_tokens_by_segment": {},
        "prior_reviews": [
            {
                "role": "primary",
                "model": "anthropic/claude-opus-4.6",
                "items": [
                    {
                        "standard_name": "ion_pressure",
                        "score": 0.95,
                        "tier": "outstanding",
                        "scores_json": "{}",
                        "comments_per_dim_json": "{}",
                        "reasoning": "Strong grammar.",
                    }
                ],
            }
        ],
    }
    for template in ("sn/review_names", "sn/review_docs"):
        out = render_prompt(template, base_ctx)
        assert "ion_pressure" in out, (
            f"{template}: prior_reviews block did not render — bug regressed"
        )
        assert "Strong grammar" in out, (
            f"{template}: reasoning text missing from prior_reviews block"
        )
