"""Regression: names-mode user prompt MUST render per-item DD context.

Background: ``generate_name_dd_names.md`` is the user prompt selected when
``--target names`` is set. Until W36 it silently dropped ALL per-item context
that ``workers._enrich_items`` computes (hybrid_neighbours, related_neighbours,
error_fields, identifier_values, clusters, cross_ids_paths, sibling_fields,
review_feedback, previous_name) and dropped the batch-level
``reference_exemplars`` block. The rich-context full-mode prompt
``compose_dd.md`` rendered them. This silent stripping was the dominant
chronic-low-scores regressor in --target names rotations.

This test guards against a regression by asserting that every per-item context
block AND the batch-level reference_exemplars block appear in the rendered
names-mode prompt when populated.
"""

from __future__ import annotations

import pytest

from imas_codex.llm.prompt_loader import PROMPTS_DIR, render_prompt

NAMES_TEMPLATE = "sn/generate_name_dd_names"
NAMES_TEMPLATE_PATH = PROMPTS_DIR / "sn" / "generate_name_dd_names.md"


def _full_item() -> dict:
    """Build an item dict with every context channel populated."""
    return {
        "path": "core_profiles/profiles_1d/electrons/temperature",
        "description": "Electron temperature profile",
        "unit": "eV",
        "data_type": "FLT_1D",
        "node_type": "dynamic",
        "physics_domain": "core_plasma_physics",
        "ndim": 1,
        "lifecycle_status": "active",
        "keywords": ["electron", "temperature", "profile"],
        "cocos_label": None,
        "parent_path": "core_profiles/profiles_1d/electrons",
        "previous_name": {
            "name": "electron_temperature_old",
            "pipeline_status": "drafted",
        },
        "identifier_schema": "magnetics_probe_type_identifier",
        "identifier_schema_doc": "MagDiag probe taxonomy",
        "identifier_values": [
            {"name": "flux_loop", "index": 1, "description": "FL"},
            {"name": "rogowski", "index": 2, "description": "RC"},
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
            "summary/local/core/t_e",
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
        "review_feedback": {
            "previous_name": "electron_temperature_old",
            "reviewer_score": 0.42,
            "review_tier": "inadequate",
            "reviewer_comments": "Too generic; missing locus distinguisher.",
        },
    }


def _render_names_prompt(items: list[dict] | None = None, **extra) -> str:
    context = {
        "items": items if items is not None else [_full_item()],
        "facility": "iter",
        "physics_domain": "core_plasma_physics",
        "nearby_existing_names": [],
        "reference_exemplars": [],
        **extra,
    }
    return render_prompt(NAMES_TEMPLATE, context)


# ---------------------------------------------------------------------------
# Per-item context channels MUST appear when populated
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
        # review feedback
        "Prior reviewer feedback",
        "Too generic; missing locus distinguisher.",
    ],
)
def test_names_mode_renders_per_item_context(marker: str) -> None:
    rendered = _render_names_prompt()
    assert marker in rendered, (
        f"Names-mode prompt missing context marker '{marker}'. "
        f"This is the W36 regression: generate_name_dd_names.md silently strips "
        f"per-item DD context. Re-port the relevant Jinja block from "
        f"compose_dd.md."
    )


# ---------------------------------------------------------------------------
# Batch-level reference_exemplars MUST be rendered when populated
# ---------------------------------------------------------------------------


def test_names_mode_renders_reference_exemplars() -> None:
    exemplars = [
        {
            "name": "electron_thermal_velocity",
            "description": "Electron thermal velocity",
            "unit": "m.s^-1",
        }
    ]
    rendered = _render_names_prompt(reference_exemplars=exemplars)
    assert "REFERENCE EXEMPLARS" in rendered
    assert "electron_thermal_velocity" in rendered
    assert "Electron thermal velocity" in rendered


# ---------------------------------------------------------------------------
# Empty context MUST NOT inject blank section headers
# ---------------------------------------------------------------------------


def test_names_mode_minimal_item_renders_without_context_sections() -> None:
    minimal = {
        "path": "core_profiles/time",
        "description": "Time base",
        "unit": "s",
        "data_type": "FLT_1D",
        "node_type": "dynamic",
        "physics_domain": "core_plasma_physics",
    }
    rendered = _render_names_prompt(items=[minimal])
    # When no per-item context is populated, those section headers must NOT
    # appear (otherwise we waste tokens on empty stubs).
    for header in (
        "Hybrid-search neighbours",
        "Semantic clusters",
        "DD error companions",
        "Graph-relationship neighbours",
        "Identifier schema",
        "Sibling fields",
        "Prior reviewer feedback",
    ):
        assert header not in rendered, f"Empty-context render leaked header '{header}'."
