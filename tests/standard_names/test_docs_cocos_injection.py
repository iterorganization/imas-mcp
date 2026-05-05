"""Tests for COCOS context injection into docs generation enrichment.

Verifies that ``_enrich_for_docs_gen`` injects ``cocos_label`` and
``cocos_guidance`` from the StandardName node's ``cocos_transformation_type``
field, and that the docs user prompt renders the COCOS section.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── Helpers ─────────────────────────────────────────────────────────────


def _make_gc(*, cocos_label: str | None = None) -> MagicMock:
    """Return a mock GraphClient whose query() returns enrichment rows."""
    gc = MagicMock()

    def _query(cypher, **kwargs):
        if "sn.cocos_transformation_type" in cypher or "cocos_label" in cypher:
            return [
                {
                    "source_paths": ["dd:equilibrium/time_slice/profiles_1d/psi"],
                    "cocos_label": cocos_label,
                    "dd_nodes": [
                        {
                            "id": "equilibrium/time_slice/profiles_1d/psi",
                            "documentation": "Poloidal flux",
                            "description": "Poloidal flux",
                            "alias": None,
                            "unit": "Wb",
                        }
                    ],
                }
            ]
        return []

    gc.query = _query
    return gc


# ── Tests ───────────────────────────────────────────────────────────────


def test_cocos_label_injected():
    """When SN has cocos_transformation_type, it appears on item."""
    from imas_codex.standard_names.workers import _enrich_for_docs_gen

    items = [{"id": "poloidal_flux"}]
    gc = _make_gc(cocos_label="psi_like")
    _enrich_for_docs_gen(gc, items)
    assert items[0].get("cocos_label") == "psi_like"


def test_cocos_guidance_rendered_with_params():
    """When cocos_params are provided, guidance is rendered."""
    from imas_codex.standard_names.workers import _enrich_for_docs_gen

    items = [{"id": "poloidal_flux"}]
    gc = _make_gc(cocos_label="psi_like")
    cocos_params = {
        "sigma_bp": 1,
        "e_bp": 0,
        "sigma_r_phi_z": 1,
        "sigma_rho_theta_phi": 1,
        "psi_increasing_outward": True,
        "phi_increasing_ccw": True,
    }
    _enrich_for_docs_gen(gc, items, cocos_params=cocos_params)
    assert items[0].get("cocos_label") == "psi_like"
    guidance = items[0].get("cocos_guidance", "")
    assert guidance  # non-empty rendered guidance


def test_no_cocos_when_label_null():
    """When SN has no cocos_transformation_type, cocos_label is not set."""
    from imas_codex.standard_names.workers import _enrich_for_docs_gen

    items = [{"id": "electron_temperature"}]
    gc = _make_gc(cocos_label=None)
    _enrich_for_docs_gen(gc, items)
    assert "cocos_label" not in items[0]
    assert "cocos_guidance" not in items[0]


def test_no_cocos_guidance_without_params():
    """When cocos_params not provided, only label is set (no guidance)."""
    from imas_codex.standard_names.workers import _enrich_for_docs_gen

    items = [{"id": "poloidal_flux"}]
    gc = _make_gc(cocos_label="psi_like")
    _enrich_for_docs_gen(gc, items, cocos_params=None)
    assert items[0].get("cocos_label") == "psi_like"
    assert "cocos_guidance" not in items[0]


def test_docs_prompt_renders_cocos_section():
    """The generate_docs_user prompt includes COCOS section when label present."""
    from imas_codex.llm.prompt_loader import render_prompt

    context = {
        "item": {
            "name": "poloidal_flux",
            "unit": "Wb",
            "kind": "scalar",
            "physics_domain": "equilibrium",
            "cocos_label": "psi_like",
            "cocos_guidance": "Positive when poloidal flux increases outward.",
        },
        "chain_history": [],
        "nearby_existing_names": [],
    }
    rendered = render_prompt("sn/generate_docs_user", context)
    assert "COCOS Sign Convention" in rendered
    assert "psi_like" in rendered
    assert "Positive when poloidal flux increases outward" in rendered
    assert "MUST" in rendered


def test_docs_prompt_omits_cocos_when_absent():
    """The generate_docs_user prompt has no COCOS section when label absent."""
    from imas_codex.llm.prompt_loader import render_prompt

    context = {
        "item": {
            "name": "electron_temperature",
            "unit": "eV",
            "kind": "scalar",
            "physics_domain": "core_profiles",
        },
        "chain_history": [],
        "nearby_existing_names": [],
    }
    rendered = render_prompt("sn/generate_docs_user", context)
    assert "COCOS Sign Convention" not in rendered
