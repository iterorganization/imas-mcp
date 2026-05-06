"""Tests for derivative family context injection in _enrich_for_docs_gen.

Covers:
- T12: Base quantity dict populated when item has COMPONENT_OF parent edge
- T13: Derivative siblings list populated when multiple derivatives share a parent
- T14: DD_DERIVATIVE_MAP lookup for dpsi_drho_tor resolves correctly
"""

from __future__ import annotations

from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gc(query_side_effects: list) -> MagicMock:
    """Return a mock GraphClient whose .query() returns successive values."""
    gc = MagicMock()
    gc.query = MagicMock(side_effect=query_side_effects)
    return gc


def _make_item(sn_id: str, **overrides) -> dict:
    item: dict = {"id": sn_id, "description": f"Test description for {sn_id}"}
    item.update(overrides)
    return item


# Query return values used across tests
_EMPTY_ENRICH = [{"source_paths": [], "cocos_label": None, "dd_nodes": []}]
_NO_CHILDREN: list = []


# ---------------------------------------------------------------------------
# T12 – base quantity context injection
# ---------------------------------------------------------------------------


def test_t12_base_quantity_context_injection():
    """When a StandardName has a COMPONENT_OF parent, base_quantity is injected."""
    from imas_codex.standard_names.workers import _enrich_for_docs_gen

    gc = _make_gc(
        [
            # 1. _DOCS_GEN_ENRICH_QUERY
            _EMPTY_ENRICH,
            # 2. Section 4: parent COMPONENT_OF query
            [
                {
                    "name": "pressure",
                    "description": "Plasma pressure",
                    "documentation": "The kinetic pressure.",
                    "axis": None,
                }
            ],
            # 3. Section 4: child COMPONENT_OF query
            _NO_CHILDREN,
            # 4. Section 5: parent unit + siblings
            [{"unit": "Pa", "siblings": []}],
        ]
    )

    item = _make_item("dpressure_dpsi")
    _enrich_for_docs_gen(gc, [item])

    assert "base_quantity" in item, "base_quantity should be injected"
    bq = item["base_quantity"]
    assert bq["name"] == "pressure"
    assert bq["unit"] == "Pa"
    assert bq["description"] == "Plasma pressure"
    assert "The kinetic pressure." in bq["documentation"]


def test_t12_base_quantity_not_injected_without_parent():
    """When there is no COMPONENT_OF parent, base_quantity is not added."""
    from imas_codex.standard_names.workers import _enrich_for_docs_gen

    gc = _make_gc(
        [
            _EMPTY_ENRICH,
            # Section 4: no parent
            [],
            # Section 4: no children
            _NO_CHILDREN,
            # Section 5 query is NOT reached (no parent_sn set)
        ]
    )

    item = _make_item("electron_temperature")
    _enrich_for_docs_gen(gc, [item])

    assert "base_quantity" not in item


# ---------------------------------------------------------------------------
# T13 – derivative siblings
# ---------------------------------------------------------------------------


def test_t13_derivative_siblings_populated():
    """Sibling derivatives sharing the same parent are listed in derivative_context."""
    from imas_codex.standard_names.workers import _enrich_for_docs_gen

    gc = _make_gc(
        [
            # 1. _DOCS_GEN_ENRICH_QUERY
            _EMPTY_ENRICH,
            # 2. Section 4: parent COMPONENT_OF
            [
                {
                    "name": "poloidal_magnetic_flux",
                    "description": "PSI",
                    "documentation": "",
                    "axis": None,
                }
            ],
            # 3. Section 4: children
            _NO_CHILDREN,
            # 4. Section 5: parent unit + sibling derivatives
            [{"unit": "Wb", "siblings": ["darea_dpsi", "dvolume_dpsi"]}],
        ]
    )

    item = _make_item("dpressure_dpsi")
    _enrich_for_docs_gen(gc, [item])

    assert "derivative_context" in item, "derivative_context should be injected"
    ctx = item["derivative_context"]
    assert "darea_dpsi" in ctx["siblings"]
    assert "dvolume_dpsi" in ctx["siblings"]
    assert "dpressure_dpsi" not in ctx["siblings"], "self should not be a sibling"


def test_t13_null_siblings_filtered():
    """Null entries returned by Cypher collect() are filtered from siblings."""
    from imas_codex.standard_names.workers import _enrich_for_docs_gen

    gc = _make_gc(
        [
            _EMPTY_ENRICH,
            [
                {
                    "name": "poloidal_magnetic_flux",
                    "description": "",
                    "documentation": "",
                    "axis": None,
                }
            ],
            _NO_CHILDREN,
            # Cypher collect() can return [None] when no siblings exist
            [{"unit": "Wb", "siblings": [None]}],
        ]
    )

    item = _make_item("dpressure_dpsi")
    _enrich_for_docs_gen(gc, [item])

    assert "derivative_context" in item
    assert item["derivative_context"]["siblings"] == []


def test_t13_no_derivative_context_for_unknown_name():
    """Names not in DD_DERIVATIVE_MAP get base_quantity but not derivative_context."""
    from imas_codex.standard_names.workers import _enrich_for_docs_gen

    gc = _make_gc(
        [
            _EMPTY_ENRICH,
            # Section 4: has a parent (so base_quantity will be set)
            [
                {
                    "name": "some_parent",
                    "description": "Parent",
                    "documentation": "",
                    "axis": None,
                }
            ],
            _NO_CHILDREN,
            # Section 5: parent query succeeds
            [{"unit": "m", "siblings": ["other_component"]}],
        ]
    )

    item = _make_item("custom_component_not_in_map")
    _enrich_for_docs_gen(gc, [item])

    assert "base_quantity" in item, "base_quantity should still be injected"
    assert "derivative_context" not in item, (
        "derivative_context requires DD_DERIVATIVE_MAP match"
    )


# ---------------------------------------------------------------------------
# T14 – DD_DERIVATIVE_MAP direct lookup
# ---------------------------------------------------------------------------


def test_t14_dpsi_drho_tor_resolves():
    """DD_DERIVATIVE_MAP maps dpsi_drho_tor to the correct (numerator, denominator)."""
    from imas_codex.standard_names.families import DD_DERIVATIVE_MAP

    result = DD_DERIVATIVE_MAP.get("dpsi_drho_tor")
    assert result is not None, "dpsi_drho_tor should be in DD_DERIVATIVE_MAP"
    numerator, denominator = result
    assert numerator == "poloidal_magnetic_flux"
    assert denominator == "normalised_toroidal_flux_coordinate"


def test_t14_all_map_entries_are_2_tuples():
    """Every entry in DD_DERIVATIVE_MAP is a (numerator, denominator) 2-tuple."""
    from imas_codex.standard_names.families import DD_DERIVATIVE_MAP

    for key, value in DD_DERIVATIVE_MAP.items():
        assert isinstance(value, tuple), f"{key}: expected tuple, got {type(value)}"
        assert len(value) == 2, f"{key}: expected 2-tuple, got length {len(value)}"
        assert all(isinstance(v, str) and v for v in value), (
            f"{key}: both numerator and denominator must be non-empty strings"
        )


def test_t14_derivative_context_numerator_denominator():
    """_enrich_for_docs_gen sets correct numerator/denominator from DD_DERIVATIVE_MAP."""
    from imas_codex.standard_names.workers import _enrich_for_docs_gen

    gc = _make_gc(
        [
            _EMPTY_ENRICH,
            [
                {
                    "name": "area",
                    "description": "Area",
                    "documentation": "",
                    "axis": None,
                }
            ],
            _NO_CHILDREN,
            [{"unit": "m^2", "siblings": []}],
        ]
    )

    item = _make_item("darea_dpsi")
    _enrich_for_docs_gen(gc, [item])

    assert "derivative_context" in item
    ctx = item["derivative_context"]
    assert ctx["numerator"] == "area"
    assert ctx["denominator"] == "poloidal_magnetic_flux"
