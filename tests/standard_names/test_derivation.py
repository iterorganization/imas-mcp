"""Unit tests for imas_codex.standard_names.derivation (D1–D16).

Pure logic tests — no graph, no I/O.  All ISN bases used here are
confirmed parseable by rc25: temperature, pressure, current_density.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from imas_standard_names.grammar import ir as isn_ir

from imas_codex.standard_names.derivation import DerivedEdge, derive_edges

# ---------------------------------------------------------------------------
# D1 — leaf name (no operators, no projection)
# ---------------------------------------------------------------------------


def test_d1_leaf_temperature():
    """temperature is a leaf — no edges."""
    edges = derive_edges("temperature")
    assert edges == []


# ---------------------------------------------------------------------------
# D2 — unary prefix: maximum
# ---------------------------------------------------------------------------


def test_d2_maximum_of_temperature():
    """maximum_of_temperature → HAS_ARGUMENT to temperature."""
    edges = derive_edges("maximum_of_temperature")
    assert len(edges) == 1
    e = edges[0]
    assert e.edge_type == "HAS_ARGUMENT"
    assert e.from_name == "maximum_of_temperature"
    assert e.to_name == "temperature"
    assert e.props["operator"] == "maximum"
    assert e.props["operator_kind"] == "unary_prefix"


# ---------------------------------------------------------------------------
# D3 — unary prefix: time_derivative
# ---------------------------------------------------------------------------


def test_d3_time_derivative_of_temperature():
    """time_derivative_of_temperature → HAS_ARGUMENT to temperature."""
    edges = derive_edges("time_derivative_of_temperature")
    assert len(edges) == 1
    e = edges[0]
    assert e.edge_type == "HAS_ARGUMENT"
    assert e.from_name == "time_derivative_of_temperature"
    assert e.to_name == "temperature"
    assert e.props["operator"] == "time_derivative"
    assert e.props["operator_kind"] == "unary_prefix"


# ---------------------------------------------------------------------------
# D4 — stacked unary prefix: outermost only
# ---------------------------------------------------------------------------


def test_d4_time_average_of_maximum_of_temperature():
    """time_average_of_maximum_of_temperature → HAS_ARGUMENT to maximum_of_temperature."""
    edges = derive_edges("time_average_of_maximum_of_temperature")
    assert len(edges) == 1
    e = edges[0]
    assert e.edge_type == "HAS_ARGUMENT"
    assert e.from_name == "time_average_of_maximum_of_temperature"
    assert e.to_name == "maximum_of_temperature"
    assert e.props["operator"] == "time_average"
    assert e.props["operator_kind"] == "unary_prefix"


# ---------------------------------------------------------------------------
# D5 — unary postfix: magnitude
# ---------------------------------------------------------------------------


def test_d5_temperature_magnitude():
    """temperature_magnitude → HAS_ARGUMENT to temperature."""
    edges = derive_edges("temperature_magnitude")
    assert len(edges) == 1
    e = edges[0]
    assert e.edge_type == "HAS_ARGUMENT"
    assert e.from_name == "temperature_magnitude"
    assert e.to_name == "temperature"
    assert e.props["operator"] == "magnitude"
    assert e.props["operator_kind"] == "unary_postfix"


# ---------------------------------------------------------------------------
# D6 — unary postfix: moment
# ---------------------------------------------------------------------------


def test_d6_temperature_moment():
    """temperature_moment → HAS_ARGUMENT to temperature."""
    edges = derive_edges("temperature_moment")
    assert len(edges) == 1
    e = edges[0]
    assert e.edge_type == "HAS_ARGUMENT"
    assert e.from_name == "temperature_moment"
    assert e.to_name == "temperature"
    assert e.props["operator"] == "moment"
    assert e.props["operator_kind"] == "unary_postfix"


# ---------------------------------------------------------------------------
# D7 — unary postfix: reference_waveform
# ---------------------------------------------------------------------------


def test_d7_temperature_reference_waveform():
    """temperature_reference_waveform → HAS_ARGUMENT to temperature."""
    edges = derive_edges("temperature_reference_waveform")
    assert len(edges) == 1
    e = edges[0]
    assert e.edge_type == "HAS_ARGUMENT"
    assert e.from_name == "temperature_reference_waveform"
    assert e.to_name == "temperature"
    assert e.props["operator"] == "reference_waveform"
    assert e.props["operator_kind"] == "unary_postfix"


# ---------------------------------------------------------------------------
# D8 — unary postfix: bessel_0
# ---------------------------------------------------------------------------


def test_d8_temperature_bessel_0():
    """temperature_bessel_0 → HAS_ARGUMENT to temperature."""
    edges = derive_edges("temperature_bessel_0")
    assert len(edges) == 1
    e = edges[0]
    assert e.edge_type == "HAS_ARGUMENT"
    assert e.from_name == "temperature_bessel_0"
    assert e.to_name == "temperature"
    assert e.props["operator"] == "bessel_0"
    assert e.props["operator_kind"] == "unary_postfix"


# ---------------------------------------------------------------------------
# D9 — binary: ratio
# ---------------------------------------------------------------------------


def test_d9_ratio_of_temperature_to_pressure():
    """ratio_of_temperature_to_pressure → two HAS_ARGUMENT edges."""
    edges = derive_edges("ratio_of_temperature_to_pressure")
    assert len(edges) == 2

    edge_by_role = {e.props["role"]: e for e in edges}
    assert set(edge_by_role) == {"a", "b"}

    ea = edge_by_role["a"]
    assert ea.edge_type == "HAS_ARGUMENT"
    assert ea.from_name == "ratio_of_temperature_to_pressure"
    assert ea.to_name == "temperature"
    assert ea.props["operator"] == "ratio"
    assert ea.props["operator_kind"] == "binary"
    assert ea.props["separator"] == "to"

    eb = edge_by_role["b"]
    assert eb.edge_type == "HAS_ARGUMENT"
    assert eb.from_name == "ratio_of_temperature_to_pressure"
    assert eb.to_name == "pressure"
    assert eb.props["operator"] == "ratio"
    assert eb.props["operator_kind"] == "binary"
    assert eb.props["separator"] == "to"


# ---------------------------------------------------------------------------
# D10 — uncertainty prefix: upper
# ---------------------------------------------------------------------------


def test_d10_upper_uncertainty_of_temperature():
    """upper_uncertainty_of_temperature → HAS_ERROR from temperature."""
    edges = derive_edges("upper_uncertainty_of_temperature")
    assert len(edges) == 1
    e = edges[0]
    assert e.edge_type == "HAS_ERROR"
    # Direction is inner → name (NOT name → inner)
    assert e.from_name == "temperature"
    assert e.to_name == "upper_uncertainty_of_temperature"
    assert e.props["error_type"] == "upper"
    # No HAS_ARGUMENT
    ha = [x for x in edges if x.edge_type == "HAS_ARGUMENT"]
    assert ha == []


# ---------------------------------------------------------------------------
# D11 — uncertainty prefix: lower
# ---------------------------------------------------------------------------


def test_d11_lower_uncertainty_of_temperature():
    """lower_uncertainty_of_temperature → HAS_ERROR {error_type: lower}."""
    edges = derive_edges("lower_uncertainty_of_temperature")
    assert len(edges) == 1
    e = edges[0]
    assert e.edge_type == "HAS_ERROR"
    assert e.from_name == "temperature"
    assert e.to_name == "lower_uncertainty_of_temperature"
    assert e.props["error_type"] == "lower"


# ---------------------------------------------------------------------------
# D12 — uncertainty prefix: index
# ---------------------------------------------------------------------------


def test_d12_uncertainty_index_of_temperature():
    """uncertainty_index_of_temperature → HAS_ERROR {error_type: index}."""
    edges = derive_edges("uncertainty_index_of_temperature")
    assert len(edges) == 1
    e = edges[0]
    assert e.edge_type == "HAS_ERROR"
    assert e.from_name == "temperature"
    assert e.to_name == "uncertainty_index_of_temperature"
    assert e.props["error_type"] == "index"


# ---------------------------------------------------------------------------
# D13 — locus preserved through peel
# ---------------------------------------------------------------------------


def test_d13_maximum_of_temperature_at_plasma_boundary():
    """Locus is preserved: inner is temperature_at_plasma_boundary."""
    edges = derive_edges("maximum_of_temperature_at_plasma_boundary")
    assert len(edges) == 1
    e = edges[0]
    assert e.edge_type == "HAS_ARGUMENT"
    assert e.from_name == "maximum_of_temperature_at_plasma_boundary"
    assert e.to_name == "temperature_at_plasma_boundary"
    assert e.props["operator"] == "maximum"
    assert e.props["operator_kind"] == "unary_prefix"


# ---------------------------------------------------------------------------
# D14 — leaf with locus only (no operator)
# ---------------------------------------------------------------------------


def test_d14_elongation_of_plasma_boundary():
    """elongation_of_plasma_boundary is a leaf — locus only, no operator."""
    edges = derive_edges("elongation_of_plasma_boundary")
    assert edges == []


# ---------------------------------------------------------------------------
# D15 — garbage string: parser raises, caught, returns []
# ---------------------------------------------------------------------------


def test_d15_garbage_string():
    """not_a_name is unparseable — derive_edges returns []."""
    edges = derive_edges("not_a_name")
    assert edges == []


# ---------------------------------------------------------------------------
# D16 — projection (monkeypatched) readiness test
# ---------------------------------------------------------------------------


def test_d16_projection_monkeypatched():
    """Projection IR shape → HAS_ARGUMENT with operator_kind='projection'."""
    # Build a stub IR representing current_density_parallel_component
    base = isn_ir.QuantityOrCarrier(
        token="current_density", kind=isn_ir.BaseKind.QUANTITY
    )
    proj = isn_ir.AxisProjection(
        axis="parallel", shape=isn_ir.ProjectionShape.COMPONENT
    )
    stub_ir = isn_ir.StandardNameIR(
        operators=[],
        projection=proj,
        qualifiers=[],
        base=base,
        locus=None,
        mechanism=None,
    )

    # ParseResult stub
    fake_result = MagicMock()
    fake_result.ir = stub_ir

    name = "current_density_parallel_component"

    with patch(
        "imas_codex.standard_names.derivation.parser.parse",
        return_value=fake_result,
    ):
        edges = derive_edges(name)

    assert len(edges) == 1
    e = edges[0]
    assert e.edge_type == "HAS_ARGUMENT"
    assert e.from_name == name
    assert e.to_name == "current_density"
    assert e.props["operator"] == "component"
    assert e.props["operator_kind"] == "projection"
    assert e.props["axis"] == "parallel"
    assert e.props["shape"] == "component"
