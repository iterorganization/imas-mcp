"""Structural edge derivation for StandardName nodes.

Pure logic module — no graph access, no I/O.  Given a single
StandardName id string, ``derive_edges`` peels the outermost ISN
grammar operator/projection and returns the corresponding
``HAS_ARGUMENT`` or ``HAS_ERROR`` edge descriptor.

Recursion is structural: when the inner StandardName is itself written
to the graph, *its* derivation runs and emits *its* own edge.  We never
peel more than one layer here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from imas_standard_names.grammar import ir as isn_ir, parser

logger = logging.getLogger(__name__)

# Uncertainty prefix operators → HAS_ERROR error_type label
_UNCERTAINTY_OPS: dict[str, str] = {
    "upper_uncertainty": "upper",
    "lower_uncertainty": "lower",
    "uncertainty_index": "index",
}


@dataclass(frozen=True)
class DerivedEdge:
    """A single derived structural edge between two StandardName ids."""

    edge_type: str  # "HAS_ARGUMENT" or "HAS_ERROR"
    from_name: str  # source StandardName id
    to_name: str  # target StandardName id
    props: dict  # edge properties (operator, operator_kind, …)


def _strip_outer(ir: isn_ir.StandardNameIR) -> isn_ir.StandardNameIR:
    """Return *ir* with the outermost operator removed.

    All other fields (projection, qualifiers, base, locus, mechanism)
    are preserved unchanged.
    """
    return ir.model_copy(update={"operators": ir.operators[1:]})


def _strip_projection(ir: isn_ir.StandardNameIR) -> isn_ir.StandardNameIR:
    """Return *ir* with the projection cleared.

    All other fields (operators, qualifiers, base, locus, mechanism)
    are preserved unchanged.
    """
    return ir.model_copy(update={"projection": None})


def derive_edges(name: str) -> list[DerivedEdge]:
    """Return derived structural edges for a single StandardName id.

    Pure function.  The ISN parser is the sole source of structural truth.
    Names the parser cannot parse produce no edges (leaf treatment).

    Parameters
    ----------
    name:
        StandardName id string (e.g. ``"maximum_of_temperature"``).

    Returns
    -------
    list[DerivedEdge]
        Zero, one, or two edges depending on the outermost IR shape.
        Returns ``[]`` for unparseable names and leaf names.
    """
    try:
        result = parser.parse(name)
    except Exception:
        return []  # unparseable → leaf (no edges)

    ir = result.ir

    # --- Outermost operator ---
    if ir.operators:
        op = ir.operators[0]

        if op.kind == isn_ir.OperatorKind.BINARY:
            # Binary: two HAS_ARGUMENT edges, one per argument
            try:
                a = parser.compose(op.args[0])
                b = parser.compose(op.args[1])
            except Exception as exc:
                logger.debug("derive_edges compose failed for %r: %s", name, exc)
                return []
            return [
                DerivedEdge(
                    "HAS_ARGUMENT",
                    name,
                    a,
                    {
                        "operator": op.op,
                        "operator_kind": "binary",
                        "role": "a",
                        "separator": op.separator,
                    },
                ),
                DerivedEdge(
                    "HAS_ARGUMENT",
                    name,
                    b,
                    {
                        "operator": op.op,
                        "operator_kind": "binary",
                        "role": "b",
                        "separator": op.separator,
                    },
                ),
            ]

        # Unary prefix / postfix: strip outer and compose inner
        stripped = _strip_outer(ir)
        try:
            inner = parser.compose(stripped)
        except Exception as exc:
            logger.debug("derive_edges compose failed for %r: %s", name, exc)
            return []

        if op.kind == isn_ir.OperatorKind.UNARY_PREFIX and op.op in _UNCERTAINTY_OPS:
            # Uncertainty prefix — direction inverts: inner → name
            return [
                DerivedEdge(
                    "HAS_ERROR",
                    inner,
                    name,
                    {"error_type": _UNCERTAINTY_OPS[op.op]},
                )
            ]

        return [
            DerivedEdge(
                "HAS_ARGUMENT",
                name,
                inner,
                {
                    "operator": op.op,
                    "operator_kind": op.kind.value,
                },
            )
        ]

    # --- Projection (component) without operator ---
    if ir.projection is not None:
        stripped = _strip_projection(ir)
        try:
            inner = parser.compose(stripped)
        except Exception as exc:
            logger.debug("derive_edges compose failed for %r: %s", name, exc)
            return []
        return [
            DerivedEdge(
                "HAS_ARGUMENT",
                name,
                inner,
                {
                    "operator": "component",
                    "operator_kind": "projection",
                    "axis": ir.projection.axis,
                    "shape": ir.projection.shape.value,
                },
            )
        ]

    # Leaf: no operator, no projection
    return []
