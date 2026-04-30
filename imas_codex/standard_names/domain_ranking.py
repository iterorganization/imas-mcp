"""Physics-domain ranking for promote-on-higher-rank logic.

A :class:`StandardName` carries a single ``physics_domain``: when a new
source is attached to an existing name (or composed from multiple
sources at birth), the name's primary domain may be *promoted* to a
more central / higher-ranked domain.  Lower rank value = more central.

The canonical ranking comes from
:func:`imas_codex.standard_names.domain_priority.get_domain_priority_index`,
which is graph-derived from cluster ``mapping_relevance`` weights.  When
the graph is unreachable or empty (CI, fresh checkout, unit tests) we
fall back to :data:`FALLBACK_RANK_TABLE` — a hand-maintained ordering
that mirrors the operational priority.

Promotion is monotonic: a name's ``physics_domain`` only ever moves
*toward* lower rank values.  Ties are resolved by keeping the current
domain (no flapping when ranks are equal).

Every domain that has ever contributed a source to a StandardName is
recorded in ``source_domains`` (multivalued, append-only) so cross-domain
discoverability is preserved even after promotion.
"""

from __future__ import annotations

from imas_codex.standard_names.domain_priority import (
    get_domain_priority_index,
)

# Hand-maintained fallback ordering used when the graph priority index
# is empty or unreachable.  Values are PhysicsDomain enum members; the
# rank is the position in this tuple (lower index = more central).
#
# Design intent (lower rank = higher priority):
#   - Plasma core physics first (equilibrium, transport, MHD)
#   - Heating & current drive next
#   - Edge / SOL / divertor
#   - Diagnostics grouped (magnetics first as backbone)
#   - Operational / engineering / metadata last
#   - 'general' acts as the lowest-priority (most generic) domain.
FALLBACK_RANK_TABLE: tuple[str, ...] = (
    "equilibrium",
    "magnetohydrodynamics",
    "transport",
    "core_plasma_physics",
    "turbulence",
    "gyrokinetics",
    "fast_particles",
    "runaway_electrons",
    "auxiliary_heating",
    "current_drive",
    "waves",
    "edge_plasma_physics",
    "plasma_wall_interactions",
    "divertor_physics",
    "fueling",
    "magnetic_field_diagnostics",
    "plasma_measurement_diagnostics",
    "particle_measurement_diagnostics",
    "electromagnetic_wave_diagnostics",
    "radiation_measurement_diagnostics",
    "mechanical_measurement_diagnostics",
    "spectroscopy",
    "neutronics",
    "plasma_control",
    "plasma_initiation",
    "machine_operations",
    "magnetic_field_systems",
    "structural_components",
    "plant_systems",
    "data_management",
    "computational_workflow",
    "general",
)

#: Sentinel rank for unranked / unknown domains.  Higher than any
#: value in :data:`FALLBACK_RANK_TABLE` so unknowns never promote.
UNKNOWN_RANK: int = 999


def _fallback_rank_index() -> dict[str, int]:
    """Materialise :data:`FALLBACK_RANK_TABLE` as a dict for O(1) lookup."""
    return {d: i for i, d in enumerate(FALLBACK_RANK_TABLE)}


def domain_rank(domain: str | None) -> int:
    """Return the rank for *domain* (lower = more central).

    Resolution order:
      1. Graph-derived index (cluster ``mapping_relevance`` weighting).
      2. :data:`FALLBACK_RANK_TABLE` ordering.
      3. :data:`UNKNOWN_RANK` for ``None``, empty, or unknown values.
    """
    if not domain:
        return UNKNOWN_RANK
    try:
        graph_idx = get_domain_priority_index()
    except Exception:  # pragma: no cover — defensive
        graph_idx = {}
    if domain in graph_idx:
        return graph_idx[domain]
    fallback = _fallback_rank_index()
    return fallback.get(domain, UNKNOWN_RANK)


def maybe_promote_domain(current: str | None, candidate: str | None) -> str | None:
    """Return the higher-priority of *current* and *candidate*.

    Semantics:
      * If both are empty → ``None``.
      * If exactly one is empty → return the non-empty value.
      * If both are non-empty → return the lower-ranked.  On a tie,
        keep *current* (no flapping when ranks are equal).
    """
    if not current and not candidate:
        return None
    if not current:
        return candidate
    if not candidate:
        return current
    if domain_rank(candidate) < domain_rank(current):
        return candidate
    return current


def merge_source_domains(existing: list[str] | None, *new: str | None) -> list[str]:
    """Append *new* domains to *existing*, deduplicate, preserve order.

    ``None`` and empty values are skipped.  Existing order is preserved;
    new domains are appended in argument order.  Used to maintain the
    ``source_domains`` accumulator on :class:`StandardName` nodes.
    """
    out: list[str] = []
    seen: set[str] = set()
    for d in list(existing or []) + list(new):
        if not d:
            continue
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
    return out


__all__ = [
    "FALLBACK_RANK_TABLE",
    "UNKNOWN_RANK",
    "domain_rank",
    "maybe_promote_domain",
    "merge_source_domains",
]
