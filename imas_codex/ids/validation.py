"""Programmatic validation for IMAS mapping bindings.

Replaces LLM self-review (Step 3) with concrete checks:
  - Source signal group exists in graph
  - Target IMAS path exists in graph
  - Transform expression executes without error
  - Source/target units are compatible
  - Multi-target aware duplicate detection:
    * Same source → multiple targets: allowed (no warning)
    * Multiple sources → same target: warning (potential conflict)
    * Same source → same target with different transforms: error
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from imas_codex.graph.client import GraphClient
from imas_codex.ids.models import EscalationFlag, EscalationSeverity
from imas_codex.ids.tools import analyze_units, check_imas_paths
from imas_codex.ids.transforms import execute_transform

logger = logging.getLogger(__name__)


@dataclass
class BindingCheck:
    """Result of validating a single source→target binding."""

    source_id: str
    target_id: str
    source_exists: bool = False
    target_exists: bool = False
    transform_executes: bool = False
    units_compatible: bool = False
    error: str | None = None


@dataclass
class ValidationReport:
    """Aggregate result of validating all bindings in a mapping."""

    mapping_id: str
    all_passed: bool = False
    binding_checks: list[BindingCheck] = field(default_factory=list)
    duplicate_targets: list[str] = field(default_factory=list)
    escalations: list[EscalationFlag] = field(default_factory=list)


def _check_sources_exist(
    source_ids: list[str], gc: GraphClient
) -> dict[str, bool]:
    """Check which source SignalSource nodes exist in the graph."""
    result: dict[str, bool] = {}
    for sid in source_ids:
        rows = gc.query(
            "MATCH (sg:SignalSource {id: $id}) RETURN sg.id AS id LIMIT 1",
            id=sid,
        )
        result[sid] = len(rows) > 0
    return result


def validate_mapping(
    bindings: list,
    *,
    gc: GraphClient | None = None,
) -> ValidationReport:
    """Run programmatic checks on a set of mapping bindings.

    Each binding is expected to have: source_id, target_id,
    transform_expression, source_units, target_units.

    Args:
        bindings: List of binding objects (ValidatedFieldMapping or similar).
        gc: GraphClient instance (created if None).

    Returns:
        ValidationReport with per-binding results and aggregate status.
    """
    if gc is None:
        gc = GraphClient()

    report = ValidationReport(mapping_id="")
    if not bindings:
        report.all_passed = True
        return report

    # Deduplicate lookups
    source_ids = list({b.source_id for b in bindings})
    target_paths = list({b.target_id for b in bindings})

    # Batch checks
    source_exists = _check_sources_exist(source_ids, gc)
    target_results = {
        r["path"]: r for r in check_imas_paths(target_paths, gc=gc)
    }

    for b in bindings:
        check = BindingCheck(source_id=b.source_id, target_id=b.target_id)
        errors: list[str] = []

        # 1. Source exists
        check.source_exists = source_exists.get(b.source_id, False)
        if not check.source_exists:
            errors.append(f"SignalSource '{b.source_id}' not found in graph")

        # 2. Target exists
        target_info = target_results.get(b.target_id, {})
        check.target_exists = target_info.get("exists", False)
        if not check.target_exists:
            suggestion = target_info.get("suggestion")
            msg = f"IMAS path '{b.target_id}' not found"
            if suggestion:
                msg += f" (renamed → {suggestion})"
            errors.append(msg)

        # 3. Transform executes
        expr = getattr(b, "transform_expression", "value")
        try:
            execute_transform(1.0, expr)
            check.transform_executes = True
        except Exception as exc:
            check.transform_executes = False
            errors.append(f"Transform '{expr}' failed: {exc}")

        # 4. Units compatible
        src_units = getattr(b, "source_units", None)
        tgt_units = getattr(b, "target_units", None)
        if src_units or tgt_units:
            unit_result = analyze_units(src_units, tgt_units)
            check.units_compatible = unit_result.get("compatible", False)
            if not check.units_compatible:
                errors.append(
                    f"Units incompatible: {src_units} → {tgt_units}"
                )
        else:
            # No units specified — compatible by default
            check.units_compatible = True

        if errors:
            check.error = "; ".join(errors)
            report.escalations.append(
                EscalationFlag(
                    source_id=b.source_id,
                    target_id=b.target_id,
                    severity=EscalationSeverity.ERROR,
                    reason=check.error,
                )
            )

        report.binding_checks.append(check)

    # 5. Duplicate target detection (multi-target aware)
    # Same source → multiple targets: expected (no warning)
    # Multiple sources → same target: warning (potential conflict)
    # Same source → same target with different transforms: error (conflicting)
    target_to_bindings: dict[str, list] = {}
    for b in bindings:
        target_to_bindings.setdefault(b.target_id, []).append(b)

    report.duplicate_targets = []
    for target_id, target_bindings in target_to_bindings.items():
        unique_sources = {b.source_id for b in target_bindings}
        if len(unique_sources) > 1:
            # Multiple sources → same target: warning
            report.duplicate_targets.append(target_id)
            source_list = sorted(unique_sources)
            report.escalations.append(
                EscalationFlag(
                    source_id=source_list[0],
                    target_id=target_id,
                    severity=EscalationSeverity.WARNING,
                    reason=(
                        f"Multiple sources → same target: {target_id} ← "
                        f"{len(source_list)} sources ({', '.join(source_list)})"
                    ),
                )
            )
        elif len(target_bindings) > 1:
            # Same source, same target — check for conflicting transforms
            transforms = {
                getattr(b, "transform_expression", "value")
                for b in target_bindings
            }
            if len(transforms) > 1:
                report.duplicate_targets.append(target_id)
                report.escalations.append(
                    EscalationFlag(
                        source_id=target_bindings[0].source_id,
                        target_id=target_id,
                        severity=EscalationSeverity.ERROR,
                        reason=(
                            f"Conflicting transforms for same source→target: "
                            f"{target_id} has transforms: "
                            f"{', '.join(sorted(transforms))}"
                        ),
                    )
                )

    report.all_passed = all(
        c.source_exists
        and c.target_exists
        and c.transform_executes
        and c.units_compatible
        for c in report.binding_checks
    ) and not report.duplicate_targets

    logger.info(
        "Validation: %d bindings, %d passed, %d duplicates",
        len(report.binding_checks),
        sum(
            1
            for c in report.binding_checks
            if c.source_exists
            and c.target_exists
            and c.transform_executes
            and c.units_compatible
        ),
        len(report.duplicate_targets),
    )

    return report


# ---------------------------------------------------------------------------
# Coverage reporting
# ---------------------------------------------------------------------------


@dataclass
class CoverageReport:
    """How much of a target IDS is covered by the current bindings."""

    ids_name: str
    total_leaf_fields: int = 0
    mapped_fields: int = 0
    unmapped_fields: list[str] = field(default_factory=list)
    mapped_paths: list[str] = field(default_factory=list)
    percentage: float = 0.0


def compute_coverage(
    ids_name: str,
    bindings: list,
    *,
    gc: GraphClient | None = None,
) -> CoverageReport:
    """Compute coverage of target IDS leaf fields by mapping bindings.

    Queries all data-bearing (non-STRUCTURE/STRUCT_ARRAY) fields under
    the IDS and compares against the set of target_id values in bindings.

    Args:
        ids_name: IDS name (e.g. "pf_active").
        bindings: List of binding objects with target_id attribute.
        gc: GraphClient instance (created if None).

    Returns:
        CoverageReport with mapped/unmapped field counts and percentage.
    """
    if gc is None:
        gc = GraphClient()

    # Query all leaf fields for this IDS
    rows = gc.query(
        """
        MATCH (p:IMASNode)
        WHERE p.ids = $ids_name
          AND NOT p.data_type IN ['STRUCTURE', 'STRUCT_ARRAY']
        OPTIONAL MATCH (p)-[:DEPRECATED_IN]->()
        WITH p, count(*) AS dep_count
        WHERE dep_count = 0 OR NOT EXISTS { (p)-[:DEPRECATED_IN]->() }
        RETURN p.id AS id
        """,
        ids_name=ids_name,
    )

    all_fields = {r["id"] for r in rows}
    mapped_targets = {b.target_id for b in bindings}

    mapped = all_fields & mapped_targets
    unmapped = sorted(all_fields - mapped_targets)

    total = len(all_fields)
    n_mapped = len(mapped)
    pct = (n_mapped / total * 100) if total > 0 else 0.0

    return CoverageReport(
        ids_name=ids_name,
        total_leaf_fields=total,
        mapped_fields=n_mapped,
        unmapped_fields=unmapped,
        mapped_paths=sorted(mapped),
        percentage=pct,
    )


# ---------------------------------------------------------------------------
# Signal group coverage
# ---------------------------------------------------------------------------


@dataclass
class SignalCoverageReport:
    """What fraction of enriched signal groups have IMAS bindings."""

    facility: str
    total_enriched: int = 0
    mapped: int = 0
    unmapped_groups: list[str] = field(default_factory=list)
    percentage: float = 0.0


def compute_signal_coverage(
    facility: str,
    *,
    gc: GraphClient | None = None,
) -> SignalCoverageReport:
    """Query enriched SignalSources and check which have MAPS_TO_IMAS bindings.

    Args:
        facility: Facility identifier (e.g. "jet").
        gc: GraphClient instance (created if None).

    Returns:
        SignalCoverageReport with mapped/unmapped counts and percentage.
    """
    if gc is None:
        gc = GraphClient()

    rows = gc.query(
        """
        MATCH (sg:SignalSource {facility_id: $facility, status: 'enriched'})
        OPTIONAL MATCH (sg)-[:MAPS_TO_IMAS]->(ip:IMASNode)
        RETURN sg.id AS id, ip IS NOT NULL AS is_mapped
        """,
        facility=facility,
    )

    total = len(rows)
    mapped_ids = [r["id"] for r in rows if r["is_mapped"]]
    unmapped_ids = sorted(r["id"] for r in rows if not r["is_mapped"])
    n_mapped = len(mapped_ids)
    pct = (n_mapped / total * 100) if total > 0 else 0.0

    return SignalCoverageReport(
        facility=facility,
        total_enriched=total,
        mapped=n_mapped,
        unmapped_groups=unmapped_ids,
        percentage=pct,
    )
