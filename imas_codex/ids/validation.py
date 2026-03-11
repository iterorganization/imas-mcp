"""Programmatic validation for IMAS mapping bindings.

Replaces LLM self-review (Step 3) with concrete checks:
  - Source signal group exists in graph
  - Target IMAS path exists in graph
  - Transform expression executes without error
  - Source/target units are compatible
  - No duplicate target paths (two sources → same IMAS field)
"""

from __future__ import annotations

import logging
from collections import Counter
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
    """Check which source SignalGroup nodes exist in the graph."""
    result: dict[str, bool] = {}
    for sid in source_ids:
        rows = gc.query(
            "MATCH (sg:SignalGroup {id: $id}) RETURN sg.id AS id LIMIT 1",
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
            errors.append(f"SignalGroup '{b.source_id}' not found in graph")

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

    # 5. Duplicate target detection
    target_counts = Counter(b.target_id for b in bindings)
    report.duplicate_targets = [
        path for path, count in target_counts.items() if count > 1
    ]
    for dup in report.duplicate_targets:
        sources = [b.source_id for b in bindings if b.target_id == dup]
        report.escalations.append(
            EscalationFlag(
                source_id=sources[0],
                target_id=dup,
                severity=EscalationSeverity.WARNING,
                reason=f"Duplicate target: {dup} ← {len(sources)} sources ({', '.join(sources)})",
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
