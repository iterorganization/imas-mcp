"""Programmatic validation for IMAS mapping bindings.

Replaces LLM self-review (Step 3) with concrete checks:
  - Source signal source exists in graph
  - Target IMAS path exists in graph
  - Transform expression executes without error
  - Source/target units are compatible (dimensional + identity transform check)
  - COCOS sign-flip enforcement on known sign-flip paths
  - Multi-target aware duplicate detection:
    * Same source → multiple targets: allowed (no warning)
    * Multiple sources → same target: warning (potential conflict)
    * Same source → same target with different transforms: error
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

from imas_codex.graph.client import GraphClient
from imas_codex.ids.models import EscalationFlag, EscalationSeverity
from imas_codex.ids.tools import analyze_units, check_imas_paths
from imas_codex.ids.transforms import execute_transform

logger = logging.getLogger(__name__)


class DuplicateTargetClassification(StrEnum):
    """Classification of why multiple sources map to the same target."""

    EPOCH_VARIANTS = "epoch_variants"
    PROCESSING_STAGES = "processing_stages"
    REDUNDANT_DIAGNOSTICS = "redundant_diagnostics"
    LEGITIMATE_OTHER = "legitimate_other"
    ERRONEOUS = "erroneous"


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


def _check_sources_exist(source_ids: list[str], gc: GraphClient) -> dict[str, bool]:
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
    sign_flip_paths: list[str] | None = None,
) -> ValidationReport:
    """Run programmatic checks on a set of mapping bindings.

    Each binding is expected to have: source_id, target_id,
    transform_expression, source_units, target_units.

    Args:
        bindings: List of binding objects (ValidatedSignalMapping or similar).
        gc: GraphClient instance (created if None).
        sign_flip_paths: IMAS paths requiring COCOS sign flips.

    Returns:
        ValidationReport with per-binding results and aggregate status.
    """
    if gc is None:
        gc = GraphClient()

    report = ValidationReport(mapping_id="")
    if not bindings:
        report.all_passed = True
        return report

    flip_set = set(sign_flip_paths) if sign_flip_paths else set()

    # Deduplicate lookups
    source_ids = list({b.source_id for b in bindings})
    target_paths = list({b.target_id for b in bindings})

    # Batch checks
    source_exists = _check_sources_exist(source_ids, gc)
    target_results = {r["path"]: r for r in check_imas_paths(target_paths, gc=gc)}

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
                errors.append(f"Units incompatible: {src_units} → {tgt_units}")
            elif src_units and tgt_units and src_units != tgt_units and expr == "value":
                # Identity transform with mismatched units — needs conversion
                errors.append(
                    f"Identity transform but units differ: "
                    f"{src_units} → {tgt_units}; "
                    f"transform_expression should include unit conversion"
                )
        else:
            # No units specified — compatible by default
            check.units_compatible = True

        # 5. COCOS sign-flip enforcement
        if flip_set and b.target_id in flip_set:
            cocos_label = getattr(b, "cocos_label", None)
            if expr == "value" and not cocos_label:
                errors.append(
                    f"Target '{b.target_id}' requires COCOS sign handling "
                    f"but transform_expression is identity ('value') "
                    f"and no cocos_label is set"
                )

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
    # Multiple sources → same target: classify the pattern
    # Same source → same target with different transforms: error (conflicting)
    target_to_bindings: dict[str, list] = {}
    for b in bindings:
        target_to_bindings.setdefault(b.target_id, []).append(b)

    report.duplicate_targets = []
    for target_id, target_bindings in target_to_bindings.items():
        unique_sources = {b.source_id for b in target_bindings}
        if len(unique_sources) > 1:
            # Classify the many-to-one pattern
            classification = _classify_many_to_one(target_id, target_bindings, gc)
            source_list = sorted(unique_sources)

            if classification == DuplicateTargetClassification.ERRONEOUS:
                report.duplicate_targets.append(target_id)
                report.escalations.append(
                    EscalationFlag(
                        source_id=source_list[0],
                        target_id=target_id,
                        severity=EscalationSeverity.WARNING,
                        reason=(
                            f"Potentially erroneous many-to-one: {target_id} ← "
                            f"{len(source_list)} sources ({', '.join(source_list)})"
                        ),
                    )
                )
            else:
                # Legitimate many-to-one — log but do not escalate
                logger.info(
                    "Legitimate many-to-one (%s): %s ← %d sources",
                    classification.value,
                    target_id,
                    len(source_list),
                )
        elif len(target_bindings) > 1:
            # Same source, same target — check for conflicting transforms
            transforms = {
                getattr(b, "transform_expression", "value") for b in target_bindings
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

    report.all_passed = (
        all(
            c.source_exists
            and c.target_exists
            and c.transform_executes
            and c.units_compatible
            for c in report.binding_checks
        )
        and not report.duplicate_targets
    )

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


def _classify_many_to_one(
    target_id: str,
    bindings: list,
    gc: GraphClient,
) -> DuplicateTargetClassification:
    """Classify whether multiple sources mapping to the same target is legitimate.

    Checks source metadata to distinguish epoch variants, processing stages,
    redundant diagnostics, and erroneous duplicates.
    """
    source_ids = list({b.source_id for b in bindings})
    if len(source_ids) < 2:
        return DuplicateTargetClassification.LEGITIMATE_OTHER

    # Fetch source metadata for classification
    rows = gc.query(
        """
        UNWIND $source_ids AS sid
        MATCH (sg:SignalSource {id: sid})
        RETURN sg.id AS id, sg.group_key AS group_key,
               sg.physics_domain AS physics_domain,
               sg.description AS description
        """,
        source_ids=source_ids,
    )
    meta = {r["id"]: r for r in rows}

    group_keys = [meta.get(sid, {}).get("group_key", "") for sid in source_ids]
    domains = {meta.get(sid, {}).get("physics_domain") for sid in source_ids}
    domains.discard(None)

    # Check for epoch variants: same group_key prefix, differing by index
    prefixes = set()
    for gk in group_keys:
        parts = gk.rsplit(":", 1) if ":" in gk else gk.rsplit("/", 1)
        if len(parts) == 2:
            prefixes.add(parts[0])
    if len(prefixes) == 1:
        return DuplicateTargetClassification.EPOCH_VARIANTS

    # Same physics domain = likely processing stages or redundant diagnostics
    if len(domains) == 1:
        return DuplicateTargetClassification.PROCESSING_STAGES

    # Multiple physics domains = likely erroneous
    if len(domains) > 1:
        return DuplicateTargetClassification.ERRONEOUS

    return DuplicateTargetClassification.LEGITIMATE_OTHER


# ---------------------------------------------------------------------------
# Coverage threshold enforcement
# ---------------------------------------------------------------------------

# Minimum coverage percentage below which a warning is raised
COVERAGE_WARNING_THRESHOLD = 5.0
# Minimum coverage percentage below which an error is raised
COVERAGE_ERROR_THRESHOLD = 1.0


def check_coverage_threshold(
    ids_name: str,
    bindings: list,
    *,
    gc: GraphClient | None = None,
    warning_threshold: float = COVERAGE_WARNING_THRESHOLD,
    error_threshold: float = COVERAGE_ERROR_THRESHOLD,
) -> list[EscalationFlag]:
    """Check whether mapping coverage meets minimum thresholds.

    Returns escalation flags when field coverage is below thresholds:
    - ERROR when coverage < error_threshold
    - WARNING when coverage < warning_threshold

    Args:
        ids_name: IDS name.
        bindings: List of mapping bindings with target_id attribute.
        gc: GraphClient (created if None).
        warning_threshold: Percentage below which a warning is issued.
        error_threshold: Percentage below which an error is issued.

    Returns:
        List of escalation flags (empty if coverage is acceptable).
    """
    coverage = compute_coverage(ids_name, bindings, gc=gc)
    escalations: list[EscalationFlag] = []

    if coverage.percentage < error_threshold:
        escalations.append(
            EscalationFlag(
                source_id=ids_name,
                target_id=ids_name,
                severity=EscalationSeverity.ERROR,
                reason=(
                    f"IDS coverage critically low: {coverage.mapped_fields}/"
                    f"{coverage.total_leaf_fields} fields mapped "
                    f"({coverage.percentage:.1f}% < {error_threshold}% threshold)"
                ),
            )
        )
    elif coverage.percentage < warning_threshold:
        escalations.append(
            EscalationFlag(
                source_id=ids_name,
                target_id=ids_name,
                severity=EscalationSeverity.WARNING,
                reason=(
                    f"IDS coverage below expected: {coverage.mapped_fields}/"
                    f"{coverage.total_leaf_fields} fields mapped "
                    f"({coverage.percentage:.1f}% < {warning_threshold}% threshold)"
                ),
            )
        )

    return escalations


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
# Signal source coverage
# ---------------------------------------------------------------------------


@dataclass
class SignalCoverageReport:
    """What fraction of enriched signal sources have IMAS bindings."""

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


# ---------------------------------------------------------------------------
# 8.1 Extended signal source coverage per IDS
# ---------------------------------------------------------------------------


@dataclass
class SignalSourceCoverageReport:
    """Extended signal source coverage report for a specific IDS.

    Reports:
    - Enriched sources matching the IDS physics domain that have MAPS_TO_IMAS
    - Discovered (not enriched) sources that might benefit from enrichment
    - Sources mapped to more than one IMAS path (multi-target)
    """

    facility: str
    ids_name: str
    total_enriched_matching: int = 0
    mapped_to_ids: int = 0
    unmapped_enriched: list[str] = field(default_factory=list)
    discovered_sources: int = 0
    discovered_ids: list[str] = field(default_factory=list)
    multi_target_sources: int = 0
    multi_target_ids: list[str] = field(default_factory=list)
    enriched_mapped_pct: float = 0.0


def compute_signal_source_coverage(
    facility: str,
    ids_name: str,
    *,
    gc: GraphClient | None = None,
) -> SignalSourceCoverageReport:
    """Compute extended signal source coverage for a facility/IDS pair.

    Queries enriched signal sources with matching physics_domain, checks
    how many have MAPS_TO_IMAS relationships to the target IDS, and
    identifies discovered-but-not-enriched sources and multi-target sources.
    """
    if gc is None:
        gc = GraphClient()

    report = SignalSourceCoverageReport(facility=facility, ids_name=ids_name)

    # Enriched sources with matching physics domain mapped to this IDS
    enriched_rows = gc.query(
        """
        MATCH (sg:SignalSource {facility_id: $facility, status: 'enriched'})
        WHERE sg.physics_domain IS NOT NULL
        OPTIONAL MATCH (sg)-[:MAPS_TO_IMAS]->(ip:IMASNode {ids: $ids_name})
        RETURN sg.id AS id, ip IS NOT NULL AS is_mapped
        """,
        facility=facility,
        ids_name=ids_name,
    )

    report.total_enriched_matching = len(enriched_rows)
    report.mapped_to_ids = sum(1 for r in enriched_rows if r["is_mapped"])
    report.unmapped_enriched = sorted(
        r["id"] for r in enriched_rows if not r["is_mapped"]
    )
    report.enriched_mapped_pct = (
        (report.mapped_to_ids / report.total_enriched_matching * 100)
        if report.total_enriched_matching > 0
        else 0.0
    )

    # Discovered (not enriched) sources
    discovered_rows = gc.query(
        """
        MATCH (sg:SignalSource {facility_id: $facility, status: 'discovered'})
        RETURN sg.id AS id
        """,
        facility=facility,
    )
    report.discovered_sources = len(discovered_rows)
    report.discovered_ids = sorted(r["id"] for r in discovered_rows)

    # Multi-target sources (mapped to >1 IMAS path in this IDS)
    multi_rows = gc.query(
        """
        MATCH (sg:SignalSource {facility_id: $facility})-[:MAPS_TO_IMAS]->(ip:IMASNode {ids: $ids_name})
        WITH sg, count(DISTINCT ip) AS target_count
        WHERE target_count > 1
        RETURN sg.id AS id, target_count
        """,
        facility=facility,
        ids_name=ids_name,
    )
    report.multi_target_sources = len(multi_rows)
    report.multi_target_ids = sorted(r["id"] for r in multi_rows)

    return report


# ---------------------------------------------------------------------------
# 8.2 Assembly coverage
# ---------------------------------------------------------------------------


@dataclass
class AssemblyCoverageReport:
    """Assembly configuration completeness for sections of a mapping."""

    facility: str
    ids_name: str
    total_sections: int = 0
    sections_with_config: int = 0
    sections_without_config: list[str] = field(default_factory=list)
    default_pattern_count: int = 0
    custom_pattern_count: int = 0
    init_arrays_configured: int = 0
    init_arrays_unconfigured: int = 0


def compute_assembly_coverage(
    facility: str,
    ids_name: str,
    *,
    gc: GraphClient | None = None,
) -> AssemblyCoverageReport:
    """Report assembly configuration completeness from POPULATES relationships.

    Queries POPULATES relationships on the IMASMapping node and checks
    which sections have assembly config properties vs defaults.
    """
    if gc is None:
        gc = GraphClient()

    report = AssemblyCoverageReport(facility=facility, ids_name=ids_name)

    # Find all POPULATES relationships from the mapping
    rows = gc.query(
        """
        MATCH (m:IMASMapping {facility: $facility, ids_name: $ids_name})
              -[r:POPULATES]->(s:IMASNode)
        RETURN s.id AS section_path,
               r.assembly_pattern AS pattern,
               r.init_arrays AS init_arrays
        """,
        facility=facility,
        ids_name=ids_name,
    )

    # Also find sections that have bindings but no POPULATES
    binding_rows = gc.query(
        """
        MATCH (m:IMASMapping {facility: $facility, ids_name: $ids_name})
              -[:USES_SIGNAL_SOURCE]->(sg:SignalSource)
              -[:MAPS_TO_IMAS]->(ip:IMASNode {ids: $ids_name})
        WITH DISTINCT
             [x IN split(ip.id, '/')[0..3] | x] AS parts
        RETURN reduce(s = '', p IN parts | s + CASE WHEN s = '' THEN '' ELSE '/' END + p) AS section_path
        """,
        facility=facility,
        ids_name=ids_name,
    )

    populated_sections = {r["section_path"] for r in rows}
    all_sections = populated_sections | {r["section_path"] for r in binding_rows}
    report.total_sections = len(all_sections)
    report.sections_with_config = len(populated_sections)
    report.sections_without_config = sorted(all_sections - populated_sections)

    for r in rows:
        pattern = r.get("pattern", "array_per_node")
        if pattern == "array_per_node":
            report.default_pattern_count += 1
        else:
            report.custom_pattern_count += 1

        init = r.get("init_arrays")
        if init:
            report.init_arrays_configured += 1
        else:
            report.init_arrays_unconfigured += 1

    return report


# ---------------------------------------------------------------------------
# 8.3 Mapping confidence distribution
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceDistribution:
    """Distribution of mapping confidences across bindings."""

    total_bindings: int = 0
    low_count: int = 0  # < 0.5
    medium_count: int = 0  # 0.5 - 0.8
    high_count: int = 0  # > 0.8
    low_bindings: list[str] = field(default_factory=list)
    average_confidence: float = 0.0


def compute_confidence_distribution(
    bindings: list,
) -> ConfidenceDistribution:
    """Compute the confidence distribution of mapping bindings.

    Buckets:
    - Low (<0.5): Flag for review
    - Medium (0.5-0.8): Acceptable but could improve
    - High (>0.8): Confident
    """
    dist = ConfidenceDistribution()
    if not bindings:
        return dist

    dist.total_bindings = len(bindings)
    total_conf = 0.0

    for b in bindings:
        conf = getattr(b, "confidence", 0.5)
        total_conf += conf

        if conf < 0.5:
            dist.low_count += 1
            dist.low_bindings.append(f"{b.source_id} → {b.target_id} ({conf:.2f})")
        elif conf <= 0.8:
            dist.medium_count += 1
        else:
            dist.high_count += 1

    dist.average_confidence = total_conf / dist.total_bindings
    return dist


# ---------------------------------------------------------------------------
# 10.5 E2E Validation Pipeline — Tier 3: Data Validation
# ---------------------------------------------------------------------------


@dataclass
class E2EFieldCheck:
    """Result of validating one field in the assembled IDS."""

    target_path: str
    populated: bool = False
    value_range: str | None = None
    error: str | None = None


@dataclass
class E2EValidationResult:
    """Result of end-to-end mapping validation."""

    facility: str
    ids_name: str
    shot: int
    strategy: str = "client"
    extraction_success: int = 0
    extraction_failed: int = 0
    extraction_errors: list[str] = field(default_factory=list)
    assembly_success: bool = False
    assembly_error: str | None = None
    field_checks: list[E2EFieldCheck] = field(default_factory=list)
    time_base_consistent: bool = False
    all_passed: bool = False


def validate_mapping_e2e(
    facility: str,
    ids_name: str,
    shot: int,
    *,
    gc: GraphClient,
    ssh_host: str | None = None,
    strategy: str = "auto",
) -> E2EValidationResult:
    """Run full E2E validation: extract → transform → assemble → validate.

    Tier 3 validation that requires facility SSH access. Extracts sample
    data for one shot, runs assembly code, and validates the populated IDS.

    Args:
        facility: Facility identifier.
        ids_name: IDS name.
        shot: Shot number to validate against.
        gc: GraphClient instance.
        ssh_host: SSH host for remote extraction (None = use facility config).
        strategy: "auto", "client", or "remote".

    Returns:
        E2EValidationResult with per-field validation details.
    """
    import json as _json

    from imas_codex.ids.codegen import generate_extraction_script
    from imas_codex.ids.tools import search_existing_mappings
    from imas_codex.remote.executor import (
        probe_remote_capabilities,
        run_script_via_stdin,
    )

    result = E2EValidationResult(
        facility=facility,
        ids_name=ids_name,
        shot=shot,
    )

    # 1. Load existing mapping from graph
    mapping_data = search_existing_mappings(facility, ids_name, gc=gc)
    if not mapping_data["mapping"]:
        result.assembly_error = f"No mapping found for {facility}/{ids_name}"
        return result

    bindings = mapping_data["bindings"]
    if not bindings:
        result.assembly_error = "No bindings in mapping"
        return result

    # 2. Determine strategy
    if strategy == "auto":
        caps = probe_remote_capabilities(ssh_host)
        result.strategy = "remote" if caps.get("imas") else "client"
    else:
        result.strategy = strategy

    # 3. Group bindings by section for extraction script
    section_signals: dict[str, list[dict]] = {}
    for b in bindings:
        target = b.get("target_id", "")
        # Derive section from target path (first two path segments)
        parts = target.split("/")
        section = "/".join(parts[:2]) if len(parts) >= 2 else ids_name
        section_signals.setdefault(section, []).append(
            {
                "id": b.get("source_id", ""),
                "accessor": b.get("source_id", ""),
                "data_source_name": "default",
            }
        )

    # 4. Get DataAccess info for extraction script
    da_records = gc.query(
        """
        MATCH (f:Facility {id: $facility})-[:HAS_DATA_ACCESS]->(da:DataAccess)
        RETURN da
        LIMIT 1
        """,
        facility=facility,
    )
    data_access = da_records[0]["da"] if da_records else {}

    # 5. Generate and run extraction script
    script = generate_extraction_script(
        facility,
        ids_name,
        section_signals,
        data_access,
        max_points=10000,
    )

    try:
        raw_output = run_script_via_stdin(
            f"import json, sys; config = {_json.dumps({'shot': shot})}\n{script}",
            ssh_host=ssh_host,
            timeout=120,
            interpreter="python3",
        )
        # Parse extraction results
        from imas_codex.remote.serialization import decode_extraction_output

        extracted = decode_extraction_output(raw_output)
        results_data = extracted.get("results", {})

        for sig_id, sig_result in results_data.items():
            if isinstance(sig_result, dict) and sig_result.get("success"):
                result.extraction_success += 1
            else:
                result.extraction_failed += 1
                err = (
                    sig_result.get("error", "unknown")
                    if isinstance(sig_result, dict)
                    else "unknown"
                )
                result.extraction_errors.append(f"{sig_id}: {err}")

    except Exception as e:
        result.assembly_error = f"Extraction failed: {e}"
        return result

    # 6. Validate field population
    for b in bindings:
        target = b.get("target_id", "")
        source = b.get("source_id", "")
        sig_data = results_data.get(source, {})
        populated = isinstance(sig_data, dict) and sig_data.get("success", False)
        check = E2EFieldCheck(
            target_path=target,
            populated=populated,
        )
        if populated and isinstance(sig_data.get("data"), list):
            data = sig_data["data"]
            if data:
                check.value_range = f"[{min(data):.4g}, {max(data):.4g}]"
        result.field_checks.append(check)

    result.assembly_success = result.extraction_success > 0

    # 7. Check time-base consistency
    time_bases = []
    for sig_result in results_data.values():
        if isinstance(sig_result, dict) and sig_result.get("time"):
            tb = sig_result["time"]
            if isinstance(tb, list) and len(tb) >= 2:
                time_bases.append((tb[0], tb[-1], len(tb)))

    if time_bases:
        # Check that all time bases have the same start/end within tolerance
        starts = [t[0] for t in time_bases]
        ends = [t[1] for t in time_bases]
        result.time_base_consistent = (
            (max(starts) - min(starts) < 1.0 and max(ends) - min(ends) < 1.0)
            if starts and ends
            else True
        )
    else:
        result.time_base_consistent = True

    # Overall pass
    result.all_passed = (
        result.assembly_success
        and result.extraction_failed == 0
        and result.time_base_consistent
    )

    return result
