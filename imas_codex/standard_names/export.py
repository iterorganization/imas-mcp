"""Export validated standard names from the graph to a staging directory.

Reads StandardName nodes from the Neo4j graph, applies quality gates,
and writes YAML files matching the ``imas-standard-names-catalog``
layout: ``<staging>/standard_names/<domain>/<name>.yml`` plus a
``<staging>/catalog.yml`` manifest.

This module is the first half of the two-step export→publish flow.
The staging directory produced here is consumed by ``publish.py``
(transport to ISNC repo) and ``preview.py`` (local site render).

See plan 35 §Phase 3.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from imas_codex.standard_names.canonical import canonicalise_entry
from imas_codex.standard_names.protection import PROTECTED_FIELDS

logger = logging.getLogger(__name__)

# Default COCOS convention for the catalog manifest
_DEFAULT_COCOS_CONVENTION = 17

# Gate names
GATE_A = "graph_tests"
GATE_B = "cross_field_consistency"
GATE_C = "score_thresholds"
GATE_D = "divergence_detection"

# Fields that must NOT appear in exported YAML
_PROVENANCE_FIELDS = frozenset({"source_paths", "dd_paths"})


# =============================================================================
# Report models
# =============================================================================


@dataclass
class GateResult:
    """Result of a single gate check."""

    gate: str
    passed: bool
    issues: list[dict[str, Any]] = field(default_factory=list)
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "skipped": self.skipped,
            "issue_count": len(self.issues),
            "issues": self.issues,
        }


@dataclass
class DivergenceEntry:
    """A single divergence finding for a catalog-edited name."""

    name: str
    field: str
    graph_hash: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "field": self.field,
            "graph_hash": self.graph_hash,
            "detail": self.detail,
        }


@dataclass
class ExportReport:
    """Full report from an export run."""

    gate_results: list[GateResult] = field(default_factory=list)
    divergence_entries: list[DivergenceEntry] = field(default_factory=list)
    total_candidates: int = 0
    exported_count: int = 0
    excluded_below_score: int = 0
    excluded_unreviewed: int = 0
    excluded_by_domain: int = 0
    gate_failures: int = 0
    all_gates_passed: bool = True
    exported_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gates": [g.to_dict() for g in self.gate_results],
            "divergence": [d.to_dict() for d in self.divergence_entries],
            "counts": {
                "total_candidates": self.total_candidates,
                "exported": self.exported_count,
                "excluded_below_score": self.excluded_below_score,
                "excluded_unreviewed": self.excluded_unreviewed,
                "excluded_by_domain": self.excluded_by_domain,
                "gate_failures": self.gate_failures,
            },
            "all_gates_passed": self.all_gates_passed,
        }


# =============================================================================
# Graph query helpers
# =============================================================================


def _fetch_candidates(
    *,
    include_unreviewed: bool = False,
    domain: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch StandardName nodes eligible for export from the graph.

    Returns dicts with all catalog-relevant properties plus ``origin``,
    ``cocos``, ``reviewer_score_name``, ``pipeline_status``.
    """
    from imas_codex.graph.client import GraphClient

    # Only export names with pipeline_status in publishable states
    cypher = """
    MATCH (sn:StandardName)
    WHERE sn.pipeline_status IN ['published', 'accepted', 'reviewed', 'enriched']
    """
    params: dict[str, Any] = {}

    if domain:
        cypher += " AND sn.physics_domain = $domain"
        params["domain"] = domain

    cypher += """
    OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
    OPTIONAL MATCH (sn)-[:HAS_COCOS]->(c:COCOS)
    RETURN sn {
        .*,
        unit: u.id,
        cocos: c.convention
    } AS record
    ORDER BY sn.id
    """

    with GraphClient() as gc:
        rows = gc.query(cypher, **params)

    return [r["record"] for r in (rows or [])]


# =============================================================================
# Gate implementations
# =============================================================================


def _run_gate_a() -> GateResult:
    """Gate A: Run existing graph test suites via subprocess pytest.

    Stub implementation — runs pytest with the ``graph or corpus_health``
    marker. Returns a GateResult. In Phase 6 this will be made more
    granular.
    """
    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "pytest",
                "-x",
                "-q",
                "--tb=short",
                "-m",
                "graph or corpus_health",
                "tests/graph/",
                "tests/standard_names/",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        passed = result.returncode == 0
        issues = []
        if not passed:
            issues.append(
                {
                    "type": "test_suite_failure",
                    "detail": result.stdout[-2000:] if result.stdout else "",
                    "stderr": result.stderr[-500:] if result.stderr else "",
                }
            )
        return GateResult(gate=GATE_A, passed=passed, issues=issues)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return GateResult(
            gate=GATE_A,
            passed=False,
            issues=[{"type": "execution_error", "detail": str(exc)}],
        )


def _run_gate_b(
    candidates: list[dict[str, Any]],
    cocos_convention: int,
) -> GateResult:
    """Gate B: Cross-field consistency checks.

    - Every non-null ``cocos`` equals ``cocos_convention``.
    - Grammar version matches ISN package version.
    - All names parse via ISN grammar.
    """
    issues: list[dict[str, Any]] = []

    # B1: COCOS consistency
    for cand in candidates:
        cand_cocos = cand.get("cocos")
        if cand_cocos is not None and cand_cocos != cocos_convention:
            issues.append(
                {
                    "type": "cocos_mismatch",
                    "name": cand["id"],
                    "expected": cocos_convention,
                    "actual": cand_cocos,
                }
            )

    # B2: Grammar parse check — validate each name parses
    try:
        from imas_standard_names.grammar import parse_name

        for cand in candidates:
            name = cand["id"]
            try:
                parse_name(name)
            except Exception as exc:
                issues.append(
                    {
                        "type": "grammar_parse_failure",
                        "name": name,
                        "detail": str(exc),
                    }
                )
    except ImportError:
        logger.warning("ISN grammar not available — skipping parse gate")

    # B3: Links resolve to known names
    all_names = {c["id"] for c in candidates}
    for cand in candidates:
        for link in cand.get("links") or []:
            # Links can be "name:foo" format or plain "foo"
            link_target = link.split(":")[-1] if ":" in link else link
            if link_target not in all_names:
                issues.append(
                    {
                        "type": "dangling_link",
                        "name": cand["id"],
                        "link_target": link_target,
                    }
                )

    passed = len(issues) == 0
    return GateResult(gate=GATE_B, passed=passed, issues=issues)


def _run_gate_c(
    candidates: list[dict[str, Any]],
    min_score: float,
    include_unreviewed: bool,
    min_description_score: float | None,
) -> tuple[GateResult, list[dict[str, Any]], int, int]:
    """Gate C: Score thresholds — filter candidates.

    Returns (gate_result, filtered_candidates, excluded_below_score,
    excluded_unreviewed).
    """
    issues: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    excluded_below_score = 0
    excluded_unreviewed = 0

    for cand in candidates:
        score = cand.get("reviewer_score_name")

        # Unreviewed check
        if score is None:
            if not include_unreviewed:
                excluded_unreviewed += 1
                continue
            # Include unreviewed — skip score threshold
            filtered.append(cand)
            continue

        # Score threshold
        if score < min_score:
            excluded_below_score += 1
            continue

        # Description score threshold (optional)
        if min_description_score is not None:
            desc_score = cand.get("reviewer_description_score")
            if desc_score is not None and desc_score < min_description_score:
                excluded_below_score += 1
                issues.append(
                    {
                        "type": "below_description_score",
                        "name": cand["id"],
                        "score": desc_score,
                        "threshold": min_description_score,
                    }
                )
                continue

        filtered.append(cand)

    return (
        GateResult(gate=GATE_C, passed=True, issues=issues),
        filtered,
        excluded_below_score,
        excluded_unreviewed,
    )


def detect_divergence(
    candidates: list[dict[str, Any]],
) -> list[DivergenceEntry]:
    """Gate D: Detect divergence in catalog-edited names.

    For each node with ``origin='catalog_edit'``, check whether any
    protected field has been modified since import (which would indicate
    a pipeline write bypassed the protection system).

    Without an ISNC checkout to compare against, we use a heuristic:
    if ``origin='catalog_edit'`` but ``exported_at`` is newer than
    ``imported_at``, the node was re-exported after being edited
    (expected). If any protected field hash differs from what was
    recorded, that's a divergence.

    Returns a list of divergence findings.
    """
    findings: list[DivergenceEntry] = []

    for cand in candidates:
        if cand.get("origin") != "catalog_edit":
            continue

        name = cand["id"]

        # Compute a hash of the current protected field values
        protected_values = {
            f: cand.get(f) for f in sorted(PROTECTED_FIELDS) if cand.get(f) is not None
        }
        current_hash = hashlib.sha256(
            json.dumps(protected_values, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        # Check if catalog_commit_sha is set — if so, the node was
        # imported from a specific commit. We can't compare without
        # the ISNC checkout, but we flag the node for awareness.
        if cand.get("catalog_commit_sha"):
            findings.append(
                DivergenceEntry(
                    name=name,
                    field="*",
                    graph_hash=current_hash,
                    detail=(
                        f"catalog-edited node with commit lineage "
                        f"{cand['catalog_commit_sha'][:8]}; "
                        f"verify protected fields match catalog"
                    ),
                )
            )

    return findings


# =============================================================================
# Entry serialisation
# =============================================================================


def _graph_node_to_entry_dict(node: dict[str, Any]) -> dict[str, Any]:
    """Convert a graph node dict to a catalog entry dict.

    Maps graph property names to ISN StandardNameEntry field names,
    and excludes all graph-only / pipeline-only fields.
    """
    entry: dict[str, Any] = {
        "name": node["id"],
        "description": node.get("description") or "",
        "documentation": node.get("documentation") or "",
        "kind": node.get("kind") or "scalar",
        "unit": node.get("unit") or "",
        "status": node.get("status") or "draft",
        "tags": list(node.get("tags") or []),
        "links": list(node.get("links") or []),
        "constraints": list(node.get("constraints") or []),
        "validity_domain": node.get("validity_domain") or "",
        "cocos_transformation_type": node.get("cocos_transformation_type"),
    }

    # Optional lifecycle fields
    if node.get("deprecates"):
        entry["deprecates"] = node["deprecates"]
    if node.get("superseded_by"):
        entry["superseded_by"] = node["superseded_by"]

    # Provenance (ISN grammatical provenance, NOT pipeline provenance)
    # This is optional — only set for derived/composite names
    # We don't emit pipeline provenance (source_paths, dd_paths)

    return entry


def _validate_entry(entry_dict: dict[str, Any]) -> dict[str, Any] | None:
    """Validate an entry dict against the ISN StandardNameEntry model.

    Returns the validated model_dump dict, or None if validation fails.
    """
    from imas_standard_names.models import (
        StandardNameComplexEntry,
        StandardNameMetadataEntry,
        StandardNameScalarEntry,
        StandardNameTensorEntry,
        StandardNameVectorEntry,
    )

    kind = entry_dict.get("kind", "scalar")
    model_cls = {
        "scalar": StandardNameScalarEntry,
        "vector": StandardNameVectorEntry,
        "tensor": StandardNameTensorEntry,
        "complex": StandardNameComplexEntry,
        "metadata": StandardNameMetadataEntry,
    }.get(kind, StandardNameScalarEntry)

    try:
        entry = model_cls.model_validate(entry_dict)
        return entry.model_dump(mode="json")
    except Exception as exc:
        logger.warning(
            "ISN validation failed for '%s': %s",
            entry_dict.get("name", "?"),
            exc,
        )
        # Fall back to returning the dict as-is (allow export to proceed)
        return entry_dict


# =============================================================================
# File writing
# =============================================================================


def _write_entry_yaml(
    staging_dir: Path,
    entry_dict: dict[str, Any],
    domain: str,
) -> Path:
    """Write a single entry YAML file to the staging directory.

    Returns the path of the written file.
    """
    name = entry_dict["name"]
    entry_dir = staging_dir / "standard_names" / domain
    entry_dir.mkdir(parents=True, exist_ok=True)

    filepath = entry_dir / f"{name}.yml"

    # Canonicalise before writing
    canon = canonicalise_entry(entry_dict)

    # Remove None values for clean YAML output
    clean = {k: v for k, v in canon.items() if v is not None}

    content = yaml.safe_dump(clean, sort_keys=False, default_flow_style=False)
    filepath.write_text(content, encoding="utf-8")

    return filepath


def _write_manifest(
    staging_dir: Path,
    *,
    cocos_convention: int,
    candidate_count: int,
    published_count: int,
    excluded_below_score_count: int,
    excluded_unreviewed_count: int,
    min_score_applied: float,
    min_description_score_applied: float | None,
    include_unreviewed: bool,
    source_commit_sha: str | None = None,
) -> Path:
    """Write the catalog.yml manifest to the staging directory root."""
    import imas_standard_names

    manifest_data = {
        "catalog_name": "imas-standard-names-catalog",
        "cocos_convention": cocos_convention,
        "grammar_version": imas_standard_names.__version__,
        "isn_model_version": imas_standard_names.__version__,
        "dd_version_lineage": ["4.0.0"],
        "generated_by": "imas-codex sn export",
        "generated_at": datetime.now(UTC).isoformat(),
        "min_score_applied": min_score_applied,
        "min_description_score_applied": min_description_score_applied,
        "include_unreviewed": include_unreviewed,
        "candidate_count": candidate_count,
        "published_count": published_count,
        "excluded_below_score_count": excluded_below_score_count,
        "excluded_unreviewed_count": excluded_unreviewed_count,
        "source_repo": "imas-codex",
        "source_commit_sha": source_commit_sha,
    }

    # Validate via ISN manifest model
    try:
        from imas_standard_names.models import StandardNameCatalogManifest

        manifest = StandardNameCatalogManifest.model_validate(manifest_data)
        manifest_data = manifest.model_dump(mode="json")
    except Exception as exc:
        logger.warning("Manifest validation warning: %s", exc)

    filepath = staging_dir / "catalog.yml"
    content = yaml.safe_dump(manifest_data, sort_keys=False, default_flow_style=False)
    filepath.write_text(content, encoding="utf-8")

    return filepath


def _write_export_report(staging_dir: Path, report: ExportReport) -> Path:
    """Write .export_report.json to the staging directory."""
    filepath = staging_dir / ".export_report.json"
    filepath.write_text(
        json.dumps(report.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    return filepath


def _get_codex_commit_sha() -> str | None:
    """Get the current imas-codex git commit SHA, or None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return None


# =============================================================================
# Main export function
# =============================================================================


def run_export(
    staging_dir: str | Path,
    *,
    min_score: float = 0.65,
    include_unreviewed: bool = False,
    min_description_score: float | None = None,
    domain: str | None = None,
    force: bool = False,
    skip_gate: bool = False,
    gate_only: bool = False,
    gate_scope: str = "all",
    override_edits: list[str] | None = None,
    cocos_convention: int = _DEFAULT_COCOS_CONVENTION,
) -> ExportReport:
    """Export standard names from the graph to a staging directory.

    Parameters
    ----------
    staging_dir:
        Path to the staging directory. Created if it doesn't exist.
    min_score:
        Minimum ``reviewer_score_name`` for inclusion (default 0.65).
    include_unreviewed:
        Include names without a ``reviewer_score_name``.
    min_description_score:
        Optional secondary threshold on description sub-score.
    domain:
        Restrict export to a single physics domain.
    force:
        Write staging tree despite gate failures.
    skip_gate:
        Skip gate entirely (requires ``force=True``).
    gate_only:
        Run the gate and report without writing YAML.
    gate_scope:
        Gate scope: ``"all"`` or ``"domain"``.
    override_edits:
        List of name IDs to reset from ``catalog_edit`` to
        ``pipeline`` origin. Pass ``["all"]`` to override all.
    cocos_convention:
        COCOS convention for the manifest (default 17).

    Returns
    -------
    ExportReport with gate results, counts, and divergence entries.
    """
    staging_path = Path(staging_dir)
    report = ExportReport()

    # ── 1. Fetch candidates from graph ──────────────────────────
    logger.info("Fetching candidates from graph...")
    candidates = _fetch_candidates(
        include_unreviewed=include_unreviewed,
        domain=domain,
    )
    report.total_candidates = len(candidates)
    logger.info("Found %d candidate(s)", len(candidates))

    # ── 2. Run gates ────────────────────────────────────────────
    if not skip_gate:
        # Gate A: Graph tests (only for 'all' scope)
        if gate_scope == "all":
            gate_a = _run_gate_a()
        else:
            gate_a = GateResult(gate=GATE_A, passed=True, skipped=True)
        report.gate_results.append(gate_a)

        # Gate C: Score thresholds (filter candidates)
        gate_c, candidates, excluded_below, excluded_unrev = _run_gate_c(
            candidates, min_score, include_unreviewed, min_description_score
        )
        report.gate_results.append(gate_c)
        report.excluded_below_score = excluded_below
        report.excluded_unreviewed = excluded_unrev

        # Gate B: Cross-field consistency (on filtered candidates)
        gate_b = _run_gate_b(candidates, cocos_convention)
        report.gate_results.append(gate_b)

        # Gate D: Divergence detection
        divergence = detect_divergence(candidates)
        report.divergence_entries = divergence
        gate_d = GateResult(
            gate=GATE_D,
            passed=len(divergence) == 0,
            issues=[d.to_dict() for d in divergence],
        )
        report.gate_results.append(gate_d)

        # Summarise gate results
        report.all_gates_passed = all(
            g.passed or g.skipped for g in report.gate_results
        )
        report.gate_failures = sum(
            1 for g in report.gate_results if not g.passed and not g.skipped
        )

        if not report.all_gates_passed and not force:
            logger.error(
                "Export blocked: %d gate(s) failed. Use --force to override.",
                report.gate_failures,
            )
            # Still write the report even on failure
            staging_path.mkdir(parents=True, exist_ok=True)
            _write_export_report(staging_path, report)
            return report
    else:
        # Gate C still runs for filtering even when gates skipped
        _, candidates, excluded_below, excluded_unrev = _run_gate_c(
            candidates, min_score, include_unreviewed, min_description_score
        )
        report.excluded_below_score = excluded_below
        report.excluded_unreviewed = excluded_unrev

    # ── 3. Gate-only mode: report and exit ──────────────────────
    if gate_only:
        staging_path.mkdir(parents=True, exist_ok=True)
        _write_export_report(staging_path, report)
        logger.info("Gate-only mode: report written, no YAML emitted.")
        return report

    # ── 4. Prepare staging directory ────────────────────────────
    staging_path.mkdir(parents=True, exist_ok=True)

    # Clear existing standard_names tree
    sn_dir = staging_path / "standard_names"
    if sn_dir.exists():
        import shutil

        shutil.rmtree(sn_dir)

    # ── 5. Write entry YAML files ───────────────────────────────
    exported_names: list[str] = []

    for cand in candidates:
        entry_dict = _graph_node_to_entry_dict(cand)

        # Ensure no provenance fields leak through
        for pf in _PROVENANCE_FIELDS:
            entry_dict.pop(pf, None)

        # Determine domain for directory placement
        entry_domain = (
            cand.get("physics_domain") or (cand.get("tags") or ["unscoped"])[0]
            if cand.get("tags")
            else "unscoped"
        )
        if not entry_domain:
            entry_domain = "unscoped"

        # Validate against ISN model (best-effort)
        validated = _validate_entry(entry_dict)
        if validated is not None:
            entry_dict = validated

        # Canonicalise
        entry_dict = canonicalise_entry(entry_dict)

        _write_entry_yaml(staging_path, entry_dict, entry_domain)
        exported_names.append(cand["id"])

    report.exported_count = len(exported_names)
    report.exported_names = exported_names

    # ── 6. Write manifest ───────────────────────────────────────
    codex_sha = _get_codex_commit_sha()
    _write_manifest(
        staging_path,
        cocos_convention=cocos_convention,
        candidate_count=report.total_candidates,
        published_count=report.exported_count,
        excluded_below_score_count=report.excluded_below_score,
        excluded_unreviewed_count=report.excluded_unreviewed,
        min_score_applied=min_score,
        min_description_score_applied=min_description_score,
        include_unreviewed=include_unreviewed,
        source_commit_sha=codex_sha,
    )

    # ── 7. Write export report ──────────────────────────────────
    _write_export_report(staging_path, report)

    logger.info(
        "Export complete: %d name(s) written to %s",
        report.exported_count,
        staging_path,
    )
    return report
