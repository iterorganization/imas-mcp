"""Layer 3 cross-batch consolidation for ``sn review``.

Runs after all LLM review batches complete.  Purely deterministic — no LLM
calls, no graph queries.  Takes Layer 1 audit results and Layer 2 review
scores to produce a :class:`ReviewSummary` report.
"""

from __future__ import annotations

import logging
import statistics
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from imas_codex.standard_names.review.audits import DuplicateComponent
    from imas_codex.standard_names.review.state import StandardNameReviewState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

_POSITION_SUFFIXES = (
    "_at_magnetic_axis",
    "_on_axis",
    "_at_midplane",
    "_at_boundary",
    "_at_separatrix",
    "_at_lcfs",
    "_at_pedestal",
    "_on_midplane",
    "_at_wall",
    "_at_limiter",
)

_PROCESS_SUFFIXES = (
    "_reconstructed",
    "_measured",
    "_computed",
    "_simulated",
    "_predicted",
    "_modelled",
    "_modeled",
    "_fitted",
    "_inferred",
)


class DuplicateReport(BaseModel):
    """Unresolved duplicate component with review context."""

    names: list[str]
    max_similarity: float
    reviewer_confirmed: bool = False  # True if LLM reviewer flagged it
    reviewer_dismissed: bool = False  # True if reviewer said they're distinct
    detail: str = ""


class DriftWarning(BaseModel):
    """Convention inconsistency within a physics domain."""

    physics_domain: str
    drift_type: str  # "mixed_position", "inconsistent_process", "doc_depth_variance"
    examples: list[str]  # name IDs showing the issue
    detail: str


class OutlierReport(BaseModel):
    """Score outlier within its cluster/domain."""

    name_id: str
    score: float
    cluster_mean: float
    cluster_std: float
    z_score: float
    recommendation: str  # "regenerate", "review", "investigate"


class ReviewSummary(BaseModel):
    """Final summary report from ``sn review``."""

    total_reviewed: int = 0
    total_cost: float = 0.0
    tier_distribution: dict[str, int] = Field(default_factory=dict)  # tier -> count
    duplicate_candidates: list[DuplicateReport] = Field(default_factory=list)
    drift_warnings: list[DriftWarning] = Field(default_factory=list)
    outliers: list[OutlierReport] = Field(default_factory=list)
    lowest_scorers: list[dict] = Field(default_factory=list)  # [{name, score, tier}]
    coverage_pct: float = 0.0  # % of catalog reviewed this pass
    total_catalog_size: int = 0


# ---------------------------------------------------------------------------
# Layer 3 functions
# ---------------------------------------------------------------------------


def resolve_duplicates(
    components: list[DuplicateComponent],
    review_results: list[dict],
) -> list[DuplicateReport]:
    """Map Layer 1 duplicate components to :class:`DuplicateReport` objects.

    For each component, searches *review_results* for reviewer comments that
    either confirm or dismiss the duplicate relationship.

    Args:
        components: ``DuplicateComponent`` objects from Layer 1 audits.
        review_results: Per-name review dicts from Layer 2 (must have
            ``reviewer_comments`` key).

    Returns:
        List of :class:`DuplicateReport`, one per component.
    """
    reports: list[DuplicateReport] = []

    for comp in components:
        name_set = set(comp.names)
        confirmed = False
        dismissed = False
        detail_parts: list[str] = []

        for result in review_results:
            comment: str = (result.get("reviewer_comments") or "").lower()
            if not comment:
                continue

            # Check whether this result concerns any name in the component
            result_name: str = result.get("name_id") or result.get("id") or ""
            if result_name not in name_set:
                continue

            # "duplicate" keyword → confirmed
            if "duplicate" in comment:
                confirmed = True
                detail_parts.append(
                    f"reviewer flagged '{result_name}' as duplicate: {comment[:120]}"
                )

            # "distinct" or "different" keyword → dismissed
            if "distinct" in comment or "different" in comment:
                dismissed = True
                detail_parts.append(
                    f"reviewer dismissed duplicate for '{result_name}': {comment[:120]}"
                )

        reports.append(
            DuplicateReport(
                names=comp.names,
                max_similarity=comp.max_similarity,
                reviewer_confirmed=confirmed,
                reviewer_dismissed=dismissed,
                detail="; ".join(detail_parts),
            )
        )

    logger.debug(
        "resolve_duplicates: %d components → %d reports", len(components), len(reports)
    )
    return reports


def detect_convention_drift(reviewed_names: list[dict]) -> list[DriftWarning]:
    """Detect naming-convention inconsistencies within each physics domain.

    Three drift types are checked:

    * **mixed_position** — multiple position-suffix forms coexist in a domain.
    * **inconsistent_process** — process suffixes present for some names but
      not all names sharing the same physical base within a domain.
    * **doc_depth_variance** — documentation length has a high coefficient of
      variation (> 1.0) within a domain.

    Args:
        reviewed_names: Per-name review dicts; each must have at least ``id``
            and ``physics_domain``.  Documentation length is read from
            ``documentation`` or ``description``.

    Returns:
        List of :class:`DriftWarning`.
    """
    # Group by physics_domain
    by_domain: dict[str, list[dict]] = {}
    for name in reviewed_names:
        domain: str = (name.get("physics_domain") or "unknown").strip()
        by_domain.setdefault(domain, []).append(name)

    warnings: list[DriftWarning] = []

    for domain, names in by_domain.items():
        name_ids = [n.get("name_id") or n.get("id") or "" for n in names]

        # ----------------------------------------------------------------
        # a) Mixed position suffixes
        # ----------------------------------------------------------------
        pos_forms_found: dict[str, list[str]] = {}  # suffix → list of name_ids
        for nid in name_ids:
            for suffix in _POSITION_SUFFIXES:
                if nid.endswith(suffix):
                    pos_forms_found.setdefault(suffix, []).append(nid)
                    break

        if len(pos_forms_found) > 1:
            examples: list[str] = []
            for ex_list in pos_forms_found.values():
                examples.extend(ex_list[:2])
            warnings.append(
                DriftWarning(
                    physics_domain=domain,
                    drift_type="mixed_position",
                    examples=examples[:10],
                    detail=(
                        f"Multiple position suffix forms used in '{domain}': "
                        + ", ".join(pos_forms_found.keys())
                    ),
                )
            )

        # ----------------------------------------------------------------
        # b) Inconsistent process suffixes
        # ----------------------------------------------------------------
        # Detect physical_base collisions: names that differ only by process suffix
        base_map: dict[str, list[str]] = {}  # stripped base → name_ids
        for nid in name_ids:
            base = nid
            for suffix in _PROCESS_SUFFIXES:
                if nid.endswith(suffix):
                    base = nid[: -len(suffix)]
                    break
            base_map.setdefault(base, []).append(nid)

        inconsistent: list[str] = []
        for _base, members in base_map.items():
            if len(members) < 2:
                continue
            has_suffix = any(
                any(m.endswith(s) for s in _PROCESS_SUFFIXES) for m in members
            )
            has_plain = any(
                not any(m.endswith(s) for s in _PROCESS_SUFFIXES) for m in members
            )
            if has_suffix and has_plain:
                inconsistent.extend(members[:4])

        if inconsistent:
            warnings.append(
                DriftWarning(
                    physics_domain=domain,
                    drift_type="inconsistent_process",
                    examples=inconsistent[:10],
                    detail=(
                        f"Domain '{domain}' has names with and without process "
                        "suffixes sharing the same physical base."
                    ),
                )
            )

        # ----------------------------------------------------------------
        # c) Documentation depth variance
        # ----------------------------------------------------------------
        doc_lengths: list[int] = []
        for n in names:
            doc = n.get("documentation") or n.get("description") or ""
            doc_lengths.append(len(doc))

        if len(doc_lengths) >= 3:
            mean_len = statistics.mean(doc_lengths)
            if mean_len > 0:
                std_len = statistics.stdev(doc_lengths)
                cv = std_len / mean_len
                if cv > 1.0:
                    # Pick examples: shortest and longest docs
                    sorted_names = sorted(
                        zip(doc_lengths, name_ids, strict=False), key=lambda x: x[0]
                    )
                    examples_cv = [
                        sorted_names[0][1],
                        sorted_names[-1][1],
                    ]
                    warnings.append(
                        DriftWarning(
                            physics_domain=domain,
                            drift_type="doc_depth_variance",
                            examples=examples_cv,
                            detail=(
                                f"Domain '{domain}' has high doc-length variance "
                                f"(CV={cv:.2f}, mean={mean_len:.0f} chars)."
                            ),
                        )
                    )

    logger.debug(
        "detect_convention_drift: %d domains → %d warnings",
        len(by_domain),
        len(warnings),
    )
    return warnings


def detect_score_outliers(reviewed_names: list[dict]) -> list[OutlierReport]:
    """Identify names whose reviewer score is unusually low within their group.

    Names are grouped by ``physics_domain`` (falling back to ``cluster`` if
    available).  Groups with fewer than 3 members are skipped.

    Args:
        reviewed_names: Per-name review dicts with ``reviewer_score`` and
            ``physics_domain`` (or ``cluster``) fields.

    Returns:
        List of :class:`OutlierReport` for names scoring > 1σ below group mean.
    """
    # Group by physics_domain (or cluster as fallback)
    by_group: dict[str, list[dict]] = {}
    for name in reviewed_names:
        group_key: str = (
            name.get("physics_domain") or name.get("cluster") or "unknown"
        ).strip()
        by_group.setdefault(group_key, []).append(name)

    outliers: list[OutlierReport] = []

    for _group_key, members in by_group.items():
        scores: list[float] = []
        for m in members:
            raw = m.get("reviewer_score")
            if raw is not None:
                try:
                    scores.append(float(raw))
                except (TypeError, ValueError):
                    pass

        if len(scores) < 3:
            continue

        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores)

        if std_score == 0:
            continue

        for member, score in zip(members, scores, strict=False):
            z = (score - mean_score) / std_score
            if z < -1.0:
                if z < -2.0:
                    rec = "regenerate"
                elif z < -1.5:
                    rec = "review"
                else:
                    rec = "investigate"

                name_id: str = member.get("name_id") or member.get("id") or ""
                outliers.append(
                    OutlierReport(
                        name_id=name_id,
                        score=score,
                        cluster_mean=mean_score,
                        cluster_std=std_score,
                        z_score=round(z, 3),
                        recommendation=rec,
                    )
                )

    logger.debug("detect_score_outliers: %d outliers found", len(outliers))
    return outliers


def build_summary_report(
    all_names: list[dict],
    reviewed_names: list[dict],
    duplicate_reports: list[DuplicateReport],
    drift_warnings: list[DriftWarning],
    outliers: list[OutlierReport],
    total_cost: float = 0.0,
) -> ReviewSummary:
    """Assemble the final :class:`ReviewSummary`.

    Args:
        all_names: Full catalog (used for coverage calculation).
        reviewed_names: Names processed in this review pass.
        duplicate_reports: From :func:`resolve_duplicates`.
        drift_warnings: From :func:`detect_convention_drift`.
        outliers: From :func:`detect_score_outliers`.
        total_cost: Accumulated LLM spend from the budget manager.

    Returns:
        A fully populated :class:`ReviewSummary`.
    """
    # Tier distribution
    tier_distribution: dict[str, int] = {}
    for name in reviewed_names:
        tier: str = name.get("review_tier") or "unknown"
        tier_distribution[tier] = tier_distribution.get(tier, 0) + 1

    # Lowest scorers — bottom 10 by reviewer_score
    scoreable = [n for n in reviewed_names if n.get("reviewer_score") is not None]
    scoreable.sort(key=lambda n: float(n.get("reviewer_score", 0)))
    lowest_scorers: list[dict[str, Any]] = [
        {
            "name": n.get("name_id") or n.get("id") or "",
            "score": n.get("reviewer_score"),
            "tier": n.get("review_tier") or "unknown",
        }
        for n in scoreable[:10]
    ]

    # Coverage
    catalog_size = len(all_names)
    coverage_pct = (
        (len(reviewed_names) / catalog_size * 100.0) if catalog_size > 0 else 0.0
    )

    summary = ReviewSummary(
        total_reviewed=len(reviewed_names),
        total_cost=total_cost,
        tier_distribution=tier_distribution,
        duplicate_candidates=duplicate_reports,
        drift_warnings=drift_warnings,
        outliers=outliers,
        lowest_scorers=lowest_scorers,
        coverage_pct=round(coverage_pct, 2),
        total_catalog_size=catalog_size,
    )
    logger.debug(
        "build_summary_report: %d reviewed, %.1f%% coverage, cost=$%.4f",
        summary.total_reviewed,
        summary.coverage_pct,
        summary.total_cost,
    )
    return summary


def run_consolidation(state: StandardNameReviewState) -> ReviewSummary:
    """Orchestrate the full Layer 3 consolidation.

    Calls :func:`resolve_duplicates`, :func:`detect_convention_drift`,
    :func:`detect_score_outliers`, and :func:`build_summary_report` in
    sequence and returns the assembled :class:`ReviewSummary`.

    Args:
        state: The shared :class:`~imas_codex.standard_names.review.state.StandardNameReviewState`
            after all Layer 2 batches have completed.

    Returns:
        Final :class:`ReviewSummary` report.
    """
    logger.info(
        "Layer 3 consolidation started (%d review results)", len(state.review_results)
    )

    # 1. Resolve duplicates from Layer 1 audit
    duplicate_components = (
        state.audit_report.duplicate_components if state.audit_report else []
    )
    duplicate_reports = resolve_duplicates(duplicate_components, state.review_results)

    # 2. Convention drift detection
    drift_warnings = detect_convention_drift(state.review_results)

    # 3. Score outlier detection
    outliers = detect_score_outliers(state.review_results)

    # 4. Total cost from budget manager or accumulated stats
    total_cost = state.total_cost

    # 5. Assemble final report
    summary = build_summary_report(
        all_names=state.all_names,
        reviewed_names=state.review_results,
        duplicate_reports=duplicate_reports,
        drift_warnings=drift_warnings,
        outliers=outliers,
        total_cost=total_cost,
    )

    logger.info(
        "Layer 3 consolidation complete: %d reviewed, %d duplicates, "
        "%d drift warnings, %d outliers",
        summary.total_reviewed,
        len(summary.duplicate_candidates),
        len(summary.drift_warnings),
        len(summary.outliers),
    )
    return summary
