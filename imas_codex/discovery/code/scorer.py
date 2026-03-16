"""Dual-pass LLM file scoring for discovered CodeFiles.

Pass 1 (Triage): Per-dimension scoring from minimal context — parent
directory description + filename + extension.  Quick and cheap.
Files triaging over threshold proceed to enrichment+scoring.

Pass 2 (Score): Full multi-dimensional scoring with enrichment evidence
(rg pattern matches, file preview text) for files that passed triage.
The scorer receives the triage description but NOT the triage numeric
scores — it makes an independent assessment.

Both passes inject dynamic calibration examples sampled from previously-
scored CodeFiles in the graph (same pattern as the paths pipeline).
Calibration examples evolve as more files are scored (60s TTL cache).

Lifecycle: discovered → triaged → (enrich) → scored → ingested | skipped
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from imas_codex.discovery.base.scoring import (
    CODE_SCORE_DIMENSIONS,
    CodeScoreFields,
    max_composite,
)
from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score dimensions — canonical list from shared scoring module
# ---------------------------------------------------------------------------

SCORE_DIMENSION_NAMES = CODE_SCORE_DIMENSIONS

TRIAGE_DIMENSION_NAMES = [d.replace("score_", "triage_") for d in SCORE_DIMENSION_NAMES]


# ---------------------------------------------------------------------------
# Dynamic calibration (same architecture as paths/frontier.py)
# ---------------------------------------------------------------------------

_calibration_cache: dict[str, tuple[float, dict]] = {}
_CALIBRATION_TTL_SECONDS = 300.0  # 5 minutes — matches LLM provider ephemeral cache TTL


def sample_code_dimension_calibration(
    facility: str | None = None,
    per_level: int = 3,
    phase: str = "score",
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Sample calibration examples per score dimension at 5 levels.

    Triage and score operate on different cohorts of the same 0-1 scale:

    * **triage** -- draws ``triage_*`` from ALL triaged+scored CodeFiles
      (general population).  Shows the full distribution.
    * **score** -- draws ``score_*`` from scored-only CodeFiles (graduate
      cohort that passed triage+enrichment).  Calibrates among peers.

    Cached with 60s TTL -- stable within a batch, evolves over time.

    Returns:
        Nested dict: dimension -> level -> list of examples.
        Each example: path, facility, score, purpose, description.
    """
    global _calibration_cache  # noqa: PLW0603

    cache_key = f"{phase}:{facility}:{per_level}"
    now = time.monotonic()

    if cache_key in _calibration_cache:
        cached_time, cached_data = _calibration_cache[cache_key]
        if (now - cached_time) < _CALIBRATION_TTL_SECONDS:
            return cached_data

    samples = _fetch_code_dimension_calibration(facility, per_level, phase)
    _calibration_cache[cache_key] = (now, samples)
    return samples


def _fetch_code_dimension_calibration(
    facility: str | None,
    per_level: int,
    phase: str,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Fetch dimension calibration from CodeFile nodes (uncached).

    For triage: uses triage_* properties from all triaged+scored files
    For score: uses score_* properties from scored files only
    """
    if phase == "triage":
        status_clause = "cf.status IN ['triaged', 'scored', 'ingested']"
        dims_to_query = TRIAGE_DIMENSION_NAMES
    else:
        status_clause = "cf.status IN ['scored', 'ingested']"
        dims_to_query = SCORE_DIMENSION_NAMES

    buckets: list[tuple[str, float, float]] = [
        ("lowest", 0.0, 0.15),
        ("low", 0.10, 0.30),
        ("medium", 0.40, 0.60),
        ("high", 0.70, 0.90),
        ("highest", 0.90, 1.01),
    ]

    samples: dict[str, dict[str, list[dict[str, Any]]]] = {}

    with GraphClient() as gc:
        for dim, graph_prop in zip(SCORE_DIMENSION_NAMES, dims_to_query, strict=True):
            samples[dim] = {}

            for level_name, min_score, max_score in buckets:
                desc_prop = (
                    "cf.triage_description" if phase == "triage" else "cf.score_reason"
                )

                target = (min_score + max_score) / 2
                result = gc.query(
                    f"""
                    MATCH (cf:CodeFile)
                    WHERE {status_clause}
                        AND cf.{graph_prop} >= $min_score
                        AND cf.{graph_prop} < $max_score
                        AND cf.{graph_prop} IS NOT NULL
                    RETURN cf.path AS path,
                           cf.facility_id AS facility,
                           cf.{graph_prop} AS score,
                           {desc_prop} AS description
                    ORDER BY
                        CASE WHEN cf.facility_id = $facility
                             THEN 0 ELSE 1 END,
                        abs(cf.{graph_prop} - $target) ASC,
                        cf.id ASC
                    LIMIT $limit
                    """,
                    min_score=min_score,
                    max_score=max_score,
                    target=target,
                    facility=facility or "",
                    limit=per_level,
                )

                samples[dim][level_name] = [
                    {
                        "path": r["path"],
                        "facility": r["facility"],
                        "score": round(r["score"], 2),
                        "purpose": "code file",
                        "description": r["description"] or "",
                    }
                    for r in result
                ]

    return samples


# ---------------------------------------------------------------------------
# Pass 1: Triage models (per-dimension scoring, not keep/skip)
# ---------------------------------------------------------------------------


class FileTriageResult(CodeScoreFields):
    """Triage result for a single file -- per-dimension scores from minimal context.

    Inherits 9 score dimensions from CodeScoreFields.
    """

    path: str = Field(description="The file path (echo from input)")
    description: str = Field(
        default="",
        description="Brief description of what the file likely contains (1 sentence)",
    )

    @property
    def triage_composite(self) -> float:
        """Composite = max of all dimension scores."""
        return max_composite(self.get_score_dict())


class FileTriageBatch(BaseModel):
    """Batch of triage results from LLM."""

    results: list[FileTriageResult]


# ---------------------------------------------------------------------------
# Pass 2: Score models (full scoring with enrichment evidence)
# ---------------------------------------------------------------------------


class FileScoreResult(CodeScoreFields):
    """Full scoring result with enrichment evidence.

    Inherits 9 score dimensions from CodeScoreFields.
    """

    path: str = Field(description="The file path (echo from input)")
    file_category: str = Field(
        description="code, document, notebook, config, data, or other"
    )
    description: str = Field(
        default="",
        description="Brief summary of what the file likely contains (1 sentence)",
    )
    skip: bool = Field(default=False, description="Whether to skip this file entirely")

    @property
    def score_composite(self) -> float:
        """Composite = max of all dimension scores."""
        return max_composite(self.get_score_dict())


class FileScoreBatch(BaseModel):
    """Batch of file scoring results from LLM."""

    results: list[FileScoreResult]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_triage_system_prompt(
    facility: str | None = None,
    focus: str | None = None,
) -> str:
    """Build triage system prompt with dimension calibration.

    Triage calibration draws from the full population (all triaged+scored
    CodeFiles) using triage_* properties.
    """
    from imas_codex.llm.prompt_loader import render_prompt

    context: dict[str, Any] = {}
    if focus:
        context["focus"] = focus

    dimension_calibration = sample_code_dimension_calibration(
        facility=facility, per_level=5, phase="triage"
    )
    has_calibration = any(
        any(examples for examples in dim_levels.values())
        for dim_levels in dimension_calibration.values()
    )
    if has_calibration:
        context["dimension_calibration"] = dimension_calibration

    return render_prompt("code/triage", context)


def _build_score_system_prompt(
    facility: str | None = None,
    focus: str | None = None,
) -> str:
    """Build scorer system prompt with dimension calibration.

    Score calibration draws from the graduate cohort (scored-only
    CodeFiles) using score_* properties.
    """
    from imas_codex.llm.prompt_loader import render_prompt

    context: dict[str, Any] = {}
    if focus:
        context["focus"] = focus

    dimension_calibration = sample_code_dimension_calibration(
        facility=facility, per_level=5, phase="score"
    )
    has_calibration = any(
        any(examples for examples in dim_levels.values())
        for dim_levels in dimension_calibration.values()
    )
    if has_calibration:
        context["dimension_calibration"] = dimension_calibration

    return render_prompt("code/scorer", context)


def _build_triage_user_prompt(file_groups: list[dict[str, Any]]) -> str:
    """Build triage user prompt -- minimal context + sibling awareness.

    Each directory group provides:
    - Parent dir full path + description (labelled as directory-level)
    - Complete sibling file listing (all known files in dir)
    - Indication of which files need triaging vs already processed

    NO enrichment patterns, NO numeric scores from paths.
    Sibling file names give batch context -- seeing `read_mds.py`
    alongside `plot_profiles.py` is more informative than either alone.
    """
    lines = ["Triage these files. Each group shares a parent directory.\n"]

    for i, group in enumerate(file_groups, 1):
        parent_path = group.get("parent_path", "unknown")
        parent_desc = group.get("parent_description") or ""
        sibling_names = group.get("sibling_names") or []

        lines.append(f"\n## Directory {i}: {parent_path}")
        if parent_desc:
            lines.append(
                f"Directory description (applies to the directory as a whole, "
                f"not necessarily to each individual file): {parent_desc}"
            )

        # Show all siblings for neighborhood context
        if sibling_names:
            triage_paths = {f["path"] for f in group.get("files", [])}
            other_siblings = [s for s in sibling_names if s not in triage_paths]
            if other_siblings:
                lines.append(
                    f"\nOther files in this directory (already processed, "
                    f"for context only): {', '.join(other_siblings)}"
                )

        lines.append("\nFiles to triage:")
        for f in group.get("files", []):
            lang = f.get("language") or "unknown"
            lines.append(f"  - {f['path']} ({lang})")

    return "\n".join(lines)


def _build_score_user_prompt(file_groups: list[dict[str, Any]]) -> str:
    """Build scorer user prompt -- enrichment evidence + triage description.

    Each file gets: parent dir path + description, per-file enrichment
    (pattern matches, line count), triage description (qualitative,
    NO triage numeric scores), and preview text.
    """
    lines = ["Score these files using their enrichment evidence and content preview.\n"]

    for i, group in enumerate(file_groups, 1):
        parent_path = group.get("parent_path", "unknown")
        parent_desc = group.get("parent_description") or ""

        lines.append(f"\n## Directory {i}: {parent_path}")
        if parent_desc:
            lines.append(f"Directory description: {parent_desc}")

        lines.append("\nFiles:")
        for f in group.get("files", []):
            lang = f.get("language") or "unknown"
            line_count = f.get("line_count") or 0
            patterns = f.get("pattern_categories") or {}
            total_matches = f.get("total_pattern_matches") or 0
            triage_desc = f.get("triage_description") or ""
            preview = f.get("preview_text") or ""

            parts = [f"\n  ### {f['path']} ({lang}, {line_count} lines)"]

            if triage_desc:
                parts.append(f"  Triage assessment: {triage_desc}")

            if total_matches > 0:
                pattern_str = ", ".join(f"{k}: {v}" for k, v in patterns.items() if v)
                parts.append(
                    f"  Pattern matches: {pattern_str} (total: {total_matches})"
                )

            if preview:
                truncated = preview[:500]
                if len(preview) > 500:
                    truncated += "..."
                parts.append(f"  Content preview:\n  ```\n  {truncated}\n  ```")

            lines.append("\n".join(parts))

    return "\n".join(lines)


def _group_files_by_parent(
    files: list[dict],
    include_siblings: bool = False,
) -> list[dict[str, Any]]:
    """Group files by their parent FacilityPath.

    When ``include_siblings=True``, queries the graph for ALL CodeFile
    names under each parent directory.  This gives the triage LLM
    neighborhood context -- seeing what other files exist alongside
    the ones being triaged.

    Returns list of group dicts with parent context and file lists.
    """
    groups: dict[str, dict[str, Any]] = {}

    for f in files:
        parent_id = f.get("parent_path_id", "unknown")
        if parent_id not in groups:
            groups[parent_id] = {
                "parent_path_id": parent_id,
                "parent_path": f.get("parent_path") or "unknown",
                "parent_description": f.get("parent_description") or "",
                "files": [],
                "sibling_names": [],
            }

        file_entry: dict[str, Any] = {
            "id": f["id"],
            "path": f["path"],
            "language": f.get("language") or "unknown",
        }
        for key in (
            "line_count",
            "pattern_categories",
            "total_pattern_matches",
            "triage_description",
            "preview_text",
        ):
            if key in f:
                file_entry[key] = f[key]

        groups[parent_id]["files"].append(file_entry)

    # Fetch sibling file names for triage context
    if include_siblings and groups:
        parent_ids = list(groups.keys())
        with GraphClient() as gc:
            rows = gc.query(
                """
                UNWIND $parent_ids AS pid
                MATCH (cf:CodeFile)-[:IN_DIRECTORY]->(fp:FacilityPath {id: pid})
                RETURN pid AS parent_id, cf.path AS path
                """,
                parent_ids=parent_ids,
            )
            for row in rows:
                pid = row["parent_id"]
                if pid in groups:
                    groups[pid]["sibling_names"].append(row["path"])

    return list(groups.values())


# ---------------------------------------------------------------------------
# Graph persistence
# ---------------------------------------------------------------------------


def apply_triage_results(
    results: list[FileTriageResult],
    file_id_map: dict[str, str],
    threshold: float | None = None,
    batch_cost: float = 0.0,
) -> dict[str, Any]:
    """Persist triage dimension scores to CodeFile nodes.

    Files triaging above threshold -> status 'triaged' (proceed to enrichment).
    Files below threshold -> status 'skipped'.

    Args:
        results: Triage results from LLM.
        file_id_map: Mapping from path to CodeFile ID.
        threshold: Minimum triage composite to proceed.
            Defaults to ``get_triage_threshold()``.
        batch_cost: Total LLM cost for the batch, distributed per-file.

    Returns dict with triaged, skipped counts and triaged_ids.
    """
    if threshold is None:
        from imas_codex.settings import get_triage_threshold

        threshold = get_triage_threshold()
    triaged_items = []
    skipped_items = []

    matched_count = sum(1 for r in results if file_id_map.get(r.path))
    cost_per_file = batch_cost / matched_count if matched_count > 0 else 0.0

    for result in results:
        sf_id = file_id_map.get(result.path)
        if not sf_id:
            continue

        composite = result.triage_composite
        item = {
            "id": sf_id,
            "score_cost": cost_per_file,
            "triage_composite": round(composite, 4),
            "triage_description": result.description,
            "triage_modeling_code": result.score_modeling_code,
            "triage_analysis_code": result.score_analysis_code,
            "triage_operations_code": result.score_operations_code,
            "triage_data_access": result.score_data_access,
            "triage_workflow": result.score_workflow,
            "triage_visualization": result.score_visualization,
            "triage_documentation": result.score_documentation,
            "triage_imas": result.score_imas,
            "triage_convention": result.score_convention,
        }

        if composite >= threshold:
            triaged_items.append(item)
        else:
            skipped_items.append({**item, "reason": result.description})

    with GraphClient() as gc:
        if triaged_items:
            gc.query(
                """
                UNWIND $items AS item
                MATCH (sf:CodeFile {id: item.id})
                SET sf.status = 'triaged',
                    sf.score_cost = coalesce(sf.score_cost, 0) + item.score_cost,
                    sf.triage_composite = item.triage_composite,
                    sf.triage_description = item.triage_description,
                    sf.triage_modeling_code = item.triage_modeling_code,
                    sf.triage_analysis_code = item.triage_analysis_code,
                    sf.triage_operations_code = item.triage_operations_code,
                    sf.triage_data_access = item.triage_data_access,
                    sf.triage_workflow = item.triage_workflow,
                    sf.triage_visualization = item.triage_visualization,
                    sf.triage_documentation = item.triage_documentation,
                    sf.triage_imas = item.triage_imas,
                    sf.triage_convention = item.triage_convention,
                    sf.triaged_at = datetime(),
                    sf.claimed_at = null
                """,
                items=triaged_items,
            )

        if skipped_items:
            gc.query(
                """
                UNWIND $items AS item
                MATCH (sf:CodeFile {id: item.id})
                SET sf.status = 'skipped',
                    sf.score_cost = coalesce(sf.score_cost, 0) + item.score_cost,
                    sf.triage_composite = item.triage_composite,
                    sf.triage_description = item.triage_description,
                    sf.triage_modeling_code = item.triage_modeling_code,
                    sf.triage_analysis_code = item.triage_analysis_code,
                    sf.triage_operations_code = item.triage_operations_code,
                    sf.triage_data_access = item.triage_data_access,
                    sf.triage_workflow = item.triage_workflow,
                    sf.triage_visualization = item.triage_visualization,
                    sf.triage_documentation = item.triage_documentation,
                    sf.triage_imas = item.triage_imas,
                    sf.triage_convention = item.triage_convention,
                    sf.skip_reason = item.reason,
                    sf.triaged_at = datetime(),
                    sf.claimed_at = null
                """,
                items=skipped_items,
            )

    triaged_ids = [item["id"] for item in triaged_items]
    return {
        "triaged": len(triaged_items),
        "skipped": len(skipped_items),
        "triaged_ids": triaged_ids,
    }


def apply_file_scores(
    results: list[FileScoreResult],
    file_id_map: dict[str, str],
    batch_cost: float = 0.0,
) -> dict[str, int]:
    """Persist full scoring results to CodeFile nodes.

    Sets status to 'scored' and writes all dimension scores.

    Args:
        results: Score results from LLM.
        file_id_map: Mapping from path to CodeFile ID.
        batch_cost: Total LLM cost for the batch, distributed per-file.
    """
    scored_items = []
    skipped_items = []

    matched_count = sum(1 for r in results if file_id_map.get(r.path))
    cost_per_file = batch_cost / matched_count if matched_count > 0 else 0.0

    for result in results:
        sf_id = file_id_map.get(result.path)
        if not sf_id:
            continue

        if result.skip:
            skipped_items.append(
                {"id": sf_id, "score_cost": cost_per_file, "reason": result.description}
            )
        else:
            scored_items.append(
                {
                    "id": sf_id,
                    "score_cost": cost_per_file,
                    "score_composite": round(result.score_composite, 4),
                    "score_reason": result.description,
                    "file_category": result.file_category,
                    "score_modeling_code": result.score_modeling_code,
                    "score_analysis_code": result.score_analysis_code,
                    "score_operations_code": result.score_operations_code,
                    "score_data_access": result.score_data_access,
                    "score_workflow": result.score_workflow,
                    "score_visualization": result.score_visualization,
                    "score_documentation": result.score_documentation,
                    "score_imas": result.score_imas,
                    "score_convention": result.score_convention,
                }
            )

    with GraphClient() as gc:
        if scored_items:
            gc.query(
                """
                UNWIND $items AS item
                MATCH (sf:CodeFile {id: item.id})
                SET sf.status = 'scored',
                    sf.score_cost = coalesce(sf.score_cost, 0) + item.score_cost,
                    sf.score_composite = item.score_composite,
                    sf.score_reason = item.score_reason,
                    sf.file_category = item.file_category,
                    sf.score_modeling_code = item.score_modeling_code,
                    sf.score_analysis_code = item.score_analysis_code,
                    sf.score_operations_code = item.score_operations_code,
                    sf.score_data_access = item.score_data_access,
                    sf.score_workflow = item.score_workflow,
                    sf.score_visualization = item.score_visualization,
                    sf.score_documentation = item.score_documentation,
                    sf.score_imas = item.score_imas,
                    sf.score_convention = item.score_convention,
                    sf.scored_at = datetime(),
                    sf.claimed_at = null
                """,
                items=scored_items,
            )

        if skipped_items:
            gc.query(
                """
                UNWIND $items AS item
                MATCH (sf:CodeFile {id: item.id})
                SET sf.status = 'skipped',
                    sf.score_cost = coalesce(sf.score_cost, 0) + item.score_cost,
                    sf.skip_reason = item.reason,
                    sf.scored_at = datetime(),
                    sf.claimed_at = null
                """,
                items=skipped_items,
            )

    return {"scored": len(scored_items), "skipped": len(skipped_items)}
