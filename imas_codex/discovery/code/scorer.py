"""Dual-pass LLM file scoring for discovered CodeFiles.

Pass 1 (Triage): Fast keep/skip classification from file path, language,
and per-file enrichment evidence (rg pattern matches). Cheap and fast —
filters ~70-80% of files.

Pass 2 (Score): Full multi-dimensional scoring with enrichment evidence
for files that passed triage. Uses the same 9 score dimensions as
the paths pipeline.

Both passes use Jinja2 prompt templates with static content first to
maximize LLM prefix cache reuse across batches.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)

# Progress callback: (current, total, message) -> None
ProgressCallback = Callable[[int, int, str], None]

# Score dimension names (matching FacilityPath dimensions)
SCORE_DIMENSION_NAMES = [
    "score_modeling_code",
    "score_analysis_code",
    "score_operations_code",
    "score_data_access",
    "score_workflow",
    "score_visualization",
    "score_documentation",
    "score_imas",
    "score_convention",
]


# ---------------------------------------------------------------------------
# Pass 1: Triage models
# ---------------------------------------------------------------------------


class FileTriageResult(BaseModel):
    """Pass 1 triage result for a single file."""

    path: str = Field(description="The file path (echo from input)")
    keep: bool = Field(description="Whether to keep this file for detailed scoring")
    reason: str = Field(
        default="",
        description="One-line explanation (max 50 chars)",
    )


class FileTriageBatch(BaseModel):
    """Batch of triage results from LLM."""

    results: list[FileTriageResult]


# ---------------------------------------------------------------------------
# Pass 2: Score models
# ---------------------------------------------------------------------------


class FileScoreResult(BaseModel):
    """LLM scoring result for a single file with multi-dimensional scores."""

    path: str = Field(description="The file path (echo from input)")
    file_category: str = Field(
        description="code, document, notebook, config, data, or other"
    )
    description: str = Field(
        default="",
        description="Brief summary of what the file likely contains (1 sentence)",
    )
    skip: bool = Field(default=False, description="Whether to skip this file entirely")

    # Per-dimension scores (0.0-1.0 each)
    score_modeling_code: float = Field(
        default=0.0,
        description="Forward modeling/simulation code value (0.0-1.0)",
    )
    score_analysis_code: float = Field(
        default=0.0,
        description="Experimental analysis code value (0.0-1.0)",
    )
    score_operations_code: float = Field(
        default=0.0,
        description="Real-time operations code value (0.0-1.0)",
    )
    score_data_access: float = Field(
        default=0.0,
        description="Data access tools value (0.0-1.0)",
    )
    score_workflow: float = Field(
        default=0.0,
        description="Workflow/orchestration value (0.0-1.0)",
    )
    score_visualization: float = Field(
        default=0.0,
        description="Visualization tools value (0.0-1.0)",
    )
    score_documentation: float = Field(
        default=0.0,
        description="Documentation value (0.0-1.0)",
    )
    score_imas: float = Field(
        default=0.0,
        description="IMAS relevance (0.0-1.0)",
    )
    score_convention: float = Field(
        default=0.0,
        description="Convention handling value — COCOS, sign/coordinate conventions (0.0-1.0)",
    )

    @property
    def interest_score(self) -> float:
        """Composite score = max of all dimension scores."""
        return max(
            self.score_modeling_code,
            self.score_analysis_code,
            self.score_operations_code,
            self.score_data_access,
            self.score_workflow,
            self.score_visualization,
            self.score_documentation,
            self.score_imas,
            self.score_convention,
        )


class FileScoreBatch(BaseModel):
    """Batch of file scoring results from LLM."""

    results: list[FileScoreResult]


# ---------------------------------------------------------------------------
# Prompt builders  (static portions first → maximize prefix cache hits)
# ---------------------------------------------------------------------------


def _build_triage_system_prompt(focus: str | None = None) -> str:
    """Build system prompt for pass 1 triage.

    Static content first for cache reuse, focus appended at end.
    """
    from imas_codex.agentic.prompt_loader import render_prompt

    context: dict[str, Any] = {}
    if focus:
        context["focus"] = focus
    return render_prompt("discovery/file-triage", context)


def _build_system_prompt(focus: str | None = None) -> str:
    """Build system prompt for pass 2 scoring.

    Static content first for cache reuse, focus appended at end.
    """
    from imas_codex.agentic.prompt_loader import render_prompt

    context: dict[str, Any] = {}
    if focus:
        context["focus"] = focus

    return render_prompt("discovery/file-scorer", context)


def _build_triage_user_prompt(file_groups: list[dict[str, Any]]) -> str:
    """Build user prompt for pass 1 triage.

    Files grouped by parent directory with per-file enrichment evidence.
    """
    lines = ["Triage these files. Each group shows the parent directory context.\n"]

    for i, group in enumerate(file_groups, 1):
        parent_path = group.get("parent_path", "unknown")
        parent_score = group.get("parent_score") or 0
        parent_purpose = group.get("parent_purpose") or "unknown"

        lines.append(f"\n## Directory {i}: {parent_path}")
        lines.append(f"Score: {parent_score:.2f}, Purpose: {parent_purpose}")

        # Parent dimension scores for context (compact)
        dim_parts = []
        for dim in SCORE_DIMENSION_NAMES:
            parent_key = f"parent_{dim}"
            val = group.get(parent_key) or 0
            if val >= 0.3:
                dim_parts.append(f"{dim.replace('score_', '')}: {val:.2f}")
        if dim_parts:
            lines.append(f"Parent scores: {', '.join(dim_parts)}")

        lines.append("\nFiles:")
        for f in group.get("files", []):
            lang = f.get("language") or "unknown"
            line_count = f.get("line_count") or 0
            patterns = f.get("patterns") or {}
            total_matches = f.get("total_matches") or 0

            parts = [f"  - {f['path']} ({lang}, {line_count} lines)"]
            if total_matches > 0:
                pattern_str = ", ".join(f"{k}: {v}" for k, v in patterns.items())
                parts.append(f"    patterns: {pattern_str} (total: {total_matches})")
            lines.append("\n".join(parts))

    return "\n".join(lines)


def _build_user_prompt(file_groups: list[dict[str, Any]]) -> str:
    """Build user prompt for pass 2 scoring.

    Files grouped by parent directory with per-file enrichment evidence.
    Static parent context first, then per-file details.
    """
    lines = ["Score these files. Each group shows the parent directory context.\n"]

    for i, group in enumerate(file_groups, 1):
        parent_path = group.get("parent_path", "unknown")
        parent_score = group.get("parent_score") or 0
        parent_purpose = group.get("parent_purpose") or "unknown"
        parent_desc = group.get("parent_description") or ""

        lines.append(f"\n## Directory {i}: {parent_path}")
        lines.append(f"Score: {parent_score:.2f}, Purpose: {parent_purpose}")
        if parent_desc:
            lines.append(f"Description: {parent_desc}")

        # Parent pattern evidence from enrichment
        patterns = group.get("parent_patterns")
        if patterns:
            if isinstance(patterns, str):
                try:
                    patterns = json.loads(patterns)
                except json.JSONDecodeError:
                    patterns = {}
            if patterns:
                pattern_parts = [f"{k}: {v}" for k, v in patterns.items() if v]
                if pattern_parts:
                    lines.append(f"Dir pattern evidence: {', '.join(pattern_parts)}")

        read_m = group.get("parent_read_matches") or 0
        write_m = group.get("parent_write_matches") or 0
        multiformat = group.get("parent_multiformat", False)
        if read_m or write_m:
            lines.append(
                f"Read/write: {read_m} reads, {write_m} writes"
                f" (multiformat: {multiformat})"
            )

        # Parent dimension scores for context
        dim_parts = []
        for dim in SCORE_DIMENSION_NAMES:
            parent_key = f"parent_{dim}"
            val = group.get(parent_key) or 0
            if val >= 0.3:
                dim_parts.append(f"{dim.replace('score_', '')}: {val:.2f}")
        if dim_parts:
            lines.append(f"Parent scores: {', '.join(dim_parts)}")

        # Files to score — now with per-file enrichment
        lines.append("\nFiles:")
        for f in group.get("files", []):
            lang = f.get("language") or "unknown"
            line_count = f.get("line_count") or 0
            patterns = f.get("patterns") or {}
            total_matches = f.get("total_matches") or 0

            parts = [f"  - {f['path']} ({lang}, {line_count} lines)"]
            if total_matches > 0:
                pattern_str = ", ".join(f"{k}: {v}" for k, v in patterns.items())
                parts.append(f"    patterns: {pattern_str}")
            lines.append("\n".join(parts))

    return "\n".join(lines)


def _group_files_by_parent(files: list[dict]) -> list[dict[str, Any]]:
    """Group claimed files by their parent FacilityPath.

    Args:
        files: Flat list of file dicts from claim_files_for_scoring,
               each containing parent_* fields from the join query.

    Returns:
        List of group dicts with parent context and file lists.
    """
    groups: dict[str, dict[str, Any]] = {}

    for f in files:
        parent_id = f.get("parent_path_id", "unknown")
        if parent_id not in groups:
            groups[parent_id] = {
                "parent_path_id": parent_id,
                "parent_path": f.get("parent_path") or "unknown",
                "parent_score": f.get("parent_score") or 0,
                "parent_purpose": f.get("parent_purpose") or "unknown",
                "parent_description": f.get("parent_description") or "",
                "parent_patterns": f.get("parent_patterns"),
                "parent_read_matches": f.get("parent_read_matches") or 0,
                "parent_write_matches": f.get("parent_write_matches") or 0,
                "parent_multiformat": f.get("parent_multiformat") or False,
                "files": [],
            }
            # Copy parent dimension scores
            for dim in SCORE_DIMENSION_NAMES:
                groups[parent_id][f"parent_{dim}"] = f.get(f"parent_{dim}") or 0

        file_entry = {
            "id": f["id"],
            "path": f["path"],
            "language": f.get("language") or "unknown",
        }
        # Include per-file enrichment data if available
        if "line_count" in f:
            file_entry["line_count"] = f.get("line_count") or 0
        if "patterns" in f:
            file_entry["patterns"] = f.get("patterns") or {}
        if "total_matches" in f:
            file_entry["total_matches"] = f.get("total_matches") or 0

        groups[parent_id]["files"].append(file_entry)

    return list(groups.values())


def apply_triage_results(
    results: list[FileTriageResult],
    file_id_map: dict[str, str],
) -> dict[str, int]:
    """Apply triage results — mark skipped files, return kept file IDs.

    Args:
        results: List of FileTriageResult from LLM
        file_id_map: Mapping from file path to CodeFile node ID

    Returns:
        Dict with kept_count, skipped_count, kept_ids (list)
    """
    kept_ids = []
    skipped_items = []

    for result in results:
        sf_id = file_id_map.get(result.path)
        if not sf_id:
            continue

        if result.keep:
            kept_ids.append(sf_id)
        else:
            skipped_items.append({"id": sf_id, "reason": result.reason})

    if skipped_items:
        with GraphClient() as client:
            client.query(
                """
                UNWIND $items AS item
                MATCH (sf:CodeFile {id: item.id})
                SET sf.status = 'skipped',
                    sf.skip_reason = item.reason,
                    sf.claimed_at = null
                """,
                items=skipped_items,
            )

    return {
        "kept": len(kept_ids),
        "skipped": len(skipped_items),
        "kept_ids": kept_ids,
    }


def apply_file_scores(
    results: list[FileScoreResult],
    file_id_map: dict[str, str],
) -> dict[str, int]:
    """Apply multi-dimensional LLM scores to CodeFile nodes in graph.

    Args:
        results: List of FileScoreResult from LLM
        file_id_map: Mapping from file path to CodeFile node ID

    Returns:
        Dict with scored, skipped counts
    """
    scored_items = []
    skipped_items = []

    for result in results:
        sf_id = file_id_map.get(result.path)
        if not sf_id:
            continue

        if result.skip:
            skipped_items.append({"id": sf_id, "reason": result.description})
        else:
            scored_items.append(
                {
                    "id": sf_id,
                    "interest_score": result.interest_score,
                    "score_reason": result.description,
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

    with GraphClient() as client:
        if scored_items:
            client.query(
                """
                UNWIND $items AS item
                MATCH (sf:CodeFile {id: item.id})
                SET sf.interest_score = item.interest_score,
                    sf.score_reason = item.score_reason,
                    sf.score_modeling_code = item.score_modeling_code,
                    sf.score_analysis_code = item.score_analysis_code,
                    sf.score_operations_code = item.score_operations_code,
                    sf.score_data_access = item.score_data_access,
                    sf.score_workflow = item.score_workflow,
                    sf.score_visualization = item.score_visualization,
                    sf.score_documentation = item.score_documentation,
                    sf.score_imas = item.score_imas,
                    sf.score_convention = item.score_convention
                """,
                items=scored_items,
            )

        if skipped_items:
            client.query(
                """
                UNWIND $items AS item
                MATCH (sf:CodeFile {id: item.id})
                SET sf.status = 'skipped',
                    sf.skip_reason = item.reason
                """,
                items=skipped_items,
            )

    return {"scored": len(scored_items), "skipped": len(skipped_items)}


def score_facility_files(
    facility: str,
    focus: str | None = None,
    batch_size: int = 500,
    limit: int = 2000,
    cost_limit: float = 5.0,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Score discovered CodeFiles using dual-pass LLM scoring.

    Pass 1: Triage — fast keep/skip from path + enrichment evidence.
    Pass 2: Score — full multi-dimensional scoring for kept files.

    Args:
        facility: Facility ID
        focus: Natural language focus (e.g., "equilibrium reconstruction")
        batch_size: Files per LLM call
        limit: Maximum files to score
        cost_limit: Maximum LLM spend in USD
        progress_callback: Optional progress callback

    Returns:
        Dict with total_scored, total_skipped, cost, batches
    """
    from imas_codex.discovery.code.graph_ops import (
        claim_files_for_scoring,
        release_file_score_claims,
    )
    from imas_codex.settings import get_model

    model = get_model("language")

    stats: dict[str, Any] = {
        "total_scored": 0,
        "total_skipped": 0,
        "total_triaged": 0,
        "cost": 0.0,
        "batches": 0,
        "errors": [],
    }

    def report(current: int, total: int, msg: str) -> None:
        if progress_callback:
            progress_callback(current, total, msg)
        logger.info("[%d/%d] %s", current, total, msg)

    # Claim files atomically (parallel-safe), grouped by parent path
    files = claim_files_for_scoring(facility, limit=limit)
    if not files:
        report(0, 0, "No files to score (or all claimed by another worker)")
        return stats

    total = len(files)
    report(0, total, f"Scoring {total} files")

    # Build system prompts once (for prefix caching)
    triage_system_prompt = _build_triage_system_prompt(focus=focus)
    score_system_prompt = _build_system_prompt(focus=focus)

    # Group by parent path
    file_groups = _group_files_by_parent(files)

    # Build batches from groups, respecting batch_size
    current_batch: list[dict] = []
    current_batch_groups: list[dict[str, Any]] = []
    processed = 0

    for group in file_groups:
        group_files = group["files"]

        if current_batch and len(current_batch) + len(group_files) > batch_size:
            if stats["cost"] >= cost_limit:
                remaining_ids = [f["id"] for f in files[processed:]]
                release_file_score_claims(remaining_ids)
                report(processed, total, f"Cost limit reached (${stats['cost']:.2f})")
                break

            _score_batch_dual_pass(
                model,
                triage_system_prompt,
                score_system_prompt,
                current_batch_groups,
                current_batch,
                stats,
            )
            processed += len(current_batch)
            report(processed, total, f"${stats['cost']:.3f}")
            current_batch = []
            current_batch_groups = []

        current_batch_groups.append(group)
        current_batch.extend(group_files)

    # Process remaining
    if current_batch:
        _score_batch_dual_pass(
            model,
            triage_system_prompt,
            score_system_prompt,
            current_batch_groups,
            current_batch,
            stats,
        )

    release_file_score_claims([f["id"] for f in files])
    return stats


def _score_batch_dual_pass(
    model: str,
    triage_system_prompt: str,
    score_system_prompt: str,
    file_groups: list[dict[str, Any]],
    all_files: list[dict],
    stats: dict[str, Any],
) -> None:
    """Run dual-pass LLM scoring on a batch of files.

    Pass 1: Triage to filter noise.
    Pass 2: Full scoring on kept files.
    """
    from imas_codex.discovery.base.llm import call_llm_structured

    file_id_map = {f["path"]: f["id"] for f in all_files}

    # --- Pass 1: Triage ---
    triage_user_prompt = _build_triage_user_prompt(file_groups)
    try:
        triage_result, triage_cost, _ = call_llm_structured(
            model=model,
            messages=[
                {"role": "system", "content": triage_system_prompt},
                {"role": "user", "content": triage_user_prompt},
            ],
            response_model=FileTriageBatch,
            temperature=0.1,
        )
        stats["cost"] += triage_cost

        triage_applied = apply_triage_results(triage_result.results, file_id_map)
        stats["total_skipped"] += triage_applied["skipped"]
        stats["total_triaged"] += triage_applied["kept"] + triage_applied["skipped"]

        # Build kept file groups for pass 2
        kept_ids = set(triage_applied["kept_ids"])
        if not kept_ids:
            stats["batches"] += 1
            return

    except Exception as e:
        logger.error("Triage batch failed: %s", e)
        stats["errors"].append(str(e))
        stats["batches"] += 1
        # On triage failure, score all files (fail-open)
        kept_ids = set(file_id_map.values())

    # --- Pass 2: Score kept files ---
    kept_groups = []
    for group in file_groups:
        kept_files = [f for f in group["files"] if f["id"] in kept_ids]
        if kept_files:
            kept_group = {**group, "files": kept_files}
            kept_groups.append(kept_group)

    if not kept_groups:
        stats["batches"] += 1
        return

    score_user_prompt = _build_user_prompt(kept_groups)
    try:
        parsed, cost, _ = call_llm_structured(
            model=model,
            messages=[
                {"role": "system", "content": score_system_prompt},
                {"role": "user", "content": score_user_prompt},
            ],
            response_model=FileScoreBatch,
            temperature=0.1,
        )
        stats["cost"] += cost

        result = apply_file_scores(parsed.results, file_id_map)
        stats["total_scored"] += result.get("scored", 0)
        stats["total_skipped"] += result.get("skipped", 0)

    except Exception as e:
        logger.error("Score batch failed: %s", e)
        stats["errors"].append(str(e))

    stats["batches"] += 1


__all__ = [
    "FileScoreBatch",
    "FileScoreResult",
    "FileTriageBatch",
    "FileTriageResult",
    "SCORE_DIMENSION_NAMES",
    "apply_file_scores",
    "apply_triage_results",
    "score_facility_files",
]
