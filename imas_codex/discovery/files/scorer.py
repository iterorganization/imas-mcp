"""LLM-based multi-dimensional file scoring for discovered SourceFiles.

Batch scores SourceFile nodes using an LLM to assess relevance across
9 score dimensions (matching FacilityPath dimensions). Files are grouped
by parent FacilityPath so the LLM receives enrichment context (rg pattern
matches) for each directory.

Uses Jinja2 prompt templates from ``discovery/file-scorer.md`` with
schema-derived score dimensions and enrichment patterns.
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


def _build_system_prompt(focus: str | None = None) -> str:
    """Build system prompt using Jinja2 template with schema context.

    Uses render_prompt() for proper Jinja2 rendering with score dimensions
    and enrichment patterns injected from LinkML schema and PATTERN_REGISTRY.
    """
    from imas_codex.agentic.prompt_loader import render_prompt

    context: dict[str, Any] = {}
    if focus:
        context["focus"] = focus

    return render_prompt("discovery/file-scorer", context)


def _build_user_prompt(file_groups: list[dict[str, Any]]) -> str:
    """Build user prompt with files grouped by parent FacilityPath.

    Each group includes the parent directory's enrichment context
    (pattern matches, scores, description) so the LLM can use
    directory-level evidence to calibrate file-level scores.

    Args:
        file_groups: List of group dicts, each with:
            - parent_path, parent_score, parent_purpose, parent_description
            - parent_patterns (JSON dict of pattern categories)
            - parent_read_matches, parent_write_matches, parent_multiformat
            - parent score dimensions
            - files: list of file dicts with path, language
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

        # Pattern evidence from enrichment
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
                    lines.append(f"Pattern evidence: {', '.join(pattern_parts)}")

        read_m = group.get("parent_read_matches", 0)
        write_m = group.get("parent_write_matches", 0)
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
            val = group.get(parent_key, 0)
            if val and val >= 0.3:
                dim_parts.append(f"{dim.replace('score_', '')}: {val:.2f}")
        if dim_parts:
            lines.append(f"Parent scores: {', '.join(dim_parts)}")

        # Files to score
        lines.append("\nFiles:")
        for f in group.get("files", []):
            lang = f.get("language", "unknown")
            lines.append(f"  - {f['path']} ({lang})")

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
                "parent_path": f.get("parent_path", "unknown"),
                "parent_score": f.get("parent_score", 0),
                "parent_purpose": f.get("parent_purpose", "unknown"),
                "parent_description": f.get("parent_description", ""),
                "parent_patterns": f.get("parent_patterns"),
                "parent_read_matches": f.get("parent_read_matches", 0),
                "parent_write_matches": f.get("parent_write_matches", 0),
                "parent_multiformat": f.get("parent_multiformat", False),
                "files": [],
            }
            # Copy parent dimension scores
            for dim in SCORE_DIMENSION_NAMES:
                groups[parent_id][f"parent_{dim}"] = f.get(f"parent_{dim}", 0)

        groups[parent_id]["files"].append(
            {
                "id": f["id"],
                "path": f["path"],
                "language": f.get("language", "unknown"),
            }
        )

    return list(groups.values())


def apply_file_scores(
    results: list[FileScoreResult],
    file_id_map: dict[str, str],
) -> dict[str, int]:
    """Apply multi-dimensional LLM scores to SourceFile nodes in graph.

    Args:
        results: List of FileScoreResult from LLM
        file_id_map: Mapping from file path to SourceFile node ID

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
                    "file_category": result.file_category,
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
                MATCH (sf:SourceFile {id: item.id})
                SET sf.interest_score = item.interest_score,
                    sf.file_category = item.file_category,
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
                MATCH (sf:SourceFile {id: item.id})
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
    """Score discovered SourceFiles using multi-dimensional LLM scoring.

    Files are grouped by parent FacilityPath so enrichment context
    (pattern matches, directory scores) is passed once per group.

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
    from imas_codex.discovery.files.graph_ops import (
        claim_files_for_scoring,
        release_file_score_claims,
    )
    from imas_codex.settings import get_model

    model = get_model("language")

    stats: dict[str, Any] = {
        "total_scored": 0,
        "total_skipped": 0,
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

    # Build system prompt once (for prefix caching)
    system_prompt = _build_system_prompt(focus=focus)

    # Group by parent path and process in batches
    file_groups = _group_files_by_parent(files)

    # Build batches from groups, respecting batch_size
    current_batch: list[dict] = []
    current_batch_groups: list[dict[str, Any]] = []
    processed = 0

    for group in file_groups:
        group_files = group["files"]

        # If adding this group exceeds batch_size, flush current batch
        if current_batch and len(current_batch) + len(group_files) > batch_size:
            if stats["cost"] >= cost_limit:
                remaining_ids = [f["id"] for f in files[processed:]]
                release_file_score_claims(remaining_ids)
                report(processed, total, f"Cost limit reached (${stats['cost']:.2f})")
                break

            _score_batch(
                model,
                system_prompt,
                current_batch_groups,
                current_batch,
                stats,
                report,
                processed,
                total,
            )
            processed += len(current_batch)
            current_batch = []
            current_batch_groups = []

        current_batch.extend(group_files)
        current_batch_groups.append(group)

    # Flush remaining
    if current_batch and stats["cost"] < cost_limit:
        _score_batch(
            model,
            system_prompt,
            current_batch_groups,
            current_batch,
            stats,
            report,
            processed,
            total,
        )
        processed += len(current_batch)

    report(
        total,
        total,
        f"Scoring complete: {stats['total_scored']} scored, "
        f"{stats['total_skipped']} skipped",
    )
    return stats


def _score_batch(
    model: str,
    system_prompt: str,
    groups: list[dict[str, Any]],
    batch_files: list[dict],
    stats: dict[str, Any],
    report: Callable,
    processed: int,
    total: int,
) -> None:
    """Score a single batch of files grouped by parent path."""
    from imas_codex.discovery.base.llm import call_llm_structured
    from imas_codex.discovery.files.graph_ops import release_file_score_claims

    file_id_map = {f["path"]: f["id"] for f in batch_files}
    batch_ids = [f["id"] for f in batch_files]

    report(
        processed,
        total,
        f"Scoring batch {stats['batches'] + 1} ({len(batch_files)} files, "
        f"{len(groups)} dirs)",
    )

    user_prompt = _build_user_prompt(groups)

    try:
        parsed, cost, _tokens = call_llm_structured(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=FileScoreBatch,
            temperature=0.1,
        )
        stats["cost"] += cost

        result = apply_file_scores(parsed.results, file_id_map)
        stats["total_scored"] += result["scored"]
        stats["total_skipped"] += result["skipped"]

        release_file_score_claims(batch_ids)

    except Exception as e:
        logger.error("Scoring batch failed: %s", e)
        stats["errors"].append(str(e))
        release_file_score_claims(batch_ids)

    stats["batches"] += 1


__all__ = [
    "FileScoreBatch",
    "FileScoreResult",
    "SCORE_DIMENSION_NAMES",
    "apply_file_scores",
    "score_facility_files",
]
