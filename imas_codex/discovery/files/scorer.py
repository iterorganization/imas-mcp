"""LLM-based file scoring for discovered SourceFiles.

Batch scores SourceFile nodes using an LLM to assess relevance and
assign interest_score + file_category. Updates SourceFile node properties.

Supports natural language focus prompts to steer scoring towards specific
topics (e.g., "equilibrium reconstruction codes").
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


class FileScore(BaseModel):
    """LLM scoring result for a single file."""

    path: str
    interest_score: float = Field(ge=0.0, le=1.0, description="Relevance score 0-1")
    file_category: str = Field(
        description="code, document, notebook, config, data, other"
    )
    reason: str = Field(description="Brief explanation for the score")
    skip: bool = Field(default=False, description="Whether to skip this file entirely")


class BatchScoreResult(BaseModel):
    """LLM batch scoring result."""

    scores: list[FileScore]


def _get_scorable_files(
    facility: str,
    limit: int = 100,
) -> list[dict]:
    """Get SourceFile nodes needing scoring (discovered, no interest_score)."""
    with GraphClient() as client:
        result = client.query(
            """
            MATCH (sf:SourceFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'discovered'
              AND sf.interest_score IS NULL
            RETURN sf.id AS id, sf.path AS path, sf.language AS language,
                   sf.file_category AS file_category
            ORDER BY sf.discovered_at ASC
            LIMIT $limit
            """,
            facility=facility,
            limit=limit,
        )
        return list(result)


def _build_scoring_prompt(
    files: list[dict],
    facility: str,
    focus: str | None = None,
) -> str:
    """Build LLM prompt for batch file scoring.

    Args:
        files: List of file info dicts
        facility: Facility ID for context
        focus: Optional natural language focus

    Returns:
        Formatted prompt string
    """
    file_list = "\n".join(
        f"  - {f['path']} ({f.get('language', 'unknown')})" for f in files
    )

    focus_section = ""
    if focus:
        focus_section = f"""
FOCUS: The user is particularly interested in: "{focus}"
Boost scores for files likely related to this focus area.
"""

    return f"""Score these source files from facility '{facility}' for relevance to fusion plasma physics research and IMAS data workflows.

{focus_section}
FILES:
{file_list}

For each file, provide:
- interest_score (0.0-1.0): How relevant is this file?
  - 0.9+: Direct IMAS integration, IDS read/write, data mapping
  - 0.7+: MDSplus access, equilibrium codes, core physics
  - 0.5+: General analysis, visualization, utility codes
  - 0.3-0.5: Support scripts, build files
  - <0.3: Config files, documentation, unrelated
- file_category: code, document, notebook, config, data, or other
- reason: Brief explanation (1 sentence)
- skip: true if file should be excluded (binary, generated, backup)

Respond with a JSON object matching this schema:
{json.dumps(BatchScoreResult.model_json_schema(), indent=2)}
"""


def _apply_scores(
    scores: list[FileScore],
    file_id_map: dict[str, str],
) -> dict[str, int]:
    """Apply LLM scores to SourceFile nodes in graph.

    Args:
        scores: List of FileScore results
        file_id_map: Mapping from path to SourceFile ID

    Returns:
        Dict with scored, skipped counts
    """
    scored_items = []
    skipped_items = []

    for score in scores:
        sf_id = file_id_map.get(score.path)
        if not sf_id:
            continue

        if score.skip:
            skipped_items.append({"id": sf_id, "reason": score.reason})
        else:
            scored_items.append(
                {
                    "id": sf_id,
                    "interest_score": score.interest_score,
                    "file_category": score.file_category,
                    "score_reason": score.reason,
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
                    sf.score_reason = item.score_reason
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
    batch_size: int = 50,
    limit: int = 1000,
    cost_limit: float = 5.0,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Score discovered SourceFiles using LLM batch scoring.

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
    from imas_codex.agentic.llm import get_completion

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

    # Get files to score
    files = _get_scorable_files(facility, limit=limit)
    if not files:
        report(0, 0, "No files to score")
        return stats

    total = len(files)
    report(0, total, f"Scoring {total} files")

    # Process in batches
    processed = 0
    for batch_start in range(0, total, batch_size):
        if stats["cost"] >= cost_limit:
            report(processed, total, f"Cost limit reached (${stats['cost']:.2f})")
            break

        batch = files[batch_start : batch_start + batch_size]
        file_id_map = {f["path"]: f["id"] for f in batch}

        report(
            processed,
            total,
            f"Scoring batch {stats['batches'] + 1} ({len(batch)} files)",
        )

        prompt = _build_scoring_prompt(batch, facility, focus=focus)

        try:
            response = get_completion(
                prompt,
                response_format=BatchScoreResult,
                temperature=0.1,
            )

            if isinstance(response, BatchScoreResult):
                result = _apply_scores(response.scores, file_id_map)
                stats["total_scored"] += result["scored"]
                stats["total_skipped"] += result["skipped"]
            else:
                # Try parsing raw response
                try:
                    parsed = BatchScoreResult.model_validate_json(response)
                    result = _apply_scores(parsed.scores, file_id_map)
                    stats["total_scored"] += result["scored"]
                    stats["total_skipped"] += result["skipped"]
                except Exception as parse_err:
                    logger.warning("Failed to parse LLM response: %s", parse_err)
                    stats["errors"].append(str(parse_err))

        except Exception as e:
            logger.error("Scoring batch failed: %s", e)
            stats["errors"].append(str(e))

        stats["batches"] += 1
        processed += len(batch)

    report(
        total,
        total,
        f"Scoring complete: {stats['total_scored']} scored, {stats['total_skipped']} skipped",
    )
    return stats


__all__ = [
    "BatchScoreResult",
    "FileScore",
    "score_facility_files",
]
