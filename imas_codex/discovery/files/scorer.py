"""LLM-based file scoring for discovered SourceFiles.

Batch scores SourceFile nodes using an LLM to assess relevance and
assign interest_score + file_category. Updates SourceFile node properties.

Supports natural language focus prompts to steer scoring towards specific
topics (e.g., "equilibrium reconstruction codes").

Also provides path-based heuristic pre-scoring using the same pattern
registry as the paths enrichment module, allowing high-value files to
be prioritized for LLM scoring.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)

# Progress callback: (current, total, message) -> None
ProgressCallback = Callable[[int, int, str], None]

# ============================================================================
# Path-based heuristic pre-scoring (no LLM, no SSH)
# ============================================================================

# Patterns matched against file paths to estimate relevance.
# Reuses categories from imas_codex.discovery.paths.enrichment.PATTERN_REGISTRY
# but applied to file paths rather than directory contents.
_PATH_PATTERNS: dict[str, tuple[re.Pattern, float]] = {
    # High value: IMAS, data access patterns in path names
    "imas": (
        re.compile(r"imas|ids_|access_layer|data_dictionary|dd_|imasdb", re.I),
        0.3,
    ),
    "mdsplus": (re.compile(r"mdsplus|mds_|tdi_|treeshr", re.I), 0.25),
    "equilibrium": (
        re.compile(r"equil|efit|liuqe|cliste|helena|chease|eqdsk", re.I),
        0.25,
    ),
    "cocos": (re.compile(r"cocos|sign_conv|coordinate", re.I), 0.2),
    "transport": (
        re.compile(r"transp|jetto|astra|cronos|ets_|core_profile", re.I),
        0.2,
    ),
    "diagnostic": (re.compile(r"thomson|ece_|interfer|mse_|cxrs|bolom", re.I), 0.15),
    "mhd": (re.compile(r"jorek|mars_|kinx|mishka|stability|tearing", re.I), 0.15),
    "heating": (
        re.compile(r"nubeam|rabbit|nemo|pencil|toric|ecrh|icrf|nbi_", re.I),
        0.15,
    ),
    "data_format": (re.compile(r"hdf5|netcdf|ufile|shotfile|ppf_", re.I), 0.1),
    # Low-noise indicators
    "test": (re.compile(r"/tests?/|_test\.py$|test_\w+\.py$|conftest", re.I), -0.1),
    "setup": (
        re.compile(r"setup\.py$|setup\.cfg|pyproject|__pycache__|\.egg", re.I),
        -0.15,
    ),
    "migration": (re.compile(r"migration|alembic|changelog", re.I), -0.1),
    "generated": (re.compile(r"generated|auto_gen|\.bak$|\.orig$|\.swp$", re.I), -0.2),
    "vendor": (re.compile(r"/vendor/|/third_party/|/external/|/deps/", re.I), -0.15),
}

# Files with these patterns in their path are likely available via GitHub/similar
_PUBLIC_REPO_PATTERNS = re.compile(
    r"(omas|omfit|freeqdsk|uda-|pyuda|pint|numpy|scipy|matplotlib|"
    r"xarray|pandas|sklearn|tensorflow|pytorch|ansible|docker|"
    r"jenkins|travis|circleci|github)",
    re.I,
)


def compute_path_heuristic_score(file_path: str) -> float:
    """Compute a heuristic relevance score based on file path alone.

    Uses pattern matching against the file path to estimate relevance
    without SSH or LLM calls. Score ranges 0.0-1.0.

    Args:
        file_path: Full remote file path

    Returns:
        Heuristic score between 0.0 and 1.0
    """
    base_score = 0.3  # Neutral starting point

    for _name, (pattern, weight) in _PATH_PATTERNS.items():
        if pattern.search(file_path):
            base_score += weight

    return max(0.0, min(1.0, base_score))


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
    """Get SourceFile nodes needing scoring (discovered, no interest_score).

    .. deprecated::
        Use :func:`~imas_codex.discovery.files.graph_ops.claim_files_for_scoring`
        instead for parallel-safe claiming.
    """
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

    Uses claim coordination via ``claimed_at`` on SourceFile to enable
    safe parallel execution.  Only unclaimed (or stale-claimed) files
    are claimed for scoring.

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
    from imas_codex.discovery.base.llm import call_llm_structured
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

    # Claim files atomically (parallel-safe)
    files = claim_files_for_scoring(facility, limit=limit)
    if not files:
        report(0, 0, "No files to score (or all claimed by another worker)")
        return stats

    total = len(files)
    report(0, total, f"Scoring {total} files")

    # Process in batches
    processed = 0
    for batch_start in range(0, total, batch_size):
        if stats["cost"] >= cost_limit:
            # Release unclaimed files from remaining batches
            remaining_ids = [f["id"] for f in files[batch_start:]]
            release_file_score_claims(remaining_ids)
            report(processed, total, f"Cost limit reached (${stats['cost']:.2f})")
            break

        batch = files[batch_start : batch_start + batch_size]
        file_id_map = {f["path"]: f["id"] for f in batch}
        batch_ids = [f["id"] for f in batch]

        report(
            processed,
            total,
            f"Scoring batch {stats['batches'] + 1} ({len(batch)} files)",
        )

        prompt = _build_scoring_prompt(batch, facility, focus=focus)

        try:
            parsed, cost, tokens = call_llm_structured(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_model=BatchScoreResult,
                temperature=0.1,
            )
            stats["cost"] += cost

            result = _apply_scores(parsed.scores, file_id_map)
            stats["total_scored"] += result["scored"]
            stats["total_skipped"] += result["skipped"]

            # Clear claims on successfully scored files
            release_file_score_claims(batch_ids)

        except Exception as e:
            logger.error("Scoring batch failed: %s", e)
            stats["errors"].append(str(e))
            # Release claims on error so another worker can retry
            release_file_score_claims(batch_ids)

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
