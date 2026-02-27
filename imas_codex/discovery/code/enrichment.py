"""File-level enrichment via rg pattern matching.

Runs pattern matching on individual source files (non-recursive) to
discover which specific files contain data access, IMAS, modeling,
and other patterns. Complements the directory-level enrichment from
the paths pipeline by attributing matches to specific files.

Uses the same PATTERN_REGISTRY from paths enrichment for consistency.
Patterns are flattened into a single dict of category → regex and
executed via the remote enrich_files.py script.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from imas_codex.discovery.base.facility import get_facility
from imas_codex.discovery.paths.enrichment import PATTERN_REGISTRY
from imas_codex.graph import GraphClient
from imas_codex.remote.executor import async_run_python_script

logger = logging.getLogger(__name__)


def _build_flat_patterns() -> dict[str, str]:
    """Flatten PATTERN_REGISTRY into category → regex for rg.

    Combines all sub-patterns within each category into a single regex,
    keeping individual sub-categories for granular counts.
    """
    flat: dict[str, str] = {}
    for _category, (patterns_dict, _score_dim) in PATTERN_REGISTRY.items():
        for name, regex in patterns_dict.items():
            flat[name] = regex
    return flat


async def enrich_files(
    facility: str,
    file_paths: list[str],
    timeout: int = 120,
) -> list[dict[str, Any]]:
    """Enrich files with rg pattern matching on the remote facility.

    Args:
        facility: Facility identifier
        file_paths: List of remote file paths to enrich
        timeout: SSH timeout in seconds

    Returns:
        List of enrichment result dicts with pattern_categories, line_count, etc.
    """
    if not file_paths:
        return []

    config = get_facility(facility)
    ssh_host = config.get("ssh_host", facility)

    patterns = _build_flat_patterns()

    input_data = {
        "files": file_paths,
        "pattern_categories": patterns,
    }

    try:
        result = await async_run_python_script(
            "enrich_files.py",
            input_data=input_data,
            ssh_host=ssh_host,
            timeout=timeout,
        )
        if isinstance(result, str):
            parsed = json.loads(result)
            if isinstance(parsed, list):
                return parsed
        if isinstance(result, list):
            return result
        logger.warning("Unexpected enrichment result type: %s", type(result))
        return []
    except Exception as e:
        logger.error("File enrichment failed: %s", e)
        return []


def persist_file_enrichment(
    results: list[dict[str, Any]],
    file_id_map: dict[str, str],
) -> int:
    """Write enrichment results to CodeFile nodes in the graph.

    Args:
        results: List of enrichment result dicts from enrich_files()
        file_id_map: Mapping from file path to CodeFile node ID

    Returns:
        Number of files enriched
    """
    items = []
    for r in results:
        sf_id = file_id_map.get(r["path"])
        if not sf_id:
            continue
        if r.get("error"):
            continue

        items.append(
            {
                "id": sf_id,
                "is_enriched": True,
                "pattern_categories": json.dumps(r.get("pattern_categories", {})),
                "total_pattern_matches": r.get("total_pattern_matches", 0),
                "line_count": r.get("line_count", 0),
            }
        )

    if not items:
        return 0

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $items AS item
            MATCH (sf:CodeFile {id: item.id})
            SET sf.is_enriched = item.is_enriched,
                sf.enriched_at = datetime(),
                sf.pattern_categories = item.pattern_categories,
                sf.total_pattern_matches = item.total_pattern_matches,
                sf.line_count = item.line_count
            """,
            items=items,
        )

    return len(items)
