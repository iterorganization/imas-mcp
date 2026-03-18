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

_TERMINAL_ENRICH_ERRORS = {"file_not_found", "not_a_file", "permission_denied"}


def _is_terminal_enrich_error(error: str | None) -> bool:
    """Return True for per-file errors that should not be retried."""
    return bool(error) and error in _TERMINAL_ENRICH_ERRORS


def _transport_error_result(path: str, error: Exception) -> dict[str, Any]:
    """Create a synthetic per-file result for transport-level failures."""
    detail = str(error).strip() or type(error).__name__
    return {
        "path": path,
        "pattern_categories": {},
        "total_pattern_matches": 0,
        "line_count": 0,
        "preview_text": "",
        "content_hash": "",
        "error": f"transport_error:{detail[:160]}",
    }


async def _run_enrich_batch(
    ssh_host: str,
    file_paths: list[str],
    patterns: dict[str, str],
    timeout: int,
) -> list[dict[str, Any]]:
    """Run one remote enrichment batch and parse the JSON response."""
    result = await async_run_python_script(
        "enrich_files.py",
        input_data={
            "files": file_paths,
            "pattern_categories": patterns,
        },
        ssh_host=ssh_host,
        timeout=timeout,
    )
    if isinstance(result, str):
        parsed = json.loads(result)
        if isinstance(parsed, list):
            return parsed
    if isinstance(result, list):
        return result
    raise TypeError(f"Unexpected enrichment result type: {type(result)!r}")


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
    """Enrich files with rg pattern matching and preview text on the remote.

    Returns enrichment dicts including preview_text (head of file).
    The preview_text is used by the score worker but NOT persisted to
    the graph — it provides content context for scoring only.

    Args:
        facility: Facility identifier
        file_paths: List of remote file paths to enrich
        timeout: SSH timeout in seconds

    Returns:
        List of enrichment result dicts with pattern_categories,
        line_count, preview_text, etc.
    """
    if not file_paths:
        return []

    config = get_facility(facility)
    ssh_host = config.get("ssh_host", facility)

    patterns = _build_flat_patterns()

    try:
        return await _run_enrich_batch(ssh_host, file_paths, patterns, timeout)
    except Exception as e:
        logger.error(
            "File enrichment batch failed for %d files at %s: %s",
            len(file_paths),
            facility,
            e,
        )
        if len(file_paths) == 1:
            return [_transport_error_result(file_paths[0], e)]

    # Batch failed. Degrade gracefully: retry in smaller chunks so one bad
    # transport call does not cause the same entire batch to loop forever.
    chunk_size = min(10, len(file_paths))
    recovered: list[dict[str, Any]] = []
    for start in range(0, len(file_paths), chunk_size):
        chunk = file_paths[start : start + chunk_size]
        try:
            recovered.extend(
                await _run_enrich_batch(ssh_host, chunk, patterns, timeout)
            )
            continue
        except Exception as chunk_error:
            logger.warning(
                "Enrichment chunk failed for %d files at %s, retrying singly: %s",
                len(chunk),
                facility,
                chunk_error,
            )

        for path in chunk:
            try:
                single = await _run_enrich_batch(ssh_host, [path], patterns, timeout)
                recovered.extend(single)
            except Exception as single_error:
                logger.error(
                    "Single-file enrichment failed for %s: %s", path, single_error
                )
                recovered.append(_transport_error_result(path, single_error))

    return recovered


def persist_file_enrichment(
    results: list[dict[str, Any]],
    file_id_map: dict[str, str],
) -> dict[str, int]:
    """Write enrichment results to CodeFile nodes in the graph.

    Persists pattern matches, line count, and preview text (head of file).
    Preview text is used by the score worker to provide content context
    in the scoring prompt.

    Args:
        results: List of enrichment result dicts from enrich_files()
        file_id_map: Mapping from file path to CodeFile node ID

    Returns:
        Dict with counts for successful and terminally failed enrichments
    """
    items = []
    failed_items = []
    for r in results:
        sf_id = file_id_map.get(r["path"])
        if not sf_id:
            continue
        error = r.get("error")
        if error:
            if _is_terminal_enrich_error(error):
                failed_items.append(
                    {
                        "id": sf_id,
                        "error": f"enrichment:{error}",
                    }
                )
            continue

        item = {
            "id": sf_id,
            "is_enriched": True,
            "pattern_categories": json.dumps(r.get("pattern_categories", {})),
            "total_pattern_matches": r.get("total_pattern_matches", 0),
            "line_count": r.get("line_count", 0),
        }
        content_hash = r.get("content_hash", "")
        if content_hash:
            item["content_hash"] = content_hash
        preview = r.get("preview_text", "")
        if preview:
            item["preview_text"] = preview[:2000]
        items.append(item)

    if not items and not failed_items:
        return {"enriched": 0, "failed": 0}

    with GraphClient() as gc:
        if items:
            gc.query(
                """
                UNWIND $items AS item
                MATCH (sf:CodeFile {id: item.id})
                SET sf.is_enriched = item.is_enriched,
                    sf.enriched_at = datetime(),
                    sf.pattern_categories = item.pattern_categories,
                    sf.total_pattern_matches = item.total_pattern_matches,
                    sf.line_count = item.line_count,
                    sf.preview_text = item.preview_text,
                    sf.content_hash = item.content_hash
                """,
                items=items,
            )
        if failed_items:
            gc.query(
                """
                UNWIND $items AS item
                MATCH (sf:CodeFile {id: item.id})
                SET sf.status = 'failed',
                    sf.error = item.error,
                    sf.enriched_at = datetime(),
                    sf.claimed_at = null,
                    sf.claim_token = null
                """,
                items=failed_items,
            )

    return {"enriched": len(items), "failed": len(failed_items)}
