"""Cross-batch consolidation for the standard-name pipeline.

Runs AFTER all compose workers complete but BEFORE any graph writes.
Performs deduplication, conflict detection, coverage accounting, and
concept registry lookups to produce a clean set of approved candidates.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# Result dataclasses
# =============================================================================


@dataclass
class ConflictRecord:
    """A detected conflict between candidates."""

    standard_name: str
    conflict_type: str  # "unit_mismatch", "kind_mismatch", "duplicate_source"
    details: str
    candidates: list[dict]


@dataclass
class ConsolidationResult:
    """Result of cross-batch consolidation."""

    approved: list[dict] = field(default_factory=list)
    conflicts: list[ConflictRecord] = field(default_factory=list)
    coverage_gaps: list[str] = field(default_factory=list)  # unmapped source paths
    reused: list[dict] = field(default_factory=list)  # from concept registry
    skipped_vocab_gaps: list[dict] = field(default_factory=list)  # vocab_gap entries
    stats: dict = field(default_factory=dict)


# =============================================================================
# Internal helpers
# =============================================================================


def _merge_duplicates(group: list[dict]) -> dict:
    """Merge duplicate candidates for the same standard name.

    When multiple candidates produce the SAME standard_name with the SAME
    unit and kind, they are merged:
    - Keep the entry with the longest documentation
    - Union the ids_paths from all duplicates
    - Union the tags
    - Keep highest confidence
    """
    # Sort by documentation length (longest first), then confidence
    group.sort(
        key=lambda c: (len(c.get("documentation", "")), c.get("confidence", 0)),
        reverse=True,
    )
    merged = dict(group[0])

    # Union ids_paths
    all_paths: set[str] = set()
    for c in group:
        all_paths.update(c.get("imas_paths") or [])
        # Also add source_id as an imas_path
        if c.get("source_id"):
            all_paths.add(c["source_id"])
    merged["imas_paths"] = sorted(all_paths)

    # Union tags
    all_tags: set[str] = set()
    for c in group:
        all_tags.update(c.get("tags") or [])
    merged["tags"] = sorted(all_tags)

    # Keep highest confidence
    merged["confidence"] = max(c.get("confidence", 0) for c in group)

    return merged


# =============================================================================
# Main function
# =============================================================================


def consolidate_candidates(
    candidates: list[dict],
    *,
    source_paths: set[str] | None = None,
    existing_registry: dict[str, dict] | None = None,
) -> ConsolidationResult:
    """Cross-batch dedup and conflict detection.

    Runs AFTER all compose workers complete, BEFORE any graph writes.

    Args:
        candidates: All composed candidate dicts from all batches.
            Each dict has at minimum: id (standard_name), source_id,
            source_type, unit (may be None for dimensionless).
            May also have: kind, tags, description, documentation, etc.
        source_paths: Set of all source paths that were sent for composition.
            Used for coverage accounting. If None, coverage check is skipped.
        existing_registry: Map of standard_name -> existing StandardName dict
            from the graph (for concept reuse). If None, registry check skipped.

    Returns:
        ConsolidationResult with approved, conflicts, coverage_gaps, reused.

    Checks performed:
        1. No duplicate standard_name with different units
        2. No duplicate standard_name with different kind
        3. No source path claimed by multiple candidates
        4. Coverage: every source_path in source_paths is mapped or noted
        5. Concept registry: reuse existing accepted names
    """
    result = ConsolidationResult()

    if not candidates:
        # Still run coverage accounting even with no candidates
        if source_paths is not None:
            result.coverage_gaps = sorted(source_paths)
        result.stats = {
            "total_input": 0,
            "approved": 0,
            "conflicts": 0,
            "merged": 0,
            "reused": 0,
            "coverage_gaps": len(result.coverage_gaps),
        }
        return result

    # ------------------------------------------------------------------
    # Group by standard_name (the "id" field)
    # ------------------------------------------------------------------
    by_name: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_name[c["id"]].append(c)

    # Track conflicted names so we skip them in merging
    conflicted_names: set[str] = set()

    # ------------------------------------------------------------------
    # Check 1: Unit consistency
    # ------------------------------------------------------------------
    for name, group in by_name.items():
        units = {c.get("unit") for c in group}
        units.discard(None)  # None means dimensionless, not a conflict with itself
        if len(units) > 1:
            result.conflicts.append(
                ConflictRecord(
                    standard_name=name,
                    conflict_type="unit_mismatch",
                    details=f"Units: {units}",
                    candidates=group,
                )
            )
            conflicted_names.add(name)

    # ------------------------------------------------------------------
    # Check 2: Kind consistency
    # ------------------------------------------------------------------
    for name, group in by_name.items():
        if name in conflicted_names:
            continue
        kinds = {c.get("kind") for c in group}
        kinds.discard(None)
        if len(kinds) > 1:
            result.conflicts.append(
                ConflictRecord(
                    standard_name=name,
                    conflict_type="kind_mismatch",
                    details=f"Kinds: {kinds}",
                    candidates=group,
                )
            )
            conflicted_names.add(name)

    # ------------------------------------------------------------------
    # Check 3: Source path uniqueness
    #   Each source_id should map to at most one standard_name.
    #   Duplicates within the SAME name are fine (merged later).
    # ------------------------------------------------------------------
    source_to_names: dict[str, set[str]] = defaultdict(set)
    for c in candidates:
        sid = c.get("source_id")
        if sid:
            source_to_names[sid].add(c["id"])

    for sid, names in source_to_names.items():
        if len(names) > 1:
            # Collect all candidates that claim this source_id
            claiming = [c for c in candidates if c.get("source_id") == sid]
            result.conflicts.append(
                ConflictRecord(
                    standard_name=sid,
                    conflict_type="duplicate_source",
                    details=f"Source {sid} claimed by: {sorted(names)}",
                    candidates=claiming,
                )
            )
            # Mark all involved names as conflicted
            conflicted_names.update(names)

    # ------------------------------------------------------------------
    # Merge non-conflicting duplicates and build approved list
    # ------------------------------------------------------------------
    merged_count = 0
    for name, group in by_name.items():
        if name in conflicted_names:
            continue

        # Check 5: Concept registry lookup
        if existing_registry and name in existing_registry:
            existing = existing_registry[name]
            if existing.get("review_status") == "accepted":
                result.reused.append(existing)
                logger.debug("Reusing existing accepted name: %s", name)
                continue

        if len(group) == 1:
            result.approved.append(dict(group[0]))
        else:
            merged = _merge_duplicates(group)
            result.approved.append(merged)
            merged_count += 1

    # ------------------------------------------------------------------
    # Check 4: Coverage accounting
    # ------------------------------------------------------------------
    if source_paths is not None:
        mapped_paths: set[str] = set()
        for c in candidates:
            sid = c.get("source_id")
            if sid:
                mapped_paths.add(sid)
        unmapped = sorted(source_paths - mapped_paths)
        result.coverage_gaps = unmapped

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    result.stats = {
        "total_input": len(candidates),
        "approved": len(result.approved),
        "conflicts": len(result.conflicts),
        "merged": merged_count,
        "reused": len(result.reused),
        "coverage_gaps": len(result.coverage_gaps),
    }

    logger.info(
        "Consolidation: %d input → %d approved, %d conflicts, "
        "%d merged, %d reused, %d coverage gaps",
        len(candidates),
        len(result.approved),
        len(result.conflicts),
        merged_count,
        len(result.reused),
        len(result.coverage_gaps),
    )

    return result
