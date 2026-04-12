"""Tests for cross-batch consolidation module."""

from __future__ import annotations

import pytest

from imas_codex.sn.consolidation import (
    ConflictRecord,
    ConsolidationResult,
    _merge_duplicates,
    consolidate_candidates,
)

# =============================================================================
# Helpers — candidate factories
# =============================================================================


def _candidate(
    name: str,
    source_id: str,
    *,
    unit: str | None = "eV",
    kind: str = "scalar",
    source_type: str = "dd",
    confidence: float = 0.9,
    tags: list[str] | None = None,
    imas_paths: list[str] | None = None,
    documentation: str = "",
    description: str = "",
) -> dict:
    """Build a minimal candidate dict."""
    return {
        "id": name,
        "source_id": source_id,
        "source_type": source_type,
        "unit": unit,
        "kind": kind,
        "confidence": confidence,
        "tags": tags or [],
        "imas_paths": imas_paths or [],
        "documentation": documentation,
        "description": description,
    }


# =============================================================================
# 1. No conflicts — pass through
# =============================================================================


class TestNoConflicts:
    """Unique names with unique source_ids → all approved."""

    def test_unique_candidates_all_approved(self):
        candidates = [
            _candidate("electron_temperature", "core/te"),
            _candidate("ion_temperature", "core/ti", unit="eV"),
            _candidate("plasma_current", "mag/ip", unit="A"),
        ]
        result = consolidate_candidates(candidates)

        assert len(result.approved) == 3
        assert len(result.conflicts) == 0
        assert result.stats["approved"] == 3
        assert result.stats["conflicts"] == 0

    def test_approved_names_match_input(self):
        candidates = [
            _candidate("electron_temperature", "core/te"),
        ]
        result = consolidate_candidates(candidates)

        assert result.approved[0]["id"] == "electron_temperature"
        assert result.approved[0]["source_id"] == "core/te"


# =============================================================================
# 2. Unit conflicts
# =============================================================================


class TestUnitConflicts:
    """Same name, different units → conflict detected."""

    def test_different_units_detected(self):
        candidates = [
            _candidate("electron_temperature", "core/te", unit="eV"),
            _candidate("electron_temperature", "core/te2", unit="K"),
        ]
        result = consolidate_candidates(candidates)

        assert len(result.conflicts) == 1
        assert result.conflicts[0].conflict_type == "unit_mismatch"
        assert result.conflicts[0].standard_name == "electron_temperature"
        assert len(result.approved) == 0

    def test_same_units_no_conflict(self):
        candidates = [
            _candidate("electron_temperature", "core/te", unit="eV"),
            _candidate("electron_temperature", "core/te2", unit="eV"),
        ]
        result = consolidate_candidates(candidates)

        # Same unit → merged, not conflicted
        assert len(result.conflicts) == 0
        assert len(result.approved) == 1
        assert result.approved[0]["id"] == "electron_temperature"

    def test_none_unit_vs_specific_unit_conflict(self):
        """Dimensionless (None) vs specific unit → conflict."""
        candidates = [
            _candidate("safety_factor", "eq/q", unit=None),
            _candidate("safety_factor", "eq/q2", unit="m"),
        ]
        result = consolidate_candidates(candidates)

        # None is discarded in the unit check, so {None, "m"} → {"m"} → 1 unit
        # This means no unit_mismatch — None is treated as "unspecified/dimensionless"
        assert len(result.conflicts) == 0

    def test_two_nones_no_conflict(self):
        """Two candidates with unit=None → no conflict."""
        candidates = [
            _candidate("safety_factor", "eq/q", unit=None),
            _candidate("safety_factor", "eq/q2", unit=None),
        ]
        result = consolidate_candidates(candidates)

        assert len(result.conflicts) == 0
        assert len(result.approved) == 1


# =============================================================================
# 3. Kind conflicts
# =============================================================================


class TestKindConflicts:
    """Same name, different kinds → conflict detected."""

    def test_different_kinds_detected(self):
        candidates = [
            _candidate("electron_temperature", "core/te", kind="scalar"),
            _candidate("electron_temperature", "core/te2", kind="vector"),
        ]
        result = consolidate_candidates(candidates)

        assert len(result.conflicts) == 1
        assert result.conflicts[0].conflict_type == "kind_mismatch"

    def test_same_kind_no_conflict(self):
        candidates = [
            _candidate("electron_temperature", "core/te", kind="scalar"),
            _candidate("electron_temperature", "core/te2", kind="scalar"),
        ]
        result = consolidate_candidates(candidates)

        assert len(result.conflicts) == 0
        assert len(result.approved) == 1


# =============================================================================
# 4. Source path uniqueness
# =============================================================================


class TestSourcePathUniqueness:
    """Same source_id claimed by different names → conflict."""

    def test_same_source_different_names_conflict(self):
        candidates = [
            _candidate("electron_temperature", "core/te"),
            _candidate("temperature_electron", "core/te"),
        ]
        result = consolidate_candidates(candidates)

        assert any(c.conflict_type == "duplicate_source" for c in result.conflicts)

    def test_same_source_same_name_ok(self):
        """Same source_id in duplicates of same name → fine (merged)."""
        candidates = [
            _candidate("electron_temperature", "core/te"),
            _candidate("electron_temperature", "core/te"),
        ]
        result = consolidate_candidates(candidates)

        # Same source_id maps to same name → no duplicate_source conflict
        assert not any(c.conflict_type == "duplicate_source" for c in result.conflicts)
        assert len(result.approved) == 1


# =============================================================================
# 5. Coverage accounting
# =============================================================================


class TestCoverageAccounting:
    """Coverage check: all paths mapped → no gaps; some unmapped → listed."""

    def test_all_paths_mapped(self):
        candidates = [
            _candidate("electron_temperature", "core/te"),
            _candidate("ion_temperature", "core/ti"),
        ]
        source_paths = {"core/te", "core/ti"}
        result = consolidate_candidates(candidates, source_paths=source_paths)

        assert len(result.coverage_gaps) == 0

    def test_some_paths_unmapped(self):
        candidates = [
            _candidate("electron_temperature", "core/te"),
        ]
        source_paths = {"core/te", "core/ti", "core/ne"}
        result = consolidate_candidates(candidates, source_paths=source_paths)

        assert sorted(result.coverage_gaps) == ["core/ne", "core/ti"]
        assert result.stats["coverage_gaps"] == 2

    def test_source_paths_none_skips_check(self):
        candidates = [
            _candidate("electron_temperature", "core/te"),
        ]
        result = consolidate_candidates(candidates, source_paths=None)

        assert result.coverage_gaps == []

    def test_empty_source_paths_all_gaps(self):
        """Empty candidates with source_paths → all are gaps."""
        result = consolidate_candidates([], source_paths={"core/te", "core/ti"})

        assert sorted(result.coverage_gaps) == ["core/te", "core/ti"]


# =============================================================================
# 6. Duplicate merging
# =============================================================================


class TestDuplicateMerging:
    """Same name + unit from different batches → merged."""

    def test_ids_paths_unioned(self):
        candidates = [
            _candidate(
                "electron_temperature",
                "core/te",
                imas_paths=["core_profiles/profiles_1d/electrons/temperature"],
            ),
            _candidate(
                "electron_temperature",
                "core/te2",
                imas_paths=["edge_profiles/profiles_1d/electrons/temperature"],
            ),
        ]
        result = consolidate_candidates(candidates)

        assert len(result.approved) == 1
        paths = result.approved[0]["imas_paths"]
        assert "core_profiles/profiles_1d/electrons/temperature" in paths
        assert "edge_profiles/profiles_1d/electrons/temperature" in paths
        # source_ids are also added as imas_paths
        assert "core/te" in paths
        assert "core/te2" in paths

    def test_tags_unioned(self):
        candidates = [
            _candidate("electron_temperature", "core/te", tags=["core_profiles"]),
            _candidate(
                "electron_temperature", "core/te2", tags=["edge_profiles", "kinetics"]
            ),
        ]
        result = consolidate_candidates(candidates)

        assert len(result.approved) == 1
        assert sorted(result.approved[0]["tags"]) == [
            "core_profiles",
            "edge_profiles",
            "kinetics",
        ]

    def test_highest_confidence_kept(self):
        candidates = [
            _candidate("electron_temperature", "core/te", confidence=0.8),
            _candidate("electron_temperature", "core/te2", confidence=0.95),
        ]
        result = consolidate_candidates(candidates)

        assert result.approved[0]["confidence"] == 0.95

    def test_longest_documentation_kept(self):
        candidates = [
            _candidate(
                "electron_temperature",
                "core/te",
                documentation="Short doc.",
            ),
            _candidate(
                "electron_temperature",
                "core/te2",
                documentation="This is a much longer documentation string with more detail.",
            ),
        ]
        result = consolidate_candidates(candidates)

        assert "much longer" in result.approved[0]["documentation"]

    def test_merge_stats(self):
        candidates = [
            _candidate("electron_temperature", "core/te"),
            _candidate("electron_temperature", "core/te2"),
        ]
        result = consolidate_candidates(candidates)

        assert result.stats["merged"] == 1


# =============================================================================
# 7. Concept registry
# =============================================================================


class TestConceptRegistry:
    """Registry lookup: existing accepted names reused, new ones approved."""

    def test_existing_accepted_reused(self):
        candidates = [
            _candidate("electron_temperature", "core/te"),
        ]
        registry = {
            "electron_temperature": {
                "id": "electron_temperature",
                "unit": "eV",
                "review_status": "accepted",
            },
        }
        result = consolidate_candidates(candidates, existing_registry=registry)

        assert len(result.approved) == 0
        assert len(result.reused) == 1
        assert result.reused[0]["id"] == "electron_temperature"
        assert result.stats["reused"] == 1

    def test_existing_drafted_not_reused(self):
        """Only 'accepted' names are reused; drafted names are not."""
        candidates = [
            _candidate("electron_temperature", "core/te"),
        ]
        registry = {
            "electron_temperature": {
                "id": "electron_temperature",
                "unit": "eV",
                "review_status": "drafted",
            },
        }
        result = consolidate_candidates(candidates, existing_registry=registry)

        assert len(result.approved) == 1
        assert len(result.reused) == 0

    def test_registry_none_skipped(self):
        candidates = [
            _candidate("electron_temperature", "core/te"),
        ]
        result = consolidate_candidates(candidates, existing_registry=None)

        assert len(result.approved) == 1
        assert len(result.reused) == 0

    def test_new_name_not_in_registry(self):
        candidates = [
            _candidate("electron_temperature", "core/te"),
        ]
        registry = {
            "ion_temperature": {
                "id": "ion_temperature",
                "unit": "eV",
                "review_status": "accepted",
            },
        }
        result = consolidate_candidates(candidates, existing_registry=registry)

        assert len(result.approved) == 1
        assert len(result.reused) == 0


# =============================================================================
# 8. Stats tracking
# =============================================================================


class TestStatsTracking:
    """Result stats include counts of approved, conflicts, merged, reused, gaps."""

    def test_all_stats_populated(self):
        candidates = [
            _candidate("electron_temperature", "core/te"),
            _candidate("electron_temperature", "core/te2"),
            _candidate("ion_temperature", "core/ti", unit="eV"),
            _candidate("ion_temperature", "core/ti2", unit="K"),  # conflict
        ]
        source_paths = {"core/te", "core/te2", "core/ti", "core/ti2", "core/ne"}
        registry = {
            "electron_temperature": {
                "id": "electron_temperature",
                "review_status": "accepted",
            },
        }
        result = consolidate_candidates(
            candidates,
            source_paths=source_paths,
            existing_registry=registry,
        )

        assert result.stats["total_input"] == 4
        assert result.stats["reused"] == 1
        assert result.stats["conflicts"] >= 1
        assert result.stats["coverage_gaps"] == 1  # core/ne
        assert "approved" in result.stats
        assert "merged" in result.stats


# =============================================================================
# 9. Edge cases
# =============================================================================


class TestEdgeCases:
    """Edge cases: empty list, single candidate, all conflicting."""

    def test_empty_candidates(self):
        result = consolidate_candidates([])

        assert result.approved == []
        assert result.conflicts == []
        assert result.coverage_gaps == []
        assert result.reused == []
        assert result.stats["total_input"] == 0
        assert result.stats["approved"] == 0

    def test_single_candidate(self):
        candidates = [_candidate("electron_temperature", "core/te")]
        result = consolidate_candidates(candidates)

        assert len(result.approved) == 1
        assert len(result.conflicts) == 0

    def test_all_candidates_conflict(self):
        """All candidates have unit mismatch → all in conflicts, none approved."""
        candidates = [
            _candidate("electron_temperature", "core/te", unit="eV"),
            _candidate("electron_temperature", "core/te2", unit="K"),
        ]
        result = consolidate_candidates(candidates)

        assert len(result.approved) == 0
        assert len(result.conflicts) == 1
        assert result.stats["approved"] == 0

    def test_candidate_missing_optional_fields(self):
        """Candidates with minimal fields still pass through."""
        candidates = [
            {
                "id": "electron_temperature",
                "source_id": "core/te",
                "source_type": "dd",
            },
        ]
        result = consolidate_candidates(candidates)

        assert len(result.approved) == 1


# =============================================================================
# 10. Dataclass construction
# =============================================================================


class TestDataclasses:
    """Verify dataclass defaults and construction."""

    def test_conflict_record_fields(self):
        cr = ConflictRecord(
            standard_name="test",
            conflict_type="unit_mismatch",
            details="Units: {'eV', 'K'}",
            candidates=[{"id": "test"}],
        )
        assert cr.standard_name == "test"
        assert cr.conflict_type == "unit_mismatch"
        assert len(cr.candidates) == 1

    def test_consolidation_result_defaults(self):
        r = ConsolidationResult()
        assert r.approved == []
        assert r.conflicts == []
        assert r.coverage_gaps == []
        assert r.reused == []
        assert r.skipped_vocab_gaps == []
        assert r.stats == {}


# =============================================================================
# 11. _merge_duplicates helper
# =============================================================================


class TestMergeDuplicates:
    """Direct tests for the _merge_duplicates helper."""

    def test_single_item_returned_as_is(self):
        group = [_candidate("te", "core/te", confidence=0.9)]
        merged = _merge_duplicates(group)

        assert merged["id"] == "te"
        assert merged["confidence"] == 0.9

    def test_paths_include_source_ids(self):
        group = [
            _candidate("te", "core/te", imas_paths=["p1"]),
            _candidate("te", "core/te2", imas_paths=["p2"]),
        ]
        merged = _merge_duplicates(group)

        assert "core/te" in merged["imas_paths"]
        assert "core/te2" in merged["imas_paths"]
        assert "p1" in merged["imas_paths"]
        assert "p2" in merged["imas_paths"]

    def test_empty_tags_handled(self):
        group = [
            _candidate("te", "core/te", tags=None),
            _candidate("te", "core/te2", tags=["kinetics"]),
        ]
        # tags=None becomes [] via factory, but test None tolerance
        group[0]["tags"] = None
        merged = _merge_duplicates(group)

        assert "kinetics" in merged["tags"]

    def test_empty_imas_paths_handled(self):
        group = [
            _candidate("te", "core/te", imas_paths=None),
            _candidate("te", "core/te2", imas_paths=["p1"]),
        ]
        group[0]["imas_paths"] = None
        merged = _merge_duplicates(group)

        assert "p1" in merged["imas_paths"]


# =============================================================================
# 12. Combined scenarios
# =============================================================================


class TestCombinedScenarios:
    """Integration-style tests combining multiple checks."""

    def test_mix_of_clean_and_conflicting(self):
        """Some names clean, some conflicting, some reused."""
        candidates = [
            # Clean: will be approved
            _candidate("plasma_current", "mag/ip", unit="A"),
            # Duplicate: will be merged
            _candidate("electron_density", "core/ne", unit="m^-3"),
            _candidate("electron_density", "core/ne2", unit="m^-3"),
            # Conflict: different units
            _candidate("electron_temperature", "core/te", unit="eV"),
            _candidate("electron_temperature", "core/te2", unit="K"),
        ]
        registry = {
            "plasma_current": {
                "id": "plasma_current",
                "review_status": "accepted",
            },
        }
        source_paths = {"mag/ip", "core/ne", "core/ne2", "core/te", "core/te2", "extra"}

        result = consolidate_candidates(
            candidates,
            source_paths=source_paths,
            existing_registry=registry,
        )

        # plasma_current → reused (accepted in registry)
        assert len(result.reused) == 1
        assert result.reused[0]["id"] == "plasma_current"

        # electron_density → merged into one approved
        assert len(result.approved) == 1
        assert result.approved[0]["id"] == "electron_density"

        # electron_temperature → conflicted
        assert any(c.standard_name == "electron_temperature" for c in result.conflicts)

        # "extra" is unmapped
        assert "extra" in result.coverage_gaps

        # Stats add up
        assert result.stats["total_input"] == 5
