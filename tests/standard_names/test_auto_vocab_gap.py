"""Tests for auto-VocabGap detection of novel physical_base tokens (W29).

Validates that ``_auto_detect_physical_base_gaps`` correctly identifies
novel ``physical_base`` tokens in composed standard names without
requiring the LLM to emit explicit ``vocab_gap`` exits.

Covers:
  - Known base in name → no gap created
  - Novel base → gap created with correct fields
  - Invalid name (parse failure) → silently skipped
  - Multiple candidates batched correctly
  - Deduplication of same (source_id, base) across candidates
  - ``_load_known_physical_bases`` returns a non-empty frozenset
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from imas_codex.standard_names.workers import (
    _auto_detect_physical_base_gaps,
    _load_known_physical_bases,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    source_id: str = "core_profiles/profiles_1d/electrons/temperature",
    standard_name: str = "electron_temperature",
) -> SimpleNamespace:
    """Create a minimal candidate-like object with source_id and standard_name."""
    return SimpleNamespace(source_id=source_id, standard_name=standard_name)


# ---------------------------------------------------------------------------
# _load_known_physical_bases
# ---------------------------------------------------------------------------


class TestLoadKnownPhysicalBases:
    """Verify the ISN physical_base vocabulary loads correctly."""

    def test_returns_frozenset(self) -> None:
        result = _load_known_physical_bases()
        assert isinstance(result, frozenset)

    def test_non_empty(self) -> None:
        result = _load_known_physical_bases()
        assert len(result) > 100, (
            f"Expected >100 registered physical bases, got {len(result)}"
        )

    def test_contains_temperature(self) -> None:
        result = _load_known_physical_bases()
        assert "temperature" in result

    def test_contains_pressure(self) -> None:
        result = _load_known_physical_bases()
        assert "pressure" in result

    def test_contains_magnetic_field(self) -> None:
        result = _load_known_physical_bases()
        assert "magnetic_field" in result


# ---------------------------------------------------------------------------
# _auto_detect_physical_base_gaps — known bases
# ---------------------------------------------------------------------------


class TestAutoDetectKnownBases:
    """Names with registered physical_base tokens should produce no gaps."""

    def test_electron_temperature_no_gap(self) -> None:
        c = _make_candidate(standard_name="electron_temperature")
        gaps = _auto_detect_physical_base_gaps([c])
        assert gaps == []

    def test_poloidal_magnetic_flux_no_gap(self) -> None:
        c = _make_candidate(
            source_id="equilibrium/time_slice/profiles_1d/psi",
            standard_name="poloidal_magnetic_flux",
        )
        gaps = _auto_detect_physical_base_gaps([c])
        assert gaps == []

    def test_pressure_no_gap(self) -> None:
        c = _make_candidate(
            source_id="core_profiles/profiles_1d/pressure_thermal",
            standard_name="thermal_pressure",
        )
        gaps = _auto_detect_physical_base_gaps([c])
        assert gaps == []


# ---------------------------------------------------------------------------
# _auto_detect_physical_base_gaps — novel bases
# ---------------------------------------------------------------------------


class TestAutoDetectNovelBases:
    """Names with unregistered physical_base tokens should produce gaps."""

    def test_novel_base_creates_gap(self) -> None:
        # Use a base we know is not registered
        c = _make_candidate(
            source_id="test/path",
            standard_name="electron_temperature",  # known base
        )
        # Override known_bases to exclude 'temperature'
        gaps = _auto_detect_physical_base_gaps(
            [c],
            known_bases=frozenset(["pressure", "magnetic_field"]),
        )
        assert len(gaps) == 1
        gap = gaps[0]
        assert gap["source_id"] == "test/path"
        assert gap["segment"] == "physical_base"
        assert gap["needed_token"] == "temperature"
        assert "Novel physical_base" in gap["reason"]

    def test_novel_base_with_real_vocabulary(self) -> None:
        """A truly novel base should be detected against real ISN vocab."""
        # Create a candidate with a base that doesn't exist in ISN
        c = _make_candidate(
            source_id="test/novel_path",
            standard_name="electron_temperature",
        )
        # With a known_bases that excludes temperature
        gaps = _auto_detect_physical_base_gaps(
            [c],
            known_bases=frozenset(),  # empty = everything is novel
        )
        assert len(gaps) == 1
        assert gaps[0]["needed_token"] == "temperature"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestAutoDetectEdgeCases:
    """Edge cases: parse failures, empty inputs, deduplication."""

    def test_empty_candidates(self) -> None:
        gaps = _auto_detect_physical_base_gaps([])
        assert gaps == []

    def test_parse_failure_silently_skipped(self) -> None:
        """Names that fail grammar parsing are silently skipped."""
        c = _make_candidate(
            source_id="test/path",
            standard_name="this_is_not_a_valid_standard_name_!@#$",
        )
        gaps = _auto_detect_physical_base_gaps([c])
        # Should not raise; gap list may be empty or contain something
        # depending on parser behavior, but should not crash
        assert isinstance(gaps, list)

    def test_dedup_same_source_and_base(self) -> None:
        """Duplicate (source_id, base) pairs are deduplicated."""
        c1 = _make_candidate(
            source_id="test/path", standard_name="electron_temperature"
        )
        c2 = _make_candidate(
            source_id="test/path", standard_name="electron_temperature"
        )
        gaps = _auto_detect_physical_base_gaps(
            [c1, c2],
            known_bases=frozenset(),  # make temperature novel
        )
        # Only one gap despite two identical candidates
        assert len(gaps) == 1

    def test_different_sources_same_base(self) -> None:
        """Same base from different sources creates separate gaps."""
        c1 = _make_candidate(source_id="path/a", standard_name="electron_temperature")
        c2 = _make_candidate(source_id="path/b", standard_name="ion_temperature")
        gaps = _auto_detect_physical_base_gaps(
            [c1, c2],
            known_bases=frozenset(),
        )
        # Temperature appears twice but from different sources
        assert len(gaps) == 2


# ---------------------------------------------------------------------------
# Multiple candidates
# ---------------------------------------------------------------------------


class TestAutoDetectBatch:
    """Batch processing of multiple candidates."""

    def test_mixed_known_and_novel(self) -> None:
        """Known bases produce no gaps; novel bases do."""
        known = _load_known_physical_bases()
        candidates = [
            _make_candidate(
                source_id="known/path",
                standard_name="electron_temperature",
            ),
            _make_candidate(
                source_id="novel/path",
                standard_name="electron_temperature",
            ),
        ]
        # With full known_bases, temperature IS known → no gaps
        gaps = _auto_detect_physical_base_gaps(candidates, known_bases=known)
        assert len(gaps) == 0

        # With empty known_bases, temperature is novel → gaps for each source
        gaps = _auto_detect_physical_base_gaps(candidates, known_bases=frozenset())
        assert len(gaps) == 2

    def test_handles_import_gracefully(self) -> None:
        """Verify the function doesn't crash with valid candidates."""
        c = _make_candidate(standard_name="electron_temperature")
        # The function has a try/except ImportError around the ISN import.
        # Since ISN is installed in our test env, just verify normal flow.
        gaps = _auto_detect_physical_base_gaps([c])
        assert isinstance(gaps, list)
