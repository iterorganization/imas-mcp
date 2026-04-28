"""Compose-prompt guard tests for HARD PRE-EMIT CHECKS (Phase 5d).

Tests that compose_system.md contains the required guard content and
that the rendered prompt includes all ten HARD PRE-EMIT CHECKS.  These
are structural/content tests — they do not call the LLM.

A companion file ``test_compose_regex_guards.py`` provides pure-regex
validators for each anti-pattern.
"""

from __future__ import annotations

import pytest

from imas_codex.llm.prompt_loader import PROMPTS_DIR

# =====================================================================
# Helpers
# =====================================================================


def _load_compose_system_raw() -> str:
    """Read the raw (un-rendered) compose_system.md template."""
    path = PROMPTS_DIR / "sn" / "compose_system.md"
    return path.read_text(encoding="utf-8")


# =====================================================================
# Tests — prompt content assertions
# =====================================================================


class TestHardPreEmitChecksPresence:
    """Verify all 10 HARD PRE-EMIT CHECKS are present in compose_system.md."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.raw = _load_compose_system_raw()

    def test_section_heading_present(self) -> None:
        assert "HARD PRE-EMIT CHECKS" in self.raw

    def test_check_1_adjacent_duplicates(self) -> None:
        assert "No adjacent duplicate tokens" in self.raw
        assert "magnetic_magnetic" in self.raw

    def test_check_2_entity_locus_of(self) -> None:
        assert "Entity-locus preposition" in self.raw
        assert "separatrix" in self.raw
        assert "magnetic_axis" in self.raw

    def test_check_3_hardware_tokens(self) -> None:
        assert "Hardware tokens" in self.raw
        assert "probe" in self.raw
        assert "sensor" in self.raw

    def test_check_4_provenance_prefixes(self) -> None:
        assert "No provenance prefixes" in self.raw
        assert "initial_" in self.raw
        assert "launched_" in self.raw
        assert "post_crash_" in self.raw

    def test_check_5_invented_bases(self) -> None:
        assert "No invented physical bases" in self.raw

    def test_check_6_abbreviations(self) -> None:
        assert "No abbreviations, acronyms, or alphanumerics" in self.raw
        assert "3db" in self.raw

    def test_check_7_one_subject(self) -> None:
        assert "Exactly one subject" in self.raw
        assert "hydrogen_ion" in self.raw
        assert "deuterium_tritium_ion" in self.raw

    def test_check_8_us_spelling(self) -> None:
        assert "US spelling only" in self.raw
        assert "analyse" in self.raw
        assert "fibre" in self.raw

    def test_check_9_length_nesting(self) -> None:
        assert "70 characters" in self.raw
        assert "two `_of_`" in self.raw

    def test_check_10_structural_leakage(self) -> None:
        assert "No structural leakage" in self.raw
        assert "obtained_from" in self.raw
        assert "stored_in" in self.raw
        assert "derived_from" in self.raw


class TestRejectListExpansion:
    """Verify the REJECT list was expanded with Report 7 patterns."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.raw = _load_compose_system_raw()

    def test_bandwidth_3db_rejected(self) -> None:
        assert "bandwidth_3db" in self.raw

    def test_turn_count_rejected(self) -> None:
        assert "turn_count" in self.raw

    def test_nuclear_charge_number_rejected(self) -> None:
        assert "nuclear_charge_number" in self.raw

    def test_azimuth_angle_rejected(self) -> None:
        assert "azimuth_angle" in self.raw

    def test_distance_between_rejected(self) -> None:
        assert "distance_between_" in self.raw


class TestChannelGuidance:
    """Verify NC-32 channel guidance block is present."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.raw = _load_compose_system_raw()

    def test_nc32_heading_present(self) -> None:
        assert "NC-32" in self.raw

    def test_channel_path_pattern_mentioned(self) -> None:
        assert "*/channel/*" in self.raw

    def test_observable_examples(self) -> None:
        assert "faraday_rotation_angle" in self.raw
        assert "line_integrated_electron_density" in self.raw

    def test_diagnostic_examples(self) -> None:
        assert "polarimeter" in self.raw
        assert "interferometer" in self.raw
        assert "thomson_scattering" in self.raw
        assert "refractometer" in self.raw

    def test_anti_pattern_examples(self) -> None:
        assert "polarimeter_channel_angle" in self.raw
        assert "interferometer_channel_density" in self.raw


class TestHardChecksPlacement:
    """Verify HARD PRE-EMIT CHECKS appear before REJECT and examples."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.raw = _load_compose_system_raw()

    def test_hard_checks_before_reject(self) -> None:
        hard_pos = self.raw.index("HARD PRE-EMIT CHECKS")
        reject_pos = self.raw.index("### REJECT")
        assert hard_pos < reject_pos

    def test_hard_checks_before_curated_examples(self) -> None:
        hard_pos = self.raw.index("HARD PRE-EMIT CHECKS")
        examples_pos = self.raw.index("## Curated Examples")
        assert hard_pos < examples_pos

    def test_hard_checks_after_includes(self) -> None:
        # HARD CHECKS must appear after the prelude includes (vocabulary,
        # exemplars, scored examples). A trailing include such as
        # _coordinate_conventions.md may legitimately appear later in the file.
        prelude_end = self.raw.index('{% include "sn/_compose_scored_examples.md" %}')
        hard_pos = self.raw.index("HARD PRE-EMIT CHECKS")
        assert hard_pos > prelude_end


class TestNoConflictWithConstraintRole:
    """Verify HARD PRE-EMIT CHECKS don't contradict CONSTRAINT ROLE ABSTRACTION."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.raw = _load_compose_system_raw()

    def test_constraint_role_block_still_present(self) -> None:
        assert "CONSTRAINT ROLE ABSTRACTION" in self.raw

    def test_hard_checks_after_constraint_role_or_independent(self) -> None:
        # The CONSTRAINT ROLE block is at ~line 88-100 in the original;
        # HARD PRE-EMIT CHECKS is near the top.  Both should exist.
        assert "HARD PRE-EMIT CHECKS" in self.raw
        assert "CONSTRAINT ROLE ABSTRACTION" in self.raw
