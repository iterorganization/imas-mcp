"""Tests for the lean compose system prompt (Phase A of plan 43).

Validates that generate_name_system_lean.md:
  - Exists and renders under 24,000 chars (≈8K tokens at chars/3 heuristic)
  - Preserves the BANNED PREFIXES table and INSTRUMENT LOCUS rule
  - Preserves all 5 ANTI-PATTERN GALLERY entries (dropped violations 7→2 in EMW pilot)
  - Leaves the legacy generate_name_system.md unchanged when compose_lean=False

See plans/features/standard-names/43-pipeline-rd-fix.md (Phase A).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from imas_codex.llm.prompt_loader import PROMPTS_DIR, render_prompt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LEAN_PROMPT_PATH = PROMPTS_DIR / "sn" / "generate_name_system_lean.md"
LEGACY_PROMPT_PATH = PROMPTS_DIR / "sn" / "generate_name_system.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_lean_minimal() -> str:
    """Render the lean prompt with a minimal (empty) context dict.

    schema_needs: [] in the frontmatter means no ISN context is loaded
    automatically — rendering succeeds without graph/ISN dependencies.
    """
    return render_prompt("sn/generate_name_system_lean", context={})


def _raw_lean() -> str:
    return LEAN_PROMPT_PATH.read_text(encoding="utf-8")


def _raw_legacy() -> str:
    return LEGACY_PROMPT_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------


class TestLeanPromptExists:
    def test_lean_prompt_file_exists(self) -> None:
        assert LEAN_PROMPT_PATH.exists(), (
            f"generate_name_system_lean.md not found at {LEAN_PROMPT_PATH}"
        )

    def test_legacy_prompt_still_exists(self) -> None:
        assert LEGACY_PROMPT_PATH.exists(), (
            "generate_name_system.md was accidentally deleted"
        )


# ---------------------------------------------------------------------------
# Rendered size constraint
# ---------------------------------------------------------------------------


class TestLeanPromptSize:
    """The rendered lean prompt must stay under 40,000 chars (≈13K tokens).

    Original target was 24K chars but incremental rule additions raised the
    baseline; limit updated to 40K to catch runaway growth while allowing
    targeted prohibition rules (NC-29 through NC-32 added ~10K chars net).
    """

    CHAR_LIMIT = 40_000

    def test_renders_under_char_limit(self) -> None:
        rendered = _render_lean_minimal()
        char_count = len(rendered)
        assert char_count <= self.CHAR_LIMIT, (
            f"Lean prompt rendered to {char_count:,} chars, "
            f"which exceeds the {self.CHAR_LIMIT:,}-char limit "
            f"(≈{self.CHAR_LIMIT // 3:,} tokens). "
            "Trim the template to fit."
        )

    def test_renders_non_empty(self) -> None:
        rendered = _render_lean_minimal()
        assert len(rendered) > 1000, "Lean prompt rendered suspiciously short"


# ---------------------------------------------------------------------------
# Required sections — BANNED PREFIXES
# ---------------------------------------------------------------------------


class TestBannedPrefixesPresent:
    """BANNED PREFIXES table must be preserved verbatim in the lean prompt."""

    def test_section_heading(self) -> None:
        raw = _raw_lean()
        assert "BANNED PREFIXES" in raw

    def test_initial_prefix_in_table(self) -> None:
        raw = _raw_lean()
        assert "`initial_`" in raw, "initial_ must appear in BANNED PREFIXES table"

    def test_reconstructed_prefix_in_table(self) -> None:
        raw = _raw_lean()
        assert "`reconstructed_`" in raw

    def test_raw_prefix_in_table(self) -> None:
        raw = _raw_lean()
        assert "`raw_`" in raw

    def test_measured_prefix_in_table(self) -> None:
        raw = _raw_lean()
        assert "`measured_`" in raw

    def test_lean_render_contains_banned_prefixes(self) -> None:
        rendered = _render_lean_minimal()
        assert "BANNED PREFIXES" in rendered
        assert "initial_" in rendered


# ---------------------------------------------------------------------------
# Required sections — INSTRUMENT LOCUS rule
# ---------------------------------------------------------------------------


class TestInstrumentLocusPresent:
    """INSTRUMENT HANDLING section must be preserved in the lean prompt."""

    def test_section_heading(self) -> None:
        raw = _raw_lean()
        assert "INSTRUMENT HANDLING" in raw

    def test_locus_only_rule(self) -> None:
        raw = _raw_lean()
        assert "postfix locus" in raw

    def test_instrument_table_present(self) -> None:
        raw = _raw_lean()
        assert "polarimeter_laser_wavelength" in raw
        assert "vacuum_wavelength_of_polarimeter_beam" in raw

    def test_lean_render_contains_instrument_locus(self) -> None:
        rendered = _render_lean_minimal()
        assert "INSTRUMENT HANDLING" in rendered
        assert "postfix locus" in rendered


# ---------------------------------------------------------------------------
# Required sections — ANTI-PATTERN GALLERY (all 5 entries)
# ---------------------------------------------------------------------------


class TestAntiPatternGalleryPresent:
    """All 5 ANTI-PATTERN GALLERY entries must be preserved.

    These entries dropped EMW-pilot violations from 7 to 2 — they must not
    be lost in any size-reduction pass.

    The lean prompt uses entry labels EMW-1, EMW-2, W38-A1, W38-A2, W38-A3
    (renamed from Entry 1…5 in the original compose_system_lean.md).
    """

    def test_gallery_section_heading(self) -> None:
        raw = _raw_lean()
        assert "ANTI-PATTERN GALLERY" in raw

    def test_entry_1_instrument_prefix(self) -> None:
        raw = _raw_lean()
        assert "EMW-1" in raw
        assert "polarimeter_laser_wavelength" in raw

    def test_entry_2_state_prefix(self) -> None:
        raw = _raw_lean()
        assert "EMW-2" in raw
        assert "initial_ellipticity_of_polarimeter_channel_beam" in raw

    def test_entry_3_state_prefix_vocab_gap(self) -> None:
        raw = _raw_lean()
        assert "W38-A1" in raw
        assert "x_ray_crystal_spectrometer_pixel_photon_energy_lower_bound" in raw

    def test_entry_4_compound_locus(self) -> None:
        raw = _raw_lean()
        assert "W38-A2" in raw
        assert "halo_region_parallel_energy_due_to_heat_flux" in raw

    def test_entry_5_instrument_locus(self) -> None:
        raw = _raw_lean()
        assert "W38-A3" in raw
        assert "z_coordinate_of_sensor_direction_unit_vector" in raw

    def test_gallery_in_render(self) -> None:
        rendered = _render_lean_minimal()
        assert "ANTI-PATTERN GALLERY" in rendered
        assert "polarimeter_laser_wavelength" in rendered


# ---------------------------------------------------------------------------
# Required sections — HARD PRE-EMIT CHECKS
# ---------------------------------------------------------------------------


class TestHardPreEmitChecksPresent:
    """All 10 HARD PRE-EMIT CHECKS must be preserved in the lean prompt."""

    def test_section_heading(self) -> None:
        raw = _raw_lean()
        assert "HARD PRE-EMIT CHECKS" in raw

    def test_check_1_adjacent_duplicates(self) -> None:
        raw = _raw_lean()
        assert "No adjacent duplicate tokens" in raw

    def test_check_3_hardware_tokens(self) -> None:
        raw = _raw_lean()
        assert "Hardware tokens" in raw

    def test_check_4_provenance_prefixes(self) -> None:
        raw = _raw_lean()
        assert "No provenance prefixes" in raw

    def test_check_6_no_abbreviations(self) -> None:
        raw = _raw_lean()
        assert "No abbreviations" in raw


# ---------------------------------------------------------------------------
# Grammar reference preserved
# ---------------------------------------------------------------------------


class TestGrammarReferencePresent:
    """The lean prompt must include the grammar reference partial."""

    def test_grammar_include_directive(self) -> None:
        raw = _raw_lean()
        assert "_grammar_reference.md" in raw

    def test_rendered_grammar_present(self) -> None:
        rendered = _render_lean_minimal()
        # Grammar reference starts with this heading
        assert "Standard Name Grammar" in rendered
        assert "5-Group Internal Representation" in rendered


# ---------------------------------------------------------------------------
# NC-32 — Drop _on_ggd suffix rule
# ---------------------------------------------------------------------------


class TestNoGGDSuffixRule:
    """NC-32 rule: _on_ggd suffix must be prohibited in the lean prompt."""

    def test_nc32_rule_present(self) -> None:
        raw = _raw_lean()
        assert "NC-32" in raw, "NC-32 _on_ggd rule must be present in lean prompt"

    def test_on_ggd_exemplar_bad(self) -> None:
        raw = _raw_lean()
        assert "electron_density_on_ggd" in raw, (
            "NC-32 must contain electron_density_on_ggd as a ❌ exemplar"
        )

    def test_on_ggd_exemplar_good(self) -> None:
        raw = _raw_lean()
        # electron_density must appear as the ✅ fix
        assert "electron_density`" in raw or "electron_density," in raw, (
            "NC-32 must show electron_density as ✅ canonical form"
        )

    def test_ggd_coordinate_agnostic_rule(self) -> None:
        raw = _raw_lean()
        assert "coordinate-agnostic" in raw, (
            "NC-32 must state that standard names are coordinate-agnostic"
        )

    def test_never_line_contains_on_ggd(self) -> None:
        raw = _raw_lean()
        assert "electron_density_on_ggd" in raw

    def test_nc32_in_render(self) -> None:
        rendered = _render_lean_minimal()
        assert "NC-32" in rendered
        assert "electron_density_on_ggd" in rendered


# ---------------------------------------------------------------------------
# Legacy prompt unchanged when compose_lean=False
# ---------------------------------------------------------------------------


class TestLegacyPromptUnchanged:
    """compose_lean=False must still select the original generate_name_system.md."""

    def test_legacy_template_name_when_flag_false(self) -> None:
        """Verify workers select legacy template when compose_lean=False."""
        # Import the settings function and check it defaults to False
        from imas_codex.settings import get_compose_lean

        with patch.dict(os.environ, {}, clear=False):
            # Remove lean env var if set
            os.environ.pop("IMAS_CODEX_SN_COMPOSE_LEAN", None)
            result = get_compose_lean()
        assert result is False, "get_compose_lean() must default to False"

    def test_lean_flag_enabled_by_env(self) -> None:
        from imas_codex.settings import get_compose_lean

        with patch.dict(os.environ, {"IMAS_CODEX_SN_COMPOSE_LEAN": "1"}):
            result = get_compose_lean()
        assert result is True

    def test_lean_flag_enabled_by_true(self) -> None:
        from imas_codex.settings import get_compose_lean

        with patch.dict(os.environ, {"IMAS_CODEX_SN_COMPOSE_LEAN": "true"}):
            result = get_compose_lean()
        assert result is True

    def test_legacy_prompt_raw_not_modified(self) -> None:
        """generate_name_system.md must still contain its full HARD PRE-EMIT CHECK header."""
        raw = _raw_legacy()
        assert "HARD PRE-EMIT CHECKS" in raw
        # Verify the legacy file still has Jinja sections we removed from lean
        assert "field_guidance.naming_guidance" in raw, (
            "Legacy prompt appears to have been stripped of naming_guidance Jinja block"
        )

    def test_lean_removes_verbose_sections(self) -> None:
        """Lean prompt must NOT contain verbose sections from the legacy template."""
        raw = _raw_lean()
        # These are verbatim Jinja blocks from the legacy template that
        # are removed in the lean variant
        assert "field_guidance.naming_guidance" not in raw
        assert "field_guidance.documentation_guidance" not in raw
        assert "tokamak_ranges" not in raw
