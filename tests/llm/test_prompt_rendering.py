"""Tests for SN compose prompt rendering — vNext grammar reference + exemplar alignment."""

from __future__ import annotations

import pytest


@pytest.fixture()
def rendered_compose_system() -> str:
    """Render sn/compose_system.md with full grammar context."""
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.context import (
        build_compose_context,
        clear_context_cache,
    )

    clear_context_cache()
    context = build_compose_context()
    return render_prompt("sn/compose_system", context)


class TestTransformationInjection:
    """W4b — Verify vNext grammar reference replaces rc20 transformation injection."""

    def test_vnext_grammar_heading_present(self, rendered_compose_system: str) -> None:
        """The vNext grammar reference heading must be present."""
        assert "Standard Name Grammar (vNext" in rendered_compose_system

    def test_old_static_list_not_present(self, rendered_compose_system: str) -> None:
        """The old static heading 'Transformation Boundaries' must be gone."""
        assert "Transformation Boundaries" not in rendered_compose_system
        assert "Only these 4 transformation tokens" not in rendered_compose_system

    def test_old_live_heading_gone(self, rendered_compose_system: str) -> None:
        """The old rc20 heading must be gone — replaced by vNext grammar."""
        assert (
            "Transformations (live from imas-standard-names)"
            not in rendered_compose_system
        )

    def test_five_group_ir_present(self, rendered_compose_system: str) -> None:
        """The 5-Group IR table must appear in the rendered prompt."""
        assert "5-Group Internal Representation" in rendered_compose_system
        for group in ["operators", "projection", "qualifiers", "base", "locus"]:
            assert group in rendered_compose_system

    def test_operator_scope_examples(self, rendered_compose_system: str) -> None:
        """Key operator examples must appear (prefix _of_, postfix concat)."""
        assert "gradient_of_" in rendered_compose_system
        assert "_magnitude" in rendered_compose_system


class TestExemplarAlignment:
    """W4b — Verify vNext exemplars and anti-patterns landed."""

    def test_no_vertical_position_of_x_point(
        self, rendered_compose_system: str
    ) -> None:
        """Zero hits for the deprecated vertical_position_of_x_point form."""
        assert "vertical_position_of_x_point" not in rendered_compose_system

    def test_vertical_coordinate_of_x_point_present(
        self, rendered_compose_system: str
    ) -> None:
        """The canonical vertical_coordinate_of_x_point must appear."""
        assert "vertical_coordinate_of_x_point" in rendered_compose_system

    def test_at_magnetic_axis_in_good_context(
        self, rendered_compose_system: str
    ) -> None:
        """at_magnetic_axis should appear in GOOD/field-at-locus context."""
        # Must appear (in P3 or similar)
        assert "at_magnetic_axis" in rendered_compose_system
        # Must NOT appear in a FORBIDDEN-only context without contrast
        # (The BAD example in P3 is "major_radius_at_magnetic_axis" which is
        # presented as BAD with a fix — so the word "BAD" appears nearby)
        idx = rendered_compose_system.find("major_radius_at_magnetic_axis")
        if idx != -1:
            # It should be in a BAD/FORBIDDEN context
            nearby = rendered_compose_system[max(0, idx - 100) : idx + 100]
            assert "BAD" in nearby or "FORBIDDEN" in nearby or "→" in nearby

    def test_p3_of_vs_at_semantic_split(self, rendered_compose_system: str) -> None:
        """P3 must teach the of_ vs at_ semantic distinction."""
        assert (
            "of_<position>" in rendered_compose_system
            or "of_\\<position\\>" in rendered_compose_system
        )
        assert (
            "at_<position>" in rendered_compose_system
            or "at_\\<position\\>" in rendered_compose_system
        )

    def test_p8_canonical_coordinates(self, rendered_compose_system: str) -> None:
        """P8 must list the three canonical coordinate forms."""
        assert "vertical_coordinate_of_<position>" in rendered_compose_system
        assert "major_radius_of_<position>" in rendered_compose_system
        assert "toroidal_angle_of_<position>" in rendered_compose_system

    def test_forbidden_patterns_section(self, rendered_compose_system: str) -> None:
        """The anti-exemplar section must be present with key patterns."""
        assert "Forbidden patterns (anti-exemplars)" in rendered_compose_system
        assert "due_to_<adjective>" in rendered_compose_system
        assert "_ggd_coefficients" in rendered_compose_system
        assert "_reference_waveform" in rendered_compose_system
        assert "diamagnetic_component_of_<vector>" in rendered_compose_system
        assert "deuterium_tritium" in rendered_compose_system


class TestRuleAlignment:
    """B.4 — NC-27/NC-28 and Rule 22 demotion."""

    def test_nc27_compound_subject(self, rendered_compose_system: str) -> None:
        """NC-27 on compound-subject tokens must be present."""
        assert "NC-27" in rendered_compose_system
        assert "deuterium_deuterium" in rendered_compose_system
        assert "tritium_tritium" in rendered_compose_system

    def test_nc28_reference_waveform(self, rendered_compose_system: str) -> None:
        """NC-28 on _reference_waveform must be present."""
        assert "NC-28" in rendered_compose_system
        assert "controller setpoint" in rendered_compose_system

    def test_rule22_demoted_to_informational(
        self, rendered_compose_system: str
    ) -> None:
        """Rule 22 should be informational, not 'critical'."""
        # Find the Rule 22 section
        idx = rendered_compose_system.find("22.")
        assert idx != -1, "Rule 22 not found"
        rule_text = rendered_compose_system[idx : idx + 300]
        assert "informational" in rule_text
        # The old "(physics semantic — critical)" should be gone
        assert "physics semantic — critical" not in rule_text

    def test_over_region_exception(self, rendered_compose_system: str) -> None:
        """The _over_ rule must cross-reference the Region segment."""
        assert (
            "over_halo_region" in rendered_compose_system
            or "over_<region>" in rendered_compose_system
        )


class TestAntiPatternHardening:
    """w1-prompt-antipatterns — Banned prefixes, instrument-as-locus, gallery."""

    def test_banned_prefixes_section_present(
        self, rendered_compose_system: str
    ) -> None:
        """BANNED PREFIXES section must appear in the rendered system prompt."""
        assert "BANNED PREFIXES" in rendered_compose_system

    def test_state_prefixes_enumerated(self, rendered_compose_system: str) -> None:
        """Key banned state/provenance prefixes must be explicitly listed."""
        for prefix in (
            "initial_",
            "final_",
            "raw_",
            "calibrated_",
            "smoothed_",
            "filtered_",
        ):
            assert prefix in rendered_compose_system, (
                f"Banned prefix '{prefix}' not found"
            )

    def test_instrument_handling_section_present(
        self, rendered_compose_system: str
    ) -> None:
        """INSTRUMENT HANDLING section must appear."""
        assert "INSTRUMENT HANDLING" in rendered_compose_system

    def test_instrument_as_locus_rule_stated(
        self, rendered_compose_system: str
    ) -> None:
        """The postfix-locus rule for instrument names must be present."""
        assert "postfix locus" in rendered_compose_system
        assert "vacuum_wavelength_of_polarimeter_beam" in rendered_compose_system

    def test_anti_pattern_gallery_present(self, rendered_compose_system: str) -> None:
        """ANTI-PATTERN GALLERY section must appear with at least 5 entries."""
        assert "ANTI-PATTERN GALLERY" in rendered_compose_system
        # Each entry is marked 'Entry N'
        for entry_n in range(1, 6):
            assert f"Entry {entry_n}" in rendered_compose_system

    def test_gallery_real_failing_names(self, rendered_compose_system: str) -> None:
        """Gallery must contain verbatim real failing names from the EMW pilot."""
        assert "polarimeter_laser_wavelength" in rendered_compose_system
        assert (
            "initial_ellipticity_of_polarimeter_channel_beam" in rendered_compose_system
        )
        assert (
            "initial_polarization_of_polarimeter_channel_beam"
            in rendered_compose_system
        )

    def test_new_sections_before_dynamic_blocks(
        self, rendered_compose_system: str
    ) -> None:
        """Anti-pattern sections must appear before any dynamic-context content.

        The field_guidance block is the first dynamic section; the banned-prefix
        rule must precede it so the static cacheable prefix is maximised.
        """
        banned_idx = rendered_compose_system.find("BANNED PREFIXES")
        # 'Naming Guidance' is injected by the first {% if field_guidance %} block
        naming_guidance_idx = rendered_compose_system.find("Naming Guidance")
        assert banned_idx != -1, "BANNED PREFIXES not found"
        # If field_guidance block rendered at all, verify ordering
        if naming_guidance_idx != -1:
            assert banned_idx < naming_guidance_idx, (
                "BANNED PREFIXES must appear before dynamic Naming Guidance block"
            )
