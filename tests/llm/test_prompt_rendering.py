"""Tests for SN compose prompt rendering — live transformation injection + exemplar alignment."""

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


@pytest.fixture()
def live_transformation_tokens() -> list[str]:
    """Return the full list of Transformation enum values from ISN."""
    from imas_standard_names.grammar import Transformation

    return [t.value for t in Transformation]


class TestTransformationInjection:
    """B.2 — Verify static 4-token list replaced with live enum."""

    def test_rendered_prompt_contains_live_tokens(
        self, rendered_compose_system: str, live_transformation_tokens: list[str]
    ) -> None:
        """Every ISN Transformation token must appear in the rendered prompt."""
        for token in live_transformation_tokens:
            assert token in rendered_compose_system, (
                f"Live transformation token '{token}' missing from rendered compose_system.md"
            )

    def test_old_static_list_not_present(self, rendered_compose_system: str) -> None:
        """The old static heading 'Transformation Boundaries' must be gone."""
        assert "Transformation Boundaries" not in rendered_compose_system
        assert "Only these 4 transformation tokens" not in rendered_compose_system

    def test_live_heading_present(self, rendered_compose_system: str) -> None:
        """The new heading must be present."""
        assert (
            "Transformations (live from imas-standard-names)" in rendered_compose_system
        )


class TestExemplarAlignment:
    """B.1/B.3 — Verify exemplar P3/P8 rewrite and anti-patterns landed."""

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
