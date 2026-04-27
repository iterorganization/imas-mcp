"""Tests verifying IMAS coordinate convention guidance is rendered in SN system prompts.

Regression guard: if the shared fragment path breaks or the include is removed,
these tests catch it before bad documentation is generated.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def rendered_compose_system() -> str:
    """Render sn/compose_system with full grammar context."""
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.context import (
        build_compose_context,
        clear_context_cache,
    )

    clear_context_cache()
    context = build_compose_context()
    return render_prompt("sn/compose_system", context)


@pytest.fixture()
def rendered_enrich_system() -> str:
    """Render sn/enrich_system (static — no dynamic context needed)."""
    from imas_codex.llm.prompt_loader import render_prompt

    return render_prompt("sn/enrich_system", {})


class TestCoordinateGuidanceInComposeSystem:
    """Coordinate convention guidance must appear in the compose system prompt."""

    def test_right_handed_phrase_present(self, rendered_compose_system: str) -> None:
        """The phrase 'right-handed' must appear (establishes handedness fact)."""
        assert "right-handed" in rendered_compose_system

    def test_cylindrical_tuple_present(self, rendered_compose_system: str) -> None:
        r"""The explicit (R, \phi, Z) tuple must appear."""
        assert r"(R, \phi, Z)" in rendered_compose_system

    def test_prohibition_phrase_present(self, rendered_compose_system: str) -> None:
        """A prohibition against vague coordinate phrases must be stated."""
        assert (
            "NEVER" in rendered_compose_system
            or "do NOT" in rendered_compose_system
            or "Never" in rendered_compose_system
        ), "Expected a prohibition (NEVER / Never / do NOT) in the compose prompt"

    def test_vague_phrase_named_as_bad_example(
        self, rendered_compose_system: str
    ) -> None:
        """The vague phrase 'standard cylindrical' must appear as a bad example."""
        assert "standard cylindrical" in rendered_compose_system

    def test_coordinate_conventions_section_heading(
        self, rendered_compose_system: str
    ) -> None:
        """The coordinate conventions section heading must be present."""
        assert "Coordinate Convention" in rendered_compose_system or (
            "IMAS coordinate" in rendered_compose_system
        )

    def test_flux_coordinate_guidance_present(
        self, rendered_compose_system: str
    ) -> None:
        """Flux coordinates must be mentioned explicitly (no 'the standard')."""
        assert "flux coordinate" in rendered_compose_system.lower()

    def test_coordinate_guidance_before_naming_rules(
        self, rendered_compose_system: str
    ) -> None:
        """Coordinate convention block must precede the naming Rule 17 section.

        Static cacheable prefix is maximised when the include appears before
        the dynamic naming-rule numbering.
        """
        coord_idx = rendered_compose_system.find(r"(R, \phi, Z)")
        # Rule 17 (coordinate naming) must come after the convention block
        rule17_idx = rendered_compose_system.find("ABSOLUTE RULE")
        assert coord_idx != -1, r"(R, \phi, Z) not found in compose prompt"
        assert rule17_idx != -1, "ABSOLUTE RULE not found in compose prompt"
        assert coord_idx < rule17_idx, (
            "Coordinate convention block must precede Rule 17 (ABSOLUTE RULE)"
        )


class TestCoordinateGuidanceInEnrichSystem:
    """Coordinate convention guidance must appear in the enrich system prompt."""

    def test_right_handed_phrase_present(self, rendered_enrich_system: str) -> None:
        """The phrase 'right-handed' must appear."""
        assert "right-handed" in rendered_enrich_system

    def test_cylindrical_tuple_present(self, rendered_enrich_system: str) -> None:
        r"""The explicit (R, \phi, Z) tuple must appear."""
        assert r"(R, \phi, Z)" in rendered_enrich_system

    def test_prohibition_phrase_present(self, rendered_enrich_system: str) -> None:
        """A prohibition against vague coordinate phrases must be stated."""
        assert (
            "NEVER" in rendered_enrich_system
            or "do NOT" in rendered_enrich_system
            or "Never" in rendered_enrich_system
        ), "Expected a prohibition (NEVER / Never / do NOT) in the enrich prompt"

    def test_vague_phrase_named_as_bad_example(
        self, rendered_enrich_system: str
    ) -> None:
        """The vague phrase 'standard cylindrical' must appear as a bad example."""
        assert "standard cylindrical" in rendered_enrich_system

    def test_coordinate_conventions_section_heading(
        self, rendered_enrich_system: str
    ) -> None:
        """The coordinate conventions section heading must be present."""
        assert "Coordinate Convention" in rendered_enrich_system or (
            "IMAS coordinate" in rendered_enrich_system
        )

    def test_coordinate_guidance_before_documentation_template(
        self, rendered_enrich_system: str
    ) -> None:
        """Coordinate convention block must precede the Documentation Template section.

        Ensures the conventions are injected before the per-batch dynamic content
        so the static prompt prefix is maximised for caching.
        """
        coord_idx = rendered_enrich_system.find(r"(R, \phi, Z)")
        template_idx = rendered_enrich_system.find("Documentation Template")
        assert coord_idx != -1, r"(R, \phi, Z) not found in enrich prompt"
        assert template_idx != -1, "Documentation Template not found in enrich prompt"
        assert coord_idx < template_idx, (
            "Coordinate convention block must precede Documentation Template"
        )

    def test_cartesian_basis_mentioned(self, rendered_enrich_system: str) -> None:
        """Cartesian basis guidance for sensor vectors must be present."""
        assert (
            r"\hat{x}" in rendered_enrich_system
            or "Cartesian" in rendered_enrich_system
        )
