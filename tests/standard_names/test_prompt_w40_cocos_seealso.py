"""W40 prompt-hardening tests: COCOS-N prohibition and inline-only cross-refs.

Verifies that:
1. The shared `_coordinate_conventions.md` fragment forbids citing COCOS-N
   numbers in description / documentation prose.
2. Both `generate_name_system.md` and `enrich_system.md` (which include the fragment)
   render the COCOS-N prohibition.
3. The "See also:" trailer pattern is explicitly forbidden in `enrich_system.md`
   PR-3 (cross-references must be inline, not appended as a trailing block).
4. Stale anti-pattern examples ("under COCOS-11", "see also electron_temperature")
   no longer appear in the prompts.

These are content/structural assertions — they do not call the LLM.
"""

from __future__ import annotations

import re

import pytest

from imas_codex.llm.prompt_loader import PROMPTS_DIR


def _load(relpath: str) -> str:
    return (PROMPTS_DIR / relpath).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Coordinate-conventions fragment
# ---------------------------------------------------------------------------


class TestCoordinateConventionsFragment:
    PATH = "shared/sn/_coordinate_conventions.md"

    def test_has_cocos_section_heading(self) -> None:
        raw = _load(self.PATH)
        assert "## COCOS Convention" in raw

    def test_states_imas_uses_single_cocos(self) -> None:
        raw = _load(self.PATH)
        assert "single COCOS convention" in raw
        assert "COCOS-17" in raw

    def test_explicit_prohibition_phrasing(self) -> None:
        raw = _load(self.PATH)
        # The rule must be unambiguous and use FORBIDDEN.
        assert "FORBIDDEN" in raw
        # Whitespace-tolerant check for "MUST NEVER appear ..."
        normalized = re.sub(r"\s+", " ", raw)
        assert "MUST NEVER appear" in normalized

    @pytest.mark.parametrize(
        "forbidden_phrase",
        ["COCOS-11", "COCOS 11", "under COCOS"],
    )
    def test_lists_specific_forbidden_phrases(self, forbidden_phrase: str) -> None:
        raw = _load(self.PATH)
        # The fragment MUST list these forbidden examples explicitly so the LLM
        # sees them. Their presence here is documentation-of-prohibition, not
        # leakage — context is the ❌ table.
        assert forbidden_phrase in raw

    def test_required_replacement_uses_phi(self) -> None:
        raw = _load(self.PATH)
        assert "increasing toroidal angle" in raw

    def test_validator_regex_documented(self) -> None:
        """The fragment must document the validator regex contract."""
        raw = _load(self.PATH)
        # We promise the validator rejects any /COCOS[\s-]?\d/ in prose.
        assert "cocos" in raw.lower()
        assert "[0-9]" in raw or "\\d" in raw


# ---------------------------------------------------------------------------
# enrich_system.md — See also trailer prohibition
# ---------------------------------------------------------------------------


class TestEnrichSystemSeeAlsoProhibition:
    PATH = "sn/enrich_system.md"

    def test_pr3_section_present(self) -> None:
        raw = _load(self.PATH)
        assert "PR-3 Cross-reference inline-link format" in raw

    def test_inline_only_rule_present(self) -> None:
        raw = _load(self.PATH)
        assert "Inline-only rule" in raw

    @pytest.mark.parametrize(
        "trailer_phrase",
        ["See also:", "See related:", "Related:", "Cross-references:"],
    )
    def test_forbidden_trailer_phrases_listed(self, trailer_phrase: str) -> None:
        raw = _load(self.PATH)
        # PR-3 must explicitly enumerate forbidden trailer-block phrases.
        assert trailer_phrase in raw

    def test_old_trailer_example_removed(self) -> None:
        """The previous good-example taught the trailer pattern. It must be gone."""
        raw = _load(self.PATH)
        # The old "✅ GOOD: see also [electron_temperature](name:electron_temperature)"
        # promoted the trailer pattern.
        assert "✅ GOOD: `see also" not in raw

    def test_omit_link_if_no_natural_flow(self) -> None:
        raw = _load(self.PATH)
        # The fragment must direct the LLM to put orphan links in the structured
        # `links` array rather than in trailer prose.
        assert "OMIT it from" in raw or "place it in the structured `links`" in raw

    def test_no_redundant_units_rule_present(self) -> None:
        raw = _load(self.PATH)
        normalized = re.sub(r"\s+", " ", raw)
        # Units are structured metadata (HAS_UNIT edge); they MUST NOT appear
        # in prose for either the entry's own quantity or any linked sibling.
        assert "No-inline-units rule" in raw
        # The rule must cite both the user-surfaced exemplar and the entry's
        # own-quantity case.
        assert "fast neutral perpendicular pressure (in Pa)" in raw
        assert "electron temperature (in eV)" in raw
        # State the canonical principle (whitespace-tolerant — phrase may wrap).
        assert "structured metadata" in normalized
        assert "HAS_UNIT" in raw
        # The three narrow exceptions must be spelled out.
        assert "Numeric typical-value ranges" in raw
        assert "Equation variable definitions" in raw
        assert "Unit-conversion statements" in raw


# ---------------------------------------------------------------------------
# enrich_system.md — sign-convention example must not cite COCOS-N
# ---------------------------------------------------------------------------


class TestEnrichSystemSignConventionExample:
    PATH = "sn/enrich_system.md"

    def test_old_cocos11_example_removed(self) -> None:
        raw = _load(self.PATH)
        # The previous example was: "Positive when B_phi is in the +φ direction
        # under COCOS-11." That bled directly into generated documentation.
        assert "under COCOS-11" not in raw
        assert "under COCOS-17" not in raw
        assert "COCOS 11" not in raw

    def test_replacement_uses_phi_direction(self) -> None:
        raw = _load(self.PATH)
        assert "increasing toroidal angle" in raw


# ---------------------------------------------------------------------------
# generate_name_system.md — DS documentation sections removed (compose-only)
# ---------------------------------------------------------------------------


class TestComposeSystemDocSectionsRemoved:
    """DS-1..DS-8 and the Documentation Template are documentation-scope content
    that belongs in generate_docs_system.md, not the compose prompt. Verify they
    are absent — the compose prompt produces name/grammar/description only."""

    PATH = "sn/generate_name_system.md"

    def test_ds5_removed_from_compose(self) -> None:
        raw = _load(self.PATH)
        assert "DS-5 Sign conventions" not in raw

    def test_documentation_template_removed(self) -> None:
        raw = _load(self.PATH)
        assert "### Documentation Template" not in raw

    def test_documentation_structure_removed(self) -> None:
        raw = _load(self.PATH)
        assert "### Documentation Structure" not in raw

    def test_cocos_prohibition_still_available_via_include(self) -> None:
        """The COCOS-N prohibition lives in shared/_coordinate_conventions.md
        and is included by both compose and enrich prompts — verify the
        shared fragment still carries it."""
        raw = _load("shared/sn/_coordinate_conventions.md")
        normalized = re.sub(r"\s+", " ", raw)
        assert "MUST NEVER appear" in normalized


# ---------------------------------------------------------------------------
# Smoke test: rendered enrich_system prompt contains the new rules
# ---------------------------------------------------------------------------


class TestRenderedPromptIncludesCocosProhibition:
    """End-to-end: when we render a prompt that includes the coordinate-
    conventions fragment, the COCOS-N prohibition is present in the final text.
    """

    def test_enrich_system_render_includes_prohibition(self) -> None:
        # enrich_system.md `{% include %}`s the fragment; rendering should
        # interpolate it.
        from imas_codex.llm.prompt_loader import render_prompt

        rendered = render_prompt("sn/enrich_system", {})
        normalized = re.sub(r"\s+", " ", rendered)
        assert "COCOS Convention" in rendered
        assert "MUST NEVER appear" in normalized

    def test_compose_system_render_includes_prohibition(self) -> None:
        from imas_codex.llm.prompt_loader import render_prompt
        from imas_codex.standard_names.context import (
            build_compose_context,
            clear_context_cache,
        )

        clear_context_cache()
        rendered = render_prompt("sn/generate_name_system", build_compose_context())
        normalized = re.sub(r"\s+", " ", rendered)
        assert "COCOS Convention" in rendered
        assert "MUST NEVER appear" in normalized
