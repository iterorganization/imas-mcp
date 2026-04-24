"""Integration tests for IMAS DD search quality improvements.

Tests cover: accessor classification, template enrichment, embedding text
generation, and BM25 scoring changes.
"""

from __future__ import annotations

import re

import pytest

from imas_codex.graph.dd_enrichment import (
    ACCESSOR_REGEX_PATTERNS,
    ACCESSOR_TEMPLATES,
    ACCESSOR_TERMINAL_NAMES,
    ACCESSOR_TERMINAL_SUFFIXES,
    FORCE_INCLUDE_CONCEPTS,
    classify_node,
    generate_template_description,
    is_accessor_terminal,
)

# =============================================================================
# Phase 6: Layered Accessor Classification
# =============================================================================


class TestClassifyNode:
    """Test the 5-layer classify_node pipeline."""

    # Layer 1: Error/metadata
    @pytest.mark.parametrize(
        "path_id,name",
        [
            ("eq/time_slice/psi_error_upper", "psi_error_upper"),
            ("eq/time_slice/psi_error_lower", "psi_error_lower"),
            ("eq/time_slice/psi_error_index", "psi_error_index"),
            ("eq/time_slice/psi_validity", "psi_validity"),
            ("eq/time_slice/psi_validity_timed", "psi_validity_timed"),
            ("eq/ids_properties/comment", "comment"),
        ],
    )
    def test_layer1_error_metadata(self, path_id: str, name: str) -> None:
        assert classify_node(path_id, name) == "accessor"

    # Layer 2: Force-include physics concepts
    @pytest.mark.parametrize("name", list(FORCE_INCLUDE_CONCEPTS))
    def test_layer2_force_include_concepts(self, name: str) -> None:
        """Physics concepts must always be classified as 'concept'."""
        assert classify_node(f"some_ids/{name}", name) == "concept"

    def test_psi_is_concept(self) -> None:
        """Regression: psi must be a concept, not an accessor."""
        assert (
            classify_node("equilibrium/time_slice/profiles_1d/psi", "psi") == "concept"
        )

    def test_density_is_concept(self) -> None:
        assert (
            classify_node("core_profiles/profiles_1d/electrons/density", "density")
            == "concept"
        )

    def test_temperature_is_concept(self) -> None:
        assert (
            classify_node(
                "core_profiles/profiles_1d/electrons/temperature", "temperature"
            )
            == "concept"
        )

    def test_pressure_is_concept(self) -> None:
        assert (
            classify_node("core_profiles/profiles_1d/pressure", "pressure") == "concept"
        )

    # Layer 3: Explicit accessor names
    @pytest.mark.parametrize(
        "name",
        [
            "value",
            "data",
            "time",
            "r",
            "z",
            "phi",
            "coefficients",
            "label",
            "grid_index",
            "parallel",
            "toroidal",
            "measured",
            "reconstructed",
        ],
    )
    def test_layer3_accessor_names(self, name: str) -> None:
        assert classify_node(f"some_ids/parent/{name}", name) == "accessor"

    # Layer 4: Regex patterns
    @pytest.mark.parametrize(
        "name",
        [
            "x_coefficients",
            "psi_n",
            "temperature_flag",
            "data_validate",
            "signal_scale",
            "value_offset",
        ],
    )
    def test_layer4_regex_patterns(self, name: str) -> None:
        assert classify_node(f"some_ids/{name}", name) == "accessor"

    # Layer 5: Frequency + structural heuristic
    def test_layer5_heuristic_accessor(self) -> None:
        stats = {"occurrence_count": 50, "structure_parent_ratio": 0.98}
        assert (
            classify_node("some_ids/unknown_field", "unknown_field", stats)
            == "accessor"
        )

    def test_layer5_heuristic_below_threshold(self) -> None:
        stats = {"occurrence_count": 10, "structure_parent_ratio": 0.98}
        assert classify_node("some_ids/rare_field", "rare_field", stats) == "concept"

    def test_layer5_heuristic_low_ratio(self) -> None:
        stats = {"occurrence_count": 50, "structure_parent_ratio": 0.5}
        assert classify_node("some_ids/mixed_field", "mixed_field", stats) == "concept"

    # Default: unknown names are concepts
    def test_default_is_concept(self) -> None:
        assert (
            classify_node("some_ids/unique_physics_thing", "unique_physics_thing")
            == "concept"
        )

    def test_empty_name_is_concept(self) -> None:
        assert classify_node("some_ids/", "") == "concept"

    # Layer ordering: Layer 2 overrides Layer 3
    def test_force_include_overrides_accessor_names(self) -> None:
        """'q' appears in some accessor lists but should be protected as a concept."""
        assert classify_node("eq/profiles_1d/q", "q") == "concept"


class TestIsAccessorTerminal:
    """Test the is_accessor_terminal convenience function."""

    def test_value_is_accessor(self) -> None:
        assert is_accessor_terminal("eq/parent/value", "value") is True

    def test_psi_is_not_accessor(self) -> None:
        assert is_accessor_terminal("eq/profiles_1d/psi", "psi") is False

    def test_infers_name_from_path(self) -> None:
        assert is_accessor_terminal("eq/parent/time") is True

    def test_coefficients_suffix(self) -> None:
        assert (
            is_accessor_terminal("eq/parent/x_coefficients", "x_coefficients") is True
        )


# =============================================================================
# Phase 2: Template Enrichment
# =============================================================================


class TestAccessorTemplates:
    """Test template description generation for accessor terminals."""

    def test_value_template_uses_parent_doc(self) -> None:
        result = generate_template_description(
            "eq/global_quantities/ip/value",
            {"name": "value"},
            parent_info={"name": "ip", "documentation": "Plasma current"},
        )
        assert result["enrichment_source"] == "template"
        assert "Plasma current" in result["description"]

    def test_time_template(self) -> None:
        result = generate_template_description(
            "eq/profiles_1d/time",
            {"name": "time"},
            parent_info={"name": "profiles_1d", "documentation": "1D profiles"},
        )
        assert "Time base" in result["description"]
        assert "profiles 1d" in result["description"]

    def test_r_coordinate_template(self) -> None:
        result = generate_template_description(
            "eq/boundary/x_point/r",
            {"name": "r"},
            parent_info={"name": "x_point", "documentation": "X-point location"},
        )
        assert "Major radius" in result["description"]
        assert "x point" in result["description"]

    def test_z_coordinate_template(self) -> None:
        result = generate_template_description(
            "eq/boundary/x_point/z",
            {"name": "z"},
            parent_info={"name": "x_point"},
        )
        assert "Vertical" in result["description"]

    def test_parallel_component_template(self) -> None:
        result = generate_template_description(
            "some_ids/parent/parallel",
            {"name": "parallel"},
            parent_info={"name": "velocity", "documentation": "Flow velocity"},
        )
        assert "Parallel component" in result["description"]

    def test_validity_template(self) -> None:
        result = generate_template_description(
            "eq/parent/validity",
            {"name": "validity"},
            parent_info={"name": "q_95", "documentation": "Safety factor at 95%"},
        )
        assert "validity" in result["description"].lower()

    def test_coefficients_suffix_template(self) -> None:
        result = generate_template_description(
            "eq/parent/psi_coefficients",
            {"name": "psi_coefficients"},
            parent_info={"name": "parent_node", "documentation": "Parent doc"},
        )
        assert "coefficients" in result["description"].lower()
        assert result["enrichment_source"] == "template"

    def test_normalized_suffix_template(self) -> None:
        result = generate_template_description(
            "eq/parent/rho_n",
            {"name": "rho_n"},
            parent_info={"name": "parent_node", "documentation": "Parent doc"},
        )
        assert "Normalized" in result["description"]

    def test_no_parent_info_falls_through(self) -> None:
        """Accessor templates without parent_info should fall through to error/metadata."""
        result = generate_template_description(
            "eq/parent/psi_error_upper",
            {"name": "psi_error_upper"},
        )
        assert result["enrichment_source"] == "template"
        assert (
            "uncertainty" in result["description"].lower()
            or "error" in result["description"].lower()
        )

    def test_all_accessor_names_have_templates(self) -> None:
        """Every name in ACCESSOR_TEMPLATES should be in ACCESSOR_TERMINAL_NAMES."""
        for name in ACCESSOR_TEMPLATES:
            assert name in ACCESSOR_TERMINAL_NAMES or any(
                name.endswith(s) for s in ACCESSOR_TERMINAL_SUFFIXES
            ), f"Template name '{name}' not in ACCESSOR_TERMINAL_NAMES"


# =============================================================================
# Phase 4: Embedding Text Redesign
# =============================================================================


class TestGenerateEmbeddingTextConcise:
    """Test generate_embedding_text produces description-only format.

    At dim 256 (Matryoshka), path prefixes are excluded to avoid score
    compression and lexical interference. Embedding text is just the
    enriched description (or documentation fallback).
    Path matching is handled by the BM25 fulltext index instead.
    """

    def test_returns_description_with_full_path(self) -> None:
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "eq/profiles_1d/psi",
            {
                "description": "Poloidal flux radial profile.",
                "documentation": "Old doc",
            },
        )
        assert text == "Poloidal flux radial profile."

    def test_fallback_to_documentation(self) -> None:
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "eq/profiles_1d/psi",
            {"documentation": "Poloidal flux"},
        )
        assert text == "Poloidal flux"

    def test_empty_description_uses_doc(self) -> None:
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "eq/profiles_1d/psi",
            {"description": "", "documentation": "Doc text"},
        )
        assert text == "Doc text"

    def test_empty_both_returns_empty(self) -> None:
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text("eq/x", {})
        assert text == ""

    def test_no_raw_metadata_in_output(self) -> None:
        """Verify units, data_type, doc excerpts, and keywords are excluded."""
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "eq/profiles_1d/psi",
            {
                "description": "Short desc.",
                "documentation": "Long doc about poloidal flux details.",
                "units": "Wb",
                "data_type": "FLT_1D",
                "keywords": ["flux", "equilibrium"],
                "physics_domain": "equilibrium",
            },
            ids_info={"eq": {"description": "Equilibrium IDS"}},
        )
        assert text == "Short desc."
        assert "Wb" not in text
        assert "FLT_1D" not in text
        assert "Keywords" not in text
        assert "Long doc" not in text

    def test_full_path_replaces_readable_ids_prefix(self) -> None:
        """Path is excluded entirely (not reformatted as a readable prefix)."""
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "core_profiles/profiles_1d/electrons/temperature",
            {"description": "Electron temperature."},
        )
        assert text == "Electron temperature."
        assert "core_profiles" not in text
        assert not text.startswith("core profiles:")

    def test_output_contains_full_imas_path(self) -> None:
        """Path is excluded from embedding text to avoid score compression."""
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "equilibrium/time_slice/profiles_1d/psi",
            {"description": "Poloidal flux."},
        )
        assert "equilibrium/time_slice/profiles_1d/psi" not in text
        assert text == "Poloidal flux."

    def test_documentation_excluded_even_when_different(self) -> None:
        """Doc excerpts are excluded to preserve cosine quality at dim 256."""
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "equilibrium/time_slice/profiles_1d/psi",
            {
                "description": "Poloidal flux.",
                "documentation": "This is a long documentation string with details.",
            },
        )
        assert text == "Poloidal flux."
        assert "long documentation" not in text

    def test_keywords_excluded(self) -> None:
        """Keywords are excluded to preserve cosine quality at dim 256."""
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "equilibrium/time_slice/profiles_1d/psi",
            {
                "description": "Poloidal flux.",
                "keywords": ["flux", "equilibrium", "psi"],
            },
        )
        assert "Keywords" not in text
        assert text == "Poloidal flux."

    def test_separator_is_period_space(self) -> None:
        """When both description and (at ≥512 dim) extras are included, separator is '. '."""
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "eq/psi",
            {"description": "Desc."},
        )
        assert text == "Desc."


# =============================================================================
# Phase 3: BM25 Scoring
# =============================================================================


class TestBM25ScoringConstants:
    """Test that BM25 scoring changes are correct at the code level."""

    def test_contains_scores_compressed(self) -> None:
        """Verify CONTAINS/exact-match fallback scores are in the 0.50-0.98 range."""
        import ast
        import inspect

        from imas_codex.tools.graph_search import _text_search_dd_paths

        source = inspect.getsource(_text_search_dd_paths)
        # Extract all THEN score values from the CASE expression
        then_values = re.findall(r"THEN\s+([\d.]+)", source)
        for val_str in then_values:
            val = float(val_str)
            assert 0.50 <= val <= 0.98, (
                f"CONTAINS score {val} outside expected range 0.50-0.98"
            )

    def test_no_score_floor(self) -> None:
        """Verify the BM25 score floor has been removed."""
        import inspect

        from imas_codex.tools.graph_search import _text_search_dd_paths

        source = inspect.getsource(_text_search_dd_paths)
        assert "max(raw, 0.7)" not in source, "BM25 score floor still present"

    def test_path_short_circuit(self) -> None:
        """Verify path queries skip vector search.

        The short-circuit lives in ``hybrid_dd_search`` (imas_codex.graph.dd_search),
        which ``GraphSearchTool.search_dd_paths`` delegates to.
        """
        import inspect

        from imas_codex.graph.dd_search import hybrid_dd_search

        source = inspect.getsource(hybrid_dd_search)
        assert '"/" in query' in source or "'/' in query" in source, (
            "Path short-circuit not found in hybrid_dd_search"
        )


class TestSearchHitChildren:
    """Test that SearchHit can hold children and matched_children."""

    def test_children_field_exists(self) -> None:
        from imas_codex.search.search_strategy import SearchHit

        hit = SearchHit(
            path="eq/boundary",
            documentation="Boundary",
            ids_name="equilibrium",
            score=0.9,
            rank=1,
            search_mode="auto",
            children=[
                {
                    "role": "coordinates",
                    "children": [{"name": "r", "data_type": "FLT_1D"}],
                }
            ],
        )
        assert hit.children is not None
        assert len(hit.children) == 1

    def test_matched_children_field_exists(self) -> None:
        from imas_codex.search.search_strategy import SearchHit

        hit = SearchHit(
            path="eq/boundary/x_point",
            documentation="X-point",
            ids_name="equilibrium",
            score=0.87,
            rank=1,
            search_mode="auto",
            matched_children=["r", "z"],
        )
        assert hit.matched_children == ["r", "z"]

    def test_none_by_default(self) -> None:
        from imas_codex.search.search_strategy import SearchHit

        hit = SearchHit(
            path="eq/psi",
            documentation="Psi",
            ids_name="equilibrium",
            score=0.9,
            rank=1,
            search_mode="auto",
        )
        assert hit.children is None
        assert hit.matched_children is None


# =============================================================================
# Phase 1: Prompt Quality
# =============================================================================


class TestPromptQuality:
    """Test the enrichment prompt and Pydantic model."""

    def test_pydantic_model_enforces_concise(self) -> None:
        from imas_codex.graph.dd_enrichment import IMASPathEnrichmentResult

        field_desc = IMASPathEnrichmentResult.model_fields["description"].description
        assert "150" in field_desc or "concise" in field_desc.lower()

    def test_enrichment_prompt_mentions_concise(self) -> None:
        from imas_codex.llm.prompt_loader import PROMPTS_DIR

        prompt_path = PROMPTS_DIR / "imas" / "enrichment.md"
        content = prompt_path.read_text()
        assert "concise" in content.lower()
        assert "150" in content.lower()

    def test_enrichment_prompt_has_examples(self) -> None:
        from imas_codex.llm.prompt_loader import PROMPTS_DIR

        prompt_path = PROMPTS_DIR / "imas" / "enrichment.md"
        content = prompt_path.read_text()
        assert "GOOD" in content
        assert "BAD" in content


# =============================================================================
# Regex Pattern Coverage
# =============================================================================


class TestAccessorRegexPatterns:
    """Test that ACCESSOR_REGEX_PATTERNS catch future accessor patterns."""

    @pytest.mark.parametrize(
        "name",
        [
            "psi_error_upper",
            "temperature_uncertainty_lower",
            "data_flag",
            "signal_validate",
            "value_scale",
            "signal_offset",
            "x_coefficients",
            "rho_n",
        ],
    )
    def test_regex_catches_pattern(self, name: str) -> None:
        matched = any(p.search(name) for p in ACCESSOR_REGEX_PATTERNS)
        assert matched, f"ACCESSOR_REGEX_PATTERNS missed '{name}'"

    @pytest.mark.parametrize(
        "name",
        [
            "temperature",
            "density",
            "psi",
            "electron_temperature",
            "magnetic_field",
        ],
    )
    def test_regex_does_not_catch_physics(self, name: str) -> None:
        matched = any(p.search(name) for p in ACCESSOR_REGEX_PATTERNS)
        assert not matched, f"ACCESSOR_REGEX_PATTERNS falsely caught '{name}'"
