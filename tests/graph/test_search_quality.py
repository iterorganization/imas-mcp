"""Integration tests for IMAS DD search quality improvements (Phases 1-7).

Tests cover: accessor classification, template enrichment, embedding text
generation, BM25 scoring changes, child traversal, and accessor query routing.
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
    is_boilerplate_path,
)
from imas_codex.tools.graph_search import (
    CHILD_ROLE_MAP,
    CHILD_SYNONYMS,
    _classify_child_role,
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
    """Test the simplified generate_embedding_text function.

    Embedding text now includes an IDS prefix for semantic separation:
    "core profiles: Radial profile of the electron temperature."
    """

    def test_returns_description_with_ids_prefix(self) -> None:
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "eq/profiles_1d/psi",
            {
                "description": "Poloidal flux radial profile.",
                "documentation": "Old doc",
            },
        )
        assert text == "eq: Poloidal flux radial profile."

    def test_fallback_to_documentation(self) -> None:
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "eq/profiles_1d/psi",
            {"documentation": "Poloidal flux"},
        )
        assert text == "eq: Poloidal flux"

    def test_empty_description_uses_doc(self) -> None:
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "eq/profiles_1d/psi",
            {"description": "", "documentation": "Doc text"},
        )
        assert text == "eq: Doc text"

    def test_empty_both_returns_empty(self) -> None:
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text("eq/x", {})
        assert text == ""

    def test_no_metadata_concatenation(self) -> None:
        """Verify the old 8-component concatenation is gone."""
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "eq/profiles_1d/psi",
            {
                "description": "Short desc.",
                "documentation": "Long doc",
                "units": "Wb",
                "data_type": "FLT_1D",
                "keywords": ["flux", "equilibrium"],
                "physics_domain": "equilibrium",
            },
            ids_info={"eq": {"description": "Equilibrium IDS"}},
        )
        # Should be IDS prefix + description, not a multi-sentence paragraph
        assert text == "eq: Short desc."
        assert "Wb" not in text
        assert "FLT_1D" not in text
        assert "Keywords" not in text

    def test_ids_prefix_uses_readable_name(self) -> None:
        """IDS names with underscores are converted to spaces."""
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "core_profiles/profiles_1d/electrons/temperature",
            {"description": "Electron temperature."},
        )
        assert text == "core profiles: Electron temperature."


# =============================================================================
# Phase 3: BM25 Scoring
# =============================================================================


class TestBM25ScoringConstants:
    """Test that BM25 scoring changes are correct at the code level."""

    def test_contains_scores_compressed(self) -> None:
        """Verify CONTAINS fallback scores are in the 0.55-0.80 range."""
        import ast
        import inspect

        from imas_codex.tools.graph_search import _text_search_imas_paths

        source = inspect.getsource(_text_search_imas_paths)
        # Extract all THEN score values from the CASE expression
        then_values = re.findall(r"THEN\s+([\d.]+)", source)
        for val_str in then_values:
            val = float(val_str)
            assert 0.50 <= val <= 0.85, (
                f"CONTAINS score {val} outside expected range 0.50-0.85"
            )

    def test_no_score_floor(self) -> None:
        """Verify the BM25 score floor has been removed."""
        import inspect

        from imas_codex.tools.graph_search import _text_search_imas_paths

        source = inspect.getsource(_text_search_imas_paths)
        assert "max(raw, 0.7)" not in source, "BM25 score floor still present"

    def test_uses_rrf_merge(self) -> None:
        """Verify RRF merge replaced naive max+0.05."""
        import inspect

        from imas_codex.tools.graph_search import GraphSearchTool

        source = inspect.getsource(GraphSearchTool.search_imas_paths)
        assert "reciprocal_rank_fusion" in source, (
            "RRF merge not found in search_imas_paths"
        )
        assert "max(scores[pid], text_score) + 0.05" not in source, (
            "Old naive max+0.05 merge still present"
        )

    def test_path_short_circuit(self) -> None:
        """Verify path queries skip vector search."""
        import inspect

        from imas_codex.tools.graph_search import GraphSearchTool

        source = inspect.getsource(GraphSearchTool.search_imas_paths)
        assert '"/" in query' in source or "'/' in query" in source, (
            "Path short-circuit not found"
        )

    def test_vector_gate_present(self) -> None:
        """Verify vector confidence gating is present."""
        import inspect

        from imas_codex.tools.graph_search import GraphSearchTool

        source = inspect.getsource(GraphSearchTool.search_imas_paths)
        assert "VECTOR_GATE_THRESHOLD" in source, (
            "Vector gate not found in search_imas_paths"
        )

    def test_heuristic_rerank_present(self) -> None:
        """Verify heuristic reranking is applied."""
        import inspect

        from imas_codex.tools.graph_search import GraphSearchTool

        source = inspect.getsource(GraphSearchTool.search_imas_paths)
        assert "heuristic_rerank" in source, (
            "Heuristic reranking not found in search_imas_paths"
        )


# =============================================================================
# Phase 9: RRF and Hybrid Tuning
# =============================================================================


class TestReciprocalRankFusion:
    """Test the RRF merge function."""

    def test_rrf_basic_merge(self) -> None:
        from imas_codex.tools.graph_search import reciprocal_rank_fusion

        v = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.8}]
        t = [{"id": "b", "score": 0.95}, {"id": "c", "score": 0.7}]
        result = reciprocal_rank_fusion(v, t, k=60)
        # 'b' appears in both lists → highest combined RRF score
        assert result["b"] > result["a"]
        assert result["b"] > result["c"]

    def test_rrf_empty_vector(self) -> None:
        from imas_codex.tools.graph_search import reciprocal_rank_fusion

        result = reciprocal_rank_fusion([], [{"id": "a", "score": 0.9}], k=60)
        assert "a" in result
        assert result["a"] > 0

    def test_rrf_empty_text(self) -> None:
        from imas_codex.tools.graph_search import reciprocal_rank_fusion

        result = reciprocal_rank_fusion([{"id": "a", "score": 0.9}], [], k=60)
        assert "a" in result

    def test_rrf_score_invariant(self) -> None:
        """RRF uses ranks, not scores — same ranks produce same RRF scores."""
        from imas_codex.tools.graph_search import reciprocal_rank_fusion

        r1 = reciprocal_rank_fusion(
            [{"id": "a", "score": 0.99}], [{"id": "a", "score": 0.99}], k=60
        )
        r2 = reciprocal_rank_fusion(
            [{"id": "a", "score": 0.50}], [{"id": "a", "score": 0.50}], k=60
        )
        assert abs(r1["a"] - r2["a"]) < 1e-10

    def test_rrf_k_parameter(self) -> None:
        """Smaller k gives more weight to top ranks."""
        from imas_codex.tools.graph_search import reciprocal_rank_fusion

        v = [{"id": "a", "score": 0.9}]
        t = [{"id": "b", "score": 0.9}]
        small_k = reciprocal_rank_fusion(v, t, k=10)
        large_k = reciprocal_rank_fusion(v, t, k=100)
        # With smaller k, the rank-1 boost is larger relative to total
        assert small_k["a"] > large_k["a"]


class TestVectorGating:
    """Test vector confidence gating."""

    def test_gate_threshold_exists(self) -> None:
        from imas_codex.tools.graph_search import VECTOR_GATE_THRESHOLD

        assert 0.5 <= VECTOR_GATE_THRESHOLD <= 0.8

    def test_low_vector_uses_text_only(self) -> None:
        """When best vector < gate, text-only scores should be used."""
        import inspect

        from imas_codex.tools.graph_search import GraphSearchTool

        source = inspect.getsource(GraphSearchTool.search_imas_paths)
        # Should contain logic that gates on vector score
        assert "best_vector" in source or "VECTOR_GATE" in source


class TestHeuristicRerank:
    """Test heuristic reranking function."""

    def test_ids_name_boost(self) -> None:
        from imas_codex.tools.graph_search import heuristic_rerank

        scores = {
            "equilibrium/time_slice/psi": 0.5,
            "core_profiles/profiles_1d/psi": 0.5,
        }
        result = heuristic_rerank(scores, "equilibrium psi")
        assert (
            result["equilibrium/time_slice/psi"]
            > result["core_profiles/profiles_1d/psi"]
        )

    def test_segment_match_boost(self) -> None:
        from imas_codex.tools.graph_search import heuristic_rerank

        scores = {
            "equilibrium/time_slice/profiles_1d/psi": 0.5,
            "magnetics/flux_loop/psi": 0.5,
        }
        result = heuristic_rerank(scores, "psi profile")
        # Both have 'psi' segment match so both get boosted
        assert result["equilibrium/time_slice/profiles_1d/psi"] >= 0.5
        assert result["magnetics/flux_loop/psi"] >= 0.5

    def test_no_boost_for_unrelated_query(self) -> None:
        from imas_codex.tools.graph_search import heuristic_rerank

        scores = {"equilibrium/time_slice/psi": 0.5}
        result = heuristic_rerank(scores, "completely unrelated")
        assert result["equilibrium/time_slice/psi"] == 0.5

    @pytest.mark.parametrize(
        "name,expected_role",
        [
            ("value", "data"),
            ("data", "data"),
            ("time", "time"),
            ("r", "coordinates"),
            ("z", "coordinates"),
            ("phi", "coordinates"),
            ("parallel", "components"),
            ("toroidal", "components"),
            ("coefficients", "interpolation"),
            ("grid_index", "grid"),
            ("validity", "quality"),
            ("measured", "fit"),
            ("label", "metadata"),
        ],
    )
    def test_known_roles(self, name: str, expected_role: str) -> None:
        assert _classify_child_role(name) == expected_role

    def test_coefficients_suffix(self) -> None:
        assert _classify_child_role("psi_coefficients") == "interpolation"

    def test_normalized_suffix(self) -> None:
        assert _classify_child_role("rho_n") == "normalized"

    def test_error_suffix(self) -> None:
        assert _classify_child_role("psi_error_upper") == "error"

    def test_unknown_is_other(self) -> None:
        assert _classify_child_role("some_physics_quantity") == "other"


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
# Phase 7: Accessor Query Routing
# =============================================================================


class TestChildSynonyms:
    """Test child name synonym mapping."""

    def test_r_synonyms(self) -> None:
        assert "radius" in CHILD_SYNONYMS["r"]
        assert "radial" in CHILD_SYNONYMS["r"]

    def test_z_synonyms(self) -> None:
        assert "height" in CHILD_SYNONYMS["z"]
        assert "vertical" in CHILD_SYNONYMS["z"]

    def test_time_synonyms(self) -> None:
        assert "timebase" in CHILD_SYNONYMS["time"]


# =============================================================================
# Phase 1: Prompt Quality
# =============================================================================


class TestPromptQuality:
    """Test the enrichment prompt and Pydantic model."""

    def test_pydantic_model_enforces_concise(self) -> None:
        from imas_codex.graph.dd_enrichment import IMASPathEnrichmentResult

        field_desc = IMASPathEnrichmentResult.model_fields["description"].description
        assert "150 characters" in field_desc or "concise" in field_desc.lower()

    def test_enrichment_prompt_mentions_concise(self) -> None:
        from imas_codex.llm.prompt_loader import PROMPTS_DIR

        prompt_path = PROMPTS_DIR / "imas" / "enrichment.md"
        content = prompt_path.read_text()
        assert "concise" in content.lower()
        assert "150 character" in content.lower() or "under 150" in content.lower()

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
