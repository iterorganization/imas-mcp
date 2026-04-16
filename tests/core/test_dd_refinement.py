"""Tests for DD refinement pipeline components.

Covers:
- compute_refinement_hash idempotency and sensitivity
- build_refinement_messages structure
- _RESET_CLEAR_FIELDS and _RESET_SOURCE_STATUSES dictionaries
- Pass 2 EXISTS-based query pattern (regression guard)
"""

import pytest

# ──────────────────────────────────────────────────────────────────
# compute_refinement_hash
# ──────────────────────────────────────────────────────────────────


class TestComputeRefinementHash:
    """Tests for hash-based idempotency in refinement."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from imas_codex.graph.dd_enrichment import compute_refinement_hash

        self.hash_fn = compute_refinement_hash

    def test_deterministic(self):
        h1 = self.hash_fn("desc", ["sib1"], ["peer1"], "model-a")
        h2 = self.hash_fn("desc", ["sib1"], ["peer1"], "model-a")
        assert h1 == h2

    def test_changes_with_description(self):
        h1 = self.hash_fn("desc A", ["sib"], ["peer"], "model")
        h2 = self.hash_fn("desc B", ["sib"], ["peer"], "model")
        assert h1 != h2

    def test_changes_with_model(self):
        h1 = self.hash_fn("desc", ["sib"], ["peer"], "model-a")
        h2 = self.hash_fn("desc", ["sib"], ["peer"], "model-b")
        assert h1 != h2

    def test_changes_with_siblings(self):
        h1 = self.hash_fn("desc", ["sib1"], ["peer"], "model")
        h2 = self.hash_fn("desc", ["sib1", "sib2"], ["peer"], "model")
        assert h1 != h2

    def test_order_independent_siblings(self):
        h1 = self.hash_fn("desc", ["sib1", "sib2"], [], "model")
        h2 = self.hash_fn("desc", ["sib2", "sib1"], [], "model")
        assert h1 == h2

    def test_order_independent_peers(self):
        h1 = self.hash_fn("desc", [], ["peer1", "peer2"], "model")
        h2 = self.hash_fn("desc", [], ["peer2", "peer1"], "model")
        assert h1 == h2

    def test_empty_context(self):
        h = self.hash_fn("description", [], [], "model")
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_length(self):
        h = self.hash_fn("d", ["s"], ["p"], "m")
        assert len(h) == 16  # sha256[:16]


# ──────────────────────────────────────────────────────────────────
# Reset dictionaries
# ──────────────────────────────────────────────────────────────────


class TestResetDictionaries:
    """Validate _RESET_CLEAR_FIELDS and _RESET_SOURCE_STATUSES."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from imas_codex.graph.dd_graph_ops import (
            _RESET_CLEAR_FIELDS,
            _RESET_SOURCE_STATUSES,
        )

        self.clear_fields = _RESET_CLEAR_FIELDS
        self.source_statuses = _RESET_SOURCE_STATUSES

    def test_same_keys(self):
        assert set(self.clear_fields.keys()) == set(self.source_statuses.keys())

    def test_built_clears_enrichment_and_refinement(self):
        fields = self.clear_fields["built"]
        assert "description" in fields
        assert "enrichment_hash" in fields
        assert "refinement_hash" in fields
        assert "refined_at" in fields
        assert "embedding" in fields

    def test_enriched_clears_refinement_not_enrichment(self):
        fields = self.clear_fields["enriched"]
        assert "refinement_hash" in fields
        assert "refined_at" in fields
        assert "embedding" in fields
        # Should NOT clear enrichment fields
        assert "description" not in fields
        assert "enrichment_hash" not in fields

    def test_refined_clears_embedding_not_refinement(self):
        fields = self.clear_fields["refined"]
        assert "embedding" in fields
        assert "embedded_at" in fields
        # Should NOT clear refinement fields
        assert "refinement_hash" not in fields
        assert "refined_at" not in fields
        assert "description" not in fields

    def test_built_sources_include_all_downstream(self):
        sources = self.source_statuses["built"]
        assert "enriched" in sources
        assert "refined" in sources
        assert "embedded" in sources

    def test_enriched_sources_include_refined_and_embedded(self):
        sources = self.source_statuses["enriched"]
        assert "refined" in sources
        assert "embedded" in sources
        assert "enriched" not in sources

    def test_refined_sources_include_embedded(self):
        sources = self.source_statuses["refined"]
        assert "embedded" in sources
        assert "refined" not in sources


# ──────────────────────────────────────────────────────────────────
# Node lifecycle constants
# ──────────────────────────────────────────────────────────────────


class TestNodeLifecycle:
    """Verify category constants include geometry."""

    def test_embeddable_includes_geometry(self):
        from imas_codex.core.node_categories import EMBEDDABLE_CATEGORIES

        assert "geometry" in EMBEDDABLE_CATEGORIES
        assert "quantity" in EMBEDDABLE_CATEGORIES
        assert "coordinate" not in EMBEDDABLE_CATEGORIES

    def test_enrichable_includes_coordinate_and_geometry(self):
        from imas_codex.core.node_categories import ENRICHABLE_CATEGORIES

        assert "geometry" in ENRICHABLE_CATEGORIES
        assert "quantity" in ENRICHABLE_CATEGORIES
        assert "coordinate" in ENRICHABLE_CATEGORIES

    def test_searchable_includes_coordinate_and_geometry(self):
        from imas_codex.core.node_categories import SEARCHABLE_CATEGORIES

        assert "geometry" in SEARCHABLE_CATEGORIES
        assert "quantity" in SEARCHABLE_CATEGORIES
        assert "coordinate" in SEARCHABLE_CATEGORIES


# ──────────────────────────────────────────────────────────────────
# build_refinement_messages
# ──────────────────────────────────────────────────────────────────


class TestBuildRefinementMessages:
    """Tests for refinement message construction."""

    def test_returns_messages_list(self):
        from imas_codex.graph.dd_enrichment import build_refinement_messages

        contexts = [
            {
                "id": "equilibrium/time_slice/profiles_1d/pressure",
                "name": "pressure",
                "data_type": "FLT_1D",
                "description": "Pressure profile",
                "siblings": [{"name": "q", "description": "Safety factor"}],
                "cluster_peers": [],
            }
        ]
        ids_info = {
            "equilibrium": {
                "name": "equilibrium",
                "description": "Equilibrium IDS",
            }
        }

        messages = build_refinement_messages(contexts, ids_info)
        assert isinstance(messages, list)
        assert len(messages) >= 1
        # Should have system and user messages
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles
