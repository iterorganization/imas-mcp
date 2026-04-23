"""Tests for node_categories — centralized category constants."""

from imas_codex.core.node_categories import (
    CLUSTERABLE_CATEGORIES,
    EMBEDDABLE_CATEGORIES,
    ENRICHABLE_CATEGORIES,
    IDENTIFIER_CATEGORIES,
    QUANTITY_CATEGORIES,
    SEARCHABLE_CATEGORIES,
    SN_SOURCE_CATEGORIES,
)


class TestCategoryConstants:
    """Verify category sets have correct membership and relationships."""

    def test_quantity_includes_geometry(self):
        assert "geometry" in QUANTITY_CATEGORIES

    def test_quantity_includes_quantity(self):
        assert "quantity" in QUANTITY_CATEGORIES

    def test_embeddable_superset_of_quantity(self):
        assert QUANTITY_CATEGORIES < EMBEDDABLE_CATEGORIES
        assert IDENTIFIER_CATEGORIES <= EMBEDDABLE_CATEGORIES

    def test_identifier_in_embeddable(self):
        assert "identifier" in EMBEDDABLE_CATEGORIES

    def test_identifier_in_searchable(self):
        assert "identifier" in SEARCHABLE_CATEGORIES

    def test_identifier_not_in_clusterable(self):
        assert "identifier" not in CLUSTERABLE_CATEGORIES

    def test_clusterable_equals_quantity(self):
        assert CLUSTERABLE_CATEGORIES == QUANTITY_CATEGORIES

    def test_searchable_superset_of_quantity(self):
        assert QUANTITY_CATEGORIES < SEARCHABLE_CATEGORIES

    def test_searchable_includes_coordinate(self):
        assert "coordinate" in SEARCHABLE_CATEGORIES

    def test_enrichable_includes_coordinate(self):
        assert "coordinate" in ENRICHABLE_CATEGORIES

    def test_enrichable_superset_of_quantity(self):
        assert QUANTITY_CATEGORIES < ENRICHABLE_CATEGORIES

    def test_sn_source_includes_coordinate(self):
        """Coordinate leaves deserve standard names (user directive 2026-04-23)."""
        assert "coordinate" in SN_SOURCE_CATEGORIES

    def test_sn_source_superset_of_quantity(self):
        assert QUANTITY_CATEGORIES <= SN_SOURCE_CATEGORIES

    def test_all_frozensets(self):
        for s in [
            QUANTITY_CATEGORIES,
            IDENTIFIER_CATEGORIES,
            EMBEDDABLE_CATEGORIES,
            CLUSTERABLE_CATEGORIES,
            SEARCHABLE_CATEGORIES,
            ENRICHABLE_CATEGORIES,
            SN_SOURCE_CATEGORIES,
        ]:
            assert isinstance(s, frozenset)

    def test_no_unexpected_categories(self):
        """All category sets should only contain known category values."""
        known = {"quantity", "geometry", "coordinate", "identifier"}
        all_cats = (
            QUANTITY_CATEGORIES
            | IDENTIFIER_CATEGORIES
            | EMBEDDABLE_CATEGORIES
            | CLUSTERABLE_CATEGORIES
            | SEARCHABLE_CATEGORIES
            | ENRICHABLE_CATEGORIES
            | SN_SOURCE_CATEGORIES
        )
        assert all_cats <= known
