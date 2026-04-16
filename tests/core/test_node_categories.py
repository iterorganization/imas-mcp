"""Tests for node_categories — centralized category constants."""

from imas_codex.core.node_categories import (
    EMBEDDABLE_CATEGORIES,
    ENRICHABLE_CATEGORIES,
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

    def test_embeddable_equals_quantity(self):
        assert EMBEDDABLE_CATEGORIES == QUANTITY_CATEGORIES

    def test_searchable_superset_of_quantity(self):
        assert QUANTITY_CATEGORIES < SEARCHABLE_CATEGORIES

    def test_searchable_includes_coordinate(self):
        assert "coordinate" in SEARCHABLE_CATEGORIES

    def test_enrichable_includes_coordinate(self):
        assert "coordinate" in ENRICHABLE_CATEGORIES

    def test_enrichable_superset_of_quantity(self):
        assert QUANTITY_CATEGORIES < ENRICHABLE_CATEGORIES

    def test_sn_source_equals_quantity(self):
        assert SN_SOURCE_CATEGORIES == QUANTITY_CATEGORIES

    def test_all_frozensets(self):
        for s in [
            QUANTITY_CATEGORIES,
            EMBEDDABLE_CATEGORIES,
            SEARCHABLE_CATEGORIES,
            ENRICHABLE_CATEGORIES,
            SN_SOURCE_CATEGORIES,
        ]:
            assert isinstance(s, frozenset)

    def test_no_unexpected_categories(self):
        """All category sets should only contain known category values."""
        known = {"quantity", "geometry", "coordinate"}
        all_cats = (
            QUANTITY_CATEGORIES
            | EMBEDDABLE_CATEGORIES
            | SEARCHABLE_CATEGORIES
            | ENRICHABLE_CATEGORIES
            | SN_SOURCE_CATEGORIES
        )
        assert all_cats <= known
