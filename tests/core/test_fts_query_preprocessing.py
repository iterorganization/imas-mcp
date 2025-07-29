"""
Test for FTS query preprocessing to handle minus operator correctly.
"""

from imas_mcp.search.document_store import DocumentStore


class TestFTSQueryPreprocessing:
    """Test FTS query preprocessing functionality."""

    def test_preprocess_fts_query_minus_operator(self):
        """Test that minus operator is correctly converted to NOT."""
        store = DocumentStore()

        # Test basic minus conversion
        assert (
            store._preprocess_fts_query("plasma temperature -wall")
            == "plasma temperature NOT wall"
        )

    def test_preprocess_fts_query_multiple_minus(self):
        """Test multiple minus operators."""
        store = DocumentStore()

        assert (
            store._preprocess_fts_query("transport -wall -edge")
            == "transport NOT wall NOT edge"
        )

    def test_preprocess_fts_query_leading_minus(self):
        """Test minus at start of query."""
        store = DocumentStore()

        assert (
            store._preprocess_fts_query("-wall temperature") == "NOT wall temperature"
        )

    def test_preprocess_fts_query_preserves_quoted_text(self):
        """Test that quoted text with minus is preserved."""
        store = DocumentStore()

        assert store._preprocess_fts_query('"quoted -text"') == '"quoted -text"'

    def test_preprocess_fts_query_preserves_field_syntax(self):
        """Test that field:value syntax is preserved."""
        store = DocumentStore()

        # Field syntax should be preserved
        assert (
            store._preprocess_fts_query("field:value -term") == "field:value NOT term"
        )

    def test_preprocess_fts_query_normal_queries_unchanged(self):
        """Test that normal queries without minus are unchanged."""
        store = DocumentStore()

        assert store._preprocess_fts_query("normal query") == "normal query"
        assert store._preprocess_fts_query("electron density") == "electron density"

    def test_preprocess_fts_query_complex_cases(self):
        """Test complex query combinations."""
        store = DocumentStore()

        # Mixed quoted and unquoted with minus
        query = 'plasma "high -density" -wall temperature'
        expected = 'plasma "high -density" NOT wall temperature'
        assert store._preprocess_fts_query(query) == expected
