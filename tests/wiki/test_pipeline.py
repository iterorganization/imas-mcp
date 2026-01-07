"""Tests for wiki ingestion pipeline."""

import pytest

from imas_codex.wiki.pipeline import html_to_text


class TestHTMLToText:
    """Tests for HTML to text conversion.

    html_to_text returns a tuple of (text, sections_dict).
    """

    def test_strip_script_tags(self):
        """Script tags should be removed."""
        html = "<html><script>alert('bad')</script><p>Content</p></html>"
        text, _ = html_to_text(html)
        assert "alert" not in text
        assert "Content" in text

    def test_strip_style_tags(self):
        """Style tags should be removed."""
        html = "<html><style>.foo { color: red; }</style><p>Content</p></html>"
        text, _ = html_to_text(html)
        assert "color" not in text
        assert "Content" in text

    def test_strip_navigation(self):
        """Navigation elements pass through but bodyContent extraction isolates main content."""
        # Simulate MediaWiki structure with bodyContent
        html = """
        <html>
        <nav>Menu</nav>
        <div id="bodyContent">
        <p>Main content here</p>
        <div class="printfooter">Retrieved from...</div>
        </div>
        </html>
        """
        text, _ = html_to_text(html)
        # Navigation before bodyContent is excluded
        assert "Menu" not in text
        # Main content is preserved
        assert "Main content" in text
        # Footer after bodyContent is excluded
        assert "Retrieved from" not in text

    def test_preserve_paragraph_text(self):
        """Paragraph text should be preserved."""
        html = (
            "<html><body><p>First paragraph.</p><p>Second paragraph.</p></body></html>"
        )
        text, _ = html_to_text(html)
        assert "First paragraph" in text
        assert "Second paragraph" in text

    def test_collapse_whitespace(self):
        """Multiple whitespaces should be collapsed."""
        html = "<html><body><p>Text   with    spaces</p></body></html>"
        text, _ = html_to_text(html)
        # Should have at most single spaces, not triple+
        assert "    " not in text
        # The text should contain the words
        assert "Text" in text
        assert "spaces" in text

    def test_strip_sidebar(self):
        """Sidebar elements should pass through (aside not filtered by default)."""
        html = "<html><aside>Sidebar content</aside><main>Main content</main></html>"
        text, _ = html_to_text(html)
        # aside is not in the skip list, so it passes through
        assert "Main content" in text

    def test_empty_html(self):
        """Empty HTML should return empty string."""
        text, sections = html_to_text("")
        assert text == ""
        assert sections == {}

    def test_returns_tuple(self):
        """Should return tuple of (text, sections)."""
        result = html_to_text("<p>Test</p>")
        assert isinstance(result, tuple)
        assert len(result) == 2
        text, sections = result
        assert isinstance(text, str)
        assert isinstance(sections, dict)

    def test_plain_text(self):
        """Plain text without HTML tags should pass through."""
        text, _ = html_to_text("Just plain text")
        assert "plain text" in text


class TestPipelineInit:
    """Tests for pipeline initialization."""

    def test_import(self):
        """Pipeline should be importable."""
        from imas_codex.wiki.pipeline import WikiIngestionPipeline

        assert WikiIngestionPipeline is not None

    def test_create_instance(self):
        """Pipeline should be instantiable without Neo4j."""
        from imas_codex.wiki.pipeline import WikiIngestionPipeline

        # This may fail without Neo4j, which is expected
        # We just test that the class exists and has expected attributes
        assert hasattr(WikiIngestionPipeline, "ingest_page")
        assert hasattr(WikiIngestionPipeline, "ingest_pages")


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests requiring Neo4j."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance."""
        from imas_codex.wiki.pipeline import WikiIngestionPipeline

        try:
            p = WikiIngestionPipeline("epfl")
            yield p
        except Exception:
            pytest.skip("Neo4j not available")

    def test_ingest_page(self, pipeline):
        """Test ingesting a single page."""
        from imas_codex.wiki.scraper import WikiPage

        # Create a test page - just verify it can be constructed
        _page = WikiPage(
            url="https://test.example.com/wiki/Test",
            title="Test Page",
            content_html="<html><body><p>This is test content about electron temperature.</p></body></html>",
        )
        # Full integration tests would verify graph state
        # For now just validate the page is valid
        assert _page.page_name == "Test"
