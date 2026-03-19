"""Tests for wiki ingestion pipeline."""

import pytest

from imas_codex.discovery.wiki.pipeline import html_to_text


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


class TestParseMetaField:
    """Tests for _parse_meta_field() — META:FIELD value extraction."""

    def test_basic_field(self):
        """Should extract name/value from a simple META:FIELD line."""
        from imas_codex.discovery.wiki.pipeline import _parse_meta_field

        line = '%META:FIELD{name="Proposal" title="Proposal" value="magnetic sensor calibration"}%'
        result = _parse_meta_field(line)
        assert "<b>Proposal</b>" in result
        assert "magnetic sensor calibration" in result

    def test_empty_value_skipped(self):
        """Fields with empty values should return empty string."""
        from imas_codex.discovery.wiki.pipeline import _parse_meta_field

        line = '%META:FIELD{name="Comment" title="Comment" value=""}%'
        assert _parse_meta_field(line) == ""

    def test_whitespace_only_value_skipped(self):
        """Fields with whitespace-only values should return empty string."""
        from imas_codex.discovery.wiki.pipeline import _parse_meta_field

        line = '%META:FIELD{name="Comment" title="Comment" value="   "}%'
        assert _parse_meta_field(line) == ""

    def test_url_encoded_value(self):
        """URL-encoded values (Japanese text, newlines) should be decoded."""
        from imas_codex.discovery.wiki.pipeline import _parse_meta_field

        line = '%META:FIELD{name="Text" title="Text" value="%E9%81%8B%E8%BB%A2%E6%97%A5%E8%AA%8C"}%'
        result = _parse_meta_field(line)
        assert "運転日誌" in result  # "operation log" in Japanese

    def test_newlines_converted_to_br(self):
        """Encoded newlines should become <br> tags."""
        from imas_codex.discovery.wiki.pipeline import _parse_meta_field

        line = '%META:FIELD{name="Text" title="Text" value="line1%0d%0aline2"}%'
        result = _parse_meta_field(line)
        assert "<br>" in result
        assert "line1" in result
        assert "line2" in result

    def test_system_meta_skipped(self):
        """TOPICINFO, TOPICPARENT, FORM, etc. should return empty string."""
        from imas_codex.discovery.wiki.pipeline import _parse_meta_field

        assert (
            _parse_meta_field('%META:TOPICINFO{author="admin" date="1234567890"}%')
            == ""
        )
        assert _parse_meta_field('%META:TOPICPARENT{name="WebHome"}%') == ""
        assert _parse_meta_field('%META:FORM{name="ShotAForm"}%') == ""
        assert _parse_meta_field('%META:FILEATTACHMENT{name="data.csv"}%') == ""
        assert (
            _parse_meta_field('%META:PREFERENCE{name="VIEW_TEMPLATE" value="ShotA"}%')
            == ""
        )

    def test_malformed_line(self):
        """Malformed lines should return empty string, not crash."""
        from imas_codex.discovery.wiki.pipeline import _parse_meta_field

        assert _parse_meta_field("%META:") == ""
        assert _parse_meta_field("%META:FIELD{}%") == ""
        assert _parse_meta_field("not a meta line") == ""


class TestTwikiMarkupToHtml:
    """Tests for twiki_markup_to_html() — full TWiki markup conversion."""

    def test_meta_field_included(self):
        """META:FIELD values should appear in the HTML output."""
        from imas_codex.discovery.wiki.pipeline import twiki_markup_to_html

        markup = (
            '%META:TOPICINFO{author="admin" date="1234567890"}%\n'
            '%META:FORM{name="ShotAForm"}%\n'
            '%META:FIELD{name="Proposal" title="Proposal" value="magnetic sensor calibration"}%\n'
            '%META:FIELD{name="PreComment" title="PreComment" value="OK, data collection"}%\n'
        )
        html = twiki_markup_to_html(markup)
        assert "magnetic sensor calibration" in html
        assert "OK, data collection" in html
        # System metadata should NOT appear
        assert "admin" not in html
        assert "ShotAForm" not in html

    def test_meta_field_empty_body(self):
        """Pages with only META:FIELD data (no body text) should still produce content."""
        from imas_codex.discovery.wiki.pipeline import twiki_markup_to_html

        markup = (
            '%META:TOPICINFO{author="user" date="1234567890"}%\n'
            '%META:TOPICPARENT{name="WebHome"}%\n'
            '%META:FORM{name="SystemDailyReportForm"}%\n'
            '%META:FIELD{name="Date" title="Date" value="2024-01-15"}%\n'
            '%META:FIELD{name="Group" title="Group" value="計測班"}%\n'
            '%META:FIELD{name="Kind" title="Kind" value="運転日誌"}%\n'
        )
        html = twiki_markup_to_html(markup)
        assert "2024-01-15" in html
        assert "計測班" in html
        assert "運転日誌" in html
        # Should have at least 3 <p><b> blocks
        assert html.count("<b>") >= 3

    def test_headings(self):
        """TWiki headings should convert to HTML headings."""
        from imas_codex.discovery.wiki.pipeline import twiki_markup_to_html

        markup = "---+ Main Heading\n---++ Sub Heading\nSome text"
        html = twiki_markup_to_html(markup)
        assert "<h1>" in html
        assert "<h2>" in html
        assert "Main Heading" in html

    def test_verbatim_blocks(self):
        """Verbatim blocks should become <pre> tags."""
        from imas_codex.discovery.wiki.pipeline import twiki_markup_to_html

        markup = "<verbatim>\ncode here\n</verbatim>"
        html = twiki_markup_to_html(markup)
        assert "<pre>" in html
        assert "code here" in html


class TestPipelineInit:
    """Tests for pipeline initialization."""

    def test_import(self):
        """Pipeline should be importable."""
        from imas_codex.discovery.wiki.pipeline import WikiIngestionPipeline

        assert WikiIngestionPipeline is not None

    def test_create_instance(self):
        """Pipeline should be instantiable without Neo4j."""
        from imas_codex.discovery.wiki.pipeline import WikiIngestionPipeline

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
        from imas_codex.discovery.wiki.pipeline import WikiIngestionPipeline

        try:
            p = WikiIngestionPipeline("tcv")
            yield p
        except Exception:
            pytest.skip("Neo4j not available")

    def test_ingest_page(self, pipeline):
        """Test ingesting a single page."""
        from imas_codex.discovery.wiki.scraper import WikiPage

        # Create a test page - just verify it can be constructed
        _page = WikiPage(
            url="https://test.example.com/wiki/Test",
            title="Test Page",
            content_html="<html><body><p>This is test content about electron temperature.</p></body></html>",
        )
        # Full integration tests would verify graph state
        # For now just validate the page is valid
        assert _page.page_name == "Test"
