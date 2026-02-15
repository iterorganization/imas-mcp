"""Tests for image extraction from wiki page HTML."""

import pytest

from imas_codex.discovery.wiki.workers import _extract_image_refs


class TestExtractImageRefs:
    """Test _extract_image_refs extracts correct images from HTML."""

    BASE_URL = "https://wiki.example.com/wiki/TestPage"

    def _make_html(self, *img_tags: str) -> str:
        return f"<html><body>{''.join(img_tags)}</body></html>"

    def test_basic_image_extraction(self):
        html = self._make_html(
            '<img src="https://cdn.example.com/plot.png" alt="Plasma plot">'
        )
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 1
        assert refs[0]["src"] == "https://cdn.example.com/plot.png"
        assert refs[0]["alt_text"] == "Plasma plot"

    def test_relative_path_resolution(self):
        html = self._make_html('<img src="images/fig1.jpg" alt="">')
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 1
        assert refs[0]["src"] == "https://wiki.example.com/wiki/images/fig1.jpg"

    def test_absolute_path_resolution(self):
        html = self._make_html('<img src="/wiki/images/fig1.png" alt="">')
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 1
        assert refs[0]["src"] == "https://wiki.example.com/wiki/images/fig1.png"

    def test_protocol_relative_url(self):
        html = self._make_html('<img src="//cdn.example.com/img.jpeg" alt="">')
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 1
        assert refs[0]["src"] == "https://cdn.example.com/img.jpeg"

    def test_skip_favicon(self):
        html = self._make_html('<img src="/favicon.ico" width="16" height="16">')
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 0

    def test_skip_tiny_images(self):
        html = self._make_html('<img src="/img/dot.png" width="1" height="1">')
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 0

    def test_skip_skin_assets(self):
        html = self._make_html(
            '<img src="/skins/common/arrow.png" alt="">',
            '<img src="/skin/logo.png" alt="">',
        )
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 0

    def test_skip_data_uri(self):
        html = self._make_html(
            '<img src="data:image/png;base64,iVBORw0KGgo..." alt="inline">'
        )
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 0

    def test_skip_non_image_extensions(self):
        html = self._make_html(
            '<img src="/files/document.pdf">',
            '<img src="/files/style.css">',
        )
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 0

    def test_section_context_captured(self):
        html = """
        <html><body>
        <h2>Thomson Scattering</h2>
        <p>Electron density profiles measured by TS.</p>
        <img src="https://example.com/ts_profile.png" alt="Te profile">
        </body></html>
        """
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 1
        assert refs[0]["section"] == "Thomson Scattering"
        assert "Electron density" in refs[0]["surrounding_text"]

    def test_deduplication_not_done_here(self):
        """_extract_image_refs does NOT deduplicate — that's _extract_and_persist_images' job."""
        html = self._make_html(
            '<img src="https://example.com/same.png" alt="">',
            '<img src="https://example.com/same.png" alt="">',
        )
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        # Both are returned; dedup happens in _extract_and_persist_images
        assert len(refs) == 2

    def test_multiple_valid_images(self):
        html = self._make_html(
            '<img src="https://example.com/fig1.png" alt="Figure 1">',
            '<img src="https://example.com/fig2.jpg" alt="Figure 2">',
            '<img src="https://example.com/fig3.gif" alt="Figure 3">',
        )
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 3

    def test_skip_icon_logo_bullet_patterns(self):
        html = self._make_html(
            '<img src="/images/icon_edit.png" alt="">',
            '<img src="/images/bullet_point.gif" alt="">',
            '<img src="/images/spacer.png" alt="">',
            '<img src="/images/real_plot.png" alt="Real data">',
        )
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 1
        assert "real_plot" in refs[0]["src"]

    def test_query_string_in_url(self):
        html = self._make_html(
            '<img src="https://example.com/image.png?v=2&w=800" alt="versioned">'
        )
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 1

    def test_empty_html(self):
        refs = _extract_image_refs("", self.BASE_URL, "Test")
        assert len(refs) == 0

    def test_no_images_html(self):
        html = "<html><body><p>Just text, no images.</p></body></html>"
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 0


class TestConfluenceImageExtraction:
    """Test _extract_image_refs with Confluence storage format (ac:image tags)."""

    BASE_URL = "https://confluence.iter.org/pages/viewpage.action?pageId=887861486"

    def test_ac_image_attachment(self):
        """Confluence pages embed images as <ac:image><ri:attachment>."""
        html = (
            '<ac:image ac:height="250">'
            '<ri:attachment ri:filename="plot.png" />'
            "</ac:image>"
        )
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 1
        assert refs[0]["src"] == (
            "https://confluence.iter.org/download/attachments/887861486/plot.png"
        )
        assert refs[0]["height"] == "250"

    def test_ac_image_with_alt(self):
        html = (
            '<ac:image ac:alt="EQDSK output" ac:width="400" ac:height="300">'
            '<ri:attachment ri:filename="eqdsk.jpg" />'
            "</ac:image>"
        )
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 1
        assert refs[0]["alt_text"] == "EQDSK output"
        assert refs[0]["width"] == "400"
        assert refs[0]["height"] == "300"

    def test_ac_image_external_url(self):
        """Confluence <ac:image> with <ri:url> for external images."""
        html = (
            '<ac:image ac:height="100">'
            '<ri:url ri:value="https://external.org/graph.png" />'
            "</ac:image>"
        )
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 1
        assert refs[0]["src"] == "https://external.org/graph.png"

    def test_ac_image_section_context(self):
        html = """
        <h2>Equilibrium Results</h2>
        <p>Reconstructed profiles</p>
        <ac:image ac:height="250">
            <ri:attachment ri:filename="eq_profile.png" />
        </ac:image>
        """
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 1
        assert refs[0]["section"] == "Equilibrium Results"
        assert "Reconstructed" in refs[0]["surrounding_text"]

    def test_ac_image_filename_url_encoding(self):
        """Filenames with special chars should be URL-encoded."""
        html = (
            '<ac:image ac:height="200">'
            '<ri:attachment ri:filename="image with spaces.png" />'
            "</ac:image>"
        )
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 1
        assert "image%20with%20spaces.png" in refs[0]["src"]

    def test_multiple_ac_images(self):
        html = (
            '<ac:image><ri:attachment ri:filename="a.png" /></ac:image>'
            '<ac:image><ri:attachment ri:filename="b.jpg" /></ac:image>'
            '<ac:image><ri:attachment ri:filename="c.gif" /></ac:image>'
        )
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 3

    def test_ac_image_without_page_id_skipped(self):
        """If page URL doesn't contain pageId, attachment images are skipped."""
        html = '<ac:image><ri:attachment ri:filename="plot.png" /></ac:image>'
        refs = _extract_image_refs(
            html, "https://confluence.iter.org/display/SPACE/Page", "Test"
        )
        # No pageId in URL → can't construct download URL → skipped
        assert len(refs) == 0

    def test_ac_image_non_image_extension_skipped(self):
        html = '<ac:image><ri:attachment ri:filename="document.pdf" /></ac:image>'
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 0

    def test_mixed_img_and_ac_image(self):
        """Both standard img and Confluence ac:image should be extracted."""
        html = """
        <img src="https://example.com/standard.png" alt="standard">
        <ac:image ac:height="200">
            <ri:attachment ri:filename="confluence.png" />
        </ac:image>
        """
        refs = _extract_image_refs(html, self.BASE_URL, "Test")
        assert len(refs) == 2
        srcs = {r["src"] for r in refs}
        assert "https://example.com/standard.png" in srcs
        assert any("confluence.png" in s for s in srcs)
