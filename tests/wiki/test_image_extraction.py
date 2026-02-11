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
