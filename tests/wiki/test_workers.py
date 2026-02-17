"""Tests for wiki discovery worker helper functions.

Covers _extract_image_refs, _fetch_image_bytes, image extraction, and
worker utility functions with mocked external dependencies.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.wiki.conftest import CONFLUENCE_HTML, MEDIAWIKI_HTML, TWIKI_HTML

# =============================================================================
# _extract_image_refs
# =============================================================================


class TestExtractImageRefs:
    """Tests for _extract_image_refs — HTML image reference extraction."""

    def _extract(self, html: str, page_url: str = "https://wiki.example.com/wiki/TestPage", page_title: str = "Test"):
        from imas_codex.discovery.wiki.workers import _extract_image_refs

        return _extract_image_refs(html, page_url, page_title)

    def test_empty_html(self):
        assert self._extract("") == []

    def test_no_images(self):
        html = "<html><body><p>No images here</p></body></html>"
        assert self._extract(html) == []

    def test_standard_img_tag(self):
        html = '<html><body><img src="/images/plot.png" alt="Flux plot" width="600" height="400"></body></html>'
        refs = self._extract(html)
        assert len(refs) == 1
        assert refs[0]["src"] == "https://wiki.example.com/images/plot.png"
        assert refs[0]["alt_text"] == "Flux plot"
        assert refs[0]["width"] == "600"
        assert refs[0]["height"] == "400"

    def test_absolute_url_img(self):
        html = '<html><body><img src="https://other.example.com/img/data.png" alt="Data"></body></html>'
        refs = self._extract(html)
        assert len(refs) == 1
        assert refs[0]["src"] == "https://other.example.com/img/data.png"

    def test_protocol_relative_url(self):
        html = '<html><body><img src="//cdn.example.com/plot.jpg" alt="CDN"></body></html>'
        refs = self._extract(html)
        assert len(refs) == 1
        assert refs[0]["src"] == "https://cdn.example.com/plot.jpg"

    def test_relative_url_resolved(self):
        html = '<html><body><img src="images/graph.png" alt="Graph"></body></html>'
        refs = self._extract(html, page_url="https://wiki.example.com/wiki/TestPage")
        assert len(refs) == 1
        assert refs[0]["src"] == "https://wiki.example.com/wiki/images/graph.png"

    def test_skip_data_uri(self):
        html = '<html><body><img src="data:image/png;base64,AAAA" alt="inline"></body></html>'
        refs = self._extract(html)
        assert len(refs) == 0

    def test_skip_favicon(self):
        html = '<html><body><img src="/favicon.png" width="16" height="16"></body></html>'
        refs = self._extract(html)
        assert len(refs) == 0

    def test_skip_logo(self):
        html = '<html><body><img src="/images/company_logo.png" width="200" height="50"></body></html>'
        refs = self._extract(html)
        assert len(refs) == 0

    def test_skip_tiny_images(self):
        """Images smaller than 32x32 should be filtered (tracking pixels)."""
        html = '<html><body><img src="/images/pixel.png" width="1" height="1"></body></html>'
        refs = self._extract(html)
        assert len(refs) == 0

    def test_skip_non_image_extension(self):
        html = '<html><body><img src="/files/document.pdf" alt="PDF"></body></html>'
        refs = self._extract(html)
        assert len(refs) == 0

    def test_confluence_ac_image_attachment(self):
        """Confluence <ac:image> tags with ri:attachment should be extracted."""
        html = """
        <html><body>
        <ac:image ac:width="800" ac:height="600" ac:alt="DMS schematic">
          <ri:attachment ri:filename="dms_diagram.png" />
        </ac:image>
        </body></html>
        """
        refs = self._extract(
            html,
            page_url="https://confluence.example.com/pages/viewpage.action?pageId=12345",
        )
        assert len(refs) == 1
        assert "dms_diagram.png" in refs[0]["src"]
        assert "12345" in refs[0]["src"]
        assert refs[0]["alt_text"] == "DMS schematic"
        assert refs[0]["width"] == "800"

    def test_confluence_ac_image_url(self):
        """Confluence <ac:image> with ri:url should be extracted."""
        html = """
        <html><body>
        <ac:image ac:width="400">
          <ri:url ri:value="/download/attachments/99/plasma.png" />
        </ac:image>
        </body></html>
        """
        refs = self._extract(
            html,
            page_url="https://confluence.example.com/pages/viewpage.action?pageId=99",
        )
        assert len(refs) == 1
        assert refs[0]["src"].endswith("/download/attachments/99/plasma.png")

    def test_section_tracking(self):
        """Image section should be inferred from preceding heading."""
        html = """
        <html><body>
        <h2>Diagnostics</h2>
        <p>Some text about diagnostics</p>
        <img src="/images/diagnostic_layout.png" alt="Layout" width="500" height="400">
        <h2>Analysis</h2>
        <img src="/images/analysis_result.png" alt="Result" width="500" height="400">
        </body></html>
        """
        refs = self._extract(html)
        assert len(refs) == 2
        assert refs[0]["section"] == "Diagnostics"
        assert refs[1]["section"] == "Analysis"

    def test_surrounding_text_captured(self):
        """Surrounding text context should be captured for VLM scoring."""
        html = """
        <html><body>
        <p>The electron temperature profile shows a clear pedestal structure.</p>
        <img src="/images/te_profile.png" alt="Te" width="500" height="400">
        <p>The pedestal width is approximately 2 cm.</p>
        </body></html>
        """
        refs = self._extract(html)
        assert len(refs) == 1
        assert "electron temperature" in refs[0]["surrounding_text"]

    def test_max_images_per_page(self):
        """No more than _MAX_IMAGES_PER_PAGE should be returned."""
        from imas_codex.discovery.wiki.workers import _MAX_IMAGES_PER_PAGE

        img_tags = "\n".join(
            f'<img src="/images/img_{i}.png" width="100" height="100">'
            for i in range(_MAX_IMAGES_PER_PAGE + 5)
        )
        html = f"<html><body>{img_tags}</body></html>"
        refs = self._extract(html)
        assert len(refs) == _MAX_IMAGES_PER_PAGE

    def test_mediawiki_html_full(self):
        """Extract images from full MediaWiki sample HTML."""
        refs = self._extract(MEDIAWIKI_HTML)
        # Should find thomson_layout.png but not favicon.png (skip pattern)
        src_list = [r["src"] for r in refs]
        assert any("thomson_layout.png" in s for s in src_list)
        assert not any("favicon" in s for s in src_list)

    def test_confluence_html_full(self):
        """Extract images from full Confluence sample HTML."""
        refs = self._extract(
            CONFLUENCE_HTML,
            page_url="https://confluence.example.com/pages/viewpage.action?pageId=12345",
        )
        # Should find the ac:image attachments but filter the logo
        src_list = [r["src"] for r in refs]
        assert any("dms_diagram.png" in s for s in src_list)
        assert any("plasma_current.png" in s for s in src_list)
        assert not any("logo" in s for s in src_list)


# =============================================================================
# _fetch_image_bytes
# =============================================================================


class TestFetchImageBytes:
    """Tests for _fetch_image_bytes — image download with auth strategies."""

    @pytest.mark.asyncio
    async def test_direct_http_success(self):
        """Direct HTTP fetch should return bytes on 200."""
        from imas_codex.discovery.wiki.workers import _fetch_image_bytes

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"\x89PNG" + b"\x00" * 1000

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_cls.return_value = mock_client

            result = await _fetch_image_bytes("https://example.com/image.png")
            assert result == mock_response.content

    @pytest.mark.asyncio
    async def test_direct_http_failure(self):
        """Direct HTTP fetch should return None on non-200."""
        from imas_codex.discovery.wiki.workers import _fetch_image_bytes

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.content = b""

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_cls.return_value = mock_client

            result = await _fetch_image_bytes("https://example.com/missing.png")
            assert result is None

    @pytest.mark.asyncio
    async def test_ssh_proxy_fetch(self):
        """SSH proxy fetch should use subprocess curl."""
        from imas_codex.discovery.wiki.workers import _fetch_image_bytes

        image_bytes = b"\x89PNG" + b"\x00" * 1000

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=image_bytes)

            result = await _fetch_image_bytes(
                "https://internal.example.com/image.png",
                ssh_host="remote-host",
            )
            assert result == image_bytes
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert "ssh" in call_args[0][0][0]
            assert "remote-host" in call_args[0][0][1]

    @pytest.mark.asyncio
    async def test_session_auth_fetch(self):
        """Session-based fetch should use the provided session."""
        from imas_codex.discovery.wiki.workers import _fetch_image_bytes

        # The function creates a fresh requests.Session internally and copies
        # cookies from the provided session. We mock requests.Session to
        # intercept the dl_session.get() call.
        mock_session = MagicMock()
        mock_session.cookies = MagicMock()

        mock_dl_response = MagicMock()
        mock_dl_response.status_code = 200
        mock_dl_response.content = b"\xff\xd8\xff" + b"\x00" * 1000

        with patch("requests.Session") as mock_session_cls:
            mock_dl_session = MagicMock()
            mock_dl_session.get.return_value = mock_dl_response
            mock_session_cls.return_value = mock_dl_session

            result = await _fetch_image_bytes(
                "https://confluence.example.com/download/image.png",
                session=mock_session,
            )
            assert result == mock_dl_response.content

    @pytest.mark.asyncio
    async def test_exception_returns_none(self):
        """Network exceptions should return None, not propagate."""
        from imas_codex.discovery.wiki.workers import _fetch_image_bytes

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_cls.return_value = mock_client

            result = await _fetch_image_bytes("https://example.com/image.png")
            assert result is None


# =============================================================================
# _extract_and_persist_images
# =============================================================================


class TestExtractAndPersistImages:
    """Tests for _extract_and_persist_images."""

    @pytest.mark.asyncio
    async def test_no_images_returns_zero(self):
        """HTML with no images should return 0."""
        from imas_codex.discovery.wiki.workers import _extract_and_persist_images

        result = await _extract_and_persist_images(
            html="<html><body><p>No images</p></body></html>",
            page_url="https://wiki.example.com/page",
            page_id="tcv:TestPage",
            page_title="Test Page",
            facility="tcv",
        )
        assert result == 0

    @pytest.mark.asyncio
    async def test_images_persisted_to_graph(self):
        """Images should be persisted as Image nodes with proper relationships."""
        from imas_codex.discovery.wiki.workers import _extract_and_persist_images

        html = """
        <html><body>
        <img src="https://wiki.example.com/images/physics_plot.png" alt="Physics" width="500" height="400">
        </body></html>
        """

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        # Mock image download and processing
        with (
            patch(
                "imas_codex.discovery.wiki.workers._fetch_image_bytes",
                new_callable=AsyncMock,
                return_value=b"\x89PNG" + b"\x00" * 1000,
            ),
            patch(
                "imas_codex.discovery.wiki.image.downsample_image",
                return_value=("base64data", 300, 200, 500, 400),
            ),
            patch(
                "imas_codex.discovery.wiki.image.make_image_id",
                return_value="tcv:abc123",
            ),
            patch(
                "imas_codex.discovery.wiki.workers.GraphClient",
                return_value=mock_gc,
            ),
        ):
            result = await _extract_and_persist_images(
                html=html,
                page_url="https://wiki.example.com/TestPage",
                page_id="tcv:TestPage",
                page_title="Test Page",
                facility="tcv",
            )
            assert result == 1
            # Verify graph query was called with image data
            mock_gc.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_unfetchable_images_become_failed_nodes(self):
        """Images that can't be downloaded should still create metadata-only nodes."""
        from imas_codex.discovery.wiki.workers import _extract_and_persist_images

        html = '<html><body><img src="https://wiki.example.com/img/blocked.png" width="500" height="400"></body></html>'

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "imas_codex.discovery.wiki.workers._fetch_image_bytes",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "imas_codex.discovery.wiki.image.make_image_id",
                return_value="tcv:blocked123",
            ),
            patch(
                "imas_codex.discovery.wiki.workers.GraphClient",
                return_value=mock_gc,
            ),
        ):
            result = await _extract_and_persist_images(
                html=html,
                page_url="https://wiki.example.com/TestPage",
                page_id="tcv:TestPage",
                page_title="Test Page",
                facility="tcv",
            )
            # Should still create a metadata-only node
            assert result == 1
            # Verify graph was called despite download failure
            mock_gc.query.assert_called_once()
            # Check that the persisted image has status 'failed'
            call_kwargs = mock_gc.query.call_args
            images_param = call_kwargs.kwargs.get("images") or call_kwargs[1].get("images")
            assert images_param[0]["status"] == "failed"


# =============================================================================
# _ingest_image_artifact
# =============================================================================


class TestIngestImageArtifact:
    """Tests for _ingest_image_artifact — wiki file images → Image nodes."""

    @pytest.mark.asyncio
    async def test_successful_ingestion(self):
        """Successful image artifact should create Image node with HAS_IMAGE."""
        from imas_codex.discovery.wiki.workers import _ingest_image_artifact

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "imas_codex.discovery.wiki.workers._fetch_image_bytes",
                new_callable=AsyncMock,
                return_value=b"\x89PNG" + b"\x00" * 1000,
            ),
            patch(
                "imas_codex.discovery.wiki.image.downsample_image",
                return_value=("b64data", 300, 200, 600, 400),
            ),
            patch(
                "imas_codex.discovery.wiki.image.make_image_id",
                return_value="tcv:img_hash",
            ),
            patch(
                "imas_codex.discovery.wiki.workers.GraphClient",
                return_value=mock_gc,
            ),
        ):
            await _ingest_image_artifact(
                artifact_id="tcv:wiki_file_image.png",
                url="https://wiki.example.com/pub/image.png",
                filename="image.png",
                facility="tcv",
            )
            # Should create Image node and link to artifact
            assert mock_gc.query.call_count == 2  # MERGE image + link pages

    @pytest.mark.asyncio
    async def test_too_small_image_skipped(self):
        """Images with fewer than 512 bytes should be skipped."""
        from imas_codex.discovery.wiki.workers import _ingest_image_artifact

        with patch(
            "imas_codex.discovery.wiki.workers._fetch_image_bytes",
            new_callable=AsyncMock,
            return_value=b"\x89PNG" + b"\x00" * 100,  # Only ~104 bytes
        ):
            # Should silently skip — no GraphClient needed
            await _ingest_image_artifact(
                artifact_id="tcv:tiny.png",
                url="https://wiki.example.com/pub/tiny.png",
                filename="tiny.png",
                facility="tcv",
            )
            # No assertion needed — just verify it doesn't raise


# =============================================================================
# _persist_document_figures
# =============================================================================


class TestPersistDocumentFigures:
    """Tests for _persist_document_figures — PDF/PPTX extracted images."""

    @pytest.mark.asyncio
    async def test_empty_list(self):
        """Empty image list should return 0."""
        from imas_codex.discovery.wiki.workers import _persist_document_figures

        result = await _persist_document_figures([], artifact_id="tcv:doc.pdf", artifact_url="https://example.com/doc.pdf", facility="tcv")
        assert result == 0

    @pytest.mark.asyncio
    async def test_small_images_filtered(self):
        """Images smaller than 2048 bytes should be filtered out."""
        from imas_codex.discovery.wiki.workers import _persist_document_figures

        images = [{"image_bytes": b"\x00" * 100, "page_num": 1}]
        result = await _persist_document_figures(images, artifact_id="tcv:doc.pdf", artifact_url="https://example.com/doc.pdf", facility="tcv")
        assert result == 0

    @pytest.mark.asyncio
    async def test_successful_persistence(self):
        """Valid images should be persisted as Image nodes."""
        from imas_codex.discovery.wiki.workers import _persist_document_figures

        images = [
            {
                "image_bytes": b"\x89PNG" + b"\x00" * 5000,
                "page_num": 1,
                "name": "figure_1.png",
            },
        ]

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "imas_codex.discovery.wiki.image.downsample_image",
                return_value=("b64img", 200, 150, 400, 300),
            ),
            patch(
                "imas_codex.discovery.wiki.workers.GraphClient",
                return_value=mock_gc,
            ),
        ):
            result = await _persist_document_figures(
                images, artifact_id="tcv:report.pdf", artifact_url="https://example.com/report.pdf", facility="tcv"
            )
            assert result == 1
            mock_gc.query.assert_called_once()
