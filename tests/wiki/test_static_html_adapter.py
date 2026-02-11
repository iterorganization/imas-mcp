"""Tests for StaticHtmlAdapter artifact discovery and shared utilities."""

from unittest.mock import patch

import pytest

from imas_codex.discovery.wiki.adapters import (
    DiscoveredArtifact,
    DiscoveredPage,
    StaticHtmlAdapter,
    _get_artifact_type_from_filename,
)

# ---------------------------------------------------------------------------
# _get_artifact_type_from_filename
# ---------------------------------------------------------------------------


class TestGetArtifactTypeFromFilename:
    def test_pdf(self):
        assert _get_artifact_type_from_filename("manual.pdf") == "pdf"
        assert _get_artifact_type_from_filename("MANUAL.PDF") == "pdf"

    def test_document(self):
        assert _get_artifact_type_from_filename("report.doc") == "document"
        assert _get_artifact_type_from_filename("report.docx") == "document"

    def test_presentation(self):
        assert _get_artifact_type_from_filename("slides.ppt") == "presentation"
        assert _get_artifact_type_from_filename("slides.pptx") == "presentation"

    def test_spreadsheet(self):
        assert _get_artifact_type_from_filename("data.xls") == "spreadsheet"
        assert _get_artifact_type_from_filename("data.xlsx") == "spreadsheet"

    def test_notebook(self):
        assert _get_artifact_type_from_filename("analysis.ipynb") == "notebook"

    def test_data_files(self):
        assert _get_artifact_type_from_filename("output.h5") == "data"
        assert _get_artifact_type_from_filename("output.hdf5") == "data"
        assert _get_artifact_type_from_filename("output.mat") == "data"

    def test_unknown_defaults_to_document(self):
        assert _get_artifact_type_from_filename("readme.txt") == "document"


# ---------------------------------------------------------------------------
# StaticHtmlAdapter.bulk_discover_artifacts
# ---------------------------------------------------------------------------


class TestStaticHtmlAdapterArtifactDiscovery:
    def _make_adapter(self, **kwargs):
        return StaticHtmlAdapter(
            base_url="https://example.org/docs",
            **kwargs,
        )

    @patch("imas_codex.discovery.wiki.adapters._fetch_html")
    def test_discovers_artifacts_from_pages(self, mock_fetch):
        """Artifacts linked from discovered pages are collected."""
        adapter = self._make_adapter()

        # Mock bulk_discover_pages to return 2 pages
        with patch.object(adapter, "bulk_discover_pages") as mock_pages:
            mock_pages.return_value = [
                DiscoveredPage(name="PageA", url="https://example.org/docs/a.html"),
                DiscoveredPage(name="PageB", url="https://example.org/docs/b.html"),
            ]

            # PageA has a PDF link, PageB has an xlsx link
            mock_fetch.side_effect = [
                '<html><a href="manual.pdf">Manual</a></html>',
                '<html><a href="/docs/data/results.xlsx">Results</a></html>',
            ]

            artifacts = adapter.bulk_discover_artifacts(
                "test", "https://example.org/docs"
            )

        assert len(artifacts) == 2
        assert isinstance(artifacts[0], DiscoveredArtifact)

        filenames = {a.filename for a in artifacts}
        assert "manual.pdf" in filenames
        assert "results.xlsx" in filenames

    @patch("imas_codex.discovery.wiki.adapters._fetch_html")
    def test_deduplicates_artifacts(self, mock_fetch):
        """Same artifact linked from multiple pages is only counted once."""
        adapter = self._make_adapter()

        with patch.object(adapter, "bulk_discover_pages") as mock_pages:
            mock_pages.return_value = [
                DiscoveredPage(name="PageA", url="https://example.org/docs/a.html"),
                DiscoveredPage(name="PageB", url="https://example.org/docs/b.html"),
            ]

            # Both pages link to the same PDF
            mock_fetch.side_effect = [
                '<html><a href="shared.pdf">PDF</a></html>',
                '<html><a href="shared.pdf">PDF</a></html>',
            ]

            artifacts = adapter.bulk_discover_artifacts(
                "test", "https://example.org/docs"
            )

        # Should deduplicate — same resolved URL
        assert len(artifacts) == 1
        assert artifacts[0].filename == "shared.pdf"

    @patch("imas_codex.discovery.wiki.adapters._fetch_html")
    def test_ignores_non_artifact_links(self, mock_fetch):
        """Links to HTML pages and images are ignored."""
        adapter = self._make_adapter()

        with patch.object(adapter, "bulk_discover_pages") as mock_pages:
            mock_pages.return_value = [
                DiscoveredPage(name="PageA", url="https://example.org/docs/a.html"),
            ]

            mock_fetch.return_value = """
            <html>
            <a href="other.html">Other page</a>
            <a href="logo.png">Logo</a>
            <a href="photo.jpg">Photo</a>
            <a href="actual.pdf">PDF</a>
            </html>
            """

            artifacts = adapter.bulk_discover_artifacts(
                "test", "https://example.org/docs"
            )

        assert len(artifacts) == 1
        assert artifacts[0].filename == "actual.pdf"

    @patch("imas_codex.discovery.wiki.adapters._fetch_html")
    def test_resolves_relative_urls(self, mock_fetch):
        """Relative URLs are resolved against the page URL."""
        adapter = self._make_adapter()

        with patch.object(adapter, "bulk_discover_pages") as mock_pages:
            mock_pages.return_value = [
                DiscoveredPage(
                    name="Sub",
                    url="https://example.org/docs/sub/page.html",
                ),
            ]

            mock_fetch.return_value = (
                '<html><a href="../files/report.pdf">Report</a></html>'
            )

            artifacts = adapter.bulk_discover_artifacts(
                "test", "https://example.org/docs"
            )

        assert len(artifacts) == 1
        assert artifacts[0].url == "https://example.org/docs/files/report.pdf"

    @patch("imas_codex.discovery.wiki.adapters._fetch_html")
    def test_resolves_absolute_path_urls(self, mock_fetch):
        """Absolute path URLs (/path/to/file.pdf) are resolved against the origin."""
        adapter = self._make_adapter()

        with patch.object(adapter, "bulk_discover_pages") as mock_pages:
            mock_pages.return_value = [
                DiscoveredPage(name="P", url="https://example.org/docs/p.html"),
            ]

            mock_fetch.return_value = (
                '<html><a href="/downloads/data.xlsx">Data</a></html>'
            )

            artifacts = adapter.bulk_discover_artifacts(
                "test", "https://example.org/docs"
            )

        assert len(artifacts) == 1
        assert artifacts[0].url == "https://example.org/downloads/data.xlsx"

    def test_no_base_url_returns_empty(self):
        """No base URL configured returns empty list."""
        adapter = StaticHtmlAdapter()
        assert adapter.bulk_discover_artifacts("test", "") == []

    @patch("imas_codex.discovery.wiki.adapters._fetch_html")
    def test_unreachable_pages_skipped(self, mock_fetch):
        """Pages that fail to fetch are gracefully skipped."""
        adapter = self._make_adapter()

        with patch.object(adapter, "bulk_discover_pages") as mock_pages:
            mock_pages.return_value = [
                DiscoveredPage(name="Good", url="https://example.org/docs/good.html"),
                DiscoveredPage(name="Bad", url="https://example.org/docs/bad.html"),
            ]

            # First page works, second returns None
            mock_fetch.side_effect = [
                '<html><a href="file.pdf">File</a></html>',
                None,
            ]

            artifacts = adapter.bulk_discover_artifacts(
                "test", "https://example.org/docs"
            )

        assert len(artifacts) == 1
        assert artifacts[0].filename == "file.pdf"

    @patch("imas_codex.discovery.wiki.adapters._fetch_html")
    def test_progress_callback(self, mock_fetch):
        """Progress callback is invoked during discovery."""
        adapter = self._make_adapter()
        progress_msgs = []

        with patch.object(adapter, "bulk_discover_pages") as mock_pages:
            mock_pages.return_value = [
                DiscoveredPage(name="P", url="https://example.org/docs/p.html"),
            ]
            mock_fetch.return_value = '<html><a href="f.pdf">F</a></html>'

            adapter.bulk_discover_artifacts(
                "test",
                "https://example.org/docs",
                on_progress=lambda msg, _: progress_msgs.append(msg),
            )

        # Should have at least scanning and discovered messages
        assert any("scanning" in m for m in progress_msgs)
        assert any("discovered" in m for m in progress_msgs)

    @patch("imas_codex.discovery.wiki.adapters._fetch_html")
    def test_artifact_type_mapping(self, mock_fetch):
        """All supported artifact types are correctly identified."""
        adapter = self._make_adapter()

        with patch.object(adapter, "bulk_discover_pages") as mock_pages:
            mock_pages.return_value = [
                DiscoveredPage(name="P", url="https://example.org/docs/p.html"),
            ]

            mock_fetch.return_value = """
            <html>
            <a href="a.pdf">pdf</a>
            <a href="b.docx">docx</a>
            <a href="c.pptx">pptx</a>
            <a href="d.xlsx">xlsx</a>
            <a href="e.ipynb">nb</a>
            <a href="f.h5">h5</a>
            </html>
            """

            artifacts = adapter.bulk_discover_artifacts(
                "test", "https://example.org/docs"
            )

        type_map = {a.filename: a.artifact_type for a in artifacts}
        assert type_map["a.pdf"] == "pdf"
        assert type_map["b.docx"] == "document"
        assert type_map["c.pptx"] == "presentation"
        assert type_map["d.xlsx"] == "spreadsheet"
        assert type_map["e.ipynb"] == "notebook"
        assert type_map["f.h5"] == "data"


# ---------------------------------------------------------------------------
# StaticHtmlAdapter BFS depth limit and max_pages safety net
# ---------------------------------------------------------------------------


class TestStaticHtmlAdapterBFSLimits:
    @patch("imas_codex.discovery.wiki.adapters._fetch_html")
    def test_bfs_respects_depth_limit(self, mock_fetch):
        """BFS crawl does not follow links beyond max_depth."""
        adapter = StaticHtmlAdapter(
            base_url="https://example.org",
            max_depth=1,
        )

        call_count = 0

        def fake_fetch(url, **kwargs):
            nonlocal call_count
            call_count += 1
            # Each page links to a deeper page
            return f"""<html>
            <a href="level{call_count}.html">Deeper</a>
            </html>"""

        mock_fetch.side_effect = fake_fetch

        pages = adapter.bulk_discover_pages("test", "https://example.org")

        # Depth 0: portal candidates (index.html, index-en.html, base)
        # Depth 1: pages discovered from depth-0 pages
        # Depth 2+: should NOT be followed
        # With max_depth=1, we fetch depth-0 seeds and their direct children,
        # but children's links are not followed
        assert len(pages) <= 10  # bounded, not unbounded

    @patch("imas_codex.discovery.wiki.adapters._fetch_html")
    def test_depth_0_only_fetches_seeds(self, mock_fetch):
        """max_depth=0 fetches only the seed URLs, no link following."""
        adapter = StaticHtmlAdapter(
            base_url="https://example.org",
            max_depth=0,
        )

        mock_fetch.return_value = """<html>
        <a href="page1.html">Page 1</a>
        <a href="page2.html">Page 2</a>
        <a href="page3.html">Page 3</a>
        </html>"""

        pages = adapter.bulk_discover_pages("test", "https://example.org")

        # Only seed URLs (index.html, index-en.html, base) should be fetched.
        # Links on those pages should NOT be followed since depth >= max_depth.
        assert len(pages) <= 3  # at most the 3 seed URLs
        # page1, page2, page3 should NOT appear
        page_names = {p.name for p in pages}
        assert "page1.html" not in page_names
        assert "page2.html" not in page_names

    @patch("imas_codex.discovery.wiki.adapters._fetch_html")
    def test_max_pages_still_acts_as_safety_net(self, mock_fetch):
        """max_pages stops crawl even within depth limit."""
        adapter = StaticHtmlAdapter(
            base_url="https://example.org",
            max_depth=10,  # Very deep
            max_pages=3,  # But strict page limit
        )

        call_count = 0

        def fake_fetch(url, **kwargs):
            nonlocal call_count
            call_count += 1
            return f"""<html>
            <a href="page{call_count * 2}.html">Next</a>
            <a href="page{call_count * 2 + 1}.html">Another</a>
            </html>"""

        mock_fetch.side_effect = fake_fetch

        pages = adapter.bulk_discover_pages("test", "https://example.org")
        assert len(pages) <= 3

    def test_default_max_depth_is_3(self):
        """Default max_depth is 3."""
        adapter = StaticHtmlAdapter(base_url="https://example.org")
        assert adapter._max_depth == 3

    def test_default_max_pages_is_500(self):
        """Default max_pages is 500 (safety net)."""
        adapter = StaticHtmlAdapter(base_url="https://example.org")
        assert adapter._max_pages == 500

    def test_custom_max_depth(self):
        """Custom max_depth is respected."""
        adapter = StaticHtmlAdapter(base_url="https://example.org", max_depth=5)
        assert adapter._max_depth == 5

    def test_max_depth_zero(self):
        """max_depth=0 is valid (seeds only)."""
        adapter = StaticHtmlAdapter(base_url="https://example.org", max_depth=0)
        assert adapter._max_depth == 0
