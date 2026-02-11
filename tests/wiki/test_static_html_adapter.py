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
# StaticHtmlAdapter BFS max_pages limit
# ---------------------------------------------------------------------------


class TestStaticHtmlAdapterMaxPages:
    @patch("imas_codex.discovery.wiki.adapters._fetch_html")
    def test_bfs_respects_max_pages_limit(self, mock_fetch):
        """BFS crawl stops after max_pages is reached."""
        adapter = StaticHtmlAdapter(
            base_url="https://example.org",
            max_pages=3,
        )

        # Each page links to two more, creating exponential growth
        def fake_fetch(url, **kwargs):
            # Return HTML with links to other pages
            n = len(mock_fetch.call_args_list)
            return f"""<html>
            <a href="page{n * 2}.html">Next</a>
            <a href="page{n * 2 + 1}.html">Another</a>
            </html>"""

        mock_fetch.side_effect = fake_fetch

        pages = adapter.bulk_discover_pages("test", "https://example.org")

        # Should stop at max_pages=3
        assert len(pages) <= 3

    def test_default_max_pages_is_500(self):
        """Default max_pages is 500."""
        adapter = StaticHtmlAdapter(base_url="https://example.org")
        assert adapter._max_pages == 500

    def test_custom_max_pages(self):
        """Custom max_pages is respected."""
        adapter = StaticHtmlAdapter(base_url="https://example.org", max_pages=100)
        assert adapter._max_pages == 100
