"""Tests for wiki adapter functions.

Covers get_adapter factory, _get_artifact_type_from_filename,
MediaWikiAdapter._extract_page_links, and adapter construction.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# =============================================================================
# _get_artifact_type_from_filename
# =============================================================================


class TestGetArtifactTypeFromFilename:
    """Tests for _get_artifact_type_from_filename utility."""

    def _get_type(self, filename: str) -> str:
        from imas_codex.discovery.wiki.adapters import _get_artifact_type_from_filename

        return _get_artifact_type_from_filename(filename)

    def test_pdf(self):
        assert self._get_type("paper.pdf") == "pdf"
        assert self._get_type("REPORT.PDF") == "pdf"

    def test_documents(self):
        assert self._get_type("report.doc") == "document"
        assert self._get_type("report.docx") == "document"

    def test_presentations(self):
        assert self._get_type("slides.ppt") == "presentation"
        assert self._get_type("slides.pptx") == "presentation"

    def test_spreadsheets(self):
        assert self._get_type("data.xls") == "spreadsheet"
        assert self._get_type("data.xlsx") == "spreadsheet"

    def test_images(self):
        assert self._get_type("photo.png") == "image"
        assert self._get_type("photo.jpg") == "image"
        assert self._get_type("photo.jpeg") == "image"
        assert self._get_type("anim.gif") == "image"
        assert self._get_type("vector.svg") == "image"
        assert self._get_type("modern.webp") == "image"

    def test_notebooks(self):
        assert self._get_type("analysis.ipynb") == "notebook"

    def test_json(self):
        assert self._get_type("config.json") == "json"

    def test_data_files(self):
        assert self._get_type("matrix.h5") == "data"
        assert self._get_type("dataset.hdf5") == "data"
        assert self._get_type("matlab.mat") == "data"

    def test_unknown_defaults_to_document(self):
        assert self._get_type("file.txt") == "document"
        assert self._get_type("binary.bin") == "document"
        assert self._get_type("noext") == "document"

    def test_case_insensitive(self):
        assert self._get_type("PHOTO.PNG") == "image"
        assert self._get_type("Report.PDF") == "pdf"
        assert self._get_type("Data.XLSX") == "spreadsheet"


# =============================================================================
# get_adapter factory
# =============================================================================


class TestGetAdapterFactory:
    """Tests for get_adapter — adapter factory based on site_type."""

    def _get_adapter(self, site_type: str, **kwargs):
        from imas_codex.discovery.wiki.adapters import get_adapter

        return get_adapter(site_type, **kwargs)

    def test_mediawiki(self):
        from imas_codex.discovery.wiki.adapters import MediaWikiAdapter

        adapter = self._get_adapter("mediawiki", ssh_host="remote")
        assert isinstance(adapter, MediaWikiAdapter)
        assert adapter.ssh_host == "remote"

    def test_mediawiki_with_session(self):
        from imas_codex.discovery.wiki.adapters import MediaWikiAdapter

        session = MagicMock()
        adapter = self._get_adapter("mediawiki", session=session)
        assert isinstance(adapter, MediaWikiAdapter)
        assert adapter.session is session

    def test_twiki(self):
        from imas_codex.discovery.wiki.adapters import TWikiAdapter

        adapter = self._get_adapter("twiki", ssh_host="twiki-host")
        assert isinstance(adapter, TWikiAdapter)

    def test_twiki_static(self):
        from imas_codex.discovery.wiki.adapters import TWikiStaticAdapter

        adapter = self._get_adapter(
            "twiki_static", base_url="https://example.com/twiki", ssh_host="host"
        )
        assert isinstance(adapter, TWikiStaticAdapter)

    def test_twiki_raw_requires_ssh_host(self):
        with pytest.raises(ValueError, match="twiki_raw requires ssh_host"):
            self._get_adapter("twiki_raw", data_path="/data")

    def test_twiki_raw_requires_data_path(self):
        with pytest.raises(ValueError, match="twiki_raw requires data_path"):
            self._get_adapter("twiki_raw", ssh_host="host")

    def test_twiki_raw_valid(self):
        from imas_codex.discovery.wiki.adapters import TWikiRawAdapter

        adapter = self._get_adapter("twiki_raw", ssh_host="host", data_path="/data")
        assert isinstance(adapter, TWikiRawAdapter)

    def test_static_html(self):
        from imas_codex.discovery.wiki.adapters import StaticHtmlAdapter

        adapter = self._get_adapter(
            "static_html", base_url="https://example.com", max_depth=5
        )
        assert isinstance(adapter, StaticHtmlAdapter)

    def test_confluence(self):
        from imas_codex.discovery.wiki.adapters import ConfluenceAdapter

        adapter = self._get_adapter(
            "confluence", api_token="token123", space_key="IMASDD"
        )
        assert isinstance(adapter, ConfluenceAdapter)

    def test_unknown_site_type_raises(self):
        with pytest.raises(ValueError, match="Unknown site type"):
            self._get_adapter("wordpress")


# =============================================================================
# MediaWikiAdapter._extract_page_links
# =============================================================================


class TestExtractPageLinks:
    """Tests for MediaWikiAdapter._extract_page_links HTML parsing."""

    def _extract(self, html: str, base_url: str = "https://wiki.example.com/wiki"):
        from imas_codex.discovery.wiki.adapters import MediaWikiAdapter

        adapter = MediaWikiAdapter()
        return adapter._extract_page_links(html, base_url)

    def test_standard_wiki_links(self):
        """Should extract page links from /wiki/Page_Name format."""
        html = """
        <div class="mw-allpages-body">
            <a href="/wiki/Main_Page" title="Main Page">Main Page</a>
            <a href="/wiki/Thomson_Scattering" title="Thomson">Thomson Scattering</a>
            <a href="/wiki/Equilibrium" title="Equilibrium">Equilibrium</a>
        </div>
        """
        pages = self._extract(html)
        names = [p.name for p in pages]
        assert "Main_Page" in names
        assert "Thomson_Scattering" in names
        assert "Equilibrium" in names

    def test_index_php_links(self):
        """Should extract page links from index.php?title format."""
        # The regex in _extract_page_links requires /wiki/ or title= in href
        # and a matching link text. The pattern captures the page name.
        html = """
        <a href="/w/index.php?title=My_Page" title="My Page">My Page</a>
        """
        pages = self._extract(html)
        names = [p.name for p in pages]
        # The regex captures title=<name> from the href
        assert "My_Page" in names

    def test_excludes_special_namespaces(self):
        """Should exclude Special:, File:, Template:, etc."""
        html = """
        <a href="/wiki/Special:Upload" title="Upload">Upload</a>
        <a href="/wiki/File:Image.png" title="File">Image</a>
        <a href="/wiki/Template:Header" title="Template">Header</a>
        <a href="/wiki/Category:Physics" title="Category">Physics</a>
        <a href="/wiki/Good_Page" title="Good Page">Good Page</a>
        """
        pages = self._extract(html)
        names = [p.name for p in pages]
        assert "Good_Page" in names
        assert len(names) == 1  # Only Good_Page

    def test_excludes_allpages_navigation(self):
        """Should exclude AllPages navigation links."""
        html = """
        <a href="/wiki/AllPages" title="All">All Pages</a>
        <a href="/wiki/Valid_Page" title="Valid">Valid Page</a>
        """
        pages = self._extract(html)
        names = [p.name for p in pages]
        assert "Valid_Page" in names
        assert "AllPages" not in names

    def test_url_decode(self):
        """Should decode URL-encoded page names."""
        html = """
        <a href="/wiki/Caf%C3%A9_Page" title="Café Page">Café Page</a>
        """
        pages = self._extract(html)
        names = [p.name for p in pages]
        assert any("Café" in name for name in names)

    def test_empty_html(self):
        """Empty HTML should return no pages."""
        pages = self._extract("")
        assert pages == []


# =============================================================================
# MediaWikiAdapter page discovery dispatch
# =============================================================================


class TestMediaWikiAdapterDispatch:
    """Tests for MediaWikiAdapter discovery method dispatch."""

    def test_no_auth_returns_empty(self):
        """Adapter with no SSH, client, or session returns empty list."""
        from imas_codex.discovery.wiki.adapters import MediaWikiAdapter

        adapter = MediaWikiAdapter()
        pages = adapter.bulk_discover_pages("tcv", "https://wiki.example.com")
        assert pages == []

    def test_session_dispatch(self):
        """Session-based adapter should use API discovery."""
        from imas_codex.discovery.wiki.adapters import MediaWikiAdapter

        session = MagicMock()
        response = MagicMock()
        response.status_code = 200
        response.text = '{"query": {"allpages": [{"title": "TestPage"}]}}'
        session.get.return_value = response

        adapter = MediaWikiAdapter(session=session)
        pages = adapter.bulk_discover_pages("tcv", "https://wiki.example.com")
        assert len(pages) > 0
        assert pages[0].name == "TestPage"


# =============================================================================
# StaticHtmlAdapter construction
# =============================================================================


class TestStaticHtmlAdapterConstruction:
    """Tests for StaticHtmlAdapter defaults."""

    def test_defaults(self):
        from imas_codex.discovery.wiki.adapters import StaticHtmlAdapter

        adapter = StaticHtmlAdapter(base_url="https://example.com")
        assert adapter._max_depth == 3
        assert adapter._max_pages == 500
        assert adapter._exclude_prefixes == []

    def test_custom_config(self):
        from imas_codex.discovery.wiki.adapters import StaticHtmlAdapter

        adapter = StaticHtmlAdapter(
            base_url="https://example.com",
            max_depth=5,
            max_pages=100,
            exclude_prefixes=["/twiki_html"],
        )
        assert adapter._max_depth == 5
        assert adapter._max_pages == 100
        assert adapter._exclude_prefixes == ["/twiki_html"]


# =============================================================================
# DiscoveredPage / DiscoveredArtifact dataclasses
# =============================================================================


class TestDiscoveryDataclasses:
    """Tests for DiscoveredPage and DiscoveredArtifact construction."""

    def test_discovered_page(self):
        from imas_codex.discovery.wiki.adapters import DiscoveredPage

        page = DiscoveredPage(name="TestPage", url="https://wiki.example.com/TestPage")
        assert page.name == "TestPage"
        assert page.url == "https://wiki.example.com/TestPage"
        assert page.namespace is None

    def test_discovered_page_with_namespace(self):
        from imas_codex.discovery.wiki.adapters import DiscoveredPage

        page = DiscoveredPage(name="Test", url=None, namespace="Main")
        assert page.namespace == "Main"

    def test_discovered_artifact(self):
        from imas_codex.discovery.wiki.adapters import DiscoveredArtifact

        artifact = DiscoveredArtifact(
            filename="report.pdf",
            url="https://wiki.example.com/files/report.pdf",
            artifact_type="pdf",
            size_bytes=1024,
            mime_type="application/pdf",
            linked_pages=["MainPage", "Reports"],
        )
        assert artifact.filename == "report.pdf"
        assert artifact.artifact_type == "pdf"
        assert artifact.size_bytes == 1024
        assert len(artifact.linked_pages) == 2

    def test_discovered_artifact_defaults(self):
        from imas_codex.discovery.wiki.adapters import DiscoveredArtifact

        artifact = DiscoveredArtifact(
            filename="file.xlsx",
            url="https://example.com/file.xlsx",
            artifact_type="spreadsheet",
        )
        assert artifact.size_bytes is None
        assert artifact.mime_type is None
        assert artifact.linked_pages == []
