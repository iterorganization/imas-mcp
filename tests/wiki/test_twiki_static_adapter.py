"""Tests for TWikiStaticAdapter."""

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.discovery.wiki.adapters import (
    DiscoveredPage,
    TWikiStaticAdapter,
    get_adapter,
)


class TestTWikiStaticAdapter:
    """Test TWikiStaticAdapter class."""

    def test_init_with_base_url(self):
        """Test adapter initializes with base URL."""
        adapter = TWikiStaticAdapter(base_url="https://example.org/twiki_html")
        assert adapter._base_url == "https://example.org/twiki_html"

    def test_init_without_base_url(self):
        """Test adapter initializes without base URL."""
        adapter = TWikiStaticAdapter()
        assert adapter._base_url is None

    def test_init_with_ssh_host(self):
        """Test adapter initializes with SSH host."""
        adapter = TWikiStaticAdapter(
            base_url="https://example.org/twiki_html", ssh_host="myhost"
        )
        assert adapter._ssh_host == "myhost"

    def test_site_type(self):
        """Test adapter has correct site type."""
        adapter = TWikiStaticAdapter()
        assert adapter.site_type == "twiki_static"


class TestTWikiStaticAdapterDiscovery:
    """Test TWikiStaticAdapter page discovery."""

    @patch("httpx.Client")
    def test_bulk_discover_pages_parses_topic_list(self, mock_client_class):
        """Test that bulk_discover_pages parses WebTopicList.html correctly."""
        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.text = """
        <html>
        <body>
        <ul>
            <li><a href="TopicOne.html">TopicOne</a></li>
            <li><a href="TopicTwo.html">TopicTwo</a></li>
            <li><a href="WebHome.html">WebHome</a></li>
            <li><a href="WebTopicList.html">WebTopicList</a></li>
        </ul>
        </body>
        </html>
        """
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.get.return_value = mock_response

        mock_client_class.return_value = mock_client_instance

        adapter = TWikiStaticAdapter(base_url="https://example.org/twiki_html")
        pages = adapter.bulk_discover_pages(
            "test_facility", "https://example.org/twiki_html"
        )

        # Should find 2 pages (Web* pages are skipped)
        assert len(pages) == 2
        assert isinstance(pages[0], DiscoveredPage)

        # Check page names
        names = [p.name for p in pages]
        assert "TopicOne" in names
        assert "TopicTwo" in names
        assert "WebHome" not in names  # Filtered out
        assert "WebTopicList" not in names  # Filtered out

    def test_bulk_discover_pages_handles_no_base_url(self):
        """Test that bulk_discover_pages returns empty list without base URL."""
        adapter = TWikiStaticAdapter()
        pages = adapter.bulk_discover_pages("test_facility", "")

        assert len(pages) == 0

    @patch("imas_codex.discovery.wiki.adapters._fetch_html_via_ssh")
    def test_bulk_discover_pages_via_ssh(self, mock_ssh_fetch):
        """Test that bulk_discover_pages uses SSH proxy when ssh_host is set."""
        mock_ssh_fetch.return_value = """
        <html>
        <body>
        <ul>
            <li><a href="TopicAlpha.html">TopicAlpha</a></li>
            <li><a href="TopicBeta.html">TopicBeta</a></li>
            <li><a href="WebHome.html">WebHome</a></li>
        </ul>
        </body>
        </html>
        """

        adapter = TWikiStaticAdapter(
            base_url="https://example.org/twiki_html", ssh_host="myhost"
        )
        pages = adapter.bulk_discover_pages(
            "test_facility", "https://example.org/twiki_html"
        )

        # Should call SSH fetch, not httpx
        mock_ssh_fetch.assert_called_once_with(
            "https://example.org/twiki_html/WebTopicList.html", "myhost"
        )

        # Should find 2 pages (Web* pages are skipped)
        assert len(pages) == 2
        names = [p.name for p in pages]
        assert "TopicAlpha" in names
        assert "TopicBeta" in names
        assert "WebHome" not in names

    @patch("imas_codex.discovery.wiki.adapters._fetch_html_via_ssh")
    def test_bulk_discover_pages_ssh_failure(self, mock_ssh_fetch):
        """Test that SSH failure returns empty list."""
        mock_ssh_fetch.return_value = None

        adapter = TWikiStaticAdapter(
            base_url="https://example.org/twiki_html", ssh_host="myhost"
        )
        pages = adapter.bulk_discover_pages(
            "test_facility", "https://example.org/twiki_html"
        )

        assert len(pages) == 0


class TestGetAdapter:
    """Test get_adapter factory function."""

    def test_get_adapter_twiki_static(self):
        """Test get_adapter returns TWikiStaticAdapter for twiki_static."""
        adapter = get_adapter("twiki_static", base_url="https://example.org")
        assert isinstance(adapter, TWikiStaticAdapter)

    def test_get_adapter_static_html(self):
        """Test get_adapter returns StaticHtmlAdapter for static_html."""
        from imas_codex.discovery.wiki.adapters import StaticHtmlAdapter

        adapter = get_adapter("static_html", base_url="https://example.org")
        assert isinstance(adapter, StaticHtmlAdapter)

    def test_get_adapter_mediawiki(self):
        """Test get_adapter still works for mediawiki."""
        from imas_codex.discovery.wiki.adapters import MediaWikiAdapter

        adapter = get_adapter("mediawiki", ssh_host="test")
        assert isinstance(adapter, MediaWikiAdapter)

    def test_get_adapter_unknown_raises(self):
        """Test get_adapter raises ValueError for unknown type."""
        with pytest.raises(ValueError, match="Unknown site type"):
            get_adapter("unknown_type")
