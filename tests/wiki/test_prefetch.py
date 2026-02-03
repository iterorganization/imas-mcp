"""Tests for wiki prefetch module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.discovery.wiki.prefetch import (
    extract_text_from_html,
    fetch_page_content,
    prefetch_pages,
    summarize_pages_batch,
)


class TestFetchPageContent:
    @pytest.mark.asyncio
    async def test_fetch_success(self):
        """Test successful page fetch."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.text = "<html><body>Test content</body></html>"
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            content, error = await fetch_page_content("http://example.com")

            assert content == "<html><body>Test content</body></html>"
            assert error is None

    @pytest.mark.asyncio
    async def test_fetch_timeout(self):
        """Test timeout handling."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            content, error = await fetch_page_content("http://example.com")

            assert content is None
            assert error == "Timeout"

    @pytest.mark.asyncio
    async def test_fetch_auth_required(self):
        """Test auth required (401) handling."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError(
                    "Auth required",
                    request=MagicMock(),
                    response=mock_response,
                )
            )

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Auth required",
                    request=MagicMock(),
                    response=mock_response,
                )
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            content, error = await fetch_page_content("http://example.com")

            assert content is None
            assert error == "Auth required"


class TestExtractText:
    def test_extract_basic_text(self):
        """Test basic text extraction."""
        html = "<html><body><p>Hello world</p></body></html>"
        text = extract_text_from_html(html)
        assert "Hello world" in text

    def test_remove_scripts(self):
        """Test script removal."""
        html = "<html><body><script>alert('test')</script><p>Content</p></body></html>"
        text = extract_text_from_html(html)
        assert "alert" not in text
        assert "Content" in text

    def test_remove_styles(self):
        """Test style tag removal."""
        html = "<html><body><style>.red { color: red; }</style><p>Content</p></body></html>"
        text = extract_text_from_html(html)
        assert "color" not in text
        assert "Content" in text

    def test_truncate_long_text(self):
        """Test truncation of long text."""
        html = f"<html><body><p>{'x' * 3000}</p></body></html>"
        text = extract_text_from_html(html, max_chars=2000)
        assert len(text) <= 2000

    def test_preserve_paragraph_structure(self):
        """Test that paragraph structure is preserved."""
        html = "<html><body><p>First</p><p>Second</p></body></html>"
        text = extract_text_from_html(html)
        assert "First" in text
        assert "Second" in text


class TestSummarizeBatch:
    @pytest.mark.asyncio
    async def test_summarize_single_page(self):
        """Test summarizing a single page."""
        pages = [
            {
                "id": "test:page1",
                "title": "Thomson Scattering",
                "preview_text": "Thomson scattering diagnostic measures electron temperature and density profiles.",
            }
        ]

        with patch("imas_codex.discovery.wiki.prefetch.get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_response = MagicMock()
            # Use return_value to mock str() behavior
            mock_response.return_value = (
                "1. Diagnostic for measuring electron temperature and density."
            )
            mock_llm.acomplete = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            summaries = await summarize_pages_batch(pages)

            assert len(summaries) == 1
            assert len(summaries[0]) <= 300

    @pytest.mark.asyncio
    async def test_summarize_batch_multiple(self):
        """Test summarizing multiple pages in batch."""
        pages = [
            {
                "id": "test:page1",
                "title": "Page 1",
                "preview_text": "Content 1",
            },
            {
                "id": "test:page2",
                "title": "Page 2",
                "preview_text": "Content 2",
            },
        ]

        with patch("imas_codex.discovery.wiki.prefetch.get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_response = MagicMock()
            # Use return_value to mock str() behavior
            mock_response.return_value = "1. Summary 1\n2. Summary 2"
            mock_llm.acomplete = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            summaries = await summarize_pages_batch(pages)

            assert len(summaries) == 2


class TestPrefetchPages:
    @pytest.mark.asyncio
    async def test_prefetch_no_pages(self):
        """Test prefetch when no pages need processing."""
        with patch("imas_codex.discovery.wiki.prefetch.GraphClient") as mock_gc_class:
            mock_gc = MagicMock()
            mock_gc.query = MagicMock(return_value=[])
            mock_gc_class.return_value = mock_gc

            stats = await prefetch_pages("tcv")

            assert stats["fetched"] == 0
            assert stats["summarized"] == 0

    @pytest.mark.asyncio
    async def test_prefetch_with_pages(self):
        """Test prefetch with actual pages."""
        pages = [
            {"id": "test:page1", "url": "http://example.com/page1", "title": "Page 1"}
        ]

        with (
            patch("imas_codex.discovery.wiki.prefetch.GraphClient") as mock_gc_class,
            patch(
                "imas_codex.discovery.wiki.prefetch.fetch_page_content"
            ) as mock_fetch,
            patch(
                "imas_codex.discovery.wiki.prefetch.summarize_pages_batch"
            ) as mock_summarize,
        ):
            mock_gc = MagicMock()
            mock_gc.query = MagicMock(
                side_effect=[pages, None]
            )  # First for query, then for updates
            mock_gc_class.return_value = mock_gc

            # Mock fetch to return HTML content
            mock_fetch.return_value = ("<html><body>Test</body></html>", None)

            # Mock summarize
            mock_summarize.return_value = ["Test summary"]

            stats = await prefetch_pages("tcv", max_pages=1)

            assert stats["fetched"] == 1
