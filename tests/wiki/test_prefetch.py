"""Tests for wiki prefetch content utilities."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.discovery.wiki.prefetch import (
    extract_text_from_html,
    fetch_page_content,
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
