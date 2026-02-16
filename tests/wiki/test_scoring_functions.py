"""Tests for wiki scoring functions.

Covers _extract_text_from_bytes, _score_pages_heuristic, _fetch_html
dispatch logic, and _fetch_and_summarize with mocked external deps.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# _extract_text_from_bytes
# =============================================================================


class TestExtractTextFromBytes:
    """Tests for _extract_text_from_bytes — artifact text extraction."""

    def _extract(self, content: bytes, artifact_type: str) -> str:
        from imas_codex.discovery.wiki.scoring import _extract_text_from_bytes

        return _extract_text_from_bytes(content, artifact_type)

    def test_notebook_extraction(self):
        """Jupyter notebook cells should be extracted."""
        nb = {
            "cells": [
                {"cell_type": "markdown", "source": ["# Title\n", "Description"]},
                {"cell_type": "code", "source": ["import numpy as np\n", "x = np.array([1, 2, 3])"]},
                {"cell_type": "markdown", "source": ["## Results"]},
            ]
        }
        result = self._extract(json.dumps(nb).encode(), "notebook")
        assert "# Title" in result
        assert "import numpy" in result
        assert "## Results" in result

    def test_notebook_max_cells(self):
        """Only first 20 cells should be extracted."""
        nb = {
            "cells": [
                {"cell_type": "code", "source": [f"cell_{i}"]} for i in range(30)
            ]
        }
        result = self._extract(json.dumps(nb).encode(), "notebook")
        assert "cell_0" in result
        assert "cell_19" in result
        assert "cell_20" not in result

    def test_json_extraction(self):
        """Valid JSON should be pretty-printed as preview."""
        data = {"key": "value", "nested": {"a": 1, "b": 2}}
        result = self._extract(json.dumps(data).encode(), "json")
        assert "key" in result
        assert "value" in result

    def test_json_large_preview_capped(self):
        """Large JSON should be capped at 5000 chars."""
        data = {"items": [{"id": i, "desc": f"Item {i} " * 100} for i in range(100)]}
        result = self._extract(json.dumps(data).encode(), "json")
        assert len(result) <= 5000

    def test_invalid_json_returns_empty(self):
        """Invalid JSON bytes should return empty string."""
        result = self._extract(b"not json at all {{{", "json")
        assert result == ""

    def test_invalid_notebook_returns_empty(self):
        """Invalid notebook JSON should return empty string."""
        result = self._extract(b"not a notebook", "notebook")
        assert result == ""

    def test_unknown_type_returns_empty(self):
        """Unknown artifact types should return empty string."""
        result = self._extract(b"some bytes", "unknown_type")
        assert result == ""

    def test_pdf_without_header_returns_empty(self):
        """PDF bytes without %PDF header should return empty."""
        result = self._extract(b"not a pdf file", "pdf")
        assert result == ""

    def test_empty_bytes_returns_empty(self):
        """Empty content should return empty string."""
        result = self._extract(b"", "notebook")
        assert result == ""


# =============================================================================
# _score_pages_heuristic
# =============================================================================


class TestScorePagesHeuristic:
    """Tests for _score_pages_heuristic — keyword-based fallback scoring."""

    def _score(self, pages, data_access_patterns=None):
        from imas_codex.discovery.wiki.scoring import _score_pages_heuristic

        return _score_pages_heuristic(pages, data_access_patterns)

    def test_default_score(self):
        """Pages without keywords should get 0.5 default."""
        pages = [{"id": "tcv:SomePage", "title": "Some Page", "summary": "Generic content"}]
        results = self._score(pages)
        assert len(results) == 1
        assert results[0]["score"] == 0.5

    def test_physics_keyword_boost(self):
        """Physics keywords in title should boost score."""
        pages = [
            {"id": "tcv:ThomsonScattering", "title": "Thomson Scattering", "summary": ""},
        ]
        results = self._score(pages)
        assert results[0]["score"] > 0.5

    def test_multiple_physics_keywords(self):
        """Multiple physics keywords should compound the boost."""
        pages = [
            {
                "id": "tcv:EquilibriumDiagnostic",
                "title": "Equilibrium Diagnostic Calibration",
                "summary": "MHD equilibrium measurement using Thomson scattering diagnostic",
            },
        ]
        results = self._score(pages)
        # Multiple keywords: equilibrium, diagnostic, calibration, mhd, thomson
        assert results[0]["score"] > 0.7

    def test_low_value_keyword_penalty(self):
        """Low-value keywords should reduce score."""
        pages = [
            {"id": "tcv:MeetingNotes", "title": "Meeting Notes Draft", "summary": ""},
        ]
        results = self._score(pages)
        assert results[0]["score"] < 0.5

    def test_physics_flag(self):
        """is_physics should be True when score >= 0.6."""
        pages = [
            {"id": "tcv:Plasma", "title": "Plasma Diagnostics", "summary": ""},
            {"id": "tcv:Notes", "title": "Personal Notes", "summary": ""},
        ]
        results = self._score(pages)
        plasma_result = next(r for r in results if r["id"] == "tcv:Plasma")
        notes_result = next(r for r in results if r["id"] == "tcv:Notes")
        assert plasma_result["is_physics"] is True
        assert notes_result["is_physics"] is False

    def test_facility_keywords_boost(self):
        """Facility-specific keywords from data_access_patterns should boost."""
        pages = [
            {"id": "tcv:MdsValue", "title": "Using MdsValue for data access", "summary": ""},
        ]
        patterns = {
            "primary_method": "mdsplus",
            "key_tools": ["MdsOpen", "MdsValue", "TdiExecute"],
            "code_import_patterns": ["import ppf"],
        }
        results = self._score(pages, data_access_patterns=patterns)
        assert results[0]["score"] > 0.5
        assert "facility data access" in results[0]["reasoning"]

    def test_summary_keyword_matching(self):
        """Keywords in summary (not just title) should be detected."""
        pages = [
            {
                "id": "tcv:Report",
                "title": "Annual Report",
                "summary": "This report covers equilibrium reconstruction results",
            },
        ]
        results = self._score(pages)
        assert results[0]["score"] > 0.5

    def test_score_clamping(self):
        """Scores should be clamped to [0.0, 1.0]."""
        # Many physics keywords
        pages = [
            {
                "id": "tcv:AllKeywords",
                "title": "Thomson Equilibrium MHD Plasma Diagnostic Calibration Signal",
                "summary": "liuqe node measurement",
            },
        ]
        results = self._score(pages)
        assert results[0]["score"] <= 1.0

        # Many low-value keywords
        pages = [
            {
                "id": "tcv:AllBad",
                "title": "Meeting Workshop Todo Draft Notes Test Sandbox Personal",
                "summary": "",
            },
        ]
        results = self._score(pages)
        assert results[0]["score"] >= 0.0

    def test_batch_scoring(self):
        """Multiple pages should be scored independently."""
        pages = [
            {"id": "tcv:A", "title": "Thomson Scattering", "summary": ""},
            {"id": "tcv:B", "title": "Meeting Notes", "summary": ""},
            {"id": "tcv:C", "title": "Generic Page", "summary": ""},
        ]
        results = self._score(pages)
        assert len(results) == 3
        # Verify IDs preserved
        assert {r["id"] for r in results} == {"tcv:A", "tcv:B", "tcv:C"}


# =============================================================================
# _fetch_html dispatch
# =============================================================================


class TestFetchHtmlDispatch:
    """Tests for _fetch_html — authentication-aware content fetching."""

    @pytest.mark.asyncio
    async def test_confluence_dispatch(self):
        """auth_type='session' with confluence_client should use Confluence REST API."""
        from imas_codex.discovery.wiki.scoring import _fetch_html

        mock_confluence = MagicMock()
        mock_page = MagicMock()
        mock_page.content_html = "<html>Confluence content</html>"
        mock_confluence.get_page_content = MagicMock(return_value=mock_page)

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_page

            result = await _fetch_html(
                url="https://confluence.example.com/pages/viewpage.action?pageId=12345",
                ssh_host=None,
                auth_type="session",
                confluence_client=mock_confluence,
            )
            assert "Confluence content" in result

    @pytest.mark.asyncio
    async def test_ssh_dispatch(self):
        """No auth + ssh_host should use SSH curl."""
        from imas_codex.discovery.wiki.scoring import _fetch_html

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=b"<html>SSH content</html>",
            )

            result = await _fetch_html(
                url="https://internal.example.com/wiki/Page",
                ssh_host="remote-host",
                auth_type=None,
            )
            assert "SSH content" in result

    @pytest.mark.asyncio
    async def test_keycloak_dispatch(self):
        """auth_type='keycloak' should use keycloak_client."""
        from imas_codex.discovery.wiki.scoring import _fetch_html

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Keycloak content</html>"
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await _fetch_html(
            url="https://wiki.example.com/page",
            ssh_host=None,
            auth_type="keycloak",
            keycloak_client=mock_client,
        )
        assert "Keycloak content" in result

    @pytest.mark.asyncio
    async def test_basic_auth_dispatch(self):
        """auth_type='basic' should use basic_auth_client."""
        from imas_codex.discovery.wiki.scoring import _fetch_html

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Basic auth content</html>"
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await _fetch_html(
            url="https://wiki.example.com/page",
            ssh_host=None,
            auth_type="basic",
            basic_auth_client=mock_client,
        )
        assert "Basic auth content" in result

    @pytest.mark.asyncio
    async def test_direct_http_fallback(self):
        """No auth, no SSH should use direct HTTP fetch."""
        from imas_codex.discovery.wiki.scoring import _fetch_html

        with patch(
            "imas_codex.discovery.wiki.prefetch.fetch_page_content",
            new_callable=AsyncMock,
            return_value=("<html>Direct content</html>", None),
        ):
            result = await _fetch_html(
                url="https://public.example.com/page.html",
                ssh_host=None,
                auth_type=None,
            )
            assert "Direct content" in result

    @pytest.mark.asyncio
    async def test_ssh_fetch_failure(self):
        """SSH fetch failure should return empty string."""
        from imas_codex.discovery.wiki.scoring import _fetch_html

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout=b"")

            result = await _fetch_html(
                url="https://internal.example.com/page",
                ssh_host="remote-host",
                auth_type=None,
            )
            assert result == ""

    @pytest.mark.asyncio
    async def test_keycloak_without_client(self):
        """Keycloak auth without client should return empty string."""
        from imas_codex.discovery.wiki.scoring import _fetch_html

        result = await _fetch_html(
            url="https://wiki.example.com/page",
            ssh_host=None,
            auth_type="keycloak",
            keycloak_client=None,
        )
        assert result == ""


# =============================================================================
# _fetch_and_summarize
# =============================================================================


class TestFetchAndSummarize:
    """Tests for _fetch_and_summarize — content preview extraction."""

    @pytest.mark.asyncio
    async def test_extracts_text_from_html(self):
        """Should extract clean text from HTML content."""
        from imas_codex.discovery.wiki.scoring import _fetch_and_summarize

        with patch(
            "imas_codex.discovery.wiki.scoring._fetch_html",
            new_callable=AsyncMock,
            return_value="<html><body><p>Electron temperature profile measured by Thomson scattering.</p></body></html>",
        ):
            result = await _fetch_and_summarize(
                url="https://wiki.example.com/page",
                ssh_host=None,
                max_chars=1000,
            )
            assert "Electron temperature" in result

    @pytest.mark.asyncio
    async def test_empty_html_returns_empty(self):
        """Empty HTML should return empty string."""
        from imas_codex.discovery.wiki.scoring import _fetch_and_summarize

        with patch(
            "imas_codex.discovery.wiki.scoring._fetch_html",
            new_callable=AsyncMock,
            return_value="",
        ):
            result = await _fetch_and_summarize(
                url="https://wiki.example.com/page",
                ssh_host=None,
            )
            assert result == ""

    @pytest.mark.asyncio
    async def test_max_chars_respected(self):
        """Output should be limited to max_chars."""
        from imas_codex.discovery.wiki.scoring import _fetch_and_summarize

        long_content = "<html><body>" + "<p>A paragraph of text. </p>" * 100 + "</body></html>"

        with patch(
            "imas_codex.discovery.wiki.scoring._fetch_html",
            new_callable=AsyncMock,
            return_value=long_content,
        ):
            result = await _fetch_and_summarize(
                url="https://wiki.example.com/page",
                ssh_host=None,
                max_chars=200,
            )
            assert len(result) <= 200
