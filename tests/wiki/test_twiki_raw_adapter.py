"""Tests for TWikiRawAdapter and twiki_markup_to_html converter."""

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.discovery.wiki.adapters import (
    DiscoveredPage,
    TWikiRawAdapter,
    fetch_twiki_raw_content,
    get_adapter,
)
from imas_codex.discovery.wiki.pipeline import (
    _strip_twiki_formatting,
    twiki_markup_to_html,
)

# ──────────────────────────────────────────────────────────────────
# TWikiRawAdapter tests
# ──────────────────────────────────────────────────────────────────


class TestTWikiRawAdapterInit:
    """Test TWikiRawAdapter initialization."""

    def test_init_basic(self):
        adapter = TWikiRawAdapter(
            ssh_host="myhost",
            data_path="/twiki/data/Main",
        )
        assert adapter._ssh_host == "myhost"
        assert adapter._data_path == "/twiki/data/Main"
        assert adapter._pub_path is None
        assert adapter._web_name == "Main"

    def test_init_with_pub_path(self):
        adapter = TWikiRawAdapter(
            ssh_host="myhost",
            data_path="/twiki/data/Main",
            pub_path="/twiki/pub/Main",
        )
        assert adapter._pub_path == "/twiki/pub/Main"

    def test_init_strips_trailing_slash(self):
        adapter = TWikiRawAdapter(
            ssh_host="myhost",
            data_path="/twiki/data/Main/",
            pub_path="/twiki/pub/Main/",
        )
        assert adapter._data_path == "/twiki/data/Main"
        assert adapter._pub_path == "/twiki/pub/Main"

    def test_init_custom_excludes(self):
        adapter = TWikiRawAdapter(
            ssh_host="myhost",
            data_path="/twiki/data/Main",
            exclude_patterns=["^Custom"],
        )
        assert adapter._should_skip("CustomPage")
        assert not adapter._should_skip("NormalPage")

    def test_site_type(self):
        adapter = TWikiRawAdapter(ssh_host="h", data_path="/d")
        assert adapter.site_type == "twiki_raw"


class TestTWikiRawAdapterExclude:
    """Test topic exclusion logic."""

    def setup_method(self):
        self.adapter = TWikiRawAdapter(ssh_host="h", data_path="/d")

    def test_skip_web_pages(self):
        assert self.adapter._should_skip("WebHome")
        assert self.adapter._should_skip("WebTopicList")

    def test_skip_watchlists(self):
        assert self.adapter._should_skip("JohnDoeWatchlist")

    def test_skip_twiki_admin(self):
        assert self.adapter._should_skip("TWikiAdminGroup")

    def test_skip_bookmarks(self):
        assert self.adapter._should_skip("JohnDoeBookmarks")

    def test_skip_templates(self):
        assert self.adapter._should_skip("DatabaseReportTemplate")

    def test_keep_normal_pages(self):
        assert not self.adapter._should_skip("AnalysisDB")
        assert not self.adapter._should_skip("EGIS1Introduction")
        assert not self.adapter._should_skip("EddbWrapperEddbOpen")


class TestTWikiRawAdapterDiscovery:
    """Test TWikiRawAdapter page discovery via SSH."""

    @patch("subprocess.run")
    def test_bulk_discover_pages(self, mock_run):
        """Test page discovery lists .txt files and filters."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"/twiki/data/Main/AnalysisDB.txt\n"
            b"/twiki/data/Main/EGIS.txt\n"
            b"/twiki/data/Main/WebHome.txt\n"
            b"/twiki/data/Main/JohnDoeWatchlist.txt\n"
            b"/twiki/data/Main/TWikiPreferences.txt\n",
        )

        adapter = TWikiRawAdapter(ssh_host="myhost", data_path="/twiki/data/Main")
        pages = adapter.bulk_discover_pages("jt60sa", "")

        # Should find 2 pages (WebHome, Watchlist, TWiki filtered out)
        assert len(pages) == 2
        names = [p.name for p in pages]
        assert "AnalysisDB" in names
        assert "EGIS" in names
        assert "WebHome" not in names
        assert "JohnDoeWatchlist" not in names

    @patch("subprocess.run")
    def test_bulk_discover_pages_url_scheme(self, mock_run):
        """Test discovered pages have ssh:// URLs."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"/twiki/data/Main/AnalysisDB.txt\n",
        )

        adapter = TWikiRawAdapter(ssh_host="myhost", data_path="/twiki/data/Main")
        pages = adapter.bulk_discover_pages("jt60sa", "")

        assert len(pages) == 1
        assert pages[0].url == "ssh://myhost/twiki/data/Main/AnalysisDB.txt"
        assert pages[0].namespace == "Main"

    @patch("subprocess.run")
    def test_bulk_discover_pages_ssh_failure(self, mock_run):
        """Test graceful handling of SSH failure."""
        mock_run.return_value = MagicMock(returncode=1, stdout=b"")

        adapter = TWikiRawAdapter(ssh_host="myhost", data_path="/twiki/data/Main")
        pages = adapter.bulk_discover_pages("jt60sa", "")

        assert len(pages) == 0

    @patch("subprocess.run")
    def test_bulk_discover_pages_timeout(self, mock_run):
        """Test graceful handling of SSH timeout."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=30)

        adapter = TWikiRawAdapter(ssh_host="myhost", data_path="/twiki/data/Main")
        pages = adapter.bulk_discover_pages("jt60sa", "")

        assert len(pages) == 0

    @patch("subprocess.run")
    def test_bulk_discover_pages_custom_excludes(self, mock_run):
        """Test custom exclude patterns from config."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"/d/DailyReport00001.txt\n/d/DailyReport00002.txt\n/d/EGIS.txt\n",
        )

        adapter = TWikiRawAdapter(
            ssh_host="h",
            data_path="/d",
            exclude_patterns=["^DailyReport"],
        )
        pages = adapter.bulk_discover_pages("jt60sa", "")

        assert len(pages) == 1
        assert pages[0].name == "EGIS"


class TestTWikiRawAdapterArtifacts:
    """Test TWikiRawAdapter artifact discovery."""

    def test_no_pub_path_returns_empty(self):
        adapter = TWikiRawAdapter(ssh_host="h", data_path="/d")
        artifacts = adapter.bulk_discover_artifacts("jt60sa", "")
        assert len(artifacts) == 0

    @patch("subprocess.run")
    def test_discovers_artifacts(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"/pub/Main/EGIS/fig1-1.png\n/pub/Main/EGIS/manual.pdf\n",
        )

        adapter = TWikiRawAdapter(
            ssh_host="h",
            data_path="/d",
            pub_path="/pub/Main",
        )
        artifacts = adapter.bulk_discover_artifacts("jt60sa", "")

        assert len(artifacts) == 2
        types = {a.artifact_type for a in artifacts}
        assert "pdf" in types

        # Should link to topic name
        assert artifacts[0].linked_pages == ["EGIS"]


# ──────────────────────────────────────────────────────────────────
# fetch_twiki_raw_content tests
# ──────────────────────────────────────────────────────────────────


class TestFetchTWikiRawContent:
    """Test SSH-based raw content fetching."""

    @patch("subprocess.run")
    def test_fetch_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b'%META:TOPICINFO{author="x"}%\n---+ Hello\nContent here\n',
        )

        content = fetch_twiki_raw_content("myhost", "/twiki/data/Main/Hello.txt")
        assert content is not None
        assert "%META:" in content
        assert "Content here" in content

    @patch("subprocess.run")
    def test_fetch_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout=b"")

        content = fetch_twiki_raw_content("myhost", "/twiki/data/Main/Missing.txt")
        assert content is None


# ──────────────────────────────────────────────────────────────────
# get_adapter factory tests
# ──────────────────────────────────────────────────────────────────


class TestGetAdapterTWikiRaw:
    """Test get_adapter factory for twiki_raw."""

    def test_get_adapter_twiki_raw(self):
        adapter = get_adapter(
            "twiki_raw",
            ssh_host="myhost",
            data_path="/twiki/data/Main",
        )
        assert isinstance(adapter, TWikiRawAdapter)

    def test_get_adapter_twiki_raw_requires_ssh(self):
        with pytest.raises(ValueError, match="ssh_host"):
            get_adapter("twiki_raw", data_path="/twiki/data/Main")

    def test_get_adapter_twiki_raw_requires_data_path(self):
        with pytest.raises(ValueError, match="data_path"):
            get_adapter("twiki_raw", ssh_host="myhost")


# ──────────────────────────────────────────────────────────────────
# twiki_markup_to_html converter tests
# ──────────────────────────────────────────────────────────────────


class TestTWikiMarkupToHtml:
    """Test TWiki markup to HTML conversion."""

    def test_meta_lines_stripped(self):
        markup = '%META:TOPICINFO{author="x" date="123"}%\n---+ Title\nContent'
        html = twiki_markup_to_html(markup)
        assert "%META:" not in html
        assert "Content" in html

    def test_headings_converted(self):
        markup = "---+ H1 Title\n---++ H2 Title\n---+++ H3 Title"
        html = twiki_markup_to_html(markup)
        assert "<h1>H1 Title</h1>" in html
        assert "<h2>H2 Title</h2>" in html
        assert "<h3>H3 Title</h3>" in html

    def test_title_extracted(self):
        markup = "---+ My Page Title\nSome content"
        html = twiki_markup_to_html(markup)
        assert "<title>My Page Title</title>" in html

    def test_bullet_lists(self):
        markup = "   * First item\n   * Second item"
        html = twiki_markup_to_html(markup)
        assert "<li>First item</li>" in html
        assert "<li>Second item</li>" in html

    def test_tables(self):
        markup = "| Name | Value |\n| foo | bar |"
        html = twiki_markup_to_html(markup)
        assert "<tr>" in html
        assert "<td>Name</td>" in html
        assert "<td>foo</td>" in html

    def test_verbatim_blocks(self):
        markup = '<verbatim>\nimport eddb\nprint("hello")\n</verbatim>'
        html = twiki_markup_to_html(markup)
        assert "<pre>" in html
        assert "import eddb" in html
        assert "</pre>" in html

    def test_literal_blocks_passthrough(self):
        markup = "<literal>\n<div>Raw HTML</div>\n</literal>"
        html = twiki_markup_to_html(markup)
        assert "<div>Raw HTML</div>" in html

    def test_variables_stripped(self):
        markup = "---+ Title\nSee %PARENTBC% for details"
        html = twiki_markup_to_html(markup)
        assert "%PARENTBC%" not in html
        assert "details" in html

    def test_br_converted(self):
        markup = "Line one%BR%Line two"
        html = twiki_markup_to_html(markup)
        assert "<br>" in html

    def test_color_variables_stripped(self):
        markup = "---+ %BLACK%Title Text%ENDCOLOR%"
        html = twiki_markup_to_html(markup)
        assert "%BLACK%" not in html
        assert "%ENDCOLOR%" not in html
        assert "Title Text" in html

    def test_full_page(self):
        """Test a realistic TWiki page."""
        markup = (
            '%META:TOPICINFO{author="x" date="1580200126" format="1.1"}%\n'
            '%META:TOPICPARENT{name="EGIS"}%\n'
            "---++ 4.1 eddbOpen\n\n"
            "実験データサーバとのソケット接続を開始します。\n\n"
            "---+++ 1) 引数/戻り値\n\n"
            "| 引数 | 型 | 概要 |\n"
            "| 第１戻り値 | bool | 成功/失敗 |\n\n"
            "---+++ 3) 使用例\n\n"
            "<verbatim>\nfrom eddb_pwrapper import eddbWrapper\n\n"
            "eddb = eddbWrapper()\nrtn = eddb.eddbOpen()\n</verbatim>\n"
        )
        html = twiki_markup_to_html(markup)

        # Should have proper HTML structure
        assert "<html>" in html
        assert "<title>" in html
        assert "</html>" in html

        # Content preserved
        assert "eddbOpen" in html
        assert "eddbWrapper" in html
        assert "ソケット接続" in html

        # No TWiki artifacts
        assert "%META:" not in html


class TestStripTWikiFormatting:
    """Test _strip_twiki_formatting helper."""

    def test_br_replacement(self):
        assert "<br>" in _strip_twiki_formatting("text%BR%more")

    def test_variable_stripping(self):
        assert _strip_twiki_formatting("%PARENTBC%") == ""

    def test_bold_formatting(self):
        result = _strip_twiki_formatting("*bold text*")
        assert "<b>bold text</b>" in result

    def test_italic_formatting(self):
        result = _strip_twiki_formatting("_italic text_")
        assert "<i>italic text</i>" in result

    def test_usersig_stripped(self):
        result = _strip_twiki_formatting("%USERSIG{JohnDoe - 2020-01-01}%")
        assert "USERSIG" not in result
