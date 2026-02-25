"""Tests for TWikiAdapter (live TWiki via HTTP/SSH)."""

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.discovery.wiki.adapters import (
    DiscoveredPage,
    TWikiAdapter,
    get_adapter,
)

# ──────────────────────────────────────────────────────────────────
# TWikiAdapter init tests
# ──────────────────────────────────────────────────────────────────


class TestTWikiAdapterInit:
    """Test TWikiAdapter initialization."""

    def test_site_type(self):
        assert TWikiAdapter.site_type == "twiki"

    def test_init_defaults(self):
        adapter = TWikiAdapter(ssh_host="myhost")
        assert adapter.ssh_host == "myhost"
        assert adapter.webs == ["Main"]
        assert adapter._base_url is None

    def test_init_custom_webs(self):
        adapter = TWikiAdapter(ssh_host="myhost", webs=["Main", "Code"])
        assert adapter.webs == ["Main", "Code"]

    def test_init_with_base_url(self):
        adapter = TWikiAdapter(
            ssh_host="myhost",
            base_url="http://157.111.10.188/twiki",
        )
        assert adapter._base_url == "http://157.111.10.188/twiki"


# ──────────────────────────────────────────────────────────────────
# SSH curl helper tests
# ──────────────────────────────────────────────────────────────────


class TestSSHCurl:
    """Test _ssh_curl method."""

    def test_ssh_curl_success(self):
        adapter = TWikiAdapter(ssh_host="myhost")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="<html>content</html>",
            )
            result = adapter._ssh_curl("http://host/twiki/bin/view/Main/WebHome")
            assert result == "<html>content</html>"

            # Check --noproxy is included
            cmd_arg = mock_run.call_args[0][0]
            assert cmd_arg[0] == "ssh"
            assert cmd_arg[1] == "myhost"
            assert '--noproxy "*"' in cmd_arg[2]

    def test_ssh_curl_no_host(self):
        adapter = TWikiAdapter(ssh_host=None)
        result = adapter._ssh_curl("http://host/page")
        assert result is None

    def test_ssh_curl_failure(self):
        adapter = TWikiAdapter(ssh_host="myhost")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            result = adapter._ssh_curl("http://host/page")
            assert result is None

    def test_ssh_curl_timeout(self):
        import subprocess

        adapter = TWikiAdapter(ssh_host="myhost")
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)):
            result = adapter._ssh_curl("http://host/page")
            assert result is None


# ──────────────────────────────────────────────────────────────────
# Page discovery tests
# ──────────────────────────────────────────────────────────────────


SAMPLE_TOPIC_LIST_HTML = """
<html>
<head><title>WebTopicList < Main < TWiki</title></head>
<body>
<div class="patternTopic">
<a href="/twiki/bin/view/Main/WebHome">WebHome</a>
<a href="/twiki/bin/view/Main/DBAccessLibraries">DBAccessLibraries</a>
<a href="/twiki/bin/view/Main/DailyReport00001">DailyReport00001</a>
<a href="/twiki/bin/view/Main/ShotE00100">ShotE00100</a>
<a href="/twiki/bin/view/Main/AnalysisDB">AnalysisDB</a>
<a href="/twiki/bin/view/Main/WebTopicList">WebTopicList</a>
<a href="/twiki/bin/view/Main/WebIndex">WebIndex</a>
<a href="/twiki/bin/view/Main/WebRss">WebRss</a>
<a href="/twiki/bin/view/Main/WebNotify">WebNotify</a>
<a href="/twiki/bin/view/Main/WebPreferences">WebPreferences</a>
</div><!-- /patternTopic-->
</body>
</html>
"""


class TestTWikiAdapterDiscovery:
    """Test bulk_discover_pages."""

    def test_discover_pages_filters_utility(self):
        adapter = TWikiAdapter(
            ssh_host="myhost",
            base_url="http://157.111.10.188/twiki",
        )
        with patch.object(adapter, "_ssh_curl", return_value=SAMPLE_TOPIC_LIST_HTML):
            pages = adapter.bulk_discover_pages(
                "jt-60sa", "http://157.111.10.188/twiki"
            )

        names = [p.name for p in pages]
        # Should include content topics
        assert "Main/WebHome" in names
        assert "Main/DBAccessLibraries" in names
        assert "Main/DailyReport00001" in names
        assert "Main/ShotE00100" in names
        assert "Main/AnalysisDB" in names

        # Should exclude utility pages
        assert "Main/WebTopicList" not in names
        assert "Main/WebIndex" not in names
        assert "Main/WebRss" not in names
        assert "Main/WebNotify" not in names
        assert "Main/WebPreferences" not in names

    def test_discover_pages_no_duplicates(self):
        html = """
        <a href="/twiki/bin/view/Main/Topic1">Topic1</a>
        <a href="/twiki/bin/view/Main/Topic1">Topic1 again</a>
        <a href="/twiki/bin/view/Main/Topic2">Topic2</a>
        """
        adapter = TWikiAdapter(
            ssh_host="myhost",
            base_url="http://host/twiki",
        )
        with patch.object(adapter, "_ssh_curl", return_value=html):
            pages = adapter.bulk_discover_pages("fac", "http://host/twiki")

        names = [p.name for p in pages]
        assert names.count("Main/Topic1") == 1
        assert len(pages) == 2

    def test_discover_pages_multi_web(self):
        main_html = '<a href="/twiki/bin/view/Main/TopicA">A</a>'
        code_html = '<a href="/twiki/bin/view/Code/TopicB">B</a>'

        adapter = TWikiAdapter(
            ssh_host="myhost",
            webs=["Main", "Code"],
            base_url="http://host/twiki",
        )

        def mock_curl(url, timeout=30):
            if "Main/WebTopicList" in url:
                return main_html
            if "Code/WebTopicList" in url:
                return code_html
            return None

        with patch.object(adapter, "_ssh_curl", side_effect=mock_curl):
            pages = adapter.bulk_discover_pages("fac", "http://host/twiki")

        names = [p.name for p in pages]
        assert "Main/TopicA" in names
        assert "Code/TopicB" in names

    def test_discover_pages_url_construction(self):
        html = '<a href="/twiki/bin/view/Main/MyTopic">MyTopic</a>'
        adapter = TWikiAdapter(
            ssh_host="myhost",
            base_url="http://157.111.10.188/twiki",
        )
        with patch.object(adapter, "_ssh_curl", return_value=html):
            pages = adapter.bulk_discover_pages("fac", "http://157.111.10.188/twiki")

        assert len(pages) == 1
        assert pages[0].url == "http://157.111.10.188/twiki/bin/view/Main/MyTopic"
        assert pages[0].name == "Main/MyTopic"

    def test_discover_pages_empty_web(self):
        adapter = TWikiAdapter(
            ssh_host="myhost",
            base_url="http://host/twiki",
        )
        with patch.object(adapter, "_ssh_curl", return_value=None):
            pages = adapter.bulk_discover_pages("fac", "http://host/twiki")

        assert pages == []

    def test_discover_pages_no_ssh_host(self):
        adapter = TWikiAdapter(ssh_host=None)
        pages = adapter.bulk_discover_pages("fac", "http://host/twiki")
        assert pages == []

    def test_discover_pages_progress_callback(self):
        html = """
        <a href="/twiki/bin/view/Main/Topic1">T1</a>
        <a href="/twiki/bin/view/Main/Topic2">T2</a>
        """
        adapter = TWikiAdapter(
            ssh_host="myhost",
            base_url="http://host/twiki",
        )
        progress_msgs = []

        def on_progress(msg, _):
            progress_msgs.append(msg)

        with patch.object(adapter, "_ssh_curl", return_value=html):
            adapter.bulk_discover_pages("fac", "http://host/twiki", on_progress)

        assert any("2 topics" in m for m in progress_msgs)
        assert any("2 pages total" in m for m in progress_msgs)

    def test_discover_uses_base_url_parameter_if_no_init_url(self):
        html = '<a href="/twiki/bin/view/Main/Topic">T</a>'
        adapter = TWikiAdapter(ssh_host="myhost")
        with patch.object(adapter, "_ssh_curl", return_value=html) as mock:
            adapter.bulk_discover_pages("fac", "http://fallback/twiki")

        called_url = mock.call_args[0][0]
        assert called_url.startswith("http://fallback/twiki")


# ──────────────────────────────────────────────────────────────────
# get_adapter factory tests
# ──────────────────────────────────────────────────────────────────


class TestGetAdapterTWiki:
    """Test get_adapter factory for twiki type."""

    def test_get_adapter_twiki(self):
        adapter = get_adapter(
            "twiki",
            ssh_host="myhost",
            base_url="http://host/twiki",
            webs=["Main", "Code"],
        )
        assert isinstance(adapter, TWikiAdapter)
        assert adapter.ssh_host == "myhost"
        assert adapter.webs == ["Main", "Code"]
        assert adapter._base_url == "http://host/twiki"

    def test_get_adapter_twiki_defaults(self):
        adapter = get_adapter("twiki", ssh_host="myhost")
        assert isinstance(adapter, TWikiAdapter)
        assert adapter.webs == ["Main"]


# ──────────────────────────────────────────────────────────────────
# html_to_text TWiki support
# ──────────────────────────────────────────────────────────────────


class TestHTMLToTextTWiki:
    """Test html_to_text extracts content from TWiki rendered HTML."""

    def test_extracts_twiki_pattern_topic(self):
        from imas_codex.discovery.wiki.pipeline import html_to_text

        html = """
        <html><head><title>Test</title></head><body>
        <div class="patternTopic">
        <h1>Data Access Libraries</h1>
        <p>This is the main content about EDAS.</p>
        <table><tr><td>Version</td><td>1.7.0</td></tr></table>
        </div><!-- /patternTopic-->
        <div class="patternTopicActions">edit buttons</div>
        </body></html>
        """
        text, sections = html_to_text(html)
        assert "Data Access Libraries" in text
        assert "EDAS" in text
        assert "1.7.0" in text
        # Should NOT include patternTopicActions
        assert "edit buttons" not in text

    def test_falls_back_to_mediawiki(self):
        from imas_codex.discovery.wiki.pipeline import html_to_text

        html = """
        <div id="bodyContent">
        <p>MediaWiki content here</p>
        </div>
        <div class="printfooter">footer</div>
        """
        text, _ = html_to_text(html)
        assert "MediaWiki content" in text
        assert "footer" not in text
