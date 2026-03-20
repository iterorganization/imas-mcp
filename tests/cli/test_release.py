"""Tests for the release CLI version computation logic."""

import pytest

from imas_codex.cli.release import _format_git_tag, _format_pep440, _parse_version


class TestParseVersion:
    def test_simple_release(self):
        assert _parse_version("v5.0.0") == (5, 0, 0, None)

    def test_release_candidate(self):
        assert _parse_version("v5.0.0-rc1") == (5, 0, 0, 1)

    def test_multi_digit_rc(self):
        assert _parse_version("v5.0.0-rc12") == (5, 0, 0, 12)

    def test_patch_version(self):
        assert _parse_version("v5.2.3") == (5, 2, 3, None)

    def test_strips_v_prefix(self):
        assert _parse_version("v1.0.0") == (1, 0, 0, None)

    def test_invalid_format(self):
        with pytest.raises(Exception, match="Cannot parse"):
            _parse_version("invalid")


class TestFormatGitTag:
    def test_release(self):
        assert _format_git_tag(5, 0, 0, None) == "v5.0.0"

    def test_rc(self):
        assert _format_git_tag(5, 0, 0, 1) == "v5.0.0-rc1"

    def test_patch_rc(self):
        assert _format_git_tag(5, 0, 1, 3) == "v5.0.1-rc3"


class TestFormatPep440:
    def test_release(self):
        assert _format_pep440(5, 0, 0, None) == "5.0.0"

    def test_rc(self):
        assert _format_pep440(5, 0, 0, 1) == "5.0.0rc1"

    def test_patch_rc(self):
        assert _format_pep440(5, 0, 1, 3) == "5.0.1rc3"
