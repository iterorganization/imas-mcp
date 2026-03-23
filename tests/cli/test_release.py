"""Tests for the release CLI version computation logic."""

from unittest.mock import patch

import pytest

from imas_codex.cli.release import (
    _apply_bump,
    _detect_state,
    _format_git_tag,
    _format_pep440,
    _parse_version,
    compute_next_version,
)


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


class TestApplyBump:
    def test_major(self):
        assert _apply_bump(5, 2, 3, "major") == (6, 0, 0)

    def test_minor(self):
        assert _apply_bump(5, 2, 3, "minor") == (5, 3, 0)

    def test_patch(self):
        assert _apply_bump(5, 2, 3, "patch") == (5, 2, 4)

    def test_invalid(self):
        with pytest.raises(Exception, match="Invalid bump"):
            _apply_bump(1, 0, 0, "invalid")


class TestDetectState:
    @patch("imas_codex.cli.release._get_latest_tag", return_value="v5.0.0")
    def test_stable(self, _):
        info = _detect_state()
        assert info["state"] == "stable"
        assert info["tag"] == "v5.0.0"
        assert info["rc"] is None

    @patch("imas_codex.cli.release._get_latest_tag", return_value="v5.0.0-rc3")
    def test_rc_mode(self, _):
        info = _detect_state()
        assert info["state"] == "rc"
        assert info["tag"] == "v5.0.0-rc3"
        assert info["rc"] == 3

    @patch("imas_codex.cli.release._get_latest_tag", return_value=None)
    def test_no_tags(self, _):
        info = _detect_state()
        assert info["state"] is None
        assert info["tag"] is None


class TestComputeNextVersion:
    """Test state machine transitions for compute_next_version."""

    # --- Stable state transitions ---

    @patch("imas_codex.cli.release._tag_exists", return_value=False)
    @patch("imas_codex.cli.release._get_latest_tag", return_value="v5.0.0")
    def test_stable_bump_major(self, *_):
        tag, pep = compute_next_version("major")
        assert tag == "v6.0.0-rc1"
        assert pep == "6.0.0rc1"

    @patch("imas_codex.cli.release._tag_exists", return_value=False)
    @patch("imas_codex.cli.release._get_latest_tag", return_value="v5.0.0")
    def test_stable_bump_minor(self, *_):
        tag, pep = compute_next_version("minor")
        assert tag == "v5.1.0-rc1"
        assert pep == "5.1.0rc1"

    @patch("imas_codex.cli.release._tag_exists", return_value=False)
    @patch("imas_codex.cli.release._get_latest_tag", return_value="v5.0.0")
    def test_stable_bump_patch(self, *_):
        tag, pep = compute_next_version("patch")
        assert tag == "v5.0.1-rc1"
        assert pep == "5.0.1rc1"

    @patch("imas_codex.cli.release._tag_exists", return_value=False)
    @patch("imas_codex.cli.release._get_latest_tag", return_value="v5.0.0")
    def test_stable_bump_major_final(self, *_):
        tag, pep = compute_next_version("major", final=True)
        assert tag == "v6.0.0"
        assert pep == "6.0.0"

    @patch("imas_codex.cli.release._tag_exists", return_value=False)
    @patch("imas_codex.cli.release._get_latest_tag", return_value="v5.0.0")
    def test_stable_bump_patch_final(self, *_):
        tag, pep = compute_next_version("patch", final=True)
        assert tag == "v5.0.1"
        assert pep == "5.0.1"

    @patch("imas_codex.cli.release._get_latest_tag", return_value="v5.0.0")
    def test_stable_no_bump_errors(self, _):
        with pytest.raises(Exception, match="Specify --bump"):
            compute_next_version(None)

    @patch("imas_codex.cli.release._get_latest_tag", return_value="v5.0.0")
    def test_stable_final_only_errors(self, _):
        with pytest.raises(Exception, match="Not in RC mode"):
            compute_next_version(None, final=True)

    # --- RC mode transitions ---

    @patch("imas_codex.cli.release._tag_exists", return_value=False)
    @patch("imas_codex.cli.release._get_latest_tag", return_value="v5.0.0-rc1")
    def test_rc_increment(self, *_):
        tag, pep = compute_next_version(None)
        assert tag == "v5.0.0-rc2"
        assert pep == "5.0.0rc2"

    @patch("imas_codex.cli.release._get_latest_tag", return_value="v5.0.0-rc3")
    def test_rc_finalize(self, _):
        tag, pep = compute_next_version(None, final=True)
        assert tag == "v5.0.0"
        assert pep == "5.0.0"

    # --- RC mode + --bump: abandon and bump from last STABLE ---

    @patch("imas_codex.cli.release._tag_exists", return_value=False)
    @patch("imas_codex.cli.release._get_latest_stable_tag", return_value="v5.0.0")
    @patch("imas_codex.cli.release._get_latest_tag", return_value="v6.0.0-rc3")
    def test_rc_bump_minor_from_stable(self, *_):
        """v6.0.0-rc3 + --bump minor → v5.1.0-rc1 (from stable v5.0.0)."""
        tag, pep = compute_next_version("minor")
        assert tag == "v5.1.0-rc1"
        assert pep == "5.1.0rc1"

    @patch("imas_codex.cli.release._tag_exists", return_value=False)
    @patch("imas_codex.cli.release._get_latest_stable_tag", return_value="v5.0.0")
    @patch("imas_codex.cli.release._get_latest_tag", return_value="v6.0.0-rc3")
    def test_rc_bump_patch_from_stable(self, *_):
        """v6.0.0-rc3 + --bump patch → v5.0.1-rc1 (from stable v5.0.0)."""
        tag, pep = compute_next_version("patch")
        assert tag == "v5.0.1-rc1"
        assert pep == "5.0.1rc1"

    @patch("imas_codex.cli.release._tag_exists", return_value=False)
    @patch("imas_codex.cli.release._get_latest_stable_tag", return_value="v5.0.0")
    @patch("imas_codex.cli.release._get_latest_tag", return_value="v6.0.0-rc3")
    def test_rc_bump_major_from_stable(self, *_):
        """v6.0.0-rc3 + --bump major → v6.0.0-rc1 (from stable v5.0.0)."""
        tag, pep = compute_next_version("major")
        assert tag == "v6.0.0-rc1"
        assert pep == "6.0.0rc1"

    @patch("imas_codex.cli.release._tag_exists", return_value=False)
    @patch("imas_codex.cli.release._get_latest_stable_tag", return_value="v5.0.0")
    @patch("imas_codex.cli.release._get_latest_tag", return_value="v6.0.0-rc3")
    def test_rc_bump_patch_final_from_stable(self, *_):
        """v6.0.0-rc3 + --bump patch --final → v5.0.1 (from stable v5.0.0)."""
        tag, pep = compute_next_version("patch", final=True)
        assert tag == "v5.0.1"
        assert pep == "5.0.1"

    @patch("imas_codex.cli.release._get_latest_stable_tag", return_value=None)
    @patch("imas_codex.cli.release._get_latest_tag", return_value="v5.0.0-rc1")
    def test_rc_bump_no_stable_tag_errors(self, *_):
        """Error when bumping in RC mode with no stable tags."""
        with pytest.raises(Exception, match="No stable"):
            compute_next_version("patch")

    # --- Edge cases ---

    @patch("imas_codex.cli.release._get_latest_tag", return_value=None)
    def test_no_tags_errors(self, _):
        with pytest.raises(Exception, match="No existing version tags"):
            compute_next_version("major")
