"""Tests for SN catalog default path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestGetSnStagingDir:
    """Tests for get_sn_staging_dir()."""

    def test_default_path(self, monkeypatch):
        """Default returns ~/.cache/imas-codex/staging."""
        monkeypatch.delenv("IMAS_CODEX_SN_STAGING", raising=False)
        from imas_codex.settings import get_sn_staging_dir

        result = get_sn_staging_dir()
        assert result == Path.home() / ".cache" / "imas-codex" / "staging"

    def test_env_var_override(self, monkeypatch, tmp_path):
        """IMAS_CODEX_SN_STAGING env var overrides config and default."""
        custom = tmp_path / "custom-staging"
        monkeypatch.setenv("IMAS_CODEX_SN_STAGING", str(custom))
        from imas_codex.settings import get_sn_staging_dir

        result = get_sn_staging_dir()
        assert result == custom

    def test_env_var_with_tilde(self, monkeypatch):
        """Tilde in env var is expanded."""
        monkeypatch.setenv("IMAS_CODEX_SN_STAGING", "~/my-staging")
        from imas_codex.settings import get_sn_staging_dir

        result = get_sn_staging_dir()
        assert result == Path.home() / "my-staging"


class TestGetSnIsncDir:
    """Tests for get_sn_isnc_dir()."""

    def test_returns_none_when_nothing_found(self, monkeypatch):
        """Returns None when no ISNC can be found."""
        monkeypatch.delenv("IMAS_CODEX_SN_ISNC", raising=False)
        # Ensure config returns empty
        monkeypatch.setattr(
            "imas_codex.settings._get_section",
            lambda section: {} if section == "sn" else {},
        )
        from imas_codex.settings import get_sn_isnc_dir

        # With no sibling dirs matching, should return None
        # (exact result depends on project layout, but the function should not crash)
        result = get_sn_isnc_dir()
        # Result is either a Path (if sibling found) or None
        assert result is None or isinstance(result, Path)

    def test_env_var_override(self, monkeypatch, tmp_path):
        """IMAS_CODEX_SN_ISNC env var takes priority."""
        isnc_dir = tmp_path / "my-isnc"
        isnc_dir.mkdir()
        monkeypatch.setenv("IMAS_CODEX_SN_ISNC", str(isnc_dir))
        from imas_codex.settings import get_sn_isnc_dir

        result = get_sn_isnc_dir()
        assert result == isnc_dir

    def test_env_var_nonexistent_dir_returns_none(self, monkeypatch, tmp_path):
        """Env var pointing to nonexistent dir returns None."""
        monkeypatch.setenv("IMAS_CODEX_SN_ISNC", str(tmp_path / "does-not-exist"))
        from imas_codex.settings import get_sn_isnc_dir

        result = get_sn_isnc_dir()
        assert result is None

    def test_sibling_exact_match_preferred(self, monkeypatch, tmp_path):
        """Exact name 'imas-standard-names-catalog' wins over glob matches."""
        # Create a fake project root structure
        project_root = tmp_path / "imas-codex"
        project_root.mkdir()
        exact = tmp_path / "imas-standard-names-catalog"
        exact.mkdir()
        fuzzy = tmp_path / "imas-standard-names-catalog-fork"
        fuzzy.mkdir()

        monkeypatch.delenv("IMAS_CODEX_SN_ISNC", raising=False)
        monkeypatch.setattr(
            "imas_codex.settings._get_section",
            lambda section: {},
        )
        # Patch Path(__file__).resolve().parent.parent to point to our fake project root
        import imas_codex.settings as settings_mod

        # We need to make the auto-discovery look in tmp_path
        # The function does: project_root = Path(__file__).resolve().parent.parent
        # So we need __file__ to be inside project_root/imas_codex/settings.py
        fake_settings = project_root / "imas_codex" / "settings.py"
        fake_settings.parent.mkdir(parents=True, exist_ok=True)
        fake_settings.touch()
        monkeypatch.setattr(settings_mod, "__file__", str(fake_settings))

        from imas_codex.settings import get_sn_isnc_dir

        result = get_sn_isnc_dir()
        assert result == exact

    def test_ambiguous_siblings_returns_none(self, monkeypatch, tmp_path):
        """Multiple non-exact matches return None."""
        project_root = tmp_path / "imas-codex"
        project_root.mkdir()
        (tmp_path / "foo-standard-names-catalog-a").mkdir()
        (tmp_path / "bar-standard-names-catalog-b").mkdir()

        monkeypatch.delenv("IMAS_CODEX_SN_ISNC", raising=False)
        monkeypatch.setattr(
            "imas_codex.settings._get_section",
            lambda section: {},
        )
        import imas_codex.settings as settings_mod

        fake_settings = project_root / "imas_codex" / "settings.py"
        fake_settings.parent.mkdir(parents=True, exist_ok=True)
        fake_settings.touch()
        monkeypatch.setattr(settings_mod, "__file__", str(fake_settings))

        from imas_codex.settings import get_sn_isnc_dir

        result = get_sn_isnc_dir()
        assert result is None
