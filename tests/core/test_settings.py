"""Tests for settings.py module."""

from imas_codex import settings
from imas_codex.settings import _parse_bool


class TestSettingsFunctions:
    """Tests for settings module functions."""

    def test_get_imas_embedding_model_env_override(self, monkeypatch):
        """Environment variable overrides embedding model setting."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.setenv("IMAS_CODEX_EMBEDDING_MODEL", "test-model")
        result = settings.get_imas_embedding_model()

        assert result == "test-model"

    def test_get_language_model_env_override(self, monkeypatch):
        """Environment variable overrides language model setting."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.setenv("IMAS_CODEX_LANGUAGE_MODEL", "test-llm")
        result = settings.get_language_model()

        assert result == "test-llm"

    def test_get_labeling_batch_size_env_override(self, monkeypatch):
        """Environment variable overrides labeling batch size."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.setenv("IMAS_CODEX_LABELING_BATCH_SIZE", "100")
        result = settings.get_labeling_batch_size()

        assert result == 100

    def test_get_include_ggd_env_override(self, monkeypatch):
        """Environment variable overrides include_ggd setting."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.setenv("IMAS_CODEX_INCLUDE_GGD", "false")
        result = settings.get_include_ggd()

        assert result is False

    def test_get_include_error_fields_env_override(self, monkeypatch):
        """Environment variable overrides include_error_fields setting."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.setenv("IMAS_CODEX_INCLUDE_ERROR_FIELDS", "true")
        result = settings.get_include_error_fields()

        assert result is True


class TestParseBool:
    """Tests for the _parse_bool helper function."""

    def test_true_string_values(self):
        """True string values are parsed correctly."""
        assert _parse_bool("true") is True
        assert _parse_bool("True") is True
        assert _parse_bool("TRUE") is True
        assert _parse_bool("1") is True
        assert _parse_bool("yes") is True

    def test_false_string_values(self):
        """False string values are parsed correctly."""
        assert _parse_bool("false") is False
        assert _parse_bool("0") is False
        assert _parse_bool("no") is False

    def test_bool_values_pass_through(self):
        """Boolean values pass through unchanged."""
        assert _parse_bool(True) is True
        assert _parse_bool(False) is False
