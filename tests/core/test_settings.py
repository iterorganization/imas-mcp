"""Tests for settings.py module."""

import pytest

from imas_codex import settings
from imas_codex.settings import _parse_bool


class TestSettingsFunctions:
    """Tests for settings module functions."""

    def test_get_embedding_model_env_override(self, monkeypatch):
        """Environment variable overrides embedding model setting."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.setenv("IMAS_CODEX_EMBEDDING_MODEL", "test-model")
        result = settings.get_embedding_model()

        assert result == "test-model"

    def test_get_model_language_env_override(self, monkeypatch):
        """Environment variable overrides language model setting."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.setenv("IMAS_CODEX_LANGUAGE_MODEL", "test-llm")
        result = settings.get_model("language")

        assert result == "test-llm"

    def test_get_model_vision_env_override(self, monkeypatch):
        """Environment variable overrides vision model setting."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.setenv("IMAS_CODEX_VISION_MODEL", "test-vlm")
        result = settings.get_model("vision")

        assert result == "test-vlm"

    def test_get_model_compaction_env_override(self, monkeypatch):
        """Environment variable overrides compaction model setting."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.setenv("IMAS_CODEX_COMPACTION_MODEL", "test-compact")
        result = settings.get_model("compaction")

        assert result == "test-compact"

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

    def test_get_dd_version_env_override(self, monkeypatch):
        """Environment variable overrides DD version."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.setenv("IMAS_DD_VERSION", "3.99.0")
        result = settings.get_dd_version()

        assert result == "3.99.0"

    def test_get_embedding_model_default(self, monkeypatch):
        """get_embedding_model returns default when env not set."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.delenv("IMAS_CODEX_EMBEDDING_MODEL", raising=False)
        result = settings.get_embedding_model()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_model_language_default(self, monkeypatch):
        """get_model('language') returns default when env not set."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.delenv("IMAS_CODEX_LANGUAGE_MODEL", raising=False)
        result = settings.get_model("language")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_labeling_batch_size_default(self, monkeypatch):
        """get_labeling_batch_size returns default when env not set."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.delenv("IMAS_CODEX_LABELING_BATCH_SIZE", raising=False)
        result = settings.get_labeling_batch_size()

        assert isinstance(result, int)
        assert result > 0

    def test_get_include_ggd_default(self, monkeypatch):
        """get_include_ggd returns default when env not set."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.delenv("IMAS_CODEX_INCLUDE_GGD", raising=False)
        result = settings.get_include_ggd()

        assert isinstance(result, bool)

    def test_get_include_error_fields_default(self, monkeypatch):
        """get_include_error_fields returns default when env not set."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.delenv("IMAS_CODEX_INCLUDE_ERROR_FIELDS", raising=False)
        result = settings.get_include_error_fields()

        assert isinstance(result, bool)


class TestGetModel:
    """Tests for unified get_model(section) function."""

    def test_language_section_returns_model(self):
        """Language section returns a model string."""
        model = settings.get_model("language")
        assert isinstance(model, str)
        assert "/" in model

    def test_vision_section_returns_model(self):
        """Vision section returns a model string."""
        model = settings.get_model("vision")
        assert isinstance(model, str)
        assert "/" in model

    def test_agent_section_returns_model(self):
        """Agent section returns a model string."""
        model = settings.get_model("agent")
        assert isinstance(model, str)
        assert "/" in model

    def test_compaction_section_returns_model(self):
        """Compaction section returns a model string."""
        model = settings.get_model("compaction")
        assert isinstance(model, str)
        assert "/" in model

    def test_embedding_section_returns_model(self):
        """Embedding section returns a model string."""
        model = settings.get_model("embedding")
        assert isinstance(model, str)
        assert len(model) > 0

    def test_unknown_section_raises(self):
        """Unknown section raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model section"):
            settings.get_model("nonexistent_section")


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


class TestModuleLevelConstants:
    """Tests for module-level constants."""

    def test_module_constants_exist(self):
        """Module-level constants are defined."""
        assert hasattr(settings, "LABELING_BATCH_SIZE")
        assert hasattr(settings, "INCLUDE_GGD")
        assert hasattr(settings, "INCLUDE_ERROR_FIELDS")
        assert hasattr(settings, "EMBEDDING_BACKEND")
        assert hasattr(settings, "EMBEDDING_DIMENSION")

    def test_module_constants_have_correct_types(self):
        """Module-level constants have correct types."""
        assert isinstance(settings.LABELING_BATCH_SIZE, int)
        assert isinstance(settings.INCLUDE_GGD, bool)
        assert isinstance(settings.INCLUDE_ERROR_FIELDS, bool)
        assert isinstance(settings.EMBEDDING_BACKEND, str)
        assert isinstance(settings.EMBEDDING_DIMENSION, int)


class TestGraphSettings:
    """Tests for graph (Neo4j) settings accessors."""

    def test_get_graph_uri_default(self, monkeypatch):
        """get_graph_uri returns pyproject value or default."""
        settings._load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        result = settings.get_graph_uri()
        assert isinstance(result, str)
        assert result.startswith("bolt://")

    def test_get_graph_uri_env_override(self, monkeypatch):
        """NEO4J_URI env var overrides pyproject.toml."""
        settings._load_pyproject_settings.cache_clear()
        monkeypatch.setenv("NEO4J_URI", "bolt://remote-host:7687")
        result = settings.get_graph_uri()
        assert result == "bolt://remote-host:7687"

    def test_get_graph_username_default(self, monkeypatch):
        """get_graph_username returns pyproject value or default."""
        settings._load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        result = settings.get_graph_username()
        assert isinstance(result, str)
        assert result == "neo4j"

    def test_get_graph_username_env_override(self, monkeypatch):
        """NEO4J_USERNAME env var overrides pyproject.toml."""
        settings._load_pyproject_settings.cache_clear()
        monkeypatch.setenv("NEO4J_USERNAME", "custom-user")
        result = settings.get_graph_username()
        assert result == "custom-user"

    def test_get_graph_password_default(self, monkeypatch):
        """get_graph_password returns pyproject value or default."""
        settings._load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        result = settings.get_graph_password()
        assert isinstance(result, str)
        assert result == "imas-codex"

    def test_get_graph_password_env_override(self, monkeypatch):
        """NEO4J_PASSWORD env var overrides pyproject.toml."""
        settings._load_pyproject_settings.cache_clear()
        monkeypatch.setenv("NEO4J_PASSWORD", "secret-pw")
        result = settings.get_graph_password()
        assert result == "secret-pw"

    def test_graph_settings_from_pyproject(self, monkeypatch):
        """Graph settings are read from pyproject.toml [tool.imas-codex.graph]."""
        settings._load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        # These should resolve from pyproject.toml which has the graph section
        uri = settings.get_graph_uri()
        username = settings.get_graph_username()
        password = settings.get_graph_password()

        assert "bolt://" in uri
        assert username == "neo4j"
        assert password == "imas-codex"

    def test_get_graph_name_default(self, monkeypatch):
        """get_graph_name returns active profile name."""
        settings._load_pyproject_settings.cache_clear()
        monkeypatch.delenv("IMAS_CODEX_GRAPH", raising=False)
        name = settings.get_graph_name()
        assert name == "iter"

    def test_get_graph_name_env_override(self, monkeypatch):
        """IMAS_CODEX_GRAPH env var switches the active profile."""
        settings._load_pyproject_settings.cache_clear()
        monkeypatch.setenv("IMAS_CODEX_GRAPH", "tcv")
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        name = settings.get_graph_name()
        assert name == "tcv"
        # URI should reflect the tcv profile port
        uri = settings.get_graph_uri()
        assert ":7688" in uri

    def test_get_graph_profile_returns_profile(self, monkeypatch):
        """get_graph_profile returns a GraphProfile object."""
        settings._load_pyproject_settings.cache_clear()
        monkeypatch.delenv("IMAS_CODEX_GRAPH", raising=False)
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        profile = settings.get_graph_profile()
        assert profile.name == "iter"
        assert profile.bolt_port == 7687
