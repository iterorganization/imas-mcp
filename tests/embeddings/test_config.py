"""Tests for embeddings/config.py module."""

from unittest.mock import patch

import pytest

from imas_mcp.embeddings.config import EncoderConfig


class TestEncoderConfig:
    """Tests for the EncoderConfig class."""

    def test_initialization_defaults(self):
        """Config initializes with default values."""
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "imas_mcp.embeddings.config.IMAS_MCP_EMBEDDING_MODEL",
                "all-MiniLM-L6-v2",
            ):
                config = EncoderConfig()

                assert config.batch_size == 250
                assert config.normalize_embeddings is True
                assert config.use_half_precision is False
                assert config.enable_cache is True

    def test_initialization_explicit_values(self):
        """Config uses explicitly provided values."""
        config = EncoderConfig(
            model_name="custom-model",
            batch_size=100,
            normalize_embeddings=False,
            use_half_precision=True,
        )

        assert config.model_name == "custom-model"
        assert config.batch_size == 100
        assert config.normalize_embeddings is False
        assert config.use_half_precision is True

    def test_api_model_auto_detection_slash(self, monkeypatch):
        """Config auto-detects API model from slash in name."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.test.com")

        config = EncoderConfig(model_name="openai/text-embedding-3-small")

        assert config.use_api_embeddings is True

    def test_api_model_auto_detection_embedding_keyword(self, monkeypatch):
        """Config auto-detects API model from embedding keyword."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.test.com")

        config = EncoderConfig(model_name="text-embedding-ada-002")

        assert config.use_api_embeddings is True

    def test_local_model_detection(self, monkeypatch):
        """Config detects local model without API indicators."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        config = EncoderConfig(model_name="all-MiniLM-L6-v2")
        config.use_api_embeddings = False  # Override auto-detection

        assert config.use_api_embeddings is False

    def test_environment_variable_loading(self, monkeypatch):
        """Config loads values from environment variables."""
        monkeypatch.setenv("IMAS_MCP_EMBEDDING_MODEL", "env-model")
        monkeypatch.setenv("OPENAI_API_KEY", "env-api-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://env.api.com")

        with patch(
            "imas_mcp.embeddings.config.IMAS_MCP_EMBEDDING_MODEL", "fallback-model"
        ):
            config = EncoderConfig()

            assert config.openai_api_key == "env-api-key"
            assert config.openai_base_url == "https://env.api.com"

    def test_validate_api_config_missing_key(self):
        """validate_api_config raises error for missing API key."""
        config = EncoderConfig(
            model_name="openai/model",
            use_api_embeddings=True,
            openai_api_key=None,
        )

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            config.validate_api_config()

    def test_validate_api_config_placeholder_key(self):
        """validate_api_config rejects placeholder API key."""
        config = EncoderConfig(
            model_name="openai/model",
            use_api_embeddings=True,
            openai_api_key="your_api_key_here",
            openai_base_url="https://api.test.com",
        )

        with pytest.raises(ValueError, match="placeholder"):
            config.validate_api_config()

    def test_validate_api_config_missing_base_url(self):
        """validate_api_config raises error for missing base URL."""
        config = EncoderConfig(
            model_name="openai/model",
            use_api_embeddings=True,
            openai_api_key="valid-key",
            openai_base_url=None,
        )

        with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
            config.validate_api_config()

    def test_validate_api_config_valid(self):
        """validate_api_config passes for valid configuration."""
        config = EncoderConfig(
            model_name="openai/model",
            use_api_embeddings=True,
            openai_api_key="valid-key",
            openai_base_url="https://api.test.com",
        )

        config.validate_api_config()  # Should not raise

    def test_generate_cache_key_none_for_full_dataset(self):
        """generate_cache_key returns None for full dataset."""
        config = EncoderConfig(model_name="test-model")

        result = config.generate_cache_key()

        assert result is None

    def test_generate_cache_key_filtered_dataset(self):
        """generate_cache_key returns key for filtered dataset."""
        config = EncoderConfig(
            model_name="test-model", ids_set={"equilibrium", "core_profiles"}
        )

        result = config.generate_cache_key()

        assert result is not None
        assert "filtered" in result
        assert "core_profiles" in result
        assert "equilibrium" in result

    def test_to_dict_serialization(self):
        """to_dict serializes config correctly."""
        config = EncoderConfig(
            model_name="test-model",
            batch_size=100,
            ids_set={"equilibrium"},
            openai_api_key="secret-key",
            openai_base_url="https://api.test.com",
        )

        result = config.to_dict()

        assert result["model_name"] == "test-model"
        assert result["batch_size"] == 100
        assert result["ids_set"] == ["equilibrium"]
        assert result["openai_base_url"] == "https://api.test.com"
        assert "openai_api_key" not in result  # Should be excluded for security

    def test_from_dict_deserialization(self, monkeypatch):
        """from_dict deserializes config correctly."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")

        data = {
            "model_name": "restored-model",
            "batch_size": 150,
            "ids_set": ["core_profiles"],
            "use_api_embeddings": True,
        }

        config = EncoderConfig.from_dict(data)

        assert config.model_name == "restored-model"
        assert config.batch_size == 150
        assert config.ids_set == {"core_profiles"}

    def test_from_dict_ignores_api_key(self):
        """from_dict ignores API key in dict for security."""
        data = {
            "model_name": "test-model",
            "openai_api_key": "should-be-ignored",
        }

        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}, clear=False):
            config = EncoderConfig.from_dict(data)

            # API key should come from environment, not dict
            assert config.openai_api_key == "env-key"

    def test_from_environment_factory(self, monkeypatch):
        """from_environment creates config from environment."""
        monkeypatch.setenv("IMAS_MCP_EMBEDDING_MODEL", "env-model")
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://env.api.com")

        with patch(
            "imas_mcp.embeddings.config.IMAS_MCP_EMBEDDING_MODEL", "fallback-model"
        ):
            config = EncoderConfig.from_environment()

            assert config.openai_api_key == "env-key"
            assert config.openai_base_url == "https://env.api.com"

    def test_get_api_info(self):
        """get_api_info returns safe API configuration info."""
        config = EncoderConfig(
            model_name="openai/text-embedding-3-small",
            use_api_embeddings=True,
            openai_api_key="sk-1234567890abcdefghij",
            openai_base_url="https://api.test.com",
        )

        result = config.get_api_info()

        assert result["use_api_embeddings"] is True
        assert result["model_name"] == "openai/text-embedding-3-small"
        assert result["has_api_key"] is True
        assert result["base_url"] == "https://api.test.com"
        assert result["api_key_prefix"] == "sk-1234567..."  # Truncated for safety

    def test_get_api_info_no_key(self):
        """get_api_info handles missing API key gracefully."""
        config = EncoderConfig(
            model_name="all-MiniLM-L6-v2",
            use_api_embeddings=False,
            openai_api_key=None,
        )

        result = config.get_api_info()

        assert result["has_api_key"] is False
        assert result["api_key_prefix"] is None

    def test_ids_set_as_set(self):
        """ids_set is stored as a set."""
        config = EncoderConfig(ids_set={"a", "b", "c"})

        assert isinstance(config.ids_set, set)
        assert config.ids_set == {"a", "b", "c"}
