"""Tests for instruction-aware embedding via prompt_name parameter."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_embed_request_accepts_prompt_name():
    """EmbedRequest should accept a prompt_name field."""
    from imas_codex.embeddings.server import EmbedRequest

    req = EmbedRequest(texts=["hello"], prompt_name="query")
    assert req.prompt_name == "query"


def test_embed_request_prompt_name_defaults_none():
    """EmbedRequest prompt_name should default to None."""
    from imas_codex.embeddings.server import EmbedRequest

    req = EmbedRequest(texts=["hello"])
    assert req.prompt_name is None


def test_embed_texts_accepts_prompt_name():
    """embed_texts() should accept prompt_name without error (local backend mock)."""
    from imas_codex.embeddings.encoder import Encoder

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

    encoder = MagicMock(spec=Encoder)
    encoder.embed_texts.return_value = np.array([[0.1, 0.2, 0.3]])

    result = encoder.embed_texts(["plasma current"], prompt_name="query")
    encoder.embed_texts.assert_called_once_with(["plasma current"], prompt_name="query")
    assert result.shape == (1, 3)


def test_embed_texts_local_passes_prompt_name():
    """embed_texts() passes prompt_name to model.encode() for local backend."""
    from imas_codex.embeddings.config import EmbeddingBackend, EncoderConfig
    from imas_codex.embeddings.encoder import Encoder

    config = EncoderConfig(
        model_name="test-model",
        backend=EmbeddingBackend.LOCAL,
    )
    encoder = Encoder(config)

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    encoder._model = mock_model

    encoder.embed_texts(["electron temperature"], prompt_name="query")

    mock_model.encode.assert_called_once()
    call_kwargs = mock_model.encode.call_args[1]
    assert call_kwargs.get("prompt_name") == "query"


def test_embed_texts_local_no_prompt_name_omitted():
    """embed_texts() does NOT pass prompt_name to model.encode() when None."""
    from imas_codex.embeddings.config import EmbeddingBackend, EncoderConfig
    from imas_codex.embeddings.encoder import Encoder

    config = EncoderConfig(
        model_name="test-model",
        backend=EmbeddingBackend.LOCAL,
    )
    encoder = Encoder(config)

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    encoder._model = mock_model

    encoder.embed_texts(["electron temperature"])

    call_kwargs = mock_model.encode.call_args[1]
    assert "prompt_name" not in call_kwargs


def test_remote_client_embed_passes_prompt_name():
    """RemoteEmbeddingClient.embed() includes prompt_name in HTTP request body."""
    from imas_codex.embeddings.client import RemoteEmbeddingClient

    client = RemoteEmbeddingClient(base_url="http://localhost:8765")
    mock_http = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "embeddings": [[0.1, 0.2, 0.3]],
        "elapsed_ms": 10,
    }
    mock_http.post.return_value = mock_response
    client._client = mock_http

    client._embed_single(
        ["test query"], normalize=True, max_retries=1, prompt_name="query"
    )

    mock_http.post.assert_called_once()
    call_kwargs = mock_http.post.call_args[1]
    body = call_kwargs.get("json", {})
    assert body.get("prompt_name") == "query"


def test_remote_client_embed_omits_prompt_name_when_none():
    """RemoteEmbeddingClient._embed_single() omits prompt_name when None."""
    from imas_codex.embeddings.client import RemoteEmbeddingClient

    client = RemoteEmbeddingClient(base_url="http://localhost:8765")
    mock_http = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "embeddings": [[0.1, 0.2, 0.3]],
        "elapsed_ms": 10,
    }
    mock_http.post.return_value = mock_response
    client._client = mock_http

    client._embed_single(["test query"], normalize=True, max_retries=1)

    call_kwargs = mock_http.post.call_args[1]
    body = call_kwargs.get("json", {})
    assert "prompt_name" not in body


def test_embed_query_uses_prompt_name():
    """_embed_query() in graph_search should call embed_texts with prompt_name='query'."""
    from imas_codex.tools import graph_search

    mock_encoder = MagicMock()
    mock_encoder.embed_texts.return_value = np.array([[0.1, 0.2, 0.3]])

    with patch.object(graph_search, "_get_encoder", return_value=mock_encoder):
        # Use the GraphSearchTool as a representative class with _embed_query
        tool = graph_search.GraphSearchTool.__new__(graph_search.GraphSearchTool)
        result = tool._embed_query("electron temperature")

    mock_encoder.embed_texts.assert_called_once_with(
        ["electron temperature"], prompt_name="query"
    )
    assert result == pytest.approx([0.1, 0.2, 0.3])
