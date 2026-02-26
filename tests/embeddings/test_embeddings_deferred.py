from unittest.mock import MagicMock, patch

import pytest

from imas_codex.embeddings.embeddings import Embeddings
from imas_codex.search.document_store import DocumentStore
from tests.conftest import STANDARD_TEST_IDS_SET


def _mock_encoder(model_name: str = "all-MiniLM-L6-v2"):
    """Create a mock Encoder that doesn't require sentence_transformers."""
    mock = MagicMock()
    mock_config = MagicMock()
    mock_config.model_name = model_name
    mock_config.generate_cache_key.return_value = None
    mock.config = mock_config
    mock.build_document_embeddings.return_value = (
        __import__("numpy").zeros((0, 256)),
        [],
        False,
    )
    return mock


def test_embeddings_encoder_config_exposed(monkeypatch):
    monkeypatch.setenv("IMAS_CODEX_EMBEDDING_LOCATION", "local")
    ds = DocumentStore()
    with patch(
        "imas_codex.embeddings.embeddings.Encoder", return_value=_mock_encoder()
    ):
        emb = Embeddings(document_store=ds, load_embeddings=False)
    cfg = emb.encoder_config
    assert cfg.model_name == emb.model_name


def test_embeddings_lazy_build(monkeypatch):
    monkeypatch.setenv("IMAS_CODEX_EMBEDDING_LOCATION", "local")

    ds = DocumentStore()
    with patch(
        "imas_codex.embeddings.embeddings.Encoder",
        return_value=_mock_encoder(),
    ):
        emb = Embeddings(document_store=ds, load_embeddings=False)
    assert emb._embeddings is None
    # trigger build lazily
    _ = emb.get_embeddings_matrix()
    # After build (may still be empty if no docs) but attribute should be set
    assert emb._embeddings is not None


@pytest.mark.asyncio
async def test_health_endpoint_deferred(monkeypatch):
    """Health endpoint works with graph-only server."""
    from imas_codex.health import HealthEndpoint
    from imas_codex.server import Server

    mock_gc = MagicMock()
    mock_gc.query.return_value = [{"v.id": "4.0.0"}]

    srv = Server(ids_set=STANDARD_TEST_IDS_SET, graph_client=mock_gc)

    he = HealthEndpoint(srv)
    he.attach()
    app = srv.mcp.http_app()
    health_route = next(r for r in app.routes if getattr(r, "path", None) == "/health")
    response = await health_route.endpoint()  # type: ignore[attr-defined]
    data = response.body.decode()
    assert "imas_codex_version" in data
