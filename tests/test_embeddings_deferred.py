import pytest

from imas_mcp.embeddings.embeddings import Embeddings
from imas_mcp.search.document_store import DocumentStore


def test_embeddings_encoder_config_exposed():
    ds = DocumentStore()
    emb = Embeddings(document_store=ds, load_embeddings=False)
    cfg = emb.encoder_config
    assert cfg.model_name == emb.model_name


def test_embeddings_lazy_build(monkeypatch):
    ds = DocumentStore()
    emb = Embeddings(document_store=ds, load_embeddings=False)
    assert emb._embeddings is None
    # trigger build lazily
    _ = emb.get_embeddings_matrix()
    # After build (may still be empty if no docs) but attribute should be set
    assert emb._embeddings is not None


@pytest.mark.asyncio
async def test_health_endpoint_deferred(monkeypatch):
    from imas_mcp.server import Server

    srv = Server()
    # Replace embeddings with deferred instance sharing same document store
    srv.embeddings = Embeddings(
        document_store=srv.tools.document_store, load_embeddings=False
    )
    from imas_mcp.health import HealthEndpoint

    he = HealthEndpoint(srv)
    he.attach()
    # Construct app and call handlers directly
    app = srv.mcp.http_app()
    # Find /health route handler
    health_route = next(r for r in app.routes if getattr(r, "path", None) == "/health")
    response = await health_route.endpoint()  # type: ignore[attr-defined]
    data = response.body.decode()
    # embedding_status field removed in synchronous embeddings refactor.
    # Verify health endpoint still works with deferred embeddings and exposes model name.
    assert "embedding_model_name" in data
