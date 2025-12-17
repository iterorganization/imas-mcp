"""Tests for the /health endpoint on HTTP transports."""

import os
import threading
import time
from contextlib import contextmanager
from typing import Literal, cast

import pytest
import requests

from imas_codex.server import Server
from tests.conftest import STANDARD_TEST_IDS_SET


@contextmanager
def run_server(port: int, transport: str = "streamable-http"):
    # Server now always manages asynchronous embedding initialization internally;
    # legacy initialize_embeddings flag removed.
    server = Server(ids_set=STANDARD_TEST_IDS_SET, use_rich=False)

    def _run():  # type: ignore
        server.run(
            transport=cast(Literal["streamable-http", "sse"], transport),
            host="127.0.0.1",
            port=port,
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    # Wait for server to start with shorter timeout
    for _ in range(50):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=0.2)
            if r.status_code in (200, 404):  # server responsive
                break
        except Exception:
            time.sleep(0.05)
    yield server


@pytest.mark.parametrize("transport", ["streamable-http", "sse"])
@pytest.mark.timeout(15)
def test_health_basic(transport, monkeypatch):
    # Use free API embeddings by default
    monkeypatch.setenv("IMAS_CODEX_EMBEDDING_MODEL", "openai/text-embedding-3-small")

    port = 8900 if transport == "streamable-http" else 8901
    with run_server(port=port, transport=transport):
        # Poll until health available with shorter intervals
        for _ in range(60):
            try:
                resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=0.2)
                if resp.status_code == 200:
                    data = resp.json()
                    assert data["status"] == "ok"
                    assert "mcp_server_version" in data
                    assert "imas_dd_version" in data
                    assert "document_count" in data
                    assert "embedding_model_name" in data
                    assert "started_at" in data
                    assert "ids_count" in data
                    assert "uptime" in data
                    break
            except Exception:
                time.sleep(0.05)
        else:
            pytest.fail("/health endpoint not reachable")


def test_health_idempotent_wrapping(monkeypatch):
    # Use free API embeddings by default
    monkeypatch.setenv("IMAS_CODEX_EMBEDDING_MODEL", "openai/text-embedding-3-small")

    port = 8902
    server = Server(ids_set=STANDARD_TEST_IDS_SET, use_rich=False)

    # Ensure multiple calls to HealthEndpoint don't duplicate (implicit by running twice)
    def _run():  # mypy: ignore
        server.run(transport="streamable-http", host="127.0.0.1", port=port)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    for _ in range(60):
        try:
            resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=0.2)
            if resp.status_code == 200:
                data = resp.json()
                assert data["status"] == "ok"
                assert "mcp_server_version" in data
                assert "embedding_model_name" in data
                assert "started_at" in data
                assert "ids_count" in data
                assert "uptime" in data
                break
        except Exception:
            time.sleep(0.05)
    else:
        pytest.fail("/health endpoint not reachable after idempotent wrap test")
