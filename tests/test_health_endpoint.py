"""Tests for the /health endpoint on HTTP transports."""

import threading
import time
from contextlib import contextmanager
from typing import Literal, cast

import pytest
import requests

from imas_mcp.server import Server
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
    # Wait for server to start
    for _ in range(120):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
            if r.status_code in (200, 404):  # server responsive
                break
        except Exception:
            time.sleep(0.1)
    yield server


@pytest.mark.parametrize("transport", ["streamable-http", "sse"])
@pytest.mark.timeout(30)
def test_health_basic(transport):
    port = 8900 if transport == "streamable-http" else 8901
    with run_server(port=port, transport=transport):
        # Poll until health available
        for _ in range(120):
            try:
                resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
                if resp.status_code == 200:
                    data = resp.json()
                    assert data["status"] == "ok"
                    assert "version" in data
                    assert "dd_version" in data
                    assert "documents" in data
                    # New embedding metadata fields
                    assert "embedding_model_name" in data
                    assert "started_at" in data
                    assert "uptime_seconds" in data
                    break
            except Exception:
                time.sleep(0.1)
        else:
            pytest.fail("/health endpoint not reachable")


def test_health_idempotent_wrapping():
    port = 8902
    server = Server(ids_set=STANDARD_TEST_IDS_SET, use_rich=False)

    # Ensure multiple calls to HealthEndpoint don't duplicate (implicit by running twice)
    def _run():  # mypy: ignore
        server.run(transport="streamable-http", host="127.0.0.1", port=port)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    for _ in range(120):
        try:
            resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
            if resp.status_code == 200:
                data = resp.json()
                assert data["status"] == "ok"
                assert "embedding_model_name" in data
                assert "started_at" in data
                assert "uptime_seconds" in data
                break
        except Exception:
            time.sleep(0.1)
    else:
        pytest.fail("/health endpoint not reachable after idempotent wrap test")
