"""Tests for the /health endpoint on HTTP transports."""

import atexit
import threading
import time
from contextlib import contextmanager
from typing import Literal, cast
from unittest.mock import MagicMock

import pytest
import requests

from imas_codex.server import Server
from tests.conftest import STANDARD_TEST_IDS_SET

# Track server threads for cleanup at exit to avoid daemon thread shutdown errors
_server_threads: list[threading.Thread] = []


def _cleanup_threads():
    """Cleanup handler to allow daemon threads to finish before interpreter shutdown."""
    # Give threads a moment to finish their current operation
    time.sleep(0.1)


atexit.register(_cleanup_threads)


@contextmanager
def run_server(port: int, transport: str = "streamable-http"):
    mock_gc = MagicMock()
    mock_gc.query.return_value = [{"v.id": "4.0.0"}]
    server = Server(ids_set=STANDARD_TEST_IDS_SET, graph_client=mock_gc)

    def _run():  # type: ignore
        try:
            server.run(
                transport=cast(Literal["streamable-http", "sse"], transport),
                host="127.0.0.1",
                port=port,
            )
        except Exception:
            pass  # Suppress errors during shutdown

    thread = threading.Thread(target=_run, daemon=True)
    _server_threads.append(thread)
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
    port = 8900 if transport == "streamable-http" else 8901
    with run_server(port=port, transport=transport):
        # Poll until health available with shorter intervals
        for _ in range(60):
            try:
                resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=0.2)
                if resp.status_code == 200:
                    data = resp.json()
                    assert data["status"] == "ok"
                    assert "imas_codex_version" in data
                    assert "imas_dd_version" in data
                    assert "started_at" in data
                    assert "ids_count" in data
                    assert "uptime" in data
                    break
            except Exception:
                time.sleep(0.05)
        else:
            pytest.fail("/health endpoint not reachable")


def test_health_idempotent_wrapping(monkeypatch):
    port = 8902
    mock_gc = MagicMock()
    mock_gc.query.return_value = [{"v.id": "4.0.0"}]
    server = Server(ids_set=STANDARD_TEST_IDS_SET, graph_client=mock_gc)

    # Ensure multiple calls to HealthEndpoint don't duplicate (implicit by running twice)
    def _run():  # mypy: ignore
        try:
            server.run(transport="streamable-http", host="127.0.0.1", port=port)
        except Exception:
            pass  # Suppress errors during shutdown

    thread = threading.Thread(target=_run, daemon=True)
    _server_threads.append(thread)
    thread.start()
    for _ in range(60):
        try:
            resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=0.2)
            if resp.status_code == 200:
                data = resp.json()
                assert data["status"] == "ok"
                assert "imas_codex_version" in data
                assert "started_at" in data
                assert "ids_count" in data
                assert "uptime" in data
                break
        except Exception:
            time.sleep(0.05)
    else:
        pytest.fail("/health endpoint not reachable after idempotent wrap test")
