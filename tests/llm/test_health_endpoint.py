"""Tests for the rich /health endpoint on AgentsServer."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

DD_ENDPOINT = "https://imas-dd.iter.org"


def _endpoint_reachable() -> bool:
    """Check if the live DD endpoint is reachable."""
    try:
        r = httpx.get(f"{DD_ENDPOINT}/health", timeout=5)
        return r.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


_skip_no_endpoint = pytest.mark.skipif(
    not _endpoint_reachable(),
    reason=f"{DD_ENDPOINT} not reachable",
)


class TestFormatUptime:
    """Tests for uptime formatting logic."""

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """Mirror the _format_uptime implementation for unit testing."""
        if seconds < 0:
            seconds = 0
        remainder = int(seconds)
        days, remainder = divmod(remainder, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, secs = divmod(remainder, 60)
        parts: list[str] = []
        if days:
            parts.append(f"{days}d")
        if hours or days:
            parts.append(f"{hours}h")
        if minutes or hours or days:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")
        return " ".join(parts)

    def test_seconds_only(self):
        assert self._format_uptime(42) == "42s"

    def test_minutes_and_seconds(self):
        assert self._format_uptime(125) == "2m 5s"

    def test_hours_minutes_seconds(self):
        assert self._format_uptime(3661) == "1h 1m 1s"

    def test_days(self):
        assert self._format_uptime(90061) == "1d 1h 1m 1s"

    def test_zero(self):
        assert self._format_uptime(0) == "0s"

    def test_negative_clamps_to_zero(self):
        assert self._format_uptime(-10) == "0s"

    def test_large_value(self):
        assert self._format_uptime(86400 * 3 + 7200 + 60 + 1) == "3d 2h 1m 1s"

    def test_exact_hour(self):
        assert self._format_uptime(3600) == "1h 0m 0s"

    def test_exact_day(self):
        assert self._format_uptime(86400) == "1d 0h 0m 0s"


class TestHealthEndpointRegistration:
    """Test that the health endpoint is registered on the MCP server."""

    @patch("imas_codex.llm.server.load_prompts", return_value={})
    def test_health_route_registered(self, _mock_prompts):
        """The /health custom route is registered on the MCP server."""
        from imas_codex.llm.server import AgentsServer

        server = AgentsServer(read_only=True)
        routes = server.mcp._additional_http_routes
        health_routes = [r for r in routes if r.path == "/health"]
        assert len(health_routes) == 1
        assert "GET" in health_routes[0].methods

    @patch("imas_codex.llm.server.load_prompts", return_value={})
    def test_started_at_recorded(self, _mock_prompts):
        """Server records startup time for uptime calculation."""
        from imas_codex.llm.server import AgentsServer

        before = time.monotonic()
        server = AgentsServer(read_only=True)
        after = time.monotonic()
        assert before <= server._started_at <= after


class TestHealthEndpointResponse:
    """Test health endpoint response structure."""

    @pytest.fixture
    def mock_graph_client(self):
        """Create a mock GraphClient with typical responses."""
        gc = MagicMock()
        gc.get_stats.return_value = {"nodes": 50000, "relationships": 120000}
        gc.query.side_effect = self._mock_query
        gc.close = MagicMock()
        return gc

    @staticmethod
    def _mock_query(query, **kwargs):
        if "DDVersion" in query:
            return [
                {"version": "3.39.0", "is_current": False},
                {"version": "3.41.0", "is_current": False},
                {"version": "4.0.0", "is_current": True},
            ]
        if "IMASNode" in query:
            return [{"paths": 12345, "ids_count": 42}]
        if "GraphMeta" in query:
            return [
                {
                    "name": "codex",
                    "facilities": ["tcv", "jet"],
                    "imas": True,
                    "created_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-03-20T00:00:00",
                }
            ]
        return []

    @patch("imas_codex.llm.server.load_prompts", return_value={})
    def test_response_has_required_fields(self, _mock_prompts, mock_graph_client):
        """Health response includes version, uptime, graph, facilities."""
        from imas_codex.llm.server import AgentsServer

        server = AgentsServer(read_only=False)

        with (
            patch(
                "imas_codex.graph.client.GraphClient.from_profile",
                return_value=mock_graph_client,
            ),
        ):
            # Call _query_graph directly (it's defined inside _register_health_check)
            # We test the full endpoint via the route handler
            import asyncio

            from starlette.testclient import TestClient

            app = server.mcp.http_app()
            client = TestClient(app)
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "uptime" in data
        assert "uptime_seconds" in data
        assert "mode" not in data

    @patch("imas_codex.llm.server.load_prompts", return_value={})
    def test_response_with_graph_data(self, _mock_prompts, mock_graph_client):
        """Health response includes graph metadata when available."""
        from imas_codex.llm.server import AgentsServer

        server = AgentsServer(read_only=True)

        with patch(
            "imas_codex.graph.client.GraphClient.from_profile",
            return_value=mock_graph_client,
        ):
            from starlette.testclient import TestClient

            app = server.mcp.http_app()
            client = TestClient(app)
            response = client.get("/health")

        data = response.json()
        assert data["graph"]["status"] == "ok"
        assert data["graph"]["name"] == "codex"
        assert data["graph"]["node_count"] == 50000
        assert data["facilities"] == ["tcv", "jet"]
        assert data["imas_dd"]["version"] == "4.0.0"
        assert data["imas_dd"]["min_version"] == "3.39.0"
        assert "max" not in data["imas_dd"]
        assert data["imas_dd"]["version_count"] == 3
        assert data["imas_dd"]["ids_count"] == 42
        assert data["imas_dd"]["path_count"] == 12345
        assert "mode" not in data

    @patch("imas_codex.llm.server.load_prompts", return_value={})
    def test_response_graph_unavailable(self, _mock_prompts):
        """Health response degrades gracefully when graph is unavailable."""
        from imas_codex.llm.server import AgentsServer

        server = AgentsServer(read_only=True)

        with patch(
            "imas_codex.graph.client.GraphClient.from_profile",
            side_effect=Exception("Connection refused"),
        ):
            from starlette.testclient import TestClient

            app = server.mcp.http_app()
            client = TestClient(app)
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["graph"]["status"] == "unavailable"
        assert "Connection refused" in data["graph"]["error"]

    @patch("imas_codex.llm.server.load_prompts", return_value={})
    def test_dd_only_graph_no_facilities(self, _mock_prompts):
        """DD-only graph reports empty facilities list."""
        gc = MagicMock()
        gc.get_stats.return_value = {"nodes": 5000, "relationships": 10000}
        gc.close = MagicMock()

        def dd_only_query(query, **kwargs):
            if "DDVersion" in query:
                return [{"version": "4.0.0", "is_current": True}]
            if "IMASNode" in query:
                return [{"paths": 5000, "ids_count": 30}]
            if "GraphMeta" in query:
                return [
                    {
                        "name": "codex-imas",
                        "facilities": [],
                        "imas": True,
                        "created_at": "2026-01-01T00:00:00",
                        "updated_at": "2026-03-20T00:00:00",
                    }
                ]
            return []

        gc.query.side_effect = dd_only_query

        from imas_codex.llm.server import AgentsServer

        server = AgentsServer(read_only=True)

        with patch(
            "imas_codex.graph.client.GraphClient.from_profile",
            return_value=gc,
        ):
            from starlette.testclient import TestClient

            app = server.mcp.http_app()
            client = TestClient(app)
            response = client.get("/health")

        data = response.json()
        assert data["facilities"] == []
        assert data["graph"]["name"] == "codex-imas"


@pytest.mark.slow
@pytest.mark.integration
class TestLiveEndpoint:
    """Integration tests against the deployed DD-only server at imas-dd.iter.org.

    Run explicitly:
        uv run pytest tests/llm/test_health_endpoint.py::TestLiveEndpoint -m slow -v
    """

    @_skip_no_endpoint
    def test_health_returns_200(self):
        """GET /health returns 200 with JSON body."""
        r = httpx.get(f"{DD_ENDPOINT}/health", timeout=10)
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/json"

    @_skip_no_endpoint
    def test_health_response_structure(self):
        """Health response has required fields for a DD-only deployment."""
        r = httpx.get(f"{DD_ENDPOINT}/health", timeout=10)
        data = r.json()
        assert data["status"] == "ok"
        assert "uptime" in data
        assert isinstance(
            data.get("ids_count", data.get("imas_dd", {}).get("ids_count")), int
        )

    @_skip_no_endpoint
    def test_mcp_post_initialize(self):
        """POST /mcp with initialize succeeds (streamable-http transport)."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1.0"},
            },
        }
        r = httpx.post(
            f"{DD_ENDPOINT}/mcp",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            timeout=10,
        )
        assert r.status_code == 200
        assert "mcp-session-id" in r.headers
        # Streamable-HTTP returns SSE format
        assert "text/event-stream" in r.headers["content-type"]
        assert '"initialize"' not in r.text or '"result"' in r.text

    @_skip_no_endpoint
    def test_mcp_get_without_session_rejected(self):
        """GET /mcp without session ID is rejected (SSE not supported)."""
        r = httpx.get(
            f"{DD_ENDPOINT}/mcp",
            headers={"Accept": "text/event-stream"},
            timeout=10,
        )
        assert r.status_code == 400

    @_skip_no_endpoint
    def test_sse_endpoint_not_available(self):
        """Old SSE transport endpoint /sse does not exist."""
        r = httpx.get(f"{DD_ENDPOINT}/sse", timeout=10)
        assert r.status_code == 404
