"""Health endpoint for IMAS Codex MCP server HTTP transports.

Provides a lightweight `/health` route exposing liveness and
data dictionary metadata from the graph.
"""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from starlette.responses import JSONResponse

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from imas_codex.server import Server

logger = logging.getLogger(__name__)


@dataclass
class HealthEndpoint:
    """Attach a /health route to a FastMCP HTTP transport app."""

    server: Server

    def _get_version(self) -> str:
        try:
            return importlib.metadata.version("imas-codex")
        except Exception:  # pragma: no cover - defensive
            return "unknown"

    def attach(self) -> None:
        """Wrap the HTTP app factory to inject /health."""
        attr = "http_app"
        sentinel = "_health_wrapped_http"
        if getattr(self.server.mcp, sentinel, False):
            return
        original = getattr(self.server.mcp, attr)

        async def health_handler(request=None):  # type: ignore[unused-argument]
            gc = self.server.graph_client

            # Query graph for version and stats
            dd_version = self.server.mcp.name.removeprefix("imas-data-dictionary-")
            try:
                stats = gc.query(
                    "MATCH (p:IMASPath) RETURN count(p) AS paths, "
                    "count(DISTINCT p.ids) AS ids_count"
                )
                ids_count = stats[0]["ids_count"] if stats else 0
                path_count = stats[0]["paths"] if stats else 0
            except Exception:
                ids_count = 0
                path_count = 0

            def _format_uptime(seconds: float) -> str:
                try:
                    if seconds < 0:  # pragma: no cover - defensive
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
                except Exception:  # pragma: no cover - defensive
                    return f"{round(seconds, 3)}s"

            uptime_seconds = round(self.server.uptime_seconds(), 3)
            response = {
                "status": "ok",
                "imas_codex_version": self._get_version(),
                "imas_dd_version": dd_version,
                "ids_count": ids_count,
                "path_count": path_count,
                "started_at": self.server.started_at.isoformat(),
                "uptime": _format_uptime(uptime_seconds),
            }
            return JSONResponse(response)

        def wrapped(*args, **kwargs):  # type: ignore[override]
            app = original(*args, **kwargs)
            existing_paths = {
                getattr(r, "path", None) for r in getattr(app, "routes", [])
            }
            if "/health" not in existing_paths:
                if hasattr(app, "add_api_route"):
                    app.add_api_route(
                        "/health", health_handler, methods=["GET"], tags=["infra"]
                    )
                else:
                    app.add_route("/health", health_handler)
            return app

        setattr(self.server.mcp, attr, wrapped)
        setattr(self.server.mcp, sentinel, True)
