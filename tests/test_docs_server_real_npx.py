import asyncio
import os
import shutil

import pytest

from imas_mcp.services.docs_server_manager import DocsServerManager


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("npx") is None, reason="npx not available")
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY required for docs-mcp-server startup",
)
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
async def test_docs_server_startup_real_npx():
    """
    Test that the docs server can actually start up if npx is available.
    This test is skipped if npx is not in the PATH or OPENAI_API_KEY is not set.
    """
    # Use a high port to avoid conflicts
    manager = DocsServerManager(default_port=19200)

    try:
        await manager.start_server()
        assert manager.is_running

        # Perform a health check
        health = await manager.health_check()
        assert health["status"] == "healthy"

    finally:
        if manager.is_running:
            await manager.stop_server()
