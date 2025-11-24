import os
import shutil

import pytest

from imas_mcp.services.docs_server_manager import DocsServerManager


@pytest.mark.asyncio
@pytest.mark.skipif(
    not shutil.which("npx"),
    reason="npx not found in PATH",
)
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY", "").strip(),
    reason="Valid OPENAI_API_KEY required for docs-mcp-server startup",
)
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
async def test_docs_server_startup_real_npx():
    """Test that the docs server can start up with npx and OPENAI_API_KEY.

    This is a simple integration test - if it fails, check:
    1. OPENAI_API_KEY is set and valid
    2. npx is in PATH
    3. Network connectivity for downloading docs-mcp-server
    """
    manager = DocsServerManager(default_port=6280)

    try:
        await manager.start_server()
        assert manager.is_running

        # Verify server responds
        health = await manager.health_check()
        assert health["status"] == "healthy"
    finally:
        await manager.stop_server()
